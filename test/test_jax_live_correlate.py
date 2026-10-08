# Copyright (C) 2026 The PyCBC Collaboration

"""Regression tests for the JAX LiveBatch correlation update."""

import os
import types

import numpy as np
import pytest
from types import SimpleNamespace

from pycbc import scheme
from pycbc.filter.matchedfilter import BatchCorrelator, LiveBatchMatchedFilter
from pycbc.filter import correlate
from pycbc.types import Array, FrequencySeries
from pycbc.waveform.bank import sigma_cached

try:
    import jax
except ImportError:
    pytest.skip("JAX is unavailable", allow_module_level=True)
from pycbc.filter.matchedfilter_jax import batch_correlate_execute
from pycbc.filter import matchedfilter_jax as backend

jnp = jax.numpy

_DEVICE = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu").split(":", 1)[-1]
if _DEVICE == "jax":
    _DEVICE = "cpu"


def _jax_context(**kwargs):
    return scheme.JAXScheme(_DEVICE, **kwargs)


def test_batch_correlator_updates_sibling_views_and_preserves_tails():
    """One batched update must preserve parent storage outside output views."""
    batch, size, stride, base, tail = 4, 9, 16, 3, 7
    rng = np.random.default_rng(1234)
    xs_data = (
        rng.normal(size=(batch, size + 4))
        + 1j * rng.normal(size=(batch, size + 4))
    ).astype(np.complex64)
    y_data = (
        rng.normal(size=size + 5) + 1j * rng.normal(size=size + 5)
    ).astype(np.complex64)

    with _jax_context():
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        parent = Array(
            np.full(base + batch * stride + tail, 7 + 3j, dtype=np.complex64)
        )
        zs = [
            parent[base + i * stride: base + (i + 1) * stride]
            for i in range(batch)
        ]

        correlator = BatchCorrelator(xs, zs, size)
        assert correlator.x is None
        assert correlator.z is None
        assert correlator._jax_template_matrix is None
        correlator.batch_correlate_execute(y)
        expected = np.conj(xs_data[:, :size]) * y_data[None, :size]
        result = np.asarray(parent)
        np.testing.assert_allclose(
            result[base: base + batch * stride].reshape(batch, stride)[
               :, :size
            ],
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(result[:base], 7 + 3j)
        np.testing.assert_allclose(
            result[base: base + batch * stride].reshape(batch, stride)[
               :, size:
            ],
            7 + 3j,
        )
        np.testing.assert_allclose(result[-tail:], 7 + 3j)

        # A second call must use the current inputs rather than stale output.
        xs_data *= np.complex64(0.25 - 0.5j)
        for x, row in zip(xs, xs_data):
            x._data.set_array(jnp.asarray(row))
        y._data.set_array(jnp.asarray(y_data * (0.75 + 0.2j)))
        correlator.batch_correlate_execute(y)
        np.testing.assert_allclose(
            np.asarray(parent)[base: base + batch * stride].reshape(
                batch, stride
            )[:, :size],
            np.conj(xs_data[:, :size]) * (y_data[None, :size] * (0.75 + 0.2j)),
            rtol=2e-6,
            atol=2e-6,
        )


def test_immutable_live_correlator_packs_ordinary_jax_rows_once():
    """Live caches immutable rows; generic callers retain mutable inputs."""
    batch, size = 2, 8
    rng = np.random.default_rng(9753)
    xs_data = (
        rng.normal(size=(batch, size + 1))
        + 1j * rng.normal(size=(batch, size + 1))
    ).astype(np.complex64)
    y_data = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )

    with _jax_context():
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        zs = [
            Array(np.full(size + 2, 3 + 4j, dtype=np.complex64))
            for _ in range(batch)
        ]
        correlator = BatchCorrelator(xs, zs, size, immutable_templates=True)
        assert correlator.x is None
        assert correlator.z is None
        assert correlator._jax_template_matrix is not None
        correlator.batch_correlate_execute(y)
        for z, row in zip(zs, xs_data):
            np.testing.assert_allclose(
                np.asarray(z)[:size],
                np.conj(row[:size]) * y_data,
                rtol=2e-6,
                atol=2e-6,
            )


def test_batch_correlator_falls_back_for_unrelated_outputs():
    """Unrelated output arrays retain the general BatchCorrelator behavior."""
    batch, size = 3, 16
    rng = np.random.default_rng(5678)
    xs_data = (
        rng.normal(size=(batch, size)) + 1j * rng.normal(size=(batch, size))
    ).astype(np.complex64)
    y_data = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )
    with _jax_context():
        xs = [Array(row) for row in xs_data]
        ys = Array(y_data)
        zs = [
            Array(np.full(size + 3, 5 + 2j, dtype=np.complex64))
            for _ in range(batch)
        ]
        BatchCorrelator(xs, zs, size).batch_correlate_execute(ys)
        for z, row in zip(zs, xs_data):
            np.testing.assert_allclose(
                np.asarray(z)[:size],
                np.conj(row[:size]) * y_data[:size],
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_allclose(np.asarray(z)[size:], 5 + 2j)


def test_numpy_output_prefix_fallback_preserves_tail():
    """The general fallback must preserve NumPy output backing arrays."""

    class NumpyHolder:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, item):
            return NumpyHolder(self._data[item])

    batch, size = 2, 5
    xs = [
        np.arange(size + 2, dtype=np.complex64) + 1j * (i + 1)
        for i in range(batch)
    ]
    y = np.arange(size + 3, dtype=np.complex64) + 2j
    backing = [
        np.full(size + 3, 4 + 1j, dtype=np.complex64) for _ in range(batch)
    ]
    zs = [NumpyHolder(z) for z in backing]
    identities = [id(z._data) for z in zs]
    control = SimpleNamespace(xs=xs, zs=zs, size=size)
    batch_correlate_execute(control, y)
    for i, (z, x) in enumerate(zip(zs, xs)):
        np.testing.assert_allclose(
            z._data[:size], np.conj(x[:size]) * y[:size]
        )
        np.testing.assert_array_equal(z._data[size:], 4 + 1j)
        assert id(z._data) == identities[i]


def test_batch_correlator_full_parent_block_uses_exact_storage():
    """A parent consisting exactly of the sibling block is updated in place."""
    batch, size, stride = 2, 7, 10
    rng = np.random.default_rng(4321)
    xs_data = (
        rng.normal(size=(batch, size + 2))
        + 1j * rng.normal(size=(batch, size + 2))
    ).astype(np.complex64)
    y_data = (
        rng.normal(size=size + 3) + 1j * rng.normal(size=size + 3)
    ).astype(np.complex64)
    with _jax_context():
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        parent = Array(np.full(batch * stride, 9 + 4j, dtype=np.complex64))
        zs = [parent[i * stride: (i + 1) * stride] for i in range(batch)]
        BatchCorrelator(xs, zs, size).batch_correlate_execute(y)
        result = np.asarray(parent)[: batch * stride].reshape(batch, stride)
        np.testing.assert_allclose(
            result[:, :size],
            np.conj(xs_data[:, :size]) * y_data[:size],
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_array_equal(result[:, size:], 9 + 4j)


def test_batch_correlator_uses_bank_tensor_without_materializing_rows():
    """Aligned lazy bank rows are consumed from the shared 2-D tensor."""
    batch, size, stride = 3, 6, 9
    rng = np.random.default_rng(2468)
    bank = (
        rng.normal(size=(batch, size + 3))
        + 1j * rng.normal(size=(batch, size + 3))
    ).astype(np.complex64)
    y_data = (
        rng.normal(size=size + 2) + 1j * rng.normal(size=size + 2)
    ).astype(np.complex64)

    class LazyRow:
        dtype = np.dtype(np.complex64)

        def __init__(self, tensor, pos):
            self._batch_tensor = tensor
            self._batch_pos = pos
            self.ptr = id(self)

        @property
        def _data(self):
            raise AssertionError("contiguous bank path materialized a row")

    with _jax_context():
        bank_jax = jnp.asarray(bank)
        xs = [LazyRow(bank_jax, i) for i in range(batch)]
        y = Array(y_data)
        parent = Array(np.full(batch * stride + 4, 5 - 2j, dtype=np.complex64))
        zs = [parent[i * stride: (i + 1) * stride] for i in range(batch)]

        BatchCorrelator(xs, zs, size).batch_correlate_execute(y)
        result = np.asarray(parent)[: batch * stride].reshape(batch, stride)
        np.testing.assert_allclose(
            result[:, :size],
            np.conj(bank[:, :size]) * y_data[:size],
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_array_equal(result[:, size:], 5 - 2j)


@pytest.mark.parametrize("reference", [False, True])
def test_live_veto_correlation_preserves_full_time_buffer(reference):
    rng = np.random.default_rng(512)
    size = 65
    x = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )
    y = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )
    original = np.full(128, 3 + 4j, dtype=np.complex64)
    with scheme.CPUScheme():
        expected = Array(original)
        correlate(Array(x), Array(y), expected)
        expected = expected.numpy().copy()
    operations = ("correlate",) if reference else ()
    with _jax_context(reference_operations=operations) as ctx:
        output = Array(original)
        storage = output._data
        correlate(Array(x), Array(y), output)
        assert len(output) == len(original)
        assert output._data is storage
        assert output._data.device == ctx.jax_device
        np.testing.assert_array_equal(output.numpy()[size:], original[size:])
        if reference:
            assert output.numpy().tobytes() == expected.tobytes()
        else:
            np.testing.assert_allclose(
                output.numpy(), expected, rtol=2e-6, atol=2e-6
            )


def test_live_jax_groups_preserve_native_template_and_psd_grids():
    def templates():
        rows = []
        for index, (size, delta_f) in enumerate(
            [(65, 0.5)] * 3 + [(129, 0.25)] * 2
        ):
            row = FrequencySeries(np.ones(size, np.complex64), delta_f=delta_f)
            row.id = index
            rows.append(row)
        return rows

    def geometry(control):
        return [
            (
                tuple(row.id for row in group),
                tuple(row.delta_f for row in group),
                tuple(len(row) for row in group),
                int(size),
            )
            for group, size in zip(control.tgroups, control.chunk_tsamples)
        ]

    with scheme.CPUScheme():
        native = LiveBatchMatchedFilter(
            templates(), 5.0, 0, None, maxelements=256
        )
        native_geometry = geometry(native)
        assert not hasattr(native, "unique_delta_fs")
        reader = SimpleNamespace(required_delta_fs=("preserve",))
        native.set_data(reader)
        assert reader.required_delta_fs == ("preserve",)
        assert not hasattr(native.corr[0], "_jax_template_matrix")
    with _jax_context():
        candidate = LiveBatchMatchedFilter(
            templates(), 5.0, 0, None, maxelements=256
        )
        assert geometry(candidate) == native_geometry
        assert candidate.unique_delta_fs == (0.5, 0.25)
        reader = SimpleNamespace()
        candidate.set_data(reader)
        assert reader.required_delta_fs == (0.5, 0.25)


def _selection_control(peak, sigma, threshold, abort_threshold):
    """Hold transforms and normalization reduction fixed around selection."""
    output = np.zeros(8, np.complex64)
    output[3] = peak
    template = FrequencySeries(np.ones(5, np.complex64), delta_f=0.25)
    template.id = 50
    template.approximant = "TaylorF2"
    template.params = np.zeros((), dtype=[])
    template.sigmasq = lambda psd: float(sigma)
    template.out = Array(output)
    template.cout = Array(np.zeros(8, np.complex64))
    stilde = FrequencySeries(np.ones(5, np.complex64), delta_f=0.25)
    stilde.psd = FrequencySeries(np.ones(5, np.float32), delta_f=0.25)
    reader = SimpleNamespace(
        trim_padding=2,
        blocksize=2,
        sample_rate=2,
        start_time=100.0,
        overwhitened_data=lambda df: stilde,
    )
    control = LiveBatchMatchedFilter.__new__(LiveBatchMatchedFilter)
    control.block_id = 0
    control.tgroups = [[template]]
    control.chunk_tsamples = [8]
    control.mids = [0]
    control.out_mem = {0: template.out}
    control.cout_mem = {0: template.cout}
    control.corr = [SimpleNamespace(execute=lambda data: None)]
    control.ifts = {0: SimpleNamespace(execute=lambda: None)}
    control.snr_threshold = threshold
    control.snr_abort_threshold = abort_threshold
    control.data = reader
    return control


@pytest.mark.parametrize("abort_threshold", [None, 2.0])
def test_native_live_selection_restores_threshold_and_scalar_promotion(
    monkeypatch, abort_threshold
):
    peak = np.complex64(-0.17656055 + 0.93697107j)
    norm = 2.7859016728226313
    sigma = (1.0 / norm) ** 2
    threshold = float(abs(peak) * norm)
    with scheme.CPUScheme():
        original = _selection_control(peak, sigma, threshold, abort_threshold)
        expected, expected_veto = original._process_batch()

    def fixed_norms(*args, **kwargs):
        return jnp.asarray([sigma], dtype=jnp.float64)

    def reject_fused_selection(*args, **kwargs):
        raise AssertionError("native live selection used the JAX selector")

    monkeypatch.setattr(backend, "live_template_norms_jax", fixed_norms)
    monkeypatch.setattr(backend, "_live_select_peaks", reject_fused_selection)
    with _jax_context(reference_operations=("live_selection",)) as ctx:
        candidate = _selection_control(peak, sigma, threshold, abort_threshold)
        actual, veto = candidate._process_batch()
        assert scheme.mgr.state is ctx
        if abort_threshold is not None:
            assert expected is actual is False
            assert veto == expected_veto == []
        else:
            assert len(expected["snr"]) == 1
            for key in expected:
                assert np.asarray(actual[key]).dtype == expected[key].dtype
                assert (
                    np.asarray(actual[key]).tobytes()
                    == expected[key].tobytes()
                )
            assert actual["snr"].devices() == {ctx.jax_device}
            assert (
                np.asarray(veto[0][0]).tobytes()
                == expected_veto[0][0].tobytes()
            )
            assert float(veto[0][1]) == expected_veto[0][1]


def _live_reference_control(chisq_bins=0, sg_chisq=None):
    rng = np.random.default_rng(517)
    rows = (rng.normal(size=(2, 65)) + 1j * rng.normal(size=(2, 65))).astype(
        np.complex64
    )
    rows[:, (0, -1)] = 0
    strain = (rng.normal(size=65) + 1j * rng.normal(size=65)).astype(
        np.complex64
    )
    strain[(0, -1),] = 0
    psd = rng.uniform(0.5, 2.0, 65).astype(np.float32)

    templates = []
    for index, values in enumerate(rows):
        template = FrequencySeries(values, delta_f=0.5)
        template.id = index
        template.approximant = "TaylorF2"
        template.f_lower = 0.5
        template.min_f_lower = 0.5
        template.end_frequency = 32.0
        template.params = np.array(
            (index + 1.0,), dtype=[("mass1", np.float64)]
        )
        template.sigmasq = types.MethodType(sigma_cached, template)
        templates.append(template)
    calculator = LiveBatchMatchedFilter(
        templates, 0.0, chisq_bins, sg_chisq, maxelements=256
    )
    stilde = FrequencySeries(strain, delta_f=0.5)
    stilde.psd = FrequencySeries(psd, delta_f=0.5)
    calculator.set_data(
        SimpleNamespace(
            trim_padding=4,
            blocksize=1,
            sample_rate=64,
            start_time=100.0,
            overwhitened_data=lambda df: stilde,
        )
    )
    return calculator


def test_live_granular_components_restore_original_batch_and_veto_inputs():
    with scheme.CPUScheme():
        original = _live_reference_control()
        expected, expected_veto = original._process_batch()
    operations = (
        "correlate",
        "ifft",
        "squared_norm",
        "divide",
        "inner",
        "abs_arg_max",
        "live_selection",
    )
    with _jax_context(reference_operations=operations) as ctx:
        candidate = _live_reference_control()
        actual, veto = candidate._process_batch()
        for key in expected:
            assert np.asarray(actual[key]).dtype == expected[key].dtype
            assert np.asarray(actual[key]).tobytes() == expected[key].tobytes()
        assert actual["snr"].devices() == {ctx.jax_device}
        assert len(veto) == len(expected_veto)
        for native, selected in zip(expected_veto, veto):
            assert np.asarray(selected[0]).tobytes() == native[0].tobytes()
            assert float(selected[1]) == native[1]
            assert selected[2] == native[2]


def test_live_process_data_native_components_restore_actual_veto_results():
    from pycbc.vetoes.chisq import SingleDetPowerChisq

    class RecordingPowerChisq(SingleDetPowerChisq):
        def __init__(self):
            super().__init__("4", None)
            self.inputs = []

        def values(self, corr, snrv, norm, psd, indices, template):
            self.inputs.append(
                (np.asarray(corr).copy(), np.asarray(snrv).copy(),
                 norm, np.asarray(indices), template.id)
            )
            return super().values(corr, snrv, norm, psd, indices, template)

    def execute():
        sg_norms = []

        def disabled_sg(stilde, htilde, psd, snrv, norm, c, d, indices):
            sg_norms.append(norm)
            return None

        calculator = _live_reference_control(
            "4", SimpleNamespace(values=disabled_sg)
        )
        calculator.power_chisq = RecordingPowerChisq()
        result = calculator.process_data(calculator.data)
        return result, calculator.power_chisq.inputs, sg_norms

    with scheme.CPUScheme():
        expected, expected_inputs, expected_sg_norms = execute()
    operations = (
        "correlate", "ifft", "squared_norm", "divide", "inner",
        "abs_arg_max", "live_selection", "power_chisq_bins",
        "power_chisq_at_points",
    )
    with _jax_context(reference_operations=operations) as ctx:
        actual, inputs, sg_norms = execute()
        assert actual.keys() == expected.keys()
        for key in expected:
            got, want = np.asarray(actual[key]), np.asarray(expected[key])
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            assert got.tobytes() == want.tobytes()
            assert actual[key].device == ctx.jax_device
        assert len(inputs) == len(expected_inputs)
        for got, want in zip(inputs, expected_inputs):
            assert type(got[2]) is type(want[2]) is float
            for selected, original in zip(got, want):
                assert np.asarray(selected).dtype == np.asarray(original).dtype
                assert (np.asarray(selected).tobytes()
                        == np.asarray(original).tobytes())
        assert len(sg_norms) == len(expected_sg_norms)
        for got, want in zip(sg_norms, expected_sg_norms):
            assert type(got) is type(want) is float
            assert got == want


@pytest.fixture(params=["cpu", "cuda:0"])
def jax_device(request):
    if request.param == "cuda:0":
        try:
            available = jax.devices("gpu")
        except RuntimeError:
            available = []
        if not available:
            pytest.skip("CUDA JAX device unavailable")
    return request.param


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_live_full_construction_preserves_dtype_tail_bits_and_snapshots(
        jax_device, dtype):
    ("Full-row construction retains scatter rounding "
     "and untouched tail bits.")
    batch, size, stride = 3, 9, 16
    rng = np.random.default_rng(7261)
    bank = (rng.normal(size=(batch, size))
            + 1j * rng.normal(size=(batch, size))).astype(dtype)
    strain = (rng.normal(size=size)
              + 1j * rng.normal(size=size)).astype(dtype)
    initial = (rng.normal(size=batch * stride)
               + 1j * rng.normal(size=batch * stride)).astype(np.complex64)
    # Assignment must preserve even NaN payloads and signed zeros in row tails.
    initial.view(np.uint32)[2 * (stride - 2):2 * stride] = (
        0x80000000, 0, 0x7fc01234, 0x80000000)
    with scheme.JAXScheme(jax_device):
        templates, y, parent = (jnp.asarray(value)
                                for value in (bank, strain, initial))
        before = np.asarray(parent).view(np.uint32).copy()
        constructed = backend._live_construct_correlate_jax(
            templates, y, parent, stride)
        reference = backend._batch_correlate_update(
            templates, y, parent, 0, stride)
        assert constructed.dtype == parent.dtype
        np.testing.assert_array_equal(np.asarray(constructed).view(np.uint32),
                                      np.asarray(reference).view(np.uint32))
        np.testing.assert_array_equal(
            np.asarray(parent).view(np.uint32), before)
        actual_rows = np.asarray(constructed).reshape(batch, stride)
        expected_tails = initial.reshape(batch, stride)[:, size:]
        np.testing.assert_array_equal(actual_rows[:, size:].view(np.uint32),
                                      expected_tails.view(np.uint32))


@pytest.mark.parametrize("reference_operations", [(), ("correlate",)])
def test_owned_full_cuda_workspace_uses_construction_and_keeps_aliases(
        jax_device, monkeypatch, reference_operations):
    """Full CUDA roots construct; original correlation keeps writable aliases."""
    batch, size, stride = 2, 7, 10
    with scheme.JAXScheme(jax_device,
                          reference_operations=reference_operations):
        bank = [Array(np.arange(size, dtype=np.complex64) + 1j * (i + 1))
                for i in range(batch)]
        parent = Array(np.full(batch * stride, 9 + 4j, np.complex64))
        views = [parent[i * stride:(i + 1) * stride] for i in range(batch)]
        correlator = BatchCorrelator(
            bank, views, size, immutable_templates=True)
        backend._bind_live_correlate_workspace_jax(correlator, parent)
        original_construct = backend._live_construct_correlate_jax
        original_native = backend._cpu_correlate
        calls, native_calls = [], []

        def construct(*args):
            calls.append(True)
            return original_construct(*args)

        def native(*args):
            native_calls.append(True)
            return original_native(*args)

        monkeypatch.setattr(
            backend, "_live_construct_correlate_jax", construct)
        monkeypatch.setattr(backend, "_cpu_correlate", native)
        alias = Array(parent, copy=False)
        initial = np.asarray(parent).copy() + np.complex64(1 - 2j)
        parent._data.set_array(jnp.asarray(initial))
        retained = parent._data.array
        correlator.execute(Array(np.arange(size, dtype=np.complex64) + 2j))
        np.testing.assert_array_equal(np.asarray(retained), initial)
        np.testing.assert_array_equal(np.asarray(alias), np.asarray(parent))
        np.testing.assert_array_equal(
            np.asarray(parent).reshape(batch, stride)[:, size:],
            initial.reshape(batch, stride)[:, size:])
        expected_construct = int(jax_device != "cpu"
                                 and not reference_operations)
        assert len(calls) == expected_construct
        assert len(native_calls) == (batch if reference_operations else 0)


def _workspace(*, immutable=True, nested=False):
    """Small bank with nonzero prefix, row tails, and root suffix."""
    batch, size, stride, base, tail = 3, 5, 8, 2, 4
    outer = 4 if nested else 0
    rng = np.random.default_rng(5102)
    bank = (rng.normal(size=(batch, size)) +
            1j * rng.normal(size=(batch, size))).astype(np.complex64)
    y_data = (rng.normal(size=size + 2) +
              1j * rng.normal(size=size + 2)).astype(np.complex64)
    length = base + batch * stride + tail
    root = Array((np.arange(length + 2 * outer) * (0.2 + 0.3j)
                  + 2 - 1j).astype(np.complex64))
    owner = root[outer:outer + length] if nested else Array(root, copy=False)
    zs = [owner[base + row * stride:base + (row + 1) * stride]
          for row in range(batch)]
    correlator = BatchCorrelator(
        [Array(row) for row in bank], zs, size,
        immutable_templates=immutable,
    )
    return SimpleNamespace(
        correlator=correlator, root=root, owner=owner, bank=bank,
        y=Array(y_data), y_data=y_data, size=size, stride=stride,
        starts=[outer + base + row * stride for row in range(batch)],
    )


def _assert_writes(root, before, writes):
    """NumPy assignment oracle also checks every untouched sample exactly."""
    expected = before.copy()
    touched = np.zeros(before.size, dtype=bool)
    for start, values in writes:
        expected[start:start + len(values)] = values
        touched[start:start + len(values)] = True
    actual = np.asarray(root)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual[~touched], before[~touched])
    return expected


def test_owned_workspace_reuses_geometry_with_current_buffers(
        jax_device, monkeypatch):
    """Warm execution reuses geometry, never the previous block's values."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace()
        correlator = ws.correlator
        backend._bind_live_correlate_workspace_jax(correlator, ws.owner)
        workspace = correlator._jax_live_workspace
        assert isinstance(correlator.zs, tuple)
        assert correlator.zs is workspace.outputs
        alias = Array(ws.root, copy=False)

        def unexpected_scan(*args):
            raise AssertionError("owned warm path rescanned output geometry")

        monkeypatch.setattr(backend, "_batch_correlate_geometry_jax",
                            unexpected_scan)
        for scale in (np.complex64(1), np.complex64(0.25 - 0.75j)):
            before = (np.asarray(ws.root) + np.complex64(3 - 2j))
            ws.root._data.set_array(jnp.asarray(before))
            current_y = ws.y_data * scale
            ws.y._data.set_array(jnp.asarray(current_y))
            correlator.batch_correlate_execute(ws.y)
            products = np.conj(ws.bank) * current_y[None, :ws.size]
            expected = _assert_writes(
                ws.root, before, zip(ws.starts, products),
            )
            np.testing.assert_array_equal(np.asarray(alias),
                                          np.asarray(ws.root))
            for z, start in zip(correlator.zs, ws.starts):
                np.testing.assert_allclose(
                    np.asarray(z), expected[start:start + ws.stride],
                    rtol=2e-6, atol=2e-6,
                )
            assert correlator._jax_live_workspace is workspace


@pytest.mark.parametrize("change", [
    "reorder", "replace_outputs", "size", "xs", "matrix", "owner",
    "root_shape", "root_dtype", "root_parent",
])
def test_owned_workspace_invalidates_changed_geometry(
        jax_device, monkeypatch, change):
    """Invalidated ownership must execute the current generic contract."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace()
        correlator = ws.correlator
        backend._bind_live_correlate_workspace_jax(correlator, ws.owner)
        correlator.batch_correlate_execute(ws.y)
        previous_root = None
        bank = ws.bank
        if change == "reorder":
            correlator.zs = tuple(reversed(correlator.zs))
            ws.starts.reverse()
        elif change == "replace_outputs":
            previous_root = (ws.root, np.asarray(ws.root).copy())
            ws.root = Array(np.full(len(ws.root), 8 + 7j, np.complex64))
            correlator.zs = tuple(
                ws.root[start:start + ws.stride] for start in ws.starts
            )
        elif change == "size":
            correlator.size -= 2
        elif change == "xs":
            correlator.xs = list(correlator.xs)
        elif change == "matrix":
            bank = bank * np.complex64(0.5 + 0.25j)
            correlator._jax_template_matrix = jnp.asarray(bank)
        elif change == "owner":
            ws.owner._data = Array(
                np.full(len(ws.root), 4 + 9j, np.complex64)
            )._data
        elif change == "root_shape":
            ws.root._data.array = jnp.concatenate([
                ws.root._data.array, jnp.full(3, 9 - 5j, dtype=jnp.complex64),
            ])
        elif change == "root_dtype":
            ws.root._data.array = ws.root._data.array.astype(jnp.complex128)
            ws.root._data.dtype = np.dtype(np.complex128)
        elif change == "root_parent":
            inner = np.asarray(ws.root)
            outer = Array(np.concatenate([
                np.full(4, 6 - 5j, np.complex64), inner,
                np.full(4, -3 + 2j, np.complex64),
            ]))
            parent = ws.root._data
            parent.parent = outer._data
            parent.slice_info = slice(4, 4 + len(inner))
            parent.array = None
            ws.root = outer
            ws.starts = [start + 4 for start in ws.starts]

        scans = []
        original = backend._batch_correlate_geometry_jax

        def record_scan(outputs, size):
            scans.append((outputs, size))
            return original(outputs, size)

        monkeypatch.setattr(backend, "_batch_correlate_geometry_jax",
                            record_scan)
        before = np.asarray(ws.root).copy()
        owner_before = np.asarray(ws.owner).copy()
        correlator.batch_correlate_execute(ws.y)
        products = (np.conj(bank[:, :correlator.size])
                    * ws.y_data[None, :correlator.size])
        _assert_writes(ws.root, before, zip(ws.starts, products))
        assert correlator._jax_live_workspace is None
        assert len(scans) == 1
        if previous_root is not None:
            np.testing.assert_array_equal(np.asarray(previous_root[0]),
                                          previous_root[1])
        if change == "owner":
            np.testing.assert_array_equal(np.asarray(ws.owner), owner_before)


@pytest.mark.parametrize("unsupported", [
    "mutable_templates", "nested_owner", "wrong_owner", "unrelated_outputs",
    "matrix_shape",
])
def test_private_workspace_binding_rejects_unowned_layouts(
        jax_device, unsupported):
    """Unsupported binding leaves ordinary correlation available."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace(immutable=unsupported != "mutable_templates",
                        nested=unsupported == "nested_owner")
        correlator = ws.correlator
        owner = ws.owner
        if unsupported == "wrong_owner":
            owner = Array(np.asarray(ws.owner))
        elif unsupported == "unrelated_outputs":
            correlator.zs = [Array(np.full(ws.stride, 5 + 2j, np.complex64))
                             for _ in ws.bank]
        elif unsupported == "matrix_shape":
            correlator.size = 3
            # The binding requires the complete fixed matrix geometry.
        outputs = correlator.zs
        backend._bind_live_correlate_workspace_jax(correlator, owner)
        assert getattr(correlator, "_jax_live_workspace", None) is None
        assert correlator.zs is outputs
        before = np.asarray(ws.root).copy()
        correlator.batch_correlate_execute(ws.y)
        products = (np.conj(ws.bank[:, :correlator.size])
                    * ws.y_data[None, :correlator.size])
        if unsupported == "unrelated_outputs":
            np.testing.assert_array_equal(np.asarray(ws.root), before)
            for z, product in zip(outputs, products):
                np.testing.assert_allclose(np.asarray(z)[:ws.size], product,
                                           rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(np.asarray(z)[ws.size:], 5 + 2j)
        else:
            _assert_writes(ws.root, before, zip(ws.starts, products))


def test_failed_rebind_clears_previous_owned_workspace(jax_device):
    """Rejected ownership cannot retain a previously qualified workspace."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace()
        correlator = ws.correlator
        backend._bind_live_correlate_workspace_jax(correlator, ws.owner)
        assert correlator._jax_live_workspace is not None
        outputs = correlator.zs
        wrong_owner = Array(np.asarray(ws.owner))
        wrong_before = np.asarray(wrong_owner).copy()
        backend._bind_live_correlate_workspace_jax(correlator, wrong_owner)
        assert correlator._jax_live_workspace is None
        assert correlator.zs is outputs
        before = np.asarray(ws.root).copy()
        correlator.batch_correlate_execute(ws.y)
        products = np.conj(ws.bank) * ws.y_data[None, :ws.size]
        _assert_writes(ws.root, before, zip(ws.starts, products))
        np.testing.assert_array_equal(np.asarray(wrong_owner), wrong_before)


@pytest.mark.parametrize("immutable", [False, True])
@pytest.mark.parametrize("change", [
    "reorder", "replace_row", "row_data", "row_parent", "row_slice",
])
def test_generic_correlator_observes_output_mutations(
        jax_device, immutable, change):
    """Frozen template values do not freeze generic mutable output geometry."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace(immutable=immutable)
        correlator = ws.correlator
        correlator.batch_correlate_execute(ws.y)
        other = Array(np.full(len(ws.root), 6 + 9j, np.complex64))
        destinations = [(ws.root, start) for start in ws.starts]
        if change == "reorder":
            correlator.zs.reverse()
            destinations.reverse()
        elif change in ("replace_row", "row_data"):
            replacement = other[3:3 + ws.stride]
            if change == "replace_row":
                correlator.zs[1] = replacement
            else:
                correlator.zs[1]._data = replacement._data
            destinations[1] = (other, 3)
        elif change == "row_parent":
            correlator.zs[1]._data.parent = other._data
            destinations[1] = (other, ws.starts[1])
        elif change == "row_slice":
            start = ws.starts[2]
            correlator.zs[1]._data.slice_info = slice(start, start + ws.stride)
            destinations[1] = (ws.root, start)

        before = np.asarray(ws.root).copy()
        other_before = np.asarray(other).copy()
        current_y = ws.y_data * np.complex64(0.25 - 0.5j)
        ws.y._data.set_array(jnp.asarray(current_y))
        correlator.batch_correlate_execute(ws.y)
        products = np.conj(ws.bank) * current_y[None, :ws.size]
        for root, baseline in ((ws.root, before), (other, other_before)):
            writes = [(start, row) for (destination, start), row
                      in zip(destinations, products) if destination is root]
            expected = _assert_writes(root, baseline, writes)
            for z, (destination, start) in zip(correlator.zs, destinations):
                if destination is root:
                    np.testing.assert_allclose(
                        np.asarray(z), expected[start:start + ws.stride],
                        rtol=2e-6, atol=2e-6,
                    )
        assert getattr(correlator, "_jax_live_workspace", None) is None


@pytest.mark.parametrize("nested", [False, True])
def test_generic_mutable_template_output_aliases_use_current_values(
        jax_device, nested):
    """Inputs aliasing outputs must be consumed before any row is published."""
    with scheme.JAXScheme(jax_device):
        ws = _workspace(immutable=False, nested=nested)
        correlator = ws.correlator
        correlator.xs = list(reversed(correlator.zs))
        for scale in (np.complex64(1), np.complex64(-0.3 + 0.2j)):
            before = np.asarray(ws.root).copy()
            current_y = ws.y_data * scale
            ws.y._data.set_array(jnp.asarray(current_y))
            inputs = np.stack([np.asarray(x)[:ws.size]
                               for x in correlator.xs])
            products = np.conj(inputs) * current_y[None, :ws.size]
            correlator.batch_correlate_execute(ws.y)
            _assert_writes(ws.root, before, zip(ws.starts, products))
            assert getattr(correlator, "_jax_live_workspace", None) is None
