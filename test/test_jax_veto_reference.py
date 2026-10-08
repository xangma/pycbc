"""Original CPU controls for JAX veto and phase calculation boundaries."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_jax import JAXArrayData
from pycbc.vetoes import chisq
from pycbc.vetoes import chisq_jax
from pycbc.vetoes.sgchisq import SingleDetSGChisq
from pycbc.vetoes.sgchisq_jax import _sine_gaussian_basis
from pycbc.waveform.sinegauss import fd_sine_gaussian
from pycbc.waveform.utils import apply_fseries_time_shift


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")
    return request.param


def _exact(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("cropped", [False, True])
def test_original_point_controls(device, dtype, cropped):
    rng = np.random.default_rng(71)
    row = (rng.normal(size=256) + 1j * rng.normal(size=256)).astype(dtype)
    row[:17] = 0
    row[201:] = 0
    bins = np.array([17, 61, 113, 201], dtype=np.uint32)
    points = np.array([3, 31, 125], dtype=np.uint32)
    snrv = np.array([1 + 2j, 2 - 3j, 4 + 1j], dtype=dtype)
    with scheme.CPUScheme():
        corr = FrequencySeries(row, delta_f=.5)
        expected_sum = chisq.shift_sum(corr, points, bins)
        expected_chisq = chisq.power_chisq_at_points_from_precomputed(
            corr, snrv, .123456789, bins, points)
    for operation, expected in [("shift_sum", expected_sum),
                                ("power_chisq_at_points", expected_chisq)]:
        with scheme.JAXScheme(device, reference_operations=[operation]):
            corr = FrequencySeries(row[17:201] if cropped else row, delta_f=.5)
            if cropped:
                corr._kmin, corr._tlen = 17, 256
            actual = (
                chisq.shift_sum(corr, points, bins)
                if operation == "shift_sum"
                else chisq.power_chisq_at_points_from_precomputed(
                    corr, snrv, 0.123456789, bins, points
                )
            )
            _exact(actual, expected)
            assert actual.device == scheme.mgr.state.jax_device


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_original_accumulation_and_bins(device, dtype):
    rng = np.random.default_rng(3)
    row = (rng.normal(size=129) + 1j * rng.normal(size=129)).astype(dtype)
    real_dtype = row.real.dtype
    weight = rng.uniform(.5, 1.5, size=129).astype(real_dtype)
    initial = rng.uniform(size=129).astype(real_dtype)
    with scheme.CPUScheme():
        output = Array(initial.copy())
        chisq.chisq_accum_bin(output, Array(row))
        expected_accum = output.numpy().copy()
        expected_bins = chisq.power_chisq_bins(
            FrequencySeries(row, delta_f=.5), 4,
            FrequencySeries(weight, delta_f=.5), 3, 50)
    with scheme.JAXScheme(device, reference_operations=["chisq_accum_bin"]):
        output = Array(initial.copy())
        chisq.chisq_accum_bin(output, Array(row))
        _exact(output.numpy(), expected_accum)
    with scheme.JAXScheme(device, reference_operations=["power_chisq_bins"]):
        actual = chisq.power_chisq_bins(
            FrequencySeries(row, delta_f=.5), 4,
            FrequencySeries(weight, delta_f=.5), 3, 50)
        _exact(actual, expected_bins)


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_original_time_shift_and_buffer_contract(device, dtype, copy):
    rng = np.random.default_rng(14)
    row = (rng.normal(size=129) + 1j * rng.normal(size=129)).astype(dtype)
    with scheme.CPUScheme():
        expected = apply_fseries_time_shift(
            FrequencySeries(row, delta_f=0.5, epoch=12),
            0.037,
            kmin=11,
            copy=copy,
        )
        expected = expected.numpy().copy()
    with scheme.JAXScheme(device, reference_operations=["time_shift"]):
        original = FrequencySeries(row, delta_f=.5, epoch=12)
        actual = apply_fseries_time_shift(original, .037, kmin=11, copy=copy)
        assert (actual is original) == (not copy)
        _exact(actual.numpy(), expected)
    with scheme.JAXScheme(device):
        original = FrequencySeries(row, delta_f=.5, epoch=12)
        actual = apply_fseries_time_shift(original, .037, kmin=11, copy=copy)
        assert (actual is original) == (not copy)


def test_original_sg_tile(device):
    args = (1., 8., 100., 20., 500., .25)
    with scheme.CPUScheme():
        expected = fd_sine_gaussian(*args).numpy()
    with scheme.JAXScheme(device, reference_operations=["sg_basis"]):
        _exact(_sine_gaussian_basis(*args).numpy(), expected)


def test_default_sg_tile_supports_device_tracing():
    with scheme.JAXScheme("cpu"):
        result = jax.jit(
            lambda: _sine_gaussian_basis(
                1.0, 8.0, 100.0, 20.0, 500.0, 0.25
            )._data.array
        )()
        assert result.shape == (2000,)


@pytest.mark.parametrize("shifts", [.037, np.array([.037, -.019])])
def test_original_batched_time_shifts(device, shifts):
    rng = np.random.default_rng(45)
    rows = (rng.normal(size=(2, 129)) + 1j * rng.normal(size=(2, 129))).astype(
        np.complex64
    )
    expected = []
    with scheme.CPUScheme():
        for row, shift in zip(rows, np.broadcast_to(shifts, (2,))):
            expected.append(
                apply_fseries_time_shift(
                    FrequencySeries(row, delta_f=0.5), float(shift), kmin=3
                ).numpy()
            )
    with scheme.JAXScheme(device, reference_operations=["time_shift"]):
        series = FrequencySeries(JAXArrayData(rows), delta_f=.5, copy=False)
        actual = apply_fseries_time_shift(series, shifts, kmin=3)
        _exact(actual.numpy(), np.array(expected))


@pytest.mark.parametrize("missing_tile", [False, True])
def test_sg_original_early_returns_with_none_epoch(device, missing_tile):
    calculator = SingleDetSGChisq.__new__(SingleDetSGChisq)
    calculator.do = True
    calculator.params = {} if missing_tile else {1: "8-30"}
    calculator.snr_threshold = 100
    calculator.cached_chisq_bins = lambda template, psd: np.array(
        [10, 80, 120, 160]
    )
    if missing_tile:
        def forbidden(*args):
            raise AssertionError("inactive template requested tile bins")
        calculator.cached_chisq_bins = forbidden
    with scheme.JAXScheme(device, reference_operations=["sgchisq"]):
        template = FrequencySeries(
            np.ones(1025, dtype=np.complex64), delta_f=1.0
        )
        template._epoch = None
        template.f_lower = 10
        template.params = SimpleNamespace(template_hash=1)
        result = calculator.values(
            template,
            template,
            FrequencySeries(np.ones(1025, np.float32), delta_f=1.0),
            np.array([1 + 0j], np.complex64),
            1.0,
            np.ones(1),
            np.ones(1),
            np.zeros(1, np.uint32),
        )
        _exact(result, np.ones(1))


def test_original_sg_veto(device):
    rng = np.random.default_rng(4)
    data = (rng.normal(size=1025) + 1j * rng.normal(size=1025)).astype(
        np.complex64
    )
    template = np.ones(1025, dtype=np.complex64)
    psd = np.ones(1025, dtype=np.float32)
    bins = np.array([10, 80, 120, 160], dtype=np.uint32)
    calculator = SingleDetSGChisq.__new__(SingleDetSGChisq)
    calculator.do = True
    calculator.params = {1: "8-30,12-60"}
    calculator.snr_threshold = 0
    calculator.cached_chisq_bins = lambda template, psd: bins
    args = (np.array([10 + 1j, 20 + 2j], dtype=np.complex64), 1.,
            np.ones(2), np.full(2, 6), np.array([1, 41], dtype=np.uint32))

    def evaluate():
        h = FrequencySeries(template, delta_f=1., epoch=0)
        h.f_lower = 10
        h.params = SimpleNamespace(template_hash=1)
        return calculator.values(FrequencySeries(data, delta_f=1.), h,
                                 FrequencySeries(psd, delta_f=1.), *args)

    with scheme.CPUScheme():
        expected = evaluate()
    with scheme.JAXScheme(device, reference_operations=["sgchisq"]):
        _exact(evaluate(), expected)


def test_default_point_sum_stays_on_device(device, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default called native CPU point calculation")

    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu_compatible", forbidden)
    with scheme.JAXScheme(device):
        corr = FrequencySeries(np.ones(64, np.complex64), delta_f=1)
        result = chisq.shift_sum(corr, [3], [1, 16, 32])
        assert result.device == scheme.mgr.state.jax_device
