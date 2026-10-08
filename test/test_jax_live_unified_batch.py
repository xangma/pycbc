# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Unit tests for duration-preserving batching in JAX LiveBatchMatchedFilter."""

from types import MethodType, SimpleNamespace
import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter.matchedfilter import LiveBatchMatchedFilter
from pycbc.types import FrequencySeries

jax = pytest.importorskip("jax")
jnp = jax.numpy
from pycbc.types.array_jax import _ensure_x64, to_jax  # noqa: E402


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _make_dummy_template(template_id, duration, sample_rate=2048, flow=30.0):
    """Create a dummy frequency-domain template for testing."""
    tsamples = int(round(duration * sample_rate))
    delta_f = 1.0 / duration
    flen = tsamples // 2 + 1

    rng = np.random.default_rng(1000 + int(template_id))
    kmin = int(flow / delta_f)
    freqs = np.linspace(0, sample_rate / 2, flen)
    data = np.zeros(flen, dtype=np.complex64)
    data[kmin:] = (rng.normal(size=flen - kmin) +
                   1j * rng.normal(size=flen - kmin)).astype(np.complex64)
    data[kmin:] *= (freqs[kmin:] / flow) ** (-7.0 / 6.0)

    t = FrequencySeries(data, delta_f=delta_f)
    t.id = template_id
    t.f_lower = flow
    t.end_frequency = sample_rate / 2.0
    t.end_idx = flen
    t.approximant = "TaylorF2"
    t.params = np.array(
        [(1.4, 1.4, 0.0, 0.0, flow, template_id, "TaylorF2", duration)],
        dtype=[
            ("mass1", "<f4"), ("mass2", "<f4"), ("spin1z", "<f4"),
            ("spin2z", "<f4"), ("f_lower", "<f4"), ("template_hash", "<i8"),
            ("approximant", "O"), ("template_duration", "<f4"),
        ],
    )[0]
    return t




@pytest.mark.parametrize("device", _devices())
def test_live_batch_matched_filter_jax_preserves_duration_groups(device):
    """Batch matching durations while retaining each bank template and grid."""
    _ensure_x64()
    sample_rate = 2048
    durations = [24.0, 32.0, 24.0, 64.0, 32.0, 40.0]
    templates = [
        _make_dummy_template(i, dur, sample_rate=sample_rate)
        for i, dur in enumerate(durations)
    ]

    mock_sg = SimpleNamespace(do=False)

    original_spectra = [template.numpy().copy() for template in templates]
    with scheme.JAXScheme(device=device):
        mf_jax = LiveBatchMatchedFilter(templates, 5.5, "16", mock_sg)
        assert list(mf_jax.chunks) == [2, 2, 1, 1]
        assert mf_jax.unique_delta_fs == (1 / 24, 1 / 32, 1 / 40, 1 / 64)
        for group in mf_jax.tgroups:
            assert len({template.delta_f for template in group}) == 1
            for template in group:
                assert template is templates[template.id]
                np.testing.assert_array_equal(template.numpy(),
                                              original_spectra[template.id])

    with scheme.CPUScheme():
        # Fresh copy of templates since LiveBatchMatchedFilter modifies
        templates_cpu = [
            _make_dummy_template(i, dur, sample_rate=sample_rate)
            for i, dur in enumerate(durations)
        ]
        mf_cpu = LiveBatchMatchedFilter(templates_cpu, 5.5, "16", mock_sg)
        np.testing.assert_array_equal(mf_jax.chunks, mf_cpu.chunks)
        assert mf_jax.unique_delta_fs == tuple(sorted(
            {group[0].delta_f for group in mf_cpu.tgroups}, reverse=True))


@pytest.mark.parametrize("device", _devices())
def test_live_equal_sized_chunks_keep_first_group_triggers(device):
    """Repeated chunk geometries must transform each group's correlations."""
    _ensure_x64()
    with scheme.JAXScheme(device=device):
        templates = [_make_dummy_template(i, 4.0, 128, 8.0)
                     for i in range(4)]
        sg_chisq = SimpleNamespace(do=False, values=lambda *args: None)
        control = LiveBatchMatchedFilter(
            templates, 0.01, "4", sg_chisq,
            maxelements=2 * 512,
        )
        rng = np.random.default_rng(112)
        data = FrequencySeries(
            (rng.normal(size=257) + 1j * rng.normal(size=257))
            .astype(np.complex64), delta_f=0.25,
        )
        data.psd = FrequencySeries(np.ones(257, np.float32), delta_f=0.25)
        reader = SimpleNamespace(
            overwhitened_data=lambda delta_f: data, trim_padding=0,
            blocksize=2, sample_rate=128, start_time=100.0,
        )
        control.set_data(reader)
        first, _ = control._process_batch()
        second, _ = control._process_batch()
        np.testing.assert_array_equal(first["template_id"], [0, 1])
        np.testing.assert_array_equal(second["template_id"], [2, 3])

        # process_all evaluates scalar vetoes after every group's shared
        # workspace has been overwritten. Compare with one full bank batch.
        actual = control.process_data(reader)
        reference = LiveBatchMatchedFilter(
            [_make_dummy_template(i, 4.0, 128, 8.0) for i in range(4)],
            0.01, "4", sg_chisq, maxelements=4 * 512,
        ).process_data(reader)
        np.testing.assert_array_equal(actual["template_id"], [0, 1, 2, 3])
        for key in ("snr", "coa_phase", "end_time", "sigmasq", "chisq",
                    "chisq_dof", "sg_chisq"):
            np.testing.assert_allclose(actual[key], reference[key],
                                       rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_live_template_matrix_keeps_each_bank_source(device):
    from pycbc.filter.matchedfilter_jax import batch_template_matrix_jax

    with scheme.JAXScheme(device=device):
        templates = [_make_dummy_template(i, 4.0, 128, 8.0)
                     for i in range(2)]
        for template in templates:
            template._batch_tensor = to_jax(template)[None, :]
            template._batch_pos = 0
        matrix = batch_template_matrix_jax(templates, len(templates[0]))
        np.testing.assert_array_equal(matrix, np.stack(
            [np.asarray(to_jax(template)) for template in templates]))


@pytest.fixture(params=_devices())
def mixed_duration_results(request):
    """Filter corresponding trailing windows of two compact real signals."""
    from pycbc.waveform.bank import sigma_cached

    _ensure_x64()
    sample_rate, trim = 2048, 256
    n = 8 * sample_rate
    times = np.arange(n) / sample_rate - 8
    signals = [np.exp(-((times + 0.25) / 0.07)**2)
               * np.cos(2 * np.pi * frequency * (times + 0.25))
               for frequency in (96, 160)]
    point = n - trim - sample_rate + sample_rate // 2
    strain = np.roll(signals[0] + 0.7 * signals[1], point).astype(np.float32)

    def templates():
        result = []
        for index, duration in enumerate((4, 8)):
            template = _make_dummy_template(index, duration, sample_rate)
            spectrum = np.fft.rfft(signals[index][-duration * sample_rate:])
            template[:] = FrequencySeries(
                (spectrum / sample_rate).astype(np.complex64),
                delta_f=template.delta_f,
            )
            template.min_f_lower = template.f_lower
            template.sigmasq = MethodType(sigma_cached, template)
            result.append(template)
        return result

    def reader():
        segments, requested = {}, []

        def overwhitened(delta_f):
            requested.append(delta_f)
            if delta_f not in segments:
                length = int(round(sample_rate / delta_f))
                # White PSD=2: overwhitening divides the physical FFT by 2.
                spectrum = np.fft.rfft(strain[-length:]) / sample_rate / 2
                data = FrequencySeries(spectrum.astype(np.complex64),
                                       delta_f=delta_f)
                data.psd = FrequencySeries(
                    np.full(length // 2 + 1, 2, np.float32), delta_f=delta_f,
                )
                segments[delta_f] = data
            return segments[delta_f]

        return SimpleNamespace(overwhitened_data=overwhitened,
                               trim_padding=trim, blocksize=1,
                               sample_rate=sample_rate, start_time=100.0,
                               requested=requested)

    sg = SimpleNamespace(do=False, values=lambda *args: None)
    with scheme.CPUScheme():
        data = reader()
        reference = LiveBatchMatchedFilter(templates(), 0.01, "4", sg)
        expected = reference.process_data(data)
        assert list(reference.chunks) == [1, 1]
        assert data.requested == [0.25, 0.125]
    with scheme.JAXScheme(request.param):
        data = reader()
        control = LiveBatchMatchedFilter(templates(), 0.01, "4", sg)
        actual = control.process_data(data)
        assert list(control.chunks) == [1, 1]
        assert data.required_delta_fs == (0.25, 0.125)
        assert data.requested == [0.25, 0.125]
    return expected, actual


def test_live_mixed_duration_filtering_matches_scalar_cpu(mixed_duration_results):
    """Check corresponding trailing windows of compact flat-PSD signals."""
    expected, actual = mixed_duration_results
    np.testing.assert_array_equal(expected["template_id"], [0, 1])
    np.testing.assert_array_equal(expected["end_time"], [100.5, 100.5])
    assert np.all(expected["snr"] > 0.1)
    for key in ("template_id", "end_time", "chisq_dof", "template_hash"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("snr", "sigmasq", "coa_phase"):
        np.testing.assert_allclose(actual[key], expected[key],
                                   rtol=1e-4, atol=1e-5)


def test_live_mixed_duration_chisq_matches_scalar_cpu(mixed_duration_results):
    """The original duration preserves veto values at the frozen tolerance."""
    expected, actual = mixed_duration_results
    np.testing.assert_allclose(actual["chisq"], expected["chisq"],
                               rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_live_broadband_coloured_psd_matches_scalar_cpu(device):
    """Preserve original FFT/PSD geometry for broadband mixed-duration banks."""
    from pycbc.waveform.bank import sigma_cached

    _ensure_x64()
    sample_rate, trim = 128, 32
    strain = np.random.default_rng(917).normal(size=8 * sample_rate)

    def templates():
        result = [_make_dummy_template(i, duration, sample_rate, 8.0)
                  for i, duration in enumerate((4, 4, 8, 8))]
        for template in result:
            template.min_f_lower = template.f_lower
            template.sigmasq = MethodType(sigma_cached, template)
        return result

    def reader():
        def overwhitened(delta_f):
            length = int(round(sample_rate / delta_f))
            frequencies = np.arange(length // 2 + 1) * delta_f
            psd = (1 + (frequencies / 20)**2
                   + 0.3 * np.cos(frequencies / 7)**2).astype(np.float32)
            spectrum = np.fft.rfft(strain[-length:]) / sample_rate / psd
            data = FrequencySeries(spectrum.astype(np.complex64),
                                   delta_f=delta_f)
            data.psd = FrequencySeries(psd, delta_f=delta_f)
            return data

        return SimpleNamespace(overwhitened_data=overwhitened,
                               trim_padding=trim, blocksize=1,
                               sample_rate=sample_rate, start_time=100.0)

    sg = SimpleNamespace(do=False, values=lambda *args: None)
    with scheme.CPUScheme():
        expected = LiveBatchMatchedFilter(
            templates(), 0.001, "4", sg).process_data(reader())
    with scheme.JAXScheme(device):
        actual = LiveBatchMatchedFilter(
            templates(), 0.001, "4", sg).process_data(reader())

    np.testing.assert_array_equal(expected["template_id"], [0, 1, 2, 3])
    for key in ("template_id", "end_time", "chisq_dof", "template_hash"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("snr", "chisq"):
        np.testing.assert_allclose(actual[key], expected[key],
                                   rtol=1e-4, atol=1e-5)
    phase_difference = np.angle(np.exp(
        1j * (np.asarray(actual["coa_phase"]) - expected["coa_phase"])))
    assert np.all(np.abs(phase_difference) <= 1e-4)
    np.testing.assert_allclose(actual["sigmasq"], expected["sigmasq"],
                               rtol=1e-5, atol=1e-5)


_GROUP_DISPATCH_DEVICES = [
    pytest.param("cpu", id="cpu-orchestration"),
    pytest.param("cuda:0", id="cuda-native", marks=pytest.mark.skipif(
        "cuda:0" not in _devices(), reason="requires CUDA")),
]


def _group_dispatch_inputs(*, tied=False, empty_duration=None, offsets=True):
    """A real compact bank whose duration chunks reuse both FFT workspaces."""
    from pycbc.waveform.bank import sigma_cached

    sample_rate, trim = 128, 32
    durations = (8, 4, 8, 4, 4, 8, 4)
    templates = []
    for index, duration in enumerate(durations):
        template = _make_dummy_template(index, duration, sample_rate, 8.0)
        if tied:
            # Equal spectra/norms produce actual equal SNRs across distinct
            # chunks; neither the FFT nor peak selection is mocked.
            prototype = _make_dummy_template(41, duration, sample_rate, 8.0)
            template[:] = prototype
        template.min_f_lower = template.f_lower
        template.sigmasq = MethodType(sigma_cached, template)
        if offsets:
            template.time_offset = np.float32(index / 1024)
        templates.append(template)

    strain = np.random.default_rng(971).normal(size=8 * sample_rate)
    segments = {}
    for duration in (4, 8):
        length = duration * sample_rate
        delta_f = 1.0 / duration
        frequencies = np.arange(length // 2 + 1) * delta_f
        psd = (1 + (frequencies / 19)**2
               + .2 * np.cos(frequencies / 5)**2).astype(np.float32)
        spectrum = np.fft.rfft(strain[-length:]) / sample_rate / psd
        if empty_duration in (duration, "all"):
            spectrum[:] = 0
        segment = FrequencySeries(spectrum.astype(np.complex64),
                                  delta_f=delta_f)
        segment.psd = FrequencySeries(psd, delta_f=delta_f)
        segments[delta_f] = segment
    requested = []

    def overwhitened(delta_f):
        requested.append(delta_f)
        return segments[delta_f]

    reader = SimpleNamespace(
        overwhitened_data=overwhitened, trim_padding=trim, blocksize=1,
        sample_rate=sample_rate, start_time=100.0, requested=requested)
    return templates, reader, segments


def _group_dispatch_control(templates, **options):
    """Keep the real power veto and record its selected SG input alignment."""
    calls = []

    def sg_values(stilde, template, psd, snr, norm, chisq, dof, points):
        assert psd is stilde.psd
        calls.append((int(template.id), int(points[0]),
                      tuple(np.asarray(value).copy()
                            for value in (snr, norm, chisq, dof))))
        # Tag each selected template/time pair after its real power chi-square.
        return jnp.asarray([template.id / 16 + points[0] / 4096],
                           dtype=jnp.float32)

    control = LiveBatchMatchedFilter(
        templates, 1e-8, "4", SimpleNamespace(do=True, values=sg_values),
        maxelements=1024, **options)
    assert list(control.chunks) == [2, 2, 1, 1, 1]
    assert len(set(control.mids)) == 2
    assert control.power_chisq.do
    return control, calls


def _assert_group_dispatch_result_exact(actual, expected):
    assert list(actual) == list(expected)
    for key in expected:
        assert isinstance(actual[key], type(expected[key])), key
        if isinstance(expected[key], jax.Array):
            assert actual[key].device == expected[key].device, key
        value, reference = np.asarray(actual[key]), np.asarray(expected[key])
        assert value.shape == reference.shape, key
        assert value.dtype == reference.dtype, key
        if reference.dtype.kind in "OU":
            np.testing.assert_array_equal(value, reference, err_msg=key)
        else:
            assert value.tobytes() == reference.tobytes(), key


def _assert_group_dispatch_veto_inputs_exact(actual, expected):
    assert len(actual) == len(expected)
    for (template, point, values), (ref_template, ref_point, ref_values) in zip(
            actual, expected):
        assert (template, point) == (ref_template, ref_point)
        for value, reference in zip(values, ref_values):
            assert value.shape == reference.shape
            assert value.dtype == reference.dtype
            assert value.tobytes() == reference.tobytes()


def _assert_group_dispatch_inputs_unchanged(templates, segments, snapshots):
    values = [np.asarray(template) for template in templates]
    for segment in segments.values():
        values.extend((np.asarray(segment), np.asarray(segment.psd)))
    for value, reference in zip(values, snapshots):
        assert value.dtype == reference.dtype
        assert value.tobytes() == reference.tobytes()


def _run_group_dispatch(control, reader, device, active, monkeypatch):
    from pycbc.filter import matchedfilter_jax as module

    if device != "cpu":
        return control.process_data(reader)

    # Force only the helper's initial orchestration guard. Restore the real
    # device before any to_jax placement or device-specific veto dispatch.
    real_device = active.jax_device

    class CudaDispatchGate:
        @property
        def platform(self):
            active.jax_device = real_device
            return "cuda"

    with monkeypatch.context() as patch:
        patch.setattr(active, "jax_device", CudaDispatchGate())
        result = module.process_live_data_jax(control, reader)
        assert active.jax_device is real_device
    return result


@pytest.mark.parametrize("device", _GROUP_DISPATCH_DEVICES)
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
@pytest.mark.parametrize("case", [
    "mixed", "threshold", "part-empty", "empty", "abort", "truncate-ties",
    "newsnr",
])
def test_grouped_live_dispatch_matches_serial_outputs(
        device, chisq_mode, case, monkeypatch):
    """Actual FFT/filter/veto results retain serial bytes and candidate order.

    CPU exercises grouped orchestration with CPU numerical kernels; the CUDA
    parameter exercises the actual GPU process_data entry point when available.
    This focused bank test does not qualify a complete Live search.
    """
    from pycbc.events import ranking

    _ensure_x64()
    with scheme.JAXScheme(device, chisq_mode=chisq_mode) as active:
        fixture_options = dict(
            tied=case == "truncate-ties",
            offsets=case != "part-empty",
            empty_duration=8 if case == "part-empty" else
                "all" if case == "empty" else None)
        expected_templates, expected_reader, _ = _group_dispatch_inputs(
            **fixture_options)
        templates, reader, segments = _group_dispatch_inputs(**fixture_options)
        snapshots = [np.asarray(template).copy() for template in templates]
        for segment in segments.values():
            snapshots.extend((np.asarray(segment).copy(),
                              np.asarray(segment.psd).copy()))
        expected_control, expected_calls = _group_dispatch_control(
            expected_templates)
        control, calls = _group_dispatch_control(templates)

        if case in ("threshold", "truncate-ties", "newsnr"):
            expected_control.set_data(expected_reader)
            full = expected_control.process_all()
            assert len(full["template_id"]) == len(templates)
            if case == "threshold":
                threshold = float(np.median(np.asarray(full["snr"])))
                expected_control.snr_threshold = control.snr_threshold = threshold
            elif case == "truncate-ties":
                _, counts = np.unique(np.asarray(full["snr"]),
                                      return_counts=True)
                assert max(counts) >= 3
                expected_control.max_triggers_in_batch = 2
                control.max_triggers_in_batch = 2
            else:
                values = ranking.newsnr(full["snr"], full["chisq"])
                threshold = float(np.median(np.asarray(values)))
                assert threshold > 0
                expected_control.newsnr_threshold = threshold
                control.newsnr_threshold = threshold
            expected_calls.clear()
            expected_reader.requested.clear()
        elif case == "abort":
            expected_control.snr_abort_threshold = 0.0
            control.snr_abort_threshold = 0.0

        # process_all keeps the real serial API, independent of process_data's
        # CUDA dispatch hook. Both arms use the same unchanged veto machinery.
        expected_control.set_data(expected_reader)
        expected = expected_control.process_all()
        actual = _run_group_dispatch(control, reader, device, active, monkeypatch)
        assert control.block_id == expected_control.block_id
        assert reader.required_delta_fs == expected_reader.required_delta_fs
        _assert_group_dispatch_inputs_unchanged(templates, segments, snapshots)
        _assert_group_dispatch_veto_inputs_exact(calls, expected_calls)
        if case == "abort":
            assert actual is expected is False
            assert not calls
            return

        _assert_group_dispatch_result_exact(actual, expected)
        count = len(actual["template_id"])
        if case == "empty":
            assert count == 0
            assert not calls
        elif case == "part-empty":
            assert count == 4
            assert set(np.asarray(actual["template_duration"])) == {4.0}
        elif case in ("threshold", "newsnr"):
            assert 0 < count < len(templates)
        elif case == "truncate-ties":
            assert count == 2
            assert np.asarray(actual["snr"])[0] == np.asarray(actual["snr"])[1]
        else:
            assert count == len(templates)
        if case != "newsnr":
            np.testing.assert_array_equal(np.asarray(actual["template_id"]),
                                          [call[0] for call in calls])
            expected_tags = np.asarray(
                [template / 16 + point / 4096
                 for template, point, _ in calls], dtype=np.float32)
            assert np.asarray(actual["sg_chisq"]).tobytes() == expected_tags.tobytes()


@pytest.mark.parametrize("device", _GROUP_DISPATCH_DEVICES)
def test_grouped_live_dispatch_queues_every_group_before_host_decisions(
        device, monkeypatch):
    """Real group reductions all precede the one explicit decision download."""
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(device) as active:
        templates, reader, _ = _group_dispatch_inputs()
        control, _ = _group_dispatch_control(templates)
        original_select, original_get = module._live_select_peaks, jax.device_get
        dispatched, downloads = [], []

        def select(*args, **kwargs):
            values = original_select(*args, **kwargs)
            dispatched.append(values)
            return values

        def download(values):
            # Veto-bin caching may independently retrieve compact edge tables.
            # Count only the actual masks returned by peak selection.
            decision_ids = {
                id(mask) for group in dispatched for mask in group[1:]
            }
            retrieved = tuple(id(value)
                              for value in jax.tree_util.tree_leaves(values)
                              if id(value) in decision_ids)
            if retrieved:
                downloads.append((len(dispatched), retrieved))
            return original_get(values)

        with monkeypatch.context() as patch:
            patch.setattr(module, "_live_select_peaks", select)
            patch.setattr(jax, "device_get", download)
            result = _run_group_dispatch(
                control, reader, device, active, monkeypatch)
        assert len(result["template_id"]) == len(templates)
        assert len(dispatched) == len(control.tgroups)
        expected_masks = tuple(
            id(mask) for group in dispatched for mask in group[1:])
        assert downloads == [(len(control.tgroups), expected_masks)]
        assert control.block_id == len(control.tgroups)


def test_cpu_live_process_data_keeps_serial_decision_boundaries(monkeypatch):
    """Production JAX CPU calls retain their group-at-a-time orchestration."""
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme("cpu"):
        templates, reader, _ = _group_dispatch_inputs()
        control, _ = _group_dispatch_control(templates)
        original_select, original_get = module._live_select_peaks, jax.device_get
        dispatched, downloads = [], []

        def select(*args, **kwargs):
            values = original_select(*args, **kwargs)
            dispatched.append(values)
            return values

        def download(values):
            downloads.append(len(dispatched))
            return original_get(values)

        with monkeypatch.context() as patch:
            patch.setattr(module, "_live_select_peaks", select)
            patch.setattr(jax, "device_get", download)
            result = control.process_data(reader)
        assert len(result["template_id"]) == len(templates)
        assert downloads == list(range(1, len(control.tgroups) + 1))


@pytest.mark.parametrize("device", _GROUP_DISPATCH_DEVICES)
def test_grouped_live_dispatch_retains_empty_bank_error(device, monkeypatch):
    """An empty bank still raises the existing result-assembly error."""
    with scheme.JAXScheme(device) as active:
        sg = SimpleNamespace(do=False, values=lambda *args: None)
        serial = LiveBatchMatchedFilter([], 1e-8, "4", sg)
        control = LiveBatchMatchedFilter([], 1e-8, "4", sg)
        reader = SimpleNamespace()
        serial.set_data(reader)
        with pytest.raises(IndexError):
            serial.process_all()
        with pytest.raises(IndexError):
            _run_group_dispatch(control, reader, device, active, monkeypatch)
