"""Original-kernel controls reach strain workflows and preserve shared views."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain import strain_jax
from pycbc.strain.strain import StrainBuffer, detect_loud_glitches, execute_cached_fft, gate_data
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import to_jax


DEVICE = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu").split("jax:")[-1]


def _same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.numpy().tobytes() == expected.numpy().tobytes()
    assert actual.start_time == expected.start_time


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_cached_fft_reference_matches_original_plan(dtype, inverse, normalize):
    values = np.random.default_rng(13).normal(size=30).astype(dtype)
    with scheme.CPUScheme():
        source = TimeSeries(values, delta_t=0.2, epoch=1187007104.25)
        if inverse:
            source = execute_cached_fft(source)
        expected = execute_cached_fft(
            source, normalize_by_rate=normalize, ifft=inverse)
        input_values = source.numpy().copy()
        spacing = source.delta_f if inverse else source.delta_t
    operation = "ifft" if inverse else "fft"
    with scheme.JAXScheme(DEVICE, reference_operations=(operation,)) as context:
        source = (FrequencySeries(input_values, delta_f=spacing,
                                  epoch=1187007104.25) if inverse else
                  TimeSeries(input_values, delta_t=spacing,
                             epoch=1187007104.25))
        actual = execute_cached_fft(
            source, normalize_by_rate=normalize, ifft=inverse)
        _same(actual, expected)
        assert to_jax(actual).devices() == {context.jax_device}
        assert actual.delta_t == expected.delta_t


@pytest.mark.parametrize("reference", [False, True])
def test_gating_updates_parent_view(reference):
    gates = [(103.0, 0.5, 0.5), (103.25, 0.25, 0.25)]
    with scheme.CPUScheme():
        parent = TimeSeries(np.ones(32, np.float32), delta_t=0.25, epoch=100)
        gate_data(parent[4:28], gates)
        expected = parent.copy()
    operations = ("gate_data",) if reference else ()
    with scheme.JAXScheme(DEVICE, reference_operations=operations) as context:
        parent = TimeSeries(np.ones(32, np.float32), delta_t=0.25, epoch=100)
        view = parent[4:28]
        storage = parent.data
        assert gate_data(view, gates) is view
        assert parent.data is storage
        np.testing.assert_array_equal(parent.numpy()[4:28], view.numpy())
        assert np.count_nonzero(parent.numpy() == 0) > 0
        assert to_jax(parent).devices() == {context.jax_device}
        if reference:
            _same(parent, expected)
        else:
            np.testing.assert_allclose(parent.numpy(), expected.numpy(), atol=1e-7)


@pytest.mark.parametrize("operation", [
    "welch", "interpolate", "inverse_spectrum_truncation", "fft", "ifft",
])
def test_autogate_routes_selected_stages(operation, monkeypatch):
    values = np.random.default_rng(7).normal(size=512).astype(np.float32)
    with scheme.JAXScheme(DEVICE, reference_operations=(operation,)) as context:
        source = TimeSeries(values, delta_t=1 / 64, epoch=100)
        monkeypatch.setattr(strain_jax, "_fused_autogate_pipeline_core",
                            lambda *a, **k: pytest.fail("reference was fused"))
        conditioned, magnitude = strain_jax._autogate_magnitude_jax(
            source, psd_duration=1, psd_stride=0.5,
            low_freq_cutoff=5, corrupt_time=0.125)
        assert magnitude.shape == source.shape
        assert magnitude.devices() == {context.jax_device}
        assert conditioned.start_time == source.start_time
        np.testing.assert_array_equal(np.asarray(magnitude[:8]), 0)
        np.testing.assert_array_equal(np.asarray(magnitude[-8:]), 0)


def test_original_glitch_detection_control(monkeypatch):
    values = np.random.default_rng(24).normal(size=512).astype(np.float32)
    values[250:253] += 70
    keywords = dict(psd_duration=1, psd_stride=0.5, low_freq_cutoff=5,
                    corrupt_time=0.125, threshold=5, cluster_window=0.25)
    with scheme.CPUScheme():
        expected = detect_loud_glitches(
            TimeSeries(values, delta_t=1 / 64, epoch=1187007104), **keywords)
    with scheme.JAXScheme(DEVICE, reference_operations=("detect_loud_glitches",)):
        source = TimeSeries(values, delta_t=1 / 64, epoch=1187007104)
        monkeypatch.setattr(strain_jax, "_autogate_magnitude_jax",
                            lambda *a, **k: pytest.fail("whole stage used JAX"))
        actual = detect_loud_glitches(source, **keywords)
        assert actual == expected
        assert actual
        assert strain_jax.autogate_strain_buffer_jax(
            SimpleNamespace(strain=source)) is False


@pytest.mark.parametrize("jax_context", [False, True])
def test_psd_invalidation_preserves_original_cpu_cache_behavior(jax_context):
    context = scheme.JAXScheme(DEVICE) if jax_context else scheme.CPUScheme()
    with context:
        buffer = object.__new__(StrainBuffer)
        buffer.psd, buffer.psds = object(), {0.5: object()}
        sentinel = {0.5: object()}
        buffer.segments = sentinel
        buffer.invalidate_psd()
        assert buffer.psd is None and buffer.psds == {}
        if jax_context:
            assert buffer.segments == {}
        else:
            assert buffer.segments is sentinel


@pytest.mark.parametrize("operation", [
    "fir_zero_filter", "lfilter", "resample", "fft", "ifft",
])
def test_live_conditioning_cannot_fuse_selected_kernels(operation):
    buffer = SimpleNamespace(sample_rate=2048, corruption=42, factor=2,
                             highpass_samples=128)
    with scheme.JAXScheme(DEVICE):
        assert strain_jax.can_fuse_strain_buffer_jax(buffer, 2)
    with scheme.JAXScheme(DEVICE, reference_operations=(operation,)):
        assert not strain_jax.can_fuse_strain_buffer_jax(buffer, 2)


def test_live_advance_composes_original_conditioning(monkeypatch):
    from pycbc.frame.frame import DataBuffer

    raw = np.random.default_rng(19).normal(size=16384)
    initial = np.random.default_rng(20).normal(size=8192).astype(np.float32)

    def buffer():
        obj = object.__new__(StrainBuffer)
        obj.raw_buffer = TimeSeries(raw, delta_t=1 / 4096, epoch=100)
        obj.strain = TimeSeries(initial, delta_t=1 / 2048, epoch=100)
        obj.sample_rate, obj.factor = 2048, 2
        obj.corruption, obj.highpass_samples = 42, 128
        obj.highpass_frequency, obj.beta, obj.dyn_range_fac = 25, 5, 2.5
        obj.taper_immediate_strain = False
        obj.state = obj.dq = obj.idq = None
        obj.wait_duration = 2
        obj.autogating_threshold = None
        obj.psd, obj.detector = object(), "H1"
        return obj

    monkeypatch.setattr(DataBuffer, "attempt_advance", lambda self, size,
                        timeout=10: self.raw_buffer[
                            len(self.raw_buffer) - int(size * 4096):])
    with scheme.CPUScheme():
        expected = buffer()
        assert expected.advance(2)
        original = expected.strain.copy()
    with scheme.JAXScheme(DEVICE, reference_operations=(
            "firwin", "fir_zero_filter", "resample")) as context:
        actual = buffer()
        storage = actual.strain.data
        old_view = actual.strain[100:200]
        monkeypatch.setattr(strain_jax, "_condition_and_stitch_core",
                            lambda *a, **k: pytest.fail("reference was fused"))
        assert actual.advance(2)
        _same(actual.strain, original)
        assert actual.strain.data is not storage
        np.testing.assert_array_equal(old_view.numpy(), initial[100:200])
        assert to_jax(actual.strain).devices() == {context.jax_device}



@pytest.mark.parametrize("options", [
    dict(psd_duration=1, psd_stride=0),
    dict(psd_duration=0, psd_stride=0.5),
    dict(psd_duration=8, psd_stride=0.5),
    dict(psd_duration=1, psd_stride=0.5, psd_avg_method="unsupported"),
])
def test_autogate_rejects_invalid_welch_geometry_like_cpu(options):
    values = np.random.default_rng(4).normal(size=512).astype(np.float32)
    for context in (scheme.CPUScheme(), scheme.JAXScheme(DEVICE)):
        with context:
            source = TimeSeries(values, delta_t=1 / 64)
            with pytest.raises(ValueError):
                detect_loud_glitches(source, low_freq_cutoff=5,
                                     corrupt_time=0.125, **options)


def test_resident_autogate_honors_original_peak_clustering():
    with scheme.JAXScheme(DEVICE, reference_operations=("findchirp_cluster",)):
        assert strain_jax.autogate_strain_buffer_jax(SimpleNamespace()) is False



