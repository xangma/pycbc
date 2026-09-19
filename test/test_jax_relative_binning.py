"""Focused tests for the JAX relative-binning kernels and dispatch."""

import numpy
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pycbc.inference.models import relbin_jax  # noqa: E402


@pytest.fixture
def summary_data():
    key = jax.random.PRNGKey(8182)
    frequency_count = 129
    frequencies = jnp.linspace(
        20.0, 500.0, frequency_count, dtype=jnp.float64
    )
    edge_indices = jnp.arange(17, dtype=jnp.int64) * 8
    bins = jnp.stack((edge_indices[:-1], edge_indices[1:]), axis=1)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    hp = jax.random.normal(k1, (frequency_count,)) + 1j * jax.random.normal(k2, (frequency_count,))
    hc = jax.random.normal(k3, (frequency_count,)) + 1j * jax.random.normal(k4, (frequency_count,))
    k5, k6 = jax.random.split(k4, 2)
    reference = jax.random.normal(k5, (frequency_count,)) + 1j * jax.random.normal(k6, (frequency_count,))
    psd = jax.random.uniform(k5, (frequency_count,), dtype=jnp.float64) + 0.1

    a0, a1 = relbin_jax.summary_product(
        hp, reference, psd, frequencies, bins, 0.25
    )
    b0, b1 = relbin_jax.summary_product(
        reference, reference, psd, frequencies, bins, 0.25
    )
    return {
        "frequencies": frequencies[edge_indices],
        "hp": hp[edge_indices],
        "hc": hc[edge_indices],
        "reference": reference[edge_indices],
        "a0": a0,
        "a1": a1,
        "b0": jnp.real(b0),
        "b1": jnp.real(b1),
    }


def test_batched_likelihood_matches_scalar_evaluation(summary_data):
    sample_count = 12
    fplus = jnp.linspace(0.1, 1.0, sample_count, dtype=jnp.float64)
    fcross = jnp.linspace(-0.5, 0.5, sample_count, dtype=jnp.float64)
    delays = jnp.linspace(-0.01, 0.01, sample_count, dtype=jnp.float64)
    args = (
        summary_data["frequencies"],
        summary_data["hp"],
        summary_data["hc"],
        summary_data["reference"],
        summary_data["a0"],
        summary_data["a1"],
        summary_data["b0"],
        summary_data["b1"],
    )

    actual = relbin_jax.likelihood_parts(
        args[0], fplus, fcross, delays, *args[1:]
    )
    expected_0 = []
    expected_1 = []
    for index in range(sample_count):
        out0, out1 = relbin_jax.likelihood_parts(
            args[0], fplus[index], fcross[index], delays[index],
            *args[1:]
        )
        expected_0.append(out0)
        expected_1.append(out1)

    assert actual[0].shape == (sample_count,)
    assert actual[1].shape == (sample_count,)
    numpy.testing.assert_allclose(numpy.asarray(actual[0]), numpy.asarray(expected_0), rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(numpy.asarray(actual[1]), numpy.asarray(expected_1), rtol=1e-12, atol=1e-12)


def test_public_detector_fallback_uses_arrays_and_keeps_gradients():
    calls = []

    class PublicDetector:
        @staticmethod
        def antenna_pattern(ra, dec, polarization, arrival_time):
            calls.append("antenna")
            return jnp.cos(ra) + 0.0 * arrival_time, jnp.sin(dec) + 0.0 * arrival_time

        @staticmethod
        def time_delay_from_earth_center(ra, dec, arrival_time):
            calls.append("delay")
            return 0.01 * (ra - dec) + 0.0 * arrival_time

    detector = PublicDetector()
    like = jnp.ones(8, dtype=jnp.complex128)

    def eval_resp(ra_val, dec_val):
        return relbin_jax.detector_response(
            detector, ra_val, dec_val, 1126259462.0, like
        )

    ra = 1.1
    dec = -0.4
    fplus, fcross, delay = eval_resp(ra, dec)
    assert calls == ["antenna", "delay"]
    assert float(fplus) == pytest.approx(float(numpy.cos(ra)))
    assert float(fcross) == pytest.approx(float(numpy.sin(dec)))
    assert float(delay) == pytest.approx(0.01 * (ra - dec))

    def sum_resp(params):
        fp, fc, dl = eval_resp(params[0], params[1])
        return fp + fc + dl

    grad = jax.grad(sum_resp)(jnp.array([ra, dec]))
    assert grad is not None
    numpy.testing.assert_allclose(float(grad[0]), -numpy.sin(ra) + 0.01, rtol=1e-5)
    numpy.testing.assert_allclose(float(grad[1]), numpy.cos(dec) - 0.01, rtol=1e-5)


def test_earth_rotation_response_matches_public_detector_geometry():
    from pycbc.detector import Detector

    reference_time = 1126259462.0
    detector = Detector("H1", reference_time=reference_time)
    times = reference_time + numpy.asarray([0.0, 0.25, 1.5, 8.0])
    ra = 1.37
    dec = -1.26
    like = jnp.ones(8, dtype=jnp.complex128)

    actual = relbin_jax.detector_response(
        detector, ra, dec, times, like
    )
    expected = tuple(numpy.asarray(values) for values in zip(*(
        (
            *detector.antenna_pattern(ra, dec, 0.0, time),
            detector.time_delay_from_earth_center(ra, dec, time),
        )
        for time in times
    )))

    for result, reference in zip(actual, expected):
        numpy.testing.assert_allclose(
            numpy.asarray(result), reference, rtol=2e-12, atol=2e-12
        )
    assert not numpy.array_equal(numpy.asarray(actual[0][0]), numpy.asarray(actual[0][-1]))


@pytest.mark.parametrize("override_location", ["subclass", "instance"])
def test_detector_response_preserves_overrides_and_public_backend_values(
        monkeypatch, override_location):
    from types import MethodType
    from pycbc.detector import Detector

    class BackendValue:
        backend = "jax"

        def __init__(self, arr):
            self.backend_array = arr

    calls = []

    def antenna_pattern(self, ra, dec, polarization, times):
        calls.append("antenna")
        return jnp.cos(ra) + polarization, jnp.sin(dec) - polarization

    def time_delay(self, ra, dec, times):
        calls.append("delay")
        return 0.01 * (ra - dec)

    class CustomDetector(Detector):
        pass

    detector = CustomDetector("H1")
    for name, method in (
        ("antenna_pattern", antenna_pattern),
        ("time_delay_from_earth_center", time_delay),
    ):
        if override_location == "subclass":
            monkeypatch.setattr(CustomDetector, name, method)
        else:
            monkeypatch.setattr(detector, name, MethodType(method, detector))

    ra = 1.1
    dec = -0.4
    like = jnp.ones(8, dtype=jnp.complex128)
    result = relbin_jax.detector_response(
        detector,
        BackendValue(jnp.array(ra, dtype=jnp.float64)),
        BackendValue(jnp.array(dec, dtype=jnp.float64)),
        BackendValue(jnp.array(1126259462.0, dtype=jnp.float64)),
        BackendValue(like),
    )
    assert calls == ["antenna", "delay"]
    expected = (numpy.cos(ra), numpy.sin(dec), 0.01 * (ra - dec))
    for actual, reference in zip(result, expected):
        numpy.testing.assert_allclose(numpy.asarray(actual), reference, rtol=1e-12)


def test_summary_product_keeps_batch_dimensions():
    frequency_count = 33
    batch_count = 5
    frequencies = jnp.arange(frequency_count, dtype=jnp.float64) * 0.25
    edges = jnp.arange(0, frequency_count, 4, dtype=jnp.int64)
    bins = jnp.stack((edges[:-1], edges[1:]), axis=1)
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    first = (
        jax.random.normal(k1, (batch_count, frequency_count))
        + 1j * jax.random.normal(k2, (batch_count, frequency_count))
    )
    second = (
        jax.random.normal(k3, (frequency_count,))
        + 1j * jax.random.normal(k4, (frequency_count,))
    )
    psd = jnp.ones(frequency_count, dtype=jnp.float64)

    a0, a1 = relbin_jax.summary_product(
        first, second, psd, frequencies, bins, 0.25
    )
    assert a0.shape == (batch_count, len(bins))
    assert a1.shape == (batch_count, len(bins))

    def loss(f_in):
        out0, out1 = relbin_jax.summary_product(
            f_in, second, psd, frequencies, bins, 0.25
        )
        return jnp.sum(jnp.real(out0)) + jnp.sum(jnp.imag(out1))

    grad = jax.grad(loss)(first)
    assert grad is not None
    assert grad.shape == first.shape


def test_time_series_wrapper_preserves_storage_and_metadata():
    from pycbc import scheme
    from pycbc.inference.models.relbin import _time_series_from_values
    from pycbc.types.backend import backend_array

    values = jnp.ones(8, dtype=jnp.float64)
    context = scheme.JAXScheme()
    try:
        with context:
            series = _time_series_from_values(values, 0.25, 123.5)
            arr = backend_array(series, "jax")
            assert arr is not None
            assert series.delta_t == 0.25
            assert float(series.start_time) == 123.5
    finally:
        del context
        scheme.Scheme._single = None


def test_likelihood_parts_jit_compile():
    frequency_count = 64
    freqs = jnp.linspace(20.0, 1024.0, frequency_count, dtype=jnp.float64)
    fp = jnp.array(0.7, dtype=jnp.float64)
    fc = jnp.array(0.3, dtype=jnp.float64)
    dtc = jnp.array(0.002, dtype=jnp.float64)
    key = jax.random.PRNGKey(123)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    hp = jax.random.normal(k1, (frequency_count,)) + 1j * jax.random.normal(k2, (frequency_count,))
    hc = jax.random.normal(k3, (frequency_count,)) + 1j * jax.random.normal(k4, (frequency_count,))
    h00 = jax.random.normal(k5, (frequency_count,)) + 1j * jax.random.normal(k6, (frequency_count,))
    a0 = jax.random.normal(k1, (frequency_count - 1,)) + 1j * jax.random.normal(k2, (frequency_count - 1,))
    a1 = jax.random.normal(k3, (frequency_count - 1,)) + 1j * jax.random.normal(k4, (frequency_count - 1,))
    b0 = jax.random.normal(k5, (frequency_count - 1,), dtype=jnp.float64)
    b1 = jax.random.normal(k6, (frequency_count - 1,), dtype=jnp.float64)

    expected_filt, expected_norm = relbin_jax.likelihood_parts(
        freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1
    )

    compiled_fn = jax.jit(relbin_jax.likelihood_parts)
    compiled_filt, compiled_norm = compiled_fn(
        freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1
    )

    numpy.testing.assert_allclose(numpy.asarray(compiled_filt), numpy.asarray(expected_filt), rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(numpy.asarray(compiled_norm), numpy.asarray(expected_norm), rtol=1e-12, atol=1e-12)


def test_likelihood_parts_det_multi_jit_compile():
    frequency_count = 64
    freqs = jnp.linspace(20.0, 1024.0, frequency_count, dtype=jnp.float64)
    dtc = jnp.array(0.002, dtype=jnp.float64)
    dtc2 = jnp.array(-0.001, dtype=jnp.float64)
    key = jax.random.PRNGKey(456)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    channel = jax.random.normal(k1, (frequency_count,)) + 1j * jax.random.normal(k2, (frequency_count,))
    channel2 = jax.random.normal(k3, (frequency_count,)) + 1j * jax.random.normal(k4, (frequency_count,))
    h00 = jax.random.normal(k1, (frequency_count,)) + 1j * jax.random.normal(k2, (frequency_count,))
    h002 = jax.random.normal(k3, (frequency_count,)) + 1j * jax.random.normal(k4, (frequency_count,))
    a0 = jax.random.normal(k1, (frequency_count - 1,)) + 1j * jax.random.normal(k2, (frequency_count - 1,))
    a1 = jax.random.normal(k3, (frequency_count - 1,)) + 1j * jax.random.normal(k4, (frequency_count - 1,))

    expected = relbin_jax.likelihood_parts_det_multi(
        freqs, dtc, channel, h00, dtc2, channel2, h002, a0, a1
    )

    compiled_fn = jax.jit(relbin_jax.likelihood_parts_det_multi)
    compiled = compiled_fn(
        freqs, dtc, channel, h00, dtc2, channel2, h002, a0, a1
    )

    numpy.testing.assert_allclose(numpy.asarray(compiled), numpy.asarray(expected), rtol=1e-12, atol=1e-12)
