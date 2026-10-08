"""JAX residency checks for conditioning and resampling."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc.filter.resample import (lfilter, lowpass, highpass,
                                   highpass_fir, fir_zero_filter,
                                   resample_to_delta_t)
from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import TimeSeries

_DEVICES = ["cpu"]
if any(d.platform in ("cuda", "gpu") for d in jax.devices()):
    _DEVICES.append("cuda")


def _ts(n=256, dtype=np.float64):
    return TimeSeries(np.sin(np.arange(n) * .07).astype(dtype), delta_t=.01)


def _forbid_jax_downloads(monkeypatch):
    from pycbc.types.array_jax import JAXArrayData

    array, asarray = np.array, np.asarray

    def check(value):
        if isinstance(value, (jax.Array, JAXArrayData)):
            raise AssertionError("conditioning downloaded JAX samples")

    def checked_array(value, *args, **kwargs):
        check(value)
        return array(value, *args, **kwargs)

    def checked_asarray(value, *args, **kwargs):
        check(value)
        return asarray(value, *args, **kwargs)

    def reject_download(*args, **kwargs):
        raise AssertionError("conditioning downloaded JAX samples")

    # CPU JAX arrays expose a buffer interface which NumPy can use without
    # calling __array__; intercept the NumPy boundary as well as device_get.
    monkeypatch.setattr(np, "array", checked_array)
    monkeypatch.setattr(np, "asarray", checked_asarray)
    monkeypatch.setattr(jax, "device_get", reject_download)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_normalization_uses_compensated_sum(device):
    from pycbc.filter.resample_jax import _compensated_pairwise_sum, firwin
    from scipy.signal import firwin as scipy_firwin

    with JAXScheme(device):
        # A naive adjacent pair tree loses both unit terms.
        terms = jax.numpy.asarray([1e16, 1.0, -1e16, 1.0], dtype=jax.numpy.float64)
        assert float(_compensated_pairwise_sum(terms)) == 2.0
        actual = np.asarray(firwin(3491, 0.00634765625,
                                  window=('kaiser', 21.081260000000004),
                                  pass_zero=False))

    expected = scipy_firwin(3491, 0.00634765625,
                            window=('kaiser', 21.081260000000004),
                            pass_zero=False)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_is_device_backed_and_matches_reference(device):
    # Compute the reference outside JAXScheme so this compares distinct
    # implementations, including the native circular FFT convention.
    reference = lfilter(np.array([.2, .3, .5]), _ts()).numpy()
    with JAXScheme(device):
        # The production CPU path intentionally switches to its circular FFT
        # implementation for records >= 128 samples.
        ts = _ts()
        got = lfilter(np.array([.2, .3, .5]), ts)
        assert jax.device_get(got._data.array).shape == got.shape
        np.testing.assert_allclose(got.numpy(), reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [255, 256, 257, 258])
@pytest.mark.parametrize("taps", [
    np.array([0.125, -0.3, 0.7], dtype=np.float64),
    np.array([0.2, -0.1, 0.4, 0.05], dtype=np.float64),
])
@pytest.mark.parametrize("device", _DEVICES)
def test_jax_circular_fir_preserves_native_correlation_alignment(dtype, size,
                                                                  taps, device):
    """Exercise native reverse/roll correlation with nonsymmetric taps."""
    data = np.random.default_rng(1234).normal(size=size).astype(dtype)
    source = TimeSeries(data, delta_t=.01)
    reference = lfilter(taps, source).numpy()
    with JAXScheme(device):
        result = lfilter(taps, TimeSeries(data, delta_t=.01)).numpy()
    # FFT implementations may round differently; this checks the operation
    # ordering and alignment without requiring cross-backend bit identity.
    tolerance = 3e-6 if dtype == np.float32 else 3e-12
    np.testing.assert_allclose(result, reference, rtol=tolerance, atol=tolerance)
    assert result.dtype == dtype


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_large_record_matches_original_recursive_boundaries(device):
    samples = np.zeros(2 ** 18, dtype=np.float64)
    samples[-1] = 1
    samples[len(samples) // 2 - 1] = 2
    coefficients = np.array([.2, .3, .5])
    with CPUScheme():
        expected = lfilter(coefficients, TimeSeries(samples, delta_t=.01)).numpy()
    with JAXScheme(device):
        result = lfilter(coefficients, TimeSeries(samples, delta_t=.01)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[:2], [.6, 1.], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_circular_fir_truncates_oversized_coefficients_like_original(device):
    samples = np.linspace(-1, 1, 256)
    coefficients = np.linspace(.1, .5, 259)
    with CPUScheme():
        expected = lfilter(coefficients, TimeSeries(samples, delta_t=.01)).numpy()
    with JAXScheme(device):
        result = lfilter(coefficients, TimeSeries(samples, delta_t=.01)).numpy()
    np.testing.assert_allclose(result, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("frequency", [-1, 0, 50, 51])
def test_jax_fir_rejects_cutoffs_outside_original_range(device, frequency):
    with CPUScheme(), pytest.raises(ValueError):
        highpass_fir(_ts(), frequency, 3)
    with JAXScheme(device), pytest.raises(ValueError):
        highpass_fir(_ts(), frequency, 3)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_short_matches_scipy_reference(device):
    from scipy.signal import lfilter as scipy_lfilter
    with JAXScheme(device):
        ts = _ts(64)
        got = lfilter(np.array([.2, .3, .5]), ts)
        ref = scipy_lfilter([.2, .3, .5], 1, ts.numpy())
        assert jax.device_get(got._data.array).shape == got.shape
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_short_float32_matches_scipy_dtype(device):
    from scipy.signal import lfilter as scipy_lfilter
    samples = np.linspace(-1, 1, 64, dtype=np.float32)
    coefficients = np.array([.2, .3, .5], dtype=np.float32)
    reference = scipy_lfilter(coefficients, 1.0, samples)
    with JAXScheme(device):
        got = lfilter(coefficients, TimeSeries(samples, delta_t=.01))
        assert jax.device_get(got._data.array).shape == got.shape
        assert got.dtype == reference.dtype
        np.testing.assert_array_equal(got.numpy(), reference)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_short_complex64_matches_scipy_dtype(device):
    from scipy.signal import lfilter as scipy_lfilter
    samples = (np.linspace(-1, 1, 64) +
               1j * np.linspace(1, -1, 64)).astype(np.complex64)
    coefficients = np.array([.2, .3, .5], dtype=np.float32)
    reference = scipy_lfilter(coefficients, 1.0, samples)
    with JAXScheme(device):
        got = lfilter(coefficients, TimeSeries(samples, delta_t=.01))
        assert jax.device_get(got._data.array).shape == got.shape
        assert got.dtype == reference.dtype
        np.testing.assert_array_equal(got.numpy(), reference)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("numtaps", [3, 129])
def test_jax_fir_zero_short_preserves_length_and_filter_dtype(
        device, dtype, numtaps):
    samples = np.linspace(-1, 1, 64)
    if np.issubdtype(dtype, np.complexfloating):
        samples = samples + 1j * samples[::-1]
    samples = samples.astype(dtype)
    coefficients = np.hanning(numtaps)
    with CPUScheme():
        source = TimeSeries(samples, delta_t=.01, epoch=123)
        reference = fir_zero_filter(coefficients, source)
        expected = reference.numpy().copy()
    with JAXScheme(device) as ctx:
        source = TimeSeries(samples, delta_t=.01, epoch=123)
        result = fir_zero_filter(coefficients, source)
        assert result.data.array.devices() == {ctx.jax_device}
        assert len(result) == len(reference) == len(source)
        assert result.dtype == reference.dtype
        assert result.start_time == source.start_time
        assert result.delta_t == source.delta_t
        np.testing.assert_allclose(result.numpy(), expected,
                                   rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_lowpass_does_not_use_lal_and_returns_device_data(monkeypatch, device):
    from pycbc.types import timeseries
    monkeypatch.setattr(timeseries.TimeSeries, "lal",
                        lambda self: (_ for _ in ()).throw(AssertionError("LAL")))
    with JAXScheme(device):
        got = lowpass(_ts(), 10)
        assert jax.device_get(got._data.array).shape == got.shape


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_ldas_resample_is_device_backed(device):
    with JAXScheme(device):
        got = resample_to_delta_t(_ts(512), .02, method="ldas")
        assert len(got) == 256
        assert jax.device_get(got._data.array).shape == got.shape
        assert got.corrupted_samples == 10


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("operation", [
    "fir", "short_fir", "zero_fir", "short_zero_fir", "ldas",
    "lowpass", "highpass", "closed_form_highpass",
])
def test_jax_conditioning_outputs_do_not_download_samples(
        monkeypatch, device, dtype, operation):
    size = 64 if operation.startswith("short_") else 257
    samples = np.random.default_rng(120).normal(size=size).astype(dtype)
    coefficients = np.array([.2, .3, .5], dtype=dtype)
    mode = "closed-form" if operation == "closed_form_highpass" else "lal-coefficients"
    with JAXScheme(device, highpass_mode=mode) as ctx:
        source = TimeSeries(samples, delta_t=.01, epoch=123)

        # A resident output alone cannot detect an intermediate NumPy round
        # trip. Reject the actual array conversion while the public API runs.
        with monkeypatch.context() as guard:
            _forbid_jax_downloads(guard)
            if operation in ("fir", "short_fir"):
                result = lfilter(coefficients, source)
            elif operation in ("zero_fir", "short_zero_fir"):
                result = fir_zero_filter(coefficients, source)
            elif operation == "ldas":
                result = resample_to_delta_t(source, .02, method="ldas")
            else:
                result = (lowpass if operation == "lowpass" else highpass)(
                    source, 10, filter_order=3)

            expected_dtype = (np.float64 if operation.startswith("short_")
                              else dtype)
            assert result.dtype == expected_dtype
            assert result.start_time == source.start_time
            assert result.delta_t == (.02 if operation == "ldas" else .01)
            expected_size = (size + 1) // 2 if operation == "ldas" else size
            assert len(result) == expected_size
            assert result.data.array.devices() == {ctx.jax_device}
            assert result.data is not source.data
            # Copying a wrapper must preserve independent mutable Series
            # containers even when immutable JAX buffers can be shared.
            result[0] = 7

        assert result.numpy()[0] == 7
        np.testing.assert_array_equal(source.numpy(), samples)


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_long_fir_chain_does_not_download_samples(monkeypatch, device):
    # Exercise long FIR design and a non-power-of-two FFT before decimation.
    samples = np.random.default_rng(121).normal(size=36296)
    with JAXScheme(device) as ctx:
        source = TimeSeries(samples, delta_t=1 / 4096, epoch=123)

        with monkeypatch.context() as guard:
            _forbid_jax_downloads(guard)
            filtered = highpass_fir(source, 13, 1745,
                                    beta=21.081260000000004)
            filtered = (filtered * 1e20).astype(np.float32)
            result = resample_to_delta_t(filtered, 1 / 2048, method="ldas")
            assert result.data.array.devices() == {ctx.jax_device}
            assert result.dtype == np.float32
            assert len(result) == 18148
            assert result.delta_t == 1 / 2048
            assert result.start_time == source.start_time
            assert result.corrupted_samples == 10
        np.testing.assert_array_equal(source.numpy(), samples)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", [1, 2, 3, 8])
@pytest.mark.parametrize("kind", ["low", "high"])
@pytest.mark.parametrize("device", _DEVICES)
def test_jax_butterworth_matches_lal(dtype, order, kind, device):
    rng = np.random.default_rng(1729)
    data = rng.normal(size=4096).astype(dtype)
    data[2048] = 1
    reference = (lowpass if kind == "low" else highpass)(
        TimeSeries(data, delta_t=1 / 2048), 100, filter_order=order)
    with JAXScheme(device):
        result = (lowpass if kind == "low" else highpass)(
            TimeSeries(data, delta_t=1 / 2048), 100, filter_order=order)
    err = np.max(np.abs(result.numpy() - reference.numpy()))
    scale = np.max(np.abs(reference.numpy()))
    # Keep this test scientific: the tolerance is dtype appropriate and the
    # full record, including the impulse, is compared.
    np.testing.assert_allclose(result.numpy(), reference.numpy(),
                               rtol=2e-5 if dtype == np.float32 else 2e-10,
                               atol=2e-6 if dtype == np.float32 else 2e-11)
    assert err / max(scale, 1e-30) < (2e-5 if dtype == np.float32 else 2e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", _DEVICES)
def test_jax_fir_and_ldas_match_cpu(dtype, device):
    data = np.random.default_rng(44).normal(size=1024).astype(dtype)
    source = TimeSeries(data, delta_t=1 / 2048)
    fir_ref = highpass_fir(source, 100, 31)
    ldas_ref = resample_to_delta_t(source, 1 / 1024, method="ldas")
    with JAXScheme(device):
        fir = highpass_fir(TimeSeries(data, delta_t=1 / 2048), 100, 31)
        ldas = resample_to_delta_t(TimeSeries(data, delta_t=1 / 2048),
                                   1 / 1024, method="ldas")
    # FFT libraries round differently at cancellation zeros. Bound both the
    # energy error and worst sample error relative to the full signal scale.
    tolerance = 2e-6 if dtype == np.float32 else 3e-12
    for result, reference in ((fir, fir_ref), (ldas, ldas_ref)):
        expected = reference.numpy()
        error = result.numpy() - expected
        assert np.linalg.norm(error) / np.linalg.norm(expected) < tolerance
        assert np.max(np.abs(error)) < tolerance * np.max(np.abs(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", _DEVICES)
def test_closed_form_butterworth_long_record_matches_lal(dtype, device):
    # A long record exercises prefix composition beyond the transient response.
    data = np.random.default_rng(55).normal(size=65536).astype(dtype)
    data[32768] += 100
    reference = highpass(TimeSeries(data, delta_t=1 / 2048), 25).numpy()
    with JAXScheme(device, highpass_mode="closed-form"):
        result = highpass(TimeSeries(data, delta_t=1 / 2048), 25).numpy()
    error = result - reference
    tolerance = 2e-6 if dtype == np.float32 else 2e-10
    assert np.linalg.norm(error) / np.linalg.norm(reference) < tolerance
    assert np.max(np.abs(error)) < tolerance * np.max(np.abs(reference))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", [1, 2, 3, 8])
@pytest.mark.parametrize("device", _DEVICES)
def test_lal_coefficients_highpass_mode_matches_reference(dtype, order, device):
    # An odd-length record exercises the prefix scan tail.
    data = np.random.default_rng(818).normal(size=513).astype(dtype)
    data[256] += 10
    reference = highpass(TimeSeries(data, delta_t=1 / 2048), 25,
                         filter_order=order).numpy()
    with JAXScheme(device, highpass_mode="lal-coefficients"):
        result = highpass(TimeSeries(data, delta_t=1 / 2048), 25,
                          filter_order=order)
    assert isinstance(result._data.array, jax.Array)
    assert result.dtype == dtype
    tolerance = 3e-6 if dtype == np.float32 else 3e-10
    np.testing.assert_allclose(result.numpy(), reference, rtol=tolerance,
                               atol=tolerance)


@pytest.mark.parametrize("device", _DEVICES)
def test_default_highpass_matches_lal_coefficients(device):
    data = np.random.default_rng(819).normal(size=513).astype(np.float64)
    series = TimeSeries(data, delta_t=1 / 2048)
    reference = highpass(series, 25).numpy()
    with JAXScheme(device):
        default = highpass(series, 25).numpy()
    with JAXScheme(device, highpass_mode="lal-coefficients"):
        explicit = highpass(series, 25).numpy()
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_allclose(default, reference, rtol=3e-10, atol=3e-10)
