"""Focused numerical compatibility checks for JAX PSD stages."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.psd.estimate_jax import (
    _interp_core,
    _median_bias_numpy_compat,
    _power_half_numpy_compat,
    _reciprocal_numpy_compat,
    interpolate_jax,
    welch_jax,
)
from pycbc.psd.estimate import median_bias
from pycbc.types.array_jax import _ensure_x64


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda")
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("num_segments", [1, 3, 63, 126, 1000])
@pytest.mark.parametrize("compiled", [False, True])
def test_median_bias_division_matches_float32_reference(
        device, num_segments, compiled):
    """Public and fused division preserve the float32 PSD scalar contract."""
    _ensure_x64()
    rng = np.random.default_rng(231)
    values = np.exp2(rng.uniform(-120, 120, 20000)).astype(np.float32)
    values = np.append(values, np.array([0, np.inf, np.nan], np.float32))
    # The compatibility contract rounds the divisor to PSD storage precision.
    # NumPy 2 promotes an np.float64 scalar (returned for >=1000 segments),
    # unlike the NumPy 1.26 reference environment. Specify both operand dtypes
    # so this test checks the frozen float32 arithmetic on either version.
    expected = values / np.float32(median_bias(num_segments))
    def divide(data):
        return _median_bias_numpy_compat(data, num_segments)
    with scheme.JAXScheme(device=device) as ctx:
        result = (jax.jit(divide) if compiled else divide)(jnp.asarray(values))
        assert result.dtype == jnp.float32
        assert result.devices() == {ctx.jax_device}
        np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("device", _devices())
def test_median_bias_float64_keeps_ordinary_division(device):
    _ensure_x64()
    values = np.exp2(np.linspace(-900, 900, 1001))
    with scheme.JAXScheme(device=device):
        source = jnp.asarray(values)
        actual = jax.jit(lambda x: _median_bias_numpy_compat(x, 126))(source)
        ordinary = jax.jit(lambda x: x / median_bias(126))(source)
        assert actual.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(ordinary))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("avg_method", ["median", "median-mean", "mean"])
def test_welch_averaging_with_exact_segment_ffts(device, avg_method):
    """Impulse segments isolate averaging from FFT backend roundoff."""
    amplitudes = np.random.default_rng(421).uniform(.1, 4, 126).astype(np.float32)
    samples = np.zeros(2 * len(amplitudes), dtype=np.float32)
    samples[::2] = amplitudes
    powers = amplitudes * amplitudes * np.float32(.5)
    if avg_method == "median":
        expected = np.median(powers) / np.float32(median_bias(len(powers)))
    elif avg_method == "median-mean":
        odd, even = powers[::2], powers[1::2]
        expected = (
            np.median(odd) / np.float32(median_bias(len(odd)))
            + np.median(even) / np.float32(median_bias(len(even)))
        ) / np.float32(2)
    else:
        expected = np.mean(powers)
    with scheme.JAXScheme(device=device) as ctx:
        result = welch_jax(jnp.asarray(samples), seg_len=2, seg_stride=2,
                           window=np.ones(2), avg_method=avg_method,
                           require_exact_data_fit=True)
        assert result.dtype == jnp.float32
        assert result.devices() == {ctx.jax_device}
        if avg_method == "mean":
            # Parallel sum order is backend dependent.
            np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-7)
        else:
            np.testing.assert_array_equal(np.asarray(result),
                                          np.full(2, expected, np.float32))


@pytest.mark.parametrize("device", _devices())
def test_wide_welch_matches_independent_double_fft(device):
    """Wide spectral arithmetic preserves the input's float32 windowing."""
    _ensure_x64()
    rng = np.random.default_rng(817)
    seg_len, stride, count = 256, 128, 15
    samples = rng.normal(size=(count - 1) * stride + seg_len).astype(np.float32)
    window = np.hanning(seg_len).astype(np.float32)
    segments = np.stack([samples[i * stride:i * stride + seg_len]
                         for i in range(count)])
    windowed = (segments * window).astype(np.float64)
    spectra = np.fft.rfft(windowed, axis=-1)
    powers = np.real(spectra * np.conj(spectra))
    powers[:, (0, -1)] *= 0.5
    expected = np.median(powers, axis=0) / median_bias(count)
    expected *= 2.0 / np.sum(window.astype(np.float64) ** 2)

    with scheme.JAXScheme(device=device) as ctx:
        actual = welch_jax(jnp.asarray(samples), seg_len=seg_len,
                           seg_stride=stride, window=window,
                           avg_method="median", num_segments=count,
                           require_exact_data_fit=True, wide_fft=True)
        assert actual.dtype == jnp.float64
        assert actual.devices() == {ctx.jax_device}
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-12)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_power_half_matches_numpy_scalar_power(
    device, dtype
):
    """Complex power follows NumPy ndarray's scalar square-root path."""
    if dtype is np.complex128 and not jax.config.x64_enabled:
        pytest.skip("complex128 requires JAX x64")
    values = np.exp(np.random.default_rng(77).uniform(-20, 20, 10000))
    values = np.append(values, 0).astype(dtype)
    # PyCBC Array.__pow__ uses the ndarray operator, which special-cases 0.5;
    # np.power is a different numerical path and is not the reference here.
    expected = values ** 0.5
    with scheme.JAXScheme(device=device) as ctx:
        result = jax.jit(_power_half_numpy_compat)(jnp.asarray(values))
        assert result.dtype == dtype
        assert result.devices() == {ctx.jax_device}
        actual = np.asarray(result)
    if dtype is np.complex64:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=2 * np.finfo(float).eps, atol=0
        )


@pytest.mark.parametrize("device", _devices())
def test_jax_interpolation_uses_numpy_value_precision(device):
    """Float32 storage follows NumPy's float64 interpolation intermediates."""
    with scheme.JAXScheme(device=device):
        values = np.array([0.1, 1.234567, 3.765432, 7.25], dtype=np.float32)
        old_df = 0.3
        new_df = 0.17
        new_n = 8
        actual = np.asarray(
            _interp_core(jnp.asarray(values), old_df, new_df, new_n)
        ).astype(np.float32)
        # Preserve the old coordinates: the regression was the subtraction
        # of adjacent float32 values, not the coordinate precision.
        old_result = np.asarray(jax.jit(lambda data: jnp.interp(
            jnp.arange(new_n, dtype=jnp.float64) * new_df,
            jnp.arange(len(values), dtype=jnp.float64) * old_df,
            data,
        ))(jnp.asarray(values))).astype(np.float32)

    old_freqs = np.arange(len(values), dtype=np.float64) * old_df
    new_freqs = np.arange(new_n, dtype=np.float64) * new_df
    expected = np.interp(new_freqs, old_freqs, values.astype(np.float64))
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    assert actual.dtype == np.float32
    assert np.any(actual != old_result)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_interpolate_preserves_psd_storage_dtype(device, dtype):
    values = np.linspace(1, 4, 9, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * np.linspace(4, 1, 9, dtype=values.real.dtype)
    with scheme.JAXScheme(device=device) as ctx:
        source = FrequencySeries(
            jnp.asarray(values),
            delta_f=0.25,
        )
        result = interpolate_jax(source, 0.125, length=17)
        assert isinstance(result.data.array, jax.Array)
        assert result.data.array.devices() == {ctx.jax_device}
        actual = np.asarray(result.numpy())

    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        actual,
        np.interp(
            np.arange(17, dtype=np.float64) * 0.125,
            np.arange(9, dtype=np.float64) * 0.25,
            values,
        ).astype(dtype),
    )


@pytest.mark.parametrize("device", _devices())
def test_float32_reciprocal_matches_numpy_across_normal_range(device):
    """The compiled helper follows NumPy's float32 reciprocal rounding."""
    _ensure_x64()
    rng = np.random.default_rng(3817)
    exponents = rng.uniform(-125, 125, 20000)
    mantissas = rng.uniform(1, 2, exponents.size)
    values = (mantissas * np.exp2(exponents)).astype(np.float32)
    values = np.concatenate((values, np.linspace(.999, 1.001, 10001,
                                                 dtype=np.float32)))
    values[::2] *= -1
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        expected = np.divide(np.ones_like(values), values)
    with scheme.JAXScheme(device=device) as ctx:
        result = jax.jit(_reciprocal_numpy_compat)(jnp.asarray(values))
        assert result.dtype == jnp.float32
        assert result.devices() == {ctx.jax_device}
        actual = np.asarray(result)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", _devices())
def test_float32_reciprocal_special_values_follow_numpy(device):
    """Special values retain ordinary semantics; subnormal exactness varies."""
    _ensure_x64()
    values = np.array(
        [0.0, -0.0, np.inf, -np.inf, np.nan,
         np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
         np.nextafter(np.float32(0), np.float32(1))],
        dtype=np.float32,
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        expected = np.divide(np.ones_like(values), values)
    with scheme.JAXScheme(device=device):
        actual = np.asarray(
            jax.jit(_reciprocal_numpy_compat)(jnp.asarray(values))
        )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    non_nan = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[non_nan], expected[non_nan])
    np.testing.assert_array_equal(
        np.signbit(actual[:2]), np.signbit(expected[:2])
    )
    np.testing.assert_array_equal(
        np.signbit(actual[2:4]), np.signbit(expected[2:4])
    )


@pytest.mark.parametrize("device", _devices())
def test_float32_reciprocal_finite_and_subnormal_edges(device):
    """Exercise finite reciprocal and subnormal-result boundaries."""
    _ensure_x64()
    largest_subnormal = np.nextafter(
        np.finfo(np.float32).tiny, np.float32(0)
    )
    values = np.array([
        largest_subnormal, -largest_subnormal,
        np.finfo(np.float32).max,
        -np.finfo(np.float32).max,
    ], dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        expected = np.divide(np.ones_like(values), values)
    with scheme.JAXScheme(device=device):
        source = jnp.asarray(values)
        actual = np.asarray(jax.jit(_reciprocal_numpy_compat)(source))
        ordinary = np.asarray(jax.jit(lambda x: 1.0 / x)(source))
    assert np.isfinite(expected[0])
    # XLA may flush subnormal inputs/results; compatibility must preserve its
    # ordinary backend behavior rather than manufacture a new result.
    np.testing.assert_array_equal(actual, ordinary)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert 0 < expected[2] < np.finfo(np.float32).tiny


@pytest.mark.parametrize("device", _devices())
def test_float64_reciprocal_remains_ordinary_jax_division(device):
    """The compatibility helper does not alter float64 arithmetic."""
    _ensure_x64()
    values = np.array([-.25, 1.0, 3.5, 1e-200, 1e200], dtype=np.float64)
    with scheme.JAXScheme(device=device):
        actual = np.asarray(
            jax.jit(_reciprocal_numpy_compat)(jnp.asarray(values))
        )
        ordinary = np.asarray(
            jax.jit(lambda x: 1.0 / x)(jnp.asarray(values))
        )
    np.testing.assert_array_equal(actual, ordinary)
    assert actual.dtype == np.float64
