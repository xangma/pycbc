"""Regression tests for JAX chi-square power/PSD rounding."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc.vetoes import chisq_jax
from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series


@pytest.fixture(autouse=True)
def _x64(monkeypatch):
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _oracle(magnitude, psd):
    return np.divide(np.asarray(magnitude), np.asarray(psd))


def _devices():
    devices = []
    for backend in ("cpu", "gpu"):
        try:
            devices.extend(jax.devices(backend))
        except RuntimeError:
            pass
    return list({str(device): device for device in devices}.values())


@pytest.mark.parametrize("shape", [(17,), (3, 17), (10000,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_power_divide_matches_numpy_oracle_on_each_device(
        shape, dtype):
    rng = np.random.default_rng(3817)
    magnitude = rng.uniform(0.01, 7.0, size=shape).astype(dtype)
    psd = rng.uniform(0.2, 4.0, size=(shape[-1],)).astype(dtype)
    expected = _oracle(magnitude, psd)
    for device in _devices():
        fn = jax.jit(chisq_jax._weighted_power_divide)
        got = fn(
            jax.device_put(magnitude, device), jax.device_put(psd, device))
        if dtype == np.float32:
            np.testing.assert_array_equal(np.asarray(got), expected)
        else:
            # Double precision remains native; device math may differ from
            # NumPy by one last-place in a broadcasted quotient.
            np.testing.assert_allclose(np.asarray(got), expected,
                                       rtol=1e-14, atol=0)
        assert got.dtype == jnp.dtype(dtype)
        actual_device = (got.device if not callable(got.device)
                         else got.device())
        assert actual_device.platform == device.platform


def test_float32_helper_preserves_scalar_and_batch_bin_parity():
    rng = np.random.default_rng(901)
    h = (rng.normal(size=33) + 1j * rng.normal(size=33)).astype(np.complex64)
    psd = rng.uniform(0.4, 2.0, size=33).astype(np.float32)
    scalar = chisq_jax._power_chisq_bins_numpy(
        jnp.asarray(h), jnp.asarray(psd), 2, 31, 8, 0.25)
    batch = chisq_jax._power_chisq_bins_numpy(
        jnp.asarray(np.stack([h, h * np.complex64(1.7)])),
        jnp.asarray(psd), 2, 31, 8, 0.25)
    power = np.divide(h[2:31].real ** 2 + h[2:31].imag ** 2, psd[2:31])
    expected = power_chisq_bins_from_sigmasq_series(
        np.cumsum(power) * (4.0 * 0.25), 8, 0, len(power)) + 2
    np.testing.assert_array_equal(np.asarray(scalar), expected)
    np.testing.assert_array_equal(np.asarray(batch[0]), expected)


def test_float32_normal_exponent_range_matches_np_divide_under_jit():
    # Keep both operands and quotients in the normal float32 range while
    # exercising many exponent scales relevant to rescaled detector data.
    exponent = np.linspace(-100.0, 100.0, 10000)
    magnitude = np.power(2.0, exponent).astype(np.float32)
    psd = np.power(2.0, exponent + 20.0 * np.sin(exponent)).astype(np.float32)
    expected = np.divide(magnitude, psd)
    compiled = jax.jit(chisq_jax._weighted_power_divide)
    for device in _devices():
        got = compiled(jax.device_put(magnitude, device),
                       jax.device_put(psd, device))
        np.testing.assert_array_equal(np.asarray(got), expected)
        actual_device = (got.device if not callable(got.device)
                         else got.device())
        assert actual_device.platform == device.platform
