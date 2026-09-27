import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pycbc import scheme
from pycbc.events import ranking
from pycbc.types import Array


_JAX_DEVICES = ["cpu"]
try:
    if jax.devices("gpu"):
        _JAX_DEVICES.append("gpu")
except RuntimeError:
    pass


def _expected(snr, reduced_x2):
    snr = np.asarray(snr, dtype=np.float64)
    reduced_x2 = np.asarray(reduced_x2, dtype=np.float64)
    return snr * np.where(
        reduced_x2 > 1.0,
        (0.5 * (1.0 + reduced_x2 ** 3.0)) ** (-1.0 / 6.0),
        1.0,
    )


@pytest.mark.parametrize("device", _JAX_DEVICES)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_jax_ranking_promotes_device_inputs_to_native_float64(device, dtype):
    snr = np.array([5.499999, 5.5, 5.500001, 8.25], dtype=dtype)
    reduced_x2 = np.array([0.999999, 1.0, 1.000001, 2.75], dtype=dtype)
    with scheme.CPUScheme():
        expected_new = ranking.newsnr(snr, reduced_x2)
        expected_eff = ranking.effsnr(snr, reduced_x2)

    with scheme.JAXScheme(device=device):
        for left, right in (
            (jnp.asarray(snr), jnp.asarray(reduced_x2)),
            (Array(snr), Array(reduced_x2)),
        ):
            actual_new = ranking.newsnr(left, right)
            actual_eff = ranking.effsnr(left, right)
            assert np.dtype(actual_new.dtype) == np.dtype(np.float64)
            assert np.dtype(actual_eff.dtype) == np.dtype(np.float64)
            if isinstance(left, Array):
                assert isinstance(actual_new, Array)
                assert isinstance(actual_eff, Array)
                new_raw = actual_new._data.array
                eff_raw = actual_eff._data.array
            else:
                assert hasattr(actual_new, "devices")
                assert hasattr(actual_eff, "devices")
                new_raw = actual_new
                eff_raw = actual_eff
            assert next(iter(new_raw.devices())).platform == (
                "cpu" if device == "cpu" else "gpu"
            )
            assert next(iter(eff_raw.devices())).platform == (
                "cpu" if device == "cpu" else "gpu"
            )
            np.testing.assert_allclose(actual_new, expected_new, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(actual_eff, expected_eff, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("device", _JAX_DEVICES)
def test_jax_ranking_handles_live_scalar_inputs(device):
    with scheme.CPUScheme():
        expected = ranking.newsnr(np.float32(5.500001), np.float32(1.000001))
    with scheme.JAXScheme(device=device):
        actual = ranking.newsnr(
            jnp.asarray(5.500001, dtype=jnp.float32),
            jnp.asarray(1.000001, dtype=jnp.float32),
        )
    assert actual.ndim == 0
    assert np.dtype(actual.dtype) == np.dtype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_native_ranking_still_promotes_float32_inputs():
    snr = np.array([5.499999, 5.500001], dtype=np.float32)
    reduced_x2 = np.array([1.000001, 2.75], dtype=np.float32)
    actual = ranking.newsnr(snr, reduced_x2)
    assert actual.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(actual, _expected(snr, reduced_x2), rtol=1e-15, atol=1e-15)
