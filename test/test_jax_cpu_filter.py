"""JAX filtering storage and explicit host metadata boundaries."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.fft.jaxfft import fft, ifft, IFFT
from pycbc.types import zeros
from pycbc.types.array_jax import JAXArrayData


def test_cpu_jax_ifft_writes_jax_storage():
    """The CPU device uses jnp.fft through the class-based JAX plan."""
    rng = np.random.default_rng(120)
    values = (rng.normal(size=16) + 1j * rng.normal(size=16)).astype(np.complex64)
    with scheme.JAXScheme(device="cpu"):
        inp = zeros(16, dtype=np.complex64)
        out = zeros(16, dtype=np.complex64)
        assert isinstance(inp.data, JAXArrayData)
        inp.data.set_array(jnp.asarray(values))
        plan = IFFT(inp, out, nbatch=1, size=16)
        plan.execute()
        np.testing.assert_allclose(np.asarray(out), np.fft.ifft(values) * 16,
                                   rtol=2e-5, atol=2e-5)
        assert isinstance(out.data, JAXArrayData)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_functional_jax_fft_hooks_keep_device_storage(device):
    """Functional FFT hooks must not synchronize JAX destinations."""
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA JAX device unavailable")
    values = jnp.arange(8, dtype=jnp.complex64)
    with scheme.JAXScheme(device=device):
        inp = zeros(8, dtype=np.complex64)
        out = zeros(8, dtype=np.complex64)
        recovered = zeros(8, dtype=np.complex64)
        inp.data.set_array(values)

        fft(inp, out, None, "complex", "complex")
        ifft(out, recovered, None, "complex", "complex")
        assert isinstance(out.data, JAXArrayData)
        assert jnp.allclose(out.data.array, jnp.fft.fft(values))
        assert isinstance(recovered.data, JAXArrayData)
        assert jnp.allclose(recovered.data.array, values * 8)
