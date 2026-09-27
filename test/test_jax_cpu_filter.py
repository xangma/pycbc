"""JAX CPU filtering stays on the JAX backend."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.filter.matchedfilter import _live_template_norms
from pycbc.filter.matchedfilter_jax import batch_peak_values, batch_sigmasq_jax
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


def test_batch_peaks_remain_jax_arrays():
    """Peak reduction does not synchronize through NumPy."""
    values = jnp.asarray(np.arange(24, dtype=np.float32).reshape(3, 8))
    with scheme.JAXScheme(device="cpu"):
        indices, peaks = batch_peak_values(values, 3, 8, slice(1, 7))
    assert isinstance(indices, jax.Array)
    assert isinstance(peaks, jax.Array)
    np.testing.assert_array_equal(np.asarray(indices), [5, 5, 5])
    np.testing.assert_array_equal(np.asarray(peaks), [6, 14, 22])


def test_live_template_norms_is_jax_reduction():
    """Live normalization matches the reference sum on a JAX device."""
    delta_f = 0.25
    class Template:
        def __init__(self, f_lower, data):
            self.delta_f = delta_f
            self.f_lower = f_lower
            self.data = jnp.asarray(data)

        def __array__(self, dtype=None):
            return np.asarray(self.data, dtype=dtype)

    templates = [
        Template(0.5, [1 + 2j, 2 + 1j, 3 + 0j, 4 + 1j]),
        Template(0.75, [2 + 0j, 1 + 1j, 3 + 2j, 1 + 0j]),
    ]
    psd = jnp.ones(4, dtype=jnp.float32)
    with scheme.JAXScheme(device="cpu"):
        result = _live_template_norms(templates, psd)
    assert isinstance(result, jax.Array)
    expected = jnp.asarray([jnp.sum(jnp.abs(t.data[int(t.f_lower / delta_f):]) ** 2)
                            * 4 * delta_f for t in templates])
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected),
                               rtol=2e-6, atol=2e-6)


def test_batch_sigmasq_stays_jax():
    """Batched normalization returns the device vector without host floats."""
    class TemplateBatch(list):
        pass

    templates = TemplateBatch()
    for _ in range(2):
        template = type("Template", (), {})()
        template.delta_f = 0.25
        template.f_lower = 0.5
        templates.append(template)
    templates._batch_tensor = jnp.asarray([
        [1 + 1j, 2 + 0j, 3 + 1j, 4 + 0j],
        [2 + 0j, 1 + 2j, 3 + 0j, 1 + 1j],
    ], dtype=jnp.complex64)
    psd = jnp.ones(4, dtype=jnp.float32)
    with scheme.JAXScheme(device="cpu"):
        result = batch_sigmasq_jax(templates, psd)
    assert isinstance(result, jax.Array)
    expected = jnp.sum(jnp.abs(templates._batch_tensor[:, 2:]) ** 2,
                       axis=1) * (4 * 0.25)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))
