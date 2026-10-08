"""JAX filtering storage and explicit host metadata boundaries."""

import types

import numpy as np
import pytest

try:
    import jax
except ImportError:
    pytest.skip("JAX is unavailable", allow_module_level=True)
import jax.numpy as jnp

from pycbc import scheme
from pycbc.filter.matchedfilter_jax import (
    batch_peak_values,
    batch_sigmasq_jax,
    live_template_norms_jax,
)
from pycbc.fft.jaxfft import fft, ifft, IFFT
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform.bank import sigma_cached
from pycbc.types.array_jax import JAXArrayData


def test_cpu_jax_ifft_writes_jax_storage():
    """The CPU device uses jnp.fft through the class-based JAX plan."""
    rng = np.random.default_rng(120)
    values = (rng.normal(size=16) + 1j * rng.normal(size=16)).astype(
        np.complex64
    )
    with scheme.JAXScheme(device="cpu"):
        inp = zeros(16, dtype=np.complex64)
        out = zeros(16, dtype=np.complex64)
        assert isinstance(inp.data, JAXArrayData)
        inp.data.set_array(jnp.asarray(values))
        plan = IFFT(inp, out, nbatch=1, size=16)
        plan.execute()
        np.testing.assert_allclose(
            np.asarray(out), np.fft.ifft(values) * 16, rtol=2e-5, atol=2e-5
        )
        assert isinstance(out.data, JAXArrayData)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_functional_jax_fft_hooks_keep_device_storage(device):
    """Functional FFT hooks must not synchronize JAX destinations."""
    if device == "cuda" and not any(
        d.platform == "gpu" for d in jax.devices()
    ):
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
        result = live_template_norms_jax(templates, psd)
    assert isinstance(result, jax.Array)
    expected = jnp.asarray(
        [
            jnp.sum(jnp.abs(t.data[int(t.f_lower / delta_f):]) ** 2)
            * 4
            * delta_f
            for t in templates
        ]
    )
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(expected), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_sigmasq_returns_host_metadata_vector(device):
    """Normalization transfers once for host candidate metadata consumers."""
    if device == "cuda" and not any(
        dev.platform in ("cuda", "gpu") for dev in jax.devices()
    ):
        pytest.skip("CUDA JAX device unavailable")

    class TemplateBatch(list):
        pass

    templates = TemplateBatch()
    for _ in range(2):
        template = type("Template", (), {})()
        template.delta_f = 0.25
        template.f_lower = 0.5
        templates.append(template)
    templates._batch_tensor = jnp.asarray(
        [
            [1 + 1j, 2 + 0j, 3 + 1j, 4 + 0j],
            [2 + 0j, 1 + 2j, 3 + 0j, 1 + 1j],
        ],
        dtype=jnp.complex64,
    )
    psd = jnp.ones(4, dtype=jnp.float32)
    with scheme.JAXScheme(device=device):
        result = batch_sigmasq_jax(templates, psd)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (2,)
    expected = jnp.sum(
        jnp.abs(templates._batch_tensor[:, 2:]) ** 2, axis=1
    ) * (4 * 0.25)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_live_norm_controls_preserve_native_support_and_cache_modes(device):
    if device == "cuda" and not any(
        d.platform == "gpu" for d in jax.devices()
    ):
        pytest.skip("CUDA JAX device unavailable")
    rng = np.random.default_rng(173)
    values = (
        rng.normal(size=(3, 257)) + 1j * rng.normal(size=(3, 257))
    ).astype(np.complex64)
    psd_values = rng.uniform(0.5, 2.0, 257).astype(np.float32)

    class Templates(list):
        pass

    def templates():
        result = Templates()
        for row, flow, end in zip(
            values, (2.0, 4.0, 6.0), (90.0, 100.0, 110.0)
        ):
            template = FrequencySeries(row, delta_f=0.5)
            template.approximant = "TaylorF2"
            template.f_lower = flow
            template.min_f_lower = flow - 0.5
            template.end_frequency = end
            template.sigmasq = types.MethodType(sigma_cached, template)
            result.append(template)
        return result

    with scheme.CPUScheme():
        original = templates()
        spectrum = FrequencySeries(psd_values, delta_f=0.5)
        expected = np.asarray([t.sigmasq(spectrum) for t in original])
    candidate = templates()
    spectrum = FrequencySeries(psd_values, delta_f=0.5)
    with scheme.JAXScheme(device):
        default = batch_sigmasq_jax(candidate, spectrum)
    operations = ("squared_norm", "divide", "inner")
    with scheme.JAXScheme(device, reference_operations=operations) as ctx:
        result = live_template_norms_jax(candidate, spectrum)
        assert result.devices() == {ctx.jax_device}
        assert np.asarray(result).tobytes() == expected.tobytes()
        metadata = batch_sigmasq_jax(candidate, spectrum)
        assert metadata is not default
        assert metadata.tobytes() == expected.astype(np.float32).tobytes()
        assert batch_sigmasq_jax(candidate, spectrum) is metadata
