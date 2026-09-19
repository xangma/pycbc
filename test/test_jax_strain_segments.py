"""Tests for JAX-backed strain segment FFT batching."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain.strain import StrainSegments
from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData

jax = pytest.importorskip("jax")

if not getattr(__import__("pycbc"), "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


_jax_devices = ["cpu"]
if any(device.platform in ("gpu", "cuda") for device in jax.devices()):
    _jax_devices.append("gpu")


@pytest.fixture(params=_jax_devices, ids=lambda device: f"jax-{device}")
def jax_device(request):
    return request.param


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_fourier_segments_stay_device_native(
    monkeypatch, dtype, jax_device
):
    """Batch segment FFTs do not materialize slices on the host."""
    data = (
        np.arange(16, dtype=dtype) + np.sin(np.arange(16))
    ).astype(dtype)
    expected = [
        np.fft.rfft(data[start:stop]).astype(
            np.complex64 if dtype == np.float32 else np.complex128
        ) * 0.25
        for start, stop in ((0, 8), (8, 16))
    ]

    with scheme.JAXScheme(device=jax_device):
        strain = TimeSeries(data, delta_t=0.25, epoch=100.5)
        segments = StrainSegments(strain, segment_length=2)

        def fail_host_transfer(_self):
            raise AssertionError("JAX strain data was copied to the host")

        monkeypatch.setattr(JAXArrayData, "numpy", fail_host_transfer)
        monkeypatch.setattr(
            "pycbc.strain.strain.numpy.stack",
            lambda *_args, **_kwargs: pytest.fail(
                "JAX segment batching used NumPy stack"
            ),
        )
        actual = segments.fourier_segments()

    assert len(actual) == 2
    for index, (freq_seg, reference) in enumerate(zip(actual, expected)):
        assert isinstance(freq_seg._data, JAXArrayData)
        np.testing.assert_allclose(
            np.asarray(freq_seg._data.array), reference, rtol=1e-5, atol=1e-6
        )
        assert freq_seg.dtype == reference.dtype
        assert freq_seg.delta_f == 0.5
        assert freq_seg.start_time == 100.5 + index * 2
        assert freq_seg.analyze == slice(0, 8)
        assert freq_seg.cumulative_index == index * 8


def test_cpu_fourier_segments_keep_the_existing_fallback():
    """The non-JAX path retains its numerical result and metadata."""
    data = np.arange(16, dtype=np.float64) + np.cos(np.arange(16))
    expected = [
        np.fft.rfft(data[start:stop]) * 0.25
        for start, stop in ((0, 8), (8, 16))
    ]

    with scheme.CPUScheme():
        strain = TimeSeries(data, delta_t=0.25, epoch=100.5)
        actual = StrainSegments(strain, segment_length=2).fourier_segments()

    assert len(actual) == 2
    for index, (freq_seg, reference) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(freq_seg.numpy(), reference)
        assert freq_seg.dtype == np.complex128
        assert freq_seg.start_time == 100.5 + index * 2
        assert freq_seg.analyze == slice(0, 8)
        assert freq_seg.cumulative_index == index * 8
