import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.filter import highpass_fir, lowpass_fir, notch_fir
from pycbc.noise import frequency_noise_from_psd
from pycbc.types import Array, TimeSeries


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


def test_fft_roundtrip(torch_ctx):
    # Real sinusoid, power-of-two length for FFT
    t = np.arange(0, 1, 1 / 1024.0)
    sig = np.sin(2 * np.pi * 50 * t)

    ts_cpu = TimeSeries(sig, delta_t=1 / 1024.0)
    fs_cpu = ts_cpu.to_frequencyseries()
    ts_cpu_rt = fs_cpu.to_timeseries()

    with torch_ctx:
        ts_t = TimeSeries(sig, delta_t=1 / 1024.0)
        fs_t = ts_t.to_frequencyseries()
        ts_t_rt = fs_t.to_timeseries()

    assert isinstance(fs_t._data.tensor, torch.Tensor)
    assert fs_t._data.tensor.device.type == "cpu"
    rel = _relative_l2(ts_t_rt.numpy(), ts_cpu_rt.numpy())
    assert rel < 1e-6


def test_fir_filters_stay_on_device(torch_ctx):
    t = np.arange(0, 1, 1 / 1024.0)
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

    with torch_ctx:
        ts = TimeSeries(sig, delta_t=1 / 1024.0)
        lp = lowpass_fir(ts, 50, 128, beta=5.0)
        hp = highpass_fir(ts, 50, 128, beta=5.0)

    for out in (lp, hp):
        assert isinstance(out._data.tensor, torch.Tensor)
        assert out._data.tensor.device.type == "cpu"
        assert len(out) == len(ts)


def test_noise_from_psd_returns_torch(torch_ctx):
    # Flat PSD, ensure generated noise lives on torch device
    from pycbc.types import FrequencySeries

    psd_vals = FrequencySeries(np.ones(513), delta_f=1.0 / 1024.0)
    with torch_ctx:
        noise = frequency_noise_from_psd(psd_vals, seed=1234)
    assert isinstance(noise._data.tensor, torch.Tensor)
    assert noise._data.tensor.device.type == "cpu"
    assert len(noise) == len(psd_vals)


def test_dtype_view_shares_torch_storage(torch_ctx):
    with torch_ctx:
        real = Array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        complex_view = real.view(np.complex64)

        assert complex_view.ptr == real.ptr
        np.testing.assert_array_equal(
            complex_view.numpy(),
            np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
        )

        complex_view[0] = 5.0 + 6.0j
        np.testing.assert_array_equal(
            real.numpy(), np.array([5.0, 6.0, 3.0, 4.0], dtype=np.float32)
        )


def test_inplace_integer_operation_does_not_silently_truncate(torch_ctx):
    with torch_ctx:
        values = Array(np.array([1, 2], dtype=np.int32))
        with pytest.raises(RuntimeError, match="can't be cast"):
            values += 0.5
