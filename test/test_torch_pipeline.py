import sys
import types

import numpy as np
import pytest
import scipy.signal

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter, resample
from pycbc.psd import inverse_spectrum_truncation, welch
from pycbc.strain.strain import StrainBuffer
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture(autouse=True)
def stub_execute_cached_fft_import(monkeypatch):
    """Avoid importing the full strain stack in Welch tests."""
    fake_pkg = types.ModuleType("pycbc.strain")
    fake_mod = types.ModuleType("pycbc.strain.strain")

    def _execute_cached_fft(*args, **kwargs):
        raise AssertionError("Welch cache path should not be used in this test")

    def _execute_cached_ifft(*args, **kwargs):
        raise AssertionError("PSD cache path should not be used in this test")

    def _create_memory_and_engine_for_class_based_fft(*args, **kwargs):
        raise AssertionError("Welch cache path should not be used in this test")

    fake_mod.execute_cached_fft = _execute_cached_fft
    fake_mod.execute_cached_ifft = _execute_cached_ifft
    fake_mod.create_memory_and_engine_for_class_based_fft = (
        _create_memory_and_engine_for_class_based_fft
    )
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_pkg)
    monkeypatch.setitem(sys.modules, "pycbc.strain.strain", fake_mod)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        # Allow other tests to construct schemes after we exit
        del ctx
        scheme.Scheme._single = None


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_ctx(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    ctx = scheme.TorchScheme(device)
    try:
        yield ctx, device
    finally:
        del ctx
        scheme.Scheme._single = None


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


def _make_strain_buffer(data, sample_rate, reduced_pad):
    """Build only the state needed by ``overwhitened_data``."""
    delta_f = 1.0
    initial_len = sample_rate + 2 * reduced_pad
    assert len(data) == initial_len

    psdt = FrequencySeries(
        np.ones(initial_len // 2 + 1),
        delta_f=sample_rate / initial_len,
    )
    psd = FrequencySeries(
        np.ones(sample_rate // 2 + 1),
        delta_f=delta_f,
    )
    psd.psdt = psdt

    buffer = object.__new__(StrainBuffer)
    buffer.strain = TimeSeries(data, delta_t=1 / sample_rate, epoch=100)
    buffer.sample_rate = sample_rate
    buffer.reduced_pad = reduced_pad
    buffer.trim_padding = 0
    buffer.segments = {}
    buffer.psds = {delta_f: psd}
    return buffer


@pytest.mark.parametrize("avg_method", ["mean", "median", "median-mean"])
@pytest.mark.parametrize("num_segments", [7, 8])
def test_psd_welch_torch_matches_cpu(torch_ctx, avg_method, num_segments):
    # Short deterministic signal
    rng = np.random.default_rng(1234)
    seg_len = 256
    data = rng.standard_normal(num_segments * seg_len)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = welch(ts_cpu, seg_len, seg_stride=seg_len,
                    avg_method=avg_method)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = welch(ts_t, seg_len, seg_stride=seg_len,
                      avg_method=avg_method)

    assert isinstance(psd_t._data.tensor, torch.Tensor)
    assert psd_t._data.tensor.device.type == "cpu"
    assert psd_t.kind == psd_cpu.kind == "real"
    np.testing.assert_allclose(psd_t.numpy(), psd_cpu.numpy(), rtol=1e-12,
                               atol=1e-14)


@pytest.mark.parametrize("trunc_method", [None, "hann"])
def test_inverse_spectrum_truncation_torch_matches_cpu(torch_ctx,
                                                       trunc_method):
    values = np.linspace(1.0, 4.0, 257)
    psd_cpu = FrequencySeries(values, delta_f=1.0)
    expected = inverse_spectrum_truncation(
        psd_cpu, 64, low_frequency_cutoff=1.0,
        trunc_method=trunc_method,
    )

    with torch_ctx:
        psd_t = FrequencySeries(values, delta_f=1.0)
        actual = inverse_spectrum_truncation(
            psd_t, 64, low_frequency_cutoff=1.0,
            trunc_method=trunc_method,
        )

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-12,
                               atol=1e-12)


def test_whiten_stays_on_device(torch_ctx):
    rng = np.random.default_rng(5678)
    data = rng.standard_normal(4096)
    ts_cpu = TimeSeries(data, delta_t=1 / 2048.0)
    white_cpu = ts_cpu.whiten(2, 1)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 2048.0)
        white_t = ts_t.whiten(2, 1)

    assert isinstance(white_t._data.tensor, torch.Tensor)
    assert white_t._data.tensor.device.type == "cpu"
    assert len(white_t) == len(white_cpu)
    assert _relative_l2(white_t.numpy(), white_cpu.numpy()) < 0.1


def test_strain_buffer_trimmed_fft_matches_cpu(torch_ctx):
    sample_rate = 256
    reduced_pad = 16
    rng = np.random.default_rng(2468)
    data = rng.standard_normal(sample_rate + 2 * reduced_pad)

    cpu_buffer = _make_strain_buffer(data, sample_rate, reduced_pad)
    expected = cpu_buffer.overwhitened_data(1.0)

    with torch_ctx:
        torch_buffer = _make_strain_buffer(data, sample_rate, reduced_pad)
        actual = torch_buffer.overwhitened_data(1.0)

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual.delta_f == expected.delta_f == 1.0
    assert actual.start_time == expected.start_time
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-11,
                               atol=1e-11)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("factor", (2, 4, 8))
def test_resample_to_delta_t_torch_matches_cpu_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, factor):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_rate = 2048
    delta_t = 1 / sample_rate
    target_delta_t = factor * delta_t
    samples = np.arange(4096) * delta_t
    data = (
        np.sin(2 * np.pi * 31 * samples)
        + 0.2 * np.cos(2 * np.pi * 173 * samples)
    ).astype(dtype)
    expected = resample.resample_to_delta_t(
        TimeSeries(data, delta_t=delta_t, epoch=123),
        target_delta_t,
        method="ldas",
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=delta_t, epoch=123)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("LDAS resampling copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = resample.resample_to_delta_t(
                input_series, target_delta_t, method="ldas"
            )

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == target_delta_t
    assert actual.start_time == expected.start_time
    assert actual.corrupted_samples == 10
    assert len(actual) == len(expected)

    rtol, atol = ((5e-5, 5e-6) if dtype == np.float32
                  else (1e-11, 1e-12))
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "length,num_taps,block_size,coefficient_type",
    ((63, 7, 2**18, "numpy"), (4096, 129, 512, "array")),
    ids=("single-block", "overlap-add"),
)
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_lfilter_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, length, num_taps, block_size,
        coefficient_type):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(9182)
    single_precision = dtype in (np.float32, np.complex64)
    real_dtype = np.float32 if single_precision else np.float64
    coefficients = rng.normal(size=num_taps).astype(real_dtype)
    data = rng.normal(size=length)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=length)
    data = data.astype(dtype)
    expected = scipy.signal.lfilter(coefficients, 1.0, data).astype(dtype)

    with ctx:
        filter_coefficients = (
            Array(coefficients) if coefficient_type == "array"
            else coefficients
        )
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=456)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("lfilter copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                resample, "_TORCH_LFILTER_TARGET_BLOCK_SIZE", block_size
            )
            actual = resample.lfilter(filter_coefficients, input_series)

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    rtol, atol = ((5e-5, 5e-5) if single_precision
                  else (1e-11, 1e-11))
    np.testing.assert_allclose(actual_data, expected, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(input_data, data)


def test_matched_filter_torch_vs_cpu(torch_ctx):
    # Simple sine wave template/data with flat PSD
    t = np.arange(0, 1, 1 / 1024.0)
    data = np.sin(2 * np.pi * 50 * t)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = FrequencySeries(np.ones(len(ts_cpu) // 2 + 1), delta_f=ts_cpu.delta_f)
    snr_cpu = matchedfilter.matched_filter(ts_cpu, ts_cpu, psd=psd_cpu)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = FrequencySeries(np.ones(len(ts_t) // 2 + 1), delta_f=ts_t.delta_f)
        snr_t = matchedfilter.matched_filter(ts_t, ts_t, psd=psd_t)

    assert isinstance(snr_t._data.tensor, torch.Tensor)
    assert snr_t._data.tensor.device.type == "cpu"
    assert _relative_l2(snr_t.numpy(), snr_cpu.numpy()) < 0.05
