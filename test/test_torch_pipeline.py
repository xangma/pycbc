import sys
import types

import numpy as np
import pytest
import scipy.signal

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.filter import autocorrelation, matchedfilter, resample
from pycbc.noise import gaussian, reproduceable
from pycbc.psd import inverse_spectrum_truncation, variation, welch
from pycbc.strain import gate as strain_gate
from pycbc.strain.strain import StrainBuffer, detect_loud_glitches
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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_noise_from_psd_stays_on_device_and_is_reproducible(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    segment_length = 1024
    delta_t = 1 / 1024
    delta_f = 1 / (segment_length * delta_t)
    psd_values = np.linspace(
        0.5, 2.0, segment_length // 2 + 1, dtype=dtype
    )

    with ctx:
        psd = FrequencySeries(psd_values, delta_f=delta_f)
        original_psd = psd._data.tensor.clone()

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "noise generation copied Torch data to host"
                )

            def _reject_lal(*_args, **_kwargs):
                raise AssertionError("noise generation called LAL")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(gaussian.lal, "gsl_rng", _reject_lal)
            patch.setattr(gaussian, "lalsimulation", None)
            first = gaussian.noise_from_psd(
                1537, delta_t, psd, seed=9182
            )
            repeated = gaussian.noise_from_psd(
                1537, delta_t, psd, seed=9182
            )
            different = gaussian.noise_from_psd(
                1537, delta_t, psd, seed=9183
            )

    assert isinstance(first._data.tensor, torch.Tensor)
    assert first._data.tensor.device.type == device
    assert first.dtype == np.dtype(dtype)
    assert len(first) == 1537
    assert first.delta_t == delta_t
    assert torch.isfinite(first._data.tensor).all()
    assert torch.equal(first._data.tensor, repeated._data.tensor)
    assert not torch.equal(first._data.tensor, different._data.tensor)
    assert torch.equal(psd._data.tensor, original_psd)


def test_noise_from_psd_flat_spectrum_has_expected_variance(torch_ctx):
    segment_length = 2048
    delta_t = 1 / 1024
    delta_f = 1 / (segment_length * delta_t)
    psd_level = 2.0
    psd_values = np.full(segment_length // 2 + 1, psd_level)

    with torch_ctx:
        psd = FrequencySeries(psd_values, delta_f=delta_f)
        noise = gaussian.noise_from_psd(
            segment_length * 64, delta_t, psd, seed=1729
        )

    # DC and Nyquist are zeroed, so this is the integral of the flat
    # one-sided PSD over the remaining positive-frequency bins.
    expected_variance = (
        segment_length // 2 - 1
    ) * psd_level * delta_f
    actual_variance = noise._data.tensor.var(unbiased=False).item()
    assert actual_variance == pytest.approx(expected_variance, rel=0.05)


def test_reproducible_colored_noise_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Colored noise requires complex PyCBC arrays on MPS")

    monkeypatch.setattr(reproduceable, "BLOCK_SAMPLES", 512)
    sample_rate = 64
    start_time, end_time = 100, 102
    psd_values = np.linspace(1.0, 2.0, 65)
    parameters = dict(
        seed=1729,
        sample_rate=sample_rate,
        low_frequency_cutoff=1.0,
        filter_duration=1,
    )
    expected = reproduceable.colored_noise(
        FrequencySeries(psd_values, delta_f=0.5),
        start_time, end_time, **parameters
    )

    with ctx:
        torch_psd = FrequencySeries(psd_values, delta_f=0.5)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Reproducible colored noise copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = reproduceable.colored_noise(
                torch_psd, start_time, end_time, **parameters
            )

    assert actual._data.tensor.device.type == device
    assert actual.start_time == expected.start_time == start_time
    assert actual.end_time == expected.end_time == end_time
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=1e-11, atol=1e-12,
    )


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


def test_live_psd_variation_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    sample_rate = 64
    rng = np.random.default_rng(9182)
    strain_values = rng.standard_normal(30 * sample_rate).astype(np.float32)
    psd_values = np.linspace(1.0, 2.0, 129, dtype=np.float32)

    cpu_strain = TimeSeries(
        strain_values, delta_t=1 / sample_rate, epoch=100
    )
    cpu_psd = FrequencySeries(psd_values, delta_f=0.25)
    expected_filter = variation.live_create_filter(
        cpu_psd, 4, sample_rate, low_freq=5, high_freq=20
    )
    expected_series = variation.live_calc_psd_variation(
        cpu_strain, expected_filter, 4, data_trim=0.5
    )
    trigger_times = np.array([
        float(expected_series.start_time) - 1,
        float(expected_series.start_time) + 0.5,
        float(expected_series.end_time) - 1,
        float(expected_series.end_time) + 1,
    ])
    expected_values = variation.live_find_var_value(
        {"end_time": trigger_times}, expected_series
    )

    with ctx:
        torch_strain = TimeSeries(
            strain_values, delta_t=1 / sample_rate, epoch=100
        )
        torch_psd = FrequencySeries(psd_values, delta_f=0.25)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "PSD variation copied a full PyCBC array to host"
                )

            def _reject_scipy(*_args, **_kwargs):
                raise AssertionError("PSD variation used its SciPy data path")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(variation.sig, "fftconvolve", _reject_scipy)
            patch.setattr(variation, "interp1d", _reject_scipy)
            actual_filter = variation.live_create_filter(
                torch_psd, 4, sample_rate, low_freq=5, high_freq=20
            )
            actual_series = variation.live_calc_psd_variation(
                torch_strain, actual_filter, 4, data_trim=0.5
            )
            actual_values = variation.live_find_var_value(
                {"end_time": trigger_times}, actual_series
            )

    assert isinstance(actual_filter, torch.Tensor)
    assert actual_filter.device.type == device
    assert actual_series._data.tensor.device.type == device
    assert actual_series.dtype == np.dtype(np.float32)
    assert actual_series.start_time == expected_series.start_time
    np.testing.assert_allclose(
        actual_filter.detach().cpu().numpy(), expected_filter,
        rtol=5e-5, atol=5e-6,
    )
    np.testing.assert_allclose(
        actual_series._data.tensor.detach().cpu().numpy(),
        expected_series.numpy(), rtol=5e-5, atol=5e-6,
    )
    np.testing.assert_allclose(
        actual_values, expected_values, rtol=5e-5, atol=5e-6
    )


def test_offline_psd_variation_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Welch requires complex PyCBC arrays, unsupported on MPS")
    sample_rate = 64
    rng = np.random.default_rng(1729)
    strain_values = rng.standard_normal(30 * sample_rate).astype(np.float32)
    parameters = (4, 0.25, 24, 2, 1, "median", 5, 20)

    cpu_strain = TimeSeries(
        strain_values, delta_t=1 / sample_rate, epoch=100
    )
    expected = variation.calc_filt_psd_variation(
        cpu_strain, *parameters
    )
    trigger_times = np.array([
        float(expected.start_time) - 1,
        float(expected.start_time) + 0.5,
        float(expected.end_time) - 1,
        float(expected.end_time) + 1,
    ])
    trigger_indices = (trigger_times - 100) * sample_rate
    expected_values = variation.find_trigger_value(
        expected, trigger_indices, 100, sample_rate
    )

    with ctx:
        torch_strain = TimeSeries(
            strain_values, delta_t=1 / sample_rate, epoch=100
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "PSD variation copied a full PyCBC array to host"
                )

            def _reject_scipy(*_args, **_kwargs):
                raise AssertionError("PSD variation used its SciPy data path")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(variation.sig, "fftconvolve", _reject_scipy)
            patch.setattr(variation, "interp1d", _reject_scipy)
            actual = variation.calc_filt_psd_variation(
                torch_strain, *parameters
            )
            actual_values = variation.find_trigger_value(
                actual, trigger_indices, 100, sample_rate
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(np.float32)
    assert actual.start_time == expected.start_time
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_values, expected_values, rtol=1e-4, atol=1e-5
    )


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


@pytest.mark.parametrize("unbiased", (False, True))
def test_autocorrelation_stays_on_device(
        torch_device_ctx, monkeypatch, unbiased):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Autocorrelation requires complex PyCBC arrays on MPS")

    rng = np.random.default_rng(8675309)
    data = np.empty(257)
    data[0] = rng.normal()
    for index in range(1, len(data)):
        data[index] = 0.7 * data[index - 1] + rng.normal()
    delta_t = 1 / 1024
    expected = autocorrelation.calculate_acf(
        TimeSeries(data, delta_t=delta_t), unbiased=unbiased
    )
    expected_acl = None
    if not unbiased:
        expected_acl = autocorrelation.calculate_acl(
            TimeSeries(data, delta_t=delta_t)
        )

    with ctx:
        torch_data = TimeSeries(data, delta_t=delta_t)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Autocorrelation copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = autocorrelation.calculate_acf(
                torch_data, unbiased=unbiased
            )
            actual_acl = None
            if not unbiased:
                actual_acl = autocorrelation.calculate_acl(torch_data)

    assert actual._data.tensor.device.type == device
    assert actual.delta_t == expected.delta_t == delta_t
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=1e-10, atol=1e-11,
    )
    if not unbiased:
        assert actual_acl == expected_acl


def test_detect_loud_glitches_thresholds_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Glitch detection requires complex PyCBC arrays on MPS")

    sample_rate = 128
    rng = np.random.default_rng(1234)
    data = rng.normal(size=sample_rate * 16)
    data[sample_rate * 6] += 80
    data[sample_rate * 11] -= 100
    parameters = dict(
        psd_duration=2,
        psd_stride=1,
        low_freq_cutoff=10,
        threshold=6,
        cluster_window=0.5,
        corrupt_time=1,
    )
    expected = detect_loud_glitches(
        TimeSeries(data, delta_t=1 / sample_rate, epoch=1000),
        **parameters,
    )

    with ctx:
        torch_data = TimeSeries(
            data, delta_t=1 / sample_rate, epoch=1000
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Glitch detection copied the full series to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = detect_loud_glitches(torch_data, **parameters)

    assert actual == expected


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


@pytest.mark.parametrize("complex_system", (False, True))
def test_torch_levinson_solver_matches_scipy(
        torch_device_ctx, complex_system):
    ctx, device = torch_device_ctx
    if device == "mps" and complex_system:
        pytest.skip("Torch MPS does not support complex tensors")

    rng = np.random.default_rng(314159)
    length = 31
    rho = 0.55 * (np.exp(0.2j) if complex_system else 1.0)
    coefficients = (rho ** np.arange(length)).astype(
        np.complex64 if complex_system else np.float32
    )
    rhs = rng.normal(size=length)
    if complex_system:
        rhs = rhs + 1j * rng.normal(size=length)
    rhs = rhs.astype(np.complex64 if complex_system else np.float32)
    expected = strain_gate.linalg.solve_toeplitz(coefficients, rhs)

    with ctx:
        actual = strain_gate._torch_solve_toeplitz(
            torch.as_tensor(coefficients, device=device),
            torch.as_tensor(rhs, device=device),
        )

    rtol, atol = ((2e-5, 2e-5) if device == "mps"
                  else (1e-11, 1e-12))
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_paint_gate_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Paint gating requires complex PyCBC arrays on MPS")

    length = 256
    delta_t = 1 / 256
    rng = np.random.default_rng(20250815)
    data = rng.normal(size=length).astype(dtype)
    invpsd_values = (
        1.0 + 0.35 * np.cos(np.linspace(0, np.pi, length // 2 + 1))
    ).astype(dtype)
    lindex, rindex = 97, 121

    expected = strain_gate.gate_and_paint(
        TimeSeries(data, delta_t=delta_t, epoch=123),
        lindex,
        rindex,
        FrequencySeries(invpsd_values, delta_f=1.0),
    )

    with ctx:
        torch_data = TimeSeries(data, delta_t=delta_t, epoch=123)
        torch_invpsd = FrequencySeries(invpsd_values, delta_f=1.0)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Paint gating copied Torch data to host")

            def _reject_scipy(*_args, **_kwargs):
                raise AssertionError("Paint gating used SciPy's Toeplitz solve")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(strain_gate.linalg, "solve_toeplitz", _reject_scipy)
            actual = strain_gate.gate_and_paint(
                torch_data, lindex, rindex, torch_invpsd
            )

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.start_time == expected.start_time
    rtol, atol = ((5e-5, 5e-6) if dtype == np.float32
                  else (1e-11, 1e-12))
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )


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


@pytest.mark.parametrize(
    "length,window,threshold,expected",
    ((10, 4, 7, (1, 2)), (3, 8, 0, (0, 0))),
)
def test_followup_background_reduction_stays_on_device(
        torch_device_ctx, monkeypatch, length, window, threshold, expected):
    ctx, device = torch_device_ctx
    data = np.arange(length, dtype=np.float32)
    cpu_result = matchedfilter._count_louder_background(
        TimeSeries(data, delta_t=1 / 1024), window, threshold
    )

    with ctx:
        background = TimeSeries(data, delta_t=1 / 1024)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "followup background copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = matchedfilter._count_louder_background(
                background, window, threshold
            )

    assert cpu_result == expected
    assert actual == expected
    assert background._data.tensor.device.type == device


@pytest.mark.parametrize("detrend_type", ("constant", "linear", "c", "l"))
@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_timeseries_detrend_stays_on_device(
        torch_device_ctx, monkeypatch, dtype, detrend_type):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(161803)
    positions = np.arange(257)
    data = 3.2 + 0.03 * positions + rng.normal(scale=0.2, size=257)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * (
            -1.3 + 0.07 * positions + rng.normal(scale=0.1, size=257)
        )
    data = data.astype(dtype)
    expected = scipy.signal.detrend(data, type=detrend_type)

    with ctx:
        series = TimeSeries(data, delta_t=1 / 2048, epoch=123)
        original = series._data.tensor.clone()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("detrend copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = series.detrend(type=detrend_type)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert torch.equal(series._data.tensor, original)
    rtol, atol = ((1e-4, 1e-5) if actual.precision == "single"
                  else (1e-11, 1e-12))
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=rtol, atol=atol,
    )


@pytest.mark.parametrize("detrend_type", ("constant", "linear"))
def test_timeseries_detrend_single_sample(torch_device_ctx, detrend_type):
    ctx, device = torch_device_ctx
    with ctx:
        actual = TimeSeries(
            np.array([3.5], dtype=np.float32), delta_t=0.25
        ).detrend(type=detrend_type)

    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.item() == 0


def test_timeseries_detrend_rejects_unknown_type(torch_device_ctx):
    ctx, _ = torch_device_ctx
    with ctx:
        series = TimeSeries(
            np.arange(4, dtype=np.float32), delta_t=0.25
        )
        with pytest.raises(ValueError, match="Trend type must be"):
            series.detrend(type="quadratic")
