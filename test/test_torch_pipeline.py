import importlib.util
import sys
import types
from pathlib import Path

import lal
import numpy as np
import pytest
import scipy.interpolate
import scipy.signal
import lalsimulation
from igwn_ligolw import lsctables

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.detector import Detector
from pycbc.filter import autocorrelation, matchedfilter, resample, zpk
from pycbc.inject.inject import SGBurstInjectionSet, _InjectionAdder
from pycbc.inject.injfilterrejector import InjFilterRejector
from pycbc.noise import gaussian, reproduceable
from pycbc.psd import inverse_spectrum_truncation, variation, welch
from pycbc.strain import gate as strain_gate
from pycbc.strain import calibration, lines as strain_lines, recalibrate
from pycbc.strain.strain import StrainBuffer, detect_loud_glitches, gate_data
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
import pycbc.vetoes.chisq as chisq
from pycbc.waveform import ringdown, sinegauss, utils as waveform_utils
from pycbc.waveform import waveform as waveform_module

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
    fake_pkg.gate_data = gate_data
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


def _load_inference_model_module(name):
    """Load a model module without inference's optional dependencies."""
    module_path = (
        Path(pycbc.__file__).parent / "inference" / "models" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_pycbc_inference_{name}_torch_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INFERENCE_DATA_UTILS = _load_inference_model_module("data_utils")
_INFERENCE_TOOLS = _load_inference_model_module("tools")


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("custom_frequencies", (False, True))
def test_waveform_evolution_helpers_stay_on_device(
        torch_device_ctx, monkeypatch, dtype, custom_frequencies):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.complex128:
        pytest.skip("Torch MPS does not support complex128")

    size = 128
    delta_f = 0.25
    indices = np.arange(size)
    phase = -0.002 * indices ** 2
    phase[82:] += 0.995 * np.pi + 0.002 * (2 * 82 - 1)
    amplitude = np.zeros(size)
    amplitude[4:120] = np.linspace(0.2, 1.0, 116)
    waveform = (amplitude * np.exp(1j * phase)).astype(dtype)
    frequencies = None
    if custom_frequencies:
        frequencies = np.cumsum(
            np.linspace(0.2, 0.3, size)
        ).astype(np.float64)

    cpu_series = FrequencySeries(waveform, delta_f=delta_f, epoch=1234)
    expected_phase = waveform_utils.phase_from_frequencyseries(cpu_series)
    expected_amp = waveform_utils.amplitude_from_frequencyseries(cpu_series)
    expected_time = waveform_utils.time_from_frequencyseries(
        cpu_series, sample_frequencies=frequencies
    )

    with ctx:
        torch_series = FrequencySeries(
            waveform, delta_f=delta_f, epoch=1234
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Waveform helper copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_phase = waveform_utils.phase_from_frequencyseries(
                torch_series
            )
            actual_amp = waveform_utils.amplitude_from_frequencyseries(
                torch_series
            )
            actual_time = waveform_utils.time_from_frequencyseries(
                torch_series, sample_frequencies=frequencies
            )

    tolerance = 2e-5 if dtype == np.complex64 else 1e-12
    for actual, expected in (
        (actual_phase, expected_phase),
        (actual_amp, expected_amp),
        (actual_time, expected_time),
    ):
        assert actual._data.tensor.device.type == device
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(
            actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
            rtol=tolerance, atol=tolerance,
        )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_polarization_evolution_helpers_stay_on_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    size = 257
    delta_t = 1 / 4096
    indices = np.arange(size)
    phase = 0.01 * indices + 0.0002 * indices ** 2
    amplitude = 1 + 0.1 * np.sin(indices / 17)
    plus = (amplitude * np.cos(phase)).astype(dtype)
    cross = (amplitude * np.sin(phase)).astype(dtype)

    cpu_plus = TimeSeries(plus, delta_t=delta_t, epoch=1234)
    cpu_cross = TimeSeries(cross, delta_t=delta_t, epoch=1234)
    expected_phase = waveform_utils.phase_from_polarizations(
        cpu_plus, cpu_cross
    )
    expected_amp = waveform_utils.amplitude_from_polarizations(
        cpu_plus, cpu_cross
    )
    expected_freq = waveform_utils.frequency_from_polarizations(
        cpu_plus, cpu_cross
    )

    with ctx:
        torch_plus = TimeSeries(plus, delta_t=delta_t, epoch=1234)
        torch_cross = TimeSeries(cross, delta_t=delta_t, epoch=1234)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Polarization helper copied data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_phase = waveform_utils.phase_from_polarizations(
                torch_plus, torch_cross
            )
            actual_amp = waveform_utils.amplitude_from_polarizations(
                torch_plus, torch_cross
            )
            actual_freq = waveform_utils.frequency_from_polarizations(
                torch_plus, torch_cross
            )

    tolerance = 2e-5 if dtype == np.float32 else 1e-12
    for actual, expected in (
        (actual_phase, expected_phase),
        (actual_amp, expected_amp),
        (actual_freq, expected_freq),
    ):
        assert actual._data.tensor.device.type == device
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(
            actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
            rtol=tolerance, atol=tolerance,
        )


def test_lal_detector_projection_returns_to_torch_device(torch_device_ctx):
    ctx, device = torch_device_ctx
    delta_t = 1 / 2048
    epoch = 1_000_000_000
    times = np.arange(512) * delta_t
    hp_data = np.sin(2 * np.pi * 80 * times)
    hc_data = 0.4 * np.cos(2 * np.pi * 80 * times)
    detector = Detector("H1")
    input_dtype = np.float32 if device == "mps" else np.float64
    hp_data = hp_data.astype(input_dtype)
    hc_data = hc_data.astype(input_dtype)

    expected = detector.project_wave(
        TimeSeries(hp_data, delta_t=delta_t, epoch=epoch),
        TimeSeries(hc_data, delta_t=delta_t, epoch=epoch),
        1.2,
        -0.4,
        0.3,
        method="lal",
    ).numpy()

    with ctx:
        projected = detector.project_wave(
            TimeSeries(hp_data, delta_t=delta_t, epoch=epoch),
            TimeSeries(hc_data, delta_t=delta_t, epoch=epoch),
            1.2,
            -0.4,
            0.3,
            method="lal",
        )

    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert projected._data.tensor.device.type == device
    assert projected._data.tensor.dtype == expected_dtype
    torch.testing.assert_close(
        projected._data.tensor.detach().cpu(),
        torch.as_tensor(expected, dtype=expected_dtype),
        rtol=0,
        atol=0,
    )


def test_varying_detector_projection_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    delta_t = 1 / 2048
    epoch = 1_126_259_462
    times = np.arange(4096) * delta_t
    hp_data = np.sin(2 * np.pi * 80 * times)
    hc_data = 0.4 * np.cos(2 * np.pi * 80 * times)
    detector = Detector("H1")
    input_dtype = np.float32 if device == "mps" else np.float64
    hp_data = hp_data.astype(input_dtype)
    hc_data = hc_data.astype(input_dtype)

    expected = detector.project_wave(
        TimeSeries(hp_data, delta_t=delta_t, epoch=epoch),
        TimeSeries(hc_data, delta_t=delta_t, epoch=epoch),
        1.2,
        -0.4,
        0.3,
        method="vary_polarization",
    ).numpy()

    with ctx:
        hp = TimeSeries(hp_data, delta_t=delta_t, epoch=epoch)
        hc = TimeSeries(hc_data, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Detector projection copied data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            projected = detector.project_wave(
                hp,
                hc,
                1.2,
                -0.4,
                0.3,
                method="vary_polarization",
            )

    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert projected._data.tensor.device.type == device
    assert projected._data.tensor.dtype == expected_dtype
    torch.testing.assert_close(
        projected._data.tensor.detach().cpu(),
        torch.as_tensor(expected, dtype=expected_dtype),
        rtol=2e-6 if device == "mps" else 2e-9,
        atol=2e-6 if device == "mps" else 2e-9,
    )


def test_time_marginalization_weights_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    sample_count = 64
    epoch = 1_126_259_461
    delta_t = 1 / 1024
    times = np.arange(4096) * delta_t
    dtype = np.complex64 if device == "mps" else np.complex128
    snr_data = {
        "H1": (
            np.exp(0.2j * times)
            * (1 + 2 * np.exp(-((times - 2.0) / 0.1) ** 2))
        ).astype(dtype),
        "L1": (
            np.exp(0.3j * times)
            * (1 + 1.5 * np.exp(-((times - 2.01) / 0.12) ** 2))
        ).astype(dtype),
    }

    def make_marginalizer():
        marginalizer = _INFERENCE_TOOLS.DistMarg()
        marginalizer.marginalized_vector_priors = {
            "tc": types.SimpleNamespace(
                bounds={"tc": (epoch + 1.5, epoch + 2.0)}
            )
        }
        marginalizer._current_params = {"ra": 1.2, "dec": -0.4}
        marginalizer.vsamples = sample_count
        marginalizer.marginalize_vector_params = {}
        marginalizer.marginalize_vector_weights = np.zeros(sample_count)
        return marginalizer

    def make_snrs():
        return {
            ifo: TimeSeries(data, delta_t=delta_t, epoch=epoch)
            for ifo, data in snr_data.items()
        }

    np.random.seed(1234)
    expected = make_marginalizer().draw_times(make_snrs())
    expected = {
        key: np.array(value, copy=True) for key, value in expected.items()
    }

    with ctx:
        snrs = make_snrs()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Marginalization copied the complete SNR series to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            np.random.seed(1234)
            actual = make_marginalizer().draw_times(snrs)

    for snr in snrs.values():
        assert snr._data.tensor.device.type == device
    np.testing.assert_allclose(actual["tc"], expected["tc"], rtol=0, atol=0)
    np.testing.assert_allclose(
        actual["logw_partial"],
        expected["logw_partial"],
        rtol=2e-6 if device == "mps" else 1e-13,
        atol=2e-6 if device == "mps" else 1e-13,
    )


def test_inference_nan_check_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    finite_data = np.linspace(-1, 1, 1024, dtype=dtype)
    nan_data = finite_data.copy()
    nan_data[713] = np.nan

    with ctx:
        finite = TimeSeries(finite_data, delta_t=1 / 1024)
        contains_nan = TimeSeries(nan_data, delta_t=1 / 1024)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("NaN validation copied strain to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            _INFERENCE_DATA_UTILS.check_for_nans({"H1": finite})
            with pytest.raises(ValueError, match="NaN found in strain from L1"):
                _INFERENCE_DATA_UTILS.check_for_nans({"L1": contains_nan})

    assert finite._data.tensor.device.type == device
    assert contains_nan._data.tensor.device.type == device


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "sample_offset",
    (96.0, 96.25, 96.6, -31.3, 1800.25),
)
def test_lalsim_injection_adder_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype, sample_offset):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    rng = np.random.default_rng(173)
    delta_t = 1 / 1024
    epoch = lal.LIGOTimeGPS(1_000_000_000)
    source_epoch = epoch + sample_offset * delta_t
    target_data = rng.normal(scale=0.1, size=2048).astype(dtype)
    source_data = rng.normal(size=777).astype(dtype)

    reference = TimeSeries(
        target_data.copy(), delta_t=delta_t, epoch=epoch
    ).lal()
    reference_source = TimeSeries(
        source_data, delta_t=delta_t, epoch=source_epoch
    ).lal()
    add_reference = (
        lalsimulation.SimAddInjectionREAL4TimeSeries
        if dtype == np.float32
        else lalsimulation.SimAddInjectionREAL8TimeSeries
    )
    add_reference(reference, reference_source, None)
    expected = reference.data.data.copy()

    with ctx:
        target = TimeSeries(
            target_data, delta_t=delta_t, epoch=epoch
        )
        source = TimeSeries(
            source_data, delta_t=delta_t, epoch=source_epoch
        )

        with monkeypatch.context() as patch:
            def _reject_lal_conversion(_self):
                raise AssertionError("Torch injection converted through LAL")

            def _reject_host_transfer(_self):
                raise AssertionError("Torch injection copied data to host")

            def _reject_lalsimulation(*_args, **_kwargs):
                raise AssertionError("Torch injection used lalsimulation")

            patch.setattr(TimeSeries, "lal", _reject_lal_conversion)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                lalsimulation,
                "SimAddInjectionREAL4TimeSeries",
                _reject_lalsimulation,
            )
            patch.setattr(
                lalsimulation,
                "SimAddInjectionREAL8TimeSeries",
                _reject_lalsimulation,
            )
            adder = _InjectionAdder(target)
            adder.add(source)
            adder.finish()

    assert target._data.tensor.device.type == device
    tolerances = (
        {"rtol": 2e-5, "atol": 2e-6}
        if dtype == np.float32 or device == "mps"
        else {"rtol": 1e-11, "atol": 1e-12}
    )
    torch.testing.assert_close(
        target._data.tensor.detach().cpu(),
        torch.as_tensor(expected),
        **tolerances,
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        (
            np.array([0, 0, 1, -2, 3, 0], dtype=np.float32),
            np.array([1, -2, 3], dtype=np.float32),
        ),
        (
            np.array([0, 1 + 2j, -3j, 0], dtype=np.complex64),
            np.array([1 + 2j, -3j], dtype=np.complex64),
        ),
        (
            np.zeros(5, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        ),
        (
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32),
        ),
    ),
)
def test_trim_zeros_stays_on_torch_device(
        torch_device_ctx, monkeypatch, values, expected):
    ctx, device = torch_device_ctx
    if device == "mps" and np.iscomplexobj(values):
        pytest.skip("Torch MPS PyCBC arrays do not support complex dtypes")

    with ctx:
        array = Array(values)

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("trim_zeros copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            trimmed = array.trim_zeros()

    assert isinstance(trimmed, Array)
    assert trimmed._data.tensor.device.type == device
    torch.testing.assert_close(
        trimmed._data.tensor.detach().cpu(),
        torch.as_tensor(expected),
    )


@pytest.mark.parametrize(
    "parameters",
    (
        {
            "amp": 1.0,
            "quality": 8.0,
            "central_frequency": 150.0,
            "fmin": 20.0,
            "fmax": 512.0,
            "delta_f": 0.25,
        },
        {
            "amp": 0.3,
            "quality": 100.0,
            "central_frequency": 80.0,
            "fmin": 30.0,
            "fmax": 256.0,
            "delta_f": 0.5,
        },
        {
            "amp": 2.0,
            "quality": 4.0,
            "central_frequency": 60.0,
            "fmin": 55.0,
            "fmax": 128.0,
            "delta_f": 1.0,
        },
    ),
)
def test_fd_sine_gaussian_torch_matches_cpu_without_host_transfer(
        torch_device_ctx, monkeypatch, parameters):
    ctx, device = torch_device_ctx
    reference = sinegauss.fd_sine_gaussian(**parameters).numpy().copy()
    sinegauss._cached_torch_arange.cache_clear()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "sine-Gaussian generation copied Torch data to host"
                )

            def _reject_numpy(*_args, **_kwargs):
                raise AssertionError(
                    "sine-Gaussian generation used a NumPy array operation"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(sinegauss.numpy, "arange", _reject_numpy)
            patch.setattr(sinegauss.numpy, "zeros", _reject_numpy)
            patch.setattr(sinegauss.numpy, "exp", _reject_numpy)
            result = sinegauss.fd_sine_gaussian(**parameters)
            cast = result.astype(np.complex64)

    expected_dtype = torch.complex64 if device == "mps" else torch.complex128
    assert isinstance(result, FrequencySeries)
    assert result._data.tensor.device.type == device
    assert result._data.tensor.dtype == expected_dtype
    assert cast._data.tensor.device.type == device
    assert len(result) == round(parameters["fmax"] / parameters["delta_f"])
    assert result.delta_f == parameters["delta_f"]

    expected = torch.as_tensor(reference, dtype=expected_dtype)
    tolerances = (
        {"rtol": 2e-5, "atol": 1e-7}
        if device == "mps"
        else {"rtol": 1e-12, "atol": 1e-14}
    )
    torch.testing.assert_close(
        result._data.tensor.detach().cpu(), expected, **tolerances
    )


_SGBURST_CASES = (
    {
        "q": 9.0,
        "frequency": 150.0,
        "hrss": 1e-21,
        "eccentricity": 0.0,
        "polarization": 0.0,
        "delta_t": 1 / 4096,
    },
    {
        "q": 4.2,
        "frequency": 80.0,
        "hrss": 2e-22,
        "eccentricity": 1.0,
        "polarization": np.pi / 2,
        "delta_t": 1 / 2048,
    },
    {
        "q": 25.0,
        "frequency": 512.0,
        "hrss": 4e-21,
        "eccentricity": 0.6,
        "polarization": -0.7,
        "delta_t": 1 / 8192,
    },
)


def _lalsim_sgburst_reference(parameters):
    plus, cross = lalsimulation.SimBurstSineGaussian(
        parameters["q"],
        parameters["frequency"],
        parameters["hrss"],
        parameters["eccentricity"],
        parameters["polarization"],
        parameters["delta_t"],
    )
    return (
        plus.data.data.copy(),
        cross.data.data.copy(),
        float(plus.epoch),
    )


@pytest.mark.parametrize("parameters", _SGBURST_CASES)
def test_td_sine_gaussian_cpu_matches_lalsimulation(parameters):
    expected_plus, expected_cross, expected_epoch = (
        _lalsim_sgburst_reference(parameters)
    )

    plus, cross = waveform_module.get_sgburst_waveform(**parameters)

    assert isinstance(plus, TimeSeries)
    assert isinstance(cross, TimeSeries)
    assert plus.delta_t == parameters["delta_t"]
    assert cross.delta_t == parameters["delta_t"]
    assert float(plus.start_time) == expected_epoch
    assert float(cross.start_time) == expected_epoch
    np.testing.assert_allclose(
        plus.numpy(), expected_plus, rtol=5e-14, atol=1e-35
    )
    np.testing.assert_allclose(
        cross.numpy(), expected_cross, rtol=5e-14, atol=1e-35
    )


@pytest.mark.parametrize("parameters", _SGBURST_CASES)
def test_td_sine_gaussian_torch_matches_lalsimulation_without_host_transfer(
        torch_device_ctx, monkeypatch, parameters):
    ctx, device = torch_device_ctx
    expected_plus, expected_cross, expected_epoch = (
        _lalsim_sgburst_reference(parameters)
    )
    sinegauss._cached_torch_time_grid.cache_clear()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_lalsimulation(*_args, **_kwargs):
                raise AssertionError("sine-Gaussian generation used LAL")

            def _reject_host_transfer(_self):
                raise AssertionError(
                    "sine-Gaussian generation copied Torch data to host"
                )

            def _reject_numpy(*_args, **_kwargs):
                raise AssertionError(
                    "sine-Gaussian generation used a NumPy array operation"
                )

            patch.setattr(
                lalsimulation,
                "SimBurstSineGaussian",
                _reject_lalsimulation,
            )
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for operation in (
                "abs", "arange", "cos", "exp", "ones", "sin", "where"
            ):
                patch.setattr(sinegauss.numpy, operation, _reject_numpy)

            plus, cross = waveform_module.get_sgburst_waveform(**parameters)

    dtype = torch.float32 if device == "mps" else torch.float64
    tolerances = (
        {"rtol": 1e-4, "atol": 1e-26}
        if device == "mps"
        else {"rtol": 2e-12, "atol": 1e-35}
    )
    for actual, expected in (
        (plus, expected_plus),
        (cross, expected_cross),
    ):
        assert isinstance(actual, TimeSeries)
        assert actual._data.tensor.device.type == device
        assert actual._data.tensor.dtype == dtype
        assert actual.delta_t == parameters["delta_t"]
        assert float(actual.start_time) == expected_epoch
        torch.testing.assert_close(
            actual._data.tensor.detach().cpu(),
            torch.as_tensor(expected, dtype=dtype),
            **tolerances,
        )


def test_sgburst_requires_delta_t_and_has_a_real_registry_entry():
    with pytest.raises(ValueError, match="delta_t"):
        waveform_module.get_sgburst_waveform(
            q=9,
            frequency=150,
            hrss=1e-21,
        )

    assert waveform_module.sgburst_approximants() == ["SineGaussian"]

    with pytest.raises(ValueError, match="Unknown sine-Gaussian"):
        waveform_module.get_sgburst_waveform(
            approximant="TaylorF2",
            q=9,
            frequency=150,
            hrss=1e-21,
            delta_t=1 / 4096,
        )


def _sgburst_injection_row(central_time=1_000_000_002.123456789):
    injection = lsctables.SimBurst()
    injection.time_geocent = lal.LIGOTimeGPS(central_time)
    injection.q = 9.0
    injection.frequency = 150.0
    injection.hrss = 1e-21
    injection.pol_ellipse_e = 0.6
    injection.pol_ellipse_angle = -0.7
    injection.ra = 1.1
    injection.dec = -0.4
    injection.psi = 0.3
    return injection


def _sgburst_injection_set(injection):
    injection_set = object.__new__(SGBurstInjectionSet)
    injection_set.table = [injection]
    injection_set.extra_args = {}
    return injection_set


def _lalsim_sgburst_injection_reference(
        injection, delta_t, length, epoch, dtype, distance_scale):
    plus_lal, cross_lal = lalsimulation.SimBurstSineGaussian(
        float(injection.q),
        float(injection.frequency),
        float(injection.hrss),
        float(injection.pol_ellipse_e),
        float(injection.pol_ellipse_angle),
        delta_t,
    )
    plus = TimeSeries(
        plus_lal.data.data.copy(),
        delta_t=plus_lal.deltaT,
        epoch=plus_lal.epoch,
    )
    cross = TimeSeries(
        cross_lal.data.data.copy(),
        delta_t=cross_lal.deltaT,
        epoch=cross_lal.epoch,
    )
    plus /= distance_scale
    cross /= distance_scale

    central_time = float(injection.time_geocent)
    plus.start_time += central_time
    cross.start_time += central_time
    detector = Detector("H1")
    fp, fc = detector.antenna_pattern(
        injection.ra,
        injection.dec,
        injection.psi,
        central_time,
    )
    delay = detector.time_delay_from_earth_center(
        injection.ra,
        injection.dec,
        central_time,
    )
    signal = plus * float(fp) + cross * float(fc)
    signal.start_time = float(signal.start_time) + delay

    strain = TimeSeries(
        np.zeros(length, dtype=dtype),
        delta_t=delta_t,
        epoch=epoch,
    )
    strain.inject(signal.astype(dtype), copy=False)
    return strain.numpy().copy()


def test_sgburst_injection_matches_lalsimulation_and_detector_response():
    delta_t = 1 / 2048
    length = 4 * 2048
    epoch = 1_000_000_000
    distance_scale = 2.5
    injection = _sgburst_injection_row()
    expected = _lalsim_sgburst_injection_reference(
        injection,
        delta_t,
        length,
        epoch,
        np.float64,
        distance_scale,
    )
    strain = TimeSeries(
        np.zeros(length, dtype=np.float64),
        delta_t=delta_t,
        epoch=epoch,
    )

    _sgburst_injection_set(injection).apply(
        strain,
        "H1",
        distance_scale=distance_scale,
    )

    np.testing.assert_allclose(strain.numpy(), expected, rtol=2e-12, atol=1e-34)


def test_sgburst_injection_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    delta_t = 1 / 2048
    length = 4 * 2048
    epoch = 1_000_000_000
    distance_scale = 2.5
    injection = _sgburst_injection_row()
    expected = _lalsim_sgburst_injection_reference(
        injection,
        delta_t,
        length,
        epoch,
        np.float32,
        distance_scale,
    )
    sinegauss._cached_torch_time_grid.cache_clear()

    with ctx:
        strain = TimeSeries(
            np.zeros(length, dtype=np.float32),
            delta_t=delta_t,
            epoch=epoch,
        )
        with monkeypatch.context() as patch:
            def _reject_lalsimulation(*_args, **_kwargs):
                raise AssertionError("burst injection used lalsimulation")

            def _reject_lal_conversion(*_args, **_kwargs):
                raise AssertionError("burst injection converted through LAL")

            def _reject_host_transfer(_self):
                raise AssertionError("burst injection copied data to host")

            patch.setattr(
                lalsimulation,
                "SimBurstSineGaussian",
                _reject_lalsimulation,
            )
            patch.setattr(
                lalsimulation,
                "SimAddInjectionREAL4TimeSeries",
                _reject_lalsimulation,
            )
            patch.setattr(
                lalsimulation,
                "SimDetectorStrainREAL8TimeSeries",
                _reject_lalsimulation,
            )
            patch.setattr(TimeSeries, "lal", _reject_lal_conversion)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)

            _sgburst_injection_set(injection).apply(
                strain,
                "H1",
                distance_scale=distance_scale,
            )

    assert strain._data.tensor.device.type == device
    assert strain._data.tensor.dtype == torch.float32
    tolerances = (
        {"rtol": 3e-4, "atol": 1e-26}
        if device == "mps"
        else {"rtol": 2e-5, "atol": 3e-28}
    )
    torch.testing.assert_close(
        strain._data.tensor.detach().cpu(),
        torch.as_tensor(expected),
        **tolerances,
    )


def test_sgburst_injection_skips_disjoint_waveforms(monkeypatch):
    injection = _sgburst_injection_row(central_time=1_000_000_100)
    strain = TimeSeries(
        np.zeros(2048, dtype=np.float64),
        delta_t=1 / 2048,
        epoch=1_000_000_000,
    )

    def _reject_generation(*_args, **_kwargs):
        raise AssertionError("disjoint burst waveform was generated")

    monkeypatch.setattr(
        waveform_module,
        "get_sgburst_waveform",
        _reject_generation,
    )
    _sgburst_injection_set(injection).apply(strain, "H1")
    np.testing.assert_array_equal(strain.numpy(), 0)


def _injection_filter_rejector():
    rejector = object.__new__(InjFilterRejector)
    rejector.enabled = True
    rejector.match_threshold = 0.8
    rejector.coarsematch_deltaf = 1.0
    rejector.coarsematch_fmax = 64.0
    rejector.short_injections = {}
    rejector._short_psd_storage = {}
    return rejector


def test_injection_filter_rejector_coarsens_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    delta_t = 1 / 512
    waveform = (
        np.sin(np.linspace(0, 40 * np.pi, 777))
        * np.hanning(777)
        * 1e-21
    ).astype(np.float32)
    psd_values = np.linspace(1, 3, 513).astype(np.float32)

    cpu_rejector = _injection_filter_rejector()
    cpu_rejector.generate_short_inj_from_inj(
        TimeSeries(waveform, delta_t=delta_t), "injection"
    )
    expected_injection = cpu_rejector.short_injections["injection"]
    expected_psd = cpu_rejector._get_short_psd(
        FrequencySeries(psd_values, delta_f=0.25)
    )

    with ctx:
        rejector = _injection_filter_rejector()
        torch_waveform = TimeSeries(waveform, delta_t=delta_t)
        torch_psd = FrequencySeries(psd_values, delta_f=0.25)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "injection filter rejection copied data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            rejector.generate_short_inj_from_inj(
                torch_waveform, "injection"
            )
            actual_psd = rejector._get_short_psd(torch_psd)
            cached_psd = rejector._get_short_psd(torch_psd)

    actual_injection = rejector.short_injections["injection"]
    assert actual_injection._data.tensor.device.type == device
    assert actual_psd._data.tensor.device.type == device
    assert actual_injection.dtype == np.dtype(np.complex64)
    assert actual_psd.dtype == np.dtype(np.float32)
    assert actual_injection.delta_f == rejector.coarsematch_deltaf
    assert actual_psd.delta_f == rejector.coarsematch_deltaf
    assert cached_psd is actual_psd
    assert list(rejector._short_psd_storage) == [id(torch_psd)]
    injection_atol = 2e-8 if device == "mps" else 7e-9
    np.testing.assert_allclose(
        actual_injection._data.tensor.detach().cpu().numpy(),
        expected_injection.numpy(),
        rtol=5e-5,
        atol=injection_atol,
    )
    np.testing.assert_allclose(
        actual_psd._data.tensor.detach().cpu().numpy(),
        expected_psd.numpy(),
        rtol=0,
        atol=0,
    )


def test_power_chisq_bins_stay_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    htilde_values = np.ones(513, dtype=np.complex64)
    psd_values = np.ones(513, dtype=np.float32)
    parameters = {
        "num_bins": 8,
        "low_frequency_cutoff": 20,
        "high_frequency_cutoff": 400,
    }
    expected = chisq.power_chisq_bins(
        FrequencySeries(htilde_values, delta_f=1),
        parameters["num_bins"],
        FrequencySeries(psd_values, delta_f=1),
        parameters["low_frequency_cutoff"],
        parameters["high_frequency_cutoff"],
    )

    with ctx:
        htilde = FrequencySeries(htilde_values, delta_f=1)
        psd = FrequencySeries(psd_values, delta_f=1)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("chi-squared bins copied data to host")

            def _reject_numpy_search(*_args, **_kwargs):
                raise AssertionError("chi-squared bins used NumPy searchsorted")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(chisq.numpy, "searchsorted", _reject_numpy_search)
            actual = chisq.power_chisq_bins(
                htilde,
                parameters["num_bins"],
                psd,
                parameters["low_frequency_cutoff"],
                parameters["high_frequency_cutoff"],
            )

    np.testing.assert_array_equal(actual, expected)


def _ringdown_parameters():
    return {
        "lmns": ["221", "331", "201"],
        "amp220": 1.2,
        "phi220": 0.4,
        "f_220": 250.0,
        "tau_220": 0.02,
        "amp330": 0.35,
        "phi330": -0.3,
        "f_330": 410.0,
        "tau_330": 0.012,
        "amp200": 0.15,
        "phi200": 0.8,
        "f_200": 180.0,
        "tau_200": 0.018,
        "inclination": 0.7,
        "azimuthal": 0.2,
    }


def _assert_ringdown_close(actual, expected, device):
    dtype = torch.float32 if device == "mps" else torch.float64
    if np.iscomplexobj(expected):
        dtype = torch.complex64 if device == "mps" else torch.complex128
    tolerances = (
        {"rtol": 3e-5, "atol": 1e-6}
        if device == "mps"
        else {"rtol": 1e-12, "atol": 1e-14}
    )
    torch.testing.assert_close(
        actual.detach().cpu(),
        torch.as_tensor(expected, dtype=dtype),
        **tolerances,
    )


def test_td_ringdown_torch_matches_cpu_without_host_transfer(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = _ringdown_parameters()
    parameters.update(
        delta_t=1 / 4096,
        t_final=0.05,
        taper=True,
        dbeta=0.1,
        dphi=-0.2,
    )
    reference_plus, reference_cross = ringdown.get_td_from_freqtau(
        **parameters
    )
    plus_values = reference_plus.numpy().copy()
    cross_values = reference_cross.numpy().copy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "time-domain ringdown copied Torch data to host"
                )

            def _reject_numpy_exp(*_args, **_kwargs):
                raise AssertionError(
                    "time-domain ringdown evaluated exponentials with NumPy"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
            plus, cross = ringdown.get_td_from_freqtau(**parameters)

    expected_dtype = torch.float32 if device == "mps" else torch.float64
    for result in (plus, cross):
        assert isinstance(result, TimeSeries)
        assert result._data.tensor.device.type == device
        assert result._data.tensor.dtype == expected_dtype
    assert len(plus) == len(reference_plus)
    assert plus.delta_t == reference_plus.delta_t
    assert float(plus.start_time) == float(reference_plus.start_time)
    _assert_ringdown_close(plus._data.tensor, plus_values, device)
    _assert_ringdown_close(cross._data.tensor, cross_values, device)


def test_fd_ringdown_torch_matches_cpu_without_host_transfer(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = _ringdown_parameters()
    parameters.update(f_lower=20.0, f_final=1024.0, t_0=0.003)
    reference_plus, reference_cross = ringdown.get_fd_from_freqtau(
        **parameters
    )
    plus_values = reference_plus.numpy().copy()
    cross_values = reference_cross.numpy().copy()

    unshifted_plus, _ = ringdown.get_fd_from_freqtau(
        **(parameters | {"t_0": 0.0})
    )
    assert np.max(np.abs(plus_values - unshifted_plus.numpy())) > 0

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "frequency-domain ringdown copied Torch data to host"
                )

            def _reject_numpy_exp(*_args, **_kwargs):
                raise AssertionError(
                    "frequency-domain ringdown evaluated exponentials "
                    "with NumPy"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
            plus, cross = ringdown.get_fd_from_freqtau(**parameters)

    expected_dtype = (
        torch.complex64 if device == "mps" else torch.complex128
    )
    for result in (plus, cross):
        assert isinstance(result, FrequencySeries)
        assert result._data.tensor.device.type == device
        assert result._data.tensor.dtype == expected_dtype
    assert len(plus) == len(reference_plus)
    assert plus.delta_f == reference_plus.delta_f
    kmin = int(parameters["f_lower"] / plus.delta_f)
    assert torch.count_nonzero(plus._data.tensor[:kmin]) == 0
    _assert_ringdown_close(plus._data.tensor, plus_values, device)
    _assert_ringdown_close(cross._data.tensor, cross_values, device)


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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("factor", (2, 4, 8))
def test_butterworth_resample_torch_matches_lal_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, factor):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_rate = 2048
    delta_t = 1 / sample_rate
    target_delta_t = factor * delta_t
    samples = np.arange(4099) * delta_t
    data = (
        np.sin(2 * np.pi * 31 * samples)
        + 0.2 * np.cos(2 * np.pi * 173 * samples)
    ).astype(dtype)
    expected = resample.resample_to_delta_t(
        TimeSeries(data, delta_t=delta_t, epoch=321), target_delta_t
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=delta_t, epoch=321)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Butterworth resampling copied Torch samples to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(zpk, "_TORCH_SOS_TARGET_BLOCK_SIZE", 128)
            actual = resample.resample_to_delta_t(
                input_series, target_delta_t
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == target_delta_t
    assert actual.start_time == expected.start_time
    assert len(actual) == len(data) // factor

    if device == "mps":
        rtol = 2e-4
        atol = np.max(np.abs(expected.numpy())) * 2e-5
    elif dtype == np.float32:
        rtol, atol = 5e-6, 5e-7
    else:
        rtol, atol = 5e-10, 5e-11
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_array_equal(
        input_series._data.tensor.detach().cpu().numpy(), data
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("filter_order", (5, 8), ids=("odd", "even"))
@pytest.mark.parametrize("filter_name", ("highpass", "lowpass"))
def test_butterworth_filter_torch_matches_lal_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, filter_order, filter_name):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_rate = 2048
    delta_t = 1 / sample_rate
    samples = np.arange(4099) * delta_t
    data = (
        np.sin(2 * np.pi * 31 * samples)
        + 0.2 * np.cos(2 * np.pi * 317 * samples)
    ).astype(dtype)
    filter_func = getattr(resample, filter_name)
    expected = filter_func(
        TimeSeries(data, delta_t=delta_t, epoch=654),
        97,
        filter_order=filter_order,
        attenuation=0.17,
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=delta_t, epoch=654)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Butterworth filtering copied Torch samples to host"
                )

            def _reject_lal(*_args, **_kwargs):
                raise AssertionError("Butterworth filtering called LAL")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(TimeSeries, "lal", _reject_lal)
            patch.setattr(zpk, "_TORCH_SOS_TARGET_BLOCK_SIZE", 128)
            actual = filter_func(
                input_series,
                97,
                filter_order=filter_order,
                attenuation=0.17,
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == delta_t
    assert actual.start_time == expected.start_time
    assert len(actual) == len(expected)

    if device == "mps":
        rtol, atol = 3e-4, 3e-5
    elif dtype == np.float32:
        rtol, atol = 5e-6, 5e-7
    else:
        rtol, atol = 5e-10, 5e-11
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_array_equal(
        input_series._data.tensor.detach().cpu().numpy(), data
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


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize(
    "parameters",
    (
        ([20, 30], [1, 2, 100, 120], 5e4),
        ([40], [2, 80, 160], 1e3),
    ),
    ids=("two-second-order-sections", "first-and-second-order-sections"),
)
def test_filter_zpk_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, parameters):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(1845)
    data = rng.normal(size=4097)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=data.size)
    data = data.astype(dtype)
    expected = zpk.filter_zpk(
        TimeSeries(data, delta_t=1 / 2048, epoch=789), *parameters
    )

    with ctx:
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=789)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("filter_zpk copied Torch samples to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(zpk, "_TORCH_SOS_TARGET_BLOCK_SIZE", 128)
            actual = zpk.filter_zpk(input_series, *parameters)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    if device == "mps":
        rtol = 2e-3
        atol = np.max(np.abs(expected.numpy())) * 1e-3
    elif dtype in (np.float32, np.complex64):
        rtol, atol = 5e-5, 5e-6
    else:
        rtol, atol = 5e-10, 5e-11
    np.testing.assert_allclose(
        actual_data, expected.numpy(), rtol=rtol, atol=atol
    )
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


@pytest.mark.parametrize("use_psd", (False, True))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_optimized_match_stays_on_device(
        torch_device_ctx, monkeypatch, use_psd, dtype):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    sample_count = 2048
    frequency_count = sample_count // 2 + 1
    delta_f = 1.0
    delta_t = 1.0 / (sample_count * delta_f)
    frequencies = np.arange(frequency_count) * delta_f
    amplitude = np.exp(-0.5 * ((frequencies - 200) / 75) ** 2)
    data = (
        amplitude
        * np.exp(1j * (0.013 * frequencies + 3e-5 * frequencies**2))
    ).astype(dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    psd_data = (1.0 + (frequencies / 300) ** 2).astype(real_dtype)
    shift = 5.5 * delta_t

    cpu_waveform = FrequencySeries(data, delta_f=delta_f)
    cpu_shifted = cpu_waveform.cyclic_time_shift(shift)
    cpu_psd = FrequencySeries(psd_data, delta_f=delta_f) if use_psd else None
    expected = matchedfilter.optimized_match(
        cpu_waveform, cpu_shifted, psd=cpu_psd,
        low_frequency_cutoff=20, high_frequency_cutoff=800,
        return_phase=True,
    )

    with ctx:
        waveform = FrequencySeries(data, delta_f=delta_f)
        shifted = waveform.cyclic_time_shift(shift)
        psd = FrequencySeries(
            psd_data, delta_f=delta_f
        ) if use_psd else None
        waveform_data = waveform._data.tensor.clone()
        shifted_data = shifted._data.tensor.clone()

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("optimized match copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = matchedfilter.optimized_match(
                waveform, shifted, psd=psd,
                low_frequency_cutoff=20, high_frequency_cutoff=800,
                return_phase=True,
            )

    assert waveform._data.tensor.device.type == device
    assert shifted._data.tensor.device.type == device
    assert torch.equal(waveform._data.tensor, waveform_data)
    assert torch.equal(shifted._data.tensor, shifted_data)
    tolerance = 2e-4 if dtype == np.complex64 else 1e-6
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    )


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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "location",
    (
        "TAPER_NONE",
        "TAPER_START",
        "TAPER_END",
        "TAPER_STARTEND",
        "start",
        "end",
        "startend",
    ),
)
def test_timeseries_lal_taper_stays_on_device(
        torch_device_ctx, monkeypatch, dtype, location):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    samples = np.arange(512)
    data = np.zeros(512, dtype=dtype)
    signal_samples = samples[17:491] - 17
    data[17:491] = (
        np.sin(2 * np.pi * signal_samples / 23 + 0.3)
        * (1 + 0.001 * signal_samples)
    ).astype(dtype)
    expected = TimeSeries(
        data, delta_t=1 / 2048, epoch=123
    ).taper_timeseries(location=location).numpy()

    with ctx:
        series = TimeSeries(data, delta_t=1 / 2048, epoch=123)
        original = series._data.tensor.clone()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("taper copied Torch data to host")

            def _reject_lal(_self):
                raise AssertionError("Torch taper constructed a LAL series")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(TimeSeries, "lal", _reject_lal)
            patch.setitem(sys.modules, "lalsimulation", None)
            actual = series.taper_timeseries(location=location)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert torch.equal(series._data.tensor, original)
    tolerance = 2e-6 if dtype == np.float32 else 1e-14
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "location",
    (
        "TAPER_NONE",
        "TAPER_START",
        "TAPER_END",
        "TAPER_STARTEND",
        "start",
        "end",
        "startend",
    ),
)
def test_timeseries_constant_taper_stays_on_device(
        torch_device_ctx, monkeypatch, dtype, location):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    delta_t = 1 / 32
    taper_window = 0.25
    epoch = 123
    data = np.zeros(128, dtype=dtype)
    data[17:111] = np.linspace(0.5, 2.0, 94, dtype=dtype)

    normalized = {
        "TAPER_NONE": "none",
        "TAPER_START": "start",
        "TAPER_END": "end",
        "TAPER_STARTEND": "startend",
    }.get(location, location)
    gate_params = []
    if normalized in ("start", "startend"):
        gate_params.append((epoch + 17 * delta_t, 0, taper_window))
    if normalized in ("end", "startend"):
        gate_params.append((epoch + 110 * delta_t, 0, taper_window))
    expected = gate_data(
        TimeSeries(data.copy(), delta_t=delta_t, epoch=epoch), gate_params
    ).numpy()
    cpu_actual = TimeSeries(
        data.copy(), delta_t=delta_t, epoch=epoch
    ).taper_timeseries(
        location=location,
        tapermethod="constant",
        taper_window=taper_window,
    ).numpy()
    np.testing.assert_allclose(cpu_actual, expected, rtol=0, atol=0)

    with ctx:
        series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("constant taper copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = series.taper_timeseries(
                location=location,
                tapermethod="constant",
                taper_window=taper_window,
            )

    assert actual is series
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-14,
        atol=2e-6 if dtype == np.float32 else 1e-14,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_line_removal_stays_on_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_rate = 128
    delta_t = 1 / sample_rate
    epoch = 100
    samples = np.arange(sample_rate * 4) * delta_t
    data = (
        0.7 * np.sin(2 * np.pi * 13 * samples + 0.3)
        + 0.2 * np.cos(2 * np.pi * 5 * samples)
    ).astype(dtype)

    cpu_series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
    expected_model = strain_lines.line_model(
        13, cpu_series, epoch, amp=0.7, phi=0.3
    )
    expected_inner = strain_lines.avg_inner_product(
        cpu_series, expected_model, bin_size=0.5
    )
    expected_calibrated = strain_lines.calibration_lines(
        [13], cpu_series
    )
    expected_cleaned = strain_lines.clean_data(
        [13], cpu_series.copy(), chunk=2, avg_bin=0.5
    )

    with ctx:
        torch_series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
        clean_input = torch_series.copy()
        original = torch_series._data.tensor.clone()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("line removal copied Torch data to host")

            def _reject_numpy(*_args, **_kwargs):
                raise AssertionError("line removal called a NumPy kernel")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(strain_lines.numpy, "conjugate", _reject_numpy)
            patch.setattr(strain_lines.numpy, "exp", _reject_numpy)
            patch.setattr(strain_lines.numpy, "hanning", _reject_numpy)
            patch.setattr(strain_lines.numpy, "median", _reject_numpy)
            actual_model = strain_lines.line_model(
                13, torch_series, epoch, amp=0.7, phi=0.3
            )
            actual_inner = strain_lines.avg_inner_product(
                torch_series, actual_model, bin_size=0.5
            )
            actual_calibrated = strain_lines.calibration_lines(
                [13], torch_series
            )
            actual_cleaned = strain_lines.clean_data(
                [13], clean_input, chunk=2, avg_bin=0.5
            )

    assert actual_model._data.tensor.device.type == device
    assert actual_calibrated._data.tensor.device.type == device
    assert actual_cleaned._data.tensor.device.type == device
    assert actual_model.delta_t == delta_t
    assert actual_model.start_time == expected_model.start_time
    assert actual_calibrated.start_time == expected_calibrated.start_time
    assert actual_cleaned is clean_input
    assert torch.equal(torch_series._data.tensor, original)

    if device == "mps":
        rtol, atol = 3e-4, 3e-5
        assert actual_model.dtype == np.dtype(np.complex64)
        assert actual_calibrated.dtype == np.dtype(np.float32)
    elif dtype == np.float32:
        rtol, atol = 3e-6, 3e-7
        assert actual_model.dtype == expected_model.dtype
        assert actual_calibrated.dtype == expected_calibrated.dtype
    else:
        rtol, atol = 2e-12, 2e-13
        assert actual_model.dtype == expected_model.dtype
        assert actual_calibrated.dtype == expected_calibrated.dtype

    np.testing.assert_allclose(
        actual_model._data.tensor.detach().cpu().numpy(),
        expected_model.numpy(), rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        actual_inner[0], expected_inner[0], rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        actual_inner[1:], expected_inner[1:], rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        actual_calibrated._data.tensor.detach().cpu().numpy(),
        expected_calibrated.numpy(), rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        actual_cleaned._data.tensor.detach().cpu().numpy(),
        expected_cleaned.numpy(), rtol=rtol, atol=atol,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_calibration_spline_evaluation_stays_on_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    spline_points = np.linspace(10, 100, 10)
    parameters = np.array((0, 5, -4, 6, -5, 4, -3, 2, -1, 0))
    spline = scipy.interpolate.UnivariateSpline(spline_points, parameters)
    assert len(spline.get_knots()) > 2
    samples = np.linspace(0, 120, 257).astype(dtype)
    expected = spline(samples)

    with ctx:
        frequencies = Array(samples)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("spline evaluation copied grid to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = calibration._evaluate_spline(spline, frequencies)

    assert actual.device.type == device
    rtol, atol = ((2e-4, 2e-4) if dtype == np.float32
                  else (1e-12, 1e-12))
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=rtol, atol=atol
    )


def _cubic_calibration_model(model_class):
    model = model_class(
        ifo_name="h1", minimum_frequency=10,
        maximum_frequency=1024, n_points=8,
    )
    amplitude = (0, 4, -3, 5, -4, 3, -2, 0)
    phase = (0, -2, 3, -4, 3, -2, 1, 0)
    for index, (amp, pha) in enumerate(zip(amplitude, phase)):
        model.params[f"amplitude_h1_{index}"] = amp
        model.params[f"phase_h1_{index}"] = pha
    return model


@pytest.mark.parametrize(
    "model_class", (calibration.CubicSpline, recalibrate.CubicSpline)
)
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_cubic_calibration_stays_on_device(
        torch_device_ctx, monkeypatch, model_class, dtype):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    data = (
        np.linspace(1, 2, 129) + 1j * np.linspace(-0.5, 0.5, 129)
    ).astype(dtype)
    expected = _cubic_calibration_model(model_class).apply_calibration(
        FrequencySeries(data, delta_f=8, epoch=123)
    )

    with ctx:
        torch_strain = FrequencySeries(data, delta_f=8, epoch=123)
        original = torch_strain._data.tensor.clone()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("calibration copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = _cubic_calibration_model(
                model_class
            ).apply_calibration(torch_strain)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == expected.dtype
    assert actual.delta_f == expected.delta_f
    assert actual.epoch == expected.epoch
    assert torch.equal(torch_strain._data.tensor, original)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=2e-11, atol=2e-11,
    )


def _physical_calibration_model():
    frequencies = np.linspace(10, 1024, 33)
    return recalibrate.PhysicalModel(
        freq=frequencies,
        fc0=350,
        c0=(1 + 0.1j) * np.ones(33),
        d0=(0.6 - 0.03j) * np.ones(33),
        a_tst0=(0.4 + 0.02j) * np.ones(33),
        a_pu0=(0.3 - 0.01j) * np.ones(33),
        fs0=8,
        qinv0=0.2,
    )


def test_physical_calibration_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    data = np.linspace(1, 2, 129).astype(np.complex128)
    adjustments = {
        "delta_fs": 0.4,
        "delta_qinv": 0.01,
        "delta_fc": 5,
        "kappa_c": 0.98,
        "kappa_tst_re": 1.02,
        "kappa_tst_im": 0.01,
        "kappa_pu_re": 0.99,
        "kappa_pu_im": -0.02,
    }
    expected = _physical_calibration_model().adjust_strain(
        FrequencySeries(data, delta_f=8, epoch=123), **adjustments
    )

    with ctx:
        torch_strain = FrequencySeries(data, delta_f=8, epoch=123)
        original = torch_strain._data.tensor.clone()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("calibration copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = _physical_calibration_model().adjust_strain(
                torch_strain, **adjustments
            )

    assert actual._data.tensor.device.type == device
    assert actual.dtype == expected.dtype
    assert actual.delta_f == expected.delta_f
    assert actual.epoch == expected.epoch
    assert torch.equal(torch_strain._data.tensor, original)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=1e-12, atol=1e-12,
    )
