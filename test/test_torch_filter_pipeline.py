import sys
import types

import lal
import numpy as np
import pytest
import scipy.interpolate
import scipy.signal
import scipy.special
import scipy.stats

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

import pycbc
from pycbc import events, scheme
from pycbc.events import eventmgr, ranking, threshold_torch
from pycbc.filter import matchedfilter, resample, zpk
from pycbc.psd import (
    analytical as analytical_psd,
    analytical_space,
    estimate as psd_estimate,
    read as psd_read,
    variation,
    welch,
)
from pycbc.strain import recalibrate
from pycbc.strain.strain import detect_loud_glitches, gate_data
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
import pycbc.vetoes.autochisq as autochisq
import pycbc.vetoes.bank_chisq as bank_chisq
import pycbc.vetoes.chisq as chisq


torch = pytest.importorskip("torch")


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


def test_fir_filters_stay_on_device(torch_ctx):
    t = np.arange(0, 1, 1 / 1024.0)
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

    with torch_ctx:
        ts = TimeSeries(sig, delta_t=1 / 1024.0)
        lp = resample.lowpass_fir(ts, 50, 128, beta=5.0)
        hp = resample.highpass_fir(ts, 50, 128, beta=5.0)

    for out in (lp, hp):
        assert isinstance(out._data.tensor, torch.Tensor)
        assert out._data.tensor.device.type == "cpu"
        assert len(out) == len(ts)


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


_DIRECT_SPACE_CURVES = (
    ("sensitivity_curve_lisa_SciRD", {}),
    (
        "sensitivity_curve_tianqin_analytical",
        {
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "sensitivity_curve_taiji_analytical",
        {
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
    ("confusion_fit_lisa", {"duration": 2.0}),
    ("confusion_fit_tianqin", {"duration": 4.0}),
    ("confusion_fit_taiji", {"duration": 2.0}),
)


_COMBINED_SPACE_CURVES = (
    (
        "sensitivity_curve_lisa_confusion",
        {"base_model": "SciRD", "duration": 2.0},
    ),
    (
        "sensitivity_curve_tianqin_confusion",
        {
            "duration": 2.0,
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "sensitivity_curve_taiji_confusion",
        {
            "duration": 2.0,
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
)


_CONFUSION_SPACE_PSDS = (
    (
        "analytical_psd_tianqin_confusion_noise",
        {"duration": 2.0, "len_arm": 1.8e8},
    ),
    (
        "analytical_psd_taiji_confusion_noise",
        {"duration": 2.0, "len_arm": 3.1e9},
    ),
    (
        "analytical_psd_tianqin_tdi_AE_confusion",
        {
            "duration": 2.0,
            "len_arm": 1.8e8,
            "acc_noise_level": 1.2e-15,
            "oms_noise_level": 1.1e-12,
        },
    ),
    (
        "analytical_psd_taiji_tdi_AE_confusion",
        {
            "duration": 2.0,
            "len_arm": 3.1e9,
            "acc_noise_level": 2.8e-15,
            "oms_noise_level": 8.4e-12,
        },
    ),
)


_SYNTHETIC_LISA_RESPONSE = (
    np.array((1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.0)),
    np.array((2.35, 2.28, 2.04, 1.21, 0.12, 0.0012712348970728724)),
)


_DIRECT_SPACE_RESPONSE_HELPERS = (
    ("averaged_lisa_fplus_sq_numerical", {}),
    ("averaged_tianqin_fplus_sq_numerical", {}),
    ("averaged_response_lisa_tdi", {"tdi": "2.0"}),
    ("averaged_response_tianqin_tdi", {"tdi": "2.0"}),
    ("averaged_response_taiji_tdi", {"tdi": "2.0"}),
)


_LISA_RESPONSE_MODELS = (
    ("sensitivity_curve_lisa_semi_analytical", {}),
    (
        "sensitivity_curve_lisa_confusion",
        {"base_model": "semi", "duration": 2.0},
    ),
    (
        "sh_transformed_psd_lisa_tdi_XYZ",
        {"base_model": "semi", "duration": 2.0, "tdi": "2.0"},
    ),
    (
        "sh_transformed_psd_lisa_tdi_XYZ",
        {"base_model": "SciRD", "duration": 2.0, "tdi": "1.5"},
    ),
    (
        "semi_analytical_psd_lisa_confusion_noise",
        {"duration": 2.0, "tdi": "2.0"},
    ),
    (
        "analytical_psd_lisa_tdi_AE_confusion",
        {"duration": 2.0, "tdi": "2.0"},
    ),
)


_ALIGO_TORCH_100_HZ_PINS = {
    "aLIGOQuantumNoSRMLowPower": 2.1931470245995445e-47,
    "aLIGOQuantumNoSRMHighPower": 3.1834841378930625e-48,
    "aLIGOQuantumZeroDetLowPower": 4.6779675124272452e-47,
    "aLIGOQuantumZeroDetHighPower": 8.4358809516525952e-48,
    "aLIGOQuantumNSNSOpt": 4.5169379541865925e-48,
    "aLIGOQuantumBHBH20Deg": 2.9004873157206200e-47,
    "aLIGOQuantumHighFrequency": 1.9787138993885476e-46,
    "aLIGONoSRMLowPower": 2.8728520233912104e-47,
    "aLIGONoSRMHighPower": 9.9805341258097216e-48,
    "aLIGOZeroDetLowPower": 5.3576725112189112e-47,
    "aLIGOZeroDetHighPower": 1.5232930939569254e-47,
    "aLIGONSNSOpt": 1.1313987942103253e-47,
    "aLIGOBHBH20Deg": 3.5801923145122860e-47,
    "aLIGOHighFrequency": 2.0466843992677141e-46,
    "aLIGOThermal": 6.7970499879166597e-48,
}


_ALIGO_TORCH_CONFIGURATIONS = (
    "NoSRMLowPower",
    "NoSRMHighPower",
    "ZeroDetLowPower",
    "ZeroDetHighPower",
    "NSNSOpt",
    "BHBH20Deg",
    "HighFrequency",
)


_ILIGO_TORCH_100_HZ_PINS = {
    "iLIGOSRD": 1.7323979166300358e-45,
    "iLIGOSeismic": 2.0903586303906188e-54,
    "iLIGOThermal": 2.3207710766653360e-46,
    "iLIGOShot": 2.7077800641050937e-46,
    "eLIGOShot": 9.025933547016977e-47,
    "iLIGOModel": 5.0285511616740164e-46,
    "eLIGOModel": 3.2233644522706198e-46,
}


_GROUND_FIT_TORCH_100_HZ_PINS = {
    "Virgo": 2.9764052287135322e-45,
    "GEO": 6.1707070707089880e-45,
    "GEOHF": 1.4534562248455390e-44,
    "TAMA": 8.1417187500000004e-42,
    "KAGRA": 9.1062488995400101e-48,
    "AdvVirgo": 2.1112328638054695e-47,
}


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


def test_power_chisq_point_indices_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(9328)
    corr_values = (
        rng.normal(size=32) + 1j * rng.normal(size=32)
    ).astype(np.complex64)
    point_values = np.array([1, 4, 9], dtype=np.int64)
    bins = np.array([1, 4, 8, 14], dtype=np.int64)
    expected = np.asarray(chisq.shift_sum(
        FrequencySeries(corr_values, delta_f=0.1), point_values, bins
    ))

    with ctx:
        corr = FrequencySeries(corr_values, delta_f=0.1)
        points = Array(point_values)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("power chi-squared indices copied to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            # The optimized CPU path exposes a zero-copy NumPy ABI view of
            # Torch-owned storage; this is not a device transfer.  Non-CPU
            # devices must continue to avoid NumPy entirely.
            if device != "cpu":
                patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = chisq.shift_sum(corr, points, bins)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_moves_host_summary_to_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 12.0, 20.0, 30.0], dtype=dtype)
    reduced_chisq = np.array([0.5, 1.0, 2.0, 6.0], dtype=dtype)
    expected = ranking.newsnr(snr, reduced_chisq, q=6.0, n=2.0)

    with ctx:
        torch_chisq = Array(reduced_chisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("newSNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.newsnr(snr, torch_chisq, q=6.0, n=2.0)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_effsnr_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0], dtype=dtype)
    rchisq = np.array([0.5, 1.0, 1.5, 4.0, np.nan], dtype=dtype)
    expected = ranking.effsnr(snr, rchisq, fac=200.0)

    with ctx:
        torch_snr = Array(snr)
        torch_rchisq = Array(rchisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("effective SNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.effsnr(
                torch_snr, torch_rchisq, fac=200.0
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0], dtype=dtype)
    rchisq = np.array([-0.5, 1.0, 1.5, 4.0, np.nan], dtype=dtype)
    expected = ranking.newsnr(snr, rchisq, q=8.0, n=3.0)

    with ctx:
        torch_snr = Array(snr)
        torch_rchisq = Array(rchisq)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("newSNR copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = ranking.newsnr(
                torch_snr, torch_rchisq, q=8.0, n=3.0
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
        atol=2e-7 if dtype == np.float32 else 0.0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "function_name, include_psd, kwargs",
    (
        ("newsnr_sgveto", False, {"q": 8.0, "n": 3.0}),
        (
            "newsnr_sgveto_psdvar",
            True,
            {"min_expected_psdvar": 0.65, "q": 8.0, "n": 3.0},
        ),
        (
            "newsnr_sgveto_psdvar_threshold",
            True,
            {
                "min_expected_psdvar": 0.65,
                "brchisq_threshold": 10.0,
                "psd_var_val_threshold": 10.0,
                "q": 8.0,
                "n": 3.0,
            },
        ),
        (
            "newsnr_sgveto_psdvar_scaled",
            True,
            {
                "scaling": 0.33,
                "min_expected_psdvar": 0.65,
                "q": 8.0,
                "n": 3.0,
            },
        ),
        (
            "newsnr_sgveto_psdvar_scaled_threshold",
            True,
            {
                "threshold": 2.0,
                "scaling": 0.33,
                "min_expected_psdvar": 0.65,
                "q": 8.0,
                "n": 3.0,
            },
        ),
    ),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_newsnr_veto_family_stays_on_torch_device(
        torch_device_ctx, monkeypatch, function_name, include_psd,
        kwargs, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0, 9.0], dtype=dtype)
    brchisq = np.array([-0.5, 1.0, 1.5, 4.0, 12.0, np.nan], dtype=dtype)
    sgchisq = np.array([2.0, 4.0, 8.0, 16.0, np.nan, 5.0], dtype=dtype)
    psd_var = np.array([0.5, 0.65, 1.0, 4.0, 12.0, np.nan], dtype=dtype)
    function = getattr(ranking, function_name)
    args = (snr, brchisq, sgchisq)
    if include_psd:
        args += (psd_var,)
    expected = function(*args, **kwargs)

    with ctx:
        torch_args = tuple(Array(value) for value in args)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    f"{function_name} copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = function(*torch_args, **kwargs)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    expected_dtype = np.float32 if device == "mps" else np.float64
    assert actual.dtype == np.dtype(expected_dtype)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=tolerance, atol=0.0, equal_nan=True,
    )


def test_real_threshold_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([-2.0, 0.5, 1.5, 3.0, -4.0], dtype=np.float32)

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        assert series._data.tensor.device.type == device

        def _reject_host_transfer(_self):
            raise AssertionError("full real series transferred to NumPy")

        monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        locations, triggers = events.threshold_real(series, 1.0)

    np.testing.assert_array_equal(locations, np.array([2, 3]))
    np.testing.assert_array_equal(triggers, values[[2, 3]])


@pytest.mark.parametrize("empty", (False, True))
def test_complex_threshold_results_stay_on_torch_device(
        torch_device_ctx, monkeypatch, empty):
    ctx, device = torch_device_ctx
    if device == "mps":
        values = np.array(
            [0.0, 2.0, 3.0, -4.0, 7.0, 0.5], dtype=np.float32
        )
    else:
        values = np.array(
            [0.0, 2.0j, 3.0, -4.0j, 7.0, 0.5j],
            dtype=np.complex64,
        )
    threshold = 20.0 if empty else 1.5
    expected_locations = np.flatnonzero(np.abs(values) > threshold)
    expected_values = values[expected_locations]

    def _reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("Torch threshold results transferred to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            locations, triggers = events.threshold(series, threshold)

    assert isinstance(locations, Array)
    assert isinstance(triggers, Array)
    assert locations._data.tensor.device.type == device
    assert triggers._data.tensor.device.type == device
    assert locations.dtype == np.dtype(np.int64)
    assert triggers.dtype == np.dtype(values.dtype)
    np.testing.assert_array_equal(
        locations._data.tensor.detach().cpu().numpy(), expected_locations
    )
    np.testing.assert_array_equal(
        triggers._data.tensor.detach().cpu().numpy(), expected_values
    )


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("interface", ("function", "engine"))
def test_symmetric_threshold_cluster_results_stay_on_torch_device(
        torch_device_ctx, monkeypatch, empty, interface):
    ctx, device = torch_device_ctx
    if device == "mps":
        values = np.array(
            [0, 2, 3, -4, 7, 5, 2, -6, 1, 0, 8, 2],
            dtype=np.float32,
        )
    else:
        values = np.array(
            [0, 2j, 3, -4j, 7, 5j, 2, -6j, 1, 0, 8j, 2],
            dtype=np.complex64,
        )
    threshold = 20.0 if empty else 1.5
    expected_locations = (
        np.array([], dtype=np.int64)
        if empty else np.array([4, 10], dtype=np.int64)
    )
    expected_values = values[expected_locations]

    def _reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("Torch clustered results transferred to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            if interface == "function":
                triggers, locations = events.threshold_and_cluster(
                    series, threshold, 3
                )
            else:
                engine = events.ThresholdCluster(series)
                triggers, locations = engine.threshold_and_cluster(
                    threshold, 3
                )

    assert isinstance(locations, Array)
    assert isinstance(triggers, Array)
    assert locations._data.tensor.device.type == device
    assert triggers._data.tensor.device.type == device
    assert locations.dtype == np.dtype(np.int64)
    assert triggers.dtype == np.dtype(values.dtype)
    np.testing.assert_array_equal(
        locations._data.tensor.detach().cpu().numpy(), expected_locations
    )
    np.testing.assert_array_equal(
        triggers._data.tensor.detach().cpu().numpy(), expected_values
    )


@pytest.mark.parametrize("real", (False, True))
def test_findchirp_threshold_copies_only_torch_survivors(
        torch_device_ctx, monkeypatch, real):
    ctx, device = torch_device_ctx
    if real:
        values = np.array(
            [-8.0, 2.0, 3.0, 4.0, -7.0, 5.0, 2.0, 6.0, 1.0],
            dtype=np.float32,
        )
        candidate_mask = values > 1.5
        operation = events.threshold_real_and_cluster_findchirp
    else:
        values = np.array(
            [0.0, 2.0j, 3.0, -4.0j, 7.0, 5.0j, 2.0, -6.0j, 1.0],
            dtype=np.complex64,
        )
        candidate_mask = np.abs(values) > 1.5
        operation = events.threshold_and_cluster_findchirp

    candidate_times = np.flatnonzero(candidate_mask)
    candidate_values = values[candidate_times]
    survivor_positions = events.findchirp_cluster_over_window(
        candidate_times, candidate_values, 3
    )
    expected_times = candidate_times[survivor_positions]
    expected_values = candidate_values[survivor_positions]
    assert len(expected_times) < len(candidate_times)

    def reject_host_array(*_args, **_kwargs):
        raise AssertionError("FindChirp copied all Torch candidates to host")

    with ctx:
        series = TimeSeries(values, delta_t=1.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_array)
            patch.setattr(eventmgr, "threshold", reject_host_array)
            patch.setattr(eventmgr, "threshold_real", reject_host_array)
            patch.setattr(eventmgr, "cluster_reduce", reject_host_array)
            patch.setattr(threshold_torch, "threshold", reject_host_array)
            actual_times, actual_values = operation(series, 1.5, 3)

    np.testing.assert_array_equal(actual_times, expected_times)
    np.testing.assert_array_equal(actual_values, expected_values)
    assert series._data.tensor.device.type == device


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


def test_power_chisq_dof_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    indices = np.array([1, 4, 7], dtype=np.int32)

    with ctx:
        corr = Array(np.arange(8, dtype=np.float32))
        snrv = Array(values)
        trigger_indices = indices
        veto = object.__new__(chisq.SingleDetPowerChisq)
        veto.do = True
        veto.snr_threshold = None
        veto.cached_chisq_bins = lambda *_args: [0, 2, 4]

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "power chi-squared metadata used NumPy repeat"
                )

            patch.setattr(
                chisq,
                "power_chisq_at_points_from_precomputed",
                lambda *_args: Array(values),
            )
            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                corr, snrv, 1.0, object(), trigger_indices, object()
            )

    assert actual._data.tensor.device.type == device
    assert actual_dof._data.tensor.device.type == device
    assert actual_dof.dtype == np.dtype(np.int64)
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), 2, device=device, dtype=torch.int64),
    )


def test_skymax_chisq_dof_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    indices = np.array([1, 4, 7], dtype=np.int32)

    with ctx:
        corr_plus = Array(np.arange(8, dtype=np.float32))
        corr_cross = Array(np.arange(8, dtype=np.float32))
        snrv = Array(np.zeros(3, dtype=np.float32))
        trigger_indices = indices
        u_vals = Array(np.ones(3, dtype=np.float32))
        veto = object.__new__(chisq.SingleDetSkyMaxPowerChisq)
        veto.do = True
        veto.snr_threshold = 10.0

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "sky-max chi-squared metadata used NumPy repeat"
                )

            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                corr_plus,
                corr_cross,
                snrv,
                object(),
                trigger_indices,
                object(),
                object(),
                u_vals,
                0.0,
                1.0,
                1.0,
            )

    assert actual._data.tensor.device.type == device
    assert actual_dof._data.tensor.device.type == device
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), -100, device=device, dtype=torch.int64),
    )


def test_bank_chisq_dof_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    with ctx:
        statistic = Array(values)
        veto = object.__new__(bank_chisq.SingleDetBankVeto)
        veto.do = True
        veto.dof = 12
        veto.cache_overlaps = lambda *_args: []
        veto.cache_segment_snrs = lambda *_args: ([], [])

        with monkeypatch.context() as patch:
            def _reject_numpy_repeat(*_args, **_kwargs):
                raise AssertionError(
                    "bank chi-squared metadata used NumPy repeat"
                )

            patch.setattr(
                bank_chisq,
                "bank_chisq_from_filters",
                lambda *_args: statistic,
            )
            patch.setattr(chisq.numpy, "repeat", _reject_numpy_repeat)
            actual, actual_dof = veto.values(
                object(), object(), object(), object(), 1.0, object()
            )

    assert actual is statistic
    assert actual_dof._data.tensor.device.type == device
    torch.testing.assert_close(
        actual_dof._data.tensor,
        torch.full((3,), 12, device=device, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    "settings",
    (
        {"oneside": None, "twophase": True, "maxvalued": False},
        {"oneside": "left", "twophase": False, "maxvalued": True},
        {
            "oneside": "right", "twophase": True, "maxvalued": True,
            "real_autocorr": True,
        },
    ),
)
def test_autochisq_stays_on_device(
        torch_device_ctx, monkeypatch, settings):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS PyCBC arrays do not support complex dtypes")

    rng = np.random.default_rng(481)
    snr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    corr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values = (
        0.15 * rng.normal(size=64) + 0.15j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values[0] = 1
    indices = np.array([1, 17, 62], dtype=np.int32)
    settings = settings.copy()
    real_autocorr = settings.pop("real_autocorr", False)
    if real_autocorr:
        autocorr_values = autocorr_values.real.copy()
    parameters = dict(stride=2, num_points=4, **settings)
    expected_dof, expected = autochisq.autochisq_from_precomputed(
        Array(snr_values), Array(corr_values), Array(autocorr_values),
        indices, **parameters
    )

    with ctx:
        snr = Array(snr_values)
        corr = Array(corr_values)
        autocorr = Array(autocorr_values)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("auto-chi-squared copied data to host")

            def _reject_numpy_output(*_args, **_kwargs):
                raise AssertionError("auto-chi-squared allocated NumPy output")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(autochisq.np, "zeros", _reject_numpy_output)
            actual_dof, actual = autochisq.autochisq_from_precomputed(
                snr, corr, autocorr, indices, **parameters
            )

    assert actual_dof == expected_dof
    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-5, atol=2e-5
    )


def test_single_det_autochisq_keeps_indices_on_device(torch_ctx, monkeypatch):
    rng = np.random.default_rng(581)
    snr_values = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values = (
        0.15 * rng.normal(size=64) + 0.15j * rng.normal(size=64)
    ).astype(np.complex64)
    autocorr_values[0] = 1
    index_values = np.array([1, 17, 62], dtype=np.int32)
    norm = 0.75
    parameters = dict(
        stride=2, num_points=4, oneside=None, twophase=True,
        maxvalued=False,
    )
    expected_dof, expected = autochisq.autochisq_from_precomputed(
        Array(snr_values * norm), Array(snr_values * norm),
        Array(autocorr_values), index_values, **parameters
    )
    template = object()
    psd = object()

    with torch_ctx:
        snr = Array(snr_values)
        indices = Array(index_values)
        veto = autochisq.SingleDetAutoChisq(
            stride=parameters["stride"],
            num_points=parameters["num_points"],
            twophase=parameters["twophase"],
        )
        veto._autocor = Array(autocorr_values)
        veto._autocor_id = (id(template), id(psd))
        original_np_array = autochisq.np.array

        def _reject_host_transfer(_self):
            raise AssertionError("auto-chi-squared copied data to host")

        def _reject_index_copy(value, *args, **kwargs):
            if isinstance(value, Array):
                raise AssertionError("auto-chi-squared copied indices to NumPy")
            return original_np_array(value, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(autochisq, "make_frequency_series", lambda x: x)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(autochisq.np, "array", _reject_index_copy)
            actual, actual_dof = veto.values(
                snr, indices, template, psd, norm
            )

    assert actual_dof == expected_dof
    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == "cpu"
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-5, atol=2e-5
    )


def test_analytical_space_psd_public_dispatch_and_devices(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=257,
        delta_f=1e-4,
        low_freq_cutoff=3.5e-4,
        len_arm=2.4e9,
        acc_noise_level=3.2e-15,
        oms_noise_level=13e-12,
        tdi="2.0",
    )
    expected = analytical_space.analytical_psd_lisa_tdi_AE(
        **parameters
    ).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical PSD used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_psd.from_string(
                        "analytical_psd_lisa_tdi_AE", **parameters
                    )
                return
            actual = analytical_psd.from_string(
                "analytical_psd_lisa_tdi_AE", **parameters
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=1e-12, atol=0.0
    )


def test_lisa_response_psd_public_devices(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=257,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        duration=2.0,
        tdi="2.0",
    )
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    expected = analytical_space.analytical_psd_lisa_tdi_AE_confusion(
        **parameters
    ).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch LISA response model used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_space.analytical_psd_lisa_tdi_AE_confusion(
                        **parameters
                    )
                return
            actual = analytical_space.analytical_psd_lisa_tdi_AE_confusion(
                **parameters
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert _relative_l2(actual.numpy(), expected) < 5e-12


def test_flat_unity_psd_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    with ctx:
        with monkeypatch.context() as patch:
            def _reject_numpy_ones(*_args, **_kwargs):
                raise AssertionError("Torch flat_unity PSD used NumPy")

            patch.setattr(
                analytical_psd.numpy, "ones", _reject_numpy_ones
            )
            actual = analytical_psd.from_string(
                "flat_unity", 17, 0.25, 0.75
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert tensor.dtype == expected_dtype
    assert torch.equal(
        tensor,
        torch.tensor(
            [0.0, 0.0, 0.0] + [1.0] * 14,
            dtype=expected_dtype,
            device=device,
        ),
    )


def test_ground_detector_fit_family_torch_matches_lalsimulation(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=1031,
        delta_f=0.375,
        low_freq_cutoff=9.1,
    )

    if device == "mps":
        with ctx:
            for model_name in _GROUND_FIT_TORCH_100_HZ_PINS:
                with pytest.raises(
                    TypeError,
                    match="MPS backend only supports",
                ):
                    analytical_psd.from_string(
                        model_name,
                        **parameters,
                    )
        return

    expected = {
        model_name: analytical_psd.from_string(
            model_name,
            **parameters,
        ).numpy()
        for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
    }

    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch ground PSD fit called LALSimulation")

    with ctx:
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical_psd.lal,
                "CreateREAL8FrequencySeries",
                reject_lal_path,
            )
            patch.setattr(
                analytical_psd.lalsimulation,
                "SimNoisePSD",
                reject_lal_path,
            )
            actual = {
                model_name: analytical_psd.from_string(
                    model_name,
                    **parameters,
                )
                for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
            }
            wrapper_actual = analytical_psd.Virgo(**parameters)

    for model_name, result in actual.items():
        tensor = result._data.tensor
        assert tensor.device.type == device
        assert tensor.dtype == torch.float64
        assert result.epoch == lal.LIGOTimeGPS(0)
        assert result.delta_f == parameters["delta_f"]
        assert tensor[0] == 0
        assert tensor[-1] == 0
        np.testing.assert_allclose(
            tensor.detach().cpu().numpy(),
            expected[model_name],
            rtol=2e-14,
            atol=0.0,
        )
    torch.testing.assert_close(
        wrapper_actual._data.tensor,
        actual["Virgo"]._data.tensor,
    )


def test_ground_detector_fit_family_layout_cutoff_and_source_pins(
        torch_ctx):
    with torch_ctx:
        actual = {
            model_name: analytical_psd.from_string(
                model_name,
                length=1002,
                delta_f=1.0,
                low_freq_cutoff=10.9,
                ignored_lalsimulation_keyword=True,
            )._data.tensor
            for model_name in _GROUND_FIT_TORCH_100_HZ_PINS
        }
        one_bin = analytical_psd.Virgo(1, 1.0, 0.0)
        two_bins = analytical_psd.Virgo(2, 1.0, 0.0)
        negative_cutoff = analytical_psd.Virgo(8, 1.0, -1.0)

    for values in actual.values():
        assert torch.count_nonzero(values[:10]) == 0
        assert values[10] != 0
        assert values[-1] == 0
    torch.testing.assert_close(
        torch.stack([values[100] for values in actual.values()]),
        torch.tensor(
            list(_GROUND_FIT_TORCH_100_HZ_PINS.values()),
            dtype=torch.float64,
        ),
        rtol=2e-15,
        atol=0.0,
    )
    assert torch.count_nonzero(one_bin._data.tensor) == 0
    assert torch.count_nonzero(two_bins._data.tensor) == 0
    assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_psd_array_reader_interpolates_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    frequencies = np.array([0.5, 0.9, 1.8, 3.2, 5.0])
    noise = np.array([4.0, 2.0, 0.75, 1.5, 3.0])
    parameters = dict(
        length=25,
        delta_f=0.25,
        low_freq_cutoff=0.75,
    )
    expected = psd_read.from_numpy_arrays(
        frequencies,
        noise,
        **parameters,
    ).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD reader used NumPy/SciPy")

            patch.setattr(
                psd_read.scipy.interpolate,
                "interp1d",
                _reject_host_path,
            )
            patch.setattr(psd_read.numpy, "zeros", _reject_host_path)
            patch.setattr(psd_read.numpy, "arange", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="requires float64"):
                    psd_read.from_numpy_arrays(
                        frequencies,
                        noise,
                        **parameters,
                    )
                return
            actual = psd_read.from_numpy_arrays(
                frequencies,
                noise,
                **parameters,
            )

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert len(actual) == len(expected) == 21
    assert torch.count_nonzero(tensor[:3]) == 0
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=0)


def test_psd_text_reader_parses_host_and_interpolates_on_device(
        torch_ctx, monkeypatch, tmp_path):
    frequencies = np.array([0.5, 0.9, 1.8, 3.2, 5.0])
    noise = np.array([4.0, 2.0, 0.75, 1.5, 3.0])
    filename = tmp_path / "asd.txt"
    np.savetxt(filename, np.column_stack((frequencies, np.sqrt(noise))))
    parameters = dict(
        filename=filename,
        length=17,
        delta_f=0.25,
        low_freq_cutoff=0.75,
    )
    expected = psd_read.from_txt(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD reader used NumPy/SciPy")

            patch.setattr(
                psd_read.scipy.interpolate,
                "interp1d",
                _reject_host_path,
            )
            patch.setattr(psd_read.numpy, "zeros", _reject_host_path)
            patch.setattr(psd_read.numpy, "arange", _reject_host_path)
            actual = psd_read.from_txt(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=0)


def test_psd_interpolation_stays_on_device_and_clamps_band_edge(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([4.0, 1.0, 9.0, 16.0])
    parameters = dict(delta_f=0.5, length=8)
    expected = psd_estimate.interpolate(
        FrequencySeries(values, delta_f=0.75),
        **parameters,
    ).numpy()

    dtype = torch.float32 if device == "mps" else torch.float64
    with ctx:
        source = FrequencySeries(
            TorchArrayData(torch.tensor(values, dtype=dtype, device=device)),
            delta_f=0.75,
            copy=False,
        )
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch PSD interpolation used the host")

            patch.setattr(psd_estimate.numpy, "interp", _reject_host_path)
            patch.setattr(torch.Tensor, "cpu", _reject_host_path)
            patch.setattr(torch.Tensor, "numpy", _reject_host_path)
            actual = psd_estimate.interpolate(source, **parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == dtype
    assert actual.delta_f == parameters["delta_f"]
    assert tensor[-3:].tolist() == [16.0, 16.0, 16.0]
    np.testing.assert_allclose(
        tensor.detach().tolist(),
        expected,
        rtol=2e-6 if dtype == torch.float32 else 1e-12,
        atol=0,
    )


@pytest.mark.parametrize(
    "model_name",
    (
        "analytical_csd_lisa_tdi_XY",
        "analytical_psd_taiji_tdi_T",
        "analytical_psd_taiji_confusion_noise",
        "analytical_psd_taiji_tdi_AE_confusion",
    ),
)
def test_analytical_space_psd_rejects_invalid_tdi(torch_ctx, model_name):
    with torch_ctx:
        with pytest.raises(ValueError, match="currently only for 1.5 or 2.0"):
            getattr(analytical_space, model_name)(
                17, 1e-4, 3.5e-4, tdi="3.0"
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


def test_psd_welch_builds_work_buffers_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(8129)
    seg_len = 32
    seg_stride = 16
    values = rng.standard_normal(64).astype(np.float32)
    expected = welch(
        TimeSeries(values, delta_t=1 / 256.0),
        seg_len,
        seg_stride=seg_stride,
        avg_method="mean",
    )

    with ctx:
        timeseries = TimeSeries(values, delta_t=1 / 256.0)

        def reject_numpy_window(*args, **kwargs):
            raise AssertionError("Welch created its Hann window with NumPy")

        def reject_numpy_zeros(*args, **kwargs):
            raise AssertionError("Welch created its FFT buffer with NumPy")

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("Welch copied Torch data back to the host")

        with monkeypatch.context() as patch:
            patch.setattr(np, "hanning", reject_numpy_window)
            patch.setattr(psd_estimate.numpy, "zeros", reject_numpy_zeros)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            actual = welch(
                timeseries,
                seg_len,
                seg_stride=seg_stride,
                avg_method="mean",
            )

        assert actual._data.tensor.device.type == device

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=2e-5, atol=1e-7
    )


@pytest.mark.parametrize("batch_segments", [1, 3, 4, 9])
def test_psd_welch_respects_temporary_budget(
        torch_ctx, monkeypatch, batch_segments):
    rng = np.random.default_rng(9137)
    seg_len = 32
    seg_stride = 16
    num_segments = 9
    values = rng.standard_normal(
        seg_len + (num_segments - 1) * seg_stride
    ).astype(np.float32)
    expected = welch(
        TimeSeries(values, delta_t=1 / 256.0),
        seg_len,
        seg_stride=seg_stride,
        avg_method="median",
    )

    real_bytes = values.dtype.itemsize
    bytes_per_segment = (
        seg_len * real_bytes
        + (seg_len // 2 + 1) * 2 * real_bytes
    )
    fft_batch_sizes = []
    original_rfft = torch.fft.rfft

    def tracked_rfft(values, *args, **kwargs):
        fft_batch_sizes.append(values.shape[0])
        return original_rfft(values, *args, **kwargs)

    with torch_ctx:
        with monkeypatch.context() as patch:
            patch.setattr(
                psd_estimate,
                "_TORCH_WELCH_TEMPORARY_BYTES",
                batch_segments * bytes_per_segment,
            )
            patch.setattr(torch.fft, "rfft", tracked_rfft)
            actual = welch(
                TimeSeries(values, delta_t=1 / 256.0),
                seg_len,
                seg_stride=seg_stride,
                avg_method="median",
            )

    full_batches, remainder = divmod(num_segments, batch_segments)
    expected_batch_sizes = [batch_segments] * full_batches
    if remainder:
        expected_batch_sizes.append(remainder)
    assert fft_batch_sizes == expected_batch_sizes
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=2e-5, atol=1e-7
    )


@pytest.mark.parametrize("psd_duration", [2, 8])
def test_psd_variation_bandpass_response_stays_on_device(
        torch_device_ctx, monkeypatch, psd_duration):
    ctx, device = torch_device_ctx
    sample_rate = 64
    low_freq = 5
    high_freq = 20

    coefficients = variation.sig.firwin(
        4 * sample_rate, [low_freq, high_freq], pass_zero=False,
        window="hann", fs=sample_rate
    )
    coefficients.resize(psd_duration * sample_rate)
    expected = np.abs(np.fft.rfft(coefficients)).astype(np.float32)

    with ctx:
        reference = FrequencySeries(
            np.ones(expected.size, dtype=np.float32), delta_f=0.25
        )

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("bandpass response used its host path")

        with monkeypatch.context() as patch:
            patch.setattr(variation.sig, "firwin", reject_host_path)
            patch.setattr(variation, "rfft", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            actual = variation._torch_bandpass_response(
                reference, sample_rate, low_freq, high_freq, psd_duration
            )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == device
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=5e-5, atol=5e-6
    )


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
            patch.setattr(variation.sig, "firwin", _reject_scipy)
            patch.setattr(variation, "rfft", _reject_scipy)
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
    assert isinstance(actual_values, Array)
    assert actual_values._data.tensor.device.type == device
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
        actual_values._data.tensor.detach().cpu().numpy(),
        expected_values, rtol=5e-5, atol=5e-6,
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
            patch.setattr(variation.sig, "firwin", _reject_scipy)
            patch.setattr(variation, "rfft", _reject_scipy)
            patch.setattr(variation.sig, "fftconvolve", _reject_scipy)
            patch.setattr(variation, "interp1d", _reject_scipy)
            actual = variation.calc_filt_psd_variation(
                torch_strain, *parameters
            )
            actual_values = variation.find_trigger_value(
                actual, trigger_indices, 100, sample_rate
            )

    assert actual._data.tensor.device.type == device
    assert isinstance(actual_values, Array)
    assert actual_values._data.tensor.device.type == device
    assert actual.dtype == np.dtype(np.float32)
    assert actual.start_time == expected.start_time
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_values._data.tensor.detach().cpu().numpy(),
        expected_values, rtol=1e-4, atol=1e-5,
    )


def test_psd_variation_interpolation_preserves_torch_autograd(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = torch.tensor(
            [2.0, 4.0, 8.0, 16.0], device=device, dtype=dtype,
            requires_grad=True,
        )
        positions = torch.tensor(
            [-1.0, 0.5, 2.25, 4.0], device=device, dtype=dtype,
            requires_grad=True,
        )
        series = TimeSeries(
            TorchArrayData(values), delta_t=1.0, copy=False
        )

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("PSD interpolation copied data to the host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            actual = variation._torch_interpolate_positions(
                series, positions
            )

        result = actual._data.tensor
        assert result.device.type == device
        assert result.requires_grad
        result.sum().backward()

    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        result.detach().cpu(),
        torch.tensor([1.0, 3.0, 10.0, 1.0], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        values.grad.detach().cpu(),
        torch.tensor([0.5, 0.5, 0.75, 0.25], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        positions.grad.detach().cpu(),
        torch.tensor([0.0, 2.0, 8.0, 0.0], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )


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

            def _reject_host_event_stage(*_args, **_kwargs):
                raise AssertionError(
                    "Glitch detection used a host threshold/cluster stage"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(pycbc.events, "threshold_only",
                          _reject_host_event_stage)
            patch.setattr(pycbc.events, "findchirp_cluster_over_window",
                          _reject_host_event_stage)
            actual = detect_loud_glitches(torch_data, **parameters)

    assert actual == expected


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
    fft_input_dtypes = []
    transform_name = (
        "fft" if np.issubdtype(dtype, np.complexfloating) else "rfft"
    )
    original_transform = getattr(torch.fft, transform_name)

    def recording_transform(values, *args, **kwargs):
        fft_input_dtypes.append(values.dtype)
        return original_transform(values, *args, **kwargs)

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
            patch.setattr(torch.fft, transform_name, recording_transform)
            patch.setattr(
                resample, "_TORCH_LFILTER_TARGET_BLOCK_SIZE", block_size
            )
            actual = resample.lfilter(filter_coefficients, input_series)

    assert isinstance(actual._data.tensor, torch.Tensor)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time
    assert fft_input_dtypes
    assert all(
        call_dtype == resample._torch_lfilter_work_dtype(
            input_series._data.tensor
        )
        for call_dtype in fft_input_dtypes
    )

    actual_data = actual._data.tensor.detach().cpu().numpy()
    input_data = input_series._data.tensor.detach().cpu().numpy()
    rtol, atol = ((5e-5, 5e-5) if single_precision
                  else (1e-11, 1e-11))
    np.testing.assert_allclose(actual_data, expected, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(input_data, data)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("length,num_taps", ((63, 7), (4096, 129)))
def test_fir_zero_filter_torch_matches_scipy_without_host_transfer(
        torch_device_ctx, monkeypatch, dtype, length, num_taps):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype != np.float32:
        pytest.skip("Torch MPS only supports float32 PyCBC arrays")

    rng = np.random.default_rng(12043)
    single_precision = dtype in (np.float32, np.complex64)
    real_dtype = np.float32 if single_precision else np.float64
    coefficients = scipy.signal.windows.hann(num_taps).astype(real_dtype)
    coefficients /= coefficients.sum()
    data = rng.normal(size=length)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.normal(size=length)
    data = data.astype(dtype)
    expected = scipy.signal.lfilter(coefficients, 1.0, data).astype(dtype)
    expected[:(num_taps // 2) * 2] = 0
    expected = np.roll(expected, -num_taps // 2)

    with ctx:
        input_series = TimeSeries(data, delta_t=1 / 2048, epoch=456)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "fir_zero_filter copied Torch data to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                resample, "_TORCH_LFILTER_TARGET_BLOCK_SIZE", 512
            )
            actual = resample.fir_zero_filter(
                coefficients, input_series
            )

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


def test_zeros_pinned_and_to_cuda_async_pipeline():
    from pycbc.types.array_torch import zeros_pinned, to_cuda_async, TorchArrayData

    pinned = zeros_pinned(64, dtype=np.float32)
    assert isinstance(pinned, TorchArrayData)
    assert pinned.shape == (64,)
    assert pinned.dtype == np.dtype(np.float32)
    if torch.cuda.is_available():
        assert pinned.tensor.is_pinned()

    transferred = to_cuda_async(pinned)
    assert isinstance(transferred, TorchArrayData)
    if torch.cuda.is_available():
        assert transferred.tensor.device.type == "cuda"

    raw_transferred = to_cuda_async(pinned.tensor)
    assert isinstance(raw_transferred, torch.Tensor)
    if torch.cuda.is_available():
        assert raw_transferred.device.type == "cuda"


def test_live_batch_cuda_graph_flag_from_environment(monkeypatch):
    template = FrequencySeries(np.ones(33, dtype=np.complex64), delta_f=0.25)
    template.id = 1
    template.params = np.array(
        [(10.0,)], dtype=[("mass1", np.float32)]
    )[0]

    monkeypatch.delenv("PYCBC_ENABLE_CUDA_GRAPHS", raising=False)
    batch = matchedfilter.LiveBatchMatchedFilter(
        [template],
        snr_threshold=5.0,
        chisq_bins=None,
        sg_chisq=None,
        maxelements=64,
    )
    assert batch.enable_cuda_graphs is False
    assert batch._cuda_graphs == {}

    monkeypatch.setenv("PYCBC_ENABLE_CUDA_GRAPHS", "1")
    batch = matchedfilter.LiveBatchMatchedFilter(
        [template],
        snr_threshold=5.0,
        chisq_bins=None,
        sg_chisq=None,
        maxelements=64,
    )
    assert batch.enable_cuda_graphs is True


@pytest.mark.parametrize("enable_cuda_graphs", (True, False))
def test_live_batch_matched_filter_cuda_graphs_pipeline(torch_device_ctx, enable_cuda_graphs):
    ctx, device = torch_device_ctx
    size = 64
    fsize = size // 2 + 1
    rng = np.random.default_rng(999)
    template_values = (
        rng.normal(size=(2, fsize)) + 1j * rng.normal(size=(2, fsize))
    ).astype(np.complex64)
    data_values = (
        rng.normal(size=fsize) + 1j * rng.normal(size=fsize)
    ).astype(np.complex64)

    with ctx:
        templates = []
        for index, values in enumerate(template_values):
            template = FrequencySeries(values, delta_f=1.0 / size)
            template.id = 100 + index
            template.params = np.array(
                [(15.0 + index,)], dtype=[("mass1", np.float32)]
            )[0]
            template.sigmasq = lambda _psd: 1.0
            templates.append(template)

        batch = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=0.0,
            chisq_bins=None,
            sg_chisq=types.SimpleNamespace(values=lambda *args: None),
            maxelements=len(templates) * size,
            enable_cuda_graphs=enable_cuda_graphs,
        )
        batch.power_chisq.values = lambda *args: (
            np.array([0.0], dtype=np.float32),
            np.array([1], dtype=np.uint32),
        )
        assert batch.enable_cuda_graphs is enable_cuda_graphs

        stilde = FrequencySeries(data_values, delta_f=1.0 / size)
        stilde.psd = FrequencySeries(np.ones(fsize, dtype=np.float32), delta_f=1.0 / size)

        batch.set_data(types.SimpleNamespace(
            overwhitened_data=lambda _delta_f: stilde,
            trim_padding=0,
            blocksize=size,
            sample_rate=1,
            start_time=100.0,
        ))

        res1 = batch.process_data(types.SimpleNamespace(
            overwhitened_data=lambda _delta_f: stilde,
            trim_padding=0,
            blocksize=size,
            sample_rate=1,
            start_time=100.0,
        ))
        assert "snr" in res1
        assert len(res1["snr"]) == 2

        if device == "cuda" and enable_cuda_graphs and torch.cuda.is_available():
            assert 0 in batch._cuda_graphs
            entry = batch._cuda_graphs[0]
            replays = entry.get("replays", getattr(entry["graph"], "replay_count", 0))
            assert replays == 0

            # Second run replays graph
            res2 = batch.process_data(types.SimpleNamespace(
                overwhitened_data=lambda _delta_f: stilde,
                trim_padding=0,
                blocksize=size,
                sample_rate=1,
                start_time=100.0,
            ))
            replays = entry.get("replays", getattr(entry["graph"], "replay_count", 0))
            assert replays == 1
            np.testing.assert_allclose(res1["snr"], res2["snr"])
