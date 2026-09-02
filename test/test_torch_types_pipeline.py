import importlib.util

import operator

import subprocess

import sys

import types

import warnings

from pathlib import Path

import lal

import numpy as np

import pytest

import scipy.interpolate

import scipy.signal

import scipy.special

import scipy.stats

import lalsimulation

from igwn_ligolw import lsctables

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

import pycbc

from pycbc import boundaries

from pycbc import cosmology

from pycbc import conversions

from pycbc import coordinates

from pycbc import distributions

from pycbc import events

from pycbc import pnutils

from pycbc import scheme

from pycbc import transforms

from pycbc.coordinates import base as coordinate_base

from pycbc.coordinates import space as coordinate_space

from pycbc.detector import Detector, single_arm_frequency_response

from pycbc.detector import space as space_detector

from pycbc.detector.space import check_signal_times

from pycbc.distributions import arbitrary as distribution_arbitrary

from pycbc.distributions import external as distribution_external

from pycbc.distributions import fixedsamples as distribution_fixedsamples

from pycbc.distributions import gaussian as distribution_gaussian

from pycbc.distributions import joint as distribution_joint

from pycbc.distributions import angular as distribution_angular

from pycbc.distributions import mass as distribution_mass

from pycbc.distributions import qnm as distribution_qnm

from pycbc.distributions import uniform_log as distribution_uniform_log

from pycbc.events import (
    coherent,
    coinc,
    coinc_rate,
    cuts,
    eventmgr,
    ranking,
    significance,
    threshold_torch,
    trigger_fits,
    veto,
)

from pycbc.events import single as event_single

from pycbc.events import stat as event_stat

from pycbc.filter import autocorrelation, matchedfilter, resample, zpk

from pycbc.frame.frame import StatusBuffer, iDQBuffer

from pycbc.inject.inject import SGBurstInjectionSet, _InjectionAdder

from pycbc.inject.injfilterrejector import InjFilterRejector

from pycbc.inference import burn_in as inference_burn_in

from pycbc.inference import evidence as inference_evidence

from pycbc.inference import entropy as inference_entropy

from pycbc.inference import geweke as inference_geweke

from pycbc.inference import gelman_rubin as inference_gelman_rubin

from pycbc.inference.models import analytic as inference_analytic

from pycbc.inference.models import base as inference_model_base

from pycbc.inference.models import base_data as inference_model_base_data

from pycbc.inference.models import brute_marg as inference_brute_marg

from pycbc.inference.models import gaussian_noise as inference_gaussian_noise

from pycbc.inference.models import gated_gaussian_noise

from pycbc.inference.models import hierarchical as inference_hierarchical

from pycbc.inference.models import marginalized_gaussian_noise

from pycbc.inference.models import relbin as inference_relbin

from pycbc.inference.models import single_template

from pycbc.inference.sampler import refine as inference_refine

from pycbc.noise import gaussian, reproduceable

from pycbc.neutron_stars import eos_utils as neutron_eos

from pycbc.neutron_stars import pg_isso_solver as pg_isso

from pycbc.population import (
    fgmc_functions,
    live_pastro,
    population_models,
    rates_functions,
    scale_injections,
)

from pycbc.population.fgmc_laguerre import count_posterior

from pycbc.psd import analytical as analytical_psd

from pycbc.psd import analytical_space

from pycbc.psd import estimate as psd_estimate

from pycbc.psd import inverse_spectrum_truncation, variation, welch

from pycbc.psd import read as psd_read

from pycbc.strain import gate as strain_gate

from pycbc.strain import calibration, lines as strain_lines, recalibrate

from pycbc.strain.strain import (
    StrainBuffer,
    _hann_window_for_series,
    _linear_tapers_for_series,
    detect_loud_glitches,
    gate_data,
)

from pycbc.tmpltbank import coord_utils as tmpltbank_coord_utils

from pycbc.tmpltbank import lambda_mapping as tmpltbank_lambda_mapping

from pycbc.tmpltbank.calc_moments import get_moments

from pycbc.tmpltbank.option_utils import metricParameters

from pycbc.types import Array, FrequencySeries, TimeSeries

import pycbc.types.array as array_module

import pycbc.types.array_torch as array_torch_module

import pycbc.types.timeseries as timeseries_module

from pycbc.types.array_torch import TorchArrayData

import pycbc.vetoes.chisq as chisq

import pycbc.vetoes.autochisq as autochisq

import pycbc.vetoes.bank_chisq as bank_chisq

import pycbc.vetoes.sgchisq as sgchisq

from pycbc.waveform import ringdown, sinegauss, utils as waveform_utils

from pycbc.waveform.compress import fd_decompress, spa_compression

from pycbc.waveform import waveform as waveform_module


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


_INFERENCE_RELBIN_TORCH = _load_inference_model_module("relbin_torch")


_INFERENCE_TOOLS = _load_inference_model_module("tools")


def _native_scalar_single_template_model(
        dtype=np.complex128, detectors=("H1",)):
    """Build the minimum real-detector model needed by the scalar path."""
    epoch = 1126259462.0
    delta_t = 1.0 / 128.0
    samples = 1025
    indices = np.arange(samples, dtype=np.float64)
    model = object.__new__(single_template.SingleTemplate)
    model._current_params = {
        "inclination": np.float64(0.71),
        "polarization": np.float64(0.39),
        "coa_phase": np.float64(0.23),
        "distance": np.float64(1.8),
        "ra": np.float64(1.0),
        "dec": np.float64(-0.4),
        "tc": np.float64(epoch + 4.003),
    }
    scales = {"H1": 1.0, "L1": 1.3}
    model.sh = {
        ifo: TimeSeries(
            (
                (1.0 + scale * 0.002 * indices)
                * np.exp(scale * 0.017j * indices)
            ).astype(dtype),
            delta_t=delta_t,
            epoch=epoch,
        )
        for ifo, scale in ((ifo, scales[ifo]) for ifo in detectors)
    }
    norms = {"H1": 1.7, "L1": 2.1}
    model.hh = {ifo: np.float64(norms[ifo]) for ifo in detectors}
    model.snr = {}
    model.det = {ifo: Detector(ifo) for ifo in model.sh}
    model.dts = {}
    model.htfs = {}
    model._sh_storage_is_host = False
    model.marginalize_phase = False
    model.marginalize_distance = False
    model.distance_marginalization = False
    model.distance_interpolator = None
    model.marginalize_vector_params = {}
    model.vsamples = 1
    model.marginalize_vector_weights = 0.0
    model.reconstruct_phase = False
    model.reconstruct_distance = False
    model.reconstruct_vector = False
    model.snr_draw = lambda **_kwargs: None
    return model


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
    epoch = (
        float(plus.epoch.gpsSeconds + 1e-9 * plus.epoch.gpsNanoSeconds)
        if hasattr(plus.epoch, "gpsSeconds")
        else float(plus.epoch)
    )
    return (
        plus.data.data.copy(),
        cross.data.data.copy(),
        epoch,
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


def _injection_filter_rejector():
    rejector = object.__new__(InjFilterRejector)
    rejector.enabled = True
    rejector.match_threshold = 0.8
    rejector.coarsematch_deltaf = 1.0
    rejector.coarsematch_fmax = 64.0
    rejector.short_injections = {}
    rejector._short_psd_storage = {}
    return rejector


def _stub_live_batch_veto_calculators(
        chisq_values, dof_values, sg_values, torch_backed):
    """Build exact standard calculators with deterministic one-point output."""
    state = {"index": 0}
    power = object.__new__(chisq.SingleDetPowerChisq)
    sg_veto = object.__new__(sgchisq.SingleDetSGChisq)

    def wrap(value, dtype):
        values = np.array([value], dtype=dtype)
        return Array(values) if torch_backed else values

    def power_values(*_args):
        index = state["index"]
        return (
            wrap(chisq_values[index], np.float32),
            wrap(dof_values[index], np.int64),
        )

    def sg_veto_values(*_args):
        index = state["index"]
        state["index"] += 1
        value = sg_values[index]
        return None if value is None else wrap(value, np.float32)

    power.values = power_values
    sg_veto.values = sg_veto_values
    return power, sg_veto


def _stub_live_batch_veto_inputs(snr_values):
    results = {
        "snr": np.asarray(snr_values, dtype=np.float32),
        "template_id": np.arange(len(snr_values), dtype=np.uint64),
    }
    veto_info = []
    for index in range(len(snr_values)):
        template = types.SimpleNamespace(cout=object())
        strain = types.SimpleNamespace(psd=object())
        veto_info.append((
            np.array([1 + 0j], dtype=np.complex64),
            1.0,
            index,
            template,
            strain,
        ))
    return results, veto_info


def _stub_live_batch_veto_filter(power, sg_veto, threshold=None):
    batch = object.__new__(matchedfilter.LiveBatchMatchedFilter)
    batch.power_chisq = power
    batch.sg_chisq = sg_veto
    batch.newsnr_threshold = threshold
    return batch


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


_DIRECT_SPACE_PSD_MODELS = tuple(
    f"analytical_psd_{detector}_tdi_{channel}"
    for detector in ("lisa", "tianqin", "taiji")
    for channel in ("XYZ", "AE", "T")
)


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


def test_optional_uint32_dtype_is_feature_detected():
    torch_uint32 = getattr(torch, "uint32", None)
    assert array_torch_module._TORCH_UINT32 is torch_uint32
    assert (
        np.dtype(np.uint32) in array_torch_module._NUMPY_TO_TORCH
    ) == (torch_uint32 is not None)


def test_conversion_broadcasting_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = torch.tensor(
            [[1.0], [2.0]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        scalar = torch.tensor(4.0, device=device, dtype=dtype)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("conversion broadcasting left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "item", reject_host_or_numpy)
            patch.setattr(
                conversions.numpy, "broadcast_arrays", reject_host_or_numpy
            )
            left, right, input_is_array = conversions.ensurearray(
                values, np.array([10.0, 20.0, 30.0])
            )
            scalar_value, offset, scalar_is_array = (
                conversions.ensurearray(scalar, 2.0)
            )
            preserved = conversions.formatreturn(
                scalar_value + offset, scalar_is_array
            )

        assert input_is_array
        assert scalar_is_array
        assert left.shape == right.shape == (2, 3)
        assert all(
            value.device.type == device
            for value in (left, right, scalar_value, offset, preserved)
        )
        assert all(
            value.dtype == dtype
            for value in (left, right, scalar_value, offset, preserved)
        )
        assert preserved.ndim == 0
        (left * right).sum().backward()
        assert values.grad is not None
        assert values.grad.device.type == device
        assert torch.equal(
            values.grad, torch.full_like(values, 60.0)
        )

    left, right, input_is_array = conversions.ensurearray(2.0, 3.0)
    assert not input_is_array
    assert conversions.formatreturn(left + right, input_is_array) == 5.0


def test_inverse_mass_conversions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    expected_mass1 = np.array([1.4, 10.0, 80.0, 1.2])
    expected_mass2 = np.array([1.2, 1.4, 20.0, 0.7])
    expected_mchirp = (
        (expected_mass1 * expected_mass2) ** (3 / 5)
        / (expected_mass1 + expected_mass2) ** (1 / 5)
    )
    expected_eta = (
        expected_mass1 * expected_mass2
        / (expected_mass1 + expected_mass2) ** 2
    )

    with ctx:
        mass1 = torch.tensor(
            expected_mass1, device=device, dtype=dtype, requires_grad=True
        )
        mass2 = torch.tensor(
            expected_mass2, device=device, dtype=dtype, requires_grad=True
        )
        mchirp = torch.tensor(
            expected_mchirp, device=device, dtype=dtype, requires_grad=True
        )
        eta = torch.tensor(
            expected_eta, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("inverse mass conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "roots", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            from_mchirp = conversions.mass2_from_mchirp_mass1(
                mchirp, mass1
            )
            from_primary = conversions.mass2_from_mass1_eta(mass1, eta)
            from_secondary = conversions.mass1_from_mass2_eta(mass2, eta)
            three_real_roots = conversions.mass2_from_mchirp_mass1(
                torch.tensor(
                    [10.0], device=device, dtype=dtype, requires_grad=True
                ),
                1.2,
            )

            results = (
                from_mchirp,
                from_primary,
                from_secondary,
                three_real_roots,
            )
            assert all(isinstance(value, torch.Tensor) for value in results)
            assert all(value.device.type == device for value in results)
            assert all(value.dtype == dtype for value in results)

            sum(value.sum() for value in results).backward()
            assert all(
                value.grad is not None and torch.all(torch.isfinite(value.grad))
                for value in (mass1, mass2, mchirp, eta)
            )

        actual = [value.detach().cpu().numpy() for value in results]

    tolerance = 3e-5 if dtype == torch.float32 else 5e-12
    np.testing.assert_allclose(
        actual[0], expected_mass2, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual[1], expected_mass2, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual[2], expected_mass1, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual[3], [241.16038223333564],
        rtol=tolerance, atol=tolerance,
    )


def test_inverse_mass_conversion_tensor_edge_roots(torch_ctx, monkeypatch):
    with torch_ctx:
        known_mass = torch.full(
            (4,), 10.0, dtype=torch.float64, requires_grad=True
        )
        eta = torch.tensor(
            [0.24, 0.25, 0.3, 0.0],
            dtype=torch.float64,
            requires_grad=True,
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("inverse mass conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "roots", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            smaller = conversions.mass_from_knownmass_eta(
                known_mass, eta, force_real=False
            )
            larger = conversions.mass_from_knownmass_eta(
                known_mass, eta,
                known_is_secondary=True,
                force_real=False,
            )
            real_smaller = conversions.mass_from_knownmass_eta(
                known_mass, eta
            )
            real_larger = conversions.mass_from_knownmass_eta(
                known_mass, eta, known_is_secondary=True
            )

            assert smaller.dtype == torch.complex128
            assert larger.dtype == torch.complex128
            loss = (smaller.real + smaller.imag
                    + larger.real + larger.imag).sum()
            loss.backward()
            assert torch.all(torch.isfinite(known_mass.grad))
            # The two quadratic roots coalesce at eta=0.25, where their
            # individual derivatives are mathematically singular.
            assert torch.all(torch.isfinite(eta.grad[[0, 2, 3]]))

        actual_smaller = smaller.detach().cpu().numpy()
        actual_larger = larger.detach().cpu().numpy()
        actual_real_smaller = real_smaller.detach().cpu().numpy()
        actual_real_larger = real_larger.detach().cpu().numpy()

    expected_smaller = np.array([
        6.666666666666667,
        10.0,
        6.666666666666667 - 7.453559924999299j,
        0.0,
    ])
    expected_larger = np.array([
        15.0,
        10.0,
        6.666666666666667 + 7.453559924999299j,
        0.0,
    ])
    np.testing.assert_allclose(actual_smaller, expected_smaller, rtol=2e-14)
    np.testing.assert_allclose(actual_larger, expected_larger, rtol=2e-14)
    np.testing.assert_allclose(
        actual_real_smaller, expected_smaller.real, rtol=2e-14
    )
    np.testing.assert_allclose(
        actual_real_larger, expected_larger.real, rtol=2e-14
    )


def test_distance_to_redshift_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    converter = cosmology.DistToZ(numpoints=1000)
    converter.setup_interpolant()
    distances = np.array([1.0, 100.0, 1000.0, 10000.0, 1.0e6])
    expected = converter(distances)
    monkeypatch.setitem(
        cosmology._d2zs, cosmology.DEFAULT_COSMOLOGY, converter)

    def reject_host_or_numpy(*args, **kwargs):
        raise AssertionError("distance-to-redshift evaluation left Torch")

    with ctx:
        tensor = torch.tensor(
            distances, device=device, dtype=dtype, requires_grad=True)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(
                scipy.interpolate.interp1d, "__call__", reject_host_or_numpy)
            actual = transforms.DistanceToRedshift().transform(
                {"distance": tensor})["redshift"]

        assert isinstance(actual, torch.Tensor)
        assert actual.device == tensor.device
        assert actual.dtype == tensor.dtype
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, device=device, dtype=dtype),
            rtol=2e-6 if dtype == torch.float32 else 1e-12,
            atol=1e-7 if dtype == torch.float32 else 0.0,
        )
        actual.sum().backward()
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


def test_tov_lambda_interpolation_stays_on_torch_device(
        torch_device_ctx, monkeypatch, tmp_path):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 1e-12
    mass_knots = np.array([1.0, 1.2, 1.6, 2.0])
    lambda_knots = np.array([1000.0, 600.0, 200.0, 50.0])
    table_path = tmp_path / "mass-lambda.txt"
    np.savetxt(table_path, np.column_stack((mass_knots, lambda_knots)))

    direct_transform = transforms.LambdaFromTOVFile(
        mass_param="mass1",
        lambda_param="lambda1",
        mass_lambda_file=table_path,
        redshift_mass=False,
    )
    shifted_transform = transforms.LambdaFromTOVFile(
        mass_param="mass1",
        lambda_param="lambda1",
        mass_lambda_file=table_path,
        redshift_mass=True,
    )
    masses = np.array([0.8, 1.1, 1.4, 2.0, 3.0])
    distances = np.array([1.0, 50.0, 100.0, 500.0, 1000.0])
    expected_direct = np.interp(
        masses, mass_knots, lambda_knots, right=0.0
    )
    converter = cosmology.DistToZ(numpoints=1000)
    converter.setup_interpolant()
    redshifts = converter(distances)
    expected_shifted = np.interp(
        masses / (1.0 + redshifts),
        mass_knots,
        lambda_knots,
        right=0.0,
    )
    monkeypatch.setitem(
        cosmology._d2zs, cosmology.DEFAULT_COSMOLOGY, converter
    )

    def reject_host_or_numpy(*_args, **_kwargs):
        raise AssertionError("TOV interpolation left Torch")

    with ctx:
        mass_tensor = torch.tensor(
            masses, device=device, dtype=dtype, requires_grad=True
        )
        distance_tensor = torch.tensor(
            distances, device=device, dtype=dtype, requires_grad=True
        )
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "interp", reject_host_or_numpy)
            patch.setattr(transforms.numpy, "interp", reject_host_or_numpy)
            patch.setattr(
                scipy.interpolate.interp1d,
                "__call__",
                reject_host_or_numpy,
            )
            actual_direct = direct_transform.transform(
                {"mass1": mass_tensor}
            )["lambda1"]
            actual_shifted = shifted_transform.transform(
                {"mass1": mass_tensor, "distance": distance_tensor}
            )["lambda1"]
            actual_file = conversions.lambda_from_mass_tov_file(
                mass_tensor, table_path, distance=distance_tensor
            )

        expected_direct_tensor = torch.as_tensor(
            expected_direct, device=device, dtype=dtype
        )
        expected_shifted_tensor = torch.as_tensor(
            expected_shifted, device=device, dtype=dtype
        )
        torch.testing.assert_close(
            actual_direct,
            expected_direct_tensor,
            rtol=tolerance,
            atol=tolerance,
        )
        torch.testing.assert_close(
            actual_shifted,
            expected_shifted_tensor,
            rtol=2 * tolerance,
            atol=2 * tolerance,
        )
        expected_file = torch.as_tensor(
            np.interp(
                masses / (1.0 + redshifts), mass_knots, lambda_knots
            ),
            device=device,
            dtype=dtype,
        )
        torch.testing.assert_close(
            actual_file,
            expected_file,
            rtol=2 * tolerance,
            atol=2 * tolerance,
        )
        assert isinstance(actual_file, torch.Tensor)
        assert actual_file.device == mass_tensor.device
        assert actual_file.dtype == mass_tensor.dtype
        (
            actual_direct.sum()
            + actual_shifted.sum()
            + actual_file.sum()
        ).backward()
        assert mass_tensor.grad is not None
        assert bool(torch.isfinite(mass_tensor.grad).all())
        assert distance_tensor.grad is not None
        assert bool(torch.isfinite(distance_tensor.grad).all())

    assert direct_transform._torch_data_cache
    assert shifted_transform._torch_data_cache


def test_sampling_logjacobian_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([-1.5, 0.25, 2.0])
    sampling = inference_model_base.SamplingTransforms(
        variable_params=["x"],
        sampling_params=["logx"],
        replace_parameters=["x"],
        sampling_transforms=[transforms.Log("x", "logx")],
    )
    expected = sampling.logjacobian(logx=values)

    with ctx:
        tensor = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("sampling jacobian left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_model_base.numpy, "log", reject_host_or_numpy
            )
            actual = sampling.logjacobian(logx=tensor)

        assert isinstance(actual, torch.Tensor)
        assert actual.device.type == device
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12
        assert np.allclose(
            actual.detach().tolist(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
        actual.sum().backward()
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


def test_supernova_convex_hull_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    hull_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    values = {
        "coeff_0": np.array([0.1, 0.0, 0.9, 1.1, -0.1, np.nan]),
        "coeff_1": np.array([0.1, 0.5, 0.9, 0.6, 0.2, 0.1]),
        "coeff_2": np.array([0.1, 0.5, 0.9, 0.1, 0.2, 0.1]),
    }
    constraint = object.__new__(
        distributions.constraints.SupernovaeConvexHull
    )
    distributions.constraints.Constraint.__init__(constraint, "unused")
    constraint.hull_dimention = 3
    constraint.required_parameters = [
        "coeff_0", "coeff_1", "coeff_2"
    ]
    constraint._hull = (
        distributions.constraints.scipy.spatial.Delaunay(hull_points)
    )
    constraint._torch_hull_cache = {}
    constraint._torch_max_working_elements = 3
    expected = constraint(values)

    prior = distributions.JointDistribution(
        ("coeff_0", "coeff_1", "coeff_2"),
        distributions.Uniform(
            coeff_0=(-1.0, 1.0),
            coeff_1=(-1.0, 1.0),
            coeff_2=(-1.0, 1.0),
        ),
    )
    prior._constraints = [constraint]

    with ctx:
        tensor_values = {
            name: torch.as_tensor(value, dtype=dtype, device=device)
            for name, value in values.items()
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("convex-hull evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(
                distributions.constraints.scipy.spatial.Delaunay,
                "find_simplex",
                reject_host_transfer,
            )
            actual = constraint(tensor_values)
            joint_actual = prior.within_constraints(tensor_values)

    assert actual.device.type == device
    assert actual.dtype == torch.bool
    assert joint_actual.device.type == device
    assert joint_actual.dtype == torch.bool
    assert actual.detach().cpu().tolist() == expected.tolist()
    assert torch.equal(actual, joint_actual)
    assert len(constraint._torch_hull_cache) == 1


def test_fixed_samples_distribution_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    sample_values = {
        "x": np.array([
            -4.0, -2.0, -1.0, 0.5, 1.5, 3.0, 6.0, 10.0, 14.0
        ]),
        "y": np.array([9.0, -3.0, 8.0, 1.0, -4.0, 6.0, 0.0, 5.0, 2.0]),
    }
    unit_x = np.array([0.0, 0.18, 0.45, 0.72, 1.0])
    unit_y = np.array([0.95, 0.05, 0.51, 0.35, 1.0])
    host_prior = distributions.FixedSamples(("x", "y"), sample_values)
    expected = [
        host_prior.cdfinv(x=float(x), y=float(y))
        for x, y in zip(unit_x, unit_y)
    ]
    expected_x = np.asarray([value["x"] for value in expected])
    expected_y = np.asarray([value["y"] for value in expected])

    with ctx:
        tensor_samples = {
            p: torch.tensor(values, dtype=dtype, device=device)
            for p, values in sample_values.items()
        }
        unit_x_tensor = torch.tensor(unit_x, dtype=dtype, device=device)
        unit_y_tensor = torch.tensor(unit_y, dtype=dtype, device=device)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("fixed-sample distribution left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "item", reject_host_or_numpy)
            patch.setattr(
                distribution_fixedsamples.numpy, "unique", reject_host_or_numpy
            )
            patch.setattr(
                distribution_fixedsamples.numpy,
                "searchsorted",
                reject_host_or_numpy,
            )
            patch.setattr(
                distribution_fixedsamples.numpy.random,
                "randint",
                reject_host_or_numpy,
            )
            tensor_prior = distributions.FixedSamples(
                ("x", "y"), tensor_samples
            )
            one_dimensional = distributions.FixedSamples(
                ("x",), {"x": tensor_samples["x"]}
            )
            actual = tensor_prior.cdfinv(
                x=unit_x_tensor, y=unit_y_tensor
            )
            lazy = host_prior.cdfinv(x=unit_x_tensor, y=unit_y_tensor)
            draws = tensor_prior.rvs(size=64)
            one_dimensional_result = one_dimensional.cdfinv(
                x=unit_x_tensor
            )["x"]
            one_dimensional_draws = one_dimensional.rvs(size=(4, 3))["x"]

        for result in (actual, lazy, draws):
            assert all(isinstance(value, torch.Tensor)
                       for value in result.values())
            assert all(value.device.type == device
                       for value in result.values())
            assert all(value.dtype == dtype for value in result.values())
        assert np.allclose(
            actual["x"].tolist(), expected_x, rtol=tolerance, atol=tolerance
        )
        assert np.allclose(
            actual["y"].tolist(), expected_y, rtol=tolerance, atol=tolerance
        )
        assert torch.equal(actual["x"], lazy["x"])
        assert torch.equal(actual["y"], lazy["y"])
        one_dimensional_indices = np.clip(
            np.round(unit_x * len(sample_values["x"])), 0, 8
        ).astype(int)
        expected_one_dimensional = np.sort(sample_values["x"])[
            one_dimensional_indices
        ]
        assert np.allclose(
            one_dimensional_result.tolist(),
            expected_one_dimensional,
            rtol=tolerance,
            atol=tolerance,
        )
        assert one_dimensional_draws.shape == (4, 3)
        assert one_dimensional_draws.device.type == device
        assert one_dimensional_draws.dtype == dtype
        paired = (
            (draws["x"][:, None] == tensor_samples["x"][None, :])
            & (draws["y"][:, None] == tensor_samples["y"][None, :])
        )
        assert bool(paired.any(dim=1).all())


def test_nonlinear_tide_phase_conversions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    frequencies = np.array([20.0, 80.0, 150.0])
    onset = np.array([40.0, 60.0, 120.0])
    amplitudes = np.array([1.0e-8, 2.0e-8, 1.5e-8])
    indices = np.array([0.5, 1.0, 2.0])
    mass1_values = np.array([1.4, 1.5, 1.6])
    mass2_values = np.array([1.2, 1.3, 1.4])
    expected_phase = conversions.nltides_gw_phase_difference(
        frequencies, onset, amplitudes, indices,
        mass1_values, mass2_values,
    )
    expected_isco = conversions.nltides_gw_phase_diff_isco(
        20.0, onset, amplitudes, indices,
        mass1_values, mass2_values,
    )

    with ctx:
        tensors = [
            torch.tensor(
                values, device=device, dtype=dtype, requires_grad=True
            )
            for values in (
                frequencies, onset, amplitudes, indices,
                mass1_values, mass2_values,
            )
        ]
        frequency, f0, amplitude, n, mass1, mass2 = tensors

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("nonlinear-tide phase evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(
                conversions.numpy, "zeros", reject_host_or_numpy
            )
            actual_phase = conversions.nltides_gw_phase_difference(
                frequency, f0, amplitude, n, mass1, mass2
            )
            actual_isco = conversions.nltides_gw_phase_diff_isco(
                20.0, f0, amplitude, n, mass1, mass2
            )

        for actual, expected in (
                (actual_phase, expected_phase),
                (actual_isco, expected_isco)):
            assert isinstance(actual, torch.Tensor)
            assert actual.device.type == device
            assert actual.dtype == dtype
            assert np.allclose(
                actual.detach().tolist(), expected,
                rtol=tolerance, atol=tolerance,
            )

        (actual_phase.sum() + actual_isco.sum()).backward()
        assert all(value.grad is not None for value in tensors)
        assert all(bool(torch.isfinite(value.grad).all()) for value in tensors)


def test_chi_p_conversion_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([30.0, 8.0, 20.0])
    mass2 = np.array([10.0, 12.0, 20.0])
    spin1x = np.array([0.3, 0.1, -0.4])
    spin1y = np.array([0.4, -0.2, 0.0])
    spin2x = np.array([-0.6, 0.5, 0.1])
    spin2y = np.array([0.0, 0.12, 0.2])
    primary_mask = mass1 >= mass2
    expected_primary_mass = np.maximum(mass1, mass2)
    expected_secondary_mass = np.minimum(mass1, mass2)
    expected_primary_x = np.where(primary_mask, spin1x, spin2x)
    expected_primary_y = np.where(primary_mask, spin1y, spin2y)
    expected_secondary_x = np.where(primary_mask, spin2x, spin1x)
    expected_secondary_y = np.where(primary_mask, spin2y, spin1y)
    mass_ratio = expected_primary_mass / expected_secondary_mass
    primary_xi = np.hypot(expected_primary_x, expected_primary_y)
    secondary_perp = np.hypot(expected_secondary_x, expected_secondary_y)
    a1 = 2 + 1.5 * mass_ratio
    a2 = 2 + 1.5 / mass_ratio
    secondary_xi = a1 / (mass_ratio**2 * a2) * secondary_perp
    expected_chi_p = np.maximum(primary_xi, secondary_xi)

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (
                mass1, mass2, spin1x, spin1y, spin2x, spin2y
            )
        ]
        m1, m2, s1x, s1y, s2x, s2y = inputs

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("chi_p conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "sqrt", reject_host_or_numpy)
            primary = conversions.primary_mass(m1, m2)
            secondary = conversions.secondary_mass(m1, m2)
            primary_x = conversions.primary_spin(m1, m2, s1x, s2x)
            secondary_x = conversions.secondary_spin(m1, m2, s1x, s2x)
            transformed = transforms.CartesianSpinToChiP().transform({
                "mass1": m1,
                "mass2": m2,
                "spin1x": s1x,
                "spin1y": s1y,
                "spin2x": s2x,
                "spin2y": s2y,
            })
            chip = transformed["chi_p"]

        results = (primary, secondary, primary_x, secondary_x, chip)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        expected = (
            expected_primary_mass,
            expected_secondary_mass,
            expected_primary_x,
            expected_secondary_x,
            expected_chi_p,
        )
        for actual, target in zip(results, expected):
            assert np.allclose(
                actual.detach().tolist(), target,
                rtol=tolerance, atol=tolerance,
            )

        chip.sum().backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_neutron_star_remnant_fit_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-4 if dtype == torch.float32 else 2e-10
    spin = np.array([0.0, 0.25, 0.55, 0.8, 0.97, 1.0])
    inclination = np.array([0.0, 0.3, 1.0, 0.5 * np.pi, 2.3, 0.6])
    eta = np.array([0.11, 0.13, 0.16, 0.19, 0.21, 0.23])
    compactness = np.array([0.12, 0.15, 0.17, 0.2, 0.22, 0.16])
    baryonic_mass = np.array([1.35, 1.45, 1.55, 1.7, 1.8, 1.5])
    expected_isso = pg_isso.PG_ISSO_solver(spin, inclination)
    expected_remnant = neutron_eos.foucart18(
        eta, compactness, baryonic_mass, spin, inclination
    )

    with ctx:
        tensor_values = [
            torch.tensor(value, device=device, dtype=dtype)
            for value in (
                eta,
                compactness,
                baryonic_mass,
                spin,
                inclination,
            )
        ]
        t_eta, t_compactness, t_baryonic_mass, t_spin, t_inclination = (
            tensor_values
        )

        gradient_values = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (
                [0.12, 0.17, 0.21],
                [0.13, 0.17, 0.2],
                [1.4, 1.6, 1.8],
                [0.25, 0.6, 0.85],
                [0.4, 1.1, 2.0],
            )
        ]
        g_eta, g_compactness, g_baryonic_mass, g_spin, g_inclination = (
            gradient_values
        )

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("neutron-star remnant fit left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(pg_isso, "root_scalar", reject_host_or_scipy)
            patch.setattr(neutron_eos.np, "where", reject_host_or_scipy)
            isso = pg_isso.PG_ISSO_solver(t_spin, t_inclination)
            remnant = neutron_eos.foucart18(
                t_eta,
                t_compactness,
                t_baryonic_mass,
                t_spin,
                t_inclination,
            )
            differentiable_isso = pg_isso.PG_ISSO_solver(
                g_spin, g_inclination
            )
            differentiable_remnant = neutron_eos.foucart18(
                g_eta,
                g_compactness,
                g_baryonic_mass,
                g_spin,
                g_inclination,
            )

        assert isinstance(isso, torch.Tensor)
        assert isinstance(remnant, torch.Tensor)
        assert isso.device.type == device
        assert remnant.device.type == device
        assert isso.dtype == dtype
        assert remnant.dtype == dtype
        assert np.allclose(
            isso.detach().cpu().numpy(),
            expected_isso,
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.allclose(
            remnant.detach().cpu().numpy(),
            expected_remnant,
            rtol=tolerance,
            atol=tolerance,
        )

        (differentiable_isso.sum() + differentiable_remnant.sum()).backward()
        assert all(value.grad is not None for value in gradient_values)
        assert all(
            torch.isfinite(value.grad).all() for value in gradient_values
        )


def test_neutron_star_eos_conversion_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 5e-4 if dtype == torch.float32 else 3e-10

    eos_masses = np.array([0.075, 0.2, 1.4, 2.7])
    expected_compactness, expected_baryonic_mass = (
        neutron_eos.initialize_eos(eos_masses, "2H", extrapolate=True)
    )
    spherical_values = (
        np.array([7.0, 5.0, 1.25, 9.0]),
        np.array([1.35, 1.75, 6.0, 3.2]),
        np.array([0.2, 0.65, 0.1, 0.4]),
        np.array([0.3, 1.1, 0.2, 0.8]),
        np.array([0.05, 0.1, 0.55, 0.2]),
        np.array([0.4, 0.7, 1.0, 0.5]),
    )
    expected_spherical = (
        conversions.remnant_mass_from_mass1_mass2_spherical_spin_eos(
            *spherical_values[:4],
            spin2_a=spherical_values[4],
            spin2_polar=spherical_values[5],
            swap_companions=True,
            ns_bh_mass_boundary=2.5,
        )
    )
    cartesian_values = (
        np.array([7.0, 5.0]),
        np.array([1.35, 1.75]),
        np.array([0.1, 0.2]),
        np.array([0.2, 0.1]),
        np.array([0.3, 0.5]),
    )
    expected_cartesian = (
        conversions.remnant_mass_from_mass1_mass2_cartesian_spin_eos(
            *cartesian_values, ns_bh_mass_boundary=2.5
        )
    )
    spherical_remnant_fn = (
        conversions.remnant_mass_from_mass1_mass2_spherical_spin_eos
    )
    cartesian_remnant_fn = (
        conversions.remnant_mass_from_mass1_mass2_cartesian_spin_eos
    )

    with ctx:
        eos_mass = torch.tensor(
            eos_masses,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        spherical_inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in spherical_values
        ]
        cartesian_inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in cartesian_values
        ]

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("EOS conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(neutron_eos, "interp1d", reject_host_or_scipy)
            patch.setattr(pg_isso, "root_scalar", reject_host_or_scipy)
            patch.setattr(conversions, "ensurearray", reject_host_or_scipy)
            patch.setattr(conversions.numpy, "zeros", reject_host_or_scipy)
            compactness, baryonic_mass = neutron_eos.initialize_eos(
                eos_mass, "2H", extrapolate=True
            )
            spherical_remnant = spherical_remnant_fn(
                *spherical_inputs[:4],
                spin2_a=spherical_inputs[4],
                spin2_polar=spherical_inputs[5],
                swap_companions=True,
                ns_bh_mass_boundary=2.5,
            )
            cartesian_remnant = cartesian_remnant_fn(
                *cartesian_inputs, ns_bh_mass_boundary=2.5
            )

        results = (
            compactness,
            baryonic_mass,
            spherical_remnant,
            cartesian_remnant,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        np.testing.assert_allclose(
            compactness.detach().cpu().numpy(),
            expected_compactness,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            baryonic_mass.detach().cpu().numpy(),
            expected_baryonic_mass,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            spherical_remnant.detach().cpu().numpy(),
            expected_spherical,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            cartesian_remnant.detach().cpu().numpy(),
            expected_cartesian,
            rtol=tolerance,
            atol=tolerance,
        )

        sum(value.sum() for value in results).backward()
        gradient_inputs = [eos_mass, *spherical_inputs, *cartesian_inputs]
        assert all(value.grad is not None for value in gradient_inputs)
        assert all(
            torch.isfinite(value.grad).all() for value in gradient_inputs
        )

        masked = (
            conversions.remnant_mass_from_mass1_mass2_spherical_spin_eos(
                torch.tensor([7.0], device=device, dtype=dtype),
                torch.tensor([3.0], device=device, dtype=dtype),
                ns_bh_mass_boundary=2.5,
            )
        )
        assert torch.equal(masked, torch.zeros_like(masked))
        with pytest.raises(ValueError, match="Maximum NS mass"):
            neutron_eos.initialize_eos(
                torch.tensor([3.0], device=device, dtype=dtype), "2H"
            )
        with pytest.raises(ValueError, match="interpolation range"):
            neutron_eos.initialize_eos(
                torch.tensor([0.075], device=device, dtype=dtype), "2H"
            )


def test_effective_tidal_conversions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([1.6, 1.2, 1.5])
    mass2 = np.array([1.2, 1.6, 1.3])
    lambda1 = np.array([300.0, 900.0, 400.0])
    lambda2 = np.array([900.0, 300.0, 700.0])
    expected_lambda_tilde = conversions.lambda_tilde(
        mass1, mass2, lambda1, lambda2
    )
    expected_delta_lambda_tilde = conversions.delta_lambda_tilde(
        mass1, mass2, lambda1, lambda2
    )
    primary_mask = mass1 >= mass2
    expected_primary = np.where(primary_mask, lambda1, lambda2)
    expected_secondary = np.where(primary_mask, lambda2, lambda1)

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (mass1, mass2, lambda1, lambda2)
        ]
        m1, m2, tidal1, tidal2 = inputs

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("tidal conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "sqrt", reject_host_or_numpy)
            lambda_eff = conversions.lambda_tilde(
                m1, m2, tidal1, tidal2
            )
            delta_lambda_eff = conversions.delta_lambda_tilde(
                m1, m2, tidal1, tidal2
            )
            recovered_primary = (
                conversions.lambda1_from_delta_lambda_tilde_lambda_tilde(
                    delta_lambda_eff, lambda_eff, m1, m2
                )
            )
            recovered_secondary = (
                conversions.lambda2_from_delta_lambda_tilde_lambda_tilde(
                    delta_lambda_eff, lambda_eff, m1, m2
                )
            )

        results = (
            lambda_eff, delta_lambda_eff,
            recovered_primary, recovered_secondary,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 2e-4 if dtype == torch.float32 else 2e-12
        targets = (
            expected_lambda_tilde, expected_delta_lambda_tilde,
            expected_primary, expected_secondary,
        )
        for actual, target in zip(results, targets):
            assert np.allclose(
                actual.detach().tolist(), target,
                rtol=tolerance, atol=tolerance,
            )

        sum(value.sum() for value in results).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_primary_object_ordering_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = (
        [3.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [0.3, 0.1, 0.2],
        [-0.1, -0.4, -0.2],
        [300.0, 100.0, 200.0],
        [900.0, 400.0, 700.0],
    )
    expected = (
        [3.0, 4.0, 2.0],
        [1.0, 1.0, 2.0],
        [0.3, -0.4, 0.2],
        [-0.1, 0.1, -0.2],
        [300.0, 400.0, 200.0],
        [900.0, 100.0, 700.0],
    )

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in values
        ]

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("primary-object ordering left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "arange", reject_host_or_numpy)
            ordered = conversions.ensure_obj1_is_primary(*inputs)

        assert len(ordered) == len(expected)
        assert all(isinstance(value, torch.Tensor) for value in ordered)
        assert all(value.device.type == device for value in ordered)
        assert all(value.dtype == dtype for value in ordered)
        for actual, target in zip(ordered, expected):
            assert actual.detach().tolist() == pytest.approx(target)

        sum(value.sum() for value in ordered).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)

        broadcast = conversions.ensure_obj1_is_primary(
            inputs[0], 2.0, inputs[2], -0.5
        )
        assert all(value.shape == inputs[0].shape for value in broadcast)
        assert broadcast[0].detach().tolist() == pytest.approx(
            [3.0, 2.0, 2.0]
        )
        assert broadcast[1].detach().tolist() == pytest.approx(
            [2.0, 1.0, 2.0]
        )


def test_analytic_pn_utilities_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    values = (
        np.array([10.0, 1.4, 6.0]),
        np.array([1.4, 10.0, 2.0]),
        np.array([0.9, 0.1, -0.4]),
        np.array([0.1, 0.9, 0.6]),
    )
    expected_spins = (
        pnutils.mass1_mass2_spin1z_spin2z_to_beta_sigma_gamma(*values)
    )
    expected_cutoff = pnutils.f_BKLISCO(values[0], values[1])

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in values
        ]

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("analytic PN utility left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(pnutils.numpy, "where", reject_host_or_numpy)
            patch.setattr(pnutils.numpy, "minimum", reject_host_or_numpy)
            spins = pnutils.mass1_mass2_spin1z_spin2z_to_beta_sigma_gamma(
                *inputs
            )
            cutoff = pnutils.f_BKLISCO(inputs[0], inputs[1])
            broadcast_cutoff = pnutils.f_BKLISCO(inputs[0], 4.0)

        results = (*spins, cutoff, broadcast_cutoff)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual, target in zip(spins, expected_spins):
            assert np.allclose(
                actual.detach().tolist(), target,
                rtol=tolerance, atol=tolerance,
            )
        assert np.allclose(
            cutoff.detach().tolist(), expected_cutoff,
            rtol=tolerance, atol=tolerance,
        )
        assert broadcast_cutoff.shape == inputs[0].shape

        sum(value.sum() for value in results).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_pn_energy_evaluation_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    values = (30.0, 20.0, 0.3, -0.2)
    velocities = np.array([0.08, 0.16, 0.24])
    expected_coefficients = pnutils.energy_coefficients(*values)
    expected_energy = pnutils.energy(velocities, *values)

    with ctx:
        parameters = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in values
        ]
        velocity = torch.tensor(
            velocities, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("PN energy evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "item", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "__float__", reject_host_or_numpy)
            patch.setattr(pnutils.numpy, "zeros", reject_host_or_numpy)
            patch.setattr(pnutils.numpy, "arange", reject_host_or_numpy)
            coefficients = pnutils.energy_coefficients(*parameters)
            actual_energy = pnutils.energy(velocity, *parameters)
            mixed_energy = pnutils.energy(velocity, *values)

        results = (coefficients, actual_energy, mixed_energy)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        assert np.allclose(
            coefficients.detach().tolist(), expected_coefficients,
            rtol=tolerance, atol=tolerance,
        )
        for result in (actual_energy, mixed_energy):
            assert np.allclose(
                result.detach().tolist(), expected_energy,
                rtol=tolerance, atol=tolerance,
            )

        (coefficients.sum() + actual_energy.sum()
         + mixed_energy.sum()).backward()
        assert velocity.grad is not None
        assert torch.isfinite(velocity.grad).all()
        assert all(value.grad is not None for value in parameters)
        assert all(torch.isfinite(value.grad).all() for value in parameters)


def test_hybrid_pn_energy_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 5e-5 if dtype == torch.float32 else 3e-12
    velocities = np.array([0.08, 0.16, 0.24])
    parameters = (30.0, 20.0, 0.3, -0.2, 1.1, 0.9)
    expected = pnutils.hybridEnergy(velocities, *parameters)

    with ctx:
        velocity = torch.tensor(
            velocities, device=device, dtype=dtype, requires_grad=True
        )
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in parameters
        ]

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("hybrid PN energy left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "item", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "__float__", reject_host_or_numpy)
            patch.setattr(pnutils.numpy, "sqrt", reject_host_or_numpy)
            actual = pnutils.hybridEnergy(velocity, *inputs)
            mixed = pnutils.hybridEnergy(velocity, *parameters)

        for result in (actual, mixed):
            assert isinstance(result, torch.Tensor)
            assert result.device.type == device
            assert result.dtype == dtype
            assert np.allclose(
                result.detach().tolist(), expected,
                rtol=tolerance, atol=tolerance,
            )

        (actual.sum() + mixed.sum()).backward()
        assert velocity.grad is not None
        assert torch.isfinite(velocity.grad).all()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_likelihood_statistic_conversion_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        loglr = torch.tensor(
            [-2.0, 0.5, 8.0, float("nan")],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("likelihood conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "sqrt", reject_host_or_numpy)
            snr = conversions.snr_from_loglr(loglr)

        assert isinstance(snr, torch.Tensor)
        assert snr.device.type == device
        assert snr.detach().tolist() == [0.0, 1.0, 4.0, 0.0]

        snr.sum().backward()
        assert loglr.grad is not None
        assert torch.isfinite(loglr.grad).all()


def test_stepping_stone_estimator_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    likelihood_values = np.array([
        [[-1000.0, -5.0], [-3.0, -1.0]],
        [[-900.0, -4.0], [-2.0, 0.0]],
        [[-800.0, -3.0], [-1.0, 1.0]],
        [[-700.0, -2.0], [0.0, 2.0]],
    ])
    beta_values = np.array([0.0, 0.25, 0.6, 1.0])

    with ctx:
        log_likelihood = torch.tensor(
            likelihood_values,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        betas = torch.tensor(
            beta_values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("stepping-stone estimator left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_evidence.numpy, "argsort", reject_host_or_numpy
            )
            patch.setattr(inference_evidence.numpy, "exp", reject_host_or_numpy)
            patch.setattr(inference_evidence.numpy, "sqrt", reject_host_or_numpy)
            log_evidence, mcmc_std = (
                inference_evidence.stepping_stone_algorithm(
                    log_likelihood, betas
                )
            )

        assert log_evidence.device.type == device
        assert mcmc_std.device.type == device
        assert torch.isfinite(log_evidence)
        assert torch.isfinite(mcmc_std)

        (log_evidence + mcmc_std).backward()
        assert log_likelihood.grad is not None
        assert betas.grad is not None
        assert torch.isfinite(log_likelihood.grad).all()
        assert torch.isfinite(betas.grad).all()


@pytest.mark.parametrize(
    "method", ["trapezoid", "trapezoid_corrected", "simpsons"]
)
def test_thermodynamic_integration_stays_on_torch_device(
        torch_device_ctx, monkeypatch, method):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    likelihood_values = np.array([
        [[-5.0, -4.0], [-3.0, -2.0]],
        [[-4.0, -3.0], [-2.0, -1.0]],
        [[-3.0, -2.0], [-1.0, 0.0]],
        [[-2.0, -1.0], [0.0, 1.0]],
    ])
    beta_values = np.array([0.0, 0.25, 0.6, 1.0])

    with ctx:
        log_likelihood = torch.tensor(
            likelihood_values,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        betas = torch.tensor(
            beta_values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("thermodynamic integration left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_evidence.numpy, "argsort", reject_host_or_numpy
            )
            patch.setattr(
                inference_evidence.numpy, "average", reject_host_or_numpy
            )
            patch.setattr(
                inference_evidence.numpy, "trapz", reject_host_or_numpy
            )
            patch.setattr(inference_evidence.numpy, "var", reject_host_or_numpy)
            patch.setattr(inference_evidence.numpy, "std", reject_host_or_numpy)
            if hasattr(inference_evidence.integrate, "simpson"):
                patch.setattr(
                    inference_evidence.integrate,
                    "simpson",
                    reject_host_or_numpy,
                )
            if hasattr(inference_evidence.integrate, "simps"):
                patch.setattr(
                    inference_evidence.integrate,
                    "simps",
                    reject_host_or_numpy,
                )
            log_evidence, mcmc_std = (
                inference_evidence.thermodynamic_integration(
                    log_likelihood, betas, method=method
                )
            )

        assert log_evidence.device.type == device
        assert mcmc_std.device.type == device
        assert torch.isfinite(log_evidence)
        assert torch.isfinite(mcmc_std)

        (log_evidence + mcmc_std).backward()
        assert log_likelihood.grad is not None
        assert betas.grad is not None
        assert torch.isfinite(log_likelihood.grad).all()
        assert torch.isfinite(betas.grad).all()


def test_information_metrics_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    probabilities = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.array([4.0, 3.0, 2.0, 1.0])
    expected_entropy = scipy.stats.entropy(probabilities, base=2.0)
    expected_kl = scipy.stats.entropy(probabilities, reference, base=2.0)

    with ctx:
        pdf = torch.tensor(
            probabilities, device=device, dtype=dtype, requires_grad=True
        )
        reference_pdf = torch.tensor(
            reference, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("information metric left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(
                inference_entropy.stats, "entropy", reject_host_or_scipy
            )
            entropy_value = inference_entropy.entropy(pdf, base=2.0)
            kl_value = inference_entropy.kl(
                pdf, reference_pdf, pdf1=True, pdf2=True, base=2.0
            )

        assert entropy_value.device.type == device
        assert kl_value.device.type == device
        tolerance = 3e-5 if dtype == torch.float32 else 1e-12
        assert np.isclose(
            entropy_value.detach().item(), expected_entropy,
            rtol=tolerance, atol=tolerance,
        )
        assert np.isclose(
            kl_value.detach().item(), expected_kl,
            rtol=tolerance, atol=tolerance,
        )

        (entropy_value + kl_value).backward()
        assert pdf.grad is not None
        assert reference_pdf.grad is not None
        assert torch.isfinite(pdf.grad).all()
        assert torch.isfinite(reference_pdf.grad).all()


def test_histogram_information_metrics_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    samples = np.array([-0.9, -0.7, -0.2, 0.1, 0.3, 0.6, 0.8, 0.9])
    reference = np.array([-0.8, -0.4, -0.1, 0.2, 0.4, 0.5, 0.7, 0.95])
    expected_pdf = np.histogram(
        samples, bins=4, range=(-1.0, 1.0), density=True
    )[0]
    expected_reference = np.histogram(
        reference, bins=4, range=(-1.0, 1.0), density=True
    )[0]
    expected_kl = scipy.stats.entropy(
        expected_pdf, expected_reference, base=2.0
    )
    mixture = 0.5 * (expected_pdf + expected_reference)
    expected_js = 0.5 * (
        scipy.stats.entropy(expected_pdf, mixture, base=2.0)
        + scipy.stats.entropy(expected_reference, mixture, base=2.0)
    )

    with ctx:
        sample_tensor = torch.tensor(samples, device=device, dtype=dtype)
        reference_tensor = torch.tensor(
            reference, device=device, dtype=dtype
        )

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("histogram information metric left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(
                inference_entropy.numpy, "histogram", reject_host_or_scipy
            )
            patch.setattr(
                inference_entropy.stats, "entropy", reject_host_or_scipy
            )
            pdf = inference_entropy.compute_pdf(
                sample_tensor, "hist", 4, -1.0, 1.0
            )
            kl_value = inference_entropy.kl(
                sample_tensor,
                reference_tensor,
                bins=4,
                hist_min=-1.0,
                hist_max=1.0,
                base=2.0,
            )
            mixed_kl_value = inference_entropy.kl(
                sample_tensor,
                expected_reference,
                pdf2=True,
                bins=4,
                hist_min=-1.0,
                hist_max=1.0,
                base=2.0,
            )
            js_value = inference_entropy.js(
                sample_tensor,
                reference_tensor,
                bins=4,
                hist_min=-1.0,
                hist_max=1.0,
                base=2.0,
            )

        assert pdf.device.type == device
        assert kl_value.device.type == device
        assert mixed_kl_value.device.type == device
        assert js_value.device.type == device
        tolerance = 3e-5 if dtype == torch.float32 else 1e-12
        np.testing.assert_allclose(
            pdf.detach().tolist(), expected_pdf,
            rtol=tolerance, atol=tolerance,
        )
        assert np.isclose(
            kl_value.detach().item(), expected_kl,
            rtol=tolerance, atol=tolerance,
        )
        assert np.isclose(
            mixed_kl_value.detach().item(), expected_kl,
            rtol=tolerance, atol=tolerance,
        )
        assert np.isclose(
            js_value.detach().item(), expected_js,
            rtol=tolerance, atol=tolerance,
        )


def test_kde_pdf_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        samples = torch.tensor(
            [-0.9, -0.4, -0.1, 0.3, 0.8],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("KDE left Torch")

        torch.manual_seed(1234)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(
                inference_entropy.stats,
                "gaussian_kde",
                reject_host_or_scipy,
            )
            pdf = inference_entropy.compute_pdf(
                samples, "kde", None, None, None
            )

        assert pdf.device.type == device
        assert pdf.shape == (10_000,)
        assert torch.isfinite(pdf).all()
        assert (pdf > 0).all()

        pdf.mean().backward()
        assert samples.grad is not None
        assert torch.isfinite(samples.grad).all()


def test_hypertriangle_conversion_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = (
        np.array([0.2, 0.8]),
        np.array([0.4, 0.1]),
        np.array([0.6, 0.3]),
    )
    expected = conversions.hypertriangle(*values, bounds=(-1.0, 1.0))

    with ctx:
        params = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in values
        ]
        scalar_params = [param[0] for param in params]

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("hypertriangle conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "power", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "cumprod", reject_host_or_numpy)
            actual = conversions.hypertriangle(
                *params, bounds=(-1.0, 1.0)
            )
            scalar = conversions.hypertriangle(
                *scalar_params, bounds=(-1.0, 1.0)
            )
            with pytest.raises(
                    AssertionError,
                    match="same number of elements"):
                conversions.hypertriangle(params[0], params[1][:1])

        assert isinstance(actual, torch.Tensor)
        assert actual.device.type == device
        assert isinstance(scalar, list)
        assert all(isinstance(value, torch.Tensor) for value in scalar)
        assert all(value.device.type == device for value in scalar)
        tolerance = 3e-6 if dtype == torch.float32 else 1e-12
        assert np.allclose(
            actual.detach().tolist(), expected,
            rtol=tolerance, atol=tolerance,
        )
        assert torch.all(actual[1:] >= actual[:-1])

        actual.sum().backward()
        assert all(param.grad is not None for param in params)
        assert all(torch.isfinite(param.grad).all() for param in params)


def test_precession_spin_trigonometry_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([30.0, 18.0, 12.0])
    mass2 = np.array([10.0, 9.0, 8.0])
    xi1 = np.array([0.4, 0.25, 0.1])
    xi2 = np.array([0.2, 0.35, 0.15])
    phi_a = np.array([0.3, 1.1, 2.2])
    phi_s = np.array([1.5, 2.0, 4.0])
    phi1 = (phi_s + phi_a) / 2.0
    phi2 = (phi_s - phi_a) / 2.0
    mass_ratio = mass1 / mass2
    a1 = 2 + 1.5 * mass_ratio
    a2 = 2 + 1.5 / mass_ratio
    secondary_perp = mass_ratio**2 * a2 / a1 * xi2
    expected = (
        xi1 * np.cos(phi1),
        xi1 * np.sin(phi1),
        secondary_perp * np.cos(phi2),
        secondary_perp * np.sin(phi2),
    )
    expected_primary_phi = np.mod(
        np.arctan2(expected[1], expected[0]), 2 * np.pi
    )
    expected_secondary_phi = np.mod(
        np.arctan2(expected[3], expected[2]), 2 * np.pi
    )

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (mass1, mass2, xi1, xi2, phi_a, phi_s)
        ]
        m1, m2, primary_xi, secondary_xi, angle_a, angle_s = inputs

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("precession spin conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "arctan2", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "cos", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "sin", reject_host_or_numpy)
            spin1x = conversions.spin1x_from_xi1_phi_a_phi_s(
                primary_xi, angle_a, angle_s
            )
            spin1y = conversions.spin1y_from_xi1_phi_a_phi_s(
                primary_xi, angle_a, angle_s
            )
            spin2x = conversions.spin2x_from_mass1_mass2_xi2_phi_a_phi_s(
                m1, m2, secondary_xi, angle_a, angle_s
            )
            spin2y = conversions.spin2y_from_mass1_mass2_xi2_phi_a_phi_s(
                m1, m2, secondary_xi, angle_a, angle_s
            )
            primary_phi = conversions.phi_from_spinx_spiny(
                spin1x, spin1y
            )
            secondary_phi = conversions.phi_from_spinx_spiny(
                spin2x, spin2y
            )

        results = (
            spin1x, spin1y, spin2x, spin2y,
            primary_phi, secondary_phi,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        targets = expected + (expected_primary_phi, expected_secondary_phi)
        for actual, target in zip(results, targets):
            assert np.allclose(
                actual.detach().tolist(), target,
                rtol=tolerance, atol=tolerance,
            )

        sum(value.sum() for value in results[:4]).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_count_posterior_probabilities_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    posterior = count_posterior(
        np.array([-1.5, -0.25, 0.4]),
        laguerre_n=24,
        Lambda0=3.0,
    )
    query = np.array([-6.0, -1.0, 0.0, 2.5], dtype=np.float32)
    expected = posterior.p_bg(query)

    with ctx:
        logbf = Array(query)

        def fail_numpy(*_args, **_kwargs):
            raise AssertionError("posterior query copied through NumPy")

        monkeypatch.setattr(TorchArrayData, "numpy", fail_numpy)
        actual = posterior.p_bg(logbf)

        assert isinstance(actual, Array)
        assert isinstance(actual._data, TorchArrayData)
        assert actual._data.tensor.device.type == device
        np.testing.assert_allclose(
            actual._data.tensor.detach().cpu().numpy(),
            expected,
            rtol=3e-6 if device == "mps" else 2e-13,
            atol=0.0,
        )


def test_live_pastro_signal_pdf_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    snr = np.array([5.0, 5.5, 6.25, 8.5], dtype=dtype)
    threshold = np.array(5.0, dtype=dtype)
    expected = live_pastro.signal_pdf_from_snr(snr, threshold)

    with ctx:
        snr_tensor = torch.tensor(
            snr, dtype=torch_dtype, device=device, requires_grad=True
        )
        threshold_tensor = torch.tensor(
            threshold,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        torch_snr = Array(TorchArrayData(snr_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("live p_astro density left Torch")

        def reject_numpy_density(*_args, **_kwargs):
            raise AssertionError("live p_astro density used NumPy")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(live_pastro.numpy, "exp", reject_numpy_density)
            patch.setattr(fgmc_functions.np, "log", reject_numpy_density)
            raw = live_pastro.signal_pdf_from_snr(
                snr_tensor, threshold_tensor
            )
            wrapped = live_pastro.signal_pdf_from_snr(
                torch_snr, threshold_tensor
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    for values in (snr_tensor, threshold_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0


def test_gated_covariance_fit_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx

    class ConcreteGatedGaussian(gated_gaussian_noise.BaseGatedGaussian):
        def get_gated_waveforms(self):
            return None

        def get_waveforms(self):
            return None

    with ctx:
        bins = np.arange(33, dtype=np.float32)
        psd = FrequencySeries(
            1.0 + 0.05 * np.cos(np.pi * bins / 32),
            delta_f=0.25,
        )
        correlation = psd.astype(
            pycbc.types.complex_same_precision_as(psd)
        ).to_timeseries()
        assert correlation._data.tensor.device.type == device

        psd_values = psd._data.tensor.detach().cpu().numpy().astype(np.float64)
        correlation_values = (
            correlation._data.tensor.detach().cpu().numpy().astype(np.float64)
        )
        covariance = scipy.linalg.toeplitz(correlation_values / 2)
        size = len(correlation_values)
        sample_sizes = [size, size // 2, size // 4, size // 8]
        expected_dets = []
        for sample_size in sample_sizes:
            if sample_size == size:
                expected_dets.append(
                    2 * np.log(psd_values / (2 * psd.delta_t)).sum()
                )
            else:
                gate_size = size - sample_size
                start = (size - gate_size) // 2
                end = start + gate_size
                retained = np.r_[0:start, end:size]
                truncated = covariance[np.ix_(retained, retained)]
                factor = scipy.linalg.cholesky(truncated, lower=True)
                expected_dets.append(2 * np.log(np.diag(factor)).sum())
        design = np.vstack(
            [sample_sizes, np.ones(len(sample_sizes))]
        ).T
        expected_fit = np.linalg.lstsq(
            design, expected_dets, rcond=None
        )[0]

        model = object.__new__(ConcreteGatedGaussian)
        model._psds = {"H1": psd}
        model._Rss = {"H1": correlation}
        model._cov_samples = {}
        model._cov_regressions = {}

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("covariance fit copied a Torch array to host")

        def reject_scipy_toeplitz(*_args, **_kwargs):
            raise AssertionError("covariance fit used SciPy Toeplitz")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(
                gated_gaussian_noise.scipy.linalg,
                "toeplitz",
                reject_scipy_toeplitz,
            )
            model._set_covfit("H1")

        actual_sizes, actual_dets = model._cov_samples["H1"]
        actual_fit = model._cov_regressions["H1"]
        assert actual_sizes == sample_sizes
        np.testing.assert_allclose(
            actual_dets, expected_dets, rtol=2e-4, atol=2e-4
        )
        np.testing.assert_allclose(
            actual_fit, expected_fit, rtol=2e-4, atol=2e-4
        )


def test_higher_mode_marginalization_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    length = 28
    delta_f = 0.25
    indices = np.arange(length)
    mode_values = {
        (2, 2): (
            (0.7 + 0.015 * indices) * np.exp(0.04j * indices),
            (0.2 + 0.008 * indices) * np.exp(-0.03j * indices),
        ),
        (3, 3): (
            (0.3 + 0.009 * indices) * np.exp(0.025j * indices),
            (0.1 + 0.004 * indices) * np.exp(-0.05j * indices),
        ),
    }
    data_values = (
        (1.0 + 0.006 * indices) * np.exp(0.018j * indices)
    )
    weight_values = np.linspace(0.8, 1.2, length)
    reference_time = 1_126_259_462.4
    detector_calls = []

    class _Detector:
        def arrival_time(self, ref_tc, ra, dec, ref_frame):
            assert ref_frame == "geocentric"
            on_device = all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            detector_calls.append(("arrival", on_device))
            if isinstance(ra, torch.Tensor):
                ref_tc = torch.as_tensor(
                    ref_tc, device=ra.device, dtype=ra.dtype)
            return ref_tc + 0.02 * (ra - 1.2) + 0.03 * (dec + 0.4)

        def antenna_pattern(self, ra, dec, polarization, tc):
            on_device = all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec, polarization, tc)
            )
            detector_calls.append(("antenna", on_device))
            if isinstance(polarization, torch.Tensor):
                ra, dec, tc = (
                    torch.as_tensor(
                        value,
                        device=polarization.device,
                        dtype=polarization.dtype,
                    )
                    for value in (ra, dec, tc)
                )
                cosine = torch.cos(2.0 * polarization)
                sine = torch.sin(2.0 * polarization)
            else:
                cosine = np.cos(2.0 * polarization)
                sine = np.sin(2.0 * polarization)
            delay = tc - reference_time
            base_fp = 0.61 + 0.01 * (ra - 1.2) + 0.03 * delay
            base_fc = -0.27 + 0.02 * (dec + 0.4) + 0.04 * delay
            return (
                base_fp * cosine + base_fc * sine,
                base_fc * cosine - base_fp * sine,
            )

    def make_model():
        model = types.SimpleNamespace()
        polarization_samples = 17
        phase_samples = 19
        pol = np.linspace(0, 2 * np.pi, polarization_samples)
        phase = np.linspace(0, 2 * np.pi, phase_samples)
        model.nsamples = polarization_samples * phase_samples
        model.pol = np.resize(pol, model.nsamples)
        phase = np.resize(phase, model.nsamples).reshape(
            phase_samples, polarization_samples
        )
        model.phase = phase.T.flatten()
        model._phase_fac = {}
        model._torch_marginalization_grids = {}
        model._torch_phase_fac = {}
        model.dets = {"H1": _Detector()}
        model.current_params = {
            "tc": reference_time,
            "ra": 1.2,
            "dec": -0.4,
            "inclination": 0.9,
        }
        model._current_stats = types.SimpleNamespace()
        model._kmin = {"H1": 2}
        model._kmax = {"H1": length - 2}
        model._weight = {
            "H1": FrequencySeries(weight_values, delta_f=delta_f)
        }
        model._whitened_data = {
            "H1": FrequencySeries(data_values, delta_f=delta_f)
        }

        def generate(**_params):
            return {
                "H1": {
                    mode: tuple(
                        FrequencySeries(values, delta_f=delta_f)
                        for values in pair
                    )
                    for mode, pair in mode_values.items()
                }
            }

        model.waveform_generator = types.SimpleNamespace(generate=generate)
        model._marginalization_grids = types.MethodType(
            marginalized_gaussian_noise.MarginalizedHMPolPhase
            ._marginalization_grids,
            model,
        )
        model.phase_fac = types.MethodType(
            marginalized_gaussian_noise.MarginalizedHMPolPhase.phase_fac,
            model,
        )
        return model

    likelihood = (
        marginalized_gaussian_noise.MarginalizedHMPolPhase
        ._loglr.__wrapped__
    )
    expected_model = make_model()
    expected = likelihood(expected_model)

    with ctx:
        actual_model = make_model()

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "higher-mode marginalization copied arrays to host"
            )

        def reject_scalar_inner(*_args, **_kwargs):
            raise AssertionError(
                "higher-mode marginalization scalarized inner products"
            )

        with monkeypatch.context() as patch:
            import pycbc.types.array_torch as array_torch

            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(array_torch, "inner", reject_scalar_inner)
            patch.setattr(
                marginalized_gaussian_noise.special,
                "logsumexp",
                reject_host_transfer,
            )
            actual = likelihood(actual_model)

            gradient_model = make_model()
            right_ascension = torch.tensor(
                1.2, dtype=torch.float64, device=device,
                requires_grad=True)
            declination = torch.tensor(
                -0.4, dtype=torch.float64, device=device,
                requires_grad=True)
            gradient_model.current_params.update({
                "ra": right_ascension,
                "dec": declination,
            })
            _, _, gradient_lr, _, _ = likelihood(
                gradient_model, return_unmarginalized=True)
            gradient_lr.sum().backward()

        sky_gradients = tuple(
            value.grad.detach().cpu().numpy()
            for value in (right_ascension, declination)
        )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        np.remainder(actual_model._current_stats.maxl_polarization, np.pi),
        np.remainder(expected_model._current_stats.maxl_polarization, np.pi),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        np.remainder(actual_model._current_stats.maxl_phase, 2 * np.pi),
        np.remainder(expected_model._current_stats.maxl_phase, 2 * np.pi),
        rtol=0,
        atol=2e-15,
    )
    assert len(actual_model._torch_marginalization_grids) == 1
    grids = next(iter(actual_model._torch_marginalization_grids.values()))
    assert all(grid.device.type == device for grid in grids)
    assert len(actual_model._torch_phase_fac) == len(mode_values)
    assert all(
        factor.device.type == device
        for factor in actual_model._torch_phase_fac.values()
    )
    assert detector_calls == [
        ("arrival", False),
        ("antenna", False),
        ("arrival", True),
        ("antenna", True),
        ("arrival", True),
        ("antenna", True),
    ]
    assert all(
        np.isfinite(value) and value != 0.0 for value in sky_gradients
    )


def test_array_take_keeps_torch_indices_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = Array(np.arange(6, dtype=dtype))
        index_tensor = torch.tensor(
            [4, 0, 2], device=device, dtype=torch.int64
        )
        indices = Array(TorchArrayData(index_tensor), copy=False)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("Array.take transferred Torch indices to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = values.take(indices)

        expected = torch.tensor(
            [4.0, 0.0, 2.0], device=device, dtype=torch_dtype
        )
        assert actual._data.tensor.device.type == device
        torch.testing.assert_close(actual._data.tensor, expected)


@pytest.mark.parametrize("delta_f", (0.5, 0.3))
def test_template_bank_moments_stay_on_torch_device(
        torch_device_ctx, monkeypatch, delta_f):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Template-bank moments require float64")

    frequencies = np.arange(1024, dtype=np.float64) * delta_f
    psd = 1.0 + (frequencies / 100.0) ** 2

    expected_params = metricParameters(
        "threePointFive", 20.0, 200.0, delta_f, f0=70.0
    )
    expected_params.psd = FrequencySeries(psd, delta_f=delta_f)
    get_moments(expected_params, vary_fmax=True, vary_density=37.0)

    with ctx:
        actual_params = metricParameters(
            "threePointFive", 20.0, 200.0, delta_f, f0=70.0
        )
        actual_params.psd = FrequencySeries(psd, delta_f=delta_f)
        source = actual_params.psd._data
        assert isinstance(source, TorchArrayData)
        assert source.tensor.device.type == device

        def forbid_host_array(*args, **kwargs):
            raise AssertionError("template-bank moments copied the PSD to NumPy")

        def forbid_scalar_index(*args, **kwargs):
            raise AssertionError("template-bank moments read PSD host scalars")

        monkeypatch.setattr(TorchArrayData, "numpy", forbid_host_array)
        monkeypatch.setattr(TorchArrayData, "__getitem__", forbid_scalar_index)
        get_moments(actual_params, vary_fmax=True, vary_density=37.0)

        assert actual_params.psd._data.tensor.device.type == device

    assert actual_params.moments.keys() == expected_params.moments.keys()
    for name, expected in expected_params.moments.items():
        actual = actual_params.moments[name]
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(
            list(actual.values()),
            list(expected.values()),
            rtol=1e-12,
            atol=0.0,
        )


def test_template_bank_frequency_selection_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    input_freqs = [5.0, 14.9, 15.0, 15.1, 29.9, 30.0, 30.1, 50.0,
                   float("nan")]
    metric_freqs = [10.0, 20.0, 40.0]
    expected = tmpltbank_coord_utils.find_closest_calculated_frequencies(
        np.asarray(input_freqs), np.asarray(metric_freqs)
    )

    with ctx:
        input_tensor = torch.tensor(input_freqs, device=device, dtype=dtype)
        metric_tensor = torch.tensor(
            metric_freqs, device=device, dtype=dtype
        )

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError(
                "template-bank frequency selection used the host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(torch.Tensor, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "item", reject_host_path)
            patch.setattr(tmpltbank_coord_utils.numpy, "zeros", reject_host_path)
            patch.setattr(
                tmpltbank_coord_utils.numpy, "logical_and", reject_host_path
            )
            actual = (
                tmpltbank_coord_utils.find_closest_calculated_frequencies(
                    input_tensor, metric_freqs
                )
            )
            mixed = (
                tmpltbank_coord_utils.find_closest_calculated_frequencies(
                    input_freqs, metric_tensor
                )
            )
            scalar = (
                tmpltbank_coord_utils.find_closest_calculated_frequencies(
                    torch.tensor(14.0, device=device, dtype=dtype),
                    metric_freqs,
                )
            )
            singleton = (
                tmpltbank_coord_utils.find_closest_calculated_frequencies(
                    input_tensor,
                    torch.tensor([25.0], device=device, dtype=dtype),
                )
            )

    for result in (actual, mixed, scalar, singleton):
        assert isinstance(result, torch.Tensor)
        assert result.device.type == device
        assert result.dtype == dtype
    assert scalar.shape == (1,)
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, device=device, dtype=dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mixed, actual, rtol=0, atol=0)
    torch.testing.assert_close(
        scalar, torch.tensor([10.0], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        singleton, torch.full_like(input_tensor, 25.0), equal_nan=True
    )


def test_template_bank_cutoff_dispatch_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.asarray([20.0, 35.0, 50.0])
    mass2 = np.asarray([10.0, 15.0, 20.0])
    metric_freqs = np.asarray([50.0, 100.0, 200.0])
    expected = tmpltbank_coord_utils.return_nearest_cutoff(
        "SchwarzISCO",
        {"mass1": mass1, "mass2": mass2},
        metric_freqs,
    )

    with ctx:
        mass1_tensor = torch.as_tensor(mass1, device=device, dtype=dtype)
        mass2_tensor = torch.as_tensor(mass2, device=device, dtype=dtype)
        metric_tensor = torch.as_tensor(
            metric_freqs, device=device, dtype=dtype
        )

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("template cutoff dispatch used the host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(torch.Tensor, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "item", reject_host_path)
            patch.setattr(tmpltbank_coord_utils.numpy, "zeros", reject_host_path)
            canonical = tmpltbank_coord_utils.return_nearest_cutoff(
                "SchwarzISCO",
                {"mass1": mass1_tensor, "mass2": mass2},
                metric_freqs,
            )
            legacy = tmpltbank_coord_utils.return_nearest_cutoff(
                "SchwarzISCO",
                {"m1": mass1_tensor, "m2": mass2_tensor},
                metric_freqs,
            )
            mixed = tmpltbank_coord_utils.return_nearest_cutoff(
                "SchwarzISCO",
                {"mass1": mass1, "mass2": mass2},
                metric_tensor,
            )
            singleton = tmpltbank_coord_utils.return_nearest_cutoff(
                "ignored-for-single-frequency",
                {"mass1": mass1_tensor, "mass2": mass2_tensor},
                torch.tensor([64.0], device=device, dtype=dtype),
            )
            scalar = tmpltbank_coord_utils.return_nearest_cutoff(
                "ignored-for-single-frequency",
                {
                    "m1": torch.tensor(30.0, device=device, dtype=dtype),
                    "m2": torch.tensor(20.0, device=device, dtype=dtype),
                },
                [64.0],
            )

    for result in (canonical, legacy, mixed, singleton, scalar):
        assert isinstance(result, torch.Tensor)
        assert result.device.type == device
        assert result.dtype == dtype
    torch.testing.assert_close(
        canonical,
        torch.as_tensor(expected, device=device, dtype=dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(legacy, canonical, rtol=0, atol=0)
    torch.testing.assert_close(mixed, canonical, rtol=0, atol=0)
    torch.testing.assert_close(
        singleton, torch.full_like(mass1_tensor, 64.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scalar, torch.tensor([64.0], device=device, dtype=dtype)
    )


@pytest.mark.parametrize("index_kind", ("integer", "boolean"))
def test_array_advanced_indices_stay_on_torch_device(
        torch_device_ctx, monkeypatch, index_kind):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = Array(np.arange(6, dtype=dtype))
        if index_kind == "integer":
            tensor = torch.tensor(
                [4, 0, 2], device=device, dtype=torch.int64
            )
            expected = torch.tensor(
                [4.0, 0.0, 2.0], device=device, dtype=torch_dtype
            )
        else:
            tensor = torch.tensor(
                [True, False, True, False, False, True],
                device=device,
            )
            expected = torch.tensor(
                [0.0, 2.0, 5.0], device=device, dtype=torch_dtype
            )
        indices = (
            tensor
            if index_kind == "boolean"
            else Array(TorchArrayData(tensor), copy=False)
        )

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("advanced indexing copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            actual = values[indices]

    assert isinstance(actual, TorchArrayData)
    assert actual.tensor.device.type == device
    torch.testing.assert_close(actual.tensor, expected)


def test_coincidence_indexes_and_cuts_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    delays = {"H1": 1, "L1": 2, "V1": 2}
    host_indexes = {
        "H1": np.array([2, 4, 7, 10], dtype=np.int32),
        "L1": np.array([3, 5, 8, 11], dtype=np.int32),
        "V1": np.array([5, 8, 12], dtype=np.int32),
    }
    host_snrs = {
        "H1": np.arange(16, dtype=dtype),
        "L1": np.arange(16, dtype=dtype) * 0.5,
        "V1": np.arange(16, dtype=dtype) * 0.25,
    }
    expected_index = coherent.get_coinc_indexes(host_indexes, delays, 2)
    expected = coherent.coincident_snr(
        host_snrs, expected_index, 6.0, delays
    )

    with ctx:
        torch_indexes = {
            ifo: Array(values) for ifo, values in host_indexes.items()
        }
        torch_snrs = {
            ifo: Array(values) for ifo, values in host_snrs.items()
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("coincidence processing copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            coinc_index = coherent.get_coinc_indexes(
                torch_indexes, delays, 2
            )
            actual = coherent.coincident_snr(
                torch_snrs, coinc_index, 6.0, delays
            )
            alternate_index = Array(
                TorchArrayData(actual[1]._data.tensor + 10),
                copy=False,
            )
            selected_index = coherent.select_coherent_values(
                actual[1],
                alternate_index,
                coherent.compare_coherent_values(
                    actual[1], 8, np.less
                ),
            )

    actual_rho, actual_index, actual_triggers = actual
    expected_rho, expected_survivors, expected_triggers = expected
    assert isinstance(coinc_index, Array)
    assert coinc_index.dtype == np.dtype(np.int64)
    assert coinc_index._data.tensor.device.type == device
    np.testing.assert_array_equal(
        coinc_index._data.tensor.detach().cpu().numpy(), expected_index
    )
    assert isinstance(actual_index, Array)
    assert actual_index._data.tensor.device.type == device
    np.testing.assert_array_equal(
        actual_index._data.tensor.detach().cpu().numpy(), expected_survivors
    )
    assert isinstance(selected_index, Array)
    assert selected_index._data.tensor.device.type == device
    np.testing.assert_array_equal(
        selected_index._data.tensor.detach().cpu().numpy(),
        np.where(
            expected_survivors < 8,
            expected_survivors,
            expected_survivors + 10,
        ),
    )
    assert isinstance(actual_rho, Array)
    assert actual_rho._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual_rho._data.tensor.detach().cpu().numpy(), expected_rho,
        rtol=2e-6,
    )
    for ifo, trigger in actual_triggers.items():
        assert isinstance(trigger, Array)
        assert trigger._data.tensor.device.type == device
        np.testing.assert_allclose(
            trigger._data.tensor.detach().cpu().numpy(),
            expected_triggers[ifo],
        )


def test_quadrature_statistics_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    calculator = object.__new__(event_stat.QuadratureSumStatistic)

    first = np.array([3.0, -1.0, 6.0, 8.0], dtype=dtype)
    second = np.array([4.0, 12.0, -1.0, 15.0], dtype=dtype)
    expected = np.array([5.0, 0.0, 0.0, 17.0], dtype=dtype)
    limit_input = np.array([3.0, 5.0, 13.0, 15.0], dtype=dtype)
    expected_limit = np.sqrt(
        np.maximum(13.0 ** 2 - limit_input ** 2, 0.0)
    ).astype(dtype)

    # The NumPy fallback must apply the cut sentinel to the values rather
    # than accidentally comparing the surrounding (IFO, values) pair.
    host_result = calculator.rank_stat_coinc(
        [("H1", first.copy()), ("L1", second.copy())],
        None,
        None,
        None,
    )
    np.testing.assert_allclose(host_result, expected)

    with ctx:
        singles = [("H1", Array(first)), ("L1", Array(second))]
        threshold_singles = [("H1", Array(limit_input))]

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("quadrature statistic copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = calculator.rank_stat_coinc(
                singles, None, None, None
            )
            actual_limit = calculator.coinc_lim_for_thresh(
                threshold_singles, 13.0, "L1"
            )

    for value, reference in (
        (actual, expected),
        (actual_limit, expected_limit),
    ):
        assert isinstance(value, Array)
        assert value._data.tensor.device.type == device
        torch.testing.assert_close(
            value._data.tensor,
            torch.as_tensor(
                reference, device=device, dtype=torch_dtype
            ),
        )


@pytest.mark.parametrize(
    "host_indexes, delays, min_nifos",
    (
        ({"H1": []}, {"H1": 0}, 1),
        ({"H1": [7, 2, 7, 4]}, {"H1": 1}, 2),
        ({"H1": [], "L1": []}, {"H1": 0, "L1": 1}, 2),
        (
            {"H1": [2, 4, 7, 7], "L1": [3, 5, 9], "V1": [4, 8]},
            {"H1": 1, "L1": 2, "V1": 3},
            2,
        ),
        (
            {"H1": [2, 4, 7], "L1": [3, 5, 8], "V1": [4, 6, 9]},
            {"H1": 1, "L1": 2, "V1": 3},
            3,
        ),
    ),
)
def test_coincidence_index_torch_edge_cases(
        torch_device_ctx, monkeypatch, host_indexes, delays, min_nifos):
    expected = coherent.get_coinc_indexes(
        {
            ifo: np.asarray(values, dtype=np.int32)
            for ifo, values in host_indexes.items()
        },
        delays,
        min_nifos,
    )
    ctx, device = torch_device_ctx
    with ctx:
        indexes = {
            ifo: Array(np.asarray(values, dtype=np.int32))
            for ifo, values in host_indexes.items()
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("coincidence indexing copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = coherent.get_coinc_indexes(
                indexes, delays, min_nifos
            )

    assert isinstance(actual, Array)
    assert actual.dtype == np.dtype(np.int64)
    assert actual._data.tensor.device.type == device
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_coherent_branch_selection_and_null_cut_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    left = {
        "H1": np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=dtype),
        "L1": np.array([2 - 1j, 4 - 3j, 6 - 5j, 8 - 7j], dtype=dtype),
    }
    right = {
        "H1": np.array([8 + 7j, 6 + 5j, 4 + 3j, 2 + 1j], dtype=dtype),
        "L1": np.array([7 - 8j, 5 - 6j, 3 - 4j, 1 - 2j], dtype=dtype),
    }
    select_left = np.array([True, False, False, True])
    expected = {
        ifo: np.where(select_left, left[ifo], right[ifo])
        for ifo in left
    }
    rho_coh = np.array([10.0, 10.0, 25.0, 25.0])
    rho_coinc = np.array([11.0, 14.0, 25.5, 27.0])
    rho_coh_right = np.array([9.0, 12.0, 26.0, 24.0])
    expected_selected_coh = np.where(
        select_left, rho_coh, rho_coh_right
    )
    expected_keep = np.array([True, False, True, False])
    index = np.arange(4)

    with ctx:
        torch_left = {ifo: Array(values) for ifo, values in left.items()}
        torch_right = {ifo: Array(values) for ifo, values in right.items()}
        with monkeypatch.context() as patch:
            original_cpu = torch.Tensor.cpu
            host_selections = []

            def _reject_host_transfer(_self):
                raise AssertionError(
                    "coherent branch selection copied triggers to host"
                )

            def _guard_selection_transfer(tensor, *args, **kwargs):
                if tensor.dtype == torch.bool:
                    raise AssertionError(
                        "null SNR copied its boolean mask to host"
                    )
                host_selections.append(tensor.numel())
                return original_cpu(tensor, *args, **kwargs)

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _guard_selection_transfer)
            selected = coherent.select_coherent_triggers(
                torch_left, torch_right, select_left
            )
            selected_summary = coherent.select_coherent_values(
                Array(rho_coh), Array(rho_coh_right), select_left
            )
            null, selected_coh, selected_coinc, selected_index, selected = (
                coherent.null_snr(
                    Array(rho_coh),
                    Array(rho_coinc),
                    index=index,
                    snrv=selected,
                )
            )

    np.testing.assert_array_equal(selected_index, index[expected_keep])
    assert host_selections == [np.count_nonzero(expected_keep)]
    assert isinstance(selected_summary, Array)
    assert selected_summary._data.tensor.device.type == device
    np.testing.assert_array_equal(
        selected_summary._data.tensor.detach().cpu().numpy(),
        expected_selected_coh,
    )
    for summary, expected_summary in (
        (selected_coh, rho_coh[expected_keep]),
        (selected_coinc, rho_coinc[expected_keep]),
        (
            null,
            np.sqrt(
                rho_coinc[expected_keep] ** 2
                - rho_coh[expected_keep] ** 2
            ),
        ),
    ):
        assert isinstance(summary, Array)
        assert summary._data.tensor.device.type == device
        np.testing.assert_allclose(
            summary._data.tensor.detach().cpu().numpy(), expected_summary
        )
    for ifo, trigger in selected.items():
        assert isinstance(trigger, Array)
        assert trigger._data.tensor.device.type == device
        np.testing.assert_array_equal(
            trigger._data.tensor.detach().cpu().numpy(),
            expected[ifo][expected_keep],
        )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_ranking_moves_host_summaries_to_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    snr = np.array([8.0, 7.5, 6.0, 5.5, 4.0, 9.0], dtype=dtype)
    brchisq = np.array(
        [0.5, 1.0, 1.5, 4.0, 12.0, np.nan], dtype=dtype
    )
    sgchisq = np.array(
        [2.0, 4.0, 8.0, 16.0, np.nan, 5.0], dtype=dtype
    )
    psd_var = np.array(
        [0.5, 0.65, 1.0, 4.0, 12.0, np.nan], dtype=dtype
    )
    calls = (
        ("effsnr", (brchisq,), {"fac": 200.0}),
        ("newsnr_sgveto", (brchisq, sgchisq), {}),
        (
            "newsnr_sgveto_psdvar",
            (brchisq, sgchisq, psd_var),
            {},
        ),
        (
            "newsnr_sgveto_psdvar_threshold",
            (brchisq, sgchisq, psd_var),
            {},
        ),
        (
            "newsnr_sgveto_psdvar_scaled",
            (brchisq, sgchisq, psd_var),
            {},
        ),
        (
            "newsnr_sgveto_psdvar_scaled_threshold",
            (brchisq, sgchisq, psd_var),
            {},
        ),
    )
    expected = {
        name: getattr(ranking, name)(snr, *summaries, **kwargs)
        for name, summaries, kwargs in calls
    }

    with ctx:
        torch_snr = Array(snr)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "ranking statistic copied Torch triggers to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = {
                name: getattr(ranking, name)(
                    torch_snr, *summaries, **kwargs
                )
                for name, summaries, kwargs in calls
            }

    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    expected_dtype = np.float32 if device == "mps" else np.float64
    for name, values in actual.items():
        assert isinstance(values, Array)
        assert values._data.tensor.device.type == device
        assert values.dtype == np.dtype(expected_dtype)
        np.testing.assert_allclose(
            values._data.tensor.detach().cpu().numpy(),
            expected[name],
            rtol=tolerance,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "function_name",
    (
        "get_snr",
        "get_newsnr",
        "get_newsnr_sgveto",
        "get_newsnr_sgveto_psdvar",
        "get_newsnr_sgveto_psdvar_threshold",
        "get_newsnr_sgveto_psdvar_scaled",
        "get_newsnr_sgveto_psdvar_scaled_threshold",
    ),
)
def test_trigger_ranking_wrappers_stay_on_torch_device(
        torch_device_ctx, monkeypatch, function_name):
    ctx, device = torch_device_ctx
    host_trigs = {
        "snr": np.array([8.0, 7.5, 6.0, 5.5, 4.0], dtype=np.float32),
        "chisq": np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32),
        "chisq_dof": np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "sg_chisq": np.array([2.0, 4.0, 8.0, 16.0, 5.0], dtype=np.float32),
        "psd_var_val": np.array([0.5, 0.65, 1.0, 4.0, 12.0], dtype=np.float32),
    }
    function = getattr(ranking, function_name)
    expected = function(host_trigs)

    assert isinstance(expected, np.ndarray)
    assert expected.dtype == np.float32

    with ctx:
        torch_trigs = {
            name: Array(values) for name, values in host_trigs.items()
        }
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    f"{function_name} copied Torch triggers to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = function(torch_trigs)

    assert isinstance(actual, Array)
    assert actual.dtype == np.dtype(np.float32)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_live_single_selection_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    host_trigs = {
        "snr": np.array([5.0, 7.0, 8.0, 9.0], dtype=dtype),
        "chisq": np.ones(4, dtype=dtype),
        "chisq_dof": np.full(4, 2.0, dtype=dtype),
        "template_duration": np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype),
        "end_time": np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype),
    }

    class _Statistic:
        @staticmethod
        def get_sngl_ranking(trigs):
            return trigs["snr"] - 1.5

        @staticmethod
        def single(trigs):
            return trigs["snr"]

        @staticmethod
        def rank_stat_single(single_info):
            return single_info[1]

    class _DataReader:
        @staticmethod
        def near_hwinj():
            return False

    live = object.__new__(event_single.LiveSingle)
    live.ifo = "H1"
    live.thresholds = {
        "duration": 1.5,
        "reduced_chisq": 5.0,
        "ranking": 6.0,
    }
    live.stat_calculator = _Statistic()
    live.stat_calculator_lock = event_single.threading.Lock()
    ifar_inputs = []

    def calculate_ifar(rank, duration):
        ifar_inputs.append((rank, duration))
        return 100.0

    live.calculate_ifar = calculate_ifar

    with ctx:
        trigs = {
            name: Array(
                TorchArrayData(
                    torch.tensor(values, dtype=torch_dtype, device=device)
                ),
                copy=False,
            )
            for name, values in host_trigs.items()
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("live single trigger vector left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(event_single.np, "any", reject_host_transfer)
            candidate = live.check(trigs, _DataReader())

    assert ifar_inputs == [(9.0, 4.0)]
    assert candidate["foreground/H1/end_time"] == 40.0
    assert candidate["foreground/stat"] == 9.0
    assert candidate["foreground/ifar"] == 100.0
    assert candidate["HWINJ"] is False


@pytest.mark.parametrize("slide_step", (0.0, 10.0))
def test_time_coincidence_stays_on_torch_device(
        torch_device_ctx, monkeypatch, slide_step):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_times1 = np.array(
        [100.125, 109.75, 110.25, 119.875, 120.5, 129.75, 130.125],
        dtype=np.float64,
    )
    host_times2 = np.array(
        [100.0, 109.875, 110.375, 120.0, 130.0, 130.25],
        dtype=np.float64,
    )
    expected = coinc.time_coincidence(
        host_times1, host_times2, window=0.2, slide_step=slide_step
    )
    times1 = host_times1.astype(dtype)
    times2 = host_times2.astype(dtype)

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("time coincidence used the NumPy/Cython path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("time coincidence copied Torch data to host")

    with ctx:
        torch_times1 = Array(times1)
        torch_times2 = Array(times2)
        with monkeypatch.context() as patch:
            patch.setattr(
                coinc, "timecoincidence_constructfold", reject_host_path
            )
            patch.setattr(
                coinc, "timecoincidence_findidxlen", reject_host_path
            )
            patch.setattr(
                coinc, "timecoincidence_constructidxs", reject_host_path
            )
            patch.setattr(
                coinc, "timecoincidence_getslideint", reject_host_path
            )
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            actual = coinc.time_coincidence(
                torch_times1,
                torch_times2,
                window=0.2,
                slide_step=slide_step,
            )

    assert all(isinstance(value, Array) for value in actual)
    assert all(value._data.tensor.device.type == device for value in actual)
    assert actual[0]._data.tensor.dtype == torch.int64
    assert actual[1]._data.tensor.dtype == torch.int64
    assert actual[2]._data.tensor.dtype == torch.int32
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(
            value._data.tensor.detach().cpu().numpy(), reference
        )


def test_time_coincidence_raw_torch_empty_and_slide_rounding():
    times1 = torch.tensor([105.0, 115.0, 125.0], dtype=torch.float64)
    times2 = torch.tensor([100.0, 110.0, 130.0], dtype=torch.float64)
    expected = coinc.time_coincidence(
        times1.numpy(), times2.numpy(), window=5.1, slide_step=10.0
    )
    actual = coinc.time_coincidence(
        times1, times2, window=5.1, slide_step=10.0
    )

    assert all(isinstance(value, torch.Tensor) for value in actual)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value.numpy(), reference)

    empty = coinc.time_coincidence(
        torch.empty(0, dtype=torch.float64),
        times2,
        window=0.1,
        slide_step=10.0,
    )
    assert all(value.numel() == 0 for value in empty)


@pytest.mark.parametrize(
    "function_name", ("indices_within_times", "indices_outside_times")
)
def test_veto_time_indices_stay_on_torch_device(
        torch_device_ctx, monkeypatch, function_name):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_times = np.array(
        [5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 1.5], dtype=np.float64
    )
    starts = np.array([4.0, 2.5, 2.0, 9.0], dtype=np.float64)
    ends = np.array([5.5, 1.0, 3.0, 9.0], dtype=np.float64)
    function = getattr(veto, function_name)
    expected = function(host_times, starts, ends)

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("veto indexing used its host segment path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("veto indexing synchronized Torch data")

    with ctx:
        torch_times = Array(host_times.astype(dtype))
        torch_starts = Array(starts.astype(dtype))
        torch_ends = Array(ends.astype(dtype))
        with monkeypatch.context() as patch:
            patch.setattr(veto, "start_end_to_segments", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = function(torch_times, torch_starts, torch_ends)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )


def test_veto_time_indices_raw_torch_empty_and_validation():
    times = np.array([5.0, 1.0, 2.0, 3.0, 4.0, 1.5])
    starts = np.array([4.0, 2.5, 2.0])
    ends = np.array([5.5, 1.0, 3.0])
    expected_within = veto.indices_within_times(times, starts, ends)
    expected_outside = veto.indices_outside_times(times, starts, ends)

    torch_times = torch.tensor(times)
    actual_within = veto.indices_within_times(
        torch_times, starts, ends
    )
    actual_outside = veto.indices_outside_times(
        torch_times, starts, ends
    )
    assert isinstance(actual_within, torch.Tensor)
    assert isinstance(actual_outside, torch.Tensor)
    np.testing.assert_array_equal(actual_within.numpy(), expected_within)
    np.testing.assert_array_equal(actual_outside.numpy(), expected_outside)

    empty = np.array([], dtype=np.float64)
    assert veto.indices_within_times(torch_times, empty, empty).numel() == 0
    np.testing.assert_array_equal(
        veto.indices_outside_times(torch_times, empty, empty).numpy(),
        np.arange(len(times)),
    )

    with pytest.raises(ValueError, match="start and end arrays must match"):
        veto.indices_within_times(torch_times, starts, ends[:-1])
    with pytest.raises(TypeError, match="one-dimensional numeric"):
        veto.indices_within_times(
            torch_times.to(torch.complex128), starts, ends
        )


def test_veto_segment_file_indices_stay_on_torch_device(
        torch_ctx, monkeypatch):
    times = np.array([5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 1.5])

    def fake_select(filename, _segment_name, _ifo):
        bounds = {
            "first": [(4.0, 5.5)],
            "second": [(2.5, 1.0), (2.0, 3.0)],
        }
        return veto.segmentlist(
            [veto.segment(start, end) for start, end in bounds[filename]]
        )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("segment-file veto indices left Torch")

    expected_within = np.array([0, 1, 2, 4, 6])
    expected_outside = np.array([3, 5])
    with torch_ctx:
        torch_times = Array(times)
        with monkeypatch.context() as patch:
            patch.setattr(
                veto, "select_segments_by_definer", fake_select
            )
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            within, within_segments = veto.indices_within_segments(
                torch_times, ("first", "second")
            )
            outside, outside_segments = veto.indices_outside_segments(
                torch_times, ("first", "second")
            )

    assert isinstance(within, Array)
    assert isinstance(outside, Array)
    assert within._data.tensor.device.type == "cpu"
    assert outside._data.tensor.device.type == "cpu"
    np.testing.assert_array_equal(
        within._data.tensor.detach().cpu().numpy(), expected_within
    )
    np.testing.assert_array_equal(
        outside._data.tensor.detach().cpu().numpy(), expected_outside
    )
    assert list(within_segments) == list(outside_segments)


@pytest.mark.parametrize("method", ("python", "cython"))
def test_cluster_over_time_stays_on_torch_device(
        torch_device_ctx, monkeypatch, method):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_stat = np.array(
        [5.0, 2.0, 7.0, 7.0, -1.0, 4.0, 8.0, 3.0],
        dtype=np.float64,
    )
    host_time = np.array(
        [4.0, 0.0, 1.0, 1.25, 8.0, 4.25, 7.75, 12.0],
        dtype=np.float64,
    )
    expected = coinc.cluster_over_time(
        host_stat, host_time, window=0.5, method=method
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("clustering used the NumPy/Cython path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("clustering copied or synchronized Torch data")

    with ctx:
        torch_stat = Array(host_stat.astype(dtype))
        torch_time = Array(host_time.astype(dtype))
        with monkeypatch.context() as patch:
            patch.setattr(coinc, "timecluster_cython", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = coinc.cluster_over_time(
                torch_stat, torch_time, window=0.5, method=method
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )


def test_cluster_over_time_raw_torch_empty_nan_and_validation():
    times = np.array([0.0, 0.1, 0.2, 2.0], dtype=np.float64)
    statistics = (
        np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64),
        np.array([np.nan, 4.0, 3.0, 2.0], dtype=np.float64),
        np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float64),
    )
    for method in ("python", "cython"):
        for stat in statistics:
            expected = coinc.cluster_over_time(
                stat, times, window=0.5, method=method
            )
            actual = coinc.cluster_over_time(
                torch.tensor(stat),
                torch.tensor(times),
                window=0.5,
                method=method,
            )
            assert isinstance(actual, torch.Tensor)
            np.testing.assert_array_equal(actual.numpy(), expected)

    empty = coinc.cluster_over_time(
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        window=0.5,
    )
    assert isinstance(empty, torch.Tensor)
    assert empty.dtype == torch.int64
    assert empty.numel() == 0

    with pytest.raises(NotImplementedError):
        coinc.cluster_over_time(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            window=0.5,
            argmax=lambda value: value.argmax(),
        )


@pytest.mark.parametrize("method", ("python", "cython"))
@pytest.mark.parametrize("slide", (10.0, np.inf))
def test_cluster_coincs_stays_on_torch_device(
        torch_device_ctx, monkeypatch, method, slide):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_stat = np.array(
        [3.0, 8.0, 7.0, 4.0, 9.0, 2.0, 6.0, 5.0],
        dtype=np.float64,
    )
    slide_ids = np.array([0, 0, 0, 1, 1, 1, -1, -1], dtype=np.int32)
    pivot_times = np.array(
        [1000.0, 1000.2, 1002.0, 1000.0, 1000.2, 1003.0,
         1000.0, 1000.1],
        dtype=np.float64,
    )
    applied_slide = slide if np.isfinite(slide) else 0.0
    other_times = pivot_times - slide_ids * applied_slide
    expected = coinc.cluster_coincs(
        host_stat,
        pivot_times,
        other_times,
        slide_ids,
        slide=slide,
        window=0.5,
        method=method,
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("coincidence clustering used its host path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("coincidence clustering synchronized Torch data")

    with ctx:
        torch_inputs = (
            Array(host_stat.astype(dtype)),
            Array(pivot_times.astype(dtype)),
            Array(other_times.astype(dtype)),
            Array(slide_ids.astype(dtype if device == "mps" else np.int32)),
        )
        with monkeypatch.context() as patch:
            patch.setattr(coinc, "timecluster_cython", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = coinc.cluster_coincs(
                *torch_inputs,
                slide=slide,
                window=0.5,
                method=method,
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )


def test_cluster_coincs_multiifo_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_stat = np.array([3.0, 8.0, 7.0, 4.0, 9.0, 2.0])
    slide_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    host_times = (
        np.array([1000.0, 1000.2, 1002.0, 1010.0, 1010.2, 1013.0]),
        np.array([1000.02, 1000.18, -1.0, 1000.02, 1000.18, -1.0]),
        np.array([999.98, -1.0, 1002.05, 999.98, -1.0, 1003.05]),
    )
    expected = coinc.cluster_coincs_multiifo(
        host_stat,
        host_times,
        slide_ids,
        slide=10.0,
        window=0.5,
        method="cython",
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("multi-IFO clustering used its host path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("multi-IFO clustering synchronized Torch data")

    with ctx:
        torch_stat = Array(host_stat.astype(dtype))
        torch_times = tuple(Array(value.astype(dtype)) for value in host_times)
        torch_slides = Array(
            slide_ids.astype(dtype if device == "mps" else np.int32)
        )
        with monkeypatch.context() as patch:
            patch.setattr(coinc, "mean_if_greater_than_zero", reject_host_path)
            patch.setattr(coinc, "timecluster_cython", reject_host_path)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = coinc.cluster_coincs_multiifo(
                torch_stat,
                torch_times,
                torch_slides,
                slide=10.0,
                window=0.5,
                method="cython",
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )


@pytest.mark.parametrize(
    "distribution", ("exponential", "rayleigh", "power")
)
def test_trigger_fits_stay_on_torch_device(
        torch_device_ctx, monkeypatch, distribution):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    threshold = 4.0
    values = np.array(
        [3.5, 4.0, 4.4, 4.9, 5.7, 6.8, 8.1], dtype=dtype
    )
    weights = np.array(
        [0.5, 1.0, 1.5, 0.75, 2.0, 1.25, 0.8], dtype=dtype
    )
    xvals = np.array([3.75, 4.0, 4.6, 5.5, 7.25], dtype=dtype)

    expected_alpha, expected_sigma = trigger_fits.fit_above_thresh(
        distribution,
        values,
        thresh=threshold,
        weights=weights,
    )
    expected_fit = trigger_fits.fit_fn(
        distribution,
        xvals,
        expected_alpha,
        threshold,
    )
    expected_cumulative = trigger_fits.cum_fit(
        distribution,
        xvals,
        expected_alpha,
        threshold,
    )
    expected_ks = trigger_fits.KS_test(
        distribution,
        values,
        expected_alpha,
        threshold,
    )
    expected_ks_default_threshold = trigger_fits.KS_test(
        distribution,
        values,
        expected_alpha,
    )
    expected_tail_threshold = trigger_fits.tail_threshold(values, N=3)

    with ctx:
        values_tensor = torch.tensor(
            values,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        weights_tensor = torch.tensor(
            weights,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        xvals_tensor = torch.tensor(
            xvals,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        torch_values = Array(TorchArrayData(values_tensor), copy=False)
        torch_weights = Array(TorchArrayData(weights_tensor), copy=False)
        torch_xvals = Array(TorchArrayData(xvals_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("trigger fit vector left Torch")

        def reject_numpy_fit_math(*_args, **_kwargs):
            raise AssertionError("trigger fit used NumPy arithmetic")

        numpy_array = trigger_fits.numpy.array

        def reject_torch_numpy_array(value, *args, **kwargs):
            if isinstance(value, (Array, TorchArrayData, torch.Tensor)):
                raise AssertionError("trigger fit vector left Torch")
            return numpy_array(value, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(
                trigger_fits.numpy,
                "array",
                reject_torch_numpy_array,
            )
            for name in ("average", "exp", "log", "putmask"):
                patch.setattr(
                    trigger_fits.numpy,
                    name,
                    reject_numpy_fit_math,
                )
            patch.setattr(
                trigger_fits, "kstest", reject_numpy_fit_math
            )
            alpha, sigma = trigger_fits.fit_above_thresh(
                distribution,
                torch_values,
                thresh=threshold,
                weights=torch_weights,
            )
            fit = trigger_fits.fit_fn(
                distribution,
                torch_xvals,
                alpha,
                threshold,
            )
            cumulative = trigger_fits.cum_fit(
                distribution,
                torch_xvals,
                alpha,
                threshold,
            )
            actual_ks = trigger_fits.KS_test(
                distribution,
                torch_values,
                alpha,
                threshold,
            )
            actual_ks_default_threshold = trigger_fits.KS_test(
                distribution,
                torch_values,
                alpha,
            )
            actual_tail_threshold = trigger_fits.tail_threshold(
                torch_values,
                N=3,
            )
            loss = alpha + sigma + actual_tail_threshold
            loss = loss + fit._data.tensor.sum()
            loss = loss + cumulative._data.tensor.sum()
            loss.backward()

    assert isinstance(fit, Array)
    assert isinstance(cumulative, Array)
    assert isinstance(actual_tail_threshold, torch.Tensor)
    assert fit._data.tensor.device.type == device
    assert cumulative._data.tensor.device.type == device
    assert actual_tail_threshold.device.type == device
    assert fit._data.tensor.dtype == torch_dtype
    assert cumulative._data.tensor.dtype == torch_dtype
    tolerance = 5e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        alpha.detach().cpu().numpy(),
        expected_alpha,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        sigma.detach().cpu().numpy(),
        expected_sigma,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        fit._data.tensor.detach().cpu().numpy(),
        expected_fit,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        cumulative._data.tensor.detach().cpu().numpy(),
        expected_cumulative,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_ks.statistic,
        expected_ks.statistic,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_ks.pvalue,
        expected_ks.pvalue,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_ks.statistic_location,
        expected_ks.statistic_location,
        rtol=tolerance,
        atol=tolerance,
    )
    assert actual_ks.statistic_sign == expected_ks.statistic_sign
    np.testing.assert_allclose(
        actual_ks_default_threshold.statistic,
        expected_ks_default_threshold.statistic,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_ks_default_threshold.pvalue,
        expected_ks_default_threshold.pvalue,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_ks_default_threshold.statistic_location,
        expected_ks_default_threshold.statistic_location,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_tail_threshold.detach().cpu().numpy(),
        expected_tail_threshold,
        rtol=tolerance,
        atol=tolerance,
    )
    assert (
        actual_ks_default_threshold.statistic_sign
        == expected_ks_default_threshold.statistic_sign
    )
    for tensor in (values_tensor, weights_tensor, xvals_tensor):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize(
    ("method", "method_kwargs"),
    (
        ("n_louder", {}),
        (
            "trigger_fit",
            {"fit_function": "exponential", "fit_threshold": 4.0},
        ),
    ),
)
def test_significance_far_stays_on_torch_device(
        torch_device_ctx, monkeypatch, method, method_kwargs):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    background_time = 1234.5
    back_stat = np.array(
        [1.0, 1.7, 2.2, 3.1, 3.8, 4.2, 4.8, 5.6, 6.5],
        dtype=dtype,
    )
    fore_stat = np.array([1.4, 3.4, 4.4, 5.2, 7.0], dtype=dtype)
    dec_facs = np.array(
        [1.0, 2.0, 1.5, 0.5, 2.5, 1.0, 2.0, 1.5, 0.75],
        dtype=dtype,
    )
    expected_bg, expected_fg, expected_info = significance.get_far(
        back_stat,
        fore_stat,
        dec_facs,
        background_time,
        method=method,
        **method_kwargs,
    )

    with ctx:
        back_tensor = torch.tensor(
            back_stat,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        fore_tensor = torch.tensor(
            fore_stat,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        dec_tensor = torch.tensor(
            dec_facs,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        torch_back = Array(TorchArrayData(back_tensor), copy=False)
        torch_fore = Array(TorchArrayData(fore_tensor), copy=False)
        torch_dec = Array(TorchArrayData(dec_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("significance vector left Torch")

        def reject_numpy_significance_math(*_args, **_kwargs):
            raise AssertionError("significance used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            for name in (
                "argsort",
                "isnan",
                "logical_not",
                "searchsorted",
                "sum",
                "where",
                "zeros_like",
            ):
                patch.setattr(
                    significance.np,
                    name,
                    reject_numpy_significance_math,
                )
            for name in ("array", "average", "exp", "log", "putmask"):
                patch.setattr(
                    trigger_fits.numpy,
                    name,
                    reject_numpy_significance_math,
                )
            actual_bg, actual_fg, actual_info = significance.get_far(
                torch_back,
                torch_fore,
                torch_dec,
                background_time,
                method=method,
                **method_kwargs,
            )
            loss = actual_bg._data.tensor.sum()
            loss = loss + actual_fg._data.tensor.sum()
            loss.backward()

    assert isinstance(actual_bg, Array)
    assert isinstance(actual_fg, Array)
    assert actual_bg._data.tensor.device.type == device
    assert actual_fg._data.tensor.device.type == device
    assert actual_bg._data.tensor.dtype == torch_dtype
    assert actual_fg._data.tensor.dtype == torch_dtype
    tolerance = 5e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual_bg._data.tensor.detach().cpu().numpy(),
        expected_bg,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_fg._data.tensor.detach().cpu().numpy(),
        expected_fg,
        rtol=tolerance,
        atol=tolerance,
    )
    assert set(actual_info) == set(expected_info)
    for key, value in actual_info.items():
        np.testing.assert_allclose(
            value.detach().cpu().numpy(),
            expected_info[key],
            rtol=tolerance,
            atol=tolerance,
        )
    assert dec_tensor.grad is not None
    assert torch.isfinite(dec_tensor.grad).all()
    assert torch.count_nonzero(dec_tensor.grad) > 0
    if method == "trigger_fit":
        assert back_tensor.grad is not None
        assert fore_tensor.grad is not None
        assert torch.isfinite(back_tensor.grad).all()
        assert torch.isfinite(fore_tensor.grad).all()
        assert torch.count_nonzero(back_tensor.grad) > 0
        assert torch.count_nonzero(fore_tensor.grad) > 0
    else:
        assert back_tensor.grad is None
        assert fore_tensor.grad is None


@pytest.mark.parametrize("combo_kind", ("single", "per_event"))
def test_significance_far_limit_stays_on_torch_device(
        torch_device_ctx, monkeypatch, combo_kind):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    significance_dict = {
        "H1L1": {"far_limit": 0.5},
        "H1V1": {"far_limit": 0.2},
        "L1V1": {"far_limit": 0.0},
    }
    if combo_kind == "single":
        combo = "H1L1"
        far = np.array([0.1, 0.4, 0.7, 1.2], dtype=dtype)
    else:
        combo = np.array([b"H1L1", b"H1V1", b"H1L1", b"L1V1"])
        far = np.array([0.1, 0.1, 0.7, 0.01], dtype=dtype)
    expected = significance.apply_far_limit(far, significance_dict, combo)

    with ctx:
        far_tensor = torch.tensor(
            far,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        torch_far = Array(TorchArrayData(far_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("FAR vector left Torch")

        def reject_numpy_maximum(*_args, **_kwargs):
            raise AssertionError("FAR limit used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(
                significance.np,
                "maximum",
                reject_numpy_maximum,
            )
            actual = significance.apply_far_limit(
                torch_far,
                significance_dict,
                combo,
            )
            actual._data.tensor.sum().backward()

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch_dtype
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=0,
        atol=0,
    )
    assert far_tensor.grad is not None
    assert torch.isfinite(far_tensor.grad).all()
    assert torch.count_nonzero(far_tensor.grad) > 0


@pytest.mark.parametrize(
    "dtype", (np.bool_, np.int32, np.int64, np.uint32)
)
def test_bitwise_operators_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.uint32:
        pytest.skip("MPS PyCBC arrays do not support uint32 storage")
    values = np.array([0, 1, 2, 3], dtype=dtype)
    others = np.array([1, 0, 3, 2], dtype=dtype)

    def _reject_host_transfer(_self):
        raise AssertionError("bitwise operation transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (lambda: ~series, np.invert(values)),
            (lambda: series & other_series,
             np.bitwise_and(values, others)),
            (lambda: series | other_series,
             np.bitwise_or(values, others)),
            (lambda: series ^ other_series,
             np.bitwise_xor(values, others)),
            (lambda: 1 | series, np.bitwise_or(1, values)),
            (lambda: np.bitwise_and(series, other_series),
             np.bitwise_and(values, others)),
        )

        for operation, expected in cases:
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData, "numpy", _reject_host_transfer
                )
                actual = operation()

            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.dtype == expected.dtype
            assert actual.delta_t == series.delta_t
            assert actual.start_time == series.start_time
            np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_bitwise_operators_reject_nonintegral_torch_arrays(torch_ctx, dtype):
    with torch_ctx:
        array = Array(np.array([0, 1], dtype=dtype))
        with pytest.raises(TypeError):
            ~array
        with pytest.raises(TypeError):
            array & 1


def test_array_direct_sum_keeps_legacy_accumulation_precision(torch_ctx):
    values = np.array([1.0e8, 1.0, -1.0e8], dtype=np.float32)

    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25)
        direct = series.sum()
        numpy_sum = np.sum(series)

    assert direct == 1.0
    assert numpy_sum == np.sum(values)


def test_torch_findchirp_clustering_matches_cython(torch_ctx):
    rng = np.random.default_rng(9321)

    with torch_ctx:
        for _ in range(100):
            count = int(rng.integers(1, 150))
            times = np.cumsum(rng.integers(1, 8, size=count)).astype(
                np.int64
            )
            # Integer magnitudes deliberately exercise the earlier-tie rule.
            values = rng.integers(0, 12, size=count).astype(np.float32)
            values *= rng.choice((-1.0, 1.0), size=count).astype(np.float32)
            values = values.astype(np.complex64)
            window = int(rng.integers(1, 40))

            expected = events.findchirp_cluster_over_window(
                times, values, window
            )
            actual = threshold_torch._findchirp_cluster_indices(
                torch.as_tensor(times), torch.as_tensor(values), window
            )

            np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_findchirp_cluster_reduce_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    time_dtype = np.float32 if device == "mps" else np.int32
    times = np.array(
        [1, 2, 4, 5, 9, 10, 13, 17], dtype=time_dtype
    )
    values = np.array(
        [1.0, -5.0, 4.0j, 3.0, -2.0j, 7.0, -6.0, 2.0j],
        dtype=np.complex64,
    )
    survivor_positions = events.findchirp_cluster_over_window(
        times, values, 4
    )
    expected_times = times[survivor_positions]
    expected_values = values[survivor_positions]

    def _reject_host_transfer(_self):
        raise AssertionError("FindChirp copied full candidate arrays to host")

    with ctx:
        time_array = Array(times)
        value_array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_positions = events.findchirp_cluster_over_window(
                time_array, value_array, 4
            )
            actual_times, actual_values = events.cluster_reduce(
                time_array, value_array, 4
            )

    np.testing.assert_array_equal(actual_positions, survivor_positions)
    assert actual_times._data.tensor.device.type == device
    assert actual_values._data.tensor.device.type == device
    np.testing.assert_array_equal(
        actual_times._data.tensor.detach().cpu().numpy(), expected_times
    )
    np.testing.assert_array_equal(
        actual_values._data.tensor.detach().cpu().numpy(), expected_values
    )


@pytest.mark.parametrize(
    "statistic, extrinsic",
    (
        (
            matchedfilter.compute_max_snr_over_sky_loc_stat,
            matchedfilter.compute_u_val_for_sky_loc_stat,
        ),
        (
            matchedfilter.compute_max_snr_over_sky_loc_stat_no_phase,
            matchedfilter.compute_u_val_for_sky_loc_stat_no_phase,
        ),
    ),
)
def test_sky_max_statistics_stay_on_torch_device(
        torch_device_ctx, monkeypatch, statistic, extrinsic):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(8128)
    hplus = (
        rng.normal(size=32) + 1j * rng.normal(size=32)
    ).astype(np.complex64)
    hcross = (
        rng.normal(size=32) + 1j * rng.normal(size=32)
    ).astype(np.complex64)
    overlap = 0.23
    hpnorm = 0.8
    hcnorm = 1.1
    threshold = 1.4
    analyse_slice = slice(2, 30)
    indices = np.array([2, 7, 13, 21, 28])

    expected = statistic(
        hplus,
        hcross,
        overlap,
        hpnorm=hpnorm,
        hcnorm=hcnorm,
    ).numpy()
    expected_thresholded = statistic(
        Array(hplus),
        Array(hcross),
        overlap,
        hpnorm=hpnorm,
        hcnorm=hcnorm,
        thresh=threshold,
        analyse_slice=analyse_slice,
    )
    expected_threshold_locs = expected_thresholded.non_zero_locs.copy()
    expected_thresholded = expected_thresholded.numpy()
    expected_u, expected_phase = extrinsic(
        hplus,
        hcross,
        overlap,
        hpnorm=hpnorm,
        hcnorm=hcnorm,
        indices=indices,
    )

    with ctx:
        hplus_torch = Array(hplus)
        hcross_torch = Array(hcross)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("sky-max series transferred to NumPy")

            def _reject_public_threshold(*_args, **_kwargs):
                raise AssertionError("sky-max candidates left Torch")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(events, "threshold_only", _reject_public_threshold)
            actual = statistic(
                hplus_torch,
                hcross_torch,
                overlap,
                hpnorm=hpnorm,
                hcnorm=hcnorm,
            )
            actual_thresholded = statistic(
                hplus_torch,
                hcross_torch,
                overlap,
                hpnorm=hpnorm,
                hcnorm=hcnorm,
                thresh=threshold,
                analyse_slice=analyse_slice,
            )
            actual_threshold_tensor = (
                actual_thresholded._data.tensor.clone()
            )
            actual_threshold_locs_tensor = (
                actual_thresholded.non_zero_locs.clone()
            )
            reused = statistic(
                hplus_torch,
                hcross_torch,
                overlap,
                hpnorm=hpnorm,
                hcnorm=hcnorm,
                out=actual_thresholded,
                thresh=1e6,
                analyse_slice=analyse_slice,
            )
            actual_u, actual_phase = extrinsic(
                hplus_torch,
                hcross_torch,
                overlap,
                hpnorm=hpnorm,
                hcnorm=hcnorm,
                indices=indices,
            )
            actual_u_tensor = actual_u._data.tensor.clone()
            actual_phase_tensor = actual_phase._data.tensor.clone()

        assert actual._data.tensor.device.type == device
        assert reused is actual_thresholded
        assert actual_thresholded._data.tensor.device.type == device
        assert isinstance(actual_thresholded.non_zero_locs, torch.Tensor)
        assert actual_thresholded.non_zero_locs.device.type == device
        assert actual_thresholded.non_zero_locs.numel() == 0
        assert torch.count_nonzero(actual_thresholded._data.tensor).item() == 0
        assert isinstance(actual_u._data, TorchArrayData)
        assert actual_u._data.tensor.device.type == device
        assert isinstance(actual_phase._data, TorchArrayData)
        assert actual_phase._data.tensor.device.type == device
        actual_threshold_locs = (
            actual_threshold_locs_tensor.detach().cpu().numpy()
        )
        actual_values = actual.numpy()
        actual_thresholded_values = (
            actual_threshold_tensor.detach().cpu().numpy()
        )
        actual_u_values = actual_u_tensor.detach().cpu().numpy()
        actual_phase_values = actual_phase_tensor.detach().cpu().numpy()

    np.testing.assert_allclose(actual_values, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        actual_thresholded_values,
        expected_thresholded,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        actual_threshold_locs, expected_threshold_locs
    )
    np.testing.assert_allclose(
        actual_u_values, expected_u, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        actual_phase_values, expected_phase, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize(
    "dtype, values, other_values, scalar",
    (
        (
            np.float32,
            (-2.0, 0.5, 3.0),
            (0.5, 0.0, 4.0),
            0.5,
        ),
        (
            np.float64,
            (-2.0, 0.5, 3.0),
            (0.5, 0.0, 4.0),
            0.5,
        ),
        (
            np.complex64,
            (1 + 2j, 1 + 1j, 2 + 0j),
            (1 + 1.5j, 1 + 1j, 3 + 0j),
            1 + 1.5j,
        ),
        (
            np.complex128,
            (1 + 2j, 1 + 1j, 2 + 0j),
            (1 + 1.5j, 1 + 1j, 3 + 0j),
            1 + 1.5j,
        ),
        (
            np.int32,
            (-2**31, 0, 2**31 - 1),
            (-1, 0, 1),
            2**40,
        ),
        (
            np.uint32,
            (0, 1, 2**32 - 1),
            (1, 1, 2**32 - 2),
            -1,
        ),
    ),
)
def test_array_comparisons_stay_on_device(
        torch_device_ctx, monkeypatch, dtype, values, other_values, scalar):
    ctx, device = torch_device_ctx
    if device == "mps" and (
        dtype in (np.float64, np.complex128)
        or np.dtype(dtype).kind in "iu"
    ):
        pytest.skip("Torch MPS does not support this dtype")

    values = np.asarray(values, dtype=dtype)
    other_values = np.asarray(other_values, dtype=dtype)
    operations = (
        operator.lt,
        operator.le,
        operator.ne,
        operator.gt,
        operator.ge,
    )
    expected_scalar = [operation(values, scalar) for operation in operations]
    expected_array = [
        operation(values, other_values) for operation in operations
    ]

    with ctx:
        array = Array(values)
        other = Array(other_values)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Comparison copied numeric data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_scalar = [
                operation(array, scalar) for operation in operations
            ]
            actual_array = [
                operation(array, other) for operation in operations
            ]

    assert array._data.tensor.device.type == device
    for actual, expected in zip(
            actual_scalar + actual_array,
            expected_scalar + expected_array,
    ):
        assert isinstance(actual, (np.ndarray, Array))
        assert actual.dtype == np.bool_
        np.testing.assert_array_equal(actual.numpy() if isinstance(actual, Array) else actual, expected)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_array_equality_only_transfers_boolean(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([1.0, -2.0, 4.0], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.25j, -0.5j, 0.75j], dtype=dtype)
    changed = values.copy()
    changed[-1] += 1

    def _reject_host_transfer(_self):
        raise AssertionError("array equality transferred full data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        equal = TimeSeries(values, delta_t=0.25, epoch=10.0)
        unequal = TimeSeries(changed, delta_t=0.25, epoch=10.0)
        metadata_mismatch = TimeSeries(values, delta_t=0.5, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", _reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            assert series == equal
            assert (series == unequal) is False
            assert (series == metadata_mismatch) is False


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_array_tolerance_comparisons_only_transfer_boolean(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([1.0, -2.0, 4.0], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.25j, -0.5j, 0.75j], dtype=dtype)
    relative_close = values * np.array(1.05, dtype=dtype)
    relative_far = values * np.array(1.2, dtype=dtype)
    absolute_close = values.copy()
    absolute_close[0] += np.array(0.05, dtype=dtype)
    absolute_far = values.copy()
    absolute_far[0] += np.array(0.2, dtype=dtype)

    def _reject_host_transfer(_self):
        raise AssertionError(
            "array tolerance comparison transferred full data to NumPy"
        )

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        rel_close = TimeSeries(relative_close, delta_t=0.25, epoch=10.0)
        rel_far = TimeSeries(relative_far, delta_t=0.25, epoch=10.0)
        abs_close = TimeSeries(absolute_close, delta_t=0.25, epoch=10.0)
        abs_far = TimeSeries(absolute_far, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", _reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            assert series.almost_equal_elem(rel_close, 0.1)
            assert not series.almost_equal_elem(rel_far, 0.1)
            assert series.almost_equal_elem(abs_close, 0.1, relative=False)
            assert not series.almost_equal_elem(abs_far, 0.1, relative=False)
            assert series.almost_equal_norm(rel_close, 0.1)
            assert not series.almost_equal_norm(rel_far, 0.1)
            assert series.almost_equal_norm(abs_close, 0.1, relative=False)
            assert not series.almost_equal_norm(abs_far, 0.1, relative=False)


def test_status_buffer_validation_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support integer status arrays")

    cases = (
        (True, (0, 0), None, True),
        (True, (0, 1), None, False),
        (False, (3, 7), None, True),
        (False, (3, 1), None, False),
        (False, (4, 5), 4, True),
        (False, (), None, True),
    )
    status = StatusBuffer.__new__(StatusBuffer)
    status.valid_mask = 3

    with ctx:
        arrays = [
            Array(np.asarray(values, dtype=np.int32))
            for _, values, _, _ in cases
        ]
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Status validation copied data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = []
            for (valid_on_zero, _, flag, _), values in zip(cases, arrays):
                status.valid_on_zero = valid_on_zero
                actual.append(status.check_valid(values, flag=flag))

    assert actual == [expected for _, _, _, expected in cases]
    assert all(array._data.tensor.device.type == device for array in arrays)


@pytest.mark.parametrize(
    "valid_on_zero, status_values",
    (
        (True, (0, 0, 1, 0, 0, 2, 0, 0)),
        (False, (3, 7, 1, 3, 3, 2, 7, 3)),
    ),
)
def test_status_buffer_flag_indices_stay_on_device(
        torch_device_ctx, monkeypatch, valid_on_zero, status_values):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support integer status arrays")

    times = np.asarray(
        (100.75, 101.0, 101.49, 101.5, 102.5, 102.99, 103.0)
    )
    status = StatusBuffer.__new__(StatusBuffer)
    status.valid_mask = 3
    status.valid_on_zero = valid_on_zero
    status.raw_buffer = TimeSeries(
        np.asarray(status_values, dtype=np.int32),
        delta_t=0.5,
        epoch=100,
    )
    expected = status.indices_of_flag(100.5, 3.5, times)

    with ctx:
        status.raw_buffer = TimeSeries(
            np.asarray(status_values, dtype=np.int32),
            delta_t=0.5,
            epoch=100,
        )
        with monkeypatch.context() as patch:
            original_cpu = torch.Tensor.cpu
            transferred_positions = []

            def _reject_host_transfer(_self):
                raise AssertionError("Flag selection copied full data to host")

            def _guard_position_transfer(tensor, *args, **kwargs):
                if tensor.dtype not in (torch.int32, torch.int64):
                    raise AssertionError(
                        "Flag selection copied sample timestamps to host"
                    )
                transferred_positions.append(tensor.numel())
                return original_cpu(tensor, *args, **kwargs)

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _guard_position_transfer)
            actual = status.indices_of_flag(100.5, 3.5, times)

    np.testing.assert_array_equal(actual, expected)
    assert transferred_positions == [2]
    assert status.raw_buffer._data.tensor.device.type == device


def test_idq_flag_selection_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    fap_values = np.asarray(
        (0.9, 0.2, 0.1, 0.8, 0.05, 0.4, 0.3, 0.2),
        dtype=np.float32,
    )
    valid_values = np.asarray(
        (1, 1, 0, 1, 1, 0, 1, 1),
        dtype=np.float32,
    )
    times = np.asarray(
        (100.2, 100.25, 100.75, 101.0, 101.74, 101.75, 102.25, 102.5)
    )

    idq = iDQBuffer.__new__(iDQBuffer)
    idq.threshold = 0.25
    idq.idq = types.SimpleNamespace(
        raw_buffer=TimeSeries(fap_values, delta_t=0.5, epoch=100)
    )
    idq.idq_state = types.SimpleNamespace(
        raw_buffer=TimeSeries(valid_values, delta_t=0.5, epoch=100)
    )
    expected = idq.flag_at_times(101.0, 3.0, times, padding=0.25)

    with ctx:
        idq.idq.raw_buffer = TimeSeries(
            fap_values, delta_t=0.5, epoch=100
        )
        idq.idq_state.raw_buffer = TimeSeries(
            valid_values, delta_t=0.5, epoch=100
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("iDQ selection copied full data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = idq.flag_at_times(101.0, 3.0, times, padding=0.25)

    np.testing.assert_array_equal(actual, expected)
    assert idq.idq.raw_buffer._data.tensor.device.type == device
    assert idq.idq_state.raw_buffer._data.tensor.device.type == device


@pytest.mark.parametrize(
    "epoch, delta_t, orbit_start, orbit_end",
    (
        (100.0, 0.1, 100.25, 100.75),
        (100.0, 0.1, 100.3, 100.7),
        (
            1_000_000_000.1234568,
            1.0 / 4096,
            1_000_000_000.124,
            1_000_000_000.125,
        ),
    ),
)
def test_space_signal_trimming_stays_on_device(
        torch_device_ctx, monkeypatch, epoch, delta_t,
        orbit_start, orbit_end):
    ctx, device = torch_device_ctx
    if device == "mps" and delta_t < 1.0:
        pytest.skip("Torch MPS does not support float64 sample-time grids")

    plus = np.arange(16, dtype=np.float64)
    cross = plus + 100.0
    sample_times = np.arange(len(plus)) * delta_t + epoch
    end_idx = np.flatnonzero(sample_times <= orbit_end)[-1]
    start_idx = np.flatnonzero(sample_times[:end_idx] >= orbit_start)[0]
    expected_plus = plus[start_idx:end_idx]
    expected_cross = cross[start_idx:end_idx]
    expected_epoch = sample_times[start_idx]

    with ctx:
        hp = TimeSeries(plus, delta_t=delta_t, epoch=epoch)
        hc = TimeSeries(cross, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Signal trimming copied waveform data")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_hp, actual_hc = check_signal_times(
                hp,
                hc,
                orbit_start,
                orbit_end,
                offset=0,
            )

    assert actual_hp._data.tensor.device.type == device
    assert actual_hc._data.tensor.device.type == device
    assert float(actual_hp.start_time) == pytest.approx(expected_epoch)
    np.testing.assert_array_equal(
        actual_hp._data.tensor.detach().cpu().numpy(), expected_plus
    )
    np.testing.assert_array_equal(
        actual_hc._data.tensor.detach().cpu().numpy(), expected_cross
    )


def test_space_sample_times_do_not_materialize_device_grid(
        torch_ctx, monkeypatch):
    with torch_ctx:
        series = TimeSeries(
            np.arange(5, dtype=np.float64), delta_t=0.25, epoch=10.5
        )

        def _reject_host_transfer(_self):
            raise AssertionError("Sample-time construction copied device data")

        monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        times = space_detector._regular_sample_times_numpy(series)

    np.testing.assert_array_equal(times, [10.5, 10.75, 11.0, 11.25, 11.5])


def test_fastlisa_torch_to_cupy_uses_dlpack():
    tensor = types.SimpleNamespace(__dlpack__=lambda: "legacy-capsule")
    modern = types.SimpleNamespace(
        from_dlpack=lambda value: ("modern", value)
    )
    legacy = types.SimpleNamespace(
        fromDlpack=lambda value: ("legacy", value)
    )

    assert space_detector._torch_to_cupy(tensor, modern) == (
        "modern", tensor
    )
    assert space_detector._torch_to_cupy(tensor, legacy) == (
        "legacy", "legacy-capsule"
    )


def test_fastlisa_cupy_result_uses_dlpack(torch_ctx, monkeypatch):
    class FakeCudaArray:
        __cuda_array_interface__ = {}

        def __dlpack__(self):
            raise AssertionError("The patched DLPack importer should be used")

    tensor = torch.arange(4, dtype=torch.float64)
    monkeypatch.setattr(
        torch.utils.dlpack,
        "from_dlpack",
        lambda values: tensor,
    )

    with torch_ctx:
        result = space_detector._fastlisa_time_series(
            FakeCudaArray(), delta_t=0.5, epoch=12.0
        )

    assert result._data.tensor.data_ptr() == tensor.data_ptr()
    assert float(result.start_time) == 12.0
    assert result.delta_t == 0.5


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("custom_frequencies", (False, True))
def test_spa_compression_stays_on_device(
        torch_device_ctx, monkeypatch, dtype, custom_frequencies):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.complex128:
        pytest.skip("Torch MPS does not support complex128")

    size = 256
    delta_f = 0.25
    indices = np.arange(size)
    amplitude = np.zeros(size)
    amplitude[4:240] = np.linspace(0.2, 1.0, 236)
    phase = -0.001 * indices ** 2
    waveform = (amplitude * np.exp(1j * phase)).astype(dtype)
    frequencies = np.arange(size, dtype=np.float64) * delta_f
    expected = spa_compression(
        FrequencySeries(waveform, delta_f=delta_f),
        5.0,
        50.0,
        sample_frequencies=frequencies if custom_frequencies else None,
    )

    with ctx:
        torch_series = FrequencySeries(waveform, delta_f=delta_f)
        torch_frequencies = None
        if custom_frequencies:
            torch_frequencies = torch.arange(
                size,
                dtype=torch_series._data.tensor.real.dtype,
                device=torch_series._data.tensor.device,
            ) * delta_f

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("SPA compression copied data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = spa_compression(
                torch_series,
                5.0,
                50.0,
                sample_frequencies=torch_frequencies,
            )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=delta_f)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "interpolation",
    ("inline_linear", "inline_quadratic", "inline_cubic", "inline_quartic"),
)
def test_fd_decompression_inputs_stay_on_device(
        torch_device_ctx, monkeypatch, dtype, interpolation):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    sample_frequencies = np.linspace(0.0, 8.0, 17, dtype=dtype)
    amplitude = np.exp(-0.1 * sample_frequencies).astype(dtype)
    phase = (0.3 * sample_frequencies).astype(dtype)
    f_lower = 0.5
    delta_f = 0.25
    expected = fd_decompress(
        Array(amplitude),
        Array(phase),
        Array(sample_frequencies),
        df=delta_f,
        f_lower=f_lower,
        interpolation=interpolation,
    )

    with ctx:
        torch_amplitude = Array(amplitude)
        torch_phase = Array(phase)
        torch_frequencies = Array(sample_frequencies)

        def _reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("waveform decompression copied inputs to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(array_torch_module, "numpy", _reject_host_transfer)
            patch.setattr(np, "searchsorted", _reject_host_transfer)
            actual = fd_decompress(
                torch_amplitude,
                torch_phase,
                torch_frequencies,
                df=delta_f,
                f_lower=f_lower,
                interpolation=interpolation,
            )

    assert actual._data.tensor.device.type == device
    expected_dtype = torch.complex64 if dtype == np.float32 else torch.complex128
    assert actual._data.tensor.dtype == expected_dtype
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=1e-3,
        atol=5e-3,
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

    cpu_weights = []

    def capture_cpu_weights(values, size=None, *, host=True):
        cpu_weights.append(np.array(values, copy=True))
        return np.zeros(size, dtype=np.int64)

    with monkeypatch.context() as patch:
        patch.setattr(
            _INFERENCE_TOOLS, "draw_sample", capture_cpu_weights
        )
        np.random.seed(1234)
        zero_index_reference = make_marginalizer().draw_times(make_snrs())

    sample_calls = []
    original_draw_sample = _INFERENCE_TOOLS.draw_sample

    def record_draw_sample(values, size=None, *, host=True):
        result = original_draw_sample(values, size=size, host=host)
        sample_calls.append((host, values, result))
        return result

    selection_calls = []
    original_selected_values = _INFERENCE_TOOLS._selected_values

    def record_selected_values(values, indices, *, host=True):
        result = original_selected_values(values, indices, host=host)
        selection_calls.append((host, indices, result))
        return result

    with ctx:
        snrs = make_snrs()
        marginalizer = make_marginalizer()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Marginalization copied the complete SNR series to host"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                _INFERENCE_TOOLS, "draw_sample", record_draw_sample
            )
            patch.setattr(
                _INFERENCE_TOOLS,
                "_selected_values",
                record_selected_values,
            )
            np.random.seed(1234)
            torch.manual_seed(1234)
            actual = marginalizer.draw_times(snrs)

    for snr in snrs.values():
        assert snr._data.tensor.device.type == device
    assert len(sample_calls) == 1
    assert sample_calls[0][0] is False
    assert sample_calls[0][1].device.type == device
    assert sample_calls[0][2].device.type == device
    assert len(selection_calls) == 1
    assert selection_calls[0][0] is False
    assert selection_calls[0][1].device.type == device
    assert selection_calls[0][2].device.type == device
    assert actual["logw_partial"].device.type == device
    assert marginalizer.marginalize_vector_weights.device.type == device
    torch.testing.assert_close(
        sample_calls[0][1].detach().cpu(),
        torch.as_tensor(
            cpu_weights[0], dtype=sample_calls[0][1].dtype
        ),
        rtol=2e-6 if device == "mps" else 1e-13,
        atol=2e-6 if device == "mps" else 1e-13,
    )
    selected_indices = sample_calls[0][2].detach().cpu().numpy()
    expected_tc = (
        zero_index_reference["tc"] + selected_indices * delta_t
    )
    np.testing.assert_allclose(actual["tc"], expected_tc, rtol=0, atol=0)
    expected_logw = (
        -cpu_weights[0][selected_indices]
        + np.log(1.0 / len(cpu_weights[0]))
    )
    torch.testing.assert_close(
        actual["logw_partial"].detach().cpu(),
        torch.as_tensor(
            expected_logw,
            dtype=actual["logw_partial"].dtype,
        ),
        rtol=2e-6 if device == "mps" else 1e-13,
        atol=2e-6 if device == "mps" else 1e-13,
    )


def test_weighted_device_draws_use_torch_rng(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    size = 257

    with ctx:
        logweights = torch.linspace(-2.0, 1.0, 11, device=device, dtype=dtype)
        cdf = torch.exp(logweights - logweights.max()).cumsum(dim=0)
        cdf = cdf / cdf[-1]

        torch.manual_seed(9182)
        expected = torch.searchsorted(
            cdf, torch.rand(size, device=device, dtype=dtype)
        )

        def reject_numpy_rng(*_args, **_kwargs):
            raise AssertionError("device draw used NumPy's host RNG")

        with monkeypatch.context() as patch:
            patch.setattr(np.random, "uniform", reject_numpy_rng)
            torch.manual_seed(9182)
            actual = _INFERENCE_TOOLS.draw_sample(
                logweights, size=size, host=False
            )

        np.random.seed(9182)
        uniforms = np.random.uniform(size=size)
        expected_host = np.searchsorted(
            cdf.detach().cpu().numpy(),
            torch.as_tensor(uniforms, dtype=dtype).numpy(),
        )
        np.random.seed(9182)
        actual_host = _INFERENCE_TOOLS.draw_sample(
            logweights, size=size
        )

    assert actual.device.type == device
    assert actual.dtype == torch.int64
    assert torch.equal(actual, expected)
    np.testing.assert_array_equal(actual_host, expected_host)


def test_brute_phase_marginalization_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    loglr_values = [-3.5, -0.75, 0.25, -1.5]
    maxl_values = [-2.0, 0.5, -0.25, 1.75]
    expected = scipy.special.logsumexp(loglr_values) - np.log(
        len(loglr_values)
    )
    phases = np.linspace(0, 2 * np.pi, len(loglr_values), endpoint=False)
    seen_params = []

    with ctx:
        values = [
            (
                torch.tensor(value, device=device, dtype=dtype),
                {
                    "maxl_loglr": torch.tensor(
                        maxl, device=device, dtype=dtype
                    ),
                    "selected_mode": index,
                },
            )
            for index, (value, maxl) in enumerate(
                zip(loglr_values, maxl_values)
            )
        ]
        model = object.__new__(
            inference_brute_marg.BruteParallelGaussianMarginalize
        )
        model.phase = phases
        model._current_params = {"mass1": 30.0}
        model._current_stats = types.SimpleNamespace()
        model.model = types.SimpleNamespace(
            _extra_stats=["maxl_loglr", "selected_mode"]
        )
        model.call = object()

        def map_values(_call, params):
            seen_params.extend(params)
            return values

        model.pool = types.SimpleNamespace(map=map_values)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "brute phase marginalization copied its vector to host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = model._loglr()

    tolerance = 2e-6 if device == "mps" else 1e-13
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert model._current_stats.maxl_phase == phases[3]
    assert model._current_stats.maxl_loglr == maxl_values[3]
    assert model._current_stats.selected_mode == 3
    assert [params["coa_phase"] for params in seen_params] == list(phases)
    assert all(value[0].device.type == device for value in values)


def test_brute_lisa_marginalization_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    loglr_values = [-4.0, -1.25, 0.75, -0.5]
    expected = scipy.special.logsumexp(loglr_values) - np.log(
        len(loglr_values)
    )

    with ctx:
        likelihoods = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in loglr_values
        ]
        values = [(value, {}) for value in likelihoods]
        model = object.__new__(
            inference_brute_marg.BruteLISASkyModesMarginalize
        )
        model.num_sky_modes = len(values)
        model.reconstruct_sky_points = False
        model._current_params = {}
        model._current_stats = types.SimpleNamespace()
        model.call = object()
        model.mapfunc = lambda _call, _params: values
        model._apply_sky_point_rotation = lambda pref, mode: pref.update(
            mode=mode
        )

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "brute sky marginalization copied its vector to host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = model._loglr()
            model.reconstruct_sky_points = True
            reconstruction = model._loglr()
            reconstruction.sum().backward()

    tolerance = 2e-6 if device == "mps" else 1e-13
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert reconstruction.device.type == device
    assert reconstruction.requires_grad
    for index, value in enumerate(loglr_values):
        assert getattr(model._current_stats, f"llr_mode_{index}") == value
        assert likelihoods[index].grad.item() == 1


def test_equal_weight_resampling_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    seed = 1837

    with ctx:
        logwt_tensor = torch.tensor(
            [-2.5, -0.3, 0.4, -1.2, 0.9, -0.8],
            dtype=dtype,
            device=device,
        )
        first_tensor = torch.linspace(
            10.0, 60.0, len(logwt_tensor), dtype=dtype, device=device
        )
        second_tensor = torch.arange(
            len(logwt_tensor), dtype=torch.int64, device=device
        )
        logwt = Array(TorchArrayData(logwt_tensor), copy=False)
        samples = {
            "first": Array(TorchArrayData(first_tensor), copy=False),
            "second": second_tensor,
        }

        torch.manual_seed(9182)
        weights = torch.softmax(logwt_tensor, dim=0)
        positions = (
            torch.rand((), device=device, dtype=dtype)
            + torch.arange(len(weights), device=device, dtype=dtype)
        ) / len(weights)
        cumulative = torch.cumsum(weights, dim=0)
        cumulative = cumulative / cumulative[-1]
        expected_indices = torch.searchsorted(
            cumulative, positions, right=True
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        expected_indices = expected_indices[
            torch.randperm(
                len(weights), device=device, generator=generator
            )
        ]

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("equal-weight resampling left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(inference_refine, "logsumexp", reject_host_or_numpy)
            patch.setattr(inference_refine.numpy, "exp", reject_host_or_numpy)
            patch.setattr(
                inference_refine.numpy.random, "random", reject_host_or_numpy
            )
            torch.manual_seed(9182)
            actual = inference_refine.resample_equal(
                samples, logwt, seed=seed
            )

    assert isinstance(actual["first"], Array)
    assert isinstance(actual["first"]._data, TorchArrayData)
    assert actual["first"]._data.tensor.device.type == device
    assert isinstance(actual["second"], torch.Tensor)
    assert actual["second"].device.type == device
    torch.testing.assert_close(
        actual["first"]._data.tensor, first_tensor[expected_indices]
    )
    torch.testing.assert_close(
        actual["second"], second_tensor[expected_indices]
    )


def test_premarginalized_draws_keep_weights_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    pool_size = 19
    sample_count = 7
    host_params = {
        "tc": np.linspace(1_126_259_461.0, 1_126_259_462.0, pool_size),
        "ra": np.linspace(0.1, 2.1, pool_size),
        "dec": np.linspace(-0.8, 0.8, pool_size),
    }
    host_sample_idx = np.arange(pool_size, dtype=np.int64) * 3

    with ctx:
        logw = torch.linspace(
            -3.0, 1.0, pool_size, device=device, dtype=dtype
        )
        marginalizer = _INFERENCE_TOOLS.DistMarg()
        marginalizer.vsamples = sample_count
        marginalizer.snr_params = list(host_params)
        marginalizer.premarg = {
            **host_params,
            "logw_partial": logw,
            "sample_idx": host_sample_idx,
        }
        marginalizer.marginalize_vector_params = {}
        marginalizer.marginalize_vector_weights = -np.log(sample_count)
        marginalizer._current_params = {}

        torch.manual_seed(3814)
        expected_choice = torch.randperm(
            pool_size, device=device
        )[:sample_count]
        expected_host_choice = expected_choice.detach().cpu().numpy()
        expected_logw = logw[expected_choice] - np.log(sample_count)
        expected_logw = expected_logw - torch.logsumexp(
            expected_logw, dim=0
        )

        original_cpu = torch.Tensor.cpu
        host_transfers = []

        def allow_only_selection_indices(tensor, *args, **kwargs):
            if tensor.dtype != torch.int64 or tensor.numel() != sample_count:
                raise AssertionError(
                    "pre-marginalized weights copied to the host"
                )
            host_transfers.append(tensor)
            return original_cpu(tensor, *args, **kwargs)

        def reject_numpy_choice(*_args, **_kwargs):
            raise AssertionError("device proposal used NumPy's host RNG")

        with monkeypatch.context() as patch:
            patch.setattr(np.random, "choice", reject_numpy_choice)
            patch.setattr(torch.Tensor, "cpu", allow_only_selection_indices)
            torch.manual_seed(3814)
            actual = marginalizer.premarg_draw()

    assert len(host_transfers) == 1
    for name, values in host_params.items():
        np.testing.assert_array_equal(
            actual[name], values[expected_host_choice]
        )
        assert marginalizer._current_params[name] is actual[name]
    np.testing.assert_array_equal(
        marginalizer.sample_idx,
        host_sample_idx[expected_host_choice],
    )
    assert marginalizer.marginalize_vector_weights.device.type == device
    torch.testing.assert_close(
        marginalizer.marginalize_vector_weights.detach().cpu(),
        expected_logw.detach().cpu(),
        rtol=2e-6 if device == "mps" else 1e-13,
        atol=2e-6 if device == "mps" else 1e-13,
    )


def test_sky_time_marginalization_accumulates_weights_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    sample_count = 64
    epoch = 1_126_259_461
    delta_t = 1 / 1024
    times = np.arange(4096) * delta_t
    dtype = np.complex64 if device == "mps" else np.complex128
    snr_data = {
        "H1": np.exp(0.2j * times).astype(dtype),
        "L1": np.exp(0.3j * times).astype(dtype),
    }

    delay_bins = {(offset,): [0] for offset in range(-4096, 4097)}
    bin_prior = {offset: 1.0 for offset in delay_bins}
    fp = {ifo: np.ones(1) for ifo in snr_data}
    fc = {ifo: np.zeros(1) for ifo in snr_data}
    dtc = {ifo: np.zeros(1) for ifo in snr_data}
    sky_info = (
        delay_bins,
        epoch + 1.5,
        epoch + 2.0,
        fp,
        fc,
        np.array([1.2]),
        np.array([-0.4]),
        dtc,
        bin_prior,
    )

    def make_marginalizer():
        marginalizer = _INFERENCE_TOOLS.DistMarg()
        marginalizer.data = {ifo: None for ifo in snr_data}
        marginalizer.tinfo = {"H1L1": sky_info}
        marginalizer._current_params = {}
        marginalizer.vsamples = sample_count
        marginalizer.marginalize_vector_params = {}
        marginalizer.marginalize_vector_weights = np.zeros(sample_count)
        return marginalizer

    def make_snrs():
        return {
            ifo: TimeSeries(data, delta_t=delta_t, epoch=epoch)
            for ifo, data in snr_data.items()
        }

    np.random.seed(817)
    expected = make_marginalizer().draw_sky_times(make_snrs())
    expected = {
        key: np.array(value, copy=True) for key, value in expected.items()
    }

    transfers = []
    original_selected_values = _INFERENCE_TOOLS._selected_values

    def record_selected_values(values, indices, *, host=True):
        tensor = _INFERENCE_TOOLS._torch_tensor(values)
        if tensor is not None:
            transfers.append((host, len(indices)))
        return original_selected_values(values, indices, host=host)

    with ctx:
        snrs = make_snrs()
        marginalizer = make_marginalizer()
        with monkeypatch.context() as patch:
            def reject_full_series_transfer(_self):
                raise AssertionError(
                    "Sky marginalization copied a complete SNR series"
                )

            patch.setattr(TorchArrayData, "numpy", reject_full_series_transfer)
            patch.setattr(
                _INFERENCE_TOOLS,
                "_selected_values",
                record_selected_values,
            )
            np.random.seed(817)
            actual = marginalizer.draw_sky_times(snrs)

    assert transfers == [
        (False, sample_count),
        (False, sample_count),
        (False, sample_count),
    ]
    for snr in snrs.values():
        assert snr._data.tensor.device.type == device
    for key in ("tc", "ra", "dec"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0)
    assert actual["logw_partial"].device.type == device
    assert marginalizer.marginalize_vector_weights.device.type == device
    torch.testing.assert_close(
        actual["logw_partial"].detach().cpu(),
        torch.as_tensor(
            expected["logw_partial"],
            dtype=actual["logw_partial"].dtype,
        ),
        rtol=2e-6 if device == "mps" else 1e-13,
        atol=2e-6 if device == "mps" else 1e-13,
    )


def test_peak_lock_extent_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    epoch = 1_126_259_461
    delta_t = 1 / 1024
    times = np.arange(4096) * delta_t
    dtype = np.float32 if device == "mps" else np.float64
    snr_data = (
        1 + 9 * np.exp(-((times - 1.5) / 0.08) ** 2)
    ).astype(dtype)

    def make_marginalizer():
        marginalizer = _INFERENCE_TOOLS.DistMarg()
        marginalizer.marginalized_vector_priors = {
            "tc": types.SimpleNamespace(
                bounds={"tc": (epoch + 1.0, epoch + 2.0)}
            )
        }
        marginalizer.data = {"H1": None}
        return marginalizer

    expected = make_marginalizer()
    expected.setup_peak_lock(
        snrs={
            "H1": TimeSeries(snr_data, delta_t=delta_t, epoch=epoch)
        },
        sample_rate=1024,
        peak_lock_snr=2,
        peak_lock_ratio=1e4,
        peak_lock_region=4,
    )

    def _reject_comparison(_self, _other, _operation):
        raise AssertionError("Peak locking copied its threshold mask to host")

    def _reject_numpy(_self):
        raise AssertionError("Peak locking copied its SNR series to host")

    with ctx:
        snr = TimeSeries(snr_data, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "comparison", _reject_comparison)
            patch.setattr(TorchArrayData, "numpy", _reject_numpy)
            actual = make_marginalizer()
            actual.setup_peak_lock(
                snrs={"H1": snr},
                sample_rate=1024,
                peak_lock_snr=2,
                peak_lock_ratio=1e4,
                peak_lock_region=4,
            )

    assert snr._data.tensor.device.type == device
    assert float(actual.tstart["H1"]) == float(expected.tstart["H1"])
    assert actual.num_samples == expected.num_samples
    assert float(actual.tend["H1"]) == float(expected.tend["H1"])


@pytest.mark.parametrize("precalc_antenna_factors", (False, True))
def test_relative_time_projection_stays_differentiable(
        torch_device_ctx, monkeypatch, precalc_antenna_factors):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    torch_real_dtype = torch.float32 if device == "mps" else torch.float64
    ifo = "H1"
    fp, fc, delay = real_dtype(0.43), real_dtype(-0.27), 0.0
    inclination_value = real_dtype(0.71)
    polarization_value = real_dtype(0.39)
    hh = real_dtype(1.7)
    sh_values = (
        np.linspace(1.0, 9.0, 9, dtype=real_dtype)
        * (1.0 + 0.2j)
    ).astype(complex_dtype)

    detector_calls = []

    class _Detector:
        def antenna_pattern(self, ra, dec, polarization, _times):
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec, polarization)
            )
            detector_calls.append("antenna")
            return (
                fp + 0.01 * (ra - 1.0),
                fc + 0.02 * (dec + 0.4),
            )

        def time_delay_from_earth_center(self, ra, dec, _times):
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            detector_calls.append("delay")
            return delay + 0.001 * (ra - dec - 1.4)

    model = types.SimpleNamespace()
    model.precalc_antenna_factors = precalc_antenna_factors
    model.get_precalc_antenna_factors = lambda _ifo: (fp, fc, delay)
    model.det = {ifo: _Detector()}
    model.get_waveforms = lambda _params, keep_torch: {ifo: None}
    model.snr_draw = lambda **_kwargs: None
    model.return_sh_hh = False
    model.marginalize_loglr = lambda sh, norm: sh.real - 0.5 * norm

    with ctx:
        inclination = torch.tensor(
            inclination_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        polarization = torch.tensor(
            polarization_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        right_ascension = torch.tensor(
            1.0, dtype=torch_real_dtype, device=device, requires_grad=True)
        declination = torch.tensor(
            -0.4, dtype=torch_real_dtype, device=device, requires_grad=True)
        model.current_params = {
            "inclination": inclination,
            "polarization": polarization,
            "ra": right_ascension,
            "dec": declination,
            "tc": 1.0,
        }
        sh_series = TimeSeries(sh_values, delta_t=0.25, epoch=0.0)

        def _get_snr(_waveforms):
            model.sh = {ifo: sh_series}
            model.hh = {ifo: hh}
            return {ifo: sh_series}

        model.get_snr = _get_snr
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Relative time copied samples to host")

            def _reject_numpy_projection(_value):
                raise AssertionError("Relative time projection used NumPy")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                inference_relbin.numpy, "cos", _reject_numpy_projection)
            patch.setattr(
                inference_relbin.numpy, "exp", _reject_numpy_projection)
            actual = inference_relbin.RelativeTimeDom._loglr.__wrapped__(
                model)
            actual.backward()

        actual_value = actual.detach().cpu().numpy()
        inclination_gradient = inclination.grad.detach().cpu().numpy()
        polarization_gradient = polarization.grad.detach().cpu().numpy()
        sky_gradients = tuple(
            None if value.grad is None else value.grad.detach().cpu().numpy()
            for value in (right_ascension, declination)
        )

    cosi = np.cos(inclination_value)
    plus = 0.5 * (1.0 + cosi * cosi)
    response = (fp + 1.0j * fc) * np.exp(
        -2.0j * polarization_value)
    projection = response.real * plus + 1.0j * response.imag * cosi
    sh_at_time = sh_values[4]
    expected = (sh_at_time * projection).real - 0.5 * hh * abs(
        projection) ** 2.0

    assert actual.device.type == device
    assert np.isfinite(inclination_gradient)
    assert np.isfinite(polarization_gradient)
    assert inclination_gradient != 0.0
    assert polarization_gradient != 0.0
    if precalc_antenna_factors:
        assert detector_calls == []
        assert sky_gradients == (None, None)
    else:
        assert detector_calls == ["antenna", "delay"]
        assert all(np.isfinite(value) and value != 0.0
                   for value in sky_gradients)
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected, rtol=tolerance, atol=tolerance)


def test_single_template_projection_stays_differentiable(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    torch_real_dtype = torch.float32 if device == "mps" else torch.float64
    ifo = "H1"
    fp, fc, delay = real_dtype(0.43), real_dtype(-0.27), 0.0
    inclination_value = real_dtype(0.71)
    polarization_value = real_dtype(0.39)
    coa_phase_value = real_dtype(0.23)
    distance_value = real_dtype(1.8)
    hh = real_dtype(1.7)
    sh_values = (
        np.linspace(1.0, 9.0, 9, dtype=real_dtype)
        * (1.0 + 0.2j)
    ).astype(complex_dtype)

    class _Detector:
        def antenna_pattern(self, ra, dec, polarization, _times):
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec, polarization)
            )
            detector_calls.append("antenna")
            return (
                fp + 0.01 * (ra - 1.0),
                fc + 0.02 * (dec + 0.4),
            )

        def time_delay_from_earth_center(self, ra, dec, _times):
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            detector_calls.append("delay")
            return delay + 0.001 * (ra - dec - 1.4)

    detector_calls = []
    model = types.SimpleNamespace()
    model.sh = {}
    model.hh = {ifo: hh}
    model.snr = {}
    model.det = {ifo: _Detector()}
    model.dts = {}
    model.htfs = {}
    model.snr_draw = lambda **_kwargs: None
    model.marginalize_loglr = lambda sh, norm: sh.real - 0.5 * norm

    with ctx:
        inclination = torch.tensor(
            inclination_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        polarization = torch.tensor(
            polarization_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        coa_phase = torch.tensor(
            coa_phase_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        distance = torch.tensor(
            distance_value, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        right_ascension = torch.tensor(
            1.0, dtype=torch_real_dtype, device=device, requires_grad=True)
        declination = torch.tensor(
            -0.4, dtype=torch_real_dtype, device=device, requires_grad=True)
        model.current_params = {
            "inclination": inclination,
            "polarization": polarization,
            "coa_phase": coa_phase,
            "distance": distance,
            "ra": right_ascension,
            "dec": declination,
            "tc": 1.0,
        }
        model.sh[ifo] = TimeSeries(
            sh_values, delta_t=0.25, epoch=0.0)

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Single template copied samples to host")

            def _reject_numpy_projection(_value):
                raise AssertionError("Single template projection used NumPy")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                single_template.numpy, "cos", _reject_numpy_projection)
            patch.setattr(
                single_template.numpy, "exp", _reject_numpy_projection)
            actual = single_template.SingleTemplate._loglr(model)
            actual.backward()

        actual_value = actual.detach().cpu().numpy()
        gradients = tuple(
            value.grad.detach().cpu().numpy()
            for value in (
                inclination, polarization, coa_phase, distance,
                right_ascension, declination,
            )
        )
        stored_factor = model.htfs[ifo]

    cosi = np.cos(inclination_value)
    plus = 0.5 * (1.0 + cosi * cosi)
    response = (fp + 1.0j * fc) * np.exp(
        -2.0j * polarization_value)
    projection = response.real * plus + 1.0j * response.imag * cosi
    factor = projection * np.exp(-2.0j * coa_phase_value) / distance_value
    sh_at_time = sh_values[4]
    expected = (sh_at_time * factor).real - 0.5 * hh * abs(factor) ** 2.0

    assert actual.device.type == device
    assert stored_factor.device.type == device
    assert detector_calls == ["antenna", "delay"]
    assert all(np.isfinite(gradient) for gradient in gradients)
    assert all(gradient != 0.0 for gradient in gradients)
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "disabled_switch",
    (
        "PYCBC_IMRPHENOMD_NATIVE",
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
    ),
)
def test_single_template_limit_preserves_native_opt_out(
    monkeypatch, disabled_switch
):
    for name in (
        "PYCBC_IMRPHENOMD_NATIVE",
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(disabled_switch, "0")
    monkeypatch.setitem(
        sys.modules, "pycbc.waveform.imrphenomd_torch", None
    )
    parameters = {
        "approximant": "IMRPhenomD",
        "mass1": 30.0,
        "mass2": 30.0,
        "f_final": 512.0,
    }

    with scheme.TorchScheme("cpu"):
        single_template._limit_torch_imrphenomd_generation(
            parameters, flen=513, delta_f=0.25
        )

    assert parameters["f_final"] == 512.0


@pytest.mark.parametrize(
    "unsupported",
    (
        {"dchi0": 0.1},
        {"mode_array": [(2, 2)]},
        {"spin_order": 5},
    ),
)
def test_single_template_limit_preserves_unsupported_lal_fallback(
    monkeypatch, unsupported
):
    for name in (
        "PYCBC_IMRPHENOMD_NATIVE",
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    parameters = {
        "approximant": "IMRPhenomD",
        "mass1": 30.0,
        "mass2": 30.0,
        "f_final": 512.0,
        **unsupported,
    }

    with scheme.TorchScheme("cpu"):
        single_template._limit_torch_imrphenomd_generation(
            parameters, flen=513, delta_f=0.25
        )

    assert parameters["f_final"] == 512.0


def test_single_template_high_mass_default_avoids_inflated_layout(monkeypatch):
    for name in (
        "PYCBC_IMRPHENOMD_NATIVE",
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    delta_f = 1.0 / 32.0
    flen = 32769
    parameters = {
        "approximant": "IMRPhenomD",
        "mass1": 200.0,
        "mass2": 200.0,
        "spin1z": 0.15,
        "spin2z": -0.08,
        "f_lower": 20.0,
        "f_ref": 20.0,
        "distance": 100.0,
        "inclination": 0.65,
    }

    with scheme.TorchScheme("cpu"):
        limited_parameters = parameters.copy()
        single_template._limit_torch_imrphenomd_generation(
            limited_parameters, flen=flen, delta_f=delta_f
        )
        natural, _ = single_template.get_fd_waveform(
            delta_f=delta_f, **limited_parameters
        )
        inflated, _ = single_template.get_fd_waveform(
            delta_f=delta_f,
            f_final=flen * delta_f,
            **parameters,
        )

        assert "f_final" not in limited_parameters
        assert len(natural) == 4097
        assert len(inflated) == 65537
        natural.resize(flen)
        inflated.resize(flen)
        assert torch.equal(natural._data.tensor, inflated._data.tensor)


def test_single_template_scalar_projection_avoids_tensor_kernels(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    ifo = "H1"
    fp, fc, delay = real_dtype(0.43), real_dtype(-0.27), real_dtype(0.0)
    inclination = real_dtype(0.71)
    polarization = real_dtype(0.39)
    coa_phase = real_dtype(0.23)
    distance = real_dtype(1.8)
    hh = real_dtype(1.7)
    sh_values = (
        np.linspace(1.0, 9.0, 9, dtype=real_dtype) * (1.0 + 0.2j)
    ).astype(complex_dtype)
    detector_calls = []

    class _Detector:
        @staticmethod
        def antenna_pattern(ra, dec, pol, _times):
            if device == "mps":
                assert all(isinstance(value, torch.Tensor)
                           for value in (ra, dec, pol))
            else:
                assert all(np.ndim(value) == 0
                           for value in (ra, dec, pol))
                assert not any(isinstance(value, torch.Tensor)
                               for value in (ra, dec, pol))
            detector_calls.append("antenna")
            return fp, fc

        @staticmethod
        def time_delay_from_earth_center(ra, dec, _times):
            if device == "mps":
                assert all(isinstance(value, torch.Tensor)
                           for value in (ra, dec))
            else:
                assert all(np.ndim(value) == 0 for value in (ra, dec))
                assert not any(isinstance(value, torch.Tensor)
                               for value in (ra, dec))
            detector_calls.append("delay")
            return delay

    model = types.SimpleNamespace(
        sh={},
        hh={ifo: hh},
        snr={},
        det={ifo: _Detector()},
        dts={},
        htfs={},
        current_params={
            "inclination": inclination,
            "polarization": polarization,
            "coa_phase": coa_phase,
            "distance": distance,
            "ra": real_dtype(1.0),
            "dec": real_dtype(-0.4),
            "tc": real_dtype(1.0),
        },
    )
    model.snr_draw = lambda **_kwargs: None
    model.marginalize_loglr = lambda sh, norm: sh.real - 0.5 * norm

    with ctx:
        model.sh[ifo] = TimeSeries(
            sh_values, delta_t=0.25, epoch=0.0
        )
        from pycbc.inference.models import relbin_torch

        tensor_kernel_calls = []

        def _tensor_kernel(name, function):
            def wrapped(*args, **kwargs):
                tensor_kernel_calls.append(name)
                if device != "mps":
                    raise AssertionError("scalar projection used tensor kernels")
                return function(*args, **kwargs)
            return wrapped

        with monkeypatch.context() as patch:
            patch.setattr(
                single_template,
                "_host_scalar_extrinsics",
                lambda _parameters: pytest.fail(
                    "plain scalar parameters used generic tensor probes"
                ),
            )
            patch.setattr(
                relbin_torch, "detector_response",
                _tensor_kernel(
                    "detector_response", relbin_torch.detector_response
                ),
            )
            patch.setattr(
                relbin_torch, "dominant_mode_template_factor",
                _tensor_kernel(
                    "dominant_mode_template_factor",
                    relbin_torch.dominant_mode_template_factor,
                ),
            )
            actual = single_template.SingleTemplate._loglr(model)
        actual_value = actual.detach().cpu().numpy()
        stored_factor = model.htfs[ifo]
        if isinstance(stored_factor, torch.Tensor):
            stored_factor = stored_factor.detach().cpu().numpy()

    cosi = np.cos(inclination)
    plus = 0.5 * (1.0 + cosi * cosi)
    response = (fp + 1.0j * fc) * np.exp(-2.0j * polarization)
    projection = response.real * plus + 1.0j * response.imag * cosi
    factor = projection * np.exp(-2.0j * coa_phase) / distance
    expected = (sh_values[4] * factor).real - 0.5 * hh * abs(factor) ** 2.0

    assert actual.device.type == device
    if device == "mps":
        assert detector_calls == ["antenna", "delay"]
        assert tensor_kernel_calls == [
            "detector_response", "dominant_mode_template_factor"
        ]
    else:
        assert detector_calls == ["delay", "antenna"]
        assert tensor_kernel_calls == []
    np.testing.assert_allclose(stored_factor, factor, rtol=2e-7, atol=2e-7)
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    "time_value",
    (
        1.0,
        np.float64(1.0),
    ),
)
def test_single_template_cached_host_storage_skips_tensor_dispatch(
        monkeypatch, time_value):
    ifo = "H1"
    fp, fc = 0.43, -0.27
    inclination = 0.71
    polarization = 0.39
    coa_phase = 0.23
    distance = 1.8
    hh = 1.7
    sh_values = np.linspace(1.0, 9.0, 9) * (1.0 + 0.2j)

    class _Detector:
        @staticmethod
        def antenna_pattern(_ra, _dec, _polarization, _times):
            return fp, fc

        @staticmethod
        def time_delay_from_earth_center(_ra, _dec, _times):
            return 0.0

    model = types.SimpleNamespace(
        sh={ifo: TimeSeries(sh_values, delta_t=0.25, epoch=0.0)},
        hh={ifo: hh},
        snr={},
        det={ifo: _Detector()},
        dts={},
        htfs={},
        _sh_storage_is_host=True,
        current_params={
            "inclination": inclination,
            "polarization": polarization,
            "coa_phase": coa_phase,
            "distance": distance,
            "ra": 1.0,
            "dec": -0.4,
            "tc": time_value,
        },
    )
    model.snr_draw = lambda **_kwargs: None
    model.marginalize_loglr = (
        lambda sh, norm, skip_vector=False: sh.real - 0.5 * norm
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            single_template,
            "_host_scalar_extrinsics",
            lambda _parameters: pytest.fail(
                "host storage inspected Torch parameter types"
            ),
        )
        patch.setattr(
            single_template,
            "_torch_tensor",
            lambda _value: pytest.fail(
                "cached host storage was inspected again"
            ),
        )
        actual = single_template.SingleTemplate._loglr(
            model
        )

    cosi = np.cos(inclination)
    plus = 0.5 * (1.0 + cosi * cosi)
    response = (fp + 1.0j * fc) * np.exp(-2.0j * polarization)
    projection = response.real * plus + 1.0j * response.imag * cosi
    factor = projection * np.exp(-2.0j * coa_phase) / distance
    index = int(time_value / 0.25)
    expected = (
        (sh_values[index] * factor).real
        - 0.5 * hh * abs(factor) ** 2.0
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_single_template_plain_host_scalar_guard_rejects_tensor_and_arrays():
    parameters = {
        "inclination": 0.71,
        "polarization": np.float64(0.39),
        "coa_phase": 0.23,
        "distance": 1.8,
        "ra": 1.0,
        "dec": -0.4,
        "tc": 1.0,
    }
    assert single_template._plain_host_scalar_extrinsics(parameters)

    for value in (
        torch.tensor(1.0, dtype=torch.float64),
        np.array(1.0),
        np.array((0.75, 1.0, 1.25)),
    ):
        candidate = {**parameters, "tc": value}
        assert not single_template._plain_host_scalar_extrinsics(candidate)

    # Reusing a cached type signature must not retain the result of a prior
    # mixed NumPy/Torch parameter set.
    parameters["tc"] = np.float32(1.0)
    assert single_template._plain_host_scalar_extrinsics(parameters)
    parameters["tc"] = torch.tensor(1.0, dtype=torch.float32)
    assert not single_template._plain_host_scalar_extrinsics(parameters)
    parameters["tc"] = np.float64(1.0)
    assert single_template._plain_host_scalar_extrinsics(parameters)

    for index in range(40):
        scalar_type = type(f"_HostScalarType{index}", (float,), {})
        parameters["tc"] = scalar_type(1.0)
        assert single_template._plain_host_scalar_extrinsics(parameters)
    cache_info = single_template._plain_host_scalar_types.cache_info()
    assert cache_info.maxsize == 32
    assert cache_info.currsize <= cache_info.maxsize
    single_template._plain_host_scalar_types.cache_clear()


@pytest.mark.parametrize("detectors", (("H1",), ("H1", "L1")))
def test_single_template_torch_cpu_native_scalar_likelihood_bit_exact(
        torch_ctx, monkeypatch, detectors):
    with torch_ctx:
        model = _native_scalar_single_template_model(detectors=detectors)
        assert single_template._torch_cpu_native_scalar_likelihood_eligible(
            model, host_storage=False, plain_host_scalars=True,
            skip_vector=False,
        )

        for offset in (-0.019, -0.003, 0.0, 0.007, 0.023):
            model._current_params["tc"] = np.float64(
                1126259466.003 + offset
            )
            native = model._loglr()
            with monkeypatch.context() as patch:
                patch.setattr(
                    single_template,
                    "_torch_cpu_native_scalar_likelihood_eligible",
                    lambda *_args, **_kwargs: False,
                )
                established = model._loglr()

            assert type(native) is float
            assert type(established) is float
            assert native == established


def test_single_template_torch_storage_dispatch_is_per_call_and_mutation_safe(
        torch_ctx, monkeypatch):
    with torch_ctx:
        model = _native_scalar_single_template_model()
        series = model.sh["H1"]
        original_tensor = series._data.tensor
        original_dispatch = single_template._torch_tensor
        dispatch_values = []

        def traced_dispatch(value):
            dispatch_values.append(value)
            return original_dispatch(value)

        monkeypatch.setattr(single_template, "_torch_tensor", traced_dispatch)

        expected = model._loglr()
        assert sum(value is series for value in dispatch_values) == 1

        dispatch_values.clear()
        assert model._loglr() == expected
        assert sum(value is series for value in dispatch_values) == 1

        # A same-model storage replacement is reclassified on the next call;
        # the per-call tensor mapping cannot retain the former dtype.
        series._data._set_tensor(original_tensor.to(torch.complex64))
        dispatch_values.clear()
        result = model._loglr()
        assert torch.isfinite(torch.as_tensor(result))
        assert sum(value is series for value in dispatch_values) == 1

        # Mixed Torch/NumPy extrinsics likewise leave and re-enter the native
        # path based on their current types, not a previous evaluation.
        series._data._set_tensor(original_tensor)
        original_tc = model._current_params["tc"]
        model._current_params["tc"] = torch.tensor(
            original_tc, dtype=torch.float64, requires_grad=True
        )
        dispatch_values.clear()
        with monkeypatch.context() as patch:
            patch.setattr(
                single_template.DistMarg,
                "marginalize_loglr",
                lambda _self, sh, hh: sh.real - 0.5 * hh,
            )
            result = model._loglr()
            result.backward()
        assert model._current_params["tc"].grad is not None
        assert sum(value is series for value in dispatch_values) == 1

        model._current_params["tc"] = original_tc
        dispatch_values.clear()
        assert model._loglr() == expected
        assert sum(value is series for value in dispatch_values) == 1


def test_single_template_torch_cpu_native_scalar_likelihood_guards(
        torch_ctx):
    with torch_ctx:
        model = _native_scalar_single_template_model()

        def eligible(skip_vector=False):
            return single_template._torch_cpu_native_scalar_likelihood_eligible(
                model, host_storage=False, plain_host_scalars=True,
                skip_vector=skip_vector,
            )

        assert eligible()
        assert not eligible(skip_vector=True)
        assert not single_template._torch_cpu_native_scalar_likelihood_eligible(
            model, host_storage=True, plain_host_scalars=True,
            skip_vector=False,
        )
        assert not single_template._torch_cpu_native_scalar_likelihood_eligible(
            model, host_storage=False, plain_host_scalars=False,
            skip_vector=False,
        )
        model.marginalize_phase = True
        assert not eligible()
        model.marginalize_phase = False

        model.marginalize_loglr = lambda sh, hh: sh.real - 0.5 * hh
        assert not eligible()
        del model.marginalize_loglr

        series = model.sh["H1"]
        original = series._data.tensor
        series._data._set_tensor(original.detach().requires_grad_())
        assert not eligible()

        noncontiguous = torch.stack((original, original), dim=1)[:, 0]
        assert not noncontiguous.is_contiguous()
        series._data._set_tensor(noncontiguous)
        assert not eligible()

        series._data._set_tensor(original.conj())
        assert not eligible()

        series._data._set_tensor(torch._neg_view(original))
        assert not eligible()

        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                original, torch.ones_like(original)
            )
            series._data._set_tensor(dual)
            assert not eligible()

        series._data._set_tensor(original.to(torch.complex64))
        assert not eligible()

        series._data._set_tensor(original)
        model.hh["H1"] = np.float32(model.hh["H1"])
        assert not eligible()


def test_single_template_native_scalar_likelihood_device_guard(
        torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = np.complex64 if device == "mps" else np.complex128
    with ctx:
        model = _native_scalar_single_template_model(dtype=dtype)
        eligible = (
            single_template._torch_cpu_native_scalar_likelihood_eligible(
                model, host_storage=False, plain_host_scalars=True,
                skip_vector=False,
            )
        )
    assert eligible is (device == "cpu")


def test_single_template_host_storage_preserves_tensor_extrinsics():
    ifo = "H1"

    class _HostSeries:
        @staticmethod
        def at_time(time, interpolate):
            assert interpolate == "quadratic"
            return (1.2 + 0.3j) * time

    class _Detector:
        @staticmethod
        def antenna_pattern(ra, dec, _polarization, times):
            return 0.43 + 0.01 * ra, -0.27 + 0.02 * dec + 0.0 * times

        @staticmethod
        def time_delay_from_earth_center(ra, dec, _times):
            return 0.001 * (ra - dec)

    values = {
        "inclination": torch.tensor(0.71, dtype=torch.float64,
                                    requires_grad=True),
        "polarization": torch.tensor(0.39, dtype=torch.float64,
                                     requires_grad=True),
        "coa_phase": torch.tensor(0.23, dtype=torch.float64,
                                  requires_grad=True),
        "distance": torch.tensor(1.8, dtype=torch.float64,
                                 requires_grad=True),
        "ra": torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
        "dec": torch.tensor(-0.4, dtype=torch.float64, requires_grad=True),
        "tc": torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
    }
    model = types.SimpleNamespace(
        sh={ifo: _HostSeries()},
        hh={ifo: 1.7},
        snr={},
        det={ifo: _Detector()},
        dts={},
        htfs={},
        _sh_storage_is_host=True,
        current_params=values,
    )
    model.snr_draw = lambda **_kwargs: None
    model.marginalize_loglr = lambda sh, norm: sh.real - 0.5 * norm

    actual = single_template.SingleTemplate._loglr(model)
    actual.backward()

    assert isinstance(actual, torch.Tensor)
    assert all(value.grad is not None for value in values.values())
    assert all(torch.isfinite(value.grad) for value in values.values())


def test_single_template_batch_loglr_retains_parameter_grid():
    model = types.SimpleNamespace(
        marginalize_vector_params={},
        marginalize_distance=False,
    )
    updates = {}
    model.update = lambda **params: updates.update(params)
    model._loglr = lambda skip_vector=False: skip_vector

    distances = np.array((70.0, 90.0, 120.0))
    result = single_template.SingleTemplate.batch_loglr(
        model, distance=distances
    )

    assert result is True
    np.testing.assert_array_equal(updates["distance"], distances)


def test_single_template_batch_loglr_rejects_nested_marginalization():
    model = types.SimpleNamespace(
        marginalize_vector_params={"polarization": np.arange(4)},
        marginalize_distance=False,
    )
    with pytest.raises(ValueError, match="vector or distance"):
        single_template.SingleTemplate.batch_loglr(
            model, distance=np.array((70.0, 90.0))
        )


@pytest.mark.parametrize("static_margin", (False, True))
def test_joint_primary_marginalization_stays_on_torch_device(
        torch_device_ctx, monkeypatch, static_margin):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex likelihood vectors")
    real_dtype = torch.float32 if device == "mps" else torch.float64
    complex_dtype = torch.complex64 if device == "mps" else torch.complex128

    class _PrimaryModel:
        def __init__(self, sh, hh, phase):
            self._sh = sh
            self._hh = hh
            self.current_params = {"phase": phase, "fixed": 0.4}
            self.marginalized_params_name = ["phase"]
            self.return_sh_hh = False
            self._current_stats = types.SimpleNamespace()

        @property
        def loglr(self):
            return self._sh, self._hh

        @staticmethod
        def marginalize_loglr(sh, hh):
            return (sh.real - 0.5 * hh).sum()

    class _OtherModel:
        def __init__(self):
            self.current_params = {"offset": 0.25}
            self.return_sh_hh = False
            self._current_stats = types.SimpleNamespace()

        def update(self, **params):
            self.current_params = params

        @property
        def loglr(self):
            phase = self.current_params["phase"]
            return (0.2 + 0.1j) * phase, 0.03 * phase**2

        @staticmethod
        def marginalize_loglr(sh, hh):
            return sh.real - 0.5 * hh

    def evaluate(sh, hh, phase):
        model = types.SimpleNamespace(
            primary_model=_PrimaryModel(sh, hh, phase),
            other_models=[_OtherModel(), _OtherModel()],
            static_margin_params_in_other_models=static_margin,
        )
        result = (
            inference_hierarchical.JointPrimaryMarginalizedModel
            .total_loglr(model)
        )
        return result, model

    sh_values = np.array([1.0 + 0.2j, 5.0 - 0.3j, 2.0 + 0.4j])
    hh_values = np.array([1.0, 4.0, 4.0])
    phase_values = np.array([0.3, 0.7, 1.1])
    if static_margin:
        phase_for_others = phase_values[1]
    else:
        phase_for_others = phase_values
    sh_others = 2 * (0.2 + 0.1j) * phase_for_others
    hh_others = 2 * 0.03 * phase_for_others**2
    expected = np.sum(
        (sh_values + sh_others).real - 0.5 * (hh_values + hh_others)
    )
    numpy_result, _ = evaluate(sh_values, hh_values, phase_values)
    np.testing.assert_allclose(numpy_result, expected, rtol=0, atol=2e-15)

    with ctx:
        amplitude = torch.tensor(
            1.0, device=device, dtype=real_dtype, requires_grad=True
        )
        phase = torch.tensor(
            phase_values, device=device, dtype=real_dtype, requires_grad=True
        )
        sh = torch.tensor(
            sh_values, device=device, dtype=complex_dtype
        ) * amplitude
        hh = torch.tensor(
            hh_values, device=device, dtype=real_dtype
        ) * amplitude**2

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("joint marginalization left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "item", reject_host_or_numpy)
            patch.setattr(
                inference_hierarchical.numpy, "argmax", reject_host_or_numpy
            )
            patch.setattr(
                inference_hierarchical.numpy, "abs", reject_host_or_numpy
            )
            patch.setattr(
                inference_hierarchical.numpy, "asarray", reject_host_or_numpy
            )
            actual, model = evaluate(sh, hh, phase)

        actual.backward()

    tolerance = 5e-6 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected,
        rtol=tolerance, atol=tolerance,
    )
    assert actual.device.type == device
    assert amplitude.grad is not None and torch.isfinite(amplitude.grad)
    assert phase.grad is not None and torch.all(torch.isfinite(phase.grad))
    if static_margin:
        assert torch.count_nonzero(phase.grad).item() == 1
    else:
        assert torch.count_nonzero(phase.grad).item() == len(phase_values)
    assert model.primary_model._current_stats.loglr.device.type == device


def test_marginalized_phase_inner_products_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    waveform_values = np.asarray(
        (0.5 + 0.25j, -1.0 + 0.75j, 1.5 - 0.5j, 0.25 + 1.0j),
        dtype=complex_dtype,
    )
    data_values = np.asarray(
        (1.0 - 0.5j, 0.25 + 1.25j, -0.75 + 0.5j, 1.5 + 0.25j),
        dtype=complex_dtype,
    )
    waveform_values_by_det = {
        "H1": waveform_values,
        "L1": waveform_values * (0.75 - 0.2j),
    }
    data_values_by_det = {
        "H1": data_values,
        "L1": data_values * (-0.4 + 0.6j),
    }
    model = types.SimpleNamespace(
        all_ifodata_same_rate_length=True,
        current_params={},
        _kmin={det: 0 for det in waveform_values_by_det},
        _kmax={
            det: len(values)
            for det, values in waveform_values_by_det.items()
        },
        _current_stats=inference_model_base.ModelStats(),
        default_stats=(
            "maxl_phase", "H1_optimal_snrsq", "L1_optimal_snrsq"
        ),
    )

    with ctx:
        waveforms = {
            det: FrequencySeries(values, delta_f=0.25)
            for det, values in waveform_values_by_det.items()
        }
        data = {
            det: FrequencySeries(values, delta_f=0.25)
            for det, values in data_values_by_det.items()
        }
        data_tensors = [series._data.tensor for series in data.values()]
        for tensor in data_tensors:
            tensor.requires_grad_()
        model.waveform_generator = types.SimpleNamespace(
            generate=lambda **_params: waveforms
        )
        model._weight = {
            det: FrequencySeries(
                np.ones(len(values), dtype=values.real.dtype),
                delta_f=0.25,
            )
            for det, values in waveform_values_by_det.items()
        }
        model._whitened_data = data

        def marginalize(sh, hh, phase=False):
            assert phase
            assert isinstance(sh, torch.Tensor)
            assert isinstance(hh, torch.Tensor)
            assert sh.device.type == device
            assert hh.device.type == device
            return sh.real - 0.5 * hh

        def reject_host_scalar(*_args, **_kwargs):
            raise AssertionError("marginalized phase scalarized an inner product")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", reject_host_scalar)
            patch.setattr(torch.Tensor, "cpu", reject_host_scalar)
            patch.setattr(torch.Tensor, "numpy", reject_host_scalar)
            patch.setattr(TorchArrayData, "numpy", reject_host_scalar)
            patch.setattr(
                marginalized_gaussian_noise.numpy,
                "angle",
                reject_host_scalar,
            )
            patch.setattr(
                marginalized_gaussian_noise,
                "marginalize_likelihood",
                marginalize,
            )
            loglr = (
                marginalized_gaussian_noise.
                MarginalizedPhaseGaussianNoise._loglr
            )
            actual = loglr.__wrapped__(model)
            actual.backward()

        actual_value = actual.detach().cpu().numpy()
        phase_value = model._current_stats.maxl_phase.detach().cpu().numpy()

        # Cached reductions remain device values for subsequent likelihood
        # work. The explicit sampler/serialization boundary returns the
        # historical Python scalar values.
        assert isinstance(model._current_stats.maxl_phase, torch.Tensor)
        assert isinstance(model._current_stats.H1_optimal_snrsq, torch.Tensor)
        assert isinstance(model._current_stats.L1_optimal_snrsq, torch.Tensor)
        public_tuple = inference_model_base.BaseModel.get_current_stats(model)
        public_dict = inference_model_base.BaseModel.current_stats.fget(model)

    assert actual.device.type == device
    assert all(tensor.grad is not None for tensor in data_tensors)
    assert all(torch.isfinite(tensor.grad).all() for tensor in data_tensors)
    expected_hds = {
        det: np.vdot(values, data_values_by_det[det])
        for det, values in waveform_values_by_det.items()
    }
    expected_hhs = {
        det: np.vdot(values, values).real
        for det, values in waveform_values_by_det.items()
    }
    expected_hd = sum(expected_hds.values())
    expected_hh = sum(expected_hhs.values())
    tolerance = 2e-6 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected_hd.real - 0.5 * expected_hh,
        rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        phase_value, np.angle(expected_hd),
        rtol=tolerance, atol=tolerance,
    )
    assert all(type(value) is float for value in public_tuple)
    assert all(type(value) is float for value in public_dict.values())
    np.testing.assert_allclose(
        public_tuple,
        (
            np.angle(expected_hd),
            expected_hhs["H1"],
            expected_hhs["L1"],
        ),
        rtol=tolerance, atol=tolerance,
    )
    assert public_dict == dict(zip(model.default_stats, public_tuple))


@pytest.mark.parametrize("phase", (False, True))
def test_likelihood_marginalization_stays_on_device(
        torch_device_ctx, monkeypatch, phase):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(7734)
    sample_count = 127
    sh = (rng.normal(size=sample_count) +
          1j * rng.normal(size=sample_count)).astype(complex_dtype)
    hh = rng.uniform(0.2, 8.0, sample_count).astype(real_dtype)
    logw = rng.normal(size=sample_count).astype(real_dtype)
    logw -= np.log(np.exp(logw).sum())
    expected = _INFERENCE_TOOLS.marginalize_likelihood(
        sh, hh, logw=logw, phase=phase)
    expected_vector = _INFERENCE_TOOLS.marginalize_likelihood(
        sh, hh, logw=logw, phase=phase, skip_vector=True)
    expected_peak = _INFERENCE_TOOLS.marginalize_likelihood(
        sh, hh, logw=logw, phase=phase, return_peak=True)

    with ctx:
        sh_array = Array(sh)
        hh_array = Array(hh)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Marginalization copied samples to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_array, hh_array, logw=logw, phase=phase)
            actual_vector = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_array, hh_array, logw=logw, phase=phase,
                skip_vector=True)
            actual_peak = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_array, hh_array, logw=logw, phase=phase,
                return_peak=True)

    tolerance = 2e-5 if device == "mps" else 2e-12
    assert isinstance(actual, float)
    assert actual_vector.device.type == device
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_vector.cpu().numpy(), expected_vector,
        rtol=tolerance, atol=tolerance)
    assert actual_peak[1] == expected_peak[1]
    np.testing.assert_allclose(
        (actual_peak[0], actual_peak[2]),
        (expected_peak[0], expected_peak[2]),
        rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("phase", (False, True))
def test_scalar_distance_marginalization_uses_torch_precision(
        torch_device_ctx, phase):
    ctx, device = torch_device_ctx
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    complex_dtype = torch.complex64 if device == "mps" else torch.complex128
    sh = complex(2.75, -0.625)
    hh = 1.875
    dist_rescale = np.linspace(0.4, 1.6, 31)
    dist_weights = np.linspace(0.75, 1.25, 31)
    distance = dist_rescale, dist_weights
    expected = _INFERENCE_TOOLS.marginalize_likelihood(
        sh, hh, phase=phase, distance=distance)

    with ctx:
        hh_tensor = torch.tensor(hh, dtype=torch_dtype, device=device)
        actual = _INFERENCE_TOOLS.marginalize_likelihood(
            sh, hh_tensor, phase=phase, distance=distance)
        complex_parts = _INFERENCE_TOOLS.marginalize_likelihood(
            sh, hh_tensor, distance=distance, return_complex=True)

    tolerance = 2e-5 if device == "mps" else 2e-12
    assert isinstance(actual, float)
    assert complex_parts[0].device.type == device
    assert complex_parts[0].dtype == complex_dtype
    assert complex_parts[1].dtype == torch_dtype
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("phase", (False, True))
def test_distance_interpolator_stays_on_torch_device(
        torch_device_ctx, monkeypatch, phase):
    ctx, device = torch_device_ctx
    real_dtype = torch.float32 if device == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if device == "mps" else torch.complex128
    )
    distance_rescale = np.linspace(0.5, 1.5, 31)
    distance_weights = np.linspace(0.75, 1.25, 31)
    distance_weights /= distance_weights.sum()
    distance = distance_rescale, distance_weights
    interpolator = _INFERENCE_TOOLS.setup_distance_marg_interpolant(
        distance,
        phase=phase,
        snr_range=(1, 10),
        density=(12, 13),
    )
    sh = np.array([2.25 + 0.3j, 3.75 - 0.6j, 6.5 + 0.8j])
    hh = np.array([1.5, 2.75, 5.25])
    expected_vector = _INFERENCE_TOOLS.marginalize_likelihood(
        sh,
        hh,
        phase=phase,
        distance=distance,
        interpolator=interpolator,
        skip_vector=True,
    )
    expected = _INFERENCE_TOOLS.marginalize_likelihood(
        sh,
        hh,
        phase=phase,
        distance=distance,
        interpolator=interpolator,
    )
    expected_peak = _INFERENCE_TOOLS.marginalize_likelihood(
        sh,
        hh,
        phase=phase,
        distance=distance,
        interpolator=interpolator,
        return_peak=True,
    )

    with ctx:
        sh_tensor = torch.tensor(
            sh, device=device, dtype=complex_dtype, requires_grad=True
        )
        hh_tensor = torch.tensor(
            hh, device=device, dtype=real_dtype, requires_grad=True
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_value):
                raise AssertionError(
                    "Distance interpolation copied live values to the host"
                )

            def _reject_scipy_evaluation(*_args, **_kwargs):
                raise AssertionError(
                    "Distance interpolation evaluated live values with SciPy"
                )

            patch.setattr(
                _INFERENCE_TOOLS, "_numpy_from_torch", _reject_host_transfer
            )
            patch.setattr(
                scipy.interpolate.RectBivariateSpline,
                "__call__",
                _reject_scipy_evaluation,
            )
            actual_vector = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_tensor,
                hh_tensor,
                phase=phase,
                distance=distance,
                interpolator=interpolator,
                skip_vector=True,
            )
            actual = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_tensor,
                hh_tensor,
                phase=phase,
                distance=distance,
                interpolator=interpolator,
            )
            actual_peak = _INFERENCE_TOOLS.marginalize_likelihood(
                sh_tensor,
                hh_tensor,
                phase=phase,
                distance=distance,
                interpolator=interpolator,
                return_peak=True,
            )
            bounds = interpolator(
                torch.tensor([0.1, 2.0], device=device, dtype=real_dtype),
                torch.ones(2, device=device, dtype=real_dtype),
            )
            actual_vector.sum().backward()

    tolerance = 5e-5 if device == "mps" else 5e-12
    assert actual_vector.device.type == device
    assert actual_vector.dtype == real_dtype
    assert isinstance(actual, float)
    assert bounds.device.type == device
    assert torch.isneginf(bounds[0])
    assert torch.isfinite(bounds[1])
    assert torch.isfinite(sh_tensor.grad.real).all()
    assert torch.isfinite(hh_tensor.grad).all()
    assert sh_tensor.grad.abs().sum() > 0
    assert hh_tensor.grad.abs().sum() > 0
    np.testing.assert_allclose(
        actual_vector.detach().cpu().numpy(),
        expected_vector,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    assert actual_peak[1] == expected_peak[1]
    np.testing.assert_allclose(
        (actual_peak[0], actual_peak[2]),
        (expected_peak[0], expected_peak[2]),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("reconstruction", ("distance", "phase"))
def test_marginalized_reconstruction_stays_on_device(
        torch_device_ctx, monkeypatch, reconstruction):
    ctx, device = torch_device_ctx
    real_dtype = torch.float32 if device == "mps" else torch.float64
    complex_dtype = torch.complex64 if device == "mps" else torch.complex128
    model = _INFERENCE_TOOLS.DistMarg.__new__(_INFERENCE_TOOLS.DistMarg)
    model.marginalize_vector_params = {}
    model.marginalize_phase = reconstruction == "phase"
    model.reconstruct_phase = False
    model.lognl = -3.25
    seen = {}
    original_draw_sample = _INFERENCE_TOOLS.draw_sample

    def _record_device(values, size=None):
        tensor = _INFERENCE_TOOLS._torch_tensor(values)
        assert tensor is not None
        seen["device"] = tensor.device.type
        seen["count"] = tensor.numel()
        return original_draw_sample(values, size=size)

    monkeypatch.setattr(_INFERENCE_TOOLS, "draw_sample", _record_device)

    with ctx:
        if reconstruction == "distance":
            dist_locs = np.linspace(100.0, 900.0, 73)
            dist_weights = np.linspace(0.5, 1.5, len(dist_locs))
            dist_weights /= dist_weights.sum()
            model.dist_locs = dist_locs
            model.distance_marginalization = (
                500.0 / dist_locs, dist_weights)
            source = torch.linspace(
                -4.0, 3.0, len(dist_locs),
                dtype=real_dtype, device=device)
            result = model.reconstruct(
                seed=417, set_loglr=lambda: source)
            selected = np.flatnonzero(dist_locs == result["distance"])[0]
            expected_loglr = source[selected].item()
            expected_count = len(dist_locs)
        else:
            model.distance_marginalization = False
            sh = torch.tensor(
                complex(4.25, -0.75), dtype=complex_dtype, device=device)
            hh = torch.tensor(-1.125, dtype=real_dtype, device=device)
            result = model.reconstruct(
                seed=417, set_loglr=lambda: (sh, hh))
            expected_loglr = (
                np.exp(-2.0j * result["coa_phase"])
                * complex(sh.item())
            ).real + hh.item()
            expected_count = int(1e4)

    tolerance = 2e-5 if device == "mps" else 2e-12
    assert seen == {"device": device, "count": expected_count}
    assert isinstance(result["loglr"], float)
    assert isinstance(result["loglikelihood"], float)
    np.testing.assert_allclose(
        result["loglr"], expected_loglr,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        result["loglikelihood"], model.lognl + expected_loglr,
        rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("interpolate", (None, "linear", "quadratic"))
def test_timeseries_at_time_stays_on_device(
        torch_device_ctx, monkeypatch, interpolate):
    ctx, device = torch_device_ctx
    dtype = np.complex64 if device == "mps" else np.complex128
    epoch, delta_t = 1_000_000_000, 1 / 1024
    samples = np.arange(64, dtype=np.float64)
    data = (np.sin(samples / 7.0) + 1j * np.cos(samples / 9.0)).astype(
        dtype)
    times = epoch + delta_t * np.array((-3.2, 5.25, 17.6, 63.4, 68.0))
    reference = TimeSeries(data, delta_t=delta_t, epoch=epoch).at_time(
        times, interpolate=interpolate, extrapolate=0.0j)
    scalar_reference = TimeSeries(
        data, delta_t=delta_t, epoch=epoch
    ).at_time(
        float(times[2]), interpolate=interpolate, extrapolate=0.0j)

    with ctx:
        series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Time lookup copied its series to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = series.at_time(
                times, interpolate=interpolate, extrapolate=0.0j)
            scalar_actual = series.at_time(
                float(times[2]), interpolate=interpolate, extrapolate=0.0j)

    tolerance = 2e-5 if device == "mps" else 2e-12
    assert actual.device.type == device
    assert scalar_actual.device.type == device
    assert scalar_actual.ndim == 0
    np.testing.assert_allclose(
        actual.cpu().numpy(), reference,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        scalar_actual.cpu().numpy(), scalar_reference,
        rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("query_kind", ("tensor", "array"))
def test_timeseries_at_time_device_queries_do_not_visit_host(
        torch_device_ctx, monkeypatch, query_kind):
    ctx, device = torch_device_ctx
    epoch, delta_t = 10.0, 1 / 256
    dtype = np.float32 if device == "mps" else np.float64
    samples = np.arange(32, dtype=np.float64)
    data = np.sin(samples / 5.0).astype(dtype)
    times = epoch + delta_t * np.array((-1.0, 3.25, 11.5, 35.0))
    reference_series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
    reference = reference_series.at_time(
        times, interpolate="linear", extrapolate=-2.0)
    scalar_reference = reference_series.at_time(
        float(times[2]), interpolate="linear")

    with ctx:
        series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
        query_dtype = torch.float32 if device == "mps" else torch.float64
        query_tensor = torch.as_tensor(
            times, dtype=query_dtype, device=device)
        if query_kind == "array":
            query = Array(times, dtype=dtype)
        else:
            query = query_tensor
        scalar_query = torch.as_tensor(
            times[2], dtype=query_dtype, device=device)

        with monkeypatch.context() as patch:
            def _reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("Time lookup copied a tensor to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = series.at_time(
                query, interpolate="linear", extrapolate=-2.0)
            scalar_actual = series.at_time(
                scalar_query, interpolate="linear")

    tolerance = 2e-5 if device == "mps" else 2e-12
    assert actual.device.type == device
    assert scalar_actual.device.type == device
    assert scalar_actual.ndim == 0
    np.testing.assert_allclose(
        actual.cpu().numpy(), reference,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        scalar_actual.cpu().numpy(), scalar_reference,
        rtol=tolerance, atol=tolerance)


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
        else {"rtol": 1e-3, "atol": 1e-5}
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
    assert float(plus.start_time) == pytest.approx(expected_epoch, abs=1e-6)
    assert float(cross.start_time) == pytest.approx(expected_epoch, abs=1e-6)
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
        assert float(actual.start_time) == pytest.approx(expected_epoch, abs=1e-6)
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


def test_live_batch_veto_torch_uses_two_bulk_copies(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    chisq_values = [1.0, 4.0, np.nan, np.inf]
    dof_values = [2, 2, 2, 2]
    sg_values = [None, 1.5, 3.5, 7.0]
    snr_values = [6.0, 8.0, 6.0, 10.0]
    threshold = 5.0

    expected_power, expected_sg = _stub_live_batch_veto_calculators(
        chisq_values, dof_values, sg_values, torch_backed=False
    )
    expected_results, expected_veto_info = _stub_live_batch_veto_inputs(
        snr_values
    )
    with monkeypatch.context() as patch:
        patch.setattr(matchedfilter, "correlate", lambda *_args: None)
        expected = _stub_live_batch_veto_filter(
            expected_power, expected_sg, threshold
        )._process_vetoes(expected_results, expected_veto_info)

    cpu_calls = []
    original_cpu = torch.Tensor.cpu

    def count_bulk_copy(tensor, *args, **kwargs):
        cpu_calls.append((tuple(tensor.shape), tensor.dtype, tensor.device.type))
        return original_cpu(tensor, *args, **kwargs)

    def reject_scalar_copy(_tensor, *_args, **_kwargs):
        raise AssertionError("live veto aggregation synchronized a scalar")

    def reject_device_division(_tensor, *_args, **_kwargs):
        raise AssertionError("live veto aggregation divided on the device")

    with ctx:
        actual_power, actual_sg = _stub_live_batch_veto_calculators(
            chisq_values, dof_values, sg_values, torch_backed=True
        )
        actual_results, actual_veto_info = _stub_live_batch_veto_inputs(
            snr_values
        )
        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "correlate", lambda *_args: None)
            patch.setattr(torch.Tensor, "cpu", count_bulk_copy)
            patch.setattr(torch.Tensor, "item", reject_scalar_copy)
            patch.setattr(
                torch.Tensor, "__truediv__", reject_device_division
            )
            patch.setattr(
                TorchArrayData,
                "numpy",
                lambda _self: (_ for _ in ()).throw(
                    AssertionError("live veto aggregation copied each array")
                ),
            )
            actual = _stub_live_batch_veto_filter(
                actual_power, actual_sg, threshold
            )._process_vetoes(actual_results, actual_veto_info)

    assert cpu_calls == [
        ((2, len(chisq_values)), torch.float32, device),
        ((len(chisq_values),), torch.int64, device),
    ]
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)
        assert actual[key].dtype == expected[key].dtype


@pytest.mark.parametrize("dof_value", [0, -1, 2**32])
def test_live_batch_veto_torch_invalid_dof_keeps_scalar_error(
        torch_device_ctx, monkeypatch, dof_value):
    ctx, device = torch_device_ctx

    class InvalidDofPowerChisq(chisq.SingleDetPowerChisq):
        def values(self, *_args):
            return (
                Array(np.array([2.0], dtype=np.float32)),
                Array(np.array([dof_value], dtype=np.int64)),
            )

    class UnreachedSGChisq(sgchisq.SingleDetSGChisq):
        def values(self, *_args):
            raise AssertionError(
                "legacy invalid-DOF path evaluated SG chi-squared"
            )

    cpu_calls = []
    original_cpu = torch.Tensor.cpu

    def count_bulk_copy(tensor, *args, **kwargs):
        cpu_calls.append((tuple(tensor.shape), tensor.dtype, tensor.device.type))
        return original_cpu(tensor, *args, **kwargs)

    def reject_scalar_copy(_tensor, *_args, **_kwargs):
        raise AssertionError("invalid-DOF aggregation synchronized a scalar")

    def reject_device_division(_tensor, *_args, **_kwargs):
        raise AssertionError("invalid-DOF aggregation divided on the device")

    with ctx:
        expected_error = ZeroDivisionError if dof_value == 0 else OverflowError
        legacy_results, legacy_veto_info = _stub_live_batch_veto_inputs([8.0])
        legacy_batch = _stub_live_batch_veto_filter(
            InvalidDofPowerChisq(), object.__new__(UnreachedSGChisq)
        )
        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "correlate", lambda *_args: None)
            with pytest.raises(expected_error) as legacy_error:
                legacy_batch._process_vetoes(
                    legacy_results, legacy_veto_info
                )

        power, sg_veto = _stub_live_batch_veto_calculators(
            [2.0], [dof_value], [None], torch_backed=True
        )
        results, veto_info = _stub_live_batch_veto_inputs([8.0])
        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "correlate", lambda *_args: None)
            patch.setattr(torch.Tensor, "cpu", count_bulk_copy)
            patch.setattr(torch.Tensor, "item", reject_scalar_copy)
            patch.setattr(
                torch.Tensor, "__truediv__", reject_device_division
            )
            patch.setattr(
                TorchArrayData,
                "numpy",
                lambda _self: (_ for _ in ()).throw(
                    AssertionError("invalid-DOF aggregation copied each array")
                ),
            )
            with pytest.raises(type(legacy_error.value)) as batched_error:
                _stub_live_batch_veto_filter(
                    power, sg_veto
                )._process_vetoes(results, veto_info)

    assert str(batched_error.value) == str(legacy_error.value)
    if dof_value == 0:
        assert str(legacy_error.value) == "float division by zero"
    assert cpu_calls == [
        ((2, 1), torch.float32, device),
        ((1,), torch.int64, device),
    ]


def test_live_batch_veto_torch_requires_vector_singletons(torch_ctx):
    with torch_ctx:
        chisq_vector = Array(np.array([2.0], dtype=np.float32))
        dof_vector = Array(np.array([2], dtype=np.int64))
        sg_vector = Array(np.array([3.0], dtype=np.float32))
        chisq_matrix = Array(np.array([[2.0]], dtype=np.float32))
        dof_matrix = Array(np.array([[2]], dtype=np.int64))
        sg_matrix = Array(np.array([[3.0]], dtype=np.float32))

        assert matchedfilter._materialize_torch_veto_results([
            (chisq_matrix, dof_vector, sg_vector)
        ]) is None
        assert matchedfilter._materialize_torch_veto_results([
            (chisq_vector, dof_matrix, sg_vector)
        ]) is None
        assert matchedfilter._materialize_torch_veto_results([
            (chisq_vector, dof_vector, sg_matrix)
        ]) is None


def test_live_batch_veto_torch_custom_calculators_keep_scalar_order(
        torch_ctx, monkeypatch):
    class CustomPowerChisq(chisq.SingleDetPowerChisq):
        def values(self, *_args):
            self.last_chisq = Array(np.array([6.0], dtype=np.float32))
            return self.last_chisq, Array(np.array([3], dtype=np.int64))

    class MutatingSGChisq(sgchisq.SingleDetSGChisq):
        def values(self, *_args):
            power_chisq = _args[-3]
            power_chisq[0] = 99.0
            return Array(np.array([4.0], dtype=np.float32))

    results, veto_info = _stub_live_batch_veto_inputs([8.0])
    with torch_ctx:
        batch = _stub_live_batch_veto_filter(
            CustomPowerChisq(), object.__new__(MutatingSGChisq)
        )
        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "correlate", lambda *_args: None)
            patch.setattr(
                matchedfilter,
                "_materialize_torch_veto_results",
                lambda *_args: (_ for _ in ()).throw(
                    AssertionError("custom veto calculators were aggregated")
                ),
            )
            actual = batch._process_vetoes(results, veto_info)

    np.testing.assert_array_equal(actual["chisq"], [2.0])
    np.testing.assert_array_equal(actual["chisq_dof"], [3])
    np.testing.assert_array_equal(actual["sg_chisq"], [4.0])
    assert actual["chisq"].dtype == np.dtype(np.float32)
    assert actual["chisq_dof"].dtype == np.dtype(np.uint32)
    assert actual["sg_chisq"].dtype == np.dtype(np.float32)


def test_live_batch_veto_torch_mixed_outputs_fall_back(
        torch_ctx, monkeypatch):
    state = {"index": 0}
    power = object.__new__(chisq.SingleDetPowerChisq)
    sg_veto = object.__new__(sgchisq.SingleDetSGChisq)

    def power_values(*_args):
        index = state["index"]
        if index == 0:
            return (
                Array(np.array([6.0], dtype=np.float32)),
                Array(np.array([3], dtype=np.int64)),
            )
        return (
            np.array([9.0], dtype=np.float32),
            np.array([2], dtype=np.int64),
        )

    def sg_veto_values(*_args):
        index = state["index"]
        state["index"] += 1
        if index == 0:
            return Array(np.array([2.0], dtype=np.float32))
        return None

    power.values = power_values
    sg_veto.values = sg_veto_values
    results, veto_info = _stub_live_batch_veto_inputs([8.0, 9.0])

    with torch_ctx:
        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "correlate", lambda *_args: None)
            patch.setattr(
                torch.Tensor,
                "cpu",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("mixed veto outputs used the bulk copy")
                ),
            )
            actual = _stub_live_batch_veto_filter(
                power, sg_veto
            )._process_vetoes(results, veto_info)

    np.testing.assert_array_equal(actual["chisq"], [2.0, 4.5])
    np.testing.assert_array_equal(actual["chisq_dof"], [3, 2])
    np.testing.assert_array_equal(actual["sg_chisq"], [2.0, 0.0])
    assert actual["chisq"].dtype == np.dtype(np.float32)
    assert actual["chisq_dof"].dtype == np.dtype(np.uint32)
    assert actual["sg_chisq"].dtype == np.dtype(np.float32)


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

            def _reject_lal_harmonic(*_args, **_kwargs):
                raise AssertionError(
                    "time-domain ringdown evaluated harmonics with LAL"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
            patch.setattr(
                ringdown.lal,
                "SpinWeightedSphericalHarmonic",
                _reject_lal_harmonic,
            )
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

            def _reject_lal_harmonic(*_args, **_kwargs):
                raise AssertionError(
                    "frequency-domain ringdown evaluated harmonics with LAL"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
            patch.setattr(
                ringdown.lal,
                "SpinWeightedSphericalHarmonic",
                _reject_lal_harmonic,
            )
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


def test_reproducible_normal_is_device_native_and_overlap_safe(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    monkeypatch.setattr(reproduceable, "BLOCK_SAMPLES", 512)
    sample_rate = 64

    with ctx:
        global_rng_state = torch.random.get_rng_state()

        with monkeypatch.context() as patch:
            def _reject_numpy_block(*_args, **_kwargs):
                raise AssertionError("Torch white noise used a NumPy block")

            def _reject_host_transfer(_self):
                raise AssertionError("Torch white noise copied data to host")

            patch.setattr(reproduceable, "block", _reject_numpy_block)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            first = reproduceable.normal(
                100, 110, sample_rate=sample_rate, seed=1729
            )
            repeated = reproduceable.normal(
                100, 110, sample_rate=sample_rate, seed=1729
            )
            overlapping = reproduceable.normal(
                104, 112, sample_rate=sample_rate, seed=1729
            )
            different = reproduceable.normal(
                100, 110, sample_rate=sample_rate, seed=1730
            )
            first_overlap = first.time_slice(104, 110)._data.tensor.clone()
            second_overlap = overlapping.time_slice(
                104, 110
            )._data.tensor.clone()

        final_rng_state = torch.random.get_rng_state()

    assert first._data.tensor.device.type == device
    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert first._data.tensor.dtype == expected_dtype
    assert first.start_time == repeated.start_time == 100
    assert first.end_time == repeated.end_time == 110
    assert torch.equal(first._data.tensor, repeated._data.tensor)
    assert torch.equal(first_overlap, second_overlap)
    assert not torch.equal(first._data.tensor, different._data.tensor)
    assert torch.equal(global_rng_state, final_rng_state)


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
            repeated = reproduceable.colored_noise(
                torch_psd, start_time, end_time, **parameters
            )

    assert actual._data.tensor.device.type == device
    assert actual.start_time == repeated.start_time == start_time
    assert actual.end_time == repeated.end_time == end_time
    assert torch.equal(actual._data.tensor, repeated._data.tensor)


def test_lisa_response_data_loader_is_cached_and_padded(
        tmp_path, monkeypatch):
    import astropy.utils.data

    response_path = tmp_path / "lisa-response.npy"
    np.save(
        response_path,
        np.array(((1e-4, 1e-2), (2.25, 1.25))),
    )
    download_calls = []

    def _download_file(url, cache):
        download_calls.append((url, cache))
        return response_path

    monkeypatch.setattr(astropy.utils.data, "download_file", _download_file)
    analytical_space._load_lisa_averaged_response_data.cache_clear()
    try:
        first = analytical_space._load_lisa_averaged_response_data()
        second = analytical_space._load_lisa_averaged_response_data()
    finally:
        analytical_space._load_lisa_averaged_response_data.cache_clear()

    assert first is second
    assert len(download_calls) == 1
    np.testing.assert_array_equal(first[0], (1e-4, 1e-2, 2.0))
    np.testing.assert_array_equal(
        first[1], (2.25, 1.25, 0.0012712348970728724)
    )
    assert not first[0].flags.writeable
    assert not first[1].flags.writeable


@pytest.mark.parametrize("model_name, parameters", _DIRECT_SPACE_RESPONSE_HELPERS)
def test_analytical_space_response_helpers_preserve_torch(
        torch_device_ctx, monkeypatch, model_name, parameters):
    ctx, device = torch_device_ctx
    model = getattr(analytical_space, model_name)
    frequencies = np.array((1e-4, 1e-2, 0.4, 1.2, 1.8))
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    expected = model(frequencies, **parameters)
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        tensor_frequencies = torch.tensor(
            frequencies,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError(
                    "Torch space-detector response used NumPy/SciPy"
                )

            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            patch.setattr(analytical_space.np, "polyval", _reject_host_path)
            patch.setattr(analytical_space.np, "exp", _reject_host_path)
            patch.setattr(analytical_space.np, "sin", _reject_host_path)
            patch.setattr(analytical_space.np, "multiply", _reject_host_path)
            patch.setattr(
                analytical_space.np, "concatenate", _reject_host_path
            )
            actual = model(tensor_frequencies, **parameters)
            actual.sum().backward()

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == device
    assert actual.dtype == dtype
    assert tensor_frequencies.grad is not None
    assert torch.all(torch.isfinite(tensor_frequencies.grad))
    assert torch.count_nonzero(tensor_frequencies.grad) > 0
    if dtype == torch.float32:
        assert _relative_l2(
            actual.detach().cpu().numpy(), expected
        ) < 1e-5
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=dtype, device=device),
        rtol=2e-3 if dtype == torch.float32 else 5e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "model_name",
    (
        "averaged_response_lisa_tdi",
        "averaged_response_tianqin_tdi",
        "averaged_response_taiji_tdi",
    ),
)
def test_analytical_space_response_helpers_reject_invalid_tdi(
        monkeypatch, model_name):
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    frequencies = torch.tensor((1e-4, 1e-2), dtype=torch.float64)
    with pytest.raises(ValueError, match="currently only for 1.5 or 2.0"):
        getattr(analytical_space, model_name)(frequencies, tdi="3.0")


@pytest.mark.parametrize(
    "model_name",
    (
        "averaged_lisa_fplus_sq_numerical",
        "averaged_response_lisa_tdi",
    ),
)
def test_lisa_response_helpers_validate_arm_before_loading(
        monkeypatch, model_name):
    def _reject_loader():
        raise AssertionError("invalid LISA arm length loaded response data")

    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        _reject_loader,
    )
    frequencies = torch.tensor((1e-4, 1e-2), dtype=torch.float64)
    with pytest.raises(ValueError, match="len_arm=2.5e9"):
        getattr(analytical_space, model_name)(frequencies, len_arm=3e9)


def test_lisa_response_helper_promotes_integer_tensor(monkeypatch):
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    frequencies = torch.tensor((1, 2), dtype=torch.int64)
    actual = analytical_space.averaged_lisa_fplus_sq_numerical(frequencies)
    expected = analytical_space.averaged_lisa_fplus_sq_numerical(
        frequencies.numpy()
    )
    assert actual.dtype == torch.get_default_dtype()
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=actual.dtype),
    )


@pytest.mark.parametrize(
    "model_name, extra",
    (
        (
            "sensitivity_curve_taiji_confusion",
            {"duration": 2.0},
        ),
        (
            "analytical_psd_tianqin_tdi_AE_confusion",
            {"duration": 2.0, "tdi": "2.0"},
        ),
    ),
)
def test_analytical_space_confusion_public_devices(
        torch_device_ctx, monkeypatch, model_name, extra):
    ctx, device = torch_device_ctx
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=257,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        **extra,
    )
    expected = model(**parameters).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch confusion model used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    model(**parameters)
                return
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert _relative_l2(actual.numpy(), expected) < 5e-12


def test_analytical_space_curve_public_devices(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=257,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        duration=2.0,
    )
    expected = analytical_space.confusion_fit_taiji(**parameters).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical curve used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_space.confusion_fit_taiji(**parameters)
                return
            actual = analytical_space.confusion_fit_taiji(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    assert _relative_l2(actual.numpy(), expected) < 5e-12


@pytest.mark.parametrize(
    "model_name, duration",
    (("confusion_fit_tianqin", 3.0), ("confusion_fit_taiji", 3.0)),
)
def test_analytical_space_curve_preserves_duration_error(
        torch_ctx, model_name, duration):
    with torch_ctx:
        with pytest.raises(Warning, match="extrapolated"):
            getattr(analytical_space, model_name)(
                257,
                5e-5,
                3.7e-4,
                duration=duration,
            )


@pytest.mark.parametrize(
    "model_name, extra",
    (
        (
            "sensitivity_curve_lisa_confusion",
            {"base_model": "SciRD", "duration": 11.0},
        ),
        (
            "sensitivity_curve_tianqin_confusion",
            {"duration": 6.0},
        ),
    ),
)
def test_analytical_space_combined_curve_preserves_duration_error(
        torch_ctx, model_name, extra):
    with torch_ctx:
        with pytest.raises(ValueError, match="Must between"):
            getattr(analytical_space, model_name)(
                257,
                5e-5,
                3.7e-4,
                **extra,
            )


def test_aligo_analytical_family_torch_matches_lalsimulation(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=1031,
        delta_f=0.375,
        low_freq_cutoff=9.1,
    )

    if device == "mps":
        # Preserve the existing REAL8 adapter failure: float32 cannot
        # represent most of this PSD's physical range.
        with ctx:
            for model_name in _ALIGO_TORCH_100_HZ_PINS:
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
        for model_name in _ALIGO_TORCH_100_HZ_PINS
    }

    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch ground PSD called LALSimulation")

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
                for model_name in _ALIGO_TORCH_100_HZ_PINS
            }
            wrapper_actual = analytical_psd.aLIGOThermal(
                **parameters
            )

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
            rtol=2e-12,
            atol=0.0,
        )
    torch.testing.assert_close(
        wrapper_actual._data.tensor,
        actual["aLIGOThermal"]._data.tensor,
    )


def test_aligo_analytical_torch_layout_and_cutoff(
        torch_ctx):
    with torch_ctx:
        psd = analytical_psd.from_string(
            "aLIGOZeroDetHighPower",
            length=1002,
            delta_f=1.0,
            low_freq_cutoff=10.9,
            ignored_lalsimulation_keyword=True,
        )
        one_bin = analytical_psd.aLIGOZeroDetHighPower(1, 1.0, 0.0)
        two_bins = analytical_psd.aLIGOZeroDetHighPower(2, 1.0, 0.0)
        negative_cutoff = analytical_psd.aLIGOZeroDetHighPower(
            8,
            1.0,
            -1.0,
        )

    tensor = psd._data.tensor
    assert torch.count_nonzero(tensor[:10]) == 0
    assert tensor[10] != 0
    assert tensor[-1] == 0
    torch.testing.assert_close(
        tensor[[10, 100, 1000]],
        torch.tensor(
            [
                9.4667343534679125e-45,
                1.5232930939569254e-47,
                2.7315382749747422e-47,
            ],
            dtype=torch.float64,
        ),
        rtol=5e-15,
        atol=0.0,
    )
    assert torch.count_nonzero(one_bin._data.tensor) == 0
    assert torch.count_nonzero(two_bins._data.tensor) == 0
    # Preserve the public Python-slice behavior for negative cutoffs.
    assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_aligo_analytical_family_source_pins_and_composition(torch_ctx):
    with torch_ctx:
        actual = {
            model_name: analytical_psd.from_string(
                model_name,
                length=102,
                delta_f=1.0,
                low_freq_cutoff=0.0,
            )._data.tensor
            for model_name in _ALIGO_TORCH_100_HZ_PINS
        }

    torch.testing.assert_close(
        torch.stack([values[100] for values in actual.values()]),
        torch.tensor(
            list(_ALIGO_TORCH_100_HZ_PINS.values()),
            dtype=torch.float64,
        ),
        rtol=5e-15,
        atol=0.0,
    )
    thermal = actual["aLIGOThermal"]
    for configuration in _ALIGO_TORCH_CONFIGURATIONS:
        torch.testing.assert_close(
            actual[f"aLIGO{configuration}"],
            actual[f"aLIGOQuantum{configuration}"] + thermal,
            rtol=0.0,
            atol=0.0,
        )


def test_iligo_analytical_family_torch_matches_lalsimulation(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=1031,
        delta_f=0.375,
        low_freq_cutoff=9.1,
    )

    if device == "mps":
        with ctx:
            for model_name in _ILIGO_TORCH_100_HZ_PINS:
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
        for model_name in _ILIGO_TORCH_100_HZ_PINS
    }

    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch i/eLIGO PSD called LALSimulation")

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
                for model_name in _ILIGO_TORCH_100_HZ_PINS
            }
            wrapper_actual = analytical_psd.iLIGOModel(**parameters)

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
            rtol=2e-13,
            atol=0.0,
        )
    torch.testing.assert_close(
        wrapper_actual._data.tensor,
        actual["iLIGOModel"]._data.tensor,
    )


def test_iligo_analytical_layout_cutoff_source_pins_and_composition(
        torch_ctx):
    with torch_ctx:
        actual = {
            model_name: analytical_psd.from_string(
                model_name,
                length=1002,
                delta_f=1.0,
                low_freq_cutoff=0.0,
                ignored_lalsimulation_keyword=True,
            )._data.tensor
            for model_name in _ILIGO_TORCH_100_HZ_PINS
        }
        cutoff = analytical_psd.iLIGOModel(1002, 1.0, 10.9)
        one_bin = analytical_psd.iLIGOModel(1, 1.0, 0.0)
        two_bins = analytical_psd.iLIGOModel(2, 1.0, 0.0)
        negative_cutoff = analytical_psd.iLIGOModel(8, 1.0, -1.0)

    cutoff_values = cutoff._data.tensor
    assert torch.count_nonzero(cutoff_values[:10]) == 0
    assert cutoff_values[10] != 0
    assert cutoff_values[-1] == 0
    torch.testing.assert_close(
        torch.stack([values[100] for values in actual.values()]),
        torch.tensor(
            list(_ILIGO_TORCH_100_HZ_PINS.values()),
            dtype=torch.float64,
        ),
        rtol=6e-15,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["iLIGOSeismic"][[9, 10, 11]],
        torch.tensor(
            [
                1.7193750209941767e-29,
                2.0903586303906163e-30,
                2.1222490997393695e-31,
            ],
            dtype=torch.float64,
        ),
        rtol=6e-15,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["iLIGOModel"],
        actual["iLIGOShot"]
        + actual["iLIGOSeismic"]
        + actual["iLIGOThermal"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["eLIGOModel"],
        actual["eLIGOShot"]
        + actual["iLIGOSeismic"]
        + actual["iLIGOThermal"],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(one_bin._data.tensor) == 0
    assert torch.count_nonzero(two_bins._data.tensor) == 0
    assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_torch_analytical_preserves_lalsimulation_fallbacks(
        torch_ctx, monkeypatch):
    original = analytical_psd.lalsimulation.SimNoisePSD
    file_model_name = "aLIGOZeroDetHighPowerGWINC"
    file_model = getattr(
        analytical_psd.lalsimulation,
        f"SimNoisePSD{file_model_name}",
    )
    calls = []
    file_calls = []

    def record_lal_call(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    def record_file_call(*args, **kwargs):
        file_calls.append(args)
        return file_model(*args, **kwargs)

    monkeypatch.setattr(
        analytical_psd.lalsimulation,
        "SimNoisePSD",
        record_lal_call,
    )
    monkeypatch.setattr(
        analytical_psd.lalsimulation,
        f"SimNoisePSD{file_model_name}",
        record_file_call,
    )
    for model_name in ("aLIGOZeroDetHighPower", "Virgo"):
        analytical_psd.from_string(
            model_name,
            33,
            1.0,
            9.0,
        )
    assert len(calls) == 2
    analytical_psd.from_string(
        file_model_name,
        33,
        1.0,
        9.0,
    )
    assert len(file_calls) == 1

    with torch_ctx:
        analytical_psd.from_string(
            file_model_name,
            33,
            1.0,
            9.0,
        )
    assert len(calls) == 2
    assert len(file_calls) == 1


@pytest.mark.parametrize(
    "model_name",
    ("aLIGOThermal", "Virgo", "iLIGOModel"),
)
@pytest.mark.parametrize(
    "delta_f",
    (
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(np.array(1.0), id="zero-dimensional-array"),
        pytest.param(torch.tensor(1.0), id="zero-dimensional-tensor"),
    ),
)
def test_torch_analytical_preserves_lalsimulation_real8_errors(
        torch_ctx, model_name, delta_f):
    with pytest.raises(TypeError, match="argument 4 of type 'REAL8'"):
        analytical_psd.from_string(model_name, 8, delta_f, 1.0)

    with torch_ctx:
        with pytest.raises(TypeError, match="argument 4 of type 'REAL8'"):
            analytical_psd.from_string(model_name, 8, delta_f, 1.0)


@pytest.mark.parametrize(
    "delta_f",
    (True, 1, 1.0, np.int64(1), np.float32(1.0)),
)
def test_torch_analytical_accepts_lalsimulation_real8_scalars(
        torch_ctx, monkeypatch, delta_f):
    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch analytical PSD called LALSimulation")

    with torch_ctx:
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical_psd.lal,
                "CreateREAL8FrequencySeries",
                reject_lal_path,
            )
            result = analytical_psd.from_string(
                "Virgo",
                8,
                delta_f,
                1.0,
            )

    assert result.delta_f == delta_f
    assert result._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    "model_name",
    ("aLIGOThermal", "Virgo", "iLIGOModel"),
)
def test_torch_analytical_does_not_require_lalsimulation_discovery(
        torch_ctx, monkeypatch, model_name):
    monkeypatch.setattr(
        analytical_psd,
        "get_lalsim_psd_list",
        lambda: [],
    )
    with torch_ctx:
        result = analytical_psd.from_string(
            model_name,
            33,
            1.0,
            9.0,
        )
    assert result._data.tensor.device.type == "cpu"


def test_aligo_zero_det_high_power_noise_stays_on_torch_device(
        torch_ctx, monkeypatch):
    def reject_lal_path(*_args, **_kwargs):
        raise AssertionError("Torch noise PSD called LALSimulation")

    with torch_ctx:
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
            noise = gaussian.noise_from_string(
                "aLIGOZeroDetHighPower",
                length=256,
                delta_t=1.0 / 256.0,
                seed=8128,
                low_frequency_cutoff=10.0,
            )

    assert noise._data.tensor.device.type == "cpu"
    assert noise._data.tensor.dtype == torch.float64
    assert torch.all(torch.isfinite(noise._data.tensor))


def test_strain_hann_window_is_created_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    length = 65
    expected = np.hanning(length).astype(np.float32)

    with ctx:
        strain = TimeSeries(
            np.zeros(128, dtype=np.float32), delta_t=1 / 256.0
        )

        def reject_numpy_window(*args, **kwargs):
            raise AssertionError("strain conditioning created a host window")

        with monkeypatch.context() as patch:
            patch.setattr(np, "hanning", reject_numpy_window)
            window = _hann_window_for_series(strain, length)

        assert window._data.tensor.device.type == device

    np.testing.assert_allclose(
        window.numpy(), expected, rtol=2e-5, atol=1e-7
    )


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


@pytest.mark.parametrize("trunc_method", [None, "hann"])
def test_inverse_spectrum_truncation_float32_stays_on_torch_device(
        torch_device_ctx, trunc_method):
    ctx, device = torch_device_ctx
    values = np.linspace(1.0, 4.0, 257, dtype=np.float32)
    expected = inverse_spectrum_truncation(
        FrequencySeries(values, delta_f=1.0),
        64,
        low_frequency_cutoff=1.0,
        trunc_method=trunc_method,
    )

    with ctx:
        actual = inverse_spectrum_truncation(
            FrequencySeries(values, delta_f=1.0),
            64,
            low_frequency_cutoff=1.0,
            trunc_method=trunc_method,
        )
        assert actual._data.tensor.device.type == device
        assert actual._data.tensor.dtype == torch.float32

    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=2e-5, atol=1e-7
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


@pytest.mark.parametrize(
    "device,dtype,length,lfilter_dtype,whiten_dtype",
    (
        ("cpu", torch.float32, 4096, torch.float32, None),
        ("cuda", torch.float32, 4096, torch.float64, torch.float64),
        ("cuda", torch.float32, 4095, torch.float64, None),
        ("cuda", torch.complex64, 4096, torch.complex128, None),
        ("cuda", torch.float64, 4096, torch.float64, None),
        ("mps", torch.float32, 4096, torch.float32, None),
    ),
)
def test_torch_conditioning_work_dtype_policy(
        device, dtype, length, lfilter_dtype, whiten_dtype):
    data = types.SimpleNamespace(
        device=types.SimpleNamespace(type=device),
        dtype=dtype,
        numel=lambda: length,
    )

    assert resample._torch_lfilter_work_dtype(data) == lfilter_dtype
    assert timeseries_module._torch_whiten_work_dtype(data) == whiten_dtype


def test_whiten_double_final_fft_preserves_public_dtype_and_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support double precision")

    rng = np.random.default_rng(61784)
    data = rng.standard_normal(4096).astype(np.float32)
    rfft_dtypes = []
    irfft_dtypes = []
    original_rfft = torch.fft.rfft
    original_irfft = torch.fft.irfft

    def recording_rfft(values, *args, **kwargs):
        rfft_dtypes.append(values.dtype)
        return original_rfft(values, *args, **kwargs)

    def recording_irfft(values, *args, **kwargs):
        irfft_dtypes.append(values.dtype)
        return original_irfft(values, *args, **kwargs)

    with ctx:
        input_series = TimeSeries(data, delta_t=1 / 2048.0)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("whitening copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.fft, "rfft", recording_rfft)
            patch.setattr(torch.fft, "irfft", recording_irfft)
            if device == "cpu":
                # Exercise the CUDA-only promotion plumbing when CI has no
                # GPU; the policy itself is tested separately above.
                patch.setattr(
                    timeseries_module, "_torch_whiten_work_dtype",
                    lambda _data: torch.float64,
                )
            actual, psd = input_series.whiten(
                2, 1, remove_corrupted=False, return_psd=True
            )
        # Recreate the previous conservative implementation with the exact
        # same on-device PSD.  This isolates the direct FFT fast path from
        # expected CPU/CUDA differences in PSD estimation and conditioning.
        expected = (
            input_series.astype(np.float64).to_frequencyseries()
            / psd.astype(np.float64)**0.5
        ).to_timeseries().astype(np.float32)

    assert actual._data.tensor.device.type == device
    assert psd._data.tensor.device.type == device
    assert actual.dtype == np.dtype(np.float32)
    assert psd.dtype == np.dtype(np.float32)
    assert actual.delta_t == input_series.delta_t
    assert actual.start_time == input_series.start_time
    assert rfft_dtypes[-1] == torch.float64
    assert irfft_dtypes[-1] == torch.complex128
    torch.testing.assert_close(
        actual._data.tensor, expected._data.tensor, rtol=0, atol=0,
    )


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


@pytest.mark.parametrize("length", (0, 17))
def test_glitch_tapers_are_built_on_device(
        torch_device_ctx, monkeypatch, length):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    expected_rising = np.arange(length, dtype=dtype) / float(
        length if length else 1
    )

    with ctx:
        series = TimeSeries(
            np.ones(max(length, 1), dtype=dtype), delta_t=1 / 128
        )
        with monkeypatch.context() as patch:
            def _reject_numpy_ramp(*_args, **_kwargs):
                raise AssertionError("Glitch taper was built with NumPy")

            patch.setattr(np, "arange", _reject_numpy_ramp)
            rising, falling = _linear_tapers_for_series(series, length)

    assert rising._data.tensor.device.type == device
    assert falling._data.tensor.device.type == device
    np.testing.assert_allclose(
        rising._data.tensor.detach().cpu().numpy(), expected_rising
    )
    np.testing.assert_allclose(
        falling._data.tensor.detach().cpu().numpy(), expected_rising[::-1]
    )


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
    assert expected.corrupted_samples == 10
    assert actual.corrupted_samples == 10
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


def test_match_cache_does_not_cross_torch_scheme(torch_ctx, monkeypatch):
    """A cached Torch SNR buffer must not be copied into a later CPU call."""
    sample_count = 256
    frequency_count = sample_count // 2 + 1
    frequencies = np.arange(frequency_count, dtype=np.float64)
    data = np.exp(-frequencies / 40) * np.exp(0.03j * frequencies)

    monkeypatch.setattr(matchedfilter, "_snr", None)
    monkeypatch.setattr(matchedfilter, "_snr_scheme_key", None)

    cpu_first = FrequencySeries(data, delta_f=1.0)
    expected = matchedfilter.match(cpu_first, cpu_first)

    with torch_ctx:
        torch_series = FrequencySeries(data, delta_f=1.0)
        matchedfilter.match(torch_series, torch_series)
        assert isinstance(matchedfilter._snr._data, TorchArrayData)

    def _reject_host_transfer(_self):
        raise AssertionError("match cache copied a Torch buffer to the host")

    monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
    cpu_second = FrequencySeries(data, delta_f=1.0)
    actual = matchedfilter.match(cpu_second, cpu_second)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert isinstance(matchedfilter._snr._data, np.ndarray)


def test_fseries_time_shift_accepts_ligo_time_gps(torch_ctx):
    """Torch time shifting must accept the scalar type used for epochs."""
    data = np.array([1.0 + 0.5j, -0.25j, 0.75 - 1.0j])
    shift = lal.LIGOTimeGPS(0, 125_000_000)
    expected = waveform_utils.apply_fseries_time_shift(
        FrequencySeries(data, delta_f=0.5),
        shift,
    )

    with torch_ctx:
        actual = waveform_utils.apply_fseries_time_shift(
            FrequencySeries(data, delta_f=0.5),
            shift,
        )

    assert isinstance(actual._data, TorchArrayData)
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1e-9, atol=1e-12
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
def test_matching_line_summary_stays_on_device(
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
    expected = strain_lines.matching_line(
        13,
        TimeSeries(data, delta_t=delta_t, epoch=epoch),
        epoch,
        bin_size=0.5,
    )

    with ctx:
        series = TimeSeries(data, delta_t=delta_t, epoch=epoch)
        with monkeypatch.context() as patch:
            def _reject_host_scalar(*_args, **_kwargs):
                raise AssertionError(
                    "matching_line copied its Torch summary to the host"
                )

            patch.setattr(torch.Tensor, "cpu", _reject_host_scalar)
            patch.setattr(torch.Tensor, "tolist", _reject_host_scalar)
            patch.setattr(torch.Tensor, "item", _reject_host_scalar)
            patch.setattr(torch.Tensor, "__float__", _reject_host_scalar)
            actual = strain_lines.matching_line(
                13, series, epoch, bin_size=0.5
            )

    assert actual._data.tensor.device.type == device
    rtol = 3e-4 if device == "mps" else 3e-6
    atol = 3e-5 if device == "mps" else 3e-7
    if dtype == np.float64:
        rtol, atol = 2e-12, 2e-13
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )


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


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_physical_calibration_stays_on_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.complex128:
        pytest.skip("Torch MPS does not support complex128")

    data = np.linspace(1, 2, 137).astype(dtype)
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
        model = _physical_calibration_model()
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("calibration copied Torch data to host")

            def _reject_host_evaluation(*_args, **_kwargs):
                raise AssertionError("calibration evaluated on the host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(model, "update_r", _reject_host_evaluation)
            patch.setattr(
                recalibrate, "UnivariateSpline", _reject_host_evaluation
            )
            patch.setattr(recalibrate.np, "abs", _reject_host_evaluation)
            patch.setattr(recalibrate.np, "angle", _reject_host_evaluation)
            patch.setattr(recalibrate.np, "unwrap", _reject_host_evaluation)
            actual = model.adjust_strain(torch_strain, **adjustments)

    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_f == expected.delta_f
    assert actual.epoch == expected.epoch
    assert torch.equal(torch_strain._data.tensor, original)
    assert len(model._torch_baselines) == 1
    assert all(
        value.device.type == device
        for value in next(iter(model._torch_baselines.values())).values()
    )
    tolerance = 2e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected.numpy(),
        rtol=tolerance, atol=tolerance,
    )


def test_trigger_cuts_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    host_triggers = {
        "snr": np.array([4, 5, 8, 10, 12, 7], dtype=dtype),
        "end_time": np.array([1, 2, 3, 4, 5, 6], dtype=dtype),
        "chisq": np.array([2, 3, 8, 2, 14, 4], dtype=dtype),
        "chisq_dof": np.array([2, 2, 3, 2, 4, 2], dtype=dtype),
        "sg_chisq": np.array([1, 2, 2, 8, 2, 6], dtype=dtype),
    }
    cut_dict = {
        ("snr", np.greater_equal): 5,
        ("end_time", np.less): 6,
        ("traditional_chisq", np.less): 2.2,
        ("newsnr_sgveto", np.greater): 5,
    }
    expected = cuts.apply_trigger_cuts(host_triggers, cut_dict)

    with ctx:
        triggers = {
            key: Array(values)
            for key, values in host_triggers.items()
            if key != "end_time"
        }
        # Mixed host columns are copied directly to the active Torch device.
        triggers["end_time"] = host_triggers["end_time"]

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("trigger cuts copied Torch data to the host")

        def reject_numpy_indices(*_args, **_kwargs):
            raise AssertionError("trigger cuts built indices with NumPy")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            patch.setattr(cuts.np, "arange", reject_numpy_indices)
            actual = cuts.apply_trigger_cuts(triggers, cut_dict)
            full = cuts.apply_trigger_cuts(triggers, {})

    for result in (actual, full):
        assert isinstance(result, Array)
        assert isinstance(result._data, TorchArrayData)
        assert result._data.tensor.device.type == device
        assert result._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().cpu().numpy(), expected
    )
    np.testing.assert_array_equal(
        full._data.tensor.detach().cpu().numpy(),
        np.arange(len(host_triggers["snr"])),
    )


def test_trigger_cuts_preserve_raw_torch_indices(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    with ctx:
        triggers = {
            "snr": torch.tensor([3, 6, 9, 12], dtype=dtype, device=device),
            "end_time": torch.tensor(
                [1, 2, 3, 4], dtype=dtype, device=device
            ),
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("raw Torch trigger cuts left the device")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            actual = cuts.apply_trigger_cuts(
                triggers,
                {
                    ("snr", np.greater): 5,
                    ("end_time", np.less_equal): 3,
                },
            )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == device
    assert actual.dtype == torch.int64
    assert actual.detach().cpu().tolist() == [1, 2]

