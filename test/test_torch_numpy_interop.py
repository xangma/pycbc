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
    return (
        plus.data.data.copy(),
        cross.data.data.copy(),
        float(plus.epoch),
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


def test_base_inference_model_retains_numpy_scalar_contract():
    class _Prior:
        variable_args = ("x",)

        @staticmethod
        def apply_boundary_conditions(**params):
            return params

        def __call__(self, **_params):
            return np.nan

    class _Model(inference_model_base.BaseModel):
        likelihood_calls = 0

        def _loglikelihood(self):
            self.likelihood_calls += 1
            return 1.0

    model = _Model(("x",), prior=_Prior())
    model.update(x=0.25)
    assert model.logposterior == -np.inf
    assert model.likelihood_calls == 0


def test_lisa_coordinate_transforms_match_numpy(torch_ctx):
    ssb_values = (
        np.array([1.12625946243e9, 1.32625946243e9]),
        np.array([0.4, 4.8]),
        np.array([-0.3, 0.7]),
        np.array([0.2, 5.4]),
    )
    expected_lisa = coordinate_space.ssb_to_lisa(*ssb_values)
    expected_ssb = coordinate_space.lisa_to_ssb(*expected_lisa)

    with torch_ctx:
        torch_values = tuple(
            torch.as_tensor(value, dtype=torch.float64)
            for value in ssb_values
        )
        actual_lisa = coordinate_space.ssb_to_lisa(*torch_values)
        actual_ssb = coordinate_space.lisa_to_ssb(*actual_lisa)

    for actual, expected in zip(actual_lisa, expected_lisa):
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
    for actual, expected in zip(actual_ssb, expected_ssb):
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )


def test_numpy_take_keeps_torch_arrays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = TimeSeries(
            np.arange(6, dtype=dtype), delta_t=0.25, epoch=12.0
        )
        index_tensor = torch.tensor(
            [4, 0, 2], device=device, dtype=torch.int64
        )
        indices = Array(TorchArrayData(index_tensor), copy=False)
        output = Array(np.empty(3, dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.take copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            actual = np.take(values, indices)
            with_axis = np.take(values, indices, axis=0)
            returned = np.take(values, indices, axis=-1, out=output)

        expected = torch.tensor(
            [4.0, 0.0, 2.0], device=device, dtype=torch_dtype
        )
        for selected in (actual, with_axis, output):
            assert type(selected) is Array
            assert selected._data.tensor.device.type == device
            torch.testing.assert_close(selected._data.tensor, expected)
        assert returned is output


@pytest.mark.parametrize(
    "mode, expected_values",
    [
        ("wrap", [5.0, 5.0, 0.0, 0.0, 2.0]),
        ("clip", [0.0, 0.0, 0.0, 5.0, 5.0]),
    ],
)
def test_numpy_take_index_modes_stay_on_device(
        torch_device_ctx, monkeypatch, mode, expected_values):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = TimeSeries(
            np.arange(6, dtype=dtype), delta_t=0.25, epoch=12.0
        )
        index_tensor = torch.tensor(
            [-7, -1, 0, 6, 8], device=device, dtype=torch.int64
        )
        indices = Array(TorchArrayData(index_tensor), copy=False)
        output = Array(np.empty(5, dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.take copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            direct = values.take(indices, mode=mode)
            actual = np.take(values, indices, mode=mode)
            returned = np.take(
                values, indices, axis=-1, out=output, mode=mode
            )

        expected = torch.tensor(
            expected_values, device=device, dtype=torch_dtype
        )
        for selected in (direct, actual, output):
            assert type(selected) is Array
            assert selected._data.tensor.device.type == device
            torch.testing.assert_close(selected._data.tensor, expected)
        assert returned is output


def test_numpy_multidimensional_take_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    indices_source = np.array([[2, 0], [1, 1]], dtype=np.int64)

    with ctx:
        values = Array(source)
        indices = Array(indices_source)
        output = Array(np.empty((2, 2, 2, 4), dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.take copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.take(values, indices, axis=-2)
            returned = np.take(values, indices, axis=1, out=output)
            wrapped = np.take(values, [-5, 5], axis=-1, mode="wrap")
            clipped = np.take(values, [-5, 5], axis=0, mode="clip")

        expected = {
            "actual": np.take(source, indices_source, axis=-2),
            "wrapped": np.take(source, [-5, 5], axis=-1, mode="wrap"),
            "clipped": np.take(source, [-5, 5], axis=0, mode="clip"),
        }
        for name, selected in (
                ("actual", actual), ("output", output),
                ("wrapped", wrapped), ("clipped", clipped)):
            wanted = expected["actual" if name == "output" else name]
            assert type(selected) is Array
            assert selected._data.tensor.device.type == device
            torch.testing.assert_close(
                selected._data.tensor,
                torch.as_tensor(wanted, device=device),
            )
        assert returned is output


@pytest.mark.parametrize("complex_values", (False, True))
def test_numpy_take_along_axis_stays_on_torch_device(
        torch_device_ctx, monkeypatch, complex_values):
    ctx, device = torch_device_ctx
    if complex_values:
        dtype = np.complex64 if device == "mps" else np.complex128
    else:
        dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    if complex_values:
        source = source + 1j * (source + 0.25)
    index_values = np.array(
        [[[-1, 0], [2, 1], [0, -2]]], dtype=np.int32
    )

    with ctx:
        values = Array(source)
        indices = Array(index_values)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError(
                "numpy.take_along_axis copied Torch data to host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.take_along_axis(values, indices)

        expected = np.take_along_axis(source, index_values, axis=-1)
        assert type(actual) is Array
        assert actual._data.tensor.device.type == device
        torch.testing.assert_close(
            actual._data.tensor,
            torch.as_tensor(expected, device=device),
        )


def test_numpy_take_along_axis_autograd_and_flatten(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        leaf = torch.arange(
            12, device=device, dtype=torch_dtype, requires_grad=True
        )
        values = Array(
            TorchArrayData(leaf.reshape(3, 4)), copy=False
        )
        indices = Array(np.array([[0, 1], [2, 3], [1, 1]]))
        actual = np.take_along_axis(values, indices, axis=1)
        actual._data.tensor.sum().backward()

        flattened = np.take_along_axis(
            values, np.array([0, -1, 5, 7]), axis=None
        )

    expected_gradient = torch.tensor(
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0],
        device=device,
        dtype=torch_dtype,
    )
    torch.testing.assert_close(leaf.grad, expected_gradient)
    torch.testing.assert_close(
        flattened._data.tensor,
        torch.tensor([0, 11, 5, 7], device=device, dtype=torch_dtype),
    )
    assert flattened.dtype == dtype


def test_numpy_take_along_axis_contract_and_uint32():
    source = np.arange(12, dtype=np.int32).reshape(3, 4)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        unsigned_source = np.array(
            [[0, 2**32 - 1], [5, 7]], dtype=np.uint32
        )
        unsigned = np.take_along_axis(
            Array(unsigned_source),
            np.array([[1, 0], [0, 1]]),
            axis=1,
        )
        np.testing.assert_array_equal(
            unsigned.numpy(),
            np.take_along_axis(
                unsigned_source,
                np.array([[1, 0], [0, 1]]),
                axis=1,
            ),
        )
        assert unsigned.dtype == np.uint32

        with pytest.raises(IndexError, match="integer array"):
            np.take_along_axis(values, np.array([[0.0]]), axis=1)
        with pytest.raises(ValueError, match="same number of dimensions"):
            np.take_along_axis(values, np.array([0]), axis=1)
        with pytest.raises(ValueError, match="single dimension"):
            np.take_along_axis(values, np.array([[0]]), axis=None)
        with pytest.raises(np.exceptions.AxisError):
            np.take_along_axis(values, np.zeros((3, 1), dtype=int), axis=2)
        with pytest.raises(IndexError):
            np.take_along_axis(values, np.array([[4], [0], [0]]), axis=1)
        with pytest.raises(IndexError):
            np.take_along_axis(
                values, np.zeros((2, 2), dtype=np.int64), axis=1
            )


@pytest.mark.parametrize("complex_values", (False, True))
def test_numpy_copy_stays_on_torch_device(
        torch_device_ctx, monkeypatch, complex_values):
    ctx, device = torch_device_ctx
    if complex_values:
        dtype = torch.complex64 if device == "mps" else torch.complex128
    else:
        dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        real_dtype = torch.float32 if device == "mps" else torch.float64
        base = torch.arange(6, device=device, dtype=real_dtype)
        if complex_values:
            base = (base + 1j * (base + 0.25)).to(dtype=dtype)
        source_tensor = base.reshape(2, 3).transpose(0, 1)
        values = Array(TorchArrayData(source_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("numpy.copy copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            copied = {
                order: np.copy(values, order=order)
                for order in (None, "C", "F", "A", "K")
            }

        for actual in copied.values():
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            assert actual._data.tensor.data_ptr() != source_tensor.data_ptr()
            torch.testing.assert_close(actual._data.tensor, source_tensor)
        assert copied["C"]._data.tensor.is_contiguous()
        for order in (None, "F", "A", "K"):
            assert copied[order]._data.tensor.stride() == (1, 3)


def test_numpy_copy_autograd_and_contract():
    with scheme.TorchScheme("cpu"):
        leaf = torch.arange(
            6, dtype=torch.float64, requires_grad=True
        )
        source_tensor = leaf.reshape(2, 3).transpose(0, 1)
        values = Array(TorchArrayData(source_tensor), copy=False)

        copied = np.copy(values, "C", True)
        copied._data.tensor.sum().backward()
        torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))
        assert copied._data.tensor.stride() == (2, 1)

        unsigned_source = np.array(
            [[0, 2**32 - 1], [5, 7]], dtype=np.uint32
        ).T
        unsigned = np.copy(Array(unsigned_source), order=b"F")
        np.testing.assert_array_equal(unsigned.numpy(), unsigned_source)
        assert unsigned.dtype == np.uint32
        assert unsigned._data.tensor.stride() == (1, 2)

        with pytest.raises(ValueError, match="order must be one of"):
            np.copy(values, order="Z")
        with pytest.raises(TypeError, match="order must be str"):
            np.copy(values, order=1)


def test_numpy_searchsorted_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = Array(np.array(
            [-np.inf, -1.0, 0.0, np.inf, np.nan], dtype=dtype
        ))
        query_tensor = torch.tensor(
            [-np.inf, 0.0, np.inf, np.nan],
            device=device,
            dtype=torch_dtype,
        )
        queries = Array(TorchArrayData(query_tensor), copy=False)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.searchsorted copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            left = np.searchsorted(values, queries)
            right = values.searchsorted(queries, side="right")

        expected_left = torch.tensor(
            [0, 2, 3, 4], device=device, dtype=torch.int64
        )
        expected_right = torch.tensor(
            [1, 3, 4, 5], device=device, dtype=torch.int64
        )
        for actual, expected in (
                (left, expected_left), (right, expected_right)):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(actual._data.tensor, expected)


def test_numpy_searchsorted_sorter_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    with ctx:
        values = Array(np.array([30.0, 10.0, 20.0], dtype=dtype))
        sorter = Array(TorchArrayData(torch.tensor(
            [1, 2, 0], device=device, dtype=torch.int64
        )), copy=False)
        queries = np.array([15.0, 25.0], dtype=dtype)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.searchsorted copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.searchsorted(values, queries, sorter=sorter)

        assert type(actual) is Array
        assert actual._data.tensor.device.type == device
        torch.testing.assert_close(
            actual._data.tensor,
            torch.tensor([1, 2], device=device, dtype=torch.int64),
        )


def test_numpy_searchsorted_scalar_keeps_numpy_result(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    with ctx:
        values = Array(np.array([1.0, 3.0, 3.0, 7.0], dtype=dtype))
        assert np.searchsorted(values, 3.0) == np.intp(1)
        assert values.searchsorted(3.0, side="right") == np.intp(3)
        assert isinstance(np.searchsorted(values, 3.0), np.intp)


def test_numpy_digitize_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    with ctx:
        values = Array(np.array(
            [-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=dtype
        ))
        increasing = Array(np.array([-1.0, 0.0, 0.0, 1.0], dtype=dtype))
        decreasing = Array(np.array([1.0, 0.0, 0.0, -1.0], dtype=dtype))

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("numpy.digitize copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            increasing_left = np.digitize(values, increasing)
            increasing_right = np.digitize(
                values, increasing, right=True
            )
            decreasing_left = np.digitize(values, decreasing)
            host_bins = np.digitize(
                values, np.array([-1.0, 0.0, 1.0], dtype=dtype)
            )

        for actual in (
                increasing_left,
                increasing_right,
                decreasing_left,
                host_bins):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            assert actual.dtype == np.dtype(np.int64)

        np.testing.assert_array_equal(
            increasing_left.numpy(),
            np.digitize(values.numpy(), increasing.numpy()),
        )
        np.testing.assert_array_equal(
            increasing_right.numpy(),
            np.digitize(values.numpy(), increasing.numpy(), right=True),
        )
        np.testing.assert_array_equal(
            decreasing_left.numpy(),
            np.digitize(values.numpy(), decreasing.numpy()),
        )


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32,
        np.float16, np.float32, np.float64,
    ),
)
@pytest.mark.parametrize("right", (False, True))
def test_numpy_digitize_matches_numpy_contract(dtype, right):
    if np.dtype(dtype).kind == "b":
        source = np.array([False, True, False], dtype=dtype)
        increasing = np.array([False, True, True], dtype=dtype)
    elif np.dtype(dtype).kind == "u":
        source = np.array([0, 1, 2, 4], dtype=dtype)
        increasing = np.array([0, 1, 1, 3], dtype=dtype)
    elif np.dtype(dtype).kind == "i":
        source = np.array([-2, -1, 0, 2], dtype=dtype)
        increasing = np.array([-1, 0, 0, 1], dtype=dtype)
    else:
        source = np.array(
            [-np.inf, -1.0, -0.0, 2.0, np.inf, np.nan], dtype=dtype
        )
        increasing = np.array([-1.0, 0.0, 0.0, 1.0], dtype=dtype)
    decreasing = increasing[::-1].copy()

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        for bins in (increasing, decreasing, np.array([], dtype=dtype)):
            actual = np.digitize(values, Array(bins), right=right)
            assert type(actual) is Array
            assert actual.dtype == np.dtype(np.int64)
            np.testing.assert_array_equal(
                actual.numpy(), np.digitize(source, bins, right=right)
            )

        scalar = np.digitize(source[0], Array(increasing), right)
        assert isinstance(scalar, np.intp)
        assert scalar == np.digitize(source[0], increasing, right=right)


def test_numpy_digitize_fallback_and_errors_match_numpy():
    with scheme.TorchScheme("cpu"):
        unsigned = Array(np.array([0, 2**63, 2**64 - 1], dtype=np.uint64))
        unsigned_bins = Array(np.array([0, 2**63], dtype=np.uint64))
        unsigned_result = np.digitize(unsigned, unsigned_bins)

        nan_bins = Array(np.array([0.0, np.nan, 2.0]))
        nan_result = np.digitize(Array(np.array([1.0])), nan_bins)

        with pytest.raises(
                ValueError,
                match="bins must be monotonically increasing or decreasing"):
            np.digitize(Array(np.array([1.0])), Array(np.array([0, 2, 1])))
        with pytest.raises(ValueError):
            np.digitize(
                Array(np.array([1.0])), Array(np.array([[0, 1]]))
            )
        with pytest.raises(TypeError, match="x may not be complex"):
            np.digitize(
                Array(np.array([1 + 0j])), Array(np.array([0.0, 1.0]))
            )

    assert type(unsigned_result) is Array
    assert isinstance(unsigned_result._scheme, scheme.TorchScheme)
    np.testing.assert_array_equal(
        unsigned_result.numpy(),
        np.digitize(unsigned.numpy(), unsigned_bins.numpy()),
    )
    assert isinstance(nan_result, np.ndarray)
    np.testing.assert_array_equal(
        nan_result, np.digitize(np.array([1.0]), np.array([0.0, np.nan, 2.0]))
    )


def test_numpy_interp_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("NumPy interpolation requires double precision")

    x_source = np.array(
        [[-np.inf, -1.0, -0.25], [0.5, 3.0, np.nan]],
        dtype=np.float32,
    )
    xp_source = np.array([-1, 0, 2, 4], dtype=np.int32)
    fp_source = np.array([2, -1, 3, 7], dtype=np.int16)
    complex_source = np.array(
        [2 + 1j, -1 + 4j, 3 - 2j, 7 + 0.5j],
        dtype=np.complex64,
    )
    expected_real = np.interp(
        x_source, xp_source, fp_source, left=-8, right=11
    )
    expected_periodic = np.interp(
        x_source, xp_source, complex_source, period=7
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("numpy.interp copied Torch data to the host")

    with ctx:
        x = Array(x_source)
        xp = Array(xp_source)
        fp = Array(fp_source)
        complex_fp = Array(complex_source)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual_real = np.interp(
                x, xp, fp, left=-8, right=11
            )
            actual_periodic = np.interp(
                x, xp, complex_fp, period=7
            )
            scalar = np.interp(0.5, xp, fp)

        for actual, expected, dtype in (
                (actual_real, expected_real, np.float64),
                (actual_periodic, expected_periodic, np.complex128)):
            assert type(actual) is Array
            assert actual.dtype == np.dtype(dtype)
            assert actual._data.tensor.device.type == device
            np.testing.assert_allclose(
                actual.numpy(), expected, rtol=2e-14, atol=2e-14,
                equal_nan=True,
            )
        assert isinstance(scalar, np.float64)
        assert scalar == np.interp(0.5, xp_source, fp_source)


def test_numpy_interp_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("NumPy interpolation requires double precision")

    with ctx:
        x = torch.tensor(
            [-0.5, 0.25, 1.5, 3.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        xp = torch.tensor(
            [0.0, 1.0, 2.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        fp = torch.tensor(
            [1.0, 3.0, 2.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        result = np.interp(
            Array(TorchArrayData(x), copy=False),
            Array(TorchArrayData(xp), copy=False),
            Array(TorchArrayData(fp), copy=False),
            left=-2.0,
            right=5.0,
        )
        result._data.tensor.sum().backward()

    torch.testing.assert_close(
        x.grad,
        torch.tensor(
            [0.0, 2.0, -1.0, 0.0],
            dtype=torch.float64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        xp.grad,
        torch.tensor(
            [-1.5, 0.0, 0.5], dtype=torch.float64, device=device
        ),
    )
    torch.testing.assert_close(
        fp.grad,
        torch.tensor(
            [0.75, 0.75, 0.5], dtype=torch.float64, device=device
        ),
    )


@pytest.mark.parametrize(
    "x, xp, fp, options",
    (
        (
            np.array([[-2.0, 0.0], [0.5, 4.0]], dtype=np.float16),
            np.array([-1, 0, 2], dtype=np.int8),
            np.array([False, True, False]),
            {"left": 3, "right": -4},
        ),
        (
            np.array([], dtype=np.float32),
            np.array([0.0, 1.0]),
            np.array([1 + 2j, 3 - 4j], dtype=np.complex64),
            {},
        ),
        (
            np.array([np.nan, -np.inf, 0.0, np.inf]),
            np.array([0.0]),
            np.array([2.5]),
            {"left": -1.0, "right": 4.0},
        ),
        (
            np.array([-4.0, -0.5, 0.0, 3.0, 8.0]),
            np.array([-2.0, 0.0, 1.5, 4.0]),
            np.array([1.0, -2.0, 3.0, 0.5]),
            {"period": -5.0},
        ),
        (
            np.array([-0.5, 0.0, 0.5, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 4.0]),
            {},
        ),
    ),
)
def test_numpy_interp_matches_numpy_contract(x, xp, fp, options):
    expected = np.interp(x, xp, fp, **options)
    with scheme.TorchScheme("cpu"):
        actual = np.interp(Array(x), Array(xp), Array(fp), **options)

    assert type(actual) is Array
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=2e-14, atol=2e-14,
        equal_nan=True,
    )


def test_numpy_interp_scalar_errors_and_fallback_contract(monkeypatch):
    source_x = np.array([-1.0, 0.5, 2.0])
    with scheme.TorchScheme("cpu"):
        xp = Array(np.array([0.0, 1.0]))
        fp = Array(np.array([2.0, 4.0]))
        scalar = np.interp(0.25, xp, fp)

        with pytest.raises(ValueError, match="sample points is empty"):
            np.interp(Array(source_x), Array(np.array([])), Array(np.array([])))
        with pytest.raises(ValueError, match="not of the same length"):
            np.interp(Array(source_x), xp, Array(np.array([1.0])))
        with pytest.raises(ValueError, match="1-D sequences"):
            np.interp(Array(source_x), Array(np.array([[0.0, 1.0]])), fp)
        with pytest.raises(ValueError, match="non-zero"):
            np.interp(Array(source_x), xp, fp, period=0)

        fallback_cases = (
            (
                Array(source_x),
                Array(np.array([1.0, 0.0])),
                Array(np.array([4.0, 2.0])),
                {},
            ),
            (
                Array(source_x),
                Array(np.array([-np.inf, 1.0])),
                Array(np.array([2.0, 4.0])),
                {},
            ),
            (
                Array(source_x),
                Array(np.array([0.0, 1.0])),
                Array(np.array([np.inf, 4.0])),
                {},
            ),
            (
                Array(source_x),
                Array(np.array([0.0, 360.0])),
                Array(np.array([1.0, 2.0])),
                {"period": 360.0},
            ),
        )
        for x, xp_case, fp_case, options in fallback_cases:
            actual = np.interp(x, xp_case, fp_case, **options)
            expected = np.interp(
                x.numpy(), xp_case.numpy(), fp_case.numpy(), **options
            )
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_interp",
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = np.interp(Array(source_x), xp, fp)

    assert isinstance(scalar, np.float64)
    assert scalar == np.interp(0.25, [0.0, 1.0], [2.0, 4.0])
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_allclose(
        backend_fallback,
        np.interp(source_x, [0.0, 1.0], [2.0, 4.0]),
    )


def test_numpy_trapezoid_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    trapezoid = array_module._NUMPY_TRAPEZOID
    source = np.array([[1, 2, 4], [3, 5, 9]], dtype=np.float32)
    coordinates = np.array([0, 1, 3], dtype=np.float32)
    expected = trapezoid(source, x=coordinates, axis=1)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("trapezoidal integration copied data to the host")

    with ctx:
        values = Array(source)
        x = Array(coordinates)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = trapezoid(values, x=x, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                legacy = np.trapz(values, x=x, axis=1)

        for result in (actual, legacy):
            assert type(result) is Array
            assert result.dtype == expected.dtype
            assert result._data.tensor.device.type == device
            np.testing.assert_allclose(result.numpy(), expected)


@pytest.mark.parametrize(
    "source, x, dx, axis",
    (
        (
            np.array([[1, 2, 4], [3, 5, 9]], dtype=np.int32),
            None,
            np.int32(2),
            1,
        ),
        (
            np.array(
                [[True, False, True], [False, True, True]], dtype=np.bool_
            ),
            np.array([False, True, False], dtype=np.bool_),
            1.0,
            1,
        ),
        (
            np.array(
                [[1 + 2j, 2 - 1j, 4 + 0.5j],
                 [3 + 0j, 5 + 2j, 9 - 2j]],
                dtype=np.complex64,
            ),
            np.array([0, 1, 3], dtype=np.int32),
            1.0,
            1,
        ),
        (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            np.array([[0.0], [1.0], [3.0]], dtype=np.float32),
            1.0,
            -2,
        ),
        (
            np.empty((2, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            1.0,
            1,
        ),
    ),
)
def test_numpy_trapezoid_matches_numpy_contract(source, x, dx, axis):
    trapezoid = array_module._NUMPY_TRAPEZOID
    expected = trapezoid(source, x=x, dx=dx, axis=axis)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        coordinates = None if x is None else Array(x)
        actual = trapezoid(values, x=coordinates, dx=dx, axis=axis)

    assert type(actual) is Array
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.numpy(), expected)


def test_numpy_trapezoid_scalar_and_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    trapezoid = array_module._NUMPY_TRAPEZOID

    with ctx:
        scalar = trapezoid(Array(np.array([1, 2, 4], dtype=np.float32)))
        y_tensor = torch.tensor(
            [[1, 2, 4], [3, 5, 9]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        x_tensor = torch.tensor(
            [0, 1, 3],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        result = trapezoid(
            Array(TorchArrayData(y_tensor), copy=False),
            x=Array(TorchArrayData(x_tensor), copy=False),
            axis=1,
        )
        result._data.tensor.sum().backward()

    assert isinstance(scalar, np.float32)
    assert scalar == trapezoid(np.array([1, 2, 4], dtype=np.float32))
    torch.testing.assert_close(
        y_tensor.grad,
        torch.tensor(
            [[0.5, 1.5, 1.0], [0.5, 1.5, 1.0]],
            dtype=torch.float32,
            device=device,
        ),
    )
    torch.testing.assert_close(
        x_tensor.grad,
        torch.tensor(
            [-5.5, -4.5, 10.0], dtype=torch.float32, device=device
        ),
    )


def test_numpy_trapezoid_errors_and_fallback_contract(monkeypatch):
    trapezoid = array_module._NUMPY_TRAPEZOID
    source = np.array([[1, 2, 4], [3, 5, 9]], dtype=np.float32)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        with pytest.raises(ValueError):
            trapezoid(
                values, x=Array(np.array([0.0, 1.0, 2.0, 3.0])), axis=1
            )

        unsigned = trapezoid(
            Array(source.astype(np.uint32)), axis=1
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_trapezoid",
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = trapezoid(values, axis=1)

    assert isinstance(unsigned, np.ndarray)
    np.testing.assert_allclose(
        unsigned, trapezoid(source.astype(np.uint32), axis=1)
    )
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_allclose(backend_fallback, trapezoid(source, axis=1))


def test_numpy_argsort_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    with ctx:
        values = Array(np.array([3.0, np.nan, -1.0, 2.0], dtype=dtype))
        series = TimeSeries(
            np.array([3.0, np.nan, -1.0, 2.0], dtype=dtype),
            delta_t=0.25,
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.argsort copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            order = np.argsort(values)
            series_order = series.argsort()
            sorted_values = values[order]

        assert type(order) is Array
        assert type(series_order) is Array
        assert isinstance(sorted_values, TorchArrayData)
        assert order._data.tensor.device.type == device
        assert series_order._data.tensor.device.type == device
        assert sorted_values.tensor.device.type == device
        torch.testing.assert_close(
            order._data.tensor,
            torch.tensor([2, 3, 0, 1], device=device, dtype=torch.int64),
        )
        torch.testing.assert_close(
            series_order._data.tensor,
            torch.tensor([2, 3, 0, 1], device=device, dtype=torch.int64),
        )
        torch.testing.assert_close(
            sorted_values.tensor,
            torch.tensor(
                [-1.0, 2.0, 3.0, np.nan],
                device=device,
                dtype=sorted_values.tensor.dtype,
            ),
            equal_nan=True,
        )


def test_numpy_argsort_stable_and_axes_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64

    with ctx:
        values = Array(np.array(
            [[2.0, 1.0, 1.0], [0.0, 3.0, 2.0]], dtype=dtype
        ))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.argsort copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            stable = values.argsort(axis=-1, kind="stable")
            flattened = np.argsort(values, axis=None, stable=True)

        assert stable._data.tensor.device.type == device
        assert flattened._data.tensor.device.type == device
        torch.testing.assert_close(
            stable._data.tensor,
            torch.tensor(
                [[1, 2, 0], [0, 2, 1]],
                device=device,
                dtype=torch.int64,
            ),
        )
        torch.testing.assert_close(
            flattened._data.tensor,
            torch.tensor([3, 1, 2, 0, 5, 4], device=device),
        )


def test_numpy_argsort_unsupported_options_fall_back_to_numpy():
    with scheme.TorchScheme("cpu"):
        complex_values = Array(np.array([2 + 1j, 1 + 3j, 1 + 2j]))
        real_values = Array(np.array([3.0, 1.0, 2.0]))

        complex_order = np.argsort(complex_values)
        quick_order = real_values.argsort(kind="quicksort")

        assert isinstance(complex_order, np.ndarray)
        assert isinstance(quick_order, np.ndarray)
        np.testing.assert_array_equal(complex_order, [2, 1, 0])
        np.testing.assert_array_equal(quick_order, [1, 2, 0])

        with pytest.raises(ValueError, match="kind.*stable"):
            np.argsort(real_values, kind="stable", stable=True)


def test_numpy_sort_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.array(
        [[2.0, np.nan, 1.0], [0.0, -0.0, -1.0]], dtype=dtype
    )

    with ctx:
        values = TimeSeries(source, delta_t=0.25, epoch=10.0)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.sort copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            axis_sorted = np.sort(values, axis=-1, stable=True)
            flattened = np.sort(values, axis=None, kind="stable")

        assert type(axis_sorted) is Array
        assert type(flattened) is Array
        assert axis_sorted._data.tensor.device.type == device
        assert flattened._data.tensor.device.type == device
        torch.testing.assert_close(
            axis_sorted._data.tensor,
            torch.tensor(
                [[1.0, 2.0, np.nan], [-1.0, 0.0, -0.0]],
                device=device,
                dtype=axis_sorted._data.tensor.dtype,
            ),
            equal_nan=True,
        )
        torch.testing.assert_close(
            flattened._data.tensor,
            torch.tensor(
                [-1.0, 0.0, -0.0, 1.0, 2.0, np.nan],
                device=device,
                dtype=flattened._data.tensor.dtype,
            ),
            equal_nan=True,
        )
        assert bool(torch.signbit(axis_sorted._data.tensor[1, 1])) is False
        assert bool(torch.signbit(axis_sorted._data.tensor[1, 2])) is True


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ),
)
def test_numpy_sort_matches_numpy_on_cpu(dtype):
    if np.dtype(dtype).kind == "b":
        source = np.array([True, False, True, False], dtype=dtype)
    elif np.dtype(dtype).kind == "u":
        source = np.array([5, 0, 2, 2, 1], dtype=dtype)
    elif np.dtype(dtype).kind == "i":
        source = np.array([5, -3, 2, 2, -1], dtype=dtype)
    else:
        source = np.array([np.nan, 0.0, -0.0, 2.0, -1.0], dtype=dtype)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        expected_source = values.numpy()
        actual = np.sort(values, stable=True)

        assert type(actual) is Array
        assert actual._data.tensor.device.type == "cpu"
        assert actual.dtype == expected_source.dtype
        np.testing.assert_equal(
            actual.numpy(), np.sort(expected_source, stable=True)
        )


def test_numpy_sort_unsupported_options_fall_back_to_numpy():
    with scheme.TorchScheme("cpu"):
        complex_values = Array(np.array([2 + 1j, 1 + 3j, 1 + 2j]))
        real_values = Array(np.array([3.0, 1.0, 2.0]))

        complex_sorted = np.sort(complex_values)
        quick_sorted = np.sort(real_values, kind="quicksort")

        assert isinstance(complex_sorted, np.ndarray)
        assert isinstance(quick_sorted, np.ndarray)
        np.testing.assert_array_equal(
            complex_sorted, np.array([1 + 2j, 1 + 3j, 2 + 1j])
        )
        np.testing.assert_array_equal(quick_sorted, [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="kind.*stable"):
            np.sort(real_values, kind="stable", stable=True)


def test_numpy_dot_products_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    left_source = np.arange(6, dtype=dtype).reshape(2, 3) - 2
    right_source = np.arange(12, dtype=dtype).reshape(3, 4) - 4
    inner_source = np.arange(15, dtype=dtype).reshape(5, 3) - 6

    with ctx:
        left = Array(left_source)
        right = Array(right_source)
        inner_right = Array(inner_source)
        left._data.tensor.requires_grad_()
        right._data.tensor.requires_grad_()

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("NumPy dot product copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            dotted = np.dot(left, right)
            inner = np.inner(left, inner_right)
            outer = np.outer(left, right)
            scalar = np.vdot(left, Array(left_source + 1))
            mixed = np.dot(a=left_source, b=right)

        for result in (dotted, inner, outer, mixed):
            assert type(result) is Array
            assert result._data.tensor.device.type == device
        assert isinstance(scalar, np.floating)
        assert scalar.dtype == np.dtype(dtype)
        assert scalar == np.vdot(left_source, left_source + 1)

        torch.testing.assert_close(
            dotted._data.tensor,
            torch.tensor(
                np.dot(left_source, right_source),
                dtype=dotted._data.tensor.dtype,
                device=device,
            ),
        )
        torch.testing.assert_close(
            inner._data.tensor,
            torch.tensor(
                np.inner(left_source, inner_source),
                dtype=inner._data.tensor.dtype,
                device=device,
            ),
        )
        torch.testing.assert_close(
            outer._data.tensor,
            torch.tensor(
                np.outer(left_source, right_source),
                dtype=outer._data.tensor.dtype,
                device=device,
            ),
        )
        torch.testing.assert_close(
            mixed._data.tensor, dotted._data.tensor
        )

        dotted._data.tensor.square().sum().backward()
        assert left._data.tensor.grad is not None
        assert right._data.tensor.grad is not None
        assert left._data.tensor.grad.device.type == device
        assert right._data.tensor.grad.device.type == device


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_, np.int32, np.int64, np.uint32,
        np.float32, np.float64, np.complex64, np.complex128,
    ),
)
def test_numpy_dot_products_match_numpy_on_cpu(dtype):
    kind = np.dtype(dtype).kind
    if kind == "b":
        left_source = (np.arange(24).reshape(2, 3, 4) % 3) != 0
        right_source = (np.arange(40).reshape(5, 4, 2) % 2) != 0
    elif kind == "c":
        left_values = np.arange(24).reshape(2, 3, 4)
        right_values = np.arange(40).reshape(5, 4, 2)
        left_source = left_values - 5 + 1j * (left_values % 4 - 2)
        right_source = right_values - 7 + 1j * (right_values % 5 - 2)
    else:
        left_source = np.arange(24).reshape(2, 3, 4) % 7 - 3
        right_source = np.arange(40).reshape(5, 4, 2) % 5 - 2
    left_source = np.asarray(left_source, dtype=dtype)
    right_source = np.asarray(right_source, dtype=dtype)
    inner_source = np.moveaxis(right_source, 1, -1)

    with scheme.TorchScheme("cpu"):
        left = Array(left_source)
        right = Array(right_source)
        inner_right = Array(inner_source)
        actual = (
            np.dot(left, right),
            np.inner(left, inner_right),
            np.outer(left, right),
            np.vdot(left, Array(left_source.reshape(4, 6))),
        )
        expected = (
            np.dot(left_source, right_source),
            np.inner(left_source, inner_source),
            np.outer(left_source, right_source),
            np.vdot(left_source, left_source.reshape(4, 6)),
        )

        for result, reference in zip(actual[:3], expected[:3]):
            assert type(result) is Array
            assert result.dtype == reference.dtype
            assert result.shape == reference.shape
            np.testing.assert_allclose(
                result.numpy(), reference, rtol=2e-6, atol=2e-6
            )
        assert isinstance(actual[3], np.generic)
        assert actual[3].dtype == expected[3].dtype
        np.testing.assert_allclose(
            actual[3], expected[3], rtol=2e-6, atol=2e-6
        )


def test_numpy_dot_products_preserve_complex_and_scalar_semantics():
    left_source = np.array(
        [[1 + 2j, -3 + 4j], [2 - 1j, 5 + 3j]],
        dtype=np.complex128,
    )
    right_source = np.array(
        [[2 - 3j, 1 + 1j], [-4 + 2j, 3 - 2j]],
        dtype=np.complex128,
    )

    with scheme.TorchScheme("cpu"):
        left = Array(left_source)
        right = Array(right_source)
        inner = np.inner(left, right)
        dot_scalar = np.dot(
            Array(left_source.ravel()), Array(right_source.ravel())
        )
        inner_scalar = np.inner(
            Array(left_source.ravel()), Array(right_source.ravel())
        )
        vdot = np.vdot(left, right)
        scaled = np.dot(left, 2.0)

        np.testing.assert_allclose(inner.numpy(), np.inner(left_source, right_source))
        np.testing.assert_allclose(
            dot_scalar,
            np.dot(left_source.ravel(), right_source.ravel()),
        )
        np.testing.assert_allclose(
            inner_scalar,
            np.inner(left_source.ravel(), right_source.ravel()),
        )
        np.testing.assert_allclose(vdot, np.vdot(left_source, right_source))
        np.testing.assert_allclose(scaled.numpy(), np.dot(left_source, 2.0))
        assert inner.dtype == np.inner(left_source, right_source).dtype
        assert isinstance(dot_scalar, np.complexfloating)
        assert isinstance(inner_scalar, np.complexfloating)
        assert vdot.dtype == np.vdot(left_source, right_source).dtype
        assert scaled.dtype == np.dot(left_source, 2.0).dtype
        assert vdot != np.sum(left_source.ravel() * right_source.ravel())

        with pytest.raises(ValueError):
            np.vdot(left, Array(np.arange(3, dtype=np.complex128)))


def test_numpy_linalg_norm_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = (np.arange(24).reshape(2, 3, 4) - 11).astype(dtype)

    with ctx:
        tensor = torch.tensor(
            source, dtype=torch.float32 if device == "mps" else torch.float64,
            device=device, requires_grad=True,
        )
        reference = tensor.detach().clone().requires_grad_(True)
        values = Array(TorchArrayData(tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("NumPy norm copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            vector = np.linalg.norm(values, ord=3, axis=1)
            frobenius = np.linalg.norm(
                values, ord="fro", axis=(0, 2), keepdims=True
            )
            scalar = np.linalg.norm(values)

        for result in (vector, frobenius):
            assert type(result) is Array
            assert result._data.tensor.device.type == device
            assert result.dtype == np.dtype(dtype)
        assert type(scalar) is np.dtype(dtype).type

        expected_vector = torch.linalg.vector_norm(
            reference, ord=3, dim=1
        )
        expected_frobenius = torch.linalg.vector_norm(
            reference, ord=2, dim=(0, 2), keepdim=True
        )
        torch.testing.assert_close(vector._data.tensor, expected_vector)
        torch.testing.assert_close(
            frobenius._data.tensor, expected_frobenius
        )
        np.testing.assert_allclose(scalar, np.linalg.norm(source))

        (vector._data.tensor.square().sum()
         + frobenius._data.tensor.square().sum()).backward()
        (expected_vector.square().sum()
         + expected_frobenius.square().sum()).backward()
        torch.testing.assert_close(tensor.grad, reference.grad)


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_, np.int32, np.int64, np.uint32,
        np.float32, np.float64, np.complex64, np.complex128,
    ),
)
def test_numpy_linalg_norm_matches_numpy_on_cpu(dtype):
    values = np.arange(24).reshape(2, 3, 4) % 7 - 3
    if np.dtype(dtype).kind == "b":
        source = (values % 3) != 0
    elif np.dtype(dtype).kind == "c":
        source = values + 1j * (values % 4 - 2)
    else:
        source = values
    source = np.asarray(source, dtype=dtype)

    with scheme.TorchScheme("cpu"):
        array = Array(source)
        actual = (
            np.linalg.norm(array),
            np.linalg.norm(array, axis=1),
            np.linalg.norm(
                array, ord="fro", axis=(0, 2), keepdims=True
            ),
            np.linalg.norm(array, ord=0.5, axis=-1),
            np.linalg.norm(array, ord=np.inf, axis=-1),
        )
        expected = (
            np.linalg.norm(source),
            np.linalg.norm(source, axis=1),
            np.linalg.norm(
                source, ord="fro", axis=(0, 2), keepdims=True
            ),
            np.linalg.norm(source, ord=0.5, axis=-1),
            np.linalg.norm(source, ord=np.inf, axis=-1),
        )

        assert isinstance(actual[0], np.generic)
        assert actual[0].dtype == expected[0].dtype
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6)
        for result, reference in zip(actual[1:], expected[1:]):
            assert type(result) is Array
            assert result.dtype == reference.dtype
            assert result.shape == reference.shape
            np.testing.assert_allclose(
                result.numpy(), reference, rtol=2e-6, atol=2e-6
            )


def test_numpy_linalg_norm_empty_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        empty_source = np.empty((2, 0), dtype=np.float32)
        empty = Array(empty_source)
        positive_infinity = np.linalg.norm(
            empty, ord=np.inf, axis=1, keepdims=True
        )

        matrix_source = np.arange(6, dtype=np.float64).reshape(2, 3)
        matrix = Array(matrix_source)
        spectral = np.linalg.norm(matrix, ord=2)
        nuclear = np.linalg.norm(matrix, ord="nuc")
        negative_source = matrix_source[0] + 1
        negative = np.linalg.norm(Array(negative_source), ord=-2)

        assert type(positive_infinity) is Array
        assert positive_infinity.shape == (2, 1)
        assert positive_infinity.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(
            positive_infinity.numpy(),
            np.linalg.norm(
                empty_source, ord=np.inf, axis=1, keepdims=True
            ),
        )

    for result, reference in (
            (spectral, np.linalg.norm(matrix_source, ord=2)),
            (nuclear, np.linalg.norm(matrix_source, ord="nuc")),
            (negative, np.linalg.norm(negative_source, ord=-2))):
        assert isinstance(result, np.generic)
        np.testing.assert_allclose(result, reference)


def test_numpy_nonzero_stays_on_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx

    with ctx:
        values = Array(np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int32))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.nonzero copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            function_indices = np.nonzero(values)
            method_indices = values.nonzero()

        expected = (
            torch.tensor([0, 1, 1], device=device, dtype=torch.int64),
            torch.tensor([1, 0, 2], device=device, dtype=torch.int64),
        )
        for indices in (function_indices, method_indices):
            assert len(indices) == 2
            for actual, wanted in zip(indices, expected):
                assert type(actual) is Array
                assert actual._data.tensor.device.type == device
                torch.testing.assert_close(actual._data.tensor, wanted)


def test_numpy_nonzero_preserves_tuple_and_plain_array_contract(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.complex64

    with ctx:
        values = TimeSeries(
            np.array([0, 2, 0, -3], dtype=dtype),
            delta_t=0.25,
            epoch=12.0,
        )
        empty = Array(np.zeros(0, dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.nonzero copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            indices = values.nonzero()
            empty_indices = np.nonzero(empty)
            selected = values[indices]

        assert len(indices) == len(empty_indices) == 1
        assert type(indices[0]) is Array
        assert type(empty_indices[0]) is Array
        assert indices[0]._data.tensor.device.type == device
        assert empty_indices[0]._data.tensor.device.type == device
        torch.testing.assert_close(
            indices[0]._data.tensor,
            torch.tensor([1, 3], device=device, dtype=torch.int64),
        )
        assert isinstance(selected, TorchArrayData)
        assert selected.tensor.device.type == device
        torch.testing.assert_close(
            selected.tensor,
            torch.tensor([2, -3], device=device, dtype=selected.tensor.dtype),
        )
        assert empty_indices[0]._data.tensor.numel() == 0


def test_numpy_nonzero_unsupported_backend_falls_back_to_numpy(monkeypatch):
    with scheme.TorchScheme("cpu"):
        values = Array(np.array([0.0, 2.0, 0.0, 4.0]))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "numpy_nonzero",
                lambda self: NotImplemented,
            )
            indices = np.nonzero(values)

        assert len(indices) == 1
        assert isinstance(indices[0], np.ndarray)
        np.testing.assert_array_equal(indices[0], [1, 3])


def test_numpy_index_locations_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx

    with ctx:
        values = Array(
            np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int32)
        )
        scalar = Array(
            TorchArrayData(torch.tensor(2, device=device)), copy=False
        )
        scalar_zero = Array(
            TorchArrayData(torch.tensor(0, device=device)), copy=False
        )
        empty = Array(np.zeros((2, 0, 3), dtype=np.float32))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("index discovery copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            flattened = np.flatnonzero(values)
            grouped = np.argwhere(a=values)
            scalar_grouped = np.argwhere(scalar)
            scalar_zero_grouped = np.argwhere(scalar_zero)
            empty_grouped = np.argwhere(empty)

        expected = {
            "flattened": torch.tensor(
                [1, 3, 5], device=device, dtype=torch.int64
            ),
            "grouped": torch.tensor(
                [[0, 1], [1, 0], [1, 2]],
                device=device,
                dtype=torch.int64,
            ),
        }
        for name, actual in (
                ("flattened", flattened), ("grouped", grouped)):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(actual._data.tensor, expected[name])

        assert type(scalar_grouped) is Array
        assert scalar_grouped.shape == (1, 0)
        assert scalar_grouped._data.tensor.device.type == device
        assert type(scalar_zero_grouped) is Array
        assert scalar_zero_grouped.shape == (0, 0)
        assert scalar_zero_grouped._data.tensor.device.type == device
        assert type(empty_grouped) is Array
        assert empty_grouped.shape == (0, 3)
        assert empty_grouped._data.tensor.device.type == device


def test_numpy_argwhere_backend_fallback(monkeypatch):
    with scheme.TorchScheme("cpu"):
        values = Array(np.array([[0, 2], [3, 0]], dtype=np.int32))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_index_locations",
                lambda self, function, args, kwargs: NotImplemented,
            )
            actual = np.argwhere(values)

        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(
            actual,
            np.argwhere(np.array([[0, 2], [3, 0]], dtype=np.int32)),
        )


@pytest.mark.parametrize("function", [np.argwhere, np.flatnonzero])
def test_numpy_index_locations_unsupported_uint32_falls_back(function):
    source = np.array([[0, 2], [3, 0]], dtype=np.uint32)
    with scheme.TorchScheme("cpu"):
        actual = function(Array(source))

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, function(source))


def test_numpy_shape_transforms_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    source_values = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    squeezed_values = np.arange(12, dtype=dtype).reshape(1, 2, 1, 6)

    with ctx:
        values = Array(source_values)
        squeezable = Array(squeezed_values)
        both_contiguous = Array(
            TorchArrayData(values._data.tensor[:1, :1, :]), copy=False
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("shape transform copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            reshaped = np.reshape(values, (4, 6), copy=False)
            transposed = np.transpose(values, (2, 0, 1))
            fortran_reshaped = np.reshape(
                transposed, (3, 8), order="F"
            )
            swapped = np.swapaxes(values, 0, -1)
            moved = np.moveaxis(values, (0, 2), (-1, 0))
            rolled = np.rollaxis(values, -1, 1)
            squeezed = np.squeeze(squeezable, axis=(0, 2))
            reversed_axes = values.T
            as_reshaped = np.reshape(
                reversed_axes, (4, 6), order="A", copy=False
            )
            both_contiguous_reshaped = np.reshape(
                both_contiguous, (2, 2), order="A", copy=False
            )
            memory_view = Array(
                TorchArrayData(transposed._data.tensor[:, :, ::2]),
                copy=False,
            )
            raveled = values.ravel()
            flattened = transposed.flatten()
            fortran_raveled = np.ravel(transposed, order=b"F")
            memory_raveled = memory_view.ravel(order="K")
            as_raveled = reversed_axes.ravel(order="A")
            fortran_flattened = transposed.flatten(order="f")
            default_raveled = np.ravel(values, order=None)

        expected = {
            "reshaped": source_values.reshape(4, 6),
            "transposed": source_values.transpose(2, 0, 1),
            "fortran_reshaped": source_values.transpose(
                2, 0, 1
            ).reshape(3, 8, order="F"),
            "swapped": source_values.swapaxes(0, -1),
            "moved": np.moveaxis(source_values, (0, 2), (-1, 0)),
            "rolled": np.rollaxis(source_values, -1, 1),
            "squeezed": squeezed_values.squeeze(axis=(0, 2)),
            "reversed_axes": source_values.T,
            "as_reshaped": source_values.T.reshape(
                4, 6, order="A"
            ),
            "both_contiguous_reshaped": source_values[
                :1, :1, :
            ].reshape(2, 2, order="A"),
            "raveled": source_values.ravel(),
            "flattened": source_values.transpose(2, 0, 1).flatten(),
            "fortran_raveled": source_values.transpose(
                2, 0, 1
            ).ravel(order="F"),
            "memory_raveled": source_values.transpose(
                2, 0, 1
            )[:, :, ::2].ravel(order="K"),
            "as_raveled": source_values.T.ravel(order="A"),
            "fortran_flattened": source_values.transpose(
                2, 0, 1
            ).flatten(order="F"),
            "default_raveled": source_values.ravel(order=None),
        }
        outputs = {
            "reshaped": reshaped,
            "transposed": transposed,
            "fortran_reshaped": fortran_reshaped,
            "swapped": swapped,
            "moved": moved,
            "rolled": rolled,
            "squeezed": squeezed,
            "reversed_axes": reversed_axes,
            "as_reshaped": as_reshaped,
            "both_contiguous_reshaped": both_contiguous_reshaped,
            "raveled": raveled,
            "flattened": flattened,
            "fortran_raveled": fortran_raveled,
            "memory_raveled": memory_raveled,
            "as_raveled": as_raveled,
            "fortran_flattened": fortran_flattened,
            "default_raveled": default_raveled,
        }
        for name, actual in outputs.items():
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor,
                torch.tensor(
                    expected[name], device=device, dtype=torch_dtype
                ),
            )


def test_numpy_axis_movement_views_and_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            24, device=device, dtype=torch.float32
        ).reshape(2, 3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        unchanged = np.moveaxis(values, (), ())
        moved = np.moveaxis(values, [0, -1], [-1, 0])
        rolled = np.rollaxis(a=values, axis=2, start=1)

        assert (
            unchanged._data.tensor.data_ptr()
            == values._data.tensor.data_ptr()
        )
        assert moved._data.tensor.data_ptr() == values._data.tensor.data_ptr()
        assert rolled._data.tensor.data_ptr() == values._data.tensor.data_ptr()

        weights = torch.arange(
            1, 25, device=device, dtype=torch.float32
        ).reshape(4, 3, 2)
        (moved._data.tensor * weights).sum().backward()
        torch.testing.assert_close(
            source.grad,
            torch.movedim(weights, (0, 2), (2, 0)),
        )


def test_numpy_axis_movement_errors_and_fallback_contract():
    source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        moved = np.moveaxis(
            a=values,
            source=np.array([0, -1]),
            destination=[-1, 0],
        )
        rolled = np.rollaxis(values, np.int64(-1), np.int64(1))

        assert type(moved) is Array
        assert type(rolled) is Array
        np.testing.assert_array_equal(
            moved.numpy(), np.moveaxis(source, [0, -1], [-1, 0])
        )
        np.testing.assert_array_equal(
            rolled.numpy(), np.rollaxis(source, -1, 1)
        )

        with pytest.raises(ValueError, match="repeated axis"):
            np.moveaxis(values, (0, 0), (1, 2))
        with pytest.raises(ValueError, match="same number"):
            np.moveaxis(values, (0, 1), (2,))
        with pytest.raises(np.exceptions.AxisError):
            np.moveaxis(values, 3, 0)
        with pytest.raises(np.exceptions.AxisError):
            np.rollaxis(values, 3)
        with pytest.raises(np.exceptions.AxisError):
            np.rollaxis(values, 1, -4)
        with pytest.raises(TypeError):
            np.moveaxis(values, 1.5, 0)
        with pytest.raises(TypeError):
            np.rollaxis(values, 1, 1.5)

    moved_fallback = np.moveaxis(Array(source), 0, -1)
    rolled_fallback = np.rollaxis(Array(source), 2, 1)
    assert isinstance(moved_fallback, np.ndarray)
    assert isinstance(rolled_fallback, np.ndarray)
    np.testing.assert_array_equal(
        moved_fallback, np.moveaxis(source, 0, -1)
    )
    np.testing.assert_array_equal(
        rolled_fallback, np.rollaxis(source, 2, 1)
    )


def test_numpy_join_functions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        first = Array(np.array([[1, 2], [3, 4]], dtype=dtype))
        second = Array(np.array([[5, 6], [7, 8]], dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy join copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            concatenated = np.concatenate((first, second), axis=0)
            flattened = np.concatenate((first, second), axis=None)
            stacked = np.stack((first, second), axis=-1)

        expected = (
            (
                concatenated,
                torch.tensor(
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    device=device,
                    dtype=torch_dtype,
                ),
            ),
            (
                flattened,
                torch.tensor(
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    device=device,
                    dtype=torch_dtype,
                ),
            ),
            (
                stacked,
                torch.tensor(
                    [
                        [[1, 5], [2, 6]],
                        [[3, 7], [4, 8]],
                    ],
                    device=device,
                    dtype=torch_dtype,
                ),
            ),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(actual._data.tensor, wanted)


def test_numpy_join_functions_preserve_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        first_source = torch.tensor(
            [1.0, 2.0], device=device, dtype=dtype, requires_grad=True
        )
        second_source = torch.tensor(
            [3.0, 4.0], device=device, dtype=dtype, requires_grad=True
        )
        first = Array(TorchArrayData(first_source), copy=False)
        second = Array(TorchArrayData(second_source), copy=False)

        concatenated = np.concatenate((first, second))
        stacked = np.stack((first, second))
        (concatenated._data.tensor.sum()
         + stacked._data.tensor.sum()).backward()

        expected = torch.full((2,), 2.0, device=device, dtype=dtype)
        torch.testing.assert_close(first_source.grad, expected)
        torch.testing.assert_close(second_source.grad, expected)


def test_numpy_directional_joins_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        first = Array(np.array([1, 2], dtype=dtype))
        second = Array(np.array([3, 4], dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy directional join copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            horizontal = np.hstack((first, second))
            vertical = np.vstack((first, second))
            depth = np.dstack((first, second))
            columns = np.column_stack((first, second))

        expected = (
            (horizontal, [1, 2, 3, 4]),
            (vertical, [[1, 2], [3, 4]]),
            (depth, [[[1, 3], [2, 4]]]),
            (columns, [[1, 3], [2, 4]]),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor,
                torch.tensor(wanted, device=device, dtype=torch_dtype),
            )


def test_numpy_directional_joins_preserve_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        first_source = torch.tensor(
            [1.0, 2.0], device=device, dtype=dtype, requires_grad=True
        )
        second_source = torch.tensor(
            [3.0, 4.0], device=device, dtype=dtype, requires_grad=True
        )
        first = Array(TorchArrayData(first_source), copy=False)
        second = Array(TorchArrayData(second_source), copy=False)

        results = (
            np.hstack((first, second)),
            np.vstack((first, second)),
            np.dstack((first, second)),
            np.column_stack((first, second)),
        )
        sum(result._data.tensor.sum() for result in results).backward()

        expected = torch.full((2,), 4.0, device=device, dtype=dtype)
        torch.testing.assert_close(first_source.grad, expected)
        torch.testing.assert_close(second_source.grad, expected)


def test_numpy_directional_join_dtype_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        first = Array(np.array([1.0, 2.0], dtype=np.float32))
        second = Array(np.array([3.0, 4.0], dtype=np.float64))

        promoted = np.hstack((first, second))
        typed = np.vstack((first, first), dtype=np.float64)
        mixed = np.column_stack((first, np.array([5.0, 6.0])))

        assert type(promoted) is Array
        assert promoted.dtype == np.dtype(np.float64)
        assert type(typed) is Array
        assert typed.dtype == np.dtype(np.float64)
        assert isinstance(mixed, np.ndarray)
        np.testing.assert_array_equal(
            mixed,
            np.array([[1.0, 5.0], [2.0, 6.0]]),
        )

        with pytest.raises(TypeError, match="Cannot cast"):
            np.hstack(
                (first, second), dtype=np.float32, casting="safe"
            )
        with pytest.raises(ValueError):
            np.vstack((first, Array(np.ones(3))))


def test_numpy_diff_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    source = np.array([[1, 4, 9, 16], [2, 8, 18, 32]], dtype=dtype)
    flags_source = np.array([True, True, False, True])

    with ctx:
        values = Array(source)
        flags = Array(flags_source)
        prepend = Array(np.array([[0], [1]], dtype=dtype))
        append = dtype(40)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy diff copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            first = np.diff(values)
            second = np.diff(values, n=2, axis=1)
            vertical = np.diff(values, axis=0)
            bounded = np.diff(
                values,
                axis=1,
                prepend=prepend,
                append=append,
            )
            changed = np.diff(flags, n=2)

        expected = (
            (first, np.diff(source)),
            (second, np.diff(source, n=2, axis=1)),
            (vertical, np.diff(source, axis=0)),
            (
                bounded,
                np.diff(
                    source,
                    axis=1,
                    prepend=np.array([[0], [1]], dtype=dtype),
                    append=append,
                ),
            ),
            (changed, np.diff(flags_source, n=2)),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor,
                torch.tensor(
                    wanted,
                    device=device,
                    dtype=actual._data.tensor.dtype,
                ),
            )
        assert first._data.tensor.dtype == torch_dtype


def test_numpy_diff_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [1.0, 4.0, 9.0, 16.0],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        reference = source.detach().clone().requires_grad_(True)
        values = Array(TorchArrayData(source), copy=False)

        actual = np.diff(values, n=2)
        expected = torch.diff(reference, n=2)
        actual._data.tensor.square().sum().backward()
        expected.square().sum().backward()

        torch.testing.assert_close(actual._data.tensor, expected)
        torch.testing.assert_close(source.grad, reference.grad)


def test_numpy_diff_dtype_shape_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        unsigned_source = np.array([0, 1, 0, 3], dtype=np.uint32)
        boolean_source = np.array([True, False, False, True])
        complex_source = np.array([1 + 2j, 4 - 1j, -2 + 3j])
        unsigned = Array(unsigned_source)
        boolean = Array(boolean_source)
        complex_values = Array(complex_source)

        unsigned_result = np.diff(unsigned, n=2)
        boolean_result = np.diff(boolean)
        complex_result = np.diff(complex_values, prepend=1.5)

        assert np.diff(unsigned, n=0) is unsigned
        assert unsigned_result.dtype == np.dtype(np.uint32)
        assert boolean_result.dtype == np.dtype(np.bool_)
        assert complex_result.dtype == np.dtype(np.complex128)
        np.testing.assert_array_equal(
            unsigned_result.numpy(), np.diff(unsigned_source, n=2)
        )
        np.testing.assert_array_equal(
            boolean_result.numpy(), np.diff(boolean_source)
        )
        np.testing.assert_allclose(
            complex_result.numpy(), np.diff(complex_source, prepend=1.5)
        )

        mixed = np.diff(
            complex_values, prepend=np.array([0 + 0j])
        )
        assert isinstance(mixed, np.ndarray)
        np.testing.assert_allclose(
            mixed,
            np.diff(complex_source, prepend=np.array([0 + 0j])),
        )

        with pytest.raises(ValueError, match="non-negative"):
            np.diff(unsigned, n=-1)
        with pytest.raises(np.exceptions.AxisError):
            np.diff(unsigned, axis=2)
        with pytest.raises(ValueError):
            np.diff(
                Array(np.ones((2, 3))),
                axis=1,
                prepend=Array(np.ones((3, 1))),
            )


def test_numpy_roll_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    empty_source = np.empty((2, 0, 4), dtype=dtype)

    with ctx:
        values = Array(source)
        empty = Array(empty_source)
        original = values._data.tensor.clone()

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy roll copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            flattened = np.roll(values, (1, 2))
            repeated = np.roll(values, (1, 2), axis=(0, 0))
            multidimensional = np.roll(values, 1, axis=(0, 2))
            empty_result = np.roll(empty, -3, axis=1)

        expected = (
            (flattened, np.roll(source, (1, 2))),
            (repeated, np.roll(source, (1, 2), axis=(0, 0))),
            (multidimensional, np.roll(source, 1, axis=(0, 2))),
            (empty_result, np.roll(empty_source, -3, axis=1)),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)
        torch.testing.assert_close(values._data.tensor, original)


def test_numpy_roll_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            12,
            device=device,
            dtype=torch.float32,
        ).reshape(3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        rolled = np.roll(values, (1, -2), axis=(0, 1))
        weights = torch.arange(
            1,
            13,
            device=device,
            dtype=torch.float32,
        ).reshape(3, 4)

        (rolled._data.tensor * weights).sum().backward()

        torch.testing.assert_close(
            source.grad,
            torch.roll(weights, shifts=(-1, 2), dims=(0, 1)),
        )


def test_numpy_roll_dtype_shape_and_fallback_contract():
    source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        result = np.roll(values, np.array([1, -2]), axis=(0, 2))
        coerced = np.roll(values, (1.9, -2.1), axis=(0, 2))
        copied = np.roll(values, 0, axis=())

        assert type(result) is Array
        assert result.dtype == np.dtype(np.uint32)
        assert result.shape == source.shape
        assert type(copied) is Array
        assert copied._data.tensor.data_ptr() != values._data.tensor.data_ptr()
        np.testing.assert_array_equal(
            result.numpy(),
            np.roll(source, np.array([1, -2]), axis=(0, 2)),
        )
        np.testing.assert_array_equal(
            coerced.numpy(),
            np.roll(source, (1.9, -2.1), axis=(0, 2)),
        )
        np.testing.assert_array_equal(copied.numpy(), source)

        with pytest.raises(np.exceptions.AxisError):
            np.roll(values, 1, axis=3)
        with pytest.raises(ValueError, match="broadcast"):
            np.roll(values, (1, 2, 3), axis=(0, 2))

    fallback = np.roll(Array(source), 1, axis=0)
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(fallback, np.roll(source, 1, axis=0))


def test_numpy_flip_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        values = Array(source)
        original = values._data.tensor.clone()

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy flip copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            all_axes = np.flip(values)
            selected = np.flip(values, axis=(0, -1))
            vertical = np.flipud(values)
            horizontal = np.fliplr(values)
            keyword = np.flip(m=values, axis=np.array([0, 2]))

        expected = (
            (all_axes, np.flip(source)),
            (selected, np.flip(source, axis=(0, -1))),
            (vertical, np.flipud(source)),
            (horizontal, np.fliplr(source)),
            (keyword, np.flip(source, axis=np.array([0, 2]))),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)
        torch.testing.assert_close(values._data.tensor, original)


def test_numpy_flip_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            12,
            device=device,
            dtype=torch.float32,
        ).reshape(3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        flipped = np.flip(values, axis=(0, 1))
        weights = torch.arange(
            1,
            13,
            device=device,
            dtype=torch.float32,
        ).reshape(3, 4)

        (flipped._data.tensor * weights).sum().backward()

        torch.testing.assert_close(
            source.grad,
            torch.flip(weights, dims=(0, 1)),
        )


def test_numpy_flip_dtype_shape_and_fallback_contract():
    source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        result = np.flip(values, axis=[0, 2])
        unchanged = np.flip(values, axis=())

        assert type(result) is Array
        assert result.dtype == np.dtype(np.uint32)
        assert result.shape == source.shape
        assert type(unchanged) is Array
        assert (
            unchanged._data.tensor.data_ptr()
            == values._data.tensor.data_ptr()
        )
        np.testing.assert_array_equal(
            result.numpy(), np.flip(source, axis=[0, 2])
        )
        np.testing.assert_array_equal(unchanged.numpy(), source)

        with pytest.raises(ValueError, match="repeated axis"):
            np.flip(values, axis=(0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.flip(values, axis=3)
        with pytest.raises(ValueError, match=">= 2-d"):
            np.fliplr(Array(np.arange(4, dtype=np.uint32)))

    fallback = np.flip(Array(source), axis=0)
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(fallback, np.flip(source, axis=0))


def test_numpy_rot90_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        values = Array(source)
        original = values._data.tensor.clone()

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy rot90 copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            default = np.rot90(values)
            reversed_axes = np.rot90(values, k=-1, axes=(2, 0))
            keyword = np.rot90(m=values, k=np.int64(2), axes=[-1, 0])
            unchanged = np.rot90(values, k=4, axes=(0.0, 1.0))

        expected = (
            (default, np.rot90(source)),
            (reversed_axes, np.rot90(source, k=-1, axes=(2, 0))),
            (keyword, np.rot90(source, k=np.int64(2), axes=[-1, 0])),
            (unchanged, source),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)
        assert (
            unchanged._data.tensor.data_ptr()
            == values._data.tensor.data_ptr()
        )
        torch.testing.assert_close(values._data.tensor, original)


def test_numpy_rot90_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            24,
            device=device,
            dtype=torch.float32,
        ).reshape(2, 3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        rotated = np.rot90(values, k=1, axes=(0, 2))
        weights = torch.arange(
            1,
            25,
            device=device,
            dtype=torch.float32,
        ).reshape(4, 3, 2)

        (rotated._data.tensor * weights).sum().backward()

        torch.testing.assert_close(
            source.grad,
            torch.rot90(weights, k=-1, dims=(0, 2)),
        )


def test_numpy_rot90_dtype_errors_and_fallback_contract():
    source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        rotated = np.rot90(values, k=np.array(3), axes=(0, -1))
        float_k = np.rot90(values, k=1.5, axes=(0, 2))

        assert type(rotated) is Array
        assert rotated.dtype == np.dtype(np.uint32)
        assert rotated.shape == (4, 3, 2)
        np.testing.assert_array_equal(
            rotated, np.rot90(source, k=np.array(3), axes=(0, -1))
        )
        np.testing.assert_array_equal(
            float_k, np.rot90(source, k=1.5, axes=(0, 2))
        )

        with pytest.raises(ValueError, match=r"len\(axes\) must be 2"):
            np.rot90(values, axes=(0,))
        with pytest.raises(ValueError, match="Axes must be different"):
            np.rot90(values, axes=(0, 3))
        with pytest.raises(ValueError, match="out of range"):
            np.rot90(values, axes=(0, 4))
        with pytest.raises(IndexError):
            np.rot90(values, axes=(0.0, 1.0))

    fallback = np.rot90(Array(source), k=-1, axes=(0, 2))
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(
        fallback, np.rot90(source, k=-1, axes=(0, 2))
    )


def test_numpy_matrix_helpers_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    vector_source = np.arange(4, dtype=dtype)
    matrix_source = np.arange(12, dtype=dtype).reshape(3, 4)
    batch_source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        vector = Array(vector_source)
        matrix = Array(matrix_source)
        batch = Array(batch_source)
        noncontiguous = Array(
            TorchArrayData(matrix._data.tensor.transpose(0, 1)),
            copy=False,
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy matrix helper copied Torch data")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            constructed = np.diag(vector, k=-1)
            extracted = np.diag(v=matrix, k=np.array(1))
            flattened = np.diagflat(noncontiguous, k=2)
            lower = np.tril(batch, k=np.array([1.5]))
            upper = np.triu(m=vector, k=-0.5)

        expected = (
            (constructed, np.diag(vector_source, k=-1)),
            (extracted, np.diag(matrix_source, k=np.array(1))),
            (flattened, np.diagflat(matrix_source.T, k=2)),
            (lower, np.tril(batch_source, k=np.array([1.5]))),
            (upper, np.triu(vector_source, k=-0.5)),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)


def test_numpy_matrix_helpers_preserve_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            6,
            device=device,
            dtype=torch.float32,
        ).reshape(2, 3).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        diagonal = np.diag(values, k=1)
        flattened = np.diagflat(values, k=-1)
        upper = np.triu(values, k=1)
        diagonal_weights = torch.tensor(
            [2.0, 3.0], device=device, dtype=torch.float32
        )
        flat_weights = torch.arange(
            1,
            flattened._data.tensor.numel() + 1,
            device=device,
            dtype=torch.float32,
        ).reshape(flattened.shape)
        upper_weights = torch.arange(
            1, 7, device=device, dtype=torch.float32
        ).reshape(2, 3)

        loss = (
            (diagonal._data.tensor * diagonal_weights).sum()
            + (flattened._data.tensor * flat_weights).sum()
            + (upper._data.tensor * upper_weights).sum()
        )
        loss.backward()

        expected = torch.zeros_like(source)
        expected[0, 1] += diagonal_weights[0]
        expected[1, 2] += diagonal_weights[1]
        expected += torch.diagonal(flat_weights, offset=-1).reshape(2, 3)
        expected += torch.triu(upper_weights, diagonal=1)
        torch.testing.assert_close(source.grad, expected)


def test_numpy_matrix_helpers_dtype_errors_and_fallback_contract():
    vector_source = np.arange(4, dtype=np.uint32)
    matrix_source = np.arange(12, dtype=np.uint32).reshape(3, 4)
    batch_source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        vector = Array(vector_source)
        matrix = Array(matrix_source)
        batch = Array(batch_source)

        results = (
            (np.diag(vector, k=2), np.diag(vector_source, k=2)),
            (np.diagflat(matrix, k=-1), np.diagflat(matrix_source, k=-1)),
            (np.tril(batch, k=0.5), np.tril(batch_source, k=0.5)),
            (np.triu(vector, k=1.5), np.triu(vector_source, k=1.5)),
        )
        for actual, wanted in results:
            assert type(actual) is Array
            assert actual.dtype == np.dtype(np.uint32)
            np.testing.assert_array_equal(actual.numpy(), wanted)

        with pytest.raises(ValueError, match="Input must be 1- or 2-d"):
            np.diag(batch)
        with pytest.raises(TypeError):
            np.diag(matrix, k=1.5)
        scalar = Array(TorchArrayData(torch.tensor(3)), copy=False)
        with pytest.raises(TypeError, match=r"tri\(\) missing"):
            np.tril(scalar)

    fallback = np.triu(Array(matrix_source), k=-1)
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(fallback, np.triu(matrix_source, k=-1))


def test_numpy_pad_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        values = Array(source)
        noncontiguous = Array(
            TorchArrayData(values._data.tensor.transpose(0, 1)),
            copy=False,
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy pad copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            constant = np.pad(
                values,
                ((1, 2), (0, 1), (2, 0)),
                mode="constant",
                constant_values=((1, 2), (3, 4), (5, 6)),
            )
            edge = np.pad(
                array=noncontiguous,
                pad_width=((0, 1), (2, 1), (1, 0)),
                mode="edge",
            )

        expected = (
            (
                constant,
                np.pad(
                    source,
                    ((1, 2), (0, 1), (2, 0)),
                    mode="constant",
                    constant_values=((1, 2), (3, 4), (5, 6)),
                ),
            ),
            (
                edge,
                np.pad(
                    source.transpose(1, 0, 2),
                    ((0, 1), (2, 1), (1, 0)),
                    mode="edge",
                ),
            ),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)


def test_numpy_pad_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            6, device=device, dtype=torch.float32
        ).reshape(2, 3).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        constant = np.pad(
            values,
            ((1, 0), (0, 2)),
            mode="constant",
            constant_values=3.5,
        )
        edge = np.pad(values, ((1, 2), (2, 1)), mode="edge")
        constant_weights = torch.arange(
            1,
            constant._data.tensor.numel() + 1,
            device=device,
            dtype=torch.float32,
        ).reshape(constant.shape)
        edge_weights = torch.arange(
            1,
            edge._data.tensor.numel() + 1,
            device=device,
            dtype=torch.float32,
        ).reshape(edge.shape)

        loss = (
            (constant._data.tensor * constant_weights).sum()
            + (edge._data.tensor * edge_weights).sum()
        )
        loss.backward()

        reference = source.detach().clone().requires_grad_()
        expected_constant = torch.nn.functional.pad(
            reference, (0, 2, 1, 0), mode="constant", value=3.5
        )
        expected_edge = torch.nn.functional.pad(
            reference[None, None], (2, 1, 1, 2), mode="replicate"
        )[0, 0]
        expected_loss = (
            (expected_constant * constant_weights).sum()
            + (expected_edge * edge_weights).sum()
        )
        expected_loss.backward()
        torch.testing.assert_close(source.grad, reference.grad)


def test_numpy_pad_dtype_errors_and_fallback_contract():
    source = np.arange(6, dtype=np.uint32).reshape(2, 3)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        padded = np.pad(
            values,
            ((1, 2), (2, 0)),
            constant_values=((1.5, 2.5), (3.5, 4.5)),
        )
        unchanged = np.pad(values, 0, mode="edge")

        assert type(padded) is Array
        assert padded.dtype == np.dtype(np.uint32)
        np.testing.assert_array_equal(
            padded,
            np.pad(
                source,
                ((1, 2), (2, 0)),
                constant_values=((1.5, 2.5), (3.5, 4.5)),
            ),
        )
        assert type(unchanged) is Array
        assert (
            unchanged._data.tensor.data_ptr()
            != values._data.tensor.data_ptr()
        )
        np.testing.assert_array_equal(unchanged, source)

        with pytest.raises(TypeError, match="integral type"):
            np.pad(values, 1.5)
        with pytest.raises(ValueError, match="negative"):
            np.pad(values, -1)
        with pytest.raises(ValueError, match="can't extend empty axis"):
            np.pad(Array(np.empty((0, 2))), ((1, 0), (0, 0)), mode="edge")
        with pytest.raises(ValueError, match="unsupported keyword"):
            np.pad(values, 1, mode="edge", constant_values=2)

        reflected = np.pad(values, 1, mode="reflect")
        assert isinstance(reflected, np.ndarray)
        np.testing.assert_array_equal(
            reflected, np.pad(source, 1, mode="reflect")
        )

    fallback = np.pad(Array(source), ((1, 0), (0, 2)))
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(
        fallback, np.pad(source, ((1, 0), (0, 2)))
    )


def test_numpy_boolean_selection_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    axis_condition = np.array([1, 0, 2, 0], dtype=np.int32)
    extract_condition = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8
    )

    with ctx:
        values = Array(source)
        noncontiguous = Array(
            TorchArrayData(values._data.tensor.transpose(0, 1)),
            copy=False,
        )
        device_condition = Array(axis_condition)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError(
                "NumPy boolean selection copied Torch data to host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            compressed = np.compress(
                device_condition, noncontiguous, axis=-1
            )
            flattened = np.compress(
                [True, False, True, False, True], values
            )
            extracted = np.extract(extract_condition, values)

        expected = (
            (
                compressed,
                np.compress(
                    axis_condition, source.transpose(1, 0, 2), axis=-1
                ),
            ),
            (
                flattened,
                np.compress(
                    [True, False, True, False, True], source
                ),
            ),
            (extracted, np.extract(extract_condition, source)),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)


def test_numpy_boolean_selection_preserves_torch_autograd(
        torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            12, device=device, dtype=torch.float32
        ).reshape(3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        compressed = np.compress([1, 0, 1], values, axis=0)
        extracted = np.extract(
            [[1, 0, 0, 1], [0, 1, 0, 0]], values
        )
        loss = compressed._data.tensor.sum() + extracted._data.tensor.sum()
        loss.backward()

        reference = source.detach().clone().requires_grad_()
        expected_loss = (
            torch.index_select(
                reference, 0, torch.tensor([0, 2], device=device)
            ).sum()
            + torch.index_select(
                reference.reshape(-1),
                0,
                torch.tensor([0, 3, 5], device=device),
            ).sum()
        )
        expected_loss.backward()
        torch.testing.assert_close(source.grad, reference.grad)


def test_numpy_boolean_selection_errors_and_fallback_contract():
    source = np.arange(12, dtype=np.uint32).reshape(3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        compressed = np.compress([1, 0, 2], values, axis=0)
        extracted = np.extract(
            [[1, 0, 0], [0, 2, 0]], values
        )

        for actual, wanted in (
                (compressed, np.compress([1, 0, 2], source, axis=0)),
                (
                    extracted,
                    np.extract([[1, 0, 0], [0, 2, 0]], source),
                )):
            assert type(actual) is Array
            assert actual.dtype == np.dtype(np.uint32)
            np.testing.assert_array_equal(actual.numpy(), wanted)

        with pytest.raises(ValueError, match="condition must be a 1-d"):
            np.compress([[1, 0], [0, 1]], values)
        with pytest.raises(np.exceptions.AxisError):
            np.compress([1, 0], values, axis=2)
        with pytest.raises(IndexError):
            np.compress([0, 0, 0, 1], values, axis=0)
        with pytest.raises(IndexError):
            np.extract([0] * 12 + [1], values)

        output = np.empty((2, 4), dtype=np.uint32)
        returned = np.compress(
            [1, 0, 1], values, axis=0, out=output
        )
        assert returned is output
        np.testing.assert_array_equal(
            output, np.compress([1, 0, 1], source, axis=0)
        )

        string_condition = np.compress(["x", "", "y"], values, axis=0)
        assert isinstance(string_condition, np.ndarray)
        np.testing.assert_array_equal(
            string_condition,
            np.compress(["x", "", "y"], source, axis=0),
        )

        empty_source = np.empty((0, 2), dtype=np.float64)
        empty_selected = np.compress(
            [1, 0, 1], Array(empty_source), axis=1
        )
        assert type(empty_selected) is Array
        np.testing.assert_array_equal(
            empty_selected,
            np.compress([1, 0, 1], empty_source, axis=1),
        )

    fallback = np.extract([1, 0, 1], Array(source))
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(fallback, np.extract([1, 0, 1], source))


def test_numpy_expand_dims_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        values = Array(source)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy expand_dims copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            leading = np.expand_dims(values, axis=0)
            selected = np.expand_dims(values, axis=(0, 3))
            reordered = np.expand_dims(a=values, axis=(3, 0))
            unchanged = np.expand_dims(values, axis=())

        expected = (
            (leading, np.expand_dims(source, axis=0)),
            (selected, np.expand_dims(source, axis=(0, 3))),
            (reordered, np.expand_dims(source, axis=(3, 0))),
            (unchanged, source),
        )
        for actual, wanted in expected:
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            np.testing.assert_array_equal(actual.numpy(), wanted)
        assert (
            unchanged._data.tensor.data_ptr()
            == values._data.tensor.data_ptr()
        )


def test_numpy_expand_dims_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            12,
            device=device,
            dtype=torch.float32,
        ).reshape(3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        expanded = np.expand_dims(values, axis=(0, 2))
        weights = torch.arange(
            1,
            13,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 3, 1, 4)

        (expanded._data.tensor * weights).sum().backward()

        torch.testing.assert_close(source.grad, weights.reshape(3, 4))


def test_numpy_expand_dims_shape_errors_and_fallback_contract():
    source = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        result = np.expand_dims(values, axis=[0, -1])

        assert type(result) is Array
        assert result.dtype == np.dtype(np.uint32)
        assert result.shape == (1, 2, 3, 4, 1)
        np.testing.assert_array_equal(
            result.numpy(), np.expand_dims(source, axis=[0, -1])
        )

        with pytest.raises(ValueError, match="repeated axis"):
            np.expand_dims(values, axis=(0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.expand_dims(values, axis=4)
        with pytest.raises(TypeError):
            np.expand_dims(values, axis=np.array([0, 2]))
        with pytest.raises(TypeError):
            np.expand_dims(values, axis=1.5)

    fallback = np.expand_dims(Array(source), axis=0)
    assert isinstance(fallback, np.ndarray)
    np.testing.assert_array_equal(fallback, np.expand_dims(source, axis=0))


@pytest.mark.parametrize(
    "function, expected_shapes",
    [
        (np.atleast_1d, [(1,), (3,), (2, 3), (2, 3, 4)]),
        (np.atleast_2d, [(1, 1), (1, 3), (2, 3), (2, 3, 4)]),
        (np.atleast_3d, [(1, 1, 1), (1, 3, 1), (2, 3, 1), (2, 3, 4)]),
    ],
)
def test_numpy_atleast_nd_stays_on_torch_device(
        function, expected_shapes, torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        tensors = [
            torch.tensor(3.0, dtype=torch_dtype, device=device),
            torch.arange(3, dtype=torch_dtype, device=device),
            torch.arange(6, dtype=torch_dtype, device=device).reshape(2, 3),
            torch.arange(24, dtype=torch_dtype, device=device).reshape(2, 3, 4),
        ]
        values = [
            Array(TorchArrayData(tensor), copy=False) for tensor in tensors
        ]

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("NumPy atleast_* copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            outputs = [function(value) for value in values]
            multiple = function(values[0], values[1], values[3])

        for index, (actual, expected_shape) in enumerate(
                zip(outputs, expected_shapes)):
            assert type(actual) is Array
            assert actual.shape == expected_shape
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor.reshape(-1), tensors[index].reshape(-1)
            )
            if tensors[index].ndim >= len(expected_shape):
                assert actual is values[index]

        assert isinstance(multiple, tuple)
        assert len(multiple) == 3
        assert multiple[2] is values[3]
        for actual, expected_shape in zip(
                multiple,
                (expected_shapes[0], expected_shapes[1], expected_shapes[3])):
            assert type(actual) is Array
            assert actual.shape == expected_shape
            assert actual._data.tensor.device.type == device


def test_numpy_atleast_nd_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            6, dtype=torch.float32, device=device
        ).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        actual = np.atleast_3d(values)
        weights = torch.arange(
            1, 7, dtype=torch.float32, device=device
        ).reshape(1, 6, 1)

        (actual._data.tensor * weights).sum().backward()

        assert actual.shape == (1, 6, 1)
        torch.testing.assert_close(source.grad, weights.reshape(6))


def test_numpy_atleast_nd_mixed_and_backend_fallback(monkeypatch):
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    other = np.arange(3, dtype=np.float64)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        mixed = np.atleast_3d(values, other)

        assert isinstance(mixed, tuple)
        assert all(isinstance(value, np.ndarray) for value in mixed)
        np.testing.assert_array_equal(mixed[0], np.atleast_3d(source))
        np.testing.assert_array_equal(mixed[1], np.atleast_3d(other))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_atleast_nd",
                lambda self, function, args, kwargs: NotImplemented,
            )
            fallback = np.atleast_2d(values)

        assert isinstance(fallback, np.ndarray)
        np.testing.assert_array_equal(fallback, np.atleast_2d(source))


@pytest.mark.parametrize("function", [np.broadcast_to, np.broadcast_arrays])
def test_numpy_broadcast_views_stay_on_torch_device(
        function, torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        left_tensor = torch.arange(
            3, dtype=torch_dtype, device=device
        )
        right_tensor = torch.arange(
            2, dtype=torch_dtype, device=device
        ).reshape(2, 1)
        left = Array(TorchArrayData(left_tensor), copy=False)
        right = Array(TorchArrayData(right_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("NumPy broadcasting copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            if function is np.broadcast_to:
                outputs = (function(left, (2, 3)),)
            else:
                outputs = function(left, right)

        assert isinstance(outputs, tuple)
        assert all(type(output) is Array for output in outputs)
        assert all(output.shape == (2, 3) for output in outputs)
        assert all(
            output._data.tensor.device.type == device for output in outputs
        )
        torch.testing.assert_close(
            outputs[0]._data.tensor,
            left_tensor.reshape(1, 3).expand(2, 3),
        )
        if function is np.broadcast_arrays:
            torch.testing.assert_close(
                outputs[1]._data.tensor, right_tensor.expand(2, 3)
            )


def test_numpy_broadcast_views_preserve_identity_and_autograd(
        torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            3, dtype=torch.float32, device=device
        ).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)

        unchanged = np.broadcast_arrays(values)[0]
        same_shape = np.broadcast_arrays(values, values)
        same_shape_view = np.broadcast_to(values, (3,))
        expanded = np.broadcast_to(values, (2, 3))
        weights = torch.arange(
            1, 7, dtype=torch.float32, device=device
        ).reshape(2, 3)
        (expanded._data.tensor * weights).sum().backward()

        assert unchanged is values
        assert same_shape[0] is values
        assert same_shape[1] is values
        assert same_shape_view is not values
        assert expanded is not values
        assert expanded.shape == (2, 3)
        torch.testing.assert_close(
            source.grad, weights.sum(dim=0)
        )


def test_numpy_broadcast_shape_and_fallback_contract(monkeypatch):
    source = np.arange(3, dtype=np.float64)
    other = np.arange(2, dtype=np.float64).reshape(2, 1)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        scalar_shape = np.broadcast_to(values, np.int64(3))
        array_shape = np.broadcast_to(
            array=values, shape=np.array([2, 3]), subok=False
        )

        assert type(scalar_shape) is Array
        assert scalar_shape.shape == (3,)
        assert type(array_shape) is Array
        assert array_shape.shape == (2, 3)
        np.testing.assert_array_equal(
            array_shape.numpy(), np.broadcast_to(source, (2, 3))
        )

        mixed = np.broadcast_arrays(values, other)
        assert isinstance(mixed, tuple)
        assert all(isinstance(value, np.ndarray) for value in mixed)
        expected_mixed = np.broadcast_arrays(source, other)
        for actual, expected in zip(mixed, expected_mixed):
            np.testing.assert_array_equal(actual, expected)

        full = Array(np.broadcast_to(source, (2, 3)).copy())
        column = Array(other)
        identity_result = np.broadcast_arrays(full, column)
        assert identity_result[0] is full
        assert identity_result[1] is not column

        subclass_fallback = np.broadcast_to(
            values, (2, 3), subok=True
        )
        assert isinstance(subclass_fallback, np.ndarray)
        np.testing.assert_array_equal(
            subclass_fallback, np.broadcast_to(source, (2, 3), subok=True)
        )
        subclass_array_fallback = np.broadcast_arrays(
            values, Array(other), subok=True
        )
        assert all(
            isinstance(value, np.ndarray)
            for value in subclass_array_fallback
        )

        with pytest.raises(ValueError):
            np.broadcast_to(values, (-1, 3))
        with pytest.raises(ValueError):
            np.broadcast_arrays(values, Array(np.arange(4)))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_broadcast",
                lambda self, function, args, kwargs: NotImplemented,
            )
            backend_fallback = np.broadcast_to(values, (2, 3))

        assert isinstance(backend_fallback, np.ndarray)
        np.testing.assert_array_equal(
            backend_fallback, np.broadcast_to(source, (2, 3))
        )


def test_numpy_diagonal_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        values = Array(source)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("diagonal selection copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.diagonal(
                values, offset=np.int64(1), axis1=-2, axis2=-1
            )
            keyword = np.diagonal(
                a=values, offset=0, axis1=np.int64(1), axis2=np.int64(2)
            )
            method = values.diagonal(offset=-1, axis1=1, axis2=2)

        assert type(actual) is Array
        assert type(keyword) is Array
        assert type(method) is Array
        assert actual._data.tensor.device.type == device
        assert keyword._data.tensor.device.type == device
        assert method._data.tensor.device.type == device
        np.testing.assert_array_equal(
            actual.numpy(), np.diagonal(source, offset=1, axis1=-2, axis2=-1)
        )
        np.testing.assert_array_equal(
            keyword.numpy(), np.diagonal(source, offset=0, axis1=1, axis2=2)
        )
        np.testing.assert_array_equal(
            method.numpy(), np.diagonal(source, offset=-1, axis1=1, axis2=2)
        )


def test_numpy_diagonal_view_autograd_and_empty_offset(torch_device_ctx):
    ctx, device = torch_device_ctx

    with ctx:
        source = torch.arange(
            24, dtype=torch.float32, device=device
        ).reshape(2, 3, 4).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        diagonal = np.diagonal(values, axis1=1, axis2=2)
        empty = np.diagonal(values, offset=10, axis1=1, axis2=2)
        diagonal._data.tensor.sum().backward()

        assert diagonal.shape == (2, 3)
        assert empty.shape == (2, 0)
        assert diagonal._data.tensor.untyped_storage().data_ptr() == (
            source.untyped_storage().data_ptr()
        )
        expected_gradient = torch.zeros_like(source)
        expected_gradient[:, torch.arange(3), torch.arange(3)] = 1
        torch.testing.assert_close(source.grad, expected_gradient)


def test_numpy_diagonal_errors_and_backend_fallback(monkeypatch):
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    with scheme.TorchScheme("cpu"):
        values = Array(source)

        with pytest.raises(ValueError, match="axis1 and axis2"):
            np.diagonal(values, axis1=1, axis2=1)
        with pytest.raises(np.exceptions.AxisError):
            np.diagonal(values, axis1=3, axis2=1)
        with pytest.raises(TypeError):
            np.diagonal(values, offset=0.5)
        with pytest.raises(ValueError, match="at least two dimensions"):
            np.diagonal(Array(np.array(1.0)))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_diagonal",
                lambda self, args, kwargs: NotImplemented,
            )
            fallback = np.diagonal(values, offset=1, axis1=1, axis2=2)

        assert isinstance(fallback, np.ndarray)
        np.testing.assert_array_equal(
            fallback, np.diagonal(source, offset=1, axis1=1, axis2=2)
        )


def test_numpy_trace_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    with ctx:
        values = Array(source)
        out = Array(np.empty(4, dtype=np.float32))

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("trace reduction copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.trace(
                values, offset=np.int64(1), axis1=-2, axis2=-1
            )
            keyword = np.trace(
                a=values, offset=0, axis1=np.int64(1), axis2=np.int64(2)
            )
            method = values.trace(offset=-1, axis1=1, axis2=2)
            returned = np.trace(values, dtype=np.float32, out=out)
            scalar = np.trace(Array(source[0]))

        assert type(actual) is Array
        assert type(keyword) is Array
        assert type(method) is Array
        assert actual._data.tensor.device.type == device
        assert keyword._data.tensor.device.type == device
        assert method._data.tensor.device.type == device
        assert returned is out
        assert out._data.tensor.device.type == device
        assert isinstance(scalar, np.float32)

        np.testing.assert_array_equal(
            actual.numpy(), np.trace(
                source, offset=1, axis1=-2, axis2=-1
            )
        )
        np.testing.assert_array_equal(
            keyword.numpy(), np.trace(source, axis1=1, axis2=2)
        )
        np.testing.assert_array_equal(
            method.numpy(), np.trace(source, offset=-1, axis1=1, axis2=2)
        )
        np.testing.assert_array_equal(out.numpy(), np.trace(source))
        assert scalar == np.trace(source[0])


def test_numpy_trace_dtype_promotion_and_autograd(torch_ctx, monkeypatch):
    integer = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    boolean = (integer % 3) == 0
    unsigned = integer.astype(np.uint32)

    with torch_ctx:
        integer_array = Array(integer)
        boolean_array = Array(boolean)
        unsigned_array = Array(unsigned)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("native trace copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            integer_trace = np.trace(integer_array, axis1=1, axis2=2)
            boolean_trace = np.trace(boolean_array, axis1=1, axis2=2)
            integer_int32 = np.trace(
                integer_array, axis1=1, axis2=2, dtype=np.int32
            )
            integer_bool = np.trace(
                integer_array, axis1=1, axis2=2, dtype=np.bool_
            )
            unsigned_override = np.trace(
                unsigned_array, axis1=1, axis2=2, dtype=np.int64
            )

        assert type(integer_trace) is Array
        assert integer_trace.dtype == np.dtype(np.int64)
        assert type(boolean_trace) is Array
        assert boolean_trace.dtype == np.dtype(np.int64)
        assert integer_int32.dtype == np.dtype(np.int32)
        assert integer_bool.dtype == np.dtype(np.bool_)
        assert type(unsigned_override) is Array
        assert unsigned_override.dtype == np.dtype(np.int64)

        source = torch.arange(
            24, dtype=torch.float32
        ).reshape(2, 3, 4).requires_grad_()
        differentiable = Array(TorchArrayData(source), copy=False)
        traced = np.trace(differentiable, axis1=1, axis2=2)
        traced._data.tensor.sum().backward()

        expected_gradient = torch.zeros_like(source)
        expected_gradient[:, torch.arange(3), torch.arange(3)] = 1
        torch.testing.assert_close(source.grad, expected_gradient)

        # NumPy's default uint32 accumulator is uint64, which PyCBC Array
        # intentionally does not support, so this exact case stays a host
        # fallback rather than silently changing dtype.
        unsigned_default = np.trace(
            unsigned_array, axis1=1, axis2=2
        )

    np.testing.assert_array_equal(integer_trace.numpy(), np.trace(
        integer, axis1=1, axis2=2
    ))
    np.testing.assert_array_equal(boolean_trace.numpy(), np.trace(
        boolean, axis1=1, axis2=2
    ))
    np.testing.assert_array_equal(integer_int32.numpy(), np.trace(
        integer, axis1=1, axis2=2, dtype=np.int32
    ))
    np.testing.assert_array_equal(integer_bool.numpy(), np.trace(
        integer, axis1=1, axis2=2, dtype=np.bool_
    ))
    np.testing.assert_array_equal(unsigned_override.numpy(), np.trace(
        unsigned, axis1=1, axis2=2, dtype=np.int64
    ))
    assert isinstance(unsigned_default, np.ndarray)
    assert unsigned_default.dtype == np.dtype(np.uint64)


def test_numpy_trace_out_errors_and_backend_fallback(monkeypatch):
    source = np.arange(24, dtype=np.complex64).reshape(2, 3, 4)
    source += 1j * source

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        out = Array(np.empty(4, dtype=np.float32))
        with pytest.warns(np.exceptions.ComplexWarning):
            returned = np.trace(values, out=out)
        assert returned is out
        np.testing.assert_array_equal(
            out.numpy(), np.trace(source).real.astype(np.float32)
        )

        with pytest.raises(ValueError, match="axis1 and axis2"):
            np.trace(values, axis1=1, axis2=1)
        with pytest.raises(np.exceptions.AxisError):
            np.trace(values, axis1=3, axis2=1)
        with pytest.raises(TypeError):
            np.trace(values, offset=0.5)
        with pytest.raises(ValueError, match="at least two dimensions"):
            np.trace(Array(np.array(1.0)))
        with pytest.raises(ValueError, match="output parameter"):
            np.trace(values, out=Array(np.empty(3, dtype=np.float32)))

        host_out = np.empty(4, dtype=np.complex64)
        host_returned = np.trace(values, out=host_out)
        assert host_returned is host_out
        np.testing.assert_array_equal(host_out, np.trace(source))

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_trace",
                lambda self, args, kwargs: NotImplemented,
            )
            fallback = np.trace(values, offset=1, axis1=1, axis2=2)

        assert isinstance(fallback, np.ndarray)
        np.testing.assert_array_equal(
            fallback, np.trace(source, offset=1, axis1=1, axis2=2)
        )


def test_numpy_close_comparisons_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32
    left_values = np.array(
        [[1.0, np.nan, np.inf, -np.inf], [2.0, 3.0, 4.0, 5.0]],
        dtype=dtype,
    )
    right_values = np.array(
        [1.00001, np.nan, np.inf, np.inf], dtype=dtype
    )
    expected = np.isclose(
        left_values,
        right_values,
        rtol=2e-5,
        atol=1e-7,
        equal_nan=True,
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy close comparison copied Torch data to host")

    with ctx:
        left = Array(left_values)
        right = Array(right_values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            elementwise = np.isclose(
                left,
                right,
                rtol=2e-5,
                atol=1e-7,
                equal_nan=True,
            )
            scalar = np.isclose(left, 3.0, atol=0.0)
            reverse_scalar = np.isclose(3.0, left, atol=0.0)
            identical = np.allclose(left, left, equal_nan=True)
            different = np.allclose(
                left, right, 2e-5, 1e-7, True
            )

    for result in (elementwise, scalar, reverse_scalar):
        assert type(result) is Array
        assert result.dtype == np.dtype(np.bool_)
        assert result._data.tensor.device.type == device
    assert identical is True
    assert different is False
    np.testing.assert_array_equal(elementwise.numpy(), expected)
    np.testing.assert_array_equal(
        scalar.numpy(), np.isclose(left_values, 3.0, atol=0.0)
    )
    np.testing.assert_array_equal(
        reverse_scalar.numpy(), np.isclose(3.0, left_values, atol=0.0)
    )


@pytest.mark.parametrize(
    "dtype, values, comparison",
    (
        (np.bool_, [True, False], [True, True]),
        (
            np.int32,
            [0, np.iinfo(np.int32).max],
            [1, np.iinfo(np.int32).max],
        ),
        (np.int64, [0, 2**53], [1, 2**53 + 1]),
        (
            np.uint32,
            [0, np.iinfo(np.uint32).max],
            [1, np.iinfo(np.uint32).max - 1],
        ),
        (np.float32, [1.0, 2.0], [1.00001, 2.1]),
        (np.float64, [1.0, 2.0], [1.00001, 2.1]),
        (np.complex64, [1 + 2j, 3 - 4j], [1.00001 + 2j, 3 - 3j]),
        (np.complex128, [1 + 2j, 3 - 4j], [1.00001 + 2j, 3 - 3j]),
    ),
)
def test_numpy_close_dtype_semantics(dtype, values, comparison):
    left_values = np.asarray(values, dtype=dtype)
    right_values = np.asarray(comparison, dtype=dtype)
    expected = np.isclose(left_values, right_values)

    with scheme.TorchScheme("cpu"):
        left = Array(left_values)
        right = Array(right_values)
        actual = np.isclose(left, right)
        reduced = np.allclose(a=left, b=right)

    assert actual.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert reduced is bool(np.all(expected))


def test_numpy_close_promotion_errors_and_fallback_contract():
    left_values = np.array([[1.0], [2.0]], dtype=np.float32)
    right_values = np.array([1.0, 2.000001], dtype=np.float64)
    with scheme.TorchScheme("cpu"):
        left = Array(left_values)
        right = Array(right_values)
        promoted = np.isclose(left, right, rtol=np.float64(1e-5))

        assert type(promoted) is Array
        assert promoted.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(
            promoted.numpy(), np.isclose(left_values, right_values)
        )
        with pytest.raises(ValueError, match="size"):
            np.isclose(left, Array(np.ones((3, 2), dtype=np.float32)))

        mixed = np.isclose(left, right_values)
        array_tolerance = np.isclose(
            left, right, rtol=np.array([1e-5, 1e-4])
        )
        negative_tolerance = np.isclose(left, right, rtol=-1.0)

    assert isinstance(mixed, np.ndarray)
    assert isinstance(array_tolerance, np.ndarray)
    assert isinstance(negative_tolerance, np.ndarray)
    np.testing.assert_array_equal(
        mixed, np.isclose(left_values, right_values)
    )
    np.testing.assert_array_equal(
        array_tolerance,
        np.isclose(left_values, right_values, rtol=np.array([1e-5, 1e-4])),
    )
    np.testing.assert_array_equal(
        negative_tolerance,
        np.isclose(left_values, right_values, rtol=-1.0),
    )

    fallback = np.isclose(Array(left_values), right_values)
    reduced = np.allclose(Array(left_values), right_values)
    assert isinstance(fallback, np.ndarray)
    assert isinstance(reduced, bool)


def test_numpy_array_equality_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    left_values = np.array(
        [[1.0, np.nan, 3.0], [1.0, np.nan, 3.0]], dtype=np.float32
    )
    row_values = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    broadcast_values = np.array(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32
    )
    changed_values = left_values.copy()
    changed_values[-1, -1] = 4.0

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("array equality copied Torch data to host")

    with ctx:
        left = Array(left_values)
        equal = Array(left_values)
        row = Array(row_values)
        broadcast = Array(broadcast_values)
        broadcast_row = Array(broadcast_values[:1])
        changed = Array(changed_values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            exact = np.array_equal(left, equal)
            nan_equal = np.array_equal(left, equal, equal_nan=True)
            different = np.array_equal(left, changed, equal_nan=True)
            shape_mismatch = np.array_equal(left, row, equal_nan=True)
            nan_not_equivalent = np.array_equiv(left, row)
            equivalent = np.array_equiv(broadcast, broadcast_row)

    assert left._data.tensor.device.type == device
    assert exact is False
    assert nan_equal is True
    assert different is False
    assert shape_mismatch is False
    assert nan_not_equivalent is False
    assert equivalent is True


@pytest.mark.parametrize(
    "dtype, left_values, right_values",
    (
        (np.bool_, [True, False], [1, 0]),
        (np.int32, [-1, 2], [-1, 2]),
        (np.int64, [2**53 + 1, -2], [2**53, -2]),
        (np.uint32, [0, 2**32 - 1], [0, 2**32 - 1]),
        (np.float32, [1.0, np.nan], [1.0, np.nan]),
        (np.float64, [1.0, np.nan], [1.0, np.nan]),
        (np.complex64, [1 + 2j, np.nan + 1j], [1 + 2j, 2 + np.nan * 1j]),
        (np.complex128, [1 + 2j, np.nan + 1j], [1 + 2j, 2 + np.nan * 1j]),
    ),
)
def test_numpy_array_equality_dtype_semantics(
        dtype, left_values, right_values):
    left_values = np.asarray(left_values, dtype=dtype)
    if dtype is np.bool_:
        right_values = np.asarray(right_values, dtype=np.int32)
    elif dtype is np.int64:
        right_values = np.asarray(right_values, dtype=np.float64)
    else:
        right_values = np.asarray(right_values, dtype=dtype)

    with scheme.TorchScheme("cpu"):
        left = Array(left_values)
        right = Array(right_values)
        exact = np.array_equal(left, right)
        nan_equal = np.array_equal(a1=left, a2=right, equal_nan=True)
        equivalent = np.array_equiv(
            Array(left_values.reshape(1, -1)), right
        )

    assert exact is np.array_equal(left_values, right_values)
    assert nan_equal is np.array_equal(
        left_values, right_values, equal_nan=True
    )
    assert equivalent is np.array_equiv(
        left_values.reshape(1, -1), right_values
    )


def test_numpy_array_equality_fallback_contract():
    left_values = np.array([[1.0], [2.0]], dtype=np.float32)
    right_values = np.array([1.0, 2.0], dtype=np.float32)
    with scheme.TorchScheme("cpu"):
        left = Array(left_values)
        mixed_equal = np.array_equal(left, right_values)
        mixed_equiv = np.array_equiv(left, right_values)

    assert isinstance(mixed_equal, bool)
    assert isinstance(mixed_equiv, bool)
    assert mixed_equal is np.array_equal(left_values, right_values)
    assert mixed_equiv is np.array_equiv(left_values, right_values)


def test_numpy_count_nonzero_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    source = np.array(
        [
            [[0.0, -0.0, 2.0], [np.nan, -3.0, 0.0]],
            [[np.inf, 0.0, 4.0], [0.0, 5.0, -6.0]],
        ],
        dtype=np.float32,
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("count_nonzero copied Torch data to host")

    with ctx:
        values = Array(source)
        unsigned_values = (
            Array(np.array([[0, 2], [3, 0]], dtype=np.uint32))
            if device == "cpu" else None
        )
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            scalar = np.count_nonzero(values)
            axis_result = np.count_nonzero(values, axis=1)
            tuple_result = np.count_nonzero(
                a=values, axis=(0, 2), keepdims=True
            )
            retained = np.count_nonzero(values, axis=())
            negative_axis = np.count_nonzero(
                values, axis=-1, keepdims=True
            )
            unsigned_result = (
                np.count_nonzero(unsigned_values, keepdims=True)
                if unsigned_values is not None else None
            )

    assert type(scalar) is np.intp
    assert scalar == np.count_nonzero(source)
    for result in (axis_result, tuple_result, retained, negative_axis):
        assert type(result) is Array
        assert result.dtype == np.dtype(np.intp)
        assert result._data.tensor.dtype == torch.int64
        assert result._data.tensor.device.type == device
    np.testing.assert_array_equal(
        axis_result.numpy(), np.count_nonzero(source, axis=1)
    )
    np.testing.assert_array_equal(
        tuple_result.numpy(),
        np.count_nonzero(source, axis=(0, 2), keepdims=True),
    )
    np.testing.assert_array_equal(
        retained.numpy(), np.count_nonzero(source, axis=())
    )
    np.testing.assert_array_equal(
        negative_axis.numpy(),
        np.count_nonzero(source, axis=-1, keepdims=True),
    )
    if unsigned_result is not None:
        assert type(unsigned_result) is Array
        assert unsigned_result._data.tensor.device.type == "cpu"
        np.testing.assert_array_equal(unsigned_result.numpy(), [[2]])


def test_numpy_count_nonzero_dtype_empty_and_fallback_contract():
    sources = (
        np.array([[True, False], [True, True]]),
        np.array([[0, -2, 3], [4, 0, 0]], dtype=np.int32),
        np.array([[0, 2, 0], [4, 0, 6]], dtype=np.uint32),
        np.array([[0 + 0j, 1 - 2j], [np.nan + 0j, -0j]]),
    )
    with scheme.TorchScheme("cpu"):
        for source in sources:
            values = Array(source)
            for axis in (None, 0, -1, (0, 1), ()):
                for keepdims in (False, True):
                    actual = np.count_nonzero(
                        values, axis=axis, keepdims=keepdims
                    )
                    expected = np.count_nonzero(
                        source, axis=axis, keepdims=keepdims
                    )
                    if np.isscalar(expected):
                        assert type(actual) is np.intp
                        assert actual == expected
                    else:
                        assert type(actual) is Array
                        assert actual.dtype == np.dtype(np.intp)
                        np.testing.assert_array_equal(actual.numpy(), expected)

        empty_source = np.empty((2, 0, 3), dtype=np.float32)
        empty = Array(empty_source)
        for axis in (None, 0, 1, -1, (0, 2), ()):
            actual = np.count_nonzero(empty, axis=axis, keepdims=True)
            expected = np.count_nonzero(
                empty_source, axis=axis, keepdims=True
            )
            if np.isscalar(expected):
                assert actual == expected
            else:
                np.testing.assert_array_equal(actual.numpy(), expected)

        singleton_source = np.array([2], dtype=np.int32)
        singleton = Array(singleton_source)
        assert np.count_nonzero(singleton, axis=0) == np.intp(1)
        singleton_kept = np.count_nonzero(
            singleton, axis=-1, keepdims=True
        )
        np.testing.assert_array_equal(
            singleton_kept.numpy(),
            np.count_nonzero(singleton_source, axis=-1, keepdims=True),
        )

        with pytest.raises(ValueError, match="duplicate value"):
            np.count_nonzero(Array(sources[1]), axis=(0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.count_nonzero(Array(sources[1]), axis=2)
        with pytest.raises(TypeError):
            np.count_nonzero(
                Array(sources[1]), keepdims=np.bool_(True)
            )
        with pytest.raises(np.exceptions.AxisError):
            np.count_nonzero(singleton, axis=(1,))


def test_numpy_average_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    rng = np.random.default_rng(8127)
    source = rng.normal(size=(2, 3, 4)).astype(np.float32)
    weight_source = rng.uniform(0.25, 2.0, size=(3, 4)).astype(np.float32)
    expected_weighted = np.average(
        source,
        axis=(1, 2),
        weights=weight_source,
        returned=True,
        keepdims=True,
    )
    expected_unweighted = np.average(source, axis=-1)
    expected_scalar = np.average(source)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("average copied Torch data to host")

    with ctx:
        values = Array(source)
        weights = Array(weight_source)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            weighted, scale = np.average(
                values,
                axis=(1, 2),
                weights=weights,
                returned=True,
                keepdims=True,
            )
            unweighted = np.average(values, axis=-1)
            scalar = np.average(values)

    for result in (weighted, scale, unweighted):
        assert type(result) is Array
        assert result.dtype == np.dtype(np.float32)
        assert result._data.tensor.device.type == device
    assert type(scalar) is np.float32
    np.testing.assert_allclose(
        weighted.numpy(), expected_weighted[0], rtol=2e-6
    )
    np.testing.assert_allclose(
        scale.numpy(), expected_weighted[1], rtol=2e-6
    )
    np.testing.assert_allclose(
        unweighted.numpy(), expected_unweighted, rtol=2e-6
    )
    np.testing.assert_allclose(scalar, expected_scalar, rtol=2e-6)


def test_numpy_average_randomized_dtype_axis_and_scalar_parity():
    rng = np.random.default_rng(1776)
    real = rng.normal(size=(2, 3, 4))
    complex_source = real + 1j * rng.normal(size=(2, 3, 4))
    cases = (
        (
            real.astype(np.float32),
            {
                "axis": (1, 2),
                "weights": rng.uniform(
                    0.25, 2.0, size=(3, 4)
                ).astype(np.float32),
                "returned": True,
                "keepdims": True,
            },
        ),
        (
            np.arange(24, dtype=np.int32).reshape(2, 3, 4),
            {
                "axis": -1,
                # Python floating weights must promote like NumPy float64,
                # rather than inheriting Torch's float32 default.
                "weights": [0.5, 1.5, 2.5, 3.5],
                "returned": True,
            },
        ),
        (
            complex_source.astype(np.complex64),
            {
                "axis": (2, 0),
                "weights": rng.uniform(0.5, 1.5, size=(4, 2)),
                "keepdims": True,
            },
        ),
        (
            rng.integers(0, 2, size=(2, 3, 4), dtype=np.int32).astype(bool),
            {"axis": (0, 2), "returned": True},
        ),
        (
            real.astype(np.float64),
            {
                "axis": (),
                "weights": np.array(2.0),
                "returned": True,
                "keepdims": True,
            },
        ),
        (
            real.astype(np.float32),
            {
                "axis": None,
                "weights": rng.uniform(0.5, 1.5, size=(2, 3, 4)),
                "returned": True,
            },
        ),
    )

    def assert_parity(actual, expected):
        if np.isscalar(expected):
            assert type(actual) is type(expected)
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
        else:
            assert type(actual) is Array
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(
                actual.numpy(), expected, rtol=2e-6
            )

    with scheme.TorchScheme("cpu"):
        for source, options in cases:
            actual = np.average(Array(source), **options)
            expected = np.average(source, **options)
            if isinstance(expected, tuple):
                assert isinstance(actual, tuple)
                for actual_value, expected_value in zip(actual, expected):
                    assert_parity(actual_value, expected_value)
            else:
                assert_parity(actual, expected)


def test_numpy_average_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [
                [[1.0, 2.0, -1.0, 4.0], [2.0, -3.0, 5.0, 1.0],
                 [0.5, 3.0, 2.0, -2.0]],
                [[-1.0, 4.0, 2.0, 0.5], [3.0, 1.0, -2.0, 6.0],
                 [2.5, -1.0, 4.0, 3.0]],
            ],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        weight_source = torch.tensor(
            [
                [0.5, 1.0, 1.5],
                [1.25, 0.75, 2.0],
                [0.8, 1.1, 0.9],
                [1.6, 0.6, 1.4],
            ],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        reference = source.detach().clone().requires_grad_(True)
        reference_weights = (
            weight_source.detach().clone().requires_grad_(True)
        )
        values = Array(TorchArrayData(source), copy=False)
        weights = Array(TorchArrayData(weight_source), copy=False)

        actual, actual_scale = np.average(
            values,
            axis=(2, 1),
            weights=weights,
            returned=True,
            keepdims=True,
        )
        broadcast_weights = reference_weights.transpose(0, 1).reshape(
            1, 3, 4
        )
        reference_scale = broadcast_weights.sum(
            dim=(1, 2), keepdim=True
        ).expand(2, 1, 1).clone()
        expected = (
            (reference * broadcast_weights).sum(
                dim=(1, 2), keepdim=True
            ) / reference_scale
        )

        actual_loss = (
            actual._data.tensor.square().sum()
            + 0.05 * actual_scale._data.tensor.square().sum()
        )
        expected_loss = (
            expected.square().sum()
            + 0.05 * reference_scale.square().sum()
        )
        actual_loss.backward()
        expected_loss.backward()

        assert actual._data.tensor.device.type == device
        assert actual_scale._data.tensor.device.type == device
        torch.testing.assert_close(actual._data.tensor, expected)
        torch.testing.assert_close(
            actual_scale._data.tensor, reference_scale
        )
        torch.testing.assert_close(source.grad, reference.grad)
        torch.testing.assert_close(
            weight_source.grad, reference_weights.grad
        )


def test_numpy_average_errors_empty_and_fallback_contract(monkeypatch):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        with pytest.raises(TypeError, match="Axis must be specified"):
            np.average(values, weights=Array(np.ones(3)))
        with pytest.raises(ValueError, match="Shape of weights"):
            np.average(values, axis=1, weights=Array(np.ones(2)))
        with pytest.raises(ZeroDivisionError, match="Weights sum to zero"):
            np.average(
                values,
                axis=1,
                weights=Array(np.array([1.0, -1.0, 0.0])),
            )
        with pytest.raises(ValueError, match="repeated axis"):
            np.average(values, axis=(0, 0))
        with pytest.raises(TypeError):
            np.average(values, keepdims=np.bool_(True))

        empty_source = np.empty((2, 0, 3), dtype=np.float32)
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            empty_average, empty_scale = np.average(
                Array(empty_source),
                axis=1,
                returned=True,
                keepdims=True,
            )
        assert torch.isnan(empty_average._data.tensor).all()
        np.testing.assert_array_equal(
            empty_scale.numpy(), np.zeros((2, 1, 3), dtype=np.float32)
        )
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            np.average(Array(empty_source), axis=0, returned=True)

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_average",
                lambda *_args, **_kwargs: NotImplemented,
            )
            fallback = np.average(
                values,
                axis=1,
                weights=[0.5, 1.5, 2.5],
                returned=True,
                keepdims=True,
            )

    expected_fallback = np.average(
        source,
        axis=1,
        weights=[0.5, 1.5, 2.5],
        returned=True,
        keepdims=True,
    )
    assert all(isinstance(value, np.ndarray) for value in fallback)
    for actual, expected in zip(fallback, expected_fallback):
        np.testing.assert_allclose(actual, expected)


def test_numpy_average_preserves_host_weight_subclasses_and_scale_storage():
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    masked_weights = np.ma.array(
        [1.0, 2.0, 3.0], mask=[False, True, False]
    )
    matrix_weights = np.matrix(
        np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)
    )

    with scheme.TorchScheme("cpu"):
        masked = np.average(
            Array(source),
            axis=1,
            weights=masked_weights,
            returned=True,
        )
        matrix = np.average(
            Array(source),
            axis=(),
            weights=matrix_weights,
            returned=True,
        )

        weights = Array(matrix_weights.A.copy())
        _, scale = np.average(
            Array(source), axis=(), weights=weights, returned=True
        )
        scale._data.tensor[0, 0] = -123.0

    expected_masked = np.average(
        source, axis=1, weights=masked_weights, returned=True
    )
    expected_matrix = np.average(
        source, axis=(), weights=matrix_weights, returned=True
    )
    assert isinstance(masked[0], np.ma.MaskedArray)
    assert isinstance(matrix[0], np.matrix)
    assert isinstance(matrix[1], np.matrix)
    np.testing.assert_array_equal(masked[0], expected_masked[0])
    np.testing.assert_array_equal(masked[1], expected_masked[1])
    np.testing.assert_array_equal(matrix[0], expected_matrix[0])
    np.testing.assert_array_equal(matrix[1], expected_matrix[1])
    assert weights[0, 0] == 1.0


@pytest.mark.parametrize(
    "torch_dtype", (torch.float16, torch.uint8, torch.int8, torch.int16)
)
def test_numpy_average_unsupported_raw_cpu_torch_weights_fall_back(
        torch_dtype):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight_values = np.array([1, 2, 3], dtype=np.float64)

    with scheme.TorchScheme("cpu"):
        weights = torch.tensor(weight_values, dtype=torch_dtype)
        actual = np.average(
            Array(source), axis=1, weights=weights, returned=True
        )

    expected = np.average(
        source,
        axis=1,
        weights=weight_values.astype(weights.numpy().dtype),
        returned=True,
    )
    assert all(isinstance(value, np.ndarray) for value in actual)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value)


def test_numpy_ptp_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.array(
        [
            [[9.0, -1.0, 5.0], [2.0, 7.0, 3.0]],
            [[4.0, 8.0, -6.0], [1.0, 0.0, 10.0]],
        ],
        dtype=dtype,
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("ptp copied Torch data to host")

    with ctx:
        values = Array(source)
        output = Array(np.empty((2, 3), dtype=dtype))
        unsigned = (
            Array(np.array([[0, 7], [9, 2]], dtype=np.uint32))
            if device == "cpu" else None
        )
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            scalar = np.ptp(values)
            returned = np.ptp(values, axis=1, out=output)
            tuple_result = np.ptp(
                a=values, axis=(0, 2), keepdims=True
            )
            retained = np.ptp(values, axis=())
            negative_axis = np.ptp(values, axis=-1, keepdims=True)
            unsigned_result = (
                np.ptp(unsigned, axis=0) if unsigned is not None else None
            )

    assert type(scalar) is np.dtype(dtype).type
    assert scalar == np.ptp(source)
    assert returned is output
    for result in (output, tuple_result, retained, negative_axis):
        assert type(result) is Array
        assert result.dtype == np.dtype(dtype)
        assert result._data.tensor.device.type == device
    np.testing.assert_array_equal(output.numpy(), np.ptp(source, axis=1))
    np.testing.assert_array_equal(
        tuple_result.numpy(),
        np.ptp(source, axis=(0, 2), keepdims=True),
    )
    np.testing.assert_array_equal(retained.numpy(), np.ptp(source, axis=()))
    np.testing.assert_array_equal(
        negative_axis.numpy(), np.ptp(source, axis=-1, keepdims=True)
    )
    if unsigned_result is not None:
        assert type(unsigned_result) is Array
        assert unsigned_result.dtype == np.dtype(np.uint32)
        assert unsigned_result._data.tensor.device.type == "cpu"
        np.testing.assert_array_equal(unsigned_result.numpy(), [9, 5])


def test_numpy_ptp_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [[1.0, 5.0, -2.0], [8.0, 2.0, 4.0]],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        reference = source.detach().clone().requires_grad_(True)
        values = Array(TorchArrayData(source), copy=False)

        actual = np.ptp(values, axis=1, keepdims=True)
        expected = (
            torch.amax(reference, dim=1, keepdim=True)
            - torch.amin(reference, dim=1, keepdim=True)
        )
        actual._data.tensor.square().sum().backward()
        expected.square().sum().backward()

        torch.testing.assert_close(actual._data.tensor, expected)
        torch.testing.assert_close(source.grad, reference.grad)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in subtract:RuntimeWarning"
)
def test_numpy_ptp_dtype_empty_and_fallback_contract():
    integer_sources = (
        np.array(
            [[np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max]],
            dtype=np.int32,
        ),
        np.array([[0, 7, np.iinfo(np.uint32).max], [3, 1, 8]], np.uint32),
        np.array([[1.0, np.nan, 4.0], [-np.inf, 2.0, np.inf]]),
    )
    with scheme.TorchScheme("cpu"):
        for source in integer_sources:
            values = Array(source)
            for axis in (None, 0, -1, (0, 1), ()):
                for keepdims in (False, True):
                    actual = np.ptp(values, axis=axis, keepdims=keepdims)
                    expected = np.ptp(source, axis=axis, keepdims=keepdims)
                    if np.isscalar(expected):
                        assert type(actual) is type(expected)
                        if np.isnan(expected):
                            assert np.isnan(actual)
                        else:
                            assert actual == expected
                    else:
                        assert type(actual) is Array
                        assert actual.dtype == expected.dtype
                        np.testing.assert_array_equal(actual.numpy(), expected)

        empty_source = np.empty((2, 0, 3), dtype=np.float32)
        empty = Array(empty_source)
        reduced_nonempty_axis = np.ptp(empty, axis=0, keepdims=True)
        retained = np.ptp(empty, axis=())
        np.testing.assert_array_equal(
            reduced_nonempty_axis.numpy(),
            np.ptp(empty_source, axis=0, keepdims=True),
        )
        np.testing.assert_array_equal(
            retained.numpy(), np.ptp(empty_source, axis=())
        )
        with pytest.raises(ValueError, match="zero-size array"):
            np.ptp(empty)
        with pytest.raises(ValueError, match="zero-size array"):
            np.ptp(empty, axis=1)

        extreme = Array(integer_sources[0])
        widened = Array(np.empty(1, dtype=np.float64))
        assert np.ptp(extreme, axis=1, out=widened) is widened
        np.testing.assert_array_equal(widened.numpy(), [4294967295.0])
        with pytest.raises(TypeError, match="Cannot cast ufunc"):
            np.ptp(
                extreme,
                axis=1,
                out=Array(np.empty(1, dtype=np.uint32)),
            )

        values = Array(np.arange(6, dtype=np.int32).reshape(2, 3))
        with pytest.raises(ValueError, match="duplicate value"):
            np.ptp(values, axis=(0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.ptp(values, axis=2)
        with pytest.raises(TypeError):
            np.ptp(values, keepdims=np.bool_(True))
        with pytest.raises(ValueError, match="wrong shape"):
            np.ptp(values, axis=0, out=Array(np.empty(2)))

        host_output = np.empty(1, dtype=np.float64)
        returned = np.ptp(extreme, axis=1, out=host_output)
        boolean = Array(np.array([True, False]))
        complex_source = np.array([1 + 2j, 4 - 3j, 2 + 1j])
        complex_result = np.ptp(Array(complex_source))

    assert returned is host_output
    np.testing.assert_array_equal(host_output, [4294967295.0])
    with pytest.raises(TypeError, match="boolean subtract"):
        np.ptp(boolean)
    assert isinstance(complex_result, np.complexfloating)
    assert complex_result == np.ptp(complex_source)


def test_numpy_median_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.array(
        [
            [[9.0, 1.0], [5.0, np.nan], [3.0, 7.0]],
            [[2.0, 8.0], [6.0, 4.0], [10.0, 0.0]],
        ],
        dtype=dtype,
    )
    expected_axis = np.median(source, axis=1)
    expected_tuple = np.median(source, axis=(0, 2), keepdims=True)
    expected_scalar = np.median(source[:, :, 0])

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy median copied Torch data to host")

    with ctx:
        values = Array(source)
        scalar_values = Array(source[:, :, 0])
        output = Array(np.empty((2, 2), dtype=dtype))
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            axis_result = np.median(values, axis=1)
            tuple_result = np.median(
                a=values, axis=(0, 2), keepdims=True
            )
            scalar_result = np.median(scalar_values)
            returned = np.median(values, axis=1, out=output)

    for result in (axis_result, tuple_result, output):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
    assert returned is output
    assert type(scalar_result) is np.dtype(dtype).type
    np.testing.assert_allclose(
        axis_result.numpy(), expected_axis, equal_nan=True
    )
    np.testing.assert_allclose(
        tuple_result.numpy(), expected_tuple, equal_nan=True
    )
    np.testing.assert_allclose(output.numpy(), expected_axis, equal_nan=True)
    assert scalar_result == expected_scalar


def test_numpy_median_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [[1.0, 5.0, 3.0], [8.0, 2.0, 4.0]],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        reference = source.detach().clone().requires_grad_(True)
        values = Array(TorchArrayData(source), copy=False)

        actual = np.median(values, axis=1, keepdims=True)
        expected = torch.median(reference, dim=1, keepdim=True).values
        actual._data.tensor.square().sum().backward()
        expected.square().sum().backward()

        torch.testing.assert_close(actual._data.tensor, expected)
        torch.testing.assert_close(source.grad, reference.grad)


def test_numpy_median_dtype_empty_axes_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        integer_source = np.array([[4, 1, 3], [9, 5, 7]], dtype=np.int32)
        integer = Array(integer_source)
        boolean = Array(np.array([[True, False], [True, True]]))
        empty = Array(np.empty((2, 0, 3), dtype=np.float32))
        unchanged = np.median(integer, axis=())

        integer_result = np.median(integer, axis=0)
        boolean_scalar = np.median(boolean)
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            empty_result = np.median(empty, axis=1, keepdims=True)

        assert integer_result.dtype == np.dtype(np.float64)
        assert type(boolean_scalar) is np.float64
        assert unchanged.dtype == np.dtype(np.float64)
        assert unchanged._data.tensor.data_ptr() != integer._data.tensor.data_ptr()
        np.testing.assert_array_equal(
            integer_result.numpy(), np.median(integer_source, axis=0)
        )
        assert boolean_scalar == np.median(boolean.numpy())
        assert empty_result.shape == (2, 1, 3)
        assert torch.isnan(empty_result._data.tensor).all()

        with pytest.raises(ValueError, match="repeated axis"):
            np.median(integer, axis=(0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.median(integer, axis=2)
        with pytest.raises(ValueError, match="wrong shape"):
            np.median(integer, axis=0, out=Array(np.empty(2)))

        complex_source = np.array([1 + 2j, 4 - 3j, 2 + 1j])
        complex_result = np.median(Array(complex_source))
        foreign_out = np.empty(3, dtype=np.float64)
        foreign_result = np.median(integer, axis=0, out=foreign_out)

    assert isinstance(complex_result, np.complexfloating)
    assert complex_result == np.median(complex_source)
    assert foreign_result is foreign_out
    np.testing.assert_array_equal(
        foreign_result, np.median(integer_source, axis=0)
    )


def test_numpy_unique_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.array(
        [[np.nan, 2.0, -0.0, 2.0], [1.0, np.nan, 0.0, 1.0]],
        dtype=dtype,
    )
    expected = np.unique(
        source,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy unique copied Torch data to host")

    with ctx:
        values = Array(source)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.unique(
                ar=values,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )
            values_only = np.unique(values)

    assert all(type(result) is Array for result in actual)
    assert all(result._data.tensor.device.type == device for result in actual)
    assert type(values_only) is Array
    assert values_only._data.tensor.device.type == device
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result.numpy(), reference)
    np.testing.assert_array_equal(values_only.numpy(), expected[0])
    assert np.signbit(actual[0].numpy()[0])


def test_numpy_unique_equal_nan_false_and_dtype_semantics():
    sources = (
        np.array([3, 1, 3, 2, 1], dtype=np.int32),
        np.array([3, 1, 3, 2, 1], dtype=np.uint32),
        np.array([False, True, False], dtype=np.bool_),
        np.array([0.0, -0.0], dtype=np.float32),
        np.array([-0.0, 0.0], dtype=np.float32),
        np.array([np.nan, 2.0, np.nan, 1.0], dtype=np.float32),
        np.array([], dtype=np.float64),
    )

    with scheme.TorchScheme("cpu"):
        results = [
            np.unique(
                Array(source),
                return_index=True,
                return_inverse=True,
                return_counts=True,
                equal_nan=False,
            )
            for source in sources
        ]

    for source, result in zip(sources, results):
        expected = np.unique(
            source,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            equal_nan=False,
        )
        assert all(type(value) is Array for value in result)
        assert result[0].dtype == source.dtype
        for actual, reference in zip(result, expected):
            np.testing.assert_array_equal(actual.numpy(), reference)


def test_numpy_unique_fallback_contract():
    source = np.array(
        [[1 + 2j, 1 + 2j], [3 - 1j, 2 + 4j]],
        dtype=np.complex128,
    )
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        complex_result = np.unique(values)
        axis_result = np.unique(Array(source.real), axis=0)
        unsorted_result = np.unique(Array(source.real), sorted=False)

    assert isinstance(complex_result, np.ndarray)
    assert isinstance(axis_result, np.ndarray)
    assert isinstance(unsorted_result, np.ndarray)
    np.testing.assert_array_equal(complex_result, np.unique(source))
    np.testing.assert_array_equal(axis_result, np.unique(source.real, axis=0))
    np.testing.assert_array_equal(
        unsorted_result, np.unique(source.real, sorted=False)
    )


def test_numpy_intersect1d_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    source = np.array([[7, 1, 3], [1, 5, 9]], dtype=np.int64)
    tests = np.array([3, 8, 1, 3], dtype=np.int32)
    expected = np.intersect1d(source, tests, return_indices=True)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy intersect1d copied Torch data to host")

    with ctx:
        values = Array(source)
        test_values = Array(tests)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.intersect1d(
                ar1=values, ar2=test_values, return_indices=True
            )
            mixed = np.intersect1d(values, tests)
            assumed = np.intersect1d(
                Array(np.array([9, 1, 5, 3], dtype=np.int64)),
                [3, 8, 1],
                assume_unique=True,
            )

    for result, reference in zip(actual, expected):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
        np.testing.assert_array_equal(result.numpy(), reference)
    for result in (mixed, assumed):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
    np.testing.assert_array_equal(mixed.numpy(), expected[0])
    np.testing.assert_array_equal(assumed.numpy(), np.array([1, 3]))


def test_numpy_intersect1d_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        left = torch.tensor(
            [5.0, 1.0, 3.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        right = torch.tensor(
            [4.0, 3.0, 1.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        result = np.intersect1d(
            Array(TorchArrayData(left), copy=False),
            Array(TorchArrayData(right), copy=False),
            assume_unique=True,
        )
        result._data.tensor.sum().backward()

    assert result._data.tensor.device.type == device
    torch.testing.assert_close(
        left.grad,
        torch.tensor([0, 1, 1], dtype=dtype, device=device),
    )
    torch.testing.assert_close(right.grad, torch.zeros_like(right))


def test_numpy_intersect1d_matches_numpy_contract():
    cases = (
        (
            np.array([[3, 1, 2], [1, 5, 3]], dtype=np.int32),
            np.array([2, 4, 1], dtype=np.float32),
            {},
        ),
        (
            np.array([0, 2**31, 2**32 - 1], dtype=np.uint32),
            np.array([-1, 2**31, 2**32 - 1], dtype=np.int64),
            {"return_indices": True},
        ),
        (
            np.array([False, True, False], dtype=np.bool_),
            np.array([1, 2], dtype=np.uint32),
            {"return_indices": True},
        ),
        (
            np.array([0.0, -0.0], dtype=np.float64),
            np.array([-0.0], dtype=np.float64),
            {},
        ),
        (
            np.array([0.0], dtype=np.float64),
            np.array([-0.0], dtype=np.float64),
            {"return_indices": True},
        ),
        (
            np.array([np.nan, 2.0, np.nan, 1.0]),
            np.array([np.nan, 1.0]),
            {"return_indices": True},
        ),
        (
            np.array([3, 1, 2, 1], dtype=np.int64),
            np.array([2, 4, 1], dtype=np.int64),
            {"assume_unique": True, "return_indices": True},
        ),
        (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float64),
            {"return_indices": True},
        ),
    )

    with scheme.TorchScheme("cpu"):
        results = [
            np.intersect1d(Array(left), Array(right), **options)
            for left, right, options in cases
        ]

    for (left, right, options), result in zip(cases, results):
        expected = np.intersect1d(left, right, **options)
        actual_values = result if isinstance(result, tuple) else (result,)
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        assert all(type(value) is Array for value in actual_values)
        for actual, reference in zip(actual_values, expected_values):
            assert actual.dtype == reference.dtype
            actual_array = actual.numpy()
            np.testing.assert_array_equal(actual_array, reference)
            if reference.dtype.kind == "f":
                zero = reference == 0
                np.testing.assert_array_equal(
                    np.signbit(actual_array[zero]),
                    np.signbit(reference[zero]),
                )


def test_numpy_intersect1d_fallback_contract(monkeypatch):
    left = np.array([1 + 2j, 3 - 1j, 1 + 2j])
    right = np.array([4 + 0j, 1 + 2j])

    with scheme.TorchScheme("cpu"):
        values = Array(left)
        complex_result = np.intersect1d(
            values, right, return_indices=True
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_intersect1d",
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = np.intersect1d(
                Array(np.array([1, 2, 3])), [2, 4]
            )

    assert all(isinstance(value, np.ndarray) for value in complex_result)
    expected = np.intersect1d(left, right, return_indices=True)
    for actual, reference in zip(complex_result, expected):
        np.testing.assert_array_equal(actual, reference)
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_array_equal(backend_fallback, np.array([2]))


def test_numpy_union1d_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    left = np.array([[7, 1, 3], [1, 5, 9]], dtype=np.int64)
    right = np.array([3, 8, 1, 3], dtype=np.int32)
    expected = np.union1d(left, right)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy union1d copied Torch data to host")

    with ctx:
        values = Array(left)
        other = Array(right)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = np.union1d(ar1=values, ar2=other)
            mixed = np.union1d(values, right)

    for result in (actual, mixed):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
        np.testing.assert_array_equal(result.numpy(), expected)


def test_numpy_union1d_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        left = torch.tensor(
            [5.0, 1.0, 3.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        right = torch.tensor(
            [4.0, 3.0, 1.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        result = np.union1d(
            Array(TorchArrayData(left), copy=False),
            Array(TorchArrayData(right), copy=False),
        )
        result._data.tensor.sum().backward()

    assert result._data.tensor.device.type == device
    torch.testing.assert_close(
        left.grad.sum() + right.grad.sum(),
        torch.tensor(4.0, dtype=dtype, device=device),
    )


def test_numpy_union1d_matches_numpy_contract():
    cases = (
        (
            np.array([[3, 1, 2], [1, 5, 3]], dtype=np.int32),
            np.array([2, 4, 1], dtype=np.float32),
        ),
        (
            np.array([0, 2**31, 2**32 - 1], dtype=np.uint32),
            np.array([-1, 2**31, 2**32 - 1], dtype=np.int64),
        ),
        (
            np.array([False, True, False], dtype=np.bool_),
            np.array([1, 2], dtype=np.uint32),
        ),
        (
            np.array([0.0, -0.0], dtype=np.float64),
            np.array([2.0, 0.0], dtype=np.float64),
        ),
        (
            np.array([np.nan, 2.0, np.nan, 1.0]),
            np.array([np.nan, 3.0, 1.0]),
        ),
        (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float64),
        ),
    )

    with scheme.TorchScheme("cpu"):
        results = [
            np.union1d(Array(left), Array(right))
            for left, right in cases
        ]

    for (left, right), result in zip(cases, results):
        expected = np.union1d(left, right)
        assert type(result) is Array
        assert result.dtype == expected.dtype
        actual = result.numpy()
        np.testing.assert_array_equal(actual, expected)
        if expected.dtype.kind == "f":
            zero = expected == 0
            np.testing.assert_array_equal(
                np.signbit(actual[zero]),
                np.signbit(expected[zero]),
            )


def test_numpy_union1d_fallback_contract(monkeypatch):
    left = np.array([1 + 2j, 3 - 1j, 1 + 2j])
    right = np.array([4 + 0j, 1 + 2j])

    with scheme.TorchScheme("cpu"):
        values = Array(left)
        complex_result = np.union1d(values, right)
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_union1d",
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = np.union1d(
                Array(np.array([1, 2, 3])), [2, 4]
            )

    assert isinstance(complex_result, np.ndarray)
    np.testing.assert_array_equal(complex_result, np.union1d(left, right))
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_array_equal(backend_fallback, np.array([1, 2, 3, 4]))


def test_numpy_histogram_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    samples = np.array(
        [[-1.0, 0.0, 0.5, 1.0], [2.0, 3.0, 4.0, 5.0]],
        dtype=dtype,
    )
    weights = np.arange(1, samples.size + 1, dtype=dtype).reshape(
        samples.shape
    )
    explicit_edges = np.array([-1.0, 0.5, 2.0, 5.0], dtype=dtype)
    expected_generated = np.histogram(samples, bins=4)
    expected_density = np.histogram(
        samples, bins=4, range=(-1.0, 5.0), density=True
    )
    expected_weighted = np.histogram(
        samples, bins=explicit_edges, weights=weights
    )

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy histogram copied Torch data to the host")

    with ctx:
        values = Array(samples)
        weight_values = Array(weights)
        edge_values = Array(explicit_edges)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            generated = np.histogram(values, bins=4)
            density = np.histogram(
                values, bins=4, range=(-1.0, 5.0), density=True
            )
            weighted = np.histogram(
                values, bins=edge_values, weights=weight_values
            )

    for result, expected in (
            (generated, expected_generated),
            (density, expected_density),
            (weighted, expected_weighted)):
        assert isinstance(result, tuple)
        assert all(type(value) is Array for value in result)
        assert all(
            value._data.tensor.device.type == device for value in result
        )
        np.testing.assert_allclose(
            result[0].numpy(), expected[0], rtol=2e-5, atol=1e-7
        )
        np.testing.assert_array_equal(result[1].numpy(), expected[1])


def test_numpy_histogram_preserves_weight_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        samples = torch.tensor(
            [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0],
            dtype=dtype,
            device=device,
        )
        weights = torch.arange(
            1, 7, dtype=dtype, device=device, requires_grad=True
        )
        histogram, edges = np.histogram(
            Array(TorchArrayData(samples), copy=False),
            bins=Array(np.array(
                [0.0, 1.0, 2.0],
                dtype=np.float32 if device == "mps" else np.float64,
            )),
            weights=Array(TorchArrayData(weights), copy=False),
        )
        histogram._data.tensor.sum().backward()

    assert histogram._data.tensor.device.type == device
    assert edges._data.tensor.device.type == device
    torch.testing.assert_close(
        weights.grad,
        torch.tensor(
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            dtype=dtype,
            device=device,
        ),
    )


def test_numpy_histogram_matches_numpy_contract():
    cases = (
        (
            np.array([-2.0, -0.5, 0.0, 1.0, 4.0], dtype=np.float32),
            {"bins": 4, "range": (-2.0, 4.0)},
        ),
        (np.array([3, 3, 3], dtype=np.int32), {"bins": 3}),
        (np.array([], dtype=np.float32), {"bins": 4}),
        (
            np.array([[0, 1, 1], [2, 3, 4]], dtype=np.int64),
            {"bins": np.array([0, 1, 1, 4], dtype=np.int32)},
        ),
        (
            np.array([np.nan, -1.0, 0.0, 1.0, 2.0, 5.0]),
            {"bins": np.array([-1.0, 0.0, 2.0, 5.0])},
        ),
        (
            np.array([0, 1, 2, 3], dtype=np.uint32),
            {
                "bins": np.array([0, 2, 3], dtype=np.uint32),
                "weights": np.array([1, 2, 3, 4], dtype=np.int32),
            },
        ),
        (
            np.array([0.0, 0.5, 1.0, 2.0]),
            {
                "bins": np.array([0.0, 1.0, 2.0]),
                "weights": np.array(
                    [-0.25, 1.5, 2.25, 0.75], dtype=np.float32
                ),
            },
        ),
        (
            np.array([False, False, True, True], dtype=np.bool_),
            {"bins": 2, "density": True},
        ),
        (
            np.array([0.0, 0.5, 1.0, 2.0]),
            {
                "bins": np.array([0.0, 0.5, 2.0]),
                "weights": np.array([1.0, 2.0, 3.0, 4.0]),
                "density": True,
            },
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = [np.histogram(values, **options) for values, options in cases]
        with scheme.TorchScheme("cpu"):
            actual = [
                np.histogram(Array(values), **options)
                for values, options in cases
            ]

    for result, reference in zip(actual, expected):
        assert isinstance(result, tuple)
        assert all(type(value) is Array for value in result)
        assert result[0].dtype == reference[0].dtype
        assert result[1].dtype == reference[1].dtype
        np.testing.assert_allclose(
            result[0].numpy(),
            reference[0],
            rtol=2e-5,
            atol=1e-7,
            equal_nan=True,
        )
        np.testing.assert_array_equal(result[1].numpy(), reference[1])


@pytest.mark.parametrize(
    "values, options",
    (
        (np.array([1.0, 2.0]), {"bins": 0}),
        (np.array([1.0, 2.0]), {"bins": 2.5}),
        (np.array([1.0, 2.0]), {"bins": [0.0, 2.0, 1.0]}),
        (np.array([1.0, 2.0]), {"bins": 2, "range": (2.0, 1.0)}),
        (np.array([1.0, 2.0]), {"bins": 2, "range": (0.0, np.inf)}),
        (np.array([np.nan]), {"bins": 2}),
        (
            np.array([1.0, 2.0]),
            {"bins": 2, "weights": np.array([1.0])},
        ),
        (np.array([1.0, 2.0]), {"bins": [[0.0, 1.0], [2.0, 3.0]]}),
    ),
)
def test_numpy_histogram_errors_match_numpy_contract(values, options):
    with pytest.raises(Exception) as numpy_error:
        np.histogram(values, **options)

    with scheme.TorchScheme("cpu"):
        with pytest.raises(type(numpy_error.value)):
            np.histogram(Array(values), **options)


def test_numpy_histogram_fallback_contract(monkeypatch):
    complex_values = np.array([1 + 2j, 2 - 1j, 3 + 0j])
    real_values = np.array([1.0, 2.0, 3.0, 4.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        complex_reference = np.histogram(complex_values, bins=3)
        estimator_reference = np.histogram(real_values, bins="auto")
        backend_reference = np.histogram(real_values, bins=2)
        with scheme.TorchScheme("cpu"):
            complex_result = np.histogram(Array(complex_values), bins=3)
            estimator_result = np.histogram(
                Array(real_values), bins="auto"
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData,
                    "_numpy_histogram",
                    lambda self, args, kwargs: NotImplemented,
                )
                backend_fallback = np.histogram(
                    Array(real_values), bins=2
                )

    for result, reference in (
            (complex_result, complex_reference),
            (estimator_result, estimator_reference),
            (backend_fallback, backend_reference)):
        assert isinstance(result, tuple)
        assert all(isinstance(value, np.ndarray) for value in result)
        np.testing.assert_array_equal(result[0], reference[0])
        np.testing.assert_array_equal(result[1], reference[1])


@pytest.mark.parametrize("function", (np.setdiff1d, np.setxor1d))
def test_numpy_set_difference_functions_stay_on_torch_device(
        torch_device_ctx, monkeypatch, function):
    ctx, device = torch_device_ctx
    left = np.array([[7, 1, 3], [1, 5, 9]], dtype=np.int64)
    right = np.array([3, 8, 1], dtype=np.int32)
    expected = function(left, right)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError(f"NumPy {function.__name__} copied to host")

    with ctx:
        values = Array(left)
        other = Array(right)
        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = function(ar1=values, ar2=other)
            mixed = function(values, right)

    for result in (actual, mixed):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
        np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("function", (np.setdiff1d, np.setxor1d))
def test_numpy_set_difference_functions_preserve_autograd(
        torch_device_ctx, function):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        left = torch.tensor(
            [5.0, 1.0, 3.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        right = torch.tensor(
            [4.0, 3.0, 1.0], dtype=dtype, device=device,
            requires_grad=True,
        )
        result = function(
            Array(TorchArrayData(left), copy=False),
            Array(TorchArrayData(right), copy=False),
        )
        result._data.tensor.sum().backward()

    assert result._data.tensor.device.type == device
    total_gradient = left.grad.sum()
    if right.grad is not None:
        total_gradient += right.grad.sum()
    torch.testing.assert_close(
        total_gradient,
        torch.tensor(
            1.0 if function is np.setdiff1d else 2.0,
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize("function", (np.setdiff1d, np.setxor1d))
def test_numpy_set_difference_functions_match_numpy_contract(function):
    cases = (
        (
            np.array([[9, 1, 5], [3, 1, 7]], dtype=np.int32),
            np.array([3, 8, 1], dtype=np.float32),
            False,
        ),
        (
            np.array([0, 2**31, 2**32 - 1], dtype=np.uint32),
            np.array([-1, 2**31], dtype=np.int64),
            False,
        ),
        (
            np.array([False, True, False], dtype=np.bool_),
            np.array([1, 2], dtype=np.uint32),
            False,
        ),
        (
            np.array([0.0, -0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
            False,
        ),
        (
            np.array([np.nan, 2.0, np.nan, 1.0]),
            np.array([np.nan, 3.0, 1.0]),
            False,
        ),
        (
            np.array([9, 1, 5, 3], dtype=np.int64),
            np.array([3, 8, 1], dtype=np.int64),
            True,
        ),
        (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float64),
            False,
        ),
    )

    with scheme.TorchScheme("cpu"):
        results = [
            function(
                Array(left), Array(right), assume_unique=assume_unique
            )
            for left, right, assume_unique in cases
        ]

    for (left, right, assume_unique), result in zip(cases, results):
        expected = function(left, right, assume_unique=assume_unique)
        assert type(result) is Array
        assert result.dtype == expected.dtype
        actual = result.numpy()
        np.testing.assert_array_equal(actual, expected)
        if expected.dtype.kind == "f":
            zero = expected == 0
            np.testing.assert_array_equal(
                np.signbit(actual[zero]),
                np.signbit(expected[zero]),
            )


@pytest.mark.parametrize("function", (np.setdiff1d, np.setxor1d))
def test_numpy_set_difference_functions_fallback_contract(
        monkeypatch, function):
    left = np.array([1 + 2j, 3 - 1j, 1 + 2j])
    right = np.array([4 + 0j, 1 + 2j])
    method = f"_numpy_{function.__name__}"

    with scheme.TorchScheme("cpu"):
        values = Array(left)
        complex_result = function(values, right)
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                method,
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = function(
                Array(np.array([1, 2, 3])), [2, 4]
            )

    assert isinstance(complex_result, np.ndarray)
    np.testing.assert_array_equal(complex_result, function(left, right))
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_array_equal(
        backend_fallback, function(np.array([1, 2, 3]), [2, 4])
    )


def test_numpy_isin_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    source = np.array(
        [[1, 2, 3, 2], [5, 8, 13, 21]], dtype=np.int64
    )
    tests = np.array([2, 5, 21], dtype=np.int32)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("NumPy isin copied Torch data to the host")

    with ctx:
        values = Array(source)
        test_values = Array(tests)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            included = np.isin(values, test_values, kind="sort")
            excluded = np.isin(
                element=values,
                test_elements=[1, 8, 13],
                assume_unique=True,
                invert=True,
            )
            locations = np.argwhere(np.isin(values, test_values))

    for result in (included, excluded, locations):
        assert type(result) is Array
        assert result._data.tensor.device.type == device
    assert included.dtype == np.dtype(np.bool_)
    assert excluded.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(
        included.numpy(), np.isin(source, tests, kind="sort")
    )
    np.testing.assert_array_equal(
        excluded.numpy(),
        np.isin(source, [1, 8, 13], assume_unique=True, invert=True),
    )
    np.testing.assert_array_equal(
        locations.numpy(), np.argwhere(np.isin(source, tests))
    )


def test_numpy_isin_dtype_and_kind_semantics():
    cases = (
        (
            np.array([False, True, False], dtype=np.bool_),
            np.array([True], dtype=np.bool_),
            "table",
        ),
        (
            np.array([0, 2**31, 2**32 - 1], dtype=np.uint32),
            np.array([2**31, -1], dtype=np.int64),
            None,
        ),
        (
            np.array([1, 2**53 + 1], dtype=np.int64),
            np.array([1.0, float(2**53 + 1)], dtype=np.float64),
            "sort",
        ),
        (
            np.array([1 + 2j, 3 - 4j, complex(np.nan, 0)]),
            np.array([3 - 4j, complex(np.nan, 0)]),
            None,
        ),
        (
            np.array([], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            None,
        ),
        (
            np.array([[1.0, np.nan], [-0.0, np.inf]], dtype=np.float64),
            np.array([0.0, np.nan, np.inf], dtype=np.float64),
            None,
        ),
    )

    with scheme.TorchScheme("cpu"):
        results = [
            np.isin(Array(source), Array(tests), kind=kind)
            for source, tests, kind in cases
        ]

    for (source, tests, kind), result in zip(cases, results):
        assert type(result) is Array
        assert result.dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(
            result.numpy(), np.isin(source, tests, kind=kind)
        )


def test_numpy_isin_errors_and_fallback_contract(monkeypatch):
    source = np.array([1, 2, 3], dtype=np.int32)
    with scheme.TorchScheme("cpu"):
        values = Array(source)
        with pytest.raises(ValueError, match="Invalid kind"):
            np.isin(values, [1, 2], kind="invalid")
        with pytest.raises(TypeError, match="unhashable type"):
            np.isin(values, [1, 2], kind=[])
        with pytest.raises(ValueError, match="only supported"):
            np.isin(values, [1.0, 2.0], kind="table")
        with pytest.raises(RuntimeError, match="range of values"):
            np.isin(
                values,
                np.array(
                    [np.iinfo(np.int32).min, np.iinfo(np.int32).max],
                    dtype=np.int32,
                ),
                kind="table",
            )
        with pytest.raises(ValueError, match="Maximum allowed dimension"):
            np.isin(
                values,
                np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64),
                kind="table",
            )

        object_result = np.isin(values, {1, 2})
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "_numpy_isin",
                lambda self, args, kwargs: NotImplemented,
            )
            backend_fallback = np.isin(values, [2, 4])

    assert isinstance(object_result, np.ndarray)
    assert isinstance(backend_fallback, np.ndarray)
    np.testing.assert_array_equal(object_result, np.isin(source, {1, 2}))
    np.testing.assert_array_equal(
        backend_fallback, np.isin(source, [2, 4])
    )


def test_numpy_join_dtype_out_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        first = Array(np.array([1.0, 2.0], dtype=np.float32))
        second = Array(np.array([3.0, 4.0], dtype=np.float64))

        promoted = np.concatenate((first, second))
        typed = np.stack((first, first), dtype=np.float64)
        output = Array(np.empty(4, dtype=np.float32))
        returned = np.concatenate((first, second), out=output)
        reshaped = np.reshape(first, (2, 1))
        norm = np.linalg.norm(first)
        mixed = np.concatenate((first, np.array([5.0])))
        unsupported = np.concatenate((first, first), dtype=np.float16)

        assert type(promoted) is Array
        assert promoted.dtype == np.dtype(np.float64)
        assert type(typed) is Array
        assert typed.dtype == np.dtype(np.float64)
        assert returned is output
        torch.testing.assert_close(
            output._data.tensor,
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        )
        assert type(reshaped) is Array
        assert np.isclose(norm, np.sqrt(5.0))
        assert isinstance(mixed, np.ndarray)
        assert isinstance(unsupported, np.ndarray)
        assert unsupported.dtype == np.dtype(np.float16)

        with pytest.raises(ValueError):
            np.concatenate((first, Array(np.ones((1, 1)))))
        with pytest.raises(TypeError, match="Cannot cast"):
            np.concatenate(
                (first, second), dtype=np.float32, casting="safe"
            )


def test_numpy_where_stays_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        condition = Array(np.array([[True], [False]]))
        value_true = Array(
            np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        )
        value_false = Array(
            np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype)
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy where copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            selected = np.where(condition, value_true, value_false)
            scalar_selected = np.where(condition, value_true, -1.0)

        assert type(selected) is Array
        assert selected._data.tensor.device.type == device
        torch.testing.assert_close(
            selected._data.tensor,
            torch.tensor(
                [[1, 2, 3], [10, 11, 12]],
                device=device,
                dtype=torch_dtype,
            ),
        )
        assert type(scalar_selected) is Array
        assert scalar_selected._data.tensor.device.type == device
        torch.testing.assert_close(
            scalar_selected._data.tensor,
            torch.tensor(
                [[1, 2, 3], [-1, -1, -1]],
                device=device,
                dtype=torch_dtype,
            ),
        )


def test_numpy_where_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source_true = torch.tensor(
            [1.0, 2.0, 3.0], device=device, dtype=dtype,
            requires_grad=True,
        )
        source_false = torch.tensor(
            [4.0, 5.0, 6.0], device=device, dtype=dtype,
            requires_grad=True,
        )
        condition = Array(np.array([True, False, True]))
        value_true = Array(TorchArrayData(source_true), copy=False)
        value_false = Array(TorchArrayData(source_false), copy=False)

        selected = np.where(condition, value_true, value_false)
        selected._data.tensor.sum().backward()

        torch.testing.assert_close(
            source_true.grad,
            torch.tensor([1, 0, 1], device=device, dtype=dtype),
        )
        torch.testing.assert_close(
            source_false.grad,
            torch.tensor([0, 1, 0], device=device, dtype=dtype),
        )


def test_numpy_where_dtype_and_fallback_contract():
    with scheme.TorchScheme("cpu"):
        condition = Array(np.array([True, False]))
        value_float = Array(np.array([1.0, 2.0], dtype=np.float32))
        value_complex = Array(np.array([3.0j, 4.0j], dtype=np.complex64))
        value_unsigned = Array(np.array([1, 2], dtype=np.uint32))

        promoted = np.where(condition, value_float, value_complex)
        scalar = np.where(condition, value_float, 1.5)
        unsigned = np.where(condition, value_unsigned, 0)
        scalar_condition = np.where(True, value_float, value_complex)
        indices = np.where(condition)
        mixed = np.where(condition, value_float, np.array([3.0, 4.0]))

        assert type(promoted) is Array
        assert promoted.dtype == np.dtype(np.complex64)
        assert type(scalar) is Array
        assert scalar.dtype == np.dtype(np.float32)
        assert type(unsigned) is Array
        assert unsigned.dtype == np.dtype(np.uint32)
        assert type(scalar_condition) is Array
        assert scalar_condition.dtype == np.dtype(np.complex64)
        assert isinstance(indices, tuple)
        assert all(isinstance(index, np.ndarray) for index in indices)
        assert isinstance(mixed, np.ndarray)

        with pytest.raises(ValueError):
            np.where(
                Array(np.ones(2, dtype=bool)),
                Array(np.ones(3)),
                Array(np.zeros(3)),
            )


def test_numpy_repeat_keeps_torch_arrays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        values = TimeSeries(
            np.arange(4, dtype=dtype), delta_t=0.25, epoch=12.0
        )
        repeat_tensor = torch.tensor(
            [1, 2, 0, 3], device=device, dtype=torch.int64
        )
        repeats = Array(TorchArrayData(repeat_tensor), copy=False)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.repeat copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            scalar = np.repeat(values, 2)
            elementwise = np.repeat(values, repeats, axis=0)
            negative_axis = np.repeat(values, repeats, axis=-1)

        expected_scalar = torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            device=device,
            dtype=torch_dtype,
        )
        expected_elementwise = torch.tensor(
            [0.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            device=device,
            dtype=torch_dtype,
        )
        for actual, expected in (
            (scalar, expected_scalar),
            (elementwise, expected_elementwise),
            (negative_axis, expected_elementwise),
        ):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(actual._data.tensor, expected)


def test_numpy_repeat_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [1.0, 2.0, 3.0], device=device, dtype=dtype, requires_grad=True
        )
        values = Array(TorchArrayData(source), copy=False)
        repeated = np.repeat(values, [1, 2, 3])
        repeated._data.tensor.sum().backward()

        assert repeated._data.tensor.device.type == device
        torch.testing.assert_close(
            source.grad,
            torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype),
        )


def test_numpy_tile_keeps_torch_arrays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    real_dtype = np.float32 if device == "mps" else np.float64
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    source_values = np.arange(6, dtype=real_dtype).reshape(2, 3)
    complex_values = source_values.astype(complex_dtype) * (1.0 + 2.0j)

    with ctx:
        values = Array(source_values)
        complex_source = Array(complex_values)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.tile copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            results = (
                np.tile(values, 2),
                np.tile(A=values, reps=(2, 1)),
                np.tile(values, (2, 1, 3)),
                np.tile(values, [2, 0]),
                np.tile(values, ()),
                np.tile(complex_source, np.array([1, 2])),
            )

        expected = (
            np.tile(source_values, 2),
            np.tile(source_values, (2, 1)),
            np.tile(source_values, (2, 1, 3)),
            np.tile(source_values, [2, 0]),
            np.tile(source_values, ()),
            np.tile(complex_values, np.array([1, 2])),
        )
        for actual, target in zip(results, expected):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor,
                torch.as_tensor(target, device=device),
            )

        copied = results[4]
        assert copied._data.tensor.data_ptr() != values._data.tensor.data_ptr()
        copied._data.tensor[0, 0] = -1
        assert values._data.tensor[0, 0].item() == 0


def test_numpy_tile_preserves_autograd_and_unsigned(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.arange(
            6, device=device, dtype=dtype
        ).reshape(2, 3).requires_grad_()
        values = Array(TorchArrayData(source), copy=False)
        tiled = np.tile(values, (2, 1, 3))
        tiled._data.tensor.sum().backward()

        assert tiled._data.tensor.device.type == device
        torch.testing.assert_close(
            source.grad,
            torch.full_like(source, 6.0),
        )

        if device != "mps":
            unsigned_values = np.arange(6, dtype=np.uint32).reshape(2, 3)
            unsigned = np.tile(Array(unsigned_values), (2, 1))
            assert unsigned.dtype == np.dtype(np.uint32)
            assert unsigned._data.tensor.device.type == device
            torch.testing.assert_close(
                unsigned._data.tensor,
                torch.as_tensor(unsigned_values, device=device).repeat(2, 1),
            )


def test_numpy_tile_repetition_contract():
    source_values = np.arange(6).reshape(2, 3)
    with scheme.TorchScheme("cpu"):
        values = Array(source_values)

        for reps in (np.int64(2), True, False, np.array(2)):
            actual = np.tile(values, reps)
            np.testing.assert_array_equal(
                actual.numpy(), np.tile(source_values, reps)
            )

        with pytest.raises(ValueError, match="negative dimensions"):
            np.tile(values, (2, -1))
        with pytest.raises(TypeError):
            np.tile(values, 2.0)
        with pytest.raises(TypeError):
            np.tile(values, "2")


def test_numpy_append_and_resize_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(6, dtype=dtype).reshape(2, 3)
    extension = np.array([[6, 7, 8]], dtype=dtype)
    expected = (
        np.append(source, extension, axis=0),
        np.append(source, extension.reshape(-1)),
        np.resize(source, (3, 5)),
        np.resize(source, (2, 2)),
        np.resize(source, ()),
        np.resize(np.array([], dtype=dtype), (2, 3)),
        np.resize(source, (0, 3)),
    )

    with ctx:
        values = Array(source)
        appended_values = Array(extension)
        empty_values = Array(np.array([], dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("append/resize copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = (
                np.append(values, appended_values, axis=0),
                np.append(arr=values, values=extension.reshape(-1)),
                np.resize(values, (3, 5)),
                np.resize(a=values, new_shape=(2, 2)),
                np.resize(values, ()),
                np.resize(empty_values, (2, 3)),
                np.resize(values, (0, 3)),
            )
            copied = np.resize(values, values.shape)

        for result, target in zip(actual, expected):
            assert type(result) is Array
            assert result._data.tensor.device.type == device
            torch.testing.assert_close(
                result._data.tensor,
                torch.tensor(np.array(target, copy=True), device=device),
            )
        assert copied._data.tensor.data_ptr() != values._data.tensor.data_ptr()


def test_numpy_append_and_resize_preserve_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        left = torch.arange(
            3, device=device, dtype=dtype, requires_grad=True
        )
        right = torch.arange(
            3, 5, device=device, dtype=dtype, requires_grad=True
        )
        values = Array(TorchArrayData(left), copy=False)
        extension = Array(TorchArrayData(right), copy=False)
        appended = np.append(values, extension)
        appended._data.tensor.sum().backward()

        torch.testing.assert_close(left.grad, torch.ones_like(left))
        torch.testing.assert_close(right.grad, torch.ones_like(right))

        left.grad = None
        resized = np.resize(values, 7)
        resized._data.tensor.sum().backward()
        assert resized._data.tensor.device.type == device
        torch.testing.assert_close(
            left.grad,
            torch.tensor([3.0, 2.0, 2.0], device=device, dtype=dtype),
        )


def test_numpy_append_and_resize_match_numpy_contract():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    extension = np.array([7, 8], dtype=np.int64)

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        appended = np.append(values, extension)
        expected_append = np.append(source, extension)
        assert appended.dtype == expected_append.dtype
        np.testing.assert_array_equal(appended.numpy(), expected_append)

        for shape in (np.int64(7), [2, 4], np.array([2, 4]), (), (0, 3)):
            actual = np.resize(values, shape)
            expected = np.resize(source, shape)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual.numpy(), expected)

        with pytest.raises(ValueError):
            np.append(values, np.ones((2, 2), dtype=np.float32), axis=0)
        with pytest.raises(np.exceptions.AxisError):
            np.append(values, source, axis=2)
        with pytest.raises(TypeError, match="integer is required"):
            np.append(values, source, axis=True)
        with pytest.raises(ValueError, match="non-negative"):
            np.resize(values, (2, -1))
        with pytest.raises(TypeError):
            np.resize(values, 2.0)
        with pytest.raises(TypeError, match="integer is required"):
            np.resize(values, True)
        with pytest.raises(TypeError):
            np.resize(values, np.array(2))


def test_numpy_putmask_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(12, dtype=dtype).reshape(3, 4)
    mask = (source % 3) != 1
    replacements = np.array([10, 20, 30, 40, 50], dtype=dtype)
    expected = source.copy()
    np.putmask(expected, mask, replacements)

    with ctx:
        values = Array(source)
        mask_values = Array(mask)
        replacement_values = Array(replacements)
        pointer = values._data.tensor.data_ptr()

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("putmask copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            result = np.putmask(
                values, mask_values, replacement_values
            )

        assert result is None
        assert values._data.tensor.data_ptr() == pointer
        assert values._data.tensor.device.type == device
        torch.testing.assert_close(
            values._data.tensor,
            torch.tensor(expected, dtype=values._data.tensor.dtype,
                         device=device),
        )


def test_numpy_putmask_matches_numpy_contract():
    mask = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    dtypes = (
        np.bool_, np.float32, np.float64, np.complex64,
        np.complex128, np.int32, np.uint32,
    )

    with scheme.TorchScheme("cpu"):
        for dtype in dtypes:
            source = np.arange(6).reshape(2, 3).astype(dtype)
            replacements = np.array([10, 20, 30], dtype=dtype)
            expected = source.copy()
            np.putmask(expected, mask, replacements)

            actual = Array(source)
            assert np.putmask(
                actual, Array(mask), Array(replacements)
            ) is None
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual.numpy(), expected)

        source = np.arange(12, dtype=np.float32).reshape(3, 4)
        expected = source.copy()
        expected_view = expected[:, ::2]
        replacement = expected_view.reshape(-1)[:2].copy()
        np.putmask(
            expected_view,
            np.array([[True, False], [True, False], [True, False]]),
            replacement,
        )

        actual = Array(source)
        actual_view = Array(actual[:, ::2], copy=False)
        np.putmask(
            actual_view,
            [[True, False], [True, False], [True, False]],
            actual_view[:1],
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

        unchanged = Array(np.arange(3, dtype=np.float32))
        np.putmask(unchanged, [True, False, True], [])
        np.testing.assert_array_equal(
            unchanged.numpy(), np.arange(3, dtype=np.float32)
        )


def test_numpy_putmask_matches_numpy_errors():
    with scheme.TorchScheme("cpu"):
        values = Array(np.zeros(3, dtype=np.float32))
        with pytest.raises(ValueError, match="same size"):
            np.putmask(values, [True, False], 1)
        with pytest.raises(TypeError, match="Cannot cast array data"):
            np.putmask(
                values,
                [True, False, True],
                np.array([2.5], dtype=np.float64),
            )
        with pytest.raises(TypeError, match="Cannot cast array data"):
            np.putmask(
                values,
                [True, False, True],
                Array(np.array([2.5], dtype=np.float64)),
            )


def test_numpy_delete_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    expected = (
        np.delete(source, [0, -1, 2], axis=1),
        np.delete(source, [True, False, True, False], axis=0),
        np.delete(source, slice(None, None, -2), axis=-1),
        np.delete(source, [0, 5, 10]),
        np.delete(source, [], axis=0),
    )

    with ctx:
        values = Array(source)
        integer_indices = Array(np.array([0, -1, 2], dtype=np.int64))
        boolean_indices = Array(
            np.array([True, False, True, False], dtype=np.bool_)
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("delete copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = (
                np.delete(values, integer_indices, axis=1),
                np.delete(values, boolean_indices, axis=0),
                np.delete(values, slice(None, None, -2), axis=-1),
                np.delete(arr=values, obj=[0, 5, 10]),
                np.delete(values, [], axis=0),
            )

        for result, target in zip(actual, expected):
            assert type(result) is Array
            assert result.dtype == target.dtype
            assert result._data.tensor.device.type == device
            torch.testing.assert_close(
                result._data.tensor,
                torch.tensor(np.array(target, copy=True), device=device),
            )


def test_numpy_delete_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.arange(
            6, dtype=dtype, device=device, requires_grad=True
        )
        values = Array(TorchArrayData(source), copy=False)
        result = np.delete(values, [1, 3])
        result._data.tensor.sum().backward()

        assert result._data.tensor.device.type == device
        torch.testing.assert_close(
            source.grad,
            torch.tensor(
                [1, 0, 1, 0, 1, 1], dtype=dtype, device=device
            ),
        )


def test_numpy_delete_matches_numpy_contract():
    source = np.arange(24, dtype=np.int64).reshape(4, 6)
    cases = (
        (1, None),
        (-1, 1),
        (np.array(1), 0),
        ([0, 2, 2], 1),
        (np.array([[0, -1], [1, 1]], dtype=np.int64), 1),
        (np.array([True, False, True, False]), 0),
        (slice(None, None, -2), -1),
        (slice(2, 2), 0),
        ([], 0),
    )

    with scheme.TorchScheme("cpu"):
        values = Array(source)
        for obj, axis in cases:
            actual = np.delete(values, obj, axis=axis)
            expected = np.delete(source, obj, axis=axis)
            assert type(actual) is Array
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual.numpy(), expected)

        array_indices = Array(np.array([0, -1, 2], dtype=np.int64))
        actual = np.delete(values, array_indices, axis=1)
        expected = np.delete(source, [0, -1, 2], axis=1)
        np.testing.assert_array_equal(actual.numpy(), expected)

        scalar = Array(np.array(7, dtype=np.uint32))
        deleted_scalar = np.delete(scalar, 0)
        assert type(deleted_scalar) is Array
        assert isinstance(deleted_scalar._data, TorchArrayData)
        assert deleted_scalar.dtype == np.dtype(np.uint32)
        assert deleted_scalar.shape == (0,)

        series = TimeSeries(np.arange(6, dtype=np.float32), delta_t=0.25)
        assert type(np.delete(series, 1)) is Array


def test_numpy_delete_matches_numpy_errors():
    values = np.arange(6)

    with scheme.TorchScheme("cpu"):
        array = Array(values)
        for obj in (True, np.bool_(False), np.array([True, False])):
            with pytest.raises(ValueError, match="boolean array argument"):
                np.delete(array, obj)

        for obj in (6, -7, [0, 6], np.array([-7])):
            with pytest.raises(IndexError, match="out of bounds"):
                np.delete(array, obj)

        with pytest.raises(np.exceptions.AxisError):
            np.delete(array, 0, axis=2)
        for obj in (1.0, [1.0], np.array([], dtype=np.float64), "1"):
            with pytest.raises(IndexError):
                np.delete(array, obj)


def test_numpy_complex_components_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    complex_source = np.array(
        [-1.0 + 0.0j, 1.0 + 2.0j, 0.0 - 3.0j],
        dtype=complex_dtype,
    )
    real_source = np.array([-2.0, 0.0, 3.0], dtype=real_dtype)
    expected = (
        np.real(complex_source),
        np.imag(complex_source),
        np.angle(complex_source),
        np.angle(complex_source, deg=True),
        np.real(real_source),
        np.imag(real_source),
        np.angle(real_source),
    )

    with ctx:
        complex_values = Array(complex_source)
        real_values = Array(real_source)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("complex-component extraction left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = (
                np.real(complex_values),
                np.imag(val=complex_values),
                np.angle(complex_values),
                np.angle(z=complex_values, deg=True),
                np.real(real_values),
                np.imag(real_values),
                np.angle(real_values),
            )

        for result, target in zip(actual, expected):
            assert type(result) is Array
            assert result._data.tensor.device.type == device
            torch.testing.assert_close(
                result._data.tensor,
                torch.tensor(np.array(target, copy=True), device=device),
            )
        assert (
            actual[4]._data.tensor.data_ptr()
            == real_values._data.tensor.data_ptr()
        )


def test_numpy_complex_components_preserve_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.complex64 if device == "mps" else torch.complex128

    with ctx:
        source = torch.tensor(
            [1.0 + 2.0j, -3.0 + 4.0j],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        values = Array(TorchArrayData(source), copy=False)
        components = np.real(values) + np.imag(values)
        components._data.tensor.sum().backward()
        torch.testing.assert_close(
            source.grad,
            torch.full_like(source, 1.0 + 1.0j),
        )

        phase_source = source.detach().clone().requires_grad_()
        phase_values = Array(TorchArrayData(phase_source), copy=False)
        np.angle(phase_values)._data.tensor.sum().backward()
        expected_gradient = torch.tensor(
            [-0.4 + 0.2j, -0.16 - 0.12j],
            device=device,
            dtype=dtype,
        )
        torch.testing.assert_close(phase_source.grad, expected_gradient)


def test_numpy_angle_matches_numpy_dtype_contract():
    with scheme.TorchScheme("cpu"):
        for dtype in (
                np.bool_, np.int32, np.int64, np.uint32,
                np.float32, np.float64, np.complex64, np.complex128):
            source = np.array([0, 1, 2], dtype=dtype)
            values = Array(source)
            for degrees in (False, True):
                actual = np.angle(values, deg=degrees)
                expected = np.angle(source, deg=degrees)
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual.numpy(), expected)


def test_numpy_unwrap_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    radians = np.array(
        [[0.0, 2.5, -2.5, -0.25], [0.5, -2.75, 2.75, 0.75]],
        dtype=dtype,
    )
    degrees = np.rad2deg(radians).astype(dtype)
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    complex_phase = np.exp(1.0j * radians).astype(complex_dtype)
    expected = (
        np.unwrap(radians),
        np.unwrap(radians, discont=4.0, axis=1),
        np.unwrap(radians, axis=0),
        np.unwrap(degrees, period=360.0),
        np.unwrap(np.angle(complex_phase)),
        np.unwrap(np.array([], dtype=dtype)),
        np.unwrap(np.array([1.25], dtype=dtype)),
    )

    with ctx:
        values = Array(radians)
        degree_values = Array(degrees)
        complex_values = Array(complex_phase)
        empty = Array(np.array([], dtype=dtype))
        singleton = Array(np.array([1.25], dtype=dtype))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("phase unwrapping left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            actual = (
                np.unwrap(values),
                np.unwrap(values, 4.0, 1),
                np.unwrap(p=values, axis=0),
                np.unwrap(degree_values, period=360.0),
                np.unwrap(np.angle(complex_values)),
                np.unwrap(empty),
                np.unwrap(singleton),
            )

        for result, target in zip(actual, expected):
            assert type(result) is Array
            assert result._data.tensor.device.type == device
            torch.testing.assert_close(
                result._data.tensor,
                torch.tensor(np.array(target, copy=True), device=device),
            )
        assert (
            actual[-1]._data.tensor.data_ptr()
            != singleton._data.tensor.data_ptr()
        )


def test_numpy_unwrap_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [0.0, 2.5, -2.5, -0.25],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        values = Array(TorchArrayData(source), copy=False)
        result = np.unwrap(values)
        result._data.tensor.sum().backward()

    torch.testing.assert_close(source.grad, torch.ones_like(source))


def test_numpy_unwrap_matches_numpy_float_contract():
    with scheme.TorchScheme("cpu"):
        for dtype in (np.float32, np.float64):
            source = np.array(
                [[0.0, 2.5, -2.5], [0.25, -2.75, 2.75]],
                dtype=dtype,
            )
            values = Array(source)
            for axis, period in ((0, 2 * np.pi), (1, 4.0)):
                actual = np.unwrap(values, axis=axis, period=period)
                expected = np.unwrap(source, axis=axis, period=period)
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual.numpy(), expected)

        values = Array(np.arange(6, dtype=np.float32).reshape(2, 3))
        with pytest.raises(np.exceptions.AxisError):
            np.unwrap(values, axis=2)
        scalar = Array(
            TorchArrayData(torch.tensor(1.0, dtype=torch.float64)),
            copy=False,
        )
        with pytest.raises(ValueError, match="at least one dimensional"):
            np.unwrap(scalar)


def test_numpy_like_creators_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.arange(
            24, device=device, dtype=dtype
        ).reshape(2, 3, 4).permute(2, 0, 1)
        values = Array(TorchArrayData(source), copy=False)
        fill = Array(np.array([2.0, 3.0, 4.0], dtype=np.float32))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("NumPy like-creator copied its prototype")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            empty = np.empty_like(prototype=values)
            zeros = np.zeros_like(
                a=values, dtype=np.float32, order="F", shape=(5, 2, 3)
            )
            ones = np.ones_like(values, order="C", subok=False)
            full = np.full_like(values, fill)
            cpu_device = (
                np.zeros_like(values, device="cpu")
                if device == "cpu" else None
            )

        for actual in (empty, zeros, ones, full):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
        assert empty._data.tensor.stride() == source.stride()
        assert zeros._data.tensor.stride() == (1, 5, 10)
        assert ones._data.tensor.stride() == (6, 3, 1)
        torch.testing.assert_close(
            zeros._data.tensor,
            torch.zeros((5, 2, 3), device=device, dtype=torch.float32),
        )
        torch.testing.assert_close(ones._data.tensor, torch.ones_like(source))
        torch.testing.assert_close(
            full._data.tensor,
            torch.tensor(
                [2.0, 3.0, 4.0], device=device, dtype=dtype
            ).expand(4, 2, 3),
        )
        if cpu_device is not None:
            assert cpu_device._data.tensor.device.type == "cpu"


def test_numpy_like_creators_match_numpy_shape_dtype_and_order():
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4).transpose(2, 0, 1)
    with scheme.TorchScheme("cpu"):
        values = Array(
            np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        ).transpose(2, 0, 1)

        for function in (np.empty_like, np.zeros_like, np.ones_like):
            for order in ("C", "F", "A", "K", None, b"C"):
                for shape in (None, (5, 2, 3), (5, 1, 7), [2, 3], ()):
                    actual = function(values, order=order, shape=shape)
                    expected = function(source, order=order, shape=shape)
                    assert actual.shape == expected.shape
                    assert actual.dtype == expected.dtype
                    assert actual._data.tensor.stride() == tuple(
                        stride // expected.itemsize
                        for stride in expected.strides
                    )
                    if function is not np.empty_like:
                        np.testing.assert_array_equal(
                            actual.numpy(), expected
                        )

        for dtype in (
                np.bool_, np.float32, np.float64,
                np.complex64, np.complex128, np.uint32, np.int32, np.int64):
            actual = np.full_like(values, [1, 2, 3], dtype=dtype)
            expected = np.full_like(source, [1, 2, 3], dtype=dtype)
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual.numpy(), expected)

        with pytest.raises(ValueError, match="negative dimensions"):
            np.zeros_like(values, shape=(2, -1))
        with pytest.raises(TypeError):
            np.zeros_like(values, shape=(2.0, 3))
        with pytest.raises(TypeError):
            np.zeros_like(values, shape=True)
        with pytest.raises(ValueError, match="order must be one of"):
            np.zeros_like(values, order="Z")
        with pytest.raises(TypeError):
            np.zeros_like(values, order=1)
        with pytest.raises(TypeError):
            np.zeros_like(values, subok="yes")
        with pytest.raises(ValueError, match="Only.*cpu.*allowed"):
            np.zeros_like(values, device="cuda")

        unsupported = np.ones_like(values, dtype=np.float16)
        assert type(unsupported) is np.ndarray
        assert unsupported.dtype == np.dtype(np.float16)

        itemsize = np.dtype(np.float64).itemsize
        for shape, strides in (
                ((1, 4, 3), (12, 1, 4)),
                ((4, 3), (0, 1)),
                ((4, 1, 3), (0, 0, 1)),
                ((2, 0, 4), (4, 4, 1))):
            tensor = torch.empty_strided(
                shape, strides, dtype=torch.float64
            )
            prototype = Array(TorchArrayData(tensor), copy=False)
            numpy_prototype = np.lib.stride_tricks.as_strided(
                np.empty(100, dtype=np.float64),
                shape=shape,
                strides=tuple(stride * itemsize for stride in strides),
            )
            actual = np.empty_like(prototype, order="K")
            expected = np.empty_like(numpy_prototype, order="K")
            assert actual._data.tensor.stride() == tuple(
                stride // itemsize for stride in expected.strides
            )


def test_numpy_multidimensional_repeat_and_ravel_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    source_values = np.arange(24, dtype=dtype).reshape(2, 3, 4)

    with ctx:
        source = torch.tensor(
            source_values, device=device, requires_grad=True
        )
        values = Array(TorchArrayData(source), copy=False)
        repeats = Array(np.array([1, 2, 0], dtype=np.int64))

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("shape/index operation copied data to host")

        with monkeypatch.context() as patch:
            patch.setattr(Array, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            repeated = np.repeat(values, repeats, axis=-2)
            scalar_repeated = np.repeat(values, 2, axis=-1)
            raveled = np.ravel(values)

        expected_repeated = np.repeat(source_values, [1, 2, 0], axis=-2)
        expected_scalar = np.repeat(source_values, 2, axis=-1)
        for actual, expected in (
                (repeated, expected_repeated),
                (scalar_repeated, expected_scalar),
                (raveled, source_values.ravel())):
            assert type(actual) is Array
            assert actual._data.tensor.device.type == device
            torch.testing.assert_close(
                actual._data.tensor,
                torch.as_tensor(expected, device=device),
            )

        (
            repeated._data.tensor.sum()
            + raveled._data.tensor.sum()
        ).backward()
        expected_gradient = np.ones(source_values.shape, dtype=dtype)
        expected_gradient[:, 0, :] += 1
        expected_gradient[:, 1, :] += 2
        torch.testing.assert_close(
            source.grad,
            torch.as_tensor(expected_gradient, device=device),
        )


def test_numpy_round_keeps_torch_series_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    real_dtype = np.float32 if device == "mps" else np.float64
    complex_dtype = np.complex64 if device == "mps" else np.complex128

    real_values = np.array([1.24, -2.56, 3.45], dtype=real_dtype)
    complex_values = np.array(
        [1.234 + 0.126j, -2.555 - 0.444j, 3.456 + 0.555j],
        dtype=complex_dtype,
    )
    expected_real = np.round(real_values, decimals=1)
    expected_complex = np.round(complex_values, decimals=2)

    with ctx:
        real_series = TimeSeries(
            real_values, delta_t=0.25, epoch=12.0
        )
        complex_series = TimeSeries(
            complex_values, delta_t=0.25, epoch=12.0
        )
        output = TimeSeries(
            np.empty_like(real_values), delta_t=0.25, epoch=12.0
        )

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("numpy.round copied Torch data to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            actual_real = np.round(real_series, decimals=1)
            actual_complex = np.round(complex_series, decimals=2)
            returned = np.round(real_series, decimals=1, out=output)

        for actual, expected in (
            (actual_real, expected_real),
            (actual_complex, expected_complex),
            (output, expected_real),
        ):
            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.delta_t == 0.25
            assert actual.start_time == 12.0
            torch.testing.assert_close(
                actual._data.tensor,
                torch.as_tensor(expected, device=device),
            )
        assert returned is output


def test_numpy_round_preserves_torch_autograd(torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        source = torch.tensor(
            [1.24, -2.56, 3.45],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        values = Array(TorchArrayData(source), copy=False)
        rounded = np.round(values, decimals=1)
        rounded._data.tensor.sum().backward()

        assert rounded._data.tensor.device.type == device
        torch.testing.assert_close(source.grad, torch.zeros_like(source))


def test_coherent_chisq_cache_preserves_numpy_path():
    reference = np.zeros(6, dtype=np.float32)
    indices = np.array([4, 1], dtype=np.int64)
    cache = coherent.create_coherent_cache(reference, np.nan, np.float32)
    dof_cache = coherent.create_coherent_cache(reference, 0, np.int32)

    assert isinstance(cache, np.ndarray)
    assert isinstance(dof_cache, np.ndarray)
    np.testing.assert_array_equal(
        coherent.unavailable_coherent_indices(cache, indices), indices
    )
    coherent.update_coherent_cache(
        cache, indices, np.array([2.5, 3.5], dtype=np.float32)
    )
    coherent.update_coherent_cache(
        dof_cache, indices, np.array([4, 6], dtype=np.int32)
    )

    assert not len(coherent.unavailable_coherent_indices(cache, indices))
    np.testing.assert_array_equal(cache[indices], [2.5, 3.5])
    np.testing.assert_array_equal(dof_cache[indices], [4, 6])


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_unary_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([0.25, 0.5, 0.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    operations = (
        np.absolute,
        np.sqrt,
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log10,
        np.log2,
        np.log1p,
        np.sin,
        np.cos,
        np.tan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arctanh,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.rint,
        np.conjugate,
        np.negative,
        np.positive,
        np.square,
        np.reciprocal,
    )
    if np.dtype(dtype).kind != "c":
        operations += (np.fabs, np.floor, np.ceil, np.trunc)
    expected = {operation: operation(values) for operation in operations}

    def _reject_host_transfer(_self):
        raise AssertionError("unary ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        assert series._data.tensor.device.type == device

        for operation in operations:
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData, "numpy", _reject_host_transfer
                )
                actual = operation(series)

            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.dtype == expected[operation].dtype
            assert actual.delta_t == series.delta_t
            assert actual.start_time == series.start_time
            np.testing.assert_allclose(
                actual.numpy(), expected[operation], rtol=2e-6, atol=1e-7
            )


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_,
        np.int32,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ),
)
def test_predicate_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    if np.dtype(dtype).kind in "biu":
        values = np.array([0, 1, 0, 1], dtype=dtype)
    else:
        values = np.array([0.0, np.nan, np.inf, -np.inf], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.0j, 1.0j, np.nan * 1.0j, -1.0j], dtype=dtype)
    operations = (np.isfinite, np.isnan, np.isinf)
    expected = {operation: operation(values) for operation in operations}

    def _reject_host_transfer(_self):
        raise AssertionError("predicate ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)

        for operation in operations:
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData, "numpy", _reject_host_transfer
                )
                actual = operation(series)

            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.dtype == np.dtype(np.bool_)
            assert actual.delta_t == series.delta_t
            assert actual.start_time == series.start_time
            np.testing.assert_array_equal(actual.numpy(), expected[operation])


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_,
        np.int32,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ),
)
def test_logical_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([0, 1, 0, 2], dtype=dtype)
    others = np.array([1, 0, 2, 0], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.0j, 1.0j, 0.0j, -1.0j], dtype=dtype)
        others += np.array([1.0j, 0.0j, -1.0j, 0.0j], dtype=dtype)
    scalar = dtype(0)

    def _reject_host_transfer(_self):
        raise AssertionError("logical ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (np.logical_not, (series,), (values,)),
            (np.logical_and, (series, other_series), (values, others)),
            (np.logical_or, (series, scalar), (values, scalar)),
            (np.logical_xor, (scalar, series), (scalar, values)),
        )

        for operation, inputs, expected_inputs in cases:
            expected = operation(*expected_inputs)
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData, "numpy", _reject_host_transfer
                )
                actual = operation(*inputs)

            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.dtype == np.dtype(np.bool_)
            assert actual.delta_t == series.delta_t
            assert actual.start_time == series.start_time
            np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "dtype",
    (
        np.bool_,
        np.int32,
        np.int64,
        np.uint32,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ),
)
def test_comparison_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and (
        dtype in (np.int64, np.uint32, np.float64, np.complex128)
    ):
        pytest.skip("Torch MPS does not support this dtype")

    values = np.array([0, 1, 3], dtype=dtype)
    others = np.array([1, 1, 2], dtype=dtype)
    scalar = dtype(1)
    if np.dtype(dtype).kind == "c":
        values += np.array([1j, -1j, 0j], dtype=dtype)
        others += np.array([0j, -1j, 1j], dtype=dtype)
        scalar += dtype(0.5j)
    operations = (
        np.equal,
        np.not_equal,
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("comparison ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (series, other_series, values, others),
            (series, scalar, values, scalar),
            (scalar, series, scalar, values),
        )

        for operation in operations:
            for left, right, expected_left, expected_right in cases:
                expected = operation(expected_left, expected_right)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        TorchArrayData, "numpy", _reject_host_transfer
                    )
                    actual = operation(left, right)

                assert isinstance(actual, TimeSeries)
                assert actual._data.tensor.device.type == device
                assert actual.dtype == np.dtype(np.bool_)
                assert actual.delta_t == series.delta_t
                assert actual.start_time == series.start_time
                np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("dtype", (np.int32, np.uint32))
def test_comparison_numpy_ufuncs_match_out_of_range_integer_scalars(
        torch_ctx, monkeypatch, dtype):
    values = np.array([0, 1, np.iinfo(dtype).max], dtype=dtype)
    scalars = (-1, 2**40)

    def _reject_host_transfer(_self):
        raise AssertionError("comparison ufunc transferred data to NumPy")

    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        for operation in (
            np.equal,
            np.not_equal,
            np.less,
            np.less_equal,
            np.greater,
            np.greater_equal,
        ):
            for scalar in scalars:
                for left, right, expected_left, expected_right in (
                    (series, scalar, values, scalar),
                    (scalar, series, scalar, values),
                ):
                    expected = operation(expected_left, expected_right)
                    with monkeypatch.context() as patch:
                        patch.setattr(
                            TorchArrayData, "numpy", _reject_host_transfer
                        )
                        actual = operation(left, right)
                    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "operation",
    (
        np.fabs,
        np.cbrt,
        np.deg2rad,
        np.radians,
        np.rad2deg,
        np.degrees,
        np.floor,
        np.ceil,
        np.trunc,
    ),
)
def test_real_only_numpy_ufuncs_reject_complex(torch_ctx, operation):
    values = np.array([0.25 + 0.1j, -0.5 - 0.2j], dtype=np.complex64)

    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25)
        with pytest.raises(TypeError):
            operation(series)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_numpy_arccosh_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([1.25, 1.5, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    expected = np.arccosh(values)

    def _reject_host_transfer(_self):
        raise AssertionError("arccosh transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.arccosh(series)

    assert isinstance(actual, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == expected.dtype
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=2e-6, atol=1e-7
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_real_transform_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.float64:
        pytest.skip("MPS does not support float64")

    values = np.array([-8.0, -0.0, 27.0], dtype=dtype)
    operations = (
        np.sign,
        np.cbrt,
        np.deg2rad,
        np.radians,
        np.rad2deg,
        np.degrees,
    )
    expected = {operation: operation(values) for operation in operations}

    def _reject_host_transfer(_self):
        raise AssertionError("real transform transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        for operation in operations:
            with monkeypatch.context() as patch:
                patch.setattr(
                    TorchArrayData, "numpy", _reject_host_transfer
                )
                actual = operation(series)

            assert isinstance(actual, TimeSeries)
            assert actual._data.tensor.device.type == device
            assert actual.dtype == expected[operation].dtype
            assert actual.delta_t == series.delta_t
            assert actual.start_time == series.start_time
            np.testing.assert_allclose(
                actual.numpy(), expected[operation],
                rtol=2e-6, atol=1e-7,
            )
            np.testing.assert_array_equal(
                np.signbit(actual.numpy()),
                np.signbit(expected[operation]),
            )


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_binary_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([0.75, 1.25, 1.75], dtype=dtype)
    others = np.array([1.1, 0.8, 1.3], dtype=dtype)
    scalar = 1.5
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
        others += np.array([-0.15j, 0.25j, 0.05j], dtype=dtype)
        scalar += 0.125j
    operations = (
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.power,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("binary ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (series, other_series, values, others),
            (series, scalar, values, scalar),
            (scalar, series, scalar, values),
        )

        for operation in operations:
            for left, right, expected_left, expected_right in cases:
                expected = operation(expected_left, expected_right)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        TorchArrayData, "numpy", _reject_host_transfer
                    )
                    actual = operation(left, right)

                assert isinstance(actual, TimeSeries)
                assert actual._data.tensor.device.type == device
                assert actual.dtype == expected.dtype
                assert actual.delta_t == series.delta_t
                assert actual.start_time == series.start_time
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=2e-5, atol=2e-6
                )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_ordering_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.float64:
        pytest.skip("MPS does not support float64")

    values = np.array([0.75, np.nan, 1.75], dtype=dtype)
    others = np.array([1.1, 0.8, np.nan], dtype=dtype)
    scalar = dtype(1.5)
    operations = (np.maximum, np.minimum)

    def _reject_host_transfer(_self):
        raise AssertionError("ordering ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (series, other_series, values, others),
            (series, scalar, values, scalar),
            (scalar, series, scalar, values),
        )

        for operation in operations:
            for left, right, expected_left, expected_right in cases:
                expected = operation(expected_left, expected_right)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        TorchArrayData, "numpy", _reject_host_transfer
                    )
                    actual = operation(left, right)

                assert isinstance(actual, TimeSeries)
                assert actual._data.tensor.device.type == device
                assert actual.dtype == expected.dtype
                assert actual.delta_t == series.delta_t
                assert actual.start_time == series.start_time
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=0.0, atol=0.0,
                    equal_nan=True,
                )


@pytest.mark.parametrize("operation", (np.maximum, np.minimum))
def test_ordering_numpy_ufuncs_match_numpy_promotion(
        torch_ctx, monkeypatch, operation):
    values = np.array([0.75, 1.25, 1.75], dtype=np.float32)
    scalar = np.float64(1.5)
    expected = operation(values, scalar)

    def _reject_host_transfer(_self):
        raise AssertionError("ordering ufunc transferred data to NumPy")

    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation(series, scalar)

        assert isinstance(actual._data, TorchArrayData)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual.numpy(), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_real_binary_numpy_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.float64:
        pytest.skip("MPS does not support float64")

    values = np.array([-2.5, -0.75, 0.5, 2.25], dtype=dtype)
    others = np.array([0.75, -1.25, 1.5, -0.5], dtype=dtype)
    scalar = dtype(1.5)
    operations = (
        np.arctan2,
        np.hypot,
        np.fmod,
        np.remainder,
        np.copysign,
        np.logaddexp,
        np.logaddexp2,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("real binary ufunc transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        other_series = TimeSeries(others, delta_t=0.25, epoch=10.0)
        cases = (
            (series, other_series, values, others),
            (series, scalar, values, scalar),
            (scalar, series, scalar, values),
        )

        for operation in operations:
            for left, right, expected_left, expected_right in cases:
                expected = operation(expected_left, expected_right)
                with monkeypatch.context() as patch:
                    patch.setattr(
                        TorchArrayData, "numpy", _reject_host_transfer
                    )
                    actual = operation(left, right)

                assert isinstance(actual, TimeSeries)
                assert actual._data.tensor.device.type == device
                assert actual.dtype == expected.dtype
                assert actual.delta_t == series.delta_t
                assert actual.start_time == series.start_time
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=2e-5, atol=2e-6
                )


@pytest.mark.parametrize(
    "operation",
    (
        np.arctan2,
        np.hypot,
        np.fmod,
        np.remainder,
        np.copysign,
        np.logaddexp,
        np.logaddexp2,
    ),
)
def test_real_binary_numpy_ufuncs_reject_complex(torch_ctx, operation):
    values = np.array([0.25 + 0.1j, -0.5 - 0.2j], dtype=np.complex64)

    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25)
        with pytest.raises(TypeError):
            operation(series, series)


@pytest.mark.parametrize(
    "left_dtype,right_value",
    (
        (np.float32, np.float64(1.25)),
        (np.float32, np.array([1.0j, 0.5j, -0.25j], np.complex64)),
        (np.complex64, np.array([1.0, 0.5, 0.25], np.float64)),
    ),
)
def test_binary_numpy_ufuncs_match_numpy_promotion(
        torch_ctx, monkeypatch, left_dtype, right_value):
    left_values = np.array([0.75, 1.25, 1.75], dtype=left_dtype)
    if np.dtype(left_dtype).kind == "c":
        left_values += np.array([0.1j, -0.2j, 0.3j], dtype=left_dtype)
    expected = np.add(left_values, right_value)

    def _reject_host_transfer(_self):
        raise AssertionError("mixed-dtype ufunc transferred data to NumPy")

    with torch_ctx:
        left = TimeSeries(left_values, delta_t=0.25, epoch=10.0)
        right = (
            TimeSeries(right_value, delta_t=0.25, epoch=10.0)
            if isinstance(right_value, np.ndarray)
            else right_value
        )
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.add(left, right)

        assert isinstance(actual._data, TorchArrayData)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual.numpy(), expected)


def test_numpy_ufunc_falls_back_after_torch_scheme_exits(torch_ctx):
    values = np.array([0.5, 1.25, 2.0], dtype=np.float64)
    with torch_ctx:
        series = TimeSeries(values, delta_t=0.25)
        assert isinstance(series._data, TorchArrayData)

    actual = np.sqrt(series)

    assert not isinstance(actual._data, TorchArrayData)
    np.testing.assert_allclose(actual.numpy(), np.sqrt(values))


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("operation", (np.add, np.multiply))
def test_numpy_ufunc_reductions_only_transfer_scalar(
        torch_device_ctx, monkeypatch, dtype, operation):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")

    values = np.array([0.75, 1.25, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    expected = operation.reduce(values)

    def _reject_host_transfer(_self):
        raise AssertionError("ufunc reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.reduce(series)

    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "operation",
    (
        pytest.param(np.sum, id="sum"),
        pytest.param(np.max, id="max"),
        pytest.param(np.min, id="min"),
    ),
)
@pytest.mark.parametrize("axis", (None, 0, -1))
def test_high_level_numpy_reductions_only_transfer_scalar(
        torch_device_ctx, monkeypatch, operation, axis):
    ctx, device = torch_device_ctx
    values = np.array([0.75, -1.25, 1.75], dtype=np.float32)
    kwargs = {"axis": axis, "initial": 2.0}
    if operation is np.sum and device != "mps":
        kwargs["dtype"] = np.float64
    expected = operation(values, **kwargs)

    def _reject_host_transfer(_self):
        raise AssertionError("high-level reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = operation(series, **kwargs)

    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "operation",
    (pytest.param(np.any, id="any"), pytest.param(np.all, id="all")),
)
@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize(
    "dtype", (np.bool_, np.int32, np.float32, np.complex64)
)
def test_high_level_numpy_logical_reductions_only_transfer_scalar(
        torch_device_ctx, monkeypatch, operation, axis, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and np.dtype(dtype).kind == "c":
        pytest.skip("MPS does not support complex logical reductions")

    values = np.array([0.0, 2.0, 0.0], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values[2] = 0.5j
    expected = operation(values, axis=axis)

    def _reject_host_transfer(_self):
        raise AssertionError("logical reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = operation(series, axis=axis)
            method = "any" if operation is np.any else "all"
            direct = getattr(series, method)(axis=axis)

    assert isinstance(actual, np.bool_)
    assert actual == expected
    assert isinstance(direct, np.bool_)
    assert direct == expected


@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_high_level_numpy_mean_only_transfers_scalar(
        torch_device_ctx, monkeypatch, axis, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and np.dtype(dtype).kind == "c":
        pytest.skip("MPS does not support complex mean reduction")

    values = np.array([0.75, -1.25, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        output_dtype = np.complex64 if device == "mps" else np.complex128
    else:
        output_dtype = np.float32 if device == "mps" else np.float64
    expected_default = np.mean(values, axis=axis)
    expected = np.mean(values, axis=axis, dtype=output_dtype)

    def _reject_host_transfer(_self):
        raise AssertionError("mean transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual_default = np.mean(series, axis=axis)
            actual = np.mean(series, axis=axis, dtype=output_dtype)

    assert isinstance(actual_default, np.generic)
    assert actual_default.dtype == expected_default.dtype
    np.testing.assert_allclose(
        actual_default, expected_default, rtol=2e-6, atol=1e-7
    )
    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_high_level_numpy_mean_empty_identity_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, _ = torch_device_ctx

    def _reject_host_transfer(_self):
        raise AssertionError("empty mean transferred the full array")

    with ctx:
        array = Array(np.array([], dtype=np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            with pytest.warns(RuntimeWarning) as caught:
                actual = np.mean(array)

    assert any("Mean of empty slice" in str(item.message) for item in caught)
    assert isinstance(actual, np.float32)
    assert np.isnan(actual)


@pytest.mark.parametrize(
    "operation,method",
    (
        pytest.param(np.argmax, "argmax", id="argmax"),
        pytest.param(np.argmin, "argmin", id="argmin"),
    ),
)
@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize(
    "values",
    (
        pytest.param([0.75, -1.25, 1.75, 1.75], id="ties"),
        pytest.param([0.75, np.nan, 1.75], id="nan"),
    ),
)
def test_high_level_numpy_argument_reductions_only_transfer_scalar(
        torch_device_ctx, monkeypatch, operation, method, axis, values):
    ctx, _ = torch_device_ctx
    values = np.asarray(values, dtype=np.float32)
    expected = operation(values, axis=axis)

    def _reject_host_transfer(_self):
        raise AssertionError("argument reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = operation(series, axis=axis)
            direct = getattr(series, method)(axis=axis)

    assert isinstance(actual, np.integer)
    assert actual == expected
    assert direct == expected


@pytest.mark.parametrize("operation", (np.argmax, np.argmin))
@pytest.mark.parametrize("dtype", (np.int32, np.uint32))
def test_high_level_numpy_integer_argument_reductions_stay_on_device(
        torch_ctx, monkeypatch, operation, dtype):
    values = np.array([2, 1, 3, 3], dtype=dtype)
    expected = operation(values)

    def _reject_host_transfer(_self):
        raise AssertionError("integer argument reduction transferred to NumPy")

    with torch_ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = operation(array)

    assert isinstance(actual, np.integer)
    assert actual == expected


@pytest.mark.parametrize("operation", (np.argmax, np.argmin))
def test_high_level_numpy_argument_reduction_empty_stays_on_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, _ = torch_device_ctx

    def _reject_host_transfer(_self):
        raise AssertionError("empty argument reduction transferred to NumPy")

    with ctx:
        array = Array(np.array([], dtype=np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            with pytest.raises(
                    ValueError,
                    match="attempt to get arg(max|min) of an empty sequence"):
                operation(array)


@pytest.mark.parametrize("operation", (np.argmax, np.argmin))
def test_numpy_multidimensional_argument_reductions_stay_on_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, device = torch_device_ctx
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    values[0, 1, 2] = np.nan
    values[1, 0, 3] = np.nan

    def _reject_host_transfer(_self):
        raise AssertionError("argument reduction used NumPy data")

    with ctx:
        array = Array(values)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for axis in (0, 1, 2, -1, None):
                for keepdims in (False, True):
                    results[axis, keepdims] = operation(
                        array, axis=axis, keepdims=keepdims
                    )

    for (axis, keepdims), actual in results.items():
        expected = operation(values, axis=axis, keepdims=keepdims)
        if expected.ndim == 0:
            assert isinstance(actual, np.integer)
        else:
            assert type(actual) is Array
            assert isinstance(actual._data, TorchArrayData)
            assert actual._data.tensor.device.type == device
            actual = actual.numpy()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", (np.argmax, np.argmin))
@pytest.mark.parametrize(
    "dtype", (np.bool_, np.int32, np.uint32, np.float32)
)
def test_numpy_multidimensional_argument_reduction_dtypes(
        torch_ctx, monkeypatch, operation, dtype):
    values = np.array(
        [[2, 1, 3, 3], [5, 0, 4, 5], [1, 6, 2, 6]], dtype=dtype
    )

    def _reject_host_transfer(_self):
        raise AssertionError("argument reduction dtype used NumPy data")

    with torch_ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation(array, axis=1, keepdims=True)

    expected = operation(values, axis=1, keepdims=True)
    assert type(actual) is Array
    assert isinstance(actual._data, TorchArrayData)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("operation", (np.argmax, np.argmin))
def test_numpy_empty_multidimensional_argument_reductions_stay_on_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, device = torch_device_ctx
    values = np.empty((2, 0, 3), dtype=np.float32)

    def _reject_host_transfer(_self):
        raise AssertionError("empty argument reduction used NumPy data")

    with ctx:
        array = Array(values)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for axis in (0, 2):
                for keepdims in (False, True):
                    results[axis, keepdims] = operation(
                        array, axis=axis, keepdims=keepdims
                    )
            for axis in (1, None):
                with pytest.raises(
                        ValueError,
                        match="attempt to get arg(max|min) of an empty sequence"):
                    operation(array, axis=axis)

    for (axis, keepdims), actual in results.items():
        expected = operation(values, axis=axis, keepdims=keepdims)
        assert type(actual) is Array
        assert isinstance(actual._data, TorchArrayData)
        assert actual._data.tensor.device.type == device
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "operation", (pytest.param(np.var, id="var"), pytest.param(np.std, id="std"))
)
@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
@pytest.mark.parametrize("ddof", (0, 1))
def test_high_level_numpy_variance_only_transfers_scalar(
        torch_device_ctx, monkeypatch, operation, axis, dtype, ddof):
    ctx, device = torch_device_ctx
    if device == "mps" and np.dtype(dtype).kind == "c":
        pytest.skip("MPS does not support complex variance reduction")

    values = np.array([0.75, -1.25, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        output_dtype = np.complex64 if device == "mps" else np.complex128
    else:
        output_dtype = np.float32 if device == "mps" else np.float64
    expected_default = operation(values, axis=axis, ddof=ddof)
    expected = operation(
        values, axis=axis, dtype=output_dtype, ddof=ddof
    )

    def _reject_host_transfer(_self):
        raise AssertionError("variance transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual_default = operation(series, axis=axis, ddof=ddof)
            actual = operation(
                series, axis=axis, dtype=output_dtype, ddof=ddof
            )

    assert isinstance(actual_default, np.generic)
    assert actual_default.dtype == expected_default.dtype
    np.testing.assert_allclose(
        actual_default, expected_default, rtol=2e-6, atol=1e-7
    )
    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("operation", (np.var, np.std))
def test_high_level_numpy_variance_empty_stays_on_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, _ = torch_device_ctx

    def _reject_host_transfer(_self):
        raise AssertionError("empty variance transferred the full array")

    with ctx:
        array = Array(np.array([], dtype=np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            with pytest.warns(RuntimeWarning) as caught:
                actual = operation(array)

    assert any(
        "Degrees of freedom <= 0" in str(item.message) for item in caught
    )
    assert isinstance(actual, np.float32)
    assert np.isnan(actual)


@pytest.mark.parametrize(
    "operation,method,identity",
    ((np.logical_or, "any", False), (np.logical_and, "all", True)),
)
def test_numpy_logical_reduction_initial_and_empty_identity(
        torch_device_ctx, monkeypatch, operation, method, identity):
    ctx, _ = torch_device_ctx
    empty = np.array([], dtype=np.float32)

    def _reject_host_transfer(_self):
        raise AssertionError("logical reduction transferred the full array")

    with ctx:
        array = Array(empty)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = operation.reduce(array)
            inverted = operation.reduce(array, initial=not identity)
            direct = getattr(array, method)()

    assert isinstance(actual, np.bool_)
    assert actual == identity
    assert inverted == (not identity)
    assert isinstance(direct, np.bool_)
    assert direct == identity


@pytest.mark.parametrize("axis", (None, 0, -1))
@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_high_level_numpy_product_only_transfers_scalar(
        torch_device_ctx, monkeypatch, axis, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and np.dtype(dtype).kind == "c":
        pytest.skip("MPS does not support complex product reduction")

    values = np.array([0.75, 1.25, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    expected = np.prod(values, axis=axis, initial=2.0)

    def _reject_host_transfer(_self):
        raise AssertionError("product transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.prod(series, axis=axis, initial=2.0)
            direct = series.prod(axis=axis, initial=2.0)

    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert direct == actual


def test_high_level_numpy_product_empty_identity_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, _ = torch_device_ctx

    def _reject_host_transfer(_self):
        raise AssertionError("empty product transferred the full array")

    with ctx:
        array = Array(np.array([], dtype=np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.prod(array)

    assert isinstance(actual, np.float32)
    assert actual == 1.0


@pytest.mark.parametrize("axis", (None, 0, -1))
def test_high_level_numpy_cumsum_stays_on_torch_device(
        torch_device_ctx, monkeypatch, axis):
    ctx, device = torch_device_ctx
    values = np.array([0.75, -1.25, 1.75], dtype=np.float32)
    dtype = np.float32 if device == "mps" else np.float64
    expected = np.cumsum(values, axis=axis, dtype=dtype)

    def _reject_host_transfer(_self):
        raise AssertionError("high-level cumsum transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        source = series._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(
                array_torch_module, "numpy", _reject_host_transfer
            )
            actual = np.cumsum(series, axis=axis, dtype=dtype)
        actual._data.tensor.sum().backward()

    assert isinstance(actual, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    np.testing.assert_allclose(actual.numpy(), expected)


@pytest.mark.parametrize("axis", (None, 0, -1))
def test_high_level_numpy_cumprod_stays_on_torch_device(
        torch_device_ctx, monkeypatch, axis):
    ctx, device = torch_device_ctx
    values = np.array([0.75, 1.25, 1.75], dtype=np.float32)
    dtype = np.float32 if device == "mps" else np.float64
    expected = np.cumprod(values, axis=axis, dtype=dtype)

    def _reject_host_transfer(_self):
        raise AssertionError("high-level cumprod transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        source = series._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.cumprod(series, axis=axis, dtype=dtype)
            direct = series.cumprod(axis=axis, dtype=dtype)
        actual._data.tensor.sum().backward()

    assert isinstance(actual, TimeSeries)
    assert isinstance(direct, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    np.testing.assert_allclose(actual.numpy(), expected)
    np.testing.assert_allclose(direct.numpy(), expected)


def test_high_level_numpy_clip_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array([-3.0, -0.25, 0.5, 4.0], dtype=np.float32)
    expected = np.clip(values, -1.0, 2.0)

    def _reject_host_transfer(_self):
        raise AssertionError("high-level clip transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        source = series._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = np.clip(series, -1.0, 2.0)
            direct = series.clip(min=-1.0, max=2.0)
            upper_only = np.clip(series, None, 2.0)
            unchanged = np.clip(series, None, None)
        actual._data.tensor.sum().backward()

    for result in (actual, direct, upper_only, unchanged):
        assert isinstance(result, TimeSeries)
        assert result._data.tensor.device.type == device
        assert result.delta_t == series.delta_t
        assert result.start_time == series.start_time
    assert source.grad is not None
    np.testing.assert_array_equal(
        source.grad.detach().cpu().numpy(), [0.0, 1.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(actual.numpy(), expected)
    np.testing.assert_allclose(direct.numpy(), expected)
    np.testing.assert_allclose(
        upper_only.numpy(), np.clip(values, None, 2.0)
    )
    np.testing.assert_allclose(unchanged.numpy(), values)


def test_high_level_numpy_clip_array_bounds_and_out_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    bound_dtype = np.float32 if device == "mps" else np.float64
    values = np.array([-3.0, 0.5, 4.0], dtype=np.float32)
    lower_values = np.array([-2.0, 0.0, 1.0], dtype=bound_dtype)
    upper_values = np.array([-1.0, 1.0, 3.0], dtype=bound_dtype)
    expected = np.clip(values, lower_values, upper_values)

    def _reject_host_transfer(_self):
        raise AssertionError("array-bound clip transferred data to NumPy")

    with ctx:
        series = FrequencySeries(values, delta_f=0.5, epoch=10.0)
        lower = Array(lower_values)
        upper = Array(upper_values)
        out = FrequencySeries(
            np.zeros(3, dtype=expected.dtype), delta_f=0.5, epoch=10.0
        )
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            result = np.clip(series, lower, upper, out=out)

    assert result is out
    assert result._data.tensor.device.type == device
    assert result.dtype == expected.dtype
    assert result.delta_f == series.delta_f
    assert result.epoch == series.epoch
    np.testing.assert_allclose(result.numpy(), expected)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("operation", (np.add, np.multiply))
def test_numpy_ufunc_accumulations_stay_on_torch_device(
        torch_device_ctx, monkeypatch, dtype, operation):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("MPS does not support float64 or complex128")
    if device == "mps" and np.dtype(dtype).kind == "c":
        pytest.skip("MPS does not support complex cumulative operations")

    values = np.array([0.75, 1.25, 1.75], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values += np.array([0.1j, -0.2j, 0.3j], dtype=dtype)
    expected = operation.accumulate(values)

    def _reject_host_transfer(_self):
        raise AssertionError("ufunc accumulation transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.accumulate(series)

    assert isinstance(actual, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == expected.dtype
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=2e-6, atol=1e-7
    )


@pytest.mark.parametrize("operation", (np.add, np.multiply))
@pytest.mark.parametrize("axis", (None, 0, -1))
def test_numpy_ufunc_reduction_options_avoid_full_host_transfer(
        torch_device_ctx, monkeypatch, operation, axis):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("MPS does not support the requested float64 accumulator")

    values = np.array([0.75, 1.25, 1.75], dtype=np.float32)
    expected = operation.reduce(
        values, axis=axis, dtype=np.float64, initial=2.0
    )

    def _reject_host_transfer(_self):
        raise AssertionError("ufunc reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.reduce(
                series, axis=axis, dtype=np.float64, initial=2.0
            )

    assert isinstance(actual, np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


def test_numpy_multidimensional_reductions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.array(
        [[0.75, 1.25], [1.75, 2.25], [3.25, 4.25]],
        dtype=np.float32,
    )
    operations = (
        np.add,
        np.multiply,
        np.maximum,
        np.minimum,
        np.logical_or,
        np.logical_and,
    )
    axes = (0, 1, -1, (0, 1), (), None)

    def _reject_host_transfer(_self):
        raise AssertionError("multidimensional reduction used NumPy data")

    with ctx:
        array = Array(values)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for operation in operations:
                for axis in axes:
                    for keepdims in (False, True):
                        results[operation, axis, keepdims] = operation.reduce(
                            array, axis=axis, keepdims=keepdims
                        )

    for (operation, axis, keepdims), actual in results.items():
        expected = operation.reduce(
            values, axis=axis, keepdims=keepdims
        )
        if expected.ndim == 0:
            assert isinstance(actual, np.generic)
        else:
            assert isinstance(actual, Array)
            assert isinstance(actual._data, TorchArrayData)
            assert actual._data.tensor.device.type == device
            actual = actual.numpy()
        assert np.asarray(actual).dtype == expected.dtype
        np.testing.assert_allclose(actual, expected)


def test_numpy_multidimensional_mean_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    def _reject_host_transfer(_self):
        raise AssertionError("multidimensional mean used NumPy data")

    with ctx:
        array = Array(values)
        source = array._data.tensor
        source.requires_grad_(True)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for axis in (0, 1, -1, (0, 1), (), None):
                for keepdims in (False, True):
                    results[axis, keepdims] = np.mean(
                        array, axis=axis, keepdims=keepdims
                    )
        results[1, False]._data.tensor.sum().backward()

    assert source.grad is not None
    np.testing.assert_allclose(
        source.grad.detach().cpu().numpy(), np.full_like(values, 0.25)
    )
    for (axis, keepdims), actual in results.items():
        expected = np.mean(values, axis=axis, keepdims=keepdims)
        if expected.ndim == 0:
            assert isinstance(actual, np.generic)
        else:
            assert isinstance(actual, Array)
            assert isinstance(actual._data, TorchArrayData)
            assert actual._data.tensor.device.type == device
            actual = actual.numpy()
        assert np.asarray(actual).dtype == expected.dtype
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("operation", (np.var, np.std))
def test_numpy_multidimensional_variance_stays_on_torch_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, device = torch_device_ctx
    values = np.array(
        [[0.75, -1.25, 1.75], [2.25, -2.75, 3.25]],
        dtype=np.float32,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("multidimensional variance used NumPy data")

    with ctx:
        array = Array(values)
        source = array._data.tensor
        source.requires_grad_(True)
        results = {}
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            for axis in (0, 1, -1, (0, 1), (), None):
                for keepdims in (False, True):
                    results[axis, keepdims] = operation(
                        array, axis=axis, ddof=0.5, keepdims=keepdims
                    )
        if operation is np.var:
            results[1, False]._data.tensor.sum().backward()

    if operation is np.var:
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
    for (axis, keepdims), actual in results.items():
        expected = operation(
            values, axis=axis, ddof=0.5, keepdims=keepdims
        )
        if expected.ndim == 0:
            assert isinstance(actual, np.generic)
        else:
            assert type(actual) is Array
            assert isinstance(actual._data, TorchArrayData)
            assert actual._data.tensor.device.type == device
            actual = actual.numpy()
        assert np.asarray(actual).dtype == expected.dtype
        np.testing.assert_allclose(
            actual, expected, rtol=2e-6, atol=1e-7
        )


@pytest.mark.parametrize("operation", (np.var, np.std))
@pytest.mark.parametrize(
    "dtype", (np.bool_, np.int32, np.float32, np.complex64)
)
def test_numpy_multidimensional_variance_dtype_promotion(
        torch_ctx, monkeypatch, operation, dtype):
    values = np.arange(12).reshape(3, 4).astype(dtype)
    if np.dtype(dtype).kind == "c":
        values += 1j * np.flip(values.real, axis=1)

    def _reject_host_transfer(_self):
        raise AssertionError("variance dtype promotion used NumPy data")

    with torch_ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation(array, axis=1, keepdims=True)
            promoted = operation(
                array, axis=0, dtype=np.complex128, ddof=1
            )

    expected = operation(values, axis=1, keepdims=True)
    expected_promoted = operation(
        values, axis=0, dtype=np.complex128, ddof=1
    )
    for result, reference in (
            (actual, expected), (promoted, expected_promoted)):
        assert type(result) is Array
        assert isinstance(result._data, TorchArrayData)
        assert result.dtype == reference.dtype
        tolerance = 2e-6 if reference.dtype.itemsize <= 4 else 2e-12
        np.testing.assert_allclose(
            result.numpy(), reference, rtol=tolerance, atol=tolerance * 0.05
        )


@pytest.mark.parametrize("operation", (np.var, np.std))
def test_numpy_empty_multidimensional_variance_stays_on_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, device = torch_device_ctx
    values = np.empty((2, 0, 3), dtype=np.float32)

    def _reject_host_transfer(_self):
        raise AssertionError("empty variance used NumPy data")

    with ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            with pytest.warns(RuntimeWarning) as caught:
                actual = operation(
                    array, axis=1, ddof=0, keepdims=True
                )

    assert any(
        "Degrees of freedom <= 0" in str(item.message) for item in caught
    )
    assert type(actual) is Array
    assert isinstance(actual._data, TorchArrayData)
    assert actual._data.tensor.device.type == device
    with pytest.warns(RuntimeWarning):
        expected = operation(values, axis=1, ddof=0, keepdims=True)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("dtype", (np.bool_, np.int32, np.int64))
def test_numpy_integer_multidimensional_reduction_promotion(
        torch_ctx, monkeypatch, dtype):
    values = np.array([[1, 0, 2], [3, 4, 0]], dtype=dtype)

    def _reject_host_transfer(_self):
        raise AssertionError("integer reduction used NumPy data")

    with torch_ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_sum = np.sum(array, axis=1)
            actual_product = np.prod(array, axis=0, keepdims=True)
            actual_mean = np.mean(array, axis=1)

    for actual, expected in (
        (actual_sum, np.sum(values, axis=1)),
        (actual_product, np.prod(values, axis=0, keepdims=True)),
        (actual_mean, np.mean(values, axis=1)),
    ):
        assert isinstance(actual, Array)
        assert isinstance(actual._data, TorchArrayData)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual.numpy(), expected)


def test_numpy_uint32_empty_axis_reduction_fallback(torch_ctx):
    values = np.array([[1, 2], [3, 4]], dtype=np.uint32)

    with torch_ctx:
        array = Array(values)
        actual_sum = np.sum(array, axis=())
        actual_max = np.maximum.reduce(array, axis=(), initial=2)

    expected_sum = np.sum(values, axis=())
    expected_max = np.maximum.reduce(values, axis=(), initial=2)
    assert isinstance(actual_sum, np.ndarray)
    assert actual_sum.dtype == expected_sum.dtype
    assert isinstance(actual_max, Array)
    assert isinstance(actual_max._data, TorchArrayData)
    np.testing.assert_array_equal(actual_sum, expected_sum)
    np.testing.assert_array_equal(actual_max.numpy(), expected_max)


def test_numpy_empty_multidimensional_reductions_stay_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    values = np.empty((3, 0), dtype=np.float32)

    def _reject_host_transfer(_self):
        raise AssertionError("empty reduction used NumPy data")

    with ctx:
        array = Array(values)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            with pytest.warns(RuntimeWarning) as caught:
                actual_mean = np.mean(array, axis=1)
            actual_max = np.max(array, axis=1, initial=2.0)
            with pytest.raises(
                ValueError,
                match="zero-size array to reduction operation",
            ):
                np.max(array, axis=1)

    assert len(caught) == 2
    for actual in (actual_mean, actual_max):
        assert isinstance(actual, Array)
        assert isinstance(actual._data, TorchArrayData)
        assert actual._data.tensor.device.type == device
    with pytest.warns(RuntimeWarning) as expected_caught:
        expected_mean = np.mean(values, axis=1)
    assert len(expected_caught) == 2
    np.testing.assert_allclose(
        actual_mean.numpy(), expected_mean, equal_nan=True
    )
    np.testing.assert_allclose(
        actual_max.numpy(), np.max(values, axis=1, initial=2.0)
    )


@pytest.mark.parametrize("operation", (np.add, np.multiply))
@pytest.mark.parametrize("axis", (0, 1, -1))
def test_numpy_multidimensional_accumulations_stay_on_torch_device(
        torch_device_ctx, monkeypatch, operation, axis):
    ctx, device = torch_device_ctx
    values = np.array(
        [[0.75, 1.25, 1.75], [2.25, 2.75, 3.25]],
        dtype=np.float32,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("multidimensional accumulation used NumPy data")

    with ctx:
        array = Array(values)
        source = array._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.accumulate(array, axis=axis)
        actual._data.tensor.sum().backward()

    expected = operation.accumulate(values, axis=axis)
    assert isinstance(actual, Array)
    assert isinstance(actual._data, TorchArrayData)
    assert actual._data.tensor.device.type == device
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    np.testing.assert_allclose(actual.numpy(), expected)


@pytest.mark.parametrize("operation", (np.add, np.multiply))
@pytest.mark.parametrize("axis", (None, 0, -1))
def test_numpy_ufunc_accumulation_options_stay_on_torch_device(
        torch_device_ctx, monkeypatch, operation, axis):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("MPS does not support the requested float64 accumulator")

    values = np.array([0.75, 1.25, 1.75], dtype=np.float32)
    expected = operation.accumulate(values, axis=axis, dtype=np.float64)

    def _reject_host_transfer(_self):
        raise AssertionError("ufunc accumulation transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        source = series._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.accumulate(
                series, axis=axis, dtype=np.float64
            )
        actual._data.tensor.sum().backward()

    assert isinstance(actual, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(np.float64)
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert actual._data.tensor.requires_grad
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=2e-12, atol=1e-14
    )


@pytest.mark.parametrize("operation", (np.maximum, np.minimum))
def test_ordering_numpy_ufunc_reduction_options_stay_on_torch_device(
        torch_device_ctx, monkeypatch, operation):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("MPS does not support the requested float64 accumulator")

    values = np.array([0.75, -1.25, 1.75], dtype=np.float32)
    expected = operation.reduce(
        values, axis=None, dtype=np.float64, initial=2.0
    )
    expected_empty = operation.reduce(
        np.array([], dtype=np.float32),
        axis=None,
        dtype=np.float64,
        initial=2.0,
    )

    def _reject_host_transfer(_self):
        raise AssertionError("ordering reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        empty = Array(np.array([], dtype=np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.reduce(
                series, axis=None, dtype=np.float64, initial=2.0
            )
            empty_actual = operation.reduce(
                empty, axis=None, dtype=np.float64, initial=2.0
            )

    assert isinstance(actual, np.float64)
    assert isinstance(empty_actual, np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        empty_actual, expected_empty, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("operation", (np.maximum, np.minimum))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_ordering_numpy_ufunc_accumulations_stay_on_torch_device(
        torch_device_ctx, monkeypatch, operation, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.float64:
        pytest.skip("MPS does not support float64")

    values = np.array([0.75, -1.25, np.nan, 1.75], dtype=dtype)
    expected = operation.accumulate(values)

    def _reject_host_transfer(_self):
        raise AssertionError("ordering accumulation transferred data to NumPy")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        source = series._data.tensor
        source.requires_grad_(True)
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.accumulate(series)

        actual._data.tensor.nan_to_num().sum().backward()

    assert isinstance(actual, TimeSeries)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == expected.dtype
    assert actual.delta_t == series.delta_t
    assert actual.start_time == series.start_time
    assert actual._data.tensor.requires_grad
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=0.0, atol=0.0, equal_nan=True
    )


@pytest.mark.parametrize("operation", (np.maximum, np.minimum))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_ordering_numpy_ufunc_reductions_only_transfer_scalar(
        torch_device_ctx, monkeypatch, operation, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype is np.float64:
        pytest.skip("MPS does not support float64")

    values = np.array([0.75, -1.25, 1.75], dtype=dtype)
    expected = operation.reduce(values)

    def _reject_host_transfer(_self):
        raise AssertionError("ordering reduction transferred the full array")

    with ctx:
        series = TimeSeries(values, delta_t=0.25, epoch=10.0)
        nan_series = TimeSeries(
            np.array([0.75, np.nan, 1.75], dtype=dtype),
            delta_t=0.25,
            epoch=10.0,
        )
        empty = Array(np.array([], dtype=dtype))
        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = operation.reduce(series)
            nan_actual = operation.reduce(nan_series)
            with pytest.raises(
                ValueError,
                match="zero-size array to reduction operation",
            ):
                operation.reduce(empty)

    assert isinstance(actual, np.generic)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert np.isnan(nan_actual)


def test_equal_weight_resampling_preserves_numpy_contract():
    samples = {
        "x": np.linspace(-1.0, 1.0, 7),
        "y": np.arange(7, dtype=np.int64),
    }
    logwt = np.array([-2.0, -1.0, 0.2, -0.4, 1.1, -0.8, 0.5])

    np.random.seed(2917)
    weights = np.exp(logwt - scipy.special.logsumexp(logwt))
    positions = (np.random.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    expected_indices = np.searchsorted(cumulative, positions, side="right")
    rng = np.random.default_rng(83)
    rng.shuffle(expected_indices)

    np.random.seed(2917)
    actual = inference_refine.resample_equal(samples, logwt, seed=83)

    assert all(isinstance(values, np.ndarray) for values in actual.values())
    for key in samples:
        np.testing.assert_array_equal(
            actual[key], samples[key][expected_indices]
        )


def test_single_template_batch_loglr_accepts_numpy_grid_on_torch():
    class _Detector:
        @staticmethod
        def antenna_pattern(ra, dec, polarization, _times):
            return (
                torch.ones_like(ra) * 0.43,
                torch.ones_like(dec) * -0.27,
            )

        @staticmethod
        def time_delay_from_earth_center(ra, _dec, _times):
            return torch.zeros_like(ra)

    ifo = "H1"
    model = types.SimpleNamespace(
        marginalize_vector_params={},
        marginalize_distance=False,
        sh={},
        hh={ifo: 1.7},
        snr={},
        det={ifo: _Detector()},
        dts={},
        htfs={},
    )
    base_params = {
        "ra": 1.0,
        "dec": -0.4,
        "polarization": 0.39,
        "inclination": 0.71,
        "coa_phase": 0.23,
        "distance": 1.8,
    }

    def _update(**params):
        model.current_params = {**base_params, **params}

    model.update = _update
    model._loglr = lambda skip_vector=False: (
        single_template.SingleTemplate._loglr(model, skip_vector)
    )
    model.marginalize_loglr = (
        lambda sh, norm, skip_vector=False: sh.real - 0.5 * norm
    )

    times = np.array((0.75, 1.0, 1.25))
    with scheme.TorchScheme("cpu"):
        model.sh[ifo] = TimeSeries(
            np.linspace(1.0, 9.0, 9) * (1.0 + 0.2j),
            delta_t=0.25,
            epoch=0.0,
        )
        actual = single_template.SingleTemplate.batch_loglr(
            model, tc=times
        )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    assert actual.shape == times.shape
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("model_name", _DIRECT_SPACE_PSD_MODELS)
@pytest.mark.parametrize("tdi", ("1.5", "2.0"))
def test_analytical_space_psd_torch_matches_numpy(
        torch_ctx, monkeypatch, model_name, tdi):
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=513,
        delta_f=1e-4,
        low_freq_cutoff=3.5e-4,
        tdi=tdi,
    )
    expected = model(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical PSD used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor[:3]) == 0
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=5e-12, atol=0.0
    )


@pytest.mark.parametrize("tdi", ("1.5", "2.0"))
def test_analytical_space_csd_torch_matches_numpy(
        torch_device_ctx, monkeypatch, tdi):
    ctx, device = torch_device_ctx
    parameters = dict(
        length=513,
        delta_f=1e-4,
        low_freq_cutoff=3.5e-4,
        len_arm=2.4e9,
        acc_noise_level=3.2e-15,
        oms_noise_level=13e-12,
        tdi=tdi,
    )
    with np.errstate(invalid="ignore"):
        expected = analytical_space.analytical_csd_lisa_tdi_XY(
            **parameters
        ).numpy()

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical CSD used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            if device == "mps":
                with pytest.raises(TypeError, match="require float64"):
                    analytical_space.analytical_csd_lisa_tdi_XY(**parameters)
                return
            actual = analytical_space.analytical_csd_lisa_tdi_XY(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == device
    assert tensor.dtype == torch.float64
    np.testing.assert_array_equal(np.isnan(actual.numpy()), np.isnan(expected))
    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=5e-12, atol=0.0, equal_nan=True
    )


@pytest.mark.parametrize("model_name, extra", _DIRECT_SPACE_CURVES)
def test_analytical_space_curve_torch_matches_numpy(
        torch_ctx, monkeypatch, model_name, extra):
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=513,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        **extra,
    )
    expected = model(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch analytical curve used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor[:7]) == 0
    assert _relative_l2(actual.numpy(), expected) < 5e-12

    # The compact-support fits must retain the legacy exact-zero regions.
    if model_name in ("confusion_fit_tianqin", "confusion_fit_taiji"):
        assert np.array_equal(actual.numpy() == 0, expected == 0)


@pytest.mark.parametrize("model_name, extra", _COMBINED_SPACE_CURVES)
def test_analytical_space_combined_curve_torch_matches_numpy(
        torch_ctx, monkeypatch, model_name, extra):
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=513,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        **extra,
    )
    expected = model(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch combined curve used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor[:7]) == 0
    assert _relative_l2(actual.numpy(), expected) < 5e-12


@pytest.mark.parametrize("model_name, extra", _CONFUSION_SPACE_PSDS)
@pytest.mark.parametrize("tdi", ("1.5", "2.0"))
def test_analytical_space_confusion_psd_torch_matches_numpy(
        torch_ctx, monkeypatch, model_name, extra, tdi):
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=513,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        tdi=tdi,
        **extra,
    )
    expected = model(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch confusion PSD used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor[:7]) == 0
    assert _relative_l2(actual.numpy(), expected) < 5e-12
    assert np.array_equal(actual.numpy() == 0, expected == 0)


@pytest.mark.parametrize("model_name, extra", _LISA_RESPONSE_MODELS)
def test_lisa_response_psd_torch_matches_numpy(
        torch_ctx, monkeypatch, model_name, extra):
    model = getattr(analytical_space, model_name)
    parameters = dict(
        length=513,
        delta_f=5e-5,
        low_freq_cutoff=3.7e-4,
        **extra,
    )
    monkeypatch.setattr(
        analytical_space,
        "_load_lisa_averaged_response_data",
        lambda: _SYNTHETIC_LISA_RESPONSE,
    )
    expected = model(**parameters).numpy()

    with torch_ctx:
        with monkeypatch.context() as patch:
            def _reject_host_path(*_args, **_kwargs):
                raise AssertionError("Torch LISA response model used NumPy/SciPy")

            patch.setattr(analytical_space.np, "linspace", _reject_host_path)
            patch.setattr(
                analytical_space, "from_numpy_arrays", _reject_host_path
            )
            patch.setattr(analytical_space, "interp1d", _reject_host_path)
            actual = model(**parameters)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.count_nonzero(tensor[:7]) == 0
    assert _relative_l2(actual.numpy(), expected) < 5e-12

