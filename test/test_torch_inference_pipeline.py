# flake8: noqa: F401
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


def test_base_inference_models_keep_stats_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    class _Prior:
        variable_args = ("x",)

        def __init__(self, invalid=False):
            self.invalid = invalid
            self.calls = 0

        @staticmethod
        def apply_boundary_conditions(**params):
            return params

        def __call__(self, *, x):
            self.calls += 1
            if self.invalid:
                return x.new_full((), torch.nan)
            return -0.5 * x.square()

    class _Model(inference_model_base.BaseModel):
        def __init__(self, *args, **kwargs):
            self.likelihood_calls = 0
            super().__init__(*args, **kwargs)

        def _loglikelihood(self):
            self.likelihood_calls += 1
            return 2.0 * self.current_params["x"]

    class _DataModel(inference_model_base_data.BaseDataModel):
        def __init__(self, *args, **kwargs):
            self.lr_calls = 0
            super().__init__(*args, **kwargs)

        def _loglikelihood(self):
            return self._loglr() + self._lognl()

        def _loglr(self):
            self.lr_calls += 1
            return 3.0 * self.current_params["x"]

        def _lognl(self):
            return self.current_params["x"].new_tensor(-1.0)

    with ctx:
        x = torch.tensor(
            0.25, device=device, dtype=dtype, requires_grad=True
        )
        finite_prior = _Prior()
        invalid_prior = _Prior(invalid=True)
        finite_model = _Model(("x",), prior=finite_prior)
        no_prior_model = _Model(("x",))
        invalid_model = _Model(("x",), prior=invalid_prior)
        data_model = _DataModel(("x",), data={}, prior=_Prior())
        invalid_data_model = _DataModel(
            ("x",), data={}, prior=_Prior(invalid=True)
        )
        for model in (
            finite_model,
            no_prior_model,
            invalid_model,
            data_model,
            invalid_data_model,
        ):
            model.update(x=x)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("model statistic evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_model_base.numpy, "isnan", reject_host_or_numpy
            )
            finite_posterior = finite_model.logposterior
            cached_prior = finite_model.logprior
            no_prior_posterior = no_prior_model.logposterior
            invalid_posterior = invalid_model.logposterior
            data_logplr = data_model.logplr
            invalid_data_logplr = invalid_data_model.logplr

        finite_values = (
            finite_posterior,
            cached_prior,
            no_prior_posterior,
            data_logplr,
        )
        assert all(isinstance(value, torch.Tensor) for value in finite_values)
        assert all(value.device.type == device for value in finite_values)
        assert cached_prior is finite_model._current_stats.logprior
        assert finite_prior.calls == 1
        assert finite_model.likelihood_calls == 1
        assert no_prior_model.likelihood_calls == 1
        assert invalid_prior.calls == 1
        assert invalid_model.likelihood_calls == 0
        assert bool(torch.isneginf(invalid_posterior))
        assert data_model.lr_calls == 1
        assert invalid_data_model.lr_calls == 0
        assert bool(torch.isneginf(invalid_data_logplr))

        (finite_posterior + no_prior_posterior + data_logplr).backward()
        assert x.grad is not None
        assert bool(torch.isfinite(x.grad))


def test_analytic_inference_models_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    samples = np.array([
        [-1.0, -0.4, 0.2, 0.8, 1.3, 1.7],
        [0.5, -0.7, 0.9, -0.2, 0.3, 1.1],
    ])

    posterior = inference_analytic.TestPosterior.__new__(
        inference_analytic.TestPosterior
    )
    inference_model_base.BaseModel.__init__(posterior, ("x", "y"))
    posterior._set_kde(scipy.stats.gaussian_kde(samples))
    models = [
        inference_analytic.TestNormal(
            ("x", "y"),
            mean=(0.2, -0.1),
            cov=((1.3, 0.2), (0.2, 0.7)),
        ),
        inference_analytic.TestEggbox(("x", "y")),
        inference_analytic.TestRosenbrock(("x", "y")),
        inference_analytic.TestRosenbrock(("x",)),
        inference_analytic.TestVolcano(("x", "y")),
        inference_analytic.TestPrior(("x", "y")),
        posterior,
    ]
    scalar_params = {"x": 0.35, "y": -0.4}
    expected = []
    for model in models:
        model.update(**{
            name: scalar_params[name] for name in model.variable_params
        })
        expected.append(model.loglikelihood)

    with ctx:
        x = torch.tensor(
            scalar_params["x"], device=device, dtype=dtype,
            requires_grad=True,
        )
        y = torch.tensor(
            scalar_params["y"], device=device, dtype=dtype,
            requires_grad=True,
        )
        tensor_params = {"x": x, "y": y}
        for model in models:
            model.update(**{
                name: tensor_params[name] for name in model.variable_params
            })

        def reject_host_evaluation(*_args, **_kwargs):
            raise AssertionError("analytic likelihood evaluated on the host")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_evaluation)
            patch.setattr(torch.Tensor, "numpy", reject_host_evaluation)
            patch.setattr(models[0]._dist, "logpdf", reject_host_evaluation)
            patch.setattr(posterior.kde, "logpdf", reject_host_evaluation)
            patch.setattr(
                inference_analytic.numpy, "cos", reject_host_evaluation
            )
            patch.setattr(
                inference_analytic.numpy, "prod", reject_host_evaluation
            )
            patch.setattr(
                inference_analytic.numpy, "sqrt", reject_host_evaluation
            )
            patch.setattr(
                inference_analytic.numpy, "exp", reject_host_evaluation
            )
            patch.setattr(
                inference_analytic.numpy, "array", reject_host_evaluation
            )
            actual = [model.loglikelihood for model in models]

        assert all(isinstance(value, torch.Tensor) for value in actual)
        assert all(value.device.type == device for value in actual)
        assert all(value.dtype == dtype for value in actual)
        for value, target in zip(actual, expected):
            assert torch.allclose(
                value.detach(), value.new_tensor(target),
                rtol=tolerance, atol=tolerance,
            )

        sum(actual).backward()
        assert x.grad is not None and bool(torch.isfinite(x.grad))
        assert y.grad is not None and bool(torch.isfinite(y.grad))


def test_common_priors_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12

    uniform_prior = distributions.Uniform(u=(-2.0, 2.0), v=(0.0, 1.0))
    gaussian_prior = distributions.Gaussian(
        x=(-2.0, 2.0), x_mean=0.25, x_var=1.5
    )
    power_prior = distributions.UniformPowerLaw(dim=3, r=(1.0, 4.0))
    log_prior = distributions.UniformLog10(scale=(1.0, 100.0))
    joint_prior = distributions.JointDistribution(
        ("x", "r"), gaussian_prior, power_prior
    )
    tensor_constraint = distributions.constraints.Constraint(
        "(x >= -1.0) & (r < 3.0)"
    )
    angular_constraint = distributions.constraints.Constraint(
        "(dec + ddec >= -pi/2) & (dec + ddec <= pi/2)"
    )
    shifted_x = transforms.CustomTransform(
        ["x"], ["shifted_x"], {"shifted_x": "x + 0.5"}
    )
    transformed_constraint = distributions.constraints.Constraint(
        "shifted_x < 1.0", transforms=[shifted_x]
    )

    uniform_values = {
        "u": np.array([-3.0, 0.0, 1.0]),
        "v": np.array([0.5, 0.5, 1.0]),
    }
    x_values = np.array([-1.5, -0.25, 0.75])
    unit_values = np.array([0.05, 0.5, 0.95])
    r_values = np.array([1.1, 2.0, 3.5])
    scale_values = np.array([1.0, 10.0, 99.0])
    bounded_scale_values = np.array([0.5, 1.0, 10.0, 100.0])

    expected_uniform_log = np.array([
        uniform_prior.logpdf(
            u=uniform_values["u"][index],
            v=uniform_values["v"][index],
        )
        for index in range(3)
    ])
    expected_gaussian_log = np.array([
        gaussian_prior.logpdf(x=value) for value in x_values
    ])
    expected_gaussian_pdf = np.array([
        gaussian_prior.pdf(x=value) for value in x_values
    ])
    expected_gaussian_cdf = gaussian_prior.cdf("x", x_values)
    expected_gaussian_cdfinv = gaussian_prior.cdfinv(
        x=unit_values
    )["x"]
    expected_power_log = np.array([
        power_prior.logpdf(r=value) for value in r_values
    ])
    expected_power_pdf = np.array([
        power_prior.pdf(r=value) for value in r_values
    ])
    expected_scale_log = np.array([
        log_prior.logpdf(scale=value) for value in scale_values
    ])
    expected_bounded_scale_log = np.array([
        log_prior.logpdf(scale=value) for value in bounded_scale_values
    ])
    expected_bounded_scale_pdf = np.array([
        log_prior.pdf(scale=value) for value in bounded_scale_values
    ])
    expected_joint = np.array([
        joint_prior(x=x_value, r=r_value)
        for x_value, r_value in zip(x_values, r_values)
    ])

    with ctx:
        uniform_tensors = {
            name: torch.as_tensor(values, device=device, dtype=dtype)
            for name, values in uniform_values.items()
        }
        x = torch.tensor(
            x_values, device=device, dtype=dtype, requires_grad=True
        )
        cdf_x = torch.tensor(
            x_values, device=device, dtype=dtype, requires_grad=True
        )
        unit = torch.tensor(
            unit_values, device=device, dtype=dtype, requires_grad=True
        )
        radius = torch.tensor(
            r_values, device=device, dtype=dtype, requires_grad=True
        )
        scale = torch.tensor(
            scale_values, device=device, dtype=dtype, requires_grad=True
        )
        bounded_scale = torch.as_tensor(
            bounded_scale_values, device=device, dtype=dtype
        )
        outside_x = torch.tensor(
            [-2.1, 0.0, 1.9], device=device, dtype=dtype
        )
        outside_radius = torch.tensor(
            [2.0, 4.1, 2.0], device=device, dtype=dtype
        )
        dec = torch.tensor(
            [-1.4, 0.0, 1.4], device=device, dtype=dtype
        )
        ddec = torch.tensor(
            [-0.3, 0.2, 0.3], device=device, dtype=dtype
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("prior evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                distribution_gaussian.numpy, "array", reject_host_or_numpy
            )
            patch.setattr(
                distribution_gaussian.numpy, "exp", reject_host_or_numpy
            )
            patch.setattr(
                distribution_gaussian.numpy, "log", reject_host_or_numpy
            )
            patch.setattr(
                distribution_gaussian.numpy, "prod", reject_host_or_numpy
            )
            patch.setattr(
                distribution_gaussian, "erf", reject_host_or_numpy
            )
            patch.setattr(
                distribution_gaussian, "erfinv", reject_host_or_numpy
            )
            patch.setattr(
                distribution_joint.numpy, "ones", reject_host_or_numpy
            )
            patch.setattr(
                distribution_joint.numpy, "array", reject_host_or_numpy
            )
            uniform_log = uniform_prior.logpdf(**uniform_tensors)
            gaussian_log = gaussian_prior.logpdf(x=x)
            gaussian_pdf = gaussian_prior.pdf(x=x)
            gaussian_cdf = gaussian_prior.cdf("x", cdf_x)
            gaussian_cdfinv = gaussian_prior.cdfinv(x=unit)["x"]
            power_log = power_prior.logpdf(r=radius)
            power_pdf = power_prior.pdf(r=radius)
            scale_log = log_prior.logpdf(scale=scale)
            bounded_scale_log = log_prior.logpdf(scale=bounded_scale)
            bounded_scale_pdf = log_prior.pdf(scale=bounded_scale)
            joint_log = joint_prior(x=x, r=radius)
            boundary_mask = joint_prior.contains(
                {"x": outside_x, "r": outside_radius}
            )
            angular_mask = angular_constraint({"dec": dec, "ddec": ddec})
            transformed_mask = transformed_constraint({"x": x})
            joint_prior._constraints = [tensor_constraint]
            constraint_mask = joint_prior.within_constraints(
                {"x": x, "r": radius}
            )
            containment_mask = joint_prior.contains(
                {"x": x, "r": radius}
            )
            scalar_containment = joint_prior.contains(
                {"x": x[1], "r": radius[1]}
            )
            constrained_joint_log = joint_prior(x=x, r=radius)
            joint_prior._constraints = []

        results = (
            uniform_log,
            gaussian_log,
            gaussian_pdf,
            gaussian_cdf,
            gaussian_cdfinv,
            power_log,
            power_pdf,
            scale_log,
            bounded_scale_log,
            bounded_scale_pdf,
            joint_log,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        assert angular_mask.dtype == torch.bool
        assert angular_mask.device.type == device
        assert angular_mask.tolist() == [False, True, False]
        assert transformed_mask.dtype == torch.bool
        assert transformed_mask.device.type == device
        assert transformed_mask.tolist() == [True, True, False]
        assert constraint_mask.dtype == torch.bool
        assert constraint_mask.device.type == device
        assert constraint_mask.tolist() == [False, True, False]
        assert boundary_mask.dtype == torch.bool
        assert boundary_mask.device.type == device
        assert boundary_mask.tolist() == [False, False, True]
        assert containment_mask.dtype == torch.bool
        assert containment_mask.device.type == device
        assert containment_mask.tolist() == [False, True, False]
        assert scalar_containment.dtype == torch.bool
        assert scalar_containment.device.type == device
        assert scalar_containment.ndim == 0
        assert bool(scalar_containment)
        assert constrained_joint_log.device.type == device
        assert torch.equal(
            torch.isfinite(constrained_joint_log), constraint_mask
        )
        assert torch.allclose(
            constrained_joint_log[constraint_mask],
            joint_log[constraint_mask],
            rtol=tolerance,
            atol=tolerance,
        )

        expectations = (
            expected_uniform_log,
            expected_gaussian_log,
            expected_gaussian_pdf,
            expected_gaussian_cdf,
            expected_gaussian_cdfinv,
            expected_power_log,
            expected_power_pdf,
            expected_scale_log,
            expected_bounded_scale_log,
            expected_bounded_scale_pdf,
            expected_joint,
        )
        for actual, expected in zip(results, expectations):
            assert np.allclose(
                actual.detach().tolist(),
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )

        (joint_log.sum() + scale_log.sum()).backward()
        (gaussian_cdf.sum() + gaussian_cdfinv.sum()).backward()
        assert torch.allclose(
            x.grad,
            -(x.detach() - 0.25) / 1.5,
            rtol=tolerance,
            atol=tolerance,
        )
        assert torch.allclose(
            radius.grad,
            2.0 / radius.detach(),
            rtol=tolerance,
            atol=tolerance,
        )
        assert cdf_x.grad is not None
        assert bool(torch.all(torch.isfinite(cdf_x.grad)))
        assert unit.grad is not None
        assert bool(torch.all(torch.isfinite(unit.grad)))
        assert torch.allclose(
            scale.grad,
            -1.0 / scale.detach(),
            rtol=tolerance,
            atol=tolerance,
        )

        joint_prior._constraints = [object()]
        with pytest.raises(TypeError, match="raw Torch tensors"):
            joint_prior(x=x, r=radius)


def test_kde_priors_stay_on_torch_device(torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 4e-4 if dtype == torch.float32 else 1e-11

    rng = np.random.default_rng(1984)
    samples_x = rng.normal(size=96)
    samples_y = np.clip(
        0.5 + 0.14 * samples_x + 0.1 * rng.normal(size=96),
        0.04,
        0.96,
    )
    prior = distributions.Arbitrary(
        bounds={"y": (0.0, 1.0)}, x=samples_x, y=samples_y
    )
    spin_prior = distributions.IndependentChiPChiEff(
        mass1=(10.0, 20.0),
        mass2=(5.0, 10.0),
        nsamples=96,
        seed=17,
    )

    x_values = np.array([-1.0, 0.25, 1.4])[:, None]
    y_values = np.array([0.0, 0.2, 0.7, 1.0])[None, :]
    expected_pdf = np.array([
        [prior.pdf(x=float(x), y=float(y)) for y in y_values[0]]
        for x in x_values[:, 0]
    ])
    expected_log = np.array([
        [prior.logpdf(x=float(x), y=float(y)) for y in y_values[0]]
        for x in x_values[:, 0]
    ])
    spin_values = {
        "mass1": 15.0,
        "mass2": 8.0,
        "xi1": 0.2,
        "xi2": 0.1,
        "chi_eff": 0.0,
        "chi_a": 0.0,
        "phi_a": 1.0,
        "phi_s": 1.5,
    }
    expected_spin_log = spin_prior.logpdf(**spin_values)

    with ctx:
        x = torch.tensor(
            x_values, device=device, dtype=dtype, requires_grad=True
        )
        y = torch.as_tensor(y_values, device=device, dtype=dtype)
        spin_tensors = {
            name: torch.tensor(
                value,
                device=device,
                dtype=dtype,
                requires_grad=name == "chi_eff",
            )
            for name, value in spin_values.items()
        }
        spin_contains_tensors = {
            name: torch.tensor(
                [value, 25.0 if name == "mass1" else value],
                device=device,
                dtype=dtype,
            )
            for name, value in spin_values.items()
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("KDE prior evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(prior.kde, "evaluate", reject_host_or_numpy)
            patch.setattr(spin_prior.kde, "evaluate", reject_host_or_numpy)
            patch.setattr(
                distribution_arbitrary.numpy, "log", reject_host_or_numpy
            )
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            for transform in (
                    list(prior._transforms.values())
                    + list(spin_prior._transforms.values())):
                patch.setattr(transform, "transform", reject_host_or_numpy)
                patch.setattr(transform, "jacobian", reject_host_or_numpy)
            actual_pdf = prior.pdf(x=x, y=y)
            actual_log = prior.logpdf(x=x, y=y)
            empty_log = prior.logpdf(x=x[:0], y=y[:, 1:3])
            actual_spin_log = spin_prior.logpdf(**spin_tensors)
            actual_spin_contains = spin_prior.__contains__(
                spin_contains_tensors
            )

        results = (actual_pdf, actual_log, empty_log, actual_spin_log)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        assert empty_log.shape == (0, 2)
        assert actual_spin_contains.dtype == torch.bool
        assert actual_spin_contains.device.type == device
        assert actual_spin_contains.tolist() == [True, False]
        assert np.allclose(
            actual_pdf.detach().tolist(),
            expected_pdf,
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.allclose(
            actual_log.detach().tolist(),
            expected_log,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
        assert np.isclose(
            actual_spin_log.detach().item(),
            expected_spin_log,
            rtol=5 * tolerance,
            atol=5 * tolerance,
        )
        assert torch.equal(
            actual_pdf[:, (0, 3)], torch.zeros_like(actual_pdf[:, (0, 3)])
        )
        assert torch.isneginf(actual_log[:, (0, 3)]).all()

        (actual_log[:, 1:3].sum() + actual_spin_log).backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(spin_tensors["chi_eff"].grad)

    assert prior._torch_kde_cache
    prior.set_bandwidth("silverman")
    assert not prior._torch_kde_cache


def test_tabulated_prior_stays_on_torch_device(
        torch_device_ctx, monkeypatch, tmp_path):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 1e-12

    knots = np.array([-2.0, 0.0, 1.0, 3.0])
    density = np.array([1.0, 3.0, 2.0, 1.0])
    density_file = tmp_path / "tabulated-prior.txt"
    np.savetxt(density_file, np.column_stack((knots, density)))
    prior = distributions.DistributionFunctionFromFile(
        params=["x"], file_path=density_file, column_index=1
    )

    values = np.array([-3.0, -2.0, -1.0, 0.5, 2.0, 3.0, 4.0])
    expected_pdf = np.array([prior._pdf(value) for value in values])
    with np.errstate(divide="ignore"):
        expected_logpdf = np.log(expected_pdf)
    cdf_values = np.array([-1.75, -0.5, 0.75, 2.5])
    unit_values = np.array([0.0, 0.09, 0.41, 0.83, 1.0])
    expected_cdf = prior._cdf(cdf_values)
    expected_cdfinv = prior.cdfinv(x=unit_values)["x"]

    def reject_host_or_scipy(*_args, **_kwargs):
        raise AssertionError("tabulated prior evaluation left Torch")

    # Normalization and table construction are intentionally CPU setup work.
    # Once cached, repeated likelihood evaluation must stay on-device.
    prior.interp["pdf"] = reject_host_or_scipy
    prior.interp["cdf"] = reject_host_or_scipy
    prior.interp["cdfinv"] = reject_host_or_scipy
    with ctx:
        x = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )
        differentiable_x = torch.tensor(
            [-1.0, 0.5, 2.0],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        cdf_x = torch.tensor(
            cdf_values, device=device, dtype=dtype, requires_grad=True
        )
        unit = torch.tensor(
            unit_values, device=device, dtype=dtype, requires_grad=True
        )
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(
                distribution_external.scipy_integrate,
                "quad",
                reject_host_or_scipy,
            )
            patch.setattr(
                distribution_external.scipy_interpolate,
                "interp1d",
                reject_host_or_scipy,
            )
            actual_pdf = prior._pdf(x)
            actual_logpdf = prior.logpdf(x=x)
            differentiable_logpdf = prior.logpdf(x=differentiable_x)
            actual_cdf = prior._cdf(cdf_x)
            actual_cdfinv = prior.cdfinv(x=unit)["x"]

        differentiable_logpdf.sum().backward()
        actual_cdf.sum().backward()
        actual_cdfinv[1:-1].sum().backward()

    assert actual_pdf.device.type == device
    assert actual_logpdf.device.type == device
    assert actual_pdf.dtype == dtype
    assert actual_logpdf.dtype == dtype
    assert actual_cdf.device.type == device
    assert actual_cdfinv.device.type == device
    assert actual_cdf.dtype == dtype
    assert actual_cdfinv.dtype == dtype
    assert np.allclose(
        actual_pdf.detach().tolist(),
        expected_pdf,
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        actual_logpdf.detach().tolist(),
        expected_logpdf,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    assert np.allclose(
        actual_cdf.detach().tolist(),
        expected_cdf,
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        actual_cdfinv.detach().tolist(),
        expected_cdfinv,
        rtol=tolerance,
        atol=tolerance,
    )
    expected_gradient = torch.tensor(
        [0.5, -0.4, -1.0 / 3.0], device=device, dtype=dtype
    )
    assert torch.allclose(
        differentiable_x.grad,
        expected_gradient,
        rtol=tolerance,
        atol=tolerance,
    )
    assert cdf_x.grad is not None
    assert bool(torch.isfinite(cdf_x.grad).all())
    assert bool((cdf_x.grad > 0).all())
    assert unit.grad is not None
    assert bool(torch.isfinite(unit.grad[1:-1]).all())
    assert bool((unit.grad[1:-1] > 0).all())

    with ctx:
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            prior.cdfinv(x=torch.tensor([-0.01], device=device))


def test_angular_priors_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12

    sin_prior = distributions.SinAngle(theta=None)
    cos_prior = distributions.CosAngle(declination=None)
    angle_prior = distributions.UniformAngle(phi=None, cyclic_domain=True)
    solid_prior = distributions.UniformSolidAngle(
        azimuthal_cyclic_domain=True
    )

    theta_values = np.array([0.4, 1.2, 2.6])
    declination_values = np.array([-1.1, 0.1, 1.0])
    phi_values = np.array([0.2, 2.0, 2.0 * np.pi + 0.3])
    outside_values = np.array([-0.1, 0.5, np.pi + 0.1])

    expected_sin_pdf = np.array([
        sin_prior.pdf(theta=value) for value in theta_values
    ])
    expected_sin_log = np.array([
        sin_prior.logpdf(theta=value) for value in theta_values
    ])
    expected_cos_pdf = np.array([
        cos_prior.pdf(declination=value)
        for value in declination_values
    ])
    expected_cos_log = np.array([
        cos_prior.logpdf(declination=value)
        for value in declination_values
    ])
    expected_angle_log = np.array([
        angle_prior.logpdf(phi=value) for value in phi_values
    ])
    expected_solid_pdf = np.array([
        solid_prior.pdf(theta=theta, phi=phi)
        for theta, phi in zip(theta_values, phi_values)
    ])
    expected_solid_log = np.array([
        solid_prior.logpdf(theta=theta, phi=phi)
        for theta, phi in zip(theta_values, phi_values)
    ])
    expected_outside_pdf = np.array([
        sin_prior.pdf(theta=value) for value in outside_values
    ])
    expected_outside_log = np.array([
        sin_prior.logpdf(theta=value) for value in outside_values
    ])

    with ctx:
        theta = torch.tensor(
            theta_values, device=device, dtype=dtype, requires_grad=True
        )
        declination = torch.tensor(
            declination_values,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        phi = torch.as_tensor(phi_values, device=device, dtype=dtype)
        solid_theta = torch.tensor(
            theta_values, device=device, dtype=dtype, requires_grad=True
        )
        outside = torch.as_tensor(
            outside_values, device=device, dtype=dtype
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("angular prior evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                distribution_angular.SinAngle, "_dfunc", reject_host_or_numpy
            )
            patch.setattr(
                distribution_angular.CosAngle, "_dfunc", reject_host_or_numpy
            )
            patch.setattr(
                distribution_angular.numpy, "array", reject_host_or_numpy
            )
            patch.setattr(
                distribution_angular.numpy, "log", reject_host_or_numpy
            )
            sin_pdf = sin_prior.pdf(theta=theta)
            sin_log = sin_prior.logpdf(theta=theta)
            cos_pdf = cos_prior.pdf(declination=declination)
            cos_log = cos_prior.logpdf(declination=declination)
            angle_log = angle_prior.logpdf(phi=phi)
            solid_pdf = solid_prior.pdf(theta=solid_theta, phi=phi)
            solid_log = solid_prior.logpdf(theta=solid_theta, phi=phi)
            outside_pdf = sin_prior.pdf(theta=outside)
            outside_log = sin_prior.logpdf(theta=outside)

        results = (
            sin_pdf,
            sin_log,
            cos_pdf,
            cos_log,
            angle_log,
            solid_pdf,
            solid_log,
            outside_pdf,
            outside_log,
        )
        expectations = (
            expected_sin_pdf,
            expected_sin_log,
            expected_cos_pdf,
            expected_cos_log,
            expected_angle_log,
            expected_solid_pdf,
            expected_solid_log,
            expected_outside_pdf,
            expected_outside_log,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual, expected in zip(results, expectations):
            assert np.allclose(
                actual.detach().tolist(),
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )

        (sin_log.sum() + cos_log.sum() + solid_log.sum()).backward()
        assert torch.allclose(
            theta.grad,
            1.0 / torch.tan(theta.detach()),
            rtol=tolerance,
            atol=tolerance,
        )
        assert torch.allclose(
            declination.grad,
            -torch.tan(declination.detach()),
            rtol=tolerance,
            atol=tolerance,
        )
        assert torch.allclose(
            solid_theta.grad,
            1.0 / torch.tan(solid_theta.detach()),
            rtol=tolerance,
            atol=tolerance,
        )


def test_prior_inverse_cdfs_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 1e-12

    log_prior = distributions.UniformLog10(scale=(0.1, 100.0))
    sin_prior = distributions.SinAngle(theta=(0.2, 2.7))
    cos_prior = distributions.CosAngle(declination=(-1.2, 1.1))
    solid_prior = distributions.UniformSolidAngle(
        polar_bounds=(0.1, 0.8), azimuthal_bounds=(0.2, 1.7)
    )
    unit_values = np.array([0.05, 0.27, 0.61, 0.94])
    azimuthal_values = np.array([0.91, 0.48, 0.13, 0.72])
    expected = (
        log_prior.cdfinv(scale=unit_values)["scale"],
        sin_prior.cdfinv(theta=unit_values)["theta"],
        cos_prior.cdfinv(declination=unit_values)["declination"],
    )
    expected_solid = solid_prior.cdfinv(
        theta=unit_values, phi=azimuthal_values
    )

    with ctx:
        unit = torch.tensor(
            unit_values, device=device, dtype=dtype, requires_grad=True
        )
        azimuthal = torch.tensor(
            azimuthal_values,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("prior inverse CDF left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                distribution_uniform_log.numpy,
                "log10",
                reject_host_or_numpy,
            )
            for name in ("sin", "cos", "arcsin", "arccos"):
                patch.setattr(
                    distribution_angular.numpy, name, reject_host_or_numpy
                )
            actual = (
                log_prior.cdfinv(scale=unit)["scale"],
                sin_prior.cdfinv(theta=unit)["theta"],
                cos_prior.cdfinv(declination=unit)["declination"],
            )
            actual_solid = solid_prior.cdfinv(
                theta=unit, phi=azimuthal
            )

        results = actual + (
            actual_solid["theta"],
            actual_solid["phi"],
        )
        expectations = expected + (
            expected_solid["theta"],
            expected_solid["phi"],
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual_value, expected_value in zip(results, expectations):
            assert np.allclose(
                actual_value.detach().tolist(),
                expected_value,
                rtol=tolerance,
                atol=tolerance,
            )

        sum(value.sum() for value in results).backward()
        assert unit.grad is not None
        assert azimuthal.grad is not None
        assert bool(torch.isfinite(unit.grad).all())
        assert bool(torch.isfinite(azimuthal.grad).all())


def test_mass_ratio_prior_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    prior = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    values = np.array([1.0, 2.5, 7.5])
    outside_values = np.array([0.5, 2.0, 8.0])
    expected_pdf = np.array([
        prior.pdf(q=value) for value in values
    ])
    expected_log = np.array([
        prior.logpdf(q=value) for value in values
    ])
    expected_outside_pdf = np.array([
        prior.pdf(q=value) for value in outside_values
    ])
    expected_outside_log = np.array([
        prior.logpdf(q=value) for value in outside_values
    ])

    with ctx:
        q = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )
        outside = torch.as_tensor(
            outside_values, device=device, dtype=dtype
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("mass-ratio prior evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                distribution_mass.numpy, "prod", reject_host_or_numpy
            )
            patch.setattr(
                distribution_mass.numpy, "log", reject_host_or_numpy
            )
            pdf = prior.pdf(q=q)
            logpdf = prior.logpdf(q=q)
            outside_pdf = prior.pdf(q=outside)
            outside_log = prior.logpdf(q=outside)

        results = (pdf, logpdf, outside_pdf, outside_log)
        expectations = (
            expected_pdf,
            expected_log,
            expected_outside_pdf,
            expected_outside_log,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual, expected in zip(results, expectations):
            assert np.allclose(
                actual.detach().tolist(),
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )

        logpdf.sum().backward()
        expected_grad = 2.0 / (5.0 * (1.0 + q.detach())) \
            - 6.0 / (5.0 * q.detach())
        assert torch.allclose(
            q.grad, expected_grad, rtol=tolerance, atol=tolerance
        )


def test_mass_ratio_prior_cdfinv_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    prior = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    unit_values = np.array([0.0, 0.07, 0.31, 0.73, 0.999, 1.0])
    expected = prior.cdfinv(q=unit_values)["q"]
    assert expected[0] == 1.0
    assert expected[-1] == 8.0

    with ctx:
        unit = torch.tensor(
            unit_values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_scipy(*_args, **_kwargs):
            raise AssertionError("mass-ratio inverse CDF left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_scipy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_scipy)
            patch.setattr(
                distribution_mass.numpy, "asarray", reject_host_or_scipy
            )
            patch.setattr(
                distribution_mass, "interp1d", reject_host_or_scipy
            )
            patch.setattr(
                distribution_mass, "CubicSpline", reject_host_or_scipy
            )
            patch.setattr(
                distribution_mass, "hyp2f1", reject_host_or_scipy
            )
            actual = prior.cdfinv(q=unit)["q"]

        assert isinstance(actual, torch.Tensor)
        assert actual.device.type == device
        assert actual.dtype == dtype
        assert np.allclose(
            actual.detach().tolist(), expected,
            rtol=tolerance, atol=tolerance,
        )

        actual[1:-1].sum().backward()
        assert unit.grad is not None
        assert bool(torch.isfinite(unit.grad).all())
        assert bool((unit.grad[1:-1] > 0).all())

        with pytest.raises(ValueError, match=r"input in \[0,1\]"):
            prior.cdfinv(q=torch.tensor([-0.01], device=device))


def test_qnm_conversions_and_prior_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-4 if dtype == torch.float32 else 1e-11
    f0_values = np.array([200.0, 250.0, 300.0])
    tau_values = np.array([0.004, 0.004, 0.004])
    expected_spin = conversions.final_spin_from_f0_tau(
        f0_values, tau_values
    )
    expected_mass = conversions.final_mass_from_f0_tau(
        f0_values, tau_values
    )
    expected_mode_f0 = conversions.freqlmn_from_other_lmn(
        f0_values, tau_values, 2, 2, 3, 3
    )
    expected_mode_tau = conversions.taulmn_from_other_lmn(
        f0_values, tau_values, 2, 2, 3, 3
    )
    prior = distribution_qnm.UniformF0Tau(
        f0=(100.0, 500.0),
        tau=(0.001, 0.02),
        final_mass=(50.0, 150.0),
        final_spin=(0.0, 0.95),
        norm_tolerance=0.1,
    )
    expected_pdf = np.array([
        prior.pdf(f0=f0, tau=tau)
        for f0, tau in zip(f0_values, tau_values)
    ])
    expected_log = np.array([
        prior.logpdf(f0=f0, tau=tau)
        for f0, tau in zip(f0_values, tau_values)
    ])
    outside_f0 = np.array([450.0, 80.0])
    outside_tau = np.array([0.001, 0.004])
    expected_outside_pdf = np.array([
        prior.pdf(f0=f0, tau=tau)
        for f0, tau in zip(outside_f0, outside_tau)
    ])
    expected_outside_log = np.array([
        prior.logpdf(f0=f0, tau=tau)
        for f0, tau in zip(outside_f0, outside_tau)
    ])

    with ctx:
        f0 = torch.tensor(
            f0_values, device=device, dtype=dtype, requires_grad=True
        )
        tau = torch.tensor(
            tau_values, device=device, dtype=dtype, requires_grad=True
        )
        outside_f0_tensor = torch.as_tensor(
            outside_f0, device=device, dtype=dtype
        )
        outside_tau_tensor = torch.as_tensor(
            outside_tau, device=device, dtype=dtype
        )
        invalid_f0 = torch.tensor([500.0], device=device, dtype=dtype)
        invalid_tau = torch.tensor([0.1], device=device, dtype=dtype)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("QNM prior evaluation left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(
                conversions.numpy, "zeros", reject_host_or_numpy
            )
            patch.setattr(
                conversions.pykerr, "qnmfreq", reject_host_or_numpy
            )
            patch.setattr(
                conversions.pykerr, "qnmtau", reject_host_or_numpy
            )
            spin = conversions.final_spin_from_f0_tau(f0, tau)
            mass = conversions.final_mass_from_f0_tau(f0, tau)
            mode_f0 = conversions.freqlmn_from_other_lmn(
                f0, tau, 2, 2, 3, 3
            )
            mode_tau = conversions.taulmn_from_other_lmn(
                f0, tau, 2, 2, 3, 3
            )
            invalid_mode_f0 = conversions.freqlmn_from_other_lmn(
                invalid_f0, invalid_tau, 2, 2, 3, 3
            )
            invalid_mode_tau = conversions.taulmn_from_other_lmn(
                invalid_f0, invalid_tau, 2, 2, 3, 3
            )
            pdf = prior.pdf(f0=f0, tau=tau)
            logpdf = prior.logpdf(f0=f0, tau=tau)
            outside_pdf = prior.pdf(
                f0=outside_f0_tensor, tau=outside_tau_tensor
            )
            outside_log = prior.logpdf(
                f0=outside_f0_tensor, tau=outside_tau_tensor
            )

        results = (
            spin, mass, mode_f0, mode_tau,
            pdf, logpdf, outside_pdf, outside_log,
        )
        expectations = (
            expected_spin, expected_mass, expected_mode_f0, expected_mode_tau,
            expected_pdf,
            expected_log,
            expected_outside_pdf,
            expected_outside_log,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual, expected in zip(results, expectations):
            assert np.allclose(
                actual.detach().tolist(),
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )
        assert bool(torch.isnan(invalid_mode_f0).all())
        assert bool(torch.isnan(invalid_mode_tau).all())

        (spin.sum() + mass.sum() + mode_f0.sum() + mode_tau.sum()).backward()
        assert torch.isfinite(f0.grad).all()
        assert torch.isfinite(tau.grad).all()


def test_evidence_mean_estimators_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([-1000.0, -3.0, -1.0, 2.0])
    expected_arithmetic = scipy.special.logsumexp(values) - np.log(
        values.size
    )
    expected_harmonic = np.log(values.size) - scipy.special.logsumexp(
        -values
    )

    with ctx:
        log_likelihood = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("evidence estimator left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(inference_evidence.numpy, "max", reject_host_or_numpy)
            patch.setattr(inference_evidence.numpy, "exp", reject_host_or_numpy)
            patch.setattr(inference_evidence.numpy, "log", reject_host_or_numpy)
            arithmetic = inference_evidence.arithmetic_mean_estimator(
                log_likelihood
            )
            harmonic = inference_evidence.harmonic_mean_estimator(
                log_likelihood
            )

        assert arithmetic.device.type == device
        assert harmonic.device.type == device
        tolerance = 3e-5 if dtype == torch.float32 else 1e-12
        assert np.isclose(
            arithmetic.detach().item(), expected_arithmetic,
            rtol=tolerance, atol=tolerance,
        )
        assert np.isclose(
            harmonic.detach().item(), expected_harmonic,
            rtol=tolerance, atol=tolerance,
        )

        (arithmetic + harmonic).backward()
        assert log_likelihood.grad is not None
        assert torch.isfinite(log_likelihood.grad).all()


def test_geweke_statistic_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([
        -1.2, 0.4, 1.1, -0.7, 0.2, 1.6, -0.3, 0.8, 1.4, -0.5,
    ])
    expected = inference_geweke.geweke(values, 3, 2, 6, 6)

    with ctx:
        chain = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("Geweke statistic left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_geweke.numpy, "arange", reject_host_or_numpy
            )
            starts, ends, stats = inference_geweke.geweke(
                chain, 3, 2, 6, 6
            )

        assert starts.device.type == device
        assert ends.device.type == device
        assert stats.device.type == device
        np.testing.assert_array_equal(starts.tolist(), expected[0])
        np.testing.assert_array_equal(ends.tolist(), expected[1])
        tolerance = 3e-5 if dtype == torch.float32 else 1e-12
        np.testing.assert_allclose(
            stats.detach().tolist(), expected[2],
            rtol=tolerance, atol=tolerance,
        )

        stats.sum().backward()
        assert chain.grad is not None
        assert torch.isfinite(chain.grad).all()


def test_gelman_rubin_statistic_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    rng = np.random.default_rng(1234)
    values = (
        rng.normal(size=(4, 3, 40))
        + np.arange(4)[:, None, None] * 0.08
    )
    expected = inference_gelman_rubin.gelman_rubin(values, False)

    with ctx:
        chains = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("Gelman-Rubin statistic left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_gelman_rubin.numpy, "array", reject_host_or_numpy
            )
            patch.setattr(
                inference_gelman_rubin.numpy, "cov", reject_host_or_numpy
            )
            psrf = inference_gelman_rubin.gelman_rubin(chains, False)

        assert psrf.device.type == device
        tolerance = 5e-5 if dtype == torch.float32 else 1e-12
        np.testing.assert_allclose(
            psrf.detach().tolist(), expected,
            rtol=tolerance, atol=tolerance,
        )
        psrf.sum().backward()
        assert chains.grad is not None
        assert torch.isfinite(chains.grad).all()


def test_burn_in_helpers_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([
        [-8.0, -7.0, -6.0, -5.0],
        [-12.0, -11.0, -10.0, -9.0],
        [-7.5, -6.5, -4.0, -3.0],
    ])
    expected_max = inference_burn_in.max_posterior(values, 4)
    expected_step = inference_burn_in.posterior_step(values[0], 2)

    with ctx:
        logposts = torch.tensor(values, device=device, dtype=dtype)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("burn-in helper left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_burn_in.numpy, "empty", reject_host_or_numpy
            )
            patch.setattr(
                inference_burn_in.numpy, "where", reject_host_or_numpy
            )
            patch.setattr(
                inference_burn_in.numpy, "diff", reject_host_or_numpy
            )
            burn_idx, burned = inference_burn_in.max_posterior(
                logposts, 4
            )
            step = inference_burn_in.posterior_step(logposts[0], 2)

        assert burn_idx.device.type == device
        assert burned.device.type == device
        assert step.device.type == device
        np.testing.assert_array_equal(burn_idx.tolist(), expected_max[0])
        np.testing.assert_array_equal(burned.tolist(), expected_max[1])
        assert step.item() == expected_step


def test_burn_in_ks_test_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values1 = np.array([0.0, 0.0, 0.5, 1.0, 1.5, 2.0])
    values2 = np.array([-0.5, 0.0, 0.5, 0.5, 1.0, 3.0, 4.0])
    expected_pvalue = scipy.stats.ks_2samp(values1, values2).pvalue
    threshold = expected_pvalue - 1e-7

    with ctx:
        samples1 = {
            "x": torch.tensor(values1, device=device, dtype=dtype)
        }
        samples2 = {
            "x": torch.tensor(values2, device=device, dtype=dtype)
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("KS samples left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(
                inference_burn_in, "ks_2samp", reject_host_or_numpy
            )
            result = inference_burn_in.ks_test(
                samples1, samples2, threshold=threshold
            )

        assert result == {"x": True}


def test_fgmc_analytic_density_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    trigs = np.array([5.0, 5.5, 6.25, 8.5], dtype=dtype)
    rho_min = np.array(5.0, dtype=dtype)
    expected = fgmc_functions.log_rho_fg_analytic(trigs, rho_min)

    with ctx:
        trig_tensor = torch.tensor(
            trigs,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        rho_tensor = torch.tensor(
            rho_min,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        torch_trigs = Array(TorchArrayData(trig_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("FGMC density vector left Torch")

        def reject_numpy_density(*_args, **_kwargs):
            raise AssertionError("FGMC density used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(fgmc_functions.np, "log", reject_numpy_density)
            actual = fgmc_functions.log_rho_fg_analytic(
                torch_trigs, rho_tensor
            )
            actual._data.tensor.sum().backward()

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch_dtype
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    for values in (trig_tensor, rho_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0


def test_fgmc_bin_filter_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([-1.0, 0.5, 1.5, 3.0, np.nan], dtype=dtype)
    expected = fgmc_functions.filter_bin_lo_hi(values, 0.0, 2.0)

    with ctx:
        value_tensor = torch.tensor(
            values, dtype=torch_dtype, device=device
        )
        torch_values = Array(TorchArrayData(value_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("FGMC bin mask left Torch")

        def reject_numpy_filter(*_args, **_kwargs):
            raise AssertionError("FGMC bin mask used NumPy")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(fgmc_functions.np, "any", reject_numpy_filter)
            patch.setattr(fgmc_functions.np, "sign", reject_numpy_filter)
            actual = fgmc_functions.filter_bin_lo_hi(
                torch_values, 0.0, 2.0
            )
            reversed_actual = fgmc_functions.filter_bin_lo_hi(
                value_tensor, 2.0, 0.0
            )

    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.bool
    assert actual.device.type == device
    assert torch.equal(actual, reversed_actual)
    np.testing.assert_array_equal(actual.detach().cpu().numpy(), expected)


def test_fgmc_bin_filter_rejects_exact_edges():
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="Edge case! Bin edges"):
        fgmc_functions.filter_bin_lo_hi(values, 0.0, 2.0)

    with scheme.TorchScheme("cpu"):
        tensor = torch.tensor(values)
        with pytest.raises(RuntimeError, match="Edge case! Bin edges"):
            fgmc_functions.filter_bin_lo_hi(tensor, 0.0, 2.0)


def test_fgmc_background_density_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    trigs = np.array([5.0, 5.5, 6.25, 8.5], dtype=dtype)
    counts = np.array([4.0, 0.0, 9.0], dtype=dtype)
    bins = np.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)
    expected_log, expected_error = fgmc_functions.log_rho_bg(
        trigs, counts.copy(), bins
    )

    with ctx:
        tensors = [
            torch.tensor(
                values,
                dtype=torch_dtype,
                device=device,
                requires_grad=True,
            )
            for values in (trigs, counts, bins)
        ]
        torch_inputs = [
            Array(TorchArrayData(values), copy=False) for values in tensors
        ]
        original_counts = tensors[1].detach().clone()
        empty = Array(
            TorchArrayData(torch.empty(0, dtype=torch_dtype, device=device)),
            copy=False,
        )

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("FGMC background vector left Torch")

        def reject_numpy_density(*_args, **_kwargs):
            raise AssertionError("FGMC background used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            for name in ("all", "any", "array", "atleast_1d", "log"):
                patch.setattr(
                    fgmc_functions.np, name, reject_numpy_density
                )
            actual_log, actual_error = fgmc_functions.log_rho_bg(
                *torch_inputs
            )
            empty_log, empty_error = fgmc_functions.log_rho_bg(
                empty, torch_inputs[1], torch_inputs[2]
            )
            (actual_log._data.tensor.sum()
             + actual_error._data.tensor.sum()).backward()

    for values in (actual_log, actual_error, empty_log, empty_error):
        assert isinstance(values, Array)
        assert values._data.tensor.device.type == device
        assert values._data.tensor.dtype == torch_dtype
    assert len(empty_log) == len(empty_error) == 0
    assert torch.equal(tensors[1].detach(), original_counts)
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual_log._data.tensor.detach().cpu().numpy(),
        expected_log,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_error._data.tensor.detach().cpu().numpy(),
        expected_error,
        rtol=tolerance,
        atol=tolerance,
    )
    for values in tensors:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0


def test_fgmc_injection_density_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    trigs = np.array([5.0, 5.5, 6.0, 6.25, 8.5], dtype=dtype)
    injections = np.array(
        [4.9, 5.0, 5.1, 5.2, 5.9, 6.0, 6.8, 7.1, 8.0, 8.2],
        dtype=dtype,
    )
    bins = np.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)
    expected_log, expected_error = fgmc_functions.log_rho_fg(
        trigs, injections, bins
    )

    with ctx:
        torch_inputs = [
            Array(
                TorchArrayData(
                    torch.tensor(values, dtype=torch_dtype, device=device)
                ),
                copy=False,
            )
            for values in (trigs, injections, bins)
        ]
        empty = Array(
            TorchArrayData(torch.empty(0, dtype=torch_dtype, device=device)),
            copy=False,
        )

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("FGMC injection vector left Torch")

        def reject_numpy_density(*_args, **_kwargs):
            raise AssertionError("FGMC injection density used NumPy")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            for name in (
                "array",
                "atleast_1d",
                "diff",
                "histogram",
                "log",
                "searchsorted",
                "where",
            ):
                patch.setattr(
                    fgmc_functions.np, name, reject_numpy_density
                )
            actual_log, actual_error = fgmc_functions.log_rho_fg(
                *torch_inputs
            )
            empty_result = fgmc_functions.log_rho_fg(
                empty, torch_inputs[1], torch_inputs[2]
            )

    for values in (actual_log, actual_error, empty_result):
        assert isinstance(values, Array)
        assert values._data.tensor.device.type == device
        assert values._data.tensor.dtype == torch_dtype
    assert len(empty_result) == 0
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual_log._data.tensor.detach().cpu().numpy(),
        expected_log,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_error._data.tensor.detach().cpu().numpy(),
        expected_error,
        rtol=tolerance,
        atol=tolerance,
    )


def test_rate_fgmc_density_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    trigs = np.array([0.0, 0.5, 1.0, 2.5, 3.0, 3.5], dtype=dtype)
    injections = np.array(
        [-1.0, 0.0, 0.2, 0.3, 0.9, 1.0, 1.5, 2.5,
         3.0, 3.5, 4.0, 5.0],
        dtype=dtype,
    )
    bins = np.array([0.0, 1.0, 3.0, 4.0], dtype=dtype)
    expected = rates_functions.log_rho_fgmc(trigs, injections, bins)

    with ctx:
        trig_tensor = torch.tensor(
            trigs, dtype=torch_dtype, device=device
        )
        injection_tensor = torch.tensor(
            injections, dtype=torch_dtype, device=device
        )
        bin_tensor = torch.tensor(
            bins, dtype=torch_dtype, device=device, requires_grad=True
        )
        torch_trigs = Array(TorchArrayData(trig_tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("rate density vector left Torch")

        def reject_numpy_density(*_args, **_kwargs):
            raise AssertionError("rate density used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            for name in ("diff", "histogram", "max", "min", "searchsorted"):
                patch.setattr(
                    rates_functions.np, name, reject_numpy_density
                )
            patch.setattr(rates_functions, "log", reject_numpy_density)
            actual = rates_functions.log_rho_fgmc(
                torch_trigs, injection_tensor, bin_tensor
            )
            scalar = rates_functions.log_rho_fgmc(
                trig_tensor[0], injection_tensor, bin_tensor
            )
            with pytest.raises(AssertionError):
                rates_functions.log_rho_fgmc(
                    bin_tensor[-1], injection_tensor, bin_tensor
                )
            (actual._data.tensor.sum() + scalar).backward()

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual._data.tensor.dtype == torch_dtype
    assert isinstance(scalar, torch.Tensor)
    assert scalar.ndim == 0
    assert scalar.device.type == device
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        scalar.detach().cpu().numpy(), expected[0],
        rtol=tolerance, atol=tolerance,
    )
    assert bin_tensor.grad is not None
    assert torch.isfinite(bin_tensor.grad).all()
    assert torch.count_nonzero(bin_tensor.grad) > 0


@pytest.mark.parametrize(
    ("model", "kwargs"),
    (
        (population_models.sfr_grb_2008, {}),
        (population_models.sfr_madau_dickinson_2014, {}),
        (population_models.sfr_madau_fragos_2017, {}),
        (population_models.sfr_madau_fragos_2017, {"mode": "low"}),
    ),
)
def test_population_sfr_models_stay_on_torch_device(
        torch_device_ctx, monkeypatch, model, kwargs):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    redshifts = np.array([0.0, 0.5, 1.0, 4.0], dtype=dtype)
    expected = model(redshifts, **kwargs)

    with ctx:
        tensor = torch.tensor(
            redshifts,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        wrapped_input = Array(TorchArrayData(tensor), copy=False)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("star-formation model left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            raw = model(tensor, **kwargs)
            wrapped = model(wrapped_input, **kwargs)
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


def test_population_lookback_derivative_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    import sympy

    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    redshifts = np.array([0.0, 0.5, 1.0, 4.0], dtype=dtype)
    cosmology_model = population_models.get_cosmology()
    hubble = (
        cosmology_model.H0.value
        * (3.0856776e19) ** -1
        / (1 / 24 / 3600 / 365 * 1e-9)
    )
    expected = (
        1 / hubble / (1 + redshifts)
        / np.sqrt(
            cosmology_model.Ode0
            + cosmology_model.Om0 * (1 + redshifts) ** 3
        )
    )

    with ctx:
        tensor = torch.tensor(
            redshifts,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        wrapped_input = Array(TorchArrayData(tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("lookback derivative left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(sympy, "sqrt", reject_host_path)
            raw = population_models.diff_lookback_time(tensor)
            wrapped = population_models.diff_lookback_time(wrapped_input)
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize(
    "td_model", ("log_normal", "gaussian", "power_law", "inverse")
)
def test_population_delay_models_stay_on_torch_device(
        torch_device_ctx, monkeypatch, td_model):
    import sympy

    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    td_max = float(population_models.cosmological_quantity_from_redshift(
        0, "age"
    ))
    delays = np.array(
        [0.01, 0.02, 0.1, 2.0, td_max, td_max + 1.0],
        dtype=dtype,
    )
    if td_model == "log_normal":
        expected = np.exp(
            -(np.log(delays) - np.log(2.9)) ** 2 / (2 * 0.2 ** 2)
        ) / (np.sqrt(2 * np.pi) * 0.2)
    elif td_model == "gaussian":
        expected = np.exp(-(delays - 2) ** 2 / (2 * 0.3 ** 2)) / (
            np.sqrt(2 * np.pi) * 0.3
        )
    elif td_model == "power_law":
        expected = delays ** -0.81
    else:
        norm = 1 / np.log(td_max / 0.02)
        expected = np.where(
            (delays < 0.02) | (delays > td_max),
            0,
            norm * delays ** -0.999,
        )

    with ctx:
        tensor = torch.tensor(
            delays,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        wrapped_input = Array(TorchArrayData(tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("time-delay model left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            for name in ("exp", "log", "sqrt", "Piecewise"):
                patch.setattr(sympy, name, reject_host_path)
            patch.setattr(population_models.np, "log", reject_host_path)
            patch.setattr(population_models.np, "where", reject_host_path)
            raw = population_models.p_tau(tensor, td_model)
            wrapped = population_models.p_tau(wrapped_input, td_model)
            scalar = population_models.p_tau(tensor[3], td_model)
            (
                raw.sum()
                + wrapped._data.tensor.sum()
                + scalar
            ).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    assert isinstance(scalar, torch.Tensor)
    assert scalar.ndim == 0
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    np.testing.assert_allclose(
        scalar.detach().cpu().numpy(),
        expected[3],
        rtol=tolerance,
        atol=tolerance,
    )
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize("mass_model", ("totalMass", "componentMass", "log"))
def test_population_injection_mass_pdfs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, mass_model):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([1, 1.5, 2, 2.5, 4.9, 5], dtype=dtype)
    mass2 = np.array([1, 2.5, 3, 3.4, 1.05, 2], dtype=dtype)
    bounds = (1.0, 5.0, 1.0, 6.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = scale_injections.inj_mass_pdf(
            mass_model, mass1, mass2, *bounds
        )

    with ctx:
        mass1_tensor = torch.tensor(
            mass1, dtype=torch_dtype, device=device, requires_grad=True
        )
        mass2_tensor = torch.tensor(
            mass2, dtype=torch_dtype, device=device, requires_grad=True
        )
        wrapped_input = Array(TorchArrayData(mass1_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("injection mass PDF left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(scale_injections.np, "array", reject_host_path)
            patch.setattr(scale_injections.np, "sign", reject_host_path)
            patch.setattr(scale_injections.np, "where", reject_host_path)
            patch.setattr(scale_injections, "log", reject_host_path)
            raw = scale_injections.inj_mass_pdf(
                mass_model, mass1_tensor, mass2_tensor, *bounds
            )
            wrapped = scale_injections.inj_mass_pdf(
                mass_model, wrapped_input, mass2_tensor, *bounds
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        assert torch.isfinite(actual).all()
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    for values in (mass1_tensor, mass2_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


def test_population_log_uniform_mass_prior_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([4, 10, 30, 70, 96], dtype=dtype)
    mass2 = np.array([6, 7, 20, 10, 5], dtype=dtype)
    expected = rates_functions.prob_lnm(mass1, mass2, None, None)

    with ctx:
        mass1_tensor = torch.tensor(
            mass1, dtype=torch_dtype, device=device, requires_grad=True
        )
        mass2_tensor = torch.tensor(
            mass2, dtype=torch_dtype, device=device, requires_grad=True
        )
        wrapped_input = Array(TorchArrayData(mass1_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("log-uniform mass prior left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(rates_functions.np, "array", reject_host_path)
            patch.setattr(rates_functions.np, "minimum", reject_host_path)
            patch.setattr(rates_functions.np, "maximum", reject_host_path)
            patch.setattr(rates_functions.np, "sign", reject_host_path)
            patch.setattr(rates_functions.np, "where", reject_host_path)
            raw = rates_functions.prob_lnm(
                mass1_tensor, mass2_tensor, None, None
            )
            wrapped = rates_functions.prob_lnm(
                wrapped_input, mass2_tensor, None, None
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    for values in (mass1_tensor, mass2_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0


def test_population_flat_mass_prior_cpu_contract():
    mass1 = np.array([5., 4., 2., 6., 4., 3.])
    mass2 = np.array([3., 5., 2., 3., 2., 3.])
    expected = np.array([0.125, 0., 0., 0., 0., 0.])

    actual = rates_functions.prob_flat(
        mass1, mass2, None, None, min_mass=2., max_mass=6.
    )

    np.testing.assert_array_equal(actual, expected)
    assert rates_functions.prob_flat(
        4., 3., None, None, min_mass=2., max_mass=6.
    ) == 0.125


def test_population_flat_mass_prior_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([5., 4., 2., 6., 4., 3.], dtype=dtype)
    mass2 = np.array([3., 5., 2., 3., 2., 3.], dtype=dtype)
    expected = np.array([0.125, 0., 0., 0., 0., 0.], dtype=dtype)

    with ctx:
        mass1_tensor = torch.tensor(
            mass1, dtype=torch_dtype, device=device, requires_grad=True
        )
        mass2_tensor = torch.tensor(
            mass2, dtype=torch_dtype, device=device, requires_grad=True
        )
        wrapped_input = Array(TorchArrayData(mass1_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("flat mass prior left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(rates_functions.np, "array", reject_host_path)
            patch.setattr(rates_functions.np, "where", reject_host_path)
            raw = rates_functions.prob_flat(
                mass1_tensor,
                mass2_tensor,
                None,
                None,
                min_mass=2.,
                max_mass=6.,
            )
            wrapped = rates_functions.prob_flat(
                wrapped_input,
                mass2_tensor,
                None,
                None,
                min_mass=2.,
                max_mass=6.,
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_array_equal(actual.detach().cpu().numpy(), expected)
    for values in (mass1_tensor, mass2_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) == 0


def test_population_power_law_mass_prior_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([6, 7, 30, 70, 96], dtype=dtype)
    mass2 = np.array([5.5, 10, 20, 10, 5], dtype=dtype)
    expected = rates_functions.prob_imf(mass1, mass2, None, None)

    with ctx:
        mass1_tensor = torch.tensor(
            mass1, dtype=torch_dtype, device=device, requires_grad=True
        )
        mass2_tensor = torch.tensor(
            mass2, dtype=torch_dtype, device=device, requires_grad=True
        )
        wrapped_input = Array(TorchArrayData(mass1_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("power-law mass prior left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(rates_functions.np, "array", reject_host_path)
            patch.setattr(rates_functions.np, "minimum", reject_host_path)
            patch.setattr(rates_functions.np, "maximum", reject_host_path)
            patch.setattr(rates_functions.np, "zeros_like", reject_host_path)
            patch.setattr(rates_functions.np, "where", reject_host_path)
            raw = rates_functions.prob_imf(
                mass1_tensor, mass2_tensor, None, None
            )
            wrapped = rates_functions.prob_imf(
                wrapped_input, mass2_tensor, None, None
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    for values in (mass1_tensor, mass2_tensor):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0


@pytest.mark.parametrize("spin_model", ("precessing", "aligned", "disable_spin"))
def test_population_injection_spin_pdfs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, spin_model):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    spins = np.array([-0.79, -0.4, -0.1, 0.2, 0.79, 0.8, 1.1], dtype=dtype)
    expected = scale_injections.inj_spin_pdf(spin_model, 0.8, spins)

    with ctx:
        spin_tensor = torch.tensor(
            spins, dtype=torch_dtype, device=device, requires_grad=True
        )
        wrapped_input = Array(TorchArrayData(spin_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("injection spin PDF left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(scale_injections.np, "array", reject_host_path)
            patch.setattr(scale_injections.np, "sign", reject_host_path)
            patch.setattr(scale_injections.np, "absolute", reject_host_path)
            raw = scale_injections.inj_spin_pdf(
                spin_model, 0.8, spin_tensor
            )
            wrapped = scale_injections.inj_spin_pdf(
                spin_model, 0.8, wrapped_input
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        assert torch.isfinite(actual).all()
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    assert spin_tensor.grad is not None
    assert torch.isfinite(spin_tensor.grad).all()


@pytest.mark.parametrize("spin_model", ("precessing", "aligned", "disable_spin"))
def test_population_injection_spin_zero_shortcut_stays_differentiable(
        torch_device_ctx, spin_model):
    ctx, device = torch_device_ctx
    torch_dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        spins = torch.tensor(
            [0.0, 0.2, 0.9],
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        result = scale_injections.inj_spin_pdf(spin_model, 0.8, spins)
        result.sum().backward()

    assert result.device.type == device
    assert torch.equal(result, torch.ones_like(result))
    assert spins.grad is not None
    assert torch.equal(spins.grad, torch.zeros_like(spins.grad))


@pytest.mark.parametrize("distance_model", ("uniform", "dchirp"))
def test_population_injection_distance_pdfs_stay_on_torch_device(
        torch_device_ctx, monkeypatch, distance_model):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    distances = np.array([9, 10, 20, 30, 40, 41], dtype=dtype)
    chirp_masses = np.array([1.2, 1.4, 1.8, 2, 2.2, 2.4], dtype=dtype)
    expected = scale_injections.inj_distance_pdf(
        distance_model, distances, 10.0, 40.0, chirp_masses
    )

    with ctx:
        distance_tensor = torch.tensor(
            distances, dtype=torch_dtype, device=device, requires_grad=True
        )
        chirp_tensor = torch.tensor(
            chirp_masses,
            dtype=torch_dtype,
            device=device,
            requires_grad=True,
        )
        wrapped_input = Array(TorchArrayData(distance_tensor), copy=False)

        def reject_host_path(*_args, **_kwargs):
            raise AssertionError("injection distance PDF left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_path)
            patch.setattr(torch.Tensor, "cpu", reject_host_path)
            patch.setattr(scale_injections.np, "array", reject_host_path)
            patch.setattr(scale_injections.np, "sign", reject_host_path)
            patch.setattr(scale_injections.np, "where", reject_host_path)
            raw = scale_injections.inj_distance_pdf(
                distance_model,
                distance_tensor,
                10.0,
                40.0,
                chirp_tensor,
            )
            wrapped = scale_injections.inj_distance_pdf(
                distance_model,
                wrapped_input,
                10.0,
                40.0,
                chirp_tensor,
            )
            (raw.sum() + wrapped._data.tensor.sum()).backward()

    assert isinstance(raw, torch.Tensor)
    assert raw.device.type == device
    assert raw.dtype == torch_dtype
    assert isinstance(wrapped, Array)
    assert wrapped._data.tensor.device.type == device
    assert wrapped._data.tensor.dtype == torch_dtype
    tolerance = 5e-6 if dtype == np.float32 else 2e-12
    for actual in (raw, wrapped._data.tensor):
        assert torch.isfinite(actual).all()
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    assert distance_tensor.grad is not None
    assert torch.isfinite(distance_tensor.grad).all()
    if distance_model == "dchirp":
        assert chirp_tensor.grad is not None
        assert torch.isfinite(chirp_tensor.grad).all()
        assert torch.count_nonzero(chirp_tensor.grad) > 0


def test_population_symbolic_models_remain_symbolic():
    import sympy

    value = sympy.Symbol("value", positive=True)
    assert population_models.diff_lookback_time(value).has(value)
    for td_model in ("log_normal", "gaussian", "power_law", "inverse"):
        result = population_models.p_tau(value, td_model)
        assert isinstance(result, sympy.Expr)
        assert result.has(value)


def test_coincidence_noise_rates_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    torch_dtype = torch.float32 if device == "mps" else torch.float64
    areas = {2: 0.125, 3: 0.03125}
    raw = {
        "H1": np.array([0.4, 0.25, 0.1, 0.05], dtype=dtype),
        "L1": np.array([0.3, 0.2, 0.08, 0.04], dtype=dtype),
        "V1": np.array([0.2, 0.15, 0.06, 0.03], dtype=dtype),
    }
    logs = {ifo: np.log(values) for ifo, values in raw.items()}

    expected_log = np.log(areas[2]) + logs["H1"] + logs["L1"]
    expected_rate = np.exp(expected_log)
    expected_multi = {
        "H1 L1 V1": (
            np.log(areas[3]) + logs["H1"] + logs["L1"] + logs["V1"]
        ),
        "H1 L1": expected_log,
        "H1 V1": np.log(areas[2]) + logs["H1"] + logs["V1"],
        "L1 V1": np.log(areas[2]) + logs["L1"] + logs["V1"],
    }

    def fake_allowed_area(ifos, _slop, dets=None):
        assert dets is None
        return areas[len(ifos)]

    with ctx:
        raw_tensors = {
            ifo: torch.tensor(
                values, dtype=torch_dtype, device=device,
                requires_grad=True,
            )
            for ifo, values in raw.items()
        }
        log_tensors = {
            ifo: torch.tensor(
                values, dtype=torch_dtype, device=device,
                requires_grad=True,
            )
            for ifo, values in logs.items()
        }
        torch_raw = {
            ifo: Array(TorchArrayData(values), copy=False)
            for ifo, values in raw_tensors.items()
        }
        torch_logs = {
            ifo: Array(TorchArrayData(values), copy=False)
            for ifo, values in log_tensors.items()
        }

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("coincidence-rate vector left Torch")

        def reject_numpy_rate_math(*_args, **_kwargs):
            raise AssertionError("coincidence rate used NumPy arithmetic")

        with monkeypatch.context() as patch:
            patch.setattr(
                coinc_rate, "multiifo_noise_coincident_area",
                fake_allowed_area,
            )
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(coinc_rate.numpy, "log", reject_numpy_rate_math)
            patch.setattr(coinc_rate.numpy, "sum", reject_numpy_rate_math)
            patch.setattr(coinc_rate.numpy, "exp", reject_numpy_rate_math)
            actual_log = coinc_rate.combination_noise_lograte(
                {ifo: torch_logs[ifo] for ifo in ("H1", "L1")},
                0.002,
            )
            actual_rate = coinc_rate.combination_noise_rate(
                {ifo: torch_raw[ifo] for ifo in ("H1", "L1")},
                0.002,
            )
            actual_multi = coinc_rate.multiifo_noise_lograte(
                torch_logs, 0.002
            )
            loss = actual_log._data.tensor.sum()
            loss = loss + actual_rate._data.tensor.sum()
            loss = loss + sum(
                values._data.tensor.sum()
                for values in actual_multi.values()
            )
            loss.backward()

    outputs = [actual_log, actual_rate, *actual_multi.values()]
    assert all(isinstance(values, Array) for values in outputs)
    assert all(values._data.tensor.device.type == device for values in outputs)
    assert all(values._data.tensor.dtype == torch_dtype for values in outputs)
    assert set(actual_multi) == set(expected_multi)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        actual_log._data.tensor.detach().cpu().numpy(),
        expected_log,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual_rate._data.tensor.detach().cpu().numpy(),
        expected_rate,
        rtol=tolerance,
        atol=tolerance,
    )
    for combination, values in actual_multi.items():
        np.testing.assert_allclose(
            values._data.tensor.detach().cpu().numpy(),
            expected_multi[combination],
            rtol=tolerance,
            atol=tolerance,
        )
    differentiated = (
        raw_tensors["H1"],
        raw_tensors["L1"],
        *log_tensors.values(),
    )
    for values in differentiated:
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0
    assert raw_tensors["V1"].grad is None


def test_inference_inner_reduction_stays_differentiable(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    left_values = np.asarray(
        (1.0 + 0.5j, -0.25 + 2.0j, 0.75 - 1.5j),
        dtype=complex_dtype,
    )
    right_values = np.asarray(
        (-0.5 + 0.25j, 1.5 - 0.75j, 2.0 + 0.5j),
        dtype=complex_dtype,
    )

    with ctx:
        left = Array(left_values)
        right = Array(right_values)
        left_tensor = left._data.tensor
        right_tensor = right._data.tensor
        left_tensor.requires_grad_()
        right_tensor.requires_grad_()

        def reject_host_scalar(*_args, **_kwargs):
            raise AssertionError("likelihood inner product scalarized")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", reject_host_scalar)
            patch.setattr(torch.Tensor, "cpu", reject_host_scalar)
            patch.setattr(torch.Tensor, "numpy", reject_host_scalar)
            patch.setattr(TorchArrayData, "numpy", reject_host_scalar)
            actual = _INFERENCE_TOOLS._inner(left, right)
            actual_real = _INFERENCE_TOOLS._real_inner(left, right)
            (actual.real + 0.25 * actual.imag + actual_real).backward()

        actual_values = (
            actual.detach().cpu().numpy(),
            actual_real.detach().cpu().numpy(),
        )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == device
    assert actual.requires_grad
    assert left_tensor.grad is not None
    assert right_tensor.grad is not None
    assert torch.isfinite(left_tensor.grad).all()
    assert torch.isfinite(right_tensor.grad).all()
    tolerance = 2e-6 if device == "mps" else 2e-12
    expected = np.vdot(left_values, right_values)
    np.testing.assert_allclose(
        actual_values, (expected, expected.real),
        rtol=tolerance, atol=tolerance,
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


def test_inference_frequency_lookup_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    frequencies = np.arange(256, dtype=dtype) * dtype(0.125)

    with ctx:
        values = Array(frequencies)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Frequency lookup copied its grid")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            index = _INFERENCE_TOOLS._last_index_at_or_below(values, 17.31)
            with pytest.raises(IndexError, match="no values"):
                _INFERENCE_TOOLS._last_index_at_or_below(values, -0.1)

    assert index == np.searchsorted(frequencies, 17.31, side="right") - 1
    assert values._data.tensor.device.type == device


@pytest.mark.parametrize("torch_waveforms", (False, True))
def test_relative_binning_summaries_stay_on_device(
        torch_device_ctx, monkeypatch, torch_waveforms):
    from pycbc.inference.models.relbin import Relative

    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(721946)
    size = 257
    delta_f = real_dtype(0.125)

    h1 = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    h2 = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    psd = (1.0 + rng.random(size)).astype(real_dtype)
    freqs = np.arange(size, dtype=real_dtype) * delta_f
    bins = np.array(
        ((3, 17), (17, 68), (68, 129), (129, 256)),
        dtype=np.int64,
    )
    h12 = np.conjugate(h1) * h2 / psd
    expected_a0 = np.array([
        4.0 * delta_f * h12[low:high].sum()
        for low, high in bins
    ])
    expected_a1 = np.array([
        4.0 / (high - low) * (
            h12[low:high] * (freqs[low:high] - freqs[low])
        ).sum()
        for low, high in bins
    ])

    with ctx:
        psd_input = Array(psd)
        h1_input = Array(h1) if torch_waveforms else h1
        h2_input = Array(h2) if torch_waveforms else h2
        model = object.__new__(Relative)
        model._psds = {"H1": psd_input}
        model.f = {"H1": freqs}
        model.df = {"H1": delta_f}
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Relative-bin summaries copied to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual_a0, actual_a1 = Relative.summary_product(
                model, h1_input, h2_input, bins, "H1"
            )

        actual_a0_values = actual_a0.detach().cpu().numpy()
        actual_a1_values = actual_a1.detach().cpu().numpy()

    tolerance = 4e-4 if device == "mps" else 2e-12
    assert actual_a0.device.type == device
    assert actual_a1.device.type == device
    np.testing.assert_allclose(
        actual_a0_values, expected_a0, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_a1_values, expected_a1, rtol=tolerance, atol=tolerance)


def test_relative_binning_cross_grid_stays_on_device(
        torch_device_ctx, monkeypatch):
    from pycbc.inference.models.relbin import Relative

    class Reference:
        pass

    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(29416)
    size = 65
    delta_f = real_dtype(0.25)
    h1 = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    h2 = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    psd = (1.0 + rng.random(size)).astype(real_dtype)
    freqs = np.arange(size, dtype=real_dtype) * delta_f
    edges1 = np.array((2, 10, 18, 26, 34, 42, 50, 58, 64))
    edges2 = np.array((2, 8, 18, 24, 34, 40, 50, 56, 64))
    h1[[18, 40]] = 0
    h2[[18, 40]] = 0

    edges = np.unique([edges1, edges2])
    edges = edges[(h1[edges] != 0) | (h2[edges] != 0)]
    bins = np.stack((edges[:-1], edges[1:]), axis=1)
    h12 = np.conjugate(h1) * h2 / psd
    expected_a0 = np.array([
        4.0 * delta_f * h12[low:high].sum()
        for low, high in bins
    ])
    expected_a1 = np.array([
        4.0 / (high - low) * (
            h12[low:high] * (freqs[low:high] - freqs[low])
        ).sum()
        for low, high in bins
    ])

    with ctx:
        h1_input = Array(h1)
        h2_input = Array(h2)
        frequencies = Array(freqs)
        model = object.__new__(Relative)
        model._data = {"H1": None}
        model._psds = {"H1": Array(psd)}
        model.f = {"H1": frequencies}
        model.df = {"H1": delta_f}
        m1, m2 = Reference(), Reference()
        m1.h00, m2.h00 = {"H1": h1_input}, {"H1": h2_input}
        m1.edges, m2.edges = {"H1": edges1}, {"H1": edges2}
        m1.f = {"H1": frequencies}

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Relative-bin cross grid copied its mask to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            Relative.calculate_hihjs(model, [m1, m2])

        actual_a0, actual_a1, actual_fedge = model.hihj[(m1, m2)]["H1"]
        actual_a0_values = actual_a0.detach().cpu().numpy()
        actual_a1_values = actual_a1.detach().cpu().numpy()
        actual_fedge_values = actual_fedge.detach().cpu().numpy()

    tolerance = 4e-4 if device == "mps" else 2e-12
    assert actual_a0.device.type == device
    assert actual_a1.device.type == device
    assert actual_fedge.device.type == device
    np.testing.assert_allclose(
        actual_a0_values, expected_a0, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_a1_values, expected_a1, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(actual_fedge_values, freqs[edges])


@pytest.mark.parametrize("torch_waveform", (False, True))
def test_relative_binning_reference_data_stays_on_device(
        torch_device_ctx, monkeypatch, torch_waveform):
    from pycbc.inference.models.relbin import (
        _prepare_reference_data,
        _uniform_frequency_grid,
    )
    from pycbc.inference.models.tools import _threshold_extent

    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    size = 32
    offset = 5
    delta_f = real_dtype(0.25)
    time_shift = real_dtype(0.013)
    waveform = np.array(
        (0j, 1.5 - 0.2j, -0.4 + 0.7j, 0.3 + 0.9j, 0j),
        dtype=complex_dtype,
    )
    data = (
        np.linspace(-0.8, 1.2, size, dtype=real_dtype)
        + 1j * np.linspace(0.4, -0.6, size, dtype=real_dtype)
    ).astype(complex_dtype)
    padded = np.zeros(size, dtype=complex_dtype)
    padded[:len(waveform)] = waveform
    expected_reference = np.roll(padded, offset)
    frequencies = np.arange(size, dtype=real_dtype) * delta_f
    expected_data = data * np.conjugate(
        np.exp(-2.0j * np.pi * frequencies * time_shift)
    )

    cpu_waveform = Array(waveform)
    with ctx:
        waveform_input = Array(waveform) if torch_waveform else cpu_waveform
        data_input = FrequencySeries(data, delta_f=delta_f)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Relative-bin initialization copied to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            rebuilt_frequencies = _uniform_frequency_grid(data_input)
            first, last = _threshold_extent(waveform_input, 0.0)
            reference, shifted_data = _prepare_reference_data(
                waveform_input,
                data_input,
                size,
                offset,
                delta_f,
                time_shift,
            )

        reference_tensor = reference._data.tensor
        shifted_data_tensor = shifted_data._data.tensor
        actual_reference = reference_tensor.detach().cpu().numpy()
        actual_data = shifted_data_tensor.detach().cpu().numpy()

    tolerance = 4e-5 if device == "mps" else 2e-12
    assert (first, last) == (1, 3)
    assert rebuilt_frequencies.dtype == np.dtype(real_dtype)
    np.testing.assert_array_equal(rebuilt_frequencies, frequencies)
    assert reference_tensor.device.type == device
    assert shifted_data_tensor.device.type == device
    np.testing.assert_allclose(
        actual_reference, expected_reference, rtol=0, atol=0)
    np.testing.assert_allclose(
        actual_data, expected_data, rtol=tolerance, atol=tolerance)


def test_relative_binning_curvature_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    size = 41

    x = np.linspace(-0.7, 0.9, size, dtype=real_dtype)
    references = {
        "H1": (1.4 + 0.2 * x + 0.3j * x).astype(complex_dtype),
        "L1": (1.7 - 0.1 * x + 0.2j * x).astype(complex_dtype),
    }
    ratios = {
        "H1": (1.0 + 0.04 * x**2 + 0.03j * x**3).astype(complex_dtype),
        "L1": (1.0 - 0.02 * x**3 + 0.05j * x**2).astype(complex_dtype),
    }
    waveforms = {
        ifo: references[ifo] * ratios[ifo] for ifo in references
    }
    expected = max(
        np.abs(np.diff(ratio / np.abs(ratio).min(), n=2)).max()
        for ratio in ratios.values()
    )

    model = inference_relbin.Relative.__new__(inference_relbin.Relative)
    model._data = {ifo: None for ifo in references}

    with ctx:
        waveform_arrays = {
            ifo: Array(values) for ifo, values in waveforms.items()
        }
        model.wf_ret = {
            ifo: (waveform_arrays[ifo], None) for ifo in references
        }
        model.h00_sparse = {
            "H1": Array(references["H1"]),
            "L1": references["L1"],
        }
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Curvature check copied a waveform")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                inference_relbin, "_numpy_value", _reject_host_transfer)
            actual = model.max_curvature_from_reference()

    tolerance = 3e-5 if device == "mps" else 2e-12
    assert all(
        value._data.tensor.device.type == device
        for value in waveform_arrays.values()
    )
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance)


def test_relative_binning_layout_dedup_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    real_dtype = np.float32 if device == "mps" else np.float64
    shared = np.linspace(20.0, 256.0, 33, dtype=real_dtype)
    distinct = np.linspace(20.0, 512.0, 65, dtype=real_dtype)
    model = inference_relbin.Relative.__new__(inference_relbin.Relative)
    model.fedges = {
        "H1": shared,
        "L1": shared.copy(),
        "V1": distinct,
    }

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Layout dedup copied frequency edges")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            model.combine_layout()

        edge_tensors = [edge._data.tensor for edge in model.edge_unique]

    assert model.ifo_map == {"H1": 0, "L1": 0, "V1": 1}
    assert len(edge_tensors) == 2
    assert all(tensor.device.type == device for tensor in edge_tensors)
    np.testing.assert_array_equal(
        edge_tensors[0].detach().cpu().numpy(), shared)
    np.testing.assert_array_equal(
        edge_tensors[1].detach().cpu().numpy(), distinct)


@pytest.mark.parametrize("response_kind", ("polarization", "earth", "detector"))
def test_relative_binning_likelihood_stays_on_device(
        torch_device_ctx, monkeypatch, response_kind):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(1847)
    size = 129

    freqs = np.linspace(18.0, 512.0, size, dtype=real_dtype)
    hp = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    hc = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    h00 = (1.5 + rng.random(size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    a0 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(
        complex_dtype)
    a1 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(
        complex_dtype)
    b0 = rng.random(size - 1).astype(real_dtype)
    b1 = rng.normal(size=size - 1).astype(real_dtype)

    if response_kind == "earth":
        fp = np.linspace(0.3, 0.7, size, dtype=real_dtype)
        fc = np.linspace(-0.4, 0.2, size, dtype=real_dtype)
        dtc = np.linspace(-0.002, 0.003, size, dtype=real_dtype)
    else:
        fp, fc, dtc = 0.61, -0.27, 0.0017

    # Match the compatibility constant used by the Cython relbin kernels.
    shift = np.exp(-2.0j * 3.141592653 * dtc * freqs)
    if response_kind == "detector":
        ratio = shift * hp / h00
    else:
        ratio = shift * (fp * hp + fc * hc) / h00
    ratio_delta = ratio[1:] - ratio[:-1]
    expected_filt = np.conjugate(
        np.sum(a0 * ratio[:-1] + a1 * ratio_delta))
    power = np.abs(ratio) ** 2
    expected_norm = np.sum(
        b0 * power[:-1] + b1 * (power[1:] - power[:-1]))

    with ctx:
        hp_array = Array(hp)
        hc_array = Array(hc)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Relative binning copied its waveform")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            prepared = _INFERENCE_RELBIN_TORCH.prepare_likelihood_data(
                hp_array, freqs, h00, a0, a1, b0, b1)
            assert all(value.device.type == device for value in prepared)
            if response_kind == "detector":
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts_det(
                        prepared[0], dtc, hp_array, *prepared[1:]))
            else:
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts(
                        prepared[0], fp, fc, dtc, hp_array, hc_array,
                        *prepared[1:]))

    tolerance = 3e-4 if device == "mps" else 2e-12
    assert actual_filt.device.type == device
    assert actual_norm.device.type == device
    np.testing.assert_allclose(
        actual_filt.item(), expected_filt,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_norm.item(), expected_norm,
        rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("response_kind", ("polarization", "earth", "detector"))
def test_relative_binning_multi_likelihood_stays_on_device(
        torch_device_ctx, monkeypatch, response_kind):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(48319)
    size = 101

    freqs = np.linspace(18.0, 512.0, size, dtype=real_dtype)

    def complex_values():
        return (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
            complex_dtype)

    hp, hc = complex_values(), complex_values()
    hp2, hc2 = complex_values(), complex_values()
    h00 = (2.0 + complex_values()).astype(complex_dtype)
    h002 = (2.0 + complex_values()).astype(complex_dtype)
    a0 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)
    a1 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)

    if response_kind == "earth":
        fp = np.linspace(-0.4, 0.7, size, dtype=real_dtype)
        fc = np.linspace(0.6, -0.2, size, dtype=real_dtype)
        dtc = np.linspace(-0.002, 0.003, size, dtype=real_dtype)
        fp2 = np.linspace(0.5, -0.6, size, dtype=real_dtype)
        fc2 = np.linspace(-0.1, 0.8, size, dtype=real_dtype)
        dtc2 = np.linspace(0.001, -0.004, size, dtype=real_dtype)
    else:
        fp, fc, dtc = 0.43, -0.28, 0.0013
        fp2, fc2, dtc2 = -0.37, 0.62, -0.0021

    shift = np.exp(-2.0j * 3.141592653 * dtc * freqs)
    shift2 = np.exp(-2.0j * 3.141592653 * dtc2 * freqs)
    if response_kind == "detector":
        ratio = shift * hp / h00
        ratio2 = shift2 * hp2 / h002
        cross = np.conjugate(ratio) * ratio2
    else:
        ratio = shift * (fp * hp + fc * hc) / h00
        ratio2 = shift2 * (fp2 * hp2 + fc2 * hc2) / h002
        cross = ratio * np.conjugate(ratio2)
    expected = np.sum(
        a0 * cross[:-1] + a1 * (cross[1:] - cross[:-1]))

    with ctx:
        hp_array, hc_array = Array(hp), Array(hc)
        hp2_array, hc2_array = Array(hp2), Array(hc2)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Multi-signal relative binning copied its waveform")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            prepared = (
                _INFERENCE_RELBIN_TORCH.prepare_multi_likelihood_data(
                    hp_array, freqs, h00, h002, a0, a1))
            if response_kind == "detector":
                actual = _INFERENCE_RELBIN_TORCH.likelihood_parts_det_multi(
                    prepared[0], dtc, hp_array, prepared[1],
                    dtc2, hp2_array, prepared[2], *prepared[3:])
            else:
                kernel = _INFERENCE_RELBIN_TORCH.likelihood_parts_multi
                if response_kind == "earth":
                    kernel = _INFERENCE_RELBIN_TORCH.likelihood_parts_multi_v
                actual = kernel(
                    prepared[0], fp, fc, dtc,
                    hp_array, hc_array, prepared[1],
                    fp2, fc2, dtc2, hp2_array, hc2_array, prepared[2],
                    *prepared[3:])

    tolerance = 8e-4 if device == "mps" else 2e-12
    assert actual.device.type == device
    assert all(value.device.type == device for value in prepared)
    np.testing.assert_allclose(
        actual.item(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "response_kind",
    ("vector", "time", "polarization",
     "earth_pol", "earth_time", "earth_pol_time"),
)
def test_relative_binning_vector_likelihood_stays_on_device(
        torch_device_ctx, monkeypatch, response_kind):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(81173)
    size, sample_count = 83, 19

    freqs = np.linspace(20.0, 512.0, size, dtype=real_dtype)
    hp = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    hc = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    h00 = (1.5 + rng.random(size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    a0 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)
    a1 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)
    b0 = rng.random(size - 1).astype(real_dtype)
    b1 = rng.normal(size=size - 1).astype(real_dtype)

    fp_samples = rng.uniform(-0.8, 0.8, sample_count).astype(real_dtype)
    fc_samples = rng.uniform(-0.8, 0.8, sample_count).astype(real_dtype)
    dt_samples = rng.uniform(-0.004, 0.004, sample_count).astype(real_dtype)
    pol_phase = np.exp(
        -2.0j * rng.uniform(0, np.pi, sample_count)
    ).astype(complex_dtype)
    fp_freq = np.linspace(-0.3, 0.7, size, dtype=real_dtype)
    fc_freq = np.linspace(0.5, -0.2, size, dtype=real_dtype)
    times = np.linspace(-0.003, 0.002, size, dtype=real_dtype)

    if response_kind in ("vector", "time", "polarization"):
        fp = fp_samples if response_kind != "time" else real_dtype(0.43)
        fc = fc_samples if response_kind != "time" else real_dtype(-0.21)
        dtc = dt_samples if response_kind != "polarization" \
            else real_dtype(0.0013)
        fp_grid = np.asarray(fp)[..., None]
        fc_grid = np.asarray(fc)[..., None]
        dt_grid = np.asarray(dtc)[..., None]
        ratio = np.exp(
            -2.0j * 3.141592653 * dt_grid * freqs
        ) * (fp_grid * hp + fc_grid * hc) / h00
    else:
        response = fp_freq + 1.0j * fc_freq
        if response_kind in ("earth_pol", "earth_pol_time"):
            response = pol_phase[:, None] * response
        total_time = times
        if response_kind in ("earth_time", "earth_pol_time"):
            total_time = times + dt_samples[:, None]
        ratio = np.exp(
            -2.0j * 3.141592653 * total_time * freqs
        ) * (response.real * hp + response.imag * hc) / h00

    ratio_lo = ratio[..., :-1]
    ratio_delta = ratio[..., 1:] - ratio_lo
    expected_filt = np.conjugate(
        np.sum(a0 * ratio_lo + a1 * ratio_delta, axis=-1))
    power = np.abs(ratio) ** 2
    expected_norm = np.sum(
        b0 * power[..., :-1] +
        b1 * (power[..., 1:] - power[..., :-1]), axis=-1)

    with ctx:
        hp_array = Array(hp)
        hc_array = Array(hc)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Relative binning copied its waveform")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            prepared = _INFERENCE_RELBIN_TORCH.prepare_likelihood_data(
                hp_array, freqs, h00, a0, a1, b0, b1)
            if response_kind in ("vector", "time", "polarization"):
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts_vector(
                        prepared[0], fp, fc, dtc, hp_array, hc_array,
                        *prepared[1:]))
            elif response_kind == "earth_pol":
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts_v_pol(
                        prepared[0], fp_freq, fc_freq, times, pol_phase,
                        hp_array, hc_array, *prepared[1:]))
            elif response_kind == "earth_time":
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts_v_time(
                        prepared[0], fp_freq, fc_freq, times, dt_samples,
                        hp_array, hc_array, *prepared[1:]))
            else:
                actual_filt, actual_norm = (
                    _INFERENCE_RELBIN_TORCH.likelihood_parts_v_pol_time(
                        prepared[0], fp_freq, fc_freq, times, dt_samples,
                        pol_phase, hp_array, hc_array, *prepared[1:]))

    tolerance = 5e-4 if device == "mps" else 2e-12
    assert actual_filt.device.type == device
    assert actual_norm.device.type == device
    np.testing.assert_allclose(
        actual_filt.resolve_conj().cpu().numpy(), expected_filt,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_norm.cpu().numpy(), expected_norm,
        rtol=tolerance, atol=tolerance)


def test_relative_binning_polarization_stays_differentiable(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    torch_real_dtype = torch.float32 if device == "mps" else torch.float64
    rng = np.random.default_rng(52291)
    ifo, size, sample_count = "H1", 29, 7

    freqs = np.linspace(20.0, 256.0, size, dtype=real_dtype)
    hp = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    hc = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    h00 = (1.2 + rng.random(size) + 1j * rng.normal(size=size)).astype(
        complex_dtype)
    a0 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)
    a1 = (rng.normal(size=size - 1) +
          1j * rng.normal(size=size - 1)).astype(complex_dtype)
    b0 = rng.random(size - 1).astype(real_dtype)
    b1 = rng.normal(size=size - 1).astype(real_dtype)
    polarization_values = np.linspace(
        0.1, 1.1, sample_count, dtype=real_dtype)
    antenna = (real_dtype(0.43), real_dtype(-0.27))
    delay = real_dtype(0.0017)
    tc, end_time, ta = 3.0, 2.99, 0.002
    dtc = tc + delay - end_time - ta

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
                antenna[0] + 0.01 * (ra - 1.0),
                antenna[1] + 0.02 * (dec + 0.4),
            )

        def time_delay_from_earth_center(self, ra, dec, _times):
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            detector_calls.append("delay")
            return delay + 0.001 * (ra - dec - 1.4)

    model = inference_relbin.Relative.__new__(inference_relbin.Relative)
    model.fedges = {ifo: freqs}
    model.sdat = {
        ifo: {"a0": a0, "a1": a1, "b0": b0, "b1": b1}
    }
    model.h00_sparse = {ifo: h00}
    model.antenna_time = {ifo: 0.0}
    model.det = {ifo: _Detector()}
    model.end_time = {ifo: end_time}
    model.ta = {ifo: ta}
    model.lformat = None
    model._torch_likelihood_cache = {}

    with ctx:
        hp_array = Array(hp)
        hc_array = Array(hc)
        polarization = torch.tensor(
            polarization_values, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        right_ascension = torch.tensor(
            1.0, dtype=torch_real_dtype, device=device, requires_grad=True)
        declination = torch.tensor(
            -0.4, dtype=torch_real_dtype, device=device, requires_grad=True)
        params = {
            "ra": right_ascension,
            "dec": declination,
            "tc": tc,
            "polarization": polarization,
        }
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Polarization copied samples to host")

            def _reject_numpy_phase(_value):
                raise AssertionError("Polarization phase used NumPy")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(inference_relbin.numpy, "exp", _reject_numpy_phase)
            actual_filt, actual_norm, wf_parts = (
                model._polarization_likelihood_parts(
                    ifo, params, (hp_array, hc_array),
                    inference_relbin.likelihood_parts_vector))
            loss = actual_filt.real.sum() + actual_norm.sum()
            loss.backward()

        actual_filt_values = actual_filt.detach().resolve_conj().cpu().numpy()
        actual_norm_values = actual_norm.detach().cpu().numpy()
        gradient = polarization.grad.detach().cpu().numpy()
        sky_gradients = tuple(
            value.grad.detach().cpu().numpy()
            for value in (right_ascension, declination)
        )

    response = (antenna[0] + 1.0j * antenna[1]) * np.exp(
        -2.0j * polarization_values)
    ratio = np.exp(-2.0j * 3.141592653 * dtc * freqs) * (
        response.real[:, None] * hp + response.imag[:, None] * hc
    ) / h00
    ratio_lo = ratio[:, :-1]
    ratio_delta = ratio[:, 1:] - ratio_lo
    expected_filt = np.conjugate(
        np.sum(a0 * ratio_lo + a1 * ratio_delta, axis=-1))
    power = np.abs(ratio) ** 2
    expected_norm = np.sum(
        b0 * power[:, :-1] +
        b1 * (power[:, 1:] - power[:, :-1]), axis=-1)

    assert detector_calls == ["antenna", "delay"]
    assert all(value.device.type == device for value in wf_parts[:3])
    assert polarization.grad is not None
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient != 0.0)
    assert all(np.isfinite(value) and value != 0.0
               for value in sky_gradients)
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_filt_values, expected_filt,
        rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_norm_values, expected_norm,
        rtol=tolerance, atol=tolerance)

