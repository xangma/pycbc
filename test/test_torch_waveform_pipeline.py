# flake8: noqa: F401
import operator
import os
import subprocess
import sys
import types

import warnings

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


from pycbc.waveform import waveform as waveform_module

from torch_test_support import (
    stub_execute_cached_fft_import,
    torch_ctx,
    torch_device_ctx,
)


torch = pytest.importorskip("torch")


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)




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


def test_series_coordinate_grids_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    input_dtype = np.float32 if device == "mps" else np.float64
    expected_times = np.arange(6) * 0.25 + 10.5
    expected_frequencies = np.arange(6) * 0.125

    with ctx:
        time_series = TimeSeries(
            np.zeros(6, dtype=input_dtype), delta_t=0.25, epoch=10.5
        )
        frequency_series = FrequencySeries(
            np.zeros(6, dtype=input_dtype), delta_f=0.125
        )

        def _reject_host_array(*_args, **_kwargs):
            raise AssertionError("coordinate grid was staged through NumPy")

        def _reject_host_transfer(_self):
            raise AssertionError("coordinate grid left the Torch device")

        with monkeypatch.context() as patch:
            patch.setattr(array_module._numpy, "array", _reject_host_array)
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            sample_times = time_series.sample_times
            sample_frequencies = frequency_series.sample_frequencies

    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert sample_times._data.tensor.device.type == device
    assert sample_frequencies._data.tensor.device.type == device
    assert sample_times._data.tensor.dtype == expected_dtype
    assert sample_frequencies._data.tensor.dtype == expected_dtype
    np.testing.assert_allclose(
        sample_times._data.tensor.detach().cpu().numpy(), expected_times
    )
    np.testing.assert_allclose(
        sample_frequencies._data.tensor.detach().cpu().numpy(),
        expected_frequencies,
    )


def test_base_coordinate_transforms_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    expected_x = np.array([0.75, -1.25, 0.4])
    expected_y = np.array([-0.5, 0.8, 1.1])
    expected_z = np.array([1.2, -0.3, 0.65])
    expected_rho = np.sqrt(
        expected_x**2 + expected_y**2 + expected_z**2
    )
    expected_phi = np.arctan2(expected_y, expected_x) % (2 * np.pi)
    expected_theta = np.arccos(expected_z / expected_rho)

    with ctx:
        x = torch.tensor(
            expected_x, device=device, dtype=dtype, requires_grad=True
        )
        y = torch.tensor(
            expected_y, device=device, dtype=dtype, requires_grad=True
        )
        z = torch.tensor(
            expected_z, device=device, dtype=dtype, requires_grad=True
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("coordinate transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(coordinate_base.numpy, "sqrt", reject_host_or_numpy)
            patch.setattr(
                coordinate_base.numpy, "arctan2", reject_host_or_numpy
            )
            patch.setattr(
                coordinate_base.numpy, "arccos", reject_host_or_numpy
            )
            patch.setattr(coordinate_base.numpy, "cos", reject_host_or_numpy)
            patch.setattr(coordinate_base.numpy, "sin", reject_host_or_numpy)
            rho, phi, theta = coordinates.cartesian_to_spherical(x, y, z)
            actual_x, actual_y, actual_z = (
                coordinates.spherical_to_cartesian(rho, phi, theta)
            )
            origin_theta = coordinates.cartesian_to_spherical_polar(
                torch.zeros(3, device=device, dtype=dtype), 0.0, 0.0
            )

        results = (rho, phi, theta, actual_x, actual_y, actual_z)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert torch.equal(origin_theta, torch.zeros_like(origin_theta))

        (actual_x + actual_y + actual_z).sum().backward()
        assert all(value.grad is not None for value in (x, y, z))
        assert all(value.grad.device.type == device for value in (x, y, z))

        actual = [
            value.detach().cpu().numpy()
            for value in (rho, phi, theta, actual_x, actual_y, actual_z)
        ]

    tolerance = 2e-6 if device == "mps" else 2e-13
    for value, expected in zip(
            actual,
            (
                expected_rho,
                expected_phi,
                expected_theta,
                expected_x,
                expected_y,
                expected_z,
            )):
        np.testing.assert_allclose(
            value, expected, rtol=tolerance, atol=tolerance
        )


def test_log_transforms_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([0.25, 0.8, 2.5])
    bounded = np.array([-1.5, 0.25, 2.0])
    domain = (-2.0, 3.0)
    expected_log = np.log(values)
    expected_logit = (
        np.log(bounded - domain[0]) - np.log(domain[1] - bounded)
    )

    with ctx:
        x = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )
        p = torch.tensor(
            bounded, device=device, dtype=dtype, requires_grad=True
        )
        log = transforms.Log("x", "logx")
        exponent = transforms.Exponent("logx", "x_roundtrip")
        logit = transforms.Logit("p", "logitp", domain=domain)
        logistic = transforms.Logistic(
            "logitp", "p_roundtrip", codomain=domain
        )

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("parameter transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(transforms.numpy, "log", reject_host_or_numpy)
            patch.setattr(transforms.numpy, "exp", reject_host_or_numpy)
            log_maps = log.transform({"x": x})
            exp_maps = exponent.transform({"logx": log_maps["logx"]})
            logit_maps = logit.transform({"p": p})
            logistic_maps = logistic.transform(
                {"logitp": logit_maps["logitp"]}
            )
            derivatives = (
                log.jacobian({"x": x}),
                log.inverse_jacobian({"logx": log_maps["logx"]}),
                logit.jacobian({"p": p}),
                logit.inverse_jacobian(
                    {"logitp": logit_maps["logitp"]}
                ),
            )

        results = (
            log_maps["logx"],
            exp_maps["x_roundtrip"],
            logit_maps["logitp"],
            logistic_maps["p_roundtrip"],
        ) + derivatives
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 2e-5 if dtype == torch.float32 else 1e-12
        assert np.allclose(
            log_maps["logx"].detach().tolist(),
            expected_log,
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.allclose(
            exp_maps["x_roundtrip"].detach().tolist(),
            values,
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.allclose(
            logit_maps["logitp"].detach().tolist(),
            expected_logit,
            rtol=tolerance,
            atol=tolerance,
        )
        assert np.allclose(
            logistic_maps["p_roundtrip"].detach().tolist(),
            bounded,
            rtol=tolerance,
            atol=tolerance,
        )
        sum(value.sum() for value in results).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert p.grad is not None and torch.isfinite(p.grad).all()

        outside = torch.tensor(
            [-2.0, 0.0], device=device, dtype=dtype
        )
        with pytest.raises(ValueError, match="not in bounds"):
            logit.transform({"p": outside})


def test_boundary_conditioning_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    values = np.array([-10.0, -4.0, -1.0, 0.25, 1.0, 3.0, 8.0])
    expected_well = np.array([0.0, 0.0, -1.0, 0.25, 1.0, -1.0, 0.0])
    expected_min = np.array([8.0, 2.0, -1.0, 0.25, 1.0, 3.0, 8.0])
    expected_max = np.array([-10.0, -4.0, -1.0, 0.25, 1.0, -1.0, -6.0])
    expected_cyclic = np.array([0.0, 0.0, -1.0, 0.25, -1.0, -1.0, 0.0])

    with ctx:
        tensor = torch.tensor(
            values, device=device, dtype=dtype, requires_grad=True
        )
        well = boundaries.Bounds(
            -1.0, 1.0, btype_min="reflected", btype_max="reflected"
        )
        minimum = boundaries.Bounds(-1.0, 1.0, btype_min="reflected")
        maximum = boundaries.Bounds(-1.0, 1.0, btype_max="reflected")
        cyclic = boundaries.Bounds(-1.0, 1.0, cyclic=True)

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("boundary conditioning left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            for bound in (well, minimum, maximum):
                patch.setattr(bound, "_reflect", reject_host_or_numpy)
            results = (
                well.apply_conditions(tensor),
                minimum.apply_conditions(tensor),
                maximum.apply_conditions(tensor),
                cyclic.apply_conditions(tensor),
            )

        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        expected = (
            expected_well, expected_min, expected_max, expected_cyclic
        )
        for actual, target in zip(results, expected):
            assert np.allclose(actual.detach().tolist(), target)
        sum(value.sum() for value in results).backward()
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_boundary_torch_type_cache_is_lazy_bounded_and_subclass_safe(
        monkeypatch):
    boundaries._torch_module_for_type.cache_clear()

    # Ordinary host values must not execute a Torch import. Torch is already
    # loaded by this test module, so make an attempted import observable.
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("host boundary classification imported Torch")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", checked_import)
        assert boundaries._torch_module_for(1.0) is None

    class TensorSubclass(torch.Tensor):
        pass

    tensor = torch.arange(3.0).as_subclass(TensorSubclass)
    assert boundaries._torch_module_for(tensor) is torch

    class UnhashableType(type):
        __hash__ = None

    class UnhashableHost(metaclass=UnhashableType):
        pass

    assert boundaries._torch_module_for(UnhashableHost()) is None

    # Adversarial callers can introduce many short-lived Python types. The
    # dispatch cache must not retain an unbounded number of them.
    for index in range(40):
        host_type = type(f"_BoundaryHostType{index}", (), {})
        assert boundaries._torch_module_for(host_type()) is None
    cache_info = boundaries._torch_module_for_type.cache_info()
    assert cache_info.maxsize == 32
    assert cache_info.currsize <= cache_info.maxsize
    boundaries._torch_module_for_type.cache_clear()


def test_custom_transform_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    mchirp_values = np.array([18.0, 24.0])
    q_values = np.array([1.5, 3.0])
    redshift_values = np.array([0.1, 0.2])
    angle_values = np.array([0.2, 0.7])
    expected_mass1 = (
        q_values ** (2.0 / 5.0)
        * (1.0 + q_values) ** (1.0 / 5.0)
        * mchirp_values
        * (1.0 + redshift_values)
    )
    expected_mass2 = (
        q_values ** (-3.0 / 5.0)
        * (1.0 + q_values) ** (1.0 / 5.0)
        * mchirp_values
        * (1.0 + redshift_values)
    )
    expected_phase = np.sin(angle_values) + np.sqrt(q_values)
    expected_jacobian = np.exp(redshift_values) / q_values
    custom = transforms.CustomTransform(
        ["srcmchirp", "q", "redshift", "angle"],
        ["mass1", "mass2", "phase_term"],
        {
            "mass1": (
                "mass1_from_mchirp_q(srcmchirp, q) * (1 + redshift)"
            ),
            "mass2": (
                "mass2_from_mchirp_q(srcmchirp, q) * (1 + redshift)"
            ),
            "phase_term": "sin(angle) + sqrt(q)",
        },
        jacobian="exp(redshift) / q",
    )

    with ctx:
        inputs = {
            name: torch.tensor(
                values, device=device, dtype=dtype, requires_grad=True
            )
            for name, values in (
                ("srcmchirp", mchirp_values),
                ("q", q_values),
                ("redshift", redshift_values),
                ("angle", angle_values),
            )
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("custom transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(custom, "_copytoscratch", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            output = custom.transform(inputs)
            jacobian = custom.jacobian(inputs)

        results = (
            output["mass1"],
            output["mass2"],
            output["phase_term"],
            jacobian,
        )
        expected = (
            expected_mass1,
            expected_mass2,
            expected_phase,
            expected_jacobian,
        )
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for actual, target in zip(results, expected):
            assert np.allclose(
                actual.detach().tolist(),
                target,
                rtol=tolerance,
                atol=tolerance,
            )

        sum(value.sum() for value in results).backward()
        assert all(value.grad is not None for value in inputs.values())
        assert all(
            torch.isfinite(value.grad).all() for value in inputs.values()
        )


def test_custom_transform_cosmology_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-5 if dtype == torch.float32 else 1e-12
    model = cosmology.get_cosmology()
    redshifts = np.array([0.05, 0.2, 1.0])
    distances = model.luminosity_distance(redshifts).value
    volumes = model.comoving_volume(redshifts).value

    distance_converter = cosmology.DistToZ(numpoints=128)
    redshift_converter = cosmology.ComovingVolInterpolator(
        "redshift", numpoints=64
    )
    volume_distance_converter = cosmology.ComovingVolInterpolator(
        "luminosity_distance", numpoints=64
    )
    converters = (
        distance_converter,
        redshift_converter,
        volume_distance_converter,
    )
    for converter in converters:
        converter.setup_interpolant()

    expected = {
        "distance_redshift": distance_converter(distances),
        "volume_redshift": redshift_converter(volumes),
        "volume_distance": volume_distance_converter(volumes),
    }
    monkeypatch.setitem(
        cosmology._d2zs, cosmology.DEFAULT_COSMOLOGY, distance_converter
    )
    monkeypatch.setitem(
        cosmology._v2zs, cosmology.DEFAULT_COSMOLOGY, redshift_converter
    )
    monkeypatch.setitem(
        cosmology._v2ds,
        cosmology.DEFAULT_COSMOLOGY,
        volume_distance_converter,
    )
    custom = transforms.CustomTransform(
        ["distance", "comoving_volume"],
        ["distance_redshift", "volume_redshift", "volume_distance"],
        {
            "distance_redshift": "redshift(distance)",
            "volume_redshift": (
                "redshift_from_comoving_volume(comoving_volume)"
            ),
            "volume_distance": (
                "distance_from_comoving_volume(comoving_volume)"
            ),
        },
    )

    with ctx:
        inputs = {
            "distance": torch.tensor(
                distances, device=device, dtype=dtype, requires_grad=True
            ),
            "comoving_volume": torch.tensor(
                volumes, device=device, dtype=dtype, requires_grad=True
            ),
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("cosmology transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(custom, "_copytoscratch", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(
                scipy.interpolate.interp1d,
                "__call__",
                reject_host_or_numpy,
            )
            output = custom.transform(inputs)

        results = {
            name: output[name]
            for name in (
                "distance_redshift",
                "volume_redshift",
                "volume_distance",
            )
        }
        assert all(isinstance(value, torch.Tensor)
                   for value in results.values())
        assert all(value.device.type == device
                   for value in results.values())
        assert all(value.dtype == dtype for value in results.values())
        for name, actual in results.items():
            assert np.allclose(
                actual.detach().tolist(),
                expected[name],
                rtol=tolerance,
                atol=tolerance,
            )

        sum(value.sum() for value in results.values()).backward()
        assert all(value.grad is not None for value in inputs.values())
        assert all(
            bool(torch.isfinite(value.grad).all())
            for value in inputs.values()
        )


def test_custom_transform_preserves_unsupported_function_fallback(torch_ctx):
    custom = transforms.CustomTransform(
        ["value"],
        ["output"],
        {"output": "host_only(value, mode='square')"},
    )
    custom._scratch.add_functions(
        "host_only",
        lambda value, mode: value**2 + (mode == "square"),
    )

    with torch_ctx:
        value = torch.tensor(2.0)
        output = custom.transform({"value": value})

    assert output["value"] is value
    assert isinstance(output["output"], np.floating)
    assert output["output"] == 5.0


def test_custom_transform_binary_ufuncs_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    x_values = np.array([0.5, 2.0, -1.5])
    y_values = np.array([1.5, 1.0, 0.75])
    custom = transforms.CustomTransform(
        ["x", "y"],
        ["angle", "angle_alias", "low", "high", "radius", "direction"],
        {
            "angle": "atan2(x, y)",
            "angle_alias": "arctan2(x, y)",
            "low": "minimum(x, y)",
            "high": "maximum(x, y)",
            "radius": "hypot(x, y)",
            "direction": "sign(x - y)",
        },
    )
    expected = {
        "angle": np.arctan2(x_values, y_values),
        "angle_alias": np.arctan2(x_values, y_values),
        "low": np.minimum(x_values, y_values),
        "high": np.maximum(x_values, y_values),
        "radius": np.hypot(x_values, y_values),
        "direction": np.sign(x_values - y_values),
    }

    with ctx:
        inputs = {
            "x": torch.tensor(
                x_values, device=device, dtype=dtype, requires_grad=True
            ),
            "y": torch.tensor(
                y_values, device=device, dtype=dtype, requires_grad=True
            ),
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("binary custom transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(custom, "_copytoscratch", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            output = custom.transform(inputs)

        results = [output[name] for name in expected]
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        assert all(value.dtype == dtype for value in results)
        for name, actual in zip(expected, results):
            assert np.allclose(
                actual.detach().tolist(),
                expected[name],
                rtol=tolerance,
                atol=tolerance,
            )

        sum(value.sum() for value in results).backward()
        assert all(value.grad is not None for value in inputs.values())
        assert all(
            torch.isfinite(value.grad).all() for value in inputs.values()
        )


def test_spherical_spin_and_quadrupole_conversions_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([30.0, 8.0, 20.0])
    mass2 = np.array([10.0, 12.0, 20.0])
    spin1_a = np.array([0.4, 0.25, 0.1])
    spin1_polar = np.array([0.3, 1.1, 2.2])
    spin2_a = np.array([0.2, 0.35, 0.15])
    spin2_polar = np.array([1.5, 2.0, 0.7])
    lambdav = np.array([10.0, 300.0, 1500.0])
    expected_chi_eff = (
        mass1 * spin1_a * np.cos(spin1_polar)
        + mass2 * spin2_a * np.cos(spin2_polar)
    ) / (mass1 + mass2)
    ll = np.log(lambdav)
    expected_dquadmon = np.exp(
        0.194
        + 0.0936 * ll
        + 0.0474 * ll**2
        - 4.21e-3 * ll**3
        + 1.23e-4 * ll**4
    ) - 1

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (
                mass1, mass2, spin1_a, spin1_polar,
                spin2_a, spin2_polar, lambdav,
            )
        ]
        m1, m2, s1a, s1p, s2a, s2p, tidal_lambda = inputs

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("conversion left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "cos", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "log", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "exp", reject_host_or_numpy)
            effective_spin = conversions.chi_eff_from_spherical(
                m1, m2, s1a, s1p, s2a, s2p
            )
            dquadmon = conversions.dquadmon_from_lambda(tidal_lambda)

        results = (effective_spin, dquadmon)
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 3e-5 if dtype == torch.float32 else 1e-12
        assert np.allclose(
            effective_spin.detach().tolist(), expected_chi_eff,
            rtol=tolerance, atol=tolerance,
        )
        assert np.allclose(
            dquadmon.detach().tolist(), expected_dquadmon,
            rtol=tolerance, atol=tolerance,
        )

        (effective_spin.sum() + dquadmon.sum()).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_precession_mass_spin_transform_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    mass1 = np.array([30.0, 8.0, 20.0])
    mass2 = np.array([10.0, 12.0, 20.0])
    xi1 = np.array([0.4, 0.3, 0.2])
    xi2 = np.array([0.2, 0.35, 0.15])
    phi_a = np.array([0.3, 1.1, 2.2])
    phi_s = np.array([1.5, 2.0, 4.0])
    primary_is_one = mass1 >= mass2
    primary_mass = np.maximum(mass1, mass2)
    secondary_mass = np.minimum(mass1, mass2)
    primary_xi = np.where(primary_is_one, xi1, xi2)
    secondary_xi = np.where(primary_is_one, xi2, xi1)
    mass_ratio = primary_mass / secondary_mass
    a1 = 2 + 1.5 * mass_ratio
    a2 = 2 + 1.5 / mass_ratio
    secondary_perp = mass_ratio**2 * a2 / a1 * secondary_xi
    primary_phi = (phi_s + phi_a) / 2.0
    secondary_phi = (phi_s - phi_a) / 2.0
    primary_x = primary_xi * np.cos(primary_phi)
    primary_y = primary_xi * np.sin(primary_phi)
    secondary_x = secondary_perp * np.cos(secondary_phi)
    secondary_y = secondary_perp * np.sin(secondary_phi)
    expected = {
        "spin1x": np.where(primary_is_one, primary_x, secondary_x),
        "spin1y": np.where(primary_is_one, primary_y, secondary_y),
        "spin2x": np.where(primary_is_one, secondary_x, primary_x),
        "spin2y": np.where(primary_is_one, secondary_y, primary_y),
    }

    with ctx:
        inputs = [
            torch.tensor(
                value, device=device, dtype=dtype, requires_grad=True
            )
            for value in (mass1, mass2, xi1, xi2, phi_a, phi_s)
        ]
        maps = dict(zip(
            ("mass1", "mass2", "xi1", "xi2", "phi_a", "phi_s"),
            inputs,
        ))

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("precession transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            patch.setattr(transforms.numpy, "asarray", reject_host_or_numpy)
            patch.setattr(transforms.numpy, "where", reject_host_or_numpy)
            patch.setattr(conversions, "ensurearray", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "arctan2", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "cos", reject_host_or_numpy)
            patch.setattr(conversions.numpy, "sin", reject_host_or_numpy)
            cartesian = (
                transforms.PrecessionMassSpinToCartesianSpin().transform(maps)
            )
            roundtrip = (
                transforms.CartesianSpinToPrecessionMassSpin().transform(
                    cartesian
                )
            )

        cartesian_values = tuple(cartesian[key] for key in expected)
        roundtrip_values = tuple(
            roundtrip[key] for key in ("xi1", "xi2", "phi_a", "phi_s")
        )
        results = cartesian_values + roundtrip_values
        assert all(isinstance(value, torch.Tensor) for value in results)
        assert all(value.device.type == device for value in results)
        tolerance = 3e-6 if dtype == torch.float32 else 2e-12
        for key, actual in zip(expected, cartesian_values):
            assert np.allclose(
                actual.detach().tolist(), expected[key],
                rtol=tolerance, atol=tolerance,
            )
        for actual, target in zip(
                roundtrip_values, (xi1, xi2, phi_a, phi_s)):
            assert np.allclose(
                actual.detach().tolist(), target,
                rtol=tolerance, atol=tolerance,
            )

        sum(value.sum() for value in cartesian_values).backward()
        assert all(value.grad is not None for value in inputs)
        assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_gated_polarization_marginalization_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    length = 32
    delta_f = 0.25
    indices = np.arange(length)
    values = {
        "hp": (0.8 + 0.02 * indices) * np.exp(0.07j * indices),
        "hc": (0.3 + 0.01 * indices) * np.exp(-0.05j * indices),
        "gated_hp": (0.7 + 0.01 * indices) * np.exp(0.04j * indices),
        "gated_hc": (0.2 + 0.015 * indices) * np.exp(-0.03j * indices),
        "data": (1.1 + 0.005 * indices) * np.exp(0.02j * indices),
        "gated_data": (
            (0.9 + 0.008 * indices) * np.exp(0.01j * indices)
        ),
    }
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

    def make_model(series):
        model = types.SimpleNamespace()
        model.pol = np.linspace(0, 2 * np.pi, 257)
        model._torch_polarization_grids = {}
        model.dets = {"H1": _Detector()}
        model.current_params = {
            "tc": reference_time,
            "ra": 1.2,
            "dec": -0.4,
        }
        model._current_stats = types.SimpleNamespace()
        model._kmin = {"H1": 2}
        model._kmax = {"H1": length - 2}
        model._invpsds = {
            "H1": FrequencySeries(
                np.linspace(0.7, 1.3, length), delta_f=delta_f
            )
        }
        model._overwhitened_data = {"H1": series["data"]}
        model.get_waveforms = lambda: {
            "H1": (series["hp"], series["hc"])
        }
        model.get_gated_waveforms = lambda: {
            "H1": (series["gated_hp"], series["gated_hc"])
        }
        model.get_gated_data = lambda: {"H1": series["gated_data"]}
        model.gate_indices = lambda _det: (4, 9)
        model.det_lognorm = lambda _det, _start, _end: -0.75
        model._polarization_grid = types.MethodType(
            gated_gaussian_noise.GatedGaussianMargPol._polarization_grid,
            model,
        )
        return model

    numpy_series = {
        name: FrequencySeries(value, delta_f=delta_f)
        for name, value in values.items()
    }
    likelihood = (
        gated_gaussian_noise.GatedGaussianMargPol._loglikelihood.__wrapped__
    )
    expected_model = make_model(numpy_series)
    expected = likelihood(expected_model)
    assert detector_calls == [
        ("arrival", False),
        ("antenna", False),
    ]

    with ctx:
        torch_series = {
            name: FrequencySeries(value, delta_f=delta_f)
            for name, value in values.items()
        }
        actual_model = make_model(torch_series)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "polarization marginalization copied arrays to host"
            )

        def reject_scalar_inner(*_args, **_kwargs):
            raise AssertionError(
                "polarization marginalization scalarized inner products"
            )

        with monkeypatch.context() as patch:
            import pycbc.types.array_torch as array_torch

            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(array_torch, "inner", reject_scalar_inner)
            actual = likelihood(actual_model)

    assert detector_calls == [
        ("arrival", False),
        ("antenna", False),
        ("arrival", True),
        ("antenna", True),
    ]

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        np.remainder(
            actual_model._current_stats.maxl_polarization, np.pi
        ),
        np.remainder(
            expected_model._current_stats.maxl_polarization, np.pi
        ),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        actual_model._current_stats.maxl_logl,
        expected_model._current_stats.maxl_logl,
        rtol=2e-12,
        atol=2e-12,
    )
    assert len(actual_model._torch_polarization_grids) == 1
    grid = next(iter(actual_model._torch_polarization_grids.values()))
    assert grid.device.type == device


def test_lisa_coordinate_roundtrip_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        ssb_values = (
            torch.tensor([1.0e5, 2.0e5], device=device, dtype=dtype),
            torch.tensor([0.4, 4.8], device=device, dtype=dtype),
            torch.tensor([-0.3, 0.7], device=device, dtype=dtype),
            torch.tensor([0.2, 5.4], device=device, dtype=dtype),
        )
        with monkeypatch.context() as patch:
            def reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("LISA coordinates copied tensors to host")

            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            lisa_values = coordinate_space.ssb_to_lisa(*ssb_values)
            recovered = coordinate_space.lisa_to_ssb(*lisa_values)

        for value in (*lisa_values, *recovered):
            assert isinstance(value, torch.Tensor)
            assert value.device.type == device

        for actual, expected in zip(recovered, ssb_values):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2e-6,
                atol=2e-6,
            )


def test_torch_shape_transform_copy_and_fallback_contract(monkeypatch):
    with scheme.TorchScheme("cpu"):
        values = Array(np.arange(12, dtype=np.float64).reshape(3, 4))
        view = np.reshape(values, (2, 6), copy=False)
        copied = np.reshape(values, (2, 6), copy=np.bool_(True))
        fortran_source = values.transpose()
        fortran_view = np.reshape(
            fortran_source, (2, 6), order="F", copy=False
        )
        fortran_copied = np.reshape(
            fortran_source, (2, 6), order=b"F", copy=True
        )
        flattened = values.flatten()

        view._data.tensor[0, 0] = -1
        copied._data.tensor[0, 1] = -2
        fortran_view._data.tensor[0, 0] = -4
        before_copy_mutation = values._data.tensor.clone()
        fortran_copied._data.tensor[0, 1] = -5
        flattened._data.tensor[2] = -3

        assert values._data.tensor[0, 0].item() == -4
        assert values._data.tensor[0, 1].item() == 1
        assert values._data.tensor.reshape(-1)[2].item() == 2
        torch.testing.assert_close(
            values._data.tensor, before_copy_mutation
        )

        with pytest.raises(ValueError, match="avoid a copy"):
            values.reshape(2, 6, order="F", copy=False)

        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "numpy_reshape",
                lambda self, *args, **kwargs: NotImplemented,
            )
            fallback = np.reshape(values, (4, 3), order="F")

        assert isinstance(fallback, np.ndarray)
        np.testing.assert_array_equal(
            fallback,
            np.reshape(values.numpy(), (4, 3), order="F"),
        )

        noncontiguous = values.transpose()
        with pytest.raises(ValueError, match="avoid a copy"):
            noncontiguous.reshape(12, copy=False)


def test_template_bank_coordinate_rotations_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    numpy_dtype = np.float32 if device == "mps" else np.float64
    f_upper = 512.0
    evecs = np.array(
        [[0.8, -0.3, 0.2], [0.4, 0.9, -0.1], [-0.2, 0.1, 0.95]],
        dtype=numpy_dtype,
    )
    evals = np.array([1.5, 0.75, 2.25], dtype=numpy_dtype)
    evecs_cv = np.array(
        [[0.9, 0.2, -0.1], [-0.25, 0.95, 0.15], [0.12, -0.08, 0.98]],
        dtype=numpy_dtype,
    )
    lambdas = np.array(
        [[1.0, 1.5, -0.5, 2.0], [0.25, -0.75, 1.25, 0.5],
         [-1.0, 0.4, 0.8, -0.2]],
        dtype=numpy_dtype,
    )
    metric = types.SimpleNamespace(
        evecs={f_upper: evecs}, evals={f_upper: evals}
    )
    expected_mus = tmpltbank_coord_utils.get_mu_params(
        lambdas, metric, f_upper
    )
    expected_xis = tmpltbank_coord_utils.get_covaried_params(
        expected_mus, evecs_cv
    )
    expected_component = tmpltbank_coord_utils.rotate_vector(
        evecs_cv, expected_mus, 1.7, 1
    )

    with ctx:
        tensor = torch.tensor(
            lambdas, device=device, dtype=dtype, requires_grad=True
        )
        scale = torch.tensor(1.7, device=device, dtype=dtype)

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "template-bank coordinates copied tensors to the host"
            )

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            mus = tmpltbank_coord_utils.get_mu_params(
                tensor, metric, f_upper
            )
            xis = tmpltbank_coord_utils.get_covaried_params(mus, evecs_cv)
            component = tmpltbank_coord_utils.rotate_vector(
                evecs_cv, mus, scale, 1
            )
            component_from_list = tmpltbank_coord_utils.rotate_vector(
                evecs_cv, list(mus), scale, 1
            )
            single = tmpltbank_coord_utils.get_mu_params(
                tensor[:, 0], metric, f_upper
            )

        loss = xis.square().sum() + component.square().sum() + single.sum()
        loss.backward()

    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for result in (mus, xis, component, component_from_list, single):
        assert isinstance(result, torch.Tensor)
        assert result.device.type == device
        assert result.dtype == dtype
    assert single.shape == (3,)
    torch.testing.assert_close(
        mus,
        torch.as_tensor(expected_mus, device=device, dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        xis,
        torch.as_tensor(expected_xis, device=device, dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        component,
        torch.as_tensor(expected_component, device=device, dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(component_from_list, component)
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(tensor.grad) > 0


def test_template_bank_chirp_coordinates_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    masses1 = np.array([2.2, 7.4, 35.0])
    masses2 = np.array([1.3, 3.1, 14.0])
    spins1 = np.array([0.1, -0.4, 0.7])
    spins2 = np.array([-0.2, 0.3, -0.5])
    quads1 = np.array([1.0, 1.2, 2.0])
    quads2 = np.array([1.0, 1.5, 1.1])
    expected = tmpltbank_lambda_mapping.get_chirp_params(
        masses1,
        masses2,
        spins1,
        spins2,
        70.0,
        "threePointFivePN",
        quadparam1=quads1,
        quadparam2=quads2,
    )

    with ctx:
        inputs = [
            torch.tensor(
                values, device=device, dtype=dtype, requires_grad=True
            )
            for values in (
                masses1,
                masses2,
                spins1,
                spins2,
                quads1,
                quads2,
            )
        ]

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError(
                "template-bank chirp coordinates copied tensors to the host"
            )

        def reject_lal_phasing(*_args, **_kwargs):
            raise AssertionError(
                "template-bank chirp coordinates called LAL phasing"
            )

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "tolist", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            patch.setattr(
                tmpltbank_lambda_mapping.lalsimulation,
                "SimInspiralTaylorF2AlignedPhasingArray",
                reject_lal_phasing,
            )
            actual = tmpltbank_lambda_mapping.get_chirp_params(
                *inputs[:4],
                70.0,
                "threePointFivePN",
                quadparam1=inputs[4],
                quadparam2=inputs[5],
            )
            scalar = tmpltbank_lambda_mapping.get_chirp_params(
                *(values[0] for values in inputs[:4]),
                70.0,
                "threePointFivePN",
            )
            loss = sum(value.square().sum() for value in actual)
            loss.backward()

    tolerance = 5e-5 if dtype == torch.float32 else 2e-12
    assert len(actual) == len(expected) == 8
    for value, reference in zip(actual, expected):
        assert isinstance(value, torch.Tensor)
        assert value.device.type == device
        assert value.dtype == dtype
        assert value.shape == masses1.shape
        torch.testing.assert_close(
            value,
            torch.as_tensor(reference, device=device, dtype=dtype),
            rtol=tolerance,
            atol=tolerance,
        )
    for value in scalar:
        assert isinstance(value, torch.Tensor)
        assert value.device.type == device
        assert value.dtype == dtype
        assert value.shape == ()
    for value in inputs:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


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


def test_gaussian_noise_likelihoods_stay_on_device_until_stats_boundary(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    waveform_values = {
        "H1": np.asarray(
            (0.5 + 0.25j, -1.0 + 0.75j, 1.5 - 0.5j),
            dtype=complex_dtype,
        ),
        "L1": np.asarray(
            (-0.25 + 0.5j, 0.75 - 1.25j, 1.0 + 0.25j),
            dtype=complex_dtype,
        ),
    }
    data_values = {
        "H1": np.asarray(
            (1.0 - 0.5j, 0.25 + 1.25j, -0.75 + 0.5j),
            dtype=complex_dtype,
        ),
        "L1": np.asarray(
            (0.5 + 0.75j, -1.0 + 0.25j, 1.25 - 0.5j),
            dtype=complex_dtype,
        ),
    }
    stat_names = (
        "loglikelihood",
        "H1_cplx_loglr", "L1_cplx_loglr",
        "H1_optimal_snrsq", "L1_optimal_snrsq",
    )
    class GaussianHarness:
        _trytoget = inference_model_base.BaseModel._trytoget
        loglr = inference_model_base_data.BaseDataModel.loglr
        loglikelihood = inference_model_base.BaseModel.loglikelihood
        _loglikelihood = (
            inference_gaussian_noise.BaseGaussianNoise._loglikelihood
        )
        _loglr = (
            inference_gaussian_noise.GaussianNoise._loglr.__wrapped__
        )
        det_cplx_loglr = (
            inference_gaussian_noise.GaussianNoise.det_cplx_loglr
        )
        det_optimal_snrsq = (
            inference_gaussian_noise.GaussianNoise.det_optimal_snrsq
        )

    model = GaussianHarness()
    model._kmin = {det: 0 for det in waveform_values}
    model._kmax = {
        det: len(values) for det, values in waveform_values.items()
    }
    model._current_stats = inference_model_base.ModelStats()
    model.default_stats = stat_names
    model.lognl = -1.25
    model.waveform_transforms = None

    with ctx:
        waveforms = {
            det: FrequencySeries(values, delta_f=0.25)
            for det, values in waveform_values.items()
        }
        waveform_calls = []

        def get_waveforms():
            waveform_calls.append(None)
            return waveforms

        model.get_waveforms = get_waveforms
        model._weight = {
            det: FrequencySeries(
                np.ones(len(values), dtype=values.real.dtype),
                delta_f=0.25,
            )
            for det, values in waveform_values.items()
        }
        model._whitened_data = {
            det: FrequencySeries(values, delta_f=0.25)
            for det, values in data_values.items()
        }
        data_tensors = [
            series._data.tensor for series in model._whitened_data.values()
        ]
        for tensor in data_tensors:
            tensor.requires_grad_()

        def reject_host_method(*_args, **_kwargs):
            raise AssertionError("Gaussian inner product used a host method")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", reject_host_method)
            patch.setattr(torch.Tensor, "cpu", reject_host_method)
            patch.setattr(torch.Tensor, "numpy", reject_host_method)
            patch.setattr(TorchArrayData, "numpy", reject_host_method)
            patch.setattr(
                inference_gaussian_noise,
                "_public_stat_value",
                lambda value: value,
            )
            first_detector_value = model.det_cplx_loglr("H1")
            actual = model.loglr
            cached_loglikelihood = model.loglikelihood

        assert isinstance(actual, torch.Tensor)
        assert isinstance(cached_loglikelihood, torch.Tensor)
        assert actual.device.type == device
        assert cached_loglikelihood.device.type == device
        assert actual is model._current_stats.loglr
        assert cached_loglikelihood is model._current_stats.loglikelihood
        assert first_detector_value is model._current_stats.H1_cplx_loglr
        assert len(waveform_calls) == 1
        assert all(
            isinstance(getattr(model._current_stats, name), torch.Tensor)
            for name in stat_names
        )
        model._current_stats.loglikelihood.backward()
        actual_value = actual.detach().cpu().numpy()
        public_tuple = inference_model_base.BaseModel.get_current_stats(model)
        public_dict = inference_model_base.BaseModel.current_stats.fget(model)
        public_h1_cplx = (
            model.det_cplx_loglr("H1")
        )
        public_h1_hh = (
            model.det_optimal_snrsq("H1")
        )

    expected_hh = {
        det: np.vdot(values, values).real
        for det, values in waveform_values.items()
    }
    expected_cplx = {
        det: np.vdot(values, data_values[det]) - 0.5 * expected_hh[det]
        for det, values in waveform_values.items()
    }
    expected_lr = sum(value.real for value in expected_cplx.values())
    expected_stats = (
        expected_lr + model.lognl,
        expected_cplx["H1"], expected_cplx["L1"],
        expected_hh["H1"], expected_hh["L1"],
    )
    tolerance = 2e-6 if device == "mps" else 2e-12
    assert all(tensor.grad is not None for tensor in data_tensors)
    assert all(torch.isfinite(tensor.grad).all() for tensor in data_tensors)
    np.testing.assert_allclose(
        actual_value, expected_lr, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        public_tuple, expected_stats, rtol=tolerance, atol=tolerance
    )
    assert public_dict == dict(zip(stat_names, public_tuple))
    assert type(public_h1_cplx) is complex
    assert type(public_h1_hh) is float
    np.testing.assert_allclose(
        public_h1_cplx, expected_cplx["H1"],
        rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        public_h1_hh, expected_hh["H1"],
        rtol=tolerance, atol=tolerance,
    )


def test_marginalized_polarization_response_stays_differentiable(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    torch_real_dtype = torch.float32 if device == "mps" else torch.float64
    ifo = "H1"
    fp, fc = real_dtype(0.61), real_dtype(-0.27)
    polarization_values = np.asarray((0.17, 0.58), dtype=real_dtype)
    detector_calls = []

    class _Detector:
        def arrival_time(self, ref_tc, ra, dec, ref_frame):
            detector_calls.append("arrival")
            assert ref_frame == "geocentric"
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            return (
                torch.as_tensor(ref_tc, device=ra.device, dtype=ra.dtype)
                + 0.02 * (ra - 1.0)
                + 0.03 * (dec + 0.4)
            )

        def antenna_pattern(self, ra, dec, polarization, tc):
            detector_calls.append("antenna")
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec, polarization, tc)
            )
            base_fp = fp + 0.01 * (ra - 1.0) + 0.03 * tc
            base_fc = fc + 0.02 * (dec + 0.4) + 0.04 * tc
            cosine = torch.cos(2.0 * polarization)
            sine = torch.sin(2.0 * polarization)
            return (
                base_fp * cosine + base_fc * sine,
                base_fc * cosine - base_fp * sine,
            )

    model = types.SimpleNamespace()
    model.all_ifodata_same_rate_length = True
    model.dets = {ifo: _Detector()}
    model._kmin = {ifo: 0}
    model._kmax = {ifo: 4}
    model._current_stats = types.SimpleNamespace()

    def _marginalize(sh, hh, return_peak=False):
        assert return_peak
        values = sh.real - 0.5 * hh
        return values.sum(), 0, values[0]

    model.marginalize_loglr = _marginalize

    with ctx:
        polarization = torch.tensor(
            polarization_values, dtype=torch_real_dtype,
            device=device, requires_grad=True)
        right_ascension = torch.tensor(
            1.0, dtype=torch_real_dtype, device=device, requires_grad=True)
        declination = torch.tensor(
            -0.4, dtype=torch_real_dtype, device=device, requires_grad=True)
        model.current_params = {
            "polarization": polarization,
            "ra": right_ascension,
            "dec": declination,
            "tc": 0.0,
        }
        hp = FrequencySeries(
            np.ones(4, dtype=complex_dtype), delta_f=0.25)
        hc = FrequencySeries(
            np.full(4, 2.0j, dtype=complex_dtype), delta_f=0.25)
        model.waveform_generator = types.SimpleNamespace(
            generate=lambda **_params: {ifo: (hp, hc)})
        model._weight = {
            ifo: FrequencySeries(
                np.ones(4, dtype=real_dtype), delta_f=0.25)
        }
        model._whitened_data = {
            ifo: FrequencySeries(
                np.ones(4, dtype=complex_dtype), delta_f=0.25)
        }

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Marginalized polarization copied samples to the host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "item", _reject_host_transfer)
            loglr = (
                marginalized_gaussian_noise.MarginalizedPolarization._loglr)
            actual = loglr.__wrapped__(model)
            actual.backward()

        actual_value = actual.detach().cpu().numpy()
        polarization_gradient = polarization.grad.detach().cpu().numpy()
        sky_gradients = tuple(
            value.grad.detach().cpu().numpy()
            for value in (right_ascension, declination)
        )

    response = (fp + 1.0j * fc) * np.exp(
        -2.0j * polarization_values)
    expected_sh = 4.0 * response.real - 8.0j * response.imag
    expected_hh = 4.0 * response.real ** 2 + 16.0 * response.imag ** 2
    expected = np.sum(expected_sh.real - 0.5 * expected_hh)

    assert actual.device.type == device
    assert detector_calls == ["arrival", "antenna"]
    assert np.all(np.isfinite(polarization_gradient))
    assert np.all(polarization_gradient != 0.0)
    assert all(
        np.isfinite(value) and value != 0.0
        for value in sky_gradients
    )
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected, rtol=tolerance, atol=tolerance)


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
