import importlib.util
from pathlib import Path
import subprocess
import sys
import types

from igwn_ligolw import lsctables
import lal
import lalsimulation
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
from pycbc import conversions, events, scheme, transforms
from pycbc.detector import Detector, single_arm_frequency_response
from pycbc.events import coherent, eventmgr, ranking, threshold_torch
from pycbc.filter import matchedfilter, resample, zpk
from pycbc.inference.models import (
    gaussian_noise as inference_gaussian_noise,
    marginalized_gaussian_noise,
    single_template,
)
from pycbc.inject.inject import SGBurstInjectionSet
from pycbc.inject.injfilterrejector import InjFilterRejector
from pycbc.noise import gaussian
from pycbc.psd import (
    analytical as analytical_psd,
    analytical_space,
    estimate as psd_estimate,
    read as psd_read,
    variation,
    welch,
)
from pycbc.strain import recalibrate
from pycbc.strain.strain import StrainBuffer, detect_loud_glitches, gate_data
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
import pycbc.vetoes.autochisq as autochisq
import pycbc.vetoes.bank_chisq as bank_chisq
import pycbc.vetoes.chisq as chisq
import pycbc.vetoes.sgchisq as sgchisq


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
    if not module_path.is_file():
        return None
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


def test_custom_transform_detector_timing_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-7 if dtype == torch.float32 else 1e-12
    ra_values = np.array([0.2, 1.1, 2.4])
    dec_values = np.array([-0.4, 0.3, 0.8])
    tc = 1187008882.0
    h1_delay = conversions.det_tc(
        "H1", ra_values, dec_values, tc, relative=1
    )
    l1_delay = conversions.det_tc(
        "L1", ra_values, dec_values, tc, relative=1
    )
    expected = {
        "dh": h1_delay,
        "dhl": l1_delay - h1_delay,
    }
    custom = transforms.CustomTransform(
        ["ra", "dec"],
        ["dh", "dhl"],
        {
            "dh": "det_tc('H1', ra, dec, 1187008882.0, relative=1)",
            "dhl": (
                "det_tc('L1', ra, dec, 1187008882.0, relative=1) - "
                "det_tc('H1', ra, dec, 1187008882.0, relative=1)"
            ),
        },
    )

    with ctx:
        inputs = {
            "ra": torch.tensor(
                ra_values, device=device, dtype=dtype, requires_grad=True
            ),
            "dec": torch.tensor(
                dec_values, device=device, dtype=dtype, requires_grad=True
            ),
        }

        def reject_host_or_numpy(*_args, **_kwargs):
            raise AssertionError("detector timing transform left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(custom, "_copytoscratch", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "cpu", reject_host_or_numpy)
            patch.setattr(torch.Tensor, "numpy", reject_host_or_numpy)
            output = custom.transform(inputs)

        results = {name: output[name] for name in ("dh", "dhl")}
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


@pytest.mark.parametrize("with_psd", (False, True))
def test_gaussian_noise_psd_setup_stays_on_torch_device(
        torch_device_ctx, monkeypatch, with_psd):
    ctx, device = torch_device_ctx
    dtype = np.float32 if device == "mps" else np.float64
    delta_f = 0.125
    data_values = np.linspace(-1.0, 1.0, 9, dtype=dtype)

    class ConcreteGaussianNoise(inference_gaussian_noise.BaseGaussianNoise):
        def _loglr(self):
            return 0.

    with ctx:
        data = FrequencySeries(data_values, delta_f=delta_f)
        if with_psd:
            psd_values = (
                1.25 + 0.25 * np.cos(np.linspace(0, np.pi, len(data)))
            ).astype(dtype)
            psds = {
                "H1": FrequencySeries(psd_values, delta_f=delta_f)
            }
        else:
            psd_values = np.ones(len(data), dtype=dtype)
            psds = None

        def reject_host_allocation(*_args, **_kwargs):
            raise AssertionError("Gaussian-noise PSD setup used NumPy storage")

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("Gaussian-noise PSD setup left Torch")

        with monkeypatch.context() as patch:
            patch.setattr(
                inference_gaussian_noise.numpy,
                "ones",
                reject_host_allocation,
            )
            patch.setattr(
                inference_gaussian_noise.numpy,
                "zeros",
                reject_host_allocation,
            )
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            model = ConcreteGaussianNoise(
                variable_params=(),
                data={"H1": data},
                low_frequency_cutoff={"H1": 0.25},
                psds=psds,
            )

    expected_invp = np.zeros(9, dtype=dtype)
    analysis_slice = slice(model.kmin["H1"], model.kmax["H1"])
    expected_invp[analysis_slice] = 1. / psd_values[analysis_slice]
    expected_weight = np.sqrt(4 * delta_f * expected_invp)
    expected_whitened = data_values * expected_weight
    for series in (
            model.psds["H1"],
            model._invpsds["H1"],
            model.weight["H1"],
            model.whitened_data["H1"]):
        assert isinstance(series._data, TorchArrayData)
        assert series._data.tensor.device.type == device
        assert series.dtype == np.dtype(dtype)

    np.testing.assert_array_equal(
        model.psds["H1"]._data.tensor.detach().cpu().numpy(),
        psd_values,
    )
    np.testing.assert_array_equal(
        model._invpsds["H1"]._data.tensor.detach().cpu().numpy(),
        expected_invp,
    )
    np.testing.assert_allclose(
        model.weight["H1"]._data.tensor.detach().cpu().numpy(),
        expected_weight,
    )
    np.testing.assert_allclose(
        model.whitened_data["H1"]._data.tensor.detach().cpu().numpy(),
        expected_whitened,
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_coincident_snr_keeps_selected_triggers_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    snrs = {
        "H1": np.array(
            [1 + 1j, 3 + 4j, 2 + 0j, 0.5 + 0j,
             6 + 0j, 1 - 3j, 4 + 4j, 0 + 0j],
            dtype=dtype,
        ),
        "L1": np.array(
            [2 + 0j, 1 + 2j, 4 + 3j, 0 + 0j,
             1 + 1j, 5 + 0j, 2 - 2j, 3 + 0j],
            dtype=dtype,
        ),
    }
    index = np.arange(6)
    time_delay_idx = {"H1": 0, "L1": 1}
    expected = coherent.coincident_snr(
        snrs, index, 5.0, time_delay_idx
    )

    with ctx:
        torch_snrs = {
            ifo: TimeSeries(values, delta_t=1.0 / 4096)
            for ifo, values in snrs.items()
        }
        with monkeypatch.context() as patch:
            original_cpu = torch.Tensor.cpu
            host_selections = []

            def _reject_host_transfer(_self):
                raise AssertionError(
                    "coincident SNR copied detector triggers to host"
                )

            def _guard_selection_transfer(tensor, *args, **kwargs):
                if tensor.dtype == torch.bool:
                    raise AssertionError(
                        "coincident SNR copied its boolean mask to host"
                    )
                host_selections.append(tensor.numel())
                return original_cpu(tensor, *args, **kwargs)

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _guard_selection_transfer)
            actual = coherent.coincident_snr(
                torch_snrs, index, 5.0, time_delay_idx
            )

    actual_rho, actual_index, actual_triggers = actual
    expected_rho, expected_index, expected_triggers = expected
    assert isinstance(actual_rho, Array)
    assert actual_rho._data.tensor.device.type == device
    assert actual_rho.dtype == expected_rho.dtype
    np.testing.assert_allclose(
        actual_rho._data.tensor.detach().cpu().numpy(),
        expected_rho,
        rtol=2e-6,
    )
    np.testing.assert_array_equal(actual_index, expected_index)
    assert host_selections == [len(expected_index)]
    for ifo, trigger in actual_triggers.items():
        assert isinstance(trigger, Array)
        assert trigger._data.tensor.device.type == device
        np.testing.assert_allclose(
            trigger._data.tensor.detach().cpu().numpy(),
            expected_triggers[ifo],
        )


@pytest.mark.parametrize("projection_name", ("standard", "left"))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_coherent_snr_keeps_selected_triggers_on_torch_device(
        torch_device_ctx, monkeypatch, dtype, projection_name):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    snrs = {
        "H1": np.array(
            [3 + 4j, 2 - 0.5j, -1 + 2j, 0.5 + 0.75j], dtype=dtype
        ),
        "L1": np.array(
            [1 - 2j, -3 + 1j, 0.5 + 0.75j, 4 - 0.25j], dtype=dtype
        ),
        "V1": np.array(
            [0.5 + 1j, 1.5 - 2j, 2 + 0.5j, -1 + 3j], dtype=dtype
        ),
    }
    f_plus = {"H1": 0.7, "L1": -0.2, "V1": 0.4}
    f_cross = {"H1": 0.1, "L1": 0.8, "V1": -0.5}
    sigma = {"H1": 1.0, "L1": 0.9, "V1": 0.8}
    projection = coherent.get_projection_matrix(
        f_plus, f_cross, sigma, projection=projection_name
    )
    index = np.arange(4)
    coinc_snr = np.array([5.5, 4.5, 3.0, 5.0])
    expected = coherent.coherent_snr(
        snrs, index, 3.0, projection, coinc_snr
    )

    with ctx:
        torch_snrs = {ifo: Array(values) for ifo, values in snrs.items()}
        with monkeypatch.context() as patch:
            original_cpu = torch.Tensor.cpu
            host_selections = []

            def _reject_host_transfer(_self):
                raise AssertionError(
                    "coherent SNR copied detector triggers to host"
                )

            def _guard_selection_transfer(tensor, *args, **kwargs):
                if tensor.dtype == torch.bool:
                    raise AssertionError(
                        "coherent SNR copied its boolean mask to host"
                    )
                host_selections.append(tensor.numel())
                return original_cpu(tensor, *args, **kwargs)

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", _guard_selection_transfer)
            actual = coherent.coherent_snr(
                torch_snrs, index, 3.0, projection, Array(coinc_snr)
            )

    actual_rho, actual_index, actual_snrs, actual_coinc = actual
    expected_rho, expected_index, expected_snrs, expected_coinc = expected
    assert isinstance(actual_rho, Array)
    assert actual_rho._data.tensor.device.type == device
    assert actual_rho.dtype == expected_rho.dtype
    np.testing.assert_allclose(
        actual_rho._data.tensor.detach().cpu().numpy(),
        expected_rho,
        rtol=2e-6,
    )
    np.testing.assert_array_equal(actual_index, expected_index)
    assert host_selections == [len(expected_index)]
    assert isinstance(actual_coinc, Array)
    assert actual_coinc._data.tensor.device.type == device
    np.testing.assert_array_equal(
        actual_coinc._data.tensor.detach().cpu().numpy(), expected_coinc
    )
    for ifo, trigger in actual_snrs.items():
        assert isinstance(trigger, Array)
        assert trigger._data.tensor.device.type == device
        np.testing.assert_allclose(
            trigger._data.tensor.detach().cpu().numpy(),
            expected_snrs[ifo],
            rtol=2e-6,
        )


def test_coherent_chisq_cache_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx

    with ctx:
        reference = Array(np.zeros(8, dtype=np.float32))
        indices = Array(
            TorchArrayData(
                torch.tensor([5, 1, 3], dtype=torch.int64, device=device)
            ),
            copy=False,
        )
        values = Array(np.array([2.5, 3.5, 4.5], dtype=np.float32))
        dof_values = Array(np.array([4, 6, 8], dtype=np.int64))

        def reject_host_transfer(*_args, **_kwargs):
            raise AssertionError("coherent chi-squared cache copied to host")

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
            patch.setattr(torch.Tensor, "item", reject_host_transfer)
            cache = coherent.create_coherent_cache(
                reference, np.nan, np.float32
            )
            dof_cache = coherent.create_coherent_cache(
                reference, 0, np.int32
            )
            missing = coherent.unavailable_coherent_indices(cache, indices)
            coherent.update_coherent_cache(cache, missing, values)
            coherent.update_coherent_cache(dof_cache, missing, dof_values)
            remaining = coherent.unavailable_coherent_indices(cache, indices)
            selected = cache[indices]
            selected_dof = dof_cache[indices]

    for value in (cache, dof_cache, missing, remaining):
        assert isinstance(value, Array)
        assert value._data.tensor.device.type == device
    for value in (selected, selected_dof):
        assert isinstance(value, TorchArrayData)
        assert value.tensor.device.type == device
    assert cache.dtype == np.dtype(np.float32)
    assert dof_cache.dtype == np.dtype(np.int32)
    assert len(remaining) == 0
    torch.testing.assert_close(
        selected.tensor,
        torch.tensor([2.5, 3.5, 4.5], device=device),
    )
    torch.testing.assert_close(
        selected_dof.tensor,
        torch.tensor([4, 6, 8], dtype=torch.int32, device=device),
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


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("summaries_on_torch", (False, True))
def test_network_chisq_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype, summaries_on_torch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    snrs = {
        "H1": np.array([3 + 4j, 2 - 0.5j, -1 + 2j], dtype=dtype),
        "L1": np.array([1 - 2j, -3 + 1j, 0.5 + 0.75j], dtype=dtype),
    }
    real_dtype = np.empty((), dtype=dtype).real.dtype
    chisq = {
        "H1": np.array([2.5, 4.0, 1.5], dtype=real_dtype),
        "L1": np.array([3.0, 2.0, 5.0], dtype=real_dtype),
    }
    dof = {"H1": 4.0, "L1": 6.0}
    expected = coherent.network_chisq(chisq, dof, snrs)

    with ctx:
        torch_snrs = {
            ifo: Array(values)._data for ifo, values in snrs.items()
        }
        summaries = (
            {ifo: Array(values) for ifo, values in chisq.items()}
            if summaries_on_torch
            else chisq
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("network chisq copied Torch data to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = coherent.network_chisq(summaries, dof, torch_snrs)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(real_dtype)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=2e-6, atol=2e-7,
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
def test_null_snr_reweighting_stays_on_torch_device(
        torch_device_ctx, monkeypatch, dtype):
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64 PyCBC arrays")

    network = np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    null = np.array([2.0, 5.0, 8.0, 12.0], dtype=dtype)
    coherent_snr = np.array([10.0, 18.0, 25.0, 30.0], dtype=dtype)
    expected = coherent.reweight_snr_by_null(
        network, null, coherent_snr
    )
    expected = coherent.reweightedsnr_cut(expected, 15.0)

    with ctx:
        torch_network = Array(network)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("null-SNR reweighting copied to host")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            actual = coherent.reweight_snr_by_null(
                torch_network, null, coherent_snr
            )
            actual = coherent.reweightedsnr_cut(actual, 15.0)

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
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


def test_lal_detector_projection_is_native_on_torch(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    delta_t = 1 / 2048
    epoch = 1_126_259_462.1234567
    times = np.arange(4096) * delta_t
    hp_data = (
        np.sin(2 * np.pi * 137 * times)
        + 0.07 * np.sin(2 * np.pi * 13.3 * times)
    )
    hc_data = (
        0.4 * np.cos(2 * np.pi * 137 * times)
        - 0.02 * np.cos(2 * np.pi * 7.1 * times)
    )
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
    )

    with ctx:
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("Detector projection copied data to host")

            def _reject_lalsimulation(*_args, **_kwargs):
                raise AssertionError("Detector projection used lalsimulation")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                lalsimulation,
                "SimDetectorStrainREAL8TimeSeries",
                _reject_lalsimulation,
            )
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
    assert len(projected) == len(expected)
    assert projected.delta_t == expected.delta_t
    assert projected.start_time == expected.start_time

    actual_data = projected._data.tensor.detach().cpu().numpy()
    expected_data = expected.numpy().astype(input_dtype, copy=False)
    tolerance = 5e-5 if device == "mps" else 1e-12
    assert _relative_l2(actual_data, expected_data) < tolerance
    np.testing.assert_allclose(
        actual_data,
        expected_data,
        rtol=0,
        atol=1e-5 if device == "mps" else 5e-14,
    )


def test_detector_antenna_pattern_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    detector = Detector("H1")
    epoch = 1_126_259_462.1234567
    right_ascension = np.array((0.2, 1.2, 2.4, 5.7))
    declination = -0.4
    polarization = np.array((0.0, 0.3, 1.1, 2.7))
    expected = detector.antenna_pattern(
        right_ascension,
        declination,
        polarization,
        epoch,
    )
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        right_ascension_tensor = torch.as_tensor(
            right_ascension, device=device, dtype=dtype
        )
        polarization_tensor = torch.as_tensor(
            polarization, device=device, dtype=dtype
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("antenna pattern copied angles to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = detector.antenna_pattern(
                right_ascension_tensor,
                declination,
                polarization_tensor,
                epoch,
            )

    tolerance = 2e-6 if device == "mps" else 2e-12
    for result, reference in zip(actual, expected):
        assert result.device.type == device
        assert result.dtype == dtype
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            reference,
            rtol=tolerance,
            atol=tolerance,
        )


def test_detector_antenna_pattern_tensor_times_stays_on_torch(monkeypatch):
    detector = Detector("H1")
    reference_time = detector.reference_time
    right_ascension = np.array(((0.2,), (2.4,)))
    gps_times = reference_time + np.array(((-10.0, 0.0, 10.0),))
    declination = -0.4
    polarization = 0.3
    expected = detector.antenna_pattern(
        right_ascension, declination, polarization, gps_times
    )

    with scheme.TorchScheme("cpu"):
        right_ascension_tensor = torch.tensor(
            right_ascension, dtype=torch.float64, requires_grad=True
        )
        gps_times_tensor = torch.tensor(
            gps_times, dtype=torch.float64, requires_grad=True
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("antenna pattern copied times to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = detector.antenna_pattern(
                right_ascension_tensor,
                declination,
                polarization,
                gps_times_tensor,
            )
            sum(value.square().sum() for value in actual).backward()

    for result, reference in zip(actual, expected):
        assert result.device.type == "cpu"
        assert result.dtype == torch.float64
        np.testing.assert_allclose(
            result.detach().numpy(), reference, rtol=2e-12, atol=2e-12
        )
    assert torch.isfinite(right_ascension_tensor.grad).all()
    assert torch.isfinite(gps_times_tensor.grad).all()


def test_fused_detector_response_matches_public_torch_kernels():
    if _INFERENCE_RELBIN_TORCH is None:
        pytest.skip("pycbc.inference.models.relbin_torch is not in this PR")
    detector = Detector("H1")
    reference_time = detector.reference_time
    right_ascension = np.array(((0.2,), (2.4,)))
    declination = np.array(((-0.4,), (0.7,)))
    gps_times = reference_time + np.array(((-10.0, 0.0, 10.0),))

    with scheme.TorchScheme("cpu"):
        right_ascension = torch.tensor(
            right_ascension, dtype=torch.float64, requires_grad=True
        )
        declination = torch.tensor(
            declination, dtype=torch.float64, requires_grad=True
        )
        gps_times = torch.tensor(
            gps_times, dtype=torch.float64, requires_grad=True
        )
        like = torch.ones((), dtype=torch.complex128)
        actual = _INFERENCE_RELBIN_TORCH.detector_response(
            detector, right_ascension, declination, gps_times, like
        )
        expected = (*detector.antenna_pattern(
            right_ascension, declination, 0.0, gps_times
        ), detector.time_delay_from_earth_center(
            right_ascension, declination, gps_times
        ))
        sum(value.square().sum() for value in actual).backward()

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result.detach(), reference.detach(), rtol=2e-12, atol=2e-12
        )
    for value in (right_ascension, declination, gps_times):
        assert torch.isfinite(value.grad).all()


def test_detector_response_contraction_preserves_device_dtype_and_gradients(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 2e-6 if device == "mps" else 2e-12
    response_values = (
        torch.arange(9, dtype=torch.float64).reshape(3, 3) / 7.0 - 0.4
    )
    basis_values = (
        torch.arange(24, dtype=torch.float64).reshape(3, 2, 4) / 11.0 - 0.7
    )

    with ctx:
        monkeypatch.setattr(
            _INFERENCE_RELBIN_TORCH,
            "_EXPLICIT_RESPONSE_MIN_THREADS",
            0,
        )
        response = response_values.to(device=device, dtype=dtype)
        basis = basis_values.to(device=device, dtype=dtype)
        response.requires_grad_()
        basis.requires_grad_()
        actual = _INFERENCE_RELBIN_TORCH._contract_detector_response(
            response, basis
        )
        actual_gradients = torch.autograd.grad(
            actual.square().sum(), (response, basis)
        )

        expected_response = response_values.to(device=device, dtype=dtype)
        expected_basis = basis_values.to(device=device, dtype=dtype)
        expected_response.requires_grad_()
        expected_basis.requires_grad_()
        expected = torch.einsum(
            "ij,j...->i...", expected_response, expected_basis
        )
        expected_gradients = torch.autograd.grad(
            expected.square().sum(), (expected_response, expected_basis)
        )

    torch.testing.assert_close(
        actual.detach(), expected.detach(), rtol=tolerance, atol=tolerance
    )
    for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.detach(),
            expected_gradient.detach(),
            rtol=tolerance,
            atol=tolerance,
        )


def test_explicit_antenna_response_preserves_dtype_and_gradients(
        torch_device_ctx):
    ctx, device = torch_device_ctx
    dtype = torch.float32 if device == "mps" else torch.float64
    tolerance = 3e-6 if device == "mps" else 2e-12
    response_values = (
        torch.arange(9, dtype=torch.float64).reshape(3, 3) / 7.0 - 0.4
    )
    basis_values = (
        torch.arange(40, dtype=torch.float64).reshape(5, 2, 4) / 11.0
        - 0.7
    )

    with ctx:
        response = response_values.to(device=device, dtype=dtype)
        components = tuple(
            value.to(device=device, dtype=dtype)
            for value in basis_values
        )
        response.requires_grad_()
        components = tuple(value.requires_grad_() for value in components)
        actual = _INFERENCE_RELBIN_TORCH._explicit_antenna_response(
            response, *components
        )
        actual_gradients = torch.autograd.grad(
            sum(value.square().sum() for value in actual),
            (response, *components),
        )

        expected_response = response_values.to(device=device, dtype=dtype)
        expected_components = tuple(
            value.to(device=device, dtype=dtype)
            for value in basis_values
        )
        expected_response.requires_grad_()
        expected_components = tuple(
            value.requires_grad_() for value in expected_components
        )
        x0, x1, y0, y1, y2 = expected_components
        zero = torch.zeros_like(x0)
        x = torch.stack((x0, x1, zero))
        y = torch.stack((y0, y1, y2))
        dx = torch.einsum("ij,j...->i...", expected_response, x)
        dy = torch.einsum("ij,j...->i...", expected_response, y)
        expected = (
            torch.sum(x * dx - y * dy, dim=0),
            torch.sum(x * dy + y * dx, dim=0),
        )
        expected_gradients = torch.autograd.grad(
            sum(value.square().sum() for value in expected),
            (expected_response, *expected_components),
        )

    for result, reference in zip(actual, expected):
        assert result.device.type == device
        assert result.dtype == dtype
        torch.testing.assert_close(
            result.detach(), reference.detach(),
            rtol=tolerance, atol=tolerance,
        )
    for result, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            result.detach(), reference.detach(),
            rtol=tolerance, atol=tolerance,
        )


def test_explicit_antenna_response_preserves_high_thread_reduction_bits(
        monkeypatch):
    response = (
        torch.arange(9, dtype=torch.float64).reshape(3, 3) / 7.0 - 0.4
    )
    x0, x1, y0, y1, y2 = tuple(
        torch.arange(32, dtype=torch.float64).reshape(4, 8) / scale
        for scale in (11.0, 13.0, 17.0, 19.0, 23.0)
    )
    x = torch.stack((x0, x1, torch.zeros_like(x0)))
    y = torch.stack((y0, y1, y2))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 64)
    dx = _INFERENCE_RELBIN_TORCH._contract_detector_response(response, x)
    dy = _INFERENCE_RELBIN_TORCH._contract_detector_response(response, y)
    expected = (
        torch.sum(x * dx - y * dy, dim=0),
        torch.sum(x * dy + y * dx, dim=0),
    )

    actual = _INFERENCE_RELBIN_TORCH._explicit_antenna_response(
        response, x0, x1, y0, y1, y2
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_fused_detector_response_scalar_sky_grid_preserves_gradients():
    detector = Detector("H1")
    reference_time = detector.reference_time
    time_values = reference_time + np.linspace(-12.0, 12.0, 17)

    with scheme.TorchScheme("cpu"):
        right_ascension = torch.tensor(
            0.73, dtype=torch.float64, requires_grad=True
        )
        declination = torch.tensor(
            -0.41, dtype=torch.float64, requires_grad=True
        )
        gps_times = torch.tensor(
            time_values, dtype=torch.float64, requires_grad=True
        )
        like = torch.ones((), dtype=torch.complex128)
        actual = _INFERENCE_RELBIN_TORCH.detector_response(
            detector, right_ascension, declination, gps_times, like
        )
        actual_gradients = torch.autograd.grad(
            sum((index + 1) * value.square().sum()
                for index, value in enumerate(actual)),
            (right_ascension, declination, gps_times),
        )

        expected_right_ascension = right_ascension.detach().requires_grad_()
        expected_declination = declination.detach().requires_grad_()
        expected_times = gps_times.detach().requires_grad_()
        expected = (*detector.antenna_pattern(
            expected_right_ascension,
            expected_declination,
            0.0,
            expected_times,
        ), detector.time_delay_from_earth_center(
            expected_right_ascension,
            expected_declination,
            expected_times,
        ))
        expected_gradients = torch.autograd.grad(
            sum((index + 1) * value.square().sum()
                for index, value in enumerate(expected)),
            (
                expected_right_ascension,
                expected_declination,
                expected_times,
            ),
        )

    for result, reference in zip(actual, expected):
        assert result.shape == gps_times.shape
        torch.testing.assert_close(
            result.detach(), reference.detach(), rtol=2e-12, atol=2e-12
        )
    for result, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result.detach(), reference.detach(), rtol=2e-10, atol=2e-12
        )


def test_detector_single_arm_response_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    frequencies = np.array((0.0, 100.0, 1_000.0, 10_000.0))
    directions = np.array((-0.999, -0.5, 0.3, 0.999))
    expected = np.ones(4, dtype=np.complex128)
    expected[1:] = single_arm_frequency_response(
        frequencies[1:], directions[1:], 4_000.0
    )
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        frequency_tensor = torch.tensor(
            frequencies, device=device, dtype=dtype, requires_grad=True
        )
        direction_tensor = torch.tensor(
            directions, device=device, dtype=dtype, requires_grad=True
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("finite-arm response copied to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = single_arm_frequency_response(
                frequency_tensor, direction_tensor, 4_000.0
            )
            actual.abs().square().sum().backward()

    expected_dtype = (
        torch.complex64 if dtype == torch.float32 else torch.complex128
    )
    assert actual.device.type == device
    assert actual.dtype == expected_dtype
    assert torch.isfinite(frequency_tensor.grad).all()
    assert torch.isfinite(direction_tensor.grad).all()
    tolerance = 2e-6 if device == "mps" else 1e-10
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_detector_finite_arm_antenna_pattern_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    detector = Detector("H1")
    epoch = 1_126_259_462.1234567
    right_ascension = np.array((0.2, 0.4, 0.6, 0.8))
    declination = np.array((0.3, 0.2, 0.1, 0.0))
    polarization = np.array((0.4, 0.3, 0.2, 0.1))
    frequencies = np.array((0.0, 200.0, 1_000.0, 10_000.0))
    expected = tuple(zip(*(
        detector.antenna_pattern(
            right_ascension[index],
            declination[index],
            polarization[index],
            epoch,
            frequency=frequencies[index],
        )
        for index in range(len(frequencies))
    )))
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        right_ascension_tensor = torch.tensor(
            right_ascension,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        frequency_tensor = torch.tensor(
            frequencies,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("finite-arm antenna response left device")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = detector.antenna_pattern(
                right_ascension_tensor,
                torch.tensor(declination, device=device, dtype=dtype),
                torch.tensor(polarization, device=device, dtype=dtype),
                epoch,
                frequency=frequency_tensor,
            )
            sum(value.abs().square().sum() for value in actual).backward()

    expected_dtype = (
        torch.complex64 if dtype == torch.float32 else torch.complex128
    )
    tolerance = 3e-6 if device == "mps" else 2e-12
    for result, reference in zip(actual, expected):
        assert result.device.type == device
        assert result.dtype == expected_dtype
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            reference,
            rtol=tolerance,
            atol=tolerance,
        )
    assert torch.isfinite(right_ascension_tensor.grad).all()
    assert torch.isfinite(frequency_tensor.grad).all()


def test_detector_effective_distance_stays_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    detector = Detector("H1")
    distance = np.array((100.0, 200.0, 300.0, 400.0))
    right_ascension = np.array((0.2, 1.2, 2.4, 5.7))
    declination = np.array((-0.4, 0.1, 0.5, -0.2))
    polarization = np.array((0.0, 0.3, 1.1, 2.7))
    epoch = 1_126_259_462.1234567
    inclination = np.array((0.1, 0.5, 1.0, 1.4))
    expected = detector.effective_distance(
        distance,
        right_ascension,
        declination,
        polarization,
        epoch,
        inclination,
    )
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        distance_tensor = torch.tensor(
            distance, device=device, dtype=dtype, requires_grad=True
        )
        right_ascension_tensor = torch.tensor(
            right_ascension, device=device, dtype=dtype, requires_grad=True
        )
        inclination_tensor = torch.tensor(
            inclination, device=device, dtype=dtype, requires_grad=True
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("effective distance copied to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual = detector.effective_distance(
                distance_tensor,
                right_ascension_tensor,
                torch.tensor(declination, device=device, dtype=dtype),
                torch.tensor(polarization, device=device, dtype=dtype),
                epoch,
                inclination_tensor,
            )
            actual.sum().backward()

    tolerance = 3e-5 if device == "mps" else 2e-12
    assert actual.device.type == device
    assert actual.dtype == dtype
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    assert torch.isfinite(distance_tensor.grad).all()
    assert torch.isfinite(right_ascension_tensor.grad).all()
    assert torch.isfinite(inclination_tensor.grad).all()


def test_detector_time_delays_stay_on_torch_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    detector = Detector("H1")
    other_detector = Detector("L1")
    epoch = 1_126_259_462.1234567
    right_ascension = np.array((0.2, 1.2, 2.4, 5.7))
    declination = np.array((-0.4, 0.1, 0.5, -0.2))
    expected_earth = detector.time_delay_from_earth_center(
        right_ascension, declination, epoch
    )
    expected_detector = detector.time_delay_from_detector(
        other_detector, right_ascension, declination, epoch
    )
    dtype = torch.float32 if device == "mps" else torch.float64

    with ctx:
        right_ascension_tensor = torch.tensor(
            right_ascension,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        declination_tensor = torch.tensor(
            declination,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("detector timing copied angles to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual_earth = detector.time_delay_from_earth_center(
                right_ascension_tensor, declination_tensor, epoch
            )
            actual_detector = detector.time_delay_from_detector(
                other_detector,
                right_ascension_tensor,
                declination_tensor,
                epoch,
            )
            (actual_earth.square().sum()
             + actual_detector.square().sum()).backward()

    tolerance = 2e-7 if device == "mps" else 2e-12
    for result, reference in (
            (actual_earth, expected_earth),
            (actual_detector, expected_detector)):
        assert result.device.type == device
        assert result.dtype == dtype
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            reference,
            rtol=tolerance,
            atol=tolerance,
        )
    assert torch.isfinite(right_ascension_tensor.grad).all()
    assert torch.isfinite(declination_tensor.grad).all()


def test_detector_tensor_times_and_arrival_time_stay_on_torch(monkeypatch):
    detector = Detector("H1")
    reference_time = detector.reference_time
    right_ascension = np.array(((0.2,), (2.4,)))
    gps_times = reference_time + np.array(((-10.0, 0.0, 10.0),))
    declination = -0.4
    expected_delay = detector.time_delay_from_earth_center(
        right_ascension, declination, gps_times
    )
    expected_arrival = gps_times + expected_delay

    with scheme.TorchScheme("cpu"):
        right_ascension_tensor = torch.tensor(
            right_ascension, dtype=torch.float64, requires_grad=True
        )
        gps_times_tensor = torch.tensor(
            gps_times, dtype=torch.float64, requires_grad=True
        )
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self, *_args, **_kwargs):
                raise AssertionError("detector timing copied times to host")

            patch.setattr(torch.Tensor, "cpu", _reject_host_transfer)
            patch.setattr(torch.Tensor, "numpy", _reject_host_transfer)
            actual_delay = detector.time_delay_from_earth_center(
                right_ascension_tensor, declination, gps_times_tensor
            )
            actual_arrival = detector.arrival_time(
                gps_times_tensor, right_ascension_tensor, declination
            )
            actual_arrival.sum().backward()

    np.testing.assert_allclose(
        actual_delay.detach().numpy(), expected_delay, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual_arrival.detach().numpy(),
        expected_arrival,
        rtol=0,
        atol=2e-12,
    )
    assert torch.isfinite(right_ascension_tensor.grad).all()
    assert torch.isfinite(gps_times_tensor.grad).all()


def test_detector_tensor_time_input_validation(torch_ctx):
    with torch_ctx:
        detector = Detector("H1")
        result = detector.time_delay_from_earth_center(
            torch.tensor((0, 1)), -0.4, detector.reference_time
        )
        assert result.dtype == torch.get_default_dtype()
        with pytest.raises(TypeError, match="must be real"):
            detector.time_delay_from_earth_center(
                torch.tensor(0.2 + 0.1j), -0.4, detector.reference_time
            )
        with pytest.raises(NotImplementedError, match="GMST reference"):
            Detector("H1", reference_time=None).time_delay_from_earth_center(
                0.2, -0.4, torch.tensor(detector.reference_time)
            )


def test_detector_sine_integral_stays_on_device(torch_device_ctx):
    from pycbc.detector.ground_torch import _torch_sine_integral

    ctx, device = torch_device_ctx
    values = np.array(
        (
            -100.0,
            -8.0,
            -4.0,
            -0.25,
            -1e-12,
            0.0,
            1e-12,
            0.25,
            4.0,
            8.0,
            100.0,
        )
    )
    dtype = torch.float32 if device == "mps" else torch.float64
    expected = scipy.special.sici(values)[0]

    with ctx:
        actual = _torch_sine_integral(
            torch.as_tensor(values, dtype=dtype, device=device)
        )

    assert actual.device.type == device
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=1e-6 if device == "mps" else 2e-13,
        atol=1e-6 if device == "mps" else 2e-13,
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


def test_single_template_host_scalar_matches_generic_real_detector(
        monkeypatch):
    ifo = "H1"
    epoch = 1126259462.0
    delta_t = 1.0 / 128.0
    tc = epoch + 2.003
    samples = 513
    sample_indices = np.arange(samples, dtype=np.float64)
    sh_values = (
        (1.0 + 0.002 * sample_indices)
        * np.exp(0.017j * sample_indices)
    ).astype(np.complex128)
    parameters = {
        "inclination": np.float64(0.71),
        "polarization": np.float64(0.39),
        "coa_phase": np.float64(0.23),
        "distance": np.float64(1.8),
        "ra": np.float64(1.0),
        "dec": np.float64(-0.4),
        "tc": np.float64(tc),
    }

    def make_model(series):
        model = types.SimpleNamespace(
            sh={ifo: series},
            hh={ifo: np.float64(1.7)},
            snr={},
            det={ifo: Detector(ifo)},
            dts={},
            htfs={},
            current_params=dict(parameters),
        )
        model.snr_draw = lambda **_kwargs: None
        model.marginalize_loglr = (
            lambda sh, norm: sh.real - 0.5 * norm
        )
        return model

    with scheme.TorchScheme("cpu"):
        series = TimeSeries(sh_values, delta_t=delta_t, epoch=epoch)
        host_model = make_model(series)
        host_value = single_template.SingleTemplate._loglr(host_model)

        generic_model = make_model(series)
        with monkeypatch.context() as patch:
            patch.setattr(
                single_template, "_plain_host_scalar_extrinsics",
                lambda _parameters: False,
            )
            patch.setattr(
                single_template, "_host_scalar_extrinsics",
                lambda _parameters: False,
            )
            generic_value = single_template.SingleTemplate._loglr(
                generic_model
            )

        host_result = host_value.detach().cpu().numpy()
        generic_result = generic_value.detach().cpu().numpy()
        generic_dt = generic_model.dts[ifo].detach().cpu().numpy()
        generic_factor = generic_model.htfs[ifo].detach().cpu().numpy()

    host_dt = host_model.dts[ifo]
    relative_sample = (host_dt - epoch) / delta_t
    assert abs(host_dt - tc) > 1e-5
    assert abs(relative_sample - round(relative_sample)) > 1e-3
    np.testing.assert_allclose(host_dt, generic_dt, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(
        host_model.htfs[ifo], generic_factor, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        host_result, generic_result, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("ifo", ("H1", "L1"))
def test_single_template_torch_cpu_scalar_detector_projection_bit_exact(ifo):
    detector = Detector(ifo)
    parameters = {"ra": np.float64(1.2), "dec": np.float64(-0.4)}

    for offset in (-0.019, -0.003, 0.0, 0.007, 0.023):
        parameters["tc"] = np.float64(1126259466.003 + offset)
        expected_fp, expected_fc = detector.antenna_pattern(
            parameters["ra"], parameters["dec"], 0, parameters["tc"]
        )
        expected_dt = detector.time_delay_from_earth_center(
            parameters["ra"], parameters["dec"], parameters["tc"]
        )

        actual = single_template._torch_cpu_scalar_detector_projection(
            detector, parameters
        )
        np.testing.assert_allclose(
            actual, (expected_fp, expected_fc, expected_dt),
            rtol=1e-15, atol=1e-15
        )


def test_single_template_torch_cpu_scalar_detector_projection_guards(
        monkeypatch):
    parameters = {
        "ra": np.float64(1.2),
        "dec": np.float64(-0.4),
        "tc": np.float64(1126259466.003),
    }

    class _DetectorSubclass(Detector):
        pass

    assert single_template._torch_cpu_scalar_detector_projection(
        Detector("H1"), parameters
    ) is not None
    assert single_template._torch_cpu_scalar_detector_projection(
        _DetectorSubclass("H1"), parameters
    ) is None

    for value in (
        np.float32(parameters["tc"]),
        np.array(parameters["tc"]),
        torch.tensor(parameters["tc"], dtype=torch.float64),
        float("inf"),
    ):
        candidate = {**parameters, "tc": value}
        assert single_template._torch_cpu_scalar_detector_projection(
            Detector("H1"), candidate
        ) is None

    for name in (
        "antenna_pattern",
        "time_delay_from_earth_center",
        "time_delay_from_location",
        "gmst_estimate",
        "set_gmst_reference",
    ):
        detector = Detector("H1")
        original = getattr(detector, name)
        setattr(
            detector, name,
            types.MethodType(
                lambda _self, *args, _original=original, **kwargs:
                    _original(*args, **kwargs),
                detector,
            ),
        )
        assert single_template._torch_cpu_scalar_detector_projection(
            detector, parameters
        ) is None

    detector = Detector("H1")
    detector.response = detector.response.astype(np.float32)
    assert single_template._torch_cpu_scalar_detector_projection(
        detector, parameters
    ) is None

    detector = Detector("H1")
    detector.response = detector.response.copy()
    assert single_template._torch_cpu_scalar_detector_projection(
        detector, parameters
    ) is None

    detector = Detector("H1")
    detector.location = detector.location.astype(np.float32)
    assert single_template._torch_cpu_scalar_detector_projection(
        detector, parameters
    ) is None

    detector = Detector("H1")
    detector.location = detector.location.copy()
    assert single_template._torch_cpu_scalar_detector_projection(
        detector, parameters
    ) is None

    detector = Detector("H1", reference_time=None)
    assert single_template._torch_cpu_scalar_detector_projection(
        detector, parameters
    ) is None

    detector = Detector("H1")
    with monkeypatch.context() as patch:
        patch.setattr(
            Detector, "antenna_pattern",
            lambda self, *args, **kwargs: (0.0, 0.0),
        )
        assert single_template._torch_cpu_scalar_detector_projection(
            detector, parameters
        ) is None


def test_single_template_detector_method_sentinel_import_order():
    code = r'''
import sys
import types

import lal

try:
    import lal.utils
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.detector import Detector

assert "pycbc.inference.models.single_template" not in sys.modules
Detector.antenna_pattern = lambda self, *args, **kwargs: (0.0, 0.0)

from pycbc.inference.models import single_template

parameters = {"ra": 1.2, "dec": -0.4, "tc": 1126259466.003}
assert single_template._torch_cpu_scalar_detector_projection(
    Detector("H1"), parameters
) is None
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_single_template_torch_cpu_scalar_detector_custom_override_fallback(
        torch_ctx):
    with torch_ctx:
        model = _native_scalar_single_template_model()
        detector = model.det["H1"]
        original = detector.antenna_pattern
        calls = []

        def antenna_pattern(_self, *args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        detector.antenna_pattern = types.MethodType(
            antenna_pattern, detector
        )
        actual = model._loglr()

        assert type(actual) is float
        assert len(calls) == 1


@pytest.mark.parametrize("ad_mode", ("reverse", "forward"))
def test_single_template_torch_cpu_scalar_detector_autograd_fallback(
        torch_ctx, monkeypatch, ad_mode):
    with torch_ctx:
        model = _native_scalar_single_template_model()
        original = torch.tensor(
            model._current_params["ra"], dtype=torch.float64
        )

        def unexpected_projection(*_args, **_kwargs):
            raise AssertionError("autograd storage must use the generic path")

        monkeypatch.setattr(
            single_template,
            "_torch_cpu_scalar_detector_projection",
            unexpected_projection,
        )
        monkeypatch.setattr(
            single_template.DistMarg,
            "marginalize_loglr",
            lambda _self, sh, hh: sh.real - 0.5 * hh,
        )
        if ad_mode == "reverse":
            tensor = original.detach().requires_grad_()
            model._current_params["ra"] = tensor
            result = model._loglr()
            result.backward()
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
        else:
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(
                    original, torch.ones_like(original)
                )
                model._current_params["ra"] = dual
                result = model._loglr()
                primal, tangent = torch.autograd.forward_ad.unpack_dual(
                    result
                )
                assert torch.isfinite(primal)
                assert tangent is not None
                assert torch.isfinite(tangent)


@pytest.mark.parametrize("precalc_antenna_factors", (False, True))
def test_marginalized_time_antenna_stays_differentiable(
        torch_device_ctx, monkeypatch, precalc_antenna_factors):
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    torch_complex_dtype = (
        torch.complex64 if device == "mps" else torch.complex128
    )
    torch_real_dtype = torch.float32 if device == "mps" else torch.float64
    ifo = "H1"
    fp, fc, delay = real_dtype(0.43), real_dtype(-0.27), 0.0
    polarization_value = real_dtype(0.39)
    hpd_value = complex(1.2, 0.4)
    hcd_value = complex(-0.7, 0.9)
    detector_calls = []

    class _MatchedSeries:
        def __init__(self, tensor):
            self.tensor = tensor

        def __truediv__(self, value):
            return _MatchedSeries(self.tensor / value)

        def squared_norm(self):
            return self.tensor.abs().square().sum()

        def at_time(self, time, interpolate=None):
            assert interpolate == "quadratic"
            return self.tensor[0] * (1.0 + 0.01 * time)

    class _Detector:
        def arrival_time(self, ref_tc, ra, dec, ref_frame):
            detector_calls.append("arrival")
            if precalc_antenna_factors:
                return 0.0
            assert ref_frame == "geocentric"
            assert all(
                isinstance(value, torch.Tensor)
                and value.device.type == device
                for value in (ra, dec)
            )
            return (
                torch.as_tensor(
                    ref_tc, device=ra.device, dtype=ra.dtype)
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
            base_fp = fp + 0.01 * (ra - 1.0)
            base_fc = fc + 0.02 * (dec + 0.4)
            cosine = torch.cos(2.0 * polarization)
            sine = torch.sin(2.0 * polarization)
            return (
                base_fp * cosine + base_fc * sine,
                base_fc * cosine - base_fp * sine,
            )

    model = types.SimpleNamespace()
    model.all_ifodata_same_rate_length = True
    model.sample_rate = None
    model._kmin = {ifo: 0}
    model._kmax = {ifo: 4}
    model._f_lower = {ifo: 0.0}
    model._f_upper = {ifo: None}
    model.draw_ifos = lambda *_args, **_kwargs: None
    model.snr_draw = lambda **_kwargs: None
    model.kwargs = {}
    model.dets = {ifo: _Detector()}
    model.precalc_antenna_factors = precalc_antenna_factors
    model.get_precalc_antenna_factors = lambda _ifo: (fp, fc, delay)
    model.return_sh_hh = False
    model.marginalize_loglr = lambda sh, hh: sh.real - 0.5 * hh

    with ctx:
        polarization = torch.tensor(
            polarization_value, dtype=torch_real_dtype,
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
        matched = iter((
            _MatchedSeries(torch.tensor(
                [hpd_value], dtype=torch_complex_dtype, device=device)),
            _MatchedSeries(torch.tensor(
                [hcd_value], dtype=torch_complex_dtype, device=device)),
        ))

        def _matched_filter_core(*_args, **_kwargs):
            return next(matched), None, None

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "Marginalized time copied samples to the host")

            def _reject_numpy_projection(_value):
                raise AssertionError(
                    "Marginalized time antenna rotation used NumPy")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                marginalized_gaussian_noise.numpy,
                "exp",
                _reject_numpy_projection,
            )
            patch.setattr(
                sys.modules["pycbc.filter"],
                "matched_filter_core",
                _matched_filter_core,
            )
            loglr = marginalized_gaussian_noise.MarginalizedTime._loglr
            actual = loglr.__wrapped__(model)
            actual.backward()

        actual_value = actual.detach().cpu().numpy()
        gradient = polarization.grad.detach().cpu().numpy()
        sky_gradients = tuple(
            None if value.grad is None else value.grad.detach().cpu().numpy()
            for value in (right_ascension, declination)
        )

    response = (fp + 1.0j * fc) * np.exp(
        -2.0j * polarization_value)
    expected_sh = response.real * hpd_value + response.imag * hcd_value
    expected_hh = 4.0 * response.real ** 2 + 16.0 * response.imag ** 2
    expected = expected_sh.real - 0.5 * expected_hh

    assert actual.device.type == device
    expected_calls = (
        ["arrival"] if precalc_antenna_factors
        else ["arrival", "antenna"]
    )
    assert detector_calls == expected_calls
    assert np.isfinite(gradient)
    assert gradient != 0.0
    if precalc_antenna_factors:
        assert sky_gradients == (None, None)
    else:
        assert all(
            np.isfinite(value) and value != 0.0
            for value in sky_gradients
        )
    tolerance = 6e-4 if device == "mps" else 2e-12
    np.testing.assert_allclose(
        actual_value, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dominant_mode", (False, True))
def test_relative_time_snr_predictor_stays_on_device(
        torch_device_ctx, monkeypatch, dominant_mode):
    if _INFERENCE_RELBIN_TORCH is None:
        pytest.skip("pycbc.inference.models.relbin_torch is not in this PR")
    ctx, device = torch_device_ctx
    complex_dtype = np.complex64 if device == "mps" else np.complex128
    real_dtype = np.float32 if device == "mps" else np.float64
    rng = np.random.default_rng(72819)
    frequency_count, sample_count = 73, 37
    freqs = np.linspace(20.0, 600.0, frequency_count, dtype=real_dtype)
    hp = (rng.normal(size=frequency_count) +
          1j * rng.normal(size=frequency_count)).astype(complex_dtype)
    hc = (rng.normal(size=frequency_count) +
          1j * rng.normal(size=frequency_count)).astype(complex_dtype)
    h00 = (1.25 + rng.random(frequency_count) +
           1j * rng.normal(size=frequency_count)).astype(complex_dtype)
    a0 = (rng.normal(size=frequency_count - 1) +
          1j * rng.normal(size=frequency_count - 1)).astype(complex_dtype)
    a1 = (rng.normal(size=frequency_count - 1) +
          1j * rng.normal(size=frequency_count - 1)).astype(complex_dtype)
    b0 = rng.random(frequency_count - 1).astype(real_dtype)
    b1 = rng.normal(size=frequency_count - 1).astype(real_dtype)
    tstart, delta_t = real_dtype(-0.0173), real_dtype(1 / 2048)

    times = tstart + delta_t * np.arange(sample_count, dtype=real_dtype)
    shift = np.exp(-2.0j * 3.141592653 * times[:, None] * freqs)

    def products(waveform):
        ratio = waveform / h00
        shifted = shift * ratio
        shifted_lo = shifted[:, :-1]
        filt = np.conjugate(np.sum(
            a0 * shifted_lo
            + a1 * (shifted[:, 1:] - shifted_lo), axis=-1))
        power = np.abs(ratio) ** 2
        norm = np.sum(
            b0 * power[:-1] + b1 * (power[1:] - power[:-1]))
        return filt, norm

    expected_sh, expected_hh = products(hp)
    if not dominant_mode:
        expected_csh, expected_chh = products(hc)
        expected = np.sqrt(
            np.abs(expected_sh) ** 2 / (2.0 * expected_hh)
            + np.abs(expected_csh) ** 2 / (2.0 * expected_chh))

    with ctx:
        hp_array = Array(hp)
        hc_array = Array(hc)
        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError("SNR prediction copied its waveform")

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                _INFERENCE_RELBIN_TORCH,
                "_SNR_PREDICTOR_TARGET_ELEMENTS",
                2 * frequency_count,
            )
            prepared = _INFERENCE_RELBIN_TORCH.prepare_likelihood_data(
                hp_array, freqs, h00, a0, a1, b0, b1)
            if dominant_mode:
                actual = _INFERENCE_RELBIN_TORCH.snr_predictor_dom(
                    prepared[0], tstart, delta_t, sample_count,
                    hp_array, *prepared[1:])
            else:
                actual = _INFERENCE_RELBIN_TORCH.snr_predictor(
                    prepared[0], tstart, delta_t, sample_count,
                    hp_array, hc_array, *prepared[1:])

    tolerance = 5e-4 if device == "mps" else 3e-12
    if dominant_mode:
        actual_sh, actual_hh = actual
        assert actual_sh.device.type == device
        assert actual_hh.device.type == device
        np.testing.assert_allclose(
            actual_sh.resolve_conj().cpu().numpy(), expected_sh,
            rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            actual_hh.cpu().numpy(), expected_hh,
            rtol=tolerance, atol=tolerance)
    else:
        assert actual.device.type == device
        np.testing.assert_allclose(
            actual.cpu().numpy(), expected,
            rtol=tolerance, atol=tolerance)


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
    injection_atol = 5e-8 if device == "cuda" else (
        2e-8 if device == "mps" else 7e-9)
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


def test_sine_gaussian_chisq_stays_on_device(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    if device == "mps":
        pytest.skip("Torch MPS PyCBC arrays do not support complex dtypes")

    rng = np.random.default_rng(681)
    series_length = 513
    strain_values = (
        rng.normal(size=series_length)
        + 1j * rng.normal(size=series_length)
    ).astype(np.complex64)
    template_values = np.zeros(series_length, dtype=np.complex64)
    psd_values = (
        1.0 + rng.random(series_length)
    ).astype(np.float32)
    snr_values = np.array([12 + 2j, 3 + 0j, 10 - 1j], dtype=np.complex64)
    bchisq_values = np.array([2.0, 50.0, 5.0], dtype=np.float32)
    dof_values = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    index_values = np.array([17, 29, 53], dtype=np.int32)
    template_hash = 681
    epoch = 0.125

    def make_calculator():
        calculator = object.__new__(sgchisq.SingleDetSGChisq)
        calculator.do = True
        calculator.snr_threshold = 5.0
        calculator.params = {template_hash: "8-20,12-40"}
        calculator.cached_chisq_bins = lambda _template, _psd: np.array(
            [20, 60, 100, 140], dtype=np.int32
        )
        return calculator

    def make_series(series_epoch=epoch):
        strain = FrequencySeries(
            strain_values, delta_f=1, epoch=series_epoch
        )
        template = FrequencySeries(
            template_values, delta_f=1, epoch=series_epoch
        )
        template.params = types.SimpleNamespace(template_hash=template_hash)
        template.f_lower = 20.0
        psd = FrequencySeries(psd_values, delta_f=1, epoch=series_epoch)
        return strain, template, psd

    strain, template, psd = make_series()
    expected = make_calculator().values(
        strain,
        template,
        psd,
        snr_values,
        0.75,
        bchisq_values,
        dof_values,
        index_values,
    )

    with ctx:
        strain, template, psd = make_series()
        snr = Array(snr_values)
        bchisq = Array(bchisq_values)
        dof = Array(dof_values)
        indices = Array(index_values)

        def _reject_host_transfer(_self):
            raise AssertionError("sine-Gaussian chi-squared copied data to host")

        def _reject_scalar_reduction(*_args, **_kwargs):
            raise AssertionError(
                "sine-Gaussian chi-squared reduced a tile through Python"
            )

        with monkeypatch.context() as patch:
            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(
                pycbc.types.array_torch, "sum", _reject_scalar_reduction
            )
            patch.setattr(
                pycbc.types.array_torch,
                "weighted_inner",
                _reject_scalar_reduction,
            )
            actual = make_calculator().values(
                strain,
                template,
                psd,
                snr,
                0.75,
                bchisq,
                dof,
                indices,
            )
            missing_calculator = make_calculator()
            missing_calculator.params = {}
            missing = missing_calculator.values(
                strain,
                template,
                psd,
                snr,
                0.75,
                bchisq,
                dof,
                indices,
            )
            nyquist_calculator = make_calculator()
            nyquist_calculator.params = {template_hash: "8-400"}
            near_nyquist = nyquist_calculator.values(
                strain,
                template,
                psd,
                snr,
                0.75,
                bchisq,
                dof,
                indices,
            )

        gps_expected = gps_actual = None
        if device == "cpu":
            gps_strain, gps_template, gps_psd = make_series(
                1126259462.125
            )
            # Force the former scalar Torch implementation so that the
            # batched phase construction is checked at a realistic GPS epoch.
            with monkeypatch.context() as patch:
                patch.setattr(sgchisq, "_torch_tensor", lambda _value: None)
                gps_expected = make_calculator().values(
                    gps_strain,
                    gps_template,
                    gps_psd,
                    snr_values,
                    0.75,
                    bchisq_values,
                    dof_values,
                    index_values,
                )
            gps_actual = make_calculator().values(
                gps_strain,
                gps_template,
                gps_psd,
                snr,
                0.75,
                bchisq,
                dof,
                indices,
            )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(), expected,
        rtol=3e-5, atol=3e-5
    )
    for defaults in (missing, near_nyquist):
        assert isinstance(defaults, Array)
        assert defaults._data.tensor.device.type == device
        torch.testing.assert_close(
            defaults._data.tensor, torch.ones_like(defaults._data.tensor)
        )
    if gps_actual is not None:
        np.testing.assert_allclose(
            gps_actual._data.tensor.detach().cpu().numpy(),
            gps_expected,
            rtol=3e-5,
            atol=3e-5,
        )


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


def test_frequency_noise_from_psd_uses_local_device_rng(
        torch_device_ctx, monkeypatch):
    ctx, device = torch_device_ctx
    psd_values = np.linspace(0.5, 2.0, 513, dtype=np.float32)

    with ctx:
        psd = FrequencySeries(psd_values, delta_f=1.0)
        global_rng_state = torch.random.get_rng_state()

        with monkeypatch.context() as patch:
            def _reject_host_transfer(_self):
                raise AssertionError(
                    "frequency-domain noise copied Torch data to host"
                )

            def _reject_numpy_rng(*_args, **_kwargs):
                raise AssertionError(
                    "frequency-domain noise used NumPy's RNG"
                )

            patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
            patch.setattr(gaussian.numpy.random, "normal", _reject_numpy_rng)
            first = gaussian.frequency_noise_from_psd(psd, seed=9182)
            repeated = gaussian.frequency_noise_from_psd(psd, seed=9182)
            different = gaussian.frequency_noise_from_psd(psd, seed=9183)

        final_rng_state = torch.random.get_rng_state()

    assert first._data.tensor.device.type == device
    assert first.dtype == np.dtype(np.complex64)
    assert torch.equal(first._data.tensor, repeated._data.tensor)
    assert not torch.equal(first._data.tensor, different._data.tensor)
    assert torch.equal(global_rng_state, final_rng_state)


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
            graph = batch._cuda_graphs[0]["graph"]
            assert graph.replay_count == 0

            # Second run replays graph
            res2 = batch.process_data(types.SimpleNamespace(
                overwhitened_data=lambda _delta_f: stilde,
                trim_padding=0,
                blocksize=size,
                sample_rate=1,
                start_time=100.0,
            ))
            assert graph.replay_count == 1
            np.testing.assert_allclose(res1["snr"], res2["snr"])

