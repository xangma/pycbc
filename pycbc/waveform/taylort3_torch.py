# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Torch-native nonspinning TaylorT3 evolution, polarizations, and modes.

TaylorT3 gives frequency and orbital phase as analytic functions of the PN
time parameter ``theta``.  The initial time parameter is found with the same
bracketing and bisection construction as LALSuite, after which the uniformly
sampled trajectory is evaluated in parallel on the active Torch device.

MPS does not support float64.  On that backend, only the scalar starting root
and the discrete termination/reference indices are determined with Torch CPU
float64 tensors.  The retained trajectory and waveform are still evaluated in
bulk on MPS.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

import pycbc.scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.constants import _EULER_GAMMA, _MTSUN_SI
from pycbc.waveform.imrphenomd_torch import (
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _is_default_order,
    _is_nonzero,
)
from pycbc.waveform.pn_modes_torch import pn_modes_lal_convention
from pycbc.waveform.pn_polarization_torch import pn_polarizations
from pycbc.waveform.taylort2_torch import (
    taylor_t2_coefficients,
    taylor_t2_timing,
)


_ISCO_VELOCITY = 1.0 / math.sqrt(6.0)
_SUPPORTED_PHASE_ORDERS = frozenset((-1, 0, 2, 3, 4, 5, 6, 7))
_SUPPORTED_TIDAL_ORDERS = frozenset((-1, 0, 10, 12))
_SUPPORTED_AMPLITUDE_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6))
_SPIN_KEYS = (
    "spin1x",
    "spin1y",
    "spin1z",
    "spin2x",
    "spin2y",
    "spin2z",
)


@dataclass(frozen=True)
class TaylorT3Coefficients:
    """Coefficients of TaylorT3's analytic frequency and phasing series."""

    total_mass_seconds: torch.Tensor
    symmetric_mass_ratio: torch.Tensor
    frequency_leading: torch.Tensor
    frequency2: torch.Tensor
    frequency3: torch.Tensor
    frequency4: torch.Tensor
    frequency5: torch.Tensor
    frequency6: torch.Tensor
    frequency6_log: torch.Tensor
    frequency7: torch.Tensor
    frequency10: torch.Tensor
    frequency12: torch.Tensor
    phase_leading: torch.Tensor
    phase2: torch.Tensor
    phase3: torch.Tensor
    phase4: torch.Tensor
    phase5: torch.Tensor
    phase6: torch.Tensor
    phase6_log: torch.Tensor
    phase7: torch.Tensor
    phase10: torch.Tensor
    phase12: torch.Tensor
    phase_order: int


@dataclass(frozen=True)
class TaylorT3Orbit:
    """Uniformly sampled TaylorT3 orbital velocity and phase."""

    velocity: torch.Tensor
    phase: torch.Tensor
    delta_t: float
    epoch: float

    def __len__(self):
        return self.velocity.shape[0]


@dataclass(frozen=True)
class _TaylorT3Evolution:
    """TaylorT3 trajectory plus the metadata needed to phase it."""

    coalescence_time: torch.Tensor
    time_scale: torch.Tensor
    theta: torch.Tensor
    velocity: torch.Tensor
    phase_index: int


def _coerce_order(value, name, supported):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value not in supported:
        choices = ", ".join(str(item) for item in sorted(supported))
        raise ValueError(f"unsupported {name} {value}; expected one of {choices}")
    return value


def _phase_tidal_5pn(mass_fraction, deformability):
    return (
        deformability
        * (-3.3 * mass_fraction / 51.2 + 9.0 / 128.0)
        * mass_fraction**4
    )


def _phase_tidal_6pn(eta, mass_fraction, deformability):
    return deformability * mass_fraction**4 * (
        -1.30715 * mass_fraction / 13.76256
        - 8.745 * eta * mass_fraction / 114.688
        + 2.3325 / 22.9376
        + 4.905 * eta / 57.344
        + 3.985 * mass_fraction**2 / 114.688
        - 9.65 * mass_fraction**3 / 286.72
    )


def _frequency_tidal_5pn(mass_fraction, deformability):
    return (
        deformability
        * (-9.9 * mass_fraction / 102.4 + 2.7 / 25.6)
        * mass_fraction**4
    )


def _frequency_tidal_6pn(eta, mass_fraction, deformability):
    return deformability * mass_fraction**4 * (
        -8.579 * mass_fraction / 65.536
        - 1.947 * eta * mass_fraction / 16.384
        + 1.8453 / 13.1072
        + 4.329 * eta / 32.768
        + 2.391 * mass_fraction**2 / 65.536
        - 5.79 * mass_fraction**3 / 163.84
    )


def taylor_t3_coefficients(
    mass1,
    mass2,
    *,
    lambda1=0.0,
    lambda2=0.0,
    phase_order=-1,
    tidal_order=-1,
    device=None,
    dtype=torch.float64,
):
    """Build TaylorT3 coefficients for masses expressed in solar masses."""

    mass1 = float(mass1)
    mass2 = float(mass2)
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    if not math.isfinite(mass1) or mass1 <= 0.0:
        raise ValueError("mass1 must be finite and positive")
    if not math.isfinite(mass2) or mass2 <= 0.0:
        raise ValueError("mass2 must be finite and positive")
    if not math.isfinite(lambda1) or not math.isfinite(lambda2):
        raise ValueError("tidal deformabilities must be finite")

    phase_order = _coerce_order(phase_order, "phase_order", _SUPPORTED_PHASE_ORDERS)
    tidal_order = _coerce_order(tidal_order, "tidal_order", _SUPPORTED_TIDAL_ORDERS)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("TaylorT3 evolution requires float32 or float64")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    total_mass_seconds = total_mass * _MTSUN_SI

    phase10 = phase12 = frequency10 = frequency12 = 0.0
    if tidal_order in (-1, 12):
        phase12 = _phase_tidal_6pn(
            eta, mass1_fraction, lambda1
        ) + _phase_tidal_6pn(eta, mass2_fraction, lambda2)
        frequency12 = _frequency_tidal_6pn(
            eta, mass1_fraction, lambda1
        ) + _frequency_tidal_6pn(eta, mass2_fraction, lambda2)
    if tidal_order in (-1, 10, 12):
        phase10 = _phase_tidal_5pn(
            mass1_fraction, lambda1
        ) + _phase_tidal_5pn(mass2_fraction, lambda2)
        frequency10 = _frequency_tidal_5pn(
            mass1_fraction, lambda1
        ) + _frequency_tidal_5pn(mass2_fraction, lambda2)

    values = torch.tensor(
        (
            total_mass_seconds,
            eta,
            1.0 / (8.0 * math.pi * total_mass_seconds),
            7.43 / 26.88 + 1.1 / 3.2 * eta,
            -3.0 / 10.0 * math.pi,
            1.855099 / 14.450688
            + 5.6975 / 25.8048 * eta
            + 3.71 / 20.48 * eta**2,
            (-7.729 / 21.504 + 1.3 / 25.6 * eta) * math.pi,
            -7.20817631400877 / 2.88412611379200
            + 5.3 / 20.0 * math.pi**2
            + 1.07 / 2.80 * _EULER_GAMMA
            + (25.302017977 / 4.161798144 - 4.51 / 20.48 * math.pi**2) * eta
            - 3.0913 / 183.5008 * eta**2
            + 2.35925 / 17.69472 * eta**3,
            1.07 / 2.80,
            (
                -1.88516689 / 4.33520640
                - 9.7765 / 25.8048 * eta
                + 1.41769 / 12.90240 * eta**2
            )
            * math.pi,
            frequency10,
            frequency12,
            -1.0 / eta,
            3.715 / 8.064 + 5.5 / 9.6 * eta,
            -3.0 / 4.0 * math.pi,
            9.275495 / 14.450688
            + 2.84875 / 2.58048 * eta
            + 1.855 / 2.048 * eta**2,
            (3.8645 / 2.1504 - 6.5 / 25.6 * eta) * math.pi,
            83.1032450749357 / 5.7682522275840
            - 5.3 / 4.0 * math.pi**2
            - 10.7 / 5.6 * _EULER_GAMMA
            + (-126.510089885 / 4.161798144 + 2.255 / 2.048 * math.pi**2)
            * eta
            + 1.54565 / 18.35008 * eta**2
            - 1.179625 / 1.769472 * eta**3,
            -10.7 / 5.6,
            (
                1.88516689 / 1.73408256
                + 4.88825 / 5.16096 * eta
                - 1.41769 / 5.16096 * eta**2
            )
            * math.pi,
            phase10,
            phase12,
        ),
        device=device,
        dtype=dtype,
    )
    return TaylorT3Coefficients(*values, phase_order=phase_order)


def _series_powers(theta):
    powers = {1: theta}
    for exponent in range(2, 13):
        powers[exponent] = powers[exponent - 1] * theta
    return powers


def taylor_t3_frequency(theta, coefficients):
    """Return TaylorT3's gravitational-wave frequency at ``theta``."""

    theta = torch.as_tensor(
        theta,
        device=coefficients.frequency_leading.device,
        dtype=coefficients.frequency_leading.dtype,
    )
    powers = _series_powers(theta)
    series = torch.ones_like(theta)
    order = coefficients.phase_order
    if order in (-1, 2, 3, 4, 5, 6, 7):
        series = series + coefficients.frequency2 * powers[2]
    if order in (-1, 3, 4, 5, 6, 7):
        series = series + coefficients.frequency3 * powers[3]
    if order in (-1, 4, 5, 6, 7):
        series = series + coefficients.frequency4 * powers[4]
    if order in (-1, 5, 6, 7):
        series = series + coefficients.frequency5 * powers[5]
    if order in (-1, 6, 7):
        series = series + (
            coefficients.frequency6
            + coefficients.frequency6_log * torch.log(2.0 * theta)
        ) * powers[6]
    if order in (-1, 7):
        series = series + coefficients.frequency7 * powers[7]
    if order != 0:
        series = (
            series
            + coefficients.frequency10 * powers[10]
            + coefficients.frequency12 * powers[12]
        )
    return coefficients.frequency_leading * powers[3] * series


def taylor_t3_phase(theta, coefficients):
    """Return TaylorT3's orbital phase at ``theta``, up to a constant.

    LALSuite obtains the logarithm's reference value from a numerical TaylorT1
    duration.  That value contributes only an additive phase constant, which
    is removed when :func:`taylor_t3_orbit` applies ``coa_phase`` at ``f_ref``.
    Using a unit reference here therefore leaves every generated waveform
    unchanged while avoiding a CPU-only quadrature.
    """

    theta = torch.as_tensor(
        theta,
        device=coefficients.phase_leading.device,
        dtype=coefficients.phase_leading.dtype,
    )
    powers = _series_powers(theta)
    series = torch.ones_like(theta)
    order = coefficients.phase_order
    if order in (-1, 2, 3, 4, 5, 6, 7):
        series = series + coefficients.phase2 * powers[2]
    if order in (-1, 3, 4, 5, 6, 7):
        series = series + coefficients.phase3 * powers[3]
    if order in (-1, 4, 5, 6, 7):
        series = series + coefficients.phase4 * powers[4]
    if order in (-1, 5, 6, 7):
        series = series + coefficients.phase5 * torch.log(theta) * powers[5]
    if order in (-1, 6, 7):
        series = series + (
            coefficients.phase6
            + coefficients.phase6_log * torch.log(2.0 * theta)
        ) * powers[6]
    if order in (-1, 7):
        series = series + coefficients.phase7 * powers[7]
    if order != 0:
        series = (
            series
            + coefficients.phase10 * powers[10]
            + coefficients.phase12 * powers[12]
        )
    return coefficients.phase_leading * series / powers[5]


def _initial_time_parameter(
    mass1,
    mass2,
    f_lower,
    lambda1,
    lambda2,
    phase_order,
    tidal_order,
    coefficients,
):
    """Find the TaylorT3 coalescence time parameter as LALSuite does."""

    initial_velocity = torch.pow(
        math.pi * coefficients.total_mass_seconds * f_lower,
        1.0 / 3.0,
    )
    timing_coefficients = taylor_t2_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
        device=initial_velocity.device,
        dtype=initial_velocity.dtype,
    )
    chirp_length = -taylor_t2_timing(initial_velocity, timing_coefficients)
    scale = coefficients.symmetric_mass_ratio / (
        5.0 * coefficients.total_mass_seconds
    )
    step = scale * chirp_length / 1000.0
    upper = scale * chirp_length * 3.0 + 5.0
    step_value = float(step.item())
    upper_value = float(upper.item())
    if not math.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("TaylorT3 chirp length must be finite and positive")

    grid_count = max(1, math.ceil(upper_value / step_value))
    scan = torch.arange(
        1,
        grid_count + 1,
        device=step.device,
        dtype=step.dtype,
    ) * step
    scan = scan[scan < upper]
    residual = taylor_t3_frequency(scan.pow(-0.125), coefficients) - f_lower
    maximum, maximum_index = torch.max(residual, dim=0)
    minimum = torch.min(residual)
    if float(maximum.item()) <= 0.0 or float(minimum.item()) >= 0.0:
        raise ValueError("TaylorT3 could not bracket the starting frequency")

    lower = scan[maximum_index]
    lower_residual = maximum
    upper_residual = taylor_t3_frequency(upper.pow(-0.125), coefficients) - f_lower
    for _ in range(80):
        if float((upper - lower).item()) <= 1.0e-6:
            break
        midpoint = 0.5 * (lower + upper)
        if bool((midpoint == lower).item()) or bool((midpoint == upper).item()):
            break
        midpoint_residual = (
            taylor_t3_frequency(midpoint.pow(-0.125), coefficients) - f_lower
        )
        if float(midpoint_residual.item()) == 0.0:
            lower = upper = midpoint
            break
        if float((lower_residual * midpoint_residual).item()) < 0.0:
            upper = midpoint
            upper_residual = midpoint_residual
        elif float((upper_residual * midpoint_residual).item()) < 0.0:
            lower = midpoint
            lower_residual = midpoint_residual
        else:
            raise ValueError("TaylorT3 starting-frequency bisection failed")
    return 0.5 * (lower + upper), scale


def _build_evolution(
    mass1,
    mass2,
    delta_t,
    f_lower,
    f_ref,
    lambda1,
    lambda2,
    tidal_order,
    coefficients,
):
    """Build a trajectory and determine its discrete control metadata."""

    total_mass_seconds = (float(mass1) + float(mass2)) * _MTSUN_SI
    initial_parameter, time_scale = _initial_time_parameter(
        mass1,
        mass2,
        f_lower,
        lambda1,
        lambda2,
        coefficients.phase_order,
        tidal_order,
        coefficients,
    )
    coalescence_time = initial_parameter / time_scale
    sample_count = max(2, math.ceil(float(coalescence_time.item()) / delta_t))
    times = torch.arange(
        sample_count,
        device=initial_parameter.device,
        dtype=initial_parameter.dtype,
    ) * delta_t
    theta = (time_scale * (coalescence_time - times)).pow(-0.125)
    frequency = taylor_t3_frequency(theta, coefficients)
    scaled_frequency = math.pi * coefficients.total_mass_seconds * frequency
    velocity = torch.sign(scaled_frequency) * torch.abs(scaled_frequency).pow(
        1.0 / 3.0
    )

    termination = velocity >= _ISCO_VELOCITY
    termination[0] = False
    termination[1:] |= frequency[1:] <= frequency[:-1]
    indices = torch.nonzero(termination, as_tuple=False)
    if indices.numel() == 0:
        raise ValueError("TaylorT3 evolution did not terminate before coalescence")
    retained = int(indices[0, 0].item())
    theta = theta[:retained]
    velocity = velocity[:retained]

    if f_ref == 0.0:
        phase_index = retained - 1
    elif f_ref == f_lower:
        phase_index = 0
    else:
        reference_velocity = torch.as_tensor(
            math.cbrt(math.pi * total_mass_seconds * f_ref),
            device=velocity.device,
            dtype=velocity.dtype,
        )
        insertion = torch.searchsorted(
            velocity.contiguous(), reference_velocity, right=True
        )
        insertion_index = int(insertion.item())
        phase_index = insertion_index - 1
        if phase_index < 0 or insertion_index == retained:
            raise ValueError("f_ref must lie within the generated evolution")

    return _TaylorT3Evolution(
        coalescence_time=coalescence_time,
        time_scale=time_scale,
        theta=theta,
        velocity=velocity,
        phase_index=phase_index,
    )


def taylor_t3_orbit(
    mass1,
    mass2,
    delta_t,
    f_lower,
    *,
    coa_phase=0.0,
    f_ref=0.0,
    lambda1=0.0,
    lambda2=0.0,
    phase_order=-1,
    tidal_order=-1,
    device=None,
    dtype=torch.float64,
):
    """Evaluate a nonspinning TaylorT3 orbit on a Torch device."""

    delta_t = float(delta_t)
    f_lower = float(f_lower)
    f_ref = float(f_ref)
    coa_phase = float(coa_phase)
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    if not math.isfinite(f_lower) or f_lower <= 0.0:
        raise ValueError("f_lower must be finite and positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("f_ref must be finite and nonnegative")
    if not math.isfinite(coa_phase):
        raise ValueError("coa_phase must be finite")

    coefficients = taylor_t3_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
        device=device,
        dtype=dtype,
    )
    target = coefficients.total_mass_seconds
    total_mass_seconds = (float(mass1) + float(mass2)) * _MTSUN_SI
    isco_frequency = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    if f_ref != 0.0 and f_ref < f_lower:
        raise ValueError("f_ref must be zero or at least f_lower")
    if f_ref >= isco_frequency:
        raise ValueError("f_ref must be below Schwarzschild ISCO")

    if target.device.type == "mps":
        metadata_coefficients = taylor_t3_coefficients(
            mass1,
            mass2,
            lambda1=lambda1,
            lambda2=lambda2,
            phase_order=coefficients.phase_order,
            tidal_order=tidal_order,
            device="cpu",
            dtype=torch.float64,
        )
        metadata = _build_evolution(
            mass1,
            mass2,
            delta_t,
            f_lower,
            f_ref,
            lambda1,
            lambda2,
            tidal_order,
            metadata_coefficients,
        )
        retained = len(metadata.velocity)
        coalescence_time = metadata.coalescence_time.to(
            device=target.device,
            dtype=target.dtype,
        )
        time_scale = metadata.time_scale.to(
            device=target.device,
            dtype=target.dtype,
        )
        times = torch.arange(
            retained,
            device=target.device,
            dtype=target.dtype,
        ) * delta_t
        theta = (time_scale * (coalescence_time - times)).pow(-0.125)
        frequency = taylor_t3_frequency(theta, coefficients)
        scaled_frequency = math.pi * coefficients.total_mass_seconds * frequency
        velocity = torch.sign(scaled_frequency) * torch.abs(scaled_frequency).pow(
            1.0 / 3.0
        )
    else:
        metadata = _build_evolution(
            mass1,
            mass2,
            delta_t,
            f_lower,
            f_ref,
            lambda1,
            lambda2,
            tidal_order,
            coefficients,
        )
        retained = len(metadata.velocity)
        theta = metadata.theta
        velocity = metadata.velocity

    phase = taylor_t3_phase(theta, coefficients)
    phase = phase + (coa_phase - phase[metadata.phase_index])

    return TaylorT3Orbit(
        velocity=velocity,
        phase=phase,
        delta_t=delta_t,
        epoch=-retained * delta_t,
    )


def _supported_order(value, supported):
    try:
        return operator.index(value) in supported
    except TypeError:
        return False


def taylort3_native_supported(parameters):
    """Return whether the native TaylorT3 TD port covers ``parameters``."""

    if parameters.get("approximant", "TaylorT3") != "TaylorT3":
        return False
    if any(_is_nonzero(parameters.get(key, 0.0)) for key in _SPIN_KEYS):
        return False
    if not _is_default_order(parameters.get("spin_order", -1)):
        return False
    if not _is_default_order(parameters.get("eccentricity_order", -1)):
        return False
    if not _supported_order(parameters.get("phase_order", -1), _SUPPORTED_PHASE_ORDERS):
        return False
    if not _supported_order(
        parameters.get("amplitude_order", -1), _SUPPORTED_AMPLITUDE_ORDERS
    ):
        return False
    if not _supported_order(parameters.get("tidal_order", -1), _SUPPORTED_TIDAL_ORDERS):
        return False
    if any(
        _is_nonzero(parameters.get(key, 0.0))
        for key in (
            _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + ("frame_axis", "modes_choice", "side_bands")
        )
    ):
        return False
    if parameters.get("mode_array") is not None or parameters.get("numrel_data", ""):
        return False
    return True


def taylort3_modes_native_supported(parameters):
    """Return whether the native TaylorT3 mode port covers ``parameters``."""

    if not taylort3_native_supported(parameters):
        return False
    try:
        ell_max = operator.index(parameters.get("ell_max", 5))
    except TypeError:
        return False
    return 2 <= ell_max <= 6


def _finite_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"TaylorT3 requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"TaylorT3 requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"TaylorT3 requires finite {name}")
    return value


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native TaylorT3 requires an active TorchScheme")
    device = state.torch_device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return device, dtype


def _waveform_parameters(parameters):
    lambda1 = (
        0.0
        if parameters.get("lambda1") is None
        else _finite_float(parameters, "lambda1")
    )
    lambda2 = (
        0.0
        if parameters.get("lambda2") is None
        else _finite_float(parameters, "lambda2")
    )
    return {
        "mass1": _finite_float(parameters, "mass1"),
        "mass2": _finite_float(parameters, "mass2"),
        "distance": _finite_float(parameters, "distance", 1.0),
        "delta_t": _finite_float(parameters, "delta_t"),
        "f_lower": _finite_float(parameters, "f_lower"),
        "f_ref": _finite_float(parameters, "f_ref", 0.0),
        "lambda1": lambda1,
        "lambda2": lambda2,
        "phase_order": parameters.get("phase_order", -1),
        "amplitude_order": parameters.get("amplitude_order", -1),
        "tidal_order": parameters.get("tidal_order", -1),
    }


def taylort3_td_torch(**parameters):
    """Generate nonspinning TaylorT3 polarizations on the Torch device."""

    values = _waveform_parameters(parameters)
    inclination = _finite_float(parameters, "inclination", 0.0)
    coa_phase = _finite_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _finite_float(parameters, "long_asc_nodes", 0.0)
    device, dtype = _target_device_dtype()
    orbit = taylor_t3_orbit(
        values["mass1"],
        values["mass2"],
        values["delta_t"],
        values["f_lower"],
        coa_phase=coa_phase,
        f_ref=values["f_ref"],
        lambda1=values["lambda1"],
        lambda2=values["lambda2"],
        phase_order=values["phase_order"],
        tidal_order=values["tidal_order"],
        device=device,
        dtype=dtype,
    )
    plus, cross = pn_polarizations(
        orbit.velocity,
        orbit.phase,
        values["mass1"],
        values["mass2"],
        values["distance"],
        inclination,
        amplitude_order=values["amplitude_order"],
    )
    if long_asc_nodes:
        rotation = 2.0 * long_asc_nodes
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        plus, cross = (
            cosine * plus + sine * cross,
            cosine * cross - sine * plus,
        )

    return (
        TimeSeries(
            TorchArrayData(plus),
            delta_t=orbit.delta_t,
            epoch=orbit.epoch,
            copy=False,
        ),
        TimeSeries(
            TorchArrayData(cross),
            delta_t=orbit.delta_t,
            epoch=orbit.epoch,
            copy=False,
        ),
    )


def taylort3_modes_torch(**parameters):
    """Generate nonspinning TaylorT3 modes on the Torch device.

    LAL's legacy mode interface ignores reference phase and tidal effects for
    TaylorT3. Preserve those conventions even though polarization generation
    supports both.
    """

    values = _waveform_parameters(parameters)
    ell_max = parameters.get("ell_max", 5)
    device, dtype = _target_device_dtype()
    orbit = taylor_t3_orbit(
        values["mass1"],
        values["mass2"],
        values["delta_t"],
        values["f_lower"],
        coa_phase=0.0,
        f_ref=values["f_ref"],
        phase_order=values["phase_order"],
        tidal_order=0,
        device=device,
        dtype=dtype,
    )
    modes = pn_modes_lal_convention(
        orbit.velocity,
        orbit.phase,
        values["mass1"],
        values["mass2"],
        values["distance"],
        ell_max=ell_max,
        amplitude_order=values["amplitude_order"],
    )
    return {
        mode: (
            TimeSeries(
                TorchArrayData(real),
                delta_t=orbit.delta_t,
                epoch=orbit.epoch,
                copy=False,
            ),
            TimeSeries(
                TorchArrayData(imaginary),
                delta_t=orbit.delta_t,
                epoch=orbit.epoch,
                copy=False,
            ),
        )
        for mode, (real, imaginary) in modes.items()
    }


__all__ = [
    "TaylorT3Coefficients",
    "TaylorT3Orbit",
    "taylor_t3_coefficients",
    "taylor_t3_frequency",
    "taylor_t3_orbit",
    "taylor_t3_phase",
    "taylort3_modes_native_supported",
    "taylort3_modes_torch",
    "taylort3_native_supported",
    "taylort3_td_torch",
]
