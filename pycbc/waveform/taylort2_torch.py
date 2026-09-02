# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native nonspinning TaylorT2 evolution, polarizations, and modes.

TaylorT2 expresses time and orbital phase as analytic functions of the PN
velocity.  This implementation inverts the time relation for every requested
sample in parallel, keeping the long waveform arrays on the active Torch
device.  It also reproduces LALSuite's early termination when tidal terms make
the timing relation non-monotonic before Schwarzschild ISCO.
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
class TaylorT2Coefficients:
    """Coefficients of TaylorT2's analytic timing and phasing series."""

    total_mass_seconds: torch.Tensor
    timing_leading: torch.Tensor
    timing2: torch.Tensor
    timing3: torch.Tensor
    timing4: torch.Tensor
    timing5: torch.Tensor
    timing6: torch.Tensor
    timing6_log: torch.Tensor
    timing7: torch.Tensor
    timing10: torch.Tensor
    timing12: torch.Tensor
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
class TaylorT2Orbit:
    """Uniformly sampled TaylorT2 orbital velocity and phase."""

    velocity: torch.Tensor
    phase: torch.Tensor
    delta_t: float
    epoch: float

    def __len__(self):
        return self.velocity.shape[0]


def _coerce_order(value, name, supported):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value not in supported:
        choices = ", ".join(str(item) for item in sorted(supported))
        raise ValueError(f"unsupported {name} {value}; expected one of {choices}")
    return value


def _timing_tidal_5pn(mass_fraction, deformability):
    return deformability * (288.0 - 264.0 * mass_fraction) * mass_fraction**4


def _timing_tidal_6pn(eta, mass_fraction, deformability):
    return deformability * mass_fraction**4 * (
        -2995.0 / 4.0 * mass_fraction
        - 451.0 * eta * mass_fraction
        + 3179.0 / 4.0
        + 519.0 * eta
        + 797.0 / 2.0 * mass_fraction**2
        - 386.0 * mass_fraction**3
    )


def _phase_tidal_5pn(mass_fraction, deformability):
    return deformability * (72.0 - 66.0 * mass_fraction) * mass_fraction**4


def _phase_tidal_6pn(eta, mass_fraction, deformability):
    return deformability * mass_fraction**4 * (
        -1497.5 / 5.6 * mass_fraction
        - 225.5 / 1.4 * eta * mass_fraction
        + 1589.5 / 5.6
        + 259.5 / 1.4 * eta
        + 398.5 / 2.8 * mass_fraction**2
        - 965.0 / 7.0 * mass_fraction**3
    )


def taylor_t2_coefficients(
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
    """Build TaylorT2 coefficients for masses expressed in solar masses."""

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
        raise ValueError("TaylorT2 evolution requires float32 or float64")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass

    timing10 = timing12 = phase10 = phase12 = 0.0
    if tidal_order in (-1, 12):
        timing12 = _timing_tidal_6pn(
            eta, mass1_fraction, lambda1
        ) + _timing_tidal_6pn(eta, mass2_fraction, lambda2)
        phase12 = _phase_tidal_6pn(
            eta, mass1_fraction, lambda1
        ) + _phase_tidal_6pn(eta, mass2_fraction, lambda2)
    if tidal_order in (-1, 10, 12):
        timing10 = _timing_tidal_5pn(
            mass1_fraction, lambda1
        ) + _timing_tidal_5pn(mass2_fraction, lambda2)
        phase10 = _phase_tidal_5pn(
            mass1_fraction, lambda1
        ) + _phase_tidal_5pn(mass2_fraction, lambda2)

    values = torch.tensor(
        (
            total_mass * _MTSUN_SI,
            -5.0 * total_mass * _MTSUN_SI / (256.0 * eta),
            7.43 / 2.52 + 11.0 / 3.0 * eta,
            -32.0 / 5.0 * math.pi,
            30.58673 / 5.08032
            + 54.29 / 5.04 * eta
            + 61.7 / 7.2 * eta**2,
            -(77.29 / 2.52 - 13.0 / 3.0 * eta) * math.pi,
            -1005.2469856691 / 2.3471078400
            + 128.0 / 3.0 * math.pi**2
            + 68.48 / 1.05 * _EULER_GAMMA
            + (3147.553127 / 3.048192 - 45.1 / 1.2 * math.pi**2) * eta
            - 15.211 / 1.728 * eta**2
            + 25.565 / 1.296 * eta**3,
            34.24 / 1.05,
            (
                -154.19335 / 1.27008
                - 757.03 / 7.56 * eta
                + 148.09 / 3.78 * eta**2
            )
            * math.pi,
            timing10,
            timing12,
            -1.0 / (32.0 * eta),
            3.715 / 1.008 + 5.5 / 1.2 * eta,
            -10.0 * math.pi,
            15.293365 / 1.016064
            + 27.145 / 1.008 * eta
            + 30.85 / 1.44 * eta**2,
            (386.45 / 6.72 - 65.0 / 8.0 * eta) * math.pi,
            1234.8611926451 / 1.8776862720
            - 160.0 / 3.0 * math.pi**2
            - 171.2 / 2.1 * _EULER_GAMMA
            + (225.5 / 4.8 * math.pi**2 - 1573.7765635 / 1.2192768) * eta
            + 76.055 / 6.912 * eta**2
            - 127.825 / 5.184 * eta**3,
            -85.6 / 2.1,
            (
                77.096675 / 2.032128
                + 37.8515 / 1.2096 * eta
                - 74.045 / 6.048 * eta**2
            )
            * math.pi,
            phase10,
            phase12,
        ),
        device=device,
        dtype=dtype,
    )
    return TaylorT2Coefficients(*values, phase_order=phase_order)


def _timing_series(velocity, coefficients, *, derivative=False):
    """Return the TaylorT2 timing series and optionally its derivative."""

    velocity2 = velocity * velocity
    powers = {1: velocity, 2: velocity2}
    for exponent in range(3, 13):
        powers[exponent] = powers[exponent - 1] * velocity

    series = torch.ones_like(velocity)
    derivative_series = torch.zeros_like(velocity) if derivative else None
    order = coefficients.phase_order

    def add_power(coefficient, exponent):
        nonlocal series, derivative_series
        series = series + coefficient * powers[exponent]
        if derivative:
            derivative_series = derivative_series + (
                exponent * coefficient * powers[exponent - 1]
            )

    if order in (-1, 2, 3, 4, 5, 6, 7):
        add_power(coefficients.timing2, 2)
    if order in (-1, 3, 4, 5, 6, 7):
        add_power(coefficients.timing3, 3)
    if order in (-1, 4, 5, 6, 7):
        add_power(coefficients.timing4, 4)
    if order in (-1, 5, 6, 7):
        add_power(coefficients.timing5, 5)
    if order in (-1, 6, 7):
        logarithm = torch.log(16.0 * velocity2)
        coefficient6 = coefficients.timing6 + coefficients.timing6_log * logarithm
        series = series + coefficient6 * powers[6]
        if derivative:
            derivative_series = derivative_series + powers[5] * (
                6.0 * coefficient6 + 2.0 * coefficients.timing6_log
            )
    if order in (-1, 7):
        add_power(coefficients.timing7, 7)
    if order != 0:
        add_power(coefficients.timing10, 10)
        add_power(coefficients.timing12, 12)

    if derivative:
        return series, derivative_series
    return series


def taylor_t2_timing(velocity, coefficients):
    """Return TaylorT2's analytic time function at ``velocity``."""

    velocity = torch.as_tensor(
        velocity,
        device=coefficients.timing_leading.device,
        dtype=coefficients.timing_leading.dtype,
    )
    return coefficients.timing_leading * _timing_series(
        velocity, coefficients
    ) / velocity**8


def taylor_t2_timing_derivative(velocity, coefficients):
    """Return the derivative of TaylorT2's analytic time function."""

    return _timing_and_derivative(velocity, coefficients)[1]


def _timing_and_derivative(velocity, coefficients):
    """Evaluate the timing series and derivative while sharing its powers."""

    velocity = torch.as_tensor(
        velocity,
        device=coefficients.timing_leading.device,
        dtype=coefficients.timing_leading.dtype,
    )
    series, derivative = _timing_series(velocity, coefficients, derivative=True)
    timing = coefficients.timing_leading * series / velocity**8
    timing_derivative = coefficients.timing_leading * (
        velocity * derivative - 8.0 * series
    ) / velocity**9
    return timing, timing_derivative


def taylor_t2_phase(velocity, coefficients):
    """Return TaylorT2's analytic orbital phase at ``velocity``."""

    velocity = torch.as_tensor(
        velocity,
        device=coefficients.phase_leading.device,
        dtype=coefficients.phase_leading.dtype,
    )
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity4 = velocity3 * velocity
    velocity5 = velocity4 * velocity
    velocity6 = velocity5 * velocity
    velocity7 = velocity6 * velocity
    velocity10 = velocity5 * velocity5
    velocity12 = velocity10 * velocity2

    series = torch.ones_like(velocity)
    order = coefficients.phase_order
    if order in (-1, 2, 3, 4, 5, 6, 7):
        series = series + coefficients.phase2 * velocity2
    if order in (-1, 3, 4, 5, 6, 7):
        series = series + coefficients.phase3 * velocity3
    if order in (-1, 4, 5, 6, 7):
        series = series + coefficients.phase4 * velocity4
    if order in (-1, 5, 6, 7):
        series = series + (
            coefficients.phase5
            * torch.log(velocity / _ISCO_VELOCITY)
            * velocity5
        )
    if order in (-1, 6, 7):
        series = series + (
            coefficients.phase6
            + coefficients.phase6_log * torch.log(16.0 * velocity2)
        ) * velocity6
    if order in (-1, 7):
        series = series + coefficients.phase7 * velocity7
    if order != 0:
        series = (
            series
            + coefficients.phase10 * velocity10
            + coefficients.phase12 * velocity12
        )
    return coefficients.phase_leading * series / velocity5


def _monotonic_timing_endpoint(initial_velocity, limit_velocity, coefficients):
    """Find the first timing-series stationary point before the limit."""

    grid = torch.linspace(
        initial_velocity,
        limit_velocity,
        4097,
        device=initial_velocity.device,
        dtype=initial_velocity.dtype,
    )
    derivative = taylor_t2_timing_derivative(grid, coefficients)
    crossings = torch.nonzero(derivative <= 0.0, as_tuple=False)
    if crossings.numel() == 0:
        return limit_velocity

    upper_index = int(crossings[0, 0].item())
    if upper_index == 0:
        return initial_velocity
    lower = grid[upper_index - 1]
    upper = grid[upper_index]
    iterations = 16 if initial_velocity.dtype == torch.float32 else 40
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        increasing = taylor_t2_timing_derivative(midpoint, coefficients) > 0.0
        lower = torch.where(increasing, midpoint, lower)
        upper = torch.where(increasing, upper, midpoint)
    return 0.5 * (lower + upper)


def _invert_timing(target_times, initial_velocity, endpoint, initial_time, coefficients):
    """Invert the monotonic TaylorT2 timing branch in parallel.

    Float32 reaches its representable limit after 24 parallel bisections. For
    float64, a coarse timing table gives every sample a narrow bracket before
    safeguarded Newton updates; at a stationary tidal cutoff those updates
    retain bisection's guaranteed progress.
    """

    if target_times.dtype == torch.float32:
        lower = initial_velocity.expand_as(target_times).clone()
        upper = endpoint.expand_as(target_times).clone()
        for _ in range(24):
            midpoint = 0.5 * (lower + upper)
            midpoint_time = taylor_t2_timing(midpoint, coefficients) - initial_time
            below_target = midpoint_time < target_times
            lower = torch.where(below_target, midpoint, lower)
            upper = torch.where(below_target, upper, midpoint)
        return 0.5 * (lower + upper)

    grid_size = 4097
    iterations = 18
    grid = torch.linspace(
        initial_velocity,
        endpoint,
        grid_size,
        device=target_times.device,
        dtype=target_times.dtype,
    )
    grid_times = taylor_t2_timing(grid, coefficients) - initial_time
    insertion = torch.searchsorted(
        grid_times.contiguous(), target_times, right=True
    )
    upper_indices = torch.clamp(insertion, 1, grid_size - 1)
    lower_indices = upper_indices - 1
    lower = grid[lower_indices]
    upper = grid[upper_indices]
    lower_times = grid_times[lower_indices]
    upper_times = grid_times[upper_indices]
    time_width = upper_times - lower_times
    fraction = torch.where(
        time_width > 0.0,
        (target_times - lower_times) / time_width,
        torch.full_like(time_width, 0.5),
    )
    candidate = lower + fraction * (upper - lower)
    candidate = torch.maximum(lower, torch.minimum(upper, candidate))
    candidate = torch.where(target_times == 0.0, initial_velocity, candidate)

    for _ in range(iterations):
        timing, derivative = _timing_and_derivative(candidate, coefficients)
        residual = timing - initial_time - target_times
        lower = torch.where(residual <= 0.0, candidate, lower)
        upper = torch.where(residual >= 0.0, candidate, upper)
        midpoint = 0.5 * (lower + upper)
        newton = candidate - residual / derivative
        valid_newton = (
            torch.isfinite(newton)
            & (derivative > 0.0)
            & (newton > lower)
            & (newton < upper)
        )
        candidate = torch.where(
            lower == upper,
            lower,
            torch.where(valid_newton, newton, midpoint),
        )
    return 0.5 * (lower + upper)


def _lal_failure_length(
    velocity,
    target_times,
    frequency_limit,
    initial_time,
    coefficients,
):
    """Return the length at LAL's first non-monotonic bisection failure."""

    if velocity.shape[0] <= 1:
        return velocity.shape[0]
    frequencies = velocity**3 / (
        math.pi * coefficients.total_mass_seconds
    )
    lower_frequencies = 0.8 * frequencies[:-1]
    upper_frequencies = torch.full_like(lower_frequencies, 1.5 * frequency_limit)
    midpoint_frequencies = 0.5 * (lower_frequencies + upper_frequencies)

    def residual(frequencies_to_check):
        checked_velocity = torch.pow(
            math.pi * coefficients.total_mass_seconds * frequencies_to_check,
            1.0 / 3.0,
        )
        return (
            taylor_t2_timing(checked_velocity, coefficients)
            - initial_time
            - target_times[1:]
        )

    lower_residual = residual(lower_frequencies)
    upper_residual = residual(upper_frequencies)
    midpoint_residual = residual(midpoint_frequencies)
    valid = (
        (midpoint_residual == 0.0)
        | (lower_residual * midpoint_residual < 0.0)
        | (upper_residual * midpoint_residual < 0.0)
    )
    failures = torch.nonzero(~valid, as_tuple=False)
    if failures.numel() == 0:
        return velocity.shape[0]
    return int(failures[0, 0].item()) + 1


def taylor_t2_orbit(
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
    """Evaluate a nonspinning TaylorT2 orbit on a Torch device."""

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

    coefficients = taylor_t2_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
        device=device,
        dtype=dtype,
    )
    total_mass_seconds = (float(mass1) + float(mass2)) * _MTSUN_SI
    isco_frequency = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    frequency_limit = min(isco_frequency, 0.5 / delta_t)
    if frequency_limit <= f_lower:
        raise ValueError("sample rate must place the termination above f_lower")
    if f_ref != 0.0 and f_ref < f_lower:
        raise ValueError("f_ref must be zero or at least f_lower")
    if f_ref >= isco_frequency:
        raise ValueError("f_ref must be below Schwarzschild ISCO")

    initial_velocity = torch.as_tensor(
        math.cbrt(math.pi * total_mass_seconds * f_lower),
        device=coefficients.timing_leading.device,
        dtype=coefficients.timing_leading.dtype,
    )
    limit_velocity = torch.as_tensor(
        math.cbrt(math.pi * total_mass_seconds * frequency_limit),
        device=initial_velocity.device,
        dtype=initial_velocity.dtype,
    )
    initial_time = taylor_t2_timing(initial_velocity, coefficients)
    endpoint = _monotonic_timing_endpoint(
        initial_velocity, limit_velocity, coefficients
    )
    endpoint_time = taylor_t2_timing(endpoint, coefficients) - initial_time
    cutoff_time = torch.minimum(-initial_time, endpoint_time)
    sample_count = max(1, math.ceil(float(cutoff_time.item()) / delta_t))
    target_times = torch.arange(
        sample_count,
        device=initial_velocity.device,
        dtype=initial_velocity.dtype,
    ) * delta_t
    velocity = _invert_timing(
        target_times,
        initial_velocity,
        endpoint,
        initial_time,
        coefficients,
    )
    retained = _lal_failure_length(
        velocity,
        target_times,
        frequency_limit,
        initial_time,
        coefficients,
    )
    velocity = velocity[:retained]
    phase = taylor_t2_phase(velocity, coefficients)

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
    phase = phase + (coa_phase - phase[phase_index])

    return TaylorT2Orbit(
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


def taylort2_native_supported(parameters):
    """Return whether the native TaylorT2 TD port covers ``parameters``."""

    if parameters.get("approximant", "TaylorT2") != "TaylorT2":
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


def taylort2_modes_native_supported(parameters):
    """Return whether the native TaylorT2 mode port covers ``parameters``."""

    if not taylort2_native_supported(parameters):
        return False
    try:
        ell_max = operator.index(parameters.get("ell_max", 5))
    except TypeError:
        return False
    return 2 <= ell_max <= 6


def _finite_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"TaylorT2 requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"TaylorT2 requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"TaylorT2 requires finite {name}")
    return value


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native TaylorT2 requires an active TorchScheme")
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


def taylort2_td_torch(**parameters):
    """Generate nonspinning TaylorT2 polarizations on the Torch device."""

    values = _waveform_parameters(parameters)
    inclination = _finite_float(parameters, "inclination", 0.0)
    coa_phase = _finite_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _finite_float(parameters, "long_asc_nodes", 0.0)
    device, dtype = _target_device_dtype()
    orbit = taylor_t2_orbit(
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


def taylort2_modes_torch(**parameters):
    """Generate nonspinning TaylorT2 modes on the Torch device."""

    values = _waveform_parameters(parameters)
    ell_max = parameters.get("ell_max", 5)
    device, dtype = _target_device_dtype()
    orbit = taylor_t2_orbit(
        values["mass1"],
        values["mass2"],
        values["delta_t"],
        values["f_lower"],
        coa_phase=0.0,
        f_ref=values["f_ref"],
        lambda1=values["lambda1"],
        lambda2=values["lambda2"],
        phase_order=values["phase_order"],
        tidal_order=values["tidal_order"],
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
    "TaylorT2Coefficients",
    "TaylorT2Orbit",
    "taylor_t2_coefficients",
    "taylor_t2_orbit",
    "taylor_t2_phase",
    "taylor_t2_timing",
    "taylor_t2_timing_derivative",
    "taylort2_modes_native_supported",
    "taylort2_modes_torch",
    "taylort2_native_supported",
    "taylort2_td_torch",
]
