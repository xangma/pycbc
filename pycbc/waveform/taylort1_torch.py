# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native nonspinning TaylorT1 evolution, polarizations, and modes."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

import pycbc.scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.constants import _EULER_GAMMA, _MTSUN_SI
from pycbc.waveform.utils_torch import (
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _is_default_order,
    _is_nonzero,
)
from pycbc.waveform.pn_modes_torch import pn_modes_lal_convention
from pycbc.waveform.pn_polarization_torch import pn_polarizations
from pycbc.waveform.rkf45_torch import adaptive_rkf45_hermite


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
class TaylorT1Coefficients:
    """Coefficients of TaylorT1's un-reexpanded energy-flux ratio."""

    total_mass_seconds: torch.Tensor
    energy_leading: torch.Tensor
    energy2: torch.Tensor
    energy4: torch.Tensor
    energy6: torch.Tensor
    energy10: torch.Tensor
    energy12: torch.Tensor
    flux_leading: torch.Tensor
    flux2: torch.Tensor
    flux3: torch.Tensor
    flux4: torch.Tensor
    flux5: torch.Tensor
    flux6: torch.Tensor
    flux6_log: torch.Tensor
    flux7: torch.Tensor
    flux10: torch.Tensor
    flux12: torch.Tensor
    phase_order: int


@dataclass(frozen=True)
class TaylorT1Orbit:
    """Uniformly sampled TaylorT1 orbital velocity and phase."""

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


def _energy_tidal_5pn(mass_fraction):
    return -9.0 * mass_fraction**4 * (1.0 - mass_fraction)


def _energy_tidal_6pn(mass_fraction):
    return mass_fraction**4 * (
        -33.0 / 2.0
        + 11.0 / 2.0 * mass_fraction
        - 11.0 / 2.0 * mass_fraction**2
        + 33.0 / 2.0 * mass_fraction**3
    )


def _flux_tidal_5pn(mass_fraction):
    return 6.0 * (3.0 - 2.0 * mass_fraction) * mass_fraction**4


def _flux_tidal_6pn(mass_fraction):
    return mass_fraction**4 * (
        -176.0 / 7.0
        - 1803.0 / 28.0 * mass_fraction
        + 643.0 / 4.0 * mass_fraction**2
        - 155.0 / 2.0 * mass_fraction**3
    )


def taylor_t1_coefficients(
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
    """Build TaylorT1 coefficients for masses expressed in solar masses."""

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
        raise ValueError("TaylorT1 evolution requires float32 or float64")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass

    energy10 = 0.0
    energy12 = 0.0
    flux10 = 0.0
    flux12 = 0.0
    if tidal_order in (-1, 12):
        energy12 = lambda1 * _energy_tidal_6pn(
            mass1_fraction
        ) + lambda2 * _energy_tidal_6pn(mass2_fraction)
        flux12 = lambda1 * _flux_tidal_6pn(
            mass1_fraction
        ) + lambda2 * _flux_tidal_6pn(mass2_fraction)
    if tidal_order in (-1, 10, 12):
        energy10 = lambda1 * _energy_tidal_5pn(
            mass1_fraction
        ) + lambda2 * _energy_tidal_5pn(mass2_fraction)
        flux10 = lambda1 * _flux_tidal_5pn(
            mass1_fraction
        ) + lambda2 * _flux_tidal_5pn(mass2_fraction)

    values = torch.tensor(
        (
            total_mass * _MTSUN_SI,
            -eta / 2.0,
            -(0.75 + eta / 12.0),
            -(27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0),
            -(
                67.5 / 6.4
                - (344.45 / 5.76 - 20.5 / 9.6 * math.pi**2) * eta
                + 15.5 / 9.6 * eta**2
                + 3.5 / 518.4 * eta**3
            ),
            energy10,
            energy12,
            32.0 * eta**2 / 5.0,
            -(12.47 / 3.36 + 3.5 / 1.2 * eta),
            4.0 * math.pi,
            -(44.711 / 9.072 - 92.71 / 5.04 * eta - 6.5 / 1.8 * eta**2),
            -(81.91 / 6.72 + 58.3 / 2.4 * eta) * math.pi,
            664.3739519 / 6.9854400
            + 16.0 / 3.0 * math.pi**2
            - 17.12 / 1.05 * _EULER_GAMMA
            - 17.12 / 1.05 * math.log(4.0)
            + (4.1 / 4.8 * math.pi**2 - 134.543 / 7.776) * eta
            - 94.403 / 3.024 * eta**2
            - 7.75 / 3.24 * eta**3,
            -17.12 / 1.05,
            -(
                162.85 / 5.04
                - 214.745 / 1.728 * eta
                - 193.385 / 3.024 * eta**2
            )
            * math.pi,
            flux10,
            flux12,
        ),
        device=device,
        dtype=dtype,
    )
    return TaylorT1Coefficients(*values, phase_order=phase_order)


def taylor_t1_rhs(state, coefficients):
    """Return ``(dv/dt, dphi/dt)`` for one TaylorT1 state."""

    state = torch.as_tensor(
        state,
        device=coefficients.energy_leading.device,
        dtype=coefficients.energy_leading.dtype,
    )
    if state.shape != (2,):
        raise ValueError("TaylorT1 state must have shape (2,)")

    velocity = state[0]
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity4 = velocity2 * velocity2
    velocity5 = velocity4 * velocity
    velocity6 = velocity4 * velocity2
    velocity7 = velocity6 * velocity
    velocity10 = velocity5 * velocity5
    velocity12 = velocity6 * velocity6
    order = coefficients.phase_order

    energy_series = torch.ones_like(velocity)
    if order in (-1, 2, 3, 4, 5, 6, 7):
        energy_series = energy_series + 2.0 * coefficients.energy2 * velocity2
    if order in (-1, 4, 5, 6, 7):
        energy_series = energy_series + 3.0 * coefficients.energy4 * velocity4
    if order in (-1, 6, 7):
        energy_series = energy_series + 4.0 * coefficients.energy6 * velocity6
    if order != 0:
        energy_series = (
            energy_series
            + 6.0 * coefficients.energy10 * velocity10
            + 7.0 * coefficients.energy12 * velocity12
        )
    energy_derivative = (
        2.0 * coefficients.energy_leading * velocity * energy_series
    )

    flux_series = torch.ones_like(velocity)
    if order in (-1, 2, 3, 4, 5, 6, 7):
        flux_series = flux_series + coefficients.flux2 * velocity2
    if order in (-1, 3, 4, 5, 6, 7):
        flux_series = flux_series + coefficients.flux3 * velocity3
    if order in (-1, 4, 5, 6, 7):
        flux_series = flux_series + coefficients.flux4 * velocity4
    if order in (-1, 5, 6, 7):
        flux_series = flux_series + coefficients.flux5 * velocity5
    if order in (-1, 6, 7):
        flux_series = flux_series + (
            coefficients.flux6 + coefficients.flux6_log * torch.log(velocity)
        ) * velocity6
    if order in (-1, 7):
        flux_series = flux_series + coefficients.flux7 * velocity7
    if order != 0:
        flux_series = (
            flux_series
            + coefficients.flux10 * velocity10
            + coefficients.flux12 * velocity12
        )
    flux = coefficients.flux_leading * velocity10 * flux_series

    return torch.stack(
        (
            -flux / energy_derivative / coefficients.total_mass_seconds,
            velocity3 / coefficients.total_mass_seconds,
        )
    )


def taylor_t1_orbit(
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
    """Evolve a nonspinning TaylorT1 orbit on a Torch device."""

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

    coefficients = taylor_t1_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
        device=device,
        dtype=dtype,
    )
    total_mass_seconds = float(coefficients.total_mass_seconds.item())
    isco_frequency = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    if f_ref != 0.0 and f_ref < f_lower:
        raise ValueError("f_ref must be zero or at least f_lower")
    if f_ref >= isco_frequency:
        raise ValueError("f_ref must be below Schwarzschild ISCO")

    initial_velocity = torch.as_tensor(
        math.cbrt(math.pi * total_mass_seconds * f_lower),
        device=coefficients.energy_leading.device,
        dtype=coefficients.energy_leading.dtype,
    )
    initial_state = torch.stack(
        (initial_velocity, torch.zeros_like(initial_velocity))
    )

    def right_hand_side(time, state):
        del time
        return taylor_t1_rhs(state, coefficients)

    def reached_isco(time, state):
        del time
        return state[0] >= _ISCO_VELOCITY

    tolerance = max(1.0e-12, 8.0 * torch.finfo(dtype).eps)
    trajectory = adaptive_rkf45_hermite(
        right_hand_side,
        initial_state,
        delta_t,
        reached_isco,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    velocity = trajectory[:, 0]
    phase = trajectory[:, 1]

    if f_ref == 0.0:
        phase_index = len(trajectory) - 1
    elif f_ref == f_lower:
        phase_index = 0
    else:
        reference_velocity = torch.pow(
            torch.as_tensor(
                math.pi * f_ref,
                device=velocity.device,
                dtype=velocity.dtype,
            )
            * coefficients.total_mass_seconds,
            1.0 / 3.0,
        )
        insertion = torch.searchsorted(
            velocity.contiguous(), reference_velocity, right=True
        )
        insertion_index = int(insertion.item())
        phase_index = insertion_index - 1
        if (
            phase_index < 0
            or phase_index >= len(trajectory)
            or insertion_index == len(trajectory)
        ):
            raise ValueError("f_ref must lie between f_lower and ISCO")
    phase = phase + (coa_phase - phase[phase_index])

    return TaylorT1Orbit(
        velocity=velocity,
        phase=phase,
        delta_t=delta_t,
        epoch=-(len(trajectory) - 1) * delta_t,
    )


def _supported_order(value, supported):
    try:
        return operator.index(value) in supported
    except TypeError:
        return False


def taylort1_native_supported(parameters):
    """Return whether the native TaylorT1 TD port covers ``parameters``."""

    if parameters.get("approximant", "TaylorT1") != "TaylorT1":
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


def taylort1_modes_native_supported(parameters):
    """Return whether the native TaylorT1 mode port covers ``parameters``."""

    if not taylort1_native_supported(parameters):
        return False
    try:
        ell_max = operator.index(parameters.get("ell_max", 5))
    except TypeError:
        return False
    return 2 <= ell_max <= 6


def _finite_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"TaylorT1 requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"TaylorT1 requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"TaylorT1 requires finite {name}")
    return value


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native TaylorT1 requires an active TorchScheme")
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


def taylort1_td_torch(**parameters):
    """Generate nonspinning TaylorT1 polarizations on the Torch device."""

    values = _waveform_parameters(parameters)
    inclination = _finite_float(parameters, "inclination", 0.0)
    coa_phase = _finite_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _finite_float(parameters, "long_asc_nodes", 0.0)
    device, dtype = _target_device_dtype()
    orbit = taylor_t1_orbit(
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


def taylort1_modes_torch(**parameters):
    """Generate nonspinning TaylorT1 modes on the Torch device.

    LAL's legacy mode interface ignores coalescence phase but retains the
    requested reference frequency and tidal evolution.
    """

    values = _waveform_parameters(parameters)
    ell_max = parameters.get("ell_max", 5)
    device, dtype = _target_device_dtype()
    orbit = taylor_t1_orbit(
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
    "TaylorT1Coefficients",
    "TaylorT1Orbit",
    "taylor_t1_coefficients",
    "taylor_t1_orbit",
    "taylor_t1_rhs",
    "taylort1_modes_native_supported",
    "taylort1_modes_torch",
    "taylort1_native_supported",
    "taylort1_td_torch",
]
