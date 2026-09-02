# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native nonspinning TaylorT4 evolution and polarizations.

This module ports the two-state TaylorT4 evolution used by LALSuite.  The
state contains the post-Newtonian velocity parameter ``v`` and orbital phase
``phi`` and is advanced at the requested output cadence with classical RK4.
Because that recurrence is strictly sequential and operates on only two
scalars, it is evaluated by a small compiled host-double kernel and transferred
to the requested Torch device once.  A pure-Python equivalent is retained for
source trees where the extension has not yet been built.  This avoids thousands
of tiny device launches without moving the bulk polarization or mode
construction off device.  The sample immediately beyond Schwarzschild ISCO is
discarded, matching the reference implementation's length and epoch
conventions.

The orbital core is kept separate from the shared PN polarization builder so
it can also serve TaylorT4 modes and the other nonspinning TaylorT families.
"""

from __future__ import annotations

import bisect
import math
import operator
from array import array
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

try:
    from pycbc.waveform.taylort4_cpu import (
        evolve_taylor_t4 as _evolve_taylor_t4_compiled,
    )
except ImportError:  # pragma: no cover - exercised in unbuilt source trees
    _evolve_taylor_t4_compiled = None


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
class TaylorT4Coefficients:
    """Coefficients of the nonspinning TaylorT4 velocity derivative."""

    total_mass_seconds: torch.Tensor
    leading: torch.Tensor
    pn2: torch.Tensor
    pn3: torch.Tensor
    pn4: torch.Tensor
    pn5: torch.Tensor
    pn6: torch.Tensor
    pn6_log: torch.Tensor
    pn7: torch.Tensor
    tidal10: torch.Tensor
    tidal12: torch.Tensor
    phase_order: int


@dataclass(frozen=True)
class _TaylorT4ScalarCoefficients:
    """Host-double coefficients for the sequential orbital recurrence."""

    total_mass_seconds: float
    leading: float
    pn2: float
    pn3: float
    pn4: float
    pn5: float
    pn6: float
    pn6_log: float
    pn7: float
    tidal10: float
    tidal12: float
    phase_order: int

    def values(self):
        """Return the floating coefficients in public tensor field order."""

        return (
            self.total_mass_seconds,
            self.leading,
            self.pn2,
            self.pn3,
            self.pn4,
            self.pn5,
            self.pn6,
            self.pn6_log,
            self.pn7,
            self.tidal10,
            self.tidal12,
        )


@dataclass(frozen=True)
class TaylorT4Orbit:
    """Uniformly sampled TaylorT4 orbital velocity and phase."""

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


def _tidal_5pn_coefficient(mass_fraction):
    return 6.0 * mass_fraction**4 * (12.0 - 11.0 * mass_fraction)


def _tidal_6pn_coefficient(mass_fraction):
    return mass_fraction**4 * (
        4421.0 / 56.0
        - 12263.0 / 56.0 * mass_fraction
        + 1893.0 / 4.0 * mass_fraction**2
        - 661.0 / 2.0 * mass_fraction**3
    )


def _taylor_t4_scalar_coefficients(
    mass1,
    mass2,
    *,
    lambda1=0.0,
    lambda2=0.0,
    phase_order=-1,
    tidal_order=-1,
):
    """Build host-double TaylorT4 coefficients and validate their inputs."""

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

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    total_mass_seconds = total_mass * _MTSUN_SI

    tidal10 = 0.0
    tidal12 = 0.0
    if tidal_order in (-1, 12):
        tidal12 = lambda1 * _tidal_6pn_coefficient(
            mass1_fraction
        ) + lambda2 * _tidal_6pn_coefficient(mass2_fraction)
    if tidal_order in (-1, 10, 12):
        tidal10 = lambda1 * _tidal_5pn_coefficient(
            mass1_fraction
        ) + lambda2 * _tidal_5pn_coefficient(mass2_fraction)

    return _TaylorT4ScalarCoefficients(
        total_mass_seconds,
        32.0 / 5.0 * eta / total_mass_seconds,
        -(743.0 + 924.0 * eta) / 336.0,
        4.0 * math.pi,
        (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0,
        -math.pi * (4159.0 + 15876.0 * eta) / 672.0,
        16447.322263 / 139.7088
        - 1712.0 / 105.0 * _EULER_GAMMA
        - 561.98689 / 2.17728 * eta
        + math.pi**2 * (16.0 / 3.0 + 451.0 / 48.0 * eta)
        + 541.0 / 896.0 * eta**2
        - 5605.0 / 2592.0 * eta**3
        - 856.0 / 105.0 * math.log(16.0),
        -1712.0 / 105.0,
        math.pi / 12096.0 * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2),
        tidal10,
        tidal12,
        phase_order,
    )


def taylor_t4_coefficients(
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
    """Build TaylorT4 coefficients for masses expressed in solar masses."""

    coefficients = _taylor_t4_scalar_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
    )
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("TaylorT4 evolution requires float32 or float64")
    values = torch.tensor(
        coefficients.values(),
        device=device,
        dtype=dtype,
    )
    return TaylorT4Coefficients(*values, phase_order=coefficients.phase_order)


def taylor_t4_rhs(state, coefficients):
    """Return ``(dv/dt, dphi/dt)`` for one TaylorT4 state."""

    state = torch.as_tensor(
        state,
        device=coefficients.leading.device,
        dtype=coefficients.leading.dtype,
    )
    if state.shape != (2,):
        raise ValueError("TaylorT4 state must have shape (2,)")

    velocity = state[0]
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity4 = velocity3 * velocity
    velocity5 = velocity4 * velocity
    velocity6 = velocity5 * velocity
    velocity7 = velocity6 * velocity
    velocity9 = velocity7 * velocity2
    velocity10 = velocity9 * velocity
    velocity12 = velocity10 * velocity2

    series = torch.ones_like(velocity)
    order = coefficients.phase_order
    if order in (-1, 2, 3, 4, 5, 6, 7):
        series = series + coefficients.pn2 * velocity2
    if order in (-1, 3, 4, 5, 6, 7):
        series = series + coefficients.pn3 * velocity3
    if order in (-1, 4, 5, 6, 7):
        series = series + coefficients.pn4 * velocity4
    if order in (-1, 5, 6, 7):
        series = series + coefficients.pn5 * velocity5
    if order in (-1, 6, 7):
        series = (
            series
            + (coefficients.pn6 + coefficients.pn6_log * torch.log(velocity))
            * velocity6
        )
    if order in (-1, 7):
        series = series + coefficients.pn7 * velocity7
    # LALSuite includes tidal terms at every supported phase order except 0PN.
    if order != 0:
        series = (
            series
            + coefficients.tidal10 * velocity10
            + coefficients.tidal12 * velocity12
        )

    return torch.stack(
        (
            (coefficients.leading * series) * velocity9,
            velocity3 / coefficients.total_mass_seconds,
        )
    )


def _rk4_step(state, delta_t, coefficients):
    k1 = taylor_t4_rhs(state, coefficients)
    k2 = taylor_t4_rhs(state + 0.5 * delta_t * k1, coefficients)
    k3 = taylor_t4_rhs(state + 0.5 * delta_t * k2, coefficients)
    k4 = taylor_t4_rhs(state + delta_t * k3, coefficients)
    return state + delta_t / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _taylor_t4_scalar_rhs(velocity, coefficients):
    """Return the two TaylorT4 derivatives for one host-double velocity."""

    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity4 = velocity3 * velocity
    velocity5 = velocity4 * velocity
    velocity6 = velocity5 * velocity
    velocity7 = velocity6 * velocity
    velocity9 = velocity7 * velocity2
    velocity10 = velocity9 * velocity
    velocity12 = velocity10 * velocity2

    series = 1.0
    order = coefficients.phase_order
    if order in (-1, 2, 3, 4, 5, 6, 7):
        series += coefficients.pn2 * velocity2
    if order in (-1, 3, 4, 5, 6, 7):
        series += coefficients.pn3 * velocity3
    if order in (-1, 4, 5, 6, 7):
        series += coefficients.pn4 * velocity4
    if order in (-1, 5, 6, 7):
        series += coefficients.pn5 * velocity5
    if order in (-1, 6, 7):
        series += (
            coefficients.pn6 + coefficients.pn6_log * math.log(velocity)
        ) * velocity6
    if order in (-1, 7):
        series += coefficients.pn7 * velocity7
    if order != 0:
        series += coefficients.tidal10 * velocity10 + coefficients.tidal12 * velocity12

    return (
        coefficients.leading * series * velocity9,
        velocity3 / coefficients.total_mass_seconds,
    )


def _rk4_scalar_step(velocity, phase, delta_t, coefficients):
    """Advance the host-double TaylorT4 state by one classical RK4 step."""

    k1_velocity, k1_phase = _taylor_t4_scalar_rhs(velocity, coefficients)
    k2_velocity, k2_phase = _taylor_t4_scalar_rhs(
        velocity + 0.5 * delta_t * k1_velocity,
        coefficients,
    )
    k3_velocity, k3_phase = _taylor_t4_scalar_rhs(
        velocity + 0.5 * delta_t * k2_velocity,
        coefficients,
    )
    k4_velocity, k4_phase = _taylor_t4_scalar_rhs(
        velocity + delta_t * k3_velocity,
        coefficients,
    )
    scale = delta_t / 6.0
    return (
        velocity
        + scale * (k1_velocity + 2.0 * k2_velocity + 2.0 * k3_velocity + k4_velocity),
        phase + scale * (k1_phase + 2.0 * k2_phase + 2.0 * k3_phase + k4_phase),
    )


def _evolve_taylor_t4_python(initial_velocity, delta_t, coefficients):
    """Pure-Python fallback for the compiled scalar recurrence."""

    velocity = initial_velocity
    phase = 0.0
    velocity_samples = array("d", (velocity,))
    phase_samples = array("d", (phase,))

    while True:
        try:
            next_velocity, next_phase = _rk4_scalar_step(
                velocity,
                phase,
                delta_t,
                coefficients,
            )
        except (OverflowError, ValueError) as exc:
            raise RuntimeError(
                "TaylorT4 evolution produced a non-finite state"
            ) from exc
        if not math.isfinite(next_velocity) or not math.isfinite(next_phase):
            raise RuntimeError("TaylorT4 evolution produced a non-finite state")
        if next_velocity > _ISCO_VELOCITY:
            break
        velocity_samples.append(next_velocity)
        phase_samples.append(next_phase)
        velocity = next_velocity
        phase = next_phase

    return velocity_samples, phase_samples


def taylor_t4_orbit(
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
    """Evolve a nonspinning TaylorT4 orbit on a Torch device.

    The returned epoch is ``-len(orbit) * delta_t``, matching LALSuite rather
    than the timestamp of the first sample relative to the final sample.
    ``coa_phase`` is the orbital phase at ``f_ref``; when ``f_ref`` is zero it
    is assigned to the final retained sample.
    """

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

    coefficients = _taylor_t4_scalar_coefficients(
        mass1,
        mass2,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
    )
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("TaylorT4 evolution requires float32 or float64")
    target = torch.empty(0, device=device, dtype=dtype)
    total_mass_seconds = coefficients.total_mass_seconds
    isco_frequency = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    if f_ref != 0.0 and f_ref < f_lower:
        raise ValueError("f_ref must be zero or at least f_lower")
    if f_ref >= isco_frequency:
        raise ValueError("f_ref must be below Schwarzschild ISCO")
    initial_velocity = math.cbrt(math.pi * total_mass_seconds * f_lower)
    reference_velocity = (
        0.0
        if f_ref == 0.0
        else math.cbrt(math.pi * f_ref * coefficients.total_mass_seconds)
    )

    if _evolve_taylor_t4_compiled is not None:
        samples = _evolve_taylor_t4_compiled(
            initial_velocity,
            delta_t,
            reference_velocity,
            coa_phase,
            *coefficients.values(),
            coefficients.phase_order,
        )
        sample_count = samples.shape[1]
        trajectory = torch.as_tensor(
            samples,
            device=target.device,
            dtype=dtype,
        )
    else:
        velocity_samples, phase_samples = _evolve_taylor_t4_python(
            initial_velocity,
            delta_t,
            coefficients,
        )
        if f_ref == 0.0:
            phase_index = len(velocity_samples) - 1
        elif f_ref == f_lower:
            phase_index = 0
        else:
            insertion_index = bisect.bisect_right(
                velocity_samples,
                reference_velocity,
            )
            phase_index = insertion_index - 1
            if (
                phase_index < 0
                or phase_index >= len(velocity_samples)
                or insertion_index == len(velocity_samples)
            ):
                raise ValueError("f_ref must lie between f_lower and ISCO")
        phase_offset = coa_phase - phase_samples[phase_index]
        for index in range(len(phase_samples)):
            phase_samples[index] += phase_offset
        sample_count = len(velocity_samples)
        trajectory = torch.tensor(
            (velocity_samples, phase_samples),
            device=target.device,
            dtype=dtype,
        )
    velocity_tensor, phase_tensor = trajectory

    return TaylorT4Orbit(
        velocity=velocity_tensor,
        phase=phase_tensor,
        delta_t=delta_t,
        epoch=-sample_count * delta_t,
    )


def _supported_order(value, supported):
    try:
        return operator.index(value) in supported
    except TypeError:
        return False


def taylort4_native_supported(parameters):
    """Return whether the native TaylorT4 TD port covers ``parameters``."""

    if parameters.get("approximant", "TaylorT4") != "TaylorT4":
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


def taylort4_default_native_supported(_parameters):
    """Return whether unflagged native use is competitive on this device.

    The compiled recurrence makes CPU execution comparable to LAL. Apple MPS
    remains an explicit opt-in because its small-kernel dispatch overhead made
    the audited waveform path materially slower than LAL plus device transfer.
    """

    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


def taylort4_modes_native_supported(parameters):
    """Return whether the native TaylorT4 mode port covers ``parameters``."""

    if not taylort4_native_supported(parameters):
        return False
    try:
        ell_max = operator.index(parameters.get("ell_max", 5))
    except TypeError:
        return False
    return 2 <= ell_max <= 6


def _finite_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"TaylorT4 requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"TaylorT4 requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"TaylorT4 requires finite {name}")
    return value


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native TaylorT4 requires an active TorchScheme")
    device = state.torch_device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return device, dtype


def taylort4_td_torch(**parameters):
    """Generate nonspinning TaylorT4 polarizations on the Torch device."""

    mass1 = _finite_float(parameters, "mass1")
    mass2 = _finite_float(parameters, "mass2")
    distance = _finite_float(parameters, "distance", 1.0)
    inclination = _finite_float(parameters, "inclination", 0.0)
    coa_phase = _finite_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _finite_float(parameters, "long_asc_nodes", 0.0)
    delta_t = _finite_float(parameters, "delta_t")
    f_lower = _finite_float(parameters, "f_lower")
    f_ref = _finite_float(parameters, "f_ref", 0.0)
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
    phase_order = parameters.get("phase_order", -1)
    amplitude_order = parameters.get("amplitude_order", -1)
    tidal_order = parameters.get("tidal_order", -1)
    device, dtype = _target_device_dtype()

    orbit = taylor_t4_orbit(
        mass1,
        mass2,
        delta_t,
        f_lower,
        coa_phase=coa_phase,
        f_ref=f_ref,
        lambda1=lambda1,
        lambda2=lambda2,
        phase_order=phase_order,
        tidal_order=tidal_order,
        device=device,
        dtype=dtype,
    )
    plus, cross = pn_polarizations(
        orbit.velocity,
        orbit.phase,
        mass1,
        mass2,
        distance,
        inclination,
        amplitude_order=amplitude_order,
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
            delta_t=delta_t,
            epoch=orbit.epoch,
            copy=False,
        ),
        TimeSeries(
            TorchArrayData(cross),
            delta_t=delta_t,
            epoch=orbit.epoch,
            copy=False,
        ),
    )


def taylort4_modes_torch(**parameters):
    """Generate nonspinning TaylorT4 modes on the Torch device.

    LAL's legacy mode interface ignores reference phase and tidal effects for
    TaylorT4. Preserve those conventions even though polarization generation
    supports both.
    """

    mass1 = _finite_float(parameters, "mass1")
    mass2 = _finite_float(parameters, "mass2")
    distance = _finite_float(parameters, "distance", 1.0)
    delta_t = _finite_float(parameters, "delta_t")
    f_lower = _finite_float(parameters, "f_lower")
    f_ref = _finite_float(parameters, "f_ref", 0.0)
    phase_order = parameters.get("phase_order", -1)
    amplitude_order = parameters.get("amplitude_order", -1)
    ell_max = parameters.get("ell_max", 5)
    device, dtype = _target_device_dtype()

    orbit = taylor_t4_orbit(
        mass1,
        mass2,
        delta_t,
        f_lower,
        coa_phase=0.0,
        f_ref=f_ref,
        phase_order=phase_order,
        tidal_order=0,
        device=device,
        dtype=dtype,
    )
    modes = pn_modes_lal_convention(
        orbit.velocity,
        orbit.phase,
        mass1,
        mass2,
        distance,
        ell_max=ell_max,
        amplitude_order=amplitude_order,
    )
    return {
        mode: (
            TimeSeries(
                TorchArrayData(real),
                delta_t=delta_t,
                epoch=orbit.epoch,
                copy=False,
            ),
            TimeSeries(
                TorchArrayData(imaginary),
                delta_t=delta_t,
                epoch=orbit.epoch,
                copy=False,
            ),
        )
        for mode, (real, imaginary) in modes.items()
    }


__all__ = [
    "TaylorT4Coefficients",
    "TaylorT4Orbit",
    "taylor_t4_coefficients",
    "taylor_t4_orbit",
    "taylor_t4_rhs",
    "taylort4_default_native_supported",
    "taylort4_modes_native_supported",
    "taylort4_modes_torch",
    "taylort4_native_supported",
    "taylort4_td_torch",
]
