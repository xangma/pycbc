# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native radiation kernels for the SpinTaylor waveform families.

The precessing polarization builder is a vectorized port of LALSuite's
``XLALSimInspiralPrecessingPolarizationWaveforms``.  It consumes orbital
states in the radiation frame and implements the known amplitudes through
1.5PN, including the 1PN and 1.5PN spin terms.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

import pycbc.scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.constants import _MRSUN_SI, _MTSUN_SI, _PC_SI
from pycbc.waveform.utils_torch import (
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _is_nonzero,
)
from pycbc.waveform.imrphenomx_spintaylor_torch import (
    _check_spintaylor_physical_state,
    _integrate_uniform_time_branch,
    spintaylor_internal_spins,
    spintaylor_t1_rhs,
    spintaylor_t4_rhs,
    spintaylor_t5_rhs,
)
from pycbc.waveform.spintaylor_modes_torch import spintaylor_modes_from_orbit


_MPC_SI = 1.0e6 * _PC_SI
_SUPPORTED_AMPLITUDE_ORDERS = frozenset((-1, 0, 1, 2, 3))
_SUPPORTED_PHASE_ORDERS = frozenset((-1, 7, 8))
_SUPPORTED_SPIN_ORDERS = frozenset((-1, 6))
_SUPPORTED_TIDAL_ORDERS = frozenset((-1, 0, 10, 12))
_NL_TIDAL_KEYS = (
    "nl_tides_a1",
    "nl_tides_n1",
    "nl_tides_f1",
    "nl_tides_a2",
    "nl_tides_n2",
    "nl_tides_f2",
)


@dataclass(frozen=True)
class SpinTaylorOrbit:
    """Uniformly sampled precessing orbit in LAL's state convention.

    ``state`` stores phase, orbital angular frequency, ``LNhat``, the two
    total-mass-normalized spins, and ``E1``.  The spin properties expose the
    usual dimensionless component spins expected by the radiation kernels.
    """

    state: torch.Tensor
    mass1_fraction: float
    mass2_fraction: float
    delta_t: float
    epoch: float

    def __len__(self):
        return self.state.shape[0]

    @property
    def phase(self):
        return self.state[:, 0]

    @property
    def omega(self):
        return self.state[:, 1]

    @property
    def velocity(self):
        return torch.pow(self.omega, 1.0 / 3.0)

    @property
    def spin1(self):
        return self.state[:, 5:8] / self.mass1_fraction**2

    @property
    def spin2(self):
        return self.state[:, 8:11] / self.mass2_fraction**2

    @property
    def lnhat(self):
        return self.state[:, 2:5]

    @property
    def e1(self):
        return self.state[:, 11:14]


def _finite_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def _reference_axis_rotation(e1, lnhat, angle):
    """Rotate ``E1`` around ``LNhat`` as the LAL SpinTaylor driver does."""

    alpha = torch.atan2(lnhat[1], lnhat[0])
    cosine_alpha = torch.cos(alpha)
    sine_alpha = torch.sin(alpha)
    lxy = lnhat[0] * cosine_alpha + lnhat[1] * sine_alpha

    e1x = e1[0] * cosine_alpha + e1[1] * sine_alpha
    e1y = e1[1] * cosine_alpha - e1[0] * sine_alpha
    e1z = e1[2]
    iota = torch.atan2(lxy, lnhat[2])
    cosine_iota = torch.cos(iota)
    sine_iota = torch.sin(iota)
    rotated_x = e1x * cosine_iota - e1z * sine_iota
    rotated_z = e1z * cosine_iota + e1x * sine_iota

    cosine_angle = math.cos(angle)
    sine_angle = math.sin(angle)
    e1x = rotated_x * cosine_angle - e1y * sine_angle
    e1y = e1y * cosine_angle + rotated_x * sine_angle

    rotated_x = e1x * cosine_iota + rotated_z * sine_iota
    e1z = rotated_z * cosine_iota - e1x * sine_iota
    return torch.stack(
        (
            rotated_x * cosine_alpha - e1y * sine_alpha,
            e1y * cosine_alpha + rotated_x * sine_alpha,
            e1z,
        )
    )


def _spintaylor_orbit(
    approximant,
    mass1,
    mass2,
    delta_t,
    f_lower,
    spin1,
    spin2,
    *,
    coa_phase=0.0,
    f_ref=0.0,
    f_final=0.0,
    lnhat=(0.0, 0.0, 1.0),
    e1=(1.0, 0.0, 0.0),
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
    device=None,
    dtype=torch.float64,
    rtol=None,
    atol=None,
    max_steps=100000,
):
    """Evolve default SpinTaylor dynamics at a physical cadence.

    Masses are in solar masses, frequencies in Hz, and ``delta_t`` in
    seconds.  Input spins and frame vectors are defined at ``f_ref`` when it
    is positive and at ``f_lower`` otherwise.  As in LAL, ``coa_phase``
    rotates ``E1`` about ``LNhat`` rather than shifting the orbital phase.
    A zero ``f_final`` evolves to the first physical PN stopping condition.
    """

    rhs_function = {
        "SpinTaylorT1": spintaylor_t1_rhs,
        "SpinTaylorT4": spintaylor_t4_rhs,
        "SpinTaylorT5": spintaylor_t5_rhs,
    }.get(approximant)
    if rhs_function is None:
        raise ValueError(f"unsupported native SpinTaylor approximant {approximant!r}")

    mass1 = _finite_float(mass1, "mass1")
    mass2 = _finite_float(mass2, "mass2")
    delta_t = _finite_float(delta_t, "delta_t")
    f_lower = _finite_float(f_lower, "f_lower")
    f_ref = _finite_float(f_ref, "f_ref")
    f_final = _finite_float(f_final, "f_final")
    coa_phase = _finite_float(coa_phase, "coa_phase")
    quadrupole1 = _finite_float(quadrupole1, "quadrupole1")
    quadrupole2 = _finite_float(quadrupole2, "quadrupole2")
    lambda1 = _finite_float(lambda1, "lambda1")
    lambda2 = _finite_float(lambda2, "lambda2")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")
    if f_lower <= 0.0:
        raise ValueError("f_lower must be positive")
    if f_ref < 0.0:
        raise ValueError("f_ref must be nonnegative")
    if f_final < 0.0:
        raise ValueError("f_final must be nonnegative")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("SpinTaylor evolution requires float32 or float64")

    if device is None:
        reference = next(
            (
                value
                for value in (spin1, spin2, lnhat, e1)
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        device = reference.device if reference is not None else torch.device("cpu")
    device = torch.device(device)
    vectors = tuple(
        torch.as_tensor(value, device=device, dtype=dtype)
        for value in (spin1, spin2, lnhat, e1)
    )
    if any(value.shape != (3,) for value in vectors):
        raise ValueError("SpinTaylor spins and frame vectors must have length three")
    if not bool(torch.isfinite(torch.cat(vectors)).all().item()):
        raise ValueError("SpinTaylor spins and frame vectors must be finite")
    spin1, spin2, lnhat, e1 = vectors

    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * _MTSUN_SI
    isco_frequency = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)
    if f_ref != 0.0 and f_ref < f_lower:
        raise ValueError("f_ref must be zero or at least f_lower")
    if f_ref >= isco_frequency:
        raise ValueError("f_ref must be below Schwarzschild ISCO")
    at_start = abs(f_ref - f_lower) < torch.finfo(torch.float32).eps
    reference_frequency = f_lower if f_ref == 0.0 or at_start else f_ref
    if f_final != 0.0 and f_final <= reference_frequency:
        raise ValueError("f_final must be zero or greater than the reference frequency")

    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("max_steps must be a positive integer") from exc
    epsilon = torch.finfo(dtype).eps
    rtol = max(1.0e-12, 32.0 * epsilon) if rtol is None else float(rtol)
    atol = max(1.0e-12, 32.0 * epsilon) if atol is None else float(atol)
    if (
        not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol <= 0.0
        or atol <= 0.0
        or max_steps < 1
    ):
        raise ValueError("solver tolerances and max_steps must be positive")

    e1 = _reference_axis_rotation(e1, lnhat, coa_phase)
    reference_mf = torch.as_tensor(
        reference_frequency * total_mass_seconds,
        device=device,
        dtype=dtype,
    )
    internal_spin1, internal_spin2 = spintaylor_internal_spins(
        mass1, mass2, spin1, spin2
    )
    reference_state = torch.cat(
        (
            torch.zeros(1, device=device, dtype=dtype),
            (math.pi * reference_mf).reshape(1),
            lnhat,
            internal_spin1,
            internal_spin2,
            e1,
        )
    )
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass

    def rhs(state):
        return rhs_function(
            state,
            mass1_fraction,
            mass2_fraction,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            lambda1=lambda1,
            lambda2=lambda2,
            tidal_order=tidal_order,
        )

    def physical_check(state, derivatives):
        _check_spintaylor_physical_state(
            state,
            mass1_fraction,
            mass2_fraction,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            lambda1=lambda1,
            lambda2=lambda2,
            tidal_order=tidal_order,
            derivatives=derivatives,
        )

    output_step = delta_t / total_mass_seconds
    if reference_frequency > f_lower:
        backward, reached_start = _integrate_uniform_time_branch(
            reference_state,
            f_lower * total_mass_seconds,
            output_step,
            -1.0,
            rhs,
            physical_check,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            retain_target_crossing=True,
            retain_physical_boundary_outputs=True,
        )
        if not reached_start:
            raise RuntimeError(
                f"{approximant} reached a physical boundary before f_lower"
            )
        ordered = list(reversed(backward))
    else:
        ordered = [reference_state]

    target_mf = math.inf if f_final == 0.0 else f_final * total_mass_seconds
    forward, _ = _integrate_uniform_time_branch(
        reference_state,
        target_mf,
        output_step,
        1.0,
        rhs,
        physical_check,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        retain_target_crossing=True,
        retain_physical_boundary_outputs=True,
        allow_asymptotic_boundary=approximant in ("SpinTaylorT1", "SpinTaylorT5"),
    )
    ordered.extend(forward[1:])
    state = torch.stack(ordered)
    if f_ref == 0.0:
        state = state.clone()
        state[:, 0] -= state[-1, 0]
    if state.shape[0] > 1 and not bool((state[1:, 1] > state[:-1, 1]).all().item()):
        raise RuntimeError(f"{approximant} orbit is not frequency ordered")

    return SpinTaylorOrbit(
        state=state,
        mass1_fraction=mass1_fraction,
        mass2_fraction=mass2_fraction,
        delta_t=delta_t,
        epoch=-(state.shape[0] - 1) * delta_t,
    )


def spintaylor_t1_orbit(*args, **kwargs):
    """Evolve the default SpinTaylorT1 dynamics at a physical cadence."""

    return _spintaylor_orbit("SpinTaylorT1", *args, **kwargs)


def spintaylor_t4_orbit(*args, **kwargs):
    """Evolve the default SpinTaylorT4 dynamics at a physical cadence."""

    return _spintaylor_orbit("SpinTaylorT4", *args, **kwargs)


def spintaylor_t5_orbit(*args, **kwargs):
    """Evolve the default SpinTaylorT5 dynamics at a physical cadence."""

    return _spintaylor_orbit("SpinTaylorT5", *args, **kwargs)


def _amplitude_order(value):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("amplitude_order must be an integer") from exc
    if value not in _SUPPORTED_AMPLITUDE_ORDERS:
        choices = ", ".join(str(item) for item in sorted(_SUPPORTED_AMPLITUDE_ORDERS))
        raise ValueError(
            f"unsupported amplitude_order {value}; expected one of {choices}"
        )
    return value


def _validate_orbit_tensors(velocity, phase, spin1, spin2, lnhat, e1):
    tensors = (velocity, phase, spin1, spin2, lnhat, e1)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("SpinTaylor orbit inputs must be Torch tensors")
    if velocity.ndim != 1 or phase.shape != velocity.shape:
        raise ValueError(
            "velocity and phase must be same-shaped one-dimensional tensors"
        )
    vector_shape = velocity.shape + (3,)
    if any(value.shape != vector_shape for value in (spin1, spin2, lnhat, e1)):
        raise ValueError("SpinTaylor vector inputs must have shape (samples, 3)")
    if any(
        value.device != velocity.device or value.dtype != velocity.dtype
        for value in tensors[1:]
    ):
        raise ValueError("SpinTaylor orbit inputs must share a device and dtype")
    if velocity.dtype not in (torch.float32, torch.float64):
        raise ValueError("SpinTaylor polarizations require float32 or float64")
    combined = torch.cat(
        (
            velocity,
            phase,
            spin1.reshape(-1),
            spin2.reshape(-1),
            lnhat.reshape(-1),
            e1.reshape(-1),
        )
    )
    if not bool(torch.isfinite(combined).all().item()):
        raise ValueError("SpinTaylor orbit inputs must be finite")
    if bool((velocity <= 0.0).any().item()):
        raise ValueError("velocity must be positive")


def spintaylor_polarizations_from_orbit(
    velocity,
    phase,
    spin1,
    spin2,
    lnhat,
    e1,
    mass1,
    mass2,
    distance,
    *,
    amplitude_order=-1,
):
    """Construct precessing SpinTaylor plus and cross polarizations.

    Component masses are in solar masses and distance is in Mpc.  ``spin1``
    and ``spin2`` use the standard dimensionless component-spin convention,
    not the total-mass-normalized internal convention used by the dynamics.
    All vectors are expressed in LAL's radiation frame.
    """

    _validate_orbit_tensors(velocity, phase, spin1, spin2, lnhat, e1)
    amplitude_order = _amplitude_order(amplitude_order)
    mass1 = float(mass1)
    mass2 = float(mass2)
    distance = float(distance)
    if not all(math.isfinite(value) for value in (mass1, mass2, distance)):
        raise ValueError("SpinTaylor polarization parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if distance <= 0.0:
        raise ValueError("distance must be positive")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    dm = (mass1 - mass2) / total_mass
    amplitude_factor = 2.0 * total_mass * _MRSUN_SI * eta / (distance * _MPC_SI)

    s1x, s1y, s1z = spin1.unbind(dim=-1)
    s2x, s2y, s2z = spin2.unbind(dim=-1)
    lnhx, lnhy, lnhz = lnhat.unbind(dim=-1)
    e1x, e1y, e1z = e1.unbind(dim=-1)

    # E2 = LNhat x E1.
    e2x = lnhy * e1z - lnhz * e1y
    e2y = lnhz * e1x - lnhx * e1z
    e2z = lnhx * e1y - lnhy * e1x
    cosine = torch.cos(phase)
    sine = torch.sin(phase)

    # Instantaneous separation and velocity directions.
    nx = e1x * cosine + e2x * sine
    ny = e1y * cosine + e2y * sine
    nz = e1z * cosine + e2z * sine
    lx = e2x * cosine - e1x * sine
    ly = e2y * cosine - e1y * sine
    lz = e2z * cosine - e1z * sine

    nx2, ny2, nz2 = nx * nx, ny * ny, nz * nz
    lx2, ly2, lz2 = lx * lx, ly * ly, lz * lz
    nz3 = nz * nz2
    lz3 = lz * lz2
    zero = torch.zeros_like(velocity)
    hp05 = hp1 = hp15 = hp_spin1 = hp_spin15 = hp_tail15 = zero
    hc05 = hc1 = hc15 = hc_spin1 = hc_spin15 = hc_tail15 = zero

    if amplitude_order == -1 or amplitude_order >= 3:
        hp15 = (
            dm
            * (
                2.0
                * lx
                * nx
                * nz
                * (
                    -95.0
                    + 90.0 * lz2
                    - 65.0 * nz2
                    - 2.0 * eta * (-9.0 + 90.0 * lz2 - 65.0 * nz2)
                )
                - 2.0
                * ly
                * ny
                * nz
                * (
                    -95.0
                    + 90.0 * lz2
                    - 65.0 * nz2
                    - 2.0 * eta * (-9.0 + 90.0 * lz2 - 65.0 * nz2)
                )
                + 6.0
                * lx2
                * lz
                * (
                    13.0
                    - 4.0 * lz2
                    + 29.0 * nz2
                    + eta * (-2.0 + 8.0 * lz2 - 58.0 * nz2)
                )
                - 6.0
                * ly2
                * lz
                * (
                    13.0
                    - 4.0 * lz2
                    + 29.0 * nz2
                    + eta * (-2.0 + 8.0 * lz2 - 58.0 * nz2)
                )
                - lz
                * (nx2 - ny2)
                * (
                    83.0
                    - 6.0 * lz2
                    + 111.0 * nz2
                    + 6.0 * eta * (-1.0 + 2.0 * lz2 - 37.0 * nz2)
                )
            )
            / 24.0
        )
        hc15 = (
            dm
            * (
                lz
                * (6.0 * (19.0 - 4.0 * eta) * lx * ly + (-101.0 + 12.0 * eta) * nx * ny)
                + (-149.0 + 36.0 * eta) * (ly * nx + lx * ny) * nz
                + 6.0
                * (-3.0 + eta)
                * (
                    2.0 * lx * ly * lz
                    - lz * nx * ny
                    - 3.0 * ly * nx * nz
                    - 3.0 * lx * ny * nz
                )
                + (1.0 - 2.0 * eta)
                * (
                    6.0 * lz3 * (-4.0 * lx * ly + nx * ny)
                    + 90.0 * lz2 * (ly * nx + lx * ny) * nz
                    + 3.0 * lz * (58.0 * lx * ly - 37.0 * nx * ny) * nz2
                    - 65.0 * (ly * nx + lx * ny) * nz3
                )
            )
            / 12.0
        )
        hp_spin15 = (
            6.0 * lz * ny * s1x
            + 6.0 * dm * lz * ny * s1x
            - 3.0 * eta * lz * ny * s1x
            + 2.0 * ly2 * lnhy * s1y
            + 2.0 * dm * ly2 * lnhy * s1y
            + 2.0 * eta * ly2 * lnhy * s1y
            + 6.0 * lz * nx * s1y
            + 6.0 * dm * lz * nx * s1y
            - 3.0 * eta * lz * nx * s1y
            + 8.0 * lnhy * nx2 * s1y
            + 8.0 * dm * lnhy * nx2 * s1y
            - eta * lnhy * nx2 * s1y
            - 8.0 * lnhy * ny2 * s1y
            - 8.0 * dm * lnhy * ny2 * s1y
            + eta * lnhy * ny2 * s1y
            + 2.0 * ly2 * lnhz * s1z
            + 2.0 * dm * ly2 * lnhz * s1z
            + 2.0 * eta * ly2 * lnhz * s1z
            - 6.0 * ly * nx * s1z
            - 6.0 * dm * ly * nx * s1z
            - 9.0 * eta * ly * nx * s1z
            + 8.0 * lnhz * nx2 * s1z
            + 8.0 * dm * lnhz * nx2 * s1z
            - eta * lnhz * nx2 * s1z
            - 8.0 * lnhz * ny2 * s1z
            - 8.0 * dm * lnhz * ny2 * s1z
            + eta * lnhz * ny2 * s1z
            + 6.0 * lz * ny * s2x
            - 6.0 * dm * lz * ny * s2x
            - 3.0 * eta * lz * ny * s2x
            + lnhx
            * (
                2.0 * ly2 * ((1.0 + dm + eta) * s1x + (1.0 - dm + eta) * s2x)
                + (nx2 - ny2)
                * ((8.0 + 8.0 * dm - eta) * s1x - (-8.0 + 8.0 * dm + eta) * s2x)
            )
            + 2.0 * ly2 * lnhy * s2y
            - 2.0 * dm * ly2 * lnhy * s2y
            + 2.0 * eta * ly2 * lnhy * s2y
            + 6.0 * lz * nx * s2y
            - 6.0 * dm * lz * nx * s2y
            - 3.0 * eta * lz * nx * s2y
            + 8.0 * lnhy * nx2 * s2y
            - 8.0 * dm * lnhy * nx2 * s2y
            - eta * lnhy * nx2 * s2y
            - 8.0 * lnhy * ny2 * s2y
            + 8.0 * dm * lnhy * ny2 * s2y
            + eta * lnhy * ny2 * s2y
            + 2.0 * ly2 * lnhz * s2z
            - 2.0 * dm * ly2 * lnhz * s2z
            + 2.0 * eta * ly2 * lnhz * s2z
            - 6.0 * ly * nx * s2z
            + 6.0 * dm * ly * nx * s2z
            - 9.0 * eta * ly * nx * s2z
            + 8.0 * lnhz * nx2 * s2z
            - 8.0 * dm * lnhz * nx2 * s2z
            - eta * lnhz * nx2 * s2z
            - 8.0 * lnhz * ny2 * s2z
            + 8.0 * dm * lnhz * ny2 * s2z
            + eta * lnhz * ny2 * s2z
            - 3.0
            * lx
            * ny
            * ((2.0 + 2.0 * dm + 3.0 * eta) * s1z + (2.0 - 2.0 * dm + 3.0 * eta) * s2z)
            - 2.0
            * lx2
            * (
                lnhx * ((1.0 + dm + eta) * s1x + (1.0 - dm + eta) * s2x)
                + lnhy * ((1.0 + dm + eta) * s1y + (1.0 - dm + eta) * s2y)
                + lnhz * ((1.0 + dm + eta) * s1z + (1.0 - dm + eta) * s2z)
            )
        ) / 3.0
        hc_spin15 = (
            -3.0
            * lz
            * (
                nx * ((2.0 + 2.0 * dm - eta) * s1x - (-2.0 + 2.0 * dm + eta) * s2x)
                + ny * ((-2.0 - 2.0 * dm + eta) * s1y + (-2.0 + 2.0 * dm + eta) * s2y)
            )
            + ny
            * (
                -6.0 * ly * s1z
                - 6.0 * dm * ly * s1z
                - 9.0 * eta * ly * s1z
                + 16.0 * lnhz * nx * s1z
                + 16.0 * dm * lnhz * nx * s1z
                - 2.0 * eta * lnhz * nx * s1z
                + 2.0
                * lnhx
                * nx
                * ((8.0 + 8.0 * dm - eta) * s1x - (-8.0 + 8.0 * dm + eta) * s2x)
                + 2.0
                * lnhy
                * nx
                * ((8.0 + 8.0 * dm - eta) * s1y - (-8.0 + 8.0 * dm + eta) * s2y)
                - 6.0 * ly * s2z
                + 6.0 * dm * ly * s2z
                - 9.0 * eta * ly * s2z
                + 16.0 * lnhz * nx * s2z
                - 16.0 * dm * lnhz * nx * s2z
                - 2.0 * eta * lnhz * nx * s2z
            )
            - lx
            * (
                4.0 * lnhx * ly * ((1.0 + dm + eta) * s1x + (1.0 - dm + eta) * s2x)
                - 3.0
                * nx
                * (
                    (2.0 + 2.0 * dm + 3.0 * eta) * s1z
                    + (2.0 - 2.0 * dm + 3.0 * eta) * s2z
                )
                + 4.0
                * ly
                * (
                    lnhy * ((1.0 + dm + eta) * s1y + (1.0 - dm + eta) * s2y)
                    + lnhz * ((1.0 + dm + eta) * s1z + (1.0 - dm + eta) * s2z)
                )
            )
        ) / 3.0
        hp_tail15 = 2.0 * (lx2 - ly2 - nx2 + ny2) * math.pi
        hc_tail15 = 4.0 * (lx * ly - nx * ny) * math.pi

    if amplitude_order == -1 or amplitude_order >= 2:
        hp1 = (
            -13.0 * lx2
            + 13.0 * ly2
            + 6.0 * lx2 * lz2
            - 6.0 * ly2 * lz2
            + 13.0 * (nx2 - ny2)
            - 2.0 * lz2 * (nx2 - ny2)
            - 32.0 * lx * lz * nx * nz
            + 32.0 * ly * lz * ny * nz
            - 14.0 * lx2 * nz2
            + 14.0 * ly2 * nz2
            + 10.0 * (nx2 - ny2) * nz2
        ) / 6.0 + eta * (
            lx2
            - 18.0 * lx2 * lz2
            + 96.0 * lx * lz * nx * nz
            - 96.0 * ly * lz * ny * nz
            + 42.0 * lx2 * nz2
            + ly2 * (-1.0 + 18.0 * lz2 - 42.0 * nz2)
            + (nx2 - ny2) * (-1.0 + 6.0 * lz2 - 30.0 * nz2)
        ) / 6.0
        hc1 = (
            eta
            * (
                lx * ly
                - nx * ny
                - 6.0
                * (
                    lz2 * (3.0 * lx * ly - nx * ny)
                    - 8.0 * lz * (ly * nx + lx * ny) * nz
                    + (-7.0 * lx * ly + 5.0 * nx * ny) * nz2
                )
            )
            / 3.0
            + (
                -13.0 * (lx * ly - nx * ny)
                + 2.0
                * (
                    lz2 * (3.0 * lx * ly - nx * ny)
                    - 8.0 * lz * (ly * nx + lx * ny) * nz
                    + (-7.0 * lx * ly + 5.0 * nx * ny) * nz2
                )
            )
            / 3.0
        )
        hp_spin1 = (
            -ny * ((1.0 + dm) * s1x + (-1.0 + dm) * s2x)
            - nx * ((1.0 + dm) * s1y + (-1.0 + dm) * s2y)
        ) / 2.0
        hc_spin1 = (
            nx * ((1.0 + dm) * s1x + (-1.0 + dm) * s2x)
            - ny * ((1.0 + dm) * s1y + (-1.0 + dm) * s2y)
        ) / 2.0

    if amplitude_order == -1 or amplitude_order >= 1:
        hp05 = (
            dm
            * (
                -2.0 * lx2 * lz
                + 2.0 * ly2 * lz
                + lz * (nx2 - ny2)
                + 6.0 * lx * nx * nz
                - 6.0 * ly * ny * nz
            )
            / 2.0
        )
        hc05 = dm * (
            -2.0 * lx * ly * lz + lz * nx * ny + 3.0 * ly * nx * nz + 3.0 * lx * ny * nz
        )

    hp0 = lx2 - ly2 - nx2 + ny2
    hc0 = 2.0 * lx * ly - 2.0 * nx * ny
    velocity2 = velocity * velocity
    plus = (
        amplitude_factor
        * velocity2
        * (
            hp0
            + velocity
            * (
                hp05
                + velocity
                * (hp1 + hp_spin1 + velocity * (hp15 + hp_spin15 + hp_tail15))
            )
        )
    )
    cross = (
        amplitude_factor
        * velocity2
        * (
            hc0
            + velocity
            * (
                hc05
                + velocity
                * (hc1 + hc_spin1 + velocity * (hc15 + hc_spin15 + hc_tail15))
            )
        )
    )
    return plus, cross


def _supported_order(value, supported):
    try:
        return operator.index(value) in supported
    except TypeError:
        return False


def _spintaylor_native_supported(parameters, approximant):
    if parameters.get("approximant", approximant) != approximant:
        return False
    if not _supported_order(parameters.get("phase_order", -1), _SUPPORTED_PHASE_ORDERS):
        return False
    if not _supported_order(parameters.get("spin_order", -1), _SUPPORTED_SPIN_ORDERS):
        return False
    if not _supported_order(parameters.get("tidal_order", -1), _SUPPORTED_TIDAL_ORDERS):
        return False
    if not _supported_order(
        parameters.get("amplitude_order", -1), _SUPPORTED_AMPLITUDE_ORDERS
    ):
        return False
    if not _supported_order(parameters.get("eccentricity_order", -1), {-1}):
        return False
    try:
        frame_axis = operator.index(parameters.get("frame_axis", 0))
    except TypeError:
        return False
    # PyCBC's zero is a sentinel that leaves LAL's Orbital-L default (2)
    # untouched; an explicitly supplied 2 has the same physical convention.
    if frame_axis not in (0, 2):
        return False
    if any(
        _is_nonzero(parameters.get(key, 0.0))
        for key in (
            ("eccentricity", "mean_per_ano")
            + _NL_TIDAL_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + ("modes_choice", "side_bands")
        )
    ):
        return False
    if parameters.get("mode_array") is not None or parameters.get("numrel_data", ""):
        return False
    return True


def spintaylor_t1_native_supported(parameters):
    """Return whether the native SpinTaylorT1 port covers ``parameters``."""

    return _spintaylor_native_supported(parameters, "SpinTaylorT1")


def spintaylor_t4_native_supported(parameters):
    """Return whether the native SpinTaylorT4 port covers ``parameters``."""

    return _spintaylor_native_supported(parameters, "SpinTaylorT4")


def spintaylor_t5_native_supported(parameters):
    """Return whether the native SpinTaylorT5 port covers ``parameters``."""

    return _spintaylor_native_supported(parameters, "SpinTaylorT5")


def _spintaylor_mode_ells(parameters):
    """Return the mode degrees selected by LAL's SpinTaylor interface."""

    mode_array = parameters.get("mode_array")
    if mode_array is None:
        try:
            ell_max = operator.index(parameters.get("ell_max", 5))
        except TypeError as exc:
            raise ValueError("ell_max must be an integer") from exc
        if ell_max < 2 or ell_max > 5:
            raise ValueError("SpinTaylor ell_max must be between 2 and 5")
        return tuple(range(2, min(ell_max, 4) + 1))

    try:
        requested_modes = tuple(mode_array)
    except TypeError as exc:
        raise ValueError("mode_array must contain (ell, m) pairs") from exc
    if not requested_modes:
        raise ValueError("mode_array must not be empty")

    ells = set()
    for mode in requested_modes:
        if isinstance(mode, (str, bytes)):
            raise ValueError("mode_array must contain (ell, m) pairs")
        try:
            mode = tuple(mode)
        except TypeError as exc:
            raise ValueError("mode_array must contain (ell, m) pairs") from exc
        if len(mode) != 2:
            raise ValueError("mode_array must contain (ell, m) pairs")
        try:
            ell, emm = (operator.index(value) for value in mode)
        except TypeError as exc:
            raise ValueError("mode_array entries must be integers") from exc
        if ell < 2 or ell > 4 or abs(emm) > ell:
            raise ValueError(f"unsupported SpinTaylor mode ({ell}, {emm})")
        ells.add(ell)
    return tuple(sorted(ells))


def _spintaylor_modes_native_supported(parameters, approximant):
    polarization_parameters = dict(parameters)
    polarization_parameters["mode_array"] = None
    if not _spintaylor_native_supported(polarization_parameters, approximant):
        return False
    try:
        _spintaylor_mode_ells(parameters)
    except (TypeError, ValueError):
        return False
    return True


def spintaylor_t1_modes_native_supported(parameters):
    """Return whether the native SpinTaylorT1 mode port covers parameters."""

    return _spintaylor_modes_native_supported(parameters, "SpinTaylorT1")


def spintaylor_t4_modes_native_supported(parameters):
    """Return whether the native SpinTaylorT4 mode port covers parameters."""

    return _spintaylor_modes_native_supported(parameters, "SpinTaylorT4")


def spintaylor_t5_modes_native_supported(parameters):
    """Return whether the native SpinTaylorT5 mode port covers parameters."""

    return _spintaylor_modes_native_supported(parameters, "SpinTaylorT5")


def _rotate_z(vector, angle):
    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        x * cosine - y * sine,
        x * sine + y * cosine,
        z,
    )


def _rotate_y(vector, angle):
    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        x * cosine + z * sine,
        y,
        -x * sine + z * cosine,
    )


def _orbital_l_spin_to_radiation_frame(spin, inclination, coa_phase):
    """Apply LAL's default Orbital-L initial-spin convention."""

    spin = _rotate_z(spin, coa_phase - math.pi / 2.0)
    spin = _rotate_y(spin, -inclination)
    return _rotate_z(spin, math.pi)


def _parameter_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"SpinTaylor requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"SpinTaylor requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"SpinTaylor requires finite {name}")
    return value


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native SpinTaylor requires an active TorchScheme")
    device = state.torch_device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return device, dtype


def _spintaylor_td_torch(native_approximant, **parameters):
    """Generate public-convention SpinTaylor polarizations with Torch."""

    mass1 = _parameter_float(parameters, "mass1")
    mass2 = _parameter_float(parameters, "mass2")
    distance = _parameter_float(parameters, "distance", 1.0)
    inclination = _parameter_float(parameters, "inclination", 0.0)
    coa_phase = _parameter_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _parameter_float(parameters, "long_asc_nodes", 0.0)
    delta_t = _parameter_float(parameters, "delta_t")
    f_lower = _parameter_float(parameters, "f_lower")
    f_ref = _parameter_float(parameters, "f_ref", 0.0)
    effective_f_ref = f_lower if f_ref == 0.0 else f_ref
    spin1 = tuple(_parameter_float(parameters, f"spin1{axis}", 0.0) for axis in "xyz")
    spin2 = tuple(_parameter_float(parameters, f"spin2{axis}", 0.0) for axis in "xyz")
    spin1 = _orbital_l_spin_to_radiation_frame(spin1, inclination, coa_phase)
    spin2 = _orbital_l_spin_to_radiation_frame(spin2, inclination, coa_phase)
    lambda1 = (
        0.0
        if parameters.get("lambda1") is None
        else _parameter_float(parameters, "lambda1")
    )
    lambda2 = (
        0.0
        if parameters.get("lambda2") is None
        else _parameter_float(parameters, "lambda2")
    )
    tidal_order = parameters.get("tidal_order", -1)
    quadrupole1 = 1.0 + (
        0.0
        if parameters.get("dquad_mon1") is None
        else _parameter_float(parameters, "dquad_mon1")
    )
    quadrupole2 = 1.0 + (
        0.0
        if parameters.get("dquad_mon2") is None
        else _parameter_float(parameters, "dquad_mon2")
    )
    device, dtype = _target_device_dtype()
    lnhat = (math.sin(inclination), 0.0, math.cos(inclination))
    orbit = _spintaylor_orbit(
        native_approximant,
        mass1,
        mass2,
        delta_t,
        f_lower,
        spin1,
        spin2,
        coa_phase=coa_phase,
        f_ref=effective_f_ref,
        lnhat=lnhat,
        e1=(0.0, 1.0, 0.0),
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
        device=device,
        dtype=dtype,
    )
    plus, cross = spintaylor_polarizations_from_orbit(
        orbit.velocity,
        orbit.phase,
        orbit.spin1,
        orbit.spin2,
        orbit.lnhat,
        orbit.e1,
        mass1,
        mass2,
        distance,
        amplitude_order=parameters.get("amplitude_order", -1),
    )

    # The legacy public interface adds pi/2 before applying the standard
    # longitude-of-ascending-nodes polarization rotation.
    polarization = long_asc_nodes + math.pi / 2.0
    cosine = math.cos(2.0 * polarization)
    sine = math.sin(2.0 * polarization)
    original_plus, original_cross = plus, cross
    plus = cosine * original_plus + sine * original_cross
    cross = cosine * original_cross - sine * original_plus

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


def spintaylor_t1_td_torch(**parameters):
    """Generate public-convention SpinTaylorT1 polarizations with Torch."""

    return _spintaylor_td_torch("SpinTaylorT1", **parameters)


def spintaylor_t4_td_torch(**parameters):
    """Generate public-convention SpinTaylorT4 polarizations with Torch."""

    return _spintaylor_td_torch("SpinTaylorT4", **parameters)


def spintaylor_t5_td_torch(**parameters):
    """Generate public-convention SpinTaylorT5 polarizations with Torch."""

    return _spintaylor_td_torch("SpinTaylorT5", **parameters)


def _spintaylor_modes_torch(native_approximant, **parameters):
    """Generate precessing SpinTaylor modes on the Torch device.

    The legacy LAL mode interface ignores coalescence phase, inclination, and
    longitude of ascending nodes. A requested ``m`` selects every mode for
    that degree, matching its mode-array behavior.
    """

    mass1 = _parameter_float(parameters, "mass1")
    mass2 = _parameter_float(parameters, "mass2")
    distance = _parameter_float(parameters, "distance", 1.0)
    delta_t = _parameter_float(parameters, "delta_t")
    f_lower = _parameter_float(parameters, "f_lower")
    f_ref = _parameter_float(parameters, "f_ref", 0.0)
    spin1 = tuple(_parameter_float(parameters, f"spin1{axis}", 0.0) for axis in "xyz")
    spin2 = tuple(_parameter_float(parameters, f"spin2{axis}", 0.0) for axis in "xyz")
    lambda1 = (
        0.0
        if parameters.get("lambda1") is None
        else _parameter_float(parameters, "lambda1")
    )
    lambda2 = (
        0.0
        if parameters.get("lambda2") is None
        else _parameter_float(parameters, "lambda2")
    )
    quadrupole1 = 1.0 + (
        0.0
        if parameters.get("dquad_mon1") is None
        else _parameter_float(parameters, "dquad_mon1")
    )
    quadrupole2 = 1.0 + (
        0.0
        if parameters.get("dquad_mon2") is None
        else _parameter_float(parameters, "dquad_mon2")
    )
    ells = _spintaylor_mode_ells(parameters)
    device, dtype = _target_device_dtype()

    orbit = _spintaylor_orbit(
        native_approximant,
        mass1,
        mass2,
        delta_t,
        f_lower,
        spin1,
        spin2,
        coa_phase=0.0,
        f_ref=f_ref,
        lnhat=(0.0, 0.0, 1.0),
        e1=(1.0, 0.0, 0.0),
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=parameters.get("tidal_order", -1),
        device=device,
        dtype=dtype,
    )
    modes = spintaylor_modes_from_orbit(
        orbit.velocity,
        orbit.phase,
        orbit.spin1,
        orbit.spin2,
        orbit.lnhat,
        orbit.e1,
        mass1,
        mass2,
        distance,
        amplitude_order=parameters.get("amplitude_order", -1),
        ells=ells,
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


def spintaylor_t1_modes_torch(**parameters):
    """Generate precessing SpinTaylorT1 modes on the Torch device."""

    return _spintaylor_modes_torch("SpinTaylorT1", **parameters)


def spintaylor_t4_modes_torch(**parameters):
    """Generate precessing SpinTaylorT4 modes on the Torch device."""

    return _spintaylor_modes_torch("SpinTaylorT4", **parameters)


def spintaylor_t5_modes_torch(**parameters):
    """Generate precessing SpinTaylorT5 modes on the Torch device."""

    return _spintaylor_modes_torch("SpinTaylorT5", **parameters)
