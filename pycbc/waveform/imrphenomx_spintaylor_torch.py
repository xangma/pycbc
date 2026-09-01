# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native SpinTaylor dynamics used by public and IMRPhenomX models.

The numerical Euler angles in IMRPhenomX precession version 330 use the
orbit-averaged, two-spin SpinTaylorT4 equations; the public waveform paths also
use the unexpanded SpinTaylorT1 energy/flux ratio and inverse-series
SpinTaylorT5 dynamics. This module contains their default right-hand sides,
frequency- and uniform-time adaptive integrators, and the source-to-J-frame
Euler-angle construction. The integrators evolve outward from the reference
frequency in both directions, matching LAL's reference-frequency semantics
while keeping state and angle evaluation on the active Torch device.

Spin vectors follow LAL's internal convention: they are divided by the square
of the total mass.  Consequently, a dimensionless component spin ``chi_i`` is
passed to :func:`spintaylor_internal_spins` before evaluating the derivatives.
All derivatives are with respect to dimensionless time ``t / M``.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)
from pycbc.waveform.constants import _EULER_GAMMA


@dataclass(frozen=True)
class SpinTaylorVectorDerivatives:
    """Vector part of a SpinTaylor state derivative."""

    lnhat: torch.Tensor
    spin1: torch.Tensor
    spin2: torch.Tensor
    e1: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorOrbitalDerivatives:
    """Orbital part of a SpinTaylor state derivative."""

    phase: torch.Tensor
    omega: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorTrajectory:
    """SpinTaylor state sampled at geometric GW frequencies ``M f``.

    ``state`` follows the same 14-component ordering and internal spin
    convention as :func:`spintaylor_t4_rhs`.  Its first dimension corresponds
    to ``mf``.
    """

    mf: torch.Tensor
    state: torch.Tensor

    @property
    def phase(self):
        return self.state[..., 0]

    @property
    def omega(self):
        return self.state[..., 1]

    @property
    def lnhat(self):
        return self.state[..., 2:5]

    @property
    def spin1(self):
        return self.state[..., 5:8]

    @property
    def spin2(self):
        return self.state[..., 8:11]

    @property
    def e1(self):
        return self.state[..., 11:14]


@dataclass(frozen=True)
class SpinTaylorJFrame:
    """Source-to-J-frame rotation and reference-angle offsets."""

    phi_j_source: torch.Tensor
    theta_j_source: torch.Tensor
    kappa: torch.Tensor
    alpha0: torch.Tensor
    epsilon0: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorEulerAngles:
    """Numerical SpinTaylor Euler angles sampled at geometric frequencies."""

    mf: torch.Tensor
    alpha: torch.Tensor
    cosbeta: torch.Tensor
    gamma: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorAlphaMRD:
    """Coefficients of the PhenomX SpinTaylor alpha continuation."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorBetaMRD:
    """Coefficients of the PhenomX SpinTaylor beta continuation."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    offset: torch.Tensor
    damping_difference: torch.Tensor
    cosbeta_sign: torch.Tensor
    flat: torch.Tensor


@dataclass(frozen=True)
class SpinTaylorAngleSpline:
    """Numerical SpinTaylor angles and their generic IMR continuation.

    The inspiral angles are natural-cubic splines of the orbital direction
    rotated into the J frame.  ``fmax_inspiral`` is the final physical PN
    frequency selected by the caller; the generic merger-ringdown
    continuation starts at ``0.98 * fmax_inspiral`` as it does in LAL.
    """

    frame: SpinTaylorJFrame
    mf: torch.Tensor
    alpha: torch.Tensor
    cosbeta: torch.Tensor
    linear: torch.Tensor
    quadratic: torch.Tensor
    cubic: torch.Tensor
    fmax_inspiral: torch.Tensor
    ftrans_mrd: torch.Tensor
    alpha_mrd: SpinTaylorAlphaMRD
    beta_mrd: SpinTaylorBetaMRD


class _SpinTaylorPhysicalBoundary(RuntimeError):
    """A physical SpinTaylor stopping condition, rather than a solver failure."""


def _as_tensor_like(value, reference):
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _scale_vector(scale, vector):
    return scale.unsqueeze(-1) * vector


def _spin_dot_3pn(mass_fraction):
    return 1.5 - mass_fraction - 0.5 * mass_fraction**2


def _spin_dot_4pn_qm(mass_fraction):
    return 1.5 * (1.0 - 1.0 / mass_fraction)


def _spin_dot_5pn(mass_fraction):
    return (
        9.0 / 8.0
        - mass_fraction / 2.0
        + 7.0 * mass_fraction**2 / 12.0
        - 7.0 * mass_fraction**3 / 6.0
        - mass_fraction**4 / 24.0
    )


def _spin_dot_6pn_partner(mass_fraction):
    partner = -1.5 - mass_fraction
    partner_n = 1.5 + 2.0 * mass_fraction + mass_fraction**2
    partner_v = 1.5 + mass_fraction
    return partner + 0.5 * (partner_n + partner_v)


def _spin_dot_6pn_own_projection(mass_fraction):
    own_n = 3.5 - 3.0 / mass_fraction - 0.5 * mass_fraction**2
    own_v = 3.0 - 1.5 * mass_fraction - 1.5 / mass_fraction
    return -0.5 * (own_n + own_v)


def _spin_dot_6pn_partner_projection(mass_fraction):
    partner_n = 1.5 + 2.0 * mass_fraction + mass_fraction**2
    partner_v = 1.5 + mass_fraction
    return -0.5 * (partner_n + partner_v)


def _spin_dot_6pn_qm(mass_fraction):
    qm_n = 3.0 * (0.5 / mass_fraction + 1.0 - mass_fraction - 0.5 * mass_fraction**2)
    qm_v = 3.0 * (1.0 / mass_fraction - 1.0)
    return -0.5 * (qm_n + qm_v)


def _spin_dot_7pn(mass_fraction):
    """Historical 3.5PN spin-orbit coefficient used by PhenomTP."""

    return (
        mass_fraction**6 / 48.0
        - 3.0 / 8.0 * mass_fraction**5
        - 39.0 / 16.0 * mass_fraction**4
        - 23.0 / 6.0 * mass_fraction**3
        + 181.0 / 16.0 * mass_fraction**2
        - 51.0 / 8.0 * mass_fraction
        + 27.0 / 16.0
    )


def _omega_spin_25pn(mass_fraction):
    return (
        -809.0 / (84.0 * mass_fraction)
        + 13.795 / 1.008
        - 527.0 * mass_fraction / 24.0
        - 79.0 * mass_fraction**2 / 6.0
    )


def _omega_spin_spin_3pn(mass_fraction, quadrupole):
    inverse_mass = 1.0 / mass_fraction
    inverse_mass2 = inverse_mass**2
    self_spin = 101.9 / 6.4 * inverse_mass2 + 2.51 / 5.76 * inverse_mass + 13.33 / 5.76
    self_projection = (
        -49.3 / 6.4 * inverse_mass2 + 197.47 / 5.76 * inverse_mass + 56.45 / 5.76
    )
    quadrupole_spin = quadrupole * (
        -6.59 / 2.24 * inverse_mass2 + 7.3 / 4.8 * inverse_mass - 43.0 / 4.0
    )
    quadrupole_projection = quadrupole * (
        19.77 / 2.24 * inverse_mass2 - 7.3 / 1.6 * inverse_mass + 129.0 / 4.0
    )
    return self_spin + quadrupole_spin, self_projection + quadrupole_projection


def _omega_tidal_5pn(mass_fraction):
    return 6.0 * mass_fraction**4 * (12.0 - 11.0 * mass_fraction)


def _omega_tidal_6pn(mass_fraction):
    return mass_fraction**4 * (
        4421.0 / 56.0
        - 12263.0 / 56.0 * mass_fraction
        + 1893.0 / 4.0 * mass_fraction**2
        - 661.0 / 2.0 * mass_fraction**3
    )


_SPINTAYLOR_TIDAL_ORDERS = frozenset((-1, 0, 10, 12))


def _spintaylor_tidal_terms(tidal_order):
    """Return the LAL-compatible 5PN and 6PN tidal switches."""

    try:
        tidal_order = operator.index(tidal_order)
    except TypeError as exc:
        raise ValueError("SpinTaylor tidal_order must be an integer") from exc
    if tidal_order not in _SPINTAYLOR_TIDAL_ORDERS:
        raise ValueError("SpinTaylor tidal_order must be one of -1, 0, 10, or 12")
    return tidal_order in (-1, 10, 12), tidal_order in (-1, 12)


def spintaylor_internal_spins(mass1, mass2, chi1, chi2):
    """Convert component-normalized spins to the SpinTaylor state convention.

    ``chi1`` and ``chi2`` must have a final vector dimension of length three.
    Masses may be expressed in any common unit and may carry batch dimensions.
    """

    if not isinstance(chi1, torch.Tensor):
        if isinstance(chi2, torch.Tensor):
            chi1 = _as_tensor_like(chi1, chi2)
        else:
            chi1 = torch.as_tensor(chi1, dtype=torch.float64)
    chi2 = _as_tensor_like(chi2, chi1)
    mass1 = _as_tensor_like(mass1, chi1)
    mass2 = _as_tensor_like(mass2, chi1)

    total_mass = mass1 + mass2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    return (
        _scale_vector(mass1_fraction**2, chi1),
        _scale_vector(mass2_fraction**2, chi2),
    )


def _spintaylor_phase_derivative(
    v,
    lnhat_dot_spin1,
    lnhat_dot_spin2,
    spin1_dot_spin2,
    spin1_squared,
    spin2_squared,
    mass1_fraction,
    mass2_fraction,
):
    """Return LAL's shared spin-shifted orbital phase derivative."""

    omega = v**3
    omega_shift1 = 0.5 + 1.5 / mass1_fraction
    omega_shift2 = 0.5 + 1.5 / mass2_fraction
    phase_shift = -0.25 * (
        omega_shift1**2 * (spin1_squared - lnhat_dot_spin1**2)
        + omega_shift2**2 * (spin2_squared - lnhat_dot_spin2**2)
        + 2.0
        * omega_shift1
        * omega_shift2
        * (spin1_dot_spin2 - lnhat_dot_spin1 * lnhat_dot_spin2)
    )
    return omega * (1.0 + omega**2 * phase_shift)


def spintaylor_t4_orbital_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the default SpinTaylorT4 orbital-frequency and phase rates.

    The implementation matches the 3.5PN orbital, 3PN spin, and 6PN tidal
    configuration used by IMRPhenomX numerical precession version 330.  The
    returned derivatives use dimensionless time ``t / M`` and angular
    frequency ``M * omega``.
    """

    v = _as_tensor_like(v, lnhat)
    spin1 = _as_tensor_like(spin1, lnhat)
    spin2 = _as_tensor_like(spin2, lnhat)
    mass1_fraction = _as_tensor_like(mass1_fraction, lnhat)
    mass2_fraction = _as_tensor_like(mass2_fraction, lnhat)
    quadrupole1 = _as_tensor_like(quadrupole1, lnhat)
    quadrupole2 = _as_tensor_like(quadrupole2, lnhat)
    lambda1 = _as_tensor_like(lambda1, lnhat)
    lambda2 = _as_tensor_like(lambda2, lnhat)

    eta = mass1_fraction * mass2_fraction
    lnhat_dot_spin1 = torch.sum(lnhat * spin1, dim=-1)
    lnhat_dot_spin2 = torch.sum(lnhat * spin2, dim=-1)
    spin1_dot_spin2 = torch.sum(spin1 * spin2, dim=-1)
    spin1_squared = torch.sum(spin1 * spin1, dim=-1)
    spin2_squared = torch.sum(spin2 * spin2, dim=-1)

    spin_15pn = (-19.0 / 6.0 - 25.0 / (4.0 * mass1_fraction)) * lnhat_dot_spin1 + (
        -19.0 / 6.0 - 25.0 / (4.0 * mass2_fraction)
    ) * lnhat_dot_spin2

    spin_2pn = (
        -247.0 / (48.0 * eta) * spin1_dot_spin2
        + 721.0 / (48.0 * eta) * lnhat_dot_spin1 * lnhat_dot_spin2
    )
    for mass_fraction, quadrupole, spin_squared, projection in (
        (mass1_fraction, quadrupole1, spin1_squared, lnhat_dot_spin1),
        (mass2_fraction, quadrupole2, spin2_squared, lnhat_dot_spin2),
    ):
        inverse_mass2 = 1.0 / mass_fraction**2
        spin_2pn = spin_2pn + inverse_mass2 * (
            (7.0 / 96.0 - 2.5 * quadrupole) * spin_squared
            + (-1.0 / 96.0 + 7.5 * quadrupole) * projection**2
        )

    spin_25pn = (
        _omega_spin_25pn(mass1_fraction) * lnhat_dot_spin1
        + _omega_spin_25pn(mass2_fraction) * lnhat_dot_spin2
    )

    spin_3pn = (
        math.pi * (-37.0 / 3.0 - 151.0 / (6.0 * mass1_fraction)) * lnhat_dot_spin1
        + math.pi * (-37.0 / 3.0 - 151.0 / (6.0 * mass2_fraction)) * lnhat_dot_spin2
    )
    spin_3pn = (
        spin_3pn
        + (108.79 / (6.72 * eta) + 75.25 / 2.88) * spin1_dot_spin2
        + (162.25 / (2.24 * eta) - 129.31 / 2.88) * lnhat_dot_spin1 * lnhat_dot_spin2
    )
    for mass_fraction, quadrupole, spin_squared, projection in (
        (mass1_fraction, quadrupole1, spin1_squared, lnhat_dot_spin1),
        (mass2_fraction, quadrupole2, spin2_squared, lnhat_dot_spin2),
    ):
        spin_coefficient, projection_coefficient = _omega_spin_spin_3pn(
            mass_fraction, quadrupole
        )
        spin_3pn = (
            spin_3pn
            + spin_coefficient * spin_squared
            + projection_coefficient * projection**2
        )

    orbital_1pn = -(743.0 + 924.0 * eta) / 336.0
    orbital_2pn = (34103.0 + 122949.0 * eta + 59472.0 * eta**2) / 18144.0
    orbital_25pn = -math.pi * (4159.0 + 15876.0 * eta) / 672.0
    orbital_3pn = (
        16447.322263 / 139.7088
        - 1712.0 / 105.0 * _EULER_GAMMA
        - 561.98689 / 2.17728 * eta
        + math.pi**2 * (16.0 / 3.0 + 451.0 / 48.0 * eta)
        + 541.0 / 896.0 * eta**2
        - 5605.0 / 2592.0 * eta**3
        - 856.0 / 105.0 * math.log(16.0)
    )
    orbital_35pn = math.pi / 12096.0 * (-13245.0 + 717350.0 * eta + 731960.0 * eta**2)
    include_tidal_5pn, include_tidal_6pn = _spintaylor_tidal_terms(tidal_order)
    tidal_5pn = 0.0
    tidal_6pn = 0.0
    if include_tidal_5pn:
        tidal_5pn = lambda1 * _omega_tidal_5pn(
            mass1_fraction
        ) + lambda2 * _omega_tidal_5pn(mass2_fraction)
    if include_tidal_6pn:
        tidal_6pn = lambda1 * _omega_tidal_6pn(
            mass1_fraction
        ) + lambda2 * _omega_tidal_6pn(mass2_fraction)

    pn_series = (
        1.0
        + orbital_1pn * v**2
        + (4.0 * math.pi + spin_15pn) * v**3
        + (orbital_2pn + spin_2pn) * v**4
        + (orbital_25pn + spin_25pn) * v**5
        + (orbital_3pn + spin_3pn - 1712.0 / 105.0 * torch.log(v)) * v**6
        + orbital_35pn * v**7
        + tidal_5pn * v**10
        + tidal_6pn * v**12
    )
    omega_derivative = 96.0 / 5.0 * eta * v**11 * pn_series
    phase_derivative = _spintaylor_phase_derivative(
        v,
        lnhat_dot_spin1,
        lnhat_dot_spin2,
        spin1_dot_spin2,
        spin1_squared,
        spin2_squared,
        mass1_fraction,
        mass2_fraction,
    )

    return SpinTaylorOrbitalDerivatives(
        phase=phase_derivative,
        omega=omega_derivative,
    )


def spintaylor_t1_orbital_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the default SpinTaylorT1 orbital-frequency and phase rates.

    TaylorT1 leaves the PN energy derivative and gravitational-wave flux as
    an unexpanded ratio. This implements LAL's 3.5PN orbital, 3PN spin, and
    6PN tidal setup while preserving Torch device and batch dimensions.
    """

    v = _as_tensor_like(v, lnhat)
    spin1 = _as_tensor_like(spin1, lnhat)
    spin2 = _as_tensor_like(spin2, lnhat)
    mass1_fraction = _as_tensor_like(mass1_fraction, lnhat)
    mass2_fraction = _as_tensor_like(mass2_fraction, lnhat)
    quadrupole1 = _as_tensor_like(quadrupole1, lnhat)
    quadrupole2 = _as_tensor_like(quadrupole2, lnhat)
    lambda1 = _as_tensor_like(lambda1, lnhat)
    lambda2 = _as_tensor_like(lambda2, lnhat)

    eta = mass1_fraction * mass2_fraction
    projection1 = torch.sum(lnhat * spin1, dim=-1)
    projection2 = torch.sum(lnhat * spin2, dim=-1)
    spin_product = torch.sum(spin1 * spin2, dim=-1)
    spin1_squared = torch.sum(spin1 * spin1, dim=-1)
    spin2_squared = torch.sum(spin2 * spin2, dim=-1)

    spin_3pn = (-1.5 - 1.25 / mass1_fraction) * projection1 + (
        -1.5 - 1.25 / mass2_fraction
    ) * projection2
    spin_4pn = (
        -103.0 / (48.0 * eta) * spin_product
        + 289.0 / (48.0 * eta) * projection1 * projection2
    )
    spin_5pn = 0.0
    spin_6pn = (2123.0 / (84.0 * eta) + 821.0 / 72.0) * spin_product + (
        -5647.0 / (168.0 * eta) - 2023.0 / 72.0
    ) * projection1 * projection2
    for mass_fraction, quadrupole, spin_squared, projection in (
        (mass1_fraction, quadrupole1, spin1_squared, projection1),
        (mass2_fraction, quadrupole2, spin2_squared, projection2),
    ):
        inverse_mass = 1.0 / mass_fraction
        inverse_mass2 = inverse_mass**2
        spin_4pn = spin_4pn + (
            (7.0 / 96.0 - quadrupole) * inverse_mass2 * spin_squared
            + (-1.0 / 96.0 + 3.0 * quadrupole) * inverse_mass2 * projection**2
        )
        spin_5pn = (
            spin_5pn
            + (
                63.0 / 8.0
                - 13.0 / 16.0 * inverse_mass
                - 73.0 / 36.0 * mass_fraction
                - 157.0 / 18.0 * mass_fraction**2
            )
            * projection
        )
        spin_6pn = spin_6pn + (
            (
                189.0 / 16.0 * inverse_mass2
                - 35.0 / 144.0 * inverse_mass
                + 47.0 / 144.0
                + quadrupole
                * (
                    279.0 / 112.0 * inverse_mass2
                    + 45.0 / 16.0 * inverse_mass
                    - 43.0 / 8.0
                )
            )
            * spin_squared
            + (
                -239.0 / 16.0 * inverse_mass2
                + 293.0 / 144.0 * inverse_mass
                + 299.0 / 144.0
                + quadrupole
                * (
                    -837.0 / 112.0 * inverse_mass2
                    - 135.0 / 16.0 * inverse_mass
                    + 129.0 / 80.0
                )
            )
            * projection**2
        )

    flux_2pn = -(1247.0 / 336.0 + 35.0 / 12.0 * eta)
    flux_4pn = -(44711.0 / 9072.0 - 9271.0 / 504.0 * eta - 65.0 / 18.0 * eta**2)
    flux_5pn = -(8191.0 / 672.0 + 583.0 / 24.0 * eta) * math.pi
    flux_6pn = (
        6643739519.0 / 69854400.0
        + 16.0 / 3.0 * math.pi**2
        - 1712.0 / 105.0 * (_EULER_GAMMA + math.log(4.0))
        + (41.0 / 48.0 * math.pi**2 - 134543.0 / 7776.0) * eta
        - 94403.0 / 3024.0 * eta**2
        - 775.0 / 324.0 * eta**3
    )
    flux_7pn = (
        -(16285.0 / 504.0 - 214745.0 / 1728.0 * eta - 193385.0 / 3024.0 * eta**2)
        * math.pi
    )

    include_tidal_5pn, include_tidal_6pn = _spintaylor_tidal_terms(tidal_order)
    flux_10pn = 0.0
    flux_12pn = 0.0
    if include_tidal_5pn:
        flux_10pn = 6.0 * (
            lambda1 * (3.0 - 2.0 * mass1_fraction) * mass1_fraction**4
            + lambda2 * (3.0 - 2.0 * mass2_fraction) * mass2_fraction**4
        )
    if include_tidal_6pn:
        flux_12pn = lambda1 * mass1_fraction**4 * (
            -176.0 / 7.0
            - 1803.0 / 28.0 * mass1_fraction
            + 643.0 / 4.0 * mass1_fraction**2
            - 155.0 / 2.0 * mass1_fraction**3
        ) + lambda2 * mass2_fraction**4 * (
            -176.0 / 7.0
            - 1803.0 / 28.0 * mass2_fraction
            + 643.0 / 4.0 * mass2_fraction**2
            - 155.0 / 2.0 * mass2_fraction**3
        )

    flux_factor = (
        1.0
        + flux_2pn * v**2
        + (4.0 * math.pi + spin_3pn) * v**3
        + (flux_4pn + spin_4pn) * v**4
        + (flux_5pn + spin_5pn) * v**5
        + (flux_6pn + spin_6pn - 1712.0 / 105.0 * torch.log(v)) * v**6
        + flux_7pn * v**7
        + flux_10pn * v**10
        + flux_12pn * v**12
    )
    energy_derivative_factor = spintaylor_t4_energy_derivative_factor(
        v,
        lnhat,
        spin1,
        spin2,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )
    omega_derivative = (
        192.0 / 5.0 * eta * v**11 * flux_factor / energy_derivative_factor
    )
    phase_derivative = _spintaylor_phase_derivative(
        v,
        projection1,
        projection2,
        spin_product,
        spin1_squared,
        spin2_squared,
        mass1_fraction,
        mass2_fraction,
    )
    return SpinTaylorOrbitalDerivatives(
        phase=phase_derivative,
        omega=omega_derivative,
    )


def spintaylor_t5_orbital_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the default SpinTaylorT5 orbital-frequency and phase rates.

    TaylorT5 inverts the TaylorT2 ``dt/dv`` PN series without re-expanding
    the reciprocal. This implements LAL's 3.5PN orbital, 3PN spin, and 6PN
    tidal setup while preserving Torch device and batch dimensions.
    """

    v = _as_tensor_like(v, lnhat)
    spin1 = _as_tensor_like(spin1, lnhat)
    spin2 = _as_tensor_like(spin2, lnhat)
    mass1_fraction = _as_tensor_like(mass1_fraction, lnhat)
    mass2_fraction = _as_tensor_like(mass2_fraction, lnhat)
    quadrupole1 = _as_tensor_like(quadrupole1, lnhat)
    quadrupole2 = _as_tensor_like(quadrupole2, lnhat)
    lambda1 = _as_tensor_like(lambda1, lnhat)
    lambda2 = _as_tensor_like(lambda2, lnhat)

    eta = mass1_fraction * mass2_fraction
    projection1 = torch.sum(lnhat * spin1, dim=-1)
    projection2 = torch.sum(lnhat * spin2, dim=-1)
    spin_product = torch.sum(spin1 * spin2, dim=-1)
    spin1_squared = torch.sum(spin1 * spin1, dim=-1)
    spin2_squared = torch.sum(spin2 * spin2, dim=-1)

    spin_3pn = (19.0 / 6.0 + 25.0 / (4.0 * mass1_fraction)) * projection1 + (
        19.0 / 6.0 + 25.0 / (4.0 * mass2_fraction)
    ) * projection2
    spin_4pn = (
        247.0 / (48.0 * eta) * spin_product
        - 721.0 / (48.0 * eta) * projection1 * projection2
    )
    spin_5pn = 0.0
    spin_6pn = (52973.0 / (8064.0 * eta) + 313.0 / 144.0) * spin_product + (
        -170603.0 / (8064.0 * eta) - 2543.0 / 144.0
    ) * projection1 * projection2
    for mass_fraction, quadrupole, spin_squared, projection in (
        (mass1_fraction, quadrupole1, spin1_squared, projection1),
        (mass2_fraction, quadrupole2, spin2_squared, projection2),
    ):
        inverse_mass = 1.0 / mass_fraction
        inverse_mass2 = inverse_mass**2
        spin_4pn = spin_4pn + inverse_mass2 * (
            (-7.0 / 96.0 + 2.5 * quadrupole) * spin_squared
            + (1.0 / 96.0 - 7.5 * quadrupole) * projection**2
        )
        spin_5pn = (
            spin_5pn
            + (
                -17.0 / 4.0 * mass_fraction**2
                + 5.0 * mass_fraction
                + 1249.0 / 36.0
                + 8349.0 / 224.0 * inverse_mass
            )
            * projection
        )
        spin_6pn = (
            spin_6pn + math.pi * (-13.0 - 149.0 / 6.0 * inverse_mass) * projection
        )
        spin_6pn = spin_6pn + (
            (
                -37427.0 / 2304.0 * inverse_mass2
                - 241.0 / 288.0 * inverse_mass
                - 551.0 / 288.0
                + quadrupole
                * (9407.0 / 672.0 * inverse_mass2 + 587.0 / 48.0 * inverse_mass - 3.0)
            )
            * spin_squared
            + (
                754979.0 / 16128.0 * inverse_mass2
                + 1543.0 / 288.0 * inverse_mass
                + 49.0 / 288.0
                + quadrupole
                * (-9407.0 / 224.0 * inverse_mass2 - 587.0 / 16.0 * inverse_mass + 9.0)
            )
            * projection**2
        )

    orbital_2pn = 743.0 / 336.0 + 11.0 / 4.0 * eta
    orbital_3pn = -4.0 * math.pi
    orbital_4pn = 3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta**2
    orbital_5pn = (-7729.0 / 672.0 + 13.0 / 8.0 * eta) * math.pi
    orbital_6pn = (
        -10817850546611.0 / 93884313600.0
        + 32.0 / 3.0 * math.pi**2
        + 1712.0 / 105.0 * _EULER_GAMMA
        + (3147553127.0 / 12192768.0 - 451.0 / 48.0 * math.pi**2) * eta
        - 15211.0 / 6912.0 * eta**2
        + 25565.0 / 5184.0 * eta**3
        + 856.0 / 105.0 * math.log(16.0)
    )
    orbital_7pn = math.pi * (
        -15419335.0 / 1016064.0 - 75703.0 / 6048.0 * eta + 14809.0 / 3024.0 * eta**2
    )

    include_tidal_5pn, include_tidal_6pn = _spintaylor_tidal_terms(tidal_order)
    tidal_10pn = 0.0
    tidal_12pn = 0.0
    if include_tidal_5pn:
        tidal_10pn = 6.0 * (
            lambda1 * mass1_fraction**4 * (-12.0 + 11.0 * mass1_fraction)
            + lambda2 * mass2_fraction**4 * (-12.0 + 11.0 * mass2_fraction)
        )
    if include_tidal_6pn:
        tidal_12pn = lambda1 * mass1_fraction**4 * (
            -3179.0 / 8.0
            + 919.0 / 8.0 * mass1_fraction
            + 1143.0 / 4.0 * mass1_fraction**2
            - 65.0 / 2.0 * mass1_fraction**3
        ) + lambda2 * mass2_fraction**4 * (
            -3179.0 / 8.0
            + 919.0 / 8.0 * mass2_fraction
            + 1143.0 / 4.0 * mass2_fraction**2
            - 65.0 / 2.0 * mass2_fraction**3
        )

    inverse_series = (
        1.0
        + orbital_2pn * v**2
        + (orbital_3pn + spin_3pn) * v**3
        + (orbital_4pn + spin_4pn) * v**4
        + (orbital_5pn + spin_5pn) * v**5
        + (orbital_6pn + spin_6pn + 1712.0 / 105.0 * torch.log(v)) * v**6
        + orbital_7pn * v**7
        + tidal_10pn * v**10
        + tidal_12pn * v**12
    )
    omega_derivative = 96.0 / 5.0 * eta * v**11 / inverse_series
    phase_derivative = _spintaylor_phase_derivative(
        v,
        projection1,
        projection2,
        spin_product,
        spin1_squared,
        spin2_squared,
        mass1_fraction,
        mass2_fraction,
    )
    return SpinTaylorOrbitalDerivatives(
        phase=phase_derivative,
        omega=omega_derivative,
    )


def spintaylor_t4_energy_derivative_factor(
    v,
    lnhat,
    spin1,
    spin2,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Return LAL's dimensionless SpinTaylor orbital-energy stop test.

    The PN inspiral remains physical while this quantity is nonnegative.
    This is the complete default 3PN-spin/6PN-tidal expression used by
    numerical IMRPhenomX precession.
    """

    v = _as_tensor_like(v, lnhat)
    spin1 = _as_tensor_like(spin1, lnhat)
    spin2 = _as_tensor_like(spin2, lnhat)
    mass1_fraction = _as_tensor_like(mass1_fraction, lnhat)
    mass2_fraction = _as_tensor_like(mass2_fraction, lnhat)
    quadrupole1 = _as_tensor_like(quadrupole1, lnhat)
    quadrupole2 = _as_tensor_like(quadrupole2, lnhat)
    lambda1 = _as_tensor_like(lambda1, lnhat)
    lambda2 = _as_tensor_like(lambda2, lnhat)

    eta = mass1_fraction * mass2_fraction
    projection1 = torch.sum(lnhat * spin1, dim=-1)
    projection2 = torch.sum(lnhat * spin2, dim=-1)
    spin1_squared = torch.sum(spin1 * spin1, dim=-1)
    spin2_squared = torch.sum(spin2 * spin2, dim=-1)
    spin_product = torch.sum(spin1 * spin2, dim=-1)

    energy_2pn = -(0.75 + eta / 12.0)
    energy_4pn = -(27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0)
    energy_6pn = -(
        675.0 / 64.0
        - (34445.0 / 576.0 - 205.0 / 96.0 * math.pi**2) * eta
        + 155.0 / 96.0 * eta**2
        + 35.0 / 5184.0 * eta**3
    )
    spin_3pn = (2.0 / 3.0 + 2.0 / mass1_fraction) * projection1 + (
        2.0 / 3.0 + 2.0 / mass2_fraction
    ) * projection2
    spin_4pn = spin_product / eta - 3.0 / eta * projection1 * projection2
    spin_5pn = (
        5.0 / 3.0
        + 3.0 / mass1_fraction
        + 29.0 / 9.0 * mass1_fraction
        + mass1_fraction**2 / 9.0
    ) * projection1 + (
        5.0 / 3.0
        + 3.0 / mass2_fraction
        + 29.0 / 9.0 * mass2_fraction
        + mass2_fraction**2 / 9.0
    ) * projection2
    spin_6pn = (2.0 / eta - 11.0 / 6.0) * spin_product + (
        -11.0 / (3.0 * eta) + 23.0 / 18.0
    ) * projection1 * projection2

    for mass_fraction, quadrupole, spin_squared, projection in (
        (mass1_fraction, quadrupole1, spin1_squared, projection1),
        (mass2_fraction, quadrupole2, spin2_squared, projection2),
    ):
        inverse_mass = 1.0 / mass_fraction
        inverse_mass2 = inverse_mass**2
        spin_4pn = spin_4pn + quadrupole * inverse_mass2 * (
            0.5 * spin_squared - 1.5 * projection**2
        )
        spin_6pn = (
            spin_6pn
            + (-inverse_mass2 - inverse_mass / 6.0 - 0.5) * spin_squared
            + (6.0 * inverse_mass2 - 1.5 * inverse_mass - 11.0 / 18.0) * projection**2
        )
        spin_6pn = (
            spin_6pn
            + quadrupole
            * (1.25 * inverse_mass2 + 1.25 * inverse_mass + 5.0 / 12.0)
            * spin_squared
            + quadrupole
            * (-3.75 * inverse_mass2 - 3.75 * inverse_mass - 1.25)
            * projection**2
        )

    include_tidal_5pn, include_tidal_6pn = _spintaylor_tidal_terms(tidal_order)
    tidal_10pn = 0.0
    tidal_12pn = 0.0
    if include_tidal_5pn:
        tidal_10pn = -9.0 * (
            lambda1 * mass1_fraction**4 * (1.0 - mass1_fraction)
            + lambda2 * mass2_fraction**4 * (1.0 - mass2_fraction)
        )
    if include_tidal_6pn:
        tidal_12pn = lambda1 * mass1_fraction**4 * (
            -33.0 / 2.0
            + 11.0 / 2.0 * mass1_fraction
            - 11.0 / 2.0 * mass1_fraction**2
            + 33.0 / 2.0 * mass1_fraction**3
        ) + lambda2 * mass2_fraction**4 * (
            -33.0 / 2.0
            + 11.0 / 2.0 * mass2_fraction
            - 11.0 / 2.0 * mass2_fraction**2
            + 33.0 / 2.0 * mass2_fraction**3
        )
    return (
        2.0
        + 4.0 * energy_2pn * v**2
        + 5.0 * spin_3pn * v**3
        + 6.0 * (energy_4pn + spin_4pn) * v**4
        + 7.0 * spin_5pn * v**5
        + 8.0 * (energy_6pn + spin_6pn) * v**6
        + 12.0 * tidal_10pn * v**10
        + 14.0 * tidal_12pn * v**12
    )


def _spintaylor_t4_vector_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    e1,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    phenomtp=False,
):
    """Shared implementation of the default and historical TP fields."""

    v = _as_tensor_like(v, lnhat)
    spin1 = _as_tensor_like(spin1, lnhat)
    spin2 = _as_tensor_like(spin2, lnhat)
    e1 = _as_tensor_like(e1, lnhat)
    mass1_fraction = _as_tensor_like(mass1_fraction, lnhat)
    mass2_fraction = _as_tensor_like(mass2_fraction, lnhat)
    quadrupole1 = _as_tensor_like(quadrupole1, lnhat)
    quadrupole2 = _as_tensor_like(quadrupole2, lnhat)

    lnhat_cross_spin1 = torch.linalg.cross(lnhat, spin1, dim=-1)
    lnhat_cross_spin2 = torch.linalg.cross(lnhat, spin2, dim=-1)
    spin1_cross_spin2 = torch.linalg.cross(spin1, spin2, dim=-1)
    lnhat_dot_spin1 = torch.sum(lnhat * spin1, dim=-1)
    lnhat_dot_spin2 = torch.sum(lnhat * spin2, dim=-1)

    # Leading 1.5PN spin-orbit terms.
    dspin1_leading = _scale_vector(
        _spin_dot_3pn(mass1_fraction) * v**5,
        lnhat_cross_spin1,
    )
    dspin2_leading = _scale_vector(
        _spin_dot_3pn(mass2_fraction) * v**5,
        lnhat_cross_spin2,
    )

    # Orbit-averaged 2PN spin-spin and quadrupole-monopole terms.
    dspin1_nlo = _scale_vector(v**6, -0.5 * spin1_cross_spin2) + _scale_vector(
        v**6
        * (
            -1.5 * lnhat_dot_spin2
            + quadrupole1 * _spin_dot_4pn_qm(mass1_fraction) * lnhat_dot_spin1
        ),
        lnhat_cross_spin1,
    )
    dspin2_nlo = _scale_vector(v**6, 0.5 * spin1_cross_spin2) + _scale_vector(
        v**6
        * (
            -1.5 * lnhat_dot_spin1
            + quadrupole2 * _spin_dot_4pn_qm(mass2_fraction) * lnhat_dot_spin2
        ),
        lnhat_cross_spin2,
    )

    # 2.5PN spin-orbit terms.
    dspin1_nnlo = _scale_vector(
        _spin_dot_5pn(mass1_fraction) * v**7,
        lnhat_cross_spin1,
    )
    dspin2_nnlo = _scale_vector(
        _spin_dot_5pn(mass2_fraction) * v**7,
        lnhat_cross_spin2,
    )

    if phenomtp:
        # PhenomTP's legacy setup deliberately skips the 3PN spin-spin terms
        # and retains the otherwise unsupported 3.5PN spin-orbit terms.
        dspin1_high = _scale_vector(
            _spin_dot_7pn(mass1_fraction) * v**9,
            lnhat_cross_spin1,
        )
        dspin2_high = _scale_vector(
            _spin_dot_7pn(mass2_fraction) * v**9,
            lnhat_cross_spin2,
        )
    else:
        # Orbit-averaged 3PN spin terms.  The body-2 labels are intentionally
        # exchanged: its "partner" is body 1, matching the LAL setup.
        dspin1_high = _scale_vector(
            -_spin_dot_6pn_partner(mass1_fraction) * v**8,
            spin1_cross_spin2,
        ) + _scale_vector(
            v**8
            * (
                _spin_dot_6pn_own_projection(mass1_fraction) * lnhat_dot_spin1
                + _spin_dot_6pn_partner_projection(mass1_fraction) * lnhat_dot_spin2
                + quadrupole1 * _spin_dot_6pn_qm(mass1_fraction) * lnhat_dot_spin1
            ),
            lnhat_cross_spin1,
        )
        dspin2_high = _scale_vector(
            _spin_dot_6pn_partner(mass2_fraction) * v**8,
            spin1_cross_spin2,
        ) + _scale_vector(
            v**8
            * (
                _spin_dot_6pn_partner_projection(mass2_fraction) * lnhat_dot_spin1
                + _spin_dot_6pn_own_projection(mass2_fraction) * lnhat_dot_spin2
                + quadrupole2 * _spin_dot_6pn_qm(mass2_fraction) * lnhat_dot_spin2
            ),
            lnhat_cross_spin2,
        )

    dspin1 = dspin1_leading + dspin1_nlo + dspin1_nnlo + dspin1_high
    dspin2 = dspin2_leading + dspin2_nlo + dspin2_nnlo + dspin2_high

    eta = mass1_fraction * mass2_fraction
    newtonian_lmag = eta / v
    lmag = newtonian_lmag * (1.0 + v**2 * (1.5 + eta / 6.0))
    if phenomtp:
        lmag = lmag + newtonian_lmag * v**4 * (
            27.0 / 8.0 - 19.0 / 8.0 * eta + eta**2 / 24.0
        )
    raw_dlnhat = _scale_vector(-1.0 / lmag, dspin1 + dspin2)
    precession = torch.linalg.cross(lnhat, raw_dlnhat, dim=-1)

    return SpinTaylorVectorDerivatives(
        lnhat=torch.linalg.cross(precession, lnhat, dim=-1),
        spin1=dspin1,
        spin2=dspin2,
        e1=torch.linalg.cross(precession, e1, dim=-1),
    )


def spintaylor_t4_vector_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    e1,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
):
    """Evaluate the default orbit-averaged SpinTaylorT4 vector field.

    This is the vector part of LAL's
    ``XLALSimInspiralSpinTaylorT4DerivativesAvg`` for spin order 6,
    ``lscorr=0``, and the ordinary (non-PhenomT) setup used by XPNR.  Inputs
    may be batched, remain on their Torch device, and must use the internal
    total-mass-normalized spin convention.
    """

    return _spintaylor_t4_vector_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        e1,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
    )


def imrphenomtp_spintaylor_vector_derivatives(
    v,
    lnhat,
    spin1,
    spin2,
    e1,
    mass1_fraction,
    mass2_fraction,
):
    """Evaluate the historical orbit-averaged vector field used by PhenomTP.

    PhenomTP passes zero quadrupole parameters, omits the usual 3PN
    spin-spin terms, retains 3.5PN spin-orbit terms, and includes the 2PN
    non-spinning orbital-angular-momentum correction.  These conventions are
    intentional and differ from the default SpinTaylorT4 setup.
    """

    return _spintaylor_t4_vector_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        e1,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=0.0,
        quadrupole2=0.0,
        phenomtp=True,
    )


def _spintaylor_rhs(
    state,
    mass1_fraction,
    mass2_fraction,
    orbital_derivatives,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.float64)
    if state.ndim == 0 or state.shape[-1] != 14:
        raise ValueError("SpinTaylor state must have a final dimension of 14")

    omega = state[..., 1]
    v = torch.pow(omega, 1.0 / 3.0)
    lnhat = state[..., 2:5]
    spin1 = state[..., 5:8]
    spin2 = state[..., 8:11]
    e1 = state[..., 11:14]

    orbital = orbital_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )
    vectors = spintaylor_t4_vector_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        e1,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
    )
    return torch.cat(
        (
            orbital.phase.unsqueeze(-1),
            orbital.omega.unsqueeze(-1),
            vectors.lnhat,
            vectors.spin1,
            vectors.spin2,
            vectors.e1,
        ),
        dim=-1,
    )


def spintaylor_t1_rhs(
    state,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the complete default SpinTaylorT1 right-hand side."""

    return _spintaylor_rhs(
        state,
        mass1_fraction,
        mass2_fraction,
        spintaylor_t1_orbital_derivatives,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )


def spintaylor_t4_rhs(
    state,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the complete default SpinTaylorT4 right-hand side.

    The final state dimension follows LAL's ordering: phase, orbital angular
    frequency, ``LNh``, the two internal spin vectors, and the orbital-plane
    basis vector ``E1``. Leading batch dimensions are preserved.
    """

    return _spintaylor_rhs(
        state,
        mass1_fraction,
        mass2_fraction,
        spintaylor_t4_orbital_derivatives,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )


def spintaylor_t5_rhs(
    state,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
):
    """Evaluate the complete default SpinTaylorT5 right-hand side."""

    return _spintaylor_rhs(
        state,
        mass1_fraction,
        mass2_fraction,
        spintaylor_t5_orbital_derivatives,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )


def _check_spintaylor_physical_state(
    state,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1,
    quadrupole2,
    lambda1,
    lambda2,
    tidal_order=-1,
    derivatives=None,
):
    """Raise when a state reaches a physical SpinTaylor stopping condition."""

    omega = state[1]
    if (
        not bool(torch.isfinite(omega).detach().cpu())
        or float(omega.detach().cpu()) <= 0.0
    ):
        raise RuntimeError("SpinTaylor orbital frequency left its positive domain")
    v = torch.pow(omega, 1.0 / 3.0)
    if float(v.detach().cpu()) >= 1.0:
        raise _SpinTaylorPhysicalBoundary("SpinTaylor PN velocity reached unity")
    energy_test = spintaylor_t4_energy_derivative_factor(
        v,
        state[2:5],
        state[5:8],
        state[8:11],
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )
    if (
        not bool(torch.isfinite(energy_test).detach().cpu())
        or float(energy_test.detach().cpu()) < 0.0
    ):
        raise _SpinTaylorPhysicalBoundary(
            "SpinTaylor reached the PN orbital-energy boundary"
        )

    if derivatives is None:
        return
    omega_rate = derivatives[1]
    if (
        not bool(torch.isfinite(omega_rate).detach().cpu())
        or float(omega_rate.detach().cpu()) <= 0.0
    ):
        raise _SpinTaylorPhysicalBoundary(
            "SpinTaylor orbital frequency is not increasing"
        )


def _spintaylor_frequency_rhs(
    state,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole1,
    quadrupole2,
    lambda1,
    lambda2,
    tidal_order=-1,
):
    """Change the independent variable of the T4 system from time to omega."""

    _check_spintaylor_physical_state(
        state,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )

    derivatives = spintaylor_t4_rhs(
        state,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
    )
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
    omega_rate = derivatives[1]
    frequency_derivatives = derivatives / omega_rate
    return torch.cat(
        (
            frequency_derivatives[:1],
            torch.ones_like(frequency_derivatives[1:2]),
            frequency_derivatives[2:],
        )
    )


def _dormand_prince_step_with_slopes(state, step, rhs):
    """Take one embedded Dormand--Prince 5(4) step and retain end slopes."""

    k1 = rhs(state)
    k2 = rhs(state + step * (1.0 / 5.0) * k1)
    k3 = rhs(state + step * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2))
    k4 = rhs(state + step * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3))
    k5 = rhs(
        state
        + step
        * (
            19372.0 / 6561.0 * k1
            - 25360.0 / 2187.0 * k2
            + 64448.0 / 6561.0 * k3
            - 212.0 / 729.0 * k4
        )
    )
    k6 = rhs(
        state
        + step
        * (
            9017.0 / 3168.0 * k1
            - 355.0 / 33.0 * k2
            + 46732.0 / 5247.0 * k3
            + 49.0 / 176.0 * k4
            - 5103.0 / 18656.0 * k5
        )
    )
    fifth_order = state + step * (
        35.0 / 384.0 * k1
        + 500.0 / 1113.0 * k3
        + 125.0 / 192.0 * k4
        - 2187.0 / 6784.0 * k5
        + 11.0 / 84.0 * k6
    )
    k7 = rhs(fifth_order)
    fourth_order = state + step * (
        5179.0 / 57600.0 * k1
        + 7571.0 / 16695.0 * k3
        + 393.0 / 640.0 * k4
        - 92097.0 / 339200.0 * k5
        + 187.0 / 2100.0 * k6
        + 1.0 / 40.0 * k7
    )
    return fifth_order, fifth_order - fourth_order, k1, k7


def _dormand_prince_step(state, step, rhs):
    """Take one embedded Dormand--Prince 5(4) step."""

    fifth_order, error, _, _ = _dormand_prince_step_with_slopes(state, step, rhs)
    return fifth_order, error


def _integrate_frequency_branch(
    state,
    target_omega,
    indices,
    outputs,
    rhs,
    *,
    rtol,
    atol,
    max_steps,
    truncate_at_boundary=False,
):
    """Integrate one monotonic branch away from the reference frequency."""

    if not indices:
        return
    # The caller supplies low-frequency indices in descending order and the
    # high-frequency indices in ascending order.  Infer the physical direction
    # directly so the one-element case remains unambiguous.
    first_distance = float((target_omega[indices[0]] - state[1]).detach().cpu())
    direction = 1.0 if first_distance > 0.0 else -1.0
    initial_scale = max(abs(float(state[1].detach().cpu())) * 0.05, 1.0e-6)
    step_size = direction * min(abs(first_distance), initial_scale)
    epsilon = torch.finfo(state.dtype).eps
    steps = 0

    for index in indices:
        target = target_omega[index]
        while True:
            distance = float((target - state[1]).detach().cpu())
            if direction * distance <= 16.0 * epsilon * max(
                1.0, abs(float(target.detach().cpu()))
            ):
                state = torch.cat((state[:1], target.reshape(1), state[2:]))
                outputs[index] = state
                break
            if steps >= max_steps:
                raise RuntimeError(
                    "SpinTaylorT4 integration exceeded the maximum step count"
                )
            steps += 1
            proposed_step = direction * min(abs(step_size), abs(distance))
            try:
                candidate, error = _dormand_prince_step(
                    state,
                    proposed_step,
                    rhs,
                )
            except _SpinTaylorPhysicalBoundary:
                if not truncate_at_boundary:
                    raise
                step_size = 0.5 * proposed_step
                minimum_step = (
                    16.0
                    * epsilon
                    * max(
                        1.0,
                        abs(float(state[1].detach().cpu())),
                    )
                )
                if abs(step_size) < minimum_step:
                    return index
                continue
            scale = atol + rtol * torch.maximum(torch.abs(state), torch.abs(candidate))
            error_ratio = float(torch.max(torch.abs(error) / scale).detach().cpu())

            if math.isfinite(error_ratio) and error_ratio <= 1.0:
                state = candidate
                factor = (
                    5.0
                    if error_ratio == 0.0
                    else min(5.0, max(0.2, 0.9 * error_ratio ** (-0.2)))
                )
                step_size = proposed_step * factor
                continue

            factor = (
                0.2
                if not math.isfinite(error_ratio)
                else max(0.1, 0.9 * error_ratio ** (-0.25))
            )
            step_size = proposed_step * min(0.5, factor)
            minimum_step = (
                16.0 * epsilon * max(1.0, abs(float(state[1].detach().cpu())))
            )
            if abs(step_size) < minimum_step:
                raise RuntimeError(
                    "SpinTaylorT4 adaptive step size fell below machine precision"
                )
    return None


def _cubic_hermite_state(
    start,
    end,
    start_slope,
    end_slope,
    step,
    fraction,
):
    """Interpolate an accepted time step at one uniform output sample."""

    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    return (
        (2.0 * fraction3 - 3.0 * fraction2 + 1.0) * start
        + (fraction3 - 2.0 * fraction2 + fraction) * step * start_slope
        + (-2.0 * fraction3 + 3.0 * fraction2) * end
        + (fraction3 - fraction2) * step * end_slope
    )


def _integrate_uniform_time_branch(
    reference_state,
    target_mf,
    output_step,
    direction,
    rhs,
    physical_check,
    *,
    rtol,
    atol,
    max_steps,
    retain_target_crossing=False,
    retain_physical_boundary_outputs=False,
    allow_asymptotic_boundary=False,
):
    """Evolve one branch and sample it uniformly in dimensionless time.

    ``retain_target_crossing`` continues to the first output sample across a
    requested frequency. ``retain_physical_boundary_outputs`` mirrors LAL's
    Hermite driver by interpolating cadence samples from an error-accepted
    adaptive step before applying its physical stopping test. The legacy
    IMRPhenomX PNR support-grid builder instead stops before either crossing.
    ``allow_asymptotic_boundary`` treats adaptive-step underflow as a physical
    stop. This is needed for the unexpanded SpinTaylorT1 energy denominator
    and reciprocal SpinTaylorT5 timing series, whose poles can be approached
    before an error-accepted step crosses them.
    """

    outputs = [reference_state]
    current_state = reference_state
    current_time = 0.0
    next_output_time = direction * output_step
    step_size = direction * output_step
    epsilon = torch.finfo(reference_state.dtype).eps
    attempts = 0

    while True:
        if attempts >= max_steps:
            raise RuntimeError("SpinTaylor integration exceeded the maximum step count")
        attempts += 1
        candidate, error, start_slope, end_slope = _dormand_prince_step_with_slopes(
            current_state, step_size, rhs
        )
        physical_boundary = False
        try:
            physical_check(candidate, end_slope)
        except _SpinTaylorPhysicalBoundary:
            physical_boundary = True

        scale = atol + rtol * torch.maximum(
            torch.abs(current_state),
            torch.abs(candidate),
        )
        error_ratio = float(torch.max(torch.abs(error) / scale).detach().cpu())
        accepted = (
            (not physical_boundary or retain_physical_boundary_outputs)
            and math.isfinite(error_ratio)
            and error_ratio <= 1.0
        )
        if accepted:
            previous_time = current_time
            current_time += step_size
            tolerance = (
                16.0
                * epsilon
                * max(
                    1.0,
                    abs(current_time),
                    abs(next_output_time),
                )
            )
            while direction * (current_time - next_output_time) >= -tolerance:
                fraction = (next_output_time - previous_time) / step_size
                output = _cubic_hermite_state(
                    current_state,
                    candidate,
                    start_slope,
                    end_slope,
                    step_size,
                    fraction,
                )
                outputs.append(output)
                next_output_time += direction * output_step
                output_mf = float((output[1] / math.pi).detach().cpu())
                if direction * (output_mf - target_mf) >= 0.0:
                    return outputs, True

            if physical_boundary:
                return outputs, False
            current_state = candidate
            candidate_mf = float((candidate[1] / math.pi).detach().cpu())
            if (
                not retain_target_crossing
                and direction * (candidate_mf - target_mf) >= 0.0
            ):
                return outputs, True
            factor = (
                5.0
                if error_ratio == 0.0
                else min(5.0, max(0.2, 0.9 * error_ratio ** (-0.2)))
            )
            step_size *= factor
            continue

        if physical_boundary:
            factor = 0.5
        elif not math.isfinite(error_ratio):
            factor = 0.2
        else:
            factor = min(0.5, max(0.1, 0.9 * error_ratio ** (-0.25)))
        step_size *= factor
        minimum_step = 16.0 * epsilon * max(1.0, abs(current_time))
        if abs(step_size) < minimum_step:
            if physical_boundary:
                return outputs, False
            if allow_asymptotic_boundary:
                return outputs, False
            raise RuntimeError(
                "SpinTaylor adaptive step size fell below machine precision"
            )


def spintaylor_t4_trajectory(
    mf,
    mf_ref,
    mass1,
    mass2,
    chi1,
    chi2,
    *,
    lnhat=(0.0, 0.0, 1.0),
    e1=(1.0, 0.0, 0.0),
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
    rtol=None,
    atol=None,
    max_steps=100000,
    truncate_at_boundary=False,
):
    """Evolve SpinTaylorT4 at increasing geometric GW frequencies ``M f``.

    The input spins and orientation are defined at ``mf_ref``.  Frequencies
    below and above that point are integrated independently, just as in LAL's
    reference-frequency driver.  The adaptive system uses ``M omega`` as its
    independent variable, avoiding a CPU-side time grid without changing the
    physical trajectory.

    This routine evolves one binary at a time; ``mf`` may contain any strictly
    increasing set of positive samples and need not contain ``mf_ref``.  If
    ``truncate_at_boundary`` is true, a forward branch that reaches a physical
    PN stopping condition returns the completed frequency prefix.  Numerical
    solver failures and backward-branch stopping conditions still raise.
    """

    reference = next(
        (
            value
            for value in (mf, chi1, chi2, lnhat, e1)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        mf = torch.as_tensor(mf, dtype=torch.float64)
    else:
        dtype = reference.dtype if reference.dtype.is_floating_point else torch.float64
        mf = torch.as_tensor(mf, dtype=dtype, device=reference.device)
    if mf.ndim != 1 or mf.numel() == 0:
        raise ValueError("SpinTaylorT4 frequencies must be a nonempty vector")
    if not bool(torch.all(torch.isfinite(mf) & (mf > 0.0)).detach().cpu()):
        raise ValueError("SpinTaylorT4 frequencies must be finite and positive")
    if mf.numel() > 1 and not bool(torch.all(mf[1:] > mf[:-1]).detach().cpu()):
        raise ValueError("SpinTaylorT4 frequencies must be strictly increasing")

    mf_ref = torch.as_tensor(mf_ref, dtype=mf.dtype, device=mf.device)
    if (
        mf_ref.numel() != 1
        or not bool(torch.isfinite(mf_ref).detach().cpu())
        or float(mf_ref.detach().cpu()) <= 0.0
    ):
        raise ValueError("SpinTaylorT4 reference frequency must be finite and positive")
    mf_ref = mf_ref.reshape(())

    mass1 = torch.as_tensor(mass1, dtype=mf.dtype, device=mf.device)
    mass2 = torch.as_tensor(mass2, dtype=mf.dtype, device=mf.device)
    if mass1.numel() != 1 or mass2.numel() != 1:
        raise ValueError("SpinTaylorT4 trajectory supports one binary at a time")
    mass1 = mass1.reshape(())
    mass2 = mass2.reshape(())
    if (
        not bool(torch.isfinite(mass1 + mass2).detach().cpu())
        or float(mass1.detach().cpu()) <= 0.0
        or float(mass2.detach().cpu()) <= 0.0
    ):
        raise ValueError("SpinTaylorT4 masses must be finite and positive")

    chi1 = torch.as_tensor(chi1, dtype=mf.dtype, device=mf.device)
    chi2 = torch.as_tensor(chi2, dtype=mf.dtype, device=mf.device)
    lnhat = torch.as_tensor(lnhat, dtype=mf.dtype, device=mf.device)
    e1 = torch.as_tensor(e1, dtype=mf.dtype, device=mf.device)
    if any(vector.shape != (3,) for vector in (chi1, chi2, lnhat, e1)):
        raise ValueError("SpinTaylorT4 spins and frame vectors must have length three")
    if not bool(
        torch.all(torch.isfinite(torch.cat((chi1, chi2, lnhat, e1)))).detach().cpu()
    ):
        raise ValueError("SpinTaylorT4 initial vectors must be finite")

    total_mass = mass1 + mass2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    reference_state = torch.cat(
        (
            torch.zeros(1, dtype=mf.dtype, device=mf.device),
            (math.pi * mf_ref).reshape(1),
            lnhat,
            spin1,
            spin2,
            e1,
        )
    )

    epsilon = torch.finfo(mf.dtype).eps
    rtol = max(1.0e-10, 32.0 * epsilon) if rtol is None else float(rtol)
    atol = max(1.0e-12, 32.0 * epsilon) if atol is None else float(atol)
    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("SpinTaylorT4 max_steps must be a positive integer") from exc
    if (
        not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol <= 0.0
        or atol <= 0.0
        or max_steps < 1
    ):
        raise ValueError(
            "SpinTaylorT4 solver tolerances and max_steps must be positive"
        )
    if not isinstance(truncate_at_boundary, bool):
        raise ValueError("SpinTaylorT4 truncate_at_boundary must be boolean")
    _spintaylor_tidal_terms(tidal_order)

    quadrupole1 = torch.as_tensor(quadrupole1, dtype=mf.dtype, device=mf.device)
    quadrupole2 = torch.as_tensor(quadrupole2, dtype=mf.dtype, device=mf.device)
    lambda1 = torch.as_tensor(lambda1, dtype=mf.dtype, device=mf.device)
    lambda2 = torch.as_tensor(lambda2, dtype=mf.dtype, device=mf.device)
    if any(
        value.numel() != 1 for value in (quadrupole1, quadrupole2, lambda1, lambda2)
    ):
        raise ValueError("SpinTaylorT4 matter parameters must be scalar")
    if not bool(
        torch.all(
            torch.isfinite(torch.stack((quadrupole1, quadrupole2, lambda1, lambda2)))
        )
        .detach()
        .cpu()
    ):
        raise ValueError("SpinTaylorT4 matter parameters must be finite")

    def rhs(state):
        return _spintaylor_frequency_rhs(
            state,
            mass1_fraction,
            mass2_fraction,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            lambda1=lambda1,
            lambda2=lambda2,
            tidal_order=tidal_order,
        )

    target_omega = math.pi * mf
    left = int(torch.searchsorted(target_omega, reference_state[1], right=False).item())
    right = int(torch.searchsorted(target_omega, reference_state[1], right=True).item())
    outputs = [None] * mf.numel()
    for index in range(left, right):
        outputs[index] = reference_state
    _integrate_frequency_branch(
        reference_state,
        target_omega,
        list(range(left - 1, -1, -1)),
        outputs,
        rhs,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    boundary_index = _integrate_frequency_branch(
        reference_state,
        target_omega,
        list(range(right, mf.numel())),
        outputs,
        rhs,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        truncate_at_boundary=truncate_at_boundary,
    )
    if boundary_index is not None:
        mf = mf[:boundary_index]
        outputs = outputs[:boundary_index]
        if not outputs:
            raise RuntimeError(
                "SpinTaylorT4 reached its physical boundary before the first sample"
            )
    return SpinTaylorTrajectory(mf=mf, state=torch.stack(outputs))


def spintaylor_t4_time_trajectory(
    mf_start,
    mf_ref,
    mf_end,
    mass1,
    mass2,
    chi1,
    chi2,
    *,
    lnhat=(0.0, 0.0, 1.0),
    e1=(1.0, 0.0, 0.0),
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
    coarse_factor=1.0,
    pnr_fine_grid=False,
    rtol=None,
    atol=None,
    max_steps=100000,
):
    """Evolve SpinTaylorT4 on LAL's uniform-time support grid.

    ``mf_start``, ``mf_ref``, and ``mf_end`` are geometric GW frequencies.
    The output interval is ``0.5 * coarse_factor / mf_end`` in units of
    ``t / M``.  A lower-frequency branch is evolved backward from the
    reference state, reversed, and joined to the independently evolved
    forward branch without duplicating the reference sample.

    The forward branch stops at the first physical PN boundary when that is
    reached before ``mf_end``.  The first uniformly sampled point crossing a
    requested frequency bound is retained, matching LAL's time-series driver.

    When ``pnr_fine_grid`` is true, reproduce the support-grid construction
    used by IMRPhenomX's numerical PNR angles.  For ``coarse_factor > 1``, the
    high-frequency tail is restarted from the ninth-last coarse-grid position
    with unit spacing.  This also preserves the legacy backward/forward stitch
    used by that path.
    """

    reference = next(
        (
            value
            for value in (mf_ref, mf_start, mf_end, chi1, chi2, lnhat, e1)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        reference_grid = torch.as_tensor([mf_ref], dtype=torch.float64)
    else:
        dtype = reference.dtype if reference.dtype.is_floating_point else torch.float64
        reference_grid = torch.as_tensor(
            mf_ref,
            dtype=dtype,
            device=reference.device,
        ).reshape(-1)
    if reference_grid.numel() != 1:
        raise ValueError("SpinTaylorT4 reference frequency must be scalar")

    epsilon = torch.finfo(reference_grid.dtype).eps
    rtol = max(1.0e-12, 32.0 * epsilon) if rtol is None else float(rtol)
    atol = max(1.0e-12, 32.0 * epsilon) if atol is None else float(atol)
    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("SpinTaylorT4 max_steps must be a positive integer") from exc

    reference_trajectory = spintaylor_t4_trajectory(
        reference_grid,
        reference_grid[0],
        mass1,
        mass2,
        chi1,
        chi2,
        lnhat=lnhat,
        e1=e1,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        tidal_order=tidal_order,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    state = reference_trajectory.state[0]
    mf_start = torch.as_tensor(mf_start, dtype=state.dtype, device=state.device)
    mf_ref = reference_trajectory.mf[0]
    mf_end = torch.as_tensor(mf_end, dtype=state.dtype, device=state.device)
    coarse_factor = torch.as_tensor(
        coarse_factor,
        dtype=state.dtype,
        device=state.device,
    )
    if any(value.numel() != 1 for value in (mf_start, mf_end, coarse_factor)):
        raise ValueError("SpinTaylorT4 time-grid controls must be scalar")
    mf_start, mf_end, coarse_factor = (
        value.reshape(()) for value in (mf_start, mf_end, coarse_factor)
    )
    controls = torch.stack((mf_start, mf_ref, mf_end, coarse_factor))
    if not bool(torch.all(torch.isfinite(controls)).detach().cpu()):
        raise ValueError("SpinTaylorT4 time-grid controls must be finite")
    if (
        float(mf_start.detach().cpu()) <= 0.0
        or float(mf_start.detach().cpu()) > float(mf_ref.detach().cpu())
        or float(mf_end.detach().cpu()) <= float(mf_ref.detach().cpu())
    ):
        raise ValueError(
            "SpinTaylorT4 time grid requires 0 < mf_start <= mf_ref < mf_end"
        )
    if float(coarse_factor.detach().cpu()) < 1.0:
        raise ValueError("SpinTaylorT4 coarse_factor must be at least one")

    mass1 = torch.as_tensor(mass1, dtype=state.dtype, device=state.device).reshape(())
    mass2 = torch.as_tensor(mass2, dtype=state.dtype, device=state.device).reshape(())
    total_mass = mass1 + mass2
    mass1_fraction = mass1 / total_mass
    mass2_fraction = mass2 / total_mass
    quadrupole1 = torch.as_tensor(
        quadrupole1,
        dtype=state.dtype,
        device=state.device,
    ).reshape(())
    quadrupole2 = torch.as_tensor(
        quadrupole2,
        dtype=state.dtype,
        device=state.device,
    ).reshape(())
    lambda1 = torch.as_tensor(lambda1, dtype=state.dtype, device=state.device).reshape(
        ()
    )
    lambda2 = torch.as_tensor(lambda2, dtype=state.dtype, device=state.device).reshape(
        ()
    )

    def rhs(current_state):
        return spintaylor_t4_rhs(
            current_state,
            mass1_fraction,
            mass2_fraction,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            lambda1=lambda1,
            lambda2=lambda2,
            tidal_order=tidal_order,
        )

    def physical_check(current_state, derivatives):
        _check_spintaylor_physical_state(
            current_state,
            mass1_fraction,
            mass2_fraction,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            lambda1=lambda1,
            lambda2=lambda2,
            tidal_order=tidal_order,
            derivatives=derivatives,
        )

    coarse_factor_value = float(coarse_factor.detach().cpu())
    output_step = float((0.5 * coarse_factor / mf_end).detach().cpu())
    mf_start_value = float(mf_start.detach().cpu())
    mf_ref_value = float(mf_ref.detach().cpu())
    mf_end_value = float(mf_end.detach().cpu())
    if mf_start_value < mf_ref_value:
        backward, reached_start = _integrate_uniform_time_branch(
            state,
            mf_start_value,
            output_step,
            -1.0,
            rhs,
            physical_check,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        if not reached_start:
            raise RuntimeError(
                "SpinTaylorT4 reached a physical boundary before mf_start"
            )
        ordered = list(reversed(backward))
    else:
        ordered = [state]

    forward, _ = _integrate_uniform_time_branch(
        state,
        mf_end_value,
        output_step,
        1.0,
        rhs,
        physical_check,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    ordered.extend(forward[1:])

    if pnr_fine_grid:
        if mf_start_value < mf_ref_value and len(backward) > 1:
            # LAL's appendTS starts the forward copy at ``origlen - 2`` and
            # leaves one unused tail position.  Remove the overwritten
            # penultimate backward sample and account for that reserved slot
            # when locating the fine-grid transition.
            reference_index = len(backward) - 1
            del ordered[reference_index - 1]
            coarse_length = len(ordered) + 1
        else:
            coarse_length = len(ordered)

        if coarse_factor_value > 1.0:
            buffer_length = min(9, coarse_length - 1)
            transition_index = coarse_length - 1 - buffer_length
            transition_state = ordered[transition_index]
            fine, _ = _integrate_uniform_time_branch(
                transition_state,
                mf_end_value,
                float((0.5 / mf_end).detach().cpu()),
                1.0,
                rhs,
                physical_check,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            )
            ordered = ordered[:transition_index]
            ordered.extend(fine)
        else:
            ordered = ordered[: coarse_length - 1]

        if not ordered:
            raise RuntimeError("SpinTaylorT4 PNR support grid is empty")

    state = torch.stack(ordered)
    mf = state[:, 1] / math.pi
    if mf.numel() > 1 and not bool(torch.all(mf[1:] > mf[:-1]).detach().cpu()):
        raise RuntimeError("SpinTaylorT4 time trajectory is not frequency ordered")
    return SpinTaylorTrajectory(mf=mf, state=state)


def _rotate_y(angle, vector):
    """Apply LAL's active y-axis rotation to Torch vectors."""

    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (x * cosine + z * sine, y, -x * sine + z * cosine),
        dim=-1,
    )


def _rotate_z(angle, vector):
    """Apply LAL's active z-axis rotation to Torch vectors."""

    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (x * cosine - y * sine, x * sine + y * cosine, z),
        dim=-1,
    )


def _angle_binary_inputs(
    reference,
    mf_ref,
    mass1,
    mass2,
    chi1,
    chi2,
    inclination,
    phi_ref,
):
    """Validate and mass-order one binary used by the angle construction."""

    dtype = reference.dtype if reference.dtype.is_floating_point else torch.float64
    device = reference.device
    scalars = tuple(
        torch.as_tensor(value, dtype=dtype, device=device)
        for value in (mf_ref, mass1, mass2, inclination, phi_ref)
    )
    if any(value.numel() != 1 for value in scalars):
        raise ValueError("SpinTaylor angle source parameters must be scalar")
    mf_ref, mass1, mass2, inclination, phi_ref = (
        value.reshape(()) for value in scalars
    )
    chi1 = torch.as_tensor(chi1, dtype=dtype, device=device)
    chi2 = torch.as_tensor(chi2, dtype=dtype, device=device)
    if chi1.shape != (3,) or chi2.shape != (3,):
        raise ValueError("SpinTaylor angle spins must have length three")
    all_values = torch.cat(
        (
            torch.stack((mf_ref, mass1, mass2, inclination, phi_ref)),
            chi1,
            chi2,
        )
    )
    if not bool(torch.all(torch.isfinite(all_values)).detach().cpu()):
        raise ValueError("SpinTaylor angle source parameters must be finite")
    if (
        float(mf_ref.detach().cpu()) <= 0.0
        or float(mass1.detach().cpu()) <= 0.0
        or float(mass2.detach().cpu()) <= 0.0
    ):
        raise ValueError("SpinTaylor angle frequencies and masses must be positive")

    swapped = bool((mass2 > mass1).detach().cpu())
    if swapped:
        mass1, mass2 = mass2, mass1
        chi1, chi2 = chi2, chi1
    return (
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination,
        phi_ref,
        swapped,
    )


def _lpn_3pn(velocity, eta, chi1_l, chi2_l):
    """Return LAL's version-330 3PN orbital angular momentum."""

    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    l2 = 1.5 + eta / 6.0
    l3 = (
        5.0
        * (chi1_l * (-2.0 - 2.0 * delta + eta) + chi2_l * (-2.0 + 2.0 * delta + eta))
        / 6.0
    )
    l4 = (81.0 + (-57.0 + eta) * eta) / 24.0
    l5 = (
        -7.0
        * (
            chi1_l * (72.0 + delta * (72.0 - 31.0 * eta) + eta * (-121.0 + 2.0 * eta))
            + chi2_l
            * (72.0 + eta * (-121.0 + 2.0 * eta) + delta * (-72.0 + 31.0 * eta))
        )
        / 144.0
    )
    l6 = (
        10935.0 + eta * (-62001.0 + eta * (1674.0 + 7.0 * eta) + 2214.0 * math.pi**2)
    ) / 1296.0
    velocity2 = velocity * velocity
    polynomial = 1.0 + l2 * velocity2 + l3 * velocity2 * velocity
    polynomial += l4 * velocity2**2 + l5 * velocity2**2 * velocity
    polynomial += l6 * velocity2**3
    return eta * polynomial / velocity


def spintaylor_j_frame(
    mf_ref,
    mass1,
    mass2,
    chi1,
    chi2,
    *,
    inclination=0.0,
    phi_ref=0.0,
    convention=1,
):
    """Build the source-to-J-frame rotation used by PhenomX precession.

    The reference frequency is geometric ``M f``.  Conventions 0, 1, 5, 6,
    and 7 follow ``IMRPhenomXGetAndSetPrecessionVariables``; version 330
    normally uses convention 1.
    """

    reference = next(
        (
            value
            for value in (mf_ref, chi1, chi2, mass1, mass2)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        reference = torch.as_tensor(mf_ref, dtype=torch.float64)
    (
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination,
        phi_ref,
        _,
    ) = _angle_binary_inputs(
        reference,
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination,
        phi_ref,
    )
    try:
        convention = operator.index(convention)
    except TypeError as exc:
        raise ValueError("unsupported PhenomXP angle convention") from exc
    if convention not in (0, 1, 5, 6, 7):
        raise ValueError("unsupported PhenomXP angle convention")

    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    eta = fraction1 * fraction2
    velocity = torch.pow(math.pi * mf_ref, 1.0 / 3.0)
    orbital_momentum = _lpn_3pn(velocity, eta, chi1[2], chi2[2])
    angular_momentum = torch.stack(
        (
            fraction1**2 * chi1[0] + fraction2**2 * chi2[0],
            fraction1**2 * chi1[1] + fraction2**2 * chi2[1],
            fraction1**2 * chi1[2] + fraction2**2 * chi2[2] + orbital_momentum,
        )
    )
    magnitude = torch.linalg.vector_norm(angular_momentum)
    theta_j_source = torch.where(
        magnitude < 1.0e-10,
        torch.zeros_like(magnitude),
        torch.acos(torch.clamp(angular_momentum[2] / magnitude, -1.0, 1.0)),
    )
    aligned = (torch.abs(angular_momentum[0]) < 1.0e-15) & (
        torch.abs(angular_momentum[1]) < 1.0e-15
    )
    aligned_phi = (
        math.pi / 2.0 - phi_ref if convention in (0, 5) else torch.zeros_like(phi_ref)
    )
    phi_j_source = torch.where(
        aligned,
        aligned_phi,
        torch.atan2(angular_momentum[1], angular_momentum[0]),
    )

    line_of_sight = torch.stack(
        (
            torch.sin(inclination) * torch.sin(phi_ref),
            torch.sin(inclination) * torch.cos(phi_ref),
            torch.cos(inclination),
        )
    )
    intermediate_sight = _rotate_y(
        -theta_j_source,
        _rotate_z(-phi_j_source, line_of_sight),
    )
    sight_on_axis = (torch.abs(intermediate_sight[0]) < 1.0e-15) & (
        torch.abs(intermediate_sight[1]) < 1.0e-15
    )
    kappa = torch.where(
        sight_on_axis,
        torch.zeros_like(phi_ref),
        torch.atan2(intermediate_sight[1], intermediate_sight[0]),
    )

    rotated_orbit = _rotate_z(
        -kappa,
        _rotate_y(
            -theta_j_source,
            _rotate_z(-phi_j_source, line_of_sight.new_tensor((0.0, 0.0, 1.0))),
        ),
    )
    orbit_on_axis = (torch.abs(rotated_orbit[0]) < 1.0e-15) & (
        torch.abs(rotated_orbit[1]) < 1.0e-15
    )
    if convention in (0, 5):
        alpha0 = torch.where(
            orbit_on_axis,
            rotated_orbit.new_tensor(math.pi),
            torch.atan2(rotated_orbit[1], rotated_orbit[0]),
        )
    else:
        alpha0 = math.pi - kappa
    epsilon0 = (
        phi_j_source - math.pi
        if convention in (1, 6)
        else torch.zeros_like(phi_j_source)
    )
    return SpinTaylorJFrame(
        phi_j_source=phi_j_source,
        theta_j_source=theta_j_source,
        kappa=kappa,
        alpha0=alpha0,
        epsilon0=epsilon0,
    )


def spintaylor_unwrap_angle(angle):
    """Unwrap adjacent angle samples exactly as the PhenomX utility does."""

    if not isinstance(angle, torch.Tensor):
        angle = torch.as_tensor(angle, dtype=torch.float64)
    if angle.ndim != 1 or angle.numel() == 0:
        raise ValueError("SpinTaylor angle samples must be a nonempty vector")
    difference = angle[1:] - angle[:-1]
    difference = torch.where(
        difference > math.pi,
        difference - 2.0 * math.pi,
        torch.where(
            difference < -math.pi,
            difference + 2.0 * math.pi,
            difference,
        ),
    )
    return torch.cat((angle[:1], angle[:1] + torch.cumsum(difference, dim=0)))


def _spintaylor_continuation_spline(mf, values, fmax, minimum_fraction):
    if not isinstance(mf, torch.Tensor):
        mf = torch.as_tensor(mf, dtype=torch.float64)
    elif not mf.dtype.is_floating_point:
        mf = mf.to(dtype=torch.float64)
    values = _as_tensor_like(values, mf)
    fmax = _as_tensor_like(fmax, mf)
    if mf.ndim != 1 or values.ndim != 1 or values.shape != mf.shape:
        raise ValueError("continuation knots and values must be equal-length vectors")
    if mf.numel() < 2:
        raise ValueError("a SpinTaylor continuation requires at least two knots")
    if fmax.ndim != 0:
        raise ValueError("the SpinTaylor continuation frequency must be scalar")
    valid = (
        torch.all(torch.isfinite(mf))
        & torch.all(torch.isfinite(values))
        & torch.isfinite(fmax)
        & torch.all(mf[1:] > mf[:-1])
        & (minimum_fraction * fmax >= mf[0])
        & (fmax <= mf[-1])
        & (fmax > 0.0)
    )
    if not bool(valid.detach().cpu()):
        raise ValueError("invalid SpinTaylor continuation spline domain")
    return mf, values, fmax, _natural_cubic_coeff(mf, values)


def build_spintaylor_alpha_mrd(mf, alpha, fmax):
    """Build LAL's analytical merger-ringdown continuation for alpha.

    The continuation is constrained to the natural-cubic inspiral spline at
    ``0.97 * fmax`` and ``0.99 * fmax`` and matches its derivative at the
    first frequency.
    """

    mf, alpha, fmax, coefficients = _spintaylor_continuation_spline(
        mf,
        alpha,
        fmax,
        0.97,
    )
    linear, quadratic, cubic = coefficients
    f1 = 0.97 * fmax
    f2 = 0.99 * fmax
    alpha1 = -_spline_eval(f1, mf, alpha, linear, quadratic, cubic)
    alpha2 = -_spline_eval(f2, mf, alpha, linear, quadratic, cubic)
    derivative1 = -_spline_derivative(f1, mf, linear, quadratic, cubic)
    f1_squared = f1.square()
    f2_squared = f2.square()
    denominator = 2.0 * (f1_squared - f2_squared).square()
    a = (
        f1**3 * (f1 - f2) * (f1 + f2) * derivative1
        + 2.0 * (f1**4 - 2.0 * f1_squared * f2_squared) * alpha1
        + 2.0 * f2**4 * alpha2
    ) / denominator
    b = (
        f1**4
        * f2_squared
        * (
            f1 * (f1 - f2) * (f1 + f2) * derivative1
            + 2.0 * f2_squared * (-alpha1 + alpha2)
        )
        / denominator
    )
    c = (
        f1_squared
        * (f1 * (-(f1**4) + f2**4) * derivative1 + 4.0 * f2**4 * (alpha1 - alpha2))
        / denominator
    )
    return SpinTaylorAlphaMRD(a=a, b=b, c=c)


def spintaylor_alpha_mrd(mf, parameters):
    """Evaluate the analytical SpinTaylor merger-ringdown alpha angle."""

    mf = _as_tensor_like(mf, parameters.a)
    return -(parameters.a + parameters.b / mf**4 + parameters.c / mf.square())


def spintaylor_alpha_mrd_derivative(mf, parameters):
    """Evaluate the frequency derivative of merger-ringdown alpha."""

    mf = _as_tensor_like(mf, parameters.a)
    return 4.0 * parameters.b / mf**5 + 2.0 * parameters.c / mf**3


def build_spintaylor_beta_mrd(
    mf,
    cosbeta,
    fmax,
    *,
    damping_difference,
    ringdown_beta,
):
    """Build LAL's analytical merger-ringdown continuation for beta.

    ``damping_difference`` is the dimensionless difference between the
    remnant-frame (2, 1) and (2, 2) damping frequencies.  Version 330 supplies
    the fitted ``ringdown_beta`` asymptote; earlier numerical-angle versions
    instead use zero or pi.
    """

    mf, cosbeta, fmax, coefficients = _spintaylor_continuation_spline(
        mf,
        cosbeta,
        fmax,
        0.97,
    )
    linear, quadratic, cubic = coefficients
    damping_difference = _as_tensor_like(damping_difference, mf)
    ringdown_beta = _as_tensor_like(ringdown_beta, mf)
    if damping_difference.ndim != 0 or ringdown_beta.ndim != 0:
        raise ValueError("SpinTaylor ringdown parameters must be scalar")
    if not bool(
        (torch.isfinite(damping_difference) & torch.isfinite(ringdown_beta))
        .detach()
        .cpu()
    ):
        raise ValueError("SpinTaylor ringdown parameters must be finite")

    f1 = 0.97 * fmax
    f2 = 0.98 * fmax
    cosbeta1 = _spline_eval(f1, mf, cosbeta, linear, quadratic, cubic)
    cosbeta2 = _spline_eval(f2, mf, cosbeta, linear, quadratic, cubic)
    derivative_cosbeta2 = _spline_derivative(
        f2,
        mf,
        linear,
        quadratic,
        cubic,
    )
    cosbeta_max = _spline_eval(fmax, mf, cosbeta, linear, quadratic, cubic)
    flat = (
        (torch.abs(cosbeta1) > 1.0)
        | (torch.abs(cosbeta2) > 1.0)
        | (torch.abs(cosbeta_max) > 1.0)
    )
    beta1 = torch.acos(cosbeta1)
    beta2 = torch.acos(cosbeta2)
    square_root_argument = 1.0 - cosbeta2.square()
    derivative_beta2 = -derivative_cosbeta2 / torch.sqrt(
        torch.where(
            square_root_argument <= 0.0,
            torch.ones_like(square_root_argument),
            square_root_argument,
        )
    )
    kappa = 2.0 * math.pi * damping_difference
    exponential1 = torch.exp(kappa * f1)
    exponential2 = torch.exp(kappa * f2)
    f1_squared = f1.square()
    f2_squared = f2.square()
    denominator = (f1 - f2).square()
    a = (
        -exponential1 * f1**4 * (ringdown_beta - beta1)
        + exponential2
        * f2**3
        * (
            f2 * (-f1 + f2) * derivative_beta2
            + (-f2 * (3.0 + f2 * kappa) + f1 * (4.0 + f2 * kappa))
            * (ringdown_beta - beta2)
        )
    ) / denominator
    b = (
        2.0 * exponential1 * f1**4 * f2 * (ringdown_beta - beta1)
        + exponential2
        * f2**3
        * (
            (f1 - f2) * f2 * (f1 + f2) * derivative_beta2
            - (-f2_squared * (2.0 + f2 * kappa) + f1_squared * (4.0 + f2 * kappa))
            * (ringdown_beta - beta2)
        )
    ) / denominator
    c = (
        -exponential1 * f1**4 * f2_squared * (ringdown_beta - beta1)
        + exponential2
        * f1
        * f2**4
        * (
            f2 * (-f1 + f2) * derivative_beta2
            + (-f2 * (2.0 + f2 * kappa) + f1 * (3.0 + f2 * kappa))
            * (ringdown_beta - beta2)
        )
    ) / denominator
    zero = torch.zeros_like(a)
    return SpinTaylorBetaMRD(
        a=torch.where(flat, zero, a),
        b=torch.where(flat, zero, b),
        c=torch.where(flat, zero, c),
        offset=torch.where(flat, zero, ringdown_beta),
        damping_difference=damping_difference,
        cosbeta_sign=torch.copysign(torch.ones_like(cosbeta_max), cosbeta_max),
        flat=flat,
    )


def spintaylor_beta_mrd(mf, parameters):
    """Evaluate the analytical SpinTaylor merger-ringdown beta angle."""

    mf = _as_tensor_like(mf, parameters.a)
    kappa = 2.0 * math.pi * parameters.damping_difference
    continuation = (
        torch.exp(-mf * kappa)
        / mf
        * (parameters.a / mf + parameters.b / mf.square() + parameters.c / mf**3)
        + parameters.offset
    )
    flat = torch.acos(parameters.cosbeta_sign)
    return torch.where(parameters.flat, flat, continuation)


def spintaylor_j_frame_angles(trajectory, frame):
    """Rotate a SpinTaylor trajectory and return raw alpha and cos(beta)."""

    rotated = _rotate_z(
        -frame.kappa,
        _rotate_y(
            -frame.theta_j_source,
            _rotate_z(-frame.phi_j_source, trajectory.lnhat),
        ),
    )
    alpha = spintaylor_unwrap_angle(torch.atan2(rotated[:, 1], rotated[:, 0]))
    return alpha, rotated[:, 2]


def build_spintaylor_angle_spline(
    trajectory,
    frame,
    fmax_inspiral,
    *,
    damping_difference,
    ringdown_beta,
):
    """Build the numerical-angle spline and generic IMR continuation.

    ``fmax_inspiral`` must identify a physical point inside ``trajectory``.
    Keeping that selection explicit prevents the requested output-grid limit
    from being mistaken for LAL's final accepted PN sample.
    """

    alpha, cosbeta = spintaylor_j_frame_angles(trajectory, frame)
    mf, alpha, fmax_inspiral, _ = _spintaylor_continuation_spline(
        trajectory.mf,
        alpha,
        fmax_inspiral,
        0.97,
    )
    _, cosbeta, _, _ = _spintaylor_continuation_spline(
        mf,
        cosbeta,
        fmax_inspiral,
        0.97,
    )
    values = torch.stack((alpha, cosbeta), dim=-1)
    linear, quadratic, cubic = _natural_cubic_coeff(mf, values)
    alpha_mrd = build_spintaylor_alpha_mrd(mf, alpha, fmax_inspiral)
    beta_mrd = build_spintaylor_beta_mrd(
        mf,
        cosbeta,
        fmax_inspiral,
        damping_difference=damping_difference,
        ringdown_beta=ringdown_beta,
    )
    return SpinTaylorAngleSpline(
        frame=frame,
        mf=mf,
        alpha=alpha,
        cosbeta=cosbeta,
        linear=linear,
        quadratic=quadratic,
        cubic=cubic,
        fmax_inspiral=fmax_inspiral,
        ftrans_mrd=0.98 * fmax_inspiral,
        alpha_mrd=alpha_mrd,
        beta_mrd=beta_mrd,
    )


def spintaylor_inspiral_alpha(mf, angles):
    """Evaluate the raw numerical inspiral alpha spline."""

    mf = _as_tensor_like(mf, angles.mf)
    return _spline_eval(
        mf,
        angles.mf,
        angles.alpha,
        angles.linear[:, 0],
        angles.quadratic[:, 0],
        angles.cubic[:, 0],
    )


def spintaylor_inspiral_cosbeta(mf, angles):
    """Evaluate the numerical inspiral cos(beta) spline."""

    mf = _as_tensor_like(mf, angles.mf)
    return _spline_eval(
        mf,
        angles.mf,
        angles.cosbeta,
        angles.linear[:, 1],
        angles.quadratic[:, 1],
        angles.cubic[:, 1],
    )


def spintaylor_alpha_imr(mf, angles, *, offset=0.0):
    """Evaluate numerical alpha with LAL's generic IMR continuation."""

    mf = _as_tensor_like(mf, angles.mf)
    offset = _as_tensor_like(offset, mf)
    alpha = torch.where(
        mf < angles.ftrans_mrd,
        spintaylor_inspiral_alpha(mf, angles),
        spintaylor_alpha_mrd(mf, angles.alpha_mrd),
    )
    return alpha + offset


def spintaylor_beta_imr(mf, angles):
    """Evaluate beta with LAL's generic numerical-angle continuation."""

    mf = _as_tensor_like(mf, angles.mf)
    inspiral = torch.acos(
        torch.clamp(spintaylor_inspiral_cosbeta(mf, angles), -1.0, 1.0)
    )
    return torch.where(
        mf < angles.ftrans_mrd,
        inspiral,
        spintaylor_beta_mrd(mf, angles.beta_mrd),
    )


def spintaylor_alpha_reference_offset(mf_ref, angles):
    """Return the offset that sets alpha at reference to ``frame.alpha0``."""

    raw_reference = spintaylor_alpha_imr(mf_ref, angles)
    return angles.frame.alpha0 - raw_reference


def spintaylor_t4_inspiral_angles(
    mf,
    mf_ref,
    mass1,
    mass2,
    chi1,
    chi2,
    *,
    inclination=0.0,
    phi_ref=0.0,
    convention=1,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
    rtol=None,
    atol=None,
    max_steps=100000,
):
    """Construct version-330 inspiral Euler angles on a uniform ``M f`` grid.

    Alpha and cos(beta) are natural-cubic interpolants of the rotated
    SpinTaylorT4 trajectory.  Gamma reproduces LAL's minimal-rotation
    prescription, including its forward-shifted Boole interval: gamma sample
    ``i`` integrates from ``mf[i]`` to ``mf[i] + delta_mf``.  A guard
    trajectory sample keeps that final interval inside the interpolation
    domain.  Merger-ringdown attachment is deliberately left to the PNR angle
    layer.
    """

    reference = next(
        (
            value
            for value in (mf, mf_ref, chi1, chi2, mass1, mass2)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        mf = torch.as_tensor(mf, dtype=torch.float64)
    else:
        dtype = reference.dtype if reference.dtype.is_floating_point else torch.float64
        mf = torch.as_tensor(mf, dtype=dtype, device=reference.device)
    if mf.ndim != 1 or mf.numel() < 4:
        raise ValueError("SpinTaylor Euler angles require at least four frequencies")
    if not bool(torch.all(torch.isfinite(mf) & (mf > 0.0)).detach().cpu()):
        raise ValueError(
            "SpinTaylor Euler-angle frequencies must be finite and positive"
        )
    spacing = mf[1:] - mf[:-1]
    if not bool(torch.all(spacing > 0.0).detach().cpu()):
        raise ValueError(
            "SpinTaylor Euler-angle frequencies must be strictly increasing"
        )
    tolerance = 64.0 * torch.finfo(mf.dtype).eps
    if not bool(
        torch.allclose(
            spacing,
            spacing[0].expand_as(spacing),
            rtol=tolerance,
            atol=tolerance * max(1.0, float(spacing[0].detach().cpu())),
        )
    ):
        raise ValueError("SpinTaylor Euler-angle frequencies must be uniformly spaced")

    (
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination,
        phi_ref,
        swapped,
    ) = _angle_binary_inputs(
        mf,
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination,
        phi_ref,
    )
    if bool(((mf_ref < mf[0]) | (mf_ref > mf[-1])).detach().cpu()):
        raise ValueError("SpinTaylor reference frequency must lie on the angle grid")
    if swapped:
        quadrupole1, quadrupole2 = quadrupole2, quadrupole1
        lambda1, lambda2 = lambda2, lambda1

    frame = spintaylor_j_frame(
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        inclination=inclination,
        phi_ref=phi_ref,
        convention=convention,
    )
    support_mf = torch.cat((mf, (mf[-1] + spacing[0]).reshape(1)))
    trajectory = spintaylor_t4_trajectory(
        support_mf,
        mf_ref,
        mass1,
        mass2,
        chi1,
        chi2,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    raw_alpha, support_cosbeta = spintaylor_j_frame_angles(trajectory, frame)
    angle_values = torch.stack((raw_alpha, support_cosbeta), dim=-1)
    linear, quadratic, cubic = _natural_cubic_coeff(support_mf, angle_values)
    alpha_ref = _spline_eval(
        mf_ref.reshape(1),
        support_mf,
        raw_alpha,
        linear[:, 0],
        quadratic[:, 0],
        cubic[:, 0],
    )[0]
    alpha = raw_alpha[: mf.numel()] - alpha_ref + frame.alpha0
    cosbeta = support_cosbeta[: mf.numel()]

    width = spacing
    nodes = mf[1:, None] + width[:, None] * mf.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0))
    alpha_derivative = _spline_derivative(
        nodes,
        support_mf,
        linear[:, 0],
        quadratic[:, 0],
        cubic[:, 0],
    )
    cosbeta_nodes = _spline_eval(
        nodes,
        support_mf,
        support_cosbeta,
        linear[:, 1],
        quadratic[:, 1],
        cubic[:, 1],
    )
    weights = mf.new_tensor((7.0, 32.0, 12.0, 32.0, 7.0))
    increments = (
        -width
        / 90.0
        * torch.sum(
            alpha_derivative * cosbeta_nodes * weights,
            dim=-1,
        )
    )
    raw_gamma = torch.cat(
        (
            torch.zeros(1, dtype=mf.dtype, device=mf.device),
            torch.cumsum(increments, dim=0),
        )
    )
    gamma_linear, gamma_quadratic, gamma_cubic = _natural_cubic_coeff(
        mf,
        raw_gamma,
    )
    gamma_ref = _spline_eval(
        mf_ref.reshape(1),
        mf,
        raw_gamma,
        gamma_linear,
        gamma_quadratic,
        gamma_cubic,
    )[0]
    gamma = raw_gamma - gamma_ref - frame.epsilon0
    return SpinTaylorEulerAngles(
        mf=mf,
        alpha=alpha,
        cosbeta=cosbeta,
        gamma=gamma,
    )
