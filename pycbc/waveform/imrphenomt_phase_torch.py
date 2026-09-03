# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native dominant-mode phase model for IMRPhenomT.

This module ports the coefficient construction and analytic ansatz functions
from ``LALSimIMRPhenomTHM_internals.c`` in LALSuite 7.26.1.  All operations
remain on the input Torch device and support batched leading dimensions.
Waveform timing and sample generation are intentionally kept separate.
"""

from __future__ import annotations

import math
from functools import reduce
from typing import NamedTuple

import torch

from . import imrphenomt_fits_torch as fits
from .imrphenomx_utils_torch import qnm_fring_21

_EULER_GAMMA = 0.5772156649015329
_PI = math.pi
_INSPIRAL_THETA_POINTS = (0.33, 0.45, 0.55, 0.65, 0.75, 0.82)


class IMRPhenomTPhase22Coefficients(NamedTuple):
    """Frequency and phase coefficients for the IMRPhenomT (2, 2) mode."""

    eta: torch.Tensor
    omega_peak: torch.Tensor
    c1: torch.Tensor
    c1_prec: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    omega_1pn: torch.Tensor
    omega_1half_pn: torch.Tensor
    omega_2pn: torch.Tensor
    omega_2half_pn: torch.Tensor
    omega_3pn: torch.Tensor
    omega_3half_pn: torch.Tensor
    omega_inspiral: torch.Tensor
    omega_merger: torch.Tensor
    alpha1_rd: torch.Tensor
    alpha1_rd_prec: torch.Tensor
    domega_peak: torch.Tensor
    omega_ring: torch.Tensor
    omega_ring_prec: torch.Tensor
    euler_rd_slope: torch.Tensor
    phase_offset_inspiral: torch.Tensor
    phase_offset_merger: torch.Tensor
    phase_offset_ringdown: torch.Tensor
    t_cut: torch.Tensor
    t_early: torch.Tensor
    tt0: torch.Tensor


def _coerce_real_tensors(*values):
    tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
    if tensor_values:
        device = tensor_values[0].device
        if any(value.device != device for value in tensor_values[1:]):
            raise ValueError("IMRPhenomT parameters must be on one Torch device")
        if any(value.is_complex() for value in tensor_values):
            raise TypeError("IMRPhenomT parameters must be real")
        dtypes = [
            value.dtype
            for value in tensor_values
            if value.is_floating_point()
        ]
        dtype = reduce(torch.promote_types, dtypes) if dtypes else torch.float64
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    return torch.broadcast_tensors(
        *(torch.as_tensor(value, device=device, dtype=dtype) for value in values)
    )


def _as_coefficient_tensor(value, coefficients):
    return torch.as_tensor(
        value,
        device=coefficients.eta.device,
        dtype=coefficients.eta.dtype,
    )


def _taylor_t3(
    theta,
    omega_1pn,
    omega_1half_pn,
    omega_2pn,
    omega_2half_pn,
    omega_3pn,
    omega_3half_pn,
):
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    theta5 = theta3 * theta2
    theta6 = theta3 * theta3
    theta7 = theta4 * theta3
    logterm = (107.0 / 280.0) * torch.log(theta)
    correction = (
        1.0
        + omega_1pn * theta2
        + omega_1half_pn * theta3
        + omega_2pn * theta4
        + omega_2half_pn * theta5
        + omega_3pn * theta6
        + logterm * theta6
        + omega_3half_pn * theta7
    )
    return 0.25 * theta3 * correction


def _inspiral_omega(theta, pn, omega_inspiral):
    theta8 = theta**8
    powers = torch.stack(
        tuple(theta8 * theta**power for power in range(6)), dim=-1
    )
    extension = (omega_inspiral * powers).sum(dim=-1)
    return _taylor_t3(theta, *pn) + 0.25 * theta**3 * extension


def _inspiral_omega_derivative(theta, pn, omega_inspiral):
    """Return the analytic derivative with respect to ``theta``.

    LALSuite uses a ``1e-7`` first-order difference here.  That difference is
    dominated by roundoff after solving the ill-conditioned monomial system;
    differentiating the same ansatz analytically is stable on every backend.
    """
    (
        omega_1pn,
        omega_1half_pn,
        omega_2pn,
        omega_2half_pn,
        omega_3pn,
        omega_3half_pn,
    ) = pn
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    theta5 = theta3 * theta2
    theta6 = theta3 * theta3
    theta7 = theta4 * theta3
    log_theta = torch.log(theta)
    correction = (
        1.0
        + omega_1pn * theta2
        + omega_1half_pn * theta3
        + omega_2pn * theta4
        + omega_2half_pn * theta5
        + omega_3pn * theta6
        + (107.0 / 280.0) * log_theta * theta6
        + omega_3half_pn * theta7
    )
    correction_derivative = (
        2.0 * omega_1pn * theta
        + 3.0 * omega_1half_pn * theta2
        + 4.0 * omega_2pn * theta3
        + 5.0 * omega_2half_pn * theta4
        + 6.0 * omega_3pn * theta5
        + (107.0 / 280.0) * (6.0 * log_theta + 1.0) * theta5
        + 7.0 * omega_3half_pn * theta6
    )
    taylor_derivative = 0.25 * (
        3.0 * theta2 * correction + theta3 * correction_derivative
    )
    extension_powers = torch.stack(
        tuple(theta ** (10 + power) for power in range(6)), dim=-1
    )
    extension_orders = theta.new_tensor((11, 12, 13, 14, 15, 16))
    extension_derivative = 0.25 * (
        omega_inspiral * extension_orders * extension_powers
    ).sum(dim=-1)
    return taylor_derivative + extension_derivative


def _merger_omega(
    t,
    omega_peak,
    domega_peak,
    omega_ring,
    alpha1_rd,
    omega_merger,
):
    x = torch.asinh(alpha1_rd * t)
    powers = torch.stack((x * x, x**3, x**4), dim=-1)
    return (
        1.0
        - omega_peak / omega_ring
        + (domega_peak / alpha1_rd) * x
        + (omega_merger * powers).sum(dim=-1)
    )


def _merger_omega_derivative(
    t,
    domega_peak,
    alpha1_rd,
    omega_merger,
):
    x = torch.asinh(alpha1_rd * t)
    dx_dt = alpha1_rd / torch.sqrt(1.0 + (alpha1_rd * t) ** 2)
    c1, c2, c3 = omega_merger.unbind(dim=-1)
    return dx_dt * (
        domega_peak / alpha1_rd
        + 2.0 * c1 * x
        + 3.0 * c2 * x**2
        + 4.0 * c3 * x**3
    )


def _ringdown_omega(t, c1, c2, c3, c4, omega_ring):
    exp_c = torch.exp(-c2 * t)
    exp_c2 = exp_c * exp_c
    numerator = c1 * (-2.0 * c2 * c4 * exp_c2 - c2 * c3 * exp_c)
    denominator = 1.0 + c4 * exp_c2 + c3 * exp_c
    return numerator / denominator + omega_ring


def _ringdown_omega_derivative(t, c1, c2, c3, c4):
    exp_c = torch.exp(-c2 * t)
    exp_c2 = exp_c * exp_c
    numerator = -c1 * c2 * (2.0 * c4 * exp_c2 + c3 * exp_c)
    denominator = 1.0 + c4 * exp_c2 + c3 * exp_c
    numerator_derivative = c1 * c2**2 * (
        4.0 * c4 * exp_c2 + c3 * exp_c
    )
    denominator_derivative = -c2 * (
        2.0 * c4 * exp_c2 + c3 * exp_c
    )
    return (
        numerator_derivative * denominator
        - numerator * denominator_derivative
    ) / denominator**2


def _inspiral_phase_taylor_t3(thetabar, eta, pn):
    (
        omega_1pn,
        omega_1half_pn,
        omega_2pn,
        omega_2half_pn,
        omega_3pn,
        omega_3half_pn,
    ) = pn
    five_1_8 = 5.0**0.125
    five_1_4 = 5.0**0.25
    five_3_8 = 5.0**0.375
    five_1_2 = 5.0**0.5
    five_5_8 = 5.0**0.625
    five_3_4 = 5.0**0.75
    five_7_8 = 5.0**0.875
    bracket = (
        -168.0
        - 280.0 * omega_1pn * five_1_4 * thetabar**2
        - 420.0 * omega_1half_pn * five_3_8 * thetabar**3
        - 840.0 * omega_2pn * five_1_2 * thetabar**4
        + 840.0
        * omega_2half_pn
        * torch.log(thetabar)
        * five_5_8
        * thetabar**5
        - 321.0 * five_3_4 * thetabar**6
        + 840.0 * omega_3pn * five_3_4 * thetabar**6
        + 321.0
        * torch.log(thetabar * five_1_8)
        * five_3_4
        * thetabar**6
        + 420.0 * omega_3half_pn * five_7_8 * thetabar**7
    )
    return 5.0**-0.625 * eta**-1 * thetabar**-5 * bracket / 84.0


def _inspiral_phase(
    t,
    thetabar,
    eta,
    pn,
    omega_inspiral,
    phase_offset,
):
    (
        omega_1pn,
        omega_1half_pn,
        omega_2pn,
        omega_2half_pn,
        omega_3pn,
        omega_3half_pn,
    ) = pn
    c1, c2, c3, c4, c5, c6 = omega_inspiral.unbind(dim=-1)
    five_1_8 = 5.0**0.125
    five_1_4 = 5.0**0.25
    five_3_8 = 5.0**0.375
    five_1_2 = 5.0**0.5
    five_5_8 = 5.0**0.625
    five_3_4 = 5.0**0.75
    five_7_8 = 5.0**0.875
    bracket = (
        3.0 * (-107.0 + 280.0 * omega_3pn) * five_3_4
        + 321.0 * torch.log(thetabar * five_1_8) * five_3_4
        + 420.0 * omega_3half_pn * thetabar * five_7_8
        + 56.0 * (25.0 * c1 + 3.0 * eta * t) * thetabar**2
        + 1050.0 * c2 * five_1_8 * thetabar**3
        + 280.0
        * (3.0 * c3 + eta * omega_1pn * t)
        * five_1_4
        * thetabar**4
        + 140.0
        * (5.0 * c4 + 3.0 * eta * omega_1half_pn * t)
        * five_3_8
        * thetabar**5
        + 120.0
        * (5.0 * c5 + 7.0 * eta * omega_2pn * t)
        * five_1_2
        * thetabar**6
        + 525.0 * c6 * five_5_8 * thetabar**7
        + 105.0
        * eta
        * omega_2half_pn
        * t
        * torch.log(-t)
        * five_5_8
        * thetabar**7
    )
    phase = (
        -5.0**-0.625
        * eta**-2
        * t**-1
        * thetabar**-7
        * bracket
        / 84.0
    )
    return phase + phase_offset


def _merger_phase(
    t,
    omega_peak,
    domega_peak,
    omega_ring,
    alpha1_rd,
    omega_merger,
    phase_offset,
):
    c1, c2, c3 = omega_merger.unbind(dim=-1)
    x = torch.asinh(alpha1_rd * t)
    root = torch.sqrt(1.0 + alpha1_rd**2 * t**2)
    bracket = (
        2.0 * c1 * t
        + 24.0 * c3 * t
        + 6.0 * c2 * t * x
        + domega_peak * t * x / alpha1_rd
        + t * (1.0 - omega_peak / omega_ring)
        + c1 * t * x**2
        + 12.0 * c3 * t * x**2
        + c2 * t * x**3
        + c3 * t * x**4
        - domega_peak * root / alpha1_rd**2
        - 6.0 * c2 * root / alpha1_rd
        - 2.0 * c1 * x * root / alpha1_rd
        - 24.0 * c3 * x * root / alpha1_rd
        - 3.0 * c2 * x**2 * root / alpha1_rd
        - 4.0 * c3 * x**3 * root / alpha1_rd
    )
    return omega_ring * t - omega_ring * bracket + phase_offset


def build_phase22_coefficients(
    eta,
    chi1z,
    chi2z,
    final_mass,
    final_spin,
    *,
    final_spin_prec=None,
):
    """Build dominant-mode phase coefficients on the active Torch device.

    ``chi1z`` belongs to the larger body, matching LALSuite's IMRPhenomT
    convention. ``final_mass`` is the remnant mass divided by total mass.
    """
    if final_spin_prec is None:
        final_spin_prec = final_spin
    (
        eta,
        chi1z,
        chi2z,
        final_mass,
        final_spin,
        final_spin_prec,
    ) = _coerce_real_tensors(
        eta,
        chi1z,
        chi2z,
        final_mass,
        final_spin,
        final_spin_prec,
    )

    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    mass1 = 0.5 * (1.0 + delta)
    mass2 = 0.5 * (1.0 - delta)
    shat = (mass1**2 * chi1z + mass2**2 * chi2z) / (
        mass1**2 + mass2**2
    )
    dchi = chi1z - chi2z

    tt0 = fits.inspiral_taylor_t3_t0(eta, shat, dchi, delta)
    omega_peak = fits.peak_frequency_22(eta, shat, dchi, delta)
    c2 = fits.ringdown_freq_d2_22(eta, shat, dchi, delta)
    c3 = fits.ringdown_freq_d3_22(eta, shat, dchi, delta)
    c4 = torch.zeros_like(eta)
    omega_ring = 2.0 * _PI * fits.qnm_fring_22(final_spin) / final_mass
    omega_ring_prec = (
        2.0 * _PI * fits.qnm_fring_22(final_spin_prec) / final_mass
    )
    alpha1_rd = 2.0 * _PI * fits.qnm_fdamp_22(final_spin) / final_mass
    alpha1_rd_prec = (
        2.0 * _PI * fits.qnm_fdamp_22(final_spin_prec) / final_mass
    )
    c1 = (1.0 + c3 + c4) * (omega_ring - omega_peak) / (
        c2 * (c3 + 2.0 * c4)
    )
    c1_prec = (1.0 + c3 + c4) * (omega_ring_prec - omega_peak) / (
        c2 * (c3 + 2.0 * c4)
    )
    euler_rd_slope = (
        2.0
        * _PI
        * (
            fits.qnm_fring_22(final_spin_prec)
            - qnm_fring_21(final_spin_prec)
        )
        / final_mass
    )
    euler_rd_slope = torch.where(
        final_spin < 0.0, -euler_rd_slope, euler_rd_slope
    )

    omega_1pn = 0.27641369047619047 + 11.0 * eta / 32.0
    omega_1half_pn = (
        -19.0 * (chi1z + chi2z) * eta / 80.0
        + (
            -113.0 * (chi2z * (-1.0 + delta) - chi1z * (1.0 + delta))
            - 96.0 * _PI
        )
        / 320.0
    )
    omega_2pn = (
        1855099.0
        + 1714608.0 * chi2z**2 * (-1.0 + delta)
        - 1714608.0 * chi1z**2 * (1.0 + delta)
    ) / 1.4450688e7 + (
        (
            56975.0
            + 61236.0 * chi1z**2
            - 119448.0 * chi1z * chi2z
            + 61236.0 * chi2z**2
        )
        * eta
        / 258048.0
    ) + 371.0 * eta**2 / 2048.0
    omega_2half_pn = (
        -17.0 * (chi1z + chi2z) * eta**2 / 128.0
        + (
            -146597.0
            * (chi2z * (-1.0 + delta) - chi1z * (1.0 + delta))
            - 46374.0 * _PI
        )
        / 129024.0
        + eta
        * (
            -2.0
            * (
                chi1z * (1213.0 - 63.0 * delta)
                + chi2z * (1213.0 + 63.0 * delta)
            )
            + 117.0 * _PI
        )
        / 2304.0
    )
    omega_3pn = (
        -2.499258364444952
        - 16928263.0 * chi1z**2 / 1.376256e8
        - 16928263.0 * chi2z**2 / 1.376256e8
        - 16928263.0 * chi1z**2 * delta / 1.376256e8
        + 16928263.0 * chi2z**2 * delta / 1.376256e8
        + (
            -2318475.0
            + 18767224.0 * chi1z**2
            - 54663952.0 * chi1z * chi2z
            + 18767224.0 * chi2z**2
        )
        * eta**2
        / 1.376256e8
        + 235925.0 * eta**3 / 1.769472e6
        + 107.0 * _EULER_GAMMA / 280.0
        - 6127.0 * chi1z * _PI / 12800.0
        - 6127.0 * chi2z * _PI / 12800.0
        - 6127.0 * chi1z * delta * _PI / 12800.0
        + 6127.0 * chi2z * delta * _PI / 12800.0
        + 53.0 * _PI**2 / 200.0
        + eta
        * (
            632550449425.0
            + 35200873512.0 * chi1z**2
            - 28527282000.0 * chi1z * chi2z
            + 9605339856.0 * chi1z**2 * delta
            - 1512.0
            * chi2z**2
            * (-23281001.0 + 6352738.0 * delta)
            + 34172264448.0 * (chi1z + chi2z) * _PI
            - 22912243200.0 * _PI**2
        )
        / 1.040449536e11
        + 107.0 * math.log(2.0) / 280.0
    )
    omega_3half_pn = (
        -12029.0 * (chi1z + chi2z) * eta**3 / 92160.0
        + eta**2
        * (
            507654.0 * chi1z * chi2z**2
            - 838782.0 * chi2z**3
            + chi2z
            * (-840149.0 + 507654.0 * chi1z**2 - 870576.0 * delta)
            + chi1z
            * (-840149.0 - 838782.0 * chi1z**2 + 870576.0 * delta)
            + 1701228.0 * _PI
        )
        / 1.548288e7
        + eta
        * (
            218532006.0 * chi1z * chi2z**2 * (-1.0 + delta)
            - 1134.0 * chi2z**3 * (-206917.0 + 71931.0 * delta)
            - chi2z
            * (
                1496368361.0
                - 429508815.0 * delta
                + 218532006.0 * chi1z**2 * (1.0 + delta)
            )
            + chi1z
            * (
                -1496368361.0
                - 429508815.0 * delta
                + 1134.0 * chi1z**2 * (206917.0 + 71931.0 * delta)
            )
            - 144.0
            * (
                488825.0
                + 923076.0 * chi1z**2
                - 1782648.0 * chi1z * chi2z
                + 923076.0 * chi2z**2
            )
            * _PI
        )
        / 1.8579456e8
        + (
            -6579635551.0 * chi2z * (-1.0 + delta)
            + 535759434.0 * chi2z**3 * (-1.0 + delta)
            - chi1z
            * (-6579635551.0 + 535759434.0 * chi1z**2)
            * (1.0 + delta)
            + (
                -565550067.0
                - 465230304.0 * chi2z**2 * (-1.0 + delta)
                + 465230304.0 * chi1z**2 * (1.0 + delta)
            )
            * _PI
        )
        / 1.30056192e9
    )
    pn = (
        omega_1pn,
        omega_1half_pn,
        omega_2pn,
        omega_2half_pn,
        omega_3pn,
        omega_3half_pn,
    )

    theta_points = eta.new_tensor(_INSPIRAL_THETA_POINTS)
    theta_grid = theta_points.reshape((1,) * eta.ndim + (6,))
    pn_grid = tuple(coefficient.unsqueeze(-1) for coefficient in pn)
    t_early = -5.0 / (eta * theta_points[0] ** 8)
    theta_initial = (eta * (tt0 - t_early) / 5.0) ** (-1.0 / 8.0)
    omega_points = torch.stack(
        (
            _taylor_t3(theta_initial, *pn),
            fits.inspiral_freq_cp1_22(eta, shat, dchi, delta),
            fits.inspiral_freq_cp2_22(eta, shat, dchi, delta),
            fits.inspiral_freq_cp3_22(eta, shat, dchi, delta),
            fits.inspiral_freq_cp4_22(eta, shat, dchi, delta),
            fits.inspiral_freq_cp5_22(eta, shat, dchi, delta),
        ),
        dim=-1,
    )
    design = torch.stack(
        tuple(theta_points**power for power in range(8, 14)), dim=-1
    )
    # MPS does not reliably broadcast an unbatched coefficient matrix over a
    # batched right-hand side in ``linalg.solve``. Materialize the tiny matrix
    # for each binary; this also makes the intended batch semantics explicit.
    design = design.expand(eta.shape + design.shape).contiguous()
    rhs = 4.0 / theta_grid**3 * (
        omega_points - _taylor_t3(theta_grid, *pn_grid)
    )
    omega_inspiral = torch.linalg.solve(
        design, rhs.unsqueeze(-1)
    ).squeeze(-1)

    theta_cut = eta.new_full(eta.shape, 0.81)
    t_cut = -5.0 / (eta * theta_cut**8)
    omega_cut = _inspiral_omega(theta_cut, pn, omega_inspiral)
    theta_merger = eta.new_full(eta.shape, 0.95)
    t_merger = -5.0 / (eta * theta_merger**8)
    omega_merger_cp = 1.0 - fits.merger_freq_cp1_22(
        eta, shat, dchi, delta
    ) / omega_ring
    omega_cut_bar = 1.0 - omega_cut / omega_ring

    theta2 = (-eta * t_cut / 5.0) ** (-1.0 / 8.0)
    dtheta_dt = -theta2 / (8.0 * t_cut)
    domega_cut = -(
        _inspiral_omega_derivative(theta2, pn, omega_inspiral) * dtheta_dt
    ) / omega_ring
    zero = torch.zeros_like(eta)
    domega_peak = -_ringdown_omega_derivative(
        zero, c1, c2, c3, c4
    ) / omega_ring

    as_cut = torch.asinh(alpha1_rd * t_cut)
    as_merger = torch.asinh(alpha1_rd * t_merger)
    denominator_cut = torch.sqrt(1.0 + (alpha1_rd * t_cut) ** 2)
    merger_matrix = torch.stack(
        (
            torch.stack((as_cut**2, as_cut**3, as_cut**4), dim=-1),
            torch.stack(
                (as_merger**2, as_merger**3, as_merger**4), dim=-1
            ),
            torch.stack(
                (
                    2.0 * alpha1_rd * as_cut / denominator_cut,
                    3.0 * alpha1_rd * as_cut**2 / denominator_cut,
                    4.0 * alpha1_rd * as_cut**3 / denominator_cut,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    merger_rhs = torch.stack(
        (
            omega_cut_bar
            - (1.0 - omega_peak / omega_ring)
            - (domega_peak / alpha1_rd) * as_cut,
            omega_merger_cp
            - (1.0 - omega_peak / omega_ring)
            - (domega_peak / alpha1_rd) * as_merger,
            domega_cut - domega_peak / denominator_cut,
        ),
        dim=-1,
    )
    omega_merger = torch.linalg.solve(
        merger_matrix, merger_rhs.unsqueeze(-1)
    ).squeeze(-1)

    thetabar_initial = (eta * (tt0 - t_early)) ** (-1.0 / 8.0)
    thetabar_initial_late = (-eta * t_early) ** (-1.0 / 8.0)
    phase_offset_inspiral = _inspiral_phase_taylor_t3(
        thetabar_initial, eta, pn
    ) - _inspiral_phase(
        t_early,
        thetabar_initial_late,
        eta,
        pn,
        omega_inspiral,
        zero,
    )
    thetabar_cut = (-eta * t_cut) ** (-1.0 / 8.0)
    phase_offset_merger = _inspiral_phase(
        t_cut,
        thetabar_cut,
        eta,
        pn,
        omega_inspiral,
        phase_offset_inspiral,
    ) - _merger_phase(
        t_cut,
        omega_peak,
        domega_peak,
        omega_ring,
        alpha1_rd,
        omega_merger,
        zero,
    )
    phase_offset_ringdown = _merger_phase(
        zero,
        omega_peak,
        domega_peak,
        omega_ring,
        alpha1_rd,
        omega_merger,
        phase_offset_merger,
    )

    return IMRPhenomTPhase22Coefficients(
        eta=eta,
        omega_peak=omega_peak,
        c1=c1,
        c1_prec=c1_prec,
        c2=c2,
        c3=c3,
        c4=c4,
        omega_1pn=omega_1pn,
        omega_1half_pn=omega_1half_pn,
        omega_2pn=omega_2pn,
        omega_2half_pn=omega_2half_pn,
        omega_3pn=omega_3pn,
        omega_3half_pn=omega_3half_pn,
        omega_inspiral=omega_inspiral,
        omega_merger=omega_merger,
        alpha1_rd=alpha1_rd,
        alpha1_rd_prec=alpha1_rd_prec,
        domega_peak=domega_peak,
        omega_ring=omega_ring,
        omega_ring_prec=omega_ring_prec,
        euler_rd_slope=euler_rd_slope,
        phase_offset_inspiral=phase_offset_inspiral,
        phase_offset_merger=phase_offset_merger,
        phase_offset_ringdown=phase_offset_ringdown,
        t_cut=t_cut,
        t_early=t_early,
        tt0=tt0,
    )


def taylor_t3(theta, coefficients):
    """Evaluate the TaylorT3 angular-frequency contribution."""
    theta = _as_coefficient_tensor(theta, coefficients)
    return _taylor_t3(
        theta,
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )


def inspiral_omega(theta, coefficients):
    """Evaluate the late-inspiral angular-frequency ansatz."""
    theta = _as_coefficient_tensor(theta, coefficients)
    pn = (
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )
    return _inspiral_omega(theta, pn, coefficients.omega_inspiral)


def merger_omega(t, coefficients):
    """Evaluate the rescaled merger angular-frequency ansatz."""
    t = _as_coefficient_tensor(t, coefficients)
    return _merger_omega(
        t,
        coefficients.omega_peak,
        coefficients.domega_peak,
        coefficients.omega_ring,
        coefficients.alpha1_rd,
        coefficients.omega_merger,
    )


def ringdown_omega(t, coefficients):
    """Evaluate the ringdown angular-frequency ansatz."""
    t = _as_coefficient_tensor(t, coefficients)
    return _ringdown_omega(
        t,
        coefficients.c1,
        coefficients.c2,
        coefficients.c3,
        coefficients.c4,
        coefficients.omega_ring,
    )


def omega22(t, coefficients):
    """Evaluate the physical piecewise (2, 2) angular frequency.

    The mathematical inspiral/merger boundary is used here. LALSuite offsets
    that boundary by one sample only while assembling its discrete time
    series; the underlying ansatz and its coefficients are cadence-independent.
    """
    t = _as_coefficient_tensor(t, coefficients)
    safe_inspiral_t = torch.where(t < 0.0, t, -torch.ones_like(t))
    safe_ringdown_t = torch.where(t > 0.0, t, torch.zeros_like(t))
    theta = (-coefficients.eta * safe_inspiral_t / 5.0) ** (-1.0 / 8.0)
    pn = (
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )
    inspiral = _inspiral_omega(
        theta, pn, coefficients.omega_inspiral
    )
    merger = coefficients.omega_ring * (
        1.0
        - _merger_omega(
            t,
            coefficients.omega_peak,
            coefficients.domega_peak,
            coefficients.omega_ring,
            coefficients.alpha1_rd,
            coefficients.omega_merger,
        )
    )
    ringdown = _ringdown_omega(
        safe_ringdown_t,
        coefficients.c1,
        coefficients.c2,
        coefficients.c3,
        coefficients.c4,
        coefficients.omega_ring,
    )
    return torch.where(
        t < coefficients.t_cut,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )


def omega22_derivative(t, coefficients):
    """Evaluate the analytic time derivative of :func:`omega22`."""
    t = _as_coefficient_tensor(t, coefficients)
    safe_inspiral_t = torch.where(t < 0.0, t, -torch.ones_like(t))
    safe_ringdown_t = torch.where(t > 0.0, t, torch.zeros_like(t))
    theta = (-coefficients.eta * safe_inspiral_t / 5.0) ** (-1.0 / 8.0)
    dtheta_dt = -theta / (8.0 * safe_inspiral_t)
    pn = (
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )
    inspiral = _inspiral_omega_derivative(
        theta, pn, coefficients.omega_inspiral
    ) * dtheta_dt
    merger = -coefficients.omega_ring * _merger_omega_derivative(
        t,
        coefficients.domega_peak,
        coefficients.alpha1_rd,
        coefficients.omega_merger,
    )
    ringdown = _ringdown_omega_derivative(
        safe_ringdown_t,
        coefficients.c1,
        coefficients.c2,
        coefficients.c3,
        coefficients.c4,
    )
    return torch.where(
        t < coefficients.t_cut,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )


def inspiral_phase_taylor_t3(thetabar, coefficients):
    """Evaluate the integrated early-inspiral TaylorT3 phase."""
    thetabar = _as_coefficient_tensor(thetabar, coefficients)
    pn = (
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )
    return _inspiral_phase_taylor_t3(thetabar, coefficients.eta, pn)


def inspiral_phase(t, thetabar, coefficients):
    """Evaluate the integrated late-inspiral phase ansatz."""
    t = _as_coefficient_tensor(t, coefficients)
    thetabar = _as_coefficient_tensor(thetabar, coefficients)
    pn = (
        coefficients.omega_1pn,
        coefficients.omega_1half_pn,
        coefficients.omega_2pn,
        coefficients.omega_2half_pn,
        coefficients.omega_3pn,
        coefficients.omega_3half_pn,
    )
    return _inspiral_phase(
        t,
        thetabar,
        coefficients.eta,
        pn,
        coefficients.omega_inspiral,
        coefficients.phase_offset_inspiral,
    )


def merger_phase(t, coefficients):
    """Evaluate the integrated merger phase ansatz."""
    t = _as_coefficient_tensor(t, coefficients)
    return _merger_phase(
        t,
        coefficients.omega_peak,
        coefficients.domega_peak,
        coefficients.omega_ring,
        coefficients.alpha1_rd,
        coefficients.omega_merger,
        coefficients.phase_offset_merger,
    )


def ringdown_phase(t, coefficients):
    """Evaluate the integrated ringdown phase ansatz."""
    t = _as_coefficient_tensor(t, coefficients)
    exp_c = torch.exp(-coefficients.c2 * t)
    numerator = 1.0 + coefficients.c3 * exp_c + coefficients.c4 * exp_c**2
    denominator = 1.0 + coefficients.c3 + coefficients.c4
    return (
        coefficients.c1_prec * torch.log(numerator / denominator)
        + coefficients.omega_ring_prec * t
        + coefficients.phase_offset_ringdown
    )


def phase22(t, coefficients):
    """Evaluate the cadence-independent piecewise (2, 2) phase.

    This follows the default three-region IMRPhenomT reconstruction. The
    discrete waveform builder applies LALSuite's one-sample shift to the
    inspiral/merger boundary separately.
    """
    t = _as_coefficient_tensor(t, coefficients)
    safe_inspiral_t = torch.where(t < 0.0, t, -torch.ones_like(t))
    safe_ringdown_t = torch.where(t > 0.0, t, torch.zeros_like(t))
    thetabar = (-coefficients.eta * safe_inspiral_t) ** (-1.0 / 8.0)
    inspiral = inspiral_phase(safe_inspiral_t, thetabar, coefficients)
    merger = merger_phase(t, coefficients)
    ringdown = ringdown_phase(safe_ringdown_t, coefficients)
    return torch.where(
        t < coefficients.t_cut,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )
