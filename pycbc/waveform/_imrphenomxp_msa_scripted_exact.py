# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""TorchScript-friendly exact lanes for IMRPhenomXP MSA angles.

These functions deliberately spell the operations in the same order as the
ordinary eager implementation.  They are private implementation details of a
strict, opt-in CPU inference path in :mod:`imrphenomxp_msa_torch`.
"""

from typing import Dict

import torch


def _j_norm(lnorm: torch.Tensor, values: Dict[str, float]):
    return torch.sqrt(
        torch.clamp(
            lnorm * lnorm
            + 2.0 * lnorm * values["c1_over_eta"]
            + values["SAv2"],
            min=0.0,
        )
    )


def _orbital_angular_momentum_3pn(
    velocity: torch.Tensor,
    values: Dict[str, float],
):
    v2 = velocity * velocity
    return (
        values["eta"]
        / velocity
        * (
            1.0
            + v2
            * (
                values["L0"]
                + velocity * values["L1"]
                + v2
                * (
                    values["L2"]
                    + velocity * values["L3"]
                    + v2 * values["L4"]
                )
            )
        )
    )


def _spin_evolution_coefficients(
    lnorm: torch.Tensor,
    jnorm: torch.Tensor,
    values: Dict[str, float],
):
    j2 = jnorm * jnorm
    l2 = lnorm * lnorm
    s1n2 = values["S1_norm_2"]
    s2n2 = values["S2_norm_2"]
    q = values["qq"]
    eta = values["eta"]
    j2ml2 = j2 - l2
    j2ml2sq = j2ml2 * j2ml2
    delta = values["delta_qq"]
    seff = values["Seff"]
    b = (
        (l2 + s1n2) * q
        + 2.0 * lnorm * seff
        - 2.0 * j2
        - s1n2
        - s2n2
        + (l2 + s2n2) / q
    )
    c = (
        j2ml2sq
        - 2.0 * lnorm * seff * j2ml2
        - 2.0 * ((1.0 - q) / q) * l2 * (s1n2 - q * s2n2)
        + 4.0 * eta * l2 * seff * seff
        - 2.0 * delta * (s1n2 - s2n2) * seff * lnorm
        + 2.0 * ((1.0 - q) / q) * (q * s1n2 - s2n2) * j2
    )
    d = (
        ((1.0 - q) / q) * (s2n2 - q * s1n2) * j2ml2sq
        + delta * delta * (s1n2 - s2n2) ** 2 * l2 / eta
        + 2.0 * delta * lnorm * seff * (s1n2 - s2n2) * j2ml2
    )
    return b, c, d


def _roots(
    lnorm: torch.Tensor,
    jnorm: torch.Tensor,
    values: Dict[str, float],
    static_fallback: bool,
):
    b, c, d = _spin_evolution_coefficients(lnorm, jnorm, values)
    b2 = b * b
    p = c - b2 / 3.0
    q_coefficient = (2.0 / 27.0) * b2 * b - b * c / 3.0 + d
    sqrt_argument = torch.sqrt(torch.clamp(-p / 3.0, min=0.0))
    denominator = p * sqrt_argument
    acos_argument = torch.where(
        denominator != 0.0,
        1.5 * q_coefficient / denominator,
        torch.zeros_like(denominator),
    )
    theta = torch.acos(torch.clamp(acos_argument, -1.0, 1.0)) / 3.0
    root1 = (
        2.0 * sqrt_argument * torch.cos(theta - 2.0 * 6.283185307179586 / 3.0)
        - b / 3.0
    )
    root2 = (
        2.0 * sqrt_argument * torch.cos(theta - 6.283185307179586 / 3.0)
        - b / 3.0
    )
    root3 = 2.0 * sqrt_argument * torch.cos(theta) - b / 3.0
    roots = torch.stack((root1, root2, root3))
    maximum = torch.max(roots, dim=0).values
    minimum = torch.min(roots, dim=0).values
    root3_is_middle = (maximum > root3) & (minimum < root3)
    root1_is_middle = (maximum > root1) & (minimum < root1)
    middle = torch.where(
        root3_is_middle,
        root3,
        torch.where(root1_is_middle, root1, root2),
    )
    fallback = (p >= 0.0) | (sqrt_argument == 0.0) | ~torch.isfinite(theta)
    if static_fallback:
        fallback = torch.ones_like(fallback)
    fallback_middle = torch.full_like(middle, values["S_0_norm_2"])
    return (
        torch.where(fallback, torch.zeros_like(minimum), minimum),
        torch.where(fallback, fallback_middle, torch.abs(middle)),
        torch.where(
            fallback,
            fallback_middle + 1.0e-9,
            torch.abs(maximum),
        ),
    )


def _constants_c(
    velocity: torch.Tensor,
    jnorm: torch.Tensor,
    spl2: torch.Tensor,
    smi2: torch.Tensor,
    values: Dict[str, float],
):
    v2 = velocity * velocity
    v3 = velocity * v2
    v4 = v2 * v2
    v6 = v2 * v4
    j2 = jnorm * jnorm
    eta = values["eta"]
    eta2 = values["eta2"]
    delta = values["delta_qq"]
    seff = values["Seff"]
    c0 = (
        -0.75
        * (
            (j2 - spl2) ** 2 * v4 / eta
            - 4.0 * eta * seff * (j2 - spl2) * v3
            - 2.0
            * (
                j2
                - spl2
                + 2.0
                * (values["S1_norm_2"] - values["S2_norm_2"])
                * delta
            )
            * eta
            * v2
            + (4.0 * seff * velocity + 1.0) * eta * eta2
        )
        * jnorm
        * v2
        * (seff * velocity - 1.0)
    )
    c2 = (
        1.5
        * (smi2 - spl2)
        * jnorm
        * (
            (j2 - spl2) / eta * v2
            - 2.0 * eta * seff * velocity
            - eta
        )
        * (seff * velocity - 1.0)
        * v4
    )
    c4 = (
        -0.75
        * jnorm
        * (seff * velocity - 1.0)
        * (spl2 - smi2) ** 2
        * v6
        / eta
    )
    return c0, c2, c4


def _constants_d(
    lnorm: torch.Tensor,
    jnorm: torch.Tensor,
    spl2: torch.Tensor,
    smi2: torch.Tensor,
):
    l2 = lnorm * lnorm
    j2 = jnorm * jnorm
    spl = torch.sqrt(torch.clamp(spl2, min=0.0))
    d0 = -(j2 - (lnorm + spl) ** 2) * (j2 - (lnorm - spl) ** 2)
    d2 = -2.0 * (spl2 - smi2) * (j2 + l2 - spl2)
    d4 = -((spl2 - smi2) ** 2)
    return d0, d2, d4


def _psi(
    velocity: torch.Tensor,
    psi0: float,
    psi1: float,
    psi2: float,
    values: Dict[str, float],
):
    v2 = velocity * velocity
    return psi0 - 0.75 * values["g0"] * values["delta_qq"] * (
        1.0 + psi1 * velocity + psi2 * v2
    ) / (v2 * velocity)


def _msa_corrections(
    velocity: torch.Tensor,
    lnorm: torch.Tensor,
    jnorm: torch.Tensor,
    s32: torch.Tensor,
    spl2: torch.Tensor,
    smi2: torch.Tensor,
    values: Dict[str, float],
):
    c0, c2, c4 = _constants_c(velocity, jnorm, spl2, smi2, values)
    d0, d2, d4 = _constants_d(lnorm, jnorm, spl2, smi2)
    two_d0 = 2.0 * d0
    sd = torch.sqrt(torch.clamp(d2 * d2 - 4.0 * d0 * d4, min=0.0))
    a_theta_l = 0.5 * (
        jnorm / lnorm + lnorm / jnorm - spl2 / (jnorm * lnorm)
    )
    b_theta_l = 0.5 * (spl2 - smi2) / (jnorm * lnorm)
    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd
    nc = nc_num / nc_denom
    nd = nc_denom / two_d0
    sqrt_nc = torch.sqrt(torch.abs(nc))
    sqrt_nd = torch.sqrt(torch.abs(nd))
    phase = (
        _psi(velocity, 0.0, values["psi1"], values["psi2"], values)
        + values["psi0"]
    )
    tangent = torch.tan(phase)
    arctangent = torch.atan(tangent)
    v2 = velocity * velocity
    psi_dot = (
        -0.75
        * v2**3
        * (1.0 - velocity * values["Seff"])
        * values["sqrt_inveta"]
        * torch.sqrt(torch.clamp(spl2 - s32, min=0.0))
    )
    c2_denominator = 2.0 * d0 * sd * (d0 + d2 + d4)
    c_prefactor = torch.abs(
        (
            c4 * d0 * (two_d0 + d2 + sd)
            - c2 * d0 * (d2 + 2.0 * d4 - sd)
            - c0 * (two_d0 * d4 - (d2 + d4) * (d2 - sd))
        )
        / c2_denominator
    )
    d_prefactor = torch.abs(
        (
            -c4 * d0 * (two_d0 + d2 - sd)
            + c2 * d0 * (d2 + 2.0 * d4 + sd)
            - c0 * (-two_d0 * d4 + (d2 + d4) * (d2 + sd))
        )
        / c2_denominator
    )
    c_term = (
        c_prefactor
        * sqrt_nc
        / (nc - 1.0)
        * (arctangent - torch.atan(sqrt_nc * tangent))
        / psi_dot
    )
    d_term = (
        d_prefactor
        * sqrt_nd
        / (nd - 1.0)
        * (arctangent - torch.atan(sqrt_nd * tangent))
        / psi_dot
    )
    c_term = torch.where(
        (torch.abs(nc - 1.0) < 1.0e-14) | (psi_dot == 0.0),
        torch.zeros_like(c_term),
        c_term,
    )
    d_term = torch.where(
        (torch.abs(nd - 1.0) < 1.0e-14) | (psi_dot == 0.0),
        torch.zeros_like(d_term),
        d_term,
    )
    phiz = torch.nan_to_num(c_term + d_term)
    zeta = torch.nan_to_num(
        a_theta_l * phiz
        + 2.0
        * b_theta_l
        * d0
        * (c_term / (sd - d2) - d_term / (sd + d2))
    )
    return phiz, zeta


def _phiz(
    velocity: torch.Tensor,
    jnorm: torch.Tensor,
    values: Dict[str, float],
):
    invv = 1.0 / velocity
    invv2 = invv * invv
    l_newtonian = values["eta"] / velocity
    c1 = values["c1"]
    c12 = c1 * c1
    sav2 = values["SAv2"]
    sav = values["SAv"]
    invsav = values["invSAv"]
    invsav2 = values["invSAv2"]
    log1 = torch.log(
        torch.abs(c1 + jnorm * values["eta"] + values["eta"] * l_newtonian)
    )
    log2 = torch.log(torch.abs(c1 + jnorm * sav * velocity + sav2 * velocity))
    phiz0 = (
        jnorm
        * values["inveta4"]
        * (
            0.5 * c12
            - c1 * values["eta2"] * invv / 6.0
            - sav2 * values["eta2"] / 3.0
            - values["eta4"] * invv2 / 3.0
        )
        - 0.5
        * c1
        * values["inveta"]
        * (c12 * values["inveta4"] - sav2 * values["inveta2"])
        * log1
    )
    phiz1 = (
        -0.5
        * jnorm
        * values["inveta2"]
        * (c1 + values["eta"] * l_newtonian)
        + 0.5
        * values["inveta3"]
        * (c12 - values["eta2"] * sav2)
        * log1
    )
    phiz2 = -jnorm + sav * log2 - c1 * log1 * values["inveta"]
    phiz3 = (
        jnorm * velocity
        - values["eta"] * log1
        + c1 * log2 * invsav
    )
    phiz4 = (
        0.5 * jnorm * invsav2 * velocity * (c1 + velocity * sav2)
        - 0.5
        * (invsav2 * invsav)
        * (c12 - values["eta2"] * sav2)
        * log2
    )
    phiz5 = (
        -jnorm
        * velocity
        * (
            0.5 * c12 * invsav2**2
            - c1 * velocity * invsav2 / 6.0
            - velocity * velocity / 3.0
            - values["eta2"] * invsav2 / 3.0
        )
        + 0.5
        * c1
        * invsav2**2
        * invsav
        * (c12 - values["eta2"] * sav2)
        * log2
    )
    unshifted = (
        phiz0 * values["Omegaz0_coeff"]
        + phiz1 * values["Omegaz1_coeff"]
        + phiz2 * values["Omegaz2_coeff"]
        + phiz3 * values["Omegaz3_coeff"]
        + phiz4 * values["Omegaz4_coeff"]
        + phiz5 * values["Omegaz5_coeff"]
    )
    return torch.nan_to_num(unshifted + values["phiz_0"]), unshifted


def _zeta(velocity: torch.Tensor, values: Dict[str, float]):
    invv = 1.0 / velocity
    invv2 = invv * invv
    unshifted = values["eta"] * (
        values["Omegazeta0_coeff"] * invv2 * invv
        + values["Omegazeta1_coeff"] * invv2
        + values["Omegazeta2_coeff"] * invv
        + values["Omegazeta3_coeff"] * torch.log(velocity)
        + values["Omegazeta4_coeff"] * velocity
        + values["Omegazeta5_coeff"] * velocity * velocity
    )
    return torch.nan_to_num(unshifted + values["zeta_0"]), unshifted


def prefix(
    velocity: torch.Tensor,
    values: Dict[str, float],
    static_fallback: bool,
):
    """Return the order-sensitive MSA prefix and eager-tail intermediates."""

    lnorm = values["eta"] / velocity
    jnorm = _j_norm(lnorm, values)
    lnorm3pn = _orbital_angular_momentum_3pn(velocity, values)
    jnorm3pn = _j_norm(lnorm3pn, values)
    s32, smi2, spl2 = _roots(lnorm, jnorm, values, static_fallback)
    separated = torch.abs(smi2 - spl2) > 1.0e-5
    phiz_correction, zeta_correction = _msa_corrections(
        velocity,
        lnorm,
        jnorm,
        s32,
        spl2,
        smi2,
        values,
    )
    phiz_correction = torch.where(
        separated,
        phiz_correction,
        torch.zeros_like(phiz_correction),
    )
    zeta_correction = torch.where(
        separated,
        zeta_correction,
        torch.zeros_like(zeta_correction),
    )
    phiz, phiz_unshifted = _phiz(velocity, jnorm, values)
    zeta, zeta_unshifted = _zeta(velocity, values)
    return (
        phiz + phiz_correction,
        zeta + zeta_correction,
        jnorm3pn,
        lnorm3pn,
        s32,
        smi2,
        spl2,
        phiz_unshifted,
        phiz_correction,
        zeta_unshifted,
        zeta_correction,
    )


def pruned_jacobi(
    argument: torch.Tensor,
    parameter: torch.Tensor,
    upper_parameter: float,
) -> torch.Tensor:
    """Run converged Landen tails only on lanes which remain active."""

    parameter = torch.clamp(parameter, 0.0, upper_parameter)
    a = torch.ones_like(parameter)
    b = torch.sqrt(1.0 - parameter)
    ratios = torch.jit.annotate(list[torch.Tensor], [])
    for _ in range(5):
        next_a = 0.5 * (a + b)
        ratios.append(0.5 * (a - b) / next_a)
        b = torch.sqrt(a * b)
        a = next_a

    stable = (a == b) & (torch.sqrt(a * a) == a)
    indices = torch.nonzero((~stable).reshape(-1)).reshape(-1)
    active_a = a.reshape(-1).index_select(0, indices)
    active_b = b.reshape(-1).index_select(0, indices)
    tail_ratios = torch.jit.annotate(list[torch.Tensor], [])
    for _ in range(7):
        next_a = 0.5 * (active_a + active_b)
        tail_ratios.append(0.5 * (active_a - active_b) / next_a)
        active_b = torch.sqrt(active_a * active_b)
        active_a = next_a

    final_a = (
        a.reshape(-1)
        .clone()
        .index_copy(0, indices, active_a)
        .reshape_as(a)
    )
    complete_integral = 3.141592653589793 / (2.0 * final_a)
    reduced = (
        torch.remainder(
            argument + complete_integral,
            2.0 * complete_integral,
        )
        - complete_integral
    )
    amplitude = (2.0**12) * final_a * reduced
    active_amplitude = amplitude.reshape(-1).index_select(0, indices)
    for index in range(6, -1, -1):
        ratio = tail_ratios[index]
        active_amplitude = 0.5 * (
            active_amplitude
            + torch.asin(
                torch.clamp(
                    ratio * torch.sin(active_amplitude),
                    -1.0,
                    1.0,
                )
            )
        )

    amplitude = (
        (amplitude * (2.0**-7))
        .reshape(-1)
        .index_copy(0, indices, active_amplitude)
        .reshape_as(amplitude)
    )
    for index in range(4, -1, -1):
        ratio = ratios[index]
        amplitude = 0.5 * (
            amplitude
            + torch.asin(
                torch.clamp(
                    ratio * torch.sin(amplitude),
                    -1.0,
                    1.0,
                )
            )
        )
    return torch.sin(amplitude) ** 2
