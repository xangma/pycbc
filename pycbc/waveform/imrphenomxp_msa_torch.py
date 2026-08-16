# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch evaluation of the IMRPhenomXP multiple-scale-analysis angles.

This implements the MSA-223 branch used by IMRPhenomXP. Model initialization
and source-frame rotation are scalar host work; all frequency-dependent roots,
elliptic functions, precession angles, and opening angles execute with Torch on
the active device.

The equations follow ``LALSimIMRPhenomX_precession.c`` in LALSuite, including
the default expansion order of five. Only convention 1 is exposed here.
"""

from __future__ import annotations

import math

import torch
from scipy import special


_PI = math.pi
_TWO_PI = 2.0 * math.pi
_MASS_EQUALITY_EPS = 1.0e-16
_ROOT_SEPARATION_TOL = 1.0e-5


def _safe_sqrt(value):
    return math.sqrt(max(float(value), 0.0))


def _rotate_z(angle, vector):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y, z = vector
    return (
        x * cosine - y * sine,
        x * sine + y * cosine,
        z,
    )


def _rotate_y(angle, vector):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y, z = vector
    return (
        x * cosine + z * sine,
        y,
        -x * sine + z * cosine,
    )


def _atan2tol(y, x):
    if abs(y) < 1.0e-15 and abs(x) < 1.0e-15:
        return 0.0
    return math.atan2(y, x)


def _base_values(mass1, mass2, spin1, spin2, total_mass_seconds, f_ref):
    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    fraction1_sq = fraction1 * fraction1
    fraction2_sq = fraction2 * fraction2
    eta = fraction1 * fraction2
    q = mass2 / mass1
    chi1_norm = math.sqrt(sum(component * component for component in spin1))
    chi2_norm = math.sqrt(sum(component * component for component in spin2))
    chi1_perp = math.hypot(spin1[0], spin1[1])
    chi2_perp = math.hypot(spin2[0], spin2[1])
    spin1_perp = fraction1_sq * chi1_perp
    spin2_perp = fraction2_sq * chi2_perp
    a1 = 2.0 + 1.5 * fraction2 / fraction1
    a2 = 2.0 + 1.5 * fraction1 / fraction2
    chip = max(a1 * spin1_perp, a2 * spin2_perp) / (a1 * fraction1_sq)
    total_perp = math.hypot(
        fraction1_sq * spin1[0] + fraction2_sq * spin2[0],
        fraction1_sq * spin1[1] + fraction2_sq * spin2[1],
    )
    return {
        "qq": q,
        "invqq": 1.0 / q,
        "m1": fraction1,
        "m2": fraction2,
        "m1_2": fraction1_sq,
        "m2_2": fraction2_sq,
        "eta": eta,
        "eta2": eta * eta,
        "eta3": eta**3,
        "eta4": eta**4,
        "inveta": 1.0 / eta,
        "inveta2": 1.0 / eta**2,
        "inveta3": 1.0 / eta**3,
        "inveta4": 1.0 / eta**4,
        "sqrt_inveta": 1.0 / math.sqrt(eta),
        "delta": fraction1 - fraction2,
        "chi1_perp": chi1_perp,
        "chi2_perp": chi2_perp,
        "chi_p": chip,
        "chiTot_perp": total_perp / fraction1_sq,
        "SL": spin1[2] * fraction1_sq + spin2[2] * fraction2_sq,
        "Sperp": chip * fraction1_sq,
        "S1_norm": chi1_norm * fraction1_sq,
        "S2_norm": chi2_norm * fraction2_sq,
        "S1_norm_2": (chi1_norm * fraction1_sq) ** 2,
        "S2_norm_2": (chi2_norm * fraction2_sq) ** 2,
        "chi1x": spin1[0],
        "chi1y": spin1[1],
        "chi1z": spin1[2],
        "chi2x": spin2[0],
        "chi2y": spin2[1],
        "chi2z": spin2[2],
        "m_sec": total_mass_seconds,
        "v_ref": math.cbrt(_PI * total_mass_seconds * f_ref),
    }


def _pn_beta(a, b, values):
    return values["dotS1L"] * (a + b * values["qq"]) + values["dotS2L"] * (
        a + b / values["qq"]
    )


def _pn_sigma(a, b, values):
    return values["inveta"] * (
        a * values["dotS1S2"] - b * values["dotS1L"] * values["dotS2L"]
    )


def _pn_tau(a, b, values):
    q = values["qq"]
    return (
        q * (values["S1_norm_2"] * a - b * values["dotS1L"] * values["dotS1L"])
        + (a * values["S2_norm_2"] - b * values["dotS2L"] * values["dotS2L"]) / q
    ) / values["eta"]


def _l_coefficients_223(values):
    eta = values["eta"]
    eta2 = values["eta2"]
    delta = values["delta"]
    chi1 = values["chi1z"]
    chi2 = values["chi2z"]
    return {
        "L0": 1.0,
        "L1": 0.0,
        "L2": 1.5 + eta / 6.0,
        "L3": (
            -7.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
            + 5.0 * (chi1 + chi2) * eta
        )
        / 6.0,
        "L4": (81.0 + (-57.0 + eta) * eta) / 24.0,
        "L5": (
            -1650.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
            + 1336.0 * (chi1 + chi2) * eta
            + 511.0 * (chi1 - chi2) * delta * eta
            + 28.0 * (chi1 + chi2) * eta2
        )
        / 600.0,
        "L6": (
            10935.0 + eta * (-62001.0 + 1674.0 * eta + 7.0 * eta2 + 2214.0 * _PI * _PI)
        )
        / 1296.0,
        "L7": 0.0,
        "L8": 0.0,
        "L8L": 0.0,
    }


def _lpn_ansatz_scalar(v, values):
    x = v * v
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return (
        values["eta"]
        / v
        * (
            values["L0"]
            + values["L1"] * v
            + values["L2"] * x
            + values["L3"] * x * v
            + values["L4"] * x2
            + values["L5"] * x2 * v
            + values["L6"] * x3
            + values["L7"] * x3 * v
            + values["L8"] * x4
            + values["L8L"] * x4 * math.log(x)
        )
    )


def source_frame_parameters_msa223(
    mass1,
    mass2,
    f_ref,
    coa_phase,
    inclination,
    spin1,
    spin2,
    total_mass_seconds,
):
    """Map source-frame inputs to XP convention-1 quantities."""

    values = _base_values(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )
    values.update(_l_coefficients_223(values))
    l_ref = _lpn_ansatz_scalar(values["v_ref"], values)
    j_vector = (
        values["m1_2"] * spin1[0] + values["m2_2"] * spin2[0],
        values["m1_2"] * spin1[1] + values["m2_2"] * spin2[1],
        values["SL"] + l_ref,
    )
    j_norm = math.sqrt(sum(component * component for component in j_vector))
    theta_j_source = (
        0.0
        if j_norm < 1.0e-10
        else math.acos(max(-1.0, min(1.0, j_vector[2] / j_norm)))
    )
    phi_j_source = _atan2tol(j_vector[1], j_vector[0])
    phi_aligned = -phi_j_source

    line_of_sight = (
        math.sin(inclination) * math.sin(coa_phase),
        math.sin(inclination) * math.cos(coa_phase),
        math.cos(inclination),
    )
    rotated_sight = _rotate_y(
        -theta_j_source,
        _rotate_z(-phi_j_source, line_of_sight),
    )
    kappa = _atan2tol(rotated_sight[1], rotated_sight[0])
    alpha0 = _PI - kappa
    epsilon0 = phi_j_source - _PI

    theta_jn = (
        math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    sum(a * b for a, b in zip(j_vector, line_of_sight)) / j_norm,
                ),
            )
        )
        if j_norm > 0.0
        else 0.0
    )
    sight_x = math.sin(theta_jn)
    sight_z = math.cos(theta_jn)
    waveframe_x = (
        -math.cos(inclination) * math.sin(coa_phase),
        -math.cos(inclination) * math.cos(coa_phase),
        math.sin(inclination),
    )
    rotated_x = _rotate_z(
        -kappa,
        _rotate_y(
            -theta_j_source,
            _rotate_z(-phi_j_source, waveframe_x),
        ),
    )
    x_dot_p = rotated_x[0] * sight_z - rotated_x[2] * sight_x
    x_dot_q = rotated_x[1]
    polarization_rotation = _atan2tol(x_dot_q, x_dot_p)
    return (
        spin1[2],
        spin2[2],
        values["chi_p"],
        values["SL"],
        values["Sperp"],
        theta_jn,
        alpha0,
        epsilon0,
        phi_aligned,
        polarization_rotation,
    )


def _spin_evolution_coefficients(lnorm, jnorm, values):
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
    b = (l2 + s1n2) * q + 2.0 * lnorm * seff - 2.0 * j2 - s1n2 - s2n2 + (l2 + s2n2) / q
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


def _roots_scalar(lnorm, jnorm, values):
    b, c, d = _spin_evolution_coefficients(lnorm, jnorm, values)
    b2 = b * b
    p = c - b2 / 3.0
    q_coefficient = (2.0 / 27.0) * b2 * b - b * c / 3.0 + d
    sqrt_argument = math.sqrt(-p / 3.0) if p < 0.0 else math.nan
    if math.isfinite(sqrt_argument) and sqrt_argument != 0.0 and p != 0.0:
        acos_argument = 1.5 * q_coefficient / p / sqrt_argument
        theta = math.acos(max(-1.0, min(1.0, acos_argument))) / 3.0
    else:
        theta = math.nan
    aligned = (
        not math.isfinite(theta)
        or not math.isfinite(values["dotS1Ln"])
        or not math.isfinite(values["dotS2Ln"])
        or abs(values["dotS1Ln"]) == 1.0
        or abs(values["dotS2Ln"]) == 1.0
        or values["S1_norm_2"] == 0.0
        or values["S2_norm_2"] == 0.0
    )
    if aligned:
        smi2 = values["S_0_norm_2"]
        return 0.0, smi2, smi2 + 1.0e-9
    root1 = 2.0 * sqrt_argument * math.cos(theta - 2.0 * _TWO_PI / 3.0) - b / 3.0
    root2 = 2.0 * sqrt_argument * math.cos(theta - _TWO_PI / 3.0) - b / 3.0
    root3 = 2.0 * sqrt_argument * math.cos(theta) - b / 3.0
    roots = (root1, root2, root3)
    maximum = max(roots)
    minimum = min(roots)
    if maximum > root3 > minimum:
        middle = root3
    elif maximum > root1 > minimum:
        middle = root1
    else:
        middle = root2
    return minimum, abs(middle), abs(maximum)


def _j_norm(lnorm, values):
    return torch.sqrt(
        torch.clamp(
            lnorm * lnorm + 2.0 * lnorm * values["c1_over_eta"] + values["SAv2"],
            min=0.0,
        )
    )


def _l_norm_3pn(v, values):
    v2 = v * v
    coefficients = values["constants_L"]
    return (
        values["eta"]
        / v
        * (
            1.0
            + v2
            * (
                coefficients[0]
                + v * coefficients[1]
                + v2 * (coefficients[2] + v * coefficients[3] + v2 * coefficients[4])
            )
        )
    )


def _roots(lnorm, jnorm, values):
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
    root1 = 2.0 * sqrt_argument * torch.cos(theta - 2.0 * _TWO_PI / 3.0) - b / 3.0
    root2 = 2.0 * sqrt_argument * torch.cos(theta - _TWO_PI / 3.0) - b / 3.0
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
    static_fallback = (
        not math.isfinite(values["dotS1Ln"])
        or not math.isfinite(values["dotS2Ln"])
        or abs(values["dotS1Ln"]) == 1.0
        or abs(values["dotS2Ln"]) == 1.0
        or values["S1_norm_2"] == 0.0
        or values["S2_norm_2"] == 0.0
    )
    fallback = (p >= 0.0) | (sqrt_argument == 0.0) | ~torch.isfinite(theta)
    if static_fallback:
        fallback = torch.ones_like(fallback)
    fallback_middle = torch.full_like(middle, values["S_0_norm_2"])
    return (
        torch.where(fallback, torch.zeros_like(minimum), minimum),
        torch.where(fallback, fallback_middle, torch.abs(middle)),
        torch.where(fallback, fallback_middle + 1.0e-9, torch.abs(maximum)),
    )


def _jacobi_sn_squared(argument, parameter):
    """Evaluate ``sn(argument | parameter)**2`` with an AGM recurrence."""

    parameter = torch.clamp(parameter, 0.0, 1.0 - torch.finfo(parameter.dtype).eps)
    a = torch.ones_like(parameter)
    b = torch.sqrt(1.0 - parameter)
    ratios = []
    # Twelve fixed Landen steps converge well below float64 precision.
    for _ in range(12):
        next_a = 0.5 * (a + b)
        ratios.append(0.5 * (a - b) / next_a)
        b = torch.sqrt(a * b)
        a = next_a
    complete_integral = _PI / (2.0 * a)
    reduced = (
        torch.remainder(
            argument + complete_integral,
            2.0 * complete_integral,
        )
        - complete_integral
    )
    amplitude = (2.0**12) * a * reduced
    for ratio in reversed(ratios):
        amplitude = 0.5 * (
            amplitude + torch.asin(torch.clamp(ratio * torch.sin(amplitude), -1.0, 1.0))
        )
    return torch.sin(amplitude) ** 2


def _psi(v, psi0, psi1, psi2, values):
    v2 = v * v
    return psi0 - 0.75 * values["g0"] * values["delta_qq"] * (
        1.0 + psi1 * v + psi2 * v2
    ) / (v2 * v)


def _s_norm(v, s32, smi2, spl2, values):
    separated = torch.abs(smi2 - spl2) >= _ROOT_SEPARATION_TOL
    parameter = torch.where(
        separated,
        (smi2 - spl2) / (s32 - spl2),
        torch.zeros_like(v),
    )
    psi = _psi(v, values["psi0"], values["psi1"], values["psi2"], values)
    sn_squared = _jacobi_sn_squared(psi, parameter)
    spin_squared = spl2 + (smi2 - spl2) * sn_squared
    spin_squared = torch.where(separated, spin_squared, spl2)
    return torch.sqrt(torch.clamp(spin_squared, min=0.0))


def _constants_c(v, jnorm, spl2, smi2, values):
    v2 = v * v
    v3 = v * v2
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
            * (j2 - spl2 + 2.0 * (values["S1_norm_2"] - values["S2_norm_2"]) * delta)
            * eta
            * v2
            + (4.0 * seff * v + 1.0) * eta * eta2
        )
        * jnorm
        * v2
        * (seff * v - 1.0)
    )
    c2 = (
        1.5
        * (smi2 - spl2)
        * jnorm
        * ((j2 - spl2) / eta * v2 - 2.0 * eta * seff * v - eta)
        * (seff * v - 1.0)
        * v4
    )
    c4 = -0.75 * jnorm * (seff * v - 1.0) * (spl2 - smi2) ** 2 * v6 / eta
    return c0, c2, c4


def _constants_d(lnorm, jnorm, spl2, smi2):
    l2 = lnorm * lnorm
    j2 = jnorm * jnorm
    spl = torch.sqrt(torch.clamp(spl2, min=0.0))
    d0 = -(j2 - (lnorm + spl) ** 2) * (j2 - (lnorm - spl) ** 2)
    d2 = -2.0 * (spl2 - smi2) * (j2 + l2 - spl2)
    d4 = -((spl2 - smi2) ** 2)
    return d0, d2, d4


def _msa_corrections(v, lnorm, jnorm, s32, spl2, smi2, values):
    c0, c2, c4 = _constants_c(v, jnorm, spl2, smi2, values)
    d0, d2, d4 = _constants_d(lnorm, jnorm, spl2, smi2)
    two_d0 = 2.0 * d0
    sd = torch.sqrt(torch.clamp(d2 * d2 - 4.0 * d0 * d4, min=0.0))
    a_theta_l = 0.5 * (jnorm / lnorm + lnorm / jnorm - spl2 / (jnorm * lnorm))
    b_theta_l = 0.5 * (spl2 - smi2) / (jnorm * lnorm)
    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd
    nc = nc_num / nc_denom
    nd = nc_denom / two_d0
    sqrt_nc = torch.sqrt(torch.abs(nc))
    sqrt_nd = torch.sqrt(torch.abs(nd))
    phase = _psi(v, 0.0, values["psi1"], values["psi2"], values) + values["psi0"]
    tangent = torch.tan(phase)
    arctangent = torch.atan(tangent)
    v2 = v * v
    psi_dot = (
        -0.75
        * v2**3
        * (1.0 - v * values["Seff"])
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
        + 2.0 * b_theta_l * d0 * (c_term / (sd - d2) - d_term / (sd + d2))
    )
    return phiz, zeta


def _phiz(v, jnorm, values):
    invv = 1.0 / v
    invv2 = invv * invv
    l_newtonian = values["eta"] / v
    c1 = values["c1"]
    c12 = c1 * c1
    sav2 = values["SAv2"]
    sav = values["SAv"]
    invsav = values["invSAv"]
    invsav2 = values["invSAv2"]
    log1 = torch.log(
        torch.abs(c1 + jnorm * values["eta"] + values["eta"] * l_newtonian)
    )
    log2 = torch.log(torch.abs(c1 + jnorm * sav * v + sav2 * v))
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
        -0.5 * jnorm * values["inveta2"] * (c1 + values["eta"] * l_newtonian)
        + 0.5 * values["inveta3"] * (c12 - values["eta2"] * sav2) * log1
    )
    phiz2 = -jnorm + sav * log2 - c1 * log1 * values["inveta"]
    phiz3 = jnorm * v - values["eta"] * log1 + c1 * log2 * invsav
    phiz4 = (
        0.5 * jnorm * invsav2 * v * (c1 + v * sav2)
        - 0.5 * (invsav2 * invsav) * (c12 - values["eta2"] * sav2) * log2
    )
    phiz5 = (
        -jnorm
        * v
        * (
            0.5 * c12 * invsav2**2
            - c1 * v * invsav2 / 6.0
            - v * v / 3.0
            - values["eta2"] * invsav2 / 3.0
        )
        + 0.5 * c1 * invsav2**2 * invsav * (c12 - values["eta2"] * sav2) * log2
    )
    return torch.nan_to_num(
        phiz0 * values["Omegaz0_coeff"]
        + phiz1 * values["Omegaz1_coeff"]
        + phiz2 * values["Omegaz2_coeff"]
        + phiz3 * values["Omegaz3_coeff"]
        + phiz4 * values["Omegaz4_coeff"]
        + phiz5 * values["Omegaz5_coeff"]
        + values.get("phiz_0", 0.0)
    )


def _zeta(v, values):
    invv = 1.0 / v
    invv2 = invv * invv
    return torch.nan_to_num(
        values["eta"]
        * (
            values["Omegazeta0_coeff"] * invv2 * invv
            + values["Omegazeta1_coeff"] * invv2
            + values["Omegazeta2_coeff"] * invv
            + values["Omegazeta3_coeff"] * torch.log(v)
            + values["Omegazeta4_coeff"] * v
            + values["Omegazeta5_coeff"] * v * v
        )
        + values.get("zeta_0", 0.0)
    )


def msa_angles(v, values):
    """Return raw ``(phi_z, zeta, cos(theta_LJ))`` tensors at velocity ``v``."""

    lnorm = values["eta"] / v
    jnorm = _j_norm(lnorm, values)
    lnorm3pn = _l_norm_3pn(v, values)
    jnorm3pn = _j_norm(lnorm3pn, values)
    s32, smi2, spl2 = _roots(lnorm, jnorm, values)
    snorm = _s_norm(v, s32, smi2, spl2, values)
    separated = torch.abs(smi2 - spl2) > _ROOT_SEPARATION_TOL
    phiz_correction, zeta_correction = _msa_corrections(
        v,
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
    cos_beta = (
        0.5
        * (jnorm3pn * jnorm3pn + lnorm3pn * lnorm3pn - snorm * snorm)
        / (lnorm3pn * jnorm3pn)
    )
    return (
        _phiz(v, jnorm, values) + phiz_correction,
        _zeta(v, values) + zeta_correction,
        torch.clamp(torch.nan_to_num(cos_beta, nan=1.0), -1.0, 1.0),
    )


def build_msa_state(
    mass1,
    mass2,
    spin1,
    spin2,
    total_mass_seconds,
    f_ref,
):
    """Initialize scalar coefficients for the MSA-223 angle evolution."""

    values = _base_values(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )
    values["delta_qq"] = (1.0 - values["qq"]) / (1.0 + values["qq"])
    values["delta2_qq"] = values["delta_qq"] ** 2
    values["delta3_qq"] = values["delta_qq"] ** 3
    values["delta4_qq"] = values["delta_qq"] ** 4
    eta = values["eta"]
    eta2 = values["eta2"]
    eta3 = values["eta3"]
    eta4 = values["eta4"]
    q = values["qq"]
    spin1_vector = tuple(component * eta / q for component in spin1)
    spin2_vector = tuple(component * eta * q for component in spin2)
    total_spin = tuple(a + b for a, b in zip(spin1_vector, spin2_vector))
    values["S_0_norm_2"] = sum(component * component for component in total_spin)
    values["dotS1L"] = spin1_vector[2]
    values["dotS2L"] = spin2_vector[2]
    values["dotS1S2"] = sum(a * b for a, b in zip(spin1_vector, spin2_vector))
    spin1_norm = math.sqrt(sum(component * component for component in spin1_vector))
    spin2_norm = math.sqrt(sum(component * component for component in spin2_vector))
    values["dotS1Ln"] = values["dotS1L"] / spin1_norm if spin1_norm > 0.0 else math.nan
    values["dotS2Ln"] = values["dotS2L"] / spin2_norm if spin2_norm > 0.0 else math.nan

    nonspin = (
        3.0 / 2.0,
        1.0 / 6.0,
        27.0 / 8.0,
        -19.0 / 8.0,
        1.0 / 24.0,
        135.0 / 16.0,
        -6889.0 / 144.0 + 41.0 * _PI * _PI / 24.0,
        31.0 / 24.0,
        7.0 / 1296.0,
    )
    spin_orbit = (
        -14.0 / 6.0,
        -3.0 / 2.0,
        -11.0 / 2.0,
        133.0 / 72.0,
        -33.0 / 8.0,
        7.0 / 4.0,
    )
    values["constants_L"] = (
        nonspin[0] + eta * nonspin[1],
        _pn_beta(spin_orbit[0], spin_orbit[1], values),
        nonspin[2] + eta * nonspin[3] + eta2 * nonspin[4],
        _pn_beta(
            spin_orbit[2] + spin_orbit[3] * eta,
            spin_orbit[4] + spin_orbit[5] * eta,
            values,
        ),
        nonspin[5] + nonspin[6] * eta + nonspin[7] * eta2 + nonspin[8] * eta3,
    )
    values["Seff"] = (1.0 + q) * values["dotS1L"] + (1.0 + 1.0 / q) * values["dotS2L"]
    values["Seff2"] = values["Seff"] ** 2
    values["v_0"] = values["v_ref"]
    values["v_0_2"] = values["v_0"] ** 2
    l0norm = eta / values["v_0"]
    values["L_0_norm"] = l0norm
    j0_vector = (total_spin[0], total_spin[1], l0norm + total_spin[2])
    values["J_0_norm"] = math.sqrt(
        sum(component * component for component in j0_vector)
    )
    values["SAv2"] = values["S_0_norm_2"] or 1.0e-9
    values["c1_over_eta"] = (
        0.5 * (values["J_0_norm"] ** 2 - l0norm * l0norm - values["SAv2"]) / l0norm
        if l0norm != 0.0
        else 0.0
    )
    s32, smi2, spl2 = _roots_scalar(l0norm, values["J_0_norm"], values)
    values.update(
        {
            "S32": s32,
            "Smi2": smi2,
            "Spl2": spl2,
            "Spl2pSmi2": spl2 + smi2,
            "Spl2mSmi2": spl2 - smi2,
            "Spl": _safe_sqrt(spl2),
            "Smi": _safe_sqrt(smi2),
        }
    )
    values["SAv2"] = max(0.5 * values["Spl2pSmi2"], 1.0e-30)
    values["SAv"] = math.sqrt(values["SAv2"])
    values["invSAv2"] = 1.0 / values["SAv2"]
    values["invSAv"] = 1.0 / values["SAv"]
    values["c1"] = (
        0.5
        * (values["J_0_norm"] ** 2 - l0norm * l0norm - values["SAv2"])
        / l0norm
        * eta
        if l0norm != 0.0
        else 0.0
    )
    values["c12"] = values["c1"] ** 2
    values["c1_over_eta"] = values["c1"] / eta

    one_minus_q_squared = 1.0 - q * q + _MASS_EQUALITY_EPS
    one_minus_q_fourth = (1.0 - q) ** 2 + _MASS_EQUALITY_EPS
    common = values["c1"] * (1.0 + q)
    values["S1L_pav"] = (common - q * eta * values["Seff"]) / (
        eta * one_minus_q_squared
    )
    values["S2L_pav"] = (
        -q * (common - eta * values["Seff"]) / (eta * one_minus_q_squared)
    )
    values["S1S2_pav"] = 0.5 * values["SAv2"] - 0.5 * (
        values["S1_norm_2"] + values["S2_norm_2"]
    )
    root_range_squared = values["Spl2mSmi2"] ** 2
    values["S1Lsq_pav"] = values["S1L_pav"] ** 2 + (
        root_range_squared * values["v_0_2"]
    ) / (32.0 * eta2 * one_minus_q_fourth)
    values["S2Lsq_pav"] = values["S2L_pav"] ** 2 + (
        q * q * root_range_squared * values["v_0_2"]
    ) / (32.0 * eta2 * one_minus_q_fourth)
    values["S1LS2L_pav"] = values["S1L_pav"] * values["S2L_pav"] - (
        q * root_range_squared * values["v_0_2"]
    ) / (32.0 * eta2 * one_minus_q_fourth)

    domega_nonspin = (
        96.0 / 5.0,
        -1486.0 / 35.0,
        -264.0 / 5.0,
        384.0 * _PI / 5.0,
        34103.0 / 945.0,
        13661.0 / 105.0,
        944.0 / 15.0,
        -4159.0 * _PI / 35.0,
        -2268.0 * _PI / 5.0,
    )
    domega_spin_orbit = (
        -904.0 / 5.0,
        -120.0,
        -62638.0 / 105.0,
        4636.0 / 5.0,
        -6472.0 / 35.0,
        3372.0 / 5.0,
    )
    domega_spin_spin = (-494.0 / 5.0, -1442.0 / 5.0, -233.0 / 5.0, -719.0 / 5.0)
    values["a0"] = eta * domega_nonspin[0]
    values["a2"] = eta * (domega_nonspin[1] + eta * domega_nonspin[2])
    values["a3"] = eta * (
        domega_nonspin[3] + _pn_beta(domega_spin_orbit[0], domega_spin_orbit[1], values)
    )
    values["a4"] = eta * (
        domega_nonspin[4]
        + eta * (domega_nonspin[5] + eta * domega_nonspin[6])
        + _pn_sigma(domega_spin_spin[0], domega_spin_spin[1], values)
        + _pn_tau(domega_spin_spin[2], domega_spin_spin[3], values)
    )
    values["a5"] = eta * (
        domega_nonspin[7]
        + eta * domega_nonspin[8]
        + _pn_beta(
            domega_spin_orbit[2] + eta * domega_spin_orbit[3],
            domega_spin_orbit[4] + eta * domega_spin_orbit[5],
            values,
        )
    )
    values["g0"] = 1.0 / values["a0"]
    values["g2"] = -values["a2"] / values["a0"] ** 2
    values["g3"] = -values["a3"] / values["a0"] ** 2
    values["g4"] = (
        -(values["a4"] * values["a0"] - values["a2"] ** 2) / values["a0"] ** 3
    )
    values["g5"] = (
        -(values["a5"] * values["a0"] - 2.0 * values["a3"] * values["a2"])
        / values["a0"] ** 3
    )

    delta = values["delta_qq"]
    c1nu = values["c1_over_eta"]
    c1nu2 = c1nu * c1nu
    one_plus_q_squared = (1.0 + q) ** 2
    q2 = q * q
    one_minus_q2_squared = (1.0 - q2) ** 2
    one_minus_q_fourth = (1.0 - q) ** 4
    values["psi1"] = (
        0.0
        if abs(delta * delta) < 1.0e-14
        else 3.0 * (2.0 * eta2 * values["Seff"] - values["c1"]) / (eta * delta * delta)
    )
    del1 = 4.0 * c1nu2 * one_plus_q_squared
    del2 = 8.0 * c1nu * q * (1.0 + q) * values["Seff"]
    del3 = 4.0 * (one_minus_q2_squared * values["S1_norm_2"] - q2 * values["Seff2"])
    del4 = 4.0 * c1nu2 * q2 * one_plus_q_squared
    del5 = 8.0 * c1nu * q2 * (1.0 + q) * values["Seff"]
    del6 = 4.0 * (one_minus_q2_squared * values["S2_norm_2"] - q2 * values["Seff2"])
    values["Delta"] = _safe_sqrt(abs((del1 - del2 - del3) * (del4 - del5 - del6)))
    if abs(one_minus_q_fourth) < 1.0e-14:
        values["psi2"] = 3.0 * values["g2"] / values["g0"]
    else:
        u1 = 3.0 * values["g2"] / values["g0"]
        u2 = 0.75 * one_plus_q_squared / one_minus_q_fourth
        u3 = -20.0 * c1nu2 * q2 * one_plus_q_squared
        u4 = (
            2.0
            * one_minus_q2_squared
            * (
                q * (2.0 + q) * values["S1_norm_2"]
                + (1.0 + 2.0 * q) * values["S2_norm_2"]
                - 2.0 * q * values["SAv2"]
            )
        )
        u5 = 4.0 * q2 * (7.0 + 6.0 * q + 7.0 * q2) * c1nu * values["Seff"]
        u6 = 2.0 * q2 * (3.0 + 4.0 * q + 3.0 * q2) * values["Seff2"]
        values["psi2"] = u1 + u2 * (u3 + u4 + u5 - u6 + q * values["Delta"])

    root_range = values["Spl2"] - values["Smi2"]
    cp = values["Spl2"] * eta2 - values["c12"]
    cm = values["Smi2"] * eta2 - values["c12"]
    sqrt_cpcm = _safe_sqrt(abs(cp * cm))
    a1dd = 0.5 + 0.75 / eta
    a2dd = -0.75 * values["Seff"] / eta
    d2rmsq = (cp - sqrt_cpcm) / eta2
    d4rmsq = -0.5 * root_range * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)
    spin_norm_difference = values["S1_norm_2"] - values["S2_norm_2"]
    aw = (
        -3.0
        * (1.0 + q)
        / q
        * (
            2.0 * (1.0 + q) * eta2 * values["Seff"] * values["c1"]
            - (1.0 + q) * values["c12"]
            + (1.0 - q) * eta2 * spin_norm_difference
        )
    )
    cw = 3.0 / 32.0 / eta * root_range**2
    dw = 4.0 * cp - 4.0 * d2rmsq * eta2
    hw = -2.0 * (2.0 * d2rmsq - root_range) * values["c1"]
    fw = root_range * d2rmsq - d4rmsq - 0.25 * root_range**2
    ad = aw / dw if dw != 0.0 else 0.0
    hd = hw / dw if dw != 0.0 else 0.0
    cd = cw / dw if dw != 0.0 else 0.0
    fd = fw / dw if dw != 0.0 else 0.0
    gd = (
        3.0 / 16.0 / eta3 * root_range**2 * (values["c1"] - eta2 * values["Seff"]) / dw
        if dw != 0.0
        else 0.0
    )
    hd2 = hd * hd
    adfd = ad * fd
    values["Omegaz0"] = a1dd + ad
    values["Omegaz1"] = a2dd - ad * values["Seff"] - ad * hd
    values["Omegaz2"] = ad * hd * values["Seff"] + cd - adfd + ad * hd2
    values["Omegaz3"] = (adfd - cd - ad * hd2) * (values["Seff"] + hd) + adfd * hd
    values["Omegaz4"] = (cd + ad * hd2 - 2.0 * adfd) * (
        hd * values["Seff"] + hd2 - fd
    ) - ad * fd * fd
    values["Omegaz5"] = (
        (cd - adfd + ad * hd2) * fd * (values["Seff"] + 2.0 * hd)
        - (cd + ad * hd2 - 2.0 * adfd) * hd2 * (values["Seff"] + hd)
        - adfd * fd * hd
    )
    g0 = values["g0"]
    values["Omegaz0_coeff"] = 3.0 * g0 * values["Omegaz0"]
    values["Omegaz1_coeff"] = 3.0 * g0 * values["Omegaz1"]
    values["Omegaz2_coeff"] = 3.0 * (
        g0 * values["Omegaz2"] + values["g2"] * values["Omegaz0"]
    )
    values["Omegaz3_coeff"] = 3.0 * (
        g0 * values["Omegaz3"]
        + values["g2"] * values["Omegaz1"]
        + values["g3"] * values["Omegaz0"]
    )
    values["Omegaz4_coeff"] = 3.0 * (
        g0 * values["Omegaz4"]
        + values["g2"] * values["Omegaz2"]
        + values["g3"] * values["Omegaz1"]
        + values["g4"] * values["Omegaz0"]
    )
    values["Omegaz5_coeff"] = 0.0
    c1eta2 = values["c1"] / eta2
    values["Omegazeta0"] = values["Omegaz0"]
    values["Omegazeta1"] = values["Omegaz1"] + values["Omegaz0"] * c1eta2
    values["Omegazeta2"] = values["Omegaz2"] + values["Omegaz1"] * c1eta2
    values["Omegazeta3"] = values["Omegaz3"] + values["Omegaz2"] * c1eta2 + gd
    values["Omegazeta4"] = (
        values["Omegaz4"] + values["Omegaz3"] * c1eta2 - gd * values["Seff"] - gd * hd
    )
    values["Omegazeta5"] = (
        values["Omegaz5"]
        + values["Omegaz4"] * c1eta2
        + gd * hd * values["Seff"]
        + gd * (hd2 - fd)
    )
    values["Omegazeta0_coeff"] = -g0 * values["Omegazeta0"]
    values["Omegazeta1_coeff"] = -1.5 * g0 * values["Omegazeta1"]
    values["Omegazeta2_coeff"] = -3.0 * (
        g0 * values["Omegazeta2"] + values["g2"] * values["Omegazeta0"]
    )
    values["Omegazeta3_coeff"] = 3.0 * (
        g0 * values["Omegazeta3"]
        + values["g2"] * values["Omegazeta1"]
        + values["g3"] * values["Omegazeta0"]
    )
    values["Omegazeta4_coeff"] = 3.0 * (
        g0 * values["Omegazeta4"]
        + values["g2"] * values["Omegazeta2"]
        + values["g3"] * values["Omegazeta1"]
        + values["g4"] * values["Omegazeta0"]
    )
    values["Omegazeta5_coeff"] = 0.0

    values["psi0"] = 0.0
    if abs(values["Smi2"] - values["Spl2"]) >= _ROOT_SEPARATION_TOL:
        modulus = _safe_sqrt(
            (values["Smi2"] - values["Spl2"]) / (values["S32"] - values["Spl2"])
        )
        ratio = (values["S_0_norm_2"] - values["Spl2"]) / (
            values["Smi2"] - values["Spl2"]
        )
        orbital = (0.0, 0.0, l0norm)
        cross = (
            orbital[1] * spin1_vector[2] - orbital[2] * spin1_vector[1],
            orbital[2] * spin1_vector[0] - orbital[0] * spin1_vector[2],
            orbital[0] * spin1_vector[1] - orbital[1] * spin1_vector[0],
        )
        volume = sum(a * b for a, b in zip(cross, spin2_vector))
        volume_sign = float((volume > 0.0) - (volume < 0.0))
        psi_v0 = float(_psi(values["v_0"], 0.0, values["psi1"], values["psi2"], values))
        if ratio > 1.0 and ratio - 1.0 < 1.0e-5:
            amplitude = math.asin(volume_sign)
        elif ratio < 0.0 and ratio > -1.0e-5:
            amplitude = 0.0
        elif 0.0 <= ratio <= 1.0:
            amplitude = math.asin(volume_sign * math.sqrt(ratio))
        else:
            amplitude = None
        if amplitude is not None:
            values["psi0"] = float(
                special.ellipkinc(amplitude, modulus * modulus) - psi_v0
            )

    values["phiz_0"] = 0.0
    values["zeta_0"] = 0.0
    reference_velocity = torch.tensor([values["v_0"]], dtype=torch.float64)
    phiz0, zeta0, _ = msa_angles(reference_velocity, values)
    values["phiz_0"] = -float(phiz0[0])
    values["zeta_0"] = -float(zeta0[0])
    values.update(_l_coefficients_223(values))
    return values


__all__ = [
    "build_msa_state",
    "msa_angles",
    "source_frame_parameters_msa223",
]
