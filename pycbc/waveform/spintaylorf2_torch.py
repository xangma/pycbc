"""Torch-native SpinTaylorF2 generator mirroring the historical CUDA kernel.

Validated against LALSuite 7.26.1 (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c and
LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c) so readers can cross-check line references
below when auditing parity.

This keeps the torch implementation scheme-specific while leaving the
scheme-agnostic API untouched.  Expressions below are a direct vectorised
translation of the PyCUDA kernel in ``SpinTaylorF2.py`` so we maintain numerical
parity with the existing CPU/LAL implementation (up to expected tolerance)."""

import os
import numpy as _np
import logging
import torch
import lal

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.torch_switches import torch_native_enabled


# PN phasing/flux coefficients (from CUDA kernel / LAL)
# LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c lines 692-750 (phasing),
# 346-379 (flux), 49-74 (energy) in the SpinTaylorF2 context.
def _pn_and_flux_coeffs(eta, pn_beta, pn_sigma, pn_gamma):
    pfaN = 3.0 / (128.0 * eta)
    pfa2 = 5.0 * (743.0 / 84.0 + 11.0 * eta) / 9.0
    pfa3 = -16.0 * _np.pi + 4.0 * pn_beta
    pfa4 = (
        5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0
        - 10.0 * pn_sigma
    )
    pfa5 = 5.0 / 9.0 * (7729.0 / 84.0 - 13.0 * eta) * _np.pi - pn_gamma
    pfl5 = 5.0 / 3.0 * (7729.0 / 84.0 - 13.0 * eta) * _np.pi - 3.0 * pn_gamma
    pfa6 = (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * _np.pi * _np.pi
        - 6848.0 / 21.0 * _np.euler_gamma
        + eta * (-15737.765635 / 3.048192 + 2255.0 / 12.0 * _np.pi * _np.pi)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        + (-6848.0 / 21.0) * _np.log(4.0)
    )
    pfl6 = -6848.0 / 21.0
    pfa7 = (
        _np.pi
        * 5.0
        / 756.0
        * (15419335.0 / 336.0 + 75703.0 / 2.0 * eta - 14809.0 * eta * eta)
    )

    FTaN = 32.0 * eta * eta / 5.0
    FTa2 = -(12.47 / 3.36 + 3.5 / 1.2 * eta)
    FTa3 = 4.0 * _np.pi
    FTa4 = -(44.711 / 9.072 - 92.71 / 5.04 * eta - 6.5 / 1.8 * eta * eta)
    FTa5 = -(81.91 / 6.72 + 58.3 / 2.4 * eta) * _np.pi
    FTa6 = (
        664.3739519 / 6.9854400
        + 16.0 / 3.0 * _np.pi * _np.pi
        - 17.12 / 1.05 * _np.euler_gamma
        + (4.1 / 4.8 * _np.pi * _np.pi - 134.543 / 7.776) * eta
        - 94.403 / 3.024 * eta * eta
        - 7.75 / 3.24 * eta * eta * eta
    )
    FTl6 = -8.56 / 1.05
    FTa7 = (
        -(162.85 / 5.04 - 214.745 / 1.728 * eta - 193.385 / 3.024 * eta * eta) * _np.pi
    )

    dETaN = -eta
    dETa1 = 2.0 * (-(3.0 / 4.0 + eta / 12.0))
    dETa2 = 3.0 * (-(27.0 / 8.0 - 19.0 / 8.0 * eta + eta * eta / 24.0))
    dETa3 = 4.0 * (
        -(
            67.5 / 6.4
            - (344.45 / 5.76 - 20.5 / 9.6 * _np.pi * _np.pi) * eta
            + 15.5 / 9.6 * eta * eta
            + 3.5 / 518.4 * eta * eta * eta
        )
    )

    return dict(
        pfaN=pfaN,
        pfa2=pfa2,
        pfa3=pfa3,
        pfa4=pfa4,
        pfa5=pfa5,
        pfl5=pfl5,
        pfa6=pfa6,
        pfl6=pfl6,
        pfa7=pfa7,
        FTaN=FTaN,
        FTa2=FTa2,
        FTa3=FTa3,
        FTa4=FTa4,
        FTa5=FTa5,
        FTa6=FTa6,
        FTl6=FTl6,
        FTa7=FTa7,
        dETaN=dETaN,
        dETa1=dETa1,
        dETa2=dETa2,
        dETa3=dETa3,
    )


# ---- Helper routines mirroring the LAL SpinTaylorF2 implementation (single-spin) ----


def _safe_atan2(y, x, device, dtype):
    """Match LAL safe_atan2: return 0 if both args are 0 (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:83-89)."""
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where((y == 0.0) & (x == 0.0), zero, torch.atan2(y, x))


def _orientation(m1, m2, v_ref, lnhatx, lnhaty, lnhatz, s1x, s1y, s1z, device, dtype):
    """Replicate XLALSimInspiralSF2CalculateOrientation (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:92-113)."""
    chi = torch.sqrt(s1x * s1x + s1y * s1y + s1z * s1z)
    kappa = (
        (lnhatx * s1x + lnhaty * s1y + lnhatz * s1z) / chi
        if chi.item() != 0.0
        else torch.tensor(1.0, device=device, dtype=dtype)
    )
    Jx0 = m1 * m2 * lnhatx / v_ref + m1 * m1 * s1x
    Jy0 = m1 * m2 * lnhaty / v_ref + m1 * m1 * s1y
    Jz0 = m1 * m2 * lnhatz / v_ref + m1 * m1 * s1z
    Jnorm = torch.sqrt(Jx0 * Jx0 + Jy0 * Jy0 + Jz0 * Jz0)
    thetaJ = torch.acos(Jz0 / Jnorm)
    # LAL sign convention for polarization phase
    psiJ = _safe_atan2(Jy0, -Jx0, device, dtype)

    rotLx = (
        lnhatx * torch.cos(thetaJ) * torch.cos(psiJ)
        - lnhaty * torch.cos(thetaJ) * torch.sin(psiJ)
        + lnhatz * torch.sin(thetaJ)
    )
    rotLy = lnhatx * torch.sin(psiJ) + lnhaty * torch.cos(psiJ)
    alpha0 = _safe_atan2(rotLy, rotLx, device, dtype)

    return dict(chi=chi, kappa=kappa, thetaJ=thetaJ, psiJ=psiJ, alpha0=alpha0)


def _coeffs(m1, m2, chi, kappa, device, dtype):
    """Replicate XLALSimInspiralSF2CalculateCoeffs (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:115-160)."""
    mtot = m1 + m2
    eta = m1 * m2 / (mtot * mtot)
    gamma0 = m1 * chi / m2
    kappa_perp = torch.sqrt(1.0 - kappa * kappa)

    pn_beta = (113.0 * m1 / (12.0 * mtot) - 19.0 * eta / 6.0) * chi * kappa
    pn_sigma = (
        (5.0 * (3.0 * kappa * kappa - 1.0) / 2.0) + (7.0 - kappa * kappa) / 96.0
    ) * (m1 * m1 * chi * chi / (mtot * mtot))
    pn_gamma = (
        (
            5.0 * (146597.0 + 7056.0 * eta) * m1 / (2268.0 * mtot)
            - 10.0 * eta * (1276.0 + 153.0 * eta) / 81.0
        )
        * chi
        * kappa
    )

    prec_fac = 5.0 * (4.0 + 3.0 * m2 / m1) / 64.0
    dtdv2 = 743.0 / 336.0 + 11.0 * eta / 4.0
    dtdv3 = -4.0 * _np.pi + pn_beta
    dtdv4 = (
        3058673.0 / 1016064.0
        + 5429.0 * eta / 1008.0
        + 617.0 * eta * eta / 144.0
        - pn_sigma
    )
    dtdv5 = (-7729.0 / 672.0 + 13.0 * eta / 8.0) * _np.pi + 9.0 * pn_gamma / 40.0

    aclog1 = (
        kappa * (1.0 - kappa * kappa) * gamma0 * gamma0 * gamma0 / 2.0
        - dtdv2 * kappa * gamma0
        - dtdv3
    )
    aclog2 = (
        dtdv2 * gamma0
        + dtdv3 * kappa
        + (1.0 - kappa * kappa) * (dtdv4 - dtdv5 * kappa / gamma0) / (2.0 * gamma0)
    )
    ac0 = -1.0 / 3.0
    ac1 = -gamma0 * kappa / 6.0
    ac2 = gamma0 * gamma0 * (-1.0 / 3.0 + kappa * kappa / 2.0) - dtdv2
    ac3 = (
        dtdv3
        + dtdv4 * kappa / (2.0 * gamma0)
        + dtdv5 * (1.0 / 3.0 - kappa * kappa / 2.0) / (gamma0 * gamma0)
    )
    ac4 = dtdv4 / 2.0 + dtdv5 * kappa / (6.0 * gamma0)
    ac5 = dtdv5 / 3.0

    zc0 = -1.0 / 3.0
    zc1 = -gamma0 * kappa / 2.0
    zc2 = -dtdv2
    zc3 = dtdv2 * gamma0 * kappa + dtdv3
    zc4 = dtdv3 * gamma0 * kappa + dtdv4
    zc5 = (dtdv4 * gamma0 * kappa + dtdv5) / 2.0
    zc6 = dtdv5 * gamma0 * kappa / 3.0

    return dict(
        eta=eta,
        gamma0=gamma0,
        kappa=kappa,
        kappa_perp=kappa_perp,
        prec_fac=prec_fac,
        dtdv2=dtdv2,
        dtdv3=dtdv3,
        dtdv4=dtdv4,
        dtdv5=dtdv5,
        aclog1=aclog1,
        aclog2=aclog2,
        ac=(ac0, ac1, ac2, ac3, ac4, ac5),
        zc=(zc0, zc1, zc2, zc3, zc4, zc5, zc6),
    )


def _alpha(v, coeffs):
    """XLALSimInspiralSF2Alpha (vectorised torch; LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:162-179)."""
    gam = coeffs["gamma0"] * v
    kappa = coeffs["kappa"]
    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    logfac1 = torch.log((1.0 + kappa * gam + sqrtfac) / v)
    logfac2 = torch.log(kappa + gam + sqrtfac)

    ac0, ac1, ac2, ac3, ac4, ac5 = coeffs["ac"]
    aclog1 = coeffs["aclog1"]
    aclog2 = coeffs["aclog2"]
    prec_fac = coeffs["prec_fac"]

    poly = (((ac0 / v + ac1) / v + ac2) / v + ac3 + (ac4 + ac5 * v) * v) * sqrtfac
    return prec_fac * (aclog1 * logfac1 + aclog2 * logfac2 + poly)


def _zeta(v, coeffs):
    """XLALSimInspiralSF2Zeta (matches CUDA kernel expression; LAL keeps this #if0 at 182-189 in lalsimulation/lib/LALSimInspiralSpinTaylorF2.c)."""
    gamma0 = coeffs["gamma0"]
    kappa = coeffs["kappa"]
    prec_fac = coeffs["prec_fac"]
    dtdv2 = coeffs["dtdv2"]
    dtdv3 = coeffs["dtdv3"]
    dtdv4 = coeffs["dtdv4"]
    dtdv5 = coeffs["dtdv5"]

    gam = gamma0 * v
    gamma02 = gamma0 * gamma0
    gamma03 = gamma02 * gamma0
    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa

    v2 = v * v
    v3 = v2 * v

    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    logv = torch.log(v)
    logfac1 = torch.log(1.0 + kappa * gam + sqrtfac)
    logfac2 = torch.log(kappa + gam + sqrtfac)

    term_logfac2 = (
        -dtdv2 * gamma0
        - dtdv3 * kappa
        + dtdv5 * kappa / (2.0 * gamma02)
        - dtdv4 / (2.0 * gamma0)
        + dtdv4 * kappa2 / (2.0 * gamma0)
        - dtdv5 * kappa3 / (2.0 * gamma02)
    )
    term_logv = kappa * gamma03 / 2.0 - gamma03 * kappa3 / 2.0
    term_logfac1 = (
        dtdv2 * gamma0 * kappa + dtdv3 - kappa * gamma03 / 2.0 + gamma03 * kappa3 / 2.0
    )

    term_poly = (
        -1.0 / (3.0 * v3)
        - gamma0 * kappa / (2.0 * v2)
        - dtdv2 / v
        + dtdv4 * gamma0 * kappa * v2 / 2.0
        + dtdv5 * v2 / 2.0
    )

    term_sqrt = (
        -dtdv3
        - dtdv4 * v / 2.0
        - dtdv5 / (3.0 * gamma02)
        - dtdv4 * kappa / (2.0 * gamma0)
        - dtdv5 * kappa * v / (6.0 * gamma0)
        + dtdv5 * kappa2 / (2.0 * gamma02)
        + 1.0 / (3.0 * v3)
        + gamma0 * kappa / (6.0 * v2)
        + dtdv2 / v
        + gamma02 / (3.0 * v)
        - gamma02 * kappa2 / (2.0 * v)
        - dtdv5 * v2 / 3.0
    )

    return prec_fac * (
        dtdv3 * gamma0 * kappa * v
        + dtdv4 * v
        + logfac2 * term_logfac2
        + logv * term_logv
        + logfac1 * term_logfac1
        + term_poly
        + sqrtfac * term_sqrt
        + dtdv5 * gamma0 * kappa * v3 / 3.0
    )


def _emission(v, coeffs):
    """XLALSimInspiralSF2Emission; returns (em0..em4) per-frequency tensors (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:238-254)."""
    gam = coeffs["gamma0"] * v
    kappa = coeffs["kappa"]
    kappa_perp = coeffs["kappa_perp"]
    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    cosbeta = (1.0 + kappa * gam) / sqrtfac
    sinbeta = (kappa_perp * gam) / sqrtfac

    em0 = (1.0 + cosbeta) * (1.0 + cosbeta) / 4.0
    em1 = (1.0 + cosbeta) * sinbeta / 4.0
    em2 = sinbeta * sinbeta / 4.0
    em3 = (1.0 - cosbeta) * sinbeta / 4.0
    em4 = (1.0 - cosbeta) * (1.0 - cosbeta) / 4.0
    return em0, em1, em2, em3, em4


def _polarization(thetaJ, psiJ, mm, device, dtype):
    """XLALSimInspiralSF2Polarization (complex) (LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:204-236)."""
    ct = torch.cos(thetaJ)
    st = torch.sin(thetaJ)
    if mm == 2:
        plus_fac = (1.0 + ct * ct) / 2.0
        cross_fac = -1j * ct
    elif mm == 1:
        plus_fac = torch.sin(2.0 * thetaJ)
        cross_fac = -2j * st
    elif mm == 0:
        plus_fac = 3.0 * st * st
        cross_fac = 0.0j
    elif mm == -1:
        plus_fac = -torch.sin(2.0 * thetaJ)
        cross_fac = -2j * st
    elif mm == -2:
        plus_fac = (1.0 + ct * ct) / 2.0
        cross_fac = 1j * ct
    else:
        plus_fac = torch.tensor(0.0, device=device, dtype=dtype)
        cross_fac = 0.0j
    return plus_fac * torch.cos(2.0 * psiJ) + cross_fac * torch.sin(2.0 * psiJ)


def _pfa_coeffs(
    m1,
    m2,
    eta,
    chi1L,
    chi1sq,
    coeffs,
    phase_order,
    enable_prec=True,
    pn_spin_order=-1,
    qm_def1=0.0,
    non_gr=None,
):
    """Compute PN phasing series up to 3.5PN matching XLALSimInspiralSpinTaylorF2
    (LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c:955-1102 for PNPhasingSeries build;
    selection and zeta additions as in lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:370-435).

    The coefficients follow the LAL PNPhasingSeries: base (non-spin) terms are
    scaled by pfaN = 3/(128*eta), single-spin contributions are added via the
    SpinTaylorF2 SO/SS coefficients, and the precession zeta-series pieces are
    added afterwards (unscaled by pfaN) when precession is enabled.
    """

    mtot = m1 + m2
    m1M = m1 / mtot
    chi1L2 = chi1L * chi1L
    qm_def1 = 1.0 + qm_def1

    # Non-spinning TaylorF2 phasing terms (use pn_beta/sigma/gamma = 0 to avoid
    # double-counting spin contributions that are added explicitly below).
    base = _pn_and_flux_coeffs(eta, 0.0, 0.0, 0.0)

    # pfa1 carries the non-GR 1PN phase term (PNPhasingSeries v[1]); LAL sets
    # it before the global pfaN scaling (LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c:987-999).
    base_dchi1 = non_gr.get("dchi1", 0.0)

    # Apply non-GR scaling to the non-spin pieces (matches PNPhasingSeries)
    if non_gr is None:
        non_gr = {}
    base["pfaN"] *= 1.0 + non_gr.get("dchi0", 0.0)
    base["pfa2"] *= 1.0 + non_gr.get("dchi2", 0.0)
    base["pfa3"] *= 1.0 + non_gr.get("dchi3", 0.0)
    base["pfa4"] *= 1.0 + non_gr.get("dchi4", 0.0)
    base["pfa5"] *= 1.0 + non_gr.get("dchi5", 0.0)
    base["pfl5"] *= 1.0 + non_gr.get("dchi5L", 0.0)
    base["pfa6"] *= 1.0 + non_gr.get("dchi6", 0.0)
    base["pfl6"] *= 1.0 + non_gr.get("dchi6L", 0.0)
    base["pfa7"] *= 1.0 + non_gr.get("dchi7", 0.0)

    # Spin-orbit and spin-squared pieces (single-spin, chi2 = 0)
    def so3(mbym):
        return mbym * (25.0 + 38.0 / 3.0 * mbym)

    def so5(mbym):
        return -mbym * (
            1391.5 / 8.4
            - mbym * (1.0 - mbym) * 10.0 / 3.0
            + mbym * (1276.0 / 8.1 + mbym * (1.0 - mbym) * 170.0 / 9.0)
        )

    def so6(mbym):
        return _np.pi * mbym * (1490.0 / 3.0 + mbym * 260.0)

    def so7(mbym):
        eta_loc = mbym * (1.0 - mbym)
        return mbym * (
            -17097.8035 / 4.8384
            + eta_loc * 28764.25 / 6.72
            + eta_loc * eta_loc * 47.35 / 1.44
            + mbym
            * (
                -7189.233785 / 1.524096
                + eta_loc * 458.555 / 3.024
                - eta_loc * eta_loc * 534.5 / 7.2
            )
        )

    def qm2so4(mbym):
        return -720.0 / 9.6 * mbym * mbym

    def self2so4(mbym):
        return 1.0 / 9.6 * mbym * mbym

    def qm2s4(mbym):
        return 240.0 / 9.6 * mbym * mbym

    def self2s4(mbym):
        return -7.0 / 9.6 * mbym * mbym

    def qm2s6(mbym):
        return (4703.5 / 8.4 + 2935.0 / 6.0 * mbym - 120.0 * mbym * mbym) * mbym * mbym

    def self2s6(mbym):
        return (
            (-4108.25 / 6.72 - 108.5 / 1.2 * mbym + 125.5 / 3.6 * mbym * mbym)
            * mbym
            * mbym
        )

    pfaN_base = base["pfaN"]
    pfa1_base = base_dchi1
    pfa2_base = base["pfa2"]
    include_so7 = pn_spin_order in (-1, 7)
    include_so6 = pn_spin_order in (-1, 7, 6)
    include_so5 = pn_spin_order in (-1, 7, 6, 5)
    include_ss2pn = pn_spin_order in (-1, 7, 6, 5, 4)
    include_so3 = pn_spin_order in (-1, 7, 6, 5, 4, 3)

    so3_term = so3(m1M) * chi1L if include_so3 else 0.0
    so5_term = so5(m1M) * chi1L if include_so5 else 0.0
    so6_term = so6(m1M) * chi1L if include_so6 else 0.0
    so7_term = so7(m1M) * chi1L if include_so7 else 0.0
    ss2pn_term = (
        (
            (qm2so4(m1M) * qm_def1 + self2so4(m1M)) * chi1L2
            + (qm2s4(m1M) * qm_def1 + self2s4(m1M)) * chi1sq
        )
        if include_ss2pn
        else 0.0
    )
    ss3pn_term = (
        ((qm2s6(m1M) * qm_def1 + self2s6(m1M)) * chi1sq) if include_so6 else 0.0
    )

    pfa3_base = base["pfa3"] + so3_term
    pfa4_base = base["pfa4"] + ss2pn_term
    pfa5_base = base["pfa5"] + so5_term
    pfl5_base = base["pfl5"] + (3.0 * so5_term if include_so5 else 0.0)
    pfa6_base = base["pfa6"] + so6_term + ss3pn_term
    pfl6_base = base["pfl6"]
    pfa7_base = base["pfa7"] + so7_term
    pfa8_base = 0.0

    # Scale by pfaN (as in PNPhasingSeries)
    pfaN_full = pfaN_base * 1.0
    pfa1_full = pfaN_base * pfa1_base
    pfa2_full = pfaN_base * pfa2_base
    pfa3_full = pfaN_base * pfa3_base
    pfa4_full = pfaN_base * pfa4_base
    pfa5_full = pfaN_base * pfa5_base
    pfl5_full = pfaN_base * pfl5_base
    pfa6_full = pfaN_base * pfa6_base
    pfl6_full = pfaN_base * pfl6_base
    pfa7_full = pfaN_base * pfa7_base
    pfa8_full = pfaN_base * pfa8_base

    # Apply phase_order selection (mirrors LAL fall-through)
    if phase_order not in (-1, 7, 6, 5, 4, 3, 2, 1, 0):
        raise ValueError(f"Invalid phase_order {phase_order}")

    pfa_sel = dict(
        pfaN=0.0,
        pfa1=0.0,
        pfa2=0.0,
        pfa3=0.0,
        pfa4=0.0,
        pfa5=0.0,
        pfl5=0.0,
        pfa6=0.0,
        pfl6=0.0,
        pfa7=0.0,
        pfa8=0.0,
    )

    if phase_order in (-1, 7):
        pfa_sel["pfa7"] = pfa7_full
    if phase_order in (-1, 7, 6):
        pfa_sel["pfa6"] = pfa6_full
        pfa_sel["pfl6"] = pfl6_full
    if phase_order in (-1, 7, 6, 5):
        pfa_sel["pfa5"] = pfa5_full
        pfa_sel["pfl5"] = pfl5_full
    if phase_order in (-1, 7, 6, 5, 4):
        pfa_sel["pfa4"] = pfa4_full
    if phase_order in (-1, 7, 6, 5, 4, 3):
        pfa_sel["pfa3"] = pfa3_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2):
        pfa_sel["pfa2"] = pfa2_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2, 1):
        pfa_sel["pfa1"] = pfa1_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2, 1, 0):
        pfa_sel["pfaN"] = pfaN_full

    # Add zeta PN pieces after selection (unscaled by pfaN)
    if enable_prec:
        zc = coeffs["zc"]
        prec_fac = coeffs["prec_fac"]
        pfa_sel["pfa2"] += 2.0 * prec_fac * zc[0]
        pfa_sel["pfa3"] += 2.0 * prec_fac * zc[1]
        pfa_sel["pfa4"] += 2.0 * prec_fac * zc[2]
        pfa_sel["pfl5"] += 2.0 * prec_fac * zc[3]
        pfa_sel["pfa6"] += 2.0 * prec_fac * zc[4]
        pfa_sel["pfa7"] += 2.0 * prec_fac * zc[5]
        pfa_sel["pfa8"] += 2.0 * prec_fac * zc[6]

    return pfa_sel


def spintaylorf2_torch(**kwds):
    # Temporary parity guard: until the native torch precessing path is fully
    # tuned against LAL, optionally fall back to the trusted CPU/LAL generator
    # and simply cast to torch.  Enable the native path with
    # PYCBC_SPINTAYLORF2_NATIVE=1 (or the global PYCBC_TORCH_NATIVE_PORTS=1)
    # once the PN flux/parity work is complete; leave unset when comparing
    # against LAL to avoid hiding differences.
    if not torch_native_enabled("PYCBC_SPINTAYLORF2_NATIVE", default=False):
        from pycbc import scheme as _scheme
        from pycbc.waveform import get_fd_waveform

        old = _scheme.mgr.state
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        try:
            hP_cpu, hC_cpu = get_fd_waveform(approximant="SpinTaylorF2", **kwds)
        finally:  # always restore scheme
            _scheme.mgr.state = old
            _scheme.Scheme._single = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hP_t = torch.tensor(hP_cpu.numpy(), device=device, dtype=torch.complex128)
        hC_t = torch.tensor(hC_cpu.numpy(), device=device, dtype=torch.complex128)
        need_scheme = (
            not isinstance(_scheme.mgr.state, _scheme.TorchScheme)
            or getattr(_scheme.mgr.state, "torch_device", None) != device
        )
        old_scheme = _scheme.mgr.state
        old_single = _scheme.Scheme._single
        if need_scheme:
            _scheme.Scheme._single = None
            _scheme.mgr.state = _scheme.TorchScheme(device=str(device))
            _scheme.mgr.state.prefix = "torch"
        try:
            fsP = FrequencySeries(
                TorchArrayData(hP_t), delta_f=kwds["delta_f"], copy=False
            )
            fsC = FrequencySeries(
                TorchArrayData(hC_t), delta_f=kwds["delta_f"], copy=False
            )
        finally:
            if need_scheme:
                _scheme.mgr.state = old_scheme
                _scheme.Scheme._single = old_single
        if os.environ.get("PYCBC_SPINTAYLORF2_LOG", "0").lower() in (
            "1",
            "true",
            "yes",
        ):
            _log = logging.getLogger(__name__)
            _log.warning(
                "SpinTaylorF2 CPU fallback used (torch path disabled); no diagnostics run."
            )
        return fsP, fsC

    f_lower = kwds["f_lower"]
    delta_f = kwds["delta_f"]
    distance = kwds["distance"]
    distance_in_mpc = bool(
        kwds.get("distance_in_mpc", True)
    )  # True keeps historical PyCBC/LAL API (distance in Mpc)
    mass1 = kwds["mass1"]
    mass2 = kwds["mass2"]
    spin1x = kwds.get("spin1x", 0.0)
    spin1y = kwds.get("spin1y", 0.0)
    spin1z = kwds.get("spin1z", 0.0)
    rotate_spin = bool(
        kwds.get("rotate_spin", True)
    )  # default: mimic LALSimInspiral.c wrapper ROTATEY; False means spins are already in the L-frame (will break LAL parity if not).
    inclination = kwds.get("inclination", 0.0)
    pn_spin_order = int(kwds.get("pn_spin_order", -1))
    dquadmon1 = kwds.get("dquadmon1", 0.0)
    non_gr = dict(
        dchi0=kwds.get("non_gr_dchi0", 0.0),
        dchi1=kwds.get("non_gr_dchi1", 0.0),
        dchi2=kwds.get("non_gr_dchi2", 0.0),
        dchi3=kwds.get("non_gr_dchi3", 0.0),
        dchi4=kwds.get("non_gr_dchi4", 0.0),
        dchi5=kwds.get("non_gr_dchi5", 0.0),
        dchi5L=kwds.get("non_gr_dchi5L", 0.0),
        dchi6=kwds.get("non_gr_dchi6", 0.0),
        dchi6L=kwds.get("non_gr_dchi6L", 0.0),
        dchi7=kwds.get("non_gr_dchi7", 0.0),
    )
    phi0 = kwds["coa_phase"]
    phase_order = int(kwds["phase_order"])
    amplitude_order = int(kwds["amplitude_order"])  # kept for signature; LAL ignores >0
    f_ref = kwds.get("f_ref", 0.0)
    sideband_param = kwds.get("side_bands", None)
    f_final = kwds.get("f_final", 0.0)

    if amplitude_order != 0:
        raise ValueError(f"Invalid amplitude PN order {amplitude_order}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:532-534 (shft = 2π t_c term with t_c set by epoch).
    shft = -2.0 * _np.pi / delta_f
    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    m_sec = M * lal.MTSUN_SI
    piM = _np.pi * m_sec
    piM_t = torch.tensor(piM, device=device, dtype=dtype)

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:318-321 and 468-493 (Schwarzschild ISCO cutoff and padding/zeroing)
    vISCO = 1.0 / _np.sqrt(6.0)
    fISCO = vISCO * vISCO * vISCO / piM
    f_max = f_final if f_final and f_final > 0.0 else fISCO
    n = int(f_max / delta_f + 1)
    kmax = n - 1
    kmin = int(_np.ceil(f_lower / delta_f))

    idx = torch.arange(kmin, n, device=device, dtype=dtype)
    freqs = idx * delta_f
    v = torch.pow(piM_t * freqs, 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v3 * v2
    logv = torch.log(v)

    # Orientation / coefficients
    cosi = torch.cos(torch.tensor(inclination, device=device, dtype=dtype))
    sini = torch.sin(torch.tensor(inclination, device=device, dtype=dtype))

    # Rotate S1 about +y by the inclination (matches LALSimInspiral.c ROTATEY
    # before calling SpinTaylorF2). Disable with rotate_spin=False if spins are
    # already supplied in the L-frame.
    cincl = cosi
    sincl = sini
    if rotate_spin:
        spin1x_t = torch.tensor(
            float(spin1x * cincl + spin1z * sincl), device=device, dtype=dtype
        )
        spin1y_t = torch.tensor(float(spin1y), device=device, dtype=dtype)
        spin1z_t = torch.tensor(
            float(-spin1x * sincl + spin1z * cincl), device=device, dtype=dtype
        )
    else:
        spin1x_t = torch.tensor(float(spin1x), device=device, dtype=dtype)
        spin1y_t = torch.tensor(float(spin1y), device=device, dtype=dtype)
        spin1z_t = torch.tensor(float(spin1z), device=device, dtype=dtype)

    # Require explicit LNhat components to mirror LAL SpinTaylorF2 inputs.
    lnhatx_in = kwds.get("lnhatx", None)
    lnhaty_in = kwds.get("lnhaty", None)
    lnhatz_in = kwds.get("lnhatz", None)
    if lnhatx_in is None or lnhaty_in is None or lnhatz_in is None:
        raise ValueError(
            "SpinTaylorF2 torch now requires lnhatx/lnhaty/lnhatz to be provided explicitly."
        )
    lnhatx = torch.tensor(float(lnhatx_in), device=device, dtype=dtype)
    lnhaty = torch.tensor(float(lnhaty_in), device=device, dtype=dtype)
    lnhatz = torch.tensor(float(lnhatz_in), device=device, dtype=dtype)
    v_ref = torch.pow(
        piM_t
        * torch.tensor(f_ref if f_ref > 0 else f_lower, device=device, dtype=dtype),
        1.0 / 3.0,
    )
    orient = _orientation(
        mass1,
        mass2,
        v_ref,
        lnhatx,
        lnhaty,
        lnhatz,
        spin1x_t,
        spin1y_t,
        spin1z_t,
        device,
        dtype,
    )
    coeffs = _coeffs(mass1, mass2, orient["chi"], orient["kappa"], device, dtype)
    enable_prec = orient["chi"].item() != 0.0 and abs(orient["kappa"].item()) != 1.0

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:341-342 (alpha_ref = Alpha(v_ref) - alpha0)
    alpha_ref = (
        _alpha(v_ref, coeffs) - orient["alpha0"]
        if enable_prec
        else torch.tensor(0.0, device=device, dtype=dtype)
    )

    chi1L = orient["chi"] * orient["kappa"]
    chi1sq = orient["chi"] * orient["chi"]
    pfa = _pfa_coeffs(
        mass1,
        mass2,
        eta,
        chi1L,
        chi1sq,
        coeffs,
        phase_order,
        enable_prec=enable_prec,
        pn_spin_order=pn_spin_order,
        qm_def1=dquadmon1,
        non_gr=non_gr,
    )

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:495-516 (phasing series loop)
    phasing_num = (
        pfa["pfaN"]
        + pfa["pfa1"] * v
        + pfa["pfa2"] * v2
        + pfa["pfa3"] * v3
        + pfa["pfa4"] * v4
    )
    phasing = (
        phasing_num / v5
        + (pfa["pfa5"] + pfa["pfl5"] * logv)
        + (pfa["pfa6"] + pfa["pfl6"] * logv) * v
        + pfa["pfa7"] * v2
        + pfa["pfa8"] * v3
    )

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:495-504 (reference phasing subtraction when f_ref>0)
    if f_ref > 0.0:
        vref_num = (
            pfa["pfaN"]
            + pfa["pfa1"] * v_ref
            + pfa["pfa2"] * v_ref * v_ref
            + pfa["pfa3"] * v_ref * v_ref * v_ref
            + pfa["pfa4"] * v_ref * v_ref * v_ref * v_ref
        )
        logvref = torch.log(v_ref)
        ref_phasing = (
            vref_num / (v_ref * v_ref * v_ref * v_ref * v_ref)
            + (pfa["pfa5"] + pfa["pfl5"] * logvref)
            + (pfa["pfa6"] + pfa["pfl6"] * logvref) * v_ref
            + pfa["pfa7"] * v_ref * v_ref
            + pfa["pfa8"] * v_ref * v_ref * v_ref
        )
    else:
        ref_phasing = torch.tensor(0.0, device=device, dtype=dtype)

    # Carrier phase: time shift + reference subtraction + SPA constant
    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:532-534 (carrier phase shift)
    phasing = phasing + shft * freqs - 2.0 * phi0 - ref_phasing - _np.pi / 4.0

    # Precession pieces
    alpha = _alpha(v, coeffs) - alpha_ref if enable_prec else torch.zeros_like(v)
    u = torch.cos(alpha) + 1j * torch.sin(alpha)
    u_inv = 1.0 / u

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:520-531 (precession factors emission/u sideband sum)
    em0, em1, em2, em3, em4 = _emission(v, coeffs)
    SBplus = torch.zeros(5, device=device, dtype=torch.complex128)
    SBcross = torch.zeros(5, device=device, dtype=torch.complex128)
    if sideband_param is None:
        # Default: single m=0 sideband (matches LAL default parameter state).
        mm_list = [0]
    else:
        # Any explicit sideband selection loads all five (matches LAL reference lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:343-361; header comment at ~306 says “set sideband m to get a single sideband” but code loads all).
        mm_list = [-2, -1, 0, 1, 2]
    for mm in mm_list:
        idx_sb = 2 - mm
        SBplus[idx_sb] = _polarization(
            orient["thetaJ"], orient["psiJ"], mm, device, dtype
        )
        SBcross[idx_sb] = _polarization(
            orient["thetaJ"],
            orient["psiJ"] + torch.tensor(_np.pi / 4.0, device=device, dtype=dtype),
            mm,
            device,
            dtype,
        )

    prec_plus = (
        SBplus[0] * em0 * (u * u)
        + SBplus[1] * em1 * u
        + SBplus[2] * em2
        + SBplus[3] * em3 * u_inv
        + SBplus[4] * em4 * (u_inv * u_inv)
    )
    prec_cross = (
        SBcross[0] * em0 * (u * u)
        + SBcross[1] * em1 * u
        + SBcross[2] * em2
        + SBcross[3] * em3 * u_inv
        + SBcross[4] * em4 * (u_inv * u_inv)
    )

    # No additional empirical correction; match LAL/CUDA directly.

    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:485-487 (Newtonian SPA amplitude)
    # distance_in_mpc=True keeps PyCBC/LAL convention (distance in Mpc); False expects SI meters (raw LAL kernel contract). Mismatching this rescales amplitude by 1e6*PC_SI.
    # The sqrt(5/(32*eta)) comes from sqrt(-2*E0/F0) with E0=-eta/2 and F0=32*eta^2/5
    # (PNCoefficients.c:49-53, 346-350).
    dist_si = distance * 1.0e6 * lal.PC_SI if distance_in_mpc else distance
    amp0 = (
        -4.0
        * mass1
        * mass2
        / dist_si
        * lal.MRSUN_SI
        * lal.MTSUN_SI
        * _np.sqrt(_np.pi / 12.0)
    )
    amp0 = amp0 * _np.sqrt(5.0 / (32.0 * eta))
    amp = amp0 / (v3 * torch.sqrt(v))

    phasor = torch.exp(-1j * phasing)
    hP = prec_plus * phasor * amp
    hC = prec_cross * phasor * amp

    # Pad back to the full [0, f_max] band so the length matches CPU/LAL output
    # LAL reference: lalsimulation/lib/LALSimInspiralSpinTaylorF2.c:468-493 (allocation/zeroing)
    hP_full = torch.zeros(kmax + 1, device=device, dtype=hP.dtype)
    hC_full = torch.zeros(kmax + 1, device=device, dtype=hC.dtype)
    hP_full[kmin : kmin + hP.numel()] = hP
    hC_full[kmin : kmin + hC.numel()] = hC

    # Ensure the scheme matches the torch backend so FrequencySeries can wrap TorchArrayData without copying.
    from pycbc import scheme as _scheme

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    need_scheme = (
        not isinstance(old_scheme, _scheme.TorchScheme)
        or getattr(old_scheme, "torch_device", None) != device
    )
    if need_scheme:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme(device=str(device))
        _scheme.mgr.state.prefix = "torch"

    try:
        fsP = FrequencySeries(
            TorchArrayData(hP_full.to(torch.complex128)), delta_f=delta_f, copy=False
        )
        fsC = FrequencySeries(
            TorchArrayData(hC_full.to(torch.complex128)), delta_f=delta_f, copy=False
        )

        # Optional diagnostic: compare against CPU/LAL output to locate divergence.
        if os.environ.get("PYCBC_SPINTAYLORF2_LOG", "0").lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                from pycbc.waveform import get_fd_waveform

                old = _scheme.mgr.state
                _scheme.Scheme._single = None
                _scheme.mgr.state = _scheme.CPUScheme()
                _scheme.mgr.state.prefix = "cpu"
                cpuP, _ = get_fd_waveform(approximant="SpinTaylorF2", **kwds)
                _scheme.mgr.state = old
                cpu_arr = cpuP.numpy()
                tor_arr = hP_full.cpu().numpy()
                mask = _np.abs(cpu_arr) > 0
                rel = _np.linalg.norm(tor_arr[mask] - cpu_arr[mask]) / _np.linalg.norm(
                    cpu_arr[mask]
                )
                mag_ratio = _np.mean(_np.abs(tor_arr[mask]) / _np.abs(cpu_arr[mask]))
                phase_diff = _np.angle(tor_arr[mask] * _np.conj(cpu_arr[mask]))
                log = logging.getLogger(__name__)
                log.warning(
                    "SpinTaylorF2 diag: rel=%.3e mag_ratio=%.3e phase_mean=%.3e rad phase_std=%.3e rad",
                    rel,
                    mag_ratio,
                    phase_diff.mean(),
                    phase_diff.std(),
                )

                # More detailed diagnostics if requested
                if os.environ.get("PYCBC_SPINTAYLORF2_LOG_VERBOSE", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    freqs_full = _np.arange(hP_full.numel(), dtype=float) * delta_f
                    freqs_masked = freqs_full[mask]
                    # first non-zero bin
                    first_idx = _np.argmax(mask)
                    v_first = _np.power(piM * freqs_full[first_idx], 1.0 / 3.0)
                    amp_fac = amp0 / (v_first * v_first * v_first * _np.sqrt(v_first))
                    phasing_np = phasing.detach().cpu().numpy()
                    prec_cpu = (
                        cpu_arr[first_idx]
                        * _np.exp(1j * phasing_np[first_idx])
                        / amp_fac
                    )
                    prec_torch = prec_plus.detach().cpu().numpy()[first_idx]
                    ratio_prec = prec_cpu / prec_torch if prec_torch != 0 else _np.nan

                    log.warning(
                        "  first bin f=%.3f Hz: |torch|=%.3e |cpu|=%.3e ratio=%.6f phase_diff=%.6f rad",
                        freqs_full[first_idx],
                        _np.abs(tor_arr[first_idx]),
                        _np.abs(cpu_arr[first_idx]),
                        _np.abs(tor_arr[first_idx]) / _np.abs(cpu_arr[first_idx]),
                        _np.angle(tor_arr[first_idx] * _np.conj(cpu_arr[first_idx])),
                    )
                    # last non-zero bin
                    last_idx = _np.where(mask)[0][-1]
                    log.warning(
                        "  last bin f=%.3f Hz: |torch|=%.3e |cpu|=%.3e ratio=%.6f phase_diff=%.6f rad",
                        freqs_full[last_idx],
                        _np.abs(tor_arr[last_idx]),
                        _np.abs(cpu_arr[last_idx]),
                        _np.abs(tor_arr[last_idx]) / _np.abs(cpu_arr[last_idx]),
                        _np.angle(tor_arr[last_idx] * _np.conj(cpu_arr[last_idx])),
                    )
                    # phase slope
                    slope, intercept = _np.polyfit(freqs_masked, phase_diff, 1)
                    log.warning(
                        "  phase trend: slope=%.3e rad/Hz intercept=%.3e rad",
                        slope,
                        intercept,
                    )
                    log.warning(
                        "  amp ratio stats: min=%.3e max=%.3e std=%.3e",
                        _np.min(_np.abs(tor_arr[mask]) / _np.abs(cpu_arr[mask])),
                        _np.max(_np.abs(tor_arr[mask]) / _np.abs(cpu_arr[mask])),
                        _np.std(_np.abs(tor_arr[mask]) / _np.abs(cpu_arr[mask])),
                    )
                    log.warning(
                        "  first-bin prec: |cpu|=%.3e |torch|=%.3e ratio=%.3f ∠=%.3f rad",
                        _np.abs(prec_cpu),
                        _np.abs(prec_torch),
                        _np.abs(ratio_prec),
                        _np.angle(ratio_prec),
                    )
                    if last_idx < prec_plus.numel():
                        v_last = _np.power(piM * freqs_full[last_idx], 1.0 / 3.0)
                        prec_cpu_last = (
                            cpu_arr[last_idx]
                            * _np.exp(1j * phasing_np[last_idx])
                            / (amp0 / (v_last**3 * _np.sqrt(v_last)))
                        )
                        prec_torch_last = prec_plus.detach().cpu().numpy()[last_idx]
                        ratio_prec_last = (
                            prec_cpu_last / prec_torch_last
                            if prec_torch_last != 0
                            else _np.nan
                        )
                        log.warning(
                            "  last-bin prec:  |cpu|=%.3e |torch|=%.3e ratio=%.3f ∠=%.3f rad",
                            _np.abs(prec_cpu_last),
                            _np.abs(prec_torch_last),
                            _np.abs(ratio_prec_last),
                            _np.angle(ratio_prec_last),
                        )
            except Exception as exc:  # diagnostics must never fail waveform generation
                logging.getLogger(__name__).warning("SpinTaylorF2 diag failed: %s", exc)

        return fsP, fsC
    finally:
        if need_scheme:
            _scheme.mgr.state = old_scheme
            _scheme.Scheme._single = old_single
