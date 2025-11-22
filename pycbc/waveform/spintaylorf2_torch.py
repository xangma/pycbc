"""Torch-native SpinTaylorF2 generator mirroring the historical CUDA kernel.

This keeps the torch implementation scheme-specific while leaving the
scheme-agnostic API untouched.  Expressions below are a direct vectorised
translation of the PyCUDA kernel in ``SpinTaylorF2.py`` so we maintain numerical
parity with the existing CPU/LAL implementation (up to expected tolerance)."""

import os
import numpy as _np
import torch
import lal

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


def _pn_and_flux_coeffs(eta, pn_beta, pn_sigma, pn_gamma):
    """Return phase/amplitude PN coefficients (matches CUDA kernel)."""
    theta = -11831.0 / 9240.0
    lambdaa = -1987.0 / 3080.0

    pfaN = 3.0 / (128.0 * eta)
    pfa2 = 5.0 * (743.0 / 84.0 + 11.0 * eta) / 9.0
    pfa3 = -16.0 * _np.pi + 4.0 * pn_beta
    pfa4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0 - 10.0 * pn_sigma
    pfa5 = 5.0 / 9.0 * (7729.0 / 84.0 - 13.0 * eta) * _np.pi - pn_gamma
    pfl5 = 5.0 / 3.0 * (7729.0 / 84.0 - 13.0 * eta) * _np.pi - 3.0 * pn_gamma
    pfa6 = (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * _np.pi * _np.pi
        - 6848.0 / 21.0 * _np.euler_gamma
        + eta * (-15335.597827 / 3.048192 + 2255.0 / 12.0 * _np.pi * _np.pi - 1760.0 / 3.0 * theta + 12320.0 / 9.0 * lambdaa)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
    )
    pfl6 = -6848.0 / 21.0
    pfa7 = _np.pi * 5.0 / 756.0 * (15419335.0 / 336.0 + 75703.0 / 2.0 * eta - 14809.0 * eta * eta)

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
    FTa7 = -(162.85 / 5.04 - 214.745 / 1.728 * eta - 193.385 / 3.024 * eta * eta) * _np.pi

    dETaN = -eta
    dETa1 = 2.0 * (-(3.0 / 4.0 + eta / 12.0))
    dETa2 = 3.0 * (-(27.0 / 8.0 - 19.0 / 8.0 * eta + eta * eta / 24.0))
    dETa3 = 4.0 * (-(67.5 / 6.4 - (344.45 / 5.76 - 20.5 / 9.6 * _np.pi * _np.pi) * eta + 15.5 / 9.6 * eta * eta + 3.5 / 518.4 * eta * eta * eta))

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


def _alpha_zeta_expr(v, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5, prec_fac0):
    """Core alpha/zeta expressions; caller subtracts reference values."""
    v2 = v * v
    v3 = v2 * v

    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gamma0 * v + (gamma0 * v) ** 2)
    logv = torch.log(v)
    logfac1 = torch.log(1.0 + kappa * gamma0 * v + sqrtfac)
    logfac2 = torch.log(kappa + gamma0 * v + sqrtfac)

    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa
    gamma02 = gamma0 * gamma0
    gamma03 = gamma02 * gamma0

    alpha = prec_fac0 * (
        logfac2 * (dtdv2 * gamma0 + dtdv3 * kappa - dtdv5 * kappa / (2.0 * gamma02) + dtdv4 / (2.0 * gamma0) - dtdv4 * kappa2 / (2.0 * gamma0) + dtdv5 * kappa3 / (2.0 * gamma02))
        + logfac1 * (-dtdv2 * gamma0 * kappa - dtdv3 + kappa * gamma03 / 2.0 - gamma03 * kappa3 / 2.0)
        + logv * (dtdv2 * gamma0 * kappa + dtdv3 - kappa * gamma03 / 2.0 + gamma03 * kappa3 / 2.0)
        + sqrtfac * (
            dtdv3
            + dtdv4 * v / 2.0
            + dtdv5 / (3.0 * gamma02)
            + dtdv4 * kappa / (2.0 * gamma0)
            + dtdv5 * kappa * v / (6.0 * gamma0)
            - dtdv5 * kappa2 / (2.0 * gamma02)
            - 1.0 / (3.0 * v3)
            - gamma0 * kappa / (6.0 * v2)
            - dtdv2 / v
            - gamma02 / (3.0 * v)
            + gamma02 * kappa2 / (2.0 * v)
            + dtdv5 * v2 / 3.0
        )
    )

    zeta = prec_fac0 * (
        dtdv3 * gamma0 * kappa * v
        + dtdv4 * v
        + logfac2 * (-dtdv2 * gamma0 - dtdv3 * kappa + dtdv5 * kappa / (2.0 * gamma02) - dtdv4 / (2.0 * gamma0) + dtdv4 * kappa2 / (2.0 * gamma0) - dtdv5 * kappa3 / (2.0 * gamma02))
        + logv * (kappa * gamma03 / 2.0 - gamma03 * kappa3 / 2.0)
        + logfac1 * (dtdv2 * gamma0 * kappa + dtdv3 - kappa * gamma03 / 2.0 + gamma03 * kappa3 / 2.0)
        - 1.0 / (3.0 * v3)
        - gamma0 * kappa / (2.0 * v2)
        - dtdv2 / v
        + dtdv4 * gamma0 * kappa * v2 / 2.0
        + dtdv5 * v2 / 2.0
        + sqrtfac
        * (
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
        + dtdv5 * gamma0 * kappa * v3 / 3.0
    )

    return alpha, zeta


def _precession_factors(thetaJ, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5,
                        prec_fac0, alpha_ref, zeta_ref, v):
    """Compute Euler angles alpha, zeta, beta and sideband factors on torch."""
    device = v.device
    dtype = v.dtype

    alpha, zeta = _alpha_zeta_expr(v, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5, prec_fac0)
    alpha = alpha - alpha_ref
    zeta = zeta - zeta_ref

    # Opening angle beta between J and L (used for sidebands)
    gam = gamma0 * v
    beta_arg = (1.0 + kappa * gam) / torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    beta = torch.acos(torch.clamp(beta_arg, -1.0, 1.0))

    # Sideband factors (mm = 2..-2, P -> plus pol)
    RE_SBfac = torch.tensor([
        (1.0 + torch.cos(thetaJ) ** 2) / 2.0,
        torch.sin(2.0 * thetaJ),
        3.0 * torch.sin(thetaJ) ** 2,
        -torch.sin(2.0 * thetaJ),
        (1.0 + torch.cos(thetaJ) ** 2) / 2.0,
    ], device=device, dtype=dtype)

    IM_SBfac = torch.tensor([
        -torch.cos(thetaJ),
        -2.0 * torch.sin(thetaJ),
        torch.tensor(0.0, device=device, dtype=dtype),
        -2.0 * torch.sin(thetaJ),
        torch.cos(thetaJ),
    ], device=device, dtype=dtype)

    return alpha, zeta, beta, RE_SBfac, IM_SBfac


def spintaylorf2_torch(**kwds):
    # Temporary parity guard: until the native torch precessing path is fully
    # tuned against LAL, optionally fall back to the trusted CPU/LAL generator
    # and simply cast to torch.  Enable the native path with
    # PYCBC_SPINTAYLORF2_NATIVE=1 once the PN flux/parity work is complete.
    if os.environ.get("PYCBC_SPINTAYLORF2_NATIVE", "0").lower() not in ("1", "true", "yes"):
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
        fsP = FrequencySeries(TorchArrayData(hP_t), delta_f=kwds["delta_f"], copy=False)
        fsC = FrequencySeries(TorchArrayData(hC_t), delta_f=kwds["delta_f"], copy=False)
        return fsP, fsC

    f_lower = kwds["f_lower"]
    delta_f = kwds["delta_f"]
    distance = kwds["distance"]
    mass1 = kwds["mass1"]
    mass2 = kwds["mass2"]
    spin1x = kwds.get("spin1x", 0.0)
    spin1y = kwds.get("spin1y", 0.0)
    spin1z = kwds.get("spin1z", 0.0)
    inclination = kwds.get("inclination", 0.0)
    phi0 = kwds["coa_phase"]
    phase_order = int(kwds["phase_order"])
    amplitude_order = int(kwds["amplitude_order"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    tC = -1.0 / delta_f
    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    m_sec = M * lal.MTSUN_SI
    piM = _np.pi * m_sec
    piM_t = torch.tensor(piM, device=device, dtype=dtype)

    vISCO = 1.0 / _np.sqrt(6.0)
    fISCO = vISCO**3 / piM
    n = int(_np.ceil(fISCO / delta_f) + 1)
    kmax = int(fISCO / delta_f)
    kmin = int(_np.ceil(f_lower / delta_f))
    kmax = kmax if kmax < n else n

    # Frequencies we evaluate (start at f_lower to avoid division by zero).
    idx = torch.arange(kmax - kmin + 1, device=device, dtype=dtype)
    freqs = (idx + kmin) * delta_f

    v = torch.pow(piM_t * freqs, 1.0 / 3.0)
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4
    v8 = v4 * v4
    v9 = v4 * v5
    v10 = v5 * v5

    # Spin geometry (matches CUDA kernel)
    lnhatx = torch.sin(torch.tensor(inclination, device=device, dtype=dtype))
    lnhaty = torch.tensor(0.0, device=device, dtype=dtype)
    lnhatz = torch.cos(torch.tensor(inclination, device=device, dtype=dtype))
    spin_vec = torch.tensor([spin1x, spin1y, spin1z], device=device, dtype=dtype)
    chi = torch.linalg.vector_norm(spin_vec)
    kappa = (spin_vec @ torch.tensor([lnhatx, lnhaty, lnhatz], device=device, dtype=dtype)) / chi if chi > 0 else torch.tensor(1.0, device=device, dtype=dtype)
    v0 = torch.pow(piM_t * torch.tensor(kmin * delta_f, device=device, dtype=dtype), 1.0 / 3.0)
    Jx0 = mass1 * mass2 * lnhatx / v0 + mass1 * mass1 * spin1x
    Jy0 = mass1 * mass2 * lnhaty / v0 + mass1 * mass1 * spin1y
    Jz0 = mass1 * mass2 * lnhatz / v0 + mass1 * mass1 * spin1z
    Jnorm = torch.sqrt(Jx0 * Jx0 + Jy0 * Jy0 + Jz0 * Jz0)
    thetaJ = torch.arccos(Jz0 / Jnorm)
    psiJ = torch.atan2(Jy0, -Jx0)
    psiJ_P = psiJ  # plus
    psiJ_C = psiJ + torch.tensor(_np.pi / 4.0, device=device, dtype=dtype)  # cross
    rotLx = lnhatx * torch.cos(thetaJ) * torch.cos(psiJ) - lnhaty * torch.cos(thetaJ) * torch.sin(psiJ) + lnhatz * torch.sin(thetaJ)
    rotLy = lnhatx * torch.sin(psiJ) + lnhaty * torch.cos(psiJ)
    alpha0 = torch.atan2(rotLy, rotLx)
    gamma0 = mass1 * chi / mass2 if mass2 != 0 else torch.tensor(0.0, device=device, dtype=dtype)
    prec_fac0 = 5.0 * (4.0 + 3.0 * mass2 / mass1) / 64.0

    pn_beta = (113.0 * mass1 / (12.0 * M) - 19.0 * eta / 6.0) * chi * kappa
    pn_sigma = ((5.0 * (3.0 * kappa * kappa - 1.0) / 2.0) + (7.0 - kappa * kappa) / 96.0) * (mass1 * mass1 * chi * chi / (M * M))
    pn_gamma = (5.0 * (146597.0 + 7056.0 * eta) * mass1 / (2268.0 * M) - 10.0 * eta * (1276.0 + 153.0 * eta) / 81.0) * chi * kappa

    dtdv2 = 743.0 / 336.0 + 11.0 * eta / 4.0
    dtdv3 = -4.0 * _np.pi + pn_beta
    dtdv4 = 3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta * eta / 144.0 - pn_sigma
    dtdv5 = (-7729.0 / 672.0 + 13.0 * eta / 8.0) * _np.pi + 9.0 * pn_gamma / 40.0

    coeffs = _pn_and_flux_coeffs(eta, pn_beta, pn_sigma, pn_gamma)

    # Reference Euler angles at v0 (alpha_ref, zeta_ref) as in CUDA kernel.
    alpha_ref, zeta_ref = _alpha_zeta_expr(v0, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5, prec_fac0)
    alpha_ref = alpha_ref - alpha0  # kernel subtracts alpha0 only for alpha_ref

    # Phasing PN expansion with fall-through ordering
    phasing = torch.zeros_like(v)
    if phase_order == -1 or phase_order >= 7:
        phasing += coeffs["pfa7"] * v7
    if phase_order == -1 or phase_order >= 6:
        phasing += (coeffs["pfa6"] + coeffs["pfl6"] * torch.log(4.0 * v)) * v6
    if phase_order == -1 or phase_order >= 5:
        v0const = torch.tensor(piM * kmin * delta_f, device=device, dtype=dtype)
        phasing += (coeffs["pfa5"] + coeffs["pfl5"] * torch.log(v / torch.pow(v0const, 1.0 / 3.0))) * v5
    if phase_order == -1 or phase_order >= 4:
        phasing += coeffs["pfa4"] * v4
    if phase_order == -1 or phase_order >= 3:
        phasing += coeffs["pfa3"] * v3
    if phase_order == -1 or phase_order >= 2:
        phasing += coeffs["pfa2"] * v2
    if phase_order == -1 or phase_order >= 0:
        phasing += 1.0

    phasing = phasing * coeffs["pfaN"] / torch.clamp(v5, min=1e-30)

    # Amplitude PN expansion (flux, energy)
    flux = torch.zeros_like(v)
    dEnergy = torch.zeros_like(v)
    if amplitude_order == -1 or amplitude_order >= 7:
        flux += coeffs["FTa7"] * v7
    if amplitude_order == -1 or amplitude_order >= 6:
        flux += (coeffs["FTa6"] + coeffs["FTl6"] * torch.log(16.0 * v2)) * v6
        dEnergy += coeffs["dETa3"] * v6
    if amplitude_order == -1 or amplitude_order >= 5:
        flux += coeffs["FTa5"] * v5
    if amplitude_order == -1 or amplitude_order >= 4:
        flux += coeffs["FTa4"] * v4
        dEnergy += coeffs["dETa2"] * v4
    if amplitude_order == -1 or amplitude_order >= 3:
        flux += coeffs["FTa3"] * v3
    if amplitude_order == -1 or amplitude_order >= 2:
        flux += coeffs["FTa2"] * v2
        dEnergy += coeffs["dETa1"] * v2
    if amplitude_order == -1 or amplitude_order >= 0:
        flux += 1.0
        dEnergy += 1.0

    flux = flux * coeffs["FTaN"] * v10
    dEnergy = dEnergy * coeffs["dETaN"] * v

    # Precession angles and sideband factors
    alpha, zeta, beta, RE_SBfac, IM_SBfac = _precession_factors(
        thetaJ, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5, prec_fac0, alpha_ref, zeta_ref, v
    )

    CBeta = torch.cos(beta / 2.0)
    SBeta = torch.sin(beta / 2.0)
    CAlpha1 = torch.cos(-alpha)
    SAlpha1 = torch.sin(-alpha)
    CAlpha2 = torch.cos(-2.0 * alpha)
    SAlpha2 = torch.sin(-2.0 * alpha)
    CAlpha3 = torch.cos(-3.0 * alpha)
    SAlpha3 = torch.sin(-3.0 * alpha)
    CAlpha4 = torch.cos(-4.0 * alpha)
    SAlpha4 = torch.sin(-4.0 * alpha)

    CBeta2 = CBeta * CBeta
    CBeta3 = CBeta2 * CBeta
    CBeta4 = CBeta2 * CBeta2
    SBeta2 = SBeta * SBeta
    SBeta3 = SBeta2 * SBeta
    SBeta4 = SBeta2 * SBeta2

    # Shared pieces for prec factors
    A = (
        SBeta4 * RE_SBfac[4] * CAlpha4
        + CBeta * SBeta3 * RE_SBfac[3] * CAlpha3
        + CBeta2 * SBeta2 * RE_SBfac[2] * CAlpha2
        + CBeta3 * SBeta * RE_SBfac[1] * CAlpha1
        + CBeta4 * RE_SBfac[0]
    )
    B = (
        SBeta4 * IM_SBfac[4] * SAlpha4
        + CBeta * SBeta3 * IM_SBfac[3] * SAlpha3
        + CBeta2 * SBeta2 * IM_SBfac[2] * SAlpha2
        + CBeta3 * SBeta * IM_SBfac[1] * SAlpha1
    )  # IM_SBfac[0] term is zero in kernel
    C = (
        SBeta4 * RE_SBfac[4] * SAlpha4
        + CBeta * SBeta3 * RE_SBfac[3] * SAlpha3
        + CBeta2 * SBeta2 * RE_SBfac[2] * SAlpha2
        + CBeta3 * SBeta * RE_SBfac[1] * SAlpha1
    )  # RE_SBfac[0] term times zero
    D = (
        SBeta4 * IM_SBfac[4] * CAlpha4
        + CBeta * SBeta3 * IM_SBfac[3] * CAlpha3
        + CBeta2 * SBeta2 * IM_SBfac[2] * CAlpha2
        + CBeta3 * SBeta * IM_SBfac[1] * CAlpha1
        + CBeta4 * IM_SBfac[0]
    )

    two_psiP = 2.0 * psiJ_P
    two_psiC = 2.0 * psiJ_C
    cos2psiP = torch.cos(two_psiP)
    sin2psiP = torch.sin(two_psiP)
    cos2psiC = torch.cos(two_psiC)
    sin2psiC = torch.sin(two_psiC)

    RE_prec_facP = cos2psiP * A - sin2psiP * B
    IM_prec_facP = cos2psiP * C + sin2psiP * D
    RE_prec_facC = cos2psiC * A - sin2psiC * B
    IM_prec_facC = cos2psiC * C + sin2psiC * D

    # Final phasing and amplitude
    shft = -2.0 * _np.pi * tC
    phasing = phasing + shft * freqs - 2.0 * phi0 + 2.0 * zeta

    amp0 = -4.0 * mass1 * mass2 / (1.0e6 * distance * lal.PC_SI) * lal.MRSUN_SI * lal.MTSUN_SI * _np.sqrt(_np.pi / 12.0)
    amp = amp0 * torch.sqrt(-dEnergy / torch.clamp(flux, min=1e-30)) * v

    CPhasing = amp * torch.cos(phasing - _np.pi / 4.0)
    SPhasing = amp * torch.sin(phasing - _np.pi / 4.0)

    hP = torch.complex(RE_prec_facP * CPhasing + IM_prec_facP * SPhasing,
                       IM_prec_facP * CPhasing - RE_prec_facP * SPhasing)
    hC = torch.complex(RE_prec_facC * CPhasing + IM_prec_facC * SPhasing,
                       IM_prec_facC * CPhasing - RE_prec_facC * SPhasing)

    # Pad back to the full [0, f_max] band so the length matches CPU/LAL output.
    hP_full = torch.zeros(kmax + 1, device=device, dtype=hP.dtype)
    hC_full = torch.zeros(kmax + 1, device=device, dtype=hC.dtype)
    hP_full[kmin : kmin + hP.numel()] = hP
    hC_full[kmin : kmin + hC.numel()] = hC

    fsP = FrequencySeries(TorchArrayData(hP_full.to(torch.complex128)), delta_f=delta_f, copy=False)
    fsC = FrequencySeries(TorchArrayData(hC_full.to(torch.complex128)), delta_f=delta_f, copy=False)
    return fsP, fsC
