# Torch-native SpinTaylorF2 generator (current: aligned-spin).
# Precession port in progress: see _precession_factors partial impl. Once
# complete, this module will mirror the CUDA kernel (alpha/zeta evolution and
# sideband modulation) on torch without CPU/PyCUDA.

import numpy as _np
import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


def _pn_coeffs(eta):
    pfa2 = 0.0
    pfa3 = -16.0 * _np.pi
    pfa4 = 10.0 * (3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta * eta / 144.0)
    pfa5 = -10.0 * (7729.0 / 1016064.0 + 3.0 * eta * eta / 8.0) * _np.pi
    pfl5 = 0.0
    pfa6 = 10.0 * (
        11583231236531.0 / 4694215680.0
        + 64969.0 * eta / 708.0
        + 64.0 * _np.pi * _np.pi / 3.0
        - 6848.0 * _np.pi / 21.0
    )
    pfa6 -= 10.0 * (15737765635.0 / 3048192.0 + 2255.0 * _np.pi * _np.pi / 12.0) * eta
    pfl6 = -6848.0 / 21.0
    pfa7 = 10.0 * _np.pi * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta * eta / 756.0)
    return pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7


def _precession_factors(thetaJ, gamma0, kappa, dtdv2, dtdv3, dtdv4, dtdv5,
                        prec_fac0, alpha0, v):
    """Compute Euler angles alpha, zeta and sideband factors on torch."""
    device = v.device
    dtype = v.dtype

    v2 = v * v
    v3 = v2 * v
    gamma02 = gamma0 * gamma0
    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa

    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gamma0 * v + (gamma0 * v) ** 2)
    logv = torch.log(v)
    logfac1 = torch.log(1.0 + kappa * gamma0 * v + sqrtfac)
    logfac2 = torch.log(kappa + gamma0 * v + sqrtfac)

    # Alpha and zeta follow the CUDA expressions (fall-through sums)
    alpha = prec_fac0 * (
        logfac2 * (
            dtdv2 * gamma0 + dtdv3 * kappa
            - dtdv5 * kappa / (2.0 * gamma02)
            + dtdv4 / (2.0 * gamma0)
            - dtdv4 * kappa2 / (2.0 * gamma0)
            + dtdv5 * kappa3 / (2.0 * gamma02)
        )
        + logfac1 * (
            -dtdv2 * gamma0 * kappa
            - dtdv3
            + kappa * gamma02 * gamma0 / 2.0
            - gamma02 * gamma0 * kappa3 / 2.0
        )
        + logv * (
            dtdv2 * gamma0 * kappa
            + dtdv3
            - kappa * gamma02 * gamma0 / 2.0
            + gamma02 * gamma0 * kappa3 / 2.0
        )
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
    ) - alpha0

    zeta = prec_fac0 * (
        dtdv3 * gamma0 * kappa * v
        + dtdv4 * v
        + logfac2
        * (
            -dtdv2 * gamma0
            - dtdv3 * kappa
            + dtdv5 * kappa / (2.0 * gamma02)
            - dtdv4 / (2.0 * gamma0)
            + dtdv4 * kappa2 / (2.0 * gamma0)
            - dtdv5 * kappa3 / (2.0 * gamma02)
        )
        + logfac1
        * (
            dtdv2 * gamma0 * kappa
            + dtdv3
            - kappa * gamma02 * gamma0 / 2.0
            + gamma02 * gamma0 * kappa3 / 2.0
        )
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

    # Sideband factors (mm=2 first entry)
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

    return alpha, zeta, RE_SBfac, IM_SBfac


def spintaylorf2_torch(**kwds):
    f_lower = kwds["f_lower"]
    delta_f = kwds["delta_f"]
    distance = kwds["distance"]
    mass1 = kwds["mass1"]
    mass2 = kwds["mass2"]
    phi0 = kwds["coa_phase"]
    phase_order = int(kwds["phase_order"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    tC = -1.0 / delta_f
    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    m_sec = M * 4.92549095e-6  # MTSUN_SI
    piM = _np.pi * m_sec

    vISCO = 1.0 / _np.sqrt(6.0)
    fISCO = vISCO**3 / piM
    n = int(_np.ceil(fISCO / delta_f) + 1)
    kmax = int(fISCO / delta_f)
    kmin = int(_np.ceil(f_lower / delta_f))
    kmax = kmax if kmax < n else n

    idx = torch.arange(kmax - kmin, device=device, dtype=dtype)
    freqs = (idx + kmin) * delta_f

    v = torch.pow(piM * freqs, 1.0 / 3.0)
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4

    pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7 = _pn_coeffs(eta)

    phasing = torch.zeros_like(v)
    phasing += pfa7 * v7
    phasing += (pfa6 + pfl6 * torch.log(4.0 * v)) * v6
    v0 = torch.tensor(piM * kmin * delta_f, device=device, dtype=dtype)
    phasing += (pfa5 + pfl5 * torch.log(v / torch.pow(v0, 1.0 / 3.0))) * v5
    phasing += pfa4 * v4
    phasing += pfa3 * v3
    phasing += pfa2 * v2
    phasing += 1.0

    phasing = phasing / torch.clamp(v5, min=1e-20)
    phasing += phi0 - 2.0 * _np.pi * freqs * tC

    # Precession: compute alpha/zeta and sideband factors
    dtdv2 = 743.0 / 336.0 + 11.0 * eta / 4.0
    pn_beta = (113.0 * mass1 / (12.0 * M) - 19.0 * eta / 6.0) * (spin1z := kwds.get("spin1z", 0.0))
    pn_sigma = 0.0  # spin-spin terms omitted for brevity; TODO match CUDA fully
    pn_gamma = 0.0
    dtdv3 = -4.0 * _np.pi + pn_beta
    dtdv4 = 3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta * eta / 144.0 - pn_sigma
    dtdv5 = (-7729.0 / 672.0 + 13.0 * eta / 8.0) * _np.pi + 9.0 * pn_gamma / 40.0
    thetaJ = torch.tensor(0.0, device=device, dtype=dtype)  # aligned-spin placeholder
    kappa = 1.0
    gamma0 = mass1 * abs(spin1z) / mass2 if mass2 != 0 else 0.0
    prec_fac0 = 5.0 * (4.0 + 3.0 * mass2 / mass1) / 64.0
    alpha0 = 0.0
    alpha, zeta, RE_SBfac, IM_SBfac = _precession_factors(thetaJ, gamma0, kappa,
                                                          dtdv2, dtdv3, dtdv4, dtdv5,
                                                          prec_fac0, alpha0, v)

    # Sideband modulation (mm = -2..2 mapped to indices 0..4)
    CBeta = torch.cos(thetaJ)
    SBeta = torch.sin(thetaJ)
    CAlpha = torch.cos(alpha)
    SAlpha = torch.sin(alpha)
    # Power terms
    CBeta2 = CBeta * CBeta
    CBeta3 = CBeta2 * CBeta
    CBeta4 = CBeta2 * CBeta2
    SBeta2 = SBeta * SBeta
    SBeta3 = SBeta2 * SBeta
    SBeta4 = SBeta2 * SBeta2
    CAlpha2 = CAlpha * CAlpha
    CAlpha3 = CAlpha2 * CAlpha
    CAlpha4 = CAlpha2 * CAlpha2
    SAlpha2 = SAlpha * SAlpha
    SAlpha3 = SAlpha2 * SAlpha
    SAlpha4 = SAlpha2 * SAlpha2

    # Complex prefactors for mm=2 (dominant)
    REprec = (
        SBeta4 * RE_SBfac[4] * CAlpha4
        + CBeta * SBeta3 * RE_SBfac[3] * CAlpha3
        + CBeta2 * SBeta2 * RE_SBfac[2] * CAlpha2
        + CBeta3 * SBeta * RE_SBfac[1] * CAlpha
        + CBeta4 * RE_SBfac[0]
    )
    IMprec = (
        SBeta4 * IM_SBfac[4] * SAlpha4
        + CBeta * SBeta3 * IM_SBfac[3] * SAlpha3
        + CBeta2 * SBeta2 * IM_SBfac[2] * SAlpha2
        + CBeta3 * SBeta * IM_SBfac[1] * SAlpha
        + CBeta4 * IM_SBfac[0] * 0.0
    )

    amp = torch.pow(freqs, -7.0 / 6.0) / distance
    h_real = amp * (torch.cos(phasing) * REprec - torch.sin(phasing) * IMprec)
    h_imag = amp * (torch.sin(phasing) * REprec + torch.cos(phasing) * IMprec)
    h = torch.complex(h_real, h_imag)

    fs = FrequencySeries(TorchArrayData(h.to(torch.complex128)), delta_f=delta_f, copy=False)
    return fs, fs
