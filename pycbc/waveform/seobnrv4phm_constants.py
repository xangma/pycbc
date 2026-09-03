"""Constants and small lookup tables for SEOBNRv4PHM torch-native port.

Extracted from lalsimulation (LALSimIMRSpinPrecEOBv4P.c and related headers).
Keeping them in one place avoids drift and simplifies unit testing of the
port-in-progress.
"""

import math

# Common physical constants (re-export for convenience)
from pycbc.waveform.constants import (
    _EULER_GAMMA,
    _MRSUN_SI,
    _MSUN_SI,
    _MTSUN_SI,
    _PC_SI,
    _PI,
)

# Approximant identifiers (mirroring lalsimulation)
SEOBNRV4P_NUMBER = 401
SEOBNRV4PHM_NUMBER = 402

# Mode support
LM_DEFAULT = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
LM_MAX = 5  # _SEOB_MODES_LMAX
LM_FLUX = [(ell, emm) for ell in range(2, 9) for emm in range(1, ell + 1)]

# Minimal initial separation used by LAL to set the start frequency (10.5 M)
MIN_INIT_SEPARATION_M = 10.5

# Integrator tolerances from LALSimIMRSpinPrecEOBv4P.c
EPS_ABS = 1.0e-8
EPS_REL = 1.0e-8
DELTA_T_MIN = 8.0e-5  # minimal step dt/M for adaptive RK (matches LAL v4P)
T_STEP_BACK = 150.0    # step-back length (in M) for HiS integration

__all__ = [
    "SEOBNRV4P_NUMBER",
    "SEOBNRV4PHM_NUMBER",
    "LM_DEFAULT",
    "LM_MAX",
    "LM_FLUX",
    "MIN_INIT_SEPARATION_M",
    "EPS_ABS",
    "EPS_REL",
    "DELTA_T_MIN",
    "T_STEP_BACK",
    "_PI",
    "_EULER_GAMMA",
    "_MSUN_SI",
    "_PC_SI",
    "_MRSUN_SI",
    "_MTSUN_SI",
    "compute_spin_aligned_hcoeffs",
    "compute_D_potential",
    "compute_Q_potential",
]


_COEFFS_K = (
    1.7336,
    -1.62045,
    -1.38086,
    1.43659,
    10.2573,
    2.26831,
    0.0,
    -0.426958,
    -126.687,
    17.3736,
    6.16466,
    0.0,
    267.788,
    -27.5201,
    31.1746,
    -59.1658,
)

_COEFFS_DSO = (
    -44.5324,
    0.0,
    0.0,
    66.1987,
    0.0,
    0.0,
    -343.313,
    -568.651,
    0.0,
    2495.29,
    0.0,
    147.481,
    0.0,
    0.0,
    0.0,
    0.0,
)

_COEFFS_DSS = (
    6.06807,
    0.0,
    0.0,
    0.0,
    -36.0272,
    37.1964,
    0.0,
    -41.0003,
    0.0,
    0.0,
    -326.325,
    528.511,
    706.958,
    0.0,
    1161.78,
    0.0,
)


def _eval_hcoeff_poly(coeffs: tuple[float, ...], eta, chi):
    """Monomial evaluation of 2D polynomials in (eta, chi) matching LAL summation order."""
    chi2 = chi * chi
    chi3 = chi2 * chi
    eta2 = eta * eta
    eta3 = eta2 * eta
    c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33 = coeffs
    return (
        c00
        + c01 * chi
        + c02 * chi2
        + c03 * chi3
        + c10 * eta
        + c11 * eta * chi
        + c12 * eta * chi2
        + c13 * eta * chi3
        + c20 * eta2
        + c21 * eta2 * chi
        + c22 * eta2 * chi2
        + c23 * eta2 * chi3
        + c30 * eta3
        + c31 * eta3 * chi
        + c32 * eta3 * chi2
        + c33 * eta3 * chi3
    )


def compute_spin_aligned_hcoeffs(eta: float, a: float, chi: float | None = None, spin_aligned_version: int = 4):
    """Compute SpinEOBHCoeffs (KK, k0-5, k5l, b3/bb3, d1/d1v2/dheffSS/dheffSSv2).

    COMPLETE port of XLALSimIMRCalculateSpinPrecEOBHCoeffs_v2
    (LALSimIMRCalculateSpinPrecEOBHCoeffs.c:130-174,189-332) for
    SpinAlignedEOBversion=4. Includes calibrated spin-orbit (d1v2) and
    spin-spin (dheffSSv2) fits; d1 and dheffSS are zero for v4.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio (m1*m2 / (m1+m2)^2)
    a : float
        |sigmaKerr| as used in LAL (dimensionless)
    chi : float, optional
        Augmented spin used in the calibration polynomials. If omitted,
        defaults to a/(1-2 eta) (aligned-spin limit) to mirror LAL.
    spin_aligned_version : int, optional
        Defaults to 4 (SEOBNRv4P/PHM). Other versions are not implemented here.

    Returns
    -------
    dict with keys KK, k0, k1, k2, k3, k4, k5, k5l, b3, bb3, d1, d1v2, dheffSS, dheffSSv2
    """
    if spin_aligned_version != 4:
        raise NotImplementedError("Only SpinAlignedEOBversion=4 is implemented")

    third = 1.0 / 3.0
    fifth = 1.0 / 5.0
    ln2 = 0.6931471805599453094172321214581765680755
    chi_eff = chi
    if chi_eff is None:
        denom = 1.0 - 2.0 * eta
        chi_eff = a / denom if (isinstance(denom, (int, float)) and abs(denom) > 1.0e-12) else 0.0

    # K calibration: XLALSimIMRCalculateSpinPrecEOBHCoeffs_v2, Eq. 4.8/4.12.
    KK = _eval_hcoeff_poly(_COEFFS_K, eta, chi_eff)

    m1PlusEtaKK = -1.0 + eta * KK
    invm1PlusEtaKK = 1.0 / m1PlusEtaKK

    k0 = KK * (m1PlusEtaKK - 1.0)
    k1 = -2.0 * (k0 + KK) * m1PlusEtaKK
    k1p2 = k1 * k1
    k1p3 = k1 * k1p2

    k2 = (k1 * (k1 - 4.0 * m1PlusEtaKK)) * 0.5 - a * a * k0 * m1PlusEtaKK * m1PlusEtaKK
    k3 = (
        -(k1 * k1) * k1 * third
        + k1 * k2
        + (k1 * k1) * m1PlusEtaKK
        - 2.0 * (k2 - m1PlusEtaKK) * m1PlusEtaKK
        - a * a * k1 * (m1PlusEtaKK * m1PlusEtaKK)
    )
    k4 = (
        (24.0 / 96.0) * (k1 * k1) * (k1 * k1)
        - (96.0 / 96.0) * (k1 * k1) * k2
        + (48.0 / 96.0) * k2 * k2
        - (64.0 / 96.0) * (k1 * k1) * k1 * m1PlusEtaKK
        + (48.0 / 96.0) * (a * a) * (k1 * k1 - 2.0 * k2) * (m1PlusEtaKK * m1PlusEtaKK)
        + (96.0 / 96.0) * k1 * (k3 + 2.0 * k2 * m1PlusEtaKK)
        - m1PlusEtaKK * ((192.0 / 96.0) * k3 + m1PlusEtaKK * (-(3008.0 / 96.0) + (123.0 / 96.0) * _PI * _PI))
    )

    # k5 for version 4
    k5 = m1PlusEtaKK * m1PlusEtaKK * (
        -4237.0 / 60.0
        + 128.0 / 5.0 * _EULER_GAMMA
        + 2275.0 * _PI * _PI / 512.0
        - third * (a * a) * (k1p3 - 3.0 * (k1 * k2) + 3.0 * k3)
        - ((k1p3 * k1p2) - 5.0 * (k1p3 * k2) + 5.0 * k1 * k2 * k2 + 5.0 * k1p2 * k3 - 5.0 * k2 * k3 - 5.0 * k1 * k4)
        * fifth
        * invm1PlusEtaKK
        * invm1PlusEtaKK
        + ((k1p2 * k1p2) - 4.0 * (k1p2 * k2) + 2.0 * k2 * k2 + 4.0 * k1 * k3 - 4.0 * k4)
        * 0.5
        * invm1PlusEtaKK
        + (256.0 / 5.0) * ln2
        + (41.0 * _PI * _PI / 32.0 - 221.0 / 6.0) * eta
    )
    k5l = (m1PlusEtaKK * m1PlusEtaKK) * (64.0 / 5.0)

    if spin_aligned_version == 4:
        d1 = 0.0
        dheffSS = 0.0
        d1v2 = _eval_hcoeff_poly(_COEFFS_DSO, eta, chi_eff)
        dheffSSv2 = _eval_hcoeff_poly(_COEFFS_DSS, eta, chi_eff)
    else:  # fallback (not calibrated)
        d1 = 0.0
        d1v2 = 0.0
        dheffSS = 0.0
        dheffSSv2 = 0.0

    return {
        "KK": KK,
        "k0": k0,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "k5l": k5l,
        "b3": 0.0,
        "bb3": 0.0,
        "d1": d1,
        "d1v2": d1v2,
        "dheffSS": dheffSS,
        "dheffSSv2": dheffSSv2,
    }


def compute_D_potential(eta: float, u: float):
    """Aligned-spin tortoise potential D(u) from SEOBNRv4 (log form).

    D(u) = 1 + ln(1 + 6 η u^2 + 2 (26 - 3 η) η u^3)

    Supports float or torch.Tensor inputs (torch preferred inside dynamics).
    """
    if not hasattr(u, "device") and not hasattr(eta, "device"):
        eta_f = float(eta)
        term_f = 1.0 + 6.0 * eta_f * (u * u) + 2.0 * (26.0 - 3.0 * eta_f) * eta_f * (u * u * u)
        return 1.0 + math.log(term_f)

    # Lazy torch support to keep this module torch-agnostic at import time
    try:
        import torch

        if isinstance(u, torch.Tensor) or isinstance(eta, torch.Tensor):
            eta_t = torch.as_tensor(eta, device=getattr(u, "device", None), dtype=getattr(u, "dtype", None))
            u_t = torch.as_tensor(u, device=eta_t.device if hasattr(eta_t, "device") else getattr(u, "device", None), dtype=getattr(u, "dtype", None))
            d2 = 6.0 * eta_t
            d3 = 2.0 * (26.0 - 3.0 * eta_t) * eta_t
            term = 1.0 + d2 * u_t * u_t + d3 * u_t * u_t * u_t
            term = torch.clamp(term, min=1.0e-12)
            return 1.0 + torch.log(term)
    except Exception:
        pass

    eta_f = float(eta)
    term_f = 1.0 + 6.0 * eta_f * (u * u) + 2.0 * (26.0 - 3.0 * eta_f) * eta_f * (u * u * u)
    return 1.0 + math.log(term_f)


def compute_Q_potential(eta: float, u: float):
    """Spin-aligned quartic-momentum potential Q(u) used with tortoise pr*.

    Q(u) = 2 η (4 - 3 η) u^2

    Returns float or torch.Tensor consistent with the input.
    """
    coeff = 2.0 * eta * (4.0 - 3.0 * eta)
    if not hasattr(u, "device") and not hasattr(eta, "device"):
        return coeff * u * u
    try:
        import torch

        if isinstance(u, torch.Tensor) or isinstance(eta, torch.Tensor):
            coeff_t = torch.as_tensor(coeff, device=getattr(u, "device", None), dtype=getattr(u, "dtype", None))
            u_t = torch.as_tensor(u, device=getattr(u, "device", None), dtype=getattr(u, "dtype", None))
            return coeff_t * u_t * u_t
    except Exception:
        pass

    return coeff * u * u
