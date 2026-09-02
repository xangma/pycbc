"""Torch-native SEOBNRv4PHM dynamics helpers (precessing, TD inspiral).

This is a pragmatic, torch-first translation of the LAL precessing SEOB model.
The goal is numerical parity at the interface level (inputs / outputs) while
keeping the implementation lightweight enough to iterate on quickly. The
Hamiltonian and flux follow the spin-aligned v4P structure with leading-order
precession torques; frame rotation and NQC hooks are provided for waveform
generation in :mod:`seobnrv4phm_torch`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from pycbc.waveform._native_math import associated_legendre_at_zero
from pycbc.waveform.seobnrv4phm_constants import (
    _EULER_GAMMA,
    _MTSUN_SI,
    _PI,
    DELTA_T_MIN,
    EPS_ABS,
    EPS_REL,
    LM_DEFAULT,
    LM_FLUX,
    MIN_INIT_SEPARATION_M,
    T_STEP_BACK,
    compute_spin_aligned_hcoeffs,
)
from pycbc.waveform.seobnrv4phm_multiroot import (
    GslCblasUnavailable,
    _load_cblas as _load_gsl_cblas,
)
from pycbc.waveform.seobnrv4phm_multiroot import (
    gsl_multiroot_hybrids as _ported_gsl_multiroot_hybrids,
)
from pycbc.waveform.seobnrv4phm_nqc import solve_nqc_coeffs
from pycbc.waveform.seobnrv4phm_peak import find_peak_time, local_derivatives
from pycbc.waveform.seobnrv4phm_waveform_coeffs import (
    rho21,
    rho22,
    rho33,
    rho44,
    rho55,
    rho_lm_full,
)

EPS_ALIGN = 1.0e-4
_LAL_V4P_NUMERICAL_DERIVATIVE_STEP = 2.0e-3
_LAL_PRECESSING_CALCOMEGA_STEP = 1.0e-4
_GSL_DERIV_LIB = None
_GSL_DERIV_FUNC_TYPE = None
_GSL_DERIV_STRUCT_TYPE = None
_GSL_DERIV_UNAVAILABLE = False
_GSL_MULTIROOT_LIB = None
_GSL_MULTIROOT_FUNC_TYPE = None
_GSL_MULTIROOT_STRUCT_TYPE = None
_GSL_MULTIROOT_SOLVER_TYPE = None
_GSL_MULTIROOT_UNAVAILABLE = False


def _double_fact(n: int) -> int:
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _calc_prefix(l: int, m: int, m1: float, m2: float, eta: float):
    """COMPLETE port of CalculateThisMultipolePrefix from
    LALSimIMREOBNewtonianMultipole.c:512-628."""
    epsilon = (l + m) % 2
    sign = 1 if (m % 2 == 0) else -1
    total = m1 + m2
    x1 = m1 / total
    x2 = m2 / total
    if (m1 != m2) or sign == 1:
        c = x2 ** (l + epsilon - 1) + sign * x1 ** (l + epsilon - 1)
    else:
        c_lookup = {2: -1.0, 3: -1.0, 4: -0.5, 5: -0.5}
        c = c_lookup.get(l, 0.0)

    if epsilon == 0:
        n = (1j * m) ** l
        mult1 = 8.0 * _PI / _double_fact(2 * l + 1)
        mult2 = math.sqrt(((l + 1) * (l + 2)) / (l * (l - 1)))
        n *= mult1 * mult2
    else:
        n = (1j * m) ** l
        n = -n
        mult1 = 16.0 * _PI / _double_fact(2 * l + 1)
        mult2 = math.sqrt(((2 * l + 1) * (l + 2) * (l * l - m * m)) / ((2 * l - 1) * (l + 1) * l * (l - 1)))
        n *= 1j * mult1 * mult2
    prefix = n * eta * c
    return prefix


def _abs_scalar_sph_pi_over2(l: int, m: int) -> float:
    """COMPLETE port of XLALAbsScalarSphHarmThetaPiBy2 from
    LALSimIMREOBNewtonianMultipole.c:272-307."""
    leg = associated_legendre_at_zero(l, abs(m))
    if m < 0 and (abs(m) % 2 == 1):
        leg *= -1.0
    norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    return abs(norm * leg)


@torch.jit.script
def _factorized_rho_22(
    v: torch.Tensor,
    eta: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    dM: torch.Tensor,
    a_delta: torch.Tensor,
    a: torch.Tensor,
    eulerlog: torch.Tensor,
) -> torch.Tensor:
    pi_val = math.pi
    eta2 = eta * eta
    eta3 = eta2 * eta
    a2 = a * a
    a3 = a2 * a
    rho22v2 = -43.0 / 42.0 + (55.0 * eta) / 84.0
    rho22v3 = (-2.0 * (chiS + chiA * dM - chiS * eta)) / 3.0
    rho22v4 = -20555.0 / 10584.0 + 0.5 * a_delta * a_delta - (33025.0 * eta) / 21168.0 + (19583.0 * eta2) / 42336.0
    rho22v5 = ((-34.0 / 21.0 + 49.0 * eta / 18.0 + 209.0 * eta2 / 126.0) * chiS) + ((-34.0 / 21.0 - 19.0 * eta / 42.0) * dM * chiA)
    rho22v6 = (
        1556919113.0 / 122245200.0
        + (89.0 * a2) / 252.0
        - (48993925.0 * eta) / 9779616.0
        - (6292061.0 * eta2) / 3259872.0
        + (10620745.0 * eta3) / 39118464.0
        + (41.0 * eta * pi_val * pi_val) / 192.0
    )
    rho22v6l = -428.0 / 105.0
    rho22v7 = (
        a3 / 3.0
        + chiA * dM * (18733.0 / 15876.0 + (50140.0 * eta) / 3969.0 + (97865.0 * eta2) / 63504.0)
        + chiS * (18733.0 / 15876.0 + (74749.0 * eta) / 5292.0 - (245717.0 * eta2) / 63504.0 + (50803.0 * eta3) / 63504.0)
    )
    rho22v8 = -387216563023.0 / 160190110080.0 + (18353.0 * a2) / 21168.0 - (a2 * a2) / 8.0
    rho22v8l = 9202.0 / 2205.0
    rho22v10 = -16094530514677.0 / 533967033600.0
    rho22v10l = 439877.0 / 55566.0

    return 1.0 + v * v * (
        rho22v2
        + v
        * (
            rho22v3
            + v
            * (
                rho22v4
                + v
                * (
                    rho22v5
                    + v
                    * (
                        rho22v6
                        + rho22v6l * eulerlog
                        + v
                        * (
                            rho22v7
                            + v * (rho22v8 + rho22v8l * eulerlog + (rho22v10 + rho22v10l * eulerlog) * v * v)
                        )
                    )
                )
            )
        )
    )


@torch.jit.script
def _factorized_rho_21(
    v: torch.Tensor,
    eta: torch.Tensor,
    a_mode: torch.Tensor,
    eulerlog: torch.Tensor,
) -> torch.Tensor:
    eta2 = eta * eta
    a2_mode = a_mode * a_mode
    a3_mode = a2_mode * a_mode
    rho21v1 = 0.0
    rho21v2 = -59.0 / 56.0 + (23.0 * eta) / 84.0
    rho21v3 = 0.0
    rho21v4 = -47009.0 / 56448.0 - (865.0 * a2_mode) / 1792.0 - (405.0 * a2_mode * a2_mode) / 2048.0 - (10993.0 * eta) / 14112.0 + (617.0 * eta2) / 4704.0
    rho21v5 = (-98635.0 * a_mode) / 75264.0 + (2031.0 * a_mode * a2_mode) / 7168.0 - (1701.0 * a2_mode * a3_mode) / 8192.0
    rho21v6 = (
        7613184941.0 / 2607897600.0
        + (9032393.0 * a2_mode) / 1806336.0
        + (3897.0 * a2_mode * a2_mode) / 16384.0
        - (15309.0 * a3_mode * a3_mode) / 65536.0
    )
    rho21v6l = -107.0 / 105.0
    rho21v7 = (-3859374457.0 * a_mode) / 1159065600.0 - (55169.0 * a3_mode) / 16384.0 + (18603.0 * a2_mode * a3_mode) / 65536.0 - (72171.0 * a2_mode * a2_mode * a3_mode) / 262144.0
    rho21v7l = 107.0 * a_mode / 140.0
    rho21v8 = -1168617463883.0 / 911303737344.0
    rho21v8l = 6313.0 / 5880.0
    rho21v10 = -63735873771463.0 / 16569158860800.0
    rho21v10l = 5029963.0 / 5927040.0

    return 1.0 + v * (
        rho21v1
        + v
        * (
            rho21v2
            + v
            * (
                rho21v3
                + v
                * (
                    rho21v4
                    + v
                    * (
                        rho21v5
                        + v
                        * (
                            rho21v6
                            + rho21v6l * eulerlog
                            + v
                            * (
                                rho21v7
                                + rho21v7l * eulerlog
                                + v * (rho21v8 + rho21v8l * eulerlog + (rho21v10 + rho21v10l * eulerlog) * v * v)
                            )
                        )
                    )
                )
            )
        )
    )


@torch.jit.script
def _factorized_aux_21(
    v: torch.Tensor,
    eta: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    dM: torch.Tensor,
    waveform: bool,
    cal21: torch.Tensor,
) -> torch.Tensor:
    dM2 = dM * dM
    eta2 = eta * eta
    inv_dM = torch.where(torch.abs(dM) > 1.0e-12, 1.0 / dM, torch.zeros_like(dM))
    f21v1 = torch.where(
        dM2 > 0.0,
        (-3.0 * (chiS + chiA * inv_dM)) / 2.0,
        -1.5 * chiA,
    )
    f21v3 = torch.where(
        dM2 > 0.0,
        (chiS * dM * (427.0 + 79.0 * eta) + chiA * (147.0 + 280.0 * dM2 + 1251.0 * eta)) / (84.0 * dM),
        torch.tensor(3.0 / 8.0, device=v.device, dtype=v.dtype) * chiA,
    )
    aux = v * f21v1 + (v * v * v) * f21v3

    if waveform:
        chiS2 = chiS * chiS
        chiA2 = chiA * chiA
        chiA3 = chiA2 * chiA
        chiS3 = chiS2 * chiS
        if torch.abs(dM).item() > 1.0e-12:
            f21v4 = (-3.0 - 2.0 * eta) * chiA2 + (-3.0 + 0.5 * eta) * chiS2 + (-6.0 + 10.5 * eta) * chiS * chiA * inv_dM
            f21v5 = (
                (0.75 - 3.0 * eta) * chiA3 * inv_dM
                + (-81.0 / 16.0 + 1709.0 * eta / 1008.0 + 613.0 * eta2 / 1008.0 + (9.0 / 4.0 - 3.0 * eta) * chiA2) * chiS
                + 0.75 * chiS3
                + (-81.0 / 16.0 - 703.0 * eta2 / 112.0 + 8797.0 * eta / 1008.0 + (9.0 / 4.0 - 6.0 * eta) * chiS2) * chiA * inv_dM
            )
            f21v6 = (
                (4163.0 / 252.0 - 9287.0 * eta / 1008.0 - 85.0 * eta2 / 112.0) * chiA2
                + (4163.0 / 252.0 - 2633.0 * eta / 1008.0 + 461.0 * eta2 / 1008.0) * chiS2
                + (4163.0 / 126.0 - 1636.0 * eta / 21.0 + 1088.0 * eta2 / 63.0) * chiS * chiA * inv_dM
            )
        else:
            f21v4 = (-6.0 + 10.5 * eta) * chiS * chiA
            f21v5 = (
                (0.75 - 3.0 * eta) * chiA3
                + (-81.0 / 16.0 - 703.0 * eta2 / 112.0 + 8797.0 * eta / 1008.0 + (9.0 / 4.0 - 6.0 * eta) * chiS2) * chiA
            )
            f21v6 = (4163.0 / 126.0 - 1636.0 * eta / 21.0 + 1088.0 * eta2 / 63.0) * chiS * chiA
        aux = aux + (v ** 4) * f21v4 + (v ** 5) * f21v5 + (v ** 6) * f21v6 + (v ** 7) * cal21
    return aux


@torch.jit.script
def _factorized_rho_33(
    v: torch.Tensor,
    eta: torch.Tensor,
    a_mode: torch.Tensor,
    eulerlog: torch.Tensor,
    waveform: bool,
) -> torch.Tensor:
    pi_val = math.pi
    eta2 = eta * eta
    eta3 = eta2 * eta
    a2_mode = a_mode * a_mode
    a3_mode = a2_mode * a_mode
    rho33v2 = -7.0 / 6.0 + (2.0 * eta) / 3.0
    rho33v3 = 0.0
    rho33v4 = -6719.0 / 3960.0 + a2_mode / 2.0 - (1861.0 * eta) / 990.0 + (149.0 * eta2) / 330.0
    rho33v5 = (-4.0 * a_mode) / 3.0
    rho33v6 = 3203101567.0 / 227026800.0 + (5.0 * a2_mode) / 36.0
    rho33v6l = -26.0 / 7.0
    rho33v7 = (5297.0 * a_mode) / 2970.0 + a3_mode / 3.0
    rho33v8 = -57566572157.0 / 8562153600.0
    rho33v8l = 13.0 / 3.0
    rho33v10 = torch.zeros_like(v)
    rho33v10l = torch.zeros_like(v)
    if waveform:
        rho33v6 = rho33v6 + (-129509.0 / 25740.0 + 41.0 * pi_val * pi_val / 192.0) * eta - 274621.0 / 154440.0 * eta2 + 12011.0 / 46332.0 * eta3
        rho33v10 = torch.tensor(-903823148417327.0 / 30566888352000.0, device=v.device, dtype=v.dtype)
        rho33v10l = torch.tensor(87347.0 / 13860.0, device=v.device, dtype=v.dtype)

    return 1.0 + v * v * (
        rho33v2
        + v
        * (
            rho33v3
            + v
            * (
                rho33v4
                + v
                * (
                    rho33v5
                    + v
                    * (
                        rho33v6
                        + rho33v6l * eulerlog
                        + v
                        * (
                            rho33v7
                            + v * (rho33v8 + rho33v8l * eulerlog + (rho33v10 + rho33v10l * eulerlog) * v * v)
                        )
                    )
                )
            )
        )
    )


@torch.jit.script
def _factorized_aux_33(
    v: torch.Tensor,
    eta: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    dM: torch.Tensor,
    waveform: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dM2 = dM * dM
    eta2 = eta * eta
    inv_dM = torch.where(torch.abs(dM) > 1.0e-12, 1.0 / dM, torch.zeros_like(dM))
    f33v3 = torch.where(
        dM2 > 0.0,
        (chiS * dM * (-4.0 + 5.0 * eta) + chiA * (-4.0 + 19.0 * eta)) / (2.0 * dM),
        torch.tensor(0.375, device=v.device, dtype=v.dtype) * chiA,
    )
    f33v4 = torch.zeros_like(v)
    f33v5 = torch.zeros_like(v)
    f33v6 = torch.zeros_like(v)
    f33vh6 = torch.zeros_like(v)

    if waveform:
        chiS2 = chiS * chiS
        chiA2 = chiA * chiA
        if torch.abs(dM).item() > 1.0e-12:
            f33v4 = (1.5 * chiS2 * dM + (3.0 - 12.0 * eta) * chiA * chiS + dM * (1.5 - 6.0 * eta) * chiA2) * inv_dM
            f33v5 = (dM * (241.0 / 30.0 * eta2 + 11.0 / 20.0 * eta + 2.0 / 3.0) * chiS + (407.0 / 30.0 * eta2 - 593.0 / 60.0 * eta + 2.0 / 3.0) * chiA) * inv_dM
            f33v6 = (dM * (6.0 * eta2 - 13.5 * eta - 1.75) * chiS2 + (44.0 * eta2 - eta - 3.5) * chiA * chiS + dM * (-12.0 * eta2 + 5.5 * eta - 1.75) * chiA2) * inv_dM
            f33vh6 = (dM * (593.0 / 108.0 * eta - 81.0 / 20.0) * chiS + (7339.0 / 540.0 * eta - 81.0 / 20.0) * chiA) * inv_dM
        else:
            f33v4 = (3.0 - 12.0 * eta) * chiA * chiS
            f33v5 = (407.0 / 30.0 * eta2 - 593.0 / 60.0 * eta + 2.0 / 3.0) * chiA
            f33v6 = (44.0 * eta2 - eta - 3.5) * chiA * chiS
            f33vh6 = (7339.0 / 540.0 * eta - 81.0 / 20.0) * chiA

    aux_real = (v ** 3) * f33v3 + (v ** 4) * f33v4 + (v ** 5) * f33v5 + (v ** 6) * f33v6
    aux_imag = (v ** 6) * f33vh6
    return aux_real, aux_imag


@torch.jit.script
def _factorized_rho_44(
    v: torch.Tensor,
    eta: torch.Tensor,
    a_mode: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    dM: torch.Tensor,
    eulerlog: torch.Tensor,
    waveform: bool,
) -> torch.Tensor:
    a2_mode = a_mode * a_mode
    eta2 = eta * eta
    eta3 = eta2 * eta
    dM2 = dM * dM
    m1Plus3eta = -1.0 + 3.0 * eta
    m1Plus3eta2 = m1Plus3eta * m1Plus3eta
    rho44v2 = (1614.0 - 5870.0 * eta + 2625.0 * eta2) / (1320.0 * m1Plus3eta)
    rho44v3 = (chiA * (10.0 - 39.0 * eta) * dM + chiS * (10.0 - 41.0 * eta + 42.0 * eta2)) / (15.0 * m1Plus3eta)
    rho44v4 = (
        a2_mode / 2.0
        + (
            -511573572.0
            + 2338945704.0 * eta
            - 313857376.0 * eta2
            - 6733146000.0 * eta * eta2
            + 1252563795.0 * eta2 * eta2
        )
        / (317116800.0 * m1Plus3eta2)
    )
    rho44v5 = (-69.0 * a_mode) / 55.0
    rho44v8 = torch.zeros_like(v)
    rho44v8l = torch.zeros_like(v)
    rho44v10 = torch.zeros_like(v)
    rho44v10l = torch.zeros_like(v)
    if waveform:
        chiS2 = chiS * chiS
        chiA2 = chiA * chiA
        rho44v4 = (
            (
                -511573572.0
                + 2338945704.0 * eta
                - 313857376.0 * eta2
                - 6733146000.0 * eta3
                + 1252563795.0 * eta2 * eta2
            )
            / (317116800.0 * m1Plus3eta2)
            + 0.5 * chiS2
            + dM * chiS * chiA
            + 0.5 * dM2 * chiA2
        )
        rho44v5 = (
            chiA * dM * (-8280.0 + 42716.0 * eta - 57990.0 * eta2 + 8955.0 * eta3) / (6600.0 * m1Plus3eta2)
            + chiS * (-8280.0 + 66284.0 * eta - 176418.0 * eta2 + 128085.0 * eta3 + 88650.0 * eta2 * eta2) / (6600.0 * m1Plus3eta2)
        )
        rho44v8 = torch.tensor(-172066910136202271.0 / 19426955708160000.0, device=v.device, dtype=v.dtype)
        rho44v8l = torch.tensor(845198.0 / 190575.0, device=v.device, dtype=v.dtype)
        rho44v10 = torch.tensor(-17154485653213713419357.0 / 568432724020761600000.0, device=v.device, dtype=v.dtype)
        rho44v10l = torch.tensor(22324502267.0 / 3815311500.0, device=v.device, dtype=v.dtype)
    rho44v6 = 16600939332793.0 / 1098809712000.0 + (217.0 * a2_mode) / 3960.0
    rho44v6l = -12568.0 / 3465.0
    return 1.0 + v * v * (
        rho44v2
        + v
        * (
            rho44v3
            + v
            * (
                rho44v4
                + v
                * (
                    rho44v5
                    + v
                    * (
                        rho44v6
                        + rho44v6l * eulerlog
                        + v * v * (rho44v8 + rho44v8l * eulerlog + v * v * (rho44v10 + rho44v10l * eulerlog))
                    )
                )
            )
        )
    )


@torch.jit.script
def _factorized_rho_55(
    v: torch.Tensor,
    eta: torch.Tensor,
    a_mode: torch.Tensor,
    eulerlog: torch.Tensor,
    waveform: bool,
) -> torch.Tensor:
    a2_mode = a_mode * a_mode
    eta2 = eta * eta
    denom = (-1.0 + 2.0 * eta)
    rho55v2 = (487.0 - 1298.0 * eta + 512.0 * eta2) / (390.0 * denom)
    rho55v3 = (-2.0 * a_mode) / 3.0
    rho55v4 = -3353747.0 / 2129400.0 + a2_mode / 2.0
    rho55v5 = -241.0 * a_mode / 195.0
    rho55v6 = torch.zeros_like(v)
    rho55v6l = torch.zeros_like(v)
    rho55v8 = torch.zeros_like(v)
    rho55v8l = torch.zeros_like(v)
    rho55v10 = torch.zeros_like(v)
    rho55v10l = torch.zeros_like(v)
    if waveform:
        rho55v6 = torch.tensor(190606537999247.0 / 11957879934000.0, device=v.device, dtype=v.dtype)
        rho55v6l = torch.tensor(-1546.0 / 429.0, device=v.device, dtype=v.dtype)
        rho55v8 = torch.tensor(-1213641959949291437.0 / 118143853747920000.0, device=v.device, dtype=v.dtype)
        rho55v8l = torch.tensor(376451.0 / 83655.0, device=v.device, dtype=v.dtype)
        rho55v10 = torch.tensor(-150082616449726042201261.0 / 4837990810977324000000.0, device=v.device, dtype=v.dtype)
        rho55v10l = torch.tensor(2592446431.0 / 456756300.0, device=v.device, dtype=v.dtype)
    return 1.0 + v * v * (
        rho55v2
        + v
        * (
            rho55v3
            + v
            * (
                rho55v4
                + v
                * (
                    rho55v5
                    + v
                    * (
                        rho55v6
                        + rho55v6l * eulerlog
                        + v * v * (rho55v8 + rho55v8l * eulerlog + v * v * (rho55v10 + rho55v10l * eulerlog))
                    )
                )
            )
        )
    )


@torch.jit.script
def _factorized_aux_55(
    v: torch.Tensor,
    eta: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    dM: torch.Tensor,
    cal55: torch.Tensor,
) -> torch.Tensor:
    denom = (1.0 - 2.0 * eta)
    eta2 = eta * eta
    chiS2 = chiS * chiS
    chiA2 = chiA * chiA
    if torch.abs(dM).item() > 1.0e-12:
        f55v3 = (
            chiA / dM * (10.0 / (3.0 * denom) - 70.0 * eta / (3.0 * denom) + 110.0 * eta2 / (3.0 * denom))
            + chiS * (10.0 / (3.0 * denom) - 10.0 * eta / denom + 10.0 * eta2 / denom)
        )
        f55v4 = (
            chiS2 * (-5.0 / (2.0 * denom) + 5.0 * eta / denom)
            + chiA * chiS / dM * (-5.0 / denom + 30.0 * eta / denom - 40.0 * eta2 / denom)
            + chiA2 * (-5.0 / (2.0 * denom) + 15.0 * eta / denom - 20.0 * eta2 / denom)
        )
    else:
        f55v3 = chiA * (10.0 / (3.0 * denom) - 70.0 * eta / (3.0 * denom) + 110.0 * eta2 / (3.0 * denom))
        f55v4 = chiA * chiS * (-5.0 / denom + 30.0 * eta / denom - 40.0 * eta2 / denom)
    return (v ** 3) * f55v3 + (v ** 4) * f55v4 + (v ** 5) * cal55


@torch.jit.script
def _factorized_delta(
    l: int,
    m: int,
    v: torch.Tensor,
    vh: torch.Tensor,
    vh3: torch.Tensor,
    Omega: torch.Tensor,
    eta: torch.Tensor,
    aDelta: torch.Tensor,
    waveform: bool,
) -> torch.Tensor:
    pi_val = math.pi
    delta = torch.zeros_like(v)
    if l == 2 and m == 2:
        delta22vh3 = 7.0 / 3.0
        delta22vh6 = (-4.0 * aDelta) / 3.0 + (428.0 * pi_val) / 105.0
        delta22vh9 = -2203.0 / 81.0 + (1712.0 * pi_val * pi_val) / 315.0
        delta22v5 = -24.0 * eta
        delta22v6 = 0.0
        delta22v8 = (20.0 * aDelta) / 63.0
        delta = vh3 * (delta22vh3 + vh3 * (delta22vh6 + vh * vh * (delta22vh9 * vh))) + Omega * (delta22v5 * v * v + Omega * (delta22v6 + delta22v8 * v * v))
    elif l == 2 and m == 1:
        delta21vh3 = 2.0 / 3.0
        delta21vh6 = (-17.0 * aDelta) / 35.0 + (107.0 * pi_val) / 105.0
        delta21vh7 = (3.0 * aDelta * aDelta) / 140.0
        delta21vh9 = -272.0 / 81.0 + (214.0 * pi_val * pi_val) / 315.0
        delta21v5 = -493.0 * eta / 42.0
        delta21v7 = 0.0
        delta = vh3 * (delta21vh3 + vh3 * (delta21vh6 + vh * (delta21vh7 + delta21vh9 * vh * vh))) + (Omega * v * v) * (delta21v5 + delta21v7 * v * v)
    elif l == 3 and m == 3:
        delta33vh3 = 13.0 / 10.0
        delta33vh6 = (-81.0 * aDelta) / 20.0 + (39.0 * pi_val) / 7.0
        delta33vh9 = -227827.0 / 3000.0 + (78.0 * pi_val * pi_val) / 7.0
        delta33v5 = -80897.0 * eta / 2430.0
        delta = vh3 * (delta33vh3 + vh3 * (delta33vh6 + vh3 * delta33vh9)) + Omega * v * v * delta33v5
    elif l == 4 and m == 4:
        m1Plus3eta = -1.0 + 3.0 * eta
        delta44vh3 = (112.0 + 219.0 * eta) / (-120.0 * m1Plus3eta)
        delta44vh6 = (-464.0 * aDelta) / 75.0 + (25136.0 * pi_val) / 3465.0
        delta44vh9 = -55144.0 / 375.0 + 201088.0 * pi_val * pi_val / 10395.0 if waveform else 0.0
        delta = vh3 * (delta44vh3 + vh3 * (delta44vh6 + vh3 * delta44vh9))
    elif l == 5 and m == 5:
        denom = (1.0 - 2.0 * eta)
        delta55vh3 = (96875.0 + 857528.0 * eta) / (131250.0 * denom)
        delta55vh6 = (3865.0 * pi_val / 429.0) if waveform else 0.0
        delta55vh9 = ((-7686949127.0 + 954500400.0 * pi_val * pi_val) / 31783752.0) if waveform else 0.0
        delta = vh3 * (delta55vh3 + vh3 * (delta55vh6 + vh3 * delta55vh9))
    return delta


def _factorized_rho_aux_delta(
    l: int,
    m: int,
    v: torch.Tensor,
    params: EOBParams,
    *,
    waveform: bool,
    H: torch.Tensor | None = None,
    chiS: torch.Tensor | None = None,
    chiA: torch.Tensor | None = None,
    tplspin: torch.Tensor | None = None,
    cal21: torch.Tensor | float | None = None,
    cal55: torch.Tensor | float | None = None,
):
    """Factorized rho_lm, aux_f_lm, and delta_lm (torch-native).

    COMPLETE port of the mode blocks in
    LALSimIMRSpinEOBFactorizedWaveformPrec.c:726-851 together with the
    coefficients from LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c
    (22:144-206, 21:248-329, 33:375-433, 44:480-505, 55:568-606).
    """

    device, dtype = v.device, v.dtype
    complex_dtype = torch.complex64 if dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.complex128

    eta = torch.as_tensor(params.eta, device=device, dtype=dtype)
    m1 = torch.as_tensor(params.mass1, device=device, dtype=dtype)
    m2 = torch.as_tensor(params.mass2, device=device, dtype=dtype)
    m_tot = m1 + m2
    dM = (m1 - m2) / m_tot
    if chiS is None or chiA is None:
        chi1z = torch.as_tensor(params.spin1z, device=device, dtype=dtype)
        chi2z = torch.as_tensor(params.spin2z, device=device, dtype=dtype)
        chiS = 0.5 * (chi1z + chi2z)
        chiA = 0.5 * (chi1z - chi2z)
    else:
        chiS = torch.as_tensor(chiS, device=device, dtype=dtype)
        chiA = torch.as_tensor(chiA, device=device, dtype=dtype)
    a_delta = chiS + chiA * dM
    a = torch.as_tensor(tplspin, device=device, dtype=dtype) if tplspin is not None else a_delta
    zero_a = torch.zeros_like(a)

    abs_m = float(abs(m))
    eulerlog = torch.as_tensor(_EULER_GAMMA, device=device, dtype=dtype) + torch.log(
        torch.as_tensor(2.0 * abs_m, device=device, dtype=dtype) * torch.clamp(torch.abs(v), min=1.0e-16)
    )

    aux = torch.zeros_like(v, dtype=complex_dtype if waveform else dtype, device=device)

    if l == 2 and m == 2:
        rho = _factorized_rho_22(v, eta, chiS, chiA, dM, a_delta, a, eulerlog)

    elif l == 2 and m == 1:
        a_mode = zero_a if waveform else a
        rho = _factorized_rho_21(v, eta, a_mode, eulerlog)
        cal21_t = torch.as_tensor(0.0 if cal21 is None else cal21, device=device, dtype=dtype)
        aux = _factorized_aux_21(v, eta, chiS, chiA, dM, waveform, cal21_t).to(aux.dtype)

    elif l == 3 and m == 3:
        a_mode = zero_a if waveform else a
        rho = _factorized_rho_33(v, eta, a_mode, eulerlog, waveform)
        aux_real, aux_imag = _factorized_aux_33(v, eta, chiS, chiA, dM, waveform)
        aux = (aux_real + 1j * aux_imag).to(complex_dtype) if waveform else aux_real.to(dtype)

    elif l == 4 and m == 4:
        a_mode = zero_a if waveform else a
        rho = _factorized_rho_44(v, eta, a_mode, chiS, chiA, dM, eulerlog, waveform)

    elif l == 5 and m == 5:
        a_mode = zero_a if waveform else a
        rho = _factorized_rho_55(v, eta, a_mode, eulerlog, waveform)
        if waveform:
            cal55_t = torch.as_tensor(0.0 if cal55 is None else cal55, device=device, dtype=dtype)
            aux = _factorized_aux_55(v, eta, chiS, chiA, dM, cal55_t).to(complex_dtype)

    else:
        raise ValueError(f"Unsupported mode ({l},{m}) for rho/aux/delta torch path")

    # delta_lm phase (aligned backbone)
    Omega = v * v * v
    vh3 = (H * Omega) if H is not None else Omega
    vh = torch.clamp(torch.abs(vh3), min=1.0e-15) ** (1.0 / 3.0)
    aDelta = chiA * dM + chiS * (1.0 - 2.0 * eta)

    delta = _factorized_delta(l, m, v, vh, vh3, Omega, eta, aDelta, waveform)

    return rho.to(dtype=dtype), aux, delta.to(dtype=dtype)


def _rho_aux_flux(
    l: int,
    m: int,
    v: torch.Tensor,
    params: EOBParams,
    chi1z: torch.Tensor | None = None,
    chi2z: torch.Tensor | None = None,
    *,
    H: torch.Tensor | None = None,
    chiS: torch.Tensor | None = None,
    chiA: torch.Tensor | None = None,
    tplspin: torch.Tensor | None = None,
):
    """Return LAL's flux residual amplitude pieces."""
    chi1_eval = params.spin1z if chi1z is None else chi1z
    chi2_eval = params.spin2z if chi2z is None else chi2z
    chiS_eval = 0.5 * (chi1_eval + chi2_eval) if chiS is None else chiS
    chiA_eval = 0.5 * (chi1_eval - chi2_eval) if chiA is None else chiA
    m1 = torch.as_tensor(params.mass1, device=v.device, dtype=v.dtype)
    m2 = torch.as_tensor(params.mass2, device=v.device, dtype=v.dtype)
    eta = torch.as_tensor(params.eta, device=v.device, dtype=v.dtype)
    dM = (m1 - m2) / torch.clamp(m1 + m2, min=1.0e-15)
    tplspin_eval = (1.0 - 2.0 * eta) * chiS_eval + dM * chiA_eval if tplspin is None else tplspin
    try:
        rho_t, aux_t, _ = _factorized_rho_aux_delta(
            l,
            m,
            v,
            params,
            waveform=False,
            H=H,
            chiS=chiS_eval,
            chiA=chiA_eval,
            tplspin=tplspin_eval,
        )
        return rho_t, aux_t
    except ValueError:
        pass

    rho, aux = rho_lm_full(
        l,
        m,
        params.eta,
        chi1_eval,
        chi2_eval,
        params.mass1,
        params.mass2,
        v,
        tplspin=tplspin_eval,
    )
    complex_dtype = torch.complex64 if v.dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.complex128
    rho_complex = isinstance(rho, complex) or (torch.is_tensor(rho) and rho.is_complex())
    aux_complex = isinstance(aux, complex) or (torch.is_tensor(aux) and aux.is_complex())
    rho_t = torch.as_tensor(rho, device=v.device, dtype=complex_dtype if rho_complex else v.dtype)
    aux_t = torch.as_tensor(aux, device=v.device, dtype=complex_dtype if aux_complex else v.dtype)
    return rho_t, aux_t


def normalize_mode_array(mode_array):
    """Normalize PHM mode selection like LAL's mode-array validation.

    SEOBNRv4PHM accepts only the positive co-precessing modes in ``LM_DEFAULT``.
    The corresponding negative-m modes are generated internally by the waveform
    rotation path, matching LAL's ModeArray convention.
    """

    if mode_array is None or (isinstance(mode_array, str) and mode_array == ""):
        return tuple(LM_DEFAULT)
    mode_array = tuple(mode_array)
    if len(mode_array) == 0:
        return tuple(LM_DEFAULT)

    requested = set()
    for raw_mode in mode_array:
        try:
            ell, emm = raw_mode
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SEOBNRv4PHM mode entry {raw_mode!r}") from exc
        ell = int(ell)
        emm = int(emm)
        if emm < 0:
            raise ValueError(
                f"Mode ({ell},{emm}) has negative m; specify ({ell},{abs(emm)}) "
                "and let SEOBNRv4PHM generate the -m partner internally"
            )
        if (ell, emm) not in LM_DEFAULT:
            raise ValueError(f"Mode ({ell},{emm}) is not available for SEOBNRv4PHM")
        requested.add((ell, emm))

    return tuple(mode for mode in LM_DEFAULT if mode in requested)


def _lal_spin_weight(spin, mass: float, total_mass: float):
    """LAL v4P spin state order: chi * m_i * m_i / M / M."""

    return spin * mass * mass / total_mass / total_mass


def _lal_spin_scale(mass: float, total_mass: float) -> float:
    """Scalar inverse for LAL's v4P spin-state weighting."""

    return mass * mass / total_mass / total_mass


@dataclass
class EOBParams:
    """Container for SEOBNRv4PHM physical parameters (solar-mass units)."""

    mass1: float
    mass2: float
    spin1x: float
    spin1y: float
    spin1z: float
    spin2x: float
    spin2y: float
    spin2z: float
    distance: float
    inclination: float
    f_lower: float
    f_ref: float
    mode_array: tuple = tuple(LM_DEFAULT)
    tortoise: int = 1

    def __post_init__(self):
        self.M = float(self.mass1 + self.mass2)
        self.eta = float(self.mass1 * self.mass2 / (self.M * self.M))
        self.M_sec = self.M * _MTSUN_SI  # total mass in seconds
        self.mode_array = normalize_mode_array(self.mode_array)
        self.aligned_spins = (math.hypot(self.spin1x, self.spin1y) < EPS_ALIGN) and (
            math.hypot(self.spin2x, self.spin2y) < EPS_ALIGN
        )
        if self.aligned_spins:
            # LAL's SpinsAlmostAligned branch zeroes in-plane spins before ICs,
            # dynamics, and Euler rotations.
            self.spin1x = 0.0
            self.spin1y = 0.0
            self.spin2x = 0.0
            self.spin2y = 0.0
        # LAL precomputes the first Hamiltonian coefficient cache before the
        # IC root using the full initial sigmaKerr and Lhat=(0,0,1).
        s1_m2 = (
            _lal_spin_weight(self.spin1x, self.mass1, self.M),
            _lal_spin_weight(self.spin1y, self.mass1, self.M),
            _lal_spin_weight(self.spin1z, self.mass1, self.M),
        )
        s2_m2 = (
            _lal_spin_weight(self.spin2x, self.mass2, self.M),
            _lal_spin_weight(self.spin2y, self.mass2, self.M),
            _lal_spin_weight(self.spin2z, self.mass2, self.M),
        )
        sigma_vec = tuple(s1_m2[i] + s2_m2[i] for i in range(3))
        self.a_sigma = math.sqrt(sum(x * x for x in sigma_vec))
        denom = 1.0 - 2.0 * self.eta
        if self.a_sigma > 1.0e-6:
            chi0 = sigma_vec[2] / denom
            s_perp_dot_sigma = (
                s1_m2[0] * sigma_vec[0]
                + s1_m2[1] * sigma_vec[1]
                + s2_m2[0] * sigma_vec[0]
                + s2_m2[1] * sigma_vec[1]
            )
            chi0 += s_perp_dot_sigma / self.a_sigma / denom / 2.0
        else:
            chi0 = 0.0
        self.hcoeffs = compute_spin_aligned_hcoeffs(self.eta, self.a_sigma, chi=chi0)
        # spin combinations used by waveform/NQC fits
        self.chiS = 0.5 * (self.spin1z + self.spin2z)
        self.chiA = 0.5 * (self.spin1z - self.spin2z)
        # precompute simplest NQC amplitude coefficients (a1-a3) for 22 mode
        try:
            self.nqc_a = solve_nqc_coeffs(self.eta, self.chiS, self.chiA, mode_l=2, mode_m=2)
        except Exception:
            self.nqc_a = {"a1": 0.0, "a2": 0.0, "a3": 0.0}
        self.nqc_b = {"b1": 0.0, "b2": 0.0, "b3": 0.0, "b4": 0.0}


def settings():
    """Return default integrator settings (matches LAL defaults)."""
    return dict(rtol=EPS_REL, atol=EPS_ABS, h_min=DELTA_T_MIN, t_step_back=T_STEP_BACK)


def highest_initial_freq(m_total_msun: float) -> float:
    """COMPLETE port of XLALEOBHighestInitialFreq
    (LALSimIMRSpinPrecEOBv4P.c:189-195). Chooses the initial 22-mode
    frequency for a minimum separation of 10.5 M via Kepler's law."""

    mTScaled = m_total_msun * _MTSUN_SI
    return (MIN_INIT_SEPARATION_M ** (-1.5)) / (_PI * mTScaled)


@torch.jit.script
def _dot3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """LAL-order 3-vector inner product."""

    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


@torch.jit.script
def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """LAL-order 3-vector cross product."""

    return torch.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


@torch.jit.script
def _matvec3_lal_order(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """3x3 matrix/vector product with the same loop order used in LAL."""

    zero = torch.zeros_like(vec[..., 0])
    out = [
        zero + vec[..., 0] * mat[..., 0, 0] + vec[..., 1] * mat[..., 0, 1] + vec[..., 2] * mat[..., 0, 2],
        zero + vec[..., 0] * mat[..., 1, 0] + vec[..., 1] * mat[..., 1, 1] + vec[..., 2] * mat[..., 1, 2],
        zero + vec[..., 0] * mat[..., 2, 0] + vec[..., 1] * mat[..., 2, 1] + vec[..., 2] * mat[..., 2, 2],
    ]
    return torch.stack(out, dim=-1)


@torch.jit.script
def _pdot_t1_lal_order(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """LAL loop-order contraction for the first tortoise pDot term."""

    zero = torch.zeros((), device=vec.device, dtype=vec.dtype)
    out: List[torch.Tensor] = []
    for i in range(3):
        acc = zero
        for j in range(3):
            acc = acc + (-vec[j]) * mat[i, j]
        out.append(acc)
    return torch.stack(out)


@torch.jit.script
def _pdot_t3_lal_order(
    dTijdXk: torch.Tensor,
    invTmat: torch.Tensor,
    p_vec: torch.Tensor,
    dxdt: torch.Tensor,
) -> torch.Tensor:
    """LAL loop-order contraction for the third tortoise pDot term."""

    zero = torch.zeros((), device=p_vec.device, dtype=p_vec.dtype)
    out: List[torch.Tensor] = []
    for i in range(3):
        acc_i = zero
        for j in range(3):
            acc_ij = zero
            for l in range(3):
                acc_ijl = zero
                for k in range(3):
                    acc_ijl = acc_ijl + dTijdXk[i, k, j] * invTmat[k, l]
                acc_ij = acc_ij + acc_ijl * p_vec[l]
            acc_i = acc_i + acc_ij * dxdt[j]
        out.append(acc_i)
    return torch.stack(out)


def _phase_split_lal_scalar_order(
    r_vec: torch.Tensor,
    p_vec: torch.Tensor,
    dxdt: torch.Tensor,
    dpdt: torch.Tensor,
    omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LAL scalar-order phi/zeta split from dLhat/dt."""

    scalar_inputs = (r_vec, p_vec, dxdt, dpdt, omega)
    if (
        all(x.device.type == "cpu" and x.dtype == torch.float64 and not x.requires_grad for x in scalar_inputs)
        and r_vec.numel() == p_vec.numel() == dxdt.numel() == dpdt.numel() == 3
        and omega.numel() == 1
    ):
        values = [float(v) for v in r_vec.detach()] + [float(v) for v in p_vec.detach()]
        dvalues = [float(v) for v in dxdt.detach()] + [float(v) for v in dpdt.detach()]

        Lx = values[1] * values[5] - values[2] * values[4]
        Ly = values[2] * values[3] - values[0] * values[5]
        Lz = values[0] * values[4] - values[1] * values[3]
        magL = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        Lhatx = Lx / magL
        Lhaty = Ly / magL
        Lhatz = Lz / magL

        dLx = dvalues[1] * values[5] - dvalues[2] * values[4]
        dLx = dLx + values[1] * dvalues[5] - values[2] * dvalues[4]

        dLy = dvalues[2] * values[3] - dvalues[0] * values[5]
        dLy = dLy + values[2] * dvalues[3] - values[0] * dvalues[5]

        dLz = dvalues[0] * values[4] - dvalues[1] * values[3]
        dLz = dLz + values[0] * dvalues[4] - values[1] * dvalues[3]

        dMagL = (Lx * dLx + Ly * dLy + Lz * dLz) / magL
        dLhatx = (dLx * magL - Lx * dMagL) / (magL * magL)
        dLhaty = (dLy * magL - Ly * dMagL) / (magL * magL)
        if Lhatx == 0.0 and Lhaty == 0.0:
            alphadotcosi_f = 0.0
        else:
            alphadotcosi_f = Lhatz * (Lhatx * dLhaty - Lhaty * dLhatx) / (Lhatx * Lhatx + Lhaty * Lhaty)
        omega_f = float(omega.detach())
        phase_dot_f = omega_f - alphadotcosi_f
        return (
            torch.as_tensor(phase_dot_f, device=omega.device, dtype=omega.dtype),
            torch.as_tensor(alphadotcosi_f, device=omega.device, dtype=omega.dtype),
        )

    L_vec = _cross3(r_vec, p_vec)
    Lx, Ly, Lz = L_vec[0], L_vec[1], L_vec[2]
    L_mag = _lal_scalar_sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    Lhatx = Lx / L_mag
    Lhaty = Ly / L_mag
    Lhatz = Lz / L_mag
    dLx = dxdt[1] * p_vec[2] - dxdt[2] * p_vec[1] + r_vec[1] * dpdt[2] - r_vec[2] * dpdt[1]
    dLy = dxdt[2] * p_vec[0] - dxdt[0] * p_vec[2] + r_vec[2] * dpdt[0] - r_vec[0] * dpdt[2]
    dLz = dxdt[0] * p_vec[1] - dxdt[1] * p_vec[0] + r_vec[0] * dpdt[1] - r_vec[1] * dpdt[0]
    dMagL = (Lx * dLx + Ly * dLy + Lz * dLz) / L_mag
    dLhatx = (dLx * L_mag - Lx * dMagL) / (L_mag * L_mag)
    dLhaty = (dLy * L_mag - Ly * dMagL) / (L_mag * L_mag)
    lhat_xy2 = Lhatx * Lhatx + Lhaty * Lhaty
    alphadotcosi = torch.where(
        (Lhatx == 0.0) & (Lhaty == 0.0),
        torch.zeros((), device=omega.device, dtype=omega.dtype),
        Lhatz * (Lhatx * dLhaty - Lhaty * dLhatx) / lhat_xy2,
    )
    return omega - alphadotcosi, alphadotcosi


def _lal_scalar_sqrt(x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
    if (
        x.device.type == "cpu"
        and x.dtype == torch.float64
        and (detach or not x.requires_grad)
    ):
        x_scalar = x.detach() if detach else x
        if x_scalar.numel() == 1:
            return torch.as_tensor(math.sqrt(float(x_scalar)), device=x.device, dtype=x.dtype)
        flat = x_scalar.reshape(-1)
        values = [math.sqrt(float(v)) for v in flat]
        return torch.tensor(values, device=x.device, dtype=x.dtype).reshape(x.shape)
    return torch.sqrt(x)


def _lal_scalar_unary(
    x: torch.Tensor,
    math_fn: Callable[[float], float],
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    detach: bool = False,
) -> torch.Tensor:
    if (
        x.device.type == "cpu"
        and x.dtype == torch.float64
        and (detach or not x.requires_grad)
    ):
        x_scalar = x.detach() if detach else x
        if x_scalar.numel() == 1:
            return torch.as_tensor(math_fn(float(x_scalar)), device=x.device, dtype=x.dtype)
        flat = x_scalar.reshape(-1)
        values = [math_fn(float(v)) for v in flat]
        return torch.tensor(values, device=x.device, dtype=x.dtype).reshape(x.shape)
    return torch_fn(x)


def _lal_scalar_log(x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
    return _lal_scalar_unary(x, math.log, torch.log, detach=detach)


def _lal_scalar_log1p(x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
    return _lal_scalar_unary(x, math.log1p, torch.log1p, detach=detach)


def _lal_scalar_log2(x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
    return _lal_scalar_unary(x, math.log2, torch.log2, detach=detach)


@torch.jit.script
def _safe_norm(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Euclidean norm with lower bound clamp."""
    return torch.sqrt(torch.clamp(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1] + x[..., 2] * x[..., 2], min=eps))


@torch.jit.script
def _unit_vector3(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Unit 3-vector normalized by safe norm."""
    norm = torch.sqrt(torch.clamp(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1] + x[..., 2] * x[..., 2], min=eps))
    return x / torch.clamp(norm.unsqueeze(-1), min=eps)


def _assert_finite(label: str, *tensors: torch.Tensor):
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            raise RuntimeError(f"NaN/inf detected in {label}")


def _env_on(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "", "false", "False")


def _gsl_library_candidates():
    """Prefer the GSL dylib bundled with LAL before system search paths."""

    env_path = os.environ.get("PYCBC_SEOBNRV4PHM_GSL_LIBRARY")
    if env_path:
        yield env_path
    try:
        import lal as _lal

        lal_dir = os.path.dirname(_lal.__file__)
        for lib_dir in (os.path.join(lal_dir, ".dylibs"), os.path.join(lal_dir, ".libs")):
            if os.path.isdir(lib_dir):
                for name in os.listdir(lib_dir):
                    if name.startswith("libgsl") and "gslcblas" not in name and (
                        name.endswith(".dylib") or ".so" in name
                    ):
                        yield os.path.join(lib_dir, name)
    except Exception:
        pass
    lib_path = ctypes.util.find_library("gsl")
    if lib_path is not None:
        yield lib_path


def _gsl_cblas_library_candidates():
    """Yield the companion CBLAS library used by the selected GSL build."""

    candidates = []
    env_path = os.environ.get("PYCBC_SEOBNRV4PHM_GSLCBLAS_LIBRARY")
    if env_path:
        candidates.append(env_path)
    for gsl_path in _gsl_library_candidates():
        lib_dir = os.path.dirname(gsl_path)
        if not lib_dir or not os.path.isdir(lib_dir):
            continue
        for name in sorted(os.listdir(lib_dir)):
            lower = name.lower()
            if "gslcblas" in lower and (
                ".so" in lower
                or lower.endswith(".dylib")
                or lower.endswith(".dll")
            ):
                candidates.append(os.path.join(lib_dir, name))
    lib_path = ctypes.util.find_library("gslcblas")
    if lib_path is not None:
        candidates.append(lib_path)

    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            yield path


def _load_gsl_deriv_central():
    """Return a ctypes handle for installed GSL's gsl_deriv_central."""

    global _GSL_DERIV_LIB
    global _GSL_DERIV_FUNC_TYPE
    global _GSL_DERIV_STRUCT_TYPE
    global _GSL_DERIV_UNAVAILABLE

    if _GSL_DERIV_UNAVAILABLE:
        return None
    if _GSL_DERIV_LIB is not None:
        return _GSL_DERIV_LIB, _GSL_DERIV_FUNC_TYPE, _GSL_DERIV_STRUCT_TYPE
    for lib_path in _gsl_library_candidates():
        try:
            lib = ctypes.CDLL(lib_path)
            func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_void_p)

            class _GSLFunction(ctypes.Structure):
                _fields_ = [("function", func_type), ("params", ctypes.c_void_p)]

            lib.gsl_deriv_central.argtypes = [
                ctypes.POINTER(_GSLFunction),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.gsl_deriv_central.restype = ctypes.c_int
        except Exception:
            continue

        _GSL_DERIV_LIB = lib
        _GSL_DERIV_FUNC_TYPE = func_type
        _GSL_DERIV_STRUCT_TYPE = _GSLFunction
        return _GSL_DERIV_LIB, _GSL_DERIV_FUNC_TYPE, _GSL_DERIV_STRUCT_TYPE

    _GSL_DERIV_UNAVAILABLE = True
    return None


def _load_gsl_multiroot_hybrids():
    """Return ctypes bindings for LAL's gsl_multiroot_fsolver_hybrids."""

    global _GSL_MULTIROOT_LIB
    global _GSL_MULTIROOT_FUNC_TYPE
    global _GSL_MULTIROOT_STRUCT_TYPE
    global _GSL_MULTIROOT_SOLVER_TYPE
    global _GSL_MULTIROOT_UNAVAILABLE

    if _GSL_MULTIROOT_UNAVAILABLE:
        return None
    if _GSL_MULTIROOT_LIB is not None:
        return (
            _GSL_MULTIROOT_LIB,
            _GSL_MULTIROOT_FUNC_TYPE,
            _GSL_MULTIROOT_STRUCT_TYPE,
            _GSL_MULTIROOT_SOLVER_TYPE,
        )

    for lib_path in _gsl_library_candidates():
        try:
            lib = ctypes.CDLL(lib_path)
            func_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

            class _GSLMultiRootFunction(ctypes.Structure):
                _fields_ = [("f", func_type), ("n", ctypes.c_size_t), ("params", ctypes.c_void_p)]

            lib.gsl_vector_alloc.argtypes = [ctypes.c_size_t]
            lib.gsl_vector_alloc.restype = ctypes.c_void_p
            lib.gsl_vector_free.argtypes = [ctypes.c_void_p]
            lib.gsl_vector_set.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]
            lib.gsl_vector_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.gsl_vector_get.restype = ctypes.c_double
            lib.gsl_multiroot_fsolver_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.gsl_multiroot_fsolver_alloc.restype = ctypes.c_void_p
            lib.gsl_multiroot_fsolver_free.argtypes = [ctypes.c_void_p]
            lib.gsl_multiroot_fsolver_set.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_GSLMultiRootFunction),
                ctypes.c_void_p,
            ]
            lib.gsl_multiroot_fsolver_set.restype = ctypes.c_int
            lib.gsl_multiroot_fsolver_iterate.argtypes = [ctypes.c_void_p]
            lib.gsl_multiroot_fsolver_iterate.restype = ctypes.c_int
            lib.gsl_multiroot_fsolver_root.argtypes = [ctypes.c_void_p]
            lib.gsl_multiroot_fsolver_root.restype = ctypes.c_void_p
            lib.gsl_multiroot_fsolver_f.argtypes = [ctypes.c_void_p]
            lib.gsl_multiroot_fsolver_f.restype = ctypes.c_void_p
            lib.gsl_multiroot_test_residual.argtypes = [ctypes.c_void_p, ctypes.c_double]
            lib.gsl_multiroot_test_residual.restype = ctypes.c_int
            solver_type = ctypes.c_void_p.in_dll(lib, "gsl_multiroot_fsolver_hybrids")
        except Exception:
            continue

        _GSL_MULTIROOT_LIB = lib
        _GSL_MULTIROOT_FUNC_TYPE = func_type
        _GSL_MULTIROOT_STRUCT_TYPE = _GSLMultiRootFunction
        _GSL_MULTIROOT_SOLVER_TYPE = solver_type
        return (
            _GSL_MULTIROOT_LIB,
            _GSL_MULTIROOT_FUNC_TYPE,
            _GSL_MULTIROOT_STRUCT_TYPE,
            _GSL_MULTIROOT_SOLVER_TYPE,
        )

    _GSL_MULTIROOT_UNAVAILABLE = True
    return None


def _native_gsl_multiroot_hybrids(
    residual: Callable[[list[float]], list[float]],
    guess: list[float],
    *,
    epsabs: float,
    max_iter: int,
) -> tuple[list[float], list[float]] | None:
    """Call an installed GSL ``hybrids`` solver through ctypes."""

    loaded = _load_gsl_multiroot_hybrids()
    if loaded is None:
        return None
    lib, func_type, struct_type, solver_type = loaded
    callback_error = []

    def _callback(x_vec, _params, f_vec):
        try:
            vals = [lib.gsl_vector_get(x_vec, i) for i in range(3)]
            out = residual(vals)
            for i, value in enumerate(out):
                lib.gsl_vector_set(f_vec, i, float(value))
            return 0
        except Exception as exc:  # pragma: no cover - defensive bridge
            callback_error.append(exc)
            return 1

    callback = func_type(_callback)
    gsl_fn = struct_type(callback, 3, None)
    x_vec = lib.gsl_vector_alloc(3)
    solver = None
    if not x_vec:
        return None
    try:
        for i, value in enumerate(guess):
            lib.gsl_vector_set(x_vec, i, float(value))
        solver = lib.gsl_multiroot_fsolver_alloc(solver_type, 3)
        if not solver:
            return None
        status = lib.gsl_multiroot_fsolver_set(solver, ctypes.byref(gsl_fn), x_vec)
        if status != 0:
            return None
        gsl_continue = -2
        for _ in range(max_iter):
            status = lib.gsl_multiroot_fsolver_iterate(solver)
            if callback_error:
                raise callback_error[0]
            if status != 0:
                return None
            f_vec = lib.gsl_multiroot_fsolver_f(solver)
            test_status = lib.gsl_multiroot_test_residual(f_vec, ctypes.c_double(epsabs))
            if test_status == 0:
                root_vec = lib.gsl_multiroot_fsolver_root(solver)
                root = [lib.gsl_vector_get(root_vec, i) for i in range(3)]
                final_res = [lib.gsl_vector_get(f_vec, i) for i in range(3)]
                return root, final_res
            if test_status != gsl_continue:
                return None
    finally:
        if solver:
            lib.gsl_multiroot_fsolver_free(solver)
        lib.gsl_vector_free(x_vec)
    return None


def _gsl_multiroot_hybrids(
    residual: Callable[[list[float]], list[float]],
    guess: list[float],
    *,
    epsabs: float,
    max_iter: int,
) -> tuple[list[float], list[float]] | None:
    """Mirror LAL's deterministic GSL ``hybrids`` residual-stop loop."""

    try:
        return _ported_gsl_multiroot_hybrids(
            residual,
            guess,
            epsabs=epsabs,
            max_iter=max_iter,
            cblas_candidates=_gsl_cblas_library_candidates(),
        )
    except GslCblasUnavailable:
        # Most Unix GSL builds link their companion CBLAS normally. Keep the
        # native solver as a fallback when that library cannot be found.
        return _native_gsl_multiroot_hybrids(
            residual,
            guess,
            epsabs=epsabs,
            max_iter=max_iter,
        )


def _ported_gsl_multiroot_available() -> bool:
    """Return whether the deterministic scaled-hybrids backend is available.

    The source-faithful controller uses a companion GSL CBLAS when loadable
    and its exact three-vector scalar BLAS subset otherwise, so it is available
    on every supported Torch device without switching root algorithms.
    """

    _load_gsl_cblas(_gsl_cblas_library_candidates())
    return True


def _gsl_multiroot_hybrids_ported(
    residual: Callable[[list[float]], list[float]],
    guess: list[float],
    *,
    epsabs: float,
    max_iter: int,
) -> tuple[list[float], list[float]] | None:
    """Run the deterministic hybrids mirror without native-GSL fallback."""

    return _ported_gsl_multiroot_hybrids(
        residual,
        guess,
        epsabs=epsabs,
        max_iter=max_iter,
        cblas_candidates=_gsl_cblas_library_candidates(),
    )


def _installed_gsl_deriv_central(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use installed GSL for the strict CPU scalar parity path."""

    if _GSL_DERIV_UNAVAILABLE:
        return None
    if not (
        x.device.type == "cpu"
        and x.dtype == torch.float64
        and x.numel() == 1
        and h.numel() == 1
        and not x.requires_grad
    ):
        return None

    use_installed_gsl = _env_on("PYCBC_SEOBNRV4PHM_INSTALLED_GSL_DERIV", False)
    if not use_installed_gsl:
        return None

    loaded = _load_gsl_deriv_central()
    if loaded is None:
        return None
    lib, func_type, struct_type = loaded
    device, dtype = x.device, x.dtype
    callback_error = []

    def _callback(x_eval: float, _params) -> float:
        try:
            x_t = torch.as_tensor(x_eval, device=device, dtype=dtype)
            return float(fn(x_t).detach())
        except Exception as exc:  # pragma: no cover - defensive bridge
            callback_error.append(exc)
            return float("nan")

    callback = func_type(_callback)
    gsl_fn = struct_type(callback, None)
    result = ctypes.c_double()
    abserr = ctypes.c_double()
    status = lib.gsl_deriv_central(
        ctypes.byref(gsl_fn),
        ctypes.c_double(float(x.detach())),
        ctypes.c_double(float(h.detach())),
        ctypes.byref(result),
        ctypes.byref(abserr),
    )
    if callback_error:
        raise callback_error[0]
    if status != 0 or not (math.isfinite(result.value) and math.isfinite(abserr.value)):
        return None
    return (
        torch.as_tensor(result.value, device=device, dtype=dtype),
        torch.as_tensor(abserr.value, device=device, dtype=dtype),
    )


def _gsl_deriv_central_with_error(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    h: torch.Tensor,
    *,
    allow_installed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch mirror of GSL's gsl_deriv_central 5-point derivative and error."""

    if allow_installed and not _GSL_DERIV_UNAVAILABLE:
        installed_gsl = _installed_gsl_deriv_central(fn, x, h)
        if installed_gsl is not None:
            return installed_gsl

    if (
        x.device.type == "cpu"
        and x.dtype == torch.float64
        and x.numel() == 1
        and h.numel() == 1
        and not x.requires_grad
    ):
        eps_f = 2.2204460492503131e-16
        x_f = float(x.detach())
        h_f = float(h.detach())

        def _eval_float(x_eval: float) -> float:
            return float(fn(torch.as_tensor(x_eval, device=x.device, dtype=x.dtype)).detach())

        def _central_float(step: float):
            fm1 = _eval_float(x_f - step)
            fp1 = _eval_float(x_f + step)
            fmh = _eval_float(x_f - step / 2.0)
            fph = _eval_float(x_f + step / 2.0)
            r3 = 0.5 * (fp1 - fm1)
            r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3
            e3 = (abs(fp1) + abs(fm1)) * eps_f
            e5 = 2.0 * (abs(fph) + abs(fmh)) * eps_f + e3
            dy = max(abs(r3 / step), abs(r5 / step)) * (abs(x_f) / step) * eps_f
            result = r5 / step
            trunc = abs((r5 - r3) / step)
            round_err = abs(e5 / step) + dy
            return result, round_err, trunc

        r0, round_err, trunc = _central_float(h_f)
        error = round_err + trunc
        if round_err < trunc and round_err > 0.0 and trunc > 0.0:
            h_opt = h_f * math.pow(round_err / (2.0 * trunc), 1.0 / 3.0)
            r_opt, round_opt, trunc_opt = _central_float(h_opt)
            error_opt = round_opt + trunc_opt
            if error_opt < error and abs(r_opt - r0) < 4.0 * error:
                r0 = r_opt
                error = error_opt
        return (
            torch.as_tensor(r0, device=x.device, dtype=x.dtype),
            torch.as_tensor(error, device=x.device, dtype=x.dtype),
        )

    eps = torch.as_tensor(torch.finfo(x.dtype).eps, device=x.device, dtype=x.dtype)

    def _central(step: torch.Tensor):
        fm1 = fn(x - step)
        fp1 = fn(x + step)
        fmh = fn(x - step / 2.0)
        fph = fn(x + step / 2.0)
        r3 = 0.5 * (fp1 - fm1)
        r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3
        e3 = (torch.abs(fp1) + torch.abs(fm1)) * eps
        e5 = 2.0 * (torch.abs(fph) + torch.abs(fmh)) * eps + e3
        dy = torch.maximum(torch.abs(r3 / step), torch.abs(r5 / step)) * (torch.abs(x) / step) * eps
        result = r5 / step
        trunc = torch.abs((r5 - r3) / step)
        round_err = torch.abs(e5 / step) + dy
        return result, round_err, trunc

    r0, round_err, trunc = _central(h)
    error = round_err + trunc
    valid = (round_err < trunc) & (round_err > 0.0) & (trunc > 0.0)
    if not torch.any(valid):
        return r0, error

    h_base = h + torch.zeros_like(round_err)
    if round_err.ndim == 0:
        h_opt = torch.as_tensor(
            float(h.detach().cpu()) * math.pow(float((round_err / (2.0 * trunc)).detach().cpu()), 1.0 / 3.0),
            dtype=h.dtype,
            device=h.device,
        )
    else:
        h_opt = h_base * torch.pow(round_err / (2.0 * trunc), 1.0 / 3.0)
    h_eval = torch.where(valid, h_opt, h_base)
    r_opt, round_opt, trunc_opt = _central(h_eval)
    error_opt = round_opt + trunc_opt
    accept = valid & (error_opt < error) & (torch.abs(r_opt - r0) < 4.0 * error)
    return torch.where(accept, r_opt, r0), torch.where(accept, error_opt, error)


def _gsl_deriv_central(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Torch mirror of GSL's gsl_deriv_central 5-point derivative."""

    return _gsl_deriv_central_with_error(fn, x, h)[0]


def _gsl_deriv_central_pure(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Deterministic derivative mirror which ignores the debug GSL switch."""

    return _gsl_deriv_central_with_error(
        fn,
        x,
        h,
        allow_installed=False,
    )[0]


def _rhs_derivative_options():
    """Derivative options for the precessing RHS.

    LAL's v4P default uses numerical Hamiltonian derivatives for x, p, and
    spins.  The torch-native path keeps the faster defaults unless the grouped
    parity switch below, or the individual derivative switches, are enabled.
    """
    lal_numerical = _env_on("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", False)
    return {
        "fd_dpvec": _env_on("PYCBC_SEOBNRV4PHM_FD_DP", lal_numerical),
        "use_hamiltonian_spin": _env_on("PYCBC_SEOBNRV4PHM_HAMILTONIAN_SPIN", True),
        "use_fd_spin": _env_on("PYCBC_SEOBNRV4PHM_FD_SPIN", lal_numerical),
        "use_tortoise_pdot": _env_on("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", True),
        "use_fd_x": _env_on("PYCBC_SEOBNRV4PHM_FD_X", lal_numerical),
        "use_cartesian_x_grad": _env_on("PYCBC_SEOBNRV4PHM_X_GRAD", True),
    }


@torch.jit.script
def _phi_hat(r_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit phi-hat, sin(theta), and r_mag for a Cartesian position (z-axis reference)."""
    r_mag = torch.clamp(_safe_norm(r_vec), min=1.0e-15)
    r_hat = r_vec / r_mag.unsqueeze(-1)
    zhat = torch.zeros_like(r_hat)
    zhat[..., 2] = 1.0
    cos_theta = torch.sum(r_hat * zhat, dim=-1)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=1.0e-16))
    phi_hat = torch.cross(zhat, r_hat, dim=-1)
    phi_hat = _unit_vector3(phi_hat)
    # Fallback when r nearly aligned with z
    aligned = sin_theta < 1.0e-8
    if aligned.any():
        xhat = torch.zeros_like(r_hat)
        xhat[..., 0] = 1.0
        phi_hat = torch.where(aligned.unsqueeze(-1), xhat, phi_hat)
        sin_theta = torch.where(aligned, torch.ones_like(sin_theta), sin_theta)
    return phi_hat, sin_theta, r_mag


def _augmented_spin_chi(L_vec: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor, params: EOBParams):
    """Augmented spin S_con used in the calibrated hcoeffs (LALSimIMRSpinPrecEOBv4P.c:1806-1834)."""
    total_mass = params.mass1 + params.mass2
    s1_m2 = _lal_spin_weight(S1, params.mass1, total_mass)
    s2_m2 = _lal_spin_weight(S2, params.mass2, total_mass)
    sigma_vec = s1_m2 + s2_m2
    sKerr_norm = _safe_norm(sigma_vec)
    chi_aug = torch.tensor(0.0, device=L_vec.device, dtype=L_vec.dtype)
    if float(sKerr_norm.detach()) > 1.0e-10:
        Lhat = L_vec / torch.clamp(_safe_norm(L_vec).unsqueeze(-1), min=1.0e-15)
        S1_dot_L = torch.sum(s1_m2 * Lhat)
        S2_dot_L = torch.sum(s2_m2 * Lhat)
        S1_perp = s1_m2 - S1_dot_L * Lhat
        S2_perp = s2_m2 - S2_dot_L * Lhat
        denom = torch.clamp(torch.tensor(1.0 - 2.0 * params.eta, device=L_vec.device, dtype=L_vec.dtype), min=1.0e-12)
        chi_aug = torch.sum(sigma_vec * Lhat) / denom
        chi_aug = chi_aug + (torch.sum(S1_perp * sigma_vec) + torch.sum(S2_perp * sigma_vec)) / (sKerr_norm * denom * 2.0)
    return chi_aug, sigma_vec, sKerr_norm


def _refresh_hcoeffs(
    params: EOBParams,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    S1_weighted: torch.Tensor | None = None,
    S2_weighted: torch.Tensor | None = None,
):
    """Refresh SpinEOBHCoeffs using the augmented spin mapping (torch-friendly)."""
    hcoeffs, chi_aug, sigma_vec = _instantaneous_hcoeffs(
        params, L_vec, S1, S2, S1_weighted=S1_weighted, S2_weighted=S2_weighted
    )
    scalar_hcoeffs = {}
    for key, value in hcoeffs.items():
        value_t = torch.as_tensor(value).detach()
        if value_t.numel() != 1:
            return chi_aug, sigma_vec
        scalar_hcoeffs[key] = float(value_t)
    params.hcoeffs = scalar_hcoeffs
    return chi_aug, sigma_vec


def _instantaneous_hcoeffs(
    params: EOBParams,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    S1_weighted: torch.Tensor | None = None,
    S2_weighted: torch.Tensor | None = None,
    combine_perpendicular_before_dot: bool = False,
):
    """Return LAL's per-state calibrated Hamiltonian coefficients."""

    device, dtype = L_vec.device, L_vec.dtype
    total_mass = params.mass1 + params.mass2
    s1_m2 = (
        _lal_spin_weight(S1, params.mass1, total_mass)
        if S1_weighted is None
        else S1_weighted.to(device=device, dtype=dtype)
    )
    s2_m2 = (
        _lal_spin_weight(S2, params.mass2, total_mass)
        if S2_weighted is None
        else S2_weighted.to(device=device, dtype=dtype)
    )
    sigma_vec = s1_m2 + s2_m2
    sigma_norm = _lal_scalar_sqrt(torch.clamp(_dot3(sigma_vec, sigma_vec), min=0.0))

    Lhat = L_vec / torch.clamp(_safe_norm(L_vec).unsqueeze(-1), min=1.0e-15)
    S1_dot_L = _dot3(s1_m2, Lhat)
    S2_dot_L = _dot3(s2_m2, Lhat)
    S1_perp = s1_m2 - S1_dot_L.unsqueeze(-1) * Lhat
    S2_perp = s2_m2 - S2_dot_L.unsqueeze(-1) * Lhat
    denom = torch.as_tensor(1.0 - 2.0 * params.eta, device=L_vec.device, dtype=L_vec.dtype)
    denom = torch.where(torch.abs(denom) < 1.0e-12, torch.sign(denom) * 1.0e-12, denom)
    chi_raw = _dot3(sigma_vec, Lhat) / denom
    if combine_perpendicular_before_dot:
        # XLALSimIMRSpinPrecEOBHamiltonian forms S_perp component-by-component
        # before taking its dot product. The shared numerical-derivative
        # coefficient refresh instead adds two separate dot products.
        perpendicular_projection = _dot3(S1_perp + S2_perp, sigma_vec)
    else:
        perpendicular_projection = _dot3(S1_perp, sigma_vec) + _dot3(S2_perp, sigma_vec)
    chi_raw = chi_raw + perpendicular_projection / sigma_norm / denom / 2.0
    chi_aug = torch.where(sigma_norm > 1.0e-6, chi_raw, torch.zeros_like(chi_raw))
    return compute_spin_aligned_hcoeffs(params.eta, sigma_norm, chi=chi_aug), chi_aug, sigma_vec


def _lal_numerical_derivative_tortoise_prelude(
    r: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    hcoeffs: dict | None = None,
    *,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Tortoise csi/dcsi block from XLALSpinPrecHcapNumericalDerivativePrec."""

    current_hcoeffs, _, sigma_vec = _instantaneous_hcoeffs(
        params,
        L_vec,
        S1,
        S2,
        S1_weighted=S1_weighted_override,
        S2_weighted=S2_weighted_override,
    )
    coeffs = current_hcoeffs if hcoeffs is None else hcoeffs
    h = {k: torch.as_tensor(v, device=r.device, dtype=r.dtype) for k, v in coeffs.items()}
    eta_t = torch.as_tensor(params.eta, device=r.device, dtype=r.dtype)

    r2 = r * r
    u = 1.0 / r
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    logu = _lal_scalar_log(u)
    a = _safe_norm(sigma_vec)
    a2 = a * a
    w2 = r2 + a2

    D_term = 1.0 + 6.0 * eta_t * u2 + 2.0 * (26.0 - 3.0 * eta_t) * eta_t * u3
    D = 1.0 + _lal_scalar_log(D_term)
    eobD_r = (u2 / (D * D)) * (12.0 * eta_t * u + 6.0 * (26.0 - 3.0 * eta_t) * eta_t * u2) / D_term

    m1PlusetaKK = -1.0 + eta_t * h["KK"]
    bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + (2.0 * u) / m1PlusetaKK + a2 * u2
    log_arg = (
        1.0
        + h["k1"] * u
        + h["k2"] * u2
        + h["k3"] * u3
        + h["k4"] * u4
        + h["k5"] * u5
        + h["k5l"] * u5 * logu
    )
    logTerms = 1.0 + eta_t * h["k0"] + eta_t * _lal_scalar_log(log_arg)
    deltaU = bulk * logTerms
    deltaT = r2 * deltaU
    dlogarg_du = h["k1"] + u * (
        2.0 * h["k2"]
        + u * (3.0 * h["k3"] + u * (4.0 * h["k4"] + 5.0 * (h["k5"] + h["k5l"] * logu) * u))
    )
    deltaU_u = 2.0 * (1.0 / m1PlusetaKK + a2 * u) * logTerms + bulk * (eta_t * dlogarg_du) / log_arg
    deltaU_r = -u2 * deltaU_u
    deltaR = deltaT * D
    csi = _lal_scalar_sqrt(deltaT * deltaR) / w2 if getattr(params, "tortoise", 2) else torch.ones_like(r)
    dcsi = (
        csi * (2.0 / r + deltaU_r / deltaU)
        + (csi * csi * csi)
        / (2.0 * r2 * r2 * deltaU * deltaU)
        * (r * (-4.0 * w2) / D - eobD_r * (w2 * w2))
    )
    return {"csi": csi, "dcsi": dcsi}


def _lal_hamiltonian_delta_t(
    hcoeffs: dict,
    r: torch.Tensor,
    eta: float,
    a: torch.Tensor,
) -> torch.Tensor:
    """Standalone ``DeltaT`` used by LAL's final IC tortoise transform."""

    h = {k: torch.as_tensor(v, device=r.device, dtype=r.dtype) for k, v in hcoeffs.items()}
    eta_t = torch.as_tensor(eta, device=r.device, dtype=r.dtype)
    u = 1.0 / r
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    a2 = a * a
    m1_plus_eta_KK = -1.0 + eta_t * h["KK"]
    bulk = (
        1.0 / (m1_plus_eta_KK * m1_plus_eta_KK)
        + (2.0 * u) / m1_plus_eta_KK
        + a2 * u2
    )
    log_arg = (
        1.0
        + h["k1"] * u
        + h["k2"] * u2
        + h["k3"] * u3
        + h["k4"] * u4
        + h["k5"] * u5
        + h["k5l"] * u5 * _lal_scalar_log(u)
    )
    log_terms = 1.0 + eta_t * h["k0"] + eta_t * _lal_scalar_log(torch.abs(log_arg))
    delta_u = torch.abs(bulk * log_terms)
    return r * r * delta_u


def _lal_hamiltonian_delta_r(
    hcoeffs: dict,
    r: torch.Tensor,
    eta: float,
    a: torch.Tensor,
) -> torch.Tensor:
    """Standalone ``DeltaR`` used by LAL's final IC tortoise transform."""

    eta_t = torch.as_tensor(eta, device=r.device, dtype=r.dtype)
    # Keep the standalone routine's order: it uses 1/(r*r), not (1/r)^2.
    u2 = 1.0 / (r * r)
    u3 = u2 / r
    D = 1.0 + _lal_scalar_log(
        1.0 + 6.0 * eta_t * u2 + 2.0 * (26.0 - 3.0 * eta_t) * eta_t * u3
    )
    return _lal_hamiltonian_delta_t(hcoeffs, r, eta, a) * D


def _eob_potentials(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    p_vec: torch.Tensor | None = None,
    r_vec: torch.Tensor | None = None,
    compute_grad_p: bool | None = None,
    compute_grad_spin: bool = False,
    p_is_tortoise: bool = True,
    fd_dpvec: bool = True,
    fd_pphi: bool = True,
    compute_grad_x: bool = False,
    compute_base_grad: bool = True,
    hcoeffs_override: dict | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """COMPLETE port of deltaT/deltaR/D/qq/csi/H + Hs/Hss tilt terms from
    LALSimIMRSpinEOBHamiltonianPrec.c:323-406, 458-575, 600-671.

    Includes non-equatorial xi^2 != 1, p_theta terms, and the precessing
    spin-orbit/spin-spin pieces (Hs/Hss). Tortoise flag is fixed to 2.
    """

    device, dtype = r.device, r.dtype
    if compute_grad_spin:
        S1 = S1.detach().clone().requires_grad_(True)
        S2 = S2.detach().clone().requires_grad_(True)
    eta_t = torch.as_tensor(params.eta, device=device, dtype=dtype)
    inv_eta = torch.as_tensor(1.0 / params.eta, device=device, dtype=dtype)
    m1 = torch.as_tensor(params.mass1, device=device, dtype=dtype)
    m2 = torch.as_tensor(params.mass2, device=device, dtype=dtype)
    M = m1 + m2
    mass1_norm = m1 / M
    mass2_norm = m2 / M
    norm_total_mass = mass1_norm + mass2_norm
    norm_total_mass2 = norm_total_mass * norm_total_mass

    # Spin combinations (Sigma_Kerr / Sigma_Star) using the same normalized
    # mass and evolved-spin order as XLALSimIMRSpinEOBCalculateSigmaKerr/Star.
    def _effective_weighted_spin(spin, mass, override):
        weighted = _lal_spin_weight(spin, mass, params.M)
        if override is None:
            return weighted
        weighted_override = override.to(device=device, dtype=dtype)
        if compute_grad_spin:
            # The Cartesian LAL state carries S_i / M^2, while this function's
            # spin arguments are chi_i.  Keep the state value exactly, but
            # retain d(S_i / M^2)/d chi_i for the analytic spin derivative.
            return weighted_override + (weighted - weighted.detach())
        return weighted_override

    s1_m2_evolved = _effective_weighted_spin(
        S1, params.mass1, S1_weighted_override
    )
    s2_m2_evolved = _effective_weighted_spin(
        S2, params.mass2, S2_weighted_override
    )
    sigmaKerr_vec = (s1_m2_evolved + s2_m2_evolved) / norm_total_mass2
    sigmaStar_vec = (
        (mass2_norm / mass1_norm) * s1_m2_evolved
        + (mass1_norm / mass2_norm) * s2_m2_evolved
    ) / norm_total_mass2

    a2 = torch.clamp(_dot3(sigmaKerr_vec, sigmaKerr_vec), min=1e-16)
    a = _lal_scalar_sqrt(a2)

    # Unit vectors: Lhat (orbital), e3 (Kerr spin), n (radial), xi = e3 x n, v = n x xi
    L_mag = _safe_norm(L_vec)
    Lhat = L_vec / torch.clamp(L_mag.unsqueeze(-1), min=1e-15)
    r2_source = _dot3(r_vec, r_vec) if r_vec is not None else r * r

    a_mag = _safe_norm(sigmaKerr_vec)
    inv_a_mag = 1.0 / torch.clamp(a_mag, min=1.0e-15)
    e3_hat = sigmaKerr_vec * inv_a_mag.unsqueeze(-1)
    fallback_e3 = torch.full_like(e3_hat, 1.0 / math.sqrt(3.0))
    e3_hat = torch.where((a_mag < 1.0e-12).unsqueeze(-1), fallback_e3, e3_hat)

    # Build an in-plane orthonormal basis (e1, e2) to define n from phase phi
    zhat = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    xhat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    # If a snapshot r_vec is provided, rebuild n_hat from it to mirror LAL's geometry
    use_r_vec = r_vec is not None
    if use_r_vec:
        inv_r_for_n = 1.0 / torch.clamp(_safe_norm(r_vec), min=1.0e-15)
        n_hat = r_vec * inv_r_for_n.unsqueeze(-1)
        # enforce orthonormal basis using Lhat and e3_hat
        e1 = _cross3(Lhat, e3_hat)
    else:
        e1 = _cross3(Lhat, e3_hat)
    # Do not use _safe_norm here: its non-zero floor masks the exactly
    # parallel L/e3 case and leaves e1 as the zero vector.  Reduced aligned
    # states then lose their tangential momentum when p is reconstructed.
    e1_norm = _lal_scalar_sqrt(torch.clamp(_dot3(e1, e1), min=0.0))
    ref = torch.where((torch.abs(Lhat[..., 2]) < 0.9).unsqueeze(-1), zhat, xhat)
    e1_fallback = _cross3(Lhat, ref)
    e1 = torch.where((e1_norm < 1.0e-14).unsqueeze(-1), e1_fallback, e1)
    e1 = e1 / torch.clamp(_safe_norm(e1).unsqueeze(-1), min=1.0e-15)
    e2 = _cross3(Lhat, e1)
    if use_r_vec:
        n_hat = n_hat  # already set
    else:
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        n_hat = cosphi.unsqueeze(-1) * e1 + sinphi.unsqueeze(-1) * e2
        n_hat = n_hat / torch.clamp(_safe_norm(n_hat).unsqueeze(-1), min=1.0e-15)
    lambda_hat = _cross3(Lhat, n_hat)
    lambda_hat = lambda_hat / torch.clamp(_safe_norm(lambda_hat).unsqueeze(-1), min=1.0e-15)

    costheta = _dot3(e3_hat, n_hat)
    xi_vec = _cross3(e3_hat, n_hat)
    # LAL forms xi2 from the scalar product, not from |e3 x n|^2; the
    # one-ulp difference matters inside GSL finite differences.
    xi2 = 1.0 - costheta * costheta
    mask_aligned = (1.0 - torch.abs(costheta)) <= 1.0e-8
    if mask_aligned.any():
        angle = torch.as_tensor(1.8e-3, device=device, dtype=dtype)
        kcrossv = _cross3(lambda_hat, e3_hat)
        kdotv = _dot3(lambda_hat, e3_hat)
        e3_rot = (
            e3_hat * torch.cos(angle)
            + kcrossv * torch.sin(angle)
            + lambda_hat * kdotv.unsqueeze(-1) * (1.0 - torch.cos(angle))
        )
        e3_hat = torch.where(mask_aligned.unsqueeze(-1), e3_rot, e3_hat)
        xi_vec = _cross3(e3_hat, n_hat)
        costheta = _dot3(e3_hat, n_hat)
        xi2 = 1.0 - costheta * costheta
    _xi2_safe = torch.clamp(xi2, min=1.0e-12)
    v_vec = _cross3(n_hat, xi_vec)

    # Scalar helpers
    u = torch.clamp(1.0 / torch.clamp(r, min=1e-9), min=1e-9)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u

    # Spin-aligned coefficients (torch tensors for broadcasting). LAL refreshes
    # these inside the Hamiltonian whenever updateHCoeffs is set. The full RHS
    # also has a separate tortoise-matrix prelude that uses the entry coeffs
    # before the later per-state refresh, so callers can override this block.
    if hcoeffs_override is None:
        hcoeffs, _, _ = _instantaneous_hcoeffs(
            params,
            L_vec,
            S1,
            S2,
            S1_weighted=s1_m2_evolved,
            S2_weighted=s2_m2_evolved,
            combine_perpendicular_before_dot=True,
        )
    else:
        hcoeffs = hcoeffs_override
    h = {k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in hcoeffs.items()}
    d1 = h.get("d1", torch.tensor(0.0, device=device, dtype=dtype))
    d1v2 = h.get("d1v2", torch.tensor(0.0, device=device, dtype=dtype))
    dheffSS = h.get("dheffSS", torch.tensor(0.0, device=device, dtype=dtype))
    dheffSSv2 = h.get("dheffSSv2", torch.tensor(0.0, device=device, dtype=dtype))

    # deltaT (Eq. 5.73 + 4PN log terms) and derivatives
    denom_KK = -1.0 + eta_t * h["KK"]
    invm1PlusEtaKK = 1.0 / torch.where(torch.abs(denom_KK) < 1.0e-14, torch.sign(denom_KK) * 1.0e-14, denom_KK)
    logu = _lal_scalar_log(u)
    logarg = h["k1"] * u + h["k2"] * u2 + h["k3"] * u3 + h["k4"] * u4 + h["k5"] * u5 + h["k5l"] * u5 * logu
    logTerms = 1.0 + eta_t * h["k0"] + eta_t * _lal_scalar_log1p(torch.abs(1.0 + logarg) - 1.0)
    bulk = invm1PlusEtaKK * (invm1PlusEtaKK + 2.0 * u) + a2 * u2
    deltaU = torch.abs(bulk * logTerms)
    deltaT = r2_source * deltaU
    dlogarg_du = h["k1"] + u * (2.0 * h["k2"] + u * (3.0 * h["k3"] + u * (4.0 * h["k4"] + 5.0 * (h["k5"] + h["k5l"] * logu) * u)))
    deltaU_u = 2.0 * (invm1PlusEtaKK + a2 * u) * logTerms + bulk * (eta_t * dlogarg_du) / (1.0 + logarg)
    deltaT_r = 2.0 * r * deltaU - deltaU_u

    # deltaR and auxiliary potentials
    D_term = torch.clamp(1.0 + 6.0 * eta_t * u2 + 2.0 * (26.0 - 3.0 * eta_t) * eta_t * u3, min=1.0e-15)
    D = 1.0 + _lal_scalar_log(D_term)
    dD_du = (12.0 * eta_t * u + 6.0 * (26.0 - 3.0 * eta_t) * eta_t * u2) / D_term
    dD_dr = -u2 * dD_du
    _deltaR = deltaT * D
    _deltaR_r = deltaT_r * D + deltaT * dD_dr
    w2 = r2_source + a2
    _rho2 = r2_source + a2 * costheta * costheta
    Lambda = torch.abs(w2 * w2 - a2 * deltaT * xi2)
    invLambda = 1.0 / torch.clamp(Lambda, min=1.0e-15)
    dLambda_dr = 4.0 * r * w2 - a2 * deltaT_r * xi2
    _dinvLambda_dr = -dLambda_dr * (invLambda * invLambda)

    # Canonical momenta components (pf = L·e3; ptheta from pvr)
    compute_grad_p = (p_vec is not None and r_vec is not None) if compute_grad_p is None else compute_grad_p
    needs_r_grad = compute_base_grad or compute_grad_x
    if needs_r_grad and compute_grad_x and getattr(r, "requires_grad", False):
        r_var = r
        r_var.requires_grad_(True)
    elif needs_r_grad:
        r_var = r.detach().clone().requires_grad_(True)
    else:
        r_var = r.detach().clone()

    # Potentials using r_var (LAL v4P uses tortoise flag=1)
    u_b = torch.clamp(1.0 / torch.clamp(r_var, min=1e-9), min=1e-9)
    u2_b = u_b * u_b
    u3_b = u2_b * u_b
    u4_b = u2_b * u2_b
    u5_b = u4_b * u_b
    # The Hamiltonian source uses log2(u) multiplied by ln(2), while the
    # standalone RHS tortoise prelude above uses log(u). Keep the distinction:
    # tiny roundoff changes here feed directly into RKF45 parity.
    invlog_2e = torch.as_tensor(
        0.69314718055994530941723212145817656807550013436026,
        device=device,
        dtype=dtype,
    )
    logu_b = _lal_scalar_log2(u_b) * invlog_2e
    logarg_b = h["k1"] * u_b + h["k2"] * u2_b + h["k3"] * u3_b + h["k4"] * u4_b + h["k5"] * u5_b + h["k5l"] * u5_b * logu_b
    logTerms_b = 1.0 + eta_t * h["k0"] + eta_t * _lal_scalar_log1p(torch.abs(1.0 + logarg_b) - 1.0)
    bulk_b = invm1PlusEtaKK * (invm1PlusEtaKK + 2.0 * u_b) + a2 * u2_b
    deltaU_b = torch.abs(bulk_b * logTerms_b)
    r2_b = r2_source if (r_vec is not None and not compute_grad_x) else r_var * r_var
    deltaT_b = r2_b * deltaU_b
    deltaU_u_b = 2.0 * (invm1PlusEtaKK + a2 * u_b) * logTerms_b + bulk_b * (eta_t * (h["k1"] + u_b * (2.0 * h["k2"] + u_b * (3.0 * h["k3"] + u_b * (4.0 * h["k4"] + 5.0 * (h["k5"] + h["k5l"] * logu_b) * u_b))))) / (1.0 + logarg_b)
    deltaT_r_b = 2.0 * r_var * deltaU_b - deltaU_u_b
    D_b_arg = 6.0 * eta_t * u2_b + 2.0 * (26.0 - 3.0 * eta_t) * eta_t * u3_b
    D_b_term = 1.0 + D_b_arg
    D_b = 1.0 + _lal_scalar_log1p(D_b_arg)
    deltaR_b = deltaT_b * D_b
    w2_b = r2_b + a2
    rho2_b = r2_b + a2 * costheta * costheta
    Lambda_b = torch.abs(w2_b * w2_b - a2 * deltaT_b * xi2)
    invrho2xi2Lambda_b = 1.0 / (rho2_b * xi2 * Lambda_b)
    invrho2 = xi2 * (Lambda_b * invrho2xi2Lambda_b)
    invxi2 = rho2_b * (Lambda_b * invrho2xi2Lambda_b)
    invLambda_b = xi2 * rho2_b * invrho2xi2Lambda_b
    csi_b = _lal_scalar_sqrt(torch.abs(deltaT_b * deltaR_b)) / w2_b
    # dcsi diagnostic (Eq. A5, LALSimIMRSpinEOBHamiltonianPrec.c:1307-1316)
    deltaU_r_b = -u2_b * deltaU_u_b
    eobD_r_b = (u2_b / (D_b * D_b)) * (12.0 * eta_t * u_b + 6.0 * (26.0 - 3.0 * eta_t) * eta_t * u2_b) / torch.clamp(D_b_term, min=1.0e-15)
    dcsi_b = (
        csi_b * (2.0 / torch.clamp(r_var, min=1.0e-15) + deltaU_r_b / torch.clamp(deltaU_b, min=1.0e-20))
        + (csi_b ** 3)
        / (2.0 * torch.clamp(r_var * r_var * r_var * r_var, min=1.0e-20) * torch.clamp(deltaU_b * deltaU_b, min=1.0e-20))
        * (r_var * (-4.0 * w2_b) / torch.clamp(D_b, min=1.0e-20) - eobD_r_b * (w2_b * w2_b))
    )
    tortoise = getattr(params, "tortoise", 2)
    # Split the tortoise factors the same way as LALSimIMRSpinEOBHamiltonianPrec.c:360-384
    if (p_vec is not None) and (r_vec is not None) and (not p_is_tortoise) and tortoise != 2:
        # Inputs are already non-tortoise with tortoise disabled. LAL's IC
        # solver uses this convention before converting to tortoise momenta.
        csi1_b = torch.ones_like(csi_b)
        csi2_b = torch.ones_like(csi_b)
        csi_out = torch.ones_like(csi_b)
    elif tortoise == 1:
        csi1_b = csi_b
        csi2_b = torch.ones_like(csi_b)
        csi_out = csi_b
    elif tortoise == 2:
        csi1_b = torch.ones_like(csi_b)
        csi2_b = csi_b
        csi_out = csi_b
    else:
        csi1_b = torch.ones_like(csi_b)
        csi2_b = torch.ones_like(csi_b)
        csi_out = csi_b
    if p_vec is None:
        pr_var = pr.detach().clone().requires_grad_(True)
        p_vec_base = pr_var.unsqueeze(-1) * n_hat - _cross3(n_hat, L_vec) / torch.clamp(r_var.unsqueeze(-1), min=1e-12)
        pr_from_p = _dot3(p_vec_base, n_hat)
        prT = pr_from_p * csi2_b
        scale = (1.0 - 1.0 / torch.clamp(csi1_b, min=1.0e-15)).unsqueeze(-1)
        p_vec_use = p_vec_base - n_hat * prT.unsqueeze(-1) * scale
    else:
        p_vec_base = p_vec.detach().clone()
        if compute_grad_p:
            p_vec_base.requires_grad_(True)
        pr_proj = _dot3(p_vec_base, n_hat)  # P_r* in LAL snapshot
        csi_fac = torch.clamp(csi_out, min=1.0e-15)
        if p_is_tortoise:
            prT = pr_proj * csi2_b
            radial_scale = 1.0 - 1.0 / torch.clamp(csi1_b, min=1.0e-15)
            p_vec_use = p_vec_base - n_hat * prT.unsqueeze(-1) * radial_scale.unsqueeze(-1)
            pr_canon = _dot3(p_vec_use, n_hat)
        else:
            p_vec_use = p_vec_base
            pr_canon = pr_proj
            prT = pr_canon * csi_fac
        if compute_grad_p or compute_grad_x:
            pr_var = pr_canon
            if compute_grad_p:
                pr_var.requires_grad_(True)
        elif compute_base_grad:
            pr_var = pr_canon.detach().clone().requires_grad_(True)
        else:
            pr_var = pr_canon.detach().clone()

    tmpP = p_vec_use
    pn = _dot3(tmpP, n_hat)
    pvr = _dot3(tmpP, v_vec) * r_var
    ptheta2 = pvr * pvr * invxi2
    qq = 2.0 * eta_t * (4.0 - 3.0 * eta_t)
    b3 = h.get("b3", torch.tensor(0.0, device=device, dtype=dtype))
    bb3 = h.get("bb3", torch.tensor(0.0, device=device, dtype=dtype))
    ww = 2.0 * a * r_var + b3 * eta_t * a2 * a * u_b + bb3 * eta_t * a * u_b

    pf0 = _dot3(tmpP, xi_vec) * r_var
    if compute_grad_p or compute_grad_x:
        pf_var = pf0
    elif compute_base_grad:
        pf_var = pf0.detach().clone().requires_grad_(True)
    else:
        pf_var = pf0.detach().clone()
    pf_use = pf_var
    prT2 = prT * prT
    pf2 = pf_use * pf_use
    pn2_raw = pn * pn
    base = (
        1.0
        + (prT2 * prT2) * qq * u2_b
        + ptheta2 * invrho2
        + pf2 * rho2_b * invLambda_b * invxi2
        + pn2_raw * deltaR_b * invrho2
    )
    Hns = _lal_scalar_sqrt(torch.clamp(base * (rho2_b * deltaT_b) * invLambda_b, min=1.0e-16)) + pf_use * ww * invLambda_b

    Q = 1.0 + pvr * pvr * invrho2 * invxi2 + pf2 * rho2_b * invLambda_b * invxi2 + pn2_raw * deltaR_b * invrho2
    pn2 = pn2_raw * deltaR_b * invrho2
    pp = Q - 1.0

    deltaSigmaStar = eta_t.unsqueeze(-1) * (
        (-8.0 - 3.0 * r_var * (12.0 * pn2 - pp)).unsqueeze(-1) * sigmaKerr_vec
        + (14.0 + (-30.0 * pn2 + 4.0 * pp) * r_var).unsqueeze(-1) * sigmaStar_vec
    ) * (1.0 / 12.0) * u_b.unsqueeze(-1)

    sMultiplier1 = (
        -706.0
        + (206.0 * pp - 282.0 * pn2 + (-96.0 * pn2 * pp + 23.0 * pp * pp) * r_var) * r_var
        + (54.0 + (-120.0 * pp + 324.0 * pn2 + (-360.0 * pn2 * pn2 + 126.0 * pn2 * pp + 3.0 * pp * pp) * r_var) * r_var) * eta_t
    )
    sMultiplier1 = sMultiplier1 * eta_t * u2_b * (-1.0 / 72.0)

    sMultiplier2 = (
        -56.0 / 9.0 * u2_b
        + (-2.0 / 3.0 * pn2 * u2_b - 109.0 / 36.0 * pp * u2_b + (pn2 * pp * u2_b / 4.0 - 5.0 / 16.0 * pp * pp * u2_b) * r_var) * r_var
        + (-7.0 / 3.0 * u2_b + (-49.0 / 8.0 * pn2 * u2_b + 17.0 / 12.0 * pp * u2_b + (45.0 / 8.0 * pn2 * pn2 * u2_b - 13.0 / 8.0 * pn2 * pp * u2_b) * r_var) * r_var) * eta_t
    )
    sMultiplier2 = sMultiplier2 * eta_t

    d1_vec = d1.unsqueeze(-1) if getattr(d1, "ndim", 0) > 0 else d1
    d1v2_vec = d1v2.unsqueeze(-1) if getattr(d1v2, "ndim", 0) > 0 else d1v2
    deltaSigmaStar = deltaSigmaStar + (
        sMultiplier1.unsqueeze(-1) * sigmaStar_vec + sMultiplier2.unsqueeze(-1) * sigmaKerr_vec
    )
    deltaSigmaStar = deltaSigmaStar + d1_vec * eta_t.unsqueeze(-1) * sigmaStar_vec * u3_b.unsqueeze(-1)
    deltaSigmaStar = deltaSigmaStar + d1v2_vec * eta_t.unsqueeze(-1) * sigmaKerr_vec * u3_b.unsqueeze(-1)

    s_vec = sigmaStar_vec + deltaSigmaStar
    sx, sy, sz = s_vec[..., 0], s_vec[..., 1], s_vec[..., 2]
    sxi = _dot3(s_vec, xi_vec)
    sv = _dot3(s_vec, v_vec)
    sn = _dot3(s_vec, n_hat)
    s3 = _dot3(s_vec, e3_hat)

    sqrtdeltaT = _lal_scalar_sqrt(torch.clamp(deltaT_b, min=1.0e-16))
    sqrtdeltaR = _lal_scalar_sqrt(torch.clamp(deltaR_b, min=1.0e-16))
    invdeltaTsqrtdeltaTsqrtdeltaR = 1.0 / torch.clamp(sqrtdeltaT * deltaT_b * sqrtdeltaR, min=1.0e-16)
    invsqrtdeltaT = deltaT_b * (sqrtdeltaR * invdeltaTsqrtdeltaTsqrtdeltaR)
    invsqrtdeltaR = deltaT_b * sqrtdeltaT * invdeltaTsqrtdeltaTsqrtdeltaR
    invdeltaT = sqrtdeltaT * (sqrtdeltaR * invdeltaTsqrtdeltaTsqrtdeltaR)

    w = ww * invLambda_b
    expnu = _lal_scalar_sqrt(torch.clamp(deltaT_b * rho2_b * invLambda_b, min=1.0e-16))
    expMU = _lal_scalar_sqrt(torch.clamp(rho2_b, min=1.0e-16))
    invexpnuexpMU = 1.0 / torch.clamp(expnu * expMU, min=1.0e-16)
    invexpnu = expMU * invexpnuexpMU
    invexpMU = expnu * invexpnuexpMU

    Lambda_r = 4.0 * r_var * w2_b - a2 * deltaT_r_b * xi2
    ww_r = 2.0 * a - (a2 * a * b3 * eta_t) * u2_b - bb3 * eta_t * a * u2_b
    BR = (-deltaT_b * invsqrtdeltaR + deltaT_r_b * 0.5) * invsqrtdeltaT
    wr = (-Lambda_r * ww + Lambda_b * ww_r) * (invLambda_b * invLambda_b)
    nur = (r_var * invrho2 + (w2_b * (-4.0 * r_var * deltaT_b + w2_b * deltaT_r_b)) * 0.5 * invdeltaT * invLambda_b)
    mur = (r_var * invrho2 - invsqrtdeltaR)
    wcos = -2.0 * (a2 * costheta) * deltaT_b * ww * (invLambda_b * invLambda_b)
    nucos = (a2 * costheta) * w2_b * (w2_b - deltaT_b) * (invrho2 * invLambda_b)
    mucos = (a2 * costheta) * invrho2

    sqrtQ = _lal_scalar_sqrt(torch.clamp(Q, min=1.0e-16))
    inv2B1psqrtQsqrtQ = 1.0 / torch.clamp(2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ, min=1.0e-16)
    invxi2_loc = invxi2

    Hwr = (
        (invexpMU * invexpMU * invexpMU * invexpnu)
        * sqrtdeltaR
        * (
            (expMU * expMU) * (expnu * expnu) * (pf_use * pf_use) * sv
            - sqrtdeltaT * (expMU * expnu) * pf_use * pvr * sxi
            + sqrtdeltaT
            * sqrtdeltaT
            * xi2
            * ((expMU * expMU) * (sqrtQ + Q) * sv + pn * pvr * sn * sqrtdeltaR - pn * pn * sv * deltaR_b)
        )
    )
    Hwr = Hwr * inv2B1psqrtQsqrtQ * invxi2_loc

    Hwcos = (
        (invexpMU * invexpMU * invexpMU * invexpnu)
        * (
            sn * (-(expMU * expMU) * (expnu * expnu) * (pf_use * pf_use) + sqrtdeltaT * sqrtdeltaT * (pvr * pvr - (expMU * expMU) * (sqrtQ + Q) * xi2))
            - sqrtdeltaT * pn * (sqrtdeltaT * pvr * sv - (expMU * expnu) * pf_use * sxi) * sqrtdeltaR
        )
    )
    Hwcos = Hwcos * inv2B1psqrtQsqrtQ

    HSOL = ((expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) * pf_use * s3) / (deltaT_b * sqrtQ) * invxi2_loc

    HSONL = (
        (expnu * (invexpMU * invexpMU))
        * (
            -(sqrtdeltaT * expMU * expnu * nucos * pf_use * (1.0 + 2.0 * sqrtQ) * sn * xi2)
            + (
                -(BR * (expMU * expnu) * pf_use * (1.0 + sqrtQ) * sv)
                + sqrtdeltaT
                * ((expMU * expnu) * nur * pf_use * (1.0 + 2.0 * sqrtQ) * sv + sqrtdeltaT * mur * pvr * sxi + sqrtdeltaT * sxi * (-(mucos * pn * xi2) + sqrtQ * (mur * pvr - nur * pvr + (-mucos + nucos) * pn * xi2)))
            )
            * sqrtdeltaR
        )
    )
    HSONL = HSONL * invxi2_loc / (deltaT_b * (sqrtQ + Q))

    Hs = w * s3 + Hwr * wr + Hwcos * wcos + HSOL + HSONL
    Hss = -0.5 * u3_b * (sx * sx + sy * sy + sz * sz - 3.0 * sn * sn)

    H_eff = Hns + Hs + Hss
    H_eff = H_eff + dheffSS * eta_t * _dot3(sigmaKerr_vec, sigmaStar_vec) * u4_b
    s1s2_square_sum = (
        s1_m2_evolved[..., 0] * s1_m2_evolved[..., 0]
        + s1_m2_evolved[..., 1] * s1_m2_evolved[..., 1]
        + s1_m2_evolved[..., 2] * s1_m2_evolved[..., 2]
        + s2_m2_evolved[..., 0] * s2_m2_evolved[..., 0]
        + s2_m2_evolved[..., 1] * s2_m2_evolved[..., 1]
        + s2_m2_evolved[..., 2] * s2_m2_evolved[..., 2]
    )
    H_eff = H_eff + dheffSSv2 * eta_t * u4_b * (
        s1s2_square_sum
    )

    # LAL uses fabs(H) in the real Hamiltonian map. The effective Hamiltonian
    # can cross negative values in tilted-spin coordinates; preserving the
    # absolute map keeps the radial force physical instead of falling through
    # to the Newtonian guard below.
    Hreal_arg = 1.0 + 2.0 * eta_t * (torch.abs(H_eff) - 1.0)
    needs_hreal_grad = compute_base_grad or compute_grad_p or compute_grad_x or compute_grad_spin
    Hreal = _lal_scalar_sqrt(torch.clamp(Hreal_arg, min=1.0e-16), detach=not needs_hreal_grad)
    r_newton = torch.clamp(torch.abs(r), min=1.0e-12)
    Hreal_fallback = _lal_scalar_sqrt(torch.clamp(1.0 + (pf0 * pf0) / (r_newton * r_newton), min=1.0e-12))
    Hreal = torch.where(Hreal <= 1.0e-8, Hreal_fallback, Hreal)

    dH_dpr = dH_dpf = dH_dr = None
    dH_dpvec = dH_dx = dH_dS1 = dH_dS2 = None
    grad_outputs = torch.ones_like(Hreal)
    grad_inputs = []
    if compute_base_grad:
        grad_inputs.extend([pr_var, pf_var, r_var])
    if compute_grad_p:
        grad_inputs.append(p_vec_base)
    if compute_grad_x and r_vec is not None:
        grad_inputs.append(r_vec)
    if compute_grad_spin:
        grad_inputs.extend([S1, S2])
    if grad_inputs:
        grads = torch.autograd.grad(
            Hreal,
            tuple(grad_inputs),
            grad_outputs=grad_outputs,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        grad_idx = 0
        if compute_base_grad:
            dH_dpr, dH_dpf, dH_dr = grads[:3]
            grad_idx = 3
        if compute_grad_p:
            dH_dpvec = grads[grad_idx]
            grad_idx += 1
        if compute_grad_x and r_vec is not None:
            dH_dx = grads[grad_idx]
            grad_idx += 1
        if compute_grad_spin:
            dH_dS1 = grads[grad_idx]
            dH_dS2 = grads[grad_idx + 1]

    # Finite-difference dH/dpphi in the polar basis (LAL GSL mirror) when snapshot is provided
    dH_dpf_fd = None
    if fd_pphi and (p_vec is not None) and (r_vec is not None):
        phi_hat, sin_theta, r_mag = _phi_hat(r_vec)
        step_pphi = torch.as_tensor(1.0e-4, device=device, dtype=dtype)
        pphi0 = torch.clamp(_safe_norm(L_vec), min=1.0e-15)
        hcoeffs_saved = params.hcoeffs

        def _ham_at_pphi(pphi_eval: torch.Tensor) -> torch.Tensor:
            scale = (pphi_eval - pphi0) / torch.clamp(r_mag * sin_theta, min=1.0e-12)
            delta_p = phi_hat * scale.unsqueeze(-1)
            L_eval = _cross3(r_vec, p_vec + delta_p)
            _refresh_hcoeffs(
                params,
                L_eval,
                S1,
                S2,
                S1_weighted=S1_weighted_override,
                S2_weighted=S2_weighted_override,
            )
            return _eob_potentials(
                r,
                pr,
                phi,
                L_eval,
                S1,
                S2,
                params,
                p_vec=p_vec + delta_p,
                r_vec=r_vec,
                compute_grad_p=False,
                p_is_tortoise=p_is_tortoise,
                fd_dpvec=False,
                fd_pphi=False,
                compute_base_grad=False,
                S1_weighted_override=S1_weighted_override,
                S2_weighted_override=S2_weighted_override,
            )["H"] / eta_t

        dH_dpf_fd = _gsl_deriv_central(_ham_at_pphi, pphi0, step_pphi)
        params.hcoeffs = hcoeffs_saved
        # Keep FD result available for debugging parity or override of dphi/dt.

    # Finite-difference dH/dP (tortoise Cartesian) to mirror GSL path when snapshot p_vec is given
    if fd_dpvec and (p_vec is not None) and (r_vec is not None):
        step_base = torch.as_tensor(_LAL_V4P_NUMERICAL_DERIVATIVE_STEP, device=device, dtype=dtype)
        hcoeffs_saved = params.hcoeffs
        # Match LAL v4P's GSL STEP_SIZE for Cartesian momentum axes.
        def _fd_step(param_idx: int) -> torch.Tensor:
            if 6 <= param_idx < 9:
                return step_base * (m1 * m1)
            if param_idx >= 9:
                return step_base * (m2 * m2)
            return step_base
        dH_dp = []
        for axis in range(3):
            # Map axis -> LAL parameter index (3..5 are momenta); keep mass-scaled spin steps available
            param_idx = axis + 3
            step = _fd_step(param_idx)
            p_axis0 = p_vec[..., axis]

            def _ham_at_p_axis(p_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                p_eval = p_vec.clone()
                p_eval[..., axis] = p_axis_eval
                L_eval = _cross3(r_vec, p_eval)
                _refresh_hcoeffs(
                    params,
                    L_eval,
                    S1,
                    S2,
                    S1_weighted=S1_weighted_override,
                    S2_weighted=S2_weighted_override,
                )
                return _eob_potentials(
                    r,
                    pr,
                    phi,
                    L_eval,
                    S1,
                    S2,
                    params,
                    p_vec=p_eval,
                    r_vec=r_vec,
                    compute_grad_p=False,
                    p_is_tortoise=p_is_tortoise,
                    fd_dpvec=False,
                    fd_pphi=False,
                    compute_base_grad=False,
                    S1_weighted_override=S1_weighted_override,
                    S2_weighted_override=S2_weighted_override,
                )["H"] / eta_t

            dH_dp.append(_gsl_deriv_central(_ham_at_p_axis, p_axis0, step))
        dH_dpvec = torch.stack(dH_dp, dim=-1)
        params.hcoeffs = hcoeffs_saved

    def _zero_if_none(x):
        if x is None:
            return torch.zeros_like(Hreal)
        return x

    def _zero_vec_if_none(x, ref):
        if x is None:
            return torch.zeros_like(ref)
        return x

    return {
        "u": u_b.detach(),
        "deltaT": deltaT_b.detach(),
        "deltaR": deltaR_b.detach(),
        "D": D_b.detach(),
        "Lambda": Lambda_b.detach(),
        "csi": csi_out.detach(),
        "dcsi": dcsi_b.detach(),
        "n_hat": n_hat.detach(),
        "xi_vec": xi_vec.detach(),
        "prT": prT.detach(),
        "pf": pf0,
        "ptheta2": ptheta2.detach(),
        "Heff": H_eff.detach(),
        "H": Hreal.detach(),
        "dH_dpr": (_zero_if_none(dH_dpr) * inv_eta).detach(),
        "dH_dpf": (
            dH_dpf_fd.detach()
            if dH_dpf_fd is not None
            else torch.where(
                torch.isnan(_zero_if_none(dH_dpf)) | (_zero_if_none(dH_dpf).abs() < 1.0e-14),
                (pf0 / torch.clamp(r_newton * r_newton * Hreal_fallback, min=1.0e-12) * inv_eta).detach(),
                (_zero_if_none(dH_dpf) * inv_eta).detach(),
            )
        ),
        "dH_dr": (_zero_if_none(dH_dr) * inv_eta).detach(),
        "dH_dpvec": None if dH_dpvec is None else (dH_dpvec if fd_dpvec else dH_dpvec * inv_eta).detach(),
        "dH_dx": None if dH_dx is None else (_zero_vec_if_none(dH_dx, r_vec) * inv_eta).detach(),
        "dH_dS1": None if dH_dS1 is None else (_zero_vec_if_none(dH_dS1, S1) * inv_eta).detach(),
        "dH_dS2": None if dH_dS2 is None else (_zero_vec_if_none(dH_dS2, S2) * inv_eta).detach(),
    }


def _ic_cartesian_potential(
    params: EOBParams,
    r: float,
    px: float,
    py: float,
    pz: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    device,
    dtype,
    compute_grad_p: bool,
    compute_grad_x: bool = False,
    p_is_tortoise: bool,
    fd_dpvec: bool = False,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Hamiltonian helper for the LAL precessing IC equations."""

    zero = torch.zeros((), device=device, dtype=dtype)
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    p_vec = torch.stack(
        [
            torch.as_tensor(px, device=device, dtype=dtype),
            torch.as_tensor(py, device=device, dtype=dtype),
            torch.as_tensor(pz, device=device, dtype=dtype),
        ]
    )
    r_vec = torch.stack([r_t, zero, zero])
    if compute_grad_x:
        r_vec = r_vec.detach().clone().requires_grad_(True)
        r_t = _safe_norm(r_vec)
    L_vec = _cross3(r_vec, p_vec)
    _refresh_hcoeffs(
        params,
        L_vec,
        S1,
        S2,
        S1_weighted=S1_weighted_override,
        S2_weighted=S2_weighted_override,
    )
    return _eob_potentials(
        r_t,
        p_vec[0],
        zero,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=compute_grad_p,
        compute_grad_x=compute_grad_x,
        p_is_tortoise=p_is_tortoise,
        fd_dpvec=fd_dpvec,
        fd_pphi=False,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
    ), L_vec


def _aligned_ic_spherical_derivatives(
    params: EOBParams,
    r: float,
    ptheta: float,
    pphi: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    device,
    dtype,
    S1_weighted_override: torch.Tensor,
    S2_weighted_override: torch.Tensor,
):
    """Optimized-v4 Hamiltonian derivatives in LAL's aligned IC frame."""

    py = pphi / r
    pz = -ptheta / r
    pot, _ = _ic_cartesian_potential(
        params,
        r,
        0.0,
        py,
        pz,
        S1,
        S2,
        device=device,
        dtype=dtype,
        compute_grad_p=True,
        compute_grad_x=True,
        p_is_tortoise=False,
        fd_dpvec=False,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
    )
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    ptheta_t = torch.as_tensor(ptheta, device=device, dtype=dtype)
    pphi_t = torch.as_tensor(pphi, device=device, dtype=dtype)
    dHdx = pot["dH_dx"][0]
    dHdpy = pot["dH_dpvec"][1]
    dHdpz = pot["dH_dpvec"][2]
    r2 = torch.clamp(r_t * r_t, min=1.0e-15)
    dHdr = dHdx - dHdpy * pphi_t / r2 + dHdpz * ptheta_t / r2
    dHdptheta = -dHdpz / torch.clamp(r_t, min=1.0e-15)
    dHdpphi = dHdpy / torch.clamp(r_t, min=1.0e-15)
    return torch.stack([dHdr, dHdptheta, dHdpphi])


def _aligned_ic_spherical_second_derivative(
    params: EOBParams,
    idx2: int,
    r: float,
    ptheta: float,
    pphi: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    device,
    dtype,
    S1_weighted_override: torch.Tensor,
    S2_weighted_override: torch.Tensor,
) -> torch.Tensor:
    """LAL's ``XLALCalculateSphHamiltonianDeriv2Hybrid`` for v4 ICs."""

    component = {0: 0, 5: 2}[idx2]
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    step = torch.as_tensor(1.0e-4, device=device, dtype=dtype)

    def _wrapped(r_eval: torch.Tensor) -> torch.Tensor:
        return _aligned_ic_spherical_derivatives(
            params,
            float(r_eval.detach()),
            ptheta,
            pphi,
            S1,
            S2,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_weighted_override,
            S2_weighted_override=S2_weighted_override,
        )[component]

    return _gsl_deriv_central(_wrapped, r_t, step)


def _ic_spherical_derivatives(
    params: EOBParams,
    r: float,
    ptheta: float,
    pphi: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    device,
    dtype,
    py_cartesian: float | None = None,
    pz_cartesian: float | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Return dH/dr, dH/dptheta, dH/dpphi in the LAL IC spherical frame."""

    # XLALFindSphericalOrbitPrec retains the Cartesian momenta supplied by the
    # root solver. Reconstructing them from ptheta/pphi can round by one ulp,
    # which is enough to send GSL's ill-conditioned hybrid solve down a
    # measurably different path. The second-derivative wrapper, by contrast,
    # really does perform this spherical-to-Cartesian conversion.
    py = pphi / r if py_cartesian is None else py_cartesian
    pz = -ptheta / r if pz_cartesian is None else pz_cartesian
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    r_vec = torch.stack([r_t, zero, zero])
    p_vec = torch.stack(
        [
            zero,
            torch.as_tensor(py, device=device, dtype=dtype),
            torch.as_tensor(pz, device=device, dtype=dtype),
        ]
    )
    L_vec = _cross3(r_vec, p_vec)
    entry_hcoeffs = dict(params.hcoeffs)
    _refresh_hcoeffs(
        params,
        L_vec,
        S1,
        S2,
        S1_weighted=S1_weighted_override,
        S2_weighted=S2_weighted_override,
    )

    # LAL's XLALFindSphericalOrbitPrec calls XLALSpinPrecHcapNumericalDerivative
    # with ignoreflux=1, then maps dvalues[1], dvalues[2], -dvalues[3] into
    # dH/dpy, dH/dpz, dH/dx for the spherical residual.
    # LAL builds this prelude from the seobCoeffs state present on entry. After
    # the numerical derivatives, it refreshes the shared coefficients at the
    # base state for use by the next root-function evaluation.
    tortoise_prelude = _lal_numerical_derivative_tortoise_prelude(
        r_t,
        L_vec,
        S1,
        S2,
        params,
        entry_hcoeffs,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
    )
    csi_fac = torch.clamp(tortoise_prelude["csi"], min=1.0e-15)
    dcsi = tortoise_prelude["dcsi"]
    Tmat, invTmat, dTijdXk = _tortoise_matrices(r_vec, csi_fac, dcsi)
    n_hat = r_vec / _safe_norm(r_vec)
    pr_star = torch.dot(p_vec, n_hat)
    p_non_tortoise = p_vec - n_hat * pr_star * (csi_fac - 1.0) / csi_fac

    # XLALSpinPrecHcapNumericalDerivative evaluates the fixed-P Cartesian
    # x-derivatives first, leaves the shared tortoise flag at 1, and only then
    # evaluates the p-derivatives used for dx/dt.
    dH_dx = _dH_dx_cartesian_fd(
        r_vec,
        p_non_tortoise,
        S1,
        S2,
        params,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
    )
    params.tortoise = 1
    pot = _eob_potentials(
        r_t,
        p_vec[0],
        zero,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=False,
        p_is_tortoise=True,
        fd_dpvec=True,
        fd_pphi=False,
        compute_base_grad=False,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
    )
    dxdt = _matvec3_lal_order(Tmat, pot["dH_dpvec"])
    pdot_t1 = _pdot_t1_lal_order(Tmat, dH_dx)
    pdot_t3 = _pdot_t3_lal_order(dTijdXk, invTmat, p_vec, dxdt)
    dpdt = pdot_t1 + pdot_t3

    dHdx = -dpdt[0]
    dHdpy = dxdt[1]
    dHdpz = dxdt[2]
    ptheta_t = torch.as_tensor(ptheta, device=device, dtype=dtype)
    pphi_t = torch.as_tensor(pphi, device=device, dtype=dtype)
    dHdr = dHdx - dHdpy * pphi_t / torch.clamp(r_t * r_t, min=1.0e-15) + dHdpz * ptheta_t / torch.clamp(r_t * r_t, min=1.0e-15)
    dHdptheta = -dHdpz / torch.clamp(r_t, min=1.0e-15)
    dHdpphi = dHdpy / torch.clamp(r_t, min=1.0e-15)
    return torch.stack([dHdr, dHdptheta, dHdpphi])


def _precessing_ic_root_radius(x_scaled: float) -> float:
    """LAL's Python extension root callback uses split multiply/add here."""

    return math.sqrt(x_scaled * x_scaled + 36.0)


def _robust_gsl_derivative(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """LAL's XLALRobustDerivative retry wrapper around gsl_deriv_central."""

    result, abs_err = _gsl_deriv_central_with_error(fn, x, h)
    frac = 0.01

    def _bad(res: torch.Tensor, err: torch.Tensor) -> bool:
        res_f = abs(float(res.detach()))
        err_f = abs(float(err.detach()))
        return err_f > frac * res_f

    if not _bad(result, abs_err):
        return result

    for n in range(1, 11):
        h1 = h * float(2 * n)
        h2 = h / float(2 * n)
        temp1, abs_err1 = _gsl_deriv_central_with_error(fn, x, h1)
        temp2, abs_err2 = _gsl_deriv_central_with_error(fn, x, h2)
        t1 = abs(float(temp1.detach()))
        t2 = abs(float(temp2.detach()))
        e1 = abs(float(abs_err1.detach()))
        e2 = abs(float(abs_err2.detach()))
        rel1 = math.inf if t1 == 0.0 else e1 / t1
        rel2 = math.inf if t2 == 0.0 else e2 / t2
        if rel1 < rel2 and e1 < frac * t1:
            return temp1
        if rel1 > rel2 and e2 < frac * t2:
            return temp2

    raise RuntimeError("second derivative computation failed")


def _ic_spherical_second_derivative(
    params: EOBParams,
    idx1: int,
    idx2: int,
    r: float,
    ptheta: float,
    pphi: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    device,
    dtype,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """LAL-order XLALCalculateSphHamiltonianDeriv2Prec for ICs."""

    values = [float(r), 0.5 * _PI, 0.0, 0.0, float(ptheta), float(pphi)]
    x0 = torch.as_tensor(values[idx1], device=device, dtype=dtype)
    h = torch.as_tensor(3.0e-3, device=device, dtype=dtype)
    component = {0: 0, 4: 1, 5: 2}[idx2]

    def _wrapped(x_eval: torch.Tensor) -> torch.Tensor:
        sph = list(values)
        sph[idx1] = float(x_eval.detach())
        return _ic_spherical_derivatives(
            params,
            sph[0],
            sph[4],
            sph[5],
            S1,
            S2,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_weighted_override,
            S2_weighted_override=S2_weighted_override,
        )[component]

    return _robust_gsl_derivative(_wrapped, x0, h)


def _precessing_ic_cartesian_state(
    params: EOBParams,
    r: float,
    py: float,
    pz: float,
    radial_summary: dict,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    final_tortoise: int,
    device,
    dtype,
) -> torch.Tensor:
    """Complete LAL IC Steps 3--5 without a reduced-state round trip."""

    zero = torch.zeros((), device=device, dtype=dtype)
    r_t = torch.as_tensor(r, device=device, dtype=dtype)
    py_t = torch.as_tensor(py, device=device, dtype=dtype)
    pz_t = torch.as_tensor(pz, device=device, dtype=dtype)
    q_cart = torch.stack([r_t, zero, zero])
    p_cart = torch.stack([zero, py_t, pz_t])

    q_hat = q_cart / _lal_scalar_sqrt(_dot3(q_cart, q_cart))
    p_hat = p_cart / _lal_scalar_sqrt(_dot3(p_cart, p_cart))
    L_hat = _cross3(q_hat, p_hat)
    L_hat = L_hat / _lal_scalar_sqrt(_dot3(L_hat, L_hat))
    rot_matrix2 = torch.stack([q_hat, p_hat, L_hat])
    inv_matrix2 = rot_matrix2.transpose(0, 1).contiguous()

    mass1 = torch.as_tensor(params.mass1, device=device, dtype=dtype)
    mass2 = torch.as_tensor(params.mass2, device=device, dtype=dtype)
    total_mass = mass1 + mass2
    total_mass2 = total_mass * total_mass
    S1_dimensional = S1 * mass1 * mass1
    S2_dimensional = S2 * mass2 * mass2
    S1_norm = S1_dimensional / total_mass2
    S2_norm = S2_dimensional / total_mass2

    S1_dimensional_ic = _matvec3_lal_order(rot_matrix2, S1_dimensional)
    S2_dimensional_ic = _matvec3_lal_order(rot_matrix2, S2_dimensional)
    S1_norm_ic = _matvec3_lal_order(rot_matrix2, S1_norm)
    S2_norm_ic = _matvec3_lal_order(rot_matrix2, S2_norm)
    q_cart_ic = _matvec3_lal_order(rot_matrix2, q_cart)
    p_cart_ic = _matvec3_lal_order(rot_matrix2, p_cart)

    # CartesianToSphericalPrec followed by SphericalToCartesianPrec.  Keeping
    # both operations matters: p_phi/r and -p_theta/r round independently.
    q_sph_r = q_cart_ic[0]
    p_theta = -q_sph_r * p_cart_ic[2]
    p_phi = q_sph_r * p_cart_ic[1]
    p_cart_ic = torch.stack(
        [
            radial_summary["pr_non_tortoise"].to(device=device, dtype=dtype),
            p_phi / q_sph_r,
            -p_theta / q_sph_r,
        ]
    )
    q_cart_ic = torch.stack([q_sph_r, zero, zero])

    # Undo Step 3.  The v4P caller fixes the Step-1 IC inclination to zero,
    # so its first rotation and inverse are exactly the identity.
    q_cart = _matvec3_lal_order(inv_matrix2, q_cart_ic)
    p_cart = _matvec3_lal_order(inv_matrix2, p_cart_ic)
    S1_norm = _matvec3_lal_order(inv_matrix2, S1_norm_ic)
    S2_norm = _matvec3_lal_order(inv_matrix2, S2_norm_ic)

    if final_tortoise:
        sigma_kerr_ic = (S1_dimensional_ic + S2_dimensional_ic) / total_mass2
        a = _lal_scalar_sqrt(_dot3(sigma_kerr_ic, sigma_kerr_ic))
        r_final = _lal_scalar_sqrt(_dot3(q_cart, q_cart))
        delta_r = _lal_hamiltonian_delta_r(params.hcoeffs, r_final, params.eta, a)
        delta_t = _lal_hamiltonian_delta_t(params.hcoeffs, r_final, params.eta, a)
        csi = _lal_scalar_sqrt(delta_t * delta_r) / (r_final * r_final + a * a)
        pr = _dot3(q_cart, p_cart) / r_final
        p_cart = torch.stack(
            [
                p_cart[i] + q_cart[i] * pr * (csi - 1.0) / r_final
                for i in range(3)
            ]
        )

    return torch.cat([q_cart, p_cart, S1_norm, S2_norm, zero.view(1), zero.view(1)])


def _precessing_spherical_initial_conditions(
    params: EOBParams,
    omega_target: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    *,
    return_cartesian: bool = False,
    device,
    dtype,
):
    """Solve the LAL precessing spherical-orbit IC equations in scaled variables."""

    omega = float(omega_target.detach().cpu())
    if not math.isfinite(omega) or omega <= 0.0:
        return None
    # LAL seeds the GSL hybrid root with cbrt(omega) and this exact scalar
    # order; equivalent powers perturb the loose residual-stop path.
    v0 = math.cbrt(omega)
    v0_sq = v0 * v0
    x0_sq = (1.0 / v0_sq) * (1.0 / v0_sq) - 36.0
    if not math.isfinite(x0_sq) or x0_sq <= 0.0:
        return None
    x0 = math.sqrt(x0_sq)

    old_tortoise = getattr(params, "tortoise", 2)
    params.tortoise = 0

    def residual(scaled):
        try:
            x_scaled, py_scaled, pz_scaled = [float(v) for v in scaled]
            r = _precessing_ic_root_radius(x_scaled)
            py = py_scaled / 2.0
            pz = pz_scaled / 200.0
            derivs = _ic_spherical_derivatives(
                params,
                r,
                -r * pz,
                r * py,
                S1,
                S2,
                device=device,
                dtype=dtype,
                py_cartesian=py,
                pz_cartesian=pz,
            )
            # LAL's x-derivative block temporarily sets tortoise=2 and then
            # leaves the shared params object at tortoise=1.
            params.tortoise = 1
            out = torch.stack([derivs[0], derivs[1], derivs[2] - omega_target]).detach().cpu().numpy()
            if not all(math.isfinite(float(v)) for v in out):
                return [1.0e3, 1.0e3, 1.0e3]
            return out
        except Exception:
            return [1.0e3, 1.0e3, 1.0e3]

    try:
        guess = [x0, 2.0 * v0, 200.0e-3]
        # The controller and its three-vector linear algebra are host scalar
        # work, while ``residual`` evaluates the Hamiltonian on the active
        # Torch device.  This is therefore the same source-faithful solve on
        # CPU, CUDA, and MPS.
        root_result = _gsl_multiroot_hybrids(
            residual,
            guess,
            epsabs=1.0e-9,
            max_iter=10000,
        )
        if root_result is not None:
            sol_x, final_res = root_result
            sol_success = True
            sol_message = "gsl_multiroot_fsolver_hybrids"
        else:
            # The source algorithm is GSL's scaled hybrids solver.  Returning
            # failure preserves its stop semantics; switching to MINPACK's
            # distinct ``hybr`` iteration changes the carried EOB state.
            sol_x = guess
            final_res = [math.inf, math.inf, math.inf]
            sol_success = False
            sol_message = "gsl_multiroot_fsolver_hybrids failed"
        # Do not re-evaluate residual(sol.x) here: LAL carries the hcoeffs and
        # tortoise side effects from the final root-solver evaluation directly
        # into the radial-momentum step.
        res_norm = math.sqrt(sum(float(v) * float(v) for v in final_res))
        if (not sol_success) or (not math.isfinite(res_norm)) or res_norm > 1.0e-7:
            if _debug_enabled():
                _dbg(f"precessing IC root failed: {sol_message}; residual={final_res}")
            return None

        x_scaled, py_scaled, pz_scaled = [float(v) for v in sol_x]
        r = _precessing_ic_root_radius(x_scaled)
        py = py_scaled / 2.0
        pz = pz_scaled / 200.0
        r_t = torch.as_tensor(r, device=device, dtype=dtype)
        py_t = torch.as_tensor(py, device=device, dtype=dtype)
        pz_t = torch.as_tensor(pz, device=device, dtype=dtype)
        r_vec = torch.stack([r_t, torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)])
        p_vec = torch.stack([torch.zeros((), device=device, dtype=dtype), py_t, pz_t])
        L_vec = _cross3(r_vec, p_vec)
        radial_summary = _precessing_ic_radial_momentum_summary(
            params,
            r,
            py,
            pz,
            L_vec,
            S1,
            S2,
            omega_target,
            final_tortoise=old_tortoise,
            device=device,
            dtype=dtype,
        )
        if radial_summary is None:
            pr_star = torch.as_tensor(-(64.0 / 5.0) * params.eta / max(r ** 3, 1.0e-12), device=device, dtype=dtype)
        else:
            pr_star = radial_summary["pr_star"]
        if return_cartesian and radial_summary is not None:
            return _precessing_ic_cartesian_state(
                params,
                r,
                py,
                pz,
                radial_summary,
                S1,
                S2,
                final_tortoise=old_tortoise,
                device=device,
                dtype=dtype,
            )
        xhat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        e1, _, _ = _orbital_basis_from_L_phi(torch.zeros((), device=device, dtype=dtype), L_vec, S1, S2, params)
        e2, _, _ = _orbital_basis_from_L_phi(torch.as_tensor(0.5 * _PI, device=device, dtype=dtype), L_vec, S1, S2, params)
        phi0 = torch.atan2(torch.dot(xhat, e2), torch.dot(xhat, e1))
        zero_t = torch.zeros((), device=device, dtype=dtype)
        phi0 = torch.where(torch.abs(phi0) <= 1.0e-15, zero_t, phi0)
        L_vec = torch.where(torch.abs(L_vec) <= 1.0e-15, torch.zeros_like(L_vec), L_vec)
        return r_t, pr_star, phi0, L_vec
    finally:
        params.tortoise = old_tortoise


def _precessing_ic_radial_momentum_summary(
    params: EOBParams,
    r: float,
    py: float,
    pz: float,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    omega_target: torch.Tensor,
    *,
    final_tortoise: int | None = None,
    include_flux_modes: bool = False,
    device,
    dtype,
):
    """Energy-balance radial momentum ingredients for the precessing IC root."""

    try:
        # Step 3 of XLALSimIMRSpinEOBInitialConditionsPrec normalizes pCart,
        # installs that vector as the second row of rotMatrix2, and applies
        # the matrix through gsl_blas_dgemv.  Reusing the normalization as
        # hypot(py, pz) is mathematically equivalent, but the BLAS dot product
        # rounds the rotated tangential momentum one ulp differently.
        p_unrotated_norm = math.sqrt(py * py + pz * pz)
        if p_unrotated_norm <= 0.0:
            return None
        zero = torch.zeros((), device=device, dtype=dtype)
        one = torch.ones((), device=device, dtype=dtype)
        py_hat_f = py / p_unrotated_norm
        pz_hat_f = pz / p_unrotated_norm
        p_norm = py_hat_f * py + pz_hat_f * pz
        py_hat = torch.as_tensor(py_hat_f, device=device, dtype=dtype)
        pz_hat = torch.as_tensor(pz_hat_f, device=device, dtype=dtype)
        rot_matrix2 = torch.stack(
            [
                torch.stack([one, zero, zero]),
                torch.stack([zero, py_hat, pz_hat]),
                torch.stack([zero, -pz_hat, py_hat]),
            ]
        )
        S1_ic = _matvec3_lal_order(rot_matrix2, S1)
        S2_ic = _matvec3_lal_order(rot_matrix2, S2)
        # LAL rotates three representations independently in Step 3: chi for
        # the returned state, dimensionful S_i=chi_i*m_i^2 for numerical
        # derivatives, and S_i/M^2 for the direct Hamiltonian and flux.  The
        # alternatives are algebraically equivalent, but differ by a few ulps
        # after the BLAS rotation and those ulps steer GSL's derivative retry.
        mass1 = torch.as_tensor(params.mass1, device=device, dtype=dtype)
        mass2 = torch.as_tensor(params.mass2, device=device, dtype=dtype)
        total_mass = mass1 + mass2
        total_mass2 = total_mass * total_mass
        S1_dimensional = S1 * mass1 * mass1
        S2_dimensional = S2 * mass2 * mass2
        S1_deriv_ic = _matvec3_lal_order(rot_matrix2, S1_dimensional) / total_mass2
        S2_deriv_ic = _matvec3_lal_order(rot_matrix2, S2_dimensional) / total_mass2
        S1_norm_ic = _matvec3_lal_order(rot_matrix2, S1_dimensional / total_mass2)
        S2_norm_ic = _matvec3_lal_order(rot_matrix2, S2_dimensional / total_mass2)
        py_ic = p_norm
        pz_ic = 0.0
        ptheta = 0.0
        pphi = r * p_norm
        L_vec_ic = torch.stack(
            [
                zero,
                zero,
                torch.as_tensor(pphi, device=device, dtype=dtype),
            ]
        )
        d2Hdr2 = _ic_spherical_second_derivative(
            params,
            0,
            0,
            r,
            ptheta,
            pphi,
            S1_ic,
            S2_ic,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_deriv_ic,
            S2_weighted_override=S2_deriv_ic,
        )
        d2Hdrdpphi = _ic_spherical_second_derivative(
            params,
            0,
            5,
            r,
            ptheta,
            pphi,
            S1_ic,
            S2_ic,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_deriv_ic,
            S2_weighted_override=S2_deriv_ic,
        )
        r_t = torch.as_tensor(r, device=device, dtype=dtype)
        pot, _ = _ic_cartesian_potential(
            params,
            r,
            0.0,
            py_ic,
            pz_ic,
            S1_ic,
            S2_ic,
            device=device,
            dtype=dtype,
            compute_grad_p=False,
            p_is_tortoise=True,
            fd_dpvec=True,
            S1_weighted_override=S1_deriv_ic,
            S2_weighted_override=S2_deriv_ic,
        )
        dHdpphi = pot["dH_dpvec"][1] / r_t
        if torch.abs(d2Hdrdpphi) < 1.0e-14 or torch.abs(d2Hdr2) < 1.0e-14:
            return None
        dEdr = -dHdpphi * d2Hdr2 / d2Hdrdpphi
        r_vec_ic = torch.stack([r_t, zero, zero])
        p_vec_ic = torch.stack(
            [
                zero,
                torch.as_tensor(py_ic, device=device, dtype=dtype),
                zero,
            ]
        )
        flux_modes = {} if include_flux_modes else None
        flux = _factorized_flux(
            r_t,
            zero,
            zero,
            L_vec_ic,
            S1_ic,
            S2_ic,
            params,
            omega_target,
            pot["H"],
            deltaT=pot["deltaT"],
            D=pot["D"],
            r_vec=r_vec_ic,
            p_vec=p_vec_ic,
            mode_contributions=flux_modes,
            S1_weighted_override=S1_norm_ic,
            S2_weighted_override=S2_norm_ic,
        )
        pr_probe = 1.0e-3
        pot_pr, _ = _ic_cartesian_potential(
            params,
            r,
            pr_probe,
            py_ic,
            pz_ic,
            S1_ic,
            S2_ic,
            device=device,
            dtype=dtype,
            compute_grad_p=False,
            p_is_tortoise=True,
            fd_dpvec=True,
            S1_weighted_override=S1_deriv_ic,
            S2_weighted_override=S2_deriv_ic,
        )
        ic_probe_hcoeffs = dict(params.hcoeffs)
        saved_tortoise = getattr(params, "tortoise", 2)
        try:
            if final_tortoise is not None:
                params.tortoise = final_tortoise
            ic_probe_tortoise = _lal_numerical_derivative_tortoise_prelude(
                r_t,
                L_vec_ic,
                S1_ic,
                S2_ic,
                params,
                ic_probe_hcoeffs,
                S1_weighted_override=S1_deriv_ic,
                S2_weighted_override=S2_deriv_ic,
            )
        finally:
            params.tortoise = saved_tortoise
        csi_ic = torch.clamp(ic_probe_tortoise["csi"], min=1.0e-15)
        Tmat_ic, _, _ = _tortoise_matrices(
            torch.stack([r_t, zero, zero]),
            csi_ic,
            ic_probe_tortoise["dcsi"],
        )
        dxdt_probe = _matvec3_lal_order(Tmat_ic, pot_pr["dH_dpvec"])
        dHdpr = csi_ic * dxdt_probe[0]
        if torch.abs(dHdpr) < 1.0e-14 or torch.abs(dEdr) < 1.0e-14:
            return None
        r_dot = -flux / dEdr
        pr_non_tortoise = r_dot / (dHdpr / torch.as_tensor(pr_probe, device=device, dtype=dtype))
        pr_star = csi_ic * pr_non_tortoise
        if not bool(torch.isfinite(pr_star).item()):
            return None
        # LAL does not restore seobCoeffs after this final IC derivative. The
        # first RHS tortoise prelude consumes this cached coefficient state.
        params.hcoeffs = ic_probe_hcoeffs
        result = {
            "pr_star": pr_star.detach(),
            "pr_non_tortoise": pr_non_tortoise.detach(),
            "r_dot": r_dot.detach(),
            "d2Hdr2": d2Hdr2.detach(),
            "d2Hdrdpphi": d2Hdrdpphi.detach(),
            "dHdpphi": dHdpphi.detach(),
            "dEdr": dEdr.detach(),
            "H": pot["H"].detach(),
            "flux": flux.detach(),
            "probe_dH_dpvec0": pot_pr["dH_dpvec"][0].detach(),
            "dxdt_probe0": dxdt_probe[0].detach(),
            "csi": csi_ic.detach(),
            "dHdpr": dHdpr.detach(),
            "p_norm": torch.as_tensor(p_norm, device=device, dtype=dtype),
            "pphi": torch.as_tensor(pphi, device=device, dtype=dtype),
        }
        if flux_modes is not None:
            result["flux_modes"] = flux_modes
        return result
    except Exception:
        return None


_orig_precessing_ic_radial_momentum_summary = _precessing_ic_radial_momentum_summary


def _precessing_ic_radial_momentum(
    params: EOBParams,
    r: float,
    py: float,
    pz: float,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    omega_target: torch.Tensor,
    *,
    final_tortoise: int | None = None,
    device,
    dtype,
):
    """Energy-balance radial tortoise momentum for the precessing IC root."""

    summary = _precessing_ic_radial_momentum_summary(
        params,
        r,
        py,
        pz,
        L_vec,
        S1,
        S2,
        omega_target,
        final_tortoise=final_tortoise,
        device=device,
        dtype=dtype,
    )
    if summary is None:
        return None
    return summary["pr_star"]


def _aligned_ic_radial_momentum_summary(
    params: EOBParams,
    r: float,
    pphi: float,
    S1: torch.Tensor,
    S2: torch.Tensor,
    omega_target: torch.Tensor,
    *,
    device,
    dtype,
):
    """Energy-balance radial momentum from aligned EOB IC step 4."""

    old_tortoise = getattr(params, "tortoise", 1)
    hcoeffs_saved = dict(params.hcoeffs)
    try:
        # XLALSimIMRSpinEOBInitialConditions solves with ordinary canonical
        # momentum, then applies the requested tortoise transform at the end.
        params.tortoise = 0
        zero = torch.zeros((), device=device, dtype=dtype)
        r_t = torch.as_tensor(r, device=device, dtype=dtype)
        pphi_t = torch.as_tensor(pphi, device=device, dtype=dtype)
        S1_weighted = _lal_spin_weight(S1, params.mass1, params.M)
        S2_weighted = _lal_spin_weight(S2, params.mass2, params.M)

        d2Hdr2 = _aligned_ic_spherical_second_derivative(
            params,
            0,
            r,
            0.0,
            pphi,
            S1,
            S2,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )
        d2Hdrdpphi = _aligned_ic_spherical_second_derivative(
            params,
            5,
            r,
            0.0,
            pphi,
            S1,
            S2,
            device=device,
            dtype=dtype,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )
        if torch.abs(d2Hdr2) < 1.0e-14 or torch.abs(d2Hdrdpphi) < 1.0e-14:
            return None

        py = pphi / r
        pot, L_vec = _ic_cartesian_potential(
            params,
            r,
            0.0,
            py,
            0.0,
            S1,
            S2,
            device=device,
            dtype=dtype,
            compute_grad_p=True,
            p_is_tortoise=False,
            fd_dpvec=False,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )
        dHdpphi = pot["dH_dpvec"][1] / r_t
        dEdr = -dHdpphi * d2Hdr2 / d2Hdrdpphi
        if torch.abs(dEdr) < 1.0e-14:
            return None

        r_vec = torch.stack([r_t, zero, zero])
        p_vec = torch.stack([zero, torch.as_tensor(py, device=device, dtype=dtype), zero])
        flux = _factorized_flux(
            r_t,
            zero,
            zero,
            L_vec,
            S1,
            S2,
            params,
            omega_target,
            pot["H"],
            deltaT=pot["deltaT"],
            D=pot["D"],
            r_vec=r_vec,
            p_vec=p_vec,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )

        pr_probe = 1.0e-3
        pot_pr, _ = _ic_cartesian_potential(
            params,
            r,
            pr_probe,
            py,
            0.0,
            S1,
            S2,
            device=device,
            dtype=dtype,
            compute_grad_p=True,
            p_is_tortoise=False,
            fd_dpvec=False,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )
        dHdpr = pot_pr["dH_dpvec"][0]
        if torch.abs(dHdpr) < 1.0e-14:
            return None

        r_dot = -flux / dEdr
        pr_non_tortoise = r_dot / (dHdpr / pr_probe)

        # This is the final transform in XLALSimIMRSpinEOBInitialConditions,
        # using standalone DeltaT/DeltaR routines and the aligned Kerr spin.
        sigma_kerr = S1_weighted + S2_weighted
        a = _safe_norm(sigma_kerr)
        delta_t = _lal_hamiltonian_delta_t(params.hcoeffs, r_t, params.eta, a)
        delta_r = _lal_hamiltonian_delta_r(params.hcoeffs, r_t, params.eta, a)
        csi = _lal_scalar_sqrt(delta_t * delta_r) / (r_t * r_t + a * a)
        pr_star = csi * pr_non_tortoise if old_tortoise else pr_non_tortoise
        if not bool(torch.isfinite(pr_star).item()):
            return None
        return {
            "pr_star": pr_star.detach(),
            "pr_non_tortoise": pr_non_tortoise.detach(),
            "r_dot": r_dot.detach(),
            "d2Hdr2": d2Hdr2.detach(),
            "d2Hdrdpphi": d2Hdrdpphi.detach(),
            "dHdpphi": dHdpphi.detach(),
            "dEdr": dEdr.detach(),
            "H": pot["H"].detach(),
            "flux": flux.detach(),
            "dHdpr": dHdpr.detach(),
            "csi": csi.detach(),
            "pphi": pphi_t,
        }
    except Exception:
        return None
    finally:
        params.tortoise = old_tortoise
        params.hcoeffs = hcoeffs_saved


def initial_conditions(params, *, device, dtype):
    """Quasicircular initial conditions at f_lower using force-balance solve."""

    hcoeffs_saved = dict(params.hcoeffs)

    def _return_fallback_without_probe_hcoeff_side_effect(y: torch.Tensor) -> torch.Tensor:
        # The fallback solver is torch-local; keep its probe state out of the
        # subsequent RHS. The precessing root path above mirrors LAL's leak.
        params.hcoeffs = hcoeffs_saved
        return y

    omega_target = torch.tensor(_PI * params.f_lower * params.M_sec, device=device, dtype=dtype)
    use_projected_rhs = _env_on("PYCBC_SEOBNRV4PHM_PROJECTED_IC", True)
    use_projected_rhs = use_projected_rhs and _env_on("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", True)
    use_precessing_root = (
        use_projected_rhs
        and (not getattr(params, "aligned_spins", False))
        and _env_on("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", True)
    )
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)

    if use_precessing_root:
        rooted = _precessing_spherical_initial_conditions(params, omega_target, S1, S2, device=device, dtype=dtype)
        if rooted is not None:
            r, pr0, phi0, L0 = rooted
            return torch.cat([r.view(1), pr0.view(1), phi0.view(1), L0, S1, S2])

    def rhs_for_ic(t, y):
        if use_projected_rhs:
            return rhs_cartesian_projected(t, y, params)
        return rhs(t, y, params)

    def omega_from_state(r, pphi):
        y = torch.tensor(
            [
                r,
                0.0,
                0.0,
                0.0,
                0.0,
                pphi,
                params.spin1x,
                params.spin1y,
                params.spin1z,
                params.spin2x,
                params.spin2y,
                params.spin2z,
            ],
            device=device,
            dtype=dtype,
        )
        return rhs_for_ic(torch.tensor(0.0, device=device, dtype=dtype), y)[2]

    def dprdt_from_state(r, pphi):
        y = torch.tensor([r, 0.0, 0.0, 0.0, 0.0, pphi, params.spin1x, params.spin1y, params.spin1z, params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
        return rhs_for_ic(torch.tensor(0.0, device=device, dtype=dtype), y)[1]

    def pphi_circular(r):
        p = torch.sqrt(torch.clamp(r, min=1e-9))
        for _ in range(50):
            F = dprdt_from_state(r, p)
            if torch.abs(F) < 1e-12:
                break
            dp = torch.clamp(p * 1e-5 + 1e-9, min=1e-7)
            Fp = dprdt_from_state(r, p + dp)
            dF = (Fp - F) / dp
            dF_safe = torch.sign(dF) * torch.clamp(torch.abs(dF), min=1e-12)
            step = torch.clamp(F / dF_safe, -p * 0.5, p * 0.5)
            p = torch.clamp(p - step, min=1e-6)
        return p

    r = torch.pow(torch.clamp(omega_target, min=1e-12), -2.0 / 3.0)
    for _ in range(30):
        pphi = pphi_circular(r)
        omega = omega_from_state(r, pphi)
        err = omega - omega_target
        if torch.abs(err) < 1e-10:
            break
        dr = torch.clamp(r * 5e-3 + 1e-4, min=1e-4)
        pphi_dr = pphi_circular(r + dr)
        omega_dr = omega_from_state(r + dr, pphi_dr)
        domega = (omega_dr - omega) / dr
        if torch.abs(domega) < 1e-12:
            r = torch.clamp(r * (omega_target / torch.clamp(omega, min=1e-12)) ** (-2.0 / 3.0), min=3.0)
        else:
            r = torch.clamp(r - err / domega, min=3.0)
    pphi = pphi_circular(r)

    if use_projected_rhs and getattr(params, "aligned_spins", False):
        radial_summary = _aligned_ic_radial_momentum_summary(
            params,
            float(r),
            float(pphi),
            S1,
            S2,
            omega_target,
            device=device,
            dtype=dtype,
        )
        if radial_summary is None:
            pr0 = torch.tensor(
                -(64.0 / 5.0) * params.eta / max(float(r) ** 3, 1.0e-12),
                device=device,
                dtype=dtype,
            )
        else:
            pr0 = radial_summary["pr_star"]
    elif use_projected_rhs:
        pr0 = torch.tensor(-(64.0 / 5.0) * params.eta / max(float(r) ** 3, 1.0e-12), device=device, dtype=dtype)
    else:
        pr0 = torch.tensor(0.0, device=device, dtype=dtype)
    phi0 = torch.tensor(0.0, device=device, dtype=dtype)
    L0 = torch.stack([torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype), pphi])
    return _return_fallback_without_probe_hcoeff_side_effect(
        torch.cat([r.view(1), pr0.view(1), phi0.view(1), L0, S1, S2])
    )


def initial_cartesian_conditions(params, *, device, dtype):
    """LAL-order 14-component initial state for full Cartesian dynamics."""

    hcoeffs_saved = dict(params.hcoeffs)
    use_precessing_root = (
        _env_on("PYCBC_SEOBNRV4PHM_PROJECTED_IC", True)
        and _env_on("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", True)
        and (not getattr(params, "aligned_spins", False))
        and _env_on("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", True)
    )
    if use_precessing_root:
        if (
            isinstance(device, torch.device)
            and device.type == "cpu"
            and dtype == torch.float64
            and _env_on("PYCBC_SEOBNR_NATIVE_ODE", True)
            and _precessing_ic_radial_momentum_summary is _orig_precessing_ic_radial_momentum_summary
        ):
            try:
                from pycbc.waveform._seobnr_native_ode import get_extension

                ext = get_extension()
                if ext is not None and hasattr(ext, "initial_cartesian_conditions_native"):
                    res = ext.initial_cartesian_conditions_native(
                        float(params.mass1),
                        float(params.mass2),
                        float(params.spin1x),
                        float(params.spin1y),
                        float(params.spin1z),
                        float(params.spin2x),
                        float(params.spin2y),
                        float(params.spin2z),
                        float(params.f_lower),
                    )
                    if res is not None and res.numel() == 14:
                        return res.to(device=device, dtype=dtype)
            except Exception:
                pass

        omega_target = torch.tensor(_PI * params.f_lower * params.M_sec, device=device, dtype=dtype)
        S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
        S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
        rooted = _precessing_spherical_initial_conditions(
            params,
            omega_target,
            S1,
            S2,
            return_cartesian=True,
            device=device,
            dtype=dtype,
        )
        if rooted is not None and rooted.numel() == 14:
            return rooted
        params.hcoeffs = hcoeffs_saved

    return reduced_state_to_cartesian_state(initial_conditions(params, device=device, dtype=dtype), params)


def _rho_for_mode(l: int, m: int) -> Callable:
    if l == 2 and m == 2:
        return rho22
    if l == 2 and m == 1:
        return rho21
    if l == 3 and m == 3:
        return rho33
    if l == 4 and m == 4:
        return rho44
    if l == 5 and m == 5:
        return rho55
    raise ValueError(f"Unsupported mode ({l},{m}) for flux")


def _debug_enabled():
    return os.environ.get("PYCBC_SEOBNRV4PHM_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(msg: str):
    if _debug_enabled():
        print(f"[seobnrv4phm] {msg}", flush=True)


@torch.jit.script
def _tail_factor_mag(l: int, m: int, omega: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Absolute |T_lm| matching LALSimIMRSpinEOBFactorizedWaveformPrec.c:675-709."""
    k = torch.abs(float(m) * omega)
    hathatk = torch.clamp(H * k, min=1e-12)
    pi_val = math.pi
    four_pi_k = 4.0 * pi_val * hathatk
    pref = torch.sqrt(four_pi_k / torch.clamp(1.0 - torch.exp(-four_pi_k), min=1e-16))
    log_fact = torch.lgamma(torch.tensor(float(l) + 1.0, device=omega.device, dtype=omega.dtype))
    pref = pref / torch.exp(log_fact)
    prod = torch.ones_like(pref)
    for s in range(1, l + 1):
        prod = prod * (4.0 * hathatk * hathatk + float(s * s))
    return pref * torch.sqrt(prod)


def _factorized_flux(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    omega: torch.Tensor,
    H: torch.Tensor,
    *,
    deltaT: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    r_vec: torch.Tensor | None = None,
    p_vec: torch.Tensor | None = None,
    rdot_vec: torch.Tensor | None = None,
    velocity_vec: torch.Tensor | None = None,
    mode_contributions: dict[str, torch.Tensor] | None = None,
    tplspin_override: torch.Tensor | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
    aligned_derivative: Callable[
        [Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor],
        torch.Tensor,
    ] = _gsl_deriv_central,
):
    """LAL radiation-reaction flux from all l<=8 positive-m dynamics modes."""

    if rdot_vec is None:
        rdot_vec = velocity_vec
    eta = torch.as_tensor(params.eta, device=r.device, dtype=r.dtype)
    v = torch.clamp(torch.abs(omega), min=1e-12) ** (1.0 / 3.0)
    flux = torch.zeros_like(v)
    vphi_nk = non_keplerian_vphi(
        r,
        omega,
        phi,
        L_vec,
        S1,
        S2,
        params,
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=rdot_vec,
        S1_weighted_override=S1_weighted_override,
        S2_weighted_override=S2_weighted_override,
        aligned_derivative=aligned_derivative,
    )
    # XLALSpinPrecHcapNumericalDerivative uses Lhat = r x p for the v4P
    # waveform coefficients that feed the factorized flux.
    spin_axis = L_vec / torch.clamp(_safe_norm(L_vec), min=1.0e-15)
    mass1_norm = torch.as_tensor(params.mass1 / params.M, device=r.device, dtype=r.dtype)
    mass2_norm = torch.as_tensor(params.mass2 / params.M, device=r.device, dtype=r.dtype)
    if S1_weighted_override is None:
        chi1_flux = torch.dot(S1, spin_axis)
    else:
        chi1_flux = torch.dot(S1_weighted_override, spin_axis) / (mass1_norm * mass1_norm)
    if S2_weighted_override is None:
        chi2_flux = torch.dot(S2, spin_axis)
    else:
        chi2_flux = torch.dot(S2_weighted_override, spin_axis) / (mass2_norm * mass2_norm)
    chiS_flux = 0.5 * (chi1_flux + chi2_flux)
    chiA_flux = 0.5 * (chi1_flux - chi2_flux)
    total_mass = torch.as_tensor(params.mass1 + params.mass2, device=r.device, dtype=r.dtype)
    dM_flux = torch.as_tensor(params.mass1 - params.mass2, device=r.device, dtype=r.dtype) / total_mass
    tplspin_flux = (
        (1.0 - 2.0 * eta) * chiS_flux + dM_flux * chiA_flux
        if tplspin_override is None
        else tplspin_override.to(device=r.device, dtype=r.dtype)
    )

    # LAL's dynamics flux is summed independently of the output mode_array.
    for l, m in LM_FLUX:
        eps = 0 if ((l + m) % 2 == 0) else 1
        rho_mag, aux_t = _rho_aux_flux(
            l,
            m,
            v,
            params,
            chi1_flux,
            chi2_flux,
            H=H,
            chiS=chiS_flux,
            chiA=chiA_flux,
            tplspin=tplspin_flux,
        )
        if aux_t.is_complex():
            target = torch.complex64 if v.dtype in (torch.float16, torch.float32) else torch.complex128
            rho_mag = rho_mag.to(target)
            aux_t = aux_t.to(target)
        else:
            aux_t = aux_t.to(dtype=v.dtype)
        if abs(float(eta) - 0.25) < 1e-12 and (m % 2):
            rholm_pwrl = aux_t
        else:
            rholm_pwrl = rho_mag ** l + aux_t

        pref = _calc_prefix(l, m, params.mass1, params.mass2, params.eta)
        yabs = _abs_scalar_sph_pi_over2(l - eps, -m)
        h_newt = abs(pref) * yabs * (vphi_nk ** (l + eps))

        pphi_eff = _safe_norm(L_vec)
        S_eff = ((H * H - 1.0) / (2.0 * eta) + 1.0) if eps == 0 else v * pphi_eff
        tail = _tail_factor_mag(l, m, omega, H)
        amp = h_newt * S_eff * rholm_pwrl * tail
        amp_sq = torch.abs(amp) * torch.abs(amp)
        flux_mode = (m * omega) * (m * omega) * amp_sq
        flux = flux + flux_mode
        if mode_contributions is not None:
            mode_contributions[f"{l},{m}"] = (flux_mode / (8.0 * _PI * eta)).detach()
        if _debug_enabled() and torch.isnan(flux).any():
            deltaT_msg = float(deltaT) if deltaT is not None else float("nan")
            D_msg = float(D) if D is not None else float("nan")
            _dbg(
                f"flux nan l={l} m={m} r={float(r):.3f} pr={float(pr):.3e} "
                f"omega={float(omega):.4e} v={float(v):.4e} deltaT={deltaT_msg:.3e} D={D_msg:.3e} "
                f"rho_mag={float(rho_mag):.3e} S_eff={float(S_eff):.3e} tail={float(tail):.3e}"
            )

    return flux / (8.0 * _PI * eta)


def _aligned_non_keplerian_omega(
    r: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    derivative: Callable[
        [Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor],
        torch.Tensor,
    ] = _gsl_deriv_central,
) -> torch.Tensor:
    """Circular frequency used by LAL's aligned non-Keplerian factor.

    This mirrors ``XLALSimIMRSpinAlignedEOBCalcOmega``: put the binary at
    ``x = r`` with zero radial momentum, then use GSL's central derivative of
    the aligned Hamiltonian with respect to ``p_y``.
    """

    zero = torch.zeros_like(r)
    r_vec = torch.stack((r, zero, zero), dim=-1)
    pphi = _safe_norm(L_vec)
    py = pphi / torch.clamp(r, min=1.0e-15)
    step = torch.as_tensor(
        _LAL_PRECESSING_CALCOMEGA_STEP,
        device=r.device,
        dtype=r.dtype,
    )
    eta = torch.as_tensor(params.eta, device=r.device, dtype=r.dtype)
    hcoeffs_override = params.hcoeffs

    def _ham_at_py(py_eval: torch.Tensor) -> torch.Tensor:
        py_eval = py_eval + zero
        p_vec = torch.stack((zero, py_eval, zero), dim=-1)
        L_eval = _cross3(r_vec, p_vec)
        return _eob_potentials(
            r,
            zero,
            zero,
            L_eval,
            S1,
            S2,
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            compute_grad_p=False,
            compute_base_grad=False,
            fd_dpvec=False,
            fd_pphi=False,
            hcoeffs_override=hcoeffs_override,
        )["H"] / eta

    return torch.abs(
        derivative(_ham_at_py, py, step)
        / torch.clamp(r, min=1.0e-15)
    )


def non_keplerian_vphi(
    r: torch.Tensor,
    omega: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    r_vec: torch.Tensor | None = None,
    p_vec: torch.Tensor | None = None,
    rdot_vec: torch.Tensor | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
    aligned_derivative: Callable[
        [Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor],
        torch.Tensor,
    ] = _gsl_deriv_central,
):
    """Non-Keplerian vPhi using LAL's aligned or precessing coefficient.

    Mirrors ``XLALSimIMRSpinAlignedEOBNonKeplerCoeff`` or
    ``XLALSimIMRSpinPrecEOBNonKeplerCoeff`` plus the follow-up scaling in
    ``LALSimIMRSpinEOBFactorizedWaveformPrec``.
    """

    if (
        r.device.type == "cpu"
        and r.dtype == torch.float64
        and (not torch.is_grad_enabled() or not r.requires_grad)
        and aligned_derivative is _gsl_deriv_central
        and not _env_on("PYCBC_SEOBNRV4PHM_FD_CALCOMEGA", False)
        and os.environ.get("PYCBC_SEOBNR_NATIVE_ODE", "1") not in ("0", "", "false", "False")
    ):
        try:
            from pycbc.waveform._seobnr_native_ode import get_extension
            ext = get_extension()
            if ext is not None and hasattr(ext, "non_keplerian_vphi_native"):
                return ext.non_keplerian_vphi_native(
                    r.contiguous(),
                    omega.contiguous(),
                    phi.contiguous(),
                    L_vec.contiguous(),
                    S1.contiguous(),
                    S2.contiguous(),
                    float(params.mass1),
                    float(params.mass2),
                    bool(params.aligned_spins),
                    None if r_vec is None else r_vec.contiguous(),
                    None if p_vec is None else p_vec.contiguous(),
                    None if rdot_vec is None else rdot_vec.contiguous(),
                    None if S1_weighted_override is None else S1_weighted_override.contiguous(),
                    None if S2_weighted_override is None else S2_weighted_override.contiguous(),
                )
        except Exception:
            pass

    pr0 = torch.zeros_like(r)
    if params.aligned_spins:
        omega_circ = _aligned_non_keplerian_omega(
            r,
            L_vec,
            S1,
            S2,
            params,
            derivative=aligned_derivative,
        )
    elif r_vec is None or p_vec is None:
        aligned_gauge = (
            (torch.abs(L_vec[..., 0]) + torch.abs(L_vec[..., 1])) < 1.0e-14
        ) & (
            (torch.abs(S1[..., 0]) + torch.abs(S1[..., 1]) + torch.abs(S2[..., 0]) + torch.abs(S2[..., 1]))
            < 1.0e-14
        )
        phi_eval = torch.where(aligned_gauge, torch.zeros_like(phi), phi)
        r_eval, p_eval, _, _, _ = _reduced_to_cartesian(r, pr0, phi_eval, L_vec, S1, S2, params)
        omega_circ = torch.abs(
            dphi_dt_fd(
                r,
                pr0,
                phi_eval,
                L_vec,
                S1,
                S2,
                params,
                p_vec=p_eval,
                r_vec=r_eval,
                step=_LAL_PRECESSING_CALCOMEGA_STEP,
            )
        )
    else:
        r_eval = r_vec
        p_eval = p_vec
        omega_circ = torch.abs(
            _calcomega_lal_polar_derivative(
                r_eval,
                p_eval,
                S1,
                S2,
                params,
                rdot_vec=rdot_vec,
                S1_weighted_override=S1_weighted_override,
                S2_weighted_override=S2_weighted_override,
            )
        )
    coeff = torch.clamp(1.0 / (torch.clamp(omega_circ, min=1e-12) ** 2 * torch.clamp(r, min=1e-12) ** 3), min=1e-12)
    vphi = torch.clamp(r * coeff.pow(1.0 / 3.0) * torch.abs(omega), min=1e-12)
    return torch.nan_to_num(vphi, nan=0.0, posinf=0.0, neginf=0.0)


def dphi_dt_fd(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    p_vec: torch.Tensor,
    r_vec: torch.Tensor,
    step: float = 1.0e-6,
):
    """Finite-difference dphi/dt by nudging pphi along phi-hat (LAL GSL mirror)."""

    phi_hat, sin_theta, r_mag = _phi_hat(r_vec)
    delta_pphi = torch.as_tensor(step, device=r.device, dtype=r.dtype)
    eta_t = torch.as_tensor(params.eta, device=r.device, dtype=r.dtype)
    pphi0 = torch.clamp(_safe_norm(L_vec), min=1.0e-15)
    hcoeffs_saved = params.hcoeffs

    def _ham_at_pphi(pphi_eval):
        scale = (pphi_eval - pphi0) / torch.clamp(r_mag * sin_theta, min=1.0e-12)
        p_eval = p_vec + phi_hat * scale.unsqueeze(-1)
        L_eval = _cross3(r_vec, p_eval)
        _refresh_hcoeffs(params, L_eval, S1, S2)
        return _eob_potentials(
            r,
            pr,
            phi,
            L_eval,
            S1,
            S2,
            params,
            p_vec=p_eval,
            r_vec=r_vec,
            compute_grad_p=False,
            fd_dpvec=False,
            fd_pphi=False,
            compute_base_grad=False,
        )["H"] / eta_t

    try:
        return _gsl_deriv_central(_ham_at_pphi, pphi0, delta_pphi)
    finally:
        params.hcoeffs = hcoeffs_saved


def _delta_phase(l: int, m: int, v: torch.Tensor, params: EOBParams):
    """Aligned-spin delta_lm phase (limited set)."""
    if l == 2 and m == 2:
        chiS = params.chiS
        chiA = params.chiA
        dM = (params.mass1 - params.mass2) / (params.M)
        aDelta = chiA * dM + chiS * (1.0 - 2.0 * params.eta)
        delta22vh6 = (-4.0 * aDelta) / 3.0 + (428.0 * _PI) / 105.0
        delta22v8 = (20.0 * aDelta) / 63.0
        delta22vh9 = -2203.0 / 81.0 + (1712.0 * _PI * _PI) / 315.0
        delta22v5 = -24.0 * params.eta
        delta22v6 = 0.0
        terms = [
            (3, delta22vh6),
            (5, delta22v5),
            (6, delta22v6),
            (8, delta22v8),
            (9, delta22vh9),
        ]
        phase = torch.zeros_like(v)
        for power, coeff in terms:
            phase = phase + torch.tensor(coeff, device=v.device, dtype=v.dtype) * (v ** power)
        return phase
    return torch.zeros_like(v)


@torch.jit.script
def _orbital_basis_from_L_phi_tensor(
    phi: torch.Tensor,
    L_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the orbital triad used to reconstruct LAL Cartesian x/P (pure tensor)."""

    device, dtype = L_vec.device, L_vec.dtype

    Lhat = _unit_vector3(L_vec)

    xhat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    yhat = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    # LAL's polarphi is phiDMod + phiMod from the inertial Cartesian dynamics.
    # The matching reduced-frame reference is the inertial x-axis projected into
    # the instantaneous orbital plane.
    e1 = xhat - torch.sum(xhat * Lhat, dim=-1).unsqueeze(-1) * Lhat
    e1_fallback = yhat - torch.sum(yhat * Lhat, dim=-1).unsqueeze(-1) * Lhat
    e1_norm_raw = torch.sqrt(torch.sum(e1 * e1, dim=-1))
    e1 = torch.where((e1_norm_raw < 1.0e-14).unsqueeze(-1), e1_fallback, e1)
    e1 = _unit_vector3(e1)
    e2 = torch.cross(Lhat, e1, dim=-1)

    n_hat = torch.cos(phi).unsqueeze(-1) * e1 + torch.sin(phi).unsqueeze(-1) * e2
    n_hat = _unit_vector3(n_hat)
    lambda_hat = torch.cross(Lhat, n_hat, dim=-1)
    lambda_hat = _unit_vector3(lambda_hat)
    return n_hat, lambda_hat, Lhat


def _orbital_basis_from_L_phi(
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor | None = None,
    S2: torch.Tensor | None = None,
    params: EOBParams | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the orbital triad used to reconstruct LAL Cartesian x/P."""

    del S1, S2, params
    return _orbital_basis_from_L_phi_tensor(phi, L_vec)


def _reduced_to_cartesian(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
):
    """Map the reduced PHM state to the tortoise Cartesian variables."""

    n_hat, lambda_hat, Lhat = _orbital_basis_from_L_phi(phi, L_vec, S1, S2, params)
    r_vec = r.unsqueeze(-1) * n_hat
    p_vec = pr.unsqueeze(-1) * n_hat - torch.cross(n_hat, L_vec, dim=-1) / torch.clamp(r.unsqueeze(-1), min=1.0e-12)
    return r_vec, p_vec, n_hat, lambda_hat, Lhat


@torch.jit.script
def _tortoise_matrices(
    r_vec: torch.Tensor, csi: torch.Tensor, dcsi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return T, inv(T), and dT_ij/dX_k from Pan+2010 Eq. A3/A5."""

    rmag = _safe_norm(r_vec)
    rmag2 = rmag * rmag
    csi_fac = torch.clamp(csi, min=1.0e-15)
    matrix_shape = list(r_vec.shape[:-1]) + [3, 3]
    T = torch.zeros(matrix_shape, device=r_vec.device, dtype=r_vec.dtype)
    invT = torch.zeros(matrix_shape, device=r_vec.device, dtype=r_vec.dtype)
    for i in range(3):
        for j in range(i + 1):
            tij = (r_vec[..., i] * r_vec[..., j] / rmag2) * (csi_fac - 1.0)
            inv_tij = -((csi_fac - 1.0) / csi_fac) * (r_vec[..., i] * r_vec[..., j] / rmag2)
            if i == j:
                tij = tij + 1.0
                inv_tij = inv_tij + 1.0
            T[..., i, j] = tij
            T[..., j, i] = tij
            invT[..., i, j] = inv_tij
            invT[..., j, i] = inv_tij
    dt_shape = list(r_vec.shape[:-1]) + [3, 3, 3]
    dT = torch.zeros(dt_shape, device=r_vec.device, dtype=r_vec.dtype)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                delta_jk = 1.0 if j == k else 0.0
                delta_ik = 1.0 if i == k else 0.0
                dT[..., i, j, k] = (r_vec[..., i] * delta_jk + delta_ik * r_vec[..., j]) * (csi_fac - 1.0) / rmag2
                dT[..., i, j, k] = dT[..., i, j, k] + r_vec[..., i] * r_vec[..., j] * r_vec[..., k] / rmag2 / rmag * (
                    -2.0 / rmag * (csi_fac - 1.0) + dcsi
                )
    return T, invT, dT


def _calcomega_rdot_lal_fd(
    r_vec: torch.Tensor,
    p_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Conservative rDot used by LAL's precessing CalcOmega frame rotation."""

    step = torch.as_tensor(_LAL_PRECESSING_CALCOMEGA_STEP, device=r_vec.device, dtype=r_vec.dtype)
    hcoeffs_saved = params.hcoeffs
    r = _safe_norm(r_vec)
    n_hat = r_vec / r.unsqueeze(-1)
    pr = _dot3(p_vec, n_hat)
    phi = torch.zeros_like(r)
    L_vec = _cross3(r_vec, p_vec)
    eta_t = torch.as_tensor(params.eta, device=r_vec.device, dtype=r_vec.dtype)
    try:
        _refresh_hcoeffs(
            params,
            L_vec,
            S1,
            S2,
            S1_weighted=S1_weighted_override,
            S2_weighted=S2_weighted_override,
        )
        pot0 = _eob_potentials(
            r,
            pr,
            phi,
            L_vec,
            S1,
            S2,
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            compute_grad_p=False,
            fd_dpvec=False,
            fd_pphi=False,
            compute_base_grad=False,
            S1_weighted_override=S1_weighted_override,
            S2_weighted_override=S2_weighted_override,
        )
        grads = []
        for axis in range(3):
            p_axis0 = p_vec[..., axis]

            def _ham_at_p_axis(p_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                p_eval = p_vec.clone()
                p_eval[..., axis] = p_axis_eval
                L_eval = _cross3(r_vec, p_eval)
                # XLALSpinPrecHcapRvecDerivative swaps in a temporary coeff
                # struct, then XLALSimIMRSpinPrecEOBHamiltonian recomputes the
                # local hcoeffs for each perturbed momentum when updateHCoeffs
                # is set.  Do not freeze the entry coeffs here.
                return _eob_potentials(
                    r,
                    pr,
                    phi,
                    L_eval,
                    S1,
                    S2,
                    params,
                    p_vec=p_eval,
                    r_vec=r_vec,
                    compute_grad_p=False,
                    fd_dpvec=False,
                    fd_pphi=False,
                    compute_base_grad=False,
                    S1_weighted_override=S1_weighted_override,
                    S2_weighted_override=S2_weighted_override,
                )["H"] / eta_t

            grads.append(_gsl_deriv_central(_ham_at_p_axis, p_axis0, step))
        dH_dp = torch.stack(grads, dim=-1)
        Tmat, _, _ = _tortoise_matrices(r_vec, torch.clamp(pot0["csi"], min=1.0e-15), pot0["dcsi"])
        return _matvec3_lal_order(Tmat, dH_dp)
    finally:
        params.hcoeffs = hcoeffs_saved


@torch.jit.script
def _polar_to_cart_lal(
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    pr: torch.Tensor,
    ptheta: torch.Tensor,
    pphi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LAL's polar-to-Cartesian map from GSLSpinPrecHamiltonianWrapperFordHdpphi."""

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_safe = torch.clamp(sin_theta, min=1.0e-12)
    r_safe = torch.clamp(r, min=1.0e-12)
    rcart = torch.stack(
        [
            r * cos_theta,
            -r * sin_theta * sin_phi,
            r * sin_theta * cos_phi,
        ],
        dim=-1,
    )
    pcart = torch.stack(
        [
            pr * cos_theta - ptheta / r_safe * sin_theta,
            -pr * sin_theta * sin_phi
            - ptheta / r_safe * cos_theta * sin_phi
            - pphi / r_safe / sin_safe * cos_phi,
            pr * sin_theta * cos_phi
            + ptheta / r_safe * cos_theta * cos_phi
            - pphi / r_safe / sin_safe * sin_phi,
        ],
        dim=-1,
    )
    return rcart, pcart


@torch.jit.script
def _calcomega_polar_derivative_core(
    pphi: torch.Tensor,
    r_polar: torch.Tensor,
    theta_polar: torch.Tensor,
    phi_polar: torch.Tensor,
    ptheta_polar: torch.Tensor,
    s1_m2: torch.Tensor,
    s2_m2: torch.Tensor,
    mass1: float,
    mass2: float,
    eta: float,
    M: float,
    h_k0: torch.Tensor,
    h_k1: torch.Tensor,
    h_k2: torch.Tensor,
    h_k3: torch.Tensor,
    h_k4: torch.Tensor,
    h_k5: torch.Tensor,
    h_k5l: torch.Tensor,
    h_KK: torch.Tensor,
    h_d1: torch.Tensor,
    h_d1v2: torch.Tensor,
    h_dheffSS: torch.Tensor,
    h_dheffSSv2: torch.Tensor,
    h_b3: torch.Tensor,
    h_bb3: torch.Tensor,
) -> torch.Tensor:
    """TorchScript-compiled analytical dH/dpphi for non-Keplerian CalcOmega."""

    device = r_polar.device
    dtype = r_polar.dtype
    eta_t = torch.as_tensor(eta, device=device, dtype=dtype)
    m1 = torch.as_tensor(mass1, device=device, dtype=dtype)
    m2 = torch.as_tensor(mass2, device=device, dtype=dtype)
    mass1_norm = m1 / M
    mass2_norm = m2 / M
    norm_total_mass2 = (mass1_norm + mass2_norm) * (mass1_norm + mass2_norm)
    sigmaKerr_vec = (s1_m2 + s2_m2) / norm_total_mass2
    sigmaStar_vec = ((mass2_norm / mass1_norm) * s1_m2 + (mass1_norm / mass2_norm) * s2_m2) / norm_total_mass2
    a2 = torch.clamp(_dot3(sigmaKerr_vec, sigmaKerr_vec), min=1e-16)
    a = torch.sqrt(a2)
    a_mag = _safe_norm(sigmaKerr_vec)
    e3_hat = sigmaKerr_vec / torch.clamp(a_mag.unsqueeze(-1), min=1.0e-15)

    sin_th = torch.sin(theta_polar)
    cos_th = torch.cos(theta_polar)
    sin_ph = torch.sin(phi_polar)
    cos_ph = torch.cos(phi_polar)
    sin_s = torch.clamp(sin_th, min=1.0e-12)
    r_s = torch.clamp(r_polar, min=1.0e-12)

    rcart = torch.stack([
        r_polar * cos_th,
        -r_polar * sin_th * sin_ph,
        r_polar * sin_th * cos_ph,
    ], dim=-1)

    pcart = torch.stack([
        -ptheta_polar / r_s * sin_th,
        -ptheta_polar / r_s * cos_th * sin_ph - pphi / r_s / sin_s * cos_ph,
        ptheta_polar / r_s * cos_th * cos_ph - pphi / r_s / sin_s * sin_ph,
    ], dim=-1)

    u_pphi = torch.stack([
        torch.zeros_like(r_polar),
        -cos_ph / (r_s * sin_s),
        -sin_ph / (r_s * sin_s),
    ], dim=-1)

    L_eval = _cross3(rcart, pcart)
    n_hat = rcart / torch.clamp(_safe_norm(rcart).unsqueeze(-1), min=1.0e-15)
    lambda_hat = _cross3(L_eval / torch.clamp(_safe_norm(L_eval).unsqueeze(-1), min=1.0e-15), n_hat)
    lambda_hat = lambda_hat / torch.clamp(_safe_norm(lambda_hat).unsqueeze(-1), min=1.0e-15)
    costheta = _dot3(e3_hat, n_hat)
    xi_vec = _cross3(e3_hat, n_hat)
    xi2 = 1.0 - costheta * costheta
    mask_aligned = (1.0 - torch.abs(costheta)) <= 1.0e-8
    if mask_aligned.any():
        angle = torch.as_tensor(1.8e-3, device=device, dtype=dtype)
        kcrossv = _cross3(lambda_hat, e3_hat)
        kdotv = _dot3(lambda_hat, e3_hat)
        e3_rot = (
            e3_hat * torch.cos(angle)
            + kcrossv * torch.sin(angle)
            + lambda_hat * kdotv.unsqueeze(-1) * (1.0 - torch.cos(angle))
        )
        e3_hat = torch.where(mask_aligned.unsqueeze(-1), e3_rot, e3_hat)
        xi_vec = _cross3(e3_hat, n_hat)
        costheta = _dot3(e3_hat, n_hat)
        xi2 = 1.0 - costheta * costheta
    v_vec = _cross3(n_hat, xi_vec)

    r_var = r_polar
    u_b = torch.clamp(1.0 / torch.clamp(r_var, min=1e-9), min=1e-9)
    u2_b = u_b * u_b
    u3_b = u2_b * u_b
    u4_b = u2_b * u2_b
    u5_b = u4_b * u_b
    invlog_2e = torch.as_tensor(0.69314718055994530941723212145817656807550013436026, device=device, dtype=dtype)
    logu_b = torch.log2(u_b) * invlog_2e

    denom_KK = -1.0 + eta_t * h_KK
    invm1PlusEtaKK = 1.0 / torch.where(torch.abs(denom_KK) < 1.0e-14, torch.sign(denom_KK) * 1.0e-14, denom_KK)
    logarg_b = h_k1 * u_b + h_k2 * u2_b + h_k3 * u3_b + h_k4 * u4_b + h_k5 * u5_b + h_k5l * u5_b * logu_b
    logTerms_b = 1.0 + eta_t * h_k0 + eta_t * torch.log1p(torch.abs(1.0 + logarg_b) - 1.0)
    bulk_b = invm1PlusEtaKK * (invm1PlusEtaKK + 2.0 * u_b) + a2 * u2_b
    deltaU_b = torch.abs(bulk_b * logTerms_b)
    r2_b = r_var * r_var
    deltaT_b = r2_b * deltaU_b
    deltaU_u_b = 2.0 * (invm1PlusEtaKK + a2 * u_b) * logTerms_b + bulk_b * (eta_t * (h_k1 + u_b * (2.0 * h_k2 + u_b * (3.0 * h_k3 + u_b * (4.0 * h_k4 + 5.0 * (h_k5 + h_k5l * logu_b) * u_b))))) / (1.0 + logarg_b)
    deltaT_r_b = 2.0 * r_var * deltaU_b - deltaU_u_b
    D_b_arg = 6.0 * eta_t * u2_b + 2.0 * (26.0 - 3.0 * eta_t) * eta_t * u3_b
    D_b = 1.0 + torch.log1p(D_b_arg)
    deltaR_b = deltaT_b * D_b
    w2_b = r2_b + a2
    rho2_b = r2_b + a2 * costheta * costheta
    Lambda_b = torch.abs(w2_b * w2_b - a2 * deltaT_b * xi2)
    invrho2xi2Lambda_b = 1.0 / (rho2_b * xi2 * Lambda_b)
    invrho2 = xi2 * (Lambda_b * invrho2xi2Lambda_b)
    invxi2 = rho2_b * (Lambda_b * invrho2xi2Lambda_b)
    invLambda_b = xi2 * rho2_b * invrho2xi2Lambda_b

    # Momenta and directional projections (note: pr = 0 in polar frame)
    tmpP = pcart
    pn = _dot3(tmpP, n_hat)
    pvr = _dot3(tmpP, v_vec) * r_var
    dpvr_dpphi = _dot3(u_pphi, v_vec) * r_var
    pf = _dot3(tmpP, xi_vec) * r_var
    dpf_dpphi = _dot3(u_pphi, xi_vec) * r_var

    ww = 2.0 * a * r_var + h_b3 * eta_t * a2 * a * u_b + h_bb3 * eta_t * a * u_b

    pf2 = pf * pf
    Q = 1.0 + pvr * pvr * invrho2 * invxi2 + pf2 * rho2_b * invLambda_b * invxi2
    dQ_dpphi = 2.0 * pvr * dpvr_dpphi * invrho2 * invxi2 + 2.0 * pf * dpf_dpphi * rho2_b * invLambda_b * invxi2
    pp = Q - 1.0
    dpp_dpphi = dQ_dpphi

    # Hns and derivative
    expnu2 = (rho2_b * deltaT_b) * invLambda_b
    expnu = torch.sqrt(torch.clamp(expnu2, min=1.0e-16))
    sqrtQ = torch.sqrt(torch.clamp(Q, min=1.0e-16))
    Hns = sqrtQ * expnu + pf * ww * invLambda_b
    dHns_dpphi = (0.5 * expnu / sqrtQ) * dQ_dpphi + dpf_dpphi * ww * invLambda_b

    # deltaSigmaStar and derivative w.r.t pp
    d_sM1_dpp = (206.0 * r_var + 46.0 * pp * r_var * r_var + (-120.0 * r_var + 6.0 * pp * r_var * r_var) * eta_t) * eta_t * u2_b * (-1.0 / 72.0)
    d_sM2_dpp = (-109.0 / 36.0 * u2_b * r_var - 10.0 / 16.0 * pp * u2_b * r_var * r_var + 17.0 / 12.0 * u2_b * r_var * eta_t) * eta_t

    sMultiplier1 = (
        -706.0
        + (206.0 * pp + 23.0 * pp * pp * r_var) * r_var
        + (54.0 + (-120.0 * pp + 3.0 * pp * pp * r_var) * r_var) * eta_t
    ) * eta_t * u2_b * (-1.0 / 72.0)

    sMultiplier2 = (
        -56.0 / 9.0 * u2_b
        + (-109.0 / 36.0 * pp * u2_b - 5.0 / 16.0 * pp * pp * u2_b * r_var) * r_var
        + (-7.0 / 3.0 * u2_b + 17.0 / 12.0 * pp * u2_b * r_var) * eta_t
    ) * eta_t

    d1_vec = h_d1.unsqueeze(-1) if h_d1.ndim > 0 else h_d1
    d1v2_vec = h_d1v2.unsqueeze(-1) if h_d1v2.ndim > 0 else h_d1v2

    deltaSigmaStar = eta_t.unsqueeze(-1) * (
        (-8.0 + 3.0 * r_var * pp).unsqueeze(-1) * sigmaKerr_vec
        + (14.0 + 4.0 * pp * r_var).unsqueeze(-1) * sigmaStar_vec
    ) * (1.0 / 12.0) * u_b.unsqueeze(-1)
    deltaSigmaStar = deltaSigmaStar + sMultiplier1.unsqueeze(-1) * sigmaStar_vec + sMultiplier2.unsqueeze(-1) * sigmaKerr_vec
    deltaSigmaStar = deltaSigmaStar + d1_vec * eta_t.unsqueeze(-1) * sigmaStar_vec * u3_b.unsqueeze(-1)
    deltaSigmaStar = deltaSigmaStar + d1v2_vec * eta_t.unsqueeze(-1) * sigmaKerr_vec * u3_b.unsqueeze(-1)

    d_deltaSigmaStar_dpp = (
        eta_t.unsqueeze(-1) * (3.0 * r_var.unsqueeze(-1) * sigmaKerr_vec + 4.0 * r_var.unsqueeze(-1) * sigmaStar_vec) * (1.0 / 12.0) * u_b.unsqueeze(-1)
        + d_sM1_dpp.unsqueeze(-1) * sigmaStar_vec
        + d_sM2_dpp.unsqueeze(-1) * sigmaKerr_vec
    )

    s_vec = sigmaStar_vec + deltaSigmaStar
    ds_vec_dpphi = d_deltaSigmaStar_dpp * dpp_dpphi.unsqueeze(-1)

    sxi = _dot3(s_vec, xi_vec)
    sv = _dot3(s_vec, v_vec)
    sn = _dot3(s_vec, n_hat)
    s3 = _dot3(s_vec, e3_hat)

    dsxi_dpphi = _dot3(ds_vec_dpphi, xi_vec)
    dsv_dpphi = _dot3(ds_vec_dpphi, v_vec)
    dsn_dpphi = _dot3(ds_vec_dpphi, n_hat)
    ds3_dpphi = _dot3(ds_vec_dpphi, e3_hat)

    sqrtdeltaT = torch.sqrt(torch.clamp(deltaT_b, min=1.0e-16))
    sqrtdeltaR = torch.sqrt(torch.clamp(deltaR_b, min=1.0e-16))
    invsqrtdeltaT = 1.0 / sqrtdeltaT
    invsqrtdeltaR = 1.0 / sqrtdeltaR
    invdeltaT = 1.0 / deltaT_b

    w = ww * invLambda_b
    expMU = torch.sqrt(torch.clamp(rho2_b, min=1.0e-16))
    invexpnu = 1.0 / expnu
    invexpMU = 1.0 / expMU

    Lambda_r = 4.0 * r_var * w2_b - a2 * deltaT_r_b * xi2
    ww_r = 2.0 * a - (a2 * a * h_b3 * eta_t) * u2_b - h_bb3 * eta_t * a * u2_b
    BR = (-deltaT_b * invsqrtdeltaR + deltaT_r_b * 0.5) * invsqrtdeltaT
    wr = (-Lambda_r * ww + Lambda_b * ww_r) * (invLambda_b * invLambda_b)
    nur = (r_var * invrho2 + (w2_b * (-4.0 * r_var * deltaT_b + w2_b * deltaT_r_b)) * 0.5 * invdeltaT * invLambda_b)
    mur = (r_var * invrho2 - invsqrtdeltaR)
    wcos = -2.0 * (a2 * costheta) * deltaT_b * ww * (invLambda_b * invLambda_b)
    nucos = (a2 * costheta) * w2_b * (w2_b - deltaT_b) * (invrho2 * invLambda_b)
    mucos = (a2 * costheta) * invrho2

    # Hs components
    dHs_dpphi = w * ds3_dpphi

    denom_Q = 2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ * xi2
    d_denom_Q_dpphi = 2.0 * sqrtdeltaT * xi2 * (1.0 + 2.0 * sqrtQ) / (2.0 * sqrtQ) * dQ_dpphi

    term1 = (expMU * expMU) * (expnu * expnu) * (pf * pf) * sv
    d_term1 = (expMU * expMU) * (expnu * expnu) * (2.0 * pf * dpf_dpphi * sv + pf * pf * dsv_dpphi)

    term2 = sqrtdeltaT * (expMU * expnu) * pf * pvr * sxi
    d_term2 = sqrtdeltaT * (expMU * expnu) * (dpf_dpphi * pvr * sxi + pf * dpvr_dpphi * sxi + pf * pvr * dsxi_dpphi)

    term3 = (sqrtdeltaT * sqrtdeltaT) * xi2 * (expMU * expMU) * (sqrtQ + Q) * sv
    d_sqrtQ_plus_Q = (1.0 / (2.0 * sqrtQ) + 1.0) * dQ_dpphi
    d_term3 = (sqrtdeltaT * sqrtdeltaT) * xi2 * (expMU * expMU) * (d_sqrtQ_plus_Q * sv + (sqrtQ + Q) * dsv_dpphi)

    Hwr_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sqrtdeltaR * (term1 - term2 + term3)
    d_Hwr_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sqrtdeltaR * (d_term1 - d_term2 + d_term3)
    dHwr_dpphi = (d_Hwr_num * denom_Q - Hwr_num * d_denom_Q_dpphi) / (denom_Q * denom_Q)
    dHs_dpphi = dHs_dpphi + dHwr_dpphi * wr

    denom_wcos = 2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ
    d_denom_wcos_dpphi = 2.0 * sqrtdeltaT * (1.0 + 2.0 * sqrtQ) / (2.0 * sqrtQ) * dQ_dpphi

    wcos_inner = -(expMU * expMU) * (expnu * expnu) * (pf * pf) + (sqrtdeltaT * sqrtdeltaT) * (pvr * pvr - (expMU * expMU) * (sqrtQ + Q) * xi2)
    d_wcos_inner = -(expMU * expMU) * (expnu * expnu) * (2.0 * pf * dpf_dpphi) + (sqrtdeltaT * sqrtdeltaT) * (2.0 * pvr * dpvr_dpphi - (expMU * expMU) * d_sqrtQ_plus_Q * xi2)
    Hwcos_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sn * wcos_inner
    d_Hwcos_num = (invexpMU * invexpMU * invexpMU * invexpnu) * (dsn_dpphi * wcos_inner + sn * d_wcos_inner)
    dHwcos_dpphi = (d_Hwcos_num * denom_wcos - Hwcos_num * d_denom_wcos_dpphi) / (denom_wcos * denom_wcos)
    dHs_dpphi = dHs_dpphi + dHwcos_dpphi * wcos

    fac_SOL = (expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) / (deltaT_b * xi2)
    HSOL_num = pf * s3
    d_HSOL_num = dpf_dpphi * s3 + pf * ds3_dpphi
    dHSOL_dpphi = fac_SOL * (d_HSOL_num * sqrtQ - HSOL_num * (0.5 / sqrtQ * dQ_dpphi)) / Q
    dHs_dpphi = dHs_dpphi + dHSOL_dpphi

    denom_SONL = deltaT_b * (sqrtQ + Q) * xi2
    d_denom_SONL = deltaT_b * xi2 * d_sqrtQ_plus_Q

    p_SONL_1 = -(sqrtdeltaT * expMU * expnu * nucos * xi2) * pf * (1.0 + 2.0 * sqrtQ) * sn
    d_p_SONL_1 = -(sqrtdeltaT * expMU * expnu * nucos * xi2) * (
        (dpf_dpphi * (1.0 + 2.0 * sqrtQ) + pf * (1.0 / sqrtQ * dQ_dpphi)) * sn
        + pf * (1.0 + 2.0 * sqrtQ) * dsn_dpphi
    )

    p_SONL_2a = -(BR * expMU * expnu) * pf * (1.0 + sqrtQ) * sv
    d_p_SONL_2a = -(BR * expMU * expnu) * (
        (dpf_dpphi * (1.0 + sqrtQ) + pf * (0.5 / sqrtQ * dQ_dpphi)) * sv
        + pf * (1.0 + sqrtQ) * dsv_dpphi
    )

    p_SONL_2b1 = (expMU * expnu * nur) * pf * (1.0 + 2.0 * sqrtQ) * sv
    d_p_SONL_2b1 = (expMU * expnu * nur) * (
        (dpf_dpphi * (1.0 + 2.0 * sqrtQ) + pf * (1.0 / sqrtQ * dQ_dpphi)) * sv
        + pf * (1.0 + 2.0 * sqrtQ) * dsv_dpphi
    )

    p_SONL_2b2 = sqrtdeltaT * mur * pvr * sxi
    d_p_SONL_2b2 = sqrtdeltaT * mur * (dpvr_dpphi * sxi + pvr * dsxi_dpphi)

    p_SONL_2b3 = sqrtdeltaT * sxi * sqrtQ * (mur - nur) * pvr
    d_p_SONL_2b3 = sqrtdeltaT * (mur - nur) * (
        (dsxi_dpphi * sqrtQ + sxi * (0.5 / sqrtQ * dQ_dpphi)) * pvr
        + sxi * sqrtQ * dpvr_dpphi
    )

    p_SONL_2 = (p_SONL_2a + sqrtdeltaT * (p_SONL_2b1 + p_SONL_2b2 + p_SONL_2b3)) * sqrtdeltaR
    d_p_SONL_2 = (d_p_SONL_2a + sqrtdeltaT * (d_p_SONL_2b1 + d_p_SONL_2b2 + d_p_SONL_2b3)) * sqrtdeltaR

    HSONL_bracket = p_SONL_1 + p_SONL_2
    d_HSONL_bracket = d_p_SONL_1 + d_p_SONL_2

    fac_SONL = expnu * (invexpMU * invexpMU)
    HSONL_num = fac_SONL * HSONL_bracket
    d_HSONL_num = fac_SONL * d_HSONL_bracket
    dHSONL_dpphi = (d_HSONL_num * denom_SONL - HSONL_num * d_denom_SONL) / (denom_SONL * denom_SONL)
    dHs_dpphi = dHs_dpphi + dHSONL_dpphi

    dHss_dpphi = -0.5 * u3_b * (2.0 * _dot3(s_vec, ds_vec_dpphi) - 6.0 * sn * dsn_dpphi)

    dHeff_dpphi = dHns_dpphi + dHs_dpphi + dHss_dpphi

    Hs_val = w * s3 + (Hwr_num / denom_Q) * wr + (Hwcos_num / denom_wcos) * wcos + (HSOL_num / (deltaT_b * sqrtQ * xi2)) * (expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) + (HSONL_num / denom_SONL)
    Hss_val = -0.5 * u3_b * (_dot3(s_vec, s_vec) - 3.0 * sn * sn)
    Heff_val = Hns + Hs_val + Hss_val + h_dheffSS * eta_t * _dot3(sigmaKerr_vec, sigmaStar_vec) * u4_b + h_dheffSSv2 * eta_t * u4_b * (
        s1_m2[..., 0] * s1_m2[..., 0] + s1_m2[..., 1] * s1_m2[..., 1] + s1_m2[..., 2] * s1_m2[..., 2] + s2_m2[..., 0] * s2_m2[..., 0] + s2_m2[..., 1] * s2_m2[..., 1] + s2_m2[..., 2] * s2_m2[..., 2]
    )
    Hreal_arg = 1.0 + 2.0 * eta_t * (torch.abs(Heff_val) - 1.0)
    Hreal = torch.sqrt(torch.clamp(Hreal_arg, min=1.0e-16))

    return dHeff_dpphi / Hreal


def _calcomega_lal_polar_derivative(
    r_vec: torch.Tensor,
    p_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    rdot_vec: torch.Tensor | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Port of XLALSimIMRSpinPrecEOBCalcOmega for the non-Keplerian flux factor."""

    if (
        r_vec.device.type == "cpu"
        and r_vec.dtype == torch.float64
        and (not torch.is_grad_enabled() or not r_vec.requires_grad)
        and not _env_on("PYCBC_SEOBNRV4PHM_FD_CALCOMEGA", False)
        and os.environ.get("PYCBC_SEOBNR_NATIVE_ODE", "1") not in ("0", "", "false", "False")
    ):
        try:
            from pycbc.waveform._seobnr_native_ode import get_extension
            ext = get_extension()
            if ext is not None and hasattr(ext, "calcomega_lal_polar_derivative_native"):
                return ext.calcomega_lal_polar_derivative_native(
                    r_vec.contiguous(),
                    p_vec.contiguous(),
                    S1.contiguous(),
                    S2.contiguous(),
                    float(params.mass1),
                    float(params.mass2),
                    None if rdot_vec is None else rdot_vec.contiguous(),
                    None if S1_weighted_override is None else S1_weighted_override.contiguous(),
                    None if S2_weighted_override is None else S2_weighted_override.contiguous(),
                )
        except Exception:
            pass

    hcoeffs_saved = params.hcoeffs
    device, dtype = r_vec.device, r_vec.dtype
    xhat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    step = torch.as_tensor(_LAL_PRECESSING_CALCOMEGA_STEP, device=device, dtype=dtype)
    eta_t = torch.as_tensor(params.eta, device=device, dtype=dtype)
    try:
        rdot = (
            rdot_vec
            if rdot_vec is not None
            else _calcomega_rdot_lal_fd(
                r_vec,
                p_vec,
                S1,
                S2,
                params,
                S1_weighted_override=S1_weighted_override,
                S2_weighted_override=S2_weighted_override,
            )
        )
        LNhat = _cross3(r_vec, rdot)
        LNhat = LNhat / torch.clamp(_safe_norm(LNhat).unsqueeze(-1), min=1.0e-15)

        use_rot1 = _dot3(LNhat, xhat) >= 0.9
        invsqrt2 = torch.as_tensor(1.0 / math.sqrt(2.0), device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)
        Rot1_alt = torch.stack(
            [
                torch.stack([invsqrt2, -invsqrt2, zero]),
                torch.stack([invsqrt2, invsqrt2, zero]),
                torch.stack([zero, zero, torch.ones((), device=device, dtype=dtype)]),
            ]
        )
        Rot1_eye = torch.eye(3, device=device, dtype=dtype)
        Rot1 = torch.where(use_rot1[..., None, None], Rot1_alt, Rot1_eye)
        Xprime_alt = _matvec3_lal_order(Rot1_alt, LNhat)
        Xprime = torch.where(use_rot1[..., None], Xprime_alt, LNhat)

        Yprime = _cross3(Xprime, xhat)
        Yprime = Yprime / torch.clamp(_safe_norm(Yprime).unsqueeze(-1), min=1.0e-15)
        Zprime = _cross3(Xprime, Yprime)
        Zprime = Zprime / torch.clamp(_safe_norm(Zprime).unsqueeze(-1), min=1.0e-15)
        Rot2 = torch.stack([Xprime, Yprime, Zprime], dim=-2)

        def _rotate(vec: torch.Tensor) -> torch.Tensor:
            return _matvec3_lal_order(Rot2, _matvec3_lal_order(Rot1, vec))

        r_prime = _rotate(r_vec)
        p_prime = _rotate(p_vec)
        S1_prime = _rotate(S1)
        S2_prime = _rotate(S2)
        S1_weighted_prime = (
            None if S1_weighted_override is None else _rotate(S1_weighted_override)
        )
        S2_weighted_prime = (
            None if S2_weighted_override is None else _rotate(S2_weighted_override)
        )

        r_polar = _safe_norm(r_prime)
        theta_polar = torch.acos(torch.clamp(r_prime[..., 0] / r_polar, min=-1.0, max=1.0))
        phi_polar = torch.atan2(-r_prime[..., 1], r_prime[..., 2])
        pr_polar = torch.zeros_like(r_polar)
        r_cross_x = _cross3(r_prime, xhat)
        r_cross_x_cross_r = _cross3(r_cross_x, r_prime)
        sin_theta = torch.clamp(torch.sin(theta_polar), min=1.0e-12)
        ptheta_polar = -_dot3(r_cross_x_cross_r, p_prime) / r_polar / sin_theta
        pphi_polar = -_dot3(r_cross_x, p_prime)

        if not _env_on("PYCBC_SEOBNRV4PHM_FD_CALCOMEGA", False):
            s1_m2_prime = (
                S1_weighted_prime
                if S1_weighted_prime is not None
                else _lal_spin_weight(S1_prime, params.mass1, params.M)
            )
            s2_m2_prime = (
                S2_weighted_prime
                if S2_weighted_prime is not None
                else _lal_spin_weight(S2_prime, params.mass2, params.M)
            )
            r_eval, p_eval = _polar_to_cart_lal(
                r_polar,
                theta_polar,
                phi_polar,
                pr_polar,
                ptheta_polar,
                pphi_polar,
            )
            L_eval = _cross3(r_eval, p_eval)
            _refresh_hcoeffs(
                params,
                L_eval,
                S1_prime,
                S2_prime,
                S1_weighted=S1_weighted_prime,
                S2_weighted=S2_weighted_prime,
            )
            h = {k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in params.hcoeffs.items()}
            zero_tensor = torch.zeros((), device=device, dtype=dtype)
            return _calcomega_polar_derivative_core(
                pphi_polar,
                r_polar,
                theta_polar,
                phi_polar,
                ptheta_polar,
                s1_m2_prime,
                s2_m2_prime,
                params.mass1,
                params.mass2,
                params.eta,
                params.M,
                h.get("k0", zero_tensor),
                h.get("k1", zero_tensor),
                h.get("k2", zero_tensor),
                h.get("k3", zero_tensor),
                h.get("k4", zero_tensor),
                h.get("k5", zero_tensor),
                h.get("k5l", zero_tensor),
                h.get("KK", zero_tensor),
                h.get("d1", zero_tensor),
                h.get("d1v2", zero_tensor),
                h.get("dheffSS", zero_tensor),
                h.get("dheffSSv2", zero_tensor),
                h.get("b3", zero_tensor),
                h.get("bb3", zero_tensor),
            )

        def _ham_at_pphi(pphi_eval: torch.Tensor) -> torch.Tensor:
            r_eval, p_eval = _polar_to_cart_lal(
                r_polar,
                theta_polar,
                phi_polar,
                pr_polar,
                ptheta_polar,
                pphi_eval,
            )
            n_eval = r_eval / _safe_norm(r_eval).unsqueeze(-1)
            pr_eval = _dot3(p_eval, n_eval)
            L_eval = _cross3(r_eval, p_eval)
            _refresh_hcoeffs(
                params,
                L_eval,
                S1_prime,
                S2_prime,
                S1_weighted=S1_weighted_prime,
                S2_weighted=S2_weighted_prime,
            )
            return _eob_potentials(
                r_polar,
                pr_eval,
                phi_polar,
                L_eval,
                S1_prime,
                S2_prime,
                params,
                p_vec=p_eval,
                r_vec=r_eval,
                compute_grad_p=False,
                fd_dpvec=False,
                fd_pphi=False,
                compute_base_grad=False,
                S1_weighted_override=S1_weighted_prime,
                S2_weighted_override=S2_weighted_prime,
            )["H"] / eta_t

        return _gsl_deriv_central(_ham_at_pphi, pphi_polar, step)
    finally:
        params.hcoeffs = hcoeffs_saved


def _hamiltonian_spin_rates_to_chi(
    params: EOBParams,
    dS1dt: torch.Tensor,
    dS2dt: torch.Tensor,
):
    """Convert Hamiltonian spin rates from mass-weighted spins to chi rates."""

    scale1 = torch.as_tensor(
        (params.M / params.mass1) ** 2, device=dS1dt.device, dtype=dS1dt.dtype
    )
    scale2 = torch.as_tensor(
        (params.M / params.mass2) ** 2, device=dS2dt.device, dtype=dS2dt.dtype
    )
    return dS1dt * scale1, dS2dt * scale2


def _cartesian_hamiltonian_over_eta(
    r_vec: torch.Tensor,
    p_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    p_is_tortoise: bool,
    tortoise: int | None = None,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Hamiltonian snapshot used by optional Cartesian finite differences."""

    old_tortoise = getattr(params, "tortoise", 2)
    if tortoise is not None:
        params.tortoise = tortoise
    try:
        r = _safe_norm(r_vec)
        n_hat = r_vec / r
        pr = _dot3(p_vec, n_hat)
        phi = torch.zeros_like(r)
        L_vec = _cross3(r_vec, p_vec)
        _refresh_hcoeffs(
            params,
            L_vec,
            S1,
            S2,
            S1_weighted=S1_weighted_override,
            S2_weighted=S2_weighted_override,
        )
        pot = _eob_potentials(
            r,
            pr,
            phi,
            L_vec,
            S1,
            S2,
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            compute_grad_p=False,
            compute_grad_spin=False,
            p_is_tortoise=p_is_tortoise,
            fd_dpvec=False,
            fd_pphi=False,
            compute_base_grad=False,
            S1_weighted_override=S1_weighted_override,
            S2_weighted_override=S2_weighted_override,
        )
        return pot["H"] / torch.as_tensor(params.eta, device=r_vec.device, dtype=r_vec.dtype)
    finally:
        if tortoise is not None:
            params.tortoise = old_tortoise


def _dH_dx_cartesian_fd(
    r_vec: torch.Tensor,
    p_non_tortoise: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Finite-difference d(H/eta)/dX at fixed non-tortoise P."""

    # LAL's v4P integrator RHS uses XLALSpinPrecHcapNumericalDerivative with
    # STEP_SIZE = 2.0e-3 for Cartesian x/P/spin derivatives.
    step = torch.as_tensor(_LAL_V4P_NUMERICAL_DERIVATIVE_STEP, device=r_vec.device, dtype=r_vec.dtype)
    grads = []
    hcoeffs_saved = params.hcoeffs
    for axis in range(3):
        r_axis0 = r_vec[axis]

        def _ham_at_x_axis(r_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
            r_eval = r_vec.clone()
            r_eval[axis] = r_axis_eval
            return _cartesian_hamiltonian_over_eta(
                r_eval,
                p_non_tortoise,
                S1,
                S2,
                params,
                p_is_tortoise=False,
                tortoise=2,
                S1_weighted_override=S1_weighted_override,
                S2_weighted_override=S2_weighted_override,
            )

        grads.append(_gsl_deriv_central(_ham_at_x_axis, r_axis0, step))
    params.hcoeffs = hcoeffs_saved
    return torch.stack(grads)


def _dH_dx_cartesian_autograd(
    r_vec: torch.Tensor,
    p_non_tortoise: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
):
    """Direct/analytical port of LAL's exact fixed-P Cartesian d(H/eta)/dX block."""
    return _dH_dx_cartesian_fd(r_vec, p_non_tortoise, S1, S2, params)


def _dH_dspin_cartesian_fd(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    L_vec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: EOBParams,
    *,
    p_vec: torch.Tensor,
    r_vec: torch.Tensor,
    weighted_spins: bool = False,
    S1_weighted_override: torch.Tensor | None = None,
    S2_weighted_override: torch.Tensor | None = None,
):
    """Finite-difference d(H/eta)/dspin with hcoeff refresh, matching LAL GSL."""

    step_base = torch.as_tensor(_LAL_V4P_NUMERICAL_DERIVATIVE_STEP, device=S1.device, dtype=S1.dtype)
    mass1_norm = torch.as_tensor(params.mass1 / params.M, device=S1.device, dtype=S1.dtype)
    mass2_norm = torch.as_tensor(params.mass2 / params.M, device=S2.device, dtype=S2.dtype)
    s1_scale = torch.as_tensor(_lal_spin_scale(params.mass1, params.M), device=S1.device, dtype=S1.dtype)
    s2_scale = torch.as_tensor(_lal_spin_scale(params.mass2, params.M), device=S2.device, dtype=S2.dtype)
    # LAL differentiates the evolved weighted spins S_i/M^2 with a step of
    # STEP_SIZE * (m_i/M)^2.  The legacy reduced-state path differentiates chi_i
    # directly, where the equivalent chi step is just STEP_SIZE.
    step1 = (step_base * mass1_norm) * mass1_norm if weighted_spins else step_base
    step2 = (step_base * mass2_norm) * mass2_norm if weighted_spins else step_base
    S1_weighted = (
        _lal_spin_weight(S1, params.mass1, params.M)
        if S1_weighted_override is None
        else S1_weighted_override.to(device=S1.device, dtype=S1.dtype)
    )
    S2_weighted = (
        _lal_spin_weight(S2, params.mass2, params.M)
        if S2_weighted_override is None
        else S2_weighted_override.to(device=S2.device, dtype=S2.dtype)
    )
    eta_t = torch.as_tensor(params.eta, device=S1.device, dtype=S1.dtype)
    hcoeffs_saved = params.hcoeffs

    def _ham_over_eta(
        s1_eval: torch.Tensor,
        s2_eval: torch.Tensor,
        *,
        s1_weighted_eval: torch.Tensor | None = None,
        s2_weighted_eval: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _refresh_hcoeffs(
            params,
            L_vec,
            s1_eval,
            s2_eval,
            S1_weighted=s1_weighted_eval,
            S2_weighted=s2_weighted_eval,
        )
        return (
            _eob_potentials(
                r,
                pr,
                phi,
                L_vec,
                s1_eval,
                s2_eval,
                params,
                p_vec=p_vec,
                r_vec=r_vec,
                compute_grad_p=False,
                compute_grad_spin=False,
                fd_dpvec=False,
                fd_pphi=False,
                compute_base_grad=False,
                S1_weighted_override=s1_weighted_eval,
                S2_weighted_override=s2_weighted_eval,
            )["H"]
            / eta_t
        )

    try:
        dS1 = []
        for axis in range(3):
            if weighted_spins:
                s_axis0 = S1_weighted[axis]

                def _ham_at_s1_axis(s_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                    s1_weighted_eval = S1_weighted.clone()
                    s1_weighted_eval[axis] = s_axis_eval
                    s1_eval = s1_weighted_eval / torch.clamp(s1_scale, min=1.0e-15)
                    return _ham_over_eta(
                        s1_eval,
                        S2,
                        s1_weighted_eval=s1_weighted_eval,
                        s2_weighted_eval=S2_weighted,
                    )
            else:
                s_axis0 = S1[axis]

                def _ham_at_s1_axis(s_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                    s1_eval = S1.clone()
                    s1_eval[axis] = s_axis_eval
                    return _ham_over_eta(s1_eval, S2)

            dS1.append(_gsl_deriv_central(_ham_at_s1_axis, s_axis0, step1))

        dS2 = []
        for axis in range(3):
            if weighted_spins:
                s_axis0 = S2_weighted[axis]

                def _ham_at_s2_axis(s_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                    s2_weighted_eval = S2_weighted.clone()
                    s2_weighted_eval[axis] = s_axis_eval
                    s2_eval = s2_weighted_eval / torch.clamp(s2_scale, min=1.0e-15)
                    return _ham_over_eta(
                        S1,
                        s2_eval,
                        s1_weighted_eval=S1_weighted,
                        s2_weighted_eval=s2_weighted_eval,
                    )
            else:
                s_axis0 = S2[axis]

                def _ham_at_s2_axis(s_axis_eval: torch.Tensor, axis: int = axis) -> torch.Tensor:
                    s2_eval = S2.clone()
                    s2_eval[axis] = s_axis_eval
                    return _ham_over_eta(S1, s2_eval)

            dS2.append(_gsl_deriv_central(_ham_at_s2_axis, s_axis0, step2))

        return torch.stack(dS1), torch.stack(dS2)
    finally:
        params.hcoeffs = hcoeffs_saved


def reduced_state_to_cartesian_state(y: torch.Tensor, params: EOBParams):
    """Convert reduced PHM state to an internal Cartesian trajectory state."""

    r = y[0]
    pr = y[1]
    phi = y[2]
    L_vec = y[3:6]
    S1 = y[6:9]
    S2 = y[9:12]
    r_vec, p_vec, _, _, _ = _reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)
    S1_weighted = _lal_spin_weight(S1, params.mass1, params.M)
    S2_weighted = _lal_spin_weight(S2, params.mass2, params.M)
    zeta = torch.zeros_like(phi)
    return torch.cat([r_vec, p_vec, S1_weighted, S2_weighted, phi.view(1), zeta.view(1)])


def cartesian_state_to_reduced_state(y: torch.Tensor, params: EOBParams | None = None):
    """Project internal Cartesian trajectory state back to the reduced PHM state."""

    r_vec = y[0:3]
    p_vec = y[3:6]
    if y.numel() >= 14:
        S1 = y[6:9]
        S2 = y[9:12]
        if params is not None:
            s1_scale = torch.as_tensor(_lal_spin_scale(params.mass1, params.M), device=y.device, dtype=y.dtype)
            s2_scale = torch.as_tensor(_lal_spin_scale(params.mass2, params.M), device=y.device, dtype=y.dtype)
            S1 = S1 / torch.clamp(s1_scale, min=1.0e-15)
            S2 = S2 / torch.clamp(s2_scale, min=1.0e-15)
        phi = y[12] + y[13]
    else:
        phi = y[6]
        S1 = y[7:10]
        S2 = y[10:13]
    r = _safe_norm(r_vec)
    n_hat = r_vec / r
    pr = _dot3(p_vec, n_hat)
    L_vec = _cross3(r_vec, p_vec)
    return torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])


def rhs_cartesian_full(t: torch.Tensor, y: torch.Tensor, params: EOBParams):
    """Generic-spin Cartesian RHS using x/P/S as the integrated state."""

    r_vec = y[0:3]
    p_vec = y[3:6]
    lal_phase_state = y.numel() >= 14
    if lal_phase_state:
        s1_scale = torch.as_tensor(_lal_spin_scale(params.mass1, params.M), device=y.device, dtype=y.dtype)
        s2_scale = torch.as_tensor(_lal_spin_scale(params.mass2, params.M), device=y.device, dtype=y.dtype)
        S1_weighted = y[6:9]
        S2_weighted = y[9:12]
        S1 = S1_weighted / torch.clamp(s1_scale, min=1.0e-15)
        S2 = S2_weighted / torch.clamp(s2_scale, min=1.0e-15)
        phi = y[12]
    else:
        phi = y[6]
        S1 = y[7:10]
        S2 = y[10:13]
        S1_weighted = _lal_spin_weight(S1, params.mass1, params.M)
        S2_weighted = _lal_spin_weight(S2, params.mass2, params.M)

    eta = torch.as_tensor(params.eta, device=y.device, dtype=y.dtype)
    r = _safe_norm(r_vec)
    n_hat = r_vec / r
    pr = _dot3(p_vec, n_hat)
    L_vec = _cross3(r_vec, p_vec)
    Lx, Ly, Lz = L_vec[0], L_vec[1], L_vec[2]
    L_mag = _lal_scalar_sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    Lhatx = Lx / L_mag
    Lhaty = Ly / L_mag
    Lhatz = Lz / L_mag
    Lhat = torch.stack([Lhatx, Lhaty, Lhatz])

    deriv_opts = _rhs_derivative_options()
    fd_dpvec = deriv_opts["fd_dpvec"]
    use_hamiltonian_spin = deriv_opts["use_hamiltonian_spin"]
    use_fd_spin = deriv_opts["use_fd_spin"]
    use_tortoise_pdot = deriv_opts["use_tortoise_pdot"]
    use_fd_x = deriv_opts["use_fd_x"]
    use_cartesian_x_grad = deriv_opts["use_cartesian_x_grad"]
    need_base_grad = (not use_tortoise_pdot) or (use_tortoise_pdot and not (use_fd_x or use_cartesian_x_grad))
    lagged_hcoeffs = getattr(params, "hcoeffs", None)
    use_lagged_tortoise = _env_on("PYCBC_SEOBNRV4PHM_LAGGED_TORTOISE", True)
    tortoise_hcoeffs = lagged_hcoeffs if (use_lagged_tortoise and lagged_hcoeffs is not None) else None
    tortoise_prelude = _lal_numerical_derivative_tortoise_prelude(
        r,
        L_vec,
        S1,
        S2,
        params,
        tortoise_hcoeffs,
        S1_weighted_override=S1_weighted,
        S2_weighted_override=S2_weighted,
    )
    pot = _eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=not fd_dpvec,
        compute_grad_spin=use_hamiltonian_spin and not use_fd_spin,
        fd_dpvec=fd_dpvec,
        fd_pphi=False,
        compute_base_grad=need_base_grad,
        S1_weighted_override=S1_weighted,
        S2_weighted_override=S2_weighted,
    )
    if use_hamiltonian_spin and use_fd_spin:
        pot["dH_dS1"], pot["dH_dS2"] = _dH_dspin_cartesian_fd(
            r,
            pr,
            phi,
            L_vec,
            S1,
            S2,
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            weighted_spins=lal_phase_state,
            S1_weighted_override=S1_weighted,
            S2_weighted_override=S2_weighted,
        )

    csi_fac = torch.clamp(tortoise_prelude["csi"], min=1.0e-15)
    Tmat, invTmat, dTijdXk = _tortoise_matrices(r_vec, csi_fac, tortoise_prelude["dcsi"])
    dxdt = _matvec3_lal_order(Tmat, pot["dH_dpvec"])
    rCrossV_x = r_vec[1] * dxdt[2] - r_vec[2] * dxdt[1]
    rCrossV_y = r_vec[2] * dxdt[0] - r_vec[0] * dxdt[2]
    rCrossV_z = r_vec[0] * dxdt[1] - r_vec[1] * dxdt[0]
    r_cross_v = torch.stack([rCrossV_x, rCrossV_y, rCrossV_z])
    omega_signed = _dot3(r_cross_v, Lhat) / torch.clamp(r * r, min=1.0e-12)
    # LAL computes the precessing orbital frequency from |r x rdot| / r^2,
    # not from the projection of r x rdot onto Lhat.
    omega = torch.clamp(
        _lal_scalar_sqrt(rCrossV_x * rCrossV_x + rCrossV_y * rCrossV_y + rCrossV_z * rCrossV_z) / (r * r),
        min=1.0e-12,
    )
    _refresh_hcoeffs(
        params,
        L_vec,
        S1,
        S2,
        S1_weighted=S1_weighted,
        S2_weighted=S2_weighted,
    )
    flux = _factorized_flux(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        omega,
        pot["H"],
        deltaT=pot["deltaT"],
        D=pot["D"],
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=dxdt,
        velocity_vec=dxdt,
        S1_weighted_override=S1_weighted,
        S2_weighted_override=S2_weighted,
    )

    pr_star = _dot3(p_vec, n_hat)
    p_non_tortoise = p_vec - n_hat * pr_star * (csi_fac - 1.0) / csi_fac
    if use_tortoise_pdot:
        if use_fd_x:
            dH_dx = _dH_dx_cartesian_fd(
                r_vec,
                p_non_tortoise,
                S1,
                S2,
                params,
                S1_weighted_override=S1_weighted,
                S2_weighted_override=S2_weighted,
            )
            _refresh_hcoeffs(
                params,
                L_vec,
                S1,
                S2,
                S1_weighted=S1_weighted,
                S2_weighted=S2_weighted,
            )
        elif use_cartesian_x_grad:
            dH_dx = _dH_dx_cartesian_autograd(r_vec, p_non_tortoise, S1, S2, params)
            _refresh_hcoeffs(params, L_vec, S1, S2)
        else:
            dH_dx = pot["dH_dr"] * n_hat
        pdot_t1 = _pdot_t1_lal_order(Tmat, dH_dx)
        pdot_rr = -flux * p_vec / torch.clamp(omega * L_mag, min=1.0e-12)
        pdot_t3 = _pdot_t3_lal_order(dTijdXk, invTmat, p_vec, dxdt)
        dpdt = pdot_t1 + pdot_rr + pdot_t3
    else:
        dpdt = -csi_fac * pot["dH_dr"] * n_hat
        dpdt = dpdt - flux * p_vec / torch.clamp(omega * L_mag, min=1.0e-12)

    dH_dS1 = pot.get("dH_dS1", None)
    dH_dS2 = pot.get("dH_dS2", None)
    dS1_weighted = None
    dS2_weighted = None
    if (not use_hamiltonian_spin) or dH_dS1 is None or dH_dS2 is None:
        pref1 = (2.0 + 3.0 * params.mass2 / (2.0 * params.mass1)) / torch.clamp(r ** 3, min=1e-12)
        pref2 = (2.0 + 3.0 * params.mass1 / (2.0 * params.mass2)) / torch.clamp(r ** 3, min=1e-12)
        dS1dt = _cross3(pref1 * Lhat, S1)
        dS2dt = _cross3(pref2 * Lhat, S2)
        if lal_phase_state:
            dS1_weighted = s1_scale * dS1dt
            dS2_weighted = s2_scale * dS2dt
    else:
        if lal_phase_state:
            if use_fd_spin:
                # The LAL-style finite differences above differentiate with
                # respect to the evolved, mass-weighted spins S_i/M^2.
                spin1_for_torque = y[6:9]
                spin2_for_torque = y[9:12]
            else:
                # Autograd differentiates with respect to chi_i.  Pair that
                # gradient with chi_i here so the returned rate is d(S_i/M^2)/dt;
                # using the weighted state would suppress each torque by
                # (m_i/M)^2.
                spin1_for_torque = S1
                spin2_for_torque = S2
            dS1_weighted = eta * _cross3(dH_dS1, spin1_for_torque)
            dS2_weighted = eta * _cross3(dH_dS2, spin2_for_torque)
            dS1dt = dS1_weighted / s1_scale
            dS2dt = dS2_weighted / s2_scale
        else:
            dS1dt = eta * _cross3(dH_dS1, S1)
            dS2dt = eta * _cross3(dH_dS2, S2)
            dS1dt, dS2dt = _hamiltonian_spin_rates_to_chi(params, dS1dt, dS2dt)

    phase_dot, zeta_dot = _phase_split_lal_scalar_order(r_vec, p_vec, dxdt, dpdt, omega)

    if os.environ.get("PYCBC_SEOBNRV4PHM_CART_SPIN_TORQUE", "0") not in ("0", "", "false", "False"):
        s1_scale = _lal_spin_scale(params.mass1, params.M)
        s2_scale = _lal_spin_scale(params.mass2, params.M)
        dL_prec = -(s1_scale * dS1dt + s2_scale * dS2dt)
        dpdt = dpdt - _cross3(r_vec, dL_prec) / torch.clamp(r * r, min=1.0e-12)

    _assert_finite("rhs_cartesian_full", dxdt, dpdt, omega_signed, dS1dt, dS2dt, phase_dot, flux, pot["H"])
    if lal_phase_state:
        if dS1_weighted is None:
            dS1_weighted = s1_scale * dS1dt
            dS2_weighted = s2_scale * dS2dt
        return torch.cat([dxdt, dpdt, dS1_weighted, dS2_weighted, phase_dot.view(1), zeta_dot.view(1)])
    return torch.cat([dxdt, dpdt, omega_signed.view(1), dS1dt, dS2dt])


def rhs_cartesian_projected(t: torch.Tensor, y: torch.Tensor, params: EOBParams):
    """LAL-style Cartesian x/P RHS projected back to the reduced PHM state."""

    r = y[0]
    pr = y[1]
    phi = y[2]
    L_vec = y[3:6]
    S1 = y[6:9]
    S2 = y[9:12]

    eta = torch.as_tensor(params.eta, device=r.device, dtype=r.dtype)
    L_mag = _safe_norm(L_vec)
    _refresh_hcoeffs(params, L_vec, S1, S2)

    r_vec, p_vec, n_hat, lambda_hat, Lhat = _reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)
    deriv_opts = _rhs_derivative_options()
    fd_dpvec = deriv_opts["fd_dpvec"]
    use_hamiltonian_spin = deriv_opts["use_hamiltonian_spin"]
    use_fd_spin = deriv_opts["use_fd_spin"]
    use_tortoise_pdot = deriv_opts["use_tortoise_pdot"]
    use_fd_x = deriv_opts["use_fd_x"]
    use_cartesian_x_grad = deriv_opts["use_cartesian_x_grad"]
    need_base_grad = (not use_tortoise_pdot) or (use_tortoise_pdot and not (use_fd_x or use_cartesian_x_grad))
    pot = _eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=not fd_dpvec,
        compute_grad_spin=use_hamiltonian_spin and not use_fd_spin,
        fd_dpvec=fd_dpvec,
        fd_pphi=False,
        compute_base_grad=need_base_grad,
    )
    if use_hamiltonian_spin and use_fd_spin:
        pot["dH_dS1"], pot["dH_dS2"] = _dH_dspin_cartesian_fd(
            r, pr, phi, L_vec, S1, S2, params, p_vec=p_vec, r_vec=r_vec
        )

    dH_dpvec = pot["dH_dpvec"]
    csi_fac = torch.clamp(pot["csi"], min=1.0e-15)
    Tmat, invTmat, dTijdXk = _tortoise_matrices(r_vec, csi_fac, pot["dcsi"])
    dxdt = _matvec3_lal_order(Tmat, dH_dpvec)

    drdt = torch.dot(dxdt, n_hat)
    dphidt = torch.dot(dxdt, lambda_hat) / torch.clamp(r, min=1.0e-12)
    rCrossV_x = r_vec[1] * dxdt[2] - r_vec[2] * dxdt[1]
    rCrossV_y = r_vec[2] * dxdt[0] - r_vec[0] * dxdt[2]
    rCrossV_z = r_vec[0] * dxdt[1] - r_vec[1] * dxdt[0]
    omega = torch.clamp(
        _lal_scalar_sqrt(rCrossV_x * rCrossV_x + rCrossV_y * rCrossV_y + rCrossV_z * rCrossV_z) / (r * r),
        min=1.0e-12,
    )
    flux = _factorized_flux(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        omega,
        pot["H"],
        deltaT=pot["deltaT"],
        D=pot["D"],
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=dxdt,
        velocity_vec=dxdt,
    )

    if use_tortoise_pdot:
        pr_star = _dot3(p_vec, n_hat)
        p_non_tortoise = p_vec - n_hat * pr_star * (csi_fac - 1.0) / csi_fac
        if use_fd_x:
            dH_dx = _dH_dx_cartesian_fd(r_vec, p_non_tortoise, S1, S2, params)
            _refresh_hcoeffs(params, L_vec, S1, S2)
        elif use_cartesian_x_grad:
            dH_dx = _dH_dx_cartesian_autograd(r_vec, p_non_tortoise, S1, S2, params)
            _refresh_hcoeffs(params, L_vec, S1, S2)
        else:
            dH_dx = pot["dH_dr"] * n_hat

        pdot_t1 = _pdot_t1_lal_order(Tmat, dH_dx)
        pdot_rr = -flux * p_vec / torch.clamp(omega * L_mag, min=1.0e-12)
        pdot_t3 = _pdot_t3_lal_order(dTijdXk, invTmat, p_vec, dxdt)
        dpdt = pdot_t1 + pdot_rr + pdot_t3
    else:
        pdot_cons = -csi_fac * pot["dH_dr"] * n_hat
        pdot_rr = -flux * p_vec / torch.clamp(omega * L_mag, min=1.0e-12)
        dpdt = pdot_cons + pdot_rr

    dprdt = torch.dot(dpdt, n_hat) + torch.dot(p_vec, lambda_hat) * dphidt
    dL_orb = _cross3(dxdt, p_vec) + _cross3(r_vec, dpdt)

    dH_dS1 = pot.get("dH_dS1", None)
    dH_dS2 = pot.get("dH_dS2", None)
    if (not use_hamiltonian_spin) or dH_dS1 is None or dH_dS2 is None:
        pref1 = (2.0 + 3.0 * params.mass2 / (2.0 * params.mass1)) / torch.clamp(r ** 3, min=1e-12)
        pref2 = (2.0 + 3.0 * params.mass1 / (2.0 * params.mass2)) / torch.clamp(r ** 3, min=1e-12)
        dS1dt = _cross3(pref1 * Lhat, S1)
        dS2dt = _cross3(pref2 * Lhat, S2)
    else:
        dS1dt = eta * _cross3(dH_dS1, S1)
        dS2dt = eta * _cross3(dH_dS2, S2)
        dS1dt, dS2dt = _hamiltonian_spin_rates_to_chi(params, dS1dt, dS2dt)
    # LAL evolves Cartesian x/P and reports L = x cross P; in the reduced
    # projection the corresponding derivative is therefore d(x cross P) only.
    dLdt = dL_orb

    _assert_finite(
        "rhs_cartesian_projected",
        drdt,
        dprdt,
        dphidt,
        dLdt,
        dS1dt,
        dS2dt,
        flux,
        pot["H"],
        pot["Heff"],
        pot["deltaT"],
        pot["D"],
    )

    return torch.stack([drdt, dprdt, dphidt, dLdt[0], dLdt[1], dLdt[2], dS1dt[0], dS1dt[1], dS1dt[2], dS2dt[0], dS2dt[1], dS2dt[2]])


def rhs(
    t: torch.Tensor,
    y: torch.Tensor,
    params: EOBParams,
    *,
    p_vec: torch.Tensor | None = None,
    r_vec: torch.Tensor | None = None,
):
    """Precessing SEOBNRv4PHM RHS (geometric units).

    State ordering (default) ::
        y = [r, pr, phi, Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z]

    Optional overrides:
        - ``p_vec``: Cartesian momentum to use directly for pf/pxir/ptheta (e.g. LAL snapshots)
        - ``r_vec``: Cartesian position to define n_hat from the snapshot (used with ``p_vec``)
    """
    # Port of XLALSpinPrecHcapRvecDerivative
    # (LALSimIMRSpinEOBHamiltonianPrec.c:1021-1230).

    r = y[0]
    pr = y[1]
    phi = y[2]
    L_state = y[3:6]
    p_override = p_vec
    r_override = r_vec
    if (r_override is not None) and (p_override is not None):
        L_vec = torch.linalg.cross(r_override, p_override)
    else:
        L_vec = L_state
    S1 = y[6:9]
    S2 = y[9:12]

    L_mag = _safe_norm(L_vec)
    Lhat = L_vec / L_mag

    # Refresh hcoeffs with instantaneous precessing spins (sigma_Kerr) and
    # augmented spin chi (LALSimIMRSpinPrecEOBv4P.c:1806-1834)
    _refresh_hcoeffs(params, L_vec, S1, S2)

    # Provide Cartesian momentum vector when available to mirror LAL derivative path
    pot = _eob_potentials(r, pr, phi, L_vec, S1, S2, params, p_vec=p_override, r_vec=r_override)
    deltaT = pot["deltaT"]
    D = pot["D"]
    Heff = pot["Heff"]
    H = pot["H"]
    dH_dpr = pot["dH_dpr"]
    dH_dpf = pot["dH_dpf"]
    dH_dr = pot["dH_dr"]
    dH_dpvec = pot.get("dH_dpvec", None)

    if dH_dpvec is not None and r_override is not None:
        # Map dH/dP (tortoise momenta) through the T-matrix to get Cartesian velocities.
        # Optional dcsi-aware correction (Eq. A5 of Pan+2010) can be toggled for diagnostics.
        csi_fac = torch.clamp(pot["csi"], min=1.0e-15)
        nhat_dot_dH = torch.dot(dH_dpvec, pot["n_hat"])
        v_vec = dH_dpvec + (csi_fac - 1.0) * nhat_dot_dH * pot["n_hat"]

        # Enable with PYCBC_SEOBNRV4PHM_DCSI_VEL=1 if dcsi contraction is desired.
        if os.environ.get("PYCBC_SEOBNRV4PHM_DCSI_VEL", "0") not in ("0", "", "false", "False"):
            rmag = _safe_norm(r_override)
            dcsi = pot["dcsi"]
            corr = torch.zeros_like(v_vec)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        dT = ((r_override[i] * (1 if j == k else 0) + (1 if i == k else 0) * r_override[j]) * (csi_fac - 1.0) / (rmag * rmag))
                        dT = dT + r_override[i] * r_override[j] * r_override[k] / (rmag * rmag * rmag) * (
                            -2.0 * (csi_fac - 1.0) / rmag + dcsi
                        )
                        corr[i] = corr[i] + dT * p_override[j] * dH_dpvec[k]
            v_vec = v_vec + corr
        drdt = torch.dot(v_vec, pot["n_hat"])
        omega_vec = torch.linalg.cross(r_override, v_vec)
        dphidt = _safe_norm(omega_vec) / torch.clamp(_dot3(r_override, r_override), min=1.0e-12)
        dprdt = -dH_dr
    else:
        drdt = dH_dpr
        dphidt = dH_dpf
        dprdt = -dH_dr

    omega = dphidt
    flux = _factorized_flux(r, pr, phi, L_vec, S1, S2, params, omega, H, deltaT=deltaT, D=D)
    dL_mag_dt = -flux / torch.clamp(omega, min=1e-12)

    if _debug_enabled() and (
        torch.isnan(drdt) or torch.isnan(dprdt) or torch.isnan(dphidt) or torch.isnan(dL_mag_dt) or torch.isnan(flux)
    ):
        _dbg(
            f"rhs nan r={float(r):.4f} pr={float(pr):.4e} omega={float(omega):.4e} "
            f"deltaT={float(deltaT):.4e} D={float(D):.4e} Heff={float(Heff):.4e} H={float(H):.4e} flux={float(flux):.4e}"
        )

    # Simple precession torques (leading-order SO coupling)
    pref1 = (2.0 + 3.0 * params.mass2 / (2.0 * params.mass1)) / torch.clamp(r ** 3, min=1e-12)
    pref2 = (2.0 + 3.0 * params.mass1 / (2.0 * params.mass2)) / torch.clamp(r ** 3, min=1e-12)
    Omega1 = pref1 * Lhat
    Omega2 = pref2 * Lhat
    dS1dt = _cross3(Omega1, S1)
    dS2dt = _cross3(Omega2, S2)

    s1_scale = _lal_spin_scale(params.mass1, params.M)
    s2_scale = _lal_spin_scale(params.mass2, params.M)
    dL_prec = -(s1_scale * dS1dt + s2_scale * dS2dt)
    dL_rr = Lhat * dL_mag_dt
    dLdt = dL_prec + dL_rr

    _assert_finite(
        "rhs",
        drdt,
        dprdt,
        dphidt,
        dLdt,
        dS1dt,
        dS2dt,
        flux,
        H,
        Heff,
        deltaT,
        D,
    )

    return torch.stack([drdt, dprdt, dphidt, dLdt[0], dLdt[1], dLdt[2], dS1dt[0], dS1dt[1], dS1dt[2], dS2dt[0], dS2dt[1], dS2dt[2]])


def nqc_corrections(*args, **kwargs):
    """Compute simple NQC summary at the inspiral peak (used for diagnostics)."""
    traj = kwargs["traj"]
    params: EOBParams = kwargs["params"]
    mode_l = kwargs.get("mode_l", 2)
    mode_m = kwargs.get("mode_m", 2)

    t = torch.stack([pt[0] for pt in traj])
    y_stack = torch.stack([pt[1] for pt in traj])
    r = y_stack[:, 0]
    Lvec = y_stack[:, 3:6]
    Lmag = torch.sqrt(torch.clamp(torch.sum(Lvec * Lvec, dim=1), min=1e-14))
    omega = Lmag / torch.clamp(r * r, min=1e-12)
    amp_proxy = 1.0 / r  # monotonic proxy for amplitude

    t_peak = find_peak_time(omega, t)
    amp_peak = float(torch.max(amp_proxy))
    adot = local_derivatives(amp_proxy, t, t_peak, order=1)
    addot = local_derivatives(amp_proxy, t, t_peak, order=2)

    coeffs = solve_nqc_coeffs(params.eta, params.chiS, params.chiA, mode_l=mode_l, mode_m=mode_m)

    out = dict(
        a1=coeffs["a1"],
        a2=coeffs["a2"],
        a3=coeffs["a3"],
        a4=coeffs["a4"],
        a5=coeffs["a5"],
        b1=coeffs["b1"],
        b2=coeffs["b2"],
        b3=coeffs["b3"],
        peak_amp_EOB=amp_peak,
        peak_adot_EOB=adot,
        peak_addot_EOB=addot,
        peak_amp_NR=coeffs["peak_amp"],
        peak_adot_NR=coeffs["peak_adot"],
        peak_addot_NR=coeffs["peak_addot"],
        t_peak=t_peak,
    )
    return out


__all__ = [
    "initial_conditions",
    "initial_cartesian_conditions",
    "rhs",
    "rhs_cartesian_projected",
    "rhs_cartesian_full",
    "reduced_state_to_cartesian_state",
    "cartesian_state_to_reduced_state",
    "nqc_corrections",
    "LM_DEFAULT",
    "normalize_mode_array",
    "EOBParams",
    "EPS_ALIGN",
    "highest_initial_freq",
    "settings",
    "dphi_dt_fd",
]
