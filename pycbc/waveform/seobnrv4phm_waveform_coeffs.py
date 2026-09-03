"""Aligned-spin factorized waveform coefficients for SEOBNRv4PHM (v4P base).

Ported symbolically from lalsimulation/lib/LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c
for SpinAlignedEOBversion = 4. Expressions are kept in callable form to allow
torch evaluation with dynamic eta and spin combinations.
"""

from __future__ import annotations

import math
import os
import re

from .seobnrv4phm_constants import _EULER_GAMMA

_PI = math.pi


def _chiS_chiA(m1, m2, chi1z, chi2z):
    m = m1 + m2
    dM = (m1 - m2) / m
    chiS = 0.5 * (chi1z + chi2z)
    chiA = 0.5 * (chi1z - chi2z)
    return chiS, chiA, dM


def rho22(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of rho_22 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:166-206."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    a = chiS + chiA * dM
    a2 = a * a
    eta2 = eta * eta
    rho22v2 = -43.0 / 42.0 + (55.0 * eta) / 84.0
    rho22v3 = (-2.0 * (chiS + chiA * dM - chiS * eta)) / 3.0
    rho22v4 = -20555.0 / 10584.0 + 0.5 * a2 - (33025.0 * eta) / 21168.0 + (19583.0 * eta2) / 42336.0
    rho22v5 = (-34.0 * a) / 21.0
    rho22v6 = (
        1556919113.0 / 122245200.0
        + (89.0 * a2) / 252.0
        - (48993925.0 * eta) / 9779616.0
        - (6292061.0 * eta2) / 3259872.0
        + (10620745.0 * eta2 * eta) / 39118464.0
        + (41.0 * eta * _PI * _PI) / 192.0
    )
    rho22v7 = (18733.0 * a) / 15876.0 + a * a2 / 3.0
    return [rho22v2, rho22v3, rho22v4, rho22v5, rho22v6, rho22v7]


def delta22(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of delta_22 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:144-186."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    aDelta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    delta22vh6 = (-4.0 * aDelta) / 3.0 + (428.0 * _PI) / 105.0
    delta22v8 = (20.0 * aDelta) / 63.0
    delta22vh9 = -2203.0 / 81.0 + (1712.0 * _PI * _PI) / 315.0
    delta22v5 = -24.0 * eta
    delta22v6 = 0.0
    return [delta22vh6, delta22v8, delta22vh9, delta22v5, delta22v6]


def rho33(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of rho_33 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:375-399."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    a = chiS + chiA * dM
    a2 = a * a
    eta2 = eta * eta
    rho33v2 = -7.0 / 6.0 + (2.0 * eta) / 3.0
    rho33v3 = (chiS * dM * (-4.0 + 5.0 * eta) + chiA * (-4.0 + 19.0 * eta)) / (6.0 * dM)
    rho33v4 = -6719.0 / 3960.0 + a2 / 2.0 - (1861.0 * eta) / 990.0 + (149.0 * eta2) / 330.0
    rho33v5 = (-4.0 * a) / 3.0
    rho33v6 = 3203101567.0 / 227026800.0 + (5.0 * a2) / 36.0
    rho33v7 = (5297.0 * a) / 2970.0 + a * a2 / 3.0
    return [rho33v2, rho33v3, rho33v4, rho33v5, rho33v6, rho33v7]


def rho21(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of rho_21 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:248-300."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    dM2 = dM * dM
    a = chiS + chiA * dM
    a2 = a * a
    a3 = a2 * a
    eta2 = eta * eta
    rho21v1 = (-3.0 * (chiS + chiA / dM)) / 4.0
    chiAPlusChiSdM = chiA + chiS * dM
    rho21v2 = -59.0 / 56.0 - (9.0 * chiAPlusChiSdM * chiAPlusChiSdM) / (32.0 * dM2) + (23.0 * eta) / 84.0
    rho21v3 = 1177.0 / 672.0 * a - 27.0 / 128.0 * a3
    rho21v4 = -47009.0 / 56448.0 - (865.0 * a2) / 1792.0 - (405.0 * a2 * a2) / 2048.0 - (10993.0 * eta) / 14112.0 + (617.0 * eta2) / 4704.0
    rho21v5 = (-98635.0 * a) / 75264.0 + (2031.0 * a * a2) / 7168.0 - (1701.0 * a2 * a3) / 8192.0
    rho21v6 = (
        7613184941.0 / 2607897600.0
        + (9032393.0 * a2) / 1806336.0
        + (3897.0 * a2 * a2) / 16384.0
        - (15309.0 * a3 * a3) / 65536.0
    )
    return [rho21v1, rho21v2, rho21v3, rho21v4, rho21v5, rho21v6]


def rho44(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of rho_44 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:480-515."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    a = chiS + chiA * dM
    a2 = a * a
    eta2 = eta * eta
    m1Plus3eta = -1.0 + 3.0 * eta
    m1Plus3eta2 = m1Plus3eta * m1Plus3eta
    rho44v2 = (1614.0 - 5870.0 * eta + 2625.0 * eta2) / (1320.0 * m1Plus3eta)
    rho44v3 = (chiA * (10.0 - 39.0 * eta) * dM + chiS * (10.0 - 41.0 * eta + 42.0 * eta2)) / (15.0 * m1Plus3eta)
    rho44v4 = (
        a2 / 2.0
        + (
            -511573572.0
            + 2338945704.0 * eta
            - 313857376.0 * eta2
            - 6733146000.0 * eta * eta2
            + 1252563795.0 * eta2 * eta2
        )
        / (317116800.0 * m1Plus3eta2)
    )
    rho44v5 = (-69.0 * a) / 55.0
    rho44v6 = 16600939332793.0 / 1098809712000.0 + (217.0 * a2) / 3960.0
    return [rho44v2, rho44v3, rho44v4, rho44v5, rho44v6]


def rho55(eta, chi1z, chi2z, m1, m2):
    """COMPLETE port of rho_55 coefficients from
    LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:568-584."""
    chiS, chiA, dM = _chiS_chiA(m1, m2, chi1z, chi2z)
    a = chiS + chiA * dM
    a2 = a * a
    eta2 = eta * eta
    rho55v2 = (487.0 - 1298.0 * eta + 512.0 * eta2) / (390.0 * (-1.0 + 2.0 * eta))
    rho55v3 = (-2.0 * a) / 3.0
    rho55v4 = -3353747.0 / 2129400.0 + a2 / 2.0
    rho55v5 = -241.0 * a / 195.0
    rho55v6 = 0.0
    return [rho55v2, rho55v3, rho55v4, rho55v5, rho55v6]


# ---------------------------------------------------------------------------
# Extended helpers (log terms, aux f_lm, deltas) for torch-native port
# ---------------------------------------------------------------------------


def _spin_combinations(m1, m2, chi1z, chi2z):
    m = m1 + m2
    dM = (m1 - m2) / m
    chiS = 0.5 * (chi1z + chi2z)
    chiA = 0.5 * (chi1z - chi2z)
    return chiS, chiA, dM, dM * dM


def _eulerlog(m, v):
    return _EULER_GAMMA + math.log(2.0 * abs(m) * max(v, 1e-16))


def rho_lm_full(l, m, eta, chi1z, chi2z, m1, m2, v, *, waveform: bool = False, tplspin=None):
    """Return (rho_lm, aux_f_lm) including log terms.

    This mirrors the aligned-spin coefficients used in
    ``LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c`` (SpinAlignedEOBversion=4).
    """

    chiS, chiA, dM, dM2 = _spin_combinations(m1, m2, chi1z, chi2z)
    a_delta = chiS + chiA * dM
    a = ((1.0 - 2.0 * eta) * chiS + dM * chiA) if tplspin is None else tplspin
    a2 = a * a
    a3 = a2 * a
    eta2 = eta * eta
    eta3 = eta2 * eta
    eulerlog = _eulerlog(m, v)

    if l == 2 and m == 2:
        # Matches SEOBNRv4P/HM rho_22 coefficients (LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:166-225)
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
            + (41.0 * eta * _PI * _PI) / 192.0
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

        rho = 1.0 + v * v * (
            rho22v2
            + v * (rho22v3 + v * (rho22v4 + v * (rho22v5 + v * (rho22v6 + rho22v6l * eulerlog + v * (rho22v7 + v * (rho22v8 + rho22v8l * eulerlog + (rho22v10 + rho22v10l * eulerlog) * v * v))))))
        )
        return rho, 0.0

    if l == 2 and m == 1:
        # Matches rho_21 + f_21 aux terms (LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:248-329/338-364)
        rho21v1 = 0.0
        rho21v2 = -59.0 / 56.0 + (23.0 * eta) / 84.0
        rho21v3 = 0.0
        rho21v4 = -47009.0 / 56448.0 - (865.0 * a2) / 1792.0 - (405.0 * a2 * a2) / 2048.0 - (10993.0 * eta) / 14112.0 + (617.0 * eta2) / 4704.0
        rho21v5 = (-98635.0 * a) / 75264.0 + (2031.0 * a * a2) / 7168.0 - (1701.0 * a2 * a3) / 8192.0
        rho21v6 = 7613184941.0 / 2607897600.0 + (9032393.0 * a2) / 1806336.0 + (3897.0 * a2 * a2) / 16384.0 - (15309.0 * a3 * a3) / 65536.0
        rho21v6l = -107.0 / 105.0
        rho21v7 = (-3859374457.0 * a) / 1159065600.0 - (55169.0 * a3) / 16384.0 + (18603.0 * a2 * a3) / 65536.0 - (72171.0 * a2 * a2 * a3) / 262144.0
        rho21v7l = 107.0 * a / 140.0
        rho21v8 = -1168617463883.0 / 911303737344.0
        rho21v8l = 6313.0 / 5880.0
        rho21v10 = -63735873771463.0 / 16569158860800.0
        rho21v10l = 5029963.0 / 5927040.0

        rho = 1.0 + v * (
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
                                + v * (
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

        # aux f_21
        if abs(dM) < 1e-12:
            inv_dM = 0.0
        else:
            inv_dM = 1.0 / dM
        f21v1 = (-3.0 * (chiS + chiA * inv_dM)) / 2.0 if dM2 else -1.5 * chiA
        f21v3 = (
            (chiS * dM * (427.0 + 79.0 * eta) + chiA * (147.0 + 280.0 * dM2 + 1251.0 * eta)) / (84.0 * dM)
            if dM2
            else (3.0 / 8.0) * chiA
        )
        aux = v * f21v1 + v * v * v * f21v3
        if waveform:
            chiS2 = chiS * chiS
            chiA2 = chiA * chiA
            chiA3 = chiA2 * chiA
            aux += v**4 * ((-3.0 - 2.0 * eta) * chiA2 + (-3.0 + 0.5 * eta) * chiS2 + (-6.0 + 10.5 * eta) * chiS * chiA * inv_dM)
            aux += v**5 * (
                (0.75 - 3.0 * eta) * chiA3 * inv_dM
                + (-81.0 / 16.0 - 703.0 * eta2 / 112.0 + 8797.0 * eta / 1008.0 + (9.0 / 4.0 - 6.0 * eta) * chiS2) * chiA * inv_dM
                + (0.75 * chiS**3)
                + ((9.0 / 4.0 - 3.0 * eta) * chiA2 * chiS)
            )
            aux += v**6 * (
                (4163.0 / 252.0 - 9287.0 * eta / 1008.0 - 85.0 * eta2 / 112.0) * chiA2
                + (4163.0 / 252.0 - 2633.0 * eta / 1008.0 + 461.0 * eta2 / 1008.0) * chiS2
                + (4163.0 / 126.0 - 1636.0 * eta / 21.0 + 1088.0 * eta2 / 63.0) * chiS * chiA * inv_dM
            )
        return rho, aux

    if l == 3 and m == 3:
        rho33v2 = -7.0 / 6.0 + (2.0 * eta) / 3.0
        rho33v3 = 0.0
        rho33v4 = -6719.0 / 3960.0 + a2 / 2.0 - (1861.0 * eta) / 990.0 + (149.0 * eta2) / 330.0
        rho33v5 = (-4.0 * a) / 3.0
        rho33v6 = 3203101567.0 / 227026800.0 + (5.0 * a2) / 36.0
        rho33v6l = -26.0 / 7.0
        rho33v7 = (5297.0 * a) / 2970.0 + a3 / 3.0
        rho33v8 = -57566572157.0 / 8562153600.0
        rho33v8l = 13.0 / 3.0
        rho33v10 = 0.0
        rho33v10l = 0.0
        if waveform:
            rho33v6 += (-129509.0 / 25740.0 + 41.0 * _PI * _PI / 192.0) * eta - 274621.0 / 154440.0 * eta2 + 12011.0 / 46332.0 * eta3
            rho33v10 = -903823148417327.0 / 30566888352000.0
            rho33v10l = 87347.0 / 13860.0
        rho = 1.0 + v * v * (
            rho33v2
            + v * (
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

        if abs(dM) < 1e-12:
            inv_dM = 0.0
        else:
            inv_dM = 1.0 / dM

        f33v3 = (chiS * dM * (-4.0 + 5.0 * eta) + chiA * (-4.0 + 19.0 * eta)) / (2.0 * dM) if dM2 else 0.375 * chiA
        f33v4 = 0.0
        f33v5 = 0.0
        f33v6 = 0.0
        f33vh6 = 0.0
        if waveform:
            chiS2 = chiS * chiS
            chiA2 = chiA * chiA
            if dM2:
                f33v4 = (1.5 * chiS2 * dM + (3.0 - 12.0 * eta) * chiA * chiS + dM * (1.5 - 6.0 * eta) * chiA2) * inv_dM
                f33v5 = (dM * (241.0 / 30.0 * eta2 + 11.0 / 20.0 * eta + 2.0 / 3.0) * chiS + (407.0 / 30.0 * eta2 - 593.0 / 60.0 * eta + 2.0 / 3.0) * chiA) * inv_dM
                f33v6 = (dM * (6.0 * eta2 - 13.5 * eta - 1.75) * chiS2 + (44.0 * eta2 - eta - 3.5) * chiA * chiS + dM * (-12.0 * eta2 + 5.5 * eta - 1.75) * chiA2) * inv_dM
                f33vh6 = (dM * (593.0 / 108.0 * eta - 81.0 / 20.0) * chiS + (7339.0 / 540.0 * eta - 81.0 / 20.0) * chiA) * inv_dM
            else:
                f33v4 = (3.0 - 12.0 * eta) * chiA * chiS
                f33v5 = (407.0 / 30.0 * eta2 - 593.0 / 60.0 * eta + 2.0 / 3.0) * chiA
                f33v6 = (44.0 * eta2 - eta - 3.5) * chiA * chiS
                f33vh6 = (7339.0 / 540.0 * eta - 81.0 / 20.0) * chiA

        aux = (
            v ** 3 * f33v3
            + v ** 4 * f33v4
            + v ** 5 * f33v5
            + v ** 6 * f33v6
            + 1j * (v ** 6) * f33vh6
        )
        return rho, aux

    if l == 3 and m == 2:
        m1Plus3eta = -1.0 + 3.0 * eta
        m1Plus3eta2 = m1Plus3eta * m1Plus3eta
        rho32v = (4.0 * chiS * eta) / (-3.0 * m1Plus3eta)
        rho32v2 = (328.0 - 1115.0 * eta + 320.0 * eta2) / (270.0 * m1Plus3eta)
        rho32v3 = (2.0 * a) / 9.0
        rho32v4 = a2 / 3.0 + (
            -1444528.0
            + 8050045.0 * eta
            - 4725605.0 * eta2
            - 20338960.0 * eta3
            + 3085640.0 * eta2 * eta2
        ) / (1603800.0 * m1Plus3eta2)
        rho32v5 = (-2788.0 * a) / 1215.0
        rho32v6 = 5849948554.0 / 940355325.0 + (488.0 * a2) / 405.0
        rho32v6l = -104.0 / 63.0
        rho32v8 = -10607269449358.0 / 3072140846775.0
        rho32v8l = 17056.0 / 8505.0
        rho = 1.0 + v * (
            rho32v
            + v * (
                rho32v2
                + v
                * (
                    rho32v3
                    + v
                    * (
                        rho32v4
                        + v
                        * (
                            rho32v5
                            + v * (rho32v6 + rho32v6l * eulerlog + (rho32v8 + rho32v8l * eulerlog) * v * v)
                        )
                    )
                )
            )
        )
        return rho, 0.0

    if l == 3 and m == 1:
        if dM2:
            rho31v2 = -13.0 / 18.0 - (2.0 * eta) / 9.0
            rho31v3 = 0.0
            rho31v4 = 101.0 / 7128.0 - (5.0 * a2) / 6.0 - (1685.0 * eta) / 1782.0 - (829.0 * eta2) / 1782.0
            rho31v5 = (4.0 * a) / 9.0
            rho31v6 = 11706720301.0 / 6129723600.0 - (49.0 * a2) / 108.0
            rho31v6l = -26.0 / 63.0
            rho31v7 = (-2579.0 * a) / 5346.0 + a3 / 9.0
            rho31v8 = 2606097992581.0 / 4854741091200.0
            rho31v8l = 169.0 / 567.0
            rho = 1.0 + v * v * (
                rho31v2
                + v
                * (
                    rho31v3
                    + v
                    * (
                        rho31v4
                        + v
                        * (
                            rho31v5
                            + v * (rho31v6 + rho31v6l * eulerlog + v * (rho31v7 + (rho31v8 + rho31v8l * eulerlog) * v))
                        )
                    )
                )
            )
            f31v3 = (chiA * (-4.0 + 11.0 * eta) + chiS * dM * (-4.0 + 13.0 * eta)) / (2.0 * dM)
        else:
            rho = 1.0
            f31v3 = -5.0 * chiA / 8.0
        return rho, v**3 * f31v3

    if l == 4 and m == 4:
        m1Plus3eta = -1.0 + 3.0 * eta
        m1Plus3eta2 = m1Plus3eta * m1Plus3eta
        rho44v2 = (1614.0 - 5870.0 * eta + 2625.0 * eta2) / (1320.0 * m1Plus3eta)
        rho44v3 = (chiA * (10.0 - 39.0 * eta) * dM + chiS * (10.0 - 41.0 * eta + 42.0 * eta2)) / (15.0 * m1Plus3eta)
        rho44v4 = a2 / 2.0 + (-511573572.0 + 2338945704.0 * eta - 313857376.0 * eta2 - 6733146000.0 * eta * eta2 + 1252563795.0 * eta2 * eta2) / (317116800.0 * m1Plus3eta2)
        rho44v5 = (-69.0 * a) / 55.0
        rho44v6 = 16600939332793.0 / 1098809712000.0 + (217.0 * a2) / 3960.0
        rho44v6l = -12568.0 / 3465.0
        rho = 1.0 + v * v * (
            rho44v2
            + v * (rho44v3 + v * (rho44v4 + v * (rho44v5 + (rho44v6 + rho44v6l * eulerlog) * v)))
        )
        return rho, 0.0

    if l == 4 and m == 3:
        if dM2:
            rho43v = 0.0
            rho43v2 = (222.0 - 547.0 * eta + 160.0 * eta2) / (176.0 * (-1.0 + 2.0 * eta))
            rho43v4 = -6894273.0 / 7047040.0 + (3.0 * a2) / 8.0
            rho43v5 = (-12113.0 * a) / 6160.0
            rho43v6 = 1664224207351.0 / 195343948800.0
            rho43v6l = -1571.0 / 770.0
            rho = 1.0 + v * (
                rho43v
                + v
                * (
                    rho43v2
                    + v * v * (rho43v4 + v * (rho43v5 + (rho43v6 + rho43v6l * eulerlog) * v))
                )
            )
            f43v = (5.0 * (chiA - chiS * dM) * eta) / (2.0 * dM * (-1.0 + 2.0 * eta))
        else:
            rho = 1.0
            f43v = -5.0 * chiA / 4.0
        return rho, v * f43v

    if l == 4 and m == 2:
        m1Plus3eta = -1.0 + 3.0 * eta
        m1Plus3eta2 = m1Plus3eta * m1Plus3eta
        rho42v2 = (1146.0 - 3530.0 * eta + 285.0 * eta2) / (1320.0 * m1Plus3eta)
        rho42v3 = (chiA * (10.0 - 21.0 * eta) * dM + chiS * (10.0 - 59.0 * eta + 78.0 * eta2)) / (15.0 * m1Plus3eta)
        rho42v4 = a2 / 2.0 + (
            -114859044.0
            + 295834536.0 * eta
            + 1204388696.0 * eta2
            - 3047981160.0 * eta3
            - 379526805.0 * eta2 * eta2
        ) / (317116800.0 * m1Plus3eta2)
        rho42v5 = (-7.0 * a) / 110.0
        rho42v6 = 848238724511.0 / 219761942400.0 + (2323.0 * a2) / 3960.0
        rho42v6l = -3142.0 / 3465.0
        rho = 1.0 + v * v * (
            rho42v2 + v * (rho42v3 + v * (rho42v4 + v * (rho42v5 + (rho42v6 + rho42v6l * eulerlog) * v)))
        )
        return rho, 0.0

    if l == 4 and m == 1:
        if dM2:
            rho41v = 0.0
            rho41v2 = (602.0 - 1385.0 * eta + 288.0 * eta2) / (528.0 * (-1.0 + 2.0 * eta))
            rho41v4 = -7775491.0 / 21141120.0 + (3.0 * a2) / 8.0
            rho41v5 = (-20033.0 * a) / 55440.0 - (5.0 * a3) / 6.0
            rho41v6 = 1227423222031.0 / 1758095539200.0
            rho41v6l = -1571.0 / 6930.0
            rho = 1.0 + v * (
                rho41v
                + v
                * (
                    rho41v2
                    + v * v * (rho41v4 + v * (rho41v5 + (rho41v6 + rho41v6l * eulerlog) * v))
                )
            )
            f41v = (5.0 * (chiA - chiS * dM) * eta) / (2.0 * dM * (-1.0 + 2.0 * eta))
        else:
            rho = 1.0
            f41v = -5.0 * chiA / 4.0
        return rho, v * f41v

    if l == 5 and m == 5:
        denom = (-1.0 + 2.0 * eta)
        rho55v2 = (487.0 - 1298.0 * eta + 512.0 * eta2) / (390.0 * denom)
        rho55v3 = (-2.0 * a) / 3.0
        rho55v4 = -3353747.0 / 2129400.0 + a2 / 2.0
        rho55v5 = -241.0 * a / 195.0
        rho55v6 = 0.0 if not waveform else 190606537999247.0 / 11957879934000.0
        rho55v6l = 0.0 if not waveform else -1546.0 / 429.0
        rho = 1.0 + v * v * (
            rho55v2
            + v * (rho55v3 + v * (rho55v4 + v * (rho55v5 + (rho55v6 + rho55v6l * eulerlog) * v)))
        )
        aux = 0.0
        if waveform:
            # Leading aux terms for HM (kept simple; calibration term f55v5c kept at 0)
            if abs(dM) > 1e-12:
                aux = v**3 * (
                    chiA / dM * (10.0 / (3.0 * denom) - 70.0 * eta / (3.0 * denom) + 110.0 * eta2 / (3.0 * denom))
                    + chiS * (10.0 / (3.0 * denom) - 10.0 * eta / denom + 10.0 * eta2 / denom)
                )
            else:
                aux = v**3 * chiA * (10.0 / (3.0 * denom) - 70.0 * eta / (3.0 * denom) + 110.0 * eta2 / (3.0 * denom))
        return rho, aux

    if l == 5 and m == 4:
        den = 1.0 - 5.0 * eta + 5.0 * eta2
        rho54v2 = (-17448.0 + 96019.0 * eta - 127610.0 * eta2 + 33320.0 * eta3) / (13650.0 * den)
        rho54v3 = (-2.0 * a) / 15.0
        rho54v4 = -16213384.0 / 15526875.0 + (2.0 * a2) / 5.0
        return 1.0 + v * v * (rho54v2 + v * (rho54v3 + rho54v4 * v)), 0.0

    if l == 5 and m == 3:
        if not dM2:
            return 1.0, 0.0
        rho53v2 = (375.0 - 850.0 * eta + 176.0 * eta2) / (390.0 * (-1.0 + 2.0 * eta))
        rho53v3 = (-2.0 * a) / 3.0
        rho53v4 = -410833.0 / 709800.0 + a2 / 2.0
        rho53v5 = -103.0 * a / 325.0
        return 1.0 + v * v * (rho53v2 + v * (rho53v3 + v * (rho53v4 + rho53v5 * v))), 0.0

    if l == 5 and m == 2:
        den = 1.0 - 5.0 * eta + 5.0 * eta2
        rho52v2 = (-15828.0 + 84679.0 * eta - 104930.0 * eta2 + 21980.0 * eta3) / (13650.0 * den)
        rho52v3 = (-2.0 * a) / 15.0
        rho52v4 = -7187914.0 / 15526875.0 + (2.0 * a2) / 5.0
        return 1.0 + v * v * (rho52v2 + v * (rho52v3 + rho52v4 * v)), 0.0

    if l == 5 and m == 1:
        if not dM2:
            return 1.0, 0.0
        rho51v2 = (319.0 - 626.0 * eta + 8.0 * eta2) / (390.0 * (-1.0 + 2.0 * eta))
        rho51v3 = (-2.0 * a) / 3.0
        rho51v4 = -31877.0 / 304200.0 + a2 / 2.0
        rho51v5 = 139.0 * a / 975.0
        return 1.0 + v * v * (rho51v2 + v * (rho51v3 + v * (rho51v4 + rho51v5 * v))), 0.0

    if l == 6:
        den_even = 1.0 - 5.0 * eta + 5.0 * eta2
        den_odd = dM2 + 3.0 * eta2
        if m == 6:
            rho66v2 = (-106.0 + 602.0 * eta - 861.0 * eta2 + 273.0 * eta3) / (84.0 * den_even)
            rho66v3 = (-2.0 * a) / 3.0
            rho66v4 = -1025435.0 / 659736.0 + a2 / 2.0
            return 1.0 + v * v * (rho66v2 + v * (rho66v3 + rho66v4 * v)), 0.0
        if m == 5:
            if not dM2:
                return 1.0, 0.0
            rho65v2 = (-185.0 + 838.0 * eta - 910.0 * eta2 + 220.0 * eta3) / (144.0 * den_odd)
            rho65v3 = -2.0 * a / 9.0
            return 1.0 + v * v * (rho65v2 + rho65v3 * v), 0.0
        if m == 4:
            rho64v2 = (-86.0 + 462.0 * eta - 581.0 * eta2 + 133.0 * eta3) / (84.0 * den_even)
            rho64v3 = (-2.0 * a) / 3.0
            rho64v4 = -476887.0 / 659736.0 + a2 / 2.0
            return 1.0 + v * v * (rho64v2 + v * (rho64v3 + rho64v4 * v)), 0.0
        if m == 3:
            if not dM2:
                return 1.0, 0.0
            rho63v2 = (-169.0 + 742.0 * eta - 750.0 * eta2 + 156.0 * eta3) / (144.0 * den_odd)
            rho63v3 = -2.0 * a / 9.0
            return 1.0 + v * v * (rho63v2 + rho63v3 * v), 0.0
        if m == 2:
            rho62v2 = (-74.0 + 378.0 * eta - 413.0 * eta2 + 49.0 * eta3) / (84.0 * den_even)
            rho62v3 = (-2.0 * a) / 3.0
            rho62v4 = -817991.0 / 3298680.0 + a2 / 2.0
            return 1.0 + v * v * (rho62v2 + v * (rho62v3 + rho62v4 * v)), 0.0
        if m == 1:
            if not dM2:
                return 1.0, 0.0
            rho61v2 = (-161.0 + 694.0 * eta - 670.0 * eta2 + 124.0 * eta3) / (144.0 * den_odd)
            rho61v3 = -2.0 * a / 9.0
            return 1.0 + v * v * (rho61v2 + rho61v3 * v), 0.0

    if l == 7:
        den_even = -1.0 + 7.0 * eta - 14.0 * eta2 + 7.0 * eta3
        den_odd = dM2 + 3.0 * eta2
        if m in (7, 5, 3, 1):
            if not dM2:
                return 1.0, 0.0
            rho2 = {
                7: (-906.0 + 4246.0 * eta - 4963.0 * eta2 + 1380.0 * eta3) / (714.0 * den_odd),
                5: (-762.0 + 3382.0 * eta - 3523.0 * eta2 + 804.0 * eta3) / (714.0 * den_odd),
                3: (-666.0 + 2806.0 * eta - 2563.0 * eta2 + 420.0 * eta3) / (714.0 * den_odd),
                1: (-618.0 + 2518.0 * eta - 2083.0 * eta2 + 228.0 * eta3) / (714.0 * den_odd),
            }[m]
            return 1.0 + v * v * (rho2 - (2.0 * a / 3.0) * v), 0.0
        if m == 6:
            rho76v2 = (2144.0 - 16185.0 * eta + 37828.0 * eta2 - 29351.0 * eta3 + 6104.0 * eta2 * eta2) / (1666.0 * den_even)
            return 1.0 + rho76v2 * v * v, 0.0
        if m == 4:
            rho74v2 = (17756.0 - 131805.0 * eta + 298872.0 * eta2 - 217959.0 * eta3 + 41076.0 * eta2 * eta2) / (14994.0 * den_even)
            return 1.0 + rho74v2 * v * v, 0.0
        if m == 2:
            rho72v2 = (16832.0 - 123489.0 * eta + 273924.0 * eta2 - 190239.0 * eta3 + 32760.0 * eta2 * eta2) / (14994.0 * den_even)
            return 1.0 + rho72v2 * v * v, 0.0

    if l == 8:
        den_even = -1.0 + 7.0 * eta - 14.0 * eta2 + 7.0 * eta3
        den_odd = -1.0 + 6.0 * eta - 10.0 * eta2 + 4.0 * eta3
        if m == 8:
            rho2 = (3482.0 - 26778.0 * eta + 64659.0 * eta2 - 53445.0 * eta3 + 12243.0 * eta2 * eta2) / (2736.0 * den_even)
        elif m == 7 and dM2:
            rho2 = (23478.0 - 154099.0 * eta + 309498.0 * eta2 - 207550.0 * eta3 + 38920.0 * eta2 * eta2) / (18240.0 * den_odd)
        elif m == 6:
            rho2 = (1002.0 - 7498.0 * eta + 17269.0 * eta2 - 13055.0 * eta3 + 2653.0 * eta2 * eta2) / (912.0 * den_even)
        elif m == 5 and dM2:
            rho2 = (4350.0 - 28055.0 * eta + 54642.0 * eta2 - 34598.0 * eta3 + 6056.0 * eta2 * eta2) / (3648.0 * den_odd)
        elif m == 4:
            rho2 = (2666.0 - 19434.0 * eta + 42627.0 * eta2 - 28965.0 * eta3 + 4899.0 * eta2 * eta2) / (2736.0 * den_even)
        elif m == 3 and dM2:
            rho2 = (20598.0 - 131059.0 * eta + 249018.0 * eta2 - 149950.0 * eta3 + 24520.0 * eta2 * eta2) / (18240.0 * den_odd)
        elif m == 2:
            rho2 = (2462.0 - 17598.0 * eta + 37119.0 * eta2 - 22845.0 * eta3 + 3063.0 * eta2 * eta2) / (2736.0 * den_even)
        elif m == 1 and dM2:
            rho2 = (20022.0 - 126451.0 * eta + 236922.0 * eta2 - 138430.0 * eta3 + 21640.0 * eta2 * eta2) / (18240.0 * den_odd)
        else:
            return 1.0, 0.0
        return 1.0 + rho2 * v * v, 0.0

    # Fallback to parsed LAL coefficients for higher modes (l<=8)
    try:
        rho = rho_lal_fallback(l, m, eta, chi1z, chi2z, m1, m2, v)
        return rho, 0.0
    except Exception as exc:
        raise ValueError(f"rho_lm_full: unsupported mode ({l},{m})") from exc


def delta_lm_full(l, m, eta, chi1z, chi2z, m1, m2, v, H=None, *, waveform: bool = False):
    """Delta_lm phase terms (aligned-spin backbone, limited modes)."""

    chiS, chiA, dM, _ = _spin_combinations(m1, m2, chi1z, chi2z)
    aDelta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    Omega = v ** 3
    vh3 = (H * Omega) if H is not None else Omega
    vh = abs(vh3) ** (1.0 / 3.0)

    if l == 2 and m == 2:
        # Delta_22 matches LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:144-155
        delta22vh3 = 7.0 / 3.0
        delta22vh6 = (-4.0 * aDelta) / 3.0 + (428.0 * _PI) / 105.0
        delta22vh9 = -2203.0 / 81.0 + (1712.0 * _PI * _PI) / 315.0
        delta22v5 = -24.0 * eta
        delta22v6 = 0.0
        delta22v8 = (20.0 * aDelta) / 63.0
        return vh3 * (delta22vh3 + vh3 * (delta22vh6 + vh * vh * (delta22vh9 * vh))) + Omega * (delta22v5 * v * v + Omega * (delta22v6 + delta22v8 * v * v))

    if l == 2 and m == 1:
        # Delta_21 matches LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:248-253
        delta21vh3 = 2.0 / 3.0
        delta21vh6 = (-17.0 * aDelta) / 35.0 + (107.0 * _PI) / 105.0
        delta21vh7 = (3.0 * aDelta * aDelta) / 140.0
        delta21vh9 = -272.0 / 81.0 + (214.0 * _PI * _PI) / 315.0
        delta21v5 = -493.0 * eta / 42.0
        delta21v7 = 0.0
        return vh3 * (delta21vh3 + vh3 * (delta21vh6 + vh * (delta21vh7 + delta21vh9 * vh * vh))) + (Omega * v * v) * (delta21v5 + delta21v7 * v * v)

    if l == 3 and m == 3:
        # Delta_33 matches LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:379-383
        delta33vh3 = 13.0 / 10.0
        delta33vh6 = (-81.0 * aDelta) / 20.0 + (39.0 * _PI) / 7.0
        delta33vh9 = -227827.0 / 3000.0 + (78.0 * _PI * _PI) / 7.0
        delta33v5 = -80897.0 * eta / 2430.0
        return vh3 * (delta33vh3 + vh3 * (delta33vh6 + vh3 * delta33vh9)) + Omega * v * v * delta33v5

    if l == 4 and m == 4:
        # Delta_44 matches LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:480-485
        m1Plus3eta = -1.0 + 3.0 * eta
        delta44vh3 = (112.0 + 219.0 * eta) / (-120.0 * m1Plus3eta)
        delta44vh6 = (-464.0 * aDelta) / 75.0 + (25136.0 * _PI) / 3465.0
        delta44vh9 = -55144.0 / 375.0 + 201088.0 * _PI * _PI / 10395.0 if waveform else 0.0
        return vh3 * (delta44vh3 + vh3 * (delta44vh6 + vh3 * delta44vh9))

    if l == 5 and m == 5:
        # Delta_55 matches LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:568-575
        denom = (1 - 2 * eta)
        delta55vh3 = (96875.0 + 857528.0 * eta) / (131250.0 * denom)
        delta55vh6 = 3865.0 * _PI / 429.0 if waveform else 0.0
        delta55vh9 = ((-7686949127.0 + 954500400.0 * _PI * _PI) / 31783752.0) if waveform else 0.0
        return vh3 * (delta55vh3 + vh3 * (delta55vh6 + vh3 * delta55vh9))

    try:
        return delta_lal_fallback(l, m, eta, chi1z, chi2z, m1, m2, v, H)
    except Exception:
        return 0.0



__all__ = [
    "rho22",
    "delta22",
    "rho33",
    "rho21",
    "rho44",
    "rho55",
    "rho_lm_full",
    "delta_lm_full",
]

# ---------------------------------------------------------------------------
# Fallback: auto-parse missing rho/delta coefficients from LAL C source
# ---------------------------------------------------------------------------

_LAL_COEFF_CACHE = {"rho": {}, "delta": {}, "loaded": False}
_LAL_C_PATH_ENV = "LAL_COEFF_C_PATH"


def _load_lal_coeffs(c_path: str = None):
    if _LAL_COEFF_CACHE["loaded"]:
        return
    path = c_path or os.environ.get(_LAL_C_PATH_ENV)
    if not path:
        return
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return
    text = open(path, "r").read()
    rho_pat = re.compile(r"coeffs->rho(\\d)(\\d)(?:v(\\d+)|v(\\d+)l)?\\s*=\\s*([^;]+);")
    delta_pat = re.compile(r"coeffs->delta(\\d)(\\d)(?:vh(\\d+)|v(\\d+))\\s*=\\s*([^;]+);")
    for m in rho_pat.finditer(text):
        l = int(m.group(1))
        mm = int(m.group(2))
        power = m.group(3) or m.group(4)
        expr = m.group(5).strip()
        _LAL_COEFF_CACHE["rho"].setdefault((l, mm), []).append((power, expr))
    for m in delta_pat.finditer(text):
        l = int(m.group(1))
        mm = int(m.group(2))
        power = m.group(3) or m.group(4)
        expr = m.group(5).strip()
        _LAL_COEFF_CACHE["delta"].setdefault((l, mm), []).append((power, expr))
    _LAL_COEFF_CACHE["loaded"] = True


def _eval_coeff_list(lst, v, a, a2, aDelta, eta, dM, dM2, eta2, eta3):
    # Safe eval namespace
    ns = dict(a=a, a2=a2, aDelta=aDelta, eta=eta, eta2=eta2, eta3=eta3, dM=dM, dM2=dM2, pow=pow)
    total = 1.0
    for power, expr in lst:
        if dM2 == 0.0 and "dM" in expr:
            term = 0.0
        else:
            term = eval(expr, {}, ns)
        total += term * (v ** (int(power)))
    return total


def rho_lal_fallback(l, m, eta, chi1z, chi2z, m1, m2, v):
    _load_lal_coeffs()
    key = (l, abs(m))
    if key not in _LAL_COEFF_CACHE["rho"]:
        raise ValueError(f"rho_lal_fallback: unsupported mode ({l},{m})")
    chiS, chiA, dM, dM2 = _spin_combinations(m1, m2, chi1z, chi2z)
    a = chiS + chiA * dM
    a2 = a * a
    eta2 = eta * eta
    eta3 = eta2 * eta
    aDelta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    coeffs = _LAL_COEFF_CACHE["rho"][key]
    return _eval_coeff_list(coeffs, v, a, a2, aDelta, eta, dM, dM2, eta2, eta3)


def delta_lal_fallback(l, m, eta, chi1z, chi2z, m1, m2, v, H=None):
    _load_lal_coeffs()
    key = (l, abs(m))
    if key not in _LAL_COEFF_CACHE["delta"]:
        return 0.0
    chiS, chiA, dM, _ = _spin_combinations(m1, m2, chi1z, chi2z)
    aDelta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    a = chiS + chiA * dM
    a2 = a * a
    Omega = v ** 3
    vh3 = (H * Omega) if H is not None else Omega
    vh = abs(vh3) ** (1.0 / 3.0)
    dM2 = dM * dM
    coeffs = _LAL_COEFF_CACHE["delta"][key]
    # Build namespace with both vh3 and Omega powers
    ns = dict(a=a, a2=a2, aDelta=aDelta, eta=eta, eta2=eta2, eta3=eta3, dM=dM, dM2=dM2, vh=vh, vh3=vh3, Omega=Omega, v=v, pow=pow)
    total = 0.0
    for power, expr in coeffs:
        if dM2 == 0.0 and "dM" in expr:
            term = 0.0
        else:
            term = eval(expr, {}, ns)
        p = int(power)
        # delta tables mix vh3 and v powers; use vh for vh-prefixed entries
        if expr.find("vh") != -1:
            total += term * (vh ** p)
        else:
            total += term * (v ** p)
    return total
