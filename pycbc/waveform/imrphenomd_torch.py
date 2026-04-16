# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch-native IMRPhenomD FD waveform (aligned-spin, dominant (2,2) mode).

This is a direct port of the reviewed LAL implementation
(``lalsimulation/lib/LALSimIMRPhenomD.c`` plus
``LALSimIMRPhenomD_internals.c``) with the physics kept intact but
implemented in pure Python/NumPy so it can run entirely inside the Torch
scheme without calling lalsimulation.

Activation
----------
- Per-model flag: ``PYCBC_IMRPHENOMD_NATIVE=1``
- Global flag   : ``PYCBC_TORCH_NATIVE_PORTS=1`` (acts as fallback)

When either flag enables the port *and* the current scheme is
``TorchScheme``, the torch-native path is taken. CPU/LAL remains the
default. The torch path is restricted to aligned-spin BBH (no tides, no
non-GR modifiers).

Limitations
-----------
- Tides/NRTidal and non-GR modifiers are not yet implemented.
- Only the dominant (2,2) mode is produced; higher modes follow the LAL
  default of zero for IMRPhenomD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as _np
import lal

from pycbc import pnutils, scheme as _scheme
from pycbc.types import FrequencySeries
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing

_PI = lal.PI
_MSUN_SI = lal.MSUN_SI
_MTSUN_SI = lal.MTSUN_SI
_MRSUN_SI = lal.MRSUN_SI

# Constants copied from LALSimIMRPhenomD.h
AMP_fJoin_INS = 0.014
PHI_fJoin_INS = 0.018
f_CUT = 0.2  # dimensionless (Mf) cutoff used by PhenomD

# For tiny safety when eta is exactly 0.25
_ETA_EPS = 1e-9


def _nudge_eta(eta: float) -> float:
    """Ensure eta <= 0.25 (mirrors PhenomInternal_nudge)."""
    if eta > 0.25:
        return 0.25 - _ETA_EPS
    return eta


@dataclass
class _Powers:
    """Common powers of an argument (pi or Mf)."""

    third: _np.ndarray
    two_thirds: _np.ndarray
    four_thirds: _np.ndarray
    five_thirds: _np.ndarray
    seven_thirds: _np.ndarray
    eight_thirds: _np.ndarray
    m_third: _np.ndarray
    m_two_thirds: _np.ndarray
    m_four_thirds: _np.ndarray
    m_five_thirds: _np.ndarray
    m_seven_sixths: _np.ndarray
    inv: _np.ndarray
    one: _np.ndarray
    two: _np.ndarray
    four: _np.ndarray


def _powers(x: _np.ndarray) -> _Powers:
    """Return useful powers of x; vectorised."""
    # Keep the inactive zero-frequency bin numerically harmless without letting
    # fractional powers underflow or inverse powers overflow.
    safe_floor = _np.finfo(float).eps
    x_safe = _np.where(x == 0.0, safe_floor, x)
    third = _np.cbrt(x_safe)
    two_thirds = third * third
    four_thirds = two_thirds * two_thirds
    five_thirds = x_safe * two_thirds
    seven_thirds = x_safe * x_safe * third
    eight_thirds = x_safe * x_safe * two_thirds
    m_third = 1.0 / third
    m_two_thirds = 1.0 / two_thirds
    m_four_thirds = 1.0 / four_thirds
    m_five_thirds = 1.0 / five_thirds
    m_seven_sixths = _np.power(x_safe, -7.0 / 6.0)
    inv = 1.0 / x_safe
    one = x
    two = x * x
    four = two * two
    return _Powers(
        third,
        two_thirds,
        four_thirds,
        five_thirds,
        seven_thirds,
        eight_thirds,
        m_third,
        m_two_thirds,
        m_four_thirds,
        m_five_thirds,
        m_seven_sixths,
        inv,
        one,
        two,
        four,
    )


@lru_cache(None)
def _pi_powers() -> _Powers:
    return _powers(_np.array(_PI))


def _select_powers(p: _Powers, mask):
    """Return a shallow copy of powers with each array masked."""
    return _Powers(
        p.third[mask],
        p.two_thirds[mask],
        p.four_thirds[mask],
        p.five_thirds[mask],
        p.seven_thirds[mask],
        p.eight_thirds[mask],
        p.m_third[mask],
        p.m_two_thirds[mask],
        p.m_four_thirds[mask],
        p.m_five_thirds[mask],
        p.m_seven_sixths[mask],
        p.inv[mask],
        p.one[mask],
        p.two[mask],
        p.four[mask],
    )


# ---------- Final spin and QNM fits -------------------------------------------------


def _final_spin0815(eta: float, chi1: float, chi2: float) -> float:
    """Final spin fit (Eq. 3.6 of arXiv:1508.07250, FinalSpin0815_s)."""
    eta2 = eta * eta
    eta3 = eta2 * eta
    seta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1 = 0.5 * (1.0 + seta)
    m2 = 0.5 * (1.0 - seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = m1s * chi1 + m2s * chi2
    s2 = s * s
    s3 = s2 * s
    return eta * (
        3.4641016151377544
        - 4.399247300629289 * eta
        + 9.397292189321194 * eta2
        - 13.180949901606242 * eta3
        + s
        * (
            (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta)
            + (0.1014665242971878 - 2.0967746996832157 * eta) * s
            + (-1.3546806617824356 + 4.108962025369336 * eta) * s2
            + (-0.8676969352555539 + 2.064046835273906 * eta) * s3
        )
    )


@lru_cache(None)
def _load_qnm_tables() -> Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Parse QNM data arrays from the bundled LAL header."""
    # Prefer the reference checkout if available; fallback to installed header.
    candidates = [
        Path("/Users/xangma/repos/lalsuite/lalsimulation/lib/LALSimIMRPhenomD.h"),
        Path(__file__).resolve().parent / "data" / "LALSimIMRPhenomD.h",
    ]
    header = next((p for p in candidates if p.exists()), None)
    if header is None:
        raise FileNotFoundError("LALSimIMRPhenomD.h not found; cannot load QNM tables")

    def _extract_array(name: str):
        text = header.read_text()
        start = text.index(f"static const double {name}[]")
        chunk = text[start:].split("{", 1)[1].split("}", 1)[0]
        vals = [float(v.replace("L", "")) for v in chunk.replace("\\\n", "").replace("\n", " ").split(",") if v.strip()]
        return _np.array(vals, dtype=_np.float64)

    a = _extract_array("QNMData_a")
    fring = _extract_array("QNMData_fring")
    fdamp = _extract_array("QNMData_fdamp")
    return a, fring, fdamp


def _interp_qnm(a: float) -> Tuple[float, float]:
    """Spline-free (linear) interpolation of fring, fdamp vs final spin."""
    xs, ys_f, ys_d = _load_qnm_tables()
    fring = _np.interp(a, xs, ys_f)
    fdamp = _np.interp(a, xs, ys_d)
    return float(fring), float(fdamp)


def _erad_rational_0815(eta: float, chi1: float, chi2: float) -> float:
    """Radiated energy fit from arXiv:1508.07250 (matches PhenomInternal_EradRational0815)."""
    seta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    m1 = 0.5 * (1.0 + seta)
    m2 = 0.5 * (1.0 - seta)
    m1s = m1 * m1
    m2s = m2 * m2
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)
    eta2 = eta * eta
    eta3 = eta2 * eta
    num = eta * (
        0.055974469826360077
        + 0.5809510763115132 * eta
        - 0.9606726679372312 * eta2
        + 3.352411249771192 * eta3
    ) * (1.0 + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta2) * s)
    den = 1.0 + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta2) * s
    return num / den


# ---------- Amplitude pieces --------------------------------------------------------


def _amp0(eta: float) -> float:
    return math.sqrt(2.0 / 3.0 * eta) * (math.pi ** (-1.0 / 6.0))


def _rho1(eta, eta2, xi):
    return (
        3931.8979897196696
        - 17395.758706812805 * eta
        + (
            3132.375545898835
            + 343965.86092361377 * eta
            - 1.2162565819981997e6 * eta2
            + (-70698.00600428853 + 1.383907177859705e6 * eta - 3.9662761890979446e6 * eta2) * xi
            + (-60017.52423652596 + 803515.1181825735 * eta - 2.091710365941658e6 * eta2) * xi * xi
        )
        * xi
    )


def _rho2(eta, eta2, xi):
    return (
        -40105.47653771657
        + 112253.0169706701 * eta
        + (
            23561.696065836168
            - 3.476180699403351e6 * eta
            + 1.137593670849482e7 * eta2
            + (754313.1127166454 - 1.308476044625268e7 * eta + 3.6444584853928134e7 * eta2) * xi
            + (596226.612472288 - 7.4277901143564405e6 * eta + 1.8928977514040343e7 * eta2) * xi * xi
        )
        * xi
    )


def _rho3(eta, eta2, xi):
    return (
        83208.35471266537
        - 191237.7264145924 * eta
        + (
            -210916.2454782992
            + 8.71797508352568e6 * eta
            - 2.6914942420669552e7 * eta2
            + (-1.9889806527362722e6 + 3.0888029960154563e7 * eta - 8.390870279256162e7 * eta2) * xi
            + (-1.4535031953446497e6 + 1.7063528990822166e7 * eta - 4.2748659731120914e7 * eta2) * xi * xi
        )
        * xi
    )


def _gamma1(eta, eta2, xi):
    return (
        0.006927402739328343
        + 0.03020474290328911 * eta
        + (
            0.006308024337706171
            - 0.12074130661131138 * eta
            + 0.26271598905781324 * eta2
            + (0.0034151773647198794 - 0.10779338611188374 * eta + 0.27098966966891747 * eta2) * xi
            + (0.0007374185938559283 - 0.02749621038376281 * eta + 0.0733150789135702 * eta2) * xi * xi
        )
        * xi
    )


def _gamma2(eta, eta2, xi):
    return (
        1.010344404799477
        + 0.0008993122007234548 * eta
        + (
            0.283949116804459
            - 4.049752962958005 * eta
            + 13.207828172665366 * eta2
            + (0.10396278486805426 - 7.025059158961947 * eta + 24.784892370130475 * eta2) * xi
            + (0.03093202475605892 - 2.6924023896851663 * eta + 9.609374464684983 * eta2) * xi * xi
        )
        * xi
    )


def _gamma3(eta, eta2, xi):
    return (
        1.3081615607036106
        - 0.005537729694807678 * eta
        + (
            -0.06782917938621007
            - 0.6689834970767117 * eta
            + 3.403147966134083 * eta2
            + (-0.05296577374411866 - 0.9923793203111362 * eta + 4.820681208409587 * eta2) * xi
            + (-0.006134139870393713 - 0.38429253308696365 * eta + 1.7561754421985984 * eta2) * xi * xi
        )
        * xi
    )


def _amp_int_col_fit(eta, eta2, chi):
    xi = -1.0 + chi
    return (
        0.8149838730507785
        + 2.5747553517454658 * eta
        + (
            1.1610198035496786
            - 2.3627771785551537 * eta
            + 6.771038707057573 * eta2
            + (0.7570782938606834 - 2.7256896890432474 * eta + 7.1140380397149965 * eta2) * xi
            + (0.1766934149293479 - 0.7978690983168183 * eta + 2.1162391502005153 * eta2) * xi * xi
        )
        * xi
    )


@dataclass
class _AmpCoeffs:
    eta: float
    chi1: float
    chi2: float
    chi: float
    eta2: float
    eta3: float
    chi12: float
    chi22: float
    seta: float
    seta_plus1: float
    rho1: float
    rho2: float
    rho3: float
    gamma1: float
    gamma2: float
    gamma3: float
    fRD: float
    fDM: float
    fmaxCalc: float
    amp0: float
    f1: float
    f2: float
    f3: float
    v1: float
    v2: float
    v3: float
    d1: float
    d2: float
    deltas: _np.ndarray


def _compute_amp_coeffs(eta: float, chi1: float, chi2: float, finspin: float) -> _AmpCoeffs:
    eta = _nudge_eta(eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    seta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    seta_plus1 = 1.0 + seta
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2
    q = 0.5 * (1.0 + seta - 2.0 * eta) / eta
    chi = 0.5 * ((chi1 + chi2) * (1.0 - 76.0 * eta / 113.0) + seta * (chi1 - chi2))
    xi = -1.0 + chi

    rho1 = _rho1(eta, eta2, xi)
    rho2 = _rho2(eta, eta2, xi)
    rho3 = _rho3(eta, eta2, xi)

    gamma1 = _gamma1(eta, eta2, xi)
    gamma2 = _gamma2(eta, eta2, xi)
    gamma3 = _gamma3(eta, eta2, xi)

    fRD, fDM = _interp_qnm(finspin)
    erad = _erad_rational_0815(eta, chi1, chi2)
    fRD /= 1.0 - erad
    fDM /= 1.0 - erad

    fmaxCalc = _fmax_calc(fRD, fDM, gamma2, gamma3)

    amp0 = _amp0(eta)

    # Collocation setup
    f1 = AMP_fJoin_INS
    f3 = fmaxCalc
    f2 = f1 + 0.5 * (f3 - f1)
    powers_f1 = _powers(_np.array(f1))
    amp_pref = _init_amp_prefactors(
        eta,
        eta2,
        eta3,
        chi1,
        chi2,
        chi12,
        chi22,
        seta,
        seta_plus1,
        gamma1,
        gamma2,
        gamma3,
        rho1,
        rho2,
        rho3,
    )
    v1 = _amp_ins_ansatz(_np.array(f1), powers_f1, amp_pref)
    d1 = _d_amp_ins_ansatz(_np.array(f1), amp_pref)
    v3 = _amp_mrd_ansatz(_np.array(f3), fRD, fDM, gamma1, gamma2, gamma3)
    d2 = _d_amp_mrd_ansatz(_np.array(f3), fRD, fDM, gamma1, gamma2, gamma3)
    v2 = _amp_int_col_fit(eta, eta2, chi)

    deltas = _solve_intermediate_polynomial(f1, f2, f3, v1, v2, v3, d1, d2)

    return _AmpCoeffs(
        eta,
        chi1,
        chi2,
        chi,
        eta2,
        eta3,
        chi12,
        chi22,
        seta,
        seta_plus1,
        rho1,
        rho2,
        rho3,
        gamma1,
        gamma2,
        gamma3,
        fRD,
        fDM,
        fmaxCalc,
        amp0,
        f1,
        f2,
        f3,
        v1,
        v2,
        v3,
        d1,
        d2,
        deltas,
    )


def _fmax_calc(fRD, fDM, gamma2, gamma3):
    if gamma2 < 1.0:
        return abs(fRD + (fDM * (-1.0 + math.sqrt(1.0 - gamma2 * gamma2)) * gamma3) / gamma2)
    return abs(fRD - (fDM * gamma3) / gamma2)


@dataclass
class _AmpPrefactors:
    amp0: float
    two_thirds: float
    one: float
    four_thirds: float
    five_thirds: float
    two: float
    seven_thirds: float
    eight_thirds: float
    three: float


def _init_amp_prefactors(
    eta,
    eta2,
    eta3,
    chi1,
    chi2,
    chi12,
    chi22,
    seta,
    seta_plus1,
    gamma1,
    gamma2,
    gamma3,
    rho1,
    rho2,
    rho3,
):
    pi = _PI
    pi2 = pi * pi
    pow_pi = _pi_powers()
    return _AmpPrefactors(
        amp0=_amp0(eta),
        two_thirds=((-969 + 1804 * eta) * pow_pi.two_thirds) / 672.0,
        one=((chi1 * (81 * seta_plus1 - 44 * eta) + chi2 * (81 - 81 * seta - 44 * eta)) * pi) / 48.0,
        four_thirds=(
            (-27312085.0 - 10287648 * chi22 - 10287648 * chi12 * seta_plus1 + 10287648 * chi22 * seta)
            + 24 * (-1975055 + 857304 * chi12 - 994896 * chi1 * chi2 + 857304 * chi22) * eta
            + 35371056 * eta2
        )
        * pow_pi.four_thirds
        / 8.128512e6,
        five_thirds=pow_pi.five_thirds
        * (
            chi2 * (-285197 * (-1 + seta) + 4 * (-91902 + 1579 * seta) * eta - 35632 * eta2)
            + chi1 * (285197 * seta_plus1 - 4 * (91902 + 1579 * seta) * eta - 35632 * eta2)
            + 42840 * (-1.0 + 4 * eta) * pi
        )
        / 32256.0,
        two=(
            -pi2
            * (
                -336
                * (-3248849057.0 + 2943675504 * chi12 - 3339284256 * chi1 * chi2 + 2943675504 * chi22)
                * eta2
                - 324322727232 * eta3
                - 7
                * (
                    -177520268561
                    + 107414046432 * chi22
                    + 107414046432 * chi12 * seta_plus1
                    - 107414046432 * chi22 * seta
                    + 11087290368 * (chi1 + chi2 + chi1 * seta - chi2 * seta) * pi
                )
                + 12
                * eta
                * (
                    -545384828789.0
                    - 176491177632 * chi1 * chi2
                    + 202603761360 * chi22
                    + 77616 * chi12 * (2610335 + 995766 * seta)
                    - 77287373856 * chi22 * seta
                    + 5841690624 * (chi1 + chi2) * pi
                    + 21384760320 * pi2
                )
            )
        )
        / 6.0085960704e10,
        seven_thirds=rho1,
        eight_thirds=rho2,
        three=rho3,
    )


def _amp_ins_ansatz(Mf, powers: _Powers, pref: _AmpPrefactors):
    return (
        1
        + powers.two_thirds * pref.two_thirds
        + powers.four_thirds * pref.four_thirds
        + powers.five_thirds * pref.five_thirds
        + powers.seven_thirds * pref.seven_thirds
        + powers.eight_thirds * pref.eight_thirds
        + Mf * (pref.one + Mf * pref.two + powers.two * pref.three)
    )


def _d_amp_ins_ansatz(Mf, pref: _AmpPrefactors):
    powers = _powers(Mf)
    return (
        (2.0 / 3.0) * pref.two_thirds * powers.m_third
        + (4.0 / 3.0) * pref.four_thirds * powers.third
        + (5.0 / 3.0) * pref.five_thirds * powers.two_thirds
        + (7.0 / 3.0) * pref.seven_thirds * powers.four_thirds
        + (8.0 / 3.0) * pref.eight_thirds * powers.five_thirds
        + pref.one
        + 2.0 * pref.two * powers.one
        + 3.0 * pref.three * powers.two
    )


def _amp_mrd_ansatz(f, fRD, fDM, gamma1, gamma2, gamma3):
    fDMgamma3 = fDM * gamma3
    fminfRD = f - fRD
    return _np.exp(-(fminfRD) * gamma2 / (fDMgamma3)) * (fDMgamma3 * gamma1) / (
        fminfRD * fminfRD + fDMgamma3 * fDMgamma3
    )


def _d_amp_mrd_ansatz(f, fRD, fDM, gamma1, gamma2, gamma3):
    fDMgamma3 = fDM * gamma3
    pow2 = fDMgamma3 * fDMgamma3
    fminfRD = f - fRD
    expfac = _np.exp(((fminfRD) * gamma2) / (fDMgamma3))
    denom = fminfRD * fminfRD + pow2
    return ((-2 * fDM * fminfRD * gamma3 * gamma1) / denom - (gamma2 * gamma1)) / (expfac * denom)


def _solve_intermediate_polynomial(f1, f2, f3, v1, v2, v3, d1, d2):
    A = _np.array(
        [
            [1, f1, f1 * f1, f1 ** 3, f1 ** 4],
            [0, 1, 2 * f1, 3 * f1 * f1, 4 * f1 ** 3],
            [1, f2, f2 * f2, f2 ** 3, f2 ** 4],
            [1, f3, f3 * f3, f3 ** 3, f3 ** 4],
            [0, 1, 2 * f3, 3 * f3 * f3, 4 * f3 ** 3],
        ],
        dtype=_np.float64,
    )
    b = _np.array([v1, d1, v2, v3, d2], dtype=_np.float64)
    return _np.linalg.solve(A, b)


def _amp_int_ansatz(Mf, deltas):
    # deltas: delta0..delta4
    Mf2 = Mf * Mf
    return deltas[0] + Mf * deltas[1] + Mf2 * (deltas[2] + Mf * deltas[3] + deltas[4] * Mf2)


# ---------- Phase pieces ------------------------------------------------------------


def _sigma1(eta, eta2, xi):
    return (
        2096.551999295543
        + 1463.7493168261553 * eta
        + (
            1312.5493286098522
            + 18307.330017082117 * eta
            - 43534.1440746107 * eta2
            + (-833.2889543511114 + 32047.31997183187 * eta - 108609.45037520859 * eta2) * xi
            + (452.25136398112204 + 8353.439546391714 * eta - 44531.3250037322 * eta2) * xi * xi
        )
        * xi
    )


def _sigma2(eta, eta2, xi):
    return (
        -10114.056472621156
        - 44631.01109458185 * eta
        + (
            -6541.308761668722
            - 266959.23419307504 * eta
            + 686328.3229317984 * eta2
            + (3405.6372187679685 - 437507.7208209015 * eta + 1.6318171307344697e6 * eta2) * xi
            + (-7462.648563007646 - 114585.25177153319 * eta + 674402.4689098676 * eta2) * xi * xi
        )
        * xi
    )


def _sigma3(eta, eta2, xi):
    return (
        22933.658273436497
        + 230960.00814979506 * eta
        + (
            14961.083974183695
            + 1.1940181342318142e6 * eta
            - 3.1042239693052764e6 * eta2
            + (-3038.166617199259 + 1.8720322849093592e6 * eta - 7.309145012085539e6 * eta2) * xi
            + (42738.22871475411 + 467502.018616601 * eta - 3.064853498512499e6 * eta2) * xi * xi
        )
        * xi
    )


def _sigma4(eta, eta2, xi):
    return (
        -14621.71522218357
        - 377812.8579387104 * eta
        + (
            -9608.682631509726
            - 1.7108925257214056e6 * eta
            + 4.332924601416521e6 * eta2
            + (-22366.683262266528 - 2.5019716386377467e6 * eta + 1.0274495902259542e7 * eta2) * xi
            + (-85360.30079034246 - 570025.3441737515 * eta + 4.396844346849777e6 * eta2) * xi * xi
        )
        * xi
    )


def _beta1(eta, eta2, xi):
    return (
        97.89747327985583
        - 42.659730877489224 * eta
        + (
            153.48421037904913
            - 1417.0620760768954 * eta
            + 2752.8614143665027 * eta2
            + (138.7406469558649 - 1433.6585075135881 * eta + 2857.7418952430758 * eta2) * xi
            + (41.025109467376126 - 423.680737974639 * eta + 850.3594335657173 * eta2) * xi * xi
        )
        * xi
    )


def _beta2(eta, eta2, xi):
    return (
        -3.282701958759534
        - 9.051384468245866 * eta
        + (
            -12.415449742258042
            + 55.4716447709787 * eta
            - 106.05109938966335 * eta2
            + (-11.953044553690658 + 76.80704618365418 * eta - 155.33172948098394 * eta2) * xi
            + (-3.4129261592393263 + 25.572377569952536 * eta - 54.408036707740465 * eta2) * xi * xi
        )
        * xi
    )


def _beta3(eta, eta2, xi):
    return (
        -0.000025156429818799565
        + 0.000019750256942201327 * eta
        + (
            -0.000018370671469295915
            + 0.000021886317041311973 * eta
            + 0.00008250240316860033 * eta2
            + (7.157371250566708e-6 - 0.000055780000112270685 * eta + 0.00019142082884072178 * eta2) * xi
            + (5.447166261464217e-6 - 0.00003220610095021982 * eta + 0.00007974016714984341 * eta2) * xi * xi
        )
        * xi
    )


def _alpha1(eta, eta2, xi):
    return (
        43.31514709695348
        + 638.6332679188081 * eta
        + (
            -32.85768747216059
            + 2415.8938269370315 * eta
            - 5766.875169379177 * eta2
            + (-61.85459307173841 + 2953.967762459948 * eta - 8986.29057591497 * eta2) * xi
            + (-21.571435779762044 + 981.2158224673428 * eta - 3239.5664895930286 * eta2) * xi * xi
        )
        * xi
    )


def _alpha2(eta, eta2, xi):
    return (
        -0.07020209449091723
        - 0.16269798450687084 * eta
        + (
            -0.1872514685185499
            + 1.138313650449945 * eta
            - 2.8334196304430046 * eta2
            + (-0.17137955686840617 + 1.7197549338119527 * eta - 4.539717148261272 * eta2) * xi
            + (-0.049983437357548705 + 0.6062072055948309 * eta - 1.682769616644546 * eta2) * xi * xi
        )
        * xi
    )


def _alpha3(eta, eta2, xi):
    return (
        9.5988072383479
        - 397.05438595557433 * eta
        + (
            16.202126189517813
            - 1574.8286986717037 * eta
            + 3600.3410843831093 * eta2
            + (27.092429659075467 - 1786.482357315139 * eta + 5152.919378666511 * eta2) * xi
            + (11.175710130033895 - 577.7999423177481 * eta + 1808.730762932043 * eta2) * xi * xi
        )
        * xi
    )


def _alpha4(eta, eta2, xi):
    return (
        -0.02989487384493607
        + 1.4022106448583738 * eta
        + (
            -0.07356049468633846
            + 0.8337006542278661 * eta
            + 0.2240008282397391 * eta2
            + (-0.055202870001177226 + 0.5667186343606578 * eta + 0.7186931973380503 * eta2) * xi
            + (-0.015507437354325743 + 0.15750322779277187 * eta + 0.21076815715176228 * eta2) * xi * xi
        )
        * xi
    )


def _alpha5(eta, eta2, xi):
    return (
        0.9974408278363099
        - 0.007884449714907203 * eta
        + (
            -0.059046901195591035
            + 1.3958712396764088 * eta
            - 4.516631601676276 * eta2
            + (-0.05585343136869692 + 1.7516580039343603 * eta - 5.990208965347804 * eta2) * xi
            + (-0.017945336522161195 + 0.5965097794825992 * eta - 2.0608879367971804 * eta2) * xi * xi
        )
        * xi
    )


@dataclass
class _PhaseCoeffs:
    eta: float
    eta_inv: float
    chi1: float
    chi2: float
    eta2: float
    seta: float
    chi: float
    sigma1: float
    sigma2: float
    sigma3: float
    sigma4: float
    beta1: float
    beta2: float
    beta3: float
    alpha1: float
    alpha2: float
    alpha3: float
    alpha4: float
    alpha5: float
    fRD: float
    fDM: float
    fInsJoin: float
    fMRDJoin: float
    C1Int: float
    C2Int: float
    C1MRD: float
    C2MRD: float


def _compute_phase_coeffs(eta: float, chi1: float, chi2: float, finspin: float, pn) -> _PhaseCoeffs:
    eta = _nudge_eta(eta)
    eta2 = eta * eta
    eta_inv = 1.0 / eta
    seta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    chi = 0.5 * ((chi1 + chi2) * (1.0 - 76.0 * eta / 113.0) + seta * (chi1 - chi2))
    xi = -1.0 + chi

    sigma1 = _sigma1(eta, eta2, xi)
    sigma2 = _sigma2(eta, eta2, xi)
    sigma3 = _sigma3(eta, eta2, xi)
    sigma4 = _sigma4(eta, eta2, xi)

    beta1 = _beta1(eta, eta2, xi)
    beta2 = _beta2(eta, eta2, xi)
    beta3 = _beta3(eta, eta2, xi)

    alpha1 = _alpha1(eta, eta2, xi)
    alpha2 = _alpha2(eta, eta2, xi)
    alpha3 = _alpha3(eta, eta2, xi)
    alpha4 = _alpha4(eta, eta2, xi)
    alpha5 = _alpha5(eta, eta2, xi)

    fRD, fDM = _interp_qnm(finspin)
    erad = _erad_rational_0815(eta, chi1, chi2)
    fRD /= 1.0 - erad
    fDM /= 1.0 - erad

    # Continuity coefficients
    pow_pi = _pi_powers()
    phi_pref = _init_phi_prefactors(sigma1, sigma2, sigma3, sigma4, pn, pow_pi)
    fInsJoin = PHI_fJoin_INS
    fMRDJoin = 0.5 * fRD
    C2Int = _d_phi_ins(fInsJoin, sigma1, sigma2, sigma3, sigma4, pn, pow_pi, eta_inv) - _d_phi_int(
        fInsJoin, beta1, beta2, beta3, eta_inv
    )
    powers_fIns = _powers(_np.array(fInsJoin))
    C1Int = (
        _phi_ins(fInsJoin, powers_fIns, phi_pref, sigma1, sigma2, sigma3, sigma4, pn, pow_pi, eta_inv)
        - eta_inv * _phi_int(fInsJoin, beta1, beta2, beta3)
        - C2Int * fInsJoin
    )

    PhiIntTempVal = eta_inv * _phi_int(fMRDJoin, beta1, beta2, beta3) + C1Int + C2Int * fMRDJoin
    DPhiIntTempVal = C2Int + (beta1 + beta3 / (fMRDJoin ** 4) + beta2 / fMRDJoin) * eta_inv
    DPhiMRDVal = _d_phi_mrd(fMRDJoin, alpha1, alpha2, alpha3, alpha4, alpha5, fRD, fDM, eta_inv)
    C2MRD = DPhiIntTempVal - DPhiMRDVal
    PhiMRDJoin = _phi_mrd(fMRDJoin, alpha1, alpha2, alpha3, alpha4, alpha5, fRD, fDM)
    C1MRD = PhiIntTempVal - eta_inv * PhiMRDJoin - C2MRD * fMRDJoin

    return _PhaseCoeffs(
        eta,
        eta_inv,
        chi1,
        chi2,
        eta2,
        seta,
        chi,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
        beta1,
        beta2,
        beta3,
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        alpha5,
        fRD,
        fDM,
        fInsJoin,
        fMRDJoin,
        C1Int,
        C2Int,
        C1MRD,
        C2MRD,
    )


@dataclass
class _PhiPref:
    initial_phasing: float
    two_thirds: float
    third: float
    third_logv: float
    logv: float
    minus_third: float
    minus_two_thirds: float
    minus_one: float
    minus_four_thirds: float
    minus_five_thirds: float
    one: float
    four_thirds: float
    five_thirds: float
    two: float


def _init_phi_prefactors(sigma1, sigma2, sigma3, sigma4, pn, pow_pi: _Powers) -> _PhiPref:
    return _PhiPref(
        initial_phasing=pn.v[5] - 0.25 * _PI,
        two_thirds=pn.v[7] * pow_pi.two_thirds,
        third=pn.v[6] * pow_pi.third,
        third_logv=pn.vlogv[6] * pow_pi.third,
        logv=pn.vlogv[5],
        minus_third=pn.v[4] * pow_pi.m_third,
        minus_two_thirds=pn.v[3] * pow_pi.m_two_thirds,
        minus_one=pn.v[2] * pow_pi.inv,
        minus_four_thirds=pn.v[1] / pow_pi.four_thirds,
        minus_five_thirds=pn.v[0] * pow_pi.m_five_thirds,
        one=sigma1,
        four_thirds=0.75 * sigma2,
        five_thirds=0.6 * sigma3,
        two=0.5 * sigma4,
    )


def _phi_ins(Mf, powers: _Powers, pref: _PhiPref, sigma1, sigma2, sigma3, sigma4, pn, pow_pi: _Powers, eta_inv: float):
    v = powers.third * pow_pi.third
    logv = _np.log(v)
    phasing = pref.initial_phasing
    phasing += pref.two_thirds * powers.two_thirds
    phasing += pref.third * powers.third
    phasing += pref.third_logv * logv * powers.third
    phasing += pref.logv * logv
    phasing += pref.minus_third * powers.m_third
    phasing += pref.minus_two_thirds * powers.m_two_thirds
    phasing += pref.minus_one * powers.inv
    phasing += pref.minus_four_thirds * powers.m_four_thirds
    phasing += pref.minus_five_thirds * powers.m_five_thirds
    phasing += eta_inv * (
        pref.one * Mf + pref.four_thirds * powers.four_thirds + pref.five_thirds * powers.five_thirds + pref.two * powers.two
    )
    return phasing


def _d_phi_ins(Mf, sigma1, sigma2, sigma3, sigma4, pn, pow_pi: _Powers, eta_inv: float):
    v = _np.cbrt(_PI * Mf)
    logv = _np.log(v)
    v2 = v * v
    v3 = v * v2
    v4 = v * v3
    v5 = v * v4
    v6 = v * v5
    v7 = v * v6
    v8 = v * v7

    Dphasing = 0.0
    Dphasing += 2.0 * pn.v[7] * v7
    Dphasing += (pn.v[6] + pn.vlogv[6] * (1.0 + logv)) * v6
    Dphasing += pn.vlogv[5] * v5
    Dphasing += -1.0 * pn.v[4] * v4
    Dphasing += -2.0 * pn.v[3] * v3
    Dphasing += -3.0 * pn.v[2] * v2
    Dphasing += -4.0 * pn.v[1] * v
    Dphasing += -5.0 * pn.v[0]
    Dphasing /= v8 * 3.0
    Dphasing *= _PI
    Dphasing += (sigma1 + sigma2 * v * pow_pi.m_third + sigma3 * v2 * pow_pi.m_two_thirds + (sigma4 * pow_pi.inv) * v3) * eta_inv
    return Dphasing


def _phi_int(Mf, beta1, beta2, beta3):
    return beta1 * Mf - beta3 / (3.0 * (Mf ** 3)) + beta2 * _np.log(Mf)


def _d_phi_int(Mf, beta1, beta2, beta3, eta_inv):
    return eta_inv * (beta1 + beta3 / (Mf ** 4) + beta2 / Mf)


def _phi_mrd(f, alpha1, alpha2, alpha3, alpha4, alpha5, fRD, fDM):
    sqrootf = _np.sqrt(f)
    fpow1_5 = f * sqrootf
    fpow0_75 = _np.sqrt(fpow1_5)
    return -(alpha2 / f) + (4.0 / 3.0) * (alpha3 * fpow0_75) + alpha1 * f + alpha4 * _np.arctan((f - alpha5 * fRD) / (fDM))


def _d_phi_mrd(f, alpha1, alpha2, alpha3, alpha4, alpha5, fRD, fDM, eta_inv):
    return eta_inv * (alpha1 + alpha2 / (f * f) + alpha3 / (f ** 0.25) + alpha4 / (fDM * (1 + ((f - alpha5 * fRD) ** 2) / (fDM * fDM))))


# ---------- PN helpers --------------------------------------------------------------


def _subtract_3pn_ss(m1, m2, M, eta, chi1, chi2):
    m1M = m1 / M
    m2M = m2 / M
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M) + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)) * m1M * m1M * chi1 * chi1
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M) + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)) * m2M * m2M * chi2 * chi2
    return pn_ss3


# ---------- Main waveform -----------------------------------------------------------


def imrphenomd_fd_torch(**p):
    """Torch-native IMRPhenomD FD waveform (hp, hc)."""
    # ensure m1 >= m2 per LAL convention
    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z

    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    f_ref = float(p.get("f_ref", 0.0))
    distance = pnutils.megaparsecs_to_meters(float(p["distance"]))
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))

    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    eta = _nudge_eta(eta)
    M_sec = M * _MTSUN_SI

    # Determine termination frequency
    f_cut_hz = f_CUT / M_sec
    f_max_prime = f_final if f_final > 0 else f_cut_hz
    f_max_prime = min(f_max_prime, f_cut_hz)
    if f_max_prime <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    # Frequency grid (power-of-two length like LAL)
    npts = int(_np.ceil(2 ** _np.ceil(_np.log2(f_max_prime / delta_f)))) + 1
    freqs = _np.arange(npts, dtype=_np.float64) * delta_f
    kmin = int(_np.ceil(f_lower / delta_f))
    kmax = int(_np.floor(f_max_prime / delta_f))
    active = (freqs >= f_lower) & (freqs <= f_max_prime)

    finspin = _final_spin0815(eta, spin1z, spin2z)
    amp_coeffs = _compute_amp_coeffs(eta, spin1z, spin2z, finspin)

    # PN phasing series (reuse TaylorF2 port)
    pn = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=-1,
        tidal_order=-1,
        dchi={},
        qm_def1=0.0,
        qm_def2=0.0,
        lambda1=0.0,
        lambda2=0.0,
    )
    # Subtract 3PN spin-spin term (matches LAL tuning)
    pn.v[6] -= _subtract_3pn_ss(mass1, mass2, M, eta, spin1z, spin2z) * pn.v[0]
    phase_coeffs = _compute_phase_coeffs(eta, spin1z, spin2z, finspin, pn)
    phi_pref = _init_phi_prefactors(
        phase_coeffs.sigma1,
        phase_coeffs.sigma2,
        phase_coeffs.sigma3,
        phase_coeffs.sigma4,
        pn,
        _pi_powers(),
    )

    # Pre-compute reference phase
    MfRef = M_sec * (f_ref if f_ref > 0 else f_lower)
    phifRef = _IMRPhenDPhase(
        _np.array([MfRef]),
        phase_coeffs,
        pn,
        phi_pref,
        _pi_powers(),
    )[0]
    phi_precalc = 2.0 * coa_phase + phifRef

    t0 = _d_phi_mrd(
        amp_coeffs.fmaxCalc,
        phase_coeffs.alpha1,
        phase_coeffs.alpha2,
        phase_coeffs.alpha3,
        phase_coeffs.alpha4,
        phase_coeffs.alpha5,
        phase_coeffs.fRD,
        phase_coeffs.fDM,
        phase_coeffs.eta_inv,
    )

    Mf = M_sec * freqs
    powers = _powers(Mf)

    # Amplitude
    amp = _IMRPhenDAmplitude(Mf, amp_coeffs, powers)

    # Phase
    phase = _IMRPhenDPhase(Mf, phase_coeffs, pn, phi_pref, _pi_powers())
    phase -= t0 * (Mf - MfRef) + phi_precalc

    # Zero outside active band
    amp = _np.where(active, amp, 0.0)
    phase = _np.where(active, phase, 0.0)

    # Overall scaling (amp0 uses masses in SI)
    amp_scale = 2.0 * math.sqrt(5.0 / (64.0 * _PI)) * M * _MRSUN_SI * M_sec / distance
    h22 = amp_scale * amp * _np.exp(-1j * phase)

    cosi = math.cos(inclination)
    hp = h22 * 0.5 * (1.0 + cosi * cosi)
    hc = -1j * h22 * cosi

    hp_fs = FrequencySeries(hp, delta_f=delta_f)
    hc_fs = FrequencySeries(hc, delta_f=delta_f)
    return hp_fs, hc_fs


def _IMRPhenDAmplitude(Mf, a: _AmpCoeffs, powers: _Powers):
    # Piecewise amplitude
    amp_pref = _init_amp_prefactors(
        a.eta,
        a.eta2,
        a.eta3,
        a.chi1,
        a.chi2,
        a.chi12,
        a.chi22,
        a.seta,
        a.seta_plus1,
        a.gamma1,
        a.gamma2,
        a.gamma3,
        a.rho1,
        a.rho2,
        a.rho3,
    )
    res = _np.empty_like(Mf)
    mask_ins = Mf < AMP_fJoin_INS
    mask_mrd = Mf >= a.fmaxCalc
    mask_int = (~mask_ins) & (~mask_mrd)

    if mask_ins.any():
        psel = _select_powers(powers, mask_ins)
        res[mask_ins] = amp_pref.amp0 * psel.m_seven_sixths * _amp_ins_ansatz(Mf[mask_ins], psel, amp_pref)
    if mask_mrd.any():
        res[mask_mrd] = amp_pref.amp0 * powers.m_seven_sixths[mask_mrd] * _amp_mrd_ansatz(
            Mf[mask_mrd], a.fRD, a.fDM, a.gamma1, a.gamma2, a.gamma3
        )
    if mask_int.any():
        res[mask_int] = amp_pref.amp0 * powers.m_seven_sixths[mask_int] * _amp_int_ansatz(Mf[mask_int], a.deltas)
    return res


def _IMRPhenDPhase(Mf, p: _PhaseCoeffs, pn, phi_pref: _PhiPref, pow_pi: _Powers):
    res = _np.empty_like(Mf)
    mask_ins = Mf < p.fInsJoin
    mask_mrd = Mf >= p.fMRDJoin
    mask_int = (~mask_ins) & (~mask_mrd)

    if mask_ins.any():
        res[mask_ins] = _phi_ins(
            Mf[mask_ins],
            _powers(Mf[mask_ins]),
            phi_pref,
            p.sigma1,
            p.sigma2,
            p.sigma3,
            p.sigma4,
            pn,
            pow_pi,
            p.eta_inv,
        )
    if mask_mrd.any():
        res[mask_mrd] = p.eta_inv * _phi_mrd(
            Mf[mask_mrd], p.alpha1, p.alpha2, p.alpha3, p.alpha4, p.alpha5, p.fRD, p.fDM
        ) + p.C1MRD + p.C2MRD * Mf[mask_mrd]
    if mask_int.any():
        res[mask_int] = p.eta_inv * _phi_int(Mf[mask_int], p.beta1, p.beta2, p.beta3) + p.C1Int + p.C2Int * Mf[
            mask_int
        ]
    return res
