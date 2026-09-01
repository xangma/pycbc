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
implemented without lalsimulation. Model coefficients and QNM interpolation
are assembled as small CPU values; all frequency-dependent work is performed
on the active Torch device.

Activation
----------
- Per-model flag: ``PYCBC_IMRPHENOMD_NATIVE=0`` opts out; ``=1`` forces native
- Global flag   : ``PYCBC_TORCH_NATIVE_PORTS=0`` opts out when the per-model
  flag is unset

Supported aligned-spin IMRPhenomD and NRTidal v1/v2 calls use this native path
by default under ``TorchScheme``. CPU schemes, disabled ports, and unsupported
options retain the lalsimulation path.

Limitations
-----------
- Non-default spin-induced quadrupoles, dynamic-tide parameters, and non-GR
  modifiers retain the lalsimulation fallback.
- Only the dominant (2,2) mode is produced; higher modes follow the LAL
  default of zero for IMRPhenomD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import numpy as _np
from pycbc import lal_compat as lal
import torch

from pycbc import pnutils, scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.nrtidal_torch import (
    nrtidal_amplitude,
    nrtidal_higher_order_spin_phase,
    nrtidal_merger_frequency,
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
    nrtidal_taper,
    nrtidal_version,
)
from pycbc.waveform._native_math import NaturalCubicSpline
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

    third: Any
    two_thirds: Any
    four_thirds: Any
    five_thirds: Any
    seven_thirds: Any
    eight_thirds: Any
    m_third: Any
    m_two_thirds: Any
    m_four_thirds: Any
    m_five_thirds: Any
    m_seven_sixths: Any
    inv: Any
    one: Any
    two: Any
    four: Any


def _powers(x) -> _Powers:
    """Return useful powers of a NumPy value or Torch tensor."""
    # Keep the inactive zero-frequency bin numerically harmless without letting
    # fractional powers underflow or inverse powers overflow.
    if isinstance(x, torch.Tensor):
        safe_floor = torch.finfo(x.dtype).eps
        x_safe = torch.where(x == 0.0, safe_floor, x)
        third = torch.pow(x_safe, 1.0 / 3.0)
        pow_fn = torch.pow
    else:
        safe_floor = _np.finfo(float).eps
        x_safe = _np.where(x == 0.0, safe_floor, x)
        third = _np.cbrt(x_safe)
        pow_fn = _np.power
    two_thirds = third * third
    four_thirds = two_thirds * two_thirds
    five_thirds = x_safe * two_thirds
    seven_thirds = x_safe * x_safe * third
    eight_thirds = x_safe * x_safe * two_thirds
    m_third = 1.0 / third
    m_two_thirds = 1.0 / two_thirds
    m_four_thirds = 1.0 / four_thirds
    m_five_thirds = 1.0 / five_thirds
    m_seven_sixths = pow_fn(x_safe, -7.0 / 6.0)
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


def _piecewise_regions(values, lower_join, upper_join, *, regular_grid=False):
    """Select the three PhenomD regions without changing join semantics.

    The regular FD generator constructs an increasing uniform grid, so its
    regions are contiguous and slices avoid repeatedly materializing boolean
    selections for every power and ansatz term. Sequence waveforms may use an
    arbitrary ordering, retaining the original boolean-mask path.
    """
    if regular_grid and isinstance(values, torch.Tensor):
        joins = values.new_tensor((lower_join, upper_join))
        lower_t, upper_t = torch.searchsorted(values, joins)
        lower, upper = int(lower_t), int(upper_t)
        return (
            slice(None, lower),
            slice(lower, upper),
            slice(upper, None),
        )

    inspiral = values < lower_join
    merger_ringdown = values >= upper_join
    intermediate = (~inspiral) & (~merger_ringdown)
    return inspiral, intermediate, merger_ringdown


# ---------- Final spin and QNM fits -------------------------------------------------


def _final_spin0815(eta, chi1, chi2):
    """Final spin fit (Eq. 3.6 of arXiv:1508.07250, FinalSpin0815_s)."""
    eta2 = eta * eta
    eta3 = eta2 * eta
    if isinstance(eta, torch.Tensor):
        seta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    else:
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
    header = Path(__file__).resolve().parent / "data" / "LALSimIMRPhenomD.h"
    if not header.is_file():
        raise FileNotFoundError(
            "Bundled IMRPhenomD QNM data are missing from the PyCBC "
            f"installation: {header}"
        )
    text = header.read_text()

    def _extract_array(name: str):
        start = text.index(f"static const double {name}[]")
        chunk = text[start:].split("{", 1)[1].split("}", 1)[0]
        vals = [
            float(value.replace("L", ""))
            for value in chunk.replace("\\\n", "").replace("\n", " ").split(",")
            if value.strip()
        ]
        return _np.array(vals, dtype=_np.float64)

    a = _extract_array("QNMData_a")
    fring = _extract_array("QNMData_fring")
    fdamp = _extract_array("QNMData_fdamp")
    return a, fring, fdamp


@lru_cache(None)
def _qnm_splines() -> Tuple[NaturalCubicSpline, NaturalCubicSpline]:
    """Build the natural cubic QNM splines used by lalsimulation."""
    xs, ys_f, ys_d = _load_qnm_tables()
    return (
        NaturalCubicSpline(xs, ys_f, extrapolate=False),
        NaturalCubicSpline(xs, ys_d, extrapolate=False),
    )


def _interp_qnm(a: float) -> Tuple[float, float]:
    """Interpolate ringdown frequency and damping rate vs final spin."""
    xs, _, _ = _load_qnm_tables()
    if not xs[0] <= a <= xs[-1]:
        raise ValueError(f"IMRPhenomD final spin {a} is outside the QNM table")
    fring_spline, fdamp_spline = _qnm_splines()
    fring = fring_spline(a)
    fdamp = fdamp_spline(a)
    return float(fring), float(fdamp)


def _erad_rational_0815(eta, chi1, chi2):
    """Radiated energy fit from arXiv:1508.07250 (matches PhenomInternal_EradRational0815)."""
    if isinstance(eta, torch.Tensor):
        seta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    else:
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
    if isinstance(gamma2, torch.Tensor):
        fmax_branch1 = torch.abs(fRD + (fDM * (-1.0 + torch.sqrt(torch.clamp(1.0 - gamma2 * gamma2, min=0.0))) * gamma3) / gamma2)
        fmax_branch2 = torch.abs(fRD - (fDM * gamma3) / gamma2)
        return torch.where(gamma2 < 1.0, fmax_branch1, fmax_branch2)
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
    exp_fn = torch.exp if isinstance(f, torch.Tensor) else _np.exp
    return exp_fn(-(fminfRD) * gamma2 / (fDMgamma3)) * (fDMgamma3 * gamma1) / (
        fminfRD * fminfRD + fDMgamma3 * fDMgamma3
    )


def _d_amp_mrd_ansatz(f, fRD, fDM, gamma1, gamma2, gamma3):
    fDMgamma3 = fDM * gamma3
    pow2 = fDMgamma3 * fDMgamma3
    fminfRD = f - fRD
    exp_fn = torch.exp if isinstance(f, torch.Tensor) else _np.exp
    expfac = exp_fn(((fminfRD) * gamma2) / (fDMgamma3))
    denom = fminfRD * fminfRD + pow2
    return ((-2 * fDM * fminfRD * gamma3 * gamma1) / denom - (gamma2 * gamma1)) / (expfac * denom)


def _solve_intermediate_polynomial(f1, f2, f3, v1, v2, v3, d1, d2):
    if isinstance(f1, torch.Tensor) or isinstance(v1, torch.Tensor):
        ones = torch.ones_like(f1)
        zeros = torch.zeros_like(f1)
        r0 = torch.stack([ones, f1, f1**2, f1**3, f1**4], dim=-1)
        r1 = torch.stack([zeros, ones, 2.0 * f1, 3.0 * f1**2, 4.0 * f1**3], dim=-1)
        r2 = torch.stack([ones, f2, f2**2, f2**3, f2**4], dim=-1)
        r3 = torch.stack([ones, f3, f3**2, f3**3, f3**4], dim=-1)
        r4 = torch.stack([zeros, ones, 2.0 * f3, 3.0 * f3**2, 4.0 * f3**3], dim=-1)
        A = torch.stack([r0, r1, r2, r3, r4], dim=1)
        b = torch.stack([v1, d1, v2, v3, d2], dim=-1)
        return torch.linalg.solve(A, b)
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


def _compute_phase_coeffs(
    eta: float,
    chi1: float,
    chi2: float,
    finspin: float,
    pn,
    Rholm: float = 1.0,
    Taulm: float = 1.0,
) -> _PhaseCoeffs:
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
    DPhiMRDVal = _d_phi_mrd(
        fMRDJoin,
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        alpha5,
        fRD,
        fDM,
        eta_inv,
        Rholm,
        Taulm,
    )
    C2MRD = DPhiIntTempVal - DPhiMRDVal
    PhiMRDJoin = _phi_mrd(
        fMRDJoin,
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        alpha5,
        fRD,
        fDM,
        Rholm,
        Taulm,
    )
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
    log_fn = torch.log if isinstance(v, torch.Tensor) else _np.log
    logv = log_fn(v)
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
    log_fn = torch.log if isinstance(Mf, torch.Tensor) else _np.log
    return beta1 * Mf - beta3 / (3.0 * (Mf ** 3)) + beta2 * log_fn(Mf)


def _d_phi_int(Mf, beta1, beta2, beta3, eta_inv):
    return eta_inv * (beta1 + beta3 / (Mf ** 4) + beta2 / Mf)


def _phi_mrd(
    f,
    alpha1,
    alpha2,
    alpha3,
    alpha4,
    alpha5,
    fRD,
    fDM,
    Rholm=1.0,
    Taulm=1.0,
):
    xp = torch if isinstance(f, torch.Tensor) else _np
    sqrootf = xp.sqrt(f)
    fpow1_5 = f * sqrootf
    fpow0_75 = xp.sqrt(fpow1_5)
    return (
        -(alpha2 / f)
        + (4.0 / 3.0) * (alpha3 * fpow0_75)
        + alpha1 * f
        + alpha4
        * Rholm
        * xp.arctan((f - alpha5 * fRD) / (Rholm * fDM * Taulm))
    )


def _d_phi_mrd(
    f,
    alpha1,
    alpha2,
    alpha3,
    alpha4,
    alpha5,
    fRD,
    fDM,
    eta_inv,
    Rholm=1.0,
    Taulm=1.0,
):
    width = fDM * Taulm
    return eta_inv * (
        alpha1
        + alpha2 / (f * f)
        + alpha3 / (f**0.25)
        + alpha4
        / (width * (1 + ((f - alpha5 * fRD) ** 2) / ((width * Rholm) ** 2)))
    )


# ---------- PN helpers --------------------------------------------------------------


def _subtract_3pn_ss(m1, m2, M, eta, chi1, chi2):
    m1M = m1 / M
    m2M = m2 / M
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M) + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)) * m1M * m1M * chi1 * chi1
    pn_ss3 += ((4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M) + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)) * m2M * m2M * chi2 * chi2
    return pn_ss3


# ---------- Public native-path boundary ---------------------------------------------


_DEFAULT_ONLY_ORDER_KEYS = (
    "spin_order",
    "tidal_order",
    "eccentricity_order",
)
_TRANSVERSE_SPIN_KEYS = ("spin1x", "spin1y", "spin2x", "spin2y")
_TIDAL_EXTENSION_KEYS = (
    "dquad_mon1",
    "dquad_mon2",
    "lambda_octu1",
    "lambda_octu2",
    "quadfmode1",
    "quadfmode2",
    "octufmode1",
    "octufmode2",
)
_NON_GR_KEYS = (
    "dchi0",
    "dchi1",
    "dchi2",
    "dchi3",
    "dchi4",
    "dchi5",
    "dchi5l",
    "dchi6",
    "dchi6l",
    "dchi7",
    "dalpha1",
    "dalpha2",
    "dalpha3",
    "dalpha4",
    "dalpha5",
    "dbeta1",
    "dbeta2",
    "dbeta3",
)


def _is_nonzero(value):
    """Return whether an optional scalar is set to a non-zero value."""
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def _is_default_order(value):
    """Return whether an integer-valued LAL order flag has its default."""
    try:
        return float(value) == -1.0 and int(value) == -1
    except (TypeError, ValueError, OverflowError):
        return False


def _is_nonnegative_finite(value):
    """Return whether an optional scalar is finite and non-negative."""

    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0.0


def imrphenomd_native_supported(params):
    """Return whether ``params`` are covered by the native Torch generator.

    The public dispatcher sends unsupported configurations to lalsimulation,
    preserving its errors and its testing-GR modifications instead of silently
    ignoring inputs. The native path covers aligned-spin IMRPhenomD and the
    standard NRTidal v1/v2 corrections, with the (2,2) mode and polarization
    rotation.
    """
    approximant = params.get("approximant", "IMRPhenomD")
    if approximant not in {
        "IMRPhenomD",
        "IMRPhenomD_NRTidal",
        "IMRPhenomD_NRTidalv2",
    }:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS + _TIDAL_EXTENSION_KEYS + _NON_GR_KEYS
        )
    ):
        return False
    lambdas = (params.get("lambda1", 0.0), params.get("lambda2", 0.0))
    if approximant == "IMRPhenomD":
        if any(_is_nonzero(value) for value in lambdas):
            return False
    elif not all(_is_nonnegative_finite(value) for value in lambdas):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in ("frame_axis", "modes_choice", "side_bands")
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


# ---------- Main waveform -----------------------------------------------------------


@dataclass
class _IMRPhenomDInputs:
    """Validated scalar inputs shared by regular and sequence generation."""

    tidal_version: int | None
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    lambda1: float
    lambda2: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    f_ref: float
    total_mass: float
    eta: float
    total_mass_seconds: float
    dquad1: float
    dquad2: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


def imrphenomd_sequence_native_supported(params):
    """Return whether arbitrary-frequency IMRPhenomD generation is native."""

    return imrphenomd_native_supported(params)


def _imrphenomd_inputs(p, *, sequence=False):
    """Validate and normalize scalar inputs shared by both public APIs."""

    if not imrphenomd_native_supported(p):
        raise ValueError(
            "IMRPhenomD parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomD requires TorchScheme")

    approximant = p.get("approximant", "IMRPhenomD")
    tidal_version = nrtidal_version(approximant)

    # Ensure m1 >= m2 per LAL convention, keeping matter parameters paired
    # with their component.
    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    lambda1 = float(p.get("lambda1") or 0.0)
    lambda2 = float(p.get("lambda2") or 0.0)
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        lambda1, lambda2 = lambda2, lambda1

    f_ref = float(p.get("f_ref", 0.0))
    distance = pnutils.megaparsecs_to_meters(float(p["distance"]))
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument and
    # ignores the corresponding PyCBC parameter.
    long_asc_nodes = 0.0 if sequence else float(p.get("long_asc_nodes", 0.0))

    if not math.isfinite(mass1) or not math.isfinite(mass2):
        raise ValueError("IMRPhenomD component masses must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomD component masses must be positive")
    if not all(
        math.isfinite(value)
        for value in (spin1z, spin2z, inclination, coa_phase, long_asc_nodes)
    ):
        raise ValueError("IMRPhenomD spins and angles must be finite")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("IMRPhenomD aligned spins must be between -1 and 1")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (lambda1, lambda2)
    ):
        raise ValueError("NRTidal deformabilities must be finite and non-negative")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("IMRPhenomD distance must be finite and positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("IMRPhenomD f_ref must be finite and non-negative")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    eta = _nudge_eta(eta)
    total_mass_seconds = total_mass * _MTSUN_SI
    dquad1 = 0.0
    dquad2 = 0.0
    if tidal_version == 2:
        dquad1 = nrtidal_quadrupole_from_lambda(lambda1) - 1.0
        dquad2 = nrtidal_quadrupole_from_lambda(lambda2) - 1.0

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _IMRPhenomDInputs(
        tidal_version=tidal_version,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        lambda1=lambda1,
        lambda2=lambda2,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        f_ref=f_ref,
        total_mass=total_mass,
        eta=eta,
        total_mass_seconds=total_mass_seconds,
        dquad1=dquad1,
        dquad2=dquad2,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _imrphenomd_samples(
    inputs,
    frequencies,
    reference_frequency,
    *,
    regular_grid=False,
):
    """Evaluate the inclination-independent waveform at device frequencies."""

    mass1 = inputs.mass1
    mass2 = inputs.mass2
    spin1z = inputs.spin1z
    spin2z = inputs.spin2z
    lambda1 = inputs.lambda1
    lambda2 = inputs.lambda2
    total_mass = inputs.total_mass
    eta = inputs.eta
    total_mass_seconds = inputs.total_mass_seconds

    finspin = _final_spin0815(eta, spin1z, spin2z)
    amp_coeffs = _compute_amp_coeffs(eta, spin1z, spin2z, finspin)

    # PN phasing series (reuse TaylorF2 port).
    pn = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=-1,
        tidal_order=-1,
        dchi={},
        qm_def1=inputs.dquad1,
        qm_def2=inputs.dquad2,
        lambda1=0.0,
        lambda2=0.0,
    )
    # Subtract 3PN spin-spin term (matches LAL tuning).
    pn.v[6] -= _subtract_3pn_ss(
        mass1,
        mass2,
        total_mass,
        eta,
        spin1z,
        spin2z,
    ) * pn.v[0]
    phase_coeffs = _compute_phase_coeffs(eta, spin1z, spin2z, finspin, pn)
    pi_powers = _pi_powers()
    phi_pref = _init_phi_prefactors(
        phase_coeffs.sigma1,
        phase_coeffs.sigma2,
        phase_coeffs.sigma3,
        phase_coeffs.sigma4,
        pn,
        pi_powers,
    )

    # A sequence with f_ref=0 uses its first sample as the reference. Keep
    # that device scalar in Torch; regular-grid generation retains the exact
    # NumPy scalar setup used by the original port.
    mf_ref = total_mass_seconds * reference_frequency
    if isinstance(mf_ref, torch.Tensor):
        reference_values = mf_ref.reshape(1)
    else:
        reference_values = _np.array([mf_ref])
    phase_at_reference = _IMRPhenDPhase(
        reference_values,
        phase_coeffs,
        pn,
        phi_pref,
        pi_powers,
    )[0]
    phase_offset = 2.0 * inputs.coa_phase + phase_at_reference

    time_shift = _d_phi_mrd(
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

    dimensionless_frequencies = total_mass_seconds * frequencies
    powers = _powers(dimensionless_frequencies)
    amplitude = _IMRPhenDAmplitude(
        dimensionless_frequencies,
        amp_coeffs,
        powers,
        regular_grid=regular_grid,
    )
    if inputs.tidal_version == 2:
        amplitude += 2.0 * math.sqrt(_PI / 5.0) * nrtidal_amplitude(
            frequencies,
            mass1,
            mass2,
            lambda1,
            lambda2,
        )

    phase = _IMRPhenDPhase(
        dimensionless_frequencies,
        phase_coeffs,
        pn,
        phi_pref,
        pi_powers,
        regular_grid=regular_grid,
    )
    phase -= time_shift * (dimensionless_frequencies - mf_ref) + phase_offset
    if inputs.tidal_version is not None:
        phase += nrtidal_phase(
            frequencies,
            mass1,
            mass2,
            lambda1,
            lambda2,
            inputs.tidal_version,
        )
        if inputs.tidal_version == 2:
            phase += nrtidal_higher_order_spin_phase(
                frequencies,
                mass1,
                mass2,
                spin1z,
                spin2z,
                inputs.dquad1 + 1.0,
                inputs.dquad2 + 1.0,
            )
        merger_frequency = nrtidal_merger_frequency(
            mass1,
            mass2,
            lambda1,
            lambda2,
        )
        amplitude *= nrtidal_taper(frequencies, merger_frequency)

    amplitude_scale = (
        2.0
        * math.sqrt(5.0 / (64.0 * _PI))
        * total_mass
        * _MRSUN_SI
        * total_mass_seconds
        / inputs.distance
    )
    scaled_amplitude = amplitude_scale * amplitude
    return torch.complex(
        scaled_amplitude * torch.cos(phase),
        -scaled_amplitude * torch.sin(phase),
    ).to(inputs.complex_dtype)


def _imrphenomd_polarizations(samples, inputs):
    """Project inclination-independent samples into plus and cross."""

    cosi = math.cos(inputs.inclination)
    plus0 = samples * 0.5 * (1.0 + cosi * cosi)
    cross0 = samples * complex(0.0, -cosi)
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        cos_nodes * plus0 + sin_nodes * cross0,
        cos_nodes * cross0 - sin_nodes * plus0,
    )


def _lal_next_power_of_two(value):
    """Match IMRPhenomD's integer-truncating ``NextPow2`` helper."""

    truncated = int(value)
    if truncated <= 0:
        return 0
    return 1 << (truncated - 1).bit_length()


def imrphenomd_cutoff_frequency(mass1, mass2):
    """Return the natural base-model cutoff frequency in Hz."""

    return f_CUT / ((float(mass1) + float(mass2)) * _MTSUN_SI)


def _zero_pad_active_samples(active_samples, npts, first_bin, stop_bin):
    """Place active waveform samples into a device-resident zero grid."""

    samples = active_samples.new_zeros(npts)
    samples[first_bin:stop_bin] = active_samples
    return samples


def imrphenomd_fd_torch(**p):
    """Generate IMRPhenomD or NRTidal polarizations with Torch."""

    inputs = _imrphenomd_inputs(p)
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomD frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomD delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomD f_final must be non-negative")

    # Determine the base-model termination and output layout. NRTidal asks the
    # base model to stop no later than 1.3 times its merger fit, then applies a
    # taper ending at 1.2 times merger. LAL still preserves a larger explicitly
    # requested f_final as zero padding in the returned power-of-two series.
    f_cut_hz = imrphenomd_cutoff_frequency(inputs.mass1, inputs.mass2)
    if inputs.tidal_version is None:
        layout_f_max = f_final if f_final > 0.0 else f_cut_hz
        active_f_max = min(layout_f_max, f_cut_hz)
    else:
        merger_frequency = nrtidal_merger_frequency(
            inputs.mass1,
            inputs.mass2,
            inputs.lambda1,
            inputs.lambda2,
        )
        tidal_f_max = 1.3 * merger_frequency
        layout_f_max = f_final if f_final > 0.0 else tidal_f_max
        active_f_max = min(layout_f_max, tidal_f_max, f_cut_hz)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    # Frequency grid (power-of-two length like LAL), resident on the active
    # device from allocation through final series construction.
    if inputs.tidal_version is not None and f_final > tidal_f_max:
        # The outer NRTidal wrapper preserves an explicitly requested high
        # frequency with a floating-point next-power-of-two calculation.
        npts = int(
            2.0 ** math.ceil(math.log2(layout_f_max / delta_f))
        ) + 1
    else:
        # The base IMRPhenomD helper accepts size_t, so the bin count is
        # truncated before it is rounded up to a power of two.
        npts = _lal_next_power_of_two(layout_f_max / delta_f) + 1
    # LAL truncates the frequency bounds to integer bins and uses an exclusive
    # upper bin in its evaluation loop. Evaluate only that active interval:
    # the power-of-two output may contain substantially more zero padding, and
    # sending those bins through every amplitude and phase operation can also
    # cross Torch's CPU parallelization grain size at high thread counts.
    first_bin = int(math.floor(f_lower / delta_f))
    stop_bin = int(math.floor(active_f_max / delta_f))
    active_frequencies = (
        torch.arange(
            first_bin,
            stop_bin,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )

    reference_frequency = inputs.f_ref if inputs.f_ref > 0.0 else f_lower
    active_samples = _imrphenomd_samples(
        inputs,
        active_frequencies,
        reference_frequency,
        regular_grid=True,
    )
    samples = _zero_pad_active_samples(
        active_samples, npts, first_bin, stop_bin
    )
    hp, hc = _imrphenomd_polarizations(samples, inputs)

    epoch = -1.0 / delta_f
    hp_fs = FrequencySeries(
        TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False
    )
    hc_fs = FrequencySeries(
        TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False
    )
    return hp_fs, hc_fs


def _imrphenomd_sequence_frequencies(sample_points, inputs):
    """Return validated sequence frequencies on the active Torch device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomD sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomD sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomD sample_points must be positive")
    return frequencies


def imrphenomd_fd_sequence_torch(**p):
    """Evaluate IMRPhenomD or NRTidal at arbitrary frequencies with Torch."""

    if not imrphenomd_sequence_native_supported(p):
        raise ValueError(
            "IMRPhenomD sequence parameters are not supported by the "
            "native Torch path"
        )
    inputs = _imrphenomd_inputs(p, sequence=True)
    frequencies = _imrphenomd_sequence_frequencies(
        p["sample_points"],
        inputs,
    )
    reference_frequency = (
        inputs.f_ref if inputs.f_ref > 0.0 else frequencies[0]
    )
    samples = _imrphenomd_samples(
        inputs,
        frequencies,
        reference_frequency,
    )
    plus, cross = _imrphenomd_polarizations(samples, inputs)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def _IMRPhenDAmplitude(
    Mf,
    a: _AmpCoeffs,
    powers: _Powers,
    *,
    regular_grid=False,
):
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
    using_torch = isinstance(Mf, torch.Tensor)
    res = torch.empty_like(Mf) if using_torch else _np.empty_like(Mf)
    select_ins, select_int, select_mrd = _piecewise_regions(
        Mf,
        AMP_fJoin_INS,
        a.fmaxCalc,
        regular_grid=regular_grid,
    )

    psel = _select_powers(powers, select_ins)
    res[select_ins] = (
        amp_pref.amp0
        * psel.m_seven_sixths
        * _amp_ins_ansatz(Mf[select_ins], psel, amp_pref)
    )
    res[select_mrd] = (
        amp_pref.amp0
        * powers.m_seven_sixths[select_mrd]
        * _amp_mrd_ansatz(
            Mf[select_mrd], a.fRD, a.fDM, a.gamma1, a.gamma2, a.gamma3
        )
    )
    deltas = (
        torch.as_tensor(a.deltas, dtype=Mf.dtype, device=Mf.device)
        if using_torch
        else a.deltas
    )
    res[select_int] = (
        amp_pref.amp0
        * powers.m_seven_sixths[select_int]
        * _amp_int_ansatz(Mf[select_int], deltas)
    )
    return res


def _IMRPhenDPhase(
    Mf,
    p: _PhaseCoeffs,
    pn,
    phi_pref: _PhiPref,
    pow_pi: _Powers,
    Rholm: float = 1.0,
    Taulm: float = 1.0,
    *,
    regular_grid=False,
):
    res = torch.empty_like(Mf) if isinstance(Mf, torch.Tensor) else _np.empty_like(Mf)
    select_ins, select_int, select_mrd = _piecewise_regions(
        Mf,
        p.fInsJoin,
        p.fMRDJoin,
        regular_grid=regular_grid,
    )

    res[select_ins] = _phi_ins(
        Mf[select_ins],
        _powers(Mf[select_ins]),
        phi_pref,
        p.sigma1,
        p.sigma2,
        p.sigma3,
        p.sigma4,
        pn,
        pow_pi,
        p.eta_inv,
    )
    res[select_mrd] = (
        p.eta_inv
        * _phi_mrd(
            Mf[select_mrd],
            p.alpha1,
            p.alpha2,
            p.alpha3,
            p.alpha4,
            p.alpha5,
            p.fRD,
            p.fDM,
            Rholm,
            Taulm,
        )
        + p.C1MRD
        + p.C2MRD * Mf[select_mrd]
    )
    res[select_int] = (
        p.eta_inv * _phi_int(Mf[select_int], p.beta1, p.beta2, p.beta3)
        + p.C1Int
        + p.C2Int * Mf[select_int]
    )
    return res


def imrphenomd_fd_batch(**params):
    """Generate a batch of IMRPhenomD frequency-domain waveforms directly as 2D PyTorch tensors.

    Parameters
    ----------
    mass1 : float or Tensor
        Primary mass in solar masses (shape (B,) or scalar).
    mass2 : float or Tensor
        Secondary mass in solar masses (shape (B,) or scalar).
    spin1z : float or Tensor, optional
        Dimensionless aligned spin of primary (shape (B,) or scalar, default 0.0).
    spin2z : float or Tensor, optional
        Dimensionless aligned spin of secondary (shape (B,) or scalar, default 0.0).
    distance : float or Tensor, optional
        Luminosity distance in Mpc (shape (B,) or scalar, default 1.0).
    inclination : float or Tensor, optional
        Inclination angle in radians (shape (B,) or scalar, default 0.0).
    coa_phase : float or Tensor, optional
        Coalescence phase in radians (shape (B,) or scalar, default 0.0).
    long_asc_nodes : float or Tensor, optional
        Longitude of ascending nodes in radians (shape (B,) or scalar, default 0.0).
    f_ref : float or Tensor, optional
        Reference frequency in Hertz (shape (B,) or scalar, default 0.0, uses f_lower).
    delta_f : float
        Frequency resolution in Hertz.
    f_lower : float
        Lower frequency cutoff in Hertz.
    f_final : float, optional
        Upper frequency cutoff in Hertz. If 0.0 or not provided, uses the maximum
        cutoff frequency across the batch.

    Returns
    -------
    hp : torch.Tensor
        Batch of plus polarizations of shape (B, length) on target device.
    hc : torch.Tensor
        Batch of cross polarizations of shape (B, length) on target device.
    """
    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = torch.float32 if getattr(device, "type", "cpu") == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    m1_in = params["mass1"]
    m2_in = params["mass2"]

    batch_size = 1
    for k in (
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
        "distance",
        "inclination",
        "coa_phase",
        "long_asc_nodes",
        "f_ref",
    ):
        v = params.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            batch_size = max(batch_size, v.shape[0])
        elif isinstance(v, (list, tuple, _np.ndarray)) and len(v) > 1:
            batch_size = max(batch_size, len(v))

    def _to_tensor(val, default=0.0):
        if val is None:
            val = default
        if isinstance(val, torch.Tensor):
            t = val.to(device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            return t
        elif isinstance(val, (list, tuple, _np.ndarray)):
            return torch.as_tensor(val, device=device, dtype=real_dtype)
        else:
            return torch.full((batch_size,), float(val), device=device, dtype=real_dtype)

    m1 = _to_tensor(m1_in)
    m2 = _to_tensor(m2_in)
    s1z = _to_tensor(params.get("spin1z", 0.0))
    s2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomD frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomD delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomD f_final must be non-negative")

    # Swap masses and spins where m2 > m1
    swap_mask = m2 > m1
    m1_eff = torch.where(swap_mask, m2, m1)
    m2_eff = torch.where(swap_mask, m1, m2)
    s1z_eff = torch.where(swap_mask, s2z, s1z)
    s2z_eff = torch.where(swap_mask, s1z, s2z)

    total_mass = m1_eff + m2_eff
    eta = (m1_eff * m2_eff) / (total_mass * total_mass)
    eta = torch.where(eta > 0.25, 0.25 - _ETA_EPS, eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta_inv = 1.0 / eta

    seta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    seta_plus1 = 1.0 + seta
    chi12 = s1z_eff * s1z_eff
    chi22 = s2z_eff * s2z_eff
    chi = 0.5 * ((s1z_eff + s2z_eff) * (1.0 - 76.0 * eta / 113.0) + seta * (s1z_eff - s2z_eff))
    xi = -1.0 + chi

    finspin = _final_spin0815(eta, s1z_eff, s2z_eff)
    erad = _erad_rational_0815(eta, s1z_eff, s2z_eff)

    fring_spline, fdamp_spline = _qnm_splines()
    finspin_np = finspin.detach().cpu().numpy()
    fring = torch.as_tensor(fring_spline(finspin_np), dtype=real_dtype, device=device)
    fdamp = torch.as_tensor(fdamp_spline(finspin_np), dtype=real_dtype, device=device)
    fRD = fring / (1.0 - erad)
    fDM = fdamp / (1.0 - erad)

    rho1 = _rho1(eta, eta2, xi)
    rho2 = _rho2(eta, eta2, xi)
    rho3 = _rho3(eta, eta2, xi)

    gamma1 = _gamma1(eta, eta2, xi)
    gamma2 = _gamma2(eta, eta2, xi)
    gamma3 = _gamma3(eta, eta2, xi)

    fmaxCalc = _fmax_calc(fRD, fDM, gamma2, gamma3)
    amp0 = torch.sqrt(2.0 / 3.0 * eta) * (_PI ** (-1.0 / 6.0))

    pow_pi = _pi_powers()
    pi2 = _PI * _PI

    pref_two_thirds = ((-969.0 + 1804.0 * eta) * pow_pi.two_thirds) / 672.0
    pref_one = ((s1z_eff * (81.0 * seta_plus1 - 44.0 * eta) + s2z_eff * (81.0 - 81.0 * seta - 44.0 * eta)) * _PI) / 48.0
    pref_four_thirds = (
        (-27312085.0 - 10287648.0 * chi22 - 10287648.0 * chi12 * seta_plus1 + 10287648.0 * chi22 * seta)
        + 24.0 * (-1975055.0 + 857304.0 * chi12 - 994896.0 * s1z_eff * s2z_eff + 857304.0 * chi22) * eta
        + 35371056.0 * eta2
    ) * pow_pi.four_thirds / 8.128512e6
    pref_five_thirds = pow_pi.five_thirds * (
        s2z_eff * (-285197.0 * (-1.0 + seta) + 4.0 * (-91902.0 + 1579.0 * seta) * eta - 35632.0 * eta2)
        + s1z_eff * (285197.0 * seta_plus1 - 4.0 * (91902.0 + 1579.0 * seta) * eta - 35632.0 * eta2)
        + 42840.0 * (-1.0 + 4.0 * eta) * _PI
    ) / 32256.0
    pref_two = (
        -pi2
        * (
            -336.0
            * (-3248849057.0 + 2943675504.0 * chi12 - 3339284256.0 * s1z_eff * s2z_eff + 2943675504.0 * chi22)
            * eta2
            - 324322727232.0 * eta3
            - 7.0
            * (
                -177520268561.0
                + 107414046432.0 * chi22
                + 107414046432.0 * chi12 * seta_plus1
                - 107414046432.0 * chi22 * seta
                + 11087290368.0 * (s1z_eff + s2z_eff + s1z_eff * seta - s2z_eff * seta) * _PI
            )
            + 12.0
            * eta
            * (
                -545384828789.0
                - 176491177632.0 * s1z_eff * s2z_eff
                + 202603761360.0 * chi22
                + 77616.0 * chi12 * (2610335.0 + 995766.0 * seta)
                - 77287373856.0 * chi22 * seta
                + 5841690624.0 * (s1z_eff + s2z_eff) * _PI
                + 21384760320.0 * pi2
            )
        )
    ) / 6.0085960704e10
    pref_seven_thirds = rho1
    pref_eight_thirds = rho2
    pref_three = rho3

    # Collocation setup
    f1 = torch.full((batch_size,), AMP_fJoin_INS, dtype=real_dtype, device=device)
    f3 = fmaxCalc
    f2 = f1 + 0.5 * (f3 - f1)

    powers_f1 = _powers(f1)
    v1 = (
        1.0
        + powers_f1.two_thirds * pref_two_thirds
        + powers_f1.four_thirds * pref_four_thirds
        + powers_f1.five_thirds * pref_five_thirds
        + powers_f1.seven_thirds * pref_seven_thirds
        + powers_f1.eight_thirds * pref_eight_thirds
        + f1 * (pref_one + f1 * pref_two + powers_f1.two * pref_three)
    )
    d1 = (
        (2.0 / 3.0) * pref_two_thirds * powers_f1.m_third
        + (4.0 / 3.0) * pref_four_thirds * powers_f1.third
        + (5.0 / 3.0) * pref_five_thirds * powers_f1.two_thirds
        + (7.0 / 3.0) * pref_seven_thirds * powers_f1.four_thirds
        + (8.0 / 3.0) * pref_eight_thirds * powers_f1.five_thirds
        + pref_one
        + 2.0 * pref_two * powers_f1.one
        + 3.0 * pref_three * powers_f1.two
    )
    fDMgamma3 = fDM * gamma3
    fminfRD_3 = f3 - fRD
    v3 = torch.exp(-fminfRD_3 * gamma2 / fDMgamma3) * (fDMgamma3 * gamma1) / (
        fminfRD_3 * fminfRD_3 + fDMgamma3 * fDMgamma3
    )
    pow2_3 = fDMgamma3 * fDMgamma3
    expfac_3 = torch.exp((fminfRD_3 * gamma2) / fDMgamma3)
    denom_3 = fminfRD_3 * fminfRD_3 + pow2_3
    d2 = ((-2.0 * fDM * fminfRD_3 * gamma3 * gamma1) / denom_3 - (gamma2 * gamma1)) / (expfac_3 * denom_3)
    v2 = _amp_int_col_fit(eta, eta2, chi)

    deltas = _solve_intermediate_polynomial(f1, f2, f3, v1, v2, v3, d1, d2)

    # Phasing
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

    pn = taylorf2_aligned_phasing(m1_eff, m2_eff, s1z_eff, s2z_eff)
    pn.v[6] -= _subtract_3pn_ss(m1_eff, m2_eff, total_mass, eta, s1z_eff, s2z_eff) * pn.v[0]

    phi_initial = pn.v[5] - 0.25 * _PI
    phi_two_thirds = pn.v[7] * pow_pi.two_thirds
    phi_third = pn.v[6] * pow_pi.third
    phi_third_logv = pn.vlogv[6] * pow_pi.third
    phi_logv = pn.vlogv[5]
    phi_minus_third = pn.v[4] * pow_pi.m_third
    phi_minus_two_thirds = pn.v[3] * pow_pi.m_two_thirds
    phi_minus_one = pn.v[2] * pow_pi.inv
    phi_minus_four_thirds = pn.v[1] / pow_pi.four_thirds
    phi_minus_five_thirds = pn.v[0] * pow_pi.m_five_thirds
    phi_one = sigma1
    phi_four_thirds = 0.75 * sigma2
    phi_five_thirds = 0.6 * sigma3
    phi_two = 0.5 * sigma4

    fInsJoin = torch.full((batch_size,), PHI_fJoin_INS, dtype=real_dtype, device=device)
    fMRDJoin = 0.5 * fRD

    # Phasing continuity at fInsJoin
    v_ins = (_PI * PHI_fJoin_INS) ** (1.0 / 3.0)
    logv_ins = math.log(v_ins)
    v2_ins = v_ins * v_ins
    v3_ins = v_ins * v2_ins
    v4_ins = v_ins * v3_ins
    v5_ins = v_ins * v4_ins
    v6_ins = v_ins * v5_ins
    v7_ins = v_ins * v6_ins
    v8_ins = v_ins * v7_ins
    Dphasing = 2.0 * pn.v[7] * v7_ins
    Dphasing = Dphasing + (pn.v[6] + pn.vlogv[6] * (1.0 + logv_ins)) * v6_ins
    Dphasing = Dphasing + pn.vlogv[5] * v5_ins
    Dphasing = Dphasing - 1.0 * pn.v[4] * v4_ins
    Dphasing = Dphasing - 2.0 * pn.v[3] * v3_ins
    Dphasing = Dphasing - 3.0 * pn.v[2] * v2_ins
    Dphasing = Dphasing - 4.0 * pn.v[1] * v_ins
    Dphasing = Dphasing - 5.0 * pn.v[0]
    Dphasing = Dphasing / (v8_ins * 3.0)
    Dphasing = Dphasing * _PI
    pow_pi_m_third = float(pow_pi.m_third)
    pow_pi_m_two_thirds = float(pow_pi.m_two_thirds)
    pow_pi_inv = float(pow_pi.inv)
    Dphasing = Dphasing + (
        sigma1
        + sigma2 * v_ins * pow_pi_m_third
        + sigma3 * v2_ins * pow_pi_m_two_thirds
        + (sigma4 * pow_pi_inv) * v3_ins
    ) * eta_inv

    d_phi_ins_val = Dphasing
    d_phi_int_val = eta_inv * (beta1 + beta3 / (PHI_fJoin_INS ** 4) + beta2 / PHI_fJoin_INS)
    C2Int = d_phi_ins_val - d_phi_int_val

    powers_fIns = _powers(fInsJoin)
    v_fIns = powers_fIns.third * pow_pi.third
    logv_fIns = torch.log(v_fIns)
    phi_ins_val = (
        phi_initial
        + phi_two_thirds * powers_fIns.two_thirds
        + phi_third * powers_fIns.third
        + phi_third_logv * logv_fIns * powers_fIns.third
        + phi_logv * logv_fIns
        + phi_minus_third * powers_fIns.m_third
        + phi_minus_two_thirds * powers_fIns.m_two_thirds
        + phi_minus_one * powers_fIns.inv
        + phi_minus_four_thirds * powers_fIns.m_four_thirds
        + phi_minus_five_thirds * powers_fIns.m_five_thirds
        + eta_inv
        * (
            phi_one * fInsJoin
            + phi_four_thirds * powers_fIns.four_thirds
            + phi_five_thirds * powers_fIns.five_thirds
            + phi_two * powers_fIns.two
        )
    )
    phi_int_val = beta1 * PHI_fJoin_INS - beta3 / (3.0 * (PHI_fJoin_INS ** 3)) + beta2 * math.log(PHI_fJoin_INS)
    C1Int = phi_ins_val - eta_inv * phi_int_val - C2Int * PHI_fJoin_INS

    PhiIntTempVal = (
        eta_inv * (beta1 * fMRDJoin - beta3 / (3.0 * (fMRDJoin ** 3)) + beta2 * torch.log(fMRDJoin))
        + C1Int
        + C2Int * fMRDJoin
    )
    DPhiIntTempVal = C2Int + (beta1 + beta3 / (fMRDJoin ** 4) + beta2 / fMRDJoin) * eta_inv

    width_mrd = fDM
    DPhiMRDVal = eta_inv * (
        alpha1
        + alpha2 / (fMRDJoin * fMRDJoin)
        + alpha3 / (fMRDJoin**0.25)
        + alpha4 / (width_mrd * (1.0 + ((fMRDJoin - alpha5 * fRD) ** 2) / (width_mrd ** 2)))
    )
    C2MRD = DPhiIntTempVal - DPhiMRDVal

    sqroot_mrd = torch.sqrt(fMRDJoin)
    fpow1_5_mrd = fMRDJoin * sqroot_mrd
    fpow0_75_mrd = torch.sqrt(fpow1_5_mrd)
    PhiMRDJoin = (
        -(alpha2 / fMRDJoin)
        + (4.0 / 3.0) * (alpha3 * fpow0_75_mrd)
        + alpha1 * fMRDJoin
        + alpha4 * torch.arctan((fMRDJoin - alpha5 * fRD) / fDM)
    )
    C1MRD = PhiIntTempVal - eta_inv * PhiMRDJoin - C2MRD * fMRDJoin

    total_mass_seconds = total_mass * _MTSUN_SI

    # Reference phase
    ref_freq = torch.where(f_ref > 0.0, f_ref, torch.full_like(f_ref, f_lower))
    mf_ref = total_mass_seconds * ref_freq

    powers_ref = _powers(mf_ref)
    v_ref = powers_ref.third * pow_pi.third
    logv_ref = torch.log(v_ref)
    phi_ins_ref = (
        phi_initial
        + phi_two_thirds * powers_ref.two_thirds
        + phi_third * powers_ref.third
        + phi_third_logv * logv_ref * powers_ref.third
        + phi_logv * logv_ref
        + phi_minus_third * powers_ref.m_third
        + phi_minus_two_thirds * powers_ref.m_two_thirds
        + phi_minus_one * powers_ref.inv
        + phi_minus_four_thirds * powers_ref.m_four_thirds
        + phi_minus_five_thirds * powers_ref.m_five_thirds
        + eta_inv
        * (
            phi_one * mf_ref
            + phi_four_thirds * powers_ref.four_thirds
            + phi_five_thirds * powers_ref.five_thirds
            + phi_two * powers_ref.two
        )
    )
    phi_int_ref = (
        eta_inv * (beta1 * mf_ref - beta3 / (3.0 * (mf_ref ** 3)) + beta2 * torch.log(mf_ref))
        + C1Int
        + C2Int * mf_ref
    )
    sqroot_ref = torch.sqrt(mf_ref)
    fpow1_5_ref = mf_ref * sqroot_ref
    fpow0_75_ref = torch.sqrt(fpow1_5_ref)
    phi_mrd_raw_ref = (
        -(alpha2 / mf_ref)
        + (4.0 / 3.0) * (alpha3 * fpow0_75_ref)
        + alpha1 * mf_ref
        + alpha4 * torch.arctan((mf_ref - alpha5 * fRD) / fDM)
    )
    phi_mrd_ref = eta_inv * phi_mrd_raw_ref + C1MRD + C2MRD * mf_ref

    phase_at_ref = torch.where(
        mf_ref < PHI_fJoin_INS,
        phi_ins_ref,
        torch.where(mf_ref >= fMRDJoin, phi_mrd_ref, phi_int_ref),
    )
    phase_offset = 2.0 * coa_phase + phase_at_ref

    # Time shift
    width_ts = fDM
    time_shift = eta_inv * (
        alpha1
        + alpha2 / (fmaxCalc * fmaxCalc)
        + alpha3 / (fmaxCalc**0.25)
        + alpha4 / (width_ts * (1.0 + ((fmaxCalc - alpha5 * fRD) ** 2) / (width_ts ** 2)))
    )

    # Cutoffs and grid layout
    f_cut_hz = f_CUT / total_mass_seconds
    if f_final > 0.0:
        layout_f_max = f_final
        active_f_max = torch.clamp(f_cut_hz, max=f_final)
    else:
        layout_f_max = float(torch.max(f_cut_hz).item())
        active_f_max = f_cut_hz

    if layout_f_max <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    npts = _lal_next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(math.floor(f_lower / delta_f))
    max_stop_bin = int(math.floor(layout_f_max / delta_f))

    if first_bin >= max_stop_bin:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    freqs = torch.arange(first_bin, max_stop_bin, dtype=real_dtype, device=device) * delta_f
    freqs_2d = freqs.unsqueeze(0)
    total_mass_seconds_2d = total_mass_seconds.unsqueeze(1)
    Mf = total_mass_seconds_2d * freqs_2d
    powers_Mf = _powers(Mf)

    # Vectorized Amplitude across (B, N_f)
    amp_ins_ansatz = (
        1.0
        + powers_Mf.two_thirds * pref_two_thirds[:, None]
        + powers_Mf.four_thirds * pref_four_thirds[:, None]
        + powers_Mf.five_thirds * pref_five_thirds[:, None]
        + powers_Mf.seven_thirds * pref_seven_thirds[:, None]
        + powers_Mf.eight_thirds * pref_eight_thirds[:, None]
        + Mf * (pref_one[:, None] + Mf * pref_two[:, None] + powers_Mf.two * pref_three[:, None])
    )
    amp_ins = amp0[:, None] * powers_Mf.m_seven_sixths * amp_ins_ansatz

    fminfRD = Mf - fRD[:, None]
    fDMgamma3_2d = (fDM * gamma3)[:, None]
    amp_mrd_ansatz = torch.exp(-fminfRD * gamma2[:, None] / fDMgamma3_2d) * (fDMgamma3_2d * gamma1[:, None]) / (
        fminfRD * fminfRD + fDMgamma3_2d * fDMgamma3_2d
    )
    amp_mrd = amp0[:, None] * powers_Mf.m_seven_sixths * amp_mrd_ansatz

    Mf2 = Mf * Mf
    amp_int_ansatz = (
        deltas[:, 0:1]
        + Mf * deltas[:, 1:2]
        + Mf2 * (deltas[:, 2:3] + Mf * deltas[:, 3:4] + deltas[:, 4:5] * Mf2)
    )
    amp_int = amp0[:, None] * powers_Mf.m_seven_sixths * amp_int_ansatz

    amp_ins_mask = Mf < AMP_fJoin_INS
    amp_mrd_mask = Mf >= fmaxCalc[:, None]
    amplitude = torch.where(amp_ins_mask, amp_ins, torch.where(amp_mrd_mask, amp_mrd, amp_int))

    # Vectorized Phase across (B, N_f)
    v_Mf = powers_Mf.third * float(pow_pi.third)
    logv_Mf = torch.log(v_Mf)
    phi_ins = (
        phi_initial[:, None]
        + phi_two_thirds[:, None] * powers_Mf.two_thirds
        + phi_third[:, None] * powers_Mf.third
        + phi_third_logv[:, None] * logv_Mf * powers_Mf.third
        + phi_logv[:, None] * logv_Mf
        + phi_minus_third[:, None] * powers_Mf.m_third
        + phi_minus_two_thirds[:, None] * powers_Mf.m_two_thirds
        + phi_minus_one[:, None] * powers_Mf.inv
        + phi_minus_four_thirds[:, None] * powers_Mf.m_four_thirds
        + phi_minus_five_thirds[:, None] * powers_Mf.m_five_thirds
        + eta_inv[:, None]
        * (
            phi_one[:, None] * Mf
            + phi_four_thirds[:, None] * powers_Mf.four_thirds
            + phi_five_thirds[:, None] * powers_Mf.five_thirds
            + phi_two[:, None] * powers_Mf.two
        )
    )

    phi_int = (
        eta_inv[:, None] * (beta1[:, None] * Mf - beta3[:, None] / (3.0 * (Mf ** 3)) + beta2[:, None] * torch.log(Mf))
        + C1Int[:, None]
        + C2Int[:, None] * Mf
    )

    sqroot_Mf = torch.sqrt(Mf)
    fpow1_5_Mf = Mf * sqroot_Mf
    fpow0_75_Mf = torch.sqrt(fpow1_5_Mf)
    phi_mrd_raw = (
        -(alpha2[:, None] / Mf)
        + (4.0 / 3.0) * (alpha3[:, None] * fpow0_75_Mf)
        + alpha1[:, None] * Mf
        + alpha4[:, None] * torch.arctan((Mf - alpha5[:, None] * fRD[:, None]) / fDM[:, None])
    )
    phi_mrd = eta_inv[:, None] * phi_mrd_raw + C1MRD[:, None] + C2MRD[:, None] * Mf

    phase_ins_mask = Mf < PHI_fJoin_INS
    phase_mrd_mask = Mf >= fMRDJoin[:, None]
    phase = torch.where(phase_ins_mask, phi_ins, torch.where(phase_mrd_mask, phi_mrd, phi_int))

    phase = phase - time_shift[:, None] * (Mf - mf_ref[:, None]) - phase_offset[:, None]

    dist_m = dist * 1.0e6 * lal.PC_SI
    amp_scale = (
        2.0
        * math.sqrt(5.0 / (64.0 * _PI))
        * total_mass
        * _MRSUN_SI
        * total_mass_seconds
        / dist_m
    ).unsqueeze(1)
    scaled_amplitude = amp_scale * amplitude

    active_samples = torch.complex(
        scaled_amplitude * torch.cos(phase),
        -scaled_amplitude * torch.sin(phase),
    ).to(complex_dtype)

    # Active frequency cutoff mask
    stop_bins = torch.floor(active_f_max / delta_f).to(torch.int64)
    bin_idx = torch.arange(first_bin, max_stop_bin, device=device).unsqueeze(0)
    valid_mask = bin_idx < stop_bins.unsqueeze(1)
    active_samples = torch.where(valid_mask, active_samples, torch.zeros_like(active_samples))

    # Polarizations
    cosi = torch.cos(incl).unsqueeze(1)
    plus0 = active_samples * 0.5 * (1.0 + cosi * cosi)
    cross0 = active_samples * torch.complex(torch.zeros_like(cosi), -cosi)

    cos_nodes = torch.cos(2.0 * long_asc_nodes).unsqueeze(1)
    sin_nodes = torch.sin(2.0 * long_asc_nodes).unsqueeze(1)

    hp_active = cos_nodes * plus0 + sin_nodes * cross0
    hc_active = cos_nodes * cross0 - sin_nodes * plus0

    hp = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hc = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hp[:, first_bin:max_stop_bin] = hp_active
    hc[:, first_bin:max_stop_bin] = hc_active
    return hp, hc
