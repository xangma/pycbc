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
Torch-friendly reimplementation of the LAL TaylorF2 aligned-spin PN phasing
coefficients.  This mirrors ``XLALSimInspiralPNPhasing_F2`` from
``lalsimulation/lib/LALSimInspiralPNCoefficients.c`` (lines 955-1109 in the
LALSuite checkout at /Users/xangma/repos/lalsuite), including spin, tidal,
and non-GR modifiers, but returns simple numpy arrays that can be consumed by
the torch SPA kernel without calling into lalsimulation.
"""

import math
import numpy as _np
import lal


# Keep the same maximum order used by PNPhasingSeries in LAL (0..15)
_MAX_PN_ORDER = 16


class _PNPhasingSeries:
    """Lightweight stand-in for LAL PNPhasingSeries."""

    def __init__(self):
        self.v = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)
        self.vlogv = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)
        self.vlogvsq = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)


# ---- Non-spinning phasing pieces (numeric coefficients copied verbatim) ----
# LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c lines 692-749
def _f2_2pn(eta):
    return 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0


def _f2_3pn(_eta):
    return -16.0 * lal.PI


def _f2_4pn(eta):
    return 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0


def _f2_5pn(eta):
    return 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * lal.PI


def _f2_5pn_log(eta):
    return 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * lal.PI


def _f2_6pn_log(_eta):
    return -684.8 / 2.1


def _f2_6pn(eta):
    return (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * lal.PI * lal.PI
        - 684.8 / 2.1 * lal.GAMMA
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * lal.PI * lal.PI)
        + eta * eta * 76.055 / 1.728
        - eta * eta * eta * 127.825 / 1.296
        + _f2_6pn_log(eta) * math.log(4.0)
    )


def _f2_7pn(eta):
    return lal.PI * (770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta)


# ---- Spin-orbit pieces (LALSimInspiralPNCoefficients.c lines 756-805) ----
def _f2_3pn_so(m_by_m):
    return m_by_m * (25.0 + 38.0 / 3.0 * m_by_m)


def _f2_5pn_so(m_by_m):
    return -m_by_m * (
        1391.5 / 8.4
        - m_by_m * (1.0 - m_by_m) * 10.0 / 3.0
        + m_by_m * (1276.0 / 8.1 + m_by_m * (1.0 - m_by_m) * 170.0 / 9.0)
    )


def _f2_6pn_so(m_by_m):
    return lal.PI * m_by_m * (1490.0 / 3.0 + m_by_m * 260.0)


def _f2_7pn_so(m_by_m, eta):
    return m_by_m * (
        -17097.8035 / 4.8384
        + eta * 28764.25 / 6.72
        + eta * eta * 47.35 / 1.44
        + m_by_m * (-7189.233785 / 1.524096 + eta * 458.555 / 3.024 - eta * eta * 534.5 / 7.2)
    )


# ---- Spin-squared / quadrupole-monopole pieces (lines 807-858) ----
def _f2_4pn_s1s2(eta):
    return 247.0 / 4.8 * eta


def _f2_4pn_s1s2_o(eta):
    return -721.0 / 4.8 * eta


def _f2_4pn_qm2_so(m_by_m):
    return -720.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_self2_so(m_by_m):
    return 1.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_qm2_s(m_by_m):
    return 240.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_self2_s(m_by_m):
    return -7.0 / 9.6 * m_by_m * m_by_m


def _f2_6pn_s1s2_o(eta):
    return (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta


def _f2_6pn_self2_s(m_by_m):
    return (-4108.25 / 6.72 - 108.5 / 1.2 * m_by_m + 125.5 / 3.6 * m_by_m * m_by_m) * m_by_m * m_by_m


def _f2_6pn_qm2_s(m_by_m):
    return (4703.5 / 8.4 + 2935.0 / 6.0 * m_by_m - 120.0 * m_by_m * m_by_m) * m_by_m * m_by_m


# ---- Tidal pieces (lines 860-915) ----
def _f2_10pn_tidal(m_by_m):
    return (-288.0 + 264.0 * m_by_m) * m_by_m**4


def _f2_12pn_tidal(m_by_m):
    return (
        (-15895.0 / 28.0 + 4595.0 / 28.0 * m_by_m + 5715.0 / 14.0 * m_by_m * m_by_m - 325.0 / 7.0 * m_by_m**3)
        * m_by_m**4
    )


def _f2_13pn_tidal(m_by_m):
    m4 = m_by_m**4
    return m4 * 24.0 * (12.0 - 11.0 * m_by_m) * lal.PI


def _f2_14pn_tidal(m_by_m):
    m3 = m_by_m * m_by_m * m_by_m
    m4 = m3 * m_by_m
    return -m4 * 5.0 * (
        193986935.0 / 571536.0
        - 14415613.0 / 381024.0 * m_by_m
        - 57859.0 / 378.0 * m_by_m * m_by_m
        - 209495.0 / 1512.0 * m3
        + 965.0 / 54.0 * m4
        - 4.0 * m4 * m_by_m
    )


def _f2_15pn_tidal(m_by_m):
    m2 = m_by_m * m_by_m
    m3 = m2 * m_by_m
    m4 = m3 * m_by_m
    return m4 * (1.0 / 28.0) * lal.PI * (27719.0 - 22415.0 * m_by_m + 7598.0 * m2 - 10520.0 * m3)


def _apply_spin_terms(pfa, m1, m2, chi1, chi2, chi1sq, chi2sq, chi1dotchi2, spin_order):
    """Apply spin contributions with fallthrough matching LAL switch."""
    mtot = m1 + m2
    eta = m1 * m2 / (mtot * mtot)
    m1m = m1 / mtot
    m2m = m2 / mtot

    if spin_order in (-1, 7):  # ALL or 3.5PN
        pfa.v[7] += _f2_7pn_so(m1m, eta) * chi1 + _f2_7pn_so(m2m, eta) * chi2
    if spin_order in (-1, 6, 7):
        pfa.v[6] += (
            _f2_6pn_so(m1m) * chi1
            + _f2_6pn_so(m2m) * chi2
            + _f2_6pn_s1s2_o(eta) * chi1 * chi2
            + (_f2_6pn_qm2_s(m1m) * (1.0) + _f2_6pn_self2_s(m1m)) * chi1sq
            + (_f2_6pn_qm2_s(m2m) * (1.0) + _f2_6pn_self2_s(m2m)) * chi2sq
        )
    if spin_order in (-1, 5, 6, 7):
        so1 = _f2_5pn_so(m1m) * chi1
        so2 = _f2_5pn_so(m2m) * chi2
        pfa.v[5] += so1 + so2
        pfa.vlogv[5] += 3.0 * (so1 + so2)
    if spin_order in (-1, 4, 5, 6, 7):
        pfa.v[4] += (
            _f2_4pn_s1s2(eta) * chi1dotchi2
            + _f2_4pn_s1s2_o(eta) * chi1 * chi2
            + (_f2_4pn_qm2_so(m1m) + _f2_4pn_self2_so(m1m)) * chi1 * chi1
            + (_f2_4pn_qm2_so(m2m) + _f2_4pn_self2_so(m2m)) * chi2 * chi2
            + (_f2_4pn_qm2_s(m1m) + _f2_4pn_self2_s(m1m)) * chi1sq
            + (_f2_4pn_qm2_s(m2m) + _f2_4pn_self2_s(m2m)) * chi2sq
        )
    if spin_order in (-1, 3, 4, 5, 6, 7):
        pfa.v[3] += _f2_3pn_so(m1m) * chi1 + _f2_3pn_so(m2m) * chi2


def _apply_tidal_terms(pfa, m1, m2, lambda1, lambda2, tidal_order):
    """Apply tidal terms following LAL fallthrough (lines 1040-1109)."""
    mtot = m1 + m2
    m1m = m1 / mtot
    m2m = m2 / mtot

    if tidal_order in (-1, 15):  # 7.5PN / ALL / DEFAULT
        pfa.v[15] = lambda1 * _f2_15pn_tidal(m1m) + lambda2 * _f2_15pn_tidal(m2m)
    if tidal_order in (-1, 14, 15):
        pfa.v[14] = lambda1 * _f2_14pn_tidal(m1m) + lambda2 * _f2_14pn_tidal(m2m)
    if tidal_order in (-1, 13, 14, 15):
        pfa.v[13] = lambda1 * _f2_13pn_tidal(m1m) + lambda2 * _f2_13pn_tidal(m2m)
    if tidal_order in (-1, 12, 13, 14, 15):
        pfa.v[12] = lambda1 * _f2_12pn_tidal(m1m) + lambda2 * _f2_12pn_tidal(m2m)
    if tidal_order in (-1, 10, 12, 13, 14, 15):
        pfa.v[10] = lambda1 * _f2_10pn_tidal(m1m) + lambda2 * _f2_10pn_tidal(m2m)


def taylorf2_aligned_phasing(mass1, mass2, chi1, chi2, *, spin_order=-1,
                             tidal_order=-1, dchi=None, qm_def1=0.0,
                             qm_def2=0.0, chi1sq=None, chi2sq=None,
                             chi1dotchi2=None, lambda1=0.0, lambda2=0.0):
    """
    Return PN phasing coefficients for aligned-spin TaylorF2 without using LAL.

    Parameters
    ----------
    mass1, mass2 : float
        Component masses in solar masses.
    chi1, chi2 : float
        Dimensionless aligned spins.
    spin_order : int, optional
        PN spin order flag (use -1 for all; matches LAL enums).
    tidal_order : int, optional
        PN tidal order flag (use -1 for default/all).
    dchi : dict, optional
        Non-GR fractional phase deviations keyed by 'dchi0'...'dchi7','dchi5l',
        'dchi6l'. Missing keys default to 0.
    qm_def1, qm_def2 : float, optional
        Spin-induced quadrupole deformation deltas (defaults to 0).
    chi1sq, chi2sq, chi1dotchi2 : float, optional
        Precomputed spin magnitudes / dot-product. If None, they are inferred
        from chi1/chi2 assuming alignment (i.e., chi1sq=chi1**2 etc.).

    Returns
    -------
    _PNPhasingSeries
        Object with arrays ``v``, ``vlogv``, ``vlogvsq`` identical to LAL.
    """
    dchi = dchi or {}
    chi1sq = chi1sq if chi1sq is not None else chi1 * chi1
    chi2sq = chi2sq if chi2sq is not None else chi2 * chi2
    chi1dotchi2 = chi1dotchi2 if chi1dotchi2 is not None else chi1 * chi2

    mtot = mass1 + mass2
    eta = (mass1 * mass2) / (mtot * mtot)
    pfaN = 3.0 / (128.0 * eta)

    pfa = _PNPhasingSeries()
    # Base non-spinning terms
    pfa.v[0] = 1.0
    pfa.v[1] = 0.0
    pfa.v[2] = _f2_2pn(eta)
    pfa.v[3] = _f2_3pn(eta)
    pfa.v[4] = _f2_4pn(eta)
    pfa.v[5] = _f2_5pn(eta)
    pfa.vlogv[5] = _f2_5pn_log(eta)
    pfa.v[6] = _f2_6pn(eta)
    pfa.vlogv[6] = _f2_6pn_log(eta)
    pfa.v[7] = _f2_7pn(eta)

    # Non-GR tweaks (LALSimInspiralPNCoefficients.c lines 972-980)
    pfa.v[0] *= 1.0 + dchi.get("dchi0", 0.0)
    pfa.v[1] = dchi.get("dchi1", 0.0)
    pfa.v[2] *= 1.0 + dchi.get("dchi2", 0.0)
    pfa.v[3] *= 1.0 + dchi.get("dchi3", 0.0)
    pfa.v[4] *= 1.0 + dchi.get("dchi4", 0.0)
    pfa.v[5] *= 1.0 + dchi.get("dchi5", 0.0)
    pfa.vlogv[5] *= 1.0 + dchi.get("dchi5l", 0.0)
    pfa.v[6] *= 1.0 + dchi.get("dchi6", 0.0)
    pfa.vlogv[6] *= 1.0 + dchi.get("dchi6l", 0.0)
    pfa.v[7] *= 1.0 + dchi.get("dchi7", 0.0)

    # Spin / tidal contributions
    _apply_spin_terms(pfa, mass1, mass2, chi1, chi2, chi1sq, chi2sq, chi1dotchi2, spin_order)
    _apply_tidal_terms(pfa, mass1, mass2, lambda1, lambda2, tidal_order)

    # Quadrupole-monopole deformation (applied via qm_defX factors)
    # Note: already folded into _apply_spin_terms via qm_def placeholders; here
    # we just scale by (1 + delta) to match LAL's qm_def1/2 definitions.
    qm1 = 1.0 + qm_def1
    qm2 = 1.0 + qm_def2
    # Re-apply the qm scaling to the SS terms that depend on qm_def*
    mtot = mass1 + mass2
    m1m = mass1 / mtot
    m2m = mass2 / mtot
    # 6PN SS/QM terms
    pfa.v[6] += (_f2_6pn_qm2_s(m1m) * (qm1 - 1.0)) * chi1sq
    pfa.v[6] += (_f2_6pn_qm2_s(m2m) * (qm2 - 1.0)) * chi2sq
    # 4PN SS/QM terms
    pfa.v[4] += (_f2_4pn_qm2_so(m1m) * (qm1 - 1.0) + _f2_4pn_qm2_s(m1m) * (qm1 - 1.0)) * chi1sq
    pfa.v[4] += (_f2_4pn_qm2_so(m2m) * (qm2 - 1.0) + _f2_4pn_qm2_s(m2m) * (qm2 - 1.0)) * chi2sq

    # Final global scaling
    pfa.v *= pfaN
    pfa.vlogv *= pfaN
    pfa.vlogvsq *= pfaN
    return pfa

