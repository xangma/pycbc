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
Torch-native (lalsimulation-free) implementation of IMRPhenomHM, with
optional fallback to lalsimulation when the native flag is disabled.

Currently implemented: skeleton placeholder (returns zeros). This will be
filled out with full amplitude/phase mapping. Meanwhile, a lalsim fallback is
provided for correctness when native is off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import lal
import numpy as _np

from pycbc import pnutils
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
import lalsimulation

# Reuse PhenomD amplitude internals (numpy-based)
from pycbc.waveform.imrphenomd_torch import (
    _compute_amp_coeffs as _pd_compute_amp_coeffs,
    _powers as _pd_powers,
    _IMRPhenDAmplitude as _pd_amp_22,
)


# ---------------------------------------------------------------------------
# Mode bookkeeping
# ---------------------------------------------------------------------------

DEFAULT_MODES: List[Tuple[int, int]] = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (4, 3),
]

AMP_fJoin_INS = 0.014  # from PhenomD amplitude model
PHI_fJoin_INS = 0.018  # from PhenomD phase model
AmpFlagTrue = 1
AmpFlagFalse = 0


# ---------------------------------------------------------------------------
# Ringdown helpers (QNM-based) and scaling factors Rholm / Taulm
# ---------------------------------------------------------------------------


def _qnm_fr_fdamp(final_mass: float, final_spin: float, l: int, m: int):
    """Return (fring, fdamp) for mode (l,m) scaled by final mass (geometric)."""
    omega = lalsimulation.SimRingdownCW_CW07102016(
        lalsimulation.SimRingdownCW_KAPPA(final_spin, l, m), l, m, 0
    )
    inv2pi = 0.5 / math.pi
    f_rd = inv2pi * omega.real / final_mass  # Hz (geom units: 1/M)
    f_damp = inv2pi * omega.imag / final_mass
    return f_rd, f_damp


def _compute_ringdown_arrays(m1, m2, chi1z, chi2z, device, dtype):
    """Compute fring/fdamp for supported HM modes and derive Rholm/Taulm."""
    Mf, af = pnutils.get_final_from_initial(m1, m2, chi1z, chi2z)
    fring = torch.zeros((5, 5), device=device, dtype=dtype)
    fdamp = torch.zeros((5, 5), device=device, dtype=dtype)
    modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
    for l, m in modes:
        fr, fd = _qnm_fr_fdamp(Mf, af, l, m)
        fring[l, m] = torch.tensor(fr, device=device, dtype=dtype)
        fdamp[l, m] = torch.tensor(fd, device=device, dtype=dtype)
    Mf_RD_22 = fring[2, 2]
    Mf_DM_22 = fdamp[2, 2]
    Rholm = torch.zeros_like(fring)
    Taulm = torch.zeros_like(fdamp)
    for l, m in modes:
        Rholm[l, m] = Mf_RD_22 / fring[l, m]
        Taulm[l, m] = fdamp[l, m] / Mf_DM_22
    return fring, fdamp, Rholm, Taulm, Mf, af


# ---------------------------------------------------------------------------
# Frequency-domain mapping (PhenomHM)
# ---------------------------------------------------------------------------


def _hm_ti(Mf: torch.Tensor, m: int) -> torch.Tensor:
    return 2.0 * Mf / float(m)


def _hm_trd(Mf: torch.Tensor, Mf_RD_22: torch.Tensor, Mf_RD_lm: torch.Tensor, AmpFlag: int, Rholm_lm: torch.Tensor) -> torch.Tensor:
    if AmpFlag == AmpFlagTrue:
        return Mf - Mf_RD_lm + Mf_RD_22
    return Rholm_lm * Mf


def _hm_slope_am_bm(mm: int, fi: torch.Tensor, fr: torch.Tensor, Mf_RD_22: torch.Tensor, Mf_RD_lm: torch.Tensor, AmpFlag: int, Rholm_lm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Trd = _hm_trd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, Rholm_lm)
    Ti = _hm_ti(fi, mm)
    Am = (Trd - Ti) / (fr - fi)
    Bm = Ti - fi * Am
    return Am, Bm


def _hm_map_params(Mflm: torch.Tensor, l: int, m: int, fring: torch.Tensor, Rholm: torch.Tensor, Mf_RD_22: torch.Tensor, AmpFlag: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Mf_RD_lm = fring[l, m]
    Rholm_lm = Rholm[l, m]
    f1_22 = AMP_fJoin_INS if AmpFlag == AmpFlagTrue else PHI_fJoin_INS
    fi = torch.as_tensor(f1_22 / Rholm_lm, device=Mflm.device, dtype=Mflm.dtype)
    fr = Mf_RD_lm
    Ai = torch.as_tensor(2.0 / float(m), device=Mflm.device, dtype=Mflm.dtype)
    Bi = torch.zeros_like(Ai)
    Am, Bm = _hm_slope_am_bm(m, fi, fr, Mf_RD_22, Mf_RD_lm, AmpFlag, Rholm_lm)
    if AmpFlag == AmpFlagTrue:
        Ar = torch.as_tensor(1.0, device=Mflm.device, dtype=Mflm.dtype)
        Br = -Mf_RD_lm + Mf_RD_22
    else:
        Ar = Rholm_lm
        Br = torch.zeros_like(Ar)
    # choose a,b depending on frequency location
    a = torch.where(Mflm > fi, torch.where(Mflm > fr, Ar, Am), Ai)
    b = torch.where(Mflm > fi, torch.where(Mflm > fr, Br, Bm), Bi)
    return a, b, fi, fr, torch.as_tensor(f1_22, device=Mflm.device, dtype=Mflm.dtype)


def _freq_map(Mflm: torch.Tensor, l: int, m: int, fring: torch.Tensor, Rholm: torch.Tensor, Mf_RD_22: torch.Tensor, AmpFlag: int) -> torch.Tensor:
    a, b, _, _, _ = _hm_map_params(Mflm, l, m, fring, Rholm, Mf_RD_22, AmpFlag)
    return a * Mflm + b


# ---------------------------------------------------------------------------
# Placeholder HM amplitude/phase (to be replaced with full fits)
# ---------------------------------------------------------------------------


def _hm_amplitude_placeholder(Mf_lm: torch.Tensor, l: int, m: int, pHM) -> torch.Tensor:
    """HM amplitude using PhenomD 22 mapped + 1.5PN HM rescaling (partial)."""
    # Map to 22 frequency domain
    Mf22 = _freq_map(Mf_lm, l, m, pHM.fring, pHM.Rholm, pHM.Mf_RD_22, AmpFlagTrue)
    amp22 = pHM.pd_amp22(Mf22)

    # PN rescaling (IMRPhenomHMOnePointFiveSpinPN)
    def hm_onepointfive_spin_pn(fM, l, m, m1, m2, chi1z, chi2z):
        M_input = m1 + m2
        m1n = m1 / M_input
        m2n = m2 / M_input
        M = m1n + m2n
        eta = m1n * m2n / (M * M)
        delta = math.sqrt(max(0.0, 1.0 - 4 * eta))
        Xs = 0.5 * (chi1z + chi2z)
        Xa = 0.5 * (chi1z - chi2z)
        v = (M * 2.0 * math.pi * fM / m) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v * v2
        if l == 2 and m == 2:
            H = 1.0
        elif l == 2 and m == 1:
            v4 = v * v3
            H = (math.sqrt(2.0) / 3.0) * (
                v * delta
                - v2 * 1.5 * (Xa + delta * Xs)
                + v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0))
                + v4
                * (
                    Xa * (3427.0 / 1344 - eta * 2101.0 / 336)
                    + delta * Xs * (3427.0 / 1344 - eta * 965 / 336)
                    + delta * (-0.5j - math.pi - 2j * 0.69314718056)
                )
            )
        elif l == 3 and m == 3:
            H = 0.75 * math.sqrt(5.0 / 7.0) * (v * delta)
        elif l == 3 and m == 2:
            H = (1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta))
        elif l == 4 and m == 4:
            H = (4.0 / 9.0) * math.sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta)
        elif l == 4 and m == 3:
            H = 0.75 * math.sqrt(3.0 / 35.0) * v3 * delta * (1.0 - 2.0 * eta)
        else:
            H = 0.0
        amp = M * M * math.pi * math.sqrt(eta * 2.0 / 3.0) * (v ** -3.5) * abs(H)
        return amp

    m1 = pHM.m1
    m2 = pHM.m2
    chi1 = pHM.chi1z
    chi2 = pHM.chi2z
    # PN terms
    beta_term1 = hm_onepointfive_spin_pn(
        Mf_lm.item(), l, m, m1, m2, chi1, chi2
    )
    beta_term2 = hm_onepointfive_spin_pn(
        2.0 * Mf_lm.item() / m, l, m, m1, m2, chi1, chi2
    )
    beta = 0.0 if beta_term1 == 0 else beta_term1 / beta_term2
    HMamp_term1 = hm_onepointfive_spin_pn(Mf22.item(), l, m, m1, m2, chi1, chi2)
    HMamp_term2 = hm_onepointfive_spin_pn(Mf22.item(), 2, 2, m1, m2, 0.0, 0.0)

    scale = beta * HMamp_term1 / HMamp_term2 if HMamp_term2 != 0 else 0.0
    return amp22 * scale


def _hm_phase_placeholder(Mf_lm: torch.Tensor, l: int, m: int, pHM) -> torch.Tensor:
    """Temporary phase: zero (will be replaced with full HM phase)."""
    # Map to 22 frequency domain (phase mapping differs by AmpFlag)
    # For now, return zeros; phase port will mirror LALSimIMRPhenomHMPhase.
    return torch.zeros_like(Mf_lm, dtype=Mf_lm.dtype)


def _sum_modes_placeholder(freqs: torch.Tensor, pHM: _HMStorage) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build hp,hc from placeholder per-mode pieces (currently zeros)."""
    hp = torch.zeros_like(freqs, dtype=torch.complex128 if freqs.dtype == torch.float64 else torch.complex64)
    hc = torch.zeros_like(hp)
    return hp, hc


# ---------------------------------------------------------------------------
# Storage container (mirrors PhenomHMStorage subset)
# ---------------------------------------------------------------------------


class _HMStorage:
    def __init__(self, m1, m2, chi1z, chi2z, device, dtype):
        self.m1 = m1
        self.m2 = m2
        self.chi1z = chi1z
        self.chi2z = chi2z
        self.fring, self.fdamp, self.Rholm, self.Taulm, self.finmass, self.finspin = _compute_ringdown_arrays(
            m1, m2, chi1z, chi2z, device, dtype
        )
        self.Mf_RD_22 = self.fring[2, 2]
        self.Mf_DM_22 = self.fdamp[2, 2]
        # Precompute PhenomD 22 amplitude coefficients (numpy)
        eta = m1 * m2 / ((m1 + m2) ** 2)
        self._pd_amp_coeffs = _pd_compute_amp_coeffs(
            eta,
            chi1z,
            chi2z,
            self.finspin,
        )

    def pd_amp22(self, Mf22_torch: torch.Tensor) -> torch.Tensor:
        """Evaluate PhenomD (2,2) amplitude at mapped Mf22 (torch in, torch out)."""
        Mf_np = _np.asarray(Mf22_torch.detach().cpu().double())
        powers = _pd_powers(Mf_np)
        amp_np = _pd_amp_22(Mf_np, self._pd_amp_coeffs, powers)
        return torch.as_tensor(amp_np, device=Mf22_torch.device, dtype=Mf22_torch.dtype)


# ---------------------------------------------------------------------------
# Main FD builder (skeleton)
# ---------------------------------------------------------------------------


def imrphenomhm_fd_torch(**p):
    """Torch-native IMRPhenomHM (phase/amplitude mapping under construction).

    For now, returns the lalsimulation waveform to keep behavior unchanged; the
    native path is being filled incrementally. Once parity is validated, this
    will switch to the pure torch build below.
    """

    # ----- temporary safety net -----
    return imrphenomhm_fd_lalsim(**p)

    # device/dtype from torch scheme
    try:
        import pycbc.scheme as _scheme  # pylint: disable=import-outside-toplevel

        device = getattr(getattr(_scheme, "mgr", None).state, "device", None)
        dtype = getattr(getattr(_scheme, "mgr", None).state, "dtype", torch.float64)
        if dtype in (torch.complex64, torch.complex128):
            dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    except Exception:
        device = None
        dtype = torch.float64

    m1 = float(p["mass1"])
    m2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    f_ref = float(p.get("f_ref", f_lower))
    distance = pnutils.megaparsecs_to_meters(float(p["distance"]))
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))

    # masses sorted
    if m1 < m2:
        m1, m2 = m2, m1
        spin1z, spin2z = spin2z, spin1z
        sign_odd = -1.0
    else:
        sign_odd = 1.0

    M = m1 + m2
    eta = m1 * m2 / (M * M)
    M_sec = M * lal.MTSUN_SI

    # Final mass/spin (reuse pnutils fit)
    Mf, af = pnutils.final_mass_spin(m1, m2, spin1z, spin2z)

    # frequency grid
    # crude fmax: use (2,2) QNM times 1.7 factor
    f_qnm22, _ = _qnm_fr_fdamp(Mf, af, 2, 2)
    f_max = f_final if f_final > 0 else 1.7 * f_qnm22
    npts = int(math.ceil(f_max / delta_f)) + 1
    freqs = torch.arange(npts, device=device, dtype=dtype) * delta_f

    hp = torch.zeros(npts, device=device, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
    hc = torch.zeros_like(hp)

    # Placeholder: use only (2,2) PhenomD-like amplitude/phase (to be replaced with full HM map)
    # For now, zero waveform; will fill in as we port amplitude/phase maps.

    hp_fs = FrequencySeries(TorchArrayData(hp), delta_f=delta_f, epoch=0, copy=False)
    hc_fs = FrequencySeries(TorchArrayData(hc), delta_f=delta_f, epoch=0, copy=False)
    return hp_fs, hc_fs


# ---------------------------------------------------------------------------
# Fallback to lalsimulation
# ---------------------------------------------------------------------------


def imrphenomhm_fd_lalsim(**p):
    hp1, hc1 = lalsimulation.SimInspiralChooseFDWaveform(
        float(pnutils.solar_mass_to_kg(p["mass1"])),
        float(pnutils.solar_mass_to_kg(p["mass2"])),
        float(p.get("spin1x", 0.0)),
        float(p.get("spin1y", 0.0)),
        float(p.get("spin1z", 0.0)),
        float(p.get("spin2x", 0.0)),
        float(p.get("spin2y", 0.0)),
        float(p.get("spin2z", 0.0)),
        pnutils.megaparsecs_to_meters(float(p["distance"])),
        float(p.get("inclination", 0.0)),
        float(p.get("coa_phase", 0.0)),
        float(p.get("long_asc_nodes", 0.0)),
        float(p.get("eccentricity", 0.0)),
        float(p.get("mean_per_ano", 0.0)),
        p["delta_f"],
        float(p["f_lower"]),
        float(p.get("f_final", 0.0)),
        float(p.get("f_ref", p["f_lower"])),
        None,
        lalsimulation.IMRPhenomHM,
    )
    hp = FrequencySeries(hp1.data.data[:], delta_f=hp1.deltaF, epoch=hp1.epoch)
    hc = FrequencySeries(hc1.data.data[:], delta_f=hc1.deltaF, epoch=hc1.epoch)
    return hp, hc


__all__ = ["imrphenomhm_fd_torch"]
