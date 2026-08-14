# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch-native (lalsimulation-free) SEOBNRv4_ROM frequency-domain waveform.

This directly evaluates the public ROM data file ``SEOBNRv4ROM_v3.0.hdf5``
with pure NumPy/SciPy, avoiding lalsimulation. It mirrors the structure of
``LALSimIMRSEOBNRv4ROM.c``: interpolate projection coefficients over
`(eta, chi1, chi2)` using tensor-product cubic B-splines, reconstruct
amplitude/phase on sparse frequency grids, glue the low/high submodels at
``Mfm=0.01``, then spline-evaluate onto the requested frequency grid.

Activation (default is CPU/LAL path):
- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` or ``PYCBC_TORCH_NATIVE=1``
- Per-model: ``PYCBC_SEOBNRV4_NATIVE=1``

Limitations:
- Tidal corrections and NRTidal variants are not implemented (BBH only).
- Uses NumPy/SciPy; output is a PyCBC FrequencySeries on the current Torch
  device via the usual scheme casting.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

import lal
from pycbc import pnutils
from pycbc.types import FrequencySeries

_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
_MFM = 0.01  # gluing frequency (geometric Mf)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_rom_file() -> Path:
    """Locate the ROM HDF5 file (local checkout first, else LAL_DATA_PATH)."""
    local = Path(__file__).resolve().parent / _ROM_FILENAME
    if local.exists():
        return local
    for base in os.environ.get("LAL_DATA_PATH", "").split(":"):
        candidate = Path(base) / _ROM_FILENAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{_ROM_FILENAME} not found (put it next to this module or in $LAL_DATA_PATH)")


def _bspline_basis(breaks: np.ndarray, x: float, k: int = 3) -> np.ndarray:
    """Return all basis function values at x for an open, clamped B-spline."""
    # For clamped splines: knot vector length = nbreak + 2k
    t = np.concatenate([np.repeat(breaks[0], k), breaks, np.repeat(breaks[-1], k)])
    n_coeff = len(breaks) + k - 1
    basis = np.empty(n_coeff)
    for i in range(n_coeff):
        c = np.zeros(n_coeff)
        c[i] = 1.0
        basis[i] = _de_boor(t, c, k, x)
    return basis


def _de_boor(t: np.ndarray, c: np.ndarray, k: int, x: float) -> float:
    """Evaluate one B-spline basis (coeff vector c) at x."""
    # Based on scipy BSpline evaluation but compact; k is degree.
    # Find span
    n = len(c)
    if x <= t[k]:
        i = k
    elif x >= t[n]:
        i = n - 1
    else:
        i = np.searchsorted(t, x) - 1
    d = c[i - k : i + 1].astype(float)
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[i + j - k]
            right = t[i + j - r + 1]
            denom = right - left
            alpha = 0.0 if denom == 0 else (x - left) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


@dataclass
class _SubModel:
    etavec: np.ndarray  # breakpoints (eta grid)
    chi1vec: np.ndarray
    chi2vec: np.ndarray
    cvec_amp: np.ndarray  # shape (nbx, nby, nbz, nk_amp)
    cvec_phi: np.ndarray  # shape (nbx, nby, nbz, nk_phi)
    Bamp: np.ndarray      # (nk_amp, nfreq_amp)
    Bphi: np.ndarray      # (nk_phi, nfreq_phi)
    gA: np.ndarray        # frequency grid for amplitude (Mf)
    gPhi: np.ndarray      # frequency grid for phase (Mf)

    @property
    def nk_amp(self):
        return self.Bamp.shape[0]

    @property
    def nk_phi(self):
        return self.Bphi.shape[0]

    @property
    def nbx(self):
        return self.etavec.size + 2  # number of basis funcs

    @property
    def nby(self):
        return self.chi1vec.size + 2

    @property
    def nbz(self):
        return self.chi2vec.size + 2

    @property
    def eta_bounds(self):
        return (self.etavec[0], self.etavec[-1])

    @property
    def chi1_bounds(self):
        return (self.chi1vec[0], self.chi1vec[-1])


@lru_cache(None)
def _load_rom():
    path = _find_rom_file()
    with h5py.File(path, "r") as f:
        subs = {}
        for name in ("sub1", "sub2", "sub3"):
            g = f[name]
            etavec = g["etavec"][:]
            chi1vec = g["chi1vec"][:]
            chi2vec = g["chi2vec"][:]
            # reshape coefficient vectors -> (nbx, nby, nbz, nk)
            nbx = etavec.size + 2
            nby = chi1vec.size + 2
            nbz = chi2vec.size + 2
            nk_amp = g["Bamp"].shape[0]
            nk_phi = g["Bphase"].shape[0]
            c_amp = g["Amp_ciall"][:].reshape(nk_amp, nbx, nby, nbz).transpose(1, 2, 3, 0)
            c_phi = g["Phase_ciall"][:].reshape(nk_phi, nbx, nby, nbz).transpose(1, 2, 3, 0)
            subs[name] = _SubModel(
                etavec=etavec,
                chi1vec=chi1vec,
                chi2vec=chi2vec,
                cvec_amp=c_amp,
                cvec_phi=c_phi,
                Bamp=g["Bamp"][:],
                Bphi=g["Bphase"][:],
                gA=g["Mf_grid_Amp"][:],
                gPhi=g["Mf_grid_Phi"][:],
            )
    return subs


# ---------------------------------------------------------------------------
# Core interpolation and waveform build
# ---------------------------------------------------------------------------


def _interp_coeffs(sub: _SubModel, eta: float, chi1: float, chi2: float) -> Tuple[np.ndarray, np.ndarray]:
    bx = _bspline_basis(sub.etavec, eta)
    by = _bspline_basis(sub.chi1vec, chi1)
    bz = _bspline_basis(sub.chi2vec, chi2)
    # Tensor contraction: (nbx,nby,nbz,nk) with basis
    c_amp = np.einsum("i,j,k,ijkn->n", bx, by, bz, sub.cvec_amp, optimize=True)
    c_phi = np.einsum("i,j,k,ijkn->n", bx, by, bz, sub.cvec_phi, optimize=True)
    return c_amp, c_phi


def _glue_amp(sub_lo: _SubModel, sub_hi: _SubModel, amp_lo: np.ndarray, amp_hi: np.ndarray):
    j_lo = np.max(np.nonzero(sub_lo.gA <= _MFM))
    j_hi = np.min(np.nonzero(sub_hi.gA > _MFM))
    gA = np.concatenate([sub_lo.gA[: j_lo + 1], sub_hi.gA[j_hi:]])
    amp = np.concatenate([amp_lo[: j_lo + 1], amp_hi[j_hi:]])
    return CubicSpline(gA, amp, bc_type="natural")


def _glue_phase(sub_lo: _SubModel, sub_hi: _SubModel, phi_lo: np.ndarray, phi_hi: np.ndarray):
    j_lo = np.max(np.nonzero(sub_lo.gPhi <= _MFM))
    j_hi = np.min(np.nonzero(sub_hi.gPhi > _MFM))
    gP = np.concatenate([sub_lo.gPhi[: j_lo + 1], sub_hi.gPhi[j_hi:]])

    # Adjust high-frequency phase to ensure C1 match at Mfm
    nn = 15
    lo_spline = CubicSpline(sub_lo.gPhi, phi_lo, bc_type="natural")
    g_hi_win = sub_hi.gPhi[j_hi - nn : j_hi + nn + 1]
    phi_lo_win = lo_spline(g_hi_win)
    phi_hi_win = phi_hi[j_hi - nn : j_hi + nn + 1]
    c_lo = np.polyfit(g_hi_win, phi_lo_win, 3)
    c_hi = np.polyfit(g_hi_win, phi_hi_win, 3)
    # derivatives and value at Mfm
    d_lo = np.polyder(c_lo)
    d_hi = np.polyder(c_hi)
    omega_lo = np.polyval(d_lo, _MFM)
    omega_hi = np.polyval(d_hi, _MFM)
    delta_omega = omega_hi - omega_lo
    phi_lo_at = np.polyval(c_lo, _MFM)
    phi_hi_at = np.polyval(c_hi, _MFM)
    delta_phi = phi_hi_at - phi_lo_at - delta_omega * _MFM
    phi_hi_adj = phi_hi - delta_omega * sub_hi.gPhi - delta_phi

    phi = np.concatenate([phi_lo[: j_lo + 1], phi_hi_adj[j_hi:]])
    return CubicSpline(gP, phi, bc_type="natural")


def seobnrv4_fd_torch(**p):
    """Return (hp, hc) FrequencySeries for SEOBNRv4 using pure NumPy/SciPy."""
    subs = _load_rom()

    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    f_ref = float(p.get("f_ref", f_lower))
    distance = pnutils.megaparsecs_to_meters(float(p["distance"]))
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    approximant = p.get("approximant", "SEOBNRv4")
    use_tides = "NRTIDAL" in approximant.upper()
    if use_tides:
        raise NotImplementedError(
            "The native SEOBNRv4 Torch port supports BBH waveforms only; "
            "use the lalsimulation fallback for NRTidal approximants."
        )

    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    M_sec = M * lal.MTSUN_SI

    # Select submodel_hi
    sub_lo = subs["sub1"]
    sub_hi = subs["sub2"] if (spin1z < subs["sub3"].chi1_bounds[0] or eta > subs["sub3"].eta_bounds[1]) else subs["sub3"]

    # Frequency bounds in geometric units
    fmax_geom = f_final * M_sec if f_final > 0 else min(sub_hi.gA[-1], sub_hi.gPhi[-1])
    fmin_geom = f_lower * M_sec
    if fmin_geom < max(sub_lo.gA[0], sub_lo.gPhi[0]):
        raise ValueError("f_lower below ROM support")
    if fmax_geom > min(sub_hi.gA[-1], sub_hi.gPhi[-1]):
        fmax_geom = min(sub_hi.gA[-1], sub_hi.gPhi[-1])

    # Interpolate coefficients
    c_amp_lo, c_phi_lo = _interp_coeffs(sub_lo, eta, spin1z, spin2z)
    c_amp_hi, c_phi_hi = _interp_coeffs(sub_hi, eta, spin1z, spin2z)

    # Evaluate amplitude/phase on sparse ROM grids
    amp_lo = sub_lo.Bamp.T @ c_amp_lo
    amp_hi = sub_hi.Bamp.T @ c_amp_hi
    phi_lo = sub_lo.Bphi.T @ c_phi_lo
    phi_hi = sub_hi.Bphi.T @ c_phi_hi

    # Glue and build splines
    amp_spline = _glue_amp(sub_lo, sub_hi, amp_lo, amp_hi)
    phi_spline = _glue_phase(sub_lo, sub_hi, phi_lo, phi_hi)

    # Build uniform frequency grid
    npts = int(np.ceil(fmax_geom / (delta_f * M_sec)))
    freqs_geom = (np.arange(npts) * delta_f) * M_sec
    mask = freqs_geom >= fmin_geom
    freqs_geom = freqs_geom[mask]
    freqs_hz = freqs_geom / M_sec

    amp = amp_spline(freqs_geom)
    phase = phi_spline(freqs_geom)

    h22 = amp * np.exp(-1j * (phase - phase[0] + 2 * coa_phase))
    cosi = math.cos(inclination)
    hp = 0.5 * (1 + cosi * cosi) * h22 * (2 * np.sqrt(5.0 / (64.0 * math.pi)) * M * lal.MRSUN_SI * M_sec / distance)
    hc = -1j * cosi * h22 * (2 * np.sqrt(5.0 / (64.0 * math.pi)) * M * lal.MRSUN_SI * M_sec / distance)

    return FrequencySeries(hp, delta_f=delta_f, epoch=0), FrequencySeries(hc, delta_f=delta_f, epoch=0)
