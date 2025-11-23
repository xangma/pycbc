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

"""Torch-native (lalsimulation-free) evaluator for SEOBNRv4HM_ROM.

This module reconstructs the higher-mode ROM directly from the public ROM
data file ``SEOBNRv4HMROM.hdf5`` using NumPy/SciPy, without calling
``lalsimulation``. It mirrors the structure of ``LALSimIMRSEOBNRv4HMROM.c``:

- tensor-product cubic B-splines over (q, chi1, chi2) to interpolate the
  projection coefficients for each ROM patch (low-f plus four high-f patches);
- hybridization of low/high patches via the f_hyb window;
- carrier phase reconstruction and per-mode approximate phasing;
- assembly of (l,-m) modes, time shift correction, and spherical-harmonic
  summation to plus/cross polarizations.

Activation (default is CPU/LAL path):
- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` (or ``PYCBC_TORCH_NATIVE=1``)
- Per-model: ``PYCBC_SEOBNRV4HM_NATIVE=1``

The ROM data file must be available as ``pycbc/waveform/SEOBNRv4HMROM.hdf5``
or on ``$LAL_DATA_PATH``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch

import lal
from pycbc import pnutils
from pycbc.conversions import get_final_from_initial
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

try:  # optional dependency used for QNM frequencies
    import pykerr
except Exception:  # pragma: no cover
    pykerr = None


_ROM_FILENAME = "SEOBNRv4HMROM.hdf5"
_F_HYB_INI = 0.003
_F_HYB_END = 0.004
_LM_MODES = [(2, 2), (3, 3), (2, 1), (4, 4), (5, 5)]  # ordering used in ROM
_CONST_PHASESHIFT = [0.0, -math.pi / 2.0, math.pi / 2.0, math.pi, math.pi / 2.0]
_CONST_FMAX = [1.7, 1.55, 1.7, 1.35, 1.25]
_MF_LOW_22 = 0.0004925491025543576


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_rom_file() -> Path:
    local = Path(__file__).resolve().parent / _ROM_FILENAME
    if local.exists():
        return local
    for base in os.environ.get("LAL_DATA_PATH", "").split(":"):
        if not base:
            continue
        cand = Path(base) / _ROM_FILENAME
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{_ROM_FILENAME} not found; place it next to this module or on $LAL_DATA_PATH")


def _knot_vector(grid: torch.Tensor, k: int = 3) -> torch.Tensor:
    return torch.cat(
        [
            torch.full((k,), grid[0], device=grid.device, dtype=grid.dtype),
            grid,
            torch.full((k,), grid[-1], device=grid.device, dtype=grid.dtype),
        ]
    )


def _de_boor(t: torch.Tensor, c: torch.Tensor, k: int, x: torch.Tensor) -> torch.Tensor:
    n = c.numel()
    if x <= t[k]:
        i = k
    elif x >= t[n]:
        i = n - 1
    else:
        i = int(torch.searchsorted(t, x) - 1)
    d = c[i - k : i + 1].clone()
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[i + j - k]
            right = t[i + j - r + 1]
            denom = right - left
            alpha = 0.0 if denom == 0 else float((x - left) / denom)
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


def _bspline_basis(grid: torch.Tensor, x: torch.Tensor, k: int = 3) -> torch.Tensor:
    t = _knot_vector(grid, k)
    n_coeff = grid.numel() + k - 1
    basis = []
    for i in range(n_coeff):
        c = torch.zeros(n_coeff, device=grid.device, dtype=grid.dtype)
        c[i] = 1.0
        basis.append(_de_boor(t, c, k, x))
    return torch.stack(basis)


def _cosine_blend(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    w = (x - x0) / (x1 - x0)
    w = torch.clamp(w, 0.0, 1.0)
    return 0.5 * (1 - torch.cos(math.pi * w))


def _unwrap_phase(ph: torch.Tensor) -> torch.Tensor:
    d = torch.diff(ph)
    dd = (d + math.pi) % (2 * math.pi) - math.pi
    dd = torch.where((dd == -math.pi) & (d > 0), math.pi, dd)
    ph_unwrapped = torch.cat([ph[:1], ph[0] + torch.cumsum(dd, dim=0)])
    return ph_unwrapped


def _qnm_omega(m1: float, m2: float, chi1: float, chi2: float, l: int, m: int) -> float:
    """Return omega_QNM (rad/s) for given remnant. Falls back to a fit if pykerr
    is unavailable."""

    Mf, af = get_final_from_initial(m1, m2, chi1, chi2)
    if pykerr is not None:
        f0 = pykerr.qnmfreq(Mf, af, l, m, 0)  # Hz
        return 2 * math.pi * f0
    # crude fallback: ringdown freq ~ (1 - 0.63*(1-af)**0.3)/(2π M)
    M_sec = (m1 + m2) * lal.MTSUN_SI
    omega = (1 - 0.63 * (1 - af) ** 0.3) / M_sec
    return omega


def _select_hf_patch(q: float, chi1: float) -> int:
    if (q > 3.0) and (chi1 <= 0.8):
        return 0  # hqls
    if (q > 3.0) and (chi1 > 0.8):
        return 1  # hqhs
    if (q <= 3.0) and (chi1 <= 0.8):
        return 2  # lqls
    return 3  # lqhs


def _compute_i_max_LF_i_min_HF(freq_lo: np.ndarray, freq_hi: np.ndarray, f_hyb_ini: float) -> Tuple[int, int]:
    i_max = np.max(np.nonzero(freq_lo <= f_hyb_ini))
    i_min = np.min(np.nonzero(freq_hi >= f_hyb_ini))
    return int(i_max), int(i_min)


# ---------------------------------------------------------------------------
# ROM data structures
# ---------------------------------------------------------------------------


@dataclass
class _SubModel:
    qvec: torch.Tensor
    chi1vec: torch.Tensor
    chi2vec: torch.Tensor
    gCMode: torch.Tensor  # sparse freq grid for cmode
    gPhase: torch.Tensor  # sparse freq grid for carrier phase
    Breal: torch.Tensor
    Bimag: torch.Tensor
    Bphase: torch.Tensor
    cvec_real: torch.Tensor  # shape (nk, nbx, nby, nbz)
    cvec_imag: torch.Tensor
    cvec_phase: torch.Tensor

    @property
    def nbx(self):
        return self.qvec.size + 2

    @property
    def nby(self):
        return self.chi1vec.size + 2

    @property
    def nbz(self):
        return self.chi2vec.size + 2


@dataclass
class _ModeROM:
    lowf: _SubModel
    hqls: _SubModel
    hqhs: _SubModel
    lqls: _SubModel
    lqhs: _SubModel


@lru_cache(None)
def _load_rom(target_dtype: torch.dtype = None, device: str | torch.device | None = None) -> Dict[str, _ModeROM]:
    path = _find_rom_file()
    data = {}
    # choose dtype/device from scheme if available
    # target dtype/device come from function arguments
    def to_torch(arr):
        t = torch.as_tensor(arr)
        if target_dtype is not None:
            t = t.to(dtype=target_dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    with h5py.File(path, "r") as f:
        for mode_name, lm in zip(["22", "33", "21", "44", "55"], _LM_MODES):
            submodels = {}
            for grp_name in ["lowf", "hqls", "hqhs", "lqls", "lqhs"]:
                g = f[grp_name]
                cgroup = g["CF_modes"][mode_name]
                phase_grp = g["phase_carrier"]
                qvec = to_torch(g["qvec"][:])
                chi1vec = to_torch(g["chi1vec"][:])
                chi2vec = to_torch(g["chi2vec"][:])
                Breal = to_torch(cgroup["basis_re"][:])
                Bimag = to_torch(cgroup["basis_im"][:])
                Bphase = to_torch(phase_grp["basis"][:])
                gCMode = to_torch(cgroup["MF_grid"][:])
                gPhase = to_torch(phase_grp["MF_grid"][:])

                # coeff_re shape (nk, nbx, nby, nbz) after transpose
                coeff_re = to_torch(cgroup["coeff_re"][:].transpose(1, 2, 3, 0))
                coeff_im = to_torch(cgroup["coeff_im"][:].transpose(1, 2, 3, 0))
                coeff_phase = to_torch(phase_grp["coeff"][:].transpose(1, 2, 3, 0))

                submodels[grp_name] = _SubModel(
                    qvec=qvec,
                    chi1vec=chi1vec,
                    chi2vec=chi2vec,
                    gCMode=gCMode,
                    gPhase=gPhase,
                    Breal=Breal,
                    Bimag=Bimag,
                    Bphase=Bphase,
                    cvec_real=coeff_re,
                    cvec_imag=coeff_im,
                    cvec_phase=coeff_phase,
                )
            data[mode_name] = _ModeROM(**submodels)
    return data


# ---------------------------------------------------------------------------
# Core interpolation helpers
# ---------------------------------------------------------------------------


def _interp_coeffs(sub: _SubModel, q: float, chi1: float, chi2: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sub.qvec.device
    dtype = sub.qvec.dtype
    bx = _bspline_basis(sub.qvec, torch.tensor(q, device=device, dtype=dtype))
    by = _bspline_basis(sub.chi1vec, torch.tensor(chi1, device=device, dtype=dtype))
    bz = _bspline_basis(sub.chi2vec, torch.tensor(chi2, device=device, dtype=dtype))
    c_real = torch.einsum("i,j,k,ijkn->n", bx, by, bz, sub.cvec_real)
    c_imag = torch.einsum("i,j,k,ijkn->n", bx, by, bz, sub.cvec_imag)
    c_phase = torch.einsum("i,j,k,ijkn->n", bx, by, bz, sub.cvec_phase)
    return c_real, c_imag, c_phase


def _eval_cmode(sub: _SubModel, q: float, chi1: float, chi2: float, kind: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c_real, c_imag, _ = _interp_coeffs(sub, q, chi1, chi2)
    real = sub.Breal.T @ c_real
    imag = sub.Bimag.T @ c_imag
    return sub.gCMode.clone(), real, imag


def _eval_phase(sub: _SubModel, q: float, chi1: float, chi2: float, inv_scaling: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, c_phase = _interp_coeffs(sub, q, chi1, chi2)
    phase = sub.Bphase.T @ c_phase
    freq = sub.gPhase * inv_scaling
    return freq, phase


def _hybridize_phase(m1: float, m2: float, chi1: float, chi2: float, roms: Dict[str, _ModeROM]) -> Tuple[torch.Tensor, torch.Tensor]:
    rom = roms["22"]
    omega_qnm = _qnm_omega(m1, m2, chi1, chi2, 2, 2)
    inv_scaling = omega_qnm / (2 * math.pi)

    q = m1 / m2
    f_lo, ph_lo = _eval_phase(rom.lowf, q, chi1, chi2)
    patch = _select_hf_patch(q, chi1)
    sub_hi = [rom.hqls, rom.hqhs, rom.lqls, rom.lqhs][patch]
    f_hi, ph_hi = _eval_phase(sub_hi, q, chi1, chi2, inv_scaling=inv_scaling)

    i_max, i_min = _compute_i_max_LF_i_min_HF(f_lo.cpu().numpy(), f_hi.cpu().numpy(), _F_HYB_INI)
    f_hyb = torch.cat([f_lo[: i_max + 1], f_hi[i_min:]])

    # align phase: fit delta = ph_lo - ph_hi over window
    f_common = torch.linspace(_F_HYB_INI, _F_HYB_END, 50, device=f_lo.device, dtype=f_lo.dtype)

    def _natural_cubic_coeff(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = x.numel()
        h = x[1:] - x[:-1]
        al = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])
        l = torch.ones(n, device=x.device, dtype=x.dtype)
        mu = torch.zeros(n, device=x.device, dtype=x.dtype)
        z = torch.zeros(n, device=x.device, dtype=x.dtype)
        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (al[i - 1] - h[i - 1] * z[i - 1]) / l[i]
        c = torch.zeros(n, device=x.device, dtype=x.dtype)
        b = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
        d = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        return b, c, d

    def _spline_eval(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # xs increasing
        idx = torch.searchsorted(xs, x.clamp(xs[0], xs[-1])) - 1
        idx = idx.clamp(0, xs.numel() - 2)
        dx = x - xs[idx]
        return ys[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

    b_lo, c_lo, d_lo = _natural_cubic_coeff(f_lo, ph_lo)
    b_hi, c_hi, d_hi = _natural_cubic_coeff(f_hi, ph_hi)

    diff = _spline_eval(f_common, f_lo, ph_lo, b_lo, c_lo, d_lo) - _spline_eval(f_common, f_hi, ph_hi, b_hi, c_hi, d_hi)
    A = torch.stack([torch.ones_like(f_common), f_common], dim=1)
    sol = torch.linalg.lstsq(A, diff.unsqueeze(1)).solution[:2]
    shift = sol[0].item()
    slope = sol[1].item()
    ph_hi_aligned = ph_hi + shift + slope * f_hi

    # blend
    w = _cosine_blend(f_hyb, _F_HYB_INI, _F_HYB_END)
    b_lo_a, c_lo_a, d_lo_a = b_lo, c_lo, d_lo
    b_hi_a, c_hi_a, d_hi_a = _natural_cubic_coeff(f_hi, ph_hi_aligned)
    ph_hyb = (1 - w) * _spline_eval(f_hyb, f_lo, ph_lo, b_lo_a, c_lo_a, d_lo_a) + w * _spline_eval(f_hyb, f_hi, ph_hi_aligned, b_hi_a, c_hi_a, d_hi_a)
    return f_hyb, ph_hyb


def _hybridize_cmode(mode_idx: int, q: float, chi1: float, chi2: float, omega_qnm: float, roms: Dict[str, _ModeROM]) -> Tuple[torch.Tensor, torch.Tensor]:
    mode_name = ["22", "33", "21", "44", "55"][mode_idx]
    rom = roms[mode_name]
    sub_hi_sel = _select_hf_patch(q, chi1)
    sub_hi = [rom.hqls, rom.hqhs, rom.lqls, rom.lqhs][sub_hi_sel]
    inv_scaling = omega_qnm / (2 * math.pi)

    f_lo, re_lo, im_lo = _eval_cmode(rom.lowf, q, chi1, chi2, "LF")
    f_hi, re_hi, im_hi = _eval_cmode(sub_hi, q, chi1, chi2, "HF")
    f_hi = f_hi * inv_scaling

    i_max, i_min = _compute_i_max_LF_i_min_HF(f_lo.cpu().numpy(), f_hi.cpu().numpy(), _F_HYB_INI * _LM_MODES[mode_idx][1])
    f_hyb = torch.cat([f_lo[: i_max + 1], f_hi[i_min:]])

    w = _cosine_blend(f_hyb, _F_HYB_INI * _LM_MODES[mode_idx][1], _F_HYB_END * _LM_MODES[mode_idx][1])

    def spline_coeff(x: torch.Tensor, y: torch.Tensor):
        n = x.numel()
        h = x[1:] - x[:-1]
        al = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])
        l = torch.ones(n, device=x.device, dtype=x.dtype)
        mu = torch.zeros(n, device=x.device, dtype=x.dtype)
        z = torch.zeros(n, device=x.device, dtype=x.dtype)
        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (al[i - 1] - h[i - 1] * z[i - 1]) / l[i]
        c = torch.zeros(n, device=x.device, dtype=x.dtype)
        b = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
        d = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
        return b, c, d

    def spline_eval(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
        idx = torch.searchsorted(xs, x.clamp(xs[0], xs[-1])) - 1
        idx = idx.clamp(0, xs.numel() - 2)
        dx = x - xs[idx]
        return ys[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

    b_re_lo, c_re_lo, d_re_lo = spline_coeff(f_lo, re_lo)
    b_re_hi, c_re_hi, d_re_hi = spline_coeff(f_hi, re_hi)
    b_im_lo, c_im_lo, d_im_lo = spline_coeff(f_lo, im_lo)
    b_im_hi, c_im_hi, d_im_hi = spline_coeff(f_hi, im_hi)

    re = (1 - w) * spline_eval(f_hyb, f_lo, re_lo, b_re_lo, c_re_lo, d_re_lo) + w * spline_eval(f_hyb, f_hi, re_hi, b_re_hi, c_re_hi, d_re_hi)
    im = (1 - w) * spline_eval(f_hyb, f_lo, im_lo, b_im_lo, c_im_lo, d_im_lo) + w * spline_eval(f_hyb, f_hi, im_hi, b_im_hi, c_im_hi, d_im_hi)
    return f_hyb, re + 1j * im


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def seobnrv4hm_fd_torch(**p):
    # masses and spins
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

    # choose dtype/device from active torch scheme if present
    target_dtype = None
    device = None
    try:
        import pycbc.scheme as _scheme  # pylint: disable=import-outside-toplevel

        target_dtype = getattr(getattr(_scheme, "mgr", None).state, "dtype", None)
        device = getattr(getattr(_scheme, "mgr", None).state, "device", None)
        if target_dtype in (torch.complex64, np.complex64):
            target_dtype = torch.float32
        elif target_dtype in (torch.complex128, np.complex128):
            target_dtype = torch.float64
    except Exception:  # pragma: no cover
        target_dtype = None
        device = None

    # ensure mass1 >= mass2 for q
    sign_odd = 1.0
    if m1 < m2:
        m1, m2 = m2, m1
        spin1z, spin2z = spin2z, spin1z
        sign_odd = -1.0

    q = m1 / m2
    eta = m1 * m2 / (m1 + m2) ** 2
    M = m1 + m2
    M_sec = M * lal.MTSUN_SI

    omega_qnm_22 = _qnm_omega(m1, m2, spin1z, spin2z, 2, 2)

    # carrier phase hybrid (22)
    roms = _load_rom(target_dtype, device)
    f_carrier, phase_carrier = _hybridize_phase(m1, m2, spin1z, spin2z, roms)

    # build modes
    fmax_modes = []
    for (l, m), c in zip(_LM_MODES, _CONST_FMAX):
        omega = _qnm_omega(m1, m2, spin1z, spin2z, l, m)
        fmax_modes.append(c * omega / (2 * math.pi))
    fmax_target = f_final if f_final > 0 else max(fmax_modes)
    npts = int(np.ceil(fmax_target / delta_f)) + 1
    rom22 = _load_rom()["22"]
    device = rom22.lowf.qvec.device
    dtype = rom22.lowf.qvec.dtype
    freqs = torch.arange(npts, device=device, dtype=dtype) * delta_f
    hp = torch.zeros_like(freqs, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
    hc = torch.zeros_like(freqs, dtype=hp.dtype)

    for idx, (l, m) in enumerate(_LM_MODES):
        omega_qnm = _qnm_omega(m1, m2, spin1z, spin2z, l, m)
        f_hyb, cmode = _hybridize_cmode(idx, q, spin1z, spin2z, omega_qnm, roms)

        # approx phase from carrier
        const_phase_shift = _CONST_PHASESHIFT[idx] + (1 - m) * math.pi / 4.0
        # natural cubic spline for carrier
        def spline_coeff(x: torch.Tensor, y: torch.Tensor):
            n = x.numel()
            h = x[1:] - x[:-1]
            al = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])
            l = torch.ones(n, device=x.device, dtype=x.dtype)
            mu = torch.zeros(n, device=x.device, dtype=x.dtype)
            z = torch.zeros(n, device=x.device, dtype=x.dtype)
            for i in range(1, n - 1):
                l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
                mu[i] = h[i] / l[i]
                z[i] = (al[i - 1] - h[i - 1] * z[i - 1]) / l[i]
            c = torch.zeros(n, device=x.device, dtype=x.dtype)
            b = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
            d = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
            for j in range(n - 2, -1, -1):
                c[j] = z[j] - mu[j] * c[j + 1]
                b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
                d[j] = (c[j + 1] - c[j]) / (3 * h[j])
            return b, c, d

        def spline_eval(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
            idx = torch.searchsorted(xs, x.clamp(xs[0], xs[-1])) - 1
            idx = idx.clamp(0, xs.numel() - 2)
            dx = x - xs[idx]
            return ys[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

        b_car, c_car, d_car = spline_coeff(f_carrier, phase_carrier)
        phase_approx = torch.empty_like(f_hyb)
        ph_max = phase_carrier[-1]
        der_max = b_car[-1]  # derivative at upper boundary for natural spline
        for i, fh in enumerate(f_hyb):
            if fh / m < f_carrier[-1]:
                phase_approx[i] = m * spline_eval(torch.tensor(fh / m, device=f_carrier.device, dtype=f_carrier.dtype), f_carrier, phase_carrier, b_car, c_car, d_car) + const_phase_shift
            else:
                phase_approx[i] = m * (ph_max + der_max * (fh / m - f_carrier[-1])) + const_phase_shift

        phase_cmode = _unwrap_phase(torch.angle(cmode))
        amp = torch.abs(cmode)
        recon_phase = phase_cmode - phase_approx

        b_amp, c_amp, d_amp = spline_coeff(f_hyb, amp)
        b_ph, c_ph, d_ph = spline_coeff(f_hyb, recon_phase)

        Mf_max = _CONST_FMAX[idx] * omega_qnm / (2 * math.pi)
        for i, f in enumerate(freqs):
            fval = float(f.item())
            if fval < f_lower or fval > Mf_max:
                continue
            if fval <= _MF_LOW_22 * m / 2.0:
                continue
            f_t = torch.tensor(fval, device=freqs.device, dtype=freqs.dtype)
            A = spline_eval(f_t, f_hyb, amp, b_amp, c_amp, d_amp)
            ph = spline_eval(f_t, f_hyb, recon_phase, b_ph, c_ph, d_ph)
            hlm = torch.polar(A.to(hp.real.dtype), ph.to(hp.real.dtype)).to(hp.dtype)
            phase_factor = torch.exp(torch.tensor(-2j * math.pi * fval * 1000.0, device=freqs.device, dtype=hp.dtype))
            hlm *= phase_factor
            hlm *= ((-1) ** l)
            if m % 2 != 0:
                hlm *= sign_odd
            ylm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, l, -m)
            hp[i] += hlm * ylm
            hc[i] -= 1j * hlm * ylm

    amp0 = M * M_sec * lal.MRSUN_SI / distance
    hp *= amp0
    hc *= amp0
    target_complex = torch.complex64 if target_dtype == torch.float32 else torch.complex128
    hp = hp.to(dtype=target_complex)
    hc = hc.to(dtype=target_complex)

    hp_fs = FrequencySeries(TorchArrayData(hp), delta_f=delta_f, epoch=0, copy=False)
    hc_fs = FrequencySeries(TorchArrayData(hc), delta_f=delta_f, epoch=0, copy=False)

    return hp_fs, hc_fs


__all__ = ["seobnrv4hm_fd_torch"]
