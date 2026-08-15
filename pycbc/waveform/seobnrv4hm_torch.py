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
data file ``SEOBNRv4HMROM_v1.0.hdf5`` with PyTorch, without calling
``lalsimulation``. HDF5 loading and scalar setup remain CPU-side; ROM
interpolation, hybridization, and waveform assembly run on the active Torch
device. It mirrors the structure of ``LALSimIMRSEOBNRv4HMROM.c``:

- tensor-product cubic B-splines over (q, chi1, chi2) to interpolate the
  projection coefficients for each ROM patch (low-f plus four high-f patches);
- hybridization of low/high patches via the f_hyb window;
- carrier phase reconstruction and per-mode approximate phasing;
- assembly of (l,-m) modes, time shift correction, and spherical-harmonic
  summation to plus/cross polarizations.

Activation (default is CPU/LAL path):
- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` (or ``PYCBC_TORCH_NATIVE=1``)
- Per-model: ``PYCBC_SEOBNRV4HM_NATIVE=1``

The ROM data file must be available next to this module or on
``$LAL_DATA_PATH``. Both its canonical name, ``SEOBNRv4HMROM_v1.0.hdf5``,
and the legacy installed name, ``SEOBNRv4HMROM.hdf5``, are recognized.
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
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._seobnrv4_qnm import seobnrv4_qnm_omega as _qnm_omega

_ROM_FILENAME = "SEOBNRv4HMROM_v1.0.hdf5"
_ROM_FILENAMES = (_ROM_FILENAME, "SEOBNRv4HMROM.hdf5")
_F_HYB_INI = 0.003
_F_HYB_END = 0.004
_LM_MODES = [(2, 2), (3, 3), (2, 1), (4, 4), (5, 5)]  # ordering used in ROM
_MODE_NAMES = ("22", "33", "21", "44", "55")
_PATCH_NAMES = ("lowf", "hqls", "hqhs", "lqls", "lqhs")
_CONST_PHASESHIFT = [0.0, -math.pi / 2.0, math.pi / 2.0, math.pi, math.pi / 2.0]
_CONST_FMAX = [1.7, 1.55, 1.7, 1.35, 1.25]
_MF_LOW_22 = 0.0004925491025543576


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_rom_file() -> Path:
    search_dirs = [Path(__file__).resolve().parent]
    search_dirs.extend(
        Path(base)
        for base in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if base
    )
    for directory in search_dirs:
        for filename in _ROM_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    names = " or ".join(_ROM_FILENAMES)
    raise FileNotFoundError(
        f"{names} not found; place the ROM next to this module or on " "$LAL_DATA_PATH"
    )


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


def _blend_weight(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    """Return the smooth step used by the LAL ROM hybridization."""

    if x1 <= x0:
        raise ValueError("blend interval must have positive width")
    weight = torch.zeros_like(x)
    weight[x >= x1] = 1.0
    interior = (x > x0) & (x < x1)
    width = x1 - x0
    xi = x[interior]
    weight[interior] = torch.sigmoid(-width / (xi - x0) - width / (xi - x1))
    return weight


def _unwrap_phase(ph: torch.Tensor) -> torch.Tensor:
    d = torch.diff(ph)
    dd = (d + math.pi) % (2 * math.pi) - math.pi
    dd = torch.where((dd == -math.pi) & (d > 0), math.pi, dd)
    ph_unwrapped = torch.cat([ph[:1], ph[0] + torch.cumsum(dd, dim=0)])
    return ph_unwrapped


def _natural_cubic_coeff(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build coefficients for a natural cubic spline."""

    n = x.numel()
    h = x[1:] - x[:-1]
    alpha = (3.0 / h[1:]) * (y[2:] - y[1:-1]) - (3.0 / h[:-1]) * (y[1:-1] - y[:-2])
    diagonal = torch.ones(n, device=x.device, dtype=x.dtype)
    upper = torch.zeros(n, device=x.device, dtype=x.dtype)
    rhs = torch.zeros(n, device=x.device, dtype=x.dtype)
    for i in range(1, n - 1):
        diagonal[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * upper[i - 1]
        upper[i] = h[i] / diagonal[i]
        rhs[i] = (alpha[i - 1] - h[i - 1] * rhs[i - 1]) / diagonal[i]

    quadratic = torch.zeros(n, device=x.device, dtype=x.dtype)
    linear = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
    cubic = torch.zeros(n - 1, device=x.device, dtype=x.dtype)
    for j in range(n - 2, -1, -1):
        quadratic[j] = rhs[j] - upper[j] * quadratic[j + 1]
        linear[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (
            quadratic[j + 1] + 2.0 * quadratic[j]
        ) / 3.0
        cubic[j] = (quadratic[j + 1] - quadratic[j]) / (3.0 * h[j])
    return linear, quadratic, cubic


def _spline_eval(
    x: torch.Tensor,
    knots: torch.Tensor,
    values: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    cubic: torch.Tensor,
) -> torch.Tensor:
    index = torch.searchsorted(knots, x.clamp(knots[0], knots[-1])) - 1
    index = index.clamp(0, knots.numel() - 2)
    offset = x - knots[index]
    return (
        values[index]
        + linear[index] * offset
        + quadratic[index] * offset**2
        + cubic[index] * offset**3
    )


def _spline_derivative_at_end(
    knots: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    cubic: torch.Tensor,
) -> torch.Tensor:
    width = knots[-1] - knots[-2]
    return linear[-1] + 2.0 * quadratic[-2] * width + 3.0 * cubic[-1] * width**2


def _select_hf_patch(q: float, chi1: float) -> int:
    if (q > 3.0) and (chi1 <= 0.8):
        return 0  # hqls
    if (q > 3.0) and (chi1 > 0.8):
        return 1  # hqhs
    if (q <= 3.0) and (chi1 <= 0.8):
        return 2  # lqls
    return 3  # lqhs


def _compute_i_max_LF_i_min_HF(
    freq_lo: torch.Tensor, freq_hi: torch.Tensor, f_hyb_ini: float
) -> Tuple[int, int]:
    i_max = int(torch.searchsorted(freq_lo, f_hyb_ini, right=False)) - 1
    i_min = int(torch.searchsorted(freq_hi, f_hyb_ini, right=False))
    if i_max < 0 or i_min >= freq_hi.numel():
        raise ValueError("ROM patches do not overlap the hybridization window")
    return i_max, i_min


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
        return self.qvec.numel() + 2

    @property
    def nby(self):
        return self.chi1vec.numel() + 2

    @property
    def nbz(self):
        return self.chi2vec.numel() + 2


@dataclass
class _ModeROM:
    lowf: _SubModel
    hqls: _SubModel
    hqhs: _SubModel
    lqls: _SubModel
    lqhs: _SubModel


def _to_rom_tensor(array, target_dtype, device):
    return torch.as_tensor(array, dtype=target_dtype, device=device)


@lru_cache(None)
def _load_shared_rom_data(target_dtype, device):
    """Load parameter grids and carrier phase shared by every mode."""

    path = _find_rom_file()
    shared = {}
    with h5py.File(path, "r") as f:
        for grp_name in _PATCH_NAMES:
            g = f[grp_name]
            phase_grp = g["phase_carrier"]
            shared[grp_name] = (
                _to_rom_tensor(g["qvec"][:], target_dtype, device),
                _to_rom_tensor(g["chi1vec"][:], target_dtype, device),
                _to_rom_tensor(g["chi2vec"][:], target_dtype, device),
                _to_rom_tensor(phase_grp["basis"][:], target_dtype, device),
                _to_rom_tensor(phase_grp["MF_grid"][:], target_dtype, device),
                _to_rom_tensor(
                    phase_grp["coeff"][:].transpose(1, 2, 3, 0),
                    target_dtype,
                    device,
                ),
            )
    return shared


@lru_cache(None)
def _load_mode_rom(mode_name, target_dtype, device):
    """Load one mode while reusing its patch-level carrier-phase tensors."""

    if mode_name not in _MODE_NAMES:
        raise ValueError(f"unknown SEOBNRv4HM ROM mode {mode_name}")

    path = _find_rom_file()
    shared = _load_shared_rom_data(target_dtype, device)
    submodels = {}
    with h5py.File(path, "r") as f:
        for grp_name in _PATCH_NAMES:
            g = f[grp_name]
            cgroup = g["CF_modes"][mode_name]
            (
                qvec,
                chi1vec,
                chi2vec,
                Bphase,
                gPhase,
                coeff_phase,
            ) = shared[grp_name]
            Breal = _to_rom_tensor(cgroup["basis_re"][:], target_dtype, device)
            Bimag = _to_rom_tensor(cgroup["basis_im"][:], target_dtype, device)
            gCMode = _to_rom_tensor(cgroup["MF_grid"][:], target_dtype, device)

            # coeff_re shape (nk, nbx, nby, nbz) after transpose
            coeff_re = _to_rom_tensor(
                cgroup["coeff_re"][:].transpose(1, 2, 3, 0),
                target_dtype,
                device,
            )
            coeff_im = _to_rom_tensor(
                cgroup["coeff_im"][:].transpose(1, 2, 3, 0),
                target_dtype,
                device,
            )

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
    return _ModeROM(**submodels)


def _load_rom(target_dtype, device, active_mode_indices):
    """Load only requested modes, plus mode 22 for the carrier phase."""

    requested = [_MODE_NAMES[index] for index in active_mode_indices]
    mode_names = tuple(dict.fromkeys(("22", *requested)))
    return {
        mode_name: _load_mode_rom(mode_name, target_dtype, device)
        for mode_name in mode_names
    }


# ---------------------------------------------------------------------------
# Core interpolation helpers
# ---------------------------------------------------------------------------


def _parameter_basis(
    sub: _SubModel, q: float, chi1: float, chi2: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sub.qvec.device
    dtype = sub.qvec.dtype
    bx = _bspline_basis(sub.qvec, torch.tensor(q, device=device, dtype=dtype))
    by = _bspline_basis(sub.chi1vec, torch.tensor(chi1, device=device, dtype=dtype))
    bz = _bspline_basis(sub.chi2vec, torch.tensor(chi2, device=device, dtype=dtype))
    return bx, by, bz


def _interpolate_coefficients(
    coefficients: torch.Tensor,
    basis: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return torch.einsum("i,j,k,ijkn->n", *basis, coefficients)


def _eval_cmode(
    sub: _SubModel, q: float, chi1: float, chi2: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = _parameter_basis(sub, q, chi1, chi2)
    c_real = _interpolate_coefficients(sub.cvec_real, basis)
    c_imag = _interpolate_coefficients(sub.cvec_imag, basis)
    real = sub.Breal.T @ c_real
    imag = sub.Bimag.T @ c_imag
    return sub.gCMode.clone(), real, imag


def _eval_phase(
    sub: _SubModel,
    q: float,
    chi1: float,
    chi2: float,
    inv_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    basis = _parameter_basis(sub, q, chi1, chi2)
    c_phase = _interpolate_coefficients(sub.cvec_phase, basis)
    phase = sub.Bphase.T @ c_phase
    freq = sub.gPhase * inv_scaling
    return freq, phase


def _hybridize_phase(
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    roms: Dict[str, _ModeROM],
) -> Tuple[torch.Tensor, torch.Tensor]:
    rom = roms["22"]
    omega_qnm = _qnm_omega(m1, m2, chi1, chi2, 2, 2)
    inv_scaling = omega_qnm / (2 * math.pi)

    q = m1 / m2
    f_lo, ph_lo = _eval_phase(rom.lowf, q, chi1, chi2)
    patch = _select_hf_patch(q, chi1)
    sub_hi = [rom.hqls, rom.hqhs, rom.lqls, rom.lqhs][patch]
    f_hi, ph_hi = _eval_phase(sub_hi, q, chi1, chi2, inv_scaling=inv_scaling)

    i_max, i_min = _compute_i_max_LF_i_min_HF(f_lo, f_hi, _F_HYB_INI)
    f_hyb = torch.cat([f_lo[: i_max + 1], f_hi[i_min:]])

    # LAL aligns the low-frequency phase to the high-frequency phase by
    # fitting phase_hi - phase_lo at ten equally spaced frequencies.
    f_common = torch.linspace(
        _F_HYB_INI,
        _F_HYB_END,
        10,
        device=f_lo.device,
        dtype=f_lo.dtype,
    )
    b_lo, c_lo, d_lo = _natural_cubic_coeff(f_lo, ph_lo)
    b_hi, c_hi, d_hi = _natural_cubic_coeff(f_hi, ph_hi)
    diff = _spline_eval(f_common, f_hi, ph_hi, b_hi, c_hi, d_hi) - _spline_eval(
        f_common, f_lo, ph_lo, b_lo, c_lo, d_lo
    )
    A = torch.stack([torch.ones_like(f_common), f_common], dim=1)
    sol = torch.linalg.lstsq(A, diff.unsqueeze(1)).solution[:2]
    ph_lo_aligned = ph_lo + sol[0, 0] + sol[1, 0] * f_lo
    b_lo, c_lo, d_lo = _natural_cubic_coeff(f_lo, ph_lo_aligned)

    weight = _blend_weight(f_hyb, _F_HYB_INI, _F_HYB_END)
    ph_hyb = (1.0 - weight) * _spline_eval(
        f_hyb, f_lo, ph_lo_aligned, b_lo, c_lo, d_lo
    ) + weight * _spline_eval(f_hyb, f_hi, ph_hi, b_hi, c_hi, d_hi)
    # The ROM stores the opposite carrier-phase convention to the assembled
    # modes. LAL flips it immediately after hybridization.
    return f_hyb, -ph_hyb


def _hybridize_cmode(
    mode_idx: int,
    q: float,
    chi1: float,
    chi2: float,
    omega_qnm: float,
    roms: Dict[str, _ModeROM],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode_name = _MODE_NAMES[mode_idx]
    rom = roms[mode_name]
    sub_hi_sel = _select_hf_patch(q, chi1)
    sub_hi = [rom.hqls, rom.hqhs, rom.lqls, rom.lqhs][sub_hi_sel]
    inv_scaling = omega_qnm / (2 * math.pi)

    f_lo, re_lo, im_lo = _eval_cmode(rom.lowf, q, chi1, chi2)
    f_hi, re_hi, im_hi = _eval_cmode(sub_hi, q, chi1, chi2)
    f_hi = f_hi * inv_scaling

    mode_m = _LM_MODES[mode_idx][1]
    blend_start = _F_HYB_INI * mode_m
    blend_end = _F_HYB_END * mode_m
    i_max, i_min = _compute_i_max_LF_i_min_HF(f_lo, f_hi, blend_start)
    f_hyb = torch.cat([f_lo[: i_max + 1], f_hi[i_min:]])

    weight = _blend_weight(f_hyb, blend_start, blend_end)
    b_re_lo, c_re_lo, d_re_lo = _natural_cubic_coeff(f_lo, re_lo)
    b_re_hi, c_re_hi, d_re_hi = _natural_cubic_coeff(f_hi, re_hi)
    b_im_lo, c_im_lo, d_im_lo = _natural_cubic_coeff(f_lo, im_lo)
    b_im_hi, c_im_hi, d_im_hi = _natural_cubic_coeff(f_hi, im_hi)

    re = (1.0 - weight) * _spline_eval(
        f_hyb, f_lo, re_lo, b_re_lo, c_re_lo, d_re_lo
    ) + weight * _spline_eval(f_hyb, f_hi, re_hi, b_re_hi, c_re_hi, d_re_hi)
    im = (1.0 - weight) * _spline_eval(
        f_hyb, f_lo, im_lo, b_im_lo, c_im_lo, d_im_lo
    ) + weight * _spline_eval(f_hyb, f_hi, im_hi, b_im_hi, c_im_hi, d_im_hi)
    return f_hyb, re + 1j * im


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _active_mode_indices(mode_array) -> Tuple[int, ...]:
    if mode_array is None:
        return tuple(range(len(_LM_MODES)))

    requested = set()
    for mode in mode_array:
        try:
            raw_ell, raw_emm = mode
        except (TypeError, ValueError):
            raise ValueError("mode_array entries must be (l, m) pairs")
        ell, emm = int(raw_ell), int(raw_emm)
        if emm >= 0:
            raise ValueError(
                "SEOBNRv4HM_ROM mode_array accepts only directly modeled "
                "(l, -|m|) modes; positive-m partners are added by symmetry"
            )
        modeled_mode = (ell, -emm)
        if modeled_mode not in _LM_MODES:
            raise ValueError(f"mode ({ell}, {emm}) is not available in SEOBNRv4HM_ROM")
        requested.add(modeled_mode)
    return tuple(index for index, mode in enumerate(_LM_MODES) if mode in requested)


def seobnrv4hm_fd_torch(**p):
    m1 = float(p["mass1"])
    m2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    distance_mpc = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    active_mode_indices = _active_mode_indices(p.get("mode_array"))

    scalar_parameters = {
        "mass1": m1,
        "mass2": m2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "delta_f": delta_f,
        "f_lower": f_lower,
        "f_final": f_final,
        "distance": distance_mpc,
        "inclination": inclination,
        "coa_phase": coa_phase,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("component masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("dimensionless component spins must lie in [-1, 1]")
    if delta_f <= 0.0:
        raise ValueError("delta_f must be positive")
    if distance_mpc <= 0.0:
        raise ValueError("distance must be positive")
    if f_final < 0.0:
        raise ValueError("f_final must be non-negative")
    distance = pnutils.megaparsecs_to_meters(distance_mpc)

    import pycbc.scheme as _scheme  # pylint: disable=import-outside-toplevel

    active_scheme = _scheme.mgr.state
    target_dtype = getattr(active_scheme, "dtype", None)
    device = getattr(active_scheme, "device", None)
    if target_dtype in (
        torch.float32,
        torch.complex64,
        np.float32,
        np.complex64,
    ):
        target_dtype = torch.float32
    elif target_dtype in (
        None,
        torch.float64,
        torch.complex128,
        np.float64,
        np.complex128,
    ):
        target_dtype = torch.float64
    else:
        raise TypeError(f"unsupported SEOBNRv4HM dtype {target_dtype}")
    device = torch.device("cpu" if device is None else device)

    sign_odd = 1.0
    if m1 < m2:
        m1, m2 = m2, m1
        spin1z, spin2z = spin2z, spin1z
        sign_odd = -1.0

    q = m1 / m2
    if q > 50.0:
        raise ValueError("SEOBNRv4HM_ROM requires a mass ratio no greater than 50")
    total_mass = m1 + m2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    f_lower_geom = f_lower * total_mass_seconds
    if f_lower_geom < _MF_LOW_22:
        raise ValueError(
            f"starting frequency M*f_lower={f_lower_geom:g} is below the "
            f"ROM minimum {_MF_LOW_22:g}"
        )

    qnm_omega = {mode: _qnm_omega(m1, m2, spin1z, spin2z, *mode) for mode in _LM_MODES}
    mf_rom_max = _CONST_FMAX[-1] * qnm_omega[_LM_MODES[-1]] / (2.0 * math.pi)
    fmax_target = f_final if f_final > 0.0 else mf_rom_max / total_mass_seconds
    fmax_target_geom = fmax_target * total_mass_seconds
    if fmax_target_geom < _MF_LOW_22:
        raise ValueError("f_final is below the ROM minimum frequency")
    if fmax_target <= f_lower:
        raise ValueError("f_final must be greater than f_lower")

    roms = _load_rom(target_dtype, device, active_mode_indices)
    f_carrier, phase_carrier = _hybridize_phase(m1, m2, spin1z, spin2z, roms)
    rom22 = roms["22"]
    device = rom22.lowf.qvec.device
    dtype = rom22.lowf.qvec.dtype
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    npts = _next_power_of_two(int(fmax_target / delta_f)) + 1
    hp = torch.zeros(npts, device=device, dtype=complex_dtype)
    hc = torch.zeros_like(hp)

    i_start = math.ceil(f_lower / delta_f)
    i_stop = min(math.ceil(fmax_target / delta_f), npts)
    bin_indices = torch.arange(i_start, i_stop, device=device)
    eval_mf = bin_indices.to(dtype=dtype) * (delta_f * total_mass_seconds)

    b_car, c_car, d_car = _natural_cubic_coeff(f_carrier, phase_carrier)
    carrier_max_frequency = f_carrier[-1]
    carrier_max_phase = phase_carrier[-1]
    carrier_end_derivative = _spline_derivative_at_end(f_carrier, b_car, c_car, d_car)
    observer_phi = math.pi / 2.0 - coa_phase

    for idx in active_mode_indices:
        ell, emm = _LM_MODES[idx]
        omega_qnm = qnm_omega[(ell, emm)]
        f_hyb, cmode = _hybridize_cmode(idx, q, spin1z, spin2z, omega_qnm, roms)

        carrier_frequency = f_hyb / emm
        carrier_phase = _spline_eval(
            carrier_frequency,
            f_carrier,
            phase_carrier,
            b_car,
            c_car,
            d_car,
        )
        carrier_extrapolation = carrier_max_phase + carrier_end_derivative * (
            carrier_frequency - carrier_max_frequency
        )
        carrier_phase = torch.where(
            carrier_frequency < carrier_max_frequency,
            carrier_phase,
            carrier_extrapolation,
        )
        const_phase_shift = _CONST_PHASESHIFT[idx] + (1.0 - emm) * math.pi / 4.0
        phase_approx = emm * carrier_phase + const_phase_shift

        phase_cmode = _unwrap_phase(torch.angle(cmode))
        amp = torch.abs(cmode)
        recon_phase = phase_cmode - phase_approx

        b_amp, c_amp, d_amp = _natural_cubic_coeff(f_hyb, amp)
        b_ph, c_ph, d_ph = _natural_cubic_coeff(f_hyb, recon_phase)

        mf_max = _CONST_FMAX[idx] * omega_qnm / (2.0 * math.pi)
        valid = (eval_mf <= mf_max) & (eval_mf > _MF_LOW_22 * emm / 2.0)
        if not torch.any(valid):
            continue
        mode_bins = bin_indices[valid]
        mode_frequencies = eval_mf[valid]
        mode_amplitude = _spline_eval(mode_frequencies, f_hyb, amp, b_amp, c_amp, d_amp)
        mode_phase = _spline_eval(
            mode_frequencies, f_hyb, recon_phase, b_ph, c_ph, d_ph
        )
        hlm = torch.polar(mode_amplitude, mode_phase).to(complex_dtype)
        time_shift = torch.exp((-2j * math.pi * 1000.0) * mode_frequencies).to(
            complex_dtype
        )
        # Store the directly modeled (l,-m) positive-frequency mode in LAL's
        # convention, including the mass-swap sign for odd-m modes.
        hlm = ((-1) ** ell) * torch.conj(hlm * time_shift)
        if emm % 2:
            hlm = hlm * sign_odd

        y_negative = complex(
            lal.SpinWeightedSphericalHarmonic(inclination, observer_phi, -2, ell, -emm)
        )
        y_positive_conjugate = complex(
            lal.SpinWeightedSphericalHarmonic(inclination, observer_phi, -2, ell, emm)
        ).conjugate()
        parity = (-1) ** ell
        factor_plus = 0.5 * (y_negative + parity * y_positive_conjugate)
        factor_cross = 0.5j * (y_negative - parity * y_positive_conjugate)
        hp[mode_bins] += factor_plus * hlm
        hc[mode_bins] += factor_cross * hlm

    amp0 = total_mass * total_mass_seconds * lal.MRSUN_SI / distance
    hp *= amp0
    hc *= amp0
    target_complex = (
        torch.complex64 if target_dtype == torch.float32 else torch.complex128
    )
    hp = hp.to(dtype=target_complex)
    hc = hc.to(dtype=target_complex)

    epoch = -1.0 / delta_f
    hp_fs = FrequencySeries(
        TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False
    )
    hc_fs = FrequencySeries(
        TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False
    )

    return hp_fs, hc_fs


__all__ = ["seobnrv4hm_fd_torch"]
