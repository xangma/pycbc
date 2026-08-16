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
``lalsimulation``. HDF5 loading and scalar model setup remain CPU-side; ROM
interpolation, hybridization, spherical-harmonic evaluation, and waveform
assembly run on the active Torch device. It mirrors the structure of
``LALSimIMRSEOBNRv4HMROM.c``:

- tensor-product cubic B-splines over (q, chi1, chi2) to interpolate the
  projection coefficients for each ROM patch (low-f plus four high-f patches);
- hybridization of low/high patches via the f_hyb window;
- carrier phase reconstruction and per-mode approximate phasing;
- low-frequency TaylorF2 amplitude/phase generation and alignment to the ROM;
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
import pycbc.scheme as _scheme
from pycbc import pnutils
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_eval,
)
from pycbc.waveform._seobnrv4_qnm import seobnrv4_qnm_omega as _qnm_omega
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing

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
_PN_HYBRID_START_FACTOR = 1.01
_PN_HYBRID_END_FACTOR = 2.0
_PN_GRID_HIGH_FACTOR = 1.1
_PN_GRID_ACCURACY = 1.0e-4

_DEFAULT_ONLY_ORDER_KEYS = (
    "phase_order",
    "spin_order",
    "tidal_order",
    "amplitude_order",
    "eccentricity_order",
)
_TRANSVERSE_SPIN_KEYS = ("spin1x", "spin1y", "spin2x", "spin2y")
_TIDAL_KEYS = (
    "lambda1",
    "lambda2",
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


def _is_nonzero(value) -> bool:
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def _is_default_order(value) -> bool:
    try:
        return float(value) == -1.0 and int(value) == -1
    except (TypeError, ValueError, OverflowError):
        return False


def _native_features_supported(params) -> bool:
    """Return whether non-sampling parameters are covered by this port."""

    if params.get("approximant", "SEOBNRv4HM_ROM") != "SEOBNRv4HM_ROM":
        return False
    try:
        active_mode_indices = _active_mode_indices(params.get("mode_array"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not active_mode_indices:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS
            + _TIDAL_KEYS
            + _NON_GR_KEYS
            + (
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    return not params.get("numrel_data", "")


def seobnrv4hm_native_supported(params) -> bool:
    """Return whether regular-grid generation is covered by this port."""

    return _native_features_supported(params)


def seobnrv4hm_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency generation is covered here."""

    return _native_features_supported(params)


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


def _linear_phase_alignment(
    freq_lo: torch.Tensor,
    phase_lo: torch.Tensor,
    freq_hi: torch.Tensor,
    phase_hi: torch.Tensor,
    window_start: float,
    window_end: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align ``phase_lo`` to ``phase_hi`` as LAL's linear fit does."""

    fit_frequency = torch.linspace(
        window_start,
        window_end,
        10,
        device=freq_lo.device,
        dtype=freq_lo.dtype,
    )
    difference = _spline_eval(
        fit_frequency,
        freq_hi,
        phase_hi,
        *_natural_cubic_coeff(freq_hi, phase_hi),
    ) - _spline_eval(
        fit_frequency,
        freq_lo,
        phase_lo,
        *_natural_cubic_coeff(freq_lo, phase_lo),
    )
    centered_frequency = fit_frequency - torch.mean(fit_frequency)
    slope = torch.sum(
        centered_frequency * (difference - torch.mean(difference))
    ) / torch.sum(centered_frequency**2)
    intercept = torch.mean(difference) - slope * torch.mean(fit_frequency)
    aligned = phase_lo + intercept + slope * freq_lo
    return aligned, slope / (2.0 * math.pi), intercept


def _phase_alignment_from_22(
    freq_lo: torch.Tensor,
    phase_lo: torch.Tensor,
    freq_hi: torch.Tensor,
    phase_hi: torch.Tensor,
    window_start: float,
    window_end: float,
    delta_time_22: torch.Tensor,
    delta_phase_22: torch.Tensor,
    mode_m: int,
) -> torch.Tensor:
    """Propagate LAL's 22 alignment and resolve the mode's pi ambiguity."""

    fit_frequency = torch.linspace(
        window_start,
        window_end,
        10,
        device=freq_lo.device,
        dtype=freq_lo.dtype,
    )
    difference = _spline_eval(
        fit_frequency,
        freq_hi,
        phase_hi,
        *_natural_cubic_coeff(freq_hi, phase_hi),
    ) - _spline_eval(
        fit_frequency,
        freq_lo,
        phase_lo,
        *_natural_cubic_coeff(freq_lo, phase_lo),
    )
    alignment = (
        2.0 * math.pi * delta_time_22 * fit_frequency
        + mode_m / 2.0 * delta_phase_22
    )
    average_residual = torch.mean(difference - alignment)
    pi_shift = torch.floor(
        (average_residual + math.pi / 2.0) / math.pi
    ) * math.pi
    return phase_lo + (
        2.0 * math.pi * delta_time_22 * freq_lo
        + mode_m / 2.0 * delta_phase_22
        + pi_shift
    )


def _hybridize_sparse_functions(
    freq_lo: torch.Tensor,
    values_lo: torch.Tensor,
    freq_hi: torch.Tensor,
    values_hi: torch.Tensor,
    window_start: float,
    window_end: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend two sparse functions using LAL's merged-grid convention."""

    i_max, i_min = _compute_i_max_LF_i_min_HF(
        freq_lo, freq_hi, window_start
    )
    frequency = torch.cat([freq_lo[: i_max + 1], freq_hi[i_min:]])
    weight = _blend_weight(frequency, window_start, window_end)
    values = torch.zeros_like(frequency)
    low_coefficients = _natural_cubic_coeff(freq_lo, values_lo)
    high_coefficients = _natural_cubic_coeff(freq_hi, values_hi)

    low_region = frequency <= window_end
    values[low_region] = _spline_eval(
        frequency[low_region],
        freq_lo,
        values_lo,
        *low_coefficients,
    ) * (1.0 - weight[low_region])

    high_region = frequency > window_end
    values[high_region] = _spline_eval(
        frequency[high_region],
        freq_hi,
        values_hi,
        *high_coefficients,
    )
    blend_region = (frequency >= window_start) & (
        frequency <= window_end
    )
    values[blend_region] += _spline_eval(
        frequency[blend_region],
        freq_hi,
        values_hi,
        *high_coefficients,
    ) * weight[blend_region]
    return frequency, values


def _mode_minimum_mf(mode_index: int) -> float:
    """Return the ROM's first geometric frequency for one mode."""

    return _MF_LOW_22 * _LM_MODES[mode_index][1] / 2.0


def _inspiral_minimum_mf(start_mf: float) -> float:
    """Return LAL's lower TaylorF2 spline boundary."""

    return min(start_mf / 2.0, _mode_minimum_mf(2))


def _inspiral_frequency_grid(
    start_mf: float,
    q: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build LAL's geometric grid for TaylorF2 spline interpolation."""

    minimum_mf = _inspiral_minimum_mf(start_mf)
    maximum_mf = (
        _PN_GRID_HIGH_FACTOR
        * _PN_HYBRID_END_FACTOR
        * _mode_minimum_mf(4)
    )
    eta = q / (1.0 + q) ** 2
    spacing = (
        3.8
        * (_PN_GRID_ACCURACY * eta) ** 0.25
        * math.pi ** (5.0 / 12.0)
    )
    transformed_span = minimum_mf ** (-5.0 / 12.0) - maximum_mf ** (
        -5.0 / 12.0
    )
    sample_count = 1 + math.ceil(12.0 / 5.0 / spacing * transformed_span)
    adjusted_spacing = 12.0 / 5.0 / (sample_count - 1) * transformed_span
    indices = torch.arange(sample_count, device=device, dtype=dtype)
    frequency = (
        minimum_mf ** (-5.0 / 12.0)
        - 5.0 / 12.0 * adjusted_spacing * indices
    ) ** (-12.0 / 5.0)
    frequency[0] = minimum_mf
    frequency[-1] = maximum_mf
    return frequency


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
    ph_lo_aligned, _, _ = _linear_phase_alignment(
        f_lo,
        ph_lo,
        f_hi,
        ph_hi,
        _F_HYB_INI,
        _F_HYB_END,
    )
    b_lo, c_lo, d_lo = _natural_cubic_coeff(f_lo, ph_lo_aligned)
    b_hi, c_hi, d_hi = _natural_cubic_coeff(f_hi, ph_hi)

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


@dataclass
class _SEOBNRv4HMInputs:
    """Validated parameters and interpolants shared by both sampling APIs."""

    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    active_mode_indices: Tuple[int, ...]
    sign_odd: float
    q: float
    total_mass: float
    total_mass_seconds: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    qnm_omega: Dict[Tuple[int, int], float]
    mf_rom_max: float
    roms: Dict[str, _ModeROM]
    f_carrier: torch.Tensor
    phase_carrier: torch.Tensor
    pn_phasing: object


def _seobnrv4hm_dtypes(state):
    """Resolve the model dtypes while respecting MPS limitations."""

    device = state.torch_device
    configured = getattr(state, "dtype", None)
    if device.type == "mps":
        real_dtype = torch.float32
    elif configured in (
        torch.float32,
        torch.complex64,
        np.float32,
        np.complex64,
    ):
        real_dtype = torch.float32
    elif configured in (
        None,
        torch.float64,
        torch.complex128,
        np.float64,
        np.complex128,
    ):
        real_dtype = torch.float64
    else:
        raise TypeError(f"unsupported SEOBNRv4HM dtype {configured}")
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return device, real_dtype, complex_dtype


def _seobnrv4hm_inputs(p, *, sequence=False):
    """Validate scalar inputs and reconstruct the selected HM ROM modes."""

    if not _native_features_supported(p):
        raise ValueError(
            "SEOBNRv4HM_ROM parameters are not supported by the native "
            "Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch SEOBNRv4HM_ROM requires TorchScheme")

    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    distance_mpc = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    f_ref = float(p.get("f_ref", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = (
        0.0 if sequence else float(p.get("long_asc_nodes", 0.0))
    )
    active_mode_indices = _active_mode_indices(p.get("mode_array"))

    scalar_parameters = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": distance_mpc,
        "inclination": inclination,
        "coa_phase": coa_phase,
        "f_ref": f_ref,
        "long_asc_nodes": long_asc_nodes,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("dimensionless component spins must lie in [-1, 1]")
    if f_ref < 0.0:
        raise ValueError("f_ref must be non-negative")
    if distance_mpc <= 0.0:
        raise ValueError("distance must be positive")

    sign_odd = 1.0
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        sign_odd = -1.0
    q = mass1 / mass2
    if q > 50.0:
        raise ValueError("SEOBNRv4HM_ROM requires a mass ratio no greater than 50")

    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    device, real_dtype, complex_dtype = _seobnrv4hm_dtypes(state)
    qnm_omega = {
        mode: _qnm_omega(mass1, mass2, spin1z, spin2z, *mode)
        for mode in _LM_MODES
    }
    mf_rom_max = (
        _CONST_FMAX[-1]
        * qnm_omega[_LM_MODES[-1]]
        / (2.0 * math.pi)
    )
    roms = _load_rom(real_dtype, device, active_mode_indices)
    f_carrier, phase_carrier = _hybridize_phase(
        mass1, mass2, spin1z, spin2z, roms
    )
    pn_phasing = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=7,
    )
    return _SEOBNRv4HMInputs(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=pnutils.megaparsecs_to_meters(distance_mpc),
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        active_mode_indices=active_mode_indices,
        sign_odd=sign_odd,
        q=q,
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        qnm_omega=qnm_omega,
        mf_rom_max=mf_rom_max,
        roms=roms,
        f_carrier=f_carrier,
        phase_carrier=phase_carrier,
        pn_phasing=pn_phasing,
    )


@dataclass
class _ModeAmpPhase:
    """Sparse amplitude and phase data for one hybridized mode."""

    amplitude_frequency: torch.Tensor
    amplitude: torch.Tensor
    phase_frequency: torch.Tensor
    phase: torch.Tensor


def _rom_mode_amp_phase(inputs, mode_index: int) -> _ModeAmpPhase:
    """Reconstruct one pure-ROM mode's amplitude and phase splines."""

    ell, emm = _LM_MODES[mode_index]
    f_hyb, cmode = _hybridize_cmode(
        mode_index,
        inputs.q,
        inputs.spin1z,
        inputs.spin2z,
        inputs.qnm_omega[(ell, emm)],
        inputs.roms,
    )
    carrier_coeff = _natural_cubic_coeff(
        inputs.f_carrier, inputs.phase_carrier
    )
    carrier_frequency = f_hyb / emm
    carrier_phase = _spline_eval(
        carrier_frequency,
        inputs.f_carrier,
        inputs.phase_carrier,
        *carrier_coeff,
    )
    carrier_end_derivative = _spline_derivative_at_end(
        inputs.f_carrier, *carrier_coeff
    )
    carrier_extrapolation = inputs.phase_carrier[-1] + (
        carrier_end_derivative
        * (carrier_frequency - inputs.f_carrier[-1])
    )
    carrier_phase = torch.where(
        carrier_frequency < inputs.f_carrier[-1],
        carrier_phase,
        carrier_extrapolation,
    )
    phase_approximation = emm * carrier_phase + (
        _CONST_PHASESHIFT[mode_index]
        + (1.0 - emm) * math.pi / 4.0
    )
    reconstructed_phase = (
        _unwrap_phase(torch.angle(cmode)) - phase_approximation
    )
    return _ModeAmpPhase(
        amplitude_frequency=f_hyb,
        amplitude=torch.abs(cmode),
        phase_frequency=f_hyb,
        phase=reconstructed_phase,
    )


def _taylorf2_mode_amp_phase(
    inputs,
    mode_index: int,
    frequency: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate LAL's leading-order higher-mode TaylorF2 construction."""

    ell, emm = _LM_MODES[mode_index]
    velocity = torch.pow(
        math.pi * (2.0 / emm) * frequency,
        1.0 / 3.0,
    )
    log_velocity = torch.log(velocity)
    orders = torch.arange(
        8,
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    powers = velocity.unsqueeze(-1) ** orders
    coefficients = torch.as_tensor(
        inputs.pn_phasing.v[:8],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    log_coefficients = torch.as_tensor(
        inputs.pn_phasing.vlogv[:8],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    phase = torch.sum(
        (
            coefficients
            + log_coefficients * log_velocity.unsqueeze(-1)
        )
        * powers,
        dim=-1,
    )
    phase = (
        phase / velocity**5 * (emm / 2.0)
        + _CONST_PHASESHIFT[mode_index]
        - math.pi / 4.0
    )

    eta = inputs.q / (1.0 + inputs.q) ** 2
    delta = (
        inputs.q - 1.0 + np.finfo(np.float64).eps
    ) / (1.0 + inputs.q)
    symmetric_spin = 0.5 * (inputs.spin1z + inputs.spin2z)
    antisymmetric_spin = 0.5 * (inputs.spin1z - inputs.spin2z)
    if (ell, emm) == (2, 2):
        mode_factor = torch.ones_like(velocity)
    elif (ell, emm) == (2, 1):
        mode_factor = velocity * (
            delta / 3.0
            - 0.5
            * velocity
            * (antisymmetric_spin + delta * symmetric_spin)
        )
    elif (ell, emm) == (3, 3):
        mode_factor = velocity * 0.75 * math.sqrt(15.0 / 14.0) * delta
    elif (ell, emm) == (4, 4):
        mode_factor = (
            velocity**2
            * 8.0
            * math.sqrt(35.0)
            / 63.0
            * (1.0 - 3.0 * eta)
        )
    else:
        mode_factor = (
            velocity**3
            * 625.0
            * math.sqrt(66.0)
            / 6336.0
            * delta
            * (1.0 - 2.0 * eta)
        )
    amplitude = (
        math.pi
        * math.sqrt(2.0 * eta / 3.0)
        * velocity ** (-3.5)
        * math.sqrt(2.0 / emm)
        * mode_factor
    )
    return amplitude, phase


def _hybridized_mode_data(inputs, start_mf: float):
    """Build the TaylorF2/ROM splines needed by the selected modes."""

    pn_frequency = _inspiral_frequency_grid(
        start_mf,
        inputs.q,
        inputs.device,
        inputs.real_dtype,
    )
    required_modes = (0,) + tuple(
        index for index in inputs.active_mode_indices if index != 0
    )
    mode_data = {}
    delta_time_22 = None
    delta_phase_22 = None

    for mode_index in required_modes:
        rom = _rom_mode_amp_phase(inputs, mode_index)
        pn_amplitude, pn_phase = _taylorf2_mode_amp_phase(
            inputs, mode_index, pn_frequency
        )
        mode_m = _LM_MODES[mode_index][1]
        window_start = (
            _mode_minimum_mf(mode_index) * _PN_HYBRID_START_FACTOR
        )
        window_end = (
            _mode_minimum_mf(mode_index) * _PN_HYBRID_END_FACTOR
        )

        if mode_index == 0:
            aligned_phase, delta_time_22, delta_phase_22 = (
                _linear_phase_alignment(
                    pn_frequency,
                    pn_phase,
                    rom.phase_frequency,
                    rom.phase,
                    window_start,
                    window_end,
                )
            )
        else:
            aligned_phase = _phase_alignment_from_22(
                pn_frequency,
                pn_phase,
                rom.phase_frequency,
                rom.phase,
                window_start,
                window_end,
                delta_time_22,
                delta_phase_22,
                mode_m,
            )
        phase_frequency, hybrid_phase = _hybridize_sparse_functions(
            pn_frequency,
            aligned_phase,
            rom.phase_frequency,
            rom.phase,
            window_start,
            window_end,
        )

        if mode_index not in inputs.active_mode_indices:
            continue
        start = pn_frequency.new_tensor([window_start])
        pn_amplitude_at_start = _spline_eval(
            start,
            pn_frequency,
            pn_amplitude,
            *_natural_cubic_coeff(pn_frequency, pn_amplitude),
        )[0]
        rom_amplitude_at_start = _spline_eval(
            start,
            rom.amplitude_frequency,
            rom.amplitude,
            *_natural_cubic_coeff(
                rom.amplitude_frequency, rom.amplitude
            ),
        )[0]
        scaled_pn_amplitude = (
            pn_amplitude * rom_amplitude_at_start / pn_amplitude_at_start
        )
        amplitude_frequency, hybrid_amplitude = (
            _hybridize_sparse_functions(
                pn_frequency,
                scaled_pn_amplitude,
                rom.amplitude_frequency,
                rom.amplitude,
                window_start,
                window_end,
            )
        )
        mode_data[mode_index] = _ModeAmpPhase(
            amplitude_frequency=amplitude_frequency,
            amplitude=hybrid_amplitude,
            phase_frequency=phase_frequency,
            phase=hybrid_phase,
        )
    return mode_data


def _seobnrv4hm_polarizations(inputs, eval_mf, start_mf: float):
    """Evaluate selected TaylorF2/ROM modes at geometric frequencies."""

    if bool(torch.any(eval_mf < _inspiral_minimum_mf(start_mf))):
        raise ValueError(
            "SEOBNRv4HM_ROM frequency lies below the TaylorF2 spline "
            "domain set by the starting frequency"
        )

    hp = torch.zeros(
        eval_mf.shape,
        device=inputs.device,
        dtype=inputs.complex_dtype,
    )
    hc = torch.zeros_like(hp)
    hybridized_modes = _hybridized_mode_data(inputs, start_mf)
    observer_phi = math.pi / 2.0 - inputs.coa_phase

    for idx in inputs.active_mode_indices:
        ell, emm = _LM_MODES[idx]
        omega_qnm = inputs.qnm_omega[(ell, emm)]
        mode = hybridized_modes[idx]
        amp_coeff = _natural_cubic_coeff(
            mode.amplitude_frequency, mode.amplitude
        )
        phase_coeff = _natural_cubic_coeff(
            mode.phase_frequency, mode.phase
        )

        mf_max = _CONST_FMAX[idx] * omega_qnm / (2.0 * math.pi)
        active = eval_mf <= mf_max
        if not bool(torch.any(active)):
            continue
        mode_frequencies = eval_mf[active]
        mode_amplitude = _spline_eval(
            mode_frequencies,
            mode.amplitude_frequency,
            mode.amplitude,
            *amp_coeff,
        )
        mode_phase = _spline_eval(
            mode_frequencies,
            mode.phase_frequency,
            mode.phase,
            *phase_coeff,
        )
        hlm = torch.complex(
            mode_amplitude * torch.cos(mode_phase),
            mode_amplitude * torch.sin(mode_phase),
        )
        time_shift = torch.exp(
            (-2j * math.pi * 1000.0) * mode_frequencies
        ).to(inputs.complex_dtype)
        # Store the directly modeled (l,-m) positive-frequency mode in LAL's
        # convention, including the mass-swap sign for odd-m modes.
        hlm = ((-1) ** ell) * torch.conj(hlm * time_shift)
        if emm % 2:
            hlm = hlm * inputs.sign_odd

        y_negative = spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            ell,
            -emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        y_positive_conjugate = spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            ell,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ).conj()
        parity = (-1) ** ell
        factor_plus = 0.5 * (
            y_negative + parity * y_positive_conjugate
        )
        factor_cross = 0.5j * (
            y_negative - parity * y_positive_conjugate
        )
        hp[active] += factor_plus * hlm
        hc[active] += factor_cross * hlm

    amplitude_scale = (
        inputs.total_mass
        * inputs.total_mass_seconds
        * lal.MRSUN_SI
        / inputs.distance
    )
    hp *= amplitude_scale
    hc *= amplitude_scale
    if inputs.long_asc_nodes:
        cosine = math.cos(2.0 * inputs.long_asc_nodes)
        sine = math.sin(2.0 * inputs.long_asc_nodes)
        plus = cosine * hp + sine * hc
        cross = cosine * hc - sine * hp
        hp, hc = plus, cross
    return hp, hc


def seobnrv4hm_fd_torch(**p):
    """Generate regular-grid ``SEOBNRv4HM_ROM`` polarizations with Torch."""

    if not seobnrv4hm_native_supported(p):
        raise ValueError(
            "SEOBNRv4HM_ROM parameters require an unsupported feature"
        )
    inputs = _seobnrv4hm_inputs(p)
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    for name, value in (
        ("delta_f", delta_f),
        ("f_lower", f_lower),
        ("f_final", f_final),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("f_final must be non-negative")

    default_final = inputs.mf_rom_max / inputs.total_mass_seconds
    final_frequency = f_final if f_final > 0.0 else default_final
    if final_frequency <= f_lower:
        raise ValueError("f_final (or the ROM cutoff) must exceed f_lower")

    npts = _next_power_of_two(int(final_frequency / delta_f)) + 1
    hp = torch.zeros(
        npts, device=inputs.device, dtype=inputs.complex_dtype
    )
    hc = torch.zeros_like(hp)
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = min(math.ceil(final_frequency / delta_f), npts)
    bin_indices = torch.arange(first_bin, stop_bin, device=inputs.device)
    eval_mf = (
        bin_indices.to(dtype=inputs.real_dtype)
        * delta_f
        * inputs.total_mass_seconds
    )
    plus, cross = _seobnrv4hm_polarizations(
        inputs,
        eval_mf,
        f_lower * inputs.total_mass_seconds,
    )
    hp[first_bin:stop_bin] = plus
    hc[first_bin:stop_bin] = cross

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(
            TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _seobnrv4hm_sequence_frequencies(sample_points, inputs):
    """Return validated arbitrary frequencies on the active Torch device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "SEOBNRv4HM_ROM sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SEOBNRv4HM_ROM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SEOBNRv4HM_ROM sample_points must be positive")
    return frequencies


def seobnrv4hm_fd_sequence_torch(**p):
    """Evaluate ``SEOBNRv4HM_ROM`` at arbitrary frequencies with Torch."""

    if not seobnrv4hm_sequence_native_supported(p):
        raise ValueError(
            "SEOBNRv4HM_ROM sequence parameters require an unsupported feature"
        )
    inputs = _seobnrv4hm_inputs(p, sequence=True)
    frequencies = _seobnrv4hm_sequence_frequencies(
        p["sample_points"], inputs
    )
    plus, cross = _seobnrv4hm_polarizations(
        inputs,
        frequencies * inputs.total_mass_seconds,
        float(frequencies[0]) * inputs.total_mass_seconds,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "seobnrv4hm_fd_sequence_torch",
    "seobnrv4hm_fd_torch",
    "seobnrv4hm_native_supported",
    "seobnrv4hm_sequence_native_supported",
]
