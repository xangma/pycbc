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

"""Device-native evaluator for ``SEOBNRv4_ROM`` and its tidal variants.

The public ROM data are loaded from ``SEOBNRv4ROM_v3.0.hdf5``. Parameter-
space interpolation, sparse-grid reconstruction, frequency interpolation,
and waveform assembly all run with Torch on the active ``TorchScheme``
device. HDF5 input and scalar parameter validation remain host-side.

The implementation follows ``LALSimIMRSEOBNRv4ROM.c`` and
``LALSimIMRSEOBNRv4ROM_NRTidal.c`` from LALSuite 7.26.1, including their
frequency-series layout, polarization normalization, reference phase, and
coalescence-time corrections. The native path includes the multiplicative
tidal-disruption amplitude fit used by ``SEOBNRv4_ROM_NRTidalv2_NSBH``. The
public dispatcher retains the LAL path for unsupported matter parameters and
the distinct time-domain ``SEOBNRv4`` approximant.
"""

from __future__ import annotations

import bisect
import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import h5py
import torch

import lal
import pycbc.scheme as _scheme
from pycbc import pnutils
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._seobnrv4_qnm import seobnrv4_qnm_omega
from pycbc.waveform.nrtidal_torch import (
    nrtidal_amplitude,
    nrtidal_higher_order_spin_phase,
    nrtidal_merger_frequency,
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
    nrtidal_self_spin_phase,
    nrtidal_taper,
    nrtidal_version,
)
from pycbc.waveform.nsbh_torch import seobnrv4_nsbh_amplitude

_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
_MFM = 0.01
_SPLINE_DEGREE = 3


# ---------------------------------------------------------------------------
# Public native-path boundary
# ---------------------------------------------------------------------------


_DEFAULT_ONLY_ORDER_KEYS = (
    "phase_order",
    "spin_order",
    "tidal_order",
    "amplitude_order",
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


def _is_nonnegative_finite(value) -> bool:
    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0.0


def seobnrv4_rom_native_supported(params) -> bool:
    """Return whether ``params`` are covered by the native ROM evaluator."""

    approximant = params.get("approximant", "SEOBNRv4_ROM")
    if approximant not in {
        "SEOBNRv4_ROM",
        "SEOBNRv4_ROM_NRTidal",
        "SEOBNRv4_ROM_NRTidalv2",
        "SEOBNRv4_ROM_NRTidalv2_NSBH",
    }:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in _TRANSVERSE_SPIN_KEYS
        + _TIDAL_EXTENSION_KEYS
        + _NON_GR_KEYS
        + ("eccentricity", "mean_per_ano")
    ):
        return False
    lambdas = (params.get("lambda1", 0.0), params.get("lambda2", 0.0))
    if approximant == "SEOBNRv4_ROM":
        if any(_is_nonzero(value) for value in lambdas):
            return False
    elif not all(_is_nonnegative_finite(value) for value in lambdas):
        return False
    if approximant == "SEOBNRv4_ROM_NRTidalv2_NSBH":
        try:
            mass1 = float(params["mass1"])
            mass2 = float(params["mass2"])
            lambda1 = float(params.get("lambda1") or 0.0)
            lambda2 = float(params.get("lambda2") or 0.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in (mass1, mass2)):
            return False
        if (
            mass1 <= 0.0
            or mass2 <= 0.0
            or mass1 < mass2
            or lambda1 != 0.0
            or lambda2 > 5000.0
            or mass2 > 3.0
            or mass1 / mass2 > 100.0
        ):
            return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in ("frame_axis", "modes_choice", "side_bands")
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


# ---------------------------------------------------------------------------
# ROM data
# ---------------------------------------------------------------------------


def _find_rom_file() -> Path:
    search_dirs = [Path(__file__).resolve().parent]
    search_dirs.extend(
        Path(base)
        for base in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if base
    )
    for directory in search_dirs:
        candidate = directory / _ROM_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{_ROM_FILENAME} not found; place it next to this module or on "
        "$LAL_DATA_PATH"
    )


@dataclass(frozen=True)
class _SubModelMetadata:
    eta_bounds: Tuple[float, float]
    chi1_bounds: Tuple[float, float]
    chi2_bounds: Tuple[float, float]
    amp_bounds: Tuple[float, float]
    phase_bounds: Tuple[float, float]


@dataclass
class _SubModel:
    eta_breaks: Tuple[float, ...]
    chi1_breaks: Tuple[float, ...]
    chi2_breaks: Tuple[float, ...]
    etavec: torch.Tensor
    chi1vec: torch.Tensor
    chi2vec: torch.Tensor
    cvec_amp: torch.Tensor  # (nk_amp, nbx, nby, nbz)
    cvec_phi: torch.Tensor  # (nk_phi, nbx, nby, nbz)
    Bamp: torch.Tensor
    Bphi: torch.Tensor
    gA: torch.Tensor
    gPhi: torch.Tensor


def _to_rom_tensor(array, dtype, device):
    return torch.as_tensor(array, dtype=dtype, device=device)


@lru_cache(None)
def _rom_metadata() -> Dict[str, _SubModelMetadata]:
    metadata = {}
    with h5py.File(_find_rom_file(), "r") as rom:
        for name in ("sub1", "sub2", "sub3"):
            group = rom[name]
            eta = group["etavec"]
            chi1 = group["chi1vec"]
            chi2 = group["chi2vec"]
            amp = group["Mf_grid_Amp"]
            phase = group["Mf_grid_Phi"]
            metadata[name] = _SubModelMetadata(
                eta_bounds=(float(eta[0]), float(eta[-1])),
                chi1_bounds=(float(chi1[0]), float(chi1[-1])),
                chi2_bounds=(float(chi2[0]), float(chi2[-1])),
                amp_bounds=(float(amp[0]), float(amp[-1])),
                phase_bounds=(float(phase[0]), float(phase[-1])),
            )
    return metadata


@lru_cache(None)
def _load_submodel(name: str, dtype: torch.dtype, device: torch.device) -> _SubModel:
    if name not in ("sub1", "sub2", "sub3"):
        raise ValueError(f"unknown SEOBNRv4 ROM submodel {name}")

    with h5py.File(_find_rom_file(), "r") as rom:
        group = rom[name]
        eta = group["etavec"][:]
        chi1 = group["chi1vec"][:]
        chi2 = group["chi2vec"][:]
        nbx = eta.size + 2
        nby = chi1.size + 2
        nbz = chi2.size + 2
        nk_amp = group["Bamp"].shape[0]
        nk_phi = group["Bphase"].shape[0]

        cvec_amp = group["Amp_ciall"][:].reshape(nk_amp, nbx, nby, nbz)
        cvec_phi = group["Phase_ciall"][:].reshape(nk_phi, nbx, nby, nbz)
        return _SubModel(
            eta_breaks=tuple(float(value) for value in eta),
            chi1_breaks=tuple(float(value) for value in chi1),
            chi2_breaks=tuple(float(value) for value in chi2),
            etavec=_to_rom_tensor(eta, dtype, device),
            chi1vec=_to_rom_tensor(chi1, dtype, device),
            chi2vec=_to_rom_tensor(chi2, dtype, device),
            cvec_amp=_to_rom_tensor(cvec_amp, dtype, device),
            cvec_phi=_to_rom_tensor(cvec_phi, dtype, device),
            Bamp=_to_rom_tensor(group["Bamp"][:], dtype, device),
            Bphi=_to_rom_tensor(group["Bphase"][:], dtype, device),
            gA=_to_rom_tensor(group["Mf_grid_Amp"][:], dtype, device),
            gPhi=_to_rom_tensor(group["Mf_grid_Phi"][:], dtype, device),
        )


def _clear_rom_cache() -> None:
    """Release cached ROM tensors, primarily for device-level tests."""

    _load_submodel.cache_clear()
    _rom_metadata.cache_clear()


# ---------------------------------------------------------------------------
# Torch interpolation
# ---------------------------------------------------------------------------


def _bspline_window(
    breaks: Tuple[float, ...], grid: torch.Tensor, value: float
) -> Tuple[int, torch.Tensor]:
    """Return the first index and four nonzero cubic B-spline values."""

    degree = _SPLINE_DEGREE
    ncoeff = len(breaks) + degree - 1
    if value <= breaks[0]:
        span = degree
    elif value >= breaks[-1]:
        span = ncoeff - 1
    else:
        # The clamped knot vector contains ``degree`` extra copies of the
        # first endpoint before the breakpoint sequence.
        span = bisect.bisect_right(breaks, value) - 1 + degree

    knots = torch.cat((grid[0].repeat(degree), grid, grid[-1].repeat(degree)))
    x = torch.as_tensor(value, dtype=grid.dtype, device=grid.device)
    basis = torch.zeros(degree + 1, dtype=grid.dtype, device=grid.device)
    left = torch.zeros_like(basis)
    right = torch.zeros_like(basis)
    basis[0] = 1.0
    for column in range(1, degree + 1):
        left[column] = x - knots[span + 1 - column]
        right[column] = knots[span + column] - x
        saved = torch.zeros((), dtype=grid.dtype, device=grid.device)
        for row in range(column):
            denominator = right[row + 1] + left[column - row]
            term = basis[row] / denominator
            basis[row] = saved + right[row + 1] * term
            saved = left[column - row] * term
        basis[column] = saved
    return span - degree, basis


def _parameter_basis(sub: _SubModel, eta: float, chi1: float, chi2: float):
    ix, bx = _bspline_window(sub.eta_breaks, sub.etavec, eta)
    iy, by = _bspline_window(sub.chi1_breaks, sub.chi1vec, chi1)
    iz, bz = _bspline_window(sub.chi2_breaks, sub.chi2vec, chi2)
    return ix, iy, iz, bx, by, bz


def _interpolate_coefficients(
    coefficients: torch.Tensor, basis
) -> torch.Tensor:
    ix, iy, iz, bx, by, bz = basis
    local = coefficients[:, ix : ix + 4, iy : iy + 4, iz : iz + 4]
    return torch.einsum("nijk,i,j,k->n", local, bx, by, bz)


def _evaluate_submodel(sub: _SubModel, eta: float, chi1: float, chi2: float):
    basis = _parameter_basis(sub, eta, chi1, chi2)
    amp_coeff = _interpolate_coefficients(sub.cvec_amp, basis)
    phase_coeff = _interpolate_coefficients(sub.cvec_phi, basis)
    return sub.Bamp.T @ amp_coeff, sub.Bphi.T @ phase_coeff


def _natural_cubic_coeff(x: torch.Tensor, y: torch.Tensor):
    """Return local coefficients for the GSL-compatible natural spline."""

    count = x.numel()
    width = x[1:] - x[:-1]
    alpha = 3.0 * (
        (y[2:] - y[1:-1]) / width[1:]
        - (y[1:-1] - y[:-2]) / width[:-1]
    )
    diagonal = torch.ones(count, dtype=x.dtype, device=x.device)
    upper = torch.zeros(count, dtype=x.dtype, device=x.device)
    rhs = torch.zeros(count, dtype=x.dtype, device=x.device)
    for index in range(1, count - 1):
        diagonal[index] = (
            2.0 * (x[index + 1] - x[index - 1])
            - width[index - 1] * upper[index - 1]
        )
        upper[index] = width[index] / diagonal[index]
        rhs[index] = (
            alpha[index - 1] - width[index - 1] * rhs[index - 1]
        ) / diagonal[index]

    quadratic = torch.zeros(count, dtype=x.dtype, device=x.device)
    linear = torch.zeros(count - 1, dtype=x.dtype, device=x.device)
    cubic = torch.zeros(count - 1, dtype=x.dtype, device=x.device)
    for index in range(count - 2, -1, -1):
        quadratic[index] = rhs[index] - upper[index] * quadratic[index + 1]
        linear[index] = (y[index + 1] - y[index]) / width[index] - (
            width[index]
            * (quadratic[index + 1] + 2.0 * quadratic[index])
            / 3.0
        )
        cubic[index] = (
            quadratic[index + 1] - quadratic[index]
        ) / (3.0 * width[index])
    return linear, quadratic, cubic


def _spline_interval(points: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(knots, points.clamp(knots[0], knots[-1])) - 1
    return indices.clamp(0, knots.numel() - 2)


def _spline_eval(points, knots, values, linear, quadratic, cubic):
    indices = _spline_interval(points, knots)
    offset = points - knots[indices]
    return (
        values[indices]
        + linear[indices] * offset
        + quadratic[indices] * offset**2
        + cubic[indices] * offset**3
    )


def _spline_derivative(points, knots, linear, quadratic, cubic):
    indices = _spline_interval(points, knots)
    offset = points - knots[indices]
    return (
        linear[indices]
        + 2.0 * quadratic[indices] * offset
        + 3.0 * cubic[indices] * offset**2
    )


def _fit_cubic_at_match(x: torch.Tensor, y: torch.Tensor):
    """Return cubic-fit value and derivative at the gluing frequency."""

    centered = x - _MFM
    design = torch.stack(
        (
            torch.ones_like(centered),
            centered,
            centered**2,
            centered**3,
        ),
        dim=1,
    )
    # Removing the large phase offset before the fit is important in float32:
    # only the value and slope mismatch matter, while the ROM phases themselves
    # are O(1e4).
    phase_center = y.mean()
    normal = design.T @ design
    coefficients = torch.linalg.solve(normal, design.T @ (y - phase_center))
    return coefficients[0] + phase_center, coefficients[1]


def _glue_amplitude(sub_lo, sub_hi, amp_lo, amp_hi):
    j_lo = int(torch.searchsorted(sub_lo.gA, _MFM, right=True)) - 1
    j_hi = int(torch.searchsorted(sub_hi.gA, _MFM, right=True))
    grid = torch.cat((sub_lo.gA[: j_lo + 1], sub_hi.gA[j_hi:]))
    values = torch.cat((amp_lo[: j_lo + 1], amp_hi[j_hi:]))
    return grid, values, _natural_cubic_coeff(grid, values)


def _glue_phase(sub_lo, sub_hi, phase_lo, phase_hi):
    j_lo = int(torch.searchsorted(sub_lo.gPhi, _MFM, right=True)) - 1
    j_hi = int(torch.searchsorted(sub_hi.gPhi, _MFM, right=True))

    low_coeff = _natural_cubic_coeff(sub_lo.gPhi, phase_lo)
    high_window = sub_hi.gPhi[j_hi - 15 : j_hi + 16]
    low_window = _spline_eval(
        high_window, sub_lo.gPhi, phase_lo, *low_coeff
    )
    high_values = phase_hi[j_hi - 15 : j_hi + 16]
    low_value, low_derivative = _fit_cubic_at_match(high_window, low_window)
    high_value, high_derivative = _fit_cubic_at_match(
        high_window, high_values
    )
    delta_value = high_value - low_value
    delta_derivative = high_derivative - low_derivative
    adjusted_high = (
        phase_hi
        - delta_derivative * (sub_hi.gPhi - _MFM)
        - delta_value
    )

    grid = torch.cat((sub_lo.gPhi[: j_lo + 1], sub_hi.gPhi[j_hi:]))
    values = torch.cat((phase_lo[: j_lo + 1], adjusted_high[j_hi:]))
    return grid, values, _natural_cubic_coeff(grid, values)


# ---------------------------------------------------------------------------
# Waveform assembly
# ---------------------------------------------------------------------------


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _nudge_eta(eta: float) -> float:
    for boundary in (0.01, 0.25):
        if math.isclose(eta, boundary, rel_tol=1e-6, abs_tol=0.0):
            return boundary
    return eta


@dataclass
class _SEOBNRv4ROMInputs:
    """Validated inputs and interpolants shared by both sampling APIs."""

    tidal_version: int | None
    is_nsbh: bool
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    lambda1: float
    lambda2: float
    f_ref: float
    distance_mpc: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    total_mass: float
    eta: float
    total_mass_seconds: float
    minimum_mf: float
    maximum_mf: float
    ringdown_mf: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    amp_grid: torch.Tensor
    amp_values: torch.Tensor
    amp_coeff: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    phase_grid: torch.Tensor
    phase_values: torch.Tensor
    phase_coeff: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def seobnrv4_rom_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency ROM generation is native."""

    return seobnrv4_rom_native_supported(params)


def _seobnrv4_rom_inputs(p, *, sequence=False):
    """Validate scalar inputs and reconstruct the parameter-space ROM."""

    if not seobnrv4_rom_native_supported(p):
        raise ValueError(
            "SEOBNRv4_ROM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch SEOBNRv4_ROM requires TorchScheme")

    approximant = p.get("approximant", "SEOBNRv4_ROM")
    tidal_version = nrtidal_version(approximant)
    is_nsbh = approximant == "SEOBNRv4_ROM_NRTidalv2_NSBH"
    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    lambda1 = float(p.get("lambda1") or 0.0)
    lambda2 = float(p.get("lambda2") or 0.0)
    f_ref = float(p.get("f_ref", 0.0))
    distance_mpc = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = (
        0.0 if sequence else float(p.get("long_asc_nodes", 0.0))
    )

    scalars = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "f_ref": f_ref,
        "distance": distance_mpc,
        "inclination": inclination,
        "coa_phase": coa_phase,
        "long_asc_nodes": long_asc_nodes,
    }
    for name, value in scalars.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("dimensionless component spins must lie in [-1, 1]")
    if lambda1 < 0.0 or lambda2 < 0.0:
        raise ValueError("tidal deformabilities must be non-negative")
    if f_ref < 0.0:
        raise ValueError("f_ref must be non-negative")
    if distance_mpc <= 0.0:
        raise ValueError("distance must be positive")

    if is_nsbh:
        if spin2z != 0.0:
            warnings.warn(
                "SEOBNRv4_ROM_NRTidalv2_NSBH is calibrated for zero "
                "neutron-star spin",
                RuntimeWarning,
                stacklevel=3,
            )
        if mass2 < 1.0:
            warnings.warn(
                "SEOBNRv4_ROM_NRTidalv2_NSBH is calibrated for neutron-star "
                "masses of at least one solar mass",
                RuntimeWarning,
                stacklevel=3,
            )
    elif mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        lambda1, lambda2 = lambda2, lambda1

    total_mass = mass1 + mass2
    eta = _nudge_eta(mass1 * mass2 / total_mass**2)
    if not 0.01 <= eta <= 0.25:
        raise ValueError("SEOBNRv4_ROM requires 0.01 <= eta <= 0.25")
    total_mass_seconds = total_mass * lal.MTSUN_SI
    if total_mass > 500.0:
        warnings.warn(
            "SEOBNRv4_ROM can disagree with SEOBNRv4 above 500 solar masses",
            RuntimeWarning,
            stacklevel=3,
        )

    metadata = _rom_metadata()
    high_name = (
        "sub2"
        if spin1z < metadata["sub3"].chi1_bounds[0]
        or eta > metadata["sub3"].eta_bounds[1]
        else "sub3"
    )
    minimum_mf = max(
        metadata["sub1"].amp_bounds[0], metadata["sub1"].phase_bounds[0]
    )
    maximum_mf = min(
        metadata[high_name].amp_bounds[1],
        metadata[high_name].phase_bounds[1],
    )

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    sub_lo = _load_submodel("sub1", real_dtype, device)
    sub_hi = _load_submodel(high_name, real_dtype, device)
    amp_lo, phase_lo = _evaluate_submodel(
        sub_lo, eta, spin1z, spin2z
    )
    amp_hi, phase_hi = _evaluate_submodel(
        sub_hi, eta, spin1z, spin2z
    )
    amp_grid, amp_values, amp_coeff = _glue_amplitude(
        sub_lo, sub_hi, amp_lo, amp_hi
    )
    phase_grid, phase_values, phase_coeff = _glue_phase(
        sub_lo, sub_hi, phase_lo, phase_hi
    )

    ringdown_mf = seobnrv4_qnm_omega(
        mass1, mass2, spin1z, spin2z, 2, 2
    ) / (2.0 * math.pi)
    ringdown_mf = min(ringdown_mf, maximum_mf)
    if ringdown_mf < minimum_mf:
        raise ValueError("SEOBNRv4 ringdown frequency is below the ROM minimum")

    return _SEOBNRv4ROMInputs(
        tidal_version=tidal_version,
        is_nsbh=is_nsbh,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        lambda1=lambda1,
        lambda2=lambda2,
        f_ref=f_ref,
        distance_mpc=distance_mpc,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        total_mass=total_mass,
        eta=eta,
        total_mass_seconds=total_mass_seconds,
        minimum_mf=minimum_mf,
        maximum_mf=maximum_mf,
        ringdown_mf=ringdown_mf,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        amp_grid=amp_grid,
        amp_values=amp_values,
        amp_coeff=amp_coeff,
        phase_grid=phase_grid,
        phase_values=phase_values,
        phase_coeff=phase_coeff,
    )


def _reference_mf(inputs, reference_frequency):
    """Clamp a scalar reference frequency to the ROM interpolation domain."""

    reference = torch.as_tensor(
        reference_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    return (reference * inputs.total_mass_seconds).clamp(
        inputs.minimum_mf, inputs.maximum_mf
    )


def _seobnrv4_rom_amplitude_phase(
    inputs, frequencies_hz, reference_frequency
):
    """Evaluate the inclination-independent base ROM on a device tensor."""

    frequencies_mf = frequencies_hz * inputs.total_mass_seconds
    amplitude = _spline_eval(
        frequencies_mf,
        inputs.amp_grid,
        inputs.amp_values,
        *inputs.amp_coeff,
    )
    if inputs.tidal_version == 2 and not inputs.is_nsbh:
        amplitude += nrtidal_amplitude(
            frequencies_hz,
            inputs.mass1,
            inputs.mass2,
            inputs.lambda1,
            inputs.lambda2,
        )

    phase = _spline_eval(
        frequencies_mf,
        inputs.phase_grid,
        inputs.phase_values,
        *inputs.phase_coeff,
    )
    reference = _reference_mf(inputs, reference_frequency)
    phase_reference = _spline_eval(
        reference,
        inputs.phase_grid,
        inputs.phase_values,
        *inputs.phase_coeff,
    ) - 2.0 * inputs.coa_phase
    ringdown = torch.as_tensor(
        inputs.ringdown_mf,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    time_correction = _spline_derivative(
        ringdown,
        inputs.phase_grid,
        *inputs.phase_coeff,
    ) / (2.0 * math.pi)
    phase -= (
        phase_reference
        + 2.0 * math.pi * (frequencies_mf - reference) * time_correction
    )
    return amplitude, phase


def _nrtidal_correction_phase(inputs, frequencies_hz):
    """Return the NRTidal and spin-induced-quadrupole phase correction."""

    phase = nrtidal_phase(
        frequencies_hz,
        inputs.mass1,
        inputs.mass2,
        inputs.lambda1,
        inputs.lambda2,
        inputs.tidal_version,
    )
    if inputs.is_nsbh:
        return phase, 0.0, 0.0
    dquad1 = nrtidal_quadrupole_from_lambda(inputs.lambda1) - 1.0
    dquad2 = nrtidal_quadrupole_from_lambda(inputs.lambda2) - 1.0
    phase += nrtidal_self_spin_phase(
        frequencies_hz,
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        dquad1,
        dquad2,
    )
    return phase, dquad1, dquad2


def _apply_nrtidal(
    inputs,
    frequencies_hz,
    amplitude,
    phase,
    correction_frequencies,
    tidal_reference_frequency,
):
    """Apply NRTidal corrections and their coalescence-time alignment."""

    correction_phase, dquad1, dquad2 = _nrtidal_correction_phase(
        inputs, correction_frequencies
    )
    if (
        correction_frequencies.shape == frequencies_hz.shape
        and correction_frequencies.data_ptr() == frequencies_hz.data_ptr()
    ):
        active_correction = correction_phase
    else:
        active_correction, _, _ = _nrtidal_correction_phase(
            inputs, frequencies_hz
        )
    phase -= active_correction
    if inputs.tidal_version == 2 and not inputs.is_nsbh:
        phase -= nrtidal_higher_order_spin_phase(
            frequencies_hz,
            inputs.mass1,
            inputs.mass2,
            inputs.spin1z,
            inputs.spin2z,
            dquad1 + 1.0,
            dquad2 + 1.0,
        )

    correction_coeff = _natural_cubic_coeff(
        correction_frequencies, correction_phase
    )
    ringdown_hz = inputs.ringdown_mf / inputs.total_mass_seconds
    correction_ringdown = torch.as_tensor(
        min(ringdown_hz, float(correction_frequencies[-1])),
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    tidal_time_correction = _spline_derivative(
        correction_ringdown,
        correction_frequencies,
        *correction_coeff,
    ) / (2.0 * math.pi)
    phase -= (
        2.0
        * math.pi
        * (frequencies_hz - tidal_reference_frequency)
        * tidal_time_correction
    )
    if inputs.is_nsbh:
        amplitude *= seobnrv4_nsbh_amplitude(
            frequencies_hz,
            inputs.mass1,
            inputs.mass2,
            inputs.spin1z,
            inputs.lambda2,
        )
    else:
        merger_frequency = nrtidal_merger_frequency(
            inputs.mass1,
            inputs.mass2,
            inputs.lambda1,
            inputs.lambda2,
        )
        amplitude *= nrtidal_taper(frequencies_hz, merger_frequency)
    return amplitude, phase


def _seobnrv4_rom_polarizations(inputs, amplitude, phase):
    """Scale and project ROM samples into the two polarizations."""

    distance = pnutils.megaparsecs_to_meters(inputs.distance_mpc)
    amplitude_scale = (
        0.5
        * inputs.total_mass
        * inputs.total_mass_seconds
        * lal.MRSUN_SI
        / distance
    )
    htilde = torch.polar(amplitude_scale * amplitude, phase).to(
        inputs.complex_dtype
    )
    cos_inclination = math.cos(inputs.inclination)
    plus = 0.5 * (1.0 + cos_inclination**2) * htilde
    cross = complex(0.0, -cos_inclination) * htilde
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        cos_nodes * plus + sin_nodes * cross,
        cos_nodes * cross - sin_nodes * plus,
    )


def seobnrv4_fd_torch(**p):
    """Generate Torch ``SEOBNRv4_ROM`` or NRTidal polarizations."""

    inputs = _seobnrv4_rom_inputs(p)
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

    lower_mf = f_lower * inputs.total_mass_seconds
    base_final_hz = f_final
    tidal_final_hz = None
    if inputs.tidal_version is not None and not inputs.is_nsbh:
        merger_frequency = nrtidal_merger_frequency(
            inputs.mass1,
            inputs.mass2,
            inputs.lambda1,
            inputs.lambda2,
        )
        tidal_final_hz = 1.3 * merger_frequency
        if base_final_hz == 0.0 or base_final_hz > tidal_final_hz:
            base_final_hz = tidal_final_hz
    requested_final_mf = base_final_hz * inputs.total_mass_seconds
    if lower_mf < inputs.minimum_mf:
        raise ValueError(
            f"starting frequency M*f_lower={lower_mf:g} is below the "
            f"ROM minimum {inputs.minimum_mf:g}"
        )
    if requested_final_mf == 0.0 or requested_final_mf > inputs.maximum_mf:
        final_mf = inputs.maximum_mf
    elif requested_final_mf < inputs.minimum_mf:
        raise ValueError("f_final is below the ROM minimum frequency")
    else:
        final_mf = requested_final_mf
    if final_mf <= lower_mf:
        raise ValueError("f_final (or the ROM cutoff) must exceed f_lower")

    final_hz = final_mf / inputs.total_mass_seconds
    npts = _next_power_of_two(int(final_hz / delta_f)) + 1
    if tidal_final_hz is not None and f_final > tidal_final_hz:
        # The outer NRTidal wrapper resizes with the untruncated ratio, unlike
        # the base ROM's size_t-valued ``NextPow2`` helper.
        npts = 2 ** math.ceil(math.log2(f_final / delta_f)) + 1
    elif f_final > final_hz:
        npts = _next_power_of_two(int(f_final / delta_f)) + 1
    hp = torch.zeros(npts, dtype=inputs.complex_dtype, device=inputs.device)
    hc = torch.zeros_like(hp)

    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = math.ceil(final_hz / delta_f)
    bin_indices = torch.arange(first_bin, stop_bin, device=inputs.device)
    frequencies_hz = bin_indices.to(inputs.real_dtype) * delta_f
    reference_frequency = inputs.f_ref if inputs.f_ref > 0.0 else f_lower
    amplitude, phase = _seobnrv4_rom_amplitude_phase(
        inputs, frequencies_hz, reference_frequency
    )
    if inputs.tidal_version is not None:
        # The outer wrapper evaluates its correction through the last allocated
        # non-Nyquist bin, including the zero-padded tail, for its time shift.
        correction_bins = torch.arange(
            first_bin, npts - 1, device=inputs.device
        )
        correction_frequencies = (
            correction_bins.to(inputs.real_dtype) * delta_f
        )
        amplitude, phase = _apply_nrtidal(
            inputs,
            frequencies_hz,
            amplitude,
            phase,
            correction_frequencies,
            reference_frequency,
        )
    plus, cross = _seobnrv4_rom_polarizations(inputs, amplitude, phase)
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


def _seobnrv4_sequence_frequencies(sample_points, inputs):
    """Return a validated arbitrary-frequency vector on the active device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() < 2:
        raise ValueError(
            "SEOBNRv4_ROM sample_points must contain at least two frequencies"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SEOBNRv4_ROM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SEOBNRv4_ROM sample_points must be positive")

    frequencies_mf = frequencies * inputs.total_mass_seconds
    if float(frequencies_mf[0]) < inputs.minimum_mf:
        raise ValueError("first sample frequency is below the ROM minimum")
    if bool(torch.any(frequencies_mf < inputs.minimum_mf)):
        raise ValueError("sample frequency is below the ROM minimum")
    final_mf = min(float(frequencies_mf[-1]), inputs.maximum_mf)
    if final_mf <= float(frequencies_mf[0]):
        raise ValueError(
            "last sample frequency must exceed the first within the ROM domain"
        )
    if inputs.tidal_version is not None and not bool(
        torch.all(frequencies[1:] > frequencies[:-1])
    ):
        # The outer LAL NRTidal time-correction spline has the same constraint.
        raise ValueError("NRTidal sample_points must be strictly increasing")
    return frequencies


def seobnrv4_fd_sequence_torch(**p):
    """Evaluate ``SEOBNRv4_ROM`` or NRTidal at arbitrary frequencies."""

    if not seobnrv4_rom_sequence_native_supported(p):
        raise ValueError(
            "SEOBNRv4_ROM sequence parameters are not supported by the "
            "native Torch path"
        )
    inputs = _seobnrv4_rom_inputs(p, sequence=True)
    frequencies_hz = _seobnrv4_sequence_frequencies(
        p["sample_points"], inputs
    )
    reference_frequency = (
        inputs.f_ref if inputs.f_ref > 0.0 else frequencies_hz[0]
    )
    # The ROM returns exact zeros above its geometric upper bound. Clamp only
    # for safe spline evaluation, then restore that support mask at the end.
    frequencies_mf = frequencies_hz * inputs.total_mass_seconds
    active = frequencies_mf <= inputs.maximum_mf
    evaluation_frequencies = frequencies_hz.clamp(
        max=inputs.maximum_mf / inputs.total_mass_seconds
    )
    amplitude, phase = _seobnrv4_rom_amplitude_phase(
        inputs, evaluation_frequencies, reference_frequency
    )
    if inputs.tidal_version is not None:
        amplitude, phase = _apply_nrtidal(
            inputs,
            frequencies_hz,
            amplitude,
            phase,
            frequencies_hz,
            reference_frequency,
        )
    plus, cross = _seobnrv4_rom_polarizations(inputs, amplitude, phase)
    zero = torch.zeros((), dtype=inputs.complex_dtype, device=inputs.device)
    plus = torch.where(active, plus, zero)
    cross = torch.where(active, cross, zero)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "seobnrv4_fd_sequence_torch",
    "seobnrv4_fd_torch",
    "seobnrv4_rom_native_supported",
    "seobnrv4_rom_sequence_native_supported",
]
