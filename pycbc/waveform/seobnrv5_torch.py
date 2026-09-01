# Copyright (C) 2026
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

"""Torch-native evaluators for ``SEOBNRv5_ROM`` and its NRTidalv3 model.

The ROM data are read from the public ``SEOBNRv5ROM_v1.0.hdf5`` file.
HDF5 discovery and validation stay host-side; cached model tensors live on
the requested Torch device. ROM interpolation, TaylorF2 hybridization, and
waveform assembly run on that device.

Supported requests use the native evaluator by default under ``TorchScheme``.
Set ``PYCBC_SEOBNRV5_NATIVE=0`` to opt out for this family, or set
``PYCBC_TORCH_NATIVE_PORTS=0`` (or ``PYCBC_TORCH_NATIVE=0``) to disable
default-native ports whose component switch is unset.

Apple MPS uses single precision. The BBH model retains the LAL path below a
mass-ratio-dependent starting-frequency boundary that limits accumulated
inspiral-phase error; sequence requests use their lowest supplied frequency
for this guard. The NRTidalv3 model retains the LAL path on MPS because its
single-precision error is not bounded by the same monotonic criterion.
"""

from __future__ import annotations

import math
import operator
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch

from pycbc import lal_compat as lal
import pycbc.scheme as _scheme
from pycbc import pnutils
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)
from pycbc.waveform._rom_hybrid_torch import (
    _bspline_window,
    _hybridize_sparse_functions,
    _interpolate_coefficients,
    _linear_phase_alignment,
    _minimum_sequence_frequency,
)
from pycbc.waveform._seobnrv5_qnm import seobnrv5_qnm_omega
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.nrtidal_torch import (
    nrtidal_amplitude,
    nrtidal_higher_order_spin_phase,
    nrtidal_merger_frequency_v3,
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
    nrtidal_self_spin_phase,
    nrtidal_taper,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing


_ROM_FILENAME = "SEOBNRv5ROM_v1.0.hdf5"
_ROM_VERSION = (1, 0, 0)
_PATCH_NAMES = ("lowf", "highf")
_PATCH_VECTOR_SIZES = {
    "lowf": (57, 12, 12),
    "highf": (57, 34, 21),
}
_SPARSE_GRID_SIZE = 300
_SPLINE_DEGREE = 3
_F_HYB_INI = 0.003
_F_HYB_END = 0.004
_MF_LOW_22 = 0.0004925491025543576
_PN_HYBRID_START_FACTOR = 1.01
_PN_HYBRID_END_FACTOR = 2.0
_PN_GRID_HIGH_FACTOR = 1.1
_PN_GRID_ACCURACY = 1.0e-4
_MODE_CUTOFF_FACTOR = 1.7
_OUTPUT_CUTOFF_FACTOR = 1.25
_MAX_MASS_RATIO = 100.0
_MAX_ALIGNED_SPIN = 0.998
_MPS_MIN_EQUAL_MASS_START_MF = 2.0e-3
_NRTIDAL_APPROXIMANT = "SEOBNRv5_ROM_NRTidalv3"
_SUPPORTED_APPROXIMANTS = frozenset(
    {"SEOBNRv5_ROM", _NRTIDAL_APPROXIMANT}
)
_NRTIDAL_MODE_NORMALIZATION = 4.0 * math.sqrt(5.0 / (64.0 * math.pi))

_DEFAULT_ONLY_ORDER_KEYS = (
    "spin_order",
    "tidal_order",
)
_INT_COERCED_ORDER_KEYS = (
    "phase_order",
    "amplitude_order",
)
_EXACT_INT_ORDER_KEYS = (
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
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return False


def _is_int4(value, *, coerce: bool) -> bool:
    """Mirror PyCBC's coercion into a signed LAL ``INT4`` order."""
    try:
        if not coerce and value == -1:
            return True
        value = int(value) if coerce else operator.index(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return False
    return -(1 << 31) <= value < (1 << 31)


def _is_nonnegative_finite(value) -> bool:
    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0.0


def _quadrupole_from_params(lambda_value, dquad_value) -> float:
    """Mirror LAL's SetQuadMonParamsFromLambdas convention."""

    lambda_value = float(lambda_value or 0.0)
    dquad_value = float(dquad_value or 0.0)
    if lambda_value > 0.0 and dquad_value == 0.0:
        return nrtidal_quadrupole_from_lambda(lambda_value)
    return 1.0 + dquad_value


def _dominant_mode_selected(mode_array) -> bool:
    """Validate the LAL mode-array convention for the dominant-mode ROM."""

    if mode_array is None:
        return True
    requested = set()
    for mode in mode_array:
        try:
            raw_ell, raw_emm = mode
        except (TypeError, ValueError):
            raise ValueError("mode_array entries must be (l, m) pairs")
        ell, emm = int(raw_ell), int(raw_emm)
        if emm >= 0:
            raise ValueError(
                "SEOBNRv5_ROM mode_array accepts only the directly modeled "
                "(2, -2) mode"
            )
        if (ell, emm) != (2, -2):
            raise ValueError(f"mode ({ell}, {emm}) is not available in SEOBNRv5_ROM")
        requested.add((ell, emm))
    return (2, -2) in requested


def _native_features_supported(params) -> bool:
    """Return whether non-sampling parameters are covered by this port."""

    approximant = params.get("approximant", "SEOBNRv5_ROM")
    if approximant not in _SUPPORTED_APPROXIMANTS:
        return False
    try:
        mode_selected = _dominant_mode_selected(params.get("mode_array"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not mode_selected:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    # The calibrated ROM accepts these PN-order flags but does not use them.
    if any(
        not _is_int4(params.get(key, -1), coerce=True)
        for key in _INT_COERCED_ORDER_KEYS
    ) or any(
        not _is_int4(params.get(key, -1), coerce=False)
        for key in _EXACT_INT_ORDER_KEYS
    ):
        return False
    zero_only_keys = (
        _TRANSVERSE_SPIN_KEYS
        + _NON_GR_KEYS
        + (
            "eccentricity",
            "mean_per_ano",
            "frame_axis",
            "modes_choice",
            "side_bands",
        )
    )
    if approximant == "SEOBNRv5_ROM":
        zero_only_keys += _TIDAL_KEYS
    else:
        zero_only_keys += _TIDAL_KEYS[4:]
        lambdas = (
            params.get("lambda1", 0.0),
            params.get("lambda2", 0.0),
        )
        if not all(_is_nonnegative_finite(value) for value in lambdas):
            return False
        try:
            quadrupoles = (
                _quadrupole_from_params(
                    lambdas[0], params.get("dquad_mon1", 0.0)
                ),
                _quadrupole_from_params(
                    lambdas[1], params.get("dquad_mon2", 0.0)
                ),
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(
            math.isfinite(value) and value > 0.0
            for value in quadrupoles
        ):
            return False
    if any(_is_nonzero(params.get(key, 0.0)) for key in zero_only_keys):
        return False
    return not params.get("numrel_data", "")


def _native_device_supported(params, *, sequence: bool) -> bool:
    """Bound single-precision phase error for native MPS evaluation."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    ):
        return True
    if params.get("approximant", "SEOBNRv5_ROM") == _NRTIDAL_APPROXIMANT:
        return False

    try:
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        total_mass = mass1 + mass2
        symmetric_mass_ratio = mass1 * mass2 / total_mass**2
        if sequence:
            start_frequency = _minimum_sequence_frequency(params["sample_points"])
        else:
            start_frequency = float(params["f_lower"])
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        ZeroDivisionError,
    ):
        return False
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (total_mass, symmetric_mass_ratio, start_frequency)
    ):
        return False

    # The leading inspiral phase scales as eta^-1 (Mf)^(-5/3), giving the
    # eta^-3/5 boundary for a fixed float32 phase-error target. The coefficient
    # conservatively bounds regular and sequence parity across q <= 100 and
    # the model's full aligned-spin range.
    minimum_start_mf = _MPS_MIN_EQUAL_MASS_START_MF * (
        0.25 / symmetric_mass_ratio
    ) ** (3.0 / 5.0)
    start_mf = total_mass * lal.MTSUN_SI * start_frequency
    return math.isfinite(start_mf) and start_mf >= minimum_start_mf


def seobnrv5_native_supported(params) -> bool:
    """Return whether regular-grid generation is covered by this port."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=False,
    )


def seobnrv5_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency generation is covered here."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=True,
    )


# ---------------------------------------------------------------------------
# ROM data
# ---------------------------------------------------------------------------


def _find_rom_file() -> Path:
    """Find the public v1.0 dominant-mode ROM data file."""

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
        f"{_ROM_FILENAME} not found; place it next to this module or on $LAL_DATA_PATH"
    )


@dataclass(frozen=True)
class _PatchMetadata:
    q_breaks: Tuple[float, ...]
    chi1_breaks: Tuple[float, ...]
    chi2_breaks: Tuple[float, ...]
    cmode_bounds: Tuple[float, float]
    phase_bounds: Tuple[float, float]


@dataclass(frozen=True)
class _RomMetadata:
    path: Path
    patches: Dict[str, _PatchMetadata]


@dataclass
class _HostSubModel:
    qvec: np.ndarray
    chi1vec: np.ndarray
    chi2vec: np.ndarray
    g_cmode: np.ndarray
    g_phase: np.ndarray
    basis_real: np.ndarray
    basis_imag: np.ndarray
    basis_phase: np.ndarray
    coeff_real: np.ndarray
    coeff_imag: np.ndarray
    coeff_phase: np.ndarray


@dataclass
class _SubModel:
    q_breaks: Tuple[float, ...]
    chi1_breaks: Tuple[float, ...]
    chi2_breaks: Tuple[float, ...]
    qvec: torch.Tensor
    chi1vec: torch.Tensor
    chi2vec: torch.Tensor
    g_cmode: torch.Tensor
    g_phase: torch.Tensor
    basis_real: torch.Tensor
    basis_imag: torch.Tensor
    basis_phase: torch.Tensor
    coeff_real: torch.Tensor  # (nk, nbx, nby, nbz)
    coeff_imag: torch.Tensor
    coeff_phase: torch.Tensor


@dataclass
class _RomData:
    lowf: _SubModel
    highf: _SubModel


@dataclass
class _PatchEvaluation:
    cmode_frequency: torch.Tensor
    cmode_real: torch.Tensor
    cmode_imag: torch.Tensor
    phase_frequency: torch.Tensor
    phase: torch.Tensor


@dataclass
class _HybridEvaluation:
    cmode_frequency: torch.Tensor
    cmode: torch.Tensor
    phase_frequency: torch.Tensor
    carrier_phase: torch.Tensor


@dataclass
class _ModeAmpPhase:
    """Sparse amplitude and phase data for the hybridized 22 mode."""

    amplitude_frequency: torch.Tensor
    amplitude: torch.Tensor
    phase_frequency: torch.Tensor
    phase: torch.Tensor


def _attribute_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _require_dataset(group, path: str, shape: Tuple[int, ...]):
    try:
        dataset = group[path]
    except KeyError as exc:
        raise ValueError(
            f"SEOBNRv5 ROM is missing dataset {group.name}/{path}"
        ) from exc
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"SEOBNRv5 ROM object {dataset.name} is not a dataset")
    if dataset.shape != shape:
        raise ValueError(
            f"SEOBNRv5 ROM dataset {dataset.name} has shape {dataset.shape}; "
            f"expected {shape}"
        )
    if not np.issubdtype(dataset.dtype, np.floating):
        raise ValueError(
            f"SEOBNRv5 ROM dataset {dataset.name} has non-floating dtype "
            f"{dataset.dtype}"
        )
    return dataset


def _read_ordered_vector(
    group, path: str, size: int, *, increasing: bool = True
) -> np.ndarray:
    values = _require_dataset(group, path, (size,))[:]
    differences = np.diff(values)
    ordered = differences > 0.0 if increasing else differences < 0.0
    if not np.all(np.isfinite(values)) or not np.all(ordered):
        direction = "increasing" if increasing else "decreasing"
        raise ValueError(
            f"SEOBNRv5 ROM dataset {group.name}/{path} must contain finite, "
            f"strictly {direction} values"
        )
    return values


def _validate_patch(group, name: str) -> _PatchMetadata:
    q_size, chi1_size, chi2_size = _PATCH_VECTOR_SIZES[name]
    qvec = _read_ordered_vector(group, "qvec", q_size)
    etavec = _read_ordered_vector(group, "etavec", q_size, increasing=False)
    chi1vec = _read_ordered_vector(group, "chi1vec", chi1_size)
    chi2vec = _read_ordered_vector(group, "chi2vec", chi2_size)
    expected_eta = qvec / np.square(1.0 + qvec)
    if not np.array_equal(etavec, expected_eta):
        raise ValueError(
            f"SEOBNRv5 ROM dataset {group.name}/etavec is inconsistent with qvec"
        )

    g_cmode = _read_ordered_vector(group, "CF_modes/22/MF_grid", _SPARSE_GRID_SIZE)
    g_phase = _read_ordered_vector(group, "phase_carrier/MF_grid", _SPARSE_GRID_SIZE)
    matrix_shape = (_SPARSE_GRID_SIZE, _SPARSE_GRID_SIZE)
    _require_dataset(group, "CF_modes/22/basis_re", matrix_shape)
    _require_dataset(group, "CF_modes/22/basis_im", matrix_shape)
    _require_dataset(group, "phase_carrier/basis", matrix_shape)

    parameter_size = (q_size + 2) * (chi1_size + 2) * (chi2_size + 2)
    coefficient_shape = (_SPARSE_GRID_SIZE * parameter_size,)
    _require_dataset(group, "CF_modes/22/coeff_re_flattened", coefficient_shape)
    _require_dataset(group, "CF_modes/22/coeff_im_flattened", coefficient_shape)
    _require_dataset(group, "phase_carrier/coeff_flattened", coefficient_shape)

    return _PatchMetadata(
        q_breaks=tuple(float(value) for value in qvec),
        chi1_breaks=tuple(float(value) for value in chi1vec),
        chi2_breaks=tuple(float(value) for value in chi2vec),
        cmode_bounds=(float(g_cmode[0]), float(g_cmode[-1])),
        phase_bounds=(float(g_phase[0]), float(g_phase[-1])),
    )


def _validate_rom_file(path: Path) -> _RomMetadata:
    """Validate the identity and complete layout of a v1.0 ROM file."""

    with h5py.File(path, "r") as rom:
        basename = _attribute_text(rom.attrs.get("CANONICAL_FILE_BASENAME", ""))
        if basename != _ROM_FILENAME:
            raise ValueError(
                "SEOBNRv5 ROM has canonical basename "
                f"{basename!r}; expected {_ROM_FILENAME!r}"
            )
        version = tuple(
            int(rom.attrs.get(f"version_{part}", -1))
            for part in ("major", "minor", "micro")
        )
        if version != _ROM_VERSION:
            raise ValueError(
                f"SEOBNRv5 ROM has version {version}; expected {_ROM_VERSION}"
            )

        patches = {}
        for name in _PATCH_NAMES:
            if name not in rom or not isinstance(rom[name], h5py.Group):
                raise ValueError(f"SEOBNRv5 ROM is missing group /{name}")
            patches[name] = _validate_patch(rom[name], name)
    return _RomMetadata(path=Path(path), patches=patches)


@lru_cache(None)
def _rom_metadata() -> _RomMetadata:
    return _validate_rom_file(_find_rom_file())


def _numpy_dtype(target_dtype: torch.dtype):
    if target_dtype == torch.float32:
        return np.float32
    if target_dtype == torch.float64:
        return np.float64
    raise TypeError(
        "SEOBNRv5 ROM tensors require torch.float32 or torch.float64, "
        f"not {target_dtype}"
    )


@lru_cache(None)
def _load_host_submodel(name: str, target_dtype: torch.dtype) -> _HostSubModel:
    """Load and cache one validated patch in the requested real precision."""

    if name not in _PATCH_NAMES:
        raise ValueError(f"unknown SEOBNRv5 ROM patch {name}")
    metadata = _rom_metadata()
    patch = metadata.patches[name]
    nbx = len(patch.q_breaks) + 2
    nby = len(patch.chi1_breaks) + 2
    nbz = len(patch.chi2_breaks) + 2
    coefficient_shape = (_SPARSE_GRID_SIZE, nbx, nby, nbz)
    dtype = _numpy_dtype(target_dtype)

    def read(group, path, *, reshape=None):
        values = np.asarray(group[path][:], dtype=dtype)
        return values.reshape(reshape) if reshape is not None else values

    with h5py.File(metadata.path, "r") as rom:
        group = rom[name]
        return _HostSubModel(
            qvec=read(group, "qvec"),
            chi1vec=read(group, "chi1vec"),
            chi2vec=read(group, "chi2vec"),
            g_cmode=read(group, "CF_modes/22/MF_grid"),
            g_phase=read(group, "phase_carrier/MF_grid"),
            basis_real=read(group, "CF_modes/22/basis_re"),
            basis_imag=read(group, "CF_modes/22/basis_im"),
            basis_phase=read(group, "phase_carrier/basis"),
            coeff_real=read(
                group,
                "CF_modes/22/coeff_re_flattened",
                reshape=coefficient_shape,
            ),
            coeff_imag=read(
                group,
                "CF_modes/22/coeff_im_flattened",
                reshape=coefficient_shape,
            ),
            coeff_phase=read(
                group,
                "phase_carrier/coeff_flattened",
                reshape=coefficient_shape,
            ),
        )


def _to_rom_tensor(array, dtype, device):
    return torch.as_tensor(array, dtype=dtype, device=device)


@lru_cache(None)
def _load_submodel(
    name: str, target_dtype: torch.dtype, device: torch.device
) -> _SubModel:
    """Return one ROM patch cached for a Torch dtype and device."""

    host = _load_host_submodel(name, target_dtype)
    metadata = _rom_metadata().patches[name]

    def convert(value):
        return _to_rom_tensor(value, target_dtype, device)

    return _SubModel(
        q_breaks=metadata.q_breaks,
        chi1_breaks=metadata.chi1_breaks,
        chi2_breaks=metadata.chi2_breaks,
        qvec=convert(host.qvec),
        chi1vec=convert(host.chi1vec),
        chi2vec=convert(host.chi2vec),
        g_cmode=convert(host.g_cmode),
        g_phase=convert(host.g_phase),
        basis_real=convert(host.basis_real),
        basis_imag=convert(host.basis_imag),
        basis_phase=convert(host.basis_phase),
        coeff_real=convert(host.coeff_real),
        coeff_imag=convert(host.coeff_imag),
        coeff_phase=convert(host.coeff_phase),
    )


@lru_cache(None)
def _load_rom(target_dtype: torch.dtype, device: torch.device) -> _RomData:
    return _RomData(
        lowf=_load_submodel("lowf", target_dtype, device),
        highf=_load_submodel("highf", target_dtype, device),
    )


def _clear_rom_cache() -> None:
    """Release cached host arrays, device tensors, and file metadata."""

    _load_rom.cache_clear()
    _load_submodel.cache_clear()
    _load_host_submodel.cache_clear()
    _rom_metadata.cache_clear()


# ---------------------------------------------------------------------------
# Parameter interpolation and sparse-grid reconstruction
# ---------------------------------------------------------------------------


def _parameter_basis(sub: _SubModel, q: float, chi1: float, chi2: float):
    ix, bx = _bspline_window(sub.q_breaks, sub.qvec, q)
    iy, by = _bspline_window(sub.chi1_breaks, sub.chi1vec, chi1)
    iz, bz = _bspline_window(sub.chi2_breaks, sub.chi2vec, chi2)
    return ix, iy, iz, bx, by, bz


def _evaluate_submodel(
    sub: _SubModel, q: float, chi1: float, chi2: float
) -> _PatchEvaluation:
    """Reconstruct a ROM patch on its two native sparse frequency grids."""

    basis = _parameter_basis(sub, q, chi1, chi2)
    coeff_real = _interpolate_coefficients(sub.coeff_real, basis)
    coeff_imag = _interpolate_coefficients(sub.coeff_imag, basis)
    coeff_phase = _interpolate_coefficients(sub.coeff_phase, basis)
    return _PatchEvaluation(
        cmode_frequency=sub.g_cmode,
        cmode_real=sub.basis_real.T @ coeff_real,
        cmode_imag=sub.basis_imag.T @ coeff_imag,
        phase_frequency=sub.g_phase,
        phase=sub.basis_phase.T @ coeff_phase,
    )


# ---------------------------------------------------------------------------
# Low/high-frequency ROM patch hybridization
# ---------------------------------------------------------------------------


def _hybridize_evaluations(
    low: _PatchEvaluation, high: _PatchEvaluation, omega_qnm: float
) -> _HybridEvaluation:
    """Join reconstructed patches after undoing the high-f QNM scaling."""

    frequency_scale = omega_qnm / (2.0 * math.pi)
    high_phase_frequency = high.phase_frequency * frequency_scale
    aligned_low_phase, _, _ = _linear_phase_alignment(
        low.phase_frequency,
        low.phase,
        high_phase_frequency,
        high.phase,
        _F_HYB_INI,
        _F_HYB_END,
    )
    phase_frequency, carrier_phase = _hybridize_sparse_functions(
        low.phase_frequency,
        aligned_low_phase,
        high_phase_frequency,
        high.phase,
        _F_HYB_INI,
        _F_HYB_END,
    )

    high_cmode_frequency = high.cmode_frequency * frequency_scale
    cmode_window_start = 2.0 * _F_HYB_INI
    cmode_window_end = 2.0 * _F_HYB_END
    cmode_frequency, cmode_real = _hybridize_sparse_functions(
        low.cmode_frequency,
        low.cmode_real,
        high_cmode_frequency,
        high.cmode_real,
        cmode_window_start,
        cmode_window_end,
    )
    imag_frequency, cmode_imag = _hybridize_sparse_functions(
        low.cmode_frequency,
        low.cmode_imag,
        high_cmode_frequency,
        high.cmode_imag,
        cmode_window_start,
        cmode_window_end,
    )
    if not torch.equal(cmode_frequency, imag_frequency):
        raise RuntimeError("real and imaginary c-mode grids do not match")

    # The stored orbital phase has the opposite sign to the carrier phase
    # used when the physical mode is assembled.
    return _HybridEvaluation(
        cmode_frequency=cmode_frequency,
        cmode=torch.complex(cmode_real, cmode_imag),
        phase_frequency=phase_frequency,
        carrier_phase=-carrier_phase,
    )


def _evaluate_sparse_rom(
    data: _RomData,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
) -> _HybridEvaluation:
    """Interpolate, reconstruct, and join both dominant-mode ROM patches."""

    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
    q = mass1 / mass2
    low = _evaluate_submodel(data.lowf, q, spin1z, spin2z)
    high = _evaluate_submodel(data.highf, q, spin1z, spin2z)
    omega_qnm = seobnrv5_qnm_omega(mass1, mass2, spin1z, spin2z, ell=2, emm=2)
    return _hybridize_evaluations(low, high, omega_qnm)


# ---------------------------------------------------------------------------
# Low-frequency TaylorF2/ROM hybridization
# ---------------------------------------------------------------------------


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Unwrap a one-dimensional phase using NumPy/LAL's branch convention."""

    differences = torch.diff(phase)
    corrected = (differences + math.pi) % (2.0 * math.pi) - math.pi
    corrected = torch.where(
        (corrected == -math.pi) & (differences > 0.0),
        math.pi,
        corrected,
    )
    return torch.cat([phase[:1], phase[0] + torch.cumsum(corrected, dim=0)])


def _spline_derivative_at_end(
    knots: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    cubic: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a natural cubic spline's derivative at its final knot."""

    width = knots[-1] - knots[-2]
    return linear[-1] + 2.0 * quadratic[-2] * width + 3.0 * cubic[-1] * width**2


def _rom_mode_amp_phase(evaluation: _HybridEvaluation) -> _ModeAmpPhase:
    """Recover the pure-ROM 22 amplitude and residual phase."""

    carrier_coefficients = _natural_cubic_coeff(
        evaluation.phase_frequency, evaluation.carrier_phase
    )
    carrier_frequency = evaluation.cmode_frequency / 2.0
    carrier_phase = _spline_eval(
        carrier_frequency,
        evaluation.phase_frequency,
        evaluation.carrier_phase,
        *carrier_coefficients,
    )
    carrier_end_derivative = _spline_derivative_at_end(
        evaluation.phase_frequency, *carrier_coefficients
    )
    carrier_extrapolation = evaluation.carrier_phase[-1] + (
        carrier_end_derivative * (carrier_frequency - evaluation.phase_frequency[-1])
    )
    carrier_phase = torch.where(
        carrier_frequency < evaluation.phase_frequency[-1],
        carrier_phase,
        carrier_extrapolation,
    )
    phase_approximation = 2.0 * carrier_phase - math.pi / 4.0
    return _ModeAmpPhase(
        amplitude_frequency=evaluation.cmode_frequency,
        amplitude=torch.abs(evaluation.cmode),
        phase_frequency=evaluation.cmode_frequency,
        phase=_unwrap_phase(torch.angle(evaluation.cmode)) - phase_approximation,
    )


def _inspiral_frequency_grid(
    start_mf: float,
    q: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build LAL's geometric TaylorF2 spline grid for the v5 ROM."""

    if not math.isfinite(start_mf) or start_mf <= 0.0:
        raise ValueError("starting geometric frequency must be positive and finite")
    if not math.isfinite(q) or q < 1.0:
        raise ValueError("mass ratio must be finite and no less than one")

    minimum_mf = min(start_mf / 2.0, _MF_LOW_22 / 2.0)
    maximum_mf = _PN_GRID_HIGH_FACTOR * _PN_HYBRID_END_FACTOR * (_MF_LOW_22 * 5.0 / 2.0)
    eta = q / (1.0 + q) ** 2
    spacing = 3.8 * (_PN_GRID_ACCURACY * eta) ** 0.25 * math.pi ** (5.0 / 12.0)
    transformed_span = minimum_mf ** (-5.0 / 12.0) - maximum_mf ** (-5.0 / 12.0)
    sample_count = 1 + math.ceil(12.0 / 5.0 / spacing * transformed_span)
    adjusted_spacing = 12.0 / 5.0 / (sample_count - 1) * transformed_span
    indices = torch.arange(sample_count, device=device, dtype=dtype)
    frequency = (
        minimum_mf ** (-5.0 / 12.0) - 5.0 / 12.0 * adjusted_spacing * indices
    ) ** (-12.0 / 5.0)
    frequency[0] = minimum_mf
    frequency[-1] = maximum_mf
    return frequency


def _taylorf2_22_amp_phase(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    frequency: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate LAL's leading-order 22-mode TaylorF2 construction."""

    phasing = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=7,
    )
    velocity = torch.pow(math.pi * frequency, 1.0 / 3.0)
    log_velocity = torch.log(velocity)
    orders = torch.arange(8, device=frequency.device, dtype=frequency.dtype)
    powers = velocity.unsqueeze(-1) ** orders
    coefficients = torch.as_tensor(
        phasing.v[:8], device=frequency.device, dtype=frequency.dtype
    )
    log_coefficients = torch.as_tensor(
        phasing.vlogv[:8], device=frequency.device, dtype=frequency.dtype
    )
    phase = torch.sum(
        (coefficients + log_coefficients * log_velocity.unsqueeze(-1)) * powers,
        dim=-1,
    )
    phase = phase / velocity**5 - math.pi / 4.0

    q = mass1 / mass2
    eta = q / (1.0 + q) ** 2
    amplitude = math.pi * math.sqrt(2.0 * eta / 3.0) * velocity ** (-3.5)
    return amplitude, phase


def _hybridize_mode_with_taylorf2(
    evaluation: _HybridEvaluation,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    start_mf: float,
) -> _ModeAmpPhase:
    """Join the reconstructed 22 ROM mode to its TaylorF2 inspiral."""

    for name, value in (
        ("mass1", mass1),
        ("mass2", mass2),
        ("spin1z", spin1z),
        ("spin2z", spin2z),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z

    q = mass1 / mass2
    pn_frequency = _inspiral_frequency_grid(
        start_mf,
        q,
        evaluation.cmode_frequency.device,
        evaluation.cmode_frequency.dtype,
    )
    pn_amplitude, pn_phase = _taylorf2_22_amp_phase(
        mass1, mass2, spin1z, spin2z, pn_frequency
    )
    rom = _rom_mode_amp_phase(evaluation)
    window_start = _MF_LOW_22 * _PN_HYBRID_START_FACTOR
    window_end = _MF_LOW_22 * _PN_HYBRID_END_FACTOR

    aligned_phase, _, _ = _linear_phase_alignment(
        pn_frequency,
        pn_phase,
        rom.phase_frequency,
        rom.phase,
        window_start,
        window_end,
    )
    phase_frequency, hybrid_phase = _hybridize_sparse_functions(
        pn_frequency,
        aligned_phase,
        rom.phase_frequency,
        rom.phase,
        window_start,
        window_end,
    )

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
        *_natural_cubic_coeff(rom.amplitude_frequency, rom.amplitude),
    )[0]
    scaled_pn_amplitude = pn_amplitude * rom_amplitude_at_start / pn_amplitude_at_start
    amplitude_frequency, hybrid_amplitude = _hybridize_sparse_functions(
        pn_frequency,
        scaled_pn_amplitude,
        rom.amplitude_frequency,
        rom.amplitude,
        window_start,
        window_end,
    )
    return _ModeAmpPhase(
        amplitude_frequency=amplitude_frequency,
        amplitude=hybrid_amplitude,
        phase_frequency=phase_frequency,
        phase=hybrid_phase,
    )


def _evaluate_hybridized_mode(
    data: _RomData,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    start_mf: float,
) -> _ModeAmpPhase:
    """Evaluate both v5 ROM patches and attach the TaylorF2 inspiral."""

    evaluation = _evaluate_sparse_rom(data, mass1, mass2, spin1z, spin2z)
    return _hybridize_mode_with_taylorf2(
        evaluation, mass1, mass2, spin1z, spin2z, start_mf
    )


# ---------------------------------------------------------------------------
# Public waveform assembly
# ---------------------------------------------------------------------------


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _inspiral_minimum_mf(start_mf: float) -> float:
    """Return LAL's lower TaylorF2 spline boundary."""

    return min(start_mf / 2.0, _MF_LOW_22 / 2.0)


@dataclass
class _SEOBNRv5Inputs:
    """Validated parameters and ROM tensors shared by both sampling APIs."""

    approximant: str
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    lambda1: float
    lambda2: float
    dquad1: float
    dquad2: float
    distance: float
    inclination: float
    coa_phase: float
    reference_frequency: float
    long_asc_nodes: float
    total_mass: float
    total_mass_seconds: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    qnm_omega_22: float
    mf_rom_max: float
    merger_frequency: float | None
    rom: _RomData

    @property
    def has_nrtidal(self) -> bool:
        return self.approximant == _NRTIDAL_APPROXIMANT


def _seobnrv5_dtypes(state):
    """Resolve model dtypes while respecting MPS limitations."""

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
        raise TypeError(f"unsupported SEOBNRv5 dtype {configured}")
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return device, real_dtype, complex_dtype


def _seobnrv5_inputs(p, *, sequence=False):
    """Validate scalar inputs and prepare the dominant-mode ROM."""

    if not _native_features_supported(p):
        raise ValueError(
            "SEOBNRv5_ROM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch SEOBNRv5_ROM requires TorchScheme")

    approximant = p.get("approximant", "SEOBNRv5_ROM")
    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    lambda1 = float(p.get("lambda1") or 0.0)
    lambda2 = float(p.get("lambda2") or 0.0)
    quadrupole1 = _quadrupole_from_params(
        lambda1, p.get("dquad_mon1", 0.0)
    )
    quadrupole2 = _quadrupole_from_params(
        lambda2, p.get("dquad_mon2", 0.0)
    )
    dquad1 = quadrupole1 - 1.0
    dquad2 = quadrupole2 - 1.0
    distance_mpc = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    f_ref = float(p.get("f_ref", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = 0.0 if sequence else float(p.get("long_asc_nodes", 0.0))

    scalar_parameters = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "dquad1": dquad1,
        "dquad2": dquad2,
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
    if abs(spin1z) > _MAX_ALIGNED_SPIN or abs(spin2z) > _MAX_ALIGNED_SPIN:
        raise ValueError(
            "SEOBNRv5_ROM component spins must lie in [-0.998, 0.998]"
        )
    if distance_mpc <= 0.0:
        raise ValueError("distance must be positive")
    if f_ref < 0.0:
        raise ValueError("f_ref must be non-negative")

    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        lambda1, lambda2 = lambda2, lambda1
        dquad1, dquad2 = dquad2, dquad1
    mass_ratio = mass1 / mass2
    if mass_ratio > _MAX_MASS_RATIO:
        raise ValueError(
            "SEOBNRv5_ROM requires a mass ratio no greater than 100"
        )

    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    device, real_dtype, complex_dtype = _seobnrv5_dtypes(state)
    qnm_omega_22 = seobnrv5_qnm_omega(
        mass1, mass2, spin1z, spin2z, ell=2, emm=2
    )
    qnm_omega_55 = seobnrv5_qnm_omega(
        mass1, mass2, spin1z, spin2z, ell=5, emm=5
    )
    mf_rom_max = _OUTPUT_CUTOFF_FACTOR * qnm_omega_55 / (2.0 * math.pi)
    merger_frequency = None
    if approximant == _NRTIDAL_APPROXIMANT:
        merger_frequency = nrtidal_merger_frequency_v3(
            mass1,
            mass2,
            lambda1,
            lambda2,
            spin1z,
            spin2z,
        )
    return _SEOBNRv5Inputs(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        lambda1=lambda1,
        lambda2=lambda2,
        dquad1=dquad1,
        dquad2=dquad2,
        distance=pnutils.megaparsecs_to_meters(distance_mpc),
        inclination=inclination,
        coa_phase=coa_phase,
        reference_frequency=f_ref,
        long_asc_nodes=long_asc_nodes,
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        qnm_omega_22=qnm_omega_22,
        mf_rom_max=mf_rom_max,
        merger_frequency=merger_frequency,
        rom=_load_rom(real_dtype, device),
    )


def _seobnrv5_nrtidal_factor(
    inputs,
    frequencies: torch.Tensor,
    reference_frequency: float,
) -> torch.Tensor:
    """Return LAL's outer NRTidalv3 phase, taper, and time correction."""

    if frequencies.numel() < 2:
        raise ValueError(
            "SEOBNRv5_ROM_NRTidalv3 requires at least two frequencies"
        )
    if not bool(torch.all(frequencies[1:] > frequencies[:-1])):
        raise ValueError(
            "SEOBNRv5_ROM_NRTidalv3 frequencies must be strictly increasing"
        )

    tidal_phase = nrtidal_phase(
        frequencies,
        inputs.mass1,
        inputs.mass2,
        inputs.lambda1,
        inputs.lambda2,
        3,
        inputs.spin1z,
        inputs.spin2z,
        frequency_series=True,
    )
    correction_phase = tidal_phase + nrtidal_self_spin_phase(
        frequencies,
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        inputs.dquad1,
        inputs.dquad2,
    )
    higher_order_spin_phase = nrtidal_higher_order_spin_phase(
        frequencies,
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        inputs.dquad1 + 1.0,
        inputs.dquad2 + 1.0,
    )
    spline_coefficients = _natural_cubic_coeff(
        frequencies, correction_phase
    )
    alignment_frequency = min(
        inputs.merger_frequency, float(frequencies[-1])
    )
    alignment_point = torch.as_tensor(
        alignment_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    time_correction = _spline_derivative(
        alignment_point,
        frequencies,
        *spline_coefficients,
    ) / (2.0 * math.pi)
    alignment_phase = (
        2.0
        * math.pi
        * (frequencies - reference_frequency)
        * time_correction
    )
    phase = -correction_phase - higher_order_spin_phase + alignment_phase
    taper = nrtidal_taper(frequencies, inputs.merger_frequency)
    return taper.to(inputs.complex_dtype) * torch.exp(1j * phase).to(
        inputs.complex_dtype
    )


def _seobnrv5_polarizations(
    inputs,
    eval_mf,
    start_mf: float,
    *,
    waveform_stop_mf: float | None = None,
    reference_frequency: float | None = None,
):
    """Evaluate the hybridized dominant mode at geometric frequencies."""

    if bool(torch.any(eval_mf < _inspiral_minimum_mf(start_mf))):
        raise ValueError(
            "SEOBNRv5_ROM frequency lies below the TaylorF2 spline domain "
            "set by the starting frequency"
        )

    hp = torch.zeros(
        eval_mf.shape,
        device=inputs.device,
        dtype=inputs.complex_dtype,
    )
    hc = torch.zeros_like(hp)
    mf_max = (
        _MODE_CUTOFF_FACTOR * inputs.qnm_omega_22 / (2.0 * math.pi)
    )
    active = eval_mf <= mf_max
    if waveform_stop_mf is not None:
        active &= eval_mf < waveform_stop_mf
    if bool(torch.any(active)):
        mode = _evaluate_hybridized_mode(
            inputs.rom,
            inputs.mass1,
            inputs.mass2,
            inputs.spin1z,
            inputs.spin2z,
            start_mf,
        )
        mode_frequencies = eval_mf[active]
        mode_amplitude = _spline_eval(
            mode_frequencies,
            mode.amplitude_frequency,
            mode.amplitude,
            *_natural_cubic_coeff(
                mode.amplitude_frequency, mode.amplitude
            ),
        )
        mode_phase = _spline_eval(
            mode_frequencies,
            mode.phase_frequency,
            mode.phase,
            *_natural_cubic_coeff(mode.phase_frequency, mode.phase),
        )
        if inputs.has_nrtidal:
            mode_amplitude += nrtidal_amplitude(
                mode_frequencies / inputs.total_mass_seconds,
                inputs.mass1,
                inputs.mass2,
                inputs.lambda1,
                inputs.lambda2,
            ) / _NRTIDAL_MODE_NORMALIZATION
        hlm = torch.complex(
            mode_amplitude * torch.cos(mode_phase),
            mode_amplitude * torch.sin(mode_phase),
        )
        time_shift = torch.exp(
            (-2j * math.pi * 1000.0) * mode_frequencies
        ).to(inputs.complex_dtype)
        # Convert the reconstructed (2,2) mode to LAL's directly modeled
        # positive-frequency (2,-2) convention.
        hlm = torch.conj(hlm * time_shift)

        observer_phi = math.pi / 2.0 - inputs.coa_phase
        y_negative = spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            2,
            -2,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        y_positive_conjugate = spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            2,
            2,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ).conj()
        hp[active] = 0.5 * (y_negative + y_positive_conjugate) * hlm
        hc[active] = 0.5j * (y_negative - y_positive_conjugate) * hlm

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
    if inputs.has_nrtidal:
        if reference_frequency is None:
            raise ValueError("NRTidalv3 requires a reference frequency")
        correction = _seobnrv5_nrtidal_factor(
            inputs,
            eval_mf / inputs.total_mass_seconds,
            reference_frequency,
        )
        hp *= correction
        hc *= correction
    return hp, hc


def seobnrv5_fd_torch(**p):
    """Generate regular-grid SEOBNRv5 ROM polarizations with Torch."""

    if not seobnrv5_native_supported(p):
        raise ValueError("SEOBNRv5_ROM parameters require an unsupported feature")
    inputs = _seobnrv5_inputs(p)
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

    if inputs.has_nrtidal:
        default_final = 1.2 * inputs.merger_frequency
        generation_final = min(
            f_final if f_final > 0.0 else default_final,
            default_final,
        )
    else:
        default_final = inputs.mf_rom_max / inputs.total_mass_seconds
        generation_final = f_final if f_final > 0.0 else default_final
    final_frequency = f_final if f_final > 0.0 else default_final
    if final_frequency <= f_lower:
        raise ValueError("f_final (or the ROM cutoff) must exceed f_lower")

    if inputs.has_nrtidal:
        npts = _next_power_of_two(
            math.ceil(final_frequency / delta_f)
        ) + 1
    else:
        npts = _next_power_of_two(int(final_frequency / delta_f)) + 1
    hp = torch.zeros(npts, device=inputs.device, dtype=inputs.complex_dtype)
    hc = torch.zeros_like(hp)
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = min(math.ceil(generation_final / delta_f), npts)
    evaluation_stop = npts - 1 if inputs.has_nrtidal else stop_bin
    bin_indices = torch.arange(
        first_bin, evaluation_stop, device=inputs.device
    )
    eval_mf = (
        bin_indices.to(dtype=inputs.real_dtype)
        * delta_f
        * inputs.total_mass_seconds
    )
    plus, cross = _seobnrv5_polarizations(
        inputs,
        eval_mf,
        f_lower * inputs.total_mass_seconds,
        waveform_stop_mf=(
            stop_bin * delta_f * inputs.total_mass_seconds
            if inputs.has_nrtidal
            else None
        ),
        reference_frequency=(
            inputs.reference_frequency or f_lower
            if inputs.has_nrtidal
            else None
        ),
    )
    hp[first_bin:evaluation_stop] = plus
    hc[first_bin:evaluation_stop] = cross

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(
            TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _seobnrv5_sequence_frequencies(sample_points, inputs):
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
        raise ValueError("SEOBNRv5_ROM sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SEOBNRv5_ROM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SEOBNRv5_ROM sample_points must be positive")
    return frequencies


def seobnrv5_fd_sequence_torch(**p):
    """Evaluate an SEOBNRv5 ROM model at arbitrary frequencies with Torch."""

    if not seobnrv5_sequence_native_supported(p):
        raise ValueError(
            "SEOBNRv5_ROM sequence parameters require an unsupported feature"
        )
    inputs = _seobnrv5_inputs(p, sequence=True)
    frequencies = _seobnrv5_sequence_frequencies(p["sample_points"], inputs)
    if inputs.has_nrtidal and not bool(
        torch.all(frequencies[1:] > frequencies[:-1])
    ):
        raise ValueError(
            "SEOBNRv5_ROM_NRTidalv3 sample_points must be strictly increasing"
        )
    plus, cross = _seobnrv5_polarizations(
        inputs,
        frequencies * inputs.total_mass_seconds,
        float(frequencies[0]) * inputs.total_mass_seconds,
        reference_frequency=(
            inputs.reference_frequency or float(frequencies[0])
            if inputs.has_nrtidal
            else None
        ),
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "seobnrv5_fd_sequence_torch",
    "seobnrv5_fd_torch",
    "seobnrv5_native_supported",
    "seobnrv5_sequence_native_supported",
]
