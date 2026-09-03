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

"""Torch-native evaluator for ``SEOBNRv5HM_ROM``.

The public model data stay in HDF5 on the host. Validated coefficient arrays
are loaded lazily by mode, cached in the requested precision, and transferred
to the selected Torch device. The carrier-phase data shared by every mode are
loaded only once per patch, dtype, and device. Parameter interpolation, sparse
basis reconstruction, and low/high-frequency patch hybridization run on that
device. Low-frequency TaylorF2 hybridization, mode assembly, spherical
harmonics, and polarization summation also remain on the selected device.

Supported requests run natively by default under ``TorchScheme``. Set
``PYCBC_SEOBNRV5HM_NATIVE=0`` to opt out. Apple MPS requests outside the
validated single-precision mass-ratio and starting-frequency range retain the
lalsimulation path.
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

from pycbc import lal_compat as lal
import pycbc.scheme as _scheme
from pycbc import pnutils
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
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
from pycbc.waveform.seobnrv5_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _EXACT_INT_ORDER_KEYS,
    _INT_COERCED_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _is_default_order,
    _is_int4,
    _is_nonzero,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing


_ROM_FILENAME = "SEOBNRv5HMROM_v1.0.hdf5"
_ROM_VERSION = (1, 0, 0)
_PATCH_NAMES = ("lowf", "highf")
_PATCH_VECTOR_SIZES = {
    "lowf": (57, 12, 12),
    "highf": (57, 34, 21),
}
_LM_MODES = ((2, 2), (3, 3), (2, 1), (4, 4), (5, 5), (3, 2), (4, 3))
_MODE_NAMES = tuple(f"{ell}{emm}" for ell, emm in _LM_MODES)
_SPARSE_GRID_SIZE = 300
_SPLINE_DEGREE = 3
_F_HYB_INI = 0.003
_F_HYB_END = 0.004
_MF_LOW_22 = 0.0004925491025543576
_PN_HYBRID_START_FACTOR = 1.01
_PN_HYBRID_END_FACTOR = 2.0
_PN_GRID_HIGH_FACTOR = 1.1
_PN_GRID_ACCURACY = 1.0e-4
_TAYLORF2_PHASE_SHIFTS = (
    0.0,
    -math.pi / 2.0,
    math.pi / 2.0,
    math.pi,
    math.pi / 2.0,
    0.0,
    -math.pi / 2.0,
)
_CONST_FMAX = (1.7, 1.55, 1.7, 1.35, 1.25, 1.7, 1.7)
_MAX_MASS_RATIO = 100.0
_MAX_ALIGNED_SPIN = 0.998
_MPS_MAX_MASS_RATIO = 10.0
_MPS_MIN_EQUAL_MASS_START_MF = 1.0e-2


def _active_mode_indices(mode_array) -> Tuple[int, ...]:
    """Validate and normalize LAL's directly modeled mode convention."""

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
                "SEOBNRv5HM_ROM mode_array accepts only directly modeled "
                "(l, -|m|) modes; positive-m partners are added by symmetry"
            )
        modeled_mode = (ell, -emm)
        if modeled_mode not in _LM_MODES:
            raise ValueError(f"mode ({ell}, {emm}) is not available in SEOBNRv5HM_ROM")
        requested.add(modeled_mode)
    return tuple(index for index, mode in enumerate(_LM_MODES) if mode in requested)


def _native_features_supported(params) -> bool:
    """Return whether non-sampling parameters are covered by this port."""

    if params.get("approximant", "SEOBNRv5HM_ROM") != "SEOBNRv5HM_ROM":
        return False
    try:
        _active_mode_indices(params.get("mode_array"))
    except (TypeError, ValueError, OverflowError):
        return False
    if any(
        not _is_default_order(params.get(key, -1)) for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        not _is_int4(params.get(key, -1), coerce=True)
        for key in _INT_COERCED_ORDER_KEYS
    ) or any(
        not _is_int4(params.get(key, -1), coerce=False) for key in _EXACT_INT_ORDER_KEYS
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


def _native_device_supported(params, *, sequence: bool) -> bool:
    """Bound single-precision interpolation and inspiral-phase error on MPS."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme) and state.torch_device.type == "mps"
    ):
        return True

    try:
        if not _active_mode_indices(params.get("mode_array")):
            return True
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        total_mass = mass1 + mass2
        symmetric_mass_ratio = mass1 * mass2 / total_mass**2
        mass_ratio = max(mass1, mass2) / min(mass1, mass2)
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
        for value in (
            total_mass,
            symmetric_mass_ratio,
            mass_ratio,
            start_frequency,
        )
    ):
        return False
    if mass_ratio > _MPS_MAX_MASS_RATIO:
        return False

    # The leading inspiral phase gives the eta^-3/5 frequency scaling. The
    # coefficient and q ceiling conservatively bound regular and sequence
    # parity across every directly modeled mode and the full aligned-spin range.
    minimum_start_mf = _MPS_MIN_EQUAL_MASS_START_MF * (0.25 / symmetric_mass_ratio) ** (
        3.0 / 5.0
    )
    start_mf = total_mass * lal.MTSUN_SI * start_frequency
    return math.isfinite(start_mf) and start_mf >= minimum_start_mf


def seobnrv5hm_native_supported(params) -> bool:
    """Return whether regular-grid generation is covered by this port."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=False,
    )


def seobnrv5hm_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency generation is covered here."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=True,
    )


def _find_rom_file() -> Path:
    """Find the public v1.0 higher-mode ROM data file."""

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
class _ModePatchMetadata:
    cmode_bounds: Tuple[float, float]


@dataclass(frozen=True)
class _PatchMetadata:
    q_breaks: Tuple[float, ...]
    chi1_breaks: Tuple[float, ...]
    chi2_breaks: Tuple[float, ...]
    phase_bounds: Tuple[float, float]
    modes: Dict[str, _ModePatchMetadata]


@dataclass(frozen=True)
class _RomMetadata:
    path: Path
    patches: Dict[str, _PatchMetadata]


@dataclass
class _HostSharedSubModel:
    qvec: np.ndarray
    chi1vec: np.ndarray
    chi2vec: np.ndarray
    g_phase: np.ndarray
    basis_phase: np.ndarray
    coeff_phase: np.ndarray


@dataclass
class _HostModeSubModel:
    g_cmode: np.ndarray
    basis_real: np.ndarray
    basis_imag: np.ndarray
    coeff_real: np.ndarray
    coeff_imag: np.ndarray


@dataclass
class _SharedSubModel:
    qvec: torch.Tensor
    chi1vec: torch.Tensor
    chi2vec: torch.Tensor
    g_phase: torch.Tensor
    basis_phase: torch.Tensor
    coeff_phase: torch.Tensor


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
    coeff_real: torch.Tensor
    coeff_imag: torch.Tensor
    coeff_phase: torch.Tensor


@dataclass
class _ModeROM:
    lowf: _SubModel
    highf: _SubModel


@dataclass
class _PatchPhaseEvaluation:
    frequency: torch.Tensor
    phase: torch.Tensor


@dataclass
class _PatchCModeEvaluation:
    frequency: torch.Tensor
    real: torch.Tensor
    imag: torch.Tensor


@dataclass
class _HybridCModeEvaluation:
    frequency: torch.Tensor
    cmode: torch.Tensor


@dataclass
class _SparseROMEvaluation:
    phase_frequency: torch.Tensor
    carrier_phase: torch.Tensor
    modes: Dict[int, _HybridCModeEvaluation]


@dataclass
class _ModeAmpPhase:
    """Sparse amplitude and phase data for one hybridized physical mode."""

    amplitude_frequency: torch.Tensor
    amplitude: torch.Tensor
    phase_frequency: torch.Tensor
    phase: torch.Tensor


@dataclass
class _SEOBNRv5HMInputs:
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
    total_mass: float
    total_mass_seconds: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    qnm_omega: Dict[Tuple[int, int], float]
    mf_rom_max: float
    evaluation: _SparseROMEvaluation | None


def _attribute_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _require_dataset(group, path: str, shape: Tuple[int, ...]):
    try:
        dataset = group[path]
    except KeyError as exc:
        raise ValueError(
            f"SEOBNRv5HM ROM is missing dataset {group.name}/{path}"
        ) from exc
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"SEOBNRv5HM ROM object {dataset.name} is not a dataset")
    if dataset.shape != shape:
        raise ValueError(
            f"SEOBNRv5HM ROM dataset {dataset.name} has shape "
            f"{dataset.shape}; expected {shape}"
        )
    if not np.issubdtype(dataset.dtype, np.floating):
        raise ValueError(
            f"SEOBNRv5HM ROM dataset {dataset.name} has non-floating dtype "
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
            f"SEOBNRv5HM ROM dataset {group.name}/{path} must contain "
            f"finite, strictly {direction} values"
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
            f"SEOBNRv5HM ROM dataset {group.name}/etavec is inconsistent with qvec"
        )

    phase_grid = _read_ordered_vector(group, "phase_carrier/MF_grid", _SPARSE_GRID_SIZE)
    matrix_shape = (_SPARSE_GRID_SIZE, _SPARSE_GRID_SIZE)
    _require_dataset(group, "phase_carrier/basis", matrix_shape)
    parameter_size = (q_size + 2) * (chi1_size + 2) * (chi2_size + 2)
    coefficient_shape = (_SPARSE_GRID_SIZE * parameter_size,)
    _require_dataset(group, "phase_carrier/coeff_flattened", coefficient_shape)

    modes = {}
    for mode_name in _MODE_NAMES:
        mode_path = f"CF_modes/{mode_name}"
        cmode_grid = _read_ordered_vector(
            group, f"{mode_path}/MF_grid", _SPARSE_GRID_SIZE
        )
        _require_dataset(group, f"{mode_path}/basis_re", matrix_shape)
        _require_dataset(group, f"{mode_path}/basis_im", matrix_shape)
        _require_dataset(group, f"{mode_path}/coeff_re_flattened", coefficient_shape)
        _require_dataset(group, f"{mode_path}/coeff_im_flattened", coefficient_shape)
        modes[mode_name] = _ModePatchMetadata(
            cmode_bounds=(float(cmode_grid[0]), float(cmode_grid[-1]))
        )

    return _PatchMetadata(
        q_breaks=tuple(float(value) for value in qvec),
        chi1_breaks=tuple(float(value) for value in chi1vec),
        chi2_breaks=tuple(float(value) for value in chi2vec),
        phase_bounds=(float(phase_grid[0]), float(phase_grid[-1])),
        modes=modes,
    )


def _validate_rom_file(path: Path) -> _RomMetadata:
    """Validate the identity and complete layout of a v1.0 ROM file."""

    with h5py.File(path, "r") as rom:
        basename = _attribute_text(rom.attrs.get("CANONICAL_FILE_BASENAME", ""))
        if basename != _ROM_FILENAME:
            raise ValueError(
                "SEOBNRv5HM ROM has canonical basename "
                f"{basename!r}; expected {_ROM_FILENAME!r}"
            )
        version = tuple(
            int(rom.attrs.get(f"version_{part}", -1))
            for part in ("major", "minor", "micro")
        )
        if version != _ROM_VERSION:
            raise ValueError(
                f"SEOBNRv5HM ROM has version {version}; expected {_ROM_VERSION}"
            )

        patches = {}
        for name in _PATCH_NAMES:
            if name not in rom or not isinstance(rom[name], h5py.Group):
                raise ValueError(f"SEOBNRv5HM ROM is missing group /{name}")
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
        "SEOBNRv5HM ROM tensors require torch.float32 or torch.float64, "
        f"not {target_dtype}"
    )


@lru_cache(None)
def _load_host_shared_submodel(
    name: str, target_dtype: torch.dtype
) -> _HostSharedSubModel:
    """Load carrier-phase arrays shared by all modes in one patch."""

    if name not in _PATCH_NAMES:
        raise ValueError(f"unknown SEOBNRv5HM ROM patch {name}")
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
        return _HostSharedSubModel(
            qvec=read(group, "qvec"),
            chi1vec=read(group, "chi1vec"),
            chi2vec=read(group, "chi2vec"),
            g_phase=read(group, "phase_carrier/MF_grid"),
            basis_phase=read(group, "phase_carrier/basis"),
            coeff_phase=read(
                group,
                "phase_carrier/coeff_flattened",
                reshape=coefficient_shape,
            ),
        )


@lru_cache(None)
def _load_host_mode_submodel(
    name: str, mode_name: str, target_dtype: torch.dtype
) -> _HostModeSubModel:
    """Load the co-orbital arrays for one mode and patch."""

    if name not in _PATCH_NAMES:
        raise ValueError(f"unknown SEOBNRv5HM ROM patch {name}")
    if mode_name not in _MODE_NAMES:
        raise ValueError(f"unknown SEOBNRv5HM ROM mode {mode_name}")
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
        mode_path = f"CF_modes/{mode_name}"
        return _HostModeSubModel(
            g_cmode=read(group, f"{mode_path}/MF_grid"),
            basis_real=read(group, f"{mode_path}/basis_re"),
            basis_imag=read(group, f"{mode_path}/basis_im"),
            coeff_real=read(
                group,
                f"{mode_path}/coeff_re_flattened",
                reshape=coefficient_shape,
            ),
            coeff_imag=read(
                group,
                f"{mode_path}/coeff_im_flattened",
                reshape=coefficient_shape,
            ),
        )


def _to_rom_tensor(array, dtype, device):
    return torch.as_tensor(array, dtype=dtype, device=device)


@lru_cache(None)
def _load_shared_submodel(
    name: str, target_dtype: torch.dtype, device: torch.device
) -> _SharedSubModel:
    host = _load_host_shared_submodel(name, target_dtype)

    def convert(value):
        return _to_rom_tensor(value, target_dtype, device)

    return _SharedSubModel(
        qvec=convert(host.qvec),
        chi1vec=convert(host.chi1vec),
        chi2vec=convert(host.chi2vec),
        g_phase=convert(host.g_phase),
        basis_phase=convert(host.basis_phase),
        coeff_phase=convert(host.coeff_phase),
    )


@lru_cache(None)
def _load_submodel(
    name: str,
    mode_name: str,
    target_dtype: torch.dtype,
    device: torch.device,
) -> _SubModel:
    """Return one mode and patch cached for a Torch dtype and device."""

    shared = _load_shared_submodel(name, target_dtype, device)
    host_mode = _load_host_mode_submodel(name, mode_name, target_dtype)
    patch = _rom_metadata().patches[name]

    def convert(value):
        return _to_rom_tensor(value, target_dtype, device)

    return _SubModel(
        q_breaks=patch.q_breaks,
        chi1_breaks=patch.chi1_breaks,
        chi2_breaks=patch.chi2_breaks,
        qvec=shared.qvec,
        chi1vec=shared.chi1vec,
        chi2vec=shared.chi2vec,
        g_cmode=convert(host_mode.g_cmode),
        g_phase=shared.g_phase,
        basis_real=convert(host_mode.basis_real),
        basis_imag=convert(host_mode.basis_imag),
        basis_phase=shared.basis_phase,
        coeff_real=convert(host_mode.coeff_real),
        coeff_imag=convert(host_mode.coeff_imag),
        coeff_phase=shared.coeff_phase,
    )


@lru_cache(None)
def _load_mode_rom(
    mode_name: str, target_dtype: torch.dtype, device: torch.device
) -> _ModeROM:
    if mode_name not in _MODE_NAMES:
        raise ValueError(f"unknown SEOBNRv5HM ROM mode {mode_name}")
    return _ModeROM(
        lowf=_load_submodel("lowf", mode_name, target_dtype, device),
        highf=_load_submodel("highf", mode_name, target_dtype, device),
    )


@lru_cache(None)
def _load_rom(
    target_dtype: torch.dtype,
    device: torch.device,
    active_mode_indices: Tuple[int, ...],
) -> Dict[str, _ModeROM]:
    requested = [_MODE_NAMES[index] for index in active_mode_indices]
    mode_names = tuple(dict.fromkeys(("22", *requested)))
    return {
        mode_name: _load_mode_rom(mode_name, target_dtype, device)
        for mode_name in mode_names
    }


def _clear_rom_cache() -> None:
    """Release cached host arrays, device tensors, and file metadata."""

    _load_rom.cache_clear()
    _load_mode_rom.cache_clear()
    _load_submodel.cache_clear()
    _load_shared_submodel.cache_clear()
    _load_host_mode_submodel.cache_clear()
    _load_host_shared_submodel.cache_clear()
    _rom_metadata.cache_clear()


# ---------------------------------------------------------------------------
# Parameter interpolation and sparse-grid reconstruction
# ---------------------------------------------------------------------------


def _parameter_basis(sub: _SubModel, q: float, chi1: float, chi2: float):
    ix, bx = _bspline_window(sub.q_breaks, sub.qvec, q)
    iy, by = _bspline_window(sub.chi1_breaks, sub.chi1vec, chi1)
    iz, bz = _bspline_window(sub.chi2_breaks, sub.chi2vec, chi2)
    return ix, iy, iz, bx, by, bz


def _evaluate_phase_submodel(sub: _SubModel, basis) -> _PatchPhaseEvaluation:
    """Reconstruct one carrier-phase patch on its sparse frequency grid."""

    coefficients = _interpolate_coefficients(sub.coeff_phase, basis)
    return _PatchPhaseEvaluation(
        frequency=sub.g_phase,
        phase=sub.basis_phase.T @ coefficients,
    )


def _evaluate_cmode_submodel(sub: _SubModel, basis) -> _PatchCModeEvaluation:
    """Reconstruct one co-orbital mode patch on its sparse frequency grid."""

    coefficients_real = _interpolate_coefficients(sub.coeff_real, basis)
    coefficients_imag = _interpolate_coefficients(sub.coeff_imag, basis)
    return _PatchCModeEvaluation(
        frequency=sub.g_cmode,
        real=sub.basis_real.T @ coefficients_real,
        imag=sub.basis_imag.T @ coefficients_imag,
    )


def _hybridize_phase(
    low: _PatchPhaseEvaluation,
    high: _PatchPhaseEvaluation,
    omega_qnm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Join the low/high carrier phase after undoing QNM scaling."""

    high_frequency = high.frequency * (omega_qnm / (2.0 * math.pi))
    aligned_low, _, _ = _linear_phase_alignment(
        low.frequency,
        low.phase,
        high_frequency,
        high.phase,
        _F_HYB_INI,
        _F_HYB_END,
    )
    frequency, phase = _hybridize_sparse_functions(
        low.frequency,
        aligned_low,
        high_frequency,
        high.phase,
        _F_HYB_INI,
        _F_HYB_END,
    )
    # The stored orbital phase has the opposite sign to the carrier phase
    # used when the physical modes are assembled.
    return frequency, -phase


def _hybridize_cmode(
    low: _PatchCModeEvaluation,
    high: _PatchCModeEvaluation,
    omega_qnm: float,
    mode_m: int,
) -> _HybridCModeEvaluation:
    """Join the low/high co-orbital mode after undoing QNM scaling."""

    high_frequency = high.frequency * (omega_qnm / (2.0 * math.pi))
    window_start = mode_m * _F_HYB_INI
    window_end = mode_m * _F_HYB_END
    frequency, real = _hybridize_sparse_functions(
        low.frequency,
        low.real,
        high_frequency,
        high.real,
        window_start,
        window_end,
    )
    imag_frequency, imag = _hybridize_sparse_functions(
        low.frequency,
        low.imag,
        high_frequency,
        high.imag,
        window_start,
        window_end,
    )
    if not torch.equal(frequency, imag_frequency):
        raise RuntimeError("real and imaginary c-mode grids do not match")
    return _HybridCModeEvaluation(
        frequency=frequency,
        cmode=torch.complex(real, imag),
    )


def _evaluate_sparse_rom(
    roms: Dict[str, _ModeROM],
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    active_mode_indices: Tuple[int, ...],
) -> _SparseROMEvaluation:
    """Interpolate, reconstruct, and join the requested v5HM ROM modes."""

    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
    q = mass1 / mass2
    requested_indices = tuple(dict.fromkeys(active_mode_indices))
    if any(index < 0 or index >= len(_LM_MODES) for index in requested_indices):
        raise ValueError("active SEOBNRv5HM ROM mode index is out of range")
    # The 22 mode fixes the common TaylorF2 time and phase alignment even when
    # the caller requests only a higher-mode subset.
    mode_indices = (0,) + tuple(index for index in requested_indices if index != 0)

    carrier_rom = roms["22"]
    low_basis = _parameter_basis(carrier_rom.lowf, q, spin1z, spin2z)
    high_basis = _parameter_basis(carrier_rom.highf, q, spin1z, spin2z)
    low_phase = _evaluate_phase_submodel(carrier_rom.lowf, low_basis)
    high_phase = _evaluate_phase_submodel(carrier_rom.highf, high_basis)
    omega_22 = seobnrv5_qnm_omega(mass1, mass2, spin1z, spin2z, ell=2, emm=2)
    phase_frequency, carrier_phase = _hybridize_phase(low_phase, high_phase, omega_22)

    modes = {}
    for mode_index in mode_indices:
        ell, emm = _LM_MODES[mode_index]
        mode_rom = roms[_MODE_NAMES[mode_index]]
        low_cmode = _evaluate_cmode_submodel(mode_rom.lowf, low_basis)
        high_cmode = _evaluate_cmode_submodel(mode_rom.highf, high_basis)
        omega_qnm = seobnrv5_qnm_omega(mass1, mass2, spin1z, spin2z, ell=ell, emm=emm)
        modes[mode_index] = _hybridize_cmode(low_cmode, high_cmode, omega_qnm, emm)

    return _SparseROMEvaluation(
        phase_frequency=phase_frequency,
        carrier_phase=carrier_phase,
        modes=modes,
    )


# ---------------------------------------------------------------------------
# Physical-mode reconstruction and low-frequency TaylorF2 hybridization
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


def _phase_alignment_from_22(
    frequency_low: torch.Tensor,
    phase_low: torch.Tensor,
    frequency_high: torch.Tensor,
    phase_high: torch.Tensor,
    window_start: float,
    window_end: float,
    delta_time_22: torch.Tensor,
    delta_phase_22: torch.Tensor,
    mode_m: int,
) -> torch.Tensor:
    """Propagate the 22 alignment and resolve a mode's pi ambiguity."""

    fit_frequency = torch.linspace(
        window_start,
        window_end,
        10,
        device=frequency_low.device,
        dtype=frequency_low.dtype,
    )
    difference = _spline_eval(
        fit_frequency,
        frequency_high,
        phase_high,
        *_natural_cubic_coeff(frequency_high, phase_high),
    ) - _spline_eval(
        fit_frequency,
        frequency_low,
        phase_low,
        *_natural_cubic_coeff(frequency_low, phase_low),
    )
    alignment = (
        2.0 * math.pi * delta_time_22 * fit_frequency + mode_m / 2.0 * delta_phase_22
    )
    average_residual = torch.mean(difference - alignment)
    pi_shift = torch.floor((average_residual + math.pi / 2.0) / math.pi) * math.pi
    return phase_low + (
        2.0 * math.pi * delta_time_22 * frequency_low
        + mode_m / 2.0 * delta_phase_22
        + pi_shift
    )


def _mode_minimum_mf(mode_index: int) -> float:
    """Return the first geometric ROM frequency for one modeled mode."""

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
    """Build LAL's geometric TaylorF2 interpolation grid."""

    if not math.isfinite(start_mf) or start_mf <= 0.0:
        raise ValueError("starting geometric frequency must be positive and finite")
    if not math.isfinite(q) or q < 1.0:
        raise ValueError("mass ratio must be finite and no less than one")

    minimum_mf = _inspiral_minimum_mf(start_mf)
    maximum_mf = _PN_GRID_HIGH_FACTOR * _PN_HYBRID_END_FACTOR * _mode_minimum_mf(4)
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


def _rom_mode_amp_phase(
    evaluation: _SparseROMEvaluation, mode_index: int
) -> _ModeAmpPhase:
    """Recover one pure-ROM mode's amplitude and residual phase."""

    try:
        cmode_evaluation = evaluation.modes[mode_index]
    except KeyError as exc:
        raise ValueError(
            f"SEOBNRv5HM ROM mode {_MODE_NAMES[mode_index]} was not reconstructed"
        ) from exc

    mode_m = _LM_MODES[mode_index][1]
    carrier_coefficients = _natural_cubic_coeff(
        evaluation.phase_frequency, evaluation.carrier_phase
    )
    carrier_frequency = cmode_evaluation.frequency / mode_m
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
    # Unlike v4HM, v5HM stores no additional per-mode phase shift here.
    phase_approximation = mode_m * carrier_phase + (1.0 - mode_m) * math.pi / 4.0
    return _ModeAmpPhase(
        amplitude_frequency=cmode_evaluation.frequency,
        amplitude=torch.abs(cmode_evaluation.cmode),
        phase_frequency=cmode_evaluation.frequency,
        phase=(
            _unwrap_phase(torch.angle(cmode_evaluation.cmode)) - phase_approximation
        ),
    )


def _taylorf2_mode_amp_phase(
    q: float,
    spin1z: float,
    spin2z: float,
    phasing,
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
    phase = (
        phase / velocity**5 * (emm / 2.0)
        + _TAYLORF2_PHASE_SHIFTS[mode_index]
        - math.pi / 4.0
    )

    eta = q / (1.0 + q) ** 2
    delta = (q - 1.0 + np.finfo(np.float64).eps) / (1.0 + q)
    symmetric_spin = 0.5 * (spin1z + spin2z)
    antisymmetric_spin = 0.5 * (spin1z - spin2z)
    if (ell, emm) == (2, 2):
        mode_factor = torch.ones_like(velocity)
    elif (ell, emm) == (2, 1):
        mode_factor = velocity * (
            delta / 3.0 - 0.5 * velocity * (antisymmetric_spin + delta * symmetric_spin)
        )
    elif (ell, emm) == (3, 3):
        mode_factor = velocity * 0.75 * math.sqrt(15.0 / 14.0) * delta
    elif (ell, emm) == (4, 4):
        mode_factor = velocity**2 * 8.0 * math.sqrt(35.0) / 63.0 * (1.0 - 3.0 * eta)
    elif (ell, emm) == (5, 5):
        mode_factor = (
            velocity**3 * 625.0 * math.sqrt(66.0) / 6336.0 * delta * (1.0 - 2.0 * eta)
        )
    elif (ell, emm) == (3, 2):
        mode_factor = (
            velocity**2
            * 9.0
            / 8.0
            * math.sqrt(5.0 / 7.0)
            * 8.0
            / 27.0
            * (-1.0 + 3.0 * eta)
        )
    else:  # (4, 3)
        mode_factor = (
            velocity**3
            * 8.0
            / 9.0
            * math.sqrt(10.0 / 7.0)
            * 81.0
            / 320.0
            * delta
            * (-1.0 + 2.0 * eta)
        )
    amplitude = (
        math.pi
        * math.sqrt(2.0 * eta / 3.0)
        * velocity ** (-3.5)
        * math.sqrt(2.0 / emm)
        * mode_factor
    )
    return amplitude, phase


def _hybridized_mode_data(
    evaluation: _SparseROMEvaluation,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    active_mode_indices: Tuple[int, ...],
    start_mf: float,
) -> Dict[int, _ModeAmpPhase]:
    """Attach a TaylorF2 inspiral to each requested physical ROM mode."""

    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
    q = mass1 / mass2
    pn_frequency = _inspiral_frequency_grid(
        start_mf,
        q,
        evaluation.phase_frequency.device,
        evaluation.phase_frequency.dtype,
    )
    phasing = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=7,
    )
    active_modes = tuple(dict.fromkeys(active_mode_indices))
    required_modes = (0,) + tuple(index for index in active_modes if index != 0)
    mode_data = {}
    delta_time_22 = None
    delta_phase_22 = None

    for mode_index in required_modes:
        rom_mode = _rom_mode_amp_phase(evaluation, mode_index)
        pn_amplitude, pn_phase = _taylorf2_mode_amp_phase(
            q,
            spin1z,
            spin2z,
            phasing,
            mode_index,
            pn_frequency,
        )
        mode_m = _LM_MODES[mode_index][1]
        window_start = _mode_minimum_mf(mode_index) * _PN_HYBRID_START_FACTOR
        window_end = _mode_minimum_mf(mode_index) * _PN_HYBRID_END_FACTOR

        if mode_index == 0:
            aligned_phase, delta_time_22, delta_phase_22 = _linear_phase_alignment(
                pn_frequency,
                pn_phase,
                rom_mode.phase_frequency,
                rom_mode.phase,
                window_start,
                window_end,
            )
        else:
            aligned_phase = _phase_alignment_from_22(
                pn_frequency,
                pn_phase,
                rom_mode.phase_frequency,
                rom_mode.phase,
                window_start,
                window_end,
                delta_time_22,
                delta_phase_22,
                mode_m,
            )
        phase_frequency, hybrid_phase = _hybridize_sparse_functions(
            pn_frequency,
            aligned_phase,
            rom_mode.phase_frequency,
            rom_mode.phase,
            window_start,
            window_end,
        )

        if mode_index not in active_modes:
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
            rom_mode.amplitude_frequency,
            rom_mode.amplitude,
            *_natural_cubic_coeff(rom_mode.amplitude_frequency, rom_mode.amplitude),
        )[0]
        scaled_pn_amplitude = (
            pn_amplitude * rom_amplitude_at_start / pn_amplitude_at_start
        )
        amplitude_frequency, hybrid_amplitude = _hybridize_sparse_functions(
            pn_frequency,
            scaled_pn_amplitude,
            rom_mode.amplitude_frequency,
            rom_mode.amplitude,
            window_start,
            window_end,
        )
        mode_data[mode_index] = _ModeAmpPhase(
            amplitude_frequency=amplitude_frequency,
            amplitude=hybrid_amplitude,
            phase_frequency=phase_frequency,
            phase=hybrid_phase,
        )
    return mode_data


def _seobnrv5hm_polarizations(inputs, eval_mf, start_mf: float):
    """Evaluate selected TaylorF2/ROM modes at geometric frequencies."""

    if bool(torch.any(eval_mf < _inspiral_minimum_mf(start_mf))):
        raise ValueError(
            "SEOBNRv5HM_ROM frequency lies below the TaylorF2 spline "
            "domain set by the starting frequency"
        )

    hp = torch.zeros(
        eval_mf.shape,
        device=inputs.device,
        dtype=inputs.complex_dtype,
    )
    hc = torch.zeros_like(hp)
    if not inputs.active_mode_indices:
        return hp, hc

    hybridized_modes = _hybridized_mode_data(
        inputs.evaluation,
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        inputs.active_mode_indices,
        start_mf,
    )
    observer_phi = math.pi / 2.0 - inputs.coa_phase

    for mode_index in inputs.active_mode_indices:
        ell, emm = _LM_MODES[mode_index]
        mode = hybridized_modes[mode_index]
        mf_max = (
            _CONST_FMAX[mode_index] * inputs.qnm_omega[(ell, emm)] / (2.0 * math.pi)
        )
        active = eval_mf <= mf_max
        if not bool(torch.any(active)):
            continue

        mode_frequencies = eval_mf[active]
        mode_amplitude = _spline_eval(
            mode_frequencies,
            mode.amplitude_frequency,
            mode.amplitude,
            *_natural_cubic_coeff(mode.amplitude_frequency, mode.amplitude),
        )
        mode_phase = _spline_eval(
            mode_frequencies,
            mode.phase_frequency,
            mode.phase,
            *_natural_cubic_coeff(mode.phase_frequency, mode.phase),
        )
        hlm = torch.complex(
            mode_amplitude * torch.cos(mode_phase),
            mode_amplitude * torch.sin(mode_phase),
        )
        time_shift = torch.exp((-2j * math.pi * 1000.0) * mode_frequencies).to(
            inputs.complex_dtype
        )
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
        factor_plus = 0.5 * (y_negative + parity * y_positive_conjugate)
        factor_cross = 0.5j * (y_negative - parity * y_positive_conjugate)
        hp[active] += factor_plus * hlm
        hc[active] += factor_cross * hlm

    amplitude_scale = (
        inputs.total_mass * inputs.total_mass_seconds * lal.MRSUN_SI / inputs.distance
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


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _seobnrv5hm_dtypes(state):
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
        raise TypeError(f"unsupported SEOBNRv5HM dtype {configured}")
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return device, real_dtype, complex_dtype


def _seobnrv5hm_inputs(p, *, sequence=False):
    """Validate scalar inputs and reconstruct the requested ROM modes."""

    if not _native_features_supported(p):
        raise ValueError(
            "SEOBNRv5HM_ROM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch SEOBNRv5HM_ROM requires TorchScheme")

    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    distance_mpc = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    reference_frequency = float(p.get("f_ref", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = 0.0 if sequence else float(p.get("long_asc_nodes", 0.0))
    active_mode_indices = _active_mode_indices(p.get("mode_array"))

    scalar_parameters = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": distance_mpc,
        "inclination": inclination,
        "coa_phase": coa_phase,
        "f_ref": reference_frequency,
        "long_asc_nodes": long_asc_nodes,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if abs(spin1z) > _MAX_ALIGNED_SPIN or abs(spin2z) > _MAX_ALIGNED_SPIN:
        raise ValueError("SEOBNRv5HM_ROM component spins must lie in [-0.998, 0.998]")
    if distance_mpc <= 0.0:
        raise ValueError("distance must be positive")
    if reference_frequency < 0.0:
        raise ValueError("f_ref must be non-negative")

    sign_odd = 1.0
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        sign_odd = -1.0
    mass_ratio = mass1 / mass2
    if mass_ratio > _MAX_MASS_RATIO:
        raise ValueError("SEOBNRv5HM_ROM requires a mass ratio no greater than 100")

    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    device, real_dtype, complex_dtype = _seobnrv5hm_dtypes(state)
    qnm_omega = {
        mode: seobnrv5_qnm_omega(mass1, mass2, spin1z, spin2z, ell=mode[0], emm=mode[1])
        for mode in _LM_MODES
    }
    mf_rom_max = _CONST_FMAX[4] * qnm_omega[_LM_MODES[4]] / (2.0 * math.pi)
    evaluation = None
    if active_mode_indices:
        roms = _load_rom(real_dtype, device, active_mode_indices)
        evaluation = _evaluate_sparse_rom(
            roms,
            mass1,
            mass2,
            spin1z,
            spin2z,
            active_mode_indices,
        )
    return _SEOBNRv5HMInputs(
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
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        qnm_omega=qnm_omega,
        mf_rom_max=mf_rom_max,
        evaluation=evaluation,
    )


def seobnrv5hm_fd_torch(**p):
    """Generate regular-grid ``SEOBNRv5HM_ROM`` polarizations with Torch."""

    if not seobnrv5hm_native_supported(p):
        raise ValueError("SEOBNRv5HM_ROM parameters require an unsupported feature")
    inputs = _seobnrv5hm_inputs(p)
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
    hp = torch.zeros(npts, device=inputs.device, dtype=inputs.complex_dtype)
    hc = torch.zeros_like(hp)
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = min(math.ceil(final_frequency / delta_f), npts)
    bin_indices = torch.arange(first_bin, stop_bin, device=inputs.device)
    eval_mf = (
        bin_indices.to(dtype=inputs.real_dtype) * delta_f * inputs.total_mass_seconds
    )
    plus, cross = _seobnrv5hm_polarizations(
        inputs,
        eval_mf,
        f_lower * inputs.total_mass_seconds,
    )
    hp[first_bin:stop_bin] = plus
    hc[first_bin:stop_bin] = cross

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False),
        FrequencySeries(TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False),
    )


def _seobnrv5hm_sequence_frequencies(sample_points, inputs):
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
        raise ValueError("SEOBNRv5HM_ROM sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SEOBNRv5HM_ROM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SEOBNRv5HM_ROM sample_points must be positive")
    return frequencies


def seobnrv5hm_fd_sequence_torch(**p):
    """Evaluate ``SEOBNRv5HM_ROM`` at arbitrary frequencies with Torch."""

    if not seobnrv5hm_sequence_native_supported(p):
        raise ValueError(
            "SEOBNRv5HM_ROM sequence parameters require an unsupported feature"
        )
    inputs = _seobnrv5hm_inputs(p, sequence=True)
    frequencies = _seobnrv5hm_sequence_frequencies(p["sample_points"], inputs)
    plus, cross = _seobnrv5hm_polarizations(
        inputs,
        frequencies * inputs.total_mass_seconds,
        float(frequencies[0]) * inputs.total_mass_seconds,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "seobnrv5hm_fd_sequence_torch",
    "seobnrv5hm_fd_torch",
    "seobnrv5hm_native_supported",
    "seobnrv5hm_sequence_native_supported",
]
