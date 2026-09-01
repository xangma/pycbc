# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native waveform implementation for ``SEOBNRv4T_surrogate``.

The public surrogate stores Gaussian-process fits to its logarithmic amplitude
and phase corrections. HDF5 discovery and validation remain on the
host; cached model tensors and all regression evaluation live on the requested
Torch device. Public waveform dispatch is registered separately from this
implementation so the numerical layer can also be validated directly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch

from pycbc import lal_compat as lal
import pycbc.scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)
from pycbc.waveform.taylorf2_torch import (
    _f2_6pn_qm2_s,
    _f2_6pn_s1s2_o,
    _f2_6pn_self2_s,
    taylorf2_aligned_phasing,
)


_ROM_FILENAMES = (
    "SEOBNRv4T_surrogate_v2.0.0.hdf5",
    "SEOBNRv4T_surrogate_v1.0.0.hdf5",
)
_ROM_FILENAME = _ROM_FILENAMES[0]
_ROM_VERSIONS = {
    _ROM_FILENAMES[0]: (2, 0, 0),
    _ROM_FILENAMES[1]: (1, 0, 0),
}
_TRAINING_COUNT = 960
_PARAMETER_DIMENSION = 5
_AMPLITUDE_NODE_COUNT = 40
_PHASE_REGRESSION_COUNT = 39
_TF2_NODE_COUNT = 1000
_APPROXIMANT = "SEOBNRv4T_surrogate"
_ISCO_MF = 1.0 / (6.0**1.5 * math.pi)

_DEFAULT_ORDER_KEYS = (
    "phase_order",
    "spin_order",
    "tidal_order",
    "amplitude_order",
    "eccentricity_order",
)
_ZERO_ONLY_KEYS = (
    "spin1x",
    "spin1y",
    "spin2x",
    "spin2y",
    "eccentricity",
    "mean_per_ano",
    "frame_axis",
    "modes_choice",
    "side_bands",
    "dquad_mon1",
    "dquad_mon2",
    "lambda_octu1",
    "lambda_octu2",
    "quadfmode1",
    "quadfmode2",
    "octufmode1",
    "octufmode2",
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

_DATASET_SHAPES = {
    "hyp_amp": (_AMPLITUDE_NODE_COUNT, _PARAMETER_DIMENSION + 2),
    "hyp_phi": (_PHASE_REGRESSION_COUNT, _PARAMETER_DIMENSION + 2),
    "kinv_dot_y_amp": (_AMPLITUDE_NODE_COUNT, _TRAINING_COUNT),
    "kinv_dot_y_phi": (_PHASE_REGRESSION_COUNT, _TRAINING_COUNT),
    "x_train": (_TRAINING_COUNT, _PARAMETER_DIMENSION),
    "spline_nodes_amp": (_AMPLITUDE_NODE_COUNT,),
    "spline_nodes_phase": (_PHASE_REGRESSION_COUNT,),
    "TF2_Mf_amp_cubic": (_TF2_NODE_COUNT,),
    "TF2_Mf_phi_cubic": (_TF2_NODE_COUNT,),
    "TF2_Mf_amp_linear": (_TF2_NODE_COUNT,),
    "TF2_Mf_phi_linear": (_TF2_NODE_COUNT,),
    "q_bounds": (2,),
    "chi1_bounds": (2,),
    "chi2_bounds": (2,),
    "lambda1_bounds": (2,),
    "lambda2_bounds": (2,),
}


@dataclass(frozen=True)
class _SurrogateMetadata:
    path: Path
    mass_ratio_bounds: tuple[float, float]
    chi1_bounds: tuple[float, float]
    chi2_bounds: tuple[float, float]
    lambda1_bounds: tuple[float, float]
    lambda2_bounds: tuple[float, float]
    amplitude_frequency_bounds: tuple[float, float]
    phase_frequency_bounds: tuple[float, float]


@dataclass(frozen=True)
class _HostSurrogateData:
    hyp_amp: np.ndarray
    hyp_phi: np.ndarray
    kinv_dot_y_amp: np.ndarray
    kinv_dot_y_phi: np.ndarray
    x_train: np.ndarray
    amplitude_nodes: np.ndarray
    phase_nodes: np.ndarray
    tf2_amplitude_nodes_cubic: np.ndarray
    tf2_phase_nodes_cubic: np.ndarray
    tf2_amplitude_nodes_linear: np.ndarray
    tf2_phase_nodes_linear: np.ndarray


@dataclass(frozen=True)
class _SurrogateData:
    metadata: _SurrogateMetadata
    hyp_amp: torch.Tensor
    hyp_phi: torch.Tensor
    kinv_dot_y_amp: torch.Tensor
    kinv_dot_y_phi: torch.Tensor
    x_train: torch.Tensor
    amplitude_nodes: torch.Tensor
    phase_nodes: torch.Tensor
    tf2_amplitude_nodes_cubic: torch.Tensor
    tf2_phase_nodes_cubic: torch.Tensor
    tf2_amplitude_nodes_linear: torch.Tensor
    tf2_phase_nodes_linear: torch.Tensor


@dataclass(frozen=True)
class _IntrinsicParameters:
    mass1: float
    mass2: float
    total_mass: float
    symmetric_mass_ratio: float
    mass_ratio: float
    chi1: float
    chi2: float
    lambda1: float
    lambda2: float


@dataclass(frozen=True)
class _CubicSpline:
    knots: torch.Tensor
    values: torch.Tensor
    linear: torch.Tensor
    quadratic: torch.Tensor
    cubic: torch.Tensor


@dataclass(frozen=True)
class _WaveformModel:
    amplitude: _CubicSpline
    phase: _CubicSpline
    minimum_frequency: float
    correction_minimum_frequency: float
    correction_maximum_frequency: float
    cutoff_frequency: float


@dataclass(frozen=True)
class _WaveformInputs:
    parameters: _IntrinsicParameters
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    reference_frequency: float
    total_mass_seconds: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    model: _WaveformModel


def _find_rom_file() -> Path:
    """Find the newest available public surrogate data file."""

    search_dirs = [Path(__file__).resolve().parent]
    search_dirs.extend(
        Path(base)
        for base in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if base
    )
    for filename in _ROM_FILENAMES:
        for directory in search_dirs:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"none of {_ROM_FILENAMES!r} found; place one next to this module or "
        "on $LAL_DATA_PATH"
    )


def _require_dataset(rom, name: str, shape: tuple[int, ...]) -> h5py.Dataset:
    try:
        dataset = rom[name]
    except KeyError as exc:
        raise ValueError(f"SEOBNRv4T surrogate is missing dataset /{name}") from exc
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"SEOBNRv4T surrogate object /{name} is not a dataset")
    if dataset.shape != shape:
        raise ValueError(
            f"SEOBNRv4T surrogate dataset /{name} has shape {dataset.shape}; "
            f"expected {shape}"
        )
    if not np.issubdtype(dataset.dtype, np.floating):
        raise ValueError(
            f"SEOBNRv4T surrogate dataset /{name} has non-floating dtype "
            f"{dataset.dtype}"
        )
    return dataset


def _read_finite_dataset(rom, name: str) -> np.ndarray:
    values = _require_dataset(rom, name, _DATASET_SHAPES[name])[:]
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"SEOBNRv4T surrogate dataset /{name} contains non-finite values"
        )
    return values


def _ordered_vector(rom, name: str) -> np.ndarray:
    values = _read_finite_dataset(rom, name)
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(
            f"SEOBNRv4T surrogate dataset /{name} must be strictly increasing"
        )
    return values


def _bounds(rom, name: str) -> tuple[float, float]:
    values = _ordered_vector(rom, name)
    return float(values[0]), float(values[1])


def _validate_rom_file(path: Path) -> _SurrogateMetadata:
    """Validate the identity and complete layout of public surrogate data."""

    path = Path(path)
    if path.name not in _ROM_VERSIONS:
        raise ValueError(
            f"SEOBNRv4T surrogate has basename {path.name!r}; "
            f"expected one of {_ROM_FILENAMES!r}"
        )
    expected_version = _ROM_VERSIONS[path.name]
    with h5py.File(path, "r") as rom:
        version = tuple(
            int(rom.attrs.get(f"version_{part}", -1))
            for part in ("major", "minor", "micro")
        )
        if version != expected_version:
            raise ValueError(
                f"SEOBNRv4T surrogate has version {version}; "
                f"expected {expected_version}"
            )
        if expected_version >= (2, 0, 0):
            canonical_basename = rom.attrs.get("CANONICAL_FILE_BASENAME", "")
            if isinstance(canonical_basename, bytes):
                canonical_basename = canonical_basename.decode()
            if canonical_basename != path.name:
                raise ValueError(
                    "SEOBNRv4T surrogate has canonical basename "
                    f"{canonical_basename!r}; expected {path.name!r}"
                )

        arrays = {name: _read_finite_dataset(rom, name) for name in _DATASET_SHAPES}
        for name in (
            "spline_nodes_amp",
            "spline_nodes_phase",
            "TF2_Mf_amp_cubic",
            "TF2_Mf_phi_cubic",
            "TF2_Mf_amp_linear",
            "TF2_Mf_phi_linear",
            "q_bounds",
            "chi1_bounds",
            "chi2_bounds",
            "lambda1_bounds",
            "lambda2_bounds",
        ):
            if not np.all(np.diff(arrays[name]) > 0.0):
                raise ValueError(
                    f"SEOBNRv4T surrogate dataset /{name} must be strictly increasing"
                )

        amplitude_nodes = arrays["spline_nodes_amp"]
        phase_nodes = arrays["spline_nodes_phase"]
        if not np.array_equal(phase_nodes, amplitude_nodes[1:]):
            raise ValueError(
                "SEOBNRv4T surrogate phase nodes must equal amplitude nodes "
                "after the shared leading node"
            )
        if not np.all(arrays["hyp_amp"] > 0.0) or not np.all(arrays["hyp_phi"] > 0.0):
            raise ValueError("SEOBNRv4T surrogate GPR hyperparameters must be positive")

        for order in ("cubic", "linear"):
            tf2_amplitude = arrays[f"TF2_Mf_amp_{order}"]
            tf2_phase = arrays[f"TF2_Mf_phi_{order}"]
            lower = max(tf2_amplitude[0], tf2_phase[0])
            upper = min(tf2_amplitude[-1], tf2_phase[-1])
            if lower >= amplitude_nodes[0] or upper < amplitude_nodes[-1]:
                raise ValueError(
                    f"SEOBNRv4T surrogate {order} TaylorF2 grids do not "
                    "cover the correction grid"
                )

        inverse_mass_ratio_bounds = arrays["q_bounds"]
        if inverse_mass_ratio_bounds[0] <= 0.0:
            raise ValueError(
                "SEOBNRv4T surrogate inverse-mass-ratio bounds must be positive"
            )
        mass_ratio_bounds = (
            1.0 / float(inverse_mass_ratio_bounds[1]),
            1.0 / float(inverse_mass_ratio_bounds[0]),
        )
        return _SurrogateMetadata(
            path=path,
            mass_ratio_bounds=mass_ratio_bounds,
            chi1_bounds=_bounds(rom, "chi1_bounds"),
            chi2_bounds=_bounds(rom, "chi2_bounds"),
            lambda1_bounds=_bounds(rom, "lambda1_bounds"),
            lambda2_bounds=_bounds(rom, "lambda2_bounds"),
            amplitude_frequency_bounds=(
                float(amplitude_nodes[0]),
                float(amplitude_nodes[-1]),
            ),
            phase_frequency_bounds=(
                float(amplitude_nodes[0]),
                float(phase_nodes[-1]),
            ),
        )


@lru_cache(None)
def _surrogate_metadata() -> _SurrogateMetadata:
    return _validate_rom_file(_find_rom_file())


def _numpy_dtype(dtype: torch.dtype):
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    raise TypeError(
        "SEOBNRv4T surrogate tensors require torch.float32 or torch.float64, "
        f"not {dtype}"
    )


@lru_cache(None)
def _load_host_data(dtype: torch.dtype) -> _HostSurrogateData:
    """Load and cache the validated model in one host precision."""

    metadata = _surrogate_metadata()
    numpy_dtype = _numpy_dtype(dtype)

    def read(rom, name):
        return np.asarray(rom[name][:], dtype=numpy_dtype)

    with h5py.File(metadata.path, "r") as rom:
        amplitude_nodes = read(rom, "spline_nodes_amp")
        phase_nodes = np.concatenate(
            (amplitude_nodes[:1], read(rom, "spline_nodes_phase"))
        )
        return _HostSurrogateData(
            hyp_amp=read(rom, "hyp_amp"),
            hyp_phi=read(rom, "hyp_phi"),
            kinv_dot_y_amp=read(rom, "kinv_dot_y_amp"),
            kinv_dot_y_phi=read(rom, "kinv_dot_y_phi"),
            x_train=read(rom, "x_train"),
            amplitude_nodes=amplitude_nodes,
            phase_nodes=phase_nodes,
            tf2_amplitude_nodes_cubic=read(rom, "TF2_Mf_amp_cubic"),
            tf2_phase_nodes_cubic=read(rom, "TF2_Mf_phi_cubic"),
            tf2_amplitude_nodes_linear=read(rom, "TF2_Mf_amp_linear"),
            tf2_phase_nodes_linear=read(rom, "TF2_Mf_phi_linear"),
        )


@lru_cache(None)
def _load_surrogate_data(dtype: torch.dtype, device: torch.device) -> _SurrogateData:
    """Return the model cached as tensors on ``device``."""

    host = _load_host_data(dtype)

    def convert(values):
        return torch.as_tensor(values, dtype=dtype, device=device)

    return _SurrogateData(
        metadata=_surrogate_metadata(),
        hyp_amp=convert(host.hyp_amp),
        hyp_phi=convert(host.hyp_phi),
        kinv_dot_y_amp=convert(host.kinv_dot_y_amp),
        kinv_dot_y_phi=convert(host.kinv_dot_y_phi),
        x_train=convert(host.x_train),
        amplitude_nodes=convert(host.amplitude_nodes),
        phase_nodes=convert(host.phase_nodes),
        tf2_amplitude_nodes_cubic=convert(host.tf2_amplitude_nodes_cubic),
        tf2_phase_nodes_cubic=convert(host.tf2_phase_nodes_cubic),
        tf2_amplitude_nodes_linear=convert(host.tf2_amplitude_nodes_linear),
        tf2_phase_nodes_linear=convert(host.tf2_phase_nodes_linear),
    )


def _clear_surrogate_cache() -> None:
    """Release cached host arrays, device tensors, and file metadata."""

    _load_surrogate_data.cache_clear()
    _load_host_data.cache_clear()
    _surrogate_metadata.cache_clear()


def _canonical_intrinsic_parameters(
    mass1,
    mass2,
    chi1,
    chi2,
    lambda1,
    lambda2,
) -> _IntrinsicParameters:
    """Put the heavier body first, matching the LAL surrogate entry point."""

    values = tuple(
        float(value) for value in (mass1, mass2, chi1, chi2, lambda1, lambda2)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SEOBNRv4T surrogate intrinsic parameters must be finite")
    mass1, mass2, chi1, chi2, lambda1, lambda2 = values
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("SEOBNRv4T surrogate component masses must be positive")
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        chi1, chi2 = chi2, chi1
        lambda1, lambda2 = lambda2, lambda1
    total_mass = mass1 + mass2
    symmetric_mass_ratio = mass1 * mass2 / (total_mass * total_mass)
    mass_ratio = (
        1.0 + math.sqrt(1.0 - 4.0 * symmetric_mass_ratio) - 2.0 * symmetric_mass_ratio
    ) / (2.0 * symmetric_mass_ratio)
    return _IntrinsicParameters(
        mass1=mass1,
        mass2=mass2,
        total_mass=total_mass,
        symmetric_mass_ratio=symmetric_mass_ratio,
        mass_ratio=mass_ratio,
        chi1=chi1,
        chi2=chi2,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def _validate_intrinsic_parameters(
    parameters: _IntrinsicParameters, metadata: _SurrogateMetadata
) -> None:
    checks = (
        ("mass ratio", parameters.mass_ratio, metadata.mass_ratio_bounds),
        ("chi1", parameters.chi1, metadata.chi1_bounds),
        ("chi2", parameters.chi2, metadata.chi2_bounds),
        ("lambda1", parameters.lambda1, metadata.lambda1_bounds),
        ("lambda2", parameters.lambda2, metadata.lambda2_bounds),
    )
    for name, value, (lower, upper) in checks:
        if value < lower or value > upper:
            raise ValueError(
                f"SEOBNRv4T surrogate {name} {value} is outside [{lower}, {upper}]"
            )


def _surrogate_coordinate(
    parameters: _IntrinsicParameters,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return LAL's ``(1/q, chi1, chi2, xi1, xi2)`` coordinate."""

    coordinate = torch.tensor(
        (
            1.0 / parameters.mass_ratio,
            parameters.chi1,
            parameters.chi2,
            parameters.lambda1 / 100.0 + 1.0,
            parameters.lambda2 / 100.0 + 1.0,
        ),
        dtype=dtype,
        device=device,
    )
    coordinate[3:] = torch.log10(coordinate[3:])
    return coordinate


def _gpr_predict(
    coordinate: torch.Tensor,
    hyperparameters: torch.Tensor,
    training_points: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a batch of Matérn-5/2 Gaussian-process regressors."""

    if coordinate.shape != (_PARAMETER_DIMENSION,):
        raise ValueError("surrogate coordinate must have shape (5,)")
    output_count = hyperparameters.shape[0]
    if hyperparameters.shape != (output_count, _PARAMETER_DIMENSION + 2):
        raise ValueError("GPR hyperparameters must have shape (outputs, 7)")
    if training_points.shape != (_TRAINING_COUNT, _PARAMETER_DIMENSION):
        raise ValueError("GPR training points must have shape (960, 5)")
    if weights.shape != (output_count, _TRAINING_COUNT):
        raise ValueError("GPR weights must have shape (outputs, 960)")
    tensors = (coordinate, hyperparameters, training_points, weights)
    if any(
        value.dtype != coordinate.dtype or value.device != coordinate.device
        for value in tensors[1:]
    ):
        raise ValueError("GPR tensors must share dtype and device")

    signal_scale = hyperparameters[:, 0]
    length_scales = hyperparameters[:, 1:-1]
    noise_scale = hyperparameters[:, -1]
    displacement = (
        coordinate[None, None, :] - training_points[None, :, :]
    ) / length_scales[:, None, :]
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    sqrt_five_radius = math.sqrt(5.0) * radius
    covariance = (
        signal_scale[:, None] ** 2
        * (1.0 + sqrt_five_radius + 5.0 * radius**2 / 3.0)
        * torch.exp(-sqrt_five_radius)
    )

    # LAL includes the fitted noise nugget only when the evaluation coordinate
    # exactly equals a training coordinate.
    diagonal = torch.all(coordinate == training_points, dim=-1)
    covariance = covariance + noise_scale[:, None] ** 2 * diagonal
    return torch.sum(covariance * weights, dim=1)


def _evaluate_corrections(
    data: _SurrogateData,
    parameters: _IntrinsicParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-amplitude and phase corrections on surrogate nodes."""

    _validate_intrinsic_parameters(parameters, data.metadata)
    coordinate = _surrogate_coordinate(
        parameters,
        dtype=data.x_train.dtype,
        device=data.x_train.device,
    )
    amplitude = _gpr_predict(
        coordinate,
        data.hyp_amp,
        data.x_train,
        data.kinv_dot_y_amp,
    )
    phase_regression = _gpr_predict(
        coordinate,
        data.hyp_phi,
        data.x_train,
        data.kinv_dot_y_phi,
    )
    phase = torch.cat((torch.zeros_like(phase_regression[:1]), phase_regression))
    return amplitude, phase


def _universal_quadrupole(lambda_tidal) -> float:
    """Return LAL's spin-quadrupole universal relation."""

    lambda_tidal = float(lambda_tidal)
    if not math.isfinite(lambda_tidal) or lambda_tidal < 0.0:
        raise ValueError("tidal deformability must be finite and non-negative")
    if lambda_tidal < 1.0:
        return 1.0 + lambda_tidal * (
            0.427688866723244
            + lambda_tidal * (-0.324336526985068 + lambda_tidal * 0.1107439432180572)
        )
    log_lambda = math.log(lambda_tidal)
    log_quadrupole = (
        0.1940
        + 0.09163 * log_lambda
        + 0.04812 * log_lambda * log_lambda
        - 4.283e-3 * log_lambda * log_lambda * log_lambda
        + 1.245e-4 * log_lambda * log_lambda * log_lambda * log_lambda
    )
    return math.exp(log_quadrupole)


def _make_cubic_spline(knots: torch.Tensor, values: torch.Tensor) -> _CubicSpline:
    linear, quadratic, cubic = _natural_cubic_coeff(knots, values)
    return _CubicSpline(knots, values, linear, quadratic, cubic)


def _evaluate_spline(spline: _CubicSpline, points: torch.Tensor) -> torch.Tensor:
    return _spline_eval(
        points,
        spline.knots,
        spline.values,
        spline.linear,
        spline.quadratic,
        spline.cubic,
    )


def _taylorf2_phase(
    parameters: _IntrinsicParameters,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the surrogate's modified TaylorF2 carrier phase."""

    quadrupole1 = _universal_quadrupole(parameters.lambda1)
    quadrupole2 = _universal_quadrupole(parameters.lambda2)
    mass_fraction1 = parameters.mass_ratio / (1.0 + parameters.mass_ratio)
    mass_fraction2 = 1.0 / (1.0 + parameters.mass_ratio)
    phasing = taylorf2_aligned_phasing(
        parameters.total_mass * mass_fraction1,
        parameters.total_mass * mass_fraction2,
        parameters.chi1,
        parameters.chi2,
        spin_order=7,
        tidal_order=12,
        qm_def1=quadrupole1 - 1.0,
        qm_def2=quadrupole2 - 1.0,
        lambda1=parameters.lambda1,
        lambda2=parameters.lambda2,
    )

    spin_spin_3pn = (
        _f2_6pn_s1s2_o(parameters.symmetric_mass_ratio)
        * parameters.chi1
        * parameters.chi2
        + (
            _f2_6pn_qm2_s(mass_fraction1) * quadrupole1
            + _f2_6pn_self2_s(mass_fraction1)
        )
        * parameters.chi1**2
        + (
            _f2_6pn_qm2_s(mass_fraction2) * quadrupole2
            + _f2_6pn_self2_s(mass_fraction2)
        )
        * parameters.chi2**2
    )
    phasing.v[6] -= spin_spin_3pn * phasing.v[0]

    coefficients = torch.as_tensor(
        phasing.v,
        dtype=frequencies.dtype,
        device=frequencies.device,
    )
    log_coefficients = torch.as_tensor(
        phasing.vlogv,
        dtype=frequencies.dtype,
        device=frequencies.device,
    )
    velocity = torch.pow(math.pi * frequencies, 1.0 / 3.0)
    log_velocity = torch.log(velocity)
    velocity2 = velocity * velocity
    velocity3 = velocity * velocity2
    velocity4 = velocity * velocity3
    velocity5 = velocity * velocity4
    velocity6 = velocity * velocity5
    velocity7 = velocity * velocity6
    velocity10 = velocity3 * velocity7
    velocity12 = velocity2 * velocity10

    # Preserve the operation order in LAL's surrogate implementation: its
    # sparse carrier omits unused PN slots and adds the tidal terms last.
    phase = coefficients[7] * velocity7
    phase += (coefficients[6] + log_coefficients[6] * log_velocity) * velocity6
    phase += (coefficients[5] + log_coefficients[5] * log_velocity) * velocity5
    phase += coefficients[4] * velocity4
    phase += coefficients[3] * velocity3
    phase += coefficients[2] * velocity2
    phase += coefficients[1] * velocity
    phase += coefficients[0]
    phase += coefficients[12] * velocity12
    phase += coefficients[10] * velocity10
    return -phase / velocity5


def _taylorf2_amplitude(
    symmetric_mass_ratio: float,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the dimensionless 1PN TaylorF2 carrier amplitude."""

    velocity = torch.pow(math.pi * frequencies, 1.0 / 3.0)
    x = velocity**2
    leading = math.sqrt(5.0 * math.pi / 24.0 * symmetric_mass_ratio)
    correction = -323.0 / 224.0 + 451.0 / 168.0 * symmetric_mass_ratio
    return -leading * torch.pow(x, -7.0 / 4.0) * (1.0 + correction * x)


def _build_waveform_model(
    data: _SurrogateData,
    parameters: _IntrinsicParameters,
) -> _WaveformModel:
    """Construct the corrected TaylorF2 amplitude and phase splines."""

    amplitude_correction, phase_correction = _evaluate_corrections(data, parameters)
    amplitude_correction_spline = _make_cubic_spline(
        data.amplitude_nodes,
        amplitude_correction,
    )
    phase_correction_spline = _make_cubic_spline(
        data.phase_nodes,
        phase_correction,
    )
    correction_minimum = max(
        data.metadata.amplitude_frequency_bounds[0],
        data.metadata.phase_frequency_bounds[0],
    )
    correction_maximum = min(
        data.metadata.amplitude_frequency_bounds[1],
        data.metadata.phase_frequency_bounds[1],
    )

    amplitude_frequencies = data.tf2_amplitude_nodes_cubic
    amplitude = _taylorf2_amplitude(
        parameters.symmetric_mass_ratio,
        amplitude_frequencies,
    )
    amplitude_mask = (amplitude_frequencies >= correction_minimum) & (
        amplitude_frequencies <= correction_maximum
    )
    amplitude = amplitude.clone()
    amplitude[amplitude_mask] *= torch.exp(
        _evaluate_spline(
            amplitude_correction_spline,
            amplitude_frequencies[amplitude_mask],
        )
    )

    phase_frequencies = data.tf2_phase_nodes_cubic
    phase = _taylorf2_phase(parameters, phase_frequencies)
    phase_mask = (phase_frequencies >= correction_minimum) & (
        phase_frequencies <= correction_maximum
    )
    phase = phase.clone()
    phase[phase_mask] += _evaluate_spline(
        phase_correction_spline,
        phase_frequencies[phase_mask],
    )

    return _WaveformModel(
        amplitude=_make_cubic_spline(amplitude_frequencies, amplitude),
        phase=_make_cubic_spline(phase_frequencies, phase),
        minimum_frequency=max(
            float(amplitude_frequencies[0]),
            float(phase_frequencies[0]),
        ),
        correction_minimum_frequency=correction_minimum,
        correction_maximum_frequency=correction_maximum,
        cutoff_frequency=data.metadata.phase_frequency_bounds[1],
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


def _native_features_supported(parameters) -> bool:
    if parameters.get("approximant", _APPROXIMANT) != _APPROXIMANT:
        return False
    if any(
        not _is_default_order(parameters.get(name, -1)) for name in _DEFAULT_ORDER_KEYS
    ):
        return False
    if any(_is_nonzero(parameters.get(name, 0.0)) for name in _ZERO_ONLY_KEYS):
        return False
    if parameters.get("mode_array") is not None:
        return False
    if parameters.get("numrel_data", ""):
        return False
    return True


def _native_device_supported() -> bool:
    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme) and state.torch_device.type == "mps"
    )


def seobnrv4t_surrogate_native_supported(parameters) -> bool:
    """Return whether regular-grid generation is covered by this port."""

    return _native_features_supported(parameters) and _native_device_supported()


def seobnrv4t_surrogate_sequence_native_supported(parameters) -> bool:
    """Return whether arbitrary-frequency generation is covered by this port."""

    return _native_features_supported(parameters) and _native_device_supported()


def _waveform_dtypes(state):
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
        raise TypeError(f"unsupported SEOBNRv4T surrogate dtype {configured}")
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return device, real_dtype, complex_dtype


def _waveform_inputs(parameters, *, sequence: bool) -> _WaveformInputs:
    if not _native_features_supported(parameters):
        raise ValueError(
            "SEOBNRv4T surrogate parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native SEOBNRv4T surrogate requires TorchScheme")

    intrinsic = _canonical_intrinsic_parameters(
        parameters["mass1"],
        parameters["mass2"],
        parameters.get("spin1z", 0.0),
        parameters.get("spin2z", 0.0),
        parameters.get("lambda1") or 0.0,
        parameters.get("lambda2") or 0.0,
    )
    distance_mpc = float(parameters.get("distance", 1.0))
    inclination = float(parameters.get("inclination", 0.0))
    coa_phase = float(parameters.get("coa_phase", 0.0))
    reference_frequency = float(parameters.get("f_ref", 0.0))
    long_asc_nodes = 0.0 if sequence else float(parameters.get("long_asc_nodes", 0.0))
    scalars = {
        "distance": distance_mpc,
        "inclination": inclination,
        "coa_phase": coa_phase,
        "f_ref": reference_frequency,
        "long_asc_nodes": long_asc_nodes,
    }
    for name, value in scalars.items():
        if not math.isfinite(value):
            raise ValueError(f"SEOBNRv4T surrogate {name} must be finite")
    if distance_mpc <= 0.0:
        raise ValueError("SEOBNRv4T surrogate distance must be positive")
    if reference_frequency < 0.0:
        raise ValueError("SEOBNRv4T surrogate f_ref must be non-negative")

    device, real_dtype, complex_dtype = _waveform_dtypes(state)
    data = _load_surrogate_data(real_dtype, device)
    model = _build_waveform_model(data, intrinsic)
    return _WaveformInputs(
        parameters=intrinsic,
        distance=distance_mpc * 1.0e6 * lal.PC_SI,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        reference_frequency=reference_frequency,
        total_mass_seconds=intrinsic.total_mass * lal.MTSUN_SI,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        model=model,
    )


def _resolve_frequency_bounds(
    inputs: _WaveformInputs,
    lower_frequency: float,
    upper_frequency: float,
) -> tuple[float, float]:
    """Return the clipped geometric upper and reference frequencies."""

    if not math.isfinite(lower_frequency) or lower_frequency <= 0.0:
        raise ValueError("SEOBNRv4T surrogate f_lower must be finite and positive")
    if not math.isfinite(upper_frequency) or upper_frequency < 0.0:
        raise ValueError("SEOBNRv4T surrogate f_final must be finite and non-negative")

    lower_mf = lower_frequency * inputs.total_mass_seconds
    upper_mf = upper_frequency * inputs.total_mass_seconds
    model = inputs.model
    if lower_mf < model.minimum_frequency:
        raise ValueError(
            "SEOBNRv4T surrogate starting frequency is below the TaylorF2 extension"
        )
    if upper_mf == 0.0 or upper_mf > model.correction_maximum_frequency:
        upper_mf = model.correction_maximum_frequency
    elif upper_mf < model.correction_minimum_frequency:
        raise ValueError(
            "SEOBNRv4T surrogate ending frequency is below the correction domain"
        )
    if upper_mf <= lower_mf:
        raise ValueError(
            "SEOBNRv4T surrogate ending frequency must exceed its starting frequency"
        )

    reference_mf = (
        inputs.reference_frequency or lower_frequency
    ) * inputs.total_mass_seconds
    reference_mf = min(reference_mf, model.correction_maximum_frequency)
    reference_mf = max(reference_mf, model.minimum_frequency)
    return upper_mf, reference_mf


def _evaluate_polarizations(
    inputs: _WaveformInputs,
    geometric_frequencies: torch.Tensor,
    reference_mf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate both polarizations at dimensionless frequencies."""

    if geometric_frequencies.ndim != 1:
        raise ValueError("SEOBNRv4T surrogate frequencies must be a vector")
    if bool(torch.any(geometric_frequencies < inputs.model.minimum_frequency)):
        raise ValueError(
            "SEOBNRv4T surrogate frequency is below the TaylorF2 extension"
        )

    plus = torch.zeros(
        geometric_frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    active = geometric_frequencies <= inputs.model.cutoff_frequency
    if bool(torch.any(active)):
        frequencies = geometric_frequencies[active]
        amplitude = _evaluate_spline(inputs.model.amplitude, frequencies)
        phase_reference = _evaluate_spline(
            inputs.model.phase,
            torch.as_tensor(
                reference_mf,
                dtype=inputs.real_dtype,
                device=inputs.device,
            ),
        )
        phase_change = phase_reference - 2.0 * inputs.coa_phase
        phase = _evaluate_spline(inputs.model.phase, frequencies) - phase_change

        amplitude_scale = (
            inputs.parameters.total_mass
            * inputs.total_mass_seconds
            * lal.MRSUN_SI
            / inputs.distance
        )
        strain_amplitude = amplitude_scale * amplitude
        strain = torch.complex(
            strain_amplitude * torch.cos(phase),
            strain_amplitude * torch.sin(phase),
        ).to(inputs.complex_dtype)

        time_correction = _spline_derivative(
            torch.as_tensor(
                _ISCO_MF,
                dtype=inputs.real_dtype,
                device=inputs.device,
            ),
            inputs.model.phase.knots,
            inputs.model.phase.linear,
            inputs.model.phase.quadratic,
            inputs.model.phase.cubic,
        ) / (2.0 * math.pi)
        time_phase = -2.0 * math.pi * (frequencies - reference_mf) * time_correction
        time_factor = torch.complex(torch.cos(time_phase), torch.sin(time_phase)).to(
            inputs.complex_dtype
        )
        strain *= time_factor

        cos_inclination = math.cos(inputs.inclination)
        plus[active] = 0.5 * (1.0 + cos_inclination**2) * strain
        cross[active] = complex(0.0, -cos_inclination) * strain

    if inputs.long_asc_nodes:
        cosine = math.cos(2.0 * inputs.long_asc_nodes)
        sine = math.sin(2.0 * inputs.long_asc_nodes)
        rotated_plus = cosine * plus + sine * cross
        rotated_cross = cosine * cross - sine * plus
        plus, cross = rotated_plus, rotated_cross
    return plus, cross


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def seobnrv4t_surrogate_fd_torch(**parameters):
    """Generate regular-grid ``SEOBNRv4T_surrogate`` with Torch."""

    inputs = _waveform_inputs(parameters, sequence=False)
    delta_f = float(parameters["delta_f"])
    f_lower = float(parameters["f_lower"])
    f_final = float(parameters.get("f_final", 0.0))
    if not math.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("SEOBNRv4T surrogate delta_f must be finite and positive")
    generation_upper_mf, reference_mf = _resolve_frequency_bounds(
        inputs,
        f_lower,
        f_final,
    )
    generation_upper = generation_upper_mf / inputs.total_mass_seconds
    allocation_upper = f_final if f_final > generation_upper else generation_upper
    length = _next_power_of_two(int(allocation_upper / delta_f)) + 1
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = math.ceil(generation_upper / delta_f)
    if stop_bin <= first_bin:
        raise ValueError("SEOBNRv4T surrogate range contains no sampled bins")

    plus = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    bins = torch.arange(first_bin, stop_bin, device=inputs.device)
    geometric_frequencies = (
        bins.to(dtype=inputs.real_dtype) * delta_f * inputs.total_mass_seconds
    )
    generated_plus, generated_cross = _evaluate_polarizations(
        inputs,
        geometric_frequencies,
        reference_mf,
    )
    plus[first_bin:stop_bin] = generated_plus
    cross[first_bin:stop_bin] = generated_cross
    epoch = -1.0 / delta_f
    return (
        FrequencySeries(
            TorchArrayData(plus),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
        FrequencySeries(
            TorchArrayData(cross),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
    )


def _sequence_frequencies(sample_points, inputs: _WaveformInputs) -> torch.Tensor:
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("SEOBNRv4T surrogate sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SEOBNRv4T surrogate sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SEOBNRv4T surrogate sample_points must be positive")
    return frequencies


def seobnrv4t_surrogate_fd_sequence_torch(**parameters):
    """Evaluate ``SEOBNRv4T_surrogate`` at arbitrary frequencies with Torch."""

    inputs = _waveform_inputs(parameters, sequence=True)
    frequencies = _sequence_frequencies(parameters["sample_points"], inputs)
    _, reference_mf = _resolve_frequency_bounds(
        inputs,
        float(frequencies[0]),
        float(frequencies[-1]),
    )
    plus, cross = _evaluate_polarizations(
        inputs,
        frequencies * inputs.total_mass_seconds,
        reference_mf,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "seobnrv4t_surrogate_fd_sequence_torch",
    "seobnrv4t_surrogate_fd_torch",
    "seobnrv4t_surrogate_native_supported",
    "seobnrv4t_surrogate_sequence_native_supported",
]
