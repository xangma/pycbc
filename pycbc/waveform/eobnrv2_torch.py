# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native EOBNRv2 reduced-order waveforms.

This module reconstructs ``EOBNRv2_ROM`` and ``EOBNRv2HM_ROM`` from the public
``EOBNRv2HMROM_*.dat`` data without calling ``lalsimulation``. Binary ROM
loading remains host-side; mass-ratio interpolation, reduced-basis
reconstruction, frequency interpolation, and polarization assembly run on the
active Torch device. Both regular grids and strictly increasing arbitrary-
frequency sequences are supported.

The native path is opt-in through ``PYCBC_EOBNRV2_NATIVE=1`` or the global
Torch-native switch. The dominant-mode approximant uses ``(2, 2)``; the
higher-mode approximant adds ``(2, 1)``, ``(3, 3)``, ``(4, 4)``, and
``(5, 5)``, and accepts explicit subsets of those positive-m modes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _is_default_order,
    _is_nonzero,
)

_DOMINANT_APPROXIMANT = "EOBNRv2_ROM"
_HIGHER_MODE_APPROXIMANT = "EOBNRv2HM_ROM"
_APPROXIMANTS = (_DOMINANT_APPROXIMANT, _HIGHER_MODE_APPROXIMANT)
_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
_MODE_SET = frozenset(_MODES)
_MODE_NAMES = tuple(f"{ell}{emm}" for ell, emm in _MODES)
_Q_MAX = 11.9894197212
_MF_ROM_MIN = 0.0003940393857519091
_MF_ROM_MAX = 0.285
_MF_22_MAX = 0.14
_ROM_TOTAL_MASS = 10.0
_Q_COUNT = 301
_FREQUENCY_COUNT = 300
_AMP_BASIS_COUNT = 10
_PHASE_BASIS_COUNT = 20
_ALL_SPIN_KEYS = _TRANSVERSE_SPIN_KEYS + ("spin1z", "spin2z")
_ROM_SHAPES = {
    "q": (_Q_COUNT,),
    **{
        f"{field}_{mode_name}": shape
        for mode_name in _MODE_NAMES
        for field, shape in (
            ("freq", (_FREQUENCY_COUNT,)),
            ("Camp", (_AMP_BASIS_COUNT, _Q_COUNT)),
            ("Cphi", (_PHASE_BASIS_COUNT, _Q_COUNT)),
            ("Bamp", (_FREQUENCY_COUNT, _AMP_BASIS_COUNT)),
            ("Bphi", (_FREQUENCY_COUNT, _PHASE_BASIS_COUNT)),
            ("shifttime", (_Q_COUNT,)),
            ("shiftphase", (_Q_COUNT,)),
        )
    },
}


def _active_mode_indices(approximant, mode_array) -> tuple[int, ...]:
    """Return requested modes in canonical ROM order."""

    if mode_array is None:
        if approximant == _HIGHER_MODE_APPROXIMANT:
            return tuple(range(len(_MODES)))
        return (0,)
    if approximant != _HIGHER_MODE_APPROXIMANT:
        raise ValueError(
            "mode_array is available only for EOBNRv2HM_ROM"
        )

    requested = set()
    for mode in mode_array:
        try:
            raw_ell, raw_emm = mode
        except (TypeError, ValueError):
            raise ValueError("mode_array entries must be (l, m) pairs")
        ell, emm = int(raw_ell), int(raw_emm)
        if (ell, emm) != (raw_ell, raw_emm):
            raise ValueError("mode_array entries must contain integers")
        if (ell, emm) not in _MODE_SET:
            raise ValueError(
                f"mode ({ell}, {emm}) is not available in EOBNRv2HM_ROM"
            )
        requested.add((ell, emm))
    return tuple(
        index for index, mode in enumerate(_MODES) if mode in requested
    )


def eobnrv2_native_supported(params) -> bool:
    """Return whether ``params`` are covered by the native ROM port."""

    if params.get("approximant", _DOMINANT_APPROXIMANT) not in _APPROXIMANTS:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in (
            *_DEFAULT_ONLY_ORDER_KEYS,
            "phase_order",
            "amplitude_order",
        )
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _ALL_SPIN_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + (
                "lambda1",
                "lambda2",
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    try:
        _active_mode_indices(
            params.get("approximant", _DOMINANT_APPROXIMANT),
            params.get("mode_array"),
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return not params.get("numrel_data", "")


def eobnrv2_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency generation is native."""

    return eobnrv2_native_supported(params)


def _find_rom_directory() -> Path:
    search_dirs = [Path(__file__).resolve().parent]
    search_dirs.extend(
        Path(directory)
        for directory in os.environ.get("LAL_DATA_PATH", "").split(
            os.pathsep
        )
        if directory
    )
    filenames = tuple(f"EOBNRv2HMROM_{name}.dat" for name in _ROM_SHAPES)
    for directory in search_dirs:
        if all((directory / filename).is_file() for filename in filenames):
            return directory
    raise FileNotFoundError(
        "EOBNRv2HMROM data files were not found next to this module or on "
        "$LAL_DATA_PATH"
    )


def _read_rom_array(directory: Path, name: str) -> np.ndarray:
    shape = _ROM_SHAPES[name]
    path = directory / f"EOBNRv2HMROM_{name}.dat"
    values = np.fromfile(path, dtype=np.float64)
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(
            f"{path} contains {values.size} doubles; expected {expected}"
        )
    return values.reshape(shape)


@dataclass(frozen=True)
class _ROMData:
    q: torch.Tensor
    frequency: torch.Tensor
    frequency_bounds: tuple[tuple[float, float], ...]
    coefficient_values: torch.Tensor
    coefficient_spline: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    amplitude_basis: torch.Tensor
    phase_basis: torch.Tensor


@lru_cache(maxsize=None)
def _load_rom(dtype: torch.dtype, device: torch.device) -> _ROMData:
    directory = _find_rom_directory()
    arrays = {
        name: _read_rom_array(directory, name) for name in _ROM_SHAPES
    }
    q = torch.as_tensor(arrays["q"], dtype=dtype, device=device)
    frequencies = np.stack(
        [arrays[f"freq_{mode_name}"] for mode_name in _MODE_NAMES]
    )
    coefficient_values = torch.as_tensor(
        np.stack(
            [
                np.column_stack(
                    (
                        arrays[f"Camp_{mode_name}"].T,
                        arrays[f"Cphi_{mode_name}"].T,
                        arrays[f"shifttime_{mode_name}"],
                        arrays[f"shiftphase_{mode_name}"],
                    )
                )
                for mode_name in _MODE_NAMES
            ],
            axis=1,
        ),
        dtype=dtype,
        device=device,
    )
    return _ROMData(
        q=q,
        frequency=torch.as_tensor(frequencies, dtype=dtype, device=device),
        frequency_bounds=tuple(
            (float(frequency[0]), float(frequency[-1]))
            for frequency in frequencies
        ),
        coefficient_values=coefficient_values,
        coefficient_spline=_natural_cubic_coeff(q, coefficient_values),
        amplitude_basis=torch.as_tensor(
            np.stack(
                [
                    arrays[f"Bamp_{mode_name}"]
                    for mode_name in _MODE_NAMES
                ]
            ),
            dtype=dtype,
            device=device,
        ),
        phase_basis=torch.as_tensor(
            np.stack(
                [
                    arrays[f"Bphi_{mode_name}"]
                    for mode_name in _MODE_NAMES
                ]
            ),
            dtype=dtype,
            device=device,
        ),
    )


@dataclass(frozen=True)
class _Inputs:
    active_mode_indices: tuple[int, ...]
    total_mass: float
    total_mass_seconds: float
    mass_ratio: float
    eta: float
    distance_m: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    f_ref: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    rom: _ROMData


def _validated_inputs(params, *, sequence=False) -> _Inputs:
    if not eobnrv2_native_supported(params):
        raise ValueError(
            "EOBNRv2 ROM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch EOBNRv2 ROM requires TorchScheme")

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    distance = float(params["distance"])
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )
    f_ref = float(params.get("f_ref", 0.0))
    if not all(
        math.isfinite(value)
        for value in (
            mass1,
            mass2,
            distance,
            inclination,
            coa_phase,
            long_asc_nodes,
            f_ref,
        )
    ):
        raise ValueError("EOBNRv2 ROM parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("EOBNRv2 ROM component masses must be positive")
    if distance <= 0.0:
        raise ValueError("EOBNRv2 ROM distance must be positive")
    if f_ref < 0.0:
        raise ValueError("EOBNRv2 ROM f_ref must be non-negative")

    mass_ratio = max(mass1 / mass2, mass2 / mass1)
    if mass_ratio > _Q_MAX:
        raise ValueError(
            f"EOBNRv2 ROM requires a mass ratio no greater than {_Q_MAX}"
        )
    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    eta = mass_ratio / (1.0 + mass_ratio) ** 2
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _Inputs(
        active_mode_indices=_active_mode_indices(
            params.get("approximant", _DOMINANT_APPROXIMANT),
            params.get("mode_array"),
        ),
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        mass_ratio=mass_ratio,
        eta=eta,
        distance_m=distance * 1.0e6 * lal.PC_SI,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        f_ref=f_ref,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        rom=_load_rom(real_dtype, device),
    )


def _reconstruct_modes(inputs: _Inputs):
    q = torch.as_tensor(
        inputs.mass_ratio, dtype=inputs.real_dtype, device=inputs.device
    )
    coefficients = _spline_eval(
        q,
        inputs.rom.q,
        inputs.rom.coefficient_values,
        *inputs.rom.coefficient_spline,
    )
    amplitude = torch.matmul(
        inputs.rom.amplitude_basis,
        coefficients[:, :_AMP_BASIS_COUNT].unsqueeze(-1),
    ).squeeze(-1)
    phase = torch.matmul(
        inputs.rom.phase_basis,
        coefficients[
            :,
            _AMP_BASIS_COUNT : _AMP_BASIS_COUNT + _PHASE_BASIS_COUNT,
        ].unsqueeze(-1),
    ).squeeze(-1)
    return amplitude, phase, coefficients[:, -2:]


def _reference_calibration(
    inputs: _Inputs,
    phase_22: torch.Tensor,
    shifts_22: torch.Tensor,
):
    frequency_22 = inputs.rom.frequency[0]
    phase_spline = _natural_cubic_coeff(frequency_22, phase_22)
    eta = inputs.eta
    peak_frequency = min(
        (0.2733 + 0.2316 * eta + 0.4463 * eta * eta)
        / (2.0 * math.pi),
        _MF_22_MAX,
    )
    peak_frequency = torch.as_tensor(
        peak_frequency, dtype=inputs.real_dtype, device=inputs.device
    )
    peak_time = -_spline_derivative(
        peak_frequency,
        frequency_22,
        *phase_spline,
    ) / (2.0 * math.pi)
    total_time_shift_22 = shifts_22[0] - 2.0 * math.pi * peak_time

    reference_frequency = inputs.f_ref * inputs.total_mass_seconds
    if reference_frequency == 0.0 or reference_frequency > _MF_22_MAX:
        reference_frequency = _MF_22_MAX
    elif reference_frequency < _MF_ROM_MIN:
        reference_frequency = _MF_ROM_MIN
    reference_frequency = torch.as_tensor(
        reference_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    phase_change = 2.0 * inputs.coa_phase + (
        _spline_eval(
            reference_frequency,
            frequency_22,
            phase_22,
            *phase_spline,
        )
        - total_time_shift_22 * reference_frequency
        - shifts_22[1]
    )
    return peak_time, phase_change


def _frequency_scale(mode: tuple[int, int], mass_ratio: float) -> float:
    decay = math.exp(-(mass_ratio - 1.0) / 5.0)
    if mode == (4, 4):
        return 1.0 - 0.25 * decay
    if mode == (5, 5):
        return 1.0 - decay / 3.0
    return 1.0


def _mode_amplitude_factor(
    mode: tuple[int, int],
    mass_ratio: float,
    eta: float,
    frequency_scale: float,
) -> float:
    base = math.sqrt(eta)
    mass_difference = (mass_ratio - 1.0) / (mass_ratio + 1.0)
    if mode == (2, 2):
        return base
    if mode in ((2, 1), (3, 3)):
        return base * mass_difference
    if mode == (4, 4):
        return math.sqrt(frequency_scale) * base * (1.0 - 3.0 * eta)
    if mode == (5, 5):
        return (
            frequency_scale ** (1.0 / 6.0)
            * base
            * mass_difference
            * (1.0 - 2.0 * eta)
        )
    raise ValueError(f"unsupported EOBNRv2 ROM mode {mode}")


def _mode_samples(
    inputs: _Inputs,
    mode_index: int,
    frequencies: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    shifts: torch.Tensor,
    peak_time_22: torch.Tensor,
    phase_change_22: torch.Tensor,
    frequency_scale: float,
) -> torch.Tensor:
    mode = _MODES[mode_index]
    mode_frequency = inputs.rom.frequency[mode_index] / frequency_scale
    phase_spline = _natural_cubic_coeff(mode_frequency, phase)
    total_time_shift = (
        shifts[0] * frequency_scale - 2.0 * math.pi * peak_time_22
    )
    constant_phase = mode[1] / 2.0 * phase_change_22 + shifts[1]

    amplitude_spline = _natural_cubic_coeff(mode_frequency, amplitude)
    interpolated_amplitude = _spline_eval(
        frequencies,
        mode_frequency,
        amplitude,
        *amplitude_spline,
    )
    interpolated_phase = (
        -_spline_eval(
            frequencies,
            mode_frequency,
            phase,
            *phase_spline,
        )
        + total_time_shift * frequencies
        + constant_phase
    )
    global_amplitude = (
        (inputs.total_mass / _ROM_TOTAL_MASS)
        * inputs.total_mass_seconds
        * 1.0e-16
        * 1.0e6
        * lal.PC_SI
        / inputs.distance_m
    )
    amplitude_scale = global_amplitude * _mode_amplitude_factor(
        mode,
        inputs.mass_ratio,
        inputs.eta,
        frequency_scale,
    )
    return amplitude_scale * interpolated_amplitude * torch.exp(
        1j * interpolated_phase
    )


def _polarizations(
    inputs: _Inputs,
    mode_number: tuple[int, int],
    mode: torch.Tensor,
):
    ell, emm = mode_number
    y_positive = spin_weighted_spherical_harmonic(
        inputs.inclination,
        0.0,
        -2,
        ell,
        emm,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    y_negative_conjugate = spin_weighted_spherical_harmonic(
        inputs.inclination,
        0.0,
        -2,
        ell,
        -emm,
        dtype=inputs.real_dtype,
        device=inputs.device,
    ).conj()
    parity = (-1) ** ell
    factor_plus = 0.5 * (
        y_positive + parity * y_negative_conjugate
    )
    factor_cross = 0.5j * (
        y_positive - parity * y_negative_conjugate
    )
    plus = torch.conj(factor_plus * mode)
    cross = torch.conj(factor_cross * mode)
    return plus, cross


def _rotate_polarizations(
    inputs: _Inputs,
    plus: torch.Tensor,
    cross: torch.Tensor,
):
    if inputs.long_asc_nodes:
        cosine = math.cos(2.0 * inputs.long_asc_nodes)
        sine = math.sin(2.0 * inputs.long_asc_nodes)
        plus, cross = (
            cosine * plus + sine * cross,
            cosine * cross - sine * plus,
        )
    return plus, cross


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def eobnrv2_fd_torch(**params):
    """Generate a regular-grid EOBNRv2 ROM waveform with Torch."""

    inputs = _validated_inputs(params)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("EOBNRv2 ROM sampling parameters must be finite")
    if delta_f <= 0.0:
        raise ValueError("EOBNRv2 ROM delta_f must be positive")
    if f_lower < 0.0 or f_final < 0.0:
        raise ValueError("EOBNRv2 ROM frequencies must be non-negative")

    # The legacy LAL implementation validates f_lower but starts every mode at
    # the first ROM frequency. Preserve that behavior for drop-in parity.

    delta_mf = delta_f * inputs.total_mass_seconds
    final_mf = (
        f_final * inputs.total_mass_seconds
        if f_final > 0.0
        else _MF_ROM_MAX
    )
    point_count = _next_power_of_two(int(final_mf / delta_mf)) + 1
    plus = torch.zeros(
        point_count, dtype=inputs.complex_dtype, device=inputs.device
    )
    cross = torch.zeros_like(plus)

    amplitudes, phases, shifts = _reconstruct_modes(inputs)
    peak_time_22, phase_change_22 = _reference_calibration(
        inputs, phases[0], shifts[0]
    )
    for mode_index in inputs.active_mode_indices:
        mode_number = _MODES[mode_index]
        frequency_scale = _frequency_scale(
            mode_number, inputs.mass_ratio
        )
        stored_low, stored_high = inputs.rom.frequency_bounds[mode_index]
        mode_low = stored_low / frequency_scale
        mode_high = min(stored_high / frequency_scale, final_mf)
        first_bin = math.ceil(mode_low / delta_mf)
        stop_bin = min(math.ceil(mode_high / delta_mf), point_count)
        if stop_bin <= first_bin:
            continue

        indices = torch.arange(first_bin, stop_bin, device=inputs.device)
        frequencies = indices.to(dtype=inputs.real_dtype) * delta_mf
        frequencies = frequencies.clone()
        frequencies[0] = max(mode_low, first_bin * delta_mf)
        frequencies[-1] = min(
            mode_high, (stop_bin - 1) * delta_mf
        )
        mode = _mode_samples(
            inputs,
            mode_index,
            frequencies,
            amplitudes[mode_index],
            phases[mode_index],
            shifts[mode_index],
            peak_time_22,
            phase_change_22,
            frequency_scale,
        )
        plus_segment, cross_segment = _polarizations(
            inputs, mode_number, mode
        )
        plus[first_bin:stop_bin] += plus_segment
        cross[first_bin:stop_bin] += cross_segment

    plus, cross = _rotate_polarizations(inputs, plus, cross)

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(
            TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _sequence_frequencies(sample_points, inputs):
    """Return a validated increasing sequence on the active Torch device."""

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
            "EOBNRv2 ROM sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("EOBNRv2 ROM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("EOBNRv2 ROM sample_points must be positive")
    if frequencies.numel() > 1 and bool(
        torch.any(frequencies[1:] <= frequencies[:-1])
    ):
        raise ValueError(
            "EOBNRv2 ROM sample_points must be strictly increasing"
        )
    return frequencies


def eobnrv2_fd_sequence_torch(**params):
    """Evaluate an EOBNRv2 ROM at arbitrary increasing frequencies."""

    inputs = _validated_inputs(params, sequence=True)
    frequencies_hz = _sequence_frequencies(params["sample_points"], inputs)
    frequencies = frequencies_hz * inputs.total_mass_seconds
    plus = torch.zeros_like(frequencies, dtype=inputs.complex_dtype)
    cross = torch.zeros_like(plus)

    amplitudes, phases, shifts = _reconstruct_modes(inputs)
    peak_time_22, phase_change_22 = _reference_calibration(
        inputs, phases[0], shifts[0]
    )
    for mode_index in inputs.active_mode_indices:
        mode_number = _MODES[mode_index]
        frequency_scale = _frequency_scale(
            mode_number, inputs.mass_ratio
        )
        stored_low, stored_high = inputs.rom.frequency_bounds[mode_index]
        mode_low = stored_low / frequency_scale
        mode_high = min(stored_high / frequency_scale, _MF_ROM_MAX)
        active = (frequencies >= mode_low) & (frequencies < mode_high)
        if not bool(torch.any(active)):
            continue
        mode = _mode_samples(
            inputs,
            mode_index,
            frequencies[active],
            amplitudes[mode_index],
            phases[mode_index],
            shifts[mode_index],
            peak_time_22,
            phase_change_22,
            frequency_scale,
        )
        plus_segment, cross_segment = _polarizations(
            inputs, mode_number, mode
        )
        plus[active] += plus_segment
        cross[active] += cross_segment

    plus, cross = _rotate_polarizations(inputs, plus, cross)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "eobnrv2_fd_sequence_torch",
    "eobnrv2_fd_torch",
    "eobnrv2_native_supported",
    "eobnrv2_sequence_native_supported",
]
