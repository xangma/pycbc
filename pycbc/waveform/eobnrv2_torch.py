# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native dominant-mode EOBNRv2 reduced-order waveform.

This module reconstructs ``EOBNRv2_ROM`` from the public
``EOBNRv2HMROM_*.dat`` data without calling ``lalsimulation``. Binary ROM
loading remains host-side; mass-ratio interpolation, reduced-basis
reconstruction, frequency interpolation, and polarization assembly run on the
active Torch device.

The native path is opt-in through ``PYCBC_EOBNRV2_NATIVE=1`` or the global
Torch-native switch. ``EOBNRv2HM_ROM`` is intentionally left to its existing
LAL path until all five modes are covered.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import lal
import pycbc.scheme as _scheme
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

_APPROXIMANT = "EOBNRv2_ROM"
_MODE = (2, 2)
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
    "freq_22": (_FREQUENCY_COUNT,),
    "Camp_22": (_AMP_BASIS_COUNT, _Q_COUNT),
    "Cphi_22": (_PHASE_BASIS_COUNT, _Q_COUNT),
    "Bamp_22": (_FREQUENCY_COUNT, _AMP_BASIS_COUNT),
    "Bphi_22": (_FREQUENCY_COUNT, _PHASE_BASIS_COUNT),
    "shifttime_22": (_Q_COUNT,),
    "shiftphase_22": (_Q_COUNT,),
}


def eobnrv2_native_supported(params) -> bool:
    """Return whether ``params`` are covered by the dominant-mode port."""

    if params.get("approximant", _APPROXIMANT) != _APPROXIMANT:
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
    return params.get("mode_array") is None and not params.get(
        "numrel_data", ""
    )


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
    coefficient_values = torch.as_tensor(
        np.column_stack(
            (
                arrays["Camp_22"].T,
                arrays["Cphi_22"].T,
                arrays["shifttime_22"],
                arrays["shiftphase_22"],
            )
        ),
        dtype=dtype,
        device=device,
    )
    return _ROMData(
        q=q,
        frequency=torch.as_tensor(
            arrays["freq_22"], dtype=dtype, device=device
        ),
        coefficient_values=coefficient_values,
        coefficient_spline=_natural_cubic_coeff(q, coefficient_values),
        amplitude_basis=torch.as_tensor(
            arrays["Bamp_22"], dtype=dtype, device=device
        ),
        phase_basis=torch.as_tensor(
            arrays["Bphi_22"], dtype=dtype, device=device
        ),
    )


@dataclass(frozen=True)
class _Inputs:
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


def _validated_inputs(params) -> _Inputs:
    if not eobnrv2_native_supported(params):
        raise ValueError(
            "EOBNRv2_ROM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch EOBNRv2_ROM requires TorchScheme")

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    distance = float(params["distance"])
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    long_asc_nodes = float(params.get("long_asc_nodes", 0.0))
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
        raise ValueError("EOBNRv2_ROM parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("EOBNRv2_ROM component masses must be positive")
    if distance <= 0.0:
        raise ValueError("EOBNRv2_ROM distance must be positive")
    if f_ref < 0.0:
        raise ValueError("EOBNRv2_ROM f_ref must be non-negative")

    mass_ratio = max(mass1 / mass2, mass2 / mass1)
    if mass_ratio > _Q_MAX:
        raise ValueError(
            f"EOBNRv2_ROM requires a mass ratio no greater than {_Q_MAX}"
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


def _reconstruct_mode(inputs: _Inputs):
    q = torch.as_tensor(
        inputs.mass_ratio, dtype=inputs.real_dtype, device=inputs.device
    )
    coefficients = _spline_eval(
        q,
        inputs.rom.q,
        inputs.rom.coefficient_values,
        *inputs.rom.coefficient_spline,
    )
    amplitude = (
        inputs.rom.amplitude_basis @ coefficients[:_AMP_BASIS_COUNT]
    )
    phase = inputs.rom.phase_basis @ coefficients[
        _AMP_BASIS_COUNT : _AMP_BASIS_COUNT + _PHASE_BASIS_COUNT
    ]
    return amplitude, phase, coefficients[-2], coefficients[-1]


def _mode_samples(
    inputs: _Inputs,
    frequencies: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    shift_time: torch.Tensor,
    shift_phase: torch.Tensor,
) -> torch.Tensor:
    phase_spline = _natural_cubic_coeff(inputs.rom.frequency, phase)
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
        inputs.rom.frequency,
        *phase_spline,
    ) / (2.0 * math.pi)
    total_time_shift = shift_time - 2.0 * math.pi * peak_time

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
            inputs.rom.frequency,
            phase,
            *phase_spline,
        )
        - total_time_shift * reference_frequency
        - shift_phase
    )
    constant_phase = phase_change + shift_phase

    amplitude_spline = _natural_cubic_coeff(
        inputs.rom.frequency, amplitude
    )
    interpolated_amplitude = _spline_eval(
        frequencies,
        inputs.rom.frequency,
        amplitude,
        *amplitude_spline,
    )
    interpolated_phase = (
        -_spline_eval(
            frequencies,
            inputs.rom.frequency,
            phase,
            *phase_spline,
        )
        + total_time_shift * frequencies
        + constant_phase
    )
    amplitude_scale = (
        (inputs.total_mass / _ROM_TOTAL_MASS)
        * inputs.total_mass_seconds
        * 1.0e-16
        * 1.0e6
        * lal.PC_SI
        / inputs.distance_m
        * math.sqrt(inputs.eta)
    )
    return amplitude_scale * interpolated_amplitude * torch.exp(
        1j * interpolated_phase
    )


def _polarizations(inputs: _Inputs, mode: torch.Tensor):
    ell, emm = _MODE
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
    """Generate a regular-grid ``EOBNRv2_ROM`` waveform with Torch."""

    inputs = _validated_inputs(params)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("EOBNRv2_ROM sampling parameters must be finite")
    if delta_f <= 0.0:
        raise ValueError("EOBNRv2_ROM delta_f must be positive")
    if f_lower < 0.0 or f_final < 0.0:
        raise ValueError("EOBNRv2_ROM frequencies must be non-negative")

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

    first_bin = math.ceil(_MF_ROM_MIN / delta_mf)
    stop_bin = math.ceil(min(_MF_22_MAX, final_mf) / delta_mf)
    stop_bin = min(stop_bin, point_count)
    if stop_bin > first_bin:
        indices = torch.arange(first_bin, stop_bin, device=inputs.device)
        frequencies = (
            indices.to(dtype=inputs.real_dtype) * delta_mf
        )
        frequencies = frequencies.clone()
        frequencies[0] = max(_MF_ROM_MIN, first_bin * delta_mf)
        frequencies[-1] = min(
            _MF_22_MAX, final_mf, (stop_bin - 1) * delta_mf
        )
        amplitude, phase, shift_time, shift_phase = _reconstruct_mode(inputs)
        mode = _mode_samples(
            inputs,
            frequencies,
            amplitude,
            phase,
            shift_time,
            shift_phase,
        )
        plus_segment, cross_segment = _polarizations(inputs, mode)
        plus[first_bin:stop_bin] = plus_segment
        cross[first_bin:stop_bin] = cross_segment

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(
            TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


__all__ = ["eobnrv2_fd_torch", "eobnrv2_native_supported"]
