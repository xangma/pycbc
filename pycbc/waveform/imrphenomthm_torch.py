# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native mode assembly for time-domain IMRPhenomTHM.

The calibrated positive-m modes are generated on the active Torch device.
Negative-m modes use aligned-spin symmetry, and the selected modes are
projected directly with Torch spin-weighted spherical harmonics. The
implementation follows LALSuite 7.26.1. Well-conditioned default-mode requests
run natively by default; explicit switches retain the broader native subset.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData

from ._spherical_harmonics_torch import spin_weighted_spherical_harmonic
from .imrphenomt_amplitude_torch import (
    amplitude_lm,
    build_hm_amplitude_coefficients,
)
from .imrphenomt_torch import (
    _IMRPhenomTCore,
    _build_imrphenomt_core,
    _estimated_frequency_root_width_samples,
    _imrphenomt_family_native_supported,
    _parse_inputs,
)
from .imrphenomthm_phase_torch import (
    build_hm_phase_coefficients,
    phase_lm,
)

_PI = math.pi
_POSITIVE_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
_DEFAULT_MODES = _POSITIVE_MODES + tuple((ell, -emm) for ell, emm in _POSITIVE_MODES)
_SUPPORTED_MODES = frozenset(_DEFAULT_MODES)
_PHASE_OFFSETS = {
    (2, 1): 0.5 * _PI,
    (3, 3): -0.5 * _PI,
    (4, 4): _PI,
    (5, 5): 0.5 * _PI,
}
_DEFAULT_ROOT_WIDTH_LIMIT_SAMPLES = 0.21
_DEFAULT_MAX_MASS_RATIO = 10.0


def _normalize_modes(mode_array, approximant="IMRPhenomTHM"):
    """Validate and deduplicate a public PhenomT higher-mode array."""
    if mode_array is None:
        return _DEFAULT_MODES
    try:
        requested = tuple(mode_array)
    except TypeError as exc:
        raise ValueError(
            f"{approximant} mode_array must be an iterable of modes"
        ) from exc
    if not requested:
        raise ValueError(f"{approximant} mode_array must not be empty")

    modes = []
    for requested_mode in requested:
        try:
            ell, emm = requested_mode
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"each {approximant} mode must be an (ell, m) pair"
            ) from exc
        if not isinstance(ell, Integral) or not isinstance(emm, Integral):
            raise ValueError(f"{approximant} mode indices must be integers")
        mode = (int(ell), int(emm))
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"unsupported {approximant} mode {mode!r}; available modes are "
                f"{_DEFAULT_MODES}"
            )
        if mode not in modes:
            modes.append(mode)
    return tuple(modes)


def imrphenomthm_native_supported(parameters):
    """Return whether the native higher-mode port covers ``parameters``."""
    if not _imrphenomt_family_native_supported(parameters, "IMRPhenomTHM"):
        return False
    try:
        _normalize_modes(parameters.get("mode_array"))
    except ValueError:
        return False
    return True


def imrphenomthm_default_native_supported(parameters):
    """Return whether an unflagged request is safe for default native use."""
    if not imrphenomthm_native_supported(parameters):
        return False
    if parameters.get("mode_array") is not None:
        return False
    try:
        inputs = _parse_inputs(parameters)
    except ValueError:
        return False
    mass_ratio = inputs.mass1 / inputs.mass2
    return (
        mass_ratio <= _DEFAULT_MAX_MASS_RATIO
        and _estimated_frequency_root_width_samples(inputs)
        <= _DEFAULT_ROOT_WIDTH_LIMIT_SAMPLES
    )


def _is_suppressed_odd_mode(core, mode):
    """Match LAL's exact equal-binary shortcut for odd-m modes."""
    if mode[1] % 2 == 0:
        return False
    inputs = core.inputs
    delta = (inputs.mass1 - inputs.mass2) / (inputs.mass1 + inputs.mass2)
    return delta < 1.0e-10 and abs(inputs.spin1z - inputs.spin2z) < 1.0e-10


def _positive_mode(core: _IMRPhenomTCore, mode):
    """Generate one calibrated positive-m mode on the core device."""
    if _is_suppressed_odd_mode(core, mode):
        zero = torch.zeros_like(core.phase22)
        return torch.complex(zero, zero)

    _, _, spin1z, spin2z = core.binary.unbind()
    amplitude_coefficients = build_hm_amplitude_coefficients(
        mode,
        spin1z,
        spin2z,
        core.final_mass,
        core.final_spin,
        core.phase_coefficients,
        final_spin_prec=core.final_spin_prec,
    )
    amplitude = amplitude_lm(
        core.time_m,
        core.x_orbital,
        amplitude_coefficients,
    )

    if mode == (2, 2):
        amplitude = torch.abs(amplitude)
        phase = core.phase22 - core.reference_phase22
    else:
        phase_coefficients = build_hm_phase_coefficients(
            mode,
            spin1z,
            spin2z,
            core.final_mass,
            core.final_spin,
            core.phase_coefficients,
            amplitude_coefficients,
            final_spin_prec=core.final_spin_prec,
        )
        phase = phase_lm(
            core.time_m,
            phase_coefficients,
            core.phase_coefficients,
            amplitude_coefficients,
        )
        phase = phase - 0.5 * mode[1] * core.reference_phase22 - _PHASE_OFFSETS[mode]

    carrier = torch.complex(torch.cos(phase), -torch.sin(phase))
    return core.amplitude_factor * amplitude * carrier


def _assemble_modes(core, modes):
    """Generate requested modes, evaluating each positive-m family once."""
    positive_modes = {}
    result = {}
    for mode in modes:
        positive_mode = (mode[0], abs(mode[1]))
        if positive_mode not in positive_modes:
            positive_modes[positive_mode] = _positive_mode(core, positive_mode)
        positive = positive_modes[positive_mode]
        result[mode] = positive if mode[1] > 0 else (-1) ** mode[0] * positive.conj()
    return result


def _project_modes(core, modes, mode_data):
    """Project selected modes into the PyCBC plus/cross convention."""
    inputs = core.inputs
    harmonic_phi = 0.5 * _PI - inputs.coa_phase
    strain = torch.zeros_like(core.phase22, dtype=mode_data[modes[0]].dtype)
    for ell, emm in modes:
        harmonic = spin_weighted_spherical_harmonic(
            inputs.inclination,
            harmonic_phi,
            -2,
            ell,
            emm,
            dtype=core.binary.dtype,
            device=core.binary.device,
        )
        strain = strain + harmonic * mode_data[(ell, emm)]

    plus = strain.real
    cross = -strain.imag
    if inputs.long_asc_nodes:
        rotation = 2.0 * inputs.long_asc_nodes
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        plus, cross = (
            cosine * plus + sine * cross,
            cosine * cross - sine * plus,
        )
    return plus, cross


def imrphenomthm_td_torch(**parameters):
    """Generate native time-domain IMRPhenomTHM polarizations with Torch."""
    modes = _normalize_modes(parameters.get("mode_array"))
    core = _build_imrphenomt_core(parameters)
    mode_data = _assemble_modes(core, modes)
    plus, cross = _project_modes(core, modes, mode_data)
    return (
        TimeSeries(
            TorchArrayData(plus),
            delta_t=core.inputs.delta_t,
            epoch=core.epoch,
            copy=False,
        ),
        TimeSeries(
            TorchArrayData(cross),
            delta_t=core.inputs.delta_t,
            epoch=core.epoch,
            copy=False,
        ),
    )


__all__ = [
    "imrphenomthm_default_native_supported",
    "imrphenomthm_native_supported",
    "imrphenomthm_td_torch",
]
