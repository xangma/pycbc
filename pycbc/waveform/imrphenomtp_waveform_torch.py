# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native waveform assembly for time-domain IMRPhenomTP."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData

from ._mode_rotation_torch import rotate_modes
from .imrphenomt_phase_torch import build_phase22_coefficients
from .imrphenomt_torch import (
    _IMRPhenomTCore,
    _build_imrphenomt_core_from_setup,
    _discrete_omega22,
    _discrete_phase22,
    _finite_float,
    _imrphenomt_family_native_supported,
    _prepare_imrphenomt_core,
    imrphenomt_td_torch,
)
from .imrphenomthm_torch import _assemble_modes, _project_modes
from .imrphenomtp_torch import (
    IMRPhenomTPEulerAngles,
    IMRPhenomTPOrbit,
    evolve_imrphenomtp_orbit,
    imrphenomtp_euler_angles,
    imrphenomtp_initial_final_spin,
)

_DOMINANT_MODES = ((2, 2), (2, -2))


@dataclass(frozen=True)
class IMRPhenomTPState:
    """Carrier, orbit, and angles used to assemble TP-family modes."""

    core: _IMRPhenomTCore
    orbit: IMRPhenomTPOrbit
    angles: IMRPhenomTPEulerAngles


def _integer_option(parameters, name, default):
    value = parameters.get(name)
    if value is None:
        return default
    try:
        return operator.index(value)
    except TypeError:
        return None


def _imrphenomtp_family_native_supported(parameters, approximant):
    """Check the shared default numerical-precession TP-family options."""
    return (
        _imrphenomt_family_native_supported(
            parameters,
            approximant,
            allow_transverse_spins=True,
        )
        and _integer_option(parameters, "phenom_x_prec_version", 300) == 300
        and _integer_option(parameters, "phenom_xp_convention", 1) == 1
        and _integer_option(parameters, "phenom_xp_final_spin_mod", 4) == 4
    )


def imrphenomtp_native_supported(parameters):
    """Return whether the native port covers these IMRPhenomTP options."""
    return _imrphenomtp_family_native_supported(
        parameters,
        "IMRPhenomTP",
    ) and parameters.get("mode_array") is None


def _ordered_spins(parameters, carrier):
    spin1 = tuple(
        _finite_float(parameters, f"spin1{axis}", 0.0)
        for axis in "xyz"
    )
    spin2 = tuple(
        _finite_float(parameters, f"spin2{axis}", 0.0)
        for axis in "xyz"
    )
    source_mass1 = _finite_float(parameters, "mass1")
    source_mass2 = _finite_float(parameters, "mass2")
    if source_mass1 < source_mass2:
        spin1, spin2 = spin2, spin1

    spin1 = carrier.binary.new_tensor(spin1)
    spin2 = carrier.binary.new_tensor(spin2)
    tolerance = 16.0 * torch.finfo(carrier.binary.dtype).eps
    if any(
        float(torch.linalg.vector_norm(spin).detach().cpu()) > 1.0 + tolerance
        for spin in (spin1, spin2)
    ):
        raise ValueError("IMRPhenomTP spin magnitudes must not exceed one")
    return spin1, spin2


def _reconstruct_merger_carrier(core, evolved_final_spin):
    """Rebuild TP's merger/ringdown carrier with its evolved remnant spin."""
    _, _, spin1z, spin2z = core.binary.unbind()
    coefficients = build_phase22_coefficients(
        core.phase_coefficients.eta,
        spin1z,
        spin2z,
        core.final_mass,
        core.final_spin,
        final_spin_prec=evolved_final_spin,
    )
    delta_time_m = core.time_m[1] - core.time_m[0]
    reconstructed_phase = _discrete_phase22(
        core.time_m,
        delta_time_m,
        coefficients,
    )
    reconstructed_omega = _discrete_omega22(
        core.time_m,
        delta_time_m,
        coefficients,
    )
    merger = core.time_m >= core.phase_coefficients.t_cut - delta_time_m
    return core._replace(
        final_spin_prec=evolved_final_spin,
        phase22=torch.where(merger, reconstructed_phase, core.phase22),
        x_orbital=torch.where(
            merger,
            torch.pow(0.5 * reconstructed_omega, 2.0 / 3.0),
            core.x_orbital,
        ),
    )


def _build_imrphenomtp_state(parameters):
    """Build the default numerical-precession TP state on the active device."""
    setup = _prepare_imrphenomt_core(parameters)
    spin1, spin2 = _ordered_spins(parameters, setup)
    initial_final_spin = imrphenomtp_initial_final_spin(
        setup,
        spin1,
        spin2,
    )
    core = _build_imrphenomt_core_from_setup(
        setup,
        final_spin_prec_override=initial_final_spin,
    )
    orbit = evolve_imrphenomtp_orbit(core, spin1, spin2)
    angles = imrphenomtp_euler_angles(core, orbit, convention=1)
    core = _reconstruct_merger_carrier(core, angles.evolved_final_spin)
    return IMRPhenomTPState(core=core, orbit=orbit, angles=angles)


def _rotation_target_modes(coprecessing_modes):
    """Return the inertial mode entries materialized by LAL's rotations.

    LAL creates both signs of an inactive co-precessing mode as zero-valued
    rotation targets, but for an active mode it creates only the explicitly
    requested signs. Preserve that custom-mode-array behavior exactly.
    """
    requested = set(coprecessing_modes)
    targets = []
    # The rotation routines materialize every lower multipole through LMAX,
    # even when none of its co-precessing modes were requested. Those entries
    # are observable through SimInspiralChooseTDModes as zero-valued modes.
    maximum_ell = max(mode[0] for mode in coprecessing_modes)
    for ell in range(2, maximum_ell + 1):
        for emm in range(ell + 1):
            positive = (ell, emm)
            negative = (ell, -emm)
            positive_active = positive in requested
            negative_active = negative in requested
            if positive_active or negative_active:
                if positive_active:
                    targets.append(positive)
                if negative_active and negative != positive:
                    targets.append(negative)
            else:
                targets.append(positive)
                if negative != positive:
                    targets.append(negative)
    return tuple(targets)


def _build_imrphenomtp_modes(parameters, coprecessing_modes):
    """Build co-precessing, J-frame, and L0-frame TP-family modes."""
    state = _build_imrphenomtp_state(parameters)
    active_coprecessing = _assemble_modes(state.core, coprecessing_modes)
    beta = torch.acos(torch.clamp(state.angles.cosbeta, -1.0, 1.0))
    target_modes = _rotation_target_modes(coprecessing_modes)
    zero_mode = torch.zeros_like(next(iter(active_coprecessing.values())))
    coprecessing = {
        mode: active_coprecessing.get(mode, zero_mode)
        for mode in target_modes
    }
    rotated_j = rotate_modes(
        coprecessing,
        state.angles.alpha,
        beta,
        state.angles.gamma,
    )
    j_frame = {mode: rotated_j[mode] for mode in target_modes}
    reference = state.angles.reference_index
    rotated_l0 = rotate_modes(
        j_frame,
        -state.angles.gamma[reference],
        -beta[reference],
        -state.angles.alpha[reference],
    )
    l0_frame = {mode: rotated_l0[mode] for mode in target_modes}
    return state, coprecessing, j_frame, l0_frame


def _aligned_limit(parameters):
    transverse_x = _finite_float(parameters, "spin1x", 0.0) + _finite_float(
        parameters, "spin2x", 0.0
    )
    transverse_y = _finite_float(parameters, "spin1y", 0.0) + _finite_float(
        parameters, "spin2y", 0.0
    )
    return math.hypot(transverse_x, transverse_y) < 1.0e-8


def _wrap_polarizations(core, polarizations):
    return tuple(
        TimeSeries(
            TorchArrayData(polarization),
            delta_t=core.inputs.delta_t,
            epoch=core.epoch,
            copy=False,
        )
        for polarization in polarizations
    )


def imrphenomtp_td_torch(**parameters):
    """Generate native time-domain IMRPhenomTP polarizations with Torch."""
    if _aligned_limit(parameters):
        return imrphenomt_td_torch(**parameters)

    state, _, _, modes = _build_imrphenomtp_modes(
        parameters,
        _DOMINANT_MODES,
    )
    selected_modes = tuple(modes)
    plus, cross = _project_modes(state.core, selected_modes, modes)
    return _wrap_polarizations(state.core, (plus, cross))


__all__ = ["imrphenomtp_native_supported", "imrphenomtp_td_torch"]
