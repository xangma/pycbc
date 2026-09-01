# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Torch-native IMRPhenomXPNR waveform generation."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils
from pycbc.waveform.imrphenomxp_torch import (
    _integer_or_default,
    _series_from_active_samples,
)
from pycbc.waveform.imrphenomxas_torch import (
    _is_lal_int4_order,
    _next_power_of_two,
)
from pycbc.waveform.imrphenomxo4a_torch import (
    _LOW_IN_PLANE_SPIN,
    _XO4aModel,
    _antisymmetric_amplitude_ratio,
    _generation_modes,
    _pnr_interpolation_frequencies,
    _pnr_mode_maps,
    _requested_coprecessing_modes,
    _sequence_frequencies,
    _twist_selected_modes,
    _xo4a_inputs,
    imrphenomxo4a_native_supported,
)
from pycbc.waveform.imrphenomxpnr_torch import (
    build_pnr_coprecessing_deviations,
    build_pnr_spintaylor_angle_model,
    pnr_coprecessing_window,
    pnr_final_spin_model7,
    pnr_ringdown_beta,
)


_PREC_VERSION = 330
_FINAL_SPIN_MOD = 7


def _native_device_supported():
    """Reject MPS, whose float32 SpinTaylor angles lose required accuracy."""

    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


def _carrier_params(params):
    """Translate an XPNR request to the shared bounded XPHM carrier."""

    carrier = dict(params)
    carrier.update(
        approximant="IMRPhenomXO4a",
        phenom_x_prec_version=300,
        phenom_xp_convention=1,
        phenom_xp_final_spin_mod=0,
    )
    return carrier


def _has_resolvable_precession(params):
    """Reject the aligned-spin branch, which follows a different LAL path."""

    try:
        in_plane = math.sqrt(
            sum(
                float(params.get(f"spin{body}{axis}") or 0.0) ** 2
                for body in (1, 2)
                for axis in "xy"
            )
        )
    except (TypeError, ValueError, OverflowError):
        return True
    return math.isfinite(in_plane) and in_plane >= _LOW_IN_PLANE_SPIN


def imrphenomxpnr_native_supported(params):
    """Return whether ``params`` select the bounded native XPNR subset."""

    if not _native_device_supported():
        return False
    if params.get("approximant") != "IMRPhenomXPNR":
        return False
    if not _requested_coprecessing_modes(params):
        return False
    if _integer_or_default(
        params.get("phenom_x_prec_version"),
        _PREC_VERSION,
    ) != _PREC_VERSION:
        return False
    if _integer_or_default(params.get("phenom_xp_convention"), 1) != 1:
        return False
    if _integer_or_default(
        params.get("phenom_xp_final_spin_mod"),
        _FINAL_SPIN_MOD,
    ) != _FINAL_SPIN_MOD:
        return False
    # Version 330 does use the phase and spin PN orders when integrating its
    # numerical SpinTaylor trajectory. The native ODE implements LAL's fixed
    # defaults (3.5PN phase and 3PN spin), so only equivalent explicit values
    # can remain native. Tidal orders are immaterial for this BBH-only subset.
    if not _is_lal_int4_order(
        params.get("phase_order", -1),
        coerce=True,
        allowed=(-1, 7),
    ) or not _is_lal_int4_order(
        params.get("spin_order", -1),
        coerce=True,
        allowed=(-1, 6),
    ):
        return False
    if not _has_resolvable_precession(params):
        return False
    return imrphenomxo4a_native_supported(_carrier_params(params))


def imrphenomxpnr_sequence_native_supported(params):
    """Return whether arbitrary-frequency XPNR generation is native."""

    return imrphenomxpnr_native_supported(params)


def _xpnr_inputs(
    params,
    *,
    sequence=False,
    default_reference_frequency=None,
):
    return _xo4a_inputs(
        _carrier_params(params),
        sequence=sequence,
        default_reference_frequency=default_reference_frequency,
    )


def _build_model(inputs, *, inclination, coa_phase, f_min, delta_f):
    """Combine XPNR's numerical angles with the shared XPHM carrier."""

    angle_model = build_pnr_spintaylor_angle_model(
        inputs.mass1,
        inputs.mass2,
        inputs.spin1,
        inputs.spin2,
        inclination,
        f_min,
        delta_f,
        inputs.f_ref,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    inputs = replace(
        inputs,
        prec_version=_PREC_VERSION,
        final_spin_mod=_FINAL_SPIN_MOD,
        epsilon0=float(angle_model.frame.epsilon0.detach().cpu()),
    )
    single_spin = angle_model.single_spin
    beta_ringdown = pnr_ringdown_beta(single_spin)
    coprecessing_window = pnr_coprecessing_window(single_spin.mass_ratio)
    aligned_final_spin = torch.as_tensor(
        IMRPhenomX_utils.get_remnant_fMs(
            inputs.mass1,
            inputs.mass2,
            inputs.chi1_l,
            inputs.chi2_l,
        ).final_spin,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    precessing_final_spin = torch.clamp(
        pnr_final_spin_model7(
            inputs.eta,
            inputs.chi1_l,
            inputs.chi2_l,
            angle_model.msa_state["chiTot_perp"],
            beta_ringdown,
        ),
        -1.0,
        1.0,
    )
    final_spin = torch.clamp(
        coprecessing_window * aligned_final_spin
        + (1.0 - coprecessing_window) * precessing_final_spin,
        -1.0,
        1.0,
    )
    return _XO4aModel(
        inputs=inputs,
        inclination=inclination,
        coa_phase=coa_phase,
        msa_state=angle_model.msa_state,
        single_spin=single_spin,
        single_spin_msa_state=angle_model.single_spin_msa_state,
        alpha_parameters=angle_model.alpha_parameters,
        beta_parameters=angle_model.beta_parameters,
        coprecessing_deviations=build_pnr_coprecessing_deviations(
            single_spin,
            prec_version=_PREC_VERSION,
        ),
        aligned_final_spin=aligned_final_spin,
        precessing_final_spin=precessing_final_spin,
        beta_ringdown=beta_ringdown,
        coprecessing_window=coprecessing_window,
        final_spin=final_spin,
        spintaylor_angle_model=angle_model,
    )


def _model_and_angle_grid(
    inputs,
    inclination,
    coa_phase,
    frequencies,
    delta_f,
    modes,
):
    """Build the numerical-angle model and any shared higher-mode grid."""

    first_frequency = float(frequencies[0].detach().cpu())
    last_frequency = float(frequencies[-1].detach().cpu())
    model = _build_model(
        inputs,
        inclination=inclination,
        coa_phase=coa_phase,
        f_min=first_frequency,
        delta_f=delta_f,
    )
    angle_frequencies = _pnr_interpolation_frequencies(
        model,
        first_frequency,
        last_frequency,
        _pnr_mode_maps(model, modes),
    )
    angle_start = float(angle_frequencies[0].detach().cpu())
    trajectory_start = float(
        model.spintaylor_angle_model.integration.starting_frequency.detach().cpu()
    )
    if angle_start < trajectory_start:
        model = _build_model(
            inputs,
            inclination=inclination,
            coa_phase=coa_phase,
            f_min=angle_start,
            delta_f=delta_f,
        )
        angle_frequencies = _pnr_interpolation_frequencies(
            model,
            first_frequency,
            last_frequency,
            _pnr_mode_maps(model, modes),
        )
    return model, angle_frequencies


def imrphenomxpnr_fd_torch(**params):
    """Generate regular-grid native IMRPhenomXPNR polarizations."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final") or 0.0)
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXPNR frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXPNR delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXPNR f_final must be non-negative")

    modes = _generation_modes(params)
    inputs = _xpnr_inputs(params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXPNR cutoff) is <= f_lower")
    npoints = _next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(f_lower / delta_f)
    stop_bin = int(active_f_max / delta_f) + 1
    frequencies = (
        torch.arange(
            first_bin,
            stop_bin,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    if frequencies.numel() < 2:
        raise ValueError("IMRPhenomXPNR requires at least two active frequencies")

    inclination = float(params.get("inclination") or 0.0)
    coa_phase = float(params.get("coa_phase") or 0.0)
    model, angle_frequencies = _model_and_angle_grid(
        inputs,
        inclination,
        coa_phase,
        frequencies,
        delta_f,
        modes,
    )
    plus, cross = _twist_selected_modes(
        model,
        frequencies,
        active_f_max,
        modes,
        angle_frequencies=angle_frequencies,
        include_antisymmetric=True,
    )
    return (
        _series_from_active_samples(
            inputs,
            plus,
            npoints,
            first_bin,
            stop_bin,
            delta_f,
        ),
        _series_from_active_samples(
            inputs,
            cross,
            npoints,
            first_bin,
            stop_bin,
            delta_f,
        ),
    )


def imrphenomxpnr_fd_sequence_torch(**params):
    """Evaluate native IMRPhenomXPNR at arbitrary frequencies."""

    frequencies = _sequence_frequencies(
        params["sample_points"],
        approximant="IMRPhenomXPNR",
    )
    first_frequency = float(frequencies[0].item())
    last_frequency = float(frequencies[-1].item())
    modes = _generation_modes(params)
    inputs = _xpnr_inputs(
        params,
        sequence=True,
        default_reference_frequency=first_frequency,
    )
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    if cutoff_frequency <= first_frequency:
        raise ValueError(
            "the IMRPhenomXPNR cutoff must exceed the first sample point"
        )
    if not first_frequency <= inputs.f_ref <= last_frequency:
        raise ValueError(
            "IMRPhenomXPNR f_ref must lie between the first and last "
            "sample points"
        )

    active_f_max = min(last_frequency, cutoff_frequency)
    active = frequencies <= active_f_max
    active_frequencies = frequencies[active]
    inclination = float(params.get("inclination") or 0.0)
    coa_phase = float(params.get("coa_phase") or 0.0)
    model, angle_frequencies = _model_and_angle_grid(
        inputs,
        inclination,
        coa_phase,
        frequencies,
        0.0,
        modes,
    )
    antisymmetric_amplitude_ratio = None
    if (2, 2) in modes:
        # LAL smooths kappa over the complete requested grid before ignoring
        # samples above the waveform cutoff.
        antisymmetric_amplitude_ratio = _antisymmetric_amplitude_ratio(
            model,
            inputs.total_mass_seconds * frequencies,
        )[active]
    active_plus, active_cross = _twist_selected_modes(
        model,
        active_frequencies,
        active_f_max,
        modes,
        angle_frequencies=angle_frequencies,
        include_antisymmetric=True,
        antisymmetric_amplitude_ratio=antisymmetric_amplitude_ratio,
    )

    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    plus[active] = active_plus
    cross[active] = active_cross
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "imrphenomxpnr_fd_sequence_torch",
    "imrphenomxpnr_fd_torch",
    "imrphenomxpnr_native_supported",
    "imrphenomxpnr_sequence_native_supported",
]
