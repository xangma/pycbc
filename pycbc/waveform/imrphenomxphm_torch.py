# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native default-mode IMRPhenomXPHM waveforms.

The native path combines the existing XHM co-precessing modes with the XP MSA
Euler angles.  Mode generation, Wigner rotations, and polarization assembly
execute on the active Torch device; scalar source-frame setup remains on the
host.  The bounded implementation covers the default XPHM mode set with MSA
precession version 223 (and its 300 alias), convention 1, and final-spin modes
0, 3, and 4.  Other configurations continue to use lalsimulation.
"""

from __future__ import annotations

import math

import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomxas_torch import (
    _XAS_MODE_POLARIZATION_FACTOR,
    _next_power_of_two,
)
from pycbc.waveform.imrphenomxhm_torch import (
    _SequenceCore,
    _active_mode_samples,
)
from pycbc.waveform.imrphenomxp_msa_torch import msa_angles
from pycbc.waveform.imrphenomxp_torch import (
    _MSA_CONVENTION,
    _MSA_FINAL_SPIN_MODES,
    _MSA_PREC_VERSIONS,
    _as_float,
    _build_model,
    _integer_or_default,
    _sequence_frequencies,
    _series_from_active_samples,
    _validated_inputs,
    _xas_samples,
    imrphenomxp_native_supported,
)
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils


_PI = math.pi
_DEFAULT_PREC_VERSION = 300
_DEFAULT_CONVENTION = 1
_DEFAULT_FINAL_SPIN = 4
_COPRECESSING_MODES = (
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
)


def _xp_params(params):
    xp = dict(params)
    xp["approximant"] = "IMRPhenomXP"
    xp["mode_array"] = None
    return xp


def imrphenomxphm_native_supported(params):
    """Return whether ``params`` select the bounded native XPHM model."""

    if params.get("approximant") != "IMRPhenomXPHM":
        return False
    if params.get("mode_array") is not None:
        return False
    prec_version = _integer_or_default(
        params.get("phenom_x_prec_version"),
        _DEFAULT_PREC_VERSION,
    )
    convention = _integer_or_default(
        params.get("phenom_xp_convention"),
        _DEFAULT_CONVENTION,
    )
    final_spin_mod = _integer_or_default(
        params.get("phenom_xp_final_spin_mod"),
        _DEFAULT_FINAL_SPIN,
    )
    return (
        prec_version in _MSA_PREC_VERSIONS
        and convention == _MSA_CONVENTION
        and final_spin_mod in _MSA_FINAL_SPIN_MODES
        and imrphenomxp_native_supported(_xp_params(params))
    )


def imrphenomxphm_sequence_native_supported(params):
    """Return whether arbitrary-frequency XPHM generation is native."""

    return imrphenomxphm_native_supported(params)


def _coprecessing_params(params, inputs):
    aligned = dict(params)
    aligned.update(
        approximant="IMRPhenomXHM",
        mass1=inputs.mass1,
        mass2=inputs.mass2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=inputs.chi1_l,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=inputs.chi2_l,
        inclination=0.0,
        coa_phase=inputs.carrier_phase,
        long_asc_nodes=0.0,
        mode_array=None,
    )
    return aligned


def _coprecessing_final_spin(model):
    if model.final_spin is not None:
        return model.final_spin
    inputs = model.inputs
    return float(
        IMRPhenomX_utils.get_remnant_fMs(
            inputs.mass1,
            inputs.mass2,
            inputs.chi1_l,
            inputs.chi2_l,
            chip=inputs.chip,
        ).final_spin.item()
    )


def _mode_angles(model, frequencies, mprime):
    inputs = model.inputs
    velocity = torch.pow(
        _PI
        * inputs.total_mass_seconds
        * frequencies
        * (2.0 / mprime),
        1.0 / 3.0,
    )
    alpha, epsilon, cos_beta = msa_angles(velocity, model.msa_state)
    # LAL initializes all default XPHM mode offsets from the m'=2 reference
    # angle, while evaluating the running angles at 2 f / m'.
    alpha = alpha - model.alpha_offset
    epsilon = epsilon - model.epsilon_offset
    cos_half = torch.sqrt(torch.abs(0.5 * (1.0 + cos_beta)))
    sin_half = torch.sqrt(torch.abs(0.5 * (1.0 - cos_beta)))
    return alpha, epsilon, cos_half, sin_half


def _wigner_columns(ell, mprime, cosine, sine):
    """Return LAL's d^l_(m,+/-m') columns for the native mode set."""

    c2 = cosine * cosine
    c3 = c2 * cosine
    c4 = c3 * cosine
    s2 = sine * sine
    s3 = s2 * sine
    s4 = s3 * sine
    if (ell, mprime) == (2, 2):
        positive = (
            s4,
            2.0 * cosine * s3,
            math.sqrt(6.0) * s2 * c2,
            2.0 * c3 * sine,
            c4,
        )
        negative = (
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    elif (ell, mprime) == (2, 1):
        positive = (
            2.0 * cosine * s3,
            3.0 * c2 * s2 - s4,
            math.sqrt(6.0) * (c3 * sine - cosine * s3),
            c2 * (c2 - 3.0 * s2),
            -2.0 * c3 * sine,
        )
        negative = (
            -positive[4],
            positive[3],
            -positive[2],
            positive[1],
            -positive[0],
        )
    elif (ell, mprime) == (3, 3):
        c5 = c4 * cosine
        c6 = c5 * cosine
        s5 = s4 * sine
        s6 = s5 * sine
        positive = (
            s6,
            math.sqrt(6.0) * cosine * s5,
            math.sqrt(15.0) * c2 * s4,
            2.0 * math.sqrt(5.0) * c3 * s3,
            math.sqrt(15.0) * c4 * s2,
            math.sqrt(6.0) * c5 * sine,
            c6,
        )
        negative = (
            positive[6],
            -positive[5],
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    elif (ell, mprime) == (3, 2):
        c5 = c4 * cosine
        c6 = c5 * cosine
        s5 = s4 * sine
        positive = (
            math.sqrt(6.0) * cosine * s5,
            s4 * (5.0 * c2 - s2),
            math.sqrt(10.0) * s3 * (2.0 * c3 - cosine * s2),
            math.sqrt(30.0) * c2 * (c2 - s2) * s2,
            math.sqrt(10.0) * c3 * (c2 * sine - 2.0 * s3),
            c4 * (c2 - 5.0 * s2),
            -math.sqrt(6.0) * c5 * sine,
        )
        negative = (
            -positive[6],
            positive[5],
            -positive[4],
            positive[3],
            -positive[2],
            positive[1],
            -positive[0],
        )
    elif (ell, mprime) == (4, 4):
        c5 = c4 * cosine
        c6 = c5 * cosine
        c7 = c6 * cosine
        c8 = c7 * cosine
        s5 = s4 * sine
        s6 = s5 * sine
        s7 = s6 * sine
        s8 = s7 * sine
        positive = (
            s8,
            2.0 * math.sqrt(2.0) * cosine * s7,
            2.0 * math.sqrt(7.0) * c2 * s6,
            2.0 * math.sqrt(14.0) * c3 * s5,
            math.sqrt(70.0) * c4 * s4,
            2.0 * math.sqrt(14.0) * c5 * s3,
            2.0 * math.sqrt(7.0) * c6 * s2,
            2.0 * math.sqrt(2.0) * c7 * sine,
            c8,
        )
        negative = (
            positive[8],
            -positive[7],
            positive[6],
            -positive[5],
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    else:  # pragma: no cover - guarded by the fixed native mode set
        raise ValueError(f"unsupported XPHM co-precessing mode {(ell, mprime)}")
    return positive, negative


def _twist_mode(model, frequencies, samples, ell, mprime):
    inputs = model.inputs
    alpha, epsilon, cosine, sine = _mode_angles(
        model,
        frequencies,
        mprime,
    )
    positive, negative = _wigner_columns(
        ell,
        mprime,
        cosine,
        sine,
    )
    plus_sum = torch.zeros_like(samples)
    cross_sum = torch.zeros_like(samples)
    for index, emm in enumerate(range(-ell, ell + 1)):
        harmonic = spin_weighted_spherical_harmonic(
            inputs.theta_jn,
            0.0,
            -2,
            ell,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        negative_term = (
            torch.exp(-1j * emm * alpha) * negative[index] * harmonic
        )
        positive_term = (
            torch.exp(1j * emm * alpha)
            * positive[index]
            * torch.conj(harmonic)
        )
        if ell % 2:
            plus_sum += negative_term - positive_term
            cross_sum += 1j * (negative_term + positive_term)
        else:
            plus_sum += negative_term + positive_term
            cross_sum += 1j * (negative_term - positive_term)

    factor = torch.exp(-1j * mprime * epsilon) * samples / 2.0
    return factor * plus_sum, factor * cross_sum


def _twist_default_modes(model, frequencies, params):
    inputs = model.inputs
    carrier = _xas_samples(model, frequencies)
    core = _SequenceCore(carrier * _XAS_MODE_POLARIZATION_FACTOR)
    active_modes = _active_mode_samples(
        core,
        _coprecessing_params(params, inputs),
        _COPRECESSING_MODES,
        frequencies=frequencies,
        reference_frequency=inputs.f_ref,
        final_spin=_coprecessing_final_spin(model),
    )
    plus = torch.zeros_like(carrier)
    cross = torch.zeros_like(carrier)
    for ell, mprime in _COPRECESSING_MODES:
        mode_plus, mode_cross = _twist_mode(
            model,
            frequencies,
            active_modes[ell, mprime],
            ell,
            mprime,
        )
        plus += mode_plus
        cross += mode_cross

    cosine = math.cos(2.0 * inputs.polarization_rotation)
    sine = math.sin(2.0 * inputs.polarization_rotation)
    plus, cross = cosine * plus + sine * cross, cosine * cross - sine * plus
    cosine = math.cos(2.0 * inputs.long_asc_nodes)
    sine = math.sin(2.0 * inputs.long_asc_nodes)
    return cosine * plus + sine * cross, cosine * cross - sine * plus


def imrphenomxphm_fd_torch(**params):
    """Generate a supported regular-grid IMRPhenomXPHM waveform with Torch."""

    if not imrphenomxphm_native_supported(params):
        raise ValueError("unsupported parameters for native Torch IMRPhenomXPHM")
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = _as_float(params.get("f_final"))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXPHM frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXPHM delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXPHM f_final must be non-negative")

    xp_params = _xp_params(params)
    inputs = _validated_inputs(xp_params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXPHM cutoff) is <= f_lower")
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
    model = _build_model(inputs)
    plus, cross = _twist_default_modes(model, frequencies, params)
    return (
        _series_from_active_samples(
            inputs, plus, npoints, first_bin, stop_bin, delta_f
        ),
        _series_from_active_samples(
            inputs, cross, npoints, first_bin, stop_bin, delta_f
        ),
    )


def imrphenomxphm_fd_sequence_torch(**params):
    """Evaluate supported IMRPhenomXPHM configurations with Torch."""

    if not imrphenomxphm_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomXPHM sequence parameters are not supported by the "
            "native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXPHM requires TorchScheme")
    frequencies = _sequence_frequencies(params["sample_points"])
    xp_params = _xp_params(params)
    inputs = _validated_inputs(
        xp_params,
        sequence=True,
        default_reference_frequency=float(frequencies[0].item()),
    )
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    active = frequencies <= cutoff_frequency
    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    if bool(torch.any(active)):
        model = _build_model(inputs)
        plus[active], cross[active] = _twist_default_modes(
            model,
            frequencies[active],
            params,
        )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "imrphenomxphm_fd_sequence_torch",
    "imrphenomxphm_fd_torch",
    "imrphenomxphm_native_supported",
    "imrphenomxphm_sequence_native_supported",
]
