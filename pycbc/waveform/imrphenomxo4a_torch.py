# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native core of IMRPhenomXO4a.

This internal generator combines the tuned PNR Euler angles and co-precessing
deviations with the aligned-spin PhenomXAS/XHM co-precessing modes and the
dominant mode's antisymmetric counterpart. Public dispatch supports explicit
positive-m (2, 2), (2, 1), (3, 3), (3, 2), and (4, 4) co-precessing-mode
requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral

import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_eval,
)
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenompv2_torch import _assemble_twisted_polarizations
from pycbc.waveform.imrphenomxp_msa_torch import (
    build_msa_state,
    remap_source_frame_parameters_pnr,
)
from pycbc.waveform.imrphenomxp_torch import (
    _build_model as _build_xp_model,
    _integer_or_default,
    _series_from_active_samples,
    _twist_up as _twist_up_xp,
    _validated_inputs,
    _xas_samples,
    imrphenomxp_native_supported,
)
from pycbc.waveform.imrphenomxpnr_torch import (
    build_pnr_alpha_parameters,
    build_pnr_beta_parameters,
    build_pnr_coprecessing_deviations,
    build_pnr_single_spin_msa_state,
    pnr_alpha,
    pnr_beta,
    pnr_coprecessing_window,
    pnr_gamma,
    pnr_higher_mode_frequency_map,
    pnr_higher_mode_transition_frequencies,
    pnr_ringdown_beta,
    pnr_single_spin_mapping,
    pnr_spintaylor_alpha,
    pnr_spintaylor_beta,
)
from pycbc.waveform.imrphenomxhm_mode21_torch import (
    _qnm_fdamp_21,
    _qnm_fring_21,
)
from pycbc.waveform.imrphenomxhm_mode32_torch import (
    qnm_fdamp32_fit,
    qnm_fring32_fit,
)
from pycbc.waveform.imrphenomxhm_mode33_torch import (
    _qnm_fdamp_33,
    _qnm_fring_33,
)
from pycbc.waveform.imrphenomxhm_mode44_torch import (
    _qnm_fdamp_44,
    _qnm_fring_44,
)
from pycbc.waveform.imrphenomxhm_mode21_2019_torch import (
    _EMR_TWO_REGION_ETA,
)
from pycbc.waveform.imrphenomxas_torch import (
    Phase,
    PhaseDerivative,
    _XAS_MODE_POLARIZATION_FACTOR,
    _get_cutoff_fMs,
    _next_power_of_two,
    _phase_alignment_terms,
)
from pycbc.waveform.imrphenomxhm_torch import (
    _SequenceCore,
    _active_mode_samples,
)
from pycbc.waveform.imrphenomxphm_torch import _wigner_columns
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils
from pycbc.waveform._torch_jax import torch_context


_PREC_VERSION = 300
_LOW_IN_PLANE_SPIN = 1.0e-7
_PNR_INTERPOLATION_TOLERANCE = 1.0e-2
_MINIMUM_INTERPOLATION_DELTA_F = 1.0e-2
_ANTISYMMETRIC_SMOOTHING_WIDTH = 80
_ANTISYMMETRIC_TRANSITION_FRACTION = 0.85
_ANTISYMMETRIC_DERIVATIVE_STEP = 0.0005
_NATIVE_COPRECESSING_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
_NATIVE_COPRECESSING_MODE_SET = frozenset(_NATIVE_COPRECESSING_MODES)


@dataclass(frozen=True)
class _XO4aModel:
    inputs: object
    inclination: float
    coa_phase: float
    msa_state: object
    single_spin: object
    single_spin_msa_state: object
    alpha_parameters: object
    beta_parameters: object
    coprecessing_deviations: object
    aligned_final_spin: object
    precessing_final_spin: object
    beta_ringdown: object
    coprecessing_window: object
    final_spin: object
    spintaylor_angle_model: object | None = None


@dataclass(frozen=True)
class _PNRAngleSplines:
    """Shared tuned-angle interpolants evaluated by every active mode."""

    frequencies: torch.Tensor
    alpha: tuple
    beta: tuple
    gamma: tuple


@dataclass(frozen=True)
class _PNRTwistFrame:
    """Common source-frame remapping and angle offsets for all modes."""

    inputs: object
    angle_splines: _PNRAngleSplines
    alpha_offset: torch.Tensor
    epsilon_offset: torch.Tensor


@dataclass(frozen=True)
class _XO4aRingdowns:
    """Effective geometric ringdown frequencies needed by the native modes."""

    carrier: torch.Tensor
    carrier_damping: torch.Tensor
    mode21: torch.Tensor
    mode21_damping: torch.Tensor
    mode33: torch.Tensor
    mode33_damping: torch.Tensor
    mode32: torch.Tensor
    mode32_damping: torch.Tensor
    mode44: torch.Tensor
    mode44_damping: torch.Tensor
    shift_per_m: torch.Tensor


@dataclass(frozen=True)
class _PNRHigherModeFrequencyMap:
    """Parameters for mapping one higher mode onto the tuned (2, 2) angles."""

    ell: int
    mprime: int
    mf_lower: torch.Tensor
    mf_upper: torch.Tensor
    mf_ring_22: torch.Tensor
    mf_ring_lm: torch.Tensor

    def evaluate(self, geometric_frequencies):
        return pnr_higher_mode_frequency_map(
            geometric_frequencies,
            self.ell,
            self.mprime,
            self.mf_lower,
            self.mf_upper,
            self.mf_ring_22,
            self.mf_ring_lm,
        )


def _requested_coprecessing_modes(params):
    """Return the canonical requested positive-m modes, or ``None``.

    An omitted mode array selects LAL's complete default set. Explicit
    requests are deduplicated in model order, matching mode-array activation.
    """

    mode_array = params.get("mode_array")
    if mode_array is None:
        return _NATIVE_COPRECESSING_MODES
    try:
        requested = set()
        for mode in mode_array:
            ell, emm = mode
            if not isinstance(ell, Integral) or not isinstance(emm, Integral):
                return None
            family = (int(ell), int(emm))
            if family not in _NATIVE_COPRECESSING_MODE_SET:
                return None
            requested.add(family)
    except (TypeError, ValueError):
        return None
    return tuple(
        mode for mode in _NATIVE_COPRECESSING_MODES if mode in requested
    )


def _generation_modes(params):
    """Resolve modes for direct internal-generator calls."""

    modes = _requested_coprecessing_modes(params)
    if not modes:
        raise ValueError(
            "native Torch IMRPhenomXO4a requires positive (2, 2), "
            "(2, 1), (3, 3), (3, 2), and/or (4, 4) modes"
        )
    return modes


def _legacy_higher_mode_source_supported(params):
    """Return whether the legacy XHM amplitudes cover the source."""

    if "mass1" not in params or "mass2" not in params:
        # Support predicates are also used to inspect option boundaries before
        # source parameters are populated. Numeric validation occurs later.
        return True
    try:
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        spin1 = tuple(
            float(params.get(f"spin1{axis}") or 0.0) for axis in "xyz"
        )
        spin2 = tuple(
            float(params.get(f"spin2{axis}") or 0.0) for axis in "xyz"
        )
    except (TypeError, ValueError, OverflowError):
        return True
    values = (mass1, mass2, *spin1, *spin2)
    if not all(math.isfinite(value) for value in values):
        return True
    in_plane_spin = math.sqrt(
        spin1[0] ** 2
        + spin1[1] ** 2
        + spin2[0] ** 2
        + spin2[1] ** 2
    )
    if in_plane_spin < _LOW_IN_PLANE_SPIN:
        return False
    if mass1 <= 0.0 or mass2 <= 0.0:
        return True
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    return not (eta < _EMR_TWO_REGION_ETA and spin1[2] <= 0.9)


def _xp_params(params):
    """Translate an XO4a request to its bounded XP input convention."""

    xp_params = dict(params)
    xp_params.update(
        approximant="IMRPhenomXP",
        phenom_x_prec_version=_PREC_VERSION,
        phenom_xp_convention=1,
        phenom_xp_final_spin_mod=0,
        mode_array=None,
    )
    return xp_params


def imrphenomxo4a_native_supported(params):
    """Return whether ``params`` select the bounded native XO4a subset."""

    if params.get("approximant") != "IMRPhenomXO4a":
        return False
    modes = _requested_coprecessing_modes(params)
    if not modes:
        return False
    if _integer_or_default(
        params.get("phenom_x_prec_version"),
        _PREC_VERSION,
    ) != _PREC_VERSION:
        return False
    if _integer_or_default(params.get("phenom_xp_convention"), 1) != 1:
        return False
    if _integer_or_default(params.get("phenom_xp_final_spin_mod"), 0) != 0:
        return False
    if any(mode != (2, 2) for mode in modes) and not (
        _legacy_higher_mode_source_supported(params)
    ):
        return False
    return imrphenomxp_native_supported(_xp_params(params))


def imrphenomxo4a_sequence_native_supported(params):
    """Return whether arbitrary-frequency XO4a generation is native."""

    return imrphenomxo4a_native_supported(params)


def _xo4a_inputs(
    params,
    *,
    sequence=False,
    default_reference_frequency=None,
):
    return _validated_inputs(
        _xp_params(params),
        sequence=sequence,
        default_reference_frequency=default_reference_frequency,
    )


def _build_model(inputs, *, inclination, coa_phase):
    msa_state = build_msa_state(
        inputs.mass1,
        inputs.mass2,
        inputs.spin1,
        inputs.spin2,
        inputs.total_mass_seconds,
        inputs.f_ref,
    )
    source = torch.tensor(
        (
            inputs.mass1,
            inputs.mass2,
            *inputs.spin1,
            *inputs.spin2,
        ),
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    single_spin = pnr_single_spin_mapping(*source.unbind())
    single_spin_msa_state = build_pnr_single_spin_msa_state(
        single_spin,
        msa_state,
    )
    alpha_parameters = build_pnr_alpha_parameters(
        single_spin,
        msa_state,
        inputs.total_mass_seconds,
    )
    beta_parameters = build_pnr_beta_parameters(
        single_spin,
        msa_state,
        single_spin_msa_state,
    )
    coprecessing_deviations = build_pnr_coprecessing_deviations(
        single_spin,
        prec_version=_PREC_VERSION,
    )

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
    precessing_final_spin = torch.as_tensor(
        IMRPhenomX_utils.get_remnant_fMs(
            inputs.mass1,
            inputs.mass2,
            inputs.chi1_l,
            inputs.chi2_l,
            chip=inputs.chip,
        ).final_spin,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    beta_ringdown = pnr_ringdown_beta(single_spin)
    signed_precessing_spin = torch.copysign(
        torch.abs(precessing_final_spin),
        torch.cos(beta_ringdown),
    )
    coprecessing_window = pnr_coprecessing_window(single_spin.mass_ratio)
    final_spin = torch.clamp(
        coprecessing_window * aligned_final_spin
        + (1.0 - coprecessing_window) * signed_precessing_spin,
        -1.0,
        1.0,
    )
    return _XO4aModel(
        inputs=inputs,
        inclination=inclination,
        coa_phase=coa_phase,
        msa_state=msa_state,
        single_spin=single_spin,
        single_spin_msa_state=single_spin_msa_state,
        alpha_parameters=alpha_parameters,
        beta_parameters=beta_parameters,
        coprecessing_deviations=coprecessing_deviations,
        aligned_final_spin=aligned_final_spin,
        precessing_final_spin=signed_precessing_spin,
        beta_ringdown=beta_ringdown,
        coprecessing_window=coprecessing_window,
        final_spin=final_spin,
    )


def _coprecessing_params(model):
    """Return the aligned-spin parameters used by XO4a's XHM modes."""

    inputs = model.inputs
    return {
        "approximant": "IMRPhenomXHM",
        "mass1": inputs.mass1,
        "mass2": inputs.mass2,
        "spin1x": 0.0,
        "spin1y": 0.0,
        "spin1z": inputs.chi1_l,
        "spin2x": 0.0,
        "spin2y": 0.0,
        "spin2z": inputs.chi2_l,
        "distance": inputs.distance,
        "inclination": 0.0,
        "coa_phase": inputs.carrier_phase,
        "long_asc_nodes": 0.0,
        "f_ref": inputs.f_ref,
        "mode_array": None,
    }


def _xo4a_ringdowns(model):
    """Return LAL's effective carrier and higher-mode ringdowns."""

    inputs = model.inputs
    precessing_remnant = IMRPhenomX_utils.get_remnant_fMs(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        final_spin=model.precessing_final_spin,
    )
    final_mass = torch.as_tensor(
        1.0 - precessing_remnant.radiated_energy,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    ringdown_22_precessing = torch.as_tensor(
        precessing_remnant.ringdown_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    ringdown_21_precessing = (
        _qnm_fring_21(model.precessing_final_spin) / final_mass
    )
    damping_21_precessing = (
        _qnm_fdamp_21(model.precessing_final_spin) / final_mass
    )
    ringdown_33_precessing = (
        _qnm_fring_33(model.precessing_final_spin) / final_mass
    )
    damping_33_precessing = (
        _qnm_fdamp_33(model.precessing_final_spin) / final_mass
    )
    ringdown_32_precessing = torch.as_tensor(
        qnm_fring32_fit(model.precessing_final_spin),
        dtype=inputs.real_dtype,
        device=inputs.device,
    ) / final_mass
    damping_32_precessing = torch.as_tensor(
        qnm_fdamp32_fit(model.precessing_final_spin),
        dtype=inputs.real_dtype,
        device=inputs.device,
    ) / final_mass
    ringdown_44_precessing = (
        _qnm_fring_44(model.precessing_final_spin) / final_mass
    )
    damping_44_precessing = (
        _qnm_fdamp_44(model.precessing_final_spin) / final_mass
    )
    shift_per_m = (
        1.0 - torch.abs(torch.cos(model.beta_ringdown))
    ) * (ringdown_22_precessing - ringdown_21_precessing)

    carrier, carrier_damping = _get_cutoff_fMs(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        inputs.chip,
        final_spin=model.final_spin,
        coprecessing_deviations=model.coprecessing_deviations,
    )[:2]
    carrier = torch.as_tensor(
        carrier,
        dtype=inputs.real_dtype,
        device=inputs.device,
    ) - (1.0 - model.coprecessing_window) * 2.0 * shift_per_m
    return _XO4aRingdowns(
        carrier=carrier,
        carrier_damping=torch.as_tensor(
            carrier_damping,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ),
        mode21=ringdown_21_precessing - shift_per_m,
        mode21_damping=damping_21_precessing,
        mode33=ringdown_33_precessing - 3.0 * shift_per_m,
        mode33_damping=damping_33_precessing,
        mode32=ringdown_32_precessing - 2.0 * shift_per_m,
        mode32_damping=damping_32_precessing,
        mode44=ringdown_44_precessing - 4.0 * shift_per_m,
        mode44_damping=damping_44_precessing,
        shift_per_m=shift_per_m,
    )


def _linear_value_at(frequencies, values, frequency):
    target = frequencies.new_tensor(frequency)
    lower = float(frequencies[0].detach().cpu())
    upper = float(frequencies[-1].detach().cpu())
    if frequency < lower or frequency > upper:
        raise ValueError("IMRPhenomXO4a f_ref must lie on the active frequency range")
    if frequencies.numel() == 1:
        return values[0]
    right = int(torch.searchsorted(frequencies, target, right=False).item())
    if right == 0:
        return values[0]
    if right == frequencies.numel():
        return values[-1]
    left = right - 1
    weight = (target - frequencies[left]) / (
        frequencies[right] - frequencies[left]
    )
    return values[left] + weight * (values[right] - values[left])


def _natural_spline(knots, values):
    return values, *_natural_cubic_coeff(knots, values)


def _spline_values_at(knots, spline, frequencies):
    values, linear, quadratic, cubic = spline
    return _spline_eval(
        frequencies,
        knots,
        values,
        linear,
        quadratic,
        cubic,
    )


def _raw_pnr_alpha(model, geometric_frequencies):
    """Evaluate the model's unshifted tuned alpha prescription."""

    if model.spintaylor_angle_model is None:
        return pnr_alpha(
            geometric_frequencies,
            model.alpha_parameters,
            model.single_spin,
            model.msa_state,
        )
    return pnr_spintaylor_alpha(
        geometric_frequencies,
        model.alpha_parameters,
        model.single_spin,
        model.spintaylor_angle_model.angles,
        alpha_offset=model.spintaylor_angle_model.alpha_offset,
    )


def _raw_pnr_beta(model, geometric_frequencies):
    """Evaluate the model's unshifted tuned beta prescription."""

    if model.spintaylor_angle_model is None:
        return pnr_beta(
            geometric_frequencies,
            model.beta_parameters,
            model.single_spin,
            model.msa_state,
            model.single_spin_msa_state,
        )
    return pnr_spintaylor_beta(
        geometric_frequencies,
        model.beta_parameters,
        model.single_spin,
        model.spintaylor_angle_model.angles,
        model.msa_state,
        model.single_spin_msa_state,
    )


def _build_pnr_angle_splines(model, angle_frequencies):
    """Build the one PNR angle spline set shared by all co-precessing modes."""

    geometric_frequencies = model.inputs.total_mass_seconds * angle_frequencies
    alpha = _raw_pnr_alpha(model, geometric_frequencies)
    beta = _raw_pnr_beta(model, geometric_frequencies)
    # LAL integrates gamma from the alpha and beta splines on the uniform-Hz
    # interpolation grid, then constructs a spline for gamma itself.
    gamma = pnr_gamma(angle_frequencies, alpha, beta)
    return _PNRAngleSplines(
        frequencies=angle_frequencies,
        alpha=_natural_spline(angle_frequencies, alpha),
        beta=_natural_spline(angle_frequencies, beta),
        gamma=_natural_spline(angle_frequencies, gamma),
    )


def _pnr_spline_angles(splines, frequencies):
    """Evaluate raw alpha, beta, and gamma from shared PNR interpolants."""

    return tuple(
        _spline_values_at(splines.frequencies, spline, frequencies)
        for spline in (splines.alpha, splines.beta, splines.gamma)
    )


def _build_pnr_twist_frame(model, angle_frequencies):
    """Build the source-frame state shared by every co-precessing mode."""

    inputs = model.inputs
    angle_splines = _build_pnr_angle_splines(model, angle_frequencies)
    reference_frequency = angle_frequencies.new_tensor(inputs.f_ref)
    alpha_ref, beta_ref, gamma_ref = _pnr_spline_angles(
        angle_splines,
        reference_frequency,
    )
    source_frame = remap_source_frame_parameters_pnr(
        inputs.mass1,
        inputs.mass2,
        inputs.f_ref,
        model.coa_phase,
        model.inclination,
        inputs.spin1,
        inputs.spin2,
        inputs.total_mass_seconds,
        float(beta_ref.detach().cpu()),
    )
    twist_inputs = replace(
        inputs,
        theta_jn=source_frame.theta_jn,
        alpha0=source_frame.alpha0,
        polarization_rotation=source_frame.polarization_rotation,
    )
    return _PNRTwistFrame(
        inputs=twist_inputs,
        angle_splines=angle_splines,
        alpha_offset=alpha_ref - source_frame.alpha_offset_shift,
        epsilon_offset=-gamma_ref - inputs.epsilon0,
    )


def _pnr_mode_angles(frame, frequencies, mode_map=None):
    """Evaluate one mode's PNR angles with the common reference offsets."""

    angle_frequencies = frequencies
    if mode_map is not None:
        total_mass_seconds = frame.inputs.total_mass_seconds
        angle_frequencies = (
            mode_map.evaluate(total_mass_seconds * frequencies)
            / total_mass_seconds
        )
    raw_alpha, beta, raw_gamma = _pnr_spline_angles(
        frame.angle_splines,
        angle_frequencies,
    )
    return (
        raw_alpha - frame.alpha_offset,
        beta,
        -raw_gamma - frame.epsilon_offset,
    )


def _twist_pnr_mode(
    frame,
    frequencies,
    samples,
    ell,
    mprime,
    *,
    mode_map=None,
):
    """Twist one positive-m co-precessing family into J-frame strain."""

    inputs = frame.inputs
    alpha, beta, epsilon = _pnr_mode_angles(
        frame,
        frequencies,
        mode_map,
    )
    cosine = torch.cos(0.5 * beta)
    sine = torch.sin(0.5 * beta)
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


def _rotate_twisted_polarizations(inputs, plus, cross):
    """Apply the common source-frame and ascending-node rotations once."""

    cosine = math.cos(2.0 * inputs.polarization_rotation)
    sine = math.sin(2.0 * inputs.polarization_rotation)
    plus, cross = cosine * plus + sine * cross, cosine * cross - sine * plus
    cosine = math.cos(2.0 * inputs.long_asc_nodes)
    sine = math.sin(2.0 * inputs.long_asc_nodes)
    return cosine * plus + sine * cross, cosine * cross - sine * plus


def _msa_has_two_spin(msa_state):
    spin1_norm = math.sqrt(
        msa_state["chi1x"] ** 2
        + msa_state["chi1y"] ** 2
        + msa_state["chi1z"] ** 2
    )
    spin2_norm = math.sqrt(
        msa_state["chi2x"] ** 2
        + msa_state["chi2y"] ** 2
        + msa_state["chi2z"] ** 2
    )
    return spin1_norm != 0.0 and spin2_norm >= 1.0e-3


def _lpn_orbital_angular_momentum(velocity, msa_state):
    velocity2 = velocity * velocity
    velocity4 = velocity2 * velocity2
    velocity6 = velocity4 * velocity2
    velocity8 = velocity4 * velocity4
    return (
        msa_state["eta"]
        / velocity
        * (
            msa_state["L0"]
            + msa_state["L1"] * velocity
            + msa_state["L2"] * velocity2
            + msa_state["L3"] * velocity2 * velocity
            + msa_state["L4"] * velocity4
            + msa_state["L5"] * velocity4 * velocity
            + msa_state["L6"] * velocity6
            + msa_state["L7"] * velocity6 * velocity
            + msa_state["L8"] * velocity8
            + msa_state["L8L"] * velocity8 * math.log(velocity2)
        )
    )


def _pnr_interpolation_delta_f(model, f_min):
    """Return LAL's PNR angle-spline spacing in Hz."""

    inputs = model.inputs
    msa_state = model.msa_state
    eta_term = math.sqrt(max(0.0, 1.0 - 4.0 * inputs.eta))
    mf_min = f_min * inputs.total_mass_seconds
    numerator = (
        3.0
        * math.pi
        * mf_min**5
        * _PNR_INTERPOLATION_TOLERANCE
        * (1.0 + eta_term)
    )
    denominator = 7.0 + 13.0 * eta_term
    geometric_delta_f = 4.0 * math.sqrt(2.0 / 5.0) * (
        numerator / denominator
    ) ** 0.25
    delta_f = geometric_delta_f / inputs.total_mass_seconds

    if _msa_has_two_spin(msa_state):
        velocity = math.cbrt(math.pi * mf_min)
        dpsi = (
            msa_state["g0"]
            * msa_state["delta_qq"]
            * math.pi
            / (4.0 * velocity**6)
            * (
                3.0
                + 2.0 * msa_state["psi1"] * velocity
                + msa_state["psi2"] * velocity * velocity
            )
        )
        if math.isfinite(dpsi) and dpsi != 0.0:
            inverse_dpsi = abs(1.0 / dpsi)
            spin1_perp = msa_state["m1_2"] * msa_state["chi1_perp"]
            spin2_perp = msa_state["m2_2"] * msa_state["chi2_perp"]
            orbital_momentum = _lpn_orbital_angular_momentum(
                velocity,
                msa_state,
            )
            beta_min = math.atan2(
                abs(spin1_perp - spin2_perp),
                orbital_momentum + msa_state["SL"],
            )
            beta_max = math.atan2(
                spin1_perp + spin2_perp,
                orbital_momentum + msa_state["SL"],
            )
            if (
                beta_min < 0.01
                and beta_max != 0.0
                and beta_min / beta_max < 0.55
            ):
                inverse_dpsi /= 4.0
            two_spin_delta_f = (
                inverse_dpsi / 4.0 / inputs.total_mass_seconds
            )
            delta_f = min(delta_f, two_spin_delta_f)

    return max(_MINIMUM_INTERPOLATION_DELTA_F, delta_f)


def _pnr_base_transition_frequencies(model, carrier_ringdown_frequency):
    """Guard LAL's shared PNR higher-mode connection frequencies."""

    inputs = model.inputs
    beta_parameters = model.beta_parameters
    if model.spintaylor_angle_model is not None:
        beta_parameters = beta_parameters.merger
    mf_high = torch.as_tensor(
        beta_parameters.mf_lower,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    mf_low = torch.as_tensor(
        model.alpha_parameters.mf_lower,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    carrier_ringdown_frequency = torch.as_tensor(
        carrier_ringdown_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    chi_effective = (
        inputs.mass1 * inputs.chi1_l + inputs.mass2 * inputs.chi2_l
    ) / (inputs.mass1 + inputs.mass2)
    cutoff = 0.33 if chi_effective > 0.99 else IMRPhenomX_utils.fM_CUT
    mf_high = torch.where(
        (mf_high > cutoff) | (mf_high < 0.1 * carrier_ringdown_frequency),
        carrier_ringdown_frequency,
        mf_high,
    )
    mf_low = torch.where(
        (mf_low > cutoff) | (mf_high < mf_low),
        0.5 * mf_high,
        mf_low,
    )
    return mf_low, mf_high


def _pnr_higher_mode_map(model, ell, mprime, carrier_ringdown, mode_ringdown):
    """Build one guarded PNR frequency map for a co-precessing mode."""

    pnr_low, pnr_high = _pnr_base_transition_frequencies(
        model,
        carrier_ringdown,
    )
    mf_lower, mf_upper = pnr_higher_mode_transition_frequencies(
        mprime,
        pnr_low,
        pnr_high,
        carrier_ringdown,
        mode_ringdown,
    )
    return _PNRHigherModeFrequencyMap(
        ell=int(ell),
        mprime=int(mprime),
        mf_lower=mf_lower,
        mf_upper=mf_upper,
        mf_ring_22=torch.as_tensor(carrier_ringdown),
        mf_ring_lm=torch.as_tensor(mode_ringdown),
    )


def _pnr_interpolation_bounds(model, f_min, f_max, mode_maps=()):
    """Include every active higher mode's mapped endpoints in spline bounds."""

    inputs = model.inputs
    physical_bounds = torch.tensor(
        (f_min, f_max),
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    mapped_bounds = [physical_bounds]
    geometric_bounds = inputs.total_mass_seconds * physical_bounds
    for mode_map in mode_maps:
        mapped_bounds.append(
            mode_map.evaluate(geometric_bounds) / inputs.total_mass_seconds
        )
    all_bounds = torch.cat(mapped_bounds)
    return (
        float(torch.min(all_bounds).detach().cpu()),
        float(torch.max(all_bounds).detach().cpu()),
    )


def _pnr_interpolation_frequencies(model, f_min, f_max, mode_maps=()):
    f_min, f_max = _pnr_interpolation_bounds(
        model,
        f_min,
        f_max,
        mode_maps,
    )
    delta_f = _pnr_interpolation_delta_f(model, f_min)
    extended_f_min = (
        f_min / 2.0 if f_min - 2.0 * delta_f < 0.0 else f_min - 2.0 * delta_f
    )
    extended_f_max = f_max + 2.0 * delta_f
    first_bin = int(extended_f_min / delta_f)
    stop_bin = int(extended_f_max / delta_f) + 1
    return (
        torch.arange(
            first_bin,
            stop_bin,
            dtype=model.inputs.real_dtype,
            device=model.inputs.device,
        )
        * delta_f
    )


def _pnr_mode_maps(model, modes):
    """Build the higher-mode frequency maps needed by ``modes``."""

    higher_modes = tuple(mode for mode in modes if mode != (2, 2))
    if not higher_modes:
        return ()
    ringdowns = _xo4a_ringdowns(model)
    mode_ringdowns = {
        (2, 1): ringdowns.mode21,
        (3, 3): ringdowns.mode33,
        (3, 2): ringdowns.mode32,
        (4, 4): ringdowns.mode44,
    }
    return tuple(
        _pnr_higher_mode_map(
            model,
            ell,
            emm,
            ringdowns.carrier,
            mode_ringdowns[ell, emm],
        )
        for ell, emm in higher_modes
    )


def _higher_mode_ringdown(ringdowns, mode):
    """Return one native higher mode's effective ringdown pair."""

    if mode == (2, 1):
        return ringdowns.mode21, ringdowns.mode21_damping
    if mode == (3, 3):
        return ringdowns.mode33, ringdowns.mode33_damping
    if mode == (3, 2):
        return ringdowns.mode32, ringdowns.mode32_damping
    if mode == (4, 4):
        return ringdowns.mode44, ringdowns.mode44_damping
    raise ValueError(f"unsupported native XO4a higher mode {mode}")


def _twist_higher_modes(
    model,
    frequencies,
    active_f_max,
    modes,
    *,
    angle_frequencies=None,
):
    """Generate and twist the selected XO4a higher-mode families."""

    modes = tuple(mode for mode in modes if mode != (2, 2))
    if not modes:
        zeros = torch.zeros(
            frequencies.shape,
            dtype=model.inputs.complex_dtype,
            device=model.inputs.device,
        )
        return zeros, torch.zeros_like(zeros)
    ringdowns = _xo4a_ringdowns(model)
    mode_ringdowns = {
        mode: _higher_mode_ringdown(ringdowns, mode) for mode in modes
    }
    mode_maps = {
        mode: _pnr_higher_mode_map(
            model,
            *mode,
            ringdowns.carrier,
            mode_ringdowns[mode][0],
        )
        for mode in modes
    }
    if angle_frequencies is None:
        angle_frequencies = _pnr_interpolation_frequencies(
            model,
            float(frequencies[0].detach().cpu()),
            float(frequencies[-1].detach().cpu()),
            tuple(mode_maps.values()),
        )
    frame = _build_pnr_twist_frame(model, angle_frequencies)
    carrier = _xas_samples(
        model,
        frequencies,
        active_f_max,
        coprecessing_deviations=model.coprecessing_deviations,
    )
    core = _SequenceCore(carrier * _XAS_MODE_POLARIZATION_FACTOR)
    samples_by_mode = _active_mode_samples(
        core,
        _coprecessing_params(model),
        modes,
        frequencies=frequencies,
        reference_frequency=model.inputs.f_ref,
        final_spin=model.final_spin,
        ringdown_frequencies={
            mode: ringdown_pair[0]
            for mode, ringdown_pair in mode_ringdowns.items()
        },
        damping_frequencies={
            mode: ringdown_pair[1]
            for mode, ringdown_pair in mode_ringdowns.items()
        },
        carrier_ringdown_frequency=ringdowns.carrier,
        carrier_damping_frequency=ringdowns.carrier_damping,
        carrier_coprecessing_deviations=model.coprecessing_deviations,
        mode21_amplitude_release=122019,
        mode33_amplitude_release=122019,
        mode32_amplitude_release=122019,
        mode44_amplitude_release=122019,
    )
    plus = torch.zeros_like(carrier)
    cross = torch.zeros_like(carrier)
    for mode in modes:
        mode_plus, mode_cross = _twist_pnr_mode(
            frame,
            frequencies,
            samples_by_mode[mode],
            *mode,
            mode_map=mode_maps[mode],
        )
        plus += mode_plus
        cross += mode_cross
    return _rotate_twisted_polarizations(frame.inputs, plus, cross)


def _pnr_tuning_enabled(inputs):
    in_plane_spin = math.sqrt(
        inputs.spin1[0] ** 2
        + inputs.spin1[1] ** 2
        + inputs.spin2[0] ** 2
        + inputs.spin2[1] ** 2
    )
    return in_plane_spin >= _LOW_IN_PLANE_SPIN


@torch.jit.script
def _scripted_smooth_antisymmetric_ratio(
    ratio: torch.Tensor,
    geometric_frequencies: torch.Tensor,
    width: int,
) -> torch.Tensor:
    length = ratio.numel()
    if width > length - 1:
        width = length // 2
    if width <= 0 or length < width + 2:
        return ratio.clone()
    half_width = width // 2
    smoothed = ratio.clone()
    dF = geometric_frequencies[1:] - geometric_frequencies[:-1]
    denom = geometric_frequencies[width + 1:] - geometric_frequencies[:-width - 1]

    for start in range(length - width - 1):
        diff_slice = dF[start : start + width + 1]
        weighted_sum = torch.sum(smoothed[start : start + width + 1] * diff_slice)
        smoothed[start + half_width] = weighted_sum / denom[start]
    return smoothed


def _antisymmetric_amplitude_ratio(model, geometric_frequencies):
    """Return LAL's smoothed antisymmetric-to-symmetric amplitude ratio.

    ``geometric_frequencies`` contains dimensionless ``Mf`` values.  LAL
    smooths the ratio in place, so earlier replacements feed later moving
    averages; the loop below deliberately preserves that order dependence.
    """

    if geometric_frequencies.ndim != 1:
        raise ValueError("antisymmetric XO4a frequencies must be one-dimensional")
    if geometric_frequencies.numel() == 0:
        return geometric_frequencies.clone()

    inputs = model.inputs
    single_spin = model.single_spin
    eta = single_spin.symmetric_mass_ratio
    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    theta = single_spin.antisymmetric_angle
    chi = single_spin.antisymmetric_magnitude
    coefficient = (
        18.0387
        + 15.4509 * eta
        + 55.1140 * theta
        - 203.6290 * eta * theta
    )

    ringdown_frequency = _get_cutoff_fMs(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        inputs.chip,
        final_spin=model.final_spin,
        coprecessing_deviations=model.coprecessing_deviations,
    )[0]
    ringdown_frequency = torch.as_tensor(
        ringdown_frequency,
        dtype=geometric_frequencies.dtype,
        device=geometric_frequencies.device,
    )
    evaluation_frequencies = torch.minimum(
        geometric_frequencies,
        ringdown_frequency,
    )
    velocity = torch.pow(math.pi * evaluation_frequencies, 1.0 / 3.0)
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity5 = velocity3 * velocity2
    numerator = (
        21.0
        * velocity2
        * (1.0 + delta)
        * chi
        * torch.sin(theta)
    )
    denominator = 2.0 * (
        42.0
        + 84.0 * math.pi * velocity3
        + velocity2 * (55.0 * eta - 107.0)
        - 28.0
        * velocity3
        * (1.0 + delta - eta)
        * chi
        * torch.cos(theta)
    )
    ratio = numerator / denominator * (1.0 + coefficient * velocity5)

    return _scripted_smooth_antisymmetric_ratio(
        ratio,
        geometric_frequencies,
        _ANTISYMMETRIC_SMOOTHING_WIDTH,
    )


def _antisymmetric_phase(model, geometric_frequencies, alpha_offset):
    """Return the XO4a antisymmetric co-precessing (2, 2) phase.

    ``alpha_offset`` is the PNR raw-alpha reference value after the source-frame
    remapping.  The final polarization rotation is deliberately left to the
    common twist-up assembly.
    """

    if geometric_frequencies.ndim != 1:
        raise ValueError("antisymmetric XO4a frequencies must be one-dimensional")
    if geometric_frequencies.numel() == 0:
        return geometric_frequencies.clone()

    inputs = model.inputs
    intrinsic = torch.tensor(
        [inputs.mass1, inputs.mass2, inputs.chi1_l, inputs.chi2_l],
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    phase_coeffs = (
        IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
        dtype=inputs.real_dtype,
        device=inputs.device,
        )
    )
    frequencies = geometric_frequencies / inputs.total_mass_seconds
    ringdown_frequency = _get_cutoff_fMs(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        inputs.chip,
        final_spin=model.final_spin,
        coprecessing_deviations=model.coprecessing_deviations,
    )[0]
    transition_frequency = torch.as_tensor(
        _ANTISYMMETRIC_TRANSITION_FRACTION * ringdown_frequency,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )

    with torch_context(geometric_frequencies):
        linear_a, linear_b, phase_offset = _phase_alignment_terms(
            intrinsic,
            phase_coeffs,
            inputs.f_ref,
            inputs.carrier_phase,
            chip=inputs.chip,
            final_spin=model.final_spin,
            coprecessing_deviations=model.coprecessing_deviations,
        )
        symmetric_phase = Phase(
            frequencies,
            intrinsic,
            phase_coeffs,
            inputs.chip,
            final_spin=model.final_spin,
            coprecessing_deviations=model.coprecessing_deviations,
        )
        symmetric_phase = (
            symmetric_phase
            + linear_b * geometric_frequencies
            + linear_a
            + phase_offset
        )
        phase_at_transition = Phase(
            transition_frequency / inputs.total_mass_seconds,
            intrinsic,
            phase_coeffs,
            inputs.chip,
            final_spin=model.final_spin,
            coprecessing_deviations=model.coprecessing_deviations,
        )
        phase_at_transition = (
            phase_at_transition
            + linear_b * transition_frequency
            + linear_a
            + phase_offset
        )
        phase_derivative_at_transition = (
            PhaseDerivative(
                transition_frequency / inputs.total_mass_seconds,
                intrinsic,
                phase_coeffs,
                inputs.chip,
                final_spin=model.final_spin,
                coprecessing_deviations=model.coprecessing_deviations,
            )
            / inputs.total_mass_seconds
            + linear_b
        )

    derivative_step = transition_frequency.new_tensor(
        _ANTISYMMETRIC_DERIVATIVE_STEP
    )
    alpha_points = _raw_pnr_alpha(
        model,
        torch.stack(
            (
                transition_frequency - derivative_step,
                transition_frequency,
                transition_frequency + derivative_step,
            )
        ),
    )
    alpha_derivative = (alpha_points[2] - alpha_points[0]) / (
        2.0 * derivative_step
    )
    connection_slope = 0.5 * phase_derivative_at_transition - alpha_derivative
    alpha_offset = torch.as_tensor(
        alpha_offset,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    high_frequency_offset = (
        alpha_points[1]
        - 0.5 * phase_at_transition
        + connection_slope * transition_frequency
        + alpha_offset
    )
    raw_alpha = _raw_pnr_alpha(model, geometric_frequencies)
    low_frequency_phase = (
        0.5 * symmetric_phase
        + raw_alpha
        + connection_slope * geometric_frequencies
        + alpha_offset
    )
    return torch.where(
        geometric_frequencies < transition_frequency,
        low_frequency_phase,
        symmetric_phase + high_frequency_offset,
    )


def _generate_samples(
    inputs,
    inclination,
    coa_phase,
    frequencies,
    active_f_max,
    modes,
    *,
    angle_frequencies=None,
    include_antisymmetric=False,
):
    if not _pnr_tuning_enabled(inputs):
        if modes != ((2, 2),):
            raise ValueError(
                "native XO4a higher-mode generation requires tuned PNR angles"
            )
        xp_inputs = replace(inputs, final_spin_mod=4)
        return _twist_up_xp(
            _build_xp_model(xp_inputs),
            frequencies,
            active_f_max,
        )
    model = _build_model(
        inputs,
        inclination=inclination,
        coa_phase=coa_phase,
    )
    if angle_frequencies is None and any(
        mode != (2, 2) for mode in modes
    ):
        angle_frequencies = _pnr_interpolation_frequencies(
            model,
            float(frequencies[0].detach().cpu()),
            float(frequencies[-1].detach().cpu()),
            _pnr_mode_maps(model, modes),
        )
    return _twist_selected_modes(
        model,
        frequencies,
        active_f_max,
        modes,
        angle_frequencies=angle_frequencies,
        include_antisymmetric=include_antisymmetric,
    )


def _twist_up(
    model,
    frequencies,
    active_f_max,
    *,
    angle_frequencies=None,
    include_antisymmetric=False,
    antisymmetric_amplitude_ratio=None,
):
    inputs = model.inputs
    geometric_frequencies = inputs.total_mass_seconds * frequencies
    if angle_frequencies is None:
        raw_alpha = _raw_pnr_alpha(model, geometric_frequencies)
        beta = _raw_pnr_beta(model, geometric_frequencies)
        raw_gamma = pnr_gamma(geometric_frequencies, raw_alpha, beta)
        alpha_ref = _linear_value_at(
            frequencies,
            raw_alpha,
            inputs.f_ref,
        )
        beta_ref = _linear_value_at(
            frequencies,
            beta,
            inputs.f_ref,
        )
        gamma_ref = _linear_value_at(
            frequencies,
            raw_gamma,
            inputs.f_ref,
        )
    else:
        angle_splines = _build_pnr_angle_splines(model, angle_frequencies)
        raw_alpha, beta, raw_gamma = _pnr_spline_angles(
            angle_splines,
            frequencies,
        )
        reference_frequency = frequencies.new_tensor(inputs.f_ref)
        alpha_ref, beta_ref, gamma_ref = _pnr_spline_angles(
            angle_splines,
            reference_frequency,
        )
    source_frame = remap_source_frame_parameters_pnr(
        inputs.mass1,
        inputs.mass2,
        inputs.f_ref,
        model.coa_phase,
        model.inclination,
        inputs.spin1,
        inputs.spin2,
        inputs.total_mass_seconds,
        float(beta_ref.detach().cpu()),
    )
    twist_inputs = replace(
        inputs,
        theta_jn=source_frame.theta_jn,
        alpha0=source_frame.alpha0,
        polarization_rotation=source_frame.polarization_rotation,
    )
    alpha_offset = alpha_ref - source_frame.alpha_offset_shift
    alpha = raw_alpha - alpha_offset
    epsilon_offset = -gamma_ref - inputs.epsilon0
    epsilon = -raw_gamma - epsilon_offset
    cos_half = torch.cos(0.5 * beta)
    sin_half = torch.sin(0.5 * beta)
    harmonics = tuple(
        spin_weighted_spherical_harmonic(
            twist_inputs.theta_jn,
            0.0,
            -2,
            2,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        for emm in range(-2, 3)
    )
    h_phenom = _xas_samples(
        model,
        frequencies,
        active_f_max,
        coprecessing_deviations=model.coprecessing_deviations,
    )
    h_phenom_antisymmetric = None
    if include_antisymmetric:
        # LAL defines the antisymmetric co-precessing phase in the
        # waveframe, so it carries the source-frame polarization rotation
        # before both contributions receive the common plus/cross rotation.
        antisymmetric_phase_offset = (
            alpha_offset + twist_inputs.polarization_rotation
        )
        if antisymmetric_amplitude_ratio is None:
            antisymmetric_amplitude_ratio = _antisymmetric_amplitude_ratio(
                model,
                geometric_frequencies,
            )
        elif antisymmetric_amplitude_ratio.shape != frequencies.shape:
            raise ValueError(
                "antisymmetric XO4a amplitude ratio must match the "
                "frequency grid"
            )
        phase = _antisymmetric_phase(
            model,
            geometric_frequencies,
            antisymmetric_phase_offset,
        )
        h_phenom_antisymmetric = (
            torch.abs(h_phenom)
            * antisymmetric_amplitude_ratio
            * torch.exp(1j * phase)
        ).to(inputs.complex_dtype)
    return _assemble_twisted_polarizations(
        twist_inputs,
        frequencies,
        h_phenom,
        alpha,
        epsilon,
        cos_half,
        sin_half,
        harmonics,
        0.0,
        h_phenom_antisymmetric=h_phenom_antisymmetric,
    )


def _twist_selected_modes(
    model,
    frequencies,
    active_f_max,
    modes,
    *,
    angle_frequencies=None,
    include_antisymmetric=False,
    antisymmetric_amplitude_ratio=None,
):
    """Generate and sum the requested native co-precessing modes."""

    plus = torch.zeros(
        frequencies.shape,
        dtype=model.inputs.complex_dtype,
        device=model.inputs.device,
    )
    cross = torch.zeros_like(plus)
    if (2, 2) in modes:
        mode_plus, mode_cross = _twist_up(
            model,
            frequencies,
            active_f_max,
            angle_frequencies=angle_frequencies,
            include_antisymmetric=include_antisymmetric,
            antisymmetric_amplitude_ratio=antisymmetric_amplitude_ratio,
        )
        plus += mode_plus
        cross += mode_cross
    if any(mode != (2, 2) for mode in modes):
        mode_plus, mode_cross = _twist_higher_modes(
            model,
            frequencies,
            active_f_max,
            modes,
            angle_frequencies=angle_frequencies,
        )
        plus += mode_plus
        cross += mode_cross
    return plus, cross


def _imrphenomxo4a_fd_torch(
    params,
    *,
    include_antisymmetric,
    selected_modes=None,
):
    """Generate regular-grid native XO4a content."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final") or 0.0)
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXO4a frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXO4a delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXO4a f_final must be non-negative")

    modes = _generation_modes(params) if selected_modes is None else selected_modes
    inputs = _xo4a_inputs(params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXO4a cutoff) is <= f_lower")
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
        raise ValueError("IMRPhenomXO4a requires at least two active frequencies")
    plus, cross = _generate_samples(
        inputs,
        float(params.get("inclination") or 0.0),
        float(params.get("coa_phase") or 0.0),
        frequencies,
        active_f_max,
        modes,
        include_antisymmetric=include_antisymmetric,
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


def imrphenomxo4a_fd_torch(**params):
    """Generate the requested native XO4a co-precessing modes."""

    return _imrphenomxo4a_fd_torch(params, include_antisymmetric=True)


def imrphenomxo4a_symmetric22_fd_torch(**params):
    """Generate XO4a with tuned symmetric (2, 2) content only."""

    return _imrphenomxo4a_fd_torch(
        params,
        include_antisymmetric=False,
        selected_modes=((2, 2),),
    )


def _sequence_frequencies(sample_points, *, approximant="IMRPhenomXO4a"):
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError(
            f"native Torch {approximant} requires TorchScheme"
        )
    real_dtype = (
        torch.float32 if state.torch_device.type == "mps" else torch.float64
    )
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=state.torch_device,
        dtype=real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() < 2:
        raise ValueError(
            f"{approximant} sample_points must contain at least two frequencies"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError(f"{approximant} sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError(f"{approximant} sample_points must be positive")

    first_frequency = frequencies[0]
    if not bool(frequencies[-1] > first_frequency):
        raise ValueError(
            f"last {approximant} sample point must exceed the first"
        )
    if bool(torch.any(frequencies < first_frequency)):
        raise ValueError(
            f"{approximant} sample points must not lie below the first"
        )
    return frequencies


def _imrphenomxo4a_fd_sequence_torch(
    params,
    *,
    include_antisymmetric,
    selected_modes=None,
):
    """Evaluate native XO4a content at arbitrary frequencies."""

    frequencies = _sequence_frequencies(params["sample_points"])
    first_frequency = float(frequencies[0].item())
    last_frequency = float(frequencies[-1].item())
    modes = _generation_modes(params) if selected_modes is None else selected_modes
    inputs = _xo4a_inputs(
        params,
        sequence=True,
        default_reference_frequency=first_frequency,
    )
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    if cutoff_frequency <= first_frequency:
        raise ValueError(
            "the IMRPhenomXO4a cutoff must exceed the first sample point"
        )
    if _pnr_tuning_enabled(inputs) and not (
        first_frequency <= inputs.f_ref <= last_frequency
    ):
        raise ValueError(
            "IMRPhenomXO4a f_ref must lie between the first and last "
            "sample points when tuned angles are enabled"
        )

    active_f_max = min(last_frequency, cutoff_frequency)
    active = frequencies <= active_f_max
    active_frequencies = frequencies[active]
    if _pnr_tuning_enabled(inputs):
        model = _build_model(
            inputs,
            inclination=float(params.get("inclination") or 0.0),
            coa_phase=float(params.get("coa_phase") or 0.0),
        )
        angle_frequencies = _pnr_interpolation_frequencies(
            model,
            first_frequency,
            last_frequency,
            _pnr_mode_maps(model, modes),
        )
        antisymmetric_amplitude_ratio = None
        if include_antisymmetric and (2, 2) in modes:
            # LAL smooths kappa over every requested point before skipping
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
            include_antisymmetric=include_antisymmetric,
            antisymmetric_amplitude_ratio=antisymmetric_amplitude_ratio,
        )
    else:
        active_plus, active_cross = _generate_samples(
            inputs,
            float(params.get("inclination") or 0.0),
            float(params.get("coa_phase") or 0.0),
            active_frequencies,
            active_f_max,
            modes,
            include_antisymmetric=include_antisymmetric,
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


def imrphenomxo4a_fd_sequence_torch(**params):
    """Evaluate the requested native XO4a co-precessing modes."""

    return _imrphenomxo4a_fd_sequence_torch(
        params,
        include_antisymmetric=True,
    )


def imrphenomxo4a_symmetric22_fd_sequence_torch(**params):
    """Evaluate tuned symmetric-(2, 2) XO4a at arbitrary frequencies."""

    return _imrphenomxo4a_fd_sequence_torch(
        params,
        include_antisymmetric=False,
        selected_modes=((2, 2),),
    )


__all__ = [
    "imrphenomxo4a_fd_sequence_torch",
    "imrphenomxo4a_fd_torch",
    "imrphenomxo4a_native_supported",
    "imrphenomxo4a_sequence_native_supported",
    "imrphenomxo4a_symmetric22_fd_sequence_torch",
    "imrphenomxo4a_symmetric22_fd_torch",
]
