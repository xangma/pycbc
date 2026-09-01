# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native IMRPhenomPv3 and IMRPhenomPv3HM waveforms.

The aligned-spin carrier is shared with :mod:`imrphenomhm_torch`.  Source-frame
setup remains scalar, while MSA angle evolution, Wigner rotations, mode
assembly, and polarization synthesis run on the active Torch device.  Both
regular frequency grids and strictly increasing arbitrary-frequency sequences
are supported without calling lalsimulation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pycbc import lal_compat as lal
import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._mode_rotation_torch import wigner_d_from_cosbeta
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _final_spin0815,
    _is_default_order,
    _is_nonzero,
)
from pycbc.waveform.imrphenomhm_torch import (
    _DEFAULT_MF_MAX,
    _active_modes,
    _imrphenomhm_inputs,
    _imrphenomhm_mode_samples,
    _imrphenomhm_model,
    _imrphenomhm_polarizations,
    _sequence_frequencies,
)
from pycbc.waveform.imrphenompv2_torch import phenomp_source_frame_parameters
from pycbc.waveform.imrphenomxp_msa_torch import (
    build_msa_state,
    msa_angles,
    orbital_angular_momentum_3pn,
)


_UNSUPPORTED_ZERO_KEYS = (
    *_TIDAL_EXTENSION_KEYS,
    *_NON_GR_KEYS,
    "lambda1",
    "lambda2",
    "eccentricity",
    "mean_per_ano",
    "frame_axis",
    "modes_choice",
    "side_bands",
    "nl_tides_a1",
    "nl_tides_a2",
    "nl_tides_n1",
    "nl_tides_n2",
    "nl_tides_f1",
    "nl_tides_f2",
)
_UNSUPPORTED_NONDEFAULT_KEYS = (
    "phenom_x_prec_version",
    "phenom_xp_convention",
    "phenom_xp_final_spin_mod",
)


class _IMRPhenomPv3Model(NamedTuple):
    """Frequency-independent state for Pv3/Pv3HM polarization synthesis."""

    carrier: object
    precessing: bool
    msa_state: dict | None
    theta_jn: float
    alpha0: float
    polarization_rotation: float


def _ordered_source_parameters(params):
    """Return mass-ordered source-frame scalars and Cartesian spins."""

    try:
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        spin1 = tuple(float(params.get(f"spin1{axis}", 0.0)) for axis in "xyz")
        spin2 = tuple(float(params.get(f"spin2{axis}", 0.0)) for axis in "xyz")
        inclination = float(params.get("inclination", 0.0))
        coa_phase = float(params.get("coa_phase", 0.0))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("IMRPhenomPv3 source parameters must be numeric") from exc

    scalars = (mass1, mass2, inclination, coa_phase, *spin1, *spin2)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("IMRPhenomPv3 source parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomPv3 component masses must be positive")
    if math.sqrt(sum(component * component for component in spin1)) > 1.0:
        raise ValueError("IMRPhenomPv3 spin1 magnitude must not exceed one")
    if math.sqrt(sum(component * component for component in spin2)) > 1.0:
        raise ValueError("IMRPhenomPv3 spin2 magnitude must not exceed one")

    # Match LAL's source-ordering tie-break: the second body wins when the
    # component masses are exactly equal.
    if mass2 >= mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1
    return mass1, mass2, spin1, spin2, inclination, coa_phase


def _active_pv3_modes(params):
    """Return the co-precessing modes selected by a Pv3-family request."""

    approximant = params.get("approximant", "IMRPhenomPv3HM")
    if approximant == "IMRPhenomPv3":
        if params.get("mode_array") is not None:
            raise ValueError("IMRPhenomPv3 does not accept a mode array")
        return ((2, 2),)
    if approximant != "IMRPhenomPv3HM":
        raise ValueError("expected IMRPhenomPv3 or IMRPhenomPv3HM")

    active_modes = _active_modes(params.get("mode_array"))
    if active_modes is None or (2, 2) not in active_modes:
        raise ValueError("IMRPhenomPv3HM requires the (2, 2) mode")
    return active_modes


def _native_features_supported(params, approximant):
    """Return whether scalar and model options are covered by this port."""

    try:
        if params.get("approximant", approximant) != approximant:
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
            for key in _UNSUPPORTED_ZERO_KEYS
        ):
            return False
        if any(
            params.get(key) is not None
            for key in _UNSUPPORTED_NONDEFAULT_KEYS
        ):
            return False
        if params.get("numrel_data", ""):
            return False

        model_params = dict(params)
        model_params["approximant"] = approximant
        _active_pv3_modes(model_params)
        _ordered_source_parameters(params)
        distance = float(params["distance"])
        f_ref = float(params.get("f_ref", 0.0))
    except (KeyError, TypeError, ValueError, OverflowError, RuntimeError):
        return False
    return (
        math.isfinite(distance)
        and distance > 0.0
        and math.isfinite(f_ref)
        and f_ref >= 0.0
    )


def _regular_sampling_supported(params):
    """Validate the regular-grid controls used by the public wrapper."""

    try:
        delta_f = float(params["delta_f"])
        f_lower = float(params["f_lower"])
        f_final = float(params.get("f_final", 0.0))
        long_asc_nodes = float(params.get("long_asc_nodes", 0.0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(
        math.isfinite(value)
        for value in (delta_f, f_lower, f_final, long_asc_nodes)
    ):
        return False
    return (
        delta_f > 0.0
        and f_lower > 0.0
        and f_final >= 0.0
        and (f_final == 0.0 or f_final >= f_lower)
    )


def _sequence_sampling_supported(params):
    """Validate arbitrary frequencies without moving them off their device."""

    try:
        values = getattr(params["sample_points"], "_data", params["sample_points"])
        if isinstance(values, TorchArrayData):
            values = values.tensor
        frequencies = torch.as_tensor(values)
        if frequencies.ndim != 1 or frequencies.numel() == 0:
            return False
        if not bool(torch.all(torch.isfinite(frequencies))):
            return False
        if bool(torch.any(frequencies <= 0.0)):
            return False
        return frequencies.numel() == 1 or not bool(
            torch.any(frequencies[1:] <= frequencies[:-1])
        )
    except (KeyError, TypeError, ValueError, OverflowError, RuntimeError):
        return False


def _native_device_supported(params, *, sequence):
    """Keep MPS on the exact aligned shortcut pending float32 validation."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    ):
        return True

    try:
        transverse_spins = tuple(
            float(params.get(f"spin{body}{axis}", 0.0))
            for body in (1, 2)
            for axis in "xy"
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return not any(transverse_spins)


def _native_supported(params, approximant, *, sequence):
    if not _native_features_supported(params, approximant):
        return False
    sampling_supported = (
        _sequence_sampling_supported(params)
        if sequence
        else _regular_sampling_supported(params)
    )
    return sampling_supported and _native_device_supported(
        params,
        sequence=sequence,
    )


def imrphenompv3_native_supported(params) -> bool:
    """Return whether regular-grid Pv3 generation is native."""

    return _native_supported(params, "IMRPhenomPv3", sequence=False)


def imrphenompv3_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency Pv3 generation is native."""

    return _native_supported(params, "IMRPhenomPv3", sequence=True)


def imrphenompv3hm_native_supported(params) -> bool:
    """Return whether regular-grid Pv3HM generation is native."""

    return _native_supported(params, "IMRPhenomPv3HM", sequence=False)


def imrphenompv3hm_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency Pv3HM generation is native."""

    return _native_supported(params, "IMRPhenomPv3HM", sequence=True)


def _precessing_carrier_final_spin(mass1, mass2, spin1z, spin2z, chip):
    """Return the in-plane-spin correction used by PhenomHM's carrier."""

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    aligned = _final_spin0815(eta, spin1z, spin2z)
    primary_fraction = max(mass1, mass2) / total_mass
    in_plane = chip * primary_fraction * primary_fraction
    return math.copysign(math.hypot(aligned, in_plane), aligned)


def _imrphenompv3hm_model(params, reference_frequency_hz, *, sequence=False):
    """Build a private Pv3/Pv3HM model without calling lalsimulation."""

    try:
        reference_frequency_hz = float(reference_frequency_hz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("IMRPhenomPv3 reference frequency must be numeric") from exc
    if not math.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0.0:
        raise ValueError("IMRPhenomPv3 reference frequency must be positive")

    active_modes = _active_pv3_modes(params)
    mass1, mass2, spin1, spin2, inclination, coa_phase = (
        _ordered_source_parameters(params)
    )
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    precessing = any(
        component != 0.0 for component in (*spin1[:2], *spin2[:2])
    )

    msa_state = None
    if precessing:
        msa_state = build_msa_state(
            mass1,
            mass2,
            spin1,
            spin2,
            total_mass_seconds,
            reference_frequency_hz,
        )

    source_kwargs = {}
    if msa_state is not None:

        def spinning_orbital_momentum(velocity, _eta):
            return orbital_angular_momentum_3pn(velocity, msa_state)

        source_kwargs["orbital_angular_momentum"] = spinning_orbital_momentum
    (
        _chi1_l,
        _chi2_l,
        chip,
        theta_jn,
        alpha0,
        _phi_aligned,
        source_rotation,
    ) = phenomp_source_frame_parameters(
        mass1,
        mass2,
        reference_frequency_hz,
        coa_phase,
        inclination,
        spin1,
        spin2,
        **source_kwargs,
    )

    carrier_params = dict(params)
    carrier_params.update(
        approximant="IMRPhenomHM",
        mass1=mass1,
        mass2=mass2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=spin1[2],
        spin2x=0.0,
        spin2y=0.0,
        spin2z=spin2[2],
        f_ref=reference_frequency_hz,
        inclination=theta_jn,
        coa_phase=coa_phase,
        long_asc_nodes=0.0,
        mode_array=active_modes,
    )
    carrier_inputs = _imrphenomhm_inputs(carrier_params, sequence=sequence)
    final_spin = _precessing_carrier_final_spin(
        mass1,
        mass2,
        spin1[2],
        spin2[2],
        chip,
    )
    carrier = _imrphenomhm_model(
        carrier_inputs,
        reference_frequency_hz,
        final_spin=final_spin,
    )
    try:
        ascending_node_rotation = (
            0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("IMRPhenomPv3 long_asc_nodes must be numeric") from exc
    if not math.isfinite(ascending_node_rotation):
        raise ValueError("IMRPhenomPv3 long_asc_nodes must be finite")
    return _IMRPhenomPv3Model(
        carrier=carrier,
        precessing=precessing,
        msa_state=msa_state,
        theta_jn=theta_jn,
        alpha0=alpha0,
        polarization_rotation=source_rotation + ascending_node_rotation,
    )


def _twist_mode(model, frequencies_hz, mode):
    """Twist one positive-m co-precessing carrier into polarizations."""

    ell, mprime = mode
    carrier = model.carrier
    inputs = carrier.inputs
    dimensionless_frequency = frequencies_hz * inputs.total_mass_seconds
    hlm = _imrphenomhm_mode_samples(carrier, dimensionless_frequency, mode)

    velocity = torch.pow(
        2.0
        * math.pi
        * inputs.total_mass_seconds
        * frequencies_hz
        / mprime,
        1.0 / 3.0,
    )
    phiz, epsilon, cos_beta = msa_angles(velocity, model.msa_state)
    alpha = phiz + model.alpha0
    d_positive, d_negative = wigner_d_from_cosbeta(ell, mprime, cos_beta)

    emm = torch.arange(
        -ell,
        ell + 1,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    alpha_phase = torch.exp(1j * emm[:, None] * alpha).to(inputs.complex_dtype)
    harmonics = torch.stack(
        [
            spin_weighted_spherical_harmonic(
                model.theta_jn,
                0.0,
                -2,
                ell,
                target_m,
                dtype=inputs.real_dtype,
                device=inputs.device,
            )
            for target_m in range(-ell, ell + 1)
        ]
    )
    term1 = torch.sum(
        harmonics[:, None] * alpha_phase * d_positive,
        dim=0,
    )
    term2 = (-1) ** ell * torch.sum(
        harmonics.conj()[:, None] * alpha_phase.conj() * d_negative,
        dim=0,
    )
    half_mode = 0.5 * hlm * torch.exp(-1j * mprime * epsilon)
    return half_mode * (term1 + term2), -1j * half_mode * (term1 - term2)


def _rotate_polarizations(plus, cross, angle):
    """Apply LAL's source-polarization convention."""

    cosine = math.cos(2.0 * angle)
    sine = math.sin(2.0 * angle)
    return cosine * plus + sine * cross, cosine * cross - sine * plus


def _imrphenompv3hm_polarizations(model, frequencies_hz):
    """Evaluate a private Pv3/Pv3HM model at positive frequencies in Hz."""

    inputs = model.carrier.inputs
    frequencies_hz = torch.as_tensor(
        frequencies_hz,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies_hz.ndim != 1 or frequencies_hz.numel() == 0:
        raise ValueError("IMRPhenomPv3 frequencies must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies_hz))):
        raise ValueError("IMRPhenomPv3 frequencies must be finite")
    if bool(torch.any(frequencies_hz <= 0.0)):
        raise ValueError("IMRPhenomPv3 frequencies must be positive")

    if not model.precessing:
        plus, cross = _imrphenomhm_polarizations(
            model.carrier,
            frequencies_hz * inputs.total_mass_seconds,
        )
        return _rotate_polarizations(
            plus,
            cross,
            model.polarization_rotation,
        )

    plus = torch.zeros(
        frequencies_hz.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    for mode in inputs.active_modes:
        mode_plus, mode_cross = _twist_mode(model, frequencies_hz, mode)
        plus += mode_plus
        cross += mode_cross

    amplitude_scale = (
        inputs.total_mass
        * lal.MRSUN_SI
        * inputs.total_mass_seconds
        / inputs.distance
    )
    plus *= amplitude_scale
    cross *= amplitude_scale
    return _rotate_polarizations(
        plus,
        cross,
        model.polarization_rotation,
    )


def _regular_waveform(params, supported):
    """Generate one Pv3-family waveform on LAL's regular FD layout."""

    if not supported(params):
        raise ValueError(
            f"{params.get('approximant')} parameters are not supported by "
            "the native Torch path"
        )

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    reference_frequency_hz = float(params.get("f_ref", 0.0)) or f_lower
    model = _imrphenompv3hm_model(
        params,
        reference_frequency_hz,
        sequence=False,
    )
    inputs = model.carrier.inputs
    layout_f_max = (
        f_final if f_final > 0.0 else _DEFAULT_MF_MAX / inputs.total_mass_seconds
    )

    frequency_bins = int(layout_f_max / delta_f)
    npts = 1
    if frequency_bins:
        npts += 1 << (frequency_bins - 1).bit_length()
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = min(math.ceil(layout_f_max / delta_f), npts)
    bins = torch.arange(first_bin, stop_bin, device=inputs.device)

    plus = torch.zeros(
        npts,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    if bins.numel():
        frequencies_hz = bins.to(dtype=inputs.real_dtype) * delta_f
        active_plus, active_cross = _imrphenompv3hm_polarizations(
            model,
            frequencies_hz,
        )
        plus[bins] = active_plus
        cross[bins] = active_cross

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _sequence_waveform(params, supported):
    """Evaluate one Pv3-family waveform at arbitrary frequencies."""

    if not supported(params):
        raise ValueError(
            f"{params.get('approximant')} parameters are not supported by "
            "the native Torch path"
        )

    initial_reference = float(params.get("f_ref", 0.0))
    if initial_reference == 0.0:
        values = getattr(params["sample_points"], "_data", params["sample_points"])
        if isinstance(values, TorchArrayData):
            values = values.tensor
        initial_reference = float(torch.as_tensor(values)[0].item())
    model = _imrphenompv3hm_model(
        params,
        initial_reference,
        sequence=True,
    )
    inputs = model.carrier.inputs
    frequencies_hz = _sequence_frequencies(params["sample_points"], inputs)
    plus, cross = _imrphenompv3hm_polarizations(model, frequencies_hz)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def imrphenompv3_fd_torch(**params):
    """Generate IMRPhenomPv3 plus/cross polarizations with Torch."""

    params.setdefault("approximant", "IMRPhenomPv3")
    return _regular_waveform(params, imrphenompv3_native_supported)


def imrphenompv3_fd_sequence_torch(**params):
    """Evaluate IMRPhenomPv3 at arbitrary increasing frequencies."""

    params.setdefault("approximant", "IMRPhenomPv3")
    return _sequence_waveform(params, imrphenompv3_sequence_native_supported)


def imrphenompv3hm_fd_torch(**params):
    """Generate IMRPhenomPv3HM plus/cross polarizations with Torch."""

    params.setdefault("approximant", "IMRPhenomPv3HM")
    return _regular_waveform(params, imrphenompv3hm_native_supported)


def imrphenompv3hm_fd_sequence_torch(**params):
    """Evaluate IMRPhenomPv3HM at arbitrary increasing frequencies."""

    params.setdefault("approximant", "IMRPhenomPv3HM")
    return _sequence_waveform(params, imrphenompv3hm_sequence_native_supported)


__all__ = (
    "imrphenompv3_fd_sequence_torch",
    "imrphenompv3_fd_torch",
    "imrphenompv3_native_supported",
    "imrphenompv3_sequence_native_supported",
    "imrphenompv3hm_fd_sequence_torch",
    "imrphenompv3hm_fd_torch",
    "imrphenompv3hm_native_supported",
    "imrphenompv3hm_sequence_native_supported",
)
