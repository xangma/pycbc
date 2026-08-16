# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native, lalsimulation-free IMRPhenomP (v1) waveforms.

The aligned-spin IMRPhenomC baseline, NNLO precession angles, small-angle
Wigner rotation, and polarization assembly run on the active Torch device.
Only scalar coefficient and time-alignment setup remain on the host.

The native path is opt-in through ``PYCBC_IMRPHENOMP_NATIVE=1`` or the global
``PYCBC_TORCH_NATIVE_PORTS=1`` switch. Unsupported options continue to use
lalsimulation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from pycbc import pnutils, scheme as _scheme
from pycbc.types import Array as PyCBCArray, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomc_torch import (
    _IMRPhenomCInputs,
    _f_rd,
    _imrphenomc_coefficients,
    _imrphenomc_components,
    _q,
)
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _MRSUN_SI,
    _MTSUN_SI,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _is_default_order,
    _is_nonzero,
    _nudge_eta,
)
from pycbc.waveform.imrphenompv2_torch import (
    _angle_series,
    _assemble_twisted_polarizations,
    _nnlo_angle_coefficients,
    _scalar_angle_series,
    _source_frame_parameters,
)


_PI = math.pi
_F_CUT = 0.15


@dataclass(frozen=True)
class _IMRPhenomPInputs:
    mass1: float
    mass2: float
    chi1_l: float
    chi2_l: float
    chi_eff: float
    chip: float
    theta_j: float
    alpha0: float
    phi_aligned: float
    polarization_rotation: float
    long_asc_nodes: float
    f_ref: float
    distance: float
    total_mass: float
    total_mass_seconds: float
    eta: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _IMRPhenomPModel:
    inputs: _IMRPhenomPInputs
    phenomc_inputs: _IMRPhenomCInputs
    phenomc_coefficients: object
    angle_coefficients: object
    alpha_offset: float
    epsilon_offset: float
    harmonics: tuple
    final_frequency: float
    time_correction: float


def _as_float(value, default=0.0):
    return float(default if value is None else value)


def _l2pnr_v1(v, eta):
    """The Kidder 2PN orbital angular momentum used by PhenomP v1."""

    v2 = v * v
    v4 = v2 * v2
    eta2 = eta * eta
    denominator = 1.0 - (3.0 - eta) * v2 / 3.0
    denominator += (4.75 + eta / 9.0) * eta * v4
    root_argument = denominator / v2
    root = (
        torch.sqrt(root_argument)
        if isinstance(root_argument, torch.Tensor)
        else math.sqrt(root_argument)
    )
    correction = (
        1.0
        + 0.5 * (1.0 - 3.0 * eta) * v2
        + 0.375 * (1.0 - 7.0 * eta + 13.0 * eta2) * v4
        + (14.0 - 41.0 * eta + 4.0 * eta2)
        * v4
        / (4.0 * denominator * denominator)
        + (3.0 + eta) * v2 / denominator
        + (7.0 - 10.0 * eta - 9.0 * eta2)
        * v4
        / (2.0 * denominator)
    )
    return eta * root * correction


def imrphenomp_native_supported(params):
    """Return whether ``params`` are covered by native IMRPhenomP."""

    if params.get("approximant", "IMRPhenomP") != "IMRPhenomP":
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    unsupported_zero = _TIDAL_EXTENSION_KEYS + _NON_GR_KEYS + (
        "lambda1",
        "lambda2",
        "eccentricity",
        "mean_per_ano",
        "frame_axis",
        "modes_choice",
        "side_bands",
    )
    if any(_is_nonzero(params.get(key, 0.0)) for key in unsupported_zero):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def _validated_inputs(
    params,
    *,
    sequence=False,
    default_reference_frequency=None,
):
    if not imrphenomp_native_supported(params):
        raise ValueError(
            "IMRPhenomP parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomP requires TorchScheme")

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    distance = pnutils.megaparsecs_to_meters(float(params["distance"]))
    inclination = _as_float(params.get("inclination"))
    coa_phase = _as_float(params.get("coa_phase"))
    long_asc_nodes = (
        0.0 if sequence else _as_float(params.get("long_asc_nodes"))
    )
    f_ref = _as_float(params.get("f_ref"))
    spin1 = tuple(_as_float(params.get(f"spin1{axis}")) for axis in "xyz")
    spin2 = tuple(_as_float(params.get(f"spin2{axis}")) for axis in "xyz")
    scalars = (
        mass1,
        mass2,
        distance,
        inclination,
        coa_phase,
        long_asc_nodes,
        f_ref,
        *spin1,
        *spin2,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("IMRPhenomP inputs must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomP component masses must be positive")
    if distance <= 0.0:
        raise ValueError("IMRPhenomP distance must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomP f_ref must be non-negative")
    if sum(value * value for value in spin1) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomP spin1 magnitude must not exceed one")
    if sum(value * value for value in spin2) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomP spin2 magnitude must not exceed one")

    reference_frequency = f_ref
    if reference_frequency == 0.0:
        reference_frequency = (
            float(params["f_lower"])
            if default_reference_frequency is None
            else float(default_reference_frequency)
        )
    if not math.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "IMRPhenomP reference frequency must be finite and positive"
        )

    (
        chi1_l,
        chi2_l,
        chip,
        theta_j,
        alpha0,
        phi_aligned,
        polarization_rotation,
    ) = _source_frame_parameters(
        mass1,
        mass2,
        reference_frequency,
        coa_phase,
        inclination,
        spin1,
        spin2,
        orbital_angular_momentum=_l2pnr_v1,
    )

    if mass1 > mass2:
        mass1, mass2 = mass2, mass1
        chi1_l, chi2_l = chi2_l, chi1_l
    total_mass = mass1 + mass2
    eta = _nudge_eta(mass1 * mass2 / (total_mass * total_mass))
    mass_ratio = mass2 / mass1
    if eta < 0.0453515:
        raise ValueError("IMRPhenomP mass ratio must not exceed 20")
    if mass_ratio > 4.0:
        warnings.warn(
            "IMRPhenomP is calibrated only for mass ratios up to 4",
            RuntimeWarning,
            stacklevel=3,
        )
    chi_eff = (mass1 * chi1_l + mass2 * chi2_l) / total_mass
    if abs(chi_eff) > 0.9:
        raise ValueError(
            "IMRPhenomP effective spin must be between -0.9 and 0.9"
        )

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _IMRPhenomPInputs(
        mass1=mass1,
        mass2=mass2,
        chi1_l=chi1_l,
        chi2_l=chi2_l,
        chi_eff=chi_eff,
        chip=chip,
        theta_j=theta_j,
        alpha0=alpha0,
        phi_aligned=phi_aligned,
        polarization_rotation=polarization_rotation,
        long_asc_nodes=long_asc_nodes,
        f_ref=reference_frequency,
        distance=distance,
        total_mass=total_mass,
        total_mass_seconds=total_mass * _MTSUN_SI,
        eta=eta,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _barausse_final_spin(eta, chi_eff, chip):
    """Barausse-Rezzolla final spin with all spin on the larger body."""

    smaller_ratio = (2.0 * eta) / (
        1.0 + math.sqrt(max(0.0, 1.0 - 4.0 * eta)) - 2.0 * eta
    )
    ratio2 = smaller_ratio * smaller_ratio
    spin2 = chi_eff * chi_eff + chip * chip
    orbital = (
        2.0 * math.sqrt(3.0)
        - 3.5171 * eta
        + 2.5763 * eta * eta
        - 0.1229 * spin2 / ((1.0 + ratio2) ** 2)
        + (0.4537 * eta - 2.8904 + 2.0)
        * chi_eff
        / (1.0 + ratio2)
    )
    final2 = (
        spin2
        + 2.0 * chi_eff * orbital * smaller_ratio
        + orbital * orbital * ratio2
    )
    return math.sqrt(max(0.0, final2)) / ((1.0 + smaller_ratio) ** 2)


def _modified_phenomc_coefficients(inputs, phenomc_inputs):
    coefficients = _imrphenomc_coefficients(phenomc_inputs)
    final_spin = min(
        1.0,
        _barausse_final_spin(inputs.eta, inputs.chi_eff, inputs.chip),
    )
    quality = _q(abs(final_spin))
    ringdown = _f_rd(abs(final_spin), inputs.total_mass)
    mf_ringdown = ringdown * inputs.total_mass_seconds
    b2 = (
        (-5.0 / 3.0) * coefficients.a1 * mf_ringdown ** (-8.0 / 3.0)
        - coefficients.a2 / (mf_ringdown * mf_ringdown)
        - (coefficients.a3 / 3.0) * mf_ringdown ** (-4.0 / 3.0)
        + (2.0 / 3.0) * coefficients.a5 * mf_ringdown ** (-1.0 / 3.0)
        + coefficients.a6
    ) / inputs.eta
    phase_at_ringdown = (
        coefficients.a1 * mf_ringdown ** (-5.0 / 3.0)
        + coefficients.a2 / mf_ringdown
        + coefficients.a3 * mf_ringdown ** (-1.0 / 3.0)
        + coefficients.a4
        + coefficients.a5 * mf_ringdown ** (2.0 / 3.0)
        + coefficients.a6 * mf_ringdown
    ) / inputs.eta
    return replace(
        coefficients,
        Q=quality,
        f_rd=ringdown,
        Mfrd=mf_ringdown,
        f1=0.1 * ringdown,
        Mf1=0.1 * mf_ringdown,
        Mf2=mf_ringdown,
        Mf0=0.98 * mf_ringdown,
        b1=phase_at_ringdown - b2 * mf_ringdown,
        b2=b2,
    )


def _angle_values(coefficients):
    alpha = (
        coefficients.alpha1,
        coefficients.alpha2,
        coefficients.alpha3,
        coefficients.alpha4,
        coefficients.alpha5,
    )
    epsilon = (
        coefficients.epsilon1,
        coefficients.epsilon2,
        coefficients.epsilon3,
        coefficients.epsilon4,
        coefficients.epsilon5,
    )
    return alpha, epsilon


def _build_model(inputs):
    # IMRPhenomC divides by this value to apply its amplitude scale. PhenomP
    # omits standalone PhenomC's additional 2*sqrt(5/(64*pi)) factor.
    amplitude_distance = inputs.distance / (
        inputs.total_mass
        * _MRSUN_SI
        * inputs.total_mass
        * _MTSUN_SI
    )
    phenomc_inputs = _IMRPhenomCInputs(
        distance=amplitude_distance,
        inclination=0.0,
        coa_phase=0.0,
        long_asc_nodes=0.0,
        total_mass=inputs.total_mass,
        eta=inputs.eta,
        xi=inputs.chi_eff,
        total_mass_seconds=inputs.total_mass_seconds,
        f_cut=_F_CUT / inputs.total_mass_seconds,
        device=inputs.device,
        real_dtype=inputs.real_dtype,
        complex_dtype=inputs.complex_dtype,
    )
    coefficients = _modified_phenomc_coefficients(inputs, phenomc_inputs)

    mass_ratio = inputs.mass2 / inputs.mass1
    chil = (1.0 + mass_ratio) * inputs.chi_eff / mass_ratio
    angles = _nnlo_angle_coefficients(mass_ratio, chil, inputs.chip)
    alpha_values, epsilon_values = _angle_values(angles)
    omega_ref = _PI * inputs.total_mass_seconds * inputs.f_ref
    alpha_offset = (
        _scalar_angle_series(omega_ref, alpha_values) - inputs.alpha0
    )
    epsilon_offset = _scalar_angle_series(omega_ref, epsilon_values)
    harmonics = tuple(
        spin_weighted_spherical_harmonic(
            inputs.theta_j,
            0.0,
            -2,
            2,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        for emm in range(-2, 3)
    )

    final_frequency = coefficients.f_rd
    fixed_frequencies = np.linspace(
        0.8 * final_frequency,
        min(1.2 * final_frequency, phenomc_inputs.f_cut),
        10,
    )
    fixed_inputs = replace(
        phenomc_inputs,
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        complex_dtype=torch.complex128,
    )
    fixed_tensor = torch.as_tensor(fixed_frequencies, dtype=torch.float64)
    _, fixed_phase = _imrphenomc_components(
        fixed_inputs,
        coefficients,
        fixed_tensor,
    )
    derivative = CubicSpline(
        fixed_frequencies,
        -fixed_phase.numpy(),
        bc_type="natural",
    )(final_frequency, 1)
    time_correction = float(derivative / (2.0 * _PI))
    return _IMRPhenomPModel(
        inputs=inputs,
        phenomc_inputs=phenomc_inputs,
        phenomc_coefficients=coefficients,
        angle_coefficients=angles,
        alpha_offset=alpha_offset,
        epsilon_offset=epsilon_offset,
        harmonics=harmonics,
        final_frequency=final_frequency,
        time_correction=time_correction,
    )


def _twist_up(model, frequencies):
    inputs = model.inputs
    amplitude, phase = _imrphenomc_components(
        model.phenomc_inputs,
        model.phenomc_coefficients,
        frequencies,
    )
    phase = phase - 2.0 * inputs.phi_aligned
    h_phenom = torch.complex(
        amplitude * torch.cos(phase),
        -amplitude * torch.sin(phase),
    ).to(inputs.complex_dtype)

    alpha_values, epsilon_values = _angle_values(model.angle_coefficients)
    omega = _PI * inputs.total_mass_seconds * frequencies
    alpha = _angle_series(omega, alpha_values) - model.alpha_offset
    epsilon = _angle_series(omega, epsilon_values) - model.epsilon_offset

    mass_ratio = inputs.mass2 / inputs.mass1
    large_fraction = mass_ratio / (1.0 + mass_ratio)
    spin_perp = inputs.chip * large_fraction * large_fraction
    spin_aligned = inputs.chi_eff * large_fraction
    velocity = torch.pow(omega, 1.0 / 3.0)
    ratio = spin_perp / (_l2pnr_v1(velocity, inputs.eta) + spin_aligned)
    denominator = 1.0 + 0.25 * ratio * ratio
    cos_half = torch.rsqrt(denominator)
    sin_half = torch.sqrt(1.0 - 1.0 / denominator)
    return _assemble_twisted_polarizations(
        inputs,
        frequencies,
        h_phenom,
        alpha,
        epsilon,
        cos_half,
        sin_half,
        model.harmonics,
        model.time_correction,
    )


def imrphenomp_fd_torch(**params):
    """Generate a regular-grid IMRPhenomP waveform with Torch."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = _as_float(params.get("f_final"))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomP frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomP delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomP f_final must be non-negative")

    inputs = _validated_inputs(params)
    f_cut = _F_CUT / inputs.total_mass_seconds
    layout_max = f_final if f_final > 0.0 else f_cut
    active_max = min(layout_max, f_cut)
    if active_max <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")
    npoints = int(2.0 ** math.ceil(math.log2(layout_max / delta_f))) + 1
    first_bin = int(f_lower / delta_f)
    stop_bin = int(active_max / delta_f)
    active_frequencies = (
        torch.arange(
            first_bin,
            stop_bin,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    model = _build_model(inputs)
    active_plus, active_cross = _twist_up(model, active_frequencies)
    plus = torch.zeros(
        npoints,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    plus[first_bin:stop_bin] = active_plus
    cross[first_bin:stop_bin] = active_cross
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


def imrphenomp_sequence_native_supported(params):
    """Return whether native IMRPhenomP covers sequence ``params``."""

    return imrphenomp_native_supported(params)


def _sequence_frequencies(sample_points):
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomP requires TorchScheme")
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
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomP sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomP sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomP sample_points must be positive")
    if frequencies.numel() > 1 and not bool(
        torch.all(frequencies[1:] > frequencies[:-1])
    ):
        raise ValueError("IMRPhenomP sample_points must be strictly increasing")
    return frequencies


def imrphenomp_fd_sequence_torch(**params):
    """Evaluate IMRPhenomP at arbitrary frequencies with Torch."""

    if not imrphenomp_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomP sequence parameters are not supported by the "
            "native Torch path"
        )
    frequencies = _sequence_frequencies(params["sample_points"])
    inputs = _validated_inputs(
        params,
        sequence=True,
        default_reference_frequency=float(frequencies[0].item()),
    )
    f_cut = _F_CUT / inputs.total_mass_seconds
    if float(frequencies[0].item()) >= f_cut:
        raise ValueError("IMRPhenomP fCut must exceed the first sample point")
    active = frequencies <= f_cut
    model = _build_model(inputs)
    active_plus, active_cross = _twist_up(model, frequencies[active])
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
    "imrphenomp_fd_sequence_torch",
    "imrphenomp_fd_torch",
    "imrphenomp_native_supported",
    "imrphenomp_sequence_native_supported",
]
