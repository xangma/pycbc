# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native IMRPhenomXP with the NNLO-v102 precession prescription.

The aligned-spin IMRPhenomXAS carrier and all frequency-dependent precession
angles, Wigner rotations, and polarization assembly execute on the active
Torch device. Scalar source-frame setup remains on the host.

This intentionally covers only the explicit LAL configuration
``PhenomXPrecVersion=102``, ``PhenomXPConvention=0``, and
``PhenomXPFinalSpinMod=0``. The public path is opt-in through
``PYCBC_IMRPHENOMXP_NATIVE=1`` or ``PYCBC_TORCH_NATIVE_PORTS=1``; the default
MSA-angle model and all other configurations continue to use lalsimulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomd_torch import (
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
)
from pycbc.waveform.imrphenompv2_torch import (
    _angle_series,
    _assemble_twisted_polarizations,
    _atan2tol,
    _nnlo_angle_coefficients,
    _rotate_y,
    _rotate_z,
    _scalar_angle_series,
)
from pycbc.waveform.imrphenomxas_torch import (
    MTSUN,
    _DEFAULT_ONLY_ORDER_KEYS,
    _XAS_MODE_POLARIZATION_FACTOR,
    _gen_IMRPhenomXAS,
    _is_default_order,
    _is_nonzero,
    _next_power_of_two,
)
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils
from pycbc.waveform._torch_jax import torch_context


_PI = math.pi
_SUPPORTED_PREC_VERSION = 102
_SUPPORTED_CONVENTION = 0
_SUPPORTED_FINAL_SPIN = 0


@dataclass(frozen=True)
class _IMRPhenomXPInputs:
    mass1: float
    mass2: float
    chi1_l: float
    chi2_l: float
    chip: float
    spin_aligned: float
    spin_perp: float
    theta_jn: float
    alpha0: float
    distance: float
    phi_aligned: float
    polarization_rotation: float
    long_asc_nodes: float
    f_ref: float
    total_mass: float
    total_mass_seconds: float
    eta: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _XPModel:
    inputs: _IMRPhenomXPInputs
    angle_coeffs: object
    alpha_offset: float
    epsilon_offset: float
    harmonics: tuple


def _as_float(value, default=0.0):
    return float(default if value is None else value)


def _is_exact_integer(value, expected):
    if value is None:
        return False
    try:
        number = float(value)
        return math.isfinite(number) and number == expected and int(number) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def imrphenomxp_native_supported(params):
    """Return whether ``params`` select the bounded native XP model."""

    if params.get("approximant", "IMRPhenomXP") != "IMRPhenomXP":
        return False
    flags = (
        ("phenom_x_prec_version", _SUPPORTED_PREC_VERSION),
        ("phenom_xp_convention", _SUPPORTED_CONVENTION),
        ("phenom_xp_final_spin_mod", _SUPPORTED_FINAL_SPIN),
    )
    if any(not _is_exact_integer(params.get(key), expected) for key, expected in flags):
        return False
    if any(
        not _is_default_order(params.get(key, -1)) for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    unsupported_zero = (
        _TIDAL_EXTENSION_KEYS
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
    if any(_is_nonzero(params.get(key, 0.0)) for key in unsupported_zero):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def imrphenomxp_sequence_native_supported(params):
    """Return whether arbitrary-frequency XP generation is native."""

    return imrphenomxp_native_supported(params)


def _lpn_v102(v, eta, chi1_l, chi2_l):
    """3PN orbital angular momentum used by XP precession version 102."""

    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    l2 = 1.5 + eta / 6.0
    l3 = (
        5.0
        * (chi1_l * (-2.0 - 2.0 * delta + eta) + chi2_l * (-2.0 + 2.0 * delta + eta))
        / 6.0
    )
    l4 = (81.0 + (-57.0 + eta) * eta) / 24.0
    l5 = (
        -7.0
        * (
            chi1_l * (72.0 + delta * (72.0 - 31.0 * eta) + eta * (-121.0 + 2.0 * eta))
            + chi2_l
            * (72.0 + eta * (-121.0 + 2.0 * eta) + delta * (-72.0 + 31.0 * eta))
        )
        / 144.0
    )
    l6 = (
        10935.0 + eta * (-62001.0 + eta * (1674.0 + 7.0 * eta) + 2214.0 * _PI * _PI)
    ) / 1296.0
    v2 = v * v
    polynomial = 1.0 + l2 * v2 + l3 * v2 * v + l4 * v2 * v2
    polynomial += l5 * v2 * v2 * v + l6 * v2 * v2 * v2
    return eta * polynomial / v


def _source_frame_parameters(
    mass1,
    mass2,
    f_ref,
    coa_phase,
    inclination,
    spin1,
    spin2,
):
    """Map LAL source-frame inputs to convention-0 XP parameters."""

    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    fraction1_sq = fraction1 * fraction1
    fraction2_sq = fraction2 * fraction2
    eta = fraction1 * fraction2
    chi1_l = spin1[2]
    chi2_l = spin2[2]

    spin1_perp = fraction1_sq * math.hypot(spin1[0], spin1[1])
    spin2_perp = fraction2_sq * math.hypot(spin2[0], spin2[1])
    a1 = 2.0 + 1.5 * fraction2 / fraction1
    a2 = 2.0 + 1.5 * fraction1 / fraction2
    chip = max(a1 * spin1_perp, a2 * spin2_perp) / (a1 * fraction1_sq)
    spin_aligned = chi1_l * fraction1_sq + chi2_l * fraction2_sq
    spin_perp = chip * fraction1_sq

    velocity = (_PI * total_mass * MTSUN * f_ref) ** (1.0 / 3.0)
    orbital_momentum = _lpn_v102(velocity, eta, chi1_l, chi2_l)
    jx = fraction1_sq * spin1[0] + fraction2_sq * spin2[0]
    jy = fraction1_sq * spin1[1] + fraction2_sq * spin2[1]
    jz = spin_aligned + orbital_momentum
    jmag = math.sqrt(jx * jx + jy * jy + jz * jz)
    theta_j_source = (
        0.0 if jmag < 1.0e-10 else math.acos(max(-1.0, min(1.0, jz / jmag)))
    )
    if abs(jx) < 1.0e-15 and abs(jy) < 1.0e-15:
        phi_j_source = _PI / 2.0 - coa_phase
    else:
        phi_j_source = math.atan2(jy, jx)
    phi_aligned = -phi_j_source

    line_of_sight = (
        math.sin(inclination) * math.sin(coa_phase),
        math.sin(inclination) * math.cos(coa_phase),
        math.cos(inclination),
    )

    def rotate_to_intermediate(vector):
        return _rotate_y(-theta_j_source, _rotate_z(-phi_j_source, vector))

    rotated_sight = rotate_to_intermediate(line_of_sight)
    kappa = _atan2tol(rotated_sight[1], rotated_sight[0])

    def rotate_to_j_frame(vector):
        return _rotate_z(-kappa, rotate_to_intermediate(vector))

    rotated_orbit = rotate_to_j_frame((0.0, 0.0, 1.0))
    if abs(rotated_orbit[0]) < 1.0e-15 and abs(rotated_orbit[1]) < 1.0e-15:
        alpha0 = _PI
    else:
        alpha0 = math.atan2(rotated_orbit[1], rotated_orbit[0])

    rotated_sight = rotate_to_j_frame(line_of_sight)
    theta_jn = math.acos(max(-1.0, min(1.0, rotated_sight[2])))
    waveframe_x = (
        -math.cos(inclination) * math.sin(coa_phase),
        -math.cos(inclination) * math.cos(coa_phase),
        math.sin(inclination),
    )
    rotated_x = rotate_to_j_frame(waveframe_x)
    x_dot_p = -rotated_x[1]
    x_dot_q = rotated_x[0] * rotated_sight[2] - rotated_x[2] * rotated_sight[0]
    polarization_rotation = math.atan2(x_dot_q, x_dot_p)
    return (
        chi1_l,
        chi2_l,
        chip,
        spin_aligned,
        spin_perp,
        theta_jn,
        alpha0,
        phi_aligned,
        polarization_rotation,
    )


def _validated_inputs(
    params,
    *,
    sequence=False,
    default_reference_frequency=None,
):
    if not imrphenomxp_native_supported(params):
        raise ValueError(
            "IMRPhenomXP parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXP requires TorchScheme")

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    distance = float(params["distance"])
    inclination = _as_float(params.get("inclination"))
    coa_phase = _as_float(params.get("coa_phase"))
    long_asc_nodes = 0.0 if sequence else _as_float(params.get("long_asc_nodes"))
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
        raise ValueError("IMRPhenomXP inputs must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomXP component masses must be positive")
    if distance <= 0.0:
        raise ValueError("IMRPhenomXP distance must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomXP f_ref must be non-negative")
    if sum(value * value for value in spin1) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomXP spin1 magnitude must not exceed one")
    if sum(value * value for value in spin2) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomXP spin2 magnitude must not exceed one")

    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1
    if mass1 / mass2 > 1000.0 + 1.0e-12:
        raise ValueError("IMRPhenomXP is not valid beyond mass ratio 1000")

    reference_frequency = f_ref
    if reference_frequency == 0.0:
        reference_frequency = (
            float(params["f_lower"])
            if default_reference_frequency is None
            else float(default_reference_frequency)
        )
    if not math.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError("IMRPhenomXP reference frequency must be finite and positive")

    (
        chi1_l,
        chi2_l,
        chip,
        spin_aligned,
        spin_perp,
        theta_jn,
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
    )
    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return _IMRPhenomXPInputs(
        mass1=mass1,
        mass2=mass2,
        chi1_l=chi1_l,
        chi2_l=chi2_l,
        chip=chip,
        spin_aligned=spin_aligned,
        spin_perp=spin_perp,
        theta_jn=theta_jn,
        alpha0=alpha0,
        distance=distance,
        phi_aligned=phi_aligned,
        polarization_rotation=polarization_rotation,
        long_asc_nodes=long_asc_nodes,
        f_ref=reference_frequency,
        total_mass=total_mass,
        total_mass_seconds=total_mass * MTSUN,
        eta=eta,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _build_model(inputs):
    q = inputs.mass1 / inputs.mass2
    chi_eff = (
        inputs.mass1 * inputs.chi1_l + inputs.mass2 * inputs.chi2_l
    ) / inputs.total_mass
    chil = (1.0 + q) * chi_eff / q
    angle_coeffs = _nnlo_angle_coefficients(q, chil, inputs.chip)
    alpha_values = (
        angle_coeffs.alpha1,
        angle_coeffs.alpha2,
        angle_coeffs.alpha3,
        angle_coeffs.alpha4,
        angle_coeffs.alpha5,
    )
    epsilon_values = (
        angle_coeffs.epsilon1,
        angle_coeffs.epsilon2,
        angle_coeffs.epsilon3,
        angle_coeffs.epsilon4,
        angle_coeffs.epsilon5,
    )
    omega_ref = _PI * inputs.total_mass_seconds * inputs.f_ref
    alpha_offset = _scalar_angle_series(omega_ref, alpha_values) - inputs.alpha0
    epsilon_offset = _scalar_angle_series(omega_ref, epsilon_values)
    harmonics = tuple(
        spin_weighted_spherical_harmonic(
            inputs.theta_jn,
            0.0,
            -2,
            2,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        for emm in range(-2, 3)
    )
    return _XPModel(
        inputs,
        angle_coeffs,
        alpha_offset,
        epsilon_offset,
        harmonics,
    )


def _xas_samples(inputs, frequencies):
    intrinsic = torch.tensor(
        [inputs.mass1, inputs.mass2, inputs.chi1_l, inputs.chi2_l],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    extrinsic = torch.tensor(
        [inputs.distance, 0.0, inputs.phi_aligned],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table.to(
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table.to(
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    with torch_context(frequencies):
        samples = _gen_IMRPhenomXAS(
            frequencies,
            intrinsic,
            extrinsic,
            phase_coeffs,
            amp_coeffs,
            inputs.f_ref,
            chip=inputs.chip,
        )
    return samples.to(inputs.complex_dtype) / _XAS_MODE_POLARIZATION_FACTOR


def _twist_up(model, frequencies):
    inputs = model.inputs
    h_phenom = _xas_samples(inputs, frequencies)
    mf = inputs.total_mass_seconds * frequencies
    omega = _PI * mf
    alpha_values = (
        model.angle_coeffs.alpha1,
        model.angle_coeffs.alpha2,
        model.angle_coeffs.alpha3,
        model.angle_coeffs.alpha4,
        model.angle_coeffs.alpha5,
    )
    epsilon_values = (
        model.angle_coeffs.epsilon1,
        model.angle_coeffs.epsilon2,
        model.angle_coeffs.epsilon3,
        model.angle_coeffs.epsilon4,
        model.angle_coeffs.epsilon5,
    )
    alpha = _angle_series(omega, alpha_values) - model.alpha_offset
    epsilon = _angle_series(omega, epsilon_values) - model.epsilon_offset

    velocity = torch.pow(omega, 1.0 / 3.0)
    orbital_momentum = _lpn_v102(
        velocity,
        inputs.eta,
        inputs.chi1_l,
        inputs.chi2_l,
    )
    denominator = orbital_momentum + inputs.spin_aligned
    ratio = inputs.spin_perp / denominator
    sign = torch.where(
        denominator >= 0.0,
        torch.ones_like(denominator),
        -torch.ones_like(denominator),
    )
    cos_beta = sign * torch.rsqrt(1.0 + ratio * ratio)
    cos_half = torch.sqrt(torch.abs(0.5 * (1.0 + cos_beta)))
    sin_half = torch.sqrt(torch.abs(0.5 * (1.0 - cos_beta)))
    return _assemble_twisted_polarizations(
        inputs,
        frequencies,
        h_phenom,
        alpha,
        epsilon,
        cos_half,
        sin_half,
        model.harmonics,
        0.0,
    )


def _series_from_active_samples(inputs, samples, npoints, first_bin, stop_bin, delta_f):
    data = torch.zeros(
        npoints,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    data[first_bin:stop_bin] = samples
    return FrequencySeries(
        TorchArrayData(data),
        delta_f=delta_f,
        epoch=-1.0 / delta_f,
        copy=False,
    )


def imrphenomxp_fd_torch(**params):
    """Generate a regular-grid NNLO-v102 IMRPhenomXP waveform with Torch."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = _as_float(params.get("f_final"))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXP frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXP delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXP f_final must be non-negative")

    inputs = _validated_inputs(params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXP cutoff) is <= f_lower")
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
    plus, cross = _twist_up(model, frequencies)
    return (
        _series_from_active_samples(
            inputs, plus, npoints, first_bin, stop_bin, delta_f
        ),
        _series_from_active_samples(
            inputs, cross, npoints, first_bin, stop_bin, delta_f
        ),
    )


def _sequence_frequencies(sample_points):
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXP requires TorchScheme")
    real_dtype = torch.float32 if state.torch_device.type == "mps" else torch.float64
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=state.torch_device,
        dtype=real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomXP sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomXP sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomXP sample_points must be positive")
    return frequencies


def imrphenomxp_fd_sequence_torch(**params):
    """Evaluate NNLO-v102 IMRPhenomXP at arbitrary frequencies with Torch."""

    if not imrphenomxp_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomXP sequence parameters are not supported by the native Torch path"
        )
    frequencies = _sequence_frequencies(params["sample_points"])
    inputs = _validated_inputs(
        params,
        sequence=True,
        default_reference_frequency=float(frequencies[0].item()),
    )
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    active_f_max = torch.minimum(
        frequencies[-1],
        frequencies.new_tensor(cutoff_frequency),
    )
    active = frequencies <= active_f_max
    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    if bool(torch.any(active)):
        model = _build_model(inputs)
        plus[active], cross[active] = _twist_up(model, frequencies[active])
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "imrphenomxp_fd_sequence_torch",
    "imrphenomxp_fd_torch",
    "imrphenomxp_native_supported",
    "imrphenomxp_sequence_native_supported",
]
