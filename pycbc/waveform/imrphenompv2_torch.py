# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native, lalsimulation-free IMRPhenomPv2 waveforms.

This ports the BBH ``IMRPhenomPv2`` implementation in
``LALSimIMRPhenomP.c``. Scalar model setup remains on the host, while the
frequency-dependent PhenomD baseline, NNLO precession angles, Wigner rotation,
and polarization assembly execute on the active Torch device.

The native path is opt-in through ``PYCBC_IMRPHENOMPV2_NATIVE=1`` or the
global ``PYCBC_TORCH_NATIVE_PORTS=1`` switch. Unsupported options continue to
fall back to lalsimulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from pycbc import pnutils, scheme as _scheme
from pycbc.types import Array as PyCBCArray, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _IMRPhenDAmplitude,
    _IMRPhenDPhase,
    _MRSUN_SI,
    _MTSUN_SI,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _compute_amp_coeffs,
    _compute_phase_coeffs,
    _final_spin0815,
    _init_phi_prefactors,
    _is_default_order,
    _is_nonzero,
    _nudge_eta,
    _pi_powers,
    _powers,
    _subtract_3pn_ss,
    f_CUT,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing


_PI = math.pi
_SQRT_6 = math.sqrt(6.0)
_MAX_TOL_ATAN = 1.0e-15


@dataclass(frozen=True)
class _NNLOAngleCoefficients:
    alpha1: float
    alpha2: float
    alpha3: float
    alpha4: float
    alpha5: float
    epsilon1: float
    epsilon2: float
    epsilon3: float
    epsilon4: float
    epsilon5: float


@dataclass(frozen=True)
class _IMRPhenomPv2Inputs:
    mass1: float
    mass2: float
    chi1_l: float
    chi2_l: float
    chip: float
    theta_j: float
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


def _rotate_y(angle, vector):
    """Apply LAL's active y-axis rotation to a three-vector."""

    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return x * cosine + z * sine, y, -x * sine + z * cosine


def _rotate_z(angle, vector):
    """Apply LAL's active z-axis rotation to a three-vector."""

    x, y, z = vector
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return x * cosine - y * sine, x * sine + y * cosine, z


def _atan2tol(y, x):
    if abs(y) < _MAX_TOL_ATAN and abs(x) < _MAX_TOL_ATAN:
        return 0.0
    return math.atan2(y, x)


def _l2pnr(v, eta):
    """Nonspinning 2PN orbital angular momentum used by PhenomPv2."""

    x = v * v
    eta2 = eta * eta
    numerator = eta * (
        1.0
        + (1.5 + eta / 6.0) * x
        + (3.375 - 19.0 * eta / 8.0 - eta2 / 24.0) * x * x
    )
    denominator = torch.sqrt(x) if isinstance(x, torch.Tensor) else math.sqrt(x)
    return numerator / denominator


def _source_frame_parameters(
    mass1,
    mass2,
    f_ref,
    coa_phase,
    inclination,
    spin1,
    spin2,
):
    """Map LAL's source-frame inputs to PhenomPv2 model parameters."""

    m1sq = mass1 * mass1
    m2sq = mass2 * mass2
    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)

    chi1_l = spin1[2]
    chi2_l = spin2[2]
    spin1_perp_mag = m1sq * math.hypot(spin1[0], spin1[1])
    spin2_perp_mag = m2sq * math.hypot(spin2[0], spin2[1])
    a1 = 2.0 + 1.5 * mass2 / mass1
    a2 = 2.0 + 1.5 * mass1 / mass2
    numerator = max(a1 * spin1_perp_mag, a2 * spin2_perp_mag)
    denominator = a2 * m2sq if mass2 > mass1 else a1 * m1sq
    chip = numerator / denominator

    velocity = (_PI * total_mass * _MTSUN_SI * f_ref) ** (1.0 / 3.0)
    orbital_momentum = total_mass * total_mass * _l2pnr(velocity, eta)
    jx = m1sq * spin1[0] + m2sq * spin2[0]
    jy = m1sq * spin1[1] + m2sq * spin2[1]
    jz = orbital_momentum + m1sq * spin1[2] + m2sq * spin2[2]
    jmag = math.sqrt(jx * jx + jy * jy + jz * jz)
    theta_j_source = 0.0 if jmag < 1.0e-10 else math.acos(jz / jmag)
    if abs(jx) < _MAX_TOL_ATAN and abs(jy) < _MAX_TOL_ATAN:
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
        vector = _rotate_z(-phi_j_source, vector)
        return _rotate_y(-theta_j_source, vector)

    rotated_sight = rotate_to_intermediate(line_of_sight)
    kappa = -_atan2tol(rotated_sight[1], rotated_sight[0])

    rotated_orbit = _rotate_z(
        kappa,
        rotate_to_intermediate((0.0, 0.0, 1.0)),
    )
    if (
        abs(rotated_orbit[0]) < _MAX_TOL_ATAN
        and abs(rotated_orbit[1]) < _MAX_TOL_ATAN
    ):
        alpha0 = _PI
    else:
        alpha0 = math.atan2(rotated_orbit[1], rotated_orbit[0])

    rotated_sight = _rotate_z(kappa, rotated_sight)
    theta_j = math.acos(max(-1.0, min(1.0, rotated_sight[2])))

    waveframe_x = (
        -math.cos(inclination) * math.sin(coa_phase),
        -math.cos(inclination) * math.cos(coa_phase),
        math.sin(inclination),
    )
    rotated_x = _rotate_z(kappa, rotate_to_intermediate(waveframe_x))
    x_dot_p = -rotated_x[1]
    x_dot_q = (
        rotated_x[0] * rotated_sight[2]
        - rotated_x[2] * rotated_sight[0]
    )
    polarization_rotation = math.atan2(x_dot_q, x_dot_p)
    return (
        chi1_l,
        chi2_l,
        chip,
        theta_j,
        alpha0,
        phi_aligned,
        polarization_rotation,
    )


def _nnlo_angle_coefficients(q, chil, chip):
    """Return the NNLO precession-angle coefficients from LAL."""

    m2 = q / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    dm = m1 - m2
    eta = m1 * m2
    eta2 = eta * eta
    dm2 = dm * dm
    dm3 = dm2 * dm
    chil2 = chil * chil
    chip2 = chip * chip
    chip4 = chip2 * chip2
    m2_2 = m2 * m2
    m2_3 = m2_2 * m2
    m2_4 = m2_3 * m2
    m2_5 = m2_4 * m2
    m2_6 = m2_5 * m2
    m2_7 = m2_6 * m2
    m2_8 = m2_7 * m2

    alpha1 = -0.18229166666666666 - 5.0 * dm / (64.0 * m2)
    alpha2 = (
        -15.0 * dm * m2 * chil / (128.0 * eta)
        - 35.0 * m2_2 * chil / (128.0 * eta)
    )
    alpha3 = (
        -1.7952473958333333
        - 4555.0 * dm / (7168.0 * m2)
        - 15.0 * chip2 * dm * m2_3 / (128.0 * eta2)
        - 35.0 * chip2 * m2_4 / (128.0 * eta2)
        - 515.0 * eta / 384.0
        - 15.0 * dm2 * eta / (256.0 * m2_2)
        - 175.0 * dm * eta / (256.0 * m2)
    )
    alpha4 = (
        -35.0 * _PI / 48.0
        - 5.0 * dm * _PI / (16.0 * m2)
        + 5.0 * dm2 * chil / 16.0
        + 5.0 * dm * m2 * chil / 3.0
        + 2545.0 * m2_2 * chil / 1152.0
        - 5.0 * chip2 * dm * m2_5 * chil / (128.0 * eta**3)
        - 35.0 * chip2 * m2_6 * chil / (384.0 * eta**3)
        + 2035.0 * dm * m2 * chil / (21504.0 * eta)
        + 2995.0 * m2_2 * chil / (9216.0 * eta)
    )
    alpha5 = (
        4.318908476114694
        + 27895885.0 * dm / (2.1676032e7 * m2)
        - 15.0 * chip4 * dm * m2_7 / (512.0 * eta**4)
        - 35.0 * chip4 * m2_8 / (512.0 * eta**4)
        - 485.0 * chip2 * dm * m2_3 / (14336.0 * eta2)
        + 475.0 * chip2 * m2_4 / (6144.0 * eta2)
        + 15.0 * chip2 * dm2 * m2_2 / (256.0 * eta)
        + 145.0 * chip2 * dm * m2_3 / (512.0 * eta)
        + 575.0 * chip2 * m2_4 / (1536.0 * eta)
        + 39695.0 * eta / 86016.0
        + 1615.0 * dm2 * eta / (28672.0 * m2_2)
        - 265.0 * dm * eta / (14336.0 * m2)
        + 955.0 * eta2 / 576.0
        + 15.0 * dm3 * eta2 / (1024.0 * m2_3)
        + 35.0 * dm2 * eta2 / (256.0 * m2_2)
        + 2725.0 * dm * eta2 / (3072.0 * m2)
        - 15.0 * dm * m2 * _PI * chil / (16.0 * eta)
        - 35.0 * m2_2 * _PI * chil / (16.0 * eta)
        + 15.0 * chip2 * dm * m2_7 * chil2 / (128.0 * eta**4)
        + 35.0 * chip2 * m2_8 * chil2 / (128.0 * eta**4)
        + 375.0 * dm2 * m2_2 * chil2 / (256.0 * eta)
        + 1815.0 * dm * m2_3 * chil2 / (256.0 * eta)
        + 1645.0 * m2_4 * chil2 / (192.0 * eta)
    )
    epsilon1 = alpha1
    epsilon2 = alpha2
    epsilon3 = (
        -1.7952473958333333
        - 4555.0 * dm / (7168.0 * m2)
        - 515.0 * eta / 384.0
        - 15.0 * dm2 * eta / (256.0 * m2_2)
        - 175.0 * dm * eta / (256.0 * m2)
    )
    epsilon4 = (
        -35.0 * _PI / 48.0
        - 5.0 * dm * _PI / (16.0 * m2)
        + 5.0 * dm2 * chil / 16.0
        + 5.0 * dm * m2 * chil / 3.0
        + 2545.0 * m2_2 * chil / 1152.0
        + 2035.0 * dm * m2 * chil / (21504.0 * eta)
        + 2995.0 * m2_2 * chil / (9216.0 * eta)
    )
    epsilon5 = (
        4.318908476114694
        + 27895885.0 * dm / (2.1676032e7 * m2)
        + 39695.0 * eta / 86016.0
        + 1615.0 * dm2 * eta / (28672.0 * m2_2)
        - 265.0 * dm * eta / (14336.0 * m2)
        + 955.0 * eta2 / 576.0
        + 15.0 * dm3 * eta2 / (1024.0 * m2_3)
        + 35.0 * dm2 * eta2 / (256.0 * m2_2)
        + 2725.0 * dm * eta2 / (3072.0 * m2)
        - 15.0 * dm * m2 * _PI * chil / (16.0 * eta)
        - 35.0 * m2_2 * _PI * chil / (16.0 * eta)
        + 375.0 * dm2 * m2_2 * chil2 / (256.0 * eta)
        + 1815.0 * dm * m2_3 * chil2 / (256.0 * eta)
        + 1645.0 * m2_4 * chil2 / (192.0 * eta)
    )
    return _NNLOAngleCoefficients(
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        alpha5,
        epsilon1,
        epsilon2,
        epsilon3,
        epsilon4,
        epsilon5,
    )


def _angle_series(omega, coefficients):
    omega_cbrt = torch.pow(omega, 1.0 / 3.0)
    omega_cbrt2 = omega_cbrt * omega_cbrt
    return (
        coefficients[0] / omega
        + coefficients[1] / omega_cbrt2
        + coefficients[2] / omega_cbrt
        + coefficients[3] * torch.log(omega)
        + coefficients[4] * omega_cbrt
    )


def _scalar_angle_series(omega, coefficients):
    omega_cbrt = omega ** (1.0 / 3.0)
    return (
        coefficients[0] / omega
        + coefficients[1] / (omega_cbrt * omega_cbrt)
        + coefficients[2] / omega_cbrt
        + coefficients[3] * math.log(omega)
        + coefficients[4] * omega_cbrt
    )


def _final_spin(mass1, mass2, chi1_l, chi2_l, chip, eta):
    """Generalize PhenomD's final spin with in-plane spin on the larger BH."""

    if mass1 >= mass2:
        mass_fraction = mass1 / (mass1 + mass2)
        aligned = _final_spin0815(eta, chi1_l, chi2_l)
    else:
        mass_fraction = mass2 / (mass1 + mass2)
        aligned = _final_spin0815(eta, chi2_l, chi1_l)
    in_plane = chip * mass_fraction * mass_fraction
    sign = 1.0 if aligned >= 0.0 else -1.0
    return max(-1.0, min(1.0, sign * math.hypot(in_plane, aligned)))


def _as_float(value, default=0.0):
    return float(default if value is None else value)


def imrphenompv2_native_supported(params):
    """Return whether a parameter set is covered by the native BBH port."""

    if params.get("approximant", "IMRPhenomPv2") != "IMRPhenomPv2":
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
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


def _validated_inputs(
    params,
    *,
    sequence=False,
    default_reference_frequency=None,
):
    if not imrphenompv2_native_supported(params):
        raise ValueError(
            "IMRPhenomPv2 parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomPv2 requires TorchScheme")

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    distance = pnutils.megaparsecs_to_meters(float(params["distance"]))
    inclination = _as_float(params.get("inclination"))
    coa_phase = _as_float(params.get("coa_phase"))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument and
    # ignores the corresponding PyCBC parameter.
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
        raise ValueError("IMRPhenomPv2 inputs must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomPv2 component masses must be positive")
    if distance <= 0.0:
        raise ValueError("IMRPhenomPv2 distance must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomPv2 f_ref must be non-negative")
    if sum(value * value for value in spin1) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomPv2 spin1 magnitude must not exceed one")
    if sum(value * value for value in spin2) > 1.0 + 1.0e-14:
        raise ValueError("IMRPhenomPv2 spin2 magnitude must not exceed one")

    reference_frequency = f_ref
    if reference_frequency == 0.0:
        if default_reference_frequency is None:
            reference_frequency = float(params["f_lower"])
        else:
            reference_frequency = float(default_reference_frequency)
    if not math.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "IMRPhenomPv2 reference frequency must be finite and positive"
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
    )

    # PhenomP's internal convention is m2 >= m1.
    if mass1 > mass2:
        mass1, mass2 = mass2, mass1
        chi1_l, chi2_l = chi2_l, chi1_l
    total_mass = mass1 + mass2
    eta = _nudge_eta(mass1 * mass2 / (total_mass * total_mass))
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _IMRPhenomPv2Inputs(
        mass1=mass1,
        mass2=mass2,
        chi1_l=chi1_l,
        chi2_l=chi2_l,
        chip=chip,
        theta_j=theta_j,
        alpha0=alpha0,
        distance=distance,
        phi_aligned=phi_aligned,
        polarization_rotation=polarization_rotation,
        long_asc_nodes=long_asc_nodes,
        f_ref=reference_frequency,
        total_mass=total_mass,
        total_mass_seconds=total_mass * _MTSUN_SI,
        eta=eta,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


@dataclass(frozen=True)
class _Pv2Model:
    inputs: _IMRPhenomPv2Inputs
    amp_coeffs: object
    phase_coeffs: object
    pn: object
    phase_prefactors: object
    angle_coeffs: _NNLOAngleCoefficients
    alpha_offset: float
    epsilon_offset: float
    harmonics: tuple
    final_frequency: float
    time_correction: float


def _raw_phase(model, frequencies):
    mf = model.inputs.total_mass_seconds * frequencies
    return _IMRPhenDPhase(
        mf,
        model.phase_coeffs,
        model.pn,
        model.phase_prefactors,
        _pi_powers(),
    ) - 2.0 * model.inputs.phi_aligned


def _build_model(inputs):
    eta = inputs.eta
    final_spin = _final_spin(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        inputs.chip,
        eta,
    )
    # IMRPhenomD's coefficient convention has the larger body first.
    amp_coeffs = _compute_amp_coeffs(
        eta,
        inputs.chi2_l,
        inputs.chi1_l,
        final_spin,
    )
    pn = taylorf2_aligned_phasing(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        spin_order=-1,
        tidal_order=-1,
        dchi={},
        qm_def1=0.0,
        qm_def2=0.0,
        lambda1=0.0,
        lambda2=0.0,
    )
    pn.v[6] -= _subtract_3pn_ss(
        inputs.mass1,
        inputs.mass2,
        inputs.total_mass,
        eta,
        inputs.chi1_l,
        inputs.chi2_l,
    ) * pn.v[0]
    phase_coeffs = _compute_phase_coeffs(
        eta,
        inputs.chi2_l,
        inputs.chi1_l,
        final_spin,
        pn,
    )
    phase_prefactors = _init_phi_prefactors(
        phase_coeffs.sigma1,
        phase_coeffs.sigma2,
        phase_coeffs.sigma3,
        phase_coeffs.sigma4,
        pn,
        _pi_powers(),
    )

    q = inputs.mass2 / inputs.mass1
    chi_eff = (
        inputs.mass1 * inputs.chi1_l
        + inputs.mass2 * inputs.chi2_l
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

    final_frequency = amp_coeffs.fRD / inputs.total_mass_seconds
    fixed_start = 0.8 * final_frequency
    fixed_stop = min(
        1.2 * final_frequency,
        f_CUT / inputs.total_mass_seconds,
    )
    fixed_frequencies = np.linspace(fixed_start, fixed_stop, 10)
    fixed_mf = inputs.total_mass_seconds * fixed_frequencies
    fixed_phase = -(
        _IMRPhenDPhase(
            fixed_mf,
            phase_coeffs,
            pn,
            phase_prefactors,
            _pi_powers(),
        )
            - 2.0 * inputs.phi_aligned
    )
    derivative = CubicSpline(
        fixed_frequencies,
        fixed_phase,
        bc_type="natural",
    )(final_frequency, 1)
    time_correction = float(derivative / (2.0 * _PI))
    return _Pv2Model(
        inputs,
        amp_coeffs,
        phase_coeffs,
        pn,
        phase_prefactors,
        angle_coeffs,
        alpha_offset,
        epsilon_offset,
        harmonics,
        final_frequency,
        time_correction,
    )


def _twist_up(model, frequencies):
    inputs = model.inputs
    mf = inputs.total_mass_seconds * frequencies
    powers = _powers(mf)
    amplitude = _IMRPhenDAmplitude(mf, model.amp_coeffs, powers)
    phase = _raw_phase(model, frequencies)
    amplitude_scale = (
        inputs.total_mass
        * _MRSUN_SI
        * inputs.total_mass
        * _MTSUN_SI
        / inputs.distance
    )
    h_phenom = torch.complex(
        amplitude_scale * amplitude * torch.cos(phase),
        -amplitude_scale * amplitude * torch.sin(phase),
    ).to(inputs.complex_dtype)

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
    omega = _PI * mf
    alpha = _angle_series(omega, alpha_values) - model.alpha_offset
    epsilon = _angle_series(omega, epsilon_values) - model.epsilon_offset

    q = inputs.mass2 / inputs.mass1
    small_fraction = 1.0 / (1.0 + q)
    large_fraction = q / (1.0 + q)
    spin_perp = inputs.chip * large_fraction * large_fraction
    spin_aligned = (
        inputs.chi1_l * small_fraction * small_fraction
        + inputs.chi2_l * large_fraction * large_fraction
    )
    velocity = torch.pow(omega, 1.0 / 3.0)
    ratio = spin_perp / (_l2pnr(velocity, inputs.eta) + spin_aligned)
    cos_beta = torch.rsqrt(1.0 + ratio * ratio)
    cos_half = torch.sqrt(0.5 * (1.0 + cos_beta))
    sin_half = torch.sqrt(0.5 * (1.0 - cos_beta))
    cos2 = cos_half * cos_half
    sin2 = sin_half * sin_half
    d2 = (
        sin2 * sin2,
        2.0 * cos_half * sin2 * sin_half,
        _SQRT_6 * sin2 * cos2,
        2.0 * cos2 * cos_half * sin_half,
        cos2 * cos2,
    )
    dm2 = (d2[4], -d2[3], d2[2], -d2[1], d2[0])
    hp_sum = torch.zeros_like(h_phenom)
    hc_sum = torch.zeros_like(h_phenom)
    for index, emm in enumerate(range(-2, 3)):
        exp_negative = torch.exp(-1j * emm * alpha)
        exp_positive = torch.exp(1j * emm * alpha)
        t2m = exp_negative * dm2[index] * model.harmonics[index]
        tm2m = exp_positive * d2[index] * torch.conj(
            model.harmonics[index]
        )
        hp_sum += t2m + tm2m
        hc_sum += 1j * (t2m - tm2m)

    factor = torch.exp(-2j * epsilon) * h_phenom / 2.0
    plus = factor * hp_sum
    cross = factor * hc_sum
    time_phase = torch.exp(-2j * _PI * frequencies * model.time_correction)
    plus *= time_phase
    cross *= time_phase

    cosine = math.cos(2.0 * inputs.polarization_rotation)
    sine = math.sin(2.0 * inputs.polarization_rotation)
    plus, cross = cosine * plus + sine * cross, cosine * cross - sine * plus

    cosine = math.cos(2.0 * inputs.long_asc_nodes)
    sine = math.sin(2.0 * inputs.long_asc_nodes)
    return cosine * plus + sine * cross, cosine * cross - sine * plus


def imrphenompv2_fd_torch(**params):
    """Generate a regular-grid BBH IMRPhenomPv2 waveform with Torch."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = _as_float(params.get("f_final"))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomPv2 frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomPv2 delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomPv2 f_final must be non-negative")

    inputs = _validated_inputs(params)
    f_cut = f_CUT / inputs.total_mass_seconds
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


def imrphenompv2_sequence_native_supported(params):
    """Return whether the native port covers a sequence parameter set."""

    return imrphenompv2_native_supported(params)


def _sequence_frequencies(sample_points):
    """Validate and move sequence frequencies to the active Torch device."""

    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomPv2 requires TorchScheme")
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
        raise ValueError(
            "IMRPhenomPv2 sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomPv2 sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomPv2 sample_points must be positive")
    if frequencies.numel() > 1 and not bool(
        torch.all(frequencies[1:] > frequencies[:-1])
    ):
        raise ValueError(
            "IMRPhenomPv2 sample_points must be strictly increasing"
        )
    return frequencies


def imrphenompv2_fd_sequence_torch(**params):
    """Evaluate BBH IMRPhenomPv2 at arbitrary frequencies with Torch."""

    if not imrphenompv2_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomPv2 sequence parameters are not supported by the "
            "native Torch path"
        )
    frequencies = _sequence_frequencies(params["sample_points"])
    inputs = _validated_inputs(
        params,
        sequence=True,
        default_reference_frequency=float(frequencies[0].item()),
    )
    f_cut = f_CUT / inputs.total_mass_seconds
    if float(frequencies[0].item()) >= f_cut:
        raise ValueError("IMRPhenomPv2 fCut must exceed the first sample point")

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
    "imrphenompv2_fd_sequence_torch",
    "imrphenompv2_fd_torch",
    "imrphenompv2_native_supported",
    "imrphenompv2_sequence_native_supported",
]
