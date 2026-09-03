# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native IMRPhenomNSBH frequency-domain waveform.

The implementation follows ``LALSimIMRPhenomNSBH.c``. The PhenomC-inspired
amplitude and the PhenomD plus NRTidalv2 phase are evaluated on the active
Torch device. Small phenomenological fits remain host-side scalar setup; the
disruption-radius polynomial is solved through Torch rather than NumPy.

Activation
----------
- Native by default under ``TorchScheme``
- Per-model opt-out: ``PYCBC_IMRPHENOMNSBH_NATIVE=0``
- Global opt-out   : ``PYCBC_TORCH_NATIVE_PORTS=0``

The native path is used only under ``TorchScheme`` and only for the aligned-
spin, dominant-mode model supported here. Unsupported waveform modifications
retain the lalsimulation path.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

from pycbc import lal_compat as lal
import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._rom_hybrid_torch import _minimum_sequence_frequency
from pycbc.waveform.imrphenomc_torch import (
    _IMRPhenomCInputs,
    _distance_scale,
    _imrphenomc_coefficients,
    _imrphenomc_spa_amplitude,
)
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _IMRPhenDPhase,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _compute_phase_coeffs,
    _d_phi_mrd,
    _imrphenomd_polarizations,
    _init_phi_prefactors,
    _is_default_order,
    _is_nonzero,
    _pi_powers,
    _subtract_3pn_ss,
)
from pycbc.waveform.nrtidal_torch import (
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
)
from pycbc.waveform.nsbh_torch import (
    bhns_mass_aligned,
    bhns_spin_aligned,
    nsbh_compactness_from_lambda,
    nsbh_torus_mass_fit,
    nsbh_xi_tide,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing


_PI = lal.PI
_MTSUN_SI = lal.MTSUN_SI
_MPS_MIN_START_MF = 5.0e-4


def _lambda2_is_supported(value) -> bool:
    """Return whether a value is a valid native NS deformability."""

    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and 0.0 <= value <= 5000.0


def _native_features_supported(params) -> bool:
    """Return whether ``params`` use features implemented by this port."""

    if params.get("approximant", "IMRPhenomNSBH") != "IMRPhenomNSBH":
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in (*_DEFAULT_ONLY_ORDER_KEYS, "phase_order", "amplitude_order")
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + (
                "lambda1",
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    if not _lambda2_is_supported(params.get("lambda2", 0.0)):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def _native_device_supported(params, *, sequence) -> bool:
    """Bound accumulated float32 phase error on Apple MPS."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    ):
        return True

    try:
        total_mass = float(params["mass1"]) + float(params["mass2"])
        if sequence:
            start_frequency = _minimum_sequence_frequency(
                params["sample_points"]
            )
        else:
            delta_f = float(params["delta_f"])
            f_lower = float(params["f_lower"])
            if delta_f <= 0.0 or f_lower <= 0.0:
                return False
            # The regular generator follows LAL's integer-bin convention and
            # may evaluate below a non-bin-aligned f_lower.
            start_frequency = int(f_lower / delta_f) * delta_f
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        ZeroDivisionError,
    ):
        return False
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (total_mass, start_frequency)
    ):
        return False
    start_mf = total_mass * _MTSUN_SI * start_frequency
    return math.isfinite(start_mf) and start_mf >= _MPS_MIN_START_MF


def imrphenomnsbh_native_supported(params) -> bool:
    """Return whether regular-grid generation is Torch-native."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=False,
    )


def imrphenomnsbh_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency generation is Torch-native."""

    return _native_features_supported(params) and _native_device_supported(
        params,
        sequence=True,
    )


@dataclass(frozen=True)
class _IMRPhenomNSBHInputs:
    """Validated scalar inputs shared by both sampling interfaces."""

    mass_bh: float
    mass_ns: float
    spin_bh: float
    spin_ns: float
    lambda_ns: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    f_ref: float
    total_mass: float
    eta: float
    total_mass_seconds: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _IMRPhenomNSBHAmplitudeParameters:
    """Frequency-independent NSBH amplitude and remnant fits."""

    compactness: float
    torus_mass: float
    final_spin: float
    final_mass_fraction: float
    ringdown_frequency: float
    tidal_frequency: float
    q_factor: float
    gamma_correction: float
    sigma: float
    epsilon_tide: float
    epsilon_ins: float
    sigma_tide: float
    transition_pn: float
    transition_pm: float
    transition_rd: float


def _imrphenomnsbh_inputs(params, *, sequence=False):
    """Validate scalar inputs and select the active Torch precision."""

    if not _native_features_supported(params):
        raise ValueError(
            "IMRPhenomNSBH parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomNSBH requires TorchScheme")

    mass_bh = float(params["mass1"])
    mass_ns = float(params["mass2"])
    spin_bh = float(params.get("spin1z", 0.0))
    spin_ns = float(params.get("spin2z", 0.0))
    lambda_ns = float(params.get("lambda2") or 0.0)
    distance_mpc = float(params["distance"])
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )
    f_ref = float(params.get("f_ref", 0.0))

    values = (
        mass_bh,
        mass_ns,
        spin_bh,
        spin_ns,
        lambda_ns,
        distance_mpc,
        inclination,
        coa_phase,
        long_asc_nodes,
        f_ref,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("IMRPhenomNSBH parameters must be finite")
    if mass_bh <= 0.0 or mass_ns <= 0.0:
        raise ValueError("IMRPhenomNSBH component masses must be positive")
    if mass_bh < mass_ns:
        raise ValueError("IMRPhenomNSBH mass1 must be the black-hole mass")
    if mass_bh > 100.0 * mass_ns:
        raise ValueError("IMRPhenomNSBH mass ratio must not exceed 100")
    if mass_ns > 3.0:
        raise ValueError("IMRPhenomNSBH neutron-star mass must not exceed 3")
    if abs(spin_bh) > 1.0 or abs(spin_ns) > 1.0:
        raise ValueError("IMRPhenomNSBH aligned spins must be between -1 and 1")
    if not 0.0 <= lambda_ns <= 5000.0:
        raise ValueError("IMRPhenomNSBH lambda2 must be between 0 and 5000")
    if distance_mpc <= 0.0:
        raise ValueError("IMRPhenomNSBH distance must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomNSBH f_ref must be non-negative")

    if spin_ns != 0.0:
        warnings.warn(
            "IMRPhenomNSBH is not calibrated for a non-zero NS spin",
            RuntimeWarning,
            stacklevel=3,
        )
    if mass_ns < 1.0:
        warnings.warn(
            "IMRPhenomNSBH is not calibrated for an NS mass below 1 solar mass",
            RuntimeWarning,
            stacklevel=3,
        )

    total_mass = mass_bh + mass_ns
    eta = mass_bh * mass_ns / total_mass**2
    total_mass_seconds = total_mass * _MTSUN_SI
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _IMRPhenomNSBHInputs(
        mass_bh=mass_bh,
        mass_ns=mass_ns,
        spin_bh=spin_bh,
        spin_ns=spin_ns,
        lambda_ns=lambda_ns,
        distance=distance_mpc * _distance_scale(total_mass),
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        f_ref=f_ref,
        total_mass=total_mass,
        eta=eta,
        total_mass_seconds=total_mass_seconds,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _phenomc_coefficients(inputs):
    """Return the PhenomC amplitude coefficients used by IMRPhenomNSBH."""

    effective_spin = (
        inputs.mass_bh * inputs.spin_bh
        + inputs.mass_ns * inputs.spin_ns
    ) / inputs.total_mass
    phenomc_inputs = _IMRPhenomCInputs(
        distance=inputs.distance,
        inclination=inputs.inclination,
        coa_phase=inputs.coa_phase,
        long_asc_nodes=inputs.long_asc_nodes,
        total_mass=inputs.total_mass,
        eta=inputs.eta,
        xi=effective_spin,
        total_mass_seconds=inputs.total_mass_seconds,
        f_cut=0.15 / inputs.total_mass_seconds,
        device=inputs.device,
        real_dtype=inputs.real_dtype,
        complex_dtype=inputs.complex_dtype,
    )
    coefficients = _imrphenomc_coefficients(phenomc_inputs)
    replacements = {}
    if coefficients.g1 < 0.0:
        warnings.warn(
            "IMRPhenomNSBH increased negative PhenomC gamma_1 to zero",
            RuntimeWarning,
            stacklevel=3,
        )
        replacements["g1"] = 0.0
    if coefficients.del1 < 0.0:
        warnings.warn(
            "IMRPhenomNSBH increased negative PhenomC delta_1 to zero",
            RuntimeWarning,
            stacklevel=3,
        )
        replacements["del1"] = 0.0
    if coefficients.del2 < 1.0e-4:
        warnings.warn(
            "IMRPhenomNSBH increased PhenomC delta_2 to 1e-4",
            RuntimeWarning,
            stacklevel=3,
        )
        replacements["del2"] = 1.0e-4
    if replacements:
        coefficients = replace(coefficients, **replacements)
    return coefficients


def _fit_window(value: float, center: float, width: float, sign: float) -> float:
    """Return the half-height scalar sigmoid used by the NSBH fits."""

    return 0.25 * (1.0 + sign * math.tanh(4.0 * (value - center) / width))


def _omega_tilde(final_spin: float) -> complex:
    """Return the fitted dimensionless (2,2,0) QNM frequency."""

    kappa = math.sqrt(math.log(2.0 - final_spin) / math.log(3.0))

    def polar(magnitude, angle):
        return magnitude * complex(math.cos(angle), math.sin(angle))

    return 1.0 + kappa * (
        polar(1.5578, 2.9031)
        + polar(1.9510, 5.9210) * kappa
        + polar(2.0997, 2.7606) * kappa**2
        + polar(1.4109, 5.9143) * kappa**3
        + polar(0.4106, 2.7952) * kappa**4
    )


def _imrphenomnsbh_amplitude_parameters(inputs, coefficients):
    """Assemble the disruption and remnant fits used by the amplitude."""

    mass_ratio = inputs.mass_bh / inputs.mass_ns
    spin = inputs.spin_bh
    lambda_ns = inputs.lambda_ns
    compactness = nsbh_compactness_from_lambda(lambda_ns)
    torus_mass = nsbh_torus_mass_fit(mass_ratio, spin, compactness)
    final_spin = bhns_spin_aligned(
        inputs.mass_bh,
        inputs.mass_ns,
        spin,
        lambda_ns,
    )
    final_mass_fraction = bhns_mass_aligned(
        inputs.mass_bh,
        inputs.mass_ns,
        spin,
        lambda_ns,
    ) / inputs.total_mass
    omega = _omega_tilde(final_spin)
    ringdown_frequency = omega.real / (2.0 * _PI * final_mass_fraction)
    q_factor = omega.real / omega.imag / 2.0

    mu = mass_ratio * compactness
    xi_tide = nsbh_xi_tide(mass_ratio, spin, mu)
    tidal_radius = xi_tide * (1.0 - 2.0 * compactness) / mu
    denominator = _PI * (spin + math.sqrt(tidal_radius**3))
    if denominator == 0.0:
        tidal_frequency = math.inf
    else:
        tidal_frequency = abs((1.0 + 1.0 / mass_ratio) / denominator)

    fitted_ringdown = 0.99 * 0.98 * ringdown_frequency
    lambda_sq = lambda_ns * lambda_ns
    if lambda_ns > 1.0:
        gamma_correction = 1.25
        frequency_ratio = (
            tidal_frequency - fitted_ringdown
        ) / fitted_ringdown
        delta2_prime = 1.62496 * _fit_window(
            frequency_ratio,
            0.0188092,
            0.338737,
            1.0,
        )
    else:
        gamma_correction = 1.0 + 0.5 * lambda_ns - 0.25 * lambda_sq
        c2 = coefficients.del2 - 0.81248
        delta2_prime = (
            coefficients.del2
            - 2.0 * c2 * lambda_ns
            + c2 * lambda_sq
        )
    sigma = delta2_prime * ringdown_frequency / q_factor

    ratio_sq = ((tidal_frequency - fitted_ringdown) / fitted_ringdown) ** 2
    x_nondisruptive = (
        ratio_sq - 0.571505 * compactness - 0.00508451 * spin
    )
    x_nondisruptive_prime = (
        ratio_sq - 0.657424 * compactness - 0.0259977 * spin
    )
    eta = mass_ratio / (1.0 + mass_ratio) ** 2
    x_disruptive = (
        torus_mass
        + 0.424912 * compactness
        + 0.363604 * math.sqrt(eta)
        - 0.0605591 * spin
    )
    x_disruptive_prime = (
        torus_mass
        - 0.132754 * compactness
        + 0.576669 * math.sqrt(eta)
        - 0.0603749 * spin
        - 0.0601185 * spin**2
        - 0.0729134 * spin**3
    )

    if tidal_frequency < ringdown_frequency:
        epsilon_tide = 0.0
        epsilon_ins = min(1.0, 1.29971 - 1.61724 * x_disruptive)
        sigma_tide = 0.137722 - 0.293237 * x_disruptive_prime
        transition_rd = 0.0
        if torus_mass > 0.0:
            transition_pn = tidal_frequency
            transition_pm = tidal_frequency
        else:
            transition_pn = (
                (1.0 - 1.0 / mass_ratio) * fitted_ringdown
                + epsilon_ins * tidal_frequency / mass_ratio
            )
            transition_pm = (
                (1.0 - 1.0 / mass_ratio) * fitted_ringdown
                + tidal_frequency / mass_ratio
            )
            sigma_tide_nondisruptive = 2.0 * _fit_window(
                x_nondisruptive_prime,
                -0.206465,
                0.226844,
                -1.0,
            )
            sigma_tide = 0.5 * (
                sigma_tide + sigma_tide_nondisruptive
            )
    else:
        if lambda_ns > 1.0:
            transition_pn = fitted_ringdown
        else:
            transition_pn = (
                1.0 - 0.02 * lambda_ns + 0.01 * lambda_sq
            ) * 0.98 * ringdown_frequency
        transition_pm = transition_pn
        transition_rd = transition_pn
        epsilon_tide = 2.0 * _fit_window(
            x_nondisruptive,
            -0.0796251,
            0.0801192,
            1.0,
        )
        sigma_tide = 2.0 * _fit_window(
            x_nondisruptive_prime,
            -0.206465,
            0.226844,
            -1.0,
        )
        epsilon_ins = (
            1.29971 - 1.61724 * x_disruptive
            if torus_mass > 0.0
            else 1.0
        )

    return _IMRPhenomNSBHAmplitudeParameters(
        compactness=compactness,
        torus_mass=torus_mass,
        final_spin=final_spin,
        final_mass_fraction=final_mass_fraction,
        ringdown_frequency=ringdown_frequency,
        tidal_frequency=tidal_frequency,
        q_factor=q_factor,
        gamma_correction=gamma_correction,
        sigma=sigma,
        epsilon_tide=epsilon_tide,
        epsilon_ins=epsilon_ins,
        sigma_tide=sigma_tide,
        transition_pn=transition_pn,
        transition_pm=transition_pm,
        transition_rd=transition_rd,
    )


def _imrphenomnsbh_amplitude(
    inputs,
    coefficients,
    amplitude_parameters,
    frequencies,
):
    """Evaluate the dimensionless NSBH amplitude on the active device."""

    dimensionless_frequencies = inputs.total_mass_seconds * frequencies
    inspiral = _imrphenomc_spa_amplitude(
        coefficients,
        frequencies,
        inputs.total_mass_seconds,
    )
    post_merger = (
        amplitude_parameters.gamma_correction
        * coefficients.g1
        * torch.pow(dimensionless_frequencies, 5.0 / 6.0)
    )
    sigma_sq = amplitude_parameters.sigma**2
    lorentzian = sigma_sq / (
        (
            dimensionless_frequencies
            - amplitude_parameters.ringdown_frequency
        )
        ** 2
        + sigma_sq / 4.0
    )
    ringdown = (
        amplitude_parameters.epsilon_tide
        * coefficients.del1
        * lorentzian
        * torch.pow(dimensionless_frequencies, -7.0 / 6.0)
    )

    width = coefficients.d0 + amplitude_parameters.sigma_tide

    def window(center, sign):
        return 0.5 * (
            1.0
            + sign
            * torch.tanh(
                4.0 * (dimensionless_frequencies - center) / width
            )
        )

    inspiral_window = window(
        amplitude_parameters.epsilon_ins
        * amplitude_parameters.transition_pn,
        -1.0,
    )
    post_merger_window = window(
        amplitude_parameters.transition_pm,
        -1.0,
    )
    ringdown_window = window(
        amplitude_parameters.transition_rd,
        1.0,
    )
    return -(
        inspiral * inspiral_window
        + post_merger * post_merger_window
        + ringdown * ringdown_window
    )


def _imrphenomnsbh_samples(
    inputs,
    coefficients,
    amplitude_parameters,
    frequencies,
    reference_frequency,
    final_frequency,
):
    """Evaluate the inclination-independent strain at device frequencies."""

    dquad_ns = nrtidal_quadrupole_from_lambda(inputs.lambda_ns) - 1.0
    pn = taylorf2_aligned_phasing(
        inputs.mass_bh,
        inputs.mass_ns,
        inputs.spin_bh,
        inputs.spin_ns,
        spin_order=7,
        tidal_order=-1,
        dchi={},
        qm_def1=0.0,
        qm_def2=dquad_ns,
        lambda1=0.0,
        lambda2=0.0,
    )
    pn.v[6] -= _subtract_3pn_ss(
        inputs.mass_bh,
        inputs.mass_ns,
        inputs.total_mass,
        inputs.eta,
        inputs.spin_bh,
        inputs.spin_ns,
    ) * pn.v[0]
    phase_coefficients = _compute_phase_coeffs(
        inputs.eta,
        inputs.spin_bh,
        inputs.spin_ns,
        amplitude_parameters.final_spin,
        pn,
    )
    pi_powers = _pi_powers()
    phase_prefactors = _init_phi_prefactors(
        phase_coefficients.sigma1,
        phase_coefficients.sigma2,
        phase_coefficients.sigma3,
        phase_coefficients.sigma4,
        pn,
        pi_powers,
    )

    reference_mf = inputs.total_mass_seconds * reference_frequency
    reference_values = torch.as_tensor(
        reference_mf,
        dtype=inputs.real_dtype,
        device=inputs.device,
    ).reshape(1)
    phase_at_reference = _IMRPhenDPhase(
        reference_values,
        phase_coefficients,
        pn,
        phase_prefactors,
        pi_powers,
    )[0]
    phase_offset = 2.0 * inputs.coa_phase + phase_at_reference
    time_shift = _d_phi_mrd(
        inputs.total_mass_seconds * final_frequency,
        phase_coefficients.alpha1,
        phase_coefficients.alpha2,
        phase_coefficients.alpha3,
        phase_coefficients.alpha4,
        phase_coefficients.alpha5,
        phase_coefficients.fRD,
        phase_coefficients.fDM,
        phase_coefficients.eta_inv,
    )

    dimensionless_frequencies = inputs.total_mass_seconds * frequencies
    phase = _IMRPhenDPhase(
        dimensionless_frequencies,
        phase_coefficients,
        pn,
        phase_prefactors,
        pi_powers,
    )
    phase += nrtidal_phase(
        frequencies,
        inputs.mass_bh,
        inputs.mass_ns,
        0.0,
        inputs.lambda_ns,
        2,
    )
    phase -= (
        time_shift * (dimensionless_frequencies - reference_mf)
        + phase_offset
    )
    amplitude = _imrphenomnsbh_amplitude(
        inputs,
        coefficients,
        amplitude_parameters,
        frequencies,
    ) / inputs.distance
    return torch.complex(
        amplitude * torch.cos(phase),
        -amplitude * torch.sin(phase),
    ).to(inputs.complex_dtype)


def imrphenomnsbh_fd_torch(**params):
    """Generate regular-grid IMRPhenomNSBH polarizations with Torch."""

    inputs = _imrphenomnsbh_inputs(params)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomNSBH frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError(
            "IMRPhenomNSBH delta_f and f_lower must be positive"
        )
    if f_final < 0.0:
        raise ValueError("IMRPhenomNSBH f_final must be non-negative")

    final_frequency = (
        f_final
        if f_final > 0.0
        else 0.2 / inputs.total_mass_seconds
    )
    if final_frequency < f_lower:
        raise ValueError("IMRPhenomNSBH f_final is below f_lower")
    first_bin = int(f_lower / delta_f)
    stop_bin = int(final_frequency / delta_f)
    if stop_bin <= first_bin:
        raise ValueError("IMRPhenomNSBH frequency interval has no samples")

    layout_bins = int(final_frequency / delta_f)
    fft_length = (
        1 if layout_bins <= 1 else 1 << (layout_bins - 1).bit_length()
    )
    nfreq = fft_length + 1
    active_frequencies = (
        torch.arange(
            first_bin,
            stop_bin,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    coefficients = _phenomc_coefficients(inputs)
    amplitude_parameters = _imrphenomnsbh_amplitude_parameters(
        inputs,
        coefficients,
    )
    reference_frequency = inputs.f_ref if inputs.f_ref > 0.0 else f_lower
    active_samples = _imrphenomnsbh_samples(
        inputs,
        coefficients,
        amplitude_parameters,
        active_frequencies,
        reference_frequency,
        final_frequency,
    )
    samples = torch.zeros(
        nfreq,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    samples[first_bin:stop_bin] = active_samples
    plus, cross = _imrphenomd_polarizations(samples, inputs)
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


def _sequence_frequencies(sample_points, inputs):
    """Return validated arbitrary frequencies on the active Torch device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "IMRPhenomNSBH sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomNSBH sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomNSBH sample_points must be positive")
    if bool(frequencies[-1] < frequencies[0]):
        raise ValueError(
            "IMRPhenomNSBH final sample must not be below the first sample"
        )
    return frequencies


def imrphenomnsbh_fd_sequence_torch(**params):
    """Evaluate IMRPhenomNSBH at arbitrary frequencies with Torch."""

    inputs = _imrphenomnsbh_inputs(params, sequence=True)
    frequencies = _sequence_frequencies(params["sample_points"], inputs)
    coefficients = _phenomc_coefficients(inputs)
    amplitude_parameters = _imrphenomnsbh_amplitude_parameters(
        inputs,
        coefficients,
    )
    reference_frequency = (
        inputs.f_ref if inputs.f_ref > 0.0 else frequencies[0]
    )
    samples = _imrphenomnsbh_samples(
        inputs,
        coefficients,
        amplitude_parameters,
        frequencies,
        reference_frequency,
        frequencies[-1],
    )
    plus, cross = _imrphenomd_polarizations(samples, inputs)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = (
    "imrphenomnsbh_fd_sequence_torch",
    "imrphenomnsbh_fd_torch",
    "imrphenomnsbh_native_supported",
    "imrphenomnsbh_sequence_native_supported",
)
