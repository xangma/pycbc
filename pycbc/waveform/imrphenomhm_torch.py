# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Torch-native IMRPhenomHM waveforms.

This ports the six aligned-spin modes modeled by LALSimIMRPhenomHM.c:
(2,2), (2,1), (3,3), (3,2), (4,4), and (4,3). Scalar model-coefficient setup
remains on the CPU, while frequency mapping, mode construction,
spherical-harmonic evaluation, and polarization assembly run on the active
Torch device without calling lalsimulation. The same kernels support both a
regular frequency grid and strictly increasing arbitrary-frequency sequences.

The native path is enabled by default under TorchScheme. Set
PYCBC_IMRPHENOMHM_NATIVE=0 to opt out. Unsupported waveform modifications
retain the lalsimulation fallback.
"""

from __future__ import annotations

import cmath
import math
from numbers import Integral
from typing import NamedTuple

from pycbc import lal_compat as lal
import numpy as np
import torch

from pycbc import pnutils, scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from pycbc.waveform.imrphenomd_torch import (
    AMP_fJoin_INS,
    PHI_fJoin_INS,
    _DEFAULT_ONLY_ORDER_KEYS,
    _IMRPhenDAmplitude,
    _IMRPhenDPhase,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _compute_amp_coeffs,
    _compute_phase_coeffs,
    _d_phi_mrd,
    _erad_rational_0815,
    _final_spin0815,
    _init_phi_prefactors,
    _is_default_order,
    _is_nonzero,
    _pi_powers,
    _powers,
    _subtract_3pn_ss,
)
from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing

_DEFAULT_MF_MAX = 0.5
_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
_MODE_SET = frozenset(_MODES)
_DEFAULT_OUTPUT_MODES = _MODES + tuple((ell, -emm) for ell, emm in _MODES)
_PHASE_SHIFT = (0.0, math.pi / 2.0, 0.0, -math.pi / 2.0, math.pi)


class _IMRPhenomHMInputs(NamedTuple):
    """Validated scalar inputs shared by regular and sequence evaluation."""

    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    f_ref: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    active_modes: tuple[tuple[int, int], ...]
    total_mass: float
    total_mass_seconds: float
    eta: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


class _IMRPhenomHMModel(NamedTuple):
    """Frequency-independent coefficients for one HM waveform."""

    inputs: _IMRPhenomHMInputs
    final_spin: float
    ringdown: dict[tuple[int, int], tuple[float, float]]
    amp_coefficients: object
    pn: object
    phi_pref: object
    reference_frequency: float
    phi0: float
    time_shift: float


def _polar_coefficient(magnitude: float, phase: float) -> complex:
    return magnitude * cmath.exp(1j * phase)


# Coefficients of the dimensionless fundamental Kerr QNM fits in ascending
# powers of kappa. These are the CW07102016 fits used by IMRPhenomHM.
_QNM_COEFFICIENTS = {
    (2, 2): (
        1.0,
        _polar_coefficient(1.557847, 2.903124),
        _polar_coefficient(1.95097051, 5.920970),
        _polar_coefficient(2.09971716, 2.760585),
        _polar_coefficient(1.41094660, 5.914340),
        _polar_coefficient(0.41063923, 2.795235),
    ),
    (2, 1): (
        _polar_coefficient(0.589113, 0.043525),
        _polar_coefficient(0.18896353, 2.289868),
        _polar_coefficient(1.15012965, 5.810057),
        _polar_coefficient(6.04585476, 2.741967),
        _polar_coefficient(11.12627777, 5.844130),
        _polar_coefficient(9.34711461, 2.669372),
        _polar_coefficient(3.03838318, 5.791518),
    ),
    (3, 3): (
        1.5,
        _polar_coefficient(2.095657, 2.964973),
        _polar_coefficient(2.46964352, 5.996734),
        _polar_coefficient(2.66552551, 2.817591),
        _polar_coefficient(1.75836443, 5.932693),
        _polar_coefficient(0.49905688, 2.781658),
    ),
    (3, 2): (
        _polar_coefficient(1.022464, 0.004870),
        _polar_coefficient(0.24731213, 0.665292),
        _polar_coefficient(1.70468239, 3.138283),
        _polar_coefficient(0.94604882, 0.163247),
        _polar_coefficient(1.53189884, 5.703573),
        _polar_coefficient(2.28052668, 2.685231),
        _polar_coefficient(0.92150314, 5.841704),
    ),
    (4, 4): (
        2.0,
        _polar_coefficient(2.658908, 3.002787),
        _polar_coefficient(2.97825567, 6.050955),
        _polar_coefficient(3.21842350, 2.877514),
        _polar_coefficient(2.12764967, 5.989669),
        _polar_coefficient(0.60338186, 2.830031),
    ),
    (4, 3): (
        1.5,
        _polar_coefficient(0.205046, 0.595328),
        _polar_coefficient(3.10333396, 3.016200),
        _polar_coefficient(4.23612166, 6.038842),
        _polar_coefficient(3.02890198, 2.826239),
        _polar_coefficient(0.90843949, 5.915164),
    ),
}


def _qnm_frequency(final_spin: float, ell: int, emm: int) -> complex:
    """Return the dimensionless complex QNM angular frequency."""

    alpha = math.log(2.0 - final_spin) / math.log(3.0)
    kappa = alpha ** (1.0 / (2.0 + ell - abs(emm)))
    coefficients = _QNM_COEFFICIENTS[(ell, abs(emm))]
    result = sum(value * kappa**power for power, value in enumerate(coefficients))
    return -result.conjugate() if emm < 0 else result


def _ringdown_frequencies(
    final_mass: float, final_spin: float
) -> dict[tuple[int, int], tuple[float, float]]:
    scale = 1.0 / (2.0 * math.pi * final_mass)
    frequencies = {}
    for mode in _MODES:
        omega = _qnm_frequency(final_spin, *mode)
        frequencies[mode] = (omega.real * scale, omega.imag * scale)
    return frequencies


def _active_modes(mode_array):
    if mode_array is None:
        return _MODES
    try:
        requested = set()
        for ell, emm in mode_array:
            mode = (int(ell), int(emm))
            if mode != (ell, emm):
                return None
            requested.add(mode)
    except (TypeError, ValueError, OverflowError):
        return None
    if not requested.issubset(_MODE_SET):
        return None
    return tuple(mode for mode in _MODES if mode in requested)


def _requested_modes(mode_array):
    """Return validated signed modes for the mode-by-mode interface."""

    if mode_array is None:
        return _DEFAULT_OUTPUT_MODES
    try:
        requested = []
        for ell, emm in mode_array:
            if not isinstance(ell, Integral) or not isinstance(emm, Integral):
                return None
            mode = (int(ell), int(emm))
            if (mode[0], abs(mode[1])) not in _MODE_SET:
                return None
            if mode not in requested:
                requested.append(mode)
    except (TypeError, ValueError, OverflowError):
        return None
    return tuple(requested)


def _mode_generation_params(params, modes):
    """Normalize signed output requests to the positive carrier families."""

    families = {(ell, abs(emm)) for ell, emm in modes}
    normalized = dict(params)
    normalized["mode_array"] = [mode for mode in _MODES if mode in families]
    # Mode output is source-frame data and must not depend on observer angles.
    normalized["inclination"] = 0.0
    normalized["long_asc_nodes"] = 0.0
    return normalized


def imrphenomhm_native_supported(params) -> bool:
    """Return whether the native implementation covers the parameters."""

    if params.get("approximant", "IMRPhenomHM") != "IMRPhenomHM":
        return False
    if _active_modes(params.get("mode_array")) is None:
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
        )
    ):
        return False
    if any(
        params.get(key) is not None
        for key in (
            "phenom_x_prec_version",
            "phenom_xp_convention",
            "phenom_xp_final_spin_mod",
        )
    ):
        return False
    return not params.get("numrel_data", "")


def imrphenomhm_modes_native_supported(params) -> bool:
    """Return whether signed FD-mode output is covered natively."""

    try:
        modes = _requested_modes(params.get("mode_array"))
        if modes is None:
            return False
        return imrphenomhm_native_supported(
            _mode_generation_params(params, modes)
        )
    except Exception:
        # Native support predicates are dispatch boundaries: malformed input
        # must select the public fallback rather than escaping from dispatch.
        return False


def imrphenomhm_sequence_native_supported(params) -> bool:
    """Return whether arbitrary-frequency HM generation is native."""

    return imrphenomhm_native_supported(params)


def _imrphenomhm_inputs(p, *, sequence=False):
    """Validate scalar inputs shared by both public sampling interfaces."""

    if not imrphenomhm_native_supported(p):
        raise ValueError(
            "IMRPhenomHM parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomHM requires TorchScheme")

    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z

    f_ref = float(p.get("f_ref", 0.0))
    distance = pnutils.megaparsecs_to_meters(float(p["distance"]))
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = 0.0 if sequence else float(p.get("long_asc_nodes", 0.0))
    active_modes = _active_modes(p.get("mode_array"))

    finite_parameters = (
        mass1,
        mass2,
        spin1z,
        spin2z,
        f_ref,
        distance,
        inclination,
        coa_phase,
        long_asc_nodes,
    )
    if not all(math.isfinite(value) for value in finite_parameters):
        raise ValueError("IMRPhenomHM parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomHM component masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("IMRPhenomHM aligned spins must be between -1 and 1")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomHM f_ref must be non-negative")
    if distance <= 0.0:
        raise ValueError("IMRPhenomHM distance must be positive")

    total_mass = mass1 + mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    eta = mass1 * mass2 / (total_mass * total_mass)
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return _IMRPhenomHMInputs(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        f_ref=f_ref,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        active_modes=active_modes,
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        eta=eta,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _mapping_parameters(
    mode: tuple[int, int],
    ringdown: dict[tuple[int, int], tuple[float, float]],
    *,
    amplitude: bool,
):
    _, emm = mode
    f_rd_22 = ringdown[(2, 2)][0]
    f_rd_lm = ringdown[mode][0]
    rho = f_rd_22 / f_rd_lm
    join_22 = AMP_fJoin_INS if amplitude else PHI_fJoin_INS
    f_inspiral = join_22 / rho
    f_ringdown = f_rd_lm

    slope_inspiral = 2.0 / emm
    value_inspiral = slope_inspiral * f_inspiral
    value_ringdown = f_rd_22
    slope_intermediate = (value_ringdown - value_inspiral) / (f_ringdown - f_inspiral)
    intercept_intermediate = value_inspiral - f_inspiral * slope_intermediate
    if amplitude:
        slope_ringdown = 1.0
        intercept_ringdown = -f_rd_lm + f_rd_22
    else:
        slope_ringdown = rho
        intercept_ringdown = 0.0

    return (
        slope_inspiral,
        0.0,
        slope_intermediate,
        intercept_intermediate,
        slope_ringdown,
        intercept_ringdown,
        f_inspiral,
        f_ringdown,
    )


def _map_frequency(
    frequency: torch.Tensor,
    parameters,
) -> torch.Tensor:
    ai, bi, am, bm, ar, br, fi, fr = parameters
    mapped_inspiral = ai * frequency + bi
    mapped_intermediate = am * frequency + bm
    mapped_ringdown = ar * frequency + br
    return torch.where(
        frequency <= fi,
        mapped_inspiral,
        torch.where(frequency <= fr, mapped_intermediate, mapped_ringdown),
    )


def _phase_at_scalar(
    frequency: float,
    coefficients,
    pn,
    phi_pref,
    rho: float,
    tau: float,
) -> float:
    value = _IMRPhenDPhase(
        np.asarray([frequency]),
        coefficients,
        pn,
        phi_pref,
        _pi_powers(),
        rho,
        tau,
    )
    return float(value[0])


def _mode_phase(
    frequency: torch.Tensor,
    mode: tuple[int, int],
    ringdown,
    coefficients,
    pn,
    phi_pref,
    rho: float,
    tau: float,
) -> torch.Tensor:
    ai, bi, am, bm, ar, br, fi, fr = _mapping_parameters(
        mode, ringdown, amplitude=False
    )
    mapped = _map_frequency(frequency, (ai, bi, am, bm, ar, br, fi, fr))
    base_phase = _IMRPhenDPhase(
        mapped,
        coefficients,
        pn,
        phi_pref,
        _pi_powers(),
        rho,
        tau,
    )

    phase_b_const = (
        _phase_at_scalar(am * fi + bm, coefficients, pn, phi_pref, rho, tau) / am
    )
    phase_c_const = (
        _phase_at_scalar(ar * fr + br, coefficients, pn, phi_pref, rho, tau) / ar
    )
    phase_ba_term = (
        _phase_at_scalar(ai * fi + bi, coefficients, pn, phi_pref, rho, tau) / ai
    )
    phase_c_join = (
        _phase_at_scalar(am * fr + bm, coefficients, pn, phi_pref, rho, tau) / am
        - phase_b_const
        + phase_ba_term
    )

    phase = torch.where(
        frequency <= fi,
        base_phase / ai,
        torch.where(
            frequency <= fr,
            base_phase / am - phase_b_const + phase_ba_term,
            base_phase / ar - phase_c_const + phase_c_join,
        ),
    )
    return phase + _PHASE_SHIFT[mode[1]]


def _pn_mode_amplitude(
    frequency: torch.Tensor,
    ell: int,
    emm: int,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
) -> torch.Tensor:
    total_mass = mass1 + mass2
    mass1 /= total_mass
    mass2 /= total_mass
    eta = mass1 * mass2
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * eta))
    spin_s = 0.5 * (spin1z + spin2z)
    spin_a = 0.5 * (spin1z - spin2z)
    velocity = torch.pow(2.0 * math.pi * frequency / emm, 1.0 / 3.0)
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity

    if (ell, emm) == (2, 2):
        h_abs = torch.ones_like(frequency)
    elif (ell, emm) == (2, 1):
        velocity4 = velocity2 * velocity2
        real = (
            velocity * delta
            - 1.5 * velocity2 * (spin_a + delta * spin_s)
            + velocity3 * delta * (335.0 / 672.0 + eta * 117.0 / 56.0)
            + velocity4
            * (
                spin_a * (3427.0 / 1344.0 - eta * 2101.0 / 336.0)
                + delta * spin_s * (3427.0 / 1344.0 - eta * 965.0 / 336.0)
                - delta * math.pi
            )
        )
        imag = velocity4 * delta * (-0.5 - 2.0 * math.log(2.0))
        h_abs = (math.sqrt(2.0) / 3.0) * torch.sqrt(real * real + imag * imag)
    elif (ell, emm) == (3, 3):
        h_abs = 0.75 * math.sqrt(5.0 / 7.0) * torch.abs(velocity * delta)
    elif (ell, emm) == (3, 2):
        h_abs = math.sqrt(5.0 / 7.0) / 3.0 * torch.abs(velocity2 * (1.0 - 3.0 * eta))
    elif (ell, emm) == (4, 4):
        h_abs = (
            4.0 / 9.0 * math.sqrt(10.0 / 7.0) * torch.abs(velocity2 * (1.0 - 3.0 * eta))
        )
    elif (ell, emm) == (4, 3):
        h_abs = (
            0.75
            * math.sqrt(3.0 / 35.0)
            * torch.abs(velocity3 * delta * (1.0 - 2.0 * eta))
        )
    else:  # pragma: no cover - guarded by native support
        raise ValueError(f"IMRPhenomHM does not model mode ({ell}, {emm})")

    return math.pi * math.sqrt(eta * 2.0 / 3.0) * torch.pow(velocity, -3.5) * h_abs


def _mode_amplitude(
    frequency: torch.Tensor,
    mode: tuple[int, int],
    ringdown,
    amp_coefficients,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
) -> torch.Tensor:
    ell, emm = mode
    mapped = _map_frequency(
        frequency, _mapping_parameters(mode, ringdown, amplitude=True)
    )
    amplitude = _IMRPhenDAmplitude(mapped, amp_coefficients, _powers(mapped))

    beta_term1 = _pn_mode_amplitude(frequency, ell, emm, mass1, mass2, spin1z, spin2z)
    beta_term2 = _pn_mode_amplitude(
        2.0 * frequency / emm,
        ell,
        emm,
        mass1,
        mass2,
        spin1z,
        spin2z,
    )
    safe_beta_term2 = torch.where(
        beta_term1 == 0.0, torch.ones_like(beta_term2), beta_term2
    )
    beta = torch.where(
        beta_term1 == 0.0,
        torch.zeros_like(beta_term1),
        beta_term1 / safe_beta_term2,
    )
    hm_term1 = _pn_mode_amplitude(mapped, ell, emm, mass1, mass2, spin1z, spin2z)
    hm_term2 = _pn_mode_amplitude(mapped, 2, 2, mass1, mass2, 0.0, 0.0)
    return amplitude * beta * hm_term1 / hm_term2


def _imrphenomhm_model(inputs, reference_frequency_hz, *, final_spin=None):
    """Build the frequency-independent HM coefficients.

    ``final_spin`` permits precessing callers to supply PhenomHM's
    in-plane-spin correction while sharing the rest of the carrier setup.
    """

    if isinstance(reference_frequency_hz, torch.Tensor):
        reference_frequency_hz = reference_frequency_hz.item()
    reference_frequency_hz = float(reference_frequency_hz)

    if final_spin is None:
        final_spin = _final_spin0815(
            inputs.eta,
            inputs.spin1z,
            inputs.spin2z,
        )
    else:
        final_spin = float(final_spin)
    if final_spin > 1.0:
        raise ValueError("IMRPhenomHM final spin exceeds one")
    final_mass = 1.0 - _erad_rational_0815(
        inputs.eta,
        inputs.spin1z,
        inputs.spin2z,
    )
    ringdown = _ringdown_frequencies(final_mass, final_spin)
    amp_coefficients = _compute_amp_coeffs(
        inputs.eta,
        inputs.spin1z,
        inputs.spin2z,
        final_spin,
    )
    pn = taylorf2_aligned_phasing(
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        spin_order=-1,
        tidal_order=-1,
        dchi={},
        qm_def1=0.0,
        qm_def2=0.0,
        lambda1=0.0,
        lambda2=0.0,
    )
    pn.v[6] -= (
        _subtract_3pn_ss(
            inputs.mass1,
            inputs.mass2,
            inputs.total_mass,
            inputs.eta,
            inputs.spin1z,
            inputs.spin2z,
        )
        * pn.v[0]
    )
    base_phase_coefficients = _compute_phase_coeffs(
        inputs.eta,
        inputs.spin1z,
        inputs.spin2z,
        final_spin,
        pn,
    )
    phi_pref = _init_phi_prefactors(
        base_phase_coefficients.sigma1,
        base_phase_coefficients.sigma2,
        base_phase_coefficients.sigma3,
        base_phase_coefficients.sigma4,
        pn,
        _pi_powers(),
    )
    reference_frequency = reference_frequency_hz * inputs.total_mass_seconds
    phase_at_reference = _phase_at_scalar(
        reference_frequency,
        base_phase_coefficients,
        pn,
        phi_pref,
        1.0,
        1.0,
    )
    phi0 = 0.5 * phase_at_reference + inputs.coa_phase
    time_shift = _d_phi_mrd(
        amp_coefficients.fmaxCalc,
        base_phase_coefficients.alpha1,
        base_phase_coefficients.alpha2,
        base_phase_coefficients.alpha3,
        base_phase_coefficients.alpha4,
        base_phase_coefficients.alpha5,
        base_phase_coefficients.fRD,
        base_phase_coefficients.fDM,
        base_phase_coefficients.eta_inv,
    )
    return _IMRPhenomHMModel(
        inputs=inputs,
        final_spin=final_spin,
        ringdown=ringdown,
        amp_coefficients=amp_coefficients,
        pn=pn,
        phi_pref=phi_pref,
        reference_frequency=reference_frequency,
        phi0=phi0,
        time_shift=time_shift,
    )


def _imrphenomhm_mode_samples(model, frequency, mode):
    """Evaluate one unscaled positive-m co-precessing HM mode."""

    ell, emm = mode
    inputs = model.inputs
    f_rd_22, f_damp_22 = model.ringdown[(2, 2)]
    f_rd_lm, f_damp_lm = model.ringdown[mode]
    rho = f_rd_22 / f_rd_lm
    tau = f_damp_lm / f_damp_22
    phase_coefficients = _compute_phase_coeffs(
        inputs.eta,
        inputs.spin1z,
        inputs.spin2z,
        model.final_spin,
        model.pn,
        rho,
        tau,
    )
    mode_phase = _mode_phase(
        frequency,
        mode,
        model.ringdown,
        phase_coefficients,
        model.pn,
        model.phi_pref,
        rho,
        tau,
    )
    mode_amplitude = _mode_amplitude(
        frequency,
        mode,
        model.ringdown,
        model.amp_coefficients,
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
    )
    total_phase = (
        -model.time_shift * (frequency - model.reference_frequency)
        + mode_phase
        - emm * model.phi0
    )
    return torch.polar(mode_amplitude, -total_phase).to(inputs.complex_dtype)


def _imrphenomhm_polarizations(model, frequency):
    """Evaluate both polarizations at dimensionless frequencies."""

    inputs = model.inputs
    hp = torch.zeros(
        frequency.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    hc = torch.zeros_like(hp)

    for ell, emm in inputs.active_modes:
        hlm = _imrphenomhm_mode_samples(model, frequency, (ell, emm))

        y_positive = spin_weighted_spherical_harmonic(
            inputs.inclination,
            0.0,
            -2,
            ell,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        y_negative_conjugate = spin_weighted_spherical_harmonic(
            inputs.inclination,
            0.0,
            -2,
            ell,
            -emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ).conj()
        parity = (-1) ** ell
        factor_plus = 0.5 * (y_positive + parity * y_negative_conjugate)
        factor_cross = -0.5j * (y_positive - parity * y_negative_conjugate)
        hp += factor_plus * hlm
        hc += factor_cross * hlm

    amplitude_scale = _strain_amplitude_scale(inputs)
    hp *= amplitude_scale
    hc *= amplitude_scale

    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    plus = cos_nodes * hp + sin_nodes * hc
    cross = cos_nodes * hc - sin_nodes * hp
    return plus, cross


def _strain_amplitude_scale(inputs):
    """Convert dimensionless HM mode amplitudes to strain."""

    return (
        inputs.total_mass
        * lal.MRSUN_SI
        * inputs.total_mass_seconds
        / inputs.distance
    )


def _regular_frequency_grid(p, inputs):
    """Validate and construct the regular one-sided HM frequency layout."""

    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomHM frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomHM delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomHM f_final must be non-negative")

    layout_f_max = (
        f_final if f_final > 0.0 else _DEFAULT_MF_MAX / inputs.total_mass_seconds
    )
    if layout_f_max < f_lower:
        raise ValueError("IMRPhenomHM f_final must not be below f_lower")

    frequency_bins = int(layout_f_max / delta_f)
    npts = 1
    if frequency_bins:
        npts += 1 << (frequency_bins - 1).bit_length()
    first_bin = math.ceil(f_lower / delta_f)
    stop_bin = min(math.ceil(layout_f_max / delta_f), npts)
    bins = torch.arange(first_bin, stop_bin, device=inputs.device)
    frequencies_hz = bins.to(dtype=inputs.real_dtype) * delta_f
    return delta_f, npts, bins, frequencies_hz


def _series_from_active_mode(inputs, samples, delta_f, npts, bins):
    """Place active mode samples into the regular one-sided layout."""

    values = torch.zeros(
        npts,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    values[bins] = samples
    return FrequencySeries(
        TorchArrayData(values),
        delta_f=delta_f,
        epoch=-1.0 / delta_f,
        copy=False,
    )


def imrphenomhm_modes_torch(**p):
    """Generate signed IMRPhenomHM FD modes with Torch.

    Each value is ``(u_lm, v_lm)``, the Fourier transforms of the real and
    imaginary parts of the complex time-domain mode.  LAL models only the six
    positive-m families; negative-m outputs follow equatorial symmetry.
    """

    if not imrphenomhm_modes_native_supported(p):
        raise ValueError(
            "IMRPhenomHM mode parameters are not supported by the native "
            "Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomHM modes require TorchScheme")

    modes = _requested_modes(p.get("mode_array"))
    if not modes:
        return {}
    inputs = _imrphenomhm_inputs(_mode_generation_params(p, modes))
    delta_f, npts, bins, frequencies_hz = _regular_frequency_grid(p, inputs)
    reference_frequency_hz = (
        inputs.f_ref if inputs.f_ref > 0.0 else float(p["f_lower"])
    )
    model = _imrphenomhm_model(inputs, reference_frequency_hz)
    dimensionless_frequencies = frequencies_hz * inputs.total_mass_seconds
    amplitude_scale = _strain_amplitude_scale(inputs)
    active = {
        mode: amplitude_scale
        * _imrphenomhm_mode_samples(model, dimensionless_frequencies, mode)
        for mode in inputs.active_modes
    }

    result = {}
    for ell, emm in modes:
        samples = active[ell, abs(emm)]
        if emm < 0:
            samples = ((-1) ** ell) * samples
        hlm = _series_from_active_mode(
            inputs,
            samples,
            delta_f,
            npts,
            bins,
        )
        ulm = 0.5 * hlm
        vlm = (0.5j if emm > 0 else -0.5j) * hlm
        result[ell, emm] = (ulm, vlm)
    return result


def imrphenomhm_fd_torch(**p):
    """Generate IMRPhenomHM plus/cross polarizations with Torch."""

    inputs = _imrphenomhm_inputs(p)
    delta_f, npts, bins, frequencies_hz = _regular_frequency_grid(p, inputs)

    reference_frequency_hz = (
        inputs.f_ref if inputs.f_ref > 0.0 else float(p["f_lower"])
    )
    model = _imrphenomhm_model(inputs, reference_frequency_hz)
    active_plus, active_cross = _imrphenomhm_polarizations(
        model,
        frequencies_hz * inputs.total_mass_seconds,
    )
    plus = torch.zeros(
        npts,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    plus[bins] = active_plus
    cross[bins] = active_cross

    epoch = -1.0 / delta_f
    return (
        FrequencySeries(TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _sequence_frequencies(sample_points, inputs):
    """Return a validated increasing sequence on the active Torch device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomHM sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomHM sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomHM sample_points must be positive")
    if frequencies.numel() > 1 and bool(torch.any(frequencies[1:] <= frequencies[:-1])):
        raise ValueError("IMRPhenomHM sample_points must be strictly increasing")
    return frequencies


def imrphenomhm_fd_sequence_torch(**p):
    """Evaluate IMRPhenomHM at arbitrary increasing frequencies."""

    inputs = _imrphenomhm_inputs(p, sequence=True)
    frequencies_hz = _sequence_frequencies(p["sample_points"], inputs)
    reference_frequency_hz = inputs.f_ref if inputs.f_ref > 0.0 else frequencies_hz[0]
    model = _imrphenomhm_model(inputs, reference_frequency_hz)
    plus, cross = _imrphenomhm_polarizations(
        model,
        frequencies_hz * inputs.total_mass_seconds,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )
