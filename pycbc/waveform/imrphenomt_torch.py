# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native dominant-mode time-domain IMRPhenomT waveform.

This module assembles the ``(2, +/-2)`` modes from the Torch ports of the
IMRPhenomT phase, amplitude, and calibration fits.  Waveform evaluation and
polarization projection remain on the active Torch device.  CPU and CUDA also
evaluate the two scalar frequency-time roots there; MPS uses Torch CPU
float64 for those roots because MPS is limited to float32.  Only the output
length and epoch are copied to the host, as required by the PyCBC
:class:`~pycbc.types.TimeSeries` interface.

The implementation follows LALSuite 7.26.1.  In particular, it preserves the
one-sample offset of the discrete inspiral/merger phase boundary and the
fitted early-time variable used by LAL's frequency-to-time root finder.

The public port is opt-in through ``PYCBC_IMRPHENOMT_NATIVE=1`` or the global
``PYCBC_TORCH_NATIVE_PORTS=1`` switch. Unsupported waveform modifications
retain the lalsimulation fallback.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pycbc import lal_compat as lal
import torch

import pycbc.scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _is_default_order,
    _is_nonzero,
)

from ._spherical_harmonics_torch import (
    spin_weighted_spherical_harmonic,
)
from ._torch_jax import torch_context
from .imrphenomt_amplitude_torch import (
    amplitude22,
    build_amplitude22_coefficients,
)
from .imrphenomt_phase_torch import (
    IMRPhenomTPhase22Coefficients,
    build_phase22_coefficients,
    inspiral_omega,
    inspiral_phase,
    merger_omega,
    merger_phase,
    ringdown_omega,
    ringdown_phase,
)
from .imrphenomx_utils_torch import get_remnant_fMs

_PI = math.pi
_END_TIME_M = 500.0
_ROOT_LOWER_TIME_M = -1.0e9
_ROOT_MAX_ITERATIONS = 128
_ROOT_RELATIVE_TOLERANCE = 1.0e-4
_MPC_SI = 1.0e6 * lal.PC_SI


class _IMRPhenomTInputs(NamedTuple):
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    delta_t: float
    f_lower: float
    f_ref: float


class _IMRPhenomTCoreSetup(NamedTuple):
    """Parsed binary and aligned remnant state used to build a T carrier."""

    inputs: _IMRPhenomTInputs
    binary: torch.Tensor
    final_mass: torch.Tensor
    final_spin: torch.Tensor


class _IMRPhenomTCore(NamedTuple):
    """Common binary, time-grid, and carrier-phase state for the T family."""

    inputs: _IMRPhenomTInputs
    binary: torch.Tensor
    final_mass: torch.Tensor
    final_spin: torch.Tensor
    final_spin_prec: torch.Tensor
    phase_coefficients: IMRPhenomTPhase22Coefficients
    reference_time_m: torch.Tensor
    time_m: torch.Tensor
    x_orbital: torch.Tensor
    phase22: torch.Tensor
    reference_phase22: torch.Tensor
    amplitude_factor: torch.Tensor
    epoch: float


def _imrphenomt_family_native_supported(
    parameters,
    approximant,
    *,
    allow_transverse_spins=False,
):
    """Check common circular IMRPhenomT-family options."""
    if parameters.get("approximant", approximant) != approximant:
        return False
    if any(
        not _is_default_order(parameters.get(key, -1))
        for key in (
            *_DEFAULT_ONLY_ORDER_KEYS,
            "phase_order",
            "amplitude_order",
        )
    ):
        return False
    if any(
        _is_nonzero(parameters.get(key, 0.0))
        for key in (
            (() if allow_transverse_spins else _TRANSVERSE_SPIN_KEYS)
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
            )
        )
    ):
        return False
    if parameters.get("numrel_data", ""):
        return False
    return True


def imrphenomt_native_supported(parameters):
    """Return whether the native dominant-mode port covers ``parameters``.

    Options outside the aligned-spin, circular, default-order model retain
    the public lalsimulation path instead of being silently ignored.
    """
    return (
        _imrphenomt_family_native_supported(parameters, "IMRPhenomT")
        and parameters.get("mode_array") is None
    )


def _finite_float(parameters, name, default=None):
    value = parameters.get(name, default)
    if value is None:
        raise ValueError(f"IMRPhenomT requires {name}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"IMRPhenomT requires a scalar {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"IMRPhenomT requires finite {name}")
    return value


def _parse_inputs(parameters):
    mass1 = _finite_float(parameters, "mass1")
    mass2 = _finite_float(parameters, "mass2")
    spin1z = _finite_float(parameters, "spin1z", 0.0)
    spin2z = _finite_float(parameters, "spin2z", 0.0)
    distance = _finite_float(parameters, "distance", 1.0)
    inclination = _finite_float(parameters, "inclination", 0.0)
    coa_phase = _finite_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _finite_float(parameters, "long_asc_nodes", 0.0)
    delta_t = _finite_float(parameters, "delta_t")
    f_lower = _finite_float(parameters, "f_lower")
    f_ref = _finite_float(parameters, "f_ref", 0.0)

    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomT masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("IMRPhenomT aligned spins must lie in [-1, 1]")
    if distance <= 0.0:
        raise ValueError("IMRPhenomT distance must be positive")
    if delta_t <= 0.0:
        raise ValueError("IMRPhenomT delta_t must be positive")
    if f_lower <= 0.0:
        raise ValueError("IMRPhenomT f_lower must be positive")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomT f_ref must be nonnegative")

    # LAL's model convention always labels body 1 as the larger body.
    if mass1 < mass2:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
    if f_ref == 0.0:
        f_ref = f_lower

    return _IMRPhenomTInputs(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        delta_t=delta_t,
        f_lower=f_lower,
        f_ref=f_ref,
    )


def _estimated_frequency_root_width_samples(inputs):
    """Estimate LAL's allowed frequency-root bracket width in samples.

    The leading-order chirp time captures the strong mass, frequency, cadence,
    and mass-ratio scaling. The aligned-spin factor conservatively bounds the
    exact IMRPhenomT root width across the default-on readiness audit without
    repeating coefficient construction and root solving during dispatch.
    """
    total_mass = inputs.mass1 + inputs.mass2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    eta = inputs.mass1 * inputs.mass2 / total_mass**2
    lowest_frequency = min(inputs.f_lower, inputs.f_ref)
    dimensionless_frequency = total_mass_seconds * lowest_frequency
    newtonian_time_m = (
        5.0
        / (256.0 * eta)
        * (_PI * dimensionless_frequency) ** (-8.0 / 3.0)
    )
    chi_effective = (
        inputs.mass1 * inputs.spin1z + inputs.mass2 * inputs.spin2z
    ) / total_mass
    spin_safety_factor = 1.06 + 0.15 * max(chi_effective, 0.0)
    delta_time_m = inputs.delta_t / total_mass_seconds
    return (
        _ROOT_RELATIVE_TOLERANCE
        * newtonian_time_m
        / delta_time_m
        * spin_safety_factor
    )


def _target_device_dtype():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native IMRPhenomT requires an active TorchScheme")
    device = state.torch_device
    # MPS linear algebra is substantially more reliable in float32, and MPS
    # does not support float64. CPU and CUDA retain LAL's double precision.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return device, dtype


def _safe_inspiral_time(t):
    return torch.where(t < 0.0, t, -torch.ones_like(t))


def _discrete_omega22(
    t,
    delta_time_m,
    coefficients: IMRPhenomTPhase22Coefficients,
    *,
    root_theta=False,
):
    """Evaluate LAL's cadence-dependent frequency construction."""
    t = torch.as_tensor(
        t, device=coefficients.eta.device, dtype=coefficients.eta.dtype
    )
    safe_time = _safe_inspiral_time(t)
    tiny = torch.finfo(t.dtype).tiny
    late_argument = torch.clamp(
        -coefficients.eta * safe_time / 5.0, min=tiny
    )
    theta = late_argument ** (-1.0 / 8.0)
    if root_theta:
        early_argument = torch.clamp(
            coefficients.eta * (coefficients.tt0 - safe_time) / 5.0,
            min=tiny,
        )
        theta = torch.where(
            t < coefficients.t_early,
            early_argument ** (-1.0 / 8.0),
            theta,
        )

    inspiral = inspiral_omega(theta, coefficients)
    merger = coefficients.omega_ring * (
        1.0 - merger_omega(t, coefficients)
    )
    ringdown_time = torch.where(t > 0.0, t, torch.zeros_like(t))
    ringdown = ringdown_omega(ringdown_time, coefficients)
    return torch.where(
        t < coefficients.t_cut - delta_time_m,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )


def _discrete_phase22(
    t,
    delta_time_m,
    coefficients: IMRPhenomTPhase22Coefficients,
):
    """Evaluate LAL's cadence-dependent default phase construction."""
    t = torch.as_tensor(
        t, device=coefficients.eta.device, dtype=coefficients.eta.dtype
    )
    safe_time = _safe_inspiral_time(t)
    thetabar = torch.clamp(
        -coefficients.eta * safe_time,
        min=torch.finfo(t.dtype).tiny,
    ) ** (-1.0 / 8.0)
    inspiral = inspiral_phase(safe_time, thetabar, coefficients)
    merger = merger_phase(t, coefficients)
    ringdown_time = torch.where(t > 0.0, t, torch.zeros_like(t))
    ringdown = ringdown_phase(ringdown_time, coefficients)
    return torch.where(
        t < coefficients.t_cut - delta_time_m,
        inspiral,
        torch.where(t > 0.0, ringdown, merger),
    )


def _time_of_frequency(
    dimensionless_frequency,
    delta_time_m,
    coefficients,
    *,
    label,
):
    """Find the geometric time of a frequency with Torch Brent iteration.

    LALSuite uses GSL's Brent solver and stops when the relative bracket width
    reaches ``1e-4``. Reproducing both choices matters: returning a more
    accurate root can shift the discrete waveform grid by a fraction of a
    sample compared with LAL.
    """
    target = 2.0 * _PI * dimensionless_frequency
    lower = coefficients.eta.new_tensor(_ROOT_LOWER_TIME_M)
    upper = coefficients.eta.new_tensor(_END_TIME_M)

    def residual(time):
        return target - _discrete_omega22(
            time,
            delta_time_m,
            coefficients,
            root_theta=True,
        )

    lower_residual = residual(lower)
    upper_residual = residual(upper)
    bracketed = bool(
        ((lower_residual >= 0.0) & (upper_residual <= 0.0)).item()
    )
    if not bracketed:
        peak_hz_m = coefficients.omega_peak / (2.0 * _PI)
        raise ValueError(
            f"IMRPhenomT {label} frequency is outside the model's "
            f"root-finding range (peak Mf={float(peak_hz_m.item()):.8g})"
        )

    # State and update order follow GSL 2.8's roots/brent.c. Tensor masks keep
    # the iteration on-device; converged state is frozen while the fixed-size
    # loop completes.
    a = lower
    b = upper
    c = upper
    fa = lower_residual
    fb = upper_residual
    fc = upper_residual
    d = upper - lower
    e = upper - lower
    root = 0.5 * (lower + upper)
    active = torch.ones((), dtype=torch.bool, device=root.device)
    converged = torch.zeros_like(active)
    epsilon = torch.finfo(root.dtype).eps

    for _ in range(_ROOT_MAX_ITERATIONS):
        same_sign = ((fb < 0.0) & (fc < 0.0)) | (
            (fb > 0.0) & (fc > 0.0)
        )
        ac_equal = same_sign
        candidate_c = torch.where(same_sign, a, c)
        candidate_fc = torch.where(same_sign, fa, fc)
        candidate_d = torch.where(same_sign, b - a, d)
        candidate_e = torch.where(same_sign, b - a, e)

        swap = torch.abs(candidate_fc) < torch.abs(fb)
        old_b = b
        old_fb = fb
        candidate_a = torch.where(swap, old_b, a)
        candidate_fa = torch.where(swap, old_fb, fa)
        candidate_b = torch.where(swap, candidate_c, b)
        candidate_fb = torch.where(swap, candidate_fc, fb)
        candidate_c = torch.where(swap, old_b, candidate_c)
        candidate_fc = torch.where(swap, old_fb, candidate_fc)
        ac_equal = ac_equal | swap

        tolerance = 0.5 * epsilon * torch.abs(candidate_b)
        midpoint = 0.5 * (candidate_c - candidate_b)
        solver_done = (candidate_fb == 0.0) | (
            torch.abs(midpoint) <= tolerance
        )

        use_bisection = (torch.abs(candidate_e) < tolerance) | (
            torch.abs(candidate_fa) <= torch.abs(candidate_fb)
        )
        ratio_s = candidate_fb / candidate_fa
        ratio_q = candidate_fa / candidate_fc
        ratio_r = candidate_fb / candidate_fc
        interpolation_p = torch.where(
            ac_equal,
            2.0 * midpoint * ratio_s,
            ratio_s
            * (
                2.0
                * midpoint
                * ratio_q
                * (ratio_q - ratio_r)
                - (candidate_b - candidate_a) * (ratio_r - 1.0)
            ),
        )
        interpolation_q = torch.where(
            ac_equal,
            1.0 - ratio_s,
            (ratio_q - 1.0) * (ratio_r - 1.0) * (ratio_s - 1.0),
        )
        interpolation_q = torch.where(
            interpolation_p > 0.0,
            -interpolation_q,
            interpolation_q,
        )
        interpolation_p = torch.abs(interpolation_p)
        accept_interpolation = (
            2.0 * interpolation_p
            < torch.minimum(
                3.0 * midpoint * interpolation_q
                - torch.abs(tolerance * interpolation_q),
                torch.abs(candidate_e * interpolation_q),
            )
        )
        interpolate = (~use_bisection) & accept_interpolation
        next_d = torch.where(
            interpolate, interpolation_p / interpolation_q, midpoint
        )
        next_e = torch.where(
            interpolate,
            candidate_d,
            midpoint,
        )

        next_a = candidate_b
        next_fa = candidate_fb
        step = torch.where(
            torch.abs(next_d) > tolerance,
            next_d,
            torch.where(midpoint > 0.0, tolerance, -tolerance),
        )
        next_b = candidate_b + step
        next_fb = residual(next_b)

        do_step = active & ~solver_done
        a = torch.where(do_step, next_a, candidate_a)
        fa = torch.where(do_step, next_fa, candidate_fa)
        b = torch.where(do_step, next_b, candidate_b)
        fb = torch.where(do_step, next_fb, candidate_fb)
        c = candidate_c
        fc = candidate_fc
        d = torch.where(do_step, next_d, candidate_d)
        e = torch.where(do_step, next_e, candidate_e)

        bound_c = torch.where(
            ((fb < 0.0) & (fc < 0.0))
            | ((fb > 0.0) & (fc > 0.0)),
            a,
            c,
        )
        x_lower = torch.minimum(b, bound_c)
        x_upper = torch.maximum(b, bound_c)
        same_side = ((x_lower > 0.0) & (x_upper > 0.0)) | (
            (x_lower < 0.0) & (x_upper < 0.0)
        )
        minimum_absolute_bound = torch.where(
            same_side,
            torch.minimum(torch.abs(x_lower), torch.abs(x_upper)),
            torch.zeros_like(x_lower),
        )
        interval_done = (
            torch.abs(x_upper - x_lower)
            < _ROOT_RELATIVE_TOLERANCE * minimum_absolute_bound
        )
        candidate_root = torch.where(solver_done, candidate_b, b)
        root = torch.where(active, candidate_root, root)
        converged = converged | (active & (solver_done | interval_done))
        active = active & ~(solver_done | interval_done)

    if not bool(converged.item()):
        raise RuntimeError(f"IMRPhenomT {label} frequency root did not converge")
    return root


def _double_precision_frequency_times(inputs, final_spin_prec_override):
    """Build the two scalar MPS roots with Torch CPU float64.

    MPS is limited to float32. Near flat portions of the frequency evolution,
    coefficient roundoff can otherwise move the starting root by several
    samples. The bulk waveform still evaluates on MPS; only these scalar roots
    and their coefficient setup use Torch CPU tensors.
    """
    binary = torch.tensor(
        (inputs.mass1, inputs.mass2, inputs.spin1z, inputs.spin2z),
        dtype=torch.float64,
    )
    mass1, mass2, spin1z, spin2z = binary.unbind()
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    delta_time_m = inputs.delta_t / total_mass_seconds
    with torch_context(binary):
        remnant = get_remnant_fMs(mass1, mass2, spin1z, spin2z)
    final_mass = 1.0 - remnant.radiated_energy
    final_spin = remnant.final_spin
    if final_spin_prec_override is None:
        final_spin_prec = final_spin
    else:
        final_spin_prec = torch.as_tensor(
            float(final_spin_prec_override.detach().cpu()),
            dtype=torch.float64,
        )
    coefficients = build_phase22_coefficients(
        mass1 * mass2 / (mass1 + mass2) ** 2,
        spin1z,
        spin2z,
        final_mass,
        final_spin,
        final_spin_prec=final_spin_prec,
    )
    minimum_time = _time_of_frequency(
        total_mass_seconds * inputs.f_lower,
        delta_time_m,
        coefficients,
        label="starting",
    )
    if inputs.f_ref == inputs.f_lower:
        reference_time = minimum_time
    else:
        reference_time = _time_of_frequency(
            total_mass_seconds * inputs.f_ref,
            delta_time_m,
            coefficients,
            label="reference",
        )
    return minimum_time, reference_time, delta_time_m, total_mass_seconds


def _prepare_imrphenomt_core(parameters):
    """Parse a T-family binary and construct its aligned remnant state."""
    inputs = _parse_inputs(parameters)
    device, dtype = _target_device_dtype()

    binary = torch.tensor(
        (inputs.mass1, inputs.mass2, inputs.spin1z, inputs.spin2z),
        device=device,
        dtype=dtype,
    )
    mass1, mass2, spin1z, spin2z = binary.unbind()
    with torch_context(binary):
        remnant = get_remnant_fMs(mass1, mass2, spin1z, spin2z)
    return _IMRPhenomTCoreSetup(
        inputs=inputs,
        binary=binary,
        final_mass=1.0 - remnant.radiated_energy,
        final_spin=remnant.final_spin,
    )


def _build_imrphenomt_core_from_setup(
    setup,
    *,
    final_spin_prec_override=None,
):
    """Construct a T-family carrier from prepared binary/remnant state.

    ``final_spin_prec_override`` supplies TP/TPHM's distinct precessing
    remnant spin.  The aligned remnant spin continues to control the merger
    carrier, matching LALSuite's default merger reconstruction.
    """
    if not isinstance(setup, _IMRPhenomTCoreSetup):
        raise TypeError("IMRPhenomT requires prepared core state")
    inputs = setup.inputs
    binary = setup.binary
    device = binary.device
    dtype = binary.dtype
    mass1, mass2, spin1z, spin2z = binary.unbind()
    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    total_mass_seconds = total_mass * lal.MTSUN_SI
    delta_time_m = inputs.delta_t / total_mass_seconds

    if final_spin_prec_override is not None:
        final_spin_prec_override = torch.as_tensor(
            final_spin_prec_override,
            device=device,
            dtype=dtype,
        )
        if final_spin_prec_override.numel() != 1:
            raise ValueError(
                "IMRPhenomT precessing final spin override must be scalar"
            )
        final_spin_prec_override = final_spin_prec_override.reshape(())
        if not bool(torch.isfinite(final_spin_prec_override).detach().cpu()):
            raise ValueError(
                "IMRPhenomT precessing final spin override must be finite"
            )
        if float(torch.abs(final_spin_prec_override).detach().cpu()) > 1.0:
            raise ValueError(
                "IMRPhenomT precessing final spin override must lie in [-1, 1]"
            )

    final_mass = setup.final_mass
    final_spin = setup.final_spin
    final_spin_prec = (
        final_spin
        if final_spin_prec_override is None
        else final_spin_prec_override
    )
    phase_coefficients = build_phase22_coefficients(
        eta,
        spin1z,
        spin2z,
        final_mass,
        final_spin,
        final_spin_prec=final_spin_prec,
    )

    if device.type == "mps":
        (
            metadata_minimum_time,
            metadata_reference_time,
            metadata_delta_time_m,
            metadata_total_mass_seconds,
        ) = _double_precision_frequency_times(
            inputs,
            final_spin_prec_override,
        )
        minimum_time = metadata_minimum_time.to(device=device, dtype=dtype)
        reference_time = metadata_reference_time.to(
            device=device,
            dtype=dtype,
        )
    else:
        minimum_mf = total_mass_seconds * inputs.f_lower
        reference_mf = total_mass_seconds * inputs.f_ref
        minimum_time = _time_of_frequency(
            minimum_mf,
            delta_time_m,
            phase_coefficients,
            label="starting",
        )
        if inputs.f_ref == inputs.f_lower:
            reference_time = minimum_time
        else:
            reference_time = _time_of_frequency(
                reference_mf,
                delta_time_m,
                phase_coefficients,
                label="reference",
            )
        metadata_minimum_time = minimum_time
        metadata_delta_time_m = delta_time_m
        metadata_total_mass_seconds = total_mass_seconds

    length = int(
        torch.floor(
            (_END_TIME_M - metadata_minimum_time) / metadata_delta_time_m
        ).item()
    )
    if length <= 0:
        raise ValueError("IMRPhenomT waveform has no samples")
    time_m = minimum_time + torch.arange(
        length, device=device, dtype=dtype
    ) * delta_time_m

    omega = _discrete_omega22(
        time_m, delta_time_m, phase_coefficients
    )
    x_orbital = torch.pow(0.5 * omega, 2.0 / 3.0)
    carrier_phase = _discrete_phase22(
        time_m, delta_time_m, phase_coefficients
    )
    reference_phase = _discrete_phase22(
        reference_time, delta_time_m, phase_coefficients
    )
    amplitude_factor = (
        total_mass * lal.MRSUN_SI / (inputs.distance * _MPC_SI)
    )
    epoch = float(
        (metadata_minimum_time * metadata_total_mass_seconds).item()
    )
    return _IMRPhenomTCore(
        inputs=inputs,
        binary=binary,
        final_mass=final_mass,
        final_spin=final_spin,
        final_spin_prec=final_spin_prec,
        phase_coefficients=phase_coefficients,
        reference_time_m=reference_time,
        time_m=time_m,
        x_orbital=x_orbital,
        phase22=carrier_phase,
        reference_phase22=reference_phase,
        amplitude_factor=amplitude_factor,
        epoch=epoch,
    )


def _build_imrphenomt_core(parameters, *, final_spin_prec_override=None):
    """Construct common IMRPhenomT-family state on the active Torch device."""
    return _build_imrphenomt_core_from_setup(
        _prepare_imrphenomt_core(parameters),
        final_spin_prec_override=final_spin_prec_override,
    )


def imrphenomt_td_torch(**parameters):
    """Generate native dominant-mode IMRPhenomT polarizations with Torch."""
    core = _build_imrphenomt_core(parameters)
    inputs = core.inputs
    _, _, spin1z, spin2z = core.binary.unbind()
    device = core.binary.device
    dtype = core.binary.dtype
    amplitude_coefficients = build_amplitude22_coefficients(
        spin1z,
        spin2z,
        core.final_mass,
        core.final_spin,
        core.phase_coefficients,
    )
    amplitude = torch.abs(
        amplitude22(core.time_m, core.x_orbital, amplitude_coefficients)
    )
    phase = core.phase22 - core.reference_phase22
    h22 = core.amplitude_factor * amplitude * torch.complex(
        torch.cos(phase), -torch.sin(phase)
    )

    harmonic_phi = 0.5 * _PI - inputs.coa_phase
    y22 = spin_weighted_spherical_harmonic(
        inputs.inclination,
        harmonic_phi,
        -2,
        2,
        2,
        dtype=dtype,
        device=device,
    )
    y2m2 = spin_weighted_spherical_harmonic(
        inputs.inclination,
        harmonic_phi,
        -2,
        2,
        -2,
        dtype=dtype,
        device=device,
    )
    strain = y22 * h22 + y2m2 * h22.conj()
    plus = strain.real
    cross = -strain.imag
    if inputs.long_asc_nodes:
        rotation = 2.0 * inputs.long_asc_nodes
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        plus, cross = (
            cosine * plus + sine * cross,
            cosine * cross - sine * plus,
        )

    return (
        TimeSeries(
            TorchArrayData(plus),
            delta_t=inputs.delta_t,
            epoch=core.epoch,
            copy=False,
        ),
        TimeSeries(
            TorchArrayData(cross),
            delta_t=inputs.delta_t,
            epoch=core.epoch,
            copy=False,
        ),
    )


__all__ = ["imrphenomt_native_supported", "imrphenomt_td_torch"]
