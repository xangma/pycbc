# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native SpinTaylor Fourier waveforms and their shared SUA core.

This module implements the shifted-uniform-asymptotics (SUA) Fourier
assembler shared by LAL's ``SpinTaylorT4Fourier`` and
``SpinTaylorT5Fourier`` approximants.  The dynamics selector keeps the
expensive and subtle SUA machinery shared by both models.

The orbit is retained at accepted time-domain RKF45 knots.  Natural cubic
splines are constructed directly on that irregular grid, avoiding an
invented time cadence (the public Fourier interface has no ``delta_t``).

The public ports are deliberately opt-in, CPU-only regular-FD implementations
with a conservative BBH, restricted-amplitude contract ending no later than
the Schwarzschild ISCO.  Arbitrary-frequency, time-domain, mode, matter, and
accelerator paths retain their existing behavior instead of silently using an
incomplete native implementation.

In LALSuite 7.26.1,
``XLALSimInspiralSpinTaylorPNEvolveOrbitIrregularIntervals`` passes
``&paramsT4``/``&paramsT5`` (the address of a local pointer) to the ODE driver
instead of ``paramsT4``/``paramsT5`` (the coefficient structure).  The
resulting undefined dynamics make an unpatched build unsuitable as an
end-to-end numeric oracle.  An isolated build changing only those pointer
assignments validates this core against both Fourier models; representative
results from that corrected build are frozen in the tests.

The fixed-shape GSL-compatible RKF45 step is traced once per waveform when a
safety probe passes and otherwise falls back to eager execution.  Low-mass
throughput and accelerator coverage still need production validation, so the
ports remain opt-in on CPU.  The reference core favors the original 1e-12
solver tolerances over throughput.
"""

from __future__ import annotations

import math
import operator
import warnings
from dataclasses import dataclass

import torch

import pycbc.scheme as _scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
)
from pycbc.waveform.constants import _MRSUN_SI, _MTSUN_SI, _PC_SI
from pycbc.waveform.imrphenomx_spintaylor_torch import (
    _SpinTaylorPhysicalBoundary,
    _check_spintaylor_physical_state,
    spintaylor_internal_spins,
    spintaylor_t4_rhs,
    spintaylor_t5_rhs,
)
from pycbc.waveform.rkf45_torch import rkf45_step
from pycbc.waveform.spintaylor_torch import (
    _NL_TIDAL_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _parameter_float,
    _rotate_y,
    spintaylor_polarizations_from_orbit,
)


_C_SI = 299792458.0
_MPC_SI = 1.0e6 * _PC_SI
_KMAX = 3
_PHASE_SAMPLES = 16
_SUA_COEFFICIENTS = (
    complex(-1.0 / 6.0, 17.0 / 18.0),
    complex(13.0 / 16.0, -7.0 / 16.0),
    complex(-1.0 / 4.0, -1.0 / 20.0),
    complex(1.0 / 48.0, 11.0 / 720.0),
)
_DYNAMICS = {
    "SpinTaylorT4": spintaylor_t4_rhs,
    "SpinTaylorT5": spintaylor_t5_rhs,
}
_SUPPORTED_PHASE_ORDERS = frozenset((-1, 7, 8))
_SUPPORTED_TIDAL_ORDERS = frozenset((-1, 0))
# The irregular averaged RHS has no 3.5PN spin term: ALL/default, 3PN, and
# explicit 3.5PN therefore produce the same corrected-LAL Fourier trajectory.
_SUPPORTED_SPIN_ORDERS = frozenset((-1, 6, 7))
_ACCESSORY_SELECTOR_KEYS = (
    "phenom_x_prec_version",
    "phenom_xp_convention",
    "phenom_xp_final_spin_mod",
)

__all__ = (
    "spintaylor_t4_fourier_fd_torch",
    "spintaylor_t4_fourier_native_supported",
    "spintaylor_t5_fourier_fd_torch",
    "spintaylor_t5_fourier_native_supported",
)


@dataclass(frozen=True)
class _SpinTaylorFourierTrajectory:
    """Irregular SpinTaylor trajectory in LAL's internal convention."""

    time: torch.Tensor
    state: torch.Tensor
    omega_rate: torch.Tensor
    mass1_fraction: float
    mass2_fraction: float


@dataclass(frozen=True)
class _Spline:
    knots: torch.Tensor
    values: torch.Tensor
    linear: torch.Tensor
    quadratic: torch.Tensor
    cubic: torch.Tensor

    @classmethod
    def build(cls, knots, values):
        knots = knots.contiguous()
        values = values.contiguous()
        linear, quadratic, cubic = _natural_cubic_coeff(knots, values)
        return cls(knots, values, linear, quadratic, cubic)

    def evaluate(self, points):
        indices = (
            torch.searchsorted(self.knots, points.clamp(self.knots[0], self.knots[-1]))
            - 1
        )
        indices = indices.clamp(0, self.knots.numel() - 2)
        offset = points - self.knots[indices]
        shape = offset.shape + (1,) * (self.values.ndim - 1)
        offset = offset.reshape(shape)
        return self.values[indices] + offset * (
            self.linear[indices]
            + offset * (self.quadratic[indices] + offset * self.cubic[indices])
        )


def _finite_order(parameters, name, default=-1):
    try:
        return operator.index(parameters.get(name, default))
    except TypeError as exc:
        raise ValueError(f"SpinTaylor requires integer {name}") from exc


def _active_float64_device():
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise TypeError("native SpinTaylorFourier requires an active TorchScheme")
    device = state.torch_device
    if device.type != "cpu":
        raise TypeError("native SpinTaylorFourier currently requires Torch CPU")
    return device


def _supported_order(parameters, name, supported, default=-1):
    try:
        return operator.index(parameters.get(name, default)) in supported
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_parameter(parameters, name, default=None):
    value = parameters.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _inactive_scalar(parameters, name):
    value = parameters.get(name)
    if value is None:
        return True
    value = _finite_parameter(parameters, name)
    return value == 0.0


def _spintaylor_fourier_native_supported(parameters, dynamics):
    """Return whether a public regular-FD call is in the native envelope."""

    try:
        expected = f"{dynamics}Fourier"
        if parameters.get("approximant", expected) != expected:
            return False

        state = _scheme.mgr.state
        if not (
            isinstance(state, _scheme.TorchScheme) and state.torch_device.type == "cpu"
        ):
            return False

        if not _supported_order(parameters, "phase_order", _SUPPORTED_PHASE_ORDERS):
            return False
        if not _supported_order(parameters, "spin_order", _SUPPORTED_SPIN_ORDERS):
            return False
        if not _supported_order(parameters, "tidal_order", _SUPPORTED_TIDAL_ORDERS):
            return False
        if not _supported_order(parameters, "amplitude_order", {0}):
            return False
        if not _supported_order(parameters, "eccentricity_order", {-1}):
            return False
        # PyCBC's zero is the unset sentinel; LAL's explicit Orbital-L
        # default has enum value two and is physically identical here.
        if not _supported_order(parameters, "frame_axis", {0, 2}, default=0):
            return False
        if not _supported_order(parameters, "modes_choice", {0}, default=0):
            return False
        if not _supported_order(parameters, "side_bands", {0}, default=0):
            return False

        mass1 = _finite_parameter(parameters, "mass1")
        mass2 = _finite_parameter(parameters, "mass2")
        distance = _finite_parameter(parameters, "distance", 1.0)
        delta_f = _finite_parameter(parameters, "delta_f")
        f_lower = _finite_parameter(parameters, "f_lower")
        f_final = _finite_parameter(parameters, "f_final", 0.0)
        f_ref = _finite_parameter(parameters, "f_ref", 0.0)
        if any(
            value is None
            for value in (mass1, mass2, distance, delta_f, f_lower, f_final, f_ref)
        ):
            return False
        if mass1 <= 0.0 or mass2 <= 0.0 or distance <= 0.0:
            return False
        if delta_f <= 0.0 or f_lower <= 0.0 or f_final < 0.0:
            return False

        total_mass = mass1 + mass2
        f_isco = 1.0 / (6.0**1.5 * math.pi * total_mass * _MTSUN_SI)
        effective_f_ref = f_lower if f_ref == 0.0 else f_ref
        if not (f_lower <= effective_f_ref < f_isco):
            return False
        f_min = delta_f * math.floor(math.nextafter(f_lower, math.inf) / delta_f)
        if f_min <= 0.0:
            return False
        effective_upper = f_final if f_final > f_min else f_isco
        if effective_upper > f_isco:
            return False

        finite_names = (
            "inclination",
            "coa_phase",
            "long_asc_nodes",
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
        )
        finite_values = {
            name: _finite_parameter(parameters, name, 0.0) for name in finite_names
        }
        if any(value is None for value in finite_values.values()):
            return False
        for prefix in ("spin1", "spin2"):
            magnitude2 = sum(finite_values[f"{prefix}{axis}"] ** 2 for axis in "xyz")
            if magnitude2 > 1.0:
                return False

        if not all(
            _inactive_scalar(parameters, name)
            for name in (
                "lambda1",
                "lambda2",
                "eccentricity",
                "mean_per_ano",
                *_NL_TIDAL_KEYS,
                *_TIDAL_EXTENSION_KEYS,
                *_NON_GR_KEYS,
            )
        ):
            return False
        if parameters.get("mode_array") is not None:
            return False
        if parameters.get("numrel_data") not in (None, ""):
            return False
        if any(parameters.get(name) is not None for name in _ACCESSORY_SELECTOR_KEYS):
            return False
        return True
    except Exception:
        # The dispatcher must preserve the LAL fallback for malformed values;
        # predicates are never allowed to turn input validation into a crash.
        return False


def spintaylor_t4_fourier_native_supported(parameters):
    """Return whether SpinTaylorT4Fourier is native for ``parameters``."""

    return _spintaylor_fourier_native_supported(parameters, "SpinTaylorT4")


def spintaylor_t5_fourier_native_supported(parameters):
    """Return whether SpinTaylorT5Fourier is native for ``parameters``."""

    return _spintaylor_fourier_native_supported(parameters, "SpinTaylorT5")


def _physical_check(state, x1, x2, matter):
    _check_spintaylor_physical_state(
        state,
        x1,
        x2,
        quadrupole1=matter[0],
        quadrupole2=matter[1],
        lambda1=matter[2],
        lambda2=matter[3],
        tidal_order=matter[4],
    )


@torch.inference_mode()
def _build_time_stepper(
    reference_state, time_rhs, *, compile_step=True, diagnostics=None
):
    """Build one fixed-shape GSL RKF45 step, tracing only when safe."""

    def finish(stepper, compiled, reason=None):
        if diagnostics is not None:
            diagnostics.update(compiled=compiled, fallback_reason=reason)
        return stepper, compiled

    zero_time = reference_state.new_zeros(())

    def ignored_time_rhs(_time, state):
        return time_rhs(state)

    def eager_step(state, step, first_slope):
        trial = rkf45_step(
            ignored_time_rhs,
            zero_time,
            state,
            step,
            first_derivative=first_slope,
            compute_final_derivative=True,
        )
        return trial.state, trial.error, trial.final_derivative

    trace = getattr(torch.jit, "trace", None)
    if not compile_step:
        return finish(eager_step, False, "disabled")
    if not callable(trace):
        return finish(eager_step, False, "torch.jit.trace is unavailable")

    example_step = reference_state.new_tensor(1.0)
    example_slope = time_rhs(reference_state)
    probe_step = reference_state.new_tensor(-0.8)
    probe_state = reference_state.clone()
    probe_state[1] *= 0.97
    probe_state[2:5] = torch.roll(probe_state[2:5], 1)
    probe_state[5] += 1.0e-4
    probe_state[9] -= 1.0e-4
    probe_state[11:14] = torch.roll(probe_state[11:14], -1)
    probe_slope = time_rhs(probe_state)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", torch.jit.TracerWarning)
            compiled = trace(
                eager_step,
                (reference_state, example_step, example_slope),
                check_trace=False,
                strict=False,
            )
        tracer_warnings = [
            item
            for item in caught
            if issubclass(item.category, torch.jit.TracerWarning)
        ]
        # The RHS has a fixed final dimension and the trace is scoped to one
        # waveform's constants, so these two warnings describe intentional
        # specialization.  Any new warning is treated as unsafe and falls back.
        expected_warning_lines = {
            ("imrphenomx_spintaylor_torch.py", 167),
            ("imrphenomx_spintaylor_torch.py", 1134),
        }
        unsafe_warnings = [
            item
            for item in tracer_warnings
            if not any(
                item.filename.endswith(filename) and item.lineno == line
                for filename, line in expected_warning_lines
            )
        ]
        if unsafe_warnings:
            messages = "; ".join(
                dict.fromkeys(
                    f"{item.filename}:{item.lineno}: {item.message}"
                    for item in unsafe_warnings
                )
            )
            return finish(eager_step, False, messages)

        expected = eager_step(probe_state, probe_step, probe_slope)
        actual = compiled(probe_state, probe_step, probe_slope)
        epsilon = torch.finfo(reference_state.dtype).eps
        if not all(
            torch.allclose(left, right, rtol=8.0 * epsilon, atol=8.0 * epsilon)
            for left, right in zip(actual, expected)
        ):
            return finish(eager_step, False, "compiled safety probe differed")
    except Exception as exc:  # optional optimization; eager is authoritative
        return finish(eager_step, False, f"{type(exc).__name__}: {exc}")
    return finish(compiled, True)


@torch.inference_mode()
def _adaptive_irregular_time_branch(
    reference_state,
    direction,
    time_rhs,
    physical_check,
    *,
    target_omega=None,
    rtol=1.0e-12,
    atol=1.0e-12,
    max_steps=1_000_000,
    stepper=None,
    compile_step=True,
    diagnostics=None,
):
    """Mirror LAL's irregular time-domain RKF45 trajectory driver.

    The independent variable is dimensionless ``t / M`` and the initial step
    is exactly ``+1`` or ``-1``.  A stop condition is evaluated at the top of
    the loop, so the first accepted knot beyond a requested or physical
    boundary remains in the returned trajectory, as it does in LAL.
    """

    if direction not in (-1.0, 1.0):
        raise ValueError("SpinTaylor integration direction must be +/-1")
    states = [reference_state]
    times = [reference_state.new_zeros(())]
    current = reference_state
    current_slope = time_rhs(reference_state)
    omega_rates = [current_slope[1]]
    prior_omega_rate = 0.0
    current_time = 0.0
    step = direction
    step_tensor = reference_state.new_empty(())
    if stepper is None:
        stepper, _ = _build_time_stepper(
            reference_state, time_rhs, compile_step=compile_step
        )

    accepted_steps = 0
    rejected_steps = 0
    derivative_retries = 0
    stop_reason = None
    check_stop = True
    attempts = 0
    while accepted_steps < max_steps:
        attempts += 1
        if check_stop:
            omega = float(current[1].detach().cpu())
            omega_rate = float(current_slope[1].detach().cpu())
            if target_omega is not None and (
                (direction < 0.0 and omega < target_omega)
                or (direction > 0.0 and omega > target_omega)
            ):
                stop_reason = "frequency"
                break

            boundary = False
            try:
                physical_check(current)
            except _SpinTaylorPhysicalBoundary:
                boundary = True
            if boundary:
                stop_reason = "physical"
                break

            omega_acceleration = omega_rate - prior_omega_rate
            if direction < 0.0 and prior_omega_rate != 0.0:
                omega_acceleration = -omega_acceleration
            prior_omega_rate = omega_rate
            if omega_acceleration <= 0.0:
                stop_reason = "omega_acceleration"
                break

        step_tensor.fill_(step)
        candidate, error, end_slope = stepper(current, step_tensor, current_slope)
        finite_trial = bool(
            (
                torch.isfinite(candidate).all()
                & torch.isfinite(error).all()
                & torch.isfinite(end_slope).all()
            )
            .detach()
            .cpu()
        )
        if not finite_trial:
            derivative_retries += 1
            if derivative_retries > 6:
                raise RuntimeError("SpinTaylor RKF45 derivative retries exhausted")
            step *= 0.1
            rejected_steps += 1
            check_stop = False
            if step == 0.0 or not math.isfinite(step):
                raise RuntimeError("SpinTaylor adaptive step size became invalid")
            continue

        derivative_retries = 0
        scale = atol + rtol * torch.abs(candidate)
        error_ratio = float(torch.max(torch.abs(error) / scale).detach().cpu())
        if error_ratio > 1.1:
            factor = max(0.2, 0.9 * error_ratio ** (-1.0 / 5.0))
            step *= factor
            rejected_steps += 1
            check_stop = False
            if step == 0.0 or not math.isfinite(step):
                raise RuntimeError("SpinTaylor adaptive step size became invalid")
            continue

        current_time += step
        current = candidate
        current_slope = end_slope
        times.append(reference_state.new_tensor(current_time))
        states.append(current)
        omega_rates.append(current_slope[1])
        accepted_steps += 1
        check_stop = True
        if error_ratio < 0.5:
            bounded_ratio = max(error_ratio, torch.finfo(reference_state.dtype).tiny)
            step *= min(5.0, max(1.0, 0.9 * bounded_ratio ** (-1.0 / 6.0)))
    else:
        raise RuntimeError("SpinTaylor integration exceeded the maximum step count")

    state = torch.stack(states)
    time = torch.stack(times)
    omega_rate_tensor = torch.stack(omega_rates)
    if direction < 0.0 and stop_reason != "frequency":
        raise RuntimeError("SpinTaylor reached a physical boundary below f_ref")
    if diagnostics is not None:
        diagnostics.update(
            attempts=attempts,
            accepted=accepted_steps,
            rejected=rejected_steps,
            stop_reason=stop_reason,
        )
    return time, state, omega_rate_tensor


def _average_duplicate_time_knots(time, state, omega_rate):
    """Average non-time rows at exact duplicate knots, matching LAL."""

    unique_time, inverse, counts = torch.unique_consecutive(
        time, return_inverse=True, return_counts=True
    )
    if unique_time.numel() == time.numel():
        return time, state, omega_rate
    combined = torch.cat((state, omega_rate.unsqueeze(-1)), dim=-1)
    averaged = combined.new_zeros((unique_time.numel(), combined.shape[-1]))
    averaged.index_add_(0, inverse, combined)
    averaged /= counts.to(averaged).unsqueeze(-1)
    return unique_time, averaged[:, :14], averaged[:, 14]


def _spintaylor_fourier_trajectory(
    dynamics,
    mass1,
    mass2,
    f_start,
    f_ref,
    spin1,
    spin2,
    lnhat,
    e1,
    coa_phase,
    matter,
    device,
    *,
    rtol=1.0e-12,
    atol=1.0e-12,
    compile_step=True,
    diagnostics=None,
):
    """Build the irregular orbit used by the common SUA assembler."""

    try:
        rhs_function = _DYNAMICS[dynamics]
    except KeyError as exc:
        raise ValueError(f"unsupported SpinTaylor dynamics {dynamics}") from exc

    dtype = torch.float64
    total_mass = mass1 + mass2
    x1 = mass1 / total_mass
    x2 = mass2 / total_mass
    mass_seconds = total_mass * _MTSUN_SI
    spin1 = torch.as_tensor(spin1, dtype=dtype, device=device)
    spin2 = torch.as_tensor(spin2, dtype=dtype, device=device)
    internal_spin1, internal_spin2 = spintaylor_internal_spins(
        mass1, mass2, spin1, spin2
    )
    reference_state = torch.cat(
        (
            torch.tensor(
                [coa_phase, math.pi * mass_seconds * f_ref], dtype=dtype, device=device
            ),
            torch.as_tensor(lnhat, dtype=dtype, device=device),
            internal_spin1,
            internal_spin2,
            torch.as_tensor(e1, dtype=dtype, device=device),
        )
    )
    matter_tensors = (
        torch.as_tensor(matter[0], dtype=dtype, device=device),
        torch.as_tensor(matter[1], dtype=dtype, device=device),
        torch.as_tensor(matter[2], dtype=dtype, device=device),
        torch.as_tensor(matter[3], dtype=dtype, device=device),
        matter[4],
    )

    def rhs(state):
        return rhs_function(
            state,
            x1,
            x2,
            quadrupole1=matter_tensors[0],
            quadrupole2=matter_tensors[1],
            lambda1=matter_tensors[2],
            lambda2=matter_tensors[3],
            tidal_order=matter_tensors[4],
        )

    def check(state):
        _physical_check(state, x1, x2, matter_tensors)

    check(reference_state)
    stepper_diagnostics = {} if diagnostics is not None else None
    stepper, is_compiled = _build_time_stepper(
        reference_state,
        rhs,
        compile_step=compile_step,
        diagnostics=stepper_diagnostics,
    )
    low_diagnostics = {} if diagnostics is not None else None
    high_diagnostics = {} if diagnostics is not None else None
    low_time, low_state, low_rate = _adaptive_irregular_time_branch(
        reference_state,
        -1.0,
        rhs,
        check,
        target_omega=math.pi * mass_seconds * f_start,
        rtol=rtol,
        atol=atol,
        stepper=stepper,
        diagnostics=low_diagnostics,
    )
    high_time, high_state, high_rate = _adaptive_irregular_time_branch(
        reference_state,
        1.0,
        rhs,
        check,
        rtol=rtol,
        atol=atol,
        stepper=stepper,
        diagnostics=high_diagnostics,
    )

    time = torch.cat((torch.flip(low_time, (0,)), high_time[1:]))
    state = torch.cat((torch.flip(low_state, (0,)), high_state[1:]))
    omega_rate = torch.cat((torch.flip(low_rate, (0,)), high_rate[1:]))
    time, state, omega_rate = _average_duplicate_time_knots(time, state, omega_rate)
    if not bool(torch.all(time[1:] > time[:-1]).detach().cpu()):
        raise RuntimeError("SpinTaylor adaptive trajectory times are not monotonic")
    if not bool(torch.all(state[1:, 1] > state[:-1, 1]).detach().cpu()):
        raise RuntimeError("SpinTaylor adaptive trajectory frequency is not monotonic")
    if diagnostics is not None:
        diagnostics.update(
            compiled_step=is_compiled,
            stepper=stepper_diagnostics,
            low=low_diagnostics,
            high=high_diagnostics,
        )
    return _SpinTaylorFourierTrajectory(time, state, omega_rate, x1, x2)


def _harmonic_numbers(amplitude_order):
    if amplitude_order == 0:
        return (2,)
    if amplitude_order == 1:
        return (1, 2, 3)
    if amplitude_order == 2:
        return (1, 2, 3, 4)
    return (1, 2, 3, 4, 5)


def _spintaylor_harmonic(
    harmonic,
    velocity,
    spin1,
    spin2,
    lnhat,
    e1,
    x1,
    x2,
    amplitude_order,
):
    """Evaluate one phase-independent LAL Fourier harmonic.

    A short exact phase DFT extracts the coefficient from the already-ported
    time-domain radiation polynomial.  Sixteen samples resolve every
    supported (n <= 5) harmonic without aliasing.  LAL's Fourier-only n=2
    logarithmic tail gauge term is added explicitly afterward.
    """

    count = velocity.numel()
    phases = (
        2.0
        * math.pi
        * torch.arange(_PHASE_SAMPLES, dtype=velocity.dtype, device=velocity.device)
        / _PHASE_SAMPLES
    )

    def repeat(value):
        return value.repeat_interleave(_PHASE_SAMPLES, dim=0)

    tiled_phase = phases.repeat(count)
    plus, cross = spintaylor_polarizations_from_orbit(
        repeat(velocity),
        tiled_phase,
        repeat(spin1),
        repeat(spin2),
        repeat(lnhat),
        repeat(e1),
        x1,
        x2,
        1.0,
        amplitude_order=amplitude_order,
    )
    amplitude_factor = 2.0 * _MRSUN_SI * x1 * x2 / _MPC_SI
    basis = torch.exp(1j * harmonic * phases).to(torch.complex128)
    plus = (plus.reshape(count, _PHASE_SAMPLES).to(torch.complex128) @ basis) / (
        _PHASE_SAMPLES * amplitude_factor
    )
    cross = (cross.reshape(count, _PHASE_SAMPLES).to(torch.complex128) @ basis) / (
        _PHASE_SAMPLES * amplitude_factor
    )

    if harmonic == 2 and (amplitude_order == -1 or amplitude_order >= 3):
        e2 = torch.cross(lnhat, e1, dim=-1)
        e1x, e1y = e1[:, 0], e1[:, 1]
        e2x, e2y = e2[:, 0], e2[:, 1]
        logarithm = torch.log(velocity)
        tail = velocity**5 * logarithm
        plus = plus + tail * (
            12.0 * (e1x * e2x - e1y * e2y) + 6j * (-(e1x**2) + e1y**2 + e2x**2 - e2y**2)
        )
        cross = cross + tail * (
            12.0 * (e1y * e2x + e1x * e2y) + 12j * (-e1x * e1y + e2x * e2y)
        )
    return plus, cross


def _gps_floor(value):
    seconds = math.floor(value)
    nanoseconds = math.floor(1.0e9 * (value - seconds))
    return seconds + nanoseconds * 1.0e-9


def _assemble_spintaylor_fourier(
    trajectory,
    frequencies,
    mass_seconds,
    distance,
    amplitude_order,
):
    """Assemble both Fourier polarizations with the common kMax=3 SUA sum."""

    time_spline = _Spline.build(trajectory.time, trajectory.state)
    rate_spline = _Spline.build(trajectory.time, trajectory.omega_rate)
    omega_time_spline = _Spline.build(trajectory.state[:, 1], trajectory.time)

    omega_isco = 1.0 / math.sqrt(216.0)
    if float(trajectory.state[-1, 1].detach().cpu()) >= omega_isco:
        isco = torch.as_tensor(
            [omega_isco], dtype=frequencies.dtype, device=frequencies.device
        )
        t_isco = omega_time_spline.evaluate(isco)[0]
    else:
        t_isco = trajectory.time[-1]

    plus = torch.zeros(
        frequencies.shape, dtype=torch.complex128, device=frequencies.device
    )
    cross = torch.zeros_like(plus)
    eta = trajectory.mass1_fraction * trajectory.mass2_fraction
    prefactor = (
        2.0
        * math.sqrt(2.0 * math.pi)
        * mass_seconds**2
        * eta
        * _C_SI
        / (distance * _MPC_SI)
    )
    coefficients = torch.as_tensor(
        _SUA_COEFFICIENTS, dtype=torch.complex128, device=frequencies.device
    )

    for harmonic in _harmonic_numbers(amplitude_order):
        stationary_omega = 2.0 * math.pi * mass_seconds * frequencies / harmonic
        active = (stationary_omega >= trajectory.state[0, 1]) & (
            stationary_omega <= trajectory.state[-1, 1]
        )
        if not bool(active.any().detach().cpu()):
            continue
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        active_omega = stationary_omega[active]
        center_time = omega_time_spline.evaluate(active_omega)
        center_state = time_spline.evaluate(center_time)
        center_rate = rate_spline.evaluate(center_time)
        shift_scale = torch.rsqrt(torch.abs(harmonic * center_rate))
        harmonic_plus = torch.zeros(
            center_time.shape, dtype=torch.complex128, device=frequencies.device
        )
        harmonic_cross = torch.zeros_like(harmonic_plus)

        for shift in range(-_KMAX, _KMAX + 1):
            shifted_time = center_time + shift * shift_scale
            in_domain = (shifted_time >= trajectory.time[0]) & (
                shifted_time <= trajectory.time[-1]
            )
            if not bool(in_domain.any().detach().cpu()):
                continue
            state = time_spline.evaluate(shifted_time[in_domain])
            rate = rate_spline.evaluate(shifted_time[in_domain])
            velocity = torch.pow(state[:, 1], 1.0 / 3.0)
            # LAL's Fourier harmonic helper consumes the trajectory's
            # total-mass-normalized x_i**2 * chi_i spins directly.  Do not
            # rescale these as the public time-domain helper normally does.
            hplus, hcross = _spintaylor_harmonic(
                harmonic,
                velocity,
                state[:, 5:8],
                state[:, 8:11],
                state[:, 2:5],
                state[:, 11:14],
                trajectory.mass1_fraction,
                trajectory.mass2_fraction,
                amplitude_order,
            )
            shifted_scale = torch.rsqrt(torch.abs(harmonic * rate))
            coefficient = coefficients[abs(shift)]
            harmonic_plus[in_domain] += coefficient * shifted_scale * hplus
            harmonic_cross[in_domain] += coefficient * shifted_scale * hcross

        active_frequencies = frequencies[active]
        phase = (
            2.0 * math.pi * mass_seconds * active_frequencies * (center_time - t_isco)
            - harmonic * center_state[:, 0]
            - math.pi / 4.0
        )
        phase_factor = prefactor * torch.exp(1j * phase)
        plus[active_indices] += torch.conj(harmonic_plus * phase_factor)
        cross[active_indices] += torch.conj(harmonic_cross * phase_factor)

    return plus, cross, t_isco


def _spintaylor_fourier_fd_torch(dynamics, **parameters):
    mass1 = _parameter_float(parameters, "mass1")
    mass2 = _parameter_float(parameters, "mass2")
    distance = _parameter_float(parameters, "distance", 1.0)
    inclination = _parameter_float(parameters, "inclination", 0.0)
    coa_phase = _parameter_float(parameters, "coa_phase", 0.0)
    long_asc_nodes = _parameter_float(parameters, "long_asc_nodes", 0.0)
    delta_f = _parameter_float(parameters, "delta_f")
    f_lower = _parameter_float(parameters, "f_lower")
    f_final = _parameter_float(parameters, "f_final", 0.0)
    f_ref = _parameter_float(parameters, "f_ref", 0.0)
    effective_f_ref = f_lower if f_ref == 0.0 else f_ref
    amplitude_order = _finite_order(parameters, "amplitude_order")
    tidal_order = _finite_order(parameters, "tidal_order")
    if mass1 <= 0.0 or mass2 <= 0.0 or distance <= 0.0:
        raise ValueError("SpinTaylor masses and distance must be positive")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("SpinTaylor frequencies must be positive")

    total_mass = mass1 + mass2
    mass_seconds = total_mass * _MTSUN_SI
    f_isco = 1.0 / (6.0**1.5 * math.pi * mass_seconds)
    if effective_f_ref < f_lower or effective_f_ref >= f_isco:
        raise ValueError("SpinTaylor f_ref must satisfy f_lower <= f_ref < f_ISCO")

    spin1 = tuple(_parameter_float(parameters, f"spin1{axis}", 0.0) for axis in "xyz")
    spin2 = tuple(_parameter_float(parameters, f"spin2{axis}", 0.0) for axis in "xyz")
    spin1 = _rotate_y(spin1, inclination)
    spin2 = _rotate_y(spin2, inclination)
    lnhat = (math.sin(inclination), 0.0, math.cos(inclination))
    e1 = (0.0, 1.0, 0.0)
    lambda1 = (
        0.0
        if parameters.get("lambda1") is None
        else _parameter_float(parameters, "lambda1")
    )
    lambda2 = (
        0.0
        if parameters.get("lambda2") is None
        else _parameter_float(parameters, "lambda2")
    )
    quadrupole1 = 1.0 + (
        0.0
        if parameters.get("dquad_mon1") is None
        else _parameter_float(parameters, "dquad_mon1")
    )
    quadrupole2 = 1.0 + (
        0.0
        if parameters.get("dquad_mon2") is None
        else _parameter_float(parameters, "dquad_mon2")
    )
    device = _active_float64_device()
    trajectory = _spintaylor_fourier_trajectory(
        dynamics,
        mass1,
        mass2,
        0.9 * f_lower,
        effective_f_ref,
        spin1,
        spin2,
        lnhat,
        e1,
        coa_phase,
        (quadrupole1, quadrupole2, lambda1, lambda2, tidal_order),
        device,
    )

    f_min = delta_f * math.floor(math.nextafter(f_lower, math.inf) / delta_f)
    first = int(f_min / delta_f)
    upper = f_final if f_final > f_min else f_isco
    sample_count = math.ceil((upper - f_min) / delta_f)
    if sample_count <= 0:
        raise ValueError("SpinTaylor Fourier frequency interval is empty")
    length = first + sample_count
    active_frequencies = delta_f * torch.arange(
        first, length, dtype=torch.float64, device=device
    )
    plus_active, cross_active, t_isco = _assemble_spintaylor_fourier(
        trajectory,
        active_frequencies,
        mass_seconds,
        distance,
        amplitude_order,
    )
    plus = torch.zeros(length, dtype=torch.complex128, device=device)
    cross = torch.zeros_like(plus)
    plus[first:] = plus_active
    cross[first:] = cross_active

    cosine = math.cos(2.0 * long_asc_nodes)
    sine = math.sin(2.0 * long_asc_nodes)
    original_plus, original_cross = plus, cross
    plus = cosine * original_plus + sine * original_cross
    cross = cosine * original_cross - sine * original_plus

    # LAL evaluates the epoch at the requested lower frequency, while the
    # returned spectrum starts at the grid bin immediately below it.
    omega_lower = torch.as_tensor(
        [math.pi * mass_seconds * f_lower], dtype=torch.float64, device=device
    )
    omega_time_spline = _Spline.build(trajectory.state[:, 1], trajectory.time)
    epoch = _gps_floor(
        float(
            ((omega_time_spline.evaluate(omega_lower)[0] - t_isco) * mass_seconds)
            .detach()
            .cpu()
        )
    )
    return (
        FrequencySeries(TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def spintaylor_t4_fourier_fd_torch(**parameters):
    """Generate a supported CPU SpinTaylorT4Fourier waveform with Torch."""

    if not spintaylor_t4_fourier_native_supported(parameters):
        raise ValueError("unsupported native SpinTaylorT4Fourier parameters")
    return _spintaylor_fourier_fd_torch("SpinTaylorT4", **parameters)


def spintaylor_t5_fourier_fd_torch(**parameters):
    """Generate a supported CPU SpinTaylorT5Fourier waveform with Torch."""

    if not spintaylor_t5_fourier_native_supported(parameters):
        raise ValueError("unsupported native SpinTaylorT5Fourier parameters")
    return _spintaylor_fourier_fd_torch("SpinTaylorT5", **parameters)
