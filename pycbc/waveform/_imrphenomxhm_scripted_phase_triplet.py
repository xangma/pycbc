# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Optional compiled phase lane for the XHM 21, 33, and 44 modes.

The three modes use the same piecewise mathematical form.  Packing only that
fixed-schema form lets TorchScript compile one three-row expression instead of
launching the same scalar and frequency-dependent operations three times.  The
matching systems themselves retain their established construction and solve
order.  Callers must qualify the request before entering this module.

This optimization is deliberately private, off by default, and fail closed.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import torch

from . import imrphenomxhm_mode21_torch as _mode21
from . import imrphenomxhm_mode33_torch as _mode33
from . import imrphenomxhm_mode44_torch as _mode44
from .torch_switches import _parse_switch


_ENV = "PYCBC_IMRPHENOMXHM_SCRIPTED_PHASE_TRIPLET"
_DYNAMIC_INDICES = _mode21._PACKED_PHASE_DYNAMIC_INDICES
_MAX_EXECUTORS = 4
_MAX_FAILED_KEYS = 8
_EXECUTORS = OrderedDict()
_FAILED_KEYS = OrderedDict()
_LOCK = threading.Lock()
_CACHE_PID = os.getpid()


def _scripted_phase_triplet_enabled():
    """Return the strict, default-off feature switch."""

    value = os.environ.get(_ENV)
    return False if value is None else _parse_switch(_ENV, value)


def _scripted_phase_triplet_version_supported(device):
    """Accept only Torch release/backend pairs with sealed qualifications."""

    if type(device) is not torch.device or device.type not in ("cpu", "cuda"):
        return False
    version = getattr(torch, "__version__", None)
    if not isinstance(version, str):
        return False
    if device.type == "cuda":
        torch_build = getattr(torch, "version", None)
        return (
            version == "2.13.0+cu126"
            and torch_build is not None
            and getattr(torch_build, "hip", None) is None
            and getattr(torch_build, "cuda", None) == "12.6"
        )
    release = version.partition("+")[0].split(".")
    if len(release) != 3 or not all(item.isdigit() for item in release):
        return False
    parsed = tuple(int(item) for item in release)
    return parsed in {(2, 9, 1), (2, 13, 0)}


def _clear_scripted_phase_triplet_cache():
    """Clear process-local compiled executors (tests and benchmarks only)."""

    global _CACHE_PID
    with _LOCK:
        _EXECUTORS.clear()
        _FAILED_KEYS.clear()
        _CACHE_PID = os.getpid()


def _reset_scripted_phase_triplet_after_fork():
    """Discard parent-process code and locks in a forked child."""

    global _CACHE_PID, _LOCK
    _LOCK = threading.Lock()
    _EXECUTORS.clear()
    _FAILED_KEYS.clear()
    _CACHE_PID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_scripted_phase_triplet_after_fork)


def _scalar_controls(
    plan,
    intermediate_row,
    ringdown_row,
    scalar_row,
    anchor_row,
    carrier_inspiral_align,
    mode_index,
):
    """Evaluate the five scalar matching and alignment controls for one row."""

    c0, c_l, c1, c2, c4 = intermediate_row.unbind()
    (
        ringdown_numerator,
        ringdown_derivative_numerator,
        ringdown_alpha_l,
        ringdown_l_damp,
    ) = ringdown_row.unbind()
    (
        output_scale,
        input_scale,
        lambda_pn,
        fmatch_in,
        fmatch_rd,
        f_ring,
        f_damp,
        f_damp_squared,
        delta_t,
        linb_fit,
        mf_ref,
        lina,
        coa_phase,
        f_align,
        m_over_2,
        sign_shift,
    ) = scalar_row.unbind()
    dphi22_ref, phase_ref_22, alignment_phase = anchor_row.unbind()

    frequency_in = torch.empty_like(output_scale)
    frequency_in.copy_(fmatch_in)
    one = torch.ones_like(frequency_in)
    transformed = input_scale * frequency_in
    phase_value, transformed_derivative = (
        _mode21._exact_inspiral_phase_value_and_derivative(
            transformed,
            plan,
            output_adjoint=one * output_scale,
        )
    )
    insp_at_in = output_scale * phase_value + lambda_pn * frequency_in
    dinsp_at_in = one * lambda_pn
    dinsp_at_in = dinsp_at_in + transformed_derivative * input_scale

    def intermediate_raw(frequency):
        return (
            c0 * frequency
            + c1 * torch.log(frequency)
            - c2 / frequency
            - c4 / (3.0 * frequency**3)
            + c_l * torch.atan((frequency - f_ring) / f_damp)
        )

    def intermediate_derivative(frequency):
        return (
            c0
            + c_l * f_damp / (f_damp_squared + (frequency - f_ring) ** 2)
            + c1 / frequency
            + c2 / frequency**2
            + c4 / frequency**4
        )

    int_at_in = intermediate_raw(frequency_in)
    dint_at_in = intermediate_derivative(frequency_in)
    c1_insp = dint_at_in - dinsp_at_in
    c_insp = -c1_insp * fmatch_in + int_at_in - insp_at_in

    frequency_rd = torch.empty_like(output_scale)
    frequency_rd.copy_(fmatch_rd)
    int_at_rd = intermediate_raw(frequency_rd)
    dint_at_rd = intermediate_derivative(frequency_rd)
    rd_at_rd = (
        ringdown_numerator / frequency_rd
        + ringdown_alpha_l
        * torch.atan((frequency_rd - f_ring) / f_damp)
    )
    drd_at_rd = (
        ringdown_derivative_numerator / frequency_rd**2
        + ringdown_l_damp
        / (f_damp_squared + (frequency_rd - f_ring) ** 2)
    )
    c1_rd = dint_at_rd - drd_at_rd
    c_rd = -c1_rd * fmatch_rd + int_at_rd - rd_at_rd

    timeshift = linb_fit - dphi22_ref + delta_t
    phiref22 = (
        -phase_ref_22
        - timeshift * mf_ref
        - lina
        + 2.0 * coa_phase
        + _mode21._PI / 4.0
    )
    mode_insp_align = (
        output_scale * carrier_inspiral_align + lambda_pn * f_align
    )
    mode_insp_align = mode_insp_align + c1_insp * f_align + c_insp
    aligned_22 = (
        m_over_2 * (alignment_phase + lina + phiref22)
        + timeshift * f_align
    )
    if mode_index == 0:
        adjusted = aligned_22 - 3.0 * _mode21._PI / 8.0
    elif mode_index == 1:
        adjusted = aligned_22 + 3.0 * _mode21._PI / 8.0
    else:
        adjusted = aligned_22 + 3.0 * _mode21._PI / 4.0
    delta_phi = torch.fmod(
        adjusted - mode_insp_align,
        2.0 * _mode21._PI,
    )
    delta_phi = delta_phi + sign_shift
    return c1_insp, c_insp, c1_rd, c_rd, delta_phi


def _phase_triplet_lane_source(
    mf,
    ringdown_21_raw,
    inspiral_coefficients,
    intermediate_coefficients,
    ringdown_coefficients,
    scalars,
    anchors,
):
    """Fixed-schema three-row expression traced once per device and length."""

    (
        phi2,
        phi3,
        phi4,
        phi5_l,
        phi6,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    ) = inspiral_coefficients
    plan = _mode21._InspiralPhasePlan(
        1.0,
        0.0,
        phi2,
        phi3,
        phi4,
        0.0,
        phi5_l,
        phi6,
        -1072.8103323596813,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    )
    c0, c_l, c1, c2, c4 = intermediate_coefficients.unbind(1)
    ringdown_numerator, _, ringdown_alpha_l, _ = (
        ringdown_coefficients.unbind(1)
    )
    (
        output_scale,
        input_scale,
        lambda_pn,
        fmatch_in,
        fmatch_rd,
        f_ring,
        f_damp,
        _,
        _,
        _,
        _,
        _,
        _,
        f_align,
        _,
        _,
    ) = scalars.unbind(1)
    carrier_inspiral_align = _mode21._evaluate_inspiral_phase(
        input_scale * f_align,
        plan,
    )
    controls = tuple(
        _scalar_controls(
            plan,
            intermediate_coefficients[index],
            ringdown_coefficients[index],
            scalars[index],
            anchors[index],
            carrier_inspiral_align[index],
            index,
        )
        for index in range(3)
    )
    c1_insp = torch.stack(tuple(value[0] for value in controls))
    c_insp = torch.stack(tuple(value[1] for value in controls))
    c1_rd = torch.stack(tuple(value[2] for value in controls))
    c_rd = torch.stack(tuple(value[3] for value in controls))
    delta_phi = torch.stack(tuple(value[4] for value in controls))

    frequency = mf.unsqueeze(0)
    scaled_frequency = input_scale.unsqueeze(1) * frequency
    carrier_phase = _mode21._evaluate_inspiral_phase(scaled_frequency, plan)
    inspiral_base = (
        output_scale.unsqueeze(1) * carrier_phase
        + lambda_pn.unsqueeze(1) * frequency
    )
    intermediate = (
        c0.unsqueeze(1) * frequency
        + c1.unsqueeze(1) * torch.log(frequency)
        - c2.unsqueeze(1) / frequency
        - c4.unsqueeze(1) / (3.0 * frequency**3)
        + c_l.unsqueeze(1)
        * torch.atan(
            (frequency - f_ring.unsqueeze(1)) / f_damp.unsqueeze(1)
        )
    )
    ringdown = (
        ringdown_numerator.unsqueeze(1) / frequency
        + ringdown_alpha_l.unsqueeze(1)
        * torch.atan(
            (frequency - f_ring.unsqueeze(1)) / f_damp.unsqueeze(1)
        )
    )
    delta = delta_phi.unsqueeze(1)
    inspiral = (
        inspiral_base
        + c1_insp.unsqueeze(1) * frequency
        + c_insp.unsqueeze(1)
        + delta
    )
    intermediate = intermediate + delta
    ringdown = (
        ringdown
        + c1_rd.unsqueeze(1) * frequency
        + c_rd.unsqueeze(1)
        + delta
    )
    result = torch.where(
        frequency <= fmatch_in.unsqueeze(1),
        inspiral,
        torch.where(
            frequency <= fmatch_rd.unsqueeze(1),
            intermediate,
            ringdown,
        ),
    )
    # Preserve mode 21's established vector arithmetic for its ringdown row.
    ringdown_21 = (
        ringdown_21_raw + c1_rd[0] * mf + c_rd[0] + delta_phi[0]
    )
    phase_21 = torch.where(mf <= fmatch_rd[0], result[0], ringdown_21)
    return phase_21.clone(), result[1].clone(), result[2].clone()


def _phase_generators(shared, states, phase_plan, anchors):
    common = (
        shared.intrinsic,
        shared.phase_coeffs,
        shared.reference_frequency,
        shared.coa_phase,
    )
    return (
        _mode21._phase_21_staged(
            shared.mf,
            states[0],
            *common,
            None,
            phase_plan,
            anchors,
            None,
        ),
        _mode33._phase_33_staged(
            shared.mf,
            states[1],
            *common,
            None,
            phase_plan,
            anchors,
            None,
            None,
            None,
        ),
        _mode44._phase_44_staged(
            shared.mf,
            states[2],
            *common,
            None,
            phase_plan,
            anchors,
            None,
            None,
            None,
        ),
    )


def _solve_phase_stages(generators):
    """Retain each established 5x5 system's construction and solve order."""

    solved = []
    frames = []
    for generator in generators:
        request = next(generator)
        frame = getattr(generator, "gi_frame", None)
        if frame is None:
            raise RuntimeError("phase generator frame is unavailable")
        values = frame.f_locals
        required = (
            "fmatch_in",
            "fmatch_rd",
            "f_ring",
            "f_damp",
            "delta_t",
            "linb_fit",
            "lina",
        )
        if any(name not in values for name in required):
            raise RuntimeError("phase generator schema changed")
        frames.append({name: values[name] for name in required})
        solved.append(_mode21._solve(*request))
    return tuple(solved), tuple(frames)


def _prepare_lane_inputs(shared, states, phase_plan, anchors):
    generators = _phase_generators(shared, states, phase_plan, anchors)
    solved, frames = _solve_phase_stages(generators)
    mf = shared.mf
    state21, state33, state44 = states
    lambdas = (
        _mode21._lambda_pn(state21),
        _mode33._lambda_pn(state33),
        _mode44._lambda_pn(state44),
    )
    alpha21, alpha_l21 = _mode21._ringdown_phase_fits(state21)
    alpha2_21_33, alpha_l33 = _mode33._ringdown_phase_fits_21(
        state33.base
    )
    alpha33 = (
        6.0 * alpha2_21_33 * state33.f_ring_21**2
        / state33.f_ring_33**2
    )
    alpha2_21_44, alpha_l44 = _mode44._ringdown_phase_fits_21(
        state44.base
    )
    alpha44 = (
        6.0 * alpha2_21_44 * state44.f_ring_21**2
        / state44.f_ring_44**2
    )
    alphas = (alpha21, alpha33, alpha44)
    alpha_ls = (alpha_l21, alpha_l33, alpha_l44)
    f_rings = tuple(values["f_ring"] for values in frames)
    f_damps = tuple(values["f_damp"] for values in frames)
    ringdown = _mode21._tensor(
        tuple(
            (
                -(f_ring**2) * alpha,
                f_ring**2 * alpha,
                alpha_l,
                alpha_l * f_damp,
            )
            for f_ring, f_damp, alpha, alpha_l in zip(
                f_rings,
                f_damps,
                alphas,
                alpha_ls,
            )
        ),
        mf,
    )
    ringdown_21_raw = (
        -(f_rings[0] ** 2) * alphas[0] / mf
        + alpha_ls[0]
        * torch.atan((mf - f_rings[0]) / f_damps[0])
    )

    dphi22_ref = (
        _mode21._carrier_phase_anchor(
            anchors,
            _mode21._CARRIER_RINGDOWN_START_DERIVATIVE,
            mf,
            lambda: _mode21.PhaseDerivative(
                _mode21._tensor(
                    (state21.f_ring_22 - state21.f_damp_22)
                    / state21.total_mass_seconds,
                    mf,
                ),
                shared.intrinsic,
                shared.phase_coeffs,
                final_spin=state21.final_spin,
                _phase_plan=phase_plan,
            ),
        )
        / state21.total_mass_seconds
    )
    phase_ref_22 = _mode21._carrier_phase_anchor(
        anchors,
        _mode21._CARRIER_REFERENCE_PHASE,
        mf,
        lambda: _mode21.Phase(
            _mode21._tensor(shared.reference_frequency, mf),
            shared.intrinsic,
            shared.phase_coeffs,
            final_spin=state21.final_spin,
            _phase_plan=phase_plan,
        ),
    )
    f_align = [values["fmatch_in"] for values in frames]
    if state21.eta > 0.05:
        f_align = [value * 0.6 for value in f_align]
    alignment21 = _mode21._carrier_phase_anchor(
        anchors,
        _mode21._CARRIER_ALIGNMENT_PHASE,
        mf,
        lambda: _mode21.Phase(
            _mode21._tensor(
                2.0 * f_align[0] / state21.total_mass_seconds,
                mf,
            ),
            shared.intrinsic,
            shared.phase_coeffs,
            final_spin=state21.final_spin,
            _phase_plan=phase_plan,
        ),
    )
    alignment33 = _mode33.Phase(
        _mode33._tensor(
            (2.0 / 3.0) * f_align[1] / state33.total_mass_seconds,
            mf,
        ),
        shared.intrinsic,
        shared.phase_coeffs,
        final_spin=state33.final_spin,
        _phase_plan=phase_plan,
    )
    alignment44 = _mode44._carrier_phase_anchor(
        anchors,
        _mode44._CARRIER_ALIGNMENT_PHASE,
        mf,
        lambda: _mode44.Phase(
            _mode44._tensor(
                0.5 * f_align[2] / state44.total_mass_seconds,
                mf,
            ),
            shared.intrinsic,
            shared.phase_coeffs,
            final_spin=state44.final_spin,
            _phase_plan=phase_plan,
        ),
    )
    anchor_values = torch.stack(
        (
            torch.stack((dphi22_ref, phase_ref_22, alignment21)),
            torch.stack((dphi22_ref, phase_ref_22, alignment33)),
            torch.stack((dphi22_ref, phase_ref_22, alignment44)),
        )
    )
    input_scales = (2.0, 2.0 / 3.0, 0.5)
    output_scales = (
        0.5 / state21.eta,
        1.5 / state33.eta,
        2.0 / state44.eta,
    )
    sign_shift = (
        _mode21._PI if _mode21._pn21_amplitude_sign(state21) > 0 else 0.0,
        0.0,
        0.0,
    )
    scalars = _mode21._tensor(
        tuple(
            (
                output_scales[index],
                input_scales[index],
                lambdas[index],
                values["fmatch_in"],
                values["fmatch_rd"],
                values["f_ring"],
                values["f_damp"],
                values["f_damp"] ** 2,
                values["delta_t"],
                values["linb_fit"],
                shared.reference_frequency
                * states[index].total_mass_seconds,
                values["lina"],
                shared.coa_phase,
                f_align[index],
                (0.5, 1.5, 2.0)[index],
                sign_shift[index],
            )
            for index, values in enumerate(frames)
        ),
        mf,
    )
    coefficients = tuple(
        phase_plan.inspiral[index] for index in _DYNAMIC_INDICES
    )
    intermediate = torch.stack(solved)
    return (
        mf,
        ringdown_21_raw,
        coefficients,
        intermediate,
        ringdown,
        scalars,
        anchor_values,
    )


def _executor_key(inputs):
    mf = inputs[0]
    device = mf.device
    return (
        getattr(torch, "__version__", None),
        device.type,
        device.index,
        mf.dtype,
        mf.numel(),
    )


def _ensure_process_cache_locked():
    """Recover conservatively if the at-fork hook was unavailable or missed."""

    global _CACHE_PID
    pid = os.getpid()
    if _CACHE_PID != pid:
        _EXECUTORS.clear()
        _FAILED_KEYS.clear()
        _CACHE_PID = pid


def _remember_failure_locked(key):
    _EXECUTORS.pop(key, None)
    _FAILED_KEYS[key] = None
    _FAILED_KEYS.move_to_end(key)
    while len(_FAILED_KEYS) > _MAX_FAILED_KEYS:
        _FAILED_KEYS.popitem(last=False)


def _remember_failure(key):
    with _LOCK:
        _ensure_process_cache_locked()
        _remember_failure_locked(key)


def _compiled_canary_supported(expected, actual):
    """Validate traced output against the eager source before cache admission."""

    if (
        type(expected) is not tuple
        or type(actual) is not tuple
        or len(expected) != 3
        or len(actual) != 3
    ):
        return False
    epsilon = torch.finfo(torch.float64).eps
    for reference, candidate in zip(expected, actual):
        if (
            type(reference) is not torch.Tensor
            or type(candidate) is not torch.Tensor
            or candidate.layout is not reference.layout
            or candidate.shape != reference.shape
            or candidate.dtype != reference.dtype
            or candidate.device != reference.device
        ):
            return False
        try:
            reference_finite = torch.isfinite(reference)
            candidate_finite = torch.isfinite(candidate)
            if not bool(torch.all(reference_finite)):
                return False
            if not bool(torch.all(candidate_finite)):
                return False
            if not torch.equal(candidate == 0, reference == 0):
                return False
            reference_norm = torch.linalg.vector_norm(reference)
            difference_norm = torch.linalg.vector_norm(candidate - reference)
            if bool(reference_norm == 0):
                if bool(difference_norm != 0):
                    return False
            elif bool(difference_norm > 256.0 * epsilon * reference_norm):
                return False
        except Exception:
            return False
    return True


def _get_executor(inputs):
    key = _executor_key(inputs)
    with _LOCK:
        _ensure_process_cache_locked()
        if key in _FAILED_KEYS:
            _FAILED_KEYS.move_to_end(key)
            return None
        executor = _EXECUTORS.get(key)
        if executor is not None:
            _EXECUTORS.move_to_end(key)
            return executor
        try:
            executor = torch.jit.trace(
                _phase_triplet_lane_source,
                inputs,
                check_trace=False,
                strict=True,
            )
            expected = _phase_triplet_lane_source(*inputs)
            actual = executor(*inputs)
            if not _compiled_canary_supported(expected, actual):
                raise RuntimeError("compiled phase-triplet canary failed")
        except Exception:
            _remember_failure_locked(key)
            return None
        _EXECUTORS[key] = executor
        _EXECUTORS.move_to_end(key)
        while len(_EXECUTORS) > _MAX_EXECUTORS:
            _EXECUTORS.popitem(last=False)
        return executor


def _evaluate_scripted_phase_triplet(shared, states, phase_plan, anchors):
    """Return three compiled phases, or ``None`` on any uncertain condition."""

    key = _executor_key((shared.mf,))
    try:
        inputs = _prepare_lane_inputs(shared, states, phase_plan, anchors)
        executor = _get_executor(inputs)
        if executor is None:
            return None
        result = executor(*inputs)
    except Exception:
        _remember_failure(key)
        return None
    if type(result) is not tuple or len(result) != 3:
        _remember_failure(key)
        return None
    mf = shared.mf
    if not all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == mf.shape
        and value.dtype == mf.dtype
        and value.device == mf.device
        and not value.requires_grad
        and value.grad_fn is None
        for value in result
    ):
        _remember_failure(key)
        return None
    return result
