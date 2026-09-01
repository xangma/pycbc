# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Exact fixed-schema evaluation of the XHM 21/33/44 amplitudes."""

from __future__ import annotations

import os
import threading
from typing import Callable, NamedTuple

import torch

from .torch_switches import _parse_switch


_FIXED_SCHEMA_AMPLITUDE_TRIPLET_ENV = (
    "PYCBC_IMRPHENOMXHM_FIXED_SCHEMA_AMPLITUDE_TRIPLET"
)


class _FixedSchemaAmplitudePlan(NamedTuple):
    """Inputs to one unchanged final amplitude expression tree."""

    coefficients: torch.Tensor
    pseudo: torch.Tensor
    intermediate_coefficients: torch.Tensor
    tail_amplitude: torch.Tensor
    tail_decay: torch.Tensor
    amp_norm: float
    global_factor: float
    pn_dominant: float
    fcut_inspiral: float
    ringdown_a0: float
    ringdown_sigma: float
    ringdown_decay: float
    f_ring: float
    f_damp: float
    ffalloff: float
    fmatch_ringdown: float


def _fixed_schema_amplitude_triplet_enabled():
    """Return the strict fixed-schema switch."""

    value = os.environ.get(_FIXED_SCHEMA_AMPLITUDE_TRIPLET_ENV)
    return (
        True
        if value is None
        else _parse_switch(_FIXED_SCHEMA_AMPLITUDE_TRIPLET_ENV, value)
    )


def _single_amplitude21(
    mf: torch.Tensor,
    cube_root: torch.Tensor,
    leading_power: torch.Tensor,
    coefficients: torch.Tensor,
    pseudo: torch.Tensor,
    intermediate_coefficients: torch.Tensor,
    tail_amplitude: torch.Tensor,
    tail_decay: torch.Tensor,
    amp_norm: float,
    global_factor: float,
    pn_dominant: float,
    fcut: float,
    a0: float,
    sigma: float,
    decay: float,
    f_ring: float,
    f_damp: float,
    ffalloff: float,
    fmatch: float,
) -> torch.Tensor:
    """Evaluate mode 21 in its original operation and reduction order."""

    series = coefficients[6]
    series = series * cube_root + coefficients[5]
    series = series * cube_root + coefficients[4]
    series = series * cube_root + coefficients[3]
    series = series * cube_root + coefficients[2]
    series = series * cube_root + coefficients[1]
    series = series * cube_root + coefficients[0]
    pn = torch.abs(series) * global_factor * leading_power * amp_norm

    ratio = mf / fcut
    pseudo_terms = (
        pseudo[0] * ratio ** (7.0 / 3.0)
        + pseudo[1] * ratio ** (8.0 / 3.0)
        + pseudo[2] * ratio**3
    )
    inspiral = pn + pn_dominant * leading_power * pseudo_terms

    polynomial = intermediate_coefficients[4]
    polynomial = polynomial * mf + intermediate_coefficients[3]
    polynomial = polynomial * mf + intermediate_coefficients[2]
    polynomial = polynomial * mf + intermediate_coefficients[1]
    polynomial = polynomial * mf + intermediate_coefficients[0]
    intermediate = leading_power * polynomial

    offset = mf - f_ring
    width = f_damp * sigma
    core = a0 * f_damp / (
        torch.exp(decay * offset / width)
        * (offset * offset + width * width)
    )
    tail = tail_amplitude * torch.exp(-tail_decay * (mf - ffalloff))
    ringdown = torch.where(mf < ffalloff, core, tail)
    amplitude = torch.where(
        mf <= fcut,
        inspiral,
        torch.where(mf <= fmatch, intermediate, ringdown),
    )
    return torch.where(amplitude < 0.0, 1.0e-15, amplitude)


def _single_amplitude3344(
    mf: torch.Tensor,
    cube_root: torch.Tensor,
    leading_power: torch.Tensor,
    intermediate_zero: torch.Tensor,
    intermediate_power0: torch.Tensor,
    intermediate_power1: torch.Tensor,
    intermediate_power2: torch.Tensor,
    intermediate_power3: torch.Tensor,
    intermediate_power4: torch.Tensor,
    intermediate_power5: torch.Tensor,
    intermediate_power6: torch.Tensor,
    intermediate_power7: torch.Tensor,
    coefficients: torch.Tensor,
    pseudo: torch.Tensor,
    intermediate_coefficients: torch.Tensor,
    tail_amplitude: torch.Tensor,
    tail_decay: torch.Tensor,
    amp_norm: float,
    global_factor: float,
    pn_dominant: float,
    fcut: float,
    a0: float,
    sigma: float,
    decay: float,
    f_ring: float,
    f_damp: float,
    ffalloff: float,
    fmatch: float,
) -> torch.Tensor:
    """Evaluate one 33/44 mode without crossing its arithmetic tree."""

    series = coefficients[6]
    series = series * cube_root + coefficients[5]
    series = series * cube_root + coefficients[4]
    series = series * cube_root + coefficients[3]
    series = series * cube_root + coefficients[2]
    series = series * cube_root + coefficients[1]
    series = series * cube_root + coefficients[0]
    pn = torch.abs(series) * global_factor * leading_power * amp_norm

    ratio = mf / fcut
    pseudo_terms = (
        pseudo[0] * ratio ** (7.0 / 3.0)
        + pseudo[1] * ratio ** (8.0 / 3.0)
        + pseudo[2] * ratio**3
    )
    inspiral = pn + pn_dominant * leading_power * pseudo_terms

    polynomial = (
        intermediate_zero
        + intermediate_coefficients[0] * intermediate_power0
    )
    polynomial = (
        polynomial + intermediate_coefficients[1] * intermediate_power1
    )
    polynomial = (
        polynomial + intermediate_coefficients[2] * intermediate_power2
    )
    polynomial = (
        polynomial + intermediate_coefficients[3] * intermediate_power3
    )
    polynomial = (
        polynomial + intermediate_coefficients[4] * intermediate_power4
    )
    polynomial = (
        polynomial + intermediate_coefficients[5] * intermediate_power5
    )
    polynomial = (
        polynomial + intermediate_coefficients[6] * intermediate_power6
    )
    polynomial = (
        polynomial + intermediate_coefficients[7] * intermediate_power7
    )
    intermediate = leading_power * polynomial

    offset = mf - f_ring
    width = f_damp * sigma
    core = a0 * f_damp / (
        torch.exp(decay * offset / width)
        * (offset * offset + width * width)
    )
    tail = tail_amplitude * torch.exp(-tail_decay * (mf - ffalloff))
    ringdown = torch.where(mf < ffalloff, core, tail)
    amplitude = torch.where(
        mf <= fcut,
        inspiral,
        torch.where(mf <= fmatch, intermediate, ringdown),
    )
    return torch.where(amplitude < 0.0, 1.0e-15, amplitude)


def _fixed_schema_amplitude_triplet(
    mf: torch.Tensor,
    coefficients21: torch.Tensor,
    pseudo21: torch.Tensor,
    intermediate21: torch.Tensor,
    tail_amplitude21: torch.Tensor,
    tail_decay21: torch.Tensor,
    amp_norm21: float,
    global_factor21: float,
    pn_dominant21: float,
    fcut21: float,
    a021: float,
    sigma21: float,
    decay21: float,
    f_ring21: float,
    f_damp21: float,
    ffalloff21: float,
    fmatch21: float,
    coefficients33: torch.Tensor,
    pseudo33: torch.Tensor,
    intermediate33: torch.Tensor,
    tail_amplitude33: torch.Tensor,
    tail_decay33: torch.Tensor,
    amp_norm33: float,
    global_factor33: float,
    pn_dominant33: float,
    fcut33: float,
    a033: float,
    sigma33: float,
    decay33: float,
    f_ring33: float,
    f_damp33: float,
    ffalloff33: float,
    fmatch33: float,
    coefficients44: torch.Tensor,
    pseudo44: torch.Tensor,
    intermediate44: torch.Tensor,
    tail_amplitude44: torch.Tensor,
    tail_decay44: torch.Tensor,
    amp_norm44: float,
    global_factor44: float,
    pn_dominant44: float,
    fcut44: float,
    a044: float,
    sigma44: float,
    decay44: float,
    f_ring44: float,
    f_damp44: float,
    ffalloff44: float,
    fmatch44: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate three independent modes while sharing identical mf powers."""

    cube_root = mf ** (1.0 / 3.0)
    leading_power = mf ** (-7.0 / 6.0)
    amplitude21 = _single_amplitude21(
        mf,
        cube_root,
        leading_power,
        coefficients21,
        pseudo21,
        intermediate21,
        tail_amplitude21,
        tail_decay21,
        amp_norm21,
        global_factor21,
        pn_dominant21,
        fcut21,
        a021,
        sigma21,
        decay21,
        f_ring21,
        f_damp21,
        ffalloff21,
        fmatch21,
    )

    intermediate_zero = torch.zeros_like(mf)
    intermediate_power0 = torch.ones_like(mf)
    intermediate_power1 = intermediate_power0 * mf
    intermediate_power2 = intermediate_power1 * mf
    intermediate_power3 = intermediate_power2 * mf
    intermediate_power4 = intermediate_power3 * mf
    intermediate_power5 = intermediate_power4 * mf
    intermediate_power6 = intermediate_power5 * mf
    intermediate_power7 = intermediate_power6 * mf
    amplitude33 = _single_amplitude3344(
        mf,
        cube_root,
        leading_power,
        intermediate_zero,
        intermediate_power0,
        intermediate_power1,
        intermediate_power2,
        intermediate_power3,
        intermediate_power4,
        intermediate_power5,
        intermediate_power6,
        intermediate_power7,
        coefficients33,
        pseudo33,
        intermediate33,
        tail_amplitude33,
        tail_decay33,
        amp_norm33,
        global_factor33,
        pn_dominant33,
        fcut33,
        a033,
        sigma33,
        decay33,
        f_ring33,
        f_damp33,
        ffalloff33,
        fmatch33,
    )
    amplitude44 = _single_amplitude3344(
        mf,
        cube_root,
        leading_power,
        intermediate_zero,
        intermediate_power0,
        intermediate_power1,
        intermediate_power2,
        intermediate_power3,
        intermediate_power4,
        intermediate_power5,
        intermediate_power6,
        intermediate_power7,
        coefficients44,
        pseudo44,
        intermediate44,
        tail_amplitude44,
        tail_decay44,
        amp_norm44,
        global_factor44,
        pn_dominant44,
        fcut44,
        a044,
        sigma44,
        decay44,
        f_ring44,
        f_damp44,
        ffalloff44,
        fmatch44,
    )
    return amplitude21, amplitude33, amplitude44


_FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR: Callable | None = None
_FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED = False
_FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED = False
_FIXED_SCHEMA_AMPLITUDE_TRIPLET_PID = os.getpid()
_FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK = threading.Lock()
_FIXED_SCHEMA_AMPLITUDE_TRIPLET_COMPILER = torch.jit.script
_AT_FORK_REGISTERED = hasattr(os, "register_at_fork")


def _after_fixed_schema_amplitude_triplet_fork():
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_PID
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK

    _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR = None
    _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED = False
    _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED = False
    _FIXED_SCHEMA_AMPLITUDE_TRIPLET_PID = os.getpid()
    _FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK = threading.Lock()


if _AT_FORK_REGISTERED:
    os.register_at_fork(after_in_child=_after_fixed_schema_amplitude_triplet_fork)


def _clear_fixed_schema_amplitude_triplet_cache():
    """Clear the process-local scripted executor and failure latch."""

    _after_fixed_schema_amplitude_triplet_fork()


def _get_fixed_schema_amplitude_triplet_executor():
    """Build the fixed-schema executor once per process, failing closed."""

    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED

    if (
        not _AT_FORK_REGISTERED
        and _FIXED_SCHEMA_AMPLITUDE_TRIPLET_PID != os.getpid()
    ):
        _after_fixed_schema_amplitude_triplet_fork()
    if _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR is not None:
        return _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
    if _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED:
        return None
    with _FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK:
        if _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR is not None:
            return _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
        if _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED:
            return None
        try:
            _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR = (
                _FIXED_SCHEMA_AMPLITUDE_TRIPLET_COMPILER(
                    _fixed_schema_amplitude_triplet
                )
            )
        except Exception:
            _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED = True
            return None
        return _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR


def _mark_fixed_schema_amplitude_triplet_failed():
    """Latch a runtime executor failure for the rest of this process."""

    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED

    with _FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK:
        _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR = None
        _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED = True
        _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED = False


def _fixed_schema_amplitude_triplet_output_supported(result, mf):
    """Validate the cheap structural contract of one executor result."""

    return (
        type(result) is tuple
        and len(result) == 3
        and all(
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.shape == mf.shape
            and value.dtype == mf.dtype
            and value.device == mf.device
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value._base is None
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and value.grad_fn is None
            for value in result
        )
    )


def _evaluate_fixed_schema_amplitude_triplet(mf, plan21, plan33, plan44):
    """Return three exact amplitudes, or ``None`` for eager fallback."""

    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED
    global _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED

    executor = _get_fixed_schema_amplitude_triplet_executor()
    if executor is None:
        return None
    inputs = (mf, *plan21, *plan33, *plan44)
    try:
        result = executor(*inputs)
    except Exception:
        _mark_fixed_schema_amplitude_triplet_failed()
        return None
    if not _fixed_schema_amplitude_triplet_output_supported(result, mf):
        _mark_fixed_schema_amplitude_triplet_failed()
        return None

    # Admit a freshly scripted executor only after its first live request is
    # byte-exact against the unchanged eager source.  This is a cold-start-only
    # check; warm calls retain only the constant-time structure validation.
    with _FIXED_SCHEMA_AMPLITUDE_TRIPLET_LOCK:
        if (
            _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED
            or executor is not _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR
        ):
            return None
        if not _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED:
            try:
                expected = _fixed_schema_amplitude_triplet(*inputs)
                exact = _fixed_schema_amplitude_triplet_output_supported(
                    expected,
                    mf,
                ) and all(
                    torch.equal(
                        reference.view(torch.uint8),
                        candidate.view(torch.uint8),
                    )
                    for reference, candidate in zip(expected, result)
                )
            except Exception:
                exact = False
            if not exact:
                _FIXED_SCHEMA_AMPLITUDE_TRIPLET_EXECUTOR = None
                _FIXED_SCHEMA_AMPLITUDE_TRIPLET_FAILED = True
                _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED = False
                return None
            _FIXED_SCHEMA_AMPLITUDE_TRIPLET_ADMITTED = True
    return result
