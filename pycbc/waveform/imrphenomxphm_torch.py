# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native default-mode IMRPhenomXPHM waveforms.

The native path combines the existing XHM co-precessing modes with the XP MSA
Euler angles.  Mode generation, Wigner rotations, and polarization assembly
execute on the active Torch device; scalar source-frame setup remains on the
host.  The bounded implementation covers the default XPHM mode set and explicit
subsets of its positive-m co-precessing families with MSA precession version
223 (and its 300 alias), convention 1, and final-spin modes 0, 3, and 4.
Other configurations continue to use lalsimulation.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import math
import os
from numbers import Integral
import struct
import sys
import threading
from typing import Any, NamedTuple

import torch
import numpy as _np

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.parameters import Parameter as _WaveformParameter
from pycbc.waveform._imrphenomxphm_cuda_aggregate_cache import (  # noqa: F401
    CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV
    as _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV,  # noqa: F401
    CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEBUG_FINGERPRINT_ENV
    as _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEBUG_FINGERPRINT_ENV,  # noqa: F401
    CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV
    as _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV,  # noqa: F401
    CUDA_AGGREGATE_PRETERMINAL_TWIST_PUBLIC_FASTPATH_ENV
    as _CUDA_AGGREGATE_PRETERMINAL_TWIST_PUBLIC_FASTPATH_ENV,  # noqa: F401
    make_cuda_aggregate_preterminal_twist_cache
    as _make_cuda_aggregate_preterminal_twist_cache,
)
from pycbc.waveform._imrphenomxphm_cuda_plan_cache import (  # noqa: F401
    CUDA_COPRECESSING_PLAN_CACHE_BYTES_ENV
    as _CUDA_COPRECESSING_PLAN_CACHE_BYTES_ENV,  # noqa: F401
    CUDA_COPRECESSING_PLAN_CACHE_DEBUG_FINGERPRINT_ENV
    as _CUDA_COPRECESSING_PLAN_CACHE_DEBUG_FINGERPRINT_ENV,  # noqa: F401
    CUDA_COPRECESSING_PLAN_CACHE_ENV
    as _CUDA_COPRECESSING_PLAN_CACHE_ENV,  # noqa: F401
    make_cuda_coprecessing_plan_cache as _make_cuda_coprecessing_plan_cache,
)
from pycbc.waveform._imrphenomxphm_public_result_cache import (  # noqa: F401
    PUBLIC_RESULT_CACHE_BYTES_ENV as _PUBLIC_RESULT_CACHE_BYTES_ENV,  # noqa: F401
    PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT_ENV
    as _PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT_ENV,  # noqa: F401
    PUBLIC_RESULT_CACHE_ENV as _PUBLIC_RESULT_CACHE_ENV,  # noqa: F401
    make_public_result_cache as _make_public_result_cache,
)
from pycbc.waveform._spherical_harmonics_torch import (
    cudagraphed_spin_minus_two_spherical_harmonics_phi_zero,
    scripted_spin_minus_two_spherical_harmonics_phi_zero,
    spin_minus_two_spherical_harmonics_phi_zero,
    spin_weighted_spherical_harmonic,
    vectorized_spin_minus_two_spherical_harmonics_phi_zero,
)
from pycbc.waveform.imrphenomxas_torch import (
    _XAS_MODE_POLARIZATION_FACTOR,
    _next_power_of_two,
    _request_proof_plan_current,
    _run_xphm_request_proof_plan,
)
from pycbc.waveform import imrphenomxas_torch as _request_proof_owner
from pycbc.waveform.imrphenomxhm_torch import (
    _SequenceCore,
    _active_mode_samples,
    _carrier_alignment_result_reuse_enabled,
    _phase_anchor_cache_enabled,
    _plain_request_runtime_supported as _xhm_plain_request_runtime_supported,
)
from pycbc.waveform.imrphenomxp_msa_torch import (
    _reference_and_mode_msa_angles,
    msa_angles,
    msa_angles_batch,
)
from pycbc.waveform.imrphenomxp_torch import (
    _MSA_CONVENTION,
    _MSA_FINAL_SPIN_MODES,
    _MSA_PREC_VERSIONS,
    _as_float,
    _build_model,
    _integer_or_default,
    _sequence_frequencies,
    _series_from_active_samples,
    _validated_inputs,
    _xas_samples,
    imrphenomxp_native_supported,
)
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils
from pycbc.waveform.torch_switches import _parse_switch


_PI = math.pi
_DEFAULT_PREC_VERSION = 300
_DEFAULT_CONVENTION = 1
_DEFAULT_FINAL_SPIN = 4
_COPRECESSING_MODES = (
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
)
_POSITIVE_COPRECESSING_MODES = frozenset(_COPRECESSING_MODES)
_INTRINSIC_CACHE_ENV = "PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE"
_CARRIER_AMP_PLAN_REUSE_ENV = (
    "PYCBC_IMRPHENOMXPHM_CARRIER_AMP_PLAN_REUSE"
)
_PACKED_REMNANT_PLAN_ENV = "PYCBC_IMRPHENOMXPHM_PACKED_REMNANT_PLAN"
_MODE_ANGLE_REUSE_ENV = "PYCBC_IMRPHENOMXPHM_MODE_ANGLE_REUSE"
_BULK_MODE_ANGLES_ENV = "PYCBC_IMRPHENOMXPHM_BULK_MODE_ANGLES"
_REFERENCE_BULK_ANGLE_LANE_ENV = (
    "PYCBC_IMRPHENOMXPHM_REFERENCE_BULK_ANGLE_LANE"
)
_TWIST_REUSE_ENV = "PYCBC_IMRPHENOMXPHM_TWIST_REUSE"
_BULK_TWIST_EXPONENTIALS_ENV = (
    "PYCBC_IMRPHENOMXPHM_BULK_TWIST_EXPONENTIALS"
)
_TWIST_EXPONENTIAL_RECURRENCE_ENV = (
    "PYCBC_IMRPHENOMXPHM_TWIST_EXPONENTIAL_RECURRENCE"
)
_BULK_TWIST_HARMONICS_ENV = "PYCBC_IMRPHENOMXPHM_BULK_TWIST_HARMONICS"
_FUSED_CPU_TWIST_ENV = "PYCBC_IMRPHENOMXPHM_FUSED_CPU_TWIST"
_SCRIPTED_TWIST_HARMONICS_ENV = (
    "PYCBC_IMRPHENOMXPHM_SCRIPTED_TWIST_HARMONICS"
)
_CUDAGRAPH_TWIST_HARMONICS_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDAGRAPH_TWIST_HARMONICS"
)
_VECTORIZED_TWIST_HARMONICS_ENV = (
    "PYCBC_IMRPHENOMXPHM_VECTORIZED_TWIST_HARMONICS"
)
_STACKED_TWIST_ENV = "PYCBC_IMRPHENOMXPHM_STACKED_TWIST"
_GROUPED_OUTER_TWIST_ENV = "PYCBC_IMRPHENOMXPHM_GROUPED_OUTER_TWIST"
_GROUPED_OUTER_TWIST_CUDA_GRAPH_ENV = (
    "PYCBC_IMRPHENOMXPHM_GROUPED_OUTER_TWIST_CUDA_GRAPH"
)
_TRUSTED_PLAIN_REQUEST_ENV = (
    "PYCBC_IMRPHENOMXPHM_TRUSTED_PLAIN_REQUEST"
)
_INFERENCE_MODE_ENV = "PYCBC_IMRPHENOMXPHM_INFERENCE_MODE"
_COPRECESSING_PLAN_CACHE_ENV = (
    "PYCBC_IMRPHENOMXPHM_COPRECESSING_PLAN_CACHE"
)
_COPRECESSING_PLAN_CACHE_COLD_MISS_ONE_PASS_ENV = (
    "PYCBC_IMRPHENOMXPHM_COPRECESSING_PLAN_CACHE_COLD_MISS_ONE_PASS"
)
_COPRECESSING_PLAN_ANGLE_CORE_ENV = (
    "PYCBC_IMRPHENOMXPHM_COPRECESSING_PLAN_ANGLE_CORE"
)
_COPRECESSING_PLAN_CACHE_BYTES_ENV = (
    "PYCBC_IMRPHENOMXPHM_COPRECESSING_PLAN_CACHE_BYTES"
)
_COPRECESSING_PLAN_CACHE_DEFAULT_BYTES = 64 * 1024 * 1024
_COPRECESSING_PLAN_CACHE_MAX_ENTRIES = 32
_COPRECESSING_PLAN_COLD_MISS_HINT_MAX_ENTRIES = 128
_COPRECESSING_PLAN_CACHE_ENTRY_OVERHEAD = 1024
_COPRECESSING_PLAN_CACHE_IMPLEMENTATION = 7
_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV = (
    "PYCBC_IMRPHENOMXPHM_AGGREGATE_PRETERMINAL_TWIST_CACHE"
)
_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV = (
    "PYCBC_IMRPHENOMXPHM_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES"
)
_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEFAULT_BYTES = 16 * 1024 * 1024
_AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES = 32
_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENTRY_OVERHEAD = 512
_AGGREGATE_PRETERMINAL_TWIST_CACHE_IMPLEMENTATION = 2
_COPRECESSING_PLAN_PHASE_TABLE_SOURCE = (
    IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
)
_COPRECESSING_PLAN_AMP_TABLE_SOURCE = (
    IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
)

_INFERENCE_REQUIRED_SWITCHES = (
    "PYCBC_IMRPHENOMX_PHASE_PLAN",
    "PYCBC_IMRPHENOMX_AMP_PLAN",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_DERIVATIVES",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_AMP_DERIVATIVES",
)
_INFERENCE_MODE32_REQUIRED_SWITCHES = (
    "PYCBC_IMRPHENOMXHM_MODE32_DERIVATIVE_REGION_SPECIALIZATION",
)
_COPRECESSING_PLAN_COLD_MISS_HINT_FIELDS = (
    "approximant",
    "mass1",
    "mass2",
    "spin1x",
    "spin1y",
    "spin1z",
    "spin2x",
    "spin2y",
    "spin2z",
    "distance",
    "coa_phase",
    "f_ref",
    "delta_f",
    "f_lower",
    "f_final",
    "lambda1",
    "lambda2",
    "dquad_mon1",
    "dquad_mon2",
    "phenom_x_prec_version",
    "phenom_xp_convention",
    "phenom_xp_final_spin_mod",
    "n_batch",
)


class _ReferenceModeMSAAngleCore(NamedTuple):
    """Raw intrinsic MSA rows before every request-local angle transform."""

    mprimes: tuple
    mode_phiz: torch.Tensor
    mode_zeta: torch.Tensor
    mode_cos_beta: torch.Tensor
    reference_phiz_residual: float
    reference_zeta_residual: float


class _CoprecessingPlan(NamedTuple):
    """Private intrinsic tensors for one exact co-precessing cache key."""

    carrier: torch.Tensor
    active_modes: tuple
    reference_angle_core: _ReferenceModeMSAAngleCore | None = None


class _CoprecessingPlanCacheEntry(NamedTuple):
    """One immutable LRU value and its byte/fingerprint accounting."""

    plan: _CoprecessingPlan
    fingerprints: tuple
    nbytes: int
    generation: int


class _CoprecessingPlanCacheToken(NamedTuple):
    """Collision-safe identity for one validated, PID-local plan entry."""

    key: tuple
    generation: int
    pid: int


class _AggregatePreterminalTwistCacheEntry(NamedTuple):
    """Private aggregate polarizations before the two terminal rotations."""

    plus: torch.Tensor
    cross: torch.Tensor
    fingerprints: tuple
    nbytes: int


class _CoprecessingPlanCacheColdMiss(RuntimeError):
    """Request one ordinary cold rebuild before populating the cache."""


def _new_coprecessing_plan_cache_counters():
    return {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
        "invalidations": 0,
        "ineligible": 0,
        "canary_failures": 0,
        "oversize": 0,
        "races": 0,
    }


_COPRECESSING_PLAN_CACHE = OrderedDict()
_COPRECESSING_PLAN_CACHE_LOCK = threading.RLock()
_COPRECESSING_PLAN_CACHE_PID = os.getpid()
_COPRECESSING_PLAN_CACHE_SIZE = 0
_COPRECESSING_PLAN_CACHE_GENERATION = 0
_COPRECESSING_PLAN_CACHE_COUNTERS = (
    _new_coprecessing_plan_cache_counters()
)
_COPRECESSING_PLAN_COLD_MISS_HINTS = OrderedDict()
_COPRECESSING_PLAN_CACHE_COLD_RETRY = ContextVar(
    "pycbc_imrphenomxphm_coprecessing_plan_cache_cold_retry",
    default=False,
)


def _new_aggregate_preterminal_twist_cache_counters():
    return {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
        "invalidations": 0,
        "ineligible": 0,
        "canary_failures": 0,
        "oversize": 0,
        "races": 0,
    }


_AGGREGATE_PRETERMINAL_TWIST_CACHE = OrderedDict()
_AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK = threading.RLock()
_AGGREGATE_PRETERMINAL_TWIST_CACHE_PID = os.getpid()
_AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE = 0
_AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS = (
    _new_aggregate_preterminal_twist_cache_counters()
)


def _intrinsic_cache_enabled():
    """Return the strict debug switch for one-call XPHM fit caching."""

    value = os.environ.get(_INTRINSIC_CACHE_ENV)
    return True if value is None else _parse_switch(_INTRINSIC_CACHE_ENV, value)


def _carrier_amp_plan_reuse_enabled():
    """Return the strict switch for sharing XAS's exact carrier amp plan."""

    value = os.environ.get(_CARRIER_AMP_PLAN_REUSE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CARRIER_AMP_PLAN_REUSE_ENV, value)
    )


def _packed_remnant_plan_enabled():
    """Return the strict switch for the request-local two-remnant lane."""

    value = os.environ.get(_PACKED_REMNANT_PLAN_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PACKED_REMNANT_PLAN_ENV, value)
    )


def _mode_angle_reuse_enabled():
    """Return the strict switch for request-local MSA mode-angle reuse."""

    value = os.environ.get(_MODE_ANGLE_REUSE_ENV)
    return False if value is None else _parse_switch(_MODE_ANGLE_REUSE_ENV, value)


def _bulk_mode_angles_enabled():
    """Return the strict switch for one-call MSA mode-angle evaluation."""

    value = os.environ.get(_BULK_MODE_ANGLES_ENV)
    return False if value is None else _parse_switch(_BULK_MODE_ANGLES_ENV, value)


def _reference_bulk_angle_lane_enabled():
    """Return the strict switch for the reference-plus-mode angle lane."""

    value = os.environ.get(_REFERENCE_BULK_ANGLE_LANE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_REFERENCE_BULK_ANGLE_LANE_ENV, value)
    )


def _twist_reuse_enabled():
    """Return the strict switch for request-local twist-term reuse."""

    value = os.environ.get(_TWIST_REUSE_ENV)
    return False if value is None else _parse_switch(_TWIST_REUSE_ENV, value)


def _bulk_twist_exponentials_enabled():
    """Return the strict switch for packed twist exponentials."""

    value = os.environ.get(_BULK_TWIST_EXPONENTIALS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_BULK_TWIST_EXPONENTIALS_ENV, value)
    )


def _twist_exponential_recurrence_enabled():
    """Return the strict switch for LAL's twist-exponential recurrence."""

    value = os.environ.get(_TWIST_EXPONENTIAL_RECURRENCE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_TWIST_EXPONENTIAL_RECURRENCE_ENV, value)
    )


def _bulk_twist_harmonics_enabled():
    """Return the strict switch for one-call twist-harmonic evaluation."""

    value = os.environ.get(_BULK_TWIST_HARMONICS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_BULK_TWIST_HARMONICS_ENV, value)
    )


def _scripted_twist_harmonics_enabled():
    """Return the strict switch for cached scripted twist harmonics."""

    value = os.environ.get(_SCRIPTED_TWIST_HARMONICS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_SCRIPTED_TWIST_HARMONICS_ENV, value)
    )


def _cudagraph_twist_harmonics_enabled():
    """Return the strict switch for exact CUDA-Graph twist harmonics."""

    value = os.environ.get(_CUDAGRAPH_TWIST_HARMONICS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CUDAGRAPH_TWIST_HARMONICS_ENV, value)
    )


def _vectorized_twist_harmonics_enabled():
    """Return the strict switch for exact internal harmonic mode lanes."""

    value = os.environ.get(_VECTORIZED_TWIST_HARMONICS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_VECTORIZED_TWIST_HARMONICS_ENV, value)
    )


def _stacked_twist_enabled():
    """Return the strict switch for exact native m-axis twist assembly."""

    value = os.environ.get(_STACKED_TWIST_ENV)
    return False if value is None else _parse_switch(_STACKED_TWIST_ENV, value)


def _grouped_outer_twist_enabled():
    """Return the strict switch for CUDA mode-lane twist assembly."""

    value = os.environ.get(_GROUPED_OUTER_TWIST_ENV)
    return (
        False
        if value is None
        else _parse_switch(_GROUPED_OUTER_TWIST_ENV, value)
    )


def _grouped_outer_twist_cuda_graph_enabled():
    """Return the strict, off-by-default grouped CUDA-Graph switch."""

    value = os.environ.get(_GROUPED_OUTER_TWIST_CUDA_GRAPH_ENV)
    return (
        False
        if value is None
        else _parse_switch(_GROUPED_OUTER_TWIST_CUDA_GRAPH_ENV, value)
    )


def _fused_cpu_twist_enabled():
    """Return the strict switch for exact fused CPU twisting."""

    value = os.environ.get(_FUSED_CPU_TWIST_ENV)
    return True if value is None else _parse_switch(_FUSED_CPU_TWIST_ENV, value)


def _fused_cpu_twist_supported(frequencies, modes, grouped_twist_device):
    """Return whether fused CPU twisting is valid and unshadowed."""

    if not _fused_cpu_twist_enabled():
        return False
    if any(
        name in os.environ
        for name in (
            _TWIST_REUSE_ENV,
            _MODE_ANGLE_REUSE_ENV,
            _BULK_MODE_ANGLES_ENV,
            _STACKED_TWIST_ENV,
            _BULK_TWIST_EXPONENTIALS_ENV,
            _BULK_TWIST_HARMONICS_ENV,
        )
    ):
        return False
    if (
        frequencies.device.type != "cpu"
        or frequencies.dtype != torch.float64
        or not all(ell in (2, 3, 4) and mprime in (1, 2, 3, 4) for ell, mprime in modes)
        or grouped_twist_device is not None
    ):
        return False
    return True


def _trusted_plain_request_enabled():
    """Return the strict switch for one validated plain request scope."""

    value = os.environ.get(_TRUSTED_PLAIN_REQUEST_ENV)
    return (
        True
        if value is None
        else _parse_switch(_TRUSTED_PLAIN_REQUEST_ENV, value)
    )


def _inference_mode_enabled():
    """Return the strict switch for validated CPU inference execution."""

    value = os.environ.get(_INFERENCE_MODE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INFERENCE_MODE_ENV, value)
    )


def _coprecessing_plan_cache_enabled():
    """Return the strict, off-by-default cross-request cache switch.

    Disable this gate or clear the cache before live debugging changes any
    transitive waveform producer.  The cache key guards supported request and
    runtime state, not arbitrary monkeypatches below its direct producer roots.
    """

    value = os.environ.get(_COPRECESSING_PLAN_CACHE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_COPRECESSING_PLAN_CACHE_ENV, value)
    )


def _coprecessing_plan_cache_cold_miss_one_pass_enabled():
    """Return the strict switch for one-pass known-cold cache requests."""

    value = os.environ.get(
        _COPRECESSING_PLAN_CACHE_COLD_MISS_ONE_PASS_ENV
    )
    return (
        False
        if value is None
        else _parse_switch(
            _COPRECESSING_PLAN_CACHE_COLD_MISS_ONE_PASS_ENV,
            value,
        )
    )


def _coprecessing_plan_angle_core_enabled():
    """Return the strict, nested switch for raw MSA-core retention."""

    value = os.environ.get(_COPRECESSING_PLAN_ANGLE_CORE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_COPRECESSING_PLAN_ANGLE_CORE_ENV, value)
    )


def _coprecessing_plan_cache_budget():
    """Return the positive byte budget, or ``None`` for invalid input."""

    value = os.environ.get(_COPRECESSING_PLAN_CACHE_BYTES_ENV)
    if value is None:
        return _COPRECESSING_PLAN_CACHE_DEFAULT_BYTES
    if not value.isascii() or not value.isdecimal():
        return None
    try:
        budget = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return budget if budget > 0 else None


def _aggregate_preterminal_twist_cache_enabled():
    """Return the strict, independently gated aggregate-twist cache switch."""

    value = os.environ.get(_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV, value)
    )


def _aggregate_preterminal_twist_cache_budget():
    """Return the positive aggregate-cache byte budget, else ``None``."""

    value = os.environ.get(
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV
    )
    if value is None:
        return _AGGREGATE_PRETERMINAL_TWIST_CACHE_DEFAULT_BYTES
    if not value.isascii() or not value.isdecimal():
        return None
    try:
        budget = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return budget if budget > 0 else None


def _plain_request_tree_supported(value):
    """Accept only exact built-in containers and scalar leaves.

    This deliberately rejects every tensor, array, scalar subclass, and
    user-defined container before either request-level promise is enabled.
    The normal implementation remains available for all such inputs.
    """

    if type(value) in (type(None), bool, int, float, str):
        return True
    if type(value) in (tuple, list):
        return all(_plain_request_tree_supported(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _plain_request_tree_supported(item)
            for key, item in value.items()
        )
    return False


def _normalize_public_request_parameter_key(key):
    """Return one exact-string public key, or ``None`` when unsafe."""

    if type(key) is str:
        return key
    if type(key) is not _WaveformParameter:
        return None
    if not (
        _WaveformParameter.__mro__
        == (_WaveformParameter, str, object)
        and _WaveformParameter.__eq__ is str.__eq__
        and _WaveformParameter.__hash__ is str.__hash__
        and _WaveformParameter.__str__ is str.__str__
    ):
        return None
    normalized_key = str.__str__(key)
    if (
        type(normalized_key) is not str
        or str.__eq__(key, normalized_key) is not True
        or str.__hash__(key) != str.__hash__(normalized_key)
    ):
        return None
    return normalized_key


def _normalize_public_request_parameter_keys(params):
    """Copy trusted public ``Parameter`` keys to exact built-in strings.

    ``waveform.props`` starts from the public default dictionary, whose keys
    are exact :class:`Parameter` instances.  They inherit immutable string
    equality and hashing, but the strict plain-request contract deliberately
    rejects arbitrary scalar subclasses.  Admit only this one exact PyCBC
    class, verify that its relevant methods still have built-in semantics,
    and rebuild only the top-level keys without touching caller-owned state.
    Any conversion collision or foreign key fails closed to the unchanged
    execution path.
    """

    if type(params) is not dict:
        return None
    normalized = {}
    changed = False
    for key, value in params.items():
        normalized_key = _normalize_public_request_parameter_key(key)
        if normalized_key is None:
            return None
        changed = changed or type(key) is _WaveformParameter
        if normalized_key in normalized:
            return None
        normalized[normalized_key] = value
    if len(normalized) != len(params):
        return None
    return normalized if changed else params


def _runtime_boolean(function, *args):
    """Call a Torch runtime predicate, returning ``None`` on uncertainty."""

    if function is None:
        return None
    try:
        return bool(function(*args))
    except Exception:
        return None


def _plain_request_runtime_supported():
    """Reject transforms, tensor modes, AD, autocast, and CUDA capture."""

    if _runtime_boolean(getattr(torch.jit, "is_scripting", None)) is not False:
        return False
    if _runtime_boolean(getattr(torch.jit, "is_tracing", None)) is not False:
        return False
    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
        return False

    compile_checks = tuple(
        function
        for function in (
            getattr(getattr(torch, "compiler", None), "is_compiling", None),
            getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
        )
        if function is not None
    )
    if not compile_checks or any(
        _runtime_boolean(function) is not False for function in compile_checks
    ):
        return False

    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False
    functorch = getattr(getattr(torch, "_C", None), "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    if dynamic_depth is not None:
        try:
            if dynamic_depth() != 0:
                return False
        except Exception:
            return False
    elif _runtime_boolean(
        getattr(
            getattr(torch, "_C", None),
            "_are_functorch_transforms_active",
            None,
        )
    ) is not False:
        return False

    torch_c = getattr(torch, "_C", None)
    for name in ("_len_torch_dispatch_stack", "_len_torch_function_stack"):
        stack_length = getattr(torch_c, name, None)
        if stack_length is None:
            return False
        try:
            if stack_length() != 0:
                return False
        except Exception:
            return False

    autocast_enabled = getattr(torch, "is_autocast_enabled", None)
    if autocast_enabled is None:
        return False
    try:
        if autocast_enabled("cpu") or autocast_enabled("cuda"):
            return False
    except (RuntimeError, TypeError):
        legacy_cpu = getattr(torch, "is_autocast_cpu_enabled", None)
        if (
            _runtime_boolean(autocast_enabled) is not False
            or _runtime_boolean(legacy_cpu) is not False
        ):
            return False
    except Exception:
        return False

    state = _scheme.mgr.state
    if type(state) is not _scheme.TorchScheme:
        return False
    device = state.torch_device
    if type(device) is not torch.device or device.type not in ("cpu", "cuda"):
        return False
    if device.type == "cuda":
        capture = getattr(torch.cuda, "is_current_stream_capturing", None)
        if _runtime_boolean(capture) is not False:
            return False
    return True


def _plain_request_supported(params):
    """Return whether a public request can make the plain-input promise."""

    return (
        type(params) is dict
        and _plain_request_tree_supported(params)
        and _plain_request_runtime_supported()
    )


def _packed_remnant_plan_supported(params, inputs, frequencies):
    """Accept one ordinary binary64 XPHM request and fail closed otherwise."""

    return (
        _packed_remnant_plan_enabled()
        and _plain_request_supported(params)
        and inputs.prec_version in _MSA_PREC_VERSIONS
        and inputs.final_spin_mod in (3, 4)
        and inputs.device.type == "cpu"
        and inputs.real_dtype == torch.float64
        and inputs.complex_dtype == torch.complex128
        and type(frequencies) is torch.Tensor
        and frequencies.layout is torch.strided
        and frequencies.device == inputs.device
        and frequencies.dtype == inputs.real_dtype
        and frequencies.ndim == 1
        and frequencies.is_contiguous()
        and not frequencies.is_conj()
        and not frequencies.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(frequencies)
    )


def _reference_bulk_angle_lane_supported(params, inputs, frequencies, modes):
    """Accept only the raw-byte-qualified CPU-f64 five-row contract."""

    if not _plain_request_supported(params):
        return False
    if (
        type(modes) is not list
        or tuple(modes) != _COPRECESSING_MODES
        or inputs.prec_version not in _MSA_PREC_VERSIONS
        or type(inputs.device) is not torch.device
        or inputs.device.type != "cpu"
        or inputs.real_dtype != torch.float64
        or inputs.complex_dtype != torch.complex128
        or type(frequencies) is not torch.Tensor
        or frequencies.layout != torch.strided
        or frequencies.device != inputs.device
        or frequencies.dtype != torch.float64
        or frequencies.ndim != 1
        or frequencies.numel() == 0
        or not frequencies.is_contiguous()
        or frequencies.storage_offset() != 0
        or frequencies._base is not None
        or frequencies.is_conj()
        or frequencies.is_neg()
        or frequencies.requires_grad
        or frequencies.grad_fn is not None
    ):
        return False
    return True


def _configured_switch_enabled(name):
    """Read one prerequisite with the same strict switch parser."""

    value = os.environ.get(name)
    return False if value is None else _parse_switch(name, value)


def _manual_exact_coverage_supported(params):
    """Require exact manual derivatives before trusting internal tensors.

    The public plain-input check cannot rule out tensors created internally by
    an autograd derivative fallback.  Requiring every exact manual derivative
    route keeps the trusted shortcut fail closed on both CPU and CUDA.
    """

    if not all(
        _configured_switch_enabled(name)
        for name in _INFERENCE_REQUIRED_SWITCHES
    ):
        return False
    modes = _requested_coprecessing_modes(params)
    if modes is None:
        return False
    return (3, 2) not in modes or all(
        _configured_switch_enabled(name)
        for name in _INFERENCE_MODE32_REQUIRED_SWITCHES
    )


def _make_cuda_public_request_proof_primitives():
    """Create one sealed CUDA public-request proof boundary.

    This proof is deliberately independent of the XAS phase-plan proof, whose
    lifecycle is CPU-only.  Only the exact public regular-grid entry point can
    retain the closure-private runner used to create a proof.
    """

    namespace = globals()
    token = object()
    barrier = object()
    active_proof = ContextVar(
        "pycbc_imrphenomxphm_cuda_public_request_proof",
        default=None,
    )
    dispatch = None
    supported = None
    target = None
    fastpath = None

    class _Proof:
        __slots__ = (
            "token",
            "process_id",
            "thread_id",
            "device_index",
            "stream_id",
            "params",
            "active",
        )

        def __init__(self, device_index, stream_id, params):
            self.token = token
            self.process_id = os.getpid()
            self.thread_id = threading.get_ident()
            self.device_index = device_index
            self.stream_id = stream_id
            self.params = params
            self.active = True

        def __setattr__(self, name, value):
            if hasattr(self, name):
                if name == "active" and value is False:
                    object.__setattr__(self, name, False)
                    return
                raise AttributeError("CUDA request proof state is write-once")
            object.__setattr__(self, name, value)

    def bindings_current():
        return (
            dispatch is not None
            and supported is not None
            and target is not None
            and fastpath is not None
            and namespace.get("_dispatch_imrphenomxphm_request") is dispatch
            and namespace.get("imrphenomxphm_native_supported") is supported
            and namespace.get("_imrphenomxphm_fd_torch") is target
            and namespace.get("_cuda_aggregate_public_fastpath") is fastpath
        )

    def runtime_state():
        """Return the exact CUDA device/stream state, else ``None``."""

        try:
            if (
                not _plain_request_runtime_supported()
                or not torch.is_grad_enabled()
                or torch.is_inference_mode_enabled()
                or sys.gettrace() is not None
                or sys.getprofile() is not None
            ):
                return None
            compile_value = os.environ.get("PYCBC_TORCH_COMPILE")
            if compile_value is not None and _parse_switch(
                "PYCBC_TORCH_COMPILE",
                compile_value,
            ):
                return None
            state = _scheme.mgr.state
            if (
                type(state) is not _scheme.TorchScheme
                or type(state.torch_device) is not torch.device
                or state.torch_device.type != "cuda"
            ):
                return None
            device_index = (
                torch.cuda.current_device()
                if state.torch_device.index is None
                else state.torch_device.index
            )
            if torch.cuda.current_device() != device_index:
                return None
            capture = torch.cuda.is_current_stream_capturing()
            if capture is not False:
                return None
            stream_id = int(
                torch.cuda.current_stream(device_index).cuda_stream
            )
        except Exception:
            return None
        return device_index, stream_id

    def resolve(proof, params=None):
        state = runtime_state()
        candidate = active_proof.get() if proof is None else proof
        if (
            state is not None
            and type(candidate) is _Proof
            and candidate.token is token
            and candidate.active
            and candidate.process_id == os.getpid()
            and candidate.thread_id == threading.get_ident()
            and candidate.device_index == state[0]
            and candidate.stream_id == state[1]
            and (params is None or candidate.params is params)
            and active_proof.get() is candidate
            and bindings_current()
        ):
            return candidate
        return None

    def current(proof, params=None):
        """Recognize only this active synchronous CUDA public request."""

        return resolve(proof, params) is not None

    def scope_supported(params, generator):
        if (
            not bindings_current()
            or generator is not dispatch
            or type(params) is not dict
            or not _plain_request_supported(params)
            or not _manual_exact_coverage_supported(params)
            or runtime_state() is None
        ):
            return False
        n_batch = params.get("n_batch")
        return (
            (n_batch is None or (type(n_batch) is int and n_batch == 1))
            and supported(params) is True
        )

    def run(params, generator):
        """Run the bound private target under one non-borrowable proof."""

        frame = active_proof.set(barrier)
        proof = None
        try:
            if not scope_supported(params, generator):
                return generator(params)
            state = runtime_state()
            if state is None:
                return generator(params)
            proof = _Proof(state[0], state[1], params)
            proof_frame = active_proof.set(proof)
            try:
                try:
                    cached = fastpath(params, proof)
                except Exception:
                    cached = None
                if cached is not None:
                    return cached
                # Preserve the established XAS request-proof wrapper and its
                # exact arithmetic.  The bound private target resolves this
                # closure-private CUDA proof when that wrapper reaches it.
                return generator(params)
            finally:
                proof.active = False
                active_proof.reset(proof_frame)
        finally:
            active_proof.reset(frame)

    def bind(bound_dispatch, bound_supported, bound_target):
        """Bind the exact production dispatch, predicate, and target once."""

        nonlocal dispatch, supported, target
        if dispatch is not None or supported is not None or target is not None:
            raise RuntimeError("CUDA public-request proof is already bound")
        expected = (
            (
                bound_dispatch,
                "_dispatch_imrphenomxphm_request",
            ),
            (
                bound_supported,
                "imrphenomxphm_native_supported",
            ),
            (
                bound_target,
                "_imrphenomxphm_fd_torch",
            ),
        )
        if not all(
            type(function) is type(run)
            and function.__module__ == __name__
            and function.__name__ == name
            and function.__qualname__ == name
            and function.__globals__ is namespace
            for function, name in expected
        ):
            raise TypeError("invalid CUDA public-request proof target")
        dispatch = bound_dispatch
        supported = bound_supported
        target = bound_target

    def bind_fastpath(bound_fastpath):
        """Bind the sole exact early aggregate lookup once."""

        nonlocal fastpath
        if fastpath is not None:
            raise RuntimeError("CUDA public aggregate fast path is already bound")
        if not (
            type(bound_fastpath) is type(run)
            and bound_fastpath.__module__ == __name__
            and bound_fastpath.__name__ == "_cuda_aggregate_public_fastpath"
            and bound_fastpath.__qualname__
            == "_cuda_aggregate_public_fastpath"
            and bound_fastpath.__globals__ is namespace
        ):
            raise TypeError("invalid CUDA public aggregate fast path")
        fastpath = bound_fastpath

    def make_public_entry(public_result_cache_runner=None):
        """Close the private runner into the sole proof-issuing API route."""

        if not bindings_current():
            raise RuntimeError("CUDA public-request proof target is not bound")

        def imrphenomxphm_fd_torch(**params):
            """Generate a regular-grid IMRPhenomXPHM waveform with Torch."""

            return _run_scoped_xphm_request(
                params,
                _dispatch_imrphenomxphm_request,
                _cuda_public_proof_runner=run,
                _public_result_cache_runner=public_result_cache_runner,
            )

        imrphenomxphm_fd_torch.__qualname__ = "imrphenomxphm_fd_torch"
        return imrphenomxphm_fd_torch

    return bind, bind_fastpath, make_public_entry, run, current


(
    _bind_cuda_public_request_proof_target,
    _bind_cuda_public_request_fastpath,
    _make_cuda_public_request_entry,
    _cuda_public_request_proof_runner,
    _cuda_public_request_proof_current,
) = _make_cuda_public_request_proof_primitives()
del _make_cuda_public_request_proof_primitives


def _inference_exact_coverage_supported(params, *, manual_coverage=None):
    """Require exact manual CPU routes before entering inference mode.

    A guarded retry outside inference mode remains the final fail-closed path
    if an exact evaluator declines at run time for a particular system.
    """

    state = _scheme.mgr.state
    if manual_coverage is None:
        manual_coverage = _manual_exact_coverage_supported(params)
    return (
        type(state) is _scheme.TorchScheme
        and state.torch_device.type == "cpu"
        and manual_coverage
    )


def _inference_autograd_failure(error):
    """Recognize only failures caused by an exact evaluator declining."""

    if type(error) is _CoprecessingPlanCacheColdMiss:
        return True
    message = str(error)
    return (
        "does not require grad and does not have a grad_fn" in message
        or "Setting requires_grad=True on inference tensor" in message
        or "Inference tensors cannot be saved for backward" in message
    )


def _coprecessing_plan_cold_miss_hint(params):
    """Return a cheap, non-authoritative physical-request shape hint.

    Observer-only inclination and node rotations deliberately do not enter
    the co-precessing carrier plan.  Every authoritative eligibility, key,
    schema, and canary check still runs inside the cache: a coarse-hint
    collision can therefore cause only the established inference miss and
    ordinary retry, never an incorrect cache hit.
    """

    if type(params) is not dict:
        return None
    hint = tuple(
        params.get(name)
        for name in _COPRECESSING_PLAN_COLD_MISS_HINT_FIELDS
    )
    if not all(
        type(value) in (type(None), bool, int, float, str)
        for value in hint
    ):
        return None
    return hint


def _coprecessing_plan_cold_miss_hint_known(hint):
    """Return whether this process has completed the hinted request shape."""

    if hint is None:
        return True
    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        return hint in _COPRECESSING_PLAN_COLD_MISS_HINTS


def _note_coprecessing_plan_cold_miss_hint(hint):
    """Remember one successful ordinary shape in bounded process memory."""

    if hint is None:
        return
    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        _COPRECESSING_PLAN_COLD_MISS_HINTS[hint] = None
        _COPRECESSING_PLAN_COLD_MISS_HINTS.move_to_end(hint)
        while (
            len(_COPRECESSING_PLAN_COLD_MISS_HINTS)
            > _COPRECESSING_PLAN_COLD_MISS_HINT_MAX_ENTRIES
        ):
            _COPRECESSING_PLAN_COLD_MISS_HINTS.popitem(last=False)


def _run_xphm_request(params, generator, *, inference_mode):
    """Run one request, rebuilding normally after an inference AD decline."""

    def generate():
        with IMRPhenomX_utils.remnant_cache_context(
            enabled=_intrinsic_cache_enabled()
        ):
            return generator(params)

    if not inference_mode:
        return generate()
    cold_miss_one_pass = (
        _coprecessing_plan_cache_cold_miss_one_pass_enabled()
    )
    if cold_miss_one_pass:
        try:
            cold_miss_one_pass = _coprecessing_plan_cache_enabled()
        except Exception:
            cold_miss_one_pass = False
    cold_miss_hint = None
    if cold_miss_one_pass:
        cold_miss_hint = _coprecessing_plan_cold_miss_hint(params)
        cold_miss_one_pass = not _coprecessing_plan_cold_miss_hint_known(
            cold_miss_hint
        )
    if cold_miss_one_pass:
        # The first occurrence of this physical request shape is known cold.
        # Build it once in ordinary mode; later occurrences retain the faster
        # scoped-inference hit route.  The deep cache remains authoritative.
        token = _COPRECESSING_PLAN_CACHE_COLD_RETRY.set(True)
        try:
            result = generate()
        finally:
            _COPRECESSING_PLAN_CACHE_COLD_RETRY.reset(token)
        _note_coprecessing_plan_cold_miss_hint(cold_miss_hint)
        return result
    try:
        with torch.inference_mode():
            result = generate()
    except RuntimeError as error:
        if not _inference_autograd_failure(error):
            raise
        # The first request-local remnant cache has been released here.
        # Rebuild outside inference mode so no inference tensor can leak into
        # autograd.  A context-local marker lets an admitted cache miss retain
        # only the ordinary rebuild's private carrier/mode artifacts.
        token = _COPRECESSING_PLAN_CACHE_COLD_RETRY.set(True)
        try:
            return generate()
        finally:
            _COPRECESSING_PLAN_CACHE_COLD_RETRY.reset(token)

    # Inference tensors reject ordinary in-place user operations after leaving
    # their scope.  Copy the two final public arrays outside inference mode so
    # the opt-in execution detail does not change output ownership or
    # mutability.  The copy preserves every result byte.
    return tuple(series.copy() for series in result)


def _cuda_aggregate_public_request_requested():
    """Return whether the CUDA public proof path was explicitly requested."""

    try:
        return _cuda_aggregate_preterminal_twist_cache_enabled()
    except Exception:
        # Let the cache's fail-closed admission record an invalid opt-in value.
        return (
            os.environ.get(
                _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV
            )
            is not None
        )


def _cuda_coprecessing_plan_requested():
    """Return whether public CUDA-plan admission was explicitly requested.

    The CUDA plan cache remains its own fail-closed authority.  This helper
    only asks the public request boundary to validate and normalize PyCBC's
    exact ``Parameter`` keys before that independently gated admission runs.
    """

    try:
        return _cuda_coprecessing_plan_cache_enabled()
    except Exception:
        # Preserve ordinary arithmetic for an invalid explicit switch while
        # still routing it through the cache's strict, default-off admission.
        return os.environ.get(_CUDA_COPRECESSING_PLAN_CACHE_ENV) is not None


def _public_result_cache_requested():
    """Return whether the public-result cache was explicitly enabled.

    Explicit false values stay on the ordinary no-cache fast path.  Invalid
    values continue into the cache admission path so it can fail closed and
    record the rejected opt-in without changing waveform arithmetic.
    """

    value = os.environ.get(_PUBLIC_RESULT_CACHE_ENV)
    if value is None:
        return False
    try:
        return _parse_switch(_PUBLIC_RESULT_CACHE_ENV, value)
    except Exception:
        return True


def _run_scoped_xphm_request(
    params,
    generator,
    *,
    _cuda_public_proof_runner=None,
    _public_result_cache_runner=None,
):
    """Apply independently gated request-local execution promises."""

    trusted_requested = _trusted_plain_request_enabled()
    inference_requested = _inference_mode_enabled()
    cuda_plan_requested = _cuda_coprecessing_plan_requested()
    cuda_aggregate_requested = (
        _cuda_public_proof_runner is not None
        and _cuda_aggregate_public_request_requested()
    )
    public_result_requested = (
        _public_result_cache_runner is not None
        and _public_result_cache_requested()
    )
    if not (
        trusted_requested
        or inference_requested
        or cuda_plan_requested
        or cuda_aggregate_requested
        or public_result_requested
    ):
        return _run_xphm_request(params, generator, inference_mode=False)

    normalized_params = _normalize_public_request_parameter_keys(params)
    plain_supported = (
        normalized_params is not None
        and _plain_request_supported(normalized_params)
    )
    exact_coverage = (
        _manual_exact_coverage_supported(normalized_params)
        if plain_supported
        and (
            trusted_requested
            or inference_requested
            or cuda_plan_requested
            or cuda_aggregate_requested
            or public_result_requested
        )
        else False
    )
    trusted = trusted_requested and plain_supported and exact_coverage
    inference = (
        inference_requested
        and plain_supported
        and exact_coverage
        and _inference_exact_coverage_supported(
            params,
            manual_coverage=exact_coverage,
        )
    )

    def scoped_generator(scoped_params):
        if cuda_aggregate_requested and plain_supported and exact_coverage:
            return _cuda_public_proof_runner(scoped_params, generator)
        return generator(scoped_params)

    scoped_params = normalized_params if plain_supported else params

    def invoke():
        with IMRPhenomX_utils.trusted_plain_request_context(enabled=trusted):
            return _run_xphm_request(
                scoped_params,
                scoped_generator,
                inference_mode=inference,
            )

    if public_result_requested and plain_supported and exact_coverage:
        return _public_result_cache_runner(
            scoped_params,
            generator,
            invoke,
        )
    return invoke()


def _bulk_twist_harmonics(inputs):
    """Return exact request-local twist harmonics, or ``None`` to fall back."""

    complex_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }.get(inputs.real_dtype)
    if (
        complex_dtype is None
        or inputs.complex_dtype != complex_dtype
        or not isinstance(inputs.device, torch.device)
        or inputs.device.type not in ("cpu", "cuda")
    ):
        return None

    theta = inputs.theta_jn
    if isinstance(theta, torch.Tensor):
        if (
            type(theta) is not torch.Tensor
            or theta.layout is not torch.strided
            or theta.ndim != 0
            or theta.dtype != inputs.real_dtype
            or theta.device != inputs.device
            or theta.is_conj()
            or theta.is_neg()
            or IMRPhenomX_utils._tree_has_autograd(theta)
        ):
            return None
    elif type(theta) is not float or not math.isfinite(theta):
        return None

    if _cudagraph_twist_harmonics_enabled():
        evaluator = cudagraphed_spin_minus_two_spherical_harmonics_phi_zero
    elif _vectorized_twist_harmonics_enabled():
        evaluator = vectorized_spin_minus_two_spherical_harmonics_phi_zero
    elif (
        _scripted_twist_harmonics_enabled()
        and inputs.device.type == "cpu"
    ):
        evaluator = scripted_spin_minus_two_spherical_harmonics_phi_zero
    else:
        evaluator = spin_minus_two_spherical_harmonics_phi_zero
    return evaluator(
        theta,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )


def _request_bulk_twist_harmonics(inputs):
    """Prepare bulk harmonics only when both exact-reuse gates are enabled."""

    if not (_bulk_twist_harmonics_enabled() and _twist_reuse_enabled()):
        return None
    return _bulk_twist_harmonics(inputs)


def _requested_coprecessing_modes(params):
    """Return the deduplicated canonical co-precessing families to twist.

    ``None`` selects the full default set. Explicit arrays must contain only
    integral members of the positive-m families; entries are deduplicated and
    returned in canonical model order, matching LAL's idempotent mode
    activation. Invalid or unsupported requests return ``None`` so the caller
    falls back to lalsimulation.
    """

    mode_array = params.get("mode_array")
    if mode_array is None:
        return list(_COPRECESSING_MODES)
    try:
        requested = set()
        for mode in mode_array:
            ell, emm = mode
            if not isinstance(ell, Integral) or not isinstance(emm, Integral):
                return None
            family = (int(ell), int(emm))
            if family not in _POSITIVE_COPRECESSING_MODES:
                return None
            requested.add(family)
    except (TypeError, ValueError):
        return None
    return [mode for mode in _COPRECESSING_MODES if mode in requested]


def _xp_params(params):
    xp = dict(params)
    xp["approximant"] = "IMRPhenomXP"
    xp["mode_array"] = None
    return xp


def imrphenomxphm_native_supported(params):
    """Return whether ``params`` select the bounded native XPHM model."""

    approximant = params.get("approximant", "IMRPhenomXPHM")
    if approximant != "IMRPhenomXPHM":
        return False
    if _requested_coprecessing_modes(params) is None:
        return False
    prec_version = _integer_or_default(
        params.get("phenom_x_prec_version"),
        _DEFAULT_PREC_VERSION,
    )
    convention = _integer_or_default(
        params.get("phenom_xp_convention"),
        _DEFAULT_CONVENTION,
    )
    final_spin_mod = _integer_or_default(
        params.get("phenom_xp_final_spin_mod"),
        _DEFAULT_FINAL_SPIN,
    )
    return (
        prec_version in _MSA_PREC_VERSIONS
        and convention == _MSA_CONVENTION
        and final_spin_mod in _MSA_FINAL_SPIN_MODES
        and imrphenomxp_native_supported(_xp_params(params))
    )


def imrphenomxphm_sequence_native_supported(params):
    """Return whether arbitrary-frequency XPHM generation is native."""

    return imrphenomxphm_native_supported(params)


def _coprecessing_params(params, inputs):
    aligned = dict(params)
    aligned.update(
        approximant="IMRPhenomXHM",
        mass1=inputs.mass1,
        mass2=inputs.mass2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=inputs.chi1_l,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=inputs.chi2_l,
        inclination=0.0,
        coa_phase=inputs.carrier_phase,
        long_asc_nodes=0.0,
        mode_array=None,
    )
    return aligned


def _coprecessing_final_spin(model):
    if model.packed_remnant_plan is not None:
        fs = model.packed_remnant_plan.carrier.final_spin
        if isinstance(fs, torch.Tensor):
            return float(fs.item()) if fs.numel() == 1 else fs
        return float(fs)
    if model.final_spin is not None:
        return model.final_spin
    inputs = model.inputs
    fs = IMRPhenomX_utils.get_remnant_fMs(
        inputs.mass1,
        inputs.mass2,
        inputs.chi1_l,
        inputs.chi2_l,
        chip=inputs.chip,
    ).final_spin
    if isinstance(fs, torch.Tensor):
        return float(fs.item()) if fs.numel() == 1 else fs
    return float(fs)


def _coprecessing_plan_tensor_bytes(value):
    """Copy one CPU tensor's exact storage-order bytes into Python memory."""

    return value.detach().reshape(-1).view(torch.uint8).numpy().tobytes()


def _coprecessing_plan_deep_size(value, seen=None):
    """Conservatively charge a retained tensor-free Python object graph."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _coprecessing_plan_deep_size(key, seen)
            + _coprecessing_plan_deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(
            _coprecessing_plan_deep_size(item, seen) for item in value
        )
    return size


def _freeze_coprecessing_plan_value(value):
    """Freeze trusted scalar/derived state without retaining any tensor."""

    value_type = type(value)
    if value is None:
        return ("none",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float64", struct.pack("!d", value))
    if value_type is str:
        return ("str", value)
    if isinstance(value, tuple) and hasattr(value_type, "_fields"):
        return (
            "namedtuple",
            value_type.__module__,
            value_type.__qualname__,
            tuple(_freeze_coprecessing_plan_value(item) for item in value),
        )
    if value_type is tuple:
        return (
            "tuple",
            tuple(_freeze_coprecessing_plan_value(item) for item in value),
        )
    if value_type is list:
        return (
            "list",
            tuple(_freeze_coprecessing_plan_value(item) for item in value),
        )
    if value_type is dict:
        return (
            "dict",
            tuple(
                (
                    key,
                    _freeze_coprecessing_plan_value(value[key]),
                )
                for key in sorted(value)
            ),
        )
    if value_type is torch.Tensor:
        if value.layout is not torch.strided or value.device.type != "cpu":
            raise TypeError("only plain CPU derived tensors may enter the key")
        return (
            "tensor",
            value.dtype,
            value.device,
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            _coprecessing_plan_tensor_bytes(value),
        )
    if value_type is torch.device:
        return ("device", value.type, value.index)
    if isinstance(value, torch.dtype):
        return ("dtype", value)
    raise TypeError(f"unsupported co-precessing cache key type {value_type!r}")


def _phenomx_torch_environment_items():
    """Snapshot every live PhenomX/Torch environment item once."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith(("PYCBC_IMRPHENOMX", "PYCBC_TORCH_"))
        )
    )


def _coprecessing_plan_environment_identity(environment_items=None):
    """Filter one PhenomX/Torch snapshot for the co-precessing plan."""

    ignored = {
        _COPRECESSING_PLAN_CACHE_BYTES_ENV,
        _PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT_ENV,
    }
    if environment_items is None:
        environment_items = _phenomx_torch_environment_items()
    return tuple(
        (name, value)
        for name, value in environment_items
        if name not in ignored
    )


def _coprecessing_plan_coefficient_tables():
    """Return the live canonical CPU coefficient-table bindings."""

    phase_master = getattr(
        IMRPhenomX_utils,
        "_PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER",
        None,
    )
    amp_master = getattr(
        IMRPhenomX_utils,
        "_PHENOMX_AMP_COEFF_TABLE_CPU_MASTER",
        None,
    )
    phase_current = (
        IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    )
    amp_current = (
        IMRPhenomX_utils._get_phenomx_amp_coeff_table_cached_master(
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    )
    return phase_master, amp_master, phase_current, amp_current


def _coprecessing_plan_coefficient_table_supported(
    value,
    source,
    current,
    shape,
):
    """Accept one unchanged, canonical, AD-free CPU coefficient table."""

    try:
        storage = value.untyped_storage()
        return (
            type(value) is torch.Tensor
            and value is source
            and value is current
            and value.layout is torch.strided
            and value.device == torch.device("cpu")
            and value.dtype is torch.float64
            and tuple(value.shape) == shape
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value._base is None
            and value._version == 0
            and not value.is_conj()
            and not value.is_neg()
            and not value.is_inference()
            and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
            and storage.data_ptr() != 0
            and storage.nbytes() == value.numel() * value.element_size()
        )
    except Exception:
        return False


def _coprecessing_plan_coefficient_tables_supported():
    """Fail closed if either private production table has drifted."""

    try:
        phase_master, amp_master, phase_current, amp_current = (
            _coprecessing_plan_coefficient_tables()
        )
        return (
            _coprecessing_plan_coefficient_table_supported(
                phase_master,
                _COPRECESSING_PLAN_PHASE_TABLE_SOURCE,
                phase_current,
                (13, 49),
            )
            and _coprecessing_plan_coefficient_table_supported(
                amp_master,
                _COPRECESSING_PLAN_AMP_TABLE_SOURCE,
                amp_current,
                (7, 42),
            )
        )
    except Exception:
        return False


def _coprecessing_plan_implementation_identity(environment_items=None):
    """Identify the bounded implementation and numerical runtime.

    Direct producer roots and coefficient tables are guarded cheaply on every
    request.  The release token covers their unmodified transitive code; live
    debugging below those roots requires disabling or clearing the cache.
    """

    def table_identity(value):
        if type(value) is not torch.Tensor:
            return id(value), None
        try:
            return (
                id(value),
                value.data_ptr(),
                value._version,
                value.device.type,
                value.device.index,
                value.dtype,
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                id(value._base) if value._base is not None else None,
                bool(value.requires_grad),
                id(value.grad_fn) if value.grad_fn is not None else None,
                bool(value.is_inference()),
                bool(IMRPhenomX_utils._tensor_has_forward_ad(value)),
            )
        except Exception:
            return id(value), None

    try:
        phase_table, amp_table, phase_current, amp_current = (
            _coprecessing_plan_coefficient_tables()
        )
    except Exception:
        phase_table = None
        amp_table = None
        phase_current = None
        amp_current = None

    direct_roots = (
        _build_model,
        _build_coprecessing_plan,
        _build_reference_mode_msa_angle_core,
        _bulk_mode_angles,
        _reference_and_mode_msa_angles,
        _xas_samples,
        _active_mode_samples,
        _carrier_amp_plan_reuse_enabled,
        _carrier_alignment_result_reuse_supported,
        _coprecessing_params,
        _coprecessing_final_spin,
        _phenomx_torch_environment_items,
        _coprecessing_plan_environment_identity,
    )
    phase_getter = getattr(
        IMRPhenomX_utils,
        "_get_phenomx_phase_coeff_table_cached_master",
        None,
    )
    amp_getter = getattr(
        IMRPhenomX_utils,
        "_get_phenomx_amp_coeff_table_cached_master",
        None,
    )

    return (
        _COPRECESSING_PLAN_CACHE_IMPLEMENTATION,
        torch.__version__,
        getattr(torch.version, "git_version", None),
        torch.get_default_dtype(),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        bool(torch.are_deterministic_algorithms_enabled()),
        torch.get_float32_matmul_precision(),
        tuple(
            (id(value), id(getattr(value, "__code__", None)))
            for value in direct_roots
        ),
        tuple(
            (
                id(value),
                id(getattr(value, "__new__", None)),
                id(getattr(getattr(value, "__new__", None), "__code__", None)),
            )
            for value in (
                _SequenceCore,
                _ReferenceModeMSAAngleCore,
                _CoprecessingPlan,
            )
        ),
        _freeze_coprecessing_plan_value(_XAS_MODE_POLARIZATION_FACTOR),
        (id(phase_getter), id(getattr(phase_getter, "__code__", None))),
        (id(amp_getter), id(getattr(amp_getter, "__code__", None))),
        table_identity(_COPRECESSING_PLAN_PHASE_TABLE_SOURCE),
        table_identity(_COPRECESSING_PLAN_AMP_TABLE_SOURCE),
        table_identity(phase_table),
        table_identity(amp_table),
        table_identity(phase_current),
        table_identity(amp_current),
        _coprecessing_plan_environment_identity(environment_items),
    )


def _coprecessing_plan_request_supported(
    model,
    frequencies,
    params,
    modes,
    active_f_max,
    uniform_grid_metadata,
):
    """Admit only the sealed CPU-f64, n_batch=1 inference contract."""

    inference_predicate = getattr(torch, "is_inference", None)
    trusted_context = getattr(
        IMRPhenomX_utils,
        "_TRUSTED_PLAIN_REQUEST",
        None,
    )
    inference_active = (
        torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
    )
    cold_retry = (
        _COPRECESSING_PLAN_CACHE_COLD_RETRY.get() is True
        and not torch.is_inference_mode_enabled()
        and torch.is_grad_enabled()
    )
    if (
        not callable(inference_predicate)
        or trusted_context is None
        or not (inference_active or cold_retry)
        or trusted_context.get() is not True
        or not _plain_request_supported(params)
        or type(uniform_grid_metadata) is not tuple
        or len(uniform_grid_metadata) != 3
        or type(active_f_max) is not float
        or not math.isfinite(active_f_max)
    ):
        return False

    first_bin, stop_bin, delta_f = uniform_grid_metadata
    n_batch = params.get("n_batch")
    inputs = model.inputs
    state = _scheme.mgr.state
    if (
        (n_batch is not None and (type(n_batch) is not int or n_batch != 1))
        or type(first_bin) is not int
        or type(stop_bin) is not int
        or type(delta_f) is not float
        or first_bin < 0
        or stop_bin <= first_bin
        or not math.isfinite(delta_f)
        or delta_f <= 0.0
        or type(state) is not _scheme.TorchScheme
        or state.torch_device != torch.device("cpu")
        or type(inputs.device) is not torch.device
        or inputs.device != torch.device("cpu")
        or inputs.real_dtype is not torch.float64
        or inputs.complex_dtype is not torch.complex128
        or type(frequencies) is not torch.Tensor
        or frequencies.layout is not torch.strided
        or frequencies.device != inputs.device
        or frequencies.dtype is not torch.float64
        or frequencies.ndim != 1
        or frequencies.numel() != stop_bin - first_bin
        or frequencies.numel() == 0
        or frequencies.stride() != (1,)
        or not frequencies.is_contiguous()
        or frequencies.storage_offset() != 0
        or frequencies._base is not None
        or frequencies.is_conj()
        or frequencies.is_neg()
        or frequencies.requires_grad
        or frequencies.grad_fn is not None
        or inference_predicate(frequencies) is not inference_active
        or IMRPhenomX_utils._tensor_has_forward_ad(frequencies)
        or type(modes) is not list
        or not modes
    ):
        return False

    try:
        canonical_modes = tuple(
            mode for mode in _COPRECESSING_MODES if mode in modes
        )
        storage_nbytes = frequencies.untyped_storage().nbytes()
        storage_pointer = frequencies.untyped_storage().data_ptr()
        first_frequency = float(frequencies[0].item())
        last_frequency = float(frequencies[-1].item())
    except Exception:
        return False
    return (
        tuple(modes) == canonical_modes
        and all(
            type(mode) is tuple
            and len(mode) == 2
            and all(type(value) is int for value in mode)
            for mode in modes
        )
        and storage_pointer != 0
        and storage_nbytes == frequencies.numel() * frequencies.element_size()
        and first_frequency == first_bin * delta_f
        and last_frequency == (stop_bin - 1) * delta_f
        and last_frequency <= active_f_max
    )


def _coprecessing_plan_key(
    model,
    frequencies,
    params,
    modes,
    active_f_max,
    uniform_grid_metadata,
    request_proof,
    *,
    angle_core_enabled,
):
    """Build a complete, tensor-free exact lookup key."""

    inputs = model.inputs
    aligned_params = _coprecessing_params(params, inputs)
    # ``msa_state`` is a deterministic output of the guarded ``_build_model``
    # root from the exact intrinsic input fields below.  Re-freezing its 121
    # derived scalars on every hit would duplicate that key at material cost.
    angle_core_identity = (
        angle_core_enabled,
        model.msa_reference_angles_deferred is True,
    )
    intrinsic_fields = (
        "tidal_version",
        "mass1",
        "mass2",
        "chi1_l",
        "chi2_l",
        "chip",
        "spin_aligned",
        "spin_perp",
        "spin1",
        "spin2",
        "lambda1",
        "lambda2",
        "dquad1",
        "dquad2",
        "prec_version",
        "final_spin_mod",
        "distance",
        "carrier_phase",
        "f_ref",
        "total_mass",
        "total_mass_seconds",
        "eta",
        "device",
        "real_dtype",
        "complex_dtype",
    )
    frequency_bytes = _coprecessing_plan_tensor_bytes(frequencies)
    key = (
        _coprecessing_plan_implementation_identity(),
        tuple(
            (name, _freeze_coprecessing_plan_value(getattr(inputs, name)))
            for name in intrinsic_fields
        ),
        _freeze_coprecessing_plan_value(aligned_params),
        _freeze_coprecessing_plan_value(model.final_spin),
        _freeze_coprecessing_plan_value(model.packed_remnant_plan),
        angle_core_identity,
        tuple(modes),
        _freeze_coprecessing_plan_value(active_f_max),
        _freeze_coprecessing_plan_value(uniform_grid_metadata),
        bool(_request_proof_plan_current(request_proof)),
        (
            frequencies.dtype,
            frequencies.device,
            tuple(frequencies.shape),
            tuple(frequencies.stride()),
            frequency_bytes,
        ),
    )
    return key


def _reference_mode_msa_angle_core_tensors(core):
    """Return the three raw tensor rows retained by one optional core."""

    return (core.mode_phiz, core.mode_zeta, core.mode_cos_beta)


def _coprecessing_plan_tensors(plan):
    tensors = (plan.carrier,) + tuple(
        samples for _, samples in plan.active_modes
    )
    if type(plan.reference_angle_core) is _ReferenceModeMSAAngleCore:
        tensors += _reference_mode_msa_angle_core_tensors(plan.reference_angle_core)
    return tensors


def _coprecessing_plan_schema_supported(
    plan,
    frequencies,
    modes,
    *,
    privately_owned,
):
    """Validate the fixed artifact schema and, for storage, ownership."""

    inference_predicate = getattr(torch, "is_inference", None)
    if (
        type(plan) is not _CoprecessingPlan
        or type(plan.active_modes) is not tuple
        or len(plan.active_modes) != len(modes)
        or plan.reference_angle_core is not None
        and type(plan.reference_angle_core) is not _ReferenceModeMSAAngleCore
        or not callable(inference_predicate)
    ):
        return False
    for expected_mode, item in zip(modes, plan.active_modes):
        if type(item) is not tuple or len(item) != 2 or item[0] != expected_mode:
            return False

    expected_mprimes = tuple(dict.fromkeys(mprime for _, mprime in modes))
    core = plan.reference_angle_core
    tensor_specs = [
        (plan.carrier, torch.complex128, tuple(frequencies.shape)),
        *(
            (samples, torch.complex128, tuple(frequencies.shape))
            for _, samples in plan.active_modes
        ),
    ]
    if core is not None:
        if (
            type(core.mprimes) is not tuple
            or core.mprimes != expected_mprimes
            or not all(type(mprime) is int for mprime in core.mprimes)
            or type(core.reference_phiz_residual) is not float
            or type(core.reference_zeta_residual) is not float
            or not math.isfinite(core.reference_phiz_residual)
            or not math.isfinite(core.reference_zeta_residual)
        ):
            return False
        row_shape = (len(expected_mprimes), frequencies.numel())
        tensor_specs.extend(
            (value, torch.float64, row_shape)
            for value in _reference_mode_msa_angle_core_tensors(core)
        )

    try:
        frequency_pointer = frequencies.untyped_storage().data_ptr()
        storage_pointers = []
        for value, expected_dtype, expected_shape in tensor_specs:
            if (
                type(value) is not torch.Tensor
                or value.layout is not torch.strided
                or value.device != torch.device("cpu")
                or value.dtype is not expected_dtype
                or tuple(value.shape) != expected_shape
                or not value.is_contiguous()
                or value.is_conj()
                or value.is_neg()
                or value.requires_grad
                or value.grad_fn is not None
                or IMRPhenomX_utils._tensor_has_forward_ad(value)
            ):
                return False
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            if pointer == 0 or pointer == frequency_pointer:
                return False
            if privately_owned and (
                inference_predicate(value) is not False
                or value._version != 0
                or value.storage_offset() != 0
                or value._base is not None
                or storage.nbytes() != value.numel() * value.element_size()
            ):
                return False
            storage_pointers.append(pointer)
    except Exception:
        return False
    return not privately_owned or len(storage_pointers) == len(
        set(storage_pointers)
    )


def _copy_coprecessing_plan_for_cache(plan):
    """Create ordinary, unaliased tensors owned solely by the cache."""

    def owned(value):
        with torch.inference_mode(False), torch.no_grad():
            return value.detach().clone(memory_format=torch.contiguous_format)

    return _CoprecessingPlan(
        owned(plan.carrier),
        tuple((mode, owned(samples)) for mode, samples in plan.active_modes),
        (
            None
            if plan.reference_angle_core is None
            else _ReferenceModeMSAAngleCore(
                tuple(plan.reference_angle_core.mprimes),
                *(
                    owned(value)
                    for value in _reference_mode_msa_angle_core_tensors(
                        plan.reference_angle_core
                    )
                ),
                plan.reference_angle_core.reference_phiz_residual,
                plan.reference_angle_core.reference_zeta_residual,
            )
        ),
    )


def _coprecessing_plan_fingerprints(plan):
    core = plan.reference_angle_core
    core_metadata = (
        None
        if core is None
        else (
            tuple(core.mprimes),
            struct.pack("!d", core.reference_phiz_residual),
            struct.pack("!d", core.reference_zeta_residual),
        )
    )
    return (
        core_metadata,
        tuple(
            hashlib.sha256(_coprecessing_plan_tensor_bytes(value)).digest()
            for value in _coprecessing_plan_tensors(plan)
        ),
    )


def _coprecessing_plan_raw_clone_canary(source, candidate):
    """Require exact bytes and metadata across the private ownership copy."""

    source_core = source.reference_angle_core
    candidate_core = candidate.reference_angle_core
    if (source_core is None) != (candidate_core is None):
        return False
    if source_core is not None and (
        source_core.mprimes != candidate_core.mprimes
        or struct.pack("!d", source_core.reference_phiz_residual)
        != struct.pack("!d", candidate_core.reference_phiz_residual)
        or struct.pack("!d", source_core.reference_zeta_residual)
        != struct.pack("!d", candidate_core.reference_zeta_residual)
    ):
        return False
    source_tensors = _coprecessing_plan_tensors(source)
    candidate_tensors = _coprecessing_plan_tensors(candidate)
    if len(source_tensors) != len(candidate_tensors):
        return False
    try:
        return all(
            source_value.dtype == candidate_value.dtype
            and source_value.device == candidate_value.device
            and source_value.shape == candidate_value.shape
            and _coprecessing_plan_tensor_bytes(source_value)
            == _coprecessing_plan_tensor_bytes(candidate_value)
            for source_value, candidate_value in zip(
                source_tensors,
                candidate_tensors,
            )
        )
    except Exception:
        return False


def _coprecessing_plan_entry_bytes(plan, fingerprints, key_bytes):
    """Charge every retained object plus conservative container overhead."""

    storage_bytes = 0
    seen_storages = set()
    for value in _coprecessing_plan_tensors(plan):
        storage = value.untyped_storage()
        storage_identity = storage._cdata
        if storage_identity in seen_storages:
            continue
        seen_storages.add(storage_identity)
        storage_bytes += storage.nbytes()
    return (
        key_bytes
        + _coprecessing_plan_deep_size(plan)
        + _coprecessing_plan_deep_size(fingerprints)
        + storage_bytes
        + _COPRECESSING_PLAN_CACHE_ENTRY_OVERHEAD
    )


def _coprecessing_plan_cache_reset_if_forked_locked():
    global _COPRECESSING_PLAN_CACHE_PID
    global _COPRECESSING_PLAN_CACHE_SIZE
    global _COPRECESSING_PLAN_CACHE_GENERATION

    process_id = os.getpid()
    if process_id == _COPRECESSING_PLAN_CACHE_PID:
        return
    _COPRECESSING_PLAN_CACHE.clear()
    _COPRECESSING_PLAN_COLD_MISS_HINTS.clear()
    _COPRECESSING_PLAN_CACHE_SIZE = 0
    _COPRECESSING_PLAN_CACHE_GENERATION = 0
    _COPRECESSING_PLAN_CACHE_COUNTERS.clear()
    _COPRECESSING_PLAN_CACHE_COUNTERS.update(
        _new_coprecessing_plan_cache_counters()
    )
    _COPRECESSING_PLAN_CACHE_PID = process_id


def _coprecessing_plan_cache_after_fork():
    """Drop inherited tensors and replace a possibly inherited locked lock."""

    global _COPRECESSING_PLAN_CACHE
    global _COPRECESSING_PLAN_CACHE_LOCK
    global _COPRECESSING_PLAN_CACHE_PID
    global _COPRECESSING_PLAN_CACHE_SIZE
    global _COPRECESSING_PLAN_CACHE_GENERATION
    global _COPRECESSING_PLAN_CACHE_COUNTERS
    global _COPRECESSING_PLAN_COLD_MISS_HINTS
    global _COPRECESSING_PLAN_CACHE_COLD_RETRY

    _COPRECESSING_PLAN_CACHE = OrderedDict()
    _COPRECESSING_PLAN_CACHE_LOCK = threading.RLock()
    _COPRECESSING_PLAN_CACHE_PID = os.getpid()
    _COPRECESSING_PLAN_CACHE_SIZE = 0
    _COPRECESSING_PLAN_CACHE_GENERATION = 0
    _COPRECESSING_PLAN_CACHE_COUNTERS = (
        _new_coprecessing_plan_cache_counters()
    )
    _COPRECESSING_PLAN_COLD_MISS_HINTS = OrderedDict()
    _COPRECESSING_PLAN_CACHE_COLD_RETRY = ContextVar(
        "pycbc_imrphenomxphm_coprecessing_plan_cache_cold_retry",
        default=False,
    )


def _clear_coprecessing_plan_cache():
    """Clear every private plan and reset diagnostic counters."""

    global _COPRECESSING_PLAN_CACHE_SIZE

    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        _COPRECESSING_PLAN_CACHE.clear()
        _COPRECESSING_PLAN_COLD_MISS_HINTS.clear()
        _COPRECESSING_PLAN_CACHE_SIZE = 0
        _COPRECESSING_PLAN_CACHE_COUNTERS.clear()
        _COPRECESSING_PLAN_CACHE_COUNTERS.update(
            _new_coprecessing_plan_cache_counters()
        )


def _invalidate_all_coprecessing_plan_cache_entries():
    """Discard retained plans while preserving diagnostic history."""

    global _COPRECESSING_PLAN_CACHE_SIZE

    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        invalidated = len(_COPRECESSING_PLAN_CACHE)
        _COPRECESSING_PLAN_CACHE.clear()
        _COPRECESSING_PLAN_COLD_MISS_HINTS.clear()
        _COPRECESSING_PLAN_CACHE_SIZE = 0
        _COPRECESSING_PLAN_CACHE_COUNTERS["invalidations"] += invalidated


def _coprecessing_plan_cache_stats():
    """Return a request-owned snapshot for tests and profiling."""

    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        result = dict(_COPRECESSING_PLAN_CACHE_COUNTERS)
        result.update(
            pid=_COPRECESSING_PLAN_CACHE_PID,
            entries=len(_COPRECESSING_PLAN_CACHE),
            bytes=_COPRECESSING_PLAN_CACHE_SIZE,
            budget_bytes=_coprecessing_plan_cache_budget() or 0,
            max_entries=_COPRECESSING_PLAN_CACHE_MAX_ENTRIES,
        )
        return result


def _coprecessing_plan_cache_note_ineligible():
    try:
        with _COPRECESSING_PLAN_CACHE_LOCK:
            _coprecessing_plan_cache_reset_if_forked_locked()
            _COPRECESSING_PLAN_CACHE_COUNTERS["ineligible"] += 1
    except Exception:
        pass


def _coprecessing_plan_cache_note_race():
    try:
        with _COPRECESSING_PLAN_CACHE_LOCK:
            _coprecessing_plan_cache_reset_if_forked_locked()
            _COPRECESSING_PLAN_CACHE_COUNTERS["races"] += 1
    except Exception:
        pass


def _trim_coprecessing_plan_cache_locked(budget):
    global _COPRECESSING_PLAN_CACHE_SIZE

    while (
        _COPRECESSING_PLAN_CACHE
        and (
            _COPRECESSING_PLAN_CACHE_SIZE > budget
            or len(_COPRECESSING_PLAN_CACHE)
            > _COPRECESSING_PLAN_CACHE_MAX_ENTRIES
        )
    ):
        _, entry = _COPRECESSING_PLAN_CACHE.popitem(last=False)
        _COPRECESSING_PLAN_CACHE_SIZE -= entry.nbytes
        _COPRECESSING_PLAN_CACHE_COUNTERS["evictions"] += 1


def _lookup_coprecessing_plan_cache(
    key,
    frequencies,
    modes,
    budget,
    *,
    token_sink=None,
):
    """Return one structurally intact immutable plan, updating its recency."""

    global _COPRECESSING_PLAN_CACHE_SIZE

    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        _trim_coprecessing_plan_cache_locked(budget)
        entry = _COPRECESSING_PLAN_CACHE.get(key)
        if entry is None:
            _COPRECESSING_PLAN_CACHE_COUNTERS["misses"] += 1
            return None
        valid = _coprecessing_plan_schema_supported(
            entry.plan,
            frequencies,
            modes,
            privately_owned=True,
        )
        if valid:
            valid = (
                _coprecessing_plan_fingerprints(entry.plan)
                == entry.fingerprints
            )
        if not valid:
            _COPRECESSING_PLAN_CACHE.pop(key, None)
            _COPRECESSING_PLAN_CACHE_SIZE -= entry.nbytes
            _COPRECESSING_PLAN_CACHE_COUNTERS["invalidations"] += 1
            _COPRECESSING_PLAN_CACHE_COUNTERS["misses"] += 1
            return None
        _COPRECESSING_PLAN_CACHE.move_to_end(key)
        _COPRECESSING_PLAN_CACHE_COUNTERS["hits"] += 1
        if type(token_sink) is list and not token_sink:
            token_sink.append(
                _CoprecessingPlanCacheToken(
                    key,
                    entry.generation,
                    _COPRECESSING_PLAN_CACHE_PID,
                )
            )
        return entry.plan


def _store_coprecessing_plan_cache(
    key,
    key_bytes,
    plan,
    frequencies,
    modes,
    budget,
    revalidate_key,
):
    """Clone, canary, and insert one private artifact plan if it fits."""

    global _COPRECESSING_PLAN_CACHE_SIZE
    global _COPRECESSING_PLAN_CACHE_GENERATION

    if not _coprecessing_plan_schema_supported(
        plan,
        frequencies,
        modes,
        privately_owned=False,
    ):
        return
    try:
        candidate = _copy_coprecessing_plan_for_cache(plan)
        if not _coprecessing_plan_schema_supported(
            candidate,
            frequencies,
            modes,
            privately_owned=True,
        ) or not _coprecessing_plan_raw_clone_canary(plan, candidate):
            with _COPRECESSING_PLAN_CACHE_LOCK:
                _coprecessing_plan_cache_reset_if_forked_locked()
                _COPRECESSING_PLAN_CACHE_COUNTERS["canary_failures"] += 1
            return
        fingerprints = _coprecessing_plan_fingerprints(candidate)
        entry_nbytes = _coprecessing_plan_entry_bytes(
            candidate,
            fingerprints,
            key_bytes,
        )
    except Exception:
        with _COPRECESSING_PLAN_CACHE_LOCK:
            _coprecessing_plan_cache_reset_if_forked_locked()
            _COPRECESSING_PLAN_CACHE_COUNTERS["canary_failures"] += 1
        return

    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        try:
            current_key = revalidate_key()
            current_key_bytes = _coprecessing_plan_deep_size(current_key)
        except Exception:
            _COPRECESSING_PLAN_CACHE_COUNTERS["races"] += 1
            return
        if current_key != key or current_key_bytes != key_bytes:
            _COPRECESSING_PLAN_CACHE_COUNTERS["races"] += 1
            return
        if entry_nbytes > budget:
            _COPRECESSING_PLAN_CACHE_COUNTERS["oversize"] += 1
            return
        if key in _COPRECESSING_PLAN_CACHE:
            _COPRECESSING_PLAN_CACHE_COUNTERS["races"] += 1
            return
        _COPRECESSING_PLAN_CACHE_GENERATION += 1
        entry = _CoprecessingPlanCacheEntry(
            candidate,
            fingerprints,
            entry_nbytes,
            _COPRECESSING_PLAN_CACHE_GENERATION,
        )
        _COPRECESSING_PLAN_CACHE[key] = entry
        _COPRECESSING_PLAN_CACHE_SIZE += entry_nbytes
        _COPRECESSING_PLAN_CACHE_COUNTERS["stores"] += 1
        _trim_coprecessing_plan_cache_locked(budget)


def _coprecessing_plan_cache_token_current(token, frequencies, modes):
    """Revalidate a generation token without retaining the plan lock."""

    global _COPRECESSING_PLAN_CACHE_SIZE

    if type(token) is not _CoprecessingPlanCacheToken:
        return False
    with _COPRECESSING_PLAN_CACHE_LOCK:
        _coprecessing_plan_cache_reset_if_forked_locked()
        if token.pid != _COPRECESSING_PLAN_CACHE_PID:
            return False
        entry = _COPRECESSING_PLAN_CACHE.get(token.key)
        if entry is None or entry.generation != token.generation:
            return False
        valid = _coprecessing_plan_schema_supported(
            entry.plan,
            frequencies,
            modes,
            privately_owned=True,
        )
        if valid:
            valid = (
                _coprecessing_plan_fingerprints(entry.plan)
                == entry.fingerprints
            )
        if valid:
            return True
        _COPRECESSING_PLAN_CACHE.pop(token.key, None)
        _COPRECESSING_PLAN_CACHE_SIZE -= entry.nbytes
        _COPRECESSING_PLAN_CACHE_COUNTERS["invalidations"] += 1
        return False


def _aggregate_preterminal_twist_implementation_identity():
    """Retain exact producer identities for aggregate-cache isolation."""

    direct_roots = (
        _bulk_mode_angles,
        _mode_angles,
        _reference_and_mode_msa_angles,
        msa_angles,
        _twist_mode,
        _twist_reuse_supported,
        _bulk_twist_harmonics,
        _packed_twist_harmonics,
        _stacked_twist_request_device,
        _stacked_twist_mode,
        _wigner_columns,
        _ordered_stacked_twist_sum,
        _bulk_twist_exponentials,
        _packed_twist_exponentials,
        _twist_exponential_recurrence,
        spin_weighted_spherical_harmonic,
        spin_minus_two_spherical_harmonics_phi_zero,
        scripted_spin_minus_two_spherical_harmonics_phi_zero,
        vectorized_spin_minus_two_spherical_harmonics_phi_zero,
    )
    return (
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_IMPLEMENTATION,
        torch.__version__,
        getattr(torch.version, "git_version", None),
        tuple(
            (function, getattr(function, "__code__", None))
            for function in direct_roots
        ),
    )


def _aggregate_preterminal_twist_scalar_tensor_key(value):
    """Freeze one plain CPU complex128 scalar harmonic exactly."""

    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.device != torch.device("cpu")
        or value.dtype is not torch.complex128
        or value.ndim != 0
        or not value.is_contiguous()
        or value.is_conj()
        or value.is_neg()
        or value.requires_grad
        or value.grad_fn is not None
        or IMRPhenomX_utils._tensor_has_forward_ad(value)
    ):
        return None
    try:
        base = value._base
        if base is not None and type(base) is not torch.Tensor:
            return None
        storage = value.untyped_storage()
        storage_pointer = storage.data_ptr()
        tensor_pointer = value.data_ptr()
        element_size = value.element_size()
        storage_offset_bytes = value.storage_offset() * element_size
        storage_nbytes = storage.nbytes()
        if (
            storage_pointer == 0
            or tensor_pointer == 0
            or storage_offset_bytes < 0
            or storage_offset_bytes > storage_nbytes - element_size
            or tensor_pointer != storage_pointer + storage_offset_bytes
            or (not value.is_inference() and value._version != 0)
        ):
            return None
        raw = _coprecessing_plan_tensor_bytes(value)
        if type(raw) is not bytes or len(raw) != element_size:
            return None
        return raw
    except Exception:
        return None


def _aggregate_preterminal_twist_harmonic_key(
    model,
    bulk_twist_harmonics,
):
    """Freeze every request-local harmonic consumed by the twist path."""

    if type(model.harmonics) is not tuple or len(model.harmonics) != 5:
        return None
    model_rows = tuple(
        _aggregate_preterminal_twist_scalar_tensor_key(value)
        for value in model.harmonics
    )
    if any(value is None for value in model_rows):
        return None
    if bulk_twist_harmonics is None:
        bulk_rows = None
    else:
        expected_modes = tuple(
            (ell, emm)
            for ell in range(2, 5)
            for emm in range(-ell, ell + 1)
        )
        if (
            type(bulk_twist_harmonics) is not dict
            or tuple(bulk_twist_harmonics) != expected_modes
        ):
            return None
        bulk_rows = tuple(
            (
                mode,
                _aggregate_preterminal_twist_scalar_tensor_key(
                    bulk_twist_harmonics[mode]
                ),
            )
            for mode in expected_modes
        )
        if any(value is None for _, value in bulk_rows):
            return None
    return model_rows, bulk_rows


def _aggregate_preterminal_twist_cache_key(
    model,
    params,
    modes,
    plan_token,
    bulk_twist_harmonics,
):
    """Build the complete exact pre-terminal result key."""

    inputs = model.inputs
    n_batch = params.get("n_batch")
    if (
        type(plan_token) is not _CoprecessingPlanCacheToken
        or type(params) is not dict
        or not _plain_request_tree_supported(params)
        or (n_batch is not None and (type(n_batch) is not int or n_batch != 1))
        or type(modes) is not list
        or not modes
        or tuple(modes)
        != tuple(mode for mode in _COPRECESSING_MODES if mode in modes)
    ):
        return None
    inclination = params.get("inclination")
    coa_phase = params.get("coa_phase")
    scalars = (
        inputs.theta_jn,
        inputs.alpha0,
        inputs.epsilon0,
        float(0.0 if inclination is None else inclination),
        float(0.0 if coa_phase is None else coa_phase),
    )
    if not all(type(value) is float and math.isfinite(value) for value in scalars):
        return None
    harmonics = _aggregate_preterminal_twist_harmonic_key(
        model,
        bulk_twist_harmonics,
    )
    if harmonics is None:
        return None
    return (
        _aggregate_preterminal_twist_implementation_identity(),
        plan_token.pid,
        plan_token.generation,
        tuple(modes),
        tuple(struct.pack("!d", value) for value in scalars),
        harmonics,
    )


def _aggregate_preterminal_twist_cache_request_supported(
    model,
    frequencies,
    params,
    modes,
    plan,
    plan_token,
    bulk_twist_harmonics,
    *,
    candidate_key=None,
):
    """Admit only plain CPU-f64, n_batch=1 inference requests."""

    inputs = model.inputs
    inference_predicate = getattr(torch, "is_inference", None)
    if (
        not callable(inference_predicate)
        or not torch.is_inference_mode_enabled()
        or torch.is_grad_enabled()
        or not _plain_request_runtime_supported()
        or not _coprecessing_plan_cache_enabled()
        or not _coprecessing_plan_angle_core_enabled()
        or not _reference_bulk_angle_lane_enabled()
        or type(plan_token) is not _CoprecessingPlanCacheToken
        or type(plan) is not _CoprecessingPlan
        or type(plan.reference_angle_core) is not _ReferenceModeMSAAngleCore
        or model.msa_reference_angles_deferred is not True
        or type(inputs.device) is not torch.device
        or inputs.device != torch.device("cpu")
        or inputs.real_dtype is not torch.float64
        or inputs.complex_dtype is not torch.complex128
        or type(inputs.polarization_rotation) is not float
        or type(inputs.long_asc_nodes) is not float
        or not math.isfinite(inputs.polarization_rotation)
        or not math.isfinite(inputs.long_asc_nodes)
        or type(frequencies) is not torch.Tensor
        or frequencies.layout is not torch.strided
        or frequencies.device != torch.device("cpu")
        or frequencies.dtype is not torch.float64
        or frequencies.ndim != 1
        or frequencies.numel() == 0
        or frequencies.stride() != (1,)
        or not frequencies.is_contiguous()
        or frequencies.storage_offset() != 0
        or frequencies._base is not None
        or frequencies.is_conj()
        or frequencies.is_neg()
        or frequencies.requires_grad
        or frequencies.grad_fn is not None
        or inference_predicate(frequencies) is not True
        or IMRPhenomX_utils._tensor_has_forward_ad(frequencies)
    ):
        return False
    try:
        if frequencies.untyped_storage().nbytes() != (
            frequencies.numel() * frequencies.element_size()
        ):
            return False
        if candidate_key is None:
            candidate_key = _aggregate_preterminal_twist_cache_key(
                model,
                params,
                modes,
                plan_token,
                bulk_twist_harmonics,
            )
        # The token was issued by the plan lookup only after schema and
        # fingerprint validation.  An aggregate hit is an independently owned
        # exact result for that generation, so re-hashing the full plan here is
        # unnecessary.  Cold aggregate stores revalidate the token and plan
        # fingerprint after computation, where a concurrent mutation matters.
        return candidate_key is not None
    except Exception:
        return False


def _aggregate_preterminal_twist_cache_tensors(entry):
    return entry.plus, entry.cross


def _aggregate_preterminal_twist_cache_schema_supported(
    plus,
    cross,
    frequencies,
    *,
    privately_owned,
):
    """Validate aggregate shape, precision, AD state, and ownership."""

    inference_predicate = getattr(torch, "is_inference", None)
    if not callable(inference_predicate):
        return False
    pointers = []
    try:
        frequency_pointer = frequencies.untyped_storage().data_ptr()
        for value in (plus, cross):
            if (
                type(value) is not torch.Tensor
                or value.layout is not torch.strided
                or value.device != torch.device("cpu")
                or value.dtype is not torch.complex128
                or value.shape != frequencies.shape
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value._base is not None
                or value.is_conj()
                or value.is_neg()
                or value.requires_grad
                or value.grad_fn is not None
                or IMRPhenomX_utils._tensor_has_forward_ad(value)
            ):
                return False
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            if (
                pointer == 0
                or pointer == frequency_pointer
                or storage.nbytes() != value.numel() * value.element_size()
            ):
                return False
            if privately_owned and (
                inference_predicate(value) is not False or value._version != 0
            ):
                return False
            pointers.append(pointer)
    except Exception:
        return False
    return len(pointers) == len(set(pointers))


def _copy_aggregate_preterminal_twist_for_cache(plus, cross):
    """Make ordinary detached clones owned only by the aggregate cache."""

    def owned(value):
        with torch.inference_mode(False), torch.no_grad():
            # A direct clone of an inference tensor starts with version 1 on
            # supported Torch CPU builds.  A private NumPy copy preserves every
            # payload bit while producing an ordinary version-0 tensor, which
            # makes any later mutation independently detectable.
            return torch.from_numpy(value.detach().numpy().copy(order="C"))

    return owned(plus), owned(cross)


def _aggregate_preterminal_twist_fingerprints(plus, cross):
    return tuple(
        hashlib.sha256(_coprecessing_plan_tensor_bytes(value)).digest()
        for value in (plus, cross)
    )


def _aggregate_preterminal_twist_raw_clone_canary(
    source_plus,
    source_cross,
    candidate_plus,
    candidate_cross,
):
    """Require the private ownership copy to preserve every raw byte."""

    try:
        return all(
            source.dtype == candidate.dtype
            and source.device == candidate.device
            and source.shape == candidate.shape
            and _coprecessing_plan_tensor_bytes(source)
            == _coprecessing_plan_tensor_bytes(candidate)
            for source, candidate in zip(
                (source_plus, source_cross),
                (candidate_plus, candidate_cross),
            )
        )
    except Exception:
        return False


def _aggregate_preterminal_twist_cache_entry_bytes(
    key,
    plus,
    cross,
    fingerprints,
):
    """Charge the exact retained graph plus conservative fixed overhead."""

    return (
        _coprecessing_plan_deep_size(key)
        + _coprecessing_plan_deep_size((plus, cross))
        + _coprecessing_plan_deep_size(fingerprints)
        + plus.untyped_storage().nbytes()
        + cross.untyped_storage().nbytes()
        + _AGGREGATE_PRETERMINAL_TWIST_CACHE_ENTRY_OVERHEAD
    )


def _aggregate_preterminal_twist_cache_reset_if_forked_locked():
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_PID
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE

    process_id = os.getpid()
    if process_id == _AGGREGATE_PRETERMINAL_TWIST_CACHE_PID:
        return
    _AGGREGATE_PRETERMINAL_TWIST_CACHE.clear()
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE = 0
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS.clear()
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS.update(
        _new_aggregate_preterminal_twist_cache_counters()
    )
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_PID = process_id


def _aggregate_preterminal_twist_cache_after_fork():
    """Drop inherited aggregates and replace the inherited lock."""

    global _AGGREGATE_PRETERMINAL_TWIST_CACHE
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_PID
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS

    _AGGREGATE_PRETERMINAL_TWIST_CACHE = OrderedDict()
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK = threading.RLock()
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_PID = os.getpid()
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE = 0
    _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS = (
        _new_aggregate_preterminal_twist_cache_counters()
    )


def _clear_aggregate_preterminal_twist_cache():
    """Clear aggregate entries and reset only their diagnostics."""

    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE

    with _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK:
        _aggregate_preterminal_twist_cache_reset_if_forked_locked()
        _AGGREGATE_PRETERMINAL_TWIST_CACHE.clear()
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE = 0
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS.clear()
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS.update(
            _new_aggregate_preterminal_twist_cache_counters()
        )


def _aggregate_preterminal_twist_cache_stats():
    """Return an owned diagnostic snapshot for tests and profiling."""

    with _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK:
        _aggregate_preterminal_twist_cache_reset_if_forked_locked()
        result = dict(_AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS)
        result.update(
            pid=_AGGREGATE_PRETERMINAL_TWIST_CACHE_PID,
            entries=len(_AGGREGATE_PRETERMINAL_TWIST_CACHE),
            bytes=_AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE,
            budget_bytes=(
                _aggregate_preterminal_twist_cache_budget() or 0
            ),
            max_entries=(
                _AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES
            ),
        )
        return result


def _aggregate_preterminal_twist_cache_note(name):
    try:
        with _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK:
            _aggregate_preterminal_twist_cache_reset_if_forked_locked()
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS[name] += 1
    except Exception:
        pass


def _trim_aggregate_preterminal_twist_cache_locked(budget):
    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE

    while (
        _AGGREGATE_PRETERMINAL_TWIST_CACHE
        and (
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE > budget
            or len(_AGGREGATE_PRETERMINAL_TWIST_CACHE)
            > _AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES
        )
    ):
        _, entry = _AGGREGATE_PRETERMINAL_TWIST_CACHE.popitem(last=False)
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE -= entry.nbytes
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["evictions"] += 1


def _lookup_aggregate_preterminal_twist_cache(key, frequencies, budget):
    """Return validated private aggregates and update their LRU recency."""

    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE

    with _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK:
        _aggregate_preterminal_twist_cache_reset_if_forked_locked()
        _trim_aggregate_preterminal_twist_cache_locked(budget)
        entry = _AGGREGATE_PRETERMINAL_TWIST_CACHE.get(key)
        if entry is None:
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["misses"] += 1
            return None
        valid = _aggregate_preterminal_twist_cache_schema_supported(
            entry.plus,
            entry.cross,
            frequencies,
            privately_owned=True,
        )
        if valid:
            valid = (
                _aggregate_preterminal_twist_fingerprints(
                    entry.plus,
                    entry.cross,
                )
                == entry.fingerprints
            )
        if not valid:
            _AGGREGATE_PRETERMINAL_TWIST_CACHE.pop(key, None)
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE -= entry.nbytes
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS[
                "invalidations"
            ] += 1
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["misses"] += 1
            return None
        _AGGREGATE_PRETERMINAL_TWIST_CACHE.move_to_end(key)
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["hits"] += 1
        return entry.plus, entry.cross


def _store_aggregate_preterminal_twist_cache(
    key,
    plus,
    cross,
    frequencies,
    budget,
    plan_token,
    modes,
    revalidate_key,
):
    """Clone, revalidate, and insert one exact aggregate if still current."""

    global _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE

    if not _aggregate_preterminal_twist_cache_schema_supported(
        plus,
        cross,
        frequencies,
        privately_owned=False,
    ):
        return
    try:
        candidate_plus, candidate_cross = (
            _copy_aggregate_preterminal_twist_for_cache(plus, cross)
        )
        if not _aggregate_preterminal_twist_cache_schema_supported(
            candidate_plus,
            candidate_cross,
            frequencies,
            privately_owned=True,
        ) or not _aggregate_preterminal_twist_raw_clone_canary(
            plus,
            cross,
            candidate_plus,
            candidate_cross,
        ):
            _aggregate_preterminal_twist_cache_note("canary_failures")
            return
        fingerprints = _aggregate_preterminal_twist_fingerprints(
            candidate_plus,
            candidate_cross,
        )
        entry_nbytes = _aggregate_preterminal_twist_cache_entry_bytes(
            key,
            candidate_plus,
            candidate_cross,
            fingerprints,
        )
        current_key = revalidate_key()
    except Exception:
        _aggregate_preterminal_twist_cache_note("canary_failures")
        return
    if (
        current_key != key
        or not _coprecessing_plan_cache_token_current(
            plan_token,
            frequencies,
            modes,
        )
    ):
        _aggregate_preterminal_twist_cache_note("races")
        return

    entry = _AggregatePreterminalTwistCacheEntry(
        candidate_plus,
        candidate_cross,
        fingerprints,
        entry_nbytes,
    )
    with _AGGREGATE_PRETERMINAL_TWIST_CACHE_LOCK:
        _aggregate_preterminal_twist_cache_reset_if_forked_locked()
        if entry.nbytes > budget:
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["oversize"] += 1
            return
        if key in _AGGREGATE_PRETERMINAL_TWIST_CACHE:
            _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["races"] += 1
            return
        _AGGREGATE_PRETERMINAL_TWIST_CACHE[key] = entry
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_SIZE += entry.nbytes
        _AGGREGATE_PRETERMINAL_TWIST_CACHE_COUNTERS["stores"] += 1
        _trim_aggregate_preterminal_twist_cache_locked(budget)


_register_at_fork = getattr(os, "register_at_fork", None)
if _register_at_fork is not None:
    _register_at_fork(after_in_child=_coprecessing_plan_cache_after_fork)
    _register_at_fork(
        after_in_child=_aggregate_preterminal_twist_cache_after_fork
    )
del _register_at_fork


_CUDA_COPRECESSING_PLAN_CACHE = _make_cuda_coprecessing_plan_cache(
    sys.modules[__name__]
)
del _make_cuda_coprecessing_plan_cache
_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE = (
    _make_cuda_aggregate_preterminal_twist_cache(
        sys.modules[__name__],
        _cuda_public_request_proof_current,
    )
)
del _make_cuda_aggregate_preterminal_twist_cache


def _cuda_coprecessing_plan_cache_enabled():
    """Return the strict, independently gated CUDA cache switch."""

    return _CUDA_COPRECESSING_PLAN_CACHE.enabled()


def _cuda_coprecessing_plan_cache_budget():
    """Return the CUDA cache byte budget, or ``None`` if invalid."""

    return _CUDA_COPRECESSING_PLAN_CACHE.budget()


def _clear_cuda_coprecessing_plan_cache():
    """Release every retained device plan and reset diagnostics."""

    _CUDA_COPRECESSING_PLAN_CACHE.clear()


def _cuda_coprecessing_plan_cache_stats():
    """Return an owned CUDA-cache diagnostic snapshot."""

    return _CUDA_COPRECESSING_PLAN_CACHE.stats()


def _cuda_aggregate_preterminal_twist_cache_enabled():
    """Return the strict, independently gated CUDA aggregate switch."""

    return _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.enabled()


def _cuda_aggregate_preterminal_twist_public_fastpath_enabled():
    """Return the strict early public aggregate-hit switch."""

    return _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.public_fastpath_enabled()


def _cuda_aggregate_preterminal_twist_cache_budget():
    """Return the CUDA aggregate byte budget, or ``None`` if invalid."""

    return _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.budget()


def _clear_cuda_aggregate_preterminal_twist_cache():
    """Release CUDA aggregate and retained-plan tensors and diagnostics."""

    _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.clear()


def _cuda_aggregate_preterminal_twist_cache_stats():
    """Return an owned CUDA aggregate-cache diagnostic snapshot."""

    return _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.stats()


def _cuda_aggregate_public_fastpath(params, proof):
    """Materialize one exact fresh public result before model construction."""

    cache = _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE
    request = cache.prepare_public(params, proof)
    if request is None:
        return None
    hit = cache.lookup_public(request)
    if hit is None:
        return None
    entry = hit.entry
    layout = entry.public_layout
    try:
        cosine = math.cos(2.0 * layout.polarization_rotation)
        sine = math.sin(2.0 * layout.polarization_rotation)
        plus, cross = (
            cosine * entry.plus + sine * entry.cross,
            cosine * entry.cross - sine * entry.plus,
        )
        cosine = math.cos(2.0 * request.long_asc_nodes)
        sine = math.sin(2.0 * request.long_asc_nodes)
        plus, cross = (
            cosine * plus + sine * cross,
            cosine * cross - sine * plus,
        )
        result = (
            _series_from_active_samples(
                layout,
                plus,
                layout.npoints,
                layout.first_bin,
                layout.stop_bin,
                layout.delta_f,
            ),
            _series_from_active_samples(
                layout,
                cross,
                layout.npoints,
                layout.first_bin,
                layout.stop_bin,
                layout.delta_f,
            ),
        )
    except Exception:
        cache._note("public_materialization_failures")
        return None
    return result if cache.finish_public_hit(hit) else None


def _mode_angles(model, frequencies, mprime):
    inputs = model.inputs
    velocity = torch.pow(
        _PI * inputs.total_mass_seconds * frequencies * (2.0 / mprime),
        1.0 / 3.0,
    )
    alpha, epsilon, cos_beta = msa_angles(velocity, model.msa_state)
    # LAL initializes all default XPHM mode offsets from the m'=2 reference
    # angle, while evaluating the running angles at 2 f / m'.
    alpha = alpha - model.alpha_offset
    epsilon = epsilon - model.epsilon_offset
    cos_half = torch.sqrt(torch.abs(0.5 * (1.0 + cos_beta)))
    sin_half = torch.sqrt(torch.abs(0.5 * (1.0 - cos_beta)))
    return alpha, epsilon, cos_half, sin_half


def _bulk_mode_angles_supported(model, frequencies, mprimes, params):
    """Admit CPU and CUDA float64 lanes sealed by the bulk-angle screen."""

    inputs = model.inputs
    if (
        not _plain_request_supported(params)
        or not (
            "n_batch" not in params
            or (type(params["n_batch"]) is int and params["n_batch"] == 1)
        )
        or type(inputs.device) is not torch.device
        or type(frequencies) is not torch.Tensor
        or type(frequencies.device) is not torch.device
        or inputs.device != _scheme.mgr.state.torch_device
        or inputs.device.type not in ("cpu", "cuda")
        or frequencies.device.type not in ("cpu", "cuda")
    ):
        return False
    try:
        # TorchScheme may store ``cuda`` without an index, while an allocated
        # tensor reports the current logical device as ``cuda:N``.
        if inputs.device.type == "cuda":
            expected_device = torch.device(
                "cuda",
                (
                    inputs.device.index
                    if inputs.device.index is not None
                    else torch.cuda.current_device()
                ),
            )
        else:
            expected_device = torch.device("cpu")
        tensor_version = frequencies._version
    except Exception:
        return False
    return (
        frequencies.device == expected_device
        and frequencies.dtype == torch.float64
        and inputs.real_dtype == torch.float64
        and inputs.complex_dtype == torch.complex128
        and inputs.prec_version in _MSA_PREC_VERSIONS
        and frequencies.layout == torch.strided
        and frequencies.ndim == 1
        and frequencies.numel() > 0
        and frequencies.stride() == (1,)
        and frequencies.is_contiguous()
        and frequencies.storage_offset() == 0
        and frequencies._base is None
        and tensor_version == 0
        and not frequencies.is_conj()
        and not frequencies.is_neg()
        and not frequencies.requires_grad
        and frequencies.grad_fn is None
        and getattr(torch.autograd.forward_ad, "_current_level", None) == -1
        and type(mprimes) is tuple
        and 0 < len(mprimes) <= len(_COPRECESSING_MODES)
        and all(type(mprime) is int and 1 <= mprime <= 4 for mprime in mprimes)
        and len(set(mprimes)) == len(mprimes)
    )


def _build_reference_mode_msa_angle_core(model, frequencies, mprimes):
    """Build raw MSA rows without request-local offsets or half angles."""

    inputs = model.inputs
    base_velocity_cubed = _PI * inputs.total_mass_seconds * frequencies
    scaled_rows = tuple(
        base_velocity_cubed * (2.0 / mprime) for mprime in mprimes
    )
    velocity_rows = torch.pow(torch.stack(scaled_rows), 1.0 / 3.0)
    try:
        result = _reference_and_mode_msa_angles(
            velocity_rows,
            model.msa_state,
            packed=True,
        )
    except Exception:
        # Preserve the existing deferred-angle fallback exactly.  A failed
        # packed evaluation is request-local and cannot mutate the MSA state.
        result = _reference_and_mode_msa_angles(
            velocity_rows,
            model.msa_state,
            packed=False,
        )
    return _ReferenceModeMSAAngleCore(tuple(mprimes), *result)


def _bulk_mode_angles(
    model,
    frequencies,
    mprimes,
    *,
    reference_angle_core=None,
):
    """Evaluate distinct ``mprime`` MSA angle rows in one Torch call."""

    inputs = model.inputs
    if reference_angle_core is None:
        base_velocity_cubed = _PI * inputs.total_mass_seconds * frequencies
        scaled_rows = tuple(
            base_velocity_cubed * (2.0 / mprime) for mprime in mprimes
        )
        if inputs.real_dtype == torch.float64:
            # The packed float64 pow kernel is raw-identical on qualified CPU and
            # CUDA devices.  CPU float32 can differ by low bits, so preserve its
            # original row-local pow calls while still packing all later work.
            velocity_rows = torch.pow(torch.stack(scaled_rows), 1.0 / 3.0)
        else:
            velocity_rows = torch.stack(
                tuple(
                    torch.pow(scaled_row, 1.0 / 3.0)
                    for scaled_row in scaled_rows
                )
            )
    elif (
        type(reference_angle_core) is not _ReferenceModeMSAAngleCore
        or reference_angle_core.mprimes != mprimes
        or model.msa_reference_angles_deferred is not True
    ):
        raise ValueError("invalid cached reference/mode MSA angle core")
    if model.msa_reference_angles_deferred:
        if reference_angle_core is None:
            try:
                (
                    alpha_rows,
                    epsilon_rows,
                    cos_beta_rows,
                    alpha_reference,
                    epsilon_reference,
                ) = _reference_and_mode_msa_angles(
                    velocity_rows,
                    model.msa_state,
                    packed=True,
                )
            except Exception:
                # Re-run the unchanged scalar-reference plus bulk-row arithmetic.
                # The helper is request-local and does not mutate the MSA state, so
                # a failed packed operation cannot leak into this eager fallback.
                (
                    alpha_rows,
                    epsilon_rows,
                    cos_beta_rows,
                    alpha_reference,
                    epsilon_reference,
                ) = _reference_and_mode_msa_angles(
                    velocity_rows,
                    model.msa_state,
                    packed=False,
                )
        else:
            (
                alpha_rows,
                epsilon_rows,
                cos_beta_rows,
                alpha_reference,
                epsilon_reference,
            ) = (
                reference_angle_core.mode_phiz,
                reference_angle_core.mode_zeta,
                reference_angle_core.mode_cos_beta,
                reference_angle_core.reference_phiz_residual,
                reference_angle_core.reference_zeta_residual,
            )
        alpha_reference = torch.tensor(
            alpha_reference,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        epsilon_reference = torch.tensor(
            epsilon_reference,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        alpha_offset = alpha_reference - inputs.alpha0
        epsilon_offset = epsilon_reference - inputs.epsilon0
    else:
        alpha_rows, epsilon_rows, cos_beta_rows = msa_angles(
            velocity_rows,
            model.msa_state,
        )
        alpha_offset = model.alpha_offset
        epsilon_offset = model.epsilon_offset
    alpha_rows = alpha_rows - alpha_offset
    epsilon_rows = epsilon_rows - epsilon_offset
    cos_half_rows = torch.sqrt(torch.abs(0.5 * (1.0 + cos_beta_rows)))
    sin_half_rows = torch.sqrt(torch.abs(0.5 * (1.0 - cos_beta_rows)))
    own_rows = frequencies.device.type == "cuda"
    return {
        mprime: (
            (alpha_rows[index].clone() if own_rows else alpha_rows[index]),
            (epsilon_rows[index].clone() if own_rows else epsilon_rows[index]),
            (
                cos_half_rows[index].clone()
                if own_rows
                else cos_half_rows[index]
            ),
            (
                sin_half_rows[index].clone()
                if own_rows
                else sin_half_rows[index]
            ),
        )
        for index, mprime in enumerate(mprimes)
    }


def _wigner_columns(ell, mprime, cosine, sine):
    """Return LAL's d^l_(m,+/-m') columns for the native mode set."""

    c2 = cosine * cosine
    c3 = c2 * cosine
    c4 = c3 * cosine
    s2 = sine * sine
    s3 = s2 * sine
    s4 = s3 * sine
    if (ell, mprime) == (2, 2):
        positive = (
            s4,
            2.0 * cosine * s3,
            math.sqrt(6.0) * s2 * c2,
            2.0 * c3 * sine,
            c4,
        )
        negative = (
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    elif (ell, mprime) == (2, 1):
        positive = (
            2.0 * cosine * s3,
            3.0 * c2 * s2 - s4,
            math.sqrt(6.0) * (c3 * sine - cosine * s3),
            c2 * (c2 - 3.0 * s2),
            -2.0 * c3 * sine,
        )
        negative = (
            -positive[4],
            positive[3],
            -positive[2],
            positive[1],
            -positive[0],
        )
    elif (ell, mprime) == (3, 3):
        c5 = c4 * cosine
        c6 = c5 * cosine
        s5 = s4 * sine
        s6 = s5 * sine
        positive = (
            s6,
            math.sqrt(6.0) * cosine * s5,
            math.sqrt(15.0) * c2 * s4,
            2.0 * math.sqrt(5.0) * c3 * s3,
            math.sqrt(15.0) * c4 * s2,
            math.sqrt(6.0) * c5 * sine,
            c6,
        )
        negative = (
            positive[6],
            -positive[5],
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    elif (ell, mprime) == (3, 2):
        c5 = c4 * cosine
        c6 = c5 * cosine
        s5 = s4 * sine
        positive = (
            math.sqrt(6.0) * cosine * s5,
            s4 * (5.0 * c2 - s2),
            math.sqrt(10.0) * s3 * (2.0 * c3 - cosine * s2),
            math.sqrt(30.0) * c2 * (c2 - s2) * s2,
            math.sqrt(10.0) * c3 * (c2 * sine - 2.0 * s3),
            c4 * (c2 - 5.0 * s2),
            -math.sqrt(6.0) * c5 * sine,
        )
        negative = (
            -positive[6],
            positive[5],
            -positive[4],
            positive[3],
            -positive[2],
            positive[1],
            -positive[0],
        )
    elif (ell, mprime) == (4, 4):
        c5 = c4 * cosine
        c6 = c5 * cosine
        c7 = c6 * cosine
        c8 = c7 * cosine
        s5 = s4 * sine
        s6 = s5 * sine
        s7 = s6 * sine
        s8 = s7 * sine
        positive = (
            s8,
            2.0 * math.sqrt(2.0) * cosine * s7,
            2.0 * math.sqrt(7.0) * c2 * s6,
            2.0 * math.sqrt(14.0) * c3 * s5,
            math.sqrt(70.0) * c4 * s4,
            2.0 * math.sqrt(14.0) * c5 * s3,
            2.0 * math.sqrt(7.0) * c6 * s2,
            2.0 * math.sqrt(2.0) * c7 * sine,
            c8,
        )
        negative = (
            positive[8],
            -positive[7],
            positive[6],
            -positive[5],
            positive[4],
            -positive[3],
            positive[2],
            -positive[1],
            positive[0],
        )
    else:  # pragma: no cover - guarded by the fixed native mode set
        raise ValueError(f"unsupported XPHM co-precessing mode {(ell, mprime)}")
    return positive, negative


@lru_cache(maxsize=None)
def _cached_twist_exponential_coefficients(ell, device, real_dtype):
    """Return exact +/- ``i m`` rows on one device and dtype."""

    complex_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[real_dtype]
    emms = range(-ell, ell + 1)
    coefficients = tuple(-1j * emm for emm in emms) + tuple(
        1j * emm for emm in range(-ell, ell + 1)
    )
    return torch.tensor(
        coefficients,
        dtype=complex_dtype,
        device=device,
    ).unsqueeze(1)


def _bulk_twist_exponentials(alpha, ell):
    """Pack the scalar ``exp(+/- i m alpha)`` calls into one native call."""

    packed = _packed_twist_exponentials(alpha, ell)
    if packed is None:
        return None
    packed_ell, negative, positive = packed
    return packed_ell, torch.unbind(negative), torch.unbind(positive)


def _packed_twist_exponentials(alpha, ell):
    """Return explicit packed ``exp(+/- i m alpha)`` tensors."""

    if (
        type(alpha) is not torch.Tensor
        or alpha.layout is not torch.strided
        or alpha.ndim != 1
        or alpha.dtype not in (torch.float32, torch.float64)
        or alpha.device.type not in ("cpu", "cuda")
        or alpha.is_conj()
        or alpha.is_neg()
        or IMRPhenomX_utils._tree_has_autograd(alpha)
    ):
        return None
    width = 2 * ell + 1
    coefficients = _cached_twist_exponential_coefficients(
        ell,
        alpha.device,
        alpha.dtype,
    )
    rows = torch.exp(coefficients * alpha.unsqueeze(0))
    return ell, rows[:width], rows[width:]


def _twist_exponential_recurrence(alpha, ell):
    """Build all ``exp(+/- i m alpha)`` rows from one exponential.

    This follows the arithmetic used by LAL's ``IMRPhenomXPHMTwistUp``:
    compute ``exp(i alpha)`` once, obtain its negative partner by reciprocal,
    and form the remaining integer powers by recurrence.
    """

    if (
        type(alpha) is not torch.Tensor
        or alpha.layout is not torch.strided
        or alpha.ndim != 1
        or alpha.dtype not in (torch.float32, torch.float64)
        or alpha.device.type not in ("cpu", "cuda")
        or alpha.is_conj()
        or alpha.is_neg()
        or IMRPhenomX_utils._tree_has_autograd(alpha)
        or type(ell) is not int
        or ell < 1
    ):
        return None

    exp_i_alpha = torch.exp(1j * alpha)
    exp_mi_alpha = 1.0 / exp_i_alpha
    one = torch.ones_like(exp_i_alpha)

    positive_powers = [one, exp_i_alpha]
    negative_powers = [one, exp_mi_alpha]
    for _ in range(2, ell + 1):
        positive_powers.append(exp_i_alpha * positive_powers[-1])
        negative_powers.append(exp_mi_alpha * negative_powers[-1])

    positive_rows = (
        tuple(reversed(negative_powers[1:]))
        + (one,)
        + tuple(positive_powers[1:])
    )
    return ell, tuple(reversed(positive_rows)), positive_rows


def _stacked_twist_device(inputs, reference):
    """Resolve a strict request device, including an unindexed CUDA device."""

    requested = inputs.device
    if (
        type(requested) is not torch.device
        or requested.type not in ("cpu", "cuda")
        or type(reference) is not torch.Tensor
        or reference.device.type != requested.type
        or (
            requested.index is not None
            and reference.device.index != requested.index
        )
    ):
        return None
    return reference.device


def _plain_twist_tensor(value, *, dtype, device, shape):
    """Return whether ``value`` is an exact-path base strided tensor."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == shape
        and value.dtype == dtype
        and value.device == device
        and not value.is_conj()
        and not value.is_neg()
    )


def _stacked_twist_request_device(model, frequencies, active_modes):
    """Return the exact-path device for one supported stacked request."""

    inputs = model.inputs
    complex_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }.get(inputs.real_dtype)
    device = _stacked_twist_device(inputs, frequencies)
    if (
        device is None
        or inputs.complex_dtype != complex_dtype
        or not _plain_twist_tensor(
            frequencies,
            dtype=inputs.real_dtype,
            device=device,
            shape=frequencies.shape,
        )
        or frequencies.ndim != 1
        or type(model.harmonics) is not tuple
        or len(model.harmonics) != 5
    ):
        return None

    theta = inputs.theta_jn
    if type(theta) is torch.Tensor:
        if not _plain_twist_tensor(
            theta,
            dtype=inputs.real_dtype,
            device=device,
            shape=torch.Size([]),
        ):
            return None
    elif type(theta) is not float or not math.isfinite(theta):
        return None

    for harmonic in model.harmonics:
        if not _plain_twist_tensor(
            harmonic,
            dtype=inputs.complex_dtype,
            device=device,
            shape=torch.Size([]),
        ):
            return None
    for samples in active_modes.values():
        if not _plain_twist_tensor(
            samples,
            dtype=inputs.complex_dtype,
            device=device,
            shape=frequencies.shape,
        ):
            return None
    if IMRPhenomX_utils._tree_has_autograd(
        (theta, frequencies, active_modes, model.harmonics)
    ):
        return None
    return device


def _grouped_outer_twist_request_device(
    model,
    frequencies,
    active_modes,
    modes,
    *,
    stacked_device=None,
):
    """Return the device only for the CUDA schema qualified byte-exact."""

    if tuple(modes) != _COPRECESSING_MODES:
        return None
    device = (
        _stacked_twist_request_device(
            model,
            frequencies,
            active_modes,
        )
        if stacked_device is None
        else stacked_device
    )
    inputs = model.inputs
    if (
        device is None
        or type(device) is not torch.device
        or device.type != "cuda"
        or inputs.real_dtype != torch.float64
        or inputs.complex_dtype != torch.complex128
        or (
            stacked_device is None
            and IMRPhenomX_utils._tree_has_autograd(
                (frequencies, active_modes)
            )
        )
    ):
        return None
    return device


def _packed_twist_harmonics(
    model,
    modes,
    device,
    bulk_twist_harmonics=None,
):
    """Pack the requested scalar harmonics explicitly by ``ell``."""

    inputs = model.inputs
    if (
        bulk_twist_harmonics is not None
        and type(bulk_twist_harmonics) is not dict
    ):
        return None
    packed = {}
    for ell in dict.fromkeys(ell for ell, _ in modes):
        harmonics = []
        for emm in range(-ell, ell + 1):
            harmonic = None
            if bulk_twist_harmonics is not None:
                harmonic = bulk_twist_harmonics.get((ell, emm))
            if harmonic is None and ell == 2:
                harmonic = model.harmonics[emm + ell]
            if harmonic is None:
                harmonic = spin_weighted_spherical_harmonic(
                    inputs.theta_jn,
                    0.0,
                    -2,
                    ell,
                    emm,
                    dtype=inputs.real_dtype,
                    device=device,
                )
            if not _plain_twist_tensor(
                harmonic,
                dtype=inputs.complex_dtype,
                device=device,
                shape=torch.Size([]),
            ):
                return None
            harmonics.append(harmonic)
        packed[ell] = torch.stack(harmonics)
    if IMRPhenomX_utils._tree_has_autograd(packed):
        return None
    return packed


@lru_cache(maxsize=None)
def _cached_stacked_twist_indices(width, device):
    """Return repeated zero indices for exact complex64 CPU accumulation."""

    return torch.zeros(width, dtype=torch.long, device=device)


def _ordered_stacked_twist_sum(terms):
    """Reduce rows in scalar-loop order without reassociation."""

    initial = torch.zeros_like(terms[:1])
    if terms.device.type == "cpu" and terms.dtype == torch.complex64:
        indices = _cached_stacked_twist_indices(
            terms.shape[0],
            terms.device,
        )
        return initial.index_add_(0, indices, terms)[0]
    return torch.cumsum(
        torch.cat((initial, terms), dim=0),
        dim=0,
    )[-1]


def _stacked_twist_mode(
    model,
    frequencies,
    samples,
    ell,
    mprime,
    mode_angles,
    packed_harmonics,
    packed_exponentials,
):
    """Assemble one mode across its m axis, or return ``None`` to fall back."""

    inputs = model.inputs
    device = _stacked_twist_device(inputs, frequencies)
    if (
        device is None
        or (ell, mprime) not in _POSITIVE_COPRECESSING_MODES
        or inputs.real_dtype not in (torch.float32, torch.float64)
        or inputs.complex_dtype
        != {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }[inputs.real_dtype]
        or not _plain_twist_tensor(
            frequencies,
            dtype=inputs.real_dtype,
            device=device,
            shape=frequencies.shape,
        )
        or frequencies.ndim != 1
        or not _plain_twist_tensor(
            samples,
            dtype=inputs.complex_dtype,
            device=device,
            shape=frequencies.shape,
        )
        or type(mode_angles) is not tuple
        or len(mode_angles) != 4
        or not _plain_twist_tensor(
            packed_harmonics,
            dtype=inputs.complex_dtype,
            device=device,
            shape=torch.Size([2 * ell + 1]),
        )
        or type(packed_exponentials) is not tuple
        or len(packed_exponentials) != 3
    ):
        return None

    alpha, epsilon, cosine, sine = mode_angles
    if any(
        not _plain_twist_tensor(
            angle,
            dtype=inputs.real_dtype,
            device=device,
            shape=frequencies.shape,
        )
        for angle in (alpha, epsilon, cosine, sine)
    ):
        return None
    exponential_ell, negative_exponentials, positive_exponentials = (
        packed_exponentials
    )
    if (
        type(exponential_ell) is not int
        or exponential_ell < ell
        or not _plain_twist_tensor(
            negative_exponentials,
            dtype=inputs.complex_dtype,
            device=device,
            shape=torch.Size([2 * exponential_ell + 1, frequencies.shape[0]]),
        )
        or not _plain_twist_tensor(
            positive_exponentials,
            dtype=inputs.complex_dtype,
            device=device,
            shape=torch.Size([2 * exponential_ell + 1, frequencies.shape[0]]),
        )
        or IMRPhenomX_utils._tree_has_autograd(
            (
                frequencies,
                samples,
                mode_angles,
                packed_harmonics,
                packed_exponentials,
            )
        )
    ):
        return None

    positive, negative = _wigner_columns(ell, mprime, cosine, sine)
    width = 2 * ell + 1
    offset = exponential_ell - ell
    negative_terms = (
        negative_exponentials[offset : offset + width]
        * torch.stack(negative)
    ) * packed_harmonics.unsqueeze(1)
    positive_terms = (
        positive_exponentials[offset : offset + width]
        * torch.stack(positive)
    ) * torch.conj(packed_harmonics).unsqueeze(1)
    if ell % 2:
        plus_terms = negative_terms - positive_terms
        cross_terms = 1j * (negative_terms + positive_terms)
    else:
        plus_terms = negative_terms + positive_terms
        cross_terms = 1j * (negative_terms - positive_terms)
    plus_sum = _ordered_stacked_twist_sum(plus_terms)
    cross_sum = _ordered_stacked_twist_sum(cross_terms)
    factor = torch.exp(-1j * mprime * epsilon) * samples / 2.0
    return factor * plus_sum, factor * cross_sum


@lru_cache(maxsize=16)
def _grouped_outer_twist_mprime_coefficients(device, complex_dtype):
    """Return the five fixed ``-i m'`` coefficients on one CUDA device."""

    return torch.tensor(
        (-2j, -1j, -3j, -2j, -4j),
        dtype=complex_dtype,
        device=device,
    ).unsqueeze(1)


def _qualified_grouped_outer_twist_calls(calls, device):
    """Validate the fixed grouped CUDA call schema without tensor algebra."""

    if (
        type(calls) is not tuple
        or len(calls) != len(_COPRECESSING_MODES)
        or type(device) is not torch.device
        or device.type != "cuda"
    ):
        return None

    first_call = calls[0]
    if type(first_call) is not tuple or len(first_call) != 2:
        return None
    first_args = first_call[0]
    if type(first_args) is not tuple or len(first_args) != 5:
        return None
    first_model, frequencies = first_args[:2]
    inputs = first_model.inputs
    if (
        inputs.real_dtype != torch.float64
        or inputs.complex_dtype != torch.complex128
        or frequencies.device != device
        or not _plain_twist_tensor(
            frequencies,
            dtype=inputs.real_dtype,
            device=device,
            shape=frequencies.shape,
        )
        or frequencies.ndim != 1
    ):
        return None

    qualified = [frequencies]
    for expected_mode, call in zip(_COPRECESSING_MODES, calls):
        if type(call) is not tuple or len(call) != 2:
            return None
        args, kwargs = call
        if (
            type(args) is not tuple
            or len(args) != 5
            or type(kwargs) is not dict
        ):
            return None
        model, call_frequencies, samples, ell, mprime = args
        if (
            model is not first_model
            or call_frequencies is not frequencies
            or (ell, mprime) != expected_mode
            or kwargs.get("stacked_twist") is not True
            or not _plain_twist_tensor(
                samples,
                dtype=inputs.complex_dtype,
                device=device,
                shape=frequencies.shape,
            )
        ):
            return None
        angles = kwargs.get("mode_angles")
        packed_harmonics = kwargs.get("packed_harmonics")
        packed_exponentials = kwargs.get("packed_exponentials")
        if (
            type(angles) is not tuple
            or len(angles) != 4
            or any(
                not _plain_twist_tensor(
                    angle,
                    dtype=inputs.real_dtype,
                    device=device,
                    shape=frequencies.shape,
                )
                for angle in angles
            )
            or not _plain_twist_tensor(
                packed_harmonics,
                dtype=inputs.complex_dtype,
                device=device,
                shape=torch.Size([2 * ell + 1]),
            )
            or type(packed_exponentials) is not tuple
            or len(packed_exponentials) != 3
        ):
            return None
        exponential_ell, negative_rows, positive_rows = packed_exponentials
        packed_shape = (
            torch.Size([2 * exponential_ell + 1, frequencies.shape[0]])
            if type(exponential_ell) is int and exponential_ell >= ell
            else None
        )
        if (
            packed_shape is None
            or not _plain_twist_tensor(
                negative_rows,
                dtype=inputs.complex_dtype,
                device=device,
                shape=packed_shape,
            )
            or not _plain_twist_tensor(
                positive_rows,
                dtype=inputs.complex_dtype,
                device=device,
                shape=packed_shape,
            )
        ):
            return None
        qualified.extend(
            (
                samples,
                *angles,
                packed_harmonics,
                negative_rows,
                positive_rows,
            )
        )
    if IMRPhenomX_utils._tree_has_autograd(tuple(qualified)):
        return None
    return inputs, frequencies


def _grouped_outer_twist_modes(calls, device):
    """Assemble the five default CUDA mode lanes in equal-width groups.

    Every frequency-lane expression keeps the arithmetic order of
    :func:`_stacked_twist_mode`.  The cumsums include an explicit initial zero,
    preserving both each mode's ``m`` accumulation and the outer mode order.
    Unsupported tensors return ``None`` so the caller can use the ordinary
    per-mode implementation.
    """

    qualified = _qualified_grouped_outer_twist_calls(calls, device)
    if qualified is None:
        return None
    inputs, frequencies = qualified

    mode_sum_groups = []
    epsilon_rows = []
    sample_rows = []
    for start, stop in ((0, 2), (2, 4), (4, 5)):
        ell = calls[start][0][3]
        width = 2 * ell + 1
        wigner_lanes = []
        exponential_lanes = []
        harmonic_lanes = []
        for args, kwargs in calls[start:stop]:
            samples, _ell, mprime = args[2:5]
            _alpha, epsilon, cosine, sine = kwargs["mode_angles"]
            positive, negative = _wigner_columns(
                ell,
                mprime,
                cosine,
                sine,
            )
            wigner_lanes.append(
                torch.stack(negative + positive).reshape(2, width, -1)
            )
            exponential_ell, negative_rows, positive_rows = kwargs[
                "packed_exponentials"
            ]
            offset = exponential_ell - ell
            exponential_lanes.append(
                torch.stack(
                    (
                        negative_rows[offset : offset + width],
                        positive_rows[offset : offset + width],
                    )
                )
            )
            harmonic_lanes.append(kwargs["packed_harmonics"])
            epsilon_rows.append(epsilon)
            sample_rows.append(samples)

        wigners = torch.stack(wigner_lanes)
        exponentials = torch.stack(exponential_lanes)
        harmonics = torch.stack(harmonic_lanes)
        harmonic_pairs = torch.stack(
            (harmonics, torch.conj(harmonics)),
            dim=1,
        )
        terms = (exponentials * wigners) * harmonic_pairs.unsqueeze(-1)
        negative_terms, positive_terms = terms.unbind(dim=1)
        if ell % 2:
            plus_terms = negative_terms - positive_terms
            cross_terms = 1j * (negative_terms + positive_terms)
        else:
            plus_terms = negative_terms + positive_terms
            cross_terms = 1j * (negative_terms - positive_terms)
        both_terms = torch.stack((plus_terms, cross_terms), dim=1)
        initial = torch.zeros_like(both_terms[:, :, :1])
        mode_sum_groups.append(
            torch.cumsum(
                torch.cat((initial, both_terms), dim=2),
                dim=2,
            )[:, :, -1]
        )

    mode_sums = torch.cat(mode_sum_groups, dim=0)
    coefficients = _grouped_outer_twist_mprime_coefficients(
        device,
        inputs.complex_dtype,
    )
    epsilons = torch.stack(epsilon_rows)
    samples = torch.stack(sample_rows)
    factors = torch.exp(coefficients * epsilons) * samples / 2.0
    mode_outputs = factors.unsqueeze(1) * mode_sums
    outer_initial = torch.zeros_like(mode_outputs[:1])
    return torch.cumsum(
        torch.cat((outer_initial, mode_outputs), dim=0),
        dim=0,
    )[-1].unbind(dim=0)


class _GroupedOuterTwistCudaGraphState(NamedTuple):
    """Owned inputs and outputs for one grouped outer-twist CUDA graph."""

    static_tensors: tuple[torch.Tensor, ...]
    topology: tuple
    unique_indices: tuple[int, ...]
    copy_groups: tuple[tuple[int, ...], ...]
    replay_stream: Any
    capture_stream: Any
    graph: Any
    outputs: tuple[torch.Tensor, torch.Tensor]


_GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE = OrderedDict()
_GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES = OrderedDict()
_GROUPED_OUTER_TWIST_CUDA_GRAPH_LOCK = threading.Lock()
_GROUPED_OUTER_TWIST_CUDA_GRAPH_MAX_ENTRIES = 4
_GROUPED_OUTER_TWIST_CUDA_GRAPH_PID = os.getpid()


def _grouped_outer_twist_cuda_graph_environment():
    """Return every PyCBC switch that can affect captured eager kernels."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("PYCBC_")
        )
    )


def _grouped_outer_twist_cuda_graph_occurrences(calls):
    """Return the 35 ordered tensor uses read by grouped twist algebra."""

    occurrences = []
    for args, kwargs in calls:
        _alpha, epsilon, cosine, sine = kwargs["mode_angles"]
        _exponential_ell, negative_rows, positive_rows = kwargs[
            "packed_exponentials"
        ]
        occurrences.extend(
            (
                args[2],
                epsilon,
                cosine,
                sine,
                kwargs["packed_harmonics"],
                negative_rows,
                positive_rows,
            )
        )
    return tuple(occurrences)


def _grouped_outer_twist_cuda_graph_tensor_metadata(value, device):
    """Return immutable view metadata for one plain contiguous CUDA input."""

    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.device != device
        or value.dtype not in (torch.float64, torch.complex128)
        or value.numel() == 0
        or not value.is_contiguous()
        or value.is_conj()
        or value.is_neg()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        return None
    base = value._base
    if base is not None and (
        type(base) is not torch.Tensor
        or base.layout is not torch.strided
        or base.device != device
        or base.dtype != value.dtype
        or base.is_conj()
        or base.is_neg()
        or base.requires_grad
        or base.grad_fn is not None
    ):
        return None
    try:
        storage_pointer = int(value.untyped_storage().data_ptr())
        if storage_pointer == 0:
            return None
        base_metadata = (
            None
            if base is None
            else (
                tuple(base.shape),
                tuple(base.stride()),
                int(base.storage_offset()),
                bool(base.is_contiguous()),
            )
        )
        metadata = (
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            bool(value.is_contiguous()),
            base_metadata,
        )
    except Exception:
        return None
    return metadata, storage_pointer


def _grouped_outer_twist_cuda_graph_topology(occurrences, device):
    """Describe exact object/view/storage aliases and safe copy lanes."""

    object_groups = {}
    storage_groups = {}
    object_aliases = []
    storage_aliases = []
    metadata = []
    unique_indices = []
    unique_ranges = []
    for index, value in enumerate(occurrences):
        tensor_metadata = _grouped_outer_twist_cuda_graph_tensor_metadata(
            value,
            device,
        )
        if tensor_metadata is None:
            return None
        one_metadata, storage_pointer = tensor_metadata
        object_identity = id(value)
        object_group = object_groups.get(object_identity)
        if object_group is None:
            object_group = len(object_groups)
            object_groups[object_identity] = object_group
            unique_indices.append(index)
            start = int(value.storage_offset()) * value.element_size()
            stop = start + value.numel() * value.element_size()
            for prior_pointer, prior_start, prior_stop in unique_ranges:
                if (
                    prior_pointer == storage_pointer
                    and max(start, prior_start) < min(stop, prior_stop)
                ):
                    return None
            unique_ranges.append((storage_pointer, start, stop))
        storage_group = storage_groups.get(storage_pointer)
        if storage_group is None:
            storage_group = len(storage_groups)
            storage_groups[storage_pointer] = storage_group
        metadata.append(one_metadata)
        object_aliases.append(object_group)
        storage_aliases.append(storage_group)

    copy_groups = OrderedDict()
    for unique_position, occurrence_index in enumerate(unique_indices):
        dtype = occurrences[occurrence_index].dtype
        copy_groups.setdefault(dtype, []).append(unique_position)
    topology = (
        tuple(metadata),
        tuple(object_aliases),
        tuple(storage_aliases),
    )
    return (
        topology,
        tuple(unique_indices),
        tuple(tuple(group) for group in copy_groups.values()),
    )


def _grouped_outer_twist_cuda_graph_runtime_supported(device):
    """Fail closed outside the narrow inference-only CUDA replay runtime."""

    if (
        os.getpid() != _GROUPED_OUTER_TWIST_CUDA_GRAPH_PID
        or type(device) is not torch.device
        or device.type != "cuda"
        or device.index is None
        or not torch.cuda.is_available()
        or not _plain_request_runtime_supported()
        or not callable(getattr(torch, "_foreach_copy_", None))
        or not callable(getattr(torch.cuda, "CUDAGraph", None))
        or not callable(getattr(torch.cuda, "Stream", None))
        or not callable(getattr(torch.cuda, "graph", None))
    ):
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
        stream = torch.cuda.current_stream(device)
        if stream.device != device or type(stream.cuda_stream) is not int:
            return False
    except Exception:
        return False
    return True


def _grouped_outer_twist_cuda_graph_key(device, topology):
    """Return one process/thread/device/stream/topology/environment key."""

    stream = torch.cuda.current_stream(device)
    return (
        os.getpid(),
        threading.get_ident(),
        device,
        stream.cuda_stream,
        topology,
        bool(torch.is_grad_enabled()),
        bool(torch.is_inference_mode_enabled()),
        _grouped_outer_twist_cuda_graph_environment(),
    )


def _grouped_outer_twist_cuda_graph_remember_failure(key):
    """Bound and remember one exact capture/copy/replay failure key."""

    _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE.pop(key, None)
    _GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES.pop(key, None)
    _GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES[key] = None
    while (
        len(_GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES)
        > _GROUPED_OUTER_TWIST_CUDA_GRAPH_MAX_ENTRIES
    ):
        _GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES.popitem(last=False)


def _grouped_outer_twist_cuda_graph_store(key, state):
    """Insert one graph into the bounded least-recently-used cache."""

    _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE.pop(key, None)
    _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE[key] = state
    while (
        len(_GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE)
        > _GROUPED_OUTER_TWIST_CUDA_GRAPH_MAX_ENTRIES
    ):
        _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE.popitem(last=False)


def _clear_grouped_outer_twist_cuda_graph_cache():
    """Release cached graphs/static buffers and remembered failures."""

    with _GROUPED_OUTER_TWIST_CUDA_GRAPH_LOCK:
        _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE.clear()
        _GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES.clear()


def _build_grouped_outer_twist_cuda_graph(
    calls,
    device,
    topology,
    unique_indices,
    copy_groups,
):
    """Warm and capture unchanged grouped outer-twist tensor algebra."""

    occurrences = _grouped_outer_twist_cuda_graph_occurrences(calls)
    static_tensors = tuple(occurrences[index] for index in unique_indices)
    replay_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(replay_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            if _grouped_outer_twist_modes(calls, device) is None:
                raise RuntimeError("grouped CUDA twist warmup was declined")
    replay_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        outputs = _grouped_outer_twist_modes(calls, device)
    if (
        type(outputs) is not tuple
        or len(outputs) != 2
        or any(type(output) is not torch.Tensor for output in outputs)
    ):
        raise RuntimeError("grouped CUDA twist capture returned an invalid schema")
    torch.cuda.synchronize(device)
    return _GroupedOuterTwistCudaGraphState(
        static_tensors=static_tensors,
        topology=topology,
        unique_indices=unique_indices,
        copy_groups=copy_groups,
        replay_stream=replay_stream,
        capture_stream=capture_stream,
        graph=graph,
        outputs=outputs,
    )


def _grouped_outer_twist_cuda_graph_sources_supported(
    state,
    current_tensors,
):
    """Reject cross-request copy overlap with retained graph inputs."""

    try:
        for target, source in zip(state.static_tensors, current_tensors):
            if target is source:
                continue
            if target.untyped_storage().data_ptr() != source.untyped_storage().data_ptr():
                continue
            target_start = target.storage_offset() * target.element_size()
            target_stop = target_start + target.numel() * target.element_size()
            source_start = source.storage_offset() * source.element_size()
            source_stop = source_start + source.numel() * source.element_size()
            if max(target_start, source_start) < min(target_stop, source_stop):
                return False
    except Exception:
        return False
    return True


def _replay_grouped_outer_twist_cuda_graph(
    state,
    occurrences,
    device,
):
    """Copy current inputs, replay once, and return request-owned outputs."""

    current_stream = torch.cuda.current_stream(device)
    if current_stream.cuda_stream != state.replay_stream.cuda_stream:
        raise RuntimeError("grouped CUDA twist replay stream changed")
    current_tensors = tuple(
        occurrences[index] for index in state.unique_indices
    )
    if not _grouped_outer_twist_cuda_graph_sources_supported(
        state,
        current_tensors,
    ):
        raise RuntimeError("grouped CUDA twist input copies overlap")
    for group in state.copy_groups:
        pairs = tuple(
            (state.static_tensors[index], current_tensors[index])
            for index in group
            if state.static_tensors[index] is not current_tensors[index]
        )
        if pairs:
            targets, sources = zip(*pairs)
            torch._foreach_copy_(targets, sources)
    # Capture itself does execute the body, but its output is not returned:
    # every candidate, including the cold one, comes from this post-copy replay.
    state.graph.replay()
    owned = tuple(output.clone() for output in state.outputs)
    for output in state.outputs:
        output.record_stream(current_stream)
    return owned


def _grouped_outer_twist_cuda_graph(calls, device):
    """Return owned graph-replayed outputs, or ``None`` for eager fallback."""

    if (
        _qualified_grouped_outer_twist_calls(calls, device) is None
        or not _grouped_outer_twist_cuda_graph_runtime_supported(device)
    ):
        return None
    try:
        occurrences = _grouped_outer_twist_cuda_graph_occurrences(calls)
        # The enclosing request may have installed the trusted-plain context,
        # which intentionally lets the eager lane skip repeated AD tree walks.
        # A persistent CUDA graph has a stricter contract: recheck its complete
        # captured input tree without consulting that request-local shortcut.
        if IMRPhenomX_utils._tree_has_autograd_untrusted(occurrences):
            return None
        topology_result = _grouped_outer_twist_cuda_graph_topology(
            occurrences,
            device,
        )
        if topology_result is None:
            return None
        topology, unique_indices, copy_groups = topology_result
        key = _grouped_outer_twist_cuda_graph_key(device, topology)
    except Exception:
        return None

    with _GROUPED_OUTER_TWIST_CUDA_GRAPH_LOCK:
        state = _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE.pop(key, None)
        if state is not None:
            _GROUPED_OUTER_TWIST_CUDA_GRAPH_CACHE[key] = state
        elif key not in _GROUPED_OUTER_TWIST_CUDA_GRAPH_FAILURES:
            try:
                state = _build_grouped_outer_twist_cuda_graph(
                    calls,
                    device,
                    topology,
                    unique_indices,
                    copy_groups,
                )
            except Exception:
                _grouped_outer_twist_cuda_graph_remember_failure(key)
            else:
                _grouped_outer_twist_cuda_graph_store(key, state)
    if state is None or state.topology != topology:
        return None
    try:
        return _replay_grouped_outer_twist_cuda_graph(
            state,
            occurrences,
            device,
        )
    except Exception:
        with _GROUPED_OUTER_TWIST_CUDA_GRAPH_LOCK:
            _grouped_outer_twist_cuda_graph_remember_failure(key)
        return None


def _twist_mode(
    model,
    frequencies,
    samples,
    ell,
    mprime,
    *,
    mode_angles=None,
    twist_harmonics=None,
    twist_exponentials=None,
    stacked_twist=False,
    packed_harmonics=None,
    packed_exponentials=None,
):
    inputs = model.inputs
    if mode_angles is None:
        mode_angles = _mode_angles(model, frequencies, mprime)
    alpha, epsilon, cosine, sine = mode_angles
    if stacked_twist:
        stacked = _stacked_twist_mode(
            model,
            frequencies,
            samples,
            ell,
            mprime,
            mode_angles,
            packed_harmonics,
            packed_exponentials,
        )
        if stacked is not None:
            return stacked
    positive, negative = _wigner_columns(
        ell,
        mprime,
        cosine,
        sine,
    )
    plus_sum = torch.zeros_like(samples)
    cross_sum = torch.zeros_like(samples)
    exponentials = {} if twist_harmonics is not None else None
    for index, emm in enumerate(range(-ell, ell + 1)):
        if twist_harmonics is None:
            harmonic = spin_weighted_spherical_harmonic(
                inputs.theta_jn,
                0.0,
                -2,
                ell,
                emm,
                dtype=inputs.real_dtype,
                device=inputs.device,
            )
            if twist_exponentials is None:
                negative_exponential = torch.exp(-1j * emm * alpha)
                positive_exponential = torch.exp(1j * emm * alpha)
            else:
                exponential_ell, negative_rows, positive_rows = (
                    twist_exponentials
                )
                row = emm + exponential_ell
                negative_exponential = negative_rows[row]
                positive_exponential = positive_rows[row]
            negative_term = (
                negative_exponential * negative[index] * harmonic
            )
            positive_term = (
                positive_exponential
                * positive[index]
                * torch.conj(harmonic)
            )
        else:
            key = (ell, emm)
            harmonic = twist_harmonics.get(key)
            if harmonic is None:
                harmonic = spin_weighted_spherical_harmonic(
                    inputs.theta_jn,
                    0.0,
                    -2,
                    ell,
                    emm,
                    dtype=inputs.real_dtype,
                    device=inputs.device,
                )
                twist_harmonics[key] = harmonic

            # Nonzero +/-m partners have identical scalar phase arguments.
            # Keep m=0 separate because its eager expressions carry distinct
            # signed-zero components on some devices and dtypes.
            if twist_exponentials is not None:
                exponential_ell, negative_rows, positive_rows = (
                    twist_exponentials
                )
                row = emm + exponential_ell
                negative_exponential = negative_rows[row]
                positive_exponential = positive_rows[row]
            elif emm == 0:
                negative_exponential = torch.exp(-1j * emm * alpha)
                positive_exponential = torch.exp(1j * emm * alpha)
            else:
                negative_exponential = exponentials.get(-emm)
                if negative_exponential is None:
                    negative_exponential = torch.exp(-1j * emm * alpha)
                    exponentials[-emm] = negative_exponential
                positive_exponential = exponentials.get(emm)
                if positive_exponential is None:
                    positive_exponential = torch.exp(1j * emm * alpha)
                    exponentials[emm] = positive_exponential
            negative_term = negative_exponential * negative[index] * harmonic
            positive_term = (
                positive_exponential * positive[index] * torch.conj(harmonic)
            )
        if ell % 2:
            plus_sum += negative_term - positive_term
            cross_sum += 1j * (negative_term + positive_term)
        else:
            plus_sum += negative_term + positive_term
            cross_sum += 1j * (negative_term - positive_term)

    factor = torch.exp(-1j * mprime * epsilon) * samples / 2.0
    return factor * plus_sum, factor * cross_sum


def _twist_reuse_supported(model, frequencies, active_modes):
    """Return whether exact request-local twist reuse is safe for this call."""

    inputs = model.inputs
    if (
        type(frequencies) is not torch.Tensor
        or frequencies.ndim != 1
        or frequencies.dtype != inputs.real_dtype
        or frequencies.device != inputs.device
    ):
        return False
    harmonics = model.harmonics
    if not isinstance(harmonics, tuple) or len(harmonics) != 5:
        return False
    for harmonic in harmonics:
        if (
            type(harmonic) is not torch.Tensor
            or harmonic.ndim != 0
            or harmonic.dtype != inputs.complex_dtype
            or harmonic.device != inputs.device
        ):
            return False
    for samples in active_modes.values():
        if (
            type(samples) is not torch.Tensor
            or samples.shape != frequencies.shape
            or samples.dtype != inputs.complex_dtype
            or samples.device != inputs.device
        ):
            return False
    return not IMRPhenomX_utils._tree_has_autograd(
        (frequencies, active_modes, harmonics)
    )


def _carrier_alignment_result_reuse_supported(
    model,
    frequencies,
    params,
    *,
    _request_proof,
):
    """Qualify the exact plain CPU request before asking XAS for a result."""

    if (
        not _carrier_alignment_result_reuse_enabled()
        or not _phase_anchor_cache_enabled()
        or _request_proof is not None
        or not _xhm_plain_request_runtime_supported()
        or type(params) is not dict
        or not _plain_request_tree_supported(params)
    ):
        return False

    inputs = model.inputs
    n_batch = params.get("n_batch")
    return (
        (n_batch is None or (type(n_batch) is int and n_batch == 1))
        and inputs.tidal_version is None
        and inputs.lambda1 == 0.0
        and inputs.lambda2 == 0.0
        and inputs.dquad1 == 0.0
        and inputs.dquad2 == 0.0
        and inputs.device == torch.device("cpu")
        and inputs.real_dtype is torch.float64
        and inputs.complex_dtype is torch.complex128
        and type(model.packed_remnant_plan)
        is IMRPhenomX_utils._PackedRemnantPlan
        and type(frequencies) is torch.Tensor
        and frequencies.layout is torch.strided
        and frequencies.device == inputs.device
        and frequencies.dtype is torch.float64
        and frequencies.ndim == 1
        and frequencies.numel() != 0
        and frequencies.is_contiguous()
        and frequencies.storage_offset() == 0
        and frequencies._base is None
        and frequencies._version == 0
        and not frequencies.is_conj()
        and not frequencies.is_neg()
        and not frequencies.requires_grad
        and frequencies.grad_fn is None
        and not IMRPhenomX_utils._tensor_has_forward_ad(frequencies)
    )


def _build_coprecessing_plan(
    model,
    frequencies,
    params,
    modes,
    active_f_max,
    *,
    uniform_grid_metadata,
    request_proof,
    cache_reference_angle_core=False,
):
    """Build the intrinsic carrier, modes, and optional raw MSA core."""

    inputs = model.inputs
    reuse_carrier_amp_plan = _carrier_amp_plan_reuse_enabled()
    reuse_carrier_alignment_result = (
        _carrier_alignment_result_reuse_supported(
            model,
            frequencies,
            params,
            _request_proof=request_proof,
        )
    )
    carrier_alignment_result = None
    if reuse_carrier_amp_plan:
        if reuse_carrier_alignment_result:
            (
                carrier,
                carrier_phase_plan,
                xas_amp_plan,
                carrier_alignment_result,
            ) = _xas_samples(
                model,
                frequencies,
                active_f_max,
                return_phase_plan=True,
                return_amp_plan=True,
                _request_proof=request_proof,
                _return_carrier_alignment_result=True,
            )
        else:
            carrier, carrier_phase_plan, xas_amp_plan = _xas_samples(
                model,
                frequencies,
                active_f_max,
                return_phase_plan=True,
                return_amp_plan=True,
                _request_proof=request_proof,
            )
        carrier_amp_plan = (
            None if xas_amp_plan is None else xas_amp_plan.mergerringdown
        )
    else:
        if reuse_carrier_alignment_result:
            (
                carrier,
                carrier_phase_plan,
                carrier_alignment_result,
            ) = _xas_samples(
                model,
                frequencies,
                active_f_max,
                return_phase_plan=True,
                _request_proof=request_proof,
                _return_carrier_alignment_result=True,
            )
        else:
            carrier, carrier_phase_plan = _xas_samples(
                model,
                frequencies,
                active_f_max,
                return_phase_plan=True,
                _request_proof=request_proof,
            )
        carrier_amp_plan = None
    core = _SequenceCore(carrier * _XAS_MODE_POLARIZATION_FACTOR)
    mode_kwargs = dict(
        frequencies=frequencies,
        reference_frequency=inputs.f_ref,
        final_spin=_coprecessing_final_spin(model),
        remnant=(
            None
            if model.packed_remnant_plan is None
            else model.packed_remnant_plan.carrier
        ),
        carrier_phase_plan=carrier_phase_plan,
        carrier_amp_plan=carrier_amp_plan,
        _uniform_grid_metadata=uniform_grid_metadata,
    )
    if carrier_alignment_result is not None:
        mode_kwargs["carrier_alignment_result"] = carrier_alignment_result
    active_modes = _active_mode_samples(
        core,
        _coprecessing_params(params, inputs),
        modes,
        **mode_kwargs,
    )
    reference_angle_core = None
    if (
        cache_reference_angle_core is True
        and model.msa_reference_angles_deferred is True
    ):
        mprimes = tuple(dict.fromkeys(mprime for _, mprime in modes))
        reference_angle_core = _build_reference_mode_msa_angle_core(
            model,
            frequencies,
            mprimes,
        )
    return _CoprecessingPlan(
        carrier,
        tuple((mode, active_modes[mode]) for mode in modes),
        reference_angle_core,
    )


def _coprecessing_plan_for_request(
    model,
    frequencies,
    params,
    modes,
    active_f_max,
    *,
    uniform_grid_metadata,
    request_proof,
    _cache_token_sink=None,
):
    """Return a warm exact plan, or run the unchanged eager builder."""

    def build(*, cache_reference_angle_core=False):
        return _build_coprecessing_plan(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            uniform_grid_metadata=uniform_grid_metadata,
            request_proof=request_proof,
            cache_reference_angle_core=cache_reference_angle_core,
        )

    if frequencies.device.type == "cuda":
        return _CUDA_COPRECESSING_PLAN_CACHE.get_or_build(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            uniform_grid_metadata=uniform_grid_metadata,
            request_proof=request_proof,
            build=build,
        )

    try:
        cache_enabled = _coprecessing_plan_cache_enabled()
    except Exception:
        cache_enabled = False
    if not cache_enabled:
        return build()

    try:
        angle_core_enabled = _coprecessing_plan_angle_core_enabled()
    except Exception:
        _coprecessing_plan_cache_note_ineligible()
        return build()

    if not _coprecessing_plan_coefficient_tables_supported():
        _invalidate_all_coprecessing_plan_cache_entries()
        _coprecessing_plan_cache_note_ineligible()
        if (
            torch.is_inference_mode_enabled()
            and _COPRECESSING_PLAN_CACHE_COLD_RETRY.get() is not True
        ):
            raise _CoprecessingPlanCacheColdMiss(
                "coefficient-table drift requires an ordinary rebuild"
            )
        return build()

    try:
        budget = _coprecessing_plan_cache_budget()
        supported = _coprecessing_plan_request_supported(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            uniform_grid_metadata,
        )
        if budget is None or not supported:
            _coprecessing_plan_cache_note_ineligible()
            return build()
        key = _coprecessing_plan_key(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            uniform_grid_metadata,
            request_proof,
            angle_core_enabled=angle_core_enabled,
        )
        cached = _lookup_coprecessing_plan_cache(
            key,
            frequencies,
            modes,
            budget,
            token_sink=_cache_token_sink,
        )
    except Exception:
        _coprecessing_plan_cache_note_ineligible()
        return build()
    if cached is not None:
        return cached

    if _COPRECESSING_PLAN_CACHE_COLD_RETRY.get() is not True:
        raise _CoprecessingPlanCacheColdMiss(
            "admitted co-precessing plan cache miss requires a cold rebuild"
        )

    plan = build(cache_reference_angle_core=angle_core_enabled)

    def revalidate_key():
        return _coprecessing_plan_key(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            uniform_grid_metadata,
            request_proof,
            angle_core_enabled=_coprecessing_plan_angle_core_enabled(),
        )

    try:
        current_key = revalidate_key()
        key_bytes = _coprecessing_plan_deep_size(key)
        current_key_bytes = _coprecessing_plan_deep_size(current_key)
    except Exception:
        _coprecessing_plan_cache_note_race()
        return plan
    if current_key != key or current_key_bytes != key_bytes:
        _coprecessing_plan_cache_note_race()
        return plan
    try:
        _store_coprecessing_plan_cache(
            key,
            key_bytes,
            plan,
            frequencies,
            modes,
            budget,
            revalidate_key,
        )
    except Exception:
        pass
    return plan


def _twist_coprecessing_modes(
    model,
    frequencies,
    params,
    modes,
    active_f_max,
    *,
    bulk_twist_harmonics=None,
    _uniform_grid_metadata=None,
    _public_layout_metadata=None,
    _request_proof=None,
    _cuda_request_proof=None,
):
    inputs = model.inputs
    aggregate_cache_requested = False
    try:
        aggregate_cache_requested = (
            _aggregate_preterminal_twist_cache_enabled()
        )
    except Exception:
        _aggregate_preterminal_twist_cache_note("ineligible")
    plan_token_sink = [] if aggregate_cache_requested else None
    coprecessing_plan = _coprecessing_plan_for_request(
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata=_uniform_grid_metadata,
        request_proof=_request_proof,
        _cache_token_sink=plan_token_sink,
    )
    carrier = coprecessing_plan.carrier
    active_modes = dict(coprecessing_plan.active_modes)
    cuda_aggregate_cache_request = None
    if frequencies.device.type == "cuda":

        def revalidate_cuda_aggregate_request():
            return _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.prepare(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                _uniform_grid_metadata,
                coprecessing_plan,
                _cuda_request_proof,
                public_layout_metadata=_public_layout_metadata,
                note=False,
            )

        try:
            cuda_aggregate_cache_request = (
                _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.prepare(
                    model,
                    frequencies,
                    params,
                    modes,
                    active_f_max,
                    _uniform_grid_metadata,
                    coprecessing_plan,
                    _cuda_request_proof,
                    public_layout_metadata=_public_layout_metadata,
                )
            )
            if cuda_aggregate_cache_request is not None:
                cached_cuda_aggregate = (
                    _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.lookup(
                        cuda_aggregate_cache_request,
                        frequencies,
                        modes,
                        revalidate_cuda_aggregate_request,
                    )
                )
                if cached_cuda_aggregate is not None:
                    plus, cross = cached_cuda_aggregate
                    cosine = math.cos(2.0 * inputs.polarization_rotation)
                    sine = math.sin(2.0 * inputs.polarization_rotation)
                    plus, cross = (
                        cosine * plus + sine * cross,
                        cosine * cross - sine * plus,
                    )
                    cosine = math.cos(2.0 * inputs.long_asc_nodes)
                    sine = math.sin(2.0 * inputs.long_asc_nodes)
                    return (
                        cosine * plus + sine * cross,
                        cosine * cross - sine * plus,
                    )
        except Exception:
            cuda_aggregate_cache_request = None
            _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE._note("ineligible")
    aggregate_cache_request = None
    if aggregate_cache_requested and plan_token_sink:
        plan_token = plan_token_sink[0]
        try:
            aggregate_budget = (
                _aggregate_preterminal_twist_cache_budget()
            )
            aggregate_key = (
                _aggregate_preterminal_twist_cache_key(
                    model,
                    params,
                    modes,
                    plan_token,
                    bulk_twist_harmonics,
                )
                if aggregate_budget is not None
                else None
            )
            aggregate_supported = (
                aggregate_key is not None
                and _aggregate_preterminal_twist_cache_request_supported(
                    model,
                    frequencies,
                    params,
                    modes,
                    coprecessing_plan,
                    plan_token,
                    bulk_twist_harmonics,
                    candidate_key=aggregate_key,
                )
            )
            if not aggregate_supported:
                _aggregate_preterminal_twist_cache_note("ineligible")
            else:
                aggregate_cache_request = (
                    aggregate_key,
                    aggregate_budget,
                    plan_token,
                )
                cached_aggregate = (
                    _lookup_aggregate_preterminal_twist_cache(
                        aggregate_key,
                        frequencies,
                        aggregate_budget,
                    )
                )
                if cached_aggregate is not None:
                    plus, cross = cached_aggregate
                    cosine = math.cos(
                        2.0 * inputs.polarization_rotation
                    )
                    sine = math.sin(2.0 * inputs.polarization_rotation)
                    plus, cross = (
                        cosine * plus + sine * cross,
                        cosine * cross - sine * plus,
                    )
                    cosine = math.cos(2.0 * inputs.long_asc_nodes)
                    sine = math.sin(2.0 * inputs.long_asc_nodes)
                    return (
                        cosine * plus + sine * cross,
                        cosine * cross - sine * plus,
                    )
        except Exception:
            aggregate_cache_request = None
            _aggregate_preterminal_twist_cache_note("ineligible")
    elif (
        aggregate_cache_requested
        and _COPRECESSING_PLAN_CACHE_COLD_RETRY.get() is not True
    ):
        _aggregate_preterminal_twist_cache_note("ineligible")
    twist_harmonics = None
    if _twist_reuse_enabled() and _twist_reuse_supported(
        model,
        frequencies,
        active_modes,
    ):
        twist_harmonics = (
            dict(bulk_twist_harmonics)
            if bulk_twist_harmonics is not None
            else {
                (2, emm): harmonic
                for emm, harmonic in zip(range(-2, 3), model.harmonics)
            }
        )
    bulk_exponentials_enabled = _bulk_twist_exponentials_enabled()
    recurrence_enabled = _twist_exponential_recurrence_enabled()
    stacked_twist = False
    stacked_harmonics = None
    if _stacked_twist_enabled() and (
        twist_harmonics is None or frequencies.device.type == "cuda"
    ):
        stacked_device = _stacked_twist_request_device(
            model,
            frequencies,
            active_modes,
        )
        if stacked_device is not None:
            stacked_harmonics = _packed_twist_harmonics(
                model,
                modes,
                stacked_device,
                bulk_twist_harmonics,
            )
            stacked_twist = stacked_harmonics is not None
    grouped_twist_device = None
    if (
        stacked_twist
        and frequencies.device.type == "cuda"
        and _grouped_outer_twist_enabled()
    ):
        grouped_twist_device = _grouped_outer_twist_request_device(
            model,
            frequencies,
            active_modes,
            modes,
            stacked_device=stacked_device,
        )
    grouped_twist_calls = [] if grouped_twist_device is not None else None
    plus = torch.zeros_like(carrier)
    cross = torch.zeros_like(carrier)
    if model.msa_reference_angles_deferred:
        mprimes = tuple(dict.fromkeys(mprime for _, mprime in modes))
        mode_angles = _bulk_mode_angles(
            model,
            frequencies,
            mprimes,
            reference_angle_core=coprecessing_plan.reference_angle_core,
        )
    elif _bulk_mode_angles_enabled():
        mprimes = tuple(dict.fromkeys(mprime for _, mprime in modes))
        mode_angles = None
        if _bulk_mode_angles_supported(model, frequencies, mprimes, params):
            try:
                mode_angles = _bulk_mode_angles(model, frequencies, mprimes)
            except Exception:
                # Bulk evaluation is request-local and does not mutate the
                # model, so the original scalar calls are a safe fallback.
                pass
        if mode_angles is None:
            mode_angles = {}
    else:
        mode_angles = (
            {}
            if (
                _mode_angle_reuse_enabled()
                or bulk_exponentials_enabled
                or recurrence_enabled
                or stacked_twist
            )
            else None
        )
    fused_cpu_twist_done = False
    if _fused_cpu_twist_supported(frequencies, modes, grouped_twist_device):
        try:
            from pycbc.waveform._imrphenomxp_msa_native import _get_extension
            from pycbc.waveform._spherical_harmonics_torch import (
                spin_minus_two_spherical_harmonics_phi_zero,
            )

            ext = _get_extension()
            if ext is not None:
                mprimes = tuple(dict.fromkeys(mprime for _, mprime in modes))
                if mode_angles is None:
                    mode_angles = {}
                missing = tuple(m for m in mprimes if m not in mode_angles)
                if missing:
                    base_v3 = _PI * inputs.total_mass_seconds * frequencies
                    scaled = [base_v3 * (2.0 / m) for m in (1, 2, 3, 4)]
                    v_rows = torch.pow(torch.stack(scaled), 1.0 / 3.0)
                    (
                        phiz_rows,
                        zeta_rows,
                        cos_beta_rows,
                        ref_phiz,
                        ref_zeta,
                    ) = _reference_and_mode_msa_angles(
                        v_rows,
                        model.msa_state,
                        packed=True,
                    )
                    alpha_off = ref_phiz - inputs.alpha0
                    eps_off = ref_zeta - inputs.epsilon0
                    for m in missing:
                        idx = m - 1
                        a = phiz_rows[idx] - alpha_off
                        e = zeta_rows[idx] - eps_off
                        cb = cos_beta_rows[idx]
                        ch = torch.sqrt(torch.abs(0.5 * (1.0 + cb)))
                        sh = torch.sqrt(torch.abs(0.5 * (1.0 - cb)))
                        mode_angles[m] = (a, e, ch, sh)

                harm_dict = spin_minus_two_spherical_harmonics_phi_zero(
                    inputs.theta_jn,
                    dtype=torch.float64,
                    device="cpu",
                )
                ordered_modes = (
                    [(2, m) for m in range(-2, 3)]
                    + [(3, m) for m in range(-3, 4)]
                    + [(4, m) for m in range(-4, 5)]
                )
                harmonics_tensor = torch.tensor(
                    [harm_dict[m] for m in ordered_modes],
                    dtype=torch.complex128,
                    device="cpu",
                )

                mode_samples = [
                    active_modes[ell, mprime].contiguous()
                    for ell, mprime in modes
                ]
                alpha_by_mprime = [
                    torch.empty(0, dtype=torch.float64, device=frequencies.device)
                    for _ in range(5)
                ]
                epsilon_by_mprime = [
                    torch.empty(0, dtype=torch.float64, device=frequencies.device)
                    for _ in range(5)
                ]
                cos_half_by_mprime = [
                    torch.empty(0, dtype=torch.float64, device=frequencies.device)
                    for _ in range(5)
                ]
                sin_half_by_mprime = [
                    torch.empty(0, dtype=torch.float64, device=frequencies.device)
                    for _ in range(5)
                ]

                for mprime in mprimes:
                    angles = mode_angles[mprime]
                    alpha_by_mprime[mprime] = angles[0].contiguous()
                    epsilon_by_mprime[mprime] = angles[1].contiguous()
                    cos_half_by_mprime[mprime] = angles[2].contiguous()
                    sin_half_by_mprime[mprime] = angles[3].contiguous()

                plus, cross = ext.fused_twist_cpu(
                    modes,
                    mode_samples,
                    alpha_by_mprime,
                    epsilon_by_mprime,
                    cos_half_by_mprime,
                    sin_half_by_mprime,
                    harmonics_tensor,
                    0.0,
                    0.0,
                    None,
                    None,
                )
                fused_cpu_twist_done = True
        except Exception:
            fused_cpu_twist_done = False

    if not fused_cpu_twist_done:
        packed_exponentials = None
        stacked_exponentials = None
        max_ell_by_mprime = None
        if recurrence_enabled or bulk_exponentials_enabled or stacked_twist:
            packed_exponentials = (
                {} if recurrence_enabled or bulk_exponentials_enabled else None
            )
            stacked_exponentials = {} if stacked_twist else None
            max_ell_by_mprime = {}
            for ell, mprime in modes:
                max_ell_by_mprime[mprime] = max(
                    ell,
                    max_ell_by_mprime.get(mprime, 0),
                )
        for ell, mprime in modes:
            angles = None
            if mode_angles is not None:
                angles = mode_angles.get(mprime)
                if angles is None:
                    angles = _mode_angles(model, frequencies, mprime)
                    mode_angles[mprime] = angles
            if recurrence_enabled and mprime not in packed_exponentials:
                packed_exponentials[mprime] = _twist_exponential_recurrence(
                    angles[0], max_ell_by_mprime[mprime]
                )
            if stacked_exponentials is not None:
                if mprime not in stacked_exponentials:
                    if recurrence_enabled:
                        scalar_rows = packed_exponentials[mprime]
                        stacked_exponentials[mprime] = (
                            scalar_rows[0],
                            torch.stack(scalar_rows[1]),
                            torch.stack(scalar_rows[2]),
                        )
                    else:
                        stacked_exponentials[mprime] = (
                            _packed_twist_exponentials(
                                angles[0],
                                max_ell_by_mprime[mprime],
                            )
                        )
                        if (
                            bulk_exponentials_enabled
                            and stacked_exponentials[mprime] is not None
                        ):
                            exponential_ell, negative_rows, positive_rows = (
                                stacked_exponentials[mprime]
                            )
                            packed_exponentials[mprime] = (
                                exponential_ell,
                                torch.unbind(negative_rows),
                                torch.unbind(positive_rows),
                            )
            if bulk_exponentials_enabled and mprime not in packed_exponentials:
                packed_exponentials[mprime] = _bulk_twist_exponentials(
                    angles[0], max_ell_by_mprime[mprime]
                )
            exponentials_for_mode = (
                None
                if packed_exponentials is None
                else packed_exponentials.get(mprime)
            )
            stacked_exponentials_for_mode = (
                None
                if stacked_exponentials is None
                else stacked_exponentials.get(mprime)
            )
            if grouped_twist_calls is None:
                mode_plus, mode_cross = _twist_mode(
                    model,
                    frequencies,
                    active_modes[ell, mprime],
                    ell,
                    mprime,
                    mode_angles=angles,
                    twist_harmonics=twist_harmonics,
                    twist_exponentials=exponentials_for_mode,
                    stacked_twist=stacked_twist,
                    packed_harmonics=(
                        None
                        if stacked_harmonics is None
                        else stacked_harmonics[ell]
                    ),
                    packed_exponentials=stacked_exponentials_for_mode,
                )
                plus += mode_plus
                cross += mode_cross
            else:
                grouped_twist_calls.append(
                    (
                        (
                            model,
                            frequencies,
                            active_modes[ell, mprime],
                            ell,
                            mprime,
                        ),
                        {
                            "mode_angles": angles,
                            "twist_harmonics": twist_harmonics,
                            "twist_exponentials": exponentials_for_mode,
                            "stacked_twist": stacked_twist,
                            "packed_harmonics": (
                                None
                                if stacked_harmonics is None
                                else stacked_harmonics[ell]
                            ),
                            "packed_exponentials": (
                                stacked_exponentials_for_mode
                            ),
                        },
                    )
                )

    if grouped_twist_calls is not None:
        grouped_twist_calls = tuple(grouped_twist_calls)
        grouped_outputs = None
        if _grouped_outer_twist_cuda_graph_enabled():
            grouped_outputs = _grouped_outer_twist_cuda_graph(
                grouped_twist_calls,
                grouped_twist_device,
            )
        if grouped_outputs is None:
            grouped_outputs = _grouped_outer_twist_modes(
                grouped_twist_calls,
                grouped_twist_device,
            )
        if grouped_outputs is None:
            for twist_args, twist_kwargs in grouped_twist_calls:
                mode_plus, mode_cross = _twist_mode(
                    *twist_args,
                    **twist_kwargs,
                )
                plus += mode_plus
                cross += mode_cross
        else:
            plus, cross = grouped_outputs

    if cuda_aggregate_cache_request is not None:
        try:
            _CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE.store(
                cuda_aggregate_cache_request,
                plus,
                cross,
                frequencies,
                modes,
                revalidate_cuda_aggregate_request,
            )
        except Exception:
            pass

    if aggregate_cache_request is not None:
        aggregate_key, aggregate_budget, plan_token = aggregate_cache_request

        def revalidate_aggregate_key():
            return _aggregate_preterminal_twist_cache_key(
                model,
                params,
                modes,
                plan_token,
                bulk_twist_harmonics,
            )

        try:
            _store_aggregate_preterminal_twist_cache(
                aggregate_key,
                plus,
                cross,
                frequencies,
                aggregate_budget,
                plan_token,
                modes,
                revalidate_aggregate_key,
            )
        except Exception:
            pass

    cosine = math.cos(2.0 * inputs.polarization_rotation)
    sine = math.sin(2.0 * inputs.polarization_rotation)
    plus, cross = cosine * plus + sine * cross, cosine * cross - sine * plus
    cosine = math.cos(2.0 * inputs.long_asc_nodes)
    sine = math.sin(2.0 * inputs.long_asc_nodes)
    return cosine * plus + sine * cross, cosine * cross - sine * plus


def _dispatch_imrphenomxphm_request(params):
    """Select current fallback code or the sealed qualified target."""

    return _run_xphm_request_proof_plan(
        params,
        imrphenomxphm_native_supported,
        _imrphenomxphm_fd_torch,
    )


def _imrphenomxphm_fd_torch(
    params,
    *,
    _request_proof=None,
    _cuda_request_proof=None,
):
    """Construct a supported regular-grid waveform inside its cache scope."""

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = _as_float(params.get("f_final"))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXPHM frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXPHM delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXPHM f_final must be non-negative")

    xp_params = _xp_params(params)
    inputs = _validated_inputs(xp_params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXPHM cutoff) is <= f_lower")
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
    modes = _requested_coprecessing_modes(params)
    if modes:
        bulk_twist_harmonics = _request_bulk_twist_harmonics(inputs)
        model_harmonics = (
            None
            if bulk_twist_harmonics is None
            else tuple(
                bulk_twist_harmonics[2, emm] for emm in range(-2, 3)
            )
        )
        defer_reference_angles = (
            _reference_bulk_angle_lane_enabled()
            and _reference_bulk_angle_lane_supported(
                params,
                inputs,
                frequencies,
                modes,
            )
        )
        model = _build_model(
            inputs,
            harmonics=model_harmonics,
            _defer_msa_reference_angles=defer_reference_angles,
            _prepare_packed_remnant_plan=_packed_remnant_plan_supported(
                params,
                inputs,
                frequencies,
            ),
        )
        plus, cross = _twist_coprecessing_modes(
            model,
            frequencies,
            params,
            modes,
            active_f_max,
            bulk_twist_harmonics=bulk_twist_harmonics,
            _uniform_grid_metadata=(first_bin, stop_bin, delta_f),
            _public_layout_metadata=(
                npoints,
                first_bin,
                stop_bin,
                delta_f,
            ),
            _request_proof=_request_proof,
            _cuda_request_proof=_cuda_request_proof,
        )
    else:
        plus = torch.zeros(
            frequencies.shape,
            dtype=inputs.complex_dtype,
            device=inputs.device,
        )
        cross = torch.zeros_like(plus)
    return (
        _series_from_active_samples(
            inputs, plus, npoints, first_bin, stop_bin, delta_f
        ),
        _series_from_active_samples(
            inputs, cross, npoints, first_bin, stop_bin, delta_f
        ),
    )


_bind_xphm_request_proof_target = getattr(
    _request_proof_owner,
    "_bind_xphm_request_proof_target",
    None,
)
del _request_proof_owner
if _bind_xphm_request_proof_target is not None:
    _bind_xphm_request_proof_target(
        imrphenomxphm_native_supported,
        _imrphenomxphm_fd_torch,
    )
del _bind_xphm_request_proof_target

_bind_cuda_public_request_fastpath(_cuda_aggregate_public_fastpath)
_bind_cuda_public_request_proof_target(
    _dispatch_imrphenomxphm_request,
    imrphenomxphm_native_supported,
    _imrphenomxphm_fd_torch,
)
(
    _bind_public_result_cache_entry,
    _public_result_cache_runner,
    _clear_public_result_cache,
    _public_result_cache_stats,
) = _make_public_result_cache(
    sys.modules[__name__],
    _dispatch_imrphenomxphm_request,
    imrphenomxphm_native_supported,
    _imrphenomxphm_fd_torch,
)
imrphenomxphm_fd_torch = _make_cuda_public_request_entry(
    _public_result_cache_runner
)
_bind_public_result_cache_entry(imrphenomxphm_fd_torch)
del _bind_cuda_public_request_proof_target
del _bind_cuda_public_request_fastpath
del _make_cuda_public_request_entry
del _cuda_public_request_proof_runner
del _bind_public_result_cache_entry
del _public_result_cache_runner
del _make_public_result_cache


def imrphenomxphm_fd_sequence_torch(**params):
    """Evaluate supported IMRPhenomXPHM configurations with Torch."""

    if not imrphenomxphm_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomXPHM sequence parameters are not supported by the "
            "native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXPHM requires TorchScheme")
    return _run_scoped_xphm_request(
        params,
        _imrphenomxphm_fd_sequence_torch,
    )


def _imrphenomxphm_fd_sequence_torch(params):
    """Construct a supported frequency sequence inside its cache scope."""

    frequencies = _sequence_frequencies(params["sample_points"])
    xp_params = _xp_params(params)
    inputs = _validated_inputs(
        xp_params,
        sequence=True,
        default_reference_frequency=float(frequencies[0].item()),
    )
    modes = _requested_coprecessing_modes(params)
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / inputs.total_mass_seconds
    active_f_max = torch.minimum(
        frequencies.max(),
        frequencies.new_tensor(cutoff_frequency),
    )
    active = frequencies <= active_f_max
    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    if modes and bool(torch.any(active)):
        bulk_twist_harmonics = _request_bulk_twist_harmonics(inputs)
        model_harmonics = (
            None
            if bulk_twist_harmonics is None
            else tuple(
                bulk_twist_harmonics[2, emm] for emm in range(-2, 3)
            )
        )
        model = _build_model(
            inputs,
            harmonics=model_harmonics,
            _prepare_packed_remnant_plan=_packed_remnant_plan_supported(
                params,
                inputs,
                frequencies,
            ),
        )
        plus[active], cross[active] = _twist_coprecessing_modes(
            model,
            frequencies[active],
            params,
            modes,
            active_f_max,
            bulk_twist_harmonics=bulk_twist_harmonics,
        )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def imrphenomxphm_fd_batch(**params):
    """Generate a batch of IMRPhenomXPHM frequency-domain waveforms directly as 2D PyTorch tensors.

    Parameters
    ----------
    mass1 : float or Tensor
        Primary mass in solar masses (shape (B,) or scalar).
    mass2 : float or Tensor
        Secondary mass in solar masses (shape (B,) or scalar).
    spin1x, spin1y, spin1z : float or Tensor, optional
        Primary dimensionless spin components (shape (B,) or scalar, default 0.0).
    spin2x, spin2y, spin2z : float or Tensor, optional
        Secondary dimensionless spin components (shape (B,) or scalar, default 0.0).
    distance : float or Tensor, optional
        Luminosity distance in Mpc (shape (B,) or scalar, default 1.0).
    inclination : float or Tensor, optional
        Inclination angle in radians (shape (B,) or scalar, default 0.0).
    coa_phase : float or Tensor, optional
        Coalescence phase in radians (shape (B,) or scalar, default 0.0).
    long_asc_nodes : float or Tensor, optional
        Longitude of ascending nodes in radians (shape (B,) or scalar, default 0.0).
    f_ref : float or Tensor, optional
        Reference frequency in Hertz (shape (B,) or scalar, default 0.0, uses f_lower).
    delta_f : float
        Frequency resolution in Hertz.
    f_lower : float
        Lower frequency cutoff in Hertz.
    f_final : float, optional
        Upper frequency cutoff in Hertz. If 0.0 or not provided, uses the maximum
        cutoff frequency across the batch.

    Returns
    -------
    hp : torch.Tensor
        Batch of plus polarizations of shape (B, length) on target device.
    hc : torch.Tensor
        Batch of cross polarizations of shape (B, length) on target device.
    """
    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = (
        torch.float32
        if getattr(device, "type", "cpu") == "mps"
        else torch.float64
    )
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )

    batch_size = 1
    for k in (
        "mass1",
        "mass2",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "distance",
        "inclination",
        "coa_phase",
        "long_asc_nodes",
        "f_ref",
    ):
        v = params.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            batch_size = max(batch_size, v.shape[0])
        elif isinstance(v, (list, tuple, _np.ndarray)) and len(v) > 1:
            batch_size = max(batch_size, len(v))

    def _to_tensor(val, default=0.0):
        if val is None:
            val = default
        if isinstance(val, torch.Tensor):
            t = val.to(device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            return t
        elif isinstance(val, (list, tuple, _np.ndarray)):
            t = torch.as_tensor(val, device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            elif t.ndim == 1 and t.shape[0] == 1 and batch_size > 1:
                t = t.repeat(batch_size)
            return t
        else:
            return torch.full(
                (batch_size,), float(val), device=device, dtype=real_dtype
            )

    mass1 = _to_tensor(params["mass1"])
    mass2 = _to_tensor(params["mass2"])
    spin1x = _to_tensor(params.get("spin1x", 0.0))
    spin1y = _to_tensor(params.get("spin1y", 0.0))
    spin1z = _to_tensor(params.get("spin1z", 0.0))
    spin2x = _to_tensor(params.get("spin2x", 0.0))
    spin2y = _to_tensor(params.get("spin2y", 0.0))
    spin2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXPHM frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXPHM delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXPHM f_final must be non-negative")

    # Swap component masses and spins if m2 > m1
    swap_mask = mass2 > mass1
    m1_eff = torch.where(swap_mask, mass2, mass1)
    m2_eff = torch.where(swap_mask, mass1, mass2)
    s1x_eff = torch.where(swap_mask, spin2x, spin1x)
    s1y_eff = torch.where(swap_mask, spin2y, spin1y)
    s1z_eff = torch.where(swap_mask, spin2z, spin1z)
    s2x_eff = torch.where(swap_mask, spin1x, spin2x)
    s2y_eff = torch.where(swap_mask, spin1y, spin2y)
    s2z_eff = torch.where(swap_mask, spin1z, spin2z)

    f_ref_eff = torch.where(f_ref > 0.0, f_ref, torch.full_like(f_ref, f_lower))

    prec_version = int(
        params.get("phenom_x_prec_version")
        if params.get("phenom_x_prec_version") is not None
        else _DEFAULT_PREC_VERSION
    )
    if prec_version not in _MSA_PREC_VERSIONS:
        raise ValueError(f"phenom_x_prec_version {prec_version} is not supported for native batch")

    chi1_l_l, chi2_l_l, chip_l, theta_jn_l, pol_rot_l, final_spin_l, models = [], [], [], [], [], [], []
    for b in range(batch_size):
        p_s = {
            "approximant": "IMRPhenomXP",
            "mass1": float(m1_eff[b]),
            "mass2": float(m2_eff[b]),
            "spin1x": float(s1x_eff[b]),
            "spin1y": float(s1y_eff[b]),
            "spin1z": float(s1z_eff[b]),
            "spin2x": float(s2x_eff[b]),
            "spin2y": float(s2y_eff[b]),
            "spin2z": float(s2z_eff[b]),
            "distance": float(dist[b]),
            "inclination": float(incl[b]),
            "coa_phase": float(coa_phase[b]),
            "f_ref": float(f_ref_eff[b]),
            "delta_f": delta_f,
            "f_lower": f_lower,
            "phenom_x_prec_version": prec_version,
        }
        inp_s = _validated_inputs(p_s)
        mod_s = _build_model(inp_s)
        chi1_l_l.append(inp_s.chi1_l)
        chi2_l_l.append(inp_s.chi2_l)
        chip_l.append(inp_s.chip)
        theta_jn_l.append(inp_s.theta_jn)
        pol_rot_l.append(inp_s.polarization_rotation)
        final_spin_l.append(mod_s.final_spin)
        models.append(mod_s)

    chi1_l = torch.tensor(chi1_l_l, dtype=real_dtype, device=device)
    chi2_l = torch.tensor(chi2_l_l, dtype=real_dtype, device=device)
    chip = torch.tensor(chip_l, dtype=real_dtype, device=device)
    theta_jn = torch.tensor(theta_jn_l, dtype=real_dtype, device=device)
    pol_rot = torch.tensor(pol_rot_l, dtype=real_dtype, device=device)
    final_spin_batch = torch.tensor(final_spin_l, dtype=real_dtype, device=device)

    total_mass_seconds = (m1_eff + m2_eff) * IMRPhenomX_utils.MTSUN
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / total_mass_seconds
    if f_final > 0.0:
        layout_f_max = f_final
        active_f_max = torch.clamp(cutoff_frequency, max=f_final)
    else:
        layout_f_max = float(torch.max(cutoff_frequency).item())
        active_f_max = cutoff_frequency

    if layout_f_max <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    npts = _next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(f_lower / delta_f)
    max_stop_bin = int(layout_f_max / delta_f) + 1

    if first_bin >= max_stop_bin:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    frequencies = torch.arange(first_bin, max_stop_bin, device=device, dtype=real_dtype) * delta_f
    frequencies_2d = frequencies.unsqueeze(0)

    from pycbc.waveform.imrphenomxhm_torch import _imrphenomxhm_modes_batch

    carrier_phase_batch = torch.zeros(batch_size, dtype=real_dtype, device=device)
    mode_dict = _imrphenomxhm_modes_batch(
        m1_eff,
        m2_eff,
        chi1_l,
        chi2_l,
        dist,
        carrier_phase_batch,
        f_ref,
        frequencies,
        delta_f,
        f_lower,
        f_final,
        final_spin_batch=final_spin_batch,
        chip_batch=chip,
    )

    # Precession angles across all active modes and batch items simultaneously
    mprime_factors = (
        2.0 / torch.arange(1, 5, device=device, dtype=real_dtype)
    ).view(4, 1, 1)
    v_3d = torch.pow(
        _PI
        * total_mass_seconds.view(1, batch_size, 1)
        * frequencies.view(1, 1, -1)
        * mprime_factors,
        1.0 / 3.0,
    )
    msa_states = [mod.msa_state for mod in models]
    alpha_3d, epsilon_3d, cos_beta_3d = msa_angles_batch(v_3d, msa_states)

    alpha_offsets = torch.tensor(
        [mod.alpha_offset for mod in models], dtype=real_dtype, device=device
    ).view(1, batch_size, 1)
    epsilon_offsets = torch.tensor(
        [mod.epsilon_offset for mod in models], dtype=real_dtype, device=device
    ).view(1, batch_size, 1)

    alpha_3d = alpha_3d - alpha_offsets
    epsilon_3d = epsilon_3d - epsilon_offsets
    cos_half_3d = torch.sqrt(torch.abs(0.5 * (1.0 + cos_beta_3d)))
    sin_half_3d = torch.sqrt(torch.abs(0.5 * (1.0 - cos_beta_3d)))

    alpha_by_mprime = {mprime: alpha_3d[mprime - 1] for mprime in (1, 2, 3, 4)}
    epsilon_by_mprime = {mprime: epsilon_3d[mprime - 1] for mprime in (1, 2, 3, 4)}
    cos_half_by_mprime = {mprime: cos_half_3d[mprime - 1] for mprime in (1, 2, 3, 4)}
    sin_half_by_mprime = {mprime: sin_half_3d[mprime - 1] for mprime in (1, 2, 3, 4)}

    plus = torch.zeros((batch_size, frequencies.shape[0]), dtype=complex_dtype, device=device)
    cross = torch.zeros((batch_size, frequencies.shape[0]), dtype=complex_dtype, device=device)

    for (ell, mprime), samples in mode_dict.items():
        cosine = cos_half_by_mprime[mprime]
        sine = sin_half_by_mprime[mprime]
        positive, negative = _wigner_columns(ell, mprime, cosine, sine)
        alpha = alpha_by_mprime[mprime]
        epsilon = epsilon_by_mprime[mprime]

        plus_sum = torch.zeros_like(samples)
        cross_sum = torch.zeros_like(samples)

        for index, emm in enumerate(range(-ell, ell + 1)):
            harmonic = spin_weighted_spherical_harmonic(
                theta_jn, 0.0, -2, ell, emm, dtype=real_dtype, device=device
            )
            neg_exp = torch.exp(-1j * emm * alpha)
            pos_exp = torch.exp(1j * emm * alpha)
            neg_term = neg_exp * negative[index] * harmonic.unsqueeze(1)
            pos_term = pos_exp * positive[index] * torch.conj(harmonic).unsqueeze(1)

            if ell % 2:
                plus_sum = plus_sum + (neg_term - pos_term)
                cross_sum = cross_sum + 1j * (neg_term + pos_term)
            else:
                plus_sum = plus_sum + (neg_term + pos_term)
                cross_sum = cross_sum + 1j * (neg_term - pos_term)

        factor = torch.exp(-1j * mprime * epsilon) * samples / 2.0
        plus = plus + factor * plus_sum
        cross = cross + factor * cross_sum

    c_pol = torch.cos(2.0 * pol_rot).unsqueeze(1)
    s_pol = torch.sin(2.0 * pol_rot).unsqueeze(1)
    plus_r = c_pol * plus + s_pol * cross
    cross_r = c_pol * cross - s_pol * plus

    c_nodes = torch.cos(2.0 * long_asc_nodes).unsqueeze(1)
    s_nodes = torch.sin(2.0 * long_asc_nodes).unsqueeze(1)
    hp_active = c_nodes * plus_r + s_nodes * cross_r
    hc_active = c_nodes * cross_r - s_nodes * plus_r

    valid_mask = frequencies_2d <= active_f_max.unsqueeze(1)
    hp_active = torch.where(valid_mask, hp_active, torch.zeros_like(hp_active))
    hc_active = torch.where(valid_mask, hc_active, torch.zeros_like(hc_active))

    hp = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hc = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hp[:, first_bin:max_stop_bin] = hp_active
    hc[:, first_bin:max_stop_bin] = hc_active

    return hp, hc


__all__ = [
    "imrphenomxphm_fd_batch",
    "imrphenomxphm_fd_sequence_torch",
    "imrphenomxphm_fd_torch",
    "imrphenomxphm_native_supported",
    "imrphenomxphm_sequence_native_supported",
]
