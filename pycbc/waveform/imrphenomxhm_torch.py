# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM modes and polarizations.

The quadrupole shares the IMRPhenomXAS implementation.  The ``(2, +/-1)``,
``(3, +/-3)``, and ``(4, +/-4)`` modes use native XHM no-mixing kernels, while
``(3, +/-2)`` includes the native spheroidal-to-spherical ringdown mixing.  The
default and explicit mode sets can be returned directly or assembled into plus
and cross polarizations on the active Torch device.
"""

import math
import os
import sys
import sysconfig
import threading
from contextvars import Context, ContextVar
from numbers import Integral
from typing import NamedTuple

import numpy as _np
import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData

from ._imrphenomxhm_fixed_schema_amplitude import (
    _FixedSchemaAmplitudePlan,
    _evaluate_fixed_schema_amplitude_triplet,
    _fixed_schema_amplitude_triplet_enabled,
)
from ._imrphenomxhm_scripted_phase_triplet import (
    _evaluate_scripted_phase_triplet,
    _scripted_phase_triplet_enabled,
    _scripted_phase_triplet_version_supported,
)
from ._spherical_harmonics_torch import (
    selected_spin_minus_two_spherical_harmonics,
    spin_weighted_spherical_harmonic,
)
from ._torch_jax import torch_context
from .imrphenomxas_torch import (
    _CarrierAlignmentResult,
    _IMRPhenomXASCarrierPlans,
    _XAS_MODE_POLARIZATION_FACTOR,
    _build_packed_frequency_plans_batch,
    _imrphenomxas_amp_plan_type_supported,
    _imrphenomxas_core_torch,
    _imrphenomxas_phase_plan_type_supported,
    _imrphenomxas_ringdown_amp_plan_type_supported,
    _imrphenomxas_sequence_samples,
    _next_power_of_two,
    _request_unqualify_top_plan,
    _series_from_active_samples,
    get_inspiral_phase,
    imrphenomxas_native_supported,
)
from . import imrphenomx_utils_torch as _xutils
from .imrphenomxhm_mode21_torch import (
    _Mode21State,
    _SharedModeInputs,
    _amplitude_21_2022_staged,
    _phase_21_staged,
    _prepare_shared_mode_inputs,
    _run_staged_solves,
    _tensor,
    imrphenomxhm_h2m1_samples,
)
from .imrphenomxhm_mode32_torch import imrphenomxhm_h3m2_samples
from .imrphenomxhm_mode33_torch import (
    _amplitude_33_2022_staged,
    _mode33_state,
    _phase_33_staged,
    imrphenomxhm_h3m3_samples,
)
from .imrphenomxhm_mode44_torch import (
    _amplitude_44_2022_staged,
    _mode44_state,
    _phase_44_staged,
    imrphenomxhm_h4m4_samples,
)
from .torch_switches import _parse_switch


_NATIVE_MODES = frozenset(
    {
        (2, -2),
        (2, -1),
        (2, 1),
        (2, 2),
        (3, -3),
        (3, -2),
        (3, 2),
        (3, 3),
        (4, -4),
        (4, 4),
    }
)
_NATIVE_MODE_FAMILIES = frozenset(
    (ell, abs(emm)) for ell, emm in _NATIVE_MODES
)

# Keep this order in sync with waveform_modes.default_modes.  Defining it here
# avoids importing waveform_modes from its native dispatch target.
_DEFAULT_MODES = (
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
)
_PHASE_ANCHOR_CACHE_ENV = "PYCBC_IMRPHENOMXHM_PHASE_ANCHOR_CACHE"
_CARRIER_ALIGNMENT_RESULT_REUSE_ENV = (
    "PYCBC_IMRPHENOMXHM_CARRIER_ALIGNMENT_RESULT_REUSE"
)
_CARRIER_INSPIRAL_LANE_ENV = (
    "PYCBC_IMRPHENOMXHM_CARRIER_INSPIRAL_LANE"
)
_SHARED_CARRIER_INSPIRAL_PHASE_ENV = (
    "PYCBC_IMRPHENOMXHM_SHARED_CARRIER_INSPIRAL_PHASE"
)
_SHARED_MODE_INPUTS_ENV = "PYCBC_IMRPHENOMXHM_SHARED_MODE_INPUTS"
_REMNANT_CACHE_ENV = "PYCBC_IMRPHENOMXHM_REMNANT_CACHE"
_CARRIER_PLAN_REUSE_ENV = "PYCBC_IMRPHENOMXHM_CARRIER_PLAN_REUSE"
_SCOPED_INFERENCE_ENV = "PYCBC_IMRPHENOMXHM_SCOPED_INFERENCE"
_BATCHED_TINY_SOLVES_ENV = "PYCBC_IMRPHENOMXHM_BATCHED_TINY_SOLVES"
_BULK_POLARIZATION_HARMONICS_ENV = (
    "PYCBC_IMRPHENOMXHM_BULK_POLARIZATION_HARMONICS"
)
_PARALLEL_MODES_CPU_ENV = "PYCBC_IMRPHENOMXHM_PARALLEL_MODES_CPU"
_SCOPED_INFERENCE_ACTIVE = ContextVar(
    "pycbc_imrphenomxhm_scoped_inference_active",
    default=False,
)
_INFERENCE_REQUIRED_SWITCHES = (
    "PYCBC_IMRPHENOMX_PHASE_PLAN",
    "PYCBC_IMRPHENOMX_AMP_PLAN",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_DERIVATIVES",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_AMP_DERIVATIVES",
    _CARRIER_PLAN_REUSE_ENV,
)
_INFERENCE_MODE32_REQUIRED_SWITCHES = (
    "PYCBC_IMRPHENOMXHM_MODE32_DERIVATIVE_REGION_SPECIALIZATION",
)
_PARALLEL_MODE_EXACT_SWITCHES = (
    *_INFERENCE_REQUIRED_SWITCHES[:-1],
    *_INFERENCE_MODE32_REQUIRED_SWITCHES,
)
_CARRIER_INSPIRAL_LANE_MODES = frozenset(
    {(2, 1), (3, 3), (3, 2), (4, 4)}
)
_SHARED_CARRIER_INSPIRAL_PHASE_MODES = frozenset(
    {(3, 3), (3, 2), (4, 4)}
)


def _phase_anchor_cache_enabled():
    """Return the strict, off-by-default carrier-anchor cache switch."""

    value = os.environ.get(_PHASE_ANCHOR_CACHE_ENV)
    return False if value is None else _parse_switch(_PHASE_ANCHOR_CACHE_ENV, value)


def _carrier_alignment_result_reuse_enabled():
    """Return the strict, off-by-default XAS-to-XHM handoff switch."""

    value = os.environ.get(_CARRIER_ALIGNMENT_RESULT_REUSE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CARRIER_ALIGNMENT_RESULT_REUSE_ENV, value)
    )


def _carrier_inspiral_lane_enabled():
    """Return the strict, off-by-default carrier-inspiral lane switch."""

    value = os.environ.get(_CARRIER_INSPIRAL_LANE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CARRIER_INSPIRAL_LANE_ENV, value)
    )


def _shared_carrier_inspiral_phase_enabled():
    """Return the strict, off-by-default shared carrier-phase switch."""

    value = os.environ.get(_SHARED_CARRIER_INSPIRAL_PHASE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_SHARED_CARRIER_INSPIRAL_PHASE_ENV, value)
    )


def _shared_mode_inputs_enabled():
    """Return the strict, off-by-default shared-input switch."""

    value = os.environ.get(_SHARED_MODE_INPUTS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_SHARED_MODE_INPUTS_ENV, value)
    )


def _batched_tiny_solves_enabled():
    """Return the strict, off-by-default cross-mode solve switch."""

    value = os.environ.get(_BATCHED_TINY_SOLVES_ENV)
    return (
        False
        if value is None
        else _parse_switch(_BATCHED_TINY_SOLVES_ENV, value)
    )


def _remnant_cache_enabled():
    """Return the strict, off-by-default request-local remnant-cache switch."""

    value = os.environ.get(_REMNANT_CACHE_ENV)
    return False if value is None else _parse_switch(_REMNANT_CACHE_ENV, value)


def _carrier_plan_reuse_enabled():
    """Return the strict, off-by-default XAS carrier-plan reuse switch."""

    value = os.environ.get(_CARRIER_PLAN_REUSE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CARRIER_PLAN_REUSE_ENV, value)
    )


def _scoped_inference_enabled():
    """Return the strict, off-by-default XHM inference-mode switch."""

    value = os.environ.get(_SCOPED_INFERENCE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_SCOPED_INFERENCE_ENV, value)
    )


def _bulk_polarization_harmonics_enabled():
    """Return the strict, off-by-default polarization-harmonic switch."""

    value = os.environ.get(_BULK_POLARIZATION_HARMONICS_ENV)
    return (
        False
        if value is None
        else _parse_switch(_BULK_POLARIZATION_HARMONICS_ENV, value)
    )


def _parallel_modes_cpu_requested():
    """Read the import-time, strict free-threaded CPU mode switch."""

    value = os.environ.get(_PARALLEL_MODES_CPU_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PARALLEL_MODES_CPU_ENV, value)
    )


def _free_threaded_cpython_runtime_supported():
    """Require a CPython free-threaded build whose GIL remains disabled."""

    if (
        sys.implementation.name != "cpython"
        or sysconfig.get_config_var("Py_GIL_DISABLED") != 1
    ):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if not callable(is_gil_enabled):
        return False
    try:
        return is_gil_enabled() is False
    except Exception:
        return False


_UNSUPPORTED_PLAIN_REQUEST = object()


def _clone_plain_request_tree(value, active_ids=None):
    """Return a private exact-built-in snapshot, or the unsupported marker."""

    if type(value) in (type(None), bool, int, float, str):
        return value
    if type(value) not in (tuple, list, dict):
        return _UNSUPPORTED_PLAIN_REQUEST
    if active_ids is None:
        active_ids = set()
    identity = id(value)
    if identity in active_ids:
        return _UNSUPPORTED_PLAIN_REQUEST
    active_ids.add(identity)
    try:
        if type(value) is dict:
            try:
                snapshot = tuple(value.items())
            except (RuntimeError, TypeError):
                return _UNSUPPORTED_PLAIN_REQUEST
            clone = {}
            for key, item in snapshot:
                if type(key) is not str:
                    return _UNSUPPORTED_PLAIN_REQUEST
                private_item = _clone_plain_request_tree(item, active_ids)
                if private_item is _UNSUPPORTED_PLAIN_REQUEST:
                    return _UNSUPPORTED_PLAIN_REQUEST
                clone[key] = private_item
            return clone
        try:
            snapshot = tuple(value)
        except (RuntimeError, TypeError):
            return _UNSUPPORTED_PLAIN_REQUEST
        private_items = []
        for item in snapshot:
            private_item = _clone_plain_request_tree(item, active_ids)
            if private_item is _UNSUPPORTED_PLAIN_REQUEST:
                return _UNSUPPORTED_PLAIN_REQUEST
            private_items.append(private_item)
        return tuple(private_items) if type(value) is tuple else private_items
    finally:
        active_ids.remove(identity)


def _runtime_boolean(function, *args):
    """Call a Torch runtime predicate, returning ``None`` on uncertainty."""

    if function is None:
        return None
    try:
        return bool(function(*args))
    except Exception:
        return None


def _plain_request_runtime_supported():
    """Reject transforms, AD modes, tensor modes, and CUDA capture."""

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

    compiler = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if _runtime_boolean(compiler) is not False:
        return False
    dynamo = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    if _runtime_boolean(dynamo) is not False:
        return False
    if _runtime_boolean(
        getattr(torch, "is_inference_mode_enabled", None)
    ) is not False:
        return False
    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False

    functorch = getattr(getattr(torch, "_C", None), "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    if dynamic_depth is None:
        return False
    try:
        if dynamic_depth() != 0:
            return False
    except Exception:
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


def _configured_switch_enabled(name):
    """Read one prerequisite using the common strict switch parser."""

    value = os.environ.get(name)
    return False if value is None else _parse_switch(name, value)


def _manual_exact_coverage_supported(params):
    """Require every manual derivative used by direct XHM's exact path."""

    if not all(
        _configured_switch_enabled(name)
        for name in _INFERENCE_REQUIRED_SWITCHES
    ):
        return False
    modes = _requested_modes(params)
    if modes is None:
        return False
    return (3, 2) not in {(ell, abs(emm)) for ell, emm in modes} or all(
        _configured_switch_enabled(name)
        for name in _INFERENCE_MODE32_REQUIRED_SWITCHES
    )


def _native_request_supported(params, generator):
    """Qualify only the three exact public-to-private XHM entrypoint pairs."""

    try:
        if generator is _imrphenomxhm_modes_torch:
            return imrphenomxhm_modes_native_supported(params)
        if generator is _imrphenomxhm_fd_torch:
            return imrphenomxhm_fd_native_supported(params)
        if generator is _imrphenomxhm_fd_sequence_torch:
            return imrphenomxhm_sequence_native_supported(params)
    except (KeyError, TypeError, ValueError):
        return False
    return False


def _inference_exact_coverage_supported(params, *, manual_coverage=None):
    """Restrict scoped inference to the raw-exact manual CPU path."""

    if manual_coverage is None:
        manual_coverage = _manual_exact_coverage_supported(params)
    state = _scheme.mgr.state
    return (
        type(state) is _scheme.TorchScheme
        and state.torch_device.type == "cpu"
        and manual_coverage
    )


def _inference_autograd_failure(error):
    """Recognize only an exact evaluator declining under inference mode."""

    message = str(error)
    return (
        "does not require grad and does not have a grad_fn" in message
        or "Setting requires_grad=True on inference tensor" in message
        or "Inference tensors cannot be saved for backward" in message
    )


def _copy_xhm_public_result(result):
    """Move public Series ownership outside inference mode recursively."""

    if type(result) is dict:
        return {
            mode: tuple(series.copy() for series in pair)
            for mode, pair in result.items()
        }
    return tuple(series.copy() for series in result)


def _run_xhm_request(params, generator, *, inference):
    """Execute one qualified request and fail closed on an AD decline."""

    def generate():
        with _xutils.remnant_cache_context(enabled=_remnant_cache_enabled()):
            return generator(**params)

    if not inference:
        return generate()
    token = _SCOPED_INFERENCE_ACTIVE.set(True)
    try:
        try:
            with torch.inference_mode():
                result = generate()
        except RuntimeError as error:
            if not _inference_autograd_failure(error):
                raise
            # Explicitly defeat inference while rebuilding with a fresh
            # request-local cache.
            with torch.inference_mode(False):
                return generate()
    finally:
        _SCOPED_INFERENCE_ACTIVE.reset(token)
    return _copy_xhm_public_result(result)


def _run_scoped_xhm_request(params, generator):
    """Apply the gated, prequalified direct-XHM inference scope."""

    inference_requested = _scoped_inference_enabled()
    # XPHM's legacy ambient promise must never confer trust on direct XHM.
    with _xutils.trusted_plain_request_context(enabled=False):
        # A trace/profiling callback can synchronously re-enter the public API
        # while the outer request owns inference mode.  Such a nested request
        # must not inherit that private execution scope or leak inference
        # tensors.  Caller-owned ambient inference has no marker and retains
        # its established semantics.
        if _SCOPED_INFERENCE_ACTIVE.get():
            with torch.inference_mode(False):
                return _run_xhm_request(
                    params,
                    generator,
                    inference=False,
                )
        if not inference_requested:
            return _run_xhm_request(
                params,
                generator,
                inference=False,
            )

        private_params = _clone_plain_request_tree(params)
        plain_supported = (
            private_params is not _UNSUPPORTED_PLAIN_REQUEST
            and _plain_request_runtime_supported()
            and _native_request_supported(private_params, generator)
        )
        exact_coverage = (
            _manual_exact_coverage_supported(private_params)
            if plain_supported
            else False
        )
        inference = (
            plain_supported
            and exact_coverage
            and _inference_exact_coverage_supported(
                private_params,
                manual_coverage=exact_coverage,
            )
        )
        call_params = private_params if inference else params
        return _run_xhm_request(
            call_params,
            generator,
            inference=inference,
        )


def _eager_scalar_runtime_supported():
    """Keep graph capture on the established eager scalar implementation."""

    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    try:
        if torch._C._get_tracing_state() is not None:
            return False
    except Exception:
        return False
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is None:
        return True
    try:
        return not is_compiling()
    except Exception:
        return False


def _carrier_inspiral_lane_version_supported():
    """Accept only stable Torch releases with qualified byte exactness."""

    version = getattr(torch, "__version__", None)
    if not isinstance(version, str):
        return False
    release = version.partition("+")[0].split(".")
    # Randomized scalar-versus-packed parity passes on Torch 2.9.1.  Torch
    # 2.10.0 and 2.13.0 change low bits in the vector ``pow`` kernel, so every
    # other or unparseable release fails closed until independently qualified.
    return (
        len(release) == 3
        and release[0] == "2"
        and release[1] == "9"
        and release[2].isdigit()
    )


def _carrier_inspiral_lane_runtime_supported():
    """Require a byte-qualified Torch version and plain eager execution."""

    return (
        _carrier_inspiral_lane_version_supported()
        and _eager_scalar_runtime_supported()
    )


def _shared_carrier_inspiral_phase_runtime_supported():
    """Reject observable Torch modes for the qualified CPU/CUDA lane."""

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

    compiler = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if _runtime_boolean(compiler) is not False:
        return False
    dynamo = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    if _runtime_boolean(dynamo) is not False:
        return False
    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False

    functorch = getattr(getattr(torch, "_C", None), "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    if dynamic_depth is None:
        return False
    try:
        if dynamic_depth() != 0:
            return False
    except Exception:
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


def _carrier_inspiral_lane_plain_request_supported(value):
    """Reject tensor subclasses and aliased tensor-valued request fields."""

    if isinstance(value, torch.Tensor):
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value._base is None
            and value.storage_offset() == 0
            and not value.is_conj()
            and not value.is_neg()
        )
    if isinstance(value, dict):
        return all(
            _carrier_inspiral_lane_plain_request_supported(item)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return all(
            _carrier_inspiral_lane_plain_request_supported(item)
            for item in value
        )
    return True


def _shared_mode_inputs_plain_request_supported(value):
    """Accept exact built-in input trees and reject every tensor leaf."""

    if type(value) in (type(None), bool, int, float, str):
        return True
    if type(value) in (tuple, list):
        return all(
            _shared_mode_inputs_plain_request_supported(item)
            for item in value
        )
    if type(value) is dict:
        return all(
            type(key) in (str, tuple)
            and _shared_mode_inputs_plain_request_supported(key)
            and _shared_mode_inputs_plain_request_supported(item)
            for key, item in value.items()
        )
    return False


def _shared_mode_inputs_request_supported(
    enabled,
    core,
    params,
    mode_families,
    frequencies,
    ringdown_frequencies,
    damping_frequencies,
    reference_frequency,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Qualify exact request-local inputs for one explicitly enabled lane."""

    if (
        not enabled
        or not _eager_scalar_runtime_supported()
        or not _CARRIER_INSPIRAL_LANE_MODES.issubset(mode_families)
        or (2, 1) in ringdown_frequencies
        or (2, 1) in damping_frequencies
        or carrier_coprecessing_deviations is not None
        or not _shared_mode_inputs_plain_request_supported(
            (
                params,
                ringdown_frequencies,
                damping_frequencies,
                reference_frequency,
                final_spin,
                carrier_ringdown_frequency,
                carrier_damping_frequency,
            )
        )
        or _xutils._tree_has_autograd_untrusted(
            (
                core,
                params,
                frequencies,
                final_spin,
                carrier_phase_plan,
                carrier_amp_plan,
            )
        )
    ):
        return False

    polarization = getattr(core, "polarization", None)
    return (
        type(polarization) is torch.Tensor
        and polarization.layout is torch.strided
        and polarization.ndim == 1
        and polarization.dtype == torch.complex128
        and polarization.device.type in ("cpu", "cuda")
        and polarization.is_contiguous()
        and polarization.storage_offset() == 0
        and polarization._base is None
        and not polarization.is_conj()
        and not polarization.is_neg()
        and type(frequencies) is torch.Tensor
        and frequencies.layout is torch.strided
        and frequencies.ndim == 1
        and frequencies.shape == polarization.shape
        and frequencies.dtype == torch.float64
        and frequencies.device == polarization.device
        and frequencies.is_contiguous()
        and frequencies.storage_offset() == 0
        and frequencies._base is None
        and not frequencies.is_conj()
        and not frequencies.is_neg()
    )


def _shared_mode_inputs_supported(
    core,
    params,
    mode_families,
    frequencies,
    ringdown_frequencies,
    damping_frequencies,
    reference_frequency,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Qualify legacy shared inputs only under their independent switch."""

    return _shared_mode_inputs_request_supported(
        _shared_mode_inputs_enabled(),
        core,
        params,
        mode_families,
        frequencies,
        ringdown_frequencies,
        damping_frequencies,
        reference_frequency,
        final_spin,
        carrier_ringdown_frequency,
        carrier_damping_frequency,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_amp_plan,
    )


def _scripted_phase_triplet_shared_inputs_supported(
    core,
    params,
    mode_families,
    frequencies,
    ringdown_frequencies,
    damping_frequencies,
    reference_frequency,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_amp_plan,
    mode21_amplitude_release,
    mode33_amplitude_release,
    mode44_amplitude_release,
):
    """Prequalify private inputs before doing any phase-only preparation."""

    # Exit graph capture before reading the process environment or parsing the
    # Torch build string.  When eager, check the cheap default-off gate before
    # the full runtime and request qualification below.
    if (
        not _eager_scalar_runtime_supported()
        or not _scripted_phase_triplet_enabled()
        or not _shared_carrier_inspiral_phase_runtime_supported()
    ):
        return False
    polarization = getattr(core, "polarization", None)
    handled_modes = ((2, 1), (3, 3), (4, 4))
    if (
        type(polarization) is not torch.Tensor
        or not _scripted_phase_triplet_version_supported(
            polarization.device
        )
        or torch.get_num_threads() != 1
        or type(params.get("n_batch", 1)) is not int
        or params.get("n_batch", 1) != 1
        or not set(handled_modes).issubset(mode_families)
        or any(
            ringdown_frequencies.get(mode) is not None
            for mode in handled_modes
        )
        or any(
            damping_frequencies.get(mode) is not None
            for mode in handled_modes
        )
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or tuple(
            type(value) is int and value == 122022
            for value in (
                mode21_amplitude_release,
                mode33_amplitude_release,
                mode44_amplitude_release,
            )
        )
        != (True, True, True)
        or not 0 < polarization.numel() < 512
    ):
        return False
    return _shared_mode_inputs_request_supported(
        True,
        core,
        params,
        mode_families,
        frequencies,
        ringdown_frequencies,
        damping_frequencies,
        reference_frequency,
        final_spin,
        carrier_ringdown_frequency,
        carrier_damping_frequency,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_amp_plan,
    )


def _shared_carrier_phase_tensor_supported(value, *, shape, dtype, device):
    """Accept one independent plain tensor used by the three-mode lane."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and (shape is None or value.shape == shape)
        and value.dtype == dtype
        and value.device == device
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
    )


def _shared_carrier_phase_plan_value_supported(value, device):
    """Validate immutable CPU-float64 phase-plan scalar leaves."""

    if type(value) is torch.Tensor:
        basic_supported = (
            value.layout is torch.strided
            and value.ndim == 0
            and value.dtype == torch.float64
            and value.device == device
            and not value.is_conj()
            and not value.is_neg()
        )
        if not basic_supported:
            return False
        base = value._base
        if base is None:
            # Some unbound scalar coefficients retain a storage offset while
            # PyTorch deliberately hides their base.  They are read-only in
            # both the packed and the established three-call paths.
            return True
        # XAS prepares a few coefficient banks with ``unbind``.  Accept only
        # those plain, one-dimensional request-local banks; arbitrary views
        # and tensor subclasses still fail closed.
        return (
            type(base) is torch.Tensor
            and base.layout is torch.strided
            and base.ndim == 1
            and base.dtype == torch.float64
            and base.device == device
            and base.is_contiguous()
            and base.storage_offset() == 0
            and base._base is None
            and not base.is_conj()
            and not base.is_neg()
        )
    if isinstance(value, tuple):
        return all(
            _shared_carrier_phase_plan_value_supported(item, device)
            for item in value
        )
    if type(value) is bool:
        return True
    return type(value) in (int, float) and math.isfinite(value)


def _shared_carrier_inspiral_phase_supported(
    core,
    params,
    mode_families,
    shared_inputs,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
):
    """Qualify exact bulk carrier-inspiral evaluation on CPU/CUDA float64."""

    # XPHM's request-local promise is enabled only after the complete public
    # input tree, runtime, and native route have been validated.  Its carrier
    # phase plan and shared inputs are then constructed privately inside that
    # scope, so avoid walking all 65 plan tensors a second time.  Direct XHM
    # deliberately clears the promise and retains the full fail-closed scan.
    trusted_request = _xutils._TRUSTED_PLAIN_REQUEST.get() is True
    if (
        not _shared_carrier_inspiral_phase_enabled()
        or not _shared_carrier_inspiral_phase_runtime_supported()
        or not _SHARED_CARRIER_INSPIRAL_PHASE_MODES.issubset(mode_families)
        or carrier_coprecessing_deviations is not None
        or type(shared_inputs) is not _SharedModeInputs
        or type(shared_inputs.state) is not _Mode21State
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or (
            not trusted_request
            and not _shared_mode_inputs_plain_request_supported(params)
        )
    ):
        return False

    polarization = getattr(core, "polarization", None)
    if type(polarization) is not torch.Tensor:
        return False
    device = polarization.device
    scheme_device = _scheme.mgr.state.torch_device
    if (
        device.type != scheme_device.type
        or (
            scheme_device.index is not None
            and device.index != scheme_device.index
        )
        or not _shared_carrier_phase_tensor_supported(
            polarization,
            shape=None,
            dtype=torch.complex128,
            device=device,
        )
        or polarization.ndim != 1
    ):
        return False
    length_shape = polarization.shape
    tensors_supported = (
        _shared_carrier_phase_tensor_supported(
            shared_inputs.frequencies,
            shape=length_shape,
            dtype=torch.float64,
            device=device,
        )
        and _shared_carrier_phase_tensor_supported(
            shared_inputs.mf,
            shape=length_shape,
            dtype=torch.float64,
            device=device,
        )
        and _shared_carrier_phase_tensor_supported(
            shared_inputs.intrinsic,
            shape=torch.Size([4]),
            dtype=torch.float64,
            device=device,
        )
        and _shared_carrier_phase_tensor_supported(
            shared_inputs.phase_coeffs,
            shape=torch.Size([13, 49]),
            dtype=torch.float64,
            device=device,
        )
    )
    if not tensors_supported:
        return False
    if any(
        value.requires_grad
        for value in (
            polarization,
            shared_inputs.frequencies,
            shared_inputs.mf,
            shared_inputs.intrinsic,
            shared_inputs.phase_coeffs,
        )
    ):
        return False
    if trusted_request:
        return True
    return _shared_carrier_phase_plan_value_supported(
        carrier_phase_plan,
        device,
    ) and not _xutils._tree_has_autograd_untrusted(
        (
            polarization,
            shared_inputs.frequencies,
            shared_inputs.mf,
            shared_inputs.intrinsic,
            shared_inputs.phase_coeffs,
            carrier_phase_plan,
        )
    )


_NO_PHASE_ANCHOR_AUTOGRAD_PROOF = object()


def _carrier_alignment_plain_tensor_supported(value, *, shape, dtype):
    """Accept one owned, non-AD binary64/complex128 CPU result tensor."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == shape
        and value.dtype == dtype
        and value.device == torch.device("cpu")
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
        and not _xutils._tensor_has_forward_ad(value)
    )


def _carrier_alignment_result_supported(
    result,
    core,
    params,
    frequencies,
    reference_frequency,
    final_spin,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
):
    """Qualify one exact request-local XAS-to-XHM result handoff."""

    if (
        not _carrier_alignment_result_reuse_enabled()
        or not _phase_anchor_cache_enabled()
        or not _phase_anchor_cache_supported(core)
        or not _plain_request_runtime_supported()
        or type(result) is not _CarrierAlignmentResult
        or result.phase_plan is not carrier_phase_plan
        or result.reference_frequency is not reference_frequency
        or carrier_coprecessing_deviations is not None
        or type(core) is not _SequenceCore
        or type(params) is not dict
        or params.get("approximant") != "IMRPhenomXHM"
        or not _shared_mode_inputs_plain_request_supported(params)
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or type(reference_frequency) is not float
        or not math.isfinite(reference_frequency)
        or type(final_spin) is not float
        or not math.isfinite(final_spin)
    ):
        return False

    n_batch = params.get("n_batch")
    if n_batch is not None and (type(n_batch) is not int or n_batch != 1):
        return False
    for name in ("lambda1", "lambda2", "dquad_mon1", "dquad_mon2"):
        value = params.get(name, 0.0)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value != 0.0
        ):
            return False

    polarization = core.polarization
    return (
        _carrier_alignment_plain_tensor_supported(
            polarization,
            shape=polarization.shape,
            dtype=torch.complex128,
        )
        and polarization.ndim == 1
        and polarization.numel() != 0
        and _carrier_alignment_plain_tensor_supported(
            frequencies,
            shape=polarization.shape,
            dtype=torch.float64,
        )
        and _carrier_alignment_plain_tensor_supported(
            result.reference_phase,
            shape=torch.Size(()),
            dtype=torch.float64,
        )
        and _carrier_alignment_plain_tensor_supported(
            result.ringdown_start_derivative,
            shape=torch.Size(()),
            dtype=torch.float64,
        )
    )


def _carrier_inspiral_lane_supported(
    core,
    params,
    mode_families,
    frequencies,
    final_spin,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    _phase_anchor_autograd=_NO_PHASE_ANCHOR_AUTOGRAD_PROOF,
):
    """Return whether four scalar carrier evaluations may be packed exactly."""

    if (
        not _carrier_inspiral_lane_enabled()
        or not _carrier_inspiral_lane_runtime_supported()
        or not _CARRIER_INSPIRAL_LANE_MODES.issubset(mode_families)
        or carrier_coprecessing_deviations is not None
        or not _carrier_inspiral_lane_plain_request_supported(
            (params, final_spin)
        )
        or (
            _phase_anchor_inputs_have_autograd(
                core,
                params,
                final_spin,
                carrier_coprecessing_deviations,
                carrier_phase_plan,
            )
            if (
                _phase_anchor_autograd
                is _NO_PHASE_ANCHOR_AUTOGRAD_PROOF
            )
            else _phase_anchor_autograd
        )
    ):
        return False

    polarization = getattr(core, "polarization", None)
    if (
        type(polarization) is not torch.Tensor
        or polarization.layout is not torch.strided
        or polarization.device.type != "cpu"
        or polarization.dtype != torch.complex128
        or not polarization.is_contiguous()
        or polarization.storage_offset() != 0
        or polarization._base is not None
        or polarization.is_conj()
        or polarization.is_neg()
    ):
        return False

    if frequencies is None:
        return all(
            hasattr(core, name)
            for name in ("first_bin", "stop_bin", "delta_f")
        )
    return (
        type(frequencies) is torch.Tensor
        and frequencies.layout is torch.strided
        and frequencies.ndim == 1
        and frequencies.dtype == torch.float64
        and frequencies.device == polarization.device
        and frequencies.is_contiguous()
        and frequencies.storage_offset() == 0
        and frequencies._base is None
        and not frequencies.is_conj()
        and not frequencies.is_neg()
    )


class _CarrierInspiralLane(NamedTuple):
    """Four exact carrier-inspiral values in native XHM mode order."""

    mode21: torch.Tensor
    mode33: torch.Tensor
    mode32: torch.Tensor
    mode44: torch.Tensor


class _SharedCarrierInspiralPhase(NamedTuple):
    """Three exact full-vector carrier phases in native XHM mode order."""

    mode33: torch.Tensor
    mode32: torch.Tensor
    mode44: torch.Tensor


def _prepare_shared_carrier_inspiral_phase(
    shared_inputs,
    carrier_phase_plan,
):
    """Evaluate three full carrier phase vectors in one native operation."""

    mf = shared_inputs.mf
    # Preserve each eager mode's frequency-scale multiplication before the
    # three independent rows enter the common XAS expression.  Mode 21 is not
    # included: its full-vector carrier phase already runs inside the cached
    # packed mode-21 phase executor rather than as a separate eager call.
    carrier_frequencies = torch.stack(
        (
            (2.0 / 3.0) * mf,
            mf,
            0.5 * mf,
        )
    )
    values = get_inspiral_phase(
        carrier_frequencies,
        shared_inputs.intrinsic,
        shared_inputs.phase_coeffs,
        _phase_plan=carrier_phase_plan,
    )
    return _SharedCarrierInspiralPhase(*values.unbind())


def _prepare_carrier_inspiral_lane(shared_inputs, carrier_phase_plan):
    """Evaluate the four mode-alignment carrier inputs in one native call.

    Each scalar input retains the arithmetic tree used by its eager mode.  The
    only packed operation is evaluation of the common XAS inspiral expression.
    """

    state = shared_inputs.state
    mf = shared_inputs.mf
    f_align_21 = 0.5 * state.f_meco_22
    f_align_33 = 1.5 * state.f_meco_22
    f_align_32 = state.f_meco_22
    f_align_44 = 2.0 * state.f_meco_22
    if state.eta > 0.05:
        f_align_21 *= 0.6
        f_align_33 *= 0.6
        f_align_32 *= 0.6
        f_align_44 *= 0.6

    # Preserve the scalar Torch multiplications performed inside each mode's
    # eager ``inspiral_raw`` closure before stacking the four independent
    # results into the fixed native lane.
    carrier_frequencies = torch.stack(
        (
            2.0 * _tensor(f_align_21, mf),
            (2.0 / 3.0) * _tensor(f_align_33, mf),
            _tensor(f_align_32, mf),
            0.5 * _tensor(f_align_44, mf),
        )
    )
    values = get_inspiral_phase(
        carrier_frequencies,
        shared_inputs.intrinsic,
        shared_inputs.phase_coeffs,
        _phase_plan=carrier_phase_plan,
    )
    return _CarrierInspiralLane(*values.unbind())


class _CarrierPhaseAnchors:
    """Request-local immutable carrier-phase results shared by XHM modes."""

    def __init__(self, carrier_alignment_result=None):
        self._values = {}
        self._carrier_alignment_result = carrier_alignment_result

    def get_or_compute(self, name, like, factory):
        """Return one anchor per name, device, and real dtype."""

        # The alignment frequency reaches the carrier through deliberately
        # different arithmetic trees in the 21, 32, and 44 mode ports.  Those
        # trees agree bitwise on CPU, but Torch 2.1 CUDA can round a reused
        # alignment anchor differently from the independent evaluations.  The
        # reference phase and ringdown-start derivative have identical inputs
        # in every mode, so they remain safe to share across either backend.
        if like.device.type == "cuda" and name == "alignment_phase":
            return factory()

        key = (name, like.device, like.dtype)
        try:
            return self._values[key]
        except KeyError:
            pass

        result = self._carrier_alignment_result
        if type(result) is _CarrierAlignmentResult and like.device.type == "cpu":
            if name == "reference_phase":
                value = result.reference_phase
            elif name == "ringdown_start_derivative":
                value = result.ringdown_start_derivative
            else:
                value = None
            if value is not None and _carrier_alignment_plain_tensor_supported(
                value,
                shape=torch.Size(()),
                dtype=like.dtype,
            ):
                self._values[key] = value
                return value

        value = factory()
        self._values[key] = value
        return value


def _phase_anchor_inputs_have_autograd(
    core,
    params,
    final_spin,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
):
    """Return whether sharing anchors could retain an autograd graph."""

    values = [core, params, final_spin, carrier_phase_plan]
    deviations = carrier_coprecessing_deviations
    if deviations is not None:
        values.append(getattr(deviations, "strength", None))
        fits = getattr(deviations, "fits", None)
        if fits is not None:
            try:
                values.extend(vars(fits).values())
            except TypeError:
                values.append(fits)
    return _xutils._tree_has_autograd(values)


def _phase_anchor_cache_supported(core):
    """Return whether carrier anchors are byte-identical when shared.

    Alignment anchors receive a narrower CUDA fallback in
    :meth:`_CarrierPhaseAnchors.get_or_compute`; the two anchors with identical
    mode-family inputs remain exact and useful on accelerators.
    """

    polarization = getattr(core, "polarization", None)
    return (
        type(polarization) is torch.Tensor
        and polarization.layout == torch.strided
        and polarization.device.type in ("cpu", "cuda")
    )


class _SequenceCore(NamedTuple):
    """Inclination-independent samples shared by the XHM mode kernels."""

    polarization: torch.Tensor


def _requested_modes(params):
    mode_array = params.get("mode_array")
    if mode_array is None:
        return list(_DEFAULT_MODES)

    modes = []
    try:
        for mode in mode_array:
            ell, emm = mode
            if not isinstance(ell, Integral) or not isinstance(emm, Integral):
                return None
            modes.append((int(ell), int(emm)))
    except (TypeError, ValueError):
        return None
    return modes


def _xas_params(params):
    xas = dict(params)
    xas["approximant"] = "IMRPhenomXAS"
    xas["mode_array"] = None
    # These angles belong to polarization assembly and are intentionally
    # ignored by the mode-by-mode interface.
    xas["inclination"] = 0.0
    xas["long_asc_nodes"] = 0.0
    return xas


def _carrier_plan_value_supported(value, like):
    """Validate immutable scalar plan leaves against the carrier tensor."""

    if type(value) is torch.Tensor:
        return (
            value.layout is torch.strided
            and value.ndim == 0
            and value.dtype == like.real.dtype
            and value.device == like.device
            and not value.is_conj()
            and not value.is_neg()
        )
    if isinstance(value, tuple):
        return all(_carrier_plan_value_supported(item, like) for item in value)
    if type(value) is bool:
        return True
    return type(value) in (int, float) and math.isfinite(value)


def _carrier_plan_reuse_supported(core, params, carrier_plans):
    """Qualify exact request-local reuse of plans built for this carrier."""

    polarization = getattr(core, "polarization", None)
    return (
        _eager_scalar_runtime_supported()
        and _shared_mode_inputs_plain_request_supported(params)
        and type(carrier_plans) is _IMRPhenomXASCarrierPlans
        and _imrphenomxas_phase_plan_type_supported(carrier_plans.phase)
        and _imrphenomxas_amp_plan_type_supported(carrier_plans.amplitude)
        and type(polarization) is torch.Tensor
        and polarization.layout is torch.strided
        and polarization.ndim == 1
        and polarization.dtype == torch.complex128
        and polarization.device.type in ("cpu", "cuda")
        and polarization.is_contiguous()
        and polarization.storage_offset() == 0
        and polarization._base is None
        and not polarization.is_conj()
        and not polarization.is_neg()
        and _carrier_plan_value_supported(
            carrier_plans,
            polarization,
        )
        and not _xutils._tree_has_autograd_untrusted(
            (core, params, carrier_plans)
        )
    )


def _carrier_plans_for_modes(core, params, carrier_plans):
    """Return exact mode-kernel plans, or fail closed to legacy rebuilding."""

    if not _carrier_plan_reuse_supported(core, params, carrier_plans):
        return None, None
    return carrier_plans.phase, carrier_plans.amplitude.mergerringdown


def _carrier_core_for_modes(params):
    """Build the XAS carrier and optionally expose its exact prepared plans."""

    xas_params = _xas_params(params)
    if not _carrier_plan_reuse_enabled():
        return _imrphenomxas_core_torch(
            xas_params,
            _manage_remnant_cache=False,
        ), None, None
    core, carrier_plans = _imrphenomxas_core_torch(
        xas_params,
        return_carrier_plans=True,
        _manage_remnant_cache=False,
    )
    phase_plan, amp_plan = _carrier_plans_for_modes(
        core,
        params,
        carrier_plans,
    )
    return core, phase_plan, amp_plan


def _carrier_sequence_for_modes(params):
    """Build the XAS sequence and optionally expose its exact prepared plans."""

    xas_params = _xas_params(params)
    if not _carrier_plan_reuse_enabled():
        sequence = _imrphenomxas_sequence_samples(
            xas_params,
            _manage_remnant_cache=False,
        )
        return sequence, None, None
    sequence, carrier_plans = _imrphenomxas_sequence_samples(
        xas_params,
        return_carrier_plans=True,
        _manage_remnant_cache=False,
    )
    core = _SequenceCore(sequence.polarization)
    phase_plan, amp_plan = _carrier_plans_for_modes(
        core,
        params,
        carrier_plans,
    )
    return sequence, phase_plan, amp_plan


def imrphenomxhm_modes_native_supported(params):
    """Return whether the requested XHM modes have a native implementation."""

    if params.get("approximant") != "IMRPhenomXHM":
        return False
    modes = _requested_modes(params)
    if modes is None or any(mode not in _NATIVE_MODES for mode in modes):
        return False
    return imrphenomxas_native_supported(_xas_params(params))


def imrphenomxhm_fd_native_supported(params):
    """Return whether native polarization generation covers the request."""

    modes = _requested_modes(params)
    return bool(modes) and imrphenomxhm_modes_native_supported(params)


def imrphenomxhm_sequence_native_supported(params):
    """Return whether arbitrary-frequency XHM generation is native."""

    return imrphenomxhm_modes_native_supported(params)


def _batched_tiny_solve_tensor_supported(value, *, shape, dtype, device):
    """Accept one independent plain CPU tensor owned by this request."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == shape
        and value.dtype == dtype
        and value.device == device
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
        and not _xutils._tensor_has_forward_ad(value)
    )


def _scripted_phase_triplet_scheme_device_supported(configured, actual):
    """Match an optionally indexed TorchScheme device to an actual device."""

    return (
        type(configured) is torch.device
        and type(actual) is torch.device
        and configured.type == actual.type
        and (configured.index is None or configured.index == actual.index)
    )


def _scripted_phase_triplet_supported(
    core,
    params,
    mode_families,
    shared_inputs,
    ringdown_frequencies,
    damping_frequencies,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_phase_anchors,
    carrier_inspiral_lane,
    shared_carrier_inspiral_phase,
    mode21_amplitude_release,
    mode33_amplitude_release,
    mode44_amplitude_release,
):
    """Qualify the compiled three-row phase expression on CPU or CUDA."""

    handled_modes = ((2, 1), (3, 3), (4, 4))
    anchors_supported = carrier_phase_anchors is None or (
        type(carrier_phase_anchors) is _CarrierPhaseAnchors
        and type(carrier_phase_anchors._values) is dict
        and not carrier_phase_anchors._values
    )
    if (
        not _scripted_phase_triplet_enabled()
        or not _shared_carrier_inspiral_phase_runtime_supported()
        or torch.get_num_threads() != 1
        or type(params.get("n_batch", 1)) is not int
        or params.get("n_batch", 1) != 1
        or not set(handled_modes).issubset(mode_families)
        or any(ringdown_frequencies.get(mode) is not None for mode in handled_modes)
        or any(damping_frequencies.get(mode) is not None for mode in handled_modes)
        or carrier_coprecessing_deviations is not None
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or not anchors_supported
        # The compiled lane deliberately owns all three carrier evaluations.
        or carrier_inspiral_lane is not None
        or shared_carrier_inspiral_phase is not None
        or type(shared_inputs) is not _SharedModeInputs
        or type(shared_inputs.state) is not _Mode21State
        or not _shared_mode_inputs_plain_request_supported(params)
        or tuple(
            type(value) is int and value == 122022
            for value in (
                mode21_amplitude_release,
                mode33_amplitude_release,
                mode44_amplitude_release,
            )
        ) != (True, True, True)
    ):
        return False

    state = shared_inputs.state
    if (
        (state.mass1 == state.mass2 and state.chi1 == state.chi2)
        or not all(
            type(value) is float and math.isfinite(value)
            for value in vars(state).values()
        )
        or type(shared_inputs.reference_frequency) is not float
        or type(shared_inputs.coa_phase) is not float
        or not math.isfinite(shared_inputs.reference_frequency)
        or not math.isfinite(shared_inputs.coa_phase)
    ):
        return False

    polarization = getattr(core, "polarization", None)
    if type(polarization) is not torch.Tensor:
        return False
    device = polarization.device
    scheme_state = _scheme.mgr.state
    if (
        type(scheme_state) is not _scheme.TorchScheme
        or not _scripted_phase_triplet_scheme_device_supported(
            scheme_state.torch_device,
            device,
        )
        or device.type not in ("cpu", "cuda")
        or not _scripted_phase_triplet_version_supported(device)
        or not _shared_carrier_phase_plan_value_supported(
            carrier_phase_plan,
            device,
        )
        or _xutils._tree_has_autograd_untrusted(carrier_phase_plan)
        or not _batched_tiny_solve_tensor_supported(
            polarization,
            shape=shared_inputs.frequencies.shape,
            dtype=torch.complex128,
            device=device,
        )
    ):
        return False
    tensor_contracts = (
        (shared_inputs.frequencies, polarization.shape),
        (shared_inputs.mf, polarization.shape),
        (shared_inputs.intrinsic, torch.Size((4,))),
        (shared_inputs.phase_coeffs, shared_inputs.phase_coeffs.shape),
    )
    if not all(
        _batched_tiny_solve_tensor_supported(
            value,
            shape=shape,
            dtype=torch.float64,
            device=device,
        )
        for value, shape in tensor_contracts
    ):
        return False
    # Below this size the established 33/44 paths retain their dense trees;
    # longer requests may use separately-qualified region-specialized trees.
    return 0 < shared_inputs.mf.numel() < 512


def _fixed_schema_amplitude_triplet_data_plane_supported(
    core,
    params,
    shared_inputs,
    carrier_phase_plan,
):
    """Qualify the exact fixed-schema amplitude data plane on CPU."""

    if (
        not _fixed_schema_amplitude_triplet_enabled()
        or not _shared_carrier_inspiral_phase_runtime_supported()
        or torch.get_num_threads() != 1
        or type(shared_inputs) is not _SharedModeInputs
        or type(shared_inputs.state) is not _Mode21State
        or not _shared_mode_inputs_plain_request_supported(params)
    ):
        return False

    state = shared_inputs.state
    if (
        (state.mass1 == state.mass2 and state.chi1 == state.chi2)
        or not all(
            type(value) is float and math.isfinite(value)
            for value in vars(state).values()
        )
        or type(shared_inputs.reference_frequency) is not float
        or type(shared_inputs.coa_phase) is not float
        or not math.isfinite(shared_inputs.reference_frequency)
        or not math.isfinite(shared_inputs.coa_phase)
    ):
        return False

    polarization = getattr(core, "polarization", None)
    device = torch.device("cpu")
    if (
        not _shared_carrier_phase_plan_value_supported(
            carrier_phase_plan,
            device,
        )
        or _xutils._tree_has_autograd_untrusted(carrier_phase_plan)
        or not _batched_tiny_solve_tensor_supported(
            polarization,
            shape=shared_inputs.frequencies.shape,
            dtype=torch.complex128,
            device=device,
        )
    ):
        return False
    tensor_contracts = (
        (shared_inputs.frequencies, polarization.shape),
        (shared_inputs.mf, polarization.shape),
        (shared_inputs.intrinsic, torch.Size((4,))),
        (shared_inputs.phase_coeffs, shared_inputs.phase_coeffs.shape),
    )
    if not all(
        _batched_tiny_solve_tensor_supported(
            value,
            shape=shape,
            dtype=torch.float64,
            device=device,
        )
        for value, shape in tensor_contracts
    ):
        return False
    # The established 33/44 paths may use region-pruned expression trees from
    # 512 samples onward.  This executor retains their dense, exact trees.
    return 0 < shared_inputs.mf.numel() < 512


def _fixed_schema_amplitude_with_scripted_phase_supported(core):
    """Qualify amplitude-specific contracts after a strict phase proof."""

    # The immediately preceding scripted-phase proof already establishes the
    # runtime, request, scalar, tensor, phase-plan, length, and device/scheme
    # contracts for these same objects.  Keep only the independent amplitude
    # gate and its separately qualified CPU restriction here.
    polarization = getattr(core, "polarization", None)
    return (
        _fixed_schema_amplitude_triplet_enabled()
        and type(polarization) is torch.Tensor
        and polarization.device == torch.device("cpu")
    )


def _fixed_schema_amplitude_triplet_supported(
    core,
    params,
    mode_families,
    shared_inputs,
    ringdown_frequencies,
    damping_frequencies,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_phase_anchors,
    carrier_inspiral_lane,
    shared_carrier_inspiral_phase,
    mode21_amplitude_release,
    mode33_amplitude_release,
    mode44_amplitude_release,
):
    """Qualify the raw-exact fixed-schema CPU amplitude triplet."""

    if not _fixed_schema_amplitude_triplet_data_plane_supported(
        core,
        params,
        shared_inputs,
        carrier_phase_plan,
    ):
        return False

    handled_modes = ((2, 1), (3, 3), (4, 4))
    anchors_supported = carrier_phase_anchors is None or (
        type(carrier_phase_anchors) is _CarrierPhaseAnchors
        and type(carrier_phase_anchors._values) is dict
        and not carrier_phase_anchors._values
    )
    if (
        not set(handled_modes).issubset(mode_families)
        or any(ringdown_frequencies.get(mode) is not None for mode in handled_modes)
        or any(damping_frequencies.get(mode) is not None for mode in handled_modes)
        or carrier_coprecessing_deviations is not None
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or not anchors_supported
        or type(carrier_inspiral_lane) is not _CarrierInspiralLane
        or shared_carrier_inspiral_phase is not None
        or tuple(
            type(value) is int and value == 122022
            for value in (
                mode21_amplitude_release,
                mode33_amplitude_release,
                mode44_amplitude_release,
            )
        ) != (True, True, True)
    ):
        return False
    device = torch.device("cpu")
    return all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype == torch.float64
        and value.device == device
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
        and not _xutils._tensor_has_forward_ad(value)
        for value in carrier_inspiral_lane
    )


def _scripted_phase_triplet_mode_samples(
    core,
    params,
    shared_inputs,
    *,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_phase_plan,
    carrier_phase_anchors,
    use_fixed_schema_amplitudes=False,
):
    """Generate modes 21, 33, and 44 through one compiled phase lane."""

    state21 = shared_inputs.state
    state33 = _mode33_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    state44 = _mode44_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    mf = shared_inputs.mf
    with torch_context(shared_inputs.frequencies):
        phases = _evaluate_scripted_phase_triplet(
            shared_inputs,
            (state21, state33, state44),
            carrier_phase_plan,
            carrier_phase_anchors,
        )
        if phases is None:
            return None
        phase21, phase33, phase44 = phases
        amplitudes = None
        if use_fixed_schema_amplitudes:
            try:
                plans = (
                    _run_staged_solves(
                        _amplitude_21_2022_staged(
                            mf,
                            state21,
                            _fixed_schema_plan=True,
                        )
                    ),
                    _run_staged_solves(
                        _amplitude_33_2022_staged(
                            mf,
                            state33,
                            region_indices=None,
                            _fixed_schema_plan=True,
                        )
                    ),
                    _run_staged_solves(
                        _amplitude_44_2022_staged(
                            mf,
                            state44,
                            region_indices=None,
                            _fixed_schema_plan=True,
                        )
                    ),
                )
                if all(
                    type(plan) is _FixedSchemaAmplitudePlan for plan in plans
                ):
                    amplitudes = _evaluate_fixed_schema_amplitude_triplet(
                        mf,
                        *plans,
                    )
            except Exception:
                # The phase executor already succeeded.  Keep its output and
                # fail closed only to the established eager amplitude trees.
                amplitudes = None
        if amplitudes is None:
            amplitude21 = _run_staged_solves(
                _amplitude_21_2022_staged(mf, state21)
            )
            amplitude33 = _run_staged_solves(
                _amplitude_33_2022_staged(mf, state33)
            )
            amplitude44 = _run_staged_solves(
                _amplitude_44_2022_staged(mf, state44)
            )
        else:
            amplitude21, amplitude33, amplitude44 = amplitudes
        samples21 = state21.amp0 * amplitude21 * torch.exp(1j * phase21)
        samples33 = -state33.amp0 * amplitude33 * torch.exp(1j * phase33)
        samples44 = state44.amp0 * amplitude44 * torch.exp(1j * phase44)
    dtype = core.polarization.dtype
    return {
        (2, 1): samples21.to(dtype),
        (3, 3): samples33.to(dtype),
        (4, 4): samples44.to(dtype),
    }


def _fixed_schema_amplitude_triplet_mode_samples(
    core,
    params,
    shared_inputs,
    *,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_phase_plan,
    carrier_phase_anchors,
    carrier_inspiral_lane,
):
    """Generate modes 21, 33, and 44 through one exact amplitude lane."""

    state21 = shared_inputs.state
    state33 = _mode33_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    state44 = _mode44_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    mf = shared_inputs.mf
    phase_args = (
        mf,
        shared_inputs.intrinsic,
        shared_inputs.phase_coeffs,
        shared_inputs.reference_frequency,
        shared_inputs.coa_phase,
    )
    with torch_context(shared_inputs.frequencies):
        phase21 = _run_staged_solves(
            _phase_21_staged(
                phase_args[0],
                state21,
                *phase_args[1:],
                None,
                carrier_phase_plan,
                carrier_phase_anchors,
                carrier_inspiral_lane.mode21,
            )
        )
        phase33 = _run_staged_solves(
            _phase_33_staged(
                phase_args[0],
                state33,
                *phase_args[1:],
                None,
                carrier_phase_plan,
                carrier_phase_anchors,
                carrier_inspiral_lane.mode33,
                None,
                None,
            )
        )
        phase44 = _run_staged_solves(
            _phase_44_staged(
                phase_args[0],
                state44,
                *phase_args[1:],
                None,
                carrier_phase_plan,
                carrier_phase_anchors,
                carrier_inspiral_lane.mode44,
                None,
                None,
            )
        )
        plan21 = _run_staged_solves(
            _amplitude_21_2022_staged(
                mf,
                state21,
                _fixed_schema_plan=True,
            )
        )
        plan33 = _run_staged_solves(
            _amplitude_33_2022_staged(
                mf,
                state33,
                region_indices=None,
                _fixed_schema_plan=True,
            )
        )
        plan44 = _run_staged_solves(
            _amplitude_44_2022_staged(
                mf,
                state44,
                region_indices=None,
                _fixed_schema_plan=True,
            )
        )
        if not all(
            type(plan) is _FixedSchemaAmplitudePlan
            for plan in (plan21, plan33, plan44)
        ):
            return None
        amplitudes = _evaluate_fixed_schema_amplitude_triplet(
            mf,
            plan21,
            plan33,
            plan44,
        )
        if amplitudes is None:
            return None
        amplitude21, amplitude33, amplitude44 = amplitudes
        samples21 = state21.amp0 * amplitude21 * torch.exp(1j * phase21)
        samples33 = -state33.amp0 * amplitude33 * torch.exp(1j * phase33)
        samples44 = state44.amp0 * amplitude44 * torch.exp(1j * phase44)
    dtype = core.polarization.dtype
    return {
        (2, 1): samples21.to(dtype),
        (3, 3): samples33.to(dtype),
        (4, 4): samples44.to(dtype),
    }


def _batched_tiny_solves_supported(
    core,
    params,
    mode_families,
    shared_inputs,
    ringdown_frequencies,
    damping_frequencies,
    carrier_coprecessing_deviations,
    carrier_phase_plan,
    carrier_phase_anchors,
    carrier_inspiral_lane,
    shared_carrier_inspiral_phase,
    mode21_amplitude_release,
    mode33_amplitude_release,
    mode44_amplitude_release,
):
    """Qualify the measured raw-exact CPU-float64 plain-input lane."""

    handled_modes = ((2, 1), (3, 3), (4, 4))
    if (
        not _batched_tiny_solves_enabled()
        or not _shared_carrier_inspiral_phase_runtime_supported()
        or not set(handled_modes).issubset(mode_families)
        or any(ringdown_frequencies.get(mode) is not None for mode in handled_modes)
        or any(damping_frequencies.get(mode) is not None for mode in handled_modes)
        or carrier_coprecessing_deviations is not None
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or type(carrier_phase_anchors) is not _CarrierPhaseAnchors
        or type(carrier_inspiral_lane) is not _CarrierInspiralLane
        or shared_carrier_inspiral_phase is not None
        or type(shared_inputs) is not _SharedModeInputs
        or type(shared_inputs.state) is not _Mode21State
        or not _shared_mode_inputs_plain_request_supported(params)
        or tuple(
            type(value) is int and value == 122022
            for value in (
                mode21_amplitude_release,
                mode33_amplitude_release,
                mode44_amplitude_release,
            )
        ) != (True, True, True)
    ):
        return False

    state = shared_inputs.state
    if (
        state.mass1 == state.mass2 and state.chi1 == state.chi2
    ) or not all(type(value) is float for value in vars(state).values()):
        return False
    if not (
        type(shared_inputs.reference_frequency) is float
        and type(shared_inputs.coa_phase) is float
    ):
        return False

    polarization = getattr(core, "polarization", None)
    device = torch.device("cpu")
    if (
        type(carrier_phase_anchors._values) is not dict
        or carrier_phase_anchors._values
        or not _shared_carrier_phase_plan_value_supported(
            carrier_phase_plan,
            device,
        )
        or _xutils._tree_has_autograd_untrusted(carrier_phase_plan)
        or not _batched_tiny_solve_tensor_supported(
            polarization,
            shape=shared_inputs.frequencies.shape,
            dtype=torch.complex128,
            device=device,
        )
    ):
        return False
    tensor_contracts = (
        (shared_inputs.frequencies, polarization.shape),
        (shared_inputs.mf, polarization.shape),
        (shared_inputs.intrinsic, torch.Size((4,))),
        (shared_inputs.phase_coeffs, shared_inputs.phase_coeffs.shape),
    )
    if not all(
        _batched_tiny_solve_tensor_supported(
            value,
            shape=shape,
            dtype=torch.float64,
            device=device,
        )
        for value, shape in tensor_contracts
    ):
        return False
    return all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype == torch.float64
        and value.device == device
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
        and not _xutils._tensor_has_forward_ad(value)
        for value in carrier_inspiral_lane
    )


def _batched_tiny_solve(requests):
    """Materialize and solve one same-shaped group at its final batch shape."""

    like = requests[0][2]
    matrices = _tensor(tuple(request[0] for request in requests), like)
    grouped_values = tuple(tuple(request[1]) for request in requests)
    if all(
        not isinstance(value, torch.Tensor)
        for values in grouped_values
        for value in values
    ):
        right_hand_sides = _tensor(grouped_values, like)
    else:
        right_hand_sides = torch.stack(
            [
                torch.stack(
                    [
                        value
                        if isinstance(value, torch.Tensor)
                        else _tensor(value, like)
                        for value in values
                    ]
                )
                for values in grouped_values
            ]
        )
    return tuple(
        solution.clone()
        for solution in torch.linalg.solve(
            matrices,
            right_hand_sides,
        ).unbind()
    )


def _next_staged_request(generator, solution):
    """Resume a two-solve amplitude and require its second request."""

    try:
        return generator.send(solution)
    except StopIteration as stopped:
        raise RuntimeError("staged amplitude ended before its second solve") from stopped


def _finish_staged_calculation(generator, solution):
    """Resume a staged calculation and return its completed tensor."""

    try:
        generator.send(solution)
    except StopIteration as stopped:
        return stopped.value
    raise RuntimeError("staged calculation requested an unexpected extra solve")


def _batched_tiny_solve_mode_samples(
    core,
    params,
    shared_inputs,
    *,
    final_spin,
    carrier_ringdown_frequency,
    carrier_damping_frequency,
    carrier_phase_plan,
    carrier_phase_anchors,
    carrier_inspiral_lane,
):
    """Generate modes 21, 33, and 44 with nine tiny solves in three batches."""

    state21 = shared_inputs.state
    state33 = _mode33_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    state44 = _mode44_state(
        params,
        final_spin=final_spin,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=state21,
    )
    mf = shared_inputs.mf
    phase_args = (
        mf,
        shared_inputs.intrinsic,
        shared_inputs.phase_coeffs,
        shared_inputs.reference_frequency,
        shared_inputs.coa_phase,
    )
    with torch_context(shared_inputs.frequencies):
        phase21 = _phase_21_staged(
            phase_args[0],
            state21,
            *phase_args[1:],
            None,
            carrier_phase_plan,
            carrier_phase_anchors,
            carrier_inspiral_lane.mode21,
        )
        phase33 = _phase_33_staged(
            phase_args[0],
            state33,
            *phase_args[1:],
            None,
            carrier_phase_plan,
            carrier_phase_anchors,
            carrier_inspiral_lane.mode33,
            None,
        )
        phase44 = _phase_44_staged(
            phase_args[0],
            state44,
            *phase_args[1:],
            None,
            carrier_phase_plan,
            carrier_phase_anchors,
            carrier_inspiral_lane.mode44,
            None,
        )
        amplitude33 = _amplitude_33_2022_staged(mf, state33)
        amplitude44 = _amplitude_44_2022_staged(mf, state44)
        amplitude21 = _amplitude_21_2022_staged(mf, state21)

        phase_requests = (next(phase21), next(phase33), next(phase44))
        amplitude_requests = (
            next(amplitude33),
            next(amplitude44),
            next(amplitude21),
        )
        amplitude_first = _batched_tiny_solve(amplitude_requests)
        amplitude33_request = _next_staged_request(
            amplitude33,
            amplitude_first[0],
        )
        amplitude44_request = _next_staged_request(
            amplitude44,
            amplitude_first[1],
        )
        amplitude21_request = _next_staged_request(
            amplitude21,
            amplitude_first[2],
        )

        phase_and_21 = _batched_tiny_solve(
            (*phase_requests, amplitude21_request)
        )
        phase21_result = _finish_staged_calculation(phase21, phase_and_21[0])
        phase33_result = _finish_staged_calculation(phase33, phase_and_21[1])
        phase44_result = _finish_staged_calculation(phase44, phase_and_21[2])
        amplitude21_result = _finish_staged_calculation(
            amplitude21,
            phase_and_21[3],
        )

        amplitude_last = _batched_tiny_solve(
            (amplitude33_request, amplitude44_request)
        )
        amplitude33_result = _finish_staged_calculation(
            amplitude33,
            amplitude_last[0],
        )
        amplitude44_result = _finish_staged_calculation(
            amplitude44,
            amplitude_last[1],
        )

        samples21 = state21.amp0 * amplitude21_result * torch.exp(
            1j * phase21_result
        )
        samples33 = -state33.amp0 * amplitude33_result * torch.exp(
            1j * phase33_result
        )
        samples44 = state44.amp0 * amplitude44_result * torch.exp(
            1j * phase44_result
        )
    dtype = core.polarization.dtype
    return {
        (2, 1): samples21.to(dtype),
        (3, 3): samples33.to(dtype),
        (4, 4): samples44.to(dtype),
    }


def _active_mode_samples_serial_impl(
    core,
    params,
    modes,
    *,
    frequencies=None,
    reference_frequency=None,
    final_spin=None,
    remnant=None,
    ringdown_frequencies=None,
    damping_frequencies=None,
    carrier_ringdown_frequency=None,
    carrier_damping_frequency=None,
    carrier_coprecessing_deviations=None,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    carrier_alignment_result=None,
    mode21_amplitude_release=122022,
    mode32_amplitude_release=122022,
    mode33_amplitude_release=122022,
    mode44_amplitude_release=122022,
    _uniform_grid_metadata=None,
):
    """Generate each requested absolute-m mode family once."""

    active_modes = {}
    if ringdown_frequencies is None:
        ringdown_frequencies = {}
    if damping_frequencies is None:
        damping_frequencies = {}
    mode_families = {(ell, abs(emm)) for ell, emm in modes}
    phase_anchor_autograd = _NO_PHASE_ANCHOR_AUTOGRAD_PROOF
    qualified_alignment_result = None
    if _carrier_alignment_result_supported(
        carrier_alignment_result,
        core,
        params,
        frequencies,
        reference_frequency,
        final_spin,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
    ):
        phase_anchor_autograd = _phase_anchor_inputs_have_autograd(
            core,
            params,
            final_spin,
            carrier_coprecessing_deviations,
            carrier_phase_plan,
        )
        if not phase_anchor_autograd:
            qualified_alignment_result = carrier_alignment_result
    legacy_shared_inputs = None
    triplet_shared_inputs = None
    prepared_shared_inputs = None
    carrier_inspiral_lane = None
    shared_carrier_inspiral_phase = None
    packed_carrier_inspiral_supported = (
        (2, 1) not in ringdown_frequencies
        and (2, 1) not in damping_frequencies
        and _carrier_inspiral_lane_supported(
            core,
            params,
            mode_families,
            frequencies,
            final_spin,
            carrier_coprecessing_deviations,
            carrier_phase_plan,
            _phase_anchor_autograd=phase_anchor_autograd,
        )
    )
    legacy_shared_inputs_supported = _shared_mode_inputs_supported(
        core,
        params,
        mode_families,
        frequencies,
        ringdown_frequencies,
        damping_frequencies,
        reference_frequency,
        final_spin,
        carrier_ringdown_frequency,
        carrier_damping_frequency,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_amp_plan,
    )
    triplet_shared_inputs_supported = (
        _scripted_phase_triplet_shared_inputs_supported(
            core,
            params,
            mode_families,
            frequencies,
            ringdown_frequencies,
            damping_frequencies,
            reference_frequency,
            final_spin,
            carrier_ringdown_frequency,
            carrier_damping_frequency,
            carrier_coprecessing_deviations,
            carrier_phase_plan,
            carrier_amp_plan,
            mode21_amplitude_release,
            mode33_amplitude_release,
            mode44_amplitude_release,
        )
    )
    if (
        packed_carrier_inspiral_supported
        or legacy_shared_inputs_supported
        or triplet_shared_inputs_supported
    ):
        phase_only_preparation = (
            triplet_shared_inputs_supported
            and not packed_carrier_inspiral_supported
            and not legacy_shared_inputs_supported
        )
        try:
            shared_frequencies = frequencies
            if shared_frequencies is None:
                shared_frequencies = (
                    torch.arange(
                        core.first_bin,
                        core.stop_bin,
                        device=core.polarization.device,
                        dtype=core.polarization.real.dtype,
                    )
                    * core.delta_f
                )
            prepared_shared_inputs = _prepare_shared_mode_inputs(
                params,
                shared_frequencies,
                reference_frequency=reference_frequency,
                final_spin=final_spin,
                _remnant=remnant,
                carrier_ringdown_frequency=carrier_ringdown_frequency,
                carrier_damping_frequency=carrier_damping_frequency,
            )
        except Exception:
            if not phase_only_preparation:
                raise
            triplet_shared_inputs_supported = False
        else:
            if legacy_shared_inputs_supported or packed_carrier_inspiral_supported:
                legacy_shared_inputs = prepared_shared_inputs
            if triplet_shared_inputs_supported:
                triplet_shared_inputs = prepared_shared_inputs
            if packed_carrier_inspiral_supported:
                carrier_inspiral_lane = _prepare_carrier_inspiral_lane(
                    prepared_shared_inputs,
                    carrier_phase_plan,
                )
    if _shared_carrier_inspiral_phase_supported(
        core,
        params,
        mode_families,
        legacy_shared_inputs,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
    ):
        shared_carrier_inspiral_phase = (
            _prepare_shared_carrier_inspiral_phase(
                legacy_shared_inputs,
                carrier_phase_plan,
            )
        )
    carrier_phase_anchors = None
    if phase_anchor_autograd is _NO_PHASE_ANCHOR_AUTOGRAD_PROOF:
        if (
            _phase_anchor_cache_enabled()
            and _phase_anchor_cache_supported(core)
            and _plain_request_runtime_supported()
            and not _phase_anchor_inputs_have_autograd(
                core,
                params,
                final_spin,
                carrier_coprecessing_deviations,
                carrier_phase_plan,
            )
        ):
            carrier_phase_anchors = _CarrierPhaseAnchors()
    elif not phase_anchor_autograd:
        carrier_phase_anchors = _CarrierPhaseAnchors(
            qualified_alignment_result
        )
    triplet_modes = None
    scripted_phase_triplet_supported = _scripted_phase_triplet_supported(
        core,
        params,
        mode_families,
        triplet_shared_inputs,
        ringdown_frequencies,
        damping_frequencies,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_phase_anchors,
        carrier_inspiral_lane,
        shared_carrier_inspiral_phase,
        mode21_amplitude_release,
        mode33_amplitude_release,
        mode44_amplitude_release,
    )
    if scripted_phase_triplet_supported:
        use_fixed_schema_amplitudes = (
            _fixed_schema_amplitude_with_scripted_phase_supported(core)
        )
        triplet_modes = _scripted_phase_triplet_mode_samples(
            core,
            params,
            triplet_shared_inputs,
            final_spin=final_spin,
            carrier_ringdown_frequency=carrier_ringdown_frequency,
            carrier_damping_frequency=carrier_damping_frequency,
            carrier_phase_plan=carrier_phase_plan,
            carrier_phase_anchors=carrier_phase_anchors,
            use_fixed_schema_amplitudes=use_fixed_schema_amplitudes,
        )
    if triplet_modes is None and _fixed_schema_amplitude_triplet_supported(
        core,
        params,
        mode_families,
        legacy_shared_inputs,
        ringdown_frequencies,
        damping_frequencies,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_phase_anchors,
        carrier_inspiral_lane,
        shared_carrier_inspiral_phase,
        mode21_amplitude_release,
        mode33_amplitude_release,
        mode44_amplitude_release,
    ):
        triplet_modes = _fixed_schema_amplitude_triplet_mode_samples(
            core,
            params,
            legacy_shared_inputs,
            final_spin=final_spin,
            carrier_ringdown_frequency=carrier_ringdown_frequency,
            carrier_damping_frequency=carrier_damping_frequency,
            carrier_phase_plan=carrier_phase_plan,
            carrier_phase_anchors=carrier_phase_anchors,
            carrier_inspiral_lane=carrier_inspiral_lane,
        )
    if triplet_modes is None and _batched_tiny_solves_supported(
        core,
        params,
        mode_families,
        legacy_shared_inputs,
        ringdown_frequencies,
        damping_frequencies,
        carrier_coprecessing_deviations,
        carrier_phase_plan,
        carrier_phase_anchors,
        carrier_inspiral_lane,
        shared_carrier_inspiral_phase,
        mode21_amplitude_release,
        mode33_amplitude_release,
        mode44_amplitude_release,
    ):
        triplet_modes = _batched_tiny_solve_mode_samples(
            core,
            params,
            legacy_shared_inputs,
            final_spin=final_spin,
            carrier_ringdown_frequency=carrier_ringdown_frequency,
            carrier_damping_frequency=carrier_damping_frequency,
            carrier_phase_plan=carrier_phase_plan,
            carrier_phase_anchors=carrier_phase_anchors,
            carrier_inspiral_lane=carrier_inspiral_lane,
        )
    if (2, 2) in mode_families:
        active_modes[2, 2] = core.polarization / _XAS_MODE_POLARIZATION_FACTOR
    if (2, 1) in mode_families:
        if triplet_modes is not None:
            active_modes[2, 1] = triplet_modes[2, 1]
        else:
            active_modes[2, 1] = imrphenomxhm_h2m1_samples(
                core,
                params,
                frequencies=frequencies,
                reference_frequency=reference_frequency,
                final_spin=final_spin,
                _remnant=remnant,
                ringdown_frequency=ringdown_frequencies.get((2, 1)),
                damping_frequency=damping_frequencies.get((2, 1)),
                carrier_ringdown_frequency=carrier_ringdown_frequency,
                carrier_damping_frequency=carrier_damping_frequency,
                carrier_coprecessing_deviations=(
                    carrier_coprecessing_deviations
                ),
                carrier_phase_plan=carrier_phase_plan,
                carrier_phase_anchors=carrier_phase_anchors,
                amplitude_release=mode21_amplitude_release,
                _shared_mode_inputs=legacy_shared_inputs,
                _carrier_inspiral_align=(
                    None
                    if carrier_inspiral_lane is None
                    else carrier_inspiral_lane.mode21
                ),
            )
    if (3, 3) in mode_families:
        if triplet_modes is not None:
            active_modes[3, 3] = triplet_modes[3, 3]
        else:
            active_modes[3, 3] = imrphenomxhm_h3m3_samples(
                core,
                params,
                frequencies=frequencies,
                reference_frequency=reference_frequency,
                final_spin=final_spin,
                _remnant=remnant,
                ringdown_frequency=ringdown_frequencies.get((3, 3)),
                damping_frequency=damping_frequencies.get((3, 3)),
                carrier_ringdown_frequency=carrier_ringdown_frequency,
                carrier_damping_frequency=carrier_damping_frequency,
                carrier_coprecessing_deviations=(
                    carrier_coprecessing_deviations
                ),
                carrier_phase_plan=carrier_phase_plan,
                carrier_phase_anchors=carrier_phase_anchors,
                amplitude_release=mode33_amplitude_release,
                _shared_mode_inputs=legacy_shared_inputs,
                _carrier_inspiral_align=(
                    None
                    if carrier_inspiral_lane is None
                    else carrier_inspiral_lane.mode33
                ),
                _shared_carrier_inspiral_phase=(
                    None
                    if shared_carrier_inspiral_phase is None
                    else shared_carrier_inspiral_phase.mode33
                ),
                _uniform_grid_metadata=_uniform_grid_metadata,
            )
    if (3, 2) in mode_families:
        active_modes[3, 2] = imrphenomxhm_h3m2_samples(
            core,
            params,
            frequencies=frequencies,
            reference_frequency=reference_frequency,
            final_spin=final_spin,
            _remnant=remnant,
            ringdown_frequency=ringdown_frequencies.get((3, 2)),
            damping_frequency=damping_frequencies.get((3, 2)),
            carrier_ringdown_frequency=carrier_ringdown_frequency,
            carrier_damping_frequency=carrier_damping_frequency,
            carrier_coprecessing_deviations=(
                carrier_coprecessing_deviations
            ),
            carrier_phase_plan=carrier_phase_plan,
            carrier_amp_plan=carrier_amp_plan,
            carrier_phase_anchors=carrier_phase_anchors,
            amplitude_release=mode32_amplitude_release,
            _shared_mode_inputs=legacy_shared_inputs,
            _carrier_inspiral_align=(
                None
                if carrier_inspiral_lane is None
                else carrier_inspiral_lane.mode32
            ),
            _shared_carrier_inspiral_phase=(
                None
                if shared_carrier_inspiral_phase is None
                else shared_carrier_inspiral_phase.mode32
            ),
            _uniform_grid_metadata=_uniform_grid_metadata,
        )
    if (4, 4) in mode_families:
        if triplet_modes is not None:
            active_modes[4, 4] = triplet_modes[4, 4]
        else:
            active_modes[4, 4] = imrphenomxhm_h4m4_samples(
                core,
                params,
                frequencies=frequencies,
                reference_frequency=reference_frequency,
                final_spin=final_spin,
                _remnant=remnant,
                ringdown_frequency=ringdown_frequencies.get((4, 4)),
                damping_frequency=damping_frequencies.get((4, 4)),
                carrier_ringdown_frequency=carrier_ringdown_frequency,
                carrier_damping_frequency=carrier_damping_frequency,
                carrier_coprecessing_deviations=(
                    carrier_coprecessing_deviations
                ),
                carrier_phase_plan=carrier_phase_plan,
                carrier_phase_anchors=carrier_phase_anchors,
                amplitude_release=mode44_amplitude_release,
                _shared_mode_inputs=legacy_shared_inputs,
                _carrier_inspiral_align=(
                    None
                    if carrier_inspiral_lane is None
                    else carrier_inspiral_lane.mode44
                ),
                _shared_carrier_inspiral_phase=(
                    None
                    if shared_carrier_inspiral_phase is None
                    else shared_carrier_inspiral_phase.mode44
                ),
                _uniform_grid_metadata=_uniform_grid_metadata,
            )
    return active_modes


def _active_mode_samples_serial(
    core,
    params,
    modes,
    *,
    frequencies=None,
    reference_frequency=None,
    final_spin=None,
    remnant=None,
    ringdown_frequencies=None,
    damping_frequencies=None,
    carrier_ringdown_frequency=None,
    carrier_damping_frequency=None,
    carrier_coprecessing_deviations=None,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    carrier_alignment_result=None,
    mode21_amplitude_release=122022,
    mode32_amplitude_release=122022,
    mode33_amplitude_release=122022,
    mode44_amplitude_release=122022,
    _uniform_grid_metadata=None,
):
    """Run the serial higher-mode implementation."""

    return _active_mode_samples_serial_impl(
        core,
        params,
        modes,
        frequencies=frequencies,
        reference_frequency=reference_frequency,
        final_spin=final_spin,
        remnant=remnant,
        ringdown_frequencies=ringdown_frequencies,
        damping_frequencies=damping_frequencies,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        carrier_coprecessing_deviations=carrier_coprecessing_deviations,
        carrier_phase_plan=carrier_phase_plan,
        carrier_amp_plan=carrier_amp_plan,
        carrier_alignment_result=carrier_alignment_result,
        mode21_amplitude_release=mode21_amplitude_release,
        mode32_amplitude_release=mode32_amplitude_release,
        mode33_amplitude_release=mode33_amplitude_release,
        mode44_amplitude_release=mode44_amplitude_release,
        _uniform_grid_metadata=_uniform_grid_metadata,
    )


_PARALLEL_MODE_SEQUENCE = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
_PARALLEL_MODE_GENERATORS = (
    ((2, 1), "imrphenomxhm_h2m1_samples", imrphenomxhm_h2m1_samples),
    ((3, 3), "imrphenomxhm_h3m3_samples", imrphenomxhm_h3m3_samples),
    ((3, 2), "imrphenomxhm_h3m2_samples", imrphenomxhm_h3m2_samples),
    ((4, 4), "imrphenomxhm_h4m4_samples", imrphenomxhm_h4m4_samples),
)


class _ParallelModeCall(NamedTuple):
    """One unchanged higher-mode generator invocation."""

    mode: tuple[int, int]
    function: object
    args: tuple
    kwargs: dict


_PARALLEL_MODE_CAPTURE = ContextVar(
    "pycbc_imrphenomxhm_parallel_mode_capture",
    default=None,
)
_PARALLEL_MODE_EXECUTOR = None
_PARALLEL_MODE_EXECUTOR_PID = None
_PARALLEL_MODE_EXECUTOR_LOCK = threading.Lock()


def _parallel_mode_tensor_supported(value, *, shape, dtype, device):
    """Accept one owned, plain, non-AD n_batch=1 CPU tensor."""

    return _batched_tiny_solve_tensor_supported(
        value,
        shape=shape,
        dtype=dtype,
        device=device,
    )


def _parallel_mode_empty_mapping(value):
    return value is None or (type(value) is dict and not value)


def _parallel_mode_request_supported(core, params, modes, **kwargs):
    """Qualify only the exact free-threaded CPU request that was measured."""

    allowed_kwargs = {
        "frequencies",
        "reference_frequency",
        "final_spin",
        "remnant",
        "ringdown_frequencies",
        "damping_frequencies",
        "carrier_ringdown_frequency",
        "carrier_damping_frequency",
        "carrier_coprecessing_deviations",
        "carrier_phase_plan",
        "carrier_amp_plan",
        "mode21_amplitude_release",
        "mode32_amplitude_release",
        "mode33_amplitude_release",
        "mode44_amplitude_release",
        "_uniform_grid_metadata",
    }
    if set(kwargs) - allowed_kwargs:
        return False

    state = _scheme.mgr.state
    if (
        not _free_threaded_cpython_runtime_supported()
        or not _shared_carrier_inspiral_phase_runtime_supported()
        or type(state) is not _scheme.TorchScheme
        or state.torch_device != torch.device("cpu")
        or type(core) is not _SequenceCore
        or type(params) is not dict
        or not _shared_mode_inputs_plain_request_supported(params)
        or type(modes) is not list
        or tuple(modes) != _PARALLEL_MODE_SEQUENCE
        or not all(
            _configured_switch_enabled(name)
            for name in _PARALLEL_MODE_EXACT_SWITCHES
        )
    ):
        return False
    try:
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            return False
        if (
            _phase_anchor_cache_enabled()
            or _batched_tiny_solves_enabled()
            or _fixed_schema_amplitude_triplet_enabled()
            or _remnant_cache_enabled()
            or _carrier_plan_reuse_enabled()
            or _shared_carrier_inspiral_phase_enabled()
            or not _shared_mode_inputs_enabled()
            # Preserve each mode's scalar carrier-anchor evaluation.  Recent
            # Torch CPU vector ``pow`` kernels are not raw-exact to scalars.
            or _carrier_inspiral_lane_enabled()
        ):
            return False
    except (AttributeError, RuntimeError):
        return False

    polarization = core.polarization
    frequencies = kwargs.get("frequencies")
    if (
        type(polarization) is not torch.Tensor
        or polarization.ndim != 1
        or polarization.dtype != torch.complex128
        or polarization.device != torch.device("cpu")
        or not _parallel_mode_tensor_supported(
            polarization,
            shape=polarization.shape,
            dtype=torch.complex128,
            device=polarization.device,
        )
        or not _parallel_mode_tensor_supported(
            frequencies,
            shape=polarization.shape,
            dtype=torch.float64,
            device=polarization.device,
        )
    ):
        return False

    reference_frequency = kwargs.get("reference_frequency")
    final_spin = kwargs.get("final_spin")
    remnant = kwargs.get("remnant")
    carrier_phase_plan = kwargs.get("carrier_phase_plan")
    carrier_amp_plan = kwargs.get("carrier_amp_plan")
    uniform_grid_metadata = kwargs.get("_uniform_grid_metadata")
    releases = tuple(
        kwargs.get(name, 122022)
        for name in (
            "mode21_amplitude_release",
            "mode32_amplitude_release",
            "mode33_amplitude_release",
            "mode44_amplitude_release",
        )
    )
    if (
        type(reference_frequency) is not float
        or not math.isfinite(reference_frequency)
        or type(final_spin) is not float
        or not math.isfinite(final_spin)
        or type(remnant) is not _xutils.IMRPhenomXRemnant
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or not _imrphenomxas_ringdown_amp_plan_type_supported(
            carrier_amp_plan
        )
        or type(uniform_grid_metadata) is not tuple
        or not _shared_mode_inputs_plain_request_supported(
            uniform_grid_metadata
        )
        or any(type(value) is not int or value != 122022 for value in releases)
        or not _parallel_mode_empty_mapping(
            kwargs.get("ringdown_frequencies")
        )
        or not _parallel_mode_empty_mapping(
            kwargs.get("damping_frequencies")
        )
        or kwargs.get("carrier_ringdown_frequency") is not None
        or kwargs.get("carrier_damping_frequency") is not None
        or kwargs.get("carrier_coprecessing_deviations") is not None
        or not _carrier_plan_value_supported(remnant, polarization)
        or not _carrier_plan_value_supported(carrier_phase_plan, polarization)
        or not _carrier_plan_value_supported(carrier_amp_plan, polarization)
        or _xutils._tree_has_autograd_untrusted(
            (
                core,
                params,
                modes,
                frequencies,
                remnant,
                carrier_phase_plan,
                carrier_amp_plan,
                uniform_grid_metadata,
            )
        )
    ):
        return False
    return True


def _parallel_mode_generator_wrapper(mode, function):
    """Capture a generator call only inside one eligible outer request."""

    def wrapped(*args, **kwargs):
        calls = _PARALLEL_MODE_CAPTURE.get()
        if calls is None:
            return function(*args, **kwargs)
        call = _ParallelModeCall(mode, function, args, kwargs)
        calls.append(call)
        return call

    wrapped.__name__ = function.__name__
    wrapped.__doc__ = function.__doc__
    return wrapped


def _install_parallel_mode_generator_wrappers():
    """Install context-local capture hooks after the import-time gate passes."""

    for mode, name, function in _PARALLEL_MODE_GENERATORS:
        globals()[name] = _parallel_mode_generator_wrapper(mode, function)


def _parallel_mode_executor_after_fork():
    """Forget parent-owned threads and locks in a fork child."""

    global _PARALLEL_MODE_EXECUTOR
    global _PARALLEL_MODE_EXECUTOR_PID
    global _PARALLEL_MODE_EXECUTOR_LOCK
    _PARALLEL_MODE_EXECUTOR = None
    _PARALLEL_MODE_EXECUTOR_PID = None
    _PARALLEL_MODE_EXECUTOR_LOCK = threading.Lock()
    _PARALLEL_MODE_CAPTURE.set(None)


def _parallel_mode_executor_get():
    """Return the lazy process-local four-worker executor."""

    global _PARALLEL_MODE_EXECUTOR
    global _PARALLEL_MODE_EXECUTOR_PID
    pid = os.getpid()
    if _PARALLEL_MODE_EXECUTOR_PID not in (None, pid):
        _parallel_mode_executor_after_fork()
    with _PARALLEL_MODE_EXECUTOR_LOCK:
        if _PARALLEL_MODE_EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor

            _PARALLEL_MODE_EXECUTOR = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="pycbc-xhm-mode",
            )
            _PARALLEL_MODE_EXECUTOR_PID = pid
        return _PARALLEL_MODE_EXECUTOR


def _parallel_mode_executor_discard(executor):
    """Retire a failed pool only after every submitted task is quiescent."""

    global _PARALLEL_MODE_EXECUTOR
    global _PARALLEL_MODE_EXECUTOR_PID
    with _PARALLEL_MODE_EXECUTOR_LOCK:
        if _PARALLEL_MODE_EXECUTOR is executor:
            _PARALLEL_MODE_EXECUTOR = None
            _PARALLEL_MODE_EXECUTOR_PID = None
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        pass


def _parallel_mode_worker_body(call):
    """Reconstruct only the two immutable promises needed by mode kernels."""

    with torch.inference_mode(), _xutils.trusted_plain_request_context(
        enabled=True
    ):
        return call.function(*call.args, **call.kwargs)


def _parallel_mode_worker(call):
    """Root every task in an empty context, even on Python 3.14t."""

    return Context().run(_parallel_mode_worker_body, call)


def _parallel_mode_capture_valid(shell, calls):
    expected_calls = _PARALLEL_MODE_SEQUENCE[1:]
    return (
        type(shell) is dict
        and tuple(shell) == _PARALLEL_MODE_SEQUENCE
        and tuple(call.mode for call in calls) == expected_calls
        and all(shell.get(call.mode) is call for call in calls)
    )


def _parallel_mode_finish_serial(shell, calls):
    """Complete a captured request synchronously after a pool failure."""

    call_by_mode = {call.mode: call for call in calls}
    result = {}
    for mode, samples in shell.items():
        call = call_by_mode.get(mode)
        result[mode] = (
            call.function(*call.args, **call.kwargs)
            if call is not None and samples is call
            else samples
        )
    return result


def _parallel_mode_drain(futures):
    for future in futures:
        future.cancel()
    for future in futures:
        try:
            future.result()
        except BaseException:
            pass


def _active_mode_samples_parallel(*args, **kwargs):
    """Generate independent XHM modes concurrently on qualified 3.14t CPU."""

    carrier_phase_plan = kwargs.get("carrier_phase_plan")
    plain_phase_plan = _request_unqualify_top_plan(carrier_phase_plan)
    if plain_phase_plan is not carrier_phase_plan:
        # Request proof markers are synchronous and thread-affine. Preserve
        # the historical same-arity Scripted/Prequalified plan schemas before
        # any call can be captured for a fresh-context worker.
        kwargs = dict(kwargs)
        kwargs["carrier_phase_plan"] = plain_phase_plan

    if (
        len(args) != 3
        or not _parallel_mode_request_supported(*args, **kwargs)
    ):
        return _active_mode_samples_serial(*args, **kwargs)

    calls = []
    token = _PARALLEL_MODE_CAPTURE.set(calls)
    try:
        shell = _active_mode_samples_serial(*args, **kwargs)
    finally:
        _PARALLEL_MODE_CAPTURE.reset(token)
    calls = tuple(calls)
    if not _parallel_mode_capture_valid(shell, calls):
        return _active_mode_samples_serial(*args, **kwargs)

    futures = []
    try:
        executor = _parallel_mode_executor_get()
        for call in calls:
            futures.append(executor.submit(_parallel_mode_worker, call))
        samples_by_mode = {
            call.mode: future.result()
            for call, future in zip(calls, futures)
        }
    except BaseException as error:
        _parallel_mode_drain(futures)
        if "executor" in locals():
            _parallel_mode_executor_discard(executor)
        if not isinstance(error, Exception):
            raise
        return _parallel_mode_finish_serial(shell, calls)

    result = {}
    for mode, samples in shell.items():
        result[mode] = samples_by_mode.get(mode, samples)
    return result


_PARALLEL_MODES_CPU_ACTIVE = (
    _parallel_modes_cpu_requested()
    and _free_threaded_cpython_runtime_supported()
)
if _PARALLEL_MODES_CPU_ACTIVE:
    _install_parallel_mode_generator_wrappers()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_parallel_mode_executor_after_fork)
    _active_mode_samples = _active_mode_samples_parallel
else:
    # This identity is deliberate: disabled and unsupported builds pay no
    # request-time branch or wrapper overhead.
    _active_mode_samples = _active_mode_samples_serial


def _bulk_polarization_tensor_supported(value, *, shape, device):
    """Accept one plain, owned complex128 tensor from this request."""

    return _batched_tiny_solve_tensor_supported(
        value,
        shape=shape,
        dtype=torch.complex128,
        device=device,
    )


def _bulk_polarization_harmonics_supported(
    core,
    params,
    modes,
    active_modes,
):
    """Qualify the measured exact CPU/CUDA float64 plain-tensor lane."""

    polarization = getattr(core, "polarization", None)
    if (
        not _shared_carrier_inspiral_phase_runtime_supported()
        or type(params) is not dict
        or not _shared_mode_inputs_plain_request_supported(params)
        or type(modes) is not list
        or any(
            type(mode) is not tuple
            or len(mode) != 2
            or type(mode[0]) is not int
            or type(mode[1]) is not int
            for mode in modes
        )
        or type(active_modes) is not dict
        or type(polarization) is not torch.Tensor
        or polarization.device.type not in ("cpu", "cuda")
        or polarization.dtype != torch.complex128
    ):
        return False
    device = polarization.device
    shape = polarization.shape
    if not _bulk_polarization_tensor_supported(
        polarization,
        shape=shape,
        device=device,
    ):
        return False
    for mode, samples in active_modes.items():
        if (
            type(mode) is not tuple
            or len(mode) != 2
            or type(mode[0]) is not int
            or type(mode[1]) is not int
            or mode not in _NATIVE_MODE_FAMILIES
            or not _bulk_polarization_tensor_supported(
                samples,
                shape=shape,
                device=device,
            )
        ):
            return False
    return True


def _request_bulk_polarization_harmonics(
    core,
    params,
    modes,
    active_modes,
    selected,
    inclination,
):
    """Return selected shared-angle harmonics, or ``None`` to fall back."""

    if (
        not _bulk_polarization_harmonics_enabled()
        or not math.isfinite(inclination)
        or not _bulk_polarization_harmonics_supported(
            core,
            params,
            modes,
            active_modes,
        )
    ):
        return None
    requested = tuple(
        mode
        for ell, emm in active_modes
        for mode in ((ell, -emm), (ell, emm))
        if mode in selected
    )
    # Sharing nine powers breaks even at three requested harmonics.  One or
    # two modes retain the lower-dispatch independent scalar implementation.
    if len(requested) < 3:
        return None
    polarization = core.polarization
    return selected_spin_minus_two_spherical_harmonics(
        inclination,
        math.pi / 2.0,
        requested,
        dtype=polarization.real.dtype,
        device=polarization.device,
    )


def _polarizations_from_active_modes(
    core,
    params,
    modes,
    active_modes,
    *,
    sequence=False,
):
    """Assemble requested mode samples into plus and cross polarizations."""

    plus = core.polarization.new_zeros(core.polarization.shape)
    cross = core.polarization.new_zeros(core.polarization.shape)

    # XHM's aligned-spin convention evaluates the spin-weighted spherical
    # harmonics at phi=pi/2.  The generated positive-frequency waveform is
    # h_l,-m; an explicitly selected +m contribution is reconstructed from
    # equatorial symmetry with the same samples.
    selected = set(modes)
    inclination = float(params.get("inclination", 0.0))
    real_dtype = plus.real.dtype
    device = plus.device
    bulk_harmonics = _request_bulk_polarization_harmonics(
        core,
        params,
        modes,
        active_modes,
        selected,
        inclination,
    )
    for (ell, emm), samples in active_modes.items():
        parity = (-1) ** ell
        factor_plus = plus.new_zeros(())
        factor_cross = plus.new_zeros(())
        if (ell, -emm) in selected:
            y_negative = (
                spin_weighted_spherical_harmonic(
                    inclination,
                    math.pi / 2.0,
                    -2,
                    ell,
                    -emm,
                    dtype=real_dtype,
                    device=device,
                )
                if bulk_harmonics is None
                else bulk_harmonics[ell, -emm]
            )
            factor_plus += 0.5 * y_negative
            factor_cross += 0.5j * y_negative
        if (ell, emm) in selected:
            y_positive_conjugate = (
                spin_weighted_spherical_harmonic(
                    inclination,
                    math.pi / 2.0,
                    -2,
                    ell,
                    emm,
                    dtype=real_dtype,
                    device=device,
                )
                if bulk_harmonics is None
                else bulk_harmonics[ell, emm]
            ).conj()
            factor_plus += 0.5 * parity * y_positive_conjugate
            factor_cross -= 0.5j * parity * y_positive_conjugate
        plus += factor_plus * samples
        cross += factor_cross * samples

    # SimInspiralChooseFDWaveformSequence has no ascending-node argument.
    long_asc_nodes = 0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    cos_nodes = math.cos(2.0 * long_asc_nodes)
    sin_nodes = math.sin(2.0 * long_asc_nodes)
    return (
        cos_nodes * plus + sin_nodes * cross,
        cos_nodes * cross - sin_nodes * plus,
    )


def imrphenomxhm_modes_torch(**params):
    """Generate the requested native XHM modes with Torch."""

    return _run_scoped_xhm_request(params, _imrphenomxhm_modes_torch)


def _imrphenomxhm_modes_torch(**params):
    """Generate the requested native XHM modes with Torch."""

    if not imrphenomxhm_modes_native_supported(params):
        raise ValueError(
            "only the default IMRPhenomXHM mode set or explicit (2, +/-1), "
            "(2, +/-2), (3, +/-2), (3, +/-3), and (4, +/-4) requests are "
            "supported by the native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM modes require TorchScheme")

    modes = _requested_modes(params)
    if not modes:
        return {}

    core, carrier_phase_plan, carrier_amp_plan = _carrier_core_for_modes(
        params
    )
    active_modes = _active_mode_samples(
        core,
        params,
        modes,
        carrier_phase_plan=carrier_phase_plan,
        carrier_amp_plan=carrier_amp_plan,
    )

    result = {}
    for ell, emm in modes:
        # LAL exposes the negative-m mode at positive frequencies. Positive-m
        # modes follow h_lm = (-1)^ell conjugate(h_l,-m).
        samples = active_modes[ell, abs(emm)]
        if emm > 0:
            samples = samples.conj()
            if ell % 2:
                samples = -samples
        hlm = _series_from_active_samples(core, samples)
        hplus = 0.5 * hlm
        hcross = (0.5j if emm < 0 else -0.5j) * hlm
        result[ell, emm] = (hplus, hcross)
    return result


def imrphenomxhm_fd_torch(**params):
    """Generate native IMRPhenomXHM plus and cross polarizations with Torch."""

    return _run_scoped_xhm_request(params, _imrphenomxhm_fd_torch)


def _imrphenomxhm_fd_torch(**params):
    """Generate native IMRPhenomXHM plus and cross polarizations with Torch."""

    if not imrphenomxhm_fd_native_supported(params):
        raise ValueError("unsupported parameters for native Torch IMRPhenomXHM")
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM requires TorchScheme")

    modes = _requested_modes(params)
    core, carrier_phase_plan, carrier_amp_plan = _carrier_core_for_modes(
        params
    )
    active_modes = _active_mode_samples(
        core,
        params,
        modes,
        carrier_phase_plan=carrier_phase_plan,
        carrier_amp_plan=carrier_amp_plan,
    )
    plus, cross = _polarizations_from_active_modes(
        core,
        params,
        modes,
        active_modes,
    )
    return (
        _series_from_active_samples(core, plus),
        _series_from_active_samples(core, cross),
    )


def imrphenomxhm_fd_sequence_torch(**params):
    """Evaluate IMRPhenomXHM polarizations at arbitrary frequencies."""

    return _run_scoped_xhm_request(params, _imrphenomxhm_fd_sequence_torch)


def _imrphenomxhm_fd_sequence_torch(**params):
    """Evaluate IMRPhenomXHM polarizations at arbitrary frequencies."""

    if not imrphenomxhm_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomXHM sequence parameters are not supported by the "
            "native Torch path"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXHM requires TorchScheme")

    modes = _requested_modes(params)
    sequence, carrier_phase_plan, carrier_amp_plan = (
        _carrier_sequence_for_modes(params)
    )
    core = _SequenceCore(sequence.polarization)
    active_modes = _active_mode_samples(
        core,
        params,
        modes,
        frequencies=sequence.frequencies,
        reference_frequency=sequence.reference_frequency,
        carrier_phase_plan=carrier_phase_plan,
        carrier_amp_plan=carrier_amp_plan,
    )
    plus, cross = _polarizations_from_active_modes(
        core,
        params,
        modes,
        active_modes,
        sequence=True,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def _imrphenomxhm_modes_batch(
    m1,
    m2,
    s1z,
    s2z,
    dist,
    coa_phase,
    f_ref,
    frequencies,
    delta_f,
    f_lower,
    f_final,
    *,
    final_spin_batch=None,
    chip_batch=None,
    modes=None,
):
    """Evaluate active IMRPhenomXHM carrier modes as batch tensors (B, N_f)."""
    batch_size = m1.shape[0]
    device = frequencies.device
    real_dtype = frequencies.dtype

    if modes is None:
        modes = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    mode_set = set(modes)

    frequencies_2d = frequencies.unsqueeze(0)

    # 1. Mode (2, 2)
    mode_dict = {}
    if (2, 2) in mode_set:
        phase_coeffs = _xutils._get_phenomx_phase_coeff_table_cached_master(
            device=device, dtype=real_dtype
        )
        amp_coeffs = _xutils._get_phenomx_amp_coeff_table_cached_master(
            device=device, dtype=real_dtype
        )
        packed_plans = _build_packed_frequency_plans_batch(
            m1,
            m2,
            s1z,
            s2z,
            dist,
            coa_phase,
            f_ref,
            f_lower,
            phase_coeffs,
            amp_coeffs,
            chip=chip_batch,
            final_spin=final_spin_batch,
        )

        total_mass_seconds_col = packed_plans[:, 0:1]
        eta_col = packed_plans[:, 1:2]
        phase_lower_col = packed_plans[:, 2:3]
        phase_upper_col = packed_plans[:, 3:4]
        phase_inspiral = packed_plans[:, 4:20]
        phase_intermediate = packed_plans[:, 20:28]
        phase_ringdown = packed_plans[:, 28:37]
        alpha0_col = packed_plans[:, 37:38]
        alpha1_col = packed_plans[:, 38:39]
        beta0_col = packed_plans[:, 39:40]
        beta1_col = packed_plans[:, 40:41]
        amp_inspiral = packed_plans[:, 41:50]
        amp_intermediate = packed_plans[:, 50:55]
        amp_ringdown = packed_plans[:, 55:60]
        linear_a_col = packed_plans[:, 60:61]
        linear_b_col = packed_plans[:, 61:62]
        phase_at_ref_col = packed_plans[:, 62:63]
        time_shift_col = packed_plans[:, 63:64]
        overall_amp_col = packed_plans[:, 64:65]
        amp_match_col = packed_plans[:, 65:66]

        Mf = total_mass_seconds_col * frequencies_2d

        f13 = Mf ** (1.0 / 3.0)
        f23 = f13 * f13
        f43 = Mf * f13
        f53 = Mf * f23
        f2 = Mf * Mf
        f73 = f2 * f13
        f83 = f2 * f23
        f3 = f2 * Mf
        f103 = f3 * f13
        f113 = f3 * f23
        log_f = torch.log(Mf)

        phase_tf2 = (
            phase_inspiral[:, 0:1]
            + phase_inspiral[:, 1:2] * f13
            + phase_inspiral[:, 2:3] * f23
            + phase_inspiral[:, 3:4] * Mf
            + phase_inspiral[:, 4:5] * f43
            + phase_inspiral[:, 5:6] * f53
            + phase_inspiral[:, 6:7] * f53 * log_f
            + phase_inspiral[:, 7:8] * f2
            + phase_inspiral[:, 8:9] * f2 * log_f
            + phase_inspiral[:, 9:10] * f73
            + phase_inspiral[:, 10:11] * f83
            + phase_inspiral[:, 11:12] * f83 * log_f
        )
        phase_inspiral_val = phase_tf2 + (
            phase_inspiral[:, 12:13] * f83
            + phase_inspiral[:, 13:14] * f3
            + phase_inspiral[:, 14:15] * f103
            + phase_inspiral[:, 15:16] * f113
        )
        normalization = -(3.0 * math.pi ** (-5.0 / 3.0)) / 128.0
        phase_ins = phase_inspiral_val * normalization / f53

        b0 = phase_intermediate[:, 0:1]
        b1 = phase_intermediate[:, 1:2]
        b2 = phase_intermediate[:, 2:3]
        b3 = phase_intermediate[:, 3:4]
        b4 = phase_intermediate[:, 4:5]
        c_l_int = phase_intermediate[:, 5:6]
        f_rd_int = phase_intermediate[:, 6:7]
        f_damp_int = phase_intermediate[:, 7:8]
        phase_int_raw = (
            b0 * Mf
            + b1 * torch.log(Mf)
            - b2 * (Mf ** -1.0)
            - b3 * (Mf ** -2.0) / 2.0
            - (b4 * (Mf ** -3.0) / 3.0)
            + (2.0 * c_l_int * torch.atan((Mf - f_rd_int) / (2.0 * f_damp_int))) / f_damp_int
        )
        phase_int = phase_int_raw + alpha1_col * Mf + alpha0_col

        c0 = phase_ringdown[:, 0:1]
        c1 = phase_ringdown[:, 1:2]
        c2 = phase_ringdown[:, 2:3]
        c4_over_3 = phase_ringdown[:, 3:4]
        c_l_over_f_damp = phase_ringdown[:, 4:5]
        f_rd_rd = phase_ringdown[:, 5:6]
        f_damp_rd = phase_ringdown[:, 6:7]
        phase_rd_raw = (
            c0 * Mf
            + 1.5 * c1 * (Mf ** (2.0 / 3.0))
            - c2 * (Mf ** -1.0)
            - c4_over_3 * (Mf ** -3.0)
            + c_l_over_f_damp * torch.atan((Mf - f_rd_rd) / f_damp_rd)
        )
        phase_rd = phase_rd_raw + beta0_col + beta1_col * Mf

        half = torch.tensor(0.5, dtype=real_dtype, device=device)
        fM_cut = torch.tensor(_xutils.fM_CUT, dtype=real_dtype, device=device)

        p_ins_mask = torch.heaviside(phase_lower_col - Mf, half)
        p_int_mask1 = torch.heaviside(Mf - phase_lower_col, half)
        p_int_mask2 = torch.heaviside(phase_upper_col - Mf, half)
        p_rd_mask1 = torch.heaviside(Mf - phase_upper_col, half)
        p_rd_mask2 = torch.heaviside(fM_cut - Mf, half)

        phase_xas = (1.0 / eta_col) * (
            phase_ins * p_ins_mask
            + p_int_mask1 * phase_int * p_int_mask2
            + phase_rd * p_rd_mask1 * p_rd_mask2
        )
        extrinsic_phase = 2.0 * math.pi * frequencies_2d * time_shift_col
        phase_xas = (
            phase_xas
            + (linear_b_col * Mf)
            + linear_a_col
            + phase_at_ref_col
            - 2.0 * math.pi
            + extrinsic_phase
        )

        a0 = amp_inspiral[:, 0:1]
        a2 = amp_inspiral[:, 1:2]
        a3 = amp_inspiral[:, 2:3]
        a4 = amp_inspiral[:, 3:4]
        a5 = amp_inspiral[:, 4:5]
        a6 = amp_inspiral[:, 5:6]
        rho1 = amp_inspiral[:, 6:7]
        rho2 = amp_inspiral[:, 7:8]
        rho3 = amp_inspiral[:, 8:9]
        amplitude_ins = (
            a0
            + a2 * (Mf ** (2.0 / 3.0))
            + a3 * Mf
            + a4 * (Mf ** (4.0 / 3.0))
            + a5 * (Mf ** (5.0 / 3.0))
            + a6 * (Mf ** 2.0)
            + rho1 * (Mf ** (7.0 / 3.0))
            + rho2 * (Mf ** (8.0 / 3.0))
            + rho3 * (Mf ** 3.0)
        )

        d0 = amp_intermediate[:, 0:1]
        d1 = amp_intermediate[:, 1:2]
        d2 = amp_intermediate[:, 2:3]
        d3 = amp_intermediate[:, 3:4]
        d4 = amp_intermediate[:, 4:5]
        amplitude_int = (Mf ** (7.0 / 6.0)) / (
            d0 + Mf * (d1 + Mf * (d2 + Mf * (d3 + Mf * d4)))
        )

        f_rd_amp = amp_ringdown[:, 0:1]
        gamma_r = amp_ringdown[:, 1:2]
        gamma_d2 = amp_ringdown[:, 2:3]
        gamma_d13 = amp_ringdown[:, 3:4]
        amp_upper = amp_ringdown[:, 4:5]
        amplitude_rd = (
            torch.exp(-(Mf - f_rd_amp) * gamma_r)
            * gamma_d13
            / ((Mf - f_rd_amp) * (Mf - f_rd_amp) + gamma_d2)
        )

        a_ins_mask = torch.heaviside(amp_match_col - Mf, half)
        a_int_mask1 = torch.heaviside(Mf - amp_match_col, half)
        a_int_mask2 = torch.heaviside(amp_upper - Mf, half)
        a_rd_mask1 = torch.heaviside(Mf - amp_upper, half)
        a_rd_mask2 = torch.heaviside(fM_cut - Mf, half)

        amplitude_xas = (
            amplitude_ins * a_ins_mask
            + a_int_mask1 * amplitude_int * a_int_mask2
            + amplitude_rd * a_rd_mask1 * a_rd_mask2
        )
        amplitude_xas = overall_amp_col * amplitude_xas * (Mf ** (-7.0 / 6.0))

        active_samples_22 = (amplitude_xas * torch.exp(1j * phase_xas)) / _XAS_MODE_POLARIZATION_FACTOR
        mode_dict[2, 2] = active_samples_22

    # Higher-order modes
    h21_l = [] if (2, 1) in mode_set else None
    h33_l = [] if (3, 3) in mode_set else None
    h32_l = [] if (3, 2) in mode_set else None
    h44_l = [] if (4, 4) in mode_set else None

    for b in range(batch_size):
        p_b = {
            "approximant": "IMRPhenomXHM",
            "mass1": float(m1[b]),
            "mass2": float(m2[b]),
            "spin1z": float(s1z[b]),
            "spin2z": float(s2z[b]),
            "distance": float(dist[b]),
            "inclination": 0.0,
            "coa_phase": float(coa_phase[b]),
            "long_asc_nodes": 0.0,
            "f_ref": float(f_ref[b]),
            "delta_f": delta_f,
            "f_lower": f_lower,
            "f_final": f_final,
        }
        core, carrier_phase_plan, carrier_amp_plan = _carrier_core_for_modes(p_b)
        fs = None if final_spin_batch is None else float(final_spin_batch[b])
        if h21_l is not None:
            h21_l.append(
                imrphenomxhm_h2m1_samples(
                    core,
                    p_b,
                    frequencies=frequencies,
                    final_spin=fs,
                    carrier_phase_plan=carrier_phase_plan,
                )
            )
        if h33_l is not None:
            h33_l.append(
                imrphenomxhm_h3m3_samples(
                    core,
                    p_b,
                    frequencies=frequencies,
                    final_spin=fs,
                    carrier_phase_plan=carrier_phase_plan,
                )
            )
        if h32_l is not None:
            h32_l.append(
                imrphenomxhm_h3m2_samples(
                    core,
                    p_b,
                    frequencies=frequencies,
                    final_spin=fs,
                    carrier_phase_plan=carrier_phase_plan,
                    carrier_amp_plan=carrier_amp_plan,
                )
            )
        if h44_l is not None:
            h44_l.append(
                imrphenomxhm_h4m4_samples(
                    core,
                    p_b,
                    frequencies=frequencies,
                    final_spin=fs,
                    carrier_phase_plan=carrier_phase_plan,
                )
            )

    if h21_l is not None:
        mode_dict[2, 1] = torch.stack(h21_l, dim=0)
    if h33_l is not None:
        mode_dict[3, 3] = torch.stack(h33_l, dim=0)
    if h32_l is not None:
        mode_dict[3, 2] = torch.stack(h32_l, dim=0)
    if h44_l is not None:
        mode_dict[4, 4] = torch.stack(h44_l, dim=0)

    return mode_dict


def imrphenomxhm_fd_batch(**params):
    """Generate a batch of IMRPhenomXHM frequency-domain waveforms directly as 2D PyTorch tensors.

    Parameters
    ----------
    mass1 : float or Tensor
        Primary mass in solar masses (shape (B,) or scalar).
    mass2 : float or Tensor
        Secondary mass in solar masses (shape (B,) or scalar).
    spin1z : float or Tensor, optional
        Dimensionless aligned spin of primary (shape (B,) or scalar, default 0.0).
    spin2z : float or Tensor, optional
        Dimensionless aligned spin of secondary (shape (B,) or scalar, default 0.0).
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
    real_dtype = torch.float32 if getattr(device, "type", "cpu") == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    batch_size = 1
    for k in (
        "mass1",
        "mass2",
        "spin1z",
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
            return torch.full((batch_size,), float(val), device=device, dtype=real_dtype)

    m1 = _to_tensor(params["mass1"])
    m2 = _to_tensor(params["mass2"])
    s1z = _to_tensor(params.get("spin1z", 0.0))
    s2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXHM frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXHM delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXHM f_final must be non-negative")

    # Swap masses and spins where m2 > m1
    swap_mask = m2 > m1
    m1_eff = torch.where(swap_mask, m2, m1)
    m2_eff = torch.where(swap_mask, m1, m2)
    s1z_eff = torch.where(swap_mask, s2z, s1z)
    s2z_eff = torch.where(swap_mask, s1z, s2z)

    total_mass_seconds = (m1_eff + m2_eff) * _xutils.MTSUN
    cutoff_frequency = _xutils.fM_CUT / total_mass_seconds
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

    mode_dict = _imrphenomxhm_modes_batch(
        m1_eff,
        m2_eff,
        s1z_eff,
        s2z_eff,
        dist,
        coa_phase,
        f_ref,
        frequencies,
        delta_f,
        f_lower,
        f_final,
    )

    plus = torch.zeros((batch_size, frequencies.shape[0]), dtype=complex_dtype, device=device)
    cross = torch.zeros((batch_size, frequencies.shape[0]), dtype=complex_dtype, device=device)

    for (ell, emm), samples in mode_dict.items():
        parity = (-1) ** ell
        y_neg = spin_weighted_spherical_harmonic(
            incl, math.pi / 2.0, -2, ell, -emm, dtype=real_dtype, device=device
        )
        y_pos_conj = spin_weighted_spherical_harmonic(
            incl, math.pi / 2.0, -2, ell, emm, dtype=real_dtype, device=device
        ).conj()
        f_plus = 0.5 * (y_neg + parity * y_pos_conj)
        f_cross = 0.5j * (y_neg - parity * y_pos_conj)

        plus = plus + f_plus.unsqueeze(1) * samples
        cross = cross + f_cross.unsqueeze(1) * samples

    cos_nodes = torch.cos(2.0 * long_asc_nodes).unsqueeze(1)
    sin_nodes = torch.sin(2.0 * long_asc_nodes).unsqueeze(1)

    hp_active = cos_nodes * plus + sin_nodes * cross
    hc_active = cos_nodes * cross - sin_nodes * plus

    valid_mask = frequencies_2d <= active_f_max.unsqueeze(1)
    hp_active = torch.where(valid_mask, hp_active, torch.zeros_like(hp_active))
    hc_active = torch.where(valid_mask, hc_active, torch.zeros_like(hc_active))

    hp = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hc = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hp[:, first_bin:max_stop_bin] = hp_active
    hc[:, first_bin:max_stop_bin] = hc_active

    return hp, hc
