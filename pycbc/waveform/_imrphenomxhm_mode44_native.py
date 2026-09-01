# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Fail-closed native CPU boundary helper for IMRPhenomXHM mode 44."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import sysconfig
import threading

import torch

from .torch_switches import _parse_switch


_NATIVE_BOUNDARY_ENV = "PYCBC_IMRPHENOMXHM_MODE44_NATIVE_CPU_BOUNDARY"
_NATIVE_CFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-builtin",
)
_NATIVE_SOURCE_PATH = Path(__file__).with_name(
    "_imrphenomxhm_mode44_native.cpp"
)
_NATIVE_SOURCE_SHA256 = (
    "30cbc1373539a120e543bbf96eae0e44576ac8319231d889f1c454cdcd938e6e"
)
_NATIVE_CACHE_MATERIAL = repr(
    (
        _NATIVE_SOURCE_SHA256,
        _NATIVE_CFLAGS,
        torch.__version__,
        getattr(torch.version, "git_version", None),
        sysconfig.get_config_var("SOABI"),
        sysconfig.get_config_var("MULTIARCH"),
        platform.python_implementation(),
        platform.system(),
        platform.release(),
        platform.machine(),
        getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
    )
).encode()
_NATIVE_CACHE_KEY = hashlib.sha256(_NATIVE_CACHE_MATERIAL).hexdigest()[:20]
_NATIVE_EXTENSION_NAME = f"pycbc_mode44_native_boundary_{_NATIVE_CACHE_KEY}"
_GLOBAL_FACTOR = (
    0.5 ** (-7.0 / 6.0) * (4.0 / 9.0) * math.sqrt(10.0 / 7.0)
)

_STATE_LOCK = threading.RLock()
_STATE_PID = os.getpid()
_EXTENSION = None
_QUALIFIED = False
_FAILED = False
_FAILURE_REASON = None


def native_boundary_enabled():
    """Return the strict, off-by-default native mode-44 switch."""

    value = os.environ.get(_NATIVE_BOUNDARY_ENV)
    return False if value is None else _parse_switch(_NATIVE_BOUNDARY_ENV, value)


def _runtime_supported():
    """Reject execution modes which can observe replacing Torch operations."""

    for function in (
        getattr(torch.jit, "is_scripting", None),
        getattr(torch.jit, "is_tracing", None),
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
        getattr(torch, "is_inference_mode_enabled", None),
    ):
        if function is None:
            return False
        try:
            if function():
                return False
        except Exception:
            return False

    for function in (
        getattr(torch, "is_anomaly_enabled", None),
        getattr(torch, "are_deterministic_algorithms_enabled", None),
    ):
        if function is None:
            return False
        try:
            if function() is not False:
                return False
        except Exception:
            return False

    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
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
        try:
            if autocast_enabled() or legacy_cpu is None or legacy_cpu():
                return False
        except Exception:
            return False
    except Exception:
        return False
    return True


def _plain_owned_tensor(value, *, dtype, shape):
    """Accept one unmodified, owned, eager CPU tensor with a fixed schema."""

    try:
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device.type == "cpu"
            and value.dtype is dtype
            and value.shape == shape
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value._base is None
            and not value._is_view()
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and value.grad_fn is None
            and value.is_leaf
            and value._version == 0
            and value.untyped_storage().nbytes()
            == value.numel() * value.element_size()
        )
    except (AttributeError, RuntimeError, TypeError):
        return False


def _plain_like_tensor(value):
    """Accept the ordinary owned binary64 CPU frequency workspace."""

    try:
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device.type == "cpu"
            and value.dtype is torch.float64
            and value.ndim == 1
            and value.numel() != 0
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value._base is None
            and not value._is_view()
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and value.grad_fn is None
            and value.is_leaf
            and value._version == 0
            and value.untyped_storage().nbytes()
            == value.numel() * value.element_size()
        )
    except (AttributeError, RuntimeError, TypeError):
        return False


def _tensor_values_finite(value):
    """Inspect the tiny fixed-schema CPU inputs without tensor dispatch."""

    try:
        values = value.tolist()
    except Exception:
        return False
    if value.is_complex():
        return all(
            math.isfinite(item.real) and math.isfinite(item.imag)
            for item in values
        )
    return all(map(math.isfinite, values))


def _arguments_supported(point, like, parameters):
    """Validate the exact request-local mode-44 boundary schema."""

    if (
        type(point) is not float
        or not math.isfinite(point)
        or point <= 2.0e-9
        or not _plain_like_tensor(like)
        or type(parameters) is not tuple
        or len(parameters) != 5
    ):
        return False
    amp_norm, fcut, coefficients, pn_dominant, pseudo = parameters
    if (
        type(amp_norm) is not float
        or type(fcut) is not float
        or type(pn_dominant) is not float
        or not math.isfinite(amp_norm)
        or not math.isfinite(fcut)
        or not math.isfinite(pn_dominant)
        or amp_norm <= 0.0
        or fcut <= 2.0e-9
        or point != fcut
        or pn_dominant != amp_norm * 0.5 ** (-7.0 / 6.0)
        or not _plain_owned_tensor(
            coefficients,
            dtype=torch.complex128,
            shape=(7,),
        )
        or not _plain_owned_tensor(pseudo, dtype=torch.float64, shape=(3,))
        or coefficients.device != like.device
        or pseudo.device != like.device
        or not _tensor_values_finite(coefficients)
        or not _tensor_values_finite(pseudo)
    ):
        return False
    return True


def _build_extension():
    """Build or load the content-addressed CPU extension."""

    source = _NATIVE_SOURCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != _NATIVE_SOURCE_SHA256:
        raise RuntimeError("mode-44 native source digest mismatch")
    from torch.utils.cpp_extension import load

    return load(
        name=_NATIVE_EXTENSION_NAME,
        sources=[str(_NATIVE_SOURCE_PATH)],
        extra_cflags=list(_NATIVE_CFLAGS),
        with_cuda=False,
        verbose=False,
        is_python_module=True,
    )


def _mark_failed_locked(reason):
    global _EXTENSION, _QUALIFIED, _FAILED, _FAILURE_REASON

    _EXTENSION = None
    _QUALIFIED = False
    _FAILED = True
    _FAILURE_REASON = f"{type(reason).__name__}: {reason}"


def _after_fork_child():
    """Discard inherited loader state and locks in a forked child."""

    global _STATE_LOCK, _STATE_PID, _EXTENSION, _QUALIFIED

    _STATE_LOCK = threading.RLock()
    _STATE_PID = os.getpid()
    _EXTENSION = None
    _QUALIFIED = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _ensure_process_state():
    if _STATE_PID != os.getpid():
        _after_fork_child()


def _get_extension():
    """Return one process-local extension, permanently disabling on failure."""

    global _EXTENSION

    _ensure_process_state()
    with _STATE_LOCK:
        if _FAILED:
            return None
        if _EXTENSION is None:
            try:
                _EXTENSION = _build_extension()
            except Exception as error:
                _mark_failed_locked(error)
                return None
        return _EXTENSION


def _native_values(extension, point, parameters):
    amp_norm, fcut, coefficients, pn_dominant, pseudo = parameters
    output = extension.evaluate(
        point,
        coefficients,
        pseudo,
        amp_norm,
        fcut,
        pn_dominant,
        _GLOBAL_FACTOR,
    )
    if not _plain_owned_tensor(output, dtype=torch.float64, shape=(2,)):
        raise RuntimeError("mode-44 native extension returned an invalid tensor")
    if not _tensor_values_finite(output):
        raise RuntimeError("mode-44 native extension returned non-finite values")
    return output.unbind()


def _raw_equal(left, right):
    if not (
        type(left) is tuple
        and type(right) is tuple
        and len(left) == len(right) == 2
    ):
        return False
    try:
        return all(
            type(reference) is torch.Tensor
            and type(candidate) is torch.Tensor
            and reference.device.type == "cpu"
            and candidate.device.type == "cpu"
            and reference.dtype is torch.float64
            and candidate.dtype is torch.float64
            and reference.shape == candidate.shape
            and torch.equal(
                reference.detach().contiguous().reshape(-1).view(torch.uint8),
                candidate.detach().contiguous().reshape(-1).view(torch.uint8),
            )
            for reference, candidate in zip(left, right)
        )
    except Exception:
        return False


def try_native_boundary(eager_boundary, point, like, parameters):
    """Return an exact native boundary, eager qualification result, or ``None``.

    The first supported call in each process evaluates both paths, compares
    raw bytes, and returns the eager objects.  A load, execution, or parity
    failure disables the helper for the remainder of that process.
    """

    global _QUALIFIED

    if (
        not native_boundary_enabled()
        or not callable(eager_boundary)
        or not _runtime_supported()
        or not _arguments_supported(point, like, parameters)
    ):
        return None
    extension = _get_extension()
    if extension is None:
        return None

    with _STATE_LOCK:
        if _FAILED:
            return None
        if not _QUALIFIED:
            reference = eager_boundary()
            try:
                candidate = _native_values(extension, point, parameters)
            except Exception as error:
                _mark_failed_locked(error)
                return reference
            if not _raw_equal(reference, candidate):
                _mark_failed_locked(
                    RuntimeError("mode-44 native boundary parity mismatch")
                )
                return reference
            _QUALIFIED = True
            return reference

        try:
            return _native_values(extension, point, parameters)
        except Exception as error:
            _mark_failed_locked(error)
            return None


def _native_boundary_state():
    """Return a read-only state snapshot for diagnostics and tests."""

    _ensure_process_state()
    with _STATE_LOCK:
        return {
            "pid": _STATE_PID,
            "loaded": _EXTENSION is not None,
            "qualified": _QUALIFIED,
            "failed": _FAILED,
            "failure_reason": _FAILURE_REASON,
        }


def _reset_native_boundary_state_for_tests():
    """Reset process-local state; never called by production code."""

    global _STATE_PID, _EXTENSION, _QUALIFIED, _FAILED, _FAILURE_REASON

    with _STATE_LOCK:
        _STATE_PID = os.getpid()
        _EXTENSION = None
        _QUALIFIED = False
        _FAILED = False
        _FAILURE_REASON = None
