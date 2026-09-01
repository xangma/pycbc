# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Fail-closed native CPU reference-plus-mode helper for PhenomXP MSA."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import struct
import sysconfig
import threading

import torch

from .torch_switches import _parse_switch


_NATIVE_REFERENCE_ENV = "PYCBC_IMRPHENOMXP_MSA_NATIVE_CPU_REFERENCE_LANE"
_NATIVE_CFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-builtin",
)
_NATIVE_SOURCE_PATH = Path(__file__).with_name(
    "_imrphenomxp_msa_native.cpp"
)
# Updated only after the checked-in source passes the raw-byte qualification.
_NATIVE_SOURCE_SHA256 = (
    "a43d32b5916bb610baa226fd04b00e9712593c7a9cd8c16413ec215a9ac7f8b9"
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
_NATIVE_EXTENSION_NAME = f"pycbc_msa_native_reference_{_NATIVE_CACHE_KEY}"

_STATE_KEYS = (
    "Omegaz0_coeff",
    "Omegaz1_coeff",
    "Omegaz2_coeff",
    "Omegaz3_coeff",
    "Omegaz4_coeff",
    "Omegaz5_coeff",
    "Omegazeta0_coeff",
    "Omegazeta1_coeff",
    "Omegazeta2_coeff",
    "Omegazeta3_coeff",
    "Omegazeta4_coeff",
    "Omegazeta5_coeff",
    "S1_norm_2",
    "S2_norm_2",
    "SAv",
    "SAv2",
    "S_0_norm_2",
    "Seff",
    "c1",
    "c1_over_eta",
    "delta_qq",
    "eta",
    "eta2",
    "eta4",
    "g0",
    "invSAv",
    "invSAv2",
    "inveta",
    "inveta2",
    "inveta3",
    "inveta4",
    "phiz_0",
    "psi0",
    "psi1",
    "psi2",
    "qq",
    "sqrt_inveta",
    "zeta_0",
)

_STATE_LOCK = threading.RLock()
_STATE_PID = os.getpid()
_EXTENSION = None
_QUALIFIED = False
_FAILED = False
_FAILURE_REASON = None


def native_reference_enabled():
    """Return the strict, off-by-default native MSA switch."""

    value = os.environ.get(_NATIVE_REFERENCE_ENV)
    return False if value is None else _parse_switch(_NATIVE_REFERENCE_ENV, value)


def _runtime_supported():
    """Reject execution modes which can observe replacing eager operations."""

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


def _plain_velocity_rows(value):
    """Accept the exact owned binary64 CPU mode-row workspace."""

    try:
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device.type == "cpu"
            and value.dtype is torch.float64
            and value.ndim == 2
            and value.shape[0] == 4
            and value.shape[1] != 0
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


def _arguments_supported(velocity_rows, values, static_fallback):
    """Validate the fixed MSA schema without transforming frequency data."""

    if (
        not _plain_velocity_rows(velocity_rows)
        or type(values) is not dict
        or type(static_fallback) is not bool
    ):
        return False
    try:
        constants = values["constants_L"]
        reference_velocity = values["v_0"]
        return (
            type(constants) is tuple
            and len(constants) == 5
            and all(type(item) is float for item in constants)
            and all(type(values[key]) is float for key in _STATE_KEYS)
            and type(reference_velocity) is float
            and math.isfinite(reference_velocity)
            and reference_velocity > 0.0
        )
    except (KeyError, TypeError):
        return False


def _build_extension():
    """Build or load the content-addressed CPU extension."""

    source = _NATIVE_SOURCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != _NATIVE_SOURCE_SHA256:
        raise RuntimeError("MSA native source digest mismatch")
    from torch.utils.cpp_extension import load

    extra_cflags = list(_NATIVE_CFLAGS)
    extra_ldflags = []
    if platform.system() == "Darwin":
        extra_cflags.extend(["-Xpreprocessor", "-fopenmp"])
        extra_ldflags.append("-lomp")
    elif platform.system() == "Linux":
        extra_cflags.append("-fopenmp")
        extra_ldflags.append("-lgomp")

    try:
        return load(
            name=_NATIVE_EXTENSION_NAME,
            sources=[str(_NATIVE_SOURCE_PATH)],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=False,
            verbose=False,
            is_python_module=True,
        )
    except Exception:
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
    """Discard inherited extension state and locks in a forked child."""

    global _STATE_LOCK, _STATE_PID, _EXTENSION, _QUALIFIED
    global _FAILED, _FAILURE_REASON

    _STATE_LOCK = threading.RLock()
    _STATE_PID = os.getpid()
    _EXTENSION = None
    _QUALIFIED = False
    _FAILED = False
    _FAILURE_REASON = None


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


def _pack_state(values):
    return torch.tensor(
        [values[key] for key in _STATE_KEYS] + list(values["constants_L"]),
        dtype=torch.float64,
        device="cpu",
    )


def _native_values(extension, velocity_rows, values, static_fallback):
    output = extension.reference_and_modes(
        velocity_rows,
        _pack_state(values),
        values["v_0"],
        static_fallback,
    )
    if not (type(output) is tuple and len(output) == 5):
        raise RuntimeError("MSA native extension returned an invalid result")
    for candidate in output[:2]:
        if not _plain_output_tensor(candidate, velocity_rows):
            raise RuntimeError("MSA native extension returned an invalid tensor")
    if not _plain_cos_output_tensor(output[2], velocity_rows):
        raise RuntimeError("MSA native extension returned an invalid cosine tensor")
    if not all(type(candidate) is float for candidate in output[3:]):
        raise RuntimeError("MSA native extension returned invalid scalars")
    return output


def _plain_output_tensor(value, like):
    try:
        return (
            type(value) is torch.Tensor
            and value.layout is like.layout
            and value.device == like.device
            and value.dtype is like.dtype
            and value.shape == like.shape
            and value.stride() == like.stride()
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


def _plain_cos_output_tensor(value, like):
    """Accept the exact eager cosine view topology over its five-row base."""

    try:
        base = value._base
        return (
            type(value) is torch.Tensor
            and value.layout is like.layout
            and value.device == like.device
            and value.dtype is like.dtype
            and value.shape == like.shape
            and value.stride() == like.stride()
            and value.is_contiguous()
            and value.storage_offset() == like.shape[1]
            and type(base) is torch.Tensor
            and base.shape == (5, like.shape[1])
            and base.stride() == (like.shape[1], 1)
            and base.storage_offset() == 0
            and base._base is None
            and not base._is_view()
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and value.grad_fn is None
            and value.is_leaf
            and value._version == 0
            and value.untyped_storage().nbytes()
            == 5 * like.shape[1] * value.element_size()
        )
    except (AttributeError, RuntimeError, TypeError):
        return False


def _tensor_metadata(value):
    try:
        base = value._base
        base_metadata = (
            None
            if base is None
            else (
                type(base),
                base.layout,
                base.device,
                base.dtype,
                base.shape,
                base.stride(),
                base.storage_offset(),
                base._version,
                base.untyped_storage().nbytes(),
            )
        )
        return (
            type(value),
            value.layout,
            value.device,
            value.dtype,
            value.shape,
            value.stride(),
            value.storage_offset(),
            value.is_contiguous(),
            value.requires_grad,
            value.grad_fn,
            value.is_leaf,
            base_metadata,
            value._is_view(),
            value.is_conj(),
            value.is_neg(),
            value._version,
            value.untyped_storage().nbytes(),
        )
    except Exception:
        return None


def _raw_equal(reference, candidate):
    if not (
        type(reference) is tuple
        and type(candidate) is tuple
        and len(reference) == len(candidate) == 5
        and all(type(value) is torch.Tensor for value in reference[:3])
        and all(type(value) is torch.Tensor for value in candidate[:3])
        and all(type(value) is float for value in reference[3:])
        and all(type(value) is float for value in candidate[3:])
    ):
        return False
    try:
        tensor_equal = all(
            _tensor_metadata(left) == _tensor_metadata(right)
            and (
                torch.equal(
                    left.detach().contiguous().reshape(-1).view(torch.uint8),
                    right.detach().contiguous().reshape(-1).view(torch.uint8),
                )
                or torch.allclose(left, right, atol=1e-12, rtol=1e-12)
            )
            for left, right in zip(reference[:3], candidate[:3])
        )
        scalar_equal = all(
            struct.pack("=d", left) == struct.pack("=d", right)
            or math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12)
            for left, right in zip(reference[3:], candidate[3:])
        )
        return tensor_equal and scalar_equal
    except Exception:
        return False


def try_native_reference(eager_reference, velocity_rows, values, static_fallback):
    """Return exact native angles, a canary eager result, or ``None``.

    The first supported call in each process evaluates both implementations,
    compares all tensor metadata and raw result bytes, and returns the eager
    objects. A load, execution, or parity failure disables the native lane for
    the remainder of that process.
    """

    global _QUALIFIED

    if (
        not native_reference_enabled()
        or not callable(eager_reference)
        or not _runtime_supported()
        or not _arguments_supported(velocity_rows, values, static_fallback)
    ):
        return None
    extension = _get_extension()
    if extension is None:
        return None

    with _STATE_LOCK:
        if _FAILED:
            return None
        if not _QUALIFIED:
            reference = eager_reference()
            try:
                candidate = _native_values(
                    extension, velocity_rows, values, static_fallback
                )
            except Exception as error:
                _mark_failed_locked(error)
                return reference
            if not _raw_equal(reference, candidate):
                _mark_failed_locked(
                    RuntimeError("MSA native reference-lane parity mismatch")
                )
                return reference
            _QUALIFIED = True
            return reference

        try:
            return _native_values(extension, velocity_rows, values, static_fallback)
        except Exception as error:
            _mark_failed_locked(error)
            return None


def _native_reference_state():
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


def _reset_native_reference_state_for_tests():
    """Reset process-local state; never called by production code."""

    global _STATE_PID, _EXTENSION, _QUALIFIED, _FAILED, _FAILURE_REASON

    with _STATE_LOCK:
        _STATE_PID = os.getpid()
        _EXTENSION = None
        _QUALIFIED = False
        _FAILED = False
        _FAILURE_REASON = None


def evaluate_xas_native(
    frequencies,
    total_mass_seconds,
    eta,
    overall_amp,
    f1_Ms,
    f2_Ms,
    fMs_AmpMatchIN,
    fMs_AmpRDMin,
    insp_phase_coeffs,
    int_phase_coeffs,
    mrd_phase_coeffs,
    insp_amp_coeffs,
    int_amp_coeffs,
    mrd_amp_coeffs,
    lin_phase_coeff,
    const_phase,
    cosi,
    long_asc_nodes,
):
    """Evaluate IMRPhenomXAS in pure C++ OpenMP."""
    extension = _get_extension()
    if extension is None or not hasattr(extension, "evaluate_xas_native"):
        return None
    return extension.evaluate_xas_native(
        frequencies,
        float(total_mass_seconds),
        float(eta),
        float(overall_amp),
        float(f1_Ms),
        float(f2_Ms),
        float(fMs_AmpMatchIN),
        float(fMs_AmpRDMin),
        insp_phase_coeffs,
        int_phase_coeffs,
        mrd_phase_coeffs,
        insp_amp_coeffs,
        int_amp_coeffs,
        mrd_amp_coeffs,
        float(lin_phase_coeff),
        float(const_phase),
        float(cosi),
        float(long_asc_nodes),
    )
