# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Lazy exact-native executor for the mode-32 mixed CPU boundary.

This module deliberately owns only build and qualification lifecycle.  The
waveform module proves the physical/runtime input contract before importing
it, and always retains the established eager implementation as fallback.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import sysconfig
import threading

import torch


_SOURCE_PATH = Path(__file__).with_name("_imrphenomxhm_mode32_native.cpp")
_CFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-builtin",
    "-fno-lto",
)
_EXTENSION_PREFIX = "pycbc_mode32_native_cpu"

_EXTENSION = None
_QUALIFIED = False
_FAILED = False
_PID = os.getpid()
_LOCK = threading.Lock()


def _identity():
    """Return the deterministic cache identity for this exact ABI/source."""

    source = _SOURCE_PATH.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    material = repr(
        (
            source_sha256,
            _CFLAGS,
            torch.__version__,
            sysconfig.get_config_var("SOABI"),
            sysconfig.get_config_var("MULTIARCH"),
            platform.python_implementation(),
            platform.system(),
            platform.release(),
            platform.machine(),
            getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
        )
    ).encode()
    key = hashlib.sha256(material).hexdigest()[:20]
    return source_sha256, key, f"{_EXTENSION_PREFIX}_{key}"


def cache_metadata():
    """Expose immutable build identity for diagnostics and qualification."""

    source_sha256, key, name = _identity()
    return {
        "source": str(_SOURCE_PATH),
        "source_sha256": source_sha256,
        "cache_key": key,
        "extension_name": name,
        "cflags": _CFLAGS,
    }


def _load_extension_unlocked():
    """Load or build the content-addressed CPU extension."""

    from torch.utils.cpp_extension import load

    _source_sha256, _key, name = _identity()
    return load(
        name=name,
        sources=[str(_SOURCE_PATH)],
        extra_cflags=list(_CFLAGS),
        with_cuda=False,
        verbose=False,
    )


def _reset_after_fork():
    """Drop inherited module state and replace a potentially held lock."""

    global _EXTENSION, _QUALIFIED, _FAILED, _PID, _LOCK

    _EXTENSION = None
    _QUALIFIED = False
    _FAILED = False
    _PID = os.getpid()
    _LOCK = threading.Lock()


def _ensure_process():
    """Reset inherited state where an at-fork hook is unavailable."""

    if _PID != os.getpid():
        _reset_after_fork()


def clear_cache():
    """Clear process-local qualification and sticky failure state."""

    global _EXTENSION, _QUALIFIED, _FAILED

    _ensure_process()
    with _LOCK:
        _EXTENSION = None
        _QUALIFIED = False
        _FAILED = False


def cache_state():
    """Return the published executor and sticky-failure state."""

    _ensure_process()
    return (_EXTENSION if _QUALIFIED else None), _FAILED


def mark_failed():
    """Fail subsequent calls closed until explicitly cleared."""

    global _EXTENSION, _QUALIFIED, _FAILED

    _ensure_process()
    with _LOCK:
        _EXTENSION = None
        _QUALIFIED = False
        _FAILED = True


def _output_supported(value):
    """Validate the immutable packed output before publishing an executor."""

    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.shape == torch.Size((2,))
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
    )


def _raw_equal(reference, candidate):
    """Compare qualified float64 outputs without a numeric conversion."""

    if not (_output_supported(reference) and _output_supported(candidate)):
        return False
    return torch.equal(
        reference.detach().view(torch.uint8),
        candidate.detach().view(torch.uint8),
    )


def get_or_build_qualified(inputs, reference):
    """Build and publish only after a first-request raw-byte self-check.

    The caller must return its already-computed eager ``reference`` for this
    request, even when qualification succeeds.  Concurrent cold callers may
    also return eager; only later requests observe the published executor.
    """

    global _EXTENSION, _QUALIFIED, _FAILED

    _ensure_process()
    if _QUALIFIED:
        return _EXTENSION
    if _FAILED:
        return None
    with _LOCK:
        if _QUALIFIED:
            return _EXTENSION
        if _FAILED:
            return None
        try:
            extension = _load_extension_unlocked()
            candidate = extension.evaluate_packed(*inputs)
            if not _raw_equal(reference, candidate):
                raise RuntimeError("native mode-32 boundary changed output bytes")
        except Exception:
            _EXTENSION = None
            _QUALIFIED = False
            _FAILED = True
            return None
        _EXTENSION = extension
        _QUALIFIED = True
        return extension


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
