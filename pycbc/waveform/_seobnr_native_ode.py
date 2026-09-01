# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Lazy exact-native C++ RKF45 ODE integrator for SEOBNR dynamics.

Manages build, JIT compilation, caching, and qualification lifecycle of
the C++ native RKF45 integrator extension.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import sysconfig
import threading

import torch


_SOURCE_PATH = Path(__file__).with_name("seobnr_native_ode.cpp")
_CFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-builtin",
    "-fno-lto",
)
_EXTENSION_PREFIX = "pycbc_seobnr_native_ode_cpu"

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


def get_extension():
    """Get or compile the C++ native ODE extension."""
    global _EXTENSION, _QUALIFIED, _FAILED

    flag = os.environ.get("PYCBC_SEOBNR_NATIVE_ODE", "1")
    if flag in ("0", "", "false", "False"):
        return None

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
            ext = _load_extension_unlocked()
            _EXTENSION = ext
            _QUALIFIED = True
            return _EXTENSION
        except Exception:
            _EXTENSION = None
            _QUALIFIED = False
            _FAILED = True
            return None


def is_available() -> bool:
    """Return whether the native C++ ODE extension is available."""
    ext = get_extension()
    return ext is not None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
