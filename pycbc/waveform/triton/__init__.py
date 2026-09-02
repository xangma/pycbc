# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Fused Triton kernels for ultra-fast GPU waveform generation and ODE stepping."""

from __future__ import annotations

from .taylorf2 import (
    _taylorf2_fused_kernel,
    _TRITON_AVAILABLE,
    is_triton_available,
)

__all__ = [
    "_taylorf2_fused_kernel",
    "_TRITON_AVAILABLE",
    "is_triton_available",
]
