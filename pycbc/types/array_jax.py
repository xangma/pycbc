# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""JAX array interoperability and DLPack conversion utilities for PyCBC.

Provides zero-copy and conversion helpers between PyCBC types (Array,
TimeSeries, FrequencySeries), NumPy arrays, and JAX arrays via DLPack.
"""

import os
import numpy as np

from .backend import backend_array


def _ensure_x64():
    """Ensure JAX is configured with 64-bit precision for GW physics."""
    import jax

    if os.environ.get("PYCBC_JAX_ENABLE_X64", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        jax.config.update("jax_enable_x64", True)


def is_jax_array(obj):
    """Return True if ``obj`` is a JAX array."""
    from .backend import jax_module_for

    return jax_module_for(obj) is not None


def to_jax(arr, device=None, dtype=None):
    """Convert an Array, Series, or numpy array to a jax.Array.

    Attempts zero-copy DLPack conversion when possible.
    """
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    if is_jax_array(arr):
        res = arr
    else:
        # Check for PyCBC Array or Series
        raw = backend_array(arr)
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        elif hasattr(raw, "__array__"):
            raw = np.asarray(raw)

        # Attempt DLPack transfer
        if hasattr(raw, "__dlpack__"):
            try:
                res = jax.dlpack.from_dlpack(raw)
            except Exception:
                res = jnp.asarray(raw)
        else:
            res = jnp.asarray(raw)

    if dtype is not None and res.dtype != dtype:
        res = res.astype(dtype)

    if device is not None:
        target_dev = device
        if isinstance(device, str):
            devices = jax.devices()
            if device in ("cpu", "cuda", "gpu", "tpu"):
                matched = [
                    d for d in devices
                    if d.platform == device
                    or (device == "cuda" and d.platform == "gpu")
                ]
                if matched:
                    target_dev = matched[0]
            elif device.isdigit():
                target_dev = devices[int(device)]
        if hasattr(jax, "device_put"):
            res = jax.device_put(res, target_dev)

    return res


def from_jax(jarr, target_type=None, copy=False):
    """Convert a JAX array to a NumPy array or PyCBC Array/Series.

    Uses DLPack when available.
    """
    if not is_jax_array(jarr):
        return jarr

    try:
        narr = np.from_dlpack(jarr)
        if copy:
            narr = narr.copy()
    except Exception:
        narr = np.asarray(jarr)
        if copy:
            narr = narr.copy()

    if target_type is None:
        return narr

    # Convert to requested PyCBC container type
    from pycbc.types import Array, FrequencySeries, TimeSeries

    if issubclass(target_type, (Array, TimeSeries, FrequencySeries)):
        return target_type(narr, copy=copy)
    return target_type(narr)


def zeros(shape, dtype=np.float64, device=None):
    """Create a JAX array of zeros with given shape and dtype."""
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    res = jnp.zeros(shape, dtype=dtype)
    if device is not None and hasattr(jax, "device_put"):
        res = jax.device_put(res, device)
    return res


def empty(shape, dtype=np.float64, device=None):
    """Create an uninitialized JAX array with given shape and dtype."""
    return zeros(shape, dtype=dtype, device=device)
