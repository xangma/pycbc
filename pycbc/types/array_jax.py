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

from .backend import backend_array, is_backend

# Delegate underlying array storage and in-place methods to array_cpu
try:
    from . import array_cpu as _array_cpu

    for _name in dir(_array_cpu):
        if not _name.startswith("__") and _name not in globals():
            globals()[_name] = getattr(_array_cpu, _name)
except ImportError:
    _array_cpu = None


class JAXArrayData:
    """Lightweight wrapper around a JAX array with NumPy compatibility."""

    __slots__ = ("array", "dtype")
    __array_priority__ = 100.0
    backend = "jax"

    def __init__(self, array):
        _ensure_x64()
        if not is_jax_array(array):
            import jax.numpy as jnp

            array = jnp.asarray(array)
        self.array = array
        self.dtype = np.dtype(array.dtype)

    @property
    def shape(self):
        return tuple(self.array.shape)

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def nbytes(self):
        return self.array.size * self.dtype.itemsize

    @property
    def device(self):
        return getattr(self.array, "device", None)

    @property
    def backend_array(self):
        """Return raw JAX array through the PyCBC backend protocol."""
        return self.array

    def __array__(self, *args, **kwargs):
        return np.asarray(self.array)

    def numpy(self):
        return np.asarray(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        res = self.array[item]
        if is_jax_array(res) and res.ndim > 0:
            return JAXArrayData(res)
        if hasattr(res, "item"):
            return res.item()
        return res

    def copy(self):
        return JAXArrayData(self.array)


def _scheme_matches_base_array(array):
    """Check whether array storage matches the JAX scheme."""
    return (
        isinstance(array, (np.ndarray, np.generic))
        or is_jax_array(array)
        or isinstance(array, JAXArrayData)
        or getattr(array, "backend", None) == "jax"
    )


def _to_device(array):
    """Convert array storage to JAX scheme base storage."""
    return np.asarray(array)


def _copy_base_array(array):
    """Copy array storage in JAX scheme."""
    return array.copy()


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


def numpy(self):
    """Return numpy array from Array under JAXScheme."""
    data = getattr(self, "_data", self)
    if isinstance(data, JAXArrayData):
        return data.numpy()
    if is_jax_array(data):
        return np.asarray(data)
    if _array_cpu is not None and hasattr(_array_cpu, "numpy"):
        return _array_cpu.numpy(self)
    return np.asarray(data)


def _getvalue(self, index):
    """Return scalar element from Array under JAXScheme."""
    data = getattr(self, "_data", self)
    if isinstance(data, JAXArrayData):
        val = data.array[index]
        return val.item() if hasattr(val, "item") else val
    if is_jax_array(data):
        val = data[index]
        return val.item() if hasattr(val, "item") else val
    if _array_cpu is not None and hasattr(_array_cpu, "_getvalue"):
        return _array_cpu._getvalue(self, index)
    return data[index]


def to_jax(arr, device=None, dtype=None):
    """Convert an Array, Series, or numpy array to a jax.Array.

    Attempts zero-copy DLPack conversion when possible.
    """
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    if is_jax_array(arr):
        res = arr
    elif isinstance(arr, JAXArrayData):
        res = arr.array
    elif isinstance(getattr(arr, "_data", None), JAXArrayData):
        res = arr._data.array
    elif is_backend(arr, "jax"):
        raw = backend_array(arr, "jax")
        if is_jax_array(raw):
            res = raw
        else:
            res = jnp.asarray(raw)
    else:
        # Check for PyCBC Array or Series
        raw = backend_array(arr)
        if is_jax_array(raw):
            res = raw
        elif hasattr(raw, "numpy"):
            raw = raw.numpy()
        elif hasattr(raw, "__array__"):
            raw = np.asarray(raw)

        if is_jax_array(raw):
            res = raw
        elif hasattr(raw, "__dlpack__"):
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
                    d
                    for d in devices
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


def from_jax(jarr, target_type=None, copy=False, **kwargs):
    """Convert a JAX array to a NumPy array or PyCBC Array/Series.

    Uses DLPack or JAXArrayData wrapper for zero-copy.
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
    from pycbc import scheme as _scheme

    if issubclass(target_type, (Array, TimeSeries, FrequencySeries)):
        if isinstance(_scheme.mgr.state, _scheme.JAXScheme) and not copy:
            return target_type(JAXArrayData(jarr), copy=False, **kwargs)
        return target_type(narr, copy=copy, **kwargs)

    return target_type(narr, **kwargs)


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
