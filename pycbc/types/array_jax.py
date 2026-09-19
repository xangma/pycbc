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
import math
from types import SimpleNamespace
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from .backend import backend_array, is_backend

# Delegate underlying array storage and in-place methods to array_cpu
try:
    from . import array_cpu as _array_cpu

    for _name in dir(_array_cpu):
        if not _name.startswith("__") and _name not in globals():
            globals()[_name] = getattr(_array_cpu, _name)
except ImportError:
    _array_cpu = None


def _reference_enabled(operation):
    """Whether this operation should use the original CPU kernel."""
    from pycbc import scheme

    return operation in getattr(scheme.mgr.state, "jax_reference_operations", ())


def _cpu_reference(self, operation, *args):
    """Run an unmodified CPU kernel on explicit host storage."""
    if _array_cpu is None:
        raise RuntimeError("CPU array kernels are unavailable for JAX validation")
    host = np.asarray(to_jax(self))
    if operation == "multiply_and_add":
        # BLAS mutates its destination; never expose an immutable device
        # allocation through NumPy's read-only, potentially shared view.
        host = host.copy()
    kind = getattr(self, "kind", "complex" if host.dtype.kind == "c" else "real")
    reference = SimpleNamespace(data=host, _data=host,
                                dtype=host.dtype, kind=kind)
    converted = [np.asarray(to_jax(value)) if not np.isscalar(value) else value
                 for value in args]
    return getattr(_array_cpu, operation)(reference, *converted)


def _as_jax_array(value):
    """Return raw jax.Array from a PyCBC array, JAXArrayData, or JAX array."""
    if isinstance(value, JAXArrayData):
        return value.array
    if hasattr(value, "_data") and isinstance(value._data, JAXArrayData):
        return value._data.array
    if is_backend(value, "jax"):
        raw = backend_array(value, "jax")
        return getattr(raw, "array", raw)
    if is_jax_array(value):
        return getattr(value, "array", value)
    return None


def _unwrap_data(val):
    """Extract raw jax.Array or scalar from JAXArrayData or Series/Array."""
    if isinstance(val, JAXArrayData):
        return val.array
    if hasattr(val, "_data"):
        d = val._data
        if isinstance(d, JAXArrayData):
            return d.array
        return d
    return val


class JAXArrayData:
    """JAX storage with write-through, lazily evaluated ordinary slices."""

    __slots__ = ("_array", "dtype", "parent", "slice_info")
    __array_priority__ = 100.0
    backend = "jax"

    def __init__(self, array, parent=None, slice_info=None):
        _ensure_x64()
        if isinstance(array, JAXArrayData):
            array = array.array
        if array is None and parent is not None and isinstance(slice_info, slice):
            # Reads already resolve against the current parent. Retaining an
            # initial device slice would only keep an unused allocation alive.
            self._array = None
            self.dtype = parent.dtype
            self.parent = parent
            self.slice_info = slice_info
            return
        if not is_jax_array(array):
            import jax
            import jax.numpy as jnp
            from pycbc import scheme

            state = getattr(scheme.mgr, "state", None)
            dev = getattr(state, "jax_device", None)
            array = jnp.asarray(array)
            if dev is not None:
                array = jax.device_put(array, dev)
        self._array = array
        self.dtype = np.dtype(array.dtype)
        self.parent = parent
        self.slice_info = slice_info

    @property
    def array(self):
        if self.parent is not None and self.slice_info is not None:
            return self.parent.array[self.slice_info]
        return self._array

    @array.setter
    def array(self, val):
        self._array = val

    def set_array(self, new_val):
        """Update wrapped array and propagate slice mutations up tree."""
        import jax.numpy as jnp

        if isinstance(new_val, JAXArrayData):
            new_val = new_val.array
        elif hasattr(new_val, "_data"):
            new_val = getattr(new_val._data, "array", new_val._data)
        # PyCBC array assignment preserves the destination dtype.  Applying
        # that rule to JAX inputs as well avoids relying on scatter's implicit
        # casts, which JAX is deprecating for incompatible precisions.
        new_val = jnp.asarray(new_val, dtype=self.dtype)

        curr = self.array
        if self.parent is not None and self.slice_info is not None:
            self.parent.set_slice(self.slice_info, new_val)
        else:
            self._array = jnp.broadcast_to(new_val, curr.shape)

    def set_slice(self, slice_info, new_val):
        """Update slice of wrapped array and propagate to parent if any."""
        import jax.numpy as jnp

        if isinstance(new_val, JAXArrayData):
            new_val = new_val.array
        elif hasattr(new_val, "_data"):
            new_val = getattr(new_val._data, "array", new_val._data)
        new_val = jnp.asarray(new_val, dtype=self.dtype)

        curr = self.array
        updated = curr.at[slice_info].set(new_val)

        if self.parent is not None and self.slice_info is not None:
            self.parent.set_slice(self.slice_info, updated)
        else:
            self._array = updated

    @property
    def shape(self):
        if self._array is None and self.parent is not None:
            shape = self.parent.shape
            length = len(range(*self.slice_info.indices(shape[0])))
            return (length,) + shape[1:]
        if self.parent is None:
            return tuple(self._array.shape)
        return tuple(self.array.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def device(self):
        if self._array is None and self.parent is not None:
            return self.parent.device
        if self.parent is None:
            return getattr(self._array, "device", None)
        return getattr(self.array, "device", None)

    @property
    def backend_array(self):
        """Return raw JAX array through the PyCBC backend protocol."""
        return self.array

    @property
    def data(self):
        return self

    @property
    def real(self):
        return JAXArrayData(self.array.real)

    @property
    def imag(self):
        return JAXArrayData(self.array.imag)

    @property
    def ptr(self):
        # PyCBC uses ptr to identify mutable storage. Independent wrappers
        # can share an immutable JAX value, and lazy slice values are transient.
        return id(self)

    def __array__(self, *args, **kwargs):
        return np.asarray(self.array)

    def numpy(self):
        return np.asarray(self.array)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return JAXArrayData(self.array.reshape(shape))

    def __len__(self):
        if self._array is None and self.parent is not None:
            return self.shape[0]
        return len(self.array)

    def __getitem__(self, item):
        if (isinstance(item, slice) and self.ndim > 0
                and all(bound is None or isinstance(bound, (int, np.integer))
                        for bound in (item.start, item.stop, item.step))):
            # Slicing can change a multi-device array's sharding. Keep its
            # eager path rather than claiming the parent's device metadata.
            device = self.device
            if device is None or isinstance(device, jax.Device):
                # Reject zero steps without dispatching an indexing kernel.
                item.indices(self.shape[0])
                return JAXArrayData(None, parent=self, slice_info=item)
        res = self.array[item]
        if is_jax_array(res) and res.ndim > 0:
            return JAXArrayData(res, parent=self, slice_info=item)
        if hasattr(res, "item"):
            return res.item()
        return res

    def __setitem__(self, item, other):
        self.set_slice(item, other)

    def fill(self, val):
        import jax.numpy as jnp

        new_val = jnp.full(self.shape, val, dtype=self.dtype)
        self.set_array(new_val)

    def __iadd__(self, other):
        self.set_array(self.array + _unwrap_data(other))
        return self

    def __isub__(self, other):
        self.set_array(self.array - _unwrap_data(other))
        return self

    def __imul__(self, other):
        self.set_array(self.array * _unwrap_data(other))
        return self

    def __itruediv__(self, other):
        self.set_array(self.array / _unwrap_data(other))
        return self

    __idiv__ = __itruediv__

    def __add__(self, other):
        return JAXArrayData(self.array + _unwrap_data(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return JAXArrayData(self.array - _unwrap_data(other))

    def __rsub__(self, other):
        return JAXArrayData(_unwrap_data(other) - self.array)

    def __mul__(self, other):
        return JAXArrayData(self.array * _unwrap_data(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return JAXArrayData(self.array / _unwrap_data(other))

    def __rtruediv__(self, other):
        return JAXArrayData(_unwrap_data(other) / self.array)

    def __neg__(self):
        return JAXArrayData(-self.array)

    def __abs__(self):
        import jax.numpy as jnp

        return JAXArrayData(jnp.abs(self.array))

    def conj(self):
        import jax.numpy as jnp

        return JAXArrayData(jnp.conj(self.array))

    def cumsum(self):
        import jax.numpy as jnp

        return JAXArrayData(jnp.cumsum(self.array))

    def sum(self, *args, **kwargs):
        return self.array.sum(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self.array.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self.array.min(*args, **kwargs)

    def __pow__(self, power):
        return JAXArrayData(self.array ** power)

    def __rpow__(self, base):
        return JAXArrayData(base ** self.array)

    def squared_norm(self):
        return JAXArrayData(self.array.real ** 2 + self.array.imag ** 2)

    def astype(self, dtype):
        return JAXArrayData(self.array.astype(dtype))

    def copy(self):
        return JAXArrayData(self.array)

    def view(self, dtype):
        """Reject byte views, which require mutable shared JAX storage."""
        raise NotImplementedError(
            "JAX arrays do not support PyCBC's write-through dtype views"
        )


def _scheme_matches_base_array(array):
    """Check whether array storage matches the JAX scheme."""
    return isinstance(array, JAXArrayData)


def _to_device(array):
    """Convert array storage to JAX scheme base storage."""
    if isinstance(array, JAXArrayData):
        return array
    return JAXArrayData(to_jax(array))


def _copy_base_array(array):
    """Copy array storage in JAX scheme."""
    if isinstance(array, JAXArrayData):
        return array.copy()
    return JAXArrayData(to_jax(array))


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
    """Return True if ``obj`` is a JAX array or JAXArrayData wrapper."""
    if isinstance(obj, JAXArrayData):
        return True
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
        if hasattr(val, "shape") and val.shape == ():
            return val.item()
        return val
    if is_jax_array(data):
        val = data[index]
        if hasattr(val, "shape") and val.shape == ():
            return val.item()
        return val
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

    if isinstance(arr, JAXArrayData):
        res = arr.array
    elif isinstance(getattr(arr, "_data", None), JAXArrayData):
        res = arr._data.array
    elif is_backend(arr, "jax"):
        raw = backend_array(arr, "jax")
        if isinstance(raw, JAXArrayData):
            res = raw.array
        elif is_jax_array(raw):
            res = getattr(raw, "array", raw)
        else:
            res = jnp.asarray(raw)
    elif is_jax_array(arr):
        res = getattr(arr, "array", arr)
    else:
        raw = backend_array(arr)
        if isinstance(raw, JAXArrayData):
            res = raw.array
        elif is_jax_array(raw):
            res = getattr(raw, "array", raw)
        else:
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            elif hasattr(raw, "__array__"):
                raw = np.asarray(raw)

            if hasattr(raw, "__dlpack__"):
                try:
                    res = jax.dlpack.from_dlpack(raw)
                except Exception:
                    res = jnp.asarray(raw)
            else:
                res = jnp.asarray(raw)

    if dtype is not None and res.dtype != dtype:
        res = res.astype(dtype)

    if device is None:
        from pycbc import scheme
        state = getattr(scheme.mgr, "state", None)
        if getattr(scheme, "JAXScheme", None) is not None and isinstance(state, scheme.JAXScheme):
            device = state.jax_device

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
        # Tracers have no concrete devices; placement remains an operation
        # in the traced calculation instead of querying their runtime value.
        if (hasattr(jax, "device_put") and
                (isinstance(res, jax.core.Tracer) or
                 res.devices() != {target_dev})):
            res = jax.device_put(res, target_dev)

    return res


def from_jax(jarr, target_type=None, copy=False, **kwargs):
    """Convert a JAX array to a NumPy array or PyCBC Array/Series.

    Uses DLPack or JAXArrayData wrapper for zero-copy.
    """
    if not is_jax_array(jarr):
        return jarr
    if isinstance(jarr, JAXArrayData):
        jarr = jarr.array

    if target_type is not None and not copy:
        from pycbc.types import Array
        from pycbc import scheme

        if issubclass(target_type, Array) and isinstance(
                scheme.mgr.state, scheme.JAXScheme):
            return target_type(JAXArrayData(jarr), copy=False, **kwargs)

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
    from pycbc.types import Array

    if issubclass(target_type, Array):
        return target_type(narr, copy=copy, **kwargs)

    return target_type(narr, **kwargs)


def abs_max_loc(self):
    """Return max absolute value and its index."""
    import jax.numpy as jnp

    if _reference_enabled("abs_max_loc"):
        return _cpu_reference(self, "abs_max_loc")
    data = getattr(self, "_data", self)
    arr = data.array if isinstance(data, JAXArrayData) else to_jax(data)
    if jnp.iscomplexobj(arr):
        mag_sq = arr.real ** 2 + arr.imag ** 2
        idx = int(jnp.argmax(mag_sq))
        return float(jnp.sqrt(mag_sq[idx])), idx
    else:
        mag = jnp.abs(arr)
        idx = int(jnp.argmax(mag))
        return float(mag[idx]), idx


def abs_arg_max(self):
    """Return the first index with greatest magnitude."""
    if _reference_enabled("abs_arg_max"):
        return _cpu_reference(self, "abs_arg_max")
    arr = to_jax(self)
    if jnp.iscomplexobj(arr):
        magnitude = arr.real ** 2 + arr.imag ** 2
        # Retain the source kernel's initial zero maximum for unordered
        # magnitudes. Exact compiled CPU behavior is available above.
        magnitude = jnp.where(jnp.isnan(magnitude), 0, magnitude)
    else:
        magnitude = jnp.abs(arr)
    return int(jnp.argmax(magnitude))


def take(self, indices):
    """Take flattened elements, retaining NumPy's bounds checking."""
    if isinstance(indices, np.ndarray):
        indices = indices.astype(np.intp, casting="safe", copy=False)
    else:
        indices = np.asarray(indices, dtype=np.intp)
    arr = to_jax(self)
    if np.any(indices < -arr.size) or np.any(indices >= arr.size):
        raise IndexError("index out of bounds for array")
    return JAXArrayData(jnp.take(arr, indices))


def multiply_and_add(self, other, mult_fac):
    """Add a scaled array in place and retain write-through storage."""
    if _reference_enabled("multiply_and_add"):
        result = _cpu_reference(self, "multiply_and_add", other, mult_fac)
        self._data.set_array(result)
        return self._data
    self._data.set_array(to_jax(self) + mult_fac * to_jax(other))
    return self._data


def cumsum(self):
    """Cumulative sum."""
    import jax.numpy as jnp

    if _reference_enabled("cumsum"):
        return JAXArrayData(_cpu_reference(self, "cumsum"))
    s_arr = to_jax(self)
    return JAXArrayData(jnp.cumsum(s_arr))


def dot(self, other):
    """Dot product."""
    import jax.numpy as jnp

    if _reference_enabled("dot"):
        return _cpu_reference(self, "dot", other)
    s_arr = to_jax(self)
    o_arr = to_jax(other)
    return jnp.dot(s_arr, o_arr)


@jax.jit
def _fast_inner(s, o):
    s_c = s.astype(jnp.complex128 if jnp.iscomplexobj(s) else jnp.float64)
    o_c = o.astype(jnp.complex128 if jnp.iscomplexobj(o) else jnp.float64)
    return jnp.sum(jnp.conj(s_c) * o_c)


@jax.jit
def _fast_inner_self(s):
    s_c = s.astype(jnp.complex128 if jnp.iscomplexobj(s) else jnp.float64)
    return jnp.sum(s_c.real ** 2 + s_c.imag ** 2)


@jax.jit
def _fast_weighted_inner(s, o, w):
    s_c = s.astype(jnp.complex128 if jnp.iscomplexobj(s) else jnp.float64)
    o_c = o.astype(jnp.complex128 if jnp.iscomplexobj(o) else jnp.float64)
    w_c = w.astype(jnp.complex128 if jnp.iscomplexobj(w) else jnp.float64)
    return jnp.sum(jnp.conj(s_c) * o_c / w_c)


@jax.jit
def _fast_weighted_inner_self(s, w):
    s_c = s.astype(jnp.complex128 if jnp.iscomplexobj(s) else jnp.float64)
    w_c = w.astype(jnp.complex128 if jnp.iscomplexobj(w) else jnp.float64)
    return jnp.sum((s_c.real ** 2 + s_c.imag ** 2) / w_c)


def inner(self, other):
    """Inner product (conjugate dot) in JAX scheme."""
    if _reference_enabled("inner"):
        return _cpu_reference(self, "inner", other)
    _ensure_x64()
    s_arr = to_jax(self)
    if self is other:
        res = _fast_inner_self(s_arr)
    else:
        o_arr = to_jax(other)
        res = _fast_inner(s_arr, o_arr)
    return res.item() if hasattr(res, "item") else res


def weighted_inner(self, other, weight):
    """Weighted inner product in JAX scheme."""
    if _reference_enabled("weighted_inner"):
        return _cpu_reference(self, "weighted_inner", other, weight)
    _ensure_x64()
    s_arr = to_jax(self)
    w_arr = to_jax(weight)
    if self is other:
        res = _fast_weighted_inner_self(s_arr, w_arr)
    else:
        o_arr = to_jax(other)
        res = _fast_weighted_inner(s_arr, o_arr, w_arr)
    return res.item() if hasattr(res, "item") else res


def squared_norm(self):
    """Sum of squares of real and imaginary parts in JAX scheme."""
    if _reference_enabled("squared_norm"):
        return JAXArrayData(_cpu_reference(self, "squared_norm"))
    s_arr = to_jax(self)
    return JAXArrayData(s_arr.real ** 2 + s_arr.imag ** 2)


def sum(self):
    """Sum using PyCBC's accumulation dtype."""
    if _reference_enabled("sum"):
        return _cpu_reference(self, "sum")
    dtype = jnp.float64 if self.kind == "real" else jnp.complex128
    return to_jax(self).sum(dtype=dtype)


def min(self):
    """Return the minimum value."""
    if _reference_enabled("min"):
        return _cpu_reference(self, "min")
    return to_jax(self).min()


def max(self):
    """Return the maximum value."""
    if _reference_enabled("max"):
        return _cpu_reference(self, "max")
    return to_jax(self).max()


def max_loc(self):
    """Return the maximum value and its first index."""
    if _reference_enabled("max_loc"):
        return _cpu_reference(self, "max_loc")
    arr = to_jax(self)
    index = int(jnp.argmax(arr))
    return arr[index].item(), index


def ptr(self):
    """Pointer/ID representation."""
    data = getattr(self, "_data", self)
    if isinstance(data, JAXArrayData):
        return data.ptr
    return id(data)


def _copy(self, self_ref, other_ref):
    """Copy other_ref into self_ref."""
    if isinstance(self_ref, JAXArrayData):
        self_ref.set_array(other_ref)
    elif (
        hasattr(self_ref, "_data")
        and isinstance(self_ref._data, JAXArrayData)
    ):
        self_ref._data.set_array(other_ref)
    elif _array_cpu is not None and hasattr(_array_cpu, "_copy"):
        _array_cpu._copy(self, self_ref, other_ref)
    else:
        self_ref[:] = np.asarray(other_ref)


def zeros(shape, dtype=np.float64, device=None):
    """Create a JAX array of zeros with given shape and dtype."""
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    # Native PyCBC allocators convert scalar lengths with int(). Frame
    # metadata, in particular, supplies sample counts as Python floats.
    if np.isscalar(shape):
        shape = int(shape)
    res = jnp.zeros(shape, dtype=dtype)
    if device is not None and hasattr(jax, "device_put"):
        res = jax.device_put(res, device)
    return JAXArrayData(res)


def empty(shape, dtype=np.float64, device=None):
    """Create an uninitialized JAX array with given shape and dtype."""
    return zeros(shape, dtype=dtype, device=device)
