# Copyright (C) 2026  The PyCBC Collaboration
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

"""JAX backend for segment-based event vetoes and time window masking."""

import jax
import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, is_jax_array
from pycbc.types.backend import backend_array, is_backend


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


def _wrap_veto_result(inputs, value):
    """Preserve PyCBC Array inputs without transferring indices to host."""
    if not any(isinstance(v, Array) for v in inputs):
        return value
    return Array(JAXArrayData(value), copy=False)


def _coalesced_segments(start, end):
    """Return sorted, coalesced half-open segments on JAX device."""
    _ensure_x64()
    lo = jnp.minimum(start, end)
    hi = jnp.maximum(start, end)
    nonempty = lo != hi
    lo = lo[nonempty]
    hi = hi[nonempty]
    if lo.size == 0:
        return lo, hi

    order = jnp.argsort(lo)
    lo = lo[order]
    try:
        running_end = jax.lax.associative_scan(jnp.maximum, hi)
    except Exception:
        running_end = jnp.maximum.accumulate(hi)
    new_segment = jnp.concatenate(
        [jnp.ones(1, dtype=bool), lo[1:] > running_end[:-1]]
    )
    last_in_segment = jnp.concatenate(
        [new_segment[1:], jnp.ones(1, dtype=bool)]
    )
    return lo[new_segment], running_end[last_in_segment]


def indices_within_times(times, start, end):
    """Return trigger indices occurring within start/end segments in JAX."""
    _ensure_x64()
    times_arr = _as_jax_array(times)
    start_arr = _as_jax_array(start)
    end_arr = _as_jax_array(end)
    reference = times_arr if times_arr is not None else start_arr
    if reference is None:
        raise TypeError("the JAX veto backend requires a JAX-backed input")

    def _as_time_arr(val, arr):
        if arr is None:
            host = val.numpy() if isinstance(val, Array) else val
            arr = jnp.asarray(host, dtype=reference.dtype)
        return arr

    times_arr = _as_time_arr(times, times_arr)
    start_arr = _as_time_arr(start, start_arr)
    end_arr = _as_time_arr(end, end_arr)

    start_c, end_c = _coalesced_segments(start_arr, end_arr)
    if times_arr.size == 0 or start_c.size == 0:
        empty = jnp.empty(0, dtype=jnp.int64)
        return _wrap_veto_result((times, start, end), empty)

    order = jnp.argsort(times_arr)
    sorted_times = times_arr[order]
    segment_id = jnp.searchsorted(start_c, sorted_times, side="right") - 1
    safe_segment_id = jnp.clip(segment_id, 0, start_c.size - 1)
    within = (segment_id >= 0) & (sorted_times < end_c[safe_segment_id])
    res = order[within]
    return _wrap_veto_result((times, start, end), res)


def indices_outside_times(times, start, end):
    """Return trigger indices outside start/end segments in JAX."""
    _ensure_x64()
    exclude = indices_within_times(times, start, end)
    exclude_raw = _as_jax_array(exclude)
    times_arr = _as_jax_array(times)
    if times_arr is None:
        times_arr = jnp.asarray(times.numpy() if isinstance(times, Array) else times)
    n = times_arr.size
    keep = jnp.ones(n, dtype=bool)
    if exclude_raw.size > 0:
        keep = keep.at[exclude_raw].set(False)
    res = jnp.flatnonzero(keep)
    return _wrap_veto_result((times, start, end), res)


def complement_indices(times, exclude):
    """Return the device-resident complement of an index vector in JAX."""
    _ensure_x64()
    times_arr = _as_jax_array(times)
    if times_arr is None:
        times_arr = jnp.asarray(times.numpy() if isinstance(times, Array) else times)
    exclude_arr = _as_jax_array(exclude)
    if exclude_arr is None:
        exclude_arr = jnp.asarray(
            exclude.numpy() if isinstance(exclude, Array) else exclude,
            dtype=jnp.int64,
        )
    n = times_arr.size
    keep = jnp.ones(n, dtype=bool)
    if exclude_arr.size > 0:
        keep = keep.at[exclude_arr.astype(jnp.int64)].set(False)
    res = jnp.flatnonzero(keep)
    return _wrap_veto_result((times, exclude), res)
