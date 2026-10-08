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
from pycbc.events.ranking_jax import _event_arrays
from pycbc.types.array_jax import (
    JAXArrayData,
    _as_jax_array,
    _ensure_x64,
    _reference_enabled,
)


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
    hi = hi[order]
    running_end = jax.lax.associative_scan(jnp.maximum, hi)
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
    if _reference_enabled('segment_veto'):
        return _native_veto(times, start, end, outside=False)
    times_arr, start_arr, end_arr = _event_arrays(times, start, end)

    start_c, end_c = _coalesced_segments(start_arr, end_arr)
    if times_arr.size == 0 or start_c.size == 0:
        dtype = jnp.uint32 if start_c.size == 0 else jnp.int64
        empty = jnp.empty(0, dtype=dtype)
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
    if _reference_enabled('segment_veto'):
        return _native_veto(times, start, end, outside=True)
    exclude = indices_within_times(times, start, end)
    exclude_raw = _as_jax_array(exclude)
    times_arr = _event_arrays(times)[0]
    n = times_arr.size
    keep = jnp.ones(n, dtype=bool)
    if exclude_raw.size > 0:
        keep = keep.at[exclude_raw].set(False)
    res = jnp.flatnonzero(keep)
    return _wrap_veto_result((times, start, end), res)


def _native_veto(times, start, end, outside):
    """Execute the original segment selection, including tie ordering."""
    from pycbc.reference_jax import cpu_reference

    inputs = (times, start, end)
    arrays = _event_arrays(*inputs)
    host = jax.device_get(arrays)
    result = cpu_reference('segment_veto', host[0], start=host[1], end=host[2],
                           outside=outside)
    return _wrap_veto_result(inputs, jax.device_put(result, arrays[0].device))
