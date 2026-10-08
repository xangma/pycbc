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

"""JAX coincidence construction and clustering."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, _reference_enabled
from .ranking_jax import _event_arrays


def _wrap_coincidence_result(t1, t2, *values):
    """Preserve PyCBC Array inputs while leaving raw arrays as JAX arrays."""
    if not isinstance(t1, Array) and not isinstance(t2, Array):
        return values
    return tuple(Array(JAXArrayData(value), copy=False) for value in values)


def _wrap_cluster_result(inputs, value):
    """Return an Array when any public clustering input was an Array."""
    if not any(isinstance(item, Array) for item in inputs):
        return value
    return Array(JAXArrayData(value), copy=False)


def _cluster_vectors(values):
    """Convert mixed clustering vectors to JAX arrays on the same device."""
    _ensure_x64()
    result = _event_arrays(*values)
    for arr in result:
        if arr.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
    return result


def _native_coincidence(operation, arrays, **options):
    """Execute an original CPU selection on the same concrete input values."""
    from pycbc.reference_jax import cpu_reference

    result = cpu_reference(operation, jax.device_get(arrays[0]), **options)
    return jax.device_put(result, arrays[0].device)


@partial(jax.jit, static_argnames=("sliding",))
def _time_coincidence_single_core(arr1, arr2, window, slide_step, sliding):
    """Pack a bounded singleton match into fixed-capacity device outputs."""
    fold1 = jnp.fmod(arr1, slide_step) if sliding else arr1
    fold2 = jnp.fmod(arr2, slide_step) if sliding else arr2
    sort1 = jnp.argsort(fold1)
    fold1 = fold1[sort1]
    if sliding:
        fold2 = jnp.concatenate((fold2 - slide_step, fold2,
                                 fold2 + slide_step))

    # Both original searches use side="left": the lower endpoint is
    # inclusive and the upper endpoint is exclusive. Searching also retains
    # the original behavior for nonfinite trigger times.
    left = jnp.searchsorted(fold2, fold1 - window)
    right = jnp.searchsorted(fold2, fold1 + window)
    positions = jnp.arange(fold2.size)[None, :]
    matched = ((positions >= left[:, None]) &
               (positions < right[:, None]))
    capacity = arr1.size * fold2.size
    rows, _ = jnp.nonzero(matched, size=capacity, fill_value=0)
    idx1 = sort1[rows]
    idx2 = jnp.zeros(capacity, dtype=jnp.int64)
    if sliding:
        difference = (arr1[idx1] - arr2[0]) / slide_step
        slide = jnp.where(
            difference >= 0,
            jnp.floor(difference + 0.5),
            jnp.ceil(difference - 0.5),
        ).astype(jnp.int32)
    else:
        slide = jnp.zeros(capacity, dtype=jnp.int32)
    return idx1, idx2, slide, jnp.sum(matched, dtype=jnp.int64)


@partial(jax.jit, static_argnames=("count",))
def _compact_time_coincidence(idx1, idx2, slide, count):
    """Compact all result columns together after reading their valid length."""
    return idx1[:count], idx2[:count], slide[:count]


def time_coincidence(t1, t2, window, slide_step=0):
    """Find coincidences on the JAX device after converting host inputs."""
    _ensure_x64()
    arr1, arr2 = _event_arrays(t1, t2)
    if _reference_enabled('time_coincidence'):
        result = _native_coincidence(
            'time_coincidence', (arr1, arr2), t2=jax.device_get(arr2),
            window=window, slide_step=slide_step)
        return _wrap_coincidence_result(t1, t2, *result)

    def _as_time_array(arr):
        if arr.ndim != 1 or not jnp.issubdtype(arr.dtype, jnp.floating):
            raise TypeError(
                "coincidence time arrays must be one-dimensional floating point"
            )
        return arr

    arr1 = _as_time_array(arr1)
    arr2 = _as_time_array(arr2)
    # Coincidence times may arrive from different detector buffers with
    # different storage precisions. Promote before applying the time window;
    # narrowing a float64 GPS vector to float32 can erase whole seconds.
    common_dtype = jnp.result_type(arr1.dtype, arr2.dtype)
    arr1 = arr1.astype(common_dtype)
    arr2 = arr2.astype(common_dtype)

    # Live matching compares one incoming time against a small template ring.
    # Bound the scratch space, reuse numeric geometry, and synchronize only
    # its output length instead of dispatching data-dependent repeats. Leave
    # larger searches and unusual window/slide values on the existing path.
    scalar_types = (int, float, np.integer, np.floating)
    if (0 < arr1.size <= 64 and arr2.size == 1 and
            isinstance(window, scalar_types) and
            isinstance(slide_step, scalar_types) and
            np.isfinite(window) and window >= 0 and
            np.isfinite(slide_step) and slide_step >= 0 and
            isinstance(arr1, jax.Array) and
            isinstance(arr2, jax.Array) and
            not isinstance(arr1, jax.core.Tracer) and
            not isinstance(arr2, jax.core.Tracer) and
            len(arr1.devices()) == 1 and arr1.devices() == arr2.devices()):
        idx1, idx2, slide, count = _time_coincidence_single_core(
            arr1, arr2, window, slide_step, sliding=bool(slide_step))
        values = _compact_time_coincidence(idx1, idx2, slide, count=int(count))
        return _wrap_coincidence_result(t1, t2, *values)

    values = _time_coincidence_general(arr1, arr2, window, slide_step)
    return _wrap_coincidence_result(t1, t2, *values)


def _time_coincidence_general(arr1, arr2, window, slide_step):
    """Retain the general eager matcher for unbounded or unusual geometry."""
    if slide_step:
        fold1 = jnp.fmod(arr1, slide_step)
        fold2 = jnp.fmod(arr2, slide_step)
    else:
        fold1 = arr1
        fold2 = arr2

    sort1 = jnp.argsort(fold1)
    sort2 = jnp.argsort(fold2)
    fold1 = fold1[sort1]
    fold2 = fold2[sort2]
    if slide_step:
        fold2 = jnp.concatenate([fold2 - slide_step, fold2, fold2 + slide_step])

    left = jnp.searchsorted(fold2, fold1 - window)
    right = jnp.searchsorted(fold2, fold1 + window)
    counts = right - left
    idx1 = jnp.repeat(sort1, counts)

    if idx1.size > 0 and sort2.size > 0:
        repeated_left = jnp.repeat(left, counts)
        group_starts = jnp.repeat(jnp.cumsum(counts) - counts, counts)
        flat_positions = jnp.arange(idx1.size, dtype=jnp.int64)
        folded_positions = repeated_left + flat_positions - group_starts
        idx2 = sort2[jnp.remainder(folded_positions, sort2.size)]
    else:
        idx2 = jnp.empty(0, dtype=jnp.int64)

    if slide_step:
        difference = (arr1[idx1] - arr2[idx2]) / slide_step
        slide = jnp.where(
            difference >= 0,
            jnp.floor(difference + 0.5),
            jnp.ceil(difference - 0.5),
        ).astype(jnp.int32)
    else:
        slide = jnp.zeros_like(idx1, dtype=jnp.int32)

    return idx1, idx2, slide


def _coincidence_cluster_time(times, slide_ids, slide, margin, multiifo,
                              sliding):
    """Construct relative times with the eager arithmetic boundaries intact."""
    barrier = jax.lax.optimization_barrier
    if multiifo:
        stacked = jnp.stack(times)
        participating = stacked > 0
        num_ifos = participating.sum(axis=0)
        first_ifo = jnp.argmax(participating.astype(jnp.int64), axis=0)
        anchor = jnp.take_along_axis(
            stacked, first_ifo[None, :], axis=0).squeeze(0)
        relative = barrier(stacked - anchor)
        relative_sum = barrier(jnp.where(
            participating, relative, jnp.zeros_like(stacked)).sum(axis=0))
        time = barrier(barrier(anchor - anchor[:1]) + barrier(
            relative_sum / num_ifos.astype(stacked.dtype)))
        if sliding:
            shift = barrier((num_ifos - 1).astype(stacked.dtype)
                            * slide_ids.astype(stacked.dtype))
            shift = barrier(shift * slide)
            shift = barrier(shift / num_ifos.astype(stacked.dtype))
            time = barrier(time + shift)
    else:
        anchor = times[0][:1]
        time = barrier(barrier(times[0] - anchor)
                       + barrier(times[1] - anchor))
        if sliding:
            time = barrier(time + barrier(slide_ids.astype(time.dtype) * slide))
        time = barrier(time * 0.5)
    # margin is evaluated on the host as in the original window * 10.
    span = barrier(barrier(time.max() - time.min()) + margin)
    return barrier(time + barrier(span * slide_ids.astype(time.dtype)))


@partial(jax.jit, static_argnames=("method", "multiifo", "sliding"))
def _coincidence_cluster_core(stat, times, slide_ids, slide, window, margin,
                              *, method, multiifo, sliding):
    """Fuse tiny coincidence geometry, selection and fixed-capacity packing."""
    time = _coincidence_cluster_time(
        times, slide_ids, slide, margin, multiifo, sliding)
    sorting, mask = _cluster_over_time_core(stat, time, window, method)
    selected = jnp.nonzero(mask, size=mask.size, fill_value=0)[0]
    return sorting[selected], jnp.sum(mask, dtype=jnp.int64)


def _can_cluster_coincs(arrays, slide, window, kwargs):
    """Keep unusual geometry and method handling on the existing public path."""
    if _reference_enabled('cluster_over_time') or arrays[0].size > 128 or any(
            not isinstance(value, jax.Array) or isinstance(value, jax.core.Tracer)
            for value in arrays):
        return False
    if any(value.dtype not in (jnp.float32, jnp.float64)
           for value in arrays[1:-1]):
        return False
    numeric = (int, float, np.integer, np.floating)
    if not isinstance(slide, numeric) or not isinstance(window, numeric):
        return False
    # longdouble is unsupported even on platforms where it occupies 8 bytes.
    if any(isinstance(value, np.floating) and
           type(value) not in (np.float16, np.float32, np.float64)
           for value in (slide, window)):
        return False
    if any(isinstance(value, (int, np.integer)) and not
           np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max
           for value in (slide, window)) or not window > 0:
        return False
    if isinstance(window, int) and not (
            np.iinfo(np.int64).min <= window * 10 <= np.iinfo(np.int64).max):
        return False
    if set(kwargs) - {"method", "argmax"}:
        return False
    method = kwargs.get("method", "python")
    if not isinstance(method, str) or method not in ("python", "cython"):
        return False
    argmax = kwargs.get("argmax", np.argmax)
    if method == "python" and argmax is not np.argmax and argmax is not jnp.argmax:
        return False
    devices = arrays[0].devices()
    return len(devices) == 1 and all(value.devices() == devices
                                   for value in arrays[1:])


def _cluster_coincidence_arrays(arrays, slide, window, kwargs, multiifo):
    """Read the compact length once and leave selected indices on-device."""
    packed, count = _coincidence_cluster_core(
        arrays[0], tuple(arrays[1:-1]), arrays[-1], slide, window, window * 10,
        method=kwargs.get("method", "python"), multiifo=multiifo,
        sliding=bool(np.isfinite(slide)))
    return packed[:int(count)]


def cluster_coincs(stat, time1, time2, timeslide_id, slide, window, **kwargs):
    """Cluster two-detector coincidences on the JAX device."""
    inputs = (stat, time1, time2, timeslide_id)
    arrays = _cluster_vectors(inputs)
    stat_arr, time1_arr, time2_arr, slide_arr = arrays
    if _reference_enabled('cluster_coincs'):
        result = _native_coincidence(
            'cluster_coincs', arrays, time1=jax.device_get(time1_arr),
            time2=jax.device_get(time2_arr),
            timeslide_id=jax.device_get(slide_arr), slide=slide, window=window,
            **kwargs)
        return _wrap_cluster_result(inputs, result)
    if time1_arr.size == 0 or time2_arr.size == 0:
        empty = jnp.empty(0, dtype=jnp.int64)
        return _wrap_cluster_result(inputs, empty)

    length = stat_arr.size
    if any(v.size != length for v in arrays[1:]):
        raise ValueError("coincidence arrays must be equal length")
    if jnp.iscomplexobj(stat_arr) or jnp.iscomplexobj(slide_arr):
        raise TypeError("coincidence statistics and slide IDs must be real")
    if not (
        jnp.issubdtype(time1_arr.dtype, jnp.floating)
        and jnp.issubdtype(time2_arr.dtype, jnp.floating)
    ):
        raise TypeError("coincidence times must be floating point")

    if _can_cluster_coincs(arrays, slide, window, kwargs):
        result = _cluster_coincidence_arrays(arrays, slide, window, kwargs, False)
        return _wrap_cluster_result(inputs, result)

    anchor = time1_arr[:1]
    time = (time1_arr - anchor) + (time2_arr - anchor)
    if np.isfinite(slide):
        time = time + slide_arr.astype(time.dtype) * slide
    time = time * 0.5

    tslide = slide_arr.astype(time.dtype)
    span = (time.max() - time.min()) + window * 10
    time = time + span * tslide
    result = cluster_over_time(stat_arr, time, window, **kwargs)
    return _wrap_cluster_result(inputs, result)


def cluster_coincs_multiifo(stat, time_coincs, timeslide_id, slide, window, **kwargs):
    """Cluster multi-detector coincidences on the JAX device."""
    inputs = (stat, *time_coincs, timeslide_id)
    arrays = _cluster_vectors(inputs)
    stat_arr = arrays[0]
    time_arrays = arrays[1:-1]
    slide_arr = arrays[-1]
    if _reference_enabled('cluster_coincs_multiifo'):
        result = _native_coincidence(
            'cluster_coincs_multiifo', arrays,
            time_coincs=jax.device_get(tuple(time_arrays)),
            timeslide_id=jax.device_get(slide_arr), slide=slide, window=window,
            **kwargs)
        return _wrap_cluster_result(inputs, result)
    if not time_arrays or any(v.size == 0 for v in time_arrays):
        empty = jnp.empty(0, dtype=jnp.int64)
        return _wrap_cluster_result(inputs, empty)

    length = stat_arr.size
    if any(v.size != length for v in arrays[1:]):
        raise ValueError("coincidence arrays must be equal length")
    if jnp.iscomplexobj(stat_arr) or jnp.iscomplexobj(slide_arr):
        raise TypeError("coincidence statistics and slide IDs must be real")
    if any(not jnp.issubdtype(v.dtype, jnp.floating) for v in time_arrays):
        raise TypeError("coincidence times must be floating point")
    if any(v.dtype != time_arrays[0].dtype for v in time_arrays):
        raise TypeError("coincidence times must use one dtype")

    if _can_cluster_coincs(arrays, slide, window, kwargs):
        result = _cluster_coincidence_arrays(arrays, slide, window, kwargs, True)
        return _wrap_cluster_result(inputs, result)

    times = jnp.stack(time_arrays)
    participating = times > 0
    num_ifos = participating.sum(axis=0)
    first_ifo = jnp.argmax(participating.astype(jnp.int64), axis=0)
    event_anchor = jnp.take_along_axis(times, first_ifo[None, :], axis=0).squeeze(0)
    relative_sum = jnp.where(
        participating, times - event_anchor, jnp.zeros_like(times)
    ).sum(axis=0)
    time_avg = event_anchor - event_anchor[:1] + relative_sum / num_ifos.astype(times.dtype)

    if np.isfinite(slide):
        time_avg = time_avg + (
            (num_ifos - 1).astype(times.dtype)
            * slide_arr.astype(times.dtype)
            * slide
            / num_ifos.astype(times.dtype)
        )

    tslide = slide_arr.astype(times.dtype)
    span = (time_avg.max() - time_avg.min()) + window * 10
    time_avg = time_avg + span * tslide
    result = cluster_over_time(stat_arr, time_avg, window, **kwargs)
    return _wrap_cluster_result(inputs, result)


def _cluster_better(first, second, stat, sentinel, nan_high=True):
    """Select the preferred maximum index between two index candidates."""
    first_valid = first != sentinel
    second_valid = second != sentinel
    first_value = stat[jnp.clip(first, 0, sentinel)]
    second_value = stat[jnp.clip(second, 0, sentinel)]

    if jnp.issubdtype(stat.dtype, jnp.floating):
        first_nan = jnp.isnan(first_value) & first_valid
        second_nan = jnp.isnan(second_value) & second_valid
        both_numeric = ~first_nan & ~second_nan
        both_nan = first_nan & second_nan
        if nan_high:
            second_better = second_nan & ~first_nan
        else:
            second_better = ~second_nan & first_nan
        second_better |= both_nan & (second < first)
        second_better |= both_numeric & (
            (second_value > first_value)
            | ((second_value == first_value) & (second < first))
        )
    else:
        second_better = (second_value > first_value) | (
            (second_value == first_value) & (second < first)
        )

    second_better = second_valid & (~first_valid | second_better)
    return jnp.where(second_better, second, first)


def _cluster_window_maxima(stat, left, right, method):
    """Return first-maximum indices for half-open ranges via segment tree."""
    length = stat.size
    sentinel = length
    stat_with_sentinel = jnp.concatenate([stat, jnp.zeros(1, dtype=stat.dtype)])
    tree_size = 1 << (length - 1).bit_length()
    tree = jnp.full(2 * tree_size, sentinel, dtype=jnp.int64)
    tree = tree.at[tree_size : tree_size + length].set(
        jnp.arange(length, dtype=jnp.int64)
    )

    nan_high = method == "python"
    width = tree_size
    while width > 1:
        updated = _cluster_better(
            tree[width : 2 * width : 2],
            tree[width + 1 : 2 * width : 2],
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        tree = tree.at[width // 2 : width].set(updated)
        width //= 2

    query_left = left + tree_size
    query_right = right + tree_size
    best = jnp.full(length, sentinel, dtype=jnp.int64)
    absent = jnp.full_like(best, sentinel)
    last_tree_index = 2 * tree_size - 1
    for _ in range(tree_size.bit_length()):
        take = (query_left < query_right) & ((query_left & 1) == 1)
        candidate = tree[jnp.clip(query_left, 0, last_tree_index)]
        best = _cluster_better(
            best,
            jnp.where(take, candidate, absent),
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        query_left = query_left + take.astype(query_left.dtype)

        take = (query_left < query_right) & ((query_right & 1) == 1)
        query_right = query_right - take.astype(query_right.dtype)
        candidate = tree[jnp.clip(query_right, 0, last_tree_index)]
        best = _cluster_better(
            best,
            jnp.where(take, candidate, absent),
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        query_left //= 2
        query_right //= 2

    if method == "cython" and jnp.issubdtype(stat.dtype, jnp.floating):
        first_is_nan = jnp.isnan(stat[left])
        best = jnp.where(first_is_nan, left, best)
    return best


@partial(jax.jit, static_argnames=("method",))
def _cluster_over_time_core(stat, time, window, method):
    """Fuse the fixed-shape clustering work; return its selection mask."""
    length = time.size
    time_sorting = jnp.argsort(time)
    sorted_stat = stat[time_sorting]
    sorted_time = time[time_sorting]
    left = jnp.searchsorted(sorted_time, sorted_time - window)
    right = jnp.searchsorted(sorted_time, sorted_time + window)
    maxima = _cluster_window_maxima(sorted_stat, left, right, method)

    # Parallelize the greedy sequential path via binary lifting
    positions = jnp.arange(length, dtype=jnp.int64)
    successor = jnp.where(
        maxima == positions,
        right,
        jnp.where(maxima > positions, maxima, positions + 1),
    )
    jump = jnp.concatenate([successor, jnp.array([length], dtype=jnp.int64)])
    steps = positions
    visited_nodes = jnp.zeros_like(steps)
    bit = 1
    while bit < length:
        visited_nodes = jnp.where(
            (steps & bit) != 0, jump[visited_nodes], visited_nodes
        )
        jump = jump[jump]
        bit <<= 1

    visited = jnp.zeros(length + 1, dtype=bool)
    visited = visited.at[visited_nodes].set(True)
    return time_sorting, visited[:length] & (maxima == positions)


def cluster_over_time(stat, time, window, method="python", argmax=np.argmax):
    """Cluster transient events with fused JAX numerical work."""
    _ensure_x64()
    stat_arr, time_arr = _event_arrays(stat, time)
    if _reference_enabled('cluster_over_time'):
        result = _native_coincidence(
            'cluster_over_time', (stat_arr, time_arr),
            time=jax.device_get(time_arr), window=window, method=method,
            argmax=np.argmax if argmax is jnp.argmax else argmax)
        return _wrap_cluster_result((stat, time), result)

    if method not in ("python", "cython"):
        raise ValueError(f"Do not recognize method {method}")

    def _as_tensor(value, arr):
        if arr is None:
            # Statistics and GPS times may have different precisions.  In
            # particular, casting float64 times to float32 loses subsecond
            # spacing at ordinary GPS epochs and changes clustering.
            arr = _event_arrays(value)[0]
        if arr.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        return arr

    stat_arr = _as_tensor(stat, stat_arr)
    time_arr = _as_tensor(time, time_arr)
    if stat_arr.size != time_arr.size:
        raise ValueError("cluster statistic and time arrays must be equal length")
    if jnp.iscomplexobj(stat_arr) or not jnp.issubdtype(time_arr.dtype, jnp.floating):
        raise TypeError("cluster statistics must be real and times floating point")

    if time_arr.size == 0:
        empty = jnp.empty(0, dtype=jnp.int64)
        return _wrap_coincidence_result(stat, time, empty)[0]
    if method == "python" and argmax not in (np.argmax, jnp.argmax):
        raise NotImplementedError("JAX clustering supports np.argmax or jnp.argmax")
    if window <= 0:
        raise ValueError("cluster window must be positive")

    time_sorting, mask = _cluster_over_time_core(
        stat_arr, time_arr, window, method)
    # Selection length is data-dependent, so keep it outside the compiled core.
    keep = jnp.flatnonzero(mask)
    result = time_sorting[keep]
    return _wrap_coincidence_result(stat, time, result)[0]


