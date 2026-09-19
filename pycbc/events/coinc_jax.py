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

"""JAX backend for coincidence construction and clustering.

Provides device-resident coincidence finding and clustering without host
synchronization or Cython dependencies, utilizing functional XLA array
transformations, segment trees, and binary lifting.
"""

import numpy as np
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
    arrays = [_as_jax_array(value) for value in values]
    reference = next((v for v in arrays if v is not None), None)
    if reference is None:
        raise TypeError("the JAX backend requires a JAX-backed input")

    result = []
    for value, arr in zip(values, arrays):
        if arr is None:
            host = value.numpy() if isinstance(value, Array) else value
            arr = jnp.asarray(host, dtype=reference.dtype)
        if arr.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        result.append(arr)
    return result


def time_coincidence(t1, t2, window, slide_step=0):
    """Find coincidences by time window on JAX device without host copies."""
    _ensure_x64()
    arr1 = _as_jax_array(t1)
    arr2 = _as_jax_array(t2)
    reference = arr1 if arr1 is not None else arr2
    if reference is None:
        raise TypeError("the JAX backend requires a JAX-backed input")

    def _as_time_array(value, arr):
        if arr is None:
            host = value.numpy() if isinstance(value, Array) else value
            arr = jnp.asarray(host, dtype=reference.dtype)
        if arr.dtype != reference.dtype:
            raise TypeError("coincidence time arrays must use one dtype")
        if arr.ndim != 1 or not jnp.issubdtype(arr.dtype, jnp.floating):
            raise TypeError(
                "coincidence time arrays must be one-dimensional floating point"
            )
        return arr

    arr1 = _as_time_array(t1, arr1)
    arr2 = _as_time_array(t2, arr2)

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

    return _wrap_coincidence_result(t1, t2, idx1, idx2, slide)


def cluster_coincs(stat, time1, time2, timeslide_id, slide, window, **kwargs):
    """Cluster two-detector coincidences on the JAX device."""
    inputs = (stat, time1, time2, timeslide_id)
    arrays = _cluster_vectors(inputs)
    stat_arr, time1_arr, time2_arr, slide_arr = arrays
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


def cluster_over_time(stat, time, window, method="python", argmax=np.argmax):
    """Cluster transient events entirely on the JAX device.

    Evaluates the greedy sequential clustering loop in parallel via binary
    lifting on the successor DAG without host transfers.
    """
    _ensure_x64()
    stat_arr = _as_jax_array(stat)
    time_arr = _as_jax_array(time)
    reference = stat_arr if stat_arr is not None else time_arr
    if reference is None:
        raise TypeError("the JAX backend requires a JAX-backed input")

    if method not in ("python", "cython"):
        raise ValueError(f"Do not recognize method {method}")

    def _as_tensor(value, arr):
        if arr is None:
            host = value.numpy() if isinstance(value, Array) else value
            arr = jnp.asarray(host, dtype=reference.dtype)
        if arr.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        return arr

    stat_arr = _as_tensor(stat, stat_arr)
    time_arr = _as_tensor(time, time_arr)
    if stat_arr.size != time_arr.size:
        raise ValueError("cluster statistic and time arrays must be equal length")
    if jnp.iscomplexobj(stat_arr) or not jnp.issubdtype(time_arr.dtype, jnp.floating):
        raise TypeError("cluster statistics must be real and times floating point")

    length = time_arr.size
    if length == 0:
        empty = jnp.empty(0, dtype=jnp.int64)
        return _wrap_coincidence_result(stat, time, empty)[0]
    if method == "python" and argmax not in (np.argmax, jnp.argmax):
        raise NotImplementedError("JAX clustering supports np.argmax or jnp.argmax")
    if window <= 0:
        raise ValueError("cluster window must be positive")

    time_sorting = jnp.argsort(time_arr)
    sorted_stat = stat_arr[time_sorting]
    sorted_time = time_arr[time_sorting]
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
    keep = jnp.flatnonzero(visited[:length] & (maxima == positions))
    result = time_sorting[keep]
    return _wrap_coincidence_result(stat, time, result)[0]
