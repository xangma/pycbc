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

Provides JAX array operations for coincidence finding and clustering,
utilizing functional XLA transformations, segment trees, and binary lifting.
"""

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, is_jax_array
from pycbc.types.backend import backend_array, is_backend


def pick_best_coinc_jax(coinc_results, logger):
    """Select the best coincidence using device-resident IFAR/stat arrays."""
    candidates = [result for result in coinc_results
                  if "coinc_possible" in result and
                  "foreground/ifar" in result]
    trials = sum("coinc_possible" in result for result in coinc_results)
    if not candidates:
        return coinc_results[0]
    ifar = jnp.asarray([result["foreground/ifar"] for result in candidates])
    stat = jnp.asarray([result["foreground/stat"] for result in candidates])
    stat = stat.reshape((len(candidates), -1))[:, 0]
    best_ifar = jnp.max(ifar)
    best = jnp.argmax(jnp.where(ifar == best_ifar, stat, -jnp.inf))
    result = candidates[int(best)]
    result["foreground/ifar"] = best_ifar / float(trials)
    logger.info("Found %s coinc with ifar %s", result["foreground/type"],
                result["foreground/ifar"])
    return result


def add_singles_to_buffer_jax(estimator, results, ifos, logger):
    """Compute and append live singles using only JAX numeric operations."""
    if not estimator.singles:
        estimator.set_singles_buffer(results)
    if not estimator.singles:
        return {}
    logger.info("adding singles to the background estimate...")
    updated_indices = {}
    for ifo in ifos:
        trigs = results[ifo]
        if len(trigs["snr"]) > 0:
            trigsc = dict(trigs)
            trigsc["ifo"] = ifo
            trigsc["chisq"] = jnp.asarray(trigs["chisq"]) * jnp.asarray(
                trigs["chisq_dof"])
            trigsc["chisq_dof"] = (jnp.asarray(trigs["chisq_dof"]) + 2) / 2
            trigsc = {key: _numeric_device_value(value)
                      for key, value in trigsc.items()}
            single_stat = estimator.stat_calculator.single(trigsc)
        else:
            single_stat = jnp.array([], dtype=estimator.stat_calculator.single_dtype)
        trigs["stat"] = single_stat
        data = {key: _numeric_device_value(value)
                for key, value in trigs.items()}
        estimator.singles[ifo].add(trigs["template_id"], data)
        updated_indices[ifo] = trigs["template_id"]
    return updated_indices


def _numeric_device_value(value):
    """Move numeric trigger columns to JAX; retain metadata on the host."""
    if isinstance(value, str):
        return value
    dtype = getattr(value, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind in "biufc":
        return jnp.asarray(value)
    return value


class JAXCoincExpireBuffer:
    """JAX resident rolling coincidence statistic buffer.

    Coincident statistics and their expiration clocks stay on the selected
    JAX device, alongside the columnar singles buffers.
    """

    def __init__(self, expiration, ifos, initial_size=2**20, dtype=np.float32):
        self.expiration = expiration
        self.ifos = ifos
        self.buffer = jnp.zeros(initial_size, dtype=dtype)
        self.timer = {ifo: jnp.zeros(initial_size, dtype=jnp.int32) for ifo in ifos}
        self.index = 0
        self.time = {ifo: 0 for ifo in ifos}

    def __len__(self):
        return self.index

    @property
    def nbytes(self):
        return self.buffer.size * self.buffer.dtype.itemsize + sum(
            value.size * value.dtype.itemsize for value in self.timer.values()
        )

    def increment(self, ifos):
        self.add(jnp.empty(0, dtype=self.buffer.dtype), {}, ifos)

    def remove(self, num):
        self.index -= num

    def add(self, values, times, ifos):
        values = jnp.asarray(values, dtype=self.buffer.dtype)
        for ifo in ifos:
            self.time[ifo] += 1
        needed = self.index + values.size
        if needed > self.buffer.size:
            size = max(needed, self.buffer.size * 2)
            self.buffer = jnp.pad(self.buffer, (0, size - self.buffer.size))
            self.timer = {
                ifo: jnp.pad(clock, (0, size - clock.size))
                for ifo, clock in self.timer.items()
            }
        self.buffer = self.buffer.at[self.index:self.index + values.size].set(values)
        for ifo in self.ifos:
            if values.size:
                self.timer[ifo] = self.timer[ifo].at[
                    self.index:self.index + values.size
                ].set(jnp.asarray(times[ifo], dtype=jnp.int32))
        self.index = needed
        if ifos:
            keep = jnp.ones(self.index, dtype=bool)
            for ifo in ifos:
                keep &= self.timer[ifo][:self.index] >= self.time[ifo] - self.expiration
            kept = jnp.flatnonzero(keep)
            self.buffer = self.buffer.at[:kept.size].set(self.buffer[kept])
            for ifo in self.ifos:
                self.timer[ifo] = self.timer[ifo].at[:kept.size].set(
                    self.timer[ifo][kept]
                )
            self.index = kept.size

    def num_greater(self, value):
        return int(jnp.sum(self.buffer[:self.index] > value))

    @property
    def data(self):
        return self.buffer[:self.index]


class JAXMultiRingBuffer:
    """Columnar JAX ring buffers for live single-detector triggers."""

    def __init__(self, num_rings, max_time, dtype=None, **kwargs):
        self.max_time = max_time
        self.num_rings = num_rings
        self.buffer = [None] * num_rings
        self.buffer_expire = [jnp.empty(0, dtype=jnp.int32) for _ in range(num_rings)]
        self.valid_ends = [0] * num_rings
        self.time = 0

    @property
    def filled_time(self):
        return min(self.time, self.max_time)

    @property
    def nbytes(self):
        return sum(
            value.size * value.dtype.itemsize
            for row in self.buffer if row is not None for value in row.values()
        )

    def add(self, indices, values):
        indices = np.asarray(indices)
        for ring in set(map(int, indices)):
            self._prune_ring(ring)
        for pos, ring in enumerate(indices):
            ring = int(ring)
            row = {key: _numeric_device_value(value[pos])
                   for key, value in values.items()}
            if self.buffer[ring] is None:
                self.buffer[ring] = {
                    key: (_numeric_device_value(value[pos:pos + 1])
                          if np.dtype(getattr(value, "dtype", object)).kind
                          in "biufc" else np.asarray(value[pos:pos + 1]))
                    for key, value in values.items()}
                self.buffer_expire[ring] = jnp.asarray([self.time], dtype=jnp.int32)
            else:
                appended = {}
                for key, new_value in row.items():
                    old_value = self.buffer[ring][key]
                    if is_jax_array(old_value):
                        appended[key] = jnp.concatenate(
                            (old_value, jnp.asarray(new_value)[None]))
                    else:
                        appended[key] = np.concatenate(
                            (old_value, np.asarray(new_value)[None]))
                self.buffer[ring] = appended
                self.buffer_expire[ring] = jnp.concatenate(
                    (self.buffer_expire[ring], jnp.asarray([self.time], dtype=jnp.int32))
                )
            self.valid_ends[ring] += 1
        self.time += 1

    def _prune_ring(self, buffer_index):
        row = self.buffer[buffer_index]
        if row is None:
            return
        keep = self._keep_mask(buffer_index)
        if bool(jnp.all(keep)):
            return
        host_keep = np.asarray(keep)
        self.buffer[buffer_index] = {
            key: (value[keep] if is_jax_array(value)
                  else value[host_keep])
            for key, value in row.items()
        }
        self.buffer_expire[buffer_index] = self.buffer_expire[buffer_index][keep]
        self.valid_ends[buffer_index] = len(self.buffer_expire[buffer_index])

    def data(self, buffer_index):
        # Compact on access or append, as native ring buffers do. This bounds
        # growing storage without synchronizing every template every block.
        self._prune_ring(buffer_index)
        row = self.buffer[buffer_index]
        if row is None:
            return {}
        return row

    def expire_vector(self, buffer_index):
        self._prune_ring(buffer_index)
        return self.buffer_expire[buffer_index]

    def _keep_mask(self, buffer_index):
        return (self.buffer_expire[buffer_index] >=
                self.time - self.max_time)

    def discard_last(self, indices):
        for ring in np.asarray(indices):
            ring = int(ring)
            if self.buffer[ring] is not None:
                self.buffer[ring] = {
                    key: value[:-1] for key, value in self.buffer[ring].items()
                }
                self.buffer_expire[ring] = self.buffer_expire[ring][:-1]
                self.valid_ends[ring] -= 1


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


def _host_array(value):
    """Convert a public boundary value without changing its dtype."""
    return value.numpy() if isinstance(value, Array) else value


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

    result = []
    for value, arr in zip(values, arrays):
        if arr is None:
            host = value.numpy() if isinstance(value, Array) else value
            # Ranking statistics, GPS times, and slide IDs may have different
            # dtypes. In particular, float64 GPS times lose subsecond spacing
            # if a float32 statistic supplies the reference array.
            arr = (jax.device_put(host, reference.device)
                   if reference is not None else jnp.asarray(host))
        if arr.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        result.append(arr)
    return result


def time_coincidence(t1, t2, window, slide_step=0):
    """Find coincidences on the JAX device after converting host inputs."""
    _ensure_x64()
    arr1 = _as_jax_array(t1)
    arr2 = _as_jax_array(t2)
    reference = arr1 if arr1 is not None else arr2

    def _as_time_array(value, arr):
        if arr is None:
            host = _host_array(value)
            arr = (jax.device_put(host, reference.device)
                   if reference is not None else jnp.asarray(host))
        if arr.ndim != 1 or not jnp.issubdtype(arr.dtype, jnp.floating):
            raise TypeError(
                "coincidence time arrays must be one-dimensional floating point"
            )
        return arr

    arr1 = _as_time_array(t1, arr1)
    arr2 = _as_time_array(t2, arr2)
    # Coincidence times may arrive from different detector buffers with
    # different storage precisions. Promote before applying the time window;
    # narrowing a float64 GPS vector to float32 can erase whole seconds.
    common_dtype = jnp.result_type(arr1.dtype, arr2.dtype)
    arr1 = arr1.astype(common_dtype)
    arr2 = arr2.astype(common_dtype)

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

    if method not in ("python", "cython"):
        raise ValueError(f"Do not recognize method {method}")

    def _as_tensor(value, arr):
        if arr is None:
            # Statistics and GPS times may have different precisions.  In
            # particular, casting float64 times to float32 loses subsecond
            # spacing at ordinary GPS epochs and changes clustering.
            arr = jnp.asarray(_host_array(value))
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


def find_coincs_jax(estimator, results, valid_ifos):
    """Look for coincs within the set of single triggers

    Parameters
    ----------
    results: dict
        Dictionary of dictionaries indexed by ifo and keys such as 'snr',
        'chisq', etc. The specific format is determined by the
        LiveBatchMatchedFilter class.
    valid_ifos: list of strs
        List of ifos for which new triggers might exist. This must be a
        subset of estimator.ifos. If an ifo is in estimator.ifos but not in this list
        either the ifo is down, or its data has been flagged as "bad".

    Returns
    -------
    num_background: int
        Number of time shifted coincidences found.
    coinc_results: dict of arrays
        A dictionary of arrays containing the coincident results.
    """
    from .coinc import cluster_coincs, time_coincidence, ppdets, logger
    from pycbc import conversions as conv
    # For each new single detector trigger find the allowed coincidences
    # Record the template and the index of the single trigger that forms
    # each coincidence

    # Initialize
    cstat = []
    offsets = []
    ctimes = {estimator.ifos[0]:[], estimator.ifos[1]:[]}
    single_expire = {estimator.ifos[0]:[], estimator.ifos[1]:[]}
    template_ids = []
    trigger_ids = {estimator.ifos[0]: [], estimator.ifos[1]: []}

    # Calculate all the permutations of coincident triggers for each
    # new single detector trigger collected
    # Currently only two detectors are supported.
    # For each ifo, check its newly added triggers for (zerolag and time
    # shift) coincs with all currently stored triggers in the other ifo.
    # Do this by keeping the ifo with new triggers fixed and time shifting
    # the other ifo. The list 'shift_vec' must be in the same order as
    # estimator.ifos and contain -1 for the shift_ifo / 0 for the fixed_ifo.
    for fixed_ifo, shift_ifo, shift_vec in zip(
        [estimator.ifos[0], estimator.ifos[1]],
        [estimator.ifos[1], estimator.ifos[0]],
        [[0, -1], [-1, 0]]
    ):
        if fixed_ifo not in valid_ifos:
            # This ifo is not online now, so no new triggers or coincs
            continue
        # Find newly added triggers in fixed_ifo
        trigs = results[fixed_ifo]
        # Calculate mchirp as a vectorized operation
        mchirps = conv.mchirp_from_mass1_mass2(
            jnp.asarray(trigs['mass1']), jnp.asarray(trigs['mass2'])
        )
        # Loop over them one trigger at a time
        for i in range(len(trigs['end_time'])):
            trig_stat = trigs['stat'][i]
            trig_time = trigs['end_time'][i]
            template = trigs['template_id'][i]
            mchirp = mchirps[i]

            # Get current shift_ifo triggers in the same template
            shifted = estimator.singles[shift_ifo].data(template)
            times = shifted.get('end_time', jnp.empty(0, dtype=jnp.float64))
            stats = shifted.get('stat', jnp.empty(0, dtype=jnp.float32))

            # Perform coincidence. i1 is the list of trigger indices in the
            # shift_ifo which make coincs, slide is the corresponding slide
            # index.
            # (The second output would just be a list of zeroes as we only
            # have one trigger in the fixed_ifo.)
            i1, _, slide = time_coincidence(times,
                             jnp.array(trig_time, ndmin=1,
                             dtype=jnp.float64),
                             estimator.time_window,
                             estimator.timeslide_interval)

            # Make a copy of the fixed ifo trig_stat for each coinc.
            # NB for some statistics the "stat" entry holds more than just
            # a ranking number. E.g. for the phase time consistency test,
            # it must also contain the phase, time and sensitivity.
            if estimator.trig_stat_memory is None:
                estimator.trig_stat_memory = jnp.zeros(
                    1,
                    dtype=trig_stat.dtype
                )
            while len(estimator.trig_stat_memory) < len(i1):
                estimator.trig_stat_memory = jnp.pad(
                    estimator.trig_stat_memory,
                    (0, len(estimator.trig_stat_memory)),
                )
            estimator.trig_stat_memory = estimator.trig_stat_memory.at[:len(i1)].set(trig_stat)

            # Force data into form needed by stat.py and then compute the
            # ranking statistic values.
            sngls_list = [[fixed_ifo, estimator.trig_stat_memory[:len(i1)]],
                          [shift_ifo, stats[i1]]]

            c = estimator.stat_calculator.rank_stat_coinc(
                sngls_list,
                slide,
                estimator.timeslide_interval,
                shift_vec,
                time_addition=estimator.coinc_window_pad,
                mchirp=mchirp,
                dets=estimator.dets
            )

            # Store data about new triggers: slide index, stat value and
            # times.
            offsets.append(slide)
            # The native coincidence path materializes this ranking array as
            # float64, even when the device-resident single-trigger columns
            # are float32.  Keep that public output contract at the backend
            # boundary; the ranking calculation above remains JAX-native.
            cstat.append(jnp.asarray(c, dtype=jnp.float64))
            ctimes[shift_ifo].append(times[i1])
            fixed_times = jnp.full(len(c), trig_time, dtype=jnp.float64)
            ctimes[fixed_ifo].append(fixed_times)

            # As background triggers are removed after a certain time, we
            # need to log when this will be for new background triggers.
            single_expire[shift_ifo].append(
                estimator.singles[shift_ifo].expire_vector(template)[i1]
            )
            single_expire[fixed_ifo].append(jnp.full(
                len(c), estimator.singles[fixed_ifo].time - 1,
                dtype=jnp.int32
            ))

            # Save the template and trigger ids to keep association
            # to singles. The trigger was just added so it must be in
            # the last position: we mark this with -1 so the
            # slicing picks the right point
            template_ids.append(jnp.zeros(len(c), dtype=jnp.int32) + template)
            trigger_ids[shift_ifo].append(i1)
            trigger_ids[fixed_ifo].append(jnp.zeros(len(c)) - 1)

    cstat = (jnp.concatenate(cstat) if cstat else
             jnp.empty(0, dtype=jnp.float64))
    template_ids = (jnp.concatenate(template_ids).astype(jnp.int32)
                    if template_ids else jnp.empty(0, dtype=jnp.int32))
    for ifo in valid_ifos:
        trigger_ids[ifo] = (jnp.concatenate(trigger_ids[ifo]).astype(jnp.int32)
                            if trigger_ids[ifo] else
                            jnp.empty(0, dtype=jnp.int32))

    logger.info(
        "%s: %s background and zerolag coincs",
        ppdets(estimator.ifos, "-"), len(cstat)
    )

    # Cluster the triggers we've found
    # (both zerolag and shifted are handled together)
    num_zerolag = 0
    num_background = 0
    if len(cstat) > 0:
        offsets = jnp.concatenate(offsets)
        ctime0 = jnp.concatenate(ctimes[estimator.ifos[0]]).astype(jnp.float64)
        ctime1 = jnp.concatenate(ctimes[estimator.ifos[1]]).astype(jnp.float64)
        logger.info("Clustering %s coincs", ppdets(estimator.ifos, "-"))
        cidx = cluster_coincs(cstat, ctime0, ctime1, offsets,
                              estimator.timeslide_interval,
                              estimator.analysis_block + 2*estimator.time_window,
                              method='cython')
        offsets = offsets[cidx]
        zerolag_idx = (offsets == 0)
        bkg_idx = (offsets != 0)

        for ifo in estimator.ifos:
            single_expire[ifo] = jnp.concatenate(single_expire[ifo])
            single_expire[ifo] = single_expire[ifo][cidx][bkg_idx]

        estimator.coincs.add(cstat[cidx][bkg_idx], single_expire, valid_ifos)
        num_zerolag = zerolag_idx.sum()
        num_background = bkg_idx.sum()
    elif len(valid_ifos) > 0:
        estimator.coincs.increment(valid_ifos)

    # Collect coinc results for saving
    coinc_results = {}
    # Save information about zerolag triggers
    if num_zerolag > 0:
        idx = cidx[zerolag_idx][0]
        zerolag_cstat = cstat[cidx][zerolag_idx]
        ifar, ifar_sat = estimator.ifar(zerolag_cstat[0])
        zerolag_results = {
            'foreground/ifar': ifar,
            'foreground/ifar_saturated': ifar_sat,
            'foreground/stat': zerolag_cstat,
            'foreground/type': '-'.join(estimator.ifos)
        }
        template = template_ids[idx]
        for ifo in estimator.ifos:
            trig_id = trigger_ids[ifo][idx]
            stored = estimator.singles[ifo].data(template)
            for key, value in stored.items():
                path = f'foreground/{ifo}/{key}'
                zerolag_results[path] = value[trig_id]
        coinc_results.update(zerolag_results)

    # Save some summary statistics about the background
    coinc_results['background/time'] = jnp.array([estimator.background_time])
    coinc_results['background/count'] = len(estimator.coincs.data)

    # Save all the background triggers
    if estimator.return_background:
        coinc_results['background/stat'] = estimator.coincs.data

    return num_background, coinc_results
