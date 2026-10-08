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

"""JAX coincidence primitives and device-resident ring buffers."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, _reference_enabled, is_jax_array
from .ranking_jax import _event_arrays
from pycbc.benchmark import host_stage


@jax.jit
def _gather_singles_columns(columns, indices):
    """Select numeric rows together without changing each column's dtype."""
    return tuple(value[indices] for value in columns)


@jax.jit
def _append_selected_singles_columns(selected, previous, expiry, clock):
    """Append selected rows without specializing on the full incoming batch."""
    clocks = jnp.full(selected[0].shape[0], clock, dtype=jnp.int32)
    if previous:
        selected = tuple(jnp.concatenate((old, new))
                         for old, new in zip(previous, selected))
    return selected, jnp.concatenate((expiry, clocks))


def _append_singles_columns(columns, previous, expiry, indices, clock):
    """Gather separately so appends reuse selected-row and history shapes."""
    selected = _gather_singles_columns(columns, indices)
    return _append_selected_singles_columns(selected, previous, expiry, clock)


@partial(jax.jit, static_argnames=("counts",))
def _gather_singles_column_groups(columns, indices, *, counts):
    """Select several rings with one index vector, retaining column dtypes."""
    selected = tuple(value[indices] for value in columns)
    rows = []
    start = 0
    for count in counts:
        rows.append(tuple(value[start:start + count] for value in selected))
        start += count
    return tuple(rows)


@jax.jit
def _append_selected_singles_groups(selected, previous, expiries, clock):
    """Reuse selected-row/history shapes independently of incoming lengths."""
    return tuple(_append_selected_singles_columns.__wrapped__(row, old, expiry,
                                                             clock)
                 for row, old, expiry in zip(selected, previous, expiries))


def _prepare_singles_ring_group(buffer, rings, columns, numeric_keys,
                                numeric_values, device):
    """Prepare at most eight resident rings; commit state in the caller."""
    with host_stage('coinc_singles_group_host_prepare'):
        if (type(buffer) is not JAXMultiRingBuffer or
                device is None or not 1 < len(rings) <= 8 or
                any(not 0 <= ring < buffer.num_rings for ring, _ in rings) or
                len({ring for ring, _ in rings}) != len(rings) or
                sum(len(selected) for _, selected in rings) > 4096):
            return {}
        # Inspect only ordinary storage before checking future rings. Malformed
        # or custom rows must still fail at their original serial update point.
        if (any(type(values) is not list for values in
                (buffer.buffer, buffer.buffer_expire, buffer._expire_times)) or
                any(ring >= len(values) for ring, _ in rings for values in
                    (buffer.buffer, buffer.buffer_expire,
                     buffer._expire_times)) or
                any(buffer.buffer[ring] is not None and
                    type(buffer.buffer[ring]) is not dict for ring, _ in rings)):
            return {}
        if (any(type(buffer._expire_times[ring]) is not np.ndarray
                for ring, _ in rings) or
                any(type(value) is not np.ndarray
                    for ring, _ in rings if buffer.buffer[ring] is not None
                    for key, value in buffer.buffer[ring].items()
                    if key not in numeric_keys)):
            # Concatenation hooks can change a later ring before its turn.
            return {}
        selection = np.concatenate(tuple(np.asarray(selected, dtype=np.int64)
                                         for _, selected in rings))
    with host_stage('coinc_singles_group_index_dispatch', rings=len(rings)):
        indices = jnp.asarray(selection)
    with host_stage('coinc_singles_group_host_prepare', rings=len(rings)):
        # The packed size is a conservative bound on each ring's addition.
        # Reuse the existing guard so unusual storage retains its old path.
        if not all(_can_append_singles_columns(
                buffer, ring, columns, numeric_keys, indices, device)
                   for ring, _ in rings):
            return {}
        previous = tuple(() if buffer.buffer[ring] is None else
                         tuple(buffer.buffer[ring][key] for key in numeric_keys)
                         for ring, _ in rings)
        expiries = tuple(buffer.buffer_expire[ring] for ring, _ in rings)
        counts = tuple(len(selected) for _, selected in rings)
    # These ranges cover Python dispatch, not completion of asynchronous work.
    with host_stage('coinc_singles_group_append_dispatch', rings=len(rings)):
        selected = _gather_singles_column_groups(numeric_values, indices,
                                                counts=counts)
        appended = _append_selected_singles_groups(
            selected, previous, expiries, np.int32(buffer.time))
    return dict(zip((ring for ring, _ in rings), appended))


def _can_append_singles_columns(buffer, ring, columns, numeric_keys,
                                indices, device):
    """Fuse bounded resident appends without changing casts or error paths."""
    if device is None or indices.devices() != {device}:
        return False
    if not 0 < len(numeric_keys) <= 32:
        return False
    if any(columns[key].shape[0] > 4096 for key in numeric_keys):
        return False
    if not isinstance(buffer.time, (int, np.integer)):
        return False
    limits = np.iinfo(np.int32)
    if not limits.min <= buffer.time <= limits.max:
        return False
    expiry = buffer.buffer_expire[ring]
    if (not isinstance(expiry, jax.Array) or
            isinstance(expiry, jax.core.Tracer) or expiry.ndim != 1 or
            expiry.dtype != jnp.int32 or expiry.devices() != {device}):
        return False
    clocks = buffer._expire_times[ring]
    if (not isinstance(clocks, np.ndarray) or clocks.ndim != 1 or
            clocks.dtype != np.int64 or clocks.size != expiry.size or
            expiry.size + indices.size > 4096):
        return False
    previous = buffer.buffer[ring]
    if previous is None:
        return expiry.size == 0
    for key, new in columns.items():
        if key not in previous:
            return False
        old = previous[key]
        if key in numeric_keys:
            if (not isinstance(old, jax.Array) or
                    isinstance(old, jax.core.Tracer) or
                    old.devices() != {device} or old.dtype != new.dtype):
                return False
        elif not isinstance(old, np.ndarray):
            return False
        if (old.ndim != new.ndim or old.ndim == 0 or
                old.shape[0] != expiry.size or
                old.shape[1:] != new.shape[1:]):
            return False
    return True


@partial(jax.jit, static_argnames=("work_size", "active_columns"))
def _append_expire_coinc_buffer(buffer, timers, values, times, index,
                                thresholds, *, work_size, active_columns):
    """Append and stably prune a bounded prefix, preserving backing tails."""
    columns = (buffer,) + timers
    additions = (values,) + times
    needed = index + values.size
    if not active_columns:
        # No expiration: retain the original full-array append semantics.
        return tuple(jax.lax.dynamic_update_slice(
            column, jnp.broadcast_to(addition, values.shape), (index,))
                     for column, addition in zip(columns, additions)), needed

    appended = tuple(jax.lax.dynamic_update_slice(
        column[:work_size], jnp.broadcast_to(addition, values.shape), (index,))
        for column, addition in zip(columns, additions))
    positions = jnp.arange(work_size)
    keep = positions < needed
    for column, threshold in zip(active_columns, thresholds):
        keep &= appended[column + 1] >= threshold
    kept = jnp.nonzero(keep, size=work_size, fill_value=0)[0]
    count = jnp.sum(keep, dtype=jnp.int64)
    compacted = tuple(jnp.where(positions < count, column[kept], column)
                      for column in appended)
    return tuple(jax.lax.dynamic_update_slice(column, prefix, (0,))
                 for column, prefix in zip(columns, compacted)), count


def _can_append_expire_coinc_buffer(buffer, values, times, ifos):
    """Admit ordinary resident live buffers; retain eager unusual cases."""
    if not isinstance(ifos, (list, tuple)) or not isinstance(
            buffer.ifos, (list, tuple)):
        return False
    if tuple(buffer.ifos) != tuple(buffer.timer):
        return False
    if not isinstance(buffer.index, (int, np.integer)):
        return False
    needed = buffer.index + values.size
    if not 0 <= buffer.index <= needed <= min(buffer.buffer.size, 4096):
        return False
    if not isinstance(buffer.expiration, (int, np.integer)):
        return False
    limits = np.iinfo(np.int32)
    for ifo in ifos:
        if ifo not in buffer.timer or not isinstance(
                buffer.time[ifo], (int, np.integer)):
            return False
        threshold = buffer.time[ifo] - buffer.expiration
        if not limits.min <= threshold <= limits.max:
            return False
    columns = (buffer.buffer, values) + tuple(buffer.timer.values())
    if values.size:
        if not isinstance(times, dict) or any(ifo not in times
                                             for ifo in buffer.ifos):
            return False
        additions = tuple(times[ifo] for ifo in buffer.ifos)
        if any(not is_jax_array(value) or value.ndim > 1 or
               (value.ndim and value.shape != values.shape)
               for value in additions):
            return False
        columns += additions
    if any(not is_jax_array(column) or isinstance(column, jax.core.Tracer)
           for column in columns):
        return False
    if any(column.ndim != 1 or column.size != buffer.buffer.size
           for column in (buffer.buffer,) + tuple(buffer.timer.values())):
        return False
    if not buffer.buffer.size or any(clock.dtype != jnp.int32
                                     for clock in buffer.timer.values()):
        return False
    if values.ndim != 1:
        return False
    devices = columns[0].devices()
    return len(devices) == 1 and all(column.devices() == devices
                                   for column in columns[1:])


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
        values = _event_arrays(values, device=self.buffer.device)[0].astype(
            self.buffer.dtype)
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
        if _can_append_expire_coinc_buffer(self, values, times, ifos):
            clocks = tuple(_event_arrays(times[ifo], device=self.buffer.device)[0].astype(jnp.int32)
                           if values.size else jnp.empty(0, dtype=jnp.int32)
                           for ifo in self.ifos)
            active = tuple(self.ifos.index(ifo) for ifo in ifos)
            thresholds = tuple(np.int32(self.time[ifo] - self.expiration)
                               for ifo in ifos)
            work_size = min(self.buffer.size,
                            1 << (int(max(needed, 1)) - 1).bit_length())
            columns, count = _append_expire_coinc_buffer(
                self.buffer, tuple(self.timer.values()), values, clocks,
                np.int64(self.index), thresholds, work_size=work_size,
                active_columns=active)
            self.buffer = columns[0]
            self.timer = dict(zip(self.ifos, columns[1:]))
            self.index = int(count) if ifos else needed
            return
        self.buffer = self.buffer.at[self.index:self.index + values.size].set(values)
        for ifo in self.ifos:
            if values.size:
                self.timer[ifo] = self.timer[ifo].at[
                    self.index:self.index + values.size
                ].set(_event_arrays(times[ifo], device=self.buffer.device)[0].astype(jnp.int32))
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
        # Public scalar API; resident live IFAR evaluation bypasses this read.
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
        # Empty JAX arrays are immutable, so untouched rings can share one.
        empty = jnp.empty(0, dtype=jnp.int32)
        self.device = getattr(empty, 'device', None)
        self.buffer_expire = [empty] * num_rings
        # These insertion clocks are control metadata, independent of the
        # scientific columns. They let pruning avoid a device scalar read.
        self._expire_times = [np.empty(0, dtype=np.int64)] * num_rings
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
        positions = {}
        for pos, ring in enumerate(indices):
            ring = int(ring)
            positions.setdefault(ring, []).append(pos)
        columns = {
            key: (_event_arrays(value, device=self.device)[0]
                  if np.dtype(getattr(value, "dtype", object)).kind in "biufc"
                  else np.asarray(value))
            for key, value in values.items()
        } if positions else {}
        numeric_keys = tuple(key for key, value in columns.items()
                             if is_jax_array(value))
        numeric_values = tuple(columns[key] for key in numeric_keys)
        # Keep prior per-column placement and validation for mixed devices,
        # tracers or malformed columns. Casts above remain outside the core.
        gather_device = None
        if (numeric_values and
                all(isinstance(value, jax.Array) and
                    not isinstance(value, jax.core.Tracer)
                    for value in numeric_values) and
                all(value.ndim > 0 and value.shape[0] >= indices.size
                    for value in columns.values())):
            devices = numeric_values[0].devices()
            if (len(devices) == 1 and
                    all(value.devices() == devices
                        for value in numeric_values[1:])):
                gather_device = next(iter(devices))
        rings = tuple(positions.items())
        group = {}
        for position, (ring, selected) in enumerate(rings):
            if position % 8 == 0:
                group = _prepare_singles_ring_group(
                    self, rings[position:position + 8], columns, numeric_keys,
                    numeric_values, gather_device)
            host_indices = np.asarray(selected, dtype=np.int64)
            prepared = group.get(ring)
            if prepared is None:
                device_indices = jnp.asarray(host_indices)
                if _can_append_singles_columns(
                        self, ring, columns, numeric_keys, device_indices,
                        gather_device):
                    previous = self.buffer[ring]
                    prior_numeric = (() if previous is None else
                                     tuple(previous[key] for key in numeric_keys))
                    prepared = _append_singles_columns(
                        numeric_values, prior_numeric, self.buffer_expire[ring],
                        device_indices, np.int32(self.time))
            if prepared is not None:
                previous = self.buffer[ring]
                appended, expiry = prepared
                numeric_row = dict(zip(numeric_keys, appended))
                row = {}
                for key, value in columns.items():
                    if key in numeric_row:
                        row[key] = numeric_row[key]
                    else:
                        selected_value = value[host_indices]
                        row[key] = (selected_value if previous is None else
                                    np.concatenate((previous[key],
                                                    selected_value)))
                self.buffer[ring] = row
                clocks = np.full(len(selected), self.time, dtype=np.int64)
                self._expire_times[ring] = np.concatenate(
                    (self._expire_times[ring], clocks))
                self.buffer_expire[ring] = expiry
                self.valid_ends[ring] += len(selected)
                continue
            gathered = {}
            if (gather_device is not None and
                    device_indices.devices() == {gather_device}):
                gathered = dict(zip(numeric_keys, _gather_singles_columns(
                    numeric_values, device_indices)))
            row = {
                key: (gathered[key] if key in gathered else
                      value[device_indices] if is_jax_array(value) else
                      value[host_indices])
                for key, value in columns.items()
            }
            if self.buffer[ring] is None:
                self.buffer[ring] = row
            else:
                appended = {}
                for key, new_value in row.items():
                    old_value = self.buffer[ring][key]
                    if is_jax_array(old_value):
                        appended[key] = jnp.concatenate(
                            (old_value, jnp.asarray(new_value)))
                    else:
                        appended[key] = np.concatenate(
                            (old_value, np.asarray(new_value)))
                self.buffer[ring] = appended
            clocks = np.full(len(selected), self.time, dtype=np.int64)
            self._expire_times[ring] = np.concatenate(
                (self._expire_times[ring], clocks))
            self.buffer_expire[ring] = jnp.concatenate(
                (self.buffer_expire[ring], jnp.asarray(clocks, dtype=jnp.int32)))
            self.valid_ends[ring] += len(selected)
        self.time += 1

    def _prune_ring(self, buffer_index):
        row = self.buffer[buffer_index]
        if row is None:
            return
        clocks = self._expire_times[buffer_index]
        start = int(np.searchsorted(clocks, self.time - self.max_time))
        if start == 0:
            return
        self.buffer[buffer_index] = {
            key: value[start:]
            for key, value in row.items()
        }
        self._expire_times[buffer_index] = clocks[start:]
        self.buffer_expire[buffer_index] = self.buffer_expire[buffer_index][start:]
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

    def discard_last(self, indices):
        for ring in np.asarray(indices):
            ring = int(ring)
            if self.buffer[ring] is not None:
                self.buffer[ring] = {
                    key: value[:-1] for key, value in self.buffer[ring].items()
                }
                self.buffer_expire[ring] = self.buffer_expire[ring][:-1]
                self._expire_times[ring] = self._expire_times[ring][:-1]
                self.valid_ends[ring] -= 1


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


