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

Native GPU live searches retain masked numeric arrays through foreground and
background calculation. Terminal controls publish Python ring lengths and
route result dictionaries; output columns remain device arrays. Standalone
compact-array APIs and custom hooks retain their compatibility boundaries.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import (
    JAXArrayData,
    _as_jax_array,
    _ensure_x64,
    _divide,
    _reference_enabled,
    is_jax_array,
)
from .stat import QuadratureSumStatistic
from .ranking_jax import _event_arrays
from . import coinc as _coinc_module
from pycbc.benchmark import host_stage
from pycbc import conversions as conv


_quadrature_rank_stat = QuadratureSumStatistic.rank_stat_coinc
_mchirp_from_masses = conv.mchirp_from_mass1_mass2
_eta_from_masses = conv.eta_from_mass1_mass2
_public_time_coincidence = _coinc_module.time_coincidence
_public_cluster_coincs = _coinc_module.cluster_coincs
_coincidence_backend = _coinc_module._coinc_backend


@jax.jit
def _prepare_live_chisq(chisq, dof):
    """Prepare both ranking columns while retaining the dof addition dtype."""
    return chisq * dof, jax.lax.optimization_barrier(dof + 2) / 2


def _can_prepare_live_chisq(chisq, dof):
    """Keep conversions, unusual shapes and placement on the eager path."""
    if (not all(isinstance(value, jax.Array) and
                not isinstance(value, jax.core.Tracer) and value.ndim == 1
                for value in (chisq, dof)) or
            chisq.shape != dof.shape or chisq.size > 4096 or
            np.dtype(chisq.dtype) not in (np.dtype(np.float32),
                                          np.dtype(np.float64)) or
            np.dtype(dof.dtype) not in (np.dtype(np.float32),
                                       np.dtype(np.float64),
                                       np.dtype(np.uint32))):
        return False
    devices = chisq.devices()
    return len(devices) == 1 and dof.devices() == devices


def _can_skip_live_mchirp(ranker, trigs):
    """Omit unused chirp values only for unchanged ordinary quadrature inputs."""
    if (type(ranker) is not QuadratureSumStatistic or
            getattr(ranker.rank_stat_coinc, "__func__", None) is not
            _quadrature_rank_stat or
            conv.mchirp_from_mass1_mass2 is not _mchirp_from_masses or
            conv.eta_from_mass1_mass2 is not _eta_from_masses or
            type(trigs) is not dict):
        return False
    columns = tuple(trigs.get(key) for key in
                    ("mass1", "mass2", "end_time", "stat"))
    if not all((type(value) is np.ndarray or
                isinstance(value, jax.Array) and
                not isinstance(value, jax.core.Tracer)) and value.ndim == 1
               for value in columns):
        return False
    if isinstance(columns[0], jax.Array) != isinstance(columns[1], jax.Array):
        return False
    if isinstance(columns[0], jax.Array):
        devices = columns[0].devices()
        if len(devices) != 1 or columns[1].devices() != devices:
            return False
    return (all(value.size == columns[0].size for value in columns[1:]) and
            all(np.dtype(value.dtype) in (np.dtype(np.float32),
                                           np.dtype(np.float64))
                for value in columns[:2]))


def pick_best_coinc_jax(coinc_results, logger):
    """Select the best coincidence using device-resident IFAR/stat arrays."""
    candidates = [result for result in coinc_results
                  if "coinc_possible" in result and
                  "foreground/ifar" in result]
    trials = sum("coinc_possible" in result for result in coinc_results)
    if not candidates:
        return coinc_results[0]
    ifar = jnp.stack(_event_arrays(*(result['foreground/ifar']
                                   for result in candidates)))
    stat = jnp.stack(_event_arrays(*(result['foreground/stat']
                                   for result in candidates)))
    stat = stat.reshape((len(candidates), -1))[:, 0]

    def select(current, row):
        index, best_ifar, best_stat = current
        candidate, value, score = row
        better = (value > best_ifar) | ((value == best_ifar) & (score > best_stat))
        return (jnp.where(better, candidate, index),
                jnp.where(better, value, best_ifar),
                jnp.where(better, score, best_stat)), None

    (best, best_ifar, _), _ = jax.lax.scan(
        select, (jnp.int64(-1), jnp.zeros((), ifar.dtype),
                 jnp.zeros((), stat.dtype)),
        (jnp.arange(len(candidates)), ifar, stat))
    corrected_ifar = _divide(best_ifar, float(trials))
    # Selection routes a Python result dictionary and logging needs a host
    # scalar. Finish numeric work first and collect both terminal values once.
    with host_stage('coinc_best_publication_readback', candidates=len(candidates)):
        best_host, ifar_host = jax.device_get((best, corrected_ifar))
    if best_host < 0:
        return coinc_results[0]
    result = candidates[int(best_host)]
    result["foreground/ifar"] = corrected_ifar
    logger.info("Found %s coinc with ifar %s", result["foreground/type"],
                ifar_host)
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
            raw_chisq = trigs["chisq"]
            raw_dof = trigs["chisq_dof"]
            chisq, dof = _event_arrays(raw_chisq, raw_dof)
            if (type(trigs) is dict and
                    _can_prepare_live_chisq(raw_chisq, raw_dof)):
                trigsc["chisq"], trigsc["chisq_dof"] = _prepare_live_chisq(
                    chisq, dof)
            else:
                trigsc["chisq"] = chisq * dof
                trigsc["chisq_dof"] = (dof + 2) / 2
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
        return _event_arrays(value)[0]
    return value


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


def _live_fixed_time_jax(times, index):
    """Read one incoming time while retaining wrapped float32 promotion."""
    if (isinstance(times, (Array, JAXArrayData)) and
            np.dtype(times.dtype) == np.dtype(np.float32)):
        # Wrapped scalar reads historically promote through a Python float.
        # Device float32-to-float64 casts may flush subnormals, so retain that
        # narrow compatibility boundary instead of changing the input time.
        return jnp.array(times[index], ndmin=1, dtype=jnp.float64)
    fixed_time = times[index:index + 1]
    device_time = _as_jax_array(fixed_time)
    return jnp.asarray(device_time if device_time is not None else fixed_time,
                       dtype=jnp.float64)


_native_live_fixed_time = _live_fixed_time_jax
_native_time_coincidence = time_coincidence
_native_ring_data = JAXMultiRingBuffer.data
_native_ring_prune = JAXMultiRingBuffer._prune_ring
_native_ring_expire = JAXMultiRingBuffer.expire_vector


@jax.jit
def _prepare_live_query_times(fixed_times, control):
    """Gather incoming times separately from the history-shape specialization."""
    return fixed_times[control[0]].astype(jnp.float64)


@partial(jax.jit, static_argnames=("sliding", "pruning"))
def _time_coincidence_query_core(times, fixed_times, control, window,
                                 slide_step, *, sliding, pruning):
    """Apply the singleton matcher to a bounded group of equal-sized rings."""
    def match(values, fixed, start):
        indices, _, slides, count = _time_coincidence_single_core.__wrapped__(
            values.astype(jnp.float64), fixed[None], window, slide_step,
            sliding)
        if pruning:
            # The ring is still pruned at its original scalar-call boundary.
            # Stable filtering retains exactly that suffix's matching order.
            valid = ((jnp.arange(indices.size) < count) & (indices >= start))
            selected = jnp.nonzero(valid, size=indices.size, fill_value=0)[0]
            indices = indices[selected] - start
            slides = slides[selected]
            count = jnp.sum(valid, dtype=jnp.int64)
        return indices, slides, count

    return jax.vmap(match)(jnp.stack(times), fixed_times, control[1])


@partial(jax.jit, static_argnames=("count",))
def _compact_live_query(indices, slides, fixed_times, index, *, count):
    """Compact a positive query without specializing on its group position."""
    return (indices[index, :count], slides[index, :count],
            jax.lax.dynamic_slice_in_dim(fixed_times, index, 1))


def _can_prepare_live_queries(ring, trigs, templates, first, last, ranker,
                              window, slide_step, minimum_queries=2):
    """Snapshot only ordinary immutable columns with unchanged live hooks."""
    if (_reference_enabled('time_coincidence') or
            not jax.config.jax_enable_x64 or
            not _can_skip_live_mchirp(ranker, trigs) or
            _coinc_module.time_coincidence is not _public_time_coincidence or
            _coinc_module._coinc_backend is not _coincidence_backend or
            time_coincidence is not _native_time_coincidence or
            _live_fixed_time_jax is not _native_live_fixed_time or
            type(ring) is not JAXMultiRingBuffer or
            getattr(ring.data, "__func__", None) is not _native_ring_data or
            getattr(ring._prune_ring, "__func__", None) is not
            _native_ring_prune or
            getattr(ring.expire_vector, "__func__", None) is not
            _native_ring_expire):
        return False
    scalar_types = (int, float, np.int32, np.int64, np.float32, np.float64)
    limits = np.iinfo(np.int64)
    if (any(type(value) not in scalar_types or not np.isfinite(value) or
            value < 0 or isinstance(value, (int, np.integer)) and
            not limits.min <= value <= limits.max
            for value in (window, slide_step)) or
            not 0 <= first < last <= len(trigs['end_time']) or
            not minimum_queries <= last - first <= 8 or
            type(templates) is not np.ndarray or templates.ndim != 1 or
            templates.dtype.kind not in 'iu' or
            templates.size != len(trigs['end_time'])):
        return False
    fixed_times, fixed_stats = trigs['end_time'], trigs['stat']
    if (not all(isinstance(value, jax.Array) and
                not isinstance(value, jax.core.Tracer) and value.ndim == 1 and
                np.dtype(value.dtype) in (np.dtype(np.float32),
                                          np.dtype(np.float64))
                for value in (fixed_times, fixed_stats)) or
            fixed_times.size > 4096 or fixed_times.shape != fixed_stats.shape):
        return False
    devices = fixed_times.devices()
    if (len(devices) != 1 or fixed_stats.devices() != devices or
            any(type(getattr(ring, key)) is not list for key in
                ('buffer', 'buffer_expire', '_expire_times', 'valid_ends')) or
            type(ring.num_rings) not in (int, np.int32, np.int64) or
            type(ring.time) not in (int, np.int32, np.int64) or
            type(ring.max_time) not in (int, np.int32, np.int64) or
            not limits.min <= int(ring.time) - int(ring.max_time) <= limits.max or
            any(len(getattr(ring, key)) != ring.num_rings for key in
                ('buffer', 'buffer_expire', '_expire_times', 'valid_ends'))):
        return False
    for template in templates[first:last]:
        template = int(template)
        if not 0 <= template < ring.num_rings:
            return False
        row = ring.buffer[template]
        if row is None:
            continue
        if type(row) is not dict or not {'end_time', 'stat'} <= row.keys():
            return False
        times, stats = row['end_time'], row['stat']
        if not all(isinstance(value, jax.Array) and
                   not isinstance(value, jax.core.Tracer) and value.ndim == 1
                   and np.dtype(value.dtype) in (np.dtype(np.float32),
                                                  np.dtype(np.float64))
                   and value.devices() == devices for value in (times, stats)):
            return False
        clocks, expiry = (ring._expire_times[template],
                          ring.buffer_expire[template])
        if (times.size > 64 or stats.shape != times.shape or
                type(clocks) is not np.ndarray or clocks.ndim != 1 or
                clocks.dtype != np.dtype(np.int64) or
                clocks.shape != times.shape or
                np.any(clocks[1:] < clocks[:-1]) or
                not isinstance(expiry, jax.Array) or
                isinstance(expiry, jax.core.Tracer) or expiry.ndim != 1 or
                expiry.shape != times.shape or expiry.devices() != devices or
                np.dtype(expiry.dtype) != np.dtype(np.int32) or
                any(not (type(value) is np.ndarray or
                         isinstance(value, jax.Array) and
                         not isinstance(value, jax.core.Tracer)) or
                    value.ndim == 0 or value.shape[0] != times.size
                    for value in row.values())):
            return False
    return True


def _prepare_live_queries(ring, trigs, templates, first, last, ranker,
                          window, slide_step):
    """Compatibility counts; native GPU search retains resident query masks."""
    if last - first < 2:
        return {}
    with host_stage('coinc_query_host_prepare', queries=last - first):
        if not _can_prepare_live_queries(ring, trigs, templates, first, last,
                                         ranker, window, slide_step):
            return {}
        groups = {}
        for index in range(first, last):
            template = int(templates[index])
            row = ring.buffer[template]
            if row is None or not row['end_time'].size:
                continue
            times = row['end_time']
            start = int(np.searchsorted(ring._expire_times[template],
                                        ring.time - ring.max_time))
            groups.setdefault((times.size, times.dtype), []).append(
                (index, times, start))
    prepared = {}
    for rows in groups.values():
        if len(rows) < 2:
            continue
        with host_stage('coinc_query_host_prepare', queries=len(rows)):
            control = np.asarray((tuple(row[0] for row in rows),
                                  tuple(row[2] for row in rows)), dtype=np.int64)
        with host_stage('coinc_query_dispatch', queries=len(rows)):
            device_control = jax.device_put(control, trigs['end_time'].device)
            fixed_times = _prepare_live_query_times(
                trigs['end_time'], device_control)
            indices, slides, counts = _time_coincidence_query_core(
                tuple(row[1] for row in rows), fixed_times, device_control,
                window, slide_step, sliding=bool(slide_step),
                pruning=bool(np.any(control[1])))
        # Count readback is the existing synchronization boundary; stage
        # recording never adds a fence to dispatch or host preparation.
        with host_stage('coinc_query_count_readback', queries=len(rows)):
            host_counts = np.asarray(counts)
        with host_stage('coinc_query_host_publish', queries=len(rows)):
            for position, (row, count) in enumerate(zip(rows, host_counts)):
                prepared[row[0]] = (indices, slides, fixed_times, position,
                                    int(count))
    return prepared


@jax.jit
def _prepare_live_match(memory, fixed_stats, mchirps, index, fixed_time,
                        times, stats, expiration, indices, template, clock):
    """Gather one matched payload without fusing ranking arithmetic."""
    count = indices.size
    memory = memory.at[:count].set(fixed_stats[index])
    payload = (times[indices], jnp.full(count, fixed_time[0], jnp.float64),
               expiration[indices], jnp.full(count, clock, jnp.int32),
               jnp.full(count, template, jnp.int32),
               indices.astype(jnp.int32), jnp.full(count, -1, jnp.int32))
    mchirp = None if mchirps is None else mchirps[index]
    return memory, memory[:count], stats[indices], mchirp, payload


def _can_prepare_live_match(values, indices, template, clock):
    """Keep unusual storage/device cases on their prior scalar path."""
    arrays = (*values[:2], *values[3:], indices) if values[2] is None else (
        *values, indices)
    if (not all(isinstance(value, jax.Array) and
                not isinstance(value, jax.core.Tracer) and value.ndim == 1
                for value in arrays) or
            indices.size > 192 or values[0].size < indices.size or
            (values[2] is not None and values[1].size != values[2].size) or
            values[4].size > 64 or
            values[4].size != values[5].size or
            values[4].size != values[6].size or
            not np.iinfo(np.int32).min <= template <= np.iinfo(np.int32).max or
            not np.iinfo(np.int32).min <= clock <= np.iinfo(np.int32).max):
        return False
    devices = values[0].devices()
    return (len(devices) == 1 and
            all(value.devices() == devices for value in arrays[1:]))


@partial(jax.jit, static_argnames=('counts',))
def _prepare_live_match_group(memory, fixed_stats, rows, control, *, counts):
    """Compact/gather a bounded chunk in original workspace-write order."""
    prepared = []
    for number, (row, count) in enumerate(zip(rows, counts)):
        indices, slides, fixed_times, times, stats, expiration = row
        position, index, start, template, clock = control[:, number]
        selected = indices[position, :count]
        slide = slides[position, :count]
        fixed_time = jax.lax.dynamic_slice_in_dim(fixed_times, position, 1)
        # Query indices refer to the pruned ring; these immutable snapshots
        # still include its expired prefix. Public IDs remain relative.
        original = selected + start
        while memory.size < count:
            memory = jnp.pad(memory, (0, memory.size))
        memory = memory.at[:count].set(fixed_stats[index])
        payload = (times[original],
                   jnp.full(count, fixed_time[0], jnp.float64),
                   expiration[original], jnp.full(count, clock, jnp.int32),
                   jnp.full(count, template, jnp.int32),
                   selected.astype(jnp.int32), jnp.full(count, -1, jnp.int32))
        prepared.append((memory, memory[:count], stats[original], selected,
                         slide, fixed_time, payload))
    return tuple(prepared)


def _prepare_live_payloads(ring, fixed_ring, trigs, templates, first, last,
                           queries, memory):
    """Gather admitted query snapshots without advancing observable state.

    Every nonempty query must be covered: independently batching interleaved
    history-shape groups would lose earlier writes in unused workspace tails.
    The caller still prunes each ring and publishes each memory snapshot at
    its original serial boundary, before calling the unchanged native ranker.
    """
    if (not queries or type(fixed_ring) is not JAXMultiRingBuffer or
            type(fixed_ring.time) not in (int, np.int32, np.int64) or
            not np.iinfo(np.int32).min <= fixed_ring.time - 1 <=
            np.iinfo(np.int32).max or
            not isinstance(memory, jax.Array) or
            isinstance(memory, jax.core.Tracer) or memory.ndim != 1 or
            not memory.size or memory.devices() != trigs['stat'].devices() or
            np.dtype(memory.dtype) not in (np.dtype(np.float32),
                                           np.dtype(np.float64))):
        return {}
    rows, controls, counts, incoming = [], [], [], []
    for index in range(first, last):
        template = int(templates[index])
        row = ring.buffer[template]
        if row is None or not row['end_time'].size:
            continue
        query = queries.get(index)
        if query is None or not 0 <= template <= np.iinfo(np.int32).max:
            return {}
        indices, slides, fixed_times, position, count = query
        if count == 0:
            continue
        if count > 192:
            return {}
        start = int(np.searchsorted(ring._expire_times[template],
                                    ring.time - ring.max_time))
        rows.append((indices, slides, fixed_times, row['end_time'], row['stat'],
                     ring.buffer_expire[template]))
        controls.append((position, index, start, template, fixed_ring.time - 1))
        counts.append(count)
        incoming.append(index)
    if len(rows) < 2:
        return {}
    control = jax.device_put(np.asarray(controls, dtype=np.int64).T,
                             trigs['stat'].device)
    prepared = _prepare_live_match_group(memory, trigs['stat'], tuple(rows),
                                         control, counts=tuple(counts))
    return dict(zip(incoming, prepared))


@jax.jit
def _pack_live_rank_inputs(rows):
    """Join equal-dtype operands without fusing any ranking arithmetic."""
    return tuple(jnp.concatenate(tuple(row[column] for row in rows))
                 for column in range(2))


@partial(jax.jit, static_argnames=('counts',))
def _split_live_ranks(ranks, *, counts):
    ranks = ranks.astype(jnp.float64)
    results, start = [], 0
    for count in counts:
        results.append(ranks[start:start + count])
        start += count
    return tuple(results)


def _rank_live_matches(ranker, payloads):
    """Group pure native ranks while retaining their eager arithmetic steps.

    Operand dtypes and weak types are grouped independently: concatenating
    different promotion rules before squaring would change the native result.
    Custom rankers retain their serial calls and observable state boundaries.
    """
    if (not jax.config.x64_enabled or len(payloads) < 2 or
            type(ranker) is not QuadratureSumStatistic or
            getattr(ranker.rank_stat_coinc, '__func__', None) is not
            _quadrature_rank_stat):
        return {}
    groups = {}
    for index, payload in payloads.items():
        fixed, shifted = payload[1:3]
        if (not all(isinstance(value, jax.Array) and
                    not isinstance(value, jax.core.Tracer) and value.ndim == 1
                    and np.dtype(value.dtype) in (np.dtype(np.float32),
                                                  np.dtype(np.float64))
                    for value in (fixed, shifted)) or
                fixed.shape != shifted.shape or not 0 < fixed.size <= 192 or
                len(fixed.devices()) != 1 or
                fixed.devices() != shifted.devices()):
            return {}
        key = (fixed.dtype, shifted.dtype, fixed.weak_type, shifted.weak_type,
               fixed.device)
        groups.setdefault(key, []).append((index, fixed, shifted))
    prepared = {}
    for group in groups.values():
        if len(group) < 2 or len(group) > 8:
            continue
        fixed, shifted = _pack_live_rank_inputs(
            tuple((row[1], row[2]) for row in group))
        # The exact native method ignores detector names, slides and kwargs.
        # Calling it unchanged preserves power/add/where dispatch boundaries.
        ranks = ranker.rank_stat_coinc([['fixed', fixed], ['shifted', shifted]],
                                       None, None, None)
        values = _split_live_ranks(
            ranks, counts=tuple(row[1].size for row in group))
        prepared.update(zip((row[0] for row in group), values))
    return prepared


@jax.jit
def _partition_live_cluster(cidx, offsets):
    """Pack clustered original indices stably; report both dynamic lengths."""
    selected_offsets = offsets[cidx]
    background = selected_offsets != 0
    zerolag = selected_offsets == 0
    background_rows = jnp.nonzero(background, size=cidx.size, fill_value=0)[0]
    zerolag_rows = jnp.nonzero(zerolag, size=cidx.size, fill_value=0)[0]
    return (cidx[background_rows], cidx[zerolag_rows],
            jnp.stack((background.sum(), zerolag.sum())))


@partial(jax.jit, static_argnames=('background_count', 'zerolag_count'))
def _select_live_cluster(cstat, expiry0, expiry1, templates, ids0, ids1,
                         background, zerolag, *, background_count,
                         zerolag_count):
    """Gather all published cluster columns in one pure device dispatch."""
    background = background[:background_count]
    zerolag = zerolag[:zerolag_count]
    if zerolag_count:
        first = zerolag[0]
        first_values = (cstat[first], templates[first], ids0[first], ids1[first])
    else:
        first_values = tuple(jnp.zeros((), dtype=value.dtype)
                             for value in (cstat, templates, ids0, ids1))
    return (cstat[background], expiry0[background], expiry1[background],
            cstat[zerolag], *first_values,
            jnp.asarray(background_count, dtype=jnp.int64),
            jnp.asarray(zerolag_count, dtype=jnp.int64))


def _prepare_live_cluster(cidx, offsets, columns):
    """Compatibility publication; resident GPU search bypasses these counts."""
    values = (offsets, *columns, cidx)
    if (not jax.config.x64_enabled or
            not all(isinstance(value, jax.Array) and
                    not isinstance(value, jax.core.Tracer) and value.ndim == 1
                    and jnp.issubdtype(value.dtype, jnp.number)
                    for value in values) or
            not jnp.issubdtype(cidx.dtype, jnp.integer) or
            any(value.shape != offsets.shape for value in columns)):
        return None
    devices = offsets.devices()
    if (len(devices) != 1 or
            any(value.devices() != devices for value in values[1:])):
        return None
    background, zerolag, counts = _partition_live_cluster(cidx, offsets)
    # One control read replaces separate boolean-index length discoveries.
    background_count, zerolag_count = map(int, np.asarray(counts))
    return _select_live_cluster(*columns, background, zerolag,
                                background_count=background_count,
                                zerolag_count=zerolag_count)


def _prepare_live_foreground(stored, index):
    """Gather selected numeric columns after pruning and terminal routing."""
    host_index = type(index) is int
    if (type(stored) is not dict or
            not all(type(value) is np.ndarray or
                    isinstance(value, jax.Array) and
                    not isinstance(value, jax.core.Tracer)
                    for value in stored.values())):
        return {}
    if host_index:
        devices = next((value.devices() for value in stored.values()
                        if isinstance(value, jax.Array) and value.ndim == 1 and
                        jnp.issubdtype(value.dtype, jnp.number) and
                        len(value.devices()) == 1), None)
        index_dtype = np.int64 if jax.config.x64_enabled else np.int32
        if (devices is None or
                not np.iinfo(index_dtype).min <= index <= np.iinfo(index_dtype).max):
            return {}
    else:
        if (not isinstance(index, jax.Array) or
                isinstance(index, jax.core.Tracer) or index.ndim != 0 or
                not jnp.issubdtype(index.dtype, jnp.integer) or
                len(index.devices()) != 1):
            return {}
        devices = index.devices()
    keys = tuple(key for key, value in stored.items()
                 if isinstance(value, jax.Array) and value.ndim == 1 and
                 jnp.issubdtype(value.dtype, jnp.number) and
                 value.devices() == devices)
    if len(keys) < 2:
        return {}
    if host_index:
        # Resident routing has already crossed the terminal control boundary.
        # Select only this pruned ring, retaining negative last-row indices.
        index = jax.device_put(index_dtype(index), next(iter(devices)))
    values = _gather_singles_columns(tuple(stored[key] for key in keys), index)
    return dict(zip(keys, values))


def _concatenate_live_match_columns(payloads):
    """Concatenate ordered numeric columns together in bounded groups."""
    columns = tuple(jnp.concatenate(tuple(row[i] for row in payloads))
                    for i in range(9))
    return (columns[0], columns[1], columns[2].astype(jnp.float64),
            columns[3].astype(jnp.float64), columns[4], columns[5],
            columns[6].astype(jnp.int32), columns[7].astype(jnp.int32),
            columns[8].astype(jnp.int32))


_concatenate_live_matches = jax.jit(_concatenate_live_match_columns)


def _join_live_matches(payloads):
    """Bound compiled tuple arity even when many incoming triggers match."""
    values = tuple(value for payload in payloads for value in payload)
    if not all(isinstance(value, jax.Array) and
               not isinstance(value, jax.core.Tracer) for value in values):
        return _concatenate_live_match_columns(payloads)
    devices = values[0].devices()
    if (len(devices) != 1 or
            any(value.devices() != devices for value in values[1:])):
        return _concatenate_live_match_columns(payloads)
    while len(payloads) > 32:
        payloads = tuple(_concatenate_live_matches(payloads[i:i + 32])
                         for i in range(0, len(payloads), 32))
    return _concatenate_live_matches(payloads)


_native_join_live_matches = _join_live_matches
_native_coinc_add = JAXCoincExpireBuffer.add
_native_coinc_greater = JAXCoincExpireBuffer.num_greater
_native_estimator_ifar = _coinc_module.LiveCoincTimeslideBackgroundEstimator.ifar
_native_background_time = _coinc_module.LiveCoincTimeslideBackgroundEstimator.background_time.fget
_native_sec_to_year = conv.sec_to_year


def _resident_live_quadrature(fixed, shifted):
    """Preserve the native ranker's separate power and addition rounding."""
    barrier = jax.lax.optimization_barrier
    fixed_square = barrier(fixed * fixed)
    shifted_square = barrier(shifted * shifted)
    summed = barrier(barrier(0 + fixed_square) + shifted_square)
    return jnp.sqrt(summed)


@partial(jax.jit, static_argnames=('sliding', 'memory_dtype', 'fixed_first'))
def _resident_live_match_group(times, stats, expiries, fixed_times, fixed_stats,
                                control, window, slide_step, *, sliding,
                                memory_dtype, fixed_first):
    """Match, rank and prepare ordered rows without eager device dispatch."""
    queries = _prepare_live_query_times.__wrapped__(fixed_times, control[:2])
    indices, slides, counts = _time_coincidence_query_core.__wrapped__(
        times, queries, control[:2], window, slide_step,
        sliding=sliding, pruning=True)
    prepared = []
    incoming_stats = fixed_stats
    fixed_stats = fixed_stats.astype(memory_dtype)
    for position, (time, stat, expiry) in enumerate(zip(times, stats, expiries)):
        start, template, clock = control[1:, position]
        original = indices[position] + start
        count = counts[position]
        valid = jnp.arange(original.size) < count
        fixed = jnp.full(original.size, fixed_stats[control[0, position]],
                         dtype=fixed_stats.dtype)
        payload = (time[original],
                   jnp.full(original.size, queries[position], jnp.float64),
                   expiry[original], jnp.full(original.size, clock, jnp.int32),
                   jnp.full(original.size, template, jnp.int32),
                   indices[position].astype(jnp.int32),
                   jnp.full(original.size, -1, jnp.int32))
        prepared.append((fixed, stat[original], slides[position], payload,
                         valid, count))
    fixed = jnp.concatenate(tuple(value[0] for value in prepared))
    shifted = jnp.concatenate(tuple(value[1] for value in prepared))
    ranks = _resident_live_quadrature(fixed, shifted).astype(jnp.float64)
    rows, begin = [], 0
    for position, match in enumerate(prepared):
        _, _, slide, payload, valid, count = match
        capacity = valid.size
        rank = ranks[begin:begin + capacity]
        begin += capacity
        shifted_time, fixed_time, shifted_expiry, fixed_expiry, template, ids, fixed_ids = payload
        columns = ((rank, slide, fixed_time, shifted_time,
                    fixed_expiry, shifted_expiry, template, fixed_ids, ids)
                   if fixed_first else
                   (rank, slide, shifted_time, fixed_time,
                    shifted_expiry, fixed_expiry, template, ids, fixed_ids))
        rows.append((columns, valid, count, incoming_stats[control[0, position]]))
    return tuple(rows)


@partial(jax.jit, static_argnames=('groups', 'memory_dtype'))
def _pack_resident_live_matches(rows, *, groups=False, memory_dtype):
    """Join columns and resident query controls with bounded input arity."""
    columns = _concatenate_live_match_columns(tuple(row[0] for row in rows))
    valid = jnp.concatenate(tuple(row[1] for row in rows))
    if groups:
        counts = jnp.concatenate(tuple(row[2] for row in rows))
        values = jnp.concatenate(tuple(row[3] for row in rows))
    else:
        counts = jnp.stack(tuple(row[2] for row in rows))
        values = jnp.stack(tuple(row[3].astype(memory_dtype) for row in rows))
    return columns, valid, counts, values


@partial(jax.jit, static_argnames=('work_size', 'groups'))
def _finish_resident_live_matches(rows, memory, *, work_size, groups):
    """Join the final group and preserve ordered native workspace overwrites."""
    columns, valid, counts, values = _pack_resident_live_matches.__wrapped__(
        rows, groups=groups, memory_dtype=memory.dtype)
    memory = jnp.pad(memory, (0, work_size - memory.size))
    memory, max_count = _resident_live_memory.__wrapped__(
        memory, counts, values.astype(memory.dtype))
    return columns, valid, memory, max_count


@jax.jit
def _resident_live_memory(memory, counts, values):
    """Preserve the native workspace's ordered prefix overwrites on-device."""
    positions = jnp.arange(memory.size)

    def update(current, row):
        count, value = row
        return jnp.where(positions < count, value, current), None

    memory, _ = jax.lax.scan(update, memory, (counts, values))
    return memory, counts.max()


def _resident_live_device_supported(value):
    """Keep padded coincidence dispatch specific to the GPU backend."""
    return value.device.platform == 'gpu'


def _resident_live_reject(estimator, reason):
    estimator._jax_resident_coinc_reason = reason
    return None


def _prepare_resident_live_matches(estimator, results, valid_ifos):
    """Record resident admission without collecting any device scalar."""
    prepared = _prepare_resident_live_matches_impl(estimator, results, valid_ifos)
    if type(estimator) is not JAXLiveCoincTimeslideBackgroundEstimator:
        return prepared
    counters = getattr(estimator, '_jax_resident_coinc_counters', None)
    if counters is None:
        counters = estimator._jax_resident_coinc_counters = {
            'admitted': 0, 'compatibility': {}}
    if prepared is None:
        reason = getattr(estimator, '_jax_resident_coinc_reason', 'unsupported')
        counters['compatibility'][reason] = counters['compatibility'].get(reason, 0) + 1
        with host_stage('coinc_resident_compatibility', reason=reason):
            pass
    else:
        counters['admitted'] += 1
        estimator._jax_resident_coinc_reason = None
        with host_stage('coinc_resident_admitted', capacity=prepared[1].size):
            pass
    return prepared


def _prepare_resident_live_matches_impl(estimator, results, valid_ifos):
    """Queue native GPU matching/ranking for both detectors before collection.

    Host template IDs route immutable ring snapshots. Scientific columns,
    query lengths and validity masks stay resident through clustering. Custom
    hooks and unbounded histories retain their observable serial path.
    """
    if type(estimator) is not JAXLiveCoincTimeslideBackgroundEstimator:
        return None
    if any(_reference_enabled(operation) for operation in (
            'time_coincidence', 'cluster_over_time', 'cluster_coincs',
            'quadrature_sum')):
        return _resident_live_reject(estimator, 'original_reference_operation')
    if (_coinc_module.cluster_coincs is not _public_cluster_coincs or
            _join_live_matches is not _native_join_live_matches):
        return _resident_live_reject(estimator, 'custom_cluster_or_join')
    if (type(estimator.coincs) is not JAXCoincExpireBuffer or
            getattr(estimator.coincs.add, '__func__', None) is not _native_coinc_add or
            getattr(estimator.coincs.num_greater, '__func__', None) is not _native_coinc_greater or
            getattr(estimator.ifar, '__func__', None) is not _native_estimator_ifar):
        return _resident_live_reject(estimator, 'custom_background_or_ifar')
    if (getattr(type(estimator).background_time, 'fget', None) is not
            _native_background_time or conv.sec_to_year is not _native_sec_to_year):
        return _resident_live_reject(estimator, 'custom_background_time_or_conversion')
    ranker = estimator.stat_calculator
    directions, first_stat = [], None
    for fixed_ifo, shift_ifo in ((estimator.ifos[0], estimator.ifos[1]),
                                (estimator.ifos[1], estimator.ifos[0])):
        if fixed_ifo not in valid_ifos:
            continue
        trigs = results[fixed_ifo]
        if not len(trigs['end_time']):
            continue
        times = trigs['end_time']
        if (not isinstance(times, jax.Array) or isinstance(times, jax.core.Tracer)
                or len(times.devices()) != 1 or not _resident_live_device_supported(times)):
            return _resident_live_reject(estimator, 'incoming_device_or_layout')
        templates = results[fixed_ifo]['template_id']
        # MPI has already supplied owned host routing IDs. Device IDs still
        # need an explicit boundary and retain the compatibility path.
        if type(templates) is not np.ndarray:
            return _resident_live_reject(estimator, 'device_template_routing_ids')
        ring, fixed_ring = (estimator.singles[shift_ifo],
                            estimator.singles[fixed_ifo])
        if (type(fixed_ring) is not JAXMultiRingBuffer or
                type(fixed_ring.time) not in (int, np.int32, np.int64) or
                not np.iinfo(np.int32).min <= fixed_ring.time - 1 <=
                np.iinfo(np.int32).max):
            return _resident_live_reject(estimator, 'fixed_ring_or_clock')
        for first in range(0, times.size, 8):
            if not _can_prepare_live_queries(
                    ring, trigs, templates, first, min(first + 8, times.size),
                    ranker, estimator.time_window, estimator.timeslide_interval, 1):
                return _resident_live_reject(estimator, 'query_geometry_dtype_or_hooks')
        if first_stat is None:
            first_stat = trigs['stat']
        directions.append((fixed_ifo, shift_ifo, trigs, templates, ring,
                           fixed_ring.time - 1))
    if first_stat is None:
        return _resident_live_reject(estimator, 'no_incoming_triggers')
    if estimator.timeslide_interval == 0:
        return _resident_live_reject(estimator, 'zero_timeslide_background_time')
    memory = estimator.trig_stat_memory
    if memory is None:
        memory = jnp.zeros(1, dtype=first_stat.dtype)
    if (not isinstance(memory, jax.Array) or isinstance(memory, jax.core.Tracer)
            or memory.ndim != 1 or not memory.size or
            memory.devices() != first_stat.devices() or
            np.dtype(memory.dtype) not in (np.dtype(np.float32), np.dtype(np.float64))):
        return _resident_live_reject(estimator, 'workspace_layout_or_placement')
    prepared, prune, maximum_capacity = [], [], memory.size
    for fixed_ifo, shift_ifo, trigs, templates, ring, clock in directions:
        groups = {}
        rows = {}
        for index, template in enumerate(templates):
            template = int(template)
            row = ring.buffer[template]
            prune.append((ring, template))
            if row is None or not row['end_time'].size:
                continue
            times, stats = row['end_time'], row['stat']
            start = int(np.searchsorted(ring._expire_times[template],
                                        ring.time - ring.max_time))
            key = (times.shape, times.dtype, stats.dtype, stats.weak_type)
            groups.setdefault(key, []).append((index, template, start, row,
                                               ring.buffer_expire[template]))
        for group in groups.values():
            for offset in range(0, len(group), 8):
                chunk = group[offset:offset + 8]
                control = jax.device_put(np.asarray(
                    [(index, start, template, clock)
                     for index, template, start, _, _ in chunk], np.int64).T,
                    first_stat.device)
                with host_stage('coinc_resident_query_dispatch', queries=len(chunk)):
                    matches = _resident_live_match_group(
                        tuple(row['end_time'] for _, _, _, row, _ in chunk),
                        tuple(row['stat'] for _, _, _, row, _ in chunk),
                        tuple(expiry for _, _, _, _, expiry in chunk),
                        trigs['end_time'], trigs['stat'],
                        control, estimator.time_window, estimator.timeslide_interval,
                        sliding=bool(estimator.timeslide_interval),
                        memory_dtype=memory.dtype,
                        fixed_first=fixed_ifo == estimator.ifos[0])
                for item, match in zip(chunk, matches):
                    capacity = match[1].size
                    rows[item[0]] = match
                    maximum_capacity = max(maximum_capacity, capacity)
        for index in sorted(rows):
            prepared.append(rows[index])
    if not prepared:
        return _resident_live_reject(estimator, 'no_candidate_histories')
    capacity = sum(row[1].size for row in prepared)
    buffer = estimator.coincs
    if (not isinstance(buffer.index, (int, np.integer)) or
            not 0 <= buffer.index <= buffer.buffer.size or
            tuple(buffer.ifos) != tuple(estimator.ifos) or
            tuple(buffer.timer) != tuple(buffer.ifos) or
            not isinstance(buffer.expiration, (int, np.integer)) or
            any(not isinstance(buffer.time[ifo], (int, np.integer)) or
                not np.iinfo(np.int32).min <=
                buffer.time[ifo] + (ifo in valid_ifos) - buffer.expiration <=
                np.iinfo(np.int32).max for ifo in buffer.ifos) or
            any(not isinstance(value, jax.Array) or
                isinstance(value, jax.core.Tracer) or value.ndim != 1 or
                value.shape != buffer.buffer.shape or
                value.devices() != first_stat.devices()
                for value in (buffer.buffer, *buffer.timer.values())) or
            any(value.dtype != jnp.int32 for value in buffer.timer.values())):
        return _resident_live_reject(estimator, 'background_storage_or_clock')
    storage = (buffer.buffer,) + tuple(buffer.timer.values())
    needed = buffer.index + capacity
    if needed > buffer.buffer.size:
        size = max(needed, buffer.buffer.size * 2)
        storage = tuple(jnp.pad(value, (0, size - value.size)) for value in storage)
    # The native workspace doubles only when an actual nonempty query needs
    # it. Preserve its exact tail now, and publish its final size with counts.
    work_size = memory.size
    while work_size < maximum_capacity:
        work_size *= 2
    groups = False
    while len(prepared) > 32:
        prepared = tuple(_pack_resident_live_matches(
            prepared[first:first + 32], groups=groups, memory_dtype=memory.dtype)
            for first in range(0, len(prepared), 32))
        groups = True
    values, valid, memory, max_count = _finish_resident_live_matches(
        tuple(prepared), memory, work_size=work_size, groups=groups)
    return values, valid, memory, max_count, prune, storage


@jax.jit
def _resident_live_cluster(columns, valid, slide, window):
    """Cluster padded matches with masks; return only final publication counts."""
    positions = jnp.arange(valid.size)
    count = valid.sum(dtype=jnp.int64)
    selected = jnp.nonzero(valid, size=valid.size, fill_value=0)[0]
    columns = tuple(value[selected] for value in columns)
    cstat, offsets, time0, time1 = columns[:4]
    barrier = jax.lax.optimization_barrier
    anchor = time0[:1]
    time = barrier(barrier(time0 - anchor) + barrier(time1 - anchor))
    time = barrier(time + barrier(offsets.astype(time.dtype) * slide))
    time = barrier(time * .5)
    admitted = positions < count
    minimum = jnp.where(admitted, time, jnp.inf).min()
    maximum = jnp.where(admitted, time, -jnp.inf).max()
    span = barrier(barrier(maximum - minimum) + window * 10)
    time = barrier(time + barrier(span * offsets.astype(time.dtype)))
    # Exclude unused capacity from searches and greedy successor traversal.
    sorting = jnp.lexsort((time, ~admitted))
    sorted_time, sorted_stat = time[sorting], cstat[sorting]
    search_time = jnp.where(positions < count, sorted_time, jnp.nan)
    left = jnp.minimum(jnp.searchsorted(search_time, sorted_time - window), count)
    right = jnp.minimum(jnp.searchsorted(search_time, sorted_time + window), count)
    maxima = _cluster_window_maxima(sorted_stat, left, right, 'cython')
    successor = jnp.where(maxima == positions, right,
                          jnp.where(maxima > positions, maxima, positions + 1))
    successor = jnp.where(positions < count, successor, valid.size)
    jump = jnp.concatenate((successor, jnp.array([valid.size], jnp.int64)))
    nodes = jnp.zeros_like(positions)
    bit = 1
    while bit < valid.size:
        nodes = jnp.where((positions & bit) != 0, jump[nodes], nodes)
        jump = jump[jump]
        bit <<= 1
    visited = jnp.zeros(valid.size + 1, bool).at[nodes].set(True)
    keep = visited[:-1] & (maxima == positions) & (positions < count)
    background = keep & (offsets[sorting] != 0)
    zerolag = keep & (offsets[sorting] == 0)
    bkg_rows = sorting[jnp.nonzero(background, size=valid.size, fill_value=0)[0]]
    zero_rows = sorting[jnp.nonzero(zerolag, size=valid.size, fill_value=0)[0]]
    return columns, bkg_rows, zero_rows, jnp.stack((background.sum(), zerolag.sum(), count))


@partial(jax.jit, static_argnames=('work_size', 'active'))
def _resident_live_background(buffer, timers, columns, background, zerolag,
                               counts, index, thresholds, max_count, timing, *,
                               work_size, active):
    """Append/expire background and evaluate IFAR before terminal control read."""
    cstat, _, _, _, expiry0, expiry1, templates, ids0, ids1 = columns
    background_count, zerolag_count, matched_count = counts
    additions = (cstat[background].astype(buffer.dtype),
                 expiry0[background], expiry1[background])
    previous = (buffer,) + timers
    positions = jnp.arange(work_size)
    incoming = jnp.clip(positions - index, 0, background.size - 1)
    appended = tuple(jnp.where((positions >= index) &
                               (positions < index + background_count),
                               addition[incoming], value[:work_size])
                     for value, addition in zip(previous, additions))
    keep = positions < index + background_count
    for column, threshold in zip(active, thresholds):
        keep &= appended[column + 1] >= threshold
    selected = jnp.nonzero(keep, size=work_size, fill_value=0)[0]
    kept_count = keep.sum(dtype=jnp.int64)
    compacted = tuple(jnp.where(positions < kept_count, value[selected], value)
                      for value in appended)
    updated = tuple(jax.lax.dynamic_update_slice(old, value, (0,))
                    for old, value in zip(previous, compacted))
    first = zerolag[0]
    first_stat = cstat[first]
    greater = ((positions < kept_count) & (compacted[0] > first_stat)).sum(
        dtype=jnp.int64)
    # Both denominators stay dynamic so XLA cannot replace the native IEEE
    # divisions with multiplication by rounded reciprocals.
    years = jax.lax.optimization_barrier(timing[0] / timing[1])
    ifar = years / (greater + 1)
    control = jnp.stack((background_count, zerolag_count, matched_count,
                         max_count, kept_count, greater, templates[first],
                         ids0[first], ids1[first])).astype(jnp.int64)
    return updated, cstat[zerolag], control, (ifar, greater == 0)


def _resident_foreground_strong_keys(estimator, candidates, template, devices):
    """Retain weak-scalar promotion using pre-pruning column metadata only."""
    templates = {key for _, key in candidates}
    result = {}
    for ifo in estimator.ifos:
        ring = estimator.singles[ifo]
        selected = ring.buffer[template]
        if type(selected) is not dict:
            continue
        keys = tuple(key for key, value in selected.items()
                     if isinstance(value, jax.Array) and value.ndim == 1 and
                     jnp.issubdtype(value.dtype, jnp.number) and value.weak_type)
        if not keys:
            continue
        rows = tuple(ring.buffer[key] for key in templates
                     if ring.buffer[key] is not None and
                     ring.buffer[key]['end_time'].size)
        result[ifo] = frozenset(
            key for key in keys if all(
                key in row and isinstance(row[key], jax.Array) and
                row[key].ndim == 1 and row[key].dtype == selected[key].dtype and
                row[key].devices() == devices for row in rows))
    return result


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
    # A custom ranker can inspect state or mutate the singles ring. Preserve
    # its existing scalar-call and post-ranking expiration-read boundaries.
    ranker = estimator.stat_calculator
    prepare_matches = (
        type(ranker) is QuadratureSumStatistic and
        getattr(ranker.rank_stat_coinc, "__func__", None) is
        _quadrature_rank_stat)
    resident = _prepare_resident_live_matches(estimator, results, valid_ifos)

    # Calculate all the permutations of coincident triggers for each
    # new single detector trigger collected
    # Currently only two detectors are supported.
    # For each ifo, check its newly added triggers for (zerolag and time
    # shift) coincs with all currently stored triggers in the other ifo.
    # Do this by keeping the ifo with new triggers fixed and time shifting
    # the other ifo. The list 'shift_vec' must be in the same order as
    # estimator.ifos and contain -1 for the shift_ifo / 0 for the fixed_ifo.
    for fixed_ifo, shift_ifo, shift_vec in (() if resident is not None else zip(
        [estimator.ifos[0], estimator.ifos[1]],
        [estimator.ifos[1], estimator.ifos[0]],
        [[0, -1], [-1, 0]]
    )):
        if fixed_ifo not in valid_ifos:
            # This ifo is not online now, so no new triggers or coincs
            continue
        # Find newly added triggers in fixed_ifo
        trigs = results[fixed_ifo]
        templates = np.asarray(trigs['template_id'])
        if len(trigs['end_time']) and estimator.trig_stat_memory is None:
            # Retain the native first-trigger dtype even when its shifted
            # template has no stored triggers.
            estimator.trig_stat_memory = jnp.zeros(1, dtype=trigs['stat'].dtype)
        # The original quadrature statistic ignores mchirp. Other rankers,
        # conversion hooks and array wrappers retain their vector/scalar path.
        skip_mchirp = _can_skip_live_mchirp(ranker, trigs)
        mchirps = None if skip_mchirp else conv.mchirp_from_mass1_mass2(
            jnp.asarray(trigs['mass1']), jnp.asarray(trigs['mass2']))
        # Loop over them one trigger at a time
        queries, prepared_payloads, prepared_ranks = {}, {}, {}
        for i in range(len(trigs['end_time'])):
            if i % 8 == 0:
                queries = _prepare_live_queries(
                    estimator.singles[shift_ifo], trigs, templates, i,
                    min(i + 8, len(trigs['end_time'])), ranker,
                    estimator.time_window, estimator.timeslide_interval)
                prepared_payloads = _prepare_live_payloads(
                    estimator.singles[shift_ifo], estimator.singles[fixed_ifo],
                    trigs, templates, i, min(i + 8, len(trigs['end_time'])),
                    queries, estimator.trig_stat_memory)
                prepared_ranks = _rank_live_matches(ranker, prepared_payloads)
            template = int(templates[i])
            shifted = estimator.singles[shift_ifo].data(template)
            if not shifted or len(shifted['end_time']) == 0:
                continue
            # Get current shift_ifo triggers in the same template
            times = shifted['end_time']
            stats = shifted['stat']

            # Perform coincidence. i1 is the list of trigger indices in the
            # shift_ifo which make coincs, slide is the corresponding slide
            # index.
            # (The second output would just be a list of zeroes as we only
            # have one trigger in the fixed_ifo.)
            query = queries.get(i)
            prepared_payload = prepared_payloads.get(i)
            if prepared_payload is not None:
                (_, _, _, i1, slide, fixed_time, _) = prepared_payload
            elif query is None:
                fixed_time = _live_fixed_time_jax(trigs['end_time'], i)
                i1, _, slide = time_coincidence(
                    times, fixed_time, estimator.time_window,
                    estimator.timeslide_interval)
            else:
                indices, slides, fixed_times, position, count = query
                if count == 0:
                    continue
                i1, slide, fixed_time = _compact_live_query(
                    indices, slides, fixed_times, np.int64(position),
                    count=count)
            if len(i1) == 0:
                continue

            # Make a copy of the fixed ifo trig_stat for each coinc.
            # NB for some statistics the "stat" entry holds more than just
            # a ranking number. E.g. for the phase time consistency test,
            # it must also contain the phase, time and sensitivity.
            while (prepared_payload is None and
                   len(estimator.trig_stat_memory) < len(i1)):
                estimator.trig_stat_memory = jnp.pad(
                    estimator.trig_stat_memory,
                    (0, len(estimator.trig_stat_memory)),
                )
            payload = None
            mchirp = None
            if prepared_payload is not None:
                (estimator.trig_stat_memory, fixed_stats, shifted_stats,
                 _, _, _, payload) = prepared_payload
            elif prepare_matches:
                expiration = estimator.singles[shift_ifo].expire_vector(template)
                clock = estimator.singles[fixed_ifo].time - 1
                values = (estimator.trig_stat_memory, trigs['stat'], mchirps,
                          fixed_time, times, stats, expiration)
                if _can_prepare_live_match(values, i1, template, clock):
                    (estimator.trig_stat_memory, fixed_stats, shifted_stats,
                     mchirp, payload) = _prepare_live_match(
                        *values[:3], i, *values[3:], i1, template, clock)
            if payload is None:
                trig_stat = trigs['stat'][i]
                trig_time = trigs['end_time'][i]
                mchirp = None if skip_mchirp else mchirps[i]
                estimator.trig_stat_memory = (
                    estimator.trig_stat_memory.at[:len(i1)].set(trig_stat))
                fixed_stats = estimator.trig_stat_memory[:len(i1)]
                shifted_stats = stats[i1]

            # Force data into form needed by stat.py and then compute the
            # ranking statistic values.
            sngls_list = [[fixed_ifo, fixed_stats],
                          [shift_ifo, shifted_stats]]

            c = prepared_ranks.get(i)
            if c is None:
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
            if payload is not None:
                (shifted_times, fixed_times, shifted_expiration,
                 fixed_expiration, matched_templates, shifted_ids,
                 fixed_ids) = payload
            else:
                shifted_times = times[i1]
                fixed_times = jnp.full(len(c), trig_time, dtype=jnp.float64)
                shifted_expiration = (
                    estimator.singles[shift_ifo].expire_vector(template)[i1])
                fixed_expiration = jnp.full(
                    len(c), estimator.singles[fixed_ifo].time - 1,
                    dtype=jnp.int32)
                matched_templates = jnp.zeros(len(c), dtype=jnp.int32) + template
                shifted_ids = i1
                fixed_ids = jnp.zeros(len(c)) - 1
            ctimes[shift_ifo].append(shifted_times)
            ctimes[fixed_ifo].append(fixed_times)

            # As background triggers are removed after a certain time, we
            # need to log when this will be for new background triggers.
            single_expire[shift_ifo].append(shifted_expiration)
            single_expire[fixed_ifo].append(fixed_expiration)

            # Save the template and trigger ids to keep association
            # to singles. The trigger was just added so it must be in
            # the last position: we mark this with -1 so the
            # slicing picks the right point
            template_ids.append(matched_templates)
            trigger_ids[shift_ifo].append(shifted_ids)
            trigger_ids[fixed_ifo].append(fixed_ids)

    if resident is not None:
        ifo0, ifo1 = estimator.ifos
        (cstat, offsets, ctimes[ifo0], ctimes[ifo1], single_expire[ifo0],
         single_expire[ifo1], template_ids, joined_ids0,
         joined_ids1) = resident[0]
        for ifo, ids in ((ifo0, joined_ids0), (ifo1, joined_ids1)):
            if ifo in valid_ifos:
                trigger_ids[ifo] = ids
    elif cstat:
        ifo0, ifo1 = estimator.ifos
        payloads = tuple(zip(cstat, offsets, ctimes[ifo0], ctimes[ifo1],
                             single_expire[ifo0], single_expire[ifo1],
                             template_ids, trigger_ids[ifo0], trigger_ids[ifo1]))
        (cstat, offsets, ctimes[ifo0], ctimes[ifo1], single_expire[ifo0],
         single_expire[ifo1], template_ids, joined_ids0,
         joined_ids1) = _join_live_matches(payloads)
        for ifo, ids in ((ifo0, joined_ids0), (ifo1, joined_ids1)):
            if ifo in valid_ifos:
                trigger_ids[ifo] = ids
    else:
        cstat = jnp.empty(0, dtype=jnp.float64)
        template_ids = jnp.empty(0, dtype=jnp.int32)
        for ifo in valid_ifos:
            trigger_ids[ifo] = jnp.empty(0, dtype=jnp.int32)

    if resident is None:
        logger.info(
            "%s: %s background and zerolag coincs",
            ppdets(estimator.ifos, "-"), len(cstat))

    # Cluster the triggers we've found
    # (both zerolag and shifted are handled together)
    num_zerolag = 0
    num_background = 0
    selected_cluster = None
    resident_ifar = None
    resident_strong_keys = {}
    if resident is not None:
        with host_stage('coinc_resident_cluster_dispatch'):
            packed, background, zerolag, counts = _resident_live_cluster(
                resident[0], resident[1], estimator.timeslide_interval,
                estimator.analysis_block + 2 * estimator.time_window)
            buffer = estimator.coincs
            active = tuple(buffer.ifos.index(ifo) for ifo in valid_ifos)
            thresholds = tuple(np.int32(buffer.time[ifo] + 1 - buffer.expiration)
                               for ifo in valid_ifos)
            storage = resident[5]
            work_size = min(storage[0].size, 1 << (
                int(max(buffer.index + resident[1].size, 1)) - 1).bit_length())
            updated, padded_zerolag, control, resident_ifar = _resident_live_background(
                storage[0], storage[1:], packed,
                background, zerolag, counts, np.int64(buffer.index), thresholds,
                resident[3], np.asarray(
                    [estimator.background_time, conv.YRJUL_SI], np.float64),
                work_size=work_size, active=active)
        with host_stage('coinc_resident_publication_readback'):
            (num_background, num_zerolag, matched_count, max_count, kept_count,
             greater, template, id0, id1) = map(int, np.asarray(control))
        if num_zerolag:
            resident_strong_keys = _resident_foreground_strong_keys(
                estimator, resident[4], template, packed[6].devices())
        logger.info("%s: %s background and zerolag coincs",
                    ppdets(estimator.ifos, "-"), matched_count)
        memory_size = (1 if estimator.trig_stat_memory is None else
                       estimator.trig_stat_memory.size)
        while memory_size < max_count:
            memory_size *= 2
        estimator.trig_stat_memory = resident[2][:memory_size]
        for ring, prune_template in resident[4]:
            ring._prune_ring(prune_template)
        # Numeric background state and IFAR are already ready. Only routing,
        # dynamic public shapes and native scalar serialization remain here.
        buffer.buffer = updated[0]
        buffer.timer = dict(zip(buffer.ifos, updated[1:]))
        buffer.index = kept_count
        for ifo in valid_ifos:
            buffer.time[ifo] += 1
        (cstat, offsets, ctimes[ifo0], ctimes[ifo1], single_expire[ifo0],
         single_expire[ifo1], template_ids, trigger_ids[ifo0],
         trigger_ids[ifo1]) = packed
        if matched_count:
            selected_cluster = True
            zerolag_cstat = padded_zerolag[:num_zerolag]
            first_stat = padded_zerolag[0]
    elif len(cstat) > 0:
        ctime0 = ctimes[estimator.ifos[0]]
        ctime1 = ctimes[estimator.ifos[1]]
        logger.info("Clustering %s coincs", ppdets(estimator.ifos, "-"))
        cidx = cluster_coincs(cstat, ctime0, ctime1, offsets,
                              estimator.timeslide_interval,
                              estimator.analysis_block + 2*estimator.time_window,
                              method='cython')
        selected_cluster = _prepare_live_cluster(
            cidx, offsets, (cstat, single_expire[ifo0], single_expire[ifo1],
                            template_ids, trigger_ids[ifo0], trigger_ids[ifo1]))
        if selected_cluster is None:
            offsets = offsets[cidx]
            zerolag_idx = (offsets == 0)
            bkg_idx = (offsets != 0)
            for ifo in estimator.ifos:
                single_expire[ifo] = single_expire[ifo][cidx][bkg_idx]
            background_stat = cstat[cidx][bkg_idx]
            num_zerolag = zerolag_idx.sum()
            num_background = bkg_idx.sum()
        else:
            (background_stat, single_expire[ifo0], single_expire[ifo1],
             zerolag_cstat, first_stat, template, id0, id1,
             num_background, num_zerolag) = selected_cluster
        estimator.coincs.add(background_stat, single_expire, valid_ifos)
    elif len(valid_ifos) > 0:
        estimator.coincs.increment(valid_ifos)

    # Collect coinc results for saving
    coinc_results = {}
    # Save information about zerolag triggers
    if num_zerolag > 0:
        if selected_cluster is None:
            idx = cidx[zerolag_idx][0]
            zerolag_cstat = cstat[cidx][zerolag_idx]
            first_stat = zerolag_cstat[0]
        ifar, ifar_sat = (estimator.ifar(first_stat) if resident_ifar is None else
                          resident_ifar)
        zerolag_results = {
            'foreground/ifar': ifar,
            'foreground/ifar_saturated': ifar_sat,
            'foreground/stat': zerolag_cstat,
            'foreground/type': '-'.join(estimator.ifos)
        }
        if selected_cluster is None:
            template = template_ids[idx]
        for ifo in estimator.ifos:
            trig_id = (trigger_ids[ifo][idx] if selected_cluster is None else
                       id0 if ifo == ifo0 else id1)
            stored = estimator.singles[ifo].data(template)
            gathered = _prepare_live_foreground(stored, trig_id)
            strong_keys = resident_strong_keys.get(ifo, ())
            for key, value in stored.items():
                path = f'foreground/{ifo}/{key}'
                selected = gathered[key] if key in gathered else value[trig_id]
                # Former resident zero/where selection strengthened common
                # weak columns, including fields on the serial gather fallback.
                if (key in strong_keys and isinstance(selected, jax.Array) and
                        selected.weak_type):
                    selected = selected.astype(selected.dtype)
                zerolag_results[path] = selected
        coinc_results.update(zerolag_results)

    # Save some summary statistics about the background
    coinc_results['background/time'] = jnp.array([estimator.background_time])
    coinc_results['background/count'] = len(estimator.coincs.data)

    # Save all the background triggers
    if estimator.return_background:
        coinc_results['background/stat'] = estimator.coincs.data

    return num_background, coinc_results


class JAXLiveCoincTimeslideBackgroundEstimator(
        _coinc_module.LiveCoincTimeslideBackgroundEstimator):
    """Device-resident two-detector background with quadrature ranking."""

    _coinc_buffer_type = JAXCoincExpireBuffer
    _singles_buffer_type = JAXMultiRingBuffer

    def __init__(self, num_templates, analysis_block, background_statistic,
                 sngl_ranking, stat_files, ifos, ifar_limit=100,
                 timeslide_interval=.035, coinc_window_pad=.002,
                 statistic_refresh_rate=None, return_background=False,
                 **kwargs):
        if (_coinc_module.pycbcstat.get_statistic(background_statistic)
                is not QuadratureSumStatistic or
                sngl_ranking not in ('snr', 'newsnr', 'new_snr')):
            raise NotImplementedError(
                'JAX live coincidence supports quadrature_sum with snr/newsnr')
        super().__init__(
            num_templates, analysis_block, background_statistic, sngl_ranking,
            stat_files, ifos, ifar_limit=ifar_limit,
            timeslide_interval=timeslide_interval,
            coinc_window_pad=coinc_window_pad,
            statistic_refresh_rate=statistic_refresh_rate,
            return_background=return_background, **kwargs)

    @classmethod
    def pick_best_coinc(cls, coinc_results):
        return pick_best_coinc_jax(coinc_results, _coinc_module.logger)

    def _add_singles_to_buffer(self, results, ifos):
        return add_singles_to_buffer_jax(self, results, ifos, _coinc_module.logger)

    def _find_coincs(self, results, valid_ifos):
        return find_coincs_jax(self, results, valid_ifos)

    def add_singles(self, results):
        # Preserve the caller's output columns while the estimator adds stat.
        values = {ifo: dict(triggers) if triggers else triggers
                  for ifo, triggers in results.items()}
        return super().add_singles(values)
