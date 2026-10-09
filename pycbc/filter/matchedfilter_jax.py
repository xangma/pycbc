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

"""JAX backend for matched filtering primitives, correlation,
sigmasq, and match calculations.

Functional helpers support JIT with the default JAX kernels. Optional CPU
validation routes perform host work and run outside JAX transformations.
"""

import functools
import math
import os
import weakref
from collections import namedtuple
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.events import ranking
from pycbc.filter.matchedfilter import (
    _BaseCorrelator, MatchedFilterControl, get_cutoff_indices,
)
from pycbc.types.array_jax import (
    JAXArrayData,
    _as_jax_array,
    _ensure_x64,
    _cpu_reference,
    _divide,
    _fast_inner,
    _fast_inner_self,
    _fast_weighted_inner,
    _fast_weighted_inner_self,
    _reference_enabled,
    to_jax,
)


_EMPTY_F32 = np.zeros(0, dtype=np.float32)
_EMPTY_U32 = np.zeros(0, dtype=np.uint32)

def _functional_device(*inputs):
    """Use the active scheme, an existing input device, or JAX's default."""
    from pycbc import scheme

    if isinstance(scheme.mgr.state, scheme.JAXScheme):
        return scheme.mgr.state.jax_device
    arrays = [_as_jax_array(value) for value in inputs]
    # Traced computations inherit their execution device from JAX rather than
    # committing intermediate arrays to the device available during tracing.
    if any(isinstance(value, jax.core.Tracer) for value in arrays):
        return None
    for value in arrays:
        if value is not None:
            return value.device
    return jax.config.jax_default_device or jax.devices()[0]


def _functional_array(value, device):
    """Place eager inputs together while leaving JAX tracers untouched."""
    array = _as_jax_array(value)
    return (array if isinstance(array, jax.core.Tracer)
            else to_jax(value, device=device))


def _live_array_metadata(value):
    """Read fixed geometry without materializing a lazy device row/view."""
    if isinstance(value, JAXArrayData):
        return value.shape[0], value.dtype
    stored = getattr(value, "__dict__", {})
    data = stored.get("_data_inst", stored.get("_data", stored.get("data")))
    if data is not None:
        return _live_array_metadata(data)
    tensor = stored.get("_batch_tensor")
    if tensor is not None:
        return tensor.shape[1], tensor.dtype
    return len(value), getattr(value, "dtype", None)


def _live_veto_executable_cache(control, power_chisq=None):
    """Share the existing retained cache across preparation and veto work."""
    if power_chisq is None:
        power_chisq = getattr(control, "power_chisq", None)
    cache = getattr(power_chisq, "_jax_live_veto_executables", None)
    if cache is None:
        cache = getattr(control, "_jax_live_veto_executables", None)
    if cache is None:
        cache = {}
    control._jax_live_veto_executables = cache
    if power_chisq is not None:
        power_chisq._jax_live_veto_executables = cache
    return cache


@functools.partial(jax.jit, static_argnames=("count",))
def _live_compact_columns(*columns, count):
    """Publish exact-length columns at the sparse trigger API boundary.

    This final slice still specializes by count; the preceding selection and
    order reductions use stable buckets and retained executable handles.
    """
    return tuple(column[:count] for column in columns)


def _live_compact_results(columns, count, cache, device):
    from pycbc.vetoes.chisq_jax import _live_chisq_executable

    if count == columns[0].shape[0]:
        return columns
    execute = _live_chisq_executable(
        _live_compact_columns, columns, dict(count=count), cache, device)
    return execute(*columns)


def batch_template_matrix_jax(templates, size):
    """Pack immutable live templates on the active JAX device."""
    if templates:
        first = templates[0]
        batch_tensor = getattr(first, "_batch_tensor", None)
        if batch_tensor is not None and all(
            getattr(template, "_batch_tensor", None) is batch_tensor
            for template in templates
        ):
            positions = [getattr(t, "_batch_pos", None) for t in templates]
            if None not in positions:
                start = positions[0]
                shape = getattr(batch_tensor, "shape", ())
                if positions == list(range(start, start + len(positions))):
                    if len(shape) == 2 and start >= 0:
                        stop = start + len(positions)
                        if stop <= shape[0] and size <= shape[1]:
                            return to_jax(batch_tensor)[start:stop, :size]
                if (
                    len(shape) == 2
                    and all(0 <= p < shape[0] for p in positions)
                    and size <= shape[1]
                ):
                    pos_idx = jnp.asarray(positions, dtype=jnp.int32)
                    return to_jax(batch_tensor)[pos_idx, :size]
    return jnp.stack(
        [to_jax(template)[:size] for template in templates], axis=0
    )


@jax.jit
def batch_template_power_jax(template_matrix, delta_f):
    """Cache the PSD-independent part of generic live normalization."""
    return (
        template_matrix.real ** 2 + template_matrix.imag ** 2
    ) * (4.0 * delta_f)


def batch_correlator_init_jax(control, immutable_templates=False):
    """Retain JAX rows without materializing native pointer tables."""
    control.x = control.z = None
    control._jax_template_matrix = None
    control._jax_template_power = None
    if immutable_templates:
        control._jax_template_matrix = batch_template_matrix_jax(
            control.xs, control.size)
        if (hasattr(control.xs[0], "delta_f")
                and not _reference_enabled("squared_norm")):
            control._jax_template_power = batch_template_power_jax(
                control._jax_template_matrix, float(control.xs[0].delta_f))



@jax.jit
def _live_generic_norms_core(template_power, psd, kmins, kmaxs):
    """Evaluate all generic live template norms in one compiled reduction."""
    from pycbc.psd.estimate_jax import _reciprocal_numpy_compat

    bins = jnp.arange(template_power.shape[1])[None, :]
    support = (bins >= kmins[:, None]) & (bins < kmaxs[:, None])
    weighted = template_power * _reciprocal_numpy_compat(psd)[None, :]
    return jnp.sum(
        jnp.where(support, weighted, 0.0), axis=-1, dtype=jnp.float64
    )


def live_template_norms_jax(
    templates, psd, template_matrix=None, template_power=None
):
    """Preserve ``sigma_cached`` frequency support and accumulation precision.

    The native generic cache forms a float32 squared-magnitude view, multiplies
    it by ``4 * delta_f``, weights it by the inverse PSD, and accumulates
    in float64. Selected squared-norm, division and inner-product controls
    independently restore the original CPU implementations.
    """
    _ensure_x64()
    if not templates:
        return jnp.zeros(0, dtype=jnp.float64)
    device = _functional_device(
        *templates, psd, template_matrix, template_power
    )
    delta_f = float(templates[0].delta_f)
    psd_j = _functional_array(psd, device)
    psd._jax_psd = psd_j
    kmins, kmaxs = [], []
    for template in templates:
        flow = getattr(template, "min_f_lower", None) or getattr(
            template, "f_lower", 0.0)
        fhigh = getattr(template, "end_frequency", None)
        size = (template_matrix.shape[1] if template_matrix is not None
                else _live_array_metadata(template)[0])
        if fhigh is None and not hasattr(template, "end_frequency"):
            kmin, kmax = (int(flow / delta_f) if flow else 1), size
        else:
            kmin, kmax = get_cutoff_indices(
                flow, fhigh, delta_f, (size - 1) * 2)
        kmins.append(kmin)
        kmaxs.append(kmax)

    if template_power is not None and not _reference_enabled("squared_norm"):
        power = _functional_array(template_power, device)
    else:
        matrix = (jnp.stack([_functional_array(t, device) for t in templates])
                  if template_matrix is None else
                  _functional_array(template_matrix, device))
        power = (jnp.stack([
            _functional_array(_cpu_reference(row, "squared_norm"), device)
            * 4.0 * delta_f for row in matrix])
            if _reference_enabled("squared_norm") else
            batch_template_power_jax(matrix, delta_f))
    if _reference_enabled("inner") or _reference_enabled("divide"):
        from pycbc.psd.estimate_jax import _reciprocal_numpy_compat
        inverse = (_divide(1.0, psd_j) if _reference_enabled("divide")
                   else _reciprocal_numpy_compat(psd_j))
        return jnp.stack([
            (_functional_array(_cpu_reference(
                row[kmin:kmax], "inner", inverse[kmin:kmax]), device)
             if _reference_enabled("inner") else jnp.sum(
                 row[kmin:kmax] * inverse[kmin:kmax], dtype=jnp.float64))
            for row, kmin, kmax in zip(power, kmins, kmaxs)])
    return _live_generic_norms_core(
        power, psd_j, jnp.asarray(kmins, dtype=jnp.int32),
        jnp.asarray(kmaxs, dtype=jnp.int32))


def _live_cached_template_norms_jax(correlator, templates, psd):
    """Reuse complete norms for one immutable Live group and PSD lifecycle.

    The cache belongs to the interpolated Live PSD, which is replaced when
    strain conditioning recalculates its PSD. Weak correlator keys distinguish
    template groups without extending their lifetime. Mutable correlators and
    the public normalization helper retain their uncached behavior. As with
    the packed immutable template matrix, template contents and normalization
    metadata remain fixed for the lifetime of each owned group.
    """
    from pycbc.scheme import current_backend_key

    backend_key = current_backend_key()
    psd._jax_psd = to_jax(psd)
    matrix = getattr(correlator, "_jax_template_matrix", None)
    power = getattr(correlator, "_jax_template_power", None)
    immutable = matrix is not None and getattr(correlator, "xs", None) is templates
    cache = getattr(psd, "_jax_live_template_norms", None)
    if immutable and cache is not None:
        cached = cache.get(correlator)
        if (cached is not None and cached[0] is matrix
                and cached[1] is power and cached[2] is templates
                and cached[3] is getattr(psd, "_jax_psd", None)
                and cached[4] == backend_key):
            return cached[5:]

    native = live_template_norms_jax(
        templates, psd, template_matrix=matrix, template_power=power)
    norms = (4.0 * templates[0].delta_f) / jnp.sqrt(native)
    if immutable:
        if cache is None:
            cache = psd._jax_live_template_norms = weakref.WeakKeyDictionary()
        cache[correlator] = (matrix, power, templates, psd._jax_psd,
                             backend_key, native, norms)
    return native, norms


def live_veto_buffer_jax(size, dtype):
    """Allocate a standalone JAX veto correlation buffer."""
    from pycbc.types import zeros
    return zeros(size, dtype=dtype)


class _LiveVetoCandidate:
    """Seven-field veto sequence with lazy scalar views of shared vectors.

    Metadata survives ``process_all`` concatenation and trigger sorting. Only
    the scalar fallback or SG veto needs to resolve individual SNR/norm rows.
    """

    def __init__(self, snrs, norms, row, point, template, stilde, source,
                 position):
        self.snrs, self.norms, self.row = snrs, norms, row
        self.metadata = (point, template, stilde, source, position)

    def __len__(self):
        return 7

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = range(len(self))[index]
        if index == 0:
            return self.snrs[self.row:self.row + 1]
        if index == 1:
            return self.norms[self.row]
        return self.metadata[index - 2]


@jax.jit
def _live_gather_veto_values(snrs, norms, rows, order):
    snr = jnp.concatenate([values[index] for values, index in zip(snrs, rows)])
    norm = jnp.concatenate([values[index]
                            for values, index in zip(norms, rows)])
    return snr[order], norm[order].astype(jnp.float64)


@jax.jit
def _live_stack_veto_values(snrs, norms):
    return (jnp.stack([jnp.ravel(value)[0] for value in snrs]),
            jnp.asarray(norms, dtype=jnp.float64))


def _live_veto_values(veto_info):
    """Gather batch vectors once, retaining final trigger order."""
    if not all(isinstance(info, _LiveVetoCandidate) for info in veto_info):
        return _live_stack_veto_values(
            tuple(info[0] for info in veto_info),
            tuple(info[1] for info in veto_info))

    groups = {}
    for index, info in enumerate(veto_info):
        groups.setdefault((id(info.snrs), id(info.norms)), []).append(index)
    snrs, norms, rows = [], [], []
    order = np.empty(len(veto_info), dtype=np.int32)
    offset = 0
    for indices in groups.values():
        first = veto_info[indices[0]]
        snrs.append(first.snrs)
        norms.append(first.norms)
        rows.append(np.asarray([veto_info[index].row for index in indices],
                               dtype=np.int32))
        order[indices] = np.arange(offset, offset + len(indices))
        offset += len(indices)
    return _live_gather_veto_values(tuple(snrs), tuple(norms), tuple(rows),
                                  order)


def _cache_live_veto_bins_jax(power_chisq, veto_info):
    """Populate exact bin caches from the live device template matrices."""
    from collections import defaultdict
    from pycbc.benchmark import stage_event
    from pycbc.vetoes.chisq_jax import (
        _collect_cached_power_chisq_bins_jax,
        cache_batch_power_chisq_bins_jax,
    )
    from pycbc.waveform.bank_jax import TemplateBatchList

    by_source = defaultdict(list)
    psds = {}
    for info in veto_info:
        template, stilde = info[3:5]
        if (id(stilde.psd) in getattr(template, "_bin_cache", {})
                and id(template.params) in getattr(
                    stilde.psd, "_chisq_cached_key", {})):
            continue
        source = info[5] if len(info) > 5 else None
        position = info[6] if len(info) > 6 else None
        key = (id(stilde.psd), id(source))
        psds[key] = stilde.psd
        by_source[key].append((template, source, position))

    pending = []
    try:
        for key, entries in by_source.items():
            templates = TemplateBatchList([entry[0] for entry in entries])
            source = entries[0][1]
            positions = [entry[2] for entry in entries]
            if source is not None and all(position is not None
                                          for position in positions):
                templates._batch_tensor = source
                templates._batch_positions = positions
            stage_event("filter_veto_bins", "start", templates=len(templates))
            cache_batch_power_chisq_bins_jax(
                power_chisq, templates, psds[key], pending=pending
            )
            stage_event("filter_veto_bins", "end", templates=len(templates))
    finally:
        if pending:
            groups = len(pending)
            stage_event("filter_veto_bins_collect", "start", groups=groups)
            try:
                _collect_cached_power_chisq_bins_jax(pending)
            finally:
                stage_event("filter_veto_bins_collect", "end", groups=groups)


@functools.partial(
    jax.jit,
    static_argnames=("base_k", "compatible", "use_pallas",
                     "snr_threshold"),
)
def _live_veto_group_core(
        source, stilde, snrs, norms, positions, points, value_rows, bins,
        valid_count, n_time, *, base_k, compatible, use_pallas,
        snr_threshold):
    """Fuse one bucket on its original Live frequency and time grids."""
    from pycbc.vetoes.chisq_jax import (
        _batched_points_chisq_core, _compatible_shift_sum_jax,
        _compatible_shift_sum_pallas,
    )

    # Preserve the materialized complex multiplication and float32 arithmetic
    # boundaries of the unfused path, including its ordered phase recurrence.
    correlations = jax.lax.optimization_barrier(
        jnp.conj(source[positions, base_k:]) * stilde[None, base_k:])
    rows = jnp.arange(points.size, dtype=jnp.int32)
    if compatible:
        ordered_sum = (_compatible_shift_sum_pallas if use_pallas
                       else _compatible_shift_sum_jax)
        shifts = ordered_sum(correlations, rows, points, bins, n_time, base_k)
    else:
        shifts = _batched_points_chisq_core(
            correlations, rows, points, bins - base_k,
            float(base_k), n_time.astype(jnp.float64))
    snr = snrs[value_rows]
    norm = norms[value_rows].astype(jnp.float64)
    num_bins = bins.shape[1] - 1
    bin_power = jax.lax.optimization_barrier(shifts * num_bins)
    snr_power = jax.lax.optimization_barrier((snr.conj() * snr).real)
    raw = jax.lax.optimization_barrier(bin_power - snr_power)
    norm_squared = jax.lax.optimization_barrier(
        (norm ** 2.0).astype(shifts.dtype))
    raw *= norm_squared
    dof = jnp.full(points.shape, num_bins * 2 - 2, dtype=jnp.int32)
    if snr_threshold:
        above = abs(snr * norm) > snr_threshold
        raw = jnp.where(above, raw, 0.0)
        dof = jnp.where(above, dof, -100)
    active = jnp.arange(points.size) < valid_count
    return jnp.where(active, raw, 0.0), jnp.where(active, dof, 0)


@jax.jit
def _live_restore_veto_order(order, *groups):
    """Restore bucket-sized trigger order; source group arity stays static."""
    midpoint = len(groups) // 2
    raw_groups, dof_groups = groups[:midpoint], groups[midpoint:]
    raw = jnp.concatenate(raw_groups)[order]
    dof = jnp.concatenate(dof_groups)[order]
    return (raw, dof, (raw / dof).astype(jnp.float32),
            dof.astype(jnp.uint32), jnp.zeros(order.shape, jnp.float32))


def _bucketed_live_vetoes(control, results, veto_info, bins_list, power_chisq):
    """Dispatch stable source/vector shapes rather than sparse scalar reads."""
    from collections import defaultdict
    from pycbc.vetoes.chisq_jax import (
        _chisq_candidate_bucket, _chisq_mode, _live_chisq_executable,
        _require_point_chisq_x64, _use_gpu_ordered_scan,
    )

    groups = defaultdict(list)
    for index, info in enumerate(veto_info):
        template, stilde = info[3:5]
        kmin = int(template.f_lower / template.delta_f)
        n_time = _live_array_metadata(
            template.cout if hasattr(template, "cout") else stilde)[0]
        template_size, template_dtype = _live_array_metadata(template)
        key = (id(info[5]), id(stilde), id(stilde.psd), kmin, n_time,
               template_size, float(template.delta_f), str(template_dtype),
               len(bins_list[index]), id(info.snrs), id(info.norms))
        groups[key].append(index)

    _require_point_chisq_x64()
    cache = _live_veto_executable_cache(control, power_chisq)
    raw_groups, dof_groups = [], []
    order = np.zeros(_chisq_candidate_bucket(len(veto_info)), dtype=np.int32)
    offset = 0
    for key, indices in groups.items():
        infos = [veto_info[index] for index in indices]
        first = infos[0]
        source = to_jax(first[5])
        bucket = _chisq_candidate_bucket(len(infos))
        positions = np.zeros(bucket, dtype=np.int32)
        points = np.zeros(bucket, dtype=np.int64)
        value_rows = np.zeros(bucket, dtype=np.int32)
        bins = np.zeros((bucket, key[8]), dtype=np.int32)
        positions[:len(infos)] = [info[6] for info in infos]
        points[:len(infos)] = [info[2] for info in infos]
        value_rows[:len(infos)] = [info.row for info in infos]
        bins[:len(infos)] = [bins_list[index] for index in indices]
        args = (source, to_jax(first[4]), first.snrs, first.norms,
                positions, points, value_rows, bins,
                np.asarray(len(infos), dtype=np.int32),
                np.asarray(key[4], dtype=np.int64))
        static_args = dict(
            base_k=key[3],
            compatible=_chisq_mode() == "cpu-compatible",
            use_pallas=(source.dtype == jnp.complex64
                        and _use_gpu_ordered_scan(source)),
            snr_threshold=getattr(power_chisq, "snr_threshold", None))
        execute = _live_chisq_executable(
            _live_veto_group_core, args, static_args, cache, source.device)
        raw, dof = execute(*args)
        raw_groups.append(raw)
        dof_groups.append(dof)
        order[indices] = np.arange(offset, offset + len(infos))
        offset += bucket

    args = (order, *raw_groups, *dof_groups)
    execute = _live_chisq_executable(
        _live_restore_veto_order, args, {}, cache, source.device)
    columns = execute(*args)
    raw, dof, reduced, dof_u32, sg = _live_compact_results(
        columns, len(veto_info), cache, source.device)
    return _store_live_veto_results(
        control, results, veto_info, raw, dof, reduced, dof_u32, sg)


def _store_live_veto_results(
        control, results, veto_info, raw, dof, reduced, dof_u32, sg):
    """Keep optional SG and ranking behavior at the existing API boundary."""
    if getattr(control.sg_chisq, "do", False):
        sg_values = []
        for i, info in enumerate(veto_info):
            snrv, norm, point, template, stilde = info[:5]
            value = control.sg_chisq.values(
                stilde, template, stilde.psd, snrv, norm,
                raw[i:i + 1], dof[i:i + 1], [point])
            sg_values.append(jnp.asarray(
                0 if value is None else value[0], dtype=jnp.float32))
        sg = jnp.stack(sg_values)
    results["chisq"] = reduced
    results["chisq_dof"] = dof_u32
    results["sg_chisq"] = sg
    if control.newsnr_threshold:
        keep_mask = ranking.newsnr(results["snr"], reduced) >= (
            control.newsnr_threshold)
        keep = jnp.asarray(np.flatnonzero(jax.device_get(keep_mask)),
                           dtype=jnp.uint32)
        for key in results:
            results[key] = results[key][keep]
    return results


def _batched_live_vetoes_gpu(control, results, veto_info, power_chisq):
    """Batch live vetoes by their template source, data, and FFT geometry."""
    if not veto_info:
        return None
    try:
        from collections import defaultdict
        from pycbc.vetoes.chisq_jax import (
            _batched_points_chisq_core, _chisq_mode,
            _compatible_shift_sum, _require_point_chisq_x64,
        )

        bins_list = []
        for info in veto_info:
            tmpl = info[3]
            stilde = info[4]
            b_edges = None
            if hasattr(power_chisq, "cached_chisq_bins"):
                b_edges = power_chisq.cached_chisq_bins(tmpl, stilde.psd)
            elif (
                hasattr(tmpl, "_bin_cache")
                and id(stilde.psd) in tmpl._bin_cache
            ):
                b_edges = tmpl._bin_cache[id(stilde.psd)]
            if b_edges is None:
                return None
            bins_list.append(b_edges)

        lazy_sources = all(isinstance(info, _LiveVetoCandidate)
                           and info[5] is not None and info[6] is not None
                           for info in veto_info)
        # The legacy gather promotes all SNR vectors together before any
        # arithmetic. Keep mixed-precision inputs on that established path.
        uniform_dtype = lazy_sources and len({
            np.dtype(_live_array_metadata(value)[1]) for info in veto_info
            for value in (info.snrs, info[3], info[4], info[5])
        }) == 1
        device = getattr(veto_info[0][5], "device", None) if lazy_sources else None
        if uniform_dtype and getattr(device, "platform", None) in ("cuda", "gpu"):
            return _bucketed_live_vetoes(
                control, results, veto_info, bins_list, power_chisq)

        snr_arr, norm_arr = _live_veto_values(veto_info)

        groups = defaultdict(list)
        for index, info in enumerate(veto_info):
            template, stilde = info[3:5]
            source = info[5] if len(info) > 6 and info[6] is not None else None
            kmin = int(template.f_lower / template.delta_f)
            n_time = _live_array_metadata(
                template.cout if hasattr(template, "cout") else stilde)[0]
            template_size, template_dtype = _live_array_metadata(template)
            key = (id(source), id(stilde), id(stilde.psd), kmin, n_time,
                   template_size, float(template.delta_f),
                   str(template_dtype), len(bins_list[index]))
            groups[key].append(index)

        _require_point_chisq_x64()
        compatible = _chisq_mode() == "cpu-compatible"
        real_dtype = np.result_type(snr_arr.real.dtype, *[
            np.empty((), dtype=_live_array_metadata(info[3])[1]).real.dtype
            for info in veto_info])
        chisq_raw = jnp.zeros(len(veto_info), dtype=real_dtype)
        dof_values = np.empty(len(veto_info), dtype=np.int32)
        for key, indices in groups.items():
            kmin, n_time = key[3:5]
            infos = [veto_info[index] for index in indices]
            bins_arr = np.asarray([bins_list[index] for index in indices],
                                  dtype=np.int32)
            max_kmax = int(np.max(bins_arr[:, -1]))
            source = infos[0][5] if len(infos[0]) > 6 else None
            if source is not None and infos[0][6] is not None:
                positions = jnp.asarray([info[6] for info in infos], jnp.int32)
                templates_mat = to_jax(source)[positions, kmin:max_kmax]
            else:
                templates_mat = jnp.stack([
                    to_jax(info[3])[kmin:max_kmax] for info in infos])
            stilde_dev = to_jax(infos[0][4])
            corr_tensor = (jnp.conj(templates_mat)
                           * stilde_dev[None, kmin:max_kmax])
            pts = jnp.asarray([int(info[2]) for info in infos], jnp.int64)
            rows = jnp.arange(len(infos), dtype=jnp.int32)
            if compatible:
                shifts = _compatible_shift_sum(
                    corr_tensor, rows, pts, jnp.asarray(bins_arr), n_time, kmin)
            else:
                shifts = _batched_points_chisq_core(
                    corr_tensor, rows, pts, jnp.asarray(bins_arr - kmin),
                    float(kmin), float(n_time))
            selected = jnp.asarray(indices, jnp.int32)
            num_bins = bins_arr.shape[1] - 1
            raw = (shifts * num_bins
                   - (snr_arr[selected].conj() * snr_arr[selected]).real)
            raw *= (norm_arr[selected] ** 2.0).astype(shifts.dtype)
            chisq_raw = chisq_raw.at[selected].set(raw)
            dof_values[indices] = num_bins * 2 - 2
        dof = jnp.asarray(dof_values, dtype=jnp.int32)
        if getattr(power_chisq, "snr_threshold", None):
            above = abs(snr_arr * norm_arr) > power_chisq.snr_threshold
            # The scalar power veto gives gated candidates zero raw chisq and
            # a -100 dof sentinel, which also feeds the sine-Gaussian veto.
            chisq_raw = jnp.where(above, chisq_raw, 0.0)
            dof = jnp.where(above, dof, -100)
        chisq_red = (chisq_raw / dof).astype(jnp.float32)

        if getattr(control.sg_chisq, "do", False):
            sg_values = []
            for i, info in enumerate(veto_info):
                snrv, norm, l, htilde, stilde_i = info[:5]
                c = chisq_raw[i:i + 1]
                d = dof[i:i + 1]
                sgv = control.sg_chisq.values(
                    stilde_i, htilde, stilde_i.psd, snrv, norm, c, d, [l]
                )
                if sgv is not None:
                    sg_values.append(jnp.asarray(sgv[0], dtype=jnp.float32))
                else:
                    sg_values.append(jnp.asarray(0, dtype=jnp.float32))
            sg_chisq = jnp.stack(sg_values)
        else:
            sg_chisq = jnp.zeros(len(veto_info), dtype=jnp.float32)

        results["chisq"] = chisq_red
        results["chisq_dof"] = dof.astype(jnp.uint32)
        results["sg_chisq"] = sg_chisq

        if control.newsnr_threshold:
            keep_mask = ranking.newsnr(results["snr"], chisq_red) >= (
                control.newsnr_threshold
            )
            keep = jnp.asarray(
                np.flatnonzero(jax.device_get(keep_mask)), dtype=jnp.uint32
            )
            for key in results:
                results[key] = results[key][keep]

        return results
    except Exception:
        return None


def process_live_vetoes_jax(control, results, veto_info):
    """Run live veto reductions while retaining numeric result columns on JAX."""
    from pycbc.benchmark import stage_event
    stage_event("filter_veto", "start", triggers=len(veto_info))
    power_chisq = control.power_chisq
    from pycbc import scheme
    device = getattr(scheme.mgr.state, "jax_device", None)
    platform = getattr(device, "platform", None)
    if (
        veto_info
        and platform in ("cuda", "gpu")
        and getattr(power_chisq, "do", False)
        and hasattr(power_chisq, "cached_chisq_bins")
        and not any(_reference_enabled(operation) for operation in (
            "live_selection", "correlate", "divide", "power_chisq_bins",
            "power_chisq_at_points", "shift_sum", "chisq_accum_bin",
            "sg_basis", "sgchisq"))
    ):
        # A live batch has at most one candidate per template. CUDA benefits
        # from replacing several executable loads and dispatches with one
        # exact batched scan. Small CPU batches retain the scalar scans, which
        # have better cache locality.
        _cache_live_veto_bins_jax(power_chisq, veto_info)
        batched_res = _batched_live_vetoes_gpu(
            control, results, veto_info, power_chisq
        )
        if batched_res is not None:
            stage_event("filter_veto", "end", triggers=len(veto_info))
            return batched_res

    chisq_values = []
    dof_values = []
    sg_values = []
    veto_corr = {}
    for info in veto_info:
        snrv, norm, l, htilde, stilde = info[:5]
        if _reference_enabled("live_selection"):
            from pycbc import waveform

            # The original generic sigma_cached returns a Python float;
            # precomputed waveform norms return a NumPy scalar. Preserve
            # that promotion boundary for both signal-consistency vetoes.
            host_norm = np.asarray(jax.device_get(norm))
            norm = (host_norm[()] if waveform.waveform_norm_exists(
                getattr(htilde, "approximant", "")) else host_norm.item())
        size = _live_array_metadata(htilde.cout)[0]
        if size not in veto_corr:
            veto_corr[size] = live_veto_buffer_jax(size, htilde.dtype)
        corr = veto_corr[size]
        correlate(htilde, stilde, corr)
        c, d = power_chisq.values(corr, snrv, norm,
                                  stilde.psd, [l], htilde)
        c = jnp.asarray(c)
        d = jnp.asarray(d)
        chisq_values.append(
            jnp.asarray(_divide(c[0], d[0]), dtype=jnp.float32))
        dof_values.append(jnp.asarray(d[0], dtype=jnp.uint32))
        sgv = control.sg_chisq.values(stilde, htilde, stilde.psd,
                                      snrv, norm, c, d, [l])
        if sgv is not None:
            sg_values.append(jnp.asarray(sgv[0], dtype=jnp.float32))
        else:
            sg_values.append(jnp.asarray(0, dtype=jnp.float32))
    if veto_info:
        chisq = jnp.stack(chisq_values)
        dof = jnp.stack(dof_values)
        sg_chisq = jnp.stack(sg_values)
    else:
        chisq = _EMPTY_F32
        dof = _EMPTY_U32
        sg_chisq = _EMPTY_F32
    results["chisq"] = chisq
    results["chisq_dof"] = dof
    results["sg_chisq"] = sg_chisq
    if control.newsnr_threshold:
        keep_mask = ranking.newsnr(results["snr"], chisq) >= (
            control.newsnr_threshold
        )
        keep = jnp.asarray(
            np.flatnonzero(jax.device_get(keep_mask)), dtype=jnp.uint32
        )
        for key in results:
            results[key] = results[key][keep]
    stage_event("filter_veto", "end", triggers=len(veto_info))
    return results


@functools.partial(jax.jit, static_argnames=("groups",))
def _live_concat_resident_columns(*values, groups):
    """Join canonicalized groups, retaining JAX's per-column promotion."""
    sizes = ((groups,) * len(range(0, len(values), groups))
             if isinstance(groups, (int, np.integer)) else groups)
    columns, start = [], 0
    for size in sizes:
        columns.append(jnp.concatenate(values[start:start + size]))
        start += size
    return tuple(columns)


def _live_prune_empty_column(values):
    """Remove empty-position specialization without changing promotion.

    Empty arrays still contribute their dtype to JAX concatenation. Retain one
    representative per dtype absent from the nonempty inputs, in a stable
    order, while preserving the original order of every nonempty input.
    """
    nonempty = [value for value in values if value.shape[0]]
    represented = {np.dtype(value.dtype).str for value in nonempty}
    empty = {}
    for value in values:
        dtype = np.dtype(value.dtype).str
        if not value.shape[0] and dtype not in represented:
            empty.setdefault(dtype, value)
    return nonempty + [empty[dtype] for dtype in sorted(empty)]


@functools.lru_cache(maxsize=128)
def _live_concat_resident_executable(shapes, groups, device, x64):
    """Retain handles by geometry, device and precision, without result data."""
    from pycbc.vetoes.chisq_jax import _live_chisq_executable

    abstracts = tuple(jax.ShapeDtypeStruct(shape, np.dtype(dtype))
                      for shape, dtype in shapes)
    return _live_chisq_executable(_live_concat_resident_columns, abstracts,
                                 dict(groups=groups), {}, device)


@functools.lru_cache(maxsize=32)
def _live_host_empty_columns(device, dtypes, x64):
    """Retain uncommitted empty host uploads, never candidate result data."""
    with jax.ensure_compile_time_eval(), jax.default_device(device):
        return tuple(jnp.asarray(np.empty(0, dtype=np.dtype(dtype)))
                     for dtype in dtypes)


def _live_host_column_parts(values, dtypes):
    """Upload matching nonempty vectors once, keeping empty JAX promotion."""
    if any(value.ndim != 1 for value in values):
        return None
    nonempty = [(value, dtype) for value, dtype in zip(values, dtypes)
                if value.shape[0]]
    if not nonempty:
        return None
    dtype = nonempty[0][1]
    if any(other != dtype for _, other in nonempty):
        return None
    uploaded = jnp.asarray(np.concatenate(
        [np.asarray(value, dtype=dtype) for value, _ in nonempty]))
    if (not isinstance(uploaded, jax.Array)
            or isinstance(uploaded, jax.core.Tracer)
            or len(uploaded.devices()) != 1):
        return None
    device = next(iter(uploaded.devices()))
    layout = getattr(getattr(uploaded, "format", None), "layout", None)
    if (device.platform != "gpu"
            or uploaded.sharding != jax.sharding.SingleDeviceSharding(device)
            or layout is None or layout.major_to_minor != (0,)
            or layout.tiling or layout.sub_byte_element_size_in_bits != 0):
        return None
    # Do not widen on the host: XLA's casts flush some subnormals and round
    # wide signed/unsigned integers differently from NumPy promotion.
    empty_dtypes = tuple(sorted({np.dtype(other).str
                                for value, other in zip(values, dtypes)
                                if not value.shape[0] and other != dtype}))
    return [uploaded, *_live_host_empty_columns(
        device, empty_dtypes, bool(jax.config.jax_enable_x64))]


def combine_live_results_jax(results):
    """Upload uniform host columns once and join CUDA columns together."""
    from pycbc import scheme

    target = getattr(scheme.mgr.state, "jax_device", None)
    gpu = (getattr(target, "platform", None) in ("cuda", "gpu")
           and jax.config.jax_default_device == target)
    combined = {}
    resident = {}
    device = None
    for key in results[0]:
        values = [batch[key] for batch in results]
        kinds = []
        for value in values:
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                dtype = np.asarray(value).dtype
            kinds.append(np.dtype(dtype).kind)
        if all(kind in "biufc" for kind in kinds):
            arrays = None
            if all(type(value) is np.ndarray and value.dtype.isnative
                   and value.ndim > 0
                   and value.shape[1:] == values[0].shape[1:]
                   for value in values):
                dtypes = [jax.dtypes.canonicalize_dtype(value.dtype)
                          for value in values]
                if all(dtype == dtypes[0] for dtype in dtypes):
                    # Canonicalize each group before combining, as the old
                    # per-group uploads did. Keep heterogeneous promotion in
                    # JAX: NumPy widening can preserve subnormals XLA flushes.
                    arrays = [np.asarray(value, dtype=dtypes[0])
                              for value in values]
                    combined[key] = jnp.asarray(np.concatenate(arrays))
                    continue
                if gpu and jax.config.jax_numpy_dtype_promotion == "standard":
                    arrays = _live_host_column_parts(values, dtypes)
            if arrays is None:
                arrays = [value if gpu and isinstance(value, jax.Array)
                          and not isinstance(value, jax.core.Tracer)
                          and not value.weak_type
                          and value.devices() == {target}
                          and value.dtype == jax.dtypes.canonicalize_dtype(
                              value.dtype) else jnp.asarray(value)
                          for value in values]
            # Keep conversions outside fusion: host canonicalization and JAX
            # promotion differ from NumPy for wide integers and subnormals.
            eligible = jax.config.jax_numpy_dtype_promotion == "standard" and all(
                (type(value) is np.ndarray and value.dtype.isnative)
                or (isinstance(value, jax.Array)
                    and not isinstance(value, jax.core.Tracer)
                    and not value.weak_type
                    and value.dtype == jax.dtypes.canonicalize_dtype(
                        value.dtype)) for value in values)
            eligible = eligible and all(
                isinstance(array, jax.Array)
                and not isinstance(array, jax.core.Tracer)
                and not array.weak_type and array.ndim == 1
                and len(array.devices()) == 1 for array in arrays)
            if eligible:
                column_device = next(iter(arrays[0].devices()))
                sharding = jax.sharding.SingleDeviceSharding(column_device)
                eligible = column_device.platform == "gpu" and all(
                    array.sharding == sharding for array in arrays)
                if eligible:
                    # A retained executable requires its default vector
                    # layout and device memory, not merely the same device.
                    layouts = [getattr(getattr(array, "format", None),
                                       "layout", None) for array in arrays]
                    eligible = all(
                        layout is not None and layout.major_to_minor == (0,)
                        and not layout.tiling
                        and layout.sub_byte_element_size_in_bits == 0
                        for layout in layouts)
                if eligible and device is None:
                    device = column_device
                eligible = eligible and column_device == device
            if eligible:
                combined[key] = None
                resident[key] = arrays
            else:
                combined[key] = jnp.concatenate(arrays)
        else:
            combined[key] = np.concatenate(values)
    if len(resident) > 1:
        columns = [_live_prune_empty_column(column)
                   for column in resident.values()]
        values = tuple(value for column in columns for value in column)
        groups = tuple(len(column) for column in columns)
        shapes = tuple((tuple(value.shape), np.dtype(value.dtype).str)
                       for value in values)
        # Closed-over concrete arrays may also be assembled under an outer
        # trace. Evaluate their retained executable without caching tracers.
        with jax.ensure_compile_time_eval():
            executable = _live_concat_resident_executable(
                shapes, groups, device, bool(jax.config.jax_enable_x64))
            outputs = executable(*values)
        for key, output in zip(resident, outputs):
            combined[key] = output
    else:
        for key, values in resident.items():
            combined[key] = jnp.concatenate(values)
    return combined


@jax.jit
def _live_select_peaks(peaks, norms, threshold, abort_threshold):
    """Scale and select a live batch without per-template synchronization."""
    scaled = (peaks * norms).astype(peaks.dtype)
    magnitudes = jnp.abs(peaks) * norms
    # Preserve the scalar path's NaN behavior: ``not (s < threshold)``.
    accepted = jnp.logical_not(magnitudes < threshold)
    abort = accepted & (magnitudes > abort_threshold)
    return scaled, accepted, abort


@jax.jit
def _live_candidate_columns(scaled_peaks, sigmasq, selected):
    """Assemble numeric columns and veto vectors in one device dispatch."""
    scaled = scaled_peaks[selected]
    return (jnp.abs(scaled), jnp.angle(scaled),
            sigmasq[selected].astype(jnp.float32))


@functools.lru_cache(maxsize=32)
def _live_empty_candidate_columns(device, dtypes, x64):
    """Retain immutable empty outputs by placement and dtype mode."""
    return tuple(jax.device_put(np.empty(0, dtype=np.dtype(dtype)), device)
                 for dtype in dtypes)


def _live_empty_candidate_device(arrays):
    """Qualify ordinary Live vectors without reading their scientific data."""
    if not all(isinstance(array, jax.Array)
               and not isinstance(array, jax.core.Tracer)
               and not array.weak_type and array.ndim == 1
               and array.dtype == jax.dtypes.canonicalize_dtype(array.dtype)
               for array in arrays):
        return None
    if arrays[0].shape[0] == 0 or any(
            array.shape != arrays[0].shape for array in arrays[1:]):
        return None
    if any(array.dtype not in (np.complex64, np.complex128)
           for array in arrays[:2]) or any(
            array.dtype not in (np.float32, np.float64)
            for array in arrays[2:]):
        return None
    if any(len(array.devices()) != 1 for array in arrays):
        return None
    device = next(iter(arrays[1].devices()))
    sharding = jax.sharding.SingleDeviceSharding(device)
    if any(array.sharding != sharding for array in arrays):
        return None
    layouts = [getattr(getattr(array, "format", None), "layout", None)
               for array in arrays]
    if not all(layout is not None and layout.major_to_minor == (0,)
               and not layout.tiling
               and layout.sub_byte_element_size_in_bits == 0
               for layout in layouts):
        return None
    return device


def _live_candidate_columns_bucketed(
        control, scaled_peaks, peaks, norms, sigmasq, selected):
    """Retain selection kernels across sparse CUDA counts in one bucket."""
    count = len(selected)
    if count == 0:
        device = _live_empty_candidate_device(
            (scaled_peaks, peaks, norms, sigmasq))
        if device is not None:
            real = np.dtype(np.float32 if scaled_peaks.dtype == np.complex64
                            else np.float64).str
            dtypes = (real, real, np.dtype(np.float32).str)
            # A concrete closed-over vector can enter under an outer trace.
            # Never retain a tracer in the empty-array cache.
            with jax.ensure_compile_time_eval():
                return _live_empty_candidate_columns(
                    device, dtypes, bool(jax.config.jax_enable_x64))

    from pycbc.vetoes.chisq_jax import (
        _chisq_candidate_bucket, _live_chisq_executable,
    )

    indices = np.zeros(_chisq_candidate_bucket(count), dtype=np.int32)
    indices[:count] = selected
    cache = _live_veto_executable_cache(control)
    args = (scaled_peaks, sigmasq, indices)
    execute = _live_chisq_executable(
        _live_candidate_columns, args, {}, cache, peaks.device)
    columns = execute(*args)
    return _live_compact_results(columns, count, cache, peaks.device)


_LivePreparedBatch = namedtuple("_LivePreparedBatch", (
    "group_index", "templates", "stilde", "template_matrix", "native_norms",
    "norms", "valid_start", "peak_values", "scaled_peaks", "selection",
    "native_selection", "correlations", "n_time"),
    defaults=(None, None, None))


_LiveBatchInputs = namedtuple("_LiveBatchInputs", (
    "group_index", "templates", "stilde", "template_matrix", "native_norms",
    "norms", "valid_start", "segment", "size", "fused"))


def _live_prepare_batch_inputs_jax(self, group_index=None):
    """Capture one group's immutable inputs without launch or publication.

    Explicit indices let detector batching gather inputs from a shared control
    without advancing its group cursor or replacing public filter buffers.
    """
    from pycbc.benchmark import stage_event
    if group_index is None:
        group_index = self.block_id
    if group_index == len(self.tgroups):
        return None

    tgroup = self.tgroups[group_index]
    psize = self.chunk_tsamples[group_index]
    mid = self.mids[group_index]
    stage_event("filter_overwhiten", "start", templates=len(tgroup))
    stilde = self.data.overwhitened_data(tgroup[0].delta_f)
    stage_event("filter_overwhiten", "end", templates=len(tgroup))
    psd = stilde.psd
    template_matrix = getattr(
        self.corr[group_index], "_jax_template_matrix", None
    )
    stage_event("filter_norms", "start", templates=len(tgroup))
    native_norms, norms = _live_cached_template_norms_jax(
        self.corr[group_index], tgroup, psd)
    stage_event("filter_norms", "end", templates=len(tgroup))

    valid_end = int(psize - self.data.trim_padding)
    valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)

    seg = slice(valid_start, valid_end)
    abort_threshold = (jnp.inf if self.snr_abort_threshold is None
                       else float(self.snr_abort_threshold))
    fused = _live_fused_batch_inputs_jax(
        self.corr[group_index], self.ifts[mid], stilde, norms, seg,
        float(self.snr_threshold), abort_threshold,
    )
    return _LiveBatchInputs(
        group_index, tgroup, stilde, template_matrix, native_norms, norms,
        valid_start, seg, psize, fused,
    )


def _live_native_selection_jax(control, templates, peaks, sigmasqs):
    """Run the original Live selection with each template's scalar contract."""
    from pycbc import waveform
    from pycbc.reference_jax import cpu_reference

    return cpu_reference(
        "live_selection", peaks, spacing=templates[0].delta_f,
        sigmasqs=np.asarray(sigmasqs),
        python_sigmasq=np.asarray([
            not waveform.waveform_norm_exists(
                getattr(template, "approximant", ""))
            for template in templates]),
        snr_threshold=control.snr_threshold,
        snr_abort_threshold=control.snr_abort_threshold)


def _live_complete_batch_jax(self, inputs, outputs=None):
    """Launch or publish captured inputs, preserving public full buffers."""
    from pycbc.benchmark import stage_event

    group_index, tgroup = inputs.group_index, inputs.templates
    mid = self.mids[group_index]
    native_selection = None
    if inputs.fused is not None:
        stage_event("filter_fused", "start", templates=len(tgroup))
        if outputs is None:
            outputs = _live_launch_fused_batch_jax(inputs.fused)
        peak_indices, peak_values, scaled_peaks, accepted, abort = (
            _live_publish_fused_batch_jax(inputs.fused, outputs))
        stage_event("filter_fused", "end", templates=len(tgroup))
    else:
        if outputs is not None:
            raise ValueError("Unqualified Live batch cannot publish fused output")
        stage_event("filter_correlation", "start", templates=len(tgroup))
        self.corr[group_index].execute(inputs.stilde)
        stage_event("filter_correlation", "end", templates=len(tgroup))
        # JAX CPU and unqualified geometries retain their original transform.
        stage_event("filter_ifft", "start", templates=len(tgroup))
        _live_ifft_execute_jax(self.ifts[mid])
        stage_event("filter_ifft", "end", templates=len(tgroup))

        stage_event("filter_peak_select", "start", templates=len(tgroup))
        jax_peaks = batch_peak_values(
            self.out_mem[mid], len(tgroup), inputs.size, inputs.segment)
        if jax_peaks is None:
            raise ValueError("JAX live peak reduction could not process the batch")
        peak_indices, peak_values = jax_peaks
        abort_threshold = (jnp.inf if self.snr_abort_threshold is None
                           else float(self.snr_abort_threshold))
        if _reference_enabled("live_selection"):
            native_selection = _live_native_selection_jax(
                self, tgroup, peak_values, inputs.native_norms)
            native_result, selected, _ = native_selection
            accepted = jnp.zeros(peak_values.shape, dtype=bool)
            accepted = accepted.at[selected].set(True)
            abort = jnp.full(peak_values.shape, native_result is False,
                             dtype=bool)
            scaled_peaks = peak_values
        else:
            scaled_peaks, accepted, abort = _live_select_peaks(
                peak_values, inputs.norms, float(self.snr_threshold),
                abort_threshold)
        stage_event("filter_peak_select", "end", templates=len(tgroup))

    # Peak vectors remain independent of subsequent shared-workspace writes.
    return _LivePreparedBatch(
        group_index, tgroup, inputs.stilde, inputs.template_matrix,
        inputs.native_norms, inputs.norms, inputs.valid_start,
        peak_values, scaled_peaks,
        (peak_indices, accepted, abort), native_selection,
        (to_jax(self.cout_mem[mid])
         if peak_values.device.platform in ("cuda", "gpu") else None),
        inputs.size)


def _live_prepare_batch_jax(self):
    """Enqueue one group's kernels without retrieving peak decisions."""
    inputs = _live_prepare_batch_inputs_jax(self)
    if inputs is None:
        return None
    prepared = _live_complete_batch_jax(self, inputs)
    self.block_id += 1
    return prepared


def _live_finish_batch_jax(self, prepared, host_selection):
    """Assemble one group's sparse metadata using its collected decisions."""
    from pycbc.filter.matchedfilter import logger

    tgroup = prepared.templates
    stilde = prepared.stilde
    template_matrix = prepared.template_matrix
    native_norms, norms = prepared.native_norms, prepared.norms
    peak_values, scaled_peaks = prepared.peak_values, prepared.scaled_peaks
    valid_start = prepared.valid_start
    host_peak_indices, host_accepted, host_abort = host_selection

    tkeys = tgroup[0].params.dtype.names
    result = {key: [] for key in tkeys}
    veto_info = []
    if _reference_enabled("live_selection"):
        selection = prepared.native_selection
        if selection is None:
            selection = _live_native_selection_jax(self, tgroup, peak_values,
                                                   native_norms)
        native_result, accepted_indices, accepted_norms = selection
        if native_result is False:
            return False, []
        norms = norms.at[accepted_indices].set(to_jax(accepted_norms))
        snr, phase, sigmasq = (to_jax(native_result[key]) for key in (
            "snr", "coa_phase", "sigmasq"))
    else:
        if np.any(host_abort):
            logger.info("We are seeing some *really* high SNRs, let's "
                        "assume they aren't signals and just give up")
            return False, []
        accepted_indices = np.flatnonzero(host_accepted)
        if getattr(peak_values.device, "platform", None) in ("cuda", "gpu"):
            columns = _live_candidate_columns_bucketed(
                self, scaled_peaks, peak_values, norms, native_norms,
                accepted_indices)
        else:
            selected = jnp.asarray(accepted_indices, dtype=jnp.int32)
            columns = _live_candidate_columns(
                scaled_peaks, native_norms, selected)
        snr, phase, sigmasq = columns

    result.update({
        "snr": snr,
        "coa_phase": phase,
        "end_time": np.asarray(
            float(self.data.start_time)
            + host_peak_indices[accepted_indices] / self.data.sample_rate,
            dtype=np.float64,
        ),
        "template_id": np.asarray(
            [tgroup[idx].id for idx in accepted_indices],
            dtype=jax.dtypes.canonicalize_dtype(np.uint64),
        ),
        "sigmasq": sigmasq,
    })

    for idx in accepted_indices:
        htilde = tgroup[idx]
        if hasattr(htilde, 'time_offset'):
            if 'time_offset' not in result:
                result['time_offset'] = []

        l = int(host_peak_indices[idx]) + valid_start
        veto_info.append(_LiveVetoCandidate(
            peak_values, norms, int(idx), l, htilde,
            stilde, template_matrix, int(idx)))
        if not hasattr(htilde, 'dict_params'):
            htilde.dict_params = {}
            for key in tkeys:
                htilde.dict_params[key] = htilde.params[key]

        for key in tkeys:
            result[key].append(htilde.dict_params[key])

        if hasattr(htilde, 'time_offset'):
            result['time_offset'].append(htilde.time_offset)

    for key in tkeys:
        result[key] = np.array(result[key])

    if 'time_offset' in result:
        result['time_offset'] = jnp.asarray(result['time_offset'])

    return result, veto_info


@functools.partial(jax.jit, static_argnames=("groups", "metadata_dtypes"))
def _live_candidate_bank_columns(*values, groups, metadata_dtypes):
    """Gather a whole bank's candidate columns in one retained dispatch.

    Calculate each group's magnitude, phase and sigma cast before joining;
    promoting mixed-precision complex inputs first changes their rounding.
    """
    selected, end_times = values[-2:]
    scaled, sigmas = values[:groups], values[groups:2 * groups]
    metadata = values[2 * groups:-2]
    return (
        jnp.concatenate([jnp.abs(value) for value in scaled])[selected],
        jnp.concatenate([jnp.angle(value) for value in scaled])[selected],
        jnp.concatenate([value.astype(jnp.float32)
                         for value in sigmas])[selected],
        end_times,
        *(value[selected].astype(dtype)
          for value, dtype in zip(metadata, metadata_dtypes)),
    )


def _live_candidate_metadata(control, prepared, selected):
    """Cache bank columns, checking only admitted rows for mutable metadata.

    The legacy lazily-created dict_params snapshot remains authoritative.
    Unselected rows are checked when they are first admitted, so changing
    metadata cannot publish a stale value and does not require scanning the
    entire bank on every block. Caches live only as long as their control.
    """
    device = prepared[0].peak_values.device
    signature = (tuple((id(batch.templates), len(batch.templates))
                       for batch in prepared), device,
                 bool(jax.config.jax_enable_x64))
    cache = getattr(control, "_jax_live_candidate_metadata", None)
    keys = prepared[0].templates[0].params.dtype.names
    if not keys or set(keys) & {
            "snr", "coa_phase", "end_time", "template_id", "sigmasq"}:
        return None

    admitted = []
    offset = 0
    for batch, rows in zip(prepared, selected):
        for row in rows:
            template = batch.templates[row]
            if (template.params.dtype.names != keys
                    or hasattr(template, "time_offset")):
                return None
            if not hasattr(template, "dict_params"):
                template.dict_params = {key: template.params[key]
                                        for key in keys}
            admitted.append((offset + int(row), template))
        offset += len(batch.templates)

    if (cache is not None and cache["signature"] == signature
            and cache["keys"] == keys):
        for row, template in admitted:
            if (template is not cache["templates"][row]
                    or template.id != cache["ids"][row]):
                cache = None
                break
            for key, host in cache["numeric_host"].items():
                value = template.dict_params[key]
                # Exact NumPy numeric scalars are immutable. Reusing the same
                # scalar therefore proves its dtype and payload are unchanged;
                # replacements still take the bitwise validation below.
                scalars = cache["numeric_scalars"][key]
                if scalars[row] is not None and value is scalars[row]:
                    continue
                if (not isinstance(value, np.generic)
                        or value.dtype != host.dtype
                        or value.tobytes() != host[row].tobytes()):
                    cache = None
                    break
                scalars[row] = (value if type(value) is host.dtype.type
                                else None)
            if cache is None:
                break
        if cache is not None:
            return cache

    templates = tuple(template for batch in prepared
                      for template in batch.templates)
    if any(template.params.dtype.names != keys
           or hasattr(template, "time_offset")
           or not hasattr(template, "id")
           or (hasattr(template, "dict_params")
               and any(key not in template.dict_params for key in keys))
           for template in templates):
        return None
    ids = [template.id for template in templates]
    id_dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.uint64))
    # Uploading unused IDs must not introduce an overflow the serial API
    # would encounter only if that template were selected.
    if any(not isinstance(value, (int, np.integer))
           or not 0 <= value <= np.iinfo(id_dtype).max for value in ids):
        return None
    numeric_host, numeric_device, numeric_scalars = {}, {}, {}
    for key in keys:
        values = [(template.dict_params[key]
                   if hasattr(template, "dict_params") else template.params[key])
                  for template in templates]
        first = values[0]
        if (templates[0].params.dtype[key].kind in "biufc"
                and not (isinstance(first, np.generic)
                         and all(isinstance(value, np.generic)
                                 and value.dtype == first.dtype
                                 for value in values))):
            return None
        if (isinstance(first, np.generic) and first.dtype.kind in "biufc"
                and all(isinstance(value, np.generic)
                        and value.dtype == first.dtype for value in values)):
            numeric_host[key] = np.array(values)
            numeric_device[key] = jnp.asarray(numeric_host[key])
            numeric_scalars[key] = [
                value if type(value) is numeric_host[key].dtype.type else None
                for value in values]
    cache = dict(signature=signature, templates=templates, ids=ids, keys=keys,
                 numeric_host=numeric_host, numeric_device=numeric_device,
                 numeric_scalars=numeric_scalars,
                 device_ids=jnp.asarray(np.asarray(ids, dtype=id_dtype)))
    control._jax_live_candidate_metadata = cache
    return cache


def _live_finish_batches_jax(control, prepared, selections):
    """Prepare and combine CUDA candidates across all duration groups.

    Return None for the existing assembly path where bank metadata cannot
    safely be cached. Numeric bank columns stay resident; timestamps retain
    the legacy NumPy float64 calculation and use one compact upload. Strings
    and Python veto references retain their original selected-row semantics.
    """
    from pycbc.vetoes.chisq_jax import (
        _chisq_candidate_bucket, _live_chisq_executable,
    )

    if (_reference_enabled("live_selection")
            or not prepared or prepared[0].peak_values.device.platform != "gpu"
            or jax.config.jax_numpy_dtype_promotion != "standard"
            or any(np.any(selection[2]) for selection in selections)):
        return None
    selected = [np.flatnonzero(selection[1]) for selection in selections]
    count = sum(len(rows) for rows in selected)
    if not count:
        return None
    metadata = _live_candidate_metadata(control, prepared, selected)
    if metadata is None:
        return None

    indices = np.zeros(_chisq_candidate_bucket(count), dtype=np.int32)
    times = np.zeros(len(indices), dtype=np.float64)
    veto_info, host_results = [], []
    offset, start = 0, 0
    host_keys = [key for key in metadata["keys"]
                 if key not in metadata["numeric_device"]]
    for batch, selection, rows in zip(prepared, selections, selected):
        stop = start + len(rows)
        indices[start:stop] = rows + offset
        times[start:stop] = (float(control.data.start_time)
                            + selection[0][rows] / control.data.sample_rate)
        host = {key: [] for key in host_keys}
        for row in rows:
            template = batch.templates[row]
            for key in host_keys:
                host[key].append(template.dict_params[key])
            veto_info.append(_LiveVetoCandidate(
                batch.peak_values, batch.norms, int(row),
                int(selection[0][row]) + batch.valid_start, template,
                batch.stilde, batch.template_matrix, int(row)))
        host_results.append({key: np.array(value)
                             for key, value in host.items()})
        start, offset = stop, offset + len(batch.templates)

    # Empty parameter groups contribute float64 to legacy concatenation,
    # including otherwise integral columns. Keep that promotion on device.
    empty = any(not len(rows) for rows in selected)
    columns = (metadata["device_ids"],
               *metadata["numeric_device"].values())
    dtypes = (np.dtype(columns[0].dtype).str, *(
        np.dtype(jnp.result_type(value.dtype, np.float64)
                 if empty else value.dtype).str for value in columns[1:]))
    args = (*(batch.scaled_peaks for batch in prepared),
            *(batch.native_norms for batch in prepared), *columns, indices,
            times.astype(jax.dtypes.canonicalize_dtype(np.float64)))
    cache = _live_veto_executable_cache(control)
    execute = _live_chisq_executable(
        _live_candidate_bank_columns, args,
        dict(groups=len(prepared), metadata_dtypes=dtypes), cache,
        prepared[0].peak_values.device)
    outputs = _live_compact_results(execute(*args), count, cache,
                                    prepared[0].peak_values.device)
    snr, phase, sigmasq, end_time, template_id, *numeric = outputs
    host = combine_live_results_jax(host_results) if host_keys else {}
    result = {key: None for key in metadata["keys"]}
    result.update(zip(metadata["numeric_device"], numeric))
    result.update(host)
    result.update(snr=snr, coa_phase=phase, end_time=end_time,
                  template_id=template_id, sigmasq=sigmasq)
    return result, veto_info


def live_process_batch_jax(self):
    """Process one batch, retaining the serial API for JAX CPU and callers."""
    from pycbc.benchmark import stage_event

    prepared = _live_prepare_batch_jax(self)
    if prepared is None:
        return None, None
    stage_event("filter_peak_collect", "start", groups=1)
    selection = jax.device_get(prepared.selection)
    stage_event("filter_peak_collect", "end", groups=1,
                accepted=int(np.count_nonzero(selection[1])))
    return _live_finish_batch_jax(self, prepared, selection)


def process_live_data_jax(self, data_reader):
    """Queue CUDA groups before collecting their peak-decision vectors."""
    from pycbc import scheme
    from pycbc.benchmark import stage_event

    self.set_data(data_reader)
    device = getattr(scheme.mgr.state, "jax_device", None)
    if getattr(device, "platform", None) not in ("cuda", "gpu"):
        return self.process_all()

    pending = []
    while self.block_id < len(self.tgroups):
        pending.append(_live_prepare_batch_jax(self))
    if not pending:
        # Preserve the existing empty-bank API, including its assembly error.
        return self.process_all()

    stage_event("filter_peak_collect", "start", groups=len(pending))
    selections = jax.device_get(tuple(batch.selection for batch in pending))
    stage_event("filter_peak_collect", "end", groups=len(pending),
                accepted=sum(np.count_nonzero(values[1])
                             for values in selections))
    results, veto_info = [], []
    stage_event("filter_candidate_prepare", "start", groups=len(pending))
    assembled = _live_finish_batches_jax(self, pending, selections)
    if assembled is None:
        for batch, selection in zip(pending, selections):
            result, veto = _live_finish_batch_jax(self, batch, selection)
            if result is False:
                self.block_id = batch.group_index + 1
                stage_event("filter_candidate_prepare", "end", aborted=True)
                return False
            results.append(result)
            veto_info += veto
    else:
        result, veto_info = assembled

    stage_event("filter_candidate_prepare", "end", triggers=len(veto_info))
    stage_event("filter_result_combine", "start", groups=len(results))
    if assembled is None:
        result = self.combine_results(results)
    if self.max_triggers_in_batch:
        sort = result['snr'].argsort()[::-1][:self.max_triggers_in_batch]
        for key in result:
            result[key] = result[key][sort]
        veto_info = [veto_info[i] for i in sort]
    stage_event("filter_result_combine", "end", triggers=len(veto_info))
    return self._process_vetoes(result, veto_info)


class _LiveEnqueuedData:
    """Own a detector's immutable inputs while a shared control advances.

    In particular, retain the correlation root itself, not its mutable PyCBC
    workspace wrapper. Another detector or block may replace that wrapper
    before this detector's vetoes or terminal snapshot run.
    """

    _jax_live_prepared = True

    def __init__(self, owner, prepared, reader, metadata):
        self.owner = owner
        self.prepared = tuple(prepared)
        self.psds = tuple(batch.stilde.psd for batch in prepared)
        self.start_time = float(reader.start_time)
        self.sample_rate = float(reader.sample_rate)
        self.metadata = metadata
        self.reader = reader


def _live_immutable_metadata_equal(value, previous):
    """Compare owned immutable metadata, preserving dtype and payload bits."""
    if value is previous:
        return True
    if type(value) is not type(previous):
        return False
    if isinstance(value, np.generic):
        return (value.dtype == previous.dtype
                and value.tobytes() == previous.tobytes())
    if type(value) in (float, complex):
        # Object fields can contain Python numeric scalars. Equality alone
        # loses signed zero and does not compare NaN payloads.
        return np.asarray(value).tobytes() == np.asarray(previous).tobytes()
    return value == previous


def _live_resident_metadata(control, prepared):
    """Freeze mutable bank configuration without reading candidate decisions.

    Structured records need one byte comparison per template. Previously
    admitted ``dict_params`` snapshots remain authoritative, including dtype,
    NaN payloads and signed zero. Unsupported mutable Python values retain the
    compatibility assembler. No dictionaries are eagerly created for rejected
    templates, and retained column tables contain only immutable values.
    Native tables stay on the host; legacy JAX promotion casts are collected
    together once when admitting a metadata revision, outside steady reuse.
    """
    if not prepared or not prepared[0].templates:
        return None
    templates = tuple(template for batch in prepared
                      for template in batch.templates)
    keys = templates[0].params.dtype.names
    if not keys or set(keys) & {
            "snr", "coa_phase", "end_time", "template_id", "sigmasq",
            "chisq", "chisq_dof", "sg_chisq"}:
        return None
    device = prepared[0].peak_values.device
    sizes = tuple(len(batch.templates) for batch in prepared)
    params_dtypes = tuple(template.params.dtype for template in templates)
    cached = getattr(control, "_jax_live_resident_metadata", None)
    if (cached is not None and cached["signature"][0] == device
            and cached["keys"] == keys and cached["sizes"] == sizes
            and cached["params_dtypes"] == params_dtypes
            and len(cached["templates"]) == len(templates)):
        # Authoritative dictionary values admitted by the slow validator are
        # immutable scalars. Identity proves their dtype and payload unchanged
        # without allocating a bytes/dtype signature for every field/block.
        # Replacements, raw records and all geometry/schema changes retain
        # exact validation below. These references are validation state only;
        # the result's owned metadata snapshot is never changed by reuse.
        unchanged = True
        published = []
        for row, template in enumerate(templates):
            previous = cached["signature"][1][row]
            params = getattr(template, "dict_params", None)
            if (id(template) != previous[0]
                    or id(template.params) != previous[1]
                    or not isinstance(template.id, (int, np.integer))
                    or int(template.id) != previous[3]
                    or template.params.dtype.names != keys
                    or hasattr(template, "time_offset")):
                unchanged = False
                break
            if id(params) != previous[2]:
                # The terminal worker lazily creates dict_params from the
                # owned snapshot. Publishing an equal authoritative view
                # changes cache bookkeeping, not the owned bank columns.
                # Replacements of existing dictionaries retain slow validation.
                if params is None or cached["validated"][row] is not None:
                    unchanged = False
                    break
                values = []
                payload = []
                for key in keys:
                    if key not in params:
                        unchanged = False
                        break
                    value = params[key]
                    if not _live_immutable_metadata_equal(
                            value, cached["host"][key][row]):
                        unchanged = False
                        break
                    values.append(value)
                    payload.append((value.dtype.str, value.tobytes())
                                   if isinstance(value, np.generic)
                                   else (type(value), value))
                if not unchanged:
                    break
                published.append((row, (
                    previous[0], previous[1], id(params), previous[3],
                    tuple(payload)), tuple(values)))
            elif params is None:
                if template.params.tobytes() != previous[4]:
                    unchanged = False
                    break
            elif any(key not in params or params[key] is not value
                     for key, value in zip(keys, cached["validated"][row])):
                unchanged = False
                break
        if unchanged:
            if published:
                signatures = list(cached["signature"][1])
                validated = list(cached["validated"])
                for row, signature, values in published:
                    signatures[row] = signature
                    validated[row] = values
                cached["signature"] = (device, tuple(signatures))
                cached["validated"] = tuple(validated)
            return cached
    signature = []
    validated = []
    for template in templates:
        if (template.params.dtype.names != keys
                or hasattr(template, "time_offset")
                or not isinstance(getattr(template, "id", None),
                                  (int, np.integer))
                or not 0 <= template.id <= np.iinfo(np.uint64).max):
            return None
        params = getattr(template, "dict_params", None)
        if params is None:
            payload = template.params.tobytes()
            validated.append(None)
        else:
            payload = []
            values = []
            for key in keys:
                if key not in params:
                    return None
                value = params[key]
                values.append(value)
                if (isinstance(value, np.generic)
                        and type(value) is value.dtype.type
                        and value.dtype.kind in "biufcUS"):
                    payload.append((value.dtype.str, value.tobytes()))
                elif type(value) in (str, bytes, int, float, complex, bool,
                                     type(None)):
                    # Numeric structured fields must retain their NumPy
                    # scalar contract; changing one to a Python scalar uses
                    # the compatibility path rather than guessing promotion.
                    if template.params.dtype[key].kind in "biufc":
                        return None
                    payload.append((type(value), value))
                else:
                    return None
            payload = tuple(payload)
            validated.append(tuple(values))
        signature.append((id(template), id(template.params), id(params),
                          int(template.id), payload))
    signature = (device, tuple(signature))
    if (cached is not None and cached["signature"] == signature
            and cached["keys"] == keys and cached["sizes"] == sizes
            and cached["params_dtypes"] == params_dtypes):
        # Equal-bit scalar replacements may reuse owned columns. Remember
        # their new immutable identities so the next block is fast again.
        cached["validated"] = tuple(validated)
        return cached

    host = {}
    numeric = {}
    promotions = []
    for key in keys:
        values = tuple((template.dict_params[key]
                        if hasattr(template, "dict_params")
                        else template.params[key]) for template in templates)
        if any(not (isinstance(value, np.generic)
                    and type(value) is value.dtype.type
                    and value.dtype.kind in "biufcUS")
               and type(value) not in (str, bytes, int, float, complex, bool,
                                      type(None)) for value in values):
            return None
        host[key] = values
        if (isinstance(values[0], np.generic)
                and values[0].dtype.kind in "biufc"
                and all(type(value) is value.dtype.type
                        and value.dtype == values[0].dtype for value in values)):
            if values[0].dtype not in tuple(map(np.dtype, (
                    np.bool_, np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64,
                    np.complex64, np.complex128))):
                return None
            # Bank parameters are immutable host configuration. Preserve
            # owned native/promoted columns for terminal publication without
            # uploading them only to copy them back after every block.
            native = np.array(values)
            native.setflags(write=False)
            promoted_dtype = np.result_type(native.dtype, np.float64)
            numeric[key] = (native, native)
            if promoted_dtype != native.dtype:
                # Preserve the retained JAX cast, including backend subnormal
                # handling. Only admission of a new metadata revision needs
                # this collection; steady blocks use the owned host snapshot.
                promotions.append((key, jnp.asarray(native).astype(promoted_dtype)))
        elif templates[0].params.dtype[key].kind in "biufc":
            return None
    if promotions:
        converted = jax.device_get(tuple(value for _, value in promotions))
        for (key, _), value in zip(promotions, converted):
            wide = np.array(value, copy=True)
            wide.setflags(write=False)
            numeric[key] = (numeric[key][0], wide)
    ids = np.asarray([template.id for template in templates], dtype=np.uint64)
    ids.setflags(write=False)
    cached = dict(signature=signature, keys=keys, templates=templates,
                  host=host, numeric=numeric, validated=tuple(validated),
                  params_dtypes=params_dtypes,
                  ids=ids,
                  sizes=sizes)
    control._jax_live_resident_metadata = cached
    return cached


@functools.partial(jax.jit, static_argnames=("groups", "limit"))
def _live_resident_candidate_plan(*values, groups, limit):
    """Select, order and limit a fixed-capacity bank entirely on device."""
    scaled, sigmas = values[:groups], values[groups:2 * groups]
    points = values[2 * groups:3 * groups]
    accepted = values[3 * groups:4 * groups]
    aborted = values[4 * groups:5 * groups]
    epoch, rate = values[-2:]
    snr = jnp.concatenate([jnp.abs(value) for value in scaled])
    phase = jnp.concatenate([jnp.angle(value) for value in scaled])
    sigma = jnp.concatenate([value.astype(jnp.float32) for value in sigmas])
    original = jnp.concatenate(accepted)
    active = original
    count = jnp.sum(active, dtype=jnp.int32)
    order = jnp.arange(snr.size, dtype=jnp.int32)
    if limit:
        # Reverse a stable ascending sort, matching argsort()[::-1], including
        # equal-SNR tie order and the native placement of accepted NaNs.
        order = jnp.argsort(jnp.where(active, snr, -jnp.inf))[::-1]
        retained = (jnp.minimum(count, limit) if limit > 0
                    else jnp.maximum(count + limit, 0))
        active = jnp.zeros_like(active).at[order].set(
            jnp.arange(snr.size) < retained)
    # Legacy timestamps divide in float64, then add the epoch. Keep the two
    # rounding boundaries separate; contraction changes large GPS values.
    relative = jax.lax.optimization_barrier(
        jnp.concatenate(points).astype(jnp.float64) / rate)
    times = jax.lax.optimization_barrier(relative + epoch)
    return (snr, phase, sigma, times, order, active, original,
            jnp.asarray([jnp.sum(mask, dtype=jnp.int32) for mask in accepted]),
            jnp.any(jnp.concatenate(aborted)))


@functools.partial(jax.jit, static_argnames=(
    "positions", "count", "n_time", "valid_start", "compatible",
    "use_pallas", "snr_threshold"))
def _live_resident_group_veto(
        correlation, peaks, norms, peak_indices, active, bins, *,
        positions, count, n_time, valid_start, compatible, use_pallas,
        snr_threshold):
    """Reuse filter correlations and skip inactive point recurrences.

    Every candidate has its own CUDA bin program. Zero-length bin intervals
    make absent candidates do zero recurrence iterations, without a host
    conditional or a dense candidate correlation gather. Fixed geometry and
    device masks also keep dispatch independent of trigger count.
    """
    from pycbc.vetoes.chisq_jax import (
        _compatible_shift_sum_jax, _compatible_shift_sum_pallas,
    )

    rows = jnp.asarray(positions, dtype=jnp.int32)
    selected = active[rows]
    points = peak_indices[rows].astype(jnp.int64) + valid_start
    snr = peaks[rows]
    norm = norms[rows].astype(jnp.float64)
    above = (abs(snr * norm) > snr_threshold if snr_threshold
             else jnp.ones(rows.shape, dtype=bool))
    edges = jnp.where((selected & above)[:, None], bins, 0)
    # Ordered phase preparation indexes bins by the original correlation row,
    # not by this geometry group's compact row ordinal. Retain that mapping
    # for interleaved bin-count/lower-cutoff groups without gathering products.
    edges = jnp.zeros((count, bins.shape[1]), bins.dtype).at[rows].set(edges)
    matrix = correlation.reshape(count, n_time)
    if not compatible:
        raise ValueError("Resident correlation reuse requires ordered chi-square")
    ordered_sum = (_compatible_shift_sum_pallas if use_pallas
                   else _compatible_shift_sum_jax)
    # The full filter root already has the materialized complex product.
    # Absolute bins exclude lower/negative frequencies, so base0 retains
    # exactly the previous cropped correlation's phase recurrence.
    shifts = ordered_sum(matrix, rows, points, edges,
                         jnp.asarray(n_time, jnp.int64), 0)
    num_bins = bins.shape[1] - 1
    bin_power = jax.lax.optimization_barrier(shifts * num_bins)
    snr_power = jax.lax.optimization_barrier((snr.conj() * snr).real)
    raw = jax.lax.optimization_barrier(bin_power - snr_power)
    norm_squared = jax.lax.optimization_barrier(
        (norm ** 2.0).astype(shifts.dtype))
    raw *= norm_squared
    dof = jnp.full(rows.shape, num_bins * 2 - 2, dtype=jnp.int32)
    if snr_threshold:
        raw = jnp.where(above, raw, 0.0)
        dof = jnp.where(above, dof, -100)
    return jnp.where(selected, raw, 0.0), jnp.where(selected, dof, 0)


@functools.partial(jax.jit, static_argnames=(
    "positions", "sizes", "newsnr_threshold"))
def _live_resident_veto_columns(
        snr, active, *values, positions, sizes, newsnr_threshold):
    """Restore original bank order and rank without sparse host indices."""
    group_count = len(positions)
    raw_groups, dof_groups = values[:group_count], values[group_count:]
    raw = jnp.zeros(snr.shape, dtype=jnp.result_type(*raw_groups))
    dof = jnp.zeros(snr.shape, dtype=jnp.int32)
    for rows, group_raw, group_dof in zip(positions, raw_groups, dof_groups):
        indices = jnp.asarray(rows, dtype=jnp.int32)
        raw = raw.at[indices].set(group_raw)
        dof = dof.at[indices].set(group_dof)
    reduced = (raw / dof).astype(jnp.float32)
    if newsnr_threshold:
        # Retain ranking_jax.newsnr_jax's float64 operations and eager stage
        # rounding while combining its dispatches into this device reduction.
        x = reduced.astype(jnp.float64)
        value = snr.astype(jnp.float64)
        power = jax.lax.optimization_barrier(x ** 3.0)
        term = jax.lax.optimization_barrier(1.0 + power)
        term = jax.lax.optimization_barrier(0.5 * term)
        weight = jax.lax.optimization_barrier(term ** (-1.0 / 6.0))
        weighted = jax.lax.optimization_barrier(value * weight)
        ranked = jnp.where(x > 1.0, weighted, value)
        active = active & (ranked >= newsnr_threshold)
    return reduced, dof.astype(jnp.uint32), jnp.zeros_like(reduced), active


class _LiveResidentResults:
    """Fixed-capacity GPU results with one explicit terminal host boundary."""

    def __init__(self, metadata, columns, plan, veto_columns):
        self.metadata = metadata
        self.columns = columns
        self.plan = plan
        self.veto_columns = veto_columns

    def materialize(self):
        """Collect final columns once, compact and publish legacy metadata."""
        from pycbc.events.live_collect_jax import collect_live_arrays

        return self._publish(collect_live_arrays(self._device_payload()))

    def _device_payload(self):
        return self.columns, self.plan, self.veto_columns

    def _publish(self, payload):
        """Publish an already collected payload, without another device call."""
        columns, plan, veto = payload
        ids = self.metadata["ids"]
        snr, phase, sigma, times = columns
        order, original, counts, aborted = plan
        chisq, dof, sg, keep = veto
        if bool(aborted):
            from pycbc.filter.matchedfilter import logger
            logger.info("We are seeing some *really* high SNRs, let's "
                        "assume they aren't signals and just give up")
            return False
        rows = order[keep[order]]
        result = {}
        promoted = bool(np.any(counts == 0))
        empty_bank_result = not bool(np.any(counts))
        numeric_values = self.metadata["numeric"]
        # Host-only parameters retain selected-row inference, including the
        # wider string dtype contributed by an empty float64 legacy group.
        accepted_rows = np.flatnonzero(original)
        accepted_order = np.cumsum(original, dtype=np.int64) - 1
        for key in self.metadata["keys"]:
            if key in numeric_values:
                native, wide = numeric_values[key]
                # With no accepted template, every legacy parameter list is
                # np.array([]), hence float64 even for complex/int fields.
                result[key] = (np.empty(0, dtype=np.float64)
                               if empty_bank_result else
                               (wide if promoted else native)[rows])
            else:
                parts, offset = [], 0
                values = self.metadata["host"][key]
                for size in self.metadata["sizes"]:
                    chosen = np.flatnonzero(original[offset:offset + size])
                    parts.append(np.array([values[offset + int(index)]
                                           for index in chosen]))
                    offset += size
                joined = np.concatenate(parts)
                result[key] = joined[accepted_order[rows]]
        # Keep lazy dict_params creation, using the owned snapshot rather than
        # a potentially advanced/mutated reader or bank view in the worker.
        for row in accepted_rows:
            template = self.metadata["templates"][int(row)]
            if not hasattr(template, "dict_params"):
                template.dict_params = {
                    key: self.metadata["host"][key][int(row)]
                    for key in self.metadata["keys"]}
        result.update(snr=snr[rows], coa_phase=phase[rows],
                      end_time=times[rows], template_id=ids[rows],
                      sigmasq=sigma[rows], chisq=chisq[rows],
                      chisq_dof=dof[rows], sg_chisq=sg[rows])
        # jax.device_get's dictionary pytree publication uses sorted keys.
        return dict(sorted(result.items()))


def _live_resident_batch_qualified(batch):
    """Correlation reuse requires the legacy veto's uniform complex64 path."""
    matrix = batch.template_matrix
    correlation = batch.correlations
    return (matrix is not None and correlation is not None
            and batch.n_time is not None
            and matrix.shape[0] == len(batch.templates)
            and correlation.size == len(batch.templates) * batch.n_time
            and all(np.dtype(dtype) == np.dtype(np.complex64) for dtype in (
                matrix.dtype, correlation.dtype, batch.peak_values.dtype,
                _live_array_metadata(batch.stilde)[1])))


def enqueue_live_data_jax(control, data_reader):
    """Queue all detector groups without collecting any peak decisions."""
    from pycbc import scheme

    device = getattr(scheme.mgr.state, "jax_device", None)
    if getattr(device, "platform", None) not in ("cuda", "gpu"):
        return process_live_data_jax(control, data_reader)
    control.set_data(data_reader)
    prepared = []
    while control.block_id < len(control.tgroups):
        prepared.append(_live_prepare_batch_jax(control))
    if not prepared:
        return control.process_all()
    metadata = None
    # The waveform reference is resolved before these filter inputs exist.
    if (os.environ.get("PYCBC_JAX_LIVE_RESIDENT", "1") != "0"
            and not (set(getattr(
                scheme.mgr.state, "jax_reference_operations", ())) - {"waveform"})
            and jax.config.jax_enable_x64
            and jax.config.jax_numpy_dtype_promotion == "standard"
            and getattr(scheme.mgr.state, "jax_chisq_mode",
                        "cpu-compatible") == "cpu-compatible"
            and not getattr(control.sg_chisq, "do", False)
            and getattr(control.power_chisq, "do", False)
            and all(_live_resident_batch_qualified(batch)
                    for batch in prepared)):
        metadata = _live_resident_metadata(control, prepared)
    return _LiveEnqueuedData(control, prepared, data_reader, metadata)


def _finish_live_data_compat(control, token):
    """Retain the sparse legacy assembler for unqualified GPU features."""
    from types import SimpleNamespace

    previous = control.data
    control.data = SimpleNamespace(start_time=token.start_time,
                                   sample_rate=token.sample_rate)
    try:
        selections = jax.device_get(tuple(batch.selection
                                         for batch in token.prepared))
        assembled = _live_finish_batches_jax(
            control, token.prepared, selections)
        if assembled is None:
            results, veto_info = [], []
            for batch, selection in zip(token.prepared, selections):
                result, veto = _live_finish_batch_jax(control, batch, selection)
                if result is False:
                    return False
                results.append(result)
                veto_info.extend(veto)
            result = control.combine_results(results)
        else:
            result, veto_info = assembled
        if control.max_triggers_in_batch:
            order = result['snr'].argsort()[::-1][:control.max_triggers_in_batch]
            result = {key: value[order] for key, value in result.items()}
            veto_info = [veto_info[index] for index in order]
        return control._process_vetoes(result, veto_info)
    finally:
        control.data = previous


def finish_live_data_jax(control, token):
    """Queue device candidate/veto plans after every detector was enqueued."""
    from pycbc.benchmark import stage_event
    from pycbc.vetoes.chisq_jax import (
        _chisq_mode, _live_chisq_executable, _use_gpu_ordered_scan,
        resident_power_chisq_bins_jax,
    )

    if not isinstance(token, _LiveEnqueuedData) or token.owner is not control:
        raise ValueError("Live prepared inputs belong to another control")
    from pycbc import scheme
    if (token.metadata is None
            or (set(getattr(
                scheme.mgr.state, "jax_reference_operations", ())) - {"waveform"})):
        return _finish_live_data_compat(control, token)
    plans = []
    for batch, psd in zip(token.prepared, token.psds):
        bins = resident_power_chisq_bins_jax(
            control.power_chisq, batch.templates, psd,
            batch.template_matrix)
        if bins is None:
            return _finish_live_data_compat(control, token)
        plans.append(bins)
    cache = _live_veto_executable_cache(control)
    groups = len(token.prepared)
    args = (*(batch.scaled_peaks for batch in token.prepared),
            *(batch.native_norms for batch in token.prepared),
            *(batch.selection[0] for batch in token.prepared),
            *(batch.selection[1] for batch in token.prepared),
            *(batch.selection[2] for batch in token.prepared),
            np.asarray(token.start_time, np.float64),
            np.asarray(token.sample_rate, np.float64))
    device = token.prepared[0].peak_values.device
    stage_event("filter_candidate_prepare", "start", groups=groups,
                resident=True)
    execute = _live_chisq_executable(
        _live_resident_candidate_plan, args,
        dict(groups=groups, limit=control.max_triggers_in_batch), cache, device)
    snr, phase, sigma, times, order, active, original, counts, aborted = (
        execute(*args))
    stage_event("filter_candidate_prepare", "end", groups=groups,
                resident=True)
    raw_groups, dof_groups, positions = [], [], []
    offset = 0
    stage_event("filter_veto", "start", groups=groups, resident=True)
    for batch, bin_groups in zip(token.prepared, plans):
        count = len(batch.templates)
        for rows, bins, _, n_time in bin_groups:
            local = tuple(map(int, rows))
            group_args = (batch.correlations, batch.peak_values, batch.norms,
                          batch.selection[0], active[offset:offset + count],
                          bins)
            execute = _live_chisq_executable(
                _live_resident_group_veto, group_args,
                dict(positions=local, count=count, n_time=n_time,
                     valid_start=batch.valid_start,
                     compatible=_chisq_mode() == "cpu-compatible",
                     use_pallas=(batch.peak_values.dtype == jnp.complex64
                                 and _use_gpu_ordered_scan(batch.peak_values)),
                     snr_threshold=control.power_chisq.snr_threshold),
                cache, device)
            raw, dof = execute(*group_args)
            raw_groups.append(raw)
            dof_groups.append(dof)
            positions.append(tuple(offset + row for row in local))
        offset += count
    args = (snr, active, *raw_groups, *dof_groups)
    execute = _live_chisq_executable(
        _live_resident_veto_columns, args,
        dict(positions=tuple(positions), sizes=token.metadata["sizes"],
             newsnr_threshold=control.newsnr_threshold), cache, device)
    veto = execute(*args)
    stage_event("filter_veto", "end", groups=groups, resident=True)
    return _LiveResidentResults(token.metadata, (snr, phase, sigma, times),
                                (order, original, counts, aborted), veto)


def materialize_live_results_jax(tree):
    """Snapshot a result tree at the terminal transport boundary only.

    This helper deliberately needs no active PyCBC scheme: the bounded result
    worker may run after the primary thread changes readers or enters a new
    block. All scientific inputs in resident result objects are immutable.
    """
    from pycbc.events.live_collect_jax import collect_live_arrays

    def device_tree(value):
        if isinstance(value, _LiveResidentResults):
            return value._device_payload()
        if isinstance(value, dict):
            return {key: device_tree(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(device_tree(item) for item in value)
        if isinstance(value, list):
            return [device_tree(item) for item in value]
        return value

    def publish(original, collected):
        if isinstance(original, _LiveResidentResults):
            return original._publish(collected)
        if isinstance(original, dict):
            return {key: publish(item, collected[key])
                    for key, item in original.items()}
        if isinstance(original, tuple):
            return tuple(publish(item, host) for item, host in
                         zip(original, collected))
        if isinstance(original, list):
            return [publish(item, host) for item, host in
                    zip(original, collected)]
        return (collected if _as_jax_array(original) is not None else original)

    return publish(tree, collect_live_arrays(device_tree(tree)))




def live_batch_matched_filter_init_jax(control, templates, maxelements=2**27):
    """Batch original duration groups with unchanged template and PSD grids."""
    from pycbc.fft import IFFT
    from pycbc.types import zeros
    from pycbc.filter.matchedfilter import BatchCorrelator

    durations = np.array([1.0 / t.delta_f for t in templates])

    lsort = durations.argsort()
    durations = durations[lsort]
    templates = [templates[li] for li in lsort]

    # Figure out how to chunk together the templates into groups to process
    _, counts = np.unique(durations, return_counts=True)
    tsamples = [(len(t) - 1) * 2 for t in templates]
    grabs = maxelements / np.unique(tsamples)

    chunks = np.array([])
    num = 0
    for count, grab in zip(counts, grabs):
        chunks = np.append(chunks, np.arange(num, count + num, grab))
        chunks = np.append(chunks, [count + num])
        num += count
    chunks = np.unique(chunks).astype(np.uint32)

    # We now have how many templates to grab at a time.
    control.chunks = chunks[1:] - chunks[0:-1]

    control.out_mem = {}
    control.cout_mem = {}
    control.ifts = {}
    chunk_durations = [durations[i] for i in chunks[:-1]]
    control.chunk_tsamples = [tsamples[int(i)] for i in chunks[:-1]]
    samples = control.chunk_tsamples * control.chunks

    # Create workspace memory for correlate and snr
    mem_ids = [(a, b) for a, b in zip(chunk_durations, control.chunks)]
    mem_types = set(zip(mem_ids, samples))

    control.tgroups, control.mids = [], []
    for i, size in mem_types:
        dur, count = i
        control.out_mem[i] = zeros(size, dtype=np.complex64)
        control.cout_mem[i] = zeros(size, dtype=np.complex64)
        control.ifts[i] = IFFT(
            control.cout_mem[i],
            control.out_mem[i],
            nbatch=count,
            size=len(control.cout_mem[i]) // count,
        )
        _bind_live_ifft_workspace_jax(control.ifts[i])

    # Split the templates into their processing groups
    for dur, count in mem_ids:
        tgroup = templates[0:count]
        control.tgroups.append(tgroup)
        control.mids.append((dur, count))
        templates = templates[count:]

    # Associate the snr and corr memory block to each template
    control.corr = []
    for i, tgroup in enumerate(control.tgroups):
        psize = control.chunk_tsamples[i]
        s = 0
        e = psize
        mid = control.mids[i]
        for htilde in tgroup:
            htilde.out = control.out_mem[mid][s:e]
            htilde.cout = control.cout_mem[mid][s:e]
            s += psize
            e += psize
        correlator = BatchCorrelator(
            tgroup, [t.cout for t in tgroup], len(tgroup[0]),
            immutable_templates=True)
        if control.cout_mem[mid]._data.device.platform in ("cuda", "gpu"):
            _bind_live_correlate_workspace_jax(correlator, control.cout_mem[mid])
        control.corr.append(correlator)

    control.unique_delta_fs = tuple(sorted(
        {group[0].delta_f for group in control.tgroups}, reverse=True))


def set_live_data_jax(control, data):
    """Tell the JAX reader which unchanged template grids are required."""
    try:
        data.required_delta_fs = control.unique_delta_fs
    except AttributeError:
        pass



@functools.partial(jax.jit, static_argnums=(1, 2))
def _live_ifft_flat_jax(flat, count, size):
    """Keep both workspace reshapes inside the unnormalized transform.

    Separate CUDA reshape dispatches copy immutable arrays. Within this JIT
    they are compiler bitcasts, preserving the frequency input without copies.
    """
    matrix = flat.reshape(count, size)
    return (jnp.fft.ifft(matrix, axis=-1) * size).reshape(-1)


_LiveIFFTWorkspace = namedtuple(
    "_LiveIFFTWorkspace",
    "invec outvec source target geometry shape dtype device compiled",
)


def _bind_live_ifft_workspace_jax(plan):
    """Qualify only the fixed, full complex64 CUDA buffers owned by Live."""
    from pycbc.fft.jaxfft import IFFT

    plan._jax_live_ifft_workspace = None
    if type(plan) is not IFFT or plan._reference is not None:
        return
    source, target = plan.invec._data, plan.outvec._data
    geometry = (plan.nbatch, plan.size, plan.idist, plan.odist)
    shape = (plan.nbatch * plan.size,)
    if (not isinstance(source, JAXArrayData)
            or not isinstance(target, JAXArrayData)
            or source is target or plan.inplace
            or source.parent is not None or target.parent is not None
            or source.slice_info is not None or target.slice_info is not None
            or source.shape != shape or target.shape != shape
            or source.dtype != np.dtype(np.complex64)
            or target.dtype != source.dtype or target.device != source.device
            or source.device.platform not in ("cuda", "gpu")
            or plan.idist != plan.size or plan.odist != plan.size):
        return
    plan._jax_live_ifft_workspace = _LiveIFFTWorkspace(
        plan.invec, plan.outvec, source, target, geometry, shape,
        source.dtype, source.device, plan._compiled,
    )


def _live_ifft_workspace_jax(plan):
    """Validate retained transform ownership without executing or publishing."""
    workspace = getattr(plan, "_jax_live_ifft_workspace", None)
    if workspace is not None:
        source, target = workspace.source, workspace.target
        if (plan.invec is workspace.invec and plan.outvec is workspace.outvec
                and plan.invec._data is source and plan.outvec._data is target
                and plan._compiled is workspace.compiled and not plan.inplace
                and (plan.nbatch, plan.size, plan.idist, plan.odist)
                == workspace.geometry
                and source.parent is None and target.parent is None
                and source.slice_info is None and target.slice_info is None
                and source.shape == target.shape == workspace.shape
                and source.dtype == target.dtype == workspace.dtype
                and source.array.dtype == target.array.dtype == workspace.dtype
                and source.device == target.device == workspace.device
                and source.array is not target.array):
            return workspace
        plan._jax_live_ifft_workspace = None
    return None


def _live_ifft_execute_jax(plan):
    """Execute a qualified Live transform, retaining the generic fallback."""
    if _reference_enabled("ifft"):
        plan.execute()
        return
    workspace = _live_ifft_workspace_jax(plan)
    if workspace is not None:
        workspace.target.set_array(_live_ifft_flat_jax(
            workspace.source.array, plan.nbatch, plan.size))
        return
    plan.execute()


def _set_output_array(z, val):
    """Store results, preserving output wrappers and exposing write errors."""
    data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
    if isinstance(data, JAXArrayData):
        data.set_array(val)
    elif data is not None:
        data[:] = np.asarray(val)
    else:
        z[:] = np.asarray(val)


def _cpu_correlate(x, y):
    """Execute original compiled correlation on explicit host arrays."""
    from pycbc.filter import matchedfilter_cpu

    # Cython's original writable memoryviews cannot accept read-only JAX views.
    x = np.asarray(x).copy()
    y = np.asarray(y).copy()
    z = np.empty_like(x)
    matchedfilter_cpu.correlate(
        SimpleNamespace(data=x), SimpleNamespace(data=y),
        SimpleNamespace(data=z))
    return to_jax(z)


def _inner_product(x, y, weight=None):
    """Share array kernels and their independent original-CPU selectors."""
    operation = "inner" if weight is None else "weighted_inner"
    if _reference_enabled(operation):
        args = (y,) if weight is None else (y, weight)
        return to_jax(_cpu_reference(x, operation, *args))
    if weight is None:
        return _fast_inner_self(x) if x is y else _fast_inner(x, y)
    return (_fast_weighted_inner_self(x, weight) if x is y
            else _fast_weighted_inner(x, y, weight))


def _inverse_transform(correlation):
    """Use the original CPU IFFT only when its validation route is selected."""
    if not _reference_enabled("ifft"):
        return jnp.fft.ifft(correlation) * len(correlation)
    from pycbc.fft import ifft
    from pycbc.types import Array

    source = Array(JAXArrayData(correlation), copy=False)
    target = Array(JAXArrayData(jnp.zeros_like(correlation)), copy=False)
    ifft(source, target)
    return to_jax(target)


@jax.jit
def correlate_jax(x, y):
    """Pure JAX elementwise conjugate multiplication: conj(x) * y."""
    return jnp.conj(x) * y


def correlate(x, y, z):
    """Elementwise z = conj(x) * y in JAX scheme."""
    _ensure_x64()
    x_arr = to_jax(x)
    y_arr = to_jax(y)
    prod = (_cpu_correlate(x_arr, y_arr) if _reference_enabled("correlate")
            else correlate_jax(x_arr, y_arr))
    _set_output_array(z[:len(prod)], prod)


class JAXCorrelator(_BaseCorrelator):
    """JAX correlator engine for persistent/repeated correlation."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def correlate(self):
        """Execute elementwise correlation."""
        correlate(self.x, self.y, self.z)


def _correlate_factory(x, y, z):
    """Factory returning the JAX correlator class."""
    return JAXCorrelator


@functools.partial(jax.jit, static_argnums=(4,))
def _batch_correlate_update(templates, y, parent, base, row_stride):
    """Correlate and publish one contiguous sibling-view block."""
    products = correlate_jax(templates, y)
    block_len = templates.shape[0] * row_stride
    block = jax.lax.dynamic_slice(parent, (base,), (block_len,))
    block = block.reshape(templates.shape[0], row_stride)
    block = block.at[:, :templates.shape[1]].set(products)
    return jax.lax.dynamic_update_slice(parent, block.reshape(-1), (base,))


@functools.partial(jax.jit, static_argnums=(3,))
def _live_construct_correlate_jax(templates, strain, parent, row_stride):
    """Build a complete Live buffer without updating an immutable old root.

    Preserve the current row tails, including nonzero negative frequencies.
    The explicit cast retains the scatter's destination rounding boundary.
    Inputs are never donated: retained raw arrays and lazy views stay valid.
    """
    products = correlate_jax(templates, strain).astype(parent.dtype)
    rows = parent.reshape(templates.shape[0], row_stride)
    return jnp.concatenate((products, rows[:, templates.shape[1]:]),
                           axis=1).reshape(-1)


def _batch_correlate_geometry_jax(outputs, size):
    """Validate sibling views without caching mutable caller geometry."""
    zdata = []
    for z in outputs:
        data = z if isinstance(z, JAXArrayData) else getattr(z, "_data", None)
        if not isinstance(data, JAXArrayData):
            return None
        info = data.slice_info
        parent = data.parent
        if parent is None or not isinstance(info, slice) or info.step not in (None, 1):
            return None
        start = 0 if info.start is None else info.start
        stop = parent.shape[0] if info.stop is None else info.stop
        zdata.append((parent, start, stop))

    if not zdata or not all(item[0] is zdata[0][0] for item in zdata):
        return None
    parent, base, end = zdata[0]
    row_stride = end - base
    if (row_stride < size
            or not all(stop - start == row_stride for _, start, stop in zdata)
            or not all(start == base + row * row_stride
                       for row, (_, start, _) in enumerate(zdata))
            or base + len(zdata) * row_stride > parent.shape[0]):
        return None
    return parent, base, row_stride


_LiveCorrelateWorkspace = namedtuple(
    "_LiveCorrelateWorkspace",
    "owner parent outputs xs matrix size shape dtype device base row_stride",
)


def _bind_live_correlate_workspace_jax(correlator, owner):
    """Retain geometry explicitly owned by the CUDA Live initializer.

    Live fixes its output views for the correlator's lifetime, just as it fixes
    its packed templates. Owning the output tuple prevents in-place row swaps;
    its members' storage and slice metadata belong to this fixed workspace.
    This contract must not be inferred from immutable_templates or a caller's
    tuple. Generic BatchCorrelators still validate every output on every call.
    No device array value is retained here: the root contents change per block.
    """
    correlator._jax_live_workspace = None
    outputs = tuple(correlator.zs)
    geometry = _batch_correlate_geometry_jax(outputs, correlator.size)
    matrix = getattr(correlator, "_jax_template_matrix", None)
    parent = getattr(owner, "_data", None)
    if (geometry is None or geometry[0] is not parent
            or parent.parent is not None or parent.slice_info is not None
            or len(parent.shape) != 1 or matrix is None
            or matrix.shape != (len(outputs), correlator.size)):
        return
    correlator.zs = outputs
    correlator._jax_live_workspace = _LiveCorrelateWorkspace(
        owner, parent, outputs, correlator.xs, matrix, correlator.size,
        parent.shape, parent.dtype, parent.device, geometry[1], geometry[2],
    )


def _live_correlate_workspace_jax(correlator, templates):
    """Reuse owned geometry, dropping ownership when the workspace changes."""
    workspace = getattr(correlator, "_jax_live_workspace", None)
    if workspace is None:
        return None
    parent = workspace.parent
    if (correlator.zs is workspace.outputs
            and correlator.xs is workspace.xs
            and getattr(correlator, "_jax_template_matrix", None) is workspace.matrix
            and correlator.size == workspace.size
            and getattr(workspace.owner, "_data", None) is parent
            and parent.parent is None and parent.slice_info is None
            and parent.shape == workspace.shape and parent.dtype == workspace.dtype
            and parent.array.dtype == workspace.dtype
            and parent.device == workspace.device
            and templates.shape == (len(workspace.outputs), workspace.size)):
        return workspace
    correlator._jax_live_workspace = None
    return None


def _live_full_correlate_workspace_jax(workspace):
    """Qualify construction only for explicitly owned complete CUDA roots."""
    return (workspace is not None
            and workspace.device.platform in ("cuda", "gpu")
            and workspace.base == 0
            and workspace.shape == (
                len(workspace.outputs) * workspace.row_stride,))


_LiveFusedBatchInputs = namedtuple("_LiveFusedBatchInputs", (
    "correlator", "plan", "correlation", "ifft", "templates", "strain",
    "parent", "norms", "threshold", "abort_threshold", "count", "size",
    "start", "stop"))


def _live_fused_batch_inputs_jax(
        correlator, plan, strain, norms, segment, threshold, abort_threshold):
    """Capture CUDA-owned inputs without dispatching the fused filter.

    Keep the original path for CPU, partial roots, changed plans and overridden
    peak helpers. The latter also preserves existing diagnostic instrumentation.
    """
    if any(_reference_enabled(operation) for operation in (
            "correlate", "ifft", "abs_arg_max", "live_selection")):
        return None
    if (_batch_peak_core, _live_select_peaks) != _LIVE_PEAK_FUNCTIONS_JAX:
        return None
    templates = getattr(correlator, "_jax_template_matrix", None)
    if templates is None:
        return None
    correlation = _live_correlate_workspace_jax(correlator, templates)
    if not _live_full_correlate_workspace_jax(correlation):
        return None
    transform = _live_ifft_workspace_jax(plan)
    if (transform is None or transform.source is not correlation.parent
            or plan.nbatch != len(correlation.outputs)
            or plan.size != correlation.row_stride
            or segment.step not in (None, 1)):
        return None
    start, stop, _ = segment.indices(plan.size)
    if start >= stop:
        return None
    strain = to_jax(strain)
    if strain.ndim != 1 or strain.size < correlation.size:
        return None
    if strain.size != correlation.size:
        strain = strain[:correlation.size]
    parent = correlation.parent.array
    if (norms.shape != (plan.nbatch,)
            or not all(isinstance(value, jax.Array)
                       and not isinstance(value, jax.core.Tracer)
                       and value.devices() == {transform.device}
                       for value in (templates, strain, parent, norms))):
        return None
    return _LiveFusedBatchInputs(
        correlator, plan, correlation, transform, templates, strain, parent,
        norms, threshold, abort_threshold, plan.nbatch, plan.size, start, stop,
    )


@functools.partial(jax.jit, static_argnames=("count", "size", "start", "stop"))
def _live_fused_batch_core_jax(
        templates, strain, parent, norms, threshold, abort_threshold,
        *, count, size, start, stop):
    """Construct, transform and select while returning both public buffers."""
    correlation = _live_construct_correlate_jax(templates, strain, parent, size)
    output = _live_ifft_flat_jax(correlation, count, size)
    # Preserve the existing JIT boundaries' rounded FFT output and peak values.
    # These barriers prevent contraction across stages without donating roots.
    output = jax.lax.optimization_barrier(output)
    indices, peaks = _batch_peak_core(output, count, start, stop)
    peaks = jax.lax.optimization_barrier(peaks)
    scaled, accepted, abort = _live_select_peaks(
        peaks, norms, threshold, abort_threshold)
    return correlation, output, indices, peaks, scaled, accepted, abort


def _live_launch_fused_batch_jax(inputs):
    """Dispatch captured inputs; publication remains a separate ordered step."""
    return _live_fused_batch_core_jax(
        inputs.templates, inputs.strain, inputs.parent, inputs.norms,
        inputs.threshold, inputs.abort_threshold,
        count=inputs.count, size=inputs.size, start=inputs.start, stop=inputs.stop,
    )


def _live_publish_fused_batch_jax(inputs, outputs):
    """Publish full roots only while their captured ownership remains valid."""
    correlation, output, indices, peaks, scaled, accepted, abort = outputs
    if (_live_correlate_workspace_jax(inputs.correlator, inputs.templates)
            is not inputs.correlation
            or _live_ifft_workspace_jax(inputs.plan) is not inputs.ifft
            or correlation.shape != inputs.correlation.shape
            or output.shape != inputs.ifft.shape
            or correlation.dtype != inputs.correlation.dtype
            or output.dtype != inputs.ifft.dtype):
        raise ValueError("Live fused workspace changed before publication")
    inputs.correlation.parent.set_array(correlation)
    inputs.ifft.target.set_array(output)
    return indices, peaks, scaled, accepted, abort


def batch_correlate_execute(self, y):
    """Vectorized batch correlation for BatchCorrelator in JAX scheme."""
    _ensure_x64()
    size = self.size
    y_arr = to_jax(y)[:size]
    if _reference_enabled("correlate"):
        for x, z in zip(self.xs, self.zs):
            products = _cpu_correlate(to_jax(x)[:size], y_arr)
            _set_output_array(z[:size], products)
        return

    # Lazy bank templates retain the complete 2-D device tensor.  When this
    # batch is an aligned contiguous run of rows, use it directly.  Reading
    # each LazyFrequencySeries first would materialize a separate static row
    # slice before stack, which is particularly costly during CUDA startup.
    templates = getattr(self, "_jax_template_matrix", None)
    if templates is not None:
        templates = templates[:, :size]

    if templates is None and self.xs:
        batch_tensor = getattr(self.xs[0], "_batch_tensor", None)
        positions = []
        if batch_tensor is not None:
            for x in self.xs:
                if getattr(x, "_batch_tensor", None) is not batch_tensor:
                    batch_tensor = None
                    break
                pos = getattr(x, "_batch_pos", None)
                if pos is None:
                    batch_tensor = None
                    break
                positions.append(int(pos))
        if batch_tensor is not None and positions:
            start = positions[0]
            aligned = positions == list(range(start, start + len(positions)))
            shape = getattr(batch_tensor, "shape", ())
            if aligned and len(shape) == 2 and start >= 0:
                stop = start + len(positions)
                if stop <= shape[0] and size <= shape[1]:
                    templates = to_jax(batch_tensor)[start:stop, :size]

    if templates is None:
        templates = jnp.stack([to_jax(x)[:size] for x in self.xs], axis=0)
    workspace = _live_correlate_workspace_jax(self, templates)
    if workspace is not None:
        if _live_full_correlate_workspace_jax(workspace):
            updated = _live_construct_correlate_jax(
                templates, y_arr, workspace.parent.array, workspace.row_stride)
        else:
            updated = _batch_correlate_update(
                templates, y_arr, workspace.parent.array,
                workspace.base, workspace.row_stride,
            )
        workspace.parent.set_array(updated)
        return
    # LiveBatchMatchedFilter passes sibling views into one contiguous output
    # allocation.  Updating each view separately makes JAX rebuild the whole
    # immutable parent for every template.  Fold the updates into one JAX
    # expression and publish the parent once, preserving any untouched tails.
    geometry = _batch_correlate_geometry_jax(self.zs, size)
    if geometry is not None:
        parent, base, row_stride = geometry
        updated = _batch_correlate_update(
            templates, y_arr, parent.array, base, row_stride,
        )
        parent.set_array(updated)
        return

    # General BatchCorrelator users may provide unrelated output arrays.
    # Retain the existing per-output behavior for those cases.
    products = correlate_jax(templates, y_arr)
    for row, z in enumerate(self.zs):
        _set_output_array(z[:size], products[row])


# ----------------------------------------------------------------------
# Pure Functional JAX API (JIT-compilable & Autodiff-compatible)
# ----------------------------------------------------------------------


def sigmasq_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of template loudness (sigma)^2."""
    _ensure_x64()
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    ht = h_arr[kmin:kmax]
    weight = (None if psd is None
              else _functional_array(psd, device)[kmin:kmax])
    return _inner_product(ht, ht, weight).real * (4.0 * delta_f)


def sigmasq_series_jax(
    htilde,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
):
    """Pure JAX calculation of cumulative power frequency series."""
    _ensure_x64()
    device = _functional_device(htilde, psd)
    h_arr = _functional_array(htilde, device)
    n_pts = (len(h_arr) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    mag_sq = (to_jax(_cpu_reference(h_arr, "squared_norm"))
              if _reference_enabled("squared_norm")
              else h_arr.real ** 2 + h_arr.imag ** 2)
    if psd is not None:
        psd_arr = _functional_array(psd, device)
        mag_sq = _divide(mag_sq, psd_arr, inplace=True)

    sub = mag_sq[kmin:kmax]
    sub = (to_jax(_cpu_reference(sub, "cumsum"))
           if _reference_enabled("cumsum") else jnp.cumsum(sub))
    norm = 4.0 * delta_f

    series = jnp.zeros(len(h_arr), dtype=h_arr.real.dtype, device=device)
    series = series.at[kmin:kmax].set(sub * norm)
    return series


def matched_filter_core_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    h_norm=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter core returning (snr, correlation, norm)."""
    _ensure_x64()
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    device = _functional_device(template, data, psd, h_norm)
    if isinstance(template, TimeSeries):
        template = make_frequency_series(template)
    if isinstance(data, TimeSeries):
        data = make_frequency_series(data)

    htilde = _functional_array(template, device)
    stilde = _functional_array(data, device)
    psd_arr = None if psd is None else _functional_array(psd, device)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    if delta_f is None:
        delta_f = getattr(data, "delta_f", 1.0)
    n_time = (len(stilde) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_time)

    qtilde = jnp.zeros(n_time, dtype=stilde.dtype, device=device)
    corr_slice = (_cpu_correlate(htilde[kmin:kmax], stilde[kmin:kmax])
                  if _reference_enabled("correlate")
                  else correlate_jax(htilde[kmin:kmax], stilde[kmin:kmax]))

    if psd_arr is not None:
        corr_slice = _divide(corr_slice, psd_arr[kmin:kmax], inplace=True)

    qtilde = qtilde.at[kmin:kmax].set(corr_slice)

    # Complex-to-complex IFFT of length n_time
    snr_time = _inverse_transform(qtilde)

    if h_norm is None:
        h_norm = sigmasq_jax(
            htilde,
            psd=psd_arr,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            delta_f=delta_f,
        )
    else:
        h_norm = _functional_array(h_norm, device)

    norm = (4.0 * delta_f) / jnp.sqrt(h_norm)
    return snr_time, qtilde, norm


def matched_filter_jax(
    template,
    data,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    sigmasq=None,
    delta_f=None,
    delta_t=None,
):
    """Pure JAX matched filter returning normalized complex/real SNR."""
    snr_time, _, norm = matched_filter_core_jax(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        h_norm=sigmasq,
        delta_f=delta_f,
        delta_t=delta_t,
    )
    return snr_time * norm


def overlap_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    normalized=True,
):
    """Calculate overlap between two frequency series in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    device = _functional_device(vec1, vec2, psd)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    v1_sub = v1[kmin:kmax]
    v2_sub = v2[kmin:kmax]
    weight = None if psd_arr is None else psd_arr[kmin:kmax]
    inn = _inner_product(v1_sub, v2_sub, weight).real * (4.0 * delta_f)
    if not normalized:
        return inn

    norm1 = sigmasq_jax(
        v1,
        psd=psd_arr,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    norm2 = sigmasq_jax(
        v2,
        psd=psd_arr,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=delta_f,
    )
    return inn / jnp.sqrt(norm1 * norm2)


def match_jax(
    vec1,
    vec2,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=1.0,
    v1_norm=None,
    v2_norm=None,
):
    """Calculate match (overlap maximized over time and phase) in JAX."""
    from pycbc.types import TimeSeries
    from pycbc.filter import make_frequency_series

    device = _functional_device(vec1, vec2, psd, v1_norm, v2_norm)
    if isinstance(vec1, TimeSeries):
        vec1 = make_frequency_series(vec1)
    if isinstance(vec2, TimeSeries):
        vec2 = make_frequency_series(vec2)

    v1 = _functional_array(vec1, device)
    v2 = _functional_array(vec2, device)
    psd_arr = None if psd is None else _functional_array(psd, device)
    n_pts = (len(v1) - 1) * 2
    flow = low_frequency_cutoff
    fhigh = high_frequency_cutoff

    kmin, kmax = get_cutoff_indices(flow, fhigh, delta_f, n_pts)

    corr = jnp.zeros(n_pts, dtype=v1.dtype, device=device)
    prod = (_cpu_correlate(v1[kmin:kmax], v2[kmin:kmax])
            if _reference_enabled("correlate")
            else correlate_jax(v1[kmin:kmax], v2[kmin:kmax]))
    if psd_arr is not None:
        prod = _divide(prod, psd_arr[kmin:kmax], inplace=True)

    corr = corr.at[kmin:kmax].set(prod)
    time_series = _inverse_transform(corr)

    if _reference_enabled("abs_max_loc"):
        max_value, max_idx = _cpu_reference(time_series, "abs_max_loc")
    else:
        magnitude = time_series.real ** 2 + time_series.imag ** 2
        max_idx = int(jnp.argmax(magnitude))
        max_value = jnp.sqrt(magnitude[max_idx])

    if v1_norm is None:
        v1_norm = sigmasq_jax(
            v1,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v1_norm = _functional_array(v1_norm, device)
    if v2_norm is None:
        v2_norm = sigmasq_jax(
            v2,
            psd=psd_arr,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            delta_f=delta_f,
        )
    else:
        v2_norm = _functional_array(v2_norm, device)

    # NumPy scalar arithmetic in the original match retains peak precision.
    peak = _functional_array(max_value, device)
    snr_norm = ((4.0 * delta_f) / jnp.sqrt(v1_norm)).astype(peak.dtype)
    second_norm = jnp.sqrt(v2_norm).astype(peak.dtype)
    return _divide(peak * snr_norm, second_norm), max_idx


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _batch_peak_core(tensor, template_count, segment_start, segment_stop):
    """Reduce a fixed-shape batch segment in one compiled JAX operation."""
    values = tensor.reshape(template_count, -1)[:, segment_start:segment_stop]
    if jnp.iscomplexobj(values):
        sq_mag = values.real ** 2 + values.imag ** 2
    else:
        sq_mag = values ** 2
    indices = jnp.argmax(sq_mag, axis=-1)
    peaks = values[jnp.arange(template_count), indices]
    return indices, peaks


_LIVE_PEAK_FUNCTIONS_JAX = (_batch_peak_core, _live_select_peaks)


def batch_peak_values(output, template_count, template_size, segment):
    """Reduce contiguous output to one peak index and value per template.

    Avoids per-template host synchronization by reducing the full 2D batch
    allocation in a single vectorized JAX kernel.
    """
    _ensure_x64()

    tensor = to_jax(output)
    template_count = int(template_count)
    template_size = int(template_size)
    if (
        template_count < 1
        or template_size < 1
        or tensor.size != template_count * template_size
    ):
        return None

    if segment.step not in (None, 1):
        return None

    segment_start, segment_stop, _ = segment.indices(template_size)
    if segment_start >= segment_stop:
        return None
    if _reference_enabled("abs_arg_max"):
        values = tensor.reshape(template_count, template_size)[
            :, segment_start:segment_stop]
        indices = to_jax(np.asarray([
            _cpu_reference(row, "abs_arg_max") for row in values], np.int64))
        peaks = values[jnp.arange(template_count), indices]
    else:
        indices, peaks = _batch_peak_core(
            tensor, template_count, segment_start, segment_stop,
        )
    # Keep reductions on the selected JAX device.  The live filter consumes
    # these values directly and must not synchronize each template through a
    # NumPy conversion before veto processing.
    return indices, peaks


def batch_matched_filter_bank(
    templates,
    strain,
    psd=None,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
    delta_f=None,
):
    """High-throughput batched template bank matched filtering in pure JAX.

    Uses the single-template calculation's band, correlation/division order,
    normalization and output precision. The default kernels run on-device;
    selected validation operations use their original CPU implementations.

    Parameters
    ----------
    templates : jax.Array or FrequencySeries
        2D array of templates with shape (num_templates, n_freq) or list of
        FrequencySeries.
    strain : jax.Array or FrequencySeries
        1D strain frequency series of length n_freq.
    psd : jax.Array or FrequencySeries, optional
        1D PSD frequency series of length n_freq.
    low_frequency_cutoff : float, optional
        Low frequency cutoff for integration in Hz.
    high_frequency_cutoff : float, optional
        High frequency cutoff for integration in Hz.
    delta_f : float, optional
        Frequency spacing in Hz.

    Returns
    -------
    norm_snr : jax.Array
        2D array of normalized complex SNR time series of shape
        (num_templates, n_time).
    sigmasq : jax.Array
        1D array of template variances (sigmasq) of length num_templates.
    """
    _ensure_x64()
    inputs = tuple(templates) if isinstance(templates, (list, tuple)) else (
        templates,)
    device = _functional_device(*inputs, strain, psd)
    if isinstance(templates, (list, tuple)):
        templates_j = jnp.stack([
            _functional_array(template, device) for template in templates])
        template_df = getattr(templates[0], "delta_f", None)
    else:
        templates_j = _functional_array(templates, device)
        template_df = getattr(templates, "delta_f", None)
    if templates_j.ndim == 1:
        templates_j = templates_j[None, :]
    df = delta_f if delta_f is not None else getattr(strain, "delta_f", None)
    if df is None:
        df = template_df if template_df is not None else 1.0

    strain_j = _functional_array(strain, device)
    n_time = (strain_j.shape[-1] - 1) * 2
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, df, n_time)
    template_band = templates_j[:, kmin:kmax]
    strain_band = strain_j[kmin:kmax]
    weight = (None if psd is None
              else _functional_array(psd, device)[kmin:kmax])
    if _reference_enabled("weighted_inner" if weight is not None else "inner"):
        sigmasqs = jnp.stack([
            _inner_product(row, row, weight).real for row in template_band])
    elif weight is None:
        sigmasqs = jax.vmap(_fast_inner_self)(template_band)
    else:
        sigmasqs = jax.vmap(_fast_weighted_inner_self, in_axes=(0, None))(
            template_band, weight
        )
    sigmasqs = sigmasqs * (4.0 * df)

    products = (jnp.stack([_cpu_correlate(row, strain_band)
                          for row in template_band])
                if _reference_enabled("correlate") else correlate_jax(
                    template_band, strain_band))
    if weight is not None:
        products = _divide(products, weight[None, :], inplace=True)
    correlation = jnp.zeros((len(templates_j), n_time), dtype=strain_j.dtype,
                            device=device)
    correlation = correlation.at[:, kmin:kmax].set(products)
    snr = (jnp.stack([_inverse_transform(row) for row in correlation])
           if _reference_enabled("ifft") else jnp.fft.ifft(
               correlation, axis=-1) * n_time)
    norm = (4.0 * df) / jnp.sqrt(sigmasqs)
    # Native matched_filter multiplies its existing complex output in place.
    norm = norm.astype(snr.real.dtype)
    return (snr * norm[:, None]).astype(snr.dtype), sigmasqs


def _batched_filter_core(
    templates_2d,
    seg_slice,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    full_templates=False,
):
    """Build the shared batched correlation and valid SNR series."""
    if full_templates:
        templates_2d = templates_2d[:, kmin:kmax]
    corr_slice = jnp.conj(templates_2d) * seg_slice[None, :]
    pad_left = kmin
    pad_right = tlen - kmax
    qtilde = jnp.pad(corr_slice, ((0, 0), (pad_left, pad_right)))
    snr_series = jnp.fft.ifft(qtilde, axis=-1) * tlen
    valid_snr = snr_series[:, valid_start:valid_stop]
    return snr_series, valid_snr, corr_slice


def _cluster_filtered_batch(valid_snr, thresh_sq, window):
    """Cluster inside the filter JIT without recomputing SNR magnitudes."""
    from pycbc.events.threshold_jax import _batched_cluster_from_magnitude

    mag_sq = valid_snr.real ** 2 + valid_snr.imag ** 2
    return _batched_cluster_from_magnitude(
        valid_snr, mag_sq, thresh_sq, window
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "window",
        "full_templates",
    ),
)
def _batched_filter_and_cluster(
    templates_2d,
    seg_slice,
    thresh_sq,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    window,
    full_templates=False,
):
    """JIT correlation, IFFT, and clustering as one compiled boundary."""
    snr_series, valid_snr, corr_slice = _batched_filter_core(
        templates_2d, seg_slice, kmin, kmax, tlen, valid_start,
        valid_stop, full_templates=full_templates,
    )
    clustered = _cluster_filtered_batch(valid_snr, thresh_sq, window)
    return (snr_series, corr_slice, *clustered)


@functools.partial(
    jax.jit,
    static_argnames=(
        "kmin", "kmax", "tlen", "valid_start", "valid_stop", "window",
        "full_templates",
    ),
)
def _batched_filter_and_cluster_lean(
    templates_2d,
    seg_slice,
    thresh_sq,
    kmin,
    kmax,
    tlen,
    valid_start,
    valid_stop,
    window,
    full_templates=False,
):
    """Compiled filter and cluster path without retaining full SNR series."""
    _, valid_snr, corr_slice = _batched_filter_core(
        templates_2d, seg_slice, kmin, kmax, tlen, valid_start,
        valid_stop, full_templates=full_templates,
    )
    clustered = _cluster_filtered_batch(valid_snr, thresh_sq, window)
    return (corr_slice, *clustered)


class JAXMatchedFilterControl(MatchedFilterControl):
    """Matched-filter control with device batch execution for JAX searches."""

    def batched_matched_filter_and_cluster(
        self, segnum, templates, sigmasqs, window, epoch=None
    ):
        return batched_matched_filter_and_cluster_jax(
            self, segnum, templates, sigmasqs, window, epoch=epoch)

    def clear_batch_cache(self):
        """Release the preceding batch's packed template tensor."""
        self._cached_templates_key = None
        self._cached_templates_2d = None


def _reference_batched_filter(control, segnum, templates, sigmasqs, window, epoch):
    """Apply selected original primitives through the scalar control API."""
    from pycbc.types import Array

    results = []
    for template, sigmasq in zip(templates, sigmasqs):
        control.htilde[control.kmin:control.kmax] = Array(
            JAXArrayData(to_jax(template)[control.kmin:control.kmax]), copy=False)
        result = control.matched_filter_and_cluster(
            segnum, sigmasq, window, epoch=epoch)
        # Scalar controls reuse workspaces; retain this template's outputs.
        results.append(tuple(value.copy() if hasattr(value, "copy") else value
                             for value in result))
    return results


def batched_matched_filter_and_cluster_jax(
    mf_control,
    segnum,
    templates,
    sigmasqs,
    window,
    epoch=None,
):
    """Batched matched filtering, thresholding, and clustering on JAX device.

    Parameters
    ----------
    mf_control : MatchedFilterControl
        The matched filter control object holding analysis configuration.
    segnum : int
        Index of the segment to filter against.
    templates : list of FrequencySeries
        Templates in the current batch.
    sigmasqs : sequence of float
        Normalization factors for each template in the batch.
    window : int
        Clustering window size in samples.
    epoch : optional
        GPS epoch for the returned TimeSeries.

    Returns
    -------
    list of tuples
        For each template, returns (snr, norm, corr, idx, snrv) matching
        MatchedFilterControl's contract.
    """
    _ensure_x64()
    from pycbc.types import Array, TimeSeries

    b = len(sigmasqs)
    if b == 0:
        return []

    if any(_reference_enabled(operation) for operation in (
            "correlate", "ifft", "threshold_cluster")):
        return _reference_batched_filter(
            mf_control, segnum, templates, sigmasqs, window, epoch)

    seg = mf_control.segments[segnum]
    kmin, kmax = mf_control.kmin, mf_control.kmax
    tlen = mf_control.tlen
    delta_f = mf_control.delta_f
    delta_t = mf_control.delta_t
    valid_start = seg.analyze.start
    valid_stop = seg.analyze.stop
    threshold = float(mf_control.snr_threshold)

    from pycbc import scheme
    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    # Pre-stack all segments into a persistent 2D tensor in VRAM during setup / first call
    segment_key = (kmin, kmax, target_dev,
                   tuple(id(to_jax(s, device=target_dev))
                         for s in mf_control.segments))
    cached_seg_tensor = getattr(mf_control, "_jax_segments_tensor", None)
    if (
        cached_seg_tensor is None
        or getattr(mf_control, "_jax_segments_key", None) != segment_key
        or (target_dev is not None
            and cached_seg_tensor.devices() != {target_dev})
    ):
        slices = []
        for s in mf_control.segments:
            s_jax = to_jax(s, device=target_dev)
            slices.append(s_jax[kmin:kmax])
        mf_control._jax_segments_tensor = jnp.stack(slices, axis=0)
        mf_control._jax_segments_key = segment_key
        if target_dev is not None:
            mf_control._jax_segments_tensor = jax.device_put(
                mf_control._jax_segments_tensor, target_dev
            )

    seg_slice = mf_control._jax_segments_tensor[segnum]

    # Cache 2D templates on mf_control across segments to avoid re-stacking 5x per batch
    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is not None:
        cache_key = (id(batch_tensor), kmin, kmax, target_dev, True)
        full_templates = True
    else:
        cache_key = (tuple(id(to_jax(t, device=target_dev)) for t in templates),
                     kmin, kmax, target_dev, False)
        full_templates = False

    if getattr(mf_control, "_cached_templates_key", None) == cache_key:
        templates_2d = mf_control._cached_templates_2d
    else:
        if batch_tensor is not None:
            # Keep the full batch tensor cached. Slicing inside the JIT avoids
            # retaining a second device allocation for the cropped templates.
            templates_2d = to_jax(batch_tensor, device=target_dev)
        elif hasattr(templates, "ndim") and templates.ndim == 2:
            templates_2d = to_jax(templates, device=target_dev)[:, kmin:kmax]
        else:
            templates_2d = jnp.stack(
                [to_jax(t, device=target_dev)[kmin:kmax] for t in templates], axis=0
            )
        if target_dev is not None:
            templates_2d = jax.device_put(templates_2d, target_dev)
        mf_control._cached_templates_key = cache_key
        mf_control._cached_templates_2d = templates_2d

    norms_host = np.asarray([(4.0 * delta_f) / math.sqrt(value)
                             for value in sigmasqs], dtype=np.float64)
    thresh_sq_np = (threshold / norms_host) ** 2
    if target_dev is not None:
        thresh_sq = jax.device_put(thresh_sq_np, target_dev)
    else:
        thresh_sq = jax.device_put(thresh_sq_np)

    # Keep correlation, IFFT, magnitude reduction, thresholding, and
    # clustering in one compiled boundary. This prevents a duplicate pass
    # over the full valid SNR matrix and avoids an intermediate launch.
    need_snr = getattr(mf_control, "need_snr_series", False)
    platform = getattr(target_dev, "platform", "cpu") if target_dev else "cpu"
    is_cuda = platform in ("cuda", "gpu")

    if not is_cuda and b > 1:
        corr_slices, max_idxs, masks, max_snrs = [], [], [], []
        snr_series_list = [] if need_snr else None
        for i in range(b):
            t_slice = templates_2d[i:i + 1]
            th_slice = thresh_sq[i:i + 1]
            if need_snr:
                (snr_s, c_s, m_idx, s_mask, m_snr) = _batched_filter_and_cluster(
                    t_slice,
                    seg_slice,
                    th_slice,
                    kmin,
                    kmax,
                    tlen,
                    valid_start,
                    valid_stop,
                    window,
                    full_templates=full_templates,
                )
                snr_series_list.append(snr_s[0])
            else:
                (c_s, m_idx, s_mask, m_snr) = (
                    _batched_filter_and_cluster_lean(
                        t_slice,
                        seg_slice,
                        th_slice,
                        kmin,
                        kmax,
                        tlen,
                        valid_start,
                        valid_stop,
                        window,
                        full_templates=full_templates,
                    )
                )
            corr_slices.append(c_s[0])
            max_idxs.append(m_idx[0])
            masks.append(s_mask[0])
            max_snrs.append(m_snr[0])
        corr_slice = jnp.stack(corr_slices, axis=0)
        batched_max_idx = jnp.stack(max_idxs, axis=0)
        batched_survivor_mask = jnp.stack(masks, axis=0)
        batched_max_snr = jnp.stack(max_snrs, axis=0)
        snr_series = (
            jnp.stack(snr_series_list, axis=0) if need_snr else None
        )
    else:
        if need_snr:
            (snr_series, corr_slice, batched_max_idx,
             batched_survivor_mask, batched_max_snr) = _batched_filter_and_cluster(
                templates_2d,
                seg_slice,
                thresh_sq,
                kmin,
                kmax,
                tlen,
                valid_start,
                valid_stop,
                window,
                full_templates=full_templates,
            )
        else:
            (corr_slice, batched_max_idx,
             batched_survivor_mask, batched_max_snr) = (
                _batched_filter_and_cluster_lean(
                    templates_2d,
                    seg_slice,
                    thresh_sq,
                    kmin,
                    kmax,
                    tlen,
                    valid_start,
                    valid_stop,
                    window,
                    full_templates=full_templates,
                )
            )
            snr_series = None

    empty_idx = np.empty(0, dtype=np.uint32)
    empty_snrv = np.empty(0, dtype=np.complex64)

    # One bounded device-to-host transfer replaces per-template boolean
    # synchronizations.  These arrays contain one candidate per clustering
    # window, not the full SNR series.
    host_max_idx, host_survivor_mask, host_max_snr = jax.device_get(
        (batched_max_idx, batched_survivor_mask, batched_max_snr)
    )

    results = []
    if not np.any(host_survivor_mask):
        for i in range(b):
            results.append(([], float(norms_host[i]), [], empty_idx, empty_snrv))
        return results

    from pycbc.waveform.bank_jax import LazyFrequencySeries

    for i in range(b):
        norm_i = float(norms_host[i])
        mask = host_survivor_mask[i]
        if not np.any(mask):
            results.append(([], norm_i, [], empty_idx, empty_snrv))
            continue

        survivor_indices = np.asarray(
            host_max_idx[i][mask], dtype=np.uint32
        )
        survivor_values = np.asarray(host_max_snr[i][mask])

        corr = LazyFrequencySeries(corr_slice, i, delta_f)
        corr._kmin = kmin
        corr._tlen = tlen

        if need_snr and snr_series is not None:
            snr = TimeSeries(
                Array(JAXArrayData(snr_series[i]), copy=False),
                epoch=epoch,
                delta_t=delta_t,
                copy=False,
            )
        else:
            snr = None

        results.append((snr, norm_i, corr, survivor_indices, survivor_values))

    del batched_max_idx, batched_survivor_mask, batched_max_snr
    if snr_series is not None:
        del snr_series
    if corr_slice is not None:
        del corr_slice

    return results


def process_batch_inspiral_jax(
    event_mgr,
    bank,
    batch_tnums,
    segments,
    matched_filter,
    power_chisq,
    cluster_window,
    next_batch_tnums=None,
    bank_chisq=None,
    autochisq=None,
    sg_chisq=None,
    inj_filter_rejector=None,
    opt=None,
):
    """Execute batched filtering, vetoes, and direct event manager insertion for JAX."""
    import logging
    from pycbc.events.eventmgr import findchirp_cluster_over_window_cython
    from pycbc.vetoes.chisq_jax import (
        batch_power_chisq_jax,
        cache_batch_power_chisq_bins_jax,
    )

    b = len(batch_tnums)
    if b == 0:
        return

    if hasattr(bank, "get_batch"):
        batch_templates = bank.get_batch(batch_tnums)
        if next_batch_tnums is not None and hasattr(bank, "prefetch_batch_jax"):
            bank.prefetch_batch_jax(
                next_batch_tnums,
                power_chisq=power_chisq,
                psd=segments[0].psd,
            )
    else:
        batch_templates = [bank[i] for i in batch_tnums]

    # Pre-cache chi-square bins for the batch
    if power_chisq is not None and getattr(power_chisq, "do", False):
        cache_batch_power_chisq_bins_jax(
            power_chisq, batch_templates, segments[0].psd
        )

    sigmasq_caches = {}
    triggered_events = [[] for _ in range(b)]
    flow = getattr(opt, "low_frequency_cutoff", None)

    for s_num, stilde in enumerate(segments):
        active_indices = []
        active_templates = []
        active_sigmasqs = []
        psd_id = id(stilde.psd)

        if psd_id not in sigmasq_caches:
            sigmasq_caches[psd_id] = np.asarray(jax.device_get(
                live_template_norms_jax(
                    batch_templates, stilde.psd,
                    template_matrix=getattr(batch_templates, "_batch_tensor", None)
                )), dtype=np.float64)
        batch_sigmasqs = sigmasq_caches[psd_id]

        if inj_filter_rejector is None:
            active_indices = list(range(b))
            active_templates = batch_templates
            active_sigmasqs = batch_sigmasqs
        else:
            for i, t_num in enumerate(batch_tnums):
                if not inj_filter_rejector.template_segment_checker(
                    bank, t_num, stilde
                ):
                    continue
                active_indices.append(i)
                active_templates.append(batch_templates[i])
                active_sigmasqs.append(batch_sigmasqs[i])

            if not active_indices:
                continue

            batch_tensor = getattr(batch_templates, "_batch_tensor", None)
            if batch_tensor is not None:
                from pycbc.waveform.bank_jax import TemplateBatchList
                act_list = TemplateBatchList(active_templates)
                if len(active_indices) == len(batch_templates):
                    act_list._batch_tensor = batch_tensor
                else:
                    act_list._batch_tensor = batch_tensor[jnp.asarray(active_indices)]
                active_templates = act_list

        if opt and getattr(opt, "update_progress", None):
            from pycbc.workflow import update_progress
            update_progress(
                (batch_tnums[0] + (s_num / float(len(segments)))) / len(bank),
                opt.update_progress,
                getattr(opt, "update_progress_file", None),
            )

        logging.info(
            "Filtering template batch %d-%d/%d segment %d/%d"
            % (batch_tnums[0] + 1, batch_tnums[-1] + 1, len(bank), s_num + 1, len(segments))
        )

        batch_results = matched_filter.batched_matched_filter_and_cluster(
            s_num,
            active_templates,
            active_sigmasqs,
            cluster_window,
            epoch=stilde._epoch,
        )

        # Check if any template in this segment produced triggers
        has_triggers = any(len(res[3]) > 0 for res in batch_results)
        if not has_triggers:
            continue

        chisq_map = None
        if power_chisq is not None and getattr(power_chisq, "do", False):
            corr_tensor = None
            for res in batch_results:
                if res[2] is not None and hasattr(res[2], "_batch_tensor"):
                    corr_tensor = res[2]._batch_tensor
                    break
            if corr_tensor is not None:
                chisq_map = batch_power_chisq_jax(
                    corr_tensor,
                    batch_results,
                    active_templates,
                    stilde.psd,
                    stilde.analyze.start,
                    snr_threshold=power_chisq.snr_threshold,
                    power_chisq=power_chisq,
                )

        for act_pos, tmpl_idx in enumerate(active_indices):
            snr, norm, corr, idx, snrv = batch_results[act_pos]
            if not len(idx):
                continue

            tmpl = active_templates[act_pos]
            sigmasq = active_sigmasqs[act_pos]
            idx_cum = idx + stilde.cumulative_index
            snr_vals = snrv * norm

            if chisq_map is not None and act_pos in chisq_map:
                chisq_val, chisq_dof_val = chisq_map[act_pos]
            elif power_chisq is not None and getattr(power_chisq, "do", False):
                chisq_val, chisq_dof_val = power_chisq.values(
                    corr, snrv, norm, stilde.psd, idx + stilde.analyze.start, tmpl
                )
            else:
                chisq_val, chisq_dof_val = None, None

            bank_chisq_val, bank_chisq_dof_val = (
                bank_chisq.values(tmpl, stilde.psd, stilde, snrv, norm, idx + stilde.analyze.start)
                if bank_chisq is not None and getattr(bank_chisq, "do", False)
                else (None, None)
            )
            sg_chisq_val = (
                sg_chisq.values(stilde, tmpl, stilde.psd, snrv, norm, chisq_val, chisq_dof_val, idx + stilde.analyze.start)
                if sg_chisq is not None and getattr(sg_chisq, "do", False)
                else None
            )
            cont_chisq_val, cont_chisq_dof_val = (
                autochisq.values(snr, idx + stilde.analyze.start, tmpl, stilde.psd, norm, stilde=stilde, low_frequency_cutoff=flow)
                if autochisq is not None and getattr(autochisq, "do", False)
                else (None, None)
            )

            triggered_events[tmpl_idx].append({
                "time_index": idx_cum,
                "snr": snr_vals,
                "chisq": chisq_val,
                "chisq_dof": chisq_dof_val,
                "sigmasq": sigmasq,
                "bank_chisq": bank_chisq_val,
                "bank_chisq_dof": bank_chisq_dof_val,
                "sg_chisq": sg_chisq_val,
                "cont_chisq": cont_chisq_val,
                "cont_chisq_dof": cont_chisq_dof_val,
            })

    # Add accumulated events directly to event_mgr per template
    for tmpl_idx, ev_list in enumerate(triggered_events):
        if not ev_list:
            continue
        tmpl = batch_templates[tmpl_idx]
        if len(ev_list) == 1:
            ev = ev_list[0]
            times = ev["time_index"]
            snrs = ev["snr"]
            chisqs = ev["chisq"]
            dofs = ev["chisq_dof"]
            sigmasq_val = ev["sigmasq"]
            sigmasqs_arr = np.full(len(times), sigmasq_val, dtype=np.float32)
            b_chisq = ev["bank_chisq"]
            b_dof = ev["bank_chisq_dof"]
            sg = ev["sg_chisq"]
            cont = ev["cont_chisq"]
            cont_dof = ev["cont_chisq_dof"]
        else:
            times = np.concatenate([e["time_index"] for e in ev_list])
            snrs = np.concatenate([e["snr"] for e in ev_list])
            chisqs = np.concatenate([e["chisq"] for e in ev_list]) if ev_list[0]["chisq"] is not None else None
            dofs = np.concatenate([e["chisq_dof"] for e in ev_list]) if ev_list[0]["chisq_dof"] is not None else None
            sigmasqs_arr = np.concatenate([np.full(len(e["time_index"]), e["sigmasq"], dtype=np.float32) for e in ev_list])
            b_chisq = np.concatenate([e["bank_chisq"] for e in ev_list]) if ev_list[0]["bank_chisq"] is not None else None
            b_dof = np.concatenate([e["bank_chisq_dof"] for e in ev_list]) if ev_list[0]["bank_chisq_dof"] is not None else None
            sg = np.concatenate([e["sg_chisq"] for e in ev_list]) if ev_list[0]["sg_chisq"] is not None else None
            cont = np.concatenate([e["cont_chisq"] for e in ev_list]) if ev_list[0]["cont_chisq"] is not None else None
            cont_dof = np.concatenate([e["cont_chisq_dof"] for e in ev_list]) if ev_list[0]["cont_chisq_dof"] is not None else None

        columns = jax.device_get((
            times, snrs, chisqs, dofs, b_chisq, b_dof, sg, cont, cont_dof))
        times, snrs, chisqs, dofs, b_chisq, b_dof, sg, cont, cont_dof = (
            None if value is None else np.asarray(value) for value in columns)

        if cluster_window > 0 and len(times) > 1:
            times_i32 = times.astype(np.int32)
            indices = np.zeros(len(times), dtype=np.int32)
            count = findchirp_cluster_over_window_cython(
                times_i32, np.asarray(abs(snrs)), cluster_window, indices, len(times)
            )
            indices = indices[:count + 1]
            times = times[indices]
            snrs = snrs[indices]
            if chisqs is not None:
                chisqs = chisqs[indices]
            if dofs is not None:
                dofs = dofs[indices]
            sigmasqs_arr = sigmasqs_arr[indices]
            if b_chisq is not None:
                b_chisq = b_chisq[indices]
            if b_dof is not None:
                b_dof = b_dof[indices]
            if sg is not None:
                sg = sg[indices]
            if cont is not None:
                cont = cont[indices]
            if cont_dof is not None:
                cont_dof = cont_dof[indices]

        event_mgr.add_template_events_direct(
            tmpl.params,
            times,
            snrs,
            chisq=chisqs,
            chisq_dof=dofs,
            sigmasq=sigmasqs_arr,
            bank_chisq=b_chisq,
            bank_chisq_dof=b_dof,
            cont_chisq=cont,
            cont_chisq_dof=cont_dof,
            sg_chisq=sg,
        )

    if hasattr(bank, "clear_batch_cache"):
        bank.clear_batch_cache(batch_tnums, collect=False)
    if hasattr(matched_filter, "clear_batch_cache"):
        matched_filter.clear_batch_cache()
