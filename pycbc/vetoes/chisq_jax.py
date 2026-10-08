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

"""JAX backend for chi-square veto primitives and point evaluation."""

import functools
from types import SimpleNamespace

from collections import OrderedDict
import threading

import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types.array_jax import (
    JAXArrayData,
    _cpu_reference,
    _divide,
    _ensure_x64,
    _reference_enabled,
    to_jax,
)


def chisq_accum_bin(chisq, q):
    """Accumulate squared magnitude of q into chisq time series in JAX."""
    _ensure_x64()
    q_arr = to_jax(q)
    if _reference_enabled("chisq_accum_bin"):
        from . import chisq_cpu

        result = np.array(jax.device_get(to_jax(chisq)), copy=True)
        chisq_cpu.chisq_accum_bin(
            SimpleNamespace(data=result),
            SimpleNamespace(data=np.array(jax.device_get(q_arr), copy=True)))
        _write_chisq(chisq, jax.device_put(result, q_arr.device))
        return
    power = q_arr.real ** 2 + q_arr.imag ** 2
    _write_chisq(chisq, to_jax(chisq) + power)


def _write_chisq(chisq, values):
    """Preserve output buffers and surface invalid write targets."""
    if isinstance(chisq, JAXArrayData):
        chisq.set_array(values)
    elif hasattr(chisq, "_data") and isinstance(chisq._data, JAXArrayData):
        chisq._data.set_array(values)
    elif isinstance(chisq, np.ndarray):
        chisq[:] = np.asarray(values)
    else:
        chisq.data[:] = np.asarray(values)


def _native_point_chisq(corr, snr, snr_norm, bins, indices):
    """Execute the original point calculation with its compiled CPU sum."""
    from pycbc.reference_jax import cpu_reference

    if getattr(corr, "_batch_tensor", None) is not None:
        values = np.array(
            jax.device_get(to_jax(corr._batch_tensor)[corr._batch_pos]),
            copy=True,
        )
    else:
        values = np.array(jax.device_get(to_jax(corr)), copy=True)
    base_k = int(getattr(corr, "_kmin", 0))
    n_time = int(getattr(corr, "_tlen", len(values)))
    if base_k or len(values) != n_time:
        full = np.zeros(n_time, dtype=values.dtype)
        full[base_k:base_k + len(values)] = values
        values = full
    return cpu_reference(
        "power_chisq_at_points", values, getattr(corr, "delta_f", 1.0),
        getattr(corr, "_epoch", None), snr=np.asarray(snr), snr_norm=snr_norm,
        bins=np.asarray(bins), indices=np.asarray(indices))


def _compatible_shift_sum(correlations, rows, points, bins, n_time, base_k):
    """Dispatch ordered point sums without changing their numerical mode."""
    if (correlations.dtype == jnp.complex64
            and _use_gpu_ordered_scan(correlations)):
        # Shape-only CUDA geometry keeps small batches spread across SMs.
        # Larger candidate buckets use independent vector lanes per bin.
        lanes, unroll = ((8, 1) if points.size >= 512 else
                         (4, 1) if points.size >= 128 else (1, 8))
        return _compatible_shift_sum_pallas(
            correlations, rows, points, bins, n_time, base_k,
            lanes=lanes, unroll=unroll)
    return _compatible_shift_sum_jax(
        correlations, rows, points, bins, n_time, base_k)


def _chisq_candidate_bucket(count):
    """Use a small set of shapes for sparse CUDA Live candidate batches."""
    return max(4, 1 << (max(1, int(count)) - 1).bit_length())


_LIVE_EXECUTABLE_CACHE_LIMIT = 128
_LIVE_EXECUTABLE_CACHE = OrderedDict()
_LIVE_EXECUTABLE_CACHE_LOCK = threading.RLock()


def _retain_live_executable(cache, key, executable):
    """Retain recently used handles without accumulating compaction shapes."""
    cache.pop(key, None)
    cache[key] = executable
    while len(cache) > _LIVE_EXECUTABLE_CACHE_LIMIT:
        cache.pop(next(iter(cache)))


def _live_chisq_executable(function, args, static_args, cache, device):
    """Retain and reuse a shape-only compiled Live executable on its device.

    Calling ``lower().compile()`` alone need not populate JIT's dispatch cache.
    Keep the compiled callable itself so later calls reuse its loaded handle,
    including after the search object is rebuilt. Both caches are bounded.
    Compilation uses abstract shapes without moving scientific data to the host.
    """
    shapes = tuple((tuple(arg.shape), np.dtype(arg.dtype).str) for arg in args)
    default_sharding = jax.sharding.SingleDeviceSharding(device)
    shardings = tuple(getattr(arg, "sharding", None) or default_sharding
                      for arg in args)
    weak_types = tuple(bool(getattr(arg, "weak_type", False)) for arg in args)
    config = tuple(getattr(jax.config, name, None) for name in (
        "jax_enable_x64", "jax_default_dtype_bits", "jax_dynamic_shapes",
        "jax_default_matmul_precision", "jax_default_dot_algorithm_preset",
        "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion",
        "jax_default_prng_impl", "jax_disable_jit"))
    static = tuple(sorted(static_args.items()))
    key = (function, device, shapes, static, shardings, weak_types, config,
           tuple(type(value) for _, value in static))
    with _LIVE_EXECUTABLE_CACHE_LOCK:
        if key in cache:
            executable = cache[key]
            _retain_live_executable(cache, key, executable)
            _retain_live_executable(_LIVE_EXECUTABLE_CACHE, key, executable)
            return executable
        if key in _LIVE_EXECUTABLE_CACHE:
            executable = _LIVE_EXECUTABLE_CACHE[key]
            _retain_live_executable(_LIVE_EXECUTABLE_CACHE, key, executable)
            _retain_live_executable(cache, key, executable)
            return executable
        from pycbc.benchmark import stage_event

        metadata = dict(kernel=function.__name__, shapes=shapes)
        abstract = tuple(jax.ShapeDtypeStruct(shape, np.dtype(dtype),
                                             sharding=sharding,
                                             weak_type=weak_type)
                         for (shape, dtype), sharding, weak_type
                         in zip(shapes, shardings, weak_types))
        with jax.default_device(device):
            stage_event('filter_veto_executable_lower', 'start',
                        synchronize=False, **metadata)
            try:
                lowered = function.lower(*abstract, **static_args)
            finally:
                stage_event('filter_veto_executable_lower', 'end',
                            synchronize=False, **metadata)
            stage_event('filter_veto_executable_load', 'start',
                        synchronize=False, **metadata)
            try:
                executable = lowered.compile()
            finally:
                stage_event('filter_veto_executable_load', 'end',
                            synchronize=False, **metadata)
        _retain_live_executable(_LIVE_EXECUTABLE_CACHE, key, executable)
        _retain_live_executable(cache, key, executable)
        return executable


def _compatible_phase_state(correlations, rows, points, bins, n_time):
    """Use the reference's real-precision shift conversion and phase seeds."""
    real_dtype = correlations.real.dtype
    edges = bins[rows].astype(jnp.int64)
    starts, ends = edges[:, :-1], edges[:, 1:]
    shifts = points.astype(real_dtype).astype(jnp.float64)[:, None]
    angle = 2 * 3.141592653 * shifts / n_time
    initial = 2 * 3.141592653 * shifts * starts / n_time
    pr, pi = jnp.cos(initial).astype(real_dtype), jnp.sin(initial).astype(
        real_dtype
    )
    rr, ri = jnp.cos(angle).astype(real_dtype), jnp.sin(angle).astype(
        real_dtype
    )
    return starts, ends, pr, pi, rr, ri


@jax.jit
def _compatible_shift_sum_jax(
    correlations, rows, points, bins, n_time, base_k
):
    """Return bin powers on the input device without copying correlations out.

    ``correlations`` and absolute ``bins`` each have a row per template.
    Cropped correlations are zero outside their stored interval, but phase
    evolution starts at the original bin boundary, including omitted zeros.
    Both complex64/float32 and complex128/float64 arithmetic are retained.
    """
    real_dtype = correlations.real.dtype
    # Empty crops and point sets have no power. Avoid tracing a gather from
    # an empty frequency axis or a reduction over an empty point/bin axis.
    if (correlations.shape[1] == 0 or points.size == 0 or
            bins.shape[1] < 2):
        return jnp.zeros(points.shape, real_dtype)
    # Signed indices also preserve negative offsets below a cropped interval.
    starts, ends, pr, pi, rr, ri = _compatible_phase_state(
        correlations, rows, points, bins, n_time)
    zero = jnp.zeros(starts.shape, real_dtype)
    width = correlations.shape[1]

    def step(offset, carry):
        pr, pi, outr, outi = carry
        k = starts + offset
        active = k < ends
        inside = (k >= base_k) & (k < base_k + width)
        value = correlations[rows[:, None], jnp.clip(k - base_k, 0, width - 1)]
        value = jnp.where(inside, value, 0)
        vr, vi = value.real, value.imag
        # Keep the CPU's three-product multiplication and update sequence.
        k1 = vr * (pr + pi)
        k2 = pr * (vi - vr)
        k3 = pi * (vr + vi)
        next_r = pr * rr - pi * ri
        next_i = pr * ri + pi * rr
        return (jnp.where(active, next_r, pr),
                jnp.where(active, next_i, pi),
                jnp.where(active, outr + (k1 - k3), outr),
                jnp.where(active, outi + (k1 + k2), outi))

    # Group dependent steps in one compiled loop body to reduce GPU launch
    # overhead. Each step still consumes the preceding phase and sum state.
    unroll = 16

    def group(index, carry):
        for offset in range(unroll):
            carry = step(index * unroll + offset, carry)
        return carry

    _, _, outr, outi = jax.lax.fori_loop(
        0, (jnp.max(ends - starts) + unroll - 1) // unroll,
        group, (pr, pi, zero, zero))
    powers = outr * outr + outi * outi
    # Summation over bins is ordered, too; a parallel reduction changes
    # rounding.
    return jax.lax.fori_loop(
        0, powers.shape[1], lambda b, total: total + powers[:, b],
        jnp.zeros(points.shape, real_dtype))


@functools.partial(
    jax.jit, static_argnames=("interpret", "lanes", "unroll"))
def _compatible_shift_sum_pallas(
        correlations, rows, points, bins, n_time, base_k, interpret=False,
        *, lanes=1, unroll=1):
    """Run independent candidate lanes in each ordered CUDA bin program.

    Phase seeds and final powers/reduction use the same JAX operations as the
    portable implementation. The Pallas loop retains every intervening phase
    update (including frequencies outside a crop) and the three-product sum;
    it does not replace the recurrence with Fourier phases or a tree reduction.
    ``interpret`` permits focused geometry/oracle checks on CPU, while actual
    CUDA tests qualify the compiled kernel's operation rounding separately.
    Geometry arguments are private tuning controls; ``lanes=1, unroll=1``
    retains the scalar program for same-input qualification and timing.
    """
    _validate_ordered_geometry(lanes, unroll)
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    real_dtype = correlations.real.dtype
    if (correlations.shape[1] == 0 or points.size == 0 or
            bins.shape[1] < 2):
        return jnp.zeros(points.shape, real_dtype)
    starts, ends, pr, pi, rr, ri = _compatible_phase_state(
        correlations, rows, points, bins, n_time)
    width = correlations.shape[1]

    def kernel(vr_ref, vi_ref, row_ref, starts_ref, ends_ref,
               pr_ref, pi_ref, rr_ref, ri_ref, base_ref,
               outr_ref, outi_ref):
        point, bin_index = pl.program_id(0), pl.program_id(1)
        row = row_ref[point]
        start, end = starts_ref[point, bin_index], ends_ref[point, bin_index]
        initial_r = pr_ref[point, bin_index]
        initial_i = pi_ref[point, bin_index]
        rotate_r, rotate_i = rr_ref[point, 0], ri_ref[point, 0]
        base = base_ref[()]

        def step(k, carry):
            phase_r, phase_i, sum_r, sum_i = carry
            inside = (k >= base) & (k < base + width)
            index = jnp.clip(k - base, 0, width - 1)
            vr = jnp.where(inside, vr_ref[row, index], 0)
            vi = jnp.where(inside, vi_ref[row, index], 0)
            k1 = vr * (phase_r + phase_i)
            k2 = phase_r * (vi - vr)
            k3 = phase_i * (vr + vi)
            next_r = phase_r * rotate_r - phase_i * rotate_i
            next_i = phase_r * rotate_i + phase_i * rotate_r
            return next_r, next_i, sum_r + (k1 - k3), sum_i + (k1 + k2)

        zero = jnp.asarray(0, real_dtype)
        initial = (initial_r, initial_i, zero, zero)
        if unroll == 1:
            _, _, sum_r, sum_i = jax.lax.fori_loop(start, end, step, initial)
        else:
            def group(index, carry):
                for offset in range(unroll):
                    carry = step(start + index * unroll + offset, carry)
                return carry

            length = jnp.maximum(end - start, 0)
            groups = length // unroll
            grouped = jax.lax.fori_loop(0, groups, group, initial)
            _, _, sum_r, sum_i = jax.lax.fori_loop(
                start + groups * unroll, end, step, grouped)
        outr_ref[point, bin_index] = sum_r
        outi_ref[point, bin_index] = sum_i

    def vector_kernel(vr_ref, vi_ref, row_ref, starts_ref, ends_ref,
                      pr_ref, pi_ref, rr_ref, ri_ref, base_ref,
                      outr_ref, outi_ref):
        point_block, bin_index = pl.program_id(0), pl.program_id(1)
        points_index = point_block * lanes + jnp.arange(lanes)
        # Clamp only unused tail lanes; valid candidates retain their order.
        point_index = jnp.minimum(points_index, points.size - 1)
        row = row_ref[point_index]
        start = starts_ref[point_index, bin_index]
        end = ends_ref[point_index, bin_index]
        initial_r = pr_ref[point_index, bin_index]
        initial_i = pi_ref[point_index, bin_index]
        rotate_r, rotate_i = rr_ref[point_index, 0], ri_ref[point_index, 0]
        base = base_ref[()]
        zero = jnp.zeros((lanes,), real_dtype)
        length = jnp.maximum(end - start, 0)

        def step(offset, carry):
            phase_r, phase_i, sum_r, sum_i = carry
            k = start + offset
            active = offset < length
            inside = (k >= base) & (k < base + width)
            index = jnp.clip(k - base, 0, width - 1)
            vr = jnp.where(inside, vr_ref[row, index], 0)
            vi = jnp.where(inside, vi_ref[row, index], 0)
            k1 = vr * (phase_r + phase_i)
            k2 = phase_r * (vi - vr)
            k3 = phase_i * (vr + vi)
            next_r = phase_r * rotate_r - phase_i * rotate_i
            next_i = phase_r * rotate_i + phase_i * rotate_r
            return (jnp.where(active, next_r, phase_r),
                    jnp.where(active, next_i, phase_i),
                    jnp.where(active, sum_r + (k1 - k3), sum_r),
                    jnp.where(active, sum_i + (k1 + k2), sum_i))

        def group(index, carry):
            for offset in range(unroll):
                carry = step(index * unroll + offset, carry)
            return carry

        _, _, sum_r, sum_i = jax.lax.fori_loop(
            0, (jnp.max(length) + unroll - 1) // unroll, group,
            (initial_r, initial_i, zero, zero))
        outr_ref[:, 0] = sum_r
        outi_ref[:, 0] = sum_i

    shape = jax.ShapeDtypeStruct(starts.shape, real_dtype)
    scalar = lanes == 1
    out_specs = (pl.no_block_spec, pl.no_block_spec) if scalar else tuple(
        pl.BlockSpec((lanes, 1), lambda block, bin_index: (block, bin_index))
        for _ in range(2))
    outr, outi = pl.pallas_call(
        kernel if scalar else vector_kernel, out_shape=(shape, shape),
        grid=starts.shape if scalar else (
            (points.size + lanes - 1) // lanes, starts.shape[1]),
        out_specs=out_specs,
        compiler_params=pltriton.CompilerParams(num_warps=1),
        interpret=interpret, name="ordered_point_chisq_recurrence",
    )(correlations.real, correlations.imag, rows, starts, ends,
      pr, pi, rr, ri, jnp.asarray(base_k, dtype=jnp.int64))
    powers = outr * outr + outi * outi
    return jax.lax.fori_loop(
        0, powers.shape[1], lambda b, total: total + powers[:, b],
        jnp.zeros(points.shape, real_dtype))


def _chisq_mode():
    from pycbc import scheme
    return getattr(scheme.mgr.state, "jax_chisq_mode", "cpu-compatible")


def _point_chisq_cpu_compatible(arr, shifts, bins, n_time, base_k=0):
    """Use the native reference on CPU devices, including zero-padded crops."""
    from . import chisq_cpu
    arr = np.asarray(arr)
    if base_k != 0 or len(arr) != n_time:
        full = np.zeros(n_time, dtype=arr.dtype)
        full[base_k:base_k + len(arr)] = arr
        arr = full
    shifts = np.asarray(shifts, dtype=arr.real.dtype)
    bins = np.asarray(bins, dtype=np.uint32)
    output = np.zeros(len(shifts), dtype=arr.real.dtype)
    if len(shifts):
        chisq_cpu.point_chisq_code(
            output, arr, len(shifts), n_time, shifts, bins, len(bins) - 1)
    return output


def _require_point_chisq_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX point chi-square requires 64-bit integer phase arithmetic. "
            "Set PYCBC_JAX_ENABLE_X64=1 (the default) or use the CPU scheme."
        )


def _time_shift_phase(n_slice, kmin, points, n_time, dtype):
    ("Form periodic phases without losing bits in large frequency/time "
     "products.")
    _require_point_chisq_x64()
    k = jnp.arange(n_slice, dtype=jnp.int64) + jnp.asarray(kmin, jnp.int64)
    cycles = (k[:, None] * points[None, :].astype(jnp.int64)) % jnp.asarray(
        n_time, jnp.int64
    )
    angle = cycles.astype(jnp.float64) * (2 * jnp.pi / n_time)
    return jnp.exp(1j * angle).astype(dtype)


@jax.jit
def _shift_sum_gpu_core(
    arr_slice, kmin_f, pts_chunk, bin_rel_indices, n_time_f, chunk_size=16
):
    ("JIT-compiled GPU power chisq prefix sum across frequency bins on "
     "corr_slice.")
    n_slice = arr_slice.shape[0]
    phases = _time_shift_phase(
        n_slice, kmin_f, pts_chunk, n_time_f, arr_slice.dtype
    )
    weighted = arr_slice[:, None] * phases
    C = jnp.cumsum(weighted, axis=0)
    C_padded = jnp.pad(C, ((1, 0), (0, 0)))
    edges = jnp.clip(bin_rel_indices, 0, n_slice)
    i0 = edges[:-1]
    i1 = edges[1:]
    zb = C_padded[i1] - C_padded[i0]
    power = jnp.sum(zb.real ** 2 + zb.imag ** 2, axis=0)
    return power


def _shift_sum_gpu_core_row(
    corr_tensor,
    row_idx,
    kmin_f,
    pts_chunk,
    bin_rel_indices,
    n_time_f,
    chunk_size=16,
):
    """GPU power chisq prefix sum indexing row on device."""
    return _shift_sum_gpu_core(
        corr_tensor[row_idx],
        kmin_f,
        pts_chunk,
        bin_rel_indices,
        n_time_f,
        chunk_size=chunk_size,
    )


@jax.jit
def _batched_points_chisq_core(
    corr_tensor,
    row_indices,
    pts,
    bins_rel_all,
    kmin_f,
    n_time_f,
    bucket_size=256,
):
    ("JIT-compiled GPU batched power chisq prefix sum across all triggered "
     """points.

    Calculates time-shifted frequency bin sums for all points across all """
     """templates
    in parallel via vmap over points, executing in a single CUDA kernel.
    """)

    n_slice = corr_tensor.shape[-1]

    def _point_chisq(row, pt, bin_rel):
        phases = _time_shift_phase(
            n_slice, kmin_f, jnp.reshape(pt, (1,)), n_time_f, jnp.complex128
        )[:, 0]
        weighted = row.astype(jnp.complex128) * phases
        C = jnp.cumsum(weighted)
        C_padded = jnp.pad(C, (1, 0))
        edges = jnp.clip(bin_rel, 0, n_slice)
        i0 = edges[:-1]
        i1 = edges[1:]
        zb = C_padded[i1] - C_padded[i0]
        return jnp.sum(zb.real ** 2 + zb.imag ** 2).astype(row.real.dtype)

    rows = corr_tensor[row_indices]
    bins_pts = bins_rel_all[row_indices]
    return jax.vmap(_point_chisq)(rows, pts, bins_pts)


def shift_sum(corr, points, bins):
    """Calculate time-shifted sum of FrequencySeries bins in JAX."""
    _ensure_x64()
    pts = np.asarray(points, dtype=np.int64)
    use_row_indexing = (
        hasattr(corr, "_batch_tensor")
        and hasattr(corr, "_batch_pos")
        and corr._batch_tensor is not None
    )

    if use_row_indexing:
        arr = to_jax(corr._batch_tensor)
    else:
        arr = to_jax(corr)
    if len(pts) == 0:
        return jnp.zeros((0,), dtype=arr.real.dtype)
    compatible = _chisq_mode() == "cpu-compatible"
    # JAXScheme owns the complete veto computation, including its CPU
    # implementation.  Keep the input on the selected JAX device rather than
    # silently routing CPU arrays through the native reference implementation.
    _require_point_chisq_x64()
    if use_row_indexing:
        corr_tensor = to_jax(corr._batch_tensor)
        row_idx = int(corr._batch_pos)
        kmin = int(corr._kmin)
        n_time = int(corr._tlen)
    elif hasattr(corr, "_kmin") and hasattr(corr, "_tlen"):
        kmin = int(corr._kmin)
        n_time = int(corr._tlen)
        arr_slice = arr
    else:
        # Correlations contain the full complex FFT work vector, even
        # when represented by a FrequencySeries.
        n_time = len(arr)
        kmin = int(bins[0])
        kmax = int(bins[-1])
        arr_slice = arr[kmin:kmax]

    if _reference_enabled("shift_sum"):
        row = corr_tensor[row_idx] if use_row_indexing else arr_slice
        result = _point_chisq_cpu_compatible(
            np.array(jax.device_get(row), copy=True),
            pts,
            bins,
            n_time,
            base_k=kmin,
        )
        return jax.device_put(result, arr.device)

    if compatible:
        tensor = corr_tensor if use_row_indexing else arr_slice[None, :]
        row = row_idx if use_row_indexing else 0
        # All points here use the same bin edges. The batch entry below handles
        # different edges per template without materializing correlation rows.
        edges = jnp.broadcast_to(
            jnp.asarray(bins), (tensor.shape[0], len(bins))
        )
        return _compatible_shift_sum(
            tensor, jnp.full(len(pts), row, dtype=jnp.int32),
            jnp.asarray(pts), edges, n_time, kmin)

    bin_indices = jnp.asarray(bins, dtype=jnp.int32)
    bin_rel = bin_indices - kmin
    kmin_f = float(kmin)
    n_time_f = float(n_time)

    pts_np = np.asarray(pts, dtype=np.int32)
    n_pts = len(pts_np)
    out_chunks = []

    offset = 0
    while offset < n_pts:
        rem = n_pts - offset
        if rem == 1:
            sz = 1
        elif rem == 2:
            sz = 2
        elif rem <= 4:
            sz = 4
        elif rem <= 8:
            sz = 8
        else:
            sz = 16
        chunk_slice = pts_np[offset:offset + min(rem, sz)]
        cur_len = len(chunk_slice)
        if cur_len < sz:
            buf = np.zeros(sz, dtype=np.int32)
            buf[:cur_len] = chunk_slice
            pts_j = jax.device_put(buf)
        else:
            pts_j = jax.device_put(chunk_slice)
        if use_row_indexing:
            res_j = _shift_sum_gpu_core_row(
                corr_tensor,
                row_idx,
                kmin_f,
                pts_j,
                bin_rel,
                n_time_f,
                chunk_size=sz,
            )
        else:
            res_j = _shift_sum_gpu_core(
                arr_slice, kmin_f, pts_j, bin_rel, n_time_f, chunk_size=sz
            )
        out_chunks.append(res_j[:cur_len])
        offset += cur_len
    return (
        jnp.concatenate(out_chunks)
        if out_chunks
        else jnp.zeros((0,), arr.real.dtype)
    )


def power_chisq_at_points_from_precomputed(
    corr, snr, snr_norm, bins, indices
):
    """Calculate chisq values at triggered points from precomputed products."""
    _ensure_x64()
    if _reference_enabled("power_chisq_at_points"):
        device = to_jax(
            corr._batch_tensor
            if getattr(corr, "_batch_tensor", None) is not None
            else corr
        ).device
        return jax.device_put(
            _native_point_chisq(corr, snr, snr_norm, bins, indices), device
        )
    num_bins = len(bins) - 1
    chisq = shift_sum(corr, indices, bins)
    # Keep the normalization arithmetic on the selected JAX device.  In
    # particular, converting ``snr`` through NumPy here would synchronize a
    # GPU result and silently move this final veto stage back to the host.
    snr_arr = jnp.asarray(snr)
    # Native NumPy computes the scalar normalization square in float64, then
    # combines it with the float32 chi-square array (NumPy 1.26's scalar
    # promotion keeps that result float32).  Preserve that arithmetic while
    # keeping the operation on the selected JAX device.
    snr_norm_sq = jnp.asarray(snr_norm, dtype=jnp.float64) ** 2.0
    snr_norm_arr = snr_norm_sq.astype(chisq.real.dtype)
    return (
        (chisq * num_bins - (snr_arr.conj() * snr_arr).real)
        * snr_norm_arr
    )


def power_chisq_bins_jax(
    htilde,
    num_bins,
    psd,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
):
    """Calculate standard-backend-compatible bin edges for a JAX template."""
    from pycbc.filter.matchedfilter import get_cutoff_indices

    _ensure_x64()
    if _reference_enabled("power_chisq_bins"):
        from pycbc.reference_jax import cpu_reference

        result = cpu_reference(
            "power_chisq_bins", jax.device_get(to_jax(htilde)), htilde.delta_f,
            getattr(htilde, "_epoch", None), num_bins=num_bins,
            psd=np.asarray(jax.device_get(to_jax(psd))),
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff)
        return jax.device_put(result, to_jax(htilde).device)
    n_pts = (len(htilde) - 1) * 2
    delta_f = float(htilde.delta_f)
    kmin, kmax = get_cutoff_indices(
        low_frequency_cutoff, high_frequency_cutoff, delta_f, n_pts
    )

    from pycbc import scheme
    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    h_arr = to_jax(htilde, device=target_dev)
    psd_arr = to_jax(psd, device=target_dev)

    if _bin_references_selected():
        return _reference_bin_stages(
            h_arr, psd_arr, kmin, kmax, num_bins, delta_f)

    return _power_chisq_bins_numpy(
        h_arr, psd_arr, int(kmin), int(kmax), int(num_bins), delta_f,
        _use_gpu_ordered_scan(h_arr),
    )


_BIN_REFERENCES = ("power_chisq_bins", "squared_norm", "divide", "cumsum")


def _bin_references_selected():
    return any(_reference_enabled(operation) for operation in _BIN_REFERENCES)


def _check_bin_reference_cache(psd):
    """Invalidate cached edges when their numerical route changes."""
    key = tuple(operation for operation in _BIN_REFERENCES
                if _reference_enabled(operation))
    if getattr(psd, "_jax_chisq_reference_key", key) != key:
        psd._chisq_cached_key = {}
    psd._jax_chisq_reference_key = key


def _reference_bin_stages(template, psd, kmin, kmax, num_bins, delta_f):
    values = to_jax(template)[kmin:kmax]
    power = (to_jax(_cpu_reference(values, "squared_norm"))
             if _reference_enabled("squared_norm") else
             values.real ** 2 + values.imag ** 2)
    spectrum = to_jax(psd)[kmin:kmax]
    power = (_divide(power, spectrum, inplace=True)
             if _reference_enabled("divide") else
             _weighted_power_divide(power, spectrum))
    cumulative = (to_jax(_cpu_reference(power, "cumsum"))
                  if _reference_enabled("cumsum") else _ordered_cumsum(power))
    cumulative = cumulative * (4.0 * delta_f)
    edges = jnp.arange(num_bins, dtype=jnp.float64) * cumulative[-1] / num_bins
    bins = jnp.searchsorted(cumulative, edges, side="right") + kmin
    return jnp.concatenate((bins, jnp.asarray([kmax]))).astype(jnp.uint32)


@functools.partial(
    jax.jit,
    static_argnames=("kmin", "kmax", "num_bins", "delta_f", "use_gpu_scan"),
)
def _power_chisq_bins_numpy(
        h_arr, psd_arr, kmin, kmax, num_bins, delta_f, use_gpu_scan=False):
    """Match the standard backend's cumulative power and discrete bin edges.

    A parallel single-precision scan changes bin boundaries in long templates.
    Compute these cached boundaries with the same serial accumulation and
    edge precision as sigmasq_series/power_chisq_bins_from_sigmasq_series.
    """
    h_arr = jnp.asarray(h_arr)
    psd_arr = jnp.asarray(psd_arr)
    h_slice = h_arr[..., kmin:kmax]
    magnitude = h_slice.real ** 2 + h_slice.imag ** 2
    power = _weighted_power_divide(magnitude, psd_arr[kmin:kmax])
    # Match the reference's ordered accumulation.  jnp.cumsum may select a
    # tree scan on some devices, which changes float32 rounding at bin edges.
    cumulative = (_ordered_cumsum_rows(power[None, :], use_gpu_scan)[0]
                  if power.ndim == 1 else
                  _ordered_cumsum_rows(power, use_gpu_scan)) * (4.0 * delta_f)

    def row_bins(row):
        # NumPy's integer arange promotes this threshold calculation to
        # float64 even when the cumulative series is float32.
        edges = jnp.arange(num_bins, dtype=jnp.float64) * row[-1] / num_bins
        bins = jnp.searchsorted(row, edges, side="right") + kmin
        return jnp.concatenate(
            (bins, jnp.asarray([kmax], dtype=bins.dtype))
        ).astype(jnp.uint32)

    if cumulative.ndim == 1:
        return row_bins(cumulative)
    return jax.vmap(row_bins)(cumulative)


@functools.partial(
    jax.jit,
    static_argnames=(
        "min_kmin",
        "kmax",
        "num_bins",
        "delta_f",
        "use_gpu_scan",
    ),
)
def _power_chisq_bins_varied_support(
        h_arr, psd_arr, kmins, min_kmin, kmax, num_bins, delta_f,
        use_gpu_scan=False):
    """Build exact batched edges for rows with different lower cutoffs."""
    h_slice = jnp.asarray(h_arr)[..., min_kmin:kmax]
    psd_slice = jnp.asarray(psd_arr)[min_kmin:kmax]
    magnitude = h_slice.real ** 2 + h_slice.imag ** 2
    power = _weighted_power_divide(magnitude, psd_slice)
    frequencies = jnp.arange(min_kmin, kmax, dtype=jnp.int32)
    power = jnp.where(frequencies[None, :] >= kmins[:, None], power, 0)
    cumulative = _ordered_cumsum_rows(power, use_gpu_scan) * (4.0 * delta_f)

    def row_bins(row):
        edges = jnp.arange(num_bins, dtype=jnp.float64) * row[-1] / num_bins
        bins = jnp.searchsorted(row, edges, side="right") + min_kmin
        return jnp.concatenate(
            (bins, jnp.asarray([kmax], dtype=bins.dtype))
        ).astype(jnp.uint32)

    return jax.vmap(row_bins)(cumulative)


def _weighted_power_divide(magnitude, psd):
    """Divide template power by PSD with the reference float32 rounding.

    The native reference uses ordinary float32 division.  Widening the
    operands and using an explicit float64 reciprocal/multiply reproduces its
    rounding on the supported JAX devices, avoiding device-specific quotient
    rounding from moving a cached chi-square bin edge.  Double-precision
    inputs retain their native operation and dtype.
    """
    magnitude = jnp.asarray(magnitude)
    psd = jnp.asarray(psd)
    if magnitude.dtype == jnp.float32:
        magnitude64 = magnitude.astype(jnp.float64)
        psd_reciprocal64 = jnp.reciprocal(psd.astype(jnp.float64))
        return (magnitude64 * psd_reciprocal64).astype(jnp.float32)
    return magnitude / psd


@jax.jit
def _ordered_cumsum(values):
    """Serial cumulative sum with the same order as NumPy's reference."""
    def step(carry, value):
        carry = carry + value
        return carry, carry
    _, result = jax.lax.scan(step, jnp.zeros((), values.dtype), values)
    return result


def _use_gpu_ordered_scan(array):
    """Use one-kernel ordered accumulation on supported NVIDIA devices."""
    device = getattr(array, "device", None)
    if getattr(device, "platform", None) not in ("cuda", "gpu"):
        return False
    try:
        capability = float(device.compute_capability)
        from jax.experimental.pallas import triton  # noqa: F401
    except (AttributeError, ImportError, TypeError, ValueError):
        return False
    return capability >= 8.0


def _validate_ordered_geometry(lanes, unroll):
    """Bound private geometry controls used for baseline qualification."""
    if lanes not in (1, 4, 8):
        raise ValueError("ordered kernel lanes must be 1, 4, or 8")
    if unroll not in (1, 8, 16):
        raise ValueError("ordered kernel unroll must be 1, 8, or 16")


@functools.partial(jax.jit, static_argnames=("unroll", "interpret"))
def _ordered_cumsum_rows_pallas(values, *, unroll=1, interpret=False):
    """Retain one program per row and explicitly unroll ordered additions.

    ``unroll=1`` is the established scalar implementation for qualification
    and timing on identical inputs. Every frequency addition and cumulative
    output remains sequential; unrolling reduces loop overhead, not accuracy.
    """
    _validate_ordered_geometry(1, unroll)
    if values.shape[0] == 0 or values.shape[1] == 0:
        return jnp.zeros_like(values)
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    rows, width = values.shape
    block_width = 1 << (width - 1).bit_length()

    def kernel(source, result):
        def step(index, carry):
            carry = carry + source[0, index]
            result[0, index] = carry
            return carry

        initial = jnp.zeros((), values.dtype)
        if unroll == 1:
            jax.lax.fori_loop(0, width, step, initial)
        else:
            def group(index, carry):
                for offset in range(unroll):
                    carry = step(index * unroll + offset, carry)
                return carry

            carry = jax.lax.fori_loop(0, width // unroll, group, initial)
            # The static tail consumes each valid frequency once, with no
            # masks or the unsupported Triton fori_loop unroll parameter.
            for offset in range(width % unroll):
                carry = step(width - width % unroll + offset, carry)

    block = pl.BlockSpec((1, block_width), lambda row: (row, 0))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(values.shape, values.dtype),
        grid=(rows,), in_specs=(block,), out_specs=block,
        compiler_params=pltriton.CompilerParams(num_warps=4),
        interpret=interpret, name="ordered_power_chisq_cumsum",
    )(values)


def _ordered_cumsum_rows(values, use_gpu_scan):
    """Preserve reference addition order without one GPU launch per sample."""
    if not use_gpu_scan or values.dtype != jnp.float32:
        return jax.vmap(_ordered_cumsum)(values)
    return _ordered_cumsum_rows_pallas(values, unroll=16)


@functools.partial(
    jax.jit,
    static_argnames=("kmax", "num_bins", "delta_f", "use_gpu_scan"),
)
def _power_chisq_bins_from_source(
        h_arr, psd_arr, positions, kmins, *, kmax, num_bins, delta_f,
        use_gpu_scan=False):
    """Gather bucketed Live rows and construct their exact cached bin edges.

    The original frequency grid is retained. A fixed zero prefix makes the
    compiled shape independent of which lower cutoffs survived selection;
    adding these zeros does not change the ordered power accumulation.
    """
    return _power_chisq_bins_varied_support(
        h_arr[positions], psd_arr, kmins, 0, kmax, num_bins, delta_f,
        use_gpu_scan)


def batch_power_chisq_bins_jax(
    templates,
    num_bins,
    psd,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
):
    """Calculate standard-backend-compatible bin edges for JAX template rows.

    ``low_frequency_cutoff`` may be one value for the whole batch or one
    value per template. The latter keeps each ordered frequency scan exact
    while allowing independent templates to execute concurrently.
    """
    from pycbc.filter.matchedfilter import get_cutoff_indices

    _ensure_x64()
    if _bin_references_selected():
        cutoffs = (
            [low_frequency_cutoff] * len(templates)
            if low_frequency_cutoff is None
            or np.isscalar(low_frequency_cutoff)
            else list(low_frequency_cutoff)
        )
        if len(cutoffs) != len(templates):
            raise ValueError("Expected one lower cutoff per template")
        return jnp.stack([power_chisq_bins_jax(
            template, num_bins, psd, cutoff, high_frequency_cutoff)
            for template, cutoff in zip(templates, cutoffs)])
    n_pts = (len(templates[0]) - 1) * 2
    delta_f = float(templates[0].delta_f)
    varied_support = (
        low_frequency_cutoff is not None
        and not np.isscalar(low_frequency_cutoff)
    )
    if varied_support:
        cutoffs = list(low_frequency_cutoff)
        if len(cutoffs) != len(templates):
            raise ValueError("Expected one lower cutoff per template")
        bounds = [
            get_cutoff_indices(
                cutoff, high_frequency_cutoff, delta_f, n_pts
            )
            for cutoff in cutoffs
        ]
        kmins = np.asarray([bound[0] for bound in bounds], dtype=np.int32)
        kmaxs = {bound[1] for bound in bounds}
        if len(kmaxs) != 1:
            raise ValueError("Batched templates must share an upper cutoff")
        kmax = kmaxs.pop()
        min_kmin = int(kmins.min())
    else:
        kmin, kmax = get_cutoff_indices(
            low_frequency_cutoff, high_frequency_cutoff, delta_f, n_pts
        )

    from pycbc import scheme
    state = getattr(scheme.mgr, "state", None)
    target_dev = getattr(state, "jax_device", None)

    batch_tensor = getattr(templates, "_batch_tensor", None)
    if batch_tensor is None:
        tmpls_tensor = jnp.stack(
            [to_jax(t, device=target_dev) for t in templates], axis=0
        )
    else:
        tmpls_tensor = to_jax(batch_tensor, device=target_dev)

    psd_arr = to_jax(psd, device=target_dev)

    if varied_support:
        return _power_chisq_bins_varied_support(
            tmpls_tensor,
            psd_arr,
            jnp.asarray(kmins),
            min_kmin,
            int(kmax),
            int(num_bins),
            delta_f,
            _use_gpu_ordered_scan(tmpls_tensor),
        )
    return _power_chisq_bins_numpy(
        tmpls_tensor, psd_arr, int(kmin), int(kmax), int(num_bins), delta_f,
        _use_gpu_ordered_scan(tmpls_tensor),
    )


def _publish_cached_power_chisq_bins_jax(edges, templates, psd):
    """Publish completed bin rows with the established identity cache keys."""
    psd_id = id(psd)
    for row, template in enumerate(templates):
        psd._chisq_cached_key[id(template.params)] = True
        template._bin_cache[psd_id] = edges[row]


def _collect_cached_power_chisq_bins_jax(pending):
    """Collect queued device bin groups once and publish them in launch order."""
    if not pending:
        return
    groups = tuple(pending)
    pending.clear()
    try:
        collected = jax.device_get(tuple(item[0] for item in groups))
    except Exception:
        # Preserve earlier successful cache writes if a later device group
        # fails, as the former sequential collection would have done.
        for result, templates, psd in groups:
            edges = np.asarray(jax.device_get(result))[:len(templates)]
            _publish_cached_power_chisq_bins_jax(edges, templates, psd)
        raise
    for result, (_, templates, psd) in zip(collected, groups):
        edges = np.asarray(result)[:len(templates)]
        _publish_cached_power_chisq_bins_jax(edges, templates, psd)








def cache_batch_power_chisq_bins_jax(power_chisq, templates, psd, *,
                                    pending=None):
    """Populate canonical chi-square bin caches in exact batched groups.

    Templates with different support or bin counts remain separate. Analytic
    PSD cumulative vectors retain the native scalar shortcut; all other rows
    use the ordered JAX scan already used by :func:`power_chisq_bins_jax`.

    An optional pending list lets native CUDA Live source groups launch before
    a shared collection. Deferred calls return ``None``; callers collect the
    list with :func:`_collect_cached_power_chisq_bins_jax`. Other paths retain
    synchronous cache publication and the existing returned bin list.
    """
    from collections import defaultdict
    from pycbc.opt import LimitedSizeDict
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    from pycbc.waveform.bank_jax import TemplateBatchList

    _check_bin_reference_cache(psd)
    native_stages = _bin_references_selected()
    source_tensor = getattr(templates, "_batch_tensor", None)
    source_positions = getattr(templates, "_batch_positions", None)
    queued_templates = {
        id(template) for _, group, _ in (pending or ()) for template in group
    }
    if any(id(template) in queued_templates for template in templates):
        # Repeated identities must observe prior publication and bounded-cache
        # eviction before this group decides whether its entries are valid.
        _collect_cached_power_chisq_bins_jax(pending)
    defer = (
        pending is not None
        and not native_stages
        and type(power_chisq) is SingleDetPowerChisq
        and power_chisq.parse_option is SingleDetPowerChisq.parse_option
        and getattr(power_chisq.cached_chisq_bins, "__func__", None)
        is SingleDetPowerChisq.cached_chisq_bins
        and source_positions is not None
        and source_tensor is not None
        and getattr(getattr(source_tensor, "device", None), "platform", None)
        in ("cuda", "gpu")
        and not any(
            getattr(template, "approximant", None)
            in getattr(psd, "sigmasq_vec", {}) for template in templates)
    )
    if pending is not None and not defer:
        _collect_cached_power_chisq_bins_jax(pending)

    psd_id = id(psd)
    if not hasattr(psd, "_chisq_cached_key"):
        psd._chisq_cached_key = {}

    groups = defaultdict(list)
    for index, template in enumerate(templates):
        if not hasattr(template, "_bin_cache"):
            template._bin_cache = LimitedSizeDict(size_limit=2**2)
        cache_valid = (
            psd_id in template._bin_cache
            and id(template.params) in psd._chisq_cached_key
        )
        if cache_valid:
            continue

        # Retain the established analytic cumulative-vector shortcut.
        if (
            hasattr(psd, "sigmasq_vec")
            and getattr(template, "approximant", None) in psd.sigmasq_vec
        ):
            power_chisq.cached_chisq_bins(template, psd)
            continue

        num_bins = int(power_chisq.parse_option(template, power_chisq.num_bins))
        key = (num_bins, len(template), float(template.delta_f))
        groups[key].append(index)

    for (num_bins, _, _), indices in groups.items():
        grouped = TemplateBatchList([templates[index] for index in indices])
        f_lowers = [getattr(template, "f_lower", None) for template in grouped]
        if (source_positions is not None and source_tensor is not None
                and not native_stages):
            from pycbc.benchmark import stage_event
            from pycbc.filter.matchedfilter import get_cutoff_indices

            stage_event('filter_veto_bins_prepare', 'start',
                        synchronize=False, templates=len(indices))
            delta_f = float(grouped[0].delta_f)
            n_time = (len(grouped[0]) - 1) * 2
            bounds = [get_cutoff_indices(flow, None, delta_f, n_time)
                      for flow in f_lowers]
            bucket = _chisq_candidate_bucket(len(indices))
            positions = np.zeros(bucket, dtype=np.int32)
            kmins = np.zeros(bucket, dtype=np.int32)
            positions[:len(indices)] = [source_positions[i] for i in indices]
            kmins[:len(indices)] = [bound[0] for bound in bounds]
            source = to_jax(source_tensor)
            args = (source, to_jax(psd), positions, kmins)
            static_args = dict(
                kmax=bounds[0][1], num_bins=num_bins, delta_f=delta_f,
                use_gpu_scan=_use_gpu_ordered_scan(source))
            cache = getattr(power_chisq, "_jax_live_veto_executables", None)
            if cache is None:
                cache = power_chisq._jax_live_veto_executables = {}
            stage_event('filter_veto_bins_prepare', 'end',
                        synchronize=False, templates=len(indices),
                        bucket=bucket)
            execute = _live_chisq_executable(
                _power_chisq_bins_from_source, args, static_args, cache,
                source.device)
            stage_event('filter_veto_bins_dispatch', 'start',
                        synchronize=False, templates=len(indices),
                        bucket=bucket)
            try:
                result = execute(*args)
            finally:
                stage_event('filter_veto_bins_dispatch', 'end',
                            synchronize=False, templates=len(indices),
                            bucket=bucket)
            if defer:
                pending.append((result, grouped, psd))
            else:
                edges = np.asarray(jax.device_get(result))[:len(indices)]
                _publish_cached_power_chisq_bins_jax(edges, grouped, psd)
            continue
        if source_tensor is not None:
            if indices == list(range(len(templates))):
                grouped._batch_tensor = source_tensor
            else:
                grouped._batch_tensor = to_jax(source_tensor)[
                    jnp.asarray(indices, dtype=jnp.int32)
                ]
        edges = np.asarray(jax.device_get(batch_power_chisq_bins_jax(
            grouped, num_bins, psd, f_lowers)))
        _publish_cached_power_chisq_bins_jax(edges, grouped, psd)

    if defer:
        return None
    return [template._bin_cache[psd_id] for template in templates]


def batch_power_chisq_jax(
    corr_tensor,
    batch_results,
    active_templates,
    psd,
    analyze_start,
    snr_threshold=None,
    power_chisq=None,
):
    ("Batched evaluation of power chisq across all triggered templates in a "
     """segment.

    Computes time-shifted frequency bin sums for triggered points across
    templates using the selected JAX chi-square mode.

    Returns:
        chisq_map: dict mapping act_pos -> (chisq_out, chisq_dof). These
        are NumPy arrays for event handling on both CPU and CUDA: chi-square
        retains the correlation's real dtype, and dof is an integer array.
        CUDA correlations remain on-device through the selected point kernel;
        only the resulting point powers are materialized on the host. Scalar
        ``shift_sum`` and precomputed point primitives instead return JAX """
     """arrays.
    """)
    _ensure_x64()
    if corr_tensor is None or len(batch_results) == 0:
        return None
    corr_dev = to_jax(corr_tensor)
    real_dtype = corr_dev.real.dtype
    compatible = _chisq_mode() == "cpu-compatible"

    psd_id = id(psd)
    b = len(active_templates)

    # 1. Gather bin edges for all active templates
    bins_list = []
    cached_key = getattr(psd, "_chisq_cached_key", None)
    for tmpl in active_templates:
        b_edges = None
        if (
            hasattr(tmpl, "_bin_cache")
            and psd_id in tmpl._bin_cache
            and cached_key is not None
            and id(tmpl.params) in cached_key
        ):
            b_edges = tmpl._bin_cache[psd_id]
        elif power_chisq is not None and hasattr(
            power_chisq, "cached_chisq_bins"
        ):
            b_edges = power_chisq.cached_chisq_bins(tmpl, psd)
        elif hasattr(tmpl, "_bin_cache") and psd_id in tmpl._bin_cache:
            b_edges = tmpl._bin_cache[psd_id]
        if b_edges is None:
            return None
        bins_list.append(b_edges)

    # Different bin counts require the caller's per-template scalar fallback.
    num_bins_set = set(len(b_edges) - 1 for b_edges in bins_list)
    if len(num_bins_set) != 1:
        return None

    num_bins = len(bins_list[0]) - 1
    dof = num_bins * 2 - 2

    # Retrieve correlation metadata
    corr_sample = None
    for res in batch_results:
        if res[2] is not None and hasattr(res[2], "_kmin"):
            corr_sample = res[2]
            break
    if corr_sample is None:
        return None

    kmin = int(corr_sample._kmin)
    tlen = int(corr_sample._tlen)
    kmin_f = float(kmin)
    n_time_f = float(tlen)

    # 2. Gather active trigger points across all templates
    chisq_map = {}
    points_to_compute = []
    total_points = 0

    empty_real = np.zeros(0, dtype=real_dtype)
    empty_dof = np.empty(0, dtype=np.int32)

    for act_pos in range(b):
        snr, norm, corr, idx, snrv = batch_results[act_pos]
        n_idx = len(idx)
        if n_idx == 0:
            chisq_map[act_pos] = (empty_real, empty_dof)
            continue

        if snr_threshold is not None:
            above = abs(snrv * norm) > snr_threshold
            n_above = int(np.sum(above))
        else:
            above = np.ones(n_idx, dtype=bool)
            n_above = n_idx

        if n_above == 0:
            chisq_map[act_pos] = (
                np.zeros(n_idx, dtype=real_dtype),
                np.repeat(dof, n_idx),
            )
            continue

        above_indices = (idx[above] + analyze_start).astype(np.int32)
        above_snrv = snrv[above]
        points_to_compute.append({
            "act_pos": act_pos,
            "above_indices": above_indices,
            "above_snrv": above_snrv,
            "norm": float(norm),
            "above_mask": above,
            "n_idx": n_idx,
            "n_above": n_above,
        })
        total_points += n_above

    if total_points == 0:
        return chisq_map

    if _reference_enabled("power_chisq_at_points"):
        for item in points_to_compute:
            act_pos = item["act_pos"]
            corr = SimpleNamespace(_batch_tensor=corr_dev, _batch_pos=act_pos,
                                   _kmin=kmin, _tlen=tlen)
            values = _native_point_chisq(
                corr, item["above_snrv"], item["norm"], bins_list[act_pos],
                item["above_indices"])
            output = np.zeros(item["n_idx"], dtype=real_dtype)
            output[item["above_mask"]] = values
            chisq_map[act_pos] = (output, np.repeat(dof, item["n_idx"]))
        return chisq_map

    # 3. Stack bins array for active templates and build flattened point arrays
    all_rows = np.empty(total_points, dtype=np.int32)
    all_pts = np.empty(total_points, dtype=np.int32)

    offset = 0
    for item in points_to_compute:
        n_ab = item["n_above"]
        all_rows[offset:offset + n_ab] = item["act_pos"]
        all_pts[offset:offset + n_ab] = item["above_indices"]
        offset += n_ab

    _require_point_chisq_x64()
    bins_rel_all = getattr(active_templates, "_cached_bins_rel", None)
    cached_kmin = getattr(active_templates, "_cached_bins_kmin", None)
    cached_psd_id = getattr(active_templates, "_cached_bins_psd_id", None)
    if (bins_rel_all is None or cached_kmin != kmin or
            cached_psd_id != psd_id):
        bins_arr = np.array(bins_list, dtype=np.int32)
        bins_rel_all = jax.device_put(bins_arr - kmin)
        try:
            active_templates._cached_bins_rel = bins_rel_all
            active_templates._cached_bins_kmin = kmin
            active_templates._cached_bins_psd_id = psd_id
            active_templates._cached_bins_arr = bins_arr
        except Exception:
            bins_arr = np.array(bins_list, dtype=np.int32)
    else:
        bins_arr = getattr(active_templates, "_cached_bins_arr", None)
        if bins_arr is None:
            bins_arr = np.array(bins_list, dtype=np.int32)
    # 4. Bucketed JAX execution. Padding keeps the compiled shapes stable,
    # while slicing remains a device operation and does not materialize the
    # correlation or veto values on the host.
    bucket_sizes = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    chosen_bucket = next(
        (bsz for bsz in bucket_sizes if total_points <= bsz), None
    )
    if chosen_bucket is None:
        chosen_bucket = int(2 ** np.ceil(np.log2(total_points)))
    r_buf = np.zeros(chosen_bucket, dtype=np.int32)
    p_buf = np.zeros(chosen_bucket, dtype=np.int32)
    r_buf[:total_points] = all_rows
    p_buf[:total_points] = all_pts
    r_dev = jax.device_put(r_buf)
    p_dev = jax.device_put(p_buf)

    if _reference_enabled("shift_sum"):
        all_shift_sums = np.concatenate(
            [
                _point_chisq_cpu_compatible(
                    np.array(
                        jax.device_get(corr_dev[item["act_pos"]]), copy=True
                    ),
                    item["above_indices"],
                    bins_arr[item["act_pos"]],
                    tlen,
                    base_k=kmin,
                )
                for item in points_to_compute
            ]
        )
    elif compatible:
        res_dev = _compatible_shift_sum(
            corr_dev, r_dev, p_dev, jnp.asarray(bins_arr), tlen, kmin)
        all_shift_sums = np.asarray(res_dev[:total_points])
    else:
        res_dev = _batched_points_chisq_core(
            corr_dev, r_dev, p_dev, bins_rel_all, kmin_f, n_time_f,
            bucket_size=chosen_bucket)
        all_shift_sums = np.asarray(res_dev[:total_points])

    # 5. Distribute results back to each template
    offset = 0
    for item in points_to_compute:
        n_ab = item["n_above"]
        act_pos = item["act_pos"]
        n_idx = item["n_idx"]
        norm = item["norm"]
        above_mask = item["above_mask"]
        above_snrv = item["above_snrv"]

        shifts = all_shift_sums[offset:offset + n_ab]
        offset += n_ab

        above_snrv_np = np.asarray(above_snrv)
        chisq_vals = (
            shifts * num_bins - (above_snrv_np.conj() * above_snrv_np).real
        ) * (norm**2.0)
        chisq_out = np.zeros(n_idx, dtype=real_dtype)
        chisq_out[above_mask] = chisq_vals.astype(real_dtype)
        chisq_dof = np.repeat(dof, n_idx)

        chisq_map[act_pos] = (chisq_out, chisq_dof)

    return chisq_map
