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
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from pycbc.types.array_jax import (
    JAXArrayData,
    _ensure_x64,
    to_jax,
)


def chisq_accum_bin(chisq, q):
    """Accumulate squared magnitude of q into chisq time series in JAX."""
    _ensure_x64()
    q_arr = to_jax(q)
    power = q_arr.real ** 2 + q_arr.imag ** 2
    if isinstance(chisq, JAXArrayData):
        chisq.set_array(chisq.array + power)
    elif hasattr(chisq, "_data") and isinstance(chisq._data, JAXArrayData):
        chisq._data.set_array(chisq._data.array + power)
    elif hasattr(chisq, "data") and isinstance(chisq.data, JAXArrayData):
        chisq.data.set_array(chisq.data.array + power)
    else:
        try:
            chisq.data[:] += np.asarray(power)
        except Exception:
            chisq[:] += np.asarray(power)


from .chisq_numpy import shift_sum as _point_chisq_numpy


@jax.jit
def _compatible_shift_sum(correlations, rows, points, bins, n_time, base_k):
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
    edges = bins[rows].astype(jnp.int64)
    starts, ends = edges[:, :-1], edges[:, 1:]
    # The CPU first converts shifts to the correlation's real precision.
    shifts = points.astype(real_dtype).astype(jnp.float64)[:, None]
    angle = 2 * 3.141592653 * shifts / n_time
    initial = 2 * 3.141592653 * shifts * starts / n_time
    pr, pi = jnp.cos(initial).astype(real_dtype), jnp.sin(initial).astype(real_dtype)
    rr, ri = jnp.cos(angle).astype(real_dtype), jnp.sin(angle).astype(real_dtype)
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
    # Summation over bins is ordered, too; a parallel reduction changes rounding.
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


def _array_is_cpu(arr):
    """Return whether an array is on CPU (including ordinary NumPy arrays)."""
    dev = getattr(arr, "device", None)
    if callable(dev):
        dev = dev()
    if dev is None:
        buffer = getattr(arr, "device_buffer", None)
        if buffer is not None:
            dev = getattr(buffer, "device", None)
            if callable(dev):
                dev = dev()
    platform = getattr(dev, "platform", None)
    return platform in (None, "cpu")


def _point_chisq_cpu(arr, shifts, bins, n_time, base_k=0):
    """Evaluate full or cropped CPU correlations with accurate Fourier sums."""
    return _point_chisq_numpy(arr, shifts, bins, n_time, base_k=base_k)


def _require_point_chisq_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX point chi-square requires 64-bit integer phase arithmetic. "
            "Set PYCBC_JAX_ENABLE_X64=1 (the default) or use the CPU scheme."
        )


def _time_shift_phase(n_slice, kmin, points, n_time, dtype):
    """Form periodic phases without losing bits in large frequency/time products."""
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
    """JIT-compiled GPU power chisq prefix sum across frequency bins on corr_slice."""
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
    corr_tensor, row_idx, kmin_f, pts_chunk, bin_rel_indices, n_time_f, chunk_size=16
):
    """GPU power chisq prefix sum indexing row on device."""
    return _shift_sum_gpu_core(
        corr_tensor[row_idx], kmin_f, pts_chunk, bin_rel_indices, n_time_f, chunk_size=chunk_size
    )


@jax.jit
def _batched_points_chisq_core(
    corr_tensor, row_indices, pts, bins_rel_all, kmin_f, n_time_f, bucket_size=256
):
    """JIT-compiled GPU batched power chisq prefix sum across all triggered points.

    Calculates time-shifted frequency bin sums for all points across all templates
    in parallel via vmap over points, executing in a single CUDA kernel.
    """
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
        return (jnp.sum(zb.real ** 2 + zb.imag ** 2)).astype(jnp.float32)

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

    if compatible:
        tensor = corr_tensor if use_row_indexing else arr_slice[None, :]
        row = row_idx if use_row_indexing else 0
        # All points here use the same bin edges. The batch entry below handles
        # different edges per template without materializing correlation rows.
        edges = jnp.broadcast_to(jnp.asarray(bins), (tensor.shape[0], len(bins)))
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
                corr_tensor, row_idx, kmin_f, pts_j, bin_rel, n_time_f, chunk_size=sz
            )
        else:
            res_j = _shift_sum_gpu_core(
                arr_slice, kmin_f, pts_j, bin_rel, n_time_f, chunk_size=sz
            )
        out_chunks.append(res_j[:cur_len])
        offset += cur_len
    return jnp.concatenate(out_chunks) if out_chunks else jnp.zeros((0,), arr.real.dtype)


def power_chisq_at_points_from_precomputed(
    corr, snr, snr_norm, bins, indices
):
    """Calculate chisq values at triggered points from precomputed products."""
    _ensure_x64()
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

    return _power_chisq_bins_numpy(
        h_arr, psd_arr, int(kmin), int(kmax), int(num_bins), delta_f,
        _use_gpu_ordered_scan(h_arr),
    )


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
        return jnp.concatenate((bins, jnp.asarray([kmax], dtype=bins.dtype))).astype(jnp.uint32)

    if cumulative.ndim == 1:
        return row_bins(cumulative)
    return jax.vmap(row_bins)(cumulative)


@functools.partial(
    jax.jit,
    static_argnames=("min_kmin", "kmax", "num_bins", "delta_f", "use_gpu_scan"),
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


def _ordered_cumsum_rows(values, use_gpu_scan):
    """Preserve reference addition order without one GPU launch per sample."""
    if not use_gpu_scan or values.dtype != jnp.float32:
        return jax.vmap(_ordered_cumsum)(values)

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    rows, width = values.shape
    block_width = 1 << (width - 1).bit_length()

    def kernel(source, result):
        def step(index, carry):
            carry = carry + source[0, index]
            result[0, index] = carry
            return carry

        jax.lax.fori_loop(0, width, step, jnp.float32(0))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(values.shape, values.dtype),
        grid=(rows,),
        in_specs=(pl.BlockSpec((1, block_width), lambda row: (row, 0)),),
        out_specs=pl.BlockSpec((1, block_width), lambda row: (row, 0)),
        compiler_params=pltriton.CompilerParams(num_warps=4),
        name="ordered_power_chisq_cumsum",
    )(values)


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
        tmpls_tensor = jnp.stack([to_jax(t, device=target_dev) for t in templates], axis=0)
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


def _batch_power_chisq_bins_host(
    templates, host_tensor, num_bins, psd, low_frequency_cutoff=None,
    high_frequency_cutoff=None,
):
    """Build exact bin edges from an existing native template batch.

    The reference calculation is an ordered NumPy cumulative sum. Compressed
    banks already pass through this host representation before one device
    transfer, so retaining it for the cache avoids a serial accelerator scan
    and a device-to-host copy without changing arithmetic.
    """
    from pycbc.filter.matchedfilter import get_cutoff_indices

    n_pts = (len(templates[0]) - 1) * 2
    delta_f = float(templates[0].delta_f)
    varied_support = (
        low_frequency_cutoff is not None
        and not np.isscalar(low_frequency_cutoff)
    )
    cutoffs = (list(low_frequency_cutoff) if varied_support else
               [low_frequency_cutoff] * len(templates))
    if len(cutoffs) != len(templates):
        raise ValueError("Expected one lower cutoff per template")
    bounds = [
        get_cutoff_indices(cutoff, high_frequency_cutoff, delta_f, n_pts)
        for cutoff in cutoffs
    ]
    kmins = np.asarray([bound[0] for bound in bounds], dtype=np.int32)
    kmaxs = {bound[1] for bound in bounds}
    if len(kmaxs) != 1:
        raise ValueError("Batched templates must share an upper cutoff")
    kmax = kmaxs.pop()
    min_kmin = int(kmins.min())
    rows = np.asarray(host_tensor)[:, min_kmin:kmax]
    if hasattr(psd, "_host_data") and psd._host_data is not None:
        psd_host_full = psd._host_data
    else:
        psd_host_full = np.asarray(jax.device_get(to_jax(psd)))
        try:
            psd._host_data = psd_host_full
        except Exception:
            pass
    psd_host = psd_host_full[min_kmin:kmax]
    cumulative = np.empty(rows.shape, dtype=rows.real.dtype)
    np.square(rows.real, out=cumulative)
    cumulative += np.square(rows.imag)
    cumulative /= psd_host[None, :]
    cumulative[np.arange(min_kmin, kmax)[None, :] < kmins[:, None]] = 0
    np.cumsum(cumulative, axis=-1, out=cumulative)
    cumulative *= 4.0 * delta_f
    bins = np.empty((len(rows), num_bins + 1), dtype=np.uint32)
    for index, row in enumerate(cumulative):
        edges = np.arange(num_bins) * row[-1] / num_bins
        bins[index, :-1] = (
            np.searchsorted(row, edges, side="right") + min_kmin
        )
        bins[index, -1] = kmax
    return bins


def cache_batch_power_chisq_bins_jax(power_chisq, templates, psd):
    """Populate canonical chi-square bin caches in exact batched groups.

    Templates with different support or bin counts remain separate. Analytic
    PSD cumulative vectors retain the native scalar shortcut; all other rows
    use the ordered JAX scan already used by :func:`power_chisq_bins_jax`.
    """
    from collections import defaultdict
    from pycbc.opt import LimitedSizeDict
    from pycbc.waveform.bank import TemplateBatchList

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

    source_tensor = getattr(templates, "_batch_tensor", None)
    source_host = getattr(templates, "_host_batch_tensor", None)
    for (num_bins, _, _), indices in groups.items():
        grouped = TemplateBatchList([templates[index] for index in indices])
        f_lowers = [getattr(template, "f_lower", None) for template in grouped]
        if source_tensor is not None:
            if indices == list(range(len(templates))):
                grouped._batch_tensor = source_tensor
            else:
                grouped._batch_tensor = to_jax(source_tensor)[
                    jnp.asarray(indices, dtype=jnp.int32)
                ]
        if source_host is not None:
            if indices == list(range(len(templates))):
                grouped_host = source_host
            else:
                grouped_host = source_host[indices]
            edges = _batch_power_chisq_bins_host(
                grouped, grouped_host, num_bins, psd, f_lowers
            )
        else:
            edges = np.asarray(jax.device_get(batch_power_chisq_bins_jax(
                grouped, num_bins, psd, f_lowers
            )))
        for row, index in enumerate(indices):
            template = templates[index]
            psd._chisq_cached_key[id(template.params)] = True
            template._bin_cache[psd_id] = edges[row]

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
    """Batched evaluation of power chisq across all triggered templates in a segment.

    Computes time-shifted frequency bin sums for triggered points across
    templates using the selected JAX chi-square mode.

    Returns:
        chisq_map: dict mapping act_pos -> (chisq_out, chisq_dof)
    """
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
        elif power_chisq is not None and hasattr(power_chisq, "cached_chisq_bins"):
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
    if bins_rel_all is None or cached_kmin != kmin:
        bins_arr = np.array(bins_list, dtype=np.int32)
        bins_rel_all = jax.device_put(bins_arr - kmin)
        try:
            active_templates._cached_bins_rel = bins_rel_all
            active_templates._cached_bins_kmin = kmin
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
    chosen_bucket = next((bsz for bsz in bucket_sizes if total_points <= bsz), None)
    if chosen_bucket is None:
        chosen_bucket = int(2 ** np.ceil(np.log2(total_points)))
    r_buf = np.zeros(chosen_bucket, dtype=np.int32)
    p_buf = np.zeros(chosen_bucket, dtype=np.int32)
    r_buf[:total_points] = all_rows
    p_buf[:total_points] = all_pts
    r_dev = jax.device_put(r_buf)
    p_dev = jax.device_put(p_buf)
    dev = getattr(corr_dev, "device", None)
    if callable(dev):
        dev = dev()
    platform = getattr(dev, "platform", "cpu")
    is_cuda = platform in ("cuda", "gpu")

    if is_cuda:
        res_dev = _batched_points_chisq_core(
            corr_dev, r_dev, p_dev, bins_rel_all, kmin_f, n_time_f,
            bucket_size=chosen_bucket)
        all_shift_sums = np.asarray(res_dev[:total_points])
    elif compatible:
        try:
            from .chisq_cpu import point_chisq_code
            have_chisq_cpu = bool(point_chisq_code)
        except ImportError:
            have_chisq_cpu = False

        if have_chisq_cpu:
            cached_rows = {}
            all_shift_sums_list = []
            for item in points_to_compute:
                act_pos = item["act_pos"]
                shifts = item["above_indices"]
                bins = bins_arr[act_pos]
                if act_pos not in cached_rows:
                    cached_rows[act_pos] = np.asarray(corr_dev[act_pos])
                shifts_out = _point_chisq_cpu_compatible(
                    cached_rows[act_pos], shifts, bins, tlen, base_k=kmin
                )
                all_shift_sums_list.append(shifts_out)
            all_shift_sums = np.concatenate(all_shift_sums_list) if all_shift_sums_list else np.empty(0, dtype=real_dtype)
        else:
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
        chisq_vals = (shifts * num_bins - (above_snrv_np.conj() * above_snrv_np).real) * (norm ** 2.0)
        chisq_out = np.zeros(n_idx, dtype=real_dtype)
        chisq_out[above_mask] = chisq_vals.astype(real_dtype)
        chisq_dof = np.repeat(dof, n_idx)

        chisq_map[act_pos] = (chisq_out, chisq_dof)

    return chisq_map
