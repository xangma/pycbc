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


import functools

try:
    from . import chisq_cpu
except (ImportError, OSError):
    chisq_cpu = None


@functools.partial(jax.jit, static_argnames=("chunk_size",))
def _shift_sum_gpu_core(
    arr_slice, kmin_f, pts_chunk, bin_rel_indices, n_time_f, chunk_size=16
):
    """JIT-compiled GPU power chisq prefix sum across frequency bins on corr_slice."""
    n_slice = arr_slice.shape[0]
    k = kmin_f + jnp.arange(n_slice, dtype=jnp.float32)
    phases = jnp.exp((2j * jnp.pi / n_time_f) * (k[:, None] * pts_chunk[None, :]))
    weighted = arr_slice[:, None] * phases
    C = jnp.cumsum(weighted, axis=0)
    C_padded = jnp.pad(C, ((1, 0), (0, 0)))
    i0 = bin_rel_indices[:-1]
    i1 = bin_rel_indices[1:]
    zb = C_padded[i1] - C_padded[i0]
    power = jnp.sum(zb.real ** 2 + zb.imag ** 2, axis=0)
    return power


@functools.partial(jax.jit, static_argnames=("chunk_size",))
def _shift_sum_gpu_core_row(
    corr_tensor, row_idx, kmin_f, pts_chunk, bin_rel_indices, n_time_f, chunk_size=16
):
    """JIT-compiled GPU power chisq prefix sum indexing row on device."""
    arr_slice = corr_tensor[row_idx]
    n_slice = arr_slice.shape[0]
    k = kmin_f + jnp.arange(n_slice, dtype=jnp.float32)
    phases = jnp.exp((2j * jnp.pi / n_time_f) * (k[:, None] * pts_chunk[None, :]))
    weighted = arr_slice[:, None] * phases
    C = jnp.cumsum(weighted, axis=0)
    C_padded = jnp.pad(C, ((1, 0), (0, 0)))
    i0 = bin_rel_indices[:-1]
    i1 = bin_rel_indices[1:]
    zb = C_padded[i1] - C_padded[i0]
    power = jnp.sum(zb.real ** 2 + zb.imag ** 2, axis=0)
    return power


@functools.partial(jax.jit, static_argnames=("bucket_size",))
def _batched_points_chisq_core(
    corr_tensor, row_indices, pts, bins_rel_all, kmin_f, n_time_f, bucket_size=256
):
    """JIT-compiled GPU batched power chisq prefix sum across all triggered points.

    Calculates time-shifted frequency bin sums for all points across all templates
    in parallel via vmap over points, executing in a single CUDA kernel.
    """
    n_slice = corr_tensor.shape[-1]
    k = kmin_f + jnp.arange(n_slice, dtype=jnp.float32)

    def _point_chisq(row, pt, bin_rel):
        phases = jnp.exp((2j * jnp.pi / n_time_f) * (k * pt.astype(jnp.float32)))
        weighted = row * phases
        C = jnp.cumsum(weighted)
        C_padded = jnp.pad(C, (1, 0))
        i0 = bin_rel[:-1]
        i1 = bin_rel[1:]
        zb = C_padded[i1] - C_padded[i0]
        return jnp.sum(zb.real ** 2 + zb.imag ** 2)

    rows = corr_tensor[row_indices]
    bins_pts = bins_rel_all[row_indices]
    return jax.vmap(_point_chisq)(rows, pts, bins_pts)


def shift_sum(corr, points, bins):
    """Calculate time-shifted sum of FrequencySeries bins in JAX."""
    _ensure_x64()
    pts = np.asarray(points, dtype=np.int64)
    if len(pts) == 0:
        return np.array([], dtype=np.float32)

    use_row_indexing = (
        hasattr(corr, "_batch_tensor")
        and hasattr(corr, "_batch_pos")
        and corr._batch_tensor is not None
    )

    if not use_row_indexing:
        arr = to_jax(corr)
        dev = getattr(arr, "device", None)
        if callable(dev):
            dev = dev()
        platform = getattr(dev, "platform", None)
        is_cpu = platform == "cpu" or platform is None
    else:
        is_cpu = False

    if is_cpu and not hasattr(corr, "_kmin") and chisq_cpu is not None and hasattr(chisq_cpu, "point_chisq_code"):
        arr_np = np.asarray(arr)
        real_type = arr_np.real.dtype
        shifts_np = np.asarray(pts, dtype=real_type)
        bins_np = np.asarray(bins, dtype=np.uint32)
        blen = len(bins_np) - 1
        n = int(len(shifts_np))
        slen = len(arr_np)
        chisq = np.zeros(n, dtype=real_type)
        chisq_cpu.point_chisq_code(
            chisq, arr_np, n, slen, shifts_np, bins_np, blen
        )
        return chisq

    n_time = getattr(corr, "_tlen", (len(arr) - 1) * 2 if not use_row_indexing and hasattr(corr, "delta_f") else 2097152)
    num_bins = len(bins) - 1
    k0 = int(bins[0])
    k1 = int(bins[-1])

    if is_cpu:
        arr_np = np.asarray(arr)
        target_dtype = arr_np.dtype
        if hasattr(corr, "_kmin"):
            base_k = int(corr._kmin)
            slice_offset = k0 - base_k
            arr_k = arr_np[slice_offset:slice_offset + (k1 - k0)]
        else:
            arr_k = arr_np[k0:k1]
        k_all = np.arange(k0, k1)
        phases = np.exp(
            (2j * np.pi / n_time) * np.outer(k_all, pts)
        ).astype(target_dtype)
        weighted = arr_k[:, None] * phases
        res = np.zeros(len(pts), dtype=arr_np.real.dtype)
        for b in range(num_bins):
            i0 = int(bins[b]) - k0
            i1 = int(bins[b + 1]) - k0
            zb = np.sum(weighted[i0:i1], axis=0)
            res += (zb.real ** 2 + zb.imag ** 2)
        return res
    else:
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
            n_time = (len(arr) - 1) * 2 if hasattr(corr, "delta_f") else len(arr)
            kmin = int(bins[0])
            kmax = int(bins[-1])
            arr_slice = arr[kmin:kmax]

        bin_indices = jnp.asarray(bins, dtype=jnp.int32)
        bin_rel = bin_indices - kmin
        kmin_f = float(kmin)
        n_time_f = float(n_time)

        pts_np = np.asarray(pts, dtype=np.int32)
        n_pts = len(pts_np)
        out_res = np.empty(n_pts, dtype=np.float32)

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
            out_res[offset:offset + cur_len] = np.asarray(res_j)[:cur_len]
            offset += cur_len
        return out_res


def power_chisq_at_points_from_precomputed(
    corr, snr, snr_norm, bins, indices
):
    """Calculate chisq values at triggered points from precomputed products."""
    num_bins = len(bins) - 1
    chisq = shift_sum(corr, indices, bins)
    snr_arr = np.asarray(snr)
    return (
        (chisq * num_bins - (snr_arr.conj() * snr_arr).real)
        * (float(snr_norm) ** 2.0)
    )


@functools.partial(jax.jit, static_argnames=("kmin", "kmax", "num_bins", "delta_f"))
def _power_chisq_bins_core(h_arr, psd_arr, kmin, kmax, num_bins, delta_f):
    h_slice = h_arr[kmin:kmax]
    psd_slice = psd_arr[kmin:kmax]
    mag_sq = (h_slice.real ** 2 + h_slice.imag ** 2) / psd_slice
    sub = jnp.cumsum(mag_sq) * (4.0 * delta_f)
    sigmasq = sub[-1]
    edge_vec = (jnp.arange(num_bins, dtype=sub.dtype) * sigmasq) / num_bins
    bins = jnp.searchsorted(sub, edge_vec, side="right") + kmin
    return jnp.append(bins, kmax)


def power_chisq_bins_jax(
    htilde,
    num_bins,
    psd,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
):
    """JAX GPU accelerated calculation of equal-power chisq bin edges."""
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

    bins_gpu = _power_chisq_bins_core(
        h_arr, psd_arr, int(kmin), int(kmax), int(num_bins), delta_f
    )
    return np.asarray(bins_gpu, dtype=np.uint32)


@functools.partial(jax.jit, static_argnames=("kmin", "kmax", "num_bins", "delta_f"))
def _batched_power_chisq_bins_core(tmpls_tensor, psd_arr, kmin, kmax, num_bins, delta_f):
    h_slice = tmpls_tensor[:, kmin:kmax]
    psd_slice = psd_arr[kmin:kmax]
    mag_sq = (h_slice.real ** 2 + h_slice.imag ** 2) / psd_slice[None, :]
    sub = jnp.cumsum(mag_sq, axis=-1) * (4.0 * delta_f)
    sigmasq = sub[:, -1:]
    edge_vec = (jnp.arange(num_bins, dtype=sub.dtype)[None, :] * sigmasq) / num_bins

    def row_search(s, e):
        return jnp.searchsorted(s, e, side="right") + kmin

    bins = jax.vmap(row_search)(sub, edge_vec)
    kmax_col = jnp.full((bins.shape[0], 1), kmax, dtype=bins.dtype)
    return jnp.concatenate([bins, kmax_col], axis=-1)


def batch_power_chisq_bins_jax(
    templates,
    num_bins,
    psd,
    low_frequency_cutoff=None,
    high_frequency_cutoff=None,
):
    """JAX GPU batched calculation of equal-power chisq bin edges for template batch."""
    from pycbc.filter.matchedfilter import get_cutoff_indices

    _ensure_x64()
    n_pts = (len(templates[0]) - 1) * 2
    delta_f = float(templates[0].delta_f)
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

    bins_gpu = _batched_power_chisq_bins_core(
        tmpls_tensor,
        psd_arr,
        int(kmin),
        int(kmax),
        int(num_bins),
        delta_f,
    )
    return np.asarray(bins_gpu, dtype=np.uint32)


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

    Computes time-shifted frequency bin sums for all triggered points across all
    templates simultaneously in a single JAX GPU kernel launch, replacing sequential
    per-template dispatch and synchronization loops.

    Returns:
        chisq_map: dict mapping act_pos -> (chisq_out, chisq_dof)
    """
    _ensure_x64()
    if corr_tensor is None or len(batch_results) == 0:
        return None

    psd_id = id(psd)
    b = len(active_templates)

    # 1. Gather bin edges for all active templates
    bins_list = []
    for tmpl in active_templates:
        b_edges = None
        if hasattr(tmpl, "_bin_cache") and psd_id in tmpl._bin_cache:
            b_edges = tmpl._bin_cache[psd_id]
        elif power_chisq is not None and hasattr(power_chisq, "cached_chisq_bins"):
            b_edges = power_chisq.cached_chisq_bins(tmpl, psd)
        if b_edges is None:
            return None
        bins_list.append(b_edges)

    # Ensure all templates have identical bin count
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

    for act_pos in range(b):
        snr, norm, corr, idx, snrv = batch_results[act_pos]
        n_idx = len(idx)
        if n_idx == 0:
            continue

        if snr_threshold is not None:
            above = abs(snrv * norm) > snr_threshold
            n_above = int(np.sum(above))
        else:
            above = np.ones(n_idx, dtype=bool)
            n_above = n_idx

        if n_above == 0:
            chisq_map[act_pos] = (
                np.zeros(n_idx, dtype=np.float32),
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
    bins_arr = np.array(bins_list, dtype=np.int32)
    bins_rel_all = jax.device_put(bins_arr - kmin)

    all_rows = np.empty(total_points, dtype=np.int32)
    all_pts = np.empty(total_points, dtype=np.int32)

    offset = 0
    for item in points_to_compute:
        n_ab = item["n_above"]
        all_rows[offset:offset + n_ab] = item["act_pos"]
        all_pts[offset:offset + n_ab] = item["above_indices"]
        offset += n_ab

    # 4. Bucketed JAX execution
    bucket_sizes = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    chosen_bucket = None
    for bsz in bucket_sizes:
        if total_points <= bsz:
            chosen_bucket = bsz
            break
    if chosen_bucket is None:
        chosen_bucket = int(2 ** np.ceil(np.log2(total_points)))

    r_buf = np.zeros(chosen_bucket, dtype=np.int32)
    p_buf = np.zeros(chosen_bucket, dtype=np.int32)
    r_buf[:total_points] = all_rows
    p_buf[:total_points] = all_pts

    r_dev = jax.device_put(r_buf)
    p_dev = jax.device_put(p_buf)

    res_dev = _batched_points_chisq_core(
        corr_tensor, r_dev, p_dev, bins_rel_all, kmin_f, n_time_f, bucket_size=chosen_bucket
    )
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

        chisq_vals = (shifts * num_bins - (above_snrv.conj() * above_snrv).real) * (norm ** 2.0)
        chisq_out = np.zeros(n_idx, dtype=np.float32)
        chisq_out[above_mask] = chisq_vals.astype(np.float32)
        chisq_dof = np.repeat(dof, n_idx)

        chisq_map[act_pos] = (chisq_out, chisq_dof)

    return chisq_map
