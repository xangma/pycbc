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

"""JAX backend for thresholding and peak clustering."""

import functools
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.events.eventmgr import _BaseThresholdCluster
from pycbc.types.array_jax import _ensure_x64, to_jax

try:
    from .simd_threshold_cython import parallel_thresh_cluster
except (ImportError, OSError):
    parallel_thresh_cluster = None


def threshold(series, value):
    """Return indices and values in series exceeding threshold in magnitude."""
    _ensure_x64()
    arr = to_jax(series)
    mag_sq = arr.real ** 2 + arr.imag ** 2
    thresh_sq = float(value) ** 2

    mask = mag_sq > thresh_sq
    locs = jnp.nonzero(mask)[0]
    vals = arr[locs]
    return np.asarray(locs, dtype=np.uint32), np.asarray(vals)


threshold_only = threshold


@functools.partial(jax.jit, static_argnames=("window",))
def _fast_cluster_core(arr, threshold_sq, window):
    """JIT-compiled reduction and clustering kernel over fixed windows."""
    length = arr.shape[0]
    num_blocks = (length + window - 1) // window
    mag_sq = arr.real ** 2 + arr.imag ** 2
    pad = num_blocks * window - length
    if pad > 0:
        mag_sq_padded = jnp.pad(mag_sq, (0, pad), constant_values=-1.0)
    else:
        mag_sq_padded = mag_sq
    blocks = mag_sq_padded.reshape((num_blocks, window))
    local_max_idx = jnp.argmax(blocks, axis=1)
    block_max_vals = jnp.max(blocks, axis=1)
    global_max_idx = jnp.arange(num_blocks) * window + local_max_idx

    survivor_mask = (block_max_vals > threshold_sq) & (global_max_idx < length)
    if num_blocks > 1:
        greater_prev = block_max_vals[1:] > block_max_vals[:-1]
        greater_eq_next = block_max_vals[:-1] >= block_max_vals[1:]
        survivor_mask = survivor_mask.at[1:].set(
            survivor_mask[1:] & greater_prev
        )
        survivor_mask = survivor_mask.at[:-1].set(
            survivor_mask[:-1] & greater_eq_next
        )
        survivor_mask = survivor_mask.at[0].set(
            survivor_mask[0] & (block_max_vals[0] > block_max_vals[1])
        )
    return global_max_idx, survivor_mask


@functools.partial(jax.jit, static_argnames=("window",))
def _batched_cluster_core(valid_snr_2d, thresh_sq_1d, window):
    """JIT-compiled batched reduction and clustering across batch and windows."""
    b, length = valid_snr_2d.shape
    num_blocks = (length + window - 1) // window
    mag_sq = valid_snr_2d.real ** 2 + valid_snr_2d.imag ** 2
    pad = num_blocks * window - length
    if pad > 0:
        mag_sq_padded = jnp.pad(mag_sq, ((0, 0), (0, pad)), constant_values=-1.0)
        valid_snr_padded = jnp.pad(valid_snr_2d, ((0, 0), (0, pad)), constant_values=0.0)
    else:
        mag_sq_padded = mag_sq
        valid_snr_padded = valid_snr_2d
    blocks = mag_sq_padded.reshape((b, num_blocks, window))
    blocks_snr = valid_snr_padded.reshape((b, num_blocks, window))
    local_max_idx = jnp.argmax(blocks, axis=2)
    block_max_vals = jnp.max(blocks, axis=2)
    global_max_idx = jnp.arange(num_blocks)[None, :] * window + local_max_idx
    block_max_snr = jnp.take_along_axis(
        blocks_snr, local_max_idx[:, :, None], axis=2
    ).squeeze(axis=2)

    thresh_sq_2d = thresh_sq_1d[:, None]
    survivor_mask = (block_max_vals > thresh_sq_2d) & (global_max_idx < length)
    if num_blocks > 1:
        greater_prev = block_max_vals[:, 1:] > block_max_vals[:, :-1]
        greater_eq_next = block_max_vals[:, :-1] >= block_max_vals[:, 1:]
        survivor_mask = survivor_mask.at[:, 1:].set(
            survivor_mask[:, 1:] & greater_prev
        )
        survivor_mask = survivor_mask.at[:, :-1].set(
            survivor_mask[:, :-1] & greater_eq_next
        )
        survivor_mask = survivor_mask.at[:, 0].set(
            survivor_mask[:, 0] & (block_max_vals[:, 0] > block_max_vals[:, 1])
        )
    return global_max_idx, survivor_mask, block_max_snr


def threshold_and_cluster(series, threshold_val, window):
    """Return clustered values and indices exceeding threshold over window."""
    _ensure_x64()
    raw = getattr(series, "_data", series)
    arr_raw = getattr(raw, "array", raw)
    dev = getattr(arr_raw, "device", None)
    is_cpu = dev is None or (
        hasattr(dev, "platform") and dev.platform == "cpu"
    )

    if is_cpu and parallel_thresh_cluster is not None:
        arr_np = np.asarray(arr_raw, dtype=np.complex64)
        slen = np.uint32(len(arr_np))
        outv = np.zeros(slen, dtype=np.complex64)
        outl = np.zeros(slen, dtype=np.uint32)
        cnt = parallel_thresh_cluster(
            arr_np,
            slen,
            outv,
            outl,
            np.float32(threshold_val),
            np.uint32(window),
            np.uint32(32768),
        )
        if cnt > 0:
            return outv[:cnt], outl[:cnt]
        return (
            np.array([], dtype=np.complex64),
            np.array([], dtype=np.uint32),
        )

    arr = to_jax(series)
    length = len(arr)
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")

    num_blocks = (length + window - 1) // window
    if num_blocks == 0:
        return (
            np.array([], dtype=arr.dtype),
            np.array([], dtype=np.uint32),
        )

    threshold_sq = float(threshold_val) ** 2
    global_max_idx, survivor_mask = _fast_cluster_core(
        arr, threshold_sq, window=window
    )
    mask_np = np.asarray(survivor_mask)
    if not np.any(mask_np):
        return (
            np.array([], dtype=arr.dtype),
            np.array([], dtype=np.uint32),
        )

    survivor_indices = np.asarray(global_max_idx)[mask_np]
    survivor_values = np.asarray(arr[survivor_indices])
    return (
        survivor_values,
        survivor_indices.astype(np.uint32),
    )


class JAXThresholdCluster(_BaseThresholdCluster):
    """JAX implementation of ThresholdCluster with compiled acceleration."""

    def __init__(self, series):
        self.series = series
        self._outv = None
        self._outl = None

    def threshold_and_cluster(self, threshold, window):
        """Find clustered peaks exceeding threshold in magnitude."""
        if parallel_thresh_cluster is not None:
            raw = getattr(self.series, "_data", self.series)
            arr = getattr(raw, "array", raw)
            dev = getattr(arr, "device", None)
            is_cpu = dev is None or (
                hasattr(dev, "platform") and dev.platform == "cpu"
            )
            if is_cpu or isinstance(arr, np.ndarray):
                arr_np = np.asarray(arr, dtype=np.complex64)
                slen = np.uint32(len(arr_np))
                if self._outv is None or len(self._outv) < len(arr_np):
                    self._outv = np.zeros(slen, dtype=np.complex64)
                    self._outl = np.zeros(slen, dtype=np.uint32)
                cnt = parallel_thresh_cluster(
                    arr_np,
                    slen,
                    self._outv,
                    self._outl,
                    np.float32(threshold),
                    np.uint32(window),
                    np.uint32(32768),
                )
                if cnt > 0:
                    return self._outv[:cnt].copy(), self._outl[:cnt].copy()
                return (
                    np.array([], dtype=np.complex64),
                    np.array([], dtype=np.uint32),
                )

        return threshold_and_cluster(self.series, threshold, window)


def _threshold_cluster_factory(series):
    """Factory returning the JAXThresholdCluster class."""
    return JAXThresholdCluster
