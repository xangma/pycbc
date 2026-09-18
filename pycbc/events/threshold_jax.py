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

import numpy as np
import jax.numpy as jnp

from pycbc.events.eventmgr import _BaseThresholdCluster
from pycbc.types.array_jax import _ensure_x64, to_jax


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


def threshold_and_cluster(series, threshold_val, window):
    """Return clustered values and indices exceeding threshold over window."""
    _ensure_x64()
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

    mag_sq = arr.real ** 2 + arr.imag ** 2
    pad = num_blocks * window - length
    if pad > 0:
        mag_sq_padded = jnp.pad(
            mag_sq, (0, pad), mode="constant", constant_values=-1.0
        )
    else:
        mag_sq_padded = mag_sq

    blocks = mag_sq_padded.reshape((num_blocks, window))
    local_max_idx = jnp.argmax(blocks, axis=1)
    block_max_vals = jnp.max(blocks, axis=1)
    global_max_idx = jnp.arange(num_blocks) * window + local_max_idx

    threshold_sq = float(threshold_val) ** 2
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

    survivor_indices = global_max_idx[survivor_mask]
    survivor_values = arr[survivor_indices]
    return (
        np.asarray(survivor_values),
        np.asarray(survivor_indices, dtype=np.uint32),
    )


class JAXThresholdCluster(_BaseThresholdCluster):
    """JAX implementation of ThresholdCluster."""

    def __init__(self, series):
        self.series = series

    def threshold_and_cluster(self, threshold, window):
        """Find clustered peaks exceeding threshold in magnitude."""
        return threshold_and_cluster(self.series, threshold, window)


def _threshold_cluster_factory(series):
    """Factory returning the JAXThresholdCluster class."""
    return JAXThresholdCluster
