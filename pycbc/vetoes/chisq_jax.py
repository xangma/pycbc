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
import jax.numpy as jnp

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


def shift_sum(corr, points, bins):
    """Calculate time-shifted sum of FrequencySeries bins in JAX."""
    _ensure_x64()
    corr_arr = to_jax(corr)
    pts = np.asarray(points, dtype=np.int64)
    if len(pts) == 0:
        return np.array([], dtype=corr_arr.real.dtype)

    n_time = len(corr_arr)
    num_bins = len(bins) - 1
    res = jnp.zeros(len(pts), dtype=corr_arr.real.dtype)

    for b in range(num_bins):
        k = jnp.arange(int(bins[b]), int(bins[b + 1]))
        corr_sub = corr_arr[k]
        phases = jnp.exp(2j * jnp.pi * jnp.outer(k, pts) / n_time)
        zb = jnp.dot(corr_sub, phases)
        res = res + jnp.abs(zb) ** 2

    return np.asarray(res)


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
