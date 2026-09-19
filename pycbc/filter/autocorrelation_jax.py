# Copyright (C) 2026 PyCBC contributors
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

"""JAX implementation of autocorrelation functions."""

import numpy
import jax.numpy as jnp

from pycbc import scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData, to_jax


def calculate_acf_jax(data, delta_t, unbiased):
    """Calculate ACF without converting JAX TimeSeries to NumPy."""
    y = to_jax(data)
    y = y - jnp.mean(y)
    ny_orig = len(y)

    npad = 1
    while npad < 2 * ny_orig:
        npad = npad << 1

    ypad = jnp.zeros(npad, dtype=y.dtype)
    ypad = ypad.at[:ny_orig].set(y)

    fy = jnp.fft.rfft(ypad)
    cy = jnp.abs(fy) ** 2
    acf = jnp.fft.irfft(cy, n=npad)[:ny_orig]

    if unbiased:
        counts = jnp.arange(ny_orig, 0, -1, dtype=y.dtype)
        var = jnp.var(y)
        acf = acf / (var * counts)
    else:
        acf = acf / acf[0]

    if isinstance(_scheme.mgr.state, _scheme.JAXScheme):
        return TimeSeries(JAXArrayData(acf), delta_t=delta_t, copy=False)
    return TimeSeries(acf, delta_t=delta_t)


def calculate_acl_jax(acf, m=5):
    """Calculate autocorrelation length on JAX array."""
    cacf = 2 * jnp.cumsum(to_jax(acf)) - 1
    win = m * cacf <= jnp.arange(len(cacf))
    locs = jnp.nonzero(win)[0]
    if len(locs) > 0:
        return float(cacf[locs[0]])
    return numpy.inf
