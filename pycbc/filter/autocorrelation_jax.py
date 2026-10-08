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

from pycbc import scheme
from pycbc.filter.matchedfilter import correlate
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.types.array_jax import (
    JAXArrayData, _cpu_reference, _divide, _reference_enabled, to_jax,
)


def calculate_acf_jax(data, delta_t, unbiased):
    """Calculate ACF without converting JAX TimeSeries to NumPy."""
    if _reference_enabled("autocorrelation"):
        from pycbc.reference_jax import cpu_reference

        values, delta_t, epoch = cpu_reference(
            "autocorrelation", to_jax(data), spacing=delta_t,
            epoch=getattr(data, "_epoch", None), unbiased=unbiased,
            is_series=isinstance(data, TimeSeries))
        return TimeSeries(JAXArrayData(to_jax(values)), delta_t=delta_t,
                          epoch=epoch, copy=False)

    y = to_jax(data)
    mean = (numpy.mean(numpy.asarray(y))
            if _reference_enabled("autocorrelation_mean") else jnp.mean(y))
    y = y - mean
    ny_orig = len(y)

    npad = 1
    while npad < 2 * ny_orig:
        npad = npad << 1

    # The original routine promotes its zero-padded FFT input to float64.
    ypad = jnp.zeros(npad, dtype=jnp.float64).at[:ny_orig].set(y)
    copy = not isinstance(scheme.mgr.state, scheme.JAXScheme)
    fdata = TimeSeries(
        JAXArrayData(ypad), delta_t=delta_t, copy=copy).to_frequencyseries()
    cdata = FrequencySeries(
        zeros(len(fdata), dtype=fdata.dtype),
        delta_f=fdata.delta_f, copy=False)
    correlate(fdata, fdata, cdata)
    acf = to_jax(cdata.to_timeseries())[:ny_orig]

    if unbiased:
        counts = jnp.arange(ny_orig, 0, -1, dtype=jnp.float64)
        var = (numpy.var(numpy.asarray(y))
               if _reference_enabled("autocorrelation_variance")
               else jnp.var(y))
        acf = _divide(acf, var * counts, inplace=True)
    else:
        acf = _divide(acf, acf[0], inplace=True)

    return TimeSeries(JAXArrayData(acf), delta_t=delta_t, copy=copy)


def calculate_acl_jax(acf, m=5):
    """Calculate autocorrelation length on JAX array."""
    values = to_jax(acf)
    cumulative = (to_jax(_cpu_reference(values, "cumsum"))
                  if _reference_enabled("cumsum") else jnp.cumsum(values))
    cacf = 2 * cumulative - 1
    win = m * cacf <= jnp.arange(len(cacf))
    locs = jnp.nonzero(win)[0]
    if len(locs) > 0:
        return float(cacf[locs[0]])
    return numpy.inf
