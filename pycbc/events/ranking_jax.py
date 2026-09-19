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

"""JAX implementations of event ranking statistics."""

import jax.numpy as jnp

from pycbc import scheme
from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _as_jax_array
from pycbc.types.backend import is_backend


def ranking_arrays_jax(*values):
    """Convert ranking inputs to JAX arrays if active."""
    if not values:
        return None
    if isinstance(scheme.mgr.state, scheme.JAXScheme) or any(
        is_backend(v, "jax") for v in values
    ):
        res = []
        for v in values:
            arr = _as_jax_array(v)
            if arr is None:
                host = v.numpy() if hasattr(v, "numpy") else v
                arr = jnp.asarray(host, dtype=jnp.float64)
            res.append(arr)
        return res
    return None


def effsnr_jax(snr, reduced_x2, fac=250.0):
    """Calculate effective SNR using JAX."""
    jax_arrs = ranking_arrays_jax(snr, reduced_x2)
    if jax_arrs is None:
        return None
    snr_j, rchisq_j = jax_arrs
    esnr = snr_j / (1.0 + snr_j**2 / fac) ** 0.25 / rchisq_j**0.25
    if isinstance(snr, Array) or isinstance(reduced_x2, Array):
        return Array(JAXArrayData(esnr), copy=False)
    return esnr


def newsnr_jax(snr, reduced_x2, q=6.0, n=2.0):
    """Calculate re-weighted new SNR using JAX."""
    jax_arrs = ranking_arrays_jax(snr, reduced_x2)
    if jax_arrs is None:
        return None
    snr_j, rchisq_j = jax_arrs
    reweight = (0.5 * (1.0 + rchisq_j ** (q / n))) ** (-1.0 / q)
    vals = jnp.where(rchisq_j > 1.0, snr_j * reweight, snr_j)
    if isinstance(snr, Array) or isinstance(reduced_x2, Array):
        return Array(JAXArrayData(vals), copy=False)
    return vals
