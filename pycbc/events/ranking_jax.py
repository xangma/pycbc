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

import jax
import jax.numpy as jnp
import numpy as np

from pycbc import scheme
from pycbc.types import Array
from pycbc.types.array_jax import (
    JAXArrayData, _as_jax_array, _divide, _ensure_x64, _reference_enabled,
)
from pycbc.types.backend import is_backend


def _event_arrays(*values, device=None):
    """Align event columns without changing their numeric precision."""
    arrays = [_as_jax_array(value) for value in values]
    if device is None:
        device = getattr(scheme.mgr.state, 'jax_device', None)
    if device is None:
        device = next((array.device for array in arrays
                       if array is not None and
                       not isinstance(array, jax.core.Tracer)), None)
    result = []
    for value, array in zip(values, arrays):
        if array is None:
            array = value.numpy() if isinstance(value, Array) else value
        array = jnp.asarray(array)
        if device is not None and not isinstance(array, jax.core.Tracer):
            array = jax.device_put(array, device)
        result.append(array)
    return result


def ranking_arrays_jax(*values):
    """Convert ranking inputs to JAX arrays if active."""
    if not values:
        return None
    if isinstance(scheme.mgr.state, scheme.JAXScheme) or any(
        is_backend(v, "jax") for v in values
    ):
        # The native ranking functions promote their numeric inputs to
        # float64.  Keep the JAX implementation numerically equivalent even
        # when its inputs arrived as device-resident float32 arrays.
        _ensure_x64()
        return [array.astype(jnp.float64) for array in _event_arrays(*values)]
    return None


def effsnr_jax(snr, reduced_x2, fac=250.0):
    """Calculate effective SNR using JAX."""
    jax_arrs = ranking_arrays_jax(snr, reduced_x2)
    if jax_arrs is None:
        return None
    snr_j, rchisq_j = jax_arrs
    if _reference_enabled("effsnr"):
        esnr = _native_ranking("effsnr", snr, snr_j, rchisq_j, fac=fac)
    else:
        # Original effsnr promotes both inputs to arrays with ndmin=1.
        snr_j, rchisq_j = jnp.atleast_1d(snr_j), jnp.atleast_1d(rchisq_j)
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
    if _reference_enabled("newsnr"):
        vals = _native_ranking("newsnr", snr, snr_j, rchisq_j, q=q, n=n)
    else:
        reweight = (0.5 * (1.0 + rchisq_j ** (q / n))) ** (-1.0 / q)
        vals = jnp.where(rchisq_j > 1.0, snr_j * reweight, snr_j)
    if isinstance(snr, Array) or isinstance(reduced_x2, Array):
        return Array(JAXArrayData(vals), copy=False)
    return vals


def get_newsnr_jax(trigs, **options):
    """Evaluate the original getter's field arithmetic on the selected device."""
    _ensure_x64()
    snr, chisq, dof = _event_arrays(*(trigs[name][:] for name in
                                    ('snr', 'chisq', 'chisq_dof')))
    reduced = _divide(chisq, 2. * dof - 2.)
    return jnp.atleast_1d(newsnr_jax(snr, reduced, **options)).astype(jnp.float32)


def _native_ranking(operation, original, snr, reduced_x2, **options):
    """Execute the original CPU ranking without changing the active scheme."""
    from pycbc.reference_jax import cpu_reference

    scalar = not isinstance(original, Array) and (
        np.isscalar(original) or isinstance(original, jax.Array) and original.ndim == 0)
    result = cpu_reference(
        operation, jax.device_get(snr), reduced_x2=jax.device_get(reduced_x2),
        is_scalar=scalar, **options)
    return jax.device_put(result, snr.device)


def quadrature_sum_jax(sngls_list):
    """Combine single-detector statistics at the event arithmetic boundary."""
    arrays = _event_arrays(*(item[1] for item in sngls_list))
    if _reference_enabled('quadrature_sum'):
        from pycbc.reference_jax import cpu_reference

        result = cpu_reference(
            'quadrature_sum', None,
            sngls_list=[(item[0], value) for item, value in
                        zip(sngls_list, jax.device_get(arrays))],
            slide=None, step=None, to_shift=None)
        return jax.device_put(result, arrays[0].device)
    result = jnp.sqrt(sum(value * value for value in arrays))
    for item in sngls_list:
        result = jnp.where(item == -1, 0, result)
    return result
