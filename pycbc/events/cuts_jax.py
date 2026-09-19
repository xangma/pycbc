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

"""JAX implementations of trigger cutting routines."""

import jax.numpy as jnp

from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, _as_jax_array, is_jax_array
from pycbc.types.backend import is_backend


def apply_trigger_cuts_jax(triggers, trigger_cut_dict):
    """Apply trigger cuts on JAX arrays without host transfers."""
    snr = triggers["snr"]
    if not (is_backend(snr, "jax") or is_jax_array(snr)):
        return None

    idx_out = jnp.arange(len(snr), dtype=jnp.int64)
    for parameter_cut_function, cut_thresh in trigger_cut_dict.items():
        parameter, cut_function = parameter_cut_function
        value = triggers[parameter]
        val_j = _as_jax_array(value)
        if val_j is None:
            host = value.numpy() if hasattr(value, "numpy") else value
            val_j = jnp.asarray(host)
        val_j = val_j[idx_out]
        mask = cut_function(val_j, cut_thresh)
        idx_out = idx_out[mask]

    if isinstance(snr, Array):
        return Array(JAXArrayData(idx_out), copy=False)
    return idx_out
