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
import numpy as np

from pycbc import scheme
from pycbc.events import ranking
from pycbc.events.ranking_jax import _event_arrays
from pycbc.types import Array
from pycbc.types.array_jax import (
    JAXArrayData, _divide, _ensure_x64,
)
from pycbc.types.backend import is_backend


def _column(value):
    return _event_arrays(value)[0]


def _reduced_chisq(triggers, choice):
    if choice == "sg":
        return _column(triggers["sg_chisq"])
    if choice not in ("traditional", "cont", "bank"):
        raise ValueError("Do not recognize --chisq-choice %s" % choice)
    name = "chisq" if choice == "traditional" else choice + "_chisq"
    numerator = _column(triggers[name])
    denominator = _column(triggers[name + "_dof"])
    if choice == "traditional":
        denominator = denominator * 2 - 2
    dtype = np.result_type(numerator.dtype, denominator.dtype)
    if dtype.kind not in "fc":
        dtype = np.dtype(np.float64)
    return _divide(numerator.astype(dtype), denominator.astype(dtype))



def apply_trigger_cuts_jax(triggers, trigger_cut_dict, statistic=None):
    """Apply trigger cuts on JAX arrays without host transfers."""
    snr = triggers["snr"]
    if not (is_backend(snr, "jax") or
            isinstance(scheme.mgr.state, scheme.JAXScheme)):
        return None

    _ensure_x64()
    idx_out = jnp.arange(len(snr), dtype=jnp.int64)
    for parameter_cut_function, cut_thresh in trigger_cut_dict.items():
        parameter, cut_function = parameter_cut_function
        if parameter.endswith('_chisq'):
            val_j = _reduced_chisq(triggers, parameter.split('_')[0])
        elif parameter == 'sigma_multiple':
            raise NotImplementedError(
                "JAX sigma_multiple cuts require a template-indexed reader")
        elif parameter in triggers:
            val_j = _column(triggers[parameter])
        elif parameter in ('snr', 'newsnr', 'new_snr'):
            columns = {name: _column(triggers[name])
                       for name in ranking.reqd_datasets[parameter]}
            val_j = ranking.get_sngls_ranking_from_trigs(columns, parameter)
        elif parameter in ranking.sngls_ranking_function_dict:
            raise NotImplementedError(
                "JAX trigger cuts support snr/newsnr ranking; %s is unavailable"
                % parameter)
        else:
            raise NotImplementedError("Parameter '%s' not recognised" % parameter)
        val_j = val_j[idx_out]
        # NumPy ufuncs otherwise copy JAX inputs to the host implicitly.
        compare = (getattr(jnp, cut_function.__name__, cut_function)
                   if isinstance(cut_function, np.ufunc) else cut_function)
        mask = compare(val_j, cut_thresh)
        idx_out = idx_out[mask]

    if isinstance(snr, Array):
        return Array(JAXArrayData(idx_out), copy=False)
    return idx_out
