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

"""JAX waveform utility functions."""

import numpy as np
import jax.numpy as jnp

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shift a uniformly sampled frequency-domain waveform in time using JAX.
    """
    _ensure_x64()
    data = to_jax(htilde)
    kmax = len(data)
    delta_f = float(htilde.delta_f)

    if kmax > kmin:
        k = jnp.arange(kmin, kmax)
        theta = -2.0 * np.pi * float(dt) * delta_f * k
        shift = jnp.exp(1j * theta)
        new_data = data.at[kmin:].set(data[kmin:] * shift)
    else:
        new_data = data

    return FrequencySeries(
        JAXArrayData(new_data),
        delta_f=htilde.delta_f,
        epoch=htilde.epoch,
        copy=False,
    )
