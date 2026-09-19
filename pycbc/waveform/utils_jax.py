# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX waveform utility functions."""

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import i0

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax
from pycbc.types.backend import backend_array


def _jax_kaiser_window(data, length, beta):
    """Return SciPy's periodic Kaiser window for JAX backend."""
    dtype = data.real.dtype
    if length <= 1:
        return jnp.ones(length, dtype=dtype)

    position = jnp.arange(length, dtype=dtype)
    radius = 2 * position / length - 1
    argument = float(beta) * jnp.sqrt(jnp.clip(1 - radius * radius, 0.0))
    normalization = i0(jnp.asarray(float(beta), dtype=dtype))
    return i0(argument) / normalization


def td_taper_jax(out, start, end, beta=8, side="left"):
    """Applies a taper to the given TimeSeries using JAX."""
    data = backend_array(out, "jax")
    if data is None:
        raise TypeError("Expected JAX-backed TimeSeries")
    width = end - start
    winlen = 2 * int(width / out.delta_t)
    xmin = int((start - out.start_time) / out.delta_t)
    xmax = xmin + winlen // 2

    window = _jax_kaiser_window(data, winlen, beta)
    target = data
    if side == "left":
        part = target[xmin:xmax] * window[: winlen // 2]
        target = target.at[xmin:xmax].set(part)
        if xmin > 0:
            target = target.at[:xmin].set(0.0)
    elif side == "right":
        part = target[xmin:xmax] * window[winlen // 2 :]
        target = target.at[xmin:xmax].set(part)
        if xmax < len(out):
            target = target.at[xmax:].set(0.0)
    else:
        raise ValueError(f"unrecognized side argument {side}")
    out._data.set_array(target)
    return out


def fd_taper_jax(out, start, end, beta=8, side="left"):
    """Applies a taper to the given FrequencySeries using JAX."""
    data = backend_array(out, "jax")
    if data is None:
        raise TypeError("Expected JAX-backed FrequencySeries")
    width = end - start
    winlen = 2 * int(width / out.delta_f)
    kmin = int(start / out.delta_f)
    kmax = kmin + winlen // 2

    window = _jax_kaiser_window(data, winlen, beta)
    target = data
    if side == "left":
        part = target[kmin:kmax] * window[: winlen // 2]
        target = target.at[kmin:kmax].set(part)
        target = target.at[:kmin].set(0.0)
    elif side == "right":
        part = target[kmin:kmax] * window[winlen // 2 :]
        target = target.at[kmin:kmax].set(part)
        target = target.at[kmax:].set(0.0)
    else:
        raise ValueError(f"unrecognized side argument {side}")
    out._data.set_array(target)
    return out


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shift a uniformly sampled frequency-domain waveform in time using JAX."""
    _ensure_x64()
    data = to_jax(htilde)
    kmax = len(data)
    delta_f = float(htilde.delta_f)

    if kmax > kmin:
        k = jnp.arange(kmin, kmax)
        theta = -2.0 * np.pi * dt * delta_f * k
        shift = jnp.exp(1j * theta)
        new_data = data.at[kmin:].set(data[kmin:] * shift)
    else:
        new_data = data

    if not copy and hasattr(htilde, "_data") and isinstance(htilde._data, JAXArrayData):
        htilde._data.set_array(new_data)
        return htilde

    return FrequencySeries(
        JAXArrayData(new_data),
        delta_f=htilde.delta_f,
        epoch=htilde.epoch,
        copy=False,
    )
