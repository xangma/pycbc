# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX-specific waveform utilities."""

import math

import jax
import jax.numpy as jnp

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData, _reference_enabled


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shift a uniformly sampled frequency-domain waveform in time using JAX.

    A non-scalar ``dt`` is aligned with the waveform's sample axes; the final
    data axis is always frequency. It cannot introduce a sample axis into a
    one-dimensional ``FrequencySeries``.
    """
    data = htilde._data.array
    if _reference_enabled("time_shift"):
        from pycbc.reference_jax import cpu_reference

        values, spacing, epoch = cpu_reference(
            "time_shift", jax.device_get(data), htilde.delta_f, htilde.epoch,
            dt=dt, kmin=kmin, copy=copy)
        if not copy:
            htilde._data.set_array(jax.device_put(values, data.device))
            return htilde
        return FrequencySeries(values, delta_f=spacing, epoch=epoch)

    if isinstance(dt, (jax.Array, jnp.ndarray)):
        real_dtype = jnp.real(data).dtype
        dt_value = jnp.asarray(dt, dtype=real_dtype)
    else:
        try:
            dt_value = float(dt)
        except (TypeError, ValueError):
            real_dtype = jnp.real(data).dtype
            dt_value = jnp.asarray(dt, dtype=real_dtype)

    if isinstance(dt_value, (jax.Array, jnp.ndarray)) and dt_value.ndim:
        sample_shape = data.shape[:-1]
        try:
            broadcast_shape = jnp.broadcast_shapes(
                tuple(dt_value.shape), tuple(sample_shape)
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "A batched time shift must broadcast across the waveform "
                "sample axes"
            ) from exc
        if tuple(broadcast_shape) != tuple(sample_shape):
            raise ValueError(
                "A batched time shift cannot introduce sample axes into a "
                "FrequencySeries"
            )
        dt_value = jnp.expand_dims(dt_value, -1)

    kmax = data.shape[-1]
    if kmax > kmin:
        indices = jnp.arange(kmin, kmax, dtype=jnp.real(data).dtype)
        theta = (-2.0 * math.pi * dt_value * float(htilde.delta_f)) * indices
        cosine = jnp.cos(theta)
        sine = jnp.sin(theta)
        shift = cosine + 1j * sine
        if kmin > 0:
            target = data[..., kmin:] * shift
            data = jnp.concatenate((data[..., :kmin], target), axis=-1)
        else:
            data = data * shift

    if not copy:
        htilde._data.set_array(data)
        return htilde
    return FrequencySeries(
        JAXArrayData(data),
        delta_f=htilde.delta_f,
        epoch=htilde.epoch,
        copy=False,
    )
