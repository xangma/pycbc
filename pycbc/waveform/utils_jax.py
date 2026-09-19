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
from jax.scipy.special import i0

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData
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
    """Shift a uniformly sampled frequency-domain waveform in time using JAX.

    A non-scalar ``dt`` is aligned with the waveform's sample axes; the final
    data axis is always frequency. It cannot introduce a sample axis into a
    one-dimensional ``FrequencySeries``.
    """
    data = htilde._data.array

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
                "A batched time shift must broadcast across the waveform sample axes"
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

    return FrequencySeries(
        JAXArrayData(data),
        delta_f=htilde.delta_f,
        epoch=htilde.epoch,
        copy=False,
    )


def fused_detector_strain_fd_jax(
    hp_tensor, hc_tensor, fp_list, fc_list, dt_list, delta_f, kmin=0
):
    """Compute detector projection and time shifts in one operation using JAX.

    Returns a complex JAX array with shape
    ``(detectors, *sample_shape, frequencies)``. Scalar detector responses
    omit ``sample_shape``.
    """
    real_dtype = jnp.real(hp_tensor).dtype
    ndet = len(fp_list)
    if ndet == 0 or len(fc_list) != ndet or len(dt_list) != ndet:
        raise ValueError(
            "fplus, fcross, and time-shift values must have the same "
            "non-zero detector count"
        )

    detector_values = [
        jnp.asarray(value, dtype=real_dtype)
        for values in (fp_list, fc_list, dt_list)
        for value in values
    ]
    try:
        detector_values = jnp.broadcast_arrays(*detector_values)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Detector responses and time shifts do not have compatible sample shapes"
        ) from exc
    fp = jnp.stack(detector_values[:ndet], axis=0)
    fc = jnp.stack(detector_values[ndet : 2 * ndet], axis=0)
    dt = jnp.stack(detector_values[2 * ndet :], axis=0)

    fp = jnp.expand_dims(fp, -1)
    fc = jnp.expand_dims(fc, -1)
    dt = jnp.expand_dims(dt, -1)

    kmax = hp_tensor.shape[-1]
    hp_r = jnp.real(hp_tensor) if jnp.iscomplexobj(hp_tensor) else hp_tensor
    hp_i = jnp.imag(hp_tensor) if jnp.iscomplexobj(hp_tensor) else jnp.zeros_like(hp_r)
    hc_r = jnp.real(hc_tensor) if jnp.iscomplexobj(hc_tensor) else hc_tensor
    hc_i = jnp.imag(hc_tensor) if jnp.iscomplexobj(hc_tensor) else jnp.zeros_like(hc_r)

    out_r = fp * hp_r + fc * hc_r
    out_i = fp * hp_i + fc * hc_i

    if kmax <= kmin:
        return out_r + 1j * out_i

    indices = jnp.arange(kmin, kmax, dtype=real_dtype)
    theta = (-2.0 * math.pi * float(delta_f) * dt) * indices
    shift_r = jnp.cos(theta)
    shift_i = jnp.sin(theta)

    if kmin == 0:
        res_r = out_r * shift_r - out_i * shift_i
        res_i = out_r * shift_i + out_i * shift_r
        return res_r + 1j * res_i

    head_r = out_r[..., :kmin]
    head_i = out_i[..., :kmin]
    tail_r = out_r[..., kmin:] * shift_r - out_i[..., kmin:] * shift_i
    tail_i = out_r[..., kmin:] * shift_i + out_i[..., kmin:] * shift_r
    res_r = jnp.concatenate((head_r, tail_r), axis=-1)
    res_i = jnp.concatenate((head_i, tail_i), axis=-1)
    return res_r + 1j * res_i
