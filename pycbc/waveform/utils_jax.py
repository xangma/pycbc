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
from pycbc.types.array_jax import JAXArrayData, _reference_enabled
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
    out = out.copy()
    if _reference_enabled("td_taper"):
        import numpy as np
        from pycbc import scheme
        from pycbc.reference_jax import cpu_reference
        values, _, _ = cpu_reference(
            "td_taper", np.asarray(out), spacing=out.delta_t, epoch=out._epoch,
            start=start, end=end, beta=beta, side=side)
        out._data.set_array(jax.device_put(values, scheme.mgr.state.jax_device))
        return out
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
    out = out.copy()
    if _reference_enabled("fd_taper"):
        import numpy as np
        from pycbc import scheme
        from pycbc.reference_jax import cpu_reference
        values, _, _ = cpu_reference(
            "fd_taper", np.asarray(out), spacing=out.delta_f, epoch=out._epoch,
            start=start, end=end, beta=beta, side=side)
        out._data.set_array(jax.device_put(values, scheme.mgr.state.jax_device))
        return out
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


def fused_detector_strain_fd_jax(hp_tensor, hc_tensor, fp_list, fc_list, dt_list, delta_f, kmin=0):
    """Project polarizations and shift each detector on its waveform device."""
    from pycbc.types.array_jax import _reference_enabled, to_jax
    from pycbc.reference_jax import cpu_reference
    import jax
    import numpy as np
    like = hp_tensor
    device = None if isinstance(like, jax.core.Tracer) else like.device
    ndet = len(fp_list)
    if not ndet or len(fc_list) != ndet or len(dt_list) != ndet:
        raise ValueError("fplus, fcross, and time-shift values must have the same non-zero detector count")
    arrays = [to_jax(value, dtype=like.real.dtype, device=device)
              for group in (fp_list, fc_list, dt_list) for value in group]
    try:
        arrays = jnp.broadcast_arrays(*arrays)
    except (ValueError, TypeError) as error:
        raise ValueError("Detector responses and time shifts do not have compatible sample shapes") from error
    outputs = []
    frequencies = jnp.arange(like.shape[-1], dtype=like.real.dtype, device=device) * delta_f
    for index in range(ndet):
        fp, fc, dt = arrays[index], arrays[index+ndet], arrays[index+2*ndet]
        if _reference_enabled("inference_projection"):
            if isinstance(fp, jax.core.Tracer):
                raise RuntimeError("Native projection cannot run inside jax.jit")
            pairs = zip(np.asarray(fp).reshape(-1), np.asarray(fc).reshape(-1))
            projected = jnp.stack([to_jax(cpu_reference("inference_projection", np.asarray(like), hc=np.asarray(hc_tensor), fp=float(a), fc=float(b), spacing=delta_f), device=device) for a,b in pairs])
            projected = projected.reshape(fp.shape + like.shape)
        else:
            projected = fp[..., None] * like + fc[..., None] * to_jax(hc_tensor, device=device)
        if _reference_enabled("time_shift"):
            if isinstance(dt, jax.core.Tracer):
                raise RuntimeError("Native time shift cannot run inside jax.jit")
            result = cpu_reference("time_shift", np.asarray(projected), spacing=delta_f, dt=np.asarray(dt), kmin=kmin)[0]
            shifted = to_jax(result, device=device)
        else:
            phase = -2.0 * jnp.pi * dt[..., None] * frequencies
            shift = jnp.cos(phase) + 1j * jnp.sin(phase)
            shifted = projected * shift
            if kmin:
                shifted = jnp.concatenate((projected[..., :kmin], shifted[..., kmin:]), axis=-1)
        outputs.append(shifted)
    return jnp.stack(outputs)
