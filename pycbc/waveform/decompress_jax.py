# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""JAX-facing decompression for compressed frequency-domain waveforms.

Compressed templates are small, irregular inputs which expand to very large
dense arrays. Running a binary search and trigonometric interpolation for every
output bin is a poor GPU workload. PyCBC's native interpolators instead walk
each compressed segment once and define the reference numerical semantics.
This module stages a complete batch with those routines and makes one explicit
transfer to the active JAX device.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pycbc.types.array_jax import JAXArrayData, _ensure_x64, _reference_enabled
from pycbc.types.backend import backend_array
from .decompress_cpu_cython import (
    decomp_ccode_double,
    decomp_ccode_float,
    decomp_qcode_double,
    decomp_qcode_float,
    decomp_tcode_double,
    decomp_tcode_float,
    decomp_Qcode_double,
    decomp_Qcode_float,
)


_INTERPOLATORS = {
    "inline_linear": (decomp_ccode_float, decomp_ccode_double),
    "inline_quadratic": (decomp_qcode_float, decomp_qcode_double),
    "inline_cubic": (decomp_tcode_float, decomp_tcode_double),
    "inline_quartic": (decomp_Qcode_float, decomp_Qcode_double),
}


def _target_device():
    """Return the active JAX device, if execution is inside JAXScheme."""
    from pycbc import scheme

    state = getattr(scheme.mgr, "state", None)
    return getattr(state, "jax_device", None)


def _host_array(value, dtype):
    """Return contiguous host storage without relying on implicit transfers."""
    raw = backend_array(value, "jax")
    if raw is not None:
        value = jax.device_get(raw)
    return np.ascontiguousarray(value, dtype=dtype)


def _decompress_host_row_into(
    out_row,
    amp, phase, frequencies, imin, start, end, count, interpolation,
    df, out_len, dtype,
):
    """Run one native interpolation directly into an output host row."""
    if interpolation not in _INTERPOLATORS:
        supported = ", ".join(_INTERPOLATORS)
        raise NotImplementedError(
            f"JAX batched decompression supports {supported}; "
            f"got {interpolation!r}"
        )

    complex_dtype = np.dtype(dtype)
    if complex_dtype == np.dtype(np.complex64):
        real_dtype = np.float32
        helper = _INTERPOLATORS[interpolation][0]
    elif complex_dtype == np.dtype(np.complex128):
        real_dtype = np.float64
        helper = _INTERPOLATORS[interpolation][1]
    else:
        raise TypeError(f"Unsupported decompression dtype {complex_dtype}")

    count = min(int(count), len(amp), len(phase), len(frequencies))
    start = max(0, int(start))
    stop = min(int(out_len), int(end))
    if count < 2 or stop <= start or not len(out_row):
        return

    if (
        not isinstance(frequencies, np.ndarray)
        or frequencies.dtype != real_dtype
        or not frequencies.flags.c_contiguous
    ):
        frequencies = _host_array(frequencies, real_dtype)[:count]
    else:
        frequencies = frequencies[:count]

    if (
        not isinstance(amp, np.ndarray)
        or amp.dtype != real_dtype
        or not amp.flags.c_contiguous
    ):
        amp = _host_array(amp, real_dtype)[:count]
    else:
        amp = amp[:count]

    if (
        not isinstance(phase, np.ndarray)
        or phase.dtype != real_dtype
        or not phase.flags.c_contiguous
    ):
        phase = _host_array(phase, real_dtype)[:count]
    else:
        phase = phase[:count]

    imin = min(max(0, int(imin)), count - 2)
    helper(
        out_row, float(df), stop, start, frequencies, amp, phase, count, imin
    )


def _decompress_host_row(
    amp, phase, frequencies, imin, start, end, count, interpolation,
    df, out_len, dtype,
):
    """Run one native interpolation into a zero-initialized host row."""
    row = np.zeros(int(out_len), dtype=np.dtype(dtype))
    _decompress_host_row_into(
        row, amp, phase, frequencies, imin, start, end, count, interpolation,
        df, out_len, dtype,
    )
    return row


def batched_inline_interp_jax(
    amps_list, phases_list, freqs_list, imins, starts, ends, counts,
    interpolations, df, out_len, dtype=jnp.complex64,
):
    """Decompress a host-staged batch and transfer it once to a JAX device."""
    _, device = stage_batched_inline_interp_jax(
        amps_list, phases_list, freqs_list, imins, starts, ends, counts,
        interpolations, df, out_len, dtype=dtype,
    )
    return device


def stage_batched_inline_interp_jax(
    amps_list, phases_list, freqs_list, imins, starts, ends, counts,
    interpolations, df, out_len, dtype=jnp.complex64,
):
    """Return the native host batch and its single-copy device image."""
    _ensure_x64()
    batch_size = len(amps_list)
    fields = (
        phases_list, freqs_list, imins, starts, ends, counts, interpolations
    )
    if any(len(field) != batch_size for field in fields):
        raise ValueError(
            "Batched decompression inputs must have equal lengths"
        )

    complex_dtype = np.dtype(dtype)
    host = np.zeros((batch_size, int(out_len)), dtype=complex_dtype)
    for index in range(batch_size):
        _decompress_host_row_into(
            host[index],
            amps_list[index], phases_list[index], freqs_list[index],
            imins[index], starts[index], ends[index], counts[index],
            interpolations[index], df, out_len, complex_dtype,
        )
    target_dev = _target_device()
    if target_dev is not None:
        device = jax.device_put(host, target_dev)
    else:
        device = jax.device_put(host)
    return host, device


def _get_device_decompress_fn(flen, pad_k, df_float, dtype_str):
    """Return a JIT-compiled on-device batch linear decompression function."""
    flen = int(flen)
    pad_k = int(pad_k)
    df_float = float(df_float)
    complex_dtype = (
        jnp.complex64 if dtype_str == "complex64" else jnp.complex128
    )
    real_dtype = jnp.float32 if dtype_str == "complex64" else jnp.float64

    bins = jnp.arange(flen)
    freqs_eval = bins.astype(real_dtype) * df_float

    def _interp_single(f_knots, a_knots, p_knots, s_idx, e_idx, fend_val):
        valid = ((bins >= s_idx) & (bins < e_idx)
                 & (freqs_eval <= fend_val))
        a_eval = jnp.interp(freqs_eval, f_knots, a_knots, left=0.0, right=0.0)
        p_eval = jnp.interp(freqs_eval, f_knots, p_knots, left=0.0, right=0.0)
        h = jnp.where(
            valid,
            a_eval * (jnp.cos(p_eval) + 1j * jnp.sin(p_eval)),
            0.0 + 0.0j,
        ).astype(complex_dtype)
        return h

    @jax.jit
    def _decompress_batch(fb, ab, pb, starts_b, ends_b, fends_b):
        return jax.vmap(_interp_single)(
            fb, ab, pb, starts_b, ends_b, fends_b
        )

    return _decompress_batch


_DEVICE_DECOMPRESS_CACHE = {}


def stage_batched_device_interp_jax(
    amps_list, phases_list, freqs_list, starts, ends, counts,
    df, out_len, dtype=jnp.complex64, target_dev=None,
):
    """Decompress with direct device interpolation and native support bounds.

    Start/end indices are inclusive/exclusive; rows with fewer than two knots
    remain zero. Direct trigonometric evaluation differs from the host native
    interpolator's recurrence arithmetic.
    """
    if _reference_enabled('decompress'):
        imins = [max(0, int(np.searchsorted(
            freq[:min(int(count), len(freq))], max(0, start) * df,
            side='right')) - 1) for freq, start, count in zip(freqs_list, starts, counts)]
        return stage_batched_inline_interp_jax(
            amps_list, phases_list, freqs_list, imins, starts, ends, counts,
            ['inline_linear'] * len(amps_list), df, out_len, dtype=dtype)
    _ensure_x64()
    batch_size = len(amps_list)
    fields = (phases_list, freqs_list, starts, ends, counts)
    if any(len(field) != batch_size for field in fields):
        raise ValueError("Batched decompression inputs must have equal lengths")
    if batch_size == 0:
        return None, None

    row_counts = [
        min(int(count), len(amp), len(phase), len(freq))
        for count, amp, phase, freq in zip(
            counts, amps_list, phases_list, freqs_list
        )
    ]
    max_k = max(row_counts)
    pad_k = 2048 if max_k <= 2048 else ((max_k + 511) // 512) * 512

    complex_dtype = np.dtype(dtype)
    if complex_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported decompression dtype {complex_dtype}")
    is_c64 = complex_dtype == np.dtype(np.complex64)
    real_dtype = np.float32 if is_c64 else np.float64

    fb = np.zeros((batch_size, pad_k), dtype=real_dtype)
    ab = np.zeros((batch_size, pad_k), dtype=real_dtype)
    pb = np.zeros((batch_size, pad_k), dtype=real_dtype)
    starts_arr = np.maximum(np.asarray(starts, dtype=np.int32), 0)
    ends_arr = np.minimum(np.asarray(ends, dtype=np.int32), int(out_len))
    fends_arr = np.full(batch_size, -np.inf, dtype=real_dtype)

    df_float = float(df)
    for i in range(batch_size):
        cnt = row_counts[i]
        if cnt < 2 or ends_arr[i] <= starts_arr[i]:
            # Keep interpolation inputs well-defined even for a masked row.
            fb[i] = np.arange(pad_k, dtype=real_dtype) * 1e4
            continue
        ab[i, :cnt] = amps_list[i][:cnt]
        pb[i, :cnt] = phases_list[i][:cnt]
        freq_i = freqs_list[i][:cnt]
        fb[i, :cnt] = freq_i
        last_f = float(freq_i[-1])
        fends_arr[i] = last_f
        tail = np.arange(1, pad_k - cnt + 1, dtype=real_dtype) * 1e4
        fb[i, cnt:] = last_f + tail

    if target_dev is None:
        target_dev = _target_device()

    if target_dev is not None:
        fb_dev = jax.device_put(fb, target_dev)
        ab_dev = jax.device_put(ab, target_dev)
        pb_dev = jax.device_put(pb, target_dev)
        starts_dev = jax.device_put(starts_arr, target_dev)
        ends_dev = jax.device_put(ends_arr, target_dev)
        fends_dev = jax.device_put(fends_arr, target_dev)
    else:
        fb_dev = jnp.asarray(fb)
        ab_dev = jnp.asarray(ab)
        pb_dev = jnp.asarray(pb)
        starts_dev = jnp.asarray(starts_arr)
        ends_dev = jnp.asarray(ends_arr)
        fends_dev = jnp.asarray(fends_arr)

    cache_key = (int(out_len), int(pad_k), df_float, str(complex_dtype))
    if cache_key not in _DEVICE_DECOMPRESS_CACHE:
        _DEVICE_DECOMPRESS_CACHE[cache_key] = _get_device_decompress_fn(
            *cache_key
        )

    fn = _DEVICE_DECOMPRESS_CACHE[cache_key]
    device_waveforms = fn(
        fb_dev, ab_dev, pb_dev, starts_dev, ends_dev, fends_dev
    )
    return None, device_waveforms


def batched_inline_linear_interp_jax(
    amps_list, phases_list, freqs_list, imins, starts, ends, counts,
    df, out_len, dtype=jnp.complex64,
):
    """Decompress a batch with native linear semantics onto a JAX device."""
    return batched_inline_interp_jax(
        amps_list, phases_list, freqs_list, imins, starts, ends, counts,
        ["inline_linear"] * len(amps_list), df, out_len, dtype=dtype,
    )


def _inline_interp(
    amp, phase, sample_frequencies, output, df, imin, start_index,
    interpolation,
):
    """Populate a JAX-backed output using the native reference interpolator."""
    out = backend_array(output, "jax")
    if out is None:
        raise TypeError("JAX decompression requires JAX-backed output")

    frequencies = _host_array(
        sample_frequencies,
        np.float32 if out.dtype == jnp.complex64 else np.float64,
    )
    host = _decompress_host_row(
        amp, phase, frequencies, imin, start_index, len(out), len(frequencies),
        interpolation, df, len(out), out.dtype,
    )
    result = jax.device_put(host, _target_device())
    if hasattr(output, "_data") and isinstance(output._data, JAXArrayData):
        output._data.set_array(result)
    elif hasattr(output, "data") and isinstance(output.data, JAXArrayData):
        output.data.set_array(result)
    return output


def inline_linear_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index,
        "inline_linear",
    )


def inline_quadratic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index,
        "inline_quadratic",
    )


def inline_cubic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index,
        "inline_cubic",
    )


def inline_quartic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index,
        "inline_quartic",
    )


def fd_decompress_jax(amp, phase, sample_frequencies, out=None, df=None,
                      f_lower=None, interpolation='inline_linear'):
    """Preserve the native decompression API while owning JAX output storage."""
    from pycbc.types import FrequencySeries
    from pycbc.reference_jax import cpu_reference
    _ensure_x64()
    amp, phase, frequencies = (np.asarray(value) for value in
                               (amp, phase, sample_frequencies))
    if _reference_enabled('decompress'):
        values, spacing, epoch = cpu_reference(
            'decompress', None if out is None else np.asarray(out),
            spacing=1. if out is None else out.delta_f,
            epoch=None if out is None else out._epoch,
            amp=amp, phase=phase, sample_frequencies=frequencies,
            df=df, f_lower=f_lower, interpolation=interpolation)
        data = jax.device_put(values, _target_device())
        if out is not None:
            out._data.set_array(data)
            return out
        return FrequencySeries(JAXArrayData(data), delta_f=spacing,
                               epoch=epoch, copy=False)
    real_dtype = frequencies.dtype
    if real_dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise KeyError(real_dtype.name)
    if amp.dtype != real_dtype or phase.dtype != real_dtype:
        raise ValueError('amp, phase, and sample_points must all have the same precision')
    if out is None:
        if df is None:
            raise ValueError('Either provide output memory or a df')
        length = int(np.ceil(frequencies.max() / df + 1))
        dtype = np.complex64 if real_dtype == np.dtype('float32') else np.complex128
        data = jax.device_put(np.zeros(length, dtype=dtype), _target_device())
        out = FrequencySeries(JAXArrayData(data), delta_f=df, copy=False)
    else:
        real_dtype = np.float32 if out.dtype == np.dtype('complex64') else np.float64
        amp, phase, frequencies = (value.astype(real_dtype) for value in
                                   (amp, phase, frequencies))
        df = out.delta_f
    if f_lower is None:
        imin, start = 0, 0
    else:
        if f_lower >= frequencies.max():
            raise ValueError('f_lower is > than the maximum sample frequency')
        if f_lower < frequencies.min():
            raise ValueError('f_lower is < than the minimum sample frequency')
        imin = int(np.searchsorted(frequencies, f_lower, side='right')) - 1
        start = int(np.ceil(f_lower / df))
    if start >= len(out):
        raise ValueError('requested f_lower >= largest frequency in out')
    return _inline_interp(amp, phase, frequencies, out, df, imin, start,
                          interpolation)
