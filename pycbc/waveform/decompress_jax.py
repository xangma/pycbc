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

from pycbc.types.array_jax import JAXArrayData, _ensure_x64
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

    if not isinstance(frequencies, np.ndarray) or frequencies.dtype != real_dtype or not frequencies.flags.c_contiguous:
        frequencies = _host_array(frequencies, real_dtype)[:count]
    else:
        frequencies = frequencies[:count]

    if not isinstance(amp, np.ndarray) or amp.dtype != real_dtype or not amp.flags.c_contiguous:
        amp = _host_array(amp, real_dtype)[:count]
    else:
        amp = amp[:count]

    if not isinstance(phase, np.ndarray) or phase.dtype != real_dtype or not phase.flags.c_contiguous:
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
        raise ValueError("Batched decompression inputs must have equal lengths")

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
