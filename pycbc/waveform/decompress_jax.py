# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""JAX interpolation backend for compressed frequency-domain waveforms."""

import functools
import numpy as np
import jax
import jax.numpy as jnp

from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax
from pycbc.types.backend import backend_array

_STENCIL_OFFSETS = {
    1: (0, 1),
    2: (-1, 0, 1),
    3: (-1, 0, 1, 2),
    4: (-1, 0, 1, 2, 3),
}


def _lagrange_weights(nodes, points):
    """Evaluate Lagrange basis weights for each row of ``nodes``."""
    weights = []
    num_cols = nodes.shape[1]
    for col in range(num_cols):
        num = jnp.ones_like(points)
        den = jnp.ones_like(points)
        for other in range(num_cols):
            if other == col:
                continue
            num = num * (points - nodes[:, other])
            den = den * (nodes[:, col] - nodes[:, other])
        weights.append(num / den)
    return jnp.stack(weights, axis=1)


def _grid_indices(frequencies, df):
    """Truncate frequencies to output-grid indices like the CPU backend."""
    ratios = np.asarray(frequencies) / df
    if ratios.dtype == np.float64:
        nearest = np.round(ratios)
        scale = np.maximum(np.abs(ratios), 1.0)
        tolerance = 8 * np.finfo(ratios.dtype).eps * scale
        ratios = np.where(
            np.abs(ratios - nearest) <= tolerance,
            nearest,
            ratios,
        )
    return np.trunc(ratios).astype(np.int64)


try:
    from .decompress_cpu_cython import (
        decomp_ccode_float,
        decomp_ccode_double,
    )
except (ImportError, OSError):
    decomp_ccode_float = None
    decomp_ccode_double = None


@functools.partial(jax.jit, static_argnames=("out_len", "dtype"))
def _decompress_batch_core(
    amps, phases, freqs, imins, starts, ends, counts, df, out_len, dtype
):
    """JIT-compiled batched linear interpolation kernel for waveforms on GPU."""
    out_idx = jnp.arange(out_len, dtype=jnp.int64)
    points = out_idx.astype(jnp.float64) * df

    def interp_one(amp, phase, freq, imin, start_idx, end_idx, count):
        ratios = freq[1:] / df
        nearest = jnp.round(ratios)
        scale = jnp.maximum(jnp.abs(ratios), 1.0)
        tol = 8 * jnp.finfo(jnp.float64).eps * scale
        ratios = jnp.where(jnp.abs(ratios - nearest) <= tol, nearest, ratios)
        seg_ends = jnp.trunc(ratios).astype(jnp.int64)

        segs = jnp.searchsorted(seg_ends, out_idx, side="right")
        segs = jnp.clip(segs, imin, count - 2)

        f0 = freq[segs]
        f1 = freq[segs + 1]
        denom = jnp.maximum(f1 - f0, 1e-12)
        w1 = (points - f0) / denom
        w0 = 1.0 - w1

        ai = amp[segs] * w0 + amp[segs + 1] * w1
        pi = phase[segs] * w0 + phase[segs + 1] * w1

        wf = (ai * jnp.cos(pi) + 1j * ai * jnp.sin(pi)).astype(dtype)
        valid = (out_idx >= start_idx) & (out_idx < end_idx)
        return jnp.where(valid, wf, 0.0)

    return jax.vmap(interp_one)(amps, phases, freqs, imins, starts, ends, counts)


def batched_inline_linear_interp_jax(
    amps_list, phases_list, freqs_list, imins, starts, ends, counts, df, out_len, dtype=jnp.complex64, pad_len=2048
):
    """Decompress an entire batch of waveforms on GPU in parallel with bitwise parity."""
    _ensure_x64()
    b = len(amps_list)
    if b == 0:
        return jnp.zeros((0, out_len), dtype=dtype)

    max_k = max(max(counts), 1)
    k_pad = max(pad_len, int(2 ** np.ceil(np.log2(max(max_k, 2)))))

    amps_padded = np.zeros((b, k_pad), dtype=np.float64)
    phases_padded = np.zeros((b, k_pad), dtype=np.float64)
    freqs_padded = np.full((b, k_pad), 1e9, dtype=np.float64)

    for i in range(b):
        c = int(counts[i])
        amps_padded[i, :c] = np.asarray(amps_list[i], dtype=np.float64)
        phases_padded[i, :c] = np.asarray(phases_list[i], dtype=np.float64)
        freqs_padded[i, :c] = np.asarray(freqs_list[i], dtype=np.float64)

    calc_df = np.float32(df).item() if dtype == jnp.complex64 else float(df)

    res = _decompress_batch_core(
        jnp.asarray(amps_padded),
        jnp.asarray(phases_padded),
        jnp.asarray(freqs_padded),
        jnp.asarray(imins, dtype=jnp.int64),
        jnp.asarray(starts, dtype=jnp.int64),
        jnp.asarray(ends, dtype=jnp.int64),
        jnp.asarray(counts, dtype=jnp.int64),
        calc_df,
        int(out_len),
        dtype,
    )
    return res


def _inline_interp(
    amp, phase, sample_frequencies, output, df, imin, start_index, degree
):
    """Interpolate amplitude and phase with CPU-backend stencil semantics."""
    _ensure_x64()
    out = backend_array(output, "jax")
    if out is None:
        raise TypeError("JAX decompression requires JAX-backed output")

    dev = out.device() if callable(getattr(out, "device", None)) else getattr(out, "device", None)
    is_cpu = getattr(dev, "platform", None) == "cpu"
    if not is_cpu and hasattr(out, "devices"):
        devs = list(out.devices()) if callable(getattr(out, "devices", None)) else getattr(out, "devices", [])
        is_cpu = all(getattr(d, "platform", None) == "cpu" for d in devs) if devs else True

    if is_cpu and degree == 1 and (
        decomp_ccode_float is not None or decomp_ccode_double is not None
    ):
        rprec = np.float32 if out.dtype == jnp.complex64 else np.float64
        cprec = np.complex64 if out.dtype == jnp.complex64 else np.complex128
        h_host = np.zeros(len(out), dtype=cprec)
        sf_host = np.asarray(sample_frequencies, dtype=rprec)
        amp_host = np.asarray(amp, dtype=rprec)
        phase_host = np.asarray(phase, dtype=rprec)
        delta_f = float(df)
        if out.dtype == jnp.complex64 and decomp_ccode_float is not None:
            decomp_ccode_float(
                h_host, delta_f, len(out), int(start_index), sf_host,
                amp_host, phase_host, len(sf_host), int(imin)
            )
        elif decomp_ccode_double is not None:
            decomp_ccode_double(
                h_host, delta_f, len(out), int(start_index), sf_host,
                amp_host, phase_host, len(sf_host), int(imin)
            )
        else:
            h_host = None

        if h_host is not None:
            if (
                hasattr(output, "_data")
                and isinstance(output._data, JAXArrayData)
            ):
                output._data.set_array(jnp.asarray(h_host))
            elif (
                hasattr(output, "data")
                and isinstance(output.data, JAXArrayData)
            ):
                output.data.set_array(jnp.asarray(h_host))
            return output

    input_dtype = jnp.float32 if out.dtype == jnp.complex64 else jnp.float64
    calc_dtype = jnp.float64
    calc_df = np.float32(df).item() if out.dtype == jnp.complex64 else float(df)

    frequencies = to_jax(sample_frequencies).astype(input_dtype).astype(calc_dtype)
    amplitudes = to_jax(amp).astype(input_dtype).astype(calc_dtype)
    phases = to_jax(phase).astype(input_dtype).astype(calc_dtype)

    sample_count = len(frequencies)
    if sample_count < 2:
        return output

    last_freq = frequencies[-1]
    last_index = int(_grid_indices(last_freq, calc_df))
    end_index = min(len(out), last_index + 1)
    if end_index <= start_index:
        return output

    if degree == 1:
        res = batched_inline_linear_interp_jax(
            [amplitudes], [phases], [frequencies],
            [int(imin)], [int(start_index)], [int(end_index)], [int(sample_count)],
            calc_df, len(out), out.dtype
        )[0]
        if hasattr(output, "_data") and isinstance(output._data, JAXArrayData):
            output._data.set_array(res)
        elif hasattr(output, "data") and isinstance(output.data, JAXArrayData):
            output.data.set_array(res)
        return output

    output_indices = jnp.arange(start_index, end_index, dtype=jnp.int64)

    segment_ends = _grid_indices(frequencies[1:], calc_df)
    segments = jnp.searchsorted(segment_ends, output_indices, side="right")
    segments = jnp.clip(segments, int(imin), sample_count - 2)

    target_array = jnp.zeros(len(out), dtype=out.dtype)
    max_degree = min(degree, sample_count - 1)
    degrees = jnp.full(segments.shape, max_degree, dtype=jnp.int32)
    degrees = jnp.where(segments == 0, 1, degrees)
    if max_degree > 3:
        degrees = jnp.where(segments >= sample_count - 3, 3, degrees)
    if max_degree > 2:
        degrees = jnp.where(segments >= sample_count - 2, 2, degrees)

    for current_degree in range(1, max_degree + 1):
        mask = (degrees == current_degree)
        if not bool(jnp.any(mask)):
            continue
        cur_segments = segments[mask]
        cur_indices = output_indices[mask]
        offsets = jnp.array(_STENCIL_OFFSETS[current_degree], dtype=jnp.int64)
        stencil = cur_segments[:, None] + offsets[None, :]
        nodes = frequencies[stencil]
        cur_points = cur_indices.astype(calc_dtype) * calc_df
        weights = _lagrange_weights(nodes, cur_points)

        interp_amp = jnp.sum(amplitudes[stencil] * weights, axis=1)
        interp_phase = jnp.sum(phases[stencil] * weights, axis=1)
        wf = (
            interp_amp * jnp.cos(interp_phase)
            + 1j * interp_amp * jnp.sin(interp_phase)
        ).astype(out.dtype)
        target_array = target_array.at[cur_indices].set(wf)

    if hasattr(output, "_data") and isinstance(output._data, JAXArrayData):
        output._data.set_array(target_array)
    return output


def inline_linear_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 1
    )


def inline_quadratic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 2
    )


def inline_cubic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 3
    )


def inline_quartic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 4
    )
