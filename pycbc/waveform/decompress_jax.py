# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""JAX interpolation backend for compressed frequency-domain waveforms."""

import numpy as np
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
    ratios = frequencies / df
    if ratios.dtype == jnp.float64:
        nearest = jnp.round(ratios)
        scale = jnp.maximum(jnp.abs(ratios), 1.0)
        tolerance = 8 * jnp.finfo(ratios.dtype).eps * scale
        ratios = jnp.where(
            jnp.abs(ratios - nearest) <= tolerance,
            nearest,
            ratios,
        )
    return jnp.trunc(ratios).astype(jnp.int64)


def _inline_interp(
    amp, phase, sample_frequencies, output, df, imin, start_index, degree
):
    """Interpolate amplitude and phase with CPU-backend stencil semantics."""
    _ensure_x64()
    out = backend_array(output, "jax")
    if out is None:
        raise TypeError("JAX decompression requires JAX-backed output")

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

    output_indices = jnp.arange(start_index, end_index, dtype=jnp.int64)

    segment_ends = _grid_indices(frequencies[1:], calc_df)
    segments = jnp.searchsorted(segment_ends, output_indices, side="right")
    segments = jnp.clip(segments, int(imin), sample_count - 2)

    target_array = jnp.zeros(len(out), dtype=out.dtype)

    if degree == 1:
        points = output_indices.astype(calc_dtype) * calc_df
        f0 = frequencies[segments]
        f1 = frequencies[segments + 1]
        w1 = (points - f0) / (f1 - f0)
        w0 = 1.0 - w1
        interp_amp = amplitudes[segments] * w0 + amplitudes[segments + 1] * w1
        interp_phase = phases[segments] * w0 + phases[segments + 1] * w1
        waveform = (
            interp_amp * jnp.cos(interp_phase)
            + 1j * interp_amp * jnp.sin(interp_phase)
        ).astype(out.dtype)
        target_array = target_array.at[start_index:start_index + len(waveform)].set(waveform)
    else:
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
