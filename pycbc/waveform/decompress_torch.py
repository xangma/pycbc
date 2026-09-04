# Copyright (C) 2026  The PyCBC team
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

"""Torch interpolation backend for compressed frequency-domain waveforms."""

import numpy
import torch

from pycbc.types.backend import backend_array


_STENCIL_OFFSETS = {
    1: (0, 1),
    2: (-1, 0, 1),
    3: (-1, 0, 1, 2),
    4: (-1, 0, 1, 2, 3),
}


def _as_tensor(value, device, dtype):
    """Return ``value`` as a tensor on the output device."""
    value = backend_array(value)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(numpy.asarray(value), device=device, dtype=dtype)


def _lagrange_weights(nodes, points):
    """Evaluate Lagrange basis weights for each row of ``nodes``."""
    weights = []
    for column in range(nodes.shape[1]):
        numerator = torch.ones_like(points)
        denominator = torch.ones_like(points)
        for other in range(nodes.shape[1]):
            if other == column:
                continue
            numerator = numerator * (points - nodes[:, other])
            denominator = denominator * (
                nodes[:, column] - nodes[:, other]
            )
        weights.append(numerator / denominator)
    return torch.stack(weights, dim=1)


def _grid_indices(frequencies, df):
    """Truncate frequencies to output-grid indices like the CPU backend."""
    ratios = frequencies / df
    if ratios.dtype == torch.float64:
        # The optimized C++ reference treats a quotient that is only a few
        # double-precision ulps from an integer as that grid boundary.  Avoid
        # assigning such a point to the adjacent interpolation segment merely
        # because tensor division rounded in the opposite direction.
        nearest = torch.round(ratios)
        scale = torch.maximum(
            torch.abs(ratios), torch.ones_like(ratios)
        )
        tolerance = 8 * torch.finfo(ratios.dtype).eps * scale
        ratios = torch.where(
            torch.abs(ratios - nearest) <= tolerance,
            nearest,
            ratios,
        )
    return torch.trunc(ratios).to(torch.int64)


def _inline_interp(amp, phase, sample_frequencies, output, df, imin,
                   start_index, degree):
    """Interpolate amplitude and phase with CPU-backend stencil semantics."""
    out = backend_array(output, "torch")
    if out is None:
        raise TypeError("Torch decompression requires Torch-backed output")

    # The CPU implementation promotes interpolation arithmetic to double,
    # including for single-precision output. MPS cannot represent float64,
    # but CPU and CUDA follow that behavior exactly.
    input_dtype = (
        torch.float32 if out.dtype == torch.complex64 else torch.float64
    )
    calc_dtype = torch.float32 if out.device.type == "mps" else torch.float64
    # The single-precision Cython entry points accept ``df`` as a C float
    # before promoting it for the C++ interpolation arithmetic.  Preserve
    # that rounding here as it also determines whether the final grid point
    # is inside the half-open output interval.
    calc_df = (
        numpy.float32(df).item()
        if out.dtype == torch.complex64
        else float(df)
    )
    frequencies = _as_tensor(
        sample_frequencies, out.device, input_dtype
    ).to(calc_dtype)
    amplitudes = _as_tensor(amp, out.device, input_dtype).to(calc_dtype)
    phases = _as_tensor(phase, out.device, input_dtype).to(calc_dtype)

    out.zero_()
    sample_count = frequencies.numel()
    if sample_count < 2:
        return output

    output_indices = torch.arange(
        start=start_index,
        end=out.numel(),
        device=out.device,
        dtype=torch.int64,
    )
    last_index = _grid_indices(frequencies[-1], calc_df)
    output_indices = output_indices[output_indices <= last_index]
    if output_indices.numel() == 0:
        return output

    # Match the CPU segment boundaries exactly. For non-final segments the
    # compiled backend stops at int(f[i + 1] / df), while the final segment
    # includes that index. ``right=True`` assigns each boundary index to the
    # following segment, and the clamp restores the final-segment exception.
    segment_ends = _grid_indices(frequencies[1:], calc_df)
    segments = torch.searchsorted(
        segment_ends.contiguous(), output_indices, right=True
    )
    segments.clamp_(min=int(imin), max=sample_count - 2)

    max_degree = min(degree, sample_count - 1)
    degrees = torch.full_like(segments, max_degree)
    degrees.masked_fill_(segments == 0, 1)
    if max_degree > 3:
        degrees.masked_fill_(segments >= sample_count - 3, 3)
    if max_degree > 2:
        degrees.masked_fill_(segments >= sample_count - 2, 2)

    for current_degree in range(1, max_degree + 1):
        positions = torch.nonzero(
            degrees == current_degree, as_tuple=True
        )[0]
        if positions.numel() == 0:
            continue
        selected_segments = segments.index_select(0, positions)
        offsets = torch.tensor(
            _STENCIL_OFFSETS[current_degree],
            device=out.device,
            dtype=torch.int64,
        )
        stencil = selected_segments[:, None] + offsets[None, :]
        nodes = frequencies[stencil]
        points = output_indices.index_select(0, positions).to(calc_dtype)
        points = points * calc_df
        weights = _lagrange_weights(nodes, points)

        interp_amp = (amplitudes[stencil] * weights).sum(dim=1)
        interp_phase = (phases[stencil] * weights).sum(dim=1)
        waveform = torch.complex(
            interp_amp * torch.cos(interp_phase),
            interp_amp * torch.sin(interp_phase),
        ).to(dtype=out.dtype)
        out.index_copy_(
            0, output_indices.index_select(0, positions), waveform
        )

    return output


def inline_linear_interp(amp, phase, sample_frequencies, output,
                         df, f_lower, imin, start_index):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 1
    )


def inline_quadratic_interp(amp, phase, sample_frequencies, output,
                            df, f_lower, imin, start_index):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 2
    )


def inline_cubic_interp(amp, phase, sample_frequencies, output,
                        df, f_lower, imin, start_index):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 3
    )


def inline_quartic_interp(amp, phase, sample_frequencies, output,
                          df, f_lower, imin, start_index):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 4
    )
