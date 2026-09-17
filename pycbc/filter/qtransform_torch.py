# Copyright (C) 2025
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

"""
Torch-backed q-transform utilities.

This mirrors ``pycbc.filter.qtransform`` but keeps all heavy lifting on the
active torch device. CPU code paths remain unchanged; callers should dispatch
here only when working under the torch scheme.
"""

from math import sqrt

import numpy as np
import torch

from pycbc.fft import ifft
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.types.array_torch import TorchArrayData

# Bound the peak temporary storage used while evaluating a Q plane.  The
# conservative per-sample estimate covers the complex input and IFFT output,
# power, median-selection workspace, and the normalized output.  This gives 64
# rows for a 32768-sample transform while automatically using fewer rows for
# longer transforms.  A minimum of one row keeps a functional low-memory path.
_QPLANE_BATCH_WORKSPACE_BYTES = 128 * 1024**2
_QPLANE_BATCH_BYTES_PER_SAMPLE = 64


def _lin_interp_indices(old_grid, new_grid):
    """Return lower/upper indices and weights for 1D linear interpolation."""
    new_grid = new_grid.clamp(old_grid[0], old_grid[-1])
    idx_hi = torch.searchsorted(old_grid, new_grid, right=False)
    idx_hi = torch.clamp(idx_hi, 1, old_grid.numel() - 1)
    idx_lo = idx_hi - 1

    x0 = old_grid[idx_lo]
    x1 = old_grid[idx_hi]
    # Add a tiny epsilon to avoid divide-by-zero if grids repeat
    w = (new_grid - x0) / (x1 - x0 + 1e-30)
    return idx_lo, idx_hi, w


def _bilinear_interp(values, freqs_old, times_old, freqs_new, times_new):
    """Bilinear interpolation using the smaller intermediate grid."""
    # Linear interpolation is separable, so either operation order is valid.
    # Choosing the smaller intermediate is particularly important when a Q
    # plane is downsampled in time before being expanded in frequency.
    frequency_first_size = freqs_new.numel() * times_old.numel()
    time_first_size = freqs_old.numel() * times_new.numel()

    if time_first_size < frequency_first_size:
        t_lo, t_hi, tw = _lin_interp_indices(times_old, times_new)
        v0 = values[:, t_lo]
        v1 = values[:, t_hi]
        time_interp = torch.lerp(v0, v1, tw)

        f_lo, f_hi, fw = _lin_interp_indices(freqs_old, freqs_new)
        v0 = time_interp[f_lo, :]
        v1 = time_interp[f_hi, :]
        return torch.lerp(v0, v1, fw.unsqueeze(-1))

    f_lo, f_hi, fw = _lin_interp_indices(freqs_old, freqs_new)
    v0 = values[f_lo, :]
    v1 = values[f_hi, :]
    freq_interp = torch.lerp(v0, v1, fw.unsqueeze(-1))

    t_lo, t_hi, tw = _lin_interp_indices(times_old, times_new)
    v0 = freq_interp[:, t_lo]
    v1 = freq_interp[:, t_hi]
    return torch.lerp(v0, v1, tw)


def _qseries_batch_row_limit(tlen, tile_count):
    """Return a workspace-bounded number of Q tiles to evaluate together."""
    row_bytes = max(1, tlen) * _QPLANE_BATCH_BYTES_PER_SAMPLE
    workspace_rows = _QPLANE_BATCH_WORKSPACE_BYTES // row_bytes
    return max(1, min(tile_count, workspace_rows))


def _midpoint_median(values):
    """Return the NumPy-style median over the final dimension.

    Q-series normalization historically uses ``numpy.median``: even-sized
    inputs average the two central values, and any NaN propagates.  Selecting
    just those values avoids the full sort on CPU and CUDA.  Retain sorting on
    MPS, where ``kthvalue`` is not consistently faster for scalar Q tiles, and
    for autograd inputs so tied medians preserve the established full-sort
    gradient routing.
    """
    length = values.shape[-1]
    if length < 1:
        raise ValueError("median input must have a non-empty final dimension")

    if values.device.type == "mps" or values.requires_grad:
        ordered = torch.sort(values, dim=-1).values
        lower = ordered[..., (length - 1) // 2]
        if length % 2:
            median = lower
        else:
            median = 0.5 * (lower + ordered[..., length // 2])
    else:
        # torch.kthvalue uses one-based ranks.  For even lengths, select both
        # central values to preserve NumPy's midpoint convention exactly.
        lower = torch.kthvalue(values, (length + 1) // 2, dim=-1).values
        if length % 2:
            median = lower
        else:
            upper = torch.kthvalue(values, length // 2 + 1, dim=-1).values
            median = 0.5 * (lower + upper)

    # kthvalue, and median on some MPS releases, can ignore a NaN.  NumPy's
    # median propagates it, which is the established Q-transform contract.
    return median.masked_fill(torch.isnan(values).any(dim=-1), float("nan"))


def _qseries_batch(fseries, Q, frequencies, return_complex=False):
    """Evaluate a group of Q tiles with batched inverse FFTs and medians."""
    device = fseries._data.tensor.device

    # Keep MPS on the established complex64 implementation.  In addition to
    # lacking complex128, supported batched sort/FFT operations vary between
    # macOS releases, and this path is not a performance target here.
    if device.type == "mps":
        return torch.stack(
            [
                qseries(fseries, Q, f0, return_complex=return_complex)._data.tensor
                for f0 in frequencies
            ]
        )

    real_dtype = torch.float64
    complex_dtype = torch.complex128
    qprime = Q / sqrt(11.0)
    tlen = (len(fseries) - 1) * 2
    row_limit = _qseries_batch_row_limit(tlen, len(frequencies))
    result_dtype = complex_dtype if return_complex else real_dtype
    result = torch.empty((len(frequencies), tlen), device=device, dtype=result_dtype)

    for chunk_start in range(0, len(frequencies), row_limit):
        chunk = frequencies[chunk_start : chunk_start + row_limit]
        padded = torch.zeros((len(chunk), tlen), device=device, dtype=complex_dtype)

        for row, f0 in enumerate(chunk):
            norm = sqrt(315.0 * qprime / (128.0 * f0))
            window_size = 2 * int(f0 / qprime * fseries.duration) + 1
            xfrequencies = torch.linspace(
                -1.0, 1.0, window_size, device=device, dtype=real_dtype
            )
            weight = ((1 - xfrequencies**2) ** 2) * norm

            start = int((f0 - (f0 / qprime)) * fseries.duration)
            end = int(start + window_size)
            center = (start + end) // 2
            windowed = fseries._data.tensor[start:end].to(dtype=complex_dtype) * weight

            # This is exactly ``resize(tlen); roll(-center)`` without making
            # and rolling a separate full-length tensor for every tile.
            if center < window_size:
                leading = window_size - center
                padded[row, :leading] = windowed[center:]
                padded[row, tlen - center :] = windowed[:center]
            else:
                destination = tlen - center
                padded[row, destination : destination + window_size] = windowed

        del xfrequencies, weight, windowed

        # Match pycbc.fft.ifft's operation order: the Torch backend first
        # converts torch.fft.ifft to an unnormalized inverse, after which the
        # wrapper applies the FrequencySeries delta_f normalization in place.
        transformed = torch.fft.ifft(padded, n=tlen, dim=-1) * tlen
        transformed.mul_(fseries.delta_f)

        if return_complex:
            result[chunk_start : chunk_start + len(chunk)].copy_(transformed)
            del padded, transformed
            continue

        energy = (
            transformed.real * transformed.real + transformed.imag * transformed.imag
        )
        median_energy = _midpoint_median(energy)
        normalized = energy / median_energy.unsqueeze(-1)
        result[chunk_start : chunk_start + len(chunk)].copy_(normalized)
        del (padded, transformed, energy, median_energy, normalized)

    return result


def qseries(fseries, Q, f0, return_complex=False):
    """Torch equivalent of :func:`pycbc.filter.qtransform.qseries`."""
    device = fseries._data.tensor.device
    # The legacy NumPy implementation forms the window in float64 and always
    # performs the inverse FFT into a complex128 TimeSeries, even for a
    # complex64 input.  Preserve that public numerical behavior on devices
    # that support it.  MPS has no float64/complex128 kernels, so it retains a
    # documented single-precision compatibility path.
    if device.type == "mps":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
        output_dtype = np.complex64
    else:
        real_dtype = torch.float64
        complex_dtype = torch.complex128
        output_dtype = np.complex128

    qprime = Q / sqrt(11.0)
    norm = sqrt(315.0 * qprime / (128.0 * f0))
    window_size = 2 * int(f0 / qprime * fseries.duration) + 1
    xfrequencies = torch.linspace(
        -1.0, 1.0, window_size, device=device, dtype=real_dtype
    )
    weight = ((1 - xfrequencies**2) ** 2) * norm

    start = int((f0 - (f0 / qprime)) * fseries.duration)
    end = int(start + window_size)
    center = (start + end) // 2

    window_slice = fseries._data.tensor[start:end].to(dtype=complex_dtype)
    windowed = window_slice * weight

    tlen = (len(fseries) - 1) * 2
    padded = torch.zeros(tlen, device=device, dtype=complex_dtype)
    padded[: windowed.shape[0]] = windowed
    padded = torch.roll(padded, -center, dims=0)

    windowed_fs = FrequencySeries(
        TorchArrayData(padded),
        delta_f=fseries.delta_f,
        epoch=fseries.start_time,
        copy=False,
    )

    ctseries = TimeSeries(
        zeros(tlen, dtype=output_dtype),
        delta_t=fseries.delta_t,
        epoch=fseries.start_time,
    )
    ifft(windowed_fs, ctseries)

    if return_complex:
        return ctseries

    energy = ctseries.squared_norm()
    flat = energy._data.tensor.flatten()
    median_energy = _midpoint_median(flat)
    normalized = energy._data.tensor / median_energy
    return energy._return(TorchArrayData(normalized))


def qplane(qplane_tile_dict, fseries, return_complex=False):
    """Torch-backed analogue of :func:`pycbc.filter.qtransform.qplane`."""
    qplanes = {}
    qkeys = []
    qmaxima = []

    for i, q in enumerate(qplane_tile_dict):
        plane = _qseries_batch(
            fseries, q, qplane_tile_dict[q], return_complex=return_complex
        )
        if return_complex:
            tile_maxima = torch.abs(plane).amax(dim=1)
        else:
            tile_maxima = plane.amax(dim=1)
        qplanes[q] = plane
        qkeys.append(q)
        # Match the NumPy implementation's initialization behavior: every
        # tile in the first Q plane replaces the provisional maximum, so its
        # final tile supplies the comparison baseline for subsequent planes.
        qmaxima.append(tile_maxima[-1] if i == 0 else tile_maxima.amax())

    max_index = torch.stack(qmaxima).argmax().item()
    max_key = qkeys[max_index]

    plane_tensor = qplanes[max_key]
    frequencies = qplane_tile_dict[max_key]
    device = plane_tensor.device
    coordinate_dtype = torch.float32 if device.type == "mps" else torch.float64
    times = torch.arange(plane_tensor.shape[-1], device=device, dtype=coordinate_dtype)
    times.mul_(fseries.delta_t)
    if fseries.start_time is not None:
        times.add_(float(fseries.start_time))
    freqs_t = torch.as_tensor(frequencies, device=times.device, dtype=times.dtype)
    return max_key, times, freqs_t, plane_tensor


def interpolate_qplane(
    q_plane, times_old, freqs_old, times_new, freqs_new, return_complex=False
):
    """Interpolate a q-plane onto new time/frequency grids using torch."""
    if return_complex:
        amp = torch.abs(q_plane)
        phase = torch.angle(q_plane)
        amp_i = _bilinear_interp(amp, freqs_old, times_old, freqs_new, times_new)
        phase_i = _bilinear_interp(phase, freqs_old, times_old, freqs_new, times_new)
        return torch.exp(1.0j * phase_i) * amp_i

    return _bilinear_interp(q_plane, freqs_old, times_old, freqs_new, times_new)
