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

import torch

from pycbc.fft import ifft
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.types.array_torch import TorchArrayData


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
    """Bilinear interpolation (freqs first, then times) on a regular grid."""
    # Interpolate along frequency axis (axis 0)
    f_lo, f_hi, fw = _lin_interp_indices(freqs_old, freqs_new)
    v0 = values[f_lo, :]
    v1 = values[f_hi, :]
    freq_interp = v0 + (v1 - v0) * fw.unsqueeze(-1)

    # Interpolate along time axis (axis 1)
    t_lo, t_hi, tw = _lin_interp_indices(times_old, times_new)
    v0 = freq_interp[:, t_lo]
    v1 = freq_interp[:, t_hi]
    return v0 + (v1 - v0) * tw


def qseries(fseries, Q, f0, return_complex=False):
    """Torch equivalent of :func:`pycbc.filter.qtransform.qseries`."""
    device = fseries._data.tensor.device
    dtype = fseries._data.tensor.dtype
    real_dtype = fseries._data.tensor.real.dtype

    qprime = Q / sqrt(11.0)
    norm = sqrt(315.0 * qprime / (128.0 * f0))
    window_size = 2 * int(f0 / qprime * fseries.duration) + 1
    xfrequencies = torch.linspace(-1.0, 1.0, window_size, device=device,
                                  dtype=torch.float64)
    weight = ((1 - xfrequencies ** 2) ** 2) * norm
    weight = weight.to(dtype=real_dtype)

    start = int((f0 - (f0 / qprime)) * fseries.duration)
    end = int(start + window_size)
    center = (start + end) // 2

    window_slice = fseries._data.tensor[start:end]
    windowed = window_slice * weight

    tlen = (len(fseries) - 1) * 2
    padded = torch.zeros(tlen, device=device, dtype=dtype)
    padded[:windowed.shape[0]] = windowed
    padded = torch.roll(padded, -center, dims=0)

    windowed_fs = FrequencySeries(TorchArrayData(padded),
                                  delta_f=fseries.delta_f,
                                  epoch=fseries.start_time,
                                  copy=False)

    ctseries = TimeSeries(zeros(tlen, dtype=fseries.dtype),
                          delta_t=fseries.delta_t,
                          epoch=fseries.start_time)
    ifft(windowed_fs, ctseries)

    if return_complex:
        return ctseries

    energy = ctseries.squared_norm()
    flat = energy._data.tensor.flatten()
    sorted_vals, _ = torch.sort(flat)
    n = sorted_vals.numel()
    if n % 2:
        median_energy = sorted_vals[n // 2]
    else:
        median_energy = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    return energy / float(median_energy.item())


def qplane(qplane_tile_dict, fseries, return_complex=False):
    """Torch-backed analogue of :func:`pycbc.filter.qtransform.qplane`."""
    qplanes = {}
    max_energy, max_key = None, None

    for i, q in enumerate(qplane_tile_dict):
        energies = []
        for f0 in qplane_tile_dict[q]:
            energy = qseries(fseries, q, f0, return_complex=return_complex)
            menergy = torch.abs(energy._data.tensor).max().item()
            energies.append(energy)

            if i == 0 or menergy > max_energy:
                max_energy = menergy
                max_key = q

        qplanes[q] = energies

    plane = qplanes[max_key]
    frequencies = qplane_tile_dict[max_key]
    times = plane[0].sample_times._data.tensor
    plane_tensor = torch.stack([v._data.tensor for v in plane], dim=0)
    freqs_t = torch.as_tensor(frequencies, device=times.device,
                              dtype=torch.float64)
    return max_key, times, freqs_t, plane_tensor


def interpolate_qplane(q_plane, times_old, freqs_old, times_new, freqs_new,
                       return_complex=False):
    """Interpolate a q-plane onto new time/frequency grids using torch."""
    if return_complex:
        amp = torch.abs(q_plane)
        phase = torch.angle(q_plane)
        amp_i = _bilinear_interp(amp, freqs_old, times_old, freqs_new, times_new)
        phase_i = _bilinear_interp(phase, freqs_old, times_old,
                                   freqs_new, times_new)
        return torch.exp(1.0j * phase_i) * amp_i

    return _bilinear_interp(q_plane, freqs_old, times_old, freqs_new, times_new)
