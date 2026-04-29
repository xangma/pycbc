# Copyright (C) 2025
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

"""
Torch backend for chisq accumulation and shift_sum.
"""

import torch
from pycbc.types import Array
from pycbc.types.array import _convert_to_scheme


def chisq_accum_bin(chisq, q):
    """Accumulate |q|^2 into chisq."""
    _convert_to_scheme(chisq)
    _convert_to_scheme(q)
    chisq._data.tensor += torch.abs(q._data.tensor) ** 2


def shift_sum(corr, points, bins):
    """
    Calculate time-shifted sums of corr over provided bins at given points.
    """
    from pycbc.types.array_torch import TorchArrayData

    device = corr._data.tensor.device
    dtype = corr._data.tensor.dtype
    N = corr._data.tensor.shape[-1]

    pts = torch.as_tensor(points, device=device, dtype=torch.float64)
    nbins = len(bins) - 1
    out = torch.zeros((pts.numel(), nbins), device=device, dtype=dtype)

    for j in range(nbins):
        s, e = int(bins[j]), int(bins[j + 1])
        idx = torch.arange(s, e, device=device, dtype=torch.float64)
        phase = torch.exp(2j * torch.pi * pts[:, None] * idx[None, :] / float(N))
        phase = phase.to(dtype)
        sl = corr._data.tensor[s:e]
        out[:, j] = torch.sum(sl * phase, dim=1)

    chisq = torch.sum(torch.conj(out) * out, dim=1).real

    # Wrap tensor so Array sees torch-backed storage and avoids host copies
    return Array(TorchArrayData(chisq), copy=False)


def power_chisq_at_points_from_precomputed(corr, snr, snr_norm, bins, indices):
    """
    Torch implementation of power_chisq_at_points_from_precomputed that keeps
    all math on the active torch device to avoid numpy round‑trips.
    """
    from pycbc.types.array_torch import TorchArrayData

    _convert_to_scheme(corr)

    device = corr._data.tensor.device
    dtype = corr._data.tensor.dtype

    num_bins = len(bins) - 1

    chisq_arr = shift_sum(corr, indices, bins)  # already returns torch-backed Array
    chisq_t = chisq_arr._data.tensor

    if hasattr(snr, "_scheme"):
        _convert_to_scheme(snr)
        snr_t = snr._data.tensor
    else:
        snr_t = torch.as_tensor(snr, device=device, dtype=dtype)

    snr_term = (torch.conj(snr_t) * snr_t).real
    chisq_t = chisq_t * num_bins - snr_term

    snr_norm_t = torch.as_tensor(snr_norm, device=device, dtype=chisq_t.dtype)
    chisq_t = chisq_t * (snr_norm_t ** 2)

    return Array(TorchArrayData(chisq_t), copy=False)
