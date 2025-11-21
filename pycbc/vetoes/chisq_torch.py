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

    return Array(out, copy=False)
