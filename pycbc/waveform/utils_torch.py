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

"""Torch-specific waveform utilities."""

import torch
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shifts a frequency domain waveform in time."""
    data = htilde._data.tensor
    if copy:
        data = data.clone()

    phi = -2j * torch.pi * dt * htilde.delta_f
    idx = torch.arange(
        kmin,
        data.shape[-1],
        device=data.device,
        dtype=data.real.dtype,
    )
    phase = torch.exp(phi * idx)
    data[kmin:] = data[kmin:] * phase

    if copy:
        wrapped = TorchArrayData(data)
        htilde = FrequencySeries(wrapped, delta_f=htilde.delta_f,
                                 epoch=htilde.epoch, copy=False)
    return htilde
