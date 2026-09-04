# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Torch-specific waveform utilities."""

import math

import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

_FREQ_GRID_CACHE = {}
_FREQ_GRID_CACHE_MAXSIZE = 64


def _get_freq_grid(kmin, kmax, device, dtype):
    """Retrieve or allocate a cached one-dimensional frequency grid."""
    key = (kmin, kmax, torch.device(device), dtype)
    grid = _FREQ_GRID_CACHE.get(key)
    if grid is None:
        grid = torch.arange(kmin, kmax, device=device, dtype=dtype)
        if len(_FREQ_GRID_CACHE) >= _FREQ_GRID_CACHE_MAXSIZE:
            _FREQ_GRID_CACHE.clear()
        _FREQ_GRID_CACHE[key] = grid
    return grid


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shift a uniformly sampled frequency-domain waveform in time."""
    data = htilde._data.tensor
    if copy:
        data = data.clone()

    if not isinstance(dt, torch.Tensor):
        dt_value = float(dt)
    else:
        dt_value = float(dt.item()) if dt.numel() == 1 else dt

    kmax = data.shape[-1]
    if kmax > kmin:
        indices = _get_freq_grid(kmin, kmax, data.device, data.real.dtype)
        theta = (-2.0 * math.pi * dt_value * float(htilde.delta_f)) * indices
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        target = data[..., kmin:] if kmin > 0 else data
        real = target.real
        imag = target.imag
        target.copy_(
            torch.complex(
                real * cosine - imag * sine,
                real * sine + imag * cosine,
            )
        )

    if copy:
        htilde = FrequencySeries(
            TorchArrayData(data),
            delta_f=htilde.delta_f,
            epoch=htilde.epoch,
            copy=False,
        )
    return htilde
