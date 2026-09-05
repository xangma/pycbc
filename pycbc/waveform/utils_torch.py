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
    """Shift a uniformly sampled frequency-domain waveform in time.

    A non-scalar ``dt`` is aligned with the waveform's sample axes; the final
    data axis is always frequency. It cannot introduce a sample axis into a
    one-dimensional ``FrequencySeries``.

    ``copy=False`` updates the existing storage, including its aliases. As
    with Torch in-place operations, this cannot modify an autograd leaf or a
    view of one; use the default ``copy=True`` for those inputs.
    """
    data = htilde._data.tensor
    if copy:
        data = data.clone()

    if isinstance(dt, torch.Tensor):
        dt_value = dt.to(device=data.device, dtype=data.real.dtype)
    else:
        try:
            dt_value = float(dt)
        except (TypeError, ValueError):
            dt_value = torch.as_tensor(dt, device=data.device, dtype=data.real.dtype)

    if isinstance(dt_value, torch.Tensor) and dt_value.ndim:
        sample_shape = data.shape[:-1]
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(dt_value.shape), tuple(sample_shape)
            )
        except RuntimeError as exc:
            raise ValueError(
                "A batched time shift must broadcast across the waveform sample axes"
            ) from exc
        if tuple(broadcast_shape) != tuple(sample_shape):
            raise ValueError(
                "A batched time shift cannot introduce sample axes into a "
                "FrequencySeries"
            )
        dt_value = dt_value.unsqueeze(-1)

    kmax = data.shape[-1]
    if kmax > kmin:
        indices = _get_freq_grid(kmin, kmax, data.device, data.real.dtype)
        theta = (-2.0 * math.pi * dt_value * float(htilde.delta_f)) * indices
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        target = data[..., kmin:] if kmin > 0 else data
        # Torch saves the unshifted samples needed by backward. Splitting
        # this into real/imaginary views and copy_ invalidates those views.
        shift = torch.complex(cosine, sine)
        target.mul_(shift)

    if copy:
        htilde = FrequencySeries(
            TorchArrayData(data),
            delta_f=htilde.delta_f,
            epoch=htilde.epoch,
            copy=False,
        )
    return htilde
