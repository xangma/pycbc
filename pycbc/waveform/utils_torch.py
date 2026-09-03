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
            dt_value = torch.as_tensor(
                dt, device=data.device, dtype=data.real.dtype
            )

    if isinstance(dt_value, torch.Tensor) and dt_value.ndim:
        sample_shape = data.shape[:-1]
        try:
            broadcast_shape = torch.broadcast_shapes(
                tuple(dt_value.shape), tuple(sample_shape)
            )
        except RuntimeError as exc:
            raise ValueError(
                "A batched time shift must broadcast across the waveform "
                "sample axes"
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
        real = target.real
        imag = target.imag
        target.copy_(torch.complex(
            real * cosine - imag * sine,
            real * sine + imag * cosine,
        ))

    if copy:
        htilde = FrequencySeries(
            TorchArrayData(data),
            delta_f=htilde.delta_f,
            epoch=htilde.epoch,
            copy=False,
        )
    return htilde


def fused_detector_strain_fd_torch(
    hp_tensor, hc_tensor, fp_list, fc_list, dt_list, delta_f, kmin=0
):
    """Compute detector projection and time shifts in one operation.

    Returns a complex tensor with shape
    ``(detectors, *sample_shape, frequencies)``. Scalar detector responses
    omit ``sample_shape``.
    """
    device = hp_tensor.device
    real_dtype = hp_tensor.real.dtype
    ndet = len(fp_list)
    if ndet == 0 or len(fc_list) != ndet or len(dt_list) != ndet:
        raise ValueError(
            "fplus, fcross, and time-shift values must have the same "
            "non-zero detector count"
        )

    detector_values = [
        torch.as_tensor(value, device=device, dtype=real_dtype)
        for values in (fp_list, fc_list, dt_list)
        for value in values
    ]
    try:
        detector_values = torch.broadcast_tensors(*detector_values)
    except RuntimeError as exc:
        raise ValueError(
            "Detector responses and time shifts do not have compatible "
            "sample shapes"
        ) from exc
    fp = torch.stack(detector_values[:ndet], dim=0)
    fc = torch.stack(detector_values[ndet:2 * ndet], dim=0)
    dt = torch.stack(detector_values[2 * ndet:], dim=0)

    # The first axis identifies detectors; every remaining axis identifies
    # samples.  The final singleton is always the frequency axis, even when a
    # sample dimension happens to equal the number of frequency bins.
    fp = fp.unsqueeze(-1)
    fc = fc.unsqueeze(-1)
    dt = dt.unsqueeze(-1)

    kmax = hp_tensor.shape[-1]
    out = fp * hp_tensor + fc * hc_tensor

    if kmax <= kmin:
        return out

    indices = _get_freq_grid(kmin, kmax, device, real_dtype)
    theta = (-2.0 * math.pi * float(delta_f) * dt) * indices
    shift = torch.complex(torch.cos(theta), torch.sin(theta))

    if kmin == 0:
        return out * shift

    return torch.cat((out[..., :kmin], out[..., kmin:] * shift), dim=-1)
