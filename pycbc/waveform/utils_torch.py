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

import math
import torch
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


_FREQ_GRID_CACHE = {}


def _get_freq_grid(kmin, kmax, device, dtype):
    """Retrieve or allocate a cached 1D frequency grid tensor."""
    key = (kmin, kmax, torch.device(device), dtype)
    grid = _FREQ_GRID_CACHE.get(key)
    if grid is None:
        grid = torch.arange(kmin, kmax, device=device, dtype=dtype)
        _FREQ_GRID_CACHE[key] = grid
    return grid


def apply_fseries_time_shift(htilde, dt, kmin=0, copy=True):
    """Shifts a frequency domain waveform in time."""
    data = htilde._data.tensor
    if copy:
        data = data.clone()

    # FrequencySeries.cyclic_time_shift also accepts lal.LIGOTimeGPS values.
    # Keep Torch scalar shifts on-device, but normalize other numeric scalar
    # types before combining them with Torch's complex phase coefficient.
    if not isinstance(dt, torch.Tensor):
        dt_val = float(dt)
    else:
        dt_val = float(dt.item()) if dt.numel() == 1 else dt

    kmax = data.shape[-1]
    if kmax > kmin:
        idx = _get_freq_grid(kmin, kmax, data.device, data.real.dtype)
        theta = (-2.0 * math.pi * dt_val * float(htilde.delta_f)) * idx
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        target = data[..., kmin:] if kmin > 0 else data
        re = target.real
        im = target.imag
        new_target = torch.complex(
            re * cos_t - im * sin_t, re * sin_t + im * cos_t
        )
        target.copy_(new_target)

    if copy:
        wrapped = TorchArrayData(data)
        htilde = FrequencySeries(wrapped, delta_f=htilde.delta_f,
                                 epoch=htilde.epoch, copy=False)
    return htilde


def fused_detector_strain_fd_torch(
    hp_tensor, hc_tensor, fp_list, fc_list, dt_list, delta_f, kmin=0
):
    """Computes multi-detector strain in a single broadcasted operation.

    Computes ``(F_+ h_+ + F_\times h_\times) * exp(-2*pi*i*f*dt)`` across
    all detectors and frequency bins simultaneously on the Torch device.

    Parameters
    ----------
    hp_tensor : torch.Tensor
        1D complex tensor of plus polarization frequency series data.
    hc_tensor : torch.Tensor
        1D complex tensor of cross polarization frequency series data.
    fp_list : list, tuple, or torch.Tensor
        Plus antenna response factors for each detector.
    fc_list : list, tuple, or torch.Tensor
        Cross antenna response factors for each detector.
    dt_list : list, tuple, or torch.Tensor
        Time offsets for each detector.
    delta_f : float
        Frequency step size in Hz.
    kmin : int, optional
        Starting frequency index. Default is 0.

    Returns
    -------
    torch.Tensor
        2D complex tensor of shape ``(D, N)`` containing the frequency-domain
        strain for each detector.
    """
    device = hp_tensor.device
    real_dtype = hp_tensor.real.dtype
    fp = torch.as_tensor(fp_list, device=device, dtype=real_dtype)
    fc = torch.as_tensor(fc_list, device=device, dtype=real_dtype)
    dt = torch.as_tensor(dt_list, device=device, dtype=real_dtype)

    if fp.ndim == 1:
        fp = fp.unsqueeze(1)
        fc = fc.unsqueeze(1)
        dt = dt.unsqueeze(1)

    kmax = hp_tensor.shape[-1]
    re = fp * hp_tensor.real + fc * hc_tensor.real
    im = fp * hp_tensor.imag + fc * hc_tensor.imag

    if kmax <= kmin:
        return torch.complex(re, im)

    idx = _get_freq_grid(kmin, kmax, device, real_dtype).unsqueeze(0)
    theta = (-2.0 * math.pi * float(delta_f)) * (dt * idx)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    if kmin == 0:
        return torch.complex(re * cos_t - im * sin_t, re * sin_t + im * cos_t)

    out = torch.complex(re, im)
    re_act = re[:, kmin:]
    im_act = im[:, kmin:]
    out[:, kmin:] = torch.complex(
        re_act * cos_t - im_act * sin_t, re_act * sin_t + im_act * cos_t
    )
    return out
