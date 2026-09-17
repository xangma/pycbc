# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Torch kernels for interpolating tabulated PSD data."""

import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


def _log_interpolate(freq_data, noise_data, length, delta_f, cutoff):
    """Match SciPy's log-log PSD interpolation on a Torch device."""
    kmin = int(cutoff / delta_f)
    output = torch.zeros(length, dtype=noise_data.dtype, device=noise_data.device)
    if kmin >= length:
        return output

    log_frequency = torch.log(freq_data)
    log_noise = torch.log(noise_data)
    samples = (
        torch.arange(
            kmin,
            length,
            dtype=freq_data.dtype,
            device=freq_data.device,
        )
        * delta_f
    )
    query = torch.log(samples)
    query = torch.clamp(query, min=log_frequency[0], max=log_frequency[-1])

    upper = torch.searchsorted(log_frequency, query, right=False)
    upper = torch.clamp(upper, 1, log_frequency.numel() - 1)
    lower = upper - 1
    x0 = log_frequency[lower]
    x1 = log_frequency[upper]
    weight = (query - x0) / (x1 - x0)
    interpolated = log_noise[lower] + (log_noise[upper] - log_noise[lower]) * weight

    # SciPy's interp1d propagates a log(0) endpoint as -inf throughout the
    # open interval. Reproduce that behavior without the ``-inf - -inf``
    # indeterminate form, so compact-support PSDs stay exactly zero.
    zero_interval = torch.isneginf(log_noise[lower]) | torch.isneginf(log_noise[upper])
    interpolated = torch.where(
        zero_interval,
        torch.full_like(interpolated, -torch.inf),
        interpolated,
    )

    # Avoid the indeterminate ``inf * 0`` form when a query lies exactly on
    # an interpolation knot.
    interpolated = torch.where(query == x0, log_noise[lower], interpolated)
    interpolated = torch.where(query == x1, log_noise[upper], interpolated)
    output[kmin:] = torch.exp(interpolated)
    return output


def from_prepared_arrays(
    freq_data,
    noise_data,
    length,
    delta_f,
    low_freq_cutoff,
    device,
):
    """Interpolate host-prepared PSD samples directly on ``device``."""
    if device.type == "mps":
        raise TypeError(
            "PSD interpolation requires float64; Torch MPS does not support float64"
        )

    frequencies = torch.tensor(freq_data, dtype=torch.float64, device=device)
    noise = torch.tensor(noise_data, dtype=torch.float64, device=device)
    output = _log_interpolate(
        frequencies,
        noise,
        length,
        delta_f,
        low_freq_cutoff,
    )
    return FrequencySeries(
        TorchArrayData(output),
        delta_f=delta_f,
        copy=False,
    )
