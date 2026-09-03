# Copyright (C) 2026
#
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

"""Torch implementation of lalsimulation's time-series injection adder."""

import math

import numpy
import torch

from pycbc.types.array_torch import TorchArrayData


_NOOP_THRESHOLD = 1e-4
_APERIODICITY_SUPPRESSION_BUFFER = 32768


def _epoch_nanoseconds(epoch):
    """Quantize a PyCBC epoch as LAL does when creating a time series."""
    value = float(epoch)
    seconds = math.floor(value)
    nanoseconds = round((value - seconds) * 1_000_000_000)
    return seconds * 1_000_000_000 + nanoseconds


def _overlap_add(target, source, start_index):
    """Add a tensor to a target tensor at an integer sample offset."""
    target_left = max(start_index, 0)
    source_left = max(-start_index, 0)
    count = min(
        target.shape[-1] - target_left,
        source.shape[-1] - source_left,
    )
    if count > 0:
        target[target_left:target_left + count].add_(
            source[source_left:source_left + count]
        )


def _fractional_shift(source, start_sample_int, start_sample_frac):
    """Match lalsimulation's padded frequency-domain interpolation."""
    padded_length = 1 << (
        source.shape[-1] + 2 * _APERIODICITY_SUPPRESSION_BUFFER - 1
    ).bit_length()
    total_padding = padded_length - source.shape[-1]
    left_padding = total_padding // 2
    start_sample_int -= left_padding

    padded = torch.zeros(
        padded_length,
        dtype=source.dtype,
        device=source.device,
    )
    padded[left_padding:left_padding + source.shape[-1]] = source

    spectrum = torch.fft.rfft(padded)
    bins = torch.arange(
        spectrum.shape[-1],
        dtype=source.dtype,
        device=source.device,
    )
    spectrum.mul_(torch.exp(
        (-2j * torch.pi * start_sample_frac / padded_length) * bins
    ))

    # A real inverse transform requires real DC and Nyquist components. LAL
    # additionally makes the DC component non-negative when no response is
    # supplied.
    spectrum[0] = spectrum[0].abs().clone()
    spectrum[-1] = spectrum[-1].real.clone()

    clip = _APERIODICITY_SUPPRESSION_BUFFER // 2
    shifted = torch.fft.irfft(spectrum, n=padded_length)[
        clip:padded_length - clip
    ]
    start_sample_int += clip

    # This is XLALCreateTukeyREAL{4,8}Window for the beta used by
    # XLALSimAddInjectionREAL{4,8}TimeSeries. Construct only the two tapered
    # edges; the middle of the window is exactly one.
    transition_length = _APERIODICITY_SUPPRESSION_BUFFER - 2
    edge_length = (transition_length + 1) // 2
    indices = torch.arange(
        edge_length,
        dtype=source.dtype,
        device=source.device,
    )
    coordinate = (
        2 * indices - (transition_length - 1)
    ) / (transition_length - 1)
    edge = torch.cos(torch.pi / 2 * coordinate).square()
    shifted[:edge_length].mul_(edge)
    shifted[-edge_length:].mul_(edge.flip(0))
    return shifted, int(start_sample_int)


def add_injection(target, source):
    """Add ``source`` to ``target`` using lalsimulation alignment semantics.

    This implements the ``response=None`` behavior of
    ``XLALSimAddInjectionREAL{4,8}TimeSeries`` without moving either series
    off its Torch device. Both series are real and are modified only through
    the in-place addition to ``target``.
    """
    if not isinstance(target._data, TorchArrayData):
        raise TypeError("Target must be backed by Torch")
    if not isinstance(source._data, TorchArrayData):
        raise TypeError("Source must be backed by Torch")
    if target.dtype not in (numpy.dtype(numpy.float32),
                            numpy.dtype(numpy.float64)):
        raise TypeError("Target dtype must be float32 or float64")
    if source.dtype != target.dtype:
        raise TypeError("Source and target dtypes must match")
    if source.delta_t != target.delta_t:
        raise ValueError("Source and target sample rates must match")

    target_data = target._data.tensor
    source_data = source._data.tensor
    if source_data.device != target_data.device:
        raise ValueError("Source and target must use the same Torch device")

    epoch_offset_ns = (
        _epoch_nanoseconds(source.start_time)
        - _epoch_nanoseconds(target.start_time)
    )
    sample_offset = epoch_offset_ns * 1e-9 / target.delta_t
    start_sample_frac, start_sample_int = math.modf(sample_offset)
    if start_sample_frac < -0.5:
        start_sample_frac += 1.0
        start_sample_int -= 1.0
    elif start_sample_frac > 0.5:
        start_sample_frac -= 1.0
        start_sample_int += 1.0

    start_sample_int = int(start_sample_int)
    if abs(start_sample_frac) > _NOOP_THRESHOLD:
        source_data, start_sample_int = _fractional_shift(
            source_data,
            start_sample_int,
            start_sample_frac,
        )

    _overlap_add(target_data, source_data, start_sample_int)
