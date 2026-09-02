# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Shared Torch helpers for sparse reduced-order models."""

from __future__ import annotations

import bisect
import math
from typing import Tuple

import torch

from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_eval,
)


def _minimum_sequence_frequency(sample_points) -> float:
    """Return the lowest sample without moving Torch data through NumPy."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("sample_points must be a non-empty vector")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("sample_points must be finite")
        return float(torch.min(values).item())

    frequencies = tuple(float(value) for value in sample_points)
    if not frequencies:
        raise ValueError("sample_points must be non-empty")
    if not all(math.isfinite(value) for value in frequencies):
        raise ValueError("sample_points must be finite")
    return min(frequencies)


def _bspline_window(
    breaks: Tuple[float, ...], grid: torch.Tensor, value: float
) -> Tuple[int, torch.Tensor]:
    """Return the first index and four nonzero cubic B-spline values."""

    degree = 3
    ncoeff = len(breaks) + degree - 1
    if value <= breaks[0]:
        span = degree
    elif value >= breaks[-1]:
        span = ncoeff - 1
    else:
        span = bisect.bisect_right(breaks, value) - 1 + degree

    knots = torch.cat((grid[0].repeat(degree), grid, grid[-1].repeat(degree)))
    x = torch.as_tensor(value, dtype=grid.dtype, device=grid.device)
    basis = torch.zeros(degree + 1, dtype=grid.dtype, device=grid.device)
    left = torch.zeros_like(basis)
    right = torch.zeros_like(basis)
    basis[0] = 1.0
    for column in range(1, degree + 1):
        left[column] = x - knots[span + 1 - column]
        right[column] = knots[span + column] - x
        saved = torch.zeros((), dtype=grid.dtype, device=grid.device)
        for row in range(column):
            denominator = right[row + 1] + left[column - row]
            term = basis[row] / denominator
            basis[row] = saved + right[row + 1] * term
            saved = left[column - row] * term
        basis[column] = saved
    return span - degree, basis


def _interpolate_coefficients(coefficients: torch.Tensor, basis) -> torch.Tensor:
    """Interpolate reduced-basis coefficients on a local cubic tensor grid."""

    ix, iy, iz, bx, by, bz = basis
    local = coefficients[:, ix : ix + 4, iy : iy + 4, iz : iz + 4]
    return torch.einsum("nijk,i,j,k->n", local, bx, by, bz)


def _blend_weight(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    """Return the smooth step used by LAL's ROM hybridization."""

    if x1 <= x0:
        raise ValueError("blend interval must have positive width")
    weight = torch.zeros_like(x)
    weight[x >= x1] = 1.0
    interior = (x > x0) & (x < x1)
    width = x1 - x0
    xi = x[interior]
    weight[interior] = torch.sigmoid(-width / (xi - x0) - width / (xi - x1))
    return weight


def _compute_i_max_LF_i_min_HF(
    freq_lo: torch.Tensor, freq_hi: torch.Tensor, window_start: float
) -> Tuple[int, int]:
    """Find the last low and first high grid points around a patch join."""

    i_max = int(torch.searchsorted(freq_lo, window_start, right=False)) - 1
    i_min = int(torch.searchsorted(freq_hi, window_start, right=False))
    if i_max < 0 or i_min >= freq_hi.numel():
        raise ValueError("ROM patches do not overlap the hybridization window")
    return i_max, i_min


def _linear_phase_alignment(
    freq_lo: torch.Tensor,
    phase_lo: torch.Tensor,
    freq_hi: torch.Tensor,
    phase_hi: torch.Tensor,
    window_start: float,
    window_end: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align ``phase_lo`` to ``phase_hi`` using LAL's ten-point fit."""

    fit_frequency = torch.linspace(
        window_start,
        window_end,
        10,
        device=freq_lo.device,
        dtype=freq_lo.dtype,
    )
    difference = _spline_eval(
        fit_frequency,
        freq_hi,
        phase_hi,
        *_natural_cubic_coeff(freq_hi, phase_hi),
    ) - _spline_eval(
        fit_frequency,
        freq_lo,
        phase_lo,
        *_natural_cubic_coeff(freq_lo, phase_lo),
    )
    centered_frequency = fit_frequency - torch.mean(fit_frequency)
    slope = torch.sum(
        centered_frequency * (difference - torch.mean(difference))
    ) / torch.sum(centered_frequency**2)
    intercept = torch.mean(difference) - slope * torch.mean(fit_frequency)
    aligned = phase_lo + intercept + slope * freq_lo
    return aligned, slope / (2.0 * math.pi), intercept


def _hybridize_sparse_functions(
    freq_lo: torch.Tensor,
    values_lo: torch.Tensor,
    freq_hi: torch.Tensor,
    values_hi: torch.Tensor,
    window_start: float,
    window_end: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend two natural-cubic splines using LAL's merged-grid convention."""

    i_max, i_min = _compute_i_max_LF_i_min_HF(freq_lo, freq_hi, window_start)
    frequency = torch.cat([freq_lo[: i_max + 1], freq_hi[i_min:]])
    weight = _blend_weight(frequency, window_start, window_end)
    values = torch.zeros_like(frequency)
    low_coefficients = _natural_cubic_coeff(freq_lo, values_lo)
    high_coefficients = _natural_cubic_coeff(freq_hi, values_hi)

    low_region = frequency <= window_end
    values[low_region] = _spline_eval(
        frequency[low_region],
        freq_lo,
        values_lo,
        *low_coefficients,
    ) * (1.0 - weight[low_region])

    high_region = frequency > window_end
    values[high_region] = _spline_eval(
        frequency[high_region],
        freq_hi,
        values_hi,
        *high_coefficients,
    )
    blend_region = (frequency >= window_start) & (frequency <= window_end)
    values[blend_region] += (
        _spline_eval(
            frequency[blend_region],
            freq_hi,
            values_hi,
            *high_coefficients,
        )
        * weight[blend_region]
    )
    return frequency, values


__all__ = [
    "_bspline_window",
    "_blend_weight",
    "_compute_i_max_LF_i_min_HF",
    "_hybridize_sparse_functions",
    "_interpolate_coefficients",
    "_linear_phase_alignment",
]
