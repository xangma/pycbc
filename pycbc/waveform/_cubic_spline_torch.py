# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Small Torch-native natural-cubic-spline helpers.

The coefficient construction follows the tridiagonal algorithm used for a
natural cubic spline and matches GSL's ``gsl_interp_cspline`` convention.
The first axis of ``values`` is the knot axis, so several splines may be
constructed together without leaving the active Torch device.
"""

from __future__ import annotations

import torch


def _natural_cubic_coeff(
    knots: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interval linear, quadratic, and cubic coefficients."""

    if knots.ndim != 1:
        raise ValueError("natural-cubic-spline knots must be one-dimensional")
    if values.ndim < 1 or values.shape[0] != knots.numel():
        raise ValueError("spline values must have one row per knot")
    if knots.numel() < 2:
        raise ValueError("a natural cubic spline requires at least two knots")
    if knots.device != values.device or knots.dtype != values.dtype:
        raise ValueError("spline knots and values must share dtype and device")

    count = knots.numel()
    width = knots[1:] - knots[:-1]
    width_shape = (width.numel(),) + (1,) * (values.ndim - 1)
    slopes = (values[1:] - values[:-1]) / width.reshape(width_shape)
    alpha = 3.0 * (slopes[1:] - slopes[:-1])

    diagonal = torch.ones(count, dtype=knots.dtype, device=knots.device)
    upper = torch.zeros(count, dtype=knots.dtype, device=knots.device)
    rhs = torch.zeros_like(values)
    for index in range(1, count - 1):
        diagonal[index] = (
            2.0 * (knots[index + 1] - knots[index - 1])
            - width[index - 1] * upper[index - 1]
        )
        upper[index] = width[index] / diagonal[index]
        rhs[index] = (
            alpha[index - 1] - width[index - 1] * rhs[index - 1]
        ) / diagonal[index]

    quadratic = torch.zeros_like(values)
    linear = torch.zeros_like(values[:-1])
    cubic = torch.zeros_like(values[:-1])
    for index in range(count - 2, -1, -1):
        quadratic[index] = rhs[index] - upper[index] * quadratic[index + 1]
        linear[index] = (values[index + 1] - values[index]) / width[
            index
        ] - width[index] * (
            quadratic[index + 1] + 2.0 * quadratic[index]
        ) / 3.0
        cubic[index] = (
            quadratic[index + 1] - quadratic[index]
        ) / (3.0 * width[index])
    return linear, quadratic, cubic


def _spline_interval(points: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(
        knots, points.clamp(knots[0], knots[-1])
    ) - 1
    return indices.clamp(0, knots.numel() - 2)


def _spline_eval(
    points: torch.Tensor,
    knots: torch.Tensor,
    values: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    cubic: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a natural cubic spline at ``points``."""

    indices = _spline_interval(points, knots)
    offset = points - knots[indices]
    return (
        values[indices]
        + linear[indices] * offset
        + quadratic[indices] * offset**2
        + cubic[indices] * offset**3
    )


def _spline_derivative(
    points: torch.Tensor,
    knots: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    cubic: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the first derivative of a natural cubic spline."""

    indices = _spline_interval(points, knots)
    offset = points - knots[indices]
    return (
        linear[indices]
        + 2.0 * quadratic[indices] * offset
        + 3.0 * cubic[indices] * offset**2
    )


__all__ = [
    "_natural_cubic_coeff",
    "_spline_derivative",
    "_spline_eval",
]
