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
    quadratic = torch.zeros_like(values)
    system_size = count - 2
    if system_size:
        diagonal = 2.0 * (width[:-1] + width[1:])
        rhs = 3.0 * (slopes[1:] - slopes[:-1])

        if system_size == 1:
            interior = rhs / diagonal[0]
        else:
            off_diagonal = width[1:-1]
            factor_diagonal = torch.empty_like(diagonal)
            factor_upper = torch.empty_like(off_diagonal)
            factor_diagonal[0] = diagonal[0]
            factor_upper[0] = off_diagonal[0] / factor_diagonal[0]
            for index in range(1, system_size - 1):
                factor_diagonal[index] = (
                    diagonal[index] - off_diagonal[index - 1] * factor_upper[index - 1]
                )
                factor_upper[index] = off_diagonal[index] / factor_diagonal[index]
            factor_diagonal[-1] = diagonal[-1] - off_diagonal[-1] * factor_upper[-1]

            transformed = torch.empty_like(rhs)
            transformed[0] = rhs[0]
            for index in range(1, system_size):
                transformed[index] = (
                    rhs[index] - factor_upper[index - 1] * transformed[index - 1]
                )

            factor_shape = (system_size,) + (1,) * (values.ndim - 1)
            normalized = transformed / factor_diagonal.reshape(factor_shape)
            interior = torch.empty_like(rhs)
            interior[-1] = normalized[-1]
            for index in range(system_size - 2, -1, -1):
                interior[index] = (
                    normalized[index] - factor_upper[index] * interior[index + 1]
                )

        quadratic[1:-1] = interior

    linear = (
        slopes
        - width.reshape(width_shape) * (quadratic[1:] + 2.0 * quadratic[:-1]) / 3.0
    )
    cubic = (quadratic[1:] - quadratic[:-1]) / (3.0 * width.reshape(width_shape))
    return linear, quadratic, cubic


def _spline_interval(points: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(knots, points.clamp(knots[0], knots[-1])) - 1
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
    return values[indices] + offset * (
        linear[indices] + offset * (quadratic[indices] + offset * cubic[indices])
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
    return linear[indices] + offset * (
        2.0 * quadratic[indices] + 3.0 * cubic[indices] * offset
    )


__all__ = [
    "_natural_cubic_coeff",
    "_spline_derivative",
    "_spline_eval",
]
