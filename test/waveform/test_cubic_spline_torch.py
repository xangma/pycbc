# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Regression tests for the shared Torch natural-cubic-spline helpers."""

import numpy as np
import pytest
import torch

from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)


def _gsl_natural_cubic_coeff(knots, values):
    """Independent NumPy translation of GSL's cspline factorization."""

    knots = np.asarray(knots, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    width = np.diff(knots)
    width_shape = (width.size,) + (1,) * (values.ndim - 1)
    slopes = np.diff(values, axis=0) / width.reshape(width_shape)
    quadratic = np.zeros_like(values)
    system_size = knots.size - 2

    if system_size:
        diagonal = 2.0 * (width[:-1] + width[1:])
        rhs = 3.0 * np.diff(slopes, axis=0)
        if system_size == 1:
            interior = rhs / diagonal[0]
        else:
            off_diagonal = width[1:-1]
            factor_diagonal = np.empty_like(diagonal)
            factor_upper = np.empty_like(off_diagonal)
            factor_diagonal[0] = diagonal[0]
            factor_upper[0] = off_diagonal[0] / factor_diagonal[0]
            for index in range(1, system_size - 1):
                factor_diagonal[index] = (
                    diagonal[index] - off_diagonal[index - 1] * factor_upper[index - 1]
                )
                factor_upper[index] = off_diagonal[index] / factor_diagonal[index]
            factor_diagonal[-1] = diagonal[-1] - off_diagonal[-1] * factor_upper[-1]

            transformed = np.empty_like(rhs)
            transformed[0] = rhs[0]
            for index in range(1, system_size):
                transformed[index] = (
                    rhs[index] - factor_upper[index - 1] * transformed[index - 1]
                )

            factor_shape = (system_size,) + (1,) * (values.ndim - 1)
            normalized = transformed / factor_diagonal.reshape(factor_shape)
            interior = np.empty_like(rhs)
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


@pytest.mark.parametrize("count", [2, 3, 8])
def test_natural_cubic_coeff_matches_gsl_factorization(count):
    knots = np.geomspace(1.0e-5, 0.07, count)
    values = np.column_stack(
        (
            -0.2 * knots ** (-5.0 / 3.0) + 7.0 * np.log(knots),
            np.sin(31.0 * knots) + 0.4 * knots,
        )
    )
    expected = _gsl_natural_cubic_coeff(knots, values)
    actual = _natural_cubic_coeff(
        torch.from_numpy(knots),
        torch.from_numpy(values),
    )

    for actual_coefficient, expected_coefficient in zip(actual, expected):
        np.testing.assert_array_equal(
            actual_coefficient.numpy(),
            expected_coefficient,
        )


def test_spline_evaluation_uses_gsl_horner_order():
    knots = np.geomspace(1.0e-5, 0.07, 1000)
    values = -0.2 * knots ** (-5.0 / 3.0) + 7.0 * np.log(knots)
    linear, quadratic, cubic = _gsl_natural_cubic_coeff(knots, values)
    point = np.float64(1.0 / (6.0**1.5 * np.pi))
    index = np.searchsorted(knots, point) - 1
    offset = point - knots[index]
    expected_value = values[index] + offset * (
        linear[index] + offset * (quadratic[index] + offset * cubic[index])
    )
    expected_derivative = linear[index] + offset * (
        2.0 * quadratic[index] + 3.0 * cubic[index] * offset
    )

    torch_knots = torch.from_numpy(knots)
    torch_values = torch.from_numpy(values)
    coefficients = _natural_cubic_coeff(torch_knots, torch_values)
    torch_point = torch.tensor(point, dtype=torch.float64)
    actual_value = _spline_eval(
        torch_point,
        torch_knots,
        torch_values,
        *coefficients,
    )
    actual_derivative = _spline_derivative(
        torch_point,
        torch_knots,
        *coefficients,
    )

    assert actual_value.item() == expected_value
    assert actual_derivative.item() == expected_derivative
