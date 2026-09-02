# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Small dependency-free numerical helpers for native waveform models.

The functions here replace three scalar/interpolation uses which otherwise
pull SciPy into Torch-native waveform generation.  They deliberately implement
the same mathematical algorithms as the external routines they replace:
natural cubic splines, integer associated Legendre functions at zero, and
Carlson's duplication algorithm for an incomplete elliptic integral.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch


def _double_factorial(value: int) -> int:
    """Return ``value!!``, including the conventional ``(-1)!! = 0!! = 1``."""

    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def associated_legendre_at_zero(degree: int, order: int) -> float:
    """Return the Ferrers function :math:`P_l^m(0)` for integer ``l, m``.

    The sign convention includes the Condon--Shortley phase, matching
    ``scipy.special.lpmv(m, l, 0.0)`` for the non-negative orders used by the
    EOB mode builders.
    """

    degree = int(degree)
    order = int(order)
    if degree < 0 or order < 0:
        raise ValueError("associated Legendre degree and order must be non-negative")
    if order > degree:
        return 0.0
    if (degree + order) % 2:
        return 0.0
    sign = -1.0 if ((degree + order) // 2) % 2 else 1.0
    return sign * (
        _double_factorial(degree + order - 1)
        / _double_factorial(degree - order)
    )


def _carlson_rf(x: float, y: float, z: float) -> float:
    """Return Carlson's symmetric integral ``R_F(x, y, z)``."""

    # Each Carlson duplication reduces the relative deviations by about four.
    # Stopping at sqrt(epsilon) makes the fifth-order correction negligible
    # compared with binary64 rounding.
    for _ in range(64):
        average = (x + y + z) / 3.0
        dx = (average - x) / average
        dy = (average - y) / average
        dz = (average - z) / average
        if max(abs(dx), abs(dy), abs(dz)) <= math.sqrt(math.ulp(1.0)):
            break
        sqrt_x = math.sqrt(x)
        sqrt_y = math.sqrt(y)
        sqrt_z = math.sqrt(z)
        lam = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
    else:  # pragma: no cover - duplication is geometrically convergent
        raise RuntimeError("Carlson R_F duplication failed to converge")

    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    correction = (
        1.0
        + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2
        + e3 / 14.0
    )
    return correction / math.sqrt(average)


def incomplete_elliptic_f(angle: float, parameter: float) -> float:
    """Return the real Legendre incomplete elliptic integral ``F(angle | m)``.

    Evaluation on the principal interval uses Carlson's identity

    ``F(phi | m) = sin(phi) R_F(cos(phi)^2, 1-m sin(phi)^2, 1)``.

    For ``m < 1`` the standard ``pi`` quasi-period is restored with the
    complete integral ``K(m) = R_F(0, 1-m, 1)``.  The real-valued domain and
    singular ``m == 1`` behavior match the scalar routine this replaces.
    """

    angle = float(angle)
    parameter = float(parameter)
    if not math.isfinite(angle) or not math.isfinite(parameter):
        return math.nan
    if parameter > 1.0:
        return math.nan
    if parameter == 0.0:
        return angle

    half_pi = 0.5 * math.pi
    if parameter == 1.0:
        if abs(angle) >= half_pi:
            return math.inf
        return math.atanh(math.sin(angle))

    turns = int(round(angle / math.pi))
    reduced_angle = math.remainder(angle, math.pi)
    sine = math.sin(reduced_angle)
    if sine == 0.0:
        principal = sine
    else:
        x = max(0.0, math.cos(reduced_angle) ** 2)
        y = 1.0 - parameter * sine * sine
        principal = sine * _carlson_rf(x, y, 1.0)

    if turns == 0:
        return principal
    complete = _carlson_rf(0.0, 1.0 - parameter, 1.0)
    return principal + 2.0 * turns * complete


def _natural_cubic_coefficients_numpy(x, y):
    """Return interval coefficients ``(a, b, c, d)`` for a natural spline."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.result_type(y, np.float64))
    if x.ndim != 1 or y.ndim == 0 or y.shape[0] != x.size:
        raise ValueError("natural cubic inputs must share a one-dimensional axis")
    if x.size < 2:
        raise ValueError("natural cubic interpolation requires at least two points")
    h = np.diff(x)
    if not np.all(np.isfinite(x)) or not np.all(h > 0.0):
        raise ValueError("natural cubic knots must be finite and strictly increasing")

    expand = (slice(None),) + (None,) * (y.ndim - 1)
    slopes = np.diff(y, axis=0) / h[expand]
    second = [np.zeros_like(y[0])]
    interior_count = x.size - 2
    if interior_count:
        rhs = 6.0 * (slopes[1:] - slopes[:-1])
        reduced_upper = []
        reduced_rhs = []
        for interior in range(interior_count):
            lower = h[interior]
            diagonal = 2.0 * (h[interior] + h[interior + 1])
            upper = h[interior + 1]
            if interior:
                diagonal -= lower * reduced_upper[-1]
                rhs_value = rhs[interior] - lower * reduced_rhs[-1]
            else:
                rhs_value = rhs[interior]
            reduced_upper.append(
                upper / diagonal if interior + 1 < interior_count else 0.0
            )
            reduced_rhs.append(rhs_value / diagonal)

        interior_second = [None] * interior_count
        interior_second[-1] = reduced_rhs[-1]
        for interior in range(interior_count - 2, -1, -1):
            interior_second[interior] = (
                reduced_rhs[interior]
                - reduced_upper[interior] * interior_second[interior + 1]
            )
        second.extend(interior_second)
    second.append(np.zeros_like(y[-1]))
    second = np.stack(second, axis=0)

    b = slopes - h[expand] * (2.0 * second[:-1] + second[1:]) / 6.0
    c = 0.5 * second[:-1]
    d = (second[1:] - second[:-1]) / (6.0 * h[expand])
    return y[:-1], b, c, d


class NaturalCubicSpline:
    """Minimal natural-cubic interpolator for scalar/table waveform setup.

    Its call convention covers the subset of ``scipy.interpolate.CubicSpline``
    used by the native waveform modules: interpolation along axis zero,
    derivatives zero through three, and optional extrapolation.
    """

    def __init__(self, x, y, *, extrapolate: bool = True):
        self.x = np.asarray(x, dtype=np.float64)
        self.extrapolate = bool(extrapolate)
        self.coefficients = _natural_cubic_coefficients_numpy(self.x, y)

    def __call__(self, values, derivative: int = 0):
        if derivative not in (0, 1, 2, 3):
            raise ValueError("natural cubic derivative order must be 0, 1, 2, or 3")
        query = np.asarray(values, dtype=np.float64)
        indices = np.searchsorted(self.x, query, side="right") - 1
        indices = np.clip(indices, 0, self.x.size - 2)
        dx = query - self.x[indices]
        trailing = (None,) * (self.coefficients[0].ndim - 1)
        dx_expanded = dx[(...,) + trailing]
        a, b, c, d = (coefficient[indices] for coefficient in self.coefficients)
        if derivative == 0:
            result = a + dx_expanded * (
                b + dx_expanded * (c + dx_expanded * d)
            )
        elif derivative == 1:
            result = b + dx_expanded * (2.0 * c + 3.0 * dx_expanded * d)
        elif derivative == 2:
            result = 2.0 * c + 6.0 * dx_expanded * d
        else:
            result = 6.0 * d

        if not self.extrapolate:
            outside = (query < self.x[0]) | (query > self.x[-1])
            if np.any(outside):
                mask = outside[(...,) + trailing]
                result = np.where(mask, np.nan, result)
        return result


def natural_cubic_interpolate_torch(
    values: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    derivative: int = 0,
    extrapolate: bool = True,
) -> torch.Tensor:
    """Evaluate a natural cubic spline entirely with Torch operations.

    The knot selection is discrete, as for every piecewise interpolator, while
    coefficient construction and polynomial evaluation retain gradients with
    respect to ``x``, ``y``, and query values inside the selected intervals.
    """

    if derivative not in (0, 1, 2, 3):
        raise ValueError("natural cubic derivative order must be 0, 1, 2, or 3")
    if x.ndim != 1 or y.ndim == 0 or y.shape[0] != x.numel():
        raise ValueError("natural cubic inputs must share a one-dimensional axis")
    if x.numel() < 2:
        raise ValueError("natural cubic interpolation requires at least two points")

    real_dtype = y.real.dtype if y.is_complex() else y.dtype
    if (
        x.device.type == "cpu"
        and x.dtype == torch.float64
        and values.device.type == "cpu"
        and values.dtype == torch.float64
        and y.device.type == "cpu"
        and real_dtype == torch.float64
        and (not torch.is_grad_enabled() or not (values.requires_grad or x.requires_grad or y.requires_grad))
        and os.environ.get("PYCBC_SEOBNR_NATIVE_ODE", "1") not in ("0", "", "false", "False")
    ):
        try:
            from pycbc.waveform._seobnr_native_ode import get_extension
            ext = get_extension()
            if ext is not None and hasattr(ext, "natural_spline_interpolate_native"):
                if not y.is_complex():
                    if y.ndim in (1, 2):
                        return ext.natural_spline_interpolate_native(
                            values.contiguous(), x.contiguous(), y.contiguous(), int(derivative), bool(extrapolate)
                        )
                else:
                    if y.ndim in (1, 2):
                        r_out = ext.natural_spline_interpolate_native(
                            values.contiguous(), x.contiguous(), y.real.contiguous(), int(derivative), bool(extrapolate)
                        )
                        i_out = ext.natural_spline_interpolate_native(
                            values.contiguous(), x.contiguous(), y.imag.contiguous(), int(derivative), bool(extrapolate)
                        )
                        return torch.complex(r_out, i_out)
        except Exception:
            pass

    real_dtype = y.real.dtype if y.is_complex() else y.dtype
    x = x.to(device=y.device, dtype=real_dtype)
    values = values.to(device=y.device, dtype=real_dtype)
    h = x[1:] - x[:-1]
    expand = (slice(None),) + (None,) * (y.ndim - 1)
    slopes = (y[1:] - y[:-1]) / h[expand]


    second = torch.zeros_like(y)
    interior_count = x.numel() - 2
    if interior_count:
        rhs = 6.0 * (slopes[1:] - slopes[:-1])
        reduced_upper = torch.empty(
            interior_count, device=y.device, dtype=real_dtype
        )
        reduced_rhs = torch.empty_like(rhs)
        prev_upper = None
        prev_rhs = None
        for interior in range(interior_count):
            lower = h[interior]
            diagonal = 2.0 * (h[interior] + h[interior + 1])
            upper = h[interior + 1]
            if interior:
                diagonal = diagonal - lower * prev_upper
                rhs_value = rhs[interior] - lower * prev_rhs
            else:
                rhs_value = rhs[interior]
            curr_upper = (
                upper / diagonal
                if interior + 1 < interior_count
                else torch.zeros_like(diagonal)
            )
            curr_rhs = rhs_value / diagonal
            reduced_upper[interior] = curr_upper
            reduced_rhs[interior] = curr_rhs
            prev_upper = curr_upper
            prev_rhs = curr_rhs

        prev_second = reduced_rhs[-1]
        second[interior_count] = prev_second
        for interior in range(interior_count - 2, -1, -1):
            curr_second = (
                reduced_rhs[interior]
                - reduced_upper[interior] * prev_second
            )
            second[interior + 1] = curr_second
            prev_second = curr_second

    b = slopes - h[expand] * (2.0 * second[:-1] + second[1:]) / 6.0
    c = 0.5 * second[:-1]
    d = (second[1:] - second[:-1]) / (6.0 * h[expand])

    indices = torch.searchsorted(x, values, right=True) - 1
    indices = torch.clamp(indices, 0, x.numel() - 2)
    dx = values - x[indices]
    dx = dx[(...,) + (None,) * (y.ndim - 1)]
    a_i = y[:-1][indices]
    b_i = b[indices]
    c_i = c[indices]
    d_i = d[indices]
    if derivative == 0:
        result = a_i + dx * (b_i + dx * (c_i + dx * d_i))
    elif derivative == 1:
        result = b_i + dx * (2.0 * c_i + 3.0 * dx * d_i)
    elif derivative == 2:
        result = 2.0 * c_i + 6.0 * dx * d_i
    else:
        result = 6.0 * d_i

    if not extrapolate:
        outside = (values < x[0]) | (values > x[-1])
        outside = outside[(...,) + (None,) * (y.ndim - 1)]
        nan = torch.full((), float("nan"), device=y.device, dtype=real_dtype)
        result = torch.where(outside, nan, result)
    return result


__all__ = [
    "NaturalCubicSpline",
    "associated_legendre_at_zero",
    "incomplete_elliptic_f",
    "natural_cubic_interpolate_torch",
]
