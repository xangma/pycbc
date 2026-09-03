# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native adaptive RKF45 integration with uniform Hermite output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class RKF45Step:
    """One accepted-or-rejected Fehlberg trial step."""

    state: torch.Tensor
    error: torch.Tensor
    first_derivative: torch.Tensor
    sixth_derivative: torch.Tensor
    final_derivative: torch.Tensor | None = None


def rkf45_step(
    function,
    time,
    state,
    step_size,
    *,
    first_derivative=None,
    compute_final_derivative=False,
):
    """Take one GSL-compatible Runge--Kutta--Fehlberg 4(5) trial step."""

    h_state = (
        step_size.view(*step_size.shape, *(1,) * (state.ndim - step_size.ndim))
        if isinstance(step_size, torch.Tensor) and state.ndim > step_size.ndim
        else step_size
    )

    first = (
        function(time, state)
        if first_derivative is None
        else first_derivative
    )
    second = function(
        time + 1.0 / 4.0 * step_size,
        state + 1.0 / 4.0 * h_state * first,
    )
    third = function(
        time + 3.0 / 8.0 * step_size,
        state + h_state * (3.0 / 32.0 * first + 9.0 / 32.0 * second),
    )
    fourth = function(
        time + 12.0 / 13.0 * step_size,
        state
        + h_state
        * (
            1932.0 / 2197.0 * first
            - 7200.0 / 2197.0 * second
            + 7296.0 / 2197.0 * third
        ),
    )
    fifth = function(
        time + step_size,
        state
        + h_state
        * (
            8341.0 / 4104.0 * first
            - 32832.0 / 4104.0 * second
            + 29440.0 / 4104.0 * third
            - 845.0 / 4104.0 * fourth
        ),
    )
    sixth = function(
        time + 1.0 / 2.0 * step_size,
        state
        + h_state
        * (
            -6080.0 / 20520.0 * first
            + 41040.0 / 20520.0 * second
            - 28352.0 / 20520.0 * third
            + 9295.0 / 20520.0 * fourth
            - 5643.0 / 20520.0 * fifth
        ),
    )

    derivative = (
        902880.0 / 7618050.0 * first
        + 3953664.0 / 7618050.0 * third
        + 3855735.0 / 7618050.0 * fourth
        - 1371249.0 / 7618050.0 * fifth
        + 277020.0 / 7618050.0 * sixth
    )
    result = state + h_state * derivative
    error = h_state * (
        1.0 / 360.0 * first
        - 128.0 / 4275.0 * third
        - 2197.0 / 75240.0 * fourth
        + 1.0 / 50.0 * fifth
        + 2.0 / 55.0 * sixth
    )
    final = (
        function(time + step_size, result)
        if compute_final_derivative
        else None
    )
    return RKF45Step(result, error, first, sixth, final)


def _as_bool(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("stop function must return one boolean")
        return bool(value.item())
    return bool(value)


def adaptive_rkf45_hermite(
    function: Callable[[float, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    delta_t: float,
    stop: Callable[[float, torch.Tensor], bool],
    *,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
    max_steps: int = 1_000_000,
):
    """Integrate forward and return uniformly sampled states.

    This follows ``XLALAdaptiveRungeKutta4Hermite``: adaptive steps use the
    GSL ``rkf45`` tableau and ``control_y`` adjustment thresholds, while the
    requested output cadence is filled with LAL's cubic Hermite interpolant.
    The stopping predicate is checked after interpolating an accepted step.
    """

    if not isinstance(initial_state, torch.Tensor):
        raise TypeError("initial_state must be a Torch tensor")
    if initial_state.ndim != 1 or not initial_state.is_floating_point():
        raise ValueError("initial_state must be a one-dimensional floating tensor")
    delta_t = float(delta_t)
    absolute_tolerance = float(absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    time = 0.0
    interpolation_time = 0.0
    step_size = delta_t
    state = initial_state

    capacity = min(max_steps + 1, 4096)
    y_buf = torch.empty(
        (capacity, *initial_state.shape),
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    t_buf = torch.empty(
        capacity,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    y_buf[0] = initial_state
    t_buf[0] = 0.0
    count = 1

    for _ in range(max_steps):
        old_time = time
        old_state = state

        while True:
            trial = rkf45_step(function, time, state, step_size)
            if not (
                torch.isfinite(trial.state).all()
                and torch.isfinite(trial.error).all()
            ):
                raise RuntimeError("RKF45 integration produced a non-finite state")

            scale = absolute_tolerance + relative_tolerance * torch.abs(
                trial.state
            )
            error_ratio = torch.max(torch.abs(trial.error) / scale)
            ratio = float(error_ratio.item())
            if ratio > 1.1:
                reduction = max(0.2, 0.9 * ratio ** (-1.0 / 5.0))
                new_step_size = step_size * reduction
                if new_step_size < step_size:
                    step_size = new_step_size
                    continue

            next_step_size = step_size
            if ratio < 0.5:
                bounded_ratio = max(ratio, torch.finfo(state.dtype).tiny)
                growth = min(
                    5.0,
                    max(1.0, 0.9 * bounded_ratio ** (-1.0 / 6.0)),
                )
                next_step_size = step_size * growth
            break

        time += step_size
        state = trial.state

        next_output_time = interpolation_time + delta_t
        while next_output_time * next_output_time < time * time:
            interpolation_time = next_output_time
            theta = (interpolation_time - old_time) / (time - old_time)
            theta2 = theta * theta
            initial_weight = 1.0 + theta2 * (3.0 - 4.0 * theta)
            first_weight = -theta * (theta - 1.0)
            sixth_weight = -4.0 * theta2 * (theta - 1.0)
            final_weight = theta2 * (4.0 * theta - 3.0)
            interpolated = (
                initial_weight * old_state
                + final_weight * state
                + (time - old_time)
                * (
                    first_weight * trial.first_derivative
                    + sixth_weight * trial.sixth_derivative
                )
            )
            if not torch.isfinite(interpolated).all():
                raise RuntimeError("RKF45 interpolation produced a non-finite state")

            if count >= capacity:
                new_capacity = min(capacity * 2, max_steps + 1)
                new_y_buf = torch.empty(
                    (new_capacity, *initial_state.shape),
                    dtype=initial_state.dtype,
                    device=initial_state.device,
                )
                new_t_buf = torch.empty(
                    new_capacity,
                    dtype=initial_state.dtype,
                    device=initial_state.device,
                )
                new_y_buf[:count] = y_buf[:count]
                new_t_buf[:count] = t_buf[:count]
                y_buf = new_y_buf
                t_buf = new_t_buf
                capacity = new_capacity

            y_buf[count] = interpolated
            t_buf[count] = interpolation_time
            count += 1
            next_output_time = interpolation_time + delta_t

        if _as_bool(stop(time, state)):
            return y_buf[:count].clone()
        step_size = next_step_size
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise RuntimeError("RKF45 step size became invalid")

    raise RuntimeError("RKF45 integration exceeded max_steps")


__all__ = [
    "RKF45Step",
    "adaptive_rkf45_hermite",
    "rkf45_step",
]
