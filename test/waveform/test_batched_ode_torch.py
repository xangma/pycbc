# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for batched ODE integration in PyTorch across (B, 14) states."""

import math
import pytest
import torch

from pycbc.waveform.rkf45_torch import rkf45_step
from pycbc.waveform.seobnrv4phm_ode import RKState, rk45_step, integrate


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def make_14d_ode_fn():
    """Create a 14-dimensional non-linear test ODE system dy/dt = A y + non-linear(y)."""

    def rhs(t, y):
        # y has shape (..., 14)
        # 7 coupled 2D harmonic-like pairs with non-linear cross-coupling
        dy = torch.zeros_like(y)
        for i in range(0, 14, 2):
            freq = 1.0 + 0.2 * i
            dy[..., i] = y[..., i + 1]
            dy[..., i + 1] = - (freq ** 2) * y[..., i] - 0.05 * torch.sin(y[..., i])
        return dy

    return rhs


@pytest.mark.parametrize("b_size", [1, 4, 16])
def test_rkf45_step_scalar_vs_batched_equivalence(b_size):
    """Verify rkf45_step yields identical results for scalar (14,) and batched (B, 14)."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    # Initial 14D state
    y_scalar = torch.tensor(
        [0.1 * (i + 1) for i in range(14)], dtype=dtype, device=device
    )
    t = torch.tensor(0.0, dtype=dtype, device=device)
    h = torch.tensor(0.05, dtype=dtype, device=device)

    # Scalar step
    trial_scalar = rkf45_step(
        rhs, t, y_scalar, h, compute_final_derivative=True
    )

    # Batched state (B, 14) with identical rows
    y_batched = y_scalar.unsqueeze(0).expand(b_size, 14).clone()
    trial_batched = rkf45_step(
        rhs, t, y_batched, h, compute_final_derivative=True
    )

    # Compare state, error, and derivatives
    for b in range(b_size):
        torch.testing.assert_close(
            trial_batched.state[b], trial_scalar.state, rtol=1e-15, atol=1e-15
        )
        torch.testing.assert_close(
            trial_batched.error[b], trial_scalar.error, rtol=1e-15, atol=1e-15
        )
        torch.testing.assert_close(
            trial_batched.first_derivative[b],
            trial_scalar.first_derivative,
            rtol=1e-15,
            atol=1e-15,
        )
        torch.testing.assert_close(
            trial_batched.sixth_derivative[b],
            trial_scalar.sixth_derivative,
            rtol=1e-15,
            atol=1e-15,
        )
        torch.testing.assert_close(
            trial_batched.final_derivative[b],
            trial_scalar.final_derivative,
            rtol=1e-15,
            atol=1e-15,
        )


@pytest.mark.parametrize("b_size", [1, 4, 16])
def test_seobnr_rk45_step_batched(b_size):
    """Verify rk45_step from seobnrv4phm_ode operates seamlessly on (B, 14)."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    y_scalar = torch.tensor(
        [0.2 * (i + 1) for i in range(14)], dtype=dtype, device=device
    )
    t = torch.tensor(0.0, dtype=dtype, device=device)
    h = torch.tensor(0.1, dtype=dtype, device=device)

    state_scalar = RKState(t, y_scalar, h)
    new_state_scalar, err_scalar = rk45_step(rhs, state_scalar)

    y_batched = y_scalar.unsqueeze(0).expand(b_size, 14).clone()
    state_batched = RKState(t, y_batched, h)
    new_state_batched, err_batched = rk45_step(rhs, state_batched)

    assert new_state_batched.y.shape == (b_size, 14)
    assert err_batched.shape == (b_size, 14)
    for b in range(b_size):
        torch.testing.assert_close(new_state_batched.y[b], new_state_scalar.y)
        torch.testing.assert_close(err_batched[b], err_scalar)


@pytest.mark.parametrize("b_size", [1, 4, 16])
def test_integrate_scalar_vs_batched_equivalence(b_size):
    """Verify integrate() produces identical trajectories and diagnostics for (14,) vs (B, 14)."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    y0_scalar = torch.tensor(
        [0.15 * (i + 1) for i in range(14)], dtype=dtype, device=device
    )
    t0, t1, h0 = 0.0, 2.0, 0.05
    rtol, atol = 1e-8, 1e-10

    traj_scalar, diag_scalar = integrate(
        rhs,
        y0_scalar,
        t0=t0,
        t1=t1,
        h0=h0,
        rtol=rtol,
        atol=atol,
        return_diagnostics=True,
    )

    y0_batched = y0_scalar.unsqueeze(0).expand(b_size, 14).clone()
    traj_batched, diag_batched = integrate(
        rhs,
        y0_batched,
        t0=t0,
        t1=t1,
        h0=h0,
        rtol=rtol,
        atol=atol,
        return_diagnostics=True,
    )

    assert len(traj_scalar) == len(traj_batched)
    assert diag_scalar["accepted_steps"] == diag_batched["accepted_steps"]
    assert diag_scalar["rejected_steps"] == diag_batched["rejected_steps"]
    assert diag_scalar["attempted_steps"] == diag_batched["attempted_steps"]
    assert diag_scalar["hit_max_steps"] == diag_batched["hit_max_steps"]
    assert diag_scalar["t_end"] == pytest.approx(diag_batched["t_end"], rel=1e-12)
    assert diag_scalar["h_final"] == pytest.approx(diag_batched["h_final"], rel=1e-12)

    for (t_s, y_s), (t_b, y_b) in zip(traj_scalar, traj_batched):
        torch.testing.assert_close(t_s, t_b, rtol=1e-14, atol=1e-14)
        assert y_b.shape == (b_size, 14)
        for b in range(b_size):
            torch.testing.assert_close(y_b[b], y_s, rtol=1e-14, atol=1e-14)


def test_integrate_batched_harmonic_oscillator_accuracy():
    """Verify batched RKF45 integration against analytical harmonic oscillator solutions."""
    dtype = torch.float64
    device = torch.device("cpu")
    b_size = 8

    # Each batch element has a different frequency omega
    omegas = torch.linspace(0.5, 3.0, b_size, dtype=dtype, device=device)
    x0 = torch.tensor([1.0] * b_size, dtype=dtype, device=device)
    v0 = torch.tensor([0.0] * b_size, dtype=dtype, device=device)

    # Embed into (B, 14) state where [..., 0]=x, [..., 1]=v, others 0
    y0 = torch.zeros(b_size, 14, dtype=dtype, device=device)
    y0[:, 0] = x0
    y0[:, 1] = v0

    def harmonic_rhs(t, y):
        # y shape: (B, 14)
        dy = torch.zeros_like(y)
        dy[:, 0] = y[:, 1]
        dy[:, 1] = - (omegas ** 2) * y[:, 0]
        return dy

    t0, t1 = 0.0, 2.0 * math.pi
    traj = integrate(
        harmonic_rhs,
        y0,
        t0=t0,
        t1=t1,
        h0=0.01,
        rtol=1e-8,
        atol=1e-10,
    )

    assert len(traj) > 5
    t_end, y_end = traj[-1]

    # Analytical solution at t_end: x(t) = x0*cos(omega*t), v(t) = -x0*omega*sin(omega*t)
    t_val = t_end
    expected_x = x0 * torch.cos(omegas * t_val)
    expected_v = -x0 * omegas * torch.sin(omegas * t_val)

    torch.testing.assert_close(y_end[:, 0], expected_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_end[:, 1], expected_v, rtol=1e-5, atol=1e-5)

    # Check energy conservation across the trajectory: E = 0.5 * (v^2 + omega^2 * x^2)
    initial_energy = 0.5 * ((omegas ** 2) * (x0 ** 2) + v0 ** 2)
    for _, y_step in traj:
        energy_step = 0.5 * ((omegas ** 2) * (y_step[:, 0] ** 2) + y_step[:, 1] ** 2)
        torch.testing.assert_close(energy_step, initial_energy, rtol=1e-5, atol=1e-5)


def test_integrate_batched_heterogeneous_independent_consistency():
    """Verify that batched integration matches independent single-trajectory solves."""
    dtype = torch.float64
    device = torch.device("cpu")
    b_size = 5
    rhs = make_14d_ode_fn()

    # Create distinct initial conditions for each batch element
    y0_batched = torch.zeros(b_size, 14, dtype=dtype, device=device)
    for b in range(b_size):
        y0_batched[b] = torch.tensor(
            [0.1 * (b + 1) * (i + 1) * 0.1 for i in range(14)],
            dtype=dtype,
            device=device,
        )

    t0, t1, h0 = 0.0, 1.0, 0.05
    rtol, atol = 1e-8, 1e-10

    # Batched solve
    traj_batched = integrate(
        rhs, y0_batched, t0=t0, t1=t1, h0=h0, rtol=rtol, atol=atol
    )

    # Independent scalar solves
    single_trajs = []
    for b in range(b_size):
        traj_single = integrate(
            rhs, y0_batched[b], t0=t0, t1=t1, h0=h0, rtol=rtol, atol=atol
        )
        single_trajs.append(traj_single)

    # Both batched and individual solves must reach t1 with high accuracy
    t_end_batched, y_end_batched = traj_batched[-1]
    for b in range(b_size):
        t_end_single, y_end_single = single_trajs[b][-1]
        torch.testing.assert_close(t_end_batched, t_end_single, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            y_end_batched[b], y_end_single, rtol=1e-5, atol=1e-5
        )


def test_integrate_error_norm_dim_minus_1():
    """Verify adaptive error norm calculation is performed along dim=-1."""
    dtype = torch.float64
    device = torch.device("cpu")
    b_size = 3

    # Fast decaying trajectory vs slow trajectory
    def decaying_rhs(t, y):
        # y: (B, 14)
        rates = torch.tensor([1.0, 10.0, 50.0], dtype=dtype, device=device).unsqueeze(1)
        return -rates * y

    y0 = torch.ones(b_size, 14, dtype=dtype, device=device)
    traj, diag = integrate(
        decaying_rhs,
        y0,
        t0=0.0,
        t1=0.5,
        h0=0.05,
        rtol=1e-6,
        atol=1e-8,
        return_diagnostics=True,
    )

    assert len(traj) > 2
    t_end, y_end = traj[-1]
    assert y_end.shape == (b_size, 14)
    # The fast decaying element (index 2) should be close to 0
    assert float(y_end[2, 0]) < float(y_end[0, 0])


def test_integrate_batched_early_stop():
    """Verify early stop predicate handles batched states properly."""
    dtype = torch.float64
    device = torch.device("cpu")
    b_size = 4
    rhs = make_14d_ode_fn()

    y0 = torch.ones(b_size, 14, dtype=dtype, device=device)
    # Stop when t >= 0.3
    traj = integrate(
        rhs,
        y0,
        t0=0.0,
        t1=2.0,
        h0=0.05,
        stop_fn=lambda t, y: t >= 0.3,
    )

    t_end, _ = traj[-1]
    assert float(t_end) >= 0.3
    assert float(t_end) < 0.5


@pytest.mark.parametrize("device_str", DEVICES)
@pytest.mark.parametrize("b_size", [1, 4])
def test_device_compatibility(device_str, b_size):
    """Verify rkf45_step and integrate function on CPU and MPS devices."""
    device = torch.device(device_str)
    # MPS does not support float64, so use float32 for MPS and float64 for CPU
    dtype = torch.float32 if device_str == "mps" else torch.float64

    def simple_rhs(t, y):
        return -0.5 * y

    # Test 1D scalar state
    y0_1d = torch.ones(14, dtype=dtype, device=device)
    t = torch.tensor(0.0, dtype=dtype, device=device)
    h = torch.tensor(0.1, dtype=dtype, device=device)

    step_1d = rkf45_step(simple_rhs, t, y0_1d, h)
    assert step_1d.state.device.type == device.type
    assert step_1d.state.shape == (14,)

    traj_1d = integrate(
        simple_rhs, y0_1d, t0=0.0, t1=0.5, h0=0.1, rtol=1e-4, atol=1e-6
    )
    assert traj_1d[-1][1].device.type == device.type
    assert traj_1d[-1][1].shape == (14,)

    # Test 2D batched state (B, 14)
    y0_2d = torch.ones(b_size, 14, dtype=dtype, device=device)
    step_2d = rkf45_step(simple_rhs, t, y0_2d, h)
    assert step_2d.state.device.type == device.type
    assert step_2d.state.shape == (b_size, 14)

    traj_2d = integrate(
        simple_rhs, y0_2d, t0=0.0, t1=0.5, h0=0.1, rtol=1e-4, atol=1e-6
    )
    assert traj_2d[-1][1].device.type == device.type
    assert traj_2d[-1][1].shape == (b_size, 14)


def test_integrate_return_tensors_optimization():
    """Verify integrate(..., return_tensors=True) returns contiguous tensors."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()
    y0 = torch.tensor(
        [0.1 * (i + 1) for i in range(14)], dtype=dtype, device=device
    )

    # Legacy list-of-tuples return
    traj_tuples, diag_tuples = integrate(
        rhs,
        y0,
        t0=0.0,
        t1=1.0,
        h0=0.05,
        rtol=1e-8,
        atol=1e-10,
        return_diagnostics=True,
        return_tensors=False,
    )

    # Optimized direct tensor return
    (t_traj, y_traj), diag_tensors = integrate(
        rhs,
        y0,
        t0=0.0,
        t1=1.0,
        h0=0.05,
        rtol=1e-8,
        atol=1e-10,
        return_diagnostics=True,
        return_tensors=True,
    )

    assert isinstance(t_traj, torch.Tensor)
    assert isinstance(y_traj, torch.Tensor)
    assert t_traj.is_contiguous()
    assert y_traj.is_contiguous()
    assert t_traj.shape == (len(traj_tuples),)
    assert y_traj.shape == (len(traj_tuples), 14)
    assert diag_tuples == diag_tensors

    for i, (t_i, y_i) in enumerate(traj_tuples):
        torch.testing.assert_close(t_traj[i], t_i)
        torch.testing.assert_close(y_traj[i], y_i)
