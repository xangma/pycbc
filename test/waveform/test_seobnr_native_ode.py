# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for the C++ native RKF45 ODE integrator for SEOBNR."""

import os
import torch

from pycbc.waveform.rkf45_torch import rkf45_step as rkf45_step_py
from pycbc.waveform.seobnrv4phm_ode import integrate
from pycbc.waveform._seobnr_native_ode import (
    cache_metadata,
    clear_cache,
    get_extension,
    is_available,
)


def make_14d_ode_fn():
    """Create a 14-dimensional non-linear test ODE system."""
    def rhs(t, y):
        dy = torch.zeros_like(y)
        for i in range(0, 14, 2):
            freq = 1.0 + 0.2 * i
            dy[..., i] = y[..., i + 1]
            f_sq = freq ** 2
            dy[..., i + 1] = -f_sq * y[..., i] - 0.05 * torch.sin(y[..., i])
        return dy

    return rhs


def test_native_extension_lifecycle_and_metadata():
    """Verify C++ extension build, caching, and metadata availability."""
    clear_cache()
    meta = cache_metadata()
    assert "source_sha256" in meta
    assert "extension_name" in meta
    assert meta["extension_name"].startswith("pycbc_seobnr_native_ode_cpu_")

    ext = get_extension()
    assert ext is not None
    assert is_available()


def test_rkf45_step_native_parity():
    """Verify bit-level parity between native C++ and PyTorch rkf45_step."""
    ext = get_extension()
    assert ext is not None

    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    init_vals = [0.15 * (i + 1) for i in range(14)]
    y0 = torch.tensor(init_vals, dtype=dtype, device=device)
    t = 0.0
    h = 0.05

    step_py = rkf45_step_py(
        rhs,
        torch.tensor(t, dtype=dtype),
        y0,
        torch.tensor(h, dtype=dtype),
        compute_final_derivative=True,
    )
    y_next_c, err_c, k1_c, k6_c, dydt_c = ext.rkf45_step_native(
        rhs, t, y0, h, None, True
    )

    torch.testing.assert_close(y_next_c, step_py.state, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(err_c, step_py.error, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(
        k1_c, step_py.first_derivative, rtol=1e-15, atol=1e-15
    )
    torch.testing.assert_close(
        k6_c, step_py.sixth_derivative, rtol=1e-15, atol=1e-15
    )
    torch.testing.assert_close(
        dydt_c, step_py.final_derivative, rtol=1e-15, atol=1e-15
    )


def test_integrate_native_parity_and_diagnostics():
    """Verify adaptive trajectory integration matches PyTorch reference."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    init_vals = [0.15 * (i + 1) for i in range(14)]
    y0 = torch.tensor(init_vals, dtype=dtype, device=device)
    t0, t1, h0 = 0.0, 50.0, 0.05
    rtol, atol = 1e-8, 1e-10

    # 1. Run with native C++ enabled
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    (t_c, y_c), diag_c = integrate(
        rhs,
        y0,
        t0=t0,
        t1=t1,
        h0=h0,
        rtol=rtol,
        atol=atol,
        return_diagnostics=True,
        return_tensors=True,
    )

    # 2. Run with native C++ disabled (pure Python fallback)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    (t_py, y_py), diag_py = integrate(
        rhs,
        y0,
        t0=t0,
        t1=t1,
        h0=h0,
        rtol=rtol,
        atol=atol,
        return_diagnostics=True,
        return_tensors=True,
    )
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"

    assert diag_c["accepted_steps"] == diag_py["accepted_steps"]
    assert diag_c["rejected_steps"] == diag_py["rejected_steps"]
    assert diag_c["attempted_steps"] == diag_py["attempted_steps"]
    assert diag_c["hit_max_steps"] == diag_py["hit_max_steps"]

    torch.testing.assert_close(t_c, t_py, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(y_c, y_py, rtol=1e-14, atol=1e-14)


def test_integrate_native_custom_stop_fn():
    """Verify custom stopping predicate works in native C++ integrator."""
    dtype = torch.float64
    device = torch.device("cpu")
    rhs = make_14d_ode_fn()

    init_vals = [0.1 * (i + 1) for i in range(14)]
    y0 = torch.tensor(init_vals, dtype=dtype, device=device)
    t0, t1, h0 = 0.0, 50.0, 0.05

    def stop_at_t5(t, y, dy=None):
        return t >= 5.0

    (t_traj, y_traj), diag = integrate(
        rhs,
        y0,
        t0=t0,
        t1=t1,
        h0=h0,
        stop_fn=stop_at_t5,
        return_diagnostics=True,
        return_tensors=True,
    )

    assert float(t_traj[-1]) >= 5.0
    assert float(t_traj[-1]) < 6.0
    assert diag["accepted_steps"] < 500


def test_pure_cpp_rhs_benchmark():
    """Verify pure C++ RHS integration runs and executes accurately."""
    ext = get_extension()
    assert ext is not None

    dtype = torch.float64
    device = torch.device("cpu")
    init_vals = [0.15 * (i + 1) for i in range(14)]
    y0 = torch.tensor(init_vals, dtype=dtype, device=device)
    t0, t1, h0 = 0.0, 50.0, 0.05
    rtol, atol = 1e-8, 1e-10

    (t_c, y_c), diag = ext.integrate_cpp_benchmark(
        y0, t0, t1, h0, rtol, atol, 200000
    )

    assert diag["accepted_steps"] == 2953
    assert diag["rejected_steps"] == 700
    assert t_c.shape == (2953,)
    assert y_c.shape == (2953, 14)
