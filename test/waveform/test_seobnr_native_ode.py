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


def test_natural_cubic_interpolate_torch_native_parity():
    """Verify natural_cubic_interpolate_torch native path matches fallback."""
    from pycbc.waveform._native_math import natural_cubic_interpolate_torch

    x = torch.linspace(0.0, 10.0, 50, dtype=torch.float64)
    query = torch.linspace(0.5, 9.5, 120, dtype=torch.float64)

    # 1. 1D real
    y_1d_real = torch.sin(x) + 0.1 * x
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    out_c_1d_r = natural_cubic_interpolate_torch(query, x, y_1d_real)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    out_py_1d_r = natural_cubic_interpolate_torch(query, x, y_1d_real)
    torch.testing.assert_close(out_c_1d_r, out_py_1d_r, rtol=1e-14, atol=1e-14)

    # 2. 2D real
    y_2d_real = torch.stack([torch.sin(x), torch.cos(x), x ** 1.5], dim=-1)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    out_c_2d_r = natural_cubic_interpolate_torch(query, x, y_2d_real)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    out_py_2d_r = natural_cubic_interpolate_torch(query, x, y_2d_real)
    torch.testing.assert_close(out_c_2d_r, out_py_2d_r, rtol=1e-14, atol=1e-14)

    # 3. 1D complex
    y_1d_c = torch.complex(torch.sin(x), torch.cos(x) * 0.5)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    out_c_1d_c = natural_cubic_interpolate_torch(query, x, y_1d_c)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    out_py_1d_c = natural_cubic_interpolate_torch(query, x, y_1d_c)
    torch.testing.assert_close(out_c_1d_c, out_py_1d_c, rtol=1e-14, atol=1e-14)

    # 4. 2D complex
    y_2d_c = torch.complex(y_2d_real, y_2d_real * 0.5)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    out_c_2d_c = natural_cubic_interpolate_torch(query, x, y_2d_c)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    out_py_2d_c = natural_cubic_interpolate_torch(query, x, y_2d_c)
    torch.testing.assert_close(out_c_2d_c, out_py_2d_c, rtol=1e-14, atol=1e-14)

    # 5. Derivatives (order 1, 2)
    for deriv in (1, 2):
        os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
        out_c_d = natural_cubic_interpolate_torch(
            query, x, y_2d_c, derivative=deriv
        )
        os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
        out_py_d = natural_cubic_interpolate_torch(
            query, x, y_2d_c, derivative=deriv
        )
        torch.testing.assert_close(out_c_d, out_py_d, rtol=1e-14, atol=1e-14)

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"


def test_initial_cartesian_conditions_native_parity():
    """Verify native C++ initial_cartesian_conditions matches Python."""
    from pycbc.waveform.seobnrv4phm_dynamics import (
        initial_cartesian_conditions,
        EOBParams,
    )

    m1, m2 = 30.0, 20.0
    s1x, s1y, s1z = 0.1, 0.2, 0.3
    s2x, s2y, s2z = -0.1, 0.15, -0.25
    f_lower = 20.0

    params = EOBParams(
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        distance=100.0,
        inclination=0.5,
        f_ref=20.0,
        f_lower=f_lower,
    )

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    res_native = initial_cartesian_conditions(
        params, device=torch.device("cpu"), dtype=torch.float64
    )

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    res_py = initial_cartesian_conditions(
        params, device=torch.device("cpu"), dtype=torch.float64
    )

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"

    # Multiroot tolerance is 1e-9
    torch.testing.assert_close(res_native, res_py, rtol=1e-7, atol=1e-7)


def test_calcomega_and_vphi_native_parity():
    """Verify C++ calcomega and non_keplerian_vphi match Python to machine precision."""
    from pycbc.waveform.seobnrv4phm_dynamics import (
        EOBParams,
        initial_conditions,
        reduced_state_to_cartesian_state,
        _calcomega_lal_polar_derivative,
        non_keplerian_vphi,
        _lal_spin_scale,
    )

    params = EOBParams(
        mass1=25.0,
        mass2=15.0,
        spin1x=0.1,
        spin1y=-0.2,
        spin1z=0.3,
        spin2x=-0.15,
        spin2y=0.1,
        spin2z=-0.25,
        distance=100.0,
        inclination=0.5,
        f_ref=30.0,
        f_lower=30.0,
    )

    state = initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    cart = reduced_state_to_cartesian_state(state, params)
    r_vec = cart[0:3]
    p_vec = cart[3:6]
    S1 = cart[6:9] / _lal_spin_scale(params.mass1, params.M)
    S2 = cart[9:12] / _lal_spin_scale(params.mass2, params.M)
    r_mag = torch.linalg.norm(r_vec)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    omega = torch.tensor(0.02, dtype=torch.float64)
    phi = torch.tensor(0.1, dtype=torch.float64)

    # 1. calcomega 1D
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    co_c_1d = _calcomega_lal_polar_derivative(r_vec, p_vec, S1, S2, params)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    co_py_1d = _calcomega_lal_polar_derivative(r_vec, p_vec, S1, S2, params)
    torch.testing.assert_close(co_c_1d, co_py_1d, rtol=1e-14, atol=1e-14)

    # 2. calcomega 2D batch
    r_vec_2d = torch.stack([r_vec, r_vec * 1.1, r_vec * 0.9], dim=0)
    p_vec_2d = torch.stack([p_vec, p_vec * 0.95, p_vec * 1.05], dim=0)
    S1_2d = torch.stack([S1, S1, S1], dim=0)
    S2_2d = torch.stack([S2, S2, S2], dim=0)

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    co_c_2d = _calcomega_lal_polar_derivative(r_vec_2d, p_vec_2d, S1_2d, S2_2d, params)
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    co_py_2d = _calcomega_lal_polar_derivative(r_vec_2d, p_vec_2d, S1_2d, S2_2d, params)
    torch.testing.assert_close(co_c_2d, co_py_2d, rtol=1e-14, atol=1e-14)

    # 3. non_keplerian_vphi
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    vphi_c = non_keplerian_vphi(
        r_mag, omega, phi, L_vec, S1, S2, params,
        r_vec=r_vec, p_vec=p_vec
    )
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    vphi_py = non_keplerian_vphi(
        r_mag, omega, phi, L_vec, S1, S2, params,
        r_vec=r_vec, p_vec=p_vec
    )
    torch.testing.assert_close(vphi_c, vphi_py, rtol=1e-14, atol=1e-14)

    # 4. non_keplerian_vphi 1D batch across trajectory
    r_mag_1d = torch.tensor([r_mag, r_mag * 1.1, r_mag * 0.9], dtype=torch.float64)
    omega_1d = torch.tensor([0.02, 0.025, 0.03], dtype=torch.float64)
    phi_1d = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    L_vec_2d = torch.stack([L_vec, L_vec * 1.05, L_vec * 0.95], dim=0)

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"
    vphi_c_batch = non_keplerian_vphi(
        r_mag_1d, omega_1d, phi_1d, L_vec_2d, S1, S2, params,
        r_vec=r_vec_2d, p_vec=p_vec_2d
    )
    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "0"
    vphi_py_batch = non_keplerian_vphi(
        r_mag_1d, omega_1d, phi_1d, L_vec_2d, S1, S2, params,
        r_vec=r_vec_2d, p_vec=p_vec_2d
    )
    torch.testing.assert_close(vphi_c_batch, vphi_py_batch, rtol=1e-14, atol=1e-14)

    os.environ["PYCBC_SEOBNR_NATIVE_ODE"] = "1"

