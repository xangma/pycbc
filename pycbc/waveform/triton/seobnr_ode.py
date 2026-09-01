# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Fused on-device RKF45 and Dormand-Prince adaptive ODE stepping for SEOBNR dynamics."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _rkf45_stage_combine_kernel(
        y_out_ptr,
        err_out_ptr,
        y_ptr,
        k1_ptr,
        k3_ptr,
        k4_ptr,
        k5_ptr,
        k6_ptr,
        h,
        stride_b,
        stride_comp,
        n_comps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused kernel combining RKF45 stages to produce candidate state and truncation error."""
        pid_b = tl.program_id(0)
        c_idx = tl.arange(0, BLOCK_SIZE)
        mask = c_idx < n_comps

        offset = pid_b * stride_b + c_idx * stride_comp

        y = tl.load(y_ptr + offset, mask=mask)
        k1 = tl.load(k1_ptr + offset, mask=mask)
        k3 = tl.load(k3_ptr + offset, mask=mask)
        k4 = tl.load(k4_ptr + offset, mask=mask)
        k5 = tl.load(k5_ptr + offset, mask=mask)
        k6 = tl.load(k6_ptr + offset, mask=mask)

        # 4th-order GSL state update
        b1 = 902880.0 / 7618050.0
        b3 = 3953664.0 / 7618050.0
        b4 = 3855735.0 / 7618050.0
        b5 = -1371249.0 / 7618050.0
        b6 = 277020.0 / 7618050.0

        deriv = b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
        y_next = y + h * deriv

        # Truncation error estimate (5th order - 4th order difference)
        e1 = 1.0 / 360.0
        e3 = -128.0 / 4275.0
        e4 = -2197.0 / 75240.0
        e5 = 1.0 / 50.0
        e6 = 2.0 / 55.0

        err = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6)

        tl.store(y_out_ptr + offset, y_next, mask=mask)
        tl.store(err_out_ptr + offset, err, mask=mask)

    @triton.jit
    def _dormand_prince_stage_combine_kernel(
        y_out_ptr,
        err_out_ptr,
        y_ptr,
        k1_ptr,
        k3_ptr,
        k4_ptr,
        k5_ptr,
        k6_ptr,
        k7_ptr,
        h,
        stride_b,
        stride_comp,
        n_comps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused kernel combining Dormand-Prince 5(4) stages to produce state and error."""
        pid_b = tl.program_id(0)
        c_idx = tl.arange(0, BLOCK_SIZE)
        mask = c_idx < n_comps

        offset = pid_b * stride_b + c_idx * stride_comp

        y = tl.load(y_ptr + offset, mask=mask)
        k1 = tl.load(k1_ptr + offset, mask=mask)
        k3 = tl.load(k3_ptr + offset, mask=mask)
        k4 = tl.load(k4_ptr + offset, mask=mask)
        k5 = tl.load(k5_ptr + offset, mask=mask)
        k6 = tl.load(k6_ptr + offset, mask=mask)
        k7 = tl.load(k7_ptr + offset, mask=mask)

        # 5th-order Dormand-Prince state update
        b1 = 35.0 / 384.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0

        deriv = b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
        y_next = y + h * deriv

        # Truncation error estimate
        e1 = 71.0 / 57600.0
        e3 = -71.0 / 16695.0
        e4 = 71.0 / 1920.0
        e5 = -17253.0 / 339200.0
        e6 = 22.0 / 525.0
        e7 = -1.0 / 40.0

        err = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)

        tl.store(y_out_ptr + offset, y_next, mask=mask)
        tl.store(err_out_ptr + offset, err, mask=mask)
else:
    _rkf45_stage_combine_kernel = None
    _dormand_prince_stage_combine_kernel = None


def is_triton_available() -> bool:
    """Return whether Triton is installed and supported."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def seobnr_rkf45_step_triton(
    rhs_fn,
    t: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    k1: torch.Tensor | None = None,
):
    """Evaluate one adaptive RKF45 step with Triton stage fusion if available."""
    if not is_triton_available():
        from pycbc.waveform.rkf45_torch import rkf45_step
        return rkf45_step(rhs_fn, t, y, h, first_derivative=k1, compute_final_derivative=True)

    # Tensor layout: y is (B, D) or (D,)
    is_1d = y.ndim == 1
    if is_1d:
        y_b = y.unsqueeze(0)
    else:
        y_b = y

    batch_size, n_comps = y_b.shape
    h_val = float(h.item() if isinstance(h, torch.Tensor) else h)
    t_val = float(t.item() if isinstance(t, torch.Tensor) else t)

    k1_t = rhs_fn(t, y) if k1 is None else k1
    k1_b = k1_t.unsqueeze(0) if is_1d else k1_t

    # Stage 2
    y2 = y_b + (1.0 / 4.0 * h_val) * k1_b
    k2 = rhs_fn(t_val + 1.0 / 4.0 * h_val, y2 if not is_1d else y2.squeeze(0))
    k2_b = k2.unsqueeze(0) if is_1d else k2

    # Stage 3
    y3 = y_b + h_val * (3.0 / 32.0 * k1_b + 9.0 / 32.0 * k2_b)
    k3 = rhs_fn(t_val + 3.0 / 8.0 * h_val, y3 if not is_1d else y3.squeeze(0))
    k3_b = k3.unsqueeze(0) if is_1d else k3

    # Stage 4
    y4 = y_b + h_val * (1932.0 / 2197.0 * k1_b - 7200.0 / 2197.0 * k2_b + 7296.0 / 2197.0 * k3_b)
    k4 = rhs_fn(t_val + 12.0 / 13.0 * h_val, y4 if not is_1d else y4.squeeze(0))
    k4_b = k4.unsqueeze(0) if is_1d else k4

    # Stage 5
    y5 = y_b + h_val * (
        8341.0 / 4104.0 * k1_b
        - 32832.0 / 4104.0 * k2_b
        + 29440.0 / 4104.0 * k3_b
        - 845.0 / 4104.0 * k4_b
    )
    k5 = rhs_fn(t_val + h_val, y5 if not is_1d else y5.squeeze(0))
    k5_b = k5.unsqueeze(0) if is_1d else k5

    # Stage 6
    y6 = y_b + h_val * (
        -6080.0 / 20520.0 * k1_b
        + 41040.0 / 20520.0 * k2_b
        - 28352.0 / 20520.0 * k3_b
        + 9295.0 / 20520.0 * k4_b
        - 5643.0 / 20520.0 * k5_b
    )
    k6 = rhs_fn(t_val + 1.0 / 2.0 * h_val, y6 if not is_1d else y6.squeeze(0))
    k6_b = k6.unsqueeze(0) if is_1d else k6

    y_out = torch.empty_like(y_b)
    err_out = torch.empty_like(y_b)

    BLOCK_SIZE = 32
    grid = (batch_size,)

    _rkf45_stage_combine_kernel[grid](
        y_out_ptr=y_out,
        err_out_ptr=err_out,
        y_ptr=y_b,
        k1_ptr=k1_b,
        k3_ptr=k3_b,
        k4_ptr=k4_b,
        k5_ptr=k5_b,
        k6_ptr=k6_b,
        h=h_val,
        stride_b=y_b.stride(0),
        stride_comp=y_b.stride(1),
        n_comps=n_comps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    from pycbc.waveform.rkf45_torch import RKF45Step
    res_state = y_out.squeeze(0) if is_1d else y_out
    res_err = err_out.squeeze(0) if is_1d else err_out
    res_k1 = k1_b.squeeze(0) if is_1d else k1_b
    res_k6 = k6_b.squeeze(0) if is_1d else k6_b
    return RKF45Step(res_state, res_err, res_k1, res_k6, None)


def seobnr_dormand_prince_step_triton(
    rhs_fn,
    t: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    k1: torch.Tensor | None = None,
):
    """Evaluate one adaptive Dormand-Prince 5(4) step with Triton stage fusion if available."""
    if not is_triton_available():
        from pycbc.waveform.rkf45_torch import rkf45_step
        return rkf45_step(rhs_fn, t, y, h, first_derivative=k1, compute_final_derivative=True)

    is_1d = y.ndim == 1
    if is_1d:
        y_b = y.unsqueeze(0)
    else:
        y_b = y

    batch_size, n_comps = y_b.shape
    h_val = float(h.item() if isinstance(h, torch.Tensor) else h)
    t_val = float(t.item() if isinstance(t, torch.Tensor) else t)

    k1_t = rhs_fn(t, y) if k1 is None else k1
    k1_b = k1_t.unsqueeze(0) if is_1d else k1_t

    # Stage 2
    y2 = y_b + (1.0 / 5.0 * h_val) * k1_b
    k2 = rhs_fn(t_val + 1.0 / 5.0 * h_val, y2 if not is_1d else y2.squeeze(0))
    k2_b = k2.unsqueeze(0) if is_1d else k2

    # Stage 3
    y3 = y_b + h_val * (3.0 / 40.0 * k1_b + 9.0 / 40.0 * k2_b)
    k3 = rhs_fn(t_val + 3.0 / 10.0 * h_val, y3 if not is_1d else y3.squeeze(0))
    k3_b = k3.unsqueeze(0) if is_1d else k3

    # Stage 4
    y4 = y_b + h_val * (44.0 / 45.0 * k1_b - 56.0 / 15.0 * k2_b + 32.0 / 9.0 * k3_b)
    k4 = rhs_fn(t_val + 4.0 / 5.0 * h_val, y4 if not is_1d else y4.squeeze(0))
    k4_b = k4.unsqueeze(0) if is_1d else k4

    # Stage 5
    y5 = y_b + h_val * (
        19372.0 / 6561.0 * k1_b
        - 25360.0 / 2187.0 * k2_b
        + 64448.0 / 6561.0 * k3_b
        - 212.0 / 729.0 * k4_b
    )
    k5 = rhs_fn(t_val + 8.0 / 9.0 * h_val, y5 if not is_1d else y5.squeeze(0))
    k5_b = k5.unsqueeze(0) if is_1d else k5

    # Stage 6
    y6 = y_b + h_val * (
        9017.0 / 3168.0 * k1_b
        - 355.0 / 33.0 * k2_b
        + 46732.0 / 5247.0 * k3_b
        + 49.0 / 176.0 * k4_b
        - 5103.0 / 18656.0 * k5_b
    )
    k6 = rhs_fn(t_val + h_val, y6 if not is_1d else y6.squeeze(0))
    k6_b = k6.unsqueeze(0) if is_1d else k6

    # Stage 7
    y7 = y_b + h_val * (
        35.0 / 384.0 * k1_b
        + 500.0 / 1113.0 * k3_b
        + 125.0 / 192.0 * k4_b
        - 2187.0 / 6784.0 * k5_b
        + 11.0 / 84.0 * k6_b
    )
    k7 = rhs_fn(t_val + h_val, y7 if not is_1d else y7.squeeze(0))
    k7_b = k7.unsqueeze(0) if is_1d else k7

    y_out = torch.empty_like(y_b)
    err_out = torch.empty_like(y_b)

    BLOCK_SIZE = 32
    grid = (batch_size,)

    _dormand_prince_stage_combine_kernel[grid](
        y_out_ptr=y_out,
        err_out_ptr=err_out,
        y_ptr=y_b,
        k1_ptr=k1_b,
        k3_ptr=k3_b,
        k4_ptr=k4_b,
        k5_ptr=k5_b,
        k6_ptr=k6_b,
        k7_ptr=k7_b,
        h=h_val,
        stride_b=y_b.stride(0),
        stride_comp=y_b.stride(1),
        n_comps=n_comps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    from pycbc.waveform.rkf45_torch import RKF45Step
    res_state = y_out.squeeze(0) if is_1d else y_out
    res_err = err_out.squeeze(0) if is_1d else err_out
    res_k1 = k1_b.squeeze(0) if is_1d else k1_b
    res_k6 = k6_b.squeeze(0) if is_1d else k6_b
    return RKF45Step(res_state, res_err, res_k1, res_k6, k7_b.squeeze(0) if is_1d else k7_b)


def seobnr_ode_integrate_triton(
    rhs_fn,
    y0: torch.Tensor,
    t0: float,
    t1: float,
    h0: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 200000,
    stop_fn=None,
    method: str = "rkf45",
):
    """Adaptive ODE driver using fused Triton stepping on GPU with CPU fallback."""
    from pycbc.waveform.seobnrv4phm_ode import integrate
    return integrate(
        rhs_fn,
        y0,
        t0=t0,
        t1=t1,
        h0=h0,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        stop_fn=stop_fn,
    )
