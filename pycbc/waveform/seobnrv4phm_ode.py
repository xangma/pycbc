"""Lightweight torch ODE integrator helpers for SEOBNRv4PHM."""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class RKState:
    t: torch.Tensor
    y: torch.Tensor
    h: torch.Tensor


def _rk45_step_impl(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state: RKState,
    *,
    dydt_in: torch.Tensor | None = None,
    compute_dydt_out: bool = False,
) -> Tuple[RKState, torch.Tensor, torch.Tensor | None]:
    """One adaptive RKF45 (Fehlberg 4(5)) step.

    COMPLETE torch port of the GSL ``rkf45`` tableau used by
    ``gsl_odeiv_step_rkf45`` (invoked via LAL's AdaptiveRungeKuttaIntegrator;
    see LALAdaptiveRungeKuttaIntegrator.c:38-78). Returns the proposed new
    state and a per-component error estimate (5th–4th order difference).
    """
    t, y, h = state.t, state.y, state.h

    # k1..k6 follow the original GSL rkf45 coefficients and arithmetic order
    # (gsl/ode-initval/rkf45.c: ah/b/c/ec tables and rkf45_apply).
    k1 = f(t, y) if dydt_in is None else dydt_in
    k2 = f(t + 1.0 / 4.0 * h, y + 1.0 / 4.0 * h * k1)

    k3 = f(
        t + 3.0 / 8.0 * h,
        y + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2),
    )

    k4 = f(
        t + 12.0 / 13.0 * h,
        y
        + h
        * (
            1932.0 / 2197.0 * k1
            - 7200.0 / 2197.0 * k2
            + 7296.0 / 2197.0 * k3
        ),
    )

    k5 = f(
        t + h,
        y
        + h
        * (
            8341.0 / 4104.0 * k1
            - 32832.0 / 4104.0 * k2
            + 29440.0 / 4104.0 * k3
            - 845.0 / 4104.0 * k4
        ),
    )

    k6 = f(
        t + 1.0 / 2.0 * h,
        y
        + h
        * (
            -6080.0 / 20520.0 * k1
            + 41040.0 / 20520.0 * k2
            - 28352.0 / 20520.0 * k3
            + 9295.0 / 20520.0 * k4
            - 5643.0 / 20520.0 * k5
        ),
    )

    # 5th-order solution.
    d_i = (
        902880.0 / 7618050.0 * k1
        + 3953664.0 / 7618050.0 * k3
        + 3855735.0 / 7618050.0 * k4
        - 1371249.0 / 7618050.0 * k5
        + 277020.0 / 7618050.0 * k6
    )
    y5 = y + h * d_i

    dydt_out = f(t + h, y5) if compute_dydt_out else None

    # GSL computes the Fehlberg error estimate directly from the embedded
    # coefficient differences, not by subtracting the two accumulated states.
    err = (
        h
        * (
            1.0 / 360.0 * k1
            - 128.0 / 4275.0 * k3
            - 2197.0 / 75240.0 * k4
            + 1.0 / 50.0 * k5
            + 2.0 / 55.0 * k6
        )
    )
    return RKState(t + h, y5, h), err, dydt_out


def rk45_step(f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], state: RKState) -> Tuple[RKState, torch.Tensor]:
    """One adaptive RKF45 (Fehlberg 4(5)) step.

    COMPLETE torch port of the GSL ``rkf45`` tableau used by
    ``gsl_odeiv_step_rkf45`` (invoked via LAL's AdaptiveRungeKuttaIntegrator;
    see LALAdaptiveRungeKuttaIntegrator.c:38-78). Returns the proposed new
    state and a per-component error estimate (5th-4th order difference).
    """

    state_out, err, _ = _rk45_step_impl(f, state)
    return state_out, torch.abs(err)


def integrate(
    f,
    y0: torch.Tensor,
    t0: float,
    t1: float,
    h0: float,
    rtol=1e-6,
    atol=1e-9,
    max_steps=200000,
    stop_fn=None,
    h_min: float | None = None,
    progress_label: str | None = None,
    return_diagnostics: bool = False,
):
    """Simple adaptive RK45 driver (kept intentionally lightweight).

    Mirrors the control flow of the LAL AdaptiveRungeKutta driver invoked
    from LALSimIMRSpinPrecEOBv4P.c (inspiral propagation) but stays on
    torch tensors/devices. Not a full port; behaviourally compatible.

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> dy/dt
    y0 : torch.Tensor
        Initial state
    t0, t1 : float
        Start and end times
    h0 : float
        Initial step
    rtol, atol : float
        Error tolerances
    max_steps : int
        Safety cap on steps
    stop_fn : callable, optional
        Early-exit predicate stop_fn(t, y) -> bool. When provided, integration
        halts as soon as the predicate returns True (after accepting a step),
        allowing event-like termination without a full event solver.
    progress_label : str, optional
        Label included in debug progress lines.
    return_diagnostics : bool, optional
        Return ``(trajectory, diagnostics)`` instead of only the trajectory.
    """
    debug = os.environ.get("PYCBC_SEOBNRV4PHM_DEBUG", "0") not in ("0", "", "false", "False")
    progress_interval = float(os.environ.get("PYCBC_SEOBNRV4PHM_PROGRESS_INTERVAL", "0.5"))
    progress_label = "" if progress_label is None else f" {progress_label}"
    from pycbc.waveform.seobnrv4phm_constants import DELTA_T_MIN

    h_max = float(os.environ.get("PYCBC_SEOBNRV4PHM_HMAX", "0.0"))
    if h_min is None:
        h_min = float(os.environ.get("PYCBC_SEOBNRV4PHM_HMIN", str(DELTA_T_MIN)))
    else:
        h_min = float(h_min)
    retries_max = int(os.environ.get("PYCBC_SEOBNRV4PHM_RETRIES", "1"))
    t = torch.tensor(t0, dtype=y0.dtype, device=y0.device)
    y = y0
    h = torch.tensor(h0, dtype=y0.dtype, device=y0.device)
    traj = []
    last_log = time.time()
    retries_left = retries_max
    accepted_steps = 0
    rejected_steps = 0
    attempted_steps = 0
    hit_max_steps = False
    trace_limit = max(0, int(os.environ.get("PYCBC_SEOBNRV4PHM_TRACE_STEPS", "0")))
    step_trace = []

    def _record_step(**entry):
        if len(step_trace) < trace_limit:
            step_trace.append(entry)

    # LAL's NoInterpolate driver evaluates dydt once before entering the step
    # loop, then passes that cached derivative as rkf45's k1.  This matters for
    # SEOBNRv4PHM because the RHS intentionally leaves hcoeff state behind.
    dydt_in = f(t, y)
    for step in range(max_steps):
        attempted_steps = step + 1
        if (t + h) > t1:
            h = t1 - t
        h_trial = h
        try:
            state, err, dydt_out = _rk45_step_impl(
                f,
                RKState(t, y, h),
                dydt_in=dydt_in,
                compute_dydt_out=True,
            )
            trial_failed = (not torch.isfinite(state.y).all()) or (
                dydt_out is not None and not torch.isfinite(dydt_out).all()
            )
            trial_error = None
        except RuntimeError as exc:
            state = None
            err = None
            dydt_out = None
            trial_failed = True
            trial_error = exc

        if trial_failed:
            rejected_steps += 1
            retries_left -= 1
            h = h / torch.tensor(10.0, dtype=h.dtype, device=h.device)
            _record_step(
                step=int(step),
                action="reject_nonfinite",
                t=float(t),
                h=float(h_trial),
                h_next=float(h),
            )
            if debug:
                try:
                    r = float(y[0])
                    pr = float(y[1])
                    Lmag = float(torch.linalg.norm(y[3:6]))
                except Exception:
                    r = pr = Lmag = float("nan")
                reason = str(trial_error) if trial_error is not None else "non-finite trial state"
                print(
                    f"[seobnrv4phm] ODE{progress_label} reject non-finite "
                    f"step={step} t={float(t):.6f} h={float(h):.3e} "
                    f"r={r:.6f} pr={pr:.3e} L={Lmag:.6f}: {reason}",
                    flush=True,
                )
            if retries_left < 0 or (h_min > 0 and h < h_min):
                if traj:
                    if debug:
                        print(f"[seobnrv4phm] ODE{progress_label} stop after non-finite rejected trial", flush=True)
                    break
                raise RuntimeError(f"NaN/inf encountered in ODE trial at t={float(t):.6f}")
            continue

        scale = atol + rtol * torch.abs(state.y)
        err_ratio = torch.abs(err) / torch.abs(scale)
        err_norm = torch.max(err_ratio)
        err_argmax = int(torch.argmax(err_ratio).item())
        err_component = float(err[err_argmax])
        scale_component = float(scale[err_argmax])
        err_nan = torch.isnan(err_norm)
        if err_nan:
            err_norm = torch.tensor(float("inf"), device=err_norm.device, dtype=err_norm.dtype)
        if err_nan and debug:
            try:
                r = float(y[0])
                pr = float(y[1])
                Lmag = float(torch.linalg.norm(y[3:6]))
            except Exception:
                r = pr = Lmag = float("nan")
            print(f"[seobnrv4phm] ODE{progress_label} err_norm nan at step={step} t={float(t):.3f} h={float(h):.3e} r={r:.3e} pr={pr:.3e} L={Lmag:.3e}", flush=True)
        # Mirror GSL's gsl_odeiv_control_y_new/std_control_hadjust for rkf45:
        # reject only above 1.1, grow only below 0.5.  The old gsl_odeiv API
        # used by LAL reports rkf45 step order 5, and the grow exponent uses
        # order + 1.
        h_next = h
        accepted = True
        if err_norm > 1.1 and h > torch.finfo(h.dtype).tiny:
            ratio = 0.9 * torch.clamp(err_norm, min=torch.finfo(err_norm.dtype).tiny).pow(-1.0 / 5.0)
            ratio = torch.clamp(ratio, min=0.2)
            h_new = h * ratio
            if h_min > 0:
                h_new = torch.maximum(h_new, torch.tensor(h_min, dtype=h.dtype, device=h.device))
            if h_new < h:
                _record_step(
                    step=int(step),
                    action="reject_error",
                    t=float(t),
                    h=float(h_trial),
                    err_norm=float(err_norm),
                    err_argmax=err_argmax,
                    err_component=err_component,
                    scale_component=scale_component,
                    h_next=float(h_new),
                )
                h = h_new
                accepted = False
                rejected_steps += 1
                if debug and step < 20:
                    try:
                        r = float(y[0])
                        pr = float(y[1])
                        Lmag = float(torch.linalg.norm(y[3:6]))
                    except Exception:
                        r = pr = Lmag = float("nan")
                    print(
                        f"[seobnrv4phm] ODE{progress_label} reject error "
                        f"step={step} t={float(t):.6f} h={float(h):.3e} "
                        f"err={float(err_norm):.3e} r={r:.6f} pr={pr:.3e} L={Lmag:.6f}",
                        flush=True,
                    )
                continue
            h_next = h_new
        elif err_norm < 0.5:
            ratio = 0.9 * torch.clamp(err_norm, min=torch.finfo(err_norm.dtype).tiny).pow(-1.0 / 6.0)
            ratio = torch.clamp(ratio, min=1.0, max=5.0)
            h_next = h * ratio

        if accepted:
            _record_step(
                step=int(step),
                action="accept",
                t=float(t),
                h=float(h_trial),
                err_norm=float(err_norm),
                err_argmax=err_argmax,
                err_component=err_component,
                scale_component=scale_component,
                h_next=float(h_next),
            )
            # accept
            t, y = state.t, state.y
            if dydt_out is None:
                dydt_out = f(t, y)
            dydt_in = dydt_out
            dy_stop = dydt_in
            traj.append((t, y))
            accepted_steps += 1
            if debug and step < 20:
                try:
                    r = float(y[0])
                    pr = float(y[1])
                    Lmag = float(torch.linalg.norm(y[3:6]))
                    omega = float((Lmag / max(r * r, 1e-9)))
                except Exception:
                    r = pr = Lmag = omega = float("nan")
                print(f"[seobnrv4phm] ODE{progress_label} accept step={step} t={float(t):.6f} h={float(h):.3e} r={r:.6f} pr={pr:.3e} L={Lmag:.6f} omega={omega:.6e}", flush=True)
            if stop_fn is not None:
                if getattr(stop_fn, "_uses_derivative_estimate", False):
                    should_stop = stop_fn(t, y, dy_stop)
                else:
                    should_stop = stop_fn(t, y)
                if should_stop:
                    break
            if t >= t1:
                break
            retries_left = retries_max  # reset after a good step
        if debug and progress_interval > 0.0 and (step < 20 or step % 20000 == 0):
            now = time.time()
            if now - last_log > progress_interval:
                last_log = now
                try:
                    r = float(y[0])
                    omega = float((torch.linalg.norm(y[3:6]) / max(r * r, 1e-9)).item())
                except Exception:
                    r = omega = float("nan")
                print(
                    f"[seobnrv4phm] ODE{progress_label} step={step} "
                    f"accepted={accepted_steps} rejected={rejected_steps} "
                    f"t={float(t):.3f} r={r:.3f} omega={omega:.4f} h={float(h):.4e}",
                    flush=True,
                )
        h = h_next
        if h_max > 0:
            h = torch.minimum(h, torch.tensor(h_max, dtype=h.dtype, device=h.device))
    else:  # pragma: no cover
        hit_max_steps = True
        warnings.warn(
            "SEOBNRv4PHM ODE integrator hit max_steps; returning partial trajectory",
            stacklevel=2,
        )
    diagnostics = {
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "attempted_steps": int(attempted_steps),
        "max_steps": int(max_steps),
        "hit_max_steps": bool(hit_max_steps),
        "t_end": float(t),
        "h_final": float(h),
    }
    if trace_limit > 0:
        diagnostics["step_trace"] = step_trace
    return (traj, diagnostics) if return_diagnostics else traj
