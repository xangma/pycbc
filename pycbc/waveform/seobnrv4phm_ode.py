"""Lightweight torch ODE integrator helpers for SEOBNRv4PHM."""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from pycbc.waveform.rkf45_torch import rkf45_step as _shared_rkf45_step


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
    """Adapt the shared GSL-compatible RKF45 kernel to ``RKState``."""

    trial = _shared_rkf45_step(
        f,
        state.t,
        state.y,
        state.h,
        first_derivative=dydt_in,
        compute_final_derivative=compute_dydt_out,
    )
    return (
        RKState(state.t + state.h, trial.state, state.h),
        trial.error,
        trial.final_derivative,
    )


_ORIGINAL_RK45_STEP_IMPL = _rk45_step_impl


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
    return_tensors: bool = False,
    initial_prev_omega: float = 0.0,
    initial_prev_dr: float = 0.0,
    initial_omega_peaked: int = 0,
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
    return_tensors : bool, optional
        If True, return ``(t_buf[:accepted_steps], y_buf[:accepted_steps])``
        directly as a tuple of contiguous PyTorch tensors instead of
        allocating Python tuples ``[(t[i], y[i]) for i in range(...)]``.
        Defaults to False.
    initial_prev_omega : float, optional
        Starting orbital frequency memory for monotonic peak stopping checks.
    initial_prev_dr : float, optional
        Starting radial derivative memory for stop checks.
    initial_omega_peaked : int, optional
        Starting flag indicating if omega has already peaked.
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
    finfo_tiny = torch.finfo(y0.dtype).tiny
    h_min_t = (
        torch.tensor(h_min, dtype=y0.dtype, device=y0.device)
        if h_min > 0
        else None
    )
    h_max_t = (
        torch.tensor(h_max, dtype=y0.dtype, device=y0.device)
        if h_max > 0
        else None
    )
    t_buf = torch.empty(max_steps, dtype=y0.dtype, device=y0.device)
    y_buf = torch.empty((max_steps, *y0.shape), dtype=y0.dtype, device=y0.device)
    last_log = time.time()
    retries_left = retries_max
    accepted_steps = 0
    rejected_steps = 0
    attempted_steps = 0
    hit_max_steps = False
    trace_limit = (
        max(
            0,
            int(os.environ.get("PYCBC_SEOBNRV4PHM_TRACE_STEPS", "0")),
        )
        if return_diagnostics
        else 0
    )
    step_trace = []

    # Fast-path: native C++ RKF45 adaptive integrator when available
    native_eligible = (
        globals().get("_rk45_step_impl") is _ORIGINAL_RK45_STEP_IMPL
        and y0.device.type == "cpu"
        and y0.dtype == torch.float64
        and y0.ndim == 1
        and not debug
        and not (return_diagnostics and trace_limit > 0)
        and os.environ.get("PYCBC_SEOBNR_NATIVE_ODE", "1") not in ("0", "", "false", "False")
    )
    if native_eligible:
        try:
            from pycbc.waveform._seobnr_native_ode import get_extension
            ext = get_extension()
            if ext is not None:
                stop_mode = getattr(stop_fn, "_native_stop_mode", 0) if stop_fn is not None else 0
                res = ext.integrate_native(
                    f,
                    y0,
                    float(t0),
                    float(t1),
                    float(h0),
                    float(rtol),
                    float(atol),
                    int(max_steps),
                    float(h_min),
                    float(h_max),
                    stop_fn,
                    int(stop_mode),
                    float(initial_prev_omega),
                    float(initial_prev_dr),
                    int(initial_omega_peaked),
                    bool(return_diagnostics),
                )
                if return_diagnostics:
                    (t_out, y_out), diagnostics = res
                    if diagnostics.get("hit_max_steps", False):
                        warnings.warn(
                            "SEOBNRv4PHM ODE integrator hit max_steps; returning partial trajectory",
                            stacklevel=2,
                        )
                else:
                    t_out, y_out = res
                    if int(t_out.numel()) == int(max_steps) and max_steps > 0:
                        if int(t_out.numel()) > 0 and float(t_out[-1].item()) < float(t1):
                            warnings.warn(
                                "SEOBNRv4PHM ODE integrator hit max_steps; returning partial trajectory",
                                stacklevel=2,
                            )
                if return_tensors:
                    traj = (t_out, y_out)
                else:
                    traj = [(t_out[i], y_out[i]) for i in range(int(t_out.numel()))]
                if not return_diagnostics:
                    return traj
                return traj, diagnostics
        except Exception:
            pass  # Transparent fallback to pure Python/PyTorch ODE solver below

    def _record_step(
        *,
        step,
        action,
        t,
        h,
        h_next,
        err_norm=None,
        err_ratio=None,
        err=None,
        scale=None,
    ):
        """Materialize optional trace scalars only when they will be kept."""

        if len(step_trace) >= trace_limit:
            return
        entry = {
            "step": int(step),
            "action": action,
            "t": float(t.item() if isinstance(t, torch.Tensor) else t),
            "h": float(h.item() if isinstance(h, torch.Tensor) else h),
        }
        if err_norm is not None:
            err_argmax = int(torch.argmax(err_ratio).item())
            err_flat = err.flatten()
            scale_flat = scale.flatten()
            norm_val = float(err_norm.max().item() if isinstance(err_norm, torch.Tensor) else err_norm)
            entry.update(
                err_norm=norm_val,
                err_argmax=err_argmax,
                err_component=float(err_flat[err_argmax]),
                scale_component=float(scale_flat[err_argmax]),
            )
        entry["h_next"] = float(h_next.item() if isinstance(h_next, torch.Tensor) else h_next)
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
            h = h / 10.0
            _record_step(
                step=step,
                action="reject_nonfinite",
                t=t,
                h=h_trial,
                h_next=h,
            )
            if debug:
                try:
                    if y.ndim > 1:
                        r = float(y[0, 0])
                        pr = float(y[0, 1])
                        Lmag = float(torch.linalg.norm(y[0, 3:6]))
                    else:
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
                if accepted_steps > 0:
                    if debug:
                        print(f"[seobnrv4phm] ODE{progress_label} stop after non-finite rejected trial", flush=True)
                    break
                raise RuntimeError(f"NaN/inf encountered in ODE trial at t={float(t):.6f}")
            continue

        scale = atol + rtol * torch.abs(state.y)
        err_ratio = torch.abs(err) / torch.abs(scale)
        err_norm = torch.max(err_ratio, dim=-1).values
        err_nan = torch.isnan(err_norm)
        err_norm = torch.where(
            err_nan,
            torch.full_like(err_norm, float("inf")),
            err_norm,
        )
        worst_err_norm = torch.max(err_norm)
        if debug and bool(err_nan.any()):
            try:
                if y.ndim > 1:
                    r = float(y[0, 0])
                    pr = float(y[0, 1])
                    Lmag = float(torch.linalg.norm(y[0, 3:6]))
                else:
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
        if worst_err_norm > 1.1 and h > finfo_tiny:
            ratio = 0.9 * torch.clamp(worst_err_norm, min=finfo_tiny).pow(-1.0 / 5.0)
            ratio = torch.clamp(ratio, min=0.2)
            h_new = h * ratio
            if h_min_t is not None:
                h_new = torch.maximum(h_new, h_min_t)
            if h_new < h:
                _record_step(
                    step=step,
                    action="reject_error",
                    t=t,
                    h=h_trial,
                    err_norm=err_norm,
                    err_ratio=err_ratio,
                    err=err,
                    scale=scale,
                    h_next=h_new,
                )
                h = h_new
                accepted = False
                rejected_steps += 1
                if debug and step < 20:
                    try:
                        if y.ndim > 1:
                            r = float(y[0, 0])
                            pr = float(y[0, 1])
                            Lmag = float(torch.linalg.norm(y[0, 3:6]))
                        else:
                            r = float(y[0])
                            pr = float(y[1])
                            Lmag = float(torch.linalg.norm(y[3:6]))
                    except Exception:
                        r = pr = Lmag = float("nan")
                    print(
                        f"[seobnrv4phm] ODE{progress_label} reject error "
                        f"step={step} t={float(t):.6f} h={float(h):.3e} "
                        f"err={float(worst_err_norm):.3e} r={r:.6f} pr={pr:.3e} L={Lmag:.6f}",
                        flush=True,
                    )
                continue
            h_next = h_new
        elif worst_err_norm < 0.5:
            ratio = 0.9 * torch.clamp(worst_err_norm, min=finfo_tiny).pow(-1.0 / 6.0)
            ratio = torch.clamp(ratio, min=1.0, max=5.0)
            h_next = h * ratio

        if accepted:
            _record_step(
                step=step,
                action="accept",
                t=t,
                h=h_trial,
                err_norm=err_norm,
                err_ratio=err_ratio,
                err=err,
                scale=scale,
                h_next=h_next,
            )
            # accept
            t, y = state.t, state.y
            if dydt_out is None:
                dydt_out = f(t, y)
            dydt_in = dydt_out
            dy_stop = dydt_in
            t_buf[accepted_steps] = t
            y_buf[accepted_steps] = y
            accepted_steps += 1
            if debug and step < 20:
                try:
                    if y.ndim > 1:
                        r = float(y[0, 0])
                        pr = float(y[0, 1])
                        Lmag = float(torch.linalg.norm(y[0, 3:6]))
                    else:
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
                if isinstance(should_stop, torch.Tensor):
                    should_stop = bool(should_stop.all().item() if should_stop.numel() > 1 else should_stop.item())
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
                    if y.ndim > 1:
                        r = float(y[0, 0])
                        omega = float((torch.linalg.norm(y[0, 3:6]) / max(r * r, 1e-9)).item())
                    else:
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
        if h_max_t is not None:
            h = torch.minimum(h, h_max_t)
    else:  # pragma: no cover
        hit_max_steps = True
        warnings.warn(
            "SEOBNRv4PHM ODE integrator hit max_steps; returning partial trajectory",
            stacklevel=2,
        )
    t_out = t_buf[:accepted_steps]
    y_out = y_buf[:accepted_steps]
    if return_tensors:
        traj = (t_out, y_out)
    else:
        traj = [(t_out[i], y_out[i]) for i in range(accepted_steps)]
    if not return_diagnostics:
        return traj
    diagnostics = {
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "attempted_steps": int(attempted_steps),
        "max_steps": int(max_steps),
        "hit_max_steps": bool(hit_max_steps),
        "t_end": float(t.item() if isinstance(t, torch.Tensor) else t),
        "h_final": float(h.item() if isinstance(h, torch.Tensor) else h),
    }
    if trace_limit > 0:
        diagnostics["step_trace"] = step_trace
    return traj, diagnostics
