"""Peak finding and local derivative utilities for SEOBNRv4PHM torch path."""

from __future__ import annotations

import torch


class _NaturalCubicSpline:
    """Minimal natural cubic spline with derivative evaluation (torch-native)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Inputs must be 1D tensors.")
        if x.numel() != y.numel():
            raise ValueError("x and y must have the same length.")
        if x.numel() < 2:
            raise ValueError("Need at least two points for a spline.")
        if not torch.all(x[1:] > x[:-1]):
            raise ValueError("x must be strictly increasing.")

        self.x = x
        n = x.numel()
        device = x.device
        dtype = x.dtype

        h = x[1:] - x[:-1]
        alpha = torch.zeros(n, device=device, dtype=dtype)
        alpha[1 : n - 1] = (3.0 / h[1:]) * (y[2:] - y[1:-1]) - (3.0 / h[:-1]) * (y[1:-1] - y[:-2])

        l = torch.ones(n, device=device, dtype=dtype)
        mu = torch.zeros(n, device=device, dtype=dtype)
        z = torch.zeros(n, device=device, dtype=dtype)
        for i in range(1, n - 1):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = torch.zeros(n, device=device, dtype=dtype)
        b = torch.zeros(n - 1, device=device, dtype=dtype)
        d = torch.zeros(n - 1, device=device, dtype=dtype)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        self.b = b
        self.c = c[:-1]
        self.d = d

    def deriv(self, xq: float | torch.Tensor, order: int = 1) -> torch.Tensor:
        """Evaluate first/second derivative at xq."""
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        xq_t = torch.as_tensor(xq, device=self.x.device, dtype=self.x.dtype)
        idx = torch.searchsorted(self.x, xq_t).item() - 1
        idx = max(0, min(idx, self.x.numel() - 2))
        dx = xq_t - self.x[idx]
        if order == 1:
            return self.b[idx] + 2.0 * self.c[idx] * dx + 3.0 * self.d[idx] * dx * dx
        return 2.0 * self.c[idx] + 6.0 * self.d[idx] * dx


def find_peak_time(omega: torch.Tensor, t: torch.Tensor, window_width: int = 3) -> float:
    """Robust peak finder (torch-native port of XLALEOBFindRobustPeak,
    LALSimIMRSpinPrecEOBv4P.c:221-303).

    Scans with a sliding window to locate interior local maxima, keeps the
    largest one, compares against the global argmax, then refines the peak time
    by solving d/dt spline(omega) = 0 via bisection."""
    if omega.numel() != t.numel():
        raise ValueError("omega and t must have the same length")
    n = omega.numel()
    if n < 2 * window_width + 1:
        idx = torch.argmax(omega)
        return float(t[idx])

    # Find the strongest local maximum away from window boundaries
    curr_max = -float("inf")
    idx_global = -1
    for kk in range(n - window_width - 1, window_width, -1):
        lo = kk - window_width
        hi = kk + window_width + 1
        sl = omega[lo:hi]
        local_argmax = int(torch.argmax(sl))
        if local_argmax == 0 or local_argmax == sl.numel() - 1:
            continue
        val = float(sl[local_argmax])
        if val > curr_max:
            curr_max = val
            idx_global = lo + local_argmax

    global_argmax = int(torch.argmax(omega))
    # Fallback to the end if no local max or if the global max is much larger
    def _bad_peak(idx_g: int) -> bool:
        if idx_g < 0:
            return True
        if idx_g < 3 or idx_g > n - 4:
            return True
        rel_diff = (float(omega[global_argmax] - omega[idx_g]) / max(abs(float(omega[idx_g])), 1e-30))
        return rel_diff > 0.1

    if _bad_peak(idx_global):
        return float(t[-1])

    spline = _NaturalCubicSpline(t, omega)
    time1 = float(t[idx_global - 3])
    time2 = float(t[idx_global + 3])
    time_peak = time2
    omega_deriv1 = float(spline.deriv(time1))
    while time2 - time1 > 1.0e-8:
        time_peak = 0.5 * (time1 + time2)
        omega_deriv_mid = float(spline.deriv(time_peak))
        if omega_deriv_mid * omega_deriv1 < 0.0:
            time2 = time_peak
        else:
            omega_deriv1 = omega_deriv_mid
            time1 = time_peak
    return time_peak


def local_derivatives(series: torch.Tensor, t: torch.Tensor, t0: float, order: int = 2):
    """Spline derivative at t0 (torch port of the derivative eval inside
    XLALEOBFindRobustPeak, LALSimIMRSpinPrecEOBv4P.c:221-303)."""
    if series.numel() != t.numel():
        raise ValueError("series and t must have the same length")
    if series.numel() < 4:
        # Fall back to simple central differences if we lack support points
        idx = torch.searchsorted(t, torch.tensor(t0, device=t.device, dtype=t.dtype)).item()
        idx = max(1, min(idx, len(t) - 2))
        dt = t[idx + 1] - t[idx]
        if order == 1:
            return float((series[idx + 1] - series[idx - 1]) / (2 * dt))
        if order == 2:
            return float((series[idx + 1] - 2 * series[idx] + series[idx - 1]) / (dt * dt))
        raise ValueError("order must be 1 or 2")

    spline = _NaturalCubicSpline(t, series)
    if order == 1:
        return float(spline.deriv(t0, order=1))
    if order == 2:
        return float(spline.deriv(t0, order=2))
    raise ValueError("order must be 1 or 2")


__all__ = ["find_peak_time", "local_derivatives"]
