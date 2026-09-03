"""Peak finding and local derivative utilities for SEOBNRv4PHM torch path."""

from __future__ import annotations

import torch


class _NaturalCubicSpline:
    """Fast natural cubic spline with derivative evaluation (torch-native / C++)."""

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
        self.y = y
        device = x.device
        dtype = x.dtype

        # Fast path via native C++ extension if available
        from pycbc.waveform._seobnr_native_ode import get_extension
        ext = get_extension() if (device.type == "cpu" and dtype == torch.float64) else None
        if ext is not None and hasattr(ext, "natural_spline_coeffs_native"):
            b, c, d = ext.natural_spline_coeffs_native(x, y)
            self.b = b.to(device=device, dtype=dtype)
            self.c = c.to(device=device, dtype=dtype)
            self.d = d.to(device=device, dtype=dtype)
            self._ext = ext
            return

        self._ext = None
        # Fallback fast NumPy Thomas algorithm
        import numpy as np
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        n = len(x_np)
        h = x_np[1:] - x_np[:-1]
        alpha = np.zeros(n, dtype=np.float64)
        alpha[1 : n - 1] = (3.0 / h[1:]) * (y_np[2:] - y_np[1:-1]) - (3.0 / h[:-1]) * (y_np[1:-1] - y_np[:-2])

        l = np.ones(n, dtype=np.float64)
        mu = np.zeros(n, dtype=np.float64)
        z = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            l[i] = 2.0 * (x_np[i + 1] - x_np[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = np.zeros(n, dtype=np.float64)
        b = np.zeros(n - 1, dtype=np.float64)
        d = np.zeros(n - 1, dtype=np.float64)
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y_np[j + 1] - y_np[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        self.b = torch.as_tensor(b, device=device, dtype=dtype)
        self.c = torch.as_tensor(c[:-1], device=device, dtype=dtype)
        self.d = torch.as_tensor(d, device=device, dtype=dtype)

    def eval_derivs(self, xq: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (value, 1st derivative, 2nd derivative) simultaneously at xq."""
        xq_t = torch.as_tensor(xq, device=self.x.device, dtype=self.x.dtype)
        if self._ext is not None and hasattr(self._ext, "natural_spline_eval_derivs_native") and xq_t.numel() == 1:
            val, d1, d2 = self._ext.natural_spline_eval_derivs_native(
                self.x, self.y, self.b, self.c, self.d, float(xq_t.item())
            )
            return (
                torch.tensor(val, device=self.x.device, dtype=self.x.dtype),
                torch.tensor(d1, device=self.x.device, dtype=self.x.dtype),
                torch.tensor(d2, device=self.x.device, dtype=self.x.dtype),
            )
        idx = torch.searchsorted(self.x, xq_t).item() - 1
        idx = max(0, min(idx, self.x.numel() - 2))
        dx = xq_t - self.x[idx]
        val = self.y[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]))
        d1 = self.b[idx] + 2.0 * self.c[idx] * dx + 3.0 * self.d[idx] * dx * dx
        d2 = 2.0 * self.c[idx] + 6.0 * self.d[idx] * dx
        return val, d1, d2

    def deriv(self, xq: float | torch.Tensor, order: int = 1) -> torch.Tensor:
        """Evaluate first/second derivative at xq."""
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        _, d1, d2 = self.eval_derivs(xq)
        return d1 if order == 1 else d2


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

    # Vectorized sliding window maximum detection (replaces scalar GPU sync loop)
    omega_3d = omega.view(1, 1, -1)
    padded = torch.nn.functional.pad(omega_3d, (window_width, window_width), mode="replicate")
    max_pooled = torch.nn.functional.max_pool1d(padded, kernel_size=2 * window_width + 1, stride=1).view(-1)

    # A candidate is an interior point equal to the local window maximum
    is_local_max = (omega == max_pooled)
    # Mask out boundary regions
    is_local_max[:window_width + 1] = False
    is_local_max[-window_width - 1:] = False

    candidates = torch.where(is_local_max, omega, torch.tensor(-float("inf"), dtype=omega.dtype, device=omega.device))
    idx_global = int(torch.argmax(candidates).item())

    # If no local maximum was found, idx_global will be 0 where candidates is -inf
    if not bool(is_local_max[idx_global].item()):
        idx_global = -1

    global_argmax = int(torch.argmax(omega).item())
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


def local_derivatives_all_tensor(
    series: torch.Tensor,
    t: torch.Tensor,
    t0: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the spline (value, 1st derivative, 2nd derivative) at ``t0`` in one pass."""
    if series.numel() != t.numel():
        raise ValueError("series and t must have the same length")
    if series.numel() < 4:
        idx = torch.searchsorted(t, torch.tensor(t0, device=t.device, dtype=t.dtype)).item()
        idx = max(1, min(idx, len(t) - 2))
        dt = t[idx + 1] - t[idx]
        val = series[idx]
        d1 = (series[idx + 1] - series[idx - 1]) / (2 * dt)
        d2 = (series[idx + 1] - 2 * series[idx] + series[idx - 1]) / (dt * dt)
        return val, d1, d2
    spline = _NaturalCubicSpline(t, series)
    return spline.eval_derivs(t0)


def local_derivatives_all(
    series: torch.Tensor,
    t: torch.Tensor,
    t0: float,
) -> tuple[float, float, float]:
    """Return host scalar (value, 1st derivative, 2nd derivative) at ``t0`` in one pass."""
    val, d1, d2 = local_derivatives_all_tensor(series, t, t0)
    return float(val), float(d1), float(d2)


def local_derivatives_tensor(
    series: torch.Tensor,
    t: torch.Tensor,
    t0: float,
    order: int = 2,
) -> torch.Tensor:
    """Return the spline derivative at ``t0`` without leaving its device."""

    if series.numel() != t.numel():
        raise ValueError("series and t must have the same length")
    if series.numel() < 4:
        # Fall back to simple central differences if we lack support points
        idx = torch.searchsorted(t, torch.tensor(t0, device=t.device, dtype=t.dtype)).item()
        idx = max(1, min(idx, len(t) - 2))
        dt = t[idx + 1] - t[idx]
        if order == 1:
            return (series[idx + 1] - series[idx - 1]) / (2 * dt)
        if order == 2:
            return (
                series[idx + 1] - 2 * series[idx] + series[idx - 1]
            ) / (dt * dt)
        raise ValueError("order must be 1 or 2")

    spline = _NaturalCubicSpline(t, series)
    if order == 1:
        return spline.deriv(t0, order=1)
    if order == 2:
        return spline.deriv(t0, order=2)
    raise ValueError("order must be 1 or 2")


def local_derivatives(
    series: torch.Tensor,
    t: torch.Tensor,
    t0: float,
    order: int = 2,
) -> float:
    """Return a host scalar spline derivative for control-flow callers."""

    return float(local_derivatives_tensor(series, t, t0, order=order))


__all__ = [
    "find_peak_time",
    "local_derivatives",
    "local_derivatives_tensor",
    "local_derivatives_all",
    "local_derivatives_all_tensor",
]
