#! /usr/bin/env python
# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch-native implementation of the SEOBNRv4PHM waveform pipeline.

The native entry points implement the LAL SEOBNRv4PHM time-domain model and
PyCBC's TD→FD conditioning with torch operations. The explicit
``seobnrv4phm_fd_from_td`` helper remains available for CPU/LAL compatibility
and reference comparisons, but it is not used by the native entry points.

Activation (default is CPU/LAL path):
- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` or ``PYCBC_TORCH_NATIVE=1``
- Per-model: ``PYCBC_SEOBNRV4PHM_NATIVE=1``
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.special
import torch
from scipy.interpolate import CubicSpline

from pycbc import scheme as _scheme
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform import seobnrv4phm_dynamics as _dyn
from pycbc.waveform.seobnrv4phm_constants import _MRSUN_SI, _PC_SI, T_STEP_BACK
from pycbc.waveform.seobnrv4phm_nqc import (
    peak_addot_v4,
    peak_adot_v4,
    peak_amp_v4,
    peak_delta_t_v4,
    peak_omega_v4,
    peak_omegadot_v4,
)
from pycbc.waveform.seobnrv4phm_ode import integrate
from pycbc.waveform.seobnrv4phm_peak import find_peak_time, local_derivatives
from pycbc.waveform.torch_switches import torch_native_enabled

# ---------------------------------------------------------------------------
# Small helpers (shared with IMRPhenom torch ports)
# ---------------------------------------------------------------------------


def _next_pow_two(x: float) -> int:
    return int(2 ** math.ceil(math.log2(max(1.0, x))))


def _torch_log_fact(x):
    return torch.lgamma(x + 1.0)


_PREFIX_CACHE = {}

_LANCZOS_G = 7.0
_LOG_SQRT_TWO_PI = 0.91893853320467274178  # 0.5 * log(2*pi)
_LANCZOS_COEFFS = torch.tensor(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ],
    dtype=torch.float64,
)

_EOB_RD_EFOLDS = 10.0  # LALSimIMR.h:71
_EOB_RD_PATCH_EFOLDS = 40.0  # LALSimIMRSpinPrecEOBv4P.c:5435-5437


@dataclass
class _TrajectorySegments:
    """AdaS/HiS trajectory pieces needed by the PHM mode-building stages."""

    traj: list
    traj_adas: list
    traj_his: list
    index_start_his: int
    tstart_his: float
    adas_stop_state: dict
    his_stop_state: dict


def _double_fact(n: int) -> int:
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _calc_prefix(l: int, m: int, m1: float, m2: float, eta: float):
    """COMPLETE port of CalculateThisMultipolePrefix
    (LALSimIMREOBNewtonianMultipole.c:512-628)."""
    key = (l, m, m1, m2, eta)
    if key in _PREFIX_CACHE:
        return _PREFIX_CACHE[key]
    epsilon = (l + m) % 2
    sign = 1 if (m % 2 == 0) else -1
    total = m1 + m2
    x1 = m1 / total
    x2 = m2 / total
    if (m1 != m2) or sign == 1:
        c = x2 ** (l + epsilon - 1) + sign * x1 ** (l + epsilon - 1)
    else:
        c_lookup = {2: -1.0, 3: -1.0, 4: -0.5, 5: -0.5}
        c = c_lookup.get(l, 0.0)

    if epsilon == 0:
        n = (1j * m) ** l
        mult1 = 8.0 * math.pi / _double_fact(2 * l + 1)
        mult2 = math.sqrt(((l + 1) * (l + 2)) / (l * (l - 1)))
        n *= mult1 * mult2
    else:
        n = (1j * m) ** l
        n = -n
        mult1 = 16.0 * math.pi / _double_fact(2 * l + 1)
        mult2 = math.sqrt(((2 * l + 1) * (l + 2) * (l * l - m * m)) / ((2 * l - 1) * (l + 1) * l * (l - 1)))
        n *= 1j * mult1 * mult2
    prefix = n * eta * c
    _PREFIX_CACHE[key] = prefix
    return prefix


def _abs_scalar_sph_pi_over2(l: int, m: int) -> float:
    """COMPLETE port of XLALAbsScalarSphHarmThetaPiBy2
    (LALSimIMREOBNewtonianMultipole.c:272-307) using SciPy."""
    key = ("Y", l, m)
    if key in _PREFIX_CACHE:
        return _PREFIX_CACHE[key]
    leg = scipy.special.lpmv(abs(m), l, 0.0)
    if m < 0 and (abs(m) % 2 == 1):
        leg *= -1.0
    norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    val = abs(norm * leg)
    _PREFIX_CACHE[key] = val
    return val


def _scalar_sph_pi_over2(l: int, m: int, phi: torch.Tensor) -> torch.Tensor:
    """COMPLETE port of XLALScalarSphHarmThetaPiBy2 for tensor phase."""
    key = ("Y_complex_coeff", l, m)
    if key in _PREFIX_CACHE:
        coeff = _PREFIX_CACHE[key]
    else:
        leg = scipy.special.lpmv(abs(m), l, 0.0)
        if m < 0 and (abs(m) % 2 == 1):
            leg *= -1.0
        norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
        coeff = norm * leg
        _PREFIX_CACHE[key] = coeff
    coeff_t = torch.tensor(float(coeff), device=phi.device, dtype=phi.dtype)
    m_t = torch.tensor(float(m), device=phi.device, dtype=phi.dtype)
    return coeff_t * torch.exp(1j * m_t * phi)


def _loggamma_complex(z: torch.Tensor) -> torch.Tensor:
    """Lanczos log-Gamma (g=7) for complex z with Re(z)>0 (torch-only)."""
    complex_dtype = (
        torch.complex64
        if z.device.type == "mps" or z.dtype in (torch.float32, torch.complex64)
        else torch.complex128
    )
    comp = z.to(complex_dtype)
    zr = comp - 1.0
    coeffs = _LANCZOS_COEFFS.to(device=comp.device, dtype=comp.real.dtype)
    x = coeffs[0]
    for i in range(1, coeffs.numel()):
        x = x + coeffs[i] / (zr + float(i))
    t = zr + _LANCZOS_G + 0.5
    log_term = (zr + 0.5) * torch.log(t) - t
    return log_term + torch.log(x) + torch.tensor(_LOG_SQRT_TWO_PI, device=comp.device, dtype=comp.real.dtype)


def _tail_factor_complex(l: int, m: int, omega: torch.Tensor, H: torch.Tensor):
    """COMPLETE torch-native Tail T_lm (LALSimIMRSpinEOBFactorizedWaveformPrec.c:675-712)."""
    device = omega.device
    if device.type == "mps":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    else:
        real_dtype = (
            torch.float64
            if omega.dtype in (torch.float16, torch.float32, torch.bfloat16)
            else omega.dtype
        )
        complex_dtype = (
            torch.complex64
            if real_dtype in (torch.float16, torch.float32, torch.bfloat16)
            else torch.complex128
        )

    omega_r = omega.to(dtype=real_dtype)
    H_r = H.to(dtype=real_dtype)
    m_t = torch.tensor(float(m), device=device, dtype=real_dtype)

    k = m_t * omega_r
    k_abs = torch.clamp(torch.abs(k), min=1e-30)
    hathatk = H_r * k

    real_part = torch.full_like(hathatk, float(l + 1), dtype=real_dtype)
    z = torch.complex(real_part, -2.0 * hathatk)

    ln_gamma = _loggamma_complex(z)
    ln_gamma_l = torch.lgamma(torch.tensor(float(l + 1), device=device, dtype=real_dtype))
    phase_log = torch.log(k_abs * 4.0 / math.sqrt(math.e))
    T = torch.exp(
        ln_gamma
        - ln_gamma_l
        + torch.tensor(math.pi, device=device, dtype=real_dtype) * hathatk
        + 1j * (2.0 * hathatk * phase_log)
    )
    return T.to(dtype=complex_dtype)


def _sYlm_torch(s: int, l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Spin-weighted spherical harmonic (s=-2) for scalar theta, phi (torch-only).

    Uses the relation {}_sY_{lm} = (-1)^s sqrt((2l+1)/(4π)) d^l_{m,-s}(theta) e^{i m phi}
    with a direct small-d implementation (no SciPy dependency, stays on device).
    """

    def _wigner_d_element(l: int, mp: int, mm: int, beta_t: torch.Tensor):
        beta_t = beta_t.to(
            torch.float32 if beta_t.device.type == "mps" else torch.float64
        )
        cosb2 = torch.cos(beta_t * 0.5)
        sinb2 = torch.sin(beta_t * 0.5)
        k_min = max(0, mm - mp)
        k_max = min(l + mm, l - mp)
        pref = math.sqrt(
            math.factorial(l + mm)
            * math.factorial(l - mm)
            * math.factorial(l + mp)
            * math.factorial(l - mp)
        )
        out = torch.zeros_like(beta_t)
        for k in range(k_min, k_max + 1):
            denom = (
                math.factorial(l + mm - k)
                * math.factorial(k)
                * math.factorial(mp - mm + k)
                * math.factorial(l - mp - k)
            )
            coef = pref / denom
            sign = -1.0 if ((k - mp + mm) % 2) else 1.0
            out = out + sign * coef * (cosb2 ** (2 * l + mm - mp - 2 * k)) * (sinb2 ** (mp - mm + 2 * k))
        return out.to(beta_t.dtype)

    mp = m
    d_l = _wigner_d_element(l, mp, -s, theta)
    norm = math.sqrt((2 * l + 1) / (4.0 * math.pi))
    return ((-1.0) ** s) * norm * d_l * torch.exp(1j * mp * phi)


def _target_device_dtypes() -> Tuple[torch.device, torch.dtype, torch.dtype]:
    """Return (device, real_dtype, complex_dtype) for the active torch scheme."""
    try:
        state = _scheme.mgr.state
        device = getattr(state, "device", None)
        dtype = getattr(state, "dtype", None)
    except Exception:  # pragma: no cover
        state = None
        device = None
        dtype = None

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Apple MPS has no float64 or complex128 kernels. TorchScheme does not
    # otherwise prescribe a dtype, so select the highest supported precision.
    if device.type == "mps":
        return device, torch.float32, torch.complex64

    if dtype in (torch.float32, torch.complex64, np.float32, np.complex64):
        return device, torch.float32, torch.complex64
    if dtype in (torch.float16, torch.bfloat16):
        return device, torch.float16, torch.complex64
    return device, torch.float64, torch.complex128


_DEFAULT_ONLY_ORDER_KEYS = (
    "phase_order",
    "spin_order",
    "tidal_order",
    "amplitude_order",
    "eccentricity_order",
)
_UNSUPPORTED_ZERO_KEYS = (
    "eccentricity",
    "mean_per_ano",
    "lambda1",
    "lambda2",
    "dquad_mon1",
    "dquad_mon2",
    "lambda_octu1",
    "lambda_octu2",
    "quadfmode1",
    "quadfmode2",
    "octufmode1",
    "octufmode2",
    "frame_axis",
    "modes_choice",
    "side_bands",
    "dchi0",
    "dchi1",
    "dchi2",
    "dchi3",
    "dchi4",
    "dchi5",
    "dchi5l",
    "dchi6",
    "dchi6l",
    "dchi7",
    "dalpha1",
    "dalpha2",
    "dalpha3",
    "dalpha4",
    "dalpha5",
    "dbeta1",
    "dbeta2",
    "dbeta3",
)


def _is_nonzero(value) -> bool:
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def _is_default_order(value) -> bool:
    try:
        return float(value) == -1.0 and int(value) == -1
    except (TypeError, ValueError, OverflowError):
        return False


def seobnrv4phm_native_supported(params) -> bool:
    """Return whether ``params`` are covered by the native PHM pipeline."""

    if params.get("approximant", "SEOBNRv4PHM") != "SEOBNRv4PHM":
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0)) for key in _UNSUPPORTED_ZERO_KEYS
    ):
        return False
    if params.get("numrel_data", ""):
        return False

    # LAL's implementation accepts either label order, but the independent
    # port currently follows the m1 >= m2 coefficient convention explicitly.
    try:
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        long_asc_nodes = float(params.get("long_asc_nodes", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (mass1, mass2, long_asc_nodes)):
        return False
    if mass1 < mass2:
        return False

    try:
        _dyn.normalize_mode_array(params.get("mode_array"))
    except (TypeError, ValueError, OverflowError):
        return False
    return True


# ---------------------------------------------------------------------------
# Vectorized rho/delta helpers (torch, no per-sample Python loops)
# ---------------------------------------------------------------------------


def _spin_combos(params: _dyn.EOBParams, device, dtype):
    m1 = torch.tensor(params.mass1, device=device, dtype=dtype)
    m2 = torch.tensor(params.mass2, device=device, dtype=dtype)
    chi1z = torch.tensor(params.spin1z, device=device, dtype=dtype)
    chi2z = torch.tensor(params.spin2z, device=device, dtype=dtype)
    m = m1 + m2
    dM = (m1 - m2) / m
    chiS = 0.5 * (chi1z + chi2z)
    chiA = 0.5 * (chi1z - chi2z)
    return chiS, chiA, dM


def _rho_aux_torch(l: int, m: int, v: torch.Tensor, params: _dyn.EOBParams, *, waveform: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return rho_lm and aux_f_lm using the shared factorized helper."""
    rho, aux, _ = _dyn._factorized_rho_aux_delta(l, m, v, params, waveform=waveform, H=None)
    return rho, aux


def _delta_torch(l: int, m: int, v: torch.Tensor, params: _dyn.EOBParams, H: torch.Tensor | None = None) -> torch.Tensor:
    """Delta_lm phase via shared factorized helper (torch)."""
    _, _, delta = _dyn._factorized_rho_aux_delta(l, m, v, params, waveform=True, H=H)
    return delta


def _debug_enabled():
    return os.environ.get("PYCBC_SEOBNRV4PHM_DEBUG", "0") not in ("0", "", "false", "False")


def _dbg(msg: str):
    if _debug_enabled():
        print(f"[seobnrv4phm] {msg}", flush=True)


def _integration_max_steps(label: str, default: int) -> int:
    """Return stage-specific diagnostic max-step cap for PHM integration."""

    label_key = f"PYCBC_SEOBNRV4PHM_{label.upper()}_MAX_STEPS"
    for key in (label_key, "PYCBC_SEOBNRV4PHM_MAX_STEPS"):
        value = os.environ.get(key)
        if value not in (None, ""):
            return max(0, int(value))
    return int(default)


def _interp_series(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear interpolation for monotonic 1D data (torch-only)."""
    idx = torch.searchsorted(x, x_new)
    idx = torch.clamp(idx, 1, len(x) - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    w = (x_new - x0) / torch.clamp(x1 - x0, min=torch.finfo(x.dtype).eps)
    while w.ndim < y0.ndim:
        w = w.unsqueeze(-1)
    return y0 + w * (y1 - y0)


def _scipy_cubic_available(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> bool:
    return (
        x.device.type == "cpu"
        and x_new.device.type == "cpu"
        and y.device.type == "cpu"
        and not x.requires_grad
        and not x_new.requires_grad
        and not y.requires_grad
        and y.dtype in (torch.float32, torch.float64)
    )


def _interp_series_cubic_scipy(
    x_new: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    derivative: bool = False,
) -> torch.Tensor:
    cs = CubicSpline(
        x.detach().numpy(),
        y.detach().numpy(),
        bc_type="natural",
        axis=0,
        extrapolate=True,
    )
    out = cs(x_new.detach().numpy(), 1 if derivative else 0)
    return torch.as_tensor(out, device=y.device, dtype=y.dtype)


def _nearest_index_increasing(x: torch.Tensor, value: float) -> int:
    """Port of FindClosestIndex for monotonic vectors."""
    if x.numel() <= 1:
        return 0
    value_t = torch.as_tensor(value, device=x.device, dtype=x.dtype)
    idx = int(torch.searchsorted(x, value_t, right=True).item()) - 1
    idx = max(0, min(idx, x.numel() - 1))
    if idx < x.numel() - 1:
        if torch.abs(x[idx] - value_t) > torch.abs(x[idx + 1] - value_t):
            idx += 1
    return idx


def _last_index_leq_increasing(x: torch.Tensor, value: float) -> int:
    """Last index with x[index] <= value for monotonic vectors."""
    if x.numel() <= 1:
        return 0
    value_t = torch.as_tensor(value, device=x.device, dtype=x.dtype)
    idx = int(torch.searchsorted(x, value_t, right=True).item()) - 1
    return max(0, min(idx, x.numel() - 1))


def _interp_series_cubic(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Natural cubic spline interpolation (dense-output) matching
    XLALAdaptiveRungeKutta4 (LALAdaptiveRungeKuttaIntegrator.c:1074-1270).

    Falls back to linear interpolation for <3 support points. CPU tensors use
    SciPy's natural spline path for speed; device/autograd tensors stay in torch.
    """

    if x.numel() < 3:
        return _interp_series(x_new, x, y)

    if _scipy_cubic_available(x_new, x, y):
        return _interp_series_cubic_scipy(
            x_new.to(dtype=y.dtype),
            x.to(dtype=y.dtype),
            y,
        )

    dtype = y.dtype
    device = y.device

    x = x.to(dtype=dtype)
    x_new = x_new.to(dtype=dtype)

    squeeze = False
    y_in = y
    if y_in.ndim == 1:
        y_in = y_in.unsqueeze(1)
        squeeze = True

    n, dim = y_in.shape
    h = torch.clamp(x[1:] - x[:-1], min=torch.finfo(dtype).tiny)

    alpha = 3.0 * (
        (y_in[2:] - y_in[1:-1]) / h[1:].unsqueeze(-1)
        - (y_in[1:-1] - y_in[:-2]) / h[:-1].unsqueeze(-1)
    )

    l = torch.ones(n, device=device, dtype=dtype)
    mu = torch.zeros(n, device=device, dtype=dtype)
    z = torch.zeros((n, dim), device=device, dtype=dtype)

    for i in range(1, n - 1):
        l_i = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        l[i] = l_i
        mu[i] = h[i] / torch.clamp(l_i, min=torch.finfo(dtype).tiny)
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / torch.clamp(l_i, min=torch.finfo(dtype).tiny)

    c = torch.zeros((n, dim), device=device, dtype=dtype)
    b = torch.zeros((n - 1, dim), device=device, dtype=dtype)
    d = torch.zeros((n - 1, dim), device=device, dtype=dtype)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        inv_h = 1.0 / h[j]
        b[j] = (y_in[j + 1] - y_in[j]) * inv_h - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) * (inv_h / 3.0)

    idx = torch.searchsorted(x, x_new, right=False)
    idx = torch.clamp(idx, 1, n - 1) - 1
    dx = (x_new - x[idx]).unsqueeze(-1)

    out = y_in[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

    if squeeze:
        out = out.squeeze(-1)
    return out


def _interp_series_cubic_derivative(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Derivative of the natural cubic spline used by _interp_series_cubic."""

    if x.numel() < 3:
        idx = torch.searchsorted(x, x_new)
        idx = torch.clamp(idx, 1, len(x) - 1) - 1
        slope = (y[idx + 1] - y[idx]) / torch.clamp(x[idx + 1] - x[idx], min=torch.finfo(y.dtype).eps)
        return slope

    if _scipy_cubic_available(x_new, x, y):
        return _interp_series_cubic_scipy(
            x_new.to(dtype=y.dtype),
            x.to(dtype=y.dtype),
            y,
            derivative=True,
        )

    dtype = y.dtype
    device = y.device

    x = x.to(dtype=dtype)
    x_new = x_new.to(dtype=dtype)

    squeeze = False
    y_in = y
    if y_in.ndim == 1:
        y_in = y_in.unsqueeze(1)
        squeeze = True

    n, dim = y_in.shape
    h = torch.clamp(x[1:] - x[:-1], min=torch.finfo(dtype).tiny)

    alpha = 3.0 * (
        (y_in[2:] - y_in[1:-1]) / h[1:].unsqueeze(-1)
        - (y_in[1:-1] - y_in[:-2]) / h[:-1].unsqueeze(-1)
    )

    l = torch.ones(n, device=device, dtype=dtype)
    mu = torch.zeros(n, device=device, dtype=dtype)
    z = torch.zeros((n, dim), device=device, dtype=dtype)

    for i in range(1, n - 1):
        l_i = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        l[i] = l_i
        mu[i] = h[i] / torch.clamp(l_i, min=torch.finfo(dtype).tiny)
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / torch.clamp(l_i, min=torch.finfo(dtype).tiny)

    c = torch.zeros((n, dim), device=device, dtype=dtype)
    b = torch.zeros((n - 1, dim), device=device, dtype=dtype)
    d = torch.zeros((n - 1, dim), device=device, dtype=dtype)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        inv_h = 1.0 / h[j]
        b[j] = (y_in[j + 1] - y_in[j]) * inv_h - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) * (inv_h / 3.0)

    idx = torch.searchsorted(x, x_new, right=False)
    idx = torch.clamp(idx, 1, n - 1) - 1
    dx = (x_new - x[idx]).unsqueeze(-1)

    out = b[idx] + 2.0 * c[idx] * dx + 3.0 * d[idx] * dx * dx

    if squeeze:
        out = out.squeeze(-1)
    return out


def _finite_diff(series: torch.Tensor, dt: float) -> torch.Tensor:
    """Centered finite difference with endpoint fallbacks."""
    out = torch.zeros_like(series)
    out[1:-1] = (series[2:] - series[:-2]) / (2.0 * dt)
    out[0] = (series[1] - series[0]) / dt
    out[-1] = (series[-1] - series[-2]) / dt
    return out


def _finite_diff_nonuniform(series: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Centered first derivative on a monotonic, non-uniform time vector."""
    out = torch.zeros_like(series)
    if len(series) < 2:
        return out
    dt0 = torch.clamp(t[1] - t[0], min=torch.finfo(t.dtype).eps)
    dtn = torch.clamp(t[-1] - t[-2], min=torch.finfo(t.dtype).eps)
    out[0] = (series[1] - series[0]) / dt0
    out[-1] = (series[-1] - series[-2]) / dtn
    if len(series) > 2:
        denom = torch.clamp(t[2:] - t[:-2], min=torch.finfo(t.dtype).eps)
        out[1:-1] = (series[2:] - series[:-2]) / denom
    return out


def _unwrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Torch-friendly unwrap akin to numpy.unwrap (2pi jumps)."""
    if len(theta) < 2:
        return theta
    diff = torch.diff(theta)
    twopi = 2.0 * math.pi
    diff_mod = (diff + math.pi) % twopi - math.pi
    diff_mod = torch.where((diff_mod == -math.pi) & (diff > 0), torch.full_like(diff_mod, math.pi), diff_mod)
    unwrapped = torch.cat([theta[:1], theta[0] + torch.cumsum(diff_mod, dim=0)])
    return unwrapped


def _euler_from_L(Lvec: torch.Tensor, dt: float):
    """Return alpha, beta, gamma (minimal-rotation) from L(t)."""
    Lmag = torch.linalg.norm(Lvec, dim=1)
    Lhat = Lvec / torch.clamp(Lmag.unsqueeze(1), min=1e-15)
    alpha = torch.atan2(Lhat[:, 1], Lhat[:, 0])
    beta = torch.acos(torch.clamp(Lhat[:, 2], min=-1.0, max=1.0))
    alpha_dot = _finite_diff(alpha, dt)
    gamma = torch.zeros_like(alpha)
    cosb = torch.cos(beta)
    for i in range(1, len(alpha)):
        gamma[i] = gamma[i - 1] - alpha_dot[i - 1] * cosb[i - 1] * dt
    return alpha, beta, gamma, Lhat


def _build_J_frame(J_hat: torch.Tensor):
    """Orthonormal final-J frame basis (SEOBBuildJframeVectors)."""
    device = J_hat.device
    dtype = J_hat.dtype
    J_norm = torch.clamp(torch.linalg.norm(J_hat), min=1e-15)
    e3 = J_hat / J_norm

    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    ex_dot = torch.dot(ex, e3)
    ey_dot = torch.dot(ey, e3)
    lambda_x = 1.0 - torch.abs(ex_dot)

    def _project_and_normalize(ref, ref_dot):
        projected = ref - ref_dot * e3
        return projected / torch.clamp(torch.linalg.norm(projected), min=1e-15)

    if float(lambda_x) > 1.0e-4:
        e1 = _project_and_normalize(ex, ex_dot)
    elif float(lambda_x) < 1.0e-5:
        e1 = _project_and_normalize(ey, ey_dot)
    else:
        weight_x = (lambda_x - 1.0e-5) / (1.0e-4 - 1.0e-5)
        weight_y = 1.0 - weight_x
        e1 = (
            weight_x * _project_and_normalize(ex, ex_dot)
            + weight_y * _project_and_normalize(ey, ey_dot)
        )

    e1 = e1 / torch.clamp(torch.linalg.norm(e1), min=1e-15)
    e2 = torch.cross(e3, e1, dim=0)
    e2 = e2 / torch.clamp(torch.linalg.norm(e2), min=1e-15)
    return e1, e2, e3


def _euler_from_basis(e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor):
    """Euler ZYZ angles from basis vectors (LALSimIMRSpinPrecEOBEulerAngles.c:7-33)."""
    alpha = torch.atan2(e3[1], e3[0])
    beta = torch.acos(torch.clamp(e3[2], min=-1.0, max=1.0))
    gamma = torch.atan2(e2[2], -e1[2])
    return alpha, beta, gamma


def _initial_gamma_from_sep_vector(
    Lhat0: torch.Tensor,
    n_hat0: torch.Tensor,
    e1J: torch.Tensor,
    e2J: torch.Tensor,
    e3J: torch.Tensor,
):
    """Initial gamma from LAL's actual initial ``(n, lambda, Z)`` P-frame."""

    e1P = n_hat0 - torch.dot(n_hat0, Lhat0) * Lhat0
    e1P = e1P / torch.clamp(torch.linalg.norm(e1P), min=1.0e-15)
    e2P = torch.cross(Lhat0, e1P, dim=0)
    e2P = e2P / torch.clamp(torch.linalg.norm(e2P), min=1.0e-15)

    e1P_J = torch.stack(
        [torch.dot(e1P, e1J), torch.dot(e1P, e2J), torch.dot(e1P, e3J)]
    )
    e2P_J = torch.stack(
        [torch.dot(e2P, e1J), torch.dot(e2P, e2J), torch.dot(e2P, e3J)]
    )
    return torch.atan2(e2P_J[2], -e1P_J[2])


def _initial_gamma_from_sep(
    Lhat0: torch.Tensor,
    e1J: torch.Tensor,
    e2J: torch.Tensor,
    e3J: torch.Tensor,
    phi0: torch.Tensor,
):
    """Compatibility helper that reconstructs ``n`` from an inertial xy angle."""

    device = Lhat0.device
    dtype = Lhat0.dtype
    n_raw = torch.stack(
        [
            torch.cos(phi0),
            torch.sin(phi0),
            torch.tensor(0.0, device=device, dtype=dtype),
        ]
    )
    return _initial_gamma_from_sep_vector(Lhat0, n_raw, e1J, e2J, e3J)


def _integrate_minimal_rotation_gamma_times(alpha: torch.Tensor, beta: torch.Tensor, t: torch.Tensor, gamma0: torch.Tensor):
    """Integrate gamma_dot = -alpha_dot*cos(beta) from cubic splines."""
    if len(alpha) == 0:
        return alpha
    gamma = torch.zeros_like(alpha)
    gamma[0] = gamma0
    if len(alpha) == 1:
        return gamma

    t0 = t[:-1]
    h = t[1:] - t0
    nodes = torch.stack(
        [
            t0,
            t0 + 0.25 * h,
            t0 + 0.5 * h,
            t0 + 0.75 * h,
            t[1:],
        ],
        dim=1,
    )
    flat_nodes = nodes.reshape(-1)
    alpha_dot = _interp_series_cubic_derivative(flat_nodes, t, alpha).reshape(-1, 5)
    beta_nodes = _interp_series_cubic(flat_nodes, t, beta).reshape(-1, 5)
    integrand = -alpha_dot * torch.cos(beta_nodes)
    weights = torch.tensor([7.0, 32.0, 12.0, 32.0, 7.0], device=alpha.device, dtype=alpha.dtype)
    increments = (h / 90.0) * torch.sum(integrand * weights.unsqueeze(0), dim=1)
    gamma[1:] = gamma0 + torch.cumsum(increments, dim=0)
    return gamma


def _integrate_minimal_rotation_gamma(alpha: torch.Tensor, beta: torch.Tensor, dt: float, gamma0: torch.Tensor):
    """Uniform-grid wrapper for the minimal-rotation gamma integral."""
    t = torch.arange(len(alpha), device=alpha.device, dtype=alpha.dtype) * dt
    return _integrate_minimal_rotation_gamma_times(alpha, beta, t, gamma0)


def _euler_j2p(
    Lvec: torch.Tensor,
    e1J: torch.Tensor,
    e2J: torch.Tensor,
    e3J: torch.Tensor,
    dt: float | None = None,
    *,
    phi0: torch.Tensor | None = None,
    n_hat0: torch.Tensor | None = None,
    t_vec: torch.Tensor | None = None,
):
    """Euler angles from J-frame to P-frame (minimal rotation).

    Mirrors SEOBEulerJ2PFromDynamics (LALSimIMRSpinPrecEOBv4P.c:3715-3831):
    Z-frame is Lhat; alpha/beta from projection on the J-frame basis; gamma via
    minimal-rotation integral with initial offset set by the (n, lambda, L)
    triad (n_hat0 gives LAL's separation direction at t=0; phi0 is retained
    for tests and legacy callers).
    """
    basis = torch.stack([e1J, e2J, e3J], dim=1)  # columns are J-frame axes
    Lmag = torch.linalg.norm(Lvec, dim=1)
    Lhat = Lvec / torch.clamp(Lmag.unsqueeze(1), min=1e-15)
    proj = Lhat @ basis  # components in J-frame
    alpha = torch.atan2(proj[:, 1], proj[:, 0])
    beta = torch.acos(torch.clamp(proj[:, 2], min=-1.0, max=1.0))
    alpha = _unwrap_angle(alpha)
    if n_hat0 is not None:
        gamma0 = _initial_gamma_from_sep_vector(Lhat[0], n_hat0, e1J, e2J, e3J)
    elif phi0 is not None:
        gamma0 = _initial_gamma_from_sep(Lhat[0], e1J, e2J, e3J, phi0)
    else:
        gamma0 = torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)
    if t_vec is None:
        if dt is None:
            raise ValueError("either dt or t_vec is required for Euler gamma integration")
        gamma = _integrate_minimal_rotation_gamma(alpha, beta, dt, gamma0)
    else:
        gamma = _integrate_minimal_rotation_gamma_times(alpha, beta, t_vec.to(dtype=alpha.dtype), gamma0)
    gamma = _unwrap_angle(gamma)
    return alpha, beta, gamma


def _extend_euler_post_merger(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, *, dt: float, n_extra: int, prec_rate: torch.Tensor):
    """Post-merger Euler extension: simple precession around J (LALSimIMRSpinPrecEOBv4P.c:3908-3951)."""
    if n_extra <= 0 or float(prec_rate) == 0.0:
        return alpha, beta, gamma
    device = alpha.device
    dtype = alpha.dtype
    t = torch.arange(1, n_extra + 1, device=device, dtype=dtype) * dt
    alpha_attach = alpha[-1]
    beta_attach = beta[-1]
    gamma_attach = gamma[-1]
    cosb = torch.cos(beta_attach)
    alpha_ext = alpha_attach + prec_rate * t
    beta_ext = torch.full_like(alpha_ext, beta_attach)
    gamma_ext = gamma_attach - cosb * prec_rate * t
    alpha_out = torch.cat([alpha, alpha_ext])
    beta_out = torch.cat([beta, beta_ext])
    gamma_out = torch.cat([gamma, gamma_ext])
    return alpha_out, beta_out, gamma_out


def _extend_euler_from_attach(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    index_start: int,
    target_len: int,
    dt: float,
    prec_rate: torch.Tensor,
):
    """Trim Euler dynamics at attachment and extend through ringdown.

    LAL's SEOBEulerJ2PFromDynamics fills J->P angles only up to
    indexJoinAttach - 1, then SEOBEulerJ2PPostMergerExtension starts at
    indexJoinAttach using the previous sample as the attachment state.
    """
    if target_len <= 0 or len(alpha) == 0:
        return alpha[:0], beta[:0], gamma[:0]

    index_start = max(1, min(int(index_start), len(alpha), int(target_len)))
    alpha_out, beta_out, gamma_out = _extend_euler_post_merger(
        alpha[:index_start],
        beta[:index_start],
        gamma[:index_start],
        dt=dt,
        n_extra=max(0, int(target_len) - index_start),
        prec_rate=prec_rate,
    )
    if len(alpha_out) < target_len:
        pad = int(target_len) - len(alpha_out)
        alpha_out = torch.cat([alpha_out, alpha_out[-1:].repeat(pad)])
        beta_out = torch.cat([beta_out, beta_out[-1:].repeat(pad)])
        gamma_out = torch.cat([gamma_out, gamma_out[-1:].repeat(pad)])
    elif len(alpha_out) > target_len:
        alpha_out = alpha_out[:target_len]
        beta_out = beta_out[:target_len]
        gamma_out = gamma_out[:target_len]
    return alpha_out, beta_out, gamma_out


def _extend_euler_from_attach_times(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    t_vec: torch.Tensor,
    *,
    index_start: int,
    prec_rate: torch.Tensor,
):
    """Post-attach Euler extension on LAL's possibly non-uniform P-mode grid."""
    target_len = int(t_vec.numel())
    if target_len <= 0 or len(alpha) == 0:
        empty = alpha[:0]
        return empty, empty, empty

    index_start = max(1, min(int(index_start), len(alpha), target_len))
    alpha_out = torch.zeros(target_len, device=alpha.device, dtype=alpha.dtype)
    beta_out = torch.zeros_like(alpha_out)
    gamma_out = torch.zeros_like(alpha_out)
    alpha_out[:index_start] = alpha[:index_start]
    beta_out[:index_start] = beta[:index_start]
    gamma_out[:index_start] = gamma[:index_start]

    if index_start < target_len:
        time_attach = t_vec[index_start - 1].to(dtype=alpha.dtype)
        dt_attach = t_vec[index_start:].to(dtype=alpha.dtype) - time_attach
        alpha_attach = alpha[index_start - 1]
        beta_attach = beta[index_start - 1]
        gamma_attach = gamma[index_start - 1]
        alpha_out[index_start:] = alpha_attach + prec_rate * dt_attach
        beta_out[index_start:] = beta_attach
        gamma_out[index_start:] = gamma_attach - torch.cos(beta_attach) * prec_rate * dt_attach
    return alpha_out, beta_out, gamma_out


def _zero_euler_angles_like(t_vec: torch.Tensor):
    """LAL SpinsAlmostAligned branch leaves all Euler angles at zero."""
    zeros = torch.zeros_like(t_vec)
    return zeros, zeros.clone(), zeros.clone()


def _lal_output_time_grid(t_modes_M: torch.Tensor, delta_t: float, M_sec: float):
    """Fixed output grid length from LAL's retLenTS calculation."""

    if len(t_modes_M) == 0 or delta_t <= 0.0 or M_sec <= 0.0:
        return t_modes_M[:0]
    delta_t_M = torch.as_tensor(delta_t / M_sec, device=t_modes_M.device, dtype=t_modes_M.dtype)
    duration_M = t_modes_M[-1] - t_modes_M[0]
    ret_len = max(0, int(torch.floor(duration_M / delta_t_M).item()))
    return torch.arange(ret_len, device=t_modes_M.device, dtype=t_modes_M.dtype) * delta_t_M


def _amplitude_peak_from_22_21(
    modes: Dict[Tuple[int, int], torch.Tensor],
    mode_array,
    t_vec_M: torch.Tensor,
):
    """Discrete LAL l=2 frame-invariant amplitude peak."""

    requested = {(int(l), int(m)) for l, m in mode_array}
    found22 = (2, 2) in requested
    found21 = (2, 1) in requested
    if not found22 or (2, 2) not in modes:
        raise ValueError("SEOBNRv4PHM amplitude peak requires requested mode (2, 2)")
    if found21 and (2, 1) not in modes:
        raise ValueError("SEOBNRv4PHM requested mode (2, 1) is missing from modes")

    h22 = modes[(2, 2)]
    if len(h22) != len(t_vec_M):
        raise ValueError("mode (2, 2) length does not match the time vector")
    amp_sq = h22.real * h22.real + h22.imag * h22.imag
    if found21:
        h21 = modes[(2, 1)]
        if len(h21) != len(t_vec_M):
            raise ValueError("mode (2, 1) length does not match the time vector")
        amp_sq = amp_sq + h21.real * h21.real + h21.imag * h21.imag

    index_peak = int(torch.argmax(amp_sq).item())
    return t_vec_M[index_peak], index_peak


def _dynamic_spin_projection_combos(
    Lvec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: _dyn.EOBParams,
    *,
    weighted_tplspin: bool,
):
    """Return L-frame spin combinations used by LAL's PHM waveform blocks."""

    Lmag = torch.linalg.norm(Lvec, dim=1)
    Lhat = Lvec / torch.clamp(Lmag.unsqueeze(-1), min=1.0e-15)
    chi1dotZ = torch.sum(S1 * Lhat, dim=1)
    chi2dotZ = torch.sum(S2 * Lhat, dim=1)
    chiS = 0.5 * (chi1dotZ + chi2dotZ)
    chiA = 0.5 * (chi1dotZ - chi2dotZ)

    total_mass = params.mass1 + params.mass2
    dM = (params.mass1 - params.mass2) / total_mass
    if weighted_tplspin:
        s1dotZ = (params.mass1 / total_mass) ** 2 * chi1dotZ
        s2dotZ = (params.mass2 / total_mass) ** 2 * chi2dotZ
        tplspin = (1.0 - 2.0 * params.eta) * (0.5 * (s1dotZ + s2dotZ)) + dM * (0.5 * (s1dotZ - s2dotZ))
    else:
        tplspin = (1.0 - 2.0 * params.eta) * chiS + dM * chiA

    return chi1dotZ, chi2dotZ, chiS, chiA, tplspin


def _factorized_residual_power(
    l: int,
    m: int,
    v: torch.Tensor,
    params: _dyn.EOBParams,
    H: torch.Tensor,
    chiS: torch.Tensor,
    chiA: torch.Tensor,
    tplspin: torch.Tensor,
    *,
    cal21=None,
    cal55=None,
):
    """Return LAL's rho_lm^l + f_lm residual and delta_lm."""

    rho_t, aux_t, delta_t = _dyn._factorized_rho_aux_delta(
        l,
        m,
        v,
        params,
        waveform=True,
        H=H,
        chiS=chiS,
        chiA=chiA,
        tplspin=tplspin,
        cal21=getattr(params, "cal21", 0.0) if cal21 is None else cal21,
        cal55=getattr(params, "cal55", 0.0) if cal55 is None else cal55,
    )
    if abs(params.eta - 0.25) < 1e-12 and (m % 2):
        # LAL equal-mass odd-m handling (LALSimIMRSpinEOBFactorizedWaveformPrec.c:504-521)
        return aux_t, delta_t
    return rho_t ** l + aux_t, delta_t


def _waveform_state_from_cartesian(
    y: torch.Tensor,
    params: _dyn.EOBParams,
) -> torch.Tensor:
    """Keep exact Cartesian geometry after projecting to the waveform state."""

    reduced = _dyn.cartesian_state_to_reduced_state(y, params)
    return torch.cat((reduced, y[0:6]))


def _retained_cartesian_vectors(
    y: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return retained x/P columns, or ``None`` for legacy reduced states."""

    if y.shape[-1] < 18:
        return None, None
    return y[..., 12:15], y[..., 15:18]


def _build_coprecessing_modes(
    phi: torch.Tensor,
    omega_orb: torch.Tensor,
    r: torch.Tensor,
    pr: torch.Tensor,
    Lvec: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    params: _dyn.EOBParams,
    mode_array,
    *,
    distance_scale: bool = True,
    weighted_tplspin: bool = True,
    H: torch.Tensor | None = None,
    r_vec: torch.Tensor | None = None,
    p_vec: torch.Tensor | None = None,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Return co-precessing h_lm(t) for requested modes (±m)."""

    if (r_vec is None) != (p_vec is None):
        raise ValueError("r_vec and p_vec must be supplied together")

    omega_dimless = omega_orb * params.M_sec
    v = torch.clamp(torch.abs(omega_dimless), min=1e-12) ** (1.0 / 3.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    _, _, chiS_dyn, chiA_dyn, tplspin_dyn = _dynamic_spin_projection_combos(
        Lvec,
        S1,
        S2,
        params,
        weighted_tplspin=weighted_tplspin,
    )
    if H is None:
        pot = _dyn._eob_potentials(
            r,
            pr,
            phi,
            Lvec,
            S1,
            S2,
            params,
            r_vec=r_vec,
            p_vec=p_vec,
            compute_grad_p=False,
            compute_base_grad=False,
            fd_pphi=False,
        )
        H = pot["H"]
    else:
        H = H.to(device=v.device, dtype=v.dtype)

    modes: Dict[Tuple[int, int], torch.Tensor] = {}
    if r_vec is None:
        phi_nk = torch.zeros_like(phi) if params.aligned_spins else phi
        r_vec, p_vec, _, _, _ = _dyn._reduced_to_cartesian(
            r, pr, phi_nk, Lvec, S1, S2, params
        )
    vphi = _dyn.non_keplerian_vphi(
        r,
        omega_dimless,
        phi,
        Lvec,
        S1,
        S2,
        params,
        r_vec=r_vec,
        p_vec=p_vec,
    )
    for l, m in mode_array:
        if m <= 0:
            continue

        eps = 0 if ((l + m) % 2 == 0) else 1
        # LAL complex Newtonian multipole hNewton.
        pref = _calc_prefix(l, m, params.mass1, params.mass2, params.eta)
        ylm = _scalar_sph_pi_over2(l - eps, -m, phi)
        pref_t = torch.tensor(pref, device=v.device, dtype=torch.complex128 if v.dtype == torch.float64 else torch.complex64)
        h_newt = pref_t * ylm.to(dtype=pref_t.dtype) * (vphi ** (l + eps)).to(dtype=pref_t.dtype)

        rholm_pwrl, delta_t = _factorized_residual_power(
            l,
            m,
            v,
            params,
            H,
            chiS_dyn,
            chiA_dyn,
            tplspin_dyn,
        )

        # LAL-style NQC correction uses pr_t and rOmega (no r-scaling of pphi)
        rOmega = torch.clamp(r * torch.abs(omega_dimless), min=1e-12)
        pr_over_rOmega = pr / rOmega
        pr_over_rOmega2 = pr_over_rOmega * pr_over_rOmega
        nqc_a_src = getattr(params, "nqc_a_map", {}).get((l, m), {})
        a1 = torch.tensor(nqc_a_src.get("a1", 0.0), device=v.device, dtype=v.dtype)
        a2 = torch.tensor(nqc_a_src.get("a2", 0.0), device=v.device, dtype=v.dtype)
        a3 = torch.tensor(nqc_a_src.get("a3", 0.0), device=v.device, dtype=v.dtype)
        a4 = torch.tensor(nqc_a_src.get("a4", 0.0), device=v.device, dtype=v.dtype)
        a5 = torch.tensor(nqc_a_src.get("a5", 0.0), device=v.device, dtype=v.dtype)
        sqrt_r = torch.sqrt(torch.clamp(r, min=1e-12))
        nqc_mag = (
            1.0
            + pr_over_rOmega2
            * (
                a1
                + a2 / torch.clamp(r, min=1e-12)
                + (a3) / (torch.clamp(r, min=1e-12) * sqrt_r)
                + a4 / torch.clamp(r, min=1e-12) ** 2.0
                + a5 / (torch.clamp(r, min=1e-12) ** 2 * sqrt_r)
            )
        )
        nqc = nqc_mag

        # Canonical pphi (no 1/r) for odd-parity source term matches LAL Slm = v*pp
        pphi_eff = torch.linalg.norm(Lvec, dim=1)
        S_eff = ((H * H - 1.0) / (2.0 * params.eta) + 1.0) if eps == 0 else v * pphi_eff
        tail = _tail_factor_complex(l, m, omega_dimless, H)
        hlm = (
            tail
            * torch.exp(1j * delta_t.to(dtype=tail.real.dtype))
            * S_eff.to(dtype=tail.dtype)
            * rholm_pwrl.to(dtype=tail.dtype)
            * h_newt.to(dtype=tail.dtype)
        )

        nqc_b_src = getattr(params, "nqc_b_map", {}).get((l, m), {})
        if nqc_b_src:
            b1 = torch.tensor(nqc_b_src.get("b1", 0.0), device=v.device, dtype=v.dtype)
            b2 = torch.tensor(nqc_b_src.get("b2", 0.0), device=v.device, dtype=v.dtype)
            b3 = torch.tensor(nqc_b_src.get("b3", 0.0), device=v.device, dtype=v.dtype)
            b4 = torch.tensor(nqc_b_src.get("b4", 0.0), device=v.device, dtype=v.dtype)
            nqc_phase = b1 * (pr / rOmega) + (pr * pr * pr / rOmega) * (b2 + b3 / sqrt_r + b4 / torch.clamp(r, min=1e-12))
            nqc = nqc * torch.exp(1j * nqc_phase.to(dtype=tail.real.dtype))

        if distance_scale:
            # distance scaling (geometric length GM/c^2 per solar mass)
            hlm = hlm * (params.M * _MRSUN_SI) / (params.distance * 1.0e6 * _PC_SI)

        # Apply NQC after factorized pieces, matching LAL layout
        h_lm = hlm * nqc.to(dtype=hlm.dtype)
        h_lm = torch.nan_to_num(h_lm, nan=0.0, posinf=0.0, neginf=0.0)
        modes[(l, m)] = h_lm
        modes[(l, -m)] = ((-1.0) ** l) * torch.conj(h_lm)
    return modes


def _wigner_d_element(l: int, mp: int, m: int, beta_t: torch.Tensor):
    """Small-d element d^l_{m,mp}(beta) using finite-sum definition (stable for low l)."""
    beta_t = beta_t.to(
        torch.float32 if beta_t.device.type == "mps" else torch.float64
    )
    cosb2 = torch.cos(beta_t * 0.5)
    sinb2 = torch.sin(beta_t * 0.5)
    k_min = max(0, m - mp)
    k_max = min(l + m, l - mp)
    pref = math.sqrt(
        math.factorial(l + m)
        * math.factorial(l - m)
        * math.factorial(l + mp)
        * math.factorial(l - mp)
    )
    out = torch.zeros_like(beta_t)
    for k in range(k_min, k_max + 1):
        denom = (
            math.factorial(l + m - k)
            * math.factorial(k)
            * math.factorial(mp - m + k)
            * math.factorial(l - mp - k)
        )
        coef = pref / denom
        sign = -1.0 if ((k - mp + m) % 2) else 1.0
        out = out + sign * coef * (cosb2 ** (2 * l + m - mp - 2 * k)) * (sinb2 ** (mp - m + 2 * k))
    if (m - mp) % 2:
        out = -out
    return out


def _rotate_modes(modes, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
    """Rotate co-precessing modes into the inertial frame via Wigner D."""

    if not modes:
        return modes

    rotated = {}
    l_values = sorted({lm[0] for lm in modes.keys()})
    for l in l_values:
        m_list = list(range(-l, l + 1))
        d_mat = torch.stack(
            [torch.stack([_wigner_d_element(l, mp, m, beta) for mp in m_list], dim=1) for m in m_list], dim=1
        )  # shape: (n,2l+1,2l+1)

        alpha_m = torch.tensor(m_list, device=alpha.device, dtype=alpha.dtype)
        gamma_m = torch.tensor(m_list, device=gamma.device, dtype=gamma.dtype)
        exp_alpha = torch.exp(-1j * alpha.unsqueeze(1) * alpha_m)
        exp_gamma = torch.exp(-1j * gamma.unsqueeze(1) * gamma_m)

        for i, m in enumerate(m_list):
            acc = None
            for j, mp in enumerate(m_list):
                h_mp = modes.get((l, mp), None)
                if h_mp is None:
                    continue
                D_vec = exp_alpha[:, i] * d_mat[:, i, j] * exp_gamma[:, j]
                term = D_vec.to(h_mp.dtype) * h_mp
                acc = term if acc is None else acc + term
            if acc is not None:
                rotated[(l, m)] = acc

    return rotated


def _rotate_interpolate_modes_jframe(
    modes,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    t_modes: torch.Tensor,
    t_out: torch.Tensor,
    mode_array,
):
    """Rotate P-frame modes to J-frame while interpolating, matching LAL order.

    This ports the core of
    ``SEOBRotateInterpolatehJlmReImFromSphHarmListhPlmAmpPhase``: P-frame
    mode amplitude/phase and Wigner coefficient amplitude/phase are spline
    interpolated separately to the output time grid before summing.
    """

    if not modes:
        return modes

    complex_dtype = next(iter(modes.values())).dtype
    device = alpha.device
    dtype = alpha.dtype
    positive_modes = sorted({(l, m) for l, m in mode_array if m > 0 and (l, m) in modes})
    if not positive_modes:
        return {}

    h_pos_out = {}
    for l, mp in positive_modes:
        h = modes[(l, mp)]
        amp = torch.abs(h)
        phase = _unwrap_angle(torch.angle(h))
        amp_u = _interp_series_cubic(t_out, t_modes, amp)
        phase_u = _interp_series_cubic(t_out, t_modes, phase)
        h_pos_out[(l, mp)] = torch.complex(amp_u * torch.cos(phase_u), amp_u * torch.sin(phase_u)).to(complex_dtype)

    out = {}
    for l in sorted({lm[0] for lm in positive_modes}):
        same_l = [(ll, mm) for ll, mm in positive_modes if ll == l]
        for m in range(-l, l + 1):
            acc = torch.zeros(len(t_out), device=device, dtype=complex_dtype)
            for _, mp in same_l:
                h_mp = h_pos_out[(l, mp)]

                d_amp = _wigner_d_element(l, mp, m, beta).to(device=device, dtype=dtype)
                d_phase = torch.as_tensor(float(m), device=device, dtype=dtype) * alpha + torch.as_tensor(float(mp), device=device, dtype=dtype) * gamma
                d_amp_u = _interp_series_cubic(t_out, t_modes, d_amp)
                d_phase_u = _interp_series_cubic(t_out, t_modes, d_phase)
                d_val = d_amp_u.to(complex_dtype) * torch.exp(-1j * d_phase_u).to(complex_dtype)
                acc = acc + d_val * h_mp

                d_amp_neg = _wigner_d_element(l, -mp, m, beta).to(device=device, dtype=dtype)
                d_phase_neg = torch.as_tensor(float(m), device=device, dtype=dtype) * alpha - torch.as_tensor(float(mp), device=device, dtype=dtype) * gamma
                d_amp_neg_u = _interp_series_cubic(t_out, t_modes, d_amp_neg)
                d_phase_neg_u = _interp_series_cubic(t_out, t_modes, d_phase_neg)
                d_neg_val = d_amp_neg_u.to(complex_dtype) * torch.exp(-1j * d_phase_neg_u).to(complex_dtype)
                acc = acc + d_neg_val * (((-1.0) ** l) * torch.conj(h_mp))
            out[(l, m)] = acc
    return out


def _rotate_modes_constant(modes, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
    """Constant-angle J→I rotation (LALSimIMRSpinPrecEOBv4P.c:4295-4388)."""
    if not modes:
        return modes

    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)
    gamma = torch.as_tensor(gamma)
    complex_dtype = next(iter(modes.values())).dtype
    device = alpha.device
    out = {}

    def _d_element(l, mp, m):
        return _wigner_d_element(l, mp, m, beta).to(device=device, dtype=alpha.dtype).squeeze()

    l_values = sorted({lm[0] for lm in modes.keys()})
    for l in l_values:
        m_list = list(range(-l, l + 1))
        n_t = len(next(v for (ll, _), v in modes.items() if ll == l))

        d_mat = torch.empty((2 * l + 1, 2 * l + 1), device=device, dtype=alpha.dtype)
        for i, m in enumerate(m_list):
            for j, mp in enumerate(m_list):
                d_mat[i, j] = _d_element(l, mp, m)
        alpha_m = torch.tensor(m_list, device=device, dtype=alpha.dtype)
        gamma_m = torch.tensor(m_list, device=device, dtype=alpha.dtype)
        exp_alpha = torch.exp(-1j * alpha * alpha_m)
        exp_gamma = torch.exp(-1j * gamma * gamma_m)
        D = (exp_alpha.unsqueeze(1) * d_mat * exp_gamma).to(dtype=complex_dtype)

        stack = torch.zeros((n_t, len(m_list)), device=device, dtype=complex_dtype)
        for j, mp in enumerate(m_list):
            hmp = modes.get((l, mp))
            if hmp is not None:
                stack[: len(hmp), j] = hmp.to(complex_dtype)

        rotated_l = torch.einsum("ij,tj->ti", D, stack)
        for i, m in enumerate(m_list):
            out[(l, m)] = rotated_l[:, i]

    return out


def _polarizations_from_modes(rot_modes, inclination: float, phi0: float, *, device, complex_dtype):
    """Project rotated modes onto plus/cross for given (theta,phi)."""
    real_dtype = (
        torch.float32 if complex_dtype == torch.complex64 else torch.float64
    )
    theta_t = torch.tensor(inclination, device=device, dtype=real_dtype)
    phi_t = torch.tensor(math.pi / 2.0 - phi0, device=device, dtype=real_dtype)
    hp = None
    for (l, m), h_lm in rot_modes.items():
        ylm = _sYlm_torch(-2, l, m, theta_t, phi_t).to(device=device)
        term = h_lm.to(complex_dtype) * ylm.to(complex_dtype)
        hp = term if hp is None else hp + term
    # Match the polarity returned by lalsimulation's TD SEOBNRv4PHM API.
    hp_real = -hp.real
    hc_real = hp.imag
    return torch.nan_to_num(hp_real, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(hc_real, nan=0.0, posinf=0.0, neginf=0.0)


def _rotate_polarizations(
    hp: torch.Tensor,
    hc: torch.Tensor,
    long_asc_nodes: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply LAL's constant polarization rotation for the ascending node."""

    angle = 2.0 * float(long_asc_nodes)
    if angle == 0.0:
        return hp, hc
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine * hp + sine * hc, cosine * hc - sine * hp


# ---------------------------------------------------------------------------
# QNM frequency tables (Cardoso; torch-native)
# COMPLETE port of XLALSimIMREOBGenerateQNMFreqV2FromFinalPrec fundamental
# mode interpolation (LALSimBlackHoleRingdownPrec.c:362-729, 1763-1770).
# ---------------------------------------------------------------------------

_QNM_SPINS = np.array([
        -0.9996000, -0.9995000, -0.9994000, -0.9992000, -0.9990000, -0.9989000, -0.9988000, -0.9987000,
        -0.9986000, -0.9985000, -0.9980000, -0.9975000, -0.9970000, -0.9960000, -0.9950000, -0.9940000,
        -0.9920000, -0.9900000, -0.9880000, -0.9860000, -0.9840000, -0.9820000, -0.9800000, -0.9750000,
        -0.9700000, -0.9600000, -0.9500000, -0.9400000, -0.9200000, -0.9000000, -0.8800000, -0.8600000,
        -0.8400000, -0.8200000, -0.8000000, -0.7800000, -0.7600000, -0.7400000, -0.7200000, -0.7000000,
        -0.6500000, -0.6000000, -0.5500000, -0.5000000, -0.4500000, -0.4000000, -0.3500000, -0.3000000,
        -0.2500000, -0.2000000, -0.1500000, -0.1000000, -0.0500000, 0.0000000, 0.0500000, 0.1000000,
        0.1500000, 0.2000000, 0.2500000, 0.3000000, 0.3500000, 0.4000000, 0.4500000, 0.5000000,
        0.5500000, 0.6000000, 0.6500000, 0.7000000, 0.7200000, 0.7400000, 0.7600000, 0.7800000,
        0.8000000, 0.8200000, 0.8400000, 0.8600000, 0.8800000, 0.9000000, 0.9200000, 0.9400000,
        0.9500000, 0.9600000, 0.9700000, 0.9750000, 0.9800000, 0.9820000, 0.9840000, 0.9860000,
        0.9880000, 0.9900000, 0.9920000, 0.9940000, 0.9950000, 0.9960000, 0.9970000, 0.9975000,
        0.9980000, 0.9985000, 0.9986000, 0.9987000, 0.9988000, 0.9989000, 0.9990000, 0.9992000,
        0.9994000, 0.9995000, 0.9996000,
    ], dtype=np.float64)

_QNM_RE = {
    (2, 2): np.array([
        0.2702280, 0.2765620, 0.2806360, 0.2852340, 0.2875480, 0.2882820, 0.2888450, 0.2892870,
        0.2896390, 0.2899240, 0.2907810, 0.2911890, 0.2914180, 0.2916580, 0.2917850, 0.2918700,
        0.2919980, 0.2921110, 0.2922210, 0.2923310, 0.2924410, 0.2925520, 0.2926640, 0.2929430,
        0.2932230, 0.2937870, 0.2943540, 0.2949250, 0.2960770, 0.2972440, 0.2984260, 0.2996240,
        0.3008370, 0.3020670, 0.3033130, 0.3045770, 0.3058570, 0.3071560, 0.3084730, 0.3098080,
        0.3132320, 0.3167840, 0.3204730, 0.3243070, 0.3282990, 0.3324580, 0.3367980, 0.3413330,
        0.3460790, 0.3510530, 0.3562750, 0.3617680, 0.3675570, 0.3736720, 0.3801460, 0.3870180,
        0.3943330, 0.4021450, 0.4105180, 0.4195270, 0.4292640, 0.4398420, 0.4514020, 0.4641230,
        0.4782350, 0.4940450, 0.5119690, 0.5326000, 0.5417940, 0.5516300, 0.5622010, 0.5736160,
        0.5860170, 0.5995800, 0.6145390, 0.6312060, 0.6500180, 0.6716140, 0.6969950, 0.7278750,
        0.7463200, 0.7676740, 0.7932080, 0.8082350, 0.8254290, 0.8331000, 0.8413430, 0.8502720,
        0.8600460, 0.8708930, 0.8831620, 0.8974460, 0.9056640, 0.9149020, 0.9255810, 0.9316890,
        0.9385240, 0.9463850, 0.9481230, 0.9499290, 0.9518130, 0.9537840, 0.9558540, 0.9603580,
        0.9655140, 0.9684380, 0.9716900,
    ], dtype=np.float64),
    (2, 1): np.array([
        0.3366090, 0.3393860, 0.3408520, 0.3422190, 0.3428180, 0.3430020, 0.3431430, 0.3432540,
        0.3433440, 0.3434170, 0.3436410, 0.3437480, 0.3438040, 0.3438530, 0.3438710, 0.3438790,
        0.3438860, 0.3438910, 0.3438960, 0.3439020, 0.3439090, 0.3439150, 0.3439220, 0.3439410,
        0.3439600, 0.3440020, 0.3440490, 0.3441010, 0.3442200, 0.3443590, 0.3445170, 0.3446960,
        0.3448960, 0.3451150, 0.3453560, 0.3456170, 0.3458990, 0.3462010, 0.3465250, 0.3468700,
        0.3478240, 0.3489110, 0.3501320, 0.3514910, 0.3529900, 0.3546330, 0.3564230, 0.3583660,
        0.3604690, 0.3627380, 0.3651830, 0.3678120, 0.3706370, 0.3736720, 0.3769310, 0.3804320,
        0.3841970, 0.3882480, 0.3926150, 0.3973300, 0.4024360, 0.4079790, 0.4140200, 0.4206320,
        0.4279090, 0.4359680, 0.4449680, 0.4551210, 0.4595690, 0.4642710, 0.4692590, 0.4745640,
        0.4802310, 0.4863080, 0.4928590, 0.4999650, 0.5077290, 0.5162910, 0.5258450, 0.5366730,
        0.5426930, 0.5492130, 0.5563290, 0.5601460, 0.5641550, 0.5658140, 0.5675050, 0.5692270,
        0.5709760, 0.5727490, 0.5745350, 0.5763220, 0.5772080, 0.5780840, 0.5789480, 0.5793740,
        0.5797950, 0.5802120, 0.5802950, 0.5803770, 0.5804600, 0.5805410, 0.5806230, 0.5807840,
        0.5809420, 0.5810180, 0.5810930,
    ], dtype=np.float64),
    (3, 3): np.array([
        0.4457680, 0.4527990, 0.4569480, 0.4609430, 0.4624620, 0.4628420, 0.4630950, 0.4632690,
        0.4633940, 0.4634880, 0.4637460, 0.4638860, 0.4639890, 0.4641440, 0.4642670, 0.4643740,
        0.4645720, 0.4647630, 0.4649520, 0.4651400, 0.4653290, 0.4655180, 0.4657070, 0.4661820,
        0.4666570, 0.4676120, 0.4685730, 0.4695400, 0.4714910, 0.4734650, 0.4754640, 0.4774870,
        0.4795350, 0.4816090, 0.4837090, 0.4858370, 0.4879910, 0.4901740, 0.4923860, 0.4946270,
        0.5003630, 0.5063000, 0.5124490, 0.5188260, 0.5254450, 0.5323230, 0.5394790, 0.5469340,
        0.5547100, 0.5628340, 0.5713350, 0.5802440, 0.5896000, 0.5994430, 0.6098230, 0.6207960,
        0.6324250, 0.6447870, 0.6579720, 0.6720860, 0.6872600, 0.7036500, 0.7214550, 0.7409210,
        0.7623690, 0.7862230, 0.8130570, 0.8436870, 0.8572540, 0.8717170, 0.8872010, 0.9038600,
        0.9218850, 0.9415210, 0.9630880, 0.9870160, 1.0139100, 1.0446400, 1.0805800, 1.1241000,
        1.1499800, 1.1798600, 1.2154700, 1.2363700, 1.2602300, 1.2708600, 1.2822700, 1.2946200,
        1.3081200, 1.3230800, 1.3399900, 1.3596500, 1.3709400, 1.3836300, 1.3982900, 1.4066600,
        1.4160300, 1.4267900, 1.4291700, 1.4316400, 1.4342200, 1.4369200, 1.4397500, 1.4459100,
        1.4529500, 1.4569500, 1.4613900,
    ], dtype=np.float64),
    (4, 4): np.array([
        0.6034850, 0.6138470, 0.6196360, 0.6239520, 0.6242190, 0.6238940, 0.6235040, 0.6231220,
        0.6227800, 0.6224870, 0.6216480, 0.6213750, 0.6213090, 0.6213650, 0.6214830, 0.6216130,
        0.6218770, 0.6221410, 0.6224040, 0.6226670, 0.6229300, 0.6231940, 0.6234580, 0.6241190,
        0.6247810, 0.6261130, 0.6274520, 0.6287990, 0.6315180, 0.6342690, 0.6370540, 0.6398720,
        0.6427260, 0.6456150, 0.6485410, 0.6515030, 0.6545040, 0.6575440, 0.6606230, 0.6637430,
        0.6717280, 0.6799890, 0.6885430, 0.6974110, 0.7066110, 0.7161680, 0.7261070, 0.7364550,
        0.7472430, 0.7585080, 0.7702860, 0.7826240, 0.7955690, 0.8091780, 0.8235170, 0.8386600,
        0.8546930, 0.8717180, 0.8898530, 0.9092420, 0.9300540, 0.9525000, 0.9768390, 1.0034000,
        1.0325900, 1.0649800, 1.1013100, 1.1426500, 1.1609200, 1.1803600, 1.2011400, 1.2234500,
        1.2475500, 1.2737400, 1.3024500, 1.3342200, 1.3698400, 1.4104200, 1.4577300, 1.5147800,
        1.5486200, 1.5875900, 1.6339000, 1.6610200, 1.6919400, 1.7057000, 1.7204500, 1.7364100,
        1.7538400, 1.7731400, 1.7949200, 1.8202200, 1.8347400, 1.8510400, 1.8698500, 1.8806000,
        1.8926000, 1.9064000, 1.9094500, 1.9126100, 1.9159200, 1.9193700, 1.9229900, 1.9308800,
        1.9398900, 1.9450000, 1.9506800,
    ], dtype=np.float64),
    (5, 5): np.array([
        0.7525320, 0.7691330, 0.7791790, 0.7870900, 0.7861490, 0.7843290, 0.7823380, 0.7804850,
        0.7789030, 0.7776150, 0.7742860, 0.7732820, 0.7729660, 0.7728900, 0.7729990, 0.7731490,
        0.7734760, 0.7738100, 0.7741450, 0.7744800, 0.7748160, 0.7751520, 0.7754880, 0.7763310,
        0.7771760, 0.7788740, 0.7805820, 0.7822990, 0.7857660, 0.7892740, 0.7928250, 0.7964200,
        0.8000590, 0.8037430, 0.8074740, 0.8112530, 0.8150800, 0.8189560, 0.8228830, 0.8268630,
        0.8370460, 0.8475820, 0.8584930, 0.8698020, 0.8815360, 0.8937240, 0.9063980, 0.9195940,
        0.9333500, 0.9477120, 0.9627280, 0.9784540, 0.9949530, 1.0123000, 1.0305600, 1.0498500,
        1.0702700, 1.0919400, 1.1150200, 1.1396800, 1.1661400, 1.1946700, 1.2255800, 1.2592800,
        1.2963100, 1.3373400, 1.3833200, 1.4355500, 1.4586100, 1.4831300, 1.5093200, 1.5374200,
        1.5677400, 1.6006700, 1.6367100, 1.6765500, 1.7211500, 1.7718800, 1.8309200, 1.9019700,
        1.9440300, 1.9923900, 2.0497700, 2.0833300, 2.1215400, 2.1385200, 2.1567300, 2.1764100,
        2.1979000, 2.2216800, 2.2484900, 2.2796100, 2.2974600, 2.3174900, 2.3405800, 2.3537700,
        2.3685000, 2.3854200, 2.3891500, 2.3930300, 2.3970800, 2.4013100, 2.4057500, 2.4154100,
        2.4264500, 2.4327100, 2.4396700,
    ], dtype=np.float64),
}

_QNM_IM = {
    (2, 2): np.array([
        0.0784607, 0.0801007, 0.0816472, 0.0840037, 0.0854954, 0.0860129, 0.0864215, 0.0867455,
        0.0870035, 0.0872100, 0.0877802, 0.0879878, 0.0880650, 0.0880965, 0.0880878, 0.0880757,
        0.0880612, 0.0880574, 0.0880589, 0.0880627, 0.0880675, 0.0880727, 0.0880780, 0.0880913,
        0.0881045, 0.0881304, 0.0881560, 0.0881813, 0.0882315, 0.0882807, 0.0883289, 0.0883763,
        0.0884226, 0.0884679, 0.0885122, 0.0885555, 0.0885976, 0.0886386, 0.0886785, 0.0887172,
        0.0888085, 0.0888917, 0.0889663, 0.0890315, 0.0890868, 0.0891313, 0.0891643, 0.0891846,
        0.0891911, 0.0891825, 0.0891574, 0.0891138, 0.0890496, 0.0889623, 0.0888489, 0.0887057,
        0.0885283, 0.0883112, 0.0880477, 0.0877293, 0.0873453, 0.0868820, 0.0863212, 0.0856388,
        0.0848021, 0.0837652, 0.0824618, 0.0807929, 0.0799908, 0.0790927, 0.0780817, 0.0769364,
        0.0756296, 0.0741258, 0.0723780, 0.0703215, 0.0678642, 0.0648692, 0.0611186, 0.0562313,
        0.0531490, 0.0494336, 0.0447904, 0.0419586, 0.0386302, 0.0371155, 0.0354676, 0.0336590,
        0.0316516, 0.0293904, 0.0267908, 0.0237095, 0.0219107, 0.0198661, 0.0174737, 0.0160919,
        0.0145340, 0.0127274, 0.0123259, 0.0119077, 0.0114708, 0.0110127, 0.0105306, 0.0094780,
        0.0082669, 0.0075770, 0.0068074,
    ], dtype=np.float64),
    (2, 1): np.array([
        0.0743116, 0.0772659, 0.0791824, 0.0812815, 0.0822652, 0.0825560, 0.0827684, 0.0829260,
        0.0830445, 0.0831347, 0.0833596, 0.0834304, 0.0834555, 0.0834714, 0.0834806, 0.0834917,
        0.0835195, 0.0835507, 0.0835831, 0.0836156, 0.0836481, 0.0836804, 0.0837126, 0.0837920,
        0.0838704, 0.0840241, 0.0841737, 0.0843193, 0.0845994, 0.0848650, 0.0851170, 0.0853561,
        0.0855831, 0.0857987, 0.0860035, 0.0861980, 0.0863828, 0.0865584, 0.0867253, 0.0868840,
        0.0872470, 0.0875662, 0.0878462, 0.0880906, 0.0883025, 0.0884840, 0.0886370, 0.0887628,
        0.0888621, 0.0889354, 0.0889828, 0.0890037, 0.0889973, 0.0889623, 0.0888968, 0.0887983,
        0.0886636, 0.0884885, 0.0882679, 0.0879952, 0.0876618, 0.0872571, 0.0867670, 0.0861730,
        0.0854501, 0.0845642, 0.0834665, 0.0820852, 0.0814304, 0.0807035, 0.0798926, 0.0789827,
        0.0779550, 0.0767847, 0.0754394, 0.0738744, 0.0720270, 0.0698043, 0.0670610, 0.0635492,
        0.0613721, 0.0587910, 0.0556436, 0.0537759, 0.0516427, 0.0506978, 0.0496908, 0.0486136,
        0.0474565, 0.0462084, 0.0448564, 0.0433879, 0.0426068, 0.0417939, 0.0409501, 0.0405173,
        0.0400777, 0.0396319, 0.0395420, 0.0394519, 0.0393616, 0.0392710, 0.0391802, 0.0389974,
        0.0388111, 0.0387140, 0.0386091,
    ], dtype=np.float64),
    (3, 3): np.array([
        0.0686120, 0.0735351, 0.0774984, 0.0829193, 0.0860369, 0.0870572, 0.0878377, 0.0884398,
        0.0889085, 0.0892767, 0.0902685, 0.0906386, 0.0907974, 0.0909101, 0.0909414, 0.0909520,
        0.0909603, 0.0909664, 0.0909729, 0.0909798, 0.0909870, 0.0909942, 0.0910015, 0.0910197,
        0.0910378, 0.0910740, 0.0911100, 0.0911457, 0.0912166, 0.0912867, 0.0913560, 0.0914243,
        0.0914917, 0.0915582, 0.0916236, 0.0916880, 0.0917514, 0.0918136, 0.0918746, 0.0919344,
        0.0920784, 0.0922137, 0.0923394, 0.0924547, 0.0925583, 0.0926492, 0.0927258, 0.0927867,
        0.0928302, 0.0928541, 0.0928562, 0.0928338, 0.0927840, 0.0927030, 0.0925869, 0.0924305,
        0.0922281, 0.0919726, 0.0916556, 0.0912666, 0.0907928, 0.0902179, 0.0895213, 0.0886763,
        0.0876470, 0.0863849, 0.0848213, 0.0828557, 0.0819248, 0.0808922, 0.0797413, 0.0784512,
        0.0769953, 0.0753390, 0.0734361, 0.0712234, 0.0686106, 0.0654629, 0.0615646, 0.0565370,
        0.0533881, 0.0496087, 0.0449046, 0.0420439, 0.0386881, 0.0371631, 0.0355053, 0.0336873,
        0.0316714, 0.0294027, 0.0267968, 0.0237110, 0.0219107, 0.0198652, 0.0174725, 0.0160908,
        0.0145330, 0.0127266, 0.0123252, 0.0119070, 0.0114701, 0.0110120, 0.0105299, 0.0094773,
        0.0082662, 0.0075764, 0.0068068,
    ], dtype=np.float64),
    (4, 4): np.array([
        0.0472887, 0.0570846, 0.0652857, 0.0769202, 0.0835313, 0.0855743, 0.0870596, 0.0881437,
        0.0889418, 0.0895359, 0.0909619, 0.0914196, 0.0916056, 0.0917410, 0.0917845, 0.0918026,
        0.0918186, 0.0918281, 0.0918366, 0.0918448, 0.0918531, 0.0918614, 0.0918696, 0.0918903,
        0.0919109, 0.0919521, 0.0919931, 0.0920340, 0.0921151, 0.0921956, 0.0922753, 0.0923543,
        0.0924324, 0.0925097, 0.0925861, 0.0926616, 0.0927361, 0.0928096, 0.0928821, 0.0929535,
        0.0931268, 0.0932920, 0.0934484, 0.0935949, 0.0937302, 0.0938532, 0.0939623, 0.0940559,
        0.0941321, 0.0941887, 0.0942233, 0.0942330, 0.0942145, 0.0941640, 0.0940768, 0.0939478,
        0.0937705, 0.0935374, 0.0932393, 0.0928650, 0.0924006, 0.0918287, 0.0911274, 0.0902679,
        0.0892124, 0.0879094, 0.0862870, 0.0842401, 0.0832692, 0.0821917, 0.0809905, 0.0796442,
        0.0781255, 0.0763992, 0.0744185, 0.0721192, 0.0694103, 0.0661560, 0.0621396, 0.0569820,
        0.0537633, 0.0499111, 0.0451315, 0.0422323, 0.0388376, 0.0372970, 0.0356236, 0.0337901,
        0.0317587, 0.0294747, 0.0268536, 0.0237529, 0.0219453, 0.0198926, 0.0174928, 0.0161075,
        0.0145463, 0.0127365, 0.0123344, 0.0119155, 0.0114779, 0.0110192, 0.0105364, 0.0094825,
        0.0082701, 0.0075796, 0.0068093,
    ], dtype=np.float64),
    (5, 5): np.array([
        0.0151971, 0.0298700, 0.0432352, 0.0648490, 0.0789971, 0.0834521, 0.0865117, 0.0885261,
        0.0898184, 0.0906380, 0.0919286, 0.0920896, 0.0921222, 0.0921485, 0.0921652, 0.0921765,
        0.0921910, 0.0922013, 0.0922106, 0.0922195, 0.0922283, 0.0922371, 0.0922459, 0.0922679,
        0.0922898, 0.0923335, 0.0923770, 0.0924205, 0.0925068, 0.0925926, 0.0926777, 0.0927620,
        0.0928457, 0.0929286, 0.0930106, 0.0930918, 0.0931722, 0.0932516, 0.0933300, 0.0934074,
        0.0935961, 0.0937773, 0.0939500, 0.0941133, 0.0942660, 0.0944066, 0.0945339, 0.0946460,
        0.0947411, 0.0948169, 0.0948709, 0.0949002, 0.0949014, 0.0948705, 0.0948030, 0.0946932,
        0.0945348, 0.0943198, 0.0940389, 0.0936804, 0.0932301, 0.0926700, 0.0919774, 0.0911228,
        0.0900671, 0.0887575, 0.0871199, 0.0850466, 0.0840612, 0.0829665, 0.0817451, 0.0803751,
        0.0788289, 0.0770706, 0.0750526, 0.0727103, 0.0699513, 0.0666388, 0.0625545, 0.0573174,
        0.0540540, 0.0501531, 0.0453206, 0.0423931, 0.0389690, 0.0374162, 0.0357305, 0.0338844,
        0.0318402, 0.0295431, 0.0269089, 0.0237947, 0.0219803, 0.0199207, 0.0175140, 0.0161252,
        0.0145605, 0.0127472, 0.0123443, 0.0119247, 0.0114865, 0.0110270, 0.0105435, 0.0094882,
        0.0082744, 0.0075832, 0.0068122,
    ], dtype=np.float64),
}

_QNM_SPLINES = {
    key: (
        CubicSpline(_QNM_SPINS, re_arr, bc_type="natural"),
        CubicSpline(_QNM_SPINS, im_arr, bc_type="natural"),
    )
    for key, re_arr in _QNM_RE.items()
    for im_key, im_arr in _QNM_IM.items()
    if im_key == key
}


def _qnm_re_im(final_spin, l: int, m: int):
    """Return (omega_R, omega_I) in units of final mass (Cardoso table)."""
    table = _QNM_SPLINES.get((l, abs(m)))
    if table is None:
        return None, None
    re_spline, im_spline = table
    spin = float(final_spin)
    spin = max(-0.9996, min(0.9996, spin))
    signm = -1.0 if m < 0 else 1.0
    s_eval = signm * spin
    omega_r = float(signm * re_spline(s_eval))
    omega_i = float(im_spline(s_eval))
    return omega_r, omega_i


def _euler_qnm_precession_rate(
    final_spin,
    final_mass_frac: float,
    cos_angle: float,
    *,
    device,
    dtype,
):
    """Return LAL's post-attachment Euler precession rate in units of 1/M."""

    omega220, _ = _qnm_re_im(final_spin, 2, 2)
    omega210, _ = _qnm_re_im(final_spin, 2, 1)
    if omega220 is None or omega210 is None or final_mass_frac <= 0.0:
        return torch.zeros((), device=device, dtype=dtype)
    rate = (omega220 - omega210) / final_mass_frac
    if cos_angle < 0.0:
        rate = -rate
    return torch.tensor(rate, device=device, dtype=dtype)


def _ringdown_omega_for_nyquist(params: _dyn.EOBParams, ell_max: int) -> float:
    """Ringdown angular frequency used by LAL's Nyquist validation."""

    final_mass, final_spin = _final_mass_spin_prec(
        params.mass1,
        params.mass2,
        (params.spin1x, params.spin1y, params.spin1z),
        (params.spin2x, params.spin2y, params.spin2z),
    )
    omega_r, _ = _qnm_re_im(final_spin, ell_max, ell_max)
    if omega_r is None or omega_r <= 0.0:
        raise ValueError(f"no QNM frequency available for Nyquist check ell={ell_max}")
    return omega_r / (final_mass * params.M_sec)


def _check_nyquist_frequency(
    params: _dyn.EOBParams,
    delta_t: float,
    ell_max_for_check: int,
    *,
    waveform_ell_max: int | None = None,
):
    """Validate that the highest checked ringdown mode is below Nyquist."""

    ell_max_for_check = int(ell_max_for_check)
    if ell_max_for_check < 2:
        raise ValueError("ellMaxForNyquistCheck must be >= 2")
    if waveform_ell_max is not None and ell_max_for_check < int(waveform_ell_max):
        _dbg(
            f"Nyquist check using ell={ell_max_for_check} "
            f"below waveform ell_max={int(waveform_ell_max)}"
        )
    omega_rd = _ringdown_omega_for_nyquist(params, ell_max_for_check)
    if delta_t > math.pi / omega_rd:
        raise ValueError("Ringdown frequency is above the Nyquist frequency")
    return omega_rd


# ---------------------------------------------------------------------------
# Ringdown coefficient fits (torch-native)
# ---------------------------------------------------------------------------

# Coefficients from LALSimIMREOBHybridRingdownPrec.c:49-207 (A1/A2/P1/P2)
_RD_A1 = {
    (2, 2): dict(c00=0.0830664, c01=-0.0196758, c02=-0.0136459, c10=0.0612892, c11=0.00146142, c20=-0.0893454),
    (2, 1): dict(c00=0.07780330893915006, c01=-0.05070638166864379, c10=0.24091006920322164, c11=0.38582622780596576, c20=-0.7456327888190485, c21=-0.9695534075470388),
    (3, 3): dict(c00=0.07638733045623343, c01=-0.030993441267953236, c10=0.2543447497371546, c11=0.2516879591102584, c20=-1.0892686061231245, c21=-0.7980907313033606),
    (4, 4): dict(c00=-0.06392710223439678, c01=-0.03646167590514318, c10=0.345195277237925, c11=1.2777441574118558, c20=-1.764352185878576, c21=-14.825262897834696, c31=40.67135475479875),
    (5, 5): dict(c00=-0.06704614393611373, c01=0.021905949257025503, c10=-0.24754936787743445, c11=-0.0943771022698497, c20=0.7588042862093705, c21=0.4357768883690394),
}

_RD_A2 = {
    (2, 2): dict(c00=-0.623953, c01=-0.371365, c10=1.39777, c11=2.40203, c20=-1.82173, c21=-5.25339),
    (2, 1): dict(c00=-1.2451852641667298, c01=-1.195786238319961, c10=6.134202504312409, c11=15.66696619631313, c20=-14.67251127485556, c21=-44.41982201432511),
    (3, 3): dict(c00=-0.8325292359346013, c01=-0.598880303198448, c10=2.767989795032018, c11=5.904371617277156, c20=-7.028151926115957, c21=-18.232606706124482),
    (4, 4): dict(c00=0.7813275473485185, c01=0.8094706044462984, c10=-5.18689829943586, c11=-5.38343327318501, c20=14.026415859369477, c21=0.1051625997942299, c31=46.978434956814006),
    (5, 5): dict(c00=1.6763424265367357, c01=0.4925695499534606, c10=-5.604559311983177, c11=-6.209095657439377, c20=16.751319143123386, c21=16.778452555342554),
}

_RD_P1 = {
    (2, 2): dict(c00=0.147584, c01=0.00779176, c02=-0.0244358, c10=0.263456, c11=-0.120853, c20=-0.808987),
    (2, 1): dict(c00=0.15601401627613815, c01=0.10219957917717622, c10=0.023346852452720928, c11=-0.9435308286367039, c20=0.15326558178697175, c21=1.7979082057513565),
    (3, 3): dict(c00=0.11085299117493969, c01=0.018959099261813613, c10=0.9999800463662053, c11=-0.729149797691242, c20=-3.3983315694441125, c21=2.5192011762934037),
    (4, 4): dict(c00=0.11498976343440313, c01=0.008389519706605305, c10=1.6126522800609633, c11=-0.8069979888526699, c20=-6.255895564079467, c21=7.595651881827078, c31=-19.32367406125053),
    (5, 5): dict(c00=0.16465380962882128, c01=-0.026574817803812007, c10=-0.19184523794508765, c11=-0.05519618962738479, c20=0.33328424135336965, c21=0.3194274548351241),
}

_RD_P2 = {
    (2, 2): dict(c00=2.46654, c01=3.13067, c02=0.581626, c10=-6.99396, c11=-9.61861, c20=17.5646),
    (2, 1): dict(
        c00=2.7886287922318105,
        c01=4.29290053494256,
        c02=2.5582321247274726,
        c12=-10.232928498772893,
        c10=-0.8145406685320334,
        c11=-15.93796979597706,
        c20=5.549338798935832,
        c21=12.649775582333442,
    ),
    (3, 3): dict(c00=2.7825237371542735, c01=2.8796835808075003, c10=-7.844741660437831, c11=-34.7670039322078, c20=27.181024362399302, c21=127.13948436435182),
    (4, 4): dict(
        c00=3.111817347262856,
        c01=5.399341180960216,
        c02=2.3832321567874686,
        c12=-9.532928476043567,
        c10=15.885333959709488,
        c11=-87.92421137153823,
        c20=-79.64931908155609,
        c21=657.7156442271963,
        c31=-1555.2968529739226,
    ),
    (5, 5): dict(c00=11.102447263357977, c01=6.015112119742853, c10=-58.605776859097084, c11=-81.68032025902797, c20=176.60643662729498, c21=266.47345742836745),
}


def _rd_poly(coeffs, l: int, m: int, eta: float, chi: float, *, device, dtype):
    """Evaluate polynomial fit for given mode."""
    c = coeffs.get((l, abs(m)))
    if c is None:
        return torch.tensor(0.0, device=device, dtype=dtype)
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi2 = chi * chi
    val = (
        c.get("c00", 0.0)
        + c.get("c01", 0.0) * chi
        + c.get("c02", 0.0) * chi2
        + c.get("c10", 0.0) * eta
        + c.get("c11", 0.0) * eta * chi
        + c.get("c12", 0.0) * eta * chi2
        + c.get("c20", 0.0) * eta2
        + c.get("c21", 0.0) * eta2 * chi
        + c.get("c31", 0.0) * eta3 * chi
    )
    return torch.tensor(val, device=device, dtype=dtype)


def _rd_amp_coeff1(l: int, m: int, eta: float, chi: float, *, device, dtype):
    return _rd_poly(_RD_A1, l, m, eta, chi, device=device, dtype=dtype)


def _rd_amp_coeff2(l: int, m: int, eta: float, chi: float, *, device, dtype):
    return _rd_poly(_RD_A2, l, m, eta, chi, device=device, dtype=dtype)


def _rd_phase_coeff1(l: int, m: int, eta: float, chi: float, *, device, dtype):
    return _rd_poly(_RD_P1, l, m, eta, chi, device=device, dtype=dtype)


def _rd_phase_coeff2(l: int, m: int, eta: float, chi: float, *, device, dtype):
    return _rd_poly(_RD_P2, l, m, eta, chi, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Final mass / spin fit for SEOBNRv4P/PHM (torch-native)
# COMPLETE port of XLALSimIMREOBFinalMassSpinPrec
# (LALSimBlackHoleRingdownPrec.c:170-310).
# ---------------------------------------------------------------------------

# k_ij coefficients from LALSimBlackHoleRingdown.h (Table I of Hofmann+2016)
_K00 = -5.977230835551017
_K01 = 3.39221
_K02 = 4.48865
_K03 = -5.77101
_K04 = -13.0459
_K10 = 35.1278
_K11 = -72.9336
_K12 = -86.0036
_K13 = 93.7371
_K14 = 200.975
_K20 = -146.822
_K21 = 387.184
_K22 = 447.009
_K23 = -467.383
_K24 = -884.339
_K30 = 223.911
_K31 = -648.502
_K32 = -697.177
_K33 = 753.738
_K34 = 1166.89


def _kerr_isco_radius(a: float) -> float:
    """XLALSimRadiusKerrISCO (LALSimBlackHoleRingdown.c:688-707)."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - (-1.0 if a < 0.0 else 1.0) * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def _kerr_isco_energy(r_isco: float) -> float:
    """XLALSimEnergyKerrISCO (LALSimBlackHoleRingdown.c:709-717)."""
    return math.sqrt(max(1.0 - 2.0 / (3.0 * r_isco), 0.0))


def _kerr_isco_angmom(r_isco: float) -> float:
    """XLALSimAngMomKerrISCO (LALSimBlackHoleRingdown.c:719-727)."""
    return (2.0 / (3.0 * math.sqrt(3.0))) * (1.0 + 2.0 * math.sqrt(max(3.0 * r_isco - 2.0, 0.0)))


def _final_mass_spin_prec(m1: float, m2: float, spin1: Tuple[float, float, float], spin2: Tuple[float, float, float]):
    """Return (Mf/Mtot, af) using the SEOBNRv4P/PHM fit."""

    # enforce primary = m1
    a1z = spin1[2]
    a2z = spin2[2]
    if m1 < m2:
        m1, m2 = m2, m1
        a1z, a2z = a2z, a1z

    q = m2 / m1  # q <= 1
    one_plus_q = 1.0 + q
    eta = m1 * m2 / (m1 + m2) ** 2
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2

    # aligned-spin proxy (atl) for final mass
    atl = (a1z + a2z * q * q) / (one_plus_q * one_plus_q)
    r_isco_aligned = _kerr_isco_radius(atl)
    e_isco_aligned = _kerr_isco_energy(r_isco_aligned)
    final_mass = 1.0 - (
        (1.0 - e_isco_aligned) * eta
        + 16.0
        * eta2
        * (0.00258 - 0.0773 / (1.0 / ((1.0 + q * q) / (one_plus_q * one_plus_q)) * atl - 1.6939) - 0.25 * (1.0 - e_isco_aligned))
    )

    # precessing-spin final spin fit (Hofmann+ 2016)
    chi1 = math.sqrt(spin1[0] ** 2 + spin1[1] ** 2 + spin1[2] ** 2)
    chi2 = math.sqrt(spin2[0] ** 2 + spin2[1] ** 2 + spin2[2] ** 2)
    beta = 0.0 if chi1 < 1e-6 else math.acos(max(min(spin1[2] / chi1, 1.0), -1.0))
    gamma = 0.0 if chi2 < 1e-6 else math.acos(max(min(spin2[2] / chi2, 1.0), -1.0))

    alpha = 0.0
    dot12 = spin1[0] * spin2[0] + spin1[1] * spin2[1] + spin1[2] * spin2[2]
    if chi1 > 1e-4 and chi2 > 1e-4:
        alpha = math.acos(max(min(dot12 / (chi1 * chi2), 1.0), -1.0))

    # empirical wobble corrections
    beta += 0.024 * math.sin(beta)
    gamma += 0.024 * math.sin(gamma)

    # swap angles/spins if original m2 > m1 (alpha unaffected)
    if m2 > m1:
        beta, gamma = gamma, beta
        chi1, chi2 = chi2, chi1

    csi = 0.474046
    a_tot_prec = (chi1 * math.cos(beta) + chi2 * math.cos(gamma) * q * q) / (one_plus_q * one_plus_q)
    aeff = a_tot_prec + csi * eta * (chi1 * math.cos(beta) + chi2 * math.cos(gamma))

    r_isco = _kerr_isco_radius(aeff)
    e_isco = _kerr_isco_energy(r_isco)
    l_isco = _kerr_isco_angmom(r_isco)

    aeff2 = aeff * aeff
    aeff3 = aeff2 * aeff
    aeff4 = aeff3 * aeff
    fitpart = (
        _K00 * eta
        + _K01 * aeff * eta
        + _K02 * aeff2 * eta
        + _K03 * aeff3 * eta
        + _K04 * aeff4 * eta
        + _K10 * eta2
        + _K11 * aeff * eta2
        + _K12 * aeff2 * eta2
        + _K13 * aeff3 * eta2
        + _K14 * aeff4 * eta2
        + _K20 * eta3
        + _K21 * aeff * eta3
        + _K22 * aeff2 * eta3
        + _K23 * aeff3 * eta3
        + _K24 * aeff4 * eta3
        + _K30 * eta4
        + _K31 * aeff * eta4
        + _K32 * aeff2 * eta4
        + _K33 * aeff3 * eta4
        + _K34 * aeff4 * eta4
    )

    ell_norm = abs(l_isco - 2.0 * a_tot_prec * (e_isco - 1.0) + fitpart)
    prefactor = 1.0 / (one_plus_q * one_plus_q)
    inside = (
        chi1 * chi1
        + chi2 * chi2 * q * q * q * q
        + 2.0 * chi1 * chi2 * q * q * math.cos(alpha)
        + 2.0 * (chi1 * math.cos(beta) + chi2 * q * q * math.cos(gamma)) * ell_norm * q
        + ell_norm * ell_norm * q * q
    )
    inside = max(inside, 0.0)
    final_spin = prefactor * math.sqrt(inside)
    final_spin = min(final_spin, 1.0 - 1e-6)

    return final_mass, final_spin


def _signed_clamped_final_spin(final_spin: float, cos_angle: float) -> float:
    """Apply LAL's final-spin sign and QNM interpolation clamp."""
    signed = -float(final_spin) if cos_angle < 0.0 else float(final_spin)
    return max(-0.9996, min(0.9996, signed))


def _l_frame_spin_vectors_from_state(y_state: torch.Tensor, params: _dyn.EOBParams):
    """Spin vectors in the local (n, lambda, L) frame used by SEOBLFrameVectors."""
    r_vec, p_vec = _retained_cartesian_vectors(y_state)
    if r_vec is None:
        _, _, n_hat, lambda_hat, Lhat = _dyn._reduced_to_cartesian(
            y_state[0],
            y_state[1],
            y_state[2],
            y_state[3:6],
            y_state[6:9],
            y_state[9:12],
            params,
        )
    else:
        n_hat = r_vec / torch.clamp(torch.linalg.norm(r_vec), min=1.0e-15)
        Lvec = torch.cross(r_vec, p_vec, dim=-1)
        Lhat = Lvec / torch.clamp(torch.linalg.norm(Lvec), min=1.0e-15)
        lambda_hat = torch.cross(Lhat, n_hat, dim=-1)
    S1 = y_state[6:9]
    S2 = y_state[9:12]
    S1_l = torch.stack([torch.dot(S1, n_hat), torch.dot(S1, lambda_hat), torch.dot(S1, Lhat)])
    S2_l = torch.stack([torch.dot(S2, n_hat), torch.dot(S2, lambda_hat), torch.dot(S2, Lhat)])
    return S1_l, S2_l


def _final_mass_spin_from_adas_10M(t_adas_M: torch.Tensor | None, y_adas: torch.Tensor | None, params: _dyn.EOBParams):
    """Final mass/spin fit using LAL's AdaS sample closest to r=10M."""
    if t_adas_M is None or y_adas is None or int(t_adas_M.numel()) < 2:
        return _final_mass_spin_prec(
            params.mass1,
            params.mass2,
            (params.spin1x, params.spin1y, params.spin1z),
            (params.spin2x, params.spin2y, params.spin2z),
        ), None, None, None

    idx_10M = _nearest_index_increasing(-y_adas[:, 0], -10.0)
    time_10M = float(t_adas_M[idx_10M].item())
    t_query = torch.tensor(time_10M, device=t_adas_M.device, dtype=t_adas_M.dtype)
    y_10M = _interp_series_cubic_lal_local(t_query, t_adas_M, y_adas)
    S1_l, S2_l = _l_frame_spin_vectors_from_state(y_10M, params)
    fit = _final_mass_spin_prec(
        params.mass1,
        params.mass2,
        tuple(float(x) for x in S1_l),
        tuple(float(x) for x in S2_l),
    )
    return fit, time_10M, S1_l, S2_l


def _attach_ringdown_modes(
    modes_inertial: Dict[Tuple[int, int], torch.Tensor],
    params: _dyn.EOBParams,
    dt: float,
    *,
    device,
    dtype,
    finspin=None,
    finmass_frac=None,
    t_attach_M: float | None = None,
    chi1L_attach: float | None = None,
    chi2L_attach: float | None = None,
    t_start_M: float = 0.0,
):
    """Attach ringdown using LAL's XLALSimIMREOBAttachFitRingdown (LALSimIMREOBHybridRingdownPrec.c:1678-2049)."""

    if not modes_inertial:
        return modes_inertial

    eta = params.eta
    if finspin is None or finmass_frac is None:
        fm_fit, af_fit = _final_mass_spin_prec(
            params.mass1,
            params.mass2,
            (params.spin1x, params.spin1y, params.spin1z),
            (params.spin2x, params.spin2y, params.spin2z),
        )
        if finspin is None:
            finspin = af_fit
        if finmass_frac is None:
            finmass_frac = fm_fit
    finmass_frac = finmass_frac if finmass_frac is not None else 0.95
    finspin_t = finspin if isinstance(finspin, torch.Tensor) else torch.tensor(finspin, device=device, dtype=dtype)

    # LAL attaches at tPeakOmega - DeltaT22, falling back to the 22 amplitude
    # peak only for diagnostic paths that do not provide merger timing.
    h22 = modes_inertial.get((2, 2), None)
    if h22 is None:
        return modes_inertial
    amp22 = torch.abs(h22)
    t_vec_22_M = torch.as_tensor(t_start_M, device=device, dtype=dtype) + torch.arange(len(h22), device=device, dtype=dtype) * (dt / params.M_sec)
    if t_attach_M is None:
        idx_attach_22 = int(torch.argmax(amp22).item())
    else:
        t_attach_M = min(max(float(t_attach_M), float(t_vec_22_M[0])), float(t_vec_22_M[-1]))
        idx_attach_22 = _nearest_index_increasing(t_vec_22_M, t_attach_M)
    t_attach_M = float(t_vec_22_M[idx_attach_22].item())

    # Precompute the 22 damping rate for the ringdown length. The embedded
    # Cardoso tables contain every mode admitted by the public support check,
    # so a missing value indicates an internal inconsistency rather than a
    # second model path to evaluate.
    re22, im22 = _qnm_re_im(finspin_t, 2, 2)
    if re22 is None or im22 is None or im22 <= 0.0:
        raise RuntimeError("missing valid (2, 2) QNM data")
    damp_rate_22 = im22 / (finmass_frac * params.M_sec)  # 1 / tau_sec
    n_rd = max(1, int(_EOB_RD_EFOLDS / (damp_rate_22 * dt + 1e-30)))
    n_rd_patch = max(
        n_rd,
        int(math.ceil(_EOB_RD_PATCH_EFOLDS / (damp_rate_22 * dt + 1e-30))),
    )
    dtM = dt / params.M_sec

    chi1 = params.spin1z if chi1L_attach is None else float(chi1L_attach)
    chi2 = params.spin2z if chi2L_attach is None else float(chi2L_attach)
    dM = (params.mass1 - params.mass2) / (params.mass1 + params.mass2)
    chi_eff = 0.5 * (chi1 + chi2) + 0.5 * (chi1 - chi2) * dM / max(1e-8, 1.0 - 2.0 * eta)

    out = {}
    for (l, m), h in list(modes_inertial.items()):
        if m <= 0:
            continue

        # mode-specific attachment time (55 gets tAttach-10M)
        attach_time_M = t_attach_M
        if l == 5 and abs(m) == 5:
            attach_time_M = max(0.0, t_attach_M - 10.0)

        t_vec_M = torch.as_tensor(t_start_M, device=device, dtype=dtype) + torch.arange(len(h), device=device, dtype=dtype) * dtM
        amp = torch.abs(h)
        phase = _unwrap_angle(torch.angle(h))
        idx_attach = _nearest_index_increasing(t_vec_M, attach_time_M)
        attach_time_M = float(t_vec_M[idx_attach].item())

        A_attach = amp[idx_attach]
        if l == 2 and abs(m) == 2:
            Adot_attach = torch.zeros((), device=device, dtype=dtype)
        else:
            Adot_attach = torch.tensor(
                local_derivatives(amp, t_vec_M, attach_time_M, order=1),
                device=device,
                dtype=dtype,
            )
        phi_attach = phase[idx_attach]
        if idx_attach > 0:
            omega_attach = (phase[idx_attach] - phase[idx_attach - 1]) / dtM
        else:
            omega_attach = (phase[1] - phase[0]) / dtM

        # QNM freq (fundamental only) in geometric units of total mass
        omega_r, omega_i = _qnm_re_im(finspin_t, l, m)
        if omega_r is None or omega_i is None or omega_i <= 0.0 or omega_r == 0.0:
            out[(l, m)] = h
            out[(l, -m)] = ((-1.0) ** l) * torch.conj(h)
            continue
        sigma_real = torch.tensor(-omega_i / finmass_frac, device=device, dtype=dtype)
        sigma_imag = torch.tensor(-omega_r / finmass_frac, device=device, dtype=dtype)

        ampcf1 = _rd_amp_coeff1(l, m, eta, chi_eff, device=device, dtype=dtype)
        ampcf2 = _rd_amp_coeff2(l, m, eta, chi_eff, device=device, dtype=dtype)
        if l == 2 and abs(m) == 2 and sigma_real > 2.0 * ampcf1 * torch.tanh(ampcf2):
            ampcf1 = sigma_real / (2.0 * torch.tanh(ampcf2))
        phasecf1 = _rd_phase_coeff1(l, m, eta, chi_eff, device=device, dtype=dtype)
        phasecf2 = _rd_phase_coeff2(l, m, eta, chi_eff, device=device, dtype=dtype)

        # Rescalings to guarantee continuity (Eqs. 22-37 of T1600383)
        Arescaled = A_attach * torch.exp(-sigma_real * 0.0) / max(eta, 1e-10)
        dArescaled = (Adot_attach - sigma_real * A_attach) * torch.exp(-sigma_real * 0.0) / max(eta, 1e-10)
        ampcc1 = dArescaled * (torch.cosh(ampcf2) ** 2) / ampcf1
        ampcc2 = (Arescaled * ampcf1 - dArescaled * torch.cosh(ampcf2) * torch.sinh(ampcf2)) / ampcf1

        rd_time = torch.arange(n_rd, device=device, dtype=dtype) * dtM
        amp_rd = eta * torch.exp(sigma_real * rd_time) * (ampcc1 * torch.tanh(ampcf1 * rd_time + ampcf2) + ampcc2)

        omega_rescaled = omega_attach - sigma_imag
        phasecc1 = omega_rescaled * (phasecf2 + 1.0) / (phasecf2 * phasecf1)
        logarg_den = 1.0 + phasecf2
        ph_rd = phi_attach - phasecc1 * torch.log((1.0 + phasecf2 * torch.exp(-phasecf1 * rd_time)) / logarg_den) + sigma_imag * rd_time

        h_rd = amp_rd * torch.exp(1j * ph_rd)
        h_new = torch.zeros(len(h) + n_rd_patch, device=device, dtype=h.dtype)
        h_new[: len(h)] = h
        h_new[idx_attach : idx_attach + n_rd] = h_rd

        out[(l, m)] = h_new
        out[(l, -m)] = ((-1.0) ** l) * torch.conj(h_new)

    return out


def _pad_modes_equal_length(modes: Dict[Tuple[int, int], torch.Tensor]):
    """Pad all modes to the same length (extend with last sample)."""
    if not modes:
        return modes
    max_len = max(len(v) for v in modes.values())
    padded = {}
    for k, v in modes.items():
        if len(v) == max_len:
            padded[k] = v
        else:
            pad_len = max_len - len(v)
            pad_tail = torch.cat([v, v[-1:].repeat(pad_len)])
            padded[k] = pad_tail
    return padded


def _time_to_fd(hp: torch.Tensor, hc: torch.Tensor, dt: float, complex_dtype, delta_f_target: float = None):
    """FFT real h+/hx into FrequencySeries on target delta_f grid.

    Match ``TimeSeries.to_frequencyseries``: zero pad to the requested
    frequency spacing, transform, and scale by ``delta_t``.
    """

    n_min = len(hp)
    if delta_f_target is None:
        delta_f_target = 1.0 / (n_min * dt)
    n_fft = int(1.0 / (delta_f_target * dt) + 0.5)
    if n_fft < n_min:
        raise ValueError(
            f"delta_f={delta_f_target} would undersample a time series with "
            f"duration={n_min * dt}"
        )
    if n_fft > n_min:
        pad = n_fft - n_min
        hp = torch.nn.functional.pad(hp, (0, pad))
        hc = torch.nn.functional.pad(hc, (0, pad))

    hp_fd = torch.fft.rfft(hp)
    hc_fd = torch.fft.rfft(hc)
    hp_fd = hp_fd * dt
    hc_fd = hc_fd * dt
    hp_fd = torch.nan_to_num(hp_fd, nan=0.0, posinf=0.0, neginf=0.0)
    hc_fd = torch.nan_to_num(hc_fd, nan=0.0, posinf=0.0, neginf=0.0)
    delta_f_out = 1.0 / (n_fft * dt)
    return hp_fd.to(complex_dtype), hc_fd.to(complex_dtype), delta_f_out


def _fd_cpu(**p):
    """Compute FD waveform using the CPU scheme fallback (no torch involvement)."""
    params = dict(p)
    approx = params.get("approximant", "SEOBNRv4PHM")
    params["approximant"] = approx

    # Supply TD-only args with sensible defaults
    def _set_default(key, value):
        if params.get(key) is None:
            params[key] = value

    _set_default("coa_phase", 0.0)
    _set_default("long_asc_nodes", 0.0)
    _set_default("eccentricity", 0.0)
    _set_default("mean_per_ano", 0.0)
    _set_default("delta_t", 1.0 / 4096.0)

    from pycbc.waveform.waveform import _lalsim_td_waveform

    env_backup = os.environ.get("PYCBC_SEOBNRV4PHM_NATIVE")
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "0"

        hp_td, hc_td = _lalsim_td_waveform(**params)
        hp_fd = hp_td.to_frequencyseries(delta_f=params["delta_f"])
        hc_fd = hc_td.to_frequencyseries(delta_f=params["delta_f"])

        f_final = params.get("f_final", None)
        if f_final and f_final > 0:
            n = int(f_final / params["delta_f"]) + 1
            hp_fd = hp_fd[:n]
            hc_fd = hc_fd[:n]
    finally:
        if env_backup is None:
            os.environ.pop("PYCBC_SEOBNRV4PHM_NATIVE", None)
        else:
            os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = env_backup
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return hp_fd, hc_fd


def seobnrv4phm_fd_from_td(*, use_torch: bool = False, **p):
    """Common TD->FD path for SEOBNRv4PHM (CPU fallback or torch cast)."""
    approx = p.get("approximant", "SEOBNRv4PHM")
    if approx != "SEOBNRv4PHM":
        raise ValueError(
            "seobnrv4phm_fd_from_td only supports SEOBNRv4PHM, "
            f"got {approx}"
        )

    hp_fd, hc_fd = _fd_cpu(**p)

    if not use_torch:
        return hp_fd, hc_fd

    device, _, complex_dtype = _target_device_dtypes()

    def _cast_fs(fs):
        tensor = torch.as_tensor(fs.numpy(), device=device, dtype=complex_dtype)
        return FrequencySeries(TorchArrayData(tensor), delta_f=fs.delta_f, epoch=fs.epoch, copy=False)

    return _cast_fs(hp_fd), _cast_fs(hc_fd)


def _nqc_coeffs_are_sane(nqc: dict, *, limit: float = 1.0e5) -> bool:
    """Reject ill-conditioned peak solves before they can dominate the waveform."""
    keys = ("a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4")
    for key in keys:
        val = float(nqc.get(key, 0.0))
        if not math.isfinite(val) or abs(val) > limit:
            return False
    return True


def _populate_nqc_coeffs_lal_v4(
    params: _dyn.EOBParams,
    mode_array,
    modes_cp_tmp: Dict[Tuple[int, int], torch.Tensor],
    t_vec_M: torch.Tensor,
    r: torch.Tensor,
    pr: torch.Tensor,
    omega_dimless: torch.Tensor,
    *,
    t_peak_omega_M: float,
    chi1L_peak: float,
    chi2L_peak: float,
):
    """Populate NQC maps by looping over requested positive modes, as LAL does."""

    params.nqc_a_map = {}
    params.nqc_b_map = {}
    for ml, mm in mode_array:
        if mm <= 0:
            continue
        hmode = modes_cp_tmp.get((ml, mm))
        if hmode is None:
            _dbg(f"NQC skipped l={ml} m={mm} missing first-pass mode")
            continue
        try:
            nqc = _solve_nqc_coeffs_lal_series(
                ml,
                mm,
                hmode,
                t_vec_M,
                r,
                pr,
                omega_dimless,
                t_peak_omega_M=t_peak_omega_M,
                chi1L_peak=chi1L_peak,
                chi2L_peak=chi2L_peak,
                params=params,
            )
        except Exception as exc:
            _dbg(f"NQC skipped l={ml} m={mm} solve failed: {exc}")
            continue
        if not _nqc_coeffs_are_sane(nqc):
            _dbg(f"NQC skipped l={ml} m={mm} ill-conditioned coeffs")
            continue
        params.nqc_a_map[(ml, mm)] = {k: nqc[k] for k in ("a1", "a2", "a3", "a4", "a5")}
        params.nqc_b_map[(ml, mm)] = {k: nqc[k] for k in ("b1", "b2", "b3", "b4")}
        if ml == 2 and mm == 2:
            params.nqc_a = params.nqc_a_map[(ml, mm)]
            params.nqc_b = params.nqc_b_map[(ml, mm)]
        _dbg(f"NQC solved l={ml} m={mm} a1={params.nqc_a_map[(ml, mm)]['a1']:.3e} b1={params.nqc_b_map[(ml, mm)]['b1']:.3e}")


def _l_frame_spin_projections(
    t_query_M: float,
    t_vec_M: torch.Tensor,
    Lvec: torch.Tensor,
    S1vec: torch.Tensor,
    S2vec: torch.Tensor,
):
    """Spin components along L at a dimensionless time, mirroring SEOBLFrameVectors."""

    t_min = float(t_vec_M[0])
    t_max = float(t_vec_M[-1])
    t_eval = min(max(float(t_query_M), t_min), t_max)
    q = torch.tensor(t_eval, device=t_vec_M.device, dtype=t_vec_M.dtype)
    L_eval = _interp_series_cubic_lal_local(q, t_vec_M, Lvec)
    S1_eval = _interp_series_cubic_lal_local(q, t_vec_M, S1vec)
    S2_eval = _interp_series_cubic_lal_local(q, t_vec_M, S2vec)
    Lhat = L_eval / torch.clamp(torch.linalg.norm(L_eval), min=1.0e-15)
    return float(torch.dot(S1_eval, Lhat)), float(torch.dot(S2_eval, Lhat))


def _phm_merger_timing(
    t_uniform: torch.Tensor,
    omega_orb: torch.Tensor,
    Lvec: torch.Tensor,
    S1vec: torch.Tensor,
    S2vec: torch.Tensor,
    params: _dyn.EOBParams,
):
    """Return LAL PHM peak-Omega and 22 attachment timing in units of total mass."""

    t_vec_M = t_uniform / params.M_sec
    omega_dimless = torch.abs(omega_orb * params.M_sec)
    try:
        t_peak_omega_M = find_peak_time(omega_dimless, t_vec_M)
    except Exception:
        t_peak_omega_M = float(t_vec_M[int(torch.argmax(omega_dimless))])

    chi1L, chi2L = _l_frame_spin_projections(t_peak_omega_M, t_vec_M, Lvec, S1vec, S2vec)
    delta_t22 = peak_delta_t_v4(2, 2, params.mass1, params.mass2, chi1L, chi2L)
    t_attach_M = t_peak_omega_M - delta_t22
    t_attach_M = min(max(t_attach_M, float(t_vec_M[0])), float(t_vec_M[-1]))
    return t_vec_M, t_peak_omega_M, t_attach_M, chi1L, chi2L


def _higher_mode_residual_calibration(
    t_vec_M: torch.Tensor,
    phi: torch.Tensor,
    omega_orb: torch.Tensor,
    r: torch.Tensor,
    pr: torch.Tensor,
    Lvec: torch.Tensor,
    S1vec: torch.Tensor,
    S2vec: torch.Tensor,
    params: _dyn.EOBParams,
    *,
    t_peak_omega_M: float,
    chi1L_peak: float,
    chi2L_peak: float,
    r_vec: torch.Tensor | None = None,
    p_vec: torch.Tensor | None = None,
):
    """Port of XLALSimIMREOBCalcCalibCoefficientHigherModesPrec for 21/55."""

    modes_to_calibrate = tuple(lm for lm in ((2, 1), (5, 5)) if lm in params.mode_array)
    if not modes_to_calibrate or t_vec_M.numel() < 3:
        return {}

    old_cal21 = getattr(params, "cal21", 0.0)
    old_cal55 = getattr(params, "cal55", 0.0)
    old_nqc_a = getattr(params, "nqc_a_map", None)
    old_nqc_b = getattr(params, "nqc_b_map", None)
    params.cal21 = 0.0
    params.cal55 = 0.0
    params.nqc_a_map = {}
    params.nqc_b_map = {}
    try:
        modes_uncal = _build_coprecessing_modes(
            phi,
            omega_orb,
            r,
            pr,
            Lvec,
            S1vec,
            S2vec,
            params,
            modes_to_calibrate,
            distance_scale=False,
            weighted_tplspin=False,
            r_vec=r_vec,
            p_vec=p_vec,
        )
    finally:
        params.cal21 = old_cal21
        params.cal55 = old_cal55
        if old_nqc_a is not None:
            params.nqc_a_map = old_nqc_a
        if old_nqc_b is not None:
            params.nqc_b_map = old_nqc_b

    omega_dimless = torch.abs(omega_orb * params.M_sec)
    v = torch.clamp(omega_dimless, min=1.0e-12) ** (1.0 / 3.0)
    pot = _dyn._eob_potentials(
        r,
        pr,
        phi,
        Lvec,
        S1vec,
        S2vec,
        params,
        r_vec=r_vec,
        p_vec=p_vec,
    )
    H = pot["H"]
    _, _, chiS_dyn, chiA_dyn, tplspin_dyn = _dynamic_spin_projection_combos(
        Lvec,
        S1vec,
        S2vec,
        params,
        weighted_tplspin=False,
    )

    chiS_peak = 0.5 * (chi1L_peak + chi2L_peak)
    chiA_peak = 0.5 * (chi1L_peak - chi2L_peak)
    t_min = float(t_vec_M[0])
    t_max = float(t_vec_M[-1])
    out = {}

    for mode_l, mode_m in modes_to_calibrate:
        hmode = modes_uncal.get((mode_l, mode_m))
        if hmode is None:
            continue

        delta_t_lm = peak_delta_t_v4(mode_l, mode_m, params.mass1, params.mass2, chi1L_peak, chi2L_peak)
        t_wave_peak_M = min(max(t_peak_omega_M - delta_t_lm, t_min), t_max)
        q = torch.tensor([t_wave_peak_M], device=t_vec_M.device, dtype=t_vec_M.dtype)

        rholm_pwrl, _ = _factorized_residual_power(
            mode_l,
            mode_m,
            v,
            params,
            H,
            chiS_dyn,
            chiA_dyn,
            tplspin_dyn,
            cal21=0.0,
            cal55=0.0,
        )
        rholm_real = torch.real(rholm_pwrl)
        hLMdivrholm = torch.abs(hmode) / torch.clamp(torch.abs(rholm_real), min=torch.finfo(rholm_real.dtype).tiny)

        hLMdiv_attach = float(_interp_series_cubic(q, t_vec_M, hLMdivrholm)[0])
        rholm_before = float(_interp_series_cubic(q, t_vec_M, rholm_real)[0])
        omega_attach = float(_interp_series_cubic(q, t_vec_M, omega_dimless)[0])
        if not (math.isfinite(hLMdiv_attach) and math.isfinite(rholm_before) and math.isfinite(omega_attach)):
            continue
        if abs(hLMdiv_attach) <= 0.0 or omega_attach <= 0.0:
            continue

        nra = float(peak_amp_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak))
        if mode_l == 2 and mode_m == 1 and abs(nra / params.eta) < 3.0e-2:
            nra = math.copysign(params.eta * 3.0e-2, nra)
        if mode_l == 5 and mode_m == 5 and abs(nra / params.eta) < 1.0e-4:
            nra = math.copysign(params.eta * 1.0e-4, nra)

        rholm_nr = nra / hLMdiv_attach
        if mode_l == 2 and mode_m == 1:
            out["cal21"] = (rholm_nr - rholm_before) / (omega_attach ** (7.0 / 3.0))
        else:
            out["cal55"] = (rholm_nr - rholm_before) / (omega_attach ** (5.0 / 3.0))

    return out


def _spline_values_derivatives(series_list, t_vec: torch.Tensor, t0: float):
    """Batch natural-cubic values while keeping local derivative parity."""

    t_min = float(t_vec[0])
    t_max = float(t_vec[-1])
    t_eval = min(max(float(t0), t_min), t_max)
    q = torch.tensor([t_eval], device=t_vec.device, dtype=t_vec.dtype)
    stacked = torch.stack(tuple(series_list), dim=1)
    vals = _interp_series_cubic(q, t_vec, stacked)[0]
    return [
        (
            float(vals[i]),
            local_derivatives(series, t_vec, t_eval, order=1),
            local_derivatives(series, t_vec, t_eval, order=2),
        )
        for i, series in enumerate(series_list)
    ]


def _solve_nqc_coeffs_lal_series(
    mode_l: int,
    mode_m: int,
    hmode: torch.Tensor,
    t_vec_M: torch.Tensor,
    r: torch.Tensor,
    pr: torch.Tensor,
    omega_dimless: torch.Tensor,
    *,
    t_peak_omega_M: float,
    chi1L_peak: float,
    chi2L_peak: float,
    params: _dyn.EOBParams,
):
    """LAL-style v4 NQC linear solve from mode/dynamics time series.

    Mirrors ``XLALSimIMRSpinEOBCalculateNQCCoefficientsV4``: build the
    ``q3/q4/q5`` and ``p3/p4`` basis vectors, evaluate natural-cubic spline
    derivatives at ``tPeakOmega - DeltaT_lm``, and solve the 3x3 amplitude and
    2x2 phase systems.  The input mode must be dimensionless, before distance
    scaling, matching LAL's P-frame hlm sequence.
    """

    chiS_peak = 0.5 * (chi1L_peak + chi2L_peak)
    chiA_peak = 0.5 * (chi1L_peak - chi2L_peak)
    q_mass = params.mass1 / params.mass2
    if q_mass < 1.005 and (mode_m % 2) and abs(chiA_peak) < 0.15:
        return {k: 0.0 for k in ("a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4")}

    delta_t_lm = peak_delta_t_v4(mode_l, mode_m, params.mass1, params.mass2, chi1L_peak, chi2L_peak)
    t_nqc_M = min(max(t_peak_omega_M - delta_t_lm, float(t_vec_M[0])), float(t_vec_M[-1]))

    amp = torch.abs(hmode)
    phase = _unwrap_angle(torch.angle(hmode))
    rOmega = torch.clamp(r * torch.abs(omega_dimless), min=1.0e-14)
    pr2 = pr * pr
    q3 = pr2 / (rOmega * rOmega)
    q4 = q3 / torch.clamp(r, min=1.0e-12)
    q5 = q4 / torch.sqrt(torch.clamp(r, min=1.0e-12))
    p3 = pr / rOmega
    p4 = p3 * pr2

    q_eval0, q_eval1, q_eval2, amp_eval, p_eval0, p_eval1, phase_eval = _spline_values_derivatives(
        (q3 * amp, q4 * amp, q5 * amp, amp, p3, p4, phase),
        t_vec_M,
        t_nqc_M,
    )
    q_matrix = np.array(
        [
            [q_eval0[0], q_eval1[0], q_eval2[0]],
            [q_eval0[1], q_eval1[1], q_eval2[1]],
            [q_eval0[2], q_eval1[2], q_eval2[2]],
        ],
        dtype=float,
    )

    amp_e, adot_e, addot_e = amp_eval
    rhs_a = np.array(
        [
            abs(peak_amp_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak)) - amp_e,
            peak_adot_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak) - adot_e,
            peak_addot_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak) - addot_e,
        ],
        dtype=float,
    )
    try:
        a1, a2, a3 = [float(x) for x in np.linalg.solve(q_matrix, rhs_a)]
    except np.linalg.LinAlgError:
        a1, a2, a3 = [float(x) for x in np.linalg.lstsq(q_matrix, rhs_a, rcond=None)[0]]

    p_matrix = np.array(
        [
            [-p_eval0[1], -p_eval1[1]],
            [-p_eval0[2], -p_eval1[2]],
        ],
        dtype=float,
    )

    _, omega_raw, omega_dot_raw = phase_eval
    omega_e = abs(omega_raw)
    omega_dot_e = abs(omega_dot_raw) if omega_raw * omega_dot_raw > 0.0 else -abs(omega_dot_raw)
    rhs_b = np.array(
        [
            peak_omega_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak) - omega_e,
            peak_omegadot_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak) - omega_dot_e,
        ],
        dtype=float,
    )
    try:
        b1, b2 = [float(x) for x in np.linalg.solve(p_matrix, rhs_b)]
    except np.linalg.LinAlgError:
        b1, b2 = [float(x) for x in np.linalg.lstsq(p_matrix, rhs_b, rcond=None)[0]]

    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": 0.0,
        "a5": 0.0,
        "b1": b1,
        "b2": b2,
        "b3": 0.0,
        "b4": 0.0,
        "peak_amp": abs(peak_amp_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak)),
        "peak_adot": peak_adot_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak),
        "peak_addot": peak_addot_v4(mode_l, mode_m, params.eta, chiS_peak, chiA_peak),
    }


def _last_traj_index_leq(traj, value: float) -> int:
    """Last trajectory sample with t <= value."""
    if not traj:
        return 0
    idx = len(traj) - 1
    while idx > 0 and float(traj[idx][0]) > value:
        idx -= 1
    return idx


def _join_adaptive_his_trajectories(traj_adas, traj_his, index_start_his: int):
    """Join AdaS prefix and HiS tail using LAL's exclusion of the join sample."""
    if not traj_his:
        return traj_adas
    prefix = traj_adas[: max(0, int(index_start_his))]
    return prefix + traj_his


def _join_uniform_dynamics_until_attach(
    t_adas_M: torch.Tensor | None,
    y_adas: torch.Tensor | None,
    t_his_M: torch.Tensor,
    y_his: torch.Tensor,
    index_start_his: int,
    t_attach_M: float,
):
    """Joined AdaS/HiS dynamics cut like LAL's ``SEOBJoinDynamics``."""

    prefix_len = 0
    if t_adas_M is not None and y_adas is not None:
        prefix_len = max(0, min(int(index_start_his), int(t_adas_M.numel())))

    if prefix_len > 0:
        t_join = torch.cat([t_adas_M[:prefix_len], t_his_M])
        y_join = torch.cat([y_adas[:prefix_len], y_his], dim=0)
    else:
        t_join = t_his_M
        y_join = y_his

    # LAL finds the last mode-grid index <= tAttach, then passes that index as
    # the excluded dynamics length to SEOBJoinDynamics.
    index_join_attach = _last_index_leq_increasing(t_join, t_attach_M)
    cut_len = max(1, min(int(index_join_attach), int(t_join.numel())))
    return t_join[:cut_len], y_join[:cut_len], index_join_attach


def _trajectory_tensors(traj):
    """Stack a reduced-state trajectory into dimensionless times and states."""
    if not traj:
        return None, None
    return torch.stack([pt[0] for pt in traj]), torch.stack([pt[1] for pt in traj])


def _sample_trajectory_states(t_query_M: torch.Tensor, t_src_M: torch.Tensor, y_src: torch.Tensor):
    """Cubic-sample reduced trajectory states at dimensionless query times."""
    return _interp_series_cubic(t_query_M, t_src_M, y_src)


def _interpolation_window_indices(t_src: torch.Tensor, value: float, half_width: int = 20):
    """Return LAL's local cubic interpolation window around a query time."""
    n = int(t_src.numel())
    if n <= 1:
        return 0, max(0, n - 1)
    value_t = torch.as_tensor(value, device=t_src.device, dtype=t_src.dtype)
    indext = int(torch.searchsorted(t_src, value_t, right=True).item()) - 1
    indext = max(0, min(indext, n - 1))
    indexstart = indext - half_width if indext - half_width > 0 else 0
    indexend = indext + half_width if indext + half_width < n - 1 else n - 1
    return indexstart, indexend


def _interp_series_cubic_lal_local(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, half_width: int = 20) -> torch.Tensor:
    """Cubic interpolation using LAL's `SEOBInterpolateDynamicsAtTime` window."""
    if x_new.ndim == 0:
        x_query = x_new.reshape(1)
        squeeze = True
    else:
        x_query = x_new
        squeeze = False

    out = []
    for q in x_query:
        start, end = _interpolation_window_indices(x, float(q), half_width=half_width)
        x_win = x[start : end + 1]
        y_win = y[start : end + 1]
        out.append(_interp_series_cubic(q.reshape(1), x_win, y_win)[0])
    result = torch.stack(out, dim=0)
    return result[0] if squeeze else result


def _interpolate_traj_at_time(traj, t_eval: float):
    """Local-window dynamics interpolation for the HiS IC transfer."""
    t_src, y_src = _trajectory_tensors(traj)
    if t_src is None:
        return None
    t_query = torch.as_tensor(t_eval, device=t_src.device, dtype=t_src.dtype)
    return _interp_series_cubic_lal_local(t_query, t_src, y_src)


def _his_uniform_time_grid(t_his_M: torch.Tensor, *, step_M: float = 1.0 / 50.0):
    """Fixed HiS mode grid, starting from the HiS handoff sample."""
    start = float(t_his_M[0])
    end = float(t_his_M[-1])
    if end <= start:
        return t_his_M[:1]
    n = max(2, int(math.floor((end - start) / step_M)) + 1)
    return torch.as_tensor(start, device=t_his_M.device, dtype=t_his_M.dtype) + torch.arange(n, device=t_his_M.device, dtype=t_his_M.dtype) * step_M


def _omega_from_hamiltonian_velocity(
    r: torch.Tensor,
    pr: torch.Tensor,
    phi: torch.Tensor,
    Lvec: torch.Tensor,
    S1vec: torch.Tensor,
    S2vec: torch.Tensor,
    params: _dyn.EOBParams,
    *,
    chunk_size: int = 512,
    r_vec: torch.Tensor | None = None,
    p_vec: torch.Tensor | None = None,
):
    """LAL omegaVec from r x rdot rather than differentiating orbital phase."""

    if (r_vec is None) != (p_vec is None):
        raise ValueError("r_vec and p_vec must be supplied together")

    omega_parts = []
    n = int(r.numel())
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sl = slice(start, end)
        r_c = r[sl]
        pr_c = pr[sl]
        phi_c = phi[sl]
        L_c = Lvec[sl]
        S1_c = S1vec[sl]
        S2_c = S2vec[sl]
        if r_vec is None:
            r_vec_c, p_vec_c, _, _, _ = _dyn._reduced_to_cartesian(
                r_c,
                pr_c,
                phi_c,
                L_c,
                S1_c,
                S2_c,
                params,
            )
        else:
            r_vec_c = r_vec[sl]
            p_vec_c = p_vec[sl]
        pot = _dyn._eob_potentials(
            r_c,
            pr_c,
            phi_c,
            L_c,
            S1_c,
            S2_c,
            params,
            p_vec=p_vec_c,
            r_vec=r_vec_c,
            compute_grad_p=False,
            compute_base_grad=False,
            fd_dpvec=True,
            fd_pphi=False,
        )
        nhat_dot_dH = torch.sum(pot["dH_dpvec"] * pot["n_hat"], dim=-1)
        vel = pot["dH_dpvec"] + (pot["csi"] - 1.0).unsqueeze(-1) * nhat_dot_dH.unsqueeze(-1) * pot["n_hat"]
        omega = torch.linalg.norm(torch.cross(r_vec_c, vel, dim=-1), dim=-1) / torch.clamp(r_c * r_c, min=1.0e-24)
        omega_parts.append(omega.detach())
    return torch.cat(omega_parts) if omega_parts else torch.empty_like(r)


def _series_from_states(t_M: torch.Tensor, y: torch.Tensor, params: _dyn.EOBParams):
    """Extract PHM mode-building series from reduced states on a given grid."""
    phi = y[:, 2]
    S1vec = y[:, 6:9]
    S2vec = y[:, 9:12]
    r_vec, p_vec = _retained_cartesian_vectors(y)
    if r_vec is None:
        r = y[:, 0]
        pr = y[:, 1]
        Lvec = y[:, 3:6]
        omega_dimless = _omega_from_hamiltonian_velocity(
            r, pr, phi, Lvec, S1vec, S2vec, params
        )
    else:
        r = torch.linalg.norm(r_vec, dim=-1)
        n_hat = r_vec / torch.clamp(r.unsqueeze(-1), min=1.0e-15)
        pr = torch.sum(p_vec * n_hat, dim=-1)
        Lvec = torch.cross(r_vec, p_vec, dim=-1)
        omega_dimless = _omega_from_hamiltonian_velocity(
            r,
            pr,
            phi,
            Lvec,
            S1vec,
            S2vec,
            params,
            r_vec=r_vec,
            p_vec=p_vec,
        )
    omega_orb = omega_dimless / params.M_sec
    return phi, r, pr, Lvec, S1vec, S2vec, omega_orb


def _interpolate_modes_complex(modes, t_src_M: torch.Tensor, t_out_M: torch.Tensor):
    """Interpolate complex modes by their real/imaginary parts."""
    out = {}
    for key, h in modes.items():
        real = _interp_series_cubic(t_out_M, t_src_M, h.real)
        imag = _interp_series_cubic(t_out_M, t_src_M, h.imag)
        out[key] = torch.complex(real, imag).to(h.dtype)
    return out


def _join_mode_dicts(modes_adas, modes_his):
    """Join AdaS-prefix modes to the HiS+RD modes on the same keys."""
    if not modes_adas:
        return modes_his
    joined = {}
    keys = set(modes_his.keys()) | set(modes_adas.keys())
    for key in keys:
        h_his = modes_his.get(key)
        h_adas = modes_adas.get(key)
        if h_his is None:
            joined[key] = h_adas
        elif h_adas is None:
            joined[key] = h_his
        else:
            joined[key] = torch.cat([h_adas, h_his])
    return joined


def _lal_stop_quantities_from_cartesian(r_vec, p_vec, rdot_vec, pdot_vec):
    """Cartesian scalars used by LAL's precessing PR stop callback."""

    r2 = torch.clamp(torch.dot(r_vec, r_vec), min=1.0e-24)
    r = torch.sqrt(r2)
    omega = torch.linalg.norm(torch.cross(r_vec, rdot_vec, dim=0)) / r2
    p_dot_r = torch.dot(p_vec, r_vec) / torch.clamp(r, min=1.0e-12)
    drdt = torch.dot(rdot_vec, r_vec) / torch.clamp(r, min=1.0e-12)
    pr_dot = (
        -torch.dot(p_vec, r_vec) * drdt / r2
        + torch.dot(pdot_vec, r_vec) / torch.clamp(r, min=1.0e-12)
        + torch.dot(rdot_vec, p_vec) / torch.clamp(r, min=1.0e-12)
    )
    return {
        "r": r,
        "r2": r2,
        "omega": omega,
        "p_dot_r": p_dot_r,
        "drdt": drdt,
        "pr_dot": pr_dot,
        "p_norm": torch.linalg.norm(p_vec),
        "dpdt_large": torch.any(torch.abs(pdot_vec) > 10.0),
        "pphi_large": p_vec[2] > 10.0,
    }


def _lal_stop_quantities_from_reduced(y, dy, params: _dyn.EOBParams):
    """Directional finite-difference bridge from reduced state to LAL stop scalars."""

    r_vec, p_vec, _, _, _ = _dyn._reduced_to_cartesian(
        y[0],
        y[1],
        y[2],
        y[3:6],
        y[6:9],
        y[9:12],
        params,
    )
    eps = torch.as_tensor(1.0e-6, device=y.device, dtype=y.dtype) / torch.clamp(
        torch.linalg.norm(dy),
        min=1.0,
    )
    y_eps = y + eps * dy
    r_vec_eps, p_vec_eps, _, _, _ = _dyn._reduced_to_cartesian(
        y_eps[0],
        y_eps[1],
        y_eps[2],
        y_eps[3:6],
        y_eps[6:9],
        y_eps[9:12],
        params,
    )
    rdot_vec = (r_vec_eps - r_vec) / eps
    pdot_vec = (p_vec_eps - p_vec) / eps
    return _lal_stop_quantities_from_cartesian(r_vec, p_vec, rdot_vec, pdot_vec)


def _lal_adas_initial_step_M(params: _dyn.EOBParams, delta_t: float | None) -> float:
    """LAL's AdaS initial step: ``INdeltaT / mTScaled`` in units of total mass."""

    delta_t_sec = 1.0 / 4096.0 if delta_t in (None, 0) else float(delta_t)
    return delta_t_sec / max(float(params.M_sec), 1.0e-30)


def _integrate_traj(
    params: _dyn.EOBParams,
    *,
    device,
    real_dtype,
    f_final: float,
    delta_t: float | None = None,
    return_segments: bool = False,
):
    """Integrate precessing EOB dynamics until stop condition (mimic LAL)."""

    aligned_spins = getattr(params, "aligned_spins", False)
    # LAL evolves the generic-precessing dynamics as a 14-component Cartesian
    # state: x, p, weighted spins, phiMod, phiDMod.  The reduced 12-state path
    # is useful for faster diagnostics, but it adapts the RK error in different
    # coordinates and cannot reproduce LAL's sparse time grid.
    full_cartesian_default = "0" if aligned_spins else "1"
    full_cartesian = os.environ.get(
        "PYCBC_SEOBNRV4PHM_FULL_CARTESIAN", full_cartesian_default
    ) not in ("0", "", "false", "False")
    if full_cartesian:
        rhs_fn = _dyn.rhs_cartesian_full
    elif os.environ.get("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1") in ("0", "", "false", "False"):
        rhs_fn = _dyn.rhs
    else:
        rhs_fn = _dyn.rhs_cartesian_projected

    if os.environ.get("PYCBC_SEOBNRV4PHM_AUTO_TMAX", "1") not in ("0", "", "false", "False"):
        omega0 = max(math.pi * params.f_lower * params.M_sec, 1.0e-12)
        t_newton = 5.0 / (256.0 * max(params.eta, 1.0e-12)) * omega0 ** (-8.0 / 3.0)
        tmax_default = min(12000.0, max(2000.0, 1.35 * t_newton))
    else:
        tmax_default = 2000.0
    tmax = float(os.environ.get("PYCBC_SEOBNRV4PHM_TMAX_M", tmax_default))

    def _run(
        y0,
        h0,
        rtol,
        atol,
        tmax,
        omega_stop=True,
        t0=0.0,
        initial_prev_omega=0.0,
        initial_prev_dr=0.0,
        initial_omega_peaked=0,
        return_stop_state=False,
        h_min=None,
        label="adas",
    ):
        t_start = time.time()
        max_steps = _integration_max_steps(label, 200000)
        prev_omega = torch.tensor(float(initial_prev_omega), device=device, dtype=real_dtype)
        prev_dr = torch.tensor(float(initial_prev_dr), device=device, dtype=real_dtype)
        omega_peaked = int(initial_omega_peaked)
        stop_reason = None

        def _stop(reason):
            nonlocal stop_reason
            stop_reason = reason
            return True

        def _stop_fn(t, y, dy_estimate=None):
            nonlocal omega_peaked
            if full_cartesian:
                r_vec = y[0:3]
                r = torch.linalg.norm(r_vec)
            else:
                r = y[0]
            try:
                dy = dy_estimate if dy_estimate is not None else rhs_fn(t, y, params)
                if full_cartesian:
                    omega = torch.abs(dy[12] + dy[13])
                    drdt = torch.dot(dy[0:3], r_vec / torch.clamp(r, min=1.0e-12))
                else:
                    omega = torch.abs(dy[2])
                    drdt = dy[0]
            except RuntimeError:
                return True
            if not (torch.isfinite(y).all() and torch.isfinite(dy).all()):
                return True
            if aligned_spins and not full_cartesian:
                if not omega_stop:
                    return False
                omega_aligned = omega
                if t0 > 0.0:
                    if r < 6.0 and omega_aligned < prev_omega:
                        omega_peaked += 1
                    should_stop = dy[1] >= 0.0 or omega_peaked == 5
                    prev_omega.copy_(omega_aligned)
                    return _stop("aligned_his") if bool(should_stop) else False
                if r < 6.0 and omega_aligned < prev_omega:
                    return _stop("aligned_omega_peak")
                prev_omega.copy_(omega_aligned)
                return False
            if omega_stop:
                if full_cartesian:
                    rdot_vec = dy[0:3]
                    pdot_vec = dy[3:6]
                    p_vec = y[3:6]
                    stop_q = _lal_stop_quantities_from_cartesian(r_vec, p_vec, rdot_vec, pdot_vec)
                else:
                    stop_q = _lal_stop_quantities_from_reduced(y, dy, params)

                r2 = stop_q["r2"]
                omega = stop_q["omega"]
                p_dot_r = stop_q["p_dot_r"]
                drdt = stop_q["drdt"]
                pr_dot = stop_q["pr_dot"]
                p_norm = stop_q["p_norm"]
                dpdt_large = stop_q["dpdt_large"]
                pphi_large = stop_q["pphi_large"]

                if r2 < 16.0 and (p_dot_r >= 0.0 or drdt >= 0.0):
                    return _stop(0 if p_dot_r >= 0.0 else 1)
                if r2 < 4.0 and pr_dot > 0.0:
                    return _stop(2)
                if r2 < 16.0 and (p_norm > 10.0 or p_norm < 1.0e-10):
                    return _stop(3 if p_norm > 10.0 else 4)
                if r2 < 16.0 and omega < prev_omega:
                    omega_peaked = 1
                if r2 < 4.0 and omega_peaked == 1 and omega > prev_omega:
                    return _stop(5)
                if (r2 < 16.0 and omega < 0.04) or (r2 < 4.0 and omega < 0.14 and omega_peaked == 1):
                    return _stop(6)
                if r2 < 16.0 and omega > 1.0:
                    return _stop(7)
                # LAL updates eobParams->omega after the omega-specific stop
                # checks, but before dpdt/pphi/rdot-increase termination.
                prev_omega.copy_(omega)
                if r2 < 25.0 and bool(dpdt_large):
                    return _stop(8)
                if r2 < 16.0 and pphi_large:
                    return _stop(9)
                if r2 < 9.0 and drdt > prev_dr:
                    prev_dr.copy_(drdt)
                    return _stop(10)
                prev_dr.copy_(drdt)
            return False

        _stop_fn._uses_derivative_estimate = full_cartesian

        traj_out, ode_diagnostics = integrate(
            lambda t, y: rhs_fn(t, y, params),
            y0,
            float(t0),
            tmax,
            h0,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            stop_fn=_stop_fn,
            h_min=h_min,
            progress_label=label,
            return_diagnostics=True,
        )
        _dbg(f"integrate pass label={label} steps={len(traj_out)} max_steps={max_steps} h0={h0} rtol={rtol} atol={atol} tmax={tmax} dt={time.time()-t_start:.3f}s")
        if return_stop_state:
            return traj_out, {
                "prev_omega": float(prev_omega.item()),
                "prev_dr": float(prev_dr.item()),
                "omega_peaked": int(omega_peaked),
                "stop_reason": stop_reason,
                "ode": ode_diagnostics,
            }
        return traj_out

    if full_cartesian:
        y0 = _dyn.initial_cartesian_conditions(params, device=device, dtype=real_dtype)
        y0_reduced = _dyn.cartesian_state_to_reduced_state(y0, params)
    else:
        y0_reduced = _dyn.initial_conditions(params, device=device, dtype=real_dtype)
        y0 = y0_reduced
    # LAL does not refresh seobCoeffs from the final 14-state here; the first
    # RHS uses the coefficient state left by the initial-condition setup.
    if _debug_enabled():
        _dbg(f"initial r0={float(y0_reduced[0]):.6f} pr0={float(y0_reduced[1]):.6e} Lz0={float(y0_reduced[5]):.6f}")
    # coarse pass to locate omega peak. LAL initializes the adaptive AdaS
    # integrator with deltaT = INdeltaT / mTScaled.
    _dbg("coarse integration start")
    adas_h0 = _lal_adas_initial_step_M(params, delta_t)
    adas_rtol = 1e-9 if aligned_spins else 1e-8
    adas_atol = 1e-10 if aligned_spins else 1e-8
    his_rtol = adas_rtol if aligned_spins else 1e-8
    his_atol = adas_atol if aligned_spins else 1e-8
    from pycbc.waveform.seobnrv4phm_constants import DELTA_T_MIN
    adas_h_min = float(os.environ.get("PYCBC_SEOBNRV4PHM_HMIN", str(DELTA_T_MIN)))
    traj_coarse_raw, adas_stop_state = _run(
        y0,
        h0=adas_h0,
        rtol=adas_rtol,
        atol=adas_atol,
        tmax=tmax,
        omega_stop=True,
        return_stop_state=True,
        h_min=adas_h_min,
        label="adas",
    )
    if not traj_coarse_raw:
        _dbg("coarse integration failed, retrying relaxed stop")
        traj_coarse_raw, adas_stop_state = _run(
            y0,
            h0=5e-3,
            rtol=5e-4,
            atol=1e-6,
            tmax=tmax,
            omega_stop=False,
            return_stop_state=True,
            h_min=adas_h_min,
            label="adas_retry",
        )
        if not traj_coarse_raw:
            traj_coarse_raw = []
            adas_stop_state = {"prev_omega": 0.0, "prev_dr": 0.0, "omega_peaked": 0, "stop_reason": None}
    # LAL's XLALAdaptiveRungeKutta4NoInterpolate stores the initial sample in
    # buffer slot 0 and, for the EOBversion=2 path used here, copies slots
    # [0, outputlength) back out. That keeps t=0 and excludes the final accepted
    # stop state.
    traj_coarse = [(torch.tensor(0.0, device=device, dtype=real_dtype), y0)]
    if traj_coarse_raw:
        traj_coarse.extend(traj_coarse_raw[:-1])
    t_stop = float(traj_coarse[-1][0].item())
    tstart_his_target = max(0.0, t_stop - float(T_STEP_BACK))
    index_start_his = _last_traj_index_leq(traj_coarse, tstart_his_target)
    tstart_his = float(traj_coarse[index_start_his][0].item())
    ystart_his = _interpolate_traj_at_time(traj_coarse, tstart_his)
    if ystart_his is None:
        ystart_his = traj_coarse[index_start_his][1]

    if os.environ.get("PYCBC_SEOBNRV4PHM_ADAS_ONLY", "0") not in ("0", "", "false", "False"):
        traj = traj_coarse
        traj_his = []
        his_stop_state = {"prev_omega": 0.0, "prev_dr": 0.0, "omega_peaked": 0, "stop_reason": "adas_only"}
        if full_cartesian:
            traj = [(ti, _waveform_state_from_cartesian(yi, params)) for ti, yi in traj]
            traj_coarse = [(ti, _waveform_state_from_cartesian(yi, params)) for ti, yi in traj_coarse]
        if return_segments:
            return _TrajectorySegments(
                traj=traj,
                traj_adas=traj_coarse,
                traj_his=traj_his,
                index_start_his=index_start_his,
                tstart_his=tstart_his,
                adas_stop_state=adas_stop_state,
                his_stop_state=his_stop_state,
            )
        return traj

    if os.environ.get("PYCBC_SEOBNRV4PHM_JOIN_HIS", "1") not in ("0", "", "false", "False"):
        # LAL steps back on the AdaS trajectory and reruns only the late segment
        # at high sampling before joining AdaS<tstartHiS with HiS. The HiS IC
        # is copied from a local cubic interpolation of AdaS dynamics, matching
        # SEOBInterpolateDynamicsAtTime/ICvaluesHiS.
        _dbg(f"HiS integration start t_stop={t_stop:.3f} tstartHiS={tstart_his:.3f} index={index_start_his}")
        traj_his_tail, his_stop_state = _run(
            ystart_his,
            h0=1.0 / 50.0,
            rtol=his_rtol,
            atol=his_atol,
            tmax=max(t_stop + 50.0, tstart_his + float(T_STEP_BACK) + 50.0),
            omega_stop=True,
            t0=tstart_his,
            initial_prev_omega=adas_stop_state["prev_omega"],
            initial_prev_dr=0.0,
            initial_omega_peaked=adas_stop_state["omega_peaked"],
            return_stop_state=True,
            h_min=0.0,
            label="his",
        )
        traj_his = [(torch.tensor(tstart_his, device=device, dtype=real_dtype), ystart_his)] + traj_his_tail
        traj = _join_adaptive_his_trajectories(traj_coarse, traj_his, index_start_his) if traj_his_tail else traj_coarse
    else:
        # Diagnostic fallback: rerun the high-accuracy trajectory from t=0.
        _dbg(f"HiS integration full-rerun start t_stop={t_stop:.3f} tstartHiS={tstart_his:.3f} index={index_start_his}")
        traj_his = _run(
            y0,
            h0=2e-3,
            rtol=1e-5,
            atol=5e-8,
            tmax=max(t_stop + 50.0, tstart_his + 200.0),
            omega_stop=True,
            label="his_full",
        )
        traj = traj_his if traj_his else traj_coarse
        his_stop_state = {"prev_omega": 0.0, "prev_dr": 0.0, "omega_peaked": 0, "stop_reason": None}
    if full_cartesian:
        traj = [(ti, _waveform_state_from_cartesian(yi, params)) for ti, yi in traj]
        traj_coarse = [(ti, _waveform_state_from_cartesian(yi, params)) for ti, yi in traj_coarse]
        traj_his = [(ti, _waveform_state_from_cartesian(yi, params)) for ti, yi in traj_his]
    if return_segments:
        return _TrajectorySegments(
            traj=traj,
            traj_adas=traj_coarse,
            traj_his=traj_his if traj_his else traj,
            index_start_his=index_start_his,
            tstart_his=tstart_his,
            adas_stop_state=adas_stop_state,
            his_stop_state=his_stop_state,
        )
    return traj


def _native_params_from_kwargs(p):
    mode_array = _dyn.normalize_mode_array(p.get("mode_array", None))
    return _dyn.EOBParams(
        mass1=p["mass1"],
        mass2=p["mass2"],
        spin1x=p.get("spin1x", 0.0),
        spin1y=p.get("spin1y", 0.0),
        spin1z=p.get("spin1z", 0.0),
        spin2x=p.get("spin2x", 0.0),
        spin2y=p.get("spin2y", 0.0),
        spin2z=p.get("spin2z", 0.0),
        distance=p["distance"],
        inclination=p.get("inclination", 0.0),
        f_lower=p["f_lower"],
        f_ref=p.get("f_ref", p["f_lower"]),
        mode_array=mode_array,
    )


def _phm_boundary_summary_from_segments(
    params: _dyn.EOBParams,
    segments: _TrajectorySegments,
    t_attach_M: float,
):
    """Summarize AdaS/HiS/join boundaries without mode or ringdown work."""

    def sample_rows(t_vec, y_vec, n_rows=5):
        n = min(int(n_rows), int(t_vec.numel()))
        out = []
        for i in range(n):
            state = y_vec[i]
            out.append(
                {
                    "t_M": float(t_vec[i].item()),
                    "r": float(state[0].item()),
                    "state_prefix": [
                        float(x)
                        for x in state[: min(6, int(state.numel()))].detach().cpu().tolist()
                    ],
                }
            )
        return out

    traj = segments.traj
    t_adas_M, y_adas = _trajectory_tensors(segments.traj_adas)
    t_his_src_M, y_his_src = _trajectory_tensors(segments.traj_his)
    if t_his_src_M is None or t_his_src_M.numel() < 3:
        t_his_src_M, y_his_src = _trajectory_tensors(traj)

    delta_t_his_M = 1.0 / 50.0
    t_his_M = _his_uniform_time_grid(t_his_src_M, step_M=delta_t_his_M)
    y_his = _sample_trajectory_states(t_his_M, t_his_src_M, y_his_src)

    t_join_dyn_M, y_join_dyn, index_join_attach = _join_uniform_dynamics_until_attach(
        t_adas_M,
        y_adas,
        t_his_M,
        y_his,
        segments.index_start_his,
        t_attach_M,
    )
    prefix_len = 0
    if t_adas_M is not None:
        prefix_len = max(0, min(int(segments.index_start_his), int(t_adas_M.numel())))
    first_his_excluded = int(index_join_attach) - prefix_len
    t_first_his_excluded_M = None
    if 0 <= first_his_excluded < int(t_his_M.numel()):
        t_first_his_excluded_M = float(t_his_M[first_his_excluded].item())

    return {
        "mode_array": tuple(params.mode_array),
        "n_traj": int(t_join_dyn_M.numel()),
        "n_adas": 0 if t_adas_M is None else int(t_adas_M.numel()),
        "n_his": int(t_his_M.numel()),
        "n_joined": int(t_join_dyn_M.numel()),
        "n_his_src": int(t_his_src_M.numel()),
        "n_his_uniform": int(t_his_M.numel()),
        "index_start_his": int(segments.index_start_his),
        "index_join_his": int(segments.index_start_his),
        "index_join_attach": int(index_join_attach),
        "t_start_his_M": float(segments.tstart_his),
        "t_join_his_M": float(t_his_M[0].item()),
        "t_adas_before_his_M": (
            None
            if t_adas_M is None or prefix_len <= 0
            else float(t_adas_M[prefix_len - 1].item())
        ),
        "t_attach_M": float(t_attach_M),
        "t_join_attach_M": t_first_his_excluded_M,
        "t_first_his_excluded_M": t_first_his_excluded_M,
        "t_adas_end_M": None if t_adas_M is None else float(t_adas_M[-1].item()),
        "t_his_src_end_M": float(t_his_src_M[-1].item()),
        "t_his_end_M": float(t_his_M[-1].item()),
        "t_end_M": float(t_join_dyn_M[-1].item()),
        "joined_end_M": float(t_join_dyn_M[-1].item()),
        "r_start": float(y_join_dyn[0, 0].item()),
        "r_his_start": float(y_his[0, 0].item()),
        "r_end": float(y_join_dyn[-1, 0].item()),
        "r_adas_end": None if y_adas is None else float(y_adas[-1, 0].item()),
        "r_his_end": float(y_his[-1, 0].item()),
        "adas_samples": [] if t_adas_M is None else sample_rows(t_adas_M, y_adas),
        "his_src_samples": sample_rows(t_his_src_M, y_his_src),
        "his_uniform_samples": sample_rows(t_his_M, y_his),
        "joined_samples": sample_rows(t_join_dyn_M, y_join_dyn),
        "adas_stop_state": segments.adas_stop_state,
        "his_stop_state": segments.his_stop_state,
        "rhs_derivative_options": _dyn._rhs_derivative_options(),
    }


def _seobnrv4phm_native_boundary_summary(**p):
    """Return native PHM trajectory boundaries, optionally using supplied tAttach."""

    params = _native_params_from_kwargs(p)
    device, real_dtype, _complex_dtype = _target_device_dtypes()

    delta_t_in = p.get("delta_t", None)
    delta_t = 1.0 / 4096.0 if delta_t_in in (None, 0) else float(delta_t_in)
    f_final = float(p.get("f_final", 0.0))

    segments = _integrate_traj(
        params,
        device=device,
        real_dtype=real_dtype,
        f_final=f_final,
        delta_t=delta_t,
        return_segments=True,
    )

    t_attach_M = p.get("t_attach_M", None)
    if t_attach_M is None:
        t_his_src_M, y_his_src = _trajectory_tensors(segments.traj_his)
        if t_his_src_M is None or t_his_src_M.numel() < 3:
            t_his_src_M, y_his_src = _trajectory_tensors(segments.traj)
        t_his_M = _his_uniform_time_grid(t_his_src_M, step_M=1.0 / 50.0)
        y_his = _sample_trajectory_states(t_his_M, t_his_src_M, y_his_src)
        (
            _phi_his,
            _r_his,
            _pr_his,
            Lvec_his,
            S1vec_his,
            S2vec_his,
            omega_his,
        ) = _series_from_states(t_his_M, y_his, params)
        _, _t_peak_omega_M, t_attach_M, _chi1L_peak, _chi2L_peak = _phm_merger_timing(
            t_his_M * params.M_sec,
            omega_his,
            Lvec_his,
            S1vec_his,
            S2vec_his,
            params,
        )

    return _phm_boundary_summary_from_segments(params, segments, float(t_attach_M))


def _seobnrv4phm_native_checkpoint_summary(**p):
    """Return early native PHM checkpoints for LAL parity diagnostics."""

    params = _native_params_from_kwargs(p)
    device, real_dtype, complex_dtype = _target_device_dtypes()

    delta_t_in = p.get("delta_t", None)
    delta_t = 1.0 / 4096.0 if delta_t_in in (None, 0) else float(delta_t_in)
    f_final = float(p.get("f_final", 0.0))

    segments = _integrate_traj(
        params,
        device=device,
        real_dtype=real_dtype,
        f_final=f_final,
        delta_t=delta_t,
        return_segments=True,
    )
    traj = segments.traj
    t_adas_M, y_adas = _trajectory_tensors(segments.traj_adas)
    t_his_src_M, y_his_src = _trajectory_tensors(segments.traj_his)
    if t_his_src_M is None or t_his_src_M.numel() < 3:
        t_his_src_M, y_his_src = _trajectory_tensors(traj)

    delta_t_his_M = 1.0 / 50.0
    t_his_M = _his_uniform_time_grid(t_his_src_M, step_M=delta_t_his_M)
    y_his = _sample_trajectory_states(t_his_M, t_his_src_M, y_his_src)
    phi_his, r_his, pr_his, Lvec_his, S1vec_his, S2vec_his, omega_his = _series_from_states(t_his_M, y_his, params)

    _t_vec_M, t_peak_omega_M, t_attach_M, chi1L_peak, chi2L_peak = _phm_merger_timing(
        t_his_M * params.M_sec,
        omega_his,
        Lvec_his,
        S1vec_his,
        S2vec_his,
        params,
    )
    _dbg(
        f"merger timing tPeakOmega={t_peak_omega_M:.3f}M "
        f"tAttach={t_attach_M:.3f}M chi1L={chi1L_peak:.6f} chi2L={chi2L_peak:.6f}"
    )
    params.cal21 = 0.0
    params.cal55 = 0.0

    s1_scale = (params.mass1 / params.M) ** 2
    s2_scale = (params.mass2 / params.M) ** 2
    t_peak_q = torch.tensor(min(max(t_peak_omega_M, float(t_his_M[0])), float(t_his_M[-1])), device=device, dtype=real_dtype)
    L_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, Lvec_his)
    S1_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, S1vec_his)
    S2_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, S2vec_his)
    J_final = params.eta * L_peak + s1_scale * S1_peak + s2_scale * S2_peak
    cos_angle = torch.dot(
        L_peak / torch.clamp(torch.linalg.norm(L_peak), min=1e-15),
        J_final / torch.clamp(torch.linalg.norm(J_final), min=1e-15),
    ).item()
    (finmass_frac, finspin_mag), time_10M, _, _ = _final_mass_spin_from_adas_10M(t_adas_M, y_adas, params)
    finspin = _signed_clamped_final_spin(finspin_mag, cos_angle)

    boundary_summary = _phm_boundary_summary_from_segments(params, segments, t_attach_M)
    t_join_dyn_M, y_join_dyn, index_join_attach = _join_uniform_dynamics_until_attach(
        t_adas_M,
        y_adas,
        t_his_M,
        y_his,
        segments.index_start_his,
        t_attach_M,
    )
    _, r_join, _, _, _, _, omega_join = _series_from_states(t_join_dyn_M, y_join_dyn, params)
    omega_join_dimless = omega_join * params.M_sec
    idx_peak_omega = int(torch.argmax(omega_join_dimless).item())
    nqc_a22 = getattr(params, "nqc_a_map", {}).get((2, 2), {})
    nqc_b22 = getattr(params, "nqc_b_map", {}).get((2, 2), {})
    summary = {
        "mode_array": tuple(params.mode_array),
        "n_traj": int(t_join_dyn_M.numel()),
        "t_end_M": float(t_join_dyn_M[-1].item()),
        "r_start": float(r_join[0].item()),
        "r_end": float(r_join[-1].item()),
        "t_peak_omega_M": float(t_peak_omega_M),
        "t_peak_omega_argmax_M": float(t_join_dyn_M[idx_peak_omega].item()),
        "omega_peak_dimless": float(omega_join_dimless[idx_peak_omega].item()),
        "t_attach_M": float(t_attach_M),
        "chi1L_peak": float(chi1L_peak),
        "chi2L_peak": float(chi2L_peak),
        "final_mass_frac": float(finmass_frac),
        "final_spin": float(finspin),
        "time_10M": None if time_10M is None else float(time_10M),
        "cal21": float(params.cal21),
        "cal55": float(params.cal55),
        "nqc_a22": {k: float(v) for k, v in nqc_a22.items()},
        "nqc_b22": {k: float(v) for k, v in nqc_b22.items()},
        "mode_peak_times_M": {},
        "mode_peak_amps": {},
    }
    summary.update(boundary_summary)
    summary["t_end_M"] = float(t_join_dyn_M[-1].item())
    summary["r_start"] = float(r_join[0].item())
    summary["r_end"] = float(r_join[-1].item())
    return summary


def _seobnrv4phm_td_native(**p):
    """Generate native SEOBNRv4PHM polarizations in the time domain."""
    params = _native_params_from_kwargs(p)
    _dbg(
        f"native start modes={params.mode_array} "
        f"f_final={p.get('f_final', 0.0)}"
    )
    device, real_dtype, complex_dtype = _target_device_dtypes()

    delta_t_in = p.get("delta_t", None)
    delta_t = 1.0 / 4096.0 if delta_t_in in (None, 0) else float(delta_t_in)
    f_final = float(p.get("f_final", 0.0))
    waveform_ell_max = max(int(l) for l, _ in params.mode_array)
    ell_check = int(p.get("ellMaxForNyquistCheck", waveform_ell_max))
    omega_nyquist_check = _check_nyquist_frequency(
        params,
        delta_t,
        ell_check,
        waveform_ell_max=waveform_ell_max,
    )
    _dbg(f"Nyquist check ell={ell_check} omega_rd={omega_nyquist_check:.6e} rad/s")

    segments = _integrate_traj(
        params,
        device=device,
        real_dtype=real_dtype,
        f_final=f_final,
        delta_t=delta_t,
        return_segments=True,
    )
    traj = segments.traj
    _dbg(f"traj len={len(traj)} t_end_M={float(traj[-1][0]) if traj else 0.0}")
    t_adas_M, y_adas = _trajectory_tensors(segments.traj_adas)
    t_his_src_M, y_his_src = _trajectory_tensors(segments.traj_his)
    if t_his_src_M is None or t_his_src_M.numel() < 3:
        t_his_src_M, y_his_src = _trajectory_tensors(traj)

    delta_t_his_M = 1.0 / 50.0
    delta_t_his = delta_t_his_M * params.M_sec
    t_his_M = _his_uniform_time_grid(t_his_src_M, step_M=delta_t_his_M)
    y_his = _sample_trajectory_states(t_his_M, t_his_src_M, y_his_src)
    phi_his, r_his, pr_his, Lvec_his, S1vec_his, S2vec_his, omega_his = _series_from_states(t_his_M, y_his, params)
    r_vec_his, p_vec_his = _retained_cartesian_vectors(y_his)

    t_vec_M, t_peak_omega_M, t_attach_M, chi1L_peak, chi2L_peak = _phm_merger_timing(
        t_his_M * params.M_sec,
        omega_his,
        Lvec_his,
        S1vec_his,
        S2vec_his,
        params,
    )
    _dbg(
        f"merger timing tPeakOmega={t_peak_omega_M:.3f}M "
        f"tAttach={t_attach_M:.3f}M chi1L={chi1L_peak:.6f} chi2L={chi2L_peak:.6f}"
    )
    params.cal21 = 0.0
    params.cal55 = 0.0
    hm_cal = _higher_mode_residual_calibration(
        t_vec_M,
        phi_his,
        omega_his,
        r_his,
        pr_his,
        Lvec_his,
        S1vec_his,
        S2vec_his,
        params,
        t_peak_omega_M=t_peak_omega_M,
        chi1L_peak=chi1L_peak,
        chi2L_peak=chi2L_peak,
        r_vec=r_vec_his,
        p_vec=p_vec_his,
    )
    params.cal21 = hm_cal.get("cal21", 0.0)
    params.cal55 = hm_cal.get("cal55", 0.0)
    if hm_cal:
        _dbg(f"higher-mode residual calibration cal21={params.cal21:.6e} cal55={params.cal55:.6e}")

    s1_scale = (params.mass1 / params.M) ** 2
    s2_scale = (params.mass2 / params.M) ** 2
    t_peak_q = torch.tensor(min(max(t_peak_omega_M, float(t_his_M[0])), float(t_his_M[-1])), device=device, dtype=real_dtype)
    L_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, Lvec_his)
    S1_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, S1vec_his)
    S2_peak = _interp_series_cubic_lal_local(t_peak_q, t_his_M, S2vec_his)
    J_final = params.eta * L_peak + s1_scale * S1_peak + s2_scale * S2_peak

    # Final J-frame basis and constant I->J angles use the HiS peak-Omega state.
    e1J, e2J, e3J = _build_J_frame(J_final)
    if getattr(params, "aligned_spins", False):
        zero_angle = torch.tensor(0.0, device=device, dtype=real_dtype)
        alphaI2J_const = zero_angle
        betaI2J_const = zero_angle
        gammaI2J_const = zero_angle
    else:
        alphaI2J_const, betaI2J_const, gammaI2J_const = _euler_from_basis(e1J, e2J, e3J)

    cos_angle = torch.dot(
        L_peak / torch.clamp(torch.linalg.norm(L_peak), min=1e-15),
        J_final / torch.clamp(torch.linalg.norm(J_final), min=1e-15),
    ).item()
    (finmass_frac, finspin_mag), time_10M, _, _ = _final_mass_spin_from_adas_10M(t_adas_M, y_adas, params)
    finspin = _signed_clamped_final_spin(finspin_mag, cos_angle)
    if time_10M is not None:
        _dbg(f"final mass/spin fit from AdaS r=10M time={time_10M:.3f}M Mf={finmass_frac:.6f} af={finspin:.6f}")

    # LAL extends the Euler angles using its Cardoso-table QNM frequencies,
    # which are also used by the ringdown attachment above.
    prec_rate = _euler_qnm_precession_rate(
        finspin,
        finmass_frac,
        cos_angle,
        device=device,
        dtype=real_dtype,
    )

    # First pass modes to measure peak and update NQC (per mode)
    modes_cp_tmp = _build_coprecessing_modes(
        phi_his,
        omega_his,
        r_his,
        pr_his,
        Lvec_his,
        S1vec_his,
        S2vec_his,
        params,
        params.mode_array,
        distance_scale=False,
        r_vec=r_vec_his,
        p_vec=p_vec_his,
    )
    omega_dimless = omega_his * params.M_sec
    _populate_nqc_coeffs_lal_v4(
        params,
        params.mode_array,
        modes_cp_tmp,
        t_vec_M,
        r_his,
        pr_his,
        omega_dimless,
        t_peak_omega_M=t_peak_omega_M,
        chi1L_peak=chi1L_peak,
        chi2L_peak=chi2L_peak,
    )

    prefix_len = 0
    t_adas_prefix_M = torch.empty(0, device=device, dtype=real_dtype)
    modes_adas = {}
    if t_adas_M is not None and y_adas is not None:
        prefix_len = max(0, min(int(segments.index_start_his), int(t_adas_M.numel())))
        if prefix_len >= 4:
            t_adas_prefix_M = t_adas_M[:prefix_len]
            y_adas_prefix = y_adas[:prefix_len]
            phi_a, r_a, pr_a, Lvec_a, S1vec_a, S2vec_a, omega_a = _series_from_states(t_adas_prefix_M, y_adas_prefix, params)
            r_vec_a, p_vec_a = _retained_cartesian_vectors(y_adas_prefix)
            modes_adas = _build_coprecessing_modes(
                phi_a,
                omega_a,
                r_a,
                pr_a,
                Lvec_a,
                S1vec_a,
                S2vec_a,
                params,
                params.mode_array,
                r_vec=r_vec_a,
                p_vec=p_vec_a,
            )

    modes_his = _build_coprecessing_modes(
        phi_his,
        omega_his,
        r_his,
        pr_his,
        Lvec_his,
        S1vec_his,
        S2vec_his,
        params,
        params.mode_array,
        r_vec=r_vec_his,
        p_vec=p_vec_his,
    )
    _dbg("modes built on joined AdaS/HiS grid (co-precessing)")

    # Attach RD in the P-frame so precession rotation applies to ringdown
    modes_his = _attach_ringdown_modes(
        modes_his,
        params,
        delta_t_his,
        device=device,
        dtype=real_dtype,
        finspin=finspin,
        finmass_frac=finmass_frac,
        t_attach_M=t_attach_M,
        chi1L_attach=chi1L_peak,
        chi2L_attach=chi2L_peak,
        t_start_M=float(t_his_M[0]),
    )
    _dbg("ringdown attached (P-frame)")
    modes_his = _pad_modes_equal_length(modes_his)
    his_rd_len = max(len(v) for v in modes_his.values()) if modes_his else int(t_his_M.numel())
    t_his_rd_M = t_his_M[0] + torch.arange(his_rd_len, device=device, dtype=real_dtype) * delta_t_his_M
    modes_cp = _join_mode_dicts(modes_adas, modes_his)
    modes_cp = _pad_modes_equal_length(modes_cp)
    if prefix_len >= 4:
        t_modes_M = torch.cat([t_adas_prefix_M, t_his_rd_M])
    else:
        t_modes_M = t_his_rd_M
    target_len = max(len(v) for v in modes_cp.values()) if modes_cp else 0
    t_peak_amp_M, index_peak_amp = _amplitude_peak_from_22_21(
        modes_cp,
        params.mode_array,
        t_modes_M,
    )
    epoch = -float(t_peak_amp_M.item()) * params.M_sec
    _dbg(
        f"amplitude peak tPeak={float(t_peak_amp_M):.3f}M "
        f"index={index_peak_amp} epoch={epoch:.6e}s"
    )

    t_euler_pre_M, y_euler_pre, _ = _join_uniform_dynamics_until_attach(
        t_adas_M,
        y_adas,
        t_his_M,
        y_his,
        segments.index_start_his,
        t_attach_M,
    )
    euler_dyn_len = int(t_euler_pre_M.numel())
    if getattr(params, "aligned_spins", False):
        alphaJ2P, betaJ2P, gammaJ2P = _zero_euler_angles_like(t_modes_M)
        _dbg("Euler J->P kept zero for SpinsAlmostAligned")
    else:
        r_vec_euler, p_vec_euler = _retained_cartesian_vectors(y_euler_pre)
        if r_vec_euler is None:
            Lvec_euler = y_euler_pre[:, 3:6]
            n_hat0, _, _ = _dyn._orbital_basis_from_L_phi(
                y_euler_pre[0, 2],
                Lvec_euler[0],
                y_euler_pre[0, 6:9],
                y_euler_pre[0, 9:12],
                params,
            )
        else:
            Lvec_euler = torch.cross(r_vec_euler, p_vec_euler, dim=-1)
            n_hat0 = r_vec_euler[0] / torch.clamp(
                torch.linalg.norm(r_vec_euler[0]), min=1.0e-15
            )
        alphaJ2P, betaJ2P, gammaJ2P = _euler_j2p(
            Lvec_euler,
            e1J,
            e2J,
            e3J,
            n_hat0=n_hat0,
            t_vec=t_euler_pre_M,
        )

        # LAL computes dynamics Euler angles only before indexJoinAttach, then
        # uses the QNM simple-precession extension through the ringdown patch.
        alphaJ2P, betaJ2P, gammaJ2P = _extend_euler_from_attach_times(
            alphaJ2P,
            betaJ2P,
            gammaJ2P,
            t_modes_M,
            index_start=euler_dyn_len,
            prec_rate=prec_rate,
        )
        _dbg(f"Euler J->P extended from attach index={euler_dyn_len} to len={target_len}")

    t_uniform_M = _lal_output_time_grid(t_modes_M, delta_t, params.M_sec)
    t_uniform = t_uniform_M * params.M_sec
    _dbg(f"rotate/interp modes to uniform n={len(t_uniform)} delta_t={delta_t}")
    # guard against extremely short waveforms
    if len(t_uniform) < 4:
        t_end = float(t_modes_M[-1].item() * params.M_sec) if len(t_modes_M) else 0.0
        t_uniform = torch.linspace(0.0, max(t_end, delta_t * 4.0), steps=8, device=device, dtype=real_dtype)
        delta_t = float(t_uniform[1] - t_uniform[0])
    if delta_t <= 0.0:
        delta_t = 1.0 / 4096.0
        t_end = float(t_modes_M[-1].item() * params.M_sec) if len(t_modes_M) else 0.0
        t_uniform = torch.arange(0.0, max(t_end, delta_t * 4.0), delta_t, device=device, dtype=real_dtype)

    modes_J = _rotate_interpolate_modes_jframe(
        modes_cp,
        alphaJ2P,
        betaJ2P,
        gammaJ2P,
        t_modes_M,
        t_uniform / params.M_sec,
        params.mode_array,
    )
    _dbg("modes rotated/interpolated to J-frame")
    modes_inertial = _rotate_modes_constant(modes_J, alphaI2J_const, betaI2J_const, gammaI2J_const)
    _dbg("modes rotated to inertial")

    hp_t, hc_t = _polarizations_from_modes(
        modes_inertial,
        params.inclination,
        p.get("coa_phase", 0.0),
        device=device,
        complex_dtype=complex_dtype,
    )
    hp_t, hc_t = _rotate_polarizations(
        hp_t,
        hc_t,
        p.get("long_asc_nodes", 0.0),
    )
    _dbg("projected to polarisations")

    hp_ts = TimeSeries(
        TorchArrayData(hp_t),
        delta_t=delta_t,
        epoch=epoch,
        copy=False,
    )
    hc_ts = TimeSeries(
        TorchArrayData(hc_t),
        delta_t=delta_t,
        epoch=epoch,
        copy=False,
    )
    return hp_ts, hc_ts


def _seobnrv4phm_fd_native(**p):
    """Generate native SEOBNRv4PHM polarizations in the frequency domain."""

    hp_td, hc_td = _seobnrv4phm_td_native(**p)
    hp_t = hp_td._data.tensor
    hc_t = hc_td._data.tensor
    complex_dtype = (
        torch.complex64 if hp_t.dtype == torch.float32 else torch.complex128
    )
    delta_f_in = p.get("delta_f", None)
    delta_f_target = None if delta_f_in is None else float(delta_f_in)
    f_final = float(p.get("f_final", 0.0))

    hp_fd, hc_fd, df_out = _time_to_fd(
        hp_t,
        hc_t,
        hp_td.delta_t,
        complex_dtype,
        delta_f_target,
    )
    if f_final > 0.0 and df_out > 0.0:
        n_final = int(f_final / df_out) + 1
        hp_fd = hp_fd[:n_final]
        hc_fd = hc_fd[:n_final]
    _dbg(f"FFT done len={len(hp_fd)} df={df_out}")

    hp_fs = FrequencySeries(
        TorchArrayData(hp_fd),
        delta_f=df_out,
        epoch=hp_td.start_time,
        copy=False,
    )
    hc_fs = FrequencySeries(
        TorchArrayData(hc_fd),
        delta_f=df_out,
        epoch=hc_td.start_time,
        copy=False,
    )
    return hp_fs, hc_fs


def _validate_public_native_request(p):
    """Reject direct native calls that public dispatch would route to LAL."""

    if not torch_native_enabled("PYCBC_SEOBNRV4PHM_NATIVE", default=False):
        raise RuntimeError(
            "SEOBNRv4PHM torch wrapper invoked without PYCBC_SEOBNRV4PHM_NATIVE "
            "(or PYCBC_TORCH_NATIVE_PORTS/PYCBC_TORCH_NATIVE) enabled."
        )
    if not seobnrv4phm_native_supported(p):
        raise ValueError(
            "SEOBNRv4PHM parameters require an unsupported native feature"
        )
    if not isinstance(_scheme.mgr.state, _scheme.TorchScheme):
        raise RuntimeError("native Torch SEOBNRv4PHM requires TorchScheme")


def seobnrv4phm_td_torch(**p):
    """Generate a torch-native SEOBNRv4PHM time-domain waveform."""

    _validate_public_native_request(p)
    return _seobnrv4phm_td_native(**p)


def seobnrv4phm_fd_torch(**p):
    """Generate a torch-native SEOBNRv4PHM frequency-domain waveform."""

    _validate_public_native_request(p)
    return _seobnrv4phm_fd_native(**p)


torch_native_td_waveform = seobnrv4phm_td_torch
torch_native_fd_waveform = seobnrv4phm_fd_torch


__all__ = [
    "seobnrv4phm_native_supported",
    "seobnrv4phm_td_torch",
    "seobnrv4phm_fd_torch",
    "seobnrv4phm_fd_from_td",
    "torch_native_td_waveform",
    "torch_native_fd_waveform",
]
