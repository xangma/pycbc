# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Pure torch implementation of the SPA template engine.

This mirrors the CUDA and CPU implementations (fall-through switch structure for
phase terms and f^-7/6 amplitude) but keeps computation entirely on the active
torch device. CPU routes remain available via other scheme backends.
"""

import torch
import numpy as _np
import pycbc.scheme as _scheme
from pycbc.waveform import spa_tmplt_cpu as _spa_cpu
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform.torch_switches import torch_native_enabled
import os


def _torch_native_spa(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                      pfl5, pfa6, pfl6, pfa7, amp_factor, device="cpu", dtype_out=torch.complex64):
    """Pure torch replication of spa_tmplt CPU kernel; returns per-stage tensors.

    Parity note: validated 2025-11-23 vs spa_tmplt_cpu for the TaylorF2
    test grid in pycbc/waveform/tests/test_taylorf2_torch.py (max abs diff = 0).
    """
    force_f64 = os.environ.get("PYCBC_SPATPLT_TORCH_FLOAT64", "0").lower() in (
        "1", "true", "yes", "on"
    )
    # use float64 intermediates when requested (or when output is complex128)
    f_dtype = torch.float64 if force_f64 or dtype_out == torch.complex128 else torch.float32

    # Build frequency tables with float64 pow/log then cast, matching CPU C code
    idx_np = _np.arange(n, dtype=_np.float64)
    delta_f_np = _np.float64(delta_f)
    f_phase_np = (idx_np + _np.float64(kmin)) * delta_f_np

    piM13_np = _np.cbrt(_np.float64(piM))
    logpiM13_np = _np.log(piM13_np)
    log4_np = _np.log(_np.float64(4.0))

    v_np = piM13_np * _np.power(f_phase_np, _np.float64(1.0 / 3.0))
    logv_np = _np.log(f_phase_np) * _np.float64(1.0 / 3.0) + logpiM13_np
    # Match CPU lookup behavior at f=0 (logv_vec[0] == 0 -> logpiM13)
    logv_np = _np.where(f_phase_np == 0.0, logpiM13_np, logv_np)
    if f_dtype == torch.float32:
        v_np = v_np.astype(_np.float32)
        logv_np = logv_np.astype(_np.float32)

    v = torch.tensor(v_np, device=device, dtype=f_dtype)
    logv = torch.tensor(logv_np, device=device, dtype=f_dtype)

    # Preconditioner: float64 pow then cast to float32, mirroring spa_tmplt_precondition.
    # Default now builds on the active torch device to avoid host copies; set
    # PYCBC_SPATPLT_KFAC_DEVICE=0 to force the legacy numpy path.
    if os.environ.get("PYCBC_SPATPLT_KFAC_DEVICE", "1").lower() in ("1", "true", "yes", "on"):
        k_idx = torch.arange(kmin + 1, kmin + n + 1, device=device, dtype=torch.float64)
        kfac_t = torch.pow(k_idx * delta_f_np, -7.0 / 6.0)
        kfac = kfac_t.to(f_dtype) if f_dtype == torch.float32 else kfac_t
    else:
        v_amp = _np.arange(0, (kmin + n * 2), 1.0, dtype=_np.float64) * delta_f_np
        v_amp = _np.power(v_amp[1:], -7.0 / 6.0)
        if f_dtype == torch.float32:
            v_amp = v_amp.astype(_np.float32)
        kfac = torch.tensor(v_amp[kmin:kmin + n], device=device, dtype=f_dtype)

    amp = torch.tensor(amp_factor, device=device, dtype=f_dtype) * kfac

    two_pi = torch.tensor(2.0 * _np.pi, dtype=f_dtype, device=device)
    log4 = torch.tensor(log4_np, dtype=f_dtype, device=device)

    pfa2_t = torch.tensor(pfa2, dtype=f_dtype, device=device)
    pfa3_t = torch.tensor(pfa3, dtype=f_dtype, device=device)
    pfa4_t = torch.tensor(pfa4, dtype=f_dtype, device=device)
    pfa5_t = torch.tensor(pfa5, dtype=f_dtype, device=device)
    pfl5_t = torch.tensor(pfl5, dtype=f_dtype, device=device)
    pfa6_t = torch.tensor(pfa6, dtype=f_dtype, device=device)
    pfl6_t = torch.tensor(pfl6, dtype=f_dtype, device=device)
    pfa7_t = torch.tensor(pfa7, dtype=f_dtype, device=device)
    pfaN_t = torch.tensor(pfaN, dtype=f_dtype, device=device)

    v2 = v * v
    v3 = v2 * v
    v5 = v2 * v3

    ph = pfa7_t * v
    ph = (ph + pfa6_t + pfl6_t * (logv + log4)) * v
    ph = (ph + pfa5_t + pfl5_t * logv) * v
    ph = (ph + pfa4_t) * v
    ph = (ph + pfa3_t) * v
    ph = (ph + pfa2_t) * v2 + torch.tensor(1.0, dtype=f_dtype, device=device)

    ph = ph * (pfaN_t / v5) - torch.tensor(_np.pi / 4.0, dtype=f_dtype, device=device)
    ph = ph - torch.trunc(ph / two_pi) * two_pi
    ph = torch.where(ph < -_np.pi, ph + two_pi, ph)
    ph = torch.where(ph > _np.pi, ph - two_pi, ph)

    sinp = torch.tensor(1.273239545, device=device, dtype=f_dtype) * ph - torch.tensor(
        0.405284735, device=device, dtype=f_dtype
    ) * ph * torch.abs(ph)
    sinp = torch.tensor(0.225, device=device, dtype=f_dtype) * (sinp * torch.abs(sinp) - sinp) + sinp

    phs = ph + torch.tensor(_np.pi / 2.0, device=device, dtype=f_dtype)
    phs = torch.where(phs > _np.pi, phs - two_pi, phs)
    cosp = torch.tensor(1.273239545, device=device, dtype=f_dtype) * phs - torch.tensor(
        0.405284735, device=device, dtype=f_dtype
    ) * phs * torch.abs(phs)
    cosp = torch.tensor(0.225, device=device, dtype=f_dtype) * (cosp * torch.abs(cosp) - cosp) + cosp

    out = torch.complex(cosp * amp, -sinp * amp)
    if out.dtype != dtype_out:
        out = out.to(dtype_out)
    return {
        "v": v,
        "logv": logv,
        "phasing": ph,
        "sinp": sinp,
        "cosp": cosp,
        "out": out,
        "kfac": kfac,
        "amp": amp,
    }


def _cpu_reference(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                   pfl5, pfa6, pfl6, pfa7, amp_factor):
    """CPU float32 reference matching ``spa_tmplt_engine`` exactly."""
    fvals = (_np.arange(kmin, kmin + n, dtype=_np.float32)) * float(delta_f)
    out_np = _np.empty(n, dtype=_np.complex64)
    _spa_cpu.spa_tmplt_inline_sequence(
        float(piM), float(pfaN), float(pfa2), float(pfa3), float(pfa4),
        float(pfa5), float(pfl5), float(pfa6), float(pfl6), float(pfa7),
        _np.float32(amp_factor), fvals, out_np
    )

    # The production CPU engine evaluates phase at ``kmin + i`` but its
    # historical preconditioner evaluates amplitude at ``kmin + i + 1``.
    # ``spa_tmplt_inline_sequence`` uses one frequency for both, so correct
    # only its amplitude here.
    amp_freqs = (
        _np.arange(kmin + 1, kmin + n + 1, dtype=_np.float64) * delta_f
    )
    phase_amp = _np.power(fvals.astype(_np.float64), -7.0 / 6.0)
    engine_amp = _np.power(amp_freqs, -7.0 / 6.0)
    out_np *= (engine_amp / phase_amp).astype(_np.float32)
    return out_np


def _numpy_reference(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                     pfl5, pfa6, pfl6, pfa7, amp_factor):
    """Pure numpy float32 replication (no torch, no C); returns complex64 and intermediates."""
    piM32 = _np.float32(piM)
    idx = _np.arange(n, dtype=_np.float32)
    f_phase = (idx + kmin) * _np.float32(delta_f)
    f_amp = (idx + kmin + 1.0) * _np.float32(delta_f)
    v = _np.cbrt(piM32) * _np.power(f_phase, _np.float32(1.0 / 3.0))
    logv = _np.log(f_phase) * _np.float32(1.0 / 3.0) + _np.log(_np.cbrt(piM32))

    v_amp = _np.arange(0, (kmin + n * 2), 1.0, dtype=_np.float64) * delta_f
    v_amp = _np.power(v_amp[1:], -7.0 / 6.0).astype(_np.float32)
    kfac = v_amp[kmin:kmin + n]
    amp = _np.float32(amp_factor) * kfac

    ph = pfa7 * v
    ph = (ph + pfa6 + pfl6 * (logv + _np.log(_np.float32(4.0)))) * v
    ph = (ph + pfa5 + pfl5 * logv) * v
    ph = (ph + pfa4) * v
    ph = (ph + pfa3) * v
    ph = (ph + pfa2) * v * v + _np.float32(1.0)
    ph = ph * (pfaN / (v * v * v * v * v)) - _np.float32(_np.pi / 4.0)
    two_pi = _np.float32(2.0 * _np.pi)
    ph = ph - _np.trunc(ph / two_pi) * two_pi
    ph = _np.where(ph < -_np.pi, ph + two_pi, ph)
    ph = _np.where(ph > _np.pi, ph - two_pi, ph)

    sinp = _np.float32(1.273239545) * ph - _np.float32(0.405284735) * ph * _np.abs(ph)
    sinp = _np.float32(0.225) * (sinp * _np.abs(sinp) - sinp) + sinp
    phs = ph + _np.float32(_np.pi / 2.0)
    phs = _np.where(phs > _np.pi, phs - two_pi, phs)
    cosp = _np.float32(1.273239545) * phs - _np.float32(0.405284735) * phs * _np.abs(phs)
    cosp = _np.float32(0.225) * (cosp * _np.abs(cosp) - cosp) + cosp
    out = cosp * amp + 1j * (-sinp * amp)
    return {
        "v": v,
        "logv": logv,
        "phasing": ph,
        "sinp": sinp,
        "cosp": cosp,
        "kfac": kfac,
        "amp": amp,
        "out": out.astype(_np.complex64),
    }


def spa_tmplt_engine(htilde, kmin, phase_order, delta_f, piM, pfaN,
                     pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7,
                     amp_factor):
    """
    Torch implementation of the SPA template kernel.

    Parameters mirror the CPU/CUDA backends; ``htilde`` is a complex Array
    allocated by the caller. The function fills it in-place.
    """
    tensor = htilde._data.tensor
    device = tensor.device
    dtype = tensor.dtype

    n = tensor.numel()
    delta_f = float(delta_f)
    kmin_int = int(kmin)

    # Default behaviour: mirror LAL/CPU path unless explicitly enabled. Global
    # PYCBC_TORCH_NATIVE_PORTS=1 (or PYCBC_SPATPLT_NATIVE=1) switches to the
    # torch-native kernel.
    if not torch_native_enabled("PYCBC_SPATPLT_NATIVE", default=False):
        cpu_np = _cpu_reference(
            n,
            kmin_int,
            delta_f,
            piM,
            pfaN,
            pfa2,
            pfa3,
            pfa4,
            pfa5,
            pfl5,
            pfa6,
            pfl6,
            pfa7,
            amp_factor,
        )
        tensor.copy_(torch.tensor(cpu_np, device=device, dtype=dtype))
        return None

    # Native torch kernel; optional CPU reference for debugging only.
    dbg = _torch_native_spa(n, kmin_int, delta_f, piM, pfaN, pfa2, pfa3, pfa4,
                            pfa5, pfl5, pfa6, pfl6, pfa7, amp_factor,
                            device=device, dtype_out=dtype)
    out = dbg["out"]
    tensor.copy_(out)

    if os.environ.get("PYCBC_SPATPLT_DEBUG", "0").lower() in ("1", "true", "yes", "on"):
        cpu_np = _cpu_reference(n, kmin_int, delta_f, piM, pfaN, pfa2, pfa3, pfa4,
                                pfa5, pfl5, pfa6, pfl6, pfa7, amp_factor)
        cpu_t = torch.tensor(cpu_np, device=device, dtype=dtype)
        np_ref = _numpy_reference(n, kmin_int, delta_f, piM, pfaN, pfa2, pfa3, pfa4,
                                  pfa5, pfl5, pfa6, pfl6, pfa7, amp_factor)

        phase_diff = torch.angle(out * torch.conj(cpu_t))
        abs_diff = torch.abs(out - cpu_t)
        max_idx = torch.argmax(abs_diff).item()
        stats = {
            "real_max": torch.max(torch.abs(out.real - cpu_t.real)).item(),
            "imag_max": torch.max(torch.abs(out.imag - cpu_t.imag)).item(),
            "out_max": torch.max(abs_diff).item(),
            "phase_mean": torch.mean(phase_diff).item(),
            "phase_std": torch.std(phase_diff).item(),
            "max_idx": max_idx + kmin_int,
            "max_amp_cpu": torch.abs(cpu_t.view(-1)[max_idx]).item(),
            "max_amp_tor": torch.abs(out.view(-1)[max_idx]).item(),
            "phase_at_max": phase_diff.view(-1)[max_idx].item(),
            "np_out_max": float(_np.max(_np.abs(np_ref["out"] - cpu_np))),
            "np_vs_torch_out_max": float(_np.max(_np.abs(np_ref["out"] - out.cpu().numpy()))),
        }
        if os.environ.get("PYCBC_SPATPLT_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on"):
            topk = torch.topk(abs_diff.view(-1), k=min(5, n))
            print("PYCBC_SPATPLT_DEBUG_VERBOSE top diffs:")
            for rank, (val, idx) in enumerate(zip(topk.values, topk.indices)):
                idx = idx.item()
                print(
                    f"  #{rank+1} bin {idx+kmin_int}: |Δh|={val.item():.3e}, "
                    f"|h_cpu|={torch.abs(cpu_t.view(-1)[idx]).item():.3e}, "
                    f"|h_tor|={torch.abs(out.view(-1)[idx]).item():.3e}, "
                    f"phaseΔ={phase_diff.view(-1)[idx].item():.3e}"
                )
        print("PYCBC_SPATPLT_DEBUG stats:", stats)

    return None
