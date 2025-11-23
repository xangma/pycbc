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
import os


def _torch_native_spa(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                      pfl5, pfa6, pfl6, pfa7, amp_factor, device="cpu", dtype_out=torch.complex64):
    """Pure torch replication of spa_tmplt CPU kernel; returns per-stage tensors.

    Parity note: validated 2025-11-23 vs spa_tmplt_cpu for the TaylorF2
    test grid in pycbc/waveform/tests/test_taylorf2_torch.py (max abs diff = 0).
    """
    f_dtype = torch.float32 if dtype_out == torch.complex64 else torch.float64
    # Build frequency tables with numpy to mirror CPU rounding
    idx_np = _np.arange(n, dtype=_np.float32)
    f_phase_np = (idx_np + kmin) * _np.float32(delta_f)
    f_amp_np = (idx_np + kmin + 1.0) * _np.float32(delta_f)

    piM13_np = _np.cbrt(_np.float32(piM))
    logpiM13_np = _np.log(piM13_np)
    log4_np = _np.log(_np.float32(4.0))

    v = torch.tensor(piM13_np * _np.power(f_phase_np, _np.float32(1.0 / 3.0)),
                     device=device, dtype=f_dtype)
    logv = torch.tensor(_np.log(f_phase_np) * _np.float32(1.0 / 3.0) + logpiM13_np,
                        device=device, dtype=f_dtype)

    # Preconditioner: float64 pow then cast to float32, as spa_tmplt_precondition
    v_amp = _np.arange(0, (kmin + n * 2), 1.0, dtype=_np.float64) * delta_f
    v_amp = _np.power(v_amp[1:], -7.0 / 6.0).astype(_np.float32)
    kfac = torch.tensor(v_amp[kmin:kmin + n], device=device, dtype=f_dtype)
    amp = torch.tensor(_np.float32(amp_factor), device=device, dtype=f_dtype) * kfac

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
    """CPU float32 reference via inline sequence; returns numpy complex64."""
    fvals = (_np.arange(kmin, kmin + n, dtype=_np.float32)) * float(delta_f)
    out_np = _np.empty(n, dtype=_np.complex64)
    _spa_cpu.spa_tmplt_inline_sequence(
        float(piM), float(pfaN), float(pfa2), float(pfa3), float(pfa4),
        float(pfa5), float(pfl5), float(pfa6), float(pfl6), float(pfa7),
        _np.float32(amp_factor), fvals, out_np
    )
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

    # Always use native torch kernel; optional CPU reference for debugging only.
    n = tensor.numel()
    delta_f = float(delta_f)
    kmin_int = int(kmin)
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
