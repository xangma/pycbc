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

import math

import numpy as _np
import torch

import pycbc.scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.torch_switches import torch_native_enabled


def _torch_native_spa(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                      pfl5, pfa6, pfl6, pfa7, amp_factor, device="cpu", dtype_out=torch.complex64):
    """Pure torch replication of spa_tmplt CPU kernel; returns per-stage tensors.

    Parity note: validated 2025-11-23 vs spa_tmplt_cpu for the TaylorF2
    test grid in test/waveform/test_taylorf2_torch.py (max abs diff = 0).
    """
    f_dtype = (
        torch.float64 if dtype_out == torch.complex128 else torch.float32
    )

    device = torch.device(device)
    # CPU and CUDA use float64 tables before casting, matching the historical
    # C/NumPy rounding. MPS has no float64 support, so evaluate there directly
    # in the float32 precision used by its complex output.
    table_dtype = (
        torch.float32 if device.type == "mps" else torch.float64
    )
    frequencies = (
        torch.arange(n, device=device, dtype=table_dtype) + kmin
    ) * delta_f
    pi_m_root = torch.pow(
        torch.tensor(piM, device=device, dtype=table_dtype), 1.0 / 3.0
    )
    log_pi_m_root = torch.log(pi_m_root)
    v = pi_m_root * torch.pow(frequencies, 1.0 / 3.0)
    safe_frequencies = torch.where(
        frequencies == 0.0, torch.ones_like(frequencies), frequencies
    )
    logv = torch.log(safe_frequencies) / 3.0 + log_pi_m_root
    # Match CPU lookup behavior at f=0 (logv_vec[0] == 0 -> logpiM13).
    logv = torch.where(frequencies == 0.0, log_pi_m_root, logv)
    if table_dtype != f_dtype:
        v = v.to(f_dtype)
        logv = logv.to(f_dtype)

    # Preconditioner: float64 pow then cast to float32, mirroring
    # spa_tmplt_precondition while keeping the frequency grid on-device.
    k_idx = torch.arange(
        kmin + 1, kmin + n + 1, device=device, dtype=table_dtype
    )
    kfac_t = torch.pow(k_idx * delta_f, -7.0 / 6.0)
    kfac = kfac_t.to(f_dtype) if f_dtype == torch.float32 else kfac_t

    amp = torch.tensor(amp_factor, device=device, dtype=f_dtype) * kfac

    two_pi = torch.tensor(2.0 * math.pi, dtype=f_dtype, device=device)
    log4 = torch.tensor(math.log(4.0), dtype=f_dtype, device=device)

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

    ph = ph * (pfaN_t / v5) - torch.tensor(
        math.pi / 4.0, dtype=f_dtype, device=device
    )
    ph = ph - torch.trunc(ph / two_pi) * two_pi
    ph = torch.where(ph < -math.pi, ph + two_pi, ph)
    ph = torch.where(ph > math.pi, ph - two_pi, ph)

    sinp = torch.tensor(1.273239545, device=device, dtype=f_dtype) * ph - torch.tensor(
        0.405284735, device=device, dtype=f_dtype
    ) * ph * torch.abs(ph)
    sinp = torch.tensor(0.225, device=device, dtype=f_dtype) * (sinp * torch.abs(sinp) - sinp) + sinp

    phs = ph + torch.tensor(math.pi / 2.0, device=device, dtype=f_dtype)
    phs = torch.where(phs > math.pi, phs - two_pi, phs)
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


def _torch_native_spa_sequence_frequencies(
    frequencies,
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
):
    """Match the legacy REAL4/COMPLEX8 arbitrary-frequency kernel."""
    device = frequencies.device
    table_dtype = torch.float32 if device.type == "mps" else torch.float64
    f32 = torch.float32
    frequency_table = frequencies.to(table_dtype)

    # Cython promotes each power/log operation to double, but stores the
    # result and every phase-polynomial stage back into a C float. Preserve
    # those rounding boundaries; they matter for long BNS inspiral phases.
    pi_m = torch.tensor(piM, dtype=f32, device=device)
    pi_m_root = torch.pow(pi_m.to(table_dtype), 1.0 / 3.0).to(f32)
    log_pi_m_root = torch.log(pi_m_root.to(table_dtype)).to(f32)
    v = (
        pi_m_root.to(table_dtype)
        * torch.pow(frequency_table, 1.0 / 3.0)
    ).to(f32)
    logv = (
        torch.log(frequency_table) / 3.0
        + log_pi_m_root.to(table_dtype)
    ).to(f32)
    amp = (
        torch.tensor(amp_factor, dtype=f32, device=device).to(table_dtype)
        * torch.pow(frequency_table, -7.0 / 6.0)
    ).to(f32)

    def coefficient(value):
        return torch.tensor(value, dtype=f32, device=device)

    phase = coefficient(pfa7) * v
    phase = (
        phase
        + coefficient(pfa6)
        + coefficient(pfl6) * (logv + coefficient(math.log(4.0)))
    ) * v
    phase = (
        phase + coefficient(pfa5) + coefficient(pfl5) * logv
    ) * v
    phase = (phase + coefficient(pfa4)) * v
    phase = (phase + coefficient(pfa3)) * v
    phase = (phase + coefficient(pfa2)) * v * v + 1.0
    v5 = v * v * v * v * v
    phase = phase * coefficient(pfaN) / v5
    phase = (phase.to(table_dtype) - math.pi / 4.0).to(f32)

    two_pi = coefficient(2.0 * math.pi)
    phase = phase - torch.trunc(phase / two_pi) * two_pi
    phase = torch.where(phase < -math.pi, phase + two_pi, phase)
    phase = torch.where(phase > math.pi, phase - two_pi, phase)

    # The inferred Cython sin/cos temporaries are doubles. MPS has no float64
    # support, so it uses the output precision for this small approximation.
    trig_dtype = torch.float32 if device.type == "mps" else torch.float64
    trig_phase = phase.to(trig_dtype)
    sinp = 1.273239545 * trig_phase - 0.405284735 * trig_phase.abs() * trig_phase
    sinp = 0.225 * (sinp * sinp.abs() - sinp) + sinp

    cosine_phase = (phase.to(table_dtype) + math.pi / 2.0).to(f32)
    cosine_phase = torch.where(
        cosine_phase > math.pi,
        cosine_phase - two_pi,
        cosine_phase,
    ).to(trig_dtype)
    cosp = (
        1.273239545 * cosine_phase
        - 0.405284735 * cosine_phase.abs() * cosine_phase
    )
    cosp = 0.225 * (cosp * cosp.abs() - cosp) + cosp

    real = (cosp * amp.to(trig_dtype)).to(f32)
    imag = (-sinp * amp.to(trig_dtype)).to(f32)
    return torch.complex(real, imag)


def spa_tmplt_sequence(
    sample_points,
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
):
    """Evaluate SPAtmplt at arbitrary frequencies on the active Torch device."""
    device = _scheme.mgr.state.torch_device
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor

    # The legacy Cython sequence API accepts REAL4 samples and returns
    # COMPLEX8 values. Round to that public precision before using float64
    # lookup arithmetic on devices which support it.
    frequencies = torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("SPAtmplt sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SPAtmplt sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SPAtmplt sample_points must be positive")

    result = _torch_native_spa_sequence_frequencies(
        frequencies,
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
    return PyCBCArray(TorchArrayData(result), copy=False)


def _cpu_reference(n, kmin, delta_f, piM, pfaN, pfa2, pfa3, pfa4, pfa5,
                   pfl5, pfa6, pfl6, pfa7, amp_factor):
    """CPU float32 reference matching ``spa_tmplt_engine`` exactly."""
    # This diagnostic reference uses the historical Cython extension, which
    # imports core LAL.  Keep it out of the production Torch engine's import
    # closure.
    from pycbc.waveform import spa_tmplt_cpu as _spa_cpu

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

    # SPAtmplt is mature enough to run natively by default under TorchScheme.
    # Global or component ``=0`` overrides retain the legacy CPU fallback.
    if not torch_native_enabled("PYCBC_SPATPLT_NATIVE", default=True):
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

    out = _torch_native_spa(
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
        device=device,
        dtype_out=dtype,
    )["out"]
    tensor.copy_(out)

    return None
