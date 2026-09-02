# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Fused single-launch Triton kernel for TaylorF2 waveform generation."""

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
    def _taylorf2_fused_kernel(
        hp_real_ptr,
        hp_imag_ptr,
        hc_real_ptr,
        hc_imag_ptr,
        delta_f,
        f_lower,
        m1_ptr,
        m2_ptr,
        dist_ptr,
        incl_ptr,
        coa_phase_ptr,
        v_coeffs_ptr,
        vlogv_coeffs_ptr,
        vlogvsq_coeffs_ptr,
        stride_b_coeffs,
        stride_out_b,
        stride_out_f,
        n_freqs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused TaylorF2 evaluation kernel over frequency blocks and batch items."""
        pid_f = tl.program_id(0)
        pid_b = tl.program_id(1)

        f_idx = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = f_idx < n_freqs

        m1 = tl.load(m1_ptr + pid_b)
        m2 = tl.load(m2_ptr + pid_b)
        distance = tl.load(dist_ptr + pid_b)
        inclination = tl.load(incl_ptr + pid_b)
        coa_phase = tl.load(coa_phase_ptr + pid_b)

        MTSUN_SI = 4.925490947641267e-06
        MRSUN_SI = 1.476625038050125e03
        PC_SI = 3.08567758149137e16
        PI = 3.14159265358979323846

        total_mass = m1 + m2
        eta = (m1 * m2) / (total_mass * total_mass)
        pi_mass = PI * total_mass * MTSUN_SI

        freqs = f_lower + f_idx.to(tl.float64) * delta_f
        v = tl.libdevice.pow(pi_mass * freqs, 1.0 / 3.0)
        log_v = tl.libdevice.log(v)
        log_v_sq = log_v * log_v

        coeff_offset = pid_b * stride_b_coeffs

        # Fused Horner polynomial evaluation
        c_v15 = tl.load(v_coeffs_ptr + coeff_offset + 15)
        c_log15 = tl.load(vlogv_coeffs_ptr + coeff_offset + 15)
        c_logsq15 = tl.load(vlogvsq_coeffs_ptr + coeff_offset + 15)
        res = c_v15 + c_log15 * log_v + c_logsq15 * log_v_sq

        for k in range(14, -1, -1):
            c_v = tl.load(v_coeffs_ptr + coeff_offset + k)
            c_log = tl.load(vlogv_coeffs_ptr + coeff_offset + k)
            c_logsq = tl.load(vlogvsq_coeffs_ptr + coeff_offset + k)
            term = c_v + c_log * log_v + c_logsq * log_v_sq
            res = res * v + term

        v5 = v * v * v * v * v
        phi = res / v5

        epoch = -1.0 / delta_f
        total_phase = phi + (2.0 * PI * epoch * freqs) - (2.0 * coa_phase) - (PI / 4.0)

        dist_m = distance * 1.0e6 * PC_SI
        amp0 = -4.0 * m1 * m2 / dist_m * MRSUN_SI * MTSUN_SI * 0.5123496421589149
        amp_factor = amp0 * tl.libdevice.sqrt(5.0 / (32.0 * eta))
        v_neg_3_5 = tl.libdevice.pow(v, -3.5)
        amplitude = amp_factor * v_neg_3_5

        cos_i = tl.libdevice.cos(inclination)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = -cos_i

        cos_phase = tl.libdevice.cos(total_phase)
        sin_phase = tl.libdevice.sin(total_phase)

        raw_r = amplitude * cos_phase
        raw_i = -amplitude * sin_phase

        hp_r = raw_r * plus_factor
        hp_i = raw_i * plus_factor
        hc_r = -raw_i * cross_factor
        hc_i = raw_r * cross_factor

        out_offset = pid_b * stride_out_b + f_idx * stride_out_f
        tl.store(hp_real_ptr + out_offset, hp_r, mask=mask)
        tl.store(hp_imag_ptr + out_offset, hp_i, mask=mask)
        tl.store(hc_real_ptr + out_offset, hc_r, mask=mask)
        tl.store(hc_imag_ptr + out_offset, hc_i, mask=mask)
else:
    _taylorf2_fused_kernel = None


def is_triton_available() -> bool:
    """Return whether Triton is installed and supported."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()
