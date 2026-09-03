# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Fused single-launch Triton kernel for IMRPhenomXAS waveform generation."""

from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _imrphenomxas_fused_kernel(
        hp_real_ptr,
        hp_imag_ptr,
        hc_real_ptr,
        hc_imag_ptr,
        delta_f,
        f_lower,
        packed_plan_ptr,
        incl_ptr,
        coa_phase_ptr,
        long_asc_nodes_ptr,
        stride_b_plan,
        stride_out_b,
        stride_out_f,
        n_freqs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused IMRPhenomXAS evaluation kernel with 17-term Horner inspiral phase in registers."""
        pid_f = tl.program_id(0)
        pid_b = tl.program_id(1)

        f_idx = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = f_idx < n_freqs

        plan_offset = pid_b * stride_b_plan

        total_mass_seconds = tl.load(packed_plan_ptr + plan_offset + 0)
        eta = tl.load(packed_plan_ptr + plan_offset + 1)
        phase_lower = tl.load(packed_plan_ptr + plan_offset + 2)
        phase_upper = tl.load(packed_plan_ptr + plan_offset + 3)

        # 16 inspiral phase coefficients
        phi0 = tl.load(packed_plan_ptr + plan_offset + 4)
        phi1 = tl.load(packed_plan_ptr + plan_offset + 5)
        phi2 = tl.load(packed_plan_ptr + plan_offset + 6)
        phi3 = tl.load(packed_plan_ptr + plan_offset + 7)
        phi4 = tl.load(packed_plan_ptr + plan_offset + 8)
        phi5 = tl.load(packed_plan_ptr + plan_offset + 9)
        phi5_l = tl.load(packed_plan_ptr + plan_offset + 10)
        phi6 = tl.load(packed_plan_ptr + plan_offset + 11)
        phi6_l = tl.load(packed_plan_ptr + plan_offset + 12)
        phi7 = tl.load(packed_plan_ptr + plan_offset + 13)
        phi8 = tl.load(packed_plan_ptr + plan_offset + 14)
        phi8_l = tl.load(packed_plan_ptr + plan_offset + 15)
        sigma1 = tl.load(packed_plan_ptr + plan_offset + 16)
        sigma2 = tl.load(packed_plan_ptr + plan_offset + 17)
        sigma3 = tl.load(packed_plan_ptr + plan_offset + 18)
        sigma4 = tl.load(packed_plan_ptr + plan_offset + 19)

        # 8 intermediate phase coefficients
        b0 = tl.load(packed_plan_ptr + plan_offset + 20)
        b1 = tl.load(packed_plan_ptr + plan_offset + 21)
        b2 = tl.load(packed_plan_ptr + plan_offset + 22)
        b3 = tl.load(packed_plan_ptr + plan_offset + 23)
        b4 = tl.load(packed_plan_ptr + plan_offset + 24)
        c_l_int = tl.load(packed_plan_ptr + plan_offset + 25)
        f_rd_int = tl.load(packed_plan_ptr + plan_offset + 26)
        f_damp_int = tl.load(packed_plan_ptr + plan_offset + 27)

        # 9 ringdown phase coefficients
        c0 = tl.load(packed_plan_ptr + plan_offset + 28)
        c1 = tl.load(packed_plan_ptr + plan_offset + 29)
        c2 = tl.load(packed_plan_ptr + plan_offset + 30)
        c4_over_3 = tl.load(packed_plan_ptr + plan_offset + 31)
        c_l_over_f_damp = tl.load(packed_plan_ptr + plan_offset + 32)
        f_rd_rd = tl.load(packed_plan_ptr + plan_offset + 33)
        f_damp_rd = tl.load(packed_plan_ptr + plan_offset + 34)

        # 4 reconnection phase coefficients
        alpha0 = tl.load(packed_plan_ptr + plan_offset + 37)
        alpha1 = tl.load(packed_plan_ptr + plan_offset + 38)
        beta0 = tl.load(packed_plan_ptr + plan_offset + 39)
        beta1 = tl.load(packed_plan_ptr + plan_offset + 40)

        # 9 inspiral amplitude coefficients
        a0 = tl.load(packed_plan_ptr + plan_offset + 41)
        a2 = tl.load(packed_plan_ptr + plan_offset + 42)
        a3 = tl.load(packed_plan_ptr + plan_offset + 43)
        a4 = tl.load(packed_plan_ptr + plan_offset + 44)
        a5 = tl.load(packed_plan_ptr + plan_offset + 45)
        a6 = tl.load(packed_plan_ptr + plan_offset + 46)
        rho1 = tl.load(packed_plan_ptr + plan_offset + 47)
        rho2 = tl.load(packed_plan_ptr + plan_offset + 48)
        rho3 = tl.load(packed_plan_ptr + plan_offset + 49)

        # 5 intermediate amplitude coefficients
        delta0 = tl.load(packed_plan_ptr + plan_offset + 50)
        delta1 = tl.load(packed_plan_ptr + plan_offset + 51)
        delta2 = tl.load(packed_plan_ptr + plan_offset + 52)
        delta3 = tl.load(packed_plan_ptr + plan_offset + 53)
        delta4 = tl.load(packed_plan_ptr + plan_offset + 54)

        # 5 ringdown amplitude coefficients
        f_rd_amp = tl.load(packed_plan_ptr + plan_offset + 55)
        gamma_r = tl.load(packed_plan_ptr + plan_offset + 56)
        gamma_d2 = tl.load(packed_plan_ptr + plan_offset + 57)
        gamma_d13 = tl.load(packed_plan_ptr + plan_offset + 58)
        amp_upper = tl.load(packed_plan_ptr + plan_offset + 59)

        # Linear alignment, reference, time shift, overall amplitude, and match frequency
        linear_a = tl.load(packed_plan_ptr + plan_offset + 60)
        linear_b = tl.load(packed_plan_ptr + plan_offset + 61)
        phase_at_reference = tl.load(packed_plan_ptr + plan_offset + 62)
        time_shift = tl.load(packed_plan_ptr + plan_offset + 63)
        overall_amp = tl.load(packed_plan_ptr + plan_offset + 64)
        amp_match = tl.load(packed_plan_ptr + plan_offset + 65)

        incl = tl.load(incl_ptr + pid_b)
        long_asc_nodes = tl.load(long_asc_nodes_ptr + pid_b)

        PI = 3.14159265358979323846
        NORM_PHIN = -0.034789526860125735  # -(3.0 * PI^(-5/3)) / 128.0

        freqs = f_lower + f_idx.to(tl.float64) * delta_f
        Mf = freqs * total_mass_seconds
        Mf_safe = tl.where(Mf > 0.0, Mf, 1e-12)

        v = tl.libdevice.pow(Mf_safe, 1.0 / 3.0)
        log_Mf = tl.libdevice.log(Mf_safe)

        # 17-term Horner Inspirial Phase Evaluation in Registers
        c11 = sigma4
        c10 = sigma3
        c9 = sigma2
        c8 = phi8 + phi8_l * log_Mf + sigma1
        c7 = phi7
        c6 = phi6 + phi6_l * log_Mf
        c5 = phi5 + phi5_l * log_Mf
        c4 = phi4
        c3 = phi3
        c2 = phi2
        c1 = phi1
        c0 = phi0

        p_ins = c11
        p_ins = p_ins * v + c10
        p_ins = p_ins * v + c9
        p_ins = p_ins * v + c8
        p_ins = p_ins * v + c7
        p_ins = p_ins * v + c6
        p_ins = p_ins * v + c5
        p_ins = p_ins * v + c4
        p_ins = p_ins * v + c3
        p_ins = p_ins * v + c2
        p_ins = p_ins * v + c1
        p_ins = p_ins * v + c0

        v5 = v * v * v * v * v
        phi_ins = p_ins * NORM_PHIN / v5

        # Intermediate Phase
        phi_int = (
            b0 * Mf_safe
            + b1 * log_Mf
            - b2 / Mf_safe
            - b3 / (2.0 * Mf_safe * Mf_safe)
            - b4 / (3.0 * Mf_safe * Mf_safe * Mf_safe)
            + (2.0 * c_l_int * tl.libdevice.atan((Mf_safe - f_rd_int) / (2.0 * f_damp_int))) / f_damp_int
            + alpha1 * Mf_safe
            + alpha0
        )

        # Ringdown Phase
        phi_rd = (
            c0 * Mf_safe
            + 1.5 * c1 * (v * v)
            - c2 / Mf_safe
            - c4_over_3 / (Mf_safe * Mf_safe * Mf_safe)
            + c_l_over_f_damp * tl.libdevice.atan((Mf_safe - f_rd_rd) / f_damp_rd)
            + beta0
            + beta1 * Mf_safe
        )

        phase_val = tl.where(
            Mf_safe < phase_lower,
            phi_ins,
            tl.where(Mf_safe >= phase_upper, phi_rd, phi_int),
        )
        extrinsic_phase = 2.0 * PI * freqs * time_shift
        total_phi = (
            (1.0 / eta) * phase_val
            + (linear_b * Mf_safe)
            + linear_a
            + phase_at_reference
            - (2.0 * PI)
            + extrinsic_phase
        )

        # Inspiral Amplitude
        amp_ins = (
            a0
            + a2 * (v * v)
            + a3 * Mf_safe
            + a4 * (v * Mf_safe)
            + a5 * (v * v * Mf_safe)
            + a6 * (Mf_safe * Mf_safe)
            + rho1 * (v * Mf_safe * Mf_safe)
            + rho2 * (v * v * Mf_safe * Mf_safe)
            + rho3 * (Mf_safe * Mf_safe * Mf_safe)
        )

        # Intermediate Amplitude
        denom_int = delta0 + Mf_safe * (
            delta1 + Mf_safe * (delta2 + Mf_safe * (delta3 + Mf_safe * delta4))
        )
        amp_int = tl.libdevice.pow(Mf_safe, 7.0 / 6.0) / denom_int

        # Ringdown Amplitude
        diff_rd = Mf_safe - f_rd_amp
        amp_rd = (
            tl.libdevice.exp(-diff_rd * gamma_r)
            * gamma_d13
            / (diff_rd * diff_rd + gamma_d2)
        )

        amp_val = tl.where(
            Mf_safe < amp_match,
            amp_ins,
            tl.where(Mf_safe >= amp_upper, amp_rd, amp_int),
        )
        amp_final = overall_amp * amp_val * tl.libdevice.pow(Mf_safe, -7.0 / 6.0)

        cos_i = tl.libdevice.cos(incl)
        cos_p = tl.libdevice.cos(total_phi)
        sin_p = tl.libdevice.sin(total_phi)

        s_r = amp_final * cos_p
        s_i = amp_final * sin_p

        plus0_r = -0.5 * (1.0 + cos_i * cos_i) * s_r
        plus0_i = -0.5 * (1.0 + cos_i * cos_i) * s_i
        cross0_r = -cos_i * s_i
        cross0_i = cos_i * s_r

        cos_2nodes = tl.libdevice.cos(2.0 * long_asc_nodes)
        sin_2nodes = tl.libdevice.sin(2.0 * long_asc_nodes)

        hp_r = cos_2nodes * plus0_r + sin_2nodes * cross0_r
        hp_i = cos_2nodes * plus0_i + sin_2nodes * cross0_i
        hc_r = cos_2nodes * cross0_r - sin_2nodes * plus0_r
        hc_i = cos_2nodes * cross0_i - sin_2nodes * plus0_i

        f_cut_hz = 0.3 / total_mass_seconds
        valid = (freqs <= f_cut_hz) & mask

        hp_r = tl.where(valid, hp_r, 0.0)
        hp_i = tl.where(valid, hp_i, 0.0)
        hc_r = tl.where(valid, hc_r, 0.0)
        hc_i = tl.where(valid, hc_i, 0.0)

        out_offset = pid_b * stride_out_b + f_idx * stride_out_f
        tl.store(hp_real_ptr + out_offset, hp_r, mask=mask)
        tl.store(hp_imag_ptr + out_offset, hp_i, mask=mask)
        tl.store(hc_real_ptr + out_offset, hc_r, mask=mask)
        tl.store(hc_imag_ptr + out_offset, hc_i, mask=mask)
else:
    _imrphenomxas_fused_kernel = None


def is_triton_available() -> bool:
    """Return whether Triton is installed and supported."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _prepare_triton_imrphenomxas_buffers(params, device, real_dtype, batch_size):
    """Assemble 66-scalar packed plan tensors for the fused Triton kernel."""
    from pycbc.waveform.imrphenomxas_torch import (
        _build_packed_frequency_plan,
        IMRPhenomX_utils,
    )

    def _to_tensor(val, default=0.0):
        if val is None:
            val = default
        if isinstance(val, torch.Tensor):
            t = val.to(device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            return t
        elif isinstance(val, (list, tuple)):
            return torch.as_tensor(val, device=device, dtype=real_dtype)
        else:
            return torch.full((batch_size,), float(val), device=device, dtype=real_dtype)

    m1 = _to_tensor(params["mass1"])
    m2 = _to_tensor(params["mass2"])
    s1z = _to_tensor(params.get("spin1z", 0.0))
    s2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))

    phase_coeffs = IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
        device=device, dtype=real_dtype
    )
    amp_coeffs = IMRPhenomX_utils._get_phenomx_amp_coeff_table_cached_master(
        device=device, dtype=real_dtype
    )

    plans = []
    for b in range(batch_size):
        m1_b = m1[b]
        m2_b = m2[b]
        s1z_b = s1z[b]
        s2z_b = s2z[b]
        if m2_b > m1_b:
            m1_b, m2_b = m2_b, m1_b
            s1z_b, s2z_b = s2z_b, s1z_b

        theta_intrinsic = torch.stack([m1_b, m2_b, s1z_b, s2z_b])
        theta_extrinsic = torch.stack([dist[b], torch.tensor(0.0, device=device, dtype=real_dtype), coa_phase[b]])
        f_ref_b = f_ref[b]
        if f_ref_b <= 0.0:
            f_ref_b = float(params["f_lower"])

        plan = _build_packed_frequency_plan(
            theta_intrinsic,
            theta_extrinsic,
            phase_coeffs,
            amp_coeffs,
            f_ref_b,
        )
        plans.append(plan)

    packed_plans = torch.stack(plans, dim=0)

    return {
        "packed_plan": packed_plans,
        "incl": incl,
        "coa_phase": coa_phase,
        "long_asc_nodes": long_asc_nodes,
        "m1": m1,
        "m2": m2,
    }


def imrphenomxas_triton_fd(**params):
    """Generate single-call IMRPhenomXAS waveform using fused Triton kernel if available."""
    if not is_triton_available():
        from pycbc.waveform.imrphenomxas_torch import imrphenomxas_fd_torch
        return imrphenomxas_fd_torch(**params)

    device = torch.device("cuda")
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    buffers = _prepare_triton_imrphenomxas_buffers(params, device, real_dtype, batch_size=1)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    from pycbc.waveform.imrphenomxas_torch import _next_power_of_two, IMRPhenomX_utils
    total_mass_seconds = (float(buffers["m1"][0]) + float(buffers["m2"][0])) * 4.925490947641267e-06
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    npts = _next_power_of_two(layout_f_max / delta_f) + 1

    first_bin = int(math.floor(f_lower / delta_f))
    stop_bin = int(math.floor(active_f_max / delta_f))
    n_freqs = stop_bin - first_bin

    hp_real = torch.zeros(npts, dtype=real_dtype, device=device)
    hp_imag = torch.zeros(npts, dtype=real_dtype, device=device)
    hc_real = torch.zeros(npts, dtype=real_dtype, device=device)
    hc_imag = torch.zeros(npts, dtype=real_dtype, device=device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_freqs, BLOCK_SIZE), 1)

    _imrphenomxas_fused_kernel[grid](
        hp_real_ptr=hp_real[first_bin:],
        hp_imag_ptr=hp_imag[first_bin:],
        hc_real_ptr=hc_real[first_bin:],
        hc_imag_ptr=hc_imag[first_bin:],
        delta_f=delta_f,
        f_lower=f_lower,
        packed_plan_ptr=buffers["packed_plan"],
        incl_ptr=buffers["incl"],
        coa_phase_ptr=buffers["coa_phase"],
        long_asc_nodes_ptr=buffers["long_asc_nodes"],
        stride_b_plan=buffers["packed_plan"].stride(0),
        stride_out_b=0,
        stride_out_f=1,
        n_freqs=n_freqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    hp = torch.complex(hp_real, hp_imag).to(complex_dtype)
    hc = torch.complex(hc_real, hc_imag).to(complex_dtype)

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData

    epoch = -1.0 / delta_f
    hp_fs = FrequencySeries(TorchArrayData(hp), delta_f=delta_f, epoch=epoch, copy=False)
    hc_fs = FrequencySeries(TorchArrayData(hc), delta_f=delta_f, epoch=epoch, copy=False)
    return hp_fs, hc_fs


def imrphenomxas_triton_fd_batch(**params):
    """Generate batch of IMRPhenomXAS waveforms using fused Triton kernel if available."""
    if not is_triton_available():
        from pycbc.waveform.imrphenomxas_torch import imrphenomxas_fd_torch
        # When called with batch parameters without Triton, evaluate iteratively or via torch
        batch_size = 1
        for k in ("mass1", "mass2", "spin1z", "spin2z", "distance", "inclination", "coa_phase", "long_asc_nodes", "f_ref"):
            v = params.get(k)
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                batch_size = max(batch_size, v.shape[0])
            elif isinstance(v, (list, tuple)) and len(v) > 1:
                batch_size = max(batch_size, len(v))

        hp_list = []
        hc_list = []
        for b in range(batch_size):
            p_single = dict(params)
            for k in ("mass1", "mass2", "spin1z", "spin2z", "distance", "inclination", "coa_phase", "long_asc_nodes", "f_ref"):
                v = params.get(k)
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    p_single[k] = float(v[b].item())
                elif isinstance(v, (list, tuple)) and len(v) > 1:
                    p_single[k] = float(v[b])
            hp_s, hc_s = imrphenomxas_fd_torch(**p_single)
            hp_list.append(torch.as_tensor(hp_s._data.tensor))
            hc_list.append(torch.as_tensor(hc_s._data.tensor))
        return torch.stack(hp_list, dim=0), torch.stack(hc_list, dim=0)

    device = torch.device("cuda")
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    batch_size = 1
    for k in (
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
        "distance",
        "inclination",
        "coa_phase",
        "long_asc_nodes",
        "f_ref",
    ):
        v = params.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            batch_size = max(batch_size, v.shape[0])
        elif isinstance(v, (list, tuple)) and len(v) > 1:
            batch_size = max(batch_size, len(v))

    buffers = _prepare_triton_imrphenomxas_buffers(params, device, real_dtype, batch_size=batch_size)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    from pycbc.waveform.imrphenomxas_torch import _next_power_of_two, IMRPhenomX_utils
    total_mass_seconds = (buffers["m1"] + buffers["m2"]) * 4.925490947641267e-06
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / total_mass_seconds
    if f_final > 0.0:
        layout_f_max = f_final
    else:
        layout_f_max = float(torch.max(cutoff_frequency).item())

    npts = _next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(math.floor(f_lower / delta_f))
    max_stop_bin = int(math.floor(layout_f_max / delta_f))
    n_freqs = max_stop_bin - first_bin

    hp_real = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hp_imag = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hc_real = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hc_imag = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_freqs, BLOCK_SIZE), batch_size)

    _imrphenomxas_fused_kernel[grid](
        hp_real_ptr=hp_real[:, first_bin:],
        hp_imag_ptr=hp_imag[:, first_bin:],
        hc_real_ptr=hc_real[:, first_bin:],
        hc_imag_ptr=hc_imag[:, first_bin:],
        delta_f=delta_f,
        f_lower=f_lower,
        packed_plan_ptr=buffers["packed_plan"],
        incl_ptr=buffers["incl"],
        coa_phase_ptr=buffers["coa_phase"],
        long_asc_nodes_ptr=buffers["long_asc_nodes"],
        stride_b_plan=buffers["packed_plan"].stride(0),
        stride_out_b=hp_real.stride(0),
        stride_out_f=hp_real.stride(1),
        n_freqs=n_freqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    hp = torch.complex(hp_real, hp_imag).to(complex_dtype)
    hc = torch.complex(hc_real, hc_imag).to(complex_dtype)
    return hp, hc
