# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Fused single-launch Triton kernel for IMRPhenomD waveform generation."""

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
    def _imrphenomd_fused_kernel(
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
        long_asc_nodes_ptr,
        amp_params_ptr,
        phase_params_ptr,
        pn_coeffs_ptr,
        stride_b_amp,
        stride_b_phase,
        stride_b_pn,
        stride_out_b,
        stride_out_f,
        n_freqs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused IMRPhenomD evaluation kernel over frequency blocks and batch items."""
        pid_f = tl.program_id(0)
        pid_b = tl.program_id(1)

        f_idx = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = f_idx < n_freqs

        m1 = tl.load(m1_ptr + pid_b)
        m2 = tl.load(m2_ptr + pid_b)
        distance = tl.load(dist_ptr + pid_b)
        inclination = tl.load(incl_ptr + pid_b)
        long_asc_nodes = tl.load(long_asc_nodes_ptr + pid_b)

        MTSUN_SI = 4.925490947641267e-06
        MRSUN_SI = 1.476625038050125e03
        PC_SI = 3.08567758149137e16

        total_mass = m1 + m2
        total_mass_seconds = total_mass * MTSUN_SI
        eta = (m1 * m2) / (total_mass * total_mass)
        eta_inv = 1.0 / eta

        freqs = f_lower + f_idx.to(tl.float64) * delta_f
        Mf = freqs * total_mass_seconds
        Mf_safe = tl.where(Mf > 0.0, Mf, 1e-12)

        # Load amplitude params for batch item pid_b
        amp_offset = pid_b * stride_b_amp
        fmaxCalc = tl.load(amp_params_ptr + amp_offset + 0)
        fRD_amp = tl.load(amp_params_ptr + amp_offset + 1)
        fDM_amp = tl.load(amp_params_ptr + amp_offset + 2)
        gamma1 = tl.load(amp_params_ptr + amp_offset + 3)
        gamma2 = tl.load(amp_params_ptr + amp_offset + 4)
        gamma3 = tl.load(amp_params_ptr + amp_offset + 5)
        amp0 = tl.load(amp_params_ptr + amp_offset + 6)
        delta0 = tl.load(amp_params_ptr + amp_offset + 7)
        delta1 = tl.load(amp_params_ptr + amp_offset + 8)
        delta2 = tl.load(amp_params_ptr + amp_offset + 9)
        delta3 = tl.load(amp_params_ptr + amp_offset + 10)
        delta4 = tl.load(amp_params_ptr + amp_offset + 11)
        pref_two_thirds = tl.load(amp_params_ptr + amp_offset + 12)
        pref_four_thirds = tl.load(amp_params_ptr + amp_offset + 13)
        pref_five_thirds = tl.load(amp_params_ptr + amp_offset + 14)
        pref_seven_thirds = tl.load(amp_params_ptr + amp_offset + 15)
        pref_eight_thirds = tl.load(amp_params_ptr + amp_offset + 16)
        pref_one = tl.load(amp_params_ptr + amp_offset + 17)
        pref_two = tl.load(amp_params_ptr + amp_offset + 18)
        pref_three = tl.load(amp_params_ptr + amp_offset + 19)

        # Load phase params
        phase_offset_b = pid_b * stride_b_phase
        fInsJoin = tl.load(phase_params_ptr + phase_offset_b + 0)
        fMRDJoin = tl.load(phase_params_ptr + phase_offset_b + 1)
        beta1 = tl.load(phase_params_ptr + phase_offset_b + 2)
        beta2 = tl.load(phase_params_ptr + phase_offset_b + 3)
        beta3 = tl.load(phase_params_ptr + phase_offset_b + 4)
        alpha1 = tl.load(phase_params_ptr + phase_offset_b + 5)
        alpha2 = tl.load(phase_params_ptr + phase_offset_b + 6)
        alpha3 = tl.load(phase_params_ptr + phase_offset_b + 7)
        alpha4 = tl.load(phase_params_ptr + phase_offset_b + 8)
        alpha5 = tl.load(phase_params_ptr + phase_offset_b + 9)
        fRD_phi = tl.load(phase_params_ptr + phase_offset_b + 10)
        fDM_phi = tl.load(phase_params_ptr + phase_offset_b + 11)
        C1Int = tl.load(phase_params_ptr + phase_offset_b + 12)
        C2Int = tl.load(phase_params_ptr + phase_offset_b + 13)
        C1MRD = tl.load(phase_params_ptr + phase_offset_b + 14)
        C2MRD = tl.load(phase_params_ptr + phase_offset_b + 15)
        time_shift = tl.load(phase_params_ptr + phase_offset_b + 16)
        mf_ref = tl.load(phase_params_ptr + phase_offset_b + 17)
        phase_offset = tl.load(phase_params_ptr + phase_offset_b + 18)

        # Load PN coefficients
        pn_offset = pid_b * stride_b_pn
        phi_initial = tl.load(pn_coeffs_ptr + pn_offset + 0)
        phi_two_thirds = tl.load(pn_coeffs_ptr + pn_offset + 1)
        phi_third = tl.load(pn_coeffs_ptr + pn_offset + 2)
        phi_third_logv = tl.load(pn_coeffs_ptr + pn_offset + 3)
        phi_logv = tl.load(pn_coeffs_ptr + pn_offset + 4)
        phi_minus_third = tl.load(pn_coeffs_ptr + pn_offset + 5)
        phi_minus_two_thirds = tl.load(pn_coeffs_ptr + pn_offset + 6)
        phi_minus_one = tl.load(pn_coeffs_ptr + pn_offset + 7)
        phi_minus_four_thirds = tl.load(pn_coeffs_ptr + pn_offset + 8)
        phi_minus_five_thirds = tl.load(pn_coeffs_ptr + pn_offset + 9)
        phi_one = tl.load(pn_coeffs_ptr + pn_offset + 10)
        phi_four_thirds = tl.load(pn_coeffs_ptr + pn_offset + 11)
        phi_five_thirds = tl.load(pn_coeffs_ptr + pn_offset + 12)
        phi_two = tl.load(pn_coeffs_ptr + pn_offset + 13)

        # Powers of Mf
        v = tl.libdevice.pow(Mf_safe, 1.0 / 3.0)
        v2 = v * v
        v3 = Mf_safe
        v4 = v3 * v
        v5 = v3 * v2
        v7 = v3 * v4
        v8 = v3 * v5
        m_v = 1.0 / v
        m_v2 = 1.0 / v2
        m_v4 = 1.0 / v4
        m_v5 = 1.0 / v5
        m_v7_6 = tl.libdevice.pow(Mf_safe, -7.0 / 6.0)
        log_Mf = tl.libdevice.log(Mf_safe)

        # Piecewise Amplitude:
        amp_ins_ansatz = (
            1.0
            + v2 * pref_two_thirds
            + v4 * pref_four_thirds
            + v5 * pref_five_thirds
            + v7 * pref_seven_thirds
            + v8 * pref_eight_thirds
            + Mf_safe * (pref_one + Mf_safe * pref_two + (Mf_safe * Mf_safe) * pref_three)
        )
        amp_ins = amp0 * m_v7_6 * amp_ins_ansatz

        fminfRD = Mf_safe - fRD_amp
        fDMgamma3 = fDM_amp * gamma3
        amp_mrd_ansatz = (
            tl.libdevice.exp(-fminfRD * gamma2 / fDMgamma3)
            * (fDMgamma3 * gamma1)
            / (fminfRD * fminfRD + fDMgamma3 * fDMgamma3)
        )
        amp_mrd = amp0 * m_v7_6 * amp_mrd_ansatz

        Mf2 = Mf_safe * Mf_safe
        amp_int_ansatz = (
            delta0
            + Mf_safe * delta1
            + Mf2 * (delta2 + Mf_safe * delta3 + delta4 * Mf2)
        )
        amp_int = amp0 * m_v7_6 * amp_int_ansatz

        AMP_fJoin_INS = 0.014
        amp = tl.where(
            Mf_safe < AMP_fJoin_INS,
            amp_ins,
            tl.where(Mf_safe >= fmaxCalc, amp_mrd, amp_int),
        )

        # Piecewise Phase:
        pow_pi_third = 1.4645918875615231
        v_pi = v * pow_pi_third
        logv = tl.libdevice.log(v_pi)
        phi_ins = (
            phi_initial
            + phi_two_thirds * v2
            + phi_third * v
            + phi_third_logv * logv * v
            + phi_logv * logv
            + phi_minus_third * m_v
            + phi_minus_two_thirds * m_v2
            + phi_minus_one * (1.0 / Mf_safe)
            + phi_minus_four_thirds * m_v4
            + phi_minus_five_thirds * m_v5
            + eta_inv
            * (
                phi_one * Mf_safe
                + phi_four_thirds * v4
                + phi_five_thirds * v5
                + phi_two * Mf2
            )
        )

        phi_int = (
            eta_inv
            * (
                beta1 * Mf_safe
                - beta3 / (3.0 * (Mf_safe * Mf2))
                + beta2 * log_Mf
            )
            + C1Int
            + C2Int * Mf_safe
        )

        sqroot_Mf = tl.libdevice.sqrt(Mf_safe)
        fpow0_75 = tl.libdevice.sqrt(Mf_safe * sqroot_Mf)
        phi_mrd_raw = (
            -(alpha2 / Mf_safe)
            + (4.0 / 3.0) * (alpha3 * fpow0_75)
            + alpha1 * Mf_safe
            + alpha4 * tl.libdevice.atan((Mf_safe - alpha5 * fRD_phi) / fDM_phi)
        )
        phi_mrd = eta_inv * phi_mrd_raw + C1MRD + C2MRD * Mf_safe

        phase_raw = tl.where(
            Mf_safe < fInsJoin,
            phi_ins,
            tl.where(Mf_safe >= fMRDJoin, phi_mrd, phi_int),
        )
        total_phase = phase_raw - time_shift * (Mf_safe - mf_ref) - phase_offset

        dist_m = distance * 1.0e6 * PC_SI
        amp_scale = (
            2.0
            * 0.15774577884845012
            * total_mass
            * MRSUN_SI
            * total_mass_seconds
            / dist_m
        )
        scaled_amplitude = amp_scale * amp

        cos_i = tl.libdevice.cos(inclination)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = -cos_i

        cos_phase = tl.libdevice.cos(total_phase)
        sin_phase = tl.libdevice.sin(total_phase)

        raw_r = scaled_amplitude * cos_phase
        raw_i = -scaled_amplitude * sin_phase

        plus0_r = raw_r * plus_factor
        plus0_i = raw_i * plus_factor
        cross0_r = -raw_i * cross_factor
        cross0_i = raw_r * cross_factor

        cos_nodes = tl.libdevice.cos(2.0 * long_asc_nodes)
        sin_nodes = tl.libdevice.sin(2.0 * long_asc_nodes)

        hp_r = cos_nodes * plus0_r + sin_nodes * cross0_r
        hp_i = cos_nodes * plus0_i + sin_nodes * cross0_i
        hc_r = cos_nodes * cross0_r - sin_nodes * plus0_r
        hc_i = cos_nodes * cross0_i - sin_nodes * plus0_i

        f_CUT = 0.2
        f_cut_hz = f_CUT / total_mass_seconds
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
    _imrphenomd_fused_kernel = None


def is_triton_available() -> bool:
    """Return whether Triton is installed and supported."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _prepare_triton_imrphenomd_buffers(params, device, real_dtype, batch_size):
    """Assemble amplitude, phase, and PN coefficient tensors for the fused Triton kernel."""
    m1_in = params["mass1"]
    m2_in = params["mass2"]

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

    m1 = _to_tensor(m1_in)
    m2 = _to_tensor(m2_in)
    s1z = _to_tensor(params.get("spin1z", 0.0))
    s2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))
    f_lower = float(params["f_lower"])

    from pycbc.waveform.imrphenomd_torch import (
        _final_spin0815,
        _erad_rational_0815,
        _qnm_splines,
        _rho1,
        _rho2,
        _rho3,
        _gamma1,
        _gamma2,
        _gamma3,
        _fmax_calc,
        _powers,
        _amp_int_col_fit,
        _solve_intermediate_polynomial,
        _sigma1,
        _sigma2,
        _sigma3,
        _sigma4,
        _beta1,
        _beta2,
        _beta3,
        _alpha1,
        _alpha2,
        _alpha3,
        _alpha4,
        _alpha5,
        _subtract_3pn_ss,
        _pi_powers,
        AMP_fJoin_INS,
        PHI_fJoin_INS,
        _PI,
        _MTSUN_SI,
        _ETA_EPS,
    )
    from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing

    swap_mask = m2 > m1
    m1_eff = torch.where(swap_mask, m2, m1)
    m2_eff = torch.where(swap_mask, m1, m2)
    s1z_eff = torch.where(swap_mask, s2z, s1z)
    s2z_eff = torch.where(swap_mask, s1z, s2z)

    total_mass = m1_eff + m2_eff
    eta = (m1_eff * m2_eff) / (total_mass * total_mass)
    eta = torch.where(eta > 0.25, 0.25 - _ETA_EPS, eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta_inv = 1.0 / eta

    seta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    seta_plus1 = 1.0 + seta
    chi12 = s1z_eff * s1z_eff
    chi22 = s2z_eff * s2z_eff
    chi = 0.5 * ((s1z_eff + s2z_eff) * (1.0 - 76.0 * eta / 113.0) + seta * (s1z_eff - s2z_eff))
    xi = -1.0 + chi

    finspin = _final_spin0815(eta, s1z_eff, s2z_eff)
    erad = _erad_rational_0815(eta, s1z_eff, s2z_eff)

    fring_spline, fdamp_spline = _qnm_splines()
    finspin_np = finspin.detach().cpu().numpy()
    fring = torch.as_tensor(fring_spline(finspin_np), dtype=real_dtype, device=device)
    fdamp = torch.as_tensor(fdamp_spline(finspin_np), dtype=real_dtype, device=device)
    fRD = fring / (1.0 - erad)
    fDM = fdamp / (1.0 - erad)

    rho1 = _rho1(eta, eta2, xi)
    rho2 = _rho2(eta, eta2, xi)
    rho3 = _rho3(eta, eta2, xi)

    gamma1 = _gamma1(eta, eta2, xi)
    gamma2 = _gamma2(eta, eta2, xi)
    gamma3 = _gamma3(eta, eta2, xi)

    fmaxCalc = _fmax_calc(fRD, fDM, gamma2, gamma3)
    amp0 = torch.sqrt(2.0 / 3.0 * eta) * (_PI ** (-1.0 / 6.0))

    pow_pi = _pi_powers()
    pi2 = _PI * _PI

    pref_two_thirds = ((-969.0 + 1804.0 * eta) * pow_pi.two_thirds) / 672.0
    pref_one = ((s1z_eff * (81.0 * seta_plus1 - 44.0 * eta) + s2z_eff * (81.0 - 81.0 * seta - 44.0 * eta)) * _PI) / 48.0
    pref_four_thirds = (
        (-27312085.0 - 10287648.0 * chi22 - 10287648.0 * chi12 * seta_plus1 + 10287648.0 * chi22 * seta)
        + 24.0 * (-1975055.0 + 857304.0 * chi12 - 994896.0 * s1z_eff * s2z_eff + 857304.0 * chi22) * eta
        + 35371056.0 * eta2
    ) * pow_pi.four_thirds / 8.128512e6
    pref_five_thirds = pow_pi.five_thirds * (
        s2z_eff * (-285197.0 * (-1.0 + seta) + 4.0 * (-91902.0 + 1579.0 * seta) * eta - 35632.0 * eta2)
        + s1z_eff * (285197.0 * seta_plus1 - 4.0 * (91902.0 + 1579.0 * seta) * eta - 35632.0 * eta2)
        + 42840.0 * (-1.0 + 4.0 * eta) * _PI
    ) / 32256.0
    pref_two = (
        -pi2
        * (
            -336.0
            * (-3248849057.0 + 2943675504.0 * chi12 - 3339284256.0 * s1z_eff * s2z_eff + 2943675504.0 * chi22)
            * eta2
            - 324322727232.0 * eta3
            - 7.0
            * (
                -177520268561.0
                + 107414046432.0 * chi22
                + 107414046432.0 * chi12 * seta_plus1
                - 107414046432.0 * chi22 * seta
                + 11087290368.0 * (s1z_eff + s2z_eff + s1z_eff * seta - s2z_eff * seta) * _PI
            )
            + 12.0
            * eta
            * (
                -545384828789.0
                - 176491177632.0 * s1z_eff * s2z_eff
                + 202603761360.0 * chi22
                + 77616.0 * chi12 * (2610335.0 + 995766.0 * seta)
                - 77287373856.0 * chi22 * seta
                + 5841690624.0 * (s1z_eff + s2z_eff) * _PI
                + 21384760320.0 * pi2
            )
        )
    ) / 6.0085960704e10
    pref_seven_thirds = rho1
    pref_eight_thirds = rho2
    pref_three = rho3

    f1 = torch.full((batch_size,), AMP_fJoin_INS, dtype=real_dtype, device=device)
    f3 = fmaxCalc
    f2 = f1 + 0.5 * (f3 - f1)

    powers_f1 = _powers(f1)
    v1 = (
        1.0
        + powers_f1.two_thirds * pref_two_thirds
        + powers_f1.four_thirds * pref_four_thirds
        + powers_f1.five_thirds * pref_five_thirds
        + powers_f1.seven_thirds * pref_seven_thirds
        + powers_f1.eight_thirds * pref_eight_thirds
        + f1 * (pref_one + f1 * pref_two + powers_f1.two * pref_three)
    )
    d1 = (
        (2.0 / 3.0) * pref_two_thirds * powers_f1.m_third
        + (4.0 / 3.0) * pref_four_thirds * powers_f1.third
        + (5.0 / 3.0) * pref_five_thirds * powers_f1.two_thirds
        + (7.0 / 3.0) * pref_seven_thirds * powers_f1.four_thirds
        + (8.0 / 3.0) * pref_eight_thirds * powers_f1.five_thirds
        + pref_one
        + 2.0 * pref_two * powers_f1.one
        + 3.0 * pref_three * powers_f1.two
    )
    fDMgamma3 = fDM * gamma3
    fminfRD_3 = f3 - fRD
    v3 = torch.exp(-fminfRD_3 * gamma2 / fDMgamma3) * (fDMgamma3 * gamma1) / (
        fminfRD_3 * fminfRD_3 + fDMgamma3 * fDMgamma3
    )
    pow2_3 = fDMgamma3 * fDMgamma3
    expfac_3 = torch.exp((fminfRD_3 * gamma2) / fDMgamma3)
    denom_3 = fminfRD_3 * fminfRD_3 + pow2_3
    d2 = ((-2.0 * fDM * fminfRD_3 * gamma3 * gamma1) / denom_3 - (gamma2 * gamma1)) / (expfac_3 * denom_3)
    v2 = _amp_int_col_fit(eta, eta2, chi)

    deltas = _solve_intermediate_polynomial(f1, f2, f3, v1, v2, v3, d1, d2)

    # Phasing
    sigma1 = _sigma1(eta, eta2, xi)
    sigma2 = _sigma2(eta, eta2, xi)
    sigma3 = _sigma3(eta, eta2, xi)
    sigma4 = _sigma4(eta, eta2, xi)
    beta1 = _beta1(eta, eta2, xi)
    beta2 = _beta2(eta, eta2, xi)
    beta3 = _beta3(eta, eta2, xi)
    alpha1 = _alpha1(eta, eta2, xi)
    alpha2 = _alpha2(eta, eta2, xi)
    alpha3 = _alpha3(eta, eta2, xi)
    alpha4 = _alpha4(eta, eta2, xi)
    alpha5 = _alpha5(eta, eta2, xi)

    pn = taylorf2_aligned_phasing(m1_eff, m2_eff, s1z_eff, s2z_eff)
    pn.v[6] -= _subtract_3pn_ss(m1_eff, m2_eff, total_mass, eta, s1z_eff, s2z_eff) * pn.v[0]

    phi_initial = pn.v[5] - 0.25 * _PI
    phi_two_thirds = pn.v[7] * pow_pi.two_thirds
    phi_third = pn.v[6] * pow_pi.third
    phi_third_logv = pn.vlogv[6] * pow_pi.third
    phi_logv = pn.vlogv[5]
    phi_minus_third = pn.v[4] * pow_pi.m_third
    phi_minus_two_thirds = pn.v[3] * pow_pi.m_two_thirds
    phi_minus_one = pn.v[2] * pow_pi.inv
    phi_minus_four_thirds = pn.v[1] / pow_pi.four_thirds
    phi_minus_five_thirds = pn.v[0] * pow_pi.m_five_thirds
    phi_one = sigma1
    phi_four_thirds = 0.75 * sigma2
    phi_five_thirds = 0.6 * sigma3
    phi_two = 0.5 * sigma4

    fInsJoin = torch.full((batch_size,), PHI_fJoin_INS, dtype=real_dtype, device=device)
    fMRDJoin = 0.5 * fRD

    v_ins = (_PI * PHI_fJoin_INS) ** (1.0 / 3.0)
    logv_ins = math.log(v_ins)
    v2_ins = v_ins * v_ins
    v3_ins = v_ins * v2_ins
    v4_ins = v_ins * v3_ins
    v5_ins = v_ins * v4_ins
    v6_ins = v_ins * v5_ins
    v7_ins = v_ins * v6_ins
    v8_ins = v_ins * v7_ins
    Dphasing = 2.0 * pn.v[7] * v7_ins
    Dphasing = Dphasing + (pn.v[6] + pn.vlogv[6] * (1.0 + logv_ins)) * v6_ins
    Dphasing = Dphasing + pn.vlogv[5] * v5_ins
    Dphasing = Dphasing - 1.0 * pn.v[4] * v4_ins
    Dphasing = Dphasing - 2.0 * pn.v[3] * v3_ins
    Dphasing = Dphasing - 3.0 * pn.v[2] * v2_ins
    Dphasing = Dphasing - 4.0 * pn.v[1] * v_ins
    Dphasing = Dphasing - 5.0 * pn.v[0]
    Dphasing = Dphasing / (v8_ins * 3.0)
    Dphasing = Dphasing * _PI
    pow_pi_m_third = float(pow_pi.m_third)
    pow_pi_m_two_thirds = float(pow_pi.m_two_thirds)
    pow_pi_inv = float(pow_pi.inv)
    Dphasing = Dphasing + (
        sigma1
        + sigma2 * v_ins * pow_pi_m_third
        + sigma3 * v2_ins * pow_pi_m_two_thirds
        + (sigma4 * pow_pi_inv) * v3_ins
    ) * eta_inv

    d_phi_ins_val = Dphasing
    d_phi_int_val = eta_inv * (beta1 + beta3 / (PHI_fJoin_INS ** 4) + beta2 / PHI_fJoin_INS)
    C2Int = d_phi_ins_val - d_phi_int_val

    powers_fIns = _powers(fInsJoin)
    v_fIns = powers_fIns.third * pow_pi.third
    logv_fIns = torch.log(v_fIns)
    phi_ins_val = (
        phi_initial
        + phi_two_thirds * powers_fIns.two_thirds
        + phi_third * powers_fIns.third
        + phi_third_logv * logv_fIns * powers_fIns.third
        + phi_logv * logv_fIns
        + phi_minus_third * powers_fIns.m_third
        + phi_minus_two_thirds * powers_fIns.m_two_thirds
        + phi_minus_one * powers_fIns.inv
        + phi_minus_four_thirds * powers_fIns.m_four_thirds
        + phi_minus_five_thirds * powers_fIns.m_five_thirds
        + eta_inv
        * (
            phi_one * fInsJoin
            + phi_four_thirds * powers_fIns.four_thirds
            + phi_five_thirds * powers_fIns.five_thirds
            + phi_two * powers_fIns.two
        )
    )
    phi_int_val = beta1 * PHI_fJoin_INS - beta3 / (3.0 * (PHI_fJoin_INS ** 3)) + beta2 * math.log(PHI_fJoin_INS)
    C1Int = phi_ins_val - eta_inv * phi_int_val - C2Int * PHI_fJoin_INS

    PhiIntTempVal = (
        eta_inv * (beta1 * fMRDJoin - beta3 / (3.0 * (fMRDJoin ** 3)) + beta2 * torch.log(fMRDJoin))
        + C1Int
        + C2Int * fMRDJoin
    )
    DPhiIntTempVal = C2Int + (beta1 + beta3 / (fMRDJoin ** 4) + beta2 / fMRDJoin) * eta_inv

    width_mrd = fDM
    DPhiMRDVal = eta_inv * (
        alpha1
        + alpha2 / (fMRDJoin * fMRDJoin)
        + alpha3 / (fMRDJoin**0.25)
        + alpha4 / (width_mrd * (1.0 + ((fMRDJoin - alpha5 * fRD) ** 2) / (width_mrd ** 2)))
    )
    C2MRD = DPhiIntTempVal - DPhiMRDVal

    sqroot_mrd = torch.sqrt(fMRDJoin)
    fpow1_5_mrd = fMRDJoin * sqroot_mrd
    fpow0_75_mrd = torch.sqrt(fpow1_5_mrd)
    PhiMRDJoin = (
        -(alpha2 / fMRDJoin)
        + (4.0 / 3.0) * (alpha3 * fpow0_75_mrd)
        + alpha1 * fMRDJoin
        + alpha4 * torch.arctan((fMRDJoin - alpha5 * fRD) / fDM)
    )
    C1MRD = PhiIntTempVal - eta_inv * PhiMRDJoin - C2MRD * fMRDJoin

    total_mass_seconds = total_mass * _MTSUN_SI

    ref_freq = torch.where(f_ref > 0.0, f_ref, torch.full_like(f_ref, f_lower))
    mf_ref = total_mass_seconds * ref_freq

    powers_ref = _powers(mf_ref)
    v_ref = powers_ref.third * pow_pi.third
    logv_ref = torch.log(v_ref)
    phi_ins_ref = (
        phi_initial
        + phi_two_thirds * powers_ref.two_thirds
        + phi_third * powers_ref.third
        + phi_third_logv * logv_ref * powers_ref.third
        + phi_logv * logv_ref
        + phi_minus_third * powers_ref.m_third
        + phi_minus_two_thirds * powers_ref.m_two_thirds
        + phi_minus_one * powers_ref.inv
        + phi_minus_four_thirds * powers_ref.m_four_thirds
        + phi_minus_five_thirds * powers_ref.m_five_thirds
        + eta_inv
        * (
            phi_one * mf_ref
            + phi_four_thirds * powers_ref.four_thirds
            + phi_five_thirds * powers_ref.five_thirds
            + phi_two * powers_ref.two
        )
    )
    phi_int_ref = (
        eta_inv * (beta1 * mf_ref - beta3 / (3.0 * (mf_ref ** 3)) + beta2 * torch.log(mf_ref))
        + C1Int
        + C2Int * mf_ref
    )
    sqroot_ref = torch.sqrt(mf_ref)
    fpow1_5_ref = mf_ref * sqroot_ref
    fpow0_75_ref = torch.sqrt(fpow1_5_ref)
    phi_mrd_raw_ref = (
        -(alpha2 / mf_ref)
        + (4.0 / 3.0) * (alpha3 * fpow0_75_ref)
        + alpha1 * mf_ref
        + alpha4 * torch.arctan((mf_ref - alpha5 * fRD) / fDM)
    )
    phi_mrd_ref = eta_inv * phi_mrd_raw_ref + C1MRD + C2MRD * mf_ref

    phase_at_ref = torch.where(
        mf_ref < PHI_fJoin_INS,
        phi_ins_ref,
        torch.where(mf_ref >= fMRDJoin, phi_mrd_ref, phi_int_ref),
    )
    phase_offset = 2.0 * coa_phase + phase_at_ref

    width_ts = fDM
    time_shift = eta_inv * (
        alpha1
        + alpha2 / (fmaxCalc * fmaxCalc)
        + alpha3 / (fmaxCalc**0.25)
        + alpha4 / (width_ts * (1.0 + ((fmaxCalc - alpha5 * fRD) ** 2) / (width_ts ** 2)))
    )

    amp_params = torch.stack(
        [
            fmaxCalc,
            fRD,
            fDM,
            gamma1,
            gamma2,
            gamma3,
            amp0,
            deltas[:, 0],
            deltas[:, 1],
            deltas[:, 2],
            deltas[:, 3],
            deltas[:, 4],
            pref_two_thirds,
            pref_four_thirds,
            pref_five_thirds,
            pref_seven_thirds,
            pref_eight_thirds,
            pref_one,
            pref_two,
            pref_three,
        ],
        dim=1,
    )

    phase_params = torch.stack(
        [
            fInsJoin,
            fMRDJoin,
            beta1,
            beta2,
            beta3,
            alpha1,
            alpha2,
            alpha3,
            alpha4,
            alpha5,
            fRD,
            fDM,
            C1Int,
            C2Int,
            C1MRD,
            C2MRD,
            time_shift,
            mf_ref,
            phase_offset,
        ],
        dim=1,
    )

    pn_coeffs = torch.stack(
        [
            phi_initial,
            phi_two_thirds,
            phi_third,
            phi_third_logv,
            phi_logv,
            phi_minus_third,
            phi_minus_two_thirds,
            phi_minus_one,
            phi_minus_four_thirds,
            phi_minus_five_thirds,
            phi_one,
            phi_four_thirds,
            phi_five_thirds,
            phi_two,
        ],
        dim=1,
    )

    return {
        "m1": m1_eff,
        "m2": m2_eff,
        "dist": dist,
        "incl": incl,
        "coa_phase": coa_phase,
        "long_asc_nodes": long_asc_nodes,
        "amp_params": amp_params,
        "phase_params": phase_params,
        "pn_coeffs": pn_coeffs,
    }


def imrphenomd_triton_fd(**params):
    """Generate single-call IMRPhenomD waveform using fused Triton kernel if available."""
    if not is_triton_available():
        from pycbc.waveform.imrphenomd_torch import imrphenomd_fd_torch
        return imrphenomd_fd_torch(**params)

    device = torch.device("cuda")
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    buffers = _prepare_triton_imrphenomd_buffers(params, device, real_dtype, batch_size=1)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    from pycbc.waveform.imrphenomd_torch import _lal_next_power_of_two, imrphenomd_cutoff_frequency
    f_cut_hz = imrphenomd_cutoff_frequency(float(buffers["m1"][0]), float(buffers["m2"][0]))
    layout_f_max = f_final if f_final > 0.0 else f_cut_hz
    npts = _lal_next_power_of_two(layout_f_max / delta_f) + 1

    first_bin = int(math.floor(f_lower / delta_f))
    stop_bin = int(math.floor(min(layout_f_max, f_cut_hz) / delta_f))
    n_freqs = stop_bin - first_bin

    hp_real = torch.zeros(npts, dtype=real_dtype, device=device)
    hp_imag = torch.zeros(npts, dtype=real_dtype, device=device)
    hc_real = torch.zeros(npts, dtype=real_dtype, device=device)
    hc_imag = torch.zeros(npts, dtype=real_dtype, device=device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_freqs, BLOCK_SIZE), 1)

    _imrphenomd_fused_kernel[grid](
        hp_real_ptr=hp_real[first_bin:],
        hp_imag_ptr=hp_imag[first_bin:],
        hc_real_ptr=hc_real[first_bin:],
        hc_imag_ptr=hc_imag[first_bin:],
        delta_f=delta_f,
        f_lower=f_lower,
        m1_ptr=buffers["m1"],
        m2_ptr=buffers["m2"],
        dist_ptr=buffers["dist"],
        incl_ptr=buffers["incl"],
        coa_phase_ptr=buffers["coa_phase"],
        long_asc_nodes_ptr=buffers["long_asc_nodes"],
        amp_params_ptr=buffers["amp_params"],
        phase_params_ptr=buffers["phase_params"],
        pn_coeffs_ptr=buffers["pn_coeffs"],
        stride_b_amp=buffers["amp_params"].stride(0),
        stride_b_phase=buffers["phase_params"].stride(0),
        stride_b_pn=buffers["pn_coeffs"].stride(0),
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


def imrphenomd_triton_fd_batch(**params):
    """Generate batch of IMRPhenomD waveforms using fused Triton kernel if available."""
    if not is_triton_available():
        from pycbc.waveform.imrphenomd_torch import imrphenomd_fd_batch
        return imrphenomd_fd_batch(**params)

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

    buffers = _prepare_triton_imrphenomd_buffers(params, device, real_dtype, batch_size=batch_size)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    from pycbc.waveform.imrphenomd_torch import _lal_next_power_of_two
    total_mass_seconds = (buffers["m1"] + buffers["m2"]) * 4.925490947641267e-06
    f_cut_hz = 0.2 / total_mass_seconds
    if f_final > 0.0:
        layout_f_max = f_final
    else:
        layout_f_max = float(torch.max(f_cut_hz).item())

    npts = _lal_next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(math.floor(f_lower / delta_f))
    max_stop_bin = int(math.floor(layout_f_max / delta_f))
    n_freqs = max_stop_bin - first_bin

    hp_real = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hp_imag = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hc_real = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)
    hc_imag = torch.zeros((batch_size, npts), dtype=real_dtype, device=device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_freqs, BLOCK_SIZE), batch_size)

    _imrphenomd_fused_kernel[grid](
        hp_real_ptr=hp_real[:, first_bin:],
        hp_imag_ptr=hp_imag[:, first_bin:],
        hc_real_ptr=hc_real[:, first_bin:],
        hc_imag_ptr=hc_imag[:, first_bin:],
        delta_f=delta_f,
        f_lower=f_lower,
        m1_ptr=buffers["m1"],
        m2_ptr=buffers["m2"],
        dist_ptr=buffers["dist"],
        incl_ptr=buffers["incl"],
        coa_phase_ptr=buffers["coa_phase"],
        long_asc_nodes_ptr=buffers["long_asc_nodes"],
        amp_params_ptr=buffers["amp_params"],
        phase_params_ptr=buffers["phase_params"],
        pn_coeffs_ptr=buffers["pn_coeffs"],
        stride_b_amp=buffers["amp_params"].stride(0),
        stride_b_phase=buffers["phase_params"].stride(0),
        stride_b_pn=buffers["pn_coeffs"].stride(0),
        stride_out_b=hp_real.stride(0),
        stride_out_f=hp_real.stride(1),
        n_freqs=n_freqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    hp = torch.complex(hp_real, hp_imag).to(complex_dtype)
    hc = torch.complex(hc_real, hc_imag).to(complex_dtype)
    return hp, hc
