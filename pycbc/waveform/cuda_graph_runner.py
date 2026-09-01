# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Static CUDA Graph runner for ultra-low latency (<5 us) waveform generation."""

from __future__ import annotations

import torch


class CUDAGraphWaveformRunner:
    """Captures and replays static waveform generation graphs on CUDA."""

    def __init__(
        self,
        approximant_or_kernel_fn="TaylorF2",
        n_freqs: int = 1024,
        device="cuda",
        delta_f: float = 0.5,
        f_lower: float = 20.0,
        batch_size: int = 1,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphWaveformRunner requires CUDA support")

        self.device = torch.device(device)
        self.n_freqs = n_freqs
        self.delta_f = delta_f
        self.f_lower = f_lower
        self.batch_size = batch_size
        self.graph = None

        if isinstance(approximant_or_kernel_fn, str):
            self.approximant = approximant_or_kernel_fn
            self.kernel_fn = self._get_approximant_kernel(self.approximant)
        else:
            self.approximant = "Custom"
            self.kernel_fn = approximant_or_kernel_fn

        # Allocate static parameter and output buffers
        self.static_params = self._allocate_static_params()
        if self.batch_size == 1:
            self.static_hp = torch.zeros(n_freqs, dtype=torch.complex128, device=self.device)
            self.static_hc = torch.zeros(n_freqs, dtype=torch.complex128, device=self.device)
        else:
            self.static_hp = torch.zeros((batch_size, n_freqs), dtype=torch.complex128, device=self.device)
            self.static_hc = torch.zeros((batch_size, n_freqs), dtype=torch.complex128, device=self.device)

        if self.device.type == "cuda":
            self._warmup_and_capture()

    def _get_approximant_kernel(self, approximant: str):
        if approximant.lower() in ("taylorf2", "taylor_f2"):
            from pycbc.waveform.triton.taylorf2 import _taylorf2_fused_kernel, _TRITON_AVAILABLE
            if _TRITON_AVAILABLE and _taylorf2_fused_kernel is not None:
                def kernel_fn(params, hp, hc):
                    BLOCK_SIZE = 128
                    import triton
                    grid = (triton.cdiv(self.n_freqs, BLOCK_SIZE), self.batch_size)
                    hp_r = hp.real
                    hp_i = hp.imag
                    hc_r = hc.real
                    hc_i = hc.imag
                    _taylorf2_fused_kernel[grid](
                        hp_real_ptr=hp_r,
                        hp_imag_ptr=hp_i,
                        hc_real_ptr=hc_r,
                        hc_imag_ptr=hc_i,
                        delta_f=self.delta_f,
                        f_lower=self.f_lower,
                        m1_ptr=params["m1"],
                        m2_ptr=params["m2"],
                        dist_ptr=params["dist"],
                        incl_ptr=params["incl"],
                        coa_phase_ptr=params["coa_phase"],
                        v_coeffs_ptr=params["v_coeffs"],
                        vlogv_coeffs_ptr=params["vlogv_coeffs"],
                        vlogvsq_coeffs_ptr=params["vlogvsq_coeffs"],
                        stride_b_coeffs=params["v_coeffs"].stride(0) if self.batch_size > 1 else 16,
                        stride_out_b=hp_r.stride(0) if self.batch_size > 1 else 0,
                        stride_out_f=hp_r.stride(-1),
                        n_freqs=self.n_freqs,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                return kernel_fn
            else:
                def fallback_fn(params, hp, hc):
                    freqs = self.f_lower + torch.arange(self.n_freqs, dtype=torch.float64, device=self.device) * self.delta_f
                    MTSUN_SI = 4.925490947641267e-06
                    MRSUN_SI = 1.476625038050125e03
                    PC_SI = 3.08567758149137e16
                    PI = 3.14159265358979323846
                    m1 = params["m1"]
                    m2 = params["m2"]
                    dist = params["dist"]
                    incl = params["incl"]
                    coa_phase = params["coa_phase"]
                    total_mass = m1 + m2
                    eta = (m1 * m2) / (total_mass * total_mass)
                    pi_mass = PI * total_mass * MTSUN_SI
                    v = torch.pow(pi_mass * freqs, 1.0 / 3.0)
                    log_v = torch.log(v)
                    log_v_sq = log_v * log_v
                    v_coeffs = params["v_coeffs"]
                    vlogv_coeffs = params["vlogv_coeffs"]
                    vlogvsq_coeffs = params["vlogvsq_coeffs"]
                    res = v_coeffs[15] + vlogv_coeffs[15] * log_v + vlogvsq_coeffs[15] * log_v_sq
                    for k in range(14, -1, -1):
                        term = v_coeffs[k] + vlogv_coeffs[k] * log_v + vlogvsq_coeffs[k] * log_v_sq
                        res = res * v + term
                    v5 = v * v * v * v * v
                    phi = res / v5
                    epoch = -1.0 / self.delta_f
                    total_phase = phi + (2.0 * PI * epoch * freqs) - (2.0 * coa_phase) - (PI / 4.0)
                    dist_m = dist * 1.0e6 * PC_SI
                    amp0 = -4.0 * m1 * m2 / dist_m * MRSUN_SI * MTSUN_SI * 0.5123496421589149
                    amp_factor = amp0 * torch.sqrt(5.0 / (32.0 * eta))
                    amplitude = amp_factor * torch.pow(v, -3.5)
                    cos_i = torch.cos(incl)
                    plus_factor = 0.5 * (1.0 + cos_i * cos_i)
                    cross_factor = -cos_i
                    raw = amplitude * torch.exp(1j * total_phase)
                    hp.copy_((raw * plus_factor).reshape(hp.shape))
                    hc.copy_((raw * cross_factor * 1j).reshape(hc.shape))
                return fallback_fn

        elif approximant.lower() in ("imrphenomd", "phenomd"):
            from pycbc.waveform.triton.imrphenomd import _imrphenomd_fused_kernel, _TRITON_AVAILABLE
            if _TRITON_AVAILABLE and _imrphenomd_fused_kernel is not None:
                def kernel_fn(params, hp, hc):
                    BLOCK_SIZE = 128
                    import triton
                    grid = (triton.cdiv(self.n_freqs, BLOCK_SIZE), self.batch_size)
                    hp_r = hp.real
                    hp_i = hp.imag
                    hc_r = hc.real
                    hc_i = hc.imag
                    _imrphenomd_fused_kernel[grid](
                        hp_real_ptr=hp_r,
                        hp_imag_ptr=hp_i,
                        hc_real_ptr=hc_r,
                        hc_imag_ptr=hc_i,
                        delta_f=self.delta_f,
                        f_lower=self.f_lower,
                        m1_ptr=params["m1"],
                        m2_ptr=params["m2"],
                        dist_ptr=params["dist"],
                        incl_ptr=params["incl"],
                        coa_phase_ptr=params["coa_phase"],
                        long_asc_nodes_ptr=params["long_asc_nodes"],
                        amp_params_ptr=params["amp_params"],
                        phase_params_ptr=params["phase_params"],
                        pn_coeffs_ptr=params["pn_coeffs"],
                        stride_b_amp=params["amp_params"].stride(0) if self.batch_size > 1 else 20,
                        stride_b_phase=params["phase_params"].stride(0) if self.batch_size > 1 else 19,
                        stride_b_pn=params["pn_coeffs"].stride(0) if self.batch_size > 1 else 14,
                        stride_out_b=hp_r.stride(0) if self.batch_size > 1 else 0,
                        stride_out_f=hp_r.stride(-1),
                        n_freqs=self.n_freqs,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                return kernel_fn
            else:
                def fallback_fn(params, hp, hc):
                    m1_val = float(params["m1"][0].item())
                    m2_val = float(params["m2"][0].item())
                    if m1_val <= 0 or m2_val <= 0:
                        return
                    from pycbc.waveform.imrphenomd_torch import imrphenomd_fd_torch
                    p = {
                        "mass1": m1_val,
                        "mass2": m2_val,
                        "distance": float(params["dist"][0].item()) or 1.0,
                        "inclination": float(params["incl"][0].item()),
                        "coa_phase": float(params["coa_phase"][0].item()),
                        "long_asc_nodes": float(params["long_asc_nodes"][0].item()),
                        "delta_f": self.delta_f,
                        "f_lower": self.f_lower,
                        "f_final": self.f_lower + self.n_freqs * self.delta_f,
                    }
                    hp_s, hc_s = imrphenomd_fd_torch(**p)
                    hp_t = torch.as_tensor(hp_s._data.tensor, device=self.device)
                    hc_t = torch.as_tensor(hc_s._data.tensor, device=self.device)
                    n = min(self.n_freqs, hp_t.shape[-1])
                    hp[:n].copy_(hp_t[:n])
                    hc[:n].copy_(hc_t[:n])
                return fallback_fn

        elif approximant.lower() in ("imrphenomxas", "phenomxas"):
            from pycbc.waveform.triton.imrphenomxas import _imrphenomxas_fused_kernel, _TRITON_AVAILABLE
            if _TRITON_AVAILABLE and _imrphenomxas_fused_kernel is not None:
                def kernel_fn(params, hp, hc):
                    BLOCK_SIZE = 128
                    import triton
                    grid = (triton.cdiv(self.n_freqs, BLOCK_SIZE), self.batch_size)
                    hp_r = hp.real
                    hp_i = hp.imag
                    hc_r = hc.real
                    hc_i = hc.imag
                    _imrphenomxas_fused_kernel[grid](
                        hp_real_ptr=hp_r,
                        hp_imag_ptr=hp_i,
                        hc_real_ptr=hc_r,
                        hc_imag_ptr=hc_i,
                        delta_f=self.delta_f,
                        f_lower=self.f_lower,
                        packed_plan_ptr=params["packed_plan"],
                        incl_ptr=params["incl"],
                        coa_phase_ptr=params["coa_phase"],
                        long_asc_nodes_ptr=params["long_asc_nodes"],
                        stride_b_plan=params["packed_plan"].stride(0) if self.batch_size > 1 else 66,
                        stride_out_b=hp_r.stride(0) if self.batch_size > 1 else 0,
                        stride_out_f=hp_r.stride(-1),
                        n_freqs=self.n_freqs,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                return kernel_fn
            else:
                def fallback_fn(params, hp, hc):
                    from pycbc.waveform.imrphenomxas_torch import _evaluate_packed_frequency_plan
                    freqs = self.f_lower + torch.arange(self.n_freqs, dtype=torch.float64, device=self.device) * self.delta_f
                    packed = params["packed_plan"][0] if self.batch_size == 1 else params["packed_plan"]
                    if packed.numel() == 66 and bool(torch.any(packed != 0)):
                        samples = _evaluate_packed_frequency_plan(freqs, packed)
                        cos_i = torch.cos(params["incl"][0])
                        plus0 = -0.5 * (1.0 + cos_i * cos_i) * samples
                        cross0 = 1j * cos_i * samples
                        hp.copy_(plus0.reshape(hp.shape))
                        hc.copy_(cross0.reshape(hc.shape))
                return fallback_fn

        return None

    def _allocate_static_params(self):
        b = self.batch_size
        if self.approximant.lower() in ("imrphenomd", "phenomd"):
            return {
                "m1": torch.zeros(b, dtype=torch.float64, device=self.device),
                "m2": torch.zeros(b, dtype=torch.float64, device=self.device),
                "dist": torch.zeros(b, dtype=torch.float64, device=self.device),
                "incl": torch.zeros(b, dtype=torch.float64, device=self.device),
                "coa_phase": torch.zeros(b, dtype=torch.float64, device=self.device),
                "long_asc_nodes": torch.zeros(b, dtype=torch.float64, device=self.device),
                "amp_params": torch.zeros((b, 20), dtype=torch.float64, device=self.device),
                "phase_params": torch.zeros((b, 19), dtype=torch.float64, device=self.device),
                "pn_coeffs": torch.zeros((b, 14), dtype=torch.float64, device=self.device),
            }
        elif self.approximant.lower() in ("imrphenomxas", "phenomxas"):
            return {
                "packed_plan": torch.zeros((b, 66), dtype=torch.float64, device=self.device),
                "incl": torch.zeros(b, dtype=torch.float64, device=self.device),
                "coa_phase": torch.zeros(b, dtype=torch.float64, device=self.device),
                "long_asc_nodes": torch.zeros(b, dtype=torch.float64, device=self.device),
            }
        else:
            # TaylorF2 / default
            return {
                "m1": torch.zeros(b, dtype=torch.float64, device=self.device),
                "m2": torch.zeros(b, dtype=torch.float64, device=self.device),
                "dist": torch.zeros(b, dtype=torch.float64, device=self.device),
                "incl": torch.zeros(b, dtype=torch.float64, device=self.device),
                "coa_phase": torch.zeros(b, dtype=torch.float64, device=self.device),
                "v_coeffs": torch.zeros((b, 16) if b > 1 else 16, dtype=torch.float64, device=self.device),
                "vlogv_coeffs": torch.zeros((b, 16) if b > 1 else 16, dtype=torch.float64, device=self.device),
                "vlogvsq_coeffs": torch.zeros((b, 16) if b > 1 else 16, dtype=torch.float64, device=self.device),
            }

    def _warmup_and_capture(self):
        if self.kernel_fn is None:
            return
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.kernel_fn(self.static_params, self.static_hp, self.static_hc)
            torch.cuda.synchronize(self.device)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=s):
                self.kernel_fn(self.static_params, self.static_hp, self.static_hc)
        torch.cuda.current_stream(self.device).wait_stream(s)

    def execute(self, *args, **kwargs):
        """Asynchronously updates static buffers with zero-copy copies and replays CUDA Graph."""
        if args:
            # Positional arguments for backwards-compatible TaylorF2 call
            keys = [
                "m1",
                "m2",
                "dist",
                "incl",
                "coa_phase",
                "v_coeffs",
                "vlogv_coeffs",
                "vlogvsq_coeffs",
            ]
            for key, val in zip(keys, args):
                if key in self.static_params and val is not None:
                    t_val = torch.as_tensor(val, device=self.device, dtype=torch.float64)
                    self.static_params[key].copy_(t_val.reshape(self.static_params[key].shape), non_blocking=True)

        for key, val in kwargs.items():
            if key in self.static_params and val is not None:
                t_val = torch.as_tensor(val, device=self.device, dtype=torch.float64)
                self.static_params[key].copy_(t_val.reshape(self.static_params[key].shape), non_blocking=True)

        if self.graph is not None:
            self.graph.replay()
        elif self.kernel_fn is not None:
            self.kernel_fn(self.static_params, self.static_hp, self.static_hc)

        return self.static_hp, self.static_hc
