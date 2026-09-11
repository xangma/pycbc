"""Prototype and benchmark fixed-capacity GPU-resident symmetric candidates.

This is an EXPERIMENT, not default integration. It prototypes keeping
candidates on-device through exact offline chi-square evaluation and final
newSNR threshold cuts using fixed-capacity per-template buffers, comparing
against the exact reference route.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from pycbc.events import ranking
from pycbc.filter.gpu_search.candidates import select_tile_candidates
from pycbc.filter.gpu_search import SelectionPolicy
from pycbc.vetoes.chisq_torch import (
    _cuda_exact_point_chisq,
    _get_cuda_exact_chisq_module,
    _search_compat_point_chisq,
)


def reference_candidate_pipeline(
    out_mem: torch.Tensor,
    norms_np: np.ndarray,
    sigmasqs_np: np.ndarray,
    corrs: torch.Tensor,
    bin_edges: List[int],
    valid_start: int,
    valid_end: int,
    policy: SelectionPolicy,
    newsnr_threshold: float = 6.0,
) -> Dict[str, Any]:
    """Exact offline reference route using production components.

    1. select_tile_candidates with out_mem, valid_start, valid_end (CPU export)
    2. Offline adapter roundtrip: raw_snrv = (cands_snr / norm_float32).astype(complex64)
    3. Points fed to exact chisq are ABSOLUTE full-transform indices (s_indices)
    4. Output SNR event is raw_snrv * norm_float32
    5. Final cut via production ranking.newsnr on float64 numpy arrays
    """
    bin_edges = tuple(int(edge) for edge in bin_edges)
    b = out_mem.shape[0]
    norms_tensor = torch.as_tensor(norms_np, device=out_mem.device, dtype=torch.float32)
    sigmasqs_tensor = torch.as_tensor(sigmasqs_np, device=out_mem.device, dtype=torch.float32)

    sel = select_tile_candidates(
        out_mem=out_mem,
        tile_norms=norms_tensor,
        tile_sigmasqs=sigmasqs_tensor,
        valid_start=valid_start,
        valid_end=valid_end,
        policy=policy,
    )
    if sel.get("aborted", False):
        return {"aborted": True, "overflow": False, "cands_before_cut": 0, "survivors": {}}

    cands = sel.get("candidates", {})
    t_indices = cands.get("template_idx", np.empty(0, dtype=np.int64))
    s_indices = cands.get("sample_idx", np.empty(0, dtype=np.int64))
    snr_vals = cands.get("snr", np.empty(0, dtype=np.complex64))
    sigmasq_vals = cands.get("sigmasq", np.empty(0, dtype=np.float32))

    num_cands = len(t_indices)
    if num_cands == 0:
        return {
            "aborted": False,
            "overflow": False,
            "cands_before_cut": 0,
            "survivors": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
                "chisq": np.empty(0, dtype=np.float32),
                "newsnr": np.empty(0, dtype=np.float64),
            },
        }

    num_bins = len(bin_edges) - 1
    dof = 2.0 * num_bins - 2.0

    cuda_mod = None
    if out_mem.device.type == "cuda":
        cuda_mod = _get_cuda_exact_chisq_module()
        if cuda_mod is None:
            raise RuntimeError("CUDA exact chisq module is unavailable on CUDA device")

    out_t = []
    out_s = []
    out_snr = []
    out_sig = []
    out_chi = []
    out_newsnr = []

    for i in range(b):
        mask = (t_indices == i)
        if not np.any(mask):
            continue
        norm_i = float(norms_np[i])
        pts_i = s_indices[mask]  # Absolute full-transform indices
        cand_snr_i = snr_vals[mask]
        sig_i = sigmasq_vals[mask]

        # Offline adapter normalization roundtrip:
        if norm_i <= 0.0:
            raw_snrv = np.zeros_like(cand_snr_i)
        else:
            raw_snrv = (cand_snr_i / np.float32(norm_i)).astype(np.complex64)
        event_snr = (raw_snrv * np.float32(norm_i)).astype(np.complex64)

        corr_i = corrs[i]
        if out_mem.device.type == "cuda":
            pts_dev = torch.as_tensor(pts_i, device=out_mem.device, dtype=torch.int64)
            raw_snr_dev = torch.as_tensor(raw_snrv, device=out_mem.device, dtype=torch.complex64)
            chisq_dev = _cuda_exact_point_chisq(
                cuda_mod, corr_i, pts_dev, bin_edges, raw_snr_dev, norm_i
            )
            chisq_np = chisq_dev.detach().cpu().numpy()
        else:
            chisq_res = _search_compat_point_chisq(
                corr_i, pts_i.astype(np.int64), bin_edges, raw_snrv, norm_i
            )
            if isinstance(chisq_res, torch.Tensor):
                chisq_np = chisq_res.detach().cpu().numpy()
            else:
                chisq_np = np.asarray(chisq_res, dtype=np.float32)

        snr_mag_f64 = np.abs(event_snr).astype(np.float64)
        rchisq_f64 = chisq_np.astype(np.float64) / dof
        newsnr_vals = ranking.newsnr(snr_mag_f64, rchisq_f64)
        newsnr_arr = np.array(newsnr_vals, ndmin=1, dtype=np.float64)

        surv_mask = newsnr_arr >= newsnr_threshold
        if np.any(surv_mask):
            out_t.append(np.full(np.count_nonzero(surv_mask), i, dtype=np.int64))
            out_s.append(pts_i[surv_mask])
            out_snr.append(event_snr[surv_mask])
            out_sig.append(sig_i[surv_mask])
            out_chi.append(chisq_np[surv_mask])
            out_newsnr.append(newsnr_arr[surv_mask])

    if len(out_t) == 0:
        return {
            "aborted": False,
            "overflow": False,
            "cands_before_cut": num_cands,
            "survivors": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
                "chisq": np.empty(0, dtype=np.float32),
                "newsnr": np.empty(0, dtype=np.float64),
            },
        }

    return {
        "aborted": False,
        "overflow": False,
        "cands_before_cut": num_cands,
        "survivors": {
            "template_idx": np.concatenate(out_t),
            "sample_idx": np.concatenate(out_s),
            "snr": np.concatenate(out_snr),
            "sigmasq": np.concatenate(out_sig),
            "chisq": np.concatenate(out_chi),
            "newsnr": np.concatenate(out_newsnr),
        },
    }


def resident_device_stage(
    out_mem: torch.Tensor,
    norms_tensor: torch.Tensor,
    sigmasqs_tensor: torch.Tensor,
    corrs: torch.Tensor,
    bin_edges: List[int],
    valid_start: int,
    valid_end: int,
    policy: SelectionPolicy,
    capacity_per_template: int,
    newsnr_threshold: float,
    norms_np: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Execute strictly on-device fixed-capacity resident stage.

    NO .item(), NO bool(tensor), NO .cpu(), NO .numpy(), NO dynamic torch.nonzero,
    NO boolean tensor indexing. Returns fixed-shape tensors and device abort/overflow/ambiguity flags.
    """
    bin_edges = tuple(int(edge) for edge in bin_edges)
    device = out_mem.device
    b = out_mem.shape[0]
    cap = int(capacity_per_template)
    num_bins = len(bin_edges) - 1
    dof = float(2.0 * num_bins - 2.0)

    # Slice valid interval matching select_tile_candidates
    valid_slice = slice(valid_start, valid_end)
    vals = out_mem[:, valid_slice]
    total_len = vals.shape[-1]
    window = max(1, int(policy.cluster_window))

    sq_mag = vals.real.square() + vals.imag.square()
    sq_mag = torch.nan_to_num(sq_mag, nan=0.0)

    num_blocks = (total_len + window - 1) // window
    if num_blocks == 0:
        empty_i64 = torch.zeros((b, cap), device=device, dtype=torch.int64)
        empty_f32 = torch.zeros((b, cap), device=device, dtype=torch.float32)
        empty_c64 = torch.zeros((b, cap), device=device, dtype=torch.complex64)
        empty_f64 = torch.zeros((b, cap), device=device, dtype=torch.float64)
        empty_bool = torch.zeros((b, cap), device=device, dtype=torch.bool)
        return {
            "device_abort": torch.tensor(False, device=device),
            "cand_counts": torch.zeros(b, device=device, dtype=torch.int64),
            "overflow_mask": torch.zeros(b, device=device, dtype=torch.bool),
            "device_ambiguity": torch.tensor(False, device=device),
            "template_idx": empty_i64,
            "sample_idx": empty_i64,
            "raw_snr": empty_c64,
            "event_snr": empty_c64,
            "sigmasq": empty_f32,
            "chisq": empty_f32,
            "newsnr": empty_f64,
            "surv_mask": empty_bool,
        }

    pad = num_blocks * window - total_len
    if pad > 0:
        pad_val = torch.full((b, pad), float("-inf"), device=device, dtype=sq_mag.dtype)
        sq_padded = torch.cat([sq_mag, pad_val], dim=-1)
    else:
        sq_padded = sq_mag

    blocks = sq_padded.view(b, num_blocks, window)
    block_max, block_idx = torch.max(blocks, dim=-1)
    block_idx = block_idx + torch.arange(num_blocks, device=device).unsqueeze(0) * window

    safe_norms = torch.clamp(norms_tensor, min=1e-12)
    thresh_sq = (policy.snr_threshold / safe_norms).square().unsqueeze(-1)
    keep = block_max > thresh_sq

    device_abort = torch.tensor(False, device=device)
    if policy.snr_abort_threshold is not None:
        abort_thresh_sq = (policy.snr_abort_threshold / safe_norms).square().unsqueeze(-1)
        device_abort = (block_max >= abort_thresh_sq).any()

    if num_blocks > 1:
        keep[:, 1:] &= block_max[:, 1:] > block_max[:, :-1]
        keep[:, :-1] &= block_max[:, :-1] >= block_max[:, 1:]
        keep[:, 0] &= block_max[:, 0] > block_max[:, 1]

    cand_counts = keep.sum(dim=-1).to(torch.int64)
    overflow_mask = cand_counts > cap

    cuda_mod = None
    if device.type == "cuda":
        cuda_mod = _get_cuda_exact_chisq_module()
        if cuda_mod is None:
            raise RuntimeError("CUDA exact chisq module is unavailable on CUDA device")

    template_idx = torch.arange(b, device=device, dtype=torch.int64).unsqueeze(1).expand(b, cap)
    sigmasq = sigmasqs_tensor.unsqueeze(1).expand(b, cap).to(torch.float32)

    sample_idx_list = []
    raw_snr_list = []
    event_snr_list = []
    chisq_list = []
    newsnr_list = []
    surv_mask_list = []
    ambiguity_list = []

    q = 6.0
    n = 2.0
    p_exp = q / n
    inv_neg_q = -1.0 / q
    thresh_tol = 1e-5 * max(1.0, abs(float(newsnr_threshold)))

    for i in range(b):
        keep_i = keep[i]
        norm_tensor_i = norms_tensor[i]
        norm_scalar_i = float(norms_np[i])

        # Fixed capacity extraction using torch.nonzero_static; padded with -1
        stat_idx = torch.nonzero_static(keep_i, size=cap, fill_value=-1).squeeze(-1)
        valid_slot = stat_idx >= 0
        safe_block_pos = torch.clamp(stat_idx, min=0)

        rel_offset = block_idx[i, safe_block_pos]
        abs_sample_idx = rel_offset + valid_start
        safe_abs_sample = torch.where(valid_slot, abs_sample_idx, torch.zeros_like(abs_sample_idx))

        peak_val = out_mem[i, safe_abs_sample]
        cand_snr = peak_val * norm_tensor_i
        # Use norm_tensor_i directly on device with safe zero-norm branch
        is_pos_norm = norm_tensor_i > 0.0
        raw_snrv = torch.where(
            is_pos_norm,
            (cand_snr / torch.where(is_pos_norm, norm_tensor_i, torch.ones_like(norm_tensor_i))).to(torch.complex64),
            torch.zeros_like(cand_snr),
        )
        event_snrv = (raw_snrv * norm_tensor_i).to(torch.complex64)

        corr_i = corrs[i]
        if device.type == "cuda":
            chisq_i = _cuda_exact_point_chisq(
                cuda_mod, corr_i, safe_abs_sample, bin_edges, raw_snrv, norm_scalar_i
            )
        else:
            chisq_res = _search_compat_point_chisq(
                corr_i, safe_abs_sample.to(torch.int64), bin_edges, raw_snrv, norm_scalar_i
            )
            if isinstance(chisq_res, torch.Tensor):
                chisq_i = chisq_res
            else:
                chisq_i = torch.as_tensor(chisq_res, device=device, dtype=torch.float32)

        # Divide chisq in float64 directly matching EventManager float64 division
        rchisq = chisq_i.to(torch.float64) / dof
        snr_mag = event_snrv.abs().to(torch.float64)
        reweight = (0.5 * (1.0 + rchisq ** p_exp)) ** inv_neg_q
        newsnr_i = torch.where(rchisq > 1.0, snr_mag * reweight, snr_mag)

        surv_i = valid_slot & (newsnr_i >= newsnr_threshold)
        # Conservative device ambiguity check: valid slot near threshold or nonfinite
        near_thresh = valid_slot & ((newsnr_i - newsnr_threshold).abs() <= thresh_tol)
        non_finite = valid_slot & (~torch.isfinite(newsnr_i))
        ambiguity_list.append((near_thresh | non_finite).any())

        sample_idx_list.append(safe_abs_sample)
        raw_snr_list.append(raw_snrv)
        event_snr_list.append(event_snrv)
        chisq_list.append(chisq_i)
        newsnr_list.append(newsnr_i)
        surv_mask_list.append(surv_i)

    device_ambiguity = torch.stack(ambiguity_list).any() if ambiguity_list else torch.tensor(False, device=device)

    return {
        "device_abort": device_abort,
        "cand_counts": cand_counts,
        "overflow_mask": overflow_mask,
        "device_ambiguity": device_ambiguity,
        "template_idx": template_idx,
        "sample_idx": torch.stack(sample_idx_list, dim=0),
        "raw_snr": torch.stack(raw_snr_list, dim=0),
        "event_snr": torch.stack(event_snr_list, dim=0),
        "sigmasq": sigmasq,
        "chisq": torch.stack(chisq_list, dim=0),
        "newsnr": torch.stack(newsnr_list, dim=0),
        "surv_mask": torch.stack(surv_mask_list, dim=0),
    }


def resident_candidate_pipeline(
    out_mem: torch.Tensor,
    norms_np: np.ndarray,
    sigmasqs_np: np.ndarray,
    corrs: torch.Tensor,
    bin_edges: List[int],
    valid_start: int,
    valid_end: int,
    policy: SelectionPolicy,
    capacity_per_template: int = 64,
    newsnr_threshold: float = 6.0,
) -> Dict[str, Any]:
    """Full resident pipeline separating device execution from host export & overflow/ambiguity fallback."""
    if capacity_per_template <= 0:
        raise ValueError("capacity_per_template must be positive")

    device = out_mem.device
    norms_tensor = torch.as_tensor(norms_np, device=device, dtype=torch.float32)
    sigmasqs_tensor = torch.as_tensor(sigmasqs_np, device=device, dtype=torch.float32)

    dev_res = resident_device_stage(
        out_mem=out_mem,
        norms_tensor=norms_tensor,
        sigmasqs_tensor=sigmasqs_tensor,
        corrs=corrs,
        bin_edges=bin_edges,
        valid_start=valid_start,
        valid_end=valid_end,
        policy=policy,
        capacity_per_template=capacity_per_template,
        newsnr_threshold=newsnr_threshold,
        norms_np=norms_np,
    )

    # Host boundary operations
    abort_val = bool(dev_res["device_abort"].item() if device.type == "cuda" else dev_res["device_abort"])
    if abort_val:
        return {
            "aborted": True,
            "overflow": False,
            "fallback": False,
            "overflow_count": 0,
            "ambiguity_count": 0,
            "cands_before_cut": 0,
            "survivors": {},
        }

    overflow_mask_np = dev_res["overflow_mask"].detach().cpu().numpy()
    cand_counts_np = dev_res["cand_counts"].detach().cpu().numpy()
    total_cands = int(np.sum(cand_counts_np))
    overflow_count = int(np.count_nonzero(overflow_mask_np))
    ambiguity_val = bool(dev_res["device_ambiguity"].item() if device.type == "cuda" else dev_res["device_ambiguity"])
    ambiguity_count = 1 if ambiguity_val else 0

    # If any template overflowed or boundary ambiguity occurred, fallback to whole-tile exact reference
    if overflow_count > 0 or ambiguity_count > 0:
        ref_fallback = reference_candidate_pipeline(
            out_mem=out_mem,
            norms_np=norms_np,
            sigmasqs_np=sigmasqs_np,
            corrs=corrs,
            bin_edges=bin_edges,
            valid_start=valid_start,
            valid_end=valid_end,
            policy=policy,
            newsnr_threshold=newsnr_threshold,
        )
        return {
            "aborted": ref_fallback["aborted"],
            "overflow": overflow_count > 0,
            "fallback": True,
            "overflow_count": overflow_count,
            "ambiguity_count": ambiguity_count,
            "cands_before_cut": total_cands,
            "survivors": ref_fallback["survivors"],
        }

    # Download padded arrays to host; do selection and sorting on host
    t_arr = dev_res["template_idx"].detach().cpu().numpy()
    s_arr = dev_res["sample_idx"].detach().cpu().numpy()
    snr_arr = dev_res["event_snr"].detach().cpu().numpy()
    sig_arr = dev_res["sigmasq"].detach().cpu().numpy()
    chi_arr = dev_res["chisq"].detach().cpu().numpy()
    newsnr_arr = dev_res["newsnr"].detach().cpu().numpy()
    surv_arr = dev_res["surv_mask"].detach().cpu().numpy()

    surv_indices = np.where(surv_arr)
    if len(surv_indices[0]) == 0:
        return {
            "aborted": False,
            "overflow": False,
            "fallback": False,
            "overflow_count": 0,
            "ambiguity_count": 0,
            "cands_before_cut": total_cands,
            "survivors": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
                "chisq": np.empty(0, dtype=np.float32),
                "newsnr": np.empty(0, dtype=np.float64),
            },
        }

    sel_t = t_arr[surv_indices]
    sel_s = s_arr[surv_indices]
    sel_snr = snr_arr[surv_indices]
    sel_sig = sig_arr[surv_indices]
    sel_chi = chi_arr[surv_indices]
    sel_newsnr = newsnr_arr[surv_indices]

    sort_order = np.lexsort((sel_s, sel_t))
    return {
        "aborted": False,
        "overflow": False,
        "fallback": False,
        "overflow_count": 0,
        "ambiguity_count": 0,
        "cands_before_cut": total_cands,
        "survivors": {
            "template_idx": sel_t[sort_order],
            "sample_idx": sel_s[sort_order],
            "snr": sel_snr[sort_order],
            "sigmasq": sel_sig[sort_order],
            "chisq": sel_chi[sort_order],
            "newsnr": sel_newsnr[sort_order],
        },
    }


def generate_self_consistent_inputs(
    batch_size: int,
    transform_len: int,
    num_bins: int = 16,
    device: str = "cpu",
    triggered: bool = True,
    num_triggers_per_template: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor, List[int]]:
    """Generate self-consistent full transform complex correlation and out_mem via torch.fft.ifft(..., norm='forward').

    Zero subsequent out_mem edits; all peaks are synthesized in corr positive-frequency bins.
    """
    rng = np.random.default_rng(seed)
    N = transform_len
    dev = torch.device(device)

    # Positive frequency support: [1, N // 2)
    k_pos_end = N // 2
    bin_edges = np.linspace(1, k_pos_end, num_bins + 1, dtype=int).tolist()
    num_pos_bins = k_pos_end - 1

    norms_np = rng.uniform(0.8, 1.2, size=batch_size).astype(np.float32)
    sigmasqs_np = rng.uniform(10.0, 50.0, size=batch_size).astype(np.float32)

    # Low complex background spectrum (e.g. 0.01 / sqrt(N)) to guarantee quiet tiles remain strictly sub-threshold
    scale = 0.01 / np.sqrt(N)
    corr_full = np.zeros((batch_size, N), dtype=np.complex64)
    background = (rng.standard_normal((batch_size, num_pos_bins)) + 1j * rng.standard_normal((batch_size, num_pos_bins))) * scale
    corr_full[:, 1:k_pos_end] = background.astype(np.complex64)

    if triggered:
        num_peaks = min(5, int(num_triggers_per_template))
        valid_start = N // 4
        valid_end = 3 * (N // 4)
        spacing = (valid_end - valid_start) // (num_peaks + 1)
        freqs = np.arange(1, k_pos_end, dtype=np.float64)

        for b in range(batch_size):
            norm_b = float(norms_np[b])
            for k in range(num_peaks):
                t_peak = valid_start + (k + 1) * spacing
                target_snr = 9.0 + (k % 3) * 2.0  # SNR ~ 9-13
                base_amp = target_snr / (norm_b * float(num_pos_bins))
                # Fourier phase ramp exp(-2j * pi * freqs * t_peak / N)
                phase = -2.0 * np.pi * freqs * (float(t_peak) / float(N))
                pulse = base_amp * np.exp(1j * phase)

                # Alternating per-bin weights 1.0 +/- 0.6 to yield modest non-zero reduced chi-square
                for bin_idx in range(num_bins):
                    b_start = bin_edges[bin_idx] - 1
                    b_end = bin_edges[bin_idx + 1] - 1
                    bin_factor = 1.6 if (bin_idx % 2 == 0) else 0.4
                    pulse[b_start:b_end] *= bin_factor

                corr_full[b, 1:k_pos_end] += pulse.astype(np.complex64)

    corr_tensor = torch.as_tensor(corr_full, device=dev)
    # Compute matched-filter output ONLY via forward IFFT without any later edits
    out_mem = torch.fft.ifft(corr_tensor, n=N, dim=-1, norm="forward")
    return out_mem, norms_np, sigmasqs_np, corr_tensor, bin_edges


def run_benchmark(
    device: str = "cpu",
    iterations: int = 5,
    warmup: int = 2,
    quick: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Rigorous benchmark comparing reference and resident pipelines."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is False")

    dev = torch.device(device)
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.5,
        snr_abort_threshold=150.0,
    )

    configs = [
        {"name": "N65536_B1_cap16", "batch": 1, "len": 65536, "cap": 16, "trig": True},
        {"name": "N65536_B16_cap1", "batch": 16, "len": 65536, "cap": 1, "trig": True},  # Deliberate overflow / small cap
        {"name": "N65536_B16_cap16", "batch": 16, "len": 65536, "cap": 16, "trig": True},
    ]
    if not quick:
        configs.extend([
            {"name": "N2097152_B1_cap16", "batch": 1, "len": 2097152, "cap": 16, "trig": True},
            {"name": "N2097152_B4_cap4", "batch": 4, "len": 2097152, "cap": 4, "trig": True},
            {"name": "N65536_B16_quiet_cap16", "batch": 16, "len": 65536, "cap": 16, "trig": False},
        ])

    results = []
    source_file = os.path.abspath(__file__)
    with open(source_file, "rb") as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Hash related production files for reproducibility
    prod_hashes = {}
    for mod_path in [
        os.path.join(sys_path_root, "pycbc", "filter", "gpu_search", "candidates.py"),
        os.path.join(sys_path_root, "pycbc", "filter", "gpu_search", "adapter.py"),
        os.path.join(sys_path_root, "pycbc", "vetoes", "chisq_torch.py"),
        os.path.join(sys_path_root, "pycbc", "events", "ranking.py"),
    ]:
        if os.path.exists(mod_path):
            with open(mod_path, "rb") as f:
                prod_hashes[os.path.basename(mod_path)] = hashlib.sha256(f.read()).hexdigest()

    # Deterministic benchmark RNG for order alternation
    order_rng = random.Random(42)

    for cfg in configs:
        b_size = cfg["batch"]
        t_len = cfg["len"]
        cap = cfg["cap"]
        trig = cfg["trig"]
        v_start = t_len // 4
        v_end = 3 * (t_len // 4)

        out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
            batch_size=b_size,
            transform_len=t_len,
            num_bins=16,
            device=device,
            triggered=trig,
            num_triggers_per_template=3,
            seed=101,
        )

        # Warmup and measure extension / kernel init
        t_w0 = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy)
            _ = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=cap)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        warmup_time_s = time.perf_counter() - t_w0

        # Correctness equality and scientific parity check outside timed loop
        ref_baseline = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy)
        res_baseline = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=cap)

        ref_s = ref_baseline.get("survivors", {})
        res_s = res_baseline.get("survivors", {})

        if ref_baseline.get("aborted", False):
            scientific_match = bool(res_baseline.get("aborted", False))
            parity_details = {"aborted": True}
            max_diffs = {}
        else:
            idx_eq = bool(
                np.array_equal(ref_s.get("template_idx", []), res_s.get("template_idx", [])) and
                np.array_equal(ref_s.get("sample_idx", []), res_s.get("sample_idx", []))
            )
            sig_eq = bool(np.array_equal(ref_s.get("sigmasq", []), res_s.get("sigmasq", [])))
            max_diffs = {}
            exact_fields = {}
            for field in ref_s:
                left, right = ref_s[field], res_s[field]
                same_shape = left.shape == right.shape
                exact_fields[field] = bool(
                    same_shape and left.dtype == right.dtype
                    and np.array_equal(left, right, equal_nan=True)
                )
                max_diffs[field] = (
                    float(np.max(np.abs(left - right))) if left.size else 0.0
                ) if same_shape else None
            parity_details = {
                "exact_indices": idx_eq,
                "exact_sigmasq": sig_eq,
                "exact_fields": exact_fields,
            }
            scientific_match = all(
                match for field, match in exact_fields.items()
                if field != "newsnr"
            )
            assert not ref_baseline["aborted"]
            if trig:
                assert 0 < ref_baseline["cands_before_cut"] <= b_size * 8
                assert len(ref_s["sample_idx"]) > 0
            else:
                assert ref_baseline["cands_before_cut"] == 0

        ref_times = []
        res_times = []
        peak_vram_ref = 0
        peak_vram_res = 0

        for _ in range(iterations):
            order_flip = order_rng.choice([True, False])
            if order_flip:
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    peak_vram_ref = max(peak_vram_ref, torch.cuda.max_memory_allocated(dev))
                ref_times.append(time.perf_counter() - t0)

                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                _ = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=cap)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    peak_vram_res = max(peak_vram_res, torch.cuda.max_memory_allocated(dev))
                res_times.append(time.perf_counter() - t2)
            else:
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                _ = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=cap)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    peak_vram_res = max(peak_vram_res, torch.cuda.max_memory_allocated(dev))
                res_times.append(time.perf_counter() - t2)

                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    peak_vram_ref = max(peak_vram_ref, torch.cuda.max_memory_allocated(dev))
                ref_times.append(time.perf_counter() - t0)

        results.append({
            "name": cfg["name"],
            "config": cfg,
            "warmup_time_s": warmup_time_s,
            "ref_p50_ms": float(np.percentile(ref_times, 50) * 1000),
            "ref_p95_ms": float(np.percentile(ref_times, 95) * 1000),
            "res_p50_ms": float(np.percentile(res_times, 50) * 1000),
            "res_p95_ms": float(np.percentile(res_times, 95) * 1000),
            "speedup_ratio_p50": float(np.percentile(ref_times, 50) / max(1e-9, np.percentile(res_times, 50))),
            "scientific_match": scientific_match,
            "valid_for_promotion": False,
            "promotion_reason": "Prototype only: parity and cost evaluation in progress; not integrated into production CLI",
            "parity_details": parity_details,
            "max_differences": max_diffs,
            "cands_before_cut": res_baseline.get("cands_before_cut", 0),
            "survivors_count": len(res_s.get("sample_idx", [])),
            "overflow": res_baseline.get("overflow", False),
            "overflow_count": res_baseline.get("overflow_count", 0),
            "ambiguity_count": res_baseline.get("ambiguity_count", 0),
            "peak_vram_ref_bytes": peak_vram_ref,
            "peak_vram_res_bytes": peak_vram_res,
            "raw_ref_times_s": ref_times,
            "raw_res_times_s": res_times,
        })

    receipt = {
        "metadata": {
            "purpose": "Prototype and measure fixed-capacity GPU-resident symmetric candidates",
            "status": "prototype_only",
            "production_cli_integration": False,
            "full_application_speedup_claim": False,
            "device": device,
            "torch_version": str(torch.__version__),
            "source_sha256": source_sha256,
            "production_source_sha256": prod_hashes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)

    return receipt


def main():
    parser = argparse.ArgumentParser(description="Experiment: GPU-resident candidate prototype")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device")
    parser.add_argument("--output", type=str, default="resident_candidates_benchmark.json", help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of cases")
    args = parser.parse_args()

    receipt = run_benchmark(
        device=args.device,
        iterations=args.iterations,
        warmup=args.warmup,
        quick=args.quick,
        output_path=args.output,
    )
    print(f"Experiment complete. Output written to {args.output}")
    for r in receipt["results"]:
        print(
            f"[{r['name']}] Ref p50: {r['ref_p50_ms']:.2f}ms | Res p50: {r['res_p50_ms']:.2f}ms "
            f"| Ratio: {r['speedup_ratio_p50']:.2f}x | Match: {r['scientific_match']} | Cands: {r['cands_before_cut']} | Surv: {r['survivors_count']}"
        )


if __name__ == "__main__":
    main()
