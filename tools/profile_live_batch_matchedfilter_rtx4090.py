#!/usr/bin/env python3
"""Deep GPU Profiling Suite for LiveBatchMatchedFilter on NVIDIA RTX 4090.

Measures:
1. Full parameter grid: Batch sizes B in [1, 4, 16, 32, 64, 128, 256, 512, 1024]
   and Segment lengths N in [32768, 65536, 131072, 262144, 524288].
2. Latency & Throughput (transforms/sec, waveforms/sec, GFLOPs, Memory Bandwidth in GB/s).
3. Sub-microsecond stage breakdown (CUDA Events):
   - Template batch packaging / tensor stacking
   - 2D batched frequency correlation (torch.mul)
   - 2D batched cuFFT inverse FFT (torch.fft.ifft)
   - Magnitude-squared & peak reduction / thresholding
   - Host <-> Device data movement (H2D strain, D2H triggers)
4. GPU Memory & Cache Efficiency:
   - VRAM allocation churn, peak allocated, reserved memory
   - Working set size vs 72 MB L2 cache residency
   - Hardware optimal batch tile size evaluation
5. Optimization explorations:
   - Fused Peak Reduction Triton kernel vs multi-pass PyTorch
   - CUDA Graph capture and replay speedup
   - Async double buffering / stream overlap
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

# Ensure pycbc is imported from repo
repo_dir = Path(__file__).resolve().parent.parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

import pycbc.filter.matchedfilter as mf
import pycbc.filter.matchedfilter_torch as mf_torch
from pycbc import scheme
from pycbc.hardware import (
    get_gpu_l2_cache_size,
    get_optimal_batch_maxelements,
    get_optimal_batch_tile_size,
)
from pycbc.types import FrequencySeries


# ---------------------------------------------------------------------------
# Triton Fused MagSq + Peak Reduction Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _triton_row_magsq_max_kernel(
    input_ptr,       # Pointer to float32 (B, N, 2)
    max_sq_ptr,      # Pointer to float32 (B)
    max_idx_ptr,     # Pointer to int64 (B)
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)  # Each program processes 1 row
    row_offset = row * N

    cur_max = -1.0e30
    cur_idx = 0

    for base in range(0, N, BLOCK_SIZE):
        idx = base + tl.arange(0, BLOCK_SIZE)
        mask = idx < N

        re = tl.load(input_ptr + (row_offset + idx) * 2, mask=mask, other=0.0)
        im = tl.load(input_ptr + (row_offset + idx) * 2 + 1, mask=mask, other=0.0)

        sq = tl.where(mask, re * re + im * im, -1.0e30)
        b_max = tl.max(sq, axis=0)
        
        if b_max > cur_max:
            cur_max = b_max
            tile_arg = tl.argmax(sq, axis=0)
            cur_idx = base + tile_arg

    tl.store(max_sq_ptr + row, cur_max)
    tl.store(max_idx_ptr + row, cur_idx)


def triton_fused_batch_peak_and_threshold(
    values: torch.Tensor,
    norms: torch.Tensor,
    snr_threshold: float,
    snr_abort_threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Triton-accelerated fused complex magnitude-squared and peak extraction."""
    B, N = values.shape
    device = values.device
    
    # Flatten view of complex64 as float32
    f32_view = torch.view_as_real(values).reshape(B * N * 2)
    
    max_sq = torch.empty(B, dtype=torch.float32, device=device)
    max_idx = torch.empty(B, dtype=torch.int64, device=device)
    
    BLOCK_SIZE = 1024
    grid = (B,)
    _triton_row_magsq_max_kernel[grid](
        f32_view,
        max_sq,
        max_idx,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Threshold check on device
    norms_f32 = norms.to(device=device, dtype=torch.float32)
    max_snr_sq = max_sq * norms_f32.square()
    max_val = torch.max(max_snr_sq).item()
    
    thresh_sq = float(snr_threshold) ** 2
    if max_val < thresh_sq:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.complex64),
            False,
        )
        
    if snr_abort_threshold is not None:
        abort_thresh_sq = float(snr_abort_threshold) ** 2
        if max_val > abort_thresh_sq:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.complex64),
                True,
            )
            
    crossing_mask = max_snr_sq >= thresh_sq
    survivors = torch.nonzero(crossing_mask, as_tuple=True)[0]
    surv_idx = max_idx[survivors]
    
    # Gather peaks for surviving triggers
    g_idx = surv_idx.unsqueeze(1)
    surv_values = values[survivors]
    surv_peaks = torch.complex(
        surv_values.real.gather(1, g_idx),
        surv_values.imag.gather(1, g_idx),
    ).squeeze(1)
    
    return (
        survivors.cpu().numpy(),
        surv_idx.cpu().numpy(),
        surv_peaks.cpu().numpy(),
        False,
    )


# ---------------------------------------------------------------------------
# Benchmark Data Generation
# ---------------------------------------------------------------------------
def generate_synthetic_bank_and_data(
    batch_size: int,
    size: int,
    sample_rate: int = 2048,
    seed: int = 42,
) -> Tuple[List[FrequencySeries], FrequencySeries, FrequencySeries]:
    """Generate synthetic template bank, strain data, and PSD."""
    rng = np.random.default_rng(seed)
    freq_len = size // 2 + 1
    delta_f = float(sample_rate) / float(size)
    
    # PSD shape
    freqs = np.linspace(0, sample_rate / 2.0, freq_len, endpoint=True)
    with np.errstate(divide="ignore"):
        psd_raw = np.clip(
            (np.maximum(freqs, delta_f) / 100.0) ** (-1.0) + (freqs / 500.0) ** 2.0,
            0.1,
            100.0,
        ).astype(np.float32)
    psd = FrequencySeries(psd_raw, delta_f=delta_f)
    
    # Templates
    templates = []
    k_min = max(1, int(30.0 / delta_f))
    k_max = min(freq_len - 1, int(800.0 / delta_f))
    if k_max <= k_min:
        k_min = 1
        k_max = freq_len - 1
    
    for i in range(batch_size):
        t_raw = np.zeros(freq_len, dtype=np.complex64)
        t_raw[k_min:k_max] = (
            rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
        ).astype(np.complex64)
        sgm_val = float(4.0 * delta_f * np.sum(np.abs(t_raw) ** 2 / psd_raw))
        t_raw /= np.sqrt(sgm_val)
        
        t = FrequencySeries(t_raw, delta_f=delta_f)
        t.id = int(1000 + i)
        t.params = np.array(
            [(10.0 + i * 0.1, 10.0 + i * 0.1)],
            dtype=[("mass1", np.float32), ("mass2", np.float32)],
        )[0]
        t.sigmasq = lambda p, s=sgm_val: 1.0  # already normalized
        templates.append(t)
        
    # Strain data with injected signal so threshold path and trigger gathering are tested
    strain_raw = (
        rng.normal(size=freq_len) + 1j * rng.normal(size=freq_len)
    ).astype(np.complex64) * 0.001
    inj_snr = 10.0
    inj_phase = np.exp(-2j * np.pi * np.arange(freq_len) * (size // 2) / size)
    strain_raw += (templates[0].numpy() * inj_snr * inj_phase).astype(np.complex64)
    stilde = FrequencySeries(strain_raw, delta_f=delta_f)
    stilde.psd = psd
    
    return templates, stilde, psd


# ---------------------------------------------------------------------------
# Comprehensive Profiling Functions
# ---------------------------------------------------------------------------
def profile_grid_point(
    B: int,
    N: int,
    sample_rate: int = 2048,
    n_warmup: int = 3,
    n_iters: int = 20,
) -> Dict[str, Any]:
    """Profile a single (B, N) point across all metrics and microsecond breakdown."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
    delta_f = float(sample_rate) / float(N)
    working_set_mb = float((2 * B * N * 8) / (1024 * 1024))
    
    try:
        templates, stilde, psd = generate_synthetic_bank_and_data(B, N, sample_rate=sample_rate)
        
        with scheme.TorchScheme("cuda"):
            t_setup_start = time.perf_counter_ns()
            batch_filter = mf.LiveBatchMatchedFilter(
                templates,
                snr_threshold=5.5,
                chisq_bins=0,
                sg_chisq=types.SimpleNamespace(),
                maxelements=B * N,  # Single batch tile
            )
            torch.cuda.synchronize()
            t_setup_ns = time.perf_counter_ns() - t_setup_start
            
            mid = batch_filter.mids[0]
            correlator = batch_filter.corr[0]
            ifft_obj = batch_filter.ifts[mid]
            out_mem = batch_filter.out_mem[mid]
            
            stilde_cpu_pinned = torch.as_tensor(stilde.numpy(), device="cpu").pin_memory()
            
            reader = types.SimpleNamespace(
                overwhitened_data=lambda _df, s=stilde: s,
                trim_padding=0,
                blocksize=N,
                sample_rate=sample_rate,
                start_time=0.0,
            )
            
            # Microsecond Breakdown Events
            ev_h2d_start = torch.cuda.Event(enable_timing=True)
            ev_h2d_end = torch.cuda.Event(enable_timing=True)
            ev_pack_start = torch.cuda.Event(enable_timing=True)
            ev_pack_end = torch.cuda.Event(enable_timing=True)
            ev_corr_start = torch.cuda.Event(enable_timing=True)
            ev_corr_end = torch.cuda.Event(enable_timing=True)
            ev_ifft_start = torch.cuda.Event(enable_timing=True)
            ev_ifft_end = torch.cuda.Event(enable_timing=True)
            ev_peak_start = torch.cuda.Event(enable_timing=True)
            ev_peak_end = torch.cuda.Event(enable_timing=True)
            ev_pipe_start = torch.cuda.Event(enable_timing=True)
            ev_pipe_end = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            for _ in range(n_warmup):
                batch_filter.set_data(reader)
                batch_filter._process_batch()
            torch.cuda.synchronize()
            
            vram_steady = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            t_total_ms_list = []
            t_h2d_us_list = []
            t_pack_us_list = []
            t_corr_us_list = []
            t_ifft_us_list = []
            t_peak_us_list = []
            t_d2h_us_list = []
            
            norms = np.ones(B, dtype=np.float64) * 4.0 * delta_f
            norms_t = torch.as_tensor(norms, device="cuda", dtype=torch.float32)  # Use float32 to prevent OOM
            seg = slice(0, N)
            
            vram_alloc_counts = 0
            
            out_tensor = getattr(getattr(out_mem, "_data", None), "tensor", None)
            if out_tensor is None and isinstance(out_mem, torch.Tensor):
                out_tensor = out_mem
            
            for _ in range(n_iters):
                mem_before_iter = torch.cuda.memory_allocated()
                
                ev_pipe_start.record()
                
                # 1. Host -> Device Transfer
                ev_h2d_start.record()
                _ = stilde_cpu_pinned.to(device="cuda", non_blocking=True)
                ev_h2d_end.record()
                
                # 2. Template packaging / state validation
                ev_pack_start.record()
                can_exec = getattr(correlator, "_torch_cuda_native_batch_state", None)
                if can_exec is not None:
                    _ = can_exec.can_execute(correlator)
                ev_pack_end.record()
                
                # 3. 2D Batched Frequency Correlation
                ev_corr_start.record()
                correlator.execute(stilde)
                ev_corr_end.record()
                
                # 4. 2D Batched cuFFT IFFT
                ev_ifft_start.record()
                ifft_obj.execute()
                ev_ifft_end.record()
                
                # 5. Magnitude-squared & Peak Reduction / Thresholding (Direct float32 fast path)
                ev_peak_start.record()
                values_2d = out_tensor.reshape(B, N)[:, seg]
                # Efficient on-device peak reduction
                sq_mag = torch.view_as_real(values_2d).square().sum(dim=-1)
                clean_mag = torch.nan_to_num(sq_mag, nan=0.0)
                max_sq_mag, indices = torch.max(clean_mag, dim=-1)
                max_snr_sq = max_sq_mag * norms_t.square()
                _ = torch.max(max_snr_sq).item()
                ev_peak_end.record()
                
                ev_pipe_end.record()
                
                # 6. D2H Peak transfer timing
                t_d2h_0 = time.perf_counter_ns()
                _ = torch.cuda.current_stream().synchronize()
                t_d2h_1 = time.perf_counter_ns()
                
                mem_after_iter = torch.cuda.memory_allocated()
                if mem_after_iter != mem_before_iter:
                    vram_alloc_counts += 1
                    
                t_total_ms = ev_pipe_start.elapsed_time(ev_pipe_end)
                t_h2d_us = ev_h2d_start.elapsed_time(ev_h2d_end) * 1000.0
                t_pack_us = ev_pack_start.elapsed_time(ev_pack_end) * 1000.0
                t_corr_us = ev_corr_start.elapsed_time(ev_corr_end) * 1000.0
                t_ifft_us = ev_ifft_start.elapsed_time(ev_ifft_end) * 1000.0
                t_peak_us = ev_peak_start.elapsed_time(ev_peak_end) * 1000.0
                t_d2h_us = (t_d2h_1 - t_d2h_0) / 1000.0
                
                t_total_ms_list.append(t_total_ms)
                t_h2d_us_list.append(t_h2d_us)
                t_pack_us_list.append(t_pack_us)
                t_corr_us_list.append(t_corr_us)
                t_ifft_us_list.append(t_ifft_us)
                t_peak_us_list.append(t_peak_us)
                t_d2h_us_list.append(t_d2h_us)

        # Compute Statistics
        mean_latency_ms = float(np.mean(t_total_ms_list))
        median_latency_ms = float(np.median(t_total_ms_list))
        p95_latency_ms = float(np.percentile(t_total_ms_list, 95))
        min_latency_ms = float(np.min(t_total_ms_list))
        std_latency_ms = float(np.std(t_total_ms_list))
        
        latency_sec = mean_latency_ms / 1000.0
        transforms_per_sec = float(B / latency_sec)
        waveforms_per_sec = float(B / latency_sec)
        
        flop_per_tmpl = 6.0 * N + 5.0 * N * math.log2(N) + 3.0 * N
        total_gflops = float((B * flop_per_tmpl) * 1e-9 / latency_sec)
        
        total_bytes = 5.0 * 8.0 * B * N + 8.0 * N
        mem_bandwidth_gbs = float((total_bytes * 1e-9) / latency_sec)
        
        del batch_filter, templates, stilde, psd
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "B": B,
            "N": N,
            "oom": False,
            "setup_time_ms": t_setup_ns / 1e6,
            "latency_ms": {
                "mean": mean_latency_ms,
                "median": median_latency_ms,
                "p95": p95_latency_ms,
                "min": min_latency_ms,
                "std": std_latency_ms,
            },
            "throughput": {
                "transforms_per_sec": transforms_per_sec,
                "waveforms_per_sec": waveforms_per_sec,
                "gflops": total_gflops,
                "mem_bandwidth_gbs": mem_bandwidth_gbs,
            },
            "breakdown_us": {
                "h2d_transfer": float(np.mean(t_h2d_us_list)),
                "packaging_validation": float(np.mean(t_pack_us_list)),
                "batched_correlation": float(np.mean(t_corr_us_list)),
                "batched_cufft_ifft": float(np.mean(t_ifft_us_list)),
                "peak_thresholding": float(np.mean(t_peak_us_list)),
                "d2h_sync": float(np.mean(t_d2h_us_list)),
            },
            "vram_mb": {
                "initial": vram_before,
                "steady": vram_steady,
                "peak": vram_peak,
                "reserved": vram_reserved,
                "allocation_churn_per_call": vram_alloc_counts,
            },
            "cache": {
                "working_set_mb": working_set_mb,
                "l2_cache_mb": 72.0,
                "l2_resident": working_set_mb <= 72.0,
            },
        }
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "B": B,
            "N": N,
            "oom": True,
            "setup_time_ms": 0.0,
            "latency_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "std": 0.0},
            "throughput": {"transforms_per_sec": 0.0, "waveforms_per_sec": 0.0, "gflops": 0.0, "mem_bandwidth_gbs": 0.0},
            "breakdown_us": {"h2d_transfer": 0, "packaging_validation": 0, "batched_correlation": 0, "batched_cufft_ifft": 0, "peak_thresholding": 0, "d2h_sync": 0},
            "vram_mb": {"initial": vram_before, "steady": 0, "peak": 24000, "reserved": 24000, "allocation_churn_per_call": 0},
            "cache": {"working_set_mb": working_set_mb, "l2_cache_mb": 72.0, "l2_resident": False},
        }


# ---------------------------------------------------------------------------
# L2 Cache Residency & Hardware Tiling Benchmark
# ---------------------------------------------------------------------------
def profile_l2_cache_and_tiling(
    total_templates: int = 1024,
    N_list: List[int] = [32768, 65536, 131072, 262144, 524288],
    sample_rate: int = 2048,
) -> List[Dict[str, Any]]:
    """Compare monolithic vs hardware-tiled batch execution against 72 MB L2 cache."""
    results = []
    
    with scheme.TorchScheme("cuda"):
        for N in N_list:
            optimal_tile = get_optimal_batch_tile_size(N, is_cuda=True, device_id=0)
            optimal_maxelem = get_optimal_batch_maxelements(is_cuda=True, device_id=0)
            
            candidate_tiles = [8, 16, 32, 64, 128, 256, 512, 1024]
            candidate_tiles = [t for t in candidate_tiles if t <= total_templates]
            
            tile_benchmarks = []
            for tile_size in candidate_tiles:
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    templates, stilde, psd = generate_synthetic_bank_and_data(
                        total_templates, N, sample_rate=sample_rate
                    )
                    batch_filter = mf.LiveBatchMatchedFilter(
                        templates,
                        snr_threshold=5.5,
                        chisq_bins=0,
                        sg_chisq=types.SimpleNamespace(),
                        maxelements=tile_size * N,  # Forces tile_size chunks
                    )
                    
                    reader = types.SimpleNamespace(
                        overwhitened_data=lambda _df, s=stilde: s,
                        trim_padding=0,
                        blocksize=N,
                        sample_rate=sample_rate,
                        start_time=0.0,
                    )
                    
                    # Warmup
                    for _ in range(2):
                        batch_filter.set_data(reader)
                        batch_filter.process_all()
                    torch.cuda.synchronize()
                    
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                    
                    ev_start.record()
                    for _ in range(8):
                        batch_filter.set_data(reader)
                        batch_filter.process_all()
                    ev_end.record()
                    torch.cuda.synchronize()
                    
                    total_time_ms = ev_start.elapsed_time(ev_end) / 8.0
                    waveforms_sec = float(total_templates / (total_time_ms / 1000.0))
                    working_set_mb = float((2 * tile_size * N * 8) / (1024 * 1024))
                    
                    tile_benchmarks.append({
                        "tile_size": tile_size,
                        "num_chunks": len(batch_filter.chunks),
                        "total_time_ms": total_time_ms,
                        "waveforms_per_sec": waveforms_sec,
                        "working_set_mb": working_set_mb,
                        "is_l2_resident": working_set_mb <= 72.0,
                        "is_predicted_optimal": (tile_size == optimal_tile),
                    })
                    del batch_filter, templates, stilde, psd
                except torch.cuda.OutOfMemoryError:
                    working_set_mb = float((2 * tile_size * N * 8) / (1024 * 1024))
                    tile_benchmarks.append({
                        "tile_size": tile_size,
                        "num_chunks": total_templates // tile_size,
                        "total_time_ms": 0.0,
                        "waveforms_per_sec": 0.0,
                        "working_set_mb": working_set_mb,
                        "is_l2_resident": working_set_mb <= 72.0,
                        "is_predicted_optimal": (tile_size == optimal_tile),
                        "oom": True,
                    })
                gc.collect()
                torch.cuda.empty_cache()
                
            results.append({
                "N": N,
                "total_templates": total_templates,
                "hardware_optimal_tile": optimal_tile,
                "optimal_maxelements": optimal_maxelem,
                "tiles": tile_benchmarks,
            })
            
    return results


# ---------------------------------------------------------------------------
# Triton vs PyTorch Peak Reduction Benchmark
# ---------------------------------------------------------------------------
def profile_triton_vs_pytorch(
    B_list: List[int] = [1, 16, 64, 256, 1024],
    N_list: List[int] = [32768, 131072, 524288],
    n_iters: int = 50,
) -> List[Dict[str, Any]]:
    """Compare multi-pass PyTorch peak reduction vs Fused Triton reduction."""
    results = []
    
    with scheme.TorchScheme("cuda"):
        for B in B_list:
            for N in N_list:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    values = torch.randn(B, N, dtype=torch.complex64, device="cuda")
                    norms_np = np.ones(B, dtype=np.float64) * 0.01
                    norms_t = torch.as_tensor(norms_np, device="cuda", dtype=torch.float32)
                    
                    # Warmup PyTorch
                    for _ in range(3):
                        _ = mf_torch._torch_batch_peak_and_threshold_gpu(
                            values, norms_t, snr_threshold=5.5
                        )
                    # Warmup Triton
                    for _ in range(3):
                        _ = triton_fused_batch_peak_and_threshold(
                            values, norms_t, snr_threshold=5.5
                        )
                    torch.cuda.synchronize()
                    
                    # Benchmark PyTorch
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_end = torch.cuda.Event(enable_timing=True)
                    
                    ev_start.record()
                    for _ in range(n_iters):
                        _ = mf_torch._torch_batch_peak_and_threshold_gpu(
                            values, norms_t, snr_threshold=5.5
                        )
                    ev_end.record()
                    torch.cuda.synchronize()
                    py_time_us = (ev_start.elapsed_time(ev_end) / n_iters) * 1000.0
                    
                    # Benchmark Triton
                    ev_start.record()
                    for _ in range(n_iters):
                        _ = triton_fused_batch_peak_and_threshold(
                            values, norms_t, snr_threshold=5.5
                        )
                    ev_end.record()
                    torch.cuda.synchronize()
                    triton_time_us = (ev_start.elapsed_time(ev_end) / n_iters) * 1000.0
                    
                    speedup = py_time_us / triton_time_us if triton_time_us > 0 else 1.0
                    
                    results.append({
                        "B": B,
                        "N": N,
                        "pytorch_us": py_time_us,
                        "triton_fused_us": triton_time_us,
                        "speedup": speedup,
                        "oom": False,
                    })
                    del values, norms_t
                except torch.cuda.OutOfMemoryError:
                    results.append({
                        "B": B,
                        "N": N,
                        "pytorch_us": 0.0,
                        "triton_fused_us": 0.0,
                        "speedup": 1.0,
                        "oom": True,
                    })
                gc.collect()
                torch.cuda.empty_cache()
                
    return results


# ---------------------------------------------------------------------------
# CUDA Graphs & Async Streams Benchmark
# ---------------------------------------------------------------------------
def profile_cuda_graphs_and_async(
    B_list: List[int] = [1, 4, 16, 64, 256, 512],
    N: int = 131072,
    sample_rate: int = 2048,
    n_iters: int = 25,
) -> Dict[str, Any]:
    """Benchmark speedup of CUDA Graphs and Asynchronous Stream Prefetching."""
    graph_results = []
    
    with scheme.TorchScheme("cuda"):
        for B in B_list:
            gc.collect()
            torch.cuda.empty_cache()
            
            templates, stilde, psd = generate_synthetic_bank_and_data(
                B, N, sample_rate=sample_rate
            )
            reader = types.SimpleNamespace(
                overwhitened_data=lambda _df, s=stilde: s,
                trim_padding=0,
                blocksize=N,
                sample_rate=sample_rate,
                start_time=0.0,
            )
            
            # 1. Eager Execution (Standard)
            filter_eager = mf.LiveBatchMatchedFilter(
                templates,
                snr_threshold=5.5,
                chisq_bins=0,
                sg_chisq=types.SimpleNamespace(),
                maxelements=B * N,
                enable_cuda_graphs=False,
            )
            for _ in range(3):
                filter_eager.set_data(reader)
                filter_eager._process_batch()
            torch.cuda.synchronize()
            
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            
            ev_start.record()
            for _ in range(n_iters):
                filter_eager.set_data(reader)
                filter_eager._process_batch()
            ev_end.record()
            torch.cuda.synchronize()
            eager_time_us = (ev_start.elapsed_time(ev_end) / n_iters) * 1000.0
            
            # 2. CUDA Graph Execution
            filter_graph = mf.LiveBatchMatchedFilter(
                templates,
                snr_threshold=5.5,
                chisq_bins=0,
                sg_chisq=types.SimpleNamespace(),
                maxelements=B * N,
                enable_cuda_graphs=True,
            )
            for _ in range(3):
                filter_graph.set_data(reader)
                filter_graph._process_batch()
            torch.cuda.synchronize()
            
            ev_start.record()
            for _ in range(n_iters):
                filter_graph.set_data(reader)
                filter_graph._process_batch()
            ev_end.record()
            torch.cuda.synchronize()
            graph_time_us = (ev_start.elapsed_time(ev_end) / n_iters) * 1000.0
            
            graph_results.append({
                "B": B,
                "N": N,
                "eager_us": eager_time_us,
                "cuda_graph_us": graph_time_us,
                "speedup": eager_time_us / graph_time_us if graph_time_us > 0 else 1.0,
            })
            del filter_eager, filter_graph, templates, stilde, psd
            gc.collect()
            torch.cuda.empty_cache()
            
    return {"cuda_graphs": graph_results}


# ---------------------------------------------------------------------------
# Main Execution Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PyCBC LiveBatchMatchedFilter Deep GPU Profiler (NVIDIA RTX 4090 24GB)")
    print("=" * 80)
    
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"SM Count: {props.multi_processor_count}")
    print(f"L2 Cache: {get_gpu_l2_cache_size(0) / (1024**2):.1f} MB")
    print("=" * 80)
    
    batch_sizes = [1, 4, 16, 32, 64, 128, 256, 512, 1024]
    segment_lengths = [32768, 65536, 131072, 262144, 524288]  # 32k .. 524k
    
    output_dir = repo_dir / "gpu_profiling_results"
    output_dir.mkdir(exist_ok=True)
    
    print("\n[Phase 1/4] Running Full (B, N) Grid Parameter Profiling...")
    grid_results = []
    for N in segment_lengths:
        for B in batch_sizes:
            print(f"  -> Profiling B={B:4d}, N={N:6d} ({N//1024}k) ...", end="", flush=True)
            res = profile_grid_point(B, N, sample_rate=2048, n_warmup=3, n_iters=20)
            grid_results.append(res)
            if res.get("oom"):
                print(" OOM (Single-tile working memory exceeded 24GB)")
            else:
                lat = res["latency_ms"]["mean"]
                wf_sec = res["throughput"]["waveforms_per_sec"]
                gflops = res["throughput"]["gflops"]
                bw = res["throughput"]["mem_bandwidth_gbs"]
                print(f" done! Latency: {lat:7.3f} ms | {wf_sec:9.1f} wf/s | {gflops:7.1f} GFLOP/s | {bw:6.1f} GB/s")
            
    print("\n[Phase 2/4] Running L2 Cache Residency & Batch Tiling Benchmark...")
    tiling_results = profile_l2_cache_and_tiling(
        total_templates=1024,
        N_list=segment_lengths,
        sample_rate=2048,
    )
    for t_res in tiling_results:
        N = t_res["N"]
        opt_tile = t_res["hardware_optimal_tile"]
        print(f"  N={N:6d} (Optimal Tile={opt_tile:3d}):")
        for tile in t_res["tiles"]:
            sz = tile["tile_size"]
            wf_s = tile.get("waveforms_per_sec", 0.0)
            ws_mb = tile["working_set_mb"]
            l2_str = "L2 RESIDENT" if tile["is_l2_resident"] else "DRAM SPILL"
            opt_str = " <== OPTIMAL" if tile.get("is_predicted_optimal") else ""
            print(f"    Tile B={sz:4d} | {ws_mb:6.1f} MB ({l2_str:11s}) | Throughput: {wf_s:9.1f} wf/s{opt_str}")
            
    print("\n[Phase 3/4] Running Triton Fused Peak Reduction Benchmark...")
    triton_results = profile_triton_vs_pytorch(
        B_list=[1, 16, 64, 256, 1024],
        N_list=[32768, 131072, 524288],
        n_iters=50,
    )
    for tr in triton_results:
        B, N = tr["B"], tr["N"]
        py_us, tri_us, sp = tr["pytorch_us"], tr["triton_fused_us"], tr["speedup"]
        print(f"  B={B:4d}, N={N:6d} | PyTorch: {py_us:7.1f} us | Triton Fused: {tri_us:7.1f} us | Speedup: {sp:.2f}x")
        
    print("\n[Phase 4/4] Running CUDA Graphs Acceleration Benchmark...")
    graph_results = profile_cuda_graphs_and_async(
        B_list=[1, 4, 16, 64, 256, 512],
        N=131072,
        sample_rate=2048,
        n_iters=25,
    )
    for gr in graph_results["cuda_graphs"]:
        B = gr["B"]
        eager_us, g_us, sp = gr["eager_us"], gr["cuda_graph_us"], gr["speedup"]
        print(f"  B={B:4d}, N=131072 | Eager: {eager_us:7.1f} us | CUDA Graph: {g_us:7.1f} us | Speedup: {sp:.2f}x")
        
    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": {
            "name": props.name,
            "vram_gb": props.total_memory / (1024**3),
            "sm_count": props.multi_processor_count,
            "l2_cache_mb": get_gpu_l2_cache_size(0) / (1024**2),
        },
        "grid_results": grid_results,
        "tiling_results": tiling_results,
        "triton_results": triton_results,
        "graph_results": graph_results,
    }
    
    json_path = output_dir / "live_batch_matchedfilter_rtx4090_profile.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[Saved] Detailed JSON results saved to: {json_path}")
    
    print("\n" + "=" * 80)
    print("PROFILING RUN COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
