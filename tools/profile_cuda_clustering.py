#!/usr/bin/env python3
"""Comprehensive CUDA profiling script for PyCBC clustering and matched filtering."""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/xangma/pycbc_bench_repo")
import pycbc.events.threshold_torch as tt
from pycbc.types import Array, FrequencySeries, zeros
import pycbc.scheme as scheme
from pycbc.events.simd_threshold_cython import parallel_thresh_cluster
import pycbc.filter.matchedfilter as mf

# Set high-precision timers
def cuda_time(func, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        func()
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    
    for i in range(n_iter):
        start_events[i].record()
        func()
        end_events[i].record()
        
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] # in milliseconds
    return float(np.median(times)), float(np.std(times)), float(np.min(times)), float(np.max(times))

def cpu_time(func, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        func()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # in ms
    return float(np.median(times)), float(np.std(times)), float(np.min(times)), float(np.max(times))

def generate_test_data_np(size, inj_snr=10.0):
    rng = np.random.default_rng(42)
    noise = (rng.normal(scale=1.0, size=size) + 1j * rng.normal(scale=1.0, size=size)).astype(np.complex64)
    center = size // 2
    noise[center] = (inj_snr + 0j)
    return noise

print("==========================================================================================")
print("0. System / Hardware Information")
print("==========================================================================================")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print("==========================================================================================\n")

print("==========================================================================================")
print("1. Standalone TorchThresholdCluster Sub-Operation Breakdown on RTX 4090")
print("==========================================================================================")

sizes = [32768, 65536, 262144, 1048576, 4194304]
windows = [16, 256, 1024, 4096]
thresholds = [("Dense (thr=2.0)", 2.0), ("Standard (thr=5.5)", 5.5), ("Sparse (thr=8.0)", 8.0)]

breakdown_results = {}

with scheme.TorchScheme(device="cuda"):
    for size in sizes:
        breakdown_results[size] = {}
        data_np = generate_test_data_np(size, inj_snr=10.0)
        pycbc_arr = Array(data_np)
        tensor = pycbc_arr._data.tensor
        engine = tt.TorchThresholdCluster(pycbc_arr)
        
        print(f"\n--- Array Size N = {size:,} ({size*8/1024/1024:.2f} MB complex64) ---")
        header = f"{'Window':<8} | {'Regime':<18} | {'GPU Total':<10} | {'MagSq %':<9} | {'CandMax %':<10} | {'SymMask %':<10} | {'SurvIdx %':<10} | {'Wrap(CPU)':<10} | {'Surv Count':<10}"
        print(header)
        print("-" * len(header))
        
        for w in windows:
            for reg_name, thr in thresholds:
                thresh_sq = torch.tensor(thr * thr, device="cuda", dtype=torch.float32)
                
                # Measure individual sub-ops with CUDA events
                # 1. Magnitude squared
                def op_mag():
                    return torch.view_as_real(tensor).square().sum(dim=-1)
                t_mag, _, _, _ = cuda_time(op_mag, n_warmup=5, n_iter=30)
                
                mag_sq = op_mag()
                
                # 2. Cluster candidates
                def op_cand():
                    return tt._cluster_candidates(mag_sq, w)
                t_cand, _, _, _ = cuda_time(op_cand, n_warmup=5, n_iter=30)
                
                block_max, block_idx = op_cand()
                
                # 3. Symmetric cluster mask
                def op_mask():
                    return tt._symmetric_cluster_mask(block_max, thresh_sq)
                t_mask, _, _, _ = cuda_time(op_mask, n_warmup=5, n_iter=30)
                
                keep = op_mask()
                
                # 4. Survivor indexing
                flat_series = tensor.reshape(-1)
                def op_idx():
                    kept_idx = block_idx[keep]
                    kept_vals = flat_series[kept_idx]
                    return kept_vals, kept_idx
                t_idx, _, _, _ = cuda_time(op_idx, n_warmup=5, n_iter=30)
                
                kept_vals, kept_idx = op_idx()
                surv_count = kept_idx.numel()
                
                # 5. PyCBC Array wrapping (host/Python overhead)
                def op_wrap():
                    return tt._array_from_tensor(kept_vals), tt._array_from_tensor(kept_idx)
                t_wrap_cpu, _, _, _ = cpu_time(op_wrap, n_warmup=5, n_iter=30)
                
                # 6. Full end-to-end threshold_and_cluster
                def op_full():
                    return engine.threshold_and_cluster(thr, w)
                t_full_gpu, _, _, _ = cuda_time(op_full, n_warmup=10, n_iter=40)
                t_full_wall, _, _, _ = cpu_time(op_full, n_warmup=10, n_iter=40)
                
                pct_mag = (t_mag / t_full_gpu) * 100
                pct_cand = (t_cand / t_full_gpu) * 100
                pct_mask = (t_mask / t_full_gpu) * 100
                pct_idx = (t_idx / t_full_gpu) * 100
                
                print(f"W={w:<6} | {reg_name:<18} | {t_full_gpu:7.4f} ms | {pct_mag:7.1f}% | {pct_cand:8.1f}% | {pct_mask:8.1f}% | {pct_idx:8.1f}% | {t_wrap_cpu:7.4f} ms | {surv_count:<10}")
                
                breakdown_results[size][(w, thr)] = {
                    "t_full_gpu_ms": t_full_gpu,
                    "t_full_wall_ms": t_full_wall,
                    "t_mag_ms": t_mag,
                    "t_cand_ms": t_cand,
                    "t_mask_ms": t_mask,
                    "t_idx_ms": t_idx,
                    "t_wrap_ms": t_wrap_cpu,
                    "surv_count": surv_count,
                }

print("\n==========================================================================================")
print("2. Host-Device Synchronization, Data Transfer Latency & Memory Churn")
print("==========================================================================================")

header2 = f"{'Array Size':<12} | {'Alloc (MB)':<11} | {'Peak Alloc (MB)':<16} | {'CUDA E2E (ms)':<14} | {'Host Sync (ms)':<15} | {'D2H Transfer (ms)':<18} | {'Transfer %':<11}"
print(header2)
print("-" * len(header2))

with scheme.TorchScheme(device="cuda"):
    for size in sizes:
        data_np = generate_test_data_np(size, inj_snr=10.0)
        pycbc_arr = Array(data_np)
        engine = tt.TorchThresholdCluster(pycbc_arr)
        w = 256
        thr = 5.5
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        vals_arr, idx_arr = engine.threshold_and_cluster(thr, w)
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
        mem_after = torch.cuda.memory_allocated() / (1024**2)
        
        # Measure async GPU call latency
        def op_async():
            engine.threshold_and_cluster(thr, w)
        t_gpu_e2e, _, _, _ = cuda_time(op_async, n_warmup=10, n_iter=50)
        
        # Measure host sync overhead
        def op_sync():
            engine.threshold_and_cluster(thr, w)
            torch.cuda.synchronize()
        t_sync_wall, _, _, _ = cpu_time(op_sync, n_warmup=10, n_iter=50)
        host_sync_overhead = max(0.0, t_sync_wall - t_gpu_e2e)
        
        # Measure D2H transfer of output survivors
        vals_t = vals_arr._data.tensor
        idx_t = idx_arr._data.tensor
        def op_d2h():
            v_cpu = vals_t.cpu()
            i_cpu = idx_t.cpu()
            return v_cpu, i_cpu
        t_d2h, _, _, _ = cuda_time(op_d2h, n_warmup=10, n_iter=50)
        
        pct_d2h = (t_d2h / (t_gpu_e2e + t_d2h)) * 100.0
        print(f"N={size:<10} | {mem_after:8.2f} MB | {mem_peak:13.2f} MB | {t_gpu_e2e:11.4f} ms | {t_sync_wall:12.4f} ms | {t_d2h:15.4f} ms | {pct_d2h:8.1f}%")

print("\n==========================================================================================")
print("3. FindChirp Clustering & F.max_pool1d Profiling (CUDA vs CPU)")
print("==========================================================================================")

fc_spans = [10000, 100000, 1000000, 10000000]
fc_windows = [16, 256, 1024, 4096]

print("\n--- F.max_pool1d Standalone Benchmark: RTX 4090 vs 16-thread CPU ---")
header3a = f"{'Span (Samples)':<15} | {'Window (Samples)':<17} | {'RTX 4090 GPU (ms)':<18} | {'CPU (16T) (ms)':<16} | {'GPU Speedup':<12}"
print(header3a)
print("-" * len(header3a))

for span in fc_spans:
    for win in [16, 256, 1024]:
        # GPU pool
        gpu_data = torch.randn((1, 1, span), device="cuda", dtype=torch.float32)
        gpu_padded = F.pad(gpu_data, (0, win), value=float("-inf"))
        def pool_gpu():
            return F.max_pool1d(gpu_padded, kernel_size=win + 1, stride=1)
        t_pool_gpu, _, _, _ = cuda_time(pool_gpu, n_warmup=5, n_iter=30)
        
        # CPU pool
        cpu_data = torch.randn((1, 1, span), device="cpu", dtype=torch.float32)
        cpu_padded = F.pad(cpu_data, (0, win), value=float("-inf"))
        def pool_cpu():
            return F.max_pool1d(cpu_padded, kernel_size=win + 1, stride=1)
        t_pool_cpu, _, _, _ = cpu_time(pool_cpu, n_warmup=3, n_iter=10 if span >= 1000000 else 20)
        
        speedup = t_pool_cpu / t_pool_gpu
        print(f"{span:<15,} | {win:<17} | {t_pool_gpu:15.4f} ms | {t_pool_cpu:13.4f} ms | {speedup:9.2f}x")

print("\n--- Full FindChirp Clustering (_threshold_and_cluster_findchirp) ---")
header3b = f"{'Array Size':<12} | {'Window':<8} | {'Density':<10} | {'Candidates':<11} | {'Survivors':<10} | {'CUDA E2E (ms)':<14} | {'CPU E2E (ms)':<13} | {'Speedup':<10}"
print(header3b)
print("-" * len(header3b))

for size in [65536, 262144, 1048576]:
    for thr_name, thr in [("Dense (2.0)", 2.0), ("Std (5.5)", 5.5), ("Sparse (8.0)", 8.0)]:
        win = 256
        data_np = generate_test_data_np(size, inj_snr=10.0)
        t_gpu = torch.from_numpy(data_np).cuda()
        t_cpu = torch.from_numpy(data_np).cpu()
        
        # Count candidates
        mask_gpu, _ = tt._threshold_mask(t_gpu, thr)
        cand_count = int(mask_gpu.sum().item())
        
        def run_fc_gpu():
            return tt._threshold_and_cluster_findchirp(t_gpu, thr, win)
        t_fc_gpu, _, _, _ = cuda_time(run_fc_gpu, n_warmup=5, n_iter=25)
        
        def run_fc_cpu():
            return tt._threshold_and_cluster_findchirp(t_cpu, thr, win)
        t_fc_cpu, _, _, _ = cpu_time(run_fc_cpu, n_warmup=2, n_iter=10)
        
        surv_times, surv_vals = run_fc_gpu()
        surv_count = len(surv_times)
        speedup = t_fc_cpu / t_fc_gpu
        print(f"N={size:<10} | W={win:<6} | {thr_name:<10} | {cand_count:<11} | {surv_count:<10} | {t_fc_gpu:11.4f} ms | {t_fc_cpu:10.4f} ms | {speedup:7.2f}x")

print("\n==========================================================================================")
print("4. CUDA vs CPU Crossover Points and Throughput Scaling (Symmetric Clustering)")
print("==========================================================================================")

header4 = f"{'Size (N)':<10} | {'Duration':<10} | {'Native Cython CPU (ms)':<23} | {'Torch CPU (16T) (ms)':<21} | {'Torch CUDA RTX4090 (ms)':<24} | {'CUDA vs Nat. CPU':<17} | {'CUDA vs Torch CPU':<17}"
print(header4)
print("-" * len(header4))

for size in [32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]:
    dur_sec = size / 2048.0
    w = 1024
    thr = 5.5
    data_np = generate_test_data_np(size, inj_snr=10.0)
    
    # 1. Native CPU Cython engine
    cands = (size + w - 1) // w
    val_buf = np.empty(cands, dtype=np.complex64)
    idx_buf = np.empty(cands, dtype=np.uint32)
    def run_native_cpu():
        return parallel_thresh_cluster(data_np, size, val_buf, idx_buf, np.float32(thr), w, 32768)
    t_nat_cpu, _, _, _ = cpu_time(run_native_cpu, n_warmup=5, n_iter=30)
    
    # 2. Torch CPU Eager
    t_cpu_tensor = torch.from_numpy(data_np)
    def run_torch_cpu():
        thresh_sq = torch.tensor(thr*thr, dtype=torch.float32)
        return tt._fixed_shape_threshold_core(t_cpu_tensor, thresh_sq, w)
    t_torch_cpu, _, _, _ = cpu_time(run_torch_cpu, n_warmup=5, n_iter=30)
    
    # 3. Torch CUDA Eager
    with scheme.TorchScheme(device="cuda"):
        arr_cuda = Array(data_np)
        engine_cuda = tt.TorchThresholdCluster(arr_cuda)
        def run_torch_cuda():
            return engine_cuda.threshold_and_cluster(thr, w)
        t_torch_cuda, _, _, _ = cuda_time(run_torch_cuda, n_warmup=10, n_iter=50)
    
    speedup_nat = t_nat_cpu / t_torch_cuda
    speedup_tcpu = t_torch_cpu / t_torch_cuda
    
    print(f"N={size:<8} | {dur_sec:6.1f}s   | {t_nat_cpu:18.4f} ms   | {t_torch_cpu:18.4f} ms   | {t_torch_cuda:19.4f} ms   | {speedup_nat:14.2f}x   | {speedup_tcpu:14.2f}x")

print("\n==========================================================================================")
print("5. Compilation & Kernel Fusion Viability (torch.compile on RTX 4090)")
print("==========================================================================================")

header5 = f"{'Size (N)':<10} | {'Eager Core (ms)':<16} | {'Compiled Core (ms)':<19} | {'Fusion Speedup':<15}"
print(header5)
print("-" * len(header5))

for size in [65536, 262144, 1048576, 4194304]:
    w = 1024
    thr = 5.5
    data_np = generate_test_data_np(size, inj_snr=10.0)
    t_cuda = torch.from_numpy(data_np).cuda()
    thresh_sq = torch.tensor(thr*thr, device="cuda", dtype=torch.float32)
    
    # Eager core
    def eager_core():
        return tt._fixed_shape_threshold_core(t_cuda, thresh_sq, w)
    t_eager, _, _, _ = cuda_time(eager_core, n_warmup=10, n_iter=40)
    
    # Compiled core with torch.compile
    try:
        def core_wrapper(tensor, thresh):
            return tt._fixed_shape_threshold_core(tensor, thresh, w)
        compiled_fn = torch.compile(core_wrapper, backend="inductor", mode="default")
        # Warmup compile
        for _ in range(5):
            compiled_fn(t_cuda, thresh_sq)
        torch.cuda.synchronize()
        
        def compiled_core():
            return compiled_fn(t_cuda, thresh_sq)
        t_compiled, _, _, _ = cuda_time(compiled_core, n_warmup=10, n_iter=40)
        speedup = t_eager / t_compiled
        print(f"N={size:<8} | {t_eager:14.4f} ms | {t_compiled:16.4f} ms   | {speedup:12.2f}x")
    except Exception as e:
        print(f"N={size:<8} | torch.compile error: {e}")

print("\n==========================================================================================")
print("6. Matched Filtering End-to-End Component Breakdown on CUDA")
print("==========================================================================================")

header6 = f"{'Size (N)':<8} | {'Total (ms)':<11} | {'Correlate (ms)':<15} | {'IFFT (ms)':<11} | {'Cluster (ms)':<13} | {'Corr %':<8} | {'IFFT %':<8} | {'Clust %':<8}"
print(header6)
print("-" * len(header6))

for size in [65536, 131072, 262144, 524288]:
    sample_rate = 2048
    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / (size * delta_t)
    freq_len = size // 2 + 1
    w = int(0.5 * sample_rate)
    
    with scheme.TorchScheme(device="cuda"):
        data_np = generate_test_data_np(freq_len, inj_snr=10.0)
        seg = FrequencySeries(Array(data_np), delta_f=delta_f)
        seg.analyze = slice(int(8 * sample_rate), int(size - 8 * sample_rate))
        
        tmpl_mem = zeros(size, dtype=np.complex64)
        tmpl_mem[:freq_len] = Array(data_np)
        
        mfc = mf.MatchedFilterControl(
            low_frequency_cutoff=30.0,
            high_frequency_cutoff=800.0,
            snr_threshold=5.5,
            tlen=size,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg],
            template_output=tmpl_mem,
            use_cluster=True,
            downsample_factor=1,
            cluster_function="symmetric",
        )
        
        # 1. Total E2E
        def run_full():
            mfc.full_matched_filter_and_cluster_symm(0, 1.0, w)
        t_full, _, _, _ = cuda_time(run_full, n_warmup=10, n_iter=40)
        
        # 2. Correlator only
        def run_corr():
            mfc.correlators[0].correlate()
        t_corr, _, _, _ = cuda_time(run_corr, n_warmup=10, n_iter=40)
        
        # 3. IFFT only
        def run_ifft():
            mfc.ifft.execute()
        t_ifft, _, _, _ = cuda_time(run_ifft, n_warmup=10, n_iter=40)
        
        # 4. Cluster only
        norm = (4.0 * mfc.delta_f) / 1.0
        def run_clust():
            mfc.threshold_and_clusterers[0].threshold_and_cluster(mfc.snr_threshold / norm, w)
        t_clust, _, _, _ = cuda_time(run_clust, n_warmup=10, n_iter=40)
        
        pct_corr = (t_corr / t_full) * 100
        pct_ifft = (t_ifft / t_full) * 100
        pct_clust = (t_clust / t_full) * 100
        
        print(f"N={size:<6} | {t_full:8.4f} ms | {t_corr:12.4f} ms | {t_ifft:8.4f} ms | {t_clust:10.4f} ms | {pct_corr:6.1f}% | {pct_ifft:6.1f}% | {pct_clust:6.1f}%")

print("\n==========================================================================================")
print("PROFILING COMPLETE")
print("==========================================================================================")
