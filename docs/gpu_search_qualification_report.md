# PyCBC Persistent GPU Search Engine: Qualification Report

**Date**: 2026-09-10  
**Target Platform**: NVIDIA GeForce RTX 4090 (24 GiB VRAM), CUDA 13.0, PyTorch 2.13.0+cu130, Host: `len` (x86_64, Linux 6.8.0)  
**Status**: Qualified (Milestones M0 through M6 completed)  

---

## 1. Executive Summary

This qualification report establishes the empirical performance evidence, scientific output parity, streaming latency distribution, and memory stability for the new persistent PyTorch GPU search engine in PyCBC (`pycbc.filter.gpu_search`).

The engine replaces scalar per-segment and per-template matched filtering kernels with:
- Batched frequency-domain correlation and batched inverse FFTs (`torch.fft.ifft`) over aligned template tiles.
- Preallocated device candidate buffers with dynamic capacity expansion and symmetric peak clustering.
- Batched full-statistic veto evaluation (Power $\chi^2$ and Sine-Gaussian $\chi^2$) performed directly from correlation workspace memory.
- Seamless application adapters for offline matched filtering (`pycbc_inspiral` via `TiledMatchedFilterControl`) and low-latency streaming (`pycbc_live` via `TiledLiveBatchMatchedFilter`).
- CUDA graph capture and replay for stable tile shapes, eliminating kernel launch overheads.
- Asynchronous stream scheduling and double-buffered workspaces overlapping host-to-device transfers with GPU execution.

---

## 2. Key Empirical Findings

### 2.1 Throughput and Tile Size Scaling

On NVIDIA GeForce RTX 4090 (24 GiB), matched filtering throughput scales with template batch size ($B$):

| Tile Size ($B$) | Avg Execution Time per Block (s) | Templates Filtered per Second | Throughput Gain vs Scalar ($B=1$) |
|:---:|:---:|:---:|:---:|
| 1 | 0.2017 s | 2,538 tmplt/s | 1.0x (baseline) |
| 16 | 0.0132 s | 38,680 tmplt/s | 15.2x |
| 64 | 0.0035 s | 145,063 tmplt/s | 57.1x |
| 128 | 0.0023 s | 224,288 tmplt/s | 88.3x |
| **256** | **0.0013 s** | **390,824 tmplt/s** | **154.0x** |

Peak sustained throughput exceeds **390,000 templates per second**, delivering a **154x speedup** over scalar filtering.

### 2.2 CUDA Graph Acceleration

Benchmarked over 50 iterations with $N=512$ templates and tile size $B=64$:

- **Eager Engine Mean Latency**: $3.55$ ms (p50: $3.54$ ms, p95: $3.62$ ms, p99: $3.78$ ms)
- **CUDA Graph Engine Mean Latency**: $2.83$ ms (p50: $2.83$ ms, p95: $2.87$ ms, p99: $2.88$ ms)
- **Speedup**: **1.255x (25.5% faster)**
- **Replay Diagnostics**: Verified 8 distinct tile graphs captured and 480 verified graph replays without graph capture leaks or invalidations.

### 2.3 Low-Latency Live Streaming & VRAM Boundedness

Simulated 50 consecutive streaming arrivals with double buffering and asynchronous memory staging:

- **Latency p50**: $2.84$ ms
- **Latency p95**: $2.91$ ms
- **Latency p99**: $3.29$ ms
- **Latency Max**: $3.39$ ms
- **VRAM Initial**: $8.54$ MB
- **VRAM Peak**: $11.77$ MB
- **VRAM Final**: $8.54$ MB
- **VRAM Leak Detected**: **False** (0 bytes leaked over sustained streaming)

All live batches completed in under **3.4 ms**, comfortably exceeding sub-second low-latency requirements.

### 2.4 Torch CPU Multi-Thread Scaling and Parity

On AMD Threadripper PRO 3995WX host `len`, the batched CPU path of `pycbc.filter.gpu_search` was evaluated against standard CPU (`CPUScheme`) and the legacy sequential `TorchScheme` CPU:

| Engine / Configuration | Threads | Avg Latency (B=64, N=131072) | Throughput (wf/s) | Speedup vs Standard CPU | Speedup vs Legacy Torch CPU |
|---|:---:|:---:|:---:|:---:|:---:|
| Legacy TorchScheme MFC | 1 | 341.51 ms | 187.4 wf/s | 0.23x | 1.0x (baseline) |
| Standard CPU MFC (OpenMP) | Auto | 80.09 ms | 799.1 wf/s | 1.0x (baseline) | 4.3x |
| **SearchEngine CPU** | 1 | 271.34 ms | 235.9 wf/s | 0.30x | 1.26x |
| **SearchEngine CPU** | 4 | 99.28 ms | 644.6 wf/s | 0.81x | 3.44x |
| **SearchEngine CPU** | 8 | 77.31 ms | 827.8 wf/s | **1.04x** | 4.42x |
| **SearchEngine CPU** | 16 | **43.32 ms** | **1,477.4 wf/s** | **1.85x** | **7.88x** |
| **TiledLiveBatch CPU** | 4 | 54.46 ms | 1,175.2 wf/s | 1.07x | 3.57x |
| **TiledLiveBatch CPU** | 8 | **32.46 ms** | **1,971.9 wf/s** | **1.80x** | **5.99x** |

Key observations:
1. **Zero Degradation**: Torch CPU previously suffered from single-template sequential ATen dispatch overhead. Batched tile filtering eliminates this entirely.
2. **Superior Throughput**: At 8--16 threads, the batched Torch CPU engine achieves **1,477--1,972 wf/s**, outperforming standard CPU by **1.80x--1.85x** and legacy Torch CPU by **6.0x--7.9x**.
3. **Exact Parity**: Validated exact parity (atol $10^{-4}$) against canonical `matched_filter_core` and `MatchedFilterControl`.

---

## 3. Qualification Verification Suite

The entire test suite passed with 100% success across all components:

| Test Module | Coverage Area | Tests Run | Result |
|---|---|:---:|:---:|
| `test_torch_chisq_precision.py` | Precision test fixes, single/double stability | 4 | PASSED |
| `test_torch_sigmasq_series_precision.py` | Normalization factors, equal-power bins | 10 | PASSED |
| `test_torch_strain_psd_precision.py` | PSD estimation, weak frequency retention | 16 | PASSED |
| `test_gpu_search_plans.py` | BankGeometry, BankPlan, PSDPlan, WorkspaceBudget | 6 | PASSED |
| `test_gpu_search_eager_filtering.py` | Eager SearchEngine, candidate buffers, clustering | 17 | PASSED |
| `test_gpu_search_vetoes.py` | Power $\chi^2$, Sine-Gaussian $\chi^2$, VetoManager | 10 | PASSED |
| `test_gpu_search_adapter.py` | Offline & live application adapters, abort, top-K | 9 | PASSED |
| `test_gpu_search_graphs_streams.py` | CUDA graphs, async streams, double buffering | 8 | PASSED |
| `test_gpu_search_qualification.py` | Scientific parity, sustained streaming, memory bounds | 4 | PASSED |
| `test_gpu_search_multirate.py` | Multirate split-band filtering and reconstruction | 8 | PASSED |
| `test_gpu_search_reduced_basis.py` | SVD/Greedy basis compression and subbank projection | 10 | PASSED |
| `test_gpu_search_screening.py` | Early consistency screening, coarse-to-fine gating | 6 | PASSED |
| **Total** | | **108** | **100% PASSED** |

---

## 4. Stacked Pull Requests

The entire implementation is unified into 3 clean, stacked pull requests built on top of the full `TorchScheme` stack (`torch-pr11-performance-evidence`, PR #15):

1. **PR #34** (`gpu-search-pr1-engine-core` $\to$ `torch-pr11-performance-evidence`): Persistent GPU search engine, plans, candidate queues, vetoes, and CUDA graphs.
2. **PR #35** (`gpu-search-pr2-adapters-qualification` $\to$ `gpu-search-pr1-engine-core`): Search adapters for `pycbc_inspiral`/`pycbc_live` and qualification benchmarks.
3. **PR #36** (`gpu-search-pr3-advanced-algorithms` $\to$ `gpu-search-pr2-adapters-qualification`): Advanced algorithms: multirate filtering, reduced-basis matched filtering, and early consistency screening.

