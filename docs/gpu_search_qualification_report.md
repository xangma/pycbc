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

---

## 3. Qualification Verification Suite

The entire test suite passed with 100% success across all components:

| Test Module | Coverage Area | Tests Run | Result |
|---|---|:---:|:---:|
| `test_torch_chisq_precision.py` | Precision test fixes, single/double stability | 4 | PASSED |
| `test_torch_sigmasq_series_precision.py` | Normalization factors, equal-power bins | 10 | PASSED |
| `test_torch_strain_psd_precision.py` | PSD estimation, weak frequency retention | 16 | PASSED |
| `test_gpu_search_plans.py` | BankGeometry, BankPlan, PSDPlan, WorkspaceBudget | 6 | PASSED |
| `test_gpu_search_eager_filtering.py` | Eager SearchEngine, candidate buffers, clustering | 16 | PASSED |
| `test_gpu_search_vetoes.py` | Power $\chi^2$, Sine-Gaussian $\chi^2$, VetoManager | 10 | PASSED |
| `test_gpu_search_adapter.py` | Offline & live application adapters, abort, top-K | 8 | PASSED |
| `test_gpu_search_graphs_streams.py` | CUDA graphs, async streams, double buffering | 8 | PASSED |
| `test_gpu_search_qualification.py` | Scientific parity, sustained streaming, memory bounds | 4 | PASSED |
| **Total** | | **82** | **100% PASSED** |

---

## 4. Stacked Pull Requests

The GPU search engine is structured as a linearly stacked series of pull requests targeting `xangma/pycbc`:

1. **PR #21** (`gpu-search-s0-harness`): Test harness, CI test selectors, and precision fixes.
2. **PR #22** (`gpu-search-s1-eager-engine`): Template bank tiling, PSD plans, workspace budgeting, and eager search engine.
3. **PR #23** (`gpu-search-s2-candidates`): Device candidate queues, dynamic buffer resizing, and symmetric clustering.
4. **PR #24** (`gpu-search-s3-vetoes`): Batched full-statistic vetoes (Power $\chi^2$ and Sine-Gaussian $\chi^2$).
5. **PR #25** (`gpu-search-s4-s5-integration`): Search application integration (`pycbc_inspiral` and `pycbc_live` adapters).
6. **PR #26** (`gpu-search-s6-graphs-streams`): CUDA graphs, multi-stream scheduling, and double-buffered workspaces.
7. **PR #27** (`gpu-search-s7-qualification`): End-to-end performance evidence, qualification benchmarks, and qualification report.
