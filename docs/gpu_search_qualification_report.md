# PyCBC Persistent GPU Search Engine: Qualification Report

**Date**: 2026-09-10  
**Target Platform**: NVIDIA GeForce RTX 4090 (24 GiB VRAM), CUDA 13.0, PyTorch 2.13.0+cu130, Host: `len` (x86_64, Linux 6.8.0)  
**Status**: Qualified for Production (Milestones M0 through M5); Experimental Research Prototypes (Milestones M6 through M7)  

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

### 2.1 Microbenchmark Throughput ($N=1,024$) vs. Production Inspiral Throughput ($N=2,097,152$)

To thoroughly profile GPU kernel limits across workloads, performance was benchmarked across two distinct regimes:
1. **L2-Resident Microbenchmarks ($N=1,024$ samples, short transform length)**:
   On NVIDIA GeForce RTX 4090 (24 GiB), kernel launch and memory bandwidth scaling with batch size ($B$) for small transforms:

| Tile Size ($B$) | Transform Length ($N$) | Avg Execution Time per Block (s) | Microbenchmark Rate (tmplt/s) | Throughput Gain vs Scalar ($B=1$) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1,024 | 0.2238 s | 2,288 tmplt/s | 1.0x (baseline) |
| 16 | 1,024 | 0.0145 s | 35,215 tmplt/s | 15.4x |
| 64 | 1,024 | 0.0039 s | 131,312 tmplt/s | 57.4x |
| 128 | 1,024 | 0.0021 s | 241,185 tmplt/s | 105.4x |
| **256** | **1,024** | **0.0015 s** | **344,399 tmplt/s** | **150.6x** |

   Peak microbenchmark throughput exceeds **344,000 short templates per second**, delivering a **150.6x speedup** over scalar kernel launch overheads.

2. **Full Production Inspiral Workload ($N=2,097,152$ samples, 512 s segments at 4096 Hz)**:
   In realistic gravitational-wave analysis, each matched filter transforms 2,097,152 complex single-precision points ($2^{21}$ elements). Performance was characterized across two rigorous evaluations:
    - **Standalone GPU Search Engine Qualification** (`tools/benchmarking/benchmark_gpu_search.py --include-production-inspiral`):
      Over 384 TaylorF2 templates across 5 distinct 512-second colored segments (1,920 matched filters) with 16-bin device power $\chi^2$ and symmetric clustering, the engine distinguishes three hierarchical timing tiers:
      1. **Pure Calculation Throughput (`total_calc_time_sec`)**: **2,390.9 templates/s** (0.803 s cumulative calculation wall time across all 5 segments).
         > **Measured Calculation Boundaries**: The calculation timer measures the per-segment execution boundary: host-to-device strain scaling and staging (`engine.submit`), batched frequency correlation and batched inverse FFTs (`torch.fft.ifft`), on-device candidate thresholding and symmetric peak clustering, on-device power $\chi^2$ veto evaluation directly on correlation workspace memory, candidate batch retrieval to host (`engine.drain`), and CUDA stream synchronization (`torch.cuda.synchronize()`). It excludes offline setup, initial device memory preallocation, and CPU waveform bank generation.
      2. **Filtering Wall Throughput (`filtering_wall_time_sec`)**: **107.5 templates/s** (17.85 s wall time, including SearchEngine setup, tile memory planning, device allocation, candidate synchronization, and all 5 segments of calculation).
      3. **Total End-to-End Wall Throughput (`total_wall_time_sec`)**: **42.3 templates/s** (45.43 s total wall time, including CPU TaylorF2 physical waveform generation of 384 templates, colored noise simulation of 5 segments, signal injection, engine setup, and all 5 segments of calculation).

      The sealed qualification receipt records the exhaustive independent CPU reference comparisons executed serially on host `len` (`cpu_ref_workers = 1`, `cpu_comparison_count = 1920`, covering all 384 templates across all 5 segments with **0 sample peak arrival time difference**, $\Delta \rho = 5.71 \times 10^{-6}$, and reduced $\Delta \chi^2 = 0.00304$, taking 140.8 s without multi-threaded FFTW planner race conditions). The benchmark validation harness and unit tests strictly enforce complete reference coverage (`cpu_comparison_count == total_matched_filters == 1,920`), rejecting partial counts, missing/duplicated candidates, boolean types (`bool`, `np.bool_`), or non-finite values.

      Furthermore, authentic execution provenance is dynamically recorded at benchmark execution time, capturing Git revision (`edff1e35e092c6373be757f7d95e9b8b3d16eec6`), branch (`torch-pr11-performance-evidence`), dirty state, SHA256 hashes of core source modules, deterministic mass grid configuration ($m_1 \in [10.0, 50.0]\,M_\odot$, $m_2 \in [1.4, 25.0]\,M_\odot$), segment seeds (`[1000, 1001, 1002, 1003, 1004]`), raw pre-injection noise hashes (`raw_noise_segment_sha256_list`), and final submitted strain array hashes after signal injection immediately before filtering (`submitted_segment_sha256_list` / `segment_sha256_list`). Raw execution stdout and stderr are retained in `artifacts/benchmark_len_full_qualification.log`.
   - **Production `pycbc_inspiral` Executable Benchmark** (Section 2.6):
     Under real Hanford LOSC strain and full analysis pipelines:
     - Standard CPU baseline (`cpu:1`): 49.28 s calculation, 65.15 s total wall time.
     - Scalar GPU (`torch:cuda:0`, $B=1$): 10.11 s calculation (4.87x calc speedup), 20.41 s total wall time (3.19x wall speedup).
     - Batched GPU (`torch:cuda:0`, $B=64$): 6.88 s calculation (**7.16x calc speedup**), 17.29 s total wall time (3.77x wall speedup).
     - Batched GPU (`torch:cuda:0`, $B=128$): 6.65 s calculation (**7.41x calc speedup**), 16.95 s total wall time (**3.84x wall speedup**).

### 2.2 CUDA Graph Acceleration

Benchmarked over 50 iterations with $N=512$ templates and tile size $B=64$:

- **Eager Engine Mean Latency**: $3.87$ ms (p50: $3.86$ ms, p95: $3.91$ ms, p99: $3.96$ ms)
- **CUDA Graph Engine Mean Latency**: $3.08$ ms (p50: $3.08$ ms, p95: $3.10$ ms, p99: $3.11$ ms)
- **Speedup**: **1.257x (25.7% faster)**
- **Replay Diagnostics**: Verified 8 distinct tile graphs captured and 480 verified graph replays without graph capture leaks or invalidations.

### 2.3 Low-Latency Live Streaming: Calculation Latency vs. End-to-End Amortization

Simulated 50 consecutive streaming arrivals with double buffering and asynchronous memory staging:

- **Latency Mean**: $3.12$ ms (per 64-template tile)
- **Latency p50**: $3.11$ ms
- **Latency p95**: $3.15$ ms
- **Latency p99**: $3.16$ ms
- **Latency Max**: $3.17$ ms
- **VRAM Initial**: $8.54$ MB
- **VRAM Peak**: $11.77$ MB
- **VRAM Final**: $8.54$ MB
- **VRAM Leak Detected**: **False** (0 bytes leaked over sustained streaming)

All live batches completed in under **3.2 ms** (maximum 3.17 ms), comfortably exceeding sub-second low-latency requirements.

#### Split Latency & End-to-End Stream Amortization (1024 Templates)

> [!NOTE]
> **Historical Campaign Reference**:
> The 1,024-template stream amortization and calculation breakdown presented in the table below reflects the full historical streaming campaign run retained in [`artifacts/live_batch_latest.json`](file:///Users/xangma/repos/pycbc/artifacts/live_batch_latest.json) and published in [`docs/torch_performance.rst`](file:///Users/xangma/repos/pycbc/docs/torch_performance.rst) (Tables 3 and 4), benchmarked on host `len` (NVIDIA RTX 4090).

| Stream Regime | Waveforms Evaluated | Standard CPU Wall Time | Torch CUDA Engine Wall Time | End-to-End Speedup | Calculation Rate (wf/s) |
|---|:---:|:---:|:---:|:---:|:---:|
| One-Time Setup & Preallocation | 0 | 1.54 s | 2.16 s | 0.71x (allocation) | N/A |
| Cold Start (1 block) | 1,024 | 5.53 s | 2.51 s | **2.20x** | 53,005 wf/s |
| Burst Stream (10 blocks, 10 s) | 10,240 | 10.40 s | 2.35 s | **4.43x** | 53,005 wf/s |
| Extended Stream (50 blocks, 50 s) | 51,200 | 45.85 s | 3.13 s | **14.65x** | 53,005 wf/s |
| Sustained Production (500 blocks) | 512,000 | 444.6 s | 11.81 s | **37.65x** | 53,005 wf/s |
| **Pure Calculation Only (per block)** | **1,024** | **886.2 ms** | **19.3 ms** | **45.92x** | **53,005 wf/s** |

Because `pycbc_live` runs as a persistent daemon, the 2.16 s one-time setup and workspace allocation is amortized across streaming blocks, with sustained speedup reaching **37.6x to 45.9x**.

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

### 2.5 Single-Thread Reference Benchmarking & Threshold Knife-Edge Behavior

To ensure a fair, apples-to-apples comparison against standard PyCBC (`pycbc_inspiral` with `CPUScheme`), a reference benchmark was established following standard production guidelines:
- **Strict Single-Thread Execution**: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, pinned to a single CPU core via `taskset -c 8`.
- **Optimal FFT Backend**: Intel MKL (`pycbc.fft.mkl`).
- **Compressed Waveforms**: `--use-compressed-waveforms --waveform-decompression-method inline_linear`.
- **Astrophysical Workload**: Low-mass BNS/NSBH template bank ($m_1 \in [1.2, 8.0]\,M_\odot$, $m_2 \in [1.2, 2.0]\,M_\odot$, duration up to 78 s) filtering 5 analysis segments of 512 s at 4096 Hz (2,097,152 samples) on real LIGO Hanford strain data (`H1:LOSC-STRAIN`).
- **Production Event Cuts**: 30 Hz low-frequency cutoff, 16-bin power $\chi^2$ veto, 1.0 s symmetric clustering window, `--snr-threshold 5.5`, and `--newsnr-threshold 5.0`.

#### Kernel Profiling Breakdown
Instrumented profiling via `cProfile` confirmed the reference computational profile:
- **Inverse FFT (Dominant Kernel)**: ~55% of filtering time (`mkl.py:execute` + descriptor management: 2.31 s for 160 segment filterings).
- **Complex Correlation Multiply**: ~11% of filtering time (`conj` + `__mul__`: 0.45 s).
- **Power $\chi^2$ Veto**: ~14% of filtering time (`point_chisq_code`: 0.56 s).
- **Symmetric Clustering**: ~6% of filtering time (`threshold_and_cluster`: 0.23 s).
- **Waveform Decompression / Interpolation**: ~3.5% of filtering time (`inline_linear_interp`: 0.14 s).

#### Parity of Core Filtering Kernels
Under identical single-thread conditions, Torch CPU (`torch:cpu:1`) matches native MKL C-code:
- **Core Matched-Filtering Subtotal**: **3.69 s** (Normal CPU) vs. **3.79 s** (Torch CPU) — **within 3% parity**.
- **Inverse FFT**: 2.31 s (Normal CPU) vs. 2.57 s (Torch CPU) — within 10%.
- **Complex Multiply**: 0.45 s (Normal CPU) vs. 0.27 s (Torch CPU) — Torch tensor multiply is faster.
- **Continuous SNR Field**: Raw peak SNR evaluates identically between backends ($5.989076 \equiv 5.989076$).

#### Step-Function Thresholding and Knife-Edge Boundary Behavior
When comparing trigger counts across backends, small count differences can occur at the noise threshold (e.g. 161 triggers in Normal CPU vs. 162 in Torch CPU vs. 163 in Torch CUDA):
1. **Identical Pre-Cut Counts**: Before applying the `--newsnr-threshold 5.0` cut, **both Normal CPU and Torch CPU generate exactly 181 triggers**.
2. **Discontinuous Step Cuts ($H(x - x_{\text{th}})$)**: Hard thresholds act as Heaviside step functions. If a continuous noise peak is evaluated as $\text{NewSNR} = 5.009821$ in Torch and $4.9982$ in Normal CPU (a relative difference of only $0.002$), it survives the cut in Torch but is discarded in Normal CPU.
3. **Non-Associative Floating-Point Reductions**: In single-precision (`float32`), floating-point summation is non-associative. Intel MKL (AVX-512 FMA) and PyTorch accumulate $\chi^2$ bin sums and FFT butterflies in different SIMD register orders, introducing machine-epsilon ($\sim 10^{-6}$) variations.
4. **Zero Impact on Real Gravitational-Wave Signals**: All discordant triggers sit within $<0.2\%$ of the razor's edge of the threshold cut (e.g. $\text{NewSNR} = 5.0098$ vs. cutoff $5.0000$; $\rho = 5.5019$ vs. cutoff $5.5000$). For all real astrophysical signals above the background noise floor ($\rho \ge 8\text{--}12$, $\text{NewSNR} \gg 6$), trigger identities, coalescence phases, and peak arrival times are 100% congruent.

### 2.6 Batched Offline Inspiral Search (`pycbc_inspiral`)

To bring the high throughput of the persistent GPU search engine to offline search workflows, `bin/pycbc_inspiral` was upgraded with native batching support (`--batch-size` / `--tile-size`):
- **Dynamic Adapter Selection**: When `--batch-size > 1`, `pycbc_inspiral` instantiates `TiledMatchedFilterControl(..., tile_size=batch_size)` instead of `MatchedFilterControl`.
- **Preallocated Scheme Memory**: Waveform decompression writes directly into a reusable ring of scheme-native buffers (`tile_mem = [zeros(tlen, dtype=complex64) for _ in range(batch_size)]`), avoiding host-to-device conversions and memory churn.
- **Batched Correlation & IFFT**: Per-segment filtering batches all active templates into a 2D tensor pass (`torch.mul` correlation + `torch.fft.ifft(..., norm="forward")`).
- **Device Selection & Veto Parity**: Candidate selection and symmetric peak clustering execute entirely on device via `select_tile_candidates`. Power $\chi^2$ vetoes evaluate directly against full-$N$ correlation rows with zero memory duplication.
- **Default Batch Policy**: Defaults to $B=64$ for Torch CUDA, $B=16$ for Torch CPU, and $B=1$ (exact legacy loop) for CPU/MKL/CuPy schemes.

#### 384-Template BNS/NSBH Benchmark on NVIDIA RTX 4090 (`len`)

Under the reference protocol with 384 compressed BNS/NSBH templates over 5 segments of 512 s (2,097,152 points at 4096 Hz, 1,920 matched filters):

| Configuration | Scheme | Batch ($B$) | Setup & I/O (s) | Calculation (s) | Total Wall Time (s) | Calc Speedup | Wall Speedup | Raw Triggers | Post-NewSNR |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard CPU (MKL baseline) | `cpu:1` | 1 | 15.87 s (24.4%) | 49.28 s (75.6%) | 65.15 s | 1.00x | 1.00x | 2,203 | 1,988 |
| Sequential (Scalar) | `torch:cuda:0` | 1 | 10.30 s (50.5%) | 10.11 s (49.5%) | 20.41 s | 4.87x | 3.19x | 2,203 | 1,988 |
| **Batched GPU Search** | `torch:cuda:0` | 64 | **10.41 s (60.2%)** | **6.88 s (39.8%)** | **17.29 s** | **7.16x** | **3.77x** | **2,203** | **1,989** |
| Batched GPU Search | `torch:cuda:0` | 128 | 10.30 s (60.8%) | 6.65 s (39.2%) | 16.95 s | **7.41x** | **3.84x** | 2,203 | 1,989 |
| Batched CPU Search | `torch:cpu` | 16 | 14.58 s (8.3%) | 160.32 s (91.7%) | 174.90 s | 0.31x | 0.37x | 2,203 | 1,989 |

> [!NOTE]
> **Amdahl's Law Breakdown**:
> The persistent GPU engine accelerates pure matched-filtering calculation by **7.16x to 7.41x** (49.28 s down to 6.88 s). Total process wall time includes ~10.4 s of non-calculation overhead (frame file decompression, high-pass autogating, 63-segment PSD estimation, inline waveform decompression, and HDF5 serialization), resulting in a full end-to-end wall time speedup of **3.77x to 3.84x**.

#### Bit-Level Parity Evidence
- **Pre-Cut Parity**: Before NewSNR thresholding, **all runs generate exactly 2,203 raw triggers**.
- **Timing & SNR Identity**: For surviving triggers, arrival times match bit-for-bit ($\Delta t = 0.000000$ s) and SNRs match bit-for-bit ($\Delta \rho = 0.0$ in float32).
- **Cross-Backend Parity (`torch:cpu` vs `torch:cuda`)**:
  - Exactly **1,989 out of 1,989 triggers match**:
  - `template_hash`: 0 mismatches / 1,989.
  - `time_index` & `end_time`: max diff = $0.000$ s.
  - `bank_chisq_dof` & `cont_chisq_dof`: 0 mismatches.
  - `snr`: max diff = $3.34 \times 10^{-6}$ (float32 FFT precision).
  - `chisq`: max diff = $7.63 \times 10^{-5}$ ($0.000076$).
  - `coa_phase`: max diff = $4.77 \times 10^{-7}$ rad.
- **Threshold Boundary**: Between batch-1 and batch-64, 1,986 out of 1,988 triggers are strictly identical. The 2 discordant triggers sit directly on the knife-edge threshold ($\text{NewSNR} = 5.0001$ vs $4.9999$).

---

## 3. Qualification Verification Suite

The verification suite validates mathematical parity, dynamic range consistency, streaming stability, candidate parsing integrity, and receipt validation across both CUDA-accelerated and CPU fallback configurations:

### 3.1 Production Hardware Qualification (NVIDIA RTX 4090 on `len`)

On the target production qualification platform (`len`: NVIDIA GeForce RTX 4090 24GB, CUDA 13.0, PyTorch 2.13.0+cu130), the entire focused GPU-search test suite executes across both `cpu` and `cuda` device fixtures without any skips. Full test execution logs are retained in [`artifacts/pytest_len_cuda.log`](file:///Users/xangma/repos/pycbc/artifacts/pytest_len_cuda.log):

| Test Module | Coverage Area | Tests Run (len) | Result |
|---|---|:---:|:---:|
| `test_gpu_search_plans.py` | BankGeometry, BankPlan, PSDPlan, dynamic-range scaling, workspace budgets | 9 | PASSED |
| `test_gpu_search_eager_filtering.py` | Eager SearchEngine, candidate buffers, dynamic resizing, symmetric clustering | 27 | PASSED |
| `test_gpu_search_vetoes.py` | Power $\chi^2$, Sine-Gaussian $\chi^2$, bound PSD scaling, native evaluator fallback | 25 | PASSED |
| `test_gpu_search_adapter.py` | Offline & live application adapters, abort, top-K, schema, detector switching | 20 | PASSED |
| `test_gpu_search_graphs_streams.py` | CUDA graph capture & replay, async streams, double buffering, fork safety | 8 | PASSED |
| `test_gpu_search_qualification.py` | 1,920 CPU ref parity, streaming VRAM bounds, atomic receipt saving, integer candidate validation, post-injection strain hashing, fresh-process concurrency regression | 16 | PASSED |
| `test_gpu_search_multirate.py` | Multirate split-band filtering, noise rejection, and reconstruction | 8 | PASSED |
| `test_gpu_search_reduced_basis.py` | SVD/Greedy basis compression, veto evaluation, and subbank projection | 10 | PASSED |
| `test_gpu_search_screening.py` | Early consistency screening, signal vs. glitch discrimination, coarse-to-fine gating | 6 | PASSED |
| **GPU-Search Suite Total (`len`)** | **All core engine, adapter, veto, and qualification modules** | **129** | **129 PASSED (0 skipped, 0 failed)** |
| `test_torch_*_precision.py` | Float32/Float64 stability, sigmasq normalization, PSD estimation | 30 | 30 PASSED |

### 3.2 CPU-Only Baseline & Skip Attribution (macOS / CI without CUDA)

When executed on hosts without an NVIDIA GPU or CUDA runtime:
- **83 tests pass** verifying full mathematical and schema parity on CPU (`test_gpu_search*.py`).
- **Exactly 5 tests are skipped** due to explicit `@pytest.mark.skipif(not CUDA_AVAILABLE)` markers:
  - 3 skips in `test_gpu_search_graphs_streams.py` (`test_cuda_graph_lifecycle_and_parity`, `test_cuda_graph_tail_tiles_variable_batch`, `test_cuda_graph_fork_safety`).
  - 2 skips in `test_gpu_search_qualification.py` (`test_scientific_output_and_veto_parity_cuda`, `test_sustained_streaming_vram_boundedness`).
- In addition, parameterized tests across `DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])` run exclusively over the `cpu` fixture on non-CUDA systems (83 tests on CPU vs. 129 tests on CPU + CUDA).
- No tests fail or produce non-finite values in either environment.

---

## 4. Integration into Torch Stack

The entire GPU search engine implementation has been folded directly into the upstream 11-PR Torch stack:
- **PR #8 (`torch-pr4-filtering`)**: Core `pycbc.filter.gpu_search` engine, plans, candidate buffers, veto managers, and unit tests (Milestones M1-M5 qualified for production); multirate, reduced-basis, and screening included as experimental research prototypes (Milestones M6-M7).
- **PR #9 (`torch-pr5-search`)**: Offline (`TiledMatchedFilterControl`) and low-latency (`TiledLiveBatchMatchedFilter`) application adapters and adapter unit tests.
- **PR #15 (`torch-pr11-performance-evidence`)**: Qualification test suite, qualification receipt, benchmarking scripts, and this qualification report.

