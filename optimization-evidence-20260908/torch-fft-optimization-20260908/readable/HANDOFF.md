# Renewed Torch optimization — completed evidence handoff

The frame-loader change reduces full executable median wall time by **4.38% for standard CPU, 2.49% for Torch CPU and 12.19% for CUDA**, compared with each corresponding pre-loader route. Each comparison has nonoverlapping observed wall-time ranges, with all three same-repeat ratios below one. Torch CPU still takes **1.56 times** the updated standard CPU wall time, so the CPU parity target is **not met**. CUDA delivers **3.08 times** the updated standard CPU full-wall throughput on this workload, using one assigned CPU core plus the GPU.

## Final loader campaign

The candidate is `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`; its comparison baseline is `2f799f0046fc36db4215bd8b8b8a774d40c0e011`, which already includes the CPU workspace and CUDA scalar scheduling changes. The loader change is applied equally to all three backends. Its measured improvement must be distinguished from the earlier optimization campaign below.

Six fresh qualification runs passed all 78 checks and eight comparison reports before timing began. All 18 fresh, unprofiled timing runs and 36 timing comparisons passed. The six qualification runs exactly reproduce their same-scheme conditioned-strain hash records and actual PSD files. All qualification and timing HDF outputs reproduce the 18 scientific H1 datasets exactly within scheme. Timing workers retain the source, input and runtime controls; they do not independently recapture conditioned-strain or PSD arrays. All 22 H1 dataset paths preserve their schema; four documented performance telemetry datasets are checked separately. Scientific tolerances and analysis options are unchanged.

| Route | Full wall median (s) | Observed wall range (s) | Internal median (s) | Setup median (s) | Full-wall templates/core | Internal templates/core |
|---|---:|---:|---:|---:|---:|---:|
| Standard CPU, before loader change | 70.000746 | 69.822451–70.570593 | 65.698221 | 15.331096 | 10,444.69 | 11,128.70 |
| Standard CPU, after loader change | 66.933429 | 66.629332–67.996605 | 62.386842 | 12.040840 | 10,923.33 | 11,719.39 |
| Torch CPU, before loader change | 107.068339 | 107.062365–107.645261 | 102.866867 | 15.178025 | 6,828.69 | 7,107.59 |
| Torch CPU, after loader change | 104.400193 | 104.212417–104.995407 | 99.905155 | 11.863055 | 7,003.21 | 7,318.30 |
| CUDA, before loader change | 24.786463 | 24.706543–25.537461 | 20.480302 | 13.378979 | 29,497.39 | 35,699.47 |
| CUDA, after loader change | 21.765736 | 21.727497–22.123757 | 17.192195 | 10.231784 | 33,591.15 | 42,527.21 |

There are three samples per route, ordered forward/reverse/forward, on shared host `len`. CPU affinity is 8; native pools and Torch threads are one. The fixed workload contains 384 compressed templates, five segments, 1,920 scalar 2M IFFTs and 1,904 valid detector seconds. Full-wall templates/core is `384 * 1904 / executable wall seconds`; internal templates/core uses the executable's HDF internal timer. These are finite-workload rates. The machine was shared, so this does not establish an empty-machine or saturated-machine throughput result. Observed ranges are not confidence intervals. Column medians are independent and need not add exactly.

The eight metrics' 144 raw sample values, 48 medians, 48 ranges and five explicit comparisons are in `loader-v1-summary.json`; `loader-v1-metrics.csv` provides all raw values and ranges in a flat table. Qualification times are excluded. Acquisition verified that all earlier qualification files remained byte-for-byte unchanged. The timing process group is absent and the shared benchmark lock was reacquired; the terminal audit binds that result to the completed status SHA-256.

Evidence: `acquired-loader-v1/`, immutable `loader-qualification-v1.tar` and `loader-timing-v1.tar`, the frozen 34-entry `loader-v1/manifest.json`, and the loader qualification/timing/summary reports in `peer-reviews/`.

## What to retain

- **CPU workspace (`612a13cade`)**: retain the private complex128 in-place workspace for the already promoted 2,097,152-point single-thread IFFT route. In the earlier five-route executable campaign it reduced Torch CPU median wall time from 113.969209 to 106.943054 seconds (6.16%). The corresponding standard CPU median was 69.873178 seconds. Isolated IFFT process medians improved from 36.974767 to 33.520219 ms, with all frozen production-route precision checks passing. Faster single-precision alternatives failed the unchanged precision gates; no tolerance was relaxed.
- **Frame loader (`ecd5d08231`)**: retain the independently measured, general improvement. It skips only the existing duration-metadata block when both time bounds are supplied. First-channel selection, default and duration behavior, actual reads, cache sieving, integrity checks and validation retain their behavior. Native validation includes 28 exact baseline/candidate reads across four dtypes and eight unchanged metadata stream-position checks.
- **CUDA scalar scheduling (`7a883441e9`, contract tests `2f799f0046`)**: optional as a bounded API optimization. The earlier full executable medians were 24.658973 versus 24.711847 seconds, with overlapping ranges, so that campaign demonstrated no executable gain from the hoist. A separate synchronized warm sparse-API diagnostic showed the clearest improvement at 2M samples/32 points: 0.676654 to 0.637705 ms (5.76%, about 39 microseconds). Only the 2M/32 and 2M/128 cells had nonoverlapping ranges; the other six overlapped. No new LiveBatch gain or native correlation-autodiff support is claimed.

Preserve the separate correctness baseline `7e56ac42417dd6f61498f917e04c5a55250ddc26`, which completes CUDA peak transfers before publishing host arrays, irrespective of timing. The remaining CPU parity limitation is explicit; the loader's CUDA improvement is a common setup improvement, not evidence of a faster GPU kernel.

## Validation and delivery

Native CUDA normalization tests passed all 32 cases; related contracts passed 255 cases with one platform skip. The promoted CPU route passed all 36 frozen scientific cases and all 108 timing-worker repeats. Frame tests passed locally and natively: 16 tests plus 48 subtests; baseline positive and negative controls confirm that the bypass tests exercise the changed behavior. Changed-file F401 lint and diff whitespace checks passed. Whole-tree F401 findings were verified as preexisting and are retained in `frame-loader-lint.json` and its log.

The earlier executable v1 and v2 controller failures, their diagnostics and terminal audits are preserved. Version 1 incorrectly forbade a dependency importing Torch. Version 2 incorrectly demanded identical whole PSD bytes across schemes; the differing bins were outside the used filter slice and each array matched its published same-scheme reference exactly. Version 3 pins the complete per-scheme references and keeps the scientific comparator unchanged. These corrections do not relax scientific precision or normalize scientific data.

The final clean source is `/private/tmp/pycbc-torch-fft-optimization-20260908`, branch `codex/torch-fft-optimization-20260908`, HEAD `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`. `final-source.json`, `final-source.diff` and the verified `final-source.bundle` preserve the four commits above baseline `7e56ac4241`. Exactly six Python source/test files changed; no C, CUDA kernel or `pycbc/lib` file changed.

`EVIDENCE-INDEX.md` maps the campaigns. `peer-reviews/manifest.json` records the original paths and SHA-256 hashes of copied independent reviews. `SHA256SUMS` covers the complete local handoff except itself and Python bytecode caches. No benchmark job remains. No integration, push, PR or documentation publication has been performed by this task.
