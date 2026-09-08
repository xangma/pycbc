# Renewed Torch optimization pass — full executable results verified

The candidate reduces full executable CPU wall time by 6.16% against the promoted Torch baseline, but still takes 1.53 times the standard CPU wall time. The CUDA scalar hoist has no demonstrated full executable gain: its median wall time increases by 0.21%, with overlapping observed ranges. Existing CUDA throughput is about 2.83 times standard CPU for this fixed workload. The CPU parity target is not met.

## Source

- Candidate: `2f799f0046fc36db4215bd8b8b8a774d40c0e011`, clean local worktree `/private/tmp/pycbc-torch-fft-optimization-20260908`.
- Comparison baseline: `7e56ac42417dd6f61498f917e04c5a55250ddc26`.
- Candidate commits in order: `7a883441e9` (CUDA scheduling), `612a13cade` (CPU workspace), `2f799f0046` (existing CUDA autodiff contract tests).
- Changes: Python FFT dispatch/ownership, Python CUDA scalar-transfer scheduling, and two test files. No C, CUDA-kernel, or `pycbc/lib` edits.
- CPU: use one retained private complex128 in-place workspace only for the already promoted 2,097,152-point, single-thread IFFT route. Input preservation, output publication, descriptor lifetime, and the separate two-argument in-place ABI are covered.
- CUDA: queue a supported scalar normalization transfer before sparse chi-square bin sums while preserving float32 cast-then-square arithmetic. Special tensor, inference, gradient, and forward-AD cases retain the previous ordering.

## Completed validation

- Native CUDA normalization suite: 32 passed. The baseline fails the eight intentional transfer-order assertions and passes the other 24 cases.
- Related contracts: 255 passed, one platform skip.
- Production CPU plan: all 36 frozen scientific cases pass; all 108 timing-worker repeats also pass.
- The standard single-precision MKL reference fails one of 36 stricter precision cases in each timing worker. It remains a timing reference, not a precision-qualified replacement.
- The final alternative FFTW in-place route fails two of 36 frozen precision cases. The CPU route search is closed without relaxing the scientific budgets.
- The existing native CUDA sparse-bin path does not propagate correlation autodiff. Candidate tests preserve the existing behavior and verify derivatives on the supported Torch fallback. This pass does not add native correlation-autodiff support.

## Completed isolated FFT timing

Three fresh processes per role, one physical core (CPU 8), loaded native pools and Torch threads set to one. Each process constructs and qualifies the actual timed plan, performs warm measurements, and rechecks all precision cases afterward.

| Route | Median of process medians (ms) |
|---|---:|
| Standard CPU reference | 16.033834 |
| Promoted Torch CPU baseline | 36.974767 |
| Torch CPU candidate | 33.520219 |

Evidence: `cpu-timing-v5-summary.json`, `acquired-cpu-timing-v5/`, and `acquired-validation-v5/`. Independent peer recomputation passed.

## Full executable campaign completed

The frozen `executable-v3/` harness compares standard CPU, Torch CPU baseline/candidate, and CUDA baseline/candidate using 384 compressed templates, five segments, 1,920 scalar 2M IFFTs, and 1,904 valid detector seconds. All five qualifications passed before the 15 fresh unprofiled timing runs began in forward/reverse/forward route order. The qualification process group is absent and the shared benchmark lock was independently reacquired before timing began.

All 65 qualification checks and four verified cross-scheme trigger comparisons passed. Both candidates exactly reproduce all 1,991 aligned triggers across all 13 H1 datasets, including values and dtypes, against their corresponding baseline. The same-scheme exact comparison is additional descriptive evidence; it does not replace the frozen comparator gates. Independent reviews are `../torch-profiling-investigation-20260908/executable-v3-qualification-peer-review.json` and `../torch-profiling-investigation-20260908/executable-v3-same-scheme-trigger-review.json`.

All 15 timing workers exited successfully and all 30 timing trigger comparisons passed. The full timing process group is absent and the shared benchmark lock was reacquired. Raw receipts, HDF outputs, PSD arrays, scientific records, source/runtime/helper pins, unchanged comparison gates and both terminal audits were independently replayed. All 120 reported timing statistics were crosschecked against receipts and HDF values.

| Route | Full wall median (s) | Observed range (s) | Internal median (s) | Setup median (s) | Full-wall templates/core | Internal templates/core |
|---|---:|---:|---:|---:|---:|---:|
| Standard CPU | 69.873178 | 69.831535–70.114786 | 65.612204 | 15.263308 | 10,463.76 | 11,143.29 |
| Torch CPU baseline | 113.969209 | 113.395970–114.334606 | 109.758820 | 15.070248 | 6,415.21 | 6,661.30 |
| Torch CPU candidate | 106.943054 | 106.897436–107.186163 | 102.728777 | 15.095749 | 6,836.69 | 7,117.15 |
| CUDA baseline | 24.658973 | 24.628002–24.862198 | 20.365057 | 13.307808 | 29,649.90 | 35,901.50 |
| CUDA candidate | 24.711847 | 24.637057–24.726278 | 20.386037 | 13.291286 | 29,586.46 | 35,864.55 |

Each median uses three fresh processes on shared host `len`, fixed CPU affinity 8, native pools and Torch threads set to one; CUDA also uses the GPU. Ranges are observed samples, not confidence intervals. Full-wall templates/core means `384 * 1904 / executable wall seconds`; the HDF internal statistic uses the executable's internal timer. These are finite-workload rates, not converged throughput. CUDA setup plus time outside the internal timer accounts for roughly 71.3% of full wall time. Column medians are computed independently and need not add exactly.

Evidence: `executable-v3-summary.json`, `acquired-executable-v3/`, and the independent `../torch-profiling-investigation-20260908/executable-v3-timing-peer-review.json` and `executable-v3-summary-crosscheck.json` reports. Qualification times are excluded.

Source, native binaries, executable bytes, consumed inputs, actual dispatch, environment, threads, strain, PSDs, geometry, and triggers are checked. Strict raw comparisons are preserved; only explicitly verified source attribution and the path to the byte-identical executable may be normalized. Scientific tolerances and analysis options remain unchanged.

Version 1 stopped after its first successful standard CPU child because a new controller assertion incorrectly forbade Torch being imported by dependencies. Version 2 removes only that two-line assertion; existing scheme/device/pool and scientific checks remain. The failed version and its terminal audit are retained in `acquired-executable-v1/`. The independent correction review is `../torch-profiling-investigation-20260908/executable-v2-harness-correction-review.json`.

Version 2 stopped after two successful CPU children because another new controller assertion required identical whole PSD bytes across schemes. Both observed arrays exactly match their respective published r3 files. Their 14,374 differing bins are entirely below the actual filter slice; the used PSD slices are exactly equal. Version 3 pins each scheme's full published scientific record, including both PSD file/data hashes, and additionally requires candidate equality with its same-scheme baseline. It introduces no numerical tolerance or scientific-data normalization. The stopped version remains in `acquired-executable-v2/`.

No new LiveBatch result is claimed: the existing workload uses zero chi-square bins and cannot exercise this scheduling change. The separate synchronized sparse-API diagnostic is complete: six fresh processes, 48 cells, and exact input/output checks passed. The clearest cell (2M samples, 32 sparse points) reduced median latency from 0.676654 to 0.637705 ms (5.76%, about 39 microseconds). The 2M/32 and 2M/128 cells had nonoverlapping observed ranges; the other six cells overlap. This supports a bounded warm API improvement, not a full executable gain or a conclusion about where waits moved. Evidence: acquired-warm-chisq-v1/ and ../torch-profiling-investigation-20260908/warm-chisq-v1-results-review.json.

## Retention and next work

Retain the qualified CPU workspace change as a measured incremental improvement. The GPU scalar hoist has a measured improvement in selected warm sparse-API cells; it remains an optional API optimization without a demonstrated full executable gain. Preserve the separate preexisting correctness fix `7e56ac4241`, which completes CUDA peak transfers before publishing host arrays, irrespective of timing.

The remaining CPU barrier is numerical: faster single-precision alternatives fail the unchanged frozen maximum-error checks. The current promoted route preserves the required accuracy but does not reach standard CPU performance. No tolerance was relaxed to obtain a speedup.

A separately reviewable loader candidate is frozen at ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f atop 2f799f0046. It guards only the existing duration-metadata block when both start and end times are supplied. First-channel selection, default/duration behavior, reader calls, cache sieving, integrity flags and validation are preserved. The two-file diff changes pycbc/frame/frame.py and test/test_frame.py. Local and native frame tests passed (16 tests plus 48 subtests); 28 native baseline/candidate read comparisons are exact across four dtypes, and all eight metadata stream-position checks are unchanged. Local baseline positive and negative controls confirm the new tests exercise the bypass. Changed-file F401 lint passes; whole-tree F401 findings are verified preexisting. Peer source, XML and exact scientific HDF controls passed. See loader-v1/ and the frame-loader review reports in ../torch-profiling-investigation-20260908/.

The frozen loader campaign measures standard CPU, Torch CPU and CUDA with the change applied equally. All six fresh science qualifications and eight strict/exact comparison reports passed. Each candidate reproduces its same-scheme baseline exactly across strain, PSDs and all 18 scientific H1 datasets. All 22 H1 dataset paths retain identical schema; only four documented performance telemetry arrays may differ. The qualification process group is absent and the shared benchmark lock was reacquired. Evidence is in acquired-loader-v1/ and the immutable loader-qualification-v1.tar snapshot. The 18 unprofiled timing workers retain all earlier workload geometry, tolerances and runtime controls; their results remain pending, so no loader speedup is claimed yet.

This draft awaits the loader full executable measurements and final evidence checksum inventory. EVIDENCE-INDEX.md maps the completed campaigns and preserved failures. The working branch now includes the separate loader commit; the earlier v3 evidence remains unchanged. No integration, push, PR, or documentation publication has been performed by this task.
