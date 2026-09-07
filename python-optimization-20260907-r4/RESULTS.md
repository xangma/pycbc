# Torch CPU squared-norm optimization: measured R4 candidate

Candidate `9e6a688a5190d6e1ddc655fbe352cc206085d5c6` reduces median full executable wall time from **119.262860 s to 114.138915 s** for the frozen workload: **4.296% less wall time; 1.044892× speedup**. All scientific comparisons pass, with zero differences in the compared trigger values. The three-repeat ranges do not overlap. These are observations on shared len, not confidence intervals or demonstrated steady-state capacity.

## Change and lineage

Baseline: `9578a710479b924e882857c4dffab6ed372a634b` (measured R4 runtime). Candidate: `9e6a688a5190d6e1ddc655fbe352cc206085d5c6`, branch `codex/torch-python-optimization-r4-20260907`, clean worktree `/private/tmp/pycbc-torch-python-optimization-r4-20260907`.

`pycbc/types/array_torch.py:squared_norm` computes separate real and imaginary squares and adds them for ordinary, non-conjugated complex CPU tensors with at least 4096 elements. Both squares and the sum retain the input's real precision. Other cases retain the previous expression. The production diff adds 11 lines and removes one, plus 159 lines of regression tests. No C extension, CUDA kernel, FFT precision, tolerance, or `pycbc/lib` source changes were made.

The two-file patch is byte-identical to original candidate `1f914b4599e0ae13357836c71ebd3c45a52bbf16` on original baseline `818331d6ec39ed6f9fb64abcb867a27ee9919785`; the original worktree and evidence remain frozen. `lineage.json` records both pairs and module/test hashes. The exact full-index diff SHA256 is `06c76d2bd544f9ac96fbb407356e59bb7ef675a1edc47860f9470cf0e92f3b09`.

## Executable measurements

len, Torch 2.13.0+cu130, Python 3.11.9, NumPy 1.26.4, Torch CPU and explicit MKL. One numerical-library thread, CPU8 affinity (SMT sibling 72). Frozen compressed BNS/NSBH bank: 384 templates, five segments, 512/112/16 geometry, 1904 unique valid seconds. Two instrumented qualification workers preceded six fresh unprofiled AB/BA/AB timing workers under the same inherited shared lock. Full wall time includes identical runtime-verification startup. Historical unwrapped measurements are not used as the denominator.

| Repeat | Execution order | Baseline wall s | Candidate wall s |
|---|---|---:|---:|
| 1 | baseline → candidate | 119.125787529 | 113.955014904 |
| 2 | candidate → baseline | 119.367811964 | 114.138915084 |
| 3 | baseline → candidate | 119.262859915 | 114.305301668 |
| Median | | 119.262859915 | 114.138915084 |
| Observed range | | 119.125787529–119.367811964 | 113.955014904–114.305301668 |

Both qualifications confirmed all 384 compressed templates exactly once, no generation fallback, all 1920 scalar IFFTs, and gap-free/non-overlapping coverage. Qualification wall times (119.462587 s baseline, 114.106658 s candidate) are excluded from the performance estimate.

Six within-revision repeat comparisons and four cross-revision comparisons (qualification plus each timed pair) pass. Each cross-revision pair matches all 1991 triggers by template and sample identity; all 11 compared fields have zero maximum absolute error, including SNR, phase, sigmasq and available veto fields. Raw strict comparisons correctly fail on the different source revision and executable path; their original FAIL receipts remain intact. The separately reviewed wrapper permits only this exact revision pair and byte-identical executable path-role substitution. It changes no numerical budget. The independent audit recomputed these raw and wrapped results exactly.

## Public API results and cutoff

All 108 raw worker cells (18 cells × six fresh workers, seven samples each) pass exact output parity, unchanged-input checks and cross-worker input/output hashes. All tested fast-path cells improve by 1.391×–18.249×, with non-overlapping worker-median ranges. At 4096 elements, complex64 contiguous/stride-3 gains are 1.391×/1.775×; complex128 gains are 4.314×/4.214×.

All six below-cutoff medians regress by **0.949316–1.502676 µs per call** (baseline/candidate ratios 0.939482×–0.990176×). This cost remains part of the result. Retain the 4096 cutoff for this candidate on the tested host/runtime; neither optimality nor portability to every x86/Torch configuration was established. The full 18-cell table, all observed ranges and hashes are in [API-REVIEW-REQUEST.md](API-REVIEW-REQUEST.md). Component speedups and the separate R4 warm live-filter API results are not executable-capacity claims.

## Validation and integrity

On local Apple CPU/Torch 2.9.1, focused candidate and baseline variants each passed 231 tests with 11 CUDA skips; broader candidate Torch array/filter/PSD/search integration passed 443 with 93 skips and two existing PSD warnings. The baseline focused worker restores the exact baseline function in the otherwise identical candidate source. Tests cover precision/extremes, storage, strides, AD, fallback contracts, reductions, chi-square and sigma precision, plus R4 LiveBatch peak/veto/FFT regressions. `local-validation.json` and logs preserve commands and import identities.

The acquisition controls passed 118 local tests; the one Linux-only test skipped locally subsequently passed on len with zero skips. Pinned Python 3.11 threadpoolctl compatibility passed. Scoped unused-import/error lint and changed-test style/whitespace checks passed. Repository-wide lint reports pre-existing errors outside the changed lines; qlty was unavailable locally.

Before/after source, eleven native binaries, input, helper, runtime, affinity and thread-control receipts all match their pins. The parent approved the exact API result and unchanged candidate via `api-review.json`. The R4 adopted-v2 numerical gate and explicit profiler release were checked before and under the shared lock. No precision/tolerance relaxation or reclassification of the prior R3 failure occurred.

The independent terminal audit revalidated all eight fresh executable workers, scientific outputs and wall statistics, found no active owned groups, and reacquired the shared lock after supervisor 3141911 exited. All 190 transferred evidence files match its manifest. Evidence transfer is complete; len is released, with no further remote work planned by this task. Candidate integration remains with the parent task.

## Evidence

- Complete remote result files: `executable-remote-evidence/`, including both phases, eight trigger HDFs, qualifications, runtime receipts, all logs, host samples, raw strict FAIL and wrapped PASS comparisons.
- Reproducible read-only terminal audit: `audit-executable.py`; result: `executable-terminal-audit.json` (contains all 190 evidence hashes).
- Earlier complete API snapshot: `api-remote-evidence/`; source bundle: `source-candidate.bundle`; source patch: `candidate.patch`.

| Artifact | SHA256 |
|---|---|
| Executable summary | `f21b5db87509c9e94701bcb050f97d3a0d104986df40bf51c47af3c2df06ce66` |
| Executable terminal status | `55015e8cc2fde4214f4e2e38cdb402f6807c993c2d3087a09d4bb66086b73727` |
| Independent terminal audit | `7810dbe5d24eb285ff83a0d1f60469bb54c1686804a3ea7312274f5b8b2dd954` |
| Complete remote evidence archive | `f502e162bcc61e5636f5764572f09b6c3ec1e0ed0de617d15eefffdd0a3c9754` |
| API summary | `a4bf978c0251f1d57785facd65fed9760ec578b22ddfbb9256fea29b617b1c31` |
| Parent executable review | `70e5aae8f2ea7df7531ca834318adafe0d47e4aa6ddbaecf7099f744ae24282a` |
| 23-helper manifest | `9c5db9e364e364281cf7dc21e168229e95447317975a7489cb111b378a5e959f` |
