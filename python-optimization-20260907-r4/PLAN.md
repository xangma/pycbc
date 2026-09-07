# Bounded squared-norm validation on the measured R4 runtime

Profiling explicitly released len after successful terminal verification. The parent reviewed and approved staging and the API phase only; see `PARENT-REVIEW.md`. Executable continuation still requires a separate review of all API cells and the cutoff assessment.

Candidate `9e6a688a5190d6e1ddc655fbe352cc206085d5c6` is based on measured R4 runtime `9578a710479b924e882857c4dffab6ed372a634b`. Its two-file change is byte-identical to original candidate `1f914b4599e0ae13357836c71ebd3c45a52bbf16`: ordinary CPU complex tensors of at least 4096 elements compute separate real/imaginary squares and add them, preserving precision. The cutoff remains provisional for x86. Original branch, acquisition helpers and evidence remain frozen. No integration into the PR stack is performed here.

## Evidence and scope

The existing 384-template CPU profile attributes 5.7998 seconds of Tensor.sum time to squared_norm; the scalar executable's FFT, bank and array paths did not change between the original baseline and R4. The runtime delta is confined to LiveBatch template-group normalization/cache identity and veto correlation geometry/cache validity. The warm R4 live-filter API uses cached sigma handling and cannot establish executable benefit from this patch.

New local validation: each focused candidate/baseline variant passes 231 tests with 11 CUDA skips; broader Torch array/filter/PSD/search integration passes 443 with 93 skips. Focused tests include squared-norm precision/storage/AD/fallback contracts, reductions, chi-square and sigma-series precision, plus R4 LiveBatch peaks/veto/FFT regressions. The baseline worker restores only the exact baseline squared_norm AST in the otherwise identical R4 candidate tree. Scoped F401, regression-test style and patch whitespace checks pass. Repository-wide lint still reports errors outside the changed lines. See `lineage.json`, `validate-local.py`, `local-validation.json` and their logs.

Prior Apple M4 Max public-API measurements remain attributable to the original source pair. Both relevant module hashes and the complete source diff are identical in the R4 pair; these measurements do not establish x86 cutoff suitability or executable speedup. No redundant Apple timing is proposed.

## Scheduling and preparation

Profiling owns len first: `/home/xangma/pycbc-torch-device-profile-20260907-r2`, PID/PGID `2663981`, reported by profiling task `01a07ac4-9675-7983-9923-90de72db82d5`. Its first locked stage passed all nine Linux controls with zero skips. Await its direct explicit release, successful terminal receipt hash, scientific/trace validation and independently verified process-group exit and lock availability. A running or successful status alone is not release. Do not fabricate `profile-release.json`.

The refreshed acquisition copies the reviewed R4 gate exactly. It requires convergence PID 340874 and R4 PID 3789112 successful and all recorded owned groups inactive; R4 must have 36 qualifications, 54 timing workers, unchanged source and exact terminal status SHA256 `a19f0d8862c7a05be182cfa3e8031ac3fe919ff210b7abede1490166adf40d32`. Profiling additionally requires scientific parity, captured trace, unchanged source, passing Linux controls, inactive groups and the explicit release matching its exact status hash.

After release and review of this concrete bundle, announce remote writes and create only `/home/xangma/pycbc-torch-python-optimization-r4-20260907`. Stage the helpers in `acquisition/` and prepare distinct clean `source-baseline` and `source-candidate` checkouts at the exact pair above. Verify unchanged native build inputs, copy the eleven qualified native extensions from R4 source, and verify the R4 provenance hashes in both new checkouts. Preserve the shared venv, existing sources, inputs, helpers and outputs. No package installation or compilation on len is proposed.

Both phases verify predecessor/release gates before and after acquiring `/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock`. The existing inherited-FD, whole-process-group cleanup and thread-control helpers are unchanged. Retain the flock throughout every stage and descendant; do not explicitly unlock or start concurrent profiling/optimization. Recheck current host state before acquisition.

## Phase 1: public Array.squared_norm API

Host: len. Cwd: `/home/xangma/pycbc-torch-python-optimization-r4-20260907`.

Command: `/home/xangma/pycbc-torch-split-20260905/venv/bin/python optimize-campaign.py --phase api --shared-host`.

Launch in a dedicated session, save `api-launch.log`, report PID/PGID and `kill -TERM <PID>`, and check within 60 seconds. Stage bound 180 seconds; phase bound 900 seconds. The first locked stage is the real Linux inherited-lock runner test; its macOS skip does not count as success. Next verify the pinned threadpoolctl runtime on Python 3.11.

Run six fresh workers in AB/BA/AB order. Each uses the public Array API, both complex dtypes, the frozen 18 length/stride cells and seven timing samples, one actual Torch intra/inter-op thread and CPU8 affinity (SMT sibling 72). Verify exact input/output parity, source/helper identities, unique workers and complete repeats. Save every median and observed range. These are shared-host component measurements, not executable capacity.

Send all 18 cells, ranges, receipt hashes and a concrete cutoff assessment to parent task `01a079bf-abfd-7cd2-bd6a-3893ab4f0d53`. Require its explicit decision before phase 2. `api-review.json` must name that parent as reviewer and bind the candidate and exact summary hash. The optimization task cannot approve its own continuation. A negative or inconclusive cutoff result is reported before any revised candidate.

## Phase 2: scientifically qualified executable

Same host/cwd. Command: `/home/xangma/pycbc-torch-split-20260905/venv/bin/python optimize-campaign.py --phase executable --shared-host`.

Save `executable-launch.log` and report the same operating fields. Stage bound 900 seconds; phase bound 10800 seconds. Recheck release, all prior groups exited, unchanged API evidence/helper hashes and parent's review before and under the same whole-job lock.

Use the exact frozen 384-template compressed BNS/NSBH bank, frame, 512/112/16 geometry, 1904 unique valid seconds, Torch CPU with one numerical-library thread, CPU8 and explicit MKL. Both instrumented qualification workers must pass template/compression/no-generation-fallback/scalar-IFFT/coverage checks and unchanged trigger/veto cross-revision comparison before six fresh unprofiled AB/BA/AB timing workers. Validate every repeat within and across revisions. Preserve raw strict revision-mismatch FAIL receipts; the separate cross-revision wrapper permits only the exact authorized source pair and byte-identical executable path-role substitution. Full wall times include the identical runtime verification startup. Historical unwrapped executable timings are not the denominator.

R4's adopted v2 remains the prerequisite for the live-filter API: actual normalized complex SNR must agree within absolute 0.001 with both the independent complex128 oracle and actual MKL, with observed normalizations and unchanged trigger/veto budgets. This scalar executable experiment retains the original scientific comparator unchanged and does not rerun or reclassify the live-API matrix. R3 remains failed under raw-v1. No precision reduction or further tolerance relaxation is proposed.

Report all wall times, medians/ranges, fixed work, ratio and scientific results, including negative/noisy outcomes. A production-benefit claim requires completed executable evidence.
