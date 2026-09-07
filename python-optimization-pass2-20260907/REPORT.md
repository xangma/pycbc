# Second Python optimization pass, 7 September 2026

**No demonstrated full-executable improvement; candidate remains isolated and unintegrated.** Baseline median 114.270831585 s versus candidate 114.251992669 s gives B/C 1.000164889× and 0.016486% less wall time. The 18.839 ms median difference is small relative to the overlapping observed ranges: baseline [114.165037497, 114.570708961] s; candidate [114.116896105, 114.308555996] s. Three repeats per revision on shared len do not establish a reliable executable benefit. The coordinating task decided to keep the candidate unintegrated and close this pass after evidence release.

Candidate `af4f0bd598ec1e2d7c6ce702d6c05850d7871039` on baseline `d2647addb884ead3249914ebc980f3c132076d93` reuses the output of `tensor.real.square()` for the subsequent `add_(tensor.imag.square())`, removing one temporary allocation inside the existing CPU complex ordinary-Tensor non-conjugate branch of length >=4096. Input precision and guards remain unchanged. Only pycbc/types/array_torch.py and test/test_torch_squared_norm.py differ. No C/CUDA kernel, pycbc/lib, FFT policy or numerical tolerance changed. Candidate source is clean at /private/tmp/pycbc-torch-python-optimization-pass2-20260907 on codex/torch-python-optimization-pass2-20260907; no integration, push, PR change or publication write was performed by this task.

The new full Torch CPU profile attributed 1.468720 s of ~118.07 s instrumented wall time to squared_norm, so the available executable gain was small. Required FFT promotion/conversion and native work were not treated as avoidable Python overhead.

## Paired executable measurements

Fixed compressed bank: 384 templates, five segments, 1,920 scalar inverse FFTs per qualification, 1,904 valid seconds, 512-second segments with 112/16-second pads. The same startup verification is included in full wall time. Two fresh qualification workers precede six fresh timing workers in AB/BA/AB order. CPU8 affinity, sibling CPU72, all observed numerical-library and Torch threads set to one. Python 3.11, Torch 2.13.0+cu130, NumPy 1.26.4. This is a finite shared-host workload, not sustained capacity or a confidence interval. Executable host observations span 181 samples with one-minute load 18.16–19.41; no exclusive host reservation was claimed.

| chronological worker | full wall s | internal s | setup s | outside internal s |
|---|---:|---:|---:|---:|
| baseline-r1 | 114.270831585 | 109.829179287 | 15.181039572 | 4.441652298 |
| candidate-r1 | 114.116896105 | 109.641326666 | 15.139410973 | 4.475569439 |
| candidate-r2 | 114.308555996 | 109.844024420 | 15.170303583 | 4.464531576 |
| baseline-r2 | 114.570708961 | 110.079461813 | 15.079032660 | 4.491247148 |
| baseline-r3 | 114.165037497 | 109.688110828 | 15.176893234 | 4.476926669 |
| candidate-r3 | 114.251992669 | 109.683238029 | 15.152178526 | 4.568754639 |

The campaign started at 2026-09-07T20:58:30.630478+00:00 and finished at 2026-09-07T21:13:49.826408+00:00. Eight distinct runtime PIDs, receipt-bound source imports, complete command/environment/affinity observations and the inherited shared-lock descriptor are retained. Total bound was 10,800 seconds, per-worker bound 900 seconds.

## API and scientific checks

Six serial fresh API workers, 20 cells each, seven timing samples per cell: all 120 outputs are bitwise identical with unchanged inputs. All 14 changed-path cell medians improve (1.016874–1.107433×); the six below-cutoff cells use unchanged runtime code and have overlapping ranges (0.981893–1.002541×). API-RESULTS.md preserves all 20 medians, ranges and ratios. Retain the existing cutoff; API throughput does not establish executable benefit.

Both qualifications pass all 13 checks, including all 384 compressed templates, no generation fallback and all 1,920 scalar IFFTs. All four cross-revision comparisons match exactly 1,991 H1 trigger identities on the sample grid, with zero differences in all 11 compared numeric fields. All six within-revision comparisons pass. Original strict comparisons remain FAIL with exactly `configuration mismatch: consumed_input_sha256` and `configuration mismatch: source_snapshot`, no review reasons. The separately authorized wrapper verifies the exact source pair and byte-identical executable and changes only those two metadata representations in memory. Its four PASS records, original FAIL records and all eight original HDF files and receipts are preserved; comparator DEFAULTS are unchanged.

## Validation, recovery and release

Candidate and exact baseline focused squared_norm suites: 235 passed, 11 CUDA skips each. Candidate array/backend/filter/PSD/search integration: 476 passed, 102 accelerator skips, two existing PSD warnings. Added coverage includes unchanged input version counters and second derivatives for complex64/complex128, contiguous and stride-3 arrays. Local acquisition controls: 142 passed, one Linux lifecycle skip; len controls: 143 passed, zero skips. Changed test lint and scoped unused-import checks pass. Full-repository flake8 reports 8,153 diagnostics, none on changed lines; qlty is unavailable.

The initial validation driver discarded worker pytest return codes. Its original bytes and raw successful logs are retained. The corrected driver propagates SystemExit; controlled failure/success through its actual worker entry point returned 1/0. Its precommit source check remains an acquisition record and is not an unattended campaign launcher. The first remote source-staging attempt failed before creating checkouts because a historical git object was absent in the compact profiler clone. Original package/launcher/log/audit are retained. The replacement verifies all 17 native build inputs against byte-identical local git blobs at both revisions and the actual len files; all 11 native binaries match. Runtime candidate and 24 acquisition-helper bytes remained fixed.

Independent terminal and export audits found all 28 staging/API/executable/export groups inactive, reacquired /home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock, and revalidated source/native/input/helper pins. The export contains 202 campaign files plus its manifest. Every local restored file hash and every scientific-audit file hash matches. No campaign remains running. Inputs, complete checkouts, native binaries and environments are represented by pinned provenance rather than bundled; safe offline restoration and comparison commands are in the publication README.

Exact full-index patch SHA256: cba88c8de4dd93df8c0c72c2aec3a661aa22f4745b91d80a94922d23729956e8. Executable audit SHA256: 0896c915b6a4fe9950f5cae67f5bf3ac285bd9e0b6cf7b7c559f29f88c3a72eb. Export release audit SHA256: b5f24452920e516e35f67aa18dcb41cba6e3795cd2bf8326763df721e1a136d7. Original remote archive SHA256: f62fc1c56c2c6820a444a0c73ec331b6b2fa54c7bafe3aefebca9f2bcbb320f1 (12,336,489 bytes).
