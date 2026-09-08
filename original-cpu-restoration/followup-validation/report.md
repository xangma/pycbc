# Isolated optional followup validation — 2026-09-08

PR16 needed one compatibility fix, committed as `dcef154a13d2d51d27d7d0b6454ff4e751e37489`. PR17 needed no changes. Both owned worktrees are clean; all jobs are finished. No primary refs/files, native sources, or network resources were modified.

## Scope and finding

- Main15: `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`.
- PR16 input: `4b4dca0e4d7278be78422d2d3540109b02cdf26c`; final: `dcef154a13d2d51d27d7d0b6454ff4e751e37489`.
- PR17 input/final: `80aae8b29cac86eea1c7ff9278601481d70061b1`.
- Worktrees: `/private/tmp/pycbc-original-cpu-followup-validation-20260908-pr16` and `/private/tmp/pycbc-original-cpu-followup-validation-20260908-pr17`.

Restoring the original FFTW wrapper removed its wisdom-I/O lock. PR16's immediate cache import/export remained protected by the Torch plan factory. However, after a failed immediate export, the CLI retry called `wisdom_cache.export_pending()` outside that factory and therefore outside the Torch planner/destructor lock. A regression test reproduced this missing serialization before the fix (`pr16-retry-regression-before.log`).

The fix takes `torchfft._FFTW_PLANNING_LOCK` around pending automatic exports. It returns before importing Torch when no export is pending. Only `pycbc/fft/wisdom_cache.py` and `test/test_fftw_wisdom_cache.py` changed. Manual wisdom operations and the restored CPU FFTW functions remain unchanged. The tests cover lock acquisition through the CLI retry lifecycle and empty-cache import isolation. The existing fake-cache retry test now explicitly skips when Torch is unavailable, since an actual pending automatic cache originates from Torch planning.

Apply the local owner-scoped fix with `git cherry-pick dcef154a13d2d51d27d7d0b6454ff4e751e37489` onto the PR16 branch. `pr16-wisdom-retry-fix.patch` is also provided.

## Validation

| Head | Selection | Passed | Skipped |
| --- | --- | ---: | ---: |
| PR16 final | 14 affected/dependent pytest modules | 257 | 16 |
| PR16 | Legacy unthreaded FFT + FFTW pthreads | 91 | 0 |
| PR17 | 20 affected/dependent pytest modules | 512 | 102 |
| PR17 | CPU FFT preservation + CPU sky-max chi-square | 12 | 0 |
| PR17 | Legacy unthreaded FFT + FFTW pthreads | 91 | 0 |
| PR17 | Legacy CPU matched filter + chi-square | 12 | 0 |
| **Total** | | **975** | **118** |

The legacy PR16 tests preceded the narrow retry fix; its final pytest selection was rerun after the fix. PR17's tests used its independent head, without PR16's optional general batch changes. Selected tests cover FFT batching, CLI wisdom, cache fingerprints/cancellation/manual precedence, planner ownership and destruction, CPU native peaks and gate-off behavior, single-point chi-square parity and fallbacks, MKL descriptor-thread selection, live batching, and dependent filtering paths. Exact module lists and commands are in `initial-runs.json`, `legacy-runs.json`, and `final-run.json`; logs and pytest XML are alongside them.

Five additional real FFTW probes passed in separate processes: cold cache measurement/export, warm import, disabled cache, direct-plan failure with promoted fallback, and initial-export failure followed by a successful CLI retry. Native cache import/export asserted ownership of the actual Torch lock. A second thread verified that plan destruction waited while that lock was held. Each probe checked preserved input storage, output version updates, numerical agreement with a complex128 NumPy reference, and release of cache-file locks. The final retry probe exercised the fixed code and exported real wisdom successfully.

Both heads passed CI-scope F401 lint for library/tests and executables, plus `git diff --check`. Qlty is unavailable in this environment. The pytest warnings were the existing explicit fork test's multi-threaded-process warning.

## Preservation and environment

Main15's `pycbc/fft/fftw.py`, `npfft.py`, and `mkl.py` remain byte-identical to original CPU baseline `40e94792b3`. Against that baseline, PR16 preserves all 21 other top-level FFTW functions/classes; only `insert_fft_options` and `verify_fft_options` differ, for the three requested wisdom flags and validator. PR17 preserves all 23 FFTW functions/classes. See `preservation-and-quality.json`.

Runtime: host `client-3792.default.vpn.port.ac.uk`, macOS; `/private/tmp/pycbc-cpu-precision-env-20260908/bin/python`, Python 3.13, Torch 2.9.1, NumPy 2.3.5. Each worktree received copied native artifacts only after all 13 tracked build inputs (11 Cython sources plus setup/pyproject) matched its donor byte-for-byte. The 11 native binaries per worktree, donor paths, and SHA-256 hashes are recorded in `native-provenance.json`. PR17 used its matching optional native kernels.

This is local compatibility validation, not new Linux/MKL or GPU performance/precision qualification. Hardware/platform-specific tests skipped where unavailable; CuPy/cuFFT batch contract tests use their existing test doubles. Real wisdom probes directly requested the eligible FFTW factory on macOS, where normal public dispatch is platform-gated. No Linux production search campaign or GPU execution was run in this bounded review.
