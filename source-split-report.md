# Standalone CPU precision correction split

Completed 2026-09-08. Four local commits start directly from frozen original
`40e94792b3edf59f39b18b65102b28a4f74433a7` on branch `codex/cpu-precision-corrections-20260908`.
The worktree is `/private/tmp/pycbc-cpu-precision-corrections-20260908`; final HEAD is
`66789ac4a7468094b0cc3ca1498a1de67e0311f6` and its tracked/untracked working status is clean.
No network calls, publications, remote changes, or PR modifications were made.
The shared checkout remains at `e1904358e1ddd0d40b4e08958437488375a1715b` with
its pre-existing untracked entries. Frozen proposed sources were only read.

| Commit | Correction |
| --- | --- |
| `bd37e124ad30d1bda06156b667a19f2d8c748915` | Improve CPU estimated PSD precision for single-precision strain |
| `597c465008d979cc82bf6a5a7e58a4d93e5853fb` | Accumulate CPU float32 template power in double precision |
| `07d0427b7d8fbd194a758df4980b813726ce1eba` | Correct CPU point chi-squared phase and accumulation precision |
| `66789ac4a7468094b0cc3ca1498a1de67e0311f6` | Transform CPU float32 strain segments in double precision |

## Scope and extraction decisions

The four corrections and reference tests were extracted from proposed code at
`123e1fb3ef1b338cada636e71c3e9c7987002402`. Only four production function bodies and their required
imports changed; an AST comparison verified the rest of those modules unchanged.
The full diff is seven files, 290 insertions and six deletions.

- `pycbc/psd/__init__.py`: CPU float32 strain is promoted before Welch estimation
  and remains float64 through interpolation and inverse-spectrum truncation.
  The return restores float32 when no explicit precision was requested; explicit
  single/double precision still applies. Only promoted input needs a restoring
  cast, so existing other-backend and float64 paths gain no extra copy. A saved
  PSD remains taken before the public return cast, as in the proposed correction.
- `pycbc/filter/matchedfilter.py`: only CPU float32 power is promoted before the
  cumulative sum. Float64 already accumulates at the desired precision, so the
  proposed unconditional CPU `astype(float64)` copy is unnecessary. Per-frequency
  squared magnitude and PSD division retain their original precision; output
  dtype, frequency spacing and out-of-band zeros remain unchanged.
- `pycbc/vetoes/chisq.py`: the CPU wrapper supplies float64 phase shifts and
  accumulation storage to the existing fused native kernel. Multiplying shifts by
  `numpy.pi / 3.141592653` corrects its shortened pi literal. Final subtraction
  uses complex128 SNR power and restores the NumPy result dtype. Both complex64
  and complex128 correlations need the phase correction. The non-CPU fallback is
  unchanged. No C, Cython, CUDA, or `pycbc/lib` source was edited.
- `pycbc/strain/strain.py`: CPU float32 chunks are promoted before the segment FFT
  and results cast back to complex64. Padding, slice geometry, epoch, frequency
  spacing, analysis metadata, caching and input ownership are preserved. The
  other-backend and float64 paths perform their original transform without an
  added cast.

There is no Torch dependency, import, or scheme check in the seven changed files.
The chi-squared kernel blob is identical in original, proposed and split:
`15dece5585a2d06bc3a17047d81328e981a6044c`.

## Tests and evidence

The three ported CPU-only test files are `test/test_sigmasq_series_precision.py`,
`test/test_chisq_precision.py` and `test/test_strain_psd_precision.py`.

| Check | Result | Evidence |
| --- | --- | --- |
| New precision tests on original source | 11 failed, 8 passed; all four numerical defects exposed | [Baseline log](cpu-baseline-precision-tests.log) |
| Existing PSD, matched-filter and strain modules on original source | 15 passed | [Baseline module log](cpu-baseline-module-tests.log) |
| New precision tests plus those existing modules after correction | 34 passed | [Corrected log](cpu-corrected-tests.log) |
| Existing chi-squared module with cached GW170814 fixtures | 2 passed | [Cached chi-squared log](cpu-cached-chisq-tests.log) |
| New-test flake8 | Passed, no diagnostics | [Log](cpu-flake8-new-tests.log) |
| CI F401 selection for modules/tests | Passed; three unrelated SyntaxWarnings | [Log](cpu-flake8-ci-f401.log) |
| Full flake8 over `pycbc/ test/` | Existing repository style failures; 9,502 output lines | [Full log](cpu-flake8-full.log) |
| Changed production-file flake8 comparison | 701 baseline and 701 corrected diagnostics; none added | [Summary](cpu-lint-summary.json) |
| Git diff whitespace check | Passed | `git diff --check` |

The new tests use an independent per-frequency power reference and long-prefix
sum, direct complex128 phase evaluation at early/late indices, SciPy median Welch
plus NumPy inverse-spectrum truncation, and direct NumPy segment FFTs. References
start from the same rounded input values. Single/double output precision and
segment metadata are covered. On this macOS ARM runtime, longdouble has the same
precision as float64; the test explicitly allows float64 cumulative roundoff.

The legacy chi-squared check is reproduced by
[run-cached-cpu-chisq-tests.py](run-cached-cpu-chisq-tests.py). Its fixture adapter
restricts catalog discovery to locally cached GWTC-1 and replaces file retrieval
with a cache-only lookup that fails on a missing URL. No PyCBC source is altered
by the adapter and no network connection is made. This is a unit-test fixture
check, not the separate exact-workload audit.

## Reproduction and remaining gates

An isolated venv at `/private/tmp/pycbc-cpu-precision-env-20260908` inherits existing
scientific dependencies from `/Users/xangma/miniconda3/envs/pycbc313`. The editable
install used `python -m pip install --no-index --no-deps --no-build-isolation -e .`
in the new worktree. The existing native sources were compiled locally without
editing them. The build is finished; there are no remaining jobs from this task.
[Build log](cpu-editable-build.log) and [build receipt](cpu-editable-build.json)
record host, command, PID and paths.

Runtime: Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, LALSuite 7.26.1,
Cython 3.2.1, pytest 9.0.1, macOS ARM64, FFTW. Both Python and native modules were
verified to load from the new worktree, with no Torch imported. The complete
[receipt](cpu-source-split-receipt.json) contains commit IDs, file SHA-256 values,
kernel blobs and runtime details.

From the new worktree, with that venv's Python:

```python
import sys
import pytest
sys.argv = ["pytest"]  # Existing unittest modules parse process arguments.
raise SystemExit(pytest.main([
    "-q", "test/test_sigmasq_series_precision.py",
    "test/test_chisq_precision.py", "test/test_strain_psd_precision.py",
    "test/test_psd.py", "test/test_matchedfilter.py", "test/test_strain.py",
    "--tb=short",
]))
```

Remaining gates belong to the next validation/publication phase:

1. Primary-owned independent numerical audit on the exact actual-data workload.
   These local unit tests do not establish injection recovery, population
   background, FAR, or production ranking equivalence.
2. Target Linux/compiler/thread configuration validation and runtime cost
   measurement. This local check used macOS, whose native build disables OpenMP.
   CUDA/OpenCL code paths were preserved by inspection and were not executed.
3. Qlty analysis when the tool is available. It is absent from PATH and the usual
   local binary locations; no tool installation or network fetch was attempted.
4. If a PR is subsequently authorized, follow the checked repository template,
   add `agent-assisted`, include "This PR was created by AI Gareth", tag the
   initiating operator, and retain the unchecked Code of Conduct box and required
   AI Agent Note. No PR was created in this bounded task.
