# TaylorF2 campaigns through batch 1024

External harnesses and verifiers for exact source revision
`4885b64560e9f39b740e85b6a976898869dd360e`. This directory contains no new timing results.

All defaults use batches **1, 8, 32, 128, 512, 1024**, three independent sequential
worker processes per cell and five timed samples per worker. Successful workers
must use the clean requested source checkout. Timing and parity operations remain
unchanged from the earlier qualified harnesses; this extension changes campaign
coverage, source provenance, capability labels and validation.

| Campaign | Requested matrix | Workers | Expected successful qualification |
| --- | --- | ---: | --- |
| Baseline waveform | 4097 bins; 3 CPU routes at 1/4 host threads; 2 CUDA routes at 1 host thread; double/single | 288 | 48 double precision timed cells (144 workers), 48 unsupported single precision cells (144 workers) |
| Triton off/on | 4097/32769 bins; CUDA batch gate off/on; 1 host thread; double | 72 | 24 timed cells (72 workers) |

Baseline routes are standard CPU/LAL scalar loop, native Torch CPU scalar loop,
native Torch CPU batch, native Torch CUDA scalar loop and native Torch CUDA batch.
`PYCBC_TAYLORF2_TRITON=0` is explicitly set for the baseline. The obsolete unsupported
Triton request was removed. Single precision remains unsupported because the public
CPU/CUDA TaylorF2 API chooses complex128 internally. The distinct Triton experiment
records gate-off/on dispatch, fresh per-worker Triton caches and both frequency grids.

Run each campaign sequentially on an otherwise idle target. Substitute absolute
paths for `PYTHON`, `SOURCE`, `SUPPORT`, `WAVE_OUT` and `TRITON_OUT`; output directories
must be fresh. Every command below requires the exact full source SHA.

```sh
PYTHON -B SUPPORT/waveform/harness/run.py --python PYTHON --root SOURCE --out WAVE_OUT --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e
PYTHON -B SUPPORT/triton/harness/run.py --python PYTHON --root SOURCE --out TRITON_OUT --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e
```

The original qualified Triton harness is also compatible when passed the explicit
`--batches 1 8 32 128 512 1024` argument. Archive the harness actually used. The report
verifies both its worker and orchestrator hashes against the manifest; do not pass
the adapted harness directory when the original was used.

```sh
PYTHON -B SUPPORT/waveform/report/waveform_report.py --input-dir WAVE_OUT --output-dir WAVE_REPORT --harness-dir USED_WAVE_HARNESS --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e --tables-only
PYTHON -B SUPPORT/triton/report/render.py --input-dir TRITON_OUT --output-dir TRITON_REPORT --harness-dir USED_TRITON_HARNESS --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e --run-date 2026-09-06 --tables-only
```

Both report commands need only the Python standard library with `--tables-only`.
Omitting that flag additionally produces figures and requires Matplotlib. The parent
campaign owns the simplified documentation figures. Baseline emits
`waveform-summary.json/md`; Triton emits `report.json/md`. Reports independently
recompute raw samples, worker medians, campaign medians/ranges and ratios, validate
recorded complex parity and dispatch, and reject incomplete matrices or mismatched
source/harness provenance. Neither old smaller campaign can qualify this extension.
The baseline CLI requires all 48 timed plus 48 unsupported cells; the Triton CLI
requires all 24 groups. Missing, failed, timed-out or unsupported measurements never
become zero-valued performance data.

At batch 1024, the two complex128 output tensors occupy about **128 MiB** at 4097
bins and **1 GiB** at 32769 bins. Torch intermediates, retained outputs, CPU parity
references and temporary copies increase peak memory by several GiB. A largest-grid
smoke check on the target GPU should precede relying on full-run completion; OOM or
timeouts remain failed measurements. Workers run sequentially and release memory
when their processes exit. Default worker timeouts remain 180 seconds for the
baseline and 240 seconds for Triton. Each manifest records worker PID, command,
output, log and runtime. Stop the orchestrator with `kill -TERM ORCHESTRATOR_PID`;
both provided runners forward termination to the active worker process group.
The existing Triton runner used with explicit batches has the same stop behavior.

Validation performed here: AST parsing and `--help` for all six scripts, unchanged
baseline measurement/operation/parity function ASTs, byte-identical Triton worker,
expanded matrix counts, unsupported batch-1024 capability record, and rejection of
prior smaller archives. Details are in `validation/checks.json`; no scientific
benchmark or remote execution was performed while preparing these files.
