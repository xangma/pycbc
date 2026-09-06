# Batch sizes through 1024: measurements and correctness

This supplement extends the batch-based waveform and live matched-filter
campaigns to **1, 8, 32, 128, 512 and 1024**. Earlier evidence remains unchanged
in the parent archive. The single-vector FFT and single-evaluation inference
measurements retain their original revisions; the new correctness suite also
exercises FFT, filtering and likelihood operations with 1024 rows.

## Source and environment

The measured assembled main revision is
`4885b64560e9f39b740e85b6a976898869dd360e`. Optional native CPU comparisons use
`d544420232428225c214a4be84fbe1262a6d307b`. The preparation record also includes
the optional FFT revision `d840198592a0ca128f4f89d90987a7469ec37c8f`.

The runner is `len`, Linux, Threadripper PRO 3995WX, RTX 4090, Python 3.11.9,
Torch 2.13.0+cu130 and Triton 3.7.1. Workers are sequential with affinity 8–11;
CPU configurations use one or four threads and CUDA uses one host thread.
These are manual measurements, not GitHub Actions results. `preparation.json`
and `finalization.json` bind clean source checkouts and native extension hashes
to the runtime. Raw records retain commands, source/module hashes, inputs,
samples, parity, runtime identity and actual route evidence.

## Acquire new measurements

Use new output directories. Install the recorded dependencies, check out and
build the exact revisions (`python setup.py build_ext --inplace -j 4`). Never
copy extension binaries from a different source revision. Substitute your own
source, interpreter and output paths below; retain the expected revision flags.

The original Triton harness runs both flag settings on the same main source:

```sh
taskset -c 8-11 python triton-original/run.py \
  --root MAIN_CHECKOUT --python PYTHON_EXECUTABLE \
  --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e \
  --batches 1 8 32 128 512 1024 --out NEW_TRITON_OUTPUT
```

`triton-launch.json` records the original absolute paths. Frequency grids are
4097 and 32769 bins, with three fresh workers per route/grid/batch.
Every worker uses a fresh Triton cache. Generated cache binaries are omitted
from this portable archive; their inventories and hashes remain in raw records.
The CUDA driver cache was retained. Warm work is five synchronized groups of
at least 50 ms; cold time is the first complete public call.

The baseline waveform harness keeps Triton disabled and retains scalar loops
and the public batch API as separate routes:

```sh
taskset -c 8-11 python wave-support/waveform/harness/run.py \
  --root MAIN_CHECKOUT --python PYTHON_EXECUTABLE \
  --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e \
  --batches 1 8 32 128 512 1024 --out NEW_WAVEFORM_OUTPUT
```

The full baseline requests 288 workers: 144 double-precision timed runs and
144 explicitly unsupported single-precision requests. Timing/parity calculations
and inputs are unchanged from the previous baseline; the updated acquisition
code binds the current source and its available-but-disabled Triton evaluator.
See `wave-support/README.md` for the harness changes and validation.

Run live matched filtering and its separate, untimed dispatch probes following
`live-support/README.md`. The fixed workload uses three blocks, FFT length
131072 and all six batches. Its complete matrix is 84 groups/252 fresh workers;
60 separate probes check actual native admission and full logical batch buffers.
Controllers retain failures, unsupported routes and memory/time-limit outcomes.
Five-second telemetry accompanies live and probe acquisition; the Triton and
baseline waveform runs retain per-worker hardware records rather than a claim
of continuous telemetry.

The matched-filter benchmark checks triggers and aggregate output norms.
Its separately recorded 1024-row correctness suite checks complete complex
outputs on short transforms. These are distinct checks with distinct scopes.

## Recompute reports

Run from this supplement directory. The report engines verify complete expected
matrices, raw file/receipt hashes, exact source, harness identity, timing and
parity before admitting performance comparisons.

```sh
python report-engines/triton/report/render.py --input-dir triton \
  --output-dir report/triton --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e \
  --harness-dir report-engines/triton/harness --run-date 2026-09-06 --tables-only
python report-engines/waveform/report/waveform_report.py --input-dir waveform \
  --output-dir report --expected-sha 4885b64560e9f39b740e85b6a976898869dd360e \
  --harness-dir report-engines/waveform/harness --tables-only
python report-engines/live/live_report.py --input-dir live --output-dir report \
  --main-revision 4885b64560e9f39b740e85b6a976898869dd360e \
  --cpu-revision d544420232428225c214a4be84fbe1262a6d307b --tables-only
python report-engines/live/probe_report.py --input-dir probes --output-dir report \
  --main-revision 4885b64560e9f39b740e85b6a976898869dd360e \
  --cpu-revision d544420232428225c214a4be84fbe1262a6d307b
```

## Correctness and figure reproduction

`run-correctness.py` records the exact commands, source imports, revisions,
environment and exit codes for all three checkouts. `correctness/` retains pytest
logs and JUnit records. `test_torch_large_batches.py` runs 1024-row checks on
real CPU/CUDA devices with bounded transform sizes; unavailable routes are
reported as skips. It is copied outside the measured checkouts so the benchmark
source revisions remain clean.

The expanded checks exposed a disabled-chi-square fallback crash. Successful
correctness runs use separate clean commits: each measured revision plus the
single guard in `correctness/NAME-runtime.patch`. `correctness-preparation.json`
records each base, corrected revision, runtime hash and native binary hashes.
To reproduce, build the exact base, apply its named patch with `git apply`,
verify `pycbc/filter/matchedfilter.py` against the recorded hash, and run the
pytest command from `correctness/status.json` with your local paths. The Python
guard changes no extension source, so that base's built extensions remain valid.
Initial fixture failures and the subsequent confirmed runtime failures are
retained in `correctness/initial-failed/` and `correctness/disabled-veto-failed/`.
The performance measurements retain their original revisions and their stubbed
SG veto does not enter the corrected bulk fallback.

The standalone `documentation/render-manifest.json` binds all five summary
inputs, including unchanged historical inference/FFT summaries. It intentionally
has no commit pin to its own archive. From this supplement directory, run:

```sh
python documentation/plot_torch_benchmark_docs.py --archive .. \
  --manifest documentation/render-manifest.json --output NEW_FIGURE_DIRECTORY
```

The renderer checks summary hashes and complete six-batch matrices before
plotting. `documentation/figures/` contains all eight rendered PNGs and the
renderer output manifest. The source documentation manifest can pin this
archive only after the archive commit exists. Do not modify the sealed remote
report scripts; `report-engines/` retains the later strict engines and their
required acquisition files separately. Whiskers and bands show observed worker
ranges, not confidence intervals. Results do not establish whole-search,
sampler, MPS or other-hardware performance.

`sealed-manifest.json` records the transfer inventory by SHA-256;
`SHA256SUMS` covers the published supplement. Timing and correctness outcomes
are recorded in the completed reports and receipts.
