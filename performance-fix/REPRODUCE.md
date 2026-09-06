# Reproducing the performance-fix supplement

Use a new working directory and retain the extracted archive unchanged. The
source bundles reconstruct the tested code without relying on moving GitHub
branches. Runtime paths in JSON command records describe the original host;
they must be mapped to your own checkout, interpreter, output and CPU affinity.

## Reconstruct the v1 sources

Set the following to absolute paths. `PERF_WORK` must be a new reproduction
directory. These commands create local checkouts; they do not contact a remote.

```sh
PERF_ARCHIVE=/absolute/path/to/extracted-supplement
PERF_WORK=/absolute/path/to/new-reproduction
mkdir -p "$PERF_WORK"
git clone --no-checkout "$PERF_ARCHIVE/source.bundle" "$PERF_WORK/objects"
git -C "$PERF_WORK/objects" bundle verify "$PERF_ARCHIVE/source.bundle"
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/baseline" dfd42bf76766cadca0eecf609a1eaeac73534676
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/patch-check" dfd42bf76766cadca0eecf609a1eaeac73534676
git -C "$PERF_WORK/patch-check" apply --check "$PERF_ARCHIVE/performance-fix.patch"
git -C "$PERF_WORK/patch-check" apply --index "$PERF_ARCHIVE/performance-fix.patch"
git -C "$PERF_WORK/patch-check" diff --cached --check
git -C "$PERF_WORK/patch-check" diff --cached --name-only
git -C "$PERF_WORK/patch-check" write-tree
```

The staged path list must be exactly:

```text
pycbc/filter/matchedfilter.py
pycbc/filter/matchedfilter_torch.py
pycbc/waveform/taylorf2_torch.py
test/test_live_batch_torch_peaks.py
test/test_torch_batch_overlap_scaling.py
test/waveform/test_taylorf2_phase_evaluation.py
```

The resulting tree must be `d9819f6f6ebf2998ac967178778cc1911757e455`.
This proves that the standalone baseline plus `performance-fix.patch` reconstructs
the complete measured candidate tree, including the two new test files. The
smaller/intermediate `candidate.patch` is not the reconstruction input.

For exact existing commit identities, import the incremental bundles after the
baseline. They require `dfd42`; cloning them alone is insufficient.

```sh
git -C "$PERF_WORK/objects" bundle verify "$PERF_ARCHIVE/dependent-sources.bundle"
git -C "$PERF_WORK/objects" fetch "$PERF_ARCHIVE/dependent-sources.bundle" '+refs/heads/*:refs/remotes/dependent/*'
git -C "$PERF_WORK/objects" bundle verify "$PERF_ARCHIVE/optional-cpu-baseline.bundle"
git -C "$PERF_WORK/objects" fetch "$PERF_ARCHIVE/optional-cpu-baseline.bundle" '+refs/heads/*:refs/remotes/cpu-baseline/*'
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/candidate" 450ab3f96ccea2783abb943b47f698578a507d59
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/fft-candidate" d6407d32742a57e4f026c461a26ef7b3929c5849
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/cpu-candidate" 1514327669fc7be125523b991c847868c3a2a17e
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/cpu-baseline" bd53914be6d2e4324cc867d52b3842b77cc6729a
```

`candidate` at `450ab3` has the same complete tree as measured commit `97bf1614`.
Do not label a new run as `97bf1614`: record its actual `450ab3` HEAD and matching
tree. If using only the standalone bundle and patch, commit the reconstructed
tree with your configured Git identity and record that new HEAD instead.

The dependent tree IDs are `4adcca5ff8c9ab3aa6769ff0cced6f7da19c2fa7` (FFT) and
`0bbf15de82c467649cb88fb6378aaf74d829cfca` (CPU). They are temporary qualification
branches, distinct from the final one-commit-per-PR publication stack.

## Reconstruct the v2 refinement

V2 changes only accelerator peak-reduction control flow in
`pycbc/filter/matchedfilter.py`; its parent is local v1 `450ab3`, and its CPU
reduction algorithm is unchanged. Retain all v1
checkouts and results. Import the separate v2 bundle into the same object store:

```sh
git -C "$PERF_WORK/objects" bundle verify "$PERF_ARCHIVE/dependent-sources-v2.bundle"
git -C "$PERF_WORK/objects" fetch "$PERF_ARCHIVE/dependent-sources-v2.bundle" '+refs/heads/*:refs/remotes/dependent-v2/*'
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/candidate-v2" 0d00581251e642a5d6b56b2497a9adad93069e6b
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/fft-candidate-v2" c3202b4b0b681a1ace527d1e6231c47470856ba6
git -C "$PERF_WORK/objects" worktree add --detach "$PERF_WORK/cpu-candidate-v2" 665fa5a0f41946c0c2873d8a79a16c8b6975a090
```

The main v2 tree must be `fa7a7c09df6d93133324f762d756842de172433d`.
The FFT v2 tree is `e5f773b444186f267d4825f59b3d2c50cee31010`; the CPU v2 tree is
`225b9b0f29832057511312db81b73ec6bae52195`. Check these against
`dependent-sources-v2.json`. This bundle requires
`dfd42` and includes v1 history. `performance-fix.patch` remains the complete
v1 reconstruction input; it must not be relabeled as the v2 tree.

## Prepare a compatible runtime

The recorded runtime is Linux x86-64, Python 3.11.9, Torch 2.13.0+cu130, NumPy
1.26.4, SciPy 1.13.0, LALSuite 7.21 and Triton 3.7.1; GPU timing used an RTX 4090
with driver 610.57.04. See `source-environment.json` and worker records for the
hardware and actual loaded runtime. Install a compatible CUDA-enabled Torch
build and driver, and the PyCBC dependencies/build prerequisites described by
each checkout's `install`, `setup.py` and `pyproject.toml`, including a compiler
and the applicable FFTW/GSL/OpenMP libraries. Pin the recorded package versions
when reproducing that environment; different hardware/builds are a new campaign.

Build PyCBC and its extensions from each reconstructed source in an appropriate
environment. A normal editable source installation, after those prerequisites
are available, is:

```sh
PERF_PY=/absolute/path/to/compatible-environment/bin/python
"$PERF_PY" -m pip install -e "$PERF_WORK/candidate"
```

Repeat the build for the baseline and dependent checkouts using suitable isolated
environments, or a carefully controlled common compatible environment. Run each
worker with its intended checkout as cwd/import root and verify `pycbc.__file__`,
native-module origins, clean tracked source, actual HEAD/tree and extension
hashes. Recheck source cleanliness after building. New binary hashes need not
match the original host; record them with compiler/runtime provenance.

The original `setup-dependent.py` copied 11 existing native shared libraries
from earlier `len` FFT/CPU checkouts only after checking every relevant native
source/build-input byte and Git mode, then hashing the copies. Its `R`, `OLD`
and interpreter paths are specific to that host. Neither that path layout nor
the old binaries are prerequisites for a fresh source build. Do not blindly
run the setup script on another system or describe copied binaries as rebuilt.
For prebuilt reproductions, use the same input/ABI/hash checks and create a new
accurate setup receipt; otherwise run the listed tests/workers directly against
your fresh builds.

## Re-run the frozen v1 comparison

`run-comparison.py` is the exact campaign orchestrator, with fixed original `R`
and `P` paths. Make a separately named working copy and replace only those path
constants with the new campaign directory and compatible interpreter. Place or
link the reconstructed `baseline` and `candidate` at those locations, copy the
archived `waveform-worker.py`, and use new empty result directories. The runner
records actual source HEADs, so it can record reconstructed `450ab3` correctly.
Retain and hash the adapted script; never overwrite archived raw results or
reuse an old completion file as evidence of a new run.

The complete matrix is:

| Component | Configuration |
|---|---|
| Batches | 1, 8, 32, 128, 512, 1024; recorded execution order 1024, 1, 32, 128, 512, 8 |
| Live, 1 thread | Torch CPU/CUDA production routes and their native opt-in routes, each before/after |
| Live, 4 threads | Torch CPU production/native routes, each before/after |
| Live controls | One baseline standard-CPU worker for every batch/thread/replicate |
| Waveforms | Torch CPU batch at 1/4 threads and CUDA batch at 1 thread, each before/after |
| Replication | Three fresh processes; replicate 2 reverses before/after order |

Workers run serially, originally pinned with `taskset -c 8-11`. Choose available
cores on a different machine and record that change consistently. Preserve
thread settings and the harness's `route_environment` feature-flag configuration.
Do not run profiles, tests or unrelated GPU jobs concurrently with this matrix.

Live workers use the checkout's `tools/bench_production_live_batch.py child`
entrypoint, FFT size 131072, three blocks, three timed iterations, one warmup,
threshold 5.5, seed 7101 and `--call-surface public`. Waveform workers use
`--samples 5`, default 50 ms calibration blocks and maximum 64 inner calls;
the runner supplies the actual expected source SHA. Both use synchronized CUDA
boundaries and separate cold observations. Preserve all route/parameter values
from the scripts, including disabled vetoes and disabled waveform Triton/autograd.

The frozen live harness uses `blocksize=56.0` seconds at 2048 Hz, within each
64-second FFT. Search reports convert raw template-blocks/s to templates/core
at real time by multiplying by 56 and dividing by the configured CPU thread
count; CUDA reports multiply by 56 for templates/GPU at real time. Use the actual
analyzed duration and core budget for any new workload. Do not apply this
conversion to standalone waveform generation, which remains waveforms/s.

A finished campaign contains 252 live JSON records, 108 waveform JSON records,
108 live parity records and a complete `comparison-status.json`. Completion and
parity must be verified; file counts alone do not qualify a result.
`build-report.py` validates every worker against the actual source identities in
that run's status record. It can therefore validate a new run with a reconstructed
candidate identity while retaining the full matrix, aggregation and parity gates.

To independently rebuild the report from the unchanged archived records:

```sh
"$PERF_PY" "$PERF_ARCHIVE/build-report.py" --input "$PERF_ARCHIVE/comparison" --status "$PERF_ARCHIVE/comparison-status.json" --out "$PERF_WORK/rebuilt-report"
```

Keep `waveform-worker.py` and `run-comparison.py` beside the renderer: it checks
the worker hash against the records and includes the local orchestrator hash in
its manifest. For a new run, retain the adapted orchestrator in that sibling
location as well. The renderer needs Matplotlib and refuses an existing output
directory. Consult
`report/report.md`, `report/report.json` and `report/input-manifest.json`; retain
slower cases and use paired replicate ratios, not a selected best run.

## Profiles, tests and optional CPU checks

Profiles are separate diagnostic runs. Per source, `profile-live.py` covers
four routes at batches 32/1024; `profile-waveform.py` covers CPU batches 32/1024
and CUDA batches 1/1024, all with one thread. They emit cProfile statistics,
operator summaries and trace JSON. The waveform profile's seven preliminary
timing samples are separate from the campaign's five-block workers; neither
instrumented live timings nor operator durations enter the reported speedups.
See `profile-summary.md` and the baseline/postcheck profile directories.

`run-cpu32-diagnostic.py` records two additional CPU-native profiles at batch 32
with four threads, using the baseline and v2 sources. It requires completed v2
timings so diagnostics cannot overlap the measurement campaign. Its
`cpu32-t4-diagnostic/` output includes actual commands, source/native hashes,
logs and profile hashes. Adapt paths and prerequisites explicitly for new
sources, and retain fresh provenance; these profiles are diagnostic attribution,
not another throughput replicate.

Run the initial affected suite using the 18 selectors in `test-command.json`,
replacing its original interpreter/cwd/affinity and preserving the selectors.
`run-postchecks.py` adds ten TaylorF2-descendant/inference files on the main
candidate, repeats the 18 affected files plus five FFT files on the FFT candidate,
and repeats the 18 affected files plus seven CPU files on the CPU candidate.
Its `--plan-only` option shows commands without running them. The actual
`postchecks/plan.json` and `postchecks/commands/` records take precedence over a
preliminary `postchecks-plan.json`, whose absolute paths may describe the machine
where that plan was drafted.

For a new run, use a working copy of the postcheck runner with explicitly updated
main-candidate HEAD expectations; do not disable source checks or retain the
original `97bf` label for `450ab3`. It supports `--root`, `--python`, `--output`
and `--skip-setup`, but skipping setup still requires a valid matching
`dependent-environment.json` receipt. Fresh source builds can instead execute the
listed test and benchmark commands directly with their own provenance records.

The optional CPU check uses batch 32/1024, one/four threads, three replicates and
three workers per group: baseline standard, baseline `torch_cpu_native`, and
candidate `torch_cpu_native`. That is 36 workers and 12 parity records. Preserve
the explicit optional native-peak gate and record admission/fallback, thread
settings, module origins and binary hashes. Results live under
`postchecks/cpu-benchmark/` and `cpu-followup-report/`.

To rebuild the optional-CPU report from the unchanged archive:

```sh
"$PERF_PY" "$PERF_ARCHIVE/build-cpu-followup-report.py" --input "$PERF_ARCHIVE/postchecks/cpu-benchmark" --out "$PERF_WORK/rebuilt-cpu-report"
```

Retain its sibling reporting/runner scripts and the complete `postchecks/`
directory. This validator pins the two archived CPU commit IDs; a reproduction
with different commits requires explicitly reviewed expectations and accurate
new source/setup receipts.

For all tests, retain stdout/stderr, exact command/environment, return code,
JUnit reports and parsed pass/skip/failure counts. Inspect `postcheck-status.json`
and before/after source/script hashes. Test plans and historical results are
not new runs, and source-tree equivalence is not a claim that every test ran on
the later publication commit.

V2 has a separate 36-worker GPU follow-on: production/native routes, all six
batches and three replicates. Preserve its actual source identity, commands,
completion ledger, relevant test outputs and links/hashes for reused original
baseline/control records. Do not pool those measurements into the v1 campaign
or present reused controls as fresh workers. Repeating both sides on new hardware
is a new campaign and should retain its own complete provenance.

`run-accelerator-refinement.py` records that follow-on under
`accelerator-refinement/`. Read `status.json`, worker `live/` and `parity/`
records, `commands/*.json`, `logs/*.log` and
`tests/test-{candidate-v2,fft-candidate-v2,cpu-candidate-v2}.xml` with their
summary JSON files. `setup-environment.json` records preparation;
`sources-before.json` and `sources-after.json` preserve source/native hashes;
`inputs-sha256.json` freezes the reused controls, original status, bundle,
manifest and scripts. The separate summary is
`cuda-v2-followup-report/report.md`. Adapt any original paths only in a working
copy of the runner, retain the archived copy, and apply the same source/build
checks described above to its three fresh v2 checkouts.
