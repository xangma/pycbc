# Torch performance-fix supplement — 6 September 2026

This supplement records a paired comparison of the published batch-1024 source
and three subsequent Python performance changes (v1), followed by an accelerator
peak-reduction refinement (v2), with regression tests,
independent diagnostic profiles and optional-CPU follow-up checks. It supplements
the original batch-1024 evidence; that historical archive and its source/timing
attribution remain intact.

Start with [report/report.md](report/report.md) for the qualified before/after
tables and figures, [cpu-followup-report/](cpu-followup-report/) for the optional
CPU comparison, [cuda-v2-followup-report/report.md](cuda-v2-followup-report/report.md)
for the v2 GPU follow-on, and [profile-summary.md](profile-summary.md) for
diagnostic attribution. [REPRODUCE.md](REPRODUCE.md) explains source reconstruction
and reruns.
All paths here are relative to the extracted supplement root.

The main comparison retains a roughly 6% slowdown for native CPU at batch 32
with four threads, and a roughly 3% lower median paired waveform-generation
rate at CPU batch 8 with one thread (overlapping worker ranges). The additional
[CPU32/T4 profiles](cpu32-t4-diagnostic/) investigate the former; they are
instrumented diagnostics and do not replace its original timing results.

## Source identity

| Role | Commit | Git tree |
|---|---|---|
| Published baseline | `dfd42bf76766cadca0eecf609a1eaeac73534676` | `b0d5ef6e3b39bfe246b4c416528839256edadf1f` |
| Measured v1 candidate on `len` | `97bf1614f3afe53a6e2edb4d8c7dff79e9661782` | `d9819f6f6ebf2998ac967178778cc1911757e455` |
| Local equivalent v1 candidate | `450ab3f96ccea2783abb943b47f698578a507d59` | `d9819f6f6ebf2998ac967178778cc1911757e455` |
| V2 accelerator refinement | `0d00581251e642a5d6b56b2497a9adad93069e6b` | `fa7a7c09df6d93133324f762d756842de172433d` |

The measured and local v1 candidates have identical complete tracked trees.
Their different commit IDs are preserved in the evidence. Later publication
commits can contain additional documentation and CI wiring; the timings belong to the measured
candidate recorded with each run. V2 is a child of local v1 and changes only
`pycbc/filter/matchedfilter.py`: accelerator reductions bypass the Python chunk
bookkeeping retained for CPU reductions. The v1 CPU reduction algorithm is
unchanged. V2 has a different tracked tree.

`source.bundle` contains the complete baseline history without prerequisites.
`performance-fix.patch` contains all six changed source/test files, including both
new regression files. `dependent-sources.bundle` contains local candidate `450ab3`
and the temporary FFT/CPU qualification branches; it requires the baseline commit.
`optional-cpu-baseline.bundle` supplies the published optional-CPU baseline and
also requires that baseline history. `dependent-sources.json` records exact heads,
trees, file hashes/modes, preserved feature patches and native build inputs.
`dependent-sources-v2.bundle` and `dependent-sources-v2.json` separately preserve
the v2 main and FFT/CPU qualification sources, with the same baseline prerequisite.

## Measurement scope

The frozen v1 campaign's complete matrix is **360 worker records**: 252 live-filter
workers, including 36 standard CPU controls, plus 108 waveform workers. It also
requires **108 live parity records**. Batches are **1, 8, 32, 128, 512 and 1024**.
For each configuration, three fresh worker processes run sequentially; the
before/after order reverses for replicate 2. CPU affinity is cores 8–11. CPU
routes use one and four threads; CUDA routes use one host thread.

Live filtering times public `LiveBatchMatchedFilter.process_data` with synthetic
seeded templates/strain, FFT length 131072, three strain blocks, complex64 and
SNR threshold 5.5. Each worker has a separate cold call, one warmup and three
timed iterations. Chi-square and sine-Gaussian vetoes are disabled. Input files,
PSD estimation, bank loading, waveform generation and CLI startup are outside
this component timing. A template/s counts one template against one strain block.
Search plots and tables express real-time capacity as **templates/core** for CPU
and **templates/GPU** for CUDA. Each block represents 56 seconds of analyzed
strain, so the conversion from the preserved raw template-blocks/s is `rate * 56
/ threads` for CPU and `rate * 56` for one GPU. The CPU divisor is the configured
one/four-thread core budget, not measured utilization; affinity cores 8–11 are
distinct physical cores. This is matched-filter component capacity with the
exclusions above, not a complete production-search capacity claim.
The raw live harness's legacy `throughput_wps_summary` unit says
`waveforms/second`; its actual numerator is template count times strain-block
count. The reports interpret this as template-blocks/s, preserving the raw
records and distinguishing them from standalone waveform-generation rates.

Native route labels indicate enabled opt-in routing. Worker metadata records
feature flags and fallback configuration; it does not provide per-call native
dispatch counts. Independent profiles supply separate diagnostic evidence.

TaylorF2 times complete public batch calls with both polarizations, complex128,
4097 frequency bins, 0.25 Hz spacing and 20–1024 Hz limits. Host parameter
conversion is included; copying outputs back to the host is excluded. Each
worker records cold timing separately, then two warmups and five calibrated
timed blocks. Triton and autograd are disabled for this comparison. It does not
replace the historical Triton or unsupported-single-precision evidence.
Its **waveforms/s** unit counts complete generated templates per wall-clock
second, including both polarizations. Waveform generation has no analyzed strain
duration and is separate from the search-capacity measurement.

CUDA timing synchronizes at its boundaries. Profiling and dispatch probes are
outside the main throughput measurements. Reported throughput is the median of
three worker medians, with their observed minimum/maximum. Paired speedup is the
median of the three candidate/baseline ratios, so it can differ from the ratio
of aggregate medians. These ranges are not confidence intervals.

The separate optional-CPU matrix has **36 workers**: baseline native, candidate
native and a baseline standard control at batches 32 and 1024, one/four threads,
and three replicates. Its 12 parity records and actual native-peak gate checks
are stored under `postchecks/cpu-benchmark/`. This smaller follow-up supplements
the six-batch main matrix.

V2 qualification uses a separate follow-on of **36 GPU workers**: two routes,
six batches and three replicates, with its own relevant tests. Comparisons reuse
explicitly identified original baseline/control records; those records are not
new concurrent controls. The v1 campaign and postchecks retain their original
source identities. V2 completion, timing and test claims require the follow-on's
actual records and must not be inferred from the v1 matrix.

V2 records are under `accelerator-refinement/`: `live/`, `parity/`,
`tests/test-{candidate-v2,fft-candidate-v2,cpu-candidate-v2}.xml` with summary
JSON files, `commands/` and `logs/`. `status.json` records completion;
`setup-environment.json`, `sources-before.json`, `sources-after.json` and
`inputs-sha256.json` retain setup, source/binary and reused-input provenance.

## Qualification and provenance

`build-report.py` requires the complete matrix and validates source identity,
raw aggregates, dispatch metadata and parity before emitting the main report.
`report/input-manifest.json` hashes its inputs. Live checks compare triggers,
template IDs, times, SNR, phase, sigma-squared and aggregate output norms with
the corresponding standard control; they do not prove pointwise equality of
every filter sample. Waveform checks cover every row and both polarizations:
batch/native-scalar relative pointwise error at most `2e-10`, native-scalar/LAL
relative L2 below `1e-11`, finite values, exact zero support and metadata. Direct
batch/LAL error diagnostics are recorded without inventing an additional gate.

`test-command.json` records the initial 18-file affected-suite command.
`postchecks/tests/` contains the additional main, FFT-dependent and CPU-dependent
test reports; `postchecks/commands/` supplies actual command/environment records
and logs. Read their result summaries and `postchecks/postcheck-status.json` for
final pass/skip/failure counts and completion, rather than treating a plan as a
completed test. Source and script snapshots before/after execution accompany
these checks. Diagnostic profiles are stored separately in `profiles-baseline/`,
`wave-profiles-baseline/` and `postchecks/profiles/`; their instrumented durations
are not substituted for campaign throughput.

`source-environment.json` and worker records describe Linux 6.8/glibc 2.39 on host
`len`, an AMD Ryzen Threadripper PRO 3995WX, NVIDIA RTX 4090 (24564 MiB), driver
610.57.04, Python
3.11.9, Torch 2.13.0+cu130, NumPy 1.26.4, SciPy 1.13.0, LALSuite 7.21 and Triton
3.7.1. Worker records preserve runtime details. Native shared-library hashes
match between the main baseline and candidate. Dependent preparation records
verify unchanged native build-input bytes/modes and hashes of each reused
binary. Those host-specific binaries and paths are not a portable installation;
the reproduction guide provides a source-build route.
