# Original-reference tuning and matched 384-template campaign

This is a finite-workload diagnostic on shared host `len`, not sustained or
reserved-machine throughput. The original CPU source is
`40e94792b3edf59f39b18b65102b28a4f74433a7`; corrected CPU and both Torch routes use
`a4d77a6d1863c0515e8dace64c5609b63d40b51e`. No runtime source was edited for this
campaign. `source.json` records the reused native binaries, whose source and
build definitions were unchanged between these revisions. Per-run receipts
record clean source state, commands, input hashes before/after, and output hashes.

Verify `SHA256SUMS` from the archive root, then follow `archive-transport/README.md`
to reconstruct large raw profiles into a new directory. The logical files retain
their original bytes; the transport inventory records both original and gzip
hashes. The external frame is deliberately excluded and must be restored below.
`summary.json` is accepted only after the campaign and required corrected,
repeat, and profiling parity checks complete successfully. Failures against
original upstream remain in `original-trigger-comparison.json`.

## Inputs and environment

`inputs/bank-compressed.hdf` contains 384 distinct templates: 256 BNS and 128 NSBH,
IMRPhenomD, 30 Hz cutoff, 4096 Hz sample rate. Its SHA-256 is
`26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f`.
`prepare-bank.py`, `compression.json` and `inputs/bank-metadata.json` record
construction, all parameters, and compression. This deterministic performance
set has no minimal-match bank placement, population weighting, tides or disruption.

The external frame `H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf` has 57,824,232 bytes and
SHA-256 `580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865`.
Obtain it from the [official GW170817 CLN release](https://dcc.ligo.org/LIGO-P1700349/public)
or the mirror recorded by the measured source's `examples/inference/single/get.sh`.
Verify the hash before using channel `H1:LOSC-STRAIN`.

Restore clean source checkouts at the exact commits and build them with the
recorded PyCBC/LALSuite/MKL/Torch dependencies. The prior
[source-bundle restoration instructions](https://github.com/xangma/pycbc/blob/639154f7ef359eb473db79660942a2cd88dd59e6/inspiral-reference-20260906/REPRODUCE.md)
provide the final-source bundle chain. `comparison-host-environment.json`, the
qualification records, and `source.json` describe the measured environment.
Rebuilding native extensions need not reproduce binary-identical compiler output;
record new build provenance and retain the new measurement separately.

## Repeating the experiment

Copy the scripts and configuration to a **new** campaign directory. Update paths
in `config.json` (`--frame-files` and `input_files`) and the source paths in stage
scripts. Select an available physical CPU, update `core`, and preserve all analysis
settings and single-thread environment variables. Do not run the stage scripts
against this immutable archive: each case intentionally requires a fresh output.

The original sweep uses segment lengths 256/512/1024 s, start padding 96/112 s and
fixed 16 s end padding, three fresh unprofiled processes per setting. Its recorded
rule selects the lowest median, treating settings within 3% as tied, then choosing
greater start padding and smaller FFT length. The selected geometry is 512/112/16 s.
The longest estimated waveform is 77.68 s; duration bounds alone do not establish
boundary correctness or a global padding optimum. Historical boundary-injection
validation is a separate experiment and is not a fresh check of this full bank.

`comparison-plan.json` records the executed four-backend qualification, separately
observed intervals, twelve unprofiled timing runs, and fresh profiling runs.
`reference-profile-plan.json` records the original CPU profiles collected first.
`reference-accepted.json` records the finite-workload scope accepted before the
matched campaign. Run a single new timing case with a compatible interpreter:

```sh
"$REF_PY" run-case.py --config "$REF_WORK/config.json" \
  --source "$REF_ORIGINAL" --bank "$REF_WORK/inputs/bank-compressed.hdf" \
  --case original-r1 --mode timing --scheme cpu:1 \
  --segment-length 512 --start-pad 112 --end-pad 16
```

Use `--mode qualify` for separate compression/dispatch qualification. Use
`run-observed-case.py --mode filter-timing` for the separately instrumented
template-loop interval and `run-profile-case.py` for filtering profiles; their
other arguments follow the same pattern. CUDA interval boundaries synchronize
the device. None of these instrumented wall times enter the timing medians.
Preserve each output and compare it against that backend's fresh timing output.

The unique valid detector interval totals 1904 seconds. For one worker, completed
work is `384 * 1904 = 731136` template-seconds. Capacity is work/full executable
wall time. Setup and time outside the executable's internal timer remain visible.
The interval timer runs from first FilterBank lookup through final event
consolidation; it is supplementary, not a replacement wall-time denominator.

`perf-report-filtering.txt` selects the recorded CLOCK_MONOTONIC interval from
the full native profile. Its percentages are rounded sampled `cycles:u` event
periods with `--no-children`; cProfile reports additive exclusive seconds. Keep
these denominators separate. Raw `perf.data` remains available via transport.

`comparison-host-samples.jsonl` records per-CPU utilization counters and process
summaries every five seconds during the matched campaign. The original sweep and
original profiles predate this observer. Logical CPU 8's SMT sibling is CPU 72;
affinity did not reserve either CPU. Process `%CPU` is a lifetime average. All
host measurements include the benchmark itself. No idle or 64-worker result is
claimed. `runner-smoke-*` directories exercise scheduling and failure handling
with small synthetic commands; they are not scientific throughput measurements.

## Recomputing the report

From a reconstructed evidence directory with Python, h5py and the same pstats
format support, validate the receipts, HDF output intervals and parity reports:

```sh
python build-controlled-summary.py --root "$REF_DATA" --output "$REF_WORK/summary.json"
```

The documentation's `tools/plot_reference_campaign.py` consumes this summary
offline. Its manifest pins the summary bytes, renderer, archive commit, and four
images. `--verify-only` checks those pins. A partial summary requires `--preview`
and receives a visible watermark; it is not publishable evidence.
