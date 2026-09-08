# Offline CUDA graph trial, 8 September 2026

The fixed CUDA graph experiment passed its prespecified acceptance rule.
Median complete-child wall time decreased from **20.257390 s eager** to
**19.944594 s graphed**, a **1.544107% reduction**. All four paired differences
were positive. The result applies to this 384-template shared-host workload
on `len`; it does not establish CPU parity, steady-state throughput, an idle
machine result, or a general CUDA graph API.

| Pair | Order | Eager full wall (s) | Graph full wall (s) | Eager minus graph (s) |
|---|---|---:|---:|---:|
| 1 | E1, G1 | 20.522021 | 19.995287 | 0.526734 |
| 2 | G2, E2 | 20.246516 | 19.951777 | 0.294740 |
| 3 | G3, E3 | 20.268264 | 19.937412 | 0.330852 |
| 4 | E4, G4 | 20.158632 | 19.932722 | 0.225909 |

Eager range: 20.158632–20.522021 s. Graph range: 19.932722–19.995287 s.
The rule required every paired E−G difference to be strictly positive and
median G below median E. All eight samples were retained. There was no retry,
sample exclusion, resampling, or result-dependent acceptance change.

## Workload and mechanism

Both arms used source commit `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`,
the same accepted MKL descriptor reuse helper, compressed waveform bank,
single-precision calculation, veto settings, PSDs and conditioned strain.
The workload was 384 templates × five segments = 1,920 filter calls over
1,904 valid detector seconds, 4,096 Hz sampling, 512 s segments with 112/16 s
padding, a 1 s symmetric clustering window, and 16 chi-square bins.
The CUDA device was an RTX 4090 with Torch 2.13.0+cu130 and CUDA 13.0.
CPU affinity was core 8; Torch and observed native thread pools used one
thread. Runtime packages, resolved libraries, source, helpers and reference
files were pinned before and after workers and again at closure.

G captured the existing correlation, inverse FFT and symmetric threshold
operations once per segment, then replayed each graph as the template changed.
Threshold conversion to float32 occurred before squaring, matching the eager
arithmetic. This is a Python experiment outside the source checkout. There
were no changes to repository source, CUDA kernels, C extensions or pycbc/lib.
The timing path used the closed workload's established bindings; detailed
per-call validation and replay counters were confined to qualification.

Each timing measured the complete fresh child chain through interpreter exit:
imports, CUDA startup, graph warmup/capture, filtering, vetoes and output,
synchronization, graph reset, descriptor cleanup and wrapper receipts were
included. Controller pin hashing and output comparisons were outside the
sample. Qualification and native-test elapsed times were excluded. These
samples are not just the executable's internal search timer.

Eager and graph median internal runtimes were 15.399736 and 15.090219 s.
Median internal setup durations were 8.387831 and 8.411961 s; median time
outside the internal timer was 4.851134 and 4.856173 s. Median overhead
fractions were 65.47% and 66.44%, so this small fixed workload remains far
from a steady-state throughput claim. Median sample throughput was 36,092.32
and 36,658.36 template-detector-seconds per wall second respectively; both
numbers use the GPU as well as one CPU core and are not CPU-only rates.
Separate medians need not add to the median total.

## Correctness and lifecycle evidence

Fresh native contracts passed before either executable qualification:
six actual captures, 30 actual replays, 17 direct kernel cases, 12 real
controller calls across default/custom CUDA streams, and 48 invalid-state
rejections before any replay attempt. Cases covered the old threshold-rounding
counterexample, strict equality, empty outputs, ties and nonfinite payloads.
The old arithmetic excluded index 35 in the counterexample; the corrected
graph and eager implementation both included it.

Fresh E and G executable qualifications both passed. G performed all 1,920
normal graph replays with exactly five captures, using a fresh eager oracle
for every call. Full correlation and SNR arrays, sparse results and unchanged
input bytes matched on every call. Graph qualification recorded 1,940 Python
IFFT executions: 1,920 eager oracles plus three warmups and one capture for
each of the five graphs. All 384 template identities were visited once.

The qualifier verified 1,309 nonempty returned pairs through the actual
consumer's cumulative-index update. It predicted the post-consumer int64
bytes, required exactly one index tensor version increment, kept sparse SNR
bytes/version fixed, and retained actual Array/backing/tensor objects and
storage bindings. Every recorded actual post-consumer certificate equalled
its prediction. The first and latest live pairs underwent 14,125 retention
checks across entry, oracle, replay and final qualification boundaries.
The archive contains byte certificates rather than every full intermediate
array; executable trigger arrays and qualification PSD arrays are included.

All 25 scientific comparisons passed and were recomputed locally from the
acquired HDF files using the unchanged comparator defaults. Fifteen also
required exact bytes, shape and dtype for all 18 H1 science datasets; four
performance telemetry datasets were excluded from byte equality. These cover
the qualified arms versus the frozen CUDA reference, E versus G, each timed
output versus its own qualified arm, and contemporaneous E/G pairs. The
remaining ten comparisons tie each qualification/timing output to the frozen
standard reference at its existing tolerances. All outputs retained 1,991
triggers. PSD data hashes and conditioned-strain records matched the frozen
scientific reference.

Every completed graph arm synchronized the bound device and reset five
graphs once; eager arms had no graphs to reset. Each run freed its three
cached MKL descriptors once and restored wrapper hooks without cleanup
errors. Whole process group 2469885 was absent before independent nonblocking
reacquisition of the existing benchmark lock. The closure audit confirmed
unchanged source, inputs, helpers and runtime pins. No benchmark remains
running. Existing unrelated GPU workloads were recorded and left in place;
the lock serialized participating benchmark owners, not the entire host.

Qualification-only first-use instrumentation recorded about 28 ms total
across five graph setup/capture events, including validation. Its CUDA memory
allocation grew from 130,026,496 to 226,785,792 bytes, peaking at 380,451,328
bytes; reserved memory grew from 551,550,976 to 616,562,688 bytes. The eager
oracles, snapshots and retained outputs affect those figures, so they do not
measure graph-only memory cost or replace complete-child timing.

## Preserved failed version and audit

Version 1 passed native tests and E qualification, then stopped on the second
graph entry because its qualifier incorrectly required returned indices to
remain unchanged. The executable intentionally performs
`idx += stilde.cumulative_index`. Version 1 captured no post-consumer byte
evidence and ran no timing samples. Its failed outputs remain sealed in
`acquired-diagnostic-v1`; they were not overwritten or counted as performance
data.

Version 2 changed only the consumer-aware qualifier, its controller assertions,
focused CPU tests and associated documentation/verification receipts. Fifteen
local CPU harness tests passed, including missing/wrong/duplicate offsets,
zero-offset version checks, storage/object replacement, value corruption and
cleanup failures. Native helper, timed wrappers, source, science comparator,
configuration, timing order and acceptance rule remained byte-identical to
version 1. Version 2 ran every native and executable gate afresh.

The owner acquisition verified all 167 archive members and 166 result-manifest
hashes before safe extraction, then rehashed the exact file set. `audit-v2.py`
recomputed the 25 science comparisons, sample metrics, paired signs and
acceptance decision into `audit-v2.json`; `offline-cuda-graph-v2-samples.csv`
contains every sample. Its draft first compared Python interval tuples to
their JSON list representation; normalizing that representation fixed the
local audit without changing any frozen input or result.

Independent read-only result review completed with no blocking finding for
this frozen workload. It reproduced archive/input hashes, runtime and lifecycle
receipts, per-call and consumer certificates, the scientific comparison
receipts, all eight timings and the decision. It also reviewed the owner HDF5
recomputation script; it did not perform another remote probe or native run.
The full review is preserved in `peer-results-review-v2.json`.
No integration or publication is included in this experiment.
A broader graph interface would need its own explicit
binding/invalidation and stream/lifetime contract and qualified workloads.

Evidence pins:

- Frozen v2 inputs manifest: `38ca072d1a59c174f029a8f8e13bd43201bff0b2e2d824b967a20f39f1c2d5da`.
- Results archive: `diagnostic-v2-results.tar`, 23,459,840 bytes, SHA256 `6149853dae5d91b48a5aaf61fea5e0621e69791cd2cf23bb020d1e83f316bf1b`.
- Results manifest: `0aff4d8412d81d0448f70e28fbc8b283584670a149989fc49e4a3d040b6953a4`.
