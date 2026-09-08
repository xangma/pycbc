# Unchanged PyCBC versus the final Torch proposal

This completed 8 September 2026 experiment compares unchanged PyCBC
`40e94792b3edf59f39b18b65102b28a4f74433a7` against the main Torch proposal
`123e1fb3ef1b338cada636e71c3e9c7987002402`. It uses fresh independent native builds and identical
scientific inputs/settings. Optional FFT/CPU follow-up PRs are excluded.

**The timing comparison is complete; scientific equivalence to unchanged
PyCBC fails.** These are descriptive execution costs. They do not establish an
equivalent-output speedup over the baseline.

| Arm | Source | Allocation | Triggers / baseline verdict | Median seconds (observed range) | Template-seconds per wall-second |
| --- | --- | --- | --- | --- | --- |
| Existing CPU | `40e94792b3` | 1 core/thread | 1988 / reference | 67.833 (66.377–76.056) | 10,778.5 |
| Proposed CPU | `123e1fb3ef` | 1 core/thread | 1991 / FAIL | 66.626 (65.531–70.279) | 10,973.7 |
| Proposed Torch CPU | `123e1fb3ef` | 1 core/thread | 1991 / FAIL | 106.597 (102.795–112.295) | 6,858.9 |
| Proposed Torch CUDA | `123e1fb3ef` | 1 core/thread + RTX 4090 | 1991 / FAIL | 20.926 (20.351–26.063) | 34,939.6 |

Four fresh, unprofiled processes per arm ran in balanced rotating orders
ABCD, BCDA, CDAB and DABC. The 384 compressed templates cover 1904 unique H1
seconds (731136 template-seconds), with five 512-second segments and 1920
template/segment pairs. Each timed output passed the frozen trigger comparator
against its own separate qualification. Qualification timings are excluded.

| Arm | Repeat 1 seconds | Repeat 2 seconds | Repeat 3 seconds | Repeat 4 seconds |
| --- | --- | --- | --- | --- |
| Existing CPU | 76.056158 | 67.792582 | 67.873176 | 66.377441 |
| Proposed CPU | 70.279260 | 67.478292 | 65.774402 | 65.530661 |
| Proposed Torch CPU | 108.298524 | 112.294662 | 104.894824 | 102.795405 |
| Proposed Torch CUDA | 21.050668 | 26.062832 | 20.800705 | 20.350912 |

## Scientific results

All four arms completed the required workload and valid interval. Conditioned
strain digests, gating metadata and segment geometry match exactly. The
baseline has 1988 triggers; each proposed arm has 1991. There are 1959 matched
identities, 29 baseline-only and 32 proposal-only identities. Two proposal-only
triggers are threshold-adjacent; that classification does not make them pass.
Among matched triggers, each baseline/proposal comparison has 1951 chi-square,
22 phase and 10 SNR budget violations. Sigmasq and degrees of freedom pass.
The raw per-field errors, unmatched examples and verdicts are retained.

The proposed CPU, Torch CPU and Torch CUDA outputs agree on all 1991 identities
and pass the frozen trigger-field budgets. Their PSD arrays are byte-identical
on the actual filter slice, bins `[15360:1048576]` (30 Hz through the bin below
Nyquist). Their full PSD arrays differ outside that slice: against proposed CPU,
Torch CPU has 2375 finite-bin budget violations and CUDA has 3105, all below
30 Hz. Full-PSD comparison remains FAIL. The original PSD also differs from the
proposal; all five full-array comparisons are retained.

The initial controller stopped on this full-PSD failure before any timing
sample. A separate, reviewed continuation then required proposed trigger
parity, exact conditioned strain/geometry and exact used PSD bins before
collecting descriptive timings. This was an explicit post-qualification
amendment to the timing gate. The original stopped status, script, checks,
tolerances and failed verdicts are unchanged. `verification.json` independently
includes full PSDs and conditioning in scientific eligibility and reports
`equal_output_speedup_eligible: false`.

The executable comparator uses `1e-4` relative and `1e-5` absolute budgets;
sigmasq uses `1e-5` relative with zero absolute floor, phase uses `1e-4` radians,
and trigger identity/sample ticks and degrees of freedom must match exactly.
PSDs use `1e-4` relative and zero absolute floor, with full nonfinite-mask,
shape and dtype checks. Only independently pinned source/executable provenance
fields are normalized for cross-revision comparison; raw verdicts and every
substitution remain saved. These are numerical comparison budgets, not a
physical waveform-accuracy guarantee.

## Timing boundary and resources

The clock spans child process launch through exit after completed HDF output.
It includes imports, frame I/O, conditioning, PSD estimation, compressed-bank
loading, filtering, vetoes, clustering and output, plus the same runtime/hash
verification wrapper in every arm. Parent source hashing and post-run HDF
comparisons are outside the clock; parent host sampling also runs concurrently
with the child. Every worker has a complete
expanded command, pre/post input hashes, runtime observations, output hash and
exit status in `acquisition/runs/<case>/receipt.json` and `runtime.json`.

The shared host is `len`, AMD Ryzen Threadripper PRO 3995WX. Each arm uses CPU 8
with SMT sibling 72 and one numerical thread; CUDA additionally uses RTX 4090
GPU 0. All native and Torch pools are observed as single-threaded. Selectors
are `cpu:1`, `torch:cpu:1` and `torch:cuda:0`, with explicit MKL and default-off
CUDA graph capture. The normal CPU route does not import Torch through the
wrapper. Runtime libraries include MKL 2020.0.4, OpenBLAS 0.3.25 and Torch
2.13.0+cu130. Both source trees use the same Python 3.11.9 environment.

The distribution inventory records Torch 2.1.1 because its name-keyed dictionary
retains the later entry from an inherited Conda installation. Every Torch
worker records the imported version as 2.13.0+cu130. A read-only diagnostic
after acquisition confirms both installations, first-match import selection,
and selected files against their wheel RECORD hashes. The verifier explicitly
reconciles these records; the original inventory remains unchanged. This
post-run inspection cannot retrospectively hash the Torch binaries loaded by
the workers.

The host, sibling CPU and GPU are not reserved. The archive retains CPU/load,
process, memory and GPU observations before/during/after every process. No
sample is discarded. Filesystem caches were not flushed. The displayed ranges
are observed minima/maxima of four samples, not confidence intervals. This
finite workload does not establish sustained or full-machine capacity.

## Evidence and reproduction

- `acquisition/config.json`: all scientific arguments, resource controls,
  original scheduling policy, input hashes and exact source commits.
- `acquisition/source-pins.json`, `*-build.json`, build logs and runtime receipts:
  clean source/native build identities and imported module hashes.
- `acquisition/runs/`: all four qualifications and 16 timed outputs, HDF files,
  full qualification PSD arrays, commands, logs and host/GPU observations.
- `acquisition/comparisons/`: original raw comparisons, declared provenance
  substitutions, full conditioning checks and every repeat comparison.
- `acquisition/status.json`, `campaign.log`, `continuation-policy.json`,
  `resume-timings.py`, `timing-status.json`: the stopped qualification and the
  explicit descriptive timing continuation.
- `verification.json`, `verify-results.py`: independent recomputation of the
  completed schedule, evidence hashes, scientific results and timing table.
- `committed-source-verification.json`: every recorded tracked file verified
  against the actual Git commit (1078 original and 1227 proposed files).
- `committed-source-proof.json`, `verify-source-commits.py`: repeatable Git
  verification bound to the SHA256 of the acquired source-pins manifest.
- `acquisition/torch-environment-diagnostic.json` and its inspection script:
  supplemental read-only observations after acquisition, resolving the
  distribution-inventory versus imported-Torch version discrepancy.
- `SHA256SUMS`, `verify-archive.py`: sealed file inventory and byte verification.

See [REPRODUCE.md](REPRODUCE.md) for clean builds, input restoration, relocated
acquisition and offline verification. The frozen bank is retained in the same
evidence branch's `reference-campaign-20260907/inputs`; the external GWOSC/LIGO
frame is restored by SHA256. Neither physical host state nor unstored strain
samples can be reconstructed from this archive; strain equality is supported
by the recorded digests and metadata, and imported source bytes by acquisition
pins plus the independent Git-content check.
