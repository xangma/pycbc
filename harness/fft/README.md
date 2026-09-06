# Fresh FFT and live-dispatch supplement

Measured heads:

- Main: `607bce53ead14f12af32552a5b2441d3bc667267`
- Optional FFT: `e6073eaf1a89cfed69af53707f52321eadf129f1`
- Optional CPU: `1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c`

These are external diagnostic harnesses, separate from the measured source
trees. They do not launch other processes or remote work, except read-only Git
identity commands. Root owns scheduling and environmental observations.

## FFT matrix

Run each cell in a fresh process, with three independent replicates and five
steady execution samples. Each sample averages ten consecutive public IFFT
executions; raw per-sample values are retained. Use the same available CPU
affinity for both heads.

| Head | Cache | Torch threads | Independent processes |
| --- | --- | ---: | ---: |
| Main | off | 1 | 3 |
| Main | off | 4 | 3 |
| Optional FFT | off | 1 | 3 |
| Optional FFT | off | 4 | 3 |
| Optional FFT | cold | 1 | 3 |
| Optional FFT | warm | 1 | 3 |

For each replicate, assign a new dedicated cache directory and run its cold
worker before its warm worker. Never share a cache directory between
replicates, other jobs, or source heads. Off cells need no cache directory.
The cold worker refuses existing wisdom; the warm worker requires existing
wisdom and confirms an actual successful import without rewriting the file.

```sh
PYTHON fft_worker.py --source-root CHECKOUT --expected-revision SHA \
  --label LABEL --output NEW.json --cache-mode cold \
  --cache-dir ISOLATED_PAIR_DIRECTORY --threads 1 \
  --samples 5 --inner 10 --warmups 3
```

The matrix measures one complex64, length-131072, out-of-place public Torch CPU
IFFT. The one-thread admission is the retained direct FFTW plan. Four threads
must select the ordinary Torch IFFT fallback and serve as an off-cache control.
Automatic wisdom is not qualified at four Torch threads, so cold/warm workers
reject that combination. MKL's main-stack direct IFFT applies only at size
32768 and is not the subject of this matrix. The optional CPU branch expands
MKL's supported sizes, which is separately covered by live timing/probes.

Report plan-construction wall time separately from steady IFFT execution.
Construction includes the cache fingerprint, wisdom import or export, FFTW
planning, and allocation done by the public constructor. Cold cache requests
MEASURE (level 1). A warm import returns the requested ESTIMATE (level 0) with
the imported wisdom available. Record these actual levels; do not describe
the warm worker as performing another MEASURE search. Python import, initial
input allocation, and NumPy reference calculation are outside both timers.

Each worker verifies a NumPy complex128 unnormalized IFFT reference using
maximum error normalized to the reference peak and relative L2, each <=2e-6;
it also checks input preservation and repeatable output. An untimed call
observes actual direct FFTW or Torch fallback execution, restoring the original
function before warmup/timing. The output's `status` also requires the expected
dispatch and cache state, so timings from failed qualification cannot support
a speed claim. These checks complement the package tests; they are not a new
scientific accuracy qualification.

Summarize independent workers using the median of worker medians. Show all
three construction samples independently. Do not pool repeated calls as if
they were independent processes or make p99 claims from this small campaign.

## Untimed live dispatch

```sh
PYTHON live_dispatch_probe.py --source-root CHECKOUT --expected-revision SHA \
  --route torch_cpu_native --threads 1 --batch 8 --output NEW_PROBE.json
```

The probe invokes the exact head's committed live child with the production
public call surface. It observes actual return values from CPU/CUDA native
correlation and peak admission, FFTW single/batch execution, and MKL execution.
It preserves return values. All timings printed by this instrumented child
are diagnostic and must be excluded from the timing campaign.

Recommended minimum probes use N=131072, B=8/32: main CPU native at 1/4 threads,
main CUDA native at 1 thread, and optional CPU native at 1/4 threads. Add B=1
CPU/default probes if describing the optional CPU MKL expansion. Successful
admissions, rather than attempts or enabled flags, establish native execution;
zero attempts can mean a route was bypassed earlier. B=1 deliberately bypasses
some batched kernels, particularly CUDA native peaks.

The main `torch_cpu_native` route enables native correlation and FFTW batching.
The optional CPU driver additionally enables native CPU peaks. Comparing these
native rows measures the documented combined configuration, not a single
isolated flag. Default rows remain appropriate main-vs-optional regression
controls. The default-off unsafe `_torch_batch_peak_and_threshold_gpu` helper
must have zero calls; the probe asserts that condition and that its flag is
false. CUDA native peaks use a separate blocking-host-copy implementation.

The committed live benchmark does not configure automatic wisdom, whose
process-local default is disabled. Optional FFT live rows would be regression
controls, not wisdom-cache measurements. The live workload also excludes
waveform generation, input I/O, PSD estimation, and substantive chi-square /
sine-Gaussian veto work; label it as library live-batch filtering.

## Local validation

Both scripts pass parsing, CLI help, and Ruff F401/F821 checks. A local macOS
off-cache four-thread smoke at FFT-format prerequisite `eba814819f1a28279352b7dfa61cf40d6cf8dcfe`
passed numerical parity and observed the expected Torch fallback. Its artifact
`local-smoke-off-t4.json` is harness validation only and must not enter Linux
campaign comparisons. Linux direct FFTW and cache admission remain for the
campaign to establish. No measured source or remote state was changed here.
