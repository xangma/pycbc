# Torch search remediation: implementation and evidence

Implementation report dated 2026-09-12. This work repairs scientific dispatch,
candidate ownership, normalization reuse, experimental defaults and evidence
gates in the Torch stack. It does not establish universal reference-decision
equality, production qualification for the TorchWave catalog, or an optimal
GPU algorithm.

The tested PyCBC source is `d0aa34d64cc848b00ebff5d3406a65449817f66e`;
TorchWave is unchanged at `84ef9b3467c8cc34b6d515b967d48646f7297d8f`.
The final target-host suite passed **482 tests, zero skips**, in 38.46 seconds.
The preceding 474-test checkpoint at `6fc453fc` is retained separately and is
not used as the final source's performance evidence.

The acquisition record is [measured results](../artifacts/torch-remediation-implementation-20260912/measurements.md),
with [raw evidence and reproduction instructions](../artifacts/torch-remediation-implementation-20260912/README.md).
A test name below identifies coverage; it does not imply another experiment.
All final numerical and performance execution used `len`; the local machine
was used for editing and Git operations. Three subagents contributed bounded
numerical, provider and evidence work; the lead integrated and restacked it.

## Issue ledger

Owners refer to the existing stack branches: PR4 filtering, PR5 search/CLI,
PR6 waveform, and PR11 evidence. They do not imply a new pull request.

| Severity / owner | Finding and reproducer | Implementation and test disposition |
|---|---|---|
| High / PR6 | The adapter ignored effective scientific options; changing `phase_order` changed reference waveforms while leaving native output unchanged. | Resolve bank and global parameters through the scalar bank's parameter path. Admit only the narrow TaylorF2 options below; unsupported rows use reference generation. `test_torchwave_integration.py` covers option precedence, heterogeneous fallback and complex parity. Implemented; finite sampled coverage. |
| High / PR6 | A complex128 request returned complex64; package availability could enable native generation without opt-in. | Require `enable_torchwave=True`; generate in float64 and honor complex64/complex128 storage, including empty batches. Diagnostics report provider and reason per row. Tests cover disabled/default activation, precision, missing optional dependency and metadata views. Implemented. |
| High / PR4 | A fixed two-bin score rejected a candidate that passed full PowerChisq and NewSNR. | Compatible screening retains every candidate. Diagnostic scoring and uncertified rejection are separate explicit options. The actual veto/ranking counterexample, odd bin counts and tensor-field preservation are regressions in `test_gpu_search_screening.py`. Contained; rejection is not certified. |
| High / PR4 | Rank truncation replaced original norms and could miss its advertised tolerance. | Retain owned original samples/metadata and norms; report aggregate SVD and per-template PSD-weighted residual failures; bind diagnostics to plan and PSD contents. Default search uses retained originals. Rank-cap, stale-PSD, metadata and default-decision tests are in `test_gpu_search_reduced_basis.py`. Contained; residual estimates do not certify reference decisions. |
| High / PR4 | A coarse multirate gate could discard a signal absent from its coarse frequency band. | Default submission delegates to the full-rate engine with its selection policy and vetoes. The coarse gate requires explicit experimental opt-in. `test_gpu_search_multirate.py` includes the absent-coarse-band signal. Contained; selective neighborhood refinement is not implemented. |
| Medium / PR4 | Candidate payloads moved to NumPy before veto evaluation converted them back to tensors. | Internal candidate arrays own their samples on the requested device; native PowerChisq retains device outputs; public drain produces owned host results. Dense-growth, empty-selection, independent veto sums and no-payload-copy tests are in `test_gpu_search_device_contract.py`; adapter/stream tests retain overflow and lifetime coverage. Implemented; trace and elapsed-time conclusions require the acquisition record. |
| High / PR4 | Complex64 cumulative endpoints could lose narrow-bin contributions; the scratch estimate omitted concurrent intermediates. | Use integer modular indices, float64 phases and complex128 bin-local blocked reductions. Enforce an explicit scratch shape budget with the exclusions below. Independent direct sums include large unrelated contributions, narrow/empty/ragged bins and late samples. Portable implementation is bounded; packed bin reductions reduce launch overhead. Further performance optimization remains open. |
| Medium / PR5 + PR6 | Offline provider metadata copied samples to NumPy, persistent reuse excluded native batches, and Live preparation buffered an entire bank on CPU. | Provider returns tensor-backed metadata views; offline batching uses an owned `InspiralSession` cache; Live generates bounded windows on the active scheme's device while preserving row order and reference fallback. Provider, session and actual CLI batch-function regressions cover mutation, ordering, subsets and metadata. Implemented; executable acquisition is reported separately. |
| High / PR5 | Reusing a mutated PSD, including across shard boundaries, could reuse stale `sigma_cached` arrays or template norms. | Hash PSD contents, invalidate inherited PSD-derived fields on first use and mutation, and key scalar norms by waveform identity and PSD digest. Batch normalization hashes once per call and rechecks on the next call. `test_torchwave_session.py` covers zero/nonzero budgets, new/same shards and fresh/reused templates. Implemented; included in the final 482-test run. |
| High / PR5 | Real Live execution exposed rolling-buffer sample counts, padded whitening grids, and output metadata/timestamps that violated downstream contracts. | Validate integral frame-buffer counts; whiten on the actual padded FFT grid; preserve serializable output arrays and timestamps. Dedicated frame/strain tests and `test_tiled_live_output_hdf_contract` cover precision, padding, cache reuse, quiet output and HDF serialization. Implemented; full Live command outcomes belong in the acquisition record. |
| High / PR11 | Verification could report success after failed rows, check only eight rows, and omit final veto decisions; historical claims exceeded their receipts. | Required failures exit 1, unavailable requested devices exit 2, and every requested row is checked. Actual child and graph outputs receive independent gates. Versioned receipts preserve raw repetitions, geometry, provenance and failure status. Adversarial evidence tests reject phase errors, missing identities, failed children, altered inputs/binaries and invalid graph claims. Historical documents now state their precise limitations. Implemented; new measurements remain separately qualified. |

## Provider eligibility and strict fallback

[The provider adapter](../pycbc/waveform/torchwave.py) admits only aligned-spin,
non-tidal TaylorF2 on CPU or CUDA. Its runtime limits are component masses
1–100 solar masses, aligned spins within ±0.99, and
`10 <= f_lower < f_final <= 4096` Hz, with finite physical/grid values.
Accepted orders are phase/spin `-1` or `7`, amplitude `-1` or `0`, and tidal
`-1`. Reference frequency and other waveform options must retain the allowed
defaults; nonzero tides, explicit modes, tapering, unknown options and other
models use the reference generator. The CLI's no-op `taper=None` is accepted.
Compressed generation takes precedence when enabled.

Both entrypoints default to disabled provider dispatch and reject contradictory
enable/disable flags. A mixed bank can contain native and reference rows;
provider diagnostics and output order identify each row. A requested native
qualification requires actual native dispatch for every required row, so a
successful reference fallback cannot silently qualify a native benchmark.

Native synthesis uses the complete zero-origin frequency vector, the adapter's
explicit PyCBC phase convention and dynamic-range-scaled distance, and
per-row inclusive upper/rounded-up lower frequency masks. Duration and timing
metadata follow the reference bank contract. Generation is float64; storage
precision is independently selectable. The current search engine still stores
complex64 filters regardless of provider output precision.

These are runtime admission rules supported by finite fixtures, not a proof
throughout the admitted domain. Strict reference-defined decisions require
reference generation or retained reference waveform samples whenever native
waveform equivalence is unestablished. Refiltering a differently generated
waveform cannot restore that contract. No additional TorchWave model family
is promoted by this work.

## Execution, memory and experimental limits

`InspiralSession` caches owned raw batches independently of PSD-dependent
statistics. Reuse depends on bank contents, resolved parameters, provider code,
grid, device and storage precision. Returned batches are cloned from cache
storage and receive independent metadata views; eviction uses an explicit LRU
byte budget. This bounds retained cache entries, not total process/device
memory or transient return values. Live preparation bounds each generation
window; that does not bound the downstream resident search bank.

Internal candidates and native PowerChisq results remain on device until the
public host boundary. Sine-Gaussian evaluation retains an explicit CPU path.
Dynamic candidate shapes, scalar validation, ragged grouping and edge extents
can synchronize. Consequently, device-resident payloads do not establish a
fully asynchronous pipeline. CUDA graph capture remains limited to
correlation/IFFT; candidate selection and vetoes are outside capture.

The portable veto uses at most 16,384 frequency elements per candidate block
(including its packed bin axis) and accounts for
128 bytes per candidate-frequency element, 128 bytes per candidate-bin and
256 bytes per candidate in its explicit scratch planner. Its default budget
is 64 MiB. The budget excludes caller inputs, O(candidate-count) result/index
arrays, normalized-SNR copies, allocator caches and backend-internal workspace.
Measured peak allocated memory minus baseline must be reported alongside
reserved memory and these exclusions. This is not a claim that total CUDA
allocation cannot exceed the scratch budget. No fused kernel or dense-case
performance advantage is established here.

Reduced-basis residual estimates use original PSD-weighted normalization and
can describe exact-arithmetic error bounds. They omit the complete waveform,
FFT and floating-point error needed to certify reference decisions near cuts
and cluster boundaries. Default dense fallback therefore remains necessary.
The optional reconstructed response workspace still costs O(bank size ×
transform length). Multirate's experimental gate still launches the full
bank after a coarse hit; its refinement-window parameter is not selective
refinement. No global no-loss claim follows from these prototypes.

## Numerical and performance evidence

[The verifier](../tools/verify_torchwave_gpu_search.py) uses fixed reference
waveforms, strain and PSD to check every row's unaligned complex waveform,
PSD-weighted error, original norm and full complex CPU-reference SNR series.
The device engine has separate candidate identity, complex SNR, PowerChisq
and NewSNR acceptance checks. Full-series CPU comparison isolates provider
differences; it is not a full-series CUDA filtering comparison.

[The provider campaign](../tools/bench_torchwave_pipeline.py) checks each actual
first and prepared drain after its timer, then validates child inputs,
dispatch, outputs and execution identity in the parent. Its launch-through-exit
time includes imports, fixture construction, qualification, provenance and
receipt output. Its inner generation-through-first-drain interval is a
synthetic provider-pipeline boundary. Neither is offline or Live executable
latency. Graph speedups require successful capture/replay and equivalent
owned outputs from every measured eager/graph submission.

Receipts record raw samples, input hashes, actual geometry, both imported
repository identities, HEAD-relative patches and relevant untracked source
bytes, plus loaded extension/shared-library hashes and selected FFT backend.
External assets outside that snapshot require a separate manifest. Inspection
of an older receipt never qualifies the current source.

| Required result | Final report status |
|---|---|
| Final tested source, environment and tests | PyCBC `d0aa34d64`, TorchWave `84ef9b346`; 482 passed, zero skipped. `len`: Threadripper PRO 3995WX (64 cores/128 threads), RTX 4090, driver 610.57.04, Python 3.11, Torch 2.13.0+cu130, NumPy 1.26.4, LALSuite 7.21. Full commands and imported binary identities accompany the receipts. |
| Independent before/after veto counterexample | Both implementations pass the clean fixture; final maximum absolute error is `8.47e-7` versus old `7.41e-6`. Unrelated excluded-prefix contamination makes the old result fail (error 19.62); the final result is unchanged and passes. Clean median elapsed time is 2.382 ms final versus 1.229 ms old, with peak new Torch allocation 7,489,536 versus 25,699,328 bytes. This is an accuracy/memory improvement with a measured latency cost. |
| Identical-input prepared generation | All six CPU1/CPU4/CUDA × N=2048/131072 comparisons pass every row's independent gates. At N=131072 on CUDA, median reference/native PyCBC/TorchWave generation is 45.400/11.810/9.371 ms. CPU1 is 44.128/142.302/232.190 ms. These APIs have the preparation/metadata differences stated in the measurements; the result does not establish whole-pipeline acceleration. |
| Actual offline and Live commands | Three offline routes complete two repeated shards each; all nine within/across-route survivor comparisons pass the explicit PyCBC policy, with one survivor per shard. Native CUDA reuses one cached batch on shard two. Live completes its two-detector, two-rank fixture in 6.627 s with all seven checks passing; the high-threshold fixture has no triggers. Neither campaign supplies repeated operational-latency measurements or general Live decision equivalence. |
| Sparse positive-candidate CUDA trace | Three eager cycles, B=2, N=131072, 16 bins; four candidates and two expected accepted injections per cycle. All candidate-array exports occur in public drain. Selection and veto each retain two scalar transfers and two stream synchronizations per cycle. Isolated veto allocation increments are 30,208 bytes at a 64 KiB scratch budget and 3,751,424 bytes at 64 MiB, identical across three repetitions. |
| First/prepared/stream measurements | Five fresh workers per route, 20 prepared drains each. CUDA reference/TorchWave generation through first drain is 391.273/456.001 ms; prepared median is 2.613/2.627 ms. No native advantage is shown for this B=16, N=2048 fixture. A separate zero-candidate graph fixture measures mean eager/graph 1.304/1.048 ms; 500 warmed blocks have p50/p99 latency 1.022/1.053 ms and equal initial/final allocated memory of 15.262 MiB. |

The offline comparison preserves exact identities, schema, dtypes and discrete
metadata. Its explicit policy uses absolute complex-SNR and wrapped-phase
tolerances of `1e-3`, and PowerChisq `atol=rtol=1e-4`; other fields default to
exact equality. Across all nine comparisons, maximum PowerChisq absolute
difference is `2.91824e-4`, wrapped phase difference is `2.32831e-10` radians,
and SNR magnitudes are identical. These are finite survivor comparisons,
not a full-series or near-boundary decision certification. The comparator's
default mode still reports unassessed floating differences instead of
silently applying that policy.

Independent trace inspection matched its SHA-256 and located all 21
`Tensor.cpu` calls, 21 `Tensor.numpy` calls and 21 array DtoH transfers inside
public drain (seven arrays, 176 bytes per cycle). Each cycle launches 29
selection kernels, 704 veto kernels and 12 other submit kernels. The complete
trace has 2,235 kernels, 39 stream synchronizations, six event synchronizations
and two device synchronizations outside the annotated stages. The trace is
instrumented evidence of residency and launch/synchronization costs; clean
timings are collected separately. It covers neither graph capture nor
sine-Gaussian vetoes. The trace's positive-candidate reference gate has zero
sample-index error, maximum complex-SNR error `2.38e-6` and PowerChisq
absolute error `5.80e-4`, within its recorded gates.

Changed source/test/tool files pass F401. The repository-wide F401 run still
reports the same 129 diagnostics as the pinned baseline, with no additions.
Full flake8 style checks are not clean: 1,331 diagnostics in the final changed
file set versus 921 in the baseline's available subset (new files make these
totals non-equivalent). The existing two undefined-name and three unused-local
diagnostics in `bank.py` are unchanged. `qlty` is unavailable on the host and
was not run. Raw logs are included; this is not a claim of a clean full lint
gate.

## Remaining work and disposition

The native provider remains opt-in. Expanding model eligibility requires
independent model-specific waveform and downstream fixtures; this acquisition
does not qualify the rest of TorchWave's catalog. Runtime admission ranges
must not be read as exhaustive accuracy coverage.

Compressed-bank samples retain their existing selected path and are covered
by dispatch regressions. A new compressed-bank performance campaign is unrun:
compression/interpolation introduces a distinct waveform representation and
needs its own fixed reconstruction-error contract before comparing timings
with regenerated waveforms. No compressed-bank speed claim is made.

Broader CPU worker/NUMA tuning, CPU/GPU generation overlap, and native-model
compilation/capture remain unqualified. CPU1 and CPU4 are measured settings,
not the best achievable use of all 64 physical cores. The observed cold
provider cost and lack of prepared-filtering improvement do not justify a
new default or an unmeasured overlap claim.

The portable veto still launches many kernels and is slower on the clean
sparse comparison. A fused or transform-based dense-case implementation
needs separate accuracy, density and elapsed-time evidence; no such kernel
was introduced. Default dense/reference behavior contains the unsafe
screening, reduced-basis and multirate shortcuts while certification and
selective refinement remain open. These dispositions preserve required
outputs without presenting experimental work avoidance as qualified.

Historical results remain in the scoped
[qualification report](gpu_search_qualification_report.md) and
[performance document](torch_performance.rst). They are not substituted for
the final acquisition. Outstanding optimization and unqualified scientific
domains must remain explicit even when implementation tests pass.
