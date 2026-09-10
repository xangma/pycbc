# GPU search engine implementation plan

Prepared 2026-09-09. Source inspected at `1229a912b`.
Audience: a coding agent implementing a shared engine for `pycbc_inspiral`
and `pycbc_live`. This is a plan; no engine changes or new performance
measurements accompany it. Recheck the current checkout before implementation.

## 1. Outcome and implementation order

Build a persistent GPU search engine whose unit of work is a data block and
a bank tile. The eventual algorithm should use duration-specific, multirate
filtering and, where profitable, local reduced bases. Keep template preparation,
normalization, filtering, candidate selection and batched veto calculation on
the device between deliberate input/output boundaries.

Implement this in two tracks with separate acceptance criteria:

1. **Compatible engine:** change scheduling, storage and batching while retaining
   the current scientific settings, full template bank, search statistics and
   established numerical/trigger contracts.
2. **Algorithm research:** introduce multirate decomposition, reduced bases and
   optional early rejection. Qualify their approximation errors, sensitivity
   and background independently before production use.

Do not make approximate candidate generation a prerequisite for the compatible
engine. A rejected candidate cannot be recovered by an accurate final veto.
Do not promise a particular speedup or GPU utilization level.

| Milestone | Deliverable | Depends on |
| --- | --- | --- |
| M0 | Frozen contracts, reference outputs and measurement harness | Current source |
| M1 | Versioned bank/data plans and bounded, eager tiled filtering | M0 |
| M2 | Device candidate queues and correct clustering/peak policies | M1 |
| M3 | Batched full-statistic veto evaluation | M2 |
| M4 | Offline and live adapters with unchanged output contracts | M3 |
| M5 | Measured graphs, stream overlap and waveform preparation choices | M4; waveform experiment can start after M1 |
| M6 | Multirate/multiband filtering with original-template refinement | Compatible engine qualified |
| M7 | Conditional reduced bases and cheap consistency screening | M6; independent feasibility work may start earlier |
| M8 | Sustained workload qualification and controlled rollout | Each selected route passes its gates |

Make each milestone a reviewable change with retained test/benchmark receipts.
Keep failed experiments and their input/source identities. Fix failed gates
before claiming equivalent-output performance; do not relax a tolerance to
complete a milestone.

Use the [PR delivery and reuse map](#8-pr-delivery-and-reuse) to turn these
milestones into reviewable changes alongside the existing Torch PRs.

Follow [AGENTS.md](../AGENTS.md), including its explicit-instruction requirement
for low-level C/CUDA kernel changes. The kernel work described below is a
design target, not an override of that requirement. Preserve CPU operation
without Torch installed and avoid adding LALSuite dependencies. Follow the
repository's PR rules if a PR is subsequently requested.

## 2. Starting points and constraints

Verify these symbols directly; this table describes the inspected source,
not a guarantee about a later branch.

| Existing location | Reuse or replace |
| --- | --- |
| [bin/pycbc_inspiral](../bin/pycbc_inspiral), `template_triggers` | Replace per-template/per-segment orchestration through an adapter. Preserve injection filtering, segment validity, veto options and event consolidation. Data FFTs and overwhitening are already reused. Templates are already reused across segments. |
| [pycbc/filter/matchedfilter.py](../pycbc/filter/matchedfilter.py), `MatchedFilterControl` | Reference normalization, filtering and clustering behavior; current offline workspaces hold one correlation/SNR row. |
| Same file, `LiveBatchMatchedFilter` | Reuse established batch behavior and public result contracts. It already batches correlation/IFFT; peak results return to the host, and `_process_vetoes` iterates over triggers. |
| [pycbc/events/threshold_torch.py](../pycbc/events/threshold_torch.py) | Reference threshold, tie and symmetric clustering behavior. Replace host-dependent dynamic compaction in the new inner loop. |
| [pycbc/vetoes/chisq.py](../pycbc/vetoes/chisq.py), `SingleDetPowerChisq` | Preserve binning, activation, normalization, degrees of freedom and output conventions. Avoid per-call CPU bin-edge round trips in the engine. |
| [pycbc/vetoes/chisq_torch.py](../pycbc/vetoes/chisq_torch.py) | Existing compatibility and mathematical reference routes. The compatibility CUDA kernel's thread 0 advances the phase through the entire bin before parallel partial sums; chunking has not removed that serial work. |
| [pycbc/waveform/bank.py](../pycbc/waveform/bank.py), `FilterBank` | Bank metadata and decompression/generation semantics. The current shared output buffer cannot serve as immutable storage for outstanding tiles. |
| [pycbc/waveform/decompress_torch.py](../pycbc/waveform/decompress_torch.py) | Existing decompression implementation; profile preparation separately from repeated filtering. |
| [pycbc/fft/torchfft.py](../pycbc/fft/torchfft.py), [pycbc/filter/_torch_cuda_graph.py](../pycbc/filter/_torch_cuda_graph.py) | Reuse FFT scaling, plan/workspace and capture-lifetime lessons. Do not assume old capture eligibility covers the new engine. |

The recorded offline result is 19.60 s versus 65.15 s for the original CPU
on a finite, shared-host workload. The recorded live result is a prepared API
experiment with different inputs and timing boundaries. These are starting
evidence, not full-machine capacity estimates. See
[performance results](torch_performance.rst),
[offline protocol](torch_benchmark_protocol.rst) and
[live numerical protocol](torch_batch_numerics.rst).

Measure new stage costs with CUDA events/profiling. Host time around an
asynchronous launch is not kernel duration. Neither a vector fitting in L2
nor its transform length proves occupancy or the best batch size.

## 3. Engine contract

The following package and API names are **proposed**, not existing interfaces:

~~~text
pycbc/filter/gpu_search/
    plans.py        bank, PSD, geometry and capability descriptions
    engine.py       submit/drain/flush lifecycle and workspace ownership
    scheduler.py    tile selection, streams, deadlines and backpressure
    candidates.py   candidate buffers and selection policies
    vetoes.py       batched veto dispatch and correlation lifetimes
    adapters.py     existing offline/live interfaces and output translation
    multirate.py    research band plans and coherent reconstruction
    basis.py        research local bases and candidate reconstruction
~~~

Create only modules needed by the current milestone. Keep public PyCBC array
and differentiability behavior outside this search-specific engine unchanged.
Start with Python orchestration of bulk Torch/cuFFT operations. Move control
code to a compiled layer only if measured host overhead justifies it.

~~~python
# Illustrative contract, not executable code or a finalized signature.
bank_plan = prepare_bank(bank, geometry, waveform_policy)
psd_plan = bind_psd(bank_plan, psd, psd_version)
engine = SearchEngine(bank_plan, selection_policy, resource_budget)
ticket = engine.submit(data_block, psd_plan, valid_interval)
ready_batches = engine.drain()   # only committed, owned candidate records
remaining_batches = engine.flush()
engine.close()
~~~

Define and test these invariants before optimization:

- **Versioned inputs.** Bank identity includes parameters, approximant/model
  version, frequency grid/support, taper/decompression settings and precision.
  PSD bindings include detector, grid, explicit version and the derived norms,
  power-bin edges and any PSD-dependent basis data. Do not key correctness
  solely on a mutable object's address. Old bindings remain alive until their
  outstanding work finishes; reuse waveform data across PSD updates when valid.
- **Owned storage.** Input and result lifetimes extend through the consumer's
  completion event. Never queue a reference to a `FilterBank` output or a
  correlation row that the next tile will overwrite. Either finish vetoes,
  retain rows, or deliberately recompute them from pinned bank/data versions.
- **Commit boundaries.** Completed tile results remain provisional until all
  selection, veto, abort and temporal-boundary dependencies are resolved.
  `drain()` exposes only committed records. In live processing, finalize the
  whole block before publication: a later tile may change global top-K or
  abort the block, in which case all provisional results are discarded.
  `flush()` waits for outstanding work and applies the declared end-of-stream
  validity policy; it must not invent future data to finish boundary windows.
- **Explicit coordinates.** Candidate records carry detector, template ID,
  data/PSD version, segment/block ID, integer sample coordinates and the epoch
  needed for conversion to output time. Keep full-transform indices distinct
  from analysis-window offsets and cumulative output indices. Preserve complex
  SNR, normalization, sigmasq and veto metadata without double normalization.
- **Bounded queues.** Use preallocated structure-of-arrays buffers, a device
  count, validity masks and an overflow flag. No per-template `.item()`, tensor
  truth test, `.cpu()`, NumPy conversion or host-sized `nonzero` result in the
  inner loop. Host decisions at declared tile/block boundaries are allowed.
- **Lossless overflow.** Do not publish a partially accepted tile. Retain its
  input, then resize within budget or rerun smaller tiles with correct halos
  and deduplication. Never silently clip candidates. If live arrivals exceed
  capacity, expose backlog/deadline failure; do not silently lower sensitivity.
- **Explicit support.** Select a supported engine route before work begins.
  Unsupported approximants, vetoes, dtypes or clustering modes use an explicit
  legacy path. Runtime faults must not silently produce incomplete output or
  turn a claimed GPU-resident measurement into an unreported CPU fallback.

Use one owning worker per GPU initially. Keep conditioning, frame I/O and HDF
serialization on the host unless profiles justify moving them. Transfer data
once per block/PSD preparation and reuse it across bank tiles. Persistent does
not mean the entire bank or every SNR series must fit on the GPU.

For complex64 correlation and SNR arrays alone, workspace is `16 * B * N`
bytes, where `B` is tile rows and `N` is the full transform length. For example,
`B=1024, N=2**21` needs 32 GiB before templates or FFT scratch. Add template
storage, data, PSDs, FFT workspace, candidate/veto scratch, retained history,
double buffering and allocator headroom to the planner. Query actual plan
requirements/peak allocation and shrink tiles deterministically on allocation
failure before publishing results. L2 capacity is not a memory-budget model.

## 4. Milestones

### M0 — Freeze behavior and establish attribution

Read [parity](torch_parity.rst), [testing](torch_testing.rst),
[search](torch_search.rst), [reference campaign](torch_reference_campaign.rst)
and the two benchmark protocols above. Record source/dependency versions,
device, configuration, input hashes and all feature flags.

Preserve the unchanged CPU reference
`40e94792b3edf59f39b18b65102b28a4f74433a7` in a separate checkout/interpreter.
Keep candidate normal CPU, Torch CPU and Torch CUDA results distinct. Freeze
scientific settings, geometry and comparator budgets before measuring changes.
Do not alter CPU arithmetic to manufacture agreement with a new oracle.

Create a capability/contract matrix for the actual target configurations:
cluster policies, optional bank/power/sine-Gaussian/continuous vetoes,
injection rejection, PSD variation, abort thresholds and output metadata.
Specify tie, threshold-equality, nonfinite-input and batch-tail behavior.

Record time/traffic for preparation, correlation, IFFT, selection, vetoes,
transfers and event handling, plus full wall time. Distinguish planning/JIT,
cold execution and warm reuse. Preserve evidence of executed routes and
synchronization counts. Profile low, ordinary and glitch-heavy candidate loads.

**Gate:** independently validated baseline artifacts and runnable comparisons.
List missing hardware/data explicitly; CPU-only success does not complete a
CUDA gate. No functional engine changes are needed for this milestone.

### M1 — Bank tiles and eager dense filtering

Implement immutable/versioned plans and an eager engine using existing
correlation and IFFT primitives. Group templates by compatible duration,
sample rate, transform geometry and frequency support; group derived values
by PSD version. Keep scientific geometry frozen for compatible comparisons.

Prepare bounded bank tiles, norms and chi-square bins in bulk. Avoid repeated
file opens and repeated PSD-dependent scans where reuse is valid. Start with
one tile and one stream; establish ownership before adding prefetch.

Compute the data FFT/overwhitening once for its existing reuse scope. Apply
batched template correlations and IFFTs. In offline work, benchmark tiling
across templates, segments, or both; do not assume larger batches are better.
Reuse plans and allocations, including a defined padded/masked final tile.

**Gate:** every complex sample, normalization, frequency support and valid
interval meets the applicable existing contract versus an independent oracle
and the unchanged CPU route; chi-square bin edges retain their exact contract.
Vary batch size, template powers, FFT lengths,
PSD versions and tile order. Peak memory is bounded by the resource plan.

### M2 — Candidate extraction without per-row host decisions

Implement separate selection policies for offline symmetric clustering and
live peak-per-template behavior. Fuse magnitude comparisons, thresholding and
selection where useful; avoid an unnecessary full real magnitude array.
With ordinary cuFFT, this still reads the dense complex IFFT output. Removing
that write requires separately justified FFT-integrated fusion.

Keep candidate counts and threshold masks on device through a tile. Evaluate
fixed-capacity buffers with masks, not a Python loop over device-discovered
survivors. Preserve normalization-dependent threshold rounding and exact
selection/tie rules. Reconcile temporal halos and adjacent blocks before
declaring boundary candidates final; live latency must include any required
look-ahead. Preserve the existing event-consolidation behavior as well as the
local clusterer's behavior.

Implement overflow/retry before performance tuning. Test empty output, every
sample qualifying, exactly full buffers, one excess candidate, long glitches,
ties across tile/time boundaries and repeated retries without duplication.

**Gate:** identical accepted identities/indices/order wherever the current
contract requires them, owned results surviving subsequent submissions, and
no hidden per-row host synchronization on the selected CUDA path.

### M3 — Batched full-statistic vetoes

First gather candidate work across templates into a bulk interface while
retaining the existing compatible arithmetic. Cache PSD-bound bin edges and
norms on device. Group heterogeneous bin counts/geometry without changing
template IDs or degrees of freedom. Preserve raw versus reduced chi-square
conventions and activation thresholds at the adapters.

Then prototype a qualified parallel bin-sum implementation over
`candidate × chi-square bin × frequency tile`. Replace the full-bin serial
phase initialization with independently anchored phases or short reanchored
recurrences and tree reductions. Control phase argument reduction and
accumulation precision explicitly; a fast trigonometric intrinsic is not
automatically accurate enough. Do not call a new reduction bitwise equivalent
to the historical recurrence without evidence.

Maintain two comparisons: historical search arithmetic and an independent
high-precision mathematical reference. If the faster implementation cannot
meet the established search gates, keep it experimental with separately
approved numerical/scientific criteria. Do not silently substitute it into
the compatible engine or weaken the CPU-compatibility tests.

At each tile, choose a documented correlation-lifetime strategy: finish vetoes
while rows are live, retain only needed rows, or recompute selected rows.
Live global trigger selection may make recomputation cheaper than retention.
Benchmark sparse point evaluation against bin-filter/IFFT evaluation for
dense glitch loads; choose the crossover from end-to-end measurements.
Neither strategy may truncate candidate counts.

Wire all configured auxiliary vetoes through an explicit supported route,
including required SNR neighborhoods and data history. A fast power chi-square
path does not justify omitting sine-Gaussian, bank or continuous chi-square.

**Gate:** veto values, activation, degrees of freedom and ranking decisions
pass their established contracts, including unequal bins, large time indices,
long transforms, empty bins and stale-workspace adversarial cases. Record any
remaining host boundary or fallback as part of the route.

### M4 — Integrate both search applications

Add an opt-in adapter using existing scheme selection; retain the legacy
default until qualification. Prefer one explicit engine setting over more
uncoordinated environment flags. Keep HDF/event schemas and CLI science
options unchanged.

For `pycbc_inspiral`, replace the scalar scheduling loop with bank/data tiles.
Preserve per-template segment rejection, progress accounting, analyzed
intervals, PSD-variation values, event consolidation and completed-work
counts. Translate owned candidate tables at a deliberate output boundary.

For `pycbc_live`, preserve one-peak policy, global `max_triggers_in_batch`
selection before vetoes, abort behavior, reweighted-SNR cuts and block IDs.
Do not apply a per-tile top-K that changes the global result. Schedule bounded
tiles against the real arrival cadence, allowing PSD updates and cancellation
without destroying storage still in use. Preserve singles needed by existing
coincidence/background estimation; do not discard them merely because they
are not immediately coincident across detectors.

**Gate:** complete offline executable comparison and frozen live API
qualification both pass with active vetoes. Test actual live arrivals,
rollover, PSD changes, cancellation and delayed output consumption. Optional
unsupported configurations take a verified explicit fallback. Include a
last-tile abort and out-of-order stream completion: neither may leak a
provisional trigger or alter final global selection.

### M5 — Graphs, overlap and waveform preparation

Only after the eager engine is correct, capture stable tile shapes with
preallocated scratch and warmed plans/kernels. Include candidate/veto work
when its bounded representation permits capture. Record actual capture and
replay; setting an environment variable is not proof of either.

Key captures by shape, dtype, device, stream, bound storage and policy.
Versioned input contents may be updated in place only after the prior consumer
finishes; invalidate captures when their bindings or dependencies change.
Test masked tail tiles, overflow, changed thresholds/PSDs, stream switches,
cleanup and capture errors. Never reuse inherited CUDA captures after fork.

Add pinned staging buffers and event-ordered double buffering only where
transfer/preparation can overlap independent compute. Verify lifetimes on
real CUDA streams, including delayed consumers; fake stream tests do not
establish correctness. Keep the number of workspaces and queued blocks bounded.

Sweep safe batch sizes against throughput, peak memory and live p99 latency.
Retain a fixed reproducible override and conservative fallback. Changing batch
size is an execution choice; changing FFT length, sample rate or padding is
a separate scientific-geometry experiment.

Compare three bank-preparation strategies over realistic reuse counts:
compressed-waveform expansion on GPU, cached expanded tiles, and existing
native GPU waveform generation where supported. Include HDF I/O, upload,
parameter preparation, compilation and repeated use. Generation is not
presumed cheaper and does not eliminate FFT memory traffic. Validate raw
complex waveforms and metadata, not only normalized overlap. Adding a new
waveform family needs its own physics validation.

**Gate:** real-stream ownership tests pass, measured performance improves on
the target workload without latency/memory regression, and every optimization
has a reproducible off/on comparison. Leave losing options disabled.

### M6 — Multirate filtering

Introduce a separate research configuration. Partition compatible templates
by duration/bandwidth, then design frequency bands or time slices: long,
low-frequency content runs at a lower sample rate; short, high-frequency
content retains adequate time resolution. This can benefit both applications.

Derive band limits, anti-alias filters, taper/overlap, whitening support,
decimation factors, group delays, FFT lengths and overlap-save/add hops from
the waveform and PSD. Do not copy a fixed three-band geometry into all banks.
Account for complex phase and fractional time shifts when interpolating and
coherently recombining bands on the output time grid. Preserve the relative
phase needed for the complex SNR and existing vetoes.

Build an independent full-rate reference and band-by-band diagnostics before
candidate selection. Test impulses, known time/phase shifts, band-edge signals,
long inspirals, short mergers, longest templates and PSD drift. Include filter
transients and valid edge support in the searched-time accounting.

Predeclare an error budget allocating loss to band approximation, resampling,
rounding and candidate localization. Evaluate original templates at full
resolution around proposed candidates for final SNR/phase/vetoes. Choose
local direct evaluation versus an IFFT from measured cost, and bound the
refinement neighborhood. Conservative candidate thresholds and a validated
coverage rule are required: final refinement cannot repair a missed proposal.

Optimize cost per **unique valid detector second**, including recombination,
refinement and vetoes. Larger hops may improve offline throughput but cannot
violate live latency. Retain full-rate filtering for groups where multirate
error, support or cost fails its gate.

**Gate:** documented bounds/empirical coverage on approximation and candidate
recall, plus representative injections and noise/background studies at a
predeclared sensitivity/false-alarm criterion. A waveform overlap test or
matching a small trigger fixture alone is insufficient. Until the scientific
criterion is approved and passed, the route remains experimental.

### M7 — Reduced bases and optional cheap consistency screening

Run a feasibility study before building a production basis compiler. Use
local, physically similar subbanks, preferably after the multirate split.
Build a PSD-weighted basis and measure rank, reconstruction error, preparation
cost and stability under PSD updates. Support high-rank groups by ordinary
filtering; do not assume aligned-spin compression transfers to precession,
eccentricity or higher modes.

For a group of `M` templates approximated by `R` basis filters,

~~~text
h_m ≈ sum_r A[m,r] u_r
d_overwhite = data / PSD
y_r = raw_ifft(conjugate(u_r) * d_overwhite)
q_m ≈ sum_r conjugate(A[m,r]) y_r
complex_SNR_m = (4 * delta_f / sqrt(sigmasq_m)) * q_m
~~~

Here `raw_ifft` uses the engine's existing unnormalized inverse-transform
convention and frequency support. Apply each original template's PSD-bound
`sigmasq_m` after reconstruction. Do not combine independently normalized
basis SNRs with bare `A` coefficients; their normalization factors differ.

Filter the `R` basis functions, then reconstruct template responses in bounded
time/template tiles with matrix multiplication. Threshold/cluster each tile
and discard dense reconstructed intermediates once no longer needed. This
creates a genuine GEMM workload; tensor-core precision modes require separate
qualification. FFTs do not become tensor-core operations merely by batching.

Include reconstruction cost `O(M * R * L)` for `L` output samples, basis
filtering, data movement, coefficient storage, PSD refresh and exact follow-up
in the break-even model. A small rank alone is not proof of a speedup.
Do not search only basis peaks: combinations can peak elsewhere. Reconstruct
the required template/time coverage or implement and verify conservative
screening bounds. Residual-template norm and data norm can supply a starting
Cauchy-Schwarz error bound, but measure whether it is useful under glitches.

Optionally reuse already-computed band/basis partial responses for a cheap
consistency screen before the full veto. Calibrate it on signal and noise
populations, then retain the standard full veto for accepted candidates.
Derive expected powers and covariance for overlapping band/basis responses;
their number alone does not determine chi-square degrees of freedom.
Reducing a sparse chi-square calculation from 16 bins to 2 does not by itself
reduce frequency summation work eightfold. No rejection fraction is assumed.

**Gate:** demonstrate net cost reduction and candidate coverage at the agreed
scientific criterion, including near-threshold injections and non-Gaussian
background tails. Requalify selection, background and false-alarm calibration
whenever acceptance changes, even if the final ranking formula is unchanged.
Reject the optimization where it loses;
the multirate/full-template engine remains a complete supported route.

### M8 — Qualify and roll out

Run each accepted route against its declared contract. Keep compatible-route
equivalence and research-route sensitivity claims in separate reports.
Extend from the frozen fixtures to representative bank distributions, long
observations, PSD changes and realistic glitch rates. Show scaling as the
number of distinct templates and unique analyzed seconds increases.

Report full executable wall time, warm engine throughput, peak memory,
host/device utilization, transfer volume and preparation amortization with
their own timing boundaries. For live operation report input-arrival-to-result
p50/p95/p99 latency, worst observed latency, deadline misses, backlog and
overflow/retry counts under sustained arrivals. Record GPU sharing and host
core allocation; compare with a realistic allocated multicore CPU baseline
before making capacity or cost claims.

Keep opt-in routing until the compatibility, scientific, resource and latency
gates for that route are satisfied. Publish limitations and fallback reasons.
Multi-GPU bank sharding is a later scheduling extension after single-GPU
correctness and scaling are established; preserve global selection semantics.

**Done:** both application adapters work for the declared supported settings;
outputs/science pass; queues and memory are bounded; live deadlines are met at
the declared load; improvement survives full-boundary repeated measurements;
and source, configurations, raw receipts and executed routes are reproducible.

## 5. Validation commands and missing tests

Use the project's editable environment and current dependency instructions.
Run commands from the repository root. These are existing test targets at the
inspected revision; verify them before use. Retain JUnit output with
`--junitxml=/absolute/new-result.xml`. Run relevant groups on Torch CPU first
where supported, then on real CUDA. Environment selection does not override
every explicit fixture; report executed devices and skips separately.

~~~sh
# CPU reference behavior
PYCBC_TEST_SCHEME=cpu python -m pytest -q -rs \
  test/test_matchedfilter.py test/test_matched_filter_symm.py \
  test/test_threshold.py test/test_fft_cpu_preservation.py \
  test/test_cpu_skymax_chisq.py

# Bank preparation, PSDs and normalization
PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q -rs \
  test/test_decompress.py test/test_torch_decompress_cpu.py \
  test/test_torch_search_power_scan.py \
  test/test_torch_sigmasq_series_precision.py \
  test/test_torch_strain_psd_precision.py

# Dense filtering
PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q -rs \
  test/test_torch_batched_fft.py test/test_torch_fft_writes.py \
  test/test_torch_fft_cuda_workspace.py test/test_torch_cuda_native_batch.py \
  test/test_live_batch_torch_fft_integration.py \
  test/test_torch_batch_overlap_scaling.py

# Candidate semantics
PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q -rs \
  test/test_threshold.py test/test_torch_search_kernels.py \
  test/test_torch_peak_contracts.py test/test_torch_cuda_native_peaks.py \
  test/test_live_batch_torch_peaks.py test/test_torch_event_kernels.py \
  test/test_torch_event_pipeline.py

# Veto arithmetic and dispatch
PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q -rs \
  test/test_chisq_torch.py test/test_torch_chisq_cpu_compat.py \
  test/test_torch_chisq_sparse_dispatch.py \
  test/test_torch_chisq_cpu_optimization.py \
  test/test_torch_chisq_cuda_normalization.py test/test_torch_chisq_precision.py

# Ownership, host visibility and graphs
python -m pytest -q -rs test/test_torch_runtime_transfers.py \
  test/test_torch_cuda_peak_host_read.py test/test_torch_offline_cuda_graph.py \
  test/test_live_batch_veto_reuse.py

# Large-batch boundaries
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -rs test/test_torch_large_batches.py
~~~

Add focused tests for the new contracts, not copies of the implementation:
candidate capacity/overflow, tiling invariance, cross-boundary ties,
version changes with outstanding work, cancellation, delayed consumers,
bounded memory, graph tails and batched veto handoff. Use independent
mathematical references plus original search outputs. Assert route execution
and forbid unintended host conversion on paths claiming residency.

Existing gaps to close when the relevant milestone is implemented:

- `test_live_batch_veto_reuse.py` covers CPU/Torch CPU/MPS, not CUDA. The async
  double-buffer test in `test_live_batch_torch_peaks.py` uses fake streams and
  events. Add real-CUDA queue/workspace/overlap tests.
- Graph execution needs suitable CUDA/Triton availability. Successful skipped
  runs do not qualify it; new batched captures need their own executed cases.
- The inspected `.github/workflows/torch-gpu.yml` names three nonexistent
  precision-test files without the `test_torch_` prefix. Verify/fix selectors
  before relying on that workflow as a gate; use the actual names above.
- For waveform preparation changes, add the relevant existing tests under
  `test/waveform/` for the approximant actually modified. Do not claim coverage
  for all waveform families from one model's tests.

Run repository-required lint/static checks for implementation changes,
including `flake8 pycbc/ test/`, and the configured qlty checks. Establish
whether any failure predates the change rather than hiding it.

## 6. Benchmark execution and evidence

Use [the reference campaign](torch_reference_campaign.rst) for the original
384-template, five-segment, 1904-valid-second offline workload. Restore its
archived inputs and full argv, rather than reconstructing a similar command.
Verify all 1920 template/segment pairs, actual compressed-waveform use,
conditioned strain, full PSDs, geometry, trigger identities and scientific
metadata before timing. Preserve each comparator's existing tolerances.

For live qualification, restore the frozen fixture/oracle linked by
[torch_batch_numerics.rst](torch_batch_numerics.rst): 1024 fixed templates,
three blocks, qualification seeds 7102/7103, timing seed 7102, active 16-bin
power chi-square and sine-Gaussian vetoes. Its complex-SNR absolute 0.001
criterion and exact trigger gates are specific to that protocol, not a
universal replacement for other tolerances.

The checked-in [tools/bench_production_live_batch.py](../tools/bench_production_live_batch.py)
is a separate diagnostic: its bank size varies with batch size and it disables
the real veto workload. It cannot qualify full-veto throughput or substitute
for the frozen fixture. The general parity matrix under
[tools/torch_parity/](../tools/torch_parity/) is useful additional coverage;
follow its manifest/two-interpreter setup in the testing guide.

After correctness, use separate profiled runs for attribution:

~~~text
python tools/profile_torch_filtering.py --output /absolute/new-profile -- \
  /absolute/candidate/bin/pycbc_inspiral <archived scientific arguments>
python tools/summarize_torch_trace.py /absolute/new-profile/trace.json \
  --receipt /absolute/new-profile/receipt.json \
  --output /absolute/new-summary.json
~~~

These are command templates, not runnable commands until paths/argv are
restored. Update profiler coverage accounting for the new batch executor:
the current helper expects one IFFT call per template/segment pair. Trace
ranges do not each synchronize. Keep host durations, CUDA device durations
and sampled CPU cycles distinct; overlapping activities are not additive.

Follow [torch_benchmark_protocol.rst](torch_benchmark_protocol.rst): at least
three fresh unprofiled processes in rotated order, medians/ranges, fixed
scientific inputs and explicit resource allocation. Include startup through
completed HDF output for executable results. Retune scientific geometry only
as a separately labeled experiment. Demonstrate workload convergence before
claiming sustained capacity; otherwise report finite-workload results.

For remote work, inspect the host read-only first and summarize intended
mutations as required by AGENTS.md. For every long job report host, cwd,
exact command, PID/job ID, log path, next check and stop command. Do not
reserve a GPU by merely setting affinity, and do not stop unrelated workloads.

Each milestone's receipt should state: source/input/config hashes, device and
executed route, tests passed/failed/skipped, numerical policy, timing boundary,
resource/sharing conditions, memory/latency observations, artifact locations
and the decision to keep, revise or disable the change.

## 7. Algorithm and runtime references

- [LLOID / early-warning detection](https://arxiv.org/abs/1107.2665): precedent
  for multirate time slicing with local low-rank filtering.
- [SVD applied to compact-binary signals](https://arxiv.org/abs/1005.0012):
  basis compression and reconstruction; measure rank and error for this bank.
- [MBTA in O4](https://arxiv.org/html/2501.04598v2): a multiband search precedent
  with coherent recombination. Its measured gains are not predictions for PyCBC.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html):
  asynchronous execution, streams and synchronization requirements.
- [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) and
  [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/introduction1.html): plan/workspace
  constraints and possible integrated FFT processing. Investigate fusion only
  after a profile identifies its traffic/launch savings.

Start implementation with M0 and the smallest M1 eager tile that can be
compared against both references. Do not begin with new physics approximants,
a monolithic compiled rewrite, or a claimed speedup inferred from GPU specs.

## 8. PR delivery and reuse

Create a new series for the search engine while retaining useful Torch code,
tests and evidence from the existing PRs. Reuse implementations deliberately;
revalidate their interfaces after moving them. The current search remains a
working comparison and fallback while the new engine is qualified.

The GitHub snapshot checked on 2026-09-09 has draft PRs #5–#15 in a linear
11-part stack. Waveform PR #10 is based on search PR #9; tooling PR #15 is
based on inference PR #14. Those branch relationships do not establish the
new engine's technical dependencies. Determine those from imports, dispatch,
fixtures and executable tests when assembling its foundation.

### Existing work to retain, separate or replace

| Existing PRs | Treatment in the new structure |
| --- | --- |
| [#5 runtime/arrays](https://github.com/xangma/pycbc/pull/5), [#6 FFT](https://github.com/xangma/pycbc/pull/6), [#7 PSD](https://github.com/xangma/pycbc/pull/7) | Retain as reusable foundations. The engine consumes prepared templates/data/PSDs, so do not require every optional PSD-generation feature before its first tile can run. Preserve the ordinary CPU and optional-Torch contracts. |
| [#8 filtering, thresholds and vetoes](https://github.com/xangma/pycbc/pull/8) | Reuse primitives, normalizations and regression tests. Separate reusable operations from existing `MatchedFilterControl` scheduling where necessary. The new candidate queue and batched veto executor belong in the engine series. Existing public filtering APIs can continue to use the legacy implementation. |
| [#9 events and search integration](https://github.com/xangma/pycbc/pull/9) | Retain event/ranking/output contracts, strain/input handling and CPU-preservation fixes. Replace search orchestration through new offline/live adapters. Do not carry the entire old controller into the engine merely because it shares a file with useful code. |
| [#10 waveform operations](https://github.com/xangma/pycbc/pull/10), [#11 TaylorF2](https://github.com/xangma/pycbc/pull/11) | Extract or reuse compressed-bank preparation needed by M1. In this stack, `compress.py` and `decompress_torch.py` changes arrive in #10, while waveform utility and sine-Gaussian support already occur in #8. Audit that boundary explicitly. Native waveform families remain a separate feature series and a measured M5 preparation option. |
| [#12 domain/prior](https://github.com/xangma/pycbc/pull/12), [#13 detector](https://github.com/xangma/pycbc/pull/13), [#14 inference](https://github.com/xangma/pycbc/pull/14) | Keep as separate Torch feature work. The search-engine series should not acquire these changes solely through stack ancestry; retain any shared prerequisite actually identified by the dependency audit. |
| [#15 validation and benchmark tooling](https://github.com/xangma/pycbc/pull/15) | Bring the required harness, comparison policy and CI coverage forward to M0. Keep feature tests with their implementation PR. Keep historical performance receipts tied to their original sources, and attach new evidence as each engine route is qualified. |
| [#16 optional FFT](https://github.com/xangma/pycbc/pull/16), [#17 optional CPU](https://github.com/xangma/pycbc/pull/17), [#18 formatting](https://github.com/xangma/pycbc/pull/18), [#19 FFT formatting](https://github.com/xangma/pycbc/pull/19) | Keep optional optimizations and formatting separate from engine behavior. #18 currently precedes #5; #19 precedes #16 after #15. Reusing that history inherits those changes, but a reconstructed minimal foundation need not make unrelated formatting or CPU tuning a prerequisite. |

Do not discard all of #8 or #9: they contain correctness repairs and useful
public functionality alongside the orchestration being redesigned. Likewise,
do not copy only their new `*_torch.py` files; dispatch, shared-module fixes
and tests can be necessary parts of a working change.

In particular, preserve or explicitly requalify #9's input FFT policy in
[`StrainSegments.fourier_segments`](../pycbc/strain/strain.py): compatible
CPU FFT execution, conditional precision promotion, output casting and
scheme-isolated FFT caches. Filtering against differently rounded input is
not the frozen comparison. Keep #8's waveform time-shift and sine-Gaussian
support with the corresponding veto primitives. Audit #9's batch-correlation
and peak-reduction refinements separately from its controller changes.

### Proposed search-engine series

The labels below are planning identifiers, not assigned GitHub PR numbers.
Each PR includes its own focused tests and documentation. M8 qualification
also applies to any later research route; it is not a one-time blanket pass.

| Proposed PR | Scope and gate | Prerequisite |
| --- | --- | --- |
| S0: contracts and harness | M0; extract the required comparisons and instrumentation from #15, fix relevant CI selectors, and pin CPU/legacy-Torch references. | Selected foundation |
| S1: eager tiled engine | M1; versioned bank/PSD plans, storage ownership, bounded workspaces and dense filtering with existing scientific settings. | S0 and needed bank-preparation primitives |
| S2: device candidates | M2; threshold/cluster policies, identities, lossless overflow and provisional/committed queues. | S1 |
| S3: batched vetoes | M3; full-statistic evaluation, binning/normalization and retained correlation lifetimes. Qualify arithmetic changes independently. | S2 |
| S4: offline adapter | Offline part of M4; opt-in `pycbc_inspiral` integration and complete-executable comparison. | S3 |
| S5: live adapter | Live part of M4; opt-in `pycbc_live` integration, block-wide selection/abort, PSD updates and deadline tests. | S3; can develop alongside S4 once shared contracts are stable |
| S6: measured runtime optimizations | M5; graphs, overlap and preparation strategies only where measurements justify them. Split independent waveform and scheduler changes when needed. | Relevant adapter and unchanged scientific gates |
| S7: qualification and rollout | Compatible-route M8; sustained workload evidence, memory/latency limits and documented enablement policy. | S4/S5 for the advertised applications; S6 only if selected |
| R1: multirate search | M6 plus route-specific M8; coherent reconstruction, approximation budget, sensitivity and background. | Qualified compatible engine |
| R2: reduced-basis search | M7 plus route-specific M8; measured rank/reconstruction cost, conservative candidate generation and original-template refinement. | R1 for the proposed multirate design; feasibility studies may start earlier |
| R3: optional early consistency rejection | Separate scientific change using available band/basis information; explicit acceptance, sensitivity and background/FAR qualification. | Required band/basis route; independent acceptance gate |

The compatible series must be useful and releasable if R1–R3 never meet their
science or performance gates. Do not make speculative compression or rejection
a dependency of ordinary GPU search support.

### Safe transition from the current stack

1. **Preserve the working comparison.** Pin the current integrated source
   (`1229a912b73c21bd601101f56d2ee9dd752aeaf8` at this inspection), the unchanged
   CPU source and their evidence. Begin additive development in an isolated
   `codex/` worktree from that pinned source if it avoids an immediate restack.
   This development base is distinct from the eventual minimal review base.
2. **Prove the boundary before restructuring.** Complete M0 and a passing M1
   prototype, then record the actual runtime/FFT/PSD/filter/bank dependencies.
   Reconstruct a minimal foundation from complete changes, including dependent
   fixes and tests. Move the required harness early and keep waveform and
   inference features on their own dependency branches.
3. **Reconcile delivery once the replacement works.** Retain existing PRs whose
   scope still matches a reusable foundation. Split or supersede the overlapping
   search parts of #8/#9 with explicit old-to-new mappings; preserve their
   reviews and evidence. Retargeting a GitHub base alone does not remove
   unwanted code inherited through ancestry. Review every reconstructed diff
   and test every resulting branch, plus the assembled engine.

Keep the existing PRs available during this process and complete the mapping
before superseding earlier work. An old passing receipt does not qualify a
rearranged or edited tree: pin its actual source, check CPU preservation and
execute the relevant new tests and workload comparisons. Verify source IDs
directly rather than assuming a PR body's recorded head is still its live head.
