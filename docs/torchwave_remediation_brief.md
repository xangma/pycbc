# PyCBC Torch remediation with native waveform providers

Design brief revised 2026-09-12 before implementation. The original review-only scope below is retained as historical context. Subsequent implementation, qualification and remaining limits are recorded in [the implementation report](torch_remediation_implementation.md).

You are the lead coding agent. Use subagents to implement and validate this work—not merely to write a plan or repeat the audit.

Repository: https://github.com/xangma/pycbc
Target branch: `torch-pr11-performance-evidence`
Reviewed PyCBC commit: `9ff3a7ec5b5643fe7b0a3b94d082799c05d31775`, plus existing uncommitted TorchWave integration.
TorchWave checkout: `~/repos/torchwave`, reviewed clean at `84ef9b3467c8cc34b6d515b967d48646f7297d8f`.
Pin and record both repositories again at implementation time. Do not assume these reviewed revisions are still current.
Target workstation: Threadripper PRO 3995WX, 256 GB host RAM, RTX 4090.

The objective is the fastest defensible calculation of the required PyCBC outputs at the required accuracy, including preparation and integration costs. PyCBC defines the scientific and behavioral contract; its existing algorithm is not a constraint. Optimize elapsed time and sustained throughput, not utilization percentages or agreement with an earlier runtime estimate. Do not claim global algorithmic optimality from a benchmark.

## 1. Authority, safety, and initial inspection

Read `AGENTS.md`, relevant nested instructions, existing tests, and the actual executable call graph. Implementation authority must be established when this brief is commissioned. The original proposed brief included narrow permission for low-level C/CUDA, optional Triton/cuFFTDx and `pycbc/lib`; the current review-only request does not exercise that permission. Apply the governing repository instructions and confirmed implementation scope. Native waveform/provider repairs should precede speculative new kernels. Preserve portable fallbacks and avoid unrelated rewrites.

Preserve existing user changes. Record the starting branch, SHA, status, and source diff. If the branch has advanced beyond the audited commit, reconcile the findings against the current source; do not reset it to the old commit. Use isolated worktrees or another safe ownership arrangement. Do not push, open a PR, change drivers/system configuration, or launch work on a remote host without authorization.

Treat every finding below as a hypothesis to reproduce. The earlier audit may contain mistakes. Correct a disproven finding rather than changing correct code to satisfy it.

Start by inspecting these paths and their callers/tests:

- `bin/pycbc_inspiral`, `pycbc/filter/gpu_search/`, and the Torch batch/live dispatch paths.
- `pycbc/waveform/decompress_torch.py`, `pycbc/waveform/bank.py`, `pycbc/waveform/taylorf2_torch.py`, `pycbc/waveform/torch_waveform_registry.py`, and waveform/bank preparation callers.
- TorchWave's public API, approximant implementations, model assets, parity harness, validation receipts and benchmark tooling.
- The existing uncommitted TorchWave integration and `test/test_torchwave_*.py`, `tools/bench_torchwave_pipeline.py`, and `tools/verify_torchwave_gpu_search.py`. Preserve these changes and verify ownership before editing.
- `tools/benchmarking/benchmark_gpu_search.py` and existing executable/worker campaign tools.
- `docs/gpu_search_qualification_report.md`, `docs/torch_performance.rst`, `docs/torch_tiled_pathways.rst`, `docs/torch_batch_numerics.rst`, and qualification artifacts.

Create a concise issue ledger: finding, reproducer/source evidence, severity, owner, intended fix, test, and final disposition. Trace which implementation the real command-line entrypoints invoke. Improvements to a standalone prototype must not be reported as executable improvements unless that integration is exercised.

## 2. Subagent organization

Use actual subagent tools when available. Do not simulate independent reviews or fabricate agent results. If delegation is unavailable, say so and execute the same workstreams sequentially.

Assign these responsibilities, with explicit file ownership:

| Agent | Responsibility |
|---|---|
| A — Numerical contract and reference tests | Establish normalization, complex-valued errors, trigger semantics, adversarial fixtures, and reference/fallback tests. |
| B — Device candidates and engine integration | Candidate buffers, residency, engine/adapter interfaces, overflow handling, graph lifetimes, and actual CLI dispatch. |
| C — Veto kernels | Accurate, memory-bounded power-chi-square and other enabled veto evaluation; sparse/dense strategy selection. |
| D — Reduced basis and screening | Original-template normalization, PSD-aware residual bounds, conservative screening/refinement, and safe experimental-mode boundaries. |
| E — Native waveform providers and preparation | Scientific parameter translation, provider eligibility, TorchWave and existing PyCBC generation, decompression, precision, metadata, caches and bounded prefetch. |
| H — Workstation execution | CPU threading, transfer overlap, GPU generation/filter contention, and cold versus persistent execution. |
| F — Benchmarking and evidence | Timing instrumentation, receipt schema, reproducible workloads, historical-claim corrections, and generated reports. |
| G — Independent reviewer | After integration, review numerical arguments, test adequacy, benchmark accounting, and claims without relying on implementer summaries. |

Use these as workstreams, not a requirement for eight concurrent agents. Schedule bounded tasks within available slots and avoid overlapping file ownership. Agent E owns bank/provider code; B owns filtering and CLI adapters; the lead owns shared interfaces and integration. Agree waveform-provider, candidate-buffer, normalization, cache-key, fallback, and receipt contracts before parallel edits. Resolve conflicts centrally. Parallelize coding and CPU tests, but serialize performance measurements on the single GPU; use a shared benchmark lock. Do not benchmark while other agents are consuming the same resources.

Each implementation agent returns code, tests, commands actually executed, results, limitations, and integration notes. Agent G must inspect code and raw evidence, not just the report.

## 3. Scientific and behavioral contract

Document the contract before optimizing. Preserve waveform samples/model options, PSD weighting, original-template norms, frequency cutoffs, transform grids, one-sided conventions, valid analysis regions, time/phase conventions, enabled vetoes and their degrees of freedom, clustering order/windows/ties, threshold inequalities, and output identities/metadata.

Keep these distinct:

**Numerical compatibility:** declared bounds for complex SNR and statistics, plus explicit reporting of decision differences. Compare complex values, not only magnitudes. Use absolute and norm-scaled errors where relative error is ill-defined. Derive tolerances from the existing contract and numerical analysis; do not relax tests to make a faster implementation pass.

**Strict reference decisions:** identical accepted trigger identities, sample positions, and clustering/cut decisions relative to a specifically pinned reference backend/configuration. This does not imply bitwise equality of every floating-point output. A more accurate direct sum is not automatically bitwise equivalent to the reference FFT.

Implement conservative reference fallback for ambiguous decisions in the strict path. Include thresholds, competing peaks, ties, and downstream ranking—not just the initially selected peak. Recompute the necessary original-template segment/neighborhood using the declared reference path. Never discard an uncertain candidate first and attempt to repair membership afterward.

A bound against mathematical exact arithmetic alone does not certify agreement with a floating-point reference; account for reference error or use sufficiently broad reference recomputation. If a safe pre-discard candidate set cannot be established, disable that screening shortcut in strict mode. An empirical guard band must not be labeled a proof.

Treat waveform generation and filtering as separate contributors to the error budget. If reference decisions use reference-provider waveforms, filtering the TorchWave waveform more accurately cannot repair a waveform-model or parameter discrepancy. Reference fallback must retain access to reference generation or stored reference samples as well as the reference filter, including PSD binding, norms and veto bins.

Retain current compatibility expectations by default. Keep uncertified approximations explicitly opt-in. Do not silently normalize boundary mismatches away, excuse them as noise, substitute a different oracle, or advertise universal equivalence from finite tests.

## 4. Native waveform generation and provider qualification

TorchWave is a real batched, device-native waveform source, not merely a CPU preparation optimization. Its reviewed catalog contains 17 model families, while PyCBC already has a native Torch TaylorF2 batch implementation. Availability in either registry does not establish scientific eligibility or performance superiority.

### Provider contract and immediate adapter repairs

Define one explicit provider interface for existing PyCBC/reference generation, PyCBC native Torch generation, TorchWave generation, and compressed-bank expansion. Preserve compressed samples when those define the selected bank; regenerating them is a separate experiment requiring scientific qualification.

The interface must carry:

- Original template IDs/order and a complete effective physical/model manifest: masses; all relevant spin components; tides; eccentricity; inclination/polarizations/modes; phase, time and reference frequency; PN/spin/tidal orders; model-specific flags; and amplitude/distance conventions. Include global bank arguments and per-row overrides with documented precedence.
- Exact transform length, delta-f, sample rate, zero-origin frequency indexing, active cutoffs/endpoints, duration/taper/termination rules and maximum-duration behavior. Handle any model's padded, truncated or compact-frequency output explicitly.
- Requested and effective provider, generation dtype, stored/filter dtype, device, version/asset identities, metadata ownership and fallback reason. Separate generation precision from filtering precision.
- Raw template tensor and required metadata with explicit lifetime and ownership. Metadata must not force every device row through a CPU sample copy.

First reproduce and repair the current adapter issues: availability checks inspect only the first model; model options such as `phase_order` are omitted; generation is forced to FP32 and output to complex64 even when complex128 is requested; `enable_torchwave=None` permits automatic use for eligible uncompressed banks; and live batching groups by grid without full model compatibility. Do not silently substitute defaults. Unsupported models/options/domains must use a documented reference fallback or raise a clear error, preserving template IDs.

Keep the new provider opt-in until its gates pass. Test explicit enable/disable, absent optional dependency, unsupported configurations and requested-versus-effective dispatch. Group heterogeneous templates by all incompatible model/grid/options while preserving original ordering; only broadcast parameters that the selected model actually supports.

### Independent scientific eligibility

Maintain an eligibility matrix by model, parameter domain, options, generation/storage precision, device, independent oracle, downstream gates and evidence revision. Choose a narrow initial scope from evidence; do not blanket-enable the catalog.

TorchWave's current catalog tests have different per-model gates; some were relaxed from complex relative L2 `1e-6` to as much as `0.35`. Some EOB reference cases call the same implementation. Do not inherit these gates as PyCBC acceptance criteria or infer external-model equivalence from an alias such as `SEOBNRv4ROM`. Preserve historical results with their exact gate and oracle. Independent reference libraries may remain test-only dependencies while production generation becomes native; do not add unnecessary new LAL runtime dependencies.

Measure complex sample errors without fitting out arbitrary time/phase differences; add PSD-weighted error, original norms and downstream statistics/decisions. Phase/time-maximized match is a useful additional diagnostic, not the waveform contract. Use separate independent waveform/reference fixtures and hold strain/injections fixed across providers.

Include long BNS signals, mass-ratio/spin extremes within declared support, nondefault PN orders, tides, higher modes/precession where claimed, varying cutoffs, partial tiles and heterogeneous batches. Verify parameter sensitivity, scalar/batch agreement, permutation invariance, precision requests, phase/time mapping, polarization projection and metadata. Apply unsupported-domain fallback instead of silently dropping physics.

### Device execution, caching and integration

Compare CPU native generation plus bounded transfer/prefetch, CUDA native generation, existing PyCBC native generation, and compressed-bank expansion under identical scientific settings. GPU generation competes with filtering and vetoes for device time and memory; measure the combined pipeline. A fast standalone generator does not establish executable acceleration.

Remove avoidable device-to-host waveform sample copies only after identifying the consumers needing norms, chi-square bins and metadata. The current offline adapter generates on the filtering device but copies rows to NumPy; live generation explicitly uses CPU and buffers the whole bank before yielding; the offline persistent session currently excludes the TorchWave branch. Exercise and report each actual dispatch path separately. Reuse existing preparation/session machinery where practical.

Cache raw waveforms separately from PSD-dependent norms, weighting and veto bins. Waveform keys include complete parameter/model/grid content, both provider code and asset versions, generation/output precision and device. PSD-bound keys additionally cover PSD contents and relevant numerical settings. Changed content with reused identifiers must invalidate the appropriate state. Bound queue, pinned-memory and device-cache budgets; preserve exception propagation and lifetime rules.

Do not assume all native models are synchronization-free, graph-capturable or safe under `inference_mode`. Inspect and test model-specific validation/control flow and internal differentiation; some EOB routines use autograd internally. Qualify scalar, eager batch, compiled and captured execution independently. Do not put model compilation into warm steady-state timing without reporting its cold cost.

## 5. Device pipeline and veto remediation

### Candidate residency and integration

Verify the suspected GPU→NumPy→GPU candidate round trip between selection and veto evaluation. Remove it from the optimized hot path: retain template IDs, sample indices, complex SNRs, normalization metadata, validity/count information, and veto results on-device until an intentional public-output boundary. Preserve existing public APIs through adapters.

Audit `.cpu()`, `.numpy()`, `.item()`, `.tolist()`, tensor-to-bool conversions, dynamic-output operations, and implicit synchronization. Source-level absence of `.cpu()` is not sufficient evidence of an asynchronous pipeline. Capture a trace.

Use bounded, reusable candidate buffers with explicit overflow detection. The current engine already grows/retries and reports overflow, and the adapter raises on unresolved overflow; validate and improve this behavior instead of claiming it is absent. Overflow must trigger resizing outside capture, splitting, or reference fallback—not dropped triggers. Exercise zero-candidate, dense-candidate, partial-tile, and exception paths. Preserve deterministic ordering and clustering behavior where required.

### Accurate, bounded-memory vetoes

Inspect `vetoes.py`. The reviewed implementation already chunks candidates, but phases, gathered correlations, weighted products and cumulative sums coexist. The nominal chunk budget accounts for one array and does not establish a total peak-memory bound. Measure the simultaneous scratch footprint and work outside contributing bins; do not characterize it as an unrestricted all-candidate allocation.

Profile before choosing a replacement. Compare segmented/pairwise reductions, blocked selected-time Fourier sums, and transform-based alternatives where candidate density warrants them. A full/bin-transform approach is not forbidden if it is the fastest accurate choice for a dense workload.

Reduce intermediate memory and unnecessary frequency work. Preserve actual bin edges, cutoffs, endpoint conventions, original correlations/norms, and enabled statistics. The existing Torch path uses FP64 phase evaluation before a complex64 cast; complex64 cumulative sums and endpoint subtraction merit targeted cancellation tests, not an unsupported declaration of numerical failure. Handle cancellation, empty/narrow bins, long transforms, PSD changes, and zero candidates correctly. Do not replace double-precision phase evaluation or stable reductions with faster approximations without error analysis and tests. If using phase recurrences, bound/control drift.

Use optional fused kernels only when they measurably beat the portable implementation without violating the contract. Record scratch-memory scaling and overflow/fallback behavior. Do not equate a particular arithmetic complexity with an efficient implementation.

### Graphs

Verify exactly which stages are captured. Distinguish correlation/IFFT capture from whole-search capture in names, metrics, and documentation. Extend capture only where safe and beneficial.

Validate stable buffers, changing inputs, graph cache keys, stream dependencies, PSD/bank invalidation, partial tiles, dynamic candidate counts, and overflow. Ensure replay does not reuse stale pointers or stale data. Invalidate according to captured dependencies: a PSD change need not rebuild a correlation/IFFT graph if weighting occurs outside capture and replay reads correctly updated stable inputs. Candidate handling and vetoes outside capture must remain visible in timing.

## 6. Reduced-basis and early-screening correctness

The reviewed reduced-basis, screening and multirate implementations are explicitly experimental, with no discovered default CLI instantiation. Their defects block promotion; they are not evidence of default executable trigger loss. First make the current prototype's semantics honest and safe. Then implement the strongest usable conservative screen within the validated accuracy contract. Do not make production remediation depend on achieving a speculative reduced-basis speedup.

### Original norms and residuals

Verify whether PSD binding derives purported original-template norms from `coefficients @ basis_data`. If rank is truncated, these are reconstructed templates. Compute/store norms from the original templates and use them for reported original-template results.

Do not treat aggregate unweighted SVD energy tolerance as a per-template SNR error certificate. Use PSD-weighted basis construction or explicitly bound residuals in the correct PSD-weighted metric. A reusable unweighted basis is acceptable when its weighted certificate is valid.

Retain access to original templates for refinement. Validate template geometry, cutoffs, zero/singular templates, and PSD conventions. If `max_rank` prevents meeting a requested tolerance, report that explicitly and reject/fallback as appropriate; never claim the tolerance was met merely because a rank cap was supplied.

Bind certificates and norms to immutable content/version identities covering templates, PSD, grid, cutoffs, dtype, and relevant math settings. Test PSD changes, including changed contents with reused identifiers. Never reuse stale certificates.

### Conservative screening and original-template refinement

Derive the bounds using the repository's exact normalization. A useful starting form is:

    g_j = sum_r c_jr u_r + e_j
    |rho_j[n] - rho_hat_j[n]| <= ||x||_2 ||e_j||_2 + epsilon_numerical

For PyCBC's common active frequency set K and positive PSD S, use original template norm sigma_j^2 = 4 delta-f sum_K |h_j|^2/S, residual norm epsilon_j^2 = 4 delta-f sum_K |h_j - h_hat_j|^2/S, and data norm D^2 = 4 delta-f sum_K |d|^2/S. If the reconstructed response uses the original sigma_j, Cauchy-Schwarz gives |rho_j - rho_hat_j| <= D epsilon_j/sigma_j before numerical/reference errors. Handle zero norms explicitly. If h_j is itself an approximate provider waveform, include its error relative to the required reference template too.

Define every quantity and normalization factor. Account for stored coefficient/basis error, normalization error, transform/reconstruction error, and bound-computation error. A truncation-only bound is incomplete. Observed residual maxima or an arbitrary safety multiplier are not rigorous certificates.

Reject only where the conservative upper bound proves the candidate cannot affect output. Refine potentially relevant points and clustering competitors using original templates, then calculate final statistics from the original-template path. A loose bound that falls back everywhere is correct but not a speedup; report that outcome honestly.

Avoid an unconditional full-bank-by-time reconstruction workspace when tiled reconstruction/reduction can serve screening. Report basis-construction cost, rank, memory, screening cost, bound tightness, refinement fraction, and amortization crossover. If no useful certificate is established, keep the approximate mode clearly experimental and route compatible execution through dense filtering.

### Early consistency screening and multirate prototypes

Verify whether the two-bin screen rejects using a fixed score such as 50 without a proof relative to the actual final selection. Disable unsafe rejection in compatible mode, or derive a conservative bound on the final statistic/ranking. Preserve the actual clustering/cut order: removing a failing peak too early can change which secondary peak survives.

Add this adversarial regression, translated carefully into the implementation's conventions:

    p = 16
    z_b = 10/16 + sqrt(60)/16 for the first 8 bins
    z_b = 10/16 - sqrt(60)/16 for the last 8 bins

These contributions have total SNR 10 and full chi-square 60, hence reduced chi-square 2 for 30 degrees of freedom. Standard NewSNR is approximately 7.78, above a cut of 5, while a two-bin score of 60 exceeds a fixed screening cutoff of 50. Verify the construction through the real functions; do not merely assert the arithmetic in a test disconnected from the implementation.

A two-group score can lower-bound full chi-square under the correct conventions; together with SNR and the actual degrees of freedom it can upper-bound NewSNR. For a group containing fraction a=m/p of bins, the coarse statistic is |Z_low - a*rho|^2/[a*(1-a)], reducing to |2*Z_low-rho|^2 for equal halves. This is a starting mathematical bound, not certification of the current implementation. Odd/ragged bins, numerical error and clustering order still require treatment.

The current multirate implementation gates an entire full-bank run on any coarse detection; it does not perform selective neighborhood refinement. Audit multirate/sub-banding and other work-avoidance prototypes under the same rule: original-output equivalence must be demonstrated or conservatively refined, not assumed from detection-efficiency results.

## 7. Preparation, reuse, and whole-workstation execution

Separate waveform generation/expansion from PSD generation, noise generation, hashing, injection, planning, transfers, and output. A timer containing all of these must not be called waveform generation.

Implement or repair reuse across template tiles and data segments where inputs permit it. Cache keys must include relevant waveform/model parameters, grid/cutoffs, PSD-dependent state, precision, and device. Preserve decompression/interpolation accuracy and ownership/lifetime rules.

Inspect existing asynchronous prefetch and worker-campaign implementations before duplicating them. Use bounded queues and pinned-memory budgets; propagate exceptions; terminate workers cleanly; prevent oversubscription and GPU starvation. Validate dependencies with events rather than assuming overlap. Do not sum overlapping stage durations into wall time.

Measure cold one-shot execution separately from persistent execution with preparation amortized. Explore sensible CPU thread/worker counts and GPU/CPU overlap on the actual host. A one-core baseline is useful but must not stand in for the best use of the workstation. Preserve portable behavior and do not assume all 256 GB of host RAM is available or GPU-resident.

## 8. Benchmarking and reproducible evidence

Repair the measurement harness before making new speed claims. Record separately:

- Process end-to-end wall time, including startup, preparation, final drain, and output under a documented boundary.
- Synchronized host time for `submit`/`drain` or equivalent engine calls.
- CUDA-event device intervals for specified stages, with correct cross-stream dependencies.
- Cold initialization, warm steady state, transfer, screening/refinement, veto, candidate, and output measurements where instrumented.

Do not call host engine time “pure GPU compute.” Avoid invasive synchronization in the production hot path; separate diagnostic instrumentation from ordinary execution and account for its overhead. Identify unattributed residuals honestly.

For each metric record the unit of work: N, sample rate, segment duration, valid duration, template count, segment count, tile size, frequency range, waveform family, PSD, candidate count, enabled cuts/vetoes, dtype, graph mode, and preparation residency/reuse. Average time per pair is amortized throughput, not single-pair latency.

Use two explicitly different comparison classes:

1. Filtering-only: byte-identical stored waveforms, PSD and strain, with identical thresholds/configuration.
2. Provider/pipeline: identical complete physical/model/grid manifests and independent fixed strain/injections, with separate hashes for each generated waveform array and qualified waveform/output differences. Do not create each route's injection from its own template.

In both cases hold the required output contract fixed and report observed differences; do not require generated waveform hashes to match when the generator is the experimental variable. Preserve historical receipts; add versioned new receipts. Include:

- Both PyCBC and TorchWave source SHA, clean/dirty status, reproducible patches plus relevant untracked source, imported module paths, provider/model-asset versions, complete commands and input manifests.
- Requested/effective provider, dtype and device, per-group eligibility, fallback counts/reasons, cache hit/miss counts, and raw generated-waveform hashes.
- Hardware, CPU affinity/thread pools, RAM configuration where discoverable, GPU/runtime/library versions, precision settings, and relevant power/clock/concurrency conditions.
- Seeds, warmups, actual repetition/block counts, per-run timings, summary statistics, memory samples, correctness metrics, and profiler artifacts when available.

Prefer a clean source checkout for qualification. Never describe a dirty old receipt as qualification of a different current SHA. Store small receipts/logs in version control and large evidence in a durable accessible location with checksums. Do not cite missing files, temporary local paths, or inaccessible `file://` URLs as reproducible evidence. Add schema/consistency and artifact-link checks; generate summary tables from receipts.

Measure representative existing synthetic and executable fixtures, keeping them distinct. Expose and validate geometry controls across every benchmark mode first: the current `--size` does not control all suites, and changing production size can leave the recorded sample rate inconsistent. Derive N, delta-f, sample rate and duration from the actual arrays/configuration. Add direct tests at N = 2^17, 2^19, 2^20, and 2^21 with resource-safe batch sizes, including relevant existing configurations. Do not allocate an enormous full Cartesian product blindly. Include sparse, glitch/candidate-heavy, injected-signal, PSD-change, and partial-tile cases.

For primary performance comparisons, aim for at least five independent cold runs and twenty warm samples when practical; record actual counts and dispersion. For a 500-block stability claim, run and record 500 blocks after warmup, with allocated/reserved/peak memory and growth over time. Equal endpoints alone do not prove absence of leaks. Separate benign allocator/plan caching from persistent growth.

Compare a serial reference, a reasonably tuned multi-core CPU baseline, the existing Torch implementation, and the optimized path where available. For generation comparisons include native PyCBC Torch TaylorF2 and TorchWave CPU/CUDA at separately qualified generation precisions; record applicability when a provider lacks a model. Report cold generation-to-device-ready preparation, prepared filtering, combined native generation/filtering, and actual offline/live executable execution as distinct experiments. Include existing compressed-bank fixtures and separate uncompressed provider fixtures. Sweep a justified subset of thread/batch/worker settings without mixing CPU affinities. Run profiler diagnostics separately from clean timing repetitions. Use hardware counters where available to distinguish bandwidth, arithmetic, launch, transfer, and host bottlenecks; measured or modeled ceilings are not proofs of optimality.

If target hardware or real fixtures are unavailable, finish CPU-testable implementation and runnable qualification tooling. Mark GPU/fixture-dependent results explicitly unrun or skipped. Do not invent measurements, silently substitute another GPU, or extrapolate a different N and label it measured.

## 9. Correct the documentation and historical claims

Verify and correct these specific issues while preserving historical context:

1. N-scaled runtime estimates were presented as directly measured N = 2^19 results or evidence of optimality.
2. Standalone engine, prepared live filtering, and complete executable timings were conflated.
3. A mixed preparation interval was labeled CPU waveform generation; uninstrumented residuals were assigned to specific causes.
4. The standalone calculation fraction was generalized into an immutable Amdahl ceiling for other workloads.
5. SNR-magnitude comparison was described as full complex-SNR agreement; candidate checks were described as full time-series checks; distinct fixture trigger counts were combined.
6. Post-cut membership differences were described as exact parity or attributed to a specific instruction/reduction order without evidence. Verify actual CPU capabilities before making ISA-specific explanations.
7. A graph benchmark for a multi-tile submission was described as one production-sized tile; throughput speedup and latency reduction were confused.
8. A short memory test or allocator-threshold flag was described as a longer run proving zero leakage.
9. GPU residency, graph coverage, reduced-basis certification, backend portability, and qualification status were overstated.
10. Missing artifacts or receipts from different/dirty revisions were cited as verification of current code.
11. TorchWave catalog pass rates were treated as independent model equivalence despite changed tolerances or self-reference; old failure reports were treated as current results without rerunning the repaired code.
12. TorchWave microbenchmarks were labeled cold-start/end-to-end while excluding bank construction or template/norm preparation; CPU generation or CPU fallback was described as GPU execution.
13. A verification script printed row-level failure and then unconditional success, and phase/time-maximized matches or peak-magnitude checks were treated as complete waveform/search parity.

The review confirmed narrower scopes that corrections must preserve: veto scratch is already chunked; graph capture covers correlation/IFFT; overflow detection exists; the old graph timing spans eight 64-template tiles; the engine memory receipt has 50 blocks, not 500; and the cited live component receipt does not establish the stream-amortization table. Some historical logs exist locally, so distinguish missing, misattributed and insufficient artifacts. Retain newer documents' accurate distinctions between prepared live, synthetic CLI and real-frame fixtures, and their explicit unattributed residuals/no-speedup results.

Do not replace unsupported positive claims with unsupported negative ones. Label measured facts, derived calculations, engineering hypotheses, historical reports, and untested claims separately.

## 10. Tests and integration gates

Add focused unit, property/adversarial, integration, and numerical-reference tests. First make required verification failures produce a nonzero process exit, remove unconditional success after failed comparisons, inspect all required rows, and report unavailable devices as skipped. Do not silently substitute CPU for a requested CUDA qualification. At minimum cover provider parameter sensitivity and nondefault options, heterogeneous model grouping, precision requests, optional dependency/default activation, metadata/grid/endpoints, reference-provider fallback, scalar/batch/permuted agreement, complex SNR/phase, original norms, PSD/cutoff invalidation, reduced-rank residuals, the screening counterexample, near-threshold decisions on both sides, clustering ties/competitors, graph replay, candidate overflow, dense-candidate memory bounds, changing inputs, and actual command-line dispatch.

Compare small problems with an independently implemented high-precision direct calculation for mathematical conventions, and compare reference decisions with the pinned reference implementation. These are different oracles. Avoid tests that reproduce the same implementation error on both sides.

Capture baseline failures before modifying code. Keep portable NumPy/Torch CPU paths and optional dependency behavior working. Run relevant lint/tests and integration checks. Report skipped GPU tests separately from passed tests. Agent G must review whether tests could pass despite candidate loss, stale plans, changed norms, or incorrect measurement boundaries.

Integrate in reviewable changes: contract/tests and trustworthy instrumentation first; contain default activation and repair provider semantics; integrate qualified native preparation and persistent reuse; then device/veto fixes and safe prototype remediation; finally qualification and generated documentation. Independent provider and filter work may proceed in parallel after interfaces are agreed. Do not activate an experimental fast path by default until its correctness and performance gates pass. Accept no-speedup results rather than weakening accuracy.

## 11. Completion report

For the future implementation task, deliver implemented code and tests, the resolved issue ledger, reproduction commands, machine-readable receipts, and updated evidence-backed documentation. Do not stop at a design document.

Report which findings were confirmed, corrected or disproven; changes by workstream; provider eligibility and exact supported domains/options; generation precision versus filtering precision; reference fallback coverage; both source identities; tests actually run; numerical/decision results; identical-workload before/after timings; memory and transfer changes; cold versus steady-state behavior; preparation/basis amortization; and remaining limitations.

Clearly distinguish implemented, CPU-validated, GPU-validated, fixture-qualified, experimental, and unrun. Explain any remaining bottleneck using evidence. List deferred work with a concrete reason, not an assumed lack of value. State precisely what accuracy/decision guarantee is established and where fallback applies.

Finish the safe, testable work available in the current environment. Do not fabricate subagent output, benchmark results, proof of certification, or a claim of fastest-possible performance.
