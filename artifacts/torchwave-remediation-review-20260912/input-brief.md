# PyCBC Torch remediation: implementation, numerical safety, and performance evidence

You are the lead coding agent. Use subagents to implement and validate this work—not merely to write a plan or repeat the audit.

Repository: https://github.com/xangma/pycbc
Target branch: `torch-pr11-performance-evidence`
Previously audited commit: `9ff3a7ec5b5643fe7b0a3b94d082799c05d31775`
Target workstation: Threadripper PRO 3995WX, 256 GB host RAM, RTX 4090.

The objective is the fastest defensible calculation of the required PyCBC outputs at the required accuracy, including preparation and integration costs. PyCBC defines the scientific and behavioral contract; its existing algorithm is not a constraint. Optimize elapsed time and sustained throughput, not utilization percentages or agreement with an earlier runtime estimate. Do not claim global algorithmic optimality from a benchmark.

## 1. Authority, safety, and initial inspection

Read `AGENTS.md`, relevant nested instructions, existing tests, and the actual executable call graph. For this task, I explicitly authorize narrowly scoped changes to low-level C/CUDA code, optional Triton/cuFFTDx implementations, and `pycbc/lib` where evidence justifies them. This permission is not a requirement to introduce new kernels. Preserve portable fallbacks and avoid unrelated rewrites.

Preserve existing user changes. Record the starting branch, SHA, status, and source diff. If the branch has advanced beyond the audited commit, reconcile the findings against the current source; do not reset it to the old commit. Use isolated worktrees or another safe ownership arrangement. Do not push, open a PR, change drivers/system configuration, or launch work on a remote host without authorization.

Treat every finding below as a hypothesis to reproduce. The earlier audit may contain mistakes. Correct a disproven finding rather than changing correct code to satisfy it.

Start by inspecting these paths and their callers/tests:

- `bin/pycbc_inspiral`, `pycbc/filter/gpu_search/`, and the Torch batch/live dispatch paths.
- `pycbc/waveform/decompress_torch.py` and waveform/bank preparation code.
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
| E — Preparation and workstation execution | Waveform expansion, caches, transfer overlap, bounded prefetch, CPU threading, and cold versus persistent execution. |
| F — Benchmarking and evidence | Timing instrumentation, receipt schema, reproducible workloads, historical-claim corrections, and generated reports. |
| G — Independent reviewer | After integration, review numerical arguments, test adequacy, benchmark accounting, and claims without relying on implementer summaries. |

The lead owns shared interfaces and integration. Agree candidate-buffer, normalization, cache-key, fallback, and receipt contracts before parallel edits. Resolve conflicts centrally. Parallelize coding and CPU tests, but serialize performance measurements on the single GPU; use a shared benchmark lock. Do not benchmark while other agents are consuming the same resources.

Each implementation agent returns code, tests, commands actually executed, results, limitations, and integration notes. Agent G must inspect code and raw evidence, not just the report.

## 3. Scientific and behavioral contract

Document the contract before optimizing. Preserve waveform samples/model options, PSD weighting, original-template norms, frequency cutoffs, transform grids, one-sided conventions, valid analysis regions, time/phase conventions, enabled vetoes and their degrees of freedom, clustering order/windows/ties, threshold inequalities, and output identities/metadata.

Keep these distinct:

**Numerical compatibility:** declared bounds for complex SNR and statistics, plus explicit reporting of decision differences. Compare complex values, not only magnitudes. Use absolute and norm-scaled errors where relative error is ill-defined. Derive tolerances from the existing contract and numerical analysis; do not relax tests to make a faster implementation pass.

**Strict reference decisions:** identical accepted trigger identities, sample positions, and clustering/cut decisions relative to a specifically pinned reference backend/configuration. This does not imply bitwise equality of every floating-point output. A more accurate direct sum is not automatically bitwise equivalent to the reference FFT.

Implement conservative reference fallback for ambiguous decisions in the strict path. Include thresholds, competing peaks, ties, and downstream ranking—not just the initially selected peak. Recompute the necessary original-template segment/neighborhood using the declared reference path. Never discard an uncertain candidate first and attempt to repair membership afterward.

A bound against mathematical exact arithmetic alone does not certify agreement with a floating-point reference; account for reference error or use sufficiently broad reference recomputation. If a safe pre-discard candidate set cannot be established, disable that screening shortcut in strict mode. An empirical guard band must not be labeled a proof.

Retain current compatibility expectations by default. Keep uncertified approximations explicitly opt-in. Do not silently normalize boundary mismatches away, excuse them as noise, substitute a different oracle, or advertise universal equivalence from finite tests.

## 4. Device pipeline and veto remediation

### Candidate residency and integration

Verify the suspected GPU→NumPy→GPU candidate round trip between selection and veto evaluation. Remove it from the optimized hot path: retain template IDs, sample indices, complex SNRs, normalization metadata, validity/count information, and veto results on-device until an intentional public-output boundary. Preserve existing public APIs through adapters.

Audit `.cpu()`, `.numpy()`, `.item()`, `.tolist()`, tensor-to-bool conversions, dynamic-output operations, and implicit synchronization. Source-level absence of `.cpu()` is not sufficient evidence of an asynchronous pipeline. Capture a trace.

Use bounded, reusable candidate buffers with explicit overflow detection. Overflow must trigger resizing outside capture, splitting, or reference fallback—not dropped triggers. Exercise zero-candidate, dense-candidate, partial-tile, and exception paths. Preserve deterministic ordering and clustering behavior where required.

### Accurate, bounded-memory vetoes

Inspect `vetoes.py`. The suspected implementation materializes candidate-by-frequency phases, gathered correlations, weighted products, and cumulative sums, including frequencies that may not contribute.

Profile before choosing a replacement. Compare segmented/pairwise reductions, blocked selected-time Fourier sums, and transform-based alternatives where candidate density warrants them. A full/bin-transform approach is not forbidden if it is the fastest accurate choice for a dense workload.

Reduce intermediate memory and unnecessary frequency work. Preserve actual bin edges, cutoffs, endpoint conventions, original correlations/norms, and enabled statistics. Handle cancellation, empty/narrow bins, long transforms, PSD changes, and zero candidates correctly. Do not replace double-precision phase evaluation or stable reductions with faster approximations without error analysis and tests. If using phase recurrences, bound/control drift.

Use optional fused kernels only when they measurably beat the portable implementation without violating the contract. Record scratch-memory scaling and overflow/fallback behavior. Do not equate a particular arithmetic complexity with an efficient implementation.

### Graphs

Verify exactly which stages are captured. Distinguish correlation/IFFT capture from whole-search capture in names, metrics, and documentation. Extend capture only where safe and beneficial.

Validate stable buffers, changing inputs, graph cache keys, stream dependencies, PSD/bank invalidation, partial tiles, dynamic candidate counts, and overflow. Ensure replay does not reuse stale pointers or stale data. Candidate handling and vetoes outside capture must remain visible in timing.

## 5. Reduced-basis and early-screening correctness

First make the current prototype's semantics honest and safe. Then implement the strongest usable conservative screen within the validated accuracy contract. Do not make production remediation depend on achieving a speculative reduced-basis speedup.

### Original norms and residuals

Verify whether PSD binding derives purported original-template norms from `coefficients @ basis_data`. If rank is truncated, these are reconstructed templates. Compute/store norms from the original templates and use them for reported original-template results.

Do not treat aggregate unweighted SVD energy tolerance as a per-template SNR error certificate. Use PSD-weighted basis construction or explicitly bound residuals in the correct PSD-weighted metric. A reusable unweighted basis is acceptable when its weighted certificate is valid.

Retain access to original templates for refinement. Validate template geometry, cutoffs, zero/singular templates, and PSD conventions. If `max_rank` prevents meeting a requested tolerance, report that explicitly and reject/fallback as appropriate; never claim the tolerance was met merely because a rank cap was supplied.

Bind certificates and norms to immutable content/version identities covering templates, PSD, grid, cutoffs, dtype, and relevant math settings. Test PSD changes, including changed contents with reused identifiers. Never reuse stale certificates.

### Conservative screening and original-template refinement

Derive the bounds using the repository's exact normalization. A useful starting form is:

    g_j = sum_r c_jr u_r + e_j
    |rho_j[n] - rho_hat_j[n]| <= ||x||_2 ||e_j||_2 + epsilon_numerical

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

Audit multirate/sub-banding and other work-avoidance prototypes under the same rule: original-output equivalence must be demonstrated or conservatively refined, not assumed from detection-efficiency results.

## 6. Preparation, reuse, and whole-workstation execution

Separate waveform generation/expansion from PSD generation, noise generation, hashing, injection, planning, transfers, and output. A timer containing all of these must not be called waveform generation.

Implement or repair reuse across template tiles and data segments where inputs permit it. Cache keys must include relevant waveform/model parameters, grid/cutoffs, PSD-dependent state, precision, and device. Preserve decompression/interpolation accuracy and ownership/lifetime rules.

Inspect existing asynchronous prefetch and worker-campaign implementations before duplicating them. Use bounded queues and pinned-memory budgets; propagate exceptions; terminate workers cleanly; prevent oversubscription and GPU starvation. Validate dependencies with events rather than assuming overlap. Do not sum overlapping stage durations into wall time.

Measure cold one-shot execution separately from persistent execution with preparation amortized. Explore sensible CPU thread/worker counts and GPU/CPU overlap on the actual host. A one-core baseline is useful but must not stand in for the best use of the workstation. Preserve portable behavior and do not assume all 256 GB of host RAM is available or GPU-resident.

## 7. Benchmarking and reproducible evidence

Repair the measurement harness before making new speed claims. Record separately:

- Process end-to-end wall time, including startup, preparation, final drain, and output under a documented boundary.
- Synchronized host time for `submit`/`drain` or equivalent engine calls.
- CUDA-event device intervals for specified stages, with correct cross-stream dependencies.
- Cold initialization, warm steady state, transfer, screening/refinement, veto, candidate, and output measurements where instrumented.

Do not call host engine time “pure GPU compute.” Avoid invasive synchronization in the production hot path; separate diagnostic instrumentation from ordinary execution and account for its overhead. Identify unattributed residuals honestly.

For each metric record the unit of work: N, sample rate, segment duration, valid duration, template count, segment count, tile size, frequency range, waveform family, PSD, candidate count, enabled cuts/vetoes, dtype, graph mode, and preparation residency/reuse. Average time per pair is amortized throughput, not single-pair latency.

Use identical inputs, hashes, thresholds, and outputs for before/after comparisons. Preserve historical receipts; add versioned new receipts. Include:

- Source SHA, clean/dirty status, and a reproducible patch plus relevant untracked source when dirty; complete commands and input manifests.
- Hardware, CPU affinity/thread pools, RAM configuration where discoverable, GPU/runtime/library versions, precision settings, and relevant power/clock/concurrency conditions.
- Seeds, warmups, actual repetition/block counts, per-run timings, summary statistics, memory samples, correctness metrics, and profiler artifacts when available.

Prefer a clean source checkout for qualification. Never describe a dirty old receipt as qualification of a different current SHA. Store small receipts/logs in version control and large evidence in a durable accessible location with checksums. Do not cite missing files, temporary local paths, or inaccessible `file://` URLs as reproducible evidence. Add schema/consistency and artifact-link checks; generate summary tables from receipts.

Measure representative existing synthetic and executable fixtures, keeping them distinct. Add direct tests at N = 2^17, 2^19, 2^20, and 2^21 with resource-safe batch sizes, including relevant existing configurations. Do not allocate an enormous full Cartesian product blindly. Include sparse, glitch/candidate-heavy, injected-signal, PSD-change, and partial-tile cases.

For primary performance comparisons, aim for at least five independent cold runs and twenty warm samples when practical; record actual counts and dispersion. For a 500-block stability claim, run and record 500 blocks after warmup, with allocated/reserved/peak memory and growth over time. Equal endpoints alone do not prove absence of leaks. Separate benign allocator/plan caching from persistent growth.

Compare a serial reference, a reasonably tuned multi-core CPU baseline, the existing Torch implementation, and the optimized path where available. Sweep a justified subset of thread/batch/worker settings without mixing CPU affinities. Run profiler diagnostics separately from clean timing repetitions. Use hardware counters where available to distinguish bandwidth, arithmetic, launch, transfer, and host bottlenecks; measured or modeled ceilings are not proofs of optimality.

If target hardware or real fixtures are unavailable, finish CPU-testable implementation and runnable qualification tooling. Mark GPU/fixture-dependent results explicitly unrun or skipped. Do not invent measurements, silently substitute another GPU, or extrapolate a different N and label it measured.

## 8. Correct the documentation and historical claims

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

Do not replace unsupported positive claims with unsupported negative ones. Label measured facts, derived calculations, engineering hypotheses, historical reports, and untested claims separately.

## 9. Tests and integration gates

Add focused unit, property/adversarial, integration, and numerical-reference tests. At minimum cover complex SNR/phase, original norms, PSD/cutoff invalidation, reduced-rank residuals, the screening counterexample, near-threshold decisions on both sides, clustering ties/competitors, graph replay, candidate overflow, dense-candidate memory bounds, changing inputs, and actual command-line dispatch.

Compare small problems with an independently implemented high-precision direct calculation for mathematical conventions, and compare reference decisions with the pinned reference implementation. These are different oracles. Avoid tests that reproduce the same implementation error on both sides.

Capture baseline failures before modifying code. Keep portable NumPy/Torch CPU paths and optional dependency behavior working. Run relevant lint/tests and integration checks. Report skipped GPU tests separately from passed tests. Agent G must review whether tests could pass despite candidate loss, stale plans, changed norms, or incorrect measurement boundaries.

Integrate in reviewable changes: contract/tests and instrumentation first; then device/veto fixes; safe screening/prototype remediation; preparation improvements; finally qualification and generated documentation. Do not activate an experimental fast path by default until its correctness and performance gates pass. Accept no-speedup results rather than weakening accuracy.

## 10. Completion report

Deliver implemented code and tests, the resolved issue ledger, reproduction commands, machine-readable receipts, and updated evidence-backed documentation. Do not stop at a design document.

Report which findings were confirmed or disproven; changes by workstream; tests actually run; numerical/decision results; identical-workload before/after timings; memory and transfer changes; cold versus steady-state behavior; preparation/basis amortization; and remaining limitations.

Clearly distinguish implemented, CPU-validated, GPU-validated, fixture-qualified, experimental, and unrun. Explain any remaining bottleneck using evidence. List deferred work with a concrete reason, not an assumed lack of value. State precisely what accuracy/decision guarantee is established and where fallback applies.

Finish the safe, testable work available in the current environment. Do not fabricate subagent output, benchmark results, proof of certification, or a claim of fastest-possible performance.