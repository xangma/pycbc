# Review of the PyCBC Torch remediation brief

Reviewed 2026-09-12. This historical review preceded implementation and performance qualification. See [the subsequent implementation report](torch_remediation_implementation.md) for fixes and final target-host evidence.

The original brief has the right objective and strong numerical/evidence safeguards. TorchWave changes preparation enough to warrant a separate provider workstream, with priority given to scientific parameter translation and activation policy. Native generation is available, but the current adapter and validation tools do not establish interchangeable waveform generation or an end-to-end GPU speedup.

The standalone revised brief is [torchwave_remediation_brief.md](/Users/xangma/repos/pycbc/docs/torchwave_remediation_brief.md). It retains the original remediation scope and incorporates the findings below.

## Reviewed state and method

- PyCBC: branch `torch-pr11-performance-evidence`, HEAD `9ff3a7ec5b5643fe7b0a3b94d082799c05d31775`, with pre-existing changes to `bin/pycbc_inspiral`, `bin/pycbc_live`, and `pycbc/waveform/bank.py`, plus untracked integration tests/tools and historical artifacts.
- TorchWave: clean HEAD `84ef9b3467c8cc34b6d515b967d48646f7297d8f`.
- Three actual subagents separately reviewed TorchWave, numerical correctness, and benchmark evidence. The lead inspected material source claims, ran integration tests and reproduced the adapter and numerical defects.
- CPU evidence only. The lead used Torch 2.13.0; the numerical subagent used the separate `pycbc313` environment with Torch 2.9.1. CUDA was unavailable in both. No RTX 4090, GPU timing or remote execution was performed.

Starting changes, source provenance, commands and small JSON probes are retained under [the review artifacts](/Users/xangma/repos/pycbc/artifacts/torchwave-remediation-review-20260912/README.md). Existing user changes were preserved.

## Findings and proposed priorities

“Confirmed” below means a source finding or the stated CPU reproduction, not GPU qualification. Owners are proposed future workstreams, not changes completed in this review.

| Priority / owner | Finding and evidence | Required action / acceptance gate |
|---|---|---|
| High / Provider | **Confirmed:** bank adapters forward masses and aligned spins but omit scientific options, including `phase_order`. In a real `FilterBank` probe, orders 0 and 7 give identical TorchWave arrays while the corresponding reference arrays differ by relative L2 **1.43007**. | Translate all effective global/per-row options; test parameter sensitivity; reject or reference-fallback unsupported configurations. |
| High / Provider | **Confirmed:** `get_batch_tensor(dtype=torch.complex128)` returns complex64; generation/grid/parameters are forced to FP32. `enable_torchwave=None` permits automatic eligibility for an uncompressed bank. Availability checks inspect the first model; live groups only by grid. | Separate generation/output precision; honor requests; explicit opt-in until qualified; group by complete compatible model/options and retain IDs. |
| High / Evidence | **Confirmed:** verification can print failed rows then unconditional `ALL CHECKS PASSED`; it checks only the first eight search rows and peak magnitudes/arrivals, with no final veto comparison. | Nonzero exit for any required failure; all required rows; complex SNR, norms, identities and enabled statistics; explicit device skips. |
| High / Prototype safety | **CPU reproduced:** two-bin score ≈60 rejects a candidate whose full chi-square ≈60, DOF=30 and NewSNR≈7.7827 pass a cut of 5. | Keep rejection out of compatible execution unless bounded against actual final ranking and clustering semantics; add the real-function regression. |
| High / Prototype safety | **CPU reproduced:** rank-one reconstruction reports `sigmasqs=[4,0]` where original templates have `[4,1]`; requested tolerance remains `1e-6` despite the rank cap. | Preserve original norms/templates, report unmet rank tolerance, and bind PSD-weighted residual certificates to content. |
| High / Evidence | **Confirmed:** current qualification claims cite a dirty older revision, magnitude-only candidate checks, a 50-block memory receipt and an unrelated live component receipt. | Scope historical claims exactly; create reconstructable, versioned receipts before new qualification. |
| Medium / Integration | **Confirmed in source:** offline TorchWave generation copies every row to NumPy for metadata; live generates on CPU and buffers the whole bank; offline persistent-session dispatch excludes TorchWave. | Qualified provider interface through real CLI paths, bounded preparation/reuse, and device-resident samples where consumers permit; trace actual transfers. |
| Medium / Device/veto | **Confirmed in source:** GPU candidate selection exports NumPy arrays that veto evaluation converts to tensors again. | Device candidate contract and overflow-safe adapters; CUDA trace and identical-output timing before claiming benefit. |

Adapter evidence: [bank provider checks and generation](/Users/xangma/repos/pycbc/pycbc/waveform/bank.py:973), [live preparation](/Users/xangma/repos/pycbc/pycbc/waveform/bank.py:693), [offline dispatch](/Users/xangma/repos/pycbc/bin/pycbc_inspiral:303).
Verification evidence: [search checks](/Users/xangma/repos/pycbc/tools/verify_torchwave_gpu_search.py:325).
Numerical evidence: [screening](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/screening.py:167), [reconstructed norms](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/reduced_basis.py:285).
Transfer evidence: [candidate export](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/candidates.py:612), [veto conversion](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/vetoes.py:285).

The adapter probe shows ignored configuration, not a measured 1.43 TorchWave-to-reference waveform error. The number compares the two reference PN-order configurations. Likewise, the screening reproduction concerns the experimental `VetoManager` screen, not an observed loss from a default CLI run.

## What TorchWave adds—and what still needs qualification

The [public API](/Users/xangma/repos/torchwave/src/torchwave/waveforms.py:431) supports batched tensors, explicit frequencies, generation precision and devices. Its reviewed catalog has 17 model families, substantially more than the README's two. PyCBC also already offers [native batch generation](/Users/xangma/repos/pycbc/pycbc/waveform/waveform.py:938) with a Torch TaylorF2 implementation. Both belong in the comparison alongside reference generation and compressed-bank expansion.

TorchWave's “catalog parity” is not a PyCBC acceptance gate. The [current harness](/Users/xangma/repos/torchwave/tests/parity_harness.py:531) uses model-dependent complex relative-L2 tolerances; commit `fc54a3c` relaxed several from `1e-6` to values between `0.02` and `0.35`. Some [EOB reference cases](/Users/xangma/repos/torchwave/tests/parity_harness.py:972) call the implementation under test. These facts limit the claims supported by those tests; they do not establish that all current models are inaccurate. Older failure reports also predate repairs.

The revised brief therefore requires a model/domain/options/precision/device eligibility matrix, an independent oracle and downstream gates. Waveform error joins the filtering/reconstruction error budget. Refiltering an inaccurate or differently configured generated waveform cannot recover decisions defined by reference waveform samples; strict fallback must include reference generation or retained reference samples.

Other necessary contracts include full zero-origin FFT layout, endpoint masks, phase/time and polarization conventions, PN/tidal settings, duration metadata and model termination. A convenience frequency grid beginning at `f_lower` must not be mistaken for a padded FFT vector. Native tensor output does not establish asynchronous execution or graph capture: input validation and model branches can synchronize, and some EOB code uses internal autograd. Capture and inference-mode support need model-specific tests.

## Corrections to the original analysis

- **Veto memory is already chunked.** The current chunk budget estimates one array, although phase angles, phases, gathered correlations, products and cumulative sums coexist. Correct the claimed peak-memory bound and measure scratch usage. Do not describe an unlimited all-candidate allocation. FP64 phase evaluation already exists; cancellation in complex64 accumulation still needs meaningful numerical tests. [Implementation](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/vetoes.py:335)
- **Overflow handling already exists.** The engine grows/retries and returns an overflow ticket; the adapter raises if unresolved. Validate these paths and improve any defects. [Engine](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/engine.py:773)
- **Graph capture currently covers correlation/IFFT.** Candidate selection and vetoes are outside it. PSD changes require correct dependency/buffer handling, not automatic graph invalidation when updated weighting lives outside capture. [Graph](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/graphs.py:201)
- **Reduced basis, screening and multirate are experimental.** No default CLI instantiation was found in the reviewed paths. Their problems block promotion; production remediation need not depend on making them fast. Multirate currently triggers an entire full-bank run after any coarse candidate, rather than selecting refinement neighborhoods. [Multirate](/Users/xangma/repos/pycbc/pycbc/filter/gpu_search/multirate.py:298)
- **Preserve newer accurate documentation.** The existing numerical document distinguishes full complex error from decision equality. Preparation reports distinguish synthetic CLI, prepared live and real-frame workloads, and already report no meaningful offline speedup in one campaign. Update unsupported claims selectively. [Numerics](/Users/xangma/repos/pycbc/docs/torch_batch_numerics.rst:61), [preparation](/Users/xangma/repos/pycbc/docs/torch_tiled_pathways.rst:138)

The revised residual-bound specification uses original normalization:
`sigma_j² = 4 df sum |h_j|²/S`, `epsilon_j² = 4 df sum |h_j-h_hat_j|²/S`, and `D² = 4 df sum |d|²/S`, giving `|rho-rho_hat| <= D epsilon_j/sigma_j` over common active bins before numerical/reference errors. This clarifies the original brief's schematic bound; it is not a certificate for the current prototype.

## Evidence and experiment changes

The revised brief distinguishes:

| Experiment | Fixed scientific inputs | Timing boundary |
|---|---|---|
| Filtering comparison | Byte-identical stored waveform, PSD and strain arrays | Prepared filtering, candidate/veto/output stages explicitly included or excluded |
| Provider comparison | Complete identical model/physical/grid manifest; independent fixed strain/injections; separate waveform hashes | Cold and warm generation through device-ready samples, metadata, norms and bins |
| Combined pipeline | Qualified provider differences on the same fixture | Generation/filtering contention, transfer, cache misses/reuse and final drain |
| Offline/live executable | Separately identified real/synthetic and compressed/uncompressed fixtures | Process startup through completed output, or clearly defined persistent block boundary |

Do not generate each provider's injection from its own waveform. That can conceal a shared waveform/analysis error. Phase/time-maximized match supplements complex waveform comparison; it cannot replace it.

Specific historical corrections are actionable:

- Standalone `generation_time_sec` includes waveform generation, PSD, noise, hashing and injection; `total_wall` is a sum of selected intervals, not measured process launch-to-output. [Timer](/Users/xangma/repos/pycbc/tools/benchmarking/benchmark_gpu_search.py:886)
- The TorchWave “cold-start” tool constructs banks outside timing and reuses instances; its “end-to-end” filter experiment prepares templates/norms before timing and uses N=4096. Retain its value as an exploratory microbenchmark. [Tool](/Users/xangma/repos/pycbc/tools/bench_torchwave_pipeline.py:84)
- The old graph receipt uses 512 templates in 64-row tiles: eight tiles per submit/drain. Its 3.868→3.078 ms change is approximately 25.66% greater throughput and 20.42% lower latency.
- The old `5.71e-6` metric measures candidate SNR magnitude differences, not full complex SNR. “1920” denotes template/segment coverage, while that receipt records 300 GPU candidates. [Comparator](/Users/xangma/repos/pycbc/tools/benchmarking/benchmark_gpu_search.py:834)
- A 50-block receipt with equal memory endpoints and a >1 MiB growth flag is not a 500-block leak proof. The cited `live_batch_latest.json` is a component microbenchmark, not evidence for the report's stream-amortization table. Some historical logs do exist; classify misattribution separately from missing files.
- The older dirty receipt identifies `edff1e35...`. Its recorded hashes differ from current engine/adapter source, so it cannot qualify this checkout. Both repositories, dirty patches/untracked integration sources, imported modules and actual provider/dtype/device belong in new receipts.
- Fix geometry controls before an N sweep: existing `--size` does not control all modes, and changing production size can leave the recorded sample rate inconsistent.

## Validation actually performed

| Check | Result | Scope |
|---|---|---|
| `test_torchwave_integration.py` and `test_torchwave_live_and_inspiral.py`, excluding `throughput` | **9 passed, 1 deselected**, 6.13 s | Lead; CPU; existing tests |
| Screening, reduced-basis and veto test modules, excluding `large_sample` | **24 passed, 1 deselected**, 2 warnings, 2.70 s | Numerical subagent; CPU; existing tests |
| Saved bank-contract probe | Ignored PN order, ignored complex128 request, automatic eligibility reproduced | Lead; current dirty adapter |
| Saved screening/reduced-norm probe | Counterexample and norm defect reproduced on NumPy and Torch CPU | Lead independently reproduced subagent findings |
| CUDA/RTX 4090 benchmarks, full CLI real-frame campaigns | **Unrun** | No speedup or production qualification claimed |

Passing tests do not contradict the defects: existing screening tests do not compare accepted sets against final ranking; reduced-basis tests check a peak magnitude/location rather than original norms/full complex output; large-veto finiteness alone is insufficient. The new brief prioritizes tests that would catch the reproduced failures.

Recommended implementation order: establish contract and trustworthy gates; contain automatic provider activation and repair parameter/precision handling; integrate qualified native generation and persistent reuse; improve device/veto paths and experimental safety; then run target-hardware qualification and regenerate evidence-backed documentation. No implementation was undertaken in this review.
