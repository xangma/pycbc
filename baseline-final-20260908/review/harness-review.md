# Baseline versus final: bounded harness review

Reviewed archived acquisition/qualification code and retained JSON only. No remote work, source edits, benchmark execution or scientific replay. The primary owns campaign setup and execution.

Use original CPU at `40e94792b3edf59f39b18b65102b28a4f74433a7`, then proposed CPU, Torch CPU and Torch CUDA at `123e1fb3ef1b338cada636e71c3e9c7987002402`. Local `git diff 9d4e4f69 123e1fb3 -- pycbc bin` is empty. Record the actual final commit in new receipts, rather than substituting the older runtime revision as the acquired source.

## Reusable files

Paths below use these exact local archive directories:

- **R** = `/Users/xangma/repos/pycbc/artifacts/torch-benchmark-controls-publication-20260907/reference-campaign-20260907`
- **N** = `/Users/xangma/repos/pycbc/artifacts/torch-benchmark-controls-publication-20260907/stack-validation-20260908/native-production`

| File | Reuse | Necessary adaptation in a new campaign directory |
|---|---|---|
| R/`config.json`, `REPRODUCE.md`, `inputs/bank-compressed.hdf` | Exact scientific arguments, bank and 512/112/16 geometry | Relocate frame paths in both `common_args` and `input_files`; retain hashes/options. Use one common absolute bank/frame path across arms where possible. |
| R/`run-case.py` | Fresh output directories, expanded argv, before/after hashes, clean-source checks, full-process `perf_counter` timer, logs and HDF receipt | Add the checked runtime wrapper and sanitized environment below; pass the inherited lock FD. Select each arm's interpreter/source explicitly. |
| R/`compare-backends-controlled.py` | Four-arm list and rotating order `ROUTES[r:] + ROUTES[:r]` | Use only qualification and at least three fresh timing repeats. Its hard-coded paths, old tuning/profile acceptance files, interval/profiling stages and reused original qualification do not belong in this campaign. It records comparison failures but can still finish with `state=complete`: preserve verdicts explicitly. |
| N/`checked-inspiral.py`, `threadpoolctl.py` | Runtime source/native-module checks, environment sanitization, Torch intra/inter-op limits, pool and scheme observations | Parameterize source, interpreter, lock, CPU and sibling checks. CPU 8 / siblings 8,72 and GPU visibility 0 are hard-coded assumptions, not portable defaults. Keep the same wrapper boundary for every arm. |
| R/`qualify-inspiral.py` (identical to N copy) | Untimed compression/no-fallback, scalar work counts, observed FFT dispatch, full PSD arrays, conditioned strain and geometry | Run separately for all four fresh arms. Add campaign assertions for the fixed counts/interval; generic success alone does not pin this workload. |
| R/`compare-triggers.py` (identical to N copy) | Strict HDF/receipt loader and trigger comparator with unchanged budgets | Retain raw results; use a separately recorded, narrowly checked cross-source metadata adapter as described below. |
| N/`science_controller_v3.py`, functions `qualify` and `compare` | Examples of validating receipts, PSD files and recording raw/adjusted comparator verdicts | Reuse their checks selectively. The controller is a three-backend science-only job with hard-coded pins, old-run reuse and graph activation; it is not a timing harness. |

Qualifier SHA256: `234ff35cc4d54ab2e9dfd5d475d825a328346183adaf516f46df4653e50450b7`.
Comparator SHA256: `f0af115a2bf2d3a5a85152cb1f61d9b7570a460efd6d420566c5a2b74ac8f1b5`.

## Exact execution boundary to retain

Keep the fixed **384 templates × 1904 unique seconds = 731136 template-seconds**, 512/112/16 geometry, five segments and 1920 scalar template/segment calls. No retuning, profiling, convergence study or additional performance arms are required for this request.

Use the R runner's outer launch-to-exit timer around the following conceptual child commands, with the same affinity/runtime wrapper in all arms:

```text
qualification:
  taskset -c CORE ARM_PYTHON checked-inspiral.py
    --receipt runtime.json --config CONFIG --source ARM_SOURCE
    --scheme SCHEME --lock-fd FD --
    qualify-inspiral.py --receipt qualification.json --
    ARM_SOURCE/bin/pycbc_inspiral FROZEN_SCIENCE_ARGV

timing:
  taskset -c CORE ARM_PYTHON checked-inspiral.py
    --receipt runtime.json --config CONFIG --source ARM_SOURCE
    --scheme SCHEME --lock-fd FD --
    ARM_SOURCE/bin/pycbc_inspiral FROZEN_SCIENCE_ARGV
```

This is a composition specification, not a directly runnable command. The parent must open/hold the coordinated lock and use `pass_fds=(FD,)`; merely adding `--lock-fd` to R's unmodified runner fails because `Popen` otherwise closes it. Include any `/usr/bin/time` wrapper consistently. The reported duration includes interpreter/imports, runtime observations, science, output and child exit. Exclude pre-launch pinning and post-exit HDF comparisons consistently. The native science controller itself has no suitable monotonic full-process timing sample.

Use schemes `cpu:1`, `cpu:1`, `torch:cpu:1`, `torch:cuda:0`, retaining explicit `--fft-backends mkl` in the common CLI (actual CUDA dispatch must be observed). Apply N's `clean_environment`, not R's inherited `os.environ.copy()` alone. Hash the checker and its vendored threadpool helper in the new manifest. The checker imports Torch and sets both pools only for Torch arms; it does not artificially import Torch for either CPU control. Keep default route flags free of inherited `PYCBC_*` overrides.

Use ordinary `qualify-inspiral.py` for the default CUDA arm. N's `config-graph.json` and `qualify_production_graph_v2.py` deliberately exercise captured production graphs; using them for headline timing would change the executed route. Graph instrumentation, oracle runs and negative probes do not belong in these timing samples.

For each qualification require all recorded checks, 384 distinct selected templates, compression success without regeneration, five segments, 1920 successful scalar IFFTs, 2097152 FFT samples, actual filter slice `[15360,1048576)`, and 1904 unique seconds without gaps/overlap. Validate saved PSD file/data hashes, shape and dtype, plus used-bin positivity/finiteness. Excluded-band positive infinity is explicitly allowed by the qualifier; rejecting every infinity changes its policy. N's exact comparison to an old per-backend `science-reference.json` must not be applied indiscriminately to the untouched baseline.

Compare each timed output with its own fresh qualification/reference and retain within-arm repeat verdicts. Compare original CPU against **all three** proposed arms, including proposed CPU; R's historical cross-source comparison omitted proposed CPU. Compare proposed CPU against both proposed Torch arms separately. Keep science comparisons outside the timing clock. Keep failed baseline equivalence visible alongside timings rather than treating proposed-backend parity as baseline equivalence.

## Allowed provenance substitutions

The comparator intentionally demands equal `source_snapshot` and `consumed_input_sha256`; without an adapter it fails legitimate cross-revision comparisons before considering the additional scientific failures. N/`science_controller_v3.py:compare` demonstrates retaining raw results, deep-copying loaded candidate metadata and recording every before/after substitution. It cannot be reused verbatim: it requires the executable hash to be identical on both sides.

The executable hashes actually differ:

- Original `40e94792...` `bin/pycbc_inspiral`: `5a714f6b5b7c945683837e4287538cb9fc1ed2e3f3d2c93b176df0ebdab9bc3e`.
- Final `123e1fb3...` `bin/pycbc_inspiral`: `d4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a`.

Permit only:

1. **Source identity:** the exact original/final commit pair, independently pinned clean before and after execution, with actual imported source and native builds verified. Store both real identities in immutable receipts; normalize only a comparison copy. Do not call these revisions runtime-equivalent to each other.
2. **Executable provenance:** the exact original/final path-and-hash pair above, verified against each pinned checkout. This is an explicit permitted code difference for a cross-source scientific comparison, not a path-only relocation or evidence of unchanged semantics.
3. **Scientific file relocation, only if needed:** map explicitly named bank/frame paths after proving the same immutable content hashes. The comparator also retains file paths in `analysis_options`, so a path-key change in `consumed_input_sha256` alone is insufficient. Common absolute paths avoid this extra allowance.

Do not remove `metadata.issues`, alter HDF values, erase unmatched triggers, loosen tolerances, normalize geometry/gating, or bypass scientific input hashes. Reject unknown substitutions. Routing/output/verbosity differences are already excluded through the comparator's explicit `ROUTING` set. Bank/frame content must remain unchanged. The stock comparator only checks source-status consistency; the controller must enforce clean, exact revisions and expected native artifacts independently.

## Actual archived baseline failures

R/`original-trigger-comparison.json` compares original `40e94792...` with historical `a4d77a6d...` Torch CPU/CUDA, three repeats each. Every comparison records:

- Original **1988**, candidate **1991**, matched identities **1959**.
- **29 baseline-only unexplained** triggers; **32 candidate-only**, comprising **30 unexplained** and **2 threshold-adjacent requiring review**. The net difference of three understates the membership changes.
- Among matched identities: **1951 chi-square**, **22 phase**, **10 SNR** numerical violations. Maximum chi-square absolute difference is approximately **9.84406**; phase approximately **0.000120878 rad** against a **0.0001 rad** budget; SNR approximately **0.00071669**.
- Additional raw metadata failures for source identity and consumed executable provenance. Allowing those two provenance differences does not eliminate the scientific failures.

The historical original qualification itself succeeded: it completed the workload. Its conditioned-strain digest matches the proposed historical routes, but its PSD digest differs. The HDF trigger comparator does **not** compare full PSD arrays; use the qualifier's saved arrays/records for that separate scientific assessment. Different PSD hashes alone establish byte differences, not their numerical size or a causal explanation for trigger changes.

R/`corrected-trigger-comparison.json` passes all six historical proposed-CPU/Torch comparisons with 1991 matched identities. This neither resolves original-baseline failures nor supplies new final-source results. No new original-versus-final science verdict was calculated in this review.

Strict comparator conditions remain: exact detector/template-hash/GPS-sample identities and valid intervals/gating; no one-sample shift; exact integer/DOF fields; ordinary float budget `1e-5 + 1e-4 * max(abs(a),abs(b))`; sigmasq relative `1e-5` without absolute floor; circular phase absolute `1e-4`. Unexplained membership differences fail; threshold-adjacent omissions or no matched triggers yield review, never pass. Preserve malformed/nonfinite/duplicate/off-grid and HDF-hash failures as failures. Do not assert byte identity between distinct backends or use N's optional exact-18-H1 certificate as a universal requirement.

## Handoff

Reuse the identical archived qualifier/comparator, R's fixed-workload runner/timer and rotation, and N's checked runtime wrapper. Adapt only campaign-local orchestration and explicitly pinned provenance comparisons. No wholesale replay of either archived controller is appropriate. Timing scope remains four arms, fixed 384×1904, full-process duration and at least three rotating fresh repeats.
