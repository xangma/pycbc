# Torch documentation evidence review — 8 September 2026

Reviewed every line of the eight assigned pages against published main `9d4e4f6905d9f109d559284326d62173a854264f`, current source and the sealed evidence. Six pages needed factual or reproduction clarifications; two remain unchanged after inspection. All eight have distinct useful purposes. No page needs removal, consolidation or a date-only rewrite.

Worktree: `/private/tmp/pycbc-torch-doc-audit-20260908`. Evidence root: `/Users/xangma/repos/pycbc/artifacts/torch-benchmark-controls-publication-20260907` (abbreviated `E` below). Only the six assigned documentation files listed below were edited. This audit made no implementation, immutable-evidence, asset, commit, push or remote changes and acquired no new measurements. Concurrent changes outside these pages belong to the parent/reviewer tasks.

## Page-by-page decisions

| Page | Decision and distinct purpose |
| --- | --- |
| `docs/torch_parity.rst` | Retain unchanged: general scientific, metadata, route and residency contracts, independent of a particular campaign. |
| `docs/torch_batch_numerics.rst` | Retain and correct restoration instructions: the particular R4 synthetic workload, adopted acceptance policy and recorded accuracy. |
| `docs/torch_performance.rst` | Retain and link integrated qualification: a compact presentation of the latest completed measurements for two different workloads. |
| `docs/torch_benchmark_protocol.rst` | Retain and clarify runnable CPU controls: prospective acquisition/capacity requirements, rather than a report of completed capacity. |
| `docs/torch_reference_campaign.rst` | Retain and clarify timing/restoration scope: exact executable inputs, scientific options and reproduction. |
| `docs/torch_profile_attribution.rst` | Retain unchanged: historical instrumentation, call ownership, timer accounting and rejected optimization evidence. |
| `docs/torch_optimizations.rst` | Retain and correct flag behavior: current defaults, eligibility and precedence. |
| `docs/torch_followups.rst` | Retain and expand integrated scientific qualification: measured source changes, separate R/G prototypes, current integrated controls and CPU precision investigations. |

### `torch_parity.rst` — reviewed, unchanged

Checked the five native FD/sequence TaylorF2 families against `pycbc/waveform/torch_waveform_registry.py`, component/global switch semantics against `torch_switches.py`, and the batch Triton route and gradient fallback against `taylorf2_torch.py` and `test/waveform/test_taylorf2_batch.py`. Eligibility, actual launch, complex polarizations, cutoffs/padding and reverse/forward-gradient distinctions are supported. The decompression discussion agrees with `pycbc/waveform/decompress_torch.py`: native inline routes and host SciPy routes remain separate, with MPS precision treated separately. The FFT residency distinction agrees with CPU staging in `pycbc/fft/torchfft.py`.

The page is prescriptive, not a claim that every matrix cell passed on current hardware. Its distinctions between numerical equivalence, byte identity and observed routes remain necessary. It explicitly treats unavailable MPS as a skip. Tests and commands belong on `torch_testing.rst`; campaign-specific budgets belong on the two workload-definition pages. No stale result or unsupported current capability required correction here.

### `torch_batch_numerics.rst` — corrected

Verified against `E/current-batch-sweep-20260907-r4/torch-current-batch-sweep-20260907-r4/{batch-worker.py,batch-campaign.py,NUMERICAL-POLICY.md,policy-decision.json}` and `tools/plot_torch_batch_sweep.py`.

The frozen helper confirms source `9578a710...`, synthetic unequal-power complex64 templates, analytic PSD, coherent injections, sample geometry, 1024 templates × three blocks, 16 power-chi-square bins and sine-Gaussian execution checks. The independent normalization uses float64 real/imaginary power and PSD; the oracle promotes stored inputs before multiplication and applies unnormalized complex128 IFFT. Both normalized complex-SNR gates use absolute `0.001`, with no percentile exceptions. Exact trigger/metadata gates and separate veto tolerances remain applicable. The v2 policy was adopted before R4 and does not convert R3 failures into passes.

Corrected both `--campaign` examples and `PYCBC_BATCH_SWEEP_SCHEMA` to identify the restored **inner** helper/run directory. Clarified that rendering requires a new/empty directory and refuses overwrites, while `--verify-only` uses only the standard library. Preserved the explicit limit that omitted `.npy` outputs cannot be scientifically replayed from the JSON publication. Offline verification accepted all recorded worker results/receipts and the checked-in plot manifest. The errors `3.3868e-6` against the oracle and `3.9178e-6` against MKL, 12 smoke + 36 full qualifications, 54 timing workers and two seeds remain correctly scoped to R4.

### `torch_performance.rst` — clarified

Compared its tables with the pinned summaries in `docs/data/torch-20260908/` using `tools/plot_torch_followups.py`, and verified the R4 figure against the restored campaign. Recomputed executable medians are 66.93342889798805, 104.40019265795127 and 21.765735947992653 seconds. Rates correctly use `384 * 1904 = 731136` template-seconds; the printed rates/ranges and 1.560 CPU ratio agree after rounding. The executable is the scalar search, whereas R4 measures prepared public live filtering before the squared-norm change.

Added a direct link to the integrated native/science qualification and an explicit statement that it adds no timing campaign. Retained source `ecd5d082...` for headline timing, source `9578a710...` for warm API timing, and the independent prototype baselines. Shared-host conditions, single-core resource accounting, observed ranges, finite-workload limits and original-upstream scientific differences are already clear. Summary-level repetition here is useful navigation; detailed acquisition and acceptance definitions remain on their own pages.

### `torch_benchmark_protocol.rst` — corrected

Traced operational instructions to `tools/benchmark_cpu_campaign.py`: `run_repeat` creates per-worker output directories and uses each as `cwd`; `command_spec` resolves the executable but does not rewrite scientific input arguments; `worker_main` applies affinity and crosses a shared gate; Linux `/proc`, sysfs and affinity are required. The tool records observations and user-supplied provenance, not a resource reservation.

Added Linux-only scope and absolute executable/input-path guidance. Qualified 64 workers as the recorded `len` topology **when all 64 physical cores are actually available and reserved**. Explained that `--reservation-note` is a user assertion and `--shared-host` permits diagnostic runs despite failed idle checks; neither grants exclusive access. Preserved the distinction among one thread, concurrent processes and full-machine capacity, plus independently verified work counts and aggregate elapsed-time denominator. This future protocol is necessary because none of the historical single-worker campaigns measures reserved full-machine capacity.

### `torch_reference_campaign.rst` — clarified

Checked input/configuration and timer claims against the reference/profile publication and loader package (`E/optimization-evidence-20260908/torch-fft-optimization-20260908`). Source/revision pins, fixed 384-template compressed bank, frame hash, GPS interval, 1904 unique valid seconds, 2M transform geometry, five segments, thresholds and optional-veto caveats agree with the retained protocol and qualification records. The 44 comparisons are eight qualification/controller plus 36 timing comparisons; timing workers themselves include their runtime wrappers but exclude separate controller comparison work. Numerical budgets and bounded provenance substitutions are not conflated.

Changed “latest matched source” to “latest matched timing campaign” so it cannot be read as current published HEAD. Clarified that `loader-v1/` and the original qualification/timing tar files are available **after restoration** of the optimization package; the publication itself uses lossless compressed/member storage. Retained the exact input and scientific specification because neither the benchmark summary nor general protocol can replace it.

### `torch_profile_attribution.rst` — reviewed, unchanged

Checked the page against `E/device-profile-20260907-r3/REPORT.md` and the linked original/second-pass squared-norm evidence. It already explicitly pins historical source `d2647add...`, runtime tree `9e6a688a...`, and its position before workspace/loader changes. The current squared-norm implementation in `pycbc/types/array_torch.py` still uses real-square plus imaginary-square for non-conjugated complex CPU tensors with at least 4096 elements; the second allocation candidate remains unintegrated.

Filtering profile totals (51.065916/96.890385/9.043454 seconds), FFT/copy ownership, 1309 deepcopy calls and 1.188421 cumulative seconds, native sampled-cycle percentages, CUDA event/range totals and median-worker clock decomposition match the retained report. Inclusive Python rows, sampled cycles, device-event sums and elapsed wall clocks are explicitly separated. The page does not subtract historical profiles from the new headline. The evidence and renderer instructions continue to serve historical attribution even though superseded plot assets are not active navigation. No freshness-only edit is warranted.

### `torch_optimizations.rst` — corrected

The material omission was the implemented promoted MKL route. `pycbc/fft/torchfft.py` defines direct complex64 size 32768 and promoted sizes 1048576, 2097152, 4194304; the latter require one Torch thread. The qualified 2097152 plan uses a private in-place complex128 workspace, while public buffers remain complex64. The previous table described only 32768. Corrected the table while retaining eligibility/fallback language; the complete buffer, layout, alignment, alias, gradient and platform predicates remain authoritative in `_can_use_mkl_cpu_ifft`.

Clarified `PYCBC_TORCH_CUDA_GRAPH`: exact `1` requests replay of an already captured eligible offline graph; it does not perform capture, and successful explicit capture also enables replay. This agrees with `pycbc/filter/matchedfilter.py` and stays distinct from live `PYCBC_ENABLE_CUDA_GRAPHS`. Added explicit `enable_async_streams` constructor precedence and the newer global native-port flag's precedence over its legacy alias. Reviewed remaining batch-size, FFT, threshold, native correlation/peak and compilation controls against `matchedfilter.py`, `torchfft.py`, `pycbc/hardware.py`, `pycbc/events/threshold_torch.py`, waveform registry/switches and the batch TaylorF2 source. Experimental defaults and mixed unset-peak behavior remain explicitly qualified.

### `torch_followups.rst` — expanded with sealed scientific qualification

Retained loader, R, G and CPU precision timing as their own frozen experiments. Recomputed R reductions are 2.21318% (standard) and 7.66571% (CUDA); G is 1.54411% against its own eager arm, which already uses R. The exact paired samples, helper hashes, comparison counts, lack of Torch CPU R timing, strict intermediate-byte workspace rejection and zero timing workers for that rejection remain supported by the five-package archive. None establishes cumulative R-to-G speedup or integrated timing.

Added the missing complete integrated qualification from `E/stack-validation-20260908/native-production/`: 43 new contracts plus 240 regressions pass; 52 regression cases skip for unavailable MPS. Three backend outputs preserve 1991 trigger identities and the frozen scientific budgets. Each backend's 18 H1 science datasets match **its own** reference bytes; this does not mean all backends or entire HDF files are byte-identical. The offline verifier replays all five comparisons and validates independently bounded source/executable provenance substitutions while retaining raw metadata failures.

Verified `qualify_production_graph_v2.py` snapshots production outputs before the eager oracle can overwrite shared buffers. All 1920 replays compare both full and sparse outputs; five captures, retention and actual downstream consumer checks pass. Suppressed-replay and pending-allocation negative controls remain explicit. Added archive verification commands with NumPy/h5py requirements and clear separation from rerunning scientific workloads or measuring performance.

Independently compared production source `fad7d8440bfde083f2e94ee62a0017492dfc4013` to published HEAD: only `pycbc/fft/torchfft.py`, `pycbc/frame/frame.py` and `pycbc/vetoes/chisq_torch.py` differ in runtime/executable trees, and their ASTs match after docstring whitespace normalization. Every other runtime/executable file is byte-identical. The new prose records this exact revision boundary. Current MKL descriptor cache capacity/configurations, ownership, cleanup and graph scratch/owned-output distinctions agree with `pycbc/fft/mkl.py` and `pycbc/filter/matchedfilter.py`.

## Validation completed

- `python3 -B -I verify_all.py` in the immutable optimization publication: PASS for all five packages, 251 physical files, 2337 logical files, 17 original archives and 2063 resolved seal entries. This verifies preservation/reconstruction, not new performance.
- `python3 -B -I verify.py` in the immutable stack publication: PASS, 306 files.
- `/private/tmp/pycbc-torch-publication-20260908-venv/bin/python -B -I native-production/verify_evidence.py` there: PASS; 115 manifest files, 283 native passes, 52 skips, three science backends, five HDF comparisons, 1920 production graph replays and two negative controls. These are replayed evidence checks; native tests/science acquisition were not rerun.
- Independent git/AST equivalence check: PASS; saved as `evidence-source-equivalence.json` beside this report.
- `python3 -B tools/plot_torch_followups.py --output docs/images/torch-executable-20260908 --verify-only`: PASS; pinned summaries, recomputed statistics and image/renderer manifest agree.
- R4 plot `--verify-only` against `/Users/xangma/repos/pycbc/artifacts/torch-r4-restore-verification-20260907/torch-current-batch-sweep-20260907-r4` and checked-in `docs/images/torch-batch-r4-20260907`: PASS, including every qualification/timing receipt and plot manifest. No rendering or `.npy` science replay was performed.
- Existing focused plot tests: **115 passed in 6.59 seconds** (32 batch, 83 historical executable). Used the published inner helper directory for `PYCBC_BATCH_SWEEP_SCHEMA`, Python 3.13.9, Matplotlib 3.10.7, and disabled pytest cache creation. JUnit output: `evidence-plot-tests.xml`.
- All **54** assigned-page `ref`, `doc`, download and image targets resolve. File hashes and checks are recorded in `evidence-document-checks.json`.
- `git diff --check`: PASS. Full Sphinx build and visual/asset integration remain with the parent as assigned.

Validation setup corrections: the default Python lacked h5py, so native evidence replay used the existing publication virtualenv. An initial test command named a nonexistent follow-up test file and ran no tests; file discovery identified the existing batch/executable suites, whose corrected command passed above. These were local validation setup issues, not failed scientific gates.

## Remaining questions and limits

No unresolved factual question blocks these edits. Current integrated runtime still has **no new timing campaign**; existing timing must not be relabelled as current-source performance. Unavailable MPS is not qualified by the 52 skips. Historical shared-host results do not establish reserved full-machine capacity, astrophysical bank coverage or original-upstream output equivalence. The R4 publication omits large scientific arrays; its offline verifier checks recorded JSON and integrity. These limits are already stated or reinforced in the reviewed pages.

Delegation/policy tools were searched for but are unavailable in this task's live tool set, so the assigned review was completed directly. The unresolved code-graph project identifier was not guessed and no indexing was performed; source discovery used local `rg` and direct file inspection as authorized.

## Reviewer-comparison steering (supersedes publication-layout recommendations above)

Documentation edits are paused. The eight retention rationales describe useful evidence purposes; they do not require eight main reviewer pages. Parent will condense the presentation and move intermediate prototype history out of main pages.

The archived original-versus-proposed comparison is `reference-campaign-20260907/source.json` and `summary.json`: original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7` versus corrected standard CPU and Torch CPU/CUDA at `a4d77a6d1863c0515e8dace64c5609b63d40b51e`. All use the same bank/frame hashes, scientific CLI configuration, 384 templates, 1904 seconds and selected 512/112/16 geometry. They are different source revisions. Original produces 1988 triggers versus 1991 on corrected/proposed routes, so original-to-proposed parity is not established; shared scientific corrections must be disclosed. This baseline is the archived original, not a verified current upstream PR base.

The latest headline is instead a same-source backend comparison: standard CPU and both Torch routes all at `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`, with the same workload and matching 1991 trigger identities. Its 66.933/104.400/21.766-second medians do not compare untouched upstream against the current merge candidate. `2f799f0046fc36db4215bd8b8b8a774d40c0e011` is the loader experiment's immediate baseline, not the original upstream baseline.

Missing: timing of the current integrated proposed runtime (`9d4e4f6905d9f109d559284326d62173a854264f`, runtime-equivalent to qualified `fad7d8440bfde083f2e94ee62a0017492dfc4013`) against an explicitly selected existing-code PR-base revision under a matched acquisition. No new timing was run. Reviewer methodology should identify the exact baseline/candidate, common inputs/settings, source differences affecting science, complete-process clock, thread/device resources, repetitions/ranges and parity reference. Existing correctness qualification can be cited independently of performance.

## Final restructure handoff

The parent's concrete restructure supersedes the earlier page-retention recommendations. This is a review of the proposed structure, not verification of its implementation. No further active-document edits or broader audit were performed for this handoff.

- Remove `torch_profile_attribution.rst` and `torch_followups.rst` from active documentation; preserve their original contents and immutable archive links in audit evidence. Archive preservation remains part of the parent's implementation.
- Move integrated correctness qualification to `torch_testing.rst`, preserving the source-equivalence boundary, 283 passes/52 unavailable-MPS skips, backend-specific reference comparisons and graph negative controls. These checks establish correctness within the recorded scope, not timing.
- Keep current MKL controls and eligibility in `torch_optimizations.rst`, including direct 32768 and promoted 1M/2M/4M routes, the one-Torch-thread requirement for promoted routes and the private complex128 workspace/public complex64 distinction.
- Make `torch_performance.rst` describe original baseline `40e94792...` versus the current proposed revision, with explicit correctness and timing status. Define four arms: original standard CPU; proposed standard CPU; proposed Torch CPU; proposed Torch CUDA. The three proposed arms must identify the same candidate revision. Remove the old timing table and figures; no historical candidate timing can fill the current candidate's missing timing cells.
- Retain exact input hashes, scientific options, how the original 512/112/16 geometry was selected, and reproduction in `torch_reference_campaign.rst`. Parent reports that the original-reference archive's `REPRODUCE.md` confirms the historical `40e94792...` versus `a4d77a6d...` comparison; this agrees with the source/configuration records inspected here. Do not relabel that candidate as the current proposal.
- Limit `torch_benchmark_protocol.rst` to measurement methodology, including complete-process clocks, matched resources, repetitions, correctness criteria and reservation/affinity limits. Remove R/G/R4 narrative.
- Retain the reproducible independent normalized-complex-SNR method in `torch_batch_numerics.rst`; remove old results/history. Preserve the float64 normalization, promotion before oracle multiplication, unnormalized complex128 IFFT, absolute 0.001 gates without percentile exceptions and the distinction between JSON receipt verification and rerunning omitted scientific arrays.

The remaining comparison caveats are unchanged: historical original/candidate outputs contain 1988/1991 triggers, respectively; `ecd5d082...` is a same-source backend comparison, not an untouched-upstream comparison; current integrated timing is absent; and the relationship of archived baseline `40e94792...` to the actual PR base must be stated accurately. There is no factual objection to the proposed consolidation. Previously recorded validation applies to the audited state; the parent owns validation of the final restructure.
