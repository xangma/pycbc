# Reference and performance-fix evidence staging plan

Plan only; no archive files, Git refs or commits have been created by this step.
Sources are `artifacts/torch-performance-fix-20260906/` (ART) and
`artifacts/torch-inspiral-reference-20260906/` (REFERENCE). Add two sibling
directories in the existing evidence checkout:

```text
performance-fix/                 supporting component measurements
inspiral-reference-20260906/      primary pycbc_inspiral comparison
```

Keep the existing `performance-fix/` naming contract and all historical records.
Both measured campaigns are complete. Baseline source is
`837f38d493420043e45fb1ad210a0ccf68bacbaa`; optimized source is
`a4d77a6d1863c0515e8dace64c5609b63d40b51e`. The refined `1e-5` compressed bank
passes both campaigns' separate scientific checks. Plans and launch records
alone do not establish completion; the stager binds the actual completed evidence.

## Supporting performance-fix include list

Promote these two files to the supplement root, without the `archive-docs/` prefix:

```text
archive-docs/README.md -> README.md
archive-docs/REPRODUCE.md -> REPRODUCE.md
```

Copy these top-level directories, preserving relative paths and every regular
file except the exclusions below. Reports need their JSON, Markdown, input
manifests and all generated figure formats; profiles need their raw traces and
Python statistics as well as text summaries.

```text
comparison/
profiles-baseline/
wave-profiles-baseline/
postchecks/
cpu32-t4-diagnostic/
accelerator-refinement/
accelerator-refinement-setup-failed/
report/
cpu-followup-report/
cuda-v2-followup-report/
```

Copy these exact top-level files:

```text
comparison-status.json
source-environment.json
source.bundle
performance-fix.patch
accelerator-fastpath.patch
final-performance-fix.diff
dependent-sources.bundle
dependent-sources.json
dependent-sources-v2.bundle
dependent-sources-v2.json
dependent-sources-v2-journal.json
dependent-v2-root-review.json
optional-cpu-baseline.bundle
run-comparison.py
waveform-worker.py
run-postchecks.py
setup-dependent.py
prepare-dependent-v2.py
run-accelerator-refinement.py
run-cpu32-diagnostic.py
profile-live.py
profile-waveform.py
probe-cpu-peaks.py
build-report.py
build-cpu-followup-report.py
build-cuda-v2-followup-report.py
summarize-profiles.py
profile-summary.json
profile-summary.md
test-command.json
tests-candidate.log
test_taylorf2_phase_evaluation.py
test_torch_batch_overlap_scaling.py
lint-f401.json
lint-f401.log
lint-f401-v2.json
lint-f401-v2.log
lint-baseline-equivalence.json
workflow-validation.json
postchecks-launch.json
accelerator-refinement-launch.json
```

The source bundle is about 50.94 MB; total size must be recalculated after final
downloads. Bundles and patches retain v1/v2 identities separately. The patch
and test snapshots are historical inputs; the source manifests identify which
full trees were actually measured and tested.

## Supporting performance-fix exclusions and checks

Exclude `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store`, atomic-write
`*.tmp` files, hidden temporary report directories, test basetemp directories
`postchecks/tests/*-temp/` and `accelerator-refinement/tests/*-temp/`, and any
temporary clone, worktree, virtualenv, build tree or native library copy. Leave
`accelerator-refinement-setup-failed-sources/`, the preserved remote clones,
outside the supplement. Keep the JUnit XML and summary JSON beside each
excluded basetemp directory. Reject
symlinks and special files instead of following them. Do not prune `.pstats`,
Chrome traces, command logs or input manifests as though they were caches.

Do not include the complete ART directory. In particular, leave `publication/`,
`publication-plan.md`, `pr15-current.json`, `candidate.patch`, the two
`before-*.rst` snapshots, `write-docs.py`, `build-docs.py`, `docs-validation/`,
`postchecks-plan.json`, `postchecks-plan-len.json` and
`accelerator-refinement-plan.json` and `accelerator-refinement-plan-len.json`
out of this scientific supplement. The first
patch is superseded/incomplete; the plans are preparatory. The actual runners'
`postchecks/plan.json` and `accelerator-refinement/plan.json` remain included.
Keep publication and documentation validation receipts locally with the
publication audit so final documentation's immutable archive link does not
create a dependency on its own future evidence commit.

Before staging, require completed v1 status, all 360 worker and 108 parity
records, successful postchecks, all 36 optional-CPU workers and 12 parity
records, and successful v2 status with 36 workers, 36 parity records and three
test summaries. Check the already validated reports' frozen input manifests,
completion receipts and source identities without rerunning numerical validators;
counts alone are insufficient. Require all twelve baseline/candidate profile
pairs and their validated summary. Preserve any skips and qualification limits
in the reports; do not infer new-head tests from source-equivalence checks.

Include `accelerator-refinement-setup-failed/` as the preserved first attempt:
its status, launch JSON/log, setup snapshots, plans, command receipts and logs,
`retry-preservation.json`, and original `run-accelerator-refinement.py`.
Require failed state with empty completed-test and completed-timing ledgers,
no timing start or test/timing output, and the recorded optional-CPU harness
comparison error. Pin the old runner to SHA-256
`a478f5b0b5865534c79d8dacf0fa4b27c592971b53d4faf442976c5b976c86d4`.
Verify the retry receipt's failed-status hash, three preserved source moves,
and setup-only failure reason. The active root runner was corrected to SHA-256
`a69ef8762c585c2040e5d6a3e91e16b371c606a2a63353b8ee57fdc83844d795`;
its successful completion remains a separate staging requirement.

Require the completed `cpu32-t4-diagnostic/` receipts and both baseline/v2
profiles, including their six-file SHA-256 manifests. These batch-32,
four-thread instrumented profiles provide diagnostic attribution only;
they do not add throughput measurements.

## Primary reference include list and checks

Preserve every regular file under `REFERENCE/inputs/`, `runs/` and
`final-report/`, and any `raw/`, `tuning/`, `accuracy/`, `parity/`, `profiles/`
or `unit-tests/` evidence directories, plus `snr-psd-diagnostic-v1/`,
`snr-psd-diagnostic-v2/`, `chisq-input-capture-cpu/` and
`chisq-input-capture-cuda/`, `chisq-capture-v4-cpu/`, `chisq-capture-v4b-cpu/`
and `chisq-capture-v4b-torch-cuda/`, subject only to the explicit cache exclusions. Keep all
attempted run directories, including failed or scientifically rejected runs.
Every attempted run must have a terminal `receipt.json`, return code and finish
time; a preserved failure does not count as an accepted timing or qualification.
Include trigger HDFs, exact qualification PSD `arrays/*.npy`, stdout/stderr,
resource logs, `.pstats`, original `perf.data`, text reports, Chrome traces,
Torch key averages, and profiling receipts. Instrumented profiles remain
separate from unprofiled timings.

Include every top-level reference script, plan, status, launch receipt/log,
input summary, validation result, tuning decision and source/environment
record. Allowed top-level evidence extensions are `.py`, `.json`, `.csv`, `.md`,
`.rst`, `.log`, `.txt`, `.xml`, `.bundle`, `.patch`, `.diff`, `.png`, `.svg` and
`.pdf`; an unknown file type or unreviewed directory stops inventory generation.
The frozen plan lists every individual included path, size, mode and SHA-256.
This includes `publication-integration-plan.md`, both compression scripts and
receipts, original and refined full/pilot banks, the unit-test receipt/log/XML
when present, and the exact qualification, validation, profiling, comparison,
summarization and report-builder scripts. Pilot banks belong to actual smoke
runs and are not incidental duplicates.

Preserve these rejected records explicitly:

```text
inputs/bank-compressed.hdf
compression.json
waveform-validation.json
scientific-validation.status.json
compression-refinement-decision.json
qualification-wrapper-v1.py
qualification-v1-failure-analysis.json
runs/qual-cpu-l256/
```

Require the original waveform validation to remain completed with `passed:
false`, its hash to match the refinement decision, and the three waveform
budgets to be unchanged in the passing refined validation. Require the original
wrapper failure, successful underlying executable exit, and original receipt
and wrapper hashes to match the failure analysis. Do not overwrite these
records with successful retries.

The final contract is `final-report/report.json` with `schema_version: 1`,
`status: pass`, and exactly eleven gates with values exactly `true`:
`source_provenance`, `unit_tests`, `campaign_complete`, `source_and_inputs_bound`,
`qualifications`, `waveform_validation`, `boundary_validation`,
`timing_counts_and_geometry`, `tuning_decision`, `trigger_parity` and `profiles`.
Its canonical REFERENCE-relative `input_sha256` mapping must be nonempty;
every entry must be archived and hash-identical. The sole final source phase
is `precision: 837f38d493420043e45fb1ad210a0ccf68bacbaa`.

Require the report's sorted `required_cases` to equal the 31 unique cases from
these six plans:

```text
precision5-reference-qualifications-plan.json
precision5-reference-timings-plan.json
precision5-final-qualifications-plan.json
precision5-reference-profiles-plan.json
precision5-matched-backends-plan.json
precision5-torch-profiles-plan.json
```

The cases comprise three CPU qualifications, nine CPU tuning timings, three
selected-backend qualifications, two CPU profiles, nine matched backend timings
and five Torch profiles. Read the selected FFT length from
`precision5-reference-tuning-decision.json`: it must be 256, 512 or 1024 seconds,
with 112-second start and 16-second end padding. Every final receipt must bind
the clean v5 source before and after execution. Fresh case prefixes are
`qual-precision5-`, `tune-precision5-`, `qual-selected5-`,
`reference-precision5-`, `matched-precision5-` and `profile-precision5-`. Both CPU qualifications at the
selected geometry remain distinct records. Require exactly 28 passing strict
parity comparisons within the three FFT geometries, including the repeated
selected CPU qualification. Preserve the original numerical budgets.

Each plan's adjacent `*-plan.status.json` must be complete with return code
zero, a finish time, an exact completed-command ledger and matching plan hash.
The report manifest must include all twelve plan/ledger files, every final run
receipt, final source/config/environment inputs, v2/v3/v4/v5 source manifests and
bundles, `source-precision-provenance-v5.json`, `setup-source-v5.py`,
`build-reference-report-v4.py`, `unit-tests-v5.json`, `unit-tests-v5.log`,
`run-unit-checks-v5.py`, `waveform-validation-precision5.json`,
`boundary-injections-precision5.json`, `profiles-precision5.json`,
`precision5-reference-tuning-decision.json` and refined compression evidence.
The v5 unit receipt must pass the exact sixteen-module regression command,
bind the eight reviewed precision paths, and retain the matching passing log.
Require 288 waveform template/PSD pairs and 36 boundary cases on the final
source, with unchanged input hashes and original accuracy budgets.

Require six PNG figures (`tuning-capacity`, `matched-capacity`,
`wall-and-internal-times`, `profile-self-time`, `native-symbols`, `cuda-events`)
and four CSV tables (`runs`, `groups`, `profiles`, `native-symbols`); every output
hash must match the report. The report builder validates numerical evidence;
staging reads its completed results without rerunning validators or profiles.
Reject any still-running top-level campaign status or unresolved attempted run.

Preserve `source-v3.json`, `inspiral-source-v3.bundle`,
`source-precision-provenance.json`, `setup-source-v3.py`, `unit-tests-v3.json`,
`unit-tests-v3.log` and `run-unit-checks-v3.py`. The v3 unit receipt records two
failed tests, 363 passed, 68 skipped and eight subtests passed; its completed
execution has return code one and `passed: false`. It remains separate from
the passing v4 retry. Preserve the v4 source manifest, bundle, provenance,
setup script, unit runner and passing unit receipt/log as historical evidence.
The v5 provenance must identify exactly eight paths relative to `968bcd` and
exactly `pycbc/vetoes/chisq_torch.py` and `test/test_chisq_precision.py` relative
to its `f2c0abe` parent. Require the reviewed single host-launch hunk using
`tl.constexpr(two_pi_over_N)`, unchanged other runtime hashes and eleven
unchanged native-module hashes. The kernel body remains unchanged. The final
proof must pass `host_launch_and_regression_only_parent_delta` alongside the
four source/cleanliness/fresh-run checks; retain its exact parent Git diff.

Preserve all numerical-investigation summaries, SNR/PSD diagnostics,
chi-squared capture arrays and diagnostic receipts, prefix-reference diagnostics,
local regression logs, launch records and runner versions. Preserve the v4
CPU smoke baseline and both `qual-precision-torch-{cpu,cuda}-l512`
qualifications with their plan/ledger. The empty
`precision-backend-smoke-parity.json` is a failed command's output and must
remain zero bytes. Its separately named `precision-backend-smoke-parity-v2.json`
must retain the CPU pass and CUDA failure, including original tolerances and
hashes of both trigger files and receipts. None of these smoke checks counts
among the 31 final v5 cases.

Keep the failed `chisq-capture-v4-cpu/` harness attempt and the completed
`chisq-capture-v4b-{cpu,torch-cuda}/` retries, including every raw array and
conditioned-strain capture. The retries observed five filter segments and
captured three chi-squared calls. Require all array and receipt input hashes.
Preserve `chisq-coefficient-v4.json`, its diagnostic script, the failed first
log and successful `chisq-coefficient-v4b.log`. The six diagnostic rows retain
four original coefficient violations and zero violations for the corrected
coefficient/fallback. This causal diagnosis is supplemental evidence from
`f2c0abe`; it does not establish final v5 science or performance results.

Also preserve the passing `precision5-backend-smoke-parity.json`, its two Torch
qualification receipts and plan/ledger. Its CPU baseline and both candidate
trigger/receipt hashes must match clean v5 runs at the original tolerances.
The two Torch smoke runs remain outside the 31-case final campaign; their
CPU baseline is the final campaign's 512-second CPU qualification.

Require `run-final-reference-campaign-v5.py`,
`precision5-final-campaign.status.json`, its orchestration log, the five
stage logs and three perf stderr logs. The final orchestration must complete
four series stages, three perf exports and one summary stage with zero exits,
clean v5 source before/after and unchanged input hashes. Bind the selector,
decision, CPU campaign, unit prerequisites and four final plans. Its output
manifest must retain all four final ledgers, 19 final receipts, six raw
Python/perf profile inputs and trigger files, three perf text exports and the
profile summary. Every recorded stdout/stderr file must be archived. This
controller receipt supplements the report's independent 31-case requirements.

Earlier source phases, science, timing, profile and failed campaign records
remain supplemental evidence. The prior deepcopy equivalence proof applies
only to its original v1/v2 transition. The precision proof explicitly records
`normal_cpu_outputs_changed: true` and `prior_science_reused: false`; the final
campaign cannot reuse old scientific results under a source-equivalence claim.
Historical waveform rejection, wrapper failure, Torch deepcopy failure,
preflight collision and v2 regression checks remain independently verified.

Exclude `source/`, `source-v2/`, `source-v3/`, `source-v4/`, `source-v5/`, copied native modules (`native/`, `native-modules/`, `*.so`,
`*.dylib`, `*.pyd`), virtual environments, `.git`, `publication/`,
`docs-validation/`, `render/`, `render-cache/`, build/dist directories,
`__pycache__`, `*.pyc`, `.pytest_cache`, `.DS_Store`, atomic `*.tmp` files,
`*-temp` directories and hidden temporary report directories. Symlinks and
special files fail the inventory, including at an excluded top-level path.
Omit only this incidental duplicate, requiring byte size and SHA-256 to match
and recording its canonical mapping in the frozen plan:

```text
qualification-v2-sample.json -> runs/qual-v2-cpu-l256/qualification.json
```

The external frame file is identified by its acquisition path/reference and
recorded SHA-256 in the run/environment evidence; do not rewrite receipt paths.
Source clones, compiled libraries and environments are not archival inputs.
The seven source bundles must remain together in this dependency order:

```text
performance-fix/source.bundle
  supplies dfd42bf76766cadca0eecf609a1eaeac73534676, no prerequisites
performance-fix/dependent-sources-v2.bundle
  requires dfd42bf76766cadca0eecf609a1eaeac73534676
  supplies 0d00581251e642a5d6b56b2497a9adad93069e6b and optional-source heads
inspiral-reference-20260906/inspiral-source.bundle
  requires 0d00581251e642a5d6b56b2497a9adad93069e6b
  supplies fb4b335eeeeaeaa907c1143b45e0191e2d977761
inspiral-reference-20260906/inspiral-source-v2.bundle
  requires fb4b335eeeeaeaa907c1143b45e0191e2d977761
  supplies 968bcd558117262af0d603710b054174659adb51
inspiral-reference-20260906/inspiral-source-v3.bundle
  requires 968bcd558117262af0d603710b054174659adb51
  supplies 6c82155044d58f3344b281869d87745f71ba2285
inspiral-reference-20260906/inspiral-source-v4.bundle
  requires 6c82155044d58f3344b281869d87745f71ba2285
  supplies f2c0abe61e787a26f41208f489c62c877bbd5667
inspiral-reference-20260906/inspiral-source-v5.bundle
  requires f2c0abe61e787a26f41208f489c62c877bbd5667
  supplies 837f38d493420043e45fb1ad210a0ccf68bacbaa
```

Staging checks the bounded bundle headers and matches each reference bundle
hash to its source manifest; it does not invoke Git. Reconstruct the source in
this order: the final incremental bundle requires every preceding bundle.

## Freeze and copy

Freeze an explicit per-file inventory with relative path, byte size, mode and
SHA-256 before copying. Verify the copied inventory byte-for-byte and review
all README/report links. Generate supplement checksums excluding their own
checksum file; the root checksum can include that supplement checksum while
excluding itself. Independently compare every historical archive file against
the pre-stage inventory. The helper makes no historical exceptions: root
`README.md` and `SHA256SUMS` are also preserved. Root may subsequently update
those two files as separately reviewed exceptions, without altering historical
supplements.

The local helper is `publication/stage-evidence.py`. From ART, run
`python3 -B publication/stage-evidence.py` to validate and freeze
`publication/evidence-staging-plan.json`. After reviewing that inventory, run
the same command with `--execute` to copy into the new
`artifacts/torch-benchmark-20260906/evidence/performance-fix/` and
`artifacts/torch-benchmark-20260906/evidence/inspiral-reference-20260906/`
directories. `--reference` overrides the reference source directory. The v6 plan
freezes both inventories, duplicate mapping, qualification result and historical
snapshot together; an older plan cannot be reused. Choose a new
`--plan` path instead of overwriting a frozen plan.
Execution requires an identical existing plan and refuses either existing
destination. Any partial copy remains for inspection and blocks automatic
retry. Both original input inventories and all historical archive files are
checked again after copying. It does not stage or commit Git changes, update the historical
root README/checksum file, invoke report builders or use the network.
The primary agent handles any root README/checksum update separately.

Commit A contains these scientific supplements. Final source/unit/doc/quality
and publication qualification receipts can be appended later in a separate
qualification archive commit B. Documentation pins A so its immutable evidence
link does not depend on the later documentation-validation commit.

`ART/build-docs.py` now checks thirteen Torch pages, four figure collections,
20 built images and four downloadable manifests, preserving image/download
hash checks and Sphinx warnings-as-errors. This staging step does not run the
documentation or quality helpers or claim that their final validation is complete.

## Optimized-source additions

The final v6 source is supplied by `inspiral-source-v6.bundle`, which requires
baseline `837f38d493420043e45fb1ad210a0ccf68bacbaa` and supplies
`a4d77a6d1863c0515e8dace64c5609b63d40b51e`. Preserve the earlier candidate bundle,
failed matrix, corrected exploratory matrix and separate failed report attempts
as historical diagnostics. They do not qualify the final source.

Include `optimized-report-v6/`, `optimized-report-v6-alias-failed/`,
`optimized-report-v6-external-metadata-failed/`, every v6 run and top-level
script/receipt/log, and both profile-interpretation files. Require all ten report
gates, 19 final cases plus two auxiliary science qualifications, 18 strict trigger
comparisons, 288 waveform/PSD checks, 36 boundary checks, 576 compressed-bank
parity cases, the 108-case large-IFFT matrix/dispatch decision and the passing
22-file unit receipt. Source, input, command and completion hashes must agree.
Normal CPU tuning reuse is separately bound to the unchanged CPU-path audit;
all optimized scientific results, backend timings and profiles are fresh.

The exact frame and five external dependency records (four LAL module files and
one generated version module) are permitted only with their recorded hashes.
Source/native files remain omitted as recorded build/environment provenance;
portable scientific inputs must exist in the archived logical inventory.

Files above 90 MiB use the reviewed lossless gzip transport helper. Freeze both
logical scientific and actual transport inventories in the v6 plan. Verify every
original byte before copying and retain the standalone restoration instructions.
`SHA256SUMS` covers actual archive files. Use reconstructed logical files for
report regeneration; published links remain on the immutable archive.
