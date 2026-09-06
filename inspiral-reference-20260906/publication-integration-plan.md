# Reference-evidence publication integration

Planning only; no staging, validators, Git changes or publication performed.
`OLD` means `/Users/xangma/repos/pycbc/artifacts/torch-performance-fix-20260906`;
`NEW` means `/Users/xangma/repos/pycbc/artifacts/torch-inspiral-reference-20260906`;
`W` means `/private/tmp/pycbc-torch-performance-fix-20260906`.
The inspected archive is `/Users/xangma/repos/pycbc/artifacts/torch-benchmark-20260906/evidence`,
clean at `7f1ea7a7aa05b4fdafe2995a43a754984a2c5817` on
`codex/torch-benchmarks-20260906`. Remote heads were not refreshed.

1. **Finish and freeze scientific evidence for archive commit A.** Retain the
   existing `performance-fix/` supplement as supporting component evidence;
   add sibling `inspiral-reference-20260906/` for the primary executable comparison.
   Preserve earlier reports, numerical claims and measured source identities.
   Earlier length/padding campaigns are supplemental. Primary report admission
   requires all new qualification, matched tuning and timing at the single final
   source `837f38d493420043e45fb1ad210a0ccf68bacbaa`; those checks are pending.
   There is no final publication completion claim here. Require the final reviewed report, its input manifest,
   scientific qualification, all attempted-run receipts and the declared tuning
   decision before freezing A. Read completed validation receipts; do not rerun
   numerical validators merely to stage files.

   Include `NEW/inputs/` (full and pilot banks, metadata, and any scientific test
   inputs), `runs/` with every attempted outcome, `config.json`, `environment.json`,
   `source.json`, all `source-v2.json` through `source-v5.json` records, all five
   `inspiral-source*.bundle` files, `setup-source-v2.py` through `setup-source-v5.py`,
   `reference-source-equivalence.json`, `source-precision-provenance.json`
   plus `source-precision-provenance-v4.json` and `source-precision-provenance-v5.json`,
   the v2/v3/v4/v5 unit runners, `unit-tests.json`/`.log`, `unit-tests-v3.json`/`.log`
   plus `unit-tests-v4.json`/`.log` and `unit-tests-v5.json`/`.log`, and bank preparation/compression scripts
   and logs, all executed campaign plans/statuses/launch records, and the exact
   runner, qualification, validation, profiling and summarization scripts used.
   Include final summaries, reports, figures and manifests. Preserve trigger
   HDFs, qualification PSD `arrays/*.npy`, stdout/stderr, resource records,
   `.pstats`, original `perf.data`, exported native reports, Chrome traces and
   Torch key averages; profiles remain distinct from unprofiled timing samples.
   Retain `runs/qual-cpu-l256/`, `qualification-wrapper-v1.py` and
   `qualification-v1-failure-analysis.json` as the original failed qualification
   and its explanation. Preserve future failed attempts equally.

   Exclude source clones/worktrees, copied native libraries, environments,
   caches, temporary render/build directories and publication working files.
   `qualification-v2-sample.json` is byte-identical to
   `runs/qual-v2-cpu-l256/qualification.json`; omit that incidental duplicate.
   Keep pilot banks because they belong to actual smoke runs. Deduplicate only
   incidental copies with an explicit canonical-path mapping; do not silently
   discard separate run provenance or change receipt paths. The frame file is
   external to NEW: retain its acquisition/reproduction reference and recorded
   SHA-256, with an archived copy only if needed and permitted for reproduction.
   The new 659-byte source bundle requires commit
   `0d00581251e642a5d6b56b2497a9adad93069e6b`. Document the reconstruction chain:
   `performance-fix/source.bundle` supplies `dfd42`,
   `performance-fix/dependent-sources-v2.bundle` supplies `0d005`, and
   `inspiral-reference-20260906/inspiral-source.bundle` supplies `fb4b335`. The added
   `inspiral-reference-20260906/inspiral-source-v2.bundle` requires `fb4b335` and supplies
   `968bcd558117262af0d603710b054174659adb51`. Preserve the original source
   record and evidence; `source-v2.json` and `reference-source-equivalence.json`
   document the two-file revision and the retained normal-reference results.
   `inspiral-source-v3.bundle` adds precision revision
   `6c82155044d58f3344b281869d87745f71ba2285` above `968bcd`;
   `inspiral-source-v4.bundle` adds the test-only reference correction
   `f2c0abe61e787a26f41208f489c62c877bbd5667` above `6c8215`.
   `inspiral-source-v5.bundle` adds the Python launch-site coefficient fix
   `837f38d493420043e45fb1ad210a0ccf68bacbaa` above `f2c0abe`. Its parent delta
   changes only `pycbc/vetoes/chisq_torch.py` and `test/test_chisq_precision.py`.
   Preserve the failed v3 unit receipt, passing v4 unit receipt, failed v4 CUDA
   smoke parity, the empty first parity output and its nonempty v2 retry, and
   all coefficient/capture diagnostics. The final source-v5 provenance binds
   eight reviewed precision paths across `968bcd..837f38d`, retaining eleven
   unchanged native modules. Every new primary run uses `837f38d`; earlier
   science and timings remain supplemental and cannot populate the final report.

2. **Adapt staging without weakening its guards.** The existing helper and plan
   are `OLD/publication/stage-evidence.py` and
   `OLD/publication/ARCHIVE_STAGING_PLAN.md` (the plan is inside `publication/`).
   The helper freezes both sibling supplements in one reviewed plan, requiring
   31 fresh precision5 cases from six plans, 28 strict comparisons, the passing
   v5 unit receipt, six PNGs and four CSVs from `build-reference-report-v4.py`.
   Also retain `precision5-backend-smoke-parity.json` and its separate Torch
   smoke receipts, plus `run-final-reference-campaign-v5.py` and the final
   campaign status/orchestration/stage logs with bound prerequisites and outputs.
   Source/config/run/report hashes, regular-file/symlink checks, absent
   destinations and byte-for-byte historical preservation remain mandatory.
   A previous frozen plan cannot be reused. The primary agent handles any root
   `README.md` and `SHA256SUMS` changes separately; each supplement gets its own
   checksum manifest. Commit A contains both scientific supplements, while
   final documentation/publication receipts follow in commit B.

3. **Write final docs against A, then assemble the latest source.**

   | Existing pending script | Required update before execution |
   | --- | --- |
   | `OLD/write-docs.py` | Its current output leads with component results and hard-codes `performance-fix/`. Add the completed reference report first, retain the supporting sections, and pin both supplement links to A. Update `W/docs/torch_inspiral_reference.rst`, `docs/torch_optimization_results.rst` and `docs/torch_benchmark_details.rst`; connect the new page through the existing results-page toctree. The reference page currently still says no measurements are complete. Add a separate reference figure/manifest collection if figures are produced. |
   | `OLD/build-docs.py` | Updated to require exactly 12 named Torch pages and three figure collections containing 8, 3 and 6 named images. Preserve warnings-as-errors, all 17 built-image hashes and three downloadable-manifest hash checks. Final build still awaits completed documentation and figures. |
   | `OLD/publication/restack.py` | Updated to freeze `837f38d493420043e45fb1ad210a0ccf68bacbaa` with 16 committed paths, including all eight precision source/test files. Validate all seven commits above the original #15 head: `dfd42 → 450ab3 → 0d0058 → fb4b335 → 968bcd → 6c8215 → f2c0abe → 837f38d`. Freeze a fresh supplementary allowlist and plan before execution. |
   | `OLD/publication/qualify-quality.py` | Updated to require passing all 16 full unit modules at `837f38d` using `unit-tests-v5.json`/`.log`, unchanged receipt/input/log hashes and 12 reviewed file hashes (the prior four plus all eight precision paths). Candidate files must match those hashes; #17 may retain only its exact recorded feature patches in `matchedfilter.py` and `chisq_torch.py`, whose individual reversals in memory must reproduce both reviewed source hashes. Preserve the completed accelerator prerequisite, exact-parent Qlty and CI F401 checks. Scientific report admission remains the staging helper’s separate responsibility. |
   | `OLD/publication/write-bodies.py` | Refresh `SUMMARY_15`, `CONTENTS_15` and reviewed body context for the primary reference, core-count reporting, context-deepcopy and spectral/chi-square precision fixes with final source/qualification identities; retain historical component attribution. |

   The new source change makes `ncores = getattr(ctx, "num_threads", None) or 1`
   when accelerator schemes leave the host count unspecified. It prevents final
   performance-output failure; it is not a kernel optimization or evidence of a
   speedup. The subsequent change preserves `TorchScheme` identity during
   `copy.deepcopy` so executable trigger dictionaries copy array data and metadata
   without cloning the Torch module or thread-restoration ownership. It changes
   only `pycbc/scheme.py` and `test/test_scheme_runtime.py`. These three newly
   included paths do not intersect any dependent feature path set. The precision
   revision adds four production paths (`pycbc/filter/matchedfilter.py`,
   `pycbc/psd/__init__.py`, `pycbc/strain/strain.py`, `pycbc/vetoes/chisq.py`) and
   three regressions (`test/test_sigmasq_series_precision.py`,
   `test/test_strain_psd_precision.py`, `test/test_chisq_precision.py`). It retains
   weak spectral power through double-precision intermediates and corrects
   long-vector CPU chi-square phases. The earlier test-only commit uses a wider
   cumulative-power reference and a justified sequential-rounding bound; it
   does not change production code or executable qualification tolerances.
   The final CUDA launch-site change prevents narrowing of the phase coefficient
   to FP32 and extends the existing chi-square precision regression. It changes
   no kernel body. The three regressions are included in both existing CPU/CUDA CI selections
   and the focused command in `docs/torch_testing.rst`. Existing shared-path
   checks for `matchedfilter.py`, the newly overlapping `chisq_torch.py`, and
   `.github/workflows/basic-tests.yml` still matter, and editing
   `docs/torch_optimizations.rst` would add an explicitly reviewable intersection.
   Keep current `W/docs/torch_testing.rst` and both workflow edits in the exact
   supplementary allowlist. The original source worktree remains the input;
   assemble separately and amend the original #15 commit after applying the
   complete `dfd42..837f38d` delta, leaving one commit above #14.

   | PR | Published branch | Recorded old head | Replacement parent |
   | --- | --- | --- | --- |
   | #15 | `torch-pr11-performance-evidence` | `dfd42bf76766cadca0eecf609a1eaeac73534676` | `78d99b5e0f540abd438e01a221de77fc02109b3f` (#14) |
   | #19 | `torch-fft-formatting-base` | `fa38dbca79f4e2e662f079e56b4d9dbb73fd9ef4` | New #15 |
   | #16 | `torch-followup-fft-optimizations` | `37d3c6b4ac1ec74a2dcd42558f44f21b92218ca5` | New #19 |
   | #17 | `torch-followup-cpu-optimizations` | `bd53914be6d2e4324cc867d52b3842b77cc6729a` | New #15 directly |

   #8 `torch-pr4-filtering`, #9 `torch-pr5-search` and #11 `torch-pr7-taylorf2`
   keep their branch heads; only their bodies gain source-qualified downstream
   evidence links. Refresh all seven PR snapshots before using these old heads
   as replay inputs or push leases. Preserve one commit per PR and exact feature
   patches, bytes and modes; identify actual tested sources separately from
   structurally equivalent publication heads.

4. **Append qualification as archive commit B; publish reviewed candidates.**
   After final-source tests, docs checks and quality checks, add their receipts,
   logs, candidate-stack/source-equivalence records and hashes in a new
   `publication-qualification/` archive directory. Preserve A's scientific files.
   Docs continue to pin A; PR bodies cite B for final qualification, avoiding a
   commit-reference cycle. Keep generated final bodies and post-publication
   receipts outside B. Freeze `allowlist.json`, `frozen-plan.json`,
   `candidate-stack.json`, body context and final-body hashes in the local
   publication audit. Adapt the publisher mechanism from
   `artifacts/torch-batch1024-20260906/publication/publish.py`; no updated publisher
   exists under OLD yet, and the older frozen inputs must not be reused.
   Root then performs the authorized archive publication, atomic four-branch
   push with refreshed explicit leases, and seven body/label updates. Retain
   titles, bases, draft state, stack index, `agent-assisted`, AI Gareth attribution
   and the unchecked Code of Conduct note tagging `@xangma`; read back all results.
