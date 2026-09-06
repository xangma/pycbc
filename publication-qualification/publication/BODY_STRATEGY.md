# Local assembly and seven PR bodies

The restack script is scaffolding only. No candidate refs, worktrees, commits,
body files or remote changes have been created. Final evidence and qualification
must be reviewed separately; the script verifies input identity and patch
preservation, not scientific claims.

## Freeze and assemble

After the final files exist atop clean committed fix `837f38d493420043e45fb1ad210a0ccf68bacbaa`,
write `allowlist.json` with exactly this schema and every changed/untracked path:

```json
{"schema": "torch-performance-fix-inputs-v1", "paths": ["<every actual supplementary path>"]}
```

Only `docs/` paths and the explicitly authorized
`.github/workflows/basic-tests.yml` and `.github/workflows/torch-gpu.yml` are
allowed. No wildcards, inferred paths, deletions, staged changes or unrelated
files are accepted. The 16 committed source/test paths remain fixed separately, including the
executable/context fixes and eight precision source/test files. The input
ancestry must be exactly `dfd42 → 450ab3 → 0d0058 → fb4b335 → 968bcd → 6c8215 → f2c0abe → 837f38d`.
The `f2c0abe` commit corrects the cumulative-power test reference and its
accumulated-rounding bound. The final commit preserves the CUDA phase
coefficient at the Python launch site in `pycbc/vetoes/chisq_torch.py` and
updates `test/test_chisq_precision.py`; the kernel body remains unchanged.

From this directory, run the read-only Git inspection and review its frozen plan:

```sh
python3 -B restack.py --allowlist allowlist.json
```

The default writes `frozen-plan.json` only. It pins script, snapshot, allowlist,
file contents/modes, input HEAD/index/status, old feature patches and assembly
location. A changed existing plan is refused. Once the primary agent has reviewed
it, the same command with `--execute` performs local assembly. Execution writes
an exclusive journal, backup refs and a verified bundle; a separate worktree
starts at `dfd42`, applies the fix and frozen supplementary files, and amends #15
with parent `78d99b5e0f`. It then replays #19, #16 and independent #17 into exactly
the four requested candidate branches. The original worktree/index/branch remain
unchanged. Any partial failure is journaled; inspect it instead of rerunning.

Dependent file sets, nonshared bytes/modes and feature patches are preserved.
Only `matchedfilter.py`, `chisq_torch.py` and `basic-tests.yml` may normalize blob IDs and hunk
start coordinates; hunk counts, context and additions/deletions must match.
The same main update is also verified across all four heads. A new overlap,
including `docs/torch_optimizations.rst`, fails until explicitly reviewed.

## Body generation after final evidence

Read `prs-current.json` as the immutable starting snapshot. Produce seven complete
UTF-8 body files under a new `bodies/` directory, plus a hash manifest and readable
diffs; do not edit the snapshot or publish while generating them. Follow the
sections in `.github/PULL_REQUEST_TEMPLATE.md` and retain the existing stack index
verbatim. Update the relevant Contents and Testing performed prose in place so
each body remains coherent and does not accumulate campaign appendices.

| PR | Scope of the evidence update |
|---|---|
| #8 filtering | Explain the later #15 overlap/normalization improvement and link its qualified evidence; its head remains unchanged. |
| #9 search | Explain the later #15 CPU peak-reduction improvement and applicable workload results; its head remains unchanged. |
| #11 TaylorF2 | Explain the later #15 sparse FP64 logarithmic-term evaluation, unchanged float32 path and parity/AD checks; its head remains unchanged. |
| #15 evidence | Describe all final fixes, regression-test/CI wiring and the immutable supplement. Keep component timing attribution at its measured `450ab3`/`0d0058` revisions, historical executable attempts at `fb4b335`/`968bcd`/`6c8215`/`f2c0abe` as supplemental, and all primary matched executable runs at `837f38d`. Include the spectral-power, PSD/strain, chi-square precision fixes, the test-only rounding-reference correction and the CUDA coefficient launch correction. Identify any tests actually rerun on replacement heads. |
| #19 FFT formatting | Retain formatting-only scope. Record its new base and feature-preservation/qualification evidence without claiming the inherited fixes as its own. |
| #16 optional FFT | Retain its optional FFT scope and historical timing attribution. Add only applicable inherited-fix and new qualification information. |
| #17 optional CPU | Retain the optional native gate, OpenMP/Cython features and existing CPU results. Explain preservation of native dispatch before the improved Torch fallback and its qualified new evidence. |

Use numbers only from the final verified report: physical workload, batch/thread
scope, actual dispatch, denominator, replicates and observed ranges must agree.
Preserve slower cases, cold/steady separation, unsupported precision and old
revision attribution. If a test was run at `450ab3`, label it that way; byte-level
restack preservation supports transfer of those results but is not a fresh run.

Before candidate quality checks, require completed passing `unit-tests-v5.json`
and its unchanged `unit-tests-v5.log` from exact source `837f38d`. The receipt must
include all 16 full unit modules and all 12 reviewed file hashes (the earlier
four executable/context files plus eight precision files). Every candidate
must retain those reviewed bytes. The shared-file exceptions are #17's
`pycbc/filter/matchedfilter.py` and `pycbc/vetoes/chisq_torch.py`. Each normalized
feature-patch hash must match its recorded original #17 patch, and reversing
that exact candidate patch in memory must reproduce the reviewed file hash. Retain the resulting source-check
receipt; this preserves the optional CPU feature without excusing precision changes.

For all seven, preserve title, base, author, OPEN/draft state, `agent-assisted`,
the sentence `This PR was created by AI Gareth`, the unchecked Code of Conduct
checkbox and this exact operator note next to it:

> *AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*

Before publication, refresh all seven PRs and remote heads independently, compare
with the frozen snapshot and review the four explicit old-head leases and body
hashes. The primary agent owns tests, atomic four-ref publication and sequential
body/label edits with immediate readback and a partial-success journal. This
scaffold makes no network calls and provides no publication command.
