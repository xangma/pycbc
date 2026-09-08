# Baseline/final benchmark publication validation

The current guide replaces the uncompleted-comparison status with the completed
unchanged-PyCBC versus final-main-proposal experiment. It reports four timing
arms and preserves failed scientific equivalence. The measured proposal remains
`123e1fb3ef1b338cada636e71c3e9c7987002402`; documentation-only descendants retain
its runtime bytes. Optional FFT/CPU branches are outside the measured proposal.

[Immutable raw benchmark, independent verifier and reproduction](https://github.com/xangma/pycbc/tree/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908).

## Validation

- Four separate executable qualifications and 16 fresh timed processes completed.
  All 16 timing trigger comparisons pass against their own qualification. The
  unchanged-baseline trigger comparisons and all five full-PSD comparisons fail;
  scientific eligibility remains false. No failed verdict or sample is discarded.
- The frozen independent verifier passes 47 focused self-tests and the full
  acquisition integrity check, including explicit dependency reconciliation,
  all 20 independent-clock checks, source/runtime/input pins, workload and
  schedule, scientific comparisons and receipt-derived timing summaries.
- A read-only remote inventory verifies all 290 raw acquisition files after
  transfer. All 311 sealed archive files pass SHA256 verification. Running the
  verifier from the archive directory also passes evidence integrity while
  preserving scientific FAIL; the archive remains byte-identical afterward.
- Every recorded tracked-source file matches its actual Git commit: 1078
  original files and 1227 proposed files. The proof binds the source-pin
  manifest SHA256. Native binaries retain build/import receipts. Torch version
  reconciliation is explicitly supported by a post-run diagnostic and cannot
  retrospectively identify the binary bytes loaded by workers.
- Fresh Sphinx `-E -a -W --keep-going` passes with zero warnings on all 12 retained
  Torch pages plus seven actual waveform/plugin/installation dependencies. It
  uses repository extensions/theme and executes plot/command directives. The
  targeted build excludes unrelated manual pages, generated includes and
  external intersphinx inventories; its API context is generated. The 19 built
  page hashes exactly match the committed main documentation head.
- The documentation diff passes `git diff --check`. No runtime changes were
  made for publication. Git confirms byte-identical implementation, executable,
  test, tool and CI trees across each previous/new head below. Earlier relevant
  unit-test receipts remain linked in the owning PRs; this documentation-only
  replacement does not require a new runtime benchmark after restacking.
- Bounded independent harness, verifier and scientific-report reviews are
  retained. The original verifier findings and their reviewed fixes remain in
  the raw archive; no current blocker was found within those reviews' scope.

## Head mapping

| PR | Previously published head | Replacement head |
| --- | --- | --- |
| #15 | `123e1fb3ef1b338cada636e71c3e9c7987002402` | `bd2d956f5c4e7c69b957181340d62d76a25717ac` |
| #19 | `aa32e7f83c8dc498d0b4b7f595ae179c192fd048` | `1b81a82b84814d6cdc6cc6d27cf47b1b4e195fde` |
| #16 | `b2972248e500b7c87dd6d8aee2a3b03418cec28f` | `52bb4742667671f178314b2bd1a23f83029cab32` |
| #17 | `0935c629c01dad44853840f0a14c402e599c845c` | `b213a6aeffa69ba0f249c8b5c883434d2371cecf` |

Only the performance results, reference-campaign reproduction links and
proposed-CPU terminology change. PR #19 remains based on #15, #16 on #19 and
#17 on #15. The other eleven PR heads are unchanged. Full tree IDs and changed
paths are in `publication-source-verification.json`; Sphinx scope, command,
source hashes and output are retained separately.

This package records completed local validation before publication. The local
publication receipt separately records guarded GitHub readback after pushing
the evidence and four branch heads and updating all 15 PR descriptions.
