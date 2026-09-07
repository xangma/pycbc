# Second Python optimization pass — no demonstrated executable gain

Candidate `af4f0bd598ec1e2d7c6ce702d6c05850d7871039` remains **isolated and unintegrated** on baseline `d2647addb884ead3249914ebc980f3c132076d93`. Reusing the Torch CPU squared-norm output buffer improves all 14 measured changed-path API medians by 1.0169–1.1074×. The full executable medians are 114.270831585 / 114.251992669 seconds (B/C 1.000164889×), with overlapping ranges across three repeats per revision on shared len. This does not demonstrate an executable benefit. The coordinating task directed that this pass close without another candidate or campaign.

All 120 API cells pass bitwise parity and unchanged-input checks. All eight executable workers qualify or complete successfully; four cross-revision comparisons match 1,991 trigger identities with zero differences in all 11 numeric fields, and six within-revision comparisons pass. Four original strict FAIL comparisons are retained alongside the authorized metadata-only wrapped PASS records. Comparator tolerances remain unchanged. REPORT.md contains the six full-wall samples, limits, validation and recovery details; API-RESULTS.md contains all 20 API rows.

## Contents and exact bytes

The unchanged original remote export, `evidence.tar.gz`, is 12,336,489 bytes, SHA256:

```text
f62fc1c56c2c6820a444a0c73ec331b6b2fa54c7bafe3aefebca9f2bcbb320f1
```

It contains 202 campaign files plus the original evidence manifest: all 24 pinned helpers, source/native/input provenance, source staging and failed staging records, launch/status/host observations, six API workers with 120 raw cells, all eight HDF outputs and their receipts/runtime/logs/times, both qualification PSD arrays, four raw FAIL and four wrapped PASS comparisons, and both within-revision records. Archive file paths and bytes remain unchanged. The original campaign root was `/home/xangma/pycbc-torch-python-optimization-pass2-20260907`.

`local-evidence.tar.gz` preserves 132 local acquisition/preparation/validation files (528,884 compressed bytes; 2,366,881 uncompressed bytes), including original and corrected validation drivers, all local test/lint logs, source lineage, original failed local copy and failed staging evidence, exact transfer packages and independent audits. SHA256:

```text
c733ee73da66f242a00b142122fc7cbc4ed609282f2ad22773be2aa7de04bdf3
```

Readable top-level copies preserve their source bytes. `publication-sources.json` maps these copies to original paths, lengths and hashes. `archive-inventory.json` lists every member of both archives. `local-evidence-manifest.json` records the local acquisition selection and omitted duplicate/cache directories. `evidence-manifest.json`, `executable-terminal-audit.json`, `export-release-audit.json` and `transfer-verification.json` bind the remote export to independent terminal and transfer checks. SHA256SUMS inventories every other flat publication file, excluding only itself.

Local tests passed: candidate and exact baseline focused suites 235 each (11 CUDA skips each); candidate integration 476 (102 accelerator skips); local acquisition controls 142 (one Linux-only skip), and len controls 143 with zero skips. The validation driver return-code defect and first source-staging failure were corrected before measurement; original bytes remain archived. All 28 owned process groups are inactive, the shared lock was independently reacquired, and all pins remained unchanged.

The bank/GWF inputs, complete source checkouts, native binaries, environments and predecessor campaigns are omitted; their hashes and dependencies remain in the receipts. Derived PSD and HDF outputs do not replace those inputs. The candidate bundle contains one commit and requires the exact trusted baseline already available in a repository; it is not a complete checkout. The full-index patch SHA256 is `cba88c8de4dd93df8c0c72c2aec3a661aa22f4745b91d80a94922d23729956e8`.

## Offline restoration

Compare the published SHA256SUMS hash and archive hashes with independently supplied publisher values, then review restore.py. It uses only the Python standard library and never imports or executes archived code. Use Python 3.10+ and an absolute unused destination whose parent exists without symlink aliases. Allow 40 MB for restored evidence:

```sh
python3 -I restore.py --restore /absolute/fresh/pass2-restored
```

The helper checks the complete flat package and both fixed archive hashes, validates every regular member/path/hash before creating the destination, and writes each file exclusively. Links, special files, traversal, duplicates, file/directory conflicts and existing destinations are rejected. Restored bytes appear under `pass2-restored/remote` and `pass2-restored/local`; tar ownership, timestamps and permissions are not restored. These are evidence files, not an installed runtime.

## Replay the ten actual HDF comparisons

Use a Python environment with NumPy and h5py. After restoration, these commands invoke the unchanged archived comparator on the eight actual HDFs and their original sibling receipts. They write results only to stdout. Four cross-revision raw comparisons must exit **1**, with exactly the two source/executable metadata mismatch reasons and no review reasons. Each of the last two commands performs three within-revision comparisons and must exit **0**.

```sh
PASS2_REMOTE=/absolute/fresh/pass2-restored/remote
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/qualify-baseline/triggers.hdf" "$PASS2_REMOTE/runs/qualify-candidate/triggers.hdf"
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/baseline-r1/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r1/triggers.hdf"
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/baseline-r2/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r2/triggers.hdf"
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/baseline-r3/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r3/triggers.hdf"
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/qualify-baseline/triggers.hdf" "$PASS2_REMOTE/runs/baseline-r1/triggers.hdf" "$PASS2_REMOTE/runs/baseline-r2/triggers.hdf" "$PASS2_REMOTE/runs/baseline-r3/triggers.hdf"
python3 -B "$PASS2_REMOTE/compare-triggers.py" "$PASS2_REMOTE/runs/qualify-candidate/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r1/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r2/triggers.hdf" "$PASS2_REMOTE/runs/candidate-r3/triggers.hdf"
```

Do not group the expected exit-1 commands in a shell configured to abort on nonzero status. Result display paths reflect the restored location; original receipt metadata is not rewritten. Scientific fields and trigger identities can be compared directly with the archived raw records. The four separately authorized wrapped PASS records and their exact metadata substitutions are preserved, with independent on-host revalidation in executable-terminal-audit.json. Offline replay checks the recorded outputs; it does not re-observe historical process ownership, native libraries, source imports or input acquisition.

The acquisition scripts contain historical paths, lock ownership, process IDs and revision pins. Restoring evidence is not authorization to rerun them. A new measurement requires a fresh reviewed acquisition root, trusted omitted inputs/runtime, current ownership coordination and repinned controls. No further measurement or integration is part of this completed pass.

The coordinating task also independently replayed all 202 remote evidence hashes, the eight worker source/runtime records, six wall samples, four exact metadata-only cross-revision comparisons and six strict within-revision comparisons. Its original verification code and JSON/log are included as parent-verify-pass2.py and parent-pass2-primary-replay.json/log. They retain their acquisition-specific local paths; the portable replay commands above use the unchanged comparator directly.
