# Torch CPU squared-norm evidence, 7 September 2026

This package preserves the measured comparison of baseline `9578a710479b924e882857c4dffab6ed372a634b` with candidate `9e6a688a5190d6e1ddc655fbe352cc206085d5c6`. On the fixed 384-template, 1904-valid-second executable workload, the three-repeat median full wall time fell from **119.262860 s to 114.138915 s: 4.296% less wall time, 1.044892× speedup**. Identical runtime-verification startup is included. These are observations on shared `len`, with one numerical-library thread and CPU8 affinity, not confidence intervals or demonstrated steady-state capacity.

All four cross-revision comparisons match 1991 triggers with zero differences in all 11 compared scientific fields; six within-revision repeat comparisons pass. The original strict FAIL receipts are preserved alongside the separately reviewed wrapped PASS receipts. The strict failures concern only the pinned source revision and executable path. No numerical budget or precision was changed. Both qualifications covered all 384 compressed templates and 1920 scalar IFFTs without generation fallback.

All 108 raw API cells pass bitwise output parity and unchanged-input checks. The six below-cutoff medians show a **0.949316–1.502676 microsecond per-call penalty**. The tested fast-path gains do not establish an optimal or universally portable 4096-element cutoff. `API-REVIEW-REQUEST.md` includes all 18 aggregate cells and their observed ranges. `RESULTS.md` retains the original full report and its limitations unchanged.

## Contents and original archive

`executable-remote-evidence.tar.gz` is the unchanged **12,223,085-byte** archive, SHA256:

```text
f502e162bcc61e5636f5764572f09b6c3ec1e0ed0de617d15eefffdd0a3c9754
```

It contains **190 regular files and 19 directories**, totaling **33,368,555 uncompressed file bytes**. The original root is `.` and each descendant is spelled `./...`; every member name and archive byte is preserved. Extracted files sit directly in the chosen fresh destination, without an extra enclosing campaign directory.

The archive includes:

- All 23 pinned acquisition helpers and their manifest, configuration, source/native provenance, staging receipt, profiler release, approved plan and review.
- Both API and executable phases: launch and terminal records, before/after receipts, host samples, six API workers and all raw API cells.
- Two qualification and six timing workers, including all eight actual trigger HDFs, runtime receipts, logs, times and both qualification PSD NumPy arrays.
- All four original raw strict FAIL comparisons and four wrapped PASS comparisons, both within-revision comparison records, and Linux lifecycle/compatibility outputs.

Readable top-level copies include RESULTS, the full API review table, both phase summaries/statuses, both terminal audits, scientific comparison receipts, scientific summary, lineage, candidate patch, parent API authorization, local tests/lint/build logs, and the parent's later independent executable review and integrated-test log. These copies retain their source bytes. The parent's later review reports independent verification of all 328 sealed hashes, all 190 archive files and eight actual HDF outputs; its integration run passed 231 tests with 11 CUDA skips. The original report predates that review and retains its original integration-status wording.

`publication-sources.json` maps each copied file to its original local path, size and hash, and distinguishes files present in the original seal from later additions. Renamed readable comparison copies are mapped to their exact archived paths. `archive-inventory.json` records every member's original name, type, size, hash and tar metadata; `archive-members.sha256` lists all 190 file hashes. `FINAL-MANIFEST.json` is the unchanged original 328-file local seal, not this publication's file list. `sealed-evidence-coverage.json` accounts for every entry of that seal, including omitted duplicates. `SHA256SUMS` inventories every other top-level publication file, including the unchanged archive, verifier, inventories and validation receipt; it excludes only itself to avoid self-reference.

## Omissions and source reconstruction

The original input bank and GWF frame, two complete source checkouts, eleven native extension binaries, Python environments and numerical libraries are omitted. Their paths, revisions and hashes remain in the receipts. The derived PSDs and trigger outputs do not replace the omitted bank/frame. The preceding convergence, failed R3, successful adopted-v2 R4 and CUDA profiling campaigns are prerequisites recorded in the gates, not archives embedded here. Their earlier classifications and numerical policies remain unchanged.

The earlier API-only transfer archive is omitted as a duplicate container: all 76 of its file contents are preserved in the final archive. Local staging transfer containers, caches and redundant preparation copies are omitted; the original seal and coverage inventory identify them. No cache or native binary is inserted into this publication. Post-transfer reports, audits and reviews are provided beside the archive rather than retroactively inserted into it.

The compact `source-candidate.bundle` (2,872 bytes) contains the single candidate commit and requires the trusted baseline commit `9578a710479b924e882857c4dffab6ed372a634b` already present in a Git repository. It is not a complete source checkout. The full-index `candidate.patch` has SHA256 `06c76d2bd544f9ac96fbb407356e59bb7ef675a1edc47860f9470cf0e92f3b09`. `lineage.json` also records a different patch hash for the original non-full-index diff representation; both describe the same source change. Neither restoration nor verification applies the patch, imports a bundle or runs scientific software.

## Safe offline restoration

Use a reviewed immutable publication and compare the archive and SHA256SUMS hashes with independently supplied publisher values. Checksums establish byte identity, not origin. Review `verify_bundle.py` before running it. It uses only Python's standard library and imports or executes no archived code.

From this publication directory, use Python 3.10 or newer and an unused destination whose parent exists without symlink aliases. Allow at least 40 MB of free space for extraction:

```sh
python3 -I verify_bundle.py --restore ../optimization-restore
```

The verifier checks the complete flat publication inventory, rejects symlink or special publication files, checks the hard-coded original archive hash and the independent terminal audit's 190 file hashes, then validates every archived path, type, metadata record and directory relationship before creating the destination. It rejects absolute paths, traversal, duplicate names, path aliases, links, special files and file/directory conflicts. It accepts only the original `.`/`./...` path convention. Every restored file is written exclusively and its SHA256 checked; the restored member set must match the inventory exactly.

Restoration preserves hierarchy and file contents, not tar ownership, timestamps or permission bits. It creates a private fresh directory and needs no network, GPU, PyCBC, Torch, input bank or frame. `publication-verification.json` records a successful fresh local restore. A second complete restore was used to check the final package after adding that receipt. Twelve standard-library verifier checks passed, including unsafe paths, links/special files, duplicate members, metadata changes, file/directory conflicts, existing or symlink destinations and publication tampering; `verifier-tests.log` preserves the result. They can be repeated with `python3 -I test-publication-verifier.py`.

Keep raw HDF/NumPy outputs and host logs inside the compressed archive for publication rather than adding the extracted directory as individual Git blobs. The verifier checks their bytes but does not parse HDFs or rerun science. The recorded parent review separately checked the actual HDF values. Repeating that scientific check requires the reviewed comparison code and suitable numerical dependencies; running the acquisition additionally requires the separately obtained trusted inputs, source/native binaries, runtime and current host coordination. The archived campaign scripts contain historical paths and process identifiers and are evidence, not unattended replay instructions.
