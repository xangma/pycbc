# Gzip archive transport review

The new v5 staging plan preserves original `logical_files` for every scientific qualification and puts only the actual copied files in `files` and supplement `SHA256SUMS`. The scientific qualification functions and original artifact files are unchanged. The helper and real-file validation did not run the real stager or copy either real archive supplement, run Git/network commands, or run scientific campaigns.

`stage-evidence.py` now freezes deterministic gzip transport for all files over 90 MiB in its publication cache. The existing plan must match exactly before `--execute`, and execute requires an existing cache. Gzip files, a portable manifest of original paths/hashes/sizes/modes, a standalone restore helper, and its guide enter the actual archive inventory. Compressed files still at or above 100 MiB stop staging. The threshold can be lowered for isolated fixtures but cannot exceed 90 MiB.

All 26 isolated tests pass, including standalone CLI reconstruction, a complete staged fixture, determinism across independent caches, immutable cache reuse, corruption controls, changed-plan rejection before archive copying, exact original modes, and unchanged historical files. AST comparison confirms every original non-main staging function is unchanged. The 3 real large perf files passed streaming decompression checks and frozen cache reuse:

| Original file | Original bytes | Gzip bytes |
| --- | ---: | ---: |
| runs/profile-precision5-torch-cpu-l512-perf/perf.data | 263631296 | 22961455 |
| runs/reference-cpu-l512-perf/perf.data | 104912964 | 6579902 |
| runs/reference-precision5-cpu-l512-perf/perf.data | 108825132 | 6334103 |

The probe cache `publication/gzip-transport-validation-cache` contains only those three files and its manifests. It is distinct from the default full staging cache `publication/transport-cache`. Let normal inventory mode build the full staging cache; do not point the full stage plan at the probe cache.

## Dependent-script review

- `publish.py` checks exact actual Git change inventories and individually named evidence blobs. It does not consume the v4 staging schema or hard-code perf.data paths; populate its reviewed archive changes with the actual transport paths.
- `restack.py`, `write-bodies.py`, and `qualify-quality.py` do not consume the staging-plan schema or require raw perf.data files at published paths. Existing named JSON/MD evidence remains at its original location.
- `write-docs.py` and `build-reference-report-v4.py` validate the original logical hashes. They continue to work on the untouched local artifacts. To rerun them from the archive, first reconstruct a new supplement with `python3 archive-transport/restore.py --destination /new/path` and use that new root. No changes made to either file for this transport task.
- `ARCHIVE_STAGING_PLAN.md` still describes v4/raw copies; update its freeze/copy description to v5 and link `ARCHIVE_TRANSPORT.md` before treating that plan prose as final.
- Add a link to `archive-transport/README.md` in the reference supplement's REPRODUCE/README, with the verify/reconstruct commands. Those files are root-owned and were not changed here. Do not add that link to the supporting supplement when it has no oversized files, because transport files are only emitted for supplements needing them. The supporting supplement's existing link validator uses its logical inventory.
- Published links to any compressed original path would be absent in Git: use the transport guide/manifest for those files. Existing raw scientific receipts deliberately retain original logical paths and hashes.

Review `gzip-transport-review.diff` and `gzip-transport-validation.json`. The real archive is unchanged; staging remains the primary agent's next action after review. Use a new plan path if an earlier v4 plan already exists.
