# torch-residual-optimization-20260908: publication evidence

Descriptor reuse passed qualification and all four paired timing repeats; median wall reductions were 2.213184% for standard CPU and 7.665707% for CUDA. The separate NumPy-copy CPU candidate was rejected.

Read [HANDOFF.md](readable/HANDOFF.md) for the original report and limitations.
`publication-summary.json` is a navigation aid; `readable-sources.json` maps every
readable copy to its exact original path, SHA256, length and mode.

Complete CPU stage/copy diagnostics, descriptor diagnostic and timing inputs/results, native and science gates, all raw HDF/timing outputs, superseded unlaunched freeze and independent reviews.

This package preserves **606 of 606 selected evidence files**
(139,212,285 original bytes) and all 9 original archives.
Only 36 interpreter-cache files are excluded, with individual hashes
in `coverage.json`. Original archives are transported losslessly as timestamp-zero,
level-9 gzip streams; original tar bytes, names, metadata and checksums reconstruct
unchanged. Files already present in those archives use exact-byte member references.
Other duplicate files share a content-addressed supplemental blob. No redundant
restored tree is shipped. Small readable report/summary/source copies are deliberate.

`manifest.json` inventories every logical file and its storage mapping.
`archive-inventory.json` inventories every original tar member plus the deterministic
supplement; `source-inventory.json` identifies exact patches, bundles and source pins.
`evidence-inventories.json` preserves every discovered flat source/input/run hash
inventory and records matching payloads or external/historical dependencies.
All 595 entries of 8 frozen input/result
manifests resolve to preserved bytes. The original outer checksum seal is preserved
when present; absence is explicit in `coverage.json`.

## Verify and reconstruct

Use Python 3.10+ and independently supplied publisher hashes for authenticity:

```sh
python3 -I verify.py
python3 -I verify.py --restore /absolute/fresh/parent/torch-residual-optimization-20260908
```

Verification checks the full physical package, decompresses and checks original
archive bytes and all members, resolves and hashes every logical file, and checks
the original outer seal. Restoration requires an unused destination whose parent
exists without symlink aliases, outside this package. It validates all payloads
before writing and rehashes every exclusively written file. Original selected file
modes are restored; tar ownership and timestamps are not applied. The helper rejects
traversal, links, duplicate members, missing/extra files and file/directory conflicts.
It never imports or executes experiment code, contacts a remote system or changes
the source evidence. `SHA256SUMS` covers every physical file except itself.

## Regenerate plots from recorded evidence

Primary machine files are listed below; they contain recorded raw samples and/or
qualified decisions. All accompanying raw worker logs, timing JSON/CSV, HDF/PSD
outputs, qualification receipts, comparators, frozen protocols and existing
summarizers present in the source inventory reconstruct at their original relative
paths. Read those inputs to generate plots in a new directory. Preserve distinctions
between qualification, warm API, native, internal and complete-child measurements;
do not pool superseded or failed campaigns with the accepted timing samples.
Existing scripts retain historical absolute paths. Review and relocate their input
and output constants in disposable working copies before any optional replay.
This preparation ran no acquisition, FFT, runtime tests, plot generator or scientific
comparator. Integrity verification does not re-observe historical scientific gates.

- [`descriptor-reuse-timing-v1-summary.json`](readable/descriptor-reuse-timing-v1-summary.json)
- [`descriptor-reuse-timing-v1-samples.csv`](readable/descriptor-reuse-timing-v1-samples.csv)
- [`descriptor-reuse-v1-summary.json`](readable/descriptor-reuse-v1-summary.json)
- [`cpu-copy-v1-summary.json`](readable/cpu-copy-v1-summary.json)
- [`cpu-stage-v1-summary.json`](readable/cpu-stage-v1-summary.json)

The public input bank and GWF are identified by their original SHA256 pins, but
their bytes are not in these five input roots. Full source baselines, native
libraries, environments and predecessor campaign roots are likewise external.
Incremental Git bundles require the exact prerequisite commits listed in
`source-inventory.json`; they are not complete checkouts. The FFT package preserves
the final four-commit bundle/diff from 7e56ac42417dd6f61498f917e04c5a55250ddc26 to
ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f. Other packages refer to that pinned source
and preserve their own experiment helpers. A new experimental rerun requires those
dependencies and a separately reviewed execution environment.

Original acquisition-time statements such as “no publication” remain verbatim;
this package is a later publication preparation, not a rewrite of the acquisition.
No source integration, PR change, remote mutation or publication is performed here.
