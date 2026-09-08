# torch-offline-cuda-graph-20260908: publication evidence

Version 2 passed the frozen rule: median complete-child wall 20.257390172453597 to 19.94459447549889 seconds, a 1.5441065916776076% reduction, with all four pairs favorable. Version 1 failed qualification and ran no timing samples.

Read [OFFLINE-CUDA-GRAPH-v2.md](readable/OFFLINE-CUDA-GRAPH-v2.md) for the original report and limitations.
`publication-summary.json` is a navigation aid; `readable-sources.json` maps every
readable copy to its exact original path, SHA256, length and mode.

Both fixed diagnostic versions, original input/results archives, frozen native/executable gates, raw HDF/science comparisons, eight timing samples and closure evidence.

This package preserves **320 of 320 selected evidence files**
(68,984,932 original bytes) and all 4 original archives.
Only 11 interpreter-cache files are excluded, with individual hashes
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
All 375 entries of 6 frozen input/result
manifests resolve to preserved bytes. The original outer checksum seal is preserved
when present; absence is explicit in `coverage.json`.

## Verify and reconstruct

Use Python 3.10+ and independently supplied publisher hashes for authenticity:

```sh
python3 -I verify.py
python3 -I verify.py --restore /absolute/fresh/parent/torch-offline-cuda-graph-20260908
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

- [`acquired-diagnostic-v2/timing-summary.json`](readable/acquired-diagnostic-v2/timing-summary.json)
- [`offline-cuda-graph-v2-samples.csv`](readable/offline-cuda-graph-v2-samples.csv)
- [`audit-v2.json`](readable/audit-v2.json)
- [`peer-results-review-v2.json`](readable/peer-results-review-v2.json)

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
