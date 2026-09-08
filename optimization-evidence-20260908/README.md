# PyCBC optimization evidence, 8 September 2026

This copy-ready tree contains the five requested evidence packages. Copy its
contents into `optimization-evidence-20260908/` in the separate evidence Git
repository, review, then commit and push to obtain immutable revision URLs.
No publication, source integration, PR change or remote mutation has occurred here.

`documentation-index.json` maps reports and machine summaries to package-relative
and suggested branch-relative paths. Each package README describes its result,
raw evidence, frozen gates, source dependencies and reconstruction commands.
`publication-layout.json` records exact package sizes and original checksum seals.

All 2,337 non-cache input files (383,045,111 logical bytes) reconstruct exactly,
including all 17 original archives and their original checksum files. All 2,063
entries of the four original outer seals pass. The 270-file profiling root had no
outer seal and is identified as a freshly hashed publication snapshot. All 1,353
entries in 33 frozen input/results manifests resolve to preserved bytes. Only 115
interpreter-cache files are omitted, with hashes and reasons in package inventories.
Original archives retain exact bytes; duplicate acquired files refer to their
archive members or shared supplemental blobs. Temporary verification trees were
removed; no redundant restored tree is shipped.

The page-backing review, protocol and summaries are included in the profiling
package. Its separate raw campaign root was outside the five requested inputs.
Public GW bank/GWF bytes, full source baselines, native binaries, environments and
predecessor campaigns are external pinned dependencies, not missing selected
files. These packages support plotting recorded evidence; a new native or
production qualification is separate work.

Verify the entire copied tree with Python 3.10+:

```sh
python3 -I verify_all.py
```

Each `verify.py` also supports `--restore /absolute/fresh/destination`. Compare the
outer SHA256SUMS hash with the parent's independently supplied publisher value.
The helper verifies physical files, original archive bytes and members, all logical
files and original seals, without executing scientific code. The complete physical
inventory excludes only this tree's root SHA256SUMS itself.

The `_publication/` folder contains the packaging source, input inventory and
verification receipts. Every selected file was reconstructed on disk, rehashed and
compared with the original input, including file modes. All 22 compressed transports
were regenerated deterministically. Ten negative checks confirmed rejection of
corruption, missing/extra files, traversal, links, duplicate JSON keys, conflicting
paths and existing restore destinations. Five unique Git bundles were strictly
verified in disposable local object storage. All six final source blobs and the
exact original patch matched their recorded hashes; applying that patch to the
pinned baseline in a disposable Git index reproduced the final source tree.

The scan covered 3,484 UTF-8 text file/member instances and 107 distinct packaged
Git objects. No credential pattern or unrelated private data was identified. Five
broad keyword hits were ordinary uses of “authorization” in experiment review prose.
Public GW identities, hostnames, filesystem paths and process/source provenance
were retained. Nothing was redacted or rewritten in the original evidence.

Packaging reproduction uses only the standard library, plus local Git for source
bundle validation. The archived preparation scripts retain their original
`archive-prep` layout and refuse an existing package destination. Run them from a
fresh equivalent preparation directory with access to the recorded source roots;
`input-inspection.json` binds all selected bytes. The order is inspect, build,
validate source bundles, then validate packages. The original source seal and
package-level verifiers remain usable independently of these preparation paths.
The aggregate layout and documentation mapping are publication navigation aids.
