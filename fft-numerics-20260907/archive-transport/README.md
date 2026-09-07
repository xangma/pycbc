# Reconstructing large evidence files

Files larger than 90 MiB are transported as deterministic gzip files so every
committed archive file remains below 100 MiB. Compression changes no scientific
input or result. Original artifact files and all scientific reports retain their
original paths, sizes and SHA-256 hashes. `manifest.json` records those original
files and maps compressed originals to their archived gzip files and hashes.

From this supplement's root, first verify the committed files using its checksum
inventory, then verify all originals without writing them:

```sh
sha256sum -c SHA256SUMS
python3 archive-transport/restore.py --verify-only
```

The standalone helper requires Python 3.11 or later and no third-party packages.
To reconstruct every original file, choose a new directory outside the archive:

```sh
python3 archive-transport/restore.py --destination /absolute/path/to/new-reconstructed-supplement
```

It checks compressed and uncompressed hashes, checks every decompressed byte
against the original size and hash, copies uncompressed files, restores original
file modes, and verifies each reconstructed file. It refuses an existing
destination and leaves any failed partial reconstruction for inspection. The
archive remains unchanged. Logical scientific inventories apply to the
reconstructed supplement; the archive's `SHA256SUMS` applies to its actual
transport files. Checksum and transport metadata are not original scientific
files and are not copied into the reconstructed directory.

Use the reconstructed supplement as the evidence root when rerunning a report
builder or documentation writer, or when opening original `perf.data` files.
Keep the immutable archive as the source of published evidence links. A direct
link to a compressed original path will not resolve in the archive: link to the
transport manifest or this guide and use the reconstruction command above.

The staging plan keeps two inventories: `logical_files` for scientific
qualification and `files` for the actual archive copies and checksums. Gzip
copies are frozen and verified in the local publication cache before staging;
execution requires the identical reviewed plan and existing cache. Gzip uses
compression level 9, timestamp zero and no original filename. The manifest
records exact compressed hashes, so verification does not require another
system's compressor to reproduce the gzip stream. A compressed file that still
exceeds the size limit stops staging rather than discarding or truncating it.
