# Assemble the batch-1024 evidence supplement

`assemble-archive.py` is a local, three-phase helper. It does not acquire data, run PyCBC tests, change Git state, or publish anything. Review the helper before using it. It has not been executed against campaign data.

Preparation requires the completed `sealed-evidence.tar.gz`, all three completed correctness receipts/JUnit files, the current local report engines and validation receipt, and the final documentation renderer/test file. The evidence checkout must be clean at `5c6c12d895fa7229fd0da2d10ade5ecfbae5153b`, with no existing `batch1024/`. Use Python 3.11+; the render interpreter also needs matplotlib.

```sh
python assemble-archive.py prepare \
  --archive /PATH/TO/EVIDENCE_CHECKOUT \
  --sealed /PATH/TO/sealed-evidence.tar.gz \
  --docs-root /PATH/TO/PYCBC_DOCUMENTATION_CHECKOUT

python assemble-archive.py render \
  --archive /PATH/TO/EVIDENCE_CHECKOUT \
  --python /PATH/TO/PYTHON_WITH_MATPLOTLIB

# Inspect all eight PNGs under batch1024/documentation/figures/,
# batch1024/README.md and batch1024/assembly/root-README.prepared.md.
python assemble-archive.py finish --archive /PATH/TO/EVIDENCE_CHECKOUT
```

Run from the helper directory, or use its absolute path. `--local-root` defaults to that directory. Each report/render child has a 300-second timeout, configurable with `--timeout`; command, host, cwd, PID, log and stop command are printed before waiting.

`prepare` validates canonical tar paths, regular-file types, duplicate members, sizes and every sealed SHA-256. It rejects symlinks, hard links, sparse files and special permission bits. Extraction is manual into a temporary sibling directory; no archive extraction API can traverse outside it. Files must be below 100,000,000 bytes, with at most 50,000 members and 20 GiB unpacked. It verifies completion/source/binary/test/log identities, preserves the sealed records byte-for-byte, and installs a fresh supplement atomically. Later strict engines and adjacent hash-matched acquisition files live separately under `report-engines/`. Local validation retains its original claims; `validation/source-matches.json` explicitly records matches or differences from the copied source files.

`render` recomputes all four strict reports. Failed reports and their logs remain available; any failed qualification prevents documentation rendering. The standalone `documentation/render-manifest.json` binds the three new summaries and retained historical inference/FFT summaries, then records all eight actual figure hashes. It has no pin to its own future archive commit. Reproduction instructions use the separate report engines and the standalone renderer CLI.

`finish` rechecks sealed inputs, reports, figures and all old files, writes the reviewed README update, and regenerates supplement/root `SHA256SUMS`. The only pre-existing files written are root `README.md` and `SHA256SUMS`; all others must match the baseline inventory. Checksums have no circular dependency: the supplement excludes its own checksum file, and the root includes that file while excluding only root `SHA256SUMS`.

Phases are guarded against repetition or overwriting. A changed helper/input, pre-existing output, unexpected old-file change, concurrent invocation or failed phase stops the operation. Failure output is kept for review; the helper performs no automatic reset, deletion or rollback. For a reviewed retry, use a fresh clean archive checkout. Do not stage/commit between phases. Once `finish` succeeds, independently review the diff and maximum file size, then commit/publish separately. Pin the final source-documentation manifest to that resulting commit afterward.
