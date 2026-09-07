# Qualified CUDA profiling evidence, 7 September 2026

This bundle preserves one successful, scientifically checked 384-template CUDA acquisition on `len`, using runtime revision `9578a710479b924e882857c4dffab6ed372a634b`. `FINAL-REPORT.rst` is the report from profiling commit `53d3591118875523673d7e60bfcfd4ac299b2708`; it also discusses earlier CPU profiles and the separate qualified R4 live-batch sweep. Those earlier campaigns are linked in the report and are not included here.

Nine Linux lifecycle controls passed before acquisition. Qualification, profiled/unprofiled CUDA comparisons, and CUDA/CPU comparisons passed with 1991 matched triggers and none unmatched. All 384 templates and 1920 IFFTs are covered. Source/native/input/helper hashes remained unchanged. The terminal audit found every owned process group inactive and independently reacquired the shared lock before optimization was released. Instrumented durations are attribution evidence, not throughput or GPU occupancy.

## Archive and readable evidence

`remote-evidence.tar.gz` is the original unchanged archive: **35,393,857 bytes**, SHA256:

```text
10298718066da7d1266a00cd874e87c7aaa0e7a8221cf1bfdd829196aa108870
```

It contains exactly **81 regular files and 10 directories**, totaling **497,478,153 uncompressed file bytes**, beneath `pycbc-torch-device-profile-20260907-r2/`:

- The complete 12-file acquisition bundle and staged helper hashes; source/native provenance receipts, configuration, launch and terminal records.
- All stage logs/receipts, Linux control output, four science runs with runtime receipts and trigger HDFs, unchanged scientific comparison results, and host observations.
- The qualification's derived PSD NumPy array and recorded template coverage.
- The 486,837,399-byte raw CUDA JSON trace, completed trace receipt, and derived attribution JSON.
- Six incidental Python bytecode cache files, preserved as evidence only.

The archive omits the original compressed template bank and GWF frame, the source checkout, eleven native extension binaries, the Python environment, CUDA/MKL/LALSuite libraries, and earlier CPU/convergence/R4 campaign archives. Input paths and hashes are recorded; those files are not embedded. The derived PSD and trigger HDFs are outputs, not replacements for the missing bank/frame. The final report and local validation were created after transfer and are supplied alongside the unchanged archive, not retroactively inserted into it.

Readable top-level copies include final campaign status, terminal audit, both scientific comparisons, trace attribution, trace receipt, provenance, Linux controls, local validation and final report. `archive-inventory.json` inventories every archive member, including file SHA256, sizes, types and original tar metadata; `archive-members.sha256` lists all 81 file hashes. `SHA256SUMS` covers every other published file, including the archive, verifier, inventory and readable copies. `publication-verification.json` records the fresh local restore check.

The raw trace, HDF/NumPy outputs, bytecode, and all other archived members remain inside the compressed tar for publication. Do not add the restored directory as individual Git blobs. No original scientific input or native extension binary is included as an individual publication file.

## Trust and offline extraction

Use a reviewed immutable publication and compare its archive/hash-manifest hashes with the publisher's independently supplied values. Checksums establish byte identity, not origin. Review `verify_bundle.py` before executing it. It uses only the Python standard library, rejects absolute/traversing/duplicate paths, links and special files, requires a fresh destination, and verifies every restored file without importing or executing archived code. Restoration preserves file contents and hierarchy; it does not restore tar ownership, timestamps or permission metadata.

Run these exact commands from this publication directory, using Python 3.10 or newer. The destination must not already exist; use a parent directory without symlink aliases. Allow roughly 0.5 GB for extraction and several GB of available memory if replaying the JSON trace.

```sh
python3 -I verify_bundle.py --restore ../profile-restore
```

The verifier checks `SHA256SUMS`, the hard-coded reviewed archive hash, all archive paths/types, and restored hashes. It prints a JSON verification result. No network connection, PyCBC installation, GPU, input bank/frame, or scientific execution is needed.

## Offline trace replay

After the verified extraction, run the archived, reviewed standard-library analyzer by absolute path in Python isolated mode. The output file must be new:

```sh
profile_restore="$(cd ../profile-restore && pwd)"
profile_evidence="$profile_restore/pycbc-torch-device-profile-20260907-r2"
python3 -I "$profile_evidence/summarize_torch_trace.py" \
  "$profile_evidence/runs/trace-current-cuda/trace/trace.json" \
  --receipt "$profile_evidence/runs/trace-current-cuda/trace/receipt.json" \
  --output "$profile_restore/replayed-device-attribution.json"
cmp "$profile_restore/replayed-device-attribution.json" \
  "$profile_evidence/device-attribution.json"
```

The analyzer verifies the completed trace receipt and trace SHA256 before processing pure JSON. Its exact SHA256 is `7f268a7bd36a4512e00b5863b21945039bec26ca1a2f3e3c7e198cdf32fc43cd`. Isolated mode prevents the extracted directory, user packages and `PYTHONPATH` from supplying imports; the analyzer imports only standard-library modules. Do not load the preserved `.pyc` files or run the campaign/qualification helpers as part of offline replay. Previous local replay produced byte-identical attribution, as recorded in `local-validation.json`; the publication restore separately verifies those same input and helper bytes.

Re-running science would require separately obtaining the trusted pinned PyCBC source, verified native binaries, original bank/frame and compatible numerical environment, then reviewing and adapting the host-specific paths and coordination gates. Those requirements are not supplied by offline trace replay. The approved R4 `source` path resolves to the preserved R3 checkout at the same clean runtime revision; this explains receipt paths and does not change the failed R3 campaign's classification.
