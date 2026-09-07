# Integrated Torch profiling evidence — 7 September 2026

[REPORT.md](REPORT.md) explains the findings, timing boundaries and limitations. Clean source `d2647addb884ead3249914ebc980f3c132076d93` was measured on `len` with one pinned logical CPU, one-thread numerical pools, 384 compressed BNS/NSBH templates and 1,904 valid detector seconds. The CPU/MKL, Torch CPU and CUDA full-process medians are **70.219288 s, 114.163224 s and 25.009828 s**, from three fresh workers each. These are shared-host finite-workload observations; their observed ranges are not confidence intervals or an old/new speedup measurement.

## Contents and integrity

The nine `evidence.tar.gz.part000`–`part008` files concatenate, in order, to the original **224,392,604-byte** archive:

```text
a86204f5163c40f9ce011406cbc9f025209b48bbc9d95b18d0784bdce4772375
```

Each part is at most 25,165,824 bytes (24 MiB). Splitting did not rewrite the archive. It contains **1,831 members: 1,633 regular files, 194 directories and four internal relative symlinks**; regular-file contents total 1,316,404,965 bytes.

| File or restored path | Purpose |
| --- | --- |
| `SHA256SUMS` | Every publication payload file, including all parts and helpers; excludes itself |
| `archive-parts.json` | Ordered part sizes/hashes and original archive identity |
| `archive-inventory.json`, `archive-members.sha256` | Every archive member's metadata and every regular file's SHA256 |
| `pycbc-torch-profile-20260907-r3/source/`, `source.bundle` | Complete frozen source checkout and Git bundle, including 11 recorded Linux native extensions |
| `pycbc-torch-profile-20260907-r3/runs/` | All 21 HDF science outputs and worker receipts; six raw cProfiles, two raw native perf profiles with full/window reports, and the complete CUDA trace |
| Archive root helpers, configuration, stages and audits | Acquisition drivers, frozen comparator, trace parser, provenance, runtime/lifecycle controls, logs and qualification results |
| `previous-cpu/`, `previous-cpu-manifest.json` | Supplemental previous corrected CPU HDF, receipt and runtime record, copied byte-for-byte and separately pinned |
| `handoff.json`, `evidence-verification.json` | Original byte-preserved machine summaries used by the documentation plot |
| `publication-replay.json`, `publication-replay.log`, `publication-verification.json` | Publication restore/replay validation, separate from acquired evidence |

Top-level attribution JSON, timing summaries, audit/export records and [RUN.md](RUN.md) make the principal evidence directly readable. The archive remains authoritative for raw acquisition files. References to `remote-evidence/` in the report and original records map to the restored `pycbc-torch-profile-20260907-r3/` directory. Absolute `len` and local acquisition paths are recorded provenance, not portable path requirements.

Ordinary worker runtime receipts hash imported native extensions and selected Python source files, not every imported Python module. The full source snapshot is independently archived and verified. Checksums establish byte identity against reviewed pins; they are not signatures proving origin. Pin the publication commit and `SHA256SUMS` through the publication's trusted reference before executing its helpers.

## Restore without executing archived code

Use Python 3.10 or later. The restore verifier uses only the standard library and needs no network. Allow at least 3 GB of free disk space for temporary reconstruction, the retained archive and extracted files. Start in this publication directory; choose a fresh destination whose existing parent is a real path without symlink aliases.

```sh
python3 -I verify_bundle.py --check-only
python3 -I verify_bundle.py --restore /absolute/real/parent/new-profile-restore
```

Both modes verify the complete publication inventory, ordered parts, combined archive identity and all archived file hashes. Restore rejects traversal, duplicate paths, hard links, special files and escaping symlinks before extraction. It preserves ordinary permission and executable bits and the four internal relative symlinks; it does not restore ownership, timestamps or elevated mode bits. It also copies the three previous CPU companions into the destination. It does not import archived Python or native code. An interrupted restore may leave a partial destination; use a new destination for another attempt.

## Replay recorded science and attribution offline

Prepare a Python environment with NumPy and h5py before disconnecting from the network. The publication replay was tested using **Python 3.14.6, NumPy 2.5.3 and h5py 3.16.0**. Compatible `pstats`/`marshal` support is needed for profiles written with Python 3.11. Allow several GB of memory to parse the 486,838,687-byte CUDA JSON trace.

After verifying the bundle and reviewing the replay scripts, run from this publication directory:

```sh
python3 -I replay_evidence.py \
  --evidence /absolute/real/parent/new-profile-restore/pycbc-torch-profile-20260907-r3 \
  --previous /absolute/real/parent/new-profile-restore/previous-cpu/triggers.hdf \
  --output /absolute/real/parent/new-replay.json
```

The output must not already exist. This executes the frozen comparison/control/trace helpers and reads profile data; it does not execute `pycbc_inspiral`, load the archived native extensions, require Torch/CUDA, or rerun the acquisition. It verifies recorded source/native/helper/receipt pins, repeats all **20 current-run HDF comparisons**, checks all three 384-template/five-segment/1,904-second qualification records, and reconstructs the nine timing samples and backend medians. It reproduces the CUDA attribution and the six-profile and focused-caller summaries exactly. Raw native perf files and their published reports are checksum-verified; native symbolization is not rerun.

With `--previous`, it also verifies the supplied previous CPU companions and repeats the corrected previous/current CPU comparison. The raw strict metadata result is preserved. Only the explicitly recorded source-revision and byte-identical executable-path substitutions are made; numerical budgets remain `rtol=1e-4`, `atol=1e-5`, normalization `rtol=1e-5`, and circular phase `atol=1e-4`. The previous reference uses revision `9578a710479b924e882857c4dffab6ed372a634b`; its complete old source/runtime is not included in this supplement.

Recorded terminal and release audits demonstrate the acquisition-time process-group and lock checks. Offline replay verifies those records; it does not inspect the current state of `len`. [publication-verification.json](publication-verification.json) records the successful local restore and replay. `verify-evidence.original.py` is retained for provenance and contains original machine-specific paths; use `replay_evidence.py` for this package.

## Deliberately omitted execution inputs and runtime

The original bank and frame bytes are **not included**. They are unnecessary for replaying saved HDF comparisons, timings and attribution, but required for a new scientific acquisition. Original locations on `len` and exact SHA256 pins are:

| Input | Original location | SHA256 |
| --- | --- | --- |
| Compressed bank | `/home/xangma/pycbc-torch-reference-protocol-20260907/inputs/bank-compressed.hdf` | `26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f` |
| H1 frame | `/home/xangma/pycbc_bench_repo/docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf` | `580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865` |

These are original host locations, not public download guarantees. Offline replay checks recorded input pins and does not rehash these omitted inputs.

The acquisition Python environment, Torch wheel, CUDA driver/runtime, MKL/LAL/system shared libraries and perf debug/build-ID caches are also **not bundled**. The acquisition used Python 3.11 and Torch `2.13.0+cu130` (CUDA 13.0) on Linux x86-64 with an RTX 4090; runtime and native provenance records carry details. The 11 archived CPython 3.11 Linux extension binaries and incidental bytecode are evidence, not a portable runtime. Rebuilding or rerunning requires compatible dependencies and freshly established resource controls. This package reproduces analysis of the captured evidence, not a self-contained executable environment.
