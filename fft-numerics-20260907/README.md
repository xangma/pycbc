# FFT numerical diagnostic evidence, 7 September 2026

This supplement preserves the completed R3 raw-output qualification and its
selected-row precision/attribution diagnostics. The numerical source is
[commit 9578a710479b924e882857c4dffab6ed372a634b](https://github.com/xangma/pycbc/commit/9578a710479b924e882857c4dffab6ed372a634b).
R3 remains **failed** under its original raw complex rule. Its six smoke checks
and all eighteen full qualification results are retained, including the twelve
failed Torch cells. All 54 timing workers were skipped. R4 uses a separately
adopted policy and is ongoing; its inputs, policy and results are outside this
archive. Packaging neither changes a verdict nor reruns the experiment.

## Contents and limits

After lossless reconstruction, original directory names are preserved:

| Directory | Evidence |
|---|---|
| `torch-fft-attribution-20260907-r2/` | Successful replay scripts, report, full replay/summary JSON, launch/status receipts and logs |
| `torch-fft-attribution-20260907/` | First failed replay attempt, including its partial JSON and failure log; provenance only |
| `torch-fft-precision-20260907/` | Diagnostic/plot/audit scripts and records, four cell JSONs/logs, and all 138 NPZ captures |
| `torch-current-batch-sweep-20260907-r3/` | Frozen worker/controller/helpers/tests, source/native provenance, input plan, all 24 stage receipts/logs and result/acquisition pairs, aggregate smoke/full JSON, and ancillary validation records |

The 138 captures cover 65 distinct template/block pairs: 36 rows each for Torch
CPU batches 1 and 8, and 33 rows each for Torch CUDA batches 1 and 8. They were
selected from three blocks using large raw tolerance ratios plus injections.
These are selected diagnostics, not a full-bank array archive. Batch-8 replay
broadcasts one captured correlation across eight rows and checks row zero; the
standalone complex128 intervention is also a subset experiment. Exact replay
equality was checked with `numpy.array_equal`; booleans named `*_bitwise_equal`
do not establish signed-zero bit equality.

The old `normalized-error-audit.json` and plotted normalized errors use a common
input-derived normalization. They exclude differences in normalization actually
used by each route and are **not complete normalized-output qualification**.
Original acquisition-time language and paths in frozen records remain unchanged;
later R4 adoption does not reinterpret an old diagnostic as an R4 pass.

All `.npy` arrays are deliberately omitted, including the full standard
`runs/qual-b1-branch_standard/outputs.npy` (shape 3 × 1024 × 131072, complex64;
3 GiB payload) and `validation/selected-reference-rows.npy`. The small smoke
reference array is also omitted. All row hashes and completed result JSON remain.
No `source/` checkout, `.git`, `.agents`, bytecode/cache tree, compiled native
library, environment or Git `source.bundle` is included. The bundle's original
hash remains in `staged-files.json`; that historical manifest describes the
original staging inputs, not the subset shipped here. Eleven native-extension
identities are retained as provenance, not binaries. Local precision
`TRANSFER.md`, `transfer.log` and `transfer-receipt.json` are omitted. Links in historical reports to
omitted source paths or the separate R4 directory need the mapping below.

## Verification and reconstruction

Files **larger than 1 MiB** are transported with the existing lossless gzip
helper, compression level 9, timestamp zero and no embedded original filename.
This archive's threshold is 1 MiB, overriding the generic helper guide's 90 MiB
default. `archive-transport/manifest.json` is authoritative: it records original
paths, byte counts, modes and SHA-256 hashes, plus compressed-file mappings.
`SHA256SUMS` covers the actual transported files; `PACKAGING-MANIFEST.json`
records the logical evidence inventory, omissions and offline checks. All frozen
scientific scripts, records and captures retain their original bytes.

From the immutable supplement root, using Python 3.11 or later:

```sh
sha256sum -c SHA256SUMS
python3 archive-transport/restore.py --verify-only
python3 archive-transport/restore.py --destination /absolute/path/to/new-restored-evidence
```

On macOS, use `shasum -a 256 -c SHA256SUMS` for the first command. Restoration
requires a new destination outside the archive. The standalone helper uses only
the standard library and verifies decompressed bytes before accepting them.
Compressed logical paths will not exist directly in the transport archive;
consult the manifest or restore first. Never rerun writers in immutable evidence.

Before packaging, offline NumPy validation requires exactly 138 NPZ files with
the recorded file SHA-256 values. Each must contain exactly `actual`, `reference`,
`correlation` (complex64) and `exact_correlation` (complex128), each with 131072
finite samples. The 276 actual/reference array hashes are compared directly
with frozen R3 row hashes; hashes for all 552 arrays are recorded in the packaging
manifest. Correlation arrays were not separately hashed in R3 result rows; their
bytes are covered by the captured NPZ file hash recorded in attribution replay.
The earlier frozen replay regenerated and checked exact products. Packaging
does not repeat that calculation or any FFT; its checks establish integrity
against recorded evidence, not a fresh scientific reproduction.
Metadata checks verify finite JSON numbers, unique coverage, original verdicts,
aggregate/result/acquisition consistency, and helper/source identity records.
This checks recorded source/native provenance, not absent native binaries.

## Relocating frozen scripts and reconstructing inputs

Frozen scripts intentionally retain their original host paths. Preserve these
copies. Create disposable working copies and edit only their path constants,
or provide equivalent symlinks in a disposable Linux environment. Record any
local patch and keep it separate from the published evidence.

| Original path | Relocated target |
|---|---|
| `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3` | Restored `torch-current-batch-sweep-20260907-r3/` |
| Its `source/` child | Clean checkout/install of the pinned commit |
| `/home/xangma/pycbc-torch-fft-precision-20260907` | Restored `torch-fft-precision-20260907/` for reading saved captures |
| `/home/xangma/pycbc-torch-fft-attribution-20260907-r2` | Disposable copy of the restored replay scripts, with a fresh output directory |
| `/home/xangma/pycbc-torch-split-20260905/venv/bin/python` | Compatible local Python environment; not included |

For source installation, obtain the linked repository revision, check it out
detached, build its native extensions locally and install editable with
`python -m pip install -e /absolute/path/to/pinned-source`. Follow the pinned
repository's installation requirements, including LALSuite. Match the recorded
Linux Python 3.11.9 / NumPy 1.26.4 environment and the recorded FFT libraries.
The `runtime` objects in the three `runs/qual-b1-*/result.json` files confirm
these versions; Torch cells record `2.13.0+cu130` and CUDA runtime `13.0`, and
the CUDA cell records an NVIDIA GeForce RTX 4090. CUDA additionally requires
compatible hardware and drivers. Review each result's `runtime`, thread settings and recorded source
hashes, plus `native-provenance.json`. Native binary hashes are build-specific;
a rebuild is a new environment, not a claim to possess the original binaries.

Input reconstruction itself needs only NumPy and the unchanged worker's
`workload` function; do not launch the controller or import PyCBC/Torch for it.
The following uses the original result arguments (seed 7101, 1024 templates,
FFT length 131072, three blocks), checks every returned identity field against
the frozen result, and writes no output arrays. It uses roughly half a GiB for
the template bank; this is an optional reproduction step, not archive validation.

```sh
export NUMERICS_RESTORED=/absolute/path/to/new-restored-evidence
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

campaign = Path(os.environ['NUMERICS_RESTORED']) / 'torch-current-batch-sweep-20260907-r3'
reference = json.loads((campaign / 'runs/qual-b1-branch_standard/result.json').read_text())
spec = importlib.util.spec_from_file_location('frozen_worker', campaign / 'batch-worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.np = np
bank, psd, sigma, blocks, geometry, injections, identity = worker.workload(
    SimpleNamespace(**reference['arguments']))
for key, value in identity.items():
    if value != reference['inputs'][key]:
        raise ValueError('Regenerated input identity differs: ' + key)
print('Verified bank, PSD, sigma, strain-block hashes and workload metadata')
PY
```

For an optional attribution replay, relocate `CAMPAIGN`, `CAPTURES` and `SOURCE`
in a disposable `replay.py`. It reconstructs inputs and uses archived reference
rows from NPZ, so it does not need the omitted full `outputs.npy`. It does need
the pinned installed source, MKL/FFTW and Torch/CUDA. Run into a fresh writable
directory, retaining original evidence for comparison. The original `run.py`
also pins Linux logical CPU 8 using `taskset`; adapt affinity only in a recorded
disposable copy. `diagnose.py` instead expects the omitted full reference array:
rerunning that acquisition requires regenerating a new reference outside the
archive and checking all its row hashes first. Do not blindly run the historical
controller or `stage-source.py`, which reference predecessor campaigns.
This is not a standalone turnkey reproduction guarantee on other platforms.

## Local packaging review workflow

The prepared script has no network, commit/push or acquisition operation.
Only `inspect` and its synthetic `self-test` may run while download is active.
`inspect` validates metadata and reports filenames without reading NPZ bytes.
After download completion, validate and prepare:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 stage-diagnostic-archive.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 stage-diagnostic-archive.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 stage-diagnostic-archive.py validate --download-complete
PYTHONDONTWRITEBYTECODE=1 python3 stage-diagnostic-archive.py prepare --download-complete
```

The last command validates captures, freezes a plan and deterministic transport
cache under `packaging/v1/`, and leaves the publication destination untouched.
Review `packaging/v1/plan.json`, its file/omission inventory and validation counts.
Then stage:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 stage-diagnostic-archive.py stage --download-complete
```

The default destination is the new
`torch-benchmark-controls-publication-20260907/fft-numerics-20260907/` under the
local artifacts root. Preparation and staging require a clean publication
checkout. Staging repeats capture validation, requires the reviewed plan/cache
to match, copies exclusively to a new directory, emits checksums and calls
`verify_archive`. It refuses an existing destination or changed input. Failed
partial plans/destinations remain for inspection; no cleanup or overwrite is
automatic. A new `--work` path inside attribution-r2's `packaging/` permits a
separately reviewed preparation attempt. No archive has been staged merely by
writing this script and README.
