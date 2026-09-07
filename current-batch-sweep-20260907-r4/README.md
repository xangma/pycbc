# R4 completed batch-sweep evidence

This package is for the **completed** R4 campaign at
[source 9578a710479b924e882857c4dffab6ed372a634b](https://github.com/xangma/pycbc/commit/9578a710479b924e882857c4dffab6ed372a634b).
Writing this README or its packaging script does not establish completion.
Preparation rejects a running, failed or incomplete local snapshot. A staged
package must pass the frozen policy-v2 evidence checks for all 12 smoke and 36
full qualifications, followed by 54 timing workers. R3 remains failed under its
original raw-output rule; its separate diagnostics are not reclassified here.
The separate [FFT diagnostic archive at immutable commit b26bbc1d](https://github.com/xangma/pycbc/tree/b26bbc1d612f92ad54377a2d687800ad207834d6/fft-numerics-20260907)
documents that investigation. The [repository renderer and documentation at
937a3f95](https://github.com/xangma/pycbc/commit/937a3f959b1a2176cd132690aa516c6cabeb0633)
were published later; the measured runtime remains source 9578a710 above.

## Contents and verification limits

After restoration, `torch-current-batch-sweep-20260907-r4/` retains all original
relative paths and bytes for:

- Frozen worker/controller/control scripts, tests, launch script, policy and
  adoption decision, review, validation, source/native identities and stage manifest.
- Original plan, status, launch/dependency/capacity/queued-input receipts,
  provenance, host-sample JSONL and launch log.
- All 102 `stages/*.{json,log,stderr}` sets, including empty stderr files,
  and all 102 `runs/*/{result,acquisition}.json` pairs.
- Both qualification aggregates, `timings.json` and `summary.json`.

`reproduction/` contains the reviewed offline verifier and lossless transport
helper/guide. `PACKAGING-MANIFEST.json` records the logical inventory, every
file's SHA-256, bytes and mode, exclusions, and verification scope. If supplied
explicitly, `plots/` contains four final PNG/SVG images and their original
`manifest.json`; these are later generated presentation artifacts, separate
from original acquisition evidence. No plot directory is inferred or generated
by the packager. The local `packaging-prep/` directory is excluded in full;
this archive README is copied separately to the supplement root.

If the parent supplies `terminal-audit.json` after campaign completion,
`supporting-provenance/` contains its unchanged bytes and the reviewed companion
`check-r4-terminal.py`. These are later final-check records, separate from the
original acquisition files. The packager checks the receipt's completion time,
status/stage-manifest hashes, source identity, counts, recorded PID/PGID sets
and declaration that no tracked process remains. It never runs the audit script.
The receipt has no script hash or cryptographic proof of execution; packaging
hashes identify the supplied receipt and companion bytes. They cannot establish
the current state of a remote host. Absent receipts are explicitly recorded as
not included in the packaging manifest's verification record.

The adopted policy requires every actual normalized complex SNR sample to agree
within absolute 0.001 with both an independent complex128 oracle and the actual
standard CPU/MKL result. Observed route normalization and the existing trigger
and veto checks are included. Raw version-1 and normwise metrics remain diagnostic.
The frozen worker policy's historical `DRAFT` description and `validation.json`
predate adoption and retain their original language and hashes. Adoption is
recorded by `policy-decision.json` and `NUMERICAL-POLICY.md`, whose current frozen
identities are bound by `stage-manifest.json`; the earlier validation receipt's
policy-document hash is not an adoption receipt.

Packaging pins the reviewed stage-manifest and offline-verifier bytes. It
checks all ten stage-manifest members, strict finite JSON/JSONL with unique keys,
exact material coverage, and final stdout receipts. The repository verifier
revalidates the complete qualification/timing matrices, aggregate/raw/acquisition
identities, policy/source/native records, worker intervals, serial execution,
qualification before timing, and the recomputed timing summary. Supplied plots
must pass its `--verify-only` CLI: consumed input hashes, renderer hash, scope
and all four image hashes must match. The packager also rejects missing/extra
plot files or manifest fields. It checks inputs again before accepting a plan
or staged copy.

These are **recorded JSON evidence and file-integrity checks**. They do not
replay arrays, recompute FFTs or triggers, rerender images, or verify absent
source/native binaries. Original completion receipts record acquisition-time
source/native checks; eleven native identities are preserved, not their binaries.

All `.npy` files are omitted, including every full and smoke `outputs.npy` and
`oracle.npy`. Each full seed's standard reference has a 3 GiB complex64 output
payload and a 6 GiB complex128 oracle payload (3 × 1024 × 131072); none is
included. Their per-row output/oracle hashes and normalization records remain
in JSON. No full array archive or cross-platform turnkey reproduction is claimed.
`source/` (including its predecessor symlink), Git bundles/checkouts, compiled
natives, environments, caches, `.git`, `.agents`, bytecode and local transfer
receipts/logs are excluded. Unknown evidence entries stop packaging for inspection;
they are not silently added or discarded. Excluded trees are never traversed.

## Lossless restoration

Files larger than **1 MiB** use the existing deterministic gzip transport
(level 9, timestamp zero, no embedded filename). This overrides the generic
helper guide's 90 MiB threshold. The transport manifest is authoritative and
records both original and compressed hashes, sizes and original file modes.
Every transported file must remain below 100 MiB; an oversized compressed file
stops preparation. No scientific bytes are truncated or rewritten.

From the immutable supplement root, with Python 3.11 or later:

```sh
sha256sum -c SHA256SUMS
python3 archive-transport/restore.py --verify-only
python3 archive-transport/restore.py --destination /absolute/path/to/new-restored-r4
```

On macOS use `shasum -a 256 -c SHA256SUMS` for the first command. Restoration
requires a new destination outside the archive and only the Python standard
library. It verifies every decompressed byte against the recorded original hash
and size, and restores modes. Direct paths to compressed logical files exist
only after restoration. Keep writers outside the immutable evidence.

If plots were included, recheck them and all campaign JSON after restoration:

```sh
export R4_RESTORED=/absolute/path/to/new-restored-r4
python3 "$R4_RESTORED/reproduction/plot_torch_batch_sweep.py" \
  --campaign "$R4_RESTORED/torch-current-batch-sweep-20260907-r4" \
  --output "$R4_RESTORED/plots" --verify-only
```

That check uses only the standard library. Without supplied plots, the same
script can validate and generate new plots by omitting `--verify-only` and
choosing a new output directory outside the restored campaign. Generation
additionally needs Matplotlib. To validate JSON alone without generating plots:

```sh
python3 -B - <<'PY'
import importlib.util
import os
from pathlib import Path
root = Path(os.environ['R4_RESTORED'])
spec = importlib.util.spec_from_file_location('r4_verifier', root / 'reproduction/plot_torch_batch_sweep.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
verifier.validate_campaign(root / 'torch-current-batch-sweep-20260907-r4')
print('Recorded campaign JSON verified; arrays and science were not replayed')
PY
```

## Input and environment reconstruction

Original scripts retain `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r4`
and predecessor source/environment paths. Use disposable copies with recorded
path changes, or equivalent symlinks in a disposable Linux environment:

| Original path | Replacement |
|---|---|
| R4 campaign root | Restored campaign for reading; a separate fresh directory for new acquisition |
| R4 `source/`, linked to R3 `source/` | Clean checkout of the linked source commit, locally built and installed |
| `/home/xangma/pycbc-torch-split-20260905/venv/bin/python` | Compatible installed Python and numerical libraries |
| `runs/*/{outputs,oracle}.npy` | New regenerated reference/oracle files, checked against original row hashes |

Install pinned source with `python -m pip install -e /absolute/path/to/pinned-source`
after building prerequisites/native extensions according to its installation
instructions, including LALSuite. The locally available full seed-7102 batch-1
result `runtime` records show Python 3.11.9, NumPy 1.26.4, Torch 2.13.0+cu130,
CUDA runtime 13.0 and an NVIDIA GeForce RTX 4090 for the CUDA route. Consult every
final result's runtime, FFT plan, affinity and thread records before reproducing;
rebuilding binaries creates a new environment, not possession of original binaries.

Input reconstruction uses only NumPy and the frozen worker's `workload`; it
does not launch acquisition or import PyCBC/Torch. This optional snippet checks
every returned identity field for both full seeds and retains no array files:

```sh
python3 -B - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

root = Path(os.environ['R4_RESTORED']) / 'torch-current-batch-sweep-20260907-r4'
spec = importlib.util.spec_from_file_location('r4_worker', root / 'batch-worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.np = np
for seed in (7102, 7103):
    reference = json.loads((root / f'runs/qual-s{seed}-b1-branch_standard/result.json').read_text())
    values = worker.workload(SimpleNamespace(**reference['arguments']))
    for key, value in values[-1].items():
        if value != reference['inputs'][key]:
            raise ValueError(f'Input identity mismatch for seed {seed}: {key}')
    del values
    print(f'Seed {seed}: bank, PSD, sigma, block hashes and workload metadata match')
PY
```

The bank needs about half a GiB per seed. This verifies the subset of `inputs`
returned by `workload`; later runtime bank-parameter records are not regenerated.
Regenerating absent outputs/oracles requires a fresh compatible numerical run
and comparison to all corresponding saved row hashes; the snippet does neither.
Do not blindly launch historical controllers: they reference predecessor
campaigns and require fresh output directories. Timing remains a shared-host,
one-thread, CPU-8-pinned warm public-filter experiment; CUDA adds one GPU.
It excludes frame I/O, PSD/waveform/bank preparation, executable overhead and
full-machine capacity. Throughput counts 3072 template-block evaluations per
iteration; worker min/max is observed spread, not a confidence interval.

## Local packaging commands

Run from the local `packaging-prep/` directory. `inspect` reports missing
material and never declares success for a partial snapshot:

```sh
python3 -B stage-r4-archive.py inspect
```

After final download completion, validate and prepare. Supply `--plots` only
when the four final plots and their manifest already exist in a separate
directory produced by the reviewed repository renderer. The agreed local path
is `/Users/xangma/repos/pycbc/artifacts/torch-current-batch-sweep-20260907-r4-plots`:

```sh
export R4_PLOTS=/Users/xangma/repos/pycbc/artifacts/torch-current-batch-sweep-20260907-r4-plots
python3 -B stage-r4-archive.py validate --download-complete --plots "$R4_PLOTS"
python3 -B stage-r4-archive.py prepare --download-complete --plots "$R4_PLOTS"
```

Omit `--plots` on both commands for a JSON-only evidence supplement. The
default verifier is the reviewed benchmark worktree's
`tools/plot_torch_batch_sweep.py`; `--verifier` may relocate identical bytes.
Changed verifier or stage-manifest bytes fail the pinned identity check.
An optional terminal audit is detected only at the campaign root. Its default
companion is `/Users/xangma/repos/pycbc/artifacts/torch-numerics-publication-20260907/check-r4-terminal.py`;
`--terminal-audit-script` may relocate identical reviewed bytes. A present audit
with a missing, changed or inconsistent companion/receipt rejects packaging.
Preparation writes only `packaging-prep/work/v1/` (plan, receipt and transport
cache), leaving publication untouched. Review `work/v1/plan.json`, its complete
inventory, omissions, verification scope and compressed mappings, then stage:

```sh
python3 -B stage-r4-archive.py stage --download-complete --plots "$R4_PLOTS"
```

Use the same plot choice and paths as preparation. The destination is the new
`torch-benchmark-controls-publication-20260907/current-batch-sweep-20260907-r4/`
under the local artifacts root. Prepare/stage require a clean publication
checkout. Stage repeats full validation, requires the unchanged plan/cache,
creates the destination exclusively, emits `SHA256SUMS` and verifies lossless
transport. It never overwrites, commits or pushes. Failed partial plans or
destinations remain for inspection; cleanup is not automatic. A fresh `--work`
path below `packaging-prep/work/` supports another preparation attempt.
