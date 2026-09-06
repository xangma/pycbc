# Reproducing the executable comparison

Keep the immutable archive unchanged. Verify both sibling supplements with
`sha256sum -c SHA256SUMS` from their roots. Original receipts record measured
host paths; adapt copies and record new receipts instead of editing evidence.

## Restore raw evidence

Large profiles use lossless gzip transport. Restore the primary supplement to
an unused directory before running its report builders or opening raw profiles:

```sh
python3 /absolute/path/to/archive-root/inspiral-reference-20260906/archive-transport/restore.py --verify-only
python3 /absolute/path/to/archive-root/inspiral-reference-20260906/archive-transport/restore.py --destination /absolute/path/to/restored-reference
```

The helper requires Python 3.11 or later and only the standard library. It checks
original and compressed hashes, sizes and paths and refuses an existing output.
See [the transport guide](archive-transport/README.md). The immutable archive's
checksums apply to transport files; scientific hashes apply to restored files.

## Source and runtime

Six incremental inspiral bundles depend on two supporting bundles. Set absolute
paths below. These commands reconstruct the optimized source without a network:

```sh
set -eu
REF_ARCHIVE=/absolute/path/to/archive-root
REF_DATA=/absolute/path/to/restored-reference
REF_WORK=/absolute/path/to/new-reproduction
mkdir -p "$REF_WORK"
git clone --no-checkout "$REF_ARCHIVE/performance-fix/source.bundle" "$REF_WORK/source-v6"
git -C "$REF_WORK/source-v6" bundle verify "$REF_ARCHIVE/performance-fix/dependent-sources-v2.bundle"
git -C "$REF_WORK/source-v6" fetch "$REF_ARCHIVE/performance-fix/dependent-sources-v2.bundle" refs/heads/codex/torch-performance-fix-20260906
for REF_BUNDLE in inspiral-source.bundle inspiral-source-v2.bundle inspiral-source-v3.bundle inspiral-source-v4.bundle inspiral-source-v5.bundle inspiral-source-v6.bundle
do
    git -C "$REF_WORK/source-v6" bundle verify "$REF_DATA/$REF_BUNDLE"
    git -C "$REF_WORK/source-v6" fetch "$REF_DATA/$REF_BUNDLE" HEAD
done
git -C "$REF_WORK/source-v6" checkout --detach a4d77a6d1863c0515e8dace64c5609b63d40b51e
git -C "$REF_WORK/source-v6" status --porcelain
```

Use Linux and dependencies in [environment.json](environment.json) and
[unit-tests-v6.json](unit-tests-v6.json). Build native modules from this source
with the installation prerequisites, then install in a compatible environment:
`python -m pip install -e /path/to/source-v6`. MKL is required for the normal CPU
comparison and the qualified large Torch CPU IFFT; CUDA/Torch/Triton are needed
for GPU cases. Record actual dependencies, compiler and binary hashes, imported
module paths and clean tracked source. Rebuilt binaries may have different
hashes; different hardware/dependencies define a new campaign.

The source manifest distinguishes the four-file optimized change from unchanged
normal CPU implementation and native modules. The earlier measured source is
`837f38d493420043e45fb1ad210a0ccf68bacbaa`; it remains reconstructable from the
same bundle chain and must retain its own receipts and outputs.

## Data and qualification

Use `inputs/bank-compressed-1e5.hdf`, SHA-256
`161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916`.
Construction/compression scripts and `compression-1e5.json` retain provenance.
The earlier `bank-compressed.hdf` was rejected and is not the final input.

The external H1 strain file is `H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf`
(57,824,232 bytes), SHA-256
`580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865`.
It is the H1 4K GWF in the official
[GW170817 CLN release](https://dcc.ligo.org/LIGO-P1700349/public).
The source's `examples/inference/single/get.sh` also records the PyCBC mirror.
Verify the exact hash before selecting `H1:LOSC-STRAIN`.

Copy `config.json`, `run-case.py`, `qualify-inspiral.py`, comparator and profiling
helpers to a new campaign directory. Map `--frame-files` and `input_files` to
your frame and choose an available physical CPU core. Preserve all analysis
settings and thread limits. Original logical CPU 8 on host `len` may not be
appropriate elsewhere. Set `REF_PY` to a compatible absolute interpreter path.
Each case requires its own unused output directory:

```sh
"$REF_PY" -B "$REF_WORK/run-case.py" \
  --config "$REF_WORK/config.json" --source "$REF_WORK/source-v6" \
  --bank "$REF_DATA/inputs/bank-compressed-1e5.hdf" \
  --case qualify-normal-512 --mode qualify --scheme cpu:1 \
  --segment-length 512 --start-pad 112 --end-pad 16
```

Check that all 96 templates decompress, all expected template/segment pairs run,
and normal CPU FFTs use MKL with one thread. Run the 22-file unit command in
`unit-tests-v6.json` from your source directory with its recorded environment;
adapt `taskset` and `PYTHONPATH`. The case runner does not configure your shell.

`run-scientific-checks-v6.py` records fresh waveform/PSD, boundary and compressed-bank parity
commands. The three validator receipts identify 288, 36 and 576 checks.
`large-ifft-v6.json` and `large-ifft-v6-decision.json` retain the 108-case matrix
and precise dispatch acceptance. Preserve all original error budgets and fallback
constraints. The final source's 22-file unit run includes dispatch/thread checks.

Validators may inspect paths inside compression receipts. Create a separate
relocation receipt recording original SHA-256 and an explicit old-to-local path
map. Preserve every file hash and remap input-before, input-after and output path
keys consistently. Keep archived receipts unchanged.

## Measurement sequence

The baseline source performed reference-only tuning at 256, 512 and 1024-second
FFT lengths with three unprofiled normal CPU processes per length. Its
`precision5-*-plan.json` files and final report retain the decision. Only this grid
was tuned; fixed 112/16-second padding does not establish a global optimum.

The optimized source proves normal CPU implementation unchanged and reuses only
the selected 512-second geometry. `campaign-v6.py` binds the tuning and fresh
unit/IFFT/science prerequisites. The qualification ledger has three selected
backend runs; two additional normal CPU qualifications at 256/1024 seconds supply
science inputs. The measurement ledger records nine fresh unprofiled processes,
three each for `cpu:1`, `torch:cpu:1` and `torch:cuda:0`, plus seven profiles.
Keep the recorded ordering, source/input hashes, triggers and resource logs.

Use `compare-triggers.py` at its archived strict tolerances against normal CPU
qualification at the same geometry. Do not compare trigger identity across FFT
lengths because PSD discretization changes. Instrumented runs are separate from
capacity samples. Native profiles use `cycles:u`, 199 Hz and DWARF stacks; the
CUDA trace uses `run-case-profiling.py` and `profile-inspiral-torch.py` with the
recorded arguments. The controller records exports and profile summarization.

Original controllers pin original paths and prerequisites and refuse existing
outputs. Adapt a copy and review its new contract for a new campaign instead of
bypassing source/hash checks or relabeling old results.

## Regenerate this report

For the restored archive, regenerate tables and figures into an unused output:

```sh
"$REF_PY" -B "$REF_DATA/build-hotpath-report-v6.py" \
  --root "$REF_DATA" \
  --source-manifest-sha256 f62d2e580fed9ba514a3df724546dc63c306b6805289c2ce37896325bcb95b2b \
  --output-dir "$REF_WORK/rebuilt-report"
```

This validates recorded evidence without repeating numerical workloads. It needs
NumPy, h5py and Matplotlib. Figure bytes may vary with rendering dependencies.
The frozen baseline can separately be rebuilt with `build-reference-report-v4.py`.
A new campaign requires a reviewed report contract matching its own sources and
input hashes; do not change archived source identities to make a new run pass.
