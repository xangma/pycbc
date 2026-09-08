# Reproduce the unchanged-baseline/final-proposal comparison

The measured sources are unchanged PyCBC `40e94792b3edf59f39b18b65102b28a4f74433a7`
and the main Torch proposal `123e1fb3ef1b338cada636e71c3e9c7987002402` in
<https://github.com/xangma/pycbc>. The proposed revision contains the complete main
runtime; the optional FFT and CPU follow-up PRs are outside this comparison.
Subsequent result-documentation commits must preserve these runtime bytes.

## Verify the saved result without running PyCBC

From this archive directory, verify the sealed inventory with `python verify-archive.py`.
Use a Python environment containing NumPy and h5py to recompute the scientific
comparisons and timing table:

```sh
python verify-results.py --evidence-root acquisition --output /tmp/baseline-final-verification.json
```

This does not execute a benchmark. It reads raw trigger HDF files, full PSD
arrays, qualifications and worker receipts. All source/executable provenance
substitutions are explicit. It does not change scientific arrays or tolerances.
Full-array PSD failures, excluded-bin differences and original-baseline trigger
failures remain distinct from successful completed-work and repeat checks.

## Sources, native builds and dependencies

Create a new directory; never run acquisition inside the immutable archive.
Clone the repository and create separate detached worktrees at the two commits:

```sh
git clone --no-checkout https://github.com/xangma/pycbc.git repo
git -C repo worktree add --detach ../original 40e94792b3edf59f39b18b65102b28a4f74433a7
git -C repo worktree add --detach ../proposed 123e1fb3ef1b338cada636e71c3e9c7987002402
```

Build each source independently with the same compatible interpreter and
dependencies using `python setup.py build_ext --inplace` from each worktree.
The acquisition used a pre-existing common Python 3.11.9 environment; it did not
install one revision over the other. `acquisition/setup-remote.py`, both build
logs and `*-build.json` record the exact executed preparation. Source pins cover
tracked content, generated version files and native extension hashes; worker
receipts also verify every imported PyCBC module's source and hash. The
`committed-source-verification.json` checks recorded tracked-file contents
against the actual two Git commits, including symlink target contents.
`committed-source-proof.json` additionally binds this check to the acquired
`source-pins.json` SHA256. With both commits available locally, repeat it using
`python verify-source-commits.py --repository /path/to/repo --output /tmp/source-proof.json`.

`acquisition/dependencies.json` inventories distribution metadata, which can
include another installation in the reused environment. Its name-keyed dictionary
keeps the later Torch 2.1.1 entry from an inherited Conda installation, while
imports select Torch 2.13.0+cu130 earlier on the search path. The supplemental
`acquisition/torch-environment-diagnostic.json` and inspection script record a
read-only check after completion: both metadata directories, selected files
against wheel RECORD hashes, and the imported paths. The independent verifier
explicitly reconciles the untouched inventory with all runtime version receipts;
the post-run check does not retrospectively hash loaded worker Torch binaries.
Runtime receipts identify
the actual loaded numerical libraries and versions: Torch `2.13.0+cu130`, CUDA
13.0, MKL 2020.0.4 and OpenBLAS 0.3.25. Rebuilds need not be binary-identical;
record fresh build and environment identities with any reproduction.

## Frozen scientific inputs

Restore `bank-compressed.hdf` from this repository's
`reference-campaign-20260907/inputs` directory. Its SHA256 is
`26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f`.
It contains 384 distinct compressed templates, 256 BNS and 128 NSBH.

Obtain `H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf` from the
[GW170817 CLN release](https://dcc.ligo.org/LIGO-P1700349/public), or the mirror
recorded by the measured source's `examples/inference/single/get.sh`.
The external frame is 57,824,232 bytes with SHA256
`580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865`.
The archived configuration uses `H1:LOSC-STRAIN` and valid GPS interval
1187007160–1187009064, totaling 1904 unique seconds. The bank is a deterministic
performance set, not a coverage-qualified search bank.

## Acquisition and timing

Copy the `.py` acquisition helpers and `config.json` from `acquisition/` into
the new campaign directory containing the `original` and `proposed` worktrees.
The scripts intentionally contain acquisition-host paths and CPU topology
assertions. Before running, relocate the frame and bank paths in `config.json`
(including `common_args`, `input_files`, `bank` and `input_pins`); set the shared
coordination-lock path consistently in `campaign.py` and `checked-inspiral.py`.
Create that lock file. Configure the selected core and its actual SMT sibling
in both the command and runtime assertions, preserving one numerical thread.
If preparation is repeated using `setup-remote.py`, adapt its local clone and
bundle import to the clean source acquisition above. Record complete build
receipts, `setup-status.json` and `dependencies.json` as its successful build
path does. Record all adapted scripts before starting fresh acquisition.

Run `python -B campaign.py`. It acquires the coordination lock and first runs
four separate instrumented qualifications. Each checks 384 compressed templates
without regeneration, five segments, 1920 template/segment pairs, 2097152 FFT
samples, zero analysis gaps/overlaps and identical valid HDF intervals.

For this acquisition, `campaign.py` stopped after qualification because full
PSD arrays differed between proposed backends below the filter's 30 Hz cutoff.
The initial `status.json` and `campaign.log` remain unchanged. The separately
reviewed `resume-timings.py` and `continuation-policy.json` document a decision
made **after** observing this failure: collect descriptive timings only after
the proposed trigger comparison passes, conditioned strain and geometry match
exactly, and PSD bins actually used by the filter match exactly. The original
full-PSD failures are not reclassified as passes. There is no automatic
continuation after any unrelated execution, source or scientific-check failure.

The continuation reused the same four qualification outputs and unchanged
worker implementation. It launched four fresh unprofiled processes per arm in
orders ABCD, BCDA, CDAB and DABC (A = original CPU; B = proposed CPU; C = Torch
CPU; D = Torch CUDA). Every arm therefore occupied each position once. All 16
trigger outputs must pass against their own qualification before completion.

The timed child command is `/usr/bin/time -v -o time.txt taskset -c 8 python
checked-inspiral.py ... -- <source>/bin/pycbc_inspiral ...`; every expanded
argument is in the worker's `receipt.json`. The comparable clock starts before
child process launch and stops after child exit. It includes interpreter and
imports, frame loading, PSD and bank setup, filtering, vetoes, clustering, HDF
output, and the common wrapper's runtime/hash checks. Parent pre/post source
hashing and scientific HDF comparison are outside that clock. Parent host
sampling also runs concurrently with the child.
Qualification clocks and `/usr/bin/time` measurements are retained but do not
enter the headline medians. Filesystem caches were not flushed.

The host was shared `len`, AMD Ryzen Threadripper PRO 3995WX, CPU 8 with SMT
sibling 72. CUDA additionally used RTX 4090 GPU 0. All observed native and Torch
intra/inter-op pools were limited to one. Graph capture was disabled. A lock
coordinates these benchmark runners; affinity and the lock do not reserve the
host, sibling CPU or GPU against other services. `host-samples.jsonl` and GPU
before/during/after observations retain concurrent load. Four observed ranges
are not confidence intervals or sustained/full-machine capacity measurements.

For each arm, recompute the median and observed minimum/maximum from its four
`elapsed_wall_seconds` values. Work is `384 * 1904 = 731136` template-seconds;
rate is that work divided by median wall seconds. Preserve all samples, all
failed scientific verdicts and any acquisition-policy amendments in the report.
