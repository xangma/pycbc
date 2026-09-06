# Final Torch versus original PyCBC

This campaign compares untouched upstream `40e94792b3edf59f39b18b65102b28a4f74433a7`
with final Torch `a4d77a6d1863c0515e8dace64c5609b63d40b51e`. It contains nine
unprofiled processes (three per backend, rotating order) and six separate
cProfile/perf processes. All use one host core, the same compressed 96-template
BNS/NSBH bank, 512-second segments, 112/16-second padding and 1904 valid detector
seconds. CUDA additionally uses one RTX 4090. Complete commands and input hashes
are in each run's `receipt.json`; `environment.json` records dependencies.

## Results and scientific scope

Median wall seconds: original CPU 29.299898, Torch CPU 44.415080, Torch CUDA
20.200899. Capacity is `96 * 1904 / wall_seconds` per allocated host core.
Startup and output writing are included. Profiles and the failed CUDA-selector
launch are excluded from these medians. `plan.json` records the initial invalid
selector; `corrected-plan.json` and `status.json` identify all completed cases.

**The original-versus-Torch strict trigger comparisons fail.** Original has 477
triggers, each Torch candidate 475, with 469 matched identities. Every matched
chi-square value exceeds the existing tolerance; some SNR and phase values also
fail. See `original-trigger-comparison.json`. The stack includes shared numerical
fixes, so its corrected CPU baseline must not be described as original upstream.
All six fresh Torch timing outputs pass the same budgets against that corrected
CPU reference, preserved in `corrected-reference/`. Its receipt identifies its
actual earlier run and source. See `final-trigger-validation.json`.

The prior independent scientific checks and exact final source bundles are in
[`inspiral-reference-20260906`](../inspiral-reference-20260906/REPRODUCE.md).
That guide identifies the compressed bank (SHA-256
`161820754117cd51b24a7bb672ea456cf7b2a3808e69690c23235d024b1a2916`) and external
GW170817 H1 frame (SHA-256
`580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865`).
Its segment tuning used corrected CPU code; original-upstream segment tuning
and a padding performance sweep remain open.

## Verify and regenerate the summary

From this directory, with Python 3.11+ and h5py installed:

```sh
sha256sum -c SHA256SUMS
python summarize.py .
python compare-triggers.py corrected-reference/triggers.hdf runs/torch-*-r[123]/triggers.hdf
python compare-triggers.py runs/original-cpu-r1/triggers.hdf runs/torch-cpu-r1/triggers.hdf runs/torch-cuda-r1/triggers.hdf
```

The final command is expected to exit 1: the original comparison fails. The
summary builder verifies source cleanliness, thread limits, CPU affinity,
geometry, input/output hashes, all 15 successful receipts, the six corrected
reference comparisons, and compressed profile integrity. It regenerates
`summary.json` and `timings.csv`; run it in a disposable archive checkout.

`profile-transport.json` records original and compressed native-profile hashes.
Restore each `runs/*-perf/perf.data.gz` with `gzip -dc` into an unused path to
inspect it with `perf`. Raw profiles remain unchanged on the measured host.
The archive includes text reports and cProfile pstats. Full-executable profiles
contain startup and I/O; they do not establish steady-state filtering costs.

## Run a new campaign

Restore the final source using the sibling guide's bundle chain. Its Git history
also contains the original upstream commit; create a separate clean checkout
at that revision. Build with compatible dependencies. This campaign reused
eleven native extensions only after checking all native sources and native build
definitions were identical between the two revisions. `source.json` records
those checks, binary hashes and the original import probe.

Copy `run-case.py` and `config.json` to a new directory, update frame/source/bank
paths and choose an available CPU core. Preserve hashes and all scientific
options. Run `--mode timing`, three fresh processes per scheme, with unique case
names and rotating order:

```sh
python run-case.py --config config.json --case original-cpu-r1 --mode timing --scheme cpu:1 --segment-length 512 --source ORIGINAL_CHECKOUT --bank BANK_PATH
python run-case.py --config config.json --case torch-cpu-r1 --mode timing --scheme torch:cpu:1 --segment-length 512 --source FINAL_CHECKOUT --bank BANK_PATH
python run-case.py --config config.json --case torch-cuda-r1 --mode timing --scheme torch:cuda:0 --segment-length 512 --source FINAL_CHECKOUT --bank BANK_PATH
```

Use separate `--mode cprofile` and `--mode perf` cases. The runner supplies the
recorded padding and one-thread environment. Adapt controllers only in a copy:
their host paths are provenance, not portable defaults. Compare triggers before
interpreting a new timing result as equivalent science.
