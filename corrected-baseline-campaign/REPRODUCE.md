# Reproduce the corrected-baseline comparison

The measured commits are standalone corrected CPU `66789ac4a7468094b0cc3ca1498a1de67e0311f6` and rebuilt main `f582b6fd250d0b82612492979e01e645d5c07afc` in [xangma/pycbc](https://github.com/xangma/pycbc). Optional PRs #16/#17 are outside this experiment. Later formatting and documentation changes have a separate source mapping; the recorded worker source identities remain unchanged.

## Inspect and verify the saved evidence

The archive contains all four qualification HDF outputs, full qualification PSD arrays, sixteen timed HDF outputs, commands, receipts, runtime checks and host observations. Verification needs Python, NumPy and h5py, but does not execute PyCBC or a benchmark. The independent verifier in `verification/` recomputes evidence checks, trigger comparisons, full-array PSD results and timing statistics. The final report is `verification/final-report.md`; earlier qualification reports in that folder are provisional snapshots. From this directory run:

```sh
python verification/verify-results.py --evidence-root acquisition --output verification/recomputed.json
```

The output must be a new file inside the verifier folder. Exit 0 means internally consistent evidence, with scientific failure reported separately. A copy of the archive is suitable for new output files; retain the published original unchanged.

From this directory, independently verify the measured source against a local clone containing both commits:

```sh
python verify-source-commits.py --repository /path/to/pycbc --output /tmp/corrected-source-proof.json
```

This compares every acquired tracked path and its bytes with Git, resolving tracked symlink targets to match the acquisition hash convention. It binds the result to `acquisition/source-pins.json`. Native binary and generated-version identities are recorded separately; Git cannot verify those untracked bytes.

## Restore inputs and source

The frozen `inputs/bank-compressed.hdf` has SHA256 `26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f`. It contains 384 distinct compressed templates (256 BNS and 128 NSBH). This is a deterministic performance set, not a coverage-qualified search bank.

Obtain `H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf` using the mirror recorded in the measured commit's [download script](https://github.com/xangma/pycbc/blob/66789ac4a7468094b0cc3ca1498a1de67e0311f6/examples/inference/single/get.sh). Require SHA256 `580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865` before acquisition. The 57,824,232-byte external frame is not included in this archive.

Create a new campaign directory. Preserve the immutable evidence directory. In the new directory:

```sh
git clone --no-checkout https://github.com/xangma/pycbc.git repo
git -C repo fetch /path/to/archive/corrected-baseline-campaign/sources.bundle refs/heads/codex/cpu-precision-corrections-20260908 refs/heads/codex/cpu-base-restack-20260908-pr15
git -C repo worktree add --detach ../corrected 66789ac4a7468094b0cc3ca1498a1de67e0311f6
git -C repo worktree add --detach ../proposed f582b6fd250d0b82612492979e01e645d5c07afc
```

The included `sources.bundle` preserves both measured heads even after PR branches move. Its SHA256 is `2847556ef9db36b3044354542a2b15040b77ac577144ec6c1f373de34f1e51fe`, matching the acquired setup receipt. It requires the frozen base `40e94792b3edf59f39b18b65102b28a4f74433a7`, available through the repository's `torch-stack-base` branch.

Build each source using the same compatible interpreter and dependency environment, with `python setup.py build_ext --inplace` in each worktree. Record fresh build receipts and generated version hashes. A reproduction need not have identical binary hashes, but it must pin and report its own builds consistently.

The recorded acquisition reused existing native extensions rather than compiling again. `acquisition/setup-remote.py` first compared native source files, build inputs and every binary hash against the frozen original/proposed build receipts, then copied the eleven verified extensions into separate clean checkouts. `corrected-build.json`, `proposed-build.json`, `setup-status.json`, source pins and per-worker imports record this provenance. Running that host-specific preparation script elsewhere requires adapting its local clone, bundle and existing-build paths. Do not claim a fresh compilation for the recorded result.

## Environment and acquisition

The experiment used a shared Python 3.11.9 environment. Distribution metadata are in `acquisition/dependencies.json`; each worker records the libraries actually loaded. The inherited name-keyed metadata inventory says Torch 2.1.1, while all Torch workers report imported Torch 2.13.0+cu130. The post-acquisition diagnostic records both distributions and checks selected imported files against wheel RECORD hashes. It explains the metadata selection without retrospectively identifying every Torch binary loaded by earlier workers. The original metadata inventory is preserved.

Copy the acquisition `.py` helpers and `config.json` into the new campaign directory containing `corrected/` and `proposed/`. These scripts intentionally preserve the original host paths and topology assertions. Adapt the frame/bank paths everywhere in `common_args`, `input_files`, `bank` and `input_pins`. Keep the input bytes and scientific arguments fixed. Use one common absolute input path across arms.

Set a shared coordination-lock path consistently in the campaign and checker, create the lock file, and configure an actual CPU/core and its SMT sibling in all command and runtime assertions. Preserve one numerical thread and record every adapted script before running. Prepare the build, setup and dependency receipts required by the campaign, using `setup-remote.py` as the recorded schema. Start with empty output directories and no previous status or timing files.

Run `python -B campaign.py`. It first performs four fresh executable qualifications: standalone corrected CPU, restacked normal CPU, Torch CPU and Torch CUDA. Each must process 384 compressed templates without regeneration, five segments, 1920 template/segment pairs and 2097152 FFT samples, with the same valid HDF intervals and no analysis gaps or overlaps. The valid interval is GPS 1187007160–1187009064: 1904 unique H1 seconds.

The recorded strict controller stopped after qualification because full PSD arrays failed the unchanged budget below 30 Hz. `acquisition/status.json` and `controller.log` retain that failure. The separate `resume-timings.py` and `continuation-policy.json` disclose a decision made after observing the failure: collect descriptive timings only after all five trigger comparisons, exact conditioned strain and geometry, and exact PSD bins actually used by filtering pass. Full-array failure remains failure. Review any new failure before choosing whether to use that continuation in a reproduction; it is not a blanket override for failed execution, inputs, source checks or scientific checks.

The continuation launches four fresh unprofiled processes per arm in rotating orders ABCD, BCDA, CDAB and DABC (A corrected CPU; B restacked CPU; C Torch CPU; D Torch CUDA). Each timed trigger output must pass against its own separate qualification. No samples are discarded.

## Boundary, budgets and interpretation

Every expanded command is saved in `acquisition/runs/<case>/receipt.json`. Timing starts before launching `/usr/bin/time -v ... taskset -c 8 python checked-inspiral.py ...` and ends after child exit following HDF output and runtime/native verification. It includes interpreter/imports, I/O, conditioning, PSD and bank loading, filtering, vetoes, clustering, output and the common checker. Parent source hashing and post-run HDF comparison are outside the clock. Parent host sampling runs concurrently with the child. Qualification times and GNU time observations are retained but excluded from the medians.

The trigger comparator uses `1e-4` relative and `1e-5` absolute budgets; sigmasq uses `1e-5` relative with zero absolute floor; phase uses `1e-4` radians. Trigger identities, sample ticks and degrees of freedom must match exactly. PSDs use `1e-4` relative with zero absolute floor and exact shape, dtype and nonfinite masks. Filtering uses bins `[15360:1048576]`, beginning at 30 Hz. Source/executable provenance substitutions are explicit and independently pinned. These budgets are numerical comparison criteria, not a physical waveform-accuracy guarantee.

The shared host `len` is an AMD Ryzen Threadripper PRO 3995WX. Every arm uses CPU 8 with SMT sibling 72 and one numerical thread; CUDA additionally uses RTX 4090 GPU 0. Native and Torch pool counts are recorded. Graph capture is disabled. Neither the lock nor affinity reserves the host, sibling or GPU against other work. Per-process host and GPU observations are archived. Filesystem caches were not flushed.

Compute each median and observed minimum/maximum from its four receipt `elapsed_wall_seconds` values. Work is `384 * 1904 = 731136` template-seconds; the displayed rate is work divided by median wall time. Four observed ranges are not confidence intervals or sustained/full-machine capacity estimates. Retain the full-PSD failure and the disclosed timing-policy amendment when reporting these descriptive costs.
