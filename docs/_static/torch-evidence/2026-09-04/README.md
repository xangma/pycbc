# Historical len benchmark receipt

These files are unchanged copies from the completed 2026-09-04 campaign at
**`ec12863ee897b529960d8579526715d8d54810e9`**. They do not validate the commit
that publishes them or any later stack revision. The original location was
`len:/home/xangma/pycbc-torch-finish-20260904/performance`.

The two JSON artifacts contain all 63 timed cells and individual samples.
`SHA256SUMS` records their original file hashes; each JSON also carries its
benchmark content seal. `environment-before.txt` records the benchmark and
native-extension hashes. The before/after environment snapshots, benchmark
logs, exact historical launcher, and six untimed native-correlation probe logs
are retained alongside them. The launcher contains the original absolute
paths; it is a command record, not a portable installation script.

See [review.md](review.md) for the workload, replicate medians and limitations.
The benchmark uses CPython 3.11.9, NumPy 1.26.4, Torch 2.13.0+cu130, an RTX 4090
and four pinned CPU cores on len. All 63 timed cells passed the benchmark's
trigger/aggregate-norm parity checks. Six separate probes confirmed batch
correlation, not every optional native component or every timed invocation.

Default Torch CUDA was the fastest measured route. Default Torch CPU and the
requested CPU-native configuration were slower than standard CPU. Requested
CUDA-native was slower than default Torch CUDA at B8 and B32. This compares
routes at one revision; it does not measure the cleanup's performance effect.
The GPU was shared with workstation processes. Aggregate output norms do not
establish pointwise parity, and this harness excludes full CLI/I/O, waveform
generation and chi-square. CUDA 12.6, MPS, other Torch versions and subsequent
revisions are unqualified by this record.

To verify the copied raw files from this directory on a qualified machine:

```sh
sha256sum -c SHA256SUMS
```

The separate scientific corpus, full test logs and qlty report are not copied
into this selected performance bundle. Do not infer their results from these
files. No plots or hosted-CI results are claimed by this receipt.
