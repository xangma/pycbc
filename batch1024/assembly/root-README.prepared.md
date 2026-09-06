**Batch sizes through 1024:** the [new supplement](batch1024/README.md) contains completed live, waveform and Triton measurements, separate dispatch probes, correctness receipts for all three clean sources, and eight simplified documentation figures. [Reproduction instructions](batch1024/REPRODUCE.md) and [standalone figure manifest](batch1024/documentation/render-manifest.json) accompany it. The historical campaign below retains its original sources and files.

# PyCBC Torch benchmarks — 6 September 2026

**TaylorF2 Triton supplement:** [60 new qualified workers](triton/README.md) show **1.109–2.972×** higher warm public-batch throughput than ordinary Torch CUDA at the same candidate revision. Cold calls cost about 2 seconds versus 0.4 seconds. The opt-in route is included in PR #11 and the updated stack. [Plots and full tables](triton/report/report.md) and [exact-head test receipts](triton/restack-validation.json) accompany the supplement.

The baseline campaign below remains pinned to its original three commits; it was not rerun after adding Triton. Its previously unsupported Triton requests remain part of that historical record.

Fresh measurements of the published assembled Torch stack and its two optional follow-ups, on `len`: AMD Threadripper PRO 3995WX, NVIDIA RTX 4090, Python 3.11.9, Torch 2.13.0+cu130, CUDA 13.0. Processes ran sequentially with affinity to CPUs 8–11. The host was shared; resource snapshots are retained with the runs.

| Measured source | Exact commit |
|---|---|
| Assembled main stack, PR #15 | `607bce53ead14f12af32552a5b2441d3bc667267` |
| Optional FFT follow-up, PR #16 | `e6073eaf1a89cfed69af53707f52321eadf129f1` |
| Optional CPU follow-up, PR #17 | `1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c` |

These are measurements of the assembled heads, not independent measurements of every intermediate PR. All three checkouts remained clean, and all 33 recorded native-extension hashes were unchanged after the campaign.

## Results

- **Synthetic live filtering:** at batch 32, default Torch CUDA achieved **41,316 template-block evaluations/s**, versus 1,192 for standard CPU with one thread and 2,728 with four threads: **34.7×** and **15.1×**, respectively. All Torch CPU live configurations were slower than their standard CPU baselines. [Full live table and limitations](report/live-summary.md).
- **TaylorF2 waveforms:** at batch 32 and 4097 frequency bins, Torch CUDA batch achieved **2,549 waveforms/s**, **1.48×** the one-thread CPU/LAL baseline. Torch CPU batch with four threads achieved **2,077 waveforms/s**, **1.21×** its corresponding CPU/LAL baseline. Most other measured Torch waveform routes were slower than CPU/LAL. [Full waveform table](report/waveform-summary.md).
- **Public inference calls:** Torch was slower for this workload. GaussianNoise delivered **328 evaluations/s** on CUDA versus **631** on standard CPU with one thread; Relative delivered **190** versus **2,271**. These timings include parameter update, waveform generation, detector projection and the returned host likelihood scalar. [Full inference table and separate cold phases](report/inference-summary.md).
- **Optional FFT planning:** for a length-131072 complex64 inverse FFT with one thread, MEASURE planning reduced steady execution from **0.737 ms to 0.571 ms**, while cold planning increased from **42 ms to 2,417 ms**. Importing cached wisdom took **76 ms**, with **0.569 ms** execution. The four-thread Torch fallback was faster at **0.297 ms**. [Full FFT table](report/fft-summary.md).
- **Optional CPU live path:** at batch 32, the optional native configuration was **1.90×** the main native configuration with one thread and **1.45×** with four threads. This comparison includes an additional enabled CPU peak kernel; it is not a comparison with identical flags. [Results](report/live-summary.md) and [actual dispatch probes](report/dispatch-summary.md).

![Main live filtering](report/main-live.png)

![TaylorF2 waveform generation](report/waveform.png)

![Public inference calls](report/inference.png)

Additional figures: [FFT planning and execution](report/fft.png), [optional CPU live filtering](report/optional-cpu.png), [inference cold phases](report/inference-cold.png). Editable SVG versions accompany every PNG.

## Validation and scope

All **126 live**, **18 FFT**, **72 supported waveform** and **30 inference** timed workers passed their recorded qualification gates. All **20 smoke checks** and **20 untimed native-dispatch probes** passed. Waveform records additionally preserve **90 unsupported requests**, covering single precision and the unavailable public TaylorF2 Triton route at these exact commits. They are not timing failures and receive no speedup claims.

Each timing cell has three fresh worker processes. Reported centers are medians of process medians; plotted ranges show the three observed process medians, not confidence intervals. Samples within a worker are not independent replicates. Cold work is reported separately.

Live measurements use the public synthetic `LiveBatchMatchedFilter.process_data` driver, three blocks, complex64, and FFT length 131072. They exclude waveform generation, PSD estimation, bank/frame I/O and startup; chi-square is disabled and sine-Gaussian postprocessing is stubbed. Live parity checks final triggers and aggregate SNR norms, not pointwise SNR arrays. The previously identified asynchronous peak-copy path is disabled throughout; this campaign does not qualify or repair it.

Waveforms retain both complex128 polarizations on the selected device and check complete outputs and metadata against native scalar and CPU/LAL references. Inference uses synthetic H1/L1 data, 32 seconds at 2048 Hz, and checks both likelihood and likelihood ratio at twelve parameter points per worker. Full sampler runs, production searches and MPS are outside this campaign.

## Evidence and reproduction

[Reproduction instructions](REPRODUCE.md) explain exact source checkout, dependency setup, commands, timing boundaries and plot regeneration. Raw outputs, process receipts, telemetry, logs, harnesses, packages and source identities are included. `sealed-manifest.json` records SHA256 and size for 566 original remote files, verified after download. `SHA256SUMS` covers the complete published artifact. Initial environment-setup diagnostics are retained separately from successful benchmark results.
