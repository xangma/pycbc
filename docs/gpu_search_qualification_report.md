# PyCBC persistent GPU search: evidence scope

Updated 2026-09-12. The measurements below are historical fixture results. They do not certify the current source revision, all waveform providers, or production search decisions.

The retained [engine receipt](../artifacts/gpu_search_qualification_receipt.json) names dirty PyCBC revision `edff1e35e092c6373be757f7d95e9b8b3d16eec6` on `torch-pr11-performance-evidence`, host `len`, RTX 4090. Its source hashes identify selected files, but the receipt does not contain the complete dirty patch. Reproducing that execution requires recovering its source state. Existing local logs are useful historical records; they are not portable artifacts simply because their paths were named in this document.

## Historical engine measurements

The physical fixture used 384 reference TaylorF2 templates and five distinct colored-noise segments, each with N = 2,097,152 at 4096 Hz (512 seconds). It enabled symmetric clustering and 16-bin PowerChisq. This is a standalone SearchEngine fixture, separate from both the real-frame executable campaign and prepared live-filter API.

| Recorded interval | Historical value | Actual scope |
|---|---:|---|
| Cumulative synchronized submit/drain | 0.803 s; 2,390.9 template-segment pairs/s | Host submit, strain staging, correlation/IFFT, candidates, PowerChisq, retrieval and synchronization |
| Setup plus submit/drain | 17.85 s; 107.5 pairs/s | Engine/plans/allocation and preceding execution interval |
| Mixed preparation plus setup/execution subtotal | 45.43 s; 42.3 pairs/s | Adds waveform generation, PSD/noise creation, hashing and injection |
| Independent CPU candidate validation | 140.8 s | Outside the preceding subtotals |

The 45.43-second field was named `total_wall_time_sec`. It is a sum of selected intervals, excluding validation, some Python work, startup and output. It is **not process end-to-end wall time**. The mixed preparation interval of about 27.58 seconds is not a measurement of waveform generation alone. These proportions do not establish a fixed CPU fraction or a speedup ceiling for another workload.

The receipt records 1,920 template-segment CPU comparisons and 300 selected candidates, including absence checks for quiet pairs. Historical candidate checks reported zero sample-index difference, maximum **SNR-magnitude** difference 5.71e-6 and reduced-chi-square difference 0.00304. They did not compare the full complex SNR time series. These counts must not be combined with the distinct executable campaign's trigger counts.

The short synthetic tile sweep used N = 1,024. Its reported peak of 344,399 pairs/s at tile size 256 is a prepared short-transform result. Scaling it with N is an estimate, not a direct measurement at N = 2^19 or proof of optimality.

## Historical graphs and memory

The graph comparison used 512 templates, eight tiles of 64, and **N = 1,024 samples**. Each synchronized submit/drain covered all eight tiles. The historical eager/graph means were about 3.87/3.08 ms: a throughput factor of 1.257 corresponds to approximately 25.7% greater throughput and 20.4% lower latency. Capture covered correlation/IFFT; candidate selection, vetoes and other host/device work must remain visible in the submission timer.

The retained streaming receipt contains **50 measured blocks**, not 500. The approximately 3.12-ms mean and 3.17-ms maximum describe each complete 512-template submission. Allocated memory was 8.54 MiB at both endpoints, with a recorded peak of 11.77 MiB. `vram_leak_detected = false` means endpoint growth did not exceed the harness's 1-MiB threshold. Equal endpoints and a short run do not prove absence of leaks.

The former 1,024-template stream-amortization table is withdrawn as measured evidence: the cited `live_batch_latest.json` component receipt does not establish its cold/10/50/500-block rows. Actual persistent-service amortization requires separate process/preparation boundaries and observed stream repetitions.

## Executable and numerical boundaries

The historical compressed-bank `pycbc_inspiral` campaign reported 65.15 s for serial CPU and 17.29/16.95 s for CUDA batches 64/128. Those are separate executable results; they are not native TorchWave preparation measurements. See [the performance document](torch_performance.rst) and its frozen campaign references for their scope.

The campaign reported 2,203 pre-cut triggers, followed by 1,988 scalar-route versus 1,989 batched-route survivors. Such membership differences preclude strict reference-decision parity, even when common candidates have small numerical differences. A particular SIMD instruction, fused accumulation or reduction order was not established as the cause. The named 3995WX does not support the previous AVX-512 attribution. No finite fixture establishes that every astrophysical signal has unchanged identity, phase and arrival time.

Historical test counts describe their recorded source, environment and parametrization only. Reading an old JSON receipt does not execute its tests or qualify current code. Multirate, reduced-basis and screening prototypes require their own explicit experimental scope and correctness gates.

## New acquisition tools

[The engine benchmark](../tools/benchmarking/benchmark_gpu_search.py) writes version-2 receipts to a new default path. It records raw repetitions, actual geometry, graph scope, complex candidate-SNR errors and sampled allocated/reserved/peak memory. `--size` controls every synthetic suite; `--production-size` and `--sample-rate` define the separate physical fixture. The default stability acquisition has 500 measured blocks after warmup. Historical inspection always reports current production qualification as false.

[The provider verifier](../tools/verify_torchwave_gpu_search.py) checks every requested row using one fixed reference-generated injection and strain fixture. Its gates cover native dispatch, requested storage precision, unaligned complex and PSD-weighted waveform error, original norms, every complex CPU-reference SNR sample, and engine candidate/PowerChisq/NewSNR identities. Required failure returns exit status 1; unavailable requested devices return a skipped receipt and status 2. CPU is never silently substituted for CUDA. The full-series comparison isolates provider differences using canonical CPU filtering; device-engine comparisons cover selected candidates and vetoes. The engine currently stores complex64 independently of provider output precision.

[The provider campaign](../tools/bench_torchwave_pipeline.py) defaults to five fresh processes per provider and twenty prepared submit/drain samples per process. It records actual bank-generation-through-first-drain time and an outer launch-through-exit timer, including child imports, fixed-fixture construction and receipt writing. These are synthetic tool-process and provider-pipeline boundaries; they do not measure the live/offline executables. Diagnostic timings may be retained when scientific gates fail, with no equivalent-output speedup claim.

Receipts record imported module paths, both repository identities when available, HEAD-relative binary patches, relevant untracked source bytes, provider diagnostics, input hashes, runtime/thread details and explicit unmeasured conditions. External model assets outside the recorded source scope need a separate manifest. Keep qualification and benchmark outputs with their exact source identities; do not relabel a historical receipt as current evidence.

Example commands, run serially on the intended device:

```sh
python tools/verify_torchwave_gpu_search.py --device cuda:0 --batch-size 16 --output artifacts/torchwave_verify_cuda_v2.json
python tools/bench_torchwave_pipeline.py --device cuda:0 --batch-size 16 --cold-runs 5 --warm-samples 20 --output artifacts/torchwave_pipeline_cuda_v2.json
python tools/benchmarking/benchmark_gpu_search.py --size 131072 --num-templates 2 --tile-size 2 --iterations 20 --num-blocks 500 --output artifacts/engine_n131072_v2.json
```

These commands are acquisition instructions, not assertions that the measurements have run. Current results must be reported from the newly produced receipts and logs.
