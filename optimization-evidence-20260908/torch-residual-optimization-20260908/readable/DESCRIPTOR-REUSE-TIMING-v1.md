# Descriptor reuse timing result

Bounded descriptor reuse improved both standard CPU and CUDA full-process time on the qualified 384-template workload. All scientific and lifecycle gates passed, and independent raw replay and summary cross-check passed. This is an artifact-only experiment; the qualified PyCBC source remains unchanged at `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`.

| Route | Original median (range), s | Cached median (range), s | Reduction from median times |
|---|---:|---:|---:|
| Standard CPU | 66.849562 (66.791962–67.038664) | 65.370059 (65.189339–65.505714) | 2.213184% |
| CUDA | 21.976242 (21.893677–22.051733) | 20.291608 (20.213787–20.307819) | 7.665707% |

All four within-family pairs improved, and each cached range lies below its original range. The median paired wall reductions are 2.252010% for standard CPU and 7.664565% for CUDA; these differ from ratios of independent medians above. Cached CUDA has a median paired full-process speedup of 3.222772× (range 3.218219–3.229945×) over contemporaneous cached standard CPU, and 3.298929× (3.290915–3.305531×) over contemporaneous original standard CPU.

Secondary setup medians fell from 12.107852 to 10.549228 s for standard CPU and 10.342075 to 8.634298 s for CUDA. The primary full-wall measurement includes interpreter execution, common cache construction, candidate installation, descriptor release, wrapper restoration and process exit. The outside-internal interval is not wholly attributed to imports.

The frozen schedule ran four fresh serial repetitions of all four cells. With A=standard original, B=standard cached, C=CUDA original and D=CUDA cached, the orders were A-B-D-C, B-C-A-D, C-D-B-A and D-A-C-B. Each cell occupied every position once. No samples or warmups were discarded. Filesystem and library state was warm from preceding qualification. All runs used CPU core 8, one numerical thread, the same 384-template bank, five segments and 1,904 valid detector seconds.

The fast helper retains the original function-API native configuration and reuses only three supported out-of-place single-precision descriptor configurations. Process/thread ownership, the full immutable descriptor key, supported-call fallback, bounded capacity, failure cleanup and restoration remain enforced. Per-call diagnostic timers, counters and histories are absent. Each cached process owned three descriptors and freed each exactly once at teardown; original processes used the untouched original functions and owned no cached descriptors. Timed cache hit counts were not recorded.

A separate fresh native gate preceded four scientific qualifications, all before the first timing worker. Nine native calls used 18 distinct live pointers, verified full-output byte equality and input preservation, and exercised coexisting class FFT and promoted Torch plans. The native gate externally observed three creations and six hits. All four qualifications verified captured strain hashes, actual PSD arrays and 1,920 successful matched-filter IFFTs. The complete campaign passed 51 unmodified scientific comparisons, including 38 exact comparisons of all 18 H1 scientific datasets. Timed runs retained exact persisted scientific outputs and work checks; they did not recapture strain or PSD arrays. Native array equality was checked in process; the arrays themselves are not archived.

Controller process group 2936779 and all 21 workers exited. An independent terminal audit reacquired the benchmark lock and rechecked source, native libraries, frozen inputs and references. The 317-file, 29,368,320-byte archive was acquired and byte verified. Owner and peer replay agreed on all 16 sample records, 80 raw values in 20 metric summaries, 48 paired values in 12 paired summaries, native results and all 20 executable lifecycle receipts.

Four repetitions on one host and one finite workload support this measured improvement, without establishing statistical significance, general workload performance, whole-host isolation or saturated multicore behavior. Torch CPU parity remains unmet from the preceding campaign; this experiment contains no Torch CPU timing. No production integration, PR, push or publication was performed. The positive result supports a separately reviewed production design for the bounded reuse mechanism.

| Evidence | SHA256 |
|---|---|
| [descriptor-reuse-timing-v1/manifest.json](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1/manifest.json) | `807406093bee08fa621eeb4c327a8c1547f66c85224db9c3d33f11e5156f244b` |
| [descriptor-reuse-timing-v1-results.tar](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1-results.tar) | `52c50829795096da3683c6b678deb5ddfc69e6f8a90d1ab8b106db84ebee18f3` |
| [acquired-descriptor-reuse-timing-v1/results-manifest.json](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/acquired-descriptor-reuse-timing-v1/results-manifest.json) | `21dd1a3f2112aba4664bea0b619af0c31fd7c947c6f7d1443658a1645531bb7f` |
| [descriptor-reuse-timing-v1-summary.json](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1-summary.json) | `71dc531bc1e08c8c93969e118276abbd6458f9cb9de76c38d9f8afd92b6463e0` |
| [peer-reviews/fast-reuse-results-v1-review.json](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/peer-reviews/fast-reuse-results-v1-review.json) | `d7e9efff7448b695d8ab316b8ed4ac028b24dd992f96319eec6b327bfda48221` |
| [peer-reviews/fast-reuse-summary-v1-crosscheck.json](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/peer-reviews/fast-reuse-summary-v1-crosscheck.json) | `5fb5cc0d90be9251d61c6b74ac053def3c186f44af73d06d8a5081b42e83ce9d` |
| [descriptor-reuse-timing-v1-samples.csv](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1-samples.csv) | `3032cf12c92ea877abb2441f5c4b92112a0e260d8efb73a9c14c824a6438e5f5` |
