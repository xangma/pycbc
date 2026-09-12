# Measured results on len

Generated from receipts by `summarize_results.py`. All times below are milliseconds unless marked otherwise.

## Provider preparation and prepared filtering

B=16, N=2048, 1024 Hz, reference-generated fixed strain. Five fresh processes per route, each with one first drain and twenty prepared drains. Every actual drain was checked outside its timer. Medians below are across workers; the prepared column is the median of worker medians.

| Execution | Provider | Generation ready | Engine preparation | Generation through first drain | Prepared submit/drain |
|---|---|---:|---:|---:|---:|
| cpu1 | reference | 25.895 | 7.152 | 91.635 | 45.752 |
| cpu1 | torchwave | 55.216 | 2.079 | 113.776 | 45.641 |
| cpu4 | reference | 26.061 | 6.826 | 59.815 | 18.926 |
| cpu4 | torchwave | 54.885 | 2.016 | 82.195 | 18.511 |
| cuda | reference | 26.178 | 17.600 | 391.273 | 2.613 |
| cuda | torchwave | 337.857 | 4.565 | 456.001 | 2.627 |

Generation includes bank construction in this experiment. Prepared filtering excludes generation and setup. Outer process wall times also include imports, fixture construction, independent qualification, provenance and output; they are retained in JSON and are not operational latency.

## Warm generation only

Prepared bank/arguments to a completed device tensor, five rotated samples per route; all rows pass unaligned complex-L2, norm, exact support, dtype and device gates before and after timing. PyCBC native prebuilds argument tensors and generates both polarizations; reference/TorchWave include FilterBank metadata. These are explicit API cost differences.

| Execution | N | Reference | PyCBC native TaylorF2 | TorchWave |
|---|---:|---:|---:|---:|
| cpu1 | 2048 | 8.832 | 4.743 | 7.325 |
| cpu1 | 131072 | 44.128 | 142.302 | 232.190 |
| cpu4 | 2048 | 8.914 | 4.645 | 7.253 |
| cpu4 | 131072 | 44.333 | 58.905 | 90.512 |
| cuda | 2048 | 8.868 | 11.226 | 8.821 |
| cuda | 131072 | 45.400 | 11.810 | 9.371 |

## Actual offline executable

One persistent worker per route, two identical injected-frame shards, four templates, N=65536 at 1024 Hz. Campaign wall time is seconds; each shard is elapsed executable time within that worker. A single campaign is dispatch/reuse evidence, not a statistically qualified speed comparison.

| Route | Campaign seconds | First shard seconds | Second shard seconds | Second-shard batch hits |
|---|---:|---:|---:|---:|
| native-cuda | 6.245 | 4.942 | 0.289 | 1 |
| reference-cuda | 6.232 | 4.918 | 0.289 | 0 |
| reference-cpu | 5.729 | 4.539 | 0.298 | 0 |

Live: two MPI ranks completed the two-detector 32-second noise fixture in 6.627 seconds. All seven dispatch, device and HDF checks passed. Its SNR threshold is 1e6; this checks the empty-trigger executable path, not detection efficiency or live trigger parity.

## Graph and streaming diagnostics

N=131072, templates=3, tile=2, 20 repetitions. Mean synchronized eager/graph submit-drain: 1.304/1.048 ms. Graph coverage is correlation/IFFT only. This normalized synthetic fixture has no selected candidates; positive-candidate qualification is separate.

Streaming measured 500 blocks after warmup, allocated memory 15.2617 to 15.2617 MiB, peak 17.5117 MiB. Raw latency and memory samples are retained; unchanged endpoints do not prove absence of leaks.

## Identical-input veto comparison

N=131072, eight candidate times, four correlation rows, sixteen bins; five synchronized calls per implementation after two warmups. Times include the common NumPy output boundary. Allocation peaks exclude caller inputs, host allocations, non-Torch allocations and allocator caches.

| Input case | Implementation | FP64 gate | Median ms | Peak new Torch bytes | Maximum absolute error |
|---|---|---|---:|---:|---:|
| seeded_clean | baseline | True | 1.229 | 25699328 | 7.40749e-06 |
| seeded_clean | final | True | 2.382 | 7489536 | 8.46628e-07 |
| excluded_prefix_contamination | baseline | False | 1.219 | 25699328 | 19.6181 |
| excluded_prefix_contamination | final | True | 2.367 | 7489536 | 8.46628e-07 |

Any case with a failed scientific gate has diagnostic timings only; no equivalent-output speed ratio is inferred for that case.
