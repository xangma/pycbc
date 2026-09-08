# Residual performance review: page-backing hypothesis closed

The huge-page workspace candidate is rejected: whole-engine calls were 29.556–29.998% slower than the identically aligned no-huge-page control in all three fresh workers. The native FFT alone was 35.611–36.033% slower. It fails the rule fixed before measurement and will not advance to executable qualification.

The experiment itself is valid. Independent review verified 34 sealed archive files, all 13 inputs, 25 native contracts, 1,728 full complex128/native and final complex64 precision comparisons, 240 page receipts, all 108 balanced timing blocks and process-group/lock closure. Every observed original and no-huge-page workspace had zero PMD huge-page backing; every advised workspace had exactly 32 MiB. Input preservation, Torch versions, native precision and the original FFTW/MKL error gates passed.

| Fresh worker | Original whole engine (ms) | No-huge control (ms) | Huge-page candidate (ms) |
|---|---:|---:|---:|
| timing-1 | 44.576467 | 33.087066 | 43.012515 |
| timing-2 | 33.414456 | 33.142225 | 43.043837 |
| timing-3 | 33.466811 | 33.079102 | 42.855908 |

Each number is a median of six block medians, with 15 calls and three warmups per block. The first original-allocation worker is visibly slower than the other two and is retained. The no-huge-page arm is a control; this result does not qualify it as a new candidate. The frozen decision is about huge-page backing. First-call/setup timing remains separate and does not change that decision.

The first attempt failed before completing any native contract because this Python build omitted two mmap constant names. Its separately sealed failure is preserved. The reviewed second attempt used checked Linux UAPI constants 14/15 through the same mmap.madvise API; computation, mapping geometry, numerical requirements, timings and thread settings were unchanged.

Previously accepted descriptor reuse remains supported by the completed whole-executable experiment: standard median wall 66.849562 → 65.370059 s, CUDA 21.976242 → 20.291608 s. These are reductions of medians of 2.21318% and 7.66571%, respectively, with all four paired repeats favorable and all scientific comparisons passing. This page-backing rejection does not alter those results or certify production integration.

CPU parity or better remains unmet. The earlier executable gap of 37.466764 s is historical; this microbenchmark supplies no new CPU executable timing or gap against descriptor-cached standard. Retain the qualified promoted complex128 plan and Tensor.copy_; precision-reducing routes and the NumPy copy candidate remain rejected. No additional CPU benchmark is proposed here.

One fixed 2**21-point CPU workload, one native/Torch thread, core 8. Worker estimates are medians of six block medians, each block containing 15 calls. Setup costs are outside steady-state timings and include four workers, one qualification and three timing. No executable speedup, CPU parity, statistical significance or production integration is established.

[Detailed CPU diagnostic and setup costs](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/page-backing-summary-v2.md), [independent raw replay](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/page-backing-results-v2-review.json), [previous descriptor-reuse result](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/RESIDUAL-ANALYSIS-v3.md), [complete evidence pins](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/residual-analysis-v4.json).
