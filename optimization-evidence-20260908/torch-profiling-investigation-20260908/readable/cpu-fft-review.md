# CPU FFT investigation, 8 September 2026

The retained promoted-MKL transform is the main remaining Torch CPU cost. Reusing its plan is already implemented. Two dtype-conversion passes remain necessary in its current dataflow. No new source change, dispatch qualification, or performance measurement is claimed by this review.

## Existing executable evidence

The frozen current profile is `d2647addb884ead3249914ebc980f3c132076d93`: 384 distinct compressed templates, five 2,097,152-point segments per template, 1,904 valid detector seconds. CPU and Torch CPU use one numerical thread pinned to logical CPU 8 on shared host `len`. Unprofiled full-process medians are 70.219288 and 114.163224 seconds respectively (three workers each); the 43.943936-second difference is the measured full-workload gap.

Independent filtering cProfiles total 51.065916 and 96.890385 seconds. Normal CPU MKL FFT execution accounts for 28.406514 seconds. Torch's promoted MKL execution accounts for 71.254080 cumulative seconds, including 62.517340 self seconds and 8.6989 seconds in 3,840 conversion copies. The 42.847566-second difference between these FFT entry points is about 93.5% of the difference between the two instrumented filtering runs. This accounting identifies a priority; it is neither a causal speedup estimate nor permission to subtract profile times from unprofiled totals.

The native samples corroborate different precision: normal CPU FFT symbols are `32fc`; Torch uses `64fc`. The frozen implementation creates the descriptor once and owns two complex128 workspaces. Each execution copies complex64 input into the first workspace, transforms into the second, publishes its mutation, and copies back to complex64. At this transform length the two private workspaces occupy 64 MiB total. The two conversions imply 96 MiB of logical read/write traffic per transform, before the FFT itself. These are array sizes, not measured memory-controller traffic.

Source: [promoted plan](/Users/xangma/repos/pycbc/artifacts/torch-profiling-20260907-r3/publication-final-restore/pycbc-torch-profile-20260907-r3/source/pycbc/fft/torchfft.py:139), [execution](/Users/xangma/repos/pycbc/artifacts/torch-profiling-20260907-r3/publication-final-restore/pycbc-torch-profile-20260907-r3/source/pycbc/fft/torchfft.py:271), [published profile](/Users/xangma/repos/pycbc/artifacts/torch-profiling-20260907-r3/publication-prep/REPORT.md).

## Earlier large-transform qualification

The final 108-case matrix is source `a4d77a6d1863c0515e8dace64c5609b63d40b51e`. Each size has four seeds, three patterns (dense, finite band, impulse), and three scales (1e-12, 1, 1e12), with one native thread. Its unchanged requirements are L2 and maximum absolute error each no larger than legacy FFTW against a complex128 NumPy inverse, with no tolerance floor; the tested MKL candidates also require bitwise matching-precision standard-MKL parity after the complex64 output cast.

| Length | Legacy FFTW c64 ESTIMATE ms | MKL c64 direct ms | Promoted MKL ms | FFTW c128 workspace ESTIMATE ms | MKL c64 failed cases |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 25.266 | 7.039 | 17.483 | 25.462 | 0 / 36 |
| 2,097,152 | 97.656 | 15.997 | 36.866 | 55.645 | 1 / 36 |
| 4,194,304 | 169.979 | 33.172 | 74.797 | 125.310 | 5 / 36 |

These are historical warm medians within one qualification process, not current throughput or replicated full-process comparisons. The production-size direct-MKL failure is seed 812, dense, scale 1e12: maximum absolute error ratio 1.0043188307533255 versus legacy FFTW, despite an L2 ratio of 0.9265943648338082 and bitwise standard-MKL equality. Preserve that failure. Passing the smaller size cannot authorize the production size. All 108 promoted-MKL cases pass.

The old c64 FFTW control uses the legacy class plan. It does not directly measure the proposed Torch retained-buffer/MEASURE route. Its slow ESTIMATE result is a reason to measure that distinction, rather than assume single precision makes FFTW faster. FFTW measure level 0 means ESTIMATE; level 1 means MEASURE. The current retained-MEASURE release is limited to 131,072 points. The old c128 FFTW workspace uses ESTIMATE and is already slower than promoted MKL; a measured plan remains a distinct experiment.

## Bounded candidate priorities

1. **Promoted MKL in-place private workspace.** Test whether one 32 MiB private workspace improves locality and native execution. Both conversion passes remain. Preserve input storage, target version updates, failure cleanup, exact-pointer/shape/dtype/thread guards and all AD exclusions. Numerical parity must be measured: in-place placement can choose a different FFT algorithm.
2. **Explicit retained FFTW MEASURE at the production length.** Measure construction separately, first execution and steady state over independent fresh processes; report the finite 1,920-transform amortization. Qualify alignment classes, source preservation, address changes and legacy error budgets. No expansion of allowlists before scientific validation. Changing the planning algorithm is a numerical-route change even if precision stays c64.
3. **Measured promoted FFTW plan.** This preserves promotion but must beat the existing promoted-MKL route including conversion and planning costs. The earlier ESTIMATE result provides no such evidence.

Removing conversion copies alone has only 8.7 profile seconds of visible headroom and cannot account for the full CPU gap. Squared norm (about 1.45 cumulative seconds) and trigger deep copying (about 1.19 full-profile cumulative seconds) are smaller opportunities; the previous deep-copy candidate failed to show a reproducible end-to-end gain. Guard micro-optimizations have negligible observed headroom.

The separate batch sweep uses 131,072-point transforms and pre-squared-norm source `9578a710479b`; its warm public-API rates are not current executable rates. Torch CPU batch one uses its qualified c64 FFTW route; multiple rows promote and chunk at 2^20 elements (eight rows at this length). The batch-one to batch-eight rate drop is consistent with that route transition. Sizes 32 to 128 show additional loss whose exact cause still needs targeted allocation/layout/dispatch attribution on the candidate source. CUDA batch growth and optional peak handling require separate completed-result timing, including safe host exports.

## Evidence and reproduction

[existing-cpu-summary.json](existing-cpu-summary.json) preserves exact historical failures, measured medians, current profile rows, and SHA256 input bindings. Regenerate in a fresh directory using [summarize-existing-cpu.py](summarize-existing-cpu.py); it reads existing local artifacts only and refuses to replace its output. The matrix SHA256 is `e156a0d659078015c56b4712b04c5f10608f94b63950f2894219a71aae5517a0`; the dispatch decision is `81b3be52929a99de7e54ccea930dd7ecece88fe65bc6350afe3d02980d81028a`. Both agree with the earlier published inventory. The source and input snapshots remain unchanged.
