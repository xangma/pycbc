# CPU conversion experiment

The guarded NumPy copy candidate did not improve the measured whole-call time and is not selected for a production patch. The existing source remains unchanged at ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f.

| Fresh worker | Current engine median | NumPy copy median | Candidate minus current |
|---|---:|---:|---:|
| cpu-copy-1 | 33.145242 ms | 33.183830 ms | +0.038588 ms |
| cpu-copy-2 | 33.450207 ms | 33.510964 ms | +0.060757 ms |
| cpu-copy-3 | 33.522067 ms | 33.558876 ms | +0.036809 ms |

All 14 storage, mutation, failure and lifetime checks passed in a separate fresh native process before public-tensor NumPy exports. All 432 strict numerical cases passed: four seeds × three patterns × three scales, for each path before and after timing. Outputs retained the unchanged FFTW error budgets and bitwise complex128 MKL parity.

Each worker measured four alternating-order blocks with three warmups and 15 whole-call samples per path per block, giving 60 samples per path. These paired path medians show no supported benefit; the small positive differences are not a claim of a statistically established regression. Source, libraries, one-core affinity, intra-op and native-pool settings were verified. Torch inter-op was observed as 64; the executable controls use 1. No executable gain or isolated conversion-time measurement follows from this microbenchmark.

Controller process group 1047042 and all workers ended; independent auditor 1175934 reacquired the campaign lock. The 27-member archive and all 26 result-manifest entries were byte verified locally. Independent peer replay passed all archive, contract, numerical, timing and closure checks; parent accepted the rejection. No production patch or executable trial is selected.

Evidence: cpu-copy-v1-summary.json SHA256 a2f1f36e273e94f17cc94e6781b8682ea983005dc5daa119f730d632fd01893b; summarize-cpu-copy.py SHA256 92b9ff6bea55a2bb58eb14a77e170f2f13804c8e1c5642581e99c17f7fe7ba4c; cpu-copy-v1-results.tar SHA256 7ba69400eecb515d9819d69897f5c5b5558a69d3bc0233a96c03d9d213fd0dba.

Independent review: peer-reviews/cpu-copy-v1-results-peer-review.json SHA256 26eb672406be3da4797dd04bebf39a7f77f380fb70c325fb871cd6e83e6c338b.
