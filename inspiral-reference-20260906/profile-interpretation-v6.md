# Optimized CPU hot-path profiles

Source `a4d77a6d1863c0515e8dace64c5609b63d40b51e` is compared with `837f38d493420043e45fb1ad210a0ccf68bacbaa` using separate cProfile runs
of the selected 96-template, 512-second, one-thread workload. The accompanying
[JSON](profile-interpretation-v6.json) binds four raw pstats files, both reports
and eight source blobs. All numbers below are instrumented cumulative seconds.

| Call path | Torch CPU before | Torch CPU after | Normal CPU after |
|---|---:|---:|---:|
| Compressed interpolation (96 calls) | 26.625 | 0.440 | 0.422 |
| Inverse FFT (480 calls) | 26.845 | 17.818 | 7.170 |

Torch CPU profile total fell from 82.852 to
47.097 seconds. These disjoint paths account for
35.211 seconds, or 98.48%
of that reduction. The existing compiled interpolation routine is now reached
through shared CPU array views. The large search inverse FFT uses retained
double-precision MKL workspaces with input promotion and output conversion.
Both paths retain their qualification and fallback constraints.

The after-profile gap to normal CPU is 12.509 seconds. The inverse-FFT
gap is 10.648 seconds (85.12% of the total gap).
Normal CPU uses single-precision MKL; Torch's qualified large transform retains
double precision. Precision, conversion and library/planning effects have not
been isolated individually, so this comparison does not assign a separate cost
to each. Nested native-plan and copy times must not be added to wrapper time.

For capacity, use the nine separate unprofiled processes in
[the optimized report](optimized-report-v6/report.json). Native perf cycle shares
and CUDA event durations use different denominators and cannot be added to these
Python profile seconds. See [REPRODUCE.md](REPRODUCE.md) to restore raw profiles.
