# Optimized single-core pycbc_inspiral reference

Start with [optimized-report-v6/report.json](optimized-report-v6/report.json),
its five CSV tables and three figures. The [baseline report](final-report/report.json)
retains the earlier campaign and its normal CPU tuning. [REPRODUCE.md](REPRODUCE.md)
explains source reconstruction and lossless restoration of large raw profiles.

The optimized source is `a4d77a6d1863c0515e8dace64c5609b63d40b51e`.
Its compressed-waveform CPU interpolation uses shared array views of the existing
compiled routine. Qualified large single-thread inverse FFTs use retained
double-precision MKL workspaces. The measured before source is
`837f38d493420043e45fb1ad210a0ccf68bacbaa`; samples from the two sources are separate.

The workload contains 96 deterministic compressed templates (64 BNS and 32 NSBH
parameter choices), aligned-spin point-particle IMRPhenomD, 30 Hz lower cutoff,
4096 Hz sample rate and 1904 unique valid detector seconds per template.
This workload does not establish search-bank coverage or tidal accuracy.
[config.json](config.json) fixes processing settings and thread limits.
Reference-only tuning selected 512-second FFTs with 112/16-second start/end
padding from the tested 256/512/1024-second grid. Only that unchanged normal CPU
tuning choice is reused. Science, triggers, matched timings and profiles are fresh.

Three unprofiled processes per backend use one allocated host core on `len`,
logical CPU 8, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. CUDA also uses one GPU.
Capacity is `96 × 1904 / full wall seconds` templates per allocated host core at
real time, including startup and output costs. It is a finite executable workload,
not a steady-state estimate or templates/GPU metric.

| Backend | Before median wall (s) | After median wall (s) | After capacity | Capacity / normal CPU |
|---|---:|---:|---:|---:|
| Normal CPU (MKL) | 32.314 | 32.297 | 5659.49 | 1.000 |
| Torch CPU | 80.628 | 44.540 | 4103.78 | 0.725 |
| Torch CUDA | 20.336 | 20.321 | 8994.65 | 1.589 |

Torch CPU improves by 1.81× in the ratio of median wall times. Normal CPU and CUDA
changes are negligible in these observations. See the report for every sample,
observed range, memory use, environment and limitations; three observations do
not establish confidence intervals.

![Before and after capacity](optimized-report-v6/capacity-before-after.png)

All ten optimized report gates pass: 19 campaign cases, 18 strict trigger
comparisons, 288 waveform/PSD checks, 36 boundary injections and 576 compressed
bank parity cases. The 108-case large-IFFT matrix covers three enabled sizes,
four seeds, three patterns and three scales. All enabled cases are bitwise equal
to the promoted MKL reference and satisfy the original error budgets. The
22-file unit run records 470 passed, 69 skipped and eight subtests passed.
The [scientific acceptance](scientific-validation-v6.json) and
[inverse-FFT decision](large-ifft-v6-decision.json) bind the detailed evidence.

Seven additional profiles are separate instrumented executions. The
[profile interpretation](profile-interpretation-v6.md) explains the two changes
and the remaining gap to normal CPU. Python seconds, native cycle shares and
CUDA event durations have different denominators and must not be combined.

All earlier attempts, failures and diagnostics remain archived, including the
rejected original compressed bank, executable context/precision defects, the
v4 CUDA chi-square failure, and exploratory large-IFFT failures. The
[earlier investigation](numerical-investigation-v3.md) and before-source report
retain their original identities. Old science never substitutes for final-source
checks. The sibling [performance-fix](../performance-fix/README.md) contains
supporting component measurements with their separately recorded sources.

`SHA256SUMS` covers actual archived files. Large originals are transported as gzip;
[the restoration guide](archive-transport/README.md) maps them to their original
paths and hashes. Restore first before rebuilding reports or opening raw profiles.
Absolute paths in receipts record provenance and are not instructions to modify
those locations.
