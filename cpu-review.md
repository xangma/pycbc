# CPU precision corrections

Four changes improve CPU numerical accuracy while retaining the existing public
single-precision outputs. They form a standalone prerequisite for the Torch work:
frozen original `40e94792b3edf59f39b18b65102b28a4f74433a7` → corrected CPU
`66789ac4a7468094b0cc3ca1498a1de67e0311f6`. The branch contains seven changed files,
four commits, no Torch dependency, and no native-kernel changes.

## Changes

1. Promote float32 strain before PSD estimation, interpolation and inverse-spectrum
   truncation, then restore the requested public precision.
2. Accumulate float32 template power in float64 before publishing the cumulative
   series. Per-frequency power still starts from the original rounded inputs.
3. Use double-precision phase shifts and accumulation for CPU point chi-squared,
   correcting the existing kernel's shortened pi constant through its Python
   wrapper. The native kernel remains unchanged.
4. Transform float32 strain segments in float64, then return complex64 spectra.

Existing float64 inputs avoid unnecessary promotion copies. Metadata, segment
geometry and output dtypes are covered by the new regression tests. Detailed
extraction and local verification are in [the split report](source-split-report.md).

## Independent numerical evidence

The reference calculations start from the same captured, rounded workload inputs.
Linux long double has 63 mantissa bits; the Welch reference explicitly uses an
extended-precision FFT. These measurements assess numerical accuracy for those
inputs, rather than search acceptance thresholds.

| Quantity | Original error | Corrected error |
| --- | ---: | ---: |
| Maximum relative estimated PSD error | 0.00302851 | 5.95648e-8 |
| Maximum relative public cumulative-power total error, original PSD | 0.00606489 | 5.78267e-8 |
| Maximum relative public cumulative-power total error, corrected PSD | 0.00606574 | 5.59648e-8 |
| Relative L2 segment FFT error, five segments | 1.60e-7–1.75e-7 | 1.83e-8–3.48e-8 |
| Maximum absolute point chi-squared error, nine captured points | 0.286298 | 1.71086e-6 |

Across 384 templates and each of two frozen PSDs, all 5,760 internal bin boundaries
are closer to the input-power reference after correction. Some corrected edges
still differ from that reference because per-frequency power and the public
prefix series are rounded to float32. They all match the separate diagnostic
that retains this public rounding. See [bin analysis](bin-report-final/bin-report.md)
and [numerical results](numerical-acquisition/numerical-audit/results.json).

The [captured-input injection probe](injection-run-1/reviewed-results.md) completed
48 matched-template cases and exactly reproduced six null cases. All 24 noiseless
peaks occur at the injected sample; all 24 noisy peaks agree with the independent
reference. Maximum noise-plus-signal chi-squared error falls from 1.812089 to
0.000115494. Each partition has its own reference statistic: changing the bins
also changes the statistic. The probe holds conditioned spectra and PSD fixed,
uses six locations selected from existing triggers, and does not measure FAR,
population sensitivity, or the response of conditioning, gating or PSD estimation
to an injection.

The numerical and injection captures used combined source `123e1fb3ef1b338cada636e71c3e9c7987002402`.
A fresh Linux executable replay of the standalone CPU branch subsequently matched
all 13 saved H1 trigger datasets, the PSD, conditioned strain and segment geometry
exactly. All 13 workload qualification checks passed, covering 384 compressed
templates, five segments and 1,904 valid seconds; no Torch was imported. See
[the replay comparison](linux-acquisition/linux-validation-v3/comparison.json)
and [runtime receipt](linux-acquisition/linux-validation-v3/runtime.json).

## Intentional scientific-output changes

On the frozen GW170817 H1 workload, the original CPU source produces 1,988 triggers
and the corrections produce 1,991: 1,959 share the same template/time identity,
29 occur only in the original output and 32 only in the corrected output.
The top 100 share 92 identities. For common triggers, the median absolute newSNR
change is 0.01855, the 95th percentile 0.13492 and the maximum 0.36312.
See [ranking consequences](trigger-ranking-consequences.md).

These changes require scientific review. More accurate arithmetic does not by
itself establish improved detection efficiency or background calibration. No
acceptance tolerance was widened to make the original and corrected outputs pass
an equivalence check.

## Unit tests and runtime validation

- macOS ARM64 / FFTW: 34 affected-module and precision tests passed, plus two
  existing chi-squared tests using cached fixtures. The original source fails 11
  of the 19 added precision cases.
- Linux x86-64 / MKL, one CPU core: all 19 precision tests passed; 32 affected-module
  tests passed in total. Two existing matched-filter timing assertions fail with
  exactly the same values on the original and corrected sources. The offset is
  0.06937270748 sample. [Corrected log](linux-acquisition/linux-validation-v2/tests.log),
  [baseline reproduction](linux-acquisition/linux-baseline-mkl-tests.log).
- CI F401 lint and new-test flake8 passed. Changed production files add no flake8
  diagnostics relative to the baseline; full-repository flake8 has existing
  failures. Qlty was unavailable.
- Linux uses unchanged native binaries copied from the frozen original build,
  with source and binary hashes verified. Source, native, harness and input hashes
  remain pinned. This is one Linux/MKL configuration; other accelerators and FFT
  implementations are not covered by this CPU validation.

The first Linux test-launch attempt incorrectly held an outer CPU scheme while
legacy tests tried to enter their own contexts. That harness error was corrected
without changing product code, tests or tolerances. Raw attempts are retained.
The final replay gate separately records the two baseline-reproduced MKL failures.

## Runtime cost

The completed timing results are reported in [the CPU cost report](cpu-cost-report.md).
The comparison measures the cost of changed arithmetic and changed trigger output;
it is not an equal-output backend speedup comparison.
