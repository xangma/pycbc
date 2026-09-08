# Equal-power bin accuracy on captured workload inputs

Validated 768 metric rows: 384 captured waveforms, each evaluated with both frozen PSDs. Summary counts were independently recomputed and agree with the analyzer. Input-file integrity is attested by the remote analyzer; this aggregation independently pins the delivered reports.

The mathematical reference uses the captured rounded complex64 waveform and float32 PSD. The scan-only reference starts with the rounded float32 power array. The public-rounding diagnostic publishes that latter longdouble prefix through float32 storage and normalization, then constructs float64 thresholds. These answer different accuracy questions.

## Boundary locations

All counts below refer to internal boundaries. Only stable reference boundaries support the exact/one-cell/larger-shift classification. An equal distance or ambiguous reference does not establish which path is better.

| PSD | Reference | Stable / total | Old closer | Current closer | Same selected edge | Unresolved |
|---|---|---:|---:|---:|---:|---:|
| original | input_power | 5760 / 5760 | 0 | 5760 | 0 | 0 |
| original | rounded_power_scan_only | 5760 / 5760 | 0 | 5760 | 0 | 0 |
| proposed | input_power | 5760 / 5760 | 0 | 5760 | 0 | 0 |
| proposed | rounded_power_scan_only | 5760 / 5760 | 0 | 5760 | 0 | 0 |

| PSD | Current versus input-power reference: exact | One cell off | More than one cell off | Maximum stable shift |
|---|---:|---:|---:|---:|
| original | 5667 | 69 | 24 | 22 |
| proposed | 5659 | 77 | 24 | 37 |

A more accurate cumulative sum does not guarantee every public edge equals the mathematical reference. Input-power rounding and float32 prefix publication can still move a boundary. The following check compares current edges with the diagnostic that retains that same public rounding:

- original PSD: 0 internal edges differ across 0 templates from the ideal-scan/public-rounding diagnostic.
- proposed PSD: 0 internal edges differ across 0 templates from the ideal-scan/public-rounding diagnostic.

## Accumulation and final public values

The values below are maximum absolute relative total errors across templates. Scan errors use the rounded-power reference; public errors use the input-power reference with exact normalization. They must not be treated as interchangeable.

| PSD | Old scan | Current scan | Old public | Current public |
|---|---:|---:|---:|---:|
| original | 0.00606489 | 3.73793e-14 | 0.00606489 | 5.78267e-08 |
| proposed | 0.00606574 | 7.08763e-14 | 0.00606574 | 5.59648e-08 |

## Actual per-bin powers and tail loss

Bin power is independently reduced over each selected interval using the input-power reference. Error is relative to total power divided by the bin count. The discrete-cell budget is the larger adjacent boundary-cell power, plus reference uncertainty; it is an analytic quantization check, not a search acceptance threshold.

| PSD | Path | Median of per-template maximum bin error | Largest bin error | Bins beyond cell budget + uncertainty | Stagnated positive scan cells |
|---|---|---:|---:|---:|---:|
| original | old | 0.0308239 | 0.0907429 | 6144 | 221499118 |
| original | current | 0.000160296 | 0.000305516 | 1 | 0 |
| proposed | old | 0.0308292 | 0.0907451 | 6144 | 221499079 |
| proposed | current | 0.000160181 | 0.000299053 | 1 | 0 |

Stagnated-cell power is not automatically the net accumulation error: later rounding can compensate. The JSON separately reports the rounded power remaining after the final scan increment, plus full-prefix and total errors.

## Analytical signal consequence of bin imbalance

For a perfectly matched noiseless signal with fixed PSD and bins, the analytical bin-imbalance noncentrality diagnostic is `lambda(rho) = rho^2 / M * sum_j(relative_to_target_j^2)`, with M=16 for this capture. Relative errors come from each partition’s true input-reference bin powers. Oracle means the mathematical input-power reference partition, including its unavoidable discrete-cell imbalance.

This calculation is not an injected-strain recovery result. It holds the waveform, PSD, partition, and SNR definition fixed; it does not simulate noise, template mismatch, trigger selection, sensitivity, or false-alarm rates.

The table gives the maximum lambda across templates for each PSD and partition; the JSON includes the median and 95th percentile, and the template CSV contains every value.

| PSD | Partition | SNR 5.5 | SNR 8 | SNR 12 | SNR 20 | SNR 50 | SNR 100 | Maximum template |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | old | 0.0166059 | 0.0351332 | 0.0790498 | 0.219583 | 1.37239 | 5.48957 | 376 |
| original | current | 4.47459e-07 | 9.46691e-07 | 2.13005e-06 | 5.91682e-06 | 3.69801e-05 | 0.00014792 | 318 |
| original | oracle | 4.47459e-07 | 9.46691e-07 | 2.13005e-06 | 5.91682e-06 | 3.69801e-05 | 0.00014792 | 318 |
| proposed | old | 0.0166068 | 0.035135 | 0.0790537 | 0.219594 | 1.37246 | 5.48984 | 376 |
| proposed | current | 4.17878e-07 | 8.84105e-07 | 1.98924e-06 | 5.52566e-06 | 3.45354e-05 | 0.000138141 | 278 |
| proposed | oracle | 4.19022e-07 | 8.86525e-07 | 1.99468e-06 | 5.54078e-06 | 3.46299e-05 | 0.00013852 | 371 |

## Coverage, precision, and PSD substitution

- original PSD: 0 native power captures, 0 captured bin checks, 0 byte-exact public-prefix checks; 384 templates use source emulation.
- proposed PSD: 381 native power captures, 381 captured bin checks, 381 byte-exact public-prefix checks; 3 templates use source emulation.
  Emulation indices: 91, 95, 246.
- Reference runtime: NumPy 1.26.4, Python 3.11.9, x86_64; longdouble has 64 significand bits (128 storage bits).

Holding each captured waveform fixed while switching PSDs changes the following selected boundaries. This isolates PSD substitution effects; it does not establish which physical PSD is correct.

| Path | Templates with changed edges | Changed internal edges | Maximum shift |
|---|---:|---:|---:|
| old | 374 | 1766 | 552 |
| current | 377 | 1828 | 2357 |
| ideal_public_rounding | 377 | 1828 | 2357 |

The findings concern numerical accuracy on these saved inputs. They do not establish end-to-end scientific equivalence, sensitivity, or false-alarm behavior. No frozen comparison tolerance has been changed. Full intermediate numbers are in bin-aggregate.json; per-template and boundary records are in the companion CSV files.

Reference boundary uncertainty assumes IEEE round-to-nearest and finite normal longdouble arithmetic. Total-error rankings and floating scalar summaries are descriptive point estimates. The aggregator checks reports; selected waveform fixtures require separate independent recomputation.

## Provenance

```json
{
  "analysis_directory": "/Users/xangma/repos/pycbc/artifacts/torch-precision-validation-20260908/numerical-acquisition/bin-audit",
  "report_sha256": {
    "metrics.jsonl": "0ae247a73d28e72ff4145037fdbb28b1b5a4306c1cadab31b56d03b18b776859",
    "summary.json": "4cdad4b9e7c1f54f228db75eafd86a73f1c27e35851a273e74af94355c4200ce"
  },
  "analyzer_sha256": "91860412bce0c3b4c5e6fac74d49f901e3d79b01d75929a18b8b5a39e0197c39",
  "aggregate_script_sha256": "ef50c21fe415ff4de8c4957d8967d880841f414c5bbbb8a442ed66b4145c3e54",
  "remote_input_file_count": 1150
}
```
