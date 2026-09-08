The independent full-executable review is complete for the fixed 384-template workload on shared host len. Each role has three fresh unprofiled processes with the original scientific and trigger checks. Qualification timings are excluded.

| Role | Full-wall median (s) | Observed range (s) |
|---|---:|---:|
| standard | 69.873178 | 69.831535–70.114786 |
| cpu-baseline | 113.969209 | 113.395970–114.334606 |
| cpu-candidate | 106.943054 | 106.897436–107.186163 |
| cuda-baseline | 24.658973 | 24.628002–24.862198 |
| cuda-candidate | 24.711847 | 24.637057–24.726278 |

Ranges describe the three observed processes; they are not confidence intervals.
The CPU candidate changes median full wall time by -6.165% versus its baseline and takes 1.531 times standard CPU wall time.
The CUDA candidate changes median full wall time by +0.214% versus its baseline and takes 0.354 times standard CPU wall time.

The production in-place promoted-MKL plan passes the unchanged 36-case numerical matrix, including matching-precision MKL parity. Its warm median is 9.343% lower than the accepted promoted baseline and remains 2.091 times standard CPU. Those are warm transform measurements, not full-process claims.

The final single-precision FFTW experiment fails two original maximum-error comparisons and is excluded. The legacy standard-MKL control failure remains recorded. Native Triton correlation AD remains unsupported on the tested route; fallback tests preserve the supported derivative contract.

Two failed executable-controller attempts are retained: the standard scheme already imports Torch, and the established PSD files differ across schemes outside the searched frequencies. The final controller compares exact frozen PSD records by scheme and each candidate to its own baseline. No numerical tolerance changed.

All scientific files, worker records, comparison reports, terminal audits and source pins are bound by [followup-review-summary.json](followup-review-summary.json). Source/harness reviews describe the revision at which they were written; their pending-validation wording is superseded only by the explicit completed evidence linked in that index.
