The frame-loader campaign passed independent scientific and runtime review for all six roles and all 18 fresh timing runs.

| Route | Full-wall median (s) | Observed range (s) |
|---|---:|---:|
| standard-baseline | 70.000746 | 69.822451–70.570593 |
| standard-candidate | 66.933429 | 66.629332–67.996605 |
| cpu-baseline | 107.068339 | 107.062365–107.645261 |
| cpu-candidate | 104.400193 | 104.212417–104.995407 |
| cuda-baseline | 24.786463 | 24.706543–25.537461 |
| cuda-candidate | 21.765736 | 21.727497–22.123757 |

The loader change is applied to standard CPU, Torch CPU and CUDA. Each result uses three fresh processes on shared host len, with one assigned CPU core and native/Torch threads fixed to one; CUDA also uses the GPU. Observed ranges are not confidence intervals.

standard-candidate takes 0.9562 times standard-baseline full wall time (4.382% reduction).
cpu-candidate takes 0.9751 times cpu-baseline full wall time (2.492% reduction).
cpu-candidate takes 1.5598 times standard-candidate full wall time (-55.976% reduction).
cuda-candidate takes 0.8781 times cuda-baseline full wall time (12.187% reduction).
cuda-candidate takes 0.3252 times standard-candidate full wall time (67.482% reduction).

All six qualifications pass the unchanged checks and exact per-scheme strain/PSD records. All eight qualification and 36 timing trigger comparisons pass; same-scheme comparisons additionally require bitwise equality for all 18 scientific H1 datasets. Four performance telemetry fields are independently validated as timing metrics.

The separate warm sparse-API diagnostic supports bounded gains in selected cells. The previous full-executable scalar-hoist result remains neutral. The CPU route retains promoted precision because the faster single-precision alternatives failed the unchanged numerical budgets.

Hashes, native validation, preserved failures, limitations and full sample statistics are bound in [loader-review-summary.json](loader-review-summary.json). The preceding full-executable evidence remains in [followup-review-summary.json](followup-review-summary.json).
