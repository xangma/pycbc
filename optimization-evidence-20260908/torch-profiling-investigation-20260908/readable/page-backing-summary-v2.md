# CPU page-backing diagnostic v2

no consistent benefit under the frozen rule; do not advance or retry this campaign.

Independent review verified all 1,728 full-native/final precision rows, 108 timing blocks, input/runtime/source pins and terminal process-group/lock closure. All workers passed their complete numerical and page-coverage gates.

The three arms use the existing Torch allocation (O), an aligned private mapping with no-huge-page advice (N), and the same mapping with huge-page advice (H). H versus N isolates page advice; comparisons with O also include allocator/alignment differences. The original FFT execution, complex128 workspace and promotion/demotion copies are preserved.

| Worker | Metric | O (ms) | N (ms) | H (ms) | H reduction vs O | H reduction vs N |
|---|---|---:|---:|---:|---:|---:|
| timing-1 | Whole engine | 44.576467 | 33.087066 | 43.012515 | 3.5085% | -29.9980% |
| timing-1 | Native FFT | 40.065777 | 28.529772 | 38.809957 | 3.1344% | -36.0332% |
| timing-2 | Whole engine | 33.414456 | 33.142225 | 43.043837 | -28.8180% | -29.8761% |
| timing-2 | Native FFT | 28.881635 | 28.576366 | 38.752787 | -34.1780% | -35.6113% |
| timing-3 | Whole engine | 33.466811 | 33.079102 | 42.855908 | -28.0549% | -29.5558% |
| timing-3 | Native FFT | 28.851672 | 28.471864 | 38.647573 | -33.9526% | -35.7395% |

Positive reductions favor H. Each entry is a median of six block medians from 15 timed calls per block, with three warmups and all six arm orders. The native attribution pass refreshes the input by promotion before every transform. Its native timer excludes that promotion; the whole-engine timer includes both copies and checks.

| Arm | Constructor including commit (ms) | First engine call (ms) |
|---|---:|---:|
| original | 33.038051 (30.888509–33.515712) | 71.917089 (69.791193–74.335795) |
| nohuge | 31.116047 (30.760527–32.983102) | 71.618318 (62.584323–72.378093) |
| huge | 31.845521 (30.678695–33.943854) | 58.166269 (56.091410–58.393816) |

Setup entries are medians (ranges) over all four workers, including qualification, and remain separate from steady-state performance. The JSON retains all samples, descriptor-finalization costs and every distinct observed page-coverage bound.

One fixed 2**21-point CPU workload, one native/Torch thread, core 8. Worker estimates are medians of six block medians, each block containing 15 calls. Setup costs are outside steady-state timings and include four workers, one qualification and three timing. No executable speedup, CPU parity, statistical significance or production integration is established.

Independent review SHA256: `d8f07c6031c32e96fe83cb7cd23a208781950d924ad9cb00c1f4c1bd2eecf313`. Sealed archive SHA256: `2a390988fea2736ea2a7f8b55ed0750fc945c2c04214b25a251c74e62a6b6267`.
