# FFTW wisdom and IFFT — 6 September 2026

Public Torch CPU IFFT, one complex64 vector of length 131072; plan construction measured separately; four-thread path uses Torch fallback.

Three independent processes; median and min/max of process medians; range is not a confidence interval.

| Route | Plan construction ms (range) | Warm IFFT ms (range) | Status |
| --- | ---: | ---: | --- |
| Main · off · 1T | 42.570 (42.422–42.586) | 0.730 (0.730–0.744) | passed |
| Optional · off · 1T | 42.235 (42.235–42.590) | 0.737 (0.730–0.751) | passed |
| Optional · cold · 1T | 2417.146 (2396.273–2475.413) | 0.571 (0.560–0.583) | passed |
| Optional · warm · 1T | 76.070 (75.165–76.259) | 0.569 (0.562–0.573) | passed |
| Main · off · 4T | 0.140 (0.125–0.141) | 0.297 (0.296–0.300) | passed |
| Optional · off · 4T | 0.129 (0.129–0.130) | 0.297 (0.296–0.298) | passed |

Cache-off and warm-cache workers use FFTW ESTIMATE; cold-cache workers use MEASURE and export wisdom. Warm workers import the corresponding cold worker’s isolated cache. The one-thread direct FFTW route and the four-thread Torch fallback are observed separately. All 18 workers must pass the numerical, dispatch and cache checks to support a complete comparison.

Parity covers the full output against a NumPy complex128 unnormalized IFFT (relative L2 and maximum error relative to reference peak ≤2e-6), input preservation and repeat-output equality. Timing excludes verification and instrumentation. Five samples of ten IFFTs follow three warmups in each process. Plan times are one construction per fresh process, after imports; they are not complete process startup times.

This single-vector case measures automatic cache behavior. It does not measure every batch-layout correction in PR #16. CPU affinity 8–11 on a shared Threadripper PRO 3995WX; execution is sequential and host telemetry is retained.
