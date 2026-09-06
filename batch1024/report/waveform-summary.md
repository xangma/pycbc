# TaylorF2 public API benchmark

Source: `4885b64560e9f39b740e85b6a976898869dd360e`.

Independent record verification: **passed**. Full campaign qualification: **passed**. 48/48 double precision cells and 48/48 unsupported single precision cells; 288/288 worker files.

4097 frequency bins, both polarizations, the same deterministic BNS parameter rows across routes. CPU/LAL and native scalar routes loop through rows; native batch routes use one public batch call. Timing includes host parameter conversion and allocation. CUDA timed blocks are synchronized.

Throughput uses the median of three process medians, each calculated from five samples. Ranges are the observed min/max of the three replicate medians, **not confidence intervals**. A speedup below 1 means slower than the standard CPU/LAL baseline at the same batch size and host-thread count.

![Verified TaylorF2 throughput](waveform.png)

## Steady-state double precision

| Actual route | Host threads | Batch | Waveforms/s | Replicate range (waveforms/s) | Speedup vs CPU/LAL | Status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Standard CPU / LAL loop | 1 | 1 | 1,941.58 | 1,896.82–1,943.21 | 1.000× | verified |
| Standard CPU / LAL loop | 1 | 8 | 1,938.25 | 1,925.27–1,941.36 | 1.000× | verified |
| Standard CPU / LAL loop | 1 | 32 | 1,780.02 | 1,728.38–1,785.83 | 1.000× | verified |
| Standard CPU / LAL loop | 1 | 128 | 1,668.21 | 1,650.56–1,673.51 | 1.000× | verified |
| Standard CPU / LAL loop | 1 | 512 | 1,650.19 | 1,641.76–1,652.12 | 1.000× | verified |
| Standard CPU / LAL loop | 1 | 1024 | 1,645.40 | 1,643.15–1,652.53 | 1.000× | verified |
| Torch CPU scalar loop | 1 | 1 | 683.26 | 681.20–695.79 | 0.352× | verified |
| Torch CPU scalar loop | 1 | 8 | 691.60 | 678.40–691.89 | 0.357× | verified |
| Torch CPU scalar loop | 1 | 32 | 680.76 | 670.82–682.44 | 0.382× | verified |
| Torch CPU scalar loop | 1 | 128 | 666.76 | 661.59–667.64 | 0.400× | verified |
| Torch CPU scalar loop | 1 | 512 | 638.35 | 635.53–641.54 | 0.387× | verified |
| Torch CPU scalar loop | 1 | 1024 | 624.61 | 593.25–640.47 | 0.380× | verified |
| Torch CPU batch | 1 | 1 | 222.26 | 221.75–224.29 | 0.114× | verified |
| Torch CPU batch | 1 | 8 | 806.57 | 761.34–861.18 | 0.416× | verified |
| Torch CPU batch | 1 | 32 | 831.24 | 828.01–833.66 | 0.467× | verified |
| Torch CPU batch | 1 | 128 | 652.96 | 652.08–654.40 | 0.391× | verified |
| Torch CPU batch | 1 | 512 | 646.81 | 644.14–647.41 | 0.392× | verified |
| Torch CPU batch | 1 | 1024 | 468.95 | 440.99–469.33 | 0.285× | verified |
| Torch CUDA scalar loop | 1 | 1 | 360.81 | 357.30–360.90 | 0.186× | verified |
| Torch CUDA scalar loop | 1 | 8 | 364.02 | 360.05–364.78 | 0.188× | verified |
| Torch CUDA scalar loop | 1 | 32 | 360.80 | 356.15–363.30 | 0.203× | verified |
| Torch CUDA scalar loop | 1 | 128 | 362.66 | 360.07–363.12 | 0.217× | verified |
| Torch CUDA scalar loop | 1 | 512 | 352.28 | 351.11–353.97 | 0.213× | verified |
| Torch CUDA scalar loop | 1 | 1024 | 350.46 | 347.35–352.62 | 0.213× | verified |
| Torch CUDA batch | 1 | 1 | 80.58 | 80.22–81.53 | 0.042× | verified |
| Torch CUDA batch | 1 | 8 | 626.36 | 625.73–633.34 | 0.323× | verified |
| Torch CUDA batch | 1 | 32 | 2,560.44 | 2,546.85–2,568.95 | 1.438× | verified |
| Torch CUDA batch | 1 | 128 | 9,997.18 | 9,986.95–10,148.34 | 5.993× | verified |
| Torch CUDA batch | 1 | 512 | 33,579.13 | 32,733.65–33,960.98 | 20.349× | verified |
| Torch CUDA batch | 1 | 1024 | 45,959.62 | 45,887.49–46,471.50 | 27.932× | verified |
| Standard CPU / LAL loop | 4 | 1 | 1,933.62 | 1,833.04–1,940.67 | 1.000× | verified |
| Standard CPU / LAL loop | 4 | 8 | 1,935.54 | 1,926.49–1,942.23 | 1.000× | verified |
| Standard CPU / LAL loop | 4 | 32 | 1,718.78 | 1,716.57–1,814.79 | 1.000× | verified |
| Standard CPU / LAL loop | 4 | 128 | 1,662.44 | 1,650.47–1,668.35 | 1.000× | verified |
| Standard CPU / LAL loop | 4 | 512 | 1,649.02 | 1,648.29–1,654.74 | 1.000× | verified |
| Standard CPU / LAL loop | 4 | 1024 | 1,641.92 | 1,628.88–1,649.04 | 1.000× | verified |
| Torch CPU scalar loop | 4 | 1 | 698.51 | 693.86–703.79 | 0.361× | verified |
| Torch CPU scalar loop | 4 | 8 | 699.64 | 687.07–702.08 | 0.361× | verified |
| Torch CPU scalar loop | 4 | 32 | 686.05 | 685.39–689.62 | 0.399× | verified |
| Torch CPU scalar loop | 4 | 128 | 653.45 | 649.36–654.16 | 0.393× | verified |
| Torch CPU scalar loop | 4 | 512 | 643.65 | 642.48–645.98 | 0.390× | verified |
| Torch CPU scalar loop | 4 | 1024 | 632.10 | 625.33–636.25 | 0.385× | verified |
| Torch CPU batch | 4 | 1 | 224.19 | 224.15–225.57 | 0.116× | verified |
| Torch CPU batch | 4 | 8 | 883.81 | 877.94–887.69 | 0.457× | verified |
| Torch CPU batch | 4 | 32 | 2,082.63 | 2,074.25–2,135.09 | 1.212× | verified |
| Torch CPU batch | 4 | 128 | 1,690.29 | 1,683.71–1,712.29 | 1.017× | verified |
| Torch CPU batch | 4 | 512 | 1,657.41 | 1,656.91–1,659.45 | 1.005× | verified |
| Torch CPU batch | 4 | 1024 | 1,084.52 | 1,082.64–1,117.69 | 0.661× | verified |

## Cold calls, separately

Milliseconds for the first waveform call in each fresh process, after imports, capability discovery and scheme entry. These are not cold driver/OS-cache measurements and are excluded from throughput and speedup calculations.

| Actual route | Host threads | Batch | Replicate 1 (ms) | Replicate 2 (ms) | Replicate 3 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard CPU / LAL loop | 1 | 1 | 0.8302 | 0.8467 | 0.8546 |
| Standard CPU / LAL loop | 1 | 8 | 4.6876 | 4.8862 | 4.6382 |
| Standard CPU / LAL loop | 1 | 32 | 19.4042 | 19.6435 | 19.5943 |
| Standard CPU / LAL loop | 1 | 128 | 78.7693 | 78.8116 | 78.6762 |
| Standard CPU / LAL loop | 1 | 512 | 313.4617 | 317.8208 | 313.9314 |
| Standard CPU / LAL loop | 1 | 1024 | 627.2871 | 628.6007 | 627.6463 |
| Torch CPU scalar loop | 1 | 1 | 59.1238 | 59.4539 | 58.4650 |
| Torch CPU scalar loop | 1 | 8 | 69.7791 | 69.4573 | 69.4752 |
| Torch CPU scalar loop | 1 | 32 | 120.6300 | 121.7607 | 125.4055 |
| Torch CPU scalar loop | 1 | 128 | 249.6795 | 252.6828 | 252.3596 |
| Torch CPU scalar loop | 1 | 512 | 866.9938 | 855.6021 | 852.1477 |
| Torch CPU scalar loop | 1 | 1024 | 1661.4832 | 1696.0859 | 1706.7121 |
| Torch CPU batch | 1 | 1 | 11.0501 | 11.0460 | 10.9139 |
| Torch CPU batch | 1 | 8 | 19.8836 | 19.4222 | 19.1432 |
| Torch CPU batch | 1 | 32 | 57.0209 | 56.0281 | 55.3276 |
| Torch CPU batch | 1 | 128 | 251.1025 | 250.3026 | 249.5841 |
| Torch CPU batch | 1 | 512 | 1005.6708 | 928.6053 | 981.7974 |
| Torch CPU batch | 1 | 1024 | 2186.1404 | 2197.6720 | 2330.0096 |
| Torch CUDA scalar loop | 1 | 1 | 221.3823 | 220.5714 | 220.9228 |
| Torch CUDA scalar loop | 1 | 8 | 242.4497 | 241.6418 | 240.2316 |
| Torch CUDA scalar loop | 1 | 32 | 309.7378 | 308.8638 | 311.4166 |
| Torch CUDA scalar loop | 1 | 128 | 576.5276 | 572.7122 | 570.1779 |
| Torch CUDA scalar loop | 1 | 512 | 1692.1779 | 1694.8876 | 1677.9802 |
| Torch CUDA scalar loop | 1 | 1024 | 3156.0736 | 3114.8670 | 3135.1054 |
| Torch CUDA batch | 1 | 1 | 404.5845 | 401.8817 | 398.6885 |
| Torch CUDA batch | 1 | 8 | 400.6448 | 402.0815 | 399.5236 |
| Torch CUDA batch | 1 | 32 | 400.1182 | 400.1706 | 400.6590 |
| Torch CUDA batch | 1 | 128 | 401.6752 | 401.5368 | 399.9860 |
| Torch CUDA batch | 1 | 512 | 402.1109 | 386.6872 | 406.7296 |
| Torch CUDA batch | 1 | 1024 | 409.8142 | 406.6954 | 415.7180 |
| Standard CPU / LAL loop | 4 | 1 | 0.8421 | 0.8316 | 0.9859 |
| Standard CPU / LAL loop | 4 | 8 | 4.6742 | 5.1923 | 4.8156 |
| Standard CPU / LAL loop | 4 | 32 | 19.2754 | 19.7208 | 19.6548 |
| Standard CPU / LAL loop | 4 | 128 | 79.5435 | 78.1716 | 78.2568 |
| Standard CPU / LAL loop | 4 | 512 | 314.3487 | 315.7219 | 315.2911 |
| Standard CPU / LAL loop | 4 | 1024 | 635.0677 | 634.7902 | 631.6340 |
| Torch CPU scalar loop | 4 | 1 | 58.6203 | 58.8756 | 60.3670 |
| Torch CPU scalar loop | 4 | 8 | 71.6925 | 70.1772 | 70.1542 |
| Torch CPU scalar loop | 4 | 32 | 105.6146 | 106.4335 | 106.7598 |
| Torch CPU scalar loop | 4 | 128 | 254.2172 | 255.4580 | 254.2852 |
| Torch CPU scalar loop | 4 | 512 | 848.4614 | 854.3624 | 853.8564 |
| Torch CPU scalar loop | 4 | 1024 | 1666.9391 | 1650.5140 | 1703.9923 |
| Torch CPU batch | 4 | 1 | 11.1105 | 11.0917 | 11.1089 |
| Torch CPU batch | 4 | 8 | 18.8795 | 19.0731 | 18.8453 |
| Torch CPU batch | 4 | 32 | 26.4934 | 26.1120 | 26.5093 |
| Torch CPU batch | 4 | 128 | 97.3325 | 99.5255 | 97.6711 |
| Torch CPU batch | 4 | 512 | 379.0245 | 381.1726 | 370.8588 |
| Torch CPU batch | 4 | 1024 | 935.9823 | 945.9460 | 910.3659 |

## Unsupported, failed or unverified cells

No numeric zeros substitute for unsupported capabilities or missing results. The public CPU/CUDA API at this source SHA selects complex128 internally; it exposes no single precision selector. Optional Triton is disabled here and measured in a separate campaign. CUDA was requested with one host thread only.

| Request | Precision | Host threads | Batch | Status | Reason / verification errors |
| --- | --- | ---: | ---: | --- | --- |
| standard-cpu | single | 1 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 1 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 1 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 1 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 1 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 1 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 1 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 1 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-scalar | single | 1 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cuda-batch | single | 1 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| standard-cpu | single | 4 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-scalar | single | 4 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 1 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 8 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 32 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 128 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 512 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |
| torch-cpu-batch | single | 4 | 1024 | unsupported | Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector |

## Verification scope

The renderer independently recomputes sample normalization, each worker median, the median/range across exactly three distinct subprocesses, throughput and baseline ratios. It checks exact clean source SHA before/after successful workers, source/harness hashes, module origins, dtype/device, full row/polarization coverage, dispatch counts, recorded numerical parity metrics, statuses and equal physical inputs. All three replicates must pass before a cell is plotted. The original summary is compared against these computations.

Recorded parity metrics and gates are checked; waveform arrays are not stored, so this renderer does not rerun waveform generation.

Native scalar versus CPU/LAL requires complex relative L2 <1e-11, exact zero support and matching metadata. Batch versus same-device native scalar requires pointwise complex relative error ≤2e-10 with zero absolute tolerance; those scalar outputs are independently checked against CPU/LAL. Direct batch-to-LAL complex, amplitude and phase metrics are retained in raw records.

Runtime inventory and SHA256 hashes of every consumed raw record are in `waveform-summary.json`. Cold timing values remain separate. This bounded waveform workload does not establish end-to-end inference/search performance.
