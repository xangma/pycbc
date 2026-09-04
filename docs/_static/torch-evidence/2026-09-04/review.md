# Final len performance review

Reviewed only the fresh campaign in
`len:/home/xangma/pycbc-torch-finish-20260904/performance`; no archived results
were used. All 63 timed cells and all six untimed correlation probes passed.

## Revision, environment and workload

- Revision: `ec12863ee897b529960d8579526715d8d54810e9`. Both artifacts record
  every selected route as clean at this revision, with dirty-source overrides
  disabled. All 63 workers use the same source checkout and distinct PIDs.
  Later qlty output made `.qlty/` entries untracked after benchmarking; that is
  separate from the clean benchmark source identities.
- Source: `/home/xangma/pycbc-torch-finish-20260904/tree`.
  Tool: `tools/bench_production_live_batch.py`, SHA-256
  `8503f204a42e8a6e1af0efd29c081e4c07237771168ed4bf3d8b24eb1bfa16de`.
- Interpreter:
  `/home/xangma/pycbc-torch-fixes-20260904-epKdaA/venv/bin/python`,
  CPython 3.11.9; NumPy 1.26.4; Torch 2.13.0+cu130 with CUDA runtime 13.0.
  PyCBC reports version `0.0a9400`; the Git revision identifies the tested code.
  Child `PYTHONPATH` is replaced with the current source root. Native extension
  hashes are recorded in `environment-before.txt`.
- Host: len, AMD Threadripper PRO 3995WX, NVIDIA RTX 4090, Linux
  6.8.0-138-generic. Affinity is CPUs 8–11. The launcher requests
  `OMP_DYNAMIC=FALSE`, OMP/MKL threads 1 or 4, and OpenBLAS/NumExpr threads 1.
- Timed interval: t1 2026-09-04 21:16:43–21:20:43 UTC; t4
  21:20:44–21:22:17 UTC. Probes completed by 21:22:47 UTC.
- Public `LiveBatchMatchedFilter.process_data`, N=131072, three blocks per
  iteration, complex64 templates/strain/output, float32 PSD. Each cell has
  one initial three-block pass, two warm-up iterations, and five measured
  iterations. There are three independent worker replicates per route/batch.

## Acceptance checks

The expected 45 t1 cells and 18 t4 cells are all present, with the requested
dimensions and sample counts. Both global parity results and all 48 individual
route comparisons pass. Every one of the 189 recorded final-iteration blocks
contains exactly one finite trigger: its injected template ID at the expected
arrival time, within 1e-4 seconds. No empty-trigger workload passed unnoticed.

Maximum differences across the 48 route comparisons:

| Metric | Maximum |
| --- | ---: |
| Absolute SNR difference | 2.86103e-6 |
| Wrapped phase difference | 1.37121e-7 |
| Relative sigma-squared difference | 2.38419e-7 |
| Relative aggregate output-norm difference | 1.86859e-7 |

## Throughput

Values are the median throughput within each of the three worker replicates,
in templates/second, rounded to two decimals. These are not pooled confidence
intervals. Route names ending in `_native` mean the requested experimental
configuration; the separately verified component is batch correlation.

| Threads | B | Route | Replicate 1 | Replicate 2 | Replicate 3 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | branch_standard | 1126.58 | 1138.65 | 1129.66 |
| 1 | 1 | torch_cpu | 489.16 | 495.40 | 497.74 |
| 1 | 1 | torch_cpu_native | 940.57 | 928.08 | 932.96 |
| 1 | 1 | torch_cuda | 2138.02 | 2132.99 | 2126.40 |
| 1 | 1 | torch_cuda_native | 2120.89 | 2105.58 | 2124.55 |
| 1 | 8 | branch_standard | 1176.87 | 1186.87 | 1183.40 |
| 1 | 8 | torch_cpu | 292.88 | 292.33 | 291.53 |
| 1 | 8 | torch_cpu_native | 963.11 | 957.29 | 958.47 |
| 1 | 8 | torch_cuda | 13440.24 | 13468.95 | 13484.29 |
| 1 | 8 | torch_cuda_native | 11521.37 | 11598.02 | 11537.86 |
| 1 | 32 | branch_standard | 1194.76 | 1192.25 | 1191.90 |
| 1 | 32 | torch_cpu | 298.65 | 297.34 | 297.49 |
| 1 | 32 | torch_cpu_native | 938.49 | 935.95 | 942.04 |
| 1 | 32 | torch_cuda | 40895.14 | 40944.09 | 40795.10 |
| 1 | 32 | torch_cuda_native | 27098.58 | 26975.78 | 27036.90 |
| 4 | 1 | branch_standard | 1953.31 | 1939.47 | 1946.98 |
| 4 | 1 | torch_cpu | 930.05 | 928.08 | 933.35 |
| 4 | 1 | torch_cpu_native | 1569.52 | 1539.75 | 1536.12 |
| 4 | 32 | branch_standard | 2667.45 | 2712.41 | 2741.35 |
| 4 | 32 | torch_cpu | 752.49 | 747.85 | 749.49 |
| 4 | 32 | torch_cpu_native | 1745.15 | 1714.08 | 1735.57 |

For this workload, default Torch CUDA is the fastest measured route. Its t1
replicate throughput ratios against the matching standard CPU replicate are
11.35–11.42 at B8 and 34.23–34.34 at B32. The requested CUDA-native configuration
is slower than default Torch CUDA at both batch sizes. Requested CPU-native
configurations improve over default Torch CPU but remain below standard CPU in
every measured cell. These results do not support enabling every native option
by default or claiming a general CPU speedup.

## Actual native correlation execution

Each log contains one successful `ROUTE_PROBE` record, no traceback, and
12 attempts / 12 native successes / zero fallbacks:

| Requested route | Threads | B8 log | B32 log |
| --- | ---: | --- | --- |
| torch_cpu_native | 1 | `probe-torch_cpu_native-t1-b8.log` | `probe-torch_cpu_native-t1-b32.log` |
| torch_cpu_native | 4 | `probe-torch_cpu_native-t4-b8.log` | `probe-torch_cpu_native-t4-b32.log` |
| torch_cuda_native | 1 | `probe-torch_cuda_native-t1-b8.log` | `probe-torch_cuda_native-t1-b32.log` |

These are separate untimed probes. They confirm correlation for the stated
dimensions and thread settings, not native FFT/peak execution or an invocation
count inside each timed worker. B1 deliberately takes the correlation fallback.

## Limits on conclusions

- This compares routes at one revision, not pre-fix/post-fix performance.
  It excludes CLI startup, I/O, waveform generation, chi-square, the stubbed
  sine-Gaussian veto, independent bank-size variation, and scalar Correlator
  performance. CPU t4 and CUDA t1 are distinct configurations.
- Only the first block of the initial pass measures the first API invocation;
  the other two blocks already reuse caches. Constructor allocation is timed
  separately. Fifteen warm block observations per worker do not establish a
  reliable production p99. Pooled bootstrap intervals mix observations within
  workers; three worker-level medians are the relevant between-run evidence.
- Parity checks final-iteration triggers and the final output's aggregate norm,
  not every timed output or pointwise output equivalence. Separate numerical
  tests remain necessary. `expected_snr` records an injection multiplier, not
  a calibrated recovered SNR: data returned as overwhitened are not divided by
  the non-unit PSD. Arrival-time/template recovery was checked independently.
- The GPU was not exclusively allocated: environment snapshots list desktop,
  llama-server and Sunshine processes. The final snapshot shows zero GPU
  utilization, but snapshots cannot establish absence of contention throughout
  the campaign. Report results as observations on this workstation rather than
  an isolated-device benchmark.

## Artifact hashes

- `live-batch-cpu-cuda-t1.json`:
  `3340a8d32e60335e8cedc0522197f302616d32886857f2cad4c8ed9ffca5bb4a`
- `live-batch-cpu-t4.json`:
  `fe598ceaf798793f08b28770d69bfb618b2722cd5bc9a0e0c412198a8bd3e4dc`

Hashes were recomputed read-only on len and match `environment-after.txt`.
