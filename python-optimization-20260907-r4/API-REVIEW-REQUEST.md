# API results and executable review request

Candidate `9e6a688a5190d6e1ddc655fbe352cc206085d5c6` versus baseline `9578a710479b924e882857c4dffab6ed372a634b`. len, Torch 2.13.0+cu130, Python 3.11.9, NumPy 1.26.4, CPU8 (SMT sibling 72), actual Torch intra/inter-op threads 1. Six fresh workers in AB/BA/AB order; seven samples per cell per worker. Shared-host component measurement.

All 108 worker cells passed exact parity and unchanged-input checks, and input/output hashes match across all six workers for every cell. All source, native, input, helper, runtime, affinity, timing-window and inherited-lock validation passed. The real Linux inherited-lock lifecycle test passed with zero skips; pinned threadpoolctl compatibility passed. An independent terminal audit recomputed the complete API summary, verified every evidence hash and unchanged source, found all owned groups inactive, and reacquired the shared lock.

Times below are microseconds per public API call: median of three worker medians, with their observed minimum–maximum. Ratios are baseline/candidate; greater than one means faster. These ranges are observations, not confidence intervals. Raw seven-sample measurements are preserved for every worker.

| Dtype | Elements | Stride | Baseline median [range] µs | Candidate median [range] µs | Ratio |
|---|---:|---:|---:|---:|---:|
| complex64 | 16 | 1 | 16.577 [16.410–17.489] | 17.645 [17.395–18.038] | 0.939× |
| complex64 | 512 | 1 | 21.070 [21.029–22.062] | 22.183 [21.942–22.484] | 0.950× |
| complex64 | 4095 | 1 | 47.635 [47.635–48.771] | 48.585 [48.573–49.205] | 0.980× |
| complex64 | 4096 | 1 | 47.688 [47.626–48.913] | 34.280 [33.687–34.488] | 1.391× |
| complex64 | 4096 | 3 | 61.394 [61.282–62.557] | 34.591 [34.526–35.341] | 1.775× |
| complex64 | 8193 | 1 | 78.249 [78.084–79.188] | 39.501 [39.304–40.018] | 1.981× |
| complex64 | 131072 | 1 | 991.133 [989.835–991.176] | 195.828 [195.336–196.623] | 5.061× |
| complex64 | 131072 | 3 | 1393.700 [1392.903–1395.031] | 212.265 [210.801–212.627] | 6.566× |
| complex64 | 1048577 | 1 | 8331.280 [8193.122–8348.529] | 1802.294 [1795.046–1810.792] | 4.623× |
| complex128 | 16 | 1 | 16.952 [16.761–17.893] | 17.953 [17.854–18.228] | 0.944× |
| complex128 | 512 | 1 | 33.902 [33.785–34.954] | 35.163 [35.015–35.441] | 0.964× |
| complex128 | 4095 | 1 | 151.455 [151.167–152.576] | 152.958 [152.488–153.312] | 0.990× |
| complex128 | 4096 | 1 | 151.811 [151.369–152.656] | 35.194 [35.177–35.833] | 4.314× |
| complex128 | 4096 | 3 | 164.515 [164.375–165.624] | 39.041 [38.638–39.494] | 4.214× |
| complex128 | 8193 | 1 | 287.180 [285.989–288.653] | 42.780 [42.378–43.186] | 6.713× |
| complex128 | 131072 | 1 | 4325.779 [4321.205–4326.948] | 237.036 [236.594–239.435] | 18.249× |
| complex128 | 131072 | 3 | 4705.658 [4704.071–4708.885] | 345.277 [343.777–348.383] | 13.629× |
| complex128 | 1048577 | 1 | 45396.836 [45387.308–50593.852] | 3444.072 [3421.053–3449.866] | 13.181× |

## Cutoff assessment

Retain the existing 4096 cutoff for this candidate. Every measured fast-path cell improves (1.391×–18.249×), with non-overlapping across-worker ranges; at 4096, complex64 gains are 1.391× contiguous and 1.775× stride-3, and complex128 gains are 4.314× contiguous and 4.214× stride-3. This supports the chosen cutoff on this x86 host and tested runtime. It does not establish the optimal cutoff or all x86/Torch versions. No lower-cutoff experiment or patch revision is proposed.

Report the fallback cost explicitly: below 4096, the same legacy expression plus dispatch guards is 0.939×–0.990× as fast, corresponding to approximately 0.95–1.50 µs additional time per call. Some repeat ranges overlap, but the median slowdown is consistent. The whole executable may or may not benefit; component ratios are not executable capacity.

Proceed recommendation: run only the already specified eight-worker executable phase (two instrumented scientific qualifications, then six fresh AB/BA/AB timings) with the unchanged comparator and provenance protocol. Keep the candidate unchanged and report full wall-time results including verification startup and any negative/noisy outcome. No executable launch has occurred. Parent approval must bind the candidate and exact API-summary hash below.

## Evidence hashes

- `api-status.json`: `44763d5599991eb2c11f6080f37fd2ebc63dea5efe24aeceafca3d089aa9b080`
- `api-summary.json`: `a4bf978c0251f1d57785facd65fed9760ec578b22ddfbb9256fea29b617b1c31`
- `api-before.json`: `47bb2d625565b102cc05baa899e2bfa611c54cd099de2b9981fdf5b0cbe64876`
- `api-after.json`: `d1f6c7c878b7b76cb2c69f5804684d49e240df2aab25a72c0c3ee61197eb0fbe`
- `source-staging.json`: `3835233895c6150d9640602a418c639606dae2a0b67d9f517c08c36b11d9d81b`
- `staged-helpers.json`: `9c5db9e364e364281cf7dc21e168229e95447317975a7489cb111b378a5e959f`
- `api-terminal-audit.json`: `f0582f95bb8f645aa521eb87fda938080bd9fdded3ddcb2d9c047fa16abd6fd8`
- `api-remote-evidence.tar.gz`: `0386da033a9575c537a68a4b46b0309fc6abc324f51e9ae03d6ffc5b28c15880`

Full transferred evidence: `api-remote-evidence/`; raw API result, launcher and stage digests (18 files) are all recorded in `api-remote-evidence/api-summary.json`. Every transferred manifest member was verified locally.
