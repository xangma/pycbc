Untimed dispatch observations from the fresh live workload probes.

20/20 cells qualified. Each counter below is **successes / attempts**, not a performance ratio. Raw counters and errors are retained in the JSON summary.

| Head | Route | Threads | Batch | CPU correlation | CUDA correlation | CPU peaks | CUDA peaks | FFTW single | FFTW batch | MKL IFFT | On-device helper excluded | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| main | torch_cpu | 1 | 8 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 1 | 32 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 8 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 32 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 8 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 32 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 8 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 32 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 8 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 32 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 8 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 32 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 8 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 32 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 8 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 32 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 8 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 32 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 8 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 32 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |

Requested configuration (effective flags and their selection, distinct from actual admissions):

| Head | Route | CPU correlation | CUDA correlation | CPU peaks | CUDA peaks | FFTW batch | MKL IFFT | On-device peaks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | torch_cpu | off (package_default) | not applicable | not recorded on head | not applicable | off (package_default) | on (package_default) | off (package_default) |
| main | torch_cpu_native | on (explicit) | not applicable | not recorded on head | not applicable | on (explicit) | on (package_default) | off (package_default) |
| cpu | torch_cpu | off (package_default) | not applicable | off (package_default) | not applicable | off (package_default) | on (package_default) | off (package_default) |
| cpu | torch_cpu_native | on (explicit) | not applicable | on (explicit) | not applicable | on (explicit) | on (package_default) | off (package_default) |
| main | torch_cuda | not applicable | off (package_default) | not recorded on head | off (package_default) | not applicable | not applicable | off (package_default) |
| main | torch_cuda_native | not applicable | on (explicit) | not recorded on head | on (explicit) | not applicable | not applicable | off (package_default) |

Fallback and exception observations (unrecorded causes remain unknown):

| Probe | Helper | Fallbacks | Exceptions | Recorded reason |
| --- | --- | ---: | ---: | --- |
| probe-main-torch_cpu-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b8.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b32.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b8.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b32.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b8.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b32.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |

Interpretation limits:

- These are instrumented dispatch observations, not timing or speed measurements. Child-driver timings are excluded.
- Success means the observed helper returned its admission/success signal; attempts include rejected admissions. Only the named helpers are observed.
- 0/0 means an available helper was not called. It does not by itself establish that a route was disabled or that a downstream fallback ran.
- Unavailable means the symbol is absent on that source head; its attempt count is unknown, not zero.
- The probe records fallback counts but does not instrument internal rejection reasons. Missing reasons remain explicitly unrecorded; requested flags are not substituted for observed reasons.
- Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks. Requested features can still be bypassed or rejected.
- The on-device peak helper exclusion requires the probe assertion, a disabled effective flag, and an available helper with zero observed attempts.
- All cells use FFT length 131072 and batches 8 or 32. Batch 1 is not included in this admission matrix.

Source revisions:

- main: `607bce53ead14f12af32552a5b2441d3bc667267`.
- cpu: `1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c`.
- Probe source SHA-256: `792ea7cb4b90bca5fb6e9dac65c5b762d905f7f3c240a95ff18698232fa66b32`.

Input hashes:

- `probe-main-torch_cpu-t1-b8.json`: `b12efe84b0fcfe0dec93f398d05540535f05b193c8bad90587b10ef452bd36a9`.
- `probe-main-torch_cpu-t1-b32.json`: `4e8350f5875426b3ce314a5b16855da57f2f117c1e07b81d3376768f0b19e7cb`.
- `probe-main-torch_cpu_native-t1-b8.json`: `370c76d157c52d684e9ed4f94fdb1cbe2f3b3365b46af528da4de883fd45075f`.
- `probe-main-torch_cpu_native-t1-b32.json`: `1af7ce0b72aa5246b8957ce80ecce38f87ebdb9b0f0a274b88b1ee582096f50d`.
- `probe-main-torch_cpu-t4-b8.json`: `61b80bb89d91e4b1eb18b90fe4ac9bf3c3823f2e58c5e55ffe8abdd9882d5052`.
- `probe-main-torch_cpu-t4-b32.json`: `43993c06b6abd0579429b05339d16c39c5819ed906804b845824d0bad185d218`.
- `probe-main-torch_cpu_native-t4-b8.json`: `cd829a4b001c688686f79e419752d09a84bef5d2c639142d92ed6584f56eef34`.
- `probe-main-torch_cpu_native-t4-b32.json`: `c1ae3d895a235c95fd87b2b5b7b74569fa7525fb732623b8d294bc0a5ac1fd4e`.
- `probe-cpu-torch_cpu-t1-b8.json`: `21b61385c79fbca4a92a8ff9a73693a12ece5056bf62ac48523c3e5f168af964`.
- `probe-cpu-torch_cpu-t1-b32.json`: `004e8616431bb4f9e5070a29cb4ad6a312d6d33a2d01c5902e4adc36ba9562ec`.
- `probe-cpu-torch_cpu_native-t1-b8.json`: `2206c4cf8593326ab57c6a0bff81baa15b2bce99396583e98cba464c4ecacac2`.
- `probe-cpu-torch_cpu_native-t1-b32.json`: `7c293ff72ceddde33c794f02eb36ee82bb0fe3f4ceb218699842806a93f6a552`.
- `probe-cpu-torch_cpu-t4-b8.json`: `49a7d09672c8b0ef9aa3ea1c8bcd1d776bb5bf11c1b87b1d64de5adc0b929386`.
- `probe-cpu-torch_cpu-t4-b32.json`: `c224758076d3a3dea1b6235589c4c12616ea0b32fd2d40c51a87d53acf659abc`.
- `probe-cpu-torch_cpu_native-t4-b8.json`: `e99c65779579b0b24a3fe71df2947ad31f2b5cf1634c14beb30ecac3b1e5b76e`.
- `probe-cpu-torch_cpu_native-t4-b32.json`: `8a3491996de87a2beba1d1d8ece3d842fb1a6c2a2a2f3a8c958290b9d899c6b1`.
- `probe-main-torch_cuda-t1-b8.json`: `a0baf7b3659c3e17e601822d92443d9d81ebe9fefb1717b7cb0768c65e67570a`.
- `probe-main-torch_cuda-t1-b32.json`: `f14f54c6e006a32adf5ac005dc172fcab792e16943da950023df2c3e8e09ecd1`.
- `probe-main-torch_cuda_native-t1-b8.json`: `a7da4b9738c5d9e033014fd36e86f87480cd4919b0f45a8e7d0f725d3f8a9a14`.
- `probe-main-torch_cuda_native-t1-b32.json`: `1a5d143030eae47b1c0a9ed7979fb12285c53ff9919fbd2ed3ca5384e7225247`.
