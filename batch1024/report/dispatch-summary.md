Untimed dispatch observations from the fresh live workload probes.

60/60 cells qualified. Each counter below is **successes / attempts**, not a performance ratio. Raw counters and errors are retained in the JSON summary.

| Head | Route | Threads | Batch | CPU correlation | CUDA correlation | CPU peaks | CUDA peaks | FFTW single | FFTW batch | MKL IFFT | On-device helper excluded | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| main | torch_cpu | 1 | 1 | 0/0 | 0/0 | unavailable | 0/0 | 12/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cpu | 1 | 8 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 1 | 32 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 1 | 128 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 1 | 512 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 1 | 1024 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 1 | 0/12 | 0/0 | unavailable | 0/0 | 12/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cpu_native | 1 | 8 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 32 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 128 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 512 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 1 | 1024 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 1 | 0/0 | 0/0 | unavailable | 0/0 | 0/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cpu | 4 | 8 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 32 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 128 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 512 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu | 4 | 1024 | 0/0 | 0/0 | unavailable | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 1 | 0/12 | 0/0 | unavailable | 0/0 | 0/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cpu_native | 4 | 8 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 32 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 128 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 512 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cpu_native | 4 | 1024 | 12/12 | 0/0 | unavailable | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 1 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/0 | 12/12 | verified | qualified |
| cpu | torch_cpu | 1 | 8 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 32 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 128 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 512 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 1 | 1024 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 1 | 0/12 | 0/0 | 12/12 | 0/0 | 0/0 | 0/0 | 12/12 | verified | qualified |
| cpu | torch_cpu_native | 1 | 8 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 32 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 128 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 512 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 1 | 1024 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 1 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/0 | 12/12 | verified | qualified |
| cpu | torch_cpu | 4 | 8 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 32 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 128 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 512 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu | 4 | 1024 | 0/0 | 0/0 | 0/12 | 0/0 | 0/0 | 0/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 1 | 0/12 | 0/0 | 12/12 | 0/0 | 0/0 | 0/0 | 12/12 | verified | qualified |
| cpu | torch_cpu_native | 4 | 8 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 32 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 128 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 512 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| cpu | torch_cpu_native | 4 | 1024 | 12/12 | 0/0 | 12/12 | 0/0 | 0/0 | 12/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 1 | 0/0 | 0/0 | unavailable | 0/12 | 0/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cuda | 1 | 8 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 32 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 128 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 512 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda | 1 | 1024 | 0/0 | 0/0 | unavailable | 0/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 1 | 0/0 | 0/12 | unavailable | 0/12 | 0/12 | 0/0 | 0/12 | verified | qualified |
| main | torch_cuda_native | 1 | 8 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 32 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 128 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 512 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |
| main | torch_cuda_native | 1 | 1024 | 0/0 | 12/12 | unavailable | 12/12 | 0/0 | 0/12 | 0/0 | verified | qualified |

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
| probe-main-torch_cpu-t1-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t1-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu_native-t1-b1.json | CPU correlation | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu_native-t1-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b1.json | FFTW single | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu-t4-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu_native-t4-b1.json | CPU correlation | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu_native-t4-b1.json | FFTW single | 12 | 0 | not recorded by probe |
| probe-main-torch_cpu_native-t4-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b1.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b8.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b32.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b128.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b512.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b1024.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t1-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu_native-t1-b1.json | CPU correlation | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b1.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b8.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b32.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b128.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b512.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b1024.json | CPU peaks | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu-t4-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-cpu-torch_cpu_native-t4-b1.json | CPU correlation | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b1.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b1.json | FFTW single | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b8.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b32.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b128.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b512.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b1024.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda-t1-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b1.json | CUDA correlation | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b1.json | CUDA peaks | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b1.json | FFTW single | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b1.json | MKL IFFT | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b8.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b32.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b128.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b512.json | FFTW batch | 12 | 0 | not recorded by probe |
| probe-main-torch_cuda_native-t1-b1024.json | FFTW batch | 12 | 0 | not recorded by probe |

Interpretation limits:

- These are instrumented dispatch observations, not timing or speed measurements. Child-driver timings are excluded.
- Success means the observed helper returned its admission/success signal; attempts include rejected admissions. Only the named helpers are observed.
- 0/0 means an available helper was not called. It does not by itself establish that a route was disabled or that a downstream fallback ran.
- Unavailable means the symbol is absent on that source head; its attempt count is unknown, not zero.
- The probe records fallback counts but does not instrument internal rejection reasons. Missing reasons remain explicitly unrecorded; requested flags are not substituted for observed reasons.
- Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks. Requested features can still be bypassed or rejected.
- The on-device peak helper exclusion requires the probe assertion, a disabled effective flag, and an available helper with zero observed attempts.
- All cells use FFT length 131072 and batches 1, 8, 32, 128, 512, or 1024. The full logical batch and allocated FFT buffer sizes are recorded.

Source revisions:

- main: `4885b64560e9f39b740e85b6a976898869dd360e`.
- cpu: `d544420232428225c214a4be84fbe1262a6d307b`.
- Probe source SHA-256: `ffb110051dfd0dcf2d32c35a39f1c9b20cfd96c0fef9988c4b87d08d279a9171`.

Input hashes:

- `probe-main-torch_cpu-t1-b1.json`: `789cc8909b0802ed7a16abd4611fc97afc2d74d4116d5adba18bb53265874e8f`.
- `probe-main-torch_cpu-t1-b8.json`: `4ea866bfca0ae17a3203a0db905a8adc2fb3c204bed881c5357845a23c117b05`.
- `probe-main-torch_cpu-t1-b32.json`: `45530b81d75126ee2ffe08d264cfd63979fc1716522c49da7fb286e94fc801cb`.
- `probe-main-torch_cpu-t1-b128.json`: `83694cbfe347294ea4a2300edbedd40eccb7df09bad5779dd428d8d74687a13c`.
- `probe-main-torch_cpu-t1-b512.json`: `534cfbcd410ab383834bcbdfe87476d16c3748255e12ad14e445419c8045896c`.
- `probe-main-torch_cpu-t1-b1024.json`: `08c6a2839cc4bc829c0645ee93dec9f29bf7099624a45a52152e6c19c948746c`.
- `probe-main-torch_cpu_native-t1-b1.json`: `da5fdf5f84730a15ca62239f6989a36ca42876e26adba48a331dce17127296cf`.
- `probe-main-torch_cpu_native-t1-b8.json`: `1d4185e833c6d29f74999e74bbb9fa86cc03fb4b84e73818124a432beb991b60`.
- `probe-main-torch_cpu_native-t1-b32.json`: `93eb4a85a6e4b5e91174c8e40cb9679025307614219c44812661a54f701c58b2`.
- `probe-main-torch_cpu_native-t1-b128.json`: `2e287cc57faa9f00cde657723c70d0d4c83bf785de88458c7ae232d64c9cd2ae`.
- `probe-main-torch_cpu_native-t1-b512.json`: `0ca1bf23d528383fbe284d6fc47f98e0cf05a7c3740a546f520066e4b66f7f9b`.
- `probe-main-torch_cpu_native-t1-b1024.json`: `04c2223b6787c1a84960765c23e75c13cf173c41ebaf86b36c22e94aba93d006`.
- `probe-main-torch_cpu-t4-b1.json`: `d570581d604d6b58e5bb89e173b56e9841c5529e6e278125aa1007512435c216`.
- `probe-main-torch_cpu-t4-b8.json`: `c4bff37c4cf2e54626029ff600709fda1d96594074e50784a3086149149daf7e`.
- `probe-main-torch_cpu-t4-b32.json`: `229df2bf483f2a5d2597740fbe9fbc1d24cd949a3f231026ec71a183e4cb0264`.
- `probe-main-torch_cpu-t4-b128.json`: `e9b35eadd8ed9314011a55f85760672d58bf35149d220f7454e2bfc43bdd9365`.
- `probe-main-torch_cpu-t4-b512.json`: `950c1333ad9f758ab481d43fd585d06566c3b36795a581752d8ca4bb10d07010`.
- `probe-main-torch_cpu-t4-b1024.json`: `1e3e5a6f22ad92df828a0216a3a6d9d908eaa7c0dfb44c69aa4e33d8649e0a88`.
- `probe-main-torch_cpu_native-t4-b1.json`: `3c9a1b69472b08e564b042f01254459d89014e4b40f0c2f56541e00625e33a14`.
- `probe-main-torch_cpu_native-t4-b8.json`: `90d596b94879be7f6f7d56a9328ce35863561c0b653d2630172d099c3df7f749`.
- `probe-main-torch_cpu_native-t4-b32.json`: `abc47f4572ddc28adc911868af86c505033d2fd2b1b095f6d2fa3a77946d0ff7`.
- `probe-main-torch_cpu_native-t4-b128.json`: `dc60bda2af225157680e6a8ee9df141b45c29e8e4db60d2949928a24da9b2dbb`.
- `probe-main-torch_cpu_native-t4-b512.json`: `2fccc0203d7991b314ae69fd9d8933c8a4b0f6995a2df94d94dfb3d7cc6dcac5`.
- `probe-main-torch_cpu_native-t4-b1024.json`: `42accce513bc1f67b6a16098eaf3f2724744bf53528732cfb9af010493a007a6`.
- `probe-cpu-torch_cpu-t1-b1.json`: `d4fd76e063215f1c0e94e124cab166e6fde1e2fd315178e6c6c9854f0959372a`.
- `probe-cpu-torch_cpu-t1-b8.json`: `ae24b9171e56e81d14992ebb030580177ec72a936ffe29616cdb8bcdb3b2d6e2`.
- `probe-cpu-torch_cpu-t1-b32.json`: `c79c7746cce3ea248bf52a67aaf34124a28c806bd4cf7f73fff9580136de5ae5`.
- `probe-cpu-torch_cpu-t1-b128.json`: `f0f5563cfd1aabf6ee50ccc316582c4ecfabe66cd48b18391d4f3adee3d13d1c`.
- `probe-cpu-torch_cpu-t1-b512.json`: `ad470e86086aa9474cc42ce16d1b1fc4a506b4f932f47200b28b7bbc6d28a5a6`.
- `probe-cpu-torch_cpu-t1-b1024.json`: `156efc35aff61747a569246042785aa2d6d331988a1e7a952ddbfc16bf88eaea`.
- `probe-cpu-torch_cpu_native-t1-b1.json`: `fb8adeb841b3ef212963713e9162fe01befad5d3c5f509215b45b5fc3893945d`.
- `probe-cpu-torch_cpu_native-t1-b8.json`: `da702b4845bb5dfa16609dd9221bf882ecb7ae2413cab659e813688e421c4827`.
- `probe-cpu-torch_cpu_native-t1-b32.json`: `43e3273fc6e04f2265737825c372117dc3fee43a4bc7c0fe2b7a991e012c2d51`.
- `probe-cpu-torch_cpu_native-t1-b128.json`: `516c4bf4e593ed28bbbed736bfeca5affe9fb36956cab5da1b3f4de7bdd1347f`.
- `probe-cpu-torch_cpu_native-t1-b512.json`: `7ec08d346b41751b278a9d3766b22a97d8af3adc6a552f0150d52b3c4ea69191`.
- `probe-cpu-torch_cpu_native-t1-b1024.json`: `77d6523af879cc0ed90a2ff3ce8d76c538bc6d6d5edadb80a49ef22b00caee19`.
- `probe-cpu-torch_cpu-t4-b1.json`: `7cd550ce0920fdea7dd55a6181163640286c9eacf073169ed8b6d5d908471943`.
- `probe-cpu-torch_cpu-t4-b8.json`: `91c9f86d4dbb777689435e7ceee204c82514e3b00460c0a206ce8776aa6c0a19`.
- `probe-cpu-torch_cpu-t4-b32.json`: `73d4b27183d47835677deccb8a5e5d42689e1b299058d26b468e24c0030a5a60`.
- `probe-cpu-torch_cpu-t4-b128.json`: `2f8c10213dc56bf699aea44819d7daba969deaaacc67215560e16d282d4ca29e`.
- `probe-cpu-torch_cpu-t4-b512.json`: `4770c95d93d0a7d46cd58716378e5afdf87371e49bd3aed363532dbbe0b49fc6`.
- `probe-cpu-torch_cpu-t4-b1024.json`: `253b75048a3b88ecfe15539e952fe5cef3429f4a455fa35a2a9bc48cb765beb5`.
- `probe-cpu-torch_cpu_native-t4-b1.json`: `ab7add958e41d600e541a3d99358b0d0094e5eacc1bf778b354ef66528fcbbf9`.
- `probe-cpu-torch_cpu_native-t4-b8.json`: `2f99e39ca123ebe965b1db73d4d9e04f570cb548ba74b8e4d1c47013fb922246`.
- `probe-cpu-torch_cpu_native-t4-b32.json`: `6f98c15f3f3dbd0de455c90fd2696b30e079ccdc067eff461825e43e52ed4f98`.
- `probe-cpu-torch_cpu_native-t4-b128.json`: `b43cac8a027cca82d1371ce8d45ef20e6b54f5d26fc0f6b6a3778218fc459728`.
- `probe-cpu-torch_cpu_native-t4-b512.json`: `5df3c4d1b792c0d89eb9cbbe3aabb6b9be339778d9d3453f6763a96ac5e905dc`.
- `probe-cpu-torch_cpu_native-t4-b1024.json`: `11e2920e5efe242bdd922768fd5682a947700d7ba83148bad1988a0f37aa4e51`.
- `probe-main-torch_cuda-t1-b1.json`: `9f81d9a9a835c2da240653900492e2c545fac65d5906d2281b0ec78b8a222086`.
- `probe-main-torch_cuda-t1-b8.json`: `87358c99b5958f9315a13455104f825bd33374e6a2f087769af67e85d6145f08`.
- `probe-main-torch_cuda-t1-b32.json`: `a36ccfb8d4776d4c57c2d24eac2cfcf46d6509f2394e5ec797efe6f6a32ed5ab`.
- `probe-main-torch_cuda-t1-b128.json`: `b64368d5be34463e0b21182ee026105dd148e460b154c70ab996e6023390cdbf`.
- `probe-main-torch_cuda-t1-b512.json`: `5dad36cb76ded6e34893d8037ae73f84038a3007089f397f59327b534147186d`.
- `probe-main-torch_cuda-t1-b1024.json`: `86ddae6a427d55b194bfd55574bf753756859c11cf43a8f26cf4f7821d7e8b56`.
- `probe-main-torch_cuda_native-t1-b1.json`: `6c55df1fef97ae16e3be67cb20c660f5b328f7b8d061f6b683cff4ae46eeea4c`.
- `probe-main-torch_cuda_native-t1-b8.json`: `c2f4a6fc31a460687084411520f98368bc8def96125d9212e23ac59199c8ee30`.
- `probe-main-torch_cuda_native-t1-b32.json`: `2258e7ca01f38bd944f1e15cb24a8aaa6a6f8efd59c99cdd8f00a221ecb27818`.
- `probe-main-torch_cuda_native-t1-b128.json`: `ff2e564c2a3d82777dae9ac3b4602ea499bce9716f30a4328db1c50ab03a39c2`.
- `probe-main-torch_cuda_native-t1-b512.json`: `abb81b7bb9046bc52c2a395eead48753f9c6f58959af0ecc14cce688fc5623b8`.
- `probe-main-torch_cuda_native-t1-b1024.json`: `050afa6d4ddba86ed755249a593fb268ebf03f8424a802fef3549293d84a5ad4`.
