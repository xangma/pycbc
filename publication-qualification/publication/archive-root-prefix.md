# Optimized single-thread pycbc_inspiral reference — 6 September 2026

The primary comparison uses the complete executable, a compressed 96-template
BNS/NSBH workload, one host core and 512-second FFTs with 112/16-second padding.
Each backend has three fresh unprofiled processes with identical inputs.

| Backend | Median full-process wall time | Templates/core at real time |
|---|---:|---:|
| Normal CPU / MKL | 32.30 s | 5659.49 |
| Torch CPU | 44.54 s | 4103.78 |
| Torch CUDA + one GPU | 20.32 s | 8994.65 |

Torch CPU improves from 80.63 to 44.54 seconds (1.81×), using compiled CPU
interpolation and qualified double-precision MKL inverse FFT workspaces.
All ten optimized report gates pass, including 18 strict trigger comparisons,
288 waveform/PSD checks, 36 boundary injections, 576 compressed-bank parity cases
and the 108-case IFFT matrix. Source is `a4d77a6d1863c0515e8dace64c5609b63d40b51e`;
the before-source results remain separate.

[Primary report and figures](inspiral-reference-20260906/README.md),
[reproduction and restoration](inspiral-reference-20260906/REPRODUCE.md) and
[profile interpretation](inspiral-reference-20260906/profile-interpretation-v6.md)
accompany the complete evidence. Earlier
[component measurements](performance-fix/README.md) and the campaigns below
retain their original source identities and scope.

---

