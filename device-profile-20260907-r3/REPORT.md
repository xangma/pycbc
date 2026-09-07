# Integrated Torch profiling pass: 7 September 2026

The remaining Torch CPU cost is dominated by the promoted double-precision MKL FFT and its two conversion copies per transform. The optimized squared norm now takes about 1.45 seconds in the filtering profile. Trigger-result deep copying is a smaller, bounded Python target. The CUDA trace continues to identify chi-square as the largest recorded device cost.

## Scope and provenance

The measured source is clean `d2647addb884ead3249914ebc980f3c132076d93`, in a fresh real directory on `len`: `/home/xangma/pycbc-torch-profile-20260907-r3/source`. Its `pycbc/` and `bin/` trees match the integrated squared-norm commit `9e6a688a5190d6e1ddc655fbe352cc206085d5c6`. The later documentation-only publication is outside this frozen acquisition. No production or kernel source was changed by this task.

The workload has 384 distinct compressed IMRPhenomD templates (256 BNS, 128 NSBH), five 2,097,152-point FFT segments per template, and 1,904 unique valid detector seconds. Sampling is 4,096 Hz, the low-frequency cutoff is 30 Hz, segment/start/end geometry is 512/112/16 seconds, and power chi-square uses 16 bins. The executable does not request sine-Gaussian chi-square. `remote-evidence/config.json` records the complete CLI and environment.

Every numerical worker is pinned to logical CPU 8, whose SMT sibling is 72, with native numerical pools limited to one thread. Torch workers also verify one intra-op and one inter-op thread. CUDA uses one RTX 4090. The host and sibling are unreserved; this remains a shared-host finite-workload diagnostic. The inherited shared flock serializes this campaign with the cooperating optimization task, but does not reserve the machine.

Normal CPU uses `cpu:1` and MKL. The checking wrapper imports Torch explicitly only for Torch routes. PyCBC itself imports optional Torch on the normal CPU path; normal CPU receipts therefore report Torch present, with wrapper-controlled Torch thread fields null. Observed native pools still satisfy the one-thread gate.

The bank SHA256 is `26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f`; the frame SHA256 is `580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865`. The executable SHA256 is `d4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a`. All 11 native extension hashes and their unchanged build-input provenance are in `remote-evidence/native-provenance.json`. Ordinary worker receipts hash imported native extensions and selected Python source files; they do not enumerate every imported Python module. The complete source snapshot is archived and verified separately.

## Python and native attribution

Full-process cProfile includes startup and the checking wrapper. Filtering cProfile starts at the first bank lookup and stops after event consolidation, before performance/output writing. All six profiles cover the qualified 384-template workload. The filtering profiles total 51.065916 seconds for normal CPU, 96.890385 for Torch CPU, and 9.043454 for CUDA. These independent instrumented runs are not throughput samples.

| Filtering operation | Normal CPU seconds | Torch CPU seconds | Calls and ownership |
| --- | ---: | ---: | --- |
| FFT execution self time | 28.4065 | 62.5173 | 1,920; native execution is charged to the Python/ctypes entry point |
| Copies inside FFT execution | — | 8.6989 | 3,840 `Tensor.copy_` calls; two per promoted transform |
| Point chi-square | 7.8446 | 7.8691 | 1,309 native CPU calls |
| Threshold wrapper self time | 3.0000 | 3.2873 | 1,920; both routes invoke native CPU thresholding |
| Decompression wrapper self time | 1.6822 | 1.7042 | 384; both routes invoke the native interpolation kernel |
| Squared norm cumulative time | Opaque native attribution | 1.4526 | 765 Torch calls; two squares total 1.0669 seconds |

Torch FFT execution has 71.2541 cumulative seconds, including the 8.6989 copy seconds. These values overlap. The full profile independently assigns 8.6351 seconds to the same 3,840 FFT copies, and just 0.1794 seconds to the remaining 383 array-copy calls. Copies cannot be removed as an ordinary allocation fix: the source intentionally promotes complex64 input to complex128 workspaces and copies the result back for the corrected precision route.

Normal CPU `scheme._scheming_function` self time includes Cython work that cProfile does not expose as separate rows. It is not a measurement of pure Python dispatch. Torch `squared_norm` self time similarly includes unexposed native/operator work; the visible `numel`, `is_complex`, and `is_conj` calls total only about 0.0017 seconds in the full profile.

Recorded caller edges locate full-profile deep copying at `bin/pycbc_inspiral:304`: `template_triggers` calls it 1,309 times, accounting for 1.1884 cumulative seconds. Total recursive `deepcopy` time is 1.2493 seconds, including 5,236 tensor deep-copy calls with 0.5025 cumulative seconds. Filtering-only total recursive deep-copy time is 1.1610 seconds. Recursive cumulative rows must not be added. The first template's enclosing function was entered before loop profiling was enabled, so its parent caller edge is incomplete; the bank-coverage receipt remains authoritative for workload coverage.

The full cProfiles also charge roughly 6.7 seconds per backend to frame APIs. MKL descriptor creation takes 4.1721 seconds on normal CPU and 2.5207 seconds on Torch CPU (self time). These are full-process attribution observations; subtracting separate full and filtering profiles would not isolate startup reliably.

Linux `perf` records 199-Hz user-cycle samples with DWARF call stacks and monotonic timestamps. Separate full-process and exactly clipped filtering-window reports are retained. The following sums use only displayed symbols at the 0.5% report threshold; percentages are rounded event-period shares, not wall-time fractions.

| Native filtering symbols | Normal CPU | Torch CPU |
| --- | ---: | ---: |
| MKL FFT | 57.62% (`32fc`) | 66.04% (`64fc`) |
| Torch copy kernels | — | 9.50% |
| Point chi-square | 16.09% | 8.28% |
| Threshold | 5.71% | 3.26% |
| Normal CPU correlation | 4.41% | — |
| Decompression | 3.15% | 1.67% |

The filtering reports record zero lost samples. The full Torch CPU report emitted five `addr2line` first-record warnings; its filtering report emitted none. Raw data and stderr are retained, so symbolization limitations are not hidden.

## CUDA device attribution

Trace SHA256: `df8d42fac08fc6f97a20b1a8c9b7fa4b1639b5b1e74fb9a5d28d86a71e223ea9`. The instrumented loop lasts 10.690923 seconds and contains all 162,006 recorded device events. Summed and union device activity both equal 1.864559 seconds in this trace.

| Kernel ownership | Recorded seconds |
| --- | ---: |
| Chi-square | 1.377272 |
| Bank lookup | 0.217452 |
| IFFT | 0.101767 |
| Correlation | 0.046151 |
| Threshold | 0.026126 |
| Squared norm | 0.016205 |
| Unattributed kernels | 0.040779 |

The 1,309 host chi-square ranges total 3.249361 seconds, including 1,955 `cudaStreamSynchronize` calls totaling 0.993084 seconds. Host ranges and device durations can overlap. An additional 0.008542 seconds of device copies remains unattributed. Ownership uses explicit external/correlation IDs and containing semantic ranges, without timestamp-only guessing. Recorded device activity is not GPU occupancy or utilization, and the difference from loop duration is not proven GPU idle time. `device-attribution.json` preserves every category, ownership path, event count, and available transfer-byte field.

## Ranked optimization handoff

1. **Promoted MKL FFT and conversion passes:** the largest Torch CPU cost. Any future change must preserve the corrected numerical route and belongs to a separately reviewed FFT effort; the current Python pass leaves it intact.
2. **Trigger-result deep copying:** approximately 1.19 cumulative seconds in the full Torch CPU profile, localized to one executable call site. A bounded candidate can investigate whether a faithful copy implementation preserves arrays, metadata, aliases, and mutation independence. This profile establishes opportunity, not a validated optimization.
3. **CPU squared norm:** about 1.45 cumulative seconds after the integrated optimization. Small eligibility-guard changes have little measured headroom; native/operator work dominates the remaining profile.
4. **CUDA chi-square and host synchronization:** dominant recorded device ownership, with substantial host synchronization. This motivates investigation of that call path, not a change to CUDA squared norm.
5. **Frame I/O and initialization:** a common cost in this small workload. Startup amortization and caching require separate scientific and resource-lifetime analysis.

## Fresh unprofiled timings

Three fresh workers per backend ran after profiling on the same clean source and qualified inputs. Each full-process clock starts immediately before launching the checked worker and stops at child exit; the runtime wrapper is included. The numerator is 384 × 1,904 = 731,136 template-seconds. No instrumented run is used as a throughput sample.

| Backend | Three wall times (s), acquisition order | Median (s) | Observed min–max (s) | Template-seconds / wall-second |
| --- | --- | ---: | --- | ---: |
| CPU / MKL | 70.219288, 69.919940, 70.271936 | 70.219288 | 69.919940–70.271936 | 10,412.18 |
| Torch CPU | 114.163224, 114.461583, 114.032041 | 114.163224 | 114.032041–114.461583 | 6,404.30 |
| Torch CUDA | 24.989981, 25.009828, 25.016771 | 25.009828 | 24.989981–25.016771 | 29,233.95 |

The observed range is 0.501% of the CPU median, 0.376% for Torch CPU, and 0.107% for CUDA. With only three samples on a shared host, these ranges are descriptive, not confidence intervals or evidence of long-term stability. This pass does not measure an old-versus-new executable speedup.

The following accounting uses the actual median-wall worker for each backend. Setup plus post-setup internal time plus time outside the internal clock equals full-process wall time. The residual includes process/wrapper startup and shutdown and work outside PyCBC’s internal timer; it cannot be labeled pure startup. Post-setup internal time is also not identical to the separately instrumented filtering scope.

| Median-wall worker | Setup (s) | Internal time after setup (s) | Outside internal clock (s) | Full wall (s) |
| --- | ---: | ---: | ---: | ---: |
| timing-1-cpu | 15.323808 | 50.261342 | 4.634138 | 70.219288 |
| timing-1-torchcpu | 15.143020 | 94.537085 | 4.483119 | 114.163224 |
| timing-2-cuda | 13.424875 | 7.011554 | 4.573399 | 25.009828 |

For the CUDA median worker, setup and the outside-clock residual account for 71.96% of full wall time. That makes workload size and startup amortization material to interpretation; the throughput above uses the full wall boundary.

## Validation and handoff

All three backend qualifications passed, covering all 384 template identities, five segments, and 1,904 valid seconds. All 21 scientific workers completed successfully and yielded the same 1,991 H1 trigger identities. Local replay against the fresh normal-CPU qualification passed all 20 other outputs, including all nine timing files. The frozen comparator checks identity and configuration exactly and retains its original numeric budgets: relative tolerance 1e-4, absolute tolerance 1e-5, sigma-squared relative tolerance 1e-5, and circular phase absolute tolerance 1e-4 radians.

The previous corrected normal-CPU output also passed locally with zero numeric differences. That comparison explicitly substitutes the authorized clean revision pair and normalizes the byte-identical executable’s path; the raw strict metadata comparison and the two substitutions remain recorded in `previous-parity-local.json`. No numerical tolerance was changed.

Linux lifecycle tests passed 9/9 with zero skips, including actual perf/time/taskset inheritance of the lock after supervisor/wrapper exit. Both runtime tests passed, exercising 18 route/failure combinations. All 15 frozen comparator fixtures passed. Remote Torch is 2.13.0+cu130 with CUDA 13.0. The raw CUDA summary was reproduced exactly from the copied trace. Stage, helper, source/native import, input, HDF, and receipt hashes passed the terminal and local evidence audits.

The final audit at 2026-09-07 20:38:55 UTC independently reacquired the shared lock and found all 52 owned science/export groups inactive. The optimization task received explicit release after the copy and local replay completed. No further remote job belongs to this profiling pass.

The full source/native snapshot, six raw pstats files, two perf data files and their full/window reports, CUDA trace, all HDF outputs, acquisition helpers, and run receipts are in `remote-evidence/`. The reproducible summaries and export records are siblings of this report.

| Evidence | SHA256 |
| --- | --- |
| `evidence.tar.gz` (224,392,604 bytes) | `a86204f5163c40f9ce011406cbc9f025209b48bbc9d95b18d0784bdce4772375` |
| `remote-evidence/profile-status.json` | `e8767efdcf7470f403b2665e2bccef18b885d780e09223d48e224cee67cb78fa` |
| `remote-evidence/terminal-audit.json` | `3617d50a45f6f09693b608d5318a6ba2b1b51cbf20e28de0292585338c1c779b` |
| `release-audit.json` | `dc31f26501b97d256705928df1c129a79986801341e51b2ab1f8d74b5193333c` |
| `evidence-verification.json` | `8b02ff2ef4df7546226d5c63a6c2e47b93568eb39138675a8a0e2abff678c59d` |
| `previous-parity-local.json` | `711506d810225b5010f58580bf9fef31f846915ad43c9c36b9251905d9bd7d61` |

Follow [README.md](README.md) for portable restore and offline replay commands. In this report, `remote-evidence/` means the restored `pycbc-torch-profile-20260907-r3/` directory. The original local verification helper is preserved as `verify-evidence.original.py`; its acquisition-machine paths are not portable. `attribution-early-full.json`, `attribution-early-loops.json`, `attribution-focused-callers.json`, and `native-attribution-summary.json` retain the detailed profile rows and caller edges used above. This publication copy clarifies source-hash coverage and replay instructions; the acquired report and the original `handoff.json` and `evidence-verification.json` remain unchanged in the acquisition directory.
