# Torch against the standalone corrected CPU baseline

All four executable qualifications and sixteen timed processes completed on 8 September 2026. All five cross-route trigger comparisons pass the unchanged budgets, and every timed output passes against its own qualification. **Full-PSD equivalence still fails below 30 Hz.** The times below are descriptive execution costs; they do not establish full-equivalence speedup.

This comparison separates the [CPU precision correction review](../cpu-review.md) from Torch support. The measured baseline is CPU `66789ac4a7468094b0cc3ca1498a1de67e0311f6`; the rebuilt main is `f582b6fd250d0b82612492979e01e645d5c07afc`. The optional FFT/CPU follow-up PRs are excluded. The source bundle preserves both measured heads, independent of later PR updates.

| Route | Allocation | Median seconds | Observed range, seconds | Template-seconds per wall-second |
| --- | --- | ---: | ---: | ---: |
| Standalone corrected CPU | 1 core/thread | 68.238 | 68.174–68.487 | 10,714.4 |
| Restacked normal CPU | 1 core/thread | 65.472 | 65.370–65.672 | 11,167.1 |
| Torch CPU | 1 core/thread | 102.446 | 102.207–102.728 | 7,136.8 |
| Torch CUDA | 1 core/thread + RTX 4090 | 20.324 | 20.298–20.353 | 35,973.9 |

Each route has four fresh unprofiled samples, scheduled in rotating orders ABCD, BCDA, CDAB and DABC. All samples are retained. The fixed work is 384 compressed templates over 1904 unique H1 seconds: 731136 template-seconds, five segments and 1920 template/segment pairs. Ranges are observed minima/maxima, not confidence intervals.

## Scientific results and the timing decision

Every route produces 1991 triggers. Exact identity/sample-tick and degrees-of-freedom checks, SNR, chi-square, sigmasq, phase, the other stored trigger columns, valid intervals and metadata checks pass across the five declared pairs. Conditioned strain digests, gating metadata and segment geometry match exactly. The standalone and restacked normal CPU full PSDs pass; Torch CPU has 2375 finite-bin budget violations and CUDA has 3105 against either normal CPU reference. These failures are below 30 Hz. The filter slice `[15360:1048576]` is byte-identical, and nonfinite masks match.

Full PSDs use `1e-4` relative tolerance and zero absolute floor. Trigger fields use `1e-4` relative and `1e-5` absolute budgets; sigmasq uses `1e-5` relative and zero absolute floor; phase uses `1e-4` radians. These are comparison budgets, not a physical accuracy guarantee. The raw arrays, declared provenance substitutions and per-field results are retained.

The original strict controller stopped on the full-PSD failure before collecting timings. A separate post-qualification continuation required all five trigger comparisons and exact conditioned strain, geometry and used PSD bins, then collected descriptive costs. The initial stopped status, unchanged scientific thresholds and explicit policy amendment remain archived. The independent verifier separates evidence completeness from scientific equivalence and keeps `equal_output_speedup_eligible: false`.

## What was measured

The clock spans child process launch through exit after HDF output and runtime/native verification. It includes imports, frame I/O, conditioning, PSD and compressed-bank loading, filtering, vetoes, clustering and output. Parent source hashing and post-run output comparisons are outside the clock; host sampling runs concurrently. Qualification clocks are excluded from the medians.

The shared host `len` is an AMD Ryzen Threadripper PRO 3995WX. Each route uses CPU 8 (SMT sibling 72), one numerical thread and explicit MKL; CUDA additionally uses RTX 4090 GPU 0. Graph capture is disabled. The host, sibling CPU and GPU were not reserved; observations before/during/after each worker are included. Filesystem caches were not flushed. The result does not establish sustained capacity or a full-machine throughput limit.

Existing native binaries were copied only after unchanged native-source and binary-hash verification; this campaign did not recompile them. Worker receipts pin imported PyCBC source, native modules, inputs and observed thread pools. Both sources used the same Python 3.11.9 environment. The inherited metadata inventory lists Torch 2.1.1 while every Torch worker reports imported 2.13.0+cu130. A separately bound post-acquisition diagnostic records both distributions and selected file/RECORD matches. It explains metadata selection without retrospectively hashing every Torch binary loaded by the workers.

## Evidence

- [Reproduction instructions](REPRODUCE.md), frozen bank in `inputs/`, and `sources.bundle`.
- [Configuration](acquisition/config.json), [source proof](source-proof.json), [source verification script](verify-source-commits.py) and [download verification](download-verification.json).
- [Raw acquisition](acquisition/): four qualifications, sixteen timings, HDF outputs, full qualification PSDs, runtime receipts, commands, host observations, source/build pins, original stopped status and explicit continuation policy.
- [Independent verification](verification/): recomputed comparisons/statistics, report and negative checks.
- [CPU-base restack validation](../cpu-base-restack/report.md): relevant unit tests, native-source and test-partition audits. Later formatting and documentation validation are recorded separately from this frozen implementation.

The earlier comparison of unchanged CPU `40e94792b3` with combined proposal `123e1fb3ef` is [historical evidence](https://github.com/xangma/pycbc/tree/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908). Its changed-trigger result must not be relabelled as a Torch-only comparison. The separate CPU cost experiment reports a 4.2611% median increase for changed arithmetic/output; it is not part of the four-route timing table above.
