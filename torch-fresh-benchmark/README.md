# Fresh Torch complete-executable benchmark — 2026-09-08

Four new scientific qualifications followed by **16 fresh unprofiled timed processes**, four per route in balanced rotating order. Original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7` is unchanged; candidate main `eb8fef9ed1d06378b59cae8439fd40af63827575` is the exact published main at launch. Optional FFT and native CPU optimization leaves are outside this measurement.

| Route | Median (s) | Range (s) | Original CPU / route | Template-seconds / wall second |
|---|---:|---:|---:|---:|
| Original CPU | 65.52 | 65.37–66.28 | 1.00× | 11,159 |
| Candidate CPU | 64.91 | 64.74–65.02 | 1.01× | 11,265 |
| Torch CPU | 103.52 | 103.48–103.86 | 0.63× | 7,063 |
| Torch CUDA | 31.97 | 31.91–32.02 | 2.05× | 22,871 |

Workload: **384 compressed templates × 1904 unique H1 seconds = 731,136 template-seconds**, five analysis segments and 1920 template/segment pairs. All four routes produce **1988 triggers**. The complete commands, input hashes and 512/112/16-second geometry are in [configuration](remote/config.json) and qualification receipts.

Host `len`: AMD Ryzen Threadripper PRO 3995WX, affinity fixed to logical CPU 8 (SMT sibling 72), numerical thread pools fixed to one, and NVIDIA GeForce RTX 4090 for CUDA. The Torch routes additionally fix intra/inter-op counts to one. Host and GPU were shared and unreserved. These are observed finite-workload medians and ranges, not confidence intervals, sustained capacity or a full-machine comparison.

Timing starts immediately before launching the checked executable process and stops at its exit. It includes imports, frame I/O, conditioning, PSD estimation, template preparation, filtering, vetoes, HDF output, runtime receipts and loaded-module verification. Qualification instrumentation, parent source/input hashing and output comparisons are outside the timing boundary. Qualification durations are excluded from the table. CUDA graphs were disabled. Raw samples, source/native/module hashes, process environment, resource snapshots and `/usr/bin/time` output are retained for every run.

## Scientific qualification and independent verification

**PASS:** all five cross-route trigger and complete-PSD comparisons under unchanged tolerances. No missing or extra trigger identities or numerical violations. Original versus candidate CPU has byte-exact equality for all 18 scientific H1 datasets and complete PSD arrays; conditioned strain matches through full-array hashes and metadata, and geometry matches. Raw conditioned-strain arrays were not archived. Four elapsed-time-derived H1/search fields are excluded from scientific byte equality. Raw provenance verdicts and the two explicitly verified source/executable provenance substitutions are retained.

Each of the 16 timed outputs passed comparison with its own route's fresh qualification. [Independent verification](independent-verification/verification.json) recomputes scientific comparisons, timing statistics and source/receipt checks. [Summary](remote/summary.json), [all comparisons](remote/comparisons/), [run receipts and outputs](remote/runs/), [transfer verification](transfer-verification.json).

The source bundle requires original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7` as a prerequisite. Generated version metadata matches each measured commit; its bytes are included in `additional-source-metadata`. Actual Torch run receipts report **Torch 2.13.0+cu130 / CUDA 13.0**. The original flat package inventory incorrectly lists Torch 2.1.1 because it collapses duplicate distribution names. [Supplemental dependency inspection](additional-source-metadata/dependencies-actual.json), collected after timing, records the selected import and every Torch distribution path/version. The original inventory is retained as collected. Input data and native binaries are referenced by location and hash; their large payloads are not duplicated in this archive. Documentation and final branch mappings will be appended separately; benchmark records are immutable.
