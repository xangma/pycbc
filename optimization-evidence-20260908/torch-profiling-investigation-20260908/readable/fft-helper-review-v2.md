# FFT diagnostic helper v2 review, 8 September 2026

Reviewed `qualify_ifft_routes.py` SHA256 `cdd5fd2cf87f176c13f6993c63429250d6b2e70f8e96f354a3cc75fe29a02c00` and contract tests SHA256 `99e1b343927f25fad956600f99045e56d8d3bb0ac12b4960ea92231ded943a6f`. The two material v1 review gaps are closed. No remaining source-level blocker was found for the already-authorized, isolated production-length diagnostics.

The exact live timing plan is now validated after its timed samples over the complete original matrix, including matching-precision MKL bitwise parity when applicable. Its samples remain recorded on failure, with candidate eligibility false. Both FFTW wisdom stores are cleared under the planning lock before creating fixed references. The timing entry gate requires qualification mode and exact ordered nonempty 36-case coverage; the timing CLI also requires the frozen seed list. Candidate gate results are checked individually, and the explicitly labelled reference-only routes remain distinct.

Qualification and timing compare host, recorded package versions, thread environment, affinity, loaded FFTW/MKL paths and hashes, and source identity. The latter includes all PyCBC Python files, native extension hashes, tracked changes and status including untracked files. The campaign wrapper still supplies the exact interpreter invocation, thread-pool verification, worker isolation, ownership lock and expected source/helper/runtime pins. A source review does not establish those live controls.

The input generator matches the archived large-transform generator: identical seed sequence, finite-band boundaries, impulses and float32 scale multiplication. Error calculations and acceptance budgets are unchanged.

The local 4,096-point direct-FFTW smoke records 36 passing qualification cases and 36 passing cases on a freshly timed plan. Its qualification SHA256 is `488317f5c52bf2d612bbcc84b32e77f28ad857490cb1d09e987458039f25260c`; its timing SHA256 is `9a474a7a302b28f536f0e1c1470fbdc4a2ba55b2607696406b245c06c45f605e`. These confirm the harness workflow locally; they do not qualify the 2,097,152-point candidates or establish their performance.

No profiling-task source edits or remote jobs were made. The optimization task owns acquisition and any candidate implementation. Original failures, scientific gates and full-workload validation remain required.
