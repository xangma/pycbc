# CPU MKL workspace policy: rejected before timing

`DFTI_WORKSPACE=DFTI_AVOID` failed the frozen native-byte parity gate on the first qualification case. It is closed as an optimization candidate under this protocol. No timing workers ran and this experiment supports no speed claim or integration change.

The diagnostic used the unchanged accepted PyCBC source at `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`, a 2^21-point complex64 IFFT promoted into its existing in-place complex128 workspace, on `len` with CPU affinity 8 and one thread in Torch and all four detected native pools. Both A (default ALLOW) and B (request AVOID) used the same private commit proxy, original constructor, Tensor allocation, guards and execute method. An unproxied original plan supplied an independent byte reference. The original stayed at value 51; B read 51 initially, set 52 once before commit, and read 52 before and after commit. Every native status was zero.

The native contract worker passed 15 check groups and 14 controlled failure paths, including descriptor ownership, single free, storage lifetime, version counters, unchanged input and failed-execution output, guard drift, wide status values, the four-byte enum-output canary, shared function signatures and an existing unrelated plan. This does not replace the separate scientific qualification.

The qualification stopped after three rows: unproxied, A and B for seed 7, dense input, scale 1e-12, using the original whole-engine path.

| Check | Unproxied / A | B |
|---|---|---|
| Full native complex128 bytes (33,554,432 bytes) | Exact match | Different |
| Final complex64 bytes (16,777,216 bytes) | Exact match | Exact match |
| Final output versus matching-precision MKL | Exact match | Exact match |
| Input preservation and version counters | Pass | Pass |
| Final L2 error | 7.495152201632806e-14 | 7.495152201632806e-14 |
| Final maximum absolute error | 3.0804501236303285e-16 | 3.0804501236303285e-16 |

The unchanged FFTW error limits for that case were L2 5.652117985554614e-13 and maximum absolute error 1.8888594884312832e-15, so B passed both final-output limits. The native byte hashes nevertheless differed: unproxied/A `ff7807101e83e90c050ff5538513ae228771c4f181c3108aaeba9aec759d3d84`, B `a812920f88163fc2d42cc6169464a2749dc57efe7c3c60f169d06a062c3315af`. All final outputs hashed to `fc91bd52777b3fc77b50447628b1b3726436ceb2e8c2befc3c75aadde6d50187`.

The protocol required full native and final byte parity. The controller therefore recorded `rejected` and launched no further qualification or timing worker. The remaining cases, separate native-call qualification and all three timing workers were unrun. Stored construction and first-call durations are setup observations only and must not be used as performance comparisons. Passing final bytes on this single case establishes nothing about the unrun cases.

All three plan owners and native Tensor references were released. Both child processes exited zero; the qualification's explicit rejected state, rather than its process exit, correctly controlled the stop. Controller PID/PGID 3912037 and both children were absent at terminal sealing. An independent auditor reacquired the same benchmark lock nonblocking and verified unchanged input, source, runtime-library and installed-header pins. No source change or gate relaxation was made.

Evidence: [native contracts](acquired-diagnostic-v1/native-contracts.json), [qualification](acquired-diagnostic-v1/qualification.json), [controller status](acquired-diagnostic-v1/workspace-status.json), [terminal audit](acquired-diagnostic-v1/terminal-audit.json), [owner recomputation](owner-results-audit.json), and [protocol](diagnostic-v1/CPU-DFTI-WORKSPACE-PROTOCOL-v2.md). The input provenance distinguishes fake/synthetic/mock tests, the corrected import-only preflight and actual native execution.

Input manifest SHA-256: `c0cf59a906575a7d89d0be4b5d2f60ee054c8d8f7953b1cec279776e20518c06` (23 files). Results manifest: `04ff0360199bd176c2f58411abf17c106438a1e57f161d12a092286ec855a4a0` (34 files). Result archive: `05a828101cce79bcded64363d20f30270d407911da032c14fc4a338ae55ddb55` (35 regular members, 399,360 bytes).

Independent peer review passed for the frozen harness and sealed terminal evidence. The reviewer verified all archive/input/result hashes, native checks and failures, the exact first-case byte/error distinction, runtime and lock receipts, absence of timing, and independent closure. See [peer review receipt](peer-results-review-v1.json); no findings remain.
