# Original CPU restoration — 2026-09-08

The Torch conversion is based on unchanged original CPU commit `40e94792b3edf59f39b18b65102b28a4f74433a7`. CPU precision proposal #20 is removed from the stack. The qualified main source is `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`; later documentation and three formatting changes are mapped separately before PR publication.

## Result

On the fixed 384-template, five-segment, 1904-second H1 workload, original CPU and restored normal CPU both produce **1988 triggers**. All 18 scientific H1 datasets match in dtype, shape and bytes; PSD arrays and segment geometry match exactly; conditioned-strain equality is supported by matching full-data hashes and metadata (raw strain is not archived). Four elapsed-time-derived `H1/search` datasets are enumerated separately and are not scientific equivalence criteria.

**Torch CPU and Torch CUDA each produce 1991 triggers and fail the original scientific tolerances.** These results do not qualify an equal-output speedup. No performance timing samples were collected; process durations in qualification receipts are instrumentation only. The proposals remain drafts.

See [qualification summary](campaign-v2-results/summary.json), [all comparisons](campaign-v2-results/qualification-summary.json), [independent verification](independent-verification/), [source preservation](source-preservation.json), and [focused local tests](local-tests-v2/manifest.json).

## Scope and verification

Main-stack CPU arithmetic, array-copy semantics, FFT validation, sky-max behavior, PSD warning policy, resource-cache identity and runtime defaults have been restored. The main CPU MKL, FFTW and NumPy FFT modules are byte-identical to the original. Native source is unchanged by this restoration. Existing optional general FFT work (#16) and optional native work (#17) remain separate leaves, with their own tests; their source is outside this main-only executable qualification.

Local integrated regressions: 507 passed, 143 skipped, plus 8 subtests. CPU tests with Torch imports blocked: 22 passed. Cached CPU chi-square: 2 passed. Additional owning-PR tests are recorded separately. Linux runtime regressions: 137 passed, one skipped. CI-scoped F401 passed. Qlty's three formatting findings were corrected with AST-equivalence checks; one B904 finding remains in a frame-reader file byte-identical to the prior published main and is reported as an existing finding, not a green gate.

## Provenance and historical records

Both campaigns retain raw trigger HDF files, full PSD arrays, source bundles, input/native/module hashes, process commands, dependency/runtime receipts and per-route qualification checks. Large frame and bank inputs are identified by immutable hashes and their original locations. The new downloaded archive SHA-256 is `c93f88b0a34d5ef89b533867f8c997553257fe56d0dda443a401e3f120d59264` and matched the remote archive.

The first restoration candidate `652206d84177f658f5fb2f9cab2ee73f6d2bc95f` is superseded. Its raw `cpu_preserved: false` result is retained unchanged: it incorrectly included four timing metadata fields. Independent reconciliation checks its scientific arrays directly and enumerates the exclusions. The second campaign corrects only that classification; scientific tolerances are unchanged. Both attempts preserve the failed Torch comparisons.

Earlier corrected-CPU numerical and performance reports elsewhere in this evidence branch remain historical evidence for withdrawn #20. They do not validate the restored Torch conversion. Final publication mapping and rendered documentation evidence are appended in a subsequent commit.
