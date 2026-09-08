# Final publication mapping

The 15 draft PR heads in `manifest-final.json` restore the main Torch conversion to original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7`, excluding PR #20. This records the prepared heads; GitHub publication is verified separately after the branch update.

`final-source-test-mapping.json` compares every prepared head to the tested restoration or its source reference. Only three Python files with exactly equal complete-module ASTs and six reviewed documentation files differ. All other tracked files are byte identical. The main scientific acquisition retains tested source `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`; no numerical rerun is implied for the final documentation and formatting commits.

CPU preservation passes the frozen workload: both CPU arms contain 1,988 triggers, all 18 scientific H1 datasets and complete PSD arrays match in dtype, shape and bytes, and conditioned-strain digests/metadata and geometry match. Raw conditioned strain is not archived. Both Torch arms contain 1,991 triggers and fail against both CPU controls under unchanged tolerances. No performance samples were collected. Qualification and independent verification are in the parent directory.

The main relevant local suite passed 507 cases plus 8 subtests (143 skips); no-Torch tests passed 22, cached CPU chi-square passed 2, and the Linux runtime suite passed 137 (1 skip). Owner-specific regressions also pass. CI F401 passes. Qlty retains one pre-existing B904 finding in unchanged `pycbc/frame/frame.py`; this is not a green Qlty claim.

Optional PR16 and PR17 remain separate from the main conversion and its scientific qualification. PR16 is mapped to its independently tested FFTW retry-lock fix; its tests passed 348 (16 skips), with five real FFTW probes. PR17 passed 627 (102 skips). Their pre-existing general FFT and native CPU changes are outside the main unchanged-CPU qualification.

The strict Sphinx build passed all 19 Torch and required dependency pages, with repository extensions/theme and real plot/command directives. Its declared scope excludes unrelated manual pages and generated include material. The exact command, runtime, source hashes and rendered HTML are under `docs/`.
