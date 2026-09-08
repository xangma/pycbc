# Independent provenance verification

PASS: all 15 pinned ancestry/native/setup checks reproduce the source-preservation receipt. Original 40e94792 is an ancestor and the CPU-correction commit 66789ac4 is absent. Main aa6b795a has byte-identical original MKL, FFTW and NumPy FFT files. PR17 retains only the acknowledged pre-existing optional native differences; the restoration changes no native source versus the published heads.

The downloaded tar SHA-256 matches the supplied remote digest, and every extracted regular-file name and digest matches the tar. All four campaign receipts are complete. Staged and downloaded campaign harness hashes match. The bundle additionally contains linux-checks.py, an unpinned separate test helper, matching its local source.

Linux runtime JUnit evidence independently counts 137 passed, 1 skipped, 0 failures at aa6b795a. This is verification of recorded results, not a fresh Linux test run.

Linux .so binaries and external bank/frame bytes are absent locally; remote rehash verification remains primary-agent evidence. No performance claim is made.
