# Torch stack publication validation, 8 September 2026

The restack assigns runtime changes to the feature PR that owns them and leaves
benchmark tooling, documentation and plots in PR #15. It retains the existing
stack baseline and separately rebases the optional CPU and general FFT follow-ups.

## Scope and source identity

`final-heads.json` records the published branch candidates. The assembled main
runtime matches production source `fad7d8440bfde083f2e94ee62a0017492dfc4013`,
apart from whitespace and docstring layout in three Python files. The normalized
AST comparison is recorded in `final-source-equivalence.json`; the restack script
also asserts exact byte identity for every other runtime, executable file. Regression tests also match except for a standalone/pytest-compatible
Torch LAL-conversion skip correction in `test/test_array_lal.py`; the test-only
restack mapping records that change.
Native qualification uses that frozen production source. Documentation-only
commits do not alter the native source identity.

`focused-to-publication-mapping.json` checks each final prefix against its
focused test revision. The only intervening change is the inherited LAL-test
skip correction. The additional changed-file tests run at the final feature
heads, and all three legacy CPU LAL conversion tests pass. The four frame I/O
tests also pass at the formatting prerequisite using its standalone runner.
The earlier pytest invocation failed in legacy argument parsing before tests
could run; its original failure log is preserved alongside the corrected run.

`final-prefix-test-results.json`, the XML reports and individual logs record tests
at each owning feature prefix. Test selections overlap, so totals across prefixes
must not be interpreted as unique tests. CUDA and MKL cases requiring unavailable
hardware/libraries skip locally on macOS; the native production qualification
provides actual MKL/CUDA coverage. Local scientific dependencies are installed in
an isolated editable environment; the complete LALSuite 7.26.1 wheel replaces an
incomplete same-version environment installation only within that environment.

## Static checks and documentation

The final main stack passes the repository CI's F401 selection across 653 files
and the Qlty CLI check against the formatting prerequisite. The formatting PR
retains 86 verified pre-existing non-formatting findings: the original 85 plus
one parent frame-reader warning exposed by the expanded formatting scope. Its
parent comparison and AST audit record this baseline; no checks were disabled.
The optional branches have their own records. Qlty used 0.644.0 and Ruff 0.14.6.
The strict Sphinx build covers all Torch guide pages with a small standalone
configuration and stubs for two external guide references; it is not a full
project documentation build. Both current figure pairs and their input manifest
pass offline verification. The PNG/SVG files were visually reviewed.

## Performance interpretation

The newest matched three-backend executable campaign measures `ecd5d082`, with
median wall times 66.933429 s (standard CPU), 104.400193 s (Torch CPU) and
21.765736 s (Torch CUDA). Descriptor-reuse and offline graph timings measure
separate pinned prototypes and paired baselines. No new timing campaign measures
the integrated runtime or this restack. Native/executable correctness qualification
is separate from a performance claim.

The original sealed optimization evidence is published at
[c4bfea522807742388dc8bcddc86473b9c03b047](https://github.com/xangma/pycbc/tree/c4bfea522807742388dc8bcddc86473b9c03b047/optimization-evidence-20260908).
That archive preserves accepted, rejected and superseded candidates. Scientific
error limits, final-output identity and intermediate-byte identity remain distinct.
The CPU gap is documented; the current measurements do not causally apportion
its entire duration or prove that every lower-precision alternative is unsuitable.

## Verification

Run `python3 verify.py` in this directory to check every published file against
`SHA256SUMS`. Verification does not execute any scientific source. Runtime sources
are available through the pinned Git commits and original evidence packages.

## Integrated production qualification

`native-production/` retains the frozen source, native unit results, raw HDF
outputs, conditioning/PSD snapshots, references, controls and every harness
revision. Its offline verifier reproduces the native counts and five scientific
comparisons without running a workload. It passed independently after copying
into this publication tree. All three backends produced 1,991 matching triggers;
the 18 H1 science datasets match each backend's own pinned reference byte for
byte. The CUDA adapter checks production outputs before the eager oracle,
covering 1,920 full/sparse comparisons and five captures, retention and actual
consumer use. Both the suppressed-replay and pending-allocation negative controls
detect their intended failures. All native jobs finished and the shared lock was
released before sealing. These runs do not provide new timing measurements.

The first scientific comparator rejected different source/executable provenance
even though the output data matched. The adapter records and independently
validates only those two provenance substitutions, preserving the raw failed
verdict and unchanged scientific budgets. A subsequent negative-control metadata
check read the wrong error field; the corrected controller verifies the original
pinned error JSON, counters and log. All original records are retained.
