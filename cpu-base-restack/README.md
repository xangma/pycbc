# CPU prerequisite and final Torch stack validation

The final publication main is `1d22031fd31c6e5bcb48a68fe11772e960bd406b`. All 15 PR heads inherit the
standalone CPU prerequisite in [#20](https://github.com/xangma/pycbc/pull/20).
The main chain is CPU → #18 → #5–#15; optional leaves are #15 → #19 → #16
and #15 → #17.

The [final publication report](../corrected-baseline-final-restack/report.md) and
[composed publication manifest](../corrected-baseline-final-restack/publication-manifest.json)
map original published heads through the measured CPU restack, formatting,
and final documentation. Each stage preserves its own commit mappings.
[Independent final review](../corrected-baseline-final-restack/primary-final-publication-review.json)
checks ancestry and unchanged native sources at all 15 heads, unchanged
standalone CPU tests, the exact seven-file difference from measured main,
complete AST equality in the two formatted Python modules, and all 19 built
documentation sources against the final commit.

The [initial CPU restack report](report.md) records local tests at measured
main `f582b6fd250d0b82612492979e01e645d5c07afc`: 377 passed and 139 capability
skips, plus 19 standalone CPU tests with Torch imports blocked. Those suites
overlap. Prefix runs, raw logs, CI lint selections and inherited lint findings
remain in this directory. The report's then-pending Linux/quality/publication
work is superseded by the completed records linked here.

[Qlty receipts](../corrected-baseline-quality-v2) retain the two formatter
findings, exact correction and successful follow-up. The final two Python
files match that checked correction byte-for-byte. The final source adds
only five documentation edits after the formatting-only main; no runtime
tests or timings are represented as reruns on this documentation commit.
[Strict Sphinx result](../corrected-baseline-final-restack/sphinx-build-result.json) and
[log](../corrected-baseline-final-restack/sphinx-build.log) record all 12 Torch guides plus seven
waveform/plugin/installation pages, using the real repository extensions.

The [corrected-baseline campaign](../corrected-baseline-campaign/README.md)
contains all 20 executable runs and independent verification. All five
cross-route trigger comparisons pass; Torch full-PSD differences below
30 Hz remain failed. The timings are descriptive execution costs.

Archive SHA256SUMS.json covers every evidence file. Scripts retain acquisition
paths for provenance; adapt workspace paths when replaying source checks.
