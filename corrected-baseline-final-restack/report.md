# Corrected-baseline final restack

Final documentation-bearing main: **`1d22031fd31c6e5bcb48a68fe11772e960bd406b`**. All 15 local refs are ready under `codex/corrected-baseline-publish-20260908-prNN`. The five documentation files are committed, strict Sphinx passed all 19 pages, and optional #19 → #16 and #17 now inherit this exact documentation commit. Publication remains with primary.

## Final heads

| PR | Parent | Final head |
| --- | --- | --- |
| #18 | CPU base | `f2eb803de6f2d34f2227e69e35d460bd481fff05` |
| #5 | #18 | `4ae42eb20638704dae2ec47882387df3d81b0b08` |
| #6 | #5 | `4865162eaa0c6f2332ec398bc30fc45773cb565c` |
| #7 | #6 | `92648253222205205c6ff96045ff1eb9ad6e30c1` |
| #8 | #7 | `6e1e08084c9d9be0e62fd701aaa662e61f1ad119` |
| #9 | #8 | `586d4f09d2962fe61287ba704e6c7ec971a8922f` |
| #10 | #9 | `6f67dc170811ff69fd6ec50816974bc4285d145b` |
| #11 | #10 | `69a96d515f1af056f74c25ef350255d89273f310` |
| #12 | #11 | `1219fe1a6fa9ee530bf2d5e57ec192b784c99474` |
| #13 | #12 | `424133803568b403cd6d30e7c54816dda2e93133` |
| #14 | #13 | `5b413c12b889e1869a897d022f67db1075c95c6d` |
| #15 | #14 | `1d22031fd31c6e5bcb48a68fe11772e960bd406b` |
| #19 | #15 | `0b5eb0c970e7d3edd726e8b3bc7abd3d6c63ddda` |
| #16 | #19 | `0a3bc0bd366c1c7fc5a4a8f06a4d0b5dc8239a36` |
| #17 | #15 | `f2180c07d7a23cf90ab597ebc1e02ab75e7d5b59` |

[manifest.json](manifest.json) contains all 15 refs, parent bases and commit mappings, plus the preserved formatting stage. Its `old_head` continues to mean the frozen CPU-base restack, not the live published branch. Primary will compose the separate publication manifest from its live snapshot.

## Evidence and source mapping

The final documentation links the immutable verified campaign at [e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c](https://github.com/xangma/pycbc/tree/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign). All 20 processes, five trigger comparisons and 16 timing-to-qualification comparisons pass. CPU versus CPU passes the full scientific comparison. Torch full-PSD comparisons fail below 30 Hz (2375/3105 finite-bin violations per segment for CPU/CUDA); timings remain descriptive and no equivalent-output Torch speedup is claimed. Independent evidence verification is PASS_WITH_LIMITATIONS and the full scientific result is FAIL. The documentation retains the timing-policy amendment, Torch distribution/import metadata discrepancy and receipt-based verification limitations, and separates the standalone CPU correction cost.

Measured main remains `f582b6fd250d0b82612492979e01e645d5c07afc`; corrected CPU remains `66789ac4a7468094b0cc3ca1498a1de67e0311f6`. Formatter commit `6e1e08084c9d9be0e62fd701aaa662e61f1ad119` belongs to #8; replayed formatted main is `6b47580146e73169cd130b601731e5ba40668d93`. Relative to the measured main, final main differs only in two Python formatting files and five documentation files. Complete module ASTs match excluding source locations; Python source bytes differ. All other tracked paths and modes, including native source, `bin/`, `test/`, `tools/` and `.github/`, are identical. No benchmark was rerun on the formatted or documentation-bearing head. Every original frozen CPU-base ref and every formatting-stage ref is preserved.

## Validation

- [Final structural audit](final-structural-audit.json): all 15 ancestry, source/AST, documentation inheritance and ref checks pass. Original CPU-base artifact hashes remain unchanged. Optional source contents match their corresponding formatted heads exactly.
- [Sphinx receipt](sphinx-build-result.json) and [log](sphinx-build.log): exit 0 with `-b html -E -a -W --keep-going`. All 19 actual Torch and required waveform/plugin/installation pages were built with the repository extensions/theme and real plot/command directives. Scope excludes unrelated manual/include generators, external intersphinx inventories and the remote logo. The runtime pin is frozen measured main; [all built-source hashes](sphinx-source-hashes.json) match the committed documentation. An initial short-title-underline warning was corrected before the successful run; its receipt remains archived.
- Browser inspection of performance/reference pages confirms rendered prose, immutable links and all timing-table values. Wide tables use the theme's horizontal scrolling. Two preexisting console syntax errors arise because unchanged `docs/conf.py` registers `terminal.css` and `theme_overrides.css` as JavaScript; this does not prevent the reviewed content rendering. Screenshots are under `/Users/xangma/repos/pycbc/output/playwright/`.
- Qlty PASS with zero issues applies to the primary run on frozen measured main plus the exact supplied formatter patch. Primary independently verified formatted-main file bytes, whole-module AST equality and all-15 ancestry/native equivalence in [its receipt](primary-final-format-source-review.json), bound to [the preserved format-only manifest](manifest-format-only.json). Final main adds only documentation. No additional Qlty run is claimed or outstanding.
- Changed-module F401 and final whitespace checks pass. Unit tests were not repeated for line wrapping and documentation; the unchanged [frozen CPU-base validation](../cpu-base-restack/report.md) remains the runtime evidence.

[Final docs diff](final-docs.diff), [final main diff](final-main.diff), and [replay log](publication-replay.log) make the completed changes reviewable. No remote mutations or PR publication were performed by this task.
