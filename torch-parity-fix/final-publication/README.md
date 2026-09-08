# Final Torch stack publication

The executable qualification measured `88878b1c38c952e63002b812058a0c7316123f70`. Final main `7a72fba101b80467ebcdabe33ee973b9816094fb` differs only in the seven reviewed documentation pages. All other final branch heads are byte-identical to their tested source outside documentation. [Source mapping](source-mapping.json) lists every tested and final head; [manifest](manifest-final.json) records the dependency stack. The bundle contains all final heads with original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7` as its prerequisite.

Strict Sphinx built all 19 scoped pages successfully using the measured runtime. Their recorded source hashes match the final main Git tree. [Build receipt](docs-audit/sphinx-build-result.json) records the exact scope and exclusions; source pages and rendered HTML are archived alongside it.

The separate optional CPU optimization leaf passed 266 tests with 42 skipped. Its [receipt](optional-pr17/provenance.json) pins the tested head, native sources and reused binaries; [XML](optional-pr17/pytest.xml) and logs are retained. This leaf and the optional FFT leaf remain outside the four-route executable qualification.

Original CPU and the withdrawn PR20 remain outside the changes. Publication keeps all existing PR numbers, branch names, draft states and agent-assisted labels. [Published PR descriptions](pr-descriptions/) and the [live GitHub verification receipt](publication-verification.json) record the completed update.
