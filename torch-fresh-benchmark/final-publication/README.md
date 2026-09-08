# Fresh benchmark documentation and final branch mapping

The fresh complete-executable benchmark measured `eb8fef9ed1d06378b59cae8439fd40af63827575`. Final main `d22c364601f3f70f3dec6dcc88be3e5f9b6d0f9d` changes only `docs/torch_performance.rst`, adding the current result table and its measurement scope. Its runtime code is byte-identical to the benchmarked main. Regression tests previously measured `88878b1c38c952e63002b812058a0c7316123f70`; all 15 final heads retain their respective tested bytes outside documentation. [Source mapping](source-mapping.json) records both kinds of validation separately.

Strict Sphinx built all 19 scoped pages with no warnings. Their recorded source hashes match final main. The [build receipt](sphinx-build-result.json) records the exact scope and exclusions. Sources and rendered HTML are included; shared HTML assets are omitted. The bundle includes all final stack heads and requires original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7`.

The four qualifications and 16 timed samples remain unchanged in the parent evidence folder. Existing PR numbers, branch names, dependencies, draft states and agent-assisted labels are retained. Optional FFT and native CPU optimization leaves remain outside the executable benchmark.
