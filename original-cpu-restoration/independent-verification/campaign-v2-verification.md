# Campaign v2 verification

Independent verification: PASS. All five archived comparison results were recomputed from local HDF/NPY evidence with unchanged tolerances.

Original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7` versus candidate `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`: **1988 triggers each, scientifically byte-identical**. Both complete H1 dataset-name sets were enumerated and compared; all 18 scientific datasets match dtype, shape, and C-order bytes. No right-only scientific field was ignored.

The only excluded H1 datasets are:

- `H1/search/filter_rate_per_core`: [0.0792893122521219] versus [0.08274252436514777].
- `H1/search/run_time`: [63.06020140647888] versus [60.42841982841492].
- `H1/search/setup_time_fraction`: [0.2309597899427871] versus [0.1853140391541363].
- `H1/search/templates_per_core`: [11594.254120553478] versus [12099.207658847336].

These four fields are elapsed-time-derived metadata; search start/end times and every gating/trigger dataset remain included.

The saved PSD arrays match exactly, including dtype, shape, bytes, and infinity masks. Conditioned-strain records match, including the SHA-256 digest of 8,323,072 float32 samples and gating metadata. Raw conditioned strain is absent, so that equality is supported by the recorded hashes. The five segment geometries match exactly and independently cover 7,798,784 samples (1904 seconds) without gaps or overlaps.

The new summary correctly reports **CPU preservation PASS**.

Both Torch arms contain 1991 triggers and retain FAIL against both CPU arms at the original tolerances. The complete scientific campaign remains FAIL; no speedup or performance claim follows.

The unadjusted comparison also retains declared source/executable-provenance differences. These are separate from the timing-field issue: after independently checking source and input pins, the original harness normalizes only those two provenance fields for its scientific verdict.

All recorded qualification checks and template/segment coverage were checked. Runtime receipts confirm affinity [8], one thread per observed native pool, and Torch intra/inter-op counts of one in Torch arms. Tracked source field sets and SHA-256 hashes match local Git blobs at the exact pinned commits; harness hashes and HDF/NPY file/data hashes match their receipts. Native binaries and external input bytes are checked through receipt consistency only; remote rehash evidence is supplied by the primary agent.

Harness review: campaign comparison logic changes only the four timing exclusions and their reporting. Separately, setup selects the v2 staging ref and config selects its commit; other harness Python files, including the comparator, are identical. Full diffs and hashes are in the JSON.

The original archive was unchanged across verification. This analysis writes only to the independent-verification directory.
