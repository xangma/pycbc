# Verified original CPU restoration publication

All 15 stack branches were published atomically with explicit expected-commit leases. Live GitHub checks verified every new head, dependency, complete PR description, draft state and `agent-assisted` label. PR #18 now starts directly from original CPU `40e94792b3edf59f39b18b65102b28a4f74433a7`. PR #20 was removed as a prerequisite and closed as withdrawn without merging. All 15 stack PRs remain open drafts, with the human Code of Conduct confirmation unchecked.

The published main head is `8f71727e28ede9de3e3bbed82bb616eab710cbe7`. The measured restoration is `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`; the previously published final-source mapping connects its tested sources to the final formatting/documentation changes. This receipt adds publication verification and does not claim a new numerical run.

The original and restored CPU runs each contain 1,988 triggers; all 18 scientific H1 datasets and complete PSD arrays are byte-identical. Conditioned strain matches by full-data SHA256 and metadata; its raw data was not archived. Both Torch CPU and Torch CUDA contain 1,991 triggers and fail the unchanged scientific gates. No performance samples or speedup claim follow from this failed qualification. The optional #16/#17 changes retain their separate regression evidence and are outside the main scientific qualification.

The main local suite passed 507 cases plus 8 subtests (143 skipped); no-Torch passed 22, cached CPU chi-square passed 2, and Linux passed 137 (1 skipped). The strict 19-page documentation build passed. CI F401 passed; the unchanged pre-existing Qlty B904 finding remains documented.

`summary.json` gives the verified branch chain and closed PR state. Full API receipts, reviewed PR bodies, the atomic push result and the publication script are retained alongside it. Scientific data, tests and final-source mapping remain unchanged in the parent evidence directories and immutable earlier commits. All 1,248 previously hashed files were verified byte-for-byte before this receipt was added.
