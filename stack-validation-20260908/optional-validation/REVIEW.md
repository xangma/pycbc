Optional PR validation — 8 September 2026

All three optional branches are ready for parent publication on main #15 at `9d4e4f6905d9f109d559284326d62173a854264f`. See `final-heads.json` for full identities, parents, diffs and receipt paths.

| PR | Rebased head | Rebased parent | Validation before inherited documentation / LAL-test fix |
| --- | --- | --- | --- |
| #19 | `d50790f51a5c39a5e96d7c15201359c159fc24fb` | `9d4e4f6905d9f109d559284326d62173a854264f` | 392 passed, 5 skipped, 178 deselected, 2 warnings, 56 subtests passed in 30.39s |
| #16 | `beec534b130bc6d52943269121b8b07892f6c04b` | `d50790f51a5c39a5e96d7c15201359c159fc24fb` | 465 passed, 5 skipped, 178 deselected, 2 warnings, 56 subtests passed in 15.74s |
| #17 | `7289c59c7e5f97354b596a256aa5c3ae6b53b031` | `9d4e4f6905d9f109d559284326d62173a854264f` | 525 passed, 5 skipped, 179 deselected, 2 warnings, 56 subtests passed in 15.98s |

Each prefix passed its isolated editable build, the full F401 file scope from `.github/workflows/check_code.yml`, and Qlty 0.644.0 against its actual parent with zero findings. Qlty used `--no-upgrade-check --no-fix --sarif --no-progress --skip-source-fetch --no-cache`; Ruff 0.14.6 was already installed. Full commands, timestamps, host, source identities and log hashes are recorded in the per-stage receipts. JUnit XML and stdout/stderr logs are retained.

The test lists came from the public prior PR validation receipts (23 common files plus the relevant optional suites), with current frame regression coverage added. The three prefixes exercised 24, 27 and 29 files respectively. Tests ran with Torch CPU and one-thread settings on macOS arm64 / Python 3.13.9, Torch 2.9.1 and LALSuite 7.26.1. CUDA/MPS-related cases were deselected; five tests skipped for unavailable CUDA or Linux/x86-64 MKL qualification. These receipts do not claim new Linux-native or GPU qualification. Each prefix also passed 56 subtests. Environment provenance and the built native extension hashes are in `environment.json`.

Review: #19 and #16 retain the exact original patch IDs. #17 merged without conflicts; its added and removed lines match the original feature per file exactly, while its patch ID differs because surrounding main runtime context changed. Both original Cython files are byte-identical to the specified feature commit. The newer in-place promoted MKL workspace, scalar normalization contracts, and CUDA host-array completion changes remain inherited from main. The CPU native peak gate remains default-off and ahead of the inherited Torch fallback. No new runtime/test/lint fix was required and no tests were weakened.

The assigned worktree is left clean on #17. Only the three authorized optional refs were rebuilt. No remote, GPU, push, or PR operation was performed. The earlier prepared ref identities are preserved in `initial-state.json`.

The three branches were finally transplanted onto main `9d4e4f6905d9f109d559284326d62173a854264f`. Relative to each original tested head, only `docs/torch_followups.rst`, `docs/torch_search.rst` and the shared `test/test_array_lal.py` skip correction differ, exactly matching the inherited main changes. Every optional feature diff is byte-identical. Runtime, build, tool, CI and all other test files are byte-identical. The exception is the parent’s replacement of unsupported Torch `simple_exit(str)` with `unittest.skipIf`; the parent reports 1,288 array legacy tests passing with one LAL skip and a passing additional 14-file PR8 suite.

`final-rebase-verification.json` and `final-heads.json` retain original tested, intermediate documentation-rebased and final identities. Original stage receipts and tested-source fingerprints remain unchanged; final fingerprints are separate `prNN-final-source-fingerprint.json` files. Full optional suites were not rerun, as requested. No optional runtime/test fixes were added.
