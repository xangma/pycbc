# CPU-base Torch stack restack

All 15 published PR heads were rebuilt locally onto standalone CPU `66789ac4a7468094b0cc3ca1498a1de67e0311f6`. The stable #15 head is **`f582b6fd250d0b82612492979e01e645d5c07afc`**, replacing `bd2d956f5c4e7c69b957181340d62d76a25717ac`. Its tracked worktree is clean. No source head changed after this stable SHA was communicated to the primary.

The reviewed main diff changes only four production functions and six precision test files. Every other tracked file matches old #15, including the latest documentation/evidence. All native source blobs match the corresponding old head across all 15 PRs. There are no new C, Cython, CUDA or `pycbc/lib` edits.

## Staging and ancestry

Worktree: `/private/tmp/pycbc-cpu-base-restack-20260908`. Each ref is `codex/cpu-base-restack-20260908-prNN`, with `NN` replaced by the PR number. The main chain is CPU → #18 → #5 → #6 → #7 → #8 → #9 → #10 → #11 → #12 → #13 → #14 → #15; optional branches are #15 → #19 → #16 and #15 → #17.

| PR | New parent | Old head | New head | Replayed commits |
| --- | --- | --- | --- | ---: |
| #18 | CPU | `5444d9fa9ead578a43a3c0c02de641f2869f394b` | `f2eb803de6f2d34f2227e69e35d460bd481fff05` | 2 |
| #5 | #18 | `13cd5a9e4dfe5bc220cbca80e54471a18d770fe1` | `4ae42eb20638704dae2ec47882387df3d81b0b08` | 2 |
| #6 | #5 | `27870cb163e8ba9f518ebf5ecff4fd0213fb993b` | `4865162eaa0c6f2332ec398bc30fc45773cb565c` | 1 |
| #7 | #6 | `f8b9256650ecee25b34ea193dad10e6c19eed7b0` | `92648253222205205c6ff96045ff1eb9ad6e30c1` | 1 |
| #8 | #7 | `d5df3d23fc2ed6a54f7376b7932e473184d6f063` | `026f4e8c9418e18cef1a44af36520936b2b2450c` | 1 |
| #9 | #8 | `6c6de2a31d62465feaf2c2c876e1e60dc3418407` | `5fea7897c352b2952581308707af6236539421ba` | 1 |
| #10 | #9 | `f27a4802cf577569062bd55603f77f5a1146d3eb` | `4d1901a58577c40ef393650d4d134a82fa9a69f0` | 1 |
| #11 | #10 | `e5c9b865c56f4e7052163af965e0a3406657bd8a` | `971f2b6036c3fddc10099481efb83377cfcb788c` | 1 |
| #12 | #11 | `3c71bb58a7972a05779590824cd859e394ea6377` | `a8181bd598f9775ef44fb277a9ac8348cba356ce` | 1 |
| #13 | #12 | `48a22169bf90800908f04d7ee23566b4ee4dbae0` | `f1d02414dbe13a0c689d7eb37b0180182e603b20` | 1 |
| #14 | #13 | `22410cd57e2bfb3a087c96917a0b93155cf80cac` | `d196f687230e1a1f3e1c7b75d25b97db1cbac924` | 1 |
| #15 | #14 | `bd2d956f5c4e7c69b957181340d62d76a25717ac` | `f582b6fd250d0b82612492979e01e645d5c07afc` | 6 |
| #19 | #15 | `1b81a82b84814d6cdc6cc6d27cf47b1b4e195fde` | `52425c6b8e5c8ffe79fa6bc4c1c3b70515163ea7` | 1 |
| #16 | #19 | `52bb4742667671f178314b2bd1a23f83029cab32` | `d4699798e44d8e4b4935d3d50c53279acde0c0b2` | 1 |
| #17 | #15 | `b213a6aeffa69ba0f249c8b5c883434d2371cecf` | `db8c849bdc4e6eb63e84e67ae11a9efd2f10db9f` | 1 |

All 22 commits in the verified old per-PR ranges have an explicit old/new mapping. [manifest.json](manifest.json) contains the full old/new heads, bases, published/staging refs, ordered commit maps, conflict decisions, commands, runtime provenance and logs. Old heads/bases were checked against the primary's supplied [live snapshot](../live-prs-before-cpu-restack.json); this sidecar made no network request. All refs present before the sidecar still resolve to the same objects.

## Functional review and conflicts

- `pycbc.psd.from_cli`: CPU restores input dtype only when float32 input was actually promoted. Published non-CPU return casts and non-MPS Torch promotion are retained.
- `pycbc.filter.matchedfilter.sigmasq_series`: CPU promotion is conditional on float32 magnitude; the published non-MPS Torch cast path is retained.
- `StrainSegments.fourier_segments`: CPU casts FFT output back only after promotion. Published non-CPU casting and Torch promotion remain in place.
- `power_chisq_at_points_from_precomputed`: the standalone CPU body is retained with the published Torch early dispatch. Splitting the `shifts` assignment and restoring CPU comments/formatting does not alter arithmetic.

The [structural audit](structural-audit.json) verifies that each affected module's AST outside these four function bodies is identical to old main. It additionally verifies the chi-squared CPU body against standalone CPU, the Torch dispatch against old main, and arithmetic identity after inlining the split assignment. Guards and dtype handling in the other three functions were reviewed directly. See [production diff](final-main-production.diff), [complete diff](final-main.diff) and [diff statistics](final-main-stat.txt).

Conflicts occurred in #18 formatting, #7 PSD, #8 filtering/chi-squared/tests, and #9 strain/tests. They were resolved within the affected functions/imports/test cases. A duplicate strain `_scheme` import found during review was removed before the stable #15 SHA was communicated; descendants were rebuilt from that corrected #9. No native conflict was resolved by editing native code. The six latest #15 documentation commits were replayed in order and their final files remain byte-identical to old #15, as historical evidence.

`test_chisq_precision.py`, `test_sigmasq_series_precision.py` and `test_strain_psd_precision.py` are byte-identical to standalone CPU on all 15 heads. They contain no Torch imports or Torch-dependent skips. Published additional Torch coverage lives in `test_torch_chisq_precision.py`, `test_torch_sigmasq_series_precision.py` and `test_torch_strain_psd_precision.py`; duplicate CPU cases were removed from those moved files, retaining Torch reference calculations, assertions and CUDA/Triton checks.

## Local validation

Existing interpreter: `/private/tmp/pycbc-cpu-precision-env-20260908/bin/python`; Python 3.13, NumPy 2.3.5, SciPy 1.16.3, macOS arm64. No shared environment install was performed. Each run set `PYTHONPATH` to the staging worktree, disabled unrelated pytest plugin autoload, and set OMP/OpenBLAS/MKL thread counts to one. Imports, Git head, generated version and native module origin were checked before tests. Eleven existing compiled modules were copied only after verifying unchanged native source blobs, and their SHA-256 hashes were checked. [prepare-runtime.py](prepare-runtime.py) and `runtime-*.json` record this preparation.

| Case | Tested head | Passed | Skipped | Failed/errors | Log |
| --- | --- | ---: | ---: | ---: | --- |
| pr7-psd | `926482532222` | 92 | 17 | 0 | [pr7-psd.log](pr7-psd.log) |
| pr8-filtering | `026f4e8c9418` | 213 | 110 | 0 | [pr8-filtering.log](pr8-filtering.log) |
| pr9-strain | `5fea7897c352` | 77 | 21 | 0 | [pr9-strain.log](pr9-strain.log) |
| pr15-main | `f582b6fd250d` | 377 | 139 | 0 | [pr15-main.log](pr15-main.log) |
| pr15-no-torch | `f582b6fd250d` | 19 | 0 | 0 | [pr15-no-torch.log](pr15-no-torch.log) |

#7 covers CPU precision, PSD and Torch PSD pipelines/protocols; #8 covers CPU/Torch precision, filtering, sparse and optimized chi-squared paths; #9 covers strain/PSD precision and search kernels. #15 runs their union plus FFT write and CPU-native tests. The separate #15 CPU run blocks Torch imports/discovery and passes all 19 standalone tests without skips. Commands and JUnit XML paths are recorded per run in the manifest. These are overlapping suites; counts should not be summed as unique coverage. CUDA is unavailable locally and capability-dependent CUDA/MPS skips are visible in the logs.

The supervisor and test processes have exited. Optional leaves #16/#17 received ancestry, exact-range replay, CPU test, documentation and native blob audits; they were not separately executed. The primary's previously reported standalone Linux result (32 passes; two legacy MKL failures reproduced identically on frozen original) is context only and was not rerun in this sidecar.

## Lint and remaining ownership

All three exact CI F401 file selections pass: 225 executables, 292 modules and 139 test files. [lint-audit.json](lint-audit.json) records commands and logs. Unfiltered `flake8 pycbc/ test/ --select F401` reports 130 diagnostics, verified unchanged against old main for tracked files; generated `pycbc/version.py` is excluded by CI. Full `flake8 pycbc/ test/` also reports inherited style findings. The changed-file comparison is 272 old versus 269 new diagnostics; eight apparently added E501 findings are verbatim historical lines relocated into Torch test files, with provenance in [lint-relocated-lines.json](lint-relocated-lines.json). The `qlty` CLI is unavailable and was not installed.

This completes the bounded local implementation and validation. Primary retains independent source review, Linux corrected-base/rebuilt CPU/Torch qualification and timing, and all publication. This report makes no new performance, sensitivity or trigger-ranking claim.
