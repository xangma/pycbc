Connect Torch filtering to event processing and offline and live search, including shared input-loading improvements.

## Standard information about the request

This is a: new feature, bug fix and efficiency update.

This change affects: the Python interfaces used by search and inference.

This change: includes focused regressions and follows the repository contribution workflow. Torch remains optional; no low-level C or Cython sources are changed in the main stack.

This change changes: Python implementation and its numerical/interface regression tests.

This change will: retain the optional Torch dependency introduced by #5.

## Motivation

Torch search needs to produce correctly normalized events through the existing interfaces while avoiding repeated setup work and preserving asynchronous output ownership.

## Contents

Coincidence, ranking, cuts, significance, event management, live batch matching and StrainBuffer integration. Filtering primitives remain in the preceding PR.

Live batches keep template powers and PSD caches distinct for groups sharing FFT workspaces; veto repairs retain full correlation geometry and invalidate stale rows. Torch CPU reductions and accelerator peak handling include the CUDA host-transfer correctness fix; ordinary-CPU reduction behavior is preserved. Shared frame reads skip unused duration discovery when explicit bounds are provided, preserving reads and validation. The ordinary CPU strain-segment FFT retains its original arithmetic; no CPU precision correction is inherited. This PR adds Torch FFT handling, keeps Torch precision coverage in a separate test file, and retains executable thread reporting.

Offline symmetric CUDA graph replay is explicitly enabled. Each replay checks storage, geometry, methods, analysis window, process, thread and stream. Captures own original storage and record streams; binding changes synchronize and release captures before eager fallback. Dynamic thresholds follow eager float32 rounding, sparse results own storage, and full SNR/correlation outputs retain their scratch-buffer contract.

Compatible single-precision Torch strain segments use the original CPU FFT arithmetic and return to the selected device with unchanged geometry and metadata. Ordinary CPU strain FFTs remain unchanged. CUDA compatibility includes synchronous host transfers and requires fresh performance measurement.

## Links to any issues or associated PRs

Depends on [#8](https://github.com/xangma/pycbc/pull/8). PR #20 remains withdrawn.

Base: `torch-pr4-filtering`. Head: `torch-pr5-search` at `29bdbe4d6e6075bd10dfcb6bcf9c367a85eb884a`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/8faf20dc51ae957e949dee409094eab6da4f22f3/torch-parity-fix); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/8faf20dc51ae957e949dee409094eab6da4f22f3/torch-parity-fix/final-publication).

## Testing performed

Owning-branch search tests: **56 passed, 12 skipped**.

Main integrated tests: **535 passed plus 8 subtests, 156 skipped**; no-Torch CPU checks **22 passed**. Linux CUDA regressions: **178 passed, 4 skipped**. CI-scoped F401 and changed-file Ruff formatting passed. The existing frame-reader B904 finding is retained and reported. Strict Sphinx built all **19 scoped main-stack pages**; rendered sources match final main documentation.

Completed main-stack campaign at `88878b1c38c952e63002b812058a0c7316123f70`, mapped to final main `7a72fba101b80467ebcdabe33ee973b9816094fb`: **CPU preservation PASS; original CPU, candidate CPU, Torch CPU and Torch CUDA each produce 1988 triggers**. All five comparisons pass unchanged trigger/full-PSD gates, with no missing or extra trigger identities or numerical violations. All 18 original-versus-candidate CPU scientific H1 datasets and full PSD arrays are byte-identical. Conditioned-strain equality is verified by full-array hashes and metadata; raw conditioned strain was not archived. Geometry matches.

Four recorded elapsed-time-derived fields are excluded from scientific byte equality; raw provenance verdicts and the two declared provenance substitutions are retained. Independent verification recomputed comparisons and checked source/receipt hashes. **No performance samples or speedup claim**; synchronous CUDA host transfers need separate performance measurement.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
