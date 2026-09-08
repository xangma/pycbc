Support Torch PSD processing with original CPU arithmetic for compatible single-precision search inputs.

## Standard information about the request

This is a: new feature, bug fix and efficiency update.

This change affects: the Python interfaces used by search and inference.

This change: includes focused regressions and follows the repository contribution workflow. Torch remains optional; no low-level C or Cython sources are changed in the main stack.

This change changes: Python implementation and its numerical/interface regression tests.

This change will: retain the optional Torch dependency introduced by #5.

## Motivation

PSD preparation must accept the same Torch-backed series as filtering, including analytical models, interpolation and data-driven estimation.

## Contents

Analytical and tabulated PSDs, interpolation, Welch estimation, variation, and the TimeSeries PSD wrappers that first consume them.

Estimated PSDs for compatible single-precision Torch strain use the original CPU PSD pipeline, including FFT rounding and inverse-spectrum truncation, then return to the selected device with preserved metadata. Other supported PSD routes retain their existing dispatch. Ordinary CPU Welch/PSD arithmetic is unchanged.

## Links to any issues or associated PRs

Depends on [#6](https://github.com/xangma/pycbc/pull/6). PR #20 remains withdrawn.

Base: `torch-pr2-fft`. Head: `torch-pr3-psd` at `c5a9da521dbaf7a4a9960b2bf877118e9557355c`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/8faf20dc51ae957e949dee409094eab6da4f22f3/torch-parity-fix); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/8faf20dc51ae957e949dee409094eab6da4f22f3/torch-parity-fix/final-publication).

## Testing performed

Owning-branch PSD tests: **73 passed, 17 skipped**.

Main integrated tests: **535 passed plus 8 subtests, 156 skipped**; no-Torch CPU checks **22 passed**. Linux CUDA regressions: **178 passed, 4 skipped**. CI-scoped F401 and changed-file Ruff formatting passed. The existing frame-reader B904 finding is retained and reported. Strict Sphinx built all **19 scoped main-stack pages**; rendered sources match final main documentation.

Completed main-stack campaign at `88878b1c38c952e63002b812058a0c7316123f70`, mapped to final main `7a72fba101b80467ebcdabe33ee973b9816094fb`: **CPU preservation PASS; original CPU, candidate CPU, Torch CPU and Torch CUDA each produce 1988 triggers**. All five comparisons pass unchanged trigger/full-PSD gates, with no missing or extra trigger identities or numerical violations. All 18 original-versus-candidate CPU scientific H1 datasets and full PSD arrays are byte-identical. Conditioned-strain equality is verified by full-array hashes and metadata; raw conditioned strain was not archived. Geometry matches.

Four recorded elapsed-time-derived fields are excluded from scientific byte equality; raw provenance verdicts and the two declared provenance substitutions are retained. Independent verification recomputed comparisons and checked source/receipt hashes. **No performance samples or speedup claim**; synchronous CUDA host transfers need separate performance measurement.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
