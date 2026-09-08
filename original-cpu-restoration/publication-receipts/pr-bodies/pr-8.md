Support Torch matched filtering, conditioning and signal-consistency vetoes.

## Standard information about the request

This is a: new feature, bug fix and efficiency update.

This change affects: the Python interfaces used by search and inference.

This change: includes focused regressions and follows the repository contribution workflow. Torch remains optional; no low-level C or Cython sources are changed in the main stack.

This change changes: Python implementation and its numerical/interface regression tests.

This change will: retain the optional Torch dependency introduced by #5.

## Motivation

Torch-backed strain must pass through conditioning, correlation and signal-consistency checks before it can produce search triggers.

## Contents

Correlation, resampling and strain conditioning, threshold/clustering primitives, chi-square vetoes and Q-transform. Includes the existing 92-line phase-shift helper at its first caller; retains SciPy’s scalar optimizer. Uses existing native kernels without Cython changes. Timing regressions use resolved broadband peaks while preserving the original sine-wave overlap checks and tolerances.

This PR adds Torch dispatch and backend implementations, with Torch precision coverage. Ordinary CPU template-power accumulation, point chi-square arithmetic and edge behavior are preserved; the three standalone CPU-correction tests from #20 are excluded from the restored stack. CUDA scalar normalization uses stream-ordered arithmetic, with optional output buffers and contract coverage.

## Links to any issues or associated PRs

Depends on [#7](https://github.com/xangma/pycbc/pull/7). PR #20 is withdrawn from this stack.

Base: `torch-pr3-psd`. Head: `torch-pr4-filtering` at `941a119e00f7da3e72daa98a8ad60e5264a4ae1c`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/a5d237cb4f4ed7a4e26d6b8ffa4aa0c1fc7b2964/original-cpu-restoration/final-publication).

## Testing performed

Owning-PR filtering/veto JUnit results: **218 successful, 110 skipped**, including subtests.

Completed main-stack campaign at `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`: **CPU preservation PASS, 1988 = 1988 triggers**; all 18 scientific H1 datasets and full PSD arrays are byte-identical, with matching geometry. Conditioned-strain identity is verified by full-data SHA256 and metadata; raw conditioned strain was not archived.

**Torch CPU and Torch CUDA: 1991 triggers each, FAIL** against both CPU controls under unchanged trigger/full-PSD gates. Only four recorded runtime/timing fields are excluded; raw and corrected verdicts are retained. No performance samples or speedup claim.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
