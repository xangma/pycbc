Provide optional Torch storage and execution through PyCBC’s existing scheme and array interfaces.

## Standard information about the request

This is a: new feature, bug fix and efficiency update.

This change affects: the Python interfaces used by search and inference.

This change: includes focused regressions and follows the repository contribution workflow. Torch remains optional; no low-level C or Cython sources are changed in the main stack.

This change changes: Python implementation and its numerical/interface regression tests.

This change will: provide Torch as an optional installation extra.

## Motivation

PyCBC's scheme and array interfaces need a shared Torch storage contract so later algorithms can select a device and exchange data through the existing container APIs.

## Contents

Scheme/device/thread selection, array storage, NumPy interoperability, backend access, metadata and conversion helpers. Torch remains an optional installation extra.

Large complex Torch CPU squared norms use separate real and imaginary squares while preserving views, dtype and differentiation. Ordinary-CPU array-copy and processing-context behavior is preserved; Torch-specific runtime setup stays in the Torch scheme.

## Links to any issues or associated PRs

Depends on [#18](https://github.com/xangma/pycbc/pull/18). PR #20 is withdrawn from this stack.

Base: `torch-formatting-base`. Head: `torch-pr1-runtime` at `16f246be7e48ad16dd9043ff6351c52ced531300`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/a5d237cb4f4ed7a4e26d6b8ffa4aa0c1fc7b2964/original-cpu-restoration/final-publication).

## Testing performed

Owning-PR runtime/array tests: **39 passed, 1 skipped**.

Completed main-stack campaign at `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`: **CPU preservation PASS, 1988 = 1988 triggers**; all 18 scientific H1 datasets and full PSD arrays are byte-identical, with matching geometry. Conditioned-strain identity is verified by full-data SHA256 and metadata; raw conditioned strain was not archived.

**Torch CPU and Torch CUDA: 1991 triggers each, FAIL** against both CPU controls under unchanged trigger/full-PSD gates. Only four recorded runtime/timing fields are excluded; raw and corrected verdicts are retained. No performance samples or speedup claim.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
