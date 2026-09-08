Execute FFT operations for Torch-backed PyCBC series.

## Standard information about the request

This is a: new feature, bug fix and efficiency update.

This change affects: the Python interfaces used by search and inference.

This change: includes focused regressions and follows the repository contribution workflow. Torch remains optional; no low-level C or Cython sources are changed in the main stack.

This change changes: Python implementation and its numerical/interface regression tests.

This change will: retain the optional Torch dependency introduced by #5.

## Motivation

Filtering and PSD routines need FFTs that accept Torch-backed series while preserving PyCBC's normalization, output-buffer and batch contracts.

## Contents

Torch FFT dispatch and batch contracts, hardware capabilities, and Torch-owned reuse of existing CPU FFTW plans. Generic NumPy/CuPy/cuFFT changes and the optional wisdom lifecycle are a separate follow-up. Ordinary-CPU FFT validation retains its original behavior.

Eligible one-thread large Torch inverse FFTs use a retained complex128 MKL workspace with preserved complex64 input. The 2M-point workspace executes in place. Native-plan, batch, output-buffer, precision and fallback contracts have focused regressions.

Torch-specific planner locking and workspaces belong to the Torch FFT implementation. The shared CPU MKL, FFTW and NumPy FFT files in the main stack are byte-identical to the original CPU baseline; no shared MKL function-descriptor cache is added.

## Links to any issues or associated PRs

Depends on [#5](https://github.com/xangma/pycbc/pull/5). PR #20 is withdrawn from this stack.

Base: `torch-pr1-runtime`. Head: `torch-pr2-fft` at `45e75586f8cf2f9ff0a6818bed706d453a48dc20`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/a5d237cb4f4ed7a4e26d6b8ffa4aa0c1fc7b2964/original-cpu-restoration/final-publication).

## Testing performed

Owning-PR FFT tests: **74 passed, 3 skipped**.

Completed main-stack campaign at `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`: **CPU preservation PASS, 1988 = 1988 triggers**; all 18 scientific H1 datasets and full PSD arrays are byte-identical, with matching geometry. Conditioned-strain identity is verified by full-data SHA256 and metadata; raw conditioned strain was not archived.

**Torch CPU and Torch CUDA: 1991 triggers each, FAIL** against both CPU controls under unchanged trigger/full-PSD gates. Only four recorded runtime/timing fields are excluded; raw and corrected verdicts are retained. No performance samples or speedup claim.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
