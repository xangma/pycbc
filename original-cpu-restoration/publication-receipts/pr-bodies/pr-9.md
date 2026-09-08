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

## Links to any issues or associated PRs

Depends on [#8](https://github.com/xangma/pycbc/pull/8). PR #20 is withdrawn from this stack.

Base: `torch-pr4-filtering`. Head: `torch-pr5-search` at `307d349013a058ad6b2aa61a111cb9c75882427c`.

Unchanged original CPU reference: `40e94792b3edf59f39b18b65102b28a4f74433a7`.

[Campaign and test evidence](https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration); [tested-source mapping and documentation checks](https://github.com/xangma/pycbc/tree/a5d237cb4f4ed7a4e26d6b8ffa4aa0c1fc7b2964/original-cpu-restoration/final-publication).

## Testing performed

Owning-PR search JUnit results: **79 successful, 21 skipped**.

Completed main-stack campaign at `aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`: **CPU preservation PASS, 1988 = 1988 triggers**; all 18 scientific H1 datasets and full PSD arrays are byte-identical, with matching geometry. Conditioned-strain identity is verified by full-data SHA256 and metadata; raw conditioned strain was not archived.

**Torch CPU and Torch CUDA: 1991 triggers each, FAIL** against both CPU controls under unchanged trigger/full-PSD gates. Only four recorded runtime/timing fields are excluded; raw and corrected verdicts are retained. No performance samples or speedup claim.

## Additional notes

This PR was created by AI Gareth

Required PR label: `agent-assisted`.

- [ ] The author of this pull request confirms they will adhere to the [code of conduct](https://github.com/gwastro/pycbc/blob/master/CODE_OF_CONDUCT.md)

*AI Agent Note: Unchecked by default. @xangma, please review this PR and check the Code of Conduct box above to confirm your agreement before requesting review.*
