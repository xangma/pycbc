# Numerical qualification policy, version 2

**Adopted on 2026-09-07 before new acquisition. This separately versioned policy applies only to the new R4 campaign.**

The original campaign remains failed under its version-1 rule (`rtol=1e-4`, `atol=1e-6` on raw unnormalized complex FFT outputs). Its inputs, outputs, verdicts and scripts are preserved. Linux diagnosis verified 138 selected frozen output hashes and reproduced all complex output values exactly and found that the standard CPU/MKL reference itself violates that envelope against a complex128 oracle. At batch 8, the more accurate Torch CPU output is penalized by comparison with that reference. This motivates a change in the measured quantity and reference, rather than a runtime change intended to mimic MKL rounding.

## Acceptance criteria

For every block, template and complex sample, compare the actual normalized filter output against an independent oracle:

`abs(actual_complex_SNR - oracle_complex_SNR) <= 0.001`.

Compute the oracle from the exact complex64 bank and overwhitened strain inputs promoted to complex128 **before** multiplication. Apply an unnormalized NumPy complex128 inverse FFT. Calculate oracle template power separately in float64 from the bank and PSD, then multiply the output by `4 * delta_f / sqrt(oracle_sigmasq)`.

Actual output must include the normalization used by the route. Qualification must observe the normalization and sigma arrays used by the default bulk path, or the actual scalar path's sigma callback, for every template. Verify the original function and group identities and the executed branch; missing or ambiguous observation fails qualification. Compare every actual sigma against the independent sigma with the existing relative `1e-3` budget. Do not infer every route's normalization from a shared fixture value. Observation is installed only during qualification and restored on exit; timing remains uninstrumented.

Also require every complex SNR sample to agree with the actual standard CPU/MKL reference within `0.001`. The standard reference must itself pass the independent oracle check. Every sample and normalization must be finite; every template must be processed exactly once in each block. There are no percentile exceptions.

The `0.001` budget comes from the existing trigger-SNR tolerance in the production benchmark. Extending it to the complete complex time series is a new engineering acceptance criterion, not a pre-existing repository guarantee. All existing trigger identity, exact peak index, end time, phase, SNR, normalization and power/sine-Gaussian chi-square comparisons remain unchanged. A small complex error alone does not guarantee unchanged threshold decisions or peak selection near a tie; the exact trigger checks remain necessary.

Report per-row relative L2 residuals against the independent oracle, and FFT-only residuals against a complex128 transform of the captured correlation input. These are diagnostics, with no invented universal norm threshold. Zero-input and nonfinite handling must be explicit. Retain the old raw pointwise metrics as separately named diagnostics, without changing their historical verdict.

## New experiment

- Source remains `9578a710479b924e882857c4dffab6ed372a634b`; no numerical implementation change is proposed.
- Freeze two previously unexamined seeds, `7102` and `7103`, before acquisition. Use the same fixed 1,024-template bank and three strain blocks for all execution batches within each seed.
- Run 12 small smoke qualifications and 36 full qualifications across standard CPU/MKL, Torch CPU and Torch CUDA, with batches 1, 8, 32, 128, 512 and 1024.
- Both complete seed matrices must pass before any timing worker starts. Use seed `7102` for the unchanged 54-worker timing design: three fresh processes per route/batch, two warmups and five measured iterations.
- Qualification may instrument output capture; timing must use the uninstrumented public `LiveBatchMatchedFilter.process_data` method and validate every call's triggers outside the clock.
- Freeze policy, worker/controller, source, native extensions, all input identities, raw and oracle row hashes. Never overwrite a prior campaign directory.

This remains a serial, one-thread, affinity-pinned shared-host experiment on len. It does not establish reserved-host throughput or full-machine capacity.
