# CPU optimization assessment after workspace-policy closure

The bounded review of existing evidence identifies no distinct CPU candidate ready for another trial under the preserved numerical requirements and one-core workload. CPU parity remains unmet. This is an assessment of the available evidence, not proof of a performance limit or an exhaustive search of possible algorithms. No benchmark, source change or gate relaxation is proposed.

The most recent completed Torch CPU executable comparison remains 104.400193 s versus contemporaneous standard CPU 66.933429 s: a historical 37.466764 s gap. Later descriptor-reuse and CUDA-graph campaigns did not measure a new Torch CPU gap. Their improvements must not be combined arithmetically with this earlier comparison. [Executable handoff](/Users/xangma/repos/pycbc/artifacts/torch-fft-optimization-20260908/HANDOFF.md).

The accepted promoted in-place IFFT measured 33.409 ms through the engine, including 28.861 ms at the native DFTI boundary and 4.492 ms in paired conversions. Bookkeeping measured about 0.011 ms, and whole-engine/direct-plan ranges overlapped. These observations support prioritizing native execution; they establish neither unavoidable precision cost nor an executable savings forecast. That stage experiment recorded inter-op 64; executable and latest diagnostics enforce 1. [Stage review](CPU-STAGE-REVIEW-v1.md).

| Examined route | Existing evidence and disposition |
|---|---|
| Direct MKL complex64 | The production-length seed-812 dense 1e12 case exceeds the unchanged maximum-error limit. Preserve the failure. |
| Private in-place FFTW complex64 MEASURE | The final native qualification fails maximum-error comparisons for seeds 812 and 20260906, dense 1e-12. No timing advancement. |
| Retained out-of-place FFTW complex64 MEASURE | Earlier qualification and actual timed-plan checks passed. Its 32.003 ms warm median came with 6.542 s median construction, versus 33.267 ms and 0.069 s for the diagnostic in-place MKL plan. This already examined route supplies no new finite-executable result. Do not describe all complex64 routes as numerical failures. |
| Promoted FFTW MEASURE | Already examined: 40.011 ms warm median and 15.899 s median construction in the same historical experiment. No evidence that it improves on the in-place MKL route. |
| NumPy conversion copies | Qualified but slower in all three fresh workers; retain Tensor.copy_. |
| Huge-page workspace | Qualified with verified page backing but whole-engine calls were 29.556–29.998% slower than the matched no-huge control. The control itself was not qualified as a new optimization. |
| DFTI workspace AVOID | First case changes native complex128 bytes, despite identical final complex64 bytes and acceptable final errors. Rejected under the frozen native/final parity rule; zero timing workers. |

The FFTW timings above are medians of three historical process medians; construction and warm execution remain separate. They are not measurements of the current production implementation. Exact input records and recomputed values are pinned in [the assessment data](cpu-constraint-assessment-v1.json). [Earlier FFT inventory](cpu-fft-review.md), [completed CPU review](FOLLOWUP-REVIEW.md), [copy result](RESIDUAL-ANALYSIS-v2.md), [page-backing result](RESIDUAL-ANALYSIS-v4.md), [workspace-policy report](/Users/xangma/repos/pycbc/artifacts/torch-cpu-workspace-policy-20260908/CPU-WORKSPACE-POLICY-v1.md).

Other recorded costs provide no concrete next mechanism for closing the filtering gap. The older profile assigns about 1.45 s to squared norm and 1.19 s to trigger deep copying; the previous deep-copy candidate had no reproducible executable gain. Repeated descriptor creation has already produced a validated shared setup improvement. Roughly one second in two first large creations was not shown reusable and is not evidence of a Torch CPU filtering optimization. These older observations are priorities only, not additive headroom estimates. [Profile inventory](cpu-fft-review.md), [setup attribution](RESIDUAL-ANALYSIS-v2.md), [accepted descriptor reuse](RESIDUAL-ANALYSIS-v3.md).

Keep the accepted source and sealed experiments unchanged. A future trial needs new evidence tying a distinct mechanism to the retained native transform or another substantial current cost, with unchanged qualification and a fair updated standard comparison. Existing small costs and rejected-route controls do not supply that evidence.

Workspace-policy closure independently passed: all 35 archive members, 34 result entries, 23 frozen inputs, native contracts, rejection/cleanup records, process-group absence and lock reacquisition were verified. The final handoff's 75 outer checksums also match. This assessment used local records only and reran no FFT. [Terminal review](workspace-policy-results-v1-review.json).
