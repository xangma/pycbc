# Cause of the remaining batch discrepancies

The captured raw-output discrepancies arise primarily from the inverse FFT's floating-point arithmetic. At the tested length, 131072, the production routes use different FFT libraries and internal precisions. CPU correlations are exactly equal elementwise to the standard reference, so their differences arise entirely in the FFT. CUDA also has a smaller difference in correlation multiplication, but its FFT error remains when both FFTs receive exactly the same correlation.

This conclusion is supported by a replay on len using source `9578a710479b924e882857c4dffab6ed372a634b`. Every one of 138 captured candidate rows was reproduced with exact elementwise equality, and all 65 distinct standard-reference row/block pairs were reproduced from regenerated inputs through the native batch correlator and MKL. Input hashes, captured output hashes, and the replay script hashes were checked. These are selected diagnostic rows from three blocks and four route/batch combinations, not a new full-bank qualification.

| Route at N=131072 | Observed FFT route | Internal precision |
|---|---|---|
| Standard CPU reference, batch 1 | PyCBC MKL `DftiComputeBackward` | complex64 |
| Torch CPU, batch 1 | `_FFTWCPUDirectPlan`, retained FFTW output | complex64 |
| Torch CPU, batch 8 | `_TorchPromotedBatchPlan`, `torch.fft.ifft`, then cast back | complex128, public complex64 |
| Torch CUDA, batches 1 and 8 | Direct `torch.fft.ifft` | complex64 |

The source excludes 131072 from the direct-MKL single-transform size list. CPU batch FFTW and direct Torch batch IFFT default off, causing larger CPU batches to use the promoted workspace. Direct Torch batch IFFT defaults on for CUDA. The same larger-batch CPU routing applies to the other tested batch sizes by source inspection; this replay directly exercises batches 1 and 8.

Relevant source: [IFFT dispatch](/private/tmp/pycbc-torch-benchmark-controls-20260907/pycbc/fft/torchfft.py:1444), [MKL eligibility](/private/tmp/pycbc-torch-benchmark-controls-20260907/pycbc/fft/torchfft.py:913), [FFTW selection](/private/tmp/pycbc-torch-benchmark-controls-20260907/pycbc/fft/torchfft.py:990), [promotion](/private/tmp/pycbc-torch-benchmark-controls-20260907/pycbc/fft/torchfft.py:1217), [direct batch gate](/private/tmp/pycbc-torch-benchmark-controls-20260907/pycbc/fft/torchfft.py:1299).

## What the replay isolates

The exact-product oracle promotes the stored complex64 template and strain inputs before multiplying, then uses a complex128 inverse FFT. A second oracle transforms the actual rounded correlation in complex128. The difference between those oracles isolates correlation rounding; the difference between the actual output and the second oracle isolates FFT error, including the final public-output rounding. The earlier diagnostic independently checked selected samples using direct DFT summation, agreeing with the double FFT to approximately 1e-11.

For comparison with the standard reference, the replay reconstructs its native correlation and uses the exact signed identity:

`candidate - reference = (candidate - F64(candidate_correlation)) + (F64(candidate_correlation) - F64(reference_correlation)) + (F64(reference_correlation) - reference)`

Here `F64` is the double-precision FFT. The JSON's signed reference term is the correction to the reference, the negative of its usual error.

The recorded terms sum exactly to the measured complex discrepancy at all 117 captured failing samples. Their magnitudes should not be added as scalar error percentages; complex errors can reinforce or cancel.

| Captured cell | Rows | Fails against MKL | Fails against exact-product oracle | Oracle fails after only promoting FFT, retaining complex64 output |
|---|---:|---:|---:|---:|
| Torch CPU B1 | 36 | 28 | 10 | 0 |
| Torch CPU B8 | 36 | 21 | 0 | 0 |
| Torch CUDA B1 | 33 | 34 | 27 | 0 |
| Torch CUDA B8 | 33 | 34 | 27 | 0 |

All counts retain the original raw criterion, `abs(error) <= 1e-6 + 1e-4*abs(reference)`, with the stated comparison target. The oracle is a diagnostic target, not an adopted replacement campaign criterion. Rows overlap across cells, and these counts must not be presented as full-bank counts. Batch-8 replay broadcasts each captured correlation across eight rows and checks row zero; it reproduces the arithmetic of an individual row, not the original mixed batch. The precision intervention is a standalone one-row complex128 FFT followed by complex64 conversion; it does not qualify a production implementation or its performance.

For CPU, all 72 captured correlations match standard CPU with exact elementwise equality. For CUDA, correlation samples differ by at most 2.11e-8; their transformed difference is at most 9.86e-7 across the captured outputs. Giving MKL and CUDA the same correlation still produces 37 raw failures per captured CUDA cell. Correlation contributes up to 27% of the pair discrepancy at an individual failing sample, so it is smaller but not universally negligible.

## Two concrete failures

For CUDA, block 0/template 1510/sample 15566:

| Quantity | Raw complex magnitude |
|---|---:|
| Candidate/reference discrepancy | 2.4126e-5 |
| Allowed discrepancy | 6.7727e-6 |
| Candidate FFT residual | 2.2884e-5 |
| Transformed CPU/CUDA correlation difference | 3.2697e-7 |
| Reference FFT residual | 1.9229e-6 |

The CUDA FFT accounts for the dominant contribution here. The transformed frequency components cancel strongly: the sum of their magnitudes is about 15136 times the magnitude of the final sample.

For CPU batch 8, block 1/template 1606/sample 106225, correlations are identical. The Torch FFT residual is only **3.76e-10**, while the MKL FFT residual is **2.6120e-6**. Their discrepancy is **2.6118e-6**, exceeding the gate's **1.8303e-6** limit. In this case the more accurate Torch result fails because the single-precision reference has the larger error.

Across the captured failures, cancellation ratios range from about 3112 to 1.69 million. Different FFT implementations and accumulation precisions produce different rounding residues; comparing those residues relative to a small, cancellation-dominated output can exceed the pointwise rule despite a small overall error. This explanation is consistent with [PyTorch's numerical-accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) and [FFTW's accuracy discussion](https://www.fftw.org/accuracy/comments.html). This investigation establishes the arithmetic stage and precision dependence; it does not identify individual internal FFT butterflies or twiddle-factor instructions.

These mismatches precede SNR normalization, peak selection, and veto evaluation. The earlier full campaign's trigger/veto comparisons all passed. Its separate common-normalization audit bounded the raw pair differences at 4.10e-6 SNR; that audit excludes route-specific normalization differences and is not a complete normalized-output qualification.

## Reproduction and status

`replay.py` reconstructs the inputs from the frozen R3 worker and reads the earlier captured NPZ arrays. `run.py` runs it serially on logical CPU 8 with a six-minute timeout. `replay.json` contains per-row metrics, exact source identity, observed plan classes, captured-file hashes, and signed error decompositions. `summary.json` is a compact aggregation. The script checks exact candidate and reference reproduction before accepting evidence. The replay booleans named `*_bitwise_equal` use `np.array_equal`: they establish exact numerical equality but do not distinguish signed-zero bits. Hashes verify the original saved candidate/reference arrays, not replay-array bit patterns. The same signed-zero qualification applies to correlation equality.

- Host: len; cwd: `/home/xangma/pycbc-torch-fft-attribution-20260907-r2`.
- Command: `/home/xangma/pycbc-torch-split-20260905/venv/bin/python run.py`.
- Supervisor PID/PGID: 1023408; completed successfully after 16.74 seconds.
- Logs: `replay.log`, `launch.log`; process group verified absent afterward.
- No stop command is needed; no diagnostic process remains.
- Source remained clean at the pinned revision. Original campaign results and criteria remain unchanged.
- Initial attempt in the sibling directory without `-r2` stopped on a diagnostic buffer-copy type error. Its failed status and log are preserved; it provides no numerical conclusions.

The evidence supports FFT precision/backend differences as the cause of these raw mismatches. The diagnostic itself does not authorize a change in internal precision or scientific qualification. On 2026-09-07 the user subsequently accepted the explanation and requested continued benchmarking. R4 therefore records a separately adopted complete-complex-SNR criterion before acquisition, uses fresh seeds 7102 and 7103, and leaves production FFT precision unchanged. See [R4 policy](../torch-current-batch-sweep-20260907-r4/NUMERICAL-POLICY.md) and [launch/status](../torch-current-batch-sweep-20260907-r4/RUN.md).
