The qualification at source `968bcd558117262af0d603710b054174659adb51` remains **failed**. This v1 summary binds eight existing diagnostic records; it reports demonstrated numerical effects and their limits. It establishes no completed fix, passing replacement qualification, or performance result. Scientific tolerances remain unchanged (`atol=1e-5`, `rtol=1e-4`, phase absolute tolerance `1e-4`, normalization relative tolerance `1e-5`).

| Backend versus CPU | CPU / candidate / matched triggers | Chi-squared violations | SNR violations | Phase violations |
| --- | --- | ---: | ---: | ---: |
| Torch CPU | 477 / 475 / 469 | 469 | 0 | 0 |
| Torch CUDA | 477 / 475 / 469 | 468 | 19 | 27 |

Both comparisons also have eight CPU-only and six candidate-only triggers. One Torch CPU candidate-only trigger is threshold-adjacent and requires review; the others are unexplained. Every unmatched trigger remains unaccepted. Maximum chi-squared differences are 7.61219 and 7.61141; CUDA also reaches SNR error 0.000844479 and phase error 0.000148296 radians.

The CPU capture wrapper failed after the full executable completed: its final assertion incorrectly expected five selected-template chi-squared calls, although only four segments contained triggers. All 44 verification checks pass, including bitwise equality of every H1 trigger column to the original CPU output. The failed receipt remains unchanged. The CUDA capture executable completed, and identities plus every other trigger column match bitwise, but 106 chi-squared entries differ, with 60 exceeding the frozen budget and maximum difference 0.167629. Successful capture execution is therefore insufficient evidence of numerical repeatability.

Identical-input tests cover one template (`4366715446675002219`), four CPU captures and eight samples. Maximum absolute errors against direct complex128 evaluation are:

| Diagnostic evaluation | Maximum absolute chi-squared error |
| --- | ---: |
| Locally compiled native float32 | 0.191856552 |
| Native float64, original pi constant | 0.00176387072 |
| Native float64, full-precision pi | 1.14563647e-09 |
| Torch CPU, identical inputs and bins | 1.71334084e-06 |

The local native float32 result does not reproduce the recorded Linux result bitwise. These controlled comparisons nevertheless demonstrate native numerical error for the captured inputs, and separate the effect of arithmetic precision from the pi constant. They do not certify the complete Torch pipeline.

A controlled bin swap at segment 3, sample 1884341 gives direct complex128 chi-squared values:

| Captured input arrays | CPU bins | CUDA bins |
| --- | ---: | ---: |
| CPU | 35.834825201 | 28.305221915 |
| CUDA | 35.836442436 | 28.306574851 |

Changing only the bins changes this statistic by approximately 7.53; changing only the other inputs changes it by approximately 0.0016. Thus bin selection explains the dominant discrepancy at this captured trigger. The CPU float32 cumulative sum reproduces its recorded edges, underestimates its float64 cumulative total by 0.5241%, and puts approximately 6.98% excess power in the final bin relative to equal power. Its penultimate edge is 273702, versus 362757 using float64 accumulation. This demonstrates error in the current CPU reference; it does not justify widening the qualification budgets or generalizing one swap to all mismatches.

On an RTX 4090 with Torch 2.13.0+cu130, 100 CUDA float32 cumulative scans of one fixed power vector produced three edge vectors: penultimate edges 362790 (81 runs), 362775 (17), and 362805 (2). Float64 accumulation **followed by a float32 cast** produced edge 362778 in all 100 runs. The fully float64 CPU reference edge is 362772. The observed stability is useful evidence, but neither an accuracy proof nor a completed fix.

The PSD/SNR v1 diagnostic completed with a clean source and its status binds the report hash. Relative L2 differences from the saved CPU PSD are 0.00169107 for Torch CPU and 0.00304023 for CUDA; each backend reproduces its own saved PSD. Welch and interpolation differences are only about 2e-7–4e-7, while differences remain after supplying common interpolated input to inverse spectrum truncation. Float64 controls also change the CPU reference. The selected waveform agrees to relative L2 3.09e-12, which is limited to that template. These results suggest conditioning precision merits further controls; they do not establish the cause of the campaign's worst SNR or phase failures. PSD substitutions hold the CPU data FFT fixed, use double ratios before complex64 rounding, and use double normalization sums. Their arithmetic differs from production. PSD/SNR conclusions remain provisional pending v2 controls.

Fresh qualifications must still resolve bin accuracy and repeatability, native chi-squared precision, unmatched triggers, and CUDA SNR/phase failures under the original budgets. The companion JSON retains exact values and scoped interpretations. The eight input files were hashed before and after synthesis without changes; four diagnostics also record equal before/after input manifests. The inspected repeatability script matches its recorded SHA-256 `4134611bc44be4fbc403df4414c9e91f5d466f5d86b34a0fb262c96008d76e2d`.

| Input, relative to this artifact directory | SHA-256 |
| --- | --- |
| `final-qualification-parity.json` | `45455fcdeb2fcf03ae8341f3ca813fd799ff6ef132c393a7117094b00561246c` |
| `chisq-input-capture-cpu/capture-verification.json` | `918dbe385cecf9ac2cb0f7b6e70e2665178432e9fc381f5e344dc3efb1afe33a` |
| `chisq-input-capture-cuda/capture-verification.json` | `d67877ed4ff9785eda48cb5fda2c994bdd40056d72b2995995e87c6e1ebd7e69` |
| `chisq-identical-input-diagnostic.json` | `8e3d5a33ed959d30196a228b487014e9adaeaa02a825ccd0751c6658ff153770` |
| `chisq-captured-bin-swap-diagnostic.json` | `5371f8bc3cf0c1db941ea7544ef8faa73ad010d940b0353c6c988d264c2a8791` |
| `chisq-cuda-prefix-repeatability.json` | `2460675d0c5d8c13dbd0f70422c41fc729483b321f4df1e3a0f95a1c4b247ba6` |
| `snr-psd-diagnostic-v1/report.json` | `6f63b3e4a9cd00cfaae5f0a1ea3852bfa352dfe987ef50e98b295b3295e646ab` |
| `snr-psd-diagnostic-v1.status.json` | `528e69c4cb6d2d7b72530cd2ea8e74d589a6ebe6a9bfc7538dc48de36f246012` |
