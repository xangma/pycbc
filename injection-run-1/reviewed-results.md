# Captured-input injection results

Completed on len in 47.47 seconds (worker 46.38 seconds), MKL, affinity [8], long-double mantissa 63 bits. Both processes exited successfully.

All six saved null cases reproduced exactly. All 48 cases completed; input and source hashes remained unchanged. Downloaded results match the remote SHA-256 hashes.

| Input | Cases | Peak at injected sample | Peak matches independent reference | Boundary peaks | Largest timing offset |
|---|---:|---:|---:|---:|---:|
| signal_only | 24 | 24 | 24 | 0 | 0 ms |
| captured_noise_plus_signal | 24 | 22 | 24 | 0 | 5.37109 ms |

The two shifted noise-plus-signal peaks were template 370: rho=5.5 shifted +22 samples (5.3711 ms), and rho=8 shifted +15 samples (3.6621 ms). Both shifts also occur in the independent reference. Original and proposed variants share the SNR calculation, so their timing is identical by construction.

Maximum signal-only independent calibration fractional error was 1.14e-10. The largest native peak-SNR error against the independent reference was 4.81e-6.

| Maximum absolute error at the common raw-SNR peak | Signal-only original | Signal-only proposed | Noise-plus-signal original | Noise-plus-signal proposed |
|---|---:|---:|---:|---:|
| Chi-square against mathematical statistic | 0.923921 | 0.0001921017 | 1.812089 | 0.0001154936 |
| Point evaluator with native correlation and SNR held fixed | 0.9239326 | 1.485494e-09 | 1.812095 | 1.900422e-06 |
| newSNR against mathematical statistic | 4.801483e-06 | 4.801483e-06 | 0.3156338 | 2.149664e-05 |

Each bin set has its own mathematical reference. The table measures numerical error for that statistic; a change between bin sets also changes the statistic itself. The stable mathematical reference can differ slightly from the point-evaluator reference because the latter retains rounded production correlations and the FFT SNR subtraction.

Signal-only newSNR was identical between variants in every case because no chi-square penalty applied. Noise-plus-signal newSNR changes ranged from -0.972324 to +0.586638 (proposed minus original); the individual values are in results.json.

The largest absolute change occurred for template 370, target rho=20, in captured noise: raw peak SNR 14.742418; original newSNR 14.742418 and proposed newSNR 13.770094. Reduced chi-square changed from 0.942813 to 1.262386. Re-evaluating the original bins with proposed arithmetic gives chi-square 28.188810, while proposed bins give 37.871593. This example is dominated by the changed bin partition, which introduces a ranking penalty; it is not evidence of improved detection efficiency.

Negative raw chi-square values are retained in results.json. Small proposed residuals in signal-only cases arise from the production FFT subtraction and rounding; their size should be assessed against the separate numerical references.

These are matched-template injections into fixed conditioned spectra using the frozen proposed PSD. The six locations were selected from existing triggers. This run establishes controlled numerical recovery for these inputs; it makes no FAR, population sensitivity, full search acceptance, conditioning/gating-response or PSD-response claim.

Raw outputs: [results.json](results.json), [worker log](probe.log), [launcher receipt](launcher-receipt.json), [generated report](report.md).
