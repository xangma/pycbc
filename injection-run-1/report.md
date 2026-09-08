# Fixed-input injection probe

Status: complete. Capture null gate: True.

The PSD, conditioned spectra, injection spectrum and captured normalization are shared. Only bin construction and point chi-squared arithmetic vary. SNR timing is shared by construction.

| Input | Cases | Peak at injection | Peak agrees with reference | Edge peaks | Max newSNR change |
|---|---:|---:|---:|---:|---:|
| signal_only | 24 | 24 | 24 | 0 | 0 |
| captured_noise_plus_signal | 24 | 22 | 24 | 0 | 0.972324 |

All per-case target/peak values, four bin/evaluator combinations, independent references, source pins, and input hashes are in results.json. Negative raw chi-squared values are preserved. An edge peak requires wider-window follow-up before interpreting recovery.

These six locations were selected from existing triggers and are not an unbiased noise sample. This checks matched-template numerics after conditioning; it does not measure FAR, detection efficiency, population sensitivity, full trigger acceptance, gating response, PSD response, or waveform/model mismatch.
