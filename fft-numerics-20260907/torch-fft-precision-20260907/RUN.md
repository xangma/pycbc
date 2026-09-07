# Linux FFT precision diagnosis

Completed at 2026-09-07 12:28:47 UTC. All four cells finished successfully and all 138 captured output hashes matched their frozen counterparts. The process group has exited. Diagnostic only; this run cannot qualify or start timing. Source remains `9578a710479b924e882857c4dffab6ed372a634b`. It captures selected outputs and correlation rows through the frozen public filter at batches 1 and 8, compares exact frozen hashes, then distinguishes correlation error from FFT error using complex128 and direct DFT checks.

- Host: len
- Cwd: /home/xangma/pycbc-torch-fft-precision-20260907
- Command: /home/xangma/pycbc-torch-split-20260905/venv/bin/python /home/xangma/pycbc-torch-fft-precision-20260907/run.py
- PID/PGID: 1933459 (exited)
- Log: /home/xangma/pycbc-torch-fft-precision-20260907/launch.log (individual cell logs alongside)
- Status: /home/xangma/pycbc-torch-fft-precision-20260907/status.json
- Stop: no active process remains.
- Next action: R4 benchmarking under adopted policy v2; no diagnostic process remains.

All measurements are serial on logical CPU 8, with one library thread and one CUDA device. The previous campaign is terminal: six full CPU/MKL passes, twelve Torch failures, all trigger comparisons passed, all 54 timing workers skipped. Its evidence is unchanged.

The independent complex128 oracle demonstrates that the CPU/MKL reference itself violates the old raw pointwise tolerance on selected cancellation-dominated samples. At batch 8, Torch CPU passes that comparison against the oracle for all selected rows, while CPU/MKL fails 22 samples. Direct DFT evaluations agree with the double FFT to about 1e-11. This shows that agreement with the single-precision MKL reference under this particular raw rule is insufficient as an accuracy target. The later attribution replay isolates the arithmetic stages in `../torch-fft-attribution-20260907-r2/REPORT.md`.

An offline audit regenerated the exact template/PSD/strain/normalization hashes and multiplied every frozen row's maximum raw error by the common input-derived SNR normalization. Across all 18 full cells and every complex sample, that maximum is 4.0996983e-6 SNR, and the maximum per-row raw relative L2 error is 3.4456342e-7. These values exclude differences in the normalization actually used by each route; they are diagnostic, not a complete normalized-output qualification. All original trigger checks pass, all values are finite, and every template is processed exactly once. This audit does not change the original failed campaign status or authorize timings. See `normalized-error-audit.json` and per-cell `diagnostic.json` for full provenance.

`plot-accuracy.py` regenerates `current-batch-accuracy.png` and `.svg` from those frozen results. The right panel uses the common normalization and labels that limitation explicitly. No throughput plot is available because all timing workers were skipped.
