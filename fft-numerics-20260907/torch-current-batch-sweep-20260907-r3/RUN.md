# Current batch-size sweep, third attempt

The campaign finished at 2026-09-07 09:57:42 UTC with **qualification failed**. All six smoke checks passed. Full qualification completed all eighteen cells: six standard CPU/MKL passes and twelve Torch CPU/CUDA failures, covering every batch size. The controller and all recorded worker process groups have exited. All 54 timing workers were skipped; there is no qualified timing result or current batch-size plot.

For batch 1, trigger identities, indices, times, phase, SNR, normalization and both chi-square vetoes all pass. The pointwise check fails on 132 Torch CPU and 541 Torch CUDA samples out of 402,653,184 complex samples per route; all samples are finite and processed exactly once. The maximum error-to-tolerance ratios are 2.204 and 3.563 respectively. Tolerances remain unchanged (`atol=1e-6`, `rtol=1e-4`). This is separate from the normalization and stale correlation defects already repaired.

The completed Linux attribution replay reproduced all complex values in 138 captured candidate rows and 65 distinct reference row/block pairs. CPU correlations match exactly; the remaining CPU discrepancy is FFT rounding. CUDA also differs slightly in correlation, with FFT rounding dominant in the captured failures. A more accurate promoted CPU FFT can fail against the single-precision MKL reference. The [attribution report](../torch-fft-attribution-20260907-r2/REPORT.md) records the signed decomposition and selected-row limitations.

The user subsequently accepted the documented diagnosis and requested continued benchmarking. The fresh [R4 campaign](../torch-current-batch-sweep-20260907-r4/RUN.md) applies a separately adopted complete-complex-SNR criterion to two new seeds, including the route's observed normalization and an independent complex128 oracle. R3's raw criterion and failed verdict are unchanged.

Source: `9578a710479b924e882857c4dffab6ed372a634b`, clean pinned checkout. The fixes keep template normalization separate across shared workspaces, preserve the full correlation length used by the veto, and invalidate stale sibling correlation rows. Local focused tests: 66 passed, 1 skipped. Pinned controller/worker tests: 31 passed. All eleven native-extension hashes and unchanged native build inputs were verified before staging.

The fixed bank and input stay identical across batches 1, 8, 32, 128, 512 and 1024. Routes are standard CPU/MKL, Torch CPU and Torch CUDA. Six small and eighteen full scientific qualifications must pass before 54 fresh timing workers (three per cell) run. Qualification checks every complex filter sample and structured trigger values with unchanged tolerances. The full workload uses 1,024 distinct templates, FFT length 131,072 and three input blocks. Each timed worker runs two warmups and five measured public `LiveBatchMatchedFilter.process_data` calls on one pinned logical CPU, with library thread counts fixed at one.

- Host: `len`
- Cwd: `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3`
- Command: `/home/xangma/pycbc-torch-split-20260905/venv/bin/python /home/xangma/pycbc-torch-current-batch-sweep-20260907-r3/batch-campaign.py --shared-host`
- PID/PGID: `1834195` / `1834195` (exited)
- Log: `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3/launch.log`
- Status: `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3/batch-status.json`
- Stop: none needed; campaign is terminal and recorded process groups are absent.
- Follow-on: the independently frozen R4 campaign; this failed campaign will not restart.

The initial and second failed campaigns are preserved as diagnostic evidence in their original directories. Their timing gates correctly prevented invalid performance results. Historical batch-size claims, two plots and their renderer paths have been removed from PR #15. Retained non-batch figures are unchanged.

The reusable production benchmark is `tools/bench_production_live_batch.py`; this campaign's `batch-worker.py` also exposes route, batch, bank size, FFT size, seed, device, thread and repeat controls. The campaign launcher still pins the source revision and len paths; it is a reproducible experiment, not yet a portable packaged command. A new plot renderer is pending a fully qualified timing summary.

Profiling and optimization remain held until this sweep completes successfully and their baseline and predecessor gates are reviewed against the new revision. These are shared-host measurements and do not establish reserved-host throughput.
