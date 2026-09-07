# Current batch-size sweep, numerical policy version 2

The campaign launched on 2026-09-07 after the user accepted the documented FFT-precision diagnosis and the new complete-complex-SNR criterion. Status is live in `batch-status.json`; a completed qualification or timing result is not implied by this launch record.

The exact criterion is frozen in `NUMERICAL-POLICY.md`, with adoption recorded in `policy-decision.json`. R3 remains failed under its original raw-output rule. R4 uses two fresh seeds (7102, 7103), with 12 smoke cells and 36 full qualification cells. Every complex SNR sample must pass the independent double-precision oracle and actual CPU/MKL reference comparisons, including observed route normalization. All existing trigger and veto checks remain unchanged. Both full seed matrices must pass before any of the 54 timing workers starts.

The bank has 1024 distinct templates and is fixed across execution batches 1, 8, 32, 128, 512 and 1024 within each seed. There are three input blocks, FFT length 131072, one affinity-pinned CPU (logical CPU 8), and one numerical-library thread. CUDA also uses one RTX 4090. Timings cover the warm public live-filter API, with three fresh processes per route/batch, two warmups and five measured calls per process. This shared-host experiment excludes frame I/O, PSD estimation, waveform generation, bank loading, executable startup and full-machine capacity.

Source is unchanged at 9578a710479b924e882857c4dffab6ed372a634b. R4 links the clean frozen R3 checkout and verifies all 11 native-library hashes before acquisition and again on successful completion. Local harness validation passed 91 tests before launch. Qualification observation is removed during timing.

- Host: `len`
- Cwd: `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r4`
- Command: `/home/xangma/pycbc-torch-split-20260905/venv/bin/python /home/xangma/pycbc-torch-current-batch-sweep-20260907-r4/batch-campaign.py --shared-host --snr-policy-v2`
- PID/PGID: `3789112/3789112`
- Log: `/home/xangma/pycbc-torch-current-batch-sweep-20260907-r4/launch.log`
- Stop: `ssh len "kill -TERM 3789112"`

Next check: completion of the smoke matrix, then approximately every five minutes during the full matrix. Progress checks are read-only; failures retain their evidence and block timing.
