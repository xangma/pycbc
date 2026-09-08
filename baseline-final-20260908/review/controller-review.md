# Baseline/final controller peer review — 2026-09-08

No qualification launch blocker identified in the bounded local review of `campaign.py`, `config.json`, `setup-remote.py`, and reused `checked-inspiral.py`. Relevant qualifier/comparator helpers were inspected. This is a harness review, not certification of completed runs or scientific parity. The operator reports both builds/preflight passed and `qual-original-cpu` is active; remote state was not independently inspected.

## Verified controls

- Original CPU pins `40e94792b3edf59f39b18b65102b28a4f74433a7`; proposed CPU, Torch CPU and Torch CUDA pin `123e1fb3ef1b338cada636e71c3e9c7987002402`. Setup builds separate source trees using one interpreter. Runtime checks enforce imported PyCBC module origins; source, native artifacts, generated version, harness and scientific inputs are pinned.
- Cross-source comparison independently verifies each revision and executable hash before normalizing only source/executable provenance in a comparison copy. Raw verdicts and substitutions remain recorded. Scientific input hashes, analysis settings, HDF values and numerical budgets remain unchanged.
- Qualifications enforce 384 templates, five segments, 1920 scalar calls, 2097152 FFT samples, 1904 unique seconds, zero gaps/overlaps, and the specified origin/filter/HDF intervals. Proposed CPU versus both Torch arms must pass trigger and conditioning checks; timed outputs must pass against their own qualifications.
- Four fresh-process repeats use balanced rotated positions, common affinity/thread controls and the checked launch-to-exit boundary. Shared-host scope is explicit. Worker process groups support timeout/signal termination. Contaminated parent CPU accounting was removed; per-worker `/usr/bin/time` remains.

All three reviewed scripts parsed successfully; configuration rotation checks passed. Reused qualifier/comparator hashes match the archived versions. No additional material issue was identified in setup, configuration or runtime checker.

## Remaining finding and agreed disposition

**P2 — acquired summary eligibility omits baseline conditioning verdicts.** `main()` records original-versus-proposed conditioning/PSD comparisons but discards their returned verdicts when computing `equal_output_speedup_eligible`. Trigger parity alone could therefore mark eligibility despite a conditioning failure. That field must not independently authorize equivalent-output speedup claims.

The operator will preserve the running, pinned harness and acquired summary unchanged, then produce an independent post-run verifier/summary incorporating all original-versus-proposed trigger and conditioning/PSD verdicts, alongside proposed-backend parity. Missing checks or non-passing verdicts must prevent eligibility. Baseline failures and their timings remain visible as descriptive work completed. This reporting-only computation requires no acquisition rerun when the complete pinned evidence is available; the new verifier and its result remain to be reviewed.

Only this report was written. No campaign/source changes, remote actions, benchmark execution or scientific reruns were performed by this peer review.
