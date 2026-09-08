# Evidence index

The final interpretation is in HANDOFF.md. HANDOFF-DRAFT.md preserves the earlier intermediate interpretation. Runs use shared host len, one assigned CPU core and one native/Torch thread; CUDA additionally uses the GPU. Finite-workload executable measurements are distinct from isolated warm API timing. Qualification runs are excluded from timing summaries.

| Evidence | Location | Purpose |
|---|---|---|
| Frozen CPU accuracy and alternatives | acquired-validation-v5/, acquired-cpu-qualification-v2/ | Unchanged precision gates; preserve failures of faster alternatives |
| Isolated CPU timing | cpu-timing-v5-summary.json, acquired-cpu-timing-v5/ | Three fresh processes per role and repeated actual plan qualification |
| CUDA contracts | acquired-cuda-v3/, gpu-hoist-local-tests.log, local-contracts.log | Scalar scheduling, source/input/output ownership and existing autodiff behavior |
| Completed five-role executable campaign | executable-v3-summary.json, acquired-executable-v3/ | Five qualifications, 15 timings and 34 verified comparisons |
| Superseded executable controllers | acquired-executable-v1/, acquired-executable-v2/ | Preserve both controller assertion failures and terminal audits |
| Warm CUDA sparse-API diagnostic | acquired-warm-chisq-v1/ | Six workers, 48 cells; synchronized input/output/runtime checks |
| Frame-loader source and native tests | loader-v1/ | Separate ecd5d082 commit, source bundle/diff, baseline/candidate XML, 28 read comparisons and eight stream-position checks |
| Frame-loader local controls | frame-loader-local-tests-final.log, frame-loader-local-final.xml, frame-loader-baseline-control.log, frame-loader-baseline-negative-control.json, frame-loader-lint.json | Actual frame data, baseline controls and changed-file lint |
| Frame-loader frozen executable harness | loader-v1/manifest.json, loader-v1/protocol.json, acquired-loader-v1/, loader-qualification-v1.tar | Equal old/new standard CPU, Torch CPU and CUDA; six qualifications, 18 timings and all 44 comparisons passed |
| Loader timing summary | loader-v1-summary.json, loader-v1-metrics.csv, loader-timing-v1.tar | All eight metrics, 144 raw values, 48 medians, 48 observed ranges and five explicit comparisons |
| Independent reviews | peer-reviews/manifest.json and peer-reviews/ | SHA-256-preserved copies of source, harness, negative-control and completed result reviews; original locations recorded in manifest |
| Complete candidate source | final-source.json, final-source.bundle, final-source.diff | Four commits from 7e56ac4241 to ecd5d08231; six Python files, clean worktree and verified bundle prerequisite |

Each completed campaign includes source/helper/input pins, worker receipts and logs, process identity, outputs and a terminal audit binding the final status hash to whole-process-group absence and successful nonblocking lock reacquisition. Earlier unsuccessful candidates and harness versions are retained.

The source checkout is /private/tmp/pycbc-torch-fft-optimization-20260908 on branch codex/torch-fft-optimization-20260908. No integration, push, PR or documentation publication is part of this handoff.
