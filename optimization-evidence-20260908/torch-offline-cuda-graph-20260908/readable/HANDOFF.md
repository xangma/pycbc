# Offline CUDA graph experiment handoff

The v2 fixed experiment passed its frozen acceptance rule: complete-child
median wall time fell from 20.257390172453597 s to 19.94459447549889 s
(1.5441065916776076%). All four paired eager-minus-graph differences are
positive. Every prescribed sample is retained.

Read `OFFLINE-CUDA-GRAPH-v2.md` for the scoped result and limitations;
`offline-cuda-graph-v2-samples.csv` contains the eight measurements.
Both arms use the same accepted descriptor reuse helper. These are shared-host
GPU measurements for 384 templates and 1,904 valid detector seconds, including
startup, capture, synchronization, cleanup and interpreter exit. CPU parity
remains unresolved. No source integration or publication was performed.

Evidence:

- `diagnostic-v2/`: 37 frozen inputs, manifest SHA256
  `38ca072d1a59c174f029a8f8e13bd43201bff0b2e2d824b967a20f39f1c2d5da`.
- `acquired-diagnostic-v2/`: all 167 archived files, including the result
  manifest; all 166 manifest entries verified during acquisition and audit.
- `diagnostic-v2-results.tar`: 23,459,840 bytes, SHA256
  `6149853dae5d91b48a5aaf61fea5e0621e69791cd2cf23bb020d1e83f316bf1b`.
- Result manifest SHA256:
  `0aff4d8412d81d0448f70e28fbc8b283584670a149989fc49e4a3d040b6953a4`.
- `audit-v2.py` and `audit-v2.json`: owner recomputation of all 25 scientific
  comparisons, 15 exact comparisons across 18 H1 datasets, qualification
  certificates, complete-child measurements and the acceptance decision.
- `peer-results-review-v2.json`: completed independent read-only evidence and
  report review, with no blocking finding. It verifies recorded native and
  closure evidence; it is not another native run or independent remote probe.
- `SHA256SUMS`: the outer file inventory and SHA256 seal, excluding Python
  bytecode caches and the checksum file itself.
- `diagnostic-v1/`, `acquired-diagnostic-v1/` and v1 archives preserve the
  failed qualifier attempt. Version 1 ran no performance samples.

Fresh v2 native and executable gates passed: six native captures, 30 native
replays, 48 invalid-state rejections before replay, 1,920 normal G replays
against fresh eager oracles, 1,309 verified consumer index transitions and
14,125 retained-output checks. All scientific outputs retained 1,991 triggers.
The local CPU harness passed 15 focused tests. Source remained clean at
`ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`.

The remote campaign is complete. Host `len`, cwd
`/home/xangma/pycbc-torch-offline-cuda-graph-20260908/diagnostic-v2`,
controller PID/PGID 2469885, start ticks 64716058. The controller was invoked
with `/home/xangma/pycbc-torch-split-20260905/venv/bin/python` and
`graph_controller.py`; its log is `controller.log` in that cwd. All 11 workers
exited zero. The terminal audit verified whole-process-group absence,
independent nonblocking lock reacquisition and unchanged final pins. No
stop command or further timing is needed for this completed campaign.

The detailed archive is immutable. Any subsequent diagnostic must use a new
directory and preserve these inputs, outputs, sample order and decision.
