# GPU setup descriptor diagnostic review

All three runs passed the independent receipt, source/native, runtime, science and terminal checks. Each captured 1,003 function-API calls. The 41-file result archive was byte verified; actual PSD files and all 18 H1 science datasets matched the qualified CUDA reference exactly. Controller PGID 861805 was absent and the owner independently reacquired the campaign lock. Source remains unchanged ecd5.

| Configuration | Calls/run | First creation median | Repeated creation median | Native total median |
|---|---:|---:|---:|---:|
| 16,384 real-to-complex, Welch | 999 | 0.002694 s | 0.503333 s (998 calls) | 0.039970 s |
| 16,777,216 complex-to-real | 2 | 0.508386 s | 0.507322 s | 0.214084 s |
| 16,777,216 real-to-complex | 2 | 0.511495 s | 0.510717 s | 0.201144 s |

Per-run sums give median total creation **2.544452 s** (range 2.534743–2.555508), repeated creation **1.522122 s** (1.516596–1.522319), and native execution **0.455636 s** (0.447844–0.464520). These use sums within each run before taking medians.

The 999 small transforms come from Welch. The first large inverse and forward calls come from inverse-spectrum truncation. Their repeats come from `FrequencySeries.to_timeseries` and `TimeSeries.to_frequencyseries`, respectively. The large repeats therefore span distinct callers; their cost must not be assigned to Welch.

A bounded descriptor-reuse diagnostic is the next GPU setup opportunity: preserve the original function-API configuration and conservative direction/size/dtype/placement/thread key, with process/thread ownership, lifetime and error cleanup verified before unchanged executable science qualification. The class API uses a different conjugate-even storage setting, so replacing function calls with class plans is not established equivalent by this evidence.

The first large creations remain distinct costs. All durations are instrumented, MKL was imported early, and repeated creation totals identify a target rather than achievable savings. No production change or unprofiled executable improvement is certified.

Evidence: `residual-gpu-results-v2-review.json` SHA256 1ec6e525bd821baff0040945ae36522fba7374bfc32135d7b50e4b1347eb78fb; `GPU-DESCRIPTOR-REVIEW-v2.json` SHA256 41e2b47181a56c76a498b2b625d7f3638dc9039890044024f28630a1f43d0e1f. Full replay script: `review-residual-gpu-results-v2.py`.
