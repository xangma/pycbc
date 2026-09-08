# Current CPU stage review

The unchanged qualified 2M promoted in-place IFFT takes **33.409 ms** through the engine, **33.379 ms** through the direct plan, and **33.373 ms** through the instrumented stage path. These are medians of three worker medians, with 45 samples per path per worker. The ranges of worker medians overlap, so they do not support a further wrapper speedup claim.

| Measured interval | Median (ms) | Range of worker medians (ms) |
|---|---:|---:|
| Promotion | 2.493279 | 2.475809–2.497214 |
| Native DFTI call boundary | 28.861175 | 28.854667–28.958477 |
| Version/status bookkeeping | 0.010670 | 0.010059–0.011051 |
| Demotion | 1.997084 | 1.972843–2.001614 |
| Paired promotion plus demotion | 4.492219 | 4.450545–4.497243 |
| Stage sum | 33.366973 | 33.334503–33.462252 |
| Outside stage intervals | 0.005511 | 0.005500–0.005602 |
| Five-clock control | 0.000461 | 0.000461–0.000461 |

The paired conversion interval is **13.4459%** of the staged whole call, using within-sample ratios followed by worker medians. Its roughly 4.49 ms cost supports a bounded conversion experiment. The native promoted transform dominates the measured route; its full time cannot be labeled an unavoidable precision penalty without a suitable causal comparison. Timers are included and no overhead is subtracted. These microbenchmarks do not establish full-executable savings or causally divide the 37.467 s executable gap. Medians of separate intervals need not sum.

Independent checks passed for all 432 fixed precision cases, including unchanged L2/max-absolute budgets and bitwise complex128 MKL parity. All twelve 36-case result lists are identical. The review reconstructed 405 timed whole calls, 135 stage observations and 3,000 timer controls and matched every owner raw sample, median and range. All 320 recorded Python files matched the clean `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f` source, and all eleven native pins matched the prior qualified runtime. The 21 archived files are byte-equal to the acquisition. The terminal audit binds the launch/status and records whole-process-group absence plus nonblocking lock reacquisition.

The fixed micro helper observes CPU core 8, Torch intra-op 1 and native pools 1. Torch inter-op remains at its recorded default of 64 before and after; this does not mean 64 active threads. The executable explicitly uses inter-op 1, so these controls must not be described as identical. This synchronous warmed comparison remains micro-attribution evidence.

A later NumPy conversion experiment must test storage resizability before any `tensor.numpy()` export: that export changes PyTorch storage semantics and the existing numerical qualifier already exports views. Retained raw-pointer views would need explicit ownership, lifetime, pointer/size invalidation, resize and version checks. No conversion candidate is approved by this result alone.

Executable review: [review-residual-cpu-results-v1.py](review-residual-cpu-results-v1.py). Result: [residual-cpu-stage-results-v1-review.json](residual-cpu-stage-results-v1-review.json), SHA-256 `8d297ec3d9436c14ed033bc8dc23e5ceb1e5614f9a8892ddf1d9ecea7cb3f00e`. The source package and completed loader benchmark remain unchanged.
