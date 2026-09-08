# Residual performance analysis after bounded diagnostics

The latest completed executable campaign still has a 37.466764 s Torch CPU
wall-time gap against the updated standard. The three matched-repeat filtering
gaps are 37.304, 37.503 and 37.995 s; setup and outside-internal time are slightly
shorter for Torch CPU in all three repeats. The residual is concentrated in filtering.

| Completed executable role | Median wall (s) | Range (s), three fresh runs |
| --- | ---: | ---: |
| standard-candidate | 66.933429 | 66.629332–67.996605 |
| cpu-candidate | 104.400193 | 104.212417–104.995407 |
| cuda-candidate | 21.765736 | 21.727497–22.123757 |

The current qualified 2**21 promoted in-place CPU plan takes 33.409 ms per
engine call in the stage experiment. The native transform stage is 28.861 ms;
paired promotion plus demotion takes 4.492 ms, or 13.446% of a staged call.
Bookkeeping is about 0.011 ms. These measurements identify where time is spent;
they do not prove any cost unavoidable or partition the executable gap.

The guarded NumPy conversion candidate passed all 14 fresh pre-export native
contracts and 432 strict precision cases. Across three workers its whole-call
medians were 0.1164%, 0.1816% and 0.1098% slower. Reject it and retain the current
Tensor.copy_ path. No further executable trial is warranted from this candidate.
CPU microbenchmarks observed inter-op64 while executable trials enforce1.

The next bounded candidate is original-function MKL descriptor reuse during
setup. The completed GPU diagnostic records 1003 calls and 2.544 s total creation,
of which 1.522 s is repeated creation within three distinct configurations.
Welch contributes 998 repeats and 0.503 s; one large inverse and one large forward
repeat contribute about 1.018 s. Those large repeats occur in other callers.
Two first large creations take about 0.51 s each and are not demonstrated reusable.

The owner is preparing an isolated diagnostic. Review must preserve successful
original operation order and settings, native identity/castability and baseline
signature transitions, current-pointer execution, and ownership/error cleanup.
Any fail-closed creation guard is a documented diagnostic error-path divergence.
Then require real descriptor reuse and fresh same-scheme standard/CUDA science
qualification before balanced unprofiled timing for both roles. A common setup
improvement must be evaluated against the contemporaneously updated standard.
The 1.522 s instrumentation result is not a predicted speedup.

No further CPU precision matrix is justified by this result. Large first-creation
cost is a lower-priority investigation after repeat reuse, with no implementation
or savings claim. There is no new executable performance result, production edit,
integration, or publication. JSON companion pins all completed source reports
and retains exact values, ranges, constraints and candidate dispositions.
