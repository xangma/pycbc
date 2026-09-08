# Current CPU IFFT stage diagnostic

Measure the accepted source `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f` without editing production code. The exact Torch IFFT engine must select the promoted, in-place 2,097,152-element MKL plan with one private complex128 workspace and disjoint complex64 user buffers.

Three fresh workers run serially on len CPU 8 under the existing campaign lock, with all discovered numerical thread pools checked at one thread before and after. This is a shared-host experiment; CPU 8 has SMT sibling 72. It is not the completely idle-machine or full-machine saturation experiment.

Each worker qualifies the original engine and instrumented reproduction before and after timing: four complete copies of the existing 36-case matrix. L2 and maximum-absolute errors must each be no larger than the legacy FFTW reference, with no tolerance floor, and outputs must be bitwise equal to complex128 MKL followed by complex64 conversion. Public input preservation, buffer identity and target version increments remain checked. No alternative single-precision route is introduced.

Measure the public engine, direct current plan, and diagnostic stages in cyclic block orders. Each path has three blocks of three warmups followed by 15 samples per worker. Stages are promotion, native DFTI call (including pointer lookup), native version/status bookkeeping, and demotion. Record complete calls, stage sums, outside-stage time and the overhead of five consecutive timer reads. Do not subtract timer overhead. These timings prioritize subsequent changes; they do not establish executable speedup or a causal decomposition of the prior full-wall gap.

Source, helpers, native objects and loaded FFT libraries are pinned; the three workers reuse neither interpreter nor descriptors. The controller uses a clean environment and passes the campaign lock to each worker. Any failure stops the campaign. Remote launch follows independent harness review. The prior completed loader evidence package stays fixed.
