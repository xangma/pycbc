# Descriptor reuse qualification v1

Passed native and scientific qualification, independently replayed. This stage was instrumented and provides no executable performance measurement.

The three supported out-of-place single-precision configurations passed nine full-output byte comparisons against original MKL functions using 18 distinct live pointers. Input preservation and coexisting class FFT and promoted Torch MKL plan operation passed.

Both standard and CUDA executions made 1,003 eligible function calls. Cached runs created and freed three descriptors with 1,000 reuse hits; uncached runs created and freed 1,003. Standard also made 133 unsupported calls through the exact original functions.

All four executions passed the unchanged scientific gates. Independent replay checked actual PSD arrays, captured conditioned-strain hashes, workload geometry, six exact comparisons of all 18 H1 science datasets within each scheme, and the cross-scheme default comparator. Native array equality is certified by the pinned native receipt; the arrays themselves were not archived.

All owned descriptors were released, wrappers restored, the whole process group exited and the benchmark lock was independently reacquired. Source, inputs, native libraries, frozen harness and references retained their pins.

Original creator returns count as creation success after commit invocation; the original ignores commit return values. Ambiguous creation failures deliberately fail closed without claiming ownership.

Evidence: `descriptor-reuse-v1-summary.json`, `peer-reviews/descriptor-reuse-results-v1-review.json`, `descriptor-reuse-v1-results.tar`. A separate reviewed experiment must remove diagnostic recording, qualify that helper, and measure standard and CUDA controls under common setup before claiming performance improvement.
