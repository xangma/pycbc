.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Corrected CPU baseline and Torch execution costs
================================================

The current comparison uses standalone corrected CPU
``66789ac4a7468094b0cc3ca1498a1de67e0311f6`` as the reference for the restacked
main Torch proposal ``f582b6fd250d0b82612492979e01e645d5c07afc``. It runs the
proposal through normal CPU, Torch CPU and Torch CUDA separately. Optional
FFT and CPU follow-up PRs are outside this comparison.

**All five trigger comparisons pass, but the full-PSD gate fails for Torch
below 30 Hz. Timings are descriptive execution costs only; full scientific
equivalence and an equivalent-output speedup are not established.** Final
timing results and the immutable campaign archive are pending verification.

Standalone CPU corrections
--------------------------

The corrected reference is a separate prerequisite, proposed in
`CPU draft PR #20 <https://github.com/xangma/pycbc/pull/20>`_. It promotes float32
PSD processing, cumulative template-power accumulation and strain-segment
FFTs, and uses double-precision phase shifts and accumulation in the CPU
point chi-square wrapper. Public single-precision outputs are retained.
These changes have no Torch dependency and do not change native kernels.
The `immutable CPU review
<https://github.com/xangma/pycbc/blob/dcd123cade49b75ce312d0a8342a8c9c5c9b6abd/cpu-review.md>`_
records numerical references, regression tests and intentional output changes.

A separate original-versus-corrected CPU experiment measured median wall time
of 65.280 versus 68.062 seconds, a **4.2611% increase**, on the same finite
workload. It measures the cost of changed arithmetic and changed output:
the original produces 1988 triggers and corrected CPU produces 1991.
This is not an equal-output comparison or a Torch speedup. Its four trials
per source and shared-host limitations are in the `CPU cost report
<https://github.com/xangma/pycbc/blob/dcd123cade49b75ce312d0a8342a8c9c5c9b6abd/cpu-cost-report.md>`_.

Current four-arm comparison
--------------------------

.. list-table:: Four fresh full-process repeats per arm; final timings pending
   :header-rows: 1
   :widths: 19 13 17 14 22 15

   * - Arm
     - Source
     - Allocation
     - Trigger comparison
     - Median seconds (observed range)
     - Template-seconds / wall-second
   * - Corrected CPU
     - ``66789ac4a7``
     - 1 core/thread
     - Reference
     - **PENDING_FINAL_TIMINGS**
     - **PENDING_FINAL_RATE**
   * - Restacked normal CPU
     - ``f582b6fd25``
     - 1 core/thread
     - PASS
     - **PENDING_FINAL_TIMINGS**
     - **PENDING_FINAL_RATE**
   * - Restacked Torch CPU
     - ``f582b6fd25``
     - 1 core/thread
     - PASS
     - **PENDING_FINAL_TIMINGS**
     - **PENDING_FINAL_RATE**
   * - Restacked Torch CUDA
     - ``f582b6fd25``
     - 1 core/thread + RTX 4090
     - PASS
     - **PENDING_FINAL_TIMINGS**
     - **PENDING_FINAL_RATE**

The fixed workload is ``pycbc_inspiral`` over 384 distinct compressed low-mass
templates and 1904 unique H1 seconds, with five segments and 1920
template/segment pairs. Its scientific definition is in
:ref:`torch-reference-campaign`. Rate is ``384 * 1904 / median_wall_seconds``.
The pending values above are not estimates or reused historical samples.
Final verification of all 16 timed outputs against their own qualifications
is **PENDING_FINAL_TIMING_VERIFICATION**. The archive containing every sample,
expanded command, source/native identity and comparison is
**PENDING_IMMUTABLE_CAMPAIGN_LINK**.

Scientific qualification and timing policy
------------------------------------------

The five trigger comparisons are corrected CPU against each of the three
restacked routes, plus restacked normal CPU against Torch CPU and Torch CUDA.
All five pass the frozen identity and trigger-field gates. Conditioned strain
and segment geometry match exactly, as do PSD values on the actual filter
slice, bins ``[15360:1048576]``. Full PSD arrays still fail the frozen budget
for Torch below the 30 Hz cutoff. Exact used bins and passing trigger
comparisons do not turn the full-array failure into a pass.

The initial strict qualification controller stopped on the full-PSD failure
before timing. An explicit policy amendment made after qualification permits
descriptive timings after all five trigger comparisons, exact conditioned
strain and geometry, and exact used PSD bins pass. The original failure,
full-array verdicts and policy amendment are retained; source, inputs,
tolerances, worker commands and rotating order are unchanged. This
continuation does not establish full-equivalence speedup eligibility.

Measurement and reproduction
----------------------------

The four arms use balanced rotating orders ABCD, BCDA, CDAB and DABC, where
A is corrected CPU, B restacked normal CPU, C Torch CPU and D Torch CUDA.
Every sample starts a fresh unprofiled process; qualification runs are
excluded. The clock starts before child launch and ends after exit following
completed HDF output, including imports, setup, frame I/O, filtering, vetoes
and the same runtime/native verification wrapper in every arm. Parent source
checks and post-run HDF comparisons are outside the clock. Filesystem caches
are not flushed.

The shared host is ``len``, AMD Ryzen Threadripper PRO 3995WX. Each arm uses
CPU 8 with SMT sibling 72 and one numerical thread. CUDA additionally uses
RTX 4090 GPU 0. Selectors are ``cpu:1``, ``torch:cpu:1`` and ``torch:cuda:0``,
with explicit MKL and verified numerical and Torch intra/inter-op thread
limits. CUDA graph capture is disabled. The host, sibling and GPU are not
reserved. Host, CPU, process, memory, load and GPU observations accompany
the runs. The pinned sources use verified unchanged native binaries copied
from the frozen build; the archive records source and binary hashes.

Current campaign reproduction instructions are
**PENDING_IMMUTABLE_REPRODUCTION_LINK**. They must include the original strict
stop, the subsequent timing policy, configuration, dependency/build receipts
and all raw worker records. No sample is discarded; reported ranges are
observed minima/maxima, not confidence intervals. This finite workload does
not establish sustained or full-machine capacity;
:ref:`torch-benchmark-protocol` defines those additional controls.

Historical unchanged-baseline comparison
----------------------------------------

The earlier comparison of unchanged PyCBC
``40e94792b3edf59f39b18b65102b28a4f74433a7`` with combined proposal
``123e1fb3ef1b338cada636e71c3e9c7987002402`` remains in the `historical benchmark
archive <https://github.com/xangma/pycbc/tree/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908>`_.
It recorded 1988 versus 1991 triggers, with 1959 shared identities, 29
original-only and 32 proposal-only identities. Its trigger-field failures,
full-PSD failures and descriptive timings remain historical evidence. Those
timings do not measure the current corrected-CPU-versus-Torch comparison.
The separate CPU review above explains the arithmetic and output changes.

Existing :ref:`native regression evidence <torch-integrated-qualification>`
also retains its own source pins and scope. A prepared live-filter API
excludes frame reading, PSD estimation, bank construction and executable
startup. Its :ref:`input and numerical method <torch-batch-numerics>` measures
a different boundary and must be reported separately from complete-search
performance.
