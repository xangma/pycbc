.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Corrected CPU baseline and Torch execution costs
================================================

The current comparison uses standalone corrected CPU
``66789ac4a7468094b0cc3ca1498a1de67e0311f6`` as the reference for measured
restacked main ``f582b6fd250d0b82612492979e01e645d5c07afc``. It runs that
source through normal CPU, Torch CPU and Torch CUDA separately. Optional
FFT and CPU follow-up PRs are outside this comparison.

**All five trigger comparisons pass, but the full-PSD gate fails for Torch
below 30 Hz. Timings are descriptive execution costs only; full scientific
equivalence across all four arms and an equivalent-output Torch speedup
are not established.** Independent evidence verification completed with
limitations; standalone versus restacked normal CPU passes all frozen
scientific gates.

Measured and publication source
-------------------------------

The measured source remains frozen at ``f582b6fd25``. Formatting-only main
``6b47580146e73169cd130b601731e5ba40668d93`` incorporates the supplied Qlty
repair in filtering PR #8 and replays its descendants. Compared with the
measured main, only ``pycbc/filter/matchedfilter.py`` and
``pycbc/vetoes/chisq.py`` differ: both complete module ASTs are identical
after excluding source locations. Their Python source bytes do differ.
Every other tracked file is byte-identical, including ``bin/``, ``test/``,
``tools/``, ``.github/`` and all native sources.

The documentation commit is based on that formatted main and changes only
documentation. This is an explicit mapping from measured source through
AST-equivalent formatting to the documentation-bearing publication, not a
new benchmark run of the publication head. All timings and worker receipts
retain the measured ``f582b6fd25`` pin. The archived ``sources.bundle`` and
frozen bank preserve the acquired source and bank after published refs advance.

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
---------------------------

.. list-table:: Four fresh full-process repeats per arm; observed execution costs
   :header-rows: 1
   :widths: 19 13 17 14 22 15

   * - Arm
     - Measured source
     - Allocation
     - Trigger comparison
     - Median seconds (observed range)
     - Template-seconds / wall-second
   * - Corrected CPU
     - ``66789ac4a7``
     - 1 core/thread
     - Reference
     - 68.238 (68.174--68.487)
     - 10,714.4
   * - Restacked normal CPU
     - ``f582b6fd25``
     - 1 core/thread
     - PASS
     - 65.472 (65.370--65.672)
     - 11,167.1
   * - Restacked Torch CPU
     - ``f582b6fd25``
     - 1 core/thread
     - PASS
     - 102.446 (102.207--102.728)
     - 7,136.8
   * - Restacked Torch CUDA
     - ``f582b6fd25``
     - 1 core/thread + RTX 4090
     - PASS
     - 20.324 (20.298--20.353)
     - 35,973.9

The fixed workload is ``pycbc_inspiral`` over 384 distinct compressed low-mass
templates and 1904 unique H1 seconds, with five segments and 1920
template/segment pairs. Its scientific definition is in
:ref:`torch-reference-campaign`. Rate is ``384 * 1904 / median_wall_seconds``.
Each route produces 1991 triggers. All four qualification processes and
16 timed processes completed; every sample is retained. Independent checks
of timed outputs against their own qualifications are **PASS for all 16
outputs**. Every sample, expanded command, source/native identity and
comparison is retained in the
`immutable corrected-baseline archive
<https://github.com/xangma/pycbc/tree/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign>`_.

Scientific qualification and timing policy
------------------------------------------

The five trigger comparisons are corrected CPU against each of the three
restacked routes, plus restacked normal CPU against Torch CPU and Torch CUDA.
All five pass the frozen identity and trigger-field gates. Conditioned-strain
digests and segment geometry match exactly, as do PSD values on the actual
filter slice, bins ``[15360:1048576]``. Standalone and restacked normal CPU
pass the full-PSD gate. Against either normal CPU reference, Torch CPU has
2375 finite-bin budget violations and Torch CUDA has 3105 in every segment,
all below the 30 Hz cutoff. The frozen full-PSD budget is ``1e-4`` relative
tolerance with zero absolute floor. Exact used bins and passing trigger
comparisons do not turn the full-array failure into a pass.

The initial strict qualification controller stopped on the full-PSD failure
before timing. An explicit policy amendment made after qualification permits
descriptive timings after all five trigger comparisons, matching
conditioned-strain digests and geometry, and exact used PSD bins pass. The
original failure, full-array verdicts and policy amendment are retained;
source, inputs,
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

Both sources use Python 3.11.9. The inherited distribution inventory lists
Torch 2.1.1, while every Torch worker reports imported ``2.13.0+cu130``. A
post-acquisition diagnostic records both installations and selected file
hashes; it explains metadata selection without retrospectively identifying
every Torch binary loaded by the workers. The independent verifier uses
acquisition receipts for native/input identity and conditioned-strain
equality; tracked source bytes were independently checked against Git.

The `immutable reproduction instructions
<https://github.com/xangma/pycbc/blob/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign/REPRODUCE.md>`_
include the original strict
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
