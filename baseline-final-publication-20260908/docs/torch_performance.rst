.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Comparing existing PyCBC with Torch
===================================

The 8 September 2026 benchmark compares unchanged PyCBC
``40e94792b3edf59f39b18b65102b28a4f74433a7`` with the final main Torch proposal
``123e1fb3ef1b338cada636e71c3e9c7987002402``, using separate clean native builds,
matching dependencies and the same scientific workload. The proposed normal
CPU route separates shared changes from the cost of selecting Torch. Optional
FFT and CPU follow-up PRs are outside this comparison.

**The timing comparison is complete; scientific equivalence to unchanged
PyCBC fails.** The table reports descriptive execution costs, with every failed
scientific comparison retained. It does not establish an equivalent-output
speedup over the baseline.

Measured comparison
-------------------

.. list-table:: Four fresh full-process repeats per arm
   :header-rows: 1
   :widths: 18 13 18 17 21 13

   * - Arm
     - Source
     - Allocation
     - Triggers / baseline verdict
     - Median seconds (observed range)
     - Template-seconds / wall-second
   * - Existing CPU
     - ``40e94792b3``
     - 1 core/thread
     - 1988 / reference
     - 67.833 (66.377--76.056)
     - 10,778.5
   * - Proposed CPU
     - ``123e1fb3ef``
     - 1 core/thread
     - 1991 / FAIL
     - 66.626 (65.531--70.279)
     - 10,973.7
   * - Proposed Torch CPU
     - ``123e1fb3ef``
     - 1 core/thread
     - 1991 / FAIL
     - 106.597 (102.795--112.295)
     - 6,858.9
   * - Proposed Torch CUDA
     - ``123e1fb3ef``
     - 1 core/thread + RTX 4090
     - 1991 / FAIL
     - 20.926 (20.351--26.063)
     - 34,939.6

The fixed workload is ``pycbc_inspiral`` over 384 distinct compressed low-mass
templates and 1904 unique H1 seconds, with five segments and 1920
template/segment pairs. Its complete scientific definition is in
:ref:`torch-reference-campaign`. Rate is ``384 * 1904 / median_wall_seconds``.
All 16 timed outputs pass against their own separate qualification. Every
sample, exact command, build identity and comparison is in the
`immutable benchmark archive <https://github.com/xangma/pycbc/tree/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908>`_.

Scientific comparison
---------------------

The baseline produces 1988 triggers; each proposed route produces 1991. There
are 1959 matched identities, 29 baseline-only and 32 proposal-only identities.
Each baseline/proposal comparison has 1951 chi-square, 22 phase and 10 SNR
budget violations among matched triggers. These failures require scientific
review before this proposal can be described as preserving baseline output.

The three proposed routes agree on all 1991 trigger identities and pass the
frozen trigger-field budgets. Conditioned strain digests, gating and geometry
match across all four arms. Proposed PSDs match exactly on the actual filter
slice, bins ``[15360:1048576]``. Full PSD arrays differ below the 30 Hz cutoff:
2375 finite-bin budget violations for Torch CPU and 3105 for CUDA relative to
proposed CPU. The full-PSD check remains failed.

The initial controller stopped on that full-PSD failure before timing. A
separately reviewed continuation collected descriptive timings after checking
proposed trigger parity and exact strain, geometry and used PSD bins. The
archive preserves this decision made after qualification, the original stop,
all full-array failures and unchanged tolerances. Its independent verifier
includes full PSD and conditioning checks and rejects equivalent-output
speedup eligibility. Existing :ref:`native regression evidence
<torch-integrated-qualification>` remains separate from baseline equivalence.

Measurement and reproduction
----------------------------

The four arms ran in balanced rotating orders ABCD, BCDA, CDAB and DABC, where
A is existing CPU, B proposed CPU, C Torch CPU and D Torch CUDA. Every sample
starts a fresh unprofiled process. Qualification runs are excluded. The clock
starts before child launch and stops after exit following completed HDF output,
including imports, setup, frame I/O, filtering, vetoes and the same runtime/hash
verification wrapper in every arm. Parent source checks and post-run HDF
comparisons are outside that clock. Filesystem caches were not flushed.

The shared host is ``len``, AMD Ryzen Threadripper PRO 3995WX. Each arm uses CPU
8 with SMT sibling 72 and one numerical thread. CUDA additionally uses RTX 4090
GPU 0. Selectors are ``cpu:1``, ``torch:cpu:1`` and ``torch:cuda:0``, with explicit
MKL. All observed numerical and Torch intra/inter-op pools are single-threaded;
CUDA graph capture is disabled. The host, sibling and GPU are not reserved.
CPU, process, memory, load and GPU observations accompany every run.

The archive's `reproduction instructions
<https://github.com/xangma/pycbc/blob/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908/REPRODUCE.md>`_ and expanded worker receipts
define the exact builds, commands and frozen inputs. No sample is discarded;
the ranges are observed minima/maxima, not confidence intervals. This finite
workload does not establish sustained or full-machine capacity;
:ref:`torch-benchmark-protocol` defines those additional controls.

A prepared live-filter API excludes frame reading, PSD estimation, bank
construction and executable startup. Its :ref:`input and numerical method
<torch-batch-numerics>` measures a different boundary and must be reported
separately from complete-search performance.
