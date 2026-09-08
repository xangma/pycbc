.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Comparing existing PyCBC with Torch
===================================

The review question is how the existing CPU implementation compares with the
proposed Torch implementation on the same scientific workload. A comparison
must identify both source revisions and include the proposed revision's normal
CPU route, so shared changes can be distinguished from the Torch backend.

Comparison status
-----------------

The existing-code baseline used for this work is
``40e94792b3edf59f39b18b65102b28a4f74433a7``. The reviewed Torch runtime is
``9d4e4f6905d9f109d559284326d62173a854264f``. **There is no completed matched
timing campaign comparing those two implementations.** Documentation changes
after that runtime revision do not supply missing timing evidence.

The integrated implementation has :ref:`native regression and executable
correctness evidence <torch-integrated-qualification>`. Those checks compare
against pinned references for each backend and scientific budgets. They do
not establish unchanged output relative to the original baseline. Earlier
original-baseline comparisons reported trigger and numerical differences;
those failures remain available in the `original-reference archive
<https://github.com/xangma/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907>`_.

Consequently, this page makes no measured speedup claim for the proposed
runtime against existing PyCBC. Timing results from development revisions and
individual optimization experiments cannot fill that comparison.

What to compare
---------------

.. list-table:: Required comparison arms
   :header-rows: 1
   :widths: 24 26 24 26

   * - Arm
     - Source
     - Resources
     - Purpose
   * - Existing CPU
     - Unchanged baseline commit
     - One host core and numerical thread; MKL
     - Existing implementation
   * - Proposed CPU
     - Proposed commit, normal CPU scheme
     - Same CPU allocation and MKL
     - Effect of shared changes in the proposal
   * - Proposed Torch CPU
     - Same proposed commit
     - Same CPU allocation and numerical thread
     - Cost of selecting Torch on CPU
   * - Proposed Torch CUDA
     - Same proposed commit
     - One host core/thread plus one named GPU
     - Cost with the additional accelerator

Use clean builds with matching dependency versions, frozen input hashes and
identical scientific arguments. Record every route flag. Default-off graph
capture and other optional routes require separate, explicitly named arms;
setting a flag does not prove that a route executed.

How the executable comparison is measured
-----------------------------------------

The fixed workload is ``pycbc_inspiral`` over 384 distinct compressed low-mass
templates and 1904 unique H1 detector seconds. The complete input and option
definition is in :ref:`torch-reference-campaign`.

#. Qualify completed work, trigger identities and numerical fields for each
   arm before timing. Preserve failures against the unchanged baseline and
   assess shared scientific changes explicitly. Passing proposed-CPU versus
   Torch comparisons alone does not demonstrate baseline equivalence.
#. Run at least three fresh, unprofiled processes per arm on the same host,
   rotating arm order between repeats. Record host load, affinity, SMT siblings,
   thread pools and the GPU model. A shared-host measurement keeps that label.
#. Time process launch through exit after completed HDF output. Include imports,
   setup, frame I/O, filtering and output. Any verification wrapper belongs in
   every comparable full-process clock. Keep qualification and profiling runs
   outside the timing sample set.
#. Report every wall-time sample, the median and observed minimum--maximum.
   For this fixed workload, rate is ``384 * 1904 / wall_seconds``
   template-seconds per wall-second. A timing ratio must name its two arms;
   CUDA uses additional hardware. Observed ranges are not confidence intervals.

The resulting report should contain one table with source revision, backend,
resources, correctness verdict, wall time and rate for each arm, followed by
the exact commands and an immutable raw-evidence link. An unmeasured or failed
arm remains visible. :ref:`torch-benchmark-protocol` specifies repetition,
capacity and profiling controls.

A prepared live-filter API measures a different boundary: it excludes frame
reading, PSD estimation, bank construction and executable startup. Its
:ref:`input and numerical method <torch-batch-numerics>` must be reported
separately from complete-search performance.
