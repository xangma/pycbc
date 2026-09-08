.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Measuring Torch performance
===========================

Compare backends using the same inputs, scientific settings and completed
work. Record the exact source revisions and test numerical agreement before
collecting timings. Device selection alone does not establish a speedup:
setup, transfers and host work can dominate small workloads.

Fresh complete-executable results
---------------------------------

Measured on **2026-09-08**, using four fresh unprofiled processes per route
in rotating order after scientific qualification. The workload contains
**384 compressed templates, five segments and 1904 unique H1 seconds**
(731,136 template-seconds; 1920 template/segment pairs).

.. list-table:: Complete-executable wall times
   :header-rows: 1
   :widths: 22 16 20 20 22

   * - Route
     - Median (s)
     - Range (s)
     - Original CPU / route
     - Template-seconds / wall second
   * - Original CPU
     - 65.52
     - 65.37–66.28
     - 1.00×
     - 11,159
   * - Candidate CPU
     - 64.91
     - 64.74–65.02
     - 1.01×
     - 11,265
   * - Torch CPU
     - 103.52
     - 103.48–103.86
     - 0.63×
     - 7,063
   * - Torch CUDA
     - 31.97
     - 31.91–32.02
     - 2.05×
     - 22,871

Each process was pinned to logical CPU 8 of an AMD Ryzen Threadripper PRO
3995WX, with numerical thread pools fixed to one. The Torch routes also set
intra/inter-op counts to one. CUDA used an NVIDIA GeForce RTX 4090 with
graphs disabled. The host and
GPU were shared and unreserved. Ranges show the four observed samples; these
results do not establish sustained or full-machine capacity.

The timing boundary runs from checked-process launch through exit, including
startup, input preparation, filtering, vetoes, HDF output and runtime
verification. Qualification instrumentation and parent-side comparisons are
outside this boundary. The rate divides 731,136 template-seconds by the
median wall time; the ratio divides the original CPU median by each route's
median.

All four routes produced **1988 triggers** and passed all five frozen
cross-route trigger and complete-PSD comparisons. Original and candidate
CPU scientific data and PSDs were byte-identical. Every timed output also
passed comparison with its route's fresh qualification.

Measured sources: original CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7`` and candidate main
``eb8fef9ed1d06378b59cae8439fd40af63827575``. The optional FFT and
native CPU optimization branches are outside this measurement. See the
`immutable benchmark evidence <https://github.com/xangma/pycbc/tree/134ecb2586b2e2fc6272924e9cffdf15e51e6f39/torch-fresh-benchmark>`_
for every sample, command, input hash, environment record and independent
verification, and :ref:`torch-reference-campaign` for the workload and gates.

Choose the measurement boundary
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Measurement
     - Included work
     - Definition
   * - Complete executable
     - Process startup, frame I/O, conditioning, PSD estimation, template
       preparation, filtering, vetoes and completed HDF output.
     - :ref:`torch-reference-campaign`
   * - Prepared live-filter API
     - Calls to ``LiveBatchMatchedFilter.process_data`` with prepared
       frequency-domain inputs, including filtering, peak selection and vetoes.
     - :ref:`torch-batch-numerics`
   * - Individual operation
     - A declared kernel or public API call, with allocation, transfer,
       compilation and synchronization costs identified explicitly.
     - :ref:`torch-benchmark-protocol`

Report these boundaries separately. A prepared API rate does not include the
startup and preparation costs of an executable. An individual operation's
speedup does not establish the speedup of a complete search.

Compare equivalent work
------------------------

Use the unchanged CPU reference to check preservation of existing behavior,
and candidate normal CPU as a dispatch control for Torch CPU and CUDA.
Require all frozen scientific gates to pass before equivalent-output timing.
The executable workload and its numerical tolerances are defined in
:ref:`torch-reference-campaign`; the live-filter fixture has a separate
:ref:`numerical method <torch-batch-numerics>`.

Run fresh unprofiled processes in rotating backend order. Separate cold calls,
warm calls and profiled runs; synchronize accelerators around timed regions.
Include transfers and host work inside the boundary being measured. Record
thread limits, allocated cores and GPUs, memory use and host load. Follow
:ref:`torch-benchmark-protocol` for convergence and full-machine experiments.

Report results with their evidence
----------------------------------

For each source and backend, retain input and output hashes, full commands,
environment and native-build identities, test results, scientific comparisons
and every timing sample. Publish medians and observed ranges with the
completed-work denominator and resource allocation. Keep raw results and
profiles in an immutable evidence archive associated with the measured
revision.

Qualification runs establish correctness within their tested scope; they are
not performance samples. Finite-workload measurements do not establish
sustained or full-machine capacity. Run the relevant checks in
:ref:`torch-testing` before benchmarking a changed API.
