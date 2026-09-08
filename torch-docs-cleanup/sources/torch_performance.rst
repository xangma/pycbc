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
