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

Fresh complete-executable results (pycbc_inspiral)
--------------------------------------------------

Measured on **2026-09-09**, using four fresh unprofiled processes per route
in rotating order after scientific qualification. The workload contains
**384 compressed templates, five segments and 1904 unique H1 seconds**
(731,136 template-seconds; 1920 template/segment pairs).

.. list-table:: Complete-executable wall times (pycbc_inspiral)
   :header-rows: 1
   :widths: 22 16 20 20 22

   * - Route
     - Median (s)
     - Range (s)
     - Original CPU / route
     - Template-seconds / wall second
   * - Original CPU
     - 65.15
     - 65.13–65.28
     - 1.00×
     - 11,222
   * - Candidate CPU
     - 63.81
     - 63.73–63.89
     - 1.02×
     - 11,459
   * - Torch CPU (optimized)
     - 70.69
     - 70.33–70.84
     - 0.92×
     - 10,343
   * - Torch CUDA (optimized)
     - 19.60
     - 19.60–19.70
     - 3.32×
     - 37,308

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

Streaming live-filter results (pycbc_live)
------------------------------------------

The low-latency online search (`pycbc_live`) uses :class:`pycbc.filter.matchedfilter.LiveBatchMatchedFilter`
on shorter frequency-domain blocks ($N = 131,072$ at 2048 Hz). Because individual
short transforms underutilize GPU Streaming Multiprocessors, batching provides
massive concurrency scaling. See :ref:`torch-batch-numerics` for full protocol and
Policy v2 qualification.

.. list-table:: Live-batch throughput scaling (1024 templates, len)
   :header-rows: 1
   :widths: 15 20 25 25 15

   * - Batch Size ($B$)
     - Standard CPU (wf/s)
     - Torch CPU (wf/s)
     - Torch CUDA (wf/s)
     - CUDA vs CPU
   * - 1
     - 1,173
     - 711
     - 2,424
     - 2.07×
   * - 8
     - 1,175
     - 371
     - 13,164
     - 11.20×
   * - 32
     - 1,180
     - 374
     - 30,393
     - 25.75×
   * - 128
     - 1,187
     - 332
     - 32,764
     - 27.60×
   * - 512
     - 1,189
     - 331
     - 33,537
     - 28.21×
   * - 1024
     - 1,188
     - 329
     - 34,260
     - 28.84×

Architectural distinction: inspiral vs live batching
----------------------------------------------------

1. **Transform length & cache dynamics**:
   - In `pycbc_inspiral`, $N = 2,097,152$ samples (512 s at 4096 Hz). A single complex64
     vector is 16.8 MiB, which fits inside the 72 MiB L2 cache of modern GPUs (e.g. RTX 4090)
     and fully saturates the 128 SMs. Batching across segments ($B=5$, 84 MiB) or templates
     ($B=16$, 268 MiB) evicts the L2 cache into GDDR6X DRAM, slowing down each transform
     (0.035 ms/waveform at $B=1$ vs 0.076 ms/waveform at $B=16$).
   - In `pycbc_live`, $N = 131,072$ samples (64 s at 2048 Hz). Each waveform is only 1.0 MiB.
     Batches of $B=32\dots 64$ fit comfortably in L2 cache while filling all GPU execution
     units, yielding a 14× scaling acceleration from $B=1$ to $B=1024$.

2. **Template generation & filtering pipeline**:
   - `pycbc_inspiral` decompresses 384 templates inline on demand to maintain a minimal,
     constant memory footprint ($\approx 150$ MiB), runs symmetric clustering over a $\pm 1$ s
     window, and calculates 16-bin power chi-square vetoes on all surviving triggers.
     IFFT accounts for $<0.5\%$ of the template loop (0.035 ms out of 8.0 ms per call).
   - `pycbc_live` maintains pre-allocated batch workspaces, performs vectorized argmax
     peak finding per template, and computes vetoes only on the loudest trigger per block.

Measured sources: original CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7`` and candidate main
``ca4bc95f7fe54a4d52d3923c3e8b0d4b283ce196``. See the
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
