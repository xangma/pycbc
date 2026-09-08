.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:

Torch benchmarks
================

These are two different tests: a complete gravitational-wave search executable
and a prepared live-filter API. Each result below identifies its measured
revision and timing boundary. The latest completed result for each test is
shown; revisions differ, so their rates and speedups cannot be combined.
All measurements use the shared, unreserved host ``len``.

.. list-table:: What each benchmark tests
   :header-rows: 1
   :widths: 18 30 28 24

   * - Test
     - Fixed work
     - Timed region
     - Question answered
   * - Full executable
     - 384 distinct compressed IMRPhenomD templates; 1904 unique H1 detector
       seconds; five segments per template
     - Fresh process through completed HDF output, including verification,
       imports, setup, filtering and output
     - How long does the complete search take on each backend?
   * - Warm live-filter API
     - 1024 templates, three synthetic strain blocks and FFT length 131072;
       execution batches 1, 8, 32, 128, 512 and 1024
     - Prepared ``LiveBatchMatchedFilter.process_data`` calls, after warmup;
       CUDA synchronized at timing boundaries
     - How does execution batch size affect filtering rate on each backend?

Neither test measures reserved-host, full-machine or sustained search capacity.
One process uses one host core and one numerical-library thread. CUDA also uses
one RTX 4090. CPU affinity does not reserve its SMT sibling.

.. _torch-benchmark-status:

Full executable: latest backend comparison
------------------------------------------

All three backends run the same frozen revision,
``ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f``, with the bounded frame loader
applied equally to every backend. This source also includes the qualified CPU
promoted-IFFT workspace change and CUDA scalar scheduling. Each backend has
three fresh unprofiled workers, drawn from the same six-role loader campaign.
Role order is forward/reverse/forward across repeats. The throughput numerator
is ``384 * 1904 = 731136`` template-seconds.

.. list-table:: Full-process wall time and rate for the fixed workload
   :header-rows: 1

   * - Backend
     - Median seconds
     - Observed minimum--maximum seconds
     - Template-seconds / wall-second
   * - Standard CPU / MKL
     - 66.933
     - 66.629--67.997
     - 10923.33
   * - Torch CPU
     - 104.400
     - 104.212--104.995
     - 7003.21
   * - Torch CUDA + one GPU
     - 21.766
     - 21.727--22.124
     - 33591.15

.. figure:: images/torch-executable-20260908/executable-wall.png
   :alt: Full executable wall time on standard CPU, Torch CPU and Torch CUDA, showing three fresh processes per backend, medians and observed ranges.
   :width: 100%

   Fixed inputs and scientific options; the full clock includes runtime
   verification, imports, setup, filtering and HDF output. Ranges are
   observations, not confidence intervals or an old/new speedup measurement.

The full loader campaign has six separate qualifications and eighteen timing
workers: baseline and candidate for each backend. All **44 scientific
comparisons pass**, preserving 1991 trigger identities and the frozen field
budgets. The qualifications check the compressed bank, all 1920 template/segment
pairs and valid-time coverage. This figure shows only the nine candidate
workers; qualification and comparison work is outside their clocks.

This rate is per assigned host core for the finite workload, including startup.
It differs from the HDF ``templates_per_core`` statistic, which uses PyCBC's
shorter internal timer. Torch CPU takes 1.560 times the standard CPU wall time.
Its qualified FFT route promotes complex64 input to complex128, performs the
transform and converts back. Native FFT execution and conversion dominate its
measured IFFT; they do not causally account for every second of the executable
gap. See :ref:`torch-cpu-precision-cost` for the precision and timer distinctions.

CUDA uses a GPU in addition to the host core. These results characterize the
complete executable's scalar filter path; they do not exercise the live-batch
API or measure equal hardware cost.
See :ref:`torch-reference-campaign` for inputs, accuracy gates and reproduction,
and :ref:`torch-profile-attribution` for the earlier full-workload profiles.

.. _torch-executable-followups:

Separate executable follow-ups
------------------------------

Descriptor reuse (R) and offline CUDA graphs (G) were measured as Python
prototypes on the same source. Their own controlled campaigns produced the
following results. **These are not timings of a later integrated revision.**
R contains standard CPU and CUDA roles; it has no Torch CPU measurement.
G compares CUDA eager execution with graph execution, with descriptor reuse
enabled in both arms.

.. figure:: images/torch-executable-20260908/prototype-followups.png
   :alt: Separate paired descriptor-reuse comparisons on standard CPU and CUDA, and a CUDA graph comparison with descriptor reuse in both arms. Four worker pairs per comparison.
   :width: 100%

   Full fresh-process wall times, including setup, cleanup and graph capture
   where enabled. Lines connect workers from the same repeat; dark marks show
   medians. Each panel expands its own time axis. R and G are separate
   campaigns, so their percentage reductions must not be added.

:ref:`torch-followup-evidence` records the exact baselines, samples, numerical
checks, limitations and plot reproduction. The live-filter API below remains
a separate workload and revision.

.. _torch-batch-throughput:

Warm live-filter API: latest completed batch sweep
--------------------------------------------------

This sweep measures revision ``9578a710479b924e882857c4dffab6ed372a634b``,
**before the squared-norm optimization**. The fixed 1024-template bank is
processed across all three blocks at every batch size. One iteration therefore
performs ``1024 * 3 = 3072`` template-block evaluations. Changing the execution
batch changes grouping, not total work.

Each backend/batch has three fresh workers. Each worker performs one cold
iteration, two warmups and five timed iterations. The reported rate is the
median of the three worker median rates; whiskers span those worker medians.
All workers run serially, with route and batch order rotated between repeats.
Frame I/O, PSD estimation, waveform/bank preparation, executable startup and
trigger validation are outside this API clock.

.. list-table:: Median template-block evaluations per second; higher is faster
   :header-rows: 1

   * - Execution batch
     - Standard CPU / MKL
     - Torch CPU
     - Torch CUDA + one GPU
   * - 1
     - 1173.4
     - 710.7
     - 2424.4
   * - 8
     - 1175.4
     - 371.3
     - 13163.7
   * - 32
     - 1180.3
     - 373.5
     - 30392.6
   * - 128
     - 1186.7
     - 331.6
     - 32764.3
   * - 512
     - 1189.3
     - 330.6
     - 33537.1
   * - 1024
     - 1187.6
     - 329.1
     - 34259.8

.. figure:: images/torch-batch-r4-20260907/batch-throughput.png
   :alt: Warm live-filter rate at six execution batch sizes for standard CPU, Torch CPU and Torch CUDA; three worker medians per cell.
   :width: 100%

   One host core and numerical thread; CUDA additionally uses one GPU.
   Whiskers show observed ranges, not confidence intervals.

All 12 smoke and 36 full numerical qualifications across seeds 7102 and 7103
passed before the 54 timing workers started. Every timing worker also passed
its trigger checks. The largest normalized complex-SNR error is ``3.3868e-6``
against the independent oracle and ``3.9178e-6`` against MKL, below the adopted
absolute ``0.001`` budget. Exact trigger gates and separate veto tolerances
also apply; :ref:`torch-batch-numerics` defines them and the synthetic inputs.

:download:`Throughput SVG <images/torch-batch-r4-20260907/batch-throughput.svg>`;
:download:`input and image manifest <images/torch-batch-r4-20260907/manifest.json>`.

Methods and evidence
--------------------

* :ref:`torch-reference-campaign`: executable inputs, scientific options and
  reproduction.
* :ref:`torch-batch-numerics`: live-filter inputs, qualification and plot
  reproduction.
* :ref:`torch-profile-attribution`: optimization evidence and profiling scopes.
* :ref:`torch-followup-evidence`: latest source changes, prototype comparisons,
  CPU precision investigation and current plot reproduction.
* :ref:`torch-benchmark-protocol`: controls for future capacity and scaling tests.

Superseded plots and campaign tables are removed from the active documentation.
Raw successful and failed results remain in the immutable archives linked by
these methods pages. No original-upstream equivalence is claimed: shared
scientific corrections change its output, as retained in the
`original-reference evidence
<https://github.com/xangma/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907>`_.
