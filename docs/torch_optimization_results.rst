.. _torch-optimization-results:

Torch performance evidence status
=================================

The repository retains one historical len campaign with raw samples and runner
records, described below. It qualifies only its exact tested revision and
workload. Older PNG files remain omitted because their raw measurements and
provenance were unavailable. No performance plots are currently published.

An image without those sources is not evidence for a speedup, regression,
crossover point, support guarantee, or optimization promotion. The distinction
between substantiated measurements, derived plots, reference snapshots, and
required evidence is defined in :ref:`torch-performance`.

Historical len campaign: 2026-09-04
-----------------------------------

Tested source: ``ec12863ee897b529960d8579526715d8d54810e9``. The later commit
publishing this receipt is **not** validated by these measurements. The
checked-in bundle contains:

* :download:`Receipt and limitations <_static/torch-evidence/2026-09-04/README.md>`
  and :download:`detailed review <_static/torch-evidence/2026-09-04/review.md>`;
* :download:`CPU/CUDA raw samples, one thread <_static/torch-evidence/2026-09-04/live-batch-cpu-cuda-t1.json>`
  and :download:`CPU raw samples, four threads <_static/torch-evidence/2026-09-04/live-batch-cpu-t4.json>`;
* :download:`original SHA-256 checksums <_static/torch-evidence/2026-09-04/SHA256SUMS>`,
  :download:`environment and native-build hashes <_static/torch-evidence/2026-09-04/environment-before.txt>`,
  :download:`final environment <_static/torch-evidence/2026-09-04/environment-after.txt>`,
  and :download:`historical launcher <_static/torch-evidence/2026-09-04/validate-performance.sh>`.

The workload exercised public ``LiveBatchMatchedFilter.process_data`` with
N=131072, complex64 data, three blocks per iteration, two warm-up iterations,
five measured iterations and three independent worker replicates. The host
was an AMD Threadripper PRO 3995WX with CPU affinity 8--11 and an RTX 4090;
the runtime was CPython 3.11.9, NumPy 1.26.4 and Torch 2.13.0+cu130.

All 63 timed cells passed trigger and aggregate-output-norm comparisons. Six
separate untimed probes confirmed native batch correlation. Default Torch CUDA
was the fastest measured route: at B8 and B32 its one-thread throughput ratios
against the matching standard-CPU replicates were 11.35--11.42 and
34.23--34.34, respectively. Default Torch CPU and the requested CPU-native
configuration were slower than standard CPU in every measured cell. Requested
CUDA-native was slower than default Torch CUDA at B8 and B32.

These are observations on a shared workstation, not isolated-device results
or a pre-change/post-change speedup. Aggregate norms do not establish pointwise
parity. The harness excludes CLI startup, I/O, waveform generation, chi-square
and the stubbed sine-Gaussian veto. The native probes establish correlation
only, and these small samples do not establish production p99 latency. The
record does not qualify cu126, MPS, other Torch versions, or later revisions;
see :ref:`torch-runtime` and :ref:`torch-testing` for current support boundaries.

Expected renderer outputs
-------------------------

The plot renderer reserves the following names when compatible source artifacts
are supplied. Their presence in documentation is conditional on publishing the
corresponding sealed artifact and provenance.

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

   * - File
     - Current status
     - Minimum source evidence
   * - ``torch_live_batch_scaling.png``
     - Not published
     - Raw samples and work counts, parity and route results, workload
       dimensions, synchronization, and the timing boundary.
   * - ``torch_latency_breakdown.png``
     - Not published
     - Raw latency samples, percentile convention, route definitions,
       synchronization, workload, and CI provenance.
   * - ``torch_cold_warm_latency.png``
     - Not published
     - Separate first-use and steady-state samples with warm-up and cache
       boundaries.
   * - ``torch_cuda_memory.png``
     - Not published
     - Allocated and reserved CUDA peaks, their exact APIs and sampling
       boundary, plus explicit OOM cells where tested.
   * - ``torch_parity_error.png``
     - Not published
     - Per-replicate parity metrics, tolerances, route identity, and complete
       pass/fail disposition.
   * - ``torch_matched_filter_symm_scaling.png``
     - Not published
     - Exact matched-filter workload, independent parity, raw samples, devices,
       software stack, warm-up, and CI run.
   * - ``torch_cpu_thread_scaling.png``
     - Not published
     - Torch and BLAS thread counts, affinity, topology, power policy, and raw
       identical-work controls.
   * - ``torch_performance_dashboard.png``
     - Not published
     - Every compatible input artifact, recomputation method, checksums, and
       provenance.
   * - ``torch_inference_acceleration.png``
     - Not published
     - Exact model, detector, and batch inputs; parity and residency results;
       raw end-to-end and resident samples; and environment provenance.

Publication gate
----------------

A result becomes substantiated only when the exact source revision is
accompanied by:

* independent parity/error and route-residency results;
* all per-sample measurements, failures, deadline misses, and OOM cells;
* complete workload, software, hardware, thread, warm-up, synchronization, and
  timing-boundary metadata;
* a retained CI or runner manifest, commands, run URLs, artifact checksums, and
  plot-generation provenance; and
* the applicable views from the required plot suite in
  :ref:`torch-performance`.

Generate plots into a review directory, never directly into documentation, and
publish the sealed source artifact beside each accepted image. If the source
artifact or provenance becomes unavailable, remove the image or explicitly
downgrade it to an unsubstantiated reference snapshot.
