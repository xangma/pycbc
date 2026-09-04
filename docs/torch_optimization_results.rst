.. _torch-optimization-results:

Torch performance evidence status
=================================

This documentation contains no published performance result. Historical PNG
files were deliberately omitted because no raw per-sample artifact,
environment manifest, or retained CI provenance was available to verify them.

An image without those sources is not evidence for a speedup, regression,
crossover point, support guarantee, or optimization promotion. The distinction
between substantiated measurements, derived plots, reference snapshots, and
required evidence is defined in :ref:`torch-performance`.

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
