.. _torch-performance-summary:

============================
Torch performance evaluation
============================

This page defines how to interpret and refresh the benchmark figures in
:doc:`torch_optimization_results`. Scientific equivalence is a prerequisite
for performance comparison; faster output that fails the applicable parity
policy is not an accepted result.

Measurement cells
=================

Use isolated processes for each source/backend/device cell so schemes, dispatch
caches, FFT plans, allocators, and compiler state do not leak between cells.

.. list-table:: Comparison cells
   :header-rows: 1
   :widths: 12 25 24 39

   * - Cell
     - Source
     - Execution
     - Purpose
   * - A
     - Recorded pre-Torch baseline
     - Standard CPU
     - Historical reference when the original revision is still reproducible
   * - B
     - Current branch
     - Standard CPU
     - Detect branch regressions unrelated to Torch execution
   * - C
     - Current branch
     - Torch CPU
     - Isolate backend cost or benefit without accelerator variation
   * - D
     - Current branch
     - Torch CUDA or MPS
     - Measure a qualified accelerator route

A current result need not include cell A when its historical environment can no
longer be reproduced. It must always retain cell B, because B-to-C and B-to-D
are the actionable comparisons on the current source tree.

Timing contract
===============

Each result must state whether it is cold or warm. Cold timing includes imports,
allocator and device initialization, FFT planning, and compilation. Warm timing
uses initialized persistent workers and is the primary measure for repeated
search and inference workloads.

Also record:

* source revision and dirty-tree state;
* Python, PyCBC, Torch, NumPy, SciPy, and LALSuite versions;
* CPU/GPU model, accelerator runtime, dtype, and device;
* Torch and BLAS thread settings plus process affinity;
* input dimensions, duration, sample rate, detector count, and batch shape;
* warm-up count, timed repetitions, and the summary statistic; and
* whether allocation, host/device transfer, and result extraction are timed.

Synchronize an asynchronous accelerator immediately before starting and after
ending a timed interval. Report throughput only for the same scientific work
and public output contract.

Accepted scope
==============

Search measurements may cover the supported Torch array, FFT, filter,
matched-filter, and live-batch paths. Waveform measurements are limited to the
six registered TaylorF2-family ports listed by the generated table in
:doc:`waveform`. Measurements may additionally cover supported inference
reductions, relative-binning batches, and detector/network response.

Do not infer support from an old benchmark label. Registry entries and focused
tests define the available waveform interfaces. Likewise, a kernel
microbenchmark does not establish end-to-end workflow readiness.

Parity qualification
====================

Before publishing a timing comparison, verify:

#. shapes, dtypes, sampling metadata, epochs, and detector ordering;
#. numerical agreement under the focused test's stated tolerances;
#. equivalent thresholds and trigger-selection semantics for search outputs;
#. the expected implementation route and output device; and
#. absence of an unintended host conversion in paths documented as resident.

See :doc:`torch_parity` for terminology and :doc:`torch_workflows` for current
commands.

Checked-in figures
==================

The PNGs embedded by :doc:`torch_optimization_results` are checked-in reference
snapshots. A documentation build reads them but does not run benchmarks or
alter them. The measurement JSON used for the current snapshots is not included
in this source tree, so the images alone are not a reproducible benchmark
record. Do not use them as evidence for a new performance claim without the
corresponding source artifacts.

``tools/generate_torch_performance_plots.py`` is a maintainer utility for
rendering the current search and inference/detector summaries. It requires an
explicit directory containing complete benchmark artifacts and fails rather
than substituting built-in measurements. Run it only when intentionally
refreshing evidence, review every changed image, and commit the source evidence
or measurement record alongside the rendered output. It writes directly to
``docs/images`` unless ``--output-dir`` is supplied.

The current documented figure set is:

* ``torch_live_batch_scaling.png``;
* ``torch_latency_breakdown.png``;
* ``torch_matched_filter_symm_scaling.png``;
* ``torch_cpu_thread_scaling.png``;
* ``torch_performance_dashboard.png``; and
* ``torch_inference_acceleration.png``.
