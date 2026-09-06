.. _torch-performance:
.. _torch-performance-summary:

Benchmarking Torch
==================

For executable measurements, start with :ref:`torch-inspiral-optimized`.
The :ref:`torch-inspiral-reference` page records the before campaign
and normal CPU tuning. The :ref:`torch-optimization-results` page
contains earlier supporting component measurements.
This page explains how to collect a reproducible benchmark. Check numerical
agreement and the selected implementation using :ref:`torch-parity` before
timing it.

Run a benchmark
---------------

The live-batch harness launches isolated workers and retains raw timings,
parity results, route settings and environment metadata:

.. code-block:: console

   python tools/bench_production_live_batch.py orchestrate \
     --root SOURCE_SET --python PYTHON --output production_live_batch.json \
     --batches 1 8 32 128 512 1024

``SOURCE_SET`` contains clean, built ``original/`` and ``branch/`` checkouts.
Optional ``branch_cpu/`` and ``branch_cuda/`` directories override the current
backend sources. To compare routes in one current checkout, pass its path and
add ``--routes branch_standard torch_cpu torch_cuda``; omit ``torch_cuda``
on CPU-only systems.

Use ``--help`` to set batches, FFT size, blocks, threads, replicates, samples,
warm-ups, threshold, affinity, seed, routes, CUDA device and call surface.
Use a new output path. Dirty sources and existing output files are rejected
unless explicitly allowed.

The default workload calls ``LiveBatchMatchedFilter.process_data``. It excludes
CLI startup, frame I/O, PSD estimation, bank loading, waveform generation and
workflow scheduling. Its trigger and aggregate-norm checks do not establish
pointwise agreement of SNR arrays. Treat it as a library benchmark.

Generate plots from retained artifacts:

.. code-block:: console

   python tools/generate_torch_performance_plots.py --artifacts-dir ARTIFACTS --output-dir OUTPUT

This renderer accepts production live-batch artifacts and older component
artifacts, labeled by scope. Matched-filter, CPU-profile and inference inputs
are optional. It writes available figures and
``torch_performance_plot_manifest.json``; it cannot supply missing measurements.
The September 2026 figures have separate reproduction instructions in
:ref:`torch-benchmark-details`.

Choose fair comparisons
-----------------------

Run each backend in a fresh process with the same scientific inputs and public
outputs. Keep fallback routes separate from native routes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Comparison
     - When to include it
   * - Standard CPU, revision under test
     - Always: the control for changes unrelated to Torch.
   * - Torch CPU, same revision
     - For backend overhead, CPU scaling or CPU optimization claims.
   * - Torch CUDA or MPS, same revision
     - For claims about that accelerator and software environment.
   * - Standard CPU, pre-Torch revision
     - Optional historical comparison, only if its environment can be rebuilt.

Small workloads can be dominated by launch, transfer or compilation costs.
Vary batch size, bank size, FFT length and dtype independently before claiming
a general crossover point. The live harness ties bank size to batch size and
uses one FFT size and thread count per invocation, so it cannot establish that
matrix in a single run.

Measurement checklist
---------------------

#. **Identify the run.** Record the full commit, dirty-tree state or patch
   digest, command, working directory, environment lock, dependency versions,
   optimization variables and initial cache state.
#. **Describe the machine.** Record OS/kernel, CPU/topology, RAM, GPU/memory,
   driver/runtime, affinity, thread and BLAS settings, accelerator visibility,
   power/clock policy and CPU governor where available.
#. **Fix the inputs.** Retain seeds and hashes, sample rate, duration, detector
   count, waveform parameters, batch/bank dimensions, FFT length, dtype and
   thresholds. Check parity, route and output device for every timed cell.
#. **Separate startup from repeated calls.** Report cold and warm timings
   separately, with imports, initialization, allocation, planning, graph capture
   and compilation boundaries stated. Record warm-up counts and cache policy.
#. **Time completed work.** Synchronize the selected CUDA or MPS device before
   and after each timed region. Include orchestration, required transfers and
   result extraction in end-to-end latency. Label resident compute separately.
#. **Keep every sample.** Retain warm-up markers, durations, work counts,
   failures, timeouts and OOMs. Compute throughput and p50/p95/p99 with the
   sample count and percentile method. Do not pool repeated samples as
   independent process runs.
#. **Control run order.** Randomize or counterbalance backends and repeat
   complete blocks where practical to reduce thermal, clock and cache bias.
#. **Verify the output.** Recompute summaries, check hashes and inspect figures
   against the manifest. Publish raw evidence with the plots.

For memory claims, name the measurement API and interval, including allocated
and reserved device memory where available. Keep OOMs at their attempted
matrix cells. For deadline claims, retain timestamped completion offsets,
deadlines, the arrival model and miss counts/rates; a standard deviation alone
does not describe deadline jitter.

Which plots are needed?
-----------------------

Match the evidence to the claim. A broad backend or optimization claim needs
the views below; a narrow change may mark a view not applicable and explain
why in its manifest.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - View
     - What it must show
   * - Throughput and latency
     - Equivalent completed work; p50/p95/p99, sample counts and uncertainty
       appropriate to the independent runs.
   * - Crossover matrix
     - Independent batch, bank, FFT-length and dtype variation, including
       unsupported cells and OOMs.
   * - Timing breakdown
     - Transfers, resident compute, allocation, orchestration and extraction
       for the same workload; separate cold and warm costs.
   * - Memory
     - Peak host/device memory, measurement APIs and OOM boundaries.
   * - Correctness
     - Error metrics and pass/fail for every timed route and input family.
   * - CPU scaling and hardware
     - Thread counts, affinity and topology; equivalent workloads on every
       hardware class covered by the claim.
   * - Deadlines, where relevant
     - Lateness over time, latency quantiles and miss counts/rates.

The current renderer supports live throughput and tails, latency breakdown,
cold/warm timing, CUDA allocator peaks and parity views when inputs contain
them. Optional inputs add matched-filter scaling, CPU scaling/profile,
dashboard and inference views. It does not yet render the independent crossover
matrix, transfer/resident breakdown, host-memory/OOM boundary, deadline jitter
or hardware matrix. Kernel benchmarks and allocator counters cannot replace
those measurements. Extend the data and renderer together.

Keep results reproducible
-------------------------

Store versioned JSON/JSONL with run/cell IDs, UTC timestamps, raw samples,
inputs, parity and route results, environment and machine records, failures,
summary algorithms and SHA-256 checksums. Keep logs and environment exports
beside the data. Include benchmark and renderer versions and generated filenames.

For CI runs, retain provider, workflow revision, run/job/attempt IDs and URLs,
runner labels, event, artifact name, retention policy and conclusion. For manual
runs, record equivalent runner provenance. Neither a hostname nor an expiring
CI link substitutes for a durable artifact.

Each figure's caption or adjacent manifest must identify the run, source
revision, input checksum, benchmark command and CI job or manual runner.
Figures combining runs must identify every input and reject incompatible
workloads or schemas. Inspect all images before publishing.

A **substantiated measurement** has enough raw data and provenance to recompute
the result. A **derived plot** identifies that data and its generation command.
An image without recoverable evidence is only a **reference snapshot**.
Required or proposed evidence describes work still to collect, not a result.
