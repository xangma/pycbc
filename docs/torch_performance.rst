.. _torch-performance:
.. _torch-performance-summary:

Torch performance evidence
==========================

This page defines the benchmark and provenance contract for Torch changes.
Scientific parity and route qualification are prerequisites; a faster result
that used different scientific work, an unintended fallback, or the wrong
device is rejected. Capability and fallback boundaries are documented
separately in :ref:`torch-scheme`.

Evidence vocabulary
-------------------

Use these terms consistently in documentation and pull requests:

Substantiated measurement
   Raw per-sample records, workload inputs, environment metadata, and retained
   CI or runner provenance are available for the exact revision. The reported
   statistic can be recomputed from them.

Derived plot
   A rendering generated from substantiated measurements. It must link to or
   identify the raw artifact and generation command.

Reference snapshot
   A checked-in image whose source measurements or provenance are unavailable.
   It can illustrate an intended presentation, but cannot substantiate a
   performance claim.

Required or proposed evidence
   A benchmark cell, plot, or acceptance check that must be collected. It is a
   protocol requirement, not a measured result.

Older binary plots remain unpublished because their source JSON and CI run
manifests are unavailable. A separate historical len campaign now retains raw
samples, runner records and explicit limitations in
:ref:`torch-optimization-results`. Those measurements apply only to their
recorded revision; they do not qualify later code or every proposed plot below.

Comparison cells
----------------

Run each cell in an isolated process so schemes, allocator state, FFT plans,
compiler caches, and global feature flags do not leak between cells.

.. list-table::
   :header-rows: 1
   :widths: 14 24 24 38

   * - Cell
     - Source
     - Execution
     - Purpose
   * - A
     - Reproducible pre-Torch revision
     - Established CPU scheme
     - Optional historical baseline. Omit it if its environment cannot be
       reconstructed rather than approximating it.
   * - B
     - Revision under test
     - Established CPU scheme
     - Required current-tree control for detecting unrelated branch changes.
   * - C
     - Revision under test
     - Torch CPU
     - Required for claims about backend overhead, CPU scaling, or CPU
       optimizations.
   * - D
     - Revision under test
     - Torch CUDA or MPS
     - Required for a claim about that exact qualified accelerator.

Compare only cells that perform the same scientific work and expose the same
public output contract. Record fallback cells separately; do not combine them
with native-route samples.

Reproducible protocol
---------------------

#. **Pin the source and environment.** Record the full commit, dirty-tree
   state or patch digest, exact command, Python environment lock, imported
   dependency versions, and every Torch optimization variable. Record whether
   compilation caches are initially empty.
#. **Describe the machine.** Record OS and kernel, CPU model and topology,
   memory, accelerator model and memory, driver, CUDA or MPS runtime, power and
   clock policy, process affinity, CPU governor where available, BLAS settings,
   Torch intra-op/inter-op threads, and accelerator visibility. A hostname is
   not a hardware description.
#. **Fix the workload.** Record random seeds and input hashes plus sample rate,
   duration, detector count, waveform model and parameters, batch size, bank
   size, FFT length, dtype, thresholds, and all dimensions affecting work. A
   crossover study must vary batch size, bank size, FFT length, and dtype
   independently rather than labeling one fixed input “representative.”
#. **Qualify correctness first.** Run the relevant checks from
   :ref:`torch-parity`. Assert the selected implementation route and output
   device, and detect prohibited host conversion. Store parity/error metrics
   with the benchmark cell.
#. **Separate cold and warm execution.** Cold timing includes stated startup
   costs such as imports, device initialization, allocation, FFT planning, and
   compilation. Warm timing uses initialized persistent workers after a
   recorded warm-up count. Never merge cold and warm samples.
#. **Define timing boundaries.** Label end-to-end, host-to-device transfer,
   resident compute, device-to-host transfer, allocation, and result extraction
   regions. Include orchestration and required synchronization in end-to-end
   latency. A resident-kernel timing is a separate metric.
#. **Synchronize accelerators.** Synchronize the selected CUDA or MPS device
   immediately before and after each timed region. Unsynchronized wall time is
   enqueue time and must not be reported as execution latency.
#. **Retain individual samples.** Store every warm-up disposition, timed
   sample, failure, timeout, and OOM. Derive throughput and p50, p95, and p99
   latency from raw samples with the sample count and percentile method. Do not
   retain only a median.
#. **Randomize or counterbalance run order.** Avoid assigning all thermal,
   clock, or cache drift to one backend. Record the order and repeat complete
   blocks when practical.
#. **Validate the artifact.** Recompute summaries from raw rows, verify artifact
   checksums, and generate plots without embedded fallback data. Review plot
   labels against the manifest before publication.

Memory measurements must report the API and sampling boundary used, including
allocated and reserved accelerator memory where available. Record an OOM at
the attempted matrix cell; do not remove failed cells or replace them with the
largest successful size.

Deadline-sensitive workloads must record the deadline, completion-time offset,
miss count/rate, and latency distribution over time. “Jitter” should be derived
from timestamped samples under a stated arrival model, not from a single
standard-deviation summary.

Required plot suite
-------------------

The following is the minimum evidence set for a broad optimization or backend
performance claim. A narrowly scoped change may mark a plot not applicable,
but must explain why in its artifact manifest.

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Required view
     - Contents
     - Prevents
   * - End-to-end throughput
     - Scientifically equivalent completed work per unit time, with workload
       dimensions and confidence or sample count.
     - Presenting kernel rate as application throughput.
   * - p50/p95/p99 latency
     - All three quantiles from retained samples for end-to-end and any
       separately claimed resident region.
     - Hiding tail regressions behind a median.
   * - Crossover heatmaps
     - CPU versus Torch CPU/accelerator outcome over batch size, bank size, FFT
       length, and dtype; unsupported and OOM cells visibly encoded.
     - Generalizing a single favorable shape.
   * - Transfer/resident breakdown
     - Host-to-device, resident compute, device-to-host, allocation, and
       orchestration contributions using the same workload.
     - Claiming residency or speed while excluding required transfer.
   * - Memory and OOM
     - Peak host and device memory, allocated/reserved values where applicable,
       and explicit OOM boundaries over the workload matrix.
     - Silently dropping impractical cells.
   * - Cold versus warm
     - First-use and steady-state distributions with import, plan, allocator,
       graph, and compile boundaries stated.
     - Amortizing setup without disclosure.
   * - Deadline jitter
     - Timestamped lateness, p50/p95/p99, and miss count/rate for any
       deadline-sensitive or streaming claim.
     - Hiding scheduling stalls and rare misses.
   * - Parity/error
     - Numerical error metrics and pass/fail boundary for every timed route,
       shape, dtype, and device family.
     - Separating performance from scientific validity.
   * - CPU thread scaling
     - End-to-end and resident behavior across recorded Torch and BLAS thread
       counts, affinity, and physical topology.
     - Treating one oversubscribed thread setting as CPU capability.
   * - Hardware matrix
     - Comparable results across every claimed CPU/GPU/MPS class with exact
       software stacks and unsupported cells shown.
     - Converting one machine's result into a support guarantee.

Matched-filter or waveform microbenchmarks may diagnose a result, but do not
replace the end-to-end, transfer, memory, and tail-latency views.

Raw artifact and CI provenance
------------------------------

Use a versioned JSON or JSONL schema. At minimum, preserve:

* a unique run and cell ID, schema version, UTC timestamps, command, working
  directory, source revision, dirty-state description, and patch digest;
* workload parameters, input hashes, random seeds, expected work count, route
  assertion, output device, dtype, and parity/error outcome;
* every raw duration and throughput numerator, warm-up marker, timeout,
  deadline outcome, exception, and OOM;
* resolved Python and package versions, Torch build and backend availability,
  driver/runtime versions, environment variables, thread settings, and cache
  policy;
* CPU, memory, accelerator, topology, affinity, power, and clock metadata;
* summary algorithm and percentile convention, benchmark and plot-generator
  versions, generated-file names, and SHA-256 checksums; and
* CI provider, workflow file and revision, run/job/attempt IDs and URLs, runner
  labels, triggering event, artifact name, retention policy, and conclusion.

Store logs and environment exports beside the samples, not inside a plot.
Anonymize a sensitive hostname if necessary, but retain a stable runner ID and
the material hardware description. A CI URL alone is insufficient because
artifacts can expire; retain the raw artifact in the project's chosen durable
location or check in an immutable manifest that resolves to it. If the artifact
is gone, downgrade derived images to reference snapshots.

Every plot caption or adjacent manifest must name its run ID, source revision,
artifact checksum, benchmark command, and CI job. Plots assembled from multiple
runs must identify every input and reject incompatible schemas or workloads.

Current benchmark and plot tooling
----------------------------------

``tools/bench_production_live_batch.py`` provides a bounded library-level
LiveBatchMatchedFilter harness. Its ``orchestrate`` mode launches isolated child
processes, counterbalances route order, identifies source trees, retains cold
and warm samples, computes p50/p95/p99 and deterministic bootstrap intervals,
records parity and route definitions, captures available runtime and peak CUDA
allocated/reserved-memory metadata, and seals schema-v3 JSON with a content
hash.

Parity checks reject nonfinite trigger fields and output norms. A failed
comparison makes the benchmark exit unsuccessfully while retaining its artifact
for diagnosis. These checks compare trigger data and aggregate output norms;
equal norms do not establish pointwise waveform equivalence.

Start a campaign with explicit source-set, interpreter, and new output path:

.. code-block:: console

   python tools/bench_production_live_batch.py orchestrate --root SOURCE_SET --python PYTHON --output production_live_batch.json

Use ``--help`` to set batches, FFT size, blocks, threads, replicates, samples,
warm-ups, threshold, affinity, seed, routes, CUDA device, and public versus
private call surface. The default public surface is
``LiveBatchMatchedFilter.process_data``. Dirty sources are rejected unless
explicitly allowed, and existing output is not overwritten by default.

This harness is not a full CLI/search benchmark. Its manifest explicitly
excludes CLI/configuration startup, file/frame input, PSD estimation,
template-bank loading or waveform generation, and workflow scheduling. It also
ties bank templates to the batch dimension, uses a fixed dtype set, and varies
one FFT size and CPU-thread count per invocation. Multiple traceable campaigns
are therefore needed for the independent batch/bank/FFT/dtype crossover, CPU
scaling, and hardware requirements above. The harness does not automatically
supply all CI run IDs, retention data, power controls, transfer regions,
deadline arrivals, or cross-machine normalization; add them to the surrounding
campaign manifest.

``tools/generate_torch_performance_plots.py`` is a renderer, not a benchmark. It
requires ``--artifacts-dir`` and a usable live-batch artifact, preferring the
public production schema when available. Older component-suite JSON is accepted
but labeled as component evidence. Matched-filter, CPU-profile, and inference
inputs are optional. ``--output-dir`` avoids writing directly to
``docs/images``.

When the corresponding source artifacts are supplied, the renderer can
produce:

* ``torch_live_batch_scaling.png``;
* ``torch_latency_breakdown.png``;
* ``torch_cold_warm_latency.png`` (when cold samples are retained);
* ``torch_cuda_memory.png`` (when CUDA allocator peaks are retained);
* ``torch_parity_error.png`` (when replicate parity is retained);
* ``torch_matched_filter_symm_scaling.png``;
* ``torch_cpu_thread_scaling.png``;
* ``torch_performance_dashboard.png``; and
* ``torch_inference_acceleration.png``.

For current production artifacts, the live-batch views include p50 and
p95/p99 latency when retained samples provide them; legacy inputs are visibly
marked when tails are unavailable. The renderer also writes
``torch_performance_plot_manifest.json`` with source scope, routes, dimensions,
metrics, and whether full CLI end-to-end measurement was present.

The production-artifact renderer keeps cold and warm distributions separate,
shows numerical parity maxima plus all-replicate pass/fail state, and reports
both CUDA allocator peak APIs. It does not infer host memory or OOM boundaries
from those allocator counters.

This still does not implement the complete required plot suite. There is no
renderer for an independent batch/bank/FFT/dtype crossover heatmap,
transfer-versus-resident regions, host-memory/OOM boundaries, deadline jitter,
or a multi-hardware matrix. Existing CPU and component figures are not
substitutes for those views. Extend the artifact schema and renderer only after
raw measurements exist; never synthesize missing samples or encode benchmark
numbers in plotting code.

An intentional refresh is:

.. code-block:: console

   python tools/generate_torch_performance_plots.py --artifacts-dir ARTIFACTS --output-dir OUTPUT

Run it from a clean, identified source revision, inspect every changed image
and the generated manifest, and publish the raw artifacts and CI provenance
with the rendered output.
