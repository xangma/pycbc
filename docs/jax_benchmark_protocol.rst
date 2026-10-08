.. _jax-benchmark-protocol:

JAX benchmark protocol
======================

Complete-search benchmarks compare an unchanged, pinned PyCBC CPU reference
with a candidate revision. Scientific qualification, unprofiled timing and
profiling are separate steps. Successful execution, matching trigger counts
or a plausible timeline does not establish scientific agreement or speedup.
This page defines the required evidence; it does not certify a particular
campaign. Incomplete or failed gates remain visible in diagnostic reports.

Reference arms and reproducibility
----------------------------------

Freeze four execution arms before collecting results:

* ``original_cpu``: a clean, built checkout of the pristine reference revision,
  using its original executable and calculations.
* ``branch_cpu``: the candidate checkout using the standard CPU executable.
* ``jax_cpu``: the candidate's dedicated JAX executable on CPU.
* ``jax_cuda``: the candidate's dedicated JAX executable on CUDA.

Check ``branch_cpu`` against ``original_cpu`` independently of JAX. Under
identical deterministic settings, require byte-exact scientific outputs and
metadata; investigate reference variability separately. The JAX numerical
tolerances below do not authorize changes to native CPU behavior. Compare
both JAX arms with the pristine reference and the branch CPU arm. An original
validation route inside JAX is an additional diagnostic; it does not replace
an independently executed pristine reference. Unsupported arms remain
unavailable rather than being replaced with another implementation.

Record source revisions and hashes, working-tree changes, native-extension
build provenance, executable and imported-module origins, dependency versions,
hardware, commands, environment, selected devices and resource bindings.
Freeze frame, bank, configuration and injection hashes. Verify source and input
identity before and after each run, including child processes and MPI ranks.
A source or numerical-configuration change requires affected qualification
and timings to be repeated. Bind every receipt, comparison and plot to the
source and workload it actually measures.

Workload and numerical configuration
------------------------------------

Use identical physical inputs, distinct templates, search geometry, PSD,
vetoes, thresholds, clustering and output options across matched arms. The
standard comparison uses 2048 Hz data and observed ``complex64`` matched-filter
arrays. Record strain, PSD, template and statistic dtypes separately; this
requirement does not disable x64 support used by phase arithmetic or other
calculations. Supplementary rates or precisions need separate labelled results.

Hold waveform physics fixed to original PyCBC/LAL. Hash stored waveform banks;
for runtime generation select the original waveform validation route in the
JAX executable, as described in :doc:`jax_waveform`. A campaign measuring a
different provider needs independent waveform qualification and its own
reporting scope. Preserve the default on-device JAX conditioning in default
JAX arms; original numerical controls are explicitly labelled validation arms.
Record decompression, generation, FFT and every selected reference operation.
For exact cross-process FFT comparisons, select ``--fft-backends numpy`` in
both searches; other original backends require controlled buffer/layout and
planning evidence, as explained in :doc:`jax_fft_numerical_differences`.

For Inspiral, freeze FFT length, segment overlap, start/end padding, waveform
duration support and the retained interval. Check completed segments and
boundary injections; waveform-duration bounds alone do not prove boundary
correctness. For Live, freeze chunk length, buffer fill, PSD refresh policy,
valid-detector intervals, coincidence configuration and MPI topology. Invalid
or buffer-fill blocks contribute no searched work. Retuning geometry for each
backend is a separate experiment, not a matched comparison.

Keep compressed-bank decompression, runtime waveform generation and prepared
filter API measurements distinct. An isolated FFT or prepared-batch rate does
not measure the complete frame-to-output search. Use banks larger than every
tested batch setting, including production partial batches; batch size does
not increase the physical work.

Frozen scientific gates
-----------------------

Evaluate the following rules against the unchanged reference. Numerical rules
apply elementwise, with the reference supplying the relative-error denominator;
nonfinite values require matching declared semantics rather than subtraction.
Do not fit thresholds after seeing the candidate. These complete-search gates
do not replace or relax tighter stage-specific validation or diagnostic budgets;
retain each separately declared rule and its verdict.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Evidence
     - Required comparison
   * - Trigger identities, detector membership, discrete fields, degrees of
       freedom, geometry and scientific metadata
     - Exact agreement; trigger times agree on the exact sample grid
   * - SNR, chi-square and other ordinary numerical observables
     - ``atol=1e-5, rtol=1e-4``
   * - Template sensitivity, ``sigmasq``
     - ``atol=1e-5, rtol=1e-5``
   * - Coalescence phase
     - Circular distance at most ``1e-4`` radians
   * - Full PSD arrays in the active analysis band
     - ``atol=0, rtol=1e-4``; verify out-of-band masks and infinity semantics
   * - Full conditioned strain and each segment's strain
     - ``atol=1e-5, rtol=1e-4``
   * - Other captured evidence arrays and normalized scientific configuration
     - Exact agreement

Use ``abs(candidate-reference) <= atol + rtol*abs(reference)`` for the stated
absolute/relative rules. Preserve original dtype, shape, epoch, spacing and
support. Capture full PSDs and conditioned strain, including intermediate
segment geometry and configuration; a saved PSD subset or trigger table cannot
satisfy these gates. Identity-aligned diagnostics must retain missing,
duplicate and off-grid triggers and must not silently erase ordering failures.

Decision invariance needs source- and workload-bound audits of candidate
selection, threshold crossings, clustering and final ranked order. Preserve
scientific output ordering and detector labels in repeatability comparisons.
Exclude only explicitly listed runtime timing/rate and execution/provenance
fields from the scientific verdict, such as an output destination; do not exclude scientific configuration, geometry or an
unexpected dataset difference. Retain raw fields and every exclusion.

Record each gate as passed, failed or unavailable. Missing evidence is not a
pass, and passing helper tests does not supply absent complete-search evidence.
Stop equivalent-output performance qualification on failure. Separately
requested diagnostic timing may continue with the failed/missing gates retained
and with equivalent-output speedup and sustained-capacity claims disabled.
Publishing diagnostic evidence does not make those gates pass. See
:doc:`jax_numerical_differences` for independent original calculation controls.

.. _jax-timing-boundaries:

Timing, completion and warmup
-----------------------------

Report the following scopes separately with equivalent boundaries across arms:

* **Full-process wall time:** launch through completed device work, collected
  results, closed output files, child/rank completion and process exit. Include
  imports, setup, I/O, compilation/cache loading, conditioning, generation or
  decompression, output and shutdown, plus disclosed observation overhead.
* **Application timers:** state the actual source boundaries and omissions.
  An internal timer ending before output or device completion is a host timer,
  not a synchronized compute measurement. Show time outside it without
  automatically calling that residual Python overhead.
* **Warmed calculation or Live loop:** state included stages and excluded setup
  and warmup. Complete preceding device work before starting; complete all
  measured work and transport/output drains before stopping. Synchronize all
  ranks and use a common interval covering the participating ranks. Count only
  work completed inside the declared measurement scope.

Declare persistent-cache isolation, priming and reuse policy. Prime every
production signature, including partial template batches, FFT geometries,
trigger/veto buckets, growing coincidence histories and dtype/static-argument
variants. Fresh-process cache hits still incur loading and may require frontend
tracing/lowering. Backend-duration notifications can include cache retrieval;
record function, duration, thread and cache events before interpreting counts.
A cache-write notification is not a complete count of backend compilations.

A fully warmed interval must demonstrate no tracing, lowering, backend compile
or persistent-cache request inside its boundary, including cache hits. Audit
runtime graph instantiation separately where applicable. A fixed number of
initial blocks, a seeded disk cache or an empty compiler log is insufficient.
If late work remains, label the interval a measured suffix containing JIT/cache
work. Retain cache evidence and setup/warmup costs beside the timing.

Warm complete replays with fresh scientific state only when the executable
supports safe state reset. Preserve compiled caches while resetting strain
buffers, filter controls, coincidence estimators, cursors and output state;
compare excluded and measured replay outputs. Do not repeatedly invoke a CLI
with surviving background threads as a substitute for that contract. A wrapper
containing several replays does not replace a separate single-search
full-process measurement. For a shorter suffix, reduce completed detector-time
and work explicitly.

.. _jax-batch-scaling:

Repetitions, scaling and resources
----------------------------------

Use at least three fresh, unprofiled processes for every compared setting,
rotating or counterbalancing arm order. Qualification and prime processes are
excluded. Report every sample, medians and observed ranges; retain failures,
timeouts and incomplete work. Do not substitute the best sample or profiled
wall time. Equivalent timing excludes the same work from every compared arm.

Increase distinct template count or unique valid detector-time geometrically
while preserving the physical distribution and scientific configuration.
Predeclare convergence. The standard sustained-capacity criterion is less
than 5% change in median capacity at two successive workload doublings, with
setup plus time outside the internal calculation timer below 10% of full wall
time in every repeat at the final three sizes. Check each backend separately.
If these requirements fail, report a scaling curve and finite-workload rates,
not sustained capacity. Sweep batch sizes separately with scientific checks,
production partial batches and measured memory headroom; do not prescribe an
optimal batch size from array sizes alone.

Record CPU topology, sockets, NUMA, physical cores, SMT siblings, affinity,
frequency policy, memory, numerical thread limits and GPU sharing. Verify
actual bindings of every process/rank. The standard CPU reference uses one
physical core and numerical thread pools limited to one. Count the Live root
rank and all worker host resources. SMT siblings count as one physical core;
affinity is not a reservation. Record host/per-core/process load before, during
and after runs and distinguish controlled, reserved and shared-host diagnostics.
Do not stop unrelated services merely to satisfy an idle-host criterion.

Single-worker isolation and full-machine CPU throughput are separate
experiments. For saturation, launch synchronized independent workers on
distinct physical cores with separate outputs. Report each worker's work/time,
the concurrent makespan and all allocated cores. Clearly identify replicated
workloads; replicas do not create a larger distinct bank or unique input
interval. Multiple threads in one process and multiple processes sharing a GPU
are different experiments and require their own resource/qualification scope.

Work and capacity
------------------

Let :math:`W` be the sum, over completed distinct physical templates and
detectors, of the union of valid searched intervals. For uniform work,
:math:`W=N_{\mathrm{distinct}}\sum_d T_{\mathrm{valid},d}`. Do not count segment
overlap, duplicate templates, invalid blocks, nominal frame length or batch
size again. Validate these quantities from actual execution and outputs,
rather than trusting requested counts.

For elapsed time :math:`t` and allocated physical CPU cores :math:`R_{CPU}`
or GPUs :math:`R_{GPU}`, report:

.. math::

   C_{CPU}=\frac{W}{tR_{CPU}}, \qquad C_{GPU}=\frac{W}{tR_{GPU}}.

These are single-detector-equivalent templates/core and templates/GPU at real
time. For :math:`D` equal-duration detectors, simultaneous network-bank
capacity is this value divided by :math:`D`; state detector scope beside the
metric. GPU figures also disclose allocated host cores. Concurrent worker
capacity uses :math:`\sum_i W_i/(t_{makespan}R_{CPU})`, not the sum of
already-normalized per-worker capacities. Report realtime factor separately
with its data-time and elapsed-time definitions.

Unpaced throughput does not establish realtime latency. Paced Live tests must
use declared arrival times and report per-block completion latency, deadline
misses, backlog/lag, drops, warmup and incomplete blocks. Separate these results
from unpaced capacity and record buffering/backpressure and final drains.

Profiles, transfers and memory
------------------------------

Profile separate executions and compare their saved scientific outputs with
qualified unprofiled runs. Include child processes/ranks and retain raw traces,
clock-alignment evidence, source/input hashes, stage logs and figure commands.
Publish a whole-process timeline, a measured calculation/Live-loop view and a
labelled 400 ms zoom. Show templates/core and templates/GPU scaling alongside
full-process and calculation timing boundaries.

Keep host call-stack time, native sampled cycles and CUDA kernel/copy time in
separate panels with their own denominators. Separate FFT planning/loading
from execution and measure conditioning, PSD, waveform preparation,
correlation/IFFT, vetoes, thresholding/clustering, coincidence, MPI, output and
wait intervals where observable. Nested host ranges overlap; cumulative call
time is not additive. Host submission markers do not imply device completion.
Do not insert per-stage fences into a throughput trace: they can serialize
prefetch and alter the gaps being investigated. Label any separately
synchronized diagnostic and bound instrumentation/sampling overhead.

Attribute transfers to the program using process-bound CUDA/CUPTI evidence
where available; system-wide PCIe counters do not establish program copy
bytes. Define whether copy bytes count completed or overlapping events. The
union of captured kernel/memcpy intervals measures recorded activity duration,
not SM occupancy, saturation or whole-device utilization; disclose exclusions
such as memsets and unrelated processes. Inspect wait/launch gaps before
attributing a bottleneck to one kernel.

Record host RAM and VRAM peaks, live allocations, allocator reservations,
compilation/cache memory, FFT scratch and concurrent process use. Account for
aliases when counting unique buffers. Sampled peaks can miss transients;
retain allocation failures and their stage. No failed allocation alone proves
an optimal batch limit or a unique memory cause.

.. _jax-campaigns:

Campaign artifacts and supported tools
--------------------------------------

The `PyCBC JAX benchmarks repository
<https://github.com/xangma/pycbc-jax-benchmarks>`_ contains the separately
versioned campaign manifests, entrypoints and evidence workflow. Pin its
revision as well as PyCBC and follow its maintained command help:

.. code-block:: console

   python -m pycbc_jax_benchmarks run CAMPAIGN.json --inputs INPUTS_DIR --sources SOURCES_DIR --output OUTPUT_DIR

The manifest supplies frozen source/input identities, native executable
arguments and declared execution arms; the directories supply their local
checkouts and data. Optional runtime/resource mappings are documented by the
runner. Running a manifest does not supply missing scientific captures or
prove a fully warmed interval: inspect each gate's availability and verdict.
A campaign must provide the arms, scientific captures/audits, repeat order, timing/cache
policy, resources, comparisons and plot recipes required above. A manifest
or successful preflight alone is not an executed qualification. Keep raw
successful and failed receipts with derived tables and a hash-bound archive.

From a PyCBC checkout, the maintained exact comparator accepts two HDF files
or directories with identical nonempty relative ``.hdf`` inventories:

.. code-block:: console

   python tools/compare_jax_search.py reference.hdf candidate.hdf --output comparison.json

It checks names, shapes, dtypes, value bits, attributes and storage properties,
preserving signed zeros and NaN payloads. It does not reorder triggers or apply
the numerical tolerances above. It exits unsuccessfully on a mismatch. Explicit
exclusions must be reviewed with the remaining result:

.. code-block:: console

   python tools/compare_jax_search.py reference.hdf candidate.hdf --ignore-attribute command_line --output comparison.json

``--ignore-attribute`` excludes that name wherever it occurs;
``--ignore-dataset`` excludes an exact HDF path without a leading slash. Both
are repeatable and every matched exclusion is reported. This comparator does
not capture full strain/PSDs or perform the independent decision audits.

The FFT tool measures public forward/inverse round trips, separating the first
call from warmed synchronized samples and checking the inverse outside timing:

.. code-block:: console

   python tools/benchmark_jax_fft.py --processing-scheme jax:cuda:0 --dtype complex64 --length 65536 --repeats 20 --output fft.json

Use ``--processing-scheme cpu`` for the original CPU calculation.
``--fft-backend`` selects its FFT implementation and that used by original
JAX controls. Repeated ``--reference-operation fft`` and
``--reference-operation ifft`` select those boundaries independently.
``--profile PROFILE_DIRECTORY`` records the warmed JAX loop with
``jax.profiler.trace``. Device results complete within each timed sample;
construction, setup, file I/O and output writing are outside this benchmark.
Record unprofiled timings separately. A passing FFT round trip or a kernel
rate does not qualify a complete search.
