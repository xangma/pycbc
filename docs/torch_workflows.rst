====================================
Torch workflow parity and benchmarks
====================================

The representative staged Torch gate runs the same deterministic analysis in
isolated source/backend/device cells and compares scientific products before it
credits performance.  It does not claim comprehensive or whole-codebase
coverage.  It is intentionally separate from the focused kernel corpus in
:doc:`torch_parity`: the kernel corpus finds narrow numerical regressions,
while this suite exercises how those kernels compose in selected analysis
workflows.

Comparison matrix
=================

.. list-table::
   :header-rows: 1

   * - Cell
     - Source
     - Backend/device
     - ``lalsimulation`` policy
   * - A
     - Frozen pre-Torch baseline
     - PyCBC CPU
     - Allowed
   * - B
     - Current Torch branch
     - PyCBC CPU
     - Allowed
   * - C
     - Current Torch branch
     - Torch on CPU
     - Recursively blocked
   * - D
     - Current Torch branch
     - Torch on CUDA
     - Recursively blocked

The reported comparisons have distinct meanings:

* A-to-B detects branch regressions that are unrelated to the Torch backend.
* B-to-C isolates the Torch implementation from accelerator differences.
* C-to-D isolates device-specific differences.
* B-to-D is the practical CPU-to-GPU science and performance comparison.

Each cell is a new subprocess.  This prevents global scheme state, dispatch
caches, FFT plans, and allocator state from crossing cell boundaries.  Cells C
and D install ``sitecustomize`` in every Python descendant.  It records Python
processes and backend imports and raises immediately if any process attempts to
import ``lalsimulation``.  Core ``lal`` and ``lalframe`` remain available for
detector geometry, times, and frame I/O; the assertion is independence from
LALSimulation waveform generation.

Coverage levels
===============

The rollout uses five separately reported stages so that a failure can be
localized without mistaking micro-benchmark success for workflow readiness.

.. list-table::
   :header-rows: 1

   * - Level
     - Scope
     - Examples
   * - L0
     - Kernels and public API intersections
     - Arrays, FFT/IFFT, shifts, filters, PSDs, waveforms, matched filtering
   * - L1
     - Deterministic composed workflows
     - Conditioning/whitening, Q-transform, two-detector search, likelihood grid
   * - SCALE
     - Bounded signal-length scale and benchmark workloads
     - 64-second conditioning/search, 3-template search, 32-point inference
   * - L2
     - Real command-line programs
     - ``pycbc_inspiral`` and ``pycbc_inference`` vertical slices
   * - L3
     - Workflow planning and scheduler capability reporting
     - Normalized Pegasus DAGs and explicit HTCondor availability evidence

The versioned profiles are ``tools/torch_workflows/suite.json`` (L1),
``suite_benchmark.json`` (SCALE), ``suite_l2.json`` (L2), and
``suite_l3.json`` (L3).  They use fixed random seeds and no network inputs.  The
L1 search slice includes a small template bank, two synthetic detector streams,
matched filtering, PyCBC's Allen chi-square implementation, trigger
thresholding/clustering, and reweighted ranking.  Its inference slice evaluates
the public single-template model on an exact parameter grid using native
``IMRPhenomD`` in the Torch cells.  SCALE increases signal lengths while
remaining bounded: its search uses three templates and its likelihood workload
uses 32 parameter points.  L2 invokes the checkout's real ``pycbc_inspiral``
and ``pycbc_inference`` entry points and compares their HDF products.  The L2
inference workload is a deterministic real-CLI smoke test using a dummy sampler;
it is not evidence of posterior convergence.  L3 runs the real offline Pegasus
planner and compares normalized DAX semantics; it does not submit jobs to an
external scheduler.

``suite_performance.json`` is a separate performance campaign, not another
rollout acceptance stage.  It exercises four larger regimes: 256 seconds of
2048 Hz conditioning, a high-rate 20--480 Hz Q-transform plane, a two-detector
64-template search, and a 256-point likelihood grid.  Those sequential
workloads keep the same A/B/C/D cells and A-B, B-C, C-D, and B-D comparisons.
The first three search templates are unchanged from SCALE, and every added
template ID and parameter set depends only on its integer index.  The
likelihood configuration retains
the SCALE points first, so their ``p0000``--``p0031`` identities remain
comparable, then deterministically expands to 256 unique prior-valid points.
The bank-64 workload filters every template and compares all semantic triggers;
only four stable representative templates retain full dense SNR series, which
bounds evidence size without reducing the computation under measurement.

The campaign also evaluates those same 256 points with one retained-grid
``SingleTemplate.batch_loglr`` invocation.  This is a distinct throughput
workload, while ``inference-grid256`` retains sequential public-model latency
representative of a scalar sampler.  The batch workload runs only current-source
B/C/D cells and compares B-C, C-D, and B-D: the frozen original A source lacks
the batch API, so there is deliberately no misleading A-to-current comparison
against a sequential emulation.  Capability detection fails closed if a listed
current cell no longer provides the API.  Its compute phase includes the small
parameter transfer performed by the API; the retained result stays on the
Torch device until the separately timed result-materialization phase.

The checked-in campaign is an exploratory single-session profile with one
cold sample, one unmeasured persistent warm-up, and three measured persistent
samples per cell.  It plans 95 attempts and 76 comparisons.  On the benchmark host, allow
roughly 45--90 minutes and 2--4 GiB for the first pilot; use the pilot's sealed
timings and byte counts to set the actual reservation.  A publishable session
uses zero cold samples, ten warm-ups, and thirty measured paired samples.  At
the same sizes that is approximately eight times the exploratory evidence
(roughly 6--12 hours and 12--24 GiB); it must be run in at least three separate
exclusive-host sessions rather than interpreted as 90 independent samples
from one thermal state.

``suite_cpu_scaling.json`` is the CPU-only companion campaign for the CPU benchmark host.  It
holds the science inputs from the performance campaign fixed and evaluates
conditioning, Q-transform, bank-64 search, and sequential grid-256 inference
at 1, 16, and 64 threads.  Each point runs A (original CPU), B (current CPU),
and C (Torch CPU), with A-B and B-C parity/performance comparisons.  D is
required by the suite schema but is deliberately absent from every workload;
CUDA measurements remain a separate lane so GPU resource use cannot be
mistaken for a CPU scaling point.

Each scaling command opts into a strict thread contract.  The active
``CPUScheme`` count and OpenMP setting must match ``--threads``; OpenMP, BLAS,
NumExpr, and Torch intra-op requests must all agree; dynamic OpenMP/MKL sizing
is disabled; and Torch inter-op remains one to avoid nested oversubscription.
Torch's observed pools are verified by the timing helper.  The artifacts also
record process affinity, native thread count, and the active thread
environment.  These checks establish requested software controls, not host
exclusivity or an absence of competing processes.

The checked-in profile has no cold samples, one unmeasured persistent warm-up,
and three measured requests.  It plans 144 attempts and 72 comparisons.  The
within-domain workload order is counterbalanced so the same thread count is
not always first.  Treat the first run as exploratory and reserve up to six
hours and 12 GiB; use its sealed elapsed time and bytes for later reservations.
The report directly summarizes A-B and B-C at each thread count.  A scaling
curve additionally groups the synchronized compute and poll-free operation
metrics by common domain prefix and cell, normalizing each cell to its own
``t01`` point; do not compare unlike domains or external request latency as if
they were kernel scaling.

``suite_cpu_refinement.json`` fills the intermediate 4-, 8-, and 32-thread
points for the same four largest representative workloads.  It preserves the
A/B/C cells, A-B and B-C science gates, persistent timing, and strict thread
attestation from ``suite_cpu_scaling.json``.  Its independent 0/1/3 profile
also plans 144 attempts and 72 comparisons.  Use it to locate a likely CPU
optimum after the topology sweep; do not treat its samples as additional
repetitions unless revision, dependency, affinity, NUMA, and host controls are
identical and the separately sealed sessions are reported.

``suite_cpu_batch_scaling.json`` is the strict CPU-only companion for the
retained-grid inference throughput path.  It compares current legacy CPU B
with Torch CPU C at batches of 8, 32, 256, and 1024 points, each with 1, 16,
and 64 threads.  A is excluded because the frozen source has no
``SingleTemplate.batch_loglr`` API, and D remains in the independent CUDA
crossover lane.  Every point reuses the crossover campaign's deterministic
parameter identities, inference comparator, persistent timing contract, and
strict thread controls.  The counterbalanced checked-in 0/1/3 profile plans
96 attempts and 36 B-C comparisons; it measures Torch-CPU implementation and
thread scaling only, not GPU speedup.

``suite_cpu_batch_refinement.json`` fills the intermediate 4-, 8-, and
32-thread points for those same retained-grid batches.  It preserves the B/C
cells, B-C science gate, persistent timing, deterministic point identities,
and strict thread attestation from ``suite_cpu_batch_scaling.json``.  Its
independent counterbalanced 0/1/3 profile also plans 96 attempts and 36
comparisons.  Use it to locate the batch-specific CPU optimum; keep its sealed
session separate from the 1/16/64 topology sweep.

``suite_crossover.json`` sweeps workload size to show where Torch CPU and CUDA
become faster than the current legacy CPU path instead of reporting only one
large point.  It covers 16, 64, and 256 seconds of conditioning; Q-planes with
approximately 0.23, 0.91, and 3.78 million pixels; 3, 16, and 64-template
two-detector searches; and sequential and retained-grid likelihood evaluations
at 8, 32, 256, and 1024 points.  The original A cell participates wherever its
public API supports the workload; retained-grid batches compare current-source
B/C/D only.  All points use one persistent warm-up and three measured warm
requests, so the campaign measures steady-state request and synchronized
compute time without conflating process startup.  It plans 256 attempts and
192 science comparisons.  Interpret a crossover only when all associated
parity comparisons pass and the timing distributions are separated; otherwise
report the point as inconclusive.

Benchmark host preflight
------------------------

The benchmark runner writes a sealed host-preflight JSON file before starting the
runner and includes it in the run's sealed evidence inputs.  In ``auto`` mode,
the checked-in performance, CPU-scaling, CPU-refinement, CPU batch-scaling,
CPU batch-refinement, crossover, preplanned-search, and persistent-inference
suite names select fail-closed ``strict`` mode; other rollout suites receive
the same snapshot in diagnostic mode.  A derived performance suite retains
its checked-in name and therefore also remains strict.  Set
``BENCHMARK_PREFLIGHT_MODE=strict`` explicitly for a custom benchmark suite.

Strict mode requires ``BENCHMARK_EXPECTED_CPU_IDS`` to exactly match the
runner's inherited Linux affinity.  The default gates require the one-minute
host load to be at most 0.10 per allowed logical CPU.  Suites requesting the
canonical CUDA cell D additionally require every detected GPU to have at least
90 percent free memory and no ``nvidia-smi`` compute process; CPU-only scaling
suites record GPU telemetry without gating on an unused device.
The evidence records the full expected and observed CPU sets, all three load
averages, GPU total/free/used bytes, and every compute process.  Thresholds may
be set before launch with ``BENCHMARK_MAX_LOAD_PER_CPU`` and
``BENCHMARK_MIN_GPU_FREE_FRACTION``.  A narrowly reviewed helper can be allowed
by process-name regex with ``BENCHMARK_ALLOWED_GPU_PROCESS_PATTERN``; its
memory still counts against the free-memory gate.

For troubleshooting only, ``BENCHMARK_PREFLIGHT_MODE=diagnostic`` is the
explicit opt-out.  It preserves a sealed invalid snapshot and runs the suite,
but that run is not controlled evidence for performance claims.  Invoking
``python -m tools.torch_workflows run`` directly does not perform this external
host preflight.

The comparator library also contains reusable semantic comparators for
trigger HDF files, likelihood grids and posterior distributions, template-bank
coverage and fitting factors, and normalized workflow graphs.  Event and DAG
row order, generated identifiers, absolute installation roots, timestamps,
and host names are not treated as science.  Detector, template, time,
statistics, bank coverage, edges, and scientifically relevant arguments are.

Statuses and acceptance
=======================

Every attempt and comparison has one canonical status:

``PASS``
   Deterministic science criteria pass.
``PASS_STATISTICAL``
   Distributional criteria pass; exact sample ordering is not required.
``SCIENTIFIC_FAIL``
   The program ran but its scientific result is outside policy.
``PERFORMANCE_REGRESSION``
   Science passes but an explicitly configured performance budget does not.
``UNSUPPORTED``
   The host or program lacks a declared capability.
``INFRA_ERROR``
   Inputs, dependencies, launch, timeout, or output evidence are incomplete.

A speedup is credited only when the corresponding scientific comparison is
``PASS`` or ``PASS_STATISTICAL``.  Raw timings remain in the JSON report after
a failure for diagnosis, but the reported speedup is null.  Missing outputs
and attempted LALSimulation imports can never be converted into a science
pass.

The Q-transform promotes single-precision real input on-device before the
forward FFT on every double-capable Torch device.  This matches or exceeds the
legacy calculation precision and prevents CPU/CUDA FFT differences from being
amplified when low-energy tiles are normalized by their median.  The SCALE gate
therefore retains its strict ``rtol=7e-5``, global ``atol=1e-8``, and
whole-plane relative-L2 limit of ``1e-6`` without an array-specific tolerance.

The L2 search comparator requires one-to-one trigger recovery within one sample
at 256 Hz.  The SNR and chi-square tolerances are fixed from an independently
repeated CPU-versus-Torch calibration: SNR uses ``atol=1e-4, rtol=3e-5`` and
chi-square uses ``atol=3e-3, rtol=3e-4``.  These cover the observed accumulated
floating-point differences without allowing lost, added, or time-shifted
events.  The L2 inference smoke comparator checks standardized effect size,
normalized Wasserstein distance, and credible-interval overlap between its
deterministic dummy-sampler products.  These checks validate output parity,
not posterior convergence.

Benchmark contract
==================

The suite records a cold fresh-process run, an unmeasured warm-up, and repeated
hot OS/cache runs.  External wall measurements are fresh-process end-to-end
latency, including interpreter startup and workload setup.  Other external
measurements include user/system CPU, an RSS upper bound, page faults, and
context switches.  Workloads synchronize Torch devices around their measured
compute regions and report throughput plus CUDA peak allocated/reserved memory
through a status sidecar.

The default conformance-suite warm repetitions are still separate processes.
They measure hot filesystem and system caches, not persistent in-process FFT
plans or allocator pools.  Every workload in ``suite_benchmark.json`` instead
provides a dedicated ``warm_command`` that reuses one worker process, performs
an unmeasured request, and then serves the repeated measured requests.  The
shared worker seals a poll-free operation timer separately from the runner's
external response latency and from the workload's synchronized compute region.
The benchmark profile also configures Torch intra-op and inter-op thread counts
inside each Torch worker and records the verified values, child affinity, and
native thread count in the workload metrics.

The fair CPU/GPU resource profiles, persistent timing boundaries, statistical
claim rules, whole-codebase coverage matrix, and reality-like workload rollout
are specified in :doc:`torch_performance`.  The conformance rollout's current
fresh-process ratios must not be presented as steady-state acceleration.

Evidence layout
===============

Setup creates two read-only root-level provenance files.  ``dependencies.json``
fingerprints the matched environments and host runtime.  ``deployment.json``
seals its SHA256 together with the exact original/current Git and import
revisions and an inventory (path, size, and SHA256) of every ignored regular
file in both source trees except explicitly harmless caches.  The exclusions
are ``.hypothesis``, ``.mypy_cache``, ``.nox``, ``.pytest_cache``,
``.ruff_cache``, ``.tox``, and ``__pycache__`` directories, the ``.coverage``
file, and ``.pyc``/``.pyo`` files.  Before every matrix or workflow launch, the
launch scripts recompute the deployment seal, dependency hash, source/import
identities, and the same ignored-file inventory.  They also re-probe the active
Python, Torch/CUDA/device/driver state and installed-package fingerprint,
compare material dependency state with setup, and write sealed launch evidence.
They remove every ambient exported ``PYCBC_*`` variable and ``PYTHONPATH``
before applying the cell's declared environment.

The runner never overwrites a non-empty run directory.  A run contains:

* ``run.json`` with the normalized plan, host, sources, revisions, and policy;
* ``inputs/manifest.json`` with SHA256 hashes of all declared inputs;
* one directory per workload/cell/phase/repetition, including command logs,
  backend trace, status, resource metrics, artifacts, and output hashes;
* semantic comparison payloads for A-B, B-C, C-D, and B-D;
* ``summary.json`` and ``report.md``.

Both root-level provenance files and the launch evidence are sealed inputs to
each workflow run and copied into L0.  The rollout regenerates each workflow
``summary.json`` from sealed run evidence before aggregation and verifies the
summary content seal.  A stage passes only when its process exits zero, its
evidence is valid, its overall scientific status passes, and every planned
comparison passes.  The rollout summary reports the separate L0, L1, SCALE,
L2, and L3 totals; exact commit bindings; valid evidence stages; scientific
statuses; benchmark eligibility/credited phases; and recursive backend
attestation totals.

Local planning and focused runs
===============================

Set the two source and interpreter pairs, then validate the complete plan:

.. code-block:: bash

   export ORIGINAL_SOURCE=/path/to/frozen/pycbc
   export ORIGINAL_PYTHON=/path/to/original/bin/python
   export CURRENT_SOURCE=/path/to/current/pycbc
   export CURRENT_PYTHON=/path/to/current/bin/python
   "$CURRENT_PYTHON" -m tools.torch_workflows plan \
       tools/torch_workflows/suite.json --run-dir /tmp/planned-run

A quick CPU implementation check can select a subset without changing the
versioned suite:

.. code-block:: bash

   "$CURRENT_PYTHON" -m tools.torch_workflows run \
       tools/torch_workflows/suite.json --run-dir /tmp/torch-smoke \
       --cell B --cell C --workload conditioning --workload search

Plan or run the realistic exploratory campaign in a new directory:

.. code-block:: bash

   "$CURRENT_PYTHON" -m tools.torch_workflows plan \
       tools/torch_workflows/suite_performance.json \
       --run-dir /tmp/torch-performance-plan
   "$CURRENT_PYTHON" -m tools.torch_workflows run \
       tools/torch_workflows/suite_performance.json \
       --run-dir /tmp/torch-performance-session-1

Plan the workload-size crossover campaign in the same way:

.. code-block:: bash

   "$CURRENT_PYTHON" -m tools.torch_workflows plan \
       tools/torch_workflows/suite_crossover.json \
       --run-dir /tmp/torch-crossover-plan

For a controlled publishable session, derive only the repetition counts while
keeping workload IDs, commands, inputs, cells, and comparison pairs unchanged.
Write the derived file beside the versioned suite so ``{suite_dir}`` continues
to resolve correctly; the runner seals that exact derived suite as evidence:

.. code-block:: bash

   jq '(.workloads[].repetitions) = \
       {"cold": 0, "warmup": 10, "warm": 30}' \
       tools/torch_workflows/suite_performance.json \
       > tools/torch_workflows/.suite_performance.publishable.json
   "$CURRENT_PYTHON" -m tools.torch_workflows run \
       tools/torch_workflows/.suite_performance.publishable.json \
       --run-dir /data/torch-performance/session-1

Preserve the derived suite with the result bundle and repeat with distinct run
directories in three controlled sessions.  As a fresh-process real-program
companion, run the existing ``pycbc-inspiral`` L2 workload in each session;
it is a deterministic CLI vertical slice, not a bank-throughput benchmark:

.. code-block:: bash

   "$CURRENT_PYTHON" -m tools.torch_workflows run \
       tools/torch_workflows/suite_l2.json \
       --run-dir /data/torch-performance/session-1-cli \
       --workload pycbc-inspiral

Run the CPU scaling pilot only on an otherwise idle host.  On Linux, restrict
the parent runner to one logical sibling from each physical core; every child
inherits that affinity and seals it in its metrics.  Review the generated list
before launch, and use a new result directory:

.. code-block:: bash

   scaling_cpu_set=$(lscpu -p=CPU,CORE,SOCKET,ONLINE | \
       awk -F, '!/^#/ && $4 == "Y" {k=$2 ":" $3; if (!seen[k]++) a[n++]=$1} \
       END {for (i=0; i<n; i++) printf "%s%s", (i ? "," : ""), a[i]}')
   test -n "$scaling_cpu_set"
   taskset -c "$scaling_cpu_set" \
       "$CURRENT_PYTHON" -m tools.torch_workflows plan \
       tools/torch_workflows/suite_cpu_scaling.json \
       --run-dir /data/torch-cpu-scaling/plan
   taskset -c "$scaling_cpu_set" \
       "$CURRENT_PYTHON" -m tools.torch_workflows run \
       tools/torch_workflows/suite_cpu_scaling.json \
       --run-dir /data/torch-cpu-scaling/session-1

Run ``suite_cpu_refinement.json`` with the same reviewed affinity and NUMA
policy in a separate result directory after the topology sweep.  It must not
run concurrently with the scaling campaign or another host benchmark.

On the benchmark host, launch the run through the runner script after the immutable parity
setup and pass that same reviewed list to its gate:

.. code-block:: bash

   RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
   BENCHMARK_EXPECTED_CPU_IDS="$scaling_cpu_set" \
       BENCHMARK_PREFLIGHT_MODE=strict \
       SUITE="$CURRENT_SOURCE/tools/torch_workflows/suite_cpu_scaling.json" \
       RUN_ID="$RUN_ID" taskset -c "$scaling_cpu_set" \
       "$CURRENT_SOURCE/tools/torch_workflows/run_benchmark.sh"

For the refinement pass, change only ``SUITE`` to
``suite_cpu_refinement.json`` and use a fresh ``RUN_ID`` and results root.  Its
checked-in name keeps automatic host preflight fail-closed.

For retained-grid batch refinement, use
``suite_cpu_batch_refinement.json`` in another fresh results root under the
same reviewed affinity and NUMA policy.  Do not run it concurrently with the
batch topology sweep or any other host benchmark.

On a multi-socket host, the selected core set spans sockets.  Record the NUMA
policy and keep it identical for every session.  If ``lscpu`` or ``taskset`` is
unavailable, stop rather than silently running a differently controlled
campaign.  For publishable claims, run at least three exclusive sessions with
rotated session order and increase warm-up/measured repetitions only in a
preserved, sealed derived suite.

Run on Remote Benchmark Host
=============================

Use the shared parity setup to create immutable original/current worktrees and
matched virtual environments under a brand-new root.  The root must not exist;
a failed or completed setup is never resumed, so choose another unique root.
Then launch the complete L0, L1, SCALE, L2, and L3 rollout as a detached,
dedicated process group:

.. code-block:: bash

   CURRENT_COMMIT=$(git rev-parse HEAD)
   RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
   ROOT_NAME="repos/pycbc-workflows-${CURRENT_COMMIT:0:12}-${RUN_ID}"
   ssh <remote-host> "PYCBC_PARITY_ROOT=\$HOME/$ROOT_NAME \
       CURRENT_COMMIT=$CURRENT_COMMIT bash -s" \
       < tools/torch_parity/run_matrix.sh
   ssh <remote-host> "root=\$HOME/$ROOT_NAME; run_id=$RUN_ID; \
       log=\$root/launcher-\$run_id.log; cd \$root/current; \
       command=\$root/current/tools/torch_workflows/run_rollout.sh; \
       nohup setsid env PYCBC_WORKFLOW_ROOT=\$root RUN_ID=\$run_id \
       \$command >\$log 2>&1 </dev/null & pid=\$!; \
       echo host=<remote-host>; echo cwd=\$root/current; echo command=\$command; \
       echo pid=\$pid; echo log=\$log; \
       echo 'next_check=tail the log in 10 minutes'; \
       echo stop_command=kill -TERM -- -\$pid"

The detached launch reports the host, working directory, command, process-group
PID, log, next check, and exact stop command.  Inspect it later with
``ssh <remote-host> 'tail -n 100 <reported-log>'``; stop the entire rollout and active
stage with the reported negative-PID ``kill`` command.

``run_rollout.sh`` attempts all five stages even when an earlier stage
fails, subject to the overall wall-clock budget, then writes one aggregate JSON
summary and Markdown report under a new timestamped ``rollout-results``
directory.  SCALE runs after L1 and retains its own status, evidence, timing,
and disk totals.  The L1, SCALE, L2, and L3 child runner detects CUDA, prints
the exact source revisions, and preserves per-attempt logs, attestations,
artifacts, hashes, and resource samples.  The entry points reject dirty source
worktrees and any drift in sealed dependency, deployment, import, runtime, or
ignored-file evidence.  To run only one workflow profile, invoke the runner script
with ``SUITE`` set to the desired JSON file.  Performance profiles also require
the controlled host-preflight settings described above.  L3 records absent
distributed scheduler capability as ``UNSUPPORTED`` inside its capability
evidence.  The
offline-only attempt is ``PASS`` only after a valid abstract graph and
non-empty concrete Pegasus planning evidence are produced, and its normalized
graph comparison must also pass; scheduler submission is never attempted.

The default overall wall-clock budget is 2,100 seconds.  Stage defaults are
controlled by ``L0_TIMEOUT_SECONDS``, ``L1_TIMEOUT_SECONDS``,
``SCALE_TIMEOUT_SECONDS``, ``L2_TIMEOUT_SECONDS``, and
``L3_TIMEOUT_SECONDS``; L1 defaults to 600 seconds and SCALE defaults to 1,200
seconds.  Disk limits use the corresponding ``*_DISK_BUDGET_BYTES`` variables.
The overall default is 8 GiB and SCALE is capped one byte below 5 GiB.  The
launcher checks free space before starting, requires a checked GNU ``timeout``
executable, monitors live usage, and records actual elapsed seconds and bytes.
A timeout, budget breach, missing/inconsistent evidence, or aggregation error
produces a terminal ``INFRA_ERROR`` marker and a nonzero exit.

Regenerate a report from sealed evidence with:

.. code-block:: bash

   "$CURRENT_PYTHON" -m tools.torch_workflows report <run-directory>
