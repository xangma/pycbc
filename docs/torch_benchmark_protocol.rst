.. _torch-benchmark-protocol:

Controlled executable benchmark protocol
========================================

Use this protocol for the corrected-CPU versus restacked-Torch comparison in
:ref:`torch-performance`. Keep the original-versus-corrected CPU arithmetic
and output changes in a separate experiment. A result is eligible for a
sustained-capacity claim only after workload convergence, scientific
qualification and host-load checks
have passed. Missing evidence remains explicit; a plausible profile or the
expected templates/core rate is not an acceptance test.

Reference and scientific scope
------------------------------

#. Prepare clean, pinned corrected-CPU and restacked-main checkouts using
   the same environment.
   Record native build provenance, executable and input hashes, dependency
   versions and the complete command. Use ``cpu:1``, MKL FFTs, compressed
   low-mass waveforms and one numerical-library thread for the normal reference.
   Preserve measured source pins when publishing later formatting or
   documentation changes. Record the final publication head and verify the
   exact changed-file set, full module AST equality for formatting, and byte
   identity of executable, test, tool, CI and native files separately.
#. Declare how the CPU reference geometry was selected before measuring Torch.
   This campaign retains the geometry selected on historical unchanged CPU;
   it does not claim a fresh optimum for corrected CPU. For a new tuning
   experiment, sweep segment length and safe start/end padding, holding bank,
   PSD, vetoes and unique output interval
   fixed. Record repeated unprofiled wall times and a selection rule before
   running candidates. Check longest-waveform duration, inverse-spectrum
   support, completed segments, boundary injections and unique search time.
   Duration bounds alone do not prove boundary correctness. A conservative
   fixed end pad is a declared constraint, not an end-padding optimum.
#. Freeze the selected geometry for matched backend comparisons. Include
   corrected CPU, restacked normal CPU, Torch CPU and Torch CUDA separately.
   Re-tuning each backend is a different experiment and needs its own table.
   Compare trigger identities, SNR, phase and chi-square under unchanged
   tolerances. Compare full PSD arrays, conditioned strain and segment
   geometry as well as the used PSD slice. Preserve the separate historical
   failures against unchanged CPU; a corrected reference does not erase them.
   Report each gate independently. Trigger parity alone does not establish
   full scientific equivalence or qualify an equivalent-output speedup.

The current campaign's initial strict controller stopped on a full-PSD
failure for Torch below 30 Hz. Its explicitly recorded policy amendment,
made after qualification, permits descriptive timings after all five
trigger comparisons, matching conditioned-strain digests and geometry, and
exact used PSD bins pass. Preserve both the initial stop and amendment; do not present the
continuation as the original policy or reclassify full-array failures.
Results under that amendment are finite-workload execution costs only.

Separate live-filter API measurements
-------------------------------------

Prepared live-filter calls have their own input construction, timing boundary
and independent numerical oracle. Apply the :ref:`live-filter method
<torch-batch-numerics>` separately; API rates do not measure full executable
performance or replace the baseline/proposed comparison.

.. toctree::
   :maxdepth: 1

   torch_batch_numerics

Workload convergence and timers
-------------------------------

Increase the number of distinct compressed templates or the unique valid
detector interval geometrically. Keep the mass/spin distribution, population
weights, sample rate, cutoff and scientific options comparable. Do not duplicate
template identities to inflate work. For every size, retain at least three
fresh unprofiled processes, rotate backend order and report medians and observed
ranges. Qualify compression without generation fallback and completed work.

Predeclare a convergence criterion; for example, require less than 5% change
in median capacity at two successive workload doublings. In every repeat at
each of the final three sizes, setup plus time outside the internal timer must
also remain below 10% of full wall time. The criterion is specific to the tested
backend, bank distribution and host; CPU-reference convergence alone does not
establish CUDA convergence. If it fails, show the scaling curve and report
finite-workload capacity.

Record both full executable wall time (imports through completed HDF output)
and a separately observed template-loop interval. The latter begins at the
first ``FilterBank`` lookup and ends after final event consolidation, before
performance metadata and output writing. It includes decompression,
normalization, filtering, vetoes and event handling. Synchronize CUDA at both
boundaries. An instrumented interval is supplementary and its enclosing wall
time must not replace the fresh unprofiled wall-time samples.

For one worker, completed work is the sum over templates of their **unique
valid detector seconds**. For this benchmark's common interval it is
``templates * valid_seconds``. Divide work by full wall seconds and allocated
physical cores to obtain templates/core at real time. Setup/analysis timer
fractions inside PyCBC exclude some process startup and output costs; show
that residual separately, without naming it Python overhead.

Host conditions and full-machine throughput
-------------------------------------------

Record CPU model, sockets, NUMA nodes, physical cores, SMT siblings, process
affinity, frequency governor, available memory and numerical-library thread
limits. Collect host and per-CPU utilization and process summaries before,
during and after each run. An affinity mask neither reserves a core nor keeps
its sibling idle. Observed low utilization alone does not prove exclusivity;
record the actual reservation or exclusive window separately.

Run these distinct CPU experiments using identical per-worker commands:

* One single-thread worker on an otherwise idle, reserved host.
* N simultaneous single-thread workers, where N is the number of physical
  cores available on the reserved host, with one logical CPU per physical core.
  The recorded ``len`` topology has 64 physical cores and 128 logical CPUs;
  use 64 workers only when all those physical cores are available and reserved.

Use a common start barrier, separate output directories and repeated runs.
Report per-worker timings, the distribution across workers, concurrent
makespan, completed work and allocated physical cores. Aggregate capacity is
``sum(completed template-seconds) / (makespan * allocated physical cores)``.
Do not sum already-normalized per-worker capacities or count SMT threads as
physical cores. N threads in one process is a separate scaling experiment.
CUDA needs an explicit GPU count and sharing policy; this CPU saturation
experiment does not establish multi-process GPU capacity.

The repository's Linux-only ``tools/benchmark_cpu_campaign.py`` implements
physical-core selection, thread limits, synchronized workers, load records and this aggregate
denominator. Supply a JSON argv array, with ``{output_dir}`` in the executable's
output filename, and run from a clean, built checkout. Workers run from their
own output directories, so use absolute executable and input paths in that
argv array. The command should invoke the executable directly, without a second
affinity or profiling wrapper.

.. code-block:: console

   python tools/benchmark_cpu_campaign.py --command-json command.json \
     --output idle-run --workers 1 --repeats 3 \
     --templates 384 --valid-seconds 1904
   python tools/benchmark_cpu_campaign.py --command-json command.json \
     --output full-machine-run --workers physical --repeats 3 \
     --templates 384 --valid-seconds 1904

Host-load checks must pass for controlled results. Explicit shared-host runs
are useful diagnostics but must retain that label. Never pause unrelated
services merely to satisfy the benchmark. Validate HDF output intervals and
template completion independently before accepting the runner's configured
work counts as scientific throughput.

The runner records observed idle conditions; it does not reserve the host.
``--reservation-note`` records an existing reservation as a user assertion,
and ``--shared-host`` explicitly permits a diagnostic despite failed idle
checks. Neither option establishes exclusive access.

Profile and presentation requirements
-------------------------------------

Collect full-process and template-loop cProfile and native ``perf`` in separate
processes after fixing the reference. Exclude all profiled runs from wall-time
medians. Check their outputs against unprofiled outputs. Keep cProfile exclusive
seconds, native sampled cycles and CUDA device-event times in separate panels
with their own denominators; cumulative call time is not additive. Report FFT
execution and planning separately, plus correlation, chi-square, thresholding,
decompression and the measured remainder. FFT dominance is a hypothesis to
check, not a shape to force onto results.

Publish the corrected-baseline and restacked-revision results together for each
defined workload, with their source revisions, timing boundaries, resource
counts and correctness verdicts. Publish the separate cost of CPU precision
corrections with its intentional arithmetic/output changes. Label earlier
unchanged-baseline comparisons as historical evidence. Keep intermediate
optimization experiments in the evidence archive, outside the main PR comparison. Retain
raw successful and failed receipts, source/input hashes, qualification,
profiles, load records and the renderer manifest in an immutable archive.
