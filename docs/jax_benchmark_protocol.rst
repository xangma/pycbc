.. _jax-benchmark-protocol:

Controlled executable benchmark protocol
========================================

Use this protocol to benchmark an unchanged CPU baseline against candidate
JAX revisions under controlled conditions. Check preservation of existing
CPU behavior separately from JAX agreement. A result is eligible for a
sustained-capacity claim only after workload convergence, scientific
qualification and host-load checks have passed. Missing evidence remains
explicit; a plausible profile or the expected templates/core rate is not an
acceptance test.

For the concrete offline inspiral search workloads (Track 1 for compressed
banks with linear decompression, and Track 2 for uncompressed banks with
on-device dynamic generation via ``diffgw``), frozen reference commits,
and scientific acceptance tolerances, see :ref:`jax-reference-campaign`.

Reference and scientific scope
------------------------------

#. Prepare clean checkouts of the unchanged CPU baseline and the candidate
   revision in the same environment. Check candidate normal CPU against
   the baseline before interpreting JAX results.
   Record native build provenance, executable and input hashes, dependency
   versions and the complete command. For the normal reference, enforce
   single-thread execution (``cpu:1``, explicit MKL FFTs, and numerical-library
   thread limits clamped via ``OMP_NUM_THREADS=1``).
   Preserve measured source pins when publishing later formatting or
   documentation changes. Record the final publication head and verify the
   exact changed-file set. For formatting-only differences, check complete
   module AST equality and byte identity outside declared formatting and
   documentation files. Any behavioral change requires new qualification.
#. Predeclare how the reference search geometry is determined before measuring
   candidate backends. When using an established benchmark workload, retain
   its declared reference geometry and input receipts (for example, the
   frozen 512/112/16-second configuration in :ref:`jax-reference-campaign`).
   For a new tuning experiment, systematically sweep segment length and
   safe start/end padding while holding the bank, PSD, vetoes and unique output
   interval fixed. Record repeated unprofiled wall times and an explicit
   selection rule before running candidates. Verify longest-waveform duration,
   inverse-spectrum support, completed segments, boundary injections and
   unique search time. Duration bounds alone do not prove boundary correctness.
   Any conservative fixed end pad is a declared constraint, not an
   unconstrained optimum.
#. Freeze the selected geometry across all matched backend comparisons. Include
   original CPU, candidate normal CPU, JAX CPU and JAX CUDA separately
   under identical geometry and inputs. Re-tuning each backend is a different
   experiment and requires its own reporting table.
   Compare trigger identities, SNR, phase and chi-square under unchanged
   tolerances. Compare full PSD arrays, conditioned strain and segment
   geometry as well as the used PSD slice. Compare the original baseline with all
   three candidate routes, plus candidate normal CPU with both JAX routes.
   Preserve failed comparisons with their original source pins. Do not change
   CPU arithmetic or relax tolerances to obtain agreement with an oracle.
   Compare scientific datasets and metadata. Exclude explicitly listed runtime
   timing fields from scientific verdicts and retain their raw values separately;
   do not exclude scientific configuration, sampling, epoch or geometry. Keep
   both original and corrected reports if the comparator included timing fields.
   Report each gate independently. Trigger parity alone does not establish
   full scientific equivalence or qualify an equivalent-output speedup.

Require all frozen scientific gates to pass before equivalent-output timing.
Stop on a failed gate and report it. Any separately authorized descriptive
measurement after failure must declare its policy, scope and retained failures;
it cannot establish equivalent-output speedup. Include host work and device
transfers inside the applicable timing boundary. Qualification processes are
not performance samples.

Precision, geometry and batching
--------------------------------

Record array dtypes, sample rate, transform length and valid output interval.
Use the same settings across a matched comparison. The compressed reference
campaign uses single-precision matched-filter arrays and 4096 Hz strain;
this choice does not establish that other search configurations must use the
same precision or sample rate. Keep double-precision experiments separate.

Compressed banks use stored waveform data and interpolation during the
search. The microbenchmark in :doc:`jax_performance` instead generates LAL
waveforms during every timed pass. Report the measured stages explicitly;
neither experiment can substitute for the other's throughput.

``--batch-size`` groups JAX templates for execution. Standard CPU remains
scalar. Start with a size known to fit the workload and sweep it under fixed
allocator settings. Template dimensions, FFT workspace, intermediate tensors,
allocator reservations and other device users all affect memory requirements;
a universal maximum batch size cannot be inferred from GPU capacity alone.
A tail batch can also introduce a different compiled shape.

Compilation and warmup in executable setup are included in full wall time
and excluded from the post-setup calculation interval. Record whether a
persistent compilation cache was present and whether warmup succeeded.
Bank decompression during filtering remains inside calculation time. See
:ref:`jax-timing-boundaries` for the exact timer scope.

The maintained campaign runner counterbalances process ordering and records
resource samples and a limited trigger comparison. Its built-in SNR/time/count
comparison does not replace the complete scientific gates above. Example:

.. code-block:: console

   python tools/bench_jax_inspiral_campaign.py \
     --original-source /path/to/pycbc-baseline \
     --branch-source . --python /path/to/venv/bin/python \
     --frame-file /path/to/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --bank-file /path/to/bank_384_compressed.hdf \
     --track track1 --batched --batch-size 128 \
     --output artifacts/benchmarks/receipt.json \
     --output-dir artifacts/benchmarks

Separate live-filter API measurements
-------------------------------------

Prepared live-filter calls have their own input construction, timing boundary
and independent numerical oracle. Apply the :ref:`live-filter method
<jax-batch-numerics>` separately; API rates do not measure full executable
performance or replace the baseline/proposed comparison.

.. toctree::
   :maxdepth: 1

   jax_batch_numerics

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
normalization, filtering, vetoes and event handling. Synchronize CUDA and JAX
device computations (via ``.block_until_ready()``) at both boundaries. An
instrumented interval is supplementary and its enclosing wall time must not
replace the fresh unprofiled wall-time samples.

For one worker, completed work is the sum over templates of their **unique
valid detector seconds**. For this benchmark's common interval it is
``templates * valid_seconds``. Divide work by full wall seconds and allocated
physical cores to obtain templates/core at real time. Setup/analysis timer
fractions inside PyCBC exclude some process startup and output costs; show
that residual separately, without naming it Python overhead (see the
comparative matrix in :ref:`jax-timing-boundaries`).

Host conditions and full-machine throughput
-------------------------------------------

Record CPU model, sockets, NUMA nodes, physical cores, SMT siblings, process
affinity, frequency governor, available memory and numerical-library thread
limits. Collect host and per-CPU utilization and process summaries before,
during and after each run. An affinity mask neither reserves a core nor keeps
its sibling idle. Observed low utilization alone does not prove exclusivity;
record the actual reservation or exclusive window separately.

Run these distinct CPU experiments using identical per-worker commands:

* **Option (a) Single-process isolation**: One single-thread worker on an otherwise
  empty, idle machine with zero competing processes to measure uncontended single-thread throughput.
* **Option (b) Full-machine saturation**: N simultaneous single-thread workers in parallel,
  where N is the number of physical cores on the machine, with each worker executing an identical
  instance of the test pinned to a distinct physical core.

Use a common start barrier, separate output directories and repeated runs.
Report per-worker timings, the distribution across workers, concurrent
makespan, completed work and allocated physical cores. Aggregate capacity is
``sum(completed template-seconds) / (makespan * allocated physical cores)``.
Do not sum already-normalized per-worker capacities or count SMT threads as
physical cores. N threads in one process is a separate scaling experiment.
CUDA needs an explicit GPU count and sharing policy; this CPU saturation
experiment does not establish multi-process GPU capacity.

Record the worker command, physical-core selection, thread limits,
synchronization, load measurements and reservation status with each campaign.
Host-load checks must pass for controlled results; explicitly shared-host
runs remain diagnostics. Validate HDF output intervals and template completion
before accepting configured work counts as scientific throughput. The former
one-off CPU saturation runner is available in Git history at ``jax-evidence-archive-20260919``.

Profile and presentation requirements
-------------------------------------

Collect full-process and template-loop cProfile and native ``perf`` in separate
processes after fixing the reference. Exclude all profiled runs from wall-time
medians. Check their outputs against unprofiled outputs. Keep cProfile exclusive
seconds, native sampled cycles and CUDA/XLA device-event times in separate
panels with their own denominators; cumulative call time is not additive. Report
FFT execution and planning separately, plus correlation, chi-square,
thresholding, decompression and the measured remainder. FFT dominance is a
hypothesis to check, not a shape to force onto results. Do not prescribe
expected percentages for the FFT or supporting stages before measuring them.

Decompose executable wall time into mutually exclusive phases (import/startup
overhead, data conditioning, waveform bank preparation, core matched filtering,
vetoes and clustering, and output serialization). Always separate unprofiled
performance measurements from profiling passes: timing samples, medians, and
speedup claims must derive solely from unprofiled runs, while profiling passes
provide kernel attribution percentages without inflating the benchmark clock.

Publish original-CPU and candidate-revision results together for each defined
workload, with source revisions, CPU-preservation evidence, timing boundaries,
resource counts and every correctness verdict. Keep intermediate optimization
experiments in the evidence archive. Retain raw successful and failed
receipts, source/input hashes, qualification, profiles, load records and the
renderer manifest in an immutable archive.
