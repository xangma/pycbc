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

Scientific search standards: precision, sample rate, and workload scope
-----------------------------------------------------------------------

Observational gravitational-wave inspiral searches (conducted in production by
``pycbc_inspiral`` throughout LIGO-Virgo-KAGRA observing runs O1–O4)
adhere to strict physics and signal-processing conventions:

#. **Single Precision Arithmetic**: Search matched filtering strictly operates in
   **single precision** (``complex64`` frequency- and time-series arrays; ``float32``
   for PSDs, variance norms, and statistics). Double precision (``complex128``) is not
   used in production offline searches as it imparts no measurable improvement to
   detection SNR, doubles memory footprint, and severely penalizes compute hardware
   (especially consumer and datacenter GPUs where single-precision compute throughput
   exceeds double precision by up to 64×). Benchmark comparisons must strictly adhere
   to single precision (``complex64``).
#. **Search Sample Rate**: For low-mass binary neutron star (BNS) and neutron star-black
   hole (NSBH) searches where gravitational-wave power is concentrated below 1000 Hz,
   PyCBC downsamples strain data to **2048 Hz** (Nyquist frequency 1024 Hz), yielding
   :math:`N = 1,048,576 = 2^{20}` samples per 512-second segment. When higher-frequency
   phenomena require 4096 Hz (:math:`N = 2,097,152 = 2^{21}`), single-precision
   ``complex64`` remains mandatory.
#. **Waveform Bank Decompression**: Production offline searches pre-generate compressed
   SVD banks. Waveform synthesis does not occur in the inner filtering loop; stored
   coefficients are decompressed via inline linear interpolation (< 1 ms per template once).
#. **Primitive Microbenchmarks vs Executable Campaigns**:

   * **Isolated Kernel Microbenchmarks**: Standalone function benchmarks (e.g. measuring
     batched cuFFT and correlation in :func:`~pycbc.filter.matchedfilter_jax.batch_matched_filter_bank`
     across batch sizes :math:`B=1\dots 1024`) quantify peak accelerator hardware scaling
     and VRAM saturation limits. They do not represent the full search pipeline.
   * **Executable Offline Campaign**: Complete, production-representative execution of
     ``pycbc_inspiral`` evaluating the full six-phase pipeline (frame reading,
     conditioning, Welch PSD estimation, bank decompression, matched filtering, power
     :math:`\chi^2` vetoes, peak clustering, and HDF5 output). This constitutes the primary
     scientific benchmark of record.
#. **Template Batching Protocol in Executable Searches (``--batch-size``)**:

   * **Motivation**: In single-template serial execution (:math:`B = 1`), GPU matched filtering is bound
     by host-device dispatch overhead and Python iteration latency (~60–75 ms per segment), severely
     underutilizing GPU execution units. Native template batching groups multiple templates into concurrent
     2D frequency-domain correlations, batched cuFFTs, and parallel peak screening.
   * **CLI Configuration**: The parameter ``--batch-size <B>`` (or ``--tile-size <B>``) controls the tile size
     processed concurrently per segment. Recommended defaults are :math:`B = 128` (or :math:`B = 64`) for CUDA accelerators,
     :math:`B = 16` for multithreaded CPU accelerators, and :math:`B = 1` for legacy single-threaded CPU backends.
   * **Scientific & Numerical Parity**: Batched matched filtering must produce identical triggers, SNR values,
     chisq statistics, and clustering time indices to unbatched execution within numerical precision tolerances.
   * **VRAM Budgeting & Saturation**: In single-precision complex64 at sample rate 4096 Hz (:math:`N = 2^{21}`),
     each template row occupies 8 MiB in frequency domain and 16 MiB for complex SNR time series. Working memory
     scales linearly with batch size: :math:`B = 16` requires ~1.0 GB, :math:`B = 64` requires ~4.1 GB, and
     :math:`B = 128` requires ~8.2 GB. Due to intermediate cuFFT plan scratch buffers, :math:`B = 128` is the
     recommended upper limit for 24 GB hardware.
   * **Decoupling Compilation & Operational Decompression**:
     To ensure timing measurements reflect true steady-state pipeline execution, JIT compilation overhead is decoupled
     from calculation time (``calc_time``) via persistent compilation caching (:file:`~/.cache/pycbc_jax_cache`) and a
     1-batch warmup pass during pipeline initialization (:math:`t_{\text{setup}}`). In contrast, waveform bank decompression
     runs on-the-fly in batches during the filtering loop (on GPU via vectorized linear interpolation, on CPU via Cython);
     it is realistically measured inside ``calc_time`` for both arms (for complete phase coverage and exclusion criteria
     across all timing metrics, see :ref:`jax-timing-boundaries`).
   * **Automated Multi-Arm Campaign Execution**:
     The campaign runner :file:`tools/bench_jax_inspiral_campaign.py` automates the protocol with counterbalanced
     process ordering, resource monitoring, and scientific trigger parity verification:

     .. code-block:: console

        # Execute batched offline inspiral reference campaign:
        python tools/bench_jax_inspiral_campaign.py \
          --original-source /path/to/pycbc-baseline \
          --branch-source . \
          --python /path/to/venv/bin/python \
          --frame-file docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
          --bank-file inputs/bank_384_compressed.hdf \
          --track track1 \
          --batched \
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

The repository's Linux-only ``tools/benchmark_cpu_campaign.py`` implements
physical-core selection, thread limits, synchronized workers, load records and this aggregate
denominator. Supply a JSON argv array, with ``{output_dir}`` in the executable's
output filename, and run from a clean, built checkout. Workers run from their
own output directories, so use absolute executable and input paths in that
argv array. The command should invoke the executable directly, without a second
affinity or profiling wrapper.

.. code-block:: console

   # Option (a): Single process on empty host
   python tools/benchmark_cpu_campaign.py --command-json command.json \
     --output idle-run --workers 1 --repeats 3 \
     --templates 384 --valid-seconds 1904

   # Option (b): N processes in parallel across all physical cores
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
seconds, native sampled cycles and CUDA/XLA device-event times in separate
panels with their own denominators; cumulative call time is not additive. Report
FFT execution and planning separately, plus correlation, chi-square,
thresholding, decompression and the measured remainder. FFT dominance is a
hypothesis to check, not a shape to force onto results. In a well-tuned reference,
core FFT filtering accounts for the majority of execution time (~50–65%), while
3–4 supporting kernels (waveform decompression, power chi-square vetoes,
auto-gated data conditioning, and peak clustering) each account for approximately
10% of runtime.

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
