.. _jax-benchmark-protocol:

JAX benchmark guide
===================

This guide defines the executable workloads, scientific gates and measurement
controls used for JAX comparisons. Current measurements and reproduction
commands are in :doc:`jax_performance`. The historical live-filter API fixture
is documented at the end; it is separate from both the executable benchmark
and the published live-sized filter estimates.

A sustained-capacity claim requires workload convergence, scientific
qualification and host-load checks. Missing evidence remains explicit;
an expected templates/core rate is not an acceptance test.

Reference and scientific scope
------------------------------

#. Declare the comparison scope. For current-stack qualification, compare
   standard CPU, JAX CPU and JAX CUDA from the same measured checkout.
   To establish preservation of existing CPU behavior across a source change,
   also prepare a clean unchanged CPU checkout in the same environment and
   compare candidate standard CPU against it. These are separate claims.
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
   standard CPU, JAX CPU and JAX CUDA under identical geometry and inputs,
   plus the original CPU arm for source-regression comparisons. Re-tuning each
   backend is a different experiment and requires its own reporting table.
   Compare trigger identities, SNR, phase and chi-square under unchanged
   tolerances. Compare full PSD arrays, conditioned strain and segment
   geometry as well as the used PSD slice. Compare current standard CPU with
   both JAX routes; when testing source regression, also compare the original
   baseline with all three candidate routes.
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
comparison does not replace the complete scientific gates above. See
:ref:`jax-campaign-commands` for the Track 1 and Track 2 commands.

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
physical cores to obtain CPU templates/core at real time. Report CUDA
capacity per allocated GPU together with its allocated host cores; a GPU is
not a CPU core. Setup/analysis timer
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

Publish all arms in the declared comparison scope together, with source
revisions, timing boundaries, resource counts and every correctness verdict.
Include original-CPU results and CPU-preservation evidence when testing
source regression. Keep intermediate optimization
experiments in the evidence archive. Retain raw successful and failed
receipts, source/input hashes, qualification, profiles, load records and the
renderer manifest in an immutable archive.

.. _jax-inspiral-reference:
.. _jax-reference-campaign:

Complete-executable offline benchmark definition (pycbc_inspiral)
------------------------------------------------------------------

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data across all pipeline stages:
data conditioning, PSD estimation, template bank preparation, normalization,
matched filtering, power chi-square, clustering, and trigger output.
It does not call ``LiveBatchMatchedFilter.process_data``.
For current-stack qualification, compare standard CPU, JAX CPU and JAX CUDA
from the same measured checkout, using identical inputs and scientific
settings. The original reference checkout
``40e94792b3edf59f39b18b65102b28a4f74433a7`` is an optional additional source
regression arm; the fresh measurements in :ref:`jax-performance` use the
current checkout's standard CPU reference.

The benchmark protocol defines two complementary tracks:

1. **Track 1: Compressed Bank Campaign (Inline Linear Decompression)**:
   Evaluates filtering, data conditioning, and linear decompression from stored
   waveform interpolation data without dynamic waveform synthesis overhead.
2. **Track 2: Dynamic Generation Campaign (On-Device Batch Generation with ``diffgw``)**:
   Evaluates dynamic on-the-fly waveform generation from physical parameters.
   Standard CPU generates through LALSimulation; the ``jax_cpu_diffgw`` and
   ``jax_cuda_diffgw`` arms use batched ``diffgw`` generation on their selected
   JAX device. The runner also supports JAX arms with LAL generation.

Track 1: Compressed bank settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bank contains **384 distinct templates: 256 BNS and 128 NSBH**.
Masses are detector-frame solar masses; spins are dimensionless aligned
components. The count ratio is a benchmark choice, not a population weight.

.. list-table:: Parameter bounds
   :header-rows: 1

   * - Region
     - Mass 1
     - Mass 2
     - Spin 1z
     - Spin 2z
   * - BNS
     - 1.2--2
     - 1.2--2
     - -0.05--0.05
     - -0.05--0.05
   * - NSBH
     - 3--8
     - 1.2--2
     - -0.5--0.5
     - -0.05--0.05

Templates use ``IMRPhenomD`` at 30 Hz and 4096 Hz sampling, with
``--use-compressed-waveforms --waveform-decompression-method inline_linear``.
This deterministic set is not a coverage-qualified search bank; it includes
neither tidal physics nor disruption. The frozen compressed bank SHA256 is
``26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f``.

The frame is ``H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf``, channel
``H1:LOSC-STRAIN``, SHA256
``580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865``.
The requested GPS input interval is 1187007048--1187009080. The unique valid
trigger interval is **1187007160--1187009064: 1904 seconds**.

.. list-table:: Fixed search configuration
   :header-rows: 1

   * - Setting
     - Value
   * - FFT geometry
     - 512-second segment; 112-second start pad; 16-second end pad;
       2097152 samples; five segments per template
   * - Strain conditioning
     - 25 Hz high-pass; 8-second data pad; autogating threshold 100,
       cluster 5 s, width/taper 0.25 s, pad 16 s, one iteration
   * - PSD
     - Median estimate; 126 segments of 32 s with 16 s stride;
       16-second inverse spectrum support; Hann truncation of inverse ASD
   * - Trigger selection
     - SNR threshold 5.5; reweighted SNR threshold 5;
       symmetric clustering with a 1-second window
   * - Vetoes
     - 16-bin power chi-square; sine-Gaussian chi-square is not requested
   * - Backend controls
     - Standard CPU ``cpu:1`` uses explicit MKL;
       JAX CPU ``jax:cpu`` and JAX CUDA ``jax:cuda:0`` use native JAX FFTs (with cuFFT on NVIDIA GPU)

Following the geometry selection methodology in :ref:`jax-benchmark-protocol`,
the configuration was selected on unchanged CPU ``40e94792b3`` by
sweeping 256/512/1024 second segments and 96/112 second start padding, with
16 second end padding across three fresh unprofiled processes per candidate.
The rule selected the lowest median, treating settings within 3% as tied, then
preferring more start padding and a smaller FFT. Retain this selected
512/112/16 second geometry for all matched backend comparisons.

Scientific qualification
~~~~~~~~~~~~~~~~~~~~~~~~

Verify that every arm decompresses the compressed bank without waveform
regeneration, completes all ``384 * 5 = 1920`` template/segment pairs, and
covers the same unique valid interval. Compare trigger identities before
comparing matched fields. Configuration and degrees of freedom must match.

The frozen executable comparator uses relative tolerance ``1e-4`` and absolute
tolerance ``1e-5``, sigmasq relative tolerance ``1e-5``, and circular phase
absolute tolerance ``1e-4`` radians. Compare complete PSD references as well as
SNR, phase, sigmasq and the available veto fields. The full-PSD gate uses
relative tolerance ``1e-4`` with zero absolute floor. Retain all failed verdicts.
Exclude runtime timing metadata fields: ``H1/search/filter_rate_per_core``,
``H1/search/run_time``, ``H1/search/setup_time_fraction`` and
``H1/search/templates_per_core``.

Commands and reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preserve measured commits, dependencies and input hashes. When including an
original-source comparison, prepare that CPU reference in a separate clean
checkout. Always identify which CPU arm is the scientific reference.

Execution arms
^^^^^^^^^^^^^^

Each campaign comparison evaluates arms under identical host isolation:

#. **Original Standard CPU (``original_cpu``)**: Baseline checkout ``40e94792b3``
   using ``--processing-scheme cpu:1`` with explicit MKL FFTs.
#. **Candidate Standard CPU (``branch_cpu``)**: Candidate branch checkout
   using ``--processing-scheme cpu:1`` with explicit MKL FFTs.
#. **Candidate JAX CPU (``jax_cpu``)**: Candidate branch checkout using
   ``--processing-scheme jax:cpu`` with JAX FFTs.
#. **Candidate JAX Batched CPU (``jax_cpu_batched``)**: Candidate branch checkout using
   ``--processing-scheme jax:cpu --batch-size 16``.
#. **Candidate JAX CUDA (``jax_cuda``)**: Candidate branch checkout using
   ``--processing-scheme jax:cuda:0`` on NVIDIA GPU with native JAX data
   conditioning and on-device linear spline decompression.
#. **Candidate JAX CUDA Batched (``jax_cuda_batched``)**: Candidate branch checkout using
   ``--processing-scheme jax:cuda:0 --batch-size 128`` with device batched filtering
   and on-device Power :math:`\chi^2` vetoes.

The ``--batched`` campaign preset keeps both standard CPU references scalar
and selects ``jax_cpu_batched`` and ``jax_cuda_batched``. Standard CPU
``--batch-size`` greater than one is not supported.

.. _jax-campaign-commands:

Campaign commands
^^^^^^^^^^^^^^^^^

The :ref:`Track 1 reproduction command <jax-executable-reproduction>` runs
standard CPU and batched JAX arms on the compressed fixture above. The runner
sets the geometry, scientific options, single-thread limits and per-scheme
FFT backend (MKL for standard CPU, JAX for JAX schemes). Preserve the full
commands and environment in each receipt.

For Track 2, supply an uncompressed parameter bank and select the DiffGW arms:

.. code-block:: console

   python tools/bench_jax_inspiral_campaign.py \
     --original-source . \
     --branch-source . --python /path/to/venv/bin/python \
     --frame-file /path/to/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --bank-file /path/to/bank_512_taylorf2.hdf \
     --track track2 --batch-size 64 --replicates 3 \
     --arms branch_cpu jax_cpu_diffgw jax_cuda_diffgw \
     --output artifacts/track2/receipt.json \
     --output-dir artifacts/track2

This preset uses TaylorF2 at order 7 without compressed waveforms. Both DiffGW
arms use the requested batch size; standard CPU remains scalar. Record the
parameter bank's hash, template distribution and row count separately from
the frozen compressed Track 1 bank. To include source regression, point
``--original-source`` at the pinned baseline and add ``original_cpu`` to
``--arms``. The example selects only current-checkout arms.

Timing and profiling boundaries are defined in :ref:`jax-timing-boundaries`.
Fresh unprofiled processes establish medians and ranges; separate profiling
passes attribute costs. Two aggregate timers alone cannot establish an
additive breakdown of all pipeline stages.

Measured campaign results
^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`jax-performance` reports the current measurements and their scientific
comparison scope. The 1536-row batch experiment repeats the 384-template
bank four times and must not be treated as a larger independent bank or as
evidence of sustained-capacity convergence.

.. _jax-batch-numerics:

Historical live-filter API fixture
----------------------------------

The one-off streaming benchmark harness is retained in Git history at
``jax-evidence-archive-20260919`` and is no longer among the maintained tools.
The following workload and numerical gates describe that historical experiment.
No timing receipts for this API are currently published; the
:ref:`live-sized capacity estimates <jax-live-capacity>` time a smaller
synthetic filter operation.

The fixture evaluates
``pycbc.filter.matchedfilter.LiveBatchMatchedFilter.process_data`` with
synthetic frequency-domain inputs. It does not time the full ``pycbc_live``
executable. Use the same frozen fixture for the original CPU reference and
each candidate backend; record source revisions and qualify each combination
before collecting timings.

Fixed workload and timing
~~~~~~~~~~~~~~~~~~~~~~~~~

Every full worker uses 1024 deterministic ``complex64`` templates and three
strain blocks. Templates contain seeded Gaussian complex values from 30 to
800 Hz, with unequal powers to exercise group normalization. They are synthetic
arrays, not IMRPhenomD waveforms. The fixture uses a smooth analytic PSD and
complex frequency-domain noise with four coherent injections per block at
nominal SNRs 8, 9.5, 11 and 12.5, then divides strain by that PSD.

The FFT has 131072 samples at 2048 Hz (64 seconds; ``delta_f = 0.015625`` Hz).
Each block advances 56 seconds and searches indices 12288 through 126975.
This differs from the executable's sample rate, FFT length and real frame data.
The fixture hashes the bank, parameters, PSD, normalization and all strain
blocks; changing batch size does not change those inputs within a seed.

The API performs correlation, IFFT, normalization, peak selection and vetoes.
SNR threshold is 5.5. It evaluates 16-bin reduced power chi-square and
sine-Gaussian chi-square with two configured tiles per trigger; activation and
valid tile support are checked. Reweighted-SNR and SNR-abort thresholds are
disabled. This tests their implemented behavior on the fixture, not detection
sensitivity across an astrophysical population.

Execution batches are 1, 8, 32, 128, 512 and 1024 on standard CPU/MKL, JAX CPU
and JAX CUDA. Run workers serially on the same recorded host, with one
allocated host core and numerical thread; CUDA additionally uses one recorded
GPU. Rotate route and batch order between repeats. Each worker performs one cold iteration, two warmups and five timed iterations. One
iteration calls the prepared API for all three blocks, completing 3072
template-block evaluations. JAX device execution synchronizes via
``.block_until_ready()`` at both timing boundaries.
Input creation, PSD estimation, waveform/bank preparation, executable startup
and trigger validation are outside the clock. The reported cell rate is the
median of three fresh-worker median rates; ranges span those three medians.

Numerical standards and accuracy oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Higher precision reduces rounding error but does not guarantee bitwise agreement
between implementations. FFT algorithms, reduction order and device arithmetic
can change the result at either precision. Report exact comparisons separately
from tolerance-based comparisons and apply the declared criteria below to the
stored inputs and observed outputs.

Adopted complex-SNR criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fixture's normalized-SNR policy requires, for every block, template and
complex sample:

.. code-block:: text

   abs(actual_complex_SNR - oracle_complex_SNR) <= 0.001
   abs(actual_complex_SNR - MKL_complex_SNR) <= 0.001

Both comparisons are mandatory, with zero relative tolerance. The standard
CPU/MKL batch-1 reference must itself pass the independent oracle gate for
this API timing policy. If unchanged CPU fails, retain the failure and stop
qualification; do not change its arithmetic to make the oracle gate pass. The
``0.001`` budget extends the existing trigger-SNR tolerance to the full complex
time series.

The oracle promotes stored complex64 template and overwhitened strain inputs
before multiplication and applies an unnormalized NumPy complex128 inverse FFT.
Its normalization is computed independently from the bank and PSD in float64:

.. code-block:: text

   oracle_sigmasq = 4 * delta_f * sum((real64(bank)**2 + imag64(bank)**2) / PSD64)
   oracle_norm = 4 * delta_f / sqrt(oracle_sigmasq)
   oracle_complex_SNR = oracle_raw_output * oracle_norm

Actual and MKL outputs use their observed route normalizations. Qualification
records the norms and sigmasqs used by the default bulk path for every row.
Every observed sigmasq must agree with the independent value within relative
``1e-3``. Norms and sigmasqs must be finite and positive, and all complex samples
finite.

Existing trigger and veto comparisons remain unchanged: exact identities,
peak indices, end times and metadata; SNR absolute ``1e-3``; circular phase
absolute ``1e-3`` radians; sigmasq relative ``1e-3``; power chi-square relative
and absolute ``1e-4``; and sine-Gaussian chi-square relative and absolute
``3e-5``. Small complex error alone cannot guarantee unchanged threshold
decisions or peak selection near a tie, so the exact trigger gates still apply.
