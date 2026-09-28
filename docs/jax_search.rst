.. _jax-search:

JAX search and tuning
=====================

The JAX search path uses JAX numerical kernels for supported strain
conditioning, template preparation, matched filtering, vetoes, clustering and
event handling. Coverage is operation- and option-specific; Python
orchestration, file I/O, metadata, MPI transport and checkpoint serialization
remain host work, and array conversions can copy or synchronize.
See :doc:`jax` for installation, device selection and precision, and
:ref:`jax-numerical-differences` for numerical parity and qualification status.

.. _jax-filtering:

Filtering API
-------------

Use the usual filtering and veto functions inside a ``JAXScheme`` context.
For prepared frequency-domain ``template`` and ``data`` arrays and a
compatible one-sided ``psd``:

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.filter import matched_filter
   from pycbc.vetoes import power_chisq

   with JAXScheme("cuda:0"):
       snr = matched_filter(template, data, psd=psd,
                            low_frequency_cutoff=20.0)
       chisq = power_chisq(template, data, num_bins=16, psd=psd,
                          low_frequency_cutoff=20.0)

The matched filter correlates strain with the conjugated template, weighted
by the inverse PSD, and returns the normalized complex SNR time series.
The power chi-square veto divides the template into frequency bins with
equal expected power and tests the signal contribution in each bin.

Execution
---------

Add the scheme and batch size to an otherwise configured ``pycbc_inspiral``
command:

.. code-block:: console

   pycbc_inspiral \
       --processing-scheme jax:cuda:0 \
       --approximant TaylorF2 \
       --batch-size 128 \
       ...

``pycbc_inspiral`` prepares conditioned strain, frequency-domain segments and
PSDs with the selected JAX implementations. Its measured qualification scope
is recorded in :doc:`jax_numerical_differences`.

``pycbc_live`` retains
JAX arrays through supported conditioning, filtering and coincidence stages;
unsupported options and orchestration stay on the host or use their documented
fallbacks. A complete path requires fresh scientific qualification: changing
FFT libraries can affect rounding, PSDs and trigger selection.

Backend-specific kernels and event state live in separate ``*_jax.py`` modules.
Shared modules contain dispatch hooks, preserving the ordinary CPU calculation.
Bounded historical replay staging lives in ``frame/frame_jax.py``; conditioning
lives in ``filter/resample_jax.py`` and ``strain/strain_jax.py``;
filtering and vetoes live in ``filter/matchedfilter_jax.py`` and
``vetoes/*_jax.py``. ``events/eventmgr_jax.py`` and ``events/coinc_jax.py`` own
JAX event storage and coincidence processing. Both JAX CPU and JAX CUDA select
these modules when execution reaches a supported JAX entry point.
Current conditioning supports Kaiser FIR filters, LDAS decimation and direct
Butterworth high/low-pass filter conventions. The default JAX high-pass runs
the LAL sample recurrence through ``jax.lax.scan``. CUDA partially unrolls
that scan to amortize loop overhead without changing the recurrence or its
rounding order; JAX CPU keeps the rolled scan. The optional
``--jax-highpass-mode parallel`` composes that recurrence with a parallel prefix
scan. The parallel mode changes floating-point grouping, so tests compare
both modes against LAL in both precisions. Butterworth resampling raises
an explicit unsupported-operation error. JAX inspiral event handling currently
rejects multiprocessing, injection-window cuts and chirp-width loudest bins.

Point chi-square defaults to ``--jax-chisq-mode cpu-compatible`` in both
``pycbc_inspiral`` and ``pycbc_live``. On either JAX device this runs an ordered JAX
recurrence with the native CPU phase initialization, multiplication and
accumulation convention. Ordinary CPU scheme dispatch is unchanged.

Use ``--jax-chisq-mode direct-phase`` to select independently evaluated
Fourier phases. This can differ from the native CPU recurrence at complex64,
so it is an explicit numerical choice rather than a compatible performance
comparison. The API equivalent is
``JAXScheme("cuda:0", chisq_mode="direct-phase")``. Both modes support
complex64 and complex128 correlations; see :doc:`jax_numerical_differences` for
measurements and qualification limits. The switch applies to point-based
chi-square, including the batched triggered-point path; the full time-series
FFT chi-square calculation is unchanged.

JAX point chi-square on CPU or CUDA requires the default
``PYCBC_JAX_ENABLE_X64=1`` even for single-precision search arrays.
Compatibility mode uses double precision for phase initialization and
direct-phase uses 64-bit frequency/time products. Both retain the requested
correlation and accumulation precision. Disabling x64 raises an error.

Waveform preparation can use compressed banks or the selected waveform
generator. Direct waveform-bank APIs that use ``diffgw`` generation require
that optional dependency and explicit ``--enable-diffgw``; selecting JAX alone
does not enable it. Provider and model support, batching and differentiability
must be checked for the chosen waveform path. See
:ref:`jax-campaigns` for the currently maintained executable
configurations.

.. _jax-search-batching:
.. _jax-tiled-pathways:
.. _jax-optimizations:

Template batching and compilation
---------------------------------

A bank can contain more templates than fit on the device simultaneously.
``--batch-size`` selects bounded execution batches. In ``pycbc_inspiral``,
sizes greater than one require a JAX scheme; the executable rejects that
combination for ordinary CPU dispatch.
Choose the size for the template length, available memory and FFT workspace,
then validate scientific output and measure the complete search.

Selected JAX filtering and waveform kernels use ``jax.jit`` for compilation
and ``jax.vmap`` for batched evaluation. Compiled kernels can be reused for
compatible shapes; a final partial batch may require another compilation.
The benchmark suite gives each JAX arm its own persistent compilation cache.
Its qualification run populates every production shape, including the final
partial batch, before any timing run. Cache thresholds are zero so even small
executables are retained. Every timed and profiled process records JAX's cache
request and hit events; a missing receipt, an empty cache, or any uncached
compile request invalidates the run. Cache lookup and executable loading remain
inside the fresh-process wall time.

A larger fitting batch need not improve throughput; validate it with a current
complete-executable campaign under the policy in
:ref:`jax-benchmark-protocol`.

The search path uses deliberately chosen fusion boundaries rather than one
whole-pipeline custom kernel. Batched correlation, inverse FFT, magnitude,
thresholding and fixed-window clustering share one compiled boundary, while
the FFT remains a vendor-library operation. ``pycbc_live`` likewise performs
template normalization and peak acceptance once per batch; only compact,
irregular trigger metadata crosses back to Python.

Compressed templates are a different workload: short irregular inputs expand
into dense rows with reference interpolation and ordered chi-square scans.
They are staged with the native interpolator on the host, transferred once per
batch, and the retained host rows are reused for exact chi-square bin caches.
Final per-template event clustering also uses the native reference on compact
host records, with one deferred global concatenation. This avoids forcing
serial or shape-varying work into GPU kernels while preserving the standard
backend's arithmetic and edge choices.

Pallas/Mosaic GPU is therefore not part of this path. Profiling identified
irregular expansion, ordered scans, scalar synchronization and repeated event
concatenation as the limiting work, not an unfused dense tile pipeline. A
persistent custom kernel would also have to cross vendor FFT and variable
trigger boundaries. Revisit that choice only if profiling exposes a stable,
dense kernel chain whose launch or memory traffic dominates after compilation.

The resulting residency contract is deliberately precise. GWF decoding remains
an unavoidable LALFrame host operation. For a bounded JAX historical replay,
``frame_jax.py`` reads a contiguous span once, transfers it once, and performs
per-block extraction and rolling-buffer updates on the selected device. The
read-ahead window is bounded and a non-contiguous span falls back to the normal
streaming reader. After input staging,
dense conditioning, normalization, filtering, FFT, thresholding and supported
veto arithmetic stay as JAX arrays on the selected device across their fused
boundaries. Host/device transfers are restricted to the exact compressed-bank
staging described above and bounded trigger indices or masks needed by Python's
variable-length metadata and output code. File I/O, command orchestration and
output serialization cannot run on the GPU. Consequently, benchmark receipts
describe this as an end-to-end JAX *numerical* path, not a literally GPU-only
process. Moving those irregular reference operations onto the GPU would change
the arithmetic or restore the decompression regression, so it is not a valid
optimization under the accuracy requirement.

Optimization boundaries
-----------------------

The current execution boundaries follow the dependencies found by profiling:

.. list-table::
   :header-rows: 1
   :widths: 20 22 22 22

   * - Component
     - Serial work
     - Parallel and fused work
     - Custom-kernel decision
   * - Frame/configuration I/O, MPI and output
     - GWF decoding, host libraries and control flow
     - Bounded replay amortizes GWF decoding across many chunks, transfers each
       decoded span once, then slices and updates the raw buffer on device
     - Keep compression on the host; use the JAX read-ahead path rather than a
       custom decoder kernel
   * - LAL-compatible high-pass
     - Each sample depends on the preceding two states; sections and the
       forward/reverse passes are ordered
     - Independent streams may be batched; CUDA partially unrolls the exact
       scan without regrouping it
     - Do not replace with a prefix scan or Pallas kernel: the former changes
       rounding and an exact persistent-kernel probe was no faster
   * - LDAS FIR resampling
     - Only setup and shape decisions
     - FFT convolution and decimation are device-parallel; XLA fuses adjacent
       elementwise operations around the vendor FFT boundary
     - Keep the existing FFT implementation; its synchronized warm cost is
       negligible beside the exact high-pass
   * - Segmentation, PSD and whitening
     - Segment construction and pipeline decisions
     - Segments and frequency bins are batched on device; elementwise work is
       fused around vendor FFT calls
     - No stable dense residual kernel currently dominates
   * - Compressed-bank expansion
     - Reference interpolation and variable row boundaries
     - Expanded dense rows are transferred once per batch and reused
     - Keep exact host staging; device interpolation caused the historical
       decompression regression
   * - Matched filter and peak finding
     - Batch scheduling and compact survivor handling
     - Templates are vectorized; correlation, inverse FFT, magnitude,
       thresholding and fixed-window clustering share one compiled boundary
     - Retain the vendor FFT and XLA fusion; a megakernel cannot profitably
       absorb the FFT or variable trigger boundary
   * - Chi-square bin construction and point evaluation
     - Frequency accumulation remains ordered within each template so its
       discrete bin edges match the native backend
     - Independent CUDA template scans, including different lower cutoffs,
       are batched; the CPU keeps locality-friendly scalar scans. Inspiral
       triggered points are vectorized and evaluated on device
     - Use the batched JAX scan before considering a custom segmented scan;
       any replacement must preserve every cached edge
   * - Event records and coincidence
     - Variable-length metadata, MPI exchange and serialization
     - Dense masks and supported coincidence arithmetic stay in JAX; only
       compact records cross to the host
     - Keep irregular record handling outside persistent kernels
   * - ``pycbc_live`` chunk loop
     - Chunks advance in time order and retain pipeline/MPI state; detector
       buffers are currently advanced serially on each rank
     - Templates and samples are parallel within each chunk. Equal-shaped
       detector conditioning is the next useful batching axis
     - Do not fuse across I/O, MPI or checkpoint boundaries. First batch the
       detector numerical stages and assess whether root/worker conditioning
       can be shared

On the measured CUDA system, increasing the exact scan unroll from 16 to 128
reduced the complete ``pycbc_inspiral`` diagnostic wall time by approximately
a factor of two while retaining its arithmetic order. A parallel-prefix
alternative was faster but changed millions of conditioned samples. A strict
single-program Pallas recurrence was bit-identical but did not improve the
production-size scan. Direct FIR convolution was faster in isolation but
changed low-order bits, while the existing synchronized FFT FIR already took
only milliseconds. These results rule out those substitutions under the
accuracy requirement and keep custom-kernel work focused on future dense,
profile-dominant chains rather than kernel size alone.

The short live profile also shows why the detector axis comes first. Each rank
currently advances the two detector buffers one after the other, while the root
and filtering rank independently repeat that work. The steady 32-template CUDA
filter is small compared with these advances. Batching the equal-shaped numeric
conditioning for the detectors can expose parallel work without changing the
per-detector arithmetic; sharing results between ranks is a larger MPI and
failure-handling change. GWF decompression cannot be absorbed into an
accelerator kernel, but bounded replay read-ahead avoids repeating it for every
chunk. Detector state and variable trigger records remain outside numerical
kernels.

Function-level profiles make the split explicit. In the 64-second diagnostic,
each live rank spent about 41 seconds in the instrumented read-and-condition
stage (the event is named ``frame_read`` but encloses ``StrainBuffer.advance``).
A function-level replay showed 40.8 seconds in 18 LALFrame GWF reads, versus
2.7 seconds total in JAX high-pass and resampling. This is why the JAX replay
path batches host decoding before pursuing a larger conditioning kernel.
In the corresponding 32-template quick diagnostic, read-ahead reduced the
steady CUDA read-and-condition median from 2.307 seconds to 0.023 seconds.
Complete CUDA wall time fell from 75.424 seconds to 36.854 seconds and capacity
rose from 13.577 to 27.786 template-seconds per second. JAX CPU wall time fell
from 64.984 seconds to 27.324 seconds, while both ordinary CPU control arms
were unchanged within one percent. Every dataset in all nine JAX CPU and CUDA
per-block evidence files was bit-for-bit identical before and after the change.
These single-repetition quick results are optimization diagnostics, not a
publishable performance claim.
The root spent 16.5 seconds blocked in the MPI gather, while its JAX coincidence
calculation took 2.3 seconds. The filtering rank spent 13.6 seconds in filtering
and vetoes;
6.55 seconds of that was six formerly scalar chi-square bin-cache builds.
Those exact ordered scans are now dispatched as one CUDA batch across
independent templates; small CPU batches retain their locality-friendly scalar
path. A follow-up trial fused live correlation products with the triggered
point scans, but changed the complete CUDA diagnostic by only 0.3 percent and
introduced trigger-count-specific executables, so that boundary was not
retained. Persistent-cache audits still distinguish compilation from fresh-
process executable loading: the measured runs had no uncached compilations,
although loading hundreds of cached executables cost roughly 1--2 seconds per
process. This evidence prioritizes detector conditioning, batched ordered
scans, and fewer compiled boundaries over a whole-search megakernel.

To disable JAX GPU preallocation, set this before starting the process:

.. code-block:: console

   export XLA_PYTHON_CLIENT_PREALLOCATE=false

Record allocator settings with measurements. Disabling preallocation does
not remove the memory requirements of templates, FFT workspace and intermediate
arrays.

Benchmark suite
---------------

The complete-search suite compares four arms for each selected executable: the
pristine CPU reference, the candidate checkout's ordinary CPU path, JAX on CPU
and JAX on CUDA. Prepare the pinned inputs first, then select a run preset
explicitly:

.. code-block:: console

   python tools/run_jax_benchmarks.py prepare --config /path/to/suite.json
   python tools/run_jax_benchmarks.py qualify --scope live \
       --config /path/to/suite.json
   python tools/run_jax_benchmarks.py run --preset quick --scope live \
       --config /path/to/suite.json
   python tools/run_jax_benchmarks.py run --preset quick --quick-profiles \
       --scope both --config /path/to/suite.json
   python tools/run_jax_benchmarks.py run --preset full --scope both \
       --config /path/to/suite.json

``--scope`` accepts ``inspiral``, ``live`` or ``both``. It limits which
executables run without changing their four comparison arms. The separate
``qualify`` phase runs the 32-distinct-template scientific preflight for the
selected scope and performs no performance measurement.

The ``quick`` preset is an optimization-loop diagnostic. For each executable
in scope, it uses the 32-template diagnostic bank and runs one scientific
qualification followed by one timing repetition for every arm; the live
workload uses eight 8-second chunks of unpaced replay. By default it does not
run a separate preflight, paced replay, utilization profiles or the production
640-second replay and bank-size convergence grid. ``--quick-profiles`` adds
separate process and stage profiles for ``pycbc_inspiral`` and unpaced
``pycbc_live`` at the same 32-template size; those runs are never included in
the timing samples.
Thus ``--scope live`` launches eight complete ``pycbc_live`` searches: four
qualification runs and four timed runs. A single timing repetition on the
small diagnostic bank is useful for smoke testing and finding large
regressions, but is not a publishable performance result.

The ``full`` preset retains the comprehensive campaign. It first runs the
32-template qualification for each executable in scope. At every configured
production bank size it then qualifies and performs the configured number of
fresh timing repetitions for ``pycbc_inspiral`` and unpaced ``pycbc_live``.
At the largest bank size it additionally runs paced ``pycbc_live`` timing and
separate profiles for ``pycbc_inspiral``, unpaced ``pycbc_live`` and paced
``pycbc_live``. The default production grid is 1536, 3072 and 6144 templates
with three timing repetitions. These successive doublings establish the
reported bank-size convergence check; paced replay supplies latency, deadline
and backlog evidence, while profile runs are excluded from timing medians.

Both presets enforce the same strict scientific gates against the pinned CPU
reference before accepting timings. A failed gate normally stops later timing
and profiling. ``--allow-unqualified-timings`` may be used when investigating a
documented numerical divergence: the suite continues, but marks the receipts
as known-divergence diagnostics, disables the performance claim and does not
turn a failed scientific gate into a pass. See
:doc:`jax_numerical_differences` for the currently documented differences.

For JAX arms, qualification also primes an isolated persistent compilation
cache. Timed and profiled runs are accepted only when every compilation-cache
request is a hit. The per-process audit receipts and aggregate hit/miss counts
are stored with each run, so a campaign cannot silently time recompilation or
resume results produced under the former cache-disabled policy.

.. _jax-search-workflows:
.. _jax-workflows:

HTCondor workflow configuration
-------------------------------

For an existing Pegasus workflow whose inspiral executable is named
``inspiral``, configure its scheme and request a GPU for those jobs:

.. code-block:: ini

   [inspiral]
   processing-scheme = jax:cuda:0
   batch-size = 128

   [pegasus_profile-inspiral]
   condor|request_gpus = 1

The worker environment must provide PyCBC, a compatible JAX CUDA installation
and a visible GPU. Adapt resource requirements to the site's scheduler and
GPU configuration. This example is an HTCondor recipe; the measured backend
coverage in this documentation is CPU and NVIDIA CUDA.
