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
Conditioning lives in ``filter/resample_jax.py`` and ``strain/strain_jax.py``;
filtering and vetoes live in ``filter/matchedfilter_jax.py`` and
``vetoes/*_jax.py``. ``events/eventmgr_jax.py`` and ``events/coinc_jax.py`` own
JAX event storage and coincidence processing. Both JAX CPU and JAX CUDA select
these modules when execution reaches a supported JAX entry point.
Current conditioning supports Kaiser FIR filters, LDAS decimation and direct
Butterworth high/low-pass filter conventions. The default JAX high-pass runs
the LAL sample recurrence through ``jax.lax.scan``; the optional
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
Include compilation in fresh-process timing and report warmed measurements
separately. A larger fitting batch need not improve throughput; validate it
with a current complete-executable campaign under the policy in
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

To disable JAX GPU preallocation, set this before starting the process:

.. code-block:: console

   export XLA_PYTHON_CLIENT_PREALLOCATE=false

Record allocator settings with measurements. Disabling preallocation does
not remove the memory requirements of templates, FFT workspace and intermediate
arrays.

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
