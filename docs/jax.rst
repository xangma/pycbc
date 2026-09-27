.. _jax:

JAX acceleration
================

PyCBC provides an optional JAX backend for selected array operations, signal
processing, waveform, veto and event operations. JAX can compile these
operations with XLA and supports automatic differentiation where the selected
implementation permits it. Selecting JAX does not make an entire workflow
device-resident or differentiable: coverage, host boundaries and dtypes are
operation-specific. The measured configurations in this documentation use CPU
and NVIDIA CUDA. Other JAX backends require compatible installations and
separate validation.

DLPack conversion is attempted for compatible arrays; NumPy conversion,
device transfer or dtype changes can copy data and synchronize execution.

.. _jax-runtime:

Quick start
-----------

Follow :doc:`install` for PyCBC's build and scientific dependencies, including
LALSuite. From a PyCBC checkout, install the optional JAX dependency:

.. code-block:: console

   python -m pip install -e ".[jax]"

Select a scheme around the work that should use JAX:

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries

   with JAXScheme("cpu"):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
       squared = series * series

Command-line applications with PyCBC's standard scheme options accept:

.. code-block:: console

   pycbc_inspiral --processing-scheme jax:cuda:0 ...

Selecting a JAX scheme dispatches operations that have JAX implementations;
unsupported operations either use their documented fallback or raise an
unsupported-operation error. Selected strain-conditioning, filtering, veto and
event paths have JAX implementations. Device waveform generation via
``diffgw`` requires the optional dependency and explicit ``--enable-diffgw``;
it is not enabled merely by selecting JAX. Python orchestration, file I/O,
metadata and MPI serialization remain host work. See :doc:`jax_search` for
supported paths and explicit limits.

.. _jax-performance:
.. _jax-parity-status:

Numerical Agreement and Qualification Status
--------------------------------------------

In the retained ``pycbc_inspiral`` and ``pycbc_live`` detector workloads, JAX
matches every retained trigger identity and every audited candidate-ranking
decision, with close SNR and phase agreement. This scoped outcome agreement is
not a complete scientific qualification: the frozen full-search comparator
still fails some retained chi-square values and exact conditioned-strain,
segment-spectrum, and PSD gates. See :ref:`jax-numerical-differences` for the
stage-by-stage attribution, frozen gate results, and selected diagnostic
receipts. Performance results must not be accepted until a fresh qualification
run passes the required gates.

For an interactive, step-by-step walkthrough of pipeline stage agreement on real
LIGO Hanford detector data (GW150914), see the companion Jupyter notebook at
``examples/jax/jax_numerical_parity.ipynb``.

.. _jax-campaigns:

Diagnostic and Benchmark Suite
------------------------------

PyCBC includes automated tools to drive benchmark campaigns, compare execution
arms against the pristine CPU reference, and profile performance:

* :download:`Suite Driver <../tools/run_jax_benchmarks.py>`: Automates campaign
  lifecycle phases (``plan``, ``prepare``, ``qualify``, ``run``, ``report``).
* :download:`Science Comparator <../tools/benchmark_science.py>`: Enforces the
  frozen numerical rules and acceptance tolerances defined in :ref:`jax-benchmark-protocol`.
* :download:`Inspiral Campaign Runner <../tools/bench_jax_inspiral_campaign.py>`: Runs
  complete ``pycbc_inspiral`` pipeline comparisons.
* :download:`Live Campaign Runner <../tools/bench_jax_live_campaign.py>`: Runs
  complete ``pycbc_live`` pipeline comparisons.
* :download:`GPU Timeline Profiler <../tools/profile_jax_gpu_timeline.py>` and
  :download:`Timeline Plotter <../tools/plot_jax_gpu_timeline.py>`: Profile and
  visualize CUDA kernel timelines.

To run the automated benchmark suite:

.. code-block:: console

   python tools/run_jax_benchmarks.py plan --config /path/to/suite.json
   python tools/run_jax_benchmarks.py prepare --config /path/to/suite.json
   python tools/run_jax_benchmarks.py qualify --config /path/to/suite.json
   python tools/run_jax_benchmarks.py run --config /path/to/suite.json
   python tools/run_jax_benchmarks.py report --config /path/to/suite.json

Choosing a device
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - Device
     - ``--processing-scheme``
     - Library context
   * - CPU
     - ``jax:cpu``
     - ``JAXScheme("cpu")``
   * - CUDA GPU
     - ``jax:cuda:0``
     - ``JAXScheme("cuda:0")``

Ordinary CPU use does not require JAX. A requested device must appear in
``jax.devices()`` in the interpreter running PyCBC. Other device names can
resolve when supplied by the installed JAX backend, but this does not establish
PyCBC support or scientific qualification for Metal or TPU.

Check the version and available devices in that interpreter:

.. code-block:: python

   import jax

   print(jax.__version__)
   print(jax.devices())

Precision
---------

``JAXScheme`` enables 64-bit support by default unless
``PYCBC_JAX_ENABLE_X64`` disables it. This permits double-precision arrays;
it does not promote explicitly single-precision inputs or computations.

High-pass compatibility mode
----------------------------

The default JAX Butterworth high-pass follows LAL's sample-ordered recurrence:

.. code-block:: console

   pycbc_inspiral --processing-scheme jax:cuda:0 ...

The equivalent library context is ``JAXScheme("cuda:0")``. The default
``lal-serial`` mode constructs LAL-order coefficients and
uses a JAX ``lax.scan`` for each forward and reverse section. It requires JAX
64-bit support. It changes Butterworth high-pass filtering only; it does not
change the FIR/LDAS path used by the measured live campaign. The mode is
intended for compatibility and has not passed complete-search qualification.
Select ``--jax-highpass-mode parallel`` (or
``JAXScheme(..., highpass_mode="parallel")``) to use the previous parallel
prefix-scan calculation. On CUDA, the serial scans unroll 16 samples per loop
iteration while retaining the sample-order recurrence; on CPU they remain
rolled. See
:ref:`jax-highpass-compat-evidence` for results and costs.

Capabilities and fallback
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Area
     - Available JAX operations
     - Limits and fallback
   * - Arrays, FFTs and Conditioning
     - Arrays, series, reductions, conversions, FFT interfaces and selected
       strain-conditioning kernels, including filters, Welch PSD estimation
       and interpolation.
     - Coverage and dtype boundaries are entry-point specific; standard JAX
       precision modes apply and 64-bit support is enabled via
       ``jax_enable_x64``.
   * - Filtering and search
     - Matched filtering, correlation, thresholding, selected chi-squared
       vetoes, peak clustering and event/coincidence state.
     - Selected kernels are JIT compiled; event metadata, conversions and
       search orchestration also perform host work.
   * - Waveforms
     - Selected JAX waveform and decompression kernels, including the
       TaylorF2, SPAtmplt and ringdown APIs where their JAX entry points are
       selected. Optional ``diffgw``/``jaxwave`` providers can supply additional
       batched generation.
     - Provider, model, precision, batching and differentiability support are
       implementation-specific and require the corresponding optional package.
   * - Decompression
     - Inline linear, quadratic, cubic, and quartic spline interpolation.
     - Vectorized evaluation across frequency bins; input packing and metadata
       may remain on the host.
   * - Domain and prior helpers
     - Selected coordinate transformations, cosmology distance/volume lookups,
       boundary conditions and prior evaluation.
     - JAX array preservation and differentiability depend on the selected
       transform and cosmology path; unsupported paths may use host libraries.
   * - Detector geometry
     - Antenna pattern, time delay, Earth rotation response, and effective
       distance calculations.
     - Numeric calculations can use JAX arrays, while detector geometry and
       other static inputs are host-provided constants.
   * - Inference
     - Gaussian likelihood, relative binning, and marginalization models.
     - Model, waveform and parameter paths determine device coverage and
       differentiability; Python orchestration remains on the host.

Documentation map
-----------------

.. toctree::
   :maxdepth: 1

   jax_search
   jax_testing
   jax_numerical_differences
