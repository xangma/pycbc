.. _jax:

JAX acceleration
================

PyCBC provides an optional JAX backend for selected array operations, signal
processing and waveform functions. JAX can compile these operations with XLA
and supports automatic differentiation where the implementation permits it.
The measured configurations in this documentation use CPU and NVIDIA CUDA.
Other JAX backends require compatible installations and separate validation.
DLPack conversion is attempted for compatible arrays; NumPy conversion,
device transfer or dtype changes can copy data and synchronize execution.

Quick start
-----------

Install PyCBC with the optional JAX dependency:

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

Selecting a JAX scheme routes supported operations through JAX, including
parts of strain conditioning and filtering. Device waveform generation via
``diffgw`` requires the optional dependency and explicit ``--enable-diffgw``;
it is not enabled merely by selecting JAX. Unsupported operations may use
host implementations, so a JAX scheme does not imply an entirely device-resident
search.

Performance results
-------------------

See :ref:`inspiral templates/core and templates/GPU <jax-search-capacity>`
and the :ref:`live-sized filter estimates <jax-live-capacity>` for capacity
tables and plots. :doc:`jax_performance` also records timing boundaries and
scientific qualification results. The
:doc:`jax_gpu_investigation` covers batch size, memory and GPU utilisation.

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

Capabilities and fallback
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Area
     - Available JAX operations
     - Limits and fallback
   * - Arrays, FFTs and Conditioning
     - Arrays, series, common reductions, conversions, FFT interfaces, and native
       strain conditioning (Welch PSD estimation, interpolation).
     - Standard JAX precision modes; 64-bit precision enabled via ``jax_enable_x64``.
   * - Filtering and search
     - Matched filtering, correlation, thresholding, chi-squared vetoes, and
       single-detector peak clustering.
     - Selected kernels are JIT compiled; search orchestration also performs host work.
   * - Waveforms
     - Direct on-device batched generation via ``diffgw.jax`` (``jaxwave``),
       SPAtmplt JAX port, and ringdown.
     - Model availability and differentiability depend on the waveform implementation.
   * - Decompression
     - Inline linear, quadratic, cubic, and quartic spline interpolation.
     - Vectorized evaluation across frequency bins.
   * - Domain and prior helpers
     - Coordinate transformations, cosmology distance/volume lookups, boundary
       conditions, and prior evaluation.
     - Preserves JAX array types and differentiability.
   * - Detector geometry
     - Antenna pattern, time delay, Earth rotation response, and effective
       distance calculations.
     - Functional detector geometry evaluation without host synchronization.
   * - Inference
     - Gaussian likelihood, relative binning, and marginalization models.
     - Differentiability depends on the selected model, waveform and parameter path.

Documentation map
-----------------

.. toctree::
   :maxdepth: 1
   :caption: JAX results

   jax_performance
   jax_gpu_investigation

.. toctree::
   :maxdepth: 1
   :caption: JAX guides

   jax_runtime
   jax_filtering
   jax_search
   jax_testing

The benchmark definitions retain separate scopes: the executable campaign
runs ``pycbc_inspiral`` through HDF output, while the streaming protocol
defines a ``LiveBatchMatchedFilter.process_data`` API benchmark. The published
live-sized estimates instead use the synthetic filter microbenchmark; they
do not measure that full API or the ``pycbc_live`` executable.

.. toctree::
   :maxdepth: 1
   :caption: JAX benchmark definitions

   jax_benchmark_protocol
   jax_reference_campaign
   jax_batch_numerics
