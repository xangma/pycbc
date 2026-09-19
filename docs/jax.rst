.. _jax:

JAX Acceleration & Functional XLA
=================================

PyCBC provides an optional JAX acceleration stack providing pure functional,
XLA-compiled, and differentiable computation for gravitational-wave signal
analysis. JAX execution enables automated vectorization (`jax.vmap`), just-in-time
compilation (`jax.jit`), hardware-agnostic acceleration across CPU, NVIDIA CUDA,
Apple Silicon (Metal), and Google Cloud TPU architectures, and seamless zero-copy
interchange with array libraries and NumPy via DLPack (`__dlpack__`, `from_dlpack`).

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

Selecting a JAX scheme automatically enables **native JAX data conditioning**
(strain gating, whitening, Welch PSD estimation, and FFT transformations
directly on JAX device arrays) and **on-device batched waveform generation via ``diffgw``**
(or ``jaxwave``).

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
   * - Metal (Apple Silicon)
     - ``jax:metal``
     - ``JAXScheme("metal")``
   * - Google Cloud TPU
     - ``jax:tpu``
     - ``JAXScheme("tpu")``

Ordinary CPU use does not require JAX. Requesting unavailable JAX, CUDA,
or TPU raises an error. Check ``import jax; jax.devices()`` in the interpreter
running PyCBC.

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
     - Pure functional transformations; supports complete XLA fusion via ``jax.jit``.
   * - Waveforms
     - Direct on-device batched generation via ``diffgw.jax`` (``jaxwave``),
       SPAtmplt JAX port, and ringdown.
     - Seamless integration with PyCBC waveform registry and autodiff.
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
     - Fully differentiable parameter estimation likelihoods.

Documentation map
-----------------

.. toctree::
   :maxdepth: 1
   :caption: JAX user guides

   jax_runtime
   jax_filtering
   jax_search

.. toctree::
   :maxdepth: 1
   :caption: JAX maintainer guides

   jax_optimizations
   jax_testing
   jax_parity
   jax_workflows

For performance measurements, scaling studies, and scientific qualification:

* :ref:`jax-benchmark-protocol` defines the general execution controls, host isolation, single-thread baseline clamping, physical-core scaling, and convergence criteria.
* :ref:`jax-reference-campaign` specifies the concrete offline inspiral search workloads (Track 1: 384 compressed BNS/NSBH templates; Track 2: 512 uncompressed BNS templates evaluated dynamically with ``diffgw``; real H1 frame data, frozen bank SHAs, and scientific acceptance gates).
* :ref:`jax-batch-numerics` defines the live-filter batch matched-filtering API benchmarks and numerical oracle.
* :ref:`jax-performance` summarizes measured performance profiles, speedup figures across backends, operational realtime factors, and :ref:`timing metric boundaries <jax-timing-boundaries>`.
* :ref:`jax-tiled-pathways` outlines acceleration strategies and memory management for large-scale template banks.

.. toctree::
   :maxdepth: 1
   :caption: JAX benchmarking

   jax_benchmark_protocol
   jax_reference_campaign
   jax_batch_numerics
   jax_performance
   jax_tiled_pathways
