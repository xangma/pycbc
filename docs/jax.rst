.. _jax:

JAX acceleration
================

PyCBC's optional JAX backend runs supported calculations on a CPU or CUDA GPU.
The original CPU processing scheme remains the default. JAX arrays, compiled
kernels and automatic differentiation are available where the selected API
supports them; Python orchestration, file I/O and metadata remain host work.

Install PyCBC and JAX in the same Python environment, following
:doc:`jax_arrays` for dependencies, device selection and precision. Waveform
generation uses the optional diffGW dependency; :doc:`jax_waveform` describes
the supported models and original waveform paths.

Search commands
---------------

Use the dedicated JAX commands for accelerated searches:

.. code-block:: console

   pycbc_inspiral_jax --processing-scheme jax:cuda:0 ...
   mpiexec -n 3 pycbc_live_jax --processing-scheme jax:cuda:0 ...

The remaining options describe the input data, template bank and analysis.
See :doc:`jax_inspiral` and :doc:`jax_live` for supported configurations.
The original ``pycbc_inspiral`` and ``pycbc_live`` retain their CPU pipelines.

Library calculations
--------------------

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries

   with JAXScheme("cpu"):
       data = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
       power = data * data

``JAXScheme`` enables 64-bit support by default. It preserves explicitly
single-precision inputs. Configure precision before constructing arrays.
Host conversion, including ``numpy()`` on a PyCBC array, synchronizes device
work; use it at a deliberate output boundary.

Numerical validation
--------------------

The default JAX path uses device calculations. Floating-point grouping,
transcendental functions, FFT implementations and interpolation can produce
numerical differences. :doc:`jax_numerical_differences` links explanations,
executed examples and the independent original-implementation controls.

``reference_operations`` is empty by default. Select a calculation to replay
its original implementation while the surrounding calculations continue
using JAX:

.. code-block:: python

   with JAXScheme("cpu", reference_operations=("fft", "ifft")):
       ...

The command-line equivalent is ``--jax-reference-operations fft,ifft``.
Original-path validation transfers data to the host and can be much slower.
It requires identical inputs and matching precision and FFT settings. See
:doc:`jax_testing` for regression checks and :doc:`jax_benchmarks` for
complete-search comparisons and timing boundaries.

.. toctree::
   :maxdepth: 1

   jax_numerical_differences
   jax_testing
   jax_benchmarks
