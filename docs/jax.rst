.. _jax:

JAX acceleration
================

PyCBC's optional JAX backend runs supported calculations on a CPU or CUDA GPU.
The original CPU processing scheme remains the default. JAX arrays, compiled
kernels and automatic differentiation are available where the selected API
supports them; Python orchestration, file I/O and metadata remain host work.

Getting started
---------------

Install PyCBC and JAX in the same Python environment. The array guide covers
dependencies, device selection, precision and storage:

.. toctree::
   :maxdepth: 1

   jax_arrays

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

Signal processing
-----------------

These guides describe the supported device calculations and their original-implementation validation controls.

.. toctree::
   :maxdepth: 1

   jax_fft
   jax_filtering
   jax_gwf

Searches
--------

The library guide describes device arrays for batched filtering, events and coincidences.

.. toctree::
   :maxdepth: 1

   jax_search

Validation and performance
--------------------------

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
It requires identical inputs and matching precision and FFT settings.

.. toctree::
   :maxdepth: 1

   jax_numerical_differences

