.. _jax-runtime:

JAX runtime
===========

PyCBC provides an optional JAX processing scheme for its core arrays, time
and frequency series, FFTs, and PSD operations. The scheme selects
the storage backend and compute device for supported operations.

Installation
------------

Follow :doc:`install` for PyCBC's build and scientific dependencies, including
LALSuite. From a PyCBC checkout, install:

.. code-block:: console

   python -m pip install -e ".[jax]"

Verify that JAX detects the expected devices:

.. code-block:: python

   import jax

   print(jax.__version__)
   print(jax.devices())

Selecting a scheme
------------------

Library callers can select a device with a context:

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries

   with JAXScheme("cpu"):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25, epoch=1000)
       squared = series * series

Use ``JAXScheme("cuda:0")`` for an available CUDA device. Other installed
JAX backends require separate compatibility and scientific validation.

Precision & 64-bit Floating Point
---------------------------------

By default, JAX uses 32-bit floating point numbers. For high-precision
gravitational-wave analysis (such as matched filtering and phase accumulation),
64-bit precision should be enabled:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

``JAXScheme`` enables 64-bit support by default unless
``PYCBC_JAX_ENABLE_X64`` disables it. This permits double-precision arrays;
it does not promote explicitly single-precision inputs or computations.

DLPack Interoperability
-----------------------

The array conversion helpers attempt DLPack interchange for compatible inputs.
They fall back to array conversion where needed; device or dtype changes and
conversion to host NumPy arrays may copy data or synchronize execution.
Zero-copy behavior is therefore conditional, not a general guarantee.
