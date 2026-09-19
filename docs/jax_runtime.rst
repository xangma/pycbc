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

Use ``JAXScheme("cuda:0")`` or ``JAXScheme("tpu")`` for available accelerators.

Precision & 64-bit Floating Point
---------------------------------

By default, JAX uses 32-bit floating point numbers. For high-precision
gravitational-wave analysis (such as matched filtering and phase accumulation),
64-bit precision should be enabled:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

PyCBC's JAX backend automatically manages float64 and complex128 precision when
64-bit precision is configured.

DLPack Interoperability
-----------------------

All PyCBC JAX array wrappers (`JAXArrayData`) implement standard DLPack protocols:
- `__dlpack__()`
- `__dlpack_device__()`
- `from_dlpack()`

This permits zero-copy memory exchange between JAX, CuPy, and NumPy.
