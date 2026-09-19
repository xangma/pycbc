.. _jax-arrays:

JAX arrays and processing schemes
=================================

PyCBC provides an optional JAX processing scheme for array and series
operations. Ordinary CPU execution does not require JAX. Selecting this scheme
uses the available JAX implementations; operations without an implementation
can raise an unsupported-operation error.

Installation and device selection
---------------------------------

Follow :doc:`install` for PyCBC's scientific dependencies, including LALSuite.
From a PyCBC checkout, install PyCBC and the CPU JAX package in the same Python
environment:

.. code-block:: console

   python -m pip install -e .
   python -m pip install jax

For NVIDIA GPUs, follow the `JAX installation instructions
<https://docs.jax.dev/en/latest/installation.html>`_ for the compatible CUDA
package and driver. Check the devices visible to that Python interpreter:

.. code-block:: python

   import jax

   print(jax.devices())

Use ``JAXScheme("cpu")`` for a CPU or ``JAXScheme("cuda:0")`` for the first
visible CUDA GPU. A requested device must be available; PyCBC raises an error
otherwise. Other JAX platforms require separate validation.

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries

   with JAXScheme("cpu"):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
       squared = series * series
       print(squared.numpy())  # [1. 4. 9.]

Applications exposing PyCBC's scheme options accept ``--processing-scheme
jax:cpu`` or ``--processing-scheme jax:cuda:0``. The scheme selects a backend;
each application's supported operations determine whether a complete run is
available.

Precision, views and host conversion
------------------------------------

``JAXScheme`` enables JAX's 64-bit support by default.
``PYCBC_JAX_ENABLE_X64=0`` disables that default. Enabling 64-bit support permits
double-precision arrays without promoting explicitly single-precision inputs.
JAX configuration applies to the Python process, so configure precision before
constructing arrays.

PyCBC wraps JAX's immutable storage to support array assignment and updates
through slices. Ordinary slices retain their relationship to the parent;
``copy()`` creates independent storage. Updating a PyCBC object replaces its
underlying JAX value, so a previously retrieved raw JAX array does not follow
later updates. These mutation semantics do not make every PyCBC operation
compatible with JAX tracing or automatic differentiation.

``Array.view(dtype)`` raises ``NotImplementedError`` in a JAX scheme: the
backend does not support PyCBC's writable views that reinterpret shared bytes
as another dtype.

``numpy()`` returns a host NumPy representation and can copy or synchronize
device data. Backend conversion can also allocate storage when the device or
dtype changes; DLPack conversion does not guarantee every conversion is free
of copies.

.. _jax-native-validation:

Native validation routes
-------------------------

The default runs the JAX implementations on the selected device. To isolate
an array operation's numerical differences, select the original CPU helper:

.. code-block:: python

   with JAXScheme("cpu", reference_operations=("inner", "cumsum")):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
       total_power = series.inner(series)

The command-line equivalent is ``--jax-reference-operations inner,cumsum``
with a JAX processing scheme. Available names are ``sum``, ``cumsum``,
``dot``, ``inner``, ``weighted_inner``, ``multiply_and_add``, ``squared_norm``,
``min``, ``max``, ``max_loc``, ``abs_max_loc`` and ``abs_arg_max``.
Each selection affects only its named array operation. Host conversion and
CPU execution can make these routes much slower and prevent JAX tracing or
differentiation through the selected operation. See
:doc:`jax_array_numerical_differences` for explanations and runnable examples.

Compilation cache
-----------------

``JAXScheme`` initializes a persistent compilation cache in the user's cache
directory and permits caching short compilations. Set
``JAX_COMPILATION_CACHE_DIR`` to choose a directory, or disable persistent
caching with ``JAX_ENABLE_COMPILATION_CACHE=false`` before starting Python.
Explicit JAX cache configuration takes precedence. Cache reuse can reduce
compilation work; process startup and execution still take time.

.. toctree::
   :maxdepth: 1

   jax_array_numerical_differences
