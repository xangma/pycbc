.. _torch-runtime:

PyTorch runtime
===============

PyCBC provides an optional Torch processing scheme for its core arrays, time
and frequency series, FFTs, and selected PSD operations. The scheme selects
the storage backend and device for supported operations. Individual APIs
still determine supported shapes, dtypes, and execution routes.

Installation
------------

Follow :doc:`install` for PyCBC's build and scientific dependencies, including
LALSuite. From a PyCBC checkout, install:

.. code-block:: console

   python -m pip install -e ".[torch]"

The extra declares ``torch>=2.6,<2.14``. This is an installation constraint;
it does not mean every version in that interval has been qualified. The extra
does not select a CUDA-specific wheel. Install the Torch build appropriate
for your device and driver, then check availability in the same interpreter:

.. code-block:: python

   import torch

   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.backends.mps.is_available())

Ordinary CPU use does not require Torch. Explicitly requesting Torch with a
missing or unloadable installation raises an error. Requesting unavailable
CUDA or MPS also raises an error.

Selecting a scheme
------------------

Library callers can select a device with a context:

.. code-block:: python

   from pycbc.scheme import TorchScheme
   from pycbc.types import TimeSeries

   with TorchScheme("cpu", num_threads=4):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25, epoch=1000)
       squared = series * series

Use ``TorchScheme("cuda:0")`` or ``TorchScheme("mps")`` for an available
accelerator. CPU thread counts must be positive. A CPU context with an
explicit count restores the prior Torch and discovered OpenMP thread settings
when it exits. Without a count, it keeps Torch's current thread setting.
PyCBC's scheme context is shared process state; overlapping active contexts
are not supported.

Applications exposing PyCBC's standard processing options accept:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - ``--processing-scheme``
     - Selection
   * - ``torch`` or ``torch:cpu``
     - Torch CPU with the current Torch thread setting.
   * - ``torch:4`` or ``torch:cpu:4``
     - Torch CPU with four threads while the context is active.
   * - ``torch:cuda``
     - CUDA using ``--processing-device-id`` (default zero).
   * - ``torch:cuda:1``
     - CUDA device 1; the explicit index takes precedence.
   * - ``torch:mps``
     - MPS using ``--processing-device-id`` (normally zero).

CPU selectors ignore ``--processing-device-id``. An explicit index in an
accelerator selector takes precedence over that option. Selecting a scheme
does not establish that an application's complete workflow supports it.

``PYCBC_SCHEME`` selects the library's default context at import time. Set it
before importing PyCBC, for example ``PYCBC_SCHEME=torch:cuda:1``. It does not
apply the command-line device-ID merge. Applications using the standard
argument parser default ``--processing-scheme`` to ``cpu``; use that option
explicitly to select Torch for their processing context.

Array and execution contract
----------------------------

Torch-backed ``Array``, ``TimeSeries``, and ``FrequencySeries`` retain the
public PyCBC interfaces and series metadata. Supported operations dispatch
through the active scheme. Scheme or device changes can require a copy;
``copy=False`` construction requires compatible storage, dtype, and scheme.
For direct access to native storage, use ``pycbc.types.backend.backend_array``
instead of inspecting private array attributes.

Torch-aware arithmetic and tensor conversions can retain autograd history.
This does not promise differentiation through every PyCBC operation: in-place
or output-buffer FFT interfaces retain Torch's autograd restrictions, and
explicit conversion to NumPy leaves the Torch graph and device. Verify the
forward and backward behavior of the particular operation you need.

Optimized FFT routes have dtype, layout, device, and autograd eligibility
checks. An ineligible route uses its defined fallback or raises an error;
there is no universal host fallback for unsupported APIs. A result stored on
an accelerator does not prove that every intermediate stayed there.

MPS does not provide the double-precision types required by some operations.
In particular, absolute ``TimeSeries.sample_times`` raises ``TypeError`` under
an active MPS scheme; use a CPU or CUDA scheme to obtain those coordinates.
The precision-promoted single-precision batched FFT path stages MPS data
through CPU memory. Choose supported dtypes explicitly and retain these
boundaries when interpreting device or performance results.
