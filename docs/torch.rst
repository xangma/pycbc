.. _torch-scheme:

Using PyCBC with PyTorch
========================

PyCBC has an optional PyTorch processing scheme for selected array, FFT,
filtering, search, waveform, detector-geometry, and inference operations. It
can run on Torch CPU, CUDA, or Apple MPS devices. Support is operation-specific:
selecting a Torch device does not promise that every PyCBC operation is native
to, or remains resident on, that device.

This page explains how to install and select the scheme and describes its
public capability boundaries. Maintainers validating a change or a performance
result should also read :ref:`torch-parity`, :ref:`torch-performance`,
:ref:`torch-optimizations`, and :ref:`torch-testing`.

Installation
------------

Install PyCBC with the optional Torch dependency:

.. code-block:: console

   python -m pip install "pycbc[torch]"

For a source checkout, run:

.. code-block:: console

   python -m pip install -e ".[torch]"

The extra declares ``torch>=2.6,<2.14`` but does not choose a CUDA-specific
wheel. This interval is an installation constraint, not evidence that every
Python, platform, device, driver, and Torch combination has been tested. Use
the `official PyTorch installer <https://pytorch.org/get-started/locally/>`_
when a particular accelerator runtime is required, then check the devices that
the installed build can actually use:

.. code-block:: console

   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"

Compatibility and qualification
-------------------------------

The following table describes checked-in implementation and CI boundaries.
“Qualified” means the named lane exercises that exact resolved environment; it
is not a general guarantee for every compatible Torch release or device.

.. list-table::
   :header-rows: 1
   :widths: 17 30 26 27

   * - Layer
     - Implemented boundary
     - Checked-in coverage
     - Qualification limit
   * - Python
     - PyCBC declares Python 3.11 or newer.
     - Focused Linux Torch CPU tests run under Python 3.11, 3.12, and 3.13.
     - This does not establish every Torch-version and platform combination.
   * - Torch CPU
     - Torch tensors and selected native numerical routes are implemented.
     - The focused CPU selection runs in the Linux unit-test matrix.
     - CPU vendor, instruction set, FFT/BLAS libraries, and thread settings can
       change behavior and performance.
   * - CUDA
     - CUDA tensors and selected native kernels, batched FFTs, and graph paths
       are implemented when their runtime predicates pass.
     - A trusted weekly and manually dispatched self-hosted workflow requests
       Torch 2.13.0 from the CUDA 12.6 wheel index and checks real availability.
     - It is not a pull-request gate and does not qualify other drivers, GPUs,
       Python versions, or Torch builds.
   * - Apple MPS
     - MPS tensors are accepted by selected operations.
     - Conditional tests run when MPS is available, but there is no dedicated
       MPS lane in the checked-in workflows.
     - Support is narrower than CPU or CUDA; see `MPS limitations`_.

For a release or performance result, record the exact Python and Torch
versions, accelerator runtime and driver, hardware, source revision, and test
results instead of replacing them with this table.

Selecting the processing scheme
-------------------------------

Use ``--processing-scheme`` for command-line applications that expose PyCBC's
standard scheme options:

.. list-table::
   :header-rows: 1
   :widths: 28 29 43

   * - Selector
     - Device
     - Notes
   * - ``torch``
     - CPU
     - Uses Torch's current CPU thread setting.
   * - ``torch:4``
     - CPU
     - Shorthand for four Torch CPU threads.
   * - ``torch:cpu``
     - CPU
     - Explicit CPU selection.
   * - ``torch:cpu:4``
     - CPU
     - Sets four Torch CPU threads while the scheme context is active.
   * - ``torch:cuda``
     - CUDA
     - Uses ``--processing-device-id``; its default is zero.
   * - ``torch:cuda:1``
     - CUDA device 1
     - The explicit index wins over ``--processing-device-id``.
   * - ``torch:mps``
     - MPS
     - Uses ``--processing-device-id``; normally select device zero.

For example:

.. code-block:: console

   pycbc_inspiral --processing-scheme torch:cuda --processing-device-id 1 ...

Device-selection precedence is deliberately narrow:

#. An index in the scheme string wins. For example, ``torch:cuda:2`` selects
   ``cuda:2`` even when ``--processing-device-id`` has another value.
#. For exactly ``torch:cuda`` or ``torch:mps``, the command-line device ID is
   appended.
#. CPU forms and bare ``torch`` ignore ``--processing-device-id``.

The ``PYCBC_SCHEME`` environment variable selects the library's import-time
default context. Command-line parsers have their own ``cpu`` default; select
Torch explicitly for a command:

.. code-block:: console

   pycbc_inspiral --processing-scheme torch:cuda:1 ...

Environment selection does not perform the command-line device-ID merge, so
include the accelerator index in ``PYCBC_SCHEME`` when it matters. Set the
variable before importing PyCBC because the default context is initialized at
import time. See :ref:`torch-runtime` for the core runtime contract.

Library code can select a context explicitly:

.. code-block:: python

   import pycbc

   with pycbc.scheme.TorchScheme("cuda:1"):
       # Selected PyCBC operations use Torch on cuda:1 here.
       ...

   with pycbc.scheme.TorchScheme("cpu", num_threads=4):
       ...

The context validates CPU, CUDA, and MPS device types and restores the prior
Torch and discovered OpenMP CPU thread state when it exits.

Capability and fallback boundaries
----------------------------------

“Fallback” below means a route explicitly implemented by the relevant public
API. It does not mean that every unsupported Torch operation silently retries
on CPU. An API with no valid route raises an error.

.. list-table::
   :header-rows: 1
   :widths: 21 31 26 22

   * - Area
     - Torch capability
     - Fallback behavior
     - Important boundary
   * - Arrays and FFTs
     - Torch-backed PyCBC arrays, common reductions and conversions, and
       compatible complex/real FFT interfaces are available.
     - Optional optimized routes retain generic Torch or established public
       routes when their predicates reject an input.
     - Shape, dtype, device, contiguity, and autograd requirements can change
       route eligibility.
   * - Filtering and search
     - Core correlation, matched filtering, thresholding, chi-squared, peak
       handling, and selected live-batch paths have Torch implementations.
     - Native, compiled, or graph routes fall back according to their explicit
       runtime predicates.
     - Host-side orchestration and compact trigger output can remain on CPU;
       one device kernel does not establish end-to-end residency.
   * - Frequency-domain waveforms
     - ``TaylorF2``, ``TaylorF2NLTides``, ``TaylorF2RedSpin``,
       ``TaylorF2RedSpinTidal``, and ``TaylorF2Ecc`` have native regular-grid
       and arbitrary-frequency ports; TaylorF2 also has an explicit batch API.
     - Ineligible native calls use an existing dispatcher only where that
       approximant and interface defines one.
     - A fallback may generate on CPU and copy the result to the selected
       device. See :doc:`waveform` for the registered families and predicates.
   * - ``SPAtmplt``
     - A separate native Torch implementation supports regular-grid and
       arbitrary-frequency generation.
     - It uses its established host route when disabled or ineligible.
     - Its controls and predicates are separate from the five registry ports.
   * - Waveform decompression
     - Inline linear, quadratic, cubic, and quartic interpolation has a native
       Torch route on supported devices and dtypes.
     - Other interpolation modes retain their established host route.
     - CPU/CUDA arithmetic is precision-promoted; MPS has a distinct
       single-precision boundary.
   * - Domain and prior helpers
     - Selected coordinate, cosmology, transform, boundary, and prior
       calculations preserve compatible Torch tensor inputs.
     - Unsupported inputs continue through the existing public implementation
       where its contract permits.
     - Coverage is deliberately scoped to the focused compatibility tests; it
       does not make every scientific helper tensor-native.
   * - Detector geometry
     - Selected single-detector and network antenna, response, delay, arrival,
       and effective-distance calculations accept compatible Torch tensors.
     - Scalar and NumPy calls retain their existing routes.
     - Geometry support is not a claim that every detector operation is
       device-resident.
   * - Inference
     - Selected model/device plumbing, likelihood utilities, waveform
       generation, Gaussian likelihoods, relative binning, and supported
       marginalization paths accept Torch-backed inputs.
     - Unsupported types or model paths retain only the fallback defined by
       the individual API.
     - Control flow, I/O, scalar decisions, and unsupported models can remain
       on the host. Whole-application residency and autograd are not implied.

Optimization switches do not expand this table. They request alternate routes
whose runtime predicates still determine eligibility. Defaults and rollback
behavior are documented in :ref:`torch-optimizations`.

Residency and parity
--------------------

A final tensor on an accelerator proves its output device, not where every
calculation ran. When residency matters, tests and benchmark artifacts must
record the selected route and check host/device conversion boundaries as well
as values, dtype, shape, and public metadata. Follow :ref:`torch-parity` before
interpreting a performance result.

MPS limitations
---------------

* Many operations do not provide ``float64`` or ``complex128`` MPS kernels.
  PyCBC therefore uses supported lower-precision paths or rejects an
  unsupported boundary where required.
* The accuracy-promoted single-precision batched FFT route can stage MPS data
  through CPU memory and must not be reported as fully resident.
* Native inline waveform decompression uses single-precision interpolation and
  ``complex64`` output on MPS.
* A waveform support predicate can choose an established host implementation
  at an accuracy boundary and then copy the result to the selected device.
* Accelerator work is asynchronous. Synchronize MPS around timed regions or a
  measurement may report enqueue latency instead of completed work.
* There is no dedicated MPS CI lane. Report the exact macOS, hardware, Python,
  and Torch versions for any MPS claim.

Troubleshooting
---------------

``Install PyTorch to use the Torch processing scheme``
   Install the optional dependency into the interpreter that launches PyCBC.
   Check ``python -m pip show torch`` and ``python -c "import pycbc, torch"``.

``Torch CUDA device requested but CUDA is unavailable``
   Check ``torch.cuda.is_available()``, the installed wheel, driver/runtime
   compatibility, container device access, and requested index. Installing the
   PyCBC extra alone does not select a CUDA wheel.

``Torch MPS device requested but MPS is unavailable``
   Check ``torch.backends.mps.is_built()`` and
   ``torch.backends.mps.is_available()``. Availability depends on the Torch
   build, macOS version, and Apple hardware.

An unexpected accelerator is selected
   Check for an explicit index in ``torch:cuda:N`` first because it wins over
   ``--processing-device-id``. Also inspect ``PYCBC_SCHEME`` and the process's
   accelerator visibility settings.

Performance is slower than the CPU route
   Small workloads can be dominated by launch, transfer, compilation, or
   allocation cost. Confirm the route and residency, separate cold and warm
   samples, synchronize accelerators, and compare identical complete work.
   Follow the protocol in :ref:`torch-performance`.

Results differ from a reference
   Record the dtype, device, and selected route before changing tolerances.
   Disable optional optimizations one at a time and follow
   :ref:`torch-parity`.

Out-of-memory failures
   Reduce batch or bank size and retain the failed cell as OOM rather than
   dropping it from the record. Relevant live-batch limits are documented in
   :ref:`torch-optimizations`.
