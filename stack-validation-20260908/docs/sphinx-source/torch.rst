.. _torch-scheme:

Using PyCBC with PyTorch
========================

The optional Torch scheme runs selected PyCBC operations on CPU, CUDA, or
Apple MPS. Support depends on the operation, dtype, and device; choosing a
device does not make an entire application run there.

Quick start
-----------

Install the optional dependency, following :doc:`install` for PyCBC's other
dependencies:

.. code-block:: console

   python -m pip install "pycbc[torch]"

For a source checkout, use ``python -m pip install -e ".[torch]"``. PyCBC
requires Python 3.11 or newer; the extra declares ``torch>=2.6,<2.14``. This
range is an installation constraint, not a guarantee that every combination
has been tested. The extra does not select a CUDA-specific wheel; use the
`PyTorch installer <https://pytorch.org/get-started/locally/>`_ for your runtime.

Select a scheme around the work that should use Torch:

.. code-block:: python

   from pycbc.scheme import TorchScheme
   from pycbc.types import TimeSeries

   with TorchScheme("cpu", num_threads=4):
       series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
       squared = series * series

Command-line applications with PyCBC's standard scheme options accept:

.. code-block:: console

   pycbc_inspiral --processing-scheme torch:cuda:0 ...

Choosing a device
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - Device
     - ``--processing-scheme``
     - Library context
   * - CPU
     - ``torch:cpu`` or ``torch:cpu:4``
     - ``TorchScheme("cpu", num_threads=4)`` for four threads
   * - CUDA
     - ``torch:cuda:0``
     - ``TorchScheme("cuda:0")``
   * - Apple MPS
     - ``torch:mps``
     - ``TorchScheme("mps")``; see `MPS limitations`_

Ordinary CPU use does not require Torch. Requesting unavailable Torch, CUDA,
or MPS raises an error. Check ``torch.cuda.is_available()`` or
``torch.backends.mps.is_available()`` in the interpreter running PyCBC.

Bare ``torch`` selects CPU. Without a thread count, Torch keeps its current
setting. See :ref:`torch-runtime` for thread restoration, device-index
precedence, and the import-time ``PYCBC_SCHEME`` environment variable.
Standard command-line parsers default to ``cpu``; select Torch explicitly.

Capabilities and fallback
-------------------------

Fallback is defined by each API. There is no universal retry on CPU: an
unsupported operation with no valid route raises an error.

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Area
     - Available Torch operations
     - Limits and fallback
   * - Arrays and FFTs
     - Arrays, series, common reductions, conversions, and FFT interfaces.
     - Dtype, layout, device, and autograd affect route eligibility.
   * - Filtering and search
     - Correlation, matched filtering, thresholds, chi-squared, peaks, and
       selected live-batch paths.
     - Optimized routes have eligibility checks; orchestration and trigger
       output can remain on CPU.
   * - Waveforms
     - Five TaylorF2-family ports and separate ``SPAtmplt`` routes support
       regular-grid and arbitrary-frequency generation; TaylorF2 has a batch API.
     - Existing host dispatch is used only where that interface defines it.
       Generation may run on CPU before a device copy. See :doc:`waveform`.
   * - Decompression
     - Inline linear, quadratic, cubic, and quartic interpolation.
     - Other modes use the established host route; MPS uses single precision.
   * - Domain and prior helpers
     - Selected coordinate, cosmology, transform, boundary, and prior helpers
       preserve compatible tensors.
     - Other inputs use existing implementations where supported.
   * - Detector geometry
     - Selected antenna, response, delay, arrival, and effective-distance
       calculations for single detectors and networks.
     - Scalar and NumPy calls retain their existing routes.
   * - Inference
     - Selected Gaussian likelihood, relative-binning, waveform, and
       marginalization paths accept Torch-backed inputs.
     - Model support varies; control flow, I/O, and scalar decisions can
       remain on the host.

An accelerator output does not prove that every intermediate stayed on that
device. Support also does not imply autograd through the complete operation.
Optional switches still obey these limits; see :ref:`torch-optimizations`
for defaults and how to disable them.

MPS limitations
---------------

* MPS cannot provide the ``float64`` and ``complex128`` types required by some
  operations. Use supported lower-precision paths or CPU/CUDA where needed.
  Absolute ``TimeSeries.sample_times`` raises ``TypeError`` on MPS.
* The precision-promoted single-precision batched FFT route stages data
  through CPU memory. Its calculation is not fully resident on MPS.
* Native inline decompression uses single-precision interpolation and
  ``complex64`` output. Waveform accuracy guards can also select CPU
  computation followed by a copy to MPS; see :doc:`waveform`.
* Synchronize MPS around timed regions to measure completed work. There is
  no dedicated MPS CI lane; record the macOS, hardware, Python, and Torch
  versions when validating a result.

Troubleshooting
---------------

Torch is missing or cannot be imported
   Install it in the interpreter launching PyCBC. Check
   ``python -m pip show torch`` and ``python -c "import pycbc, torch"``.

CUDA or MPS is unavailable
   Check the availability calls above. For CUDA, check the wheel, driver,
   device access, and index. For MPS, also check
   ``torch.backends.mps.is_built()`` and your macOS/hardware support.

The wrong accelerator is selected
   An explicit ``torch:cuda:N`` index wins over ``--processing-device-id``.
   Check accelerator visibility and the selector rules in :ref:`torch-runtime`.

Performance is slower than CPU
   Small workloads can be dominated by transfers, launches, or setup. Compare
   the same complete work and separate first-call costs from repeated work;
   see :ref:`torch-performance` for measurements and
   :ref:`torch-benchmark-protocol` for timing controls.

Results differ from a reference
   Record dtype, device, and route before changing tolerances. Disable
   optional optimizations individually and follow :ref:`torch-parity`.

Out of memory
   Reduce the batch or bank size. Live-batch controls are listed in
   :ref:`torch-optimizations`.

Maintainers should use :ref:`torch-parity` and :ref:`torch-testing` to qualify
specific operations and environments. A skipped device test does not qualify
that device, and the dependency range does not replace those results.
