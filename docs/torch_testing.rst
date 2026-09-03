========================
Torch testing quickstart
========================

PyCBC's standalone scheme-aware tests accept ``-s torch``, ``-s torch:cuda``,
or ``-s torch:mps``. Pytest selects the test scheme with the test-only
``PYCBC_TEST_SCHEME`` variable; this is deliberately separate from the runtime
``PYCBC_SCHEME`` setting.

Use the repository's Pixi tasks so the test command runs in the supported
environment:

.. code-block:: console

   pixi run -e unittest test-unittest -q test/test_torch_ops.py
   PYCBC_TEST_SCHEME=torch pixi run -e unittest test-unittest -q test/test_array.py

The second form is for pytest suites that use PyCBC's scheme fixture. A
standalone invocation of the same style of test is:

.. code-block:: console

   pixi run -e unittest python test/test_array.py -s torch

CUDA and MPS tests require compatible hardware and a Torch build that supports
the requested device. Tests should skip when an optional device is not
available; a skip is not evidence that the device path passed.

Focused regression groups
=========================

The following paths and nodes provide a useful CPU-Torch smoke gate:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/test_torch_inverse_spectrum_validation.py \
     test/test_torch_timeseries_taper.py \
     test/test_torch_filter_pipeline.py::test_matched_filter_torch_vs_cpu \
     test/test_qtransform.py \
     test/test_chisq_torch.py \
     test/test_live_batch_torch_peaks.py

The waveform registry, sequence interfaces, and TaylorF2-family ports are
covered by:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2ecc_torch.py \
     test/waveform/test_taylorf2nltides_torch.py \
     test/waveform/test_taylorf2redspin_torch.py

TaylorF2 batching, inference reductions, relative-binning batches, and detector
projection have focused nodes of their own:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_taylorf2_batch.py \
     test/test_inference_wp3_optimizations.py::TestWP3NetworkGeometry::test_torch_batched_network \
     test/test_inference_wp3_optimizations.py::TestWP3RelbinTorchBatched::test_batched_summary_product \
     test/test_inference_wp3_optimizations.py::TestWP2MultiDetectorBatchedLikelihood::test_network_geometry_integration \
     test/test_torch_inference_pipeline.py::test_relative_binning_summaries_stay_on_device \
     test/test_detector.py::TestDetector::test_antenna_pattern_and_time_delay_torch \
     test/test_detector.py::TestDetector::test_network_geometry

The runtime-integration boundary is covered separately:

.. code-block:: console

   PYCBC_TEST_SCHEME=torch pixi run -e unittest test-unittest -q \
     test/test_torch_generator.py \
     test/test_live_snr_optimizer_torch.py \
     test/test_torch_inference_pipeline.py::test_kde_priors_stay_on_torch_device \
     test/test_torch_numpy_interop.py::test_coherent_chisq_cache_stays_on_torch_device \
     test/test_torch_types_pipeline.py::test_lalsim_injection_adder_stays_on_torch_device

For the broader project gates, use the named tasks in ``pyproject.toml``:

.. code-block:: console

   pixi run -e search test-search
   pixi run -e inference test-inference
   pixi run -e docs test-docs

Numerical and device-residency contract
=======================================

Torch implementations must preserve the public PyCBC dtype and metadata
contract. They may widen an intermediate calculation, but must not silently
downcast a double-precision input. Device-specific limits, notably the lack of
float64 and complex128 kernels for some MPS operations, must be guarded or
reported as unsupported.

Device residency applies to bulk numerical work in supported paths. It does
not prohibit scalar Python control decisions, file I/O, explicit ``.numpy()``
or ``.lal()`` conversion, compact public event outputs, or a documented CPU
fallback for an unsupported operation. Tests that poison host conversion are
used where the public API is expected to remain on the selected device.
