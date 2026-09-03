====================================
Torch workflow validation
====================================

Torch changes are validated in layers. Focused tests establish kernel and
public-API parity first; composed search or inference suites then test how those
paths interact. A microbenchmark is not a substitute for either layer.

Supported workflow scope
========================

The current branch covers the Torch search stack, shared waveform
infrastructure, the six TaylorF2-family registry entries documented in
:doc:`waveform`, and the supported inference and detector paths. Workflows must
not request a Torch-native waveform outside that registry and then treat a
fallback as native coverage.

The validation levels are:

.. list-table:: Validation levels
   :header-rows: 1
   :widths: 18 34 48

   * - Level
     - Scope
     - Examples
   * - Focused
     - Kernels and public APIs
     - Arrays, FFTs, filtering, matched filtering, registered waveforms,
       inference reductions, detector response
   * - Composed
     - Deterministic module pipelines
     - Search conditioning/trigger paths and relative-binning inference
   * - Suite
     - Existing project suites
     - Unit, search, inference, and documentation Pixi tasks
   * - Benchmark
     - Qualified performance workload
     - Fixed input, device, warm-up, timing boundary, and parity policy

Current commands
================

Use Pixi's checked-in tasks. A focused search/Torch selection is:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/test_torch_inverse_spectrum_validation.py \
     test/test_torch_timeseries_taper.py \
     test/test_torch_filter_pipeline.py::test_matched_filter_torch_vs_cpu \
     test/test_qtransform.py \
     test/test_chisq_torch.py \
     test/test_live_batch_torch_peaks.py

The registered waveform ports are checked with:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2ecc_torch.py \
     test/waveform/test_taylorf2nltides_torch.py \
     test/waveform/test_taylorf2redspin_torch.py \
     test/waveform/test_spintaylorf2_wrapper_torch.py

TaylorF2 batching, inference, and detector integration has a focused selection:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_taylorf2_batch.py \
     test/test_inference_wp3_optimizations.py::TestWP3NetworkGeometry::test_torch_batched_network \
     test/test_inference_wp3_optimizations.py::TestWP3RelbinTorchBatched::test_batched_summary_product \
     test/test_inference_wp3_optimizations.py::TestWP2MultiDetectorBatchedLikelihood::test_network_geometry_integration \
     test/test_torch_inference_pipeline.py::test_relative_binning_summaries_stay_on_device \
     test/test_detector.py::TestDetector::test_antenna_pattern_and_time_delay_torch \
     test/test_detector.py::TestDetector::test_network_geometry

Run the broader suites with:

.. code-block:: console

   pixi run -e unittest test-unittest
   pixi run -e search test-search
   pixi run -e inference test-inference
   pixi run -e docs test-docs

Standalone scheme-aware scripts can still be selected explicitly, for example
``pixi run -e unittest python test/test_array.py -s torch``. CUDA and MPS
coverage requires compatible hardware and must distinguish skips from passes.

Benchmark promotion
===================

Before refreshing a checked-in benchmark figure, record the source revision,
dependency versions, hardware, dtype, input and batch dimensions, thread
settings, warm-up, timed repetitions, synchronization, and transfer boundary.
Run the matching focused parity selection first. The checked-in PNGs are
reference snapshots and are not rewritten by the documentation build. See
:doc:`torch_performance` for their provenance limitations and the complete
measurement contract.
