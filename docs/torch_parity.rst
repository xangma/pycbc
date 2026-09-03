=======================
Torch parity validation
=======================

Torch performance evidence is accepted only after the corresponding scientific
outputs pass focused parity tests. The tests compare public results and
metadata; they do not assume that two independent floating-point
implementations are byte-identical.

Result terminology
==================

``Raw-byte exact`` means that every compared output byte matches, including
signed zero. ``Numerically equivalent`` means that shapes, dtypes, metadata,
and values pass the test's explicit tolerances. ``Route-qualified`` additionally
means that the expected backend/device was used and no prohibited host fallback
occurred.

A numerical pass does not establish a speedup. A timing is reported only after
the route appropriate to that timing has passed its parity checks.

Current waveform boundary
=========================

The current public Torch waveform registry contains FD and FD-sequence entries
for ``TaylorF2``, ``TaylorF2NLTides``, ``TaylorF2RedSpin``,
``TaylorF2RedSpinTidal``, and ``TaylorF2Ecc``. Their supported interfaces and
selection behavior are described in :doc:`waveform`.

Regular-grid ports with a standard LAL path compare the supported Torch route
against that reference. A sequence interface marked ``native extension`` has
no equivalent LAL public interface; it is instead checked against its analytic
or regular-grid contract and ordering, duplicate-frequency, support, dtype,
and device invariants. Predicate guards remain authoritative for individual
requests.

Focused commands
================

Run the registry and TaylorF2-family waveform parity tests in the Pixi
unit-test environment:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2ecc_torch.py \
     test/waveform/test_taylorf2nltides_torch.py \
     test/waveform/test_taylorf2redspin_torch.py \
     test/waveform/test_spintaylorf2_wrapper_torch.py

Run the TaylorF2 batch, inference, and detector comparisons with:

.. code-block:: console

   pixi run -e unittest test-unittest -q \
     test/waveform/test_taylorf2_batch.py \
     test/test_inference_wp3_optimizations.py::TestWP3NetworkGeometry::test_torch_batched_network \
     test/test_inference_wp3_optimizations.py::TestWP3RelbinTorchBatched::test_batched_summary_product \
     test/test_inference_wp3_optimizations.py::TestWP2MultiDetectorBatchedLikelihood::test_network_geometry_integration \
     test/test_torch_inference_pipeline.py::test_relative_binning_summaries_stay_on_device \
     test/test_detector.py::TestDetector::test_antenna_pattern_and_time_delay_torch \
     test/test_detector.py::TestDetector::test_network_geometry

The wider search parity selection is listed in :doc:`torch_testing`.
Device-specific CUDA or MPS results require the corresponding hardware; a
device skip must be reported separately from a pass.

Acceptance rules
================

For every compared result, check the scientific value together with all public
metadata: shape, dtype, sample spacing, epoch, frequency support, detector
ordering, and output device where applicable. Search comparisons must also
preserve threshold, clustering, and trigger-selection semantics. Inference
comparisons must use identical detector sets, frequency bins, PSDs, and batch
dimensions.

Reproducible performance evidence additionally records the environment and
timing boundaries required by :doc:`torch_performance`.
