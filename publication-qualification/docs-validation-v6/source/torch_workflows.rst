.. _torch-workflows:

Torch maintainer workflow
=========================

Use this checklist when changing Torch support. The linked guides contain the
test commands, numerical requirements, feature flags, and benchmark protocol.

Change checklist
----------------

#. **Define the supported inputs and outputs.** Record the operation, devices,
   dtypes, shapes, gradients, metadata, and output device. Update the capability
   matrix in :ref:`torch-scheme` when support or fallback behavior changes.
#. **Make implementation selection explicit.** Use narrow eligibility checks
   and an established fallback. Raise a clear unsupported error when no correct
   fallback exists. Tests must observe the implementation that ran and any
   host/device transfers; output placement alone is insufficient.
#. **Check scientific results.** Follow :ref:`torch-parity` for independent
   references, error metrics, tolerances, and metadata checks. Exercise the
   optimization disabled, enabled and eligible, and enabled but ineligible.
   Check empty/boundary inputs and make fallback behavior visible.
#. **Run tests on the claimed devices.** Follow :ref:`torch-testing` for focused
   groups, composed pipelines, repeated/context reuse, affected project suites,
   and real-device CI limits. Check that plan, allocator, compiler, graph, and
   thread state do not leak into later calls. Keep pass, fail, skip, and
   expected-unsupported counts separate for each device.
#. **Measure the tested workload.** Use isolated processes and
   :ref:`torch-performance`. Preserve raw samples, parity rows, failures,
   deadline misses, and OOM cells. Separate cold and warm measurements and
   report transfer, memory, and tail costs for the claimed execution region.
#. **Publish reproducible evidence.** Retain immutable artifacts and validate
   their schemas, checksums, workload identities, implementation fields, and
   CI or runner provenance.
   Include the exact source revision, command, job URL, runner metadata, and
   plot generator. Render into a review directory, publish raw artifacts with
   the figures, and review every caption. Current results are in
   :ref:`torch-optimization-results`.
#. **Review default changes separately.** Follow the promotion policy in
   :ref:`torch-optimizations`, including automatic device CI, regression bounds,
   and a verified fallback and rollback. Limit changes to the device, dtype,
   and shape coverage supported by the evidence.

Testing optional settings
-------------------------

Start from the documented defaults in :ref:`torch-optimizations`. Change one
flag at a time and record all variables. Use a fresh process for each
configuration: compiler, FFT, allocator, graph, and dispatch state can persist.
Test supported flag combinations as separate matrix cells.

Set ``PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK`` explicitly because its unset behavior
differs between threshold reduction and separate peak extraction. Keep both
cold and warm samples when testing compilation.

Failure triage
--------------

Numerical failure
   Disable optional routes individually, reproduce the exact dtype and shape,
   compare metadata before values, and locate the first divergent stage.

Unexpected host conversion
   Instrument conversion boundaries, inspect waveform fallback and MPS
   promoted FFT behavior, and limit the residency claim to the measured region.

Unexpected implementation
   Capture feature variables and constructor arguments, then check device,
   dtype, shape, version, and gradient requirements. A flag requests a route;
   its eligibility checks determine whether it runs.

Flaky latency
   Preserve timestamped samples; inspect warm-up, compilation, clocks, affinity,
   competing load, synchronization, allocation, memory pressure, and deadline
   misses before summarizing.

Device-only failure
   Report hardware, software, and skips. Establish the precision contract and
   independent error distribution before changing shared tolerances.
