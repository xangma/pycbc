.. _torch-workflows:

Torch maintainer workflow
=========================

Torch changes move through capability, scientific parity, integration, device
qualification, and performance evidence in that order. A later stage cannot
repair a missing earlier one: a microbenchmark does not establish correctness,
and output copied to an accelerator does not establish a native route.

Validation stages
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 29 35 18

   * - Stage
     - Question
     - Required evidence
     - Authority
   * - Capability
     - Is this operation and interface implemented, and what happens when it
       is not eligible?
     - Public capability/fallback entry, predicates, dtype/device boundaries,
       and an explicit unsupported result where no fallback exists.
     - :ref:`torch-scheme` and focused tests.
   * - Focused parity
     - Does the route preserve scientific values and public metadata?
     - Independent reference or justified invariant, explicit error metric,
       route assertion, output device, and fallback/error cases.
     - :ref:`torch-parity`
   * - Composed integration
     - Do adjacent array, FFT, filtering, waveform, and inference paths preserve
       state and residency?
     - Deterministic pipeline tests, repeated/context reuse, host-conversion
       guards, and affected project suites.
     - :ref:`torch-testing`
   * - Device qualification
     - Does the exact CPU, CUDA, or MPS stack behave as claimed?
     - A real-device CI/job record with versions, hardware, passes, skips, and
       route results.
     - :ref:`torch-testing`
   * - Performance
     - Is a qualified route useful for the claimed workload without hidden
       tail, transfer, memory, or cold-start cost?
     - Raw samples, required plots, parity rows, and durable CI provenance.
     - :ref:`torch-performance`
   * - Promotion
     - Is evidence broad enough to change a default?
     - Scoped matrix coverage, automatic device CI, fallback and rollback,
       regression bounds, and review of every artifact.
     - :ref:`torch-optimizations`

Change workflow
---------------

#. **Define the public boundary.** Name the operation, devices, dtypes, shapes,
   autograd expectation, metadata, and output-device contract. Update the
   capability matrix when support or fallback behavior changes.
#. **Implement explicit eligibility.** Keep optional fast paths behind the
   narrowest predicate and retain an established fallback. If no correct
   fallback exists, raise a clear unsupported error instead of performing an
   invisible host conversion.
#. **Add focused tests.** Exercise the optimization disabled, enabled and
   eligible, and enabled but ineligible. Check values, metadata, route, device,
   empty/boundary input, and repeated state.
#. **Run composed and broad suites.** Use the focused groups and project tasks
   in :ref:`torch-testing`. Keep pass, fail, skip, and expected-unsupported
   counts separate for each device.
#. **Qualify real hardware.** CPU results include thread and topology metadata;
   CUDA and MPS claims require those physical devices. A skipped conditional
   test on another runner is not qualification.
#. **Benchmark only qualified routes.** Use isolated processes and the complete
   protocol in :ref:`torch-performance`. Preserve all raw samples, parity rows,
   failures, deadline misses, and OOM cells.
#. **Publish traceable evidence.** Attach the immutable artifact and checksum,
   exact command, workflow/job URL, runner metadata, source revision, and plot
   generator. A checked-in PNG alone is not measured evidence.
#. **Review promotion separately.** Defaults-off flags stay experimental until
   the promotion policy is met. Limit a promotion to the device/dtype/shape
   envelope actually tested.

Route and fallback review
-------------------------

Reviewers should be able to answer these questions from tests and artifacts:

* Which implementation ran, and how was that asserted?
* Which output device and public dtype were returned?
* Did any input, intermediate, or output cross a host/device boundary?
* What happens with the feature disabled?
* What happens just outside every eligibility boundary?
* Is fallback scientifically equivalent, and is it visible in logs/metrics?
* Does plan, allocator, compiler, graph, or thread state leak to later calls?

The capability matrix in :ref:`torch-scheme`, implementation registries,
support predicates, and focused tests define the supported boundary. Existing
host generation followed by a device copy is a fallback, not native evidence,
regardless of the final output device.

Device-specific workflow
------------------------

Torch CPU
   Keep the established CPU scheme as the current-tree control. Record
   ``TorchScheme``, BLAS, and process thread settings, affinity, topology, FFT
   library route, and oversubscription. Exercise the hardware-aware defaults
   and explicit batch/thread overrides separately.

CUDA
   Verify availability and the explicit device index before testing. Record
   Torch build, driver/runtime, GPU model and memory. Synchronize measurements,
   identify generic versus native/compiled/graph routes, and test memory/OOM
   boundaries. The checked-in CUDA workflow is weekly on the trusted default
   branch and manually dispatchable, but not a pull-request gate; link the
   exact completed run explicitly.

MPS
   Use real Apple hardware and record macOS, SoC, memory, Python, and Torch.
   Exercise supported single precision, double-precision rejection/guard
   behavior, asynchronous synchronization, and the CPU-staged promoted FFT
   exception. No dedicated checked-in MPS CI lane currently establishes this
   automatically.

Optimization experiments
------------------------

Start from the documented default environment. Change one flag at a time,
record all variables, and use a fresh process for each route because compiler,
FFT, allocator, graph, and dispatch state can persist.

For interactions that are intentionally supported, add a separate matrix cell
instead of enabling several flags in a single opaque “optimized” configuration.
``PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK`` must be set explicitly because its unset
behavior differs between threshold reduction and separate peak extraction.
Compilation measurements must preserve both cold and warm samples.

The complete flag inventory, defaults, precedence, and promotion criteria are
in :ref:`torch-optimizations`.

Publishing benchmark material
-----------------------------

Before regenerating plots:

#. finish the parity and suite runs for the same revision and environment;
#. validate the raw artifact schema, checksums, workload identities, route
   fields, and CI provenance;
#. derive the required throughput, tail-latency, crossover, transfer/resident,
   memory/OOM, cold/warm, jitter, parity/error, CPU-scaling, and hardware views;
#. render into a review directory with the command documented in
   :ref:`torch-performance`; and
#. publish raw artifacts with the images and review every changed caption.

No generated performance plot is published by this layer. The expected output
inventory in :ref:`torch-optimization-results` remains unpublished until each
image has a traceable raw artifact and CI or runner provenance.

Failure triage
--------------

Numerical failure
   Disable optional routes individually, reproduce with the exact dtype and
   shape, compare metadata before values, and locate the first divergent stage.

Unexpected host conversion
   Instrument conversion boundaries, inspect waveform fallback and MPS
   promoted FFT behavior, and reduce the claim to the region actually resident.

Unexpected route
   Capture every feature variable and construction argument, then check device,
   dtype, shape, version, and autograd predicates. An enabled flag is only a
   request.

Flaky latency
   Preserve timestamped raw samples; inspect warm-up, compilation, clocks,
   affinity, competing load, synchronization, allocation, memory pressure, and
   deadline misses before summarizing.

Device-only failure
   Report the actual hardware/software matrix and skip disposition. Do not
   weaken shared tolerances until the route's precision contract and independent
   error distribution have been established.
