.. _torch-testing:

Testing Torch changes
=====================

Torch validation is layered: focused numerical and route tests, composed
module tests, real-device qualification, parity artifacts, and only then
performance measurements. A benchmark does not replace correctness tests, and
a device skip is not a pass for that device.

Scheme selection in tests
-------------------------

The focused CI selections set the test-only ``PYCBC_TEST_SCHEME`` variable:

.. code-block:: console

   PYCBC_TEST_SCHEME=torch:cpu python -m pytest -q TEST_PATHS
   PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q TEST_PATHS

This is separate from the runtime default ``PYCBC_SCHEME`` described in
:ref:`torch-scheme`. ``parse_args_all_schemes`` in ``test/utils.py`` reads this
variable as its default, including the complete Torch device name. An explicit
``--scheme`` argument takes precedence. CPU-only test helpers continue to run
on CPU. Tests with their own explicit Torch contexts or device fixtures retain
those choices; the variable does not override them.

Domain-compatibility and relative-binning tests now have separately collected
CPU and CUDA cases, as do the detector and prior tests. The CUDA workflow runs
both CPU reference cases and CUDA cases and prints their identifiers and skip
reasons. Other selected tests may still have explicitly CPU-only contexts.
The lane therefore qualifies the CUDA cases it actually executes, not every
test in each selected file. Route and output-device assertions remain required.

Tests should clear unrelated Torch feature variables so a developer's shell
does not change route selection. A device-specific test must skip clearly when
that physical device is unavailable and must never count that skip as
qualification.

Evidence-tool tests
-------------------

The pure artifact tests do not require a Torch device or execute PyCBC's
scientific runtime. They validate the schema, percentile and bootstrap
summaries, content seal, production-artifact parsing, conditional plot
generation, route labels, cold/warm separation, parity disposition, and CUDA
allocated/reserved-memory metadata:

.. code-block:: console

   python -m pytest -q test/test_torch_performance_artifacts.py \
     test/test_torch_parity_artifacts.py

The parity negative controls reject empty records/corpora, any nonfinite
values, NumPy fallback when Torch storage was requested, and a mismatched CUDA
device index. A returned Torch tensor establishes output placement only; it
does not establish that every intermediate was resident on that device.

Run syntax, configuration, and documentation checks with the repository
environment:

.. code-block:: console

   python -m py_compile \
     tools/benchmark_artifact.py \
     tools/bench_production_live_batch.py \
     tools/generate_torch_performance_plots.py \
     tools/torch_parity/compare.py \
     tools/torch_parity/generate.py
   bash -n tools/torch_parity/run_matrix.sh
   python -m json.tool tools/torch_parity/policy.json >/dev/null
   pixi run -e docs test-docs

These checks establish that the evidence machinery is internally consistent.
They do not establish a speedup, scientific parity, or accelerator residency.

Checked-in CI matrix
--------------------

This table describes workflows in this source tree, not a universal support
statement. Exact versions, hardware, routes, and test results from a completed
job are the evidence for that run.

.. list-table::
   :header-rows: 1
   :widths: 24 18 22 36

   * - Lane
     - Python
     - Device
     - Qualification
   * - General Linux tests
     - 3.11, 3.12, 3.13
     - CPU
     - Broad PyCBC coverage, followed in unit-test jobs by a focused Torch CPU
       selection with ``PYCBC_TEST_SCHEME=torch:cpu``.
   * - General macOS tests
     - 3.11, 3.12, 3.13
     - CPU
     - Broad macOS coverage; it is not a dedicated MPS-device lane.
   * - Trusted Torch GPU
     - Workflow environment
     - Self-hosted Linux CUDA
     - Weekly on the default branch and manually dispatched. It requests the
       exact Torch/CUDA wheel declared in the workflow, verifies real CUDA
       availability, and runs CUDA regressions alongside CPU reference cases.
       Manual custom selectors qualify only the cases selected.
   * - MPS
     - Not dedicated
     - Apple MPS
     - Conditional tests can run where MPS is available, but no checked-in
       workflow qualifies MPS automatically.

The self-hosted CUDA job is scheduled and manual rather than a pull-request
gate. It is limited to trusted revisions so untrusted pull-request code is not
run on the repository's GPU host. A retained green run qualifies only the
resolved Python, Torch, driver, CUDA runtime, GPU, and source revision recorded
by that job.

Development validation on ``len`` used Torch ``2.13.0+cu130`` with CUDA 13.0.
Those results do not qualify the CUDA workflow's Torch ``2.13.0+cu126`` with
CUDA 12.6; retain qualification of that exact environment separately. The GPU
selection includes the complete backend-protocol, search-kernel, filter-pipeline,
live-batch peak, and chi-squared optimization test files so their CUDA cases run
alongside CPU references. Explicit CPU-only tests in those files remain on CPU.

Focused functional groups
-------------------------

The CPU workflow keeps the stack's focused runtime, array, FFT, filter, search,
decompression, and native-waveform tests together. Representative paths are:

.. code-block:: console

   python -m pytest -q \
     test/test_scheme_runtime.py \
     test/test_torch_optional.py \
     test/test_array_torch_reductions.py \
     test/test_torch_backend_protocol.py \
     test/test_torch_batched_fft.py \
     test/test_torch_filter_pipeline.py \
     test/test_live_batch_torch_fft_integration.py \
     test/test_live_batch_torch_peaks.py \
     test/test_decompress.py

The native waveform registry, supported TaylorF2-family ports, and batch
contract are checked with:

.. code-block:: console

   python -m pytest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2ecc_torch.py \
     test/waveform/test_taylorf2nltides_torch.py \
     test/waveform/test_taylorf2redspin_torch.py \
     test/waveform/test_taylorf2_batch.py

The slim domain, detector-geometry, and inference layers use focused tests that
match their public boundaries:

.. code-block:: console

   PYCBC_TEST_SCHEME=torch python -m pytest -q \
     test/test_torch_domain_compat.py \
     test/test_torch_prior_compat.py \
     test/test_detector_torch_geometry.py \
     test/test_torch_inference_core.py \
     test/test_torch_inference_tools.py \
     test/test_torch_waveform_generator.py \
     test/test_torch_gaussian_noise.py \
     test/test_torch_relative_binning.py \
     test/test_torch_marginalized_gaussian.py \
     test/test_torch_inference_cli.py

The checked-in workflow files are authoritative when paths evolve. Before a
performance run, execute the smallest relevant focused group, then every
project suite affected by the changed public path.

Parity matrix
-------------

The scripts under ``tools/torch_parity`` generate deterministic CPU and Torch
artifacts, compare them with policy-scoped tolerances, and preserve manifests.
The matrix runner requires two clean, identified worktrees, separate Python
interpreters, a dependency fingerprint, and a sealed deployment manifest. It
refuses to manufacture missing provenance.

A prepared campaign can be launched with:

.. code-block:: console

   ORIGINAL_SOURCE=/path/to/baseline \
   CURRENT_SOURCE=/path/to/current \
   ORIGINAL_PYTHON=/path/to/baseline/python \
   CURRENT_PYTHON=/path/to/current/python \
   DEPENDENCIES_FILE=/path/to/dependencies.json \
   DEPLOYMENT_FILE=/path/to/deployment.json \
   tools/torch_parity/run_matrix.sh

Preserve ``launch.json``, every generated NPZ/JSON pair, the comparison report,
and the matrix log. A skipped CUDA cell must remain visible as a skip and must
not be presented as device qualification. The runner reports
``matrix_result=PASS_CPU_ONLY`` when CUDA was unavailable.

The matrix's current waveform corpus has one 35+28 solar-mass TaylorF2 case.
It does not supersede ``tools/verify_lal_torch_parity.py``, which retains three
BNS/low-mass TaylorF2 cases and PSD-weighted strain-match checks. Keep both
suites until those cases and acceptance criteria have been migrated. Neither
suite alone establishes parity across the entire physical parameter space.

What each optimization test must prove
--------------------------------------

Every default-off or newly promoted route from :ref:`torch-optimizations`
needs:

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Case
     - Required assertion
   * - Flag disabled
     - The established route retains its numerical and metadata contract.
   * - Flag enabled, eligible
     - The intended route runs and passes values, metadata, output-device, and
       residency checks.
   * - Flag enabled, ineligible
     - The documented fallback or explicit unsupported error occurs without a
       partial result or silent dtype change.
   * - Boundaries
     - Empty/minimal and non-contiguous inputs, dtype limits, batch and FFT-size
       boundaries, threshold equality, and memory-budget edges.
   * - State reuse
     - Repeated calls, changed shapes, plan/cache reuse, context exit, and
       feature-state isolation behave consistently.
   * - Autograd
     - Where the public API promises gradients, forward and backward paths
       pass. Tests must not imply whole-workflow autograd otherwise.

Compilation tests must separate first-use compilation from cache reuse. CUDA
graph and stream tests require real CUDA and must verify capture/replay or
stream selection, not merely numerical output.

Device and fallback rules
-------------------------

CPU
   Record Torch and BLAS thread counts, affinity, topology, and
   oversubscription controls.

CUDA
   Require the requested device index, synchronize asynchronous work around
   timed regions, exercise optimized eligibility and fallback separately, and
   record the exact Torch, driver, runtime, and GPU stack.

MPS
   Require a real available MPS device, use supported dtypes and explicit
   tolerances, and report staged CPU work. A skip elsewhere does not provide
   MPS coverage.

Waveform fallback
   Assert whether the native registry or an established host implementation
   ran. A final tensor on the requested device is insufficient to prove native
   generation.

For every performance run, preserve machine-readable test and parity output
beside the raw timing artifact. Count pass, fail, expected-unsupported, and
skip separately by device. Follow :ref:`torch-parity` for the numerical
contract and :ref:`torch-performance` for provenance and publication.
