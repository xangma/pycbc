.. _torch-testing:

Testing Torch performance evidence
==================================

Torch validation is layered: focused numerical and route tests, composed
module tests, real-device qualification, parity artifacts, and only then
performance measurements. A benchmark does not replace tests, and a device
skip is not a pass.

Evidence-layer tests
--------------------

The pure artifact tests do not require a Torch device or execute PyCBC's
scientific runtime. They validate the schema, percentile and bootstrap
summaries, content seal, production-artifact parsing, conditional plot
generation, route labels, cold/warm separation, parity disposition, and CUDA
allocated/reserved-memory metadata:

.. code-block:: console

   python -m pytest -q test/test_torch_performance_artifacts.py

Run configuration and documentation checks with the repository environment:

.. code-block:: console

   python -m py_compile \
     tools/benchmark_artifact.py \
     tools/bench_production_live_batch.py \
     tools/generate_torch_performance_plots.py \
     tools/torch_parity/compare.py \
     tools/torch_parity/generate.py
   pixi run -e docs test-docs

These checks establish that the evidence machinery is internally consistent.
They do not establish a PyCBC speedup or scientific parity for an optimized
route.

Checked-in CI matrix
--------------------

This table describes workflows in this source tree, not a universal support
statement. Exact versions and hardware from a completed job are the evidence
for that run.

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
     - Broad PyCBC coverage. This evidence branch adds no general CPU Torch
       selector, so the lane alone does not qualify Torch performance.
   * - General macOS tests
     - 3.11, 3.12, 3.13
     - CPU
     - Broad macOS coverage; it is not a dedicated MPS lane.
   * - Trusted Torch GPU
     - Environment-resolved
     - Self-hosted Linux CUDA
     - Weekly on the default branch and manually dispatched. It installs the
       exact Torch/CUDA wheel declared in the workflow, verifies CUDA
       availability, and runs the focused CUDA regression set.
   * - MPS
     - Not dedicated
     - Apple MPS
     - No checked-in workflow qualifies MPS automatically.

The self-hosted job is scheduled and manual rather than a pull-request gate.
It is restricted to trusted revisions so untrusted pull-request code is not
run on the repository's GPU host. A retained green run qualifies only the
resolved Python, Torch, driver, CUDA, device, and source revision recorded by
that job.

Focused functional prerequisites
--------------------------------

Before measuring a route, run the smallest applicable correctness group from
the implementation stack. The PR4 baseline used by this evidence branch
contains these CPU-capable Torch groups:

.. code-block:: console

   python -m pytest -q \
     test/test_array_torch_reductions.py \
     test/test_torch_batched_fft.py \
     test/test_matched_filter_symm.py \
     test/test_live_batch_torch_fft_integration.py \
     test/test_live_batch_torch_peaks.py \
     test/test_torch_filter_pipeline.py

Waveform registry and native-family coverage is available in:

.. code-block:: console

   python -m pytest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2ecc_torch.py \
     test/waveform/test_taylorf2nltides_torch.py \
     test/waveform/test_taylorf2redspin_torch.py \
     test/waveform/test_taylorf2_batch.py

The trusted CUDA workflow additionally selects the CUDA-native batch,
threshold, chi-squared, live-batch peak, FFT-integration, and CUDA-graph tests
that exist at this branch's PR4 base. It also runs the pure artifact tests.
Inference, detector, and waveform-decompression tests from the original PR5
implementation are intentionally not selected by this standalone evidence
branch.

Exact test paths can evolve. The workflow file and current source tree remain
authoritative; do not copy a stale command into performance provenance.

Parity matrix
-------------

The scripts under ``tools/torch_parity`` generate deterministic CPU and Torch
artifacts, compare them with policy-scoped tolerances, and preserve manifests.
The matrix runner deliberately requires two clean, identified worktrees,
separate Python interpreters, a dependency fingerprint, and a sealed deployment
manifest. It refuses to manufacture either provenance input.

A prepared campaign can be launched with:

.. code-block:: console

   ORIGINAL_SOURCE=/path/to/baseline \
   CURRENT_SOURCE=/path/to/current \
   ORIGINAL_PYTHON=/path/to/baseline/python \
   CURRENT_PYTHON=/path/to/current/python \
   DEPENDENCIES_FILE=/path/to/dependencies.json \
   DEPLOYMENT_FILE=/path/to/deployment.json \
   tools/torch_parity/run_matrix.sh

Preserve ``launch.json``, every generated NPZ/JSON pair, comparison report,
and the matrix log. A skipped CUDA cell must remain visible as a skip and must
not be presented as device qualification.

What each optimization test must prove
--------------------------------------

Every defaults-off or newly promoted route from
:ref:`torch-optimizations` needs:

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
   record the exact Torch/driver/runtime stack.

MPS
   Require a real available MPS device, use supported dtypes and explicit
   tolerances, and report staged CPU work. A skip elsewhere does not provide
   MPS coverage.

For every performance run, preserve machine-readable test and parity output
beside the raw timing artifact. Count pass, fail, expected-unsupported, and
skip separately by device. Follow :ref:`torch-parity` for the numerical
contract and :ref:`torch-performance` for provenance and publication.
