.. _torch-testing:

Testing Torch changes
=====================

Run the focused tests for your changed API, then the affected project suites.
Check scientific values and the implementation that ran before benchmarking.
CUDA and MPS results require those physical devices; report skips separately.

* :ref:`torch-test-groups`: functional test commands.
* :ref:`torch-test-ci`: what the checked-in CPU and GPU jobs cover.
* :ref:`torch-parity-matrix`: prepare and run a reproducible parity campaign.
* :ref:`torch-test-evidence`: check artifact tools and documentation.

Scheme selection in tests
-------------------------

Choose test paths from :ref:`torch-test-groups` below. The focused CI selections
set the test-only ``PYCBC_TEST_SCHEME`` variable:

.. code-block:: console

   PYCBC_TEST_SCHEME=torch:cpu python -m pytest -q TEST_PATHS
   PYCBC_TEST_SCHEME=torch:cuda python -m pytest -q TEST_PATHS

``parse_args_all_schemes`` in ``test/utils.py`` reads this default, including the
complete Torch device name; an explicit ``--scheme`` takes precedence. It is
separate from the runtime default ``PYCBC_SCHEME`` in :ref:`torch-runtime`.
CPU-only helpers and tests with explicit contexts or device fixtures retain
their own choices.

Domain-compatibility, relative-binning and detector tests collect separate CPU
and CUDA cases. Prior tests add CUDA cases only when CUDA is available; absent
cases do not count as skips. The CUDA workflow prints case identifiers and skip
reasons and runs CPU references alongside CUDA cases. Only executed CUDA cases
provide CUDA coverage; assert the implementation and output device as well.
Clear unrelated Torch feature variables and skip clearly when a required
physical device is unavailable. Keep skips separate from passes.

.. _torch-test-groups:

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
     test/test_torch_runtime_transfers.py \
     test/test_torch_batched_fft.py \
     test/test_torch_fft_writes.py \
     test/test_torch_fft_cpu_native.py \
     test/test_torch_fft_cuda_workspace.py \
     test/test_torch_psd_pipeline.py \
     test/test_strain_psd_precision.py \
     test/test_sigmasq_series_precision.py \
     test/test_chisq_precision.py \
     test/test_torch_filter_pipeline.py \
     test/test_torch_event_kernels.py \
     test/test_torch_event_pipeline.py \
     test/test_torch_peak_contracts.py \
     test/test_torch_cuda_native_peaks.py \
     test/test_live_batch_torch_fft_integration.py \
     test/test_live_batch_torch_peaks.py \
     test/test_torch_batch_overlap_scaling.py \
     test/test_torch_large_ifft.py \
     test/test_torch_decompress_cpu.py \
     test/test_decompress.py

The native waveform registry, supported TaylorF2-family ports, and batch
contract are checked with:

.. code-block:: console

   python -m pytest -q \
     test/waveform/test_torch_waveform_registry.py \
     test/waveform/test_taylorf2_torch.py \
     test/waveform/test_taylorf2_phase_evaluation.py \
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

.. _torch-large-batches:

Large batches
-------------

For 1,024-row boundary and reuse checks across FFT, filtering and inference,
run the dedicated suite on a machine with the required devices:

.. code-block:: console

   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     python -m pytest -q -rs test/test_torch_large_batches.py

It uses short transforms to keep memory bounded. These are correctness and
reuse checks. Completed batch-size measurements and their separate numerical
qualification are in :ref:`torch-batch-throughput` and
:ref:`torch-batch-numerics`.
The public live-filter check also verifies that disabled chi-square vetoes
return zero values without raising an exception.

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
   Use the established CPU scheme at the same source revision as the control.
   Record ``TorchScheme``, BLAS and process thread settings, affinity, topology,
   FFT library route, and oversubscription controls. Exercise hardware-aware
   defaults and explicit batch/thread overrides separately.

CUDA
   Verify availability and the requested device index. Synchronize timed work,
   distinguish generic, native, compiled, and graph implementations, and test
   eligibility, fallback, and memory/OOM boundaries. Record the Torch build,
   driver, runtime, GPU model and memory; link the completed device job.

MPS
   Use real Apple hardware and record macOS, SoC, memory, Python, and Torch.
   Test supported single precision, explicit tolerances, double-precision
   rejection, synchronization, and the CPU-staged promoted FFT exception.
   A skip elsewhere does not provide MPS coverage.

Waveform fallback
   Assert whether the native registry or an established host implementation
   ran. A final tensor on the requested device is insufficient to prove native
   generation.

For every performance run, preserve machine-readable test and parity output
beside the raw timing artifact. Count pass, fail, expected-unsupported, and
skip separately by device. Follow :ref:`torch-parity` for the numerical
contract and :ref:`torch-performance` for provenance and publication.

.. _torch-test-ci:

Checked-in CI matrix
--------------------

The checked-in workflows provide the coverage below. Retain each completed
job's exact versions, hardware, implementations, and test results.

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

The self-hosted CUDA job accepts trusted revisions and is not a pull-request
gate. Its result applies to the recorded Python, Torch, driver, CUDA runtime,
GPU, and source revision.

Development validation on ``len`` used Torch ``2.13.0+cu130`` with CUDA 13.0.
Those results do not qualify the CUDA workflow's Torch ``2.13.0+cu126`` with
CUDA 12.6; retain qualification of that exact environment separately. The GPU
selection includes the complete backend-protocol, search-kernel, filter-pipeline,
live-batch peak, and chi-squared optimization test files so their CUDA cases run
alongside CPU references. Explicit CPU-only tests in those files remain on CPU.

.. _torch-parity-matrix:

Parity matrix
-------------

The scripts under ``tools/torch_parity`` compare deterministic CPU and Torch
artifacts with policy-scoped tolerances. The runner requires two clean, identified
worktrees, separate interpreters, a dependency fingerprint, and a sealed
deployment manifest. ``manifest.py prepare`` records these inputs after
installation and building; the runner verifies them before generating a corpus.
Preparation does not install, rebuild, or repair environments.

Create baseline and current worktrees at explicit commits, with matching Python
and dependency versions, including Torch. Keep each environment outside its
source tree. See :ref:`torch-runtime` for installation prerequisites.

.. code-block:: console

   campaign=/absolute/path/to/new-campaign
   mkdir -p "$campaign"
   git worktree add --detach "$campaign/original" BASELINE_SHA
   git worktree add --detach "$campaign/current" CURRENT_SHA
   python3 -m venv "$campaign/original-venv"
   python3 -m venv "$campaign/current-venv"

Install the same resolved runtime and build requirements, including the chosen
Torch wheel, into both environments. Install each checkout in editable mode
and finish native builds before preparing manifests:

.. code-block:: console

   requirements=/absolute/path/to/common-resolved-requirements.txt
   for label in original current; do
     "$campaign/$label-venv/bin/python" -m pip install -r "$requirements"
     "$campaign/$label-venv/bin/python" -m pip install \
       --no-deps --no-build-isolation -e "$campaign/$label"
   done

Use requirements for the chosen operating system and Torch wheel. Preparation
rejects differing package inventories, excluding PyCBC because its source
revisions are compared separately. It records installed versions without locking
wheel bytes or operating-system dependencies. With both sources clean, prepare
new manifests using the current interpreter:

.. code-block:: console

   export ORIGINAL_SOURCE="$campaign/original"
   export CURRENT_SOURCE="$campaign/current"
   export ORIGINAL_PYTHON="$campaign/original-venv/bin/python"
   export CURRENT_PYTHON="$campaign/current-venv/bin/python"
   export DEPENDENCIES_FILE="$campaign/dependencies.json"
   export DEPLOYMENT_FILE="$campaign/deployment.json"
   "$CURRENT_PYTHON" "$CURRENT_SOURCE/tools/torch_parity/manifest.py" prepare \
     --original-source "$ORIGINAL_SOURCE" --current-source "$CURRENT_SOURCE" \
     --original-python "$ORIGINAL_PYTHON" --current-python "$CURRENT_PYTHON" \
     --dependencies "$DEPENDENCIES_FILE" --deployment "$DEPLOYMENT_FILE"
   RESULTS_ROOT="$campaign/results" \
     bash "$CURRENT_SOURCE/tools/torch_parity/run_matrix.sh"

The dependency file records installed versions and runtime. The sealed
deployment file records source/import revisions, interpreter paths, and hashes
of that file and ignored build artifacts, including native extensions but
excluding bytecode and known tool caches. Keep manifests and environments
outside both worktrees. Preparation refuses to overwrite evidence; after a
rebuild or environment change, use new paths and retain previous receipts.

Both commands discard ``PYTHONPATH`` and ``PYCBC_*`` for provenance probes;
the runner selects each scheme. Set thread limits and ``CUDA_VISIBLE_DEVICES``
before preparation and keep them unchanged: both are fingerprinted. Verification
rejects changed commits, dirty sources, wrong editable imports, and changes to
packages, native binaries, runtime, driver, or device. Checksums detect changes;
they do not authenticate the machine or establish scientific correctness.

Preserve ``launch.json``, every generated NPZ/JSON pair, the comparison report,
and the matrix log. A skipped CUDA cell must remain visible as a skip and must
not be presented as device qualification. The runner reports
``matrix_result=PASS_CPU_ONLY`` when CUDA was unavailable.

The matrix's current waveform corpus has one 35+28 solar-mass TaylorF2 case.
It does not supersede ``tools/verify_lal_torch_parity.py``, which retains three
BNS/low-mass TaylorF2 cases and PSD-weighted strain-match checks. Keep both
suites until those cases and acceptance criteria have been migrated. Neither
suite alone establishes parity across the entire physical parameter space.

.. _torch-test-evidence:

Evidence-tool tests
-------------------

These device-independent tests validate artifact schemas, percentile and
bootstrap summaries, content seals, production-artifact parsing, conditional
plots, route labels, cold/warm separation, parity disposition, and CUDA
allocated/reserved-memory metadata:

.. code-block:: console

   python -m pytest -q test/test_torch_performance_artifacts.py \
     test/test_torch_parity_artifacts.py

Negative controls reject empty records/corpora, nonfinite values, NumPy
fallback when Torch storage was requested, and mismatched CUDA device indices.
Output placement alone does not establish that all intermediates stayed on the
device.

Run syntax, configuration, and documentation checks with the repository
environment:

.. code-block:: console

   python -m py_compile \
     tools/benchmark_artifact.py \
     tools/bench_production_live_batch.py \
     tools/generate_torch_performance_plots.py \
     tools/torch_parity/compare.py \
     tools/torch_parity/generate.py \
     tools/torch_parity/manifest.py
   bash -n tools/torch_parity/run_matrix.sh
   python -m json.tool tools/torch_parity/policy.json >/dev/null
   pixi run -e docs test-docs

These checks establish that the evidence machinery is internally consistent.
They do not establish a speedup, scientific parity, or accelerator residency.
