=======================
Torch parity validation
=======================

Torch parity is tested as a differential matrix, not by changing schemes in
one Python process.  Separate processes prevent PyCBC's global scheme state,
backend dispatch caches, FFT plans, and device allocations from leaking from
one cell into another.

This page defines and runs the parity test. Current measured outcomes are
summarized in :doc:`torch_optimization_results`; the full historical evidence
is in the repository benchmark archives.

Reading a parity result
=======================

The reports distinguish three claims:

* **Raw-byte exact** means all compared bytes match, including signed zero.
* **Comparator-equivalent** means structure, dense-array errors, and triggers
  pass ``tools/torch_parity/policy.json``; it does not mean byte identity.
* **Route-qualified** means the result also passed source, dependency, backend,
  device, and LAL-independence assertions.

Performance is reported only after the applicable parity claim passes. A
parity pass does not itself imply that a route is faster.

The matrix
==========

.. list-table::
   :header-rows: 1

   * - Cell
     - Source
     - Scheme/device
     - LALSimulation
     - Purpose
   * - A
     - Frozen pre-Torch fork point
     - CPU
     - Available
     - Historical reference
   * - B
     - Current Torch branch
     - CPU
     - Available
     - Detect CPU regressions independently of Torch
   * - C
     - Current Torch branch
     - Torch CPU
     - Import blocked
     - Isolate backend differences without GPU variation
   * - D
     - Current Torch branch
     - Torch CUDA
     - Import blocked
     - Validate the production accelerator path

The primary comparisons are A-to-B, B-to-C, and B-to-D.  C-to-D is also
reported to expose device-specific numerical drift.  The frozen baseline is
``b551de0d5334d7b5ed07ac775aee1351e41817db``, the merge base at which the
Torch conversion branch diverged from ``master``.

Only two virtual environments are required.  A and B need different PyCBC
source installations, while B, C, and D deliberately use the same current
installation in isolated subprocesses.  Both virtual environments inherit
the same immutable base conda environment, so NumPy, SciPy, LALSuite, Torch,
CUDA libraries, and every other dependency are identical.  This is stricter
and smaller than maintaining three independently solved conda environments.

Artifacts and acceptance criteria
=================================

``tools/torch_parity/generate.py`` runs a deterministic API-intersection
corpus covering array arithmetic, FFTs, frequency-domain time shifts, FIR and
Butterworth filtering, Welch and analytical PSDs, TaylorF2 and IMRPhenomD
sentinels, and matched filtering.  Each cell writes:

* an NPZ file containing result arrays;
* a JSON manifest containing exact shape, dtype, epoch, sampling metadata,
  storage backend, dependency versions, source revision, and timings.

``tools/torch_parity/compare.py`` first requires structural metadata parity,
then applies the numerical policy in ``tools/torch_parity/policy.json``.  It
reports relative L2, maximum absolute, and maximum relative errors.  The
Torch tolerances come from the corresponding focused unit-test tolerances;
waveforms additionally require identical zero support.

Cells C and D install an import blocker before importing PyCBC.  Any attempted
``lalsimulation`` import fails immediately, and the manifest records that it
remained absent.  This turns independence into an executable assertion rather
than an inference from output parity.

This corpus is a cross-version smoke gate.  It complements, rather than
replaces, the full unit suite and the focused no-LALSimulation tests in
``test/test_torch_core_lal_independence.py``.

Setup on ``len``
================

From a local checkout of the exact revision to test, choose a unique root and
stream the one-shot setup script to ``len``.  The selected root must not exist;
after a failed or completed setup, use a new root rather than resuming it:

.. code-block:: bash

   CURRENT_COMMIT=$(git rev-parse HEAD)
   RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
   ROOT_NAME="repos/pycbc-parity-${CURRENT_COMMIT:0:12}-${RUN_ID}"
   ssh len "PYCBC_PARITY_ROOT=\$HOME/$ROOT_NAME \
       CURRENT_COMMIT=$CURRENT_COMMIT bash -s" \
       < tools/torch_parity/setup_len.sh

The setup creates exact detached original/current worktrees and two lightweight
virtual environments based on ``~/miniconda3/envs/pycbcgpu``.  It verifies
identical dependency fingerprints, exact imported PyCBC source revisions, and
clean tracked/untracked source state.  ``deployment.json`` seals an inventory
of every ignored regular file in both source trees except explicitly harmless
caches: ``.hypothesis``, ``.mypy_cache``, ``.nox``, ``.pytest_cache``,
``.ruff_cache``, ``.tox``, and ``__pycache__`` directories, the ``.coverage``
file, and ``.pyc``/``.pyo`` files.  The inventory records paths, sizes, and
SHA256 hashes alongside the source/import revisions and SHA256 of
``dependencies.json``.  Both manifests are read-only evidence.  Every matrix
launch recomputes them, re-probes the active Python, Torch/CUDA/device/driver
state and installed-package fingerprint, compares material dependency state
with setup, and saves a sealed ``launch.json`` before running.  The setup
installs identical, pinned ``igwn-ligolw==2.1.1`` and
``igwn-segments==2.1.1`` wheels into both virtual environments without
modifying the base environment.

Run the complete matrix with:

.. code-block:: bash

   ssh len "root=\$HOME/$ROOT_NAME; \
       PYCBC_PARITY_ROOT=\$root \
       \$root/current/tools/torch_parity/run_matrix.sh"

Results are stored below
``$HOME/repos/pycbc-parity-<commit>-<run-id>/results/<UTC timestamp>/``.
``matrix.log`` is the human-readable transcript; ``compare-*.json`` files are
machine-readable reports.  The dependency, deployment, and sealed launch
manifests are stored beside them.  A nonzero exit status means provenance or
runtime verification failed, or at least one comparison failed.

To test another current revision without overwriting evidence from the first
run, select a new root:

.. code-block:: bash

   CURRENT_COMMIT=$(git rev-parse HEAD)
   NEXT_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
   NEXT_ROOT_NAME="repos/pycbc-parity-${CURRENT_COMMIT:0:12}-${NEXT_RUN_ID}"
   ssh len "PYCBC_PARITY_ROOT=\$HOME/$NEXT_ROOT_NAME \
       CURRENT_COMMIT=$CURRENT_COMMIT bash -s" \
       < tools/torch_parity/setup_len.sh

The setup does not modify the base conda environment.  Rollback consists of
removing the dedicated parity root after preserving any desired result
artifacts.
