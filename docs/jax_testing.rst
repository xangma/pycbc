.. _jax-testing:

JAX testing and validation
==========================

PyCBC includes unit tests and regression checks for the JAX backend. Report
the tested revision, Python/JAX versions, selected device, failures and skips
with each test result; an available test suite does not establish that every
configuration passes.

Running tests
-------------

Use an environment with PyCBC's dependencies and compiled extensions matching
its Python version. Select the CPU scheme explicitly:

.. code-block:: console

   PYCBC_TEST_SCHEME=jax:cpu pytest \
     test/test_jax_*.py test/test_detector_jax_*.py \
     test/waveform/test_jax_*.py test/waveform/test_spherical_harmonics_jax.py

On a configured CUDA host, use ``PYCBC_TEST_SCHEME=jax:cuda`` for tests that
honor this variable. Some tests choose their own device explicitly, including
CPU, so the variable does not force the entire suite onto CUDA. Check
``jax.devices()`` and the individual tests' scheme selection before treating
a result as GPU validation.

Precision and scientific scope
------------------------------

Tests that need double precision enable ``jax_enable_x64``. Other checks
exercise single-precision search arrays. A passing double-precision test
does not imply bitwise agreement in a single-precision production search.
Complete executable qualification is described in
:ref:`jax-reference-campaign`; its scientific comparisons are separate from
unit tests and performance measurements.

Continuous integration
----------------------

``.github/workflows/basic-tests.yml`` selects ``jax:cpu`` for its JAX suite.
``.github/workflows/jax-gpu.yml`` schedules the CUDA suite on a self-hosted
Linux GPU runner and also supports manual test selection. A configured GPU
workflow still requires an available compatible runner. Consult the actual
workflow result for pass/fail status and skips.


.. _jax-testing-parity:
.. _jax-parity:

Numerical comparison helpers
----------------------------

``tools/jax_parity/`` compares a fixed synthetic corpus between the standard
CPU and JAX execution schemes. It covers array arithmetic, FFT conversion,
frequency shifts, analytical and Welch PSDs, one TaylorF2 waveform, matched
filtering and match interpolation. It does not exercise every module,
chi-squared vetoes, inference likelihoods or complete executable searches.

Comparison policy
~~~~~~~~~~~~~~~~~

``policy.json`` is the authoritative set of acceptance targets. The ``jax``
profile uses relative L2 error, with record-specific limits ranging from
``1e-13`` for array arithmetic to ``3e-5`` for the interpolated match value.
Selected records also require elementwise ``allclose`` or exact zero patterns.
The comparator checks record sets, shape, dtype, series metadata and finiteness.
These are targets for this corpus, not a claim that production searches meet
those limits or produce identical triggers.

Complete-search qualification uses the separate frozen gates in
:ref:`jax-reference-campaign`. The fresh 384-template CPU/JAX comparisons
fail those gates; see :ref:`jax-search-qualification` for measured differences.
Passing a helper corpus cannot replace that search-level qualification.

Running the helper corpus
~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper generates standard CPU and JAX CPU artifacts in separate
processes using the current checkout, then writes a JSON comparison report:

.. code-block:: console

   CURRENT_PYTHON=python OUTPUT_DIR=artifacts/jax_parity \
     bash tools/jax_parity/run_matrix.sh

Use ``generate.py --help`` to select another available JAX device explicitly.
Keep both generated manifests and NumPy archives together with the report;
the manifests record source, environment and device information.
