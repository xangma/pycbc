.. _jax-testing:

JAX Testing & Quality Verification
==================================

PyCBC includes automated unit tests, regression suites, and parity gates for
the JAX backend.

Running Unit Tests
------------------

Run the complete JAX test suite locally:

.. code-block:: console

   pytest test/test_jax_*.py test/test_detector_jax_*.py

Testing with 64-bit Precision
-----------------------------

All tests enable 64-bit floating point mode (`jax_enable_x64=True`) to verify
double-precision equivalence with CPU and LALSuite reference calculations.

CI Verification
---------------

GitHub Actions CI runs the full JAX test suite on CPU and dedicated GPU runners,
ensuring strict compliance with PyCBC quality gates (0 unused imports, 100% test pass).
