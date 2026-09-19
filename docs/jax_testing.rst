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

On a configured CUDA host, use ``PYCBC_TEST_SCHEME=jax:cuda``. Check
``jax.devices()`` before interpreting a run as GPU validation.

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
