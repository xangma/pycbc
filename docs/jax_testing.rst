Testing JAX calculations
========================

Install the normal PyCBC test dependencies and JAX in the same environment.
Run maintained regressions from a checkout, selecting the intended device:

.. code-block:: console

   PYCBC_TEST_SCHEME=jax:cpu python -m pytest -q test/test_jax*.py test/test_gwf_*jax*.py test/test_detector_jax*.py test/waveform/test_jax*.py
   PYCBC_TEST_SCHEME=jax:cuda:0 python -m pytest -q test/test_jax*.py test/test_gwf_*jax*.py test/test_detector_jax*.py test/waveform/test_jax*.py

Tests include independent original calculations, device placement, mutation
and buffer ownership, compilation reuse, search state and output lifetime.
Hardware-specific cases skip when their required device is absent. An absent
optional waveform or decompression provider can also skip its integration
cases; inspect the summary before claiming coverage of those integrations.

The separate JAX CPU and GPU workflows install JAX explicitly. The GPU
workflow first executes and synchronizes a compiled calculation on a GPU,
then requires the detector enqueue/collection and warmed bank regressions to
run without skips. The ordinary CPU workflow keeps its original environment.

For a CUDA check outside CI, establish availability before the test run:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from pycbc.scheme import JAXScheme

   with JAXScheme("cuda:0"):
       value = jax.jit(lambda x: x + 1)(jnp.ones(4))
       value.block_until_ready()
       assert next(iter(value.devices())).platform == "gpu"

The executed notebooks linked by :doc:`jax_numerical_differences` provide
fixed-input numerical examples. Reexecute them after changing the documented
calculation or its dependencies. They complement the maintained tests rather
than replace full application validation. Use :doc:`jax_benchmarks` to compare
completed search outputs and to measure synchronized calculations.
