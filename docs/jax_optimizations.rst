.. _jax-optimizations:

JAX Optimizations & Performance Tuning
======================================

JAX provides compilation options and architectural patterns that maximize
execution speed on hardware accelerators.

XLA Compilation & JIT
---------------------

PyCBC kernels in `pycbc/filter/`, `pycbc/waveform/`, and `pycbc/inference/`
are designed as pure functions compatible with `jax.jit`. XLA compilation
fuses elementwise operations, minimizes memory round-trips, and emits optimized
assembly for GPU and TPU.

Vectorized Batching (vmap)
--------------------------

Template banks and multidimensional parameter evaluations use `jax.vmap` to
vectorize scalar implementations across batch axes without explicit loop overhead.

Memory Management
-----------------

JAX preallocates GPU memory by default. For co-located workloads or multi-process
pipelines, control allocation with environment variables:

.. code-block:: console

   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
