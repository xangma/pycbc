.. _jax-search:

JAX Search Pipeline & Execution Engine
======================================

PyCBC provides persistent, batched search engine capabilities using JAX,
optimizing template bank execution across accelerators.

Search Architecture
-------------------

The JAX search pipeline coordinates:
1. Data whitening and overwhitening via JAX FFT operations.
2. Waveform template generation using DiffGW JAX (`jaxwave`) or compressed banks.
3. Batched matched filtering and threshold detection across templates.
4. Single-detector time-window peak clustering and event consolidation.

Execution
---------

Launch JAX-accelerated search with `pycbc_inspiral`:

.. code-block:: console

   pycbc_inspiral \
       --processing-scheme jax:cuda:0 \
       --approximant TaylorF2 \
       --batch-size 128 \
       ...
