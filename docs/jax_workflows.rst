.. _jax-workflows:

JAX Workflow Integration
========================

PyCBC Pegasus workflows support JAX execution across cluster nodes equipped with
GPUs or TPUs.

Workflow Configuration
----------------------

In your workflow configuration file (`.ini`), specify the JAX processing scheme:

.. code-block:: ini

   [inspiral]
   processing-scheme = jax:cuda:0
   batch-size = 128

HTC & Slurm Execution
---------------------

Submit workflow jobs with GPU resource requests:

.. code-block:: ini

   [pegasus_profile]
   condor|request_gpus = 1
   condor|requirements = (CUDACapability >= 7.0)
