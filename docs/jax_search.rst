.. _jax-search:

JAX search and tuning
=====================

The JAX search path combines device kernels with host orchestration for strain
conditioning, template preparation, matched filtering, vetoes, clustering and
output. Selecting a JAX scheme does not make every stage device-resident.
See :doc:`jax` for installation, device selection and precision, and
:doc:`jax_performance` for measured executable performance and synthetic
filter estimates.

.. _jax-filtering:

Filtering API
-------------

Use the usual filtering and veto functions inside a ``JAXScheme`` context.
For prepared frequency-domain ``template`` and ``data`` arrays and a
compatible one-sided ``psd``:

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.filter import matched_filter
   from pycbc.vetoes import power_chisq

   with JAXScheme("cuda:0"):
       snr = matched_filter(template, data, psd=psd,
                            low_frequency_cutoff=20.0)
       chisq = power_chisq(template, data, num_bins=16, psd=psd,
                          low_frequency_cutoff=20.0)

The matched filter correlates strain with the conjugated template, weighted
by the inverse PSD, and returns the normalized complex SNR time series.
The power chi-square veto divides the template into frequency bins with
equal expected power and tests the signal contribution in each bin.

Execution
---------

Add the scheme and batch size to an otherwise configured ``pycbc_inspiral``
command:

.. code-block:: console

   pycbc_inspiral \
       --processing-scheme jax:cuda:0 \
       --approximant TaylorF2 \
       --batch-size 128 \
       ...

Waveform preparation can use compressed banks or the selected waveform
generator. On-device generation through ``diffgw`` requires that optional
dependency and explicit ``--enable-diffgw``; selecting JAX alone does not enable
it. See :ref:`jax-reference-campaign` for complete executable configurations.

.. _jax-search-batching:
.. _jax-tiled-pathways:
.. _jax-optimizations:

Template batching and compilation
---------------------------------

A bank can contain more templates than fit on the device simultaneously.
``--batch-size`` selects bounded execution batches. Standard CPU filtering
uses a batch size of one; sizes greater than one require a JAX scheme.
Choose the size for the template length, available memory and FFT workspace,
then validate scientific output and measure the complete search.

Selected JAX filtering and waveform kernels use ``jax.jit`` for compilation
and ``jax.vmap`` for batched evaluation. Compiled kernels can be reused for
compatible shapes; a final partial batch may require another compilation.
Include compilation in fresh-process timing and report warmed measurements
separately. A larger fitting batch need not improve throughput: see the
measured sweep in :ref:`jax-gpu-investigation`.

To disable JAX GPU preallocation, set this before starting the process:

.. code-block:: console

   export XLA_PYTHON_CLIENT_PREALLOCATE=false

Record allocator settings with measurements. Disabling preallocation does
not remove the memory requirements of templates, FFT workspace and intermediate
arrays. Automatic memory-based batch selection and overlapping host preparation
with device execution remain possible future experiments.

.. _jax-search-workflows:
.. _jax-workflows:

HTCondor workflow configuration
-------------------------------

For an existing Pegasus workflow whose inspiral executable is named
``inspiral``, configure its scheme and request a GPU for those jobs:

.. code-block:: ini

   [inspiral]
   processing-scheme = jax:cuda:0
   batch-size = 128

   [pegasus_profile-inspiral]
   condor|request_gpus = 1

The worker environment must provide PyCBC, a compatible JAX CUDA installation
and a visible GPU. Adapt resource requirements to the site's scheduler and
GPU configuration. This example is an HTCondor recipe; the measured backend
coverage in this documentation is CPU and NVIDIA CUDA.
