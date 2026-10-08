.. _jax-inspiral:

JAX inspiral search
===================

``pycbc_inspiral_jax`` runs a single-detector search using JAX arrays and
template batches. It accepts the ordinary inspiral bank, strain, PSD, veto,
clustering and output options. ``pycbc_inspiral`` retains the original native
pipeline. See :doc:`jax_arrays` for installation and device selection.

The JAX command defaults to ``--processing-scheme jax:cpu``. To use an
available NVIDIA GPU, select ``--processing-scheme jax:cuda:0``. Conditioning,
PSD evaluation and filtering run on the selected device by default.
Template generation uses diffGW; supported models and parameter restrictions
are in :doc:`jax_waveform`. Compressed template banks retain their separate
decompression path.

``--batch-size`` (also spelled ``--tile-size``) sets the positive number of
templates per batch. The defaults are 16 on CPU and 64 on a GPU. A value of
one uses the same filtering and event-insertion path as larger batches.
Choose a smaller batch when device memory is limited. Device batching replaces
``--multiprocessing-nprocesses``, which this command rejects.

Template parameters, file routing and final trigger serialization remain host
operations. Compact candidate columns also cross to the host for original
clustering and event insertion. This boundary does not move conditioning or
the complete SNR series off device.

Numerical validation
--------------------

``--jax-reference-operations`` selects original implementations independently.
Unselected stages continue to use JAX. For example, selecting ``waveform``
holds synthesis fixed while comparing conditioning and filtering; selecting
``ifft`` holds inverse transforms fixed while comparing the remaining stages.
Multiple names are comma-separated:

.. code-block:: console

   --jax-reference-operations waveform,correlate,ifft,inner

These switches transfer the selected inputs to the host and can be much
slower. Some original calculations run in an isolated CPU process. They keep
the selected device active for the surrounding JAX calculations and return
results to that device. An exact comparison requires all differing upstream
stages to be held fixed, as well as the same input data and numerical options. For
reproducible byte comparisons, use ``--fft-backends numpy`` in both searches;
FFTW buffer alignment can change rounding between independent CPU runs. See
:doc:`jax_fft_numerical_differences`.

The numerical guides and executed notebooks in :doc:`jax_fft`,
:doc:`jax_psd`, :doc:`jax_strain`, :doc:`jax_filtering`, :doc:`jax_waveform`
and :doc:`jax_search_numerical_differences` explain the available controls and
the differences they isolate. Original waveform generation also makes models
outside the default diffGW allowlist available for validation.

Checkpoint and final output options use the ordinary inspiral interfaces.
A checkpoint records completed template work and accumulated events; it is
not a portable cache of compiled JAX programs or device buffers.
