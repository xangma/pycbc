.. _jax-commands:

JAX search commands
===================

The dedicated JAX commands use device arrays and batch scheduling; the ordinary
search executables retain their native pipelines. See :doc:`jax_arrays` for
installation and device selection, :doc:`jax_fft` for FFT and PSD preparation,
and :doc:`jax_filtering` for conditioning, filtering and veto interfaces.

The commands default to ``--processing-scheme jax:cpu``. Select an available
NVIDIA GPU with ``--processing-scheme jax:cuda:0``. Conditioning, PSD evaluation,
filtering and numerical veto calculations use JAX on the selected device by
default. Template metadata, channel/file routing and serialization remain host
operations.

Default template generation uses the optional diffGW interface; supported
models and parameter restrictions are in :doc:`jax_waveform`. Select the
``waveform`` reference operation to use original PyCBC/LAL generation.
Compressed template banks have a separate decompression path and control.

.. _jax-command-validation:

Original-implementation validation
----------------------------------

``--jax-reference-operations`` selects original calculations independently;
unselected stages continue to use JAX. For example, ``waveform`` holds synthesis
fixed while comparing conditioning and filtering, and ``ifft`` holds inverse
transforms fixed. Multiple names are comma-separated:

.. code-block:: console

   --jax-reference-operations waveform,correlate,ifft,inner

The Python equivalent is ``JAXScheme(reference_operations=(...))``. The default
selection is empty. Selected calculations transfer inputs to the host and can
be much slower; some run in an isolated CPU process. Surrounding calculations
keep using the selected JAX device, and array results return to that device.

Exact comparison requires the same input data and numerical options, with every
differing upstream stage held fixed. For reproducible FFT byte comparisons, use
``--fft-backends numpy`` in both searches; FFTW buffer alignment can change
rounding between independent CPU runs. See
:doc:`jax_fft_numerical_differences`. The calculation guides and executed
notebooks linked from :doc:`jax_numerical_differences` explain each control and
its limits. Original waveform generation also permits models outside the
default diffGW allowlist for validation.

.. _jax-inspiral:

Inspiral search
---------------

``pycbc_inspiral_jax`` runs a single-detector search using JAX arrays and
template batches. It accepts the ordinary inspiral bank, strain, PSD, veto,
clustering and output options. ``pycbc_inspiral`` retains the original native
pipeline.

``--batch-size`` (also spelled ``--tile-size``) sets the positive number of
templates per batch. The defaults are 16 on CPU and 64 on a GPU. A value of
one uses the same filtering and event-insertion path as larger batches.
Choose a smaller batch when device memory is limited. Device batching replaces
``--multiprocessing-nprocesses``, which this command rejects.

Template parameters, file routing and final trigger serialization remain host
operations. Compact candidate columns also cross to the host for original
clustering and event insertion. This boundary does not move conditioning or
the complete SNR series off device.

Checkpoint and final output options use the ordinary inspiral interfaces.
A checkpoint records completed template work and accumulated events; it is
not a portable cache of compiled JAX programs or device buffers.

.. _jax-live:

Live search
-----------

``pycbc_live_jax`` runs the Live search with JAX scheduling. The ordinary
``pycbc_live`` executable retains its native pipeline. Both accept the usual
bank, channel, conditioning, threshold and output options.

Launch through MPI, with rank zero coordinating results and other ranks
filtering their template-bank partitions. Each rank must see its requested
device. Templates retain their original duration and frequency-spacing groups.
See :doc:`jax_search` for library batching and event interfaces and
:doc:`jax_psd_numerical_differences` for the Live PSD admission diagnostic and
its original calculation controls.

Result and output lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detector work is submitted before collecting its results. Filtering ranks
hand off immutable device results through an isolated MPI communicator;
serialization requires host buffers. The coordinator consumes blocks and
rank contributions in order. Receipt acknowledgements bound outstanding
blocks, including retained device results and serialized buffers.

On a GPU, result collection and MPI handoff use a single background worker
when MPI provides ``MPI_THREAD_MULTIPLE``. CPU execution and other MPI thread
levels use synchronous handoff. The handoff worker transfers results without
advancing coordinator science state.
MPI buffers remain owned until both the send and its acknowledgement finish.

The coordinator can also write HDF files with one ordered background writer.
Each snapshot owns mutable host metadata and retains immutable device arrays
until collection. The writer preserves the ordinary result columns, Unicode
conversion, gate fields and PSD metadata; PSD datasets use gzip level 9 and
shuffling. It never calls MPI.

Submitting a snapshot does not mean its file is complete. Draining output
waits for collection, writing and file closure. Leaving the result/output
scopes drains work and releases their threads and communicator on normal and
exceptional exits. A writer failure stops subsequent files and reaches the
calling thread. A fatal MPI handoff failure reports its cause and aborts the
isolated Live communicator so peers do not wait for a missing block.

.. list-table:: Queue controls
   :header-rows: 1
   :widths: 40 60

   * - Environment variable
     - Accepted values and default
   * - ``PYCBC_JAX_LIVE_RESULT_QUEUE_DEPTH``
     - Integer 1--4; four on a GPU, one on CPU. Counts outstanding blocks
       per filtering rank, rather than only completed MPI sends.
   * - ``PYCBC_JAX_ASYNC_OUTPUT``
     - ``1``, ``true`` or ``yes`` enable the writer; ``0``, ``false`` or
       ``no`` disable it, case-insensitively. Enabled by default on a GPU,
       disabled on CPU. The writer holds two waiting snapshots plus the one
       being written.

These controls change scheduling and retained memory, rather than the
scientific configuration. Library callers can use ``live_result_scope`` and
``live_output_scope`` from ``pycbc.events.live_pipeline_jax`` and
``pycbc.events.live_output_jax`` to own the same lifetimes.

.. _jax-live-output-validation:

Numerical validation of output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Background top-N output uses a partition, which does not define the order
within the selected partition. NumPy uses introselect; JAX implements its
partition using top-k operations. The same selected values can therefore
appear in a different HDF order. See the `NumPy partition contract
<https://numpy.org/doc/stable/reference/generated/numpy.partition.html>`_
and `JAX partition implementation
<https://docs.jax.dev/en/latest/_autosummary/jax.numpy.partition.html>`_.

Loudest-event output stores the union of the largest SNR and NewSNR indices.
For equal scores, the original NumPy default quicksort and JAX's default
stable sort can select different tied rows. Reversing the ascending sort
also reverses its tie order. These ordering differences do not require a
floating-point value difference. See `NumPy argsort
<https://numpy.org/doc/stable/reference/generated/numpy.argsort.html>`_
and `JAX argsort
<https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html>`_.

``live_output_selection`` restores the original NumPy partition, sorting and
index-union operations while holding their input columns fixed. ``newsnr``
independently restores the original ranking calculation. Select both for an
exact comparison of this output boundary:

.. code-block:: console

   --jax-reference-operations newsnr,live_output_selection

In Python, pass ``reference_operations=("newsnr", "live_output_selection")``
to ``JAXScheme``. These controls use the host boundary described above; ranking
can invoke an isolated CPU process.

The :download:`Live output comparison notebook <../examples/jax/jax_live_output.ipynb>` separates background membership from
order, demonstrates equal-score selection, checks ranking independently,
and asserts exact dtype, shape and bytes for all supplied trigger columns,
selected indices and HDF output. It records library versions and
device hardware. These component checks do not establish equivalence of an
entire search or exercise MPI. Upstream calculation controls and complete
Live filtering/veto examples are in :doc:`jax_search_numerical_differences`.
