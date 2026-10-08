.. _jax-live:

JAX Live execution and output
=============================

``pycbc_live_jax`` runs the Live search with JAX scheduling and numerical
arrays on the selected device. The ordinary ``pycbc_live`` executable retains
its native pipeline. Both accept the usual bank, channel, conditioning,
threshold and output options.

The JAX executable defaults to ``--processing-scheme jax:cpu``. Select an
available NVIDIA GPU with ``--processing-scheme jax:cuda:0``. See
:doc:`jax_arrays` for installation and device selection. Launch the search
through MPI, with rank zero coordinating results and other ranks filtering
their template-bank partitions. Each rank must see its requested device.

Conditioning, PSD evaluation, filtering and numerical veto calculations use
the JAX implementations by default. Template metadata and channel/file
routing remain host operations. Templates retain their original duration and
frequency-spacing groups. See :doc:`jax_strain`, :doc:`jax_psd`,
:doc:`jax_filtering` and :doc:`jax_search` for these calculation interfaces.
The PSD admission diagnostic and its original calculation controls are
described in :doc:`jax_live_psd_numerical_differences`.

Result and output lifetime
--------------------------

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

Numerical validation of output
------------------------------

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
to ``JAXScheme``. The default selection is empty and keeps these calculations
on device. Original controls transfer data to the host and are slower;
ranking can invoke an isolated CPU process.

The :download:`Live output comparison notebook
<../examples/jax/jax_live_output.ipynb>` separates background membership from
order, demonstrates equal-score selection, checks ranking independently,
and asserts exact dtype, shape and bytes for all supplied trigger columns,
selected indices and HDF output. It records library versions and
device hardware. These component checks do not establish equivalence of an
entire search or exercise MPI. Upstream calculation controls and complete
Live filtering/veto examples are in :doc:`jax_search_numerical_differences`.
