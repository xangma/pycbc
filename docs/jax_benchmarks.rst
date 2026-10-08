Comparing and timing JAX calculations
=====================================

Compare completed outputs before interpreting timings. Use the same input
data, bank, numerical options, output configuration and input precision for
the original search and the dedicated JAX search. Hold differing calculations
fixed with the controls in :doc:`jax_numerical_differences`. For byte comparisons,
select ``--fft-backends numpy`` in both searches.

Search output comparison
------------------------

From a checkout, compare two HDF files or two directories with identical,
nonempty relative ``.hdf`` file inventories:

.. code-block:: console

   python tools/compare_jax_search.py reference.hdf candidate.hdf --output comparison.json

The comparator checks dataset names, shapes, dtypes, value bytes, attributes
and storage properties. It preserves signed zeros and NaN payloads and does
not reorder triggers or apply a numerical tolerance. It exits unsuccessfully
when a comparison fails.

Different command lines, software provenance and measured performance fields
need explicit exclusions when they are intentionally different. For example:

.. code-block:: console

   python tools/compare_jax_search.py reference.hdf candidate.hdf --ignore-attribute command_line --output comparison.json

``--ignore-attribute`` excludes that attribute name wherever it occurs.
``--ignore-dataset`` excludes the exact HDF path, without a leading slash.
Both options can be repeated. The report records every matched exclusion;
there are no automatic exclusions. Review exclusions together with the
remaining comparison result.

FFT timing and profiling
------------------------

The small FFT tool measures public PyCBC forward/inverse round trips. It
reports the first call separately from warmed synchronized samples and checks
the inverse result outside the timing region:

.. code-block:: console

   python tools/benchmark_jax_fft.py --processing-scheme jax:cuda:0 --dtype complex64 --length 65536 --repeats 20 --output fft.json

Select ``cpu`` for the original CPU calculation. ``--fft-backend`` selects its
FFT implementation and the backend used by original JAX validation controls.
``--reference-operation fft`` and ``--reference-operation ifft`` independently
select those controls. The tool records device hardware and numerical library
versions; a kernel timing is not a complete search throughput measurement.

``--profile PROFILE_DIRECTORY`` records the warmed JAX loop using
`jax.profiler.trace <https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html>`_.
Device results are completed within each timed sample. Profiling adds overhead,
so record unprofiled timings separately. Array construction occurs before the
first timed call; setup, file I/O and output writing are outside this benchmark.

For a complete search, measure process completion, including queued result
collection and closed output files. Record whether waveform generation,
conditioning, compilation and warmup are included. Use repeated measurements
with the same workload and runtime settings; compare CPU and GPU memory use
alongside elapsed time and retain the output comparison for that workload.
