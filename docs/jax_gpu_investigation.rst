JAX GPU batch size, memory and utilisation
==========================================

This experiment uses the 1536-row repeated compressed bank and hardware
specified in :doc:`jax_performance`. Keep source revisions, input hashes,
individual trials and scientific comparisons in the compact
`batch sweep summary <_static/jax_batch_sweep.json>`_. The repeated bank tests
execution and allocation behavior; it is not a larger independent search bank.

Batch-size comparison
---------------------

.. list-table:: Separated JAX stack, three uninstrumented trials per batch
   :header-rows: 1

   * - Batch
     - Wall seconds, median [range]
     - Calculation seconds, median [range]
     - Bank templates/calculation second
   * - 128
     - 23.254 [23.203, 23.504]
     - 7.400 [7.391, 7.421]
     - 207.6
   * - 144
     - 23.602 [23.554, 23.754]
     - 7.793 [7.777, 7.819]
     - 197.1
   * - 160
     - 23.602 [23.552, 23.603]
     - 7.566 [7.564, 7.592]
     - 203.0

Batch 128 has the shortest median calculation and wall times among these
three tested sizes. Batch 160 fits but takes 2.3% longer in calculation time;
batch 144 takes 5.3% longer. This comparison does not test every possible
size or establish the maximum fitting batch. The separate general-runtime
comparison is reported in :doc:`jax_performance`.

.. figure:: _static/jax_batch_sweep.png
   :alt: Uninstrumented batch timings and separately measured GPU memory
   :width: 100%

   Timing repetitions and memory captures use separate processes. Compare
   calculation and wall times on the same workload before selecting a batch
   size; a larger batch that fits need not be faster.

Memory measurements
-------------------

The JAX path avoids unused scalar scratch arrays for batched banks, performs
template cropping inside the compiled filter, and releases segment correlation
references before allocating the next segment's workspace. These changes
reduce overlapping allocations; their effect must be measured for the chosen
shape and allocator settings.

Synchronized profiling runs group live pointers by device to avoid counting
aliases twice. At the sixth production batch, distinct live buffers use
1.338, 1.463 and 1.588 GiB for batches 128, 144 and 160. Corresponding
allocator high-water marks are 9.893, 11.062 and 12.247 GiB. The allocator's
reserved pool is 16 GiB in all three captures. The batch-128 run with general
optimizations has the same measured GPU live-buffer and allocator peaks.

The four memory runs preserve all 18 compared scientific datasets and 7984
triggers exactly. HDF attributes were not compared. These measurements do
not include a fresh pre-optimization runtime control, so they quantify the
current implementation rather than a before/after memory reduction.

.. figure:: _static/jax_memory_optimized.png
   :alt: GPU allocator, live buffers and process memory during the JAX search
   :width: 100%

   Live buffers, allocator use and reserved process memory describe different
   quantities. A settled snapshot does not capture every transient FFT or JIT
   workspace allocation and cannot alone establish the largest safe batch.

Process utilisation
-------------------

The plotted window runs from the first ``filter_batch`` start to the final
``filter_batch`` end in each capture. Clipping process CUDA kernel intervals
to that window and taking their union gives:

.. list-table:: Instrumented production filtering windows
   :header-rows: 1

   * - Batch size
     - Window from process start, seconds
     - Duration, seconds
     - Batches
     - Kernel-active union, seconds
     - Kernel-active fraction
   * - 128
     - 18.9475--26.4294
     - 7.4819
     - 12
     - 1.7663
     - 23.61%
   * - 160
     - 19.0247--26.6192
     - 7.5945
     - 10
     - 1.7581
     - 23.15%

Both captures finish successfully and preserve all 18 scientific datasets
of the prior 7984-trigger JAX output exactly; neither HDF file has
attributes. This validates output preservation under profiling, not
CPU-reference equivalence. The complete 384-template CPU qualification
still fails as detailed in :ref:`jax-search-qualification`.

.. figure:: _static/jax_gpu_timeline_optimized_128.png
   :alt: Process CUDA kernels, transfers, GPU memory and CPU use at batch 128
   :width: 100%

   Batch 128 production filtering. CUDA events belong to the search process;
   CPU and memory values are sampled. Instrumented durations are excluded
   from the uninstrumented performance table.

.. figure:: _static/jax_gpu_timeline_optimized_160.png
   :alt: Process CUDA kernels, transfers, GPU memory and CPU use at batch 160
   :width: 100%

   Batch 160 production filtering, using this capture's own phase boundaries.

Kernel-active fraction is the union of process kernel intervals divided by
the selected production interval; it is not SM occupancy. CUDA copy totals
per time bin are not link bandwidth. Process GPU memory includes reserved
allocator space, while host RSS measures a different memory domain. Gaps in
a CUDA timeline identify periods without recorded kernels but do not, by
themselves, identify the responsible host operation.

Reproducing the figures
-----------------------

The complete Nsight capture, SQLite extraction and rendering commands are in
:doc:`jax_performance`. The maintained tools collect process CPU/GPU memory
and CUDA activity together, so new utilisation plots can be produced from
new runs without recovering the removed raw profiling artifacts.

Regenerate the batch comparison from its compact summary with:

.. code-block:: console

   python tools/plot_jax_batch_sweep.py \
     --input docs/_static/jax_batch_sweep.json \
     --output artifacts/jax_batch_sweep.png

The allocation snapshot experiment used the archived standalone helpers at
``jax-evidence-archive-20260919``. To repeat it, recover the two files under ``artifacts/``:

.. code-block:: console

   mkdir -p artifacts/memory
   git show jax-evidence-archive-20260919:tools/profile_jax_memory.py > artifacts/profile_jax_memory.py
   git show jax-evidence-archive-20260919:tools/plot_jax_memory.py > artifacts/plot_jax_memory.py

Run ``profile_jax_memory.py --output-dir artifacts/memory --`` before the
same ``bin/pycbc_inspiral`` command and arguments recorded in the compact
receipt, keeping its environment and affinity. This produces
``artifacts/memory/jax_memory_profile.json``. Render it with:

.. code-block:: console

   python artifacts/plot_jax_memory.py \
     --input artifacts/memory/jax_memory_profile.json \
     --output artifacts/jax_memory_optimized.png

The maintained utilisation tools require none of these archived helpers.

Raw telemetry, CUDA events, HDF files and allocation profiles belong under
ignored ``artifacts/``. Keep the compact measurement summary and final plots
in the documentation.
