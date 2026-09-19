JAX GPU memory and batch-size results (2026-09-19)
====================================================

The latest campaign used host ``len``, an NVIDIA GeForce RTX 4090 (24 GiB),
JAX 0.4.20 and the ``pycbc3g`` environment. The workload used compressed
IMRPhenomD templates, 4096 Hz strain and ``jax:cuda:0``. CPU affinity was
core 8, numerical-library threads were limited to one, and
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` was unchanged across comparisons.
The compact `results summary <_static/jax_batch_sweep.json>`_ records the
source hashes, workload, final timing trials, parity checks and a
representative command with its environment.

Earlier work retained the warmed first template batch, moved strain staging
and autogate padding onto the device, and removed redundant per-batch garbage
collection. The results below cover the subsequent memory optimization.
Historical timelines and raw profiling dumps are omitted from the docs.

Memory reductions and batch-size comparison
---------------------------------------------

The optimized implementation removes three avoidable allocations or
overlapping lifetimes:

* Allocate the extra scalar scratch buffers only for a bank without
  ``get_batch``. This removes 127 unused buffers at batch size 128.
* Pass the full template batch into the compiled filter and slice inside the
  JIT. The compiler fuses the crop with correlation, eliminating the persistent
  cropped-template allocation. Ordinary list inputs retain their existing path.
* Release the veto lookup's ``corr_tensor`` and residual ``res`` reference at
  the end of each segment. Otherwise the previous correlation tensor stays
  alive while the next segment allocates its FFT workspace.

At batch size 128, peak allocator use falls from **13.867 to 9.893 GiB**
(28.7%). Distinct live buffers at B6 fall from **3.322 to 1.338 GiB**. Pprof
buffer totals agree exactly with the device-qualified pointer inventory at
every final snapshot. This validates the measured
allocations and aliases; it does not compare the contents of unrelated buffers.

.. figure:: _static/jax_batch_sweep.png
   :alt: Lower GPU memory at batch 128; batches 144 and 160 fit but are slower
   :width: 100%

   Memory profiles and three uninstrumented timings per final configuration.
   Dots show every timing repetition; bars show median throughput. The original
   memory measurement is the earlier baseline profile, while control timings
   were repeated in this campaign.

.. list-table:: Same workload, same allocator settings
   :header-rows: 1

   * - Configuration
     - Peak GiB
     - Median calculation seconds
     - Bank templates/s
   * - Original, B128
     - 13.867
     - 6.842
     - 224.5
   * - Optimized, B128
     - 9.893
     - 6.799
     - 225.9
   * - Optimized, B144
     - 11.062
     - 7.243
     - 212.1
   * - Optimized, B160
     - 12.247
     - 7.057
     - 217.7

**Keep B128 for this workload.** B160 now completes, increasing the validated
batch size by 25%, but is about 3.8% slower than optimized B128. The 0.6%
control-to-optimized B128 difference is small; this is a memory improvement,
not evidence of a substantial throughput improvement. The workload has five
segments and 1,536 bank rows made from 384 compressed templates repeated four
times. Results need confirmation on a larger independent bank.

Final B192 still failed: cuFFT requested 3 GiB of scratch during warmup,
followed by a 6 GiB XLA temporary allocation failure in production. B256 was
not retried after that failure. No allocator fraction or unrelated GPU
workload was changed. Transient FFT workspace and allocator layout still
constrain larger batches; settled live-buffer totals alone cannot predict fit.

All successful final memory and timing runs preserve **all 18 scientific
datasets and 7,984 triggers exactly**, including at B144 and B160. Four
performance datasets are excluded; HDF attributes are not compared. Timing
order was reversed/rotated, with CPU affinity and thread counts fixed.
Compilation warmups and instrumented runs are excluded from throughput.

Fresh Nsight captures confirm that larger batches do not close the idle gaps.
Within production filter phases, the union of this process's kernel intervals
is 1.763 seconds out of 7.560 seconds at B128 (23.3%), versus 1.758 out of
7.773 seconds at B160 (22.6%). This measures time with a kernel active, not SM
occupancy. Both captures preserve the same 18 scientific datasets exactly.
Their instrumented durations are excluded from the throughput table above.

.. figure:: _static/jax_gpu_timeline_optimized_128.png
   :alt: Process-specific kernels and transfers across twelve optimized batches
   :width: 100%

   Optimized B128, zoomed to production filtering. CUDA kernels and all three
   transfer directions belong to the target process. Memory and CPU are
   sampled. The approximately 16 GiB process VRAM line includes the allocator's
   reserved pool; it is not the live-buffer or peak-use measurement above.

The corresponding `B160 timeline
<_static/jax_gpu_timeline_optimized_160.png>`_ shows fewer batches but longer
leading gaps: approximately 0.25--0.28 seconds before the first kernel in
batches 2 onward, compared with 0.20--0.23 seconds at B128. Total kernel-active
work stays nearly unchanged. Earlier Python sampling associated these gaps
with HDF5/gzip bank preparation; the CUDA timeline alone cannot establish
their cause. The next experiment should overlap bounded host bank preparation
with GPU work, then validate on an independent bank so repeated templates do
not overstate a host-cache benefit.

.. figure:: _static/jax_memory_optimized.png
   :alt: Optimized JAX allocator, distinct live buffers and host memory snapshots
   :width: 100%

   Optimized B128 memory snapshots. Distinct live buffers exclude transient
   JIT workspace; the allocator pool is not the amount of live data.

Reproducing the results
-------------------------

The compact `batch sweep summary <_static/jax_batch_sweep.json>`_ retains
all final timing repetitions, memory totals, scientific-output comparisons,
source hashes and aggregate process-specific CUDA measurements. It also
records the failed B192 trial. Regenerate the batch comparison plot with:

.. code-block:: console

   python tools/plot_jax_batch_sweep.py \
     --input docs/_static/jax_batch_sweep.json \
     --output artifacts/jax_batch_sweep.png

The timeline and memory figures are retained as published results. Their raw
CUDA events, telemetry samples, per-array inventories, pprof snapshots and
compiler dumps are not checked in. To collect new receipts and render those
figures, follow :doc:`jax_performance` and write generated data under the
ignored ``artifacts/`` directory. The B128 production zoom uses
``--zoom-start 14.833 --zoom-end 22.3932``; B160 uses
``--zoom-start 15.3644 --zoom-end 23.137`` for the published captures.
These time windows are specific to those captures; new runs need their own
production phase boundaries.
