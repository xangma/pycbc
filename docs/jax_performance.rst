.. _jax-performance:

JAX Measured Performance Evidence
=================================

Measured execution benchmarks demonstrating calculation rate and scaling across CPU and GPU architectures.
Measurements were conducted on host ``len`` featuring an **AMD Ryzen Threadripper PRO 3995WX 64-Cores**
paired with an **NVIDIA GeForce RTX 4090 (24GB VRAM)** running CUDA 12 / JAX 0.4.20.

All measurements reflect steady-state execution time (with JIT warmup decoupled and device
synchronization enforced via ``.block_until_ready()``), reported as medians over 10 repeated trials.
The sealed measurement artifact is recorded in :file:`artifacts/jax_benchmark_results.json`.

Six-Arm Benchmark Matrix
------------------------

The benchmark evaluates six architectural configurations covering waveform generation, host-device
memory movement, and matched filtering:

.. list-table:: Benchmark Evaluation Arms
   :header-rows: 1
   :widths: 22 20 22 36

   * - Arm Identifier
     - Waveform Source
     - Filter Backend
     - Architectural Focus
   * - **Original CPU**
     - Precomputed / MKL
     - MKL CPU (1 thread)
     - Standard PyCBC pre-branch baseline (Intel MKL DFTI)
   * - **Branch CPU**
     - Precomputed / MKL
     - MKL CPU (1 thread)
     - Validates zero CPU regression on candidate branch
   * - **JAX CPU (LAL)**
     - LAL (Host CPU)
     - JAX CPU
     - Quantifies host LAL synthesis paired with JAX CPU filtering
   * - **JAX CPU (diffgw)**
     - ``diffgw`` (JAX CPU)
     - JAX CPU
     - Pure JAX CPU end-to-end execution
   * - **JAX CUDA (LAL)**
     - LAL (Host CPU)
     - JAX CUDA (GPU)
     - Quantifies host-to-device PCIe bottleneck when waveforms are regenerated
   * - **JAX CUDA (diffgw)**
     - ``diffgw`` (JAX CUDA)
     - JAX CUDA (GPU)
     - Pure on-device GPU execution in VRAM

.. note::

   **Waveform Handling in Production Searches**:
   In standard PyCBC offline inspiral searches (``pycbc_inspiral``), templates are precomputed
   and stored in **compressed banks** using singular value decomposition (SVD) coefficients.
   During search filtering, templates are decompressed via fast inline linear interpolation (< 1 ms per template once),
   resulting in **zero recurring waveform generation overhead**. Dynamic waveform synthesis (e.g. via ``diffgw``)
   is relevant for parameter estimation (Bayesian inference) or exploratory uncompressed searches, but is not part
   of the production inspiral matched filtering loop.

.. note::

   **CPU FFT Backend Verification**:
   All standard CPU baseline benchmarks (``original_cpu`` and ``branch_cpu``) explicitly use **Intel MKL**
   via ``--fft-backends mkl`` (executing through ``pycbc.fft.mkl.FFT``), pinned to a single physical core with
   ``OMP_NUM_THREADS=1`` and ``MKL_NUM_THREADS=1``. Both Intel MKL and FFTW are verified available and functional
   on host ``len``.

Performance Scaling Figures
---------------------------

Throughput Scaling Across Batch Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/jax_throughput_scaling.png
   :alt: JAX Matched Filter Throughput Scaling
   :width: 100%
   :align: center

   Figure 1: Matched filter calculation rate (templates/sec) as a function of template batch size :math:`B`
   for streaming live (:math:`N=2^{17}`) and offline inspiral (:math:`N=2^{21}`) search tracks.

Relative Speedup vs Baseline CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/jax_speedup_matrix.png
   :alt: Relative Speedup Matrix
   :width: 100%
   :align: center

   Figure 2: Relative speedup vs standard single-threaded MKL CPU baseline across evaluated arms at batch sizes
   :math:`B \in \{1, 16, 64\}` for the streaming track.

Per-Template Latency Breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/jax_latency_breakdown.png
   :alt: Per-Template Latency Breakdown
   :width: 100%
   :align: center

   Figure 3: Component latency breakdown comparing on-the-fly waveform synthesis, host-device transfer, and matched
   filtering. In production searches using pre-generated compressed banks, waveform synthesis is absent. When dynamic
   generation is required (e.g. inference), on-device ``diffgw`` synthesis eliminates the host CPU generation bottleneck.

End-to-End Execution & Hardware Telemetry Timeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/jax_gpu_timeline_optimized_128.png
   :alt: Process-specific JAX CUDA kernels, transfers, CPU and memory
   :width: 100%
   :align: center

   Figure 4: Latest optimized ``pycbc_inspiral`` capture on an RTX 4090,
   zoomed to production filtering at batch size 128 with 1,536 templates
   (the 384-template bank repeated four times). CUDA activity and all three
   transfer directions belong to the target process. Transfer bars show MiB
   completed per 100 ms bin, not PCIe bandwidth; kernel activity is
   interval-union time, not occupancy. CPU, process VRAM and host RSS are
   sampled. Instrumented timings are excluded from throughput measurements.
   See :doc:`jax_gpu_investigation` for the latest memory and batch comparison.

Streaming Live Search Workload (:math:`N = 131,072`)
-----------------------------------------------------

Evaluates complex128 matched filtering on 64-second streaming strain blocks at 2048 Hz
(:math:`N = 131,072` samples per block), representative of low-latency ``pycbc_live`` streaming analyses.

**Table 1 Column Definitions**:

* **Configuration**: The software backend and computing architecture under test (Original CPU baseline using Intel MKL, Branch CPU candidate, JAX CPU, or JAX CUDA on RTX 4090).
* **Batch :math:`B`**: The number of template frequency series processed together in a single batched 2D FFT cross-correlation call.
* **Filter Latency (ms)**: Wall-clock execution time in milliseconds required to execute the cross-correlation, inverse FFT, normalization, and peak detection across the full block. For :math:`B > 1`, total block latency is reported.
* **Filter Rate (tmpl/s)**: Effective template processing rate, defined as :math:`\text{Filter Rate} = \frac{B}{\text{Filter Latency (seconds)}}`.
* **Throughput (templates/core)**: The number of templates that one compute resource (1 CPU core or 1 GPU) can filter continuously in 1× real time (:math:`\text{Throughput} = \text{Filter Rate} \times T_{\text{block}} = \text{Filter Rate} \times 64\text{ s}`).
* **Speedup**: Relative throughput factor compared to the single-template baseline (:math:`B=1` on Original CPU, filtering at 173.6 tmpl/s).

.. list-table:: Table 1: Streaming live search throughput (:math:`N=2^{17}`, 64 s block duration)
   :header-rows: 1
   :widths: 26 10 18 20 20 12

   * - Configuration
     - Batch :math:`B`
     - Filter Latency (ms)
     - Filter Rate (tmpl/s)
     - Throughput (templates/core)
     - Speedup
   * - Original CPU (MKL baseline)
     - 1
     - 5.76 ms
     - 173.6
     - **11,110**
     - 1.00×
   * - Branch CPU (MKL candidate)
     - 1
     - 5.12 ms
     - 195.3
     - **12,499**
     - 1.12×
   * - JAX CPU
     - 1
     - 2.74 ms
     - 365.0
     - **23,360**
     - 2.10×
   * - JAX CUDA
     - 1
     - 0.21 ms
     - 4,761.9
     - **304,762**
     - 27.4×
   * - JAX CUDA
     - 16
     - 0.62 ms (total)
     - 25,806.5
     - **1,651,616**
     - 148.6×
   * - JAX CUDA
     - 64
     - 2.09 ms (total)
     - 30,622.5
     - **1,959,840**
     - 176.4×
   * - JAX CUDA
     - 128
     - 4.06 ms (total)
     - 31,512.1
     - **2,016,774**
     - 181.5×
   * - JAX CUDA
     - 256
     - 7.95 ms (total)
     - 32,179.6
     - **2,059,494**
     - 185.4×
   * - JAX CUDA
     - 512
     - 15.55 ms (total)
     - 32,931.9
     - **2,107,642**
     - 189.7×
   * - **JAX CUDA**
     - **1024**
     - **30.32 ms (total)**
     - **33,776.1**
     - **2,161,670**
     - **194.6×**

Offline Inspiral Matched-Filter Kernel Microbenchmark (:math:`N = 2,097,152`)
---------------------------------------------------------------------------------

Evaluates single-precision (``complex64``, the production search standard) matched filtering on 512-second synthetic strain segments
at 4096 Hz (:math:`N = 2,097,152 = 2^{21}` samples per segment).

.. important::

   **Table 2 Workload & Scope (Peak Hardware Saturation Limit)**:
   Table 2 is an **isolated kernel microbenchmark** measuring the pure mathematical FFT correlation latency
   and peak hardware throughput of the accelerator. In this synthetic benchmark:

   - **Zero Triggers**: No candidate peaks exceed the SNR threshold.
   - **Zero Vetoes**: The 16-bin Power :math:`\chi^2` and Auto :math:`\chi^2` vetoes are **not evaluated**.
   - **Zero I/O**: No frame files are read, no PSD is estimated, and no HDF5 files are written.
   - **Zero Decompression**: Template vectors are pre-allocated directly in device memory.

   This table establishes the theoretical "speed of light" of the hardware for raw correlation arithmetic:
   sustaining up to **5,315 tmpl/s** (a **511× speedup** over single-threaded MKL CPU, equivalent to **2.72 million templates / core** in 1× real time).

**Table 2 Column Definitions**:

* **Configuration**: The software processing backend and numerical engine:
  - *Original CPU*: PyCBC baseline executing sequential single-template matched filtering with Intel MKL DFTI on a single physical CPU thread.
  - *JAX CPU*: Vectorized 2D batched transform compiled by XLA executing across CPU cores.
  - *JAX CUDA*: Parallelized on-device execution across 16,384 CUDA cores on the NVIDIA GeForce RTX 4090.
* **Batch :math:`B`**: The number of template frequency-series vectors (:math:`2^{21}` points each) correlated and inverse-transformed concurrently in a single batched 2D FFT call.
* **Filter Latency (ms)**: The total wall-clock elapsed time in milliseconds required to compute the frequency-domain cross-correlation (:math:`s^*(f) h(f)`) and inverse FFT for the entire batch of :math:`B` templates on a 512 s segment (:math:`N = 2,097,152` complex points). For :math:`B > 1`, the average per-template latency (:math:`\frac{\text{Filter Latency}}{B}`) is shown in parentheses.
* **Filter Rate (tmpl/s)**: The raw template processing throughput of the kernel:

  .. math::

     \text{Filter Rate} = \frac{B}{\text{Filter Latency (seconds)}} \quad \left[\frac{\text{templates}}{\text{second}}\right]

  This quantifies how many template-segment correlations the hardware computes per second of pure compute time.
* **Throughput (templates/core)**: Real-time analysis throughput, representing how many template waveforms a single compute unit (1 CPU core or 1 GPU) can filter continuously in 1× real time:

  .. math::

     \text{Throughput} = \text{Filter Rate} \times T_{\text{segment}} = \frac{N_{\text{tmpl}} \times T_{\text{data}}}{T_{\text{compute}} \times N_{\text{cores}}} \quad \left[\frac{\text{templates}}{\text{core}}\right]

  where :math:`T_{\text{segment}} = 512\text{ s}` and :math:`N_{\text{cores}} = 1`. For example, JAX CUDA at :math:`B=128` achieves :math:`5,315.0\text{ tmpl/s} \times 512\text{ s} \approx 2,721,257\text{ templates/core}`.
* **Peak VRAM**: Maximum GPU device memory allocated during kernel execution (measured via ``jax.devices()[0].memory_stats()``).
* **Speedup**: The throughput acceleration ratio relative to the single-template baseline (:math:`B=1` on MKL CPU, filtering at 10.4 tmpl/s):

  .. math::

     \text{Speedup} = \frac{\text{Filter Rate}}{\text{Filter Rate}_{\text{Original CPU } (B=1)}}

.. list-table:: Table 2: Offline inspiral matched filter throughput & VRAM scaling (:math:`N=2^{21}`, single precision ``complex64``, isolated kernel)
   :header-rows: 1
   :widths: 22 10 18 18 18 12 10

   * - Configuration
     - Batch :math:`B`
     - Filter Latency (ms)
     - Filter Rate (tmpl/s)
     - Throughput (templates/core)
     - Peak VRAM
     - Speedup
   * - **Original CPU** (MKL baseline)
     - 1
     - 96.3 ms
     - 10.4
     - **5,315**
     - —
     - 1.00×
   * - Original CPU (MKL baseline)
     - 4
     - 300.4 ms (75.1 ms/tmpl)
     - 13.3
     - **6,817**
     - —
     - 1.28×
   * - Original CPU (MKL baseline)
     - 16
     - 1,100.3 ms (68.8 ms/tmpl)
     - 14.5
     - **7,445**
     - —
     - 1.39×
   * - Original CPU (MKL baseline)
     - 32
     - 2,244.5 ms (70.1 ms/tmpl)
     - 14.3
     - **7,300**
     - —
     - 1.38×
   * - Original CPU (MKL baseline)
     - 64
     - 4,452.5 ms (69.6 ms/tmpl)
     - 14.4
     - **7,359**
     - —
     - 1.38×
   * - Original CPU (MKL baseline)
     - 128
     - 11,367.9 ms (88.8 ms/tmpl)
     - 11.3
     - **5,765**
     - —
     - 1.09×
   * - **JAX CPU** (Vectorized XLA)
     - 1
     - 59.7 ms
     - 16.7
     - **8,574**
     - —
     - 1.61×
   * - JAX CPU (Vectorized XLA)
     - 4
     - 216.3 ms (54.1 ms/tmpl)
     - 18.5
     - **9,469**
     - —
     - 1.78×
   * - JAX CPU (Vectorized XLA)
     - 16
     - 935.0 ms (58.4 ms/tmpl)
     - 17.1
     - **8,762**
     - —
     - 1.64×
   * - JAX CPU (Vectorized XLA)
     - 32
     - 1,881.2 ms (58.8 ms/tmpl)
     - 17.0
     - **8,709**
     - —
     - 1.64×
   * - JAX CPU (Vectorized XLA)
     - 64
     - 3,541.7 ms (55.3 ms/tmpl)
     - 18.1
     - **9,252**
     - —
     - 1.74×
   * - JAX CPU (Vectorized XLA)
     - 128
     - 7,095.2 ms (55.4 ms/tmpl)
     - 18.0
     - **9,237**
     - —
     - 1.73×
   * - **JAX CUDA** (RTX 4090)
     - 1
     - 0.16 ms
     - 6,128.2
     - **3,137,659**
     - 68 MB
     - 589.3×
   * - JAX CUDA (RTX 4090)
     - 4
     - 0.82 ms (0.204 ms/tmpl)
     - 4,907.2
     - **2,512,510**
     - 252 MB
     - 471.8×
   * - JAX CUDA (RTX 4090)
     - 16
     - 3.10 ms (0.194 ms/tmpl)
     - 5,158.6
     - **2,641,214**
     - 972 MB
     - 496.0×
   * - JAX CUDA (RTX 4090)
     - 32
     - 6.12 ms (0.191 ms/tmpl)
     - 5,225.7
     - **2,675,573**
     - 2,060 MB
     - 502.5×
   * - JAX CUDA (RTX 4090)
     - 64
     - 12.08 ms (0.189 ms/tmpl)
     - 5,296.3
     - **2,711,705**
     - 4,108 MB
     - 509.3×
   * - **JAX CUDA** (RTX 4090)
     - **128**
     - **24.08 ms (0.188 ms/tmpl)**
     - **5,315.0**
     - **2,721,257**
     - **8,204 MB**
     - **511.1×**
   * - JAX CUDA (RTX 4090)
     - 256
     - *VRAM limit*
     - —
     - —
     - > 20 GB (OOM)
     - —

.. note::

   **CPU Batching Mechanics vs GPU Acceleration**:

   * **Original MKL CPU**: PyCBC's legacy CPU matched filtering API processes batches serially in a Python loop
     over templates. Once CPU cache warm-up stabilizes, it achieves a steady-state rate of :math:`\approx 14.4\text{ tmpl/s}`
     (:math:`\approx 69\text{ ms/tmpl}`). At :math:`B=128`, cache thrashing increases per-template latency to :math:`88.8\text{ ms}`.
   * **JAX CPU**: JAX compiles a vectorized 2D batched transform that executes across CPU cores via the multithreaded XLA
     CPU backend, sustaining :math:`\approx 18.0\text{ tmpl/s}` (:math:`\approx 55.4\text{ ms/tmpl}`) consistently across all batch sizes
     (:math:`\approx 1.25\times` faster than single-threaded MKL CPU).
   * **JAX CUDA**: Because cuFFT and tensor multiplication execute concurrently across 16,384 CUDA cores, JAX CUDA transforms
     an entire batch of 128 templates in **24.08 ms** (just **0.188 ms per template**), yielding **5,315 tmpl/s**—a **511× speedup**
     over single-template MKL CPU, sustaining **2.72 million templates in real time**.
   * **VRAM Boundary**: For single-precision :math:`N = 2^{21}`, memory scales linearly with batch size: :math:`B=16` uses 0.97 GB,
     :math:`B=64` uses 4.1 GB, and :math:`B=128` occupies **8.2 GB** in VRAM. At :math:`B=256`, the combined working buffers
     (:math:`256 \times 2^{21}` complex points for input, output, and correlation products) plus the 4 GB cuFFT internal scratch plan
     exceed the GPU allocator limit on 24 GB hardware, establishing :math:`B=128` as the optimal safe production batch ceiling.

Complete Executable Offline Campaign (pycbc_inspiral, 384 Compressed Templates)
--------------------------------------------------------------------------------

The full end-to-end executable ``pycbc_inspiral`` was evaluated across all pipeline stages
(frame reading, highpass filtering, autogating, 126-segment Welch PSD estimation, template bank reading,
scalar matched filtering, 16-bin Power :math:`\chi^2` vetoes, and symmetric clustering)
on 1,904 seconds of real LIGO Hanford (H1) data across 384 compressed BNS/NSBH templates (5 segments, totaling 1,920 template-segment evaluations).
For complete input checksums, parameter bounds, and search geometry selection, see :ref:`jax-reference-campaign`.

.. important::

   **Contrast Between Table 2 (Microbenchmark) and Table 3 (Full Pipeline)**:

   While Table 2 and Table 3 evaluate identical 512-second segment lengths (:math:`N = 2^{21}` points in single-precision ``complex64``),
   they answer two fundamentally different operational questions:

   * **Table 2 (Isolated Kernel Microbenchmark)**: Measures peak hardware compute saturation for FFT cross-correlation in total isolation.
     It evaluates synthetic data with **0 triggers, 0 thresholding, 0 vetoes, and 0 frame I/O**, establishing the physical
     hardware limit of **2,721,257 templates / core** (5,315 tmpl/s) on the RTX 4090.
   * **Table 3 (Complete Executable Search)**: Measures the operational throughput of the production search executable (``pycbc_inspiral``)
     analyzing real detector strain data. Real noise excursions generate candidate triggers exceeding threshold (:math:`\text{SNR} > 5.5`),
     engaging on-device 16-bin Power :math:`\chi^2` vetoes, Auto :math:`\chi^2` vetoes, and peak clustering. With end-to-end on-device execution,
     core calculation time across all 384 templates and 1,920 template-segment evaluations is reduced to **2.18 s**,
     delivering an operational throughput of **335,286 templates / core** (a **22.52× calculation speedup** and **6.11× wall speedup** over single-core MKL CPU).

Performance Metrics & Realtime Factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance is reported using two complementary operational metrics:

1. **Realtime Factor (s/s)**:
   Measures analysis speed relative to the physical flow of time at the observatory:

   .. math::

      \text{Realtime Factor} = \frac{T_{\text{data analyzed}}}{T_{\text{elapsed}}} \quad \left[\frac{\text{seconds of detector data}}{\text{second of execution time}}\right]

   * **Calc Realtime (s/s)**: Evaluates core computation speed (:math:`T_{\text{elapsed}} = T_{\text{calc}}`), measuring the throughput of the FFT filtering, veto, and clustering engine. For JAX CUDA, **873.2 s/s** means the GPU processes **~14.5 minutes of data every second of GPU compute time** across the 384 templates.
   * **Wall Realtime (s/s)**: Evaluates total end-to-end executable speed (:math:`T_{\text{elapsed}} = T_{\text{wall}}`), including Python startup, frame file I/O, Welch PSD estimation, autogating, and HDF5 output. For JAX CUDA, **171.0 s/s** means the complete job analyzes **1,904 s (~32 minutes) of real detector data in just 11.13 seconds**.

2. **Throughput (templates/core)**:
   Measures how many templates a single physical CPU core or GPU accelerator can sustain concurrently in 1× real time (:math:`1.0\text{ s/s}`):

   .. math::

      \text{Throughput} = \text{Realtime Factor} \times N_{\text{templates}} = \frac{N_{\text{templates}} \times T_{\text{data analyzed}}}{T_{\text{compute time}} \times N_{\text{compute resources}}} \quad \left[\frac{\text{templates}}{\text{core}}\right]

   where :math:`N_{\text{templates}} = 384` templates in the bank, :math:`T_{\text{data analyzed}} = 1904.0\text{ s}`, and :math:`N_{\text{compute resources}} = 1`.
   At :math:`B=128`, JAX CUDA sustains **335,286 templates/core** (core calculation) and **65,667 templates/core** (total process wall time) on a single RTX 4090.

.. _jax-timing-boundaries:

Timing Metrics & Scope Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To prevent ambiguity when comparing runtime metrics across pipeline components, PyCBC inspiral analyses differentiate between internal sub-timers and total executable wall-clock duration:

.. list-table:: PyCBC Inspiral Timing Boundaries and Phase Coverage
   :header-rows: 1
   :widths: 16 16 34 34

   * - Timing Metric
     - Source / Capture Point
     - Inclusions (What is Measured)
     - Exclusions (What is Excluded)
   * - **Core Calculation Time (``calc_time``)**
     - PyCBC internal: ``run_time - tsetup`` (recorded in ``H1/search``)
     - * Waveform batch decompression / generation (:math:`h(f)`)
       * Template norm (:math:`\sigma^2`) computation
       * Frequency-domain correlation (:math:`s^*(f) h(f)`)
       * Batched inverse FFT to complex SNR time series
       * Peak thresholding above SNR threshold
       * Power :math:`\chi^2` veto calculation & frequency binning
       * Single-detector trigger clustering & event consolidation
     - * Python process startup & dependency imports (~4.17 s)
       * Frame file disk I/O (~2.38 s)
       * Strain conditioning (highpass filter, resampling, autogating) (~1.0 s)
       * PSD estimation (Welch averaging, interpolation, inverse truncation) (~1.4 s)
       * Strain Fourier segment transforms (0.3 ms on GPU)
       * Template bank metadata loading
       * JAX JIT warmup pass during setup
       * Trigger serialization & writing output HDF5 (~0.02 s)
   * - **Pipeline Execution Time (``run_time``)**
     - PyCBC internal: captured between ``tstart`` and ``tstop`` of ``bin/pycbc_inspiral``
     - * All data conditioning & PSD estimation (``tsetup``)
       * Full template matched filtering & :math:`\chi^2` calculation (``calc_time``)
       * Final event consolidation
     - * Python interpreter startup & library imports (~4.17 s)
       * HDF5 trigger dataset serialization (``event_mgr.write_events``)
       * FFTW wisdom flushing & process teardown
   * - **Total Process Wall Time (``wall_sec``)**
     - External harness: process fork to termination (``time.perf_counter()``)
     - **Everything**: 100% of end-to-end execution including Python process spawn, imports, data conditioning, filtering, vetoes, clustering, disk I/O, and file closure
     - **Nothing**: Complete wall-clock duration

**Table 3 Column Definitions**:

* **Evaluation Arm**: The specific search code checkout and execution configuration being measured:
  - *Original CPU*: Frozen upstream master reference (``40e94792b3``) running sequential single-template matched filtering with Intel MKL (:math:`B=1`).
  - *Branch CPU Batched*: Candidate branch running batched CPU filtering (:math:`B=128`), verifying zero regression on CPU while demonstrating that CPU batching saturates L3 cache rather than providing speedups.
  - *JAX CPU Batched*: Pure JAX CPU execution (:math:`B=16`) running compiled XLA kernels across CPU threads.
  - *JAX CUDA Batched*: Full on-device GPU execution (:math:`B=128`) on the RTX 4090 with batched cuFFT, parallel peak clustering, and vectorized Power :math:`\chi^2` vetoes.
* **Batch :math:`B`**: Number of templates grouped together in each batch during the search filtering loop.
* **Total Wall (s)**: End-to-end executable execution duration from process launch to completion (measured externally via ``time.perf_counter()``), including startup, imports, frame I/O, strain conditioning, PSD estimation, template loading, filtering, vetoes, clustering, and HDF5 output writing.
* **Core Calc (s)**: Internal PyCBC timer (``calc_time = run_time - tsetup``) measuring exclusively the template filtering loop within ``batch_template_triggers`` (template decompression, correlation, IFFT, peak clustering, and Power :math:`\chi^2` veto evaluation).
* **Wall Realtime (s/s)**: Real-time analysis factor based on total wall time (:math:`\frac{T_{\text{data analyzed}}}{T_{\text{total wall}}}`), indicating how many seconds of detector strain data are analyzed per second of clock time.
* **Calc Realtime (s/s)**: Real-time analysis factor based on core calculation time (:math:`\frac{T_{\text{data analyzed}}}{T_{\text{core calc}}}`), indicating the filtering speed during the active search stage.
* **Throughput (templates/core)**: Full operational search throughput across all 384 templates and 1,904 seconds of analyzed data:

  .. math::

     \text{Throughput} = \text{Realtime Factor} \times N_{\text{templates}} = \frac{N_{\text{templates}} \times T_{\text{data analyzed}}}{T_{\text{compute time}} \times N_{\text{compute resources}}} \quad \left[\frac{\text{templates}}{\text{core}}\right]

  Reported for both core calculation time (``calc``) and full executable wall time (``wall``).
* **Triggers**: Total count of single-detector gravitational-wave triggers exceeding the SNR threshold (:math:`\text{SNR} \ge 5.5`) and surviving 16-bin Power :math:`\chi^2` and Auto :math:`\chi^2` vetoes and time clustering across all 1,920 segment evaluations.
* **Memory / VRAM**: Resident Set Size (RSS) host RAM consumed for CPU arms, or peak GPU device VRAM allocation for CUDA arms.

.. list-table:: Table 3: Track 1 complete executable campaign (384 compressed templates, 1904 s data, 3 counterbalanced replicates)
   :header-rows: 1
   :widths: 22 10 12 12 12 12 16 10 12

   * - Evaluation Arm
     - Batch :math:`B`
     - Total Wall (s)
     - Core Calc (s)
     - Wall Realtime (s/s)
     - Calc Realtime (s/s)
     - Throughput (templates/core)
     - Triggers
     - Memory / VRAM
   * - **Original CPU** (``40e94792b3``)
     - 1
     - 68.04 ± 0.46
     - 49.12 ± 0.48
     - 28.0 s/s
     - 38.8 s/s
     - **14,885** (calc) / 10,744 (wall)
     - 1,988
     - ~500 MB RAM
   * - **Branch CPU Batched** (Candidate)
     - 128
     - 67.30 ± 0.12
     - 50.60 ± 0.08
     - 28.3 s/s
     - 37.6 s/s
     - **14,449** (calc) / 10,864 (wall)
     - 1,988
     - ~1.2 GB RAM
   * - **JAX CPU Batched** (``jax:cpu``)
     - 16
     - 516.9
     - 470.92
     - 3.68 s/s
     - 4.04 s/s
     - **1,553** (calc) / 1,414 (wall)
     - 2,000
     - ~2.5 GB RAM
   * - **JAX CUDA Batched** (``jax:cuda:0``)
     - 128
     - **11.13 ± 0.006**
     - **2.18 ± 0.002**
     - **171.0 s/s**
     - **873.2 s/s**
     - **335,286** (calc) / 65,667 (wall)
     - 1,996
     - ~1.1 GB VRAM

.. note::

   **Analysis of Executable Campaign Scaling, Batching Dynamics, and Pipeline Bottlenecks**:

   1. **CPU Batching Mechanics vs Cache Saturation**:

      * For single-threaded CPU execution (MKL), processing templates serially (:math:`B=1`) keeps the active
        frequency series within the CPU's high-speed L2/L3 cache hierarchy (~32–64 MiB), achieving 49.12 s core calc time.
      * When batching templates on CPU (:math:`B=128`), the resident memory footprint of 128 template vectors
        exceeds CPU L3 cache capacity, inducing cache thrashing and memory bus contention that increases core calculation
        time to 50.60 s. Consequently, CPU search execution is optimal at :math:`B=1`.
      * **CPU Trigger Parity**: On CPU, candidate execution (both unbatched :math:`B=1` and batched :math:`B=128`)
        achieves **100.0% exact bit-for-bit parity** with the upstream baseline (1,988 / 1,988 triggers,
        zero floating-point difference across all 13 output datasets). In batched CPU execution (:math:`B=128`),
        correlation and SNR arrays are preserved per trigger to maintain strict veto parity while looping through
        the batch.

   2. **Zero-Transfer On-Device Filtering, Fused Autogating, and GPU Power Chi-Square Vetoes**:

      * In isolated kernel microbenchmarks (Table 2), JAX CUDA batching achieves **5,315 tmpl/s**
        at :math:`B=128` (**0.188 ms per template**, delivering a **511× speedup** over CPU matched filtering).
      * In the full executable ``pycbc_inspiral`` search, all core filtering, conditioning, and signal-consistency tests execute
        directly on the GPU without intermediate host memory allocations:

        - Matched filtering (cross-correlation and batched cuFFT) executes in **~24 ms per segment for 128 templates**
          (totaling only **~0.36 s of pure FFT correlation compute** across all 384 templates).
        - Parallel peak clustering (:func:`~pycbc.events.threshold_jax._batched_cluster_core`) screens templates
          simultaneously in **1.48 ms**.
        - Signal-consistency Power :math:`\chi^2` vetoes (:func:`~pycbc.vetoes.chisq_jax.shift_sum`) run on-device via
          a JIT-compiled prefix-sum kernel (:func:`~pycbc.vetoes.chisq_jax._shift_sum_gpu_core`), computing all 16 frequency
          bins in parallel in **1.05 ms per call** without host transfers.
        - Fused JAX autogating (:func:`~pycbc.strain.strain_jax._fused_autogate_pipeline_core`) executes Welch estimation,
          interpolation, and inverse spectrum truncation in a single pass of **13.92 ms** (176× faster than CPU).
        - Batched 2D segment FFTs on GPU complete in **0.32 ms** (1,000× faster than CPU).
        - Waveform plugins defer heavy optional imports (such as external third-party PyTorch or CuPy waveform packages) via lazy proxies, preventing unnecessary CUDA runtime initialization during standard startup.

        Together, these optimizations reduce total pipeline wall time to **11.13 s** and core calculation time to **2.18 s**,
        sustaining **335,286 templates/core (calculation) / 65,667 (wall)**—a **22.52× calculation speedup (6.11× wall speedup) over the MKL CPU baseline**.

   3. **Batching Dynamics and Throughput Accounting**:

      * The throughput metric :math:`\text{Throughput} = \frac{N_{\text{tmpl}} \times T_{\text{data}}}{T_{\text{compute}} \times N_{\text{cores}}}`
        measures the number of templates filtered per unit resource in 1× real time.
      * When batching with :math:`B=128`, all 384 templates across 5 segments (1,920 template-segment evaluations) are filtered in 3 batches in **2.18 s**
        core calculation time, yielding an operational throughput across 1,904 s of analyzed data of:
        :math:`\frac{384 \times 1904\text{ s}}{2.18\text{ s}} \approx 335,286\text{ templates/core}`.
      * Waveform preparation latency during the search loop is eliminated (0.00 s) through in-memory template batch caching in :class:`~pycbc.waveform.bank.FilterBank`.

   4. **Parity and Scientific Equivalence**:

      * JAX CUDA batched execution produces **1,996 triggers**, matching reference triggers with 99.45% parity.
        The small 0.4% trigger count difference (8 triggers) is confined to marginal candidates right at the
        :math:`\text{SNR} = 5.5` boundary due to single-precision float32 rounding differences between cuFFT and PocketFFT/MKL.

   5. **VRAM Footprint and Production Boundaries**:

      * Peak GPU memory allocation scales linearly with batch size: **68 MB** at :math:`B=1`, **972 MB** at :math:`B=16`,
        **4,108 MB** (4.01 GB) at :math:`B=64`, and **8,204 MB** (8.01 GB) at :math:`B=128`.
      * On 24 GB hardware (RTX 4090), :math:`B=64` operates comfortably with >18 GB headroom, while :math:`B=128`
        represents the optimal production ceiling before allocator fragmentation limits.

Latest Pipeline Memory and CUDA Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest full-pipeline campaign on 19 September 2026 is summarized in
:doc:`jax_gpu_investigation`. With batch size 128, the memory changes reduce
peak allocator use from **13.867 to 9.893 GiB** and distinct live buffers at
batch 6 from **3.322 to 1.338 GiB**. Median calculation time changes from
**6.842 to 6.799 s** across three uninstrumented trials per configuration;
this is primarily a memory improvement.

Batch sizes 144 and 160 now complete, but batch 128 remains faster on this
workload. All successful final memory and timing runs preserve **18 scientific
HDF5 datasets and 7,984 triggers exactly**; four performance datasets and HDF5
attributes are excluded from that comparison. The bank repeats 384 templates
four times, so these results need confirmation on a larger independent bank.

Figure 4 shows the optimized B128 production window. Process-specific kernels
are active for **1.763 of 7.560 s (23.3%)**, compared with **1.758 of 7.773 s
(22.6%)** at B160. These instrumented intervals do not measure throughput or
SM occupancy. Sampled process VRAM includes the allocator's reserved pool;
it must not be interpreted as live-buffer use. The compact
`results summary <_static/jax_batch_sweep.json>`_ retains timing repetitions,
memory totals, output comparisons, source hashes and CUDA summary statistics.

Hardware Saturation & VRAM Boundary Analysis
--------------------------------------------

The following measurements describe the standalone kernel benchmarks on the
NVIDIA GeForce RTX 4090 (24,564 MiB VRAM). Their memory limits differ from the
latest full-pipeline batch sweep in :doc:`jax_gpu_investigation`:

1. **Streaming Search (:math:`N = 131,072 = 2^{17}`)**: Memory footprint per batch is lightweight (:math:`\approx 1.07\text{ GiB}` at :math:`B=512`, :math:`\approx 2.15\text{ GiB}` at :math:`B=1024` for complex128; :math:`\approx 1.08\text{ GiB}` at :math:`B=1024` for complex64). Throughput scales monotonically up to :math:`B=1024`, reaching **33,776.1 tmpl/s** (a **194.6× speedup** over the single-threaded MKL CPU baseline, equivalent to **2,161,670 templates in real time**). Beyond :math:`B=1024`, kernel launch overhead is completely amortized and performance becomes compute/memory bandwidth bound.
2. **Offline Inspiral (:math:`N = 2,097,152 = 2^{21}`)**:
   - **Single Precision (``complex64``, Production Standard)**: Each vector row comprises :math:`2^{21}` complex64 points (8 MiB per template). VRAM scales linearly: **68 MB** at :math:`B=1`, **972 MB** at :math:`B=16`, **4,108 MB** (4.01 GB) at :math:`B=64`, and **8,204 MB** (8.01 GB) at :math:`B=128`. At :math:`B=128`, JAX CUDA delivers **5,315.0 tmpl/s** (**511.1× speedup** over single-template MKL CPU, sustaining **2,721,257 templates in real time**). At :math:`B=256`, the combined working tensors (4 GB inputs, 4 GB outputs, 4 GB correlation buffers) plus the 4 GB internal cuFFT scratch buffer hit the default GPU allocator limit on 24 GB hardware, establishing :math:`B=128` as the optimal safe maximum batch size.
   - **Double Precision (``complex128``)**: Each vector row comprises :math:`2^{21}` complex128 points (16 MiB per template). Batch size :math:`B=128` occupies :math:`\approx 8.6\text{ GiB}` in VRAM, delivering **1,779.1 tmpl/s** (**209.3× speedup**, equivalent to **910,900 templates in real time**). Scaling to :math:`B=256` requires :math:`> 20\text{ GiB}` of peak intermediate buffers, exceeding physical 24 GB capacity.
3. **Compressed Banks vs Dynamic Synthesis**: In production CBC searches, waveform generation is eliminated by using compressed banks with inline linear decompression (< 1 ms per template). When dynamic synthesis is required (e.g. for parameter estimation / inference), on-device differentiable waveforms (``diffgw``) eliminate host-to-device transfer and CPU evaluation bottlenecks.

Reproducing JAX Benchmark Measurements
--------------------------------------

The PyCBC JAX performance benchmark suite encompasses three distinct evaluation tiers.
For host isolation controls, single-thread baseline pinning, and platform convergence criteria,
see :ref:`jax-benchmark-protocol`.

1. Kernel Microbenchmarks (Tables 1 & 2) & Scaling Figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To measure the 6-arm FFT cross-correlation rate and peak VRAM across batch sizes (:math:`N \in \{2^{17}, 2^{21}\}`)
and render the scaling figures into Sphinx documentation:

.. code-block:: console

   # Execute isolated kernel microbenchmarks across batch sizes:
   python tools/bench_jax_performance.py \
     --batch-sizes 1 4 16 64 128 256 512 1024 \
     --lengths 131072 2097152 \
     --trials 10 \
     --output artifacts/jax_benchmark_results.json

   # Render publication scaling plots:
   python tools/plot_jax_benchmarks.py \
     --input artifacts/jax_benchmark_results.json \
     --output-dir docs/_static/

2. Complete Executable Search Campaign (Table 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reproduce the end-to-end production search executable (``pycbc_inspiral``) multi-arm campaign on real detector strain data
(Track 1: 384 compressed BNS/NSBH templates, 1,904 s of LIGO Hanford H1 data, 3 counterbalanced replicates):

.. code-block:: console

   # Execute the multi-arm offline inspiral campaign:
   python tools/bench_jax_inspiral_campaign.py \
     --original-source /path/to/pycbc-baseline \
     --branch-source . \
     --python $(which python) \
     --frame-file docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --bank-file inputs/bank_384_compressed.hdf \
     --track track1 \
     --batched \
     --output artifacts/benchmarks/receipt.json \
     --output-dir artifacts/benchmarks

For complete workload definitions, input checksums, and execution instructions for uncompressed dynamic synthesis (Track 2),
see :ref:`jax-reference-campaign`.

3. Prepared Live-Filter Production Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate the prepared live-filter batching API and verify numerical agreement against the standalone NumPy oracle:

.. code-block:: console

   # Run the production live-filter batching harness:
   python tools/bench_production_live_batch.py orchestrate --python $(which python)

For protocol details, streaming batch schedules, and numerical tolerances, see :ref:`jax-batch-numerics`.

4. Process CUDA Timeline and Memory Profiles (Figure 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Nsight Systems to trace CUDA events from the program and its collector.
The collector emits an NVTX clock marker; the extraction tool selects the
child PID recorded in the telemetry receipt. No device-wide transfer
counters enter the resulting CUDA panels.

.. code-block:: console

   nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
     --output=artifacts/program \
     python tools/profile_jax_gpu_timeline.py --nvtx-sync \
     --executable bin/pycbc_inspiral \
     --frame-file docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --bank-file inputs/bank_384_compressed.hdf \
     --output-json artifacts/telemetry.json

   nsys export --type=sqlite --output=artifacts/program.sqlite \
     artifacts/program.nsys-rep
   python tools/extract_jax_nsight_timeline.py \
     --sqlite artifacts/program.sqlite --telemetry artifacts/telemetry.json \
     --output artifacts/timeline.json --bin-ms 100
   python tools/plot_jax_gpu_timeline.py \
     --input artifacts/timeline.json --output artifacts/timeline.png

The published Figure 4 uses the final optimized B128 capture, zoomed to
production filtering. Raw timeline receipts are not checked in. For a new
capture, select the production phase boundaries from ``artifacts/timeline.json``
and pass ``--zoom-start`` and ``--zoom-end`` to the plotting command above.
Use the repeated 1,536-row bank and ``--batch-size 128`` to match the published
workload; the 384-row bank in this example is a smaller capture workload.

Capture memory separately, using the same executable arguments as the search:

.. code-block:: console

   python tools/profile_jax_memory.py --output-dir artifacts/memory -- \
     bin/pycbc_inspiral <search arguments>
   python tools/plot_jax_memory.py \
     --input artifacts/memory/jax_memory_profile.json \
     --output artifacts/memory.png

This produces JAX pprof snapshots and array inventories at selected batches
and after batch cleanup. Snapshot synchronization adds overhead. Allocation
stacks identify live buffers; internal JIT temporaries require compiler memory
reports. The :doc:`investigation <jax_gpu_investigation>` explains the measured
pool, live-buffer and peak-allocation differences.

Legacy telemetry receipts still render their explicitly labeled whole-device
NVML counters. They cannot be converted into process-specific copy events
without a new Nsight capture.

The subsequent memory optimization and batch sweep are documented in
:doc:`jax_gpu_investigation`: batch-128 peak allocator use falls from 13.87 to
9.89 GiB with exact scientific output parity. Batch 160 fits, but batch 128
remains faster on this workload. Reproduce the comparison with:

.. code-block:: console

   python tools/plot_jax_batch_sweep.py \
     --input docs/_static/jax_batch_sweep.json \
     --output artifacts/jax_batch_sweep.png
