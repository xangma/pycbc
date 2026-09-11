.. _torch-performance:
.. _torch-performance-summary:
.. _torch-inspiral-optimized:
.. _torch-optimization-results:
.. _torch-benchmark-status:

Measuring Torch performance
===========================

Compare backends using the same inputs, scientific settings and completed
work. Record the exact source revisions and test numerical agreement before
collecting timings. Device selection alone does not establish a speedup:
setup, transfers and host work can dominate small workloads.

Offline benchmarks: end-to-end vs calculation-stage time (pycbc_inspiral)
-------------------------------------------------------------------------

Measured on host ``len`` (NVIDIA GeForce RTX 4090, AMD Ryzen Threadripper
PRO 3995WX, pinned to logical core 8 with numerical thread pools fixed to
one). The production workload contains **384 compressed BNS/NSBH templates,
five segments of 512 s at 4096 Hz** (2,097,152 samples per segment;
1,920 template/segment pairs, 731,136 template-seconds), with 16-bin power
:math:`\chi^2` and 1.0 s symmetric clustering.

To establish an accurate performance picture, historical measurements are
split into:

1. **Full end-to-end executable wall time**: Process launch through completed
   HDF output, including startup, frame I/O, strain conditioning, PSD
   estimation, template preparation, filtering, vetoes, and output writing.
2. **Calculation-stage filtering time**: The duration of the template loop
   within ``batch_template_triggers``, including template bank lookups,
   frequency-domain correlation, inverse FFTs, peak clustering, Power
   :math:`\chi^2` veto evaluation, and event consolidation.

.. list-table:: Table 1: Historical end-to-end wall times (pycbc_inspiral)
   :header-rows: 1
   :widths: 22 14 14 14 18 18

   * - Route / Configuration
     - Batch ($B$)
     - Setup & I/O (s)
     - Total Wall (s)
     - Wall Speedup
     - Throughput (mf/s)
   * - Standard CPU (MKL baseline)
     - 1
     - 15.87 s (24.4%)
     - 65.15 s
     - 1.00×
     - 29.5 mf/s
   * - Candidate CPU
     - 1
     - 14.53 s (22.8%)
     - 63.81 s
     - 1.02×
     - 30.1 mf/s
   * - Torch CPU (single-thread)
     - 1
     - 11.60 s (11.7%)
     - 98.96 s
     - 0.66×
     - 19.4 mf/s
   * - Sequential Torch CUDA (M5)
     - 1
     - 10.30 s (50.5%)
     - 20.41 s
     - 3.19×
     - 94.1 mf/s
   * - **Batched Torch CUDA (GPU Engine)**
     - **64**
     - **10.41 s (60.2%)**
     - **17.29 s**
     - **3.77×**
     - **111.0 mf/s**
   * - Batched Torch CUDA (GPU Engine)
     - 128
     - 10.30 s (60.8%)
     - 16.95 s
     - 3.84×
     - 113.3 mf/s

.. list-table:: Table 2: Historical calculation-stage times (pycbc_inspiral)
   :header-rows: 1
   :widths: 22 14 16 18 18 14

   * - Route / Configuration
     - Batch ($B$)
     - Calculation Stage (s)
     - Stage Share
     - Stage Speedup
     - Rate (mf/s)
   * - Standard CPU (MKL baseline)
     - 1
     - 49.28 s
     - 75.6%
     - 1.00×
     - 39.0 mf/s
   * - Candidate CPU
     - 1
     - 49.28 s
     - 77.2%
     - 1.00×
     - 39.0 mf/s
   * - Torch CPU (single-thread)
     - 1
     - 87.36 s
     - 88.3%
     - 0.56×
     - 22.0 mf/s
   * - Sequential Torch CUDA (M5)
     - 1
     - 10.11 s
     - 49.5%
     - 4.87×
     - 189.9 mf/s
   * - **Batched Torch CUDA (GPU Engine)**
     - **64**
     - **6.88 s**
     - **39.8%**
     - **7.16×**
     - **279.1 mf/s**
   * - Batched Torch CUDA (GPU Engine)
     - 128
     - 6.65 s
     - 39.2%
     - 7.41×
     - 288.7 mf/s

**Amdahl's Law and Residual Interpretation**:
In these historical measurements on ``len``, the calculation-stage filtering
is accelerated **7.16× to 7.41×** on GPU via batching. In ``pycbc_inspiral``,
the reported setup timer (:math:`t_{\text{setup}}`) strictly precedes
``batch_template_triggers``, while the calculation stage encompasses
template bank lookups, template preparation, correlation, inverse FFTs,
peak clustering, vetoes, and event consolidation. The difference between
total wall time and calculation time is an unattributed residual containing
startup, Python module imports, frame I/O, strain conditioning, PSD
estimation, framework overhead, and HDF5 output writing. Without direct
sub-stage instrumentation, this residual cannot be assigned to specific
phases or assumed to be a fixed CPU fraction. The observed **3.77×** wall
speedup is an empirical measurement for this 384-template workload, not a
theoretical ceiling. For larger template banks or longer segments, fixed
startup overhead amortizes over more templates, narrowing the gap between
executable wall speedup and filtering-stage speedup.

All four routes produced exactly **2,203 pre-cut raw triggers** with
identical SNRs and arrival times. After NewSNR thresholding (5.0), scalar
routes yield 1,988 triggers and batched routes yield 1,989 triggers, with
1,986 triggers being strictly bit-for-bit identical and the single difference
arising from a knife-edge threshold crossing (:math:`\text{NewSNR} = 5.0001`
vs :math:`4.9999` from fused GPU accumulation) as detailed in the
qualification report.

Streaming live results: end-to-end vs calculation-stage time (pycbc_live)
-------------------------------------------------------------------------

The low-latency online search (``pycbc_live``) uses
:class:`~pycbc.filter.matchedfilter.LiveBatchMatchedFilter` (or
:class:`~pycbc.filter.gpu_search.adapter.TiledLiveBatchMatchedFilter`) on
shorter frequency-domain blocks (:math:`N = 131,072` at 2048 Hz).

Because ``pycbc_live`` is a long-running streaming service, its performance
separates into:

1. **Calculation-stage latency & throughput per block**: The time required to
   filter a streaming block of data against the template bank (tensor
   correlation, batched IFFTs, peak extraction, and vetoes).
2. **Full end-to-end wall time & stream amortization**: Total elapsed time
   including one-time workspace preallocation, CUDA graph capture, and memory
   staging, evaluated across different streaming durations.

.. list-table:: Table 3: Historical calculation-stage rates (1024 templates)
   :header-rows: 1
   :widths: 22 14 18 18 14

   * - Backend / Configuration
     - Batch ($B$)
     - Latency / block (ms)
     - Throughput (wf/s)
     - Stage Speedup
   * - Standard CPU (MKL/FFTW, 1 thread)
     - 1
     - 886.2 ms
     - 1,155 wf/s
     - 1.00×
   * - Torch CPU (1 thread)
     - 1
     - 4,931.0 ms
     - 208 wf/s
     - 0.18×
   * - Torch CPU (8 threads)
     - 64
     - 519.3 ms
     - 1,972 wf/s
     - 1.71×
   * - Sequential Torch CUDA (M5)
     - 1
     - 422.4 ms
     - 2,424 wf/s
     - 2.10×
   * - Batched Torch CUDA
     - 32
     - 33.7 ms
     - 30,393 wf/s
     - 26.3×
   * - Batched Torch CUDA
     - 64
     - 34.8 ms
     - 29,425 wf/s
     - 25.5×
   * - Batched Torch CUDA
     - 128
     - 31.2 ms
     - 32,764 wf/s
     - 28.4×
   * - **Batched Torch CUDA (Peak)**
     - **1024**
     - **19.3 ms**
     - **53,005 wf/s**
     - **45.9×**
   * - **Batched Torch CUDA + CUDA Graphs**
     - **64**
     - **22.6 ms**
     - **45,310 wf/s**
     - **39.2×**

.. list-table:: Table 4: Historical end-to-end stream latency (1024 templates)
   :header-rows: 1
   :widths: 24 16 20 20 20

   * - Stream Duration
     - Waveforms Evaluated
     - Standard CPU (s)
     - Torch CUDA Engine (s)
     - End-to-End Speedup
   * - Setup / Initialization
     - 0 (one-time setup)
     - 1.54 s
     - 2.16 s
     - 0.71× (allocation)
   * - Cold Start (1 block)
     - 1,024
     - 5.53 s
     - 2.51 s
     - **2.20×**
   * - Burst Stream (10 blocks, 10 s)
     - 10,240
     - 10.40 s
     - 2.35 s
     - **4.43×**
   * - Extended Stream (50 blocks, 50 s)
     - 51,200
     - 45.85 s
     - 3.13 s
     - **14.65×**
   * - Sustained Production (500 blocks)
     - 512,000
     - 444.6 s
     - 11.81 s
     - **37.65×**

**Setup Amortization in Low-Latency Analysis**:
Because ``pycbc_live`` runs as a continuous service over hours or days, its
one-time 2.16 s startup and workspace allocation overhead is rapidly
amortized across incoming data blocks. While a single cold block exhibits a
modest 2.2× speedup due to initial device memory staging, sustained production
streaming rapidly approaches the calculation-stage speedup (**37.6× to
45.9×**). Per-block calculation latencies remain under **35 ms** (and
**22.6 ms** with CUDA graphs), comfortably exceeding sub-second low-latency
alert requirements.

Architectural distinction: inspiral vs live batching
----------------------------------------------------

1. **Transform length and memory footprint**:

   - In ``pycbc_inspiral``, :math:`N = 2,097,152` samples (512 s at 4096 Hz).
     A full-length complex64 buffer is 16 MiB (:math:`N = 2^{21}`, with
     half-frequency templates taking 8 MiB). Working buffers rapidly exceed
     on-chip cache capacity at higher batch sizes.
   - In ``pycbc_live``, :math:`N = 131,072` samples (64 s at 2048 Hz).
     A single waveform tensor is 1.0 MiB. At :math:`B=64`, one complex64
     tensor is 64 MiB. Together with template inputs, input data, and
     output workspaces, the total memory footprint exceeds typical GPU L2
     cache sizes (e.g. 72 MiB on an RTX 4090). Determining memory-bound
     vs latency-bound behavior requires direct profiling with hardware
     performance counters rather than cache-residence assumptions.

2. **Template generation and filtering pipeline**:

   - ``pycbc_inspiral`` decompresses templates during the filtering loop to
     maintain a bounded host memory footprint, evaluates clustering over
     valid intervals, and computes Power :math:`\chi^2` vetoes on surviving
     candidates.
   - ``pycbc_live`` maintains pre-allocated batch workspaces, performs
     vectorized peak extraction per template, and evaluates vetoes on the
     loudest candidate per block.
   - For complete details on route dispatch across offline and live
     workflows, see :ref:`torch-tiled-pathways`.

Profiling methodology and correcting legacy performance claims
--------------------------------------------------------------

Accurate acceleration analysis requires a strict distinction between host
execution and on-device GPU kernels:

- **Host cProfile Percentages are NOT GPU Breakdowns**: Host profiling tools
  such as Python's :mod:`cProfile` measure CPU instruction execution,
  Python interpreter dispatch overhead, and C extension call latencies.
  Host cProfile percentages cannot be reported as GPU stage durations or
  device bottlenecks, because GPU kernels execute asynchronously on the
  device stream.
- **Isolated Microbenchmarks vs Host Loop Time**: Quoting an isolated kernel
  duration (such as a 35 µs FFT execution) against host-side Python loop
  durations cannot establish device bottlenecks or prove that FFT represents
  an insignificant portion of actual GPU work. Stage breakdowns must be
  measured on-device via semantic ranges in trace tools (e.g.
  :file:`tools/profile_torch_filtering.py`) rather than synthesizing host
  and device numbers.
- **Instrumented Traces vs Benchmark Throughput**: Instrumented profiling runs
  introduce measurement overhead and must remain strictly ineligible for
  throughput claims. Traces should be used solely to verify call counts,
  tile sizes, and semantic scopes, and every profiled configuration must be
  paired with an unprofiled benchmark run confirming numerical scientific
  parity.



Measured sources: original CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7`` and candidate main
``ca4bc95f7fe54a4d52d3923c3e8b0d4b283ce196``. See the
`immutable benchmark evidence <https://github.com/xangma/pycbc/tree/134ecb2586b2e2fc6272924e9cffdf15e51e6f39/torch-fresh-benchmark>`_
for every sample, command, input hash, environment record and independent
verification, and :ref:`torch-reference-campaign` for the workload and gates.

Choose the measurement boundary
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Measurement
     - Included work
     - Definition
   * - Complete executable
     - Process startup, frame I/O, conditioning, PSD estimation, template
       preparation, filtering, vetoes and completed HDF output.
     - :ref:`torch-reference-campaign`
   * - Prepared live-filter API
     - Calls to ``LiveBatchMatchedFilter.process_data`` with prepared
       frequency-domain inputs, including filtering, peak selection
       and vetoes.
     - :ref:`torch-batch-numerics`
   * - Individual operation
     - A declared kernel or public API call, with allocation, transfer,
       compilation and synchronization costs identified explicitly.
     - :ref:`torch-benchmark-protocol`

Report these boundaries separately. A prepared API rate does not include the
startup and preparation costs of an executable. An individual operation's
speedup does not establish the speedup of a complete search.

Compare equivalent work
------------------------

Use the unchanged CPU reference to check preservation of existing behavior,
and candidate normal CPU as a dispatch control for Torch CPU and CUDA.
Require all frozen scientific gates to pass before equivalent-output timing.
The executable workload and its numerical tolerances are defined in
:ref:`torch-reference-campaign`; the live-filter fixture has a separate
:ref:`numerical method <torch-batch-numerics>`.

Run fresh unprofiled processes in rotating backend order. Separate cold calls,
warm calls and profiled runs; synchronize accelerators around timed regions.
Include transfers and host work inside the boundary being measured. Record
thread limits, allocated cores and GPUs, memory use and host load. Follow
:ref:`torch-benchmark-protocol` for convergence and full-machine experiments.

Report results with their evidence
----------------------------------

For each source and backend, retain input and output hashes, full commands,
environment and native-build identities, test results, scientific comparisons
and every timing sample. Publish medians and observed ranges with the
completed-work denominator and resource allocation. Keep raw results and
profiles in an immutable evidence archive associated with the measured
revision.

Qualification runs establish correctness within their tested scope; they are
not performance samples. Finite-workload measurements do not establish
sustained or full-machine capacity. Run the relevant checks in
:ref:`torch-testing` before benchmarking a changed API.
