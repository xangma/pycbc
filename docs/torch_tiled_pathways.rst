.. _torch-tiled-pathways:

PyCBC Torch Tiled Execution Pathways
====================================

This document describes the exact execution pathways for matched filtering
in PyCBC across offline (:program:`pycbc_inspiral`) and online streaming
(:program:`pycbc_live`) analyses, detailing how backends, batch sizes, and
shared filtering cores are dispatched.

Current execution pathways matrix
---------------------------------

.. list-table:: Execution pathways by application, backend, and batch size
   :header-rows: 1
   :widths: 18 16 12 24 30

   * - Context
     - Backend
     - Batch (:math:`B`)
     - Controller / Adapter
     - Core filtering mechanism
   * - Offline
     - Standard CPU
     - 1 (default)
     - :class:`~pycbc.filter.matchedfilter.MatchedFilterControl`
     - Scalar MKL/FFTW FFT & correlation using tuned native libraries
   * - Offline
     - Torch CPU
     - 16 (default)
     - :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`
     - Tiled :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft` via
       :class:`~pycbc.filter.gpu_search.core.FilteringWorkspace`
   * - Offline
     - Torch CPU
     - 1 (explicit)
     - :class:`~pycbc.filter.matchedfilter.MatchedFilterControl`
     - Scalar :class:`~pycbc.fft.torchfft.IFFT` &
       :class:`~pycbc.filter.matchedfilter_torch.TorchCorrelator`
   * - Offline
     - Torch CUDA
     - 64 (default)
     - :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`
     - Tiled :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft` via
       batched workspaces
   * - Offline
     - Torch CUDA
     - 1 (explicit)
     - :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`
     - Tiled :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft`
       (1-row tile pass)
   * - Online / Live
     - Torch CUDA / CPU
     - Any (:math:`B \ge 1`)
     - :class:`~pycbc.filter.gpu_search.adapter.TiledLiveBatchMatchedFilter`
       / :class:`~pycbc.filter.gpu_search.engine.SearchEngine`
     - Shared :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft`
       preserving live streaming semantics

Offline matched-filtering pathways
----------------------------------

In :program:`pycbc_inspiral`, the filtering route is chosen based on the
processing scheme and batch size configuration:

1. **Standard CPU (:math:`B=1`, default)**:
   Executes single-template matched filtering through the canonical
   :class:`~pycbc.filter.matchedfilter.MatchedFilterControl`
   using tuned native CPU libraries (FFTW or Intel MKL) and existing
   clustering and veto pipelines.

2. **Torch CPU (:math:`B=16` default, or :math:`B=1` scalar)**:
   By default, selecting the Torch CPU scheme activates batched filtering
   at :math:`B=16` through
   :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`
   backed by :class:`~pycbc.filter.gpu_search.core.FilteringWorkspace`.
   If scalar filtering (:math:`B=1`) is explicitly configured, it dispatches
   through :class:`~pycbc.filter.matchedfilter.MatchedFilterControl`
   using 1D FFT wrappers (:class:`~pycbc.fft.torchfft.IFFT` and
   :class:`~pycbc.filter.matchedfilter_torch.TorchCorrelator`).

3. **CPU (:math:`B > 1`, NumPy or Torch)**:
   When batching is configured on CPU (with either NumPy or Torch CPU),
   filtering dispatches through
   :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`.
   It uses the vectorised
   :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft` kernel
   backed by :class:`~pycbc.filter.gpu_search.core.FilteringWorkspace`.

4. **Torch CUDA (:math:`B=64` default, :math:`B=1` 1-row tile)**:
   Selecting the CUDA scheme activates batched filtering at :math:`B=64` by
   default. Both :math:`B=1` and :math:`B > 1` CUDA filtering execute through the same
   unified tiled filtering core. Rather than invoking scalar FFT wrappers
   for :math:`B=1`,
   :class:`~pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl`
   executes a 1-row tile pass directly through
   :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft`. This eliminates
   redundant code paths and unifies numerical invariants across batch sizes
   on GPU.

Online streaming pathways
-------------------------

Online low-latency analysis (:program:`pycbc_live`) uses
:class:`~pycbc.filter.gpu_search.adapter.TiledLiveBatchMatchedFilter` and
:class:`~pycbc.filter.gpu_search.engine.SearchEngine`.

- **Shared Core**: The underlying correlation and inverse FFT operations use
  the exact same :func:`~pycbc.filter.gpu_search.core.correlate_and_ifft` and
  :class:`~pycbc.filter.gpu_search.core.FilteringWorkspace` implementations
  as the offline tiled adapter.
- **Preserved Live Semantics**: Live analysis retains distinct operational
  semantics: frequency-domain streaming data blocks, selection of the single
  loudest trigger per template block, fixed valid intervals, and optional
  asynchronous queue drain.

Batch and block preparation
--------------------------

The offline template loop prepares a contiguous batch lazily, when its first
active segment is encountered, and reuses it for subsequent segments. Full
batches reuse the same tensor; contiguous subsets use slices, and noncontiguous
subsets gather the requested rows in order. Template metadata and PSD-dependent
normalization remain separate. The prepared tensor belongs to the current bank
batch, so reuse of the bank's output buffers cannot retain an earlier batch.

The live engine reuses the latest reciprocal PSD for a CUDA tensor whose identity
and mutation counter are unchanged. A changed plan or tensor invalidates it.
CPU, NumPy, inference-mode, and autograd-enabled PSD tensors are recomputed;
an explicit workspace byte limit also disables the persistent reciprocal cache.
CUDA events preserve readiness across caller streams and workspace slots.
Full-band overwhitened data is copied only when a veto manager needs it, and
unit strain scaling avoids an intermediate device tensor. Caller inputs remain
unchanged, including their values outside the filtering cutoff.
Torch-backed PyCBC series expose their native tensor through the public backend
accessor, avoiding a host round trip when the data already resides on CUDA.

Preparation optimization measurements (2026-09-10)
-------------------------------------------------

Separate A/B/B/A campaigns on an RTX 4090 compared these preparation changes
with the shared tiled core at ``ed7099dd3157303445be1a00f853d0900d363f59``.
Each table entry is the median of two independent process measurements.
The :download:`raw receipts and qualification summary
<data/torch-20260910/preparation-optimization-summary.json>` retain commands,
source hashes, individual timings, and the comparison scope.

The offline campaign used 128 TaylorF2 templates, synthetic H1 noise, and five
segments of 2097152 samples. All eight runs produced 215 triggers with exact
agreement in all 13 top-level H1 trigger datasets, including power chi-square.

.. list-table:: Complete pycbc_inspiral and internal filtering time
   :header-rows: 1

   * - Batch
     - Baseline wall (s)
     - Updated wall (s)
     - Baseline filtering (s)
     - Updated filtering (s)
   * - 64
     - 44.559
     - 44.398
     - 2.107
     - 2.102
   * - 128
     - 44.212
     - 44.395
     - 2.081
     - 2.064

These differences are within the observed process variation; they do not
establish a meaningful end-to-end inspiral speedup. This synthetic campaign
does not replace the compressed-bank, real-frame reference campaign.

The live campaign measured ``SearchEngine.submit`` plus ``drain`` for 128
templates, 64-row tiles, and 65536-sample blocks, with two workspace slots,
asynchronous transfers, CUDA graphs, and no veto manager. Each process measured
120 blocks per case after eight warmup blocks. The table compares the per-run
median block latency.

.. list-table:: Live engine block latency
   :header-rows: 1

   * - Case
     - Baseline (ms)
     - Updated (ms)
     - Reduction
   * - Empty
     - 1.117
     - 1.021
     - 8.5%
   * - Triggered
     - 1.757
     - 1.630
     - 7.3%

All live result hashes matched, including 15360 candidates per triggered run,
and allocated GPU memory did not grow during any measured run. These results
cover the engine; they do not measure the complete ``pycbc_live`` executable,
data acquisition, or optional vetoes. A candidate-export coalescing experiment
was slower and is excluded from the implementation.

Defaults, scheme activation, and execution modes
------------------------------------------------

- **Scheme-Specific Batch Defaults**:
  Selecting a processing scheme activates its respective default batch size:

  - Standard CPU defaults to :math:`B=1` (scalar matched filtering).
  - Torch CPU defaults to :math:`B=16`.
  - Torch CUDA defaults to :math:`B=64`.

  Switching schemes via ``--processing-scheme`` establishes these defaults
  automatically unless overridden by explicit configuration.

- **Offline Has No Graph Capture**:
  Offline analysis (:program:`pycbc_inspiral`) has no CUDA graph-capture
  execution path. Graph capture is exclusively available in the online
  streaming engine (:class:`~pycbc.filter.gpu_search.engine.SearchEngine`
  in :program:`pycbc_live`). The offline adapter processes segments
  synchronously, and CUDA graph execution cannot be enabled offline.

- **CPU Scalar Acceleration**:
  Standard CPU scalar wrappers leverage tuned native libraries (such as
  Intel MKL or FFTW) with single-template pipelines.

- **Floating-Point Reduction and Numerical Agreement**:
  Different batch sizes alter tensor dimensions, reduction tree orders, and
  accumulation sequences in batched FFTs and correlations. Consequently,
  there is no guarantee of strict bit-identical output across all batch sizes
  :math:`B`, though physical candidates agree within validated scientific
  tolerances.

Route-aware profiling and trace interpretation
----------------------------------------------

Profiler instrumentation in :file:`tools/profile_torch_filtering.py` is
route-aware:

- **Offline pycbc_inspiral Only**:
  The profiling helper is designed specifically and exclusively for offline
  :program:`pycbc_inspiral` template loops. It does not instrument or execute
  online streaming workflows or live CUDA graphs.
- **No Scalar FFT Assumptions**: Tiled runs (including :math:`B=1` CUDA) are
  profiled without assuming each tile corresponds to a single scalar FFT call.
- **Separate Metrics**: The profiler counts executed template rows and tile
  calls separately, logging the exact sequence of tile sizes (including tail
  tiles).
- **Non-Synchronizing Semantic Ranges**: Operations are annotated using
  :func:`torch.profiler.record_function` (e.g.
  ``pycbc::core_correlate_and_ifft``, ``pycbc::core_mul``,
  ``pycbc::core_ifft``, ``pycbc::select_candidates``,
  ``pycbc::batched_power_chisq``) without device synchronization during
  the loop. CUDA synchronization occurs strictly at outer measurement
  boundaries.
- **Opt-In CPU Tracing**: The ``--cpu`` flag enables PyTorch CPU profiling
  using CPU-only profiler activities and omitting all CUDA synchronization.
- **Receipt Integrity**: Instrumented runs are explicitly flagged as
  ineligible for throughput benchmarking. Completed traces must always be
  interpreted within their defined semantic scopes and paired with
  unprofiled runs to confirm scientific numerical parity.
