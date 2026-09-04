.. _torch-optimizations:

Torch optimization controls
===========================

This is the maintainer reference for Torch feature flags. It records route
selection, not a performance promise. Every optional route remains subject to
shape, dtype, device, version, and autograd predicates; an enabled flag can
therefore fall back to an established implementation.

Set process-wide environment flags before importing PyCBC or constructing
plans, live-batch engines, and waveform generators. Some decisions are cached
or captured at construction. Use ``0`` and ``1`` for Boolean settings: several
components also accept common true/false spellings, but invalid-value handling
is component-specific.

Waveform routes
---------------

The component-specific setting has precedence over the global setting, which
has precedence over the listed default.

.. list-table::
   :header-rows: 1
   :widths: 34 15 51

   * - Variable
     - Default
     - Effect
   * - ``PYCBC_TORCH_NATIVE_PORTS``
     - Unset
     - Global request for registered native Torch waveform ports.
   * - ``PYCBC_TORCH_NATIVE``
     - Unset
     - Backward-compatible alias for ``PYCBC_TORCH_NATIVE_PORTS``.
   * - ``PYCBC_TAYLORF2_NATIVE``
     - On
     - Native ``TaylorF2`` regular-grid and sequence ports under Torch.
   * - ``PYCBC_TAYLORF2NLTIDES_NATIVE``
     - On
     - Native ``TaylorF2NLTides`` ports under Torch.
   * - ``PYCBC_TAYLORF2REDSPIN_NATIVE``
     - On
     - Native ``TaylorF2RedSpin`` ports under Torch.
   * - ``PYCBC_TAYLORF2REDSPINTIDAL_NATIVE``
     - On
     - Native ``TaylorF2RedSpinTidal`` ports under Torch.
   * - ``PYCBC_TAYLORF2ECC_NATIVE``
     - On
     - Native ``TaylorF2Ecc`` ports under Torch.
   * - ``PYCBC_SPATPLT_NATIVE``
     - On
     - Selects the separate native Torch ``SPAtmplt`` route when eligible.

“On” means the route is considered. The registered support predicate can
still reject a parameter set, dtype, device, or interface and select an
existing route. The implementation's registered support predicate remains
authoritative; user-facing capability boundaries are in :ref:`torch-scheme`.

FFT, precision, and batch sizing
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Variable
     - Default
     - Effect
   * - ``PYCBC_TORCH_CPU_MKL_IFFT``
     - On
     - Considers the direct MKL CPU inverse-FFT route for validated Linux
       x86-64, ``complex64``, and size combinations. Otherwise the established
       FFT route is used.
   * - ``PYCBC_TORCH_DIRECT_BATCH_IFFT``
     - CUDA: on; CPU/MPS: off
     - Considers direct batched inverse FFT for eligible ``complex64`` batches
       with no autograd requirement.
   * - ``PYCBC_TORCH_CPU_FFTW_BATCH``
     - Off
     - Requests the experimental CPU FFTW batched path when its library,
       shape, and dtype checks pass.
   * - ``PYCBC_TORCH_CUDA_PROMOTED_ROWS``
     - Off
     - Requests the experimental CUDA implementation for promoted
       single-precision batch rows.
   * - ``PYCBC_BATCH_MAXELEMENTS``
     - Hardware-aware
     - Overrides the maximum eligible live-batch element budget. It takes
       precedence over a constructor value, which takes precedence over the
       hardware-aware default.
   * - ``PYCBC_BATCH_TILE_SIZE``
     - Hardware-aware
     - Overrides the positive tile size used by eligible batch work. Invalid
       values are ignored in favor of the hardware-aware default.

On MPS, the accuracy-promoted single-precision batch path stages through CPU
memory. Enabling a related flag does not make that route resident.

Filtering, thresholding, and execution
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 39 17 44

   * - Variable
     - Default
     - Effect
   * - ``PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE``
     - Off
     - Requests the experimental native CPU batched-correlation route.
   * - ``PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE``
     - Off
     - Requests the experimental native CUDA batched-correlation route.
   * - ``PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK``
     - Off
     - Requests native CPU batched peak extraction.
   * - ``PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK``
     - Context-dependent when unset
     - Eligible Triton batch threshold reduction treats unset as enabled; the
       separate native peak-extraction promotion treats unset as disabled. Set
       ``0`` or ``1`` explicitly for reproducible route tests.
   * - ``PYCBC_TORCH_ONDEVICE_PEAKS``
     - Off
     - Keeps eligible peak-processing work on the selected Torch device.
   * - ``PYCBC_TORCH_CPU_THRESHOLD_TRUSTED_ARRAYS``
     - Off
     - Enables a CPU threshold fast path that assumes its array contract has
       already been validated.
   * - ``PYCBC_TORCH_ASYNC_STREAMS``
     - Off
     - Requests eligible asynchronous CUDA stream scheduling.
   * - ``PYCBC_ENABLE_CUDA_GRAPHS``
     - Off
     - Enables eligible CUDA graph capture for live-batch execution. An
       explicit constructor argument takes precedence.
   * - ``PYCBC_TORCH_CUDA_GRAPH``
     - Off
     - Low-level CUDA graph replay request; only the exact value ``1`` enables
       it. Capture and route eligibility are still required.

Compilation
-----------

Compilation is opt-in. Settings other than the master switch do not activate
compilation by themselves.

.. list-table::
   :header-rows: 1
   :widths: 35 18 47

   * - Variable
     - Default
     - Effect
   * - ``PYCBC_TORCH_COMPILE``
     - Off
     - Master switch for eligible ``torch.compile`` routes.
   * - ``PYCBC_TORCH_COMPILE_BACKEND``
     - ``inductor``
     - Backend passed to eligible compilation.
   * - ``PYCBC_TORCH_COMPILE_MODE``
     - ``default``
     - Compilation mode.
   * - ``PYCBC_TORCH_COMPILE_THRESHOLD``
     - On after the master switch
     - Allows compilation of eligible threshold kernels. It is inert while
       ``PYCBC_TORCH_COMPILE=0``.
   * - ``PYCBC_TORCH_COMPILE_VERIFY``
     - Off
     - Runs the implementation's verification behavior for compiled routes.

Compilation can make the first invocation substantially different from steady
state and can create shape-specific caches. Performance artifacts must record
these variables and report cold and warm measurements separately.

Activation and promotion policy
-------------------------------

Defaults-on routes are qualified route choices, not guarantees that an
optimized kernel runs for every input. Defaults-off routes are experimental and
should remain opt-in until all of the following evidence is attached to a
specific revision:

* raw, per-sample performance artifacts with complete software and hardware
  metadata and retained CI provenance;
* parity against an independent reference at supported accuracy boundaries,
  including route and output-device assertions;
* coverage over the claimed device, dtype, shape, batch/bank size, and FFT-size
  matrix, including ineligible fallback cases;
* end-to-end latency, tail latency, throughput, memory/OOM, cold/warm, transfer,
  residency, and deadline-jitter evidence as applicable;
* automatic CI coverage on the device class being promoted; and
* a documented rollback flag and verified established fallback.

A median-only microbenchmark or checked-in plot without its raw source artifact
is not sufficient for promotion. Promotion should be scoped as narrowly as the
evidence, with runtime predicates retaining the safe fallback. Evidence and
plot requirements are in :ref:`torch-performance`; numerical acceptance is in
:ref:`torch-parity`.
