.. _torch-optimizations:

Torch optimization controls
===========================

Use these flags to request optional implementations. Shape, dtype, device,
version, and gradient requirements determine whether they run; an enabled
flag can still use the established fallback. Flags do not guarantee a speedup.
For setup, see :ref:`torch-runtime`; for benchmark comparison requirements, see
:ref:`torch-performance`.

The `restoration evidence
<https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration>`_
records historical main ``aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`` against
original CPU ``40e94792b3edf59f39b18b65102b28a4f74433a7``. Its CPU controls
produced identical scientific output on that workload; both Torch routes
failed the unchanged scientific gates. The later Torch compatibility changes
and their qualification are in :ref:`torch-current-qualification`.
No new performance samples or speedup are claimed.

Set process-wide environment flags before importing PyCBC or constructing
plans, live-batch engines, and waveform generators. Some decisions are cached
or captured at construction. Use ``0`` and ``1`` for Boolean settings: several
components also accept common true/false spellings, but invalid-value handling
is component-specific.

Waveform routes
---------------

The component-specific setting has precedence over the global setting, which
has precedence over the listed default. If both global flags are set,
``PYCBC_TORCH_NATIVE_PORTS`` takes precedence over ``PYCBC_TORCH_NATIVE``.

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

The explicit ``get_fd_waveform_batch("TaylorF2", ...)`` interface separately
supports ``PYCBC_TAYLORF2_TRITON=1`` (default ``0``). This per-call setting selects
a fused double-precision CUDA evaluator when Triton is available and inputs
carry no reverse- or forward-mode gradients. Other calls retain the Torch
evaluator. The scalar registry flags above do not select this batch evaluator.
The first eligible call may compile a kernel; measure cold and warmed public
calls separately. Compilation and launch errors propagate to the caller.

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
     - Considers eligible single-transform Torch CPU IFFT plans on Linux x86-64:
       direct ``complex64`` at 32768 samples, or promoted ``complex128`` work
       at 1048576, 2097152 and 4194304 samples with one Torch thread. Public
       input/output remain ``complex64``; the 2097152-sample plan uses one
       private in-place workspace. Ineligible calls retain the established
       FFT route.
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
     - Requests a CPU threshold fast path that assumes its array contract has
       already been validated.
   * - ``PYCBC_TORCH_ASYNC_STREAMS``
     - Off
     - Requests eligible asynchronous CUDA stream scheduling. An explicit
       ``enable_async_streams`` constructor argument takes precedence.
   * - ``PYCBC_ENABLE_CUDA_GRAPHS``
     - Off
     - Requests eligible CUDA graph capture for live-batch execution. An
       explicit constructor argument takes precedence.
   * - ``PYCBC_TORCH_CUDA_GRAPH``
     - Off
     - Requests replay of an already captured offline symmetric-filter graph;
       only the exact value ``1`` enables this request. It does not create a
       capture. Explicit successful capture also enables replay; see
       :doc:`torch_search` for the capture and clear APIs.

Original CPU compatibility within Torch
----------------------------------------

Eligible ordinary Torch tensors on CPU/CUDA use the original CPU PSD
estimation pipeline and float32 strain-segment forward FFT on copies, NumPy's
serial float32 cumulative-power scan, and the original complex64 sparse point
chi-square calculation and postprocessing. The point calculation uses CPU
views or copies from CUDA. These are Torch-side compatibility routes; the
original CPU implementation is unchanged. They preserve the output dtype and
device, with host work and transfers for CUDA inputs. Specialized storage,
gradient-aware and MPS routes keep their existing eligibility and behavior.
See :ref:`torch-current-qualification` for the measured scope. Existing
optimization flags do not establish that the calculation stayed on the GPU.

Original CPU FFT behavior
-------------------------

In historical measured main ``aa6b795a63``, the shared CPU backend files
``pycbc/fft/mkl.py``, ``pycbc/fft/fftw.py`` and ``pycbc/fft/npfft.py`` are
byte-identical to original CPU ``40e94792b3``. The shared MKL descriptor-cache
addition was removed. Planner locking and retained workspaces for eligible
Torch routes belong to the Torch FFT implementation.

Optional #16 retains separate general FFT changes; optional #17 retains
separate CPU/native changes. Neither leaf is part of the main scientific
qualification or required to preserve its default CPU behavior.

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

Defaults-on routes are considered only when their eligibility checks pass;
enabling a route does not establish scientific qualification. Defaults-off routes are experimental and
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
plot requirements are in :ref:`torch-benchmark-protocol`; numerical acceptance
is in :ref:`torch-parity`.
