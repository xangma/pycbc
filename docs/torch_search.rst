Torch event processing and search
=================================

Torch event support builds on :doc:`torch_filtering`. Selected ranking,
coincidence, cut, significance and live-search kernels preserve compatible
tensor storage. Search orchestration, template/PSD setup and trigger assembly
can still use NumPy or host values before serialization. Support for these
kernels does not establish residency of a complete search.

``LiveBatchMatchedFilter`` supports Torch batch correlation, inverse FFTs,
peak selection and signal-consistency vetoes. Its data reader can supply
Torch-backed strain; ``StrainBuffer`` can cache overwhitened data on that device.
It clears interpolated PSD caches when replacing or invalidating its PSD, and
clears cached overwhitened segments when advancing the strain buffer.

CUDA graph and asynchronous stream paths default to disabled. Request them
with the ``enable_cuda_graphs`` and ``enable_async_streams`` constructor
arguments. Each defaults to ``None``, which reads ``PYCBC_ENABLE_CUDA_GRAPHS``
or ``PYCBC_TORCH_ASYNC_STREAMS``, respectively. The environment enables a path
with ``1``, ``true``, ``yes`` or ``on`` (ignoring case and surrounding spaces);
an explicit boolean argument takes precedence. Storage, shape, stream and
differentiation checks still determine eligibility for a particular batch.
See :ref:`torch-optimizations` for the other optional routes and their defaults.

The integration suites are ``test_live_batch_torch_fft_integration.py`` and
``test_live_batch_torch_peaks.py`` under ``test/``. Event, peak and storage
regressions are grouped in ``test_torch_event_pipeline.py``,
``test_torch_event_kernels.py``, ``test_torch_peak_contracts.py`` and
``test_torch_cuda_native_peaks.py``. ``test_live_batch_veto_reuse.py`` checks
correlation workspace reuse through veto calculation;
``test_torch_cuda_peak_host_read.py`` checks that returned host arrays are
ready for immediate use. Run the CUDA suites on a CUDA host when qualifying
graph or stream behavior.

Offline symmetric filtering
---------------------------

``MatchedFilterControl`` can capture correlation, inverse FFT and symmetric
clustering on CUDA. This path defaults to disabled and is separate from the
live-batch graph option above. Call
``control.capture_cuda_graph_symm(segnum, window, template_norm)`` to attempt
capture; it returns whether capture succeeded and enables replay on success.
``PYCBC_TORCH_CUDA_GRAPH=1`` only requests replay of existing captures. Setting
the environment variable alone does not capture a graph or accelerate an
ordinary offline executable run. Capture requires Triton,
eligible contiguous complex64 buffers on the current CUDA device, a single
FFT, a fixed analysis window and clustering window, and sequential calls in
the same process, thread and CUDA stream. Gradient, inference-mode and autocast
inputs are ineligible.

Template and segment contents, normalization and threshold can change in place.
Changing bound storage, tensor geometry, methods or the window invalidates the
capture, synchronizes its resources and falls back to eager filtering. Call
``control.clear_cuda_graphs()`` to release captures and permit a fresh opt-in.
Capture or execution failures propagate after cleanup; unsupported bindings
return to eager filtering. CUDA graphs inherited across a fork cannot be
released or reused in the child; that case raises an error. Create the CUDA
controller in the process that will use it.

Retained sparse indices and values own their storage. Copy full SNR or
correlation arrays if they must survive the controller's next operation, since
those arrays use its scratch workspace. Real CUDA coverage is in
``test/test_torch_offline_cuda_graph.py``.
