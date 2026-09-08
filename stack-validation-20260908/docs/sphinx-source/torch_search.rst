Torch event processing and search
=================================

Torch event support builds on :doc:`torch_filtering`. Ranking, coincidence,
cuts, significance and live-search code preserve tensor storage through their
numerical calculations. Existing output and event-record interfaces materialize
host values at their serialization boundaries.

``LiveBatchMatchedFilter`` supports Torch batch correlation, inverse FFTs,
peak selection and signal-consistency vetoes. Its data reader can supply
Torch-backed strain; ``StrainBuffer`` maintains device-local overwhitened data
and invalidates cached values when the PSD changes.

CUDA graph and asynchronous stream paths are optional. They can be requested
with the ``enable_cuda_graphs`` and ``enable_async_streams`` constructor
arguments; their default environment switches are ``PYCBC_ENABLE_CUDA_GRAPHS``
and ``PYCBC_TORCH_ASYNC_STREAMS``. Storage, shape, stream and differentiation
checks determine whether an accelerated path is eligible for a particular
batch. These checks do not establish that every search configuration runs
entirely on a device.

The integration suites are ``test_live_batch_torch_fft_integration.py`` and
``test_live_batch_torch_peaks.py`` under ``test/``. Event, peak and storage
regressions are grouped in ``test_torch_event_pipeline.py``,
``test_torch_event_kernels.py``, ``test_torch_peak_contracts.py`` and
``test_torch_cuda_native_peaks.py``. Run the CUDA suites on a CUDA host when
qualifying graph or stream behavior.

Offline symmetric filtering
---------------------------

``MatchedFilterControl`` can capture correlation, inverse FFT and symmetric
clustering on CUDA. Set ``PYCBC_TORCH_CUDA_GRAPH=1`` before running the offline
search, or explicitly call
``control.capture_cuda_graph_symm(segnum, window, template_norm)``. The method
returns whether capture is supported. Capture requires eligible contiguous
complex64 buffers, a fixed analysis window, and the same process, thread and
CUDA stream. Gradient, inference-mode and autocast inputs are ineligible.

Template and segment contents, normalization and threshold can change in place.
Changing bound storage, tensor geometry, methods or the window invalidates the
capture, synchronizes its resources and falls back to eager filtering. Call
``control.clear_cuda_graphs()`` to release captures and permit a fresh opt-in.
Capture or execution failures propagate after cleanup; unsupported bindings
return to eager filtering.

Retained sparse indices and values own their storage. Copy full SNR or
correlation arrays if they must survive the controller's next operation, since
those arrays use its scratch workspace. Real CUDA coverage is in
``test/test_torch_offline_cuda_graph.py``. Recorded performance and its precise
measurement scope are in :ref:`torch-followup-evidence`.
