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

Offline search (``pycbc_inspiral``) on Torch CUDA utilizes the persistent tiled
GPU search engine via ``pycbc.filter.gpu_search.adapter.TiledMatchedFilterControl``.
Waveforms are grouped into tiles ($B=64$ by default on CUDA) and filtered concurrently
with native GPU multi-rate planning, fused veto evaluation, and stream-managed
CUDA graph execution managed by ``pycbc.filter.gpu_search.graphs.CUDAGraphManager``.

On CPU, ``TorchScheme(device="cpu")`` is supported through ``TiledMatchedFilterControl``
with user-configurable batch sizes ($B=16$ default), while standard single-template
CPU processing uses ``MatchedFilterControl`` to optimize L2/L3 cache locality.
For full integration test coverage, see ``test/test_gpu_search_adapter.py`` and
``test/test_gpu_search_qualification.py``.
