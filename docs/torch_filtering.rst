Torch filtering and vetoes
==========================

Use ``pycbc.scheme.TorchScheme`` to select Torch implementations of matched
filtering, correlation, thresholding, conditioning and signal-consistency
vetoes. The public PyCBC interfaces continue to accept ``Array``,
``TimeSeries`` and ``FrequencySeries`` objects. Construct inputs inside the
scheme so their storage is allocated on the selected device.

This support includes FIR, Butterworth and zero-pole-gain filters; Q transforms;
matched-filter and sky-max statistics; FindChirp and symmetric clustering; and
power, bank, auto and sine-Gaussian chi-square calculations. Thresholding keeps
the working series on its device and transfers selected results where the
existing public interface requires host values.

The Torch backend checks storage, shape and automatic-differentiation contracts
before using existing native correlation or chi-square routines. Ineligible
inputs use the corresponding Torch calculation. The optional native batch
correlation routes are controlled by
``PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE`` and
``PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE``; both default to disabled.
CUDA graph capture for symmetric single-filter clustering is exposed through
``MatchedFilterControl.capture_cuda_graph_symm``.

The focused regression suites are ``test_torch_filter_pipeline.py``,
``test_torch_search_kernels.py``, ``test_chisq_torch.py`` and
``test_qtransform.py`` under ``test/``. Device-specific tests skip when their
device is unavailable. A test pass on one device does not qualify another.
