Torch filtering and vetoes
==========================

Use ``pycbc.scheme.TorchScheme`` to select Torch implementations of matched
filtering, correlation, thresholding, conditioning and signal-consistency
vetoes. The public PyCBC interfaces continue to accept ``Array``,
``TimeSeries`` and ``FrequencySeries`` objects. Construct inputs inside the
scheme so their storage is allocated on the selected device.

This support includes FIR, Butterworth and zero-pole-gain filters; Q transforms;
matched-filter and sky-max statistics; FindChirp and symmetric clustering; and
power, bank, auto and sine-Gaussian chi-square calculations. Supported kernels
can keep the working series on its device, but some preparation and clustering
steps use the host. Filter coefficient design uses NumPy/SciPy. FindChirp
clustering transfers successor indices for a host-side walk and returns sparse
host arrays; its full input series remains on the device. Check the specific
API's transfer boundaries before claiming device residency.

The Torch backend checks storage, shape and automatic-differentiation contracts
before using existing native correlation or chi-square routines. Ineligible
inputs use the corresponding Torch calculation. The optional native batch
correlation routes are controlled by
``PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE`` and
``PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE``; both default to disabled.
See :doc:`torch_search` for offline symmetric CUDA graph capture, eligibility
and result lifetime.

The focused regression suites are ``test_torch_filter_pipeline.py``,
``test_torch_search_kernels.py``, ``test_chisq_torch.py`` and
``test_qtransform.py`` under ``test/``. Device-specific tests skip when their
device is unavailable. A test pass on one device does not qualify another.
