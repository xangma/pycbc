# PyTorch Enablement Plan (Full Codebase) for PyCBC

Goal: Run all PyCBC paths on PyTorch (CPU/CUDA, optional MPS) while keeping core code scheme-agnostic and maintaining numerical parity with the CPU baseline. Torch logic should be isolated in torch backend files; avoid touching core modules unless necessary for scheme wiring.

## Environment & Guardrails
- Python: `/Users/xangma/miniconda3/envs/pycbc313/bin/python`
- Repo: `/Users/xangma/Library/CloudStorage/OneDrive-Personal/repos/pycbc`
- Branch: `torch`
- After modifying compiled/imported code, run: `pip install -e .`
- Do **not** use `git reset` without explicit approval.
- Always run (or add) tests for changed functionality.

## 0. Scheme & Detection
- [ ] Add `HAVE_TORCH` detection in `pycbc/__init__.py`.
- [ ] Add `TorchScheme` in `pycbc/scheme.py`; wire `scheme_prefix['torch']`; CLI selection `torch[:device]` (CPU default; CUDA if available; MPS only on request).
- [ ] Update CLI/test helpers to accept torch; ensure `_import_cache/current_prefix` handle torch prefix with clear error logging.

## 1. Core Types (Array/Series)
- [ ] Implement `pycbc/types/array_torch.py` with complete API parity (all schemed methods, arithmetic, reductions, in-place ops) without modifying `array.py`.
- [ ] Preserve dtype, promote to complex when needed; no silent float64 promotion or complex→real truncation; non-inplace ops do not mutate inputs.
- [ ] Pass `python test/test_array.py -s torch`.
- [ ] Extend TimeSeries/FrequencySeries to accept torch tensors and keep metadata scalar; ensure NumPy-only utilities convert explicitly.

## 2. FFT
- [ ] Add torch FFT backend (`backend_torch.py`, `torchfft.py`) using `torch.fft`; integrate with backend_support/class API; torch-only backend selection under torch scheme.

## 3. Filtering / Correlation
- [ ] Torch matched filter backend (`matchedfilter_torch.py`) and BatchCorrelator.
- [ ] Torch equivalents for `threshold_cuda`, `chisq_cuda` (events/vetoes); validate against CPU.

## 4. Waveforms / Noise / PSD / Resampling
- [ ] Replace pycuda waveform kernels with torch (or CPU fallback with explicit transfers); allow LAL outputs to move to torch device with correct dtype.
- [ ] Port noise generation, PSD estimation, resample/decimate, whitening to torch where feasible; explicit CPU fallbacks for SciPy-only functions.

## 5. Tests & CI
- [ ] Add torch scheme to test harness; torch variants for array/timeseries/frequencyseries/fft/matchedfilter; parity tests CPU vs torch for key metrics.
- [ ] Torch jobs in tox/CI (CPU first; CUDA optional).

## 6. Docs & Packaging
- [ ] Requirements/companion: torch CPU default; document CUDA wheel URLs.
- [ ] User docs: torch scheme selection, device flags, limitations (no autograd through LAL).

## 7. Validation & Cleanup
- [ ] Benchmarks CPU vs torch (and legacy CUDA where applicable); tolerance documentation.
- [ ] Deprecation plan for PyCUDA/MKL after torch parity.

Notes: Keep core files (e.g., `array.py`) scheme-agnostic; isolate torch logic in torch backends/modules.
