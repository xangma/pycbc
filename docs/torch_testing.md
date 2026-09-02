# Torch Testing Quickstart

PyCBC’s test helpers already understand the torch scheme. You can run any
scheme-aware test with:

```bash
python test/test_array.py -s torch           # CPU torch
python test/test_array.py -s torch:cuda      # specific CUDA device (0 default)
python test/test_array.py -s torch:mps       # Apple MPS (dtype limits apply)
```

## Non-waveform execution boundary

In compatible `TorchScheme` paths, bulk numerical computation stays resident
on the selected Torch device.  This includes PyCBC array arithmetic, indexing,
reductions, and accumulations; FFTs and digital filtering; PSD generation,
estimation, interpolation, whitening, inverse-spectrum truncation, noise,
resampling, gating, and strain-buffer processing; matched-filter correlation,
Q-transform tiles, chi-squared and veto statistics, trigger thresholding,
clustering, Torch-enabled ranking and coincidence kernels; and detector projection,
series interpolation/decompression/time shifts, calibration application, and
many inference likelihood and marginalization reductions.  Availability still
depends on the operation, dtype, device, and any optional dependency involved.

Device residency does not mean that every public call is tensor-only.  The
intentional host boundaries include:

- scalar Python control decisions and scalar public return values;
- compact survivor/event outputs when a public API requires NumPy or Python
  data, rather than a full-array transfer before selection;
- small coefficient tables, spline knots, and other model metadata prepared on
  the host and copied to the device for bulk evaluation;
- explicit file, frame, text, and HDF I/O;
- explicit ``.numpy()`` and ``.lal()`` conversions, documented NumPy/Python
  results such as comparison masks and terminal diagnostic or sample arrays,
  and legacy CPU fallbacks for unsupported paths such as some relative-binning
  variants;
- third-party adapter invocations lacking a compatible device interface,
  including Foton and CPU/non-CUDA LISA response/TDI configurations;
- the SciPy FITPACK fit used to construct calibration splines (spline evaluation
  and calibration application use Torch); and
- the SciPy Brent sub-sample refinement in `optimized_match` (the FFT,
  correlation, and candidate evaluation remain on the Torch device, with
  scalar objective values crossing the boundary).

These paths preserve the existing CPU/public behavior and are not a promise of
end-to-end autograd.  In particular, this non-waveform coverage does not imply
that waveform generation is independent of `lalsimulation`.

## Numerical precision contract

On devices that support the required dtypes, a Torch implementation must not
use lower precision than the corresponding PyCBC CPU implementation. It may
widen intermediate calculations when that improves numerical stability while
preserving the public result dtype and API contract. A parity tolerance is an
acceptance bound for independent floating-point implementations, not
permission to introduce a lower-precision algorithm.

Device dtype limitations must be explicit and covered by tests. In particular,
MPS lacks the float64 and complex128 kernels required by some PyCBC operations;
those paths use a documented compatibility implementation or report the
operation as unsupported instead of silently downcasting a double-precision
input.

Notes:

- Standalone test scripts select their scheme with `-s/--scheme`.
- Pytest and tox use the test-only `PYCBC_TEST_SCHEME` variable. This is kept
  separate from the runtime `PYCBC_SCHEME` default so pytest collection does
  not create two competing scheme contexts.
- CPU-only tests that use `parse_args_cpu_only` skip cleanly under pytest (or
  exit successfully as standalone scripts) when a non-CPU scheme is selected.
- New targeted torch regressions live in:
  - `test/test_torch_pipeline.py` (PSD/whiten/resample/matched-filter and
    reduced-padding strain-buffer FFTs)
  - `test/test_torch_ops.py` (FFT/FIR/noise and array semantics)
  - `test/test_torch_generator.py` (waveform generator casting)
  - `test/test_torch_optional.py` (imports when Torch is not installed)
  - `test/test_qtransform.py` (q-transform parity)
  - `test/test_spatmplt_torch.py` (SPA template parity)

Run the CPU-Torch tox environment with:

```bash
tox -e py-torch
```

For a focused device-residency check, including tests that poison prohibited
host conversions, run:

```bash
python -m pytest -q \
  test/test_torch_inverse_spectrum_validation.py \
  test/test_torch_timeseries_taper.py \
  test/test_torch_pipeline.py::test_matched_filter_torch_vs_cpu \
  test/test_torch_pipeline.py::test_optimized_match_stays_on_device \
  test/test_torch_pipeline.py::test_calibration_spline_evaluation_stays_on_device \
  test/test_torch_pipeline.py::test_trigger_ranking_wrappers_stay_on_torch_device \
  test/test_torch_pipeline.py::test_time_coincidence_stays_on_torch_device \
  test/test_torch_pipeline.py::test_cluster_coincs_multiifo_stays_on_torch_device \
  test/test_qtransform.py \
  test/test_chisq_torch.py \
  test/test_live_batch_torch_peaks.py
```

To select a device or one test, use:

```bash
PYCBC_TEST_SCHEME=torch:cuda tox -e py-torch -- test/test_torch_pipeline.py
```

The basic CI matrix runs `py-torch` on CPU. The manually dispatched
`torch-gpu.yml` workflow accepts a `torch:cuda[:index]` scheme and an optional
pytest path or node ID.
