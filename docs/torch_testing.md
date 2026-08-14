# Torch Testing Quickstart

PyCBC’s test helpers already understand the torch scheme. You can run any
scheme-aware test with:

```bash
python test/test_array.py -s torch           # CPU torch
python test/test_array.py -s torch:cuda      # specific CUDA device (0 default)
python test/test_array.py -s torch:mps       # Apple MPS (dtype limits apply)
```

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

To select a device or one test, use:

```bash
PYCBC_TEST_SCHEME=torch:cuda tox -e py-torch -- test/test_torch_pipeline.py
```

The basic CI matrix runs `py-torch` on CPU. The manually dispatched
`torch-gpu.yml` workflow accepts a `torch:cuda[:index]` scheme and an optional
pytest path or node ID.
