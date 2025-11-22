# Torch Testing Quickstart

PyCBC’s test helpers already understand the torch scheme. You can run any
scheme-aware test with:

```bash
python test/test_array.py -s torch           # CPU torch
python test/test_array.py -s torch:cuda      # specific CUDA device (0 default)
python test/test_array.py -s torch:mps       # Apple MPS (dtype limits apply)
```

Notes:
- The default scheme is controlled by `PYCBC_SCHEME`; setting
  `PYCBC_SCHEME=torch:cuda` will make `tools/pycbc_test_suite.sh` and
  `parse_args_all_schemes` pick torch even when `-s` isn’t passed.
- CPU-only tests that use `parse_args_cpu_only` will now exit early if run under
  `-s torch` (same behavior as CUDA).
- New targeted torch regressions live in:
  - `test/test_torch_pipeline.py` (PSD/whiten/resample/matched-filter)
  - `test/test_torch_ops.py` (FFT/FIR/noise)
  - `test/test_qtransform.py` (q-transform parity)

For CI, add a torch job that sets `PYCBC_SCHEME=torch` (or `torch:cuda` when
GPU is available) and reuses the existing test entrypoints.
