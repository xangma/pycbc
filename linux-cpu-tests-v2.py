"""Run CPU precision and affected-module tests with the target MKL backend."""
import sys
from pathlib import Path
import pytest
from pycbc.fft.backend_support import set_backend, get_backend

source = Path('/home/xangma/pycbc-torch-precision-validation-20260908/cpu-corrections')
sys.argv = ['pytest']
set_backend(['mkl'])
assert get_backend().__name__ == 'pycbc.fft.mkl'
print('FFT backend:', get_backend().__name__, flush=True)
code = pytest.main(['-q', '--tb=short', *[str(source/'test'/name) for name in (
    'test_sigmasq_series_precision.py', 'test_chisq_precision.py', 'test_strain_psd_precision.py',
    'test_psd.py', 'test_matchedfilter.py', 'test_strain.py')]])
assert 'torch' not in sys.modules
raise SystemExit(code)
