"""Ensure CPU discovery does not initialize optional JAX modules."""

import os
import subprocess
import sys
import importlib.util
import pytest


def _run(code):
    env = os.environ.copy()
    env.pop("PYCBC_SCHEME", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=45,
    ).stdout.splitlines()


def test_cpu_fft_and_psd_imports_are_jax_lazy():
    lines = _run(
        """
import sys
import pycbc.fft
assert 'jax' not in sys.modules
print('jax' in pycbc.fft.get_backend_names())
import pycbc.psd
print('jax' in sys.modules)
print('pycbc.psd.analytical_jax' in sys.modules)
print('pycbc.psd.estimate_jax' in sys.modules)
"""
    )
    assert lines[0] == str(importlib.util.find_spec("jax") is not None)
    assert lines[1:] == ["False", "False", "False"]


def test_public_jax_psd_api_remains_lazy_importable():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is unavailable")
    lines = _run(
        """
import sys
from pycbc.psd import welch_jax
assert callable(welch_jax)
print('pycbc.psd.estimate_jax' in sys.modules)
"""
    )
    assert lines == ["True"]


def test_cpu_psd_star_import_preserves_original_exports_without_jax():
    lines = _run(
        """
import sys
from pycbc.psd import *
assert callable(welch)
assert callable(MultiDetOptionAppendAction)
assert callable(copy.deepcopy)
assert 'welch_jax' not in locals()
assert 'analytical_psd_jax' not in locals()
assert 'get_jax_psd_list' not in locals()
assert 'jax' not in sys.modules
print('star-ok')
"""
    )
    assert lines == ["star-ok"]


def test_cpu_model_discovery_preserves_order_duplicates_and_lazy_imports():
    lines = _run(
        """
import sys
import pycbc.psd.analytical as analytical
analytical.get_lalsim_psd_list = lambda: ['z', 'a']
analytical.get_pycbc_psd_list = lambda: ['z', 'b']
assert analytical.get_psd_model_list() == ['z', 'a', 'z', 'b']
assert 'jax' not in sys.modules
assert 'pycbc.psd.analytical_jax' not in sys.modules
print('models-ok')
"""
    )
    assert lines == ["models-ok"]


def test_missing_optional_jax_symbol_is_not_available():
    lines = _run(
        """
import pycbc.psd
pycbc.psd._HAVE_JAX = False
try:
    from pycbc.psd import welch_jax
except ImportError:
    print('missing-ok')
else:
    raise AssertionError('optional JAX API unexpectedly available')
"""
    )
    assert lines == ["missing-ok"]
