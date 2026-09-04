import builtins

import numpy as np
import pytest

from pycbc.psd import estimate as est
from pycbc.types import FrequencySeries, TimeSeries


@pytest.fixture
def block_strain_import(monkeypatch):
    orig_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pycbc.strain.strain":
            raise AssertionError("unexpected pycbc.strain.strain import")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_welch_skips_strain_import_when_fft_caching_disabled(block_strain_import, monkeypatch):
    monkeypatch.setattr(est, "USE_CACHING_FOR_WELCH_FFTS", False)
    ts = TimeSeries(np.random.randn(1024), delta_t=1 / 1024.0)
    psd = est.welch(ts, seg_len=256, seg_stride=128, avg_method="mean")
    assert len(psd) == 129


def test_inverse_spectrum_truncation_skips_strain_import_when_caching_disabled(
    block_strain_import, monkeypatch
):
    monkeypatch.setattr(est, "USE_CACHING_FOR_INV_SPEC_TRUNC", False)
    psd = FrequencySeries(np.linspace(1.0, 2.0, 257), delta_f=0.5)
    out = est.inverse_spectrum_truncation(psd, max_filter_len=32, low_frequency_cutoff=1.0)
    assert len(out) == len(psd)
