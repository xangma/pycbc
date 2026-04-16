import sys
import types

import numpy as np
import pytest
import torch

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter, resample
from pycbc.psd import welch
from pycbc.types import FrequencySeries, TimeSeries

# Skip the entire module if torch support is unavailable
pytest.importorskip("torch")
if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture(autouse=True)
def stub_execute_cached_fft_import(monkeypatch):
    """Avoid importing the full strain stack in Welch tests."""
    fake_pkg = types.ModuleType("pycbc.strain")
    fake_mod = types.ModuleType("pycbc.strain.strain")

    def _execute_cached_fft(*args, **kwargs):
        raise AssertionError("Welch cache path should not be used in this test")

    def _execute_cached_ifft(*args, **kwargs):
        raise AssertionError("PSD cache path should not be used in this test")

    fake_mod.execute_cached_fft = _execute_cached_fft
    fake_mod.execute_cached_ifft = _execute_cached_ifft
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_pkg)
    monkeypatch.setitem(sys.modules, "pycbc.strain.strain", fake_mod)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        # Allow other tests to construct schemes after we exit
        del ctx
        scheme.Scheme._single = None


def _relative_l2(a, b):
    diff = a - b
    return np.linalg.norm(diff) / np.linalg.norm(b)


def test_psd_welch_torch_matches_cpu(torch_ctx):
    # Short deterministic signal
    rng = np.random.default_rng(1234)
    data = rng.standard_normal(2048)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = welch(ts_cpu, 256)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = welch(ts_t, 256)

    assert isinstance(psd_t._data.tensor, torch.Tensor)
    assert psd_t._data.tensor.device.type == "cpu"
    assert psd_t.kind == psd_cpu.kind == "real"
    assert _relative_l2(psd_t.numpy(), psd_cpu.numpy()) < 0.05


def test_whiten_stays_on_device(torch_ctx):
    rng = np.random.default_rng(5678)
    data = rng.standard_normal(4096)
    ts_cpu = TimeSeries(data, delta_t=1 / 2048.0)
    white_cpu = ts_cpu.whiten(2, 1)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 2048.0)
        white_t = ts_t.whiten(2, 1)

    assert isinstance(white_t._data.tensor, torch.Tensor)
    assert white_t._data.tensor.device.type == "cpu"
    assert len(white_t) == len(white_cpu)
    assert _relative_l2(white_t.numpy(), white_cpu.numpy()) < 0.1


def test_resample_to_delta_t_torch(torch_ctx):
    data = np.sin(2 * np.pi * 30 * np.arange(0, 1, 1 / 1024.0))

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        rs_t = resample.resample_to_delta_t(ts_t, 1 / 256.0, method="ldas")

    assert isinstance(rs_t._data.tensor, torch.Tensor)
    assert rs_t._data.tensor.device.type == "cpu"
    assert abs(rs_t.delta_t - 1 / 256.0) < 1e-12


def test_matched_filter_torch_vs_cpu(torch_ctx):
    # Simple sine wave template/data with flat PSD
    t = np.arange(0, 1, 1 / 1024.0)
    data = np.sin(2 * np.pi * 50 * t)
    ts_cpu = TimeSeries(data, delta_t=1 / 1024.0)
    psd_cpu = FrequencySeries(np.ones(len(ts_cpu) // 2 + 1), delta_f=ts_cpu.delta_f)
    snr_cpu = matchedfilter.matched_filter(ts_cpu, ts_cpu, psd=psd_cpu)

    with torch_ctx:
        ts_t = TimeSeries(data, delta_t=1 / 1024.0)
        psd_t = FrequencySeries(np.ones(len(ts_t) // 2 + 1), delta_f=ts_t.delta_f)
        snr_t = matchedfilter.matched_filter(ts_t, ts_t, psd=psd_t)

    assert isinstance(snr_t._data.tensor, torch.Tensor)
    assert snr_t._data.tensor.device.type == "cpu"
    assert _relative_l2(snr_t.numpy(), snr_cpu.numpy()) < 0.05
