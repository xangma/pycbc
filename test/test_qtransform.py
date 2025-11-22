import numpy as np
import pytest
import torch

from pycbc import scheme
from pycbc.types import TimeSeries


def _make_signal():
    # Two seconds of data at 1024 Hz to give the q-transform enough bandwidth
    t = np.arange(0, 2, 1 / 1024.0)
    sig = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 90 * t)
    return sig, t[1] - t[0]


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        # Explicitly drop the singleton guard so other tests can create schemes
        del ctx
        scheme.Scheme._single = None


def test_qtransform_torch_matches_cpu(torch_ctx):
    data, dt = _make_signal()

    # CPU reference
    ts_cpu = TimeSeries(data, delta_t=dt)
    t_cpu, f_cpu, plane_cpu = ts_cpu.qtransform(frange=(20, 120))

    # Torch path on CPU device
    with torch_ctx:
        ts_torch = TimeSeries(data, delta_t=dt)
        t_t, f_t, plane_t = ts_torch.qtransform(frange=(20, 120))

    assert isinstance(plane_t, torch.Tensor)
    assert plane_t.device.type == "cpu"
    diff = plane_t.cpu().numpy() - plane_cpu
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(plane_cpu)
    assert rel_l2 < 0.1
    assert np.allclose(t_t.cpu().numpy(), t_cpu)
    assert np.allclose(f_t.cpu().numpy(), f_cpu)


def test_qtransform_torch_interpolation_and_complex(torch_ctx):
    data, dt = _make_signal()
    with torch_ctx:
        ts = TimeSeries(data, delta_t=dt)
        t_t, f_t, plane_t = ts.qtransform(
            delta_t=dt / 2.0, delta_f=1.0, frange=(20, 120), return_complex=True
        )

    assert plane_t.is_complex()
    assert plane_t.shape[0] == len(f_t)
    assert plane_t.shape[1] == len(t_t)
    assert torch.isfinite(plane_t).all()
