import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import zeros
from pycbc.waveform.spa_tmplt import spa_tmplt_engine


@pytest.fixture
def params():
    return dict(
        kmin=1,
        phase_order=7,
        delta_f=1.0 / 64.0,
        piM=np.pi * 20.0,
        pfaN=1.0,
        pfa2=0.1,
        pfa3=-0.05,
        pfa4=0.01,
        pfa5=0.02,
        pfl5=-0.03,
        pfa6=0.04,
        pfl6=0.01,
        pfa7=-0.02,
        amp_factor=1.0,
    )


def _run_engine(scheme_ctx, dtype, params):
    scheme.Scheme._single = None
    with scheme_ctx:
        htilde = zeros(256, dtype=dtype)
        spa_tmplt_engine(htilde, **params)
    # reset singleton to allow another scheme
    scheme.Scheme._single = None
    return htilde.numpy()


def test_spatmplt_torch_matches_cpu(params):
    scheme.Scheme._single = None
    cpu = _run_engine(scheme.CPUScheme(), np.complex64, params)
    torch_out = _run_engine(scheme.TorchScheme("cpu"), np.complex64, params)

    rel_l2 = np.linalg.norm(torch_out - cpu) / np.linalg.norm(cpu)
    assert rel_l2 < 1e-5


def test_spatmplt_torch_dtype_and_device(params):
    scheme.Scheme._single = None
    ctx = scheme.TorchScheme("cpu")
    with ctx:
        htilde = zeros(64, dtype=np.complex64)
        spa_tmplt_engine(htilde, **params)
        # Should remain complex64 on CPU device
        import torch
        assert isinstance(htilde._data.tensor, torch.Tensor)
        assert htilde._data.tensor.dtype == torch.complex64
        assert htilde._data.tensor.device.type == "cpu"
    scheme.Scheme._single = None
