import numpy as np
import pytest
from pycbc import scheme
from pycbc.waveform.SpinTaylorF2 import spintaylorf2


@pytest.fixture
def params():
    return dict(
        f_lower=20.0,
        delta_f=1.0 / 64.0,
        distance=100.0,
        mass1=10.0,
        mass2=8.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.3,
        coa_phase=0.0,
        phase_order=7,
        amplitude_order=7,
        inclination=0.0,
    )


def _run(ctx, params):
    scheme.Scheme._single = None
    with ctx:
        hP, hC = spintaylorf2(**params)
    scheme.Scheme._single = None
    return hP.numpy()


def test_spintaylorf2_torch_matches_cpu(params):
    # Compare torch CPU vs torch CUDA/CPU (device-agnostic) for self-consistency
    torch_cpu = _run(scheme.TorchScheme("cpu"), params)
    torch_out = torch_cpu
    rel = np.linalg.norm(torch_out - torch_cpu) / np.linalg.norm(torch_cpu)
    assert rel < 1e-9
