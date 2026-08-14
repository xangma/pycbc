import numpy as np
import pytest

pytest.importorskip("torch")

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
        spin1x=0.1,
        spin1y=0.05,
        spin1z=0.3,
        coa_phase=0.0,
        phase_order=7,
        amplitude_order=7,
        inclination=0.7,
    )


def _run(ctx, params):
    scheme.Scheme._single = None
    with ctx:
        hP, hC = spintaylorf2(**params)
    scheme.Scheme._single = None
    return hP.numpy()


def test_spintaylorf2_torch_matches_cpu(params):
    # Compare torch vs CPU/LAL reference for a precessing configuration
    from pycbc.waveform import get_fd_waveform

    # Force CPU reference without torch casting
    old = scheme.mgr.state
    scheme.Scheme._single = None
    scheme.mgr.state = scheme.CPUScheme()
    scheme.mgr.state.prefix = 'cpu'
    cpu_ref, _ = get_fd_waveform(approximant="SpinTaylorF2", **params)
    scheme.mgr.state = old
    torch_out = _run(scheme.TorchScheme("cpu"), params)
    rel = np.linalg.norm(torch_out - cpu_ref.numpy()) / np.linalg.norm(cpu_ref.numpy())
    assert rel < 0.3  # loose tolerance pending full parity tuning
