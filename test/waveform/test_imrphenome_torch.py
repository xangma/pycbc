import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402


def _run_case(approximant, params):
    env_backup = {
        key: os.environ.get(key)
        for key in (
            "PYCBC_TORCH_NATIVE_PORTS",
            "PYCBC_IMRPHENOMHM_NATIVE",
        )
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMHM_NATIVE"] = "0"
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        h_cpu, _ = get_fd_waveform(approximant=approximant, **params)

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        h_torch, _ = get_fd_waveform(approximant=approximant, **params)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return h_cpu, h_torch


@pytest.mark.parametrize(
    "approximant,params",
    [
        (
            "IMRPhenomHM",
            dict(
                mass1=50.0,
                mass2=35.0,
                spin1z=0.2,
                spin2z=0.1,
                delta_f=0.5,
                f_lower=15.0,
                f_final=0.0,
                f_ref=20.0,
                distance=500.0,
                inclination=0.7,
                coa_phase=1.0,
            ),
        ),
    ],
)
def test_imrphenomhm_lalsim_fallback_is_cast_to_torch(approximant, params):
    h_cpu, h_torch = _run_case(approximant, params)
    assert isinstance(h_torch._data.tensor, torch.Tensor)
    cpu = h_cpu.numpy()
    tor = h_torch.numpy()
    n = min(len(cpu), len(tor))
    cpu = cpu[:n]
    tor = tor[:n]
    mask = (np.abs(cpu) > 1e-26) | (np.abs(tor) > 1e-26)
    assert mask.any(), "waveform contains no non-zero bins"
    np.testing.assert_allclose(tor[mask], cpu[mask], rtol=1e-5, atol=1e-9)
