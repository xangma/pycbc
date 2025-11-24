import os
import numpy as np
import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform


def _run_case(approximant, params, use_native=True):
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOME_NATIVE")}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOME_NATIVE"] = "0"
        h_cpu, _ = get_fd_waveform(approximant=approximant, **params)

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_IMRPHENOME_NATIVE"] = "1" if use_native else "0"
        h_torch, _ = get_fd_waveform(approximant=approximant, **params)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return h_cpu.numpy(), h_torch.numpy()


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
def test_imrphenome_torch_parity(approximant, params):
    cpu, tor = _run_case(approximant, params, use_native=True)
    n = min(len(cpu), len(tor))
    cpu = cpu[:n]
    tor = tor[:n]
    mask = (np.abs(cpu) > 1e-26) | (np.abs(tor) > 1e-26)
    if not mask.any():
        pytest.skip("no non-zero bins")
    np.testing.assert_allclose(tor[mask], cpu[mask], rtol=1e-5, atol=1e-9)


def test_imrphenome_torch_global_fallback():
    params = dict(
        mass1=20.0,
        mass2=15.0,
        spin1z=0.0,
        spin2z=0.0,
        delta_f=0.5,
        f_lower=20.0,
        f_final=0.0,
        f_ref=20.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.2,
    )
    cpu, tor = _run_case("IMRPhenomHM", params, use_native=False)
    n = min(len(cpu), len(tor))
    np.testing.assert_allclose(tor[:n], cpu[:n], rtol=1e-12, atol=1e-18)
