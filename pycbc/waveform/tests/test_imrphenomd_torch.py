import os
import numpy as np
import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform


def _tol(dtype):
    if dtype == np.complex64:
        return dict(rel=5e-5, mag=2e-4, phase_mean=5e-3, phase_std=5e-2)
    return dict(rel=1e-5, mag=1e-5, phase_mean=5e-5, phase_std=5e-4)


def _run_case(params, use_native=True):
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMD_NATIVE")}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        # CPU reference (LAL)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMD_NATIVE"] = "0"
        h_cpu, _ = get_fd_waveform(approximant="IMRPhenomD", **params)

        # Torch path
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_IMRPHENOMD_NATIVE"] = "1" if use_native else "0"
        h_torch, _ = get_fd_waveform(approximant="IMRPhenomD", **params)
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
    "params",
    [
        dict(
            mass1=35.0,
            mass2=28.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=1.0 / 64,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.4,
            coa_phase=1.1,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1z=0.6,
            spin2z=0.3,
            delta_f=0.5,
            f_lower=15.0,
            f_final=0.0,
            f_ref=30.0,
            distance=300.0,
            inclination=1.2,
            coa_phase=0.3,
        ),
    ],
)
def test_imrphenomd_torch_parity(params):
    cpu, tor = _run_case(params, use_native=True)
    mask = np.abs(cpu) > 1e-26
    if not mask.any():
        pytest.skip("no non-zero bins")
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    tol = _tol(tor.dtype)
    assert rel < tol["rel"]
    assert abs(mag_ratio - 1.0) < tol["mag"]
    assert abs(phase_diff.mean()) < tol["phase_mean"]
    assert phase_diff.std() < tol["phase_std"]


def test_imrphenomd_torch_global_switch_fallback():
    params = dict(
        mass1=20.0,
        mass2=18.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=1.0 / 32,
        f_lower=20.0,
        f_final=0.0,
        f_ref=20.0,
        distance=400.0,
        inclination=0.3,
        coa_phase=0.2,
    )
    cpu, tor = _run_case(params, use_native=False)
    np.testing.assert_allclose(tor, cpu, rtol=1e-12, atol=1e-18)
