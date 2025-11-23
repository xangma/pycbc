import os
import numpy as np
import pytest

from pycbc.waveform.spa_tmplt import spa_tmplt
from pycbc import scheme as _scheme


def _tol(dtype):
    """Return tolerance tuple keyed by output dtype."""
    if dtype == np.complex64:
        return dict(rel=1e-7, mag=1e-6, phase_mean=1e-3, phase_std=5e-2)
    return dict(rel=1e-11, mag=1e-10, phase_mean=1e-6, phase_std=1e-3)


def _run_case(params):
    old = _scheme.mgr.state
    # CPU reference (uses lalsimulation phasing)
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.CPUScheme()
    _scheme.mgr.state.prefix = "cpu"
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
    try:
        h_cpu = spa_tmplt(**params)
    finally:
        _scheme.mgr.state = old

    # Torch path (native phasing)
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme()
    _scheme.mgr.state.prefix = "torch"
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
    try:
        h_torch = spa_tmplt(**params)
    finally:
        _scheme.mgr.state = old

    cpu = h_cpu.numpy()
    tor = h_torch.numpy()
    # Ignore only exact/near-zero bins; keep tiny but non-zero tails.
    mask = np.abs(cpu) > 1e-26
    if not mask.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, tor.dtype
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    return (
        rel,
        mag_ratio,
        phase_diff.mean(),
        phase_diff.std(),
        np.nonzero(mask)[0][0],
        np.nonzero(mask)[0][-1],
        tor.dtype,
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.3,
            spin2z=-0.1,
            delta_f=0.2,
            f_lower=20.0,
            distance=500.0,
            phase_order=-1,
            spin_order=-1,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1z=0.0,
            spin2z=0.4,
            delta_f=0.25,
            f_lower=15.0,
            distance=400.0,
            phase_order=7,
            spin_order=5,
        ),
        dict(
            mass1=1.4,
            mass2=1.3,
            spin1z=0.0,
            spin2z=0.0,
            delta_f=0.1,
            f_lower=10.0,
            distance=100.0,
            phase_order=-1,
            spin_order=-1,
            lambda1=800.0,
            lambda2=700.0,
        ),
    ],
)
def test_taylorf2_torch_parity(params):
    rel, mag_ratio, phase_mean, phase_std, kmin, kmax, dtype = _run_case(params)
    if np.isnan(rel):
        pytest.skip("no non-zero bins for this configuration")
    tol = _tol(dtype)
    assert rel < tol["rel"]
    assert abs(mag_ratio - 1.0) < tol["mag"]
    assert abs(phase_mean) < tol["phase_mean"]
    assert phase_std < tol["phase_std"]
    # basic sanity on bin coverage
    assert kmax > kmin
