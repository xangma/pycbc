import os
import numpy as np
import pytest

from pycbc.waveform.spintaylorf2_torch import spintaylorf2_torch
from pycbc.waveform import get_fd_waveform
from pycbc import scheme as _scheme


def _run_case(params):
    # ensure CPU reference
    old = _scheme.mgr.state
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.CPUScheme()
    _scheme.mgr.state.prefix = "cpu"
    try:
        hP_c, _ = get_fd_waveform(approximant="SpinTaylorF2", **params)
    finally:
        _scheme.mgr.state = old
    hP_t, _ = spintaylorf2_torch(**params)
    cpu = hP_c.numpy()
    tor = hP_t.numpy()
    mask = np.abs(cpu) > 0
    if not mask.any():
        return np.nan, np.nan, np.nan, np.nan
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    return rel, mag_ratio, phase_diff.mean(), phase_diff.std()


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            inclination=0.7,
            coa_phase=1.0,
            delta_f=0.1,
            f_lower=20.0,
            distance=500.0,
            amplitude_order=0,
            phase_order=-1,
            f_ref=30.0,
            side_bands=0,
            f_final=0.0,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1x=0.0,
            spin1y=0.0,
            spin1z=0.4,
            inclination=0.3,
            coa_phase=2.0,
            delta_f=0.2,
            f_lower=15.0,
            distance=400.0,
            amplitude_order=0,
            phase_order=7,
            f_ref=0.0,
            side_bands=0,
            f_final=0.0,
        ),
    ],
)
def test_spintaylorf2_torch_parity(params):
    # Require native torch path
    os.environ["PYCBC_SPINTAYLORF2_NATIVE"] = "1"
    rel, mag_ratio, phase_mean, phase_std = _run_case(params)
    if np.isnan(rel):
        pytest.skip("no non-zero bins for this configuration")
    assert rel < 1e-10
    assert abs(mag_ratio - 1.0) < 1e-10
    assert abs(phase_mean) < 1e-10
    assert phase_std < 1e-10
