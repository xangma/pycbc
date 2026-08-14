import numpy as np
import pytest

pytest.importorskip("torch")

from pycbc.waveform.spintaylorf2_torch import spintaylorf2_torch
from pycbc.waveform import get_fd_waveform
from pycbc import scheme as _scheme


def _tol(dtype):
    if dtype == np.complex64:
        return dict(rel=1e-7, mag=1e-6, phase_mean=1e-3, phase_std=5e-2)
    return dict(rel=1e-11, mag=1e-10, phase_mean=1e-6, phase_std=1e-3)


def _run_case(params):
    old_scheme = _scheme.mgr.state
    try:
        _scheme.mgr.state = _scheme.CPUScheme()
        hP_c, _ = get_fd_waveform(approximant="SpinTaylorF2", **params)

        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        hP_t, _ = spintaylorf2_torch(**params)
        assert hP_t._data.tensor.device.type == "cpu"
    finally:
        _scheme.mgr.state = old_scheme
    cpu = hP_c.numpy()
    tor = hP_t.numpy()
    mask = np.abs(cpu) > 0
    assert mask.any(), "waveform contains no non-zero bins"
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    return rel, mag_ratio, phase_diff.mean(), phase_diff.std(), tor.dtype


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
            lnhatx=0.0,
            lnhaty=0.0,
            lnhatz=1.0,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1x=0.1,
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
            lnhatx=0.0,
            lnhaty=0.0,
            lnhatz=1.0,
        ),
    ],
)
def test_spintaylorf2_wrapper_fallback_parity(params, monkeypatch):
    # Use torch wrapper (CPU fallback by default for trusted parity)
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "0")
    rel, mag_ratio, phase_mean, phase_std, dtype = _run_case(params)
    tol = _tol(dtype)
    assert rel < tol["rel"]
    assert abs(mag_ratio - 1.0) < tol["mag"]
    assert abs(phase_mean) < tol["phase_mean"]
    assert phase_std < tol["phase_std"]
