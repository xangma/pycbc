import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform


_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent


def _require_rom_data():
    search_paths = [_WAVEFORM_DIR]
    search_paths.extend(
        Path(path)
        for path in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if path
    )
    if not any((path / _ROM_FILENAME).is_file() for path in search_paths):
        pytest.skip(f"{_ROM_FILENAME} is not available on LAL_DATA_PATH")


def _run_case(params, use_native=True):
    _require_rom_data()
    env_backup = {
        key: os.environ.get(key)
        for key in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        # CPU reference
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4_NATIVE"] = "0"
        params_no_apx = dict(params)
        params_no_apx.pop("approximant", None)
        h_cpu, _ = get_fd_waveform(
            approximant=params.get("approximant", "SEOBNRv4_ROM"),
            **params_no_apx,
        )

        # Torch path (native ROM)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_SEOBNRV4_NATIVE"] = "1" if use_native else "0"
        params_no_apx = dict(params_no_apx)
        apx = params.get("approximant", "SEOBNRv4_ROM")
        h_torch, _ = get_fd_waveform(approximant=apx, **params_no_apx)
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
            mass1=30.0,
            mass2=25.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=1.0 / 64,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=400.0,
            inclination=0.7,
            coa_phase=0.5,
        ),
        dict(
            mass1=12.0,
            mass2=10.0,
            spin1z=0.5,
            spin2z=0.3,
            delta_f=0.25,
            f_lower=15.0,
            f_final=0.0,
            f_ref=30.0,
            distance=200.0,
            inclination=0.2,
            coa_phase=1.2,
        ),
    ],
)
def test_seobnrv4_torch_parity(params):
    cpu, tor = _run_case(params, use_native=True)
    n = min(len(cpu), len(tor))
    cpu = cpu[:n]
    tor = tor[:n]
    mask = (np.abs(cpu) > 1e-26) | (np.abs(tor) > 1e-26)
    assert mask.any(), "waveform contains no non-zero bins"
    np.testing.assert_allclose(tor[mask], cpu[mask], rtol=5e-5, atol=1e-10)


def test_seobnrv4_torch_global_switch_fallback():
    params = dict(
        mass1=20.0,
        mass2=15.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        f_final=0.0,
        f_ref=20.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.1,
    )
    cpu, tor = _run_case(params, use_native=False)
    n = min(len(cpu), len(tor))
    np.testing.assert_allclose(tor[:n], cpu[:n], rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=1.35,
            mass2=1.25,
            spin1z=0.0,
            spin2z=0.0,
            delta_f=0.1,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=100.0,
            inclination=0.3,
            coa_phase=0.0,
            lambda1=400,
            lambda2=600,
        ),
        dict(
            mass1=2.0,
            mass2=1.6,
            spin1z=0.05,
            spin2z=-0.02,
            delta_f=0.2,
            f_lower=30.0,
            f_final=0.0,
            f_ref=30.0,
            distance=150.0,
            inclination=0.5,
            coa_phase=0.2,
            lambda1=800,
            lambda2=500,
        ),
    ],
)
def test_seobnrv4_nrtidalv2_uses_lalsim_fallback(params, monkeypatch):
    import pycbc.waveform.seobnrv4_torch as native_module

    def unexpected_native_call(**_):
        raise AssertionError("NRTidal waveform was routed to the BBH-only port")

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", unexpected_native_call)
    cpu, tor = _run_case(
        {**params, "approximant": "SEOBNRv4_ROM_NRTidalv2"},
        use_native=True,
    )
    n = min(len(cpu), len(tor))
    cpu = cpu[:n]
    tor = tor[:n]
    mask = (np.abs(cpu) > 1e-26) | (np.abs(tor) > 1e-26)
    assert mask.any(), "waveform contains no non-zero bins"
    np.testing.assert_allclose(tor[mask], cpu[mask], rtol=1e-12, atol=1e-18)
