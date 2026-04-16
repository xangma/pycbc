import os

import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform


def test_cpu_waveform_generation_does_not_require_prefix():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        assert not hasattr(_scheme.mgr.state, "prefix")

        hp, _ = get_fd_waveform(
            approximant="TaylorF2",
            mass1=10.0,
            mass2=9.0,
            delta_f=0.5,
            f_lower=30.0,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert len(hp) > 0


def test_imrphenomhm_native_flag_uses_wrapper(monkeypatch):
    import pycbc.waveform.imrphenome_torch as wrapper_mod
    import pycbc.waveform.imrphenomhm_torch as skeleton_mod

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOME_NATIVE")}

    monkeypatch.setattr(
        wrapper_mod,
        "imrphenomhm_fd_torch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("wrapper hit")),
    )
    monkeypatch.setattr(
        skeleton_mod,
        "imrphenomhm_fd_torch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("skeleton hit")),
    )

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_IMRPHENOME_NATIVE"] = "1"

        with pytest.raises(RuntimeError, match="wrapper hit"):
            get_fd_waveform(
                approximant="IMRPhenomHM",
                mass1=40.0,
                mass2=30.0,
                spin1z=0.1,
                spin2z=0.05,
                delta_f=0.5,
                f_lower=20.0,
                f_final=0.0,
                f_ref=20.0,
                distance=400.0,
                inclination=0.4,
                coa_phase=0.1,
            )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single
