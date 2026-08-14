import pytest

pytest.importorskip("torch")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform
from pycbc.waveform.torch_switches import torch_native_enabled


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


@pytest.mark.parametrize("scheme_factory", [_scheme.TorchScheme, _scheme.CPUScheme])
def test_lightweight_scheme_finalizer_does_not_release_another_scheme(
    scheme_factory,
):
    old_single = _scheme.Scheme._single
    owner = object()
    try:
        _scheme.Scheme._single = owner
        lightweight_scheme = scheme_factory()
        lightweight_scheme.__del__()
        assert _scheme.Scheme._single is owner
    finally:
        _scheme.Scheme._single = old_single


def test_imrphenomhm_uses_lalsim_fallback(monkeypatch):
    import pycbc.waveform.waveform as waveform_mod

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lalsim hit")),
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()

        with pytest.raises(RuntimeError, match="lalsim hit"):
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
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_invalid_native_switch_value_is_rejected(monkeypatch):
    monkeypatch.setenv("PYCBC_EXAMPLE_NATIVE", "sometimes")
    with pytest.raises(ValueError, match="PYCBC_EXAMPLE_NATIVE"):
        torch_native_enabled("PYCBC_EXAMPLE_NATIVE")
