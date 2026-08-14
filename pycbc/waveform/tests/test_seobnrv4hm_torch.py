import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform
from pycbc.waveform.seobnrv4hm_torch import (
    _active_mode_indices,
    _qnm_omega,
    _seobnrv4_final_mass_spin,
)

_ROM_FILENAMES = ("SEOBNRv4HMROM_v1.0.hdf5", "SEOBNRv4HMROM.hdf5")
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent


def _require_rom_data():
    search_paths = [_WAVEFORM_DIR]
    search_paths.extend(
        Path(path)
        for path in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if path
    )
    if not any(
        (path / filename).is_file()
        for path in search_paths
        for filename in _ROM_FILENAMES
    ):
        pytest.skip("SEOBNRv4HM ROM data is not available on LAL_DATA_PATH")


def _run_case(params, use_native=True):
    _require_rom_data()
    env_backup = {
        key: os.environ.get(key)
        for key in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        # CPU reference
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "0"
        params_no_apx = dict(params)
        params_no_apx.pop("approximant", None)
        hp_cpu, hc_cpu = get_fd_waveform(
            approximant=params.get("approximant", "SEOBNRv4HM_ROM"),
            **params_no_apx,
        )

        # Torch path (native wrapper)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1" if use_native else "0"
        hp_torch, hc_torch = get_fd_waveform(
            approximant=params.get("approximant", "SEOBNRv4HM_ROM"),
            **params_no_apx,
        )
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return (hp_cpu, hc_cpu), (hp_torch, hc_torch)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=35.0,
            mass2=25.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=0.25,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.7,
            coa_phase=0.3,
        ),
        dict(
            mass1=60.0,
            mass2=20.0,
            spin1z=0.6,
            spin2z=0.1,
            delta_f=0.5,
            f_lower=15.0,
            f_final=0.0,
            f_ref=25.0,
            distance=600.0,
            inclination=0.4,
            coa_phase=1.0,
            mode_array=[(2, -2), (3, -3)],
        ),
        dict(
            mass1=18.0,
            mass2=42.0,
            spin1z=-0.3,
            spin2z=0.45,
            delta_f=0.5,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=450.0,
            inclination=1.1,
            coa_phase=-0.4,
            mode_array=[(2, -1), (5, -5)],
        ),
    ],
)
def test_seobnrv4hm_torch_parity(params):
    cpu_polarizations, torch_polarizations = _run_case(params, use_native=True)
    for cpu_series, torch_series in zip(cpu_polarizations, torch_polarizations):
        assert len(torch_series) == len(cpu_series)
        assert float(torch_series.epoch) == pytest.approx(float(cpu_series.epoch))
        cpu = cpu_series.numpy()
        tor = torch_series.numpy()
        scale = np.max(np.abs(cpu))
        assert scale > 0.0, "waveform contains no non-zero bins"
        np.testing.assert_allclose(
            tor,
            cpu,
            rtol=1e-10,
            atol=scale * 1e-12,
        )


def test_seobnrv4hm_torch_global_switch_fallback():
    params = dict(
        mass1=25.0,
        mass2=20.0,
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
    cpu_polarizations, torch_polarizations = _run_case(params, use_native=False)
    for cpu_series, torch_series in zip(cpu_polarizations, torch_polarizations):
        assert len(torch_series) == len(cpu_series)
        cpu = cpu_series.numpy()
        tor = torch_series.numpy()
        scale = np.max(np.abs(cpu))
        np.testing.assert_allclose(
            tor,
            cpu,
            rtol=1e-12,
            atol=scale * 1e-12,
        )


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ((30.0, 20.0, 0.3, 0.2), (47.35105872059112, 0.752235389057258)),
        ((35.0, 25.0, 0.2, -0.05), (57.05219701196065, 0.710429947075623)),
        ((10.0, 10.0, 0.0, 0.0), (19.035713300667098, 0.6864600000000001)),
        ((40.0, 10.0, -0.8, 0.6), (49.160049909691914, 0.05457878016900919)),
    ],
)
def test_seobnrv4_remnant_fit(parameters, expected):
    final_mass, final_spin = _seobnrv4_final_mass_spin(*parameters)
    assert final_mass == pytest.approx(expected[0], rel=2e-15)
    assert final_spin == pytest.approx(expected[1], rel=2e-15)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ((2, 2), 0.5624644677145155),
        ((3, 3), 0.890714186854536),
        ((2, 1), 0.47966903955842655),
        ((4, 4), 1.206314561245682),
        ((5, 5), 1.5155659533348103),
    ],
)
def test_qnm_frequency_matches_seobnrv4_table(mode, expected):
    omega = _qnm_omega(35.0, 25.0, 0.2, -0.1, *mode)
    scaled_omega = _qnm_omega(70.0, 50.0, 0.2, -0.1, *mode)
    assert omega == pytest.approx(expected, rel=2e-14)
    assert scaled_omega == pytest.approx(omega, rel=2e-14)


def test_mode_array_uses_directly_modeled_negative_m_modes():
    assert _active_mode_indices([(2, -2), (4, -4)]) == (0, 3)
    with pytest.raises(ValueError, match="positive-m"):
        _active_mode_indices([(2, 2)])
    with pytest.raises(ValueError, match="not available"):
        _active_mode_indices([(3, -2)])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_seobnrv4hm_dtype_cast(dtype):
    _require_rom_data()
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.3,
        spin2z=0.2,
        delta_f=0.25,
        f_lower=25.0,
        f_final=0.0,
        f_ref=25.0,
        distance=400.0,
        inclination=0.5,
        coa_phase=0.2,
        mode_array=[(2, -2)],
    )
    env_backup = {
        key: os.environ.get(key)
        for key in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1"
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        _scheme.mgr.state.dtype = dtype
        params_no_apx = dict(params)
        h_torch, _ = get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params_no_apx)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert h_torch.numpy().dtype == dtype


def test_seobnrv4hm_native_emits_no_user_warning():
    _require_rom_data()
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.3,
        spin2z=0.2,
        delta_f=0.25,
        f_lower=25.0,
        f_final=0.0,
        f_ref=25.0,
        distance=400.0,
        inclination=0.5,
        coa_phase=0.2,
    )
    env_backup = {
        key: os.environ.get(key)
        for key in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1"
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user == []
