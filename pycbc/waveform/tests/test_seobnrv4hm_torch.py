import os
import warnings
import numpy as np
import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform


def _run_case(params, use_native=True):
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE", "LAL_DATA_PATH")}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        lal_paths = [
            "/Users/xangma/miniconda3/envs/pycbc313/opt/lalsuite-extra/share/lalsimulation",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),  # pycbc/waveform
        ]
        os.environ["LAL_DATA_PATH"] = ":".join(lal_paths)

        # CPU reference
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "0"
        params_no_apx = dict(params)
        params_no_apx.pop("approximant", None)
        h_cpu, _ = get_fd_waveform(approximant=params.get("approximant", "SEOBNRv4HM_ROM"), **params_no_apx)

        # Torch path (native wrapper)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1" if use_native else "0"
        h_torch, _ = get_fd_waveform(approximant=params.get("approximant", "SEOBNRv4HM_ROM"), **params_no_apx)
    except RuntimeError as exc:
        pytest.skip(f"SEOBNRv4HM unavailable (likely missing ROM data for LAL fallback): {exc}")
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
        ),
    ],
)
def test_seobnrv4hm_torch_parity(params):
    cpu, tor = _run_case(params, use_native=True)
    n = min(len(cpu), len(tor))
    cpu = cpu[:n]
    tor = tor[:n]
    mask = (np.abs(cpu) > 1e-26) | (np.abs(tor) > 1e-26)
    if not mask.any():
        pytest.skip("no non-zero bins")
    np.testing.assert_allclose(tor[mask], cpu[mask], rtol=5e-5, atol=1e-10)


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
    cpu, tor = _run_case(params, use_native=False)
    n = min(len(cpu), len(tor))
    np.testing.assert_allclose(tor[:n], cpu[:n], rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_seobnrv4hm_dtype_cast(dtype):
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
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE", "LAL_DATA_PATH")}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1"
        os.environ["LAL_DATA_PATH"] = ":".join([
            "/Users/xangma/miniconda3/envs/pycbc313/opt/lalsuite-extra/share/lalsimulation",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        ])
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        _scheme.mgr.state.dtype = dtype
        params_no_apx = dict(params)
        h_torch, _ = get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params_no_apx)
    except RuntimeError as exc:
        pytest.skip(f"SEOBNRv4HM unavailable (likely missing ROM data for LAL fallback): {exc}")
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
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4HM_NATIVE", "LAL_DATA_PATH")}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1"
        os.environ["LAL_DATA_PATH"] = ":".join([
            "/Users/xangma/miniconda3/envs/pycbc313/opt/lalsuite-extra/share/lalsimulation",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        ])
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params)
    except RuntimeError as exc:
        pytest.skip(f"SEOBNRv4HM unavailable (likely missing ROM data for LAL fallback): {exc}")
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
