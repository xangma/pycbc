import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.waveform.seobnrv4phm_constants import compute_spin_aligned_hcoeffs

PARAMS = dict(
    mass1=25.0,
    mass2=18.0,
    spin1x=0.2,
    spin1y=-0.15,
    spin1z=0.05,
    spin2x=-0.1,
    spin2y=0.07,
    spin2z=0.2,
    delta_f=0.25,
    f_lower=30.0,
    f_final=128.0,
    f_ref=50.0,
    distance=300.0,
    inclination=0.6,
    coa_phase=0.3,
)


def _phm_boundary_params():
    return dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        delta_f=1.0,
        f_lower=90.0,
        f_final=256.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.0,
        mode_array=((2, 2), (2, 1)),
    )


RUN_SLOW_NATIVE_PHM = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_NATIVE_SLOW", "0"
) not in ("0", "", "false", "False")
SLOW_NATIVE_PHM_REASON = (
    "full SEOBNRv4PHM native waveform generation is an opt-in slow test; set "
    "PYCBC_RUN_SEOBNRV4PHM_NATIVE_SLOW=1 to run it"
)
RUN_CHECKPOINT_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_CHECKPOINT_DIAG", "0"
) not in ("0", "", "false", "False")
CHECKPOINT_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM checkpoint diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_CHECKPOINT_DIAG=1 to run it"
)
RUN_BOUNDARY_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_BOUNDARY_DIAG", "0"
) not in ("0", "", "false", "False")
BOUNDARY_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM boundary diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_BOUNDARY_DIAG=1 to run it"
)
RUN_IC_STEP4_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_IC_STEP4_DIAG", "0"
) not in ("0", "", "false", "False")
IC_STEP4_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM IC Step 4 diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_IC_STEP4_DIAG=1 to run it"
)

_PHM_FD_PARITY_TOLERANCES = {
    # Independent Torch and C/GSL adaptive trajectories eventually choose
    # slightly different accepted steps from roundoff-level RHS differences.
    # These bounds retain several-fold platform margin around the validated
    # result while remaining tight enough to reject the former 11% error.
    "raw_rel": 1.0e-3,
    "amp_min": 1.0 - 1.0e-3,
    "amp_max": 1.0 + 1.0e-3,
    "raw_phase_mean": 1.0e-3,
    "raw_phase_std": 1.0e-3,
    "linear_dt_s": 5.0e-6,
    "aligned_rel": 2.5e-4,
    "aligned_phase_std": 2.5e-4,
}

_PHM_CHECKPOINT_PARITY_TOLERANCES = {
    "dt_M": 5.0e-10,
    "omega_peak_rel": 5.0e-12,
    "r": 5.0e-10,
}


def _lal_weighted_spins_to_chi(S1_lal, S2_lal, mass1, mass2):
    """Convert LAL dynamics spin columns into individual dimensionless spins."""
    total_mass = mass1 + mass2
    return S1_lal * (total_mass / mass1) ** 2, S2_lal * (total_mass / mass2) ** 2


def test_seobnrv4phm_exposes_dispatch_entrypoint():
    from pycbc.waveform import seobnrv4phm_torch

    assert seobnrv4phm_torch.torch_native_td_waveform is (
        seobnrv4phm_torch.seobnrv4phm_td_torch
    )
    assert seobnrv4phm_torch.torch_native_fd_waveform is (
        seobnrv4phm_torch.seobnrv4phm_fd_torch
    )


def test_seobnrv4phm_enabled_entrypoint_uses_native_implementation(monkeypatch):
    """The public native switch must not fall back through LAL."""
    from pycbc.waveform import seobnrv4phm_torch

    expected_td = object()
    expected_fd = object()
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        lambda **_params: expected_td,
    )
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_fd_native",
        lambda **_params: expected_fd,
    )

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        params = dict(PARAMS, approximant="SEOBNRv4PHM")
        assert seobnrv4phm_torch.seobnrv4phm_td_torch(**params) is expected_td
        assert seobnrv4phm_torch.seobnrv4phm_fd_torch(**params) is expected_fd
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_seobnrv4phm_fd_wrapper_preserves_td_start_time(monkeypatch):
    """The native TD-to-FD wrapper must preserve the PyCBC epoch property."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        hp_td = TimeSeries(
            TorchArrayData(torch.arange(4, dtype=torch.float64)),
            delta_t=0.25,
            epoch=-0.5,
            copy=False,
        )
        hc_td = TimeSeries(
            TorchArrayData(torch.arange(4, dtype=torch.float64)),
            delta_t=0.25,
            epoch=-0.5,
            copy=False,
        )
        monkeypatch.setattr(
            seobnrv4phm_torch,
            "_seobnrv4phm_td_native",
            lambda **_params: (hp_td, hc_td),
        )

        hp_fd, hc_fd = seobnrv4phm_torch._seobnrv4phm_fd_native(
            delta_f=1.0,
            f_final=2.0,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert hp_fd.start_time == hp_td.start_time
    assert hc_fd.start_time == hc_td.start_time
    assert hp_fd.delta_f == hc_fd.delta_f == 1.0
    assert len(hp_fd) == len(hc_fd) == 3


def test_seobnrv4phm_public_td_dispatch_avoids_lalsimulation(monkeypatch):
    """The public TD API must select the native Torch entry point."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    native_calls = 0
    expected = []

    def recording_native(**_params):
        nonlocal native_calls
        native_calls += 1
        result = tuple(
            TimeSeries(
                TorchArrayData(torch.arange(4, dtype=torch.float64)),
                delta_t=1.0 / 2048.0,
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4PHM called lalsimulation")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4phm_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_td_waveform(
            approximant="SEOBNRv4PHM",
            delta_t=1.0 / 2048.0,
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert native_calls == 1
    assert all(got is want for got, want in zip(actual, expected))


def test_seobnrv4phm_native_support_boundary():
    from pycbc.waveform.seobnrv4phm_torch import (
        seobnrv4phm_native_supported,
    )

    baseline = dict(PARAMS, approximant="SEOBNRv4PHM")
    assert seobnrv4phm_native_supported(baseline)

    unsupported = (
        {"approximant": "SEOBNRv4P"},
        {"mass1": 10.0, "mass2": 20.0},
        {"phase_order": 7},
        {"eccentricity": 0.1},
        {"lambda1": 100.0},
        {"dchi0": 0.1},
        {"frame_axis": 1},
        {"numrel_data": "file.h5"},
        {"mode_array": ((3, 2),)},
        {"long_asc_nodes": float("nan")},
    )
    for overrides in unsupported:
        assert not seobnrv4phm_native_supported(
            dict(baseline, **overrides)
        )


def test_seobnrv4phm_mps_selects_supported_dtypes():
    import torch

    from pycbc.waveform import seobnrv4phm_torch

    old_state = _scheme.mgr.state
    try:
        _scheme.mgr.state = type(
            "MPSState",
            (),
            {"device": torch.device("mps"), "dtype": torch.float64},
        )()
        device, real_dtype, complex_dtype = (
            seobnrv4phm_torch._target_device_dtypes()
        )
    finally:
        _scheme.mgr.state = old_state

    assert device.type == "mps"
    assert real_dtype == torch.float32
    assert complex_dtype == torch.complex64


def test_seobnrv4phm_polarization_rotation_matches_lal_convention():
    import torch

    from pycbc.waveform.seobnrv4phm_torch import _rotate_polarizations

    hp = torch.tensor([1.0, -0.5], dtype=torch.float64)
    hc = torch.tensor([0.25, 2.0], dtype=torch.float64)
    nodes = 0.37
    got_hp, got_hc = _rotate_polarizations(hp, hc, nodes)
    expected = (hp - 1j * hc) * np.exp(2j * nodes)

    torch.testing.assert_close(got_hp, expected.real)
    torch.testing.assert_close(got_hc, -expected.imag)


def test_seobnrv4phm_mode_array_normalizes_like_lal():
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    assert dyn.normalize_mode_array(None) == tuple(dyn.LM_DEFAULT)
    assert dyn.normalize_mode_array(((5, 5), (2, 1), (2, 2), (2, 1))) == (
        (2, 2),
        (2, 1),
        (5, 5),
    )


def test_seobnrv4phm_mode_array_rejects_lal_invalid_modes():
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    with pytest.raises(ValueError, match="negative m"):
        dyn.normalize_mode_array(((2, -2),))
    with pytest.raises(ValueError, match="not available"):
        dyn.normalize_mode_array(((3, 2),))


def test_seobnrv4phm_time_to_fd_matches_pycbc_timeseries_fft():
    """The native TD->FD helper should mirror PyCBC TimeSeries padding/FFT."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.waveform.seobnrv4phm_torch import _time_to_fd

    delta_t = 1.0 / 128.0
    delta_f = 0.5
    t = np.arange(127, dtype=np.float64) * delta_t
    hp = np.sin(2.0 * np.pi * 7.0 * t) + 0.25 * np.cos(2.0 * np.pi * 13.0 * t)
    hc = 0.5 * np.sin(2.0 * np.pi * 5.0 * t + 0.3)

    hp_torch, hc_torch, df_out = _time_to_fd(
        torch.as_tensor(hp, dtype=torch.float64),
        torch.as_tensor(hc, dtype=torch.float64),
        delta_t,
        torch.complex128,
        delta_f,
    )

    hp_ref = TimeSeries(hp, delta_t=delta_t).to_frequencyseries(delta_f=delta_f)
    hc_ref = TimeSeries(hc, delta_t=delta_t).to_frequencyseries(delta_f=delta_f)

    assert df_out == pytest.approx(delta_f)
    np.testing.assert_allclose(hp_torch.numpy(), hp_ref.numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_torch.numpy(), hc_ref.numpy(), rtol=1e-12, atol=1e-14)


def _cpu_waveform():
    env_backup = {
        k: os.environ.get(k)
        for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "0"
        return get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)[0]
    except (RuntimeError, ValueError, OSError) as exc:
        pytest.skip(f"SEOBNRv4PHM CPU reference unavailable: {exc}")
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _phm_fd_parity_metrics(h_cpu, h_torch):
    """Return in-band FD parity diagnostics for the native path."""
    n = min(len(h_cpu), len(h_torch))
    cpu = h_cpu.numpy()[:n]
    torch_native = h_torch.numpy()[:n]
    delta_f = float(h_torch.delta_f)
    freqs = np.arange(n, dtype=float) * delta_f

    f_final = PARAMS.get("f_final", 0.0)
    if f_final is None or f_final <= 0.0:
        f_final = freqs[-1] if len(freqs) else 0.0

    band = (freqs >= PARAMS["f_lower"]) & (freqs <= f_final)
    finite = (
        band
        & np.isfinite(cpu)
        & np.isfinite(torch_native)
        & (np.abs(cpu) > 0.0)
        & (np.abs(torch_native) > 0.0)
    )
    if not np.any(finite):
        return {
            "bins": 0,
            "f_min": float(PARAMS["f_lower"]),
            "f_max": float(f_final),
        }

    cpu_band = cpu[finite]
    torch_band = torch_native[finite]
    ratio = np.abs(torch_band / cpu_band)
    phase_diff = np.angle(torch_band * np.conj(cpu_band))
    rel = np.linalg.norm(torch_band - cpu_band) / np.linalg.norm(cpu_band)
    freq_band = freqs[finite]
    if len(freq_band) >= 2:
        phase_unwrapped = np.unwrap(phase_diff)
        fit_slope, fit_intercept = np.polyfit(freq_band, phase_unwrapped, 1)
        linear_phase = fit_slope * freq_band + fit_intercept
        aligned = torch_band * np.exp(-1j * linear_phase)
        aligned_phase = np.angle(aligned * np.conj(cpu_band))
        aligned_rel = np.linalg.norm(aligned - cpu_band) / np.linalg.norm(cpu_band)
        linear_dt_s = fit_slope / (2.0 * np.pi)
        aligned_phase_mean = aligned_phase.mean()
        aligned_phase_std = aligned_phase.std()
    else:
        fit_slope = np.nan
        fit_intercept = np.nan
        linear_dt_s = np.nan
        aligned_rel = np.inf
        aligned_phase_mean = np.nan
        aligned_phase_std = np.inf

    return {
        "bins": int(finite.sum()),
        "f_min": float(freq_band[0]),
        "f_max": float(freq_band[-1]),
        "rel": float(rel),
        "amp_min": float(ratio.min()),
        "amp_median": float(np.median(ratio)),
        "amp_max": float(ratio.max()),
        "phase_mean": float(phase_diff.mean()),
        "phase_std": float(phase_diff.std()),
        "linear_phase_slope": float(fit_slope),
        "linear_phase_intercept": float(fit_intercept),
        "linear_dt_s": float(linear_dt_s),
        "aligned_rel": float(aligned_rel),
        "aligned_phase_mean": float(aligned_phase_mean),
        "aligned_phase_std": float(aligned_phase_std),
    }


def _format_phm_fd_parity_metrics(metrics):
    if metrics["bins"] == 0:
        return (
            "SEOBNRv4PHM torch-native parity has no comparable in-band bins "
            f"for f=[{metrics['f_min']:.2f}, {metrics['f_max']:.2f}] Hz"
        )
    return (
        "SEOBNRv4PHM torch-native parity not within tolerance: "
        f"bins={metrics['bins']} "
        f"f=[{metrics['f_min']:.2f}, {metrics['f_max']:.2f}] Hz "
        f"rel={metrics['rel']:.3e} "
        f"amp_ratio=[{metrics['amp_min']:.3e}, "
        f"{metrics['amp_median']:.3e}, {metrics['amp_max']:.3e}] "
        f"phase_mean={metrics['phase_mean']:.3e} "
        f"phase_std={metrics['phase_std']:.3e} "
        f"linear_dt={metrics['linear_dt_s']:.3e}s "
        f"linear_intercept={metrics['linear_phase_intercept']:.3e} "
        f"aligned_rel={metrics['aligned_rel']:.3e} "
        f"aligned_phase_std={metrics['aligned_phase_std']:.3e}"
    )


def _isolated_lal_phm_dynamics(params, n_rows=0, vector="joined", n_cols=15):
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    payload = dict(params)
    payload["mode_array"] = list(dyn.normalize_mode_array(params.get("mode_array", None)))
    payload["n_rows"] = int(n_rows)
    payload["n_cols"] = int(n_cols)
    payload["vector"] = str(vector)
    script = r"""
import json
import sys

import lal
import lalsimulation as lalsim
import numpy as np

p = json.loads(sys.argv[1])
mode_array = lalsim.SimInspiralCreateModeArray()
for ell, emm in p["mode_array"]:
    lalsim.SimInspiralModeArrayActivateMode(mode_array, int(ell), int(emm))

outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
    0.0,
    p.get("delta_t", 1.0 / 4096.0),
    p["mass1"] * lal.MSUN_SI,
    p["mass2"] * lal.MSUN_SI,
    p["f_lower"],
    p["distance"] * lal.PC_SI,
    p.get("inclination", 0.0),
    p.get("spin1x", 0.0),
    p.get("spin1y", 0.0),
    p.get("spin1z", 0.0),
    p.get("spin2x", 0.0),
    p.get("spin2y", 0.0),
    p.get("spin2z", 0.0),
    mode_array,
    lal.CreateDict(),
)
dyn_vars = 26
vector_indices = {"adas": 4, "his": 5, "joined": 6}
selected = p.get("vector", "joined")
if selected not in vector_indices:
    raise ValueError(f"unknown dynamics vector {selected!r}")

def vector_data(name):
    vec = outs[vector_indices[name]]
    data = np.array(vec.data, copy=False)
    ret_len = vec.length // dyn_vars
    return data, ret_len

def vector_summary(name):
    data, ret_len = vector_data(name)

    def col(k):
        return data[k * ret_len : (k + 1) * ret_len]

    t_m = col(0)
    r = col(18)
    omega = col(22)
    hamiltonian = col(25)
    idx_peak = int(np.nanargmax(omega))

    def sample_rows(n_rows=5):
        n = min(int(n_rows), ret_len)
        rows = []
        for i in range(n):
            rows.append({
                "t_M": float(t_m[i]),
                "r": float(r[i]),
                "omega": float(omega[i]),
                "H": float(hamiltonian[i]),
                "state": [float(data[k * ret_len + i]) for k in range(1, 15)],
            })
        return rows

    out = {
        "n_dyn": int(ret_len),
        "t_start_M": float(t_m[0]),
        "t_end_M": float(t_m[-1]),
        "r_start": float(r[0]),
        "r_end": float(r[-1]),
        "t_peak_omega_M": float(t_m[idx_peak]),
        "omega_peak_dimless": float(omega[idx_peak]),
        "H_start": float(hamiltonian[0]),
        "H_end": float(hamiltonian[-1]),
        "samples": sample_rows(),
    }
    if ret_len > 1:
        out["dt_start_M"] = float(t_m[1] - t_m[0])
        out["dt_end_M"] = float(t_m[-1] - t_m[-2])
    return out

vectors = {name: vector_summary(name) for name in vector_indices}
merger_params = np.array(outs[21].data, copy=False)
p_mode_t = np.array(outs[7].data, copy=False)
his_data, his_len = vector_data("his")
joined_data, joined_len = vector_data("joined")
his_t = his_data[:his_len]
joined_t = joined_data[:joined_len]
index_join_his = int(np.searchsorted(joined_t, his_t[0], side="left"))
first_his_excluded = joined_len - index_join_his
join = {
    "index_join_his": index_join_his,
    "index_join_attach": int(joined_len),
    "n_his_in_join": int(first_his_excluded),
    "t_join_his_M": float(joined_t[index_join_his]),
    "t_his_start_M": float(his_t[0]),
    "t_dynamics_end_M": float(joined_t[-1]),
    "ret_len_pmodes": int(p_mode_t.size),
}
if index_join_his > 0:
    join["t_adas_before_his_M"] = float(joined_t[index_join_his - 1])
if joined_len < p_mode_t.size:
    t_join_attach = float(p_mode_t[joined_len])
    join["t_join_attach_M"] = t_join_attach
    join["t_first_his_excluded_M"] = t_join_attach

seob_vec = outs[vector_indices[selected]]
data = np.array(seob_vec.data, copy=False)
ret_len = seob_vec.length // dyn_vars

def col(k):
    return data[k * ret_len : (k + 1) * ret_len]

t_m = col(0)
r = col(18)
omega = col(22)
hamiltonian = col(25)
idx_peak = int(np.nanargmax(omega))
n_rows = min(int(p["n_rows"]), ret_len)
n_cols = min(max(int(p.get("n_cols", 15)), 0), dyn_vars)
rows = [[float(data[k * ret_len + i]) for k in range(n_cols)] for i in range(n_rows)]
print(json.dumps({
    "rows": rows,
    "vectors": vectors,
    "join": join,
    "merger": {
        "t_peak_omega_M": float(merger_params[0]),
        "t_attach_M": float(merger_params[1]),
        "t_peak_amp_M": float(merger_params[2]),
        "final_mass_frac": float(merger_params[6]),
        "final_spin": float(merger_params[7]),
        "termination_reason": int(round(merger_params[8])),
    },
    "summary": {
    "n_dyn": int(ret_len),
    "t_start_M": float(t_m[0]),
    "t_end_M": float(t_m[-1]),
    "r_start": float(r[0]),
    "r_end": float(r[-1]),
    "t_peak_omega_M": float(t_m[idx_peak]),
    "omega_peak_dimless": float(omega[idx_peak]),
    "H_start": float(hamiltonian[0]),
    "H_end": float(hamiltonian[-1]),
    },
}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        pytest.skip(f"isolated lalsimulation reference unavailable: {stderr}")
    return json.loads(proc.stdout)


def _lal_phm_checkpoint_summary(params):
    ref = _isolated_lal_phm_dynamics(params)
    summary = dict(ref["summary"])
    summary["vectors"] = ref["vectors"]
    summary["join"] = ref["join"]
    summary["merger"] = ref["merger"]
    return summary


def _lal_phm_dynamics_rows(params, n_rows, vector="joined", n_cols=15):
    return _isolated_lal_phm_dynamics(params, n_rows=n_rows, vector=vector, n_cols=n_cols)["rows"]


def _torch_replay_lal_rk_interval(params_dict, rows, idx):
    """Replay one LAL dynamics interval with the torch RK/RHS path."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform.seobnrv4phm_ode import RKState, _rk45_step_impl

    y0 = torch.tensor(rows[idx][1:15], dtype=torch.float64)
    y1_lal = torch.tensor(rows[idx + 1][1:15], dtype=torch.float64)
    h_lal = torch.tensor(rows[idx + 1][0] - rows[idx][0], dtype=torch.float64)
    t0 = torch.tensor(rows[idx][0], dtype=torch.float64)

    eob_params = {k: v for k, v in params_dict.items() if k != "delta_t"}
    params = dyn.EOBParams(**eob_params)
    if idx == 0:
        # LAL's first RHS starts from the coefficient cache left by the IC root.
        dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    else:
        s1_scale = dyn._lal_spin_scale(params.mass1, params.M)
        s2_scale = dyn._lal_spin_scale(params.mass2, params.M)
        L_vec = torch.linalg.cross(y0[0:3], y0[3:6])
        dyn._refresh_hcoeffs(params, L_vec, y0[6:9] / s1_scale, y0[9:12] / s2_scale)

    dydt0 = dyn.rhs_cartesian_full(t0, y0, params)
    state, _, _ = _rk45_step_impl(
        lambda t, y: dyn.rhs_cartesian_full(t, y, params),
        RKState(t0, y0, h_lal),
        dydt_in=dydt0,
        compute_dydt_out=True,
    )
    return state.y, y1_lal


def _phm_boundary_comparison(lal_ref, torch_summary):
    vectors = lal_ref["vectors"]
    join = lal_ref["join"]
    merger = lal_ref["merger"]
    adas_pairs = list(zip(vectors["adas"]["samples"], torch_summary["adas_samples"], strict=False))
    his_pairs = list(zip(vectors["his"]["samples"], torch_summary["his_uniform_samples"], strict=False))
    return {
        "dn_adas": torch_summary["n_adas"] - vectors["adas"]["n_dyn"],
        "dn_his": torch_summary["n_his"] - vectors["his"]["n_dyn"],
        "dn_joined": torch_summary["n_joined"] - vectors["joined"]["n_dyn"],
        "d_index_join_his": torch_summary["index_join_his"] - join["index_join_his"],
        "d_index_join_attach": torch_summary["index_join_attach"] - join["index_join_attach"],
        "dt_start_his_M": torch_summary["t_start_his_M"] - join["t_his_start_M"],
        "dt_join_his_M": torch_summary["t_join_his_M"] - join["t_join_his_M"],
        "dt_attach_M": torch_summary["t_attach_M"] - merger["t_attach_M"],
        "dt_attach_floor_M": torch_summary["t_join_attach_M"] - join["t_join_attach_M"],
        "dt_joined_end_M": torch_summary["joined_end_M"] - vectors["joined"]["t_end_M"],
        "dr_start": torch_summary["r_start"] - vectors["joined"]["r_start"],
        "dr_his_start": torch_summary["r_his_start"] - vectors["his"]["r_start"],
        "dr_joined_end": torch_summary["r_end"] - vectors["joined"]["r_end"],
        "adas_sample_dt_M": [t["t_M"] - l["t_M"] for l, t in adas_pairs],
        "adas_sample_dr": [t["r"] - l["r"] for l, t in adas_pairs],
        "his_sample_dt_M": [t["t_M"] - l["t_M"] for l, t in his_pairs],
        "his_sample_dr": [t["r"] - l["r"] for l, t in his_pairs],
    }


def _phm_ic_step4_comparison(params_dict):
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    def _native_params():
        return swt._native_params_from_kwargs(params_dict)

    def _row_fields(row):
        return {
            "r": float(row[1]),
            "px": float(row[4]),
            "py": float(row[5]),
            "pz": float(row[6]),
        }

    def _row_dynamics(row):
        keys = (
            "vx",
            "vy",
            "vz",
            "polar_r",
            "polar_phi",
            "polar_pr",
            "polar_pphi",
            "omega",
            "s1dotZ",
            "s2dotZ",
            "H",
        )
        return {key: float(row[15 + idx]) for idx, key in enumerate(keys)}

    def _rhs_from_lal_row(row):
        params = _native_params()
        device = torch.device("cpu")
        dtype = torch.float64
        # LAL's first RHS starts with the coefficient cache left by the IC root.
        dyn.initial_conditions(params, device=device, dtype=dtype)
        y_cart = torch.tensor(row[1:15], device=device, dtype=dtype)
        rhs = dyn.rhs_cartesian_full(
            torch.tensor(row[0], device=device, dtype=dtype),
            y_cart,
            params,
        )
        return {"vx": float(rhs[0]), "vy": float(rhs[1]), "vz": float(rhs[2])}

    def _clean_step4_summary(summary):
        keys = (
            "pr_star",
            "pr_non_tortoise",
            "r_dot",
            "d2Hdr2",
            "d2Hdrdpphi",
            "dHdpphi",
            "dEdr",
            "H",
            "flux",
            "probe_dH_dpvec0",
            "dxdt_probe0",
            "csi",
            "dHdpr",
            "p_norm",
            "pphi",
        )
        return {key: float(summary[key]) for key in keys if key in summary}

    def _step4_from_fields(fields):
        params = _native_params()
        device = torch.device("cpu")
        dtype = torch.float64
        omega = torch.tensor(math.pi * params.f_lower * params.M_sec, device=device, dtype=dtype)
        S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
        S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
        r = fields["r"]
        py = fields["py"]
        pz = fields["pz"]
        L_vec = torch.tensor([0.0, -r * pz, r * py], device=device, dtype=dtype)
        summary = dyn._precessing_ic_radial_momentum_summary(
            params,
            r,
            py,
            pz,
            L_vec,
            S1,
            S2,
            omega,
            final_tortoise=1,
            include_flux_modes=True,
            device=device,
            dtype=dtype,
        )
        assert summary is not None
        out = _clean_step4_summary(summary)
        out["pr_star_minus_px"] = out["pr_star"] - fields["px"]
        if "flux_modes" in summary:
            out["flux_22_frac"] = float(summary["flux_modes"]["2,2"] / summary["flux"])
        return out

    def _step4_required_scalars(summary, target_pr_star):
        pr_probe = 1.0e-3
        pr_star = float(summary["pr_star"])
        csi = float(summary["csi"])
        flux = float(summary["flux"])
        dEdr = float(summary["dEdr"])
        dHdpr = float(summary["dHdpr"])
        dHdpphi = float(summary["dHdpphi"])
        d2Hdr2 = float(summary["d2Hdr2"])
        d2Hdrdpphi = float(summary["d2Hdrdpphi"])

        required_flux = -target_pr_star * dEdr * dHdpr / (csi * pr_probe)
        required_dEdr = -csi * flux * pr_probe / (target_pr_star * dHdpr)
        required_dHdpr = -csi * flux * pr_probe / (target_pr_star * dEdr)
        required_csi = -target_pr_star * dEdr * dHdpr / (flux * pr_probe)
        required_d2Hdr2 = -required_dEdr * d2Hdrdpphi / dHdpphi
        required_d2Hdrdpphi = -dHdpphi * d2Hdr2 / required_dEdr

        return {
            "pr_star_rel_delta": (pr_star - target_pr_star) / target_pr_star,
            "required_flux_ratio": required_flux / flux,
            "required_dEdr_ratio": required_dEdr / dEdr,
            "required_dHdpr_ratio": required_dHdpr / dHdpr,
            "required_csi_ratio": required_csi / csi,
            "required_d2Hdr2_ratio": required_d2Hdr2 / d2Hdr2,
            "required_d2Hdrdpphi_ratio": required_d2Hdrdpphi / d2Hdrdpphi,
        }

    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas", n_cols=26)[0]
    lal_fields = _row_fields(lal_row)
    lal_dynamics = _row_dynamics(lal_row)
    torch_rhs_at_lal = _rhs_from_lal_row(lal_row)

    params = _native_params()
    y_reduced = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    y_cart = dyn.reduced_state_to_cartesian_state(y_reduced, params)
    torch_fields = {
        "r": float(y_cart[0]),
        "px": float(y_cart[3]),
        "py": float(y_cart[4]),
        "pz": float(y_cart[5]),
    }

    step4_at_lal = _step4_from_fields(lal_fields)
    step4_at_torch = _step4_from_fields(torch_fields)

    return {
        "lal_row0": lal_fields,
        "lal_row0_dynamics": lal_dynamics,
        "torch_rhs_at_lal_row0": torch_rhs_at_lal,
        "rhs_velocity_delta": {
            key: torch_rhs_at_lal[key] - lal_dynamics[key]
            for key in torch_rhs_at_lal
        },
        "torch_row0": torch_fields,
        "row0_delta": {key: torch_fields[key] - lal_fields[key] for key in lal_fields},
        "step4_at_lal_row0": step4_at_lal,
        "step4_lal_row0_required_scalar_ratios": _step4_required_scalars(
            step4_at_lal,
            lal_fields["px"],
        ),
        "step4_at_torch_row0": step4_at_torch,
    }


def _format_boundary_diagnostic(lal_ref, torch_summary, comparison):
    vectors = lal_ref["vectors"]
    join = lal_ref["join"]
    merger = lal_ref["merger"]
    parts = [
        "SEOBNRv4PHM AdaS/HiS/join boundary diagnostic",
        (
            f"counts lal=(adas={vectors['adas']['n_dyn']}, his={vectors['his']['n_dyn']}, "
            f"joined={vectors['joined']['n_dyn']}) "
            f"torch=(adas={torch_summary['n_adas']}, his={torch_summary['n_his']}, "
            f"joined={torch_summary['n_joined']})"
        ),
        (
            f"indices lal=(his={join['index_join_his']}, attach={join['index_join_attach']}) "
            f"torch=(his={torch_summary['index_join_his']}, "
            f"attach={torch_summary['index_join_attach']})"
        ),
        (
            f"times lal=(tstartHiS={join['t_his_start_M']:.12g}, "
            f"tAttach={merger['t_attach_M']:.12g}, "
            f"floor={join['t_join_attach_M']:.12g}, end={vectors['joined']['t_end_M']:.12g}) "
            f"torch=(tstartHiS={torch_summary['t_start_his_M']:.12g}, "
            f"tAttach={torch_summary['t_attach_M']:.12g}, "
            f"floor={torch_summary['t_join_attach_M']:.12g}, "
            f"end={torch_summary['joined_end_M']:.12g})"
        ),
        f"deltas={comparison}",
        (
            f"samples lal_adas={vectors['adas']['samples'][:3]} "
            f"torch_adas={torch_summary['adas_samples'][:3]}"
        ),
        f"torch_stops=(adas={torch_summary['adas_stop_state']}, his={torch_summary['his_stop_state']})",
        f"torch_rhs={torch_summary['rhs_derivative_options']}",
    ]
    return "; ".join(parts)


def _format_ic_step4_diagnostic(comparison):
    return "; ".join(
        [
            "SEOBNRv4PHM IC Step 4 diagnostic outside strict parity tolerance",
            f"lal_row0={comparison['lal_row0']}",
            f"lal_row0_dynamics={comparison['lal_row0_dynamics']}",
            f"torch_rhs_at_lal_row0={comparison['torch_rhs_at_lal_row0']}",
            f"rhs_velocity_delta={comparison['rhs_velocity_delta']}",
            f"torch_row0={comparison['torch_row0']}",
            f"row0_delta={comparison['row0_delta']}",
            f"step4_at_lal_row0={comparison['step4_at_lal_row0']}",
            (
                "step4_lal_row0_required_scalar_ratios="
                f"{comparison['step4_lal_row0_required_scalar_ratios']}"
            ),
            f"step4_at_torch_row0={comparison['step4_at_torch_row0']}",
        ]
    )


def _format_checkpoint_diagnostic(lal_summary, torch_summary, comparison):
    lal_join = lal_summary.get("join", {})
    parts = [
        "SEOBNRv4PHM checkpoint diagnostic outside strict parity tolerance",
        f"lal_n={lal_summary['n_dyn']} torch_n={torch_summary['n_traj']}",
        f"dt_end={comparison['dt_end_M']:.3e}M",
        f"dt_peak={comparison['dt_peak_omega_M']:.3e}M",
        f"domega_peak_rel={comparison['domega_peak_rel']:.3e}",
        f"dr_start={comparison['dr_start']:.3e}",
        f"dr_end={comparison['dr_end']:.3e}",
        f"torch_segments=(adas={torch_summary['n_adas']}, his_src={torch_summary['n_his_src']}, "
        f"his_uniform={torch_summary['n_his_uniform']}, start_his={torch_summary['index_start_his']}, "
        f"join_attach={torch_summary['index_join_attach']})",
        (
            f"lal_join=(his={lal_join.get('index_join_his')}, "
            f"attach={lal_join.get('index_join_attach')}, "
            f"t_his={lal_join.get('t_join_his_M')}, "
            f"t_attach_floor={lal_join.get('t_join_attach_M')})"
        ),
        (
            f"torch_join=(his={torch_summary.get('index_join_his')}, "
            f"attach={torch_summary.get('index_join_attach')}, "
            f"t_his={torch_summary.get('t_join_his_M')}, "
            f"t_attach_floor={torch_summary.get('t_join_attach_M')})"
        ),
        f"torch_stops=(adas={torch_summary['adas_stop_state']}, his={torch_summary['his_stop_state']})",
        f"torch_t_attach={torch_summary['t_attach_M']:.3f}M",
        f"torch_final=(Mf={torch_summary['final_mass_frac']:.6f}, af={torch_summary['final_spin']:.6f})",
        f"torch_cal=(cal21={torch_summary['cal21']:.3e}, cal55={torch_summary['cal55']:.3e})",
        f"torch_rhs={torch_summary['rhs_derivative_options']}",
    ]
    return "; ".join(parts)


def test_seobnrv4phm_torch_flag_off_fallback():
    """With native flags off, torch scheme must match CPU fallback."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")}
    try:
        h_cpu = _cpu_waveform()
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "0"
        h_torch, _ = get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single
    n = min(len(h_cpu), len(h_torch))
    np.testing.assert_allclose(h_torch.numpy()[:n], h_cpu.numpy()[:n], rtol=1e-10, atol=1e-14)


@pytest.fixture(scope="module")
def native_phm_waveform():
    """Generate the expensive native waveform once for smoke and parity."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {
        k: os.environ.get(k)
        for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")
    }
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "1"
        return get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.skipif(not RUN_SLOW_NATIVE_PHM, reason=SLOW_NATIVE_PHM_REASON)
def test_seobnrv4phm_torch_native_smoke(native_phm_waveform):
    """Native torch path should run and produce finite spectra."""
    h_torch, _ = native_phm_waveform
    arr = h_torch.numpy()
    assert np.isfinite(arr).all()
    assert arr.ndim == 1 and len(arr) > 8
    assert len(arr) == int(PARAMS["f_final"] / PARAMS["delta_f"]) + 1


@pytest.mark.skipif(not RUN_SLOW_NATIVE_PHM, reason=SLOW_NATIVE_PHM_REASON)
def test_seobnrv4phm_torch_optional_parity(native_phm_waveform):
    """Check in-band numerical parity against the CPU/LAL reference."""
    h_cpu = _cpu_waveform()
    h_torch, _ = native_phm_waveform
    metrics = _phm_fd_parity_metrics(h_cpu, h_torch)
    tol = _PHM_FD_PARITY_TOLERANCES
    within_tolerance = (
        metrics["bins"] > 0
        and metrics["rel"] < tol["raw_rel"]
        and metrics["amp_min"] > tol["amp_min"]
        and metrics["amp_max"] < tol["amp_max"]
        and abs(metrics["phase_mean"]) < tol["raw_phase_mean"]
        and metrics["phase_std"] < tol["raw_phase_std"]
        and abs(metrics["linear_dt_s"]) < tol["linear_dt_s"]
        and metrics["aligned_rel"] < tol["aligned_rel"]
        and metrics["aligned_phase_std"] < tol["aligned_phase_std"]
    )
    assert within_tolerance, _format_phm_fd_parity_metrics(metrics)


@pytest.mark.skipif(not RUN_BOUNDARY_DIAGNOSTIC, reason=BOUNDARY_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_boundary_diagnostic(monkeypatch):
    """Fast trajectory-boundary diagnostic before merger/NQC/RD work."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = _phm_boundary_params()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_AUTO_TMAX", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    monkeypatch.setenv(
        "PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS",
        os.environ.get("PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS", "4"),
    )
    monkeypatch.setenv(
        "PYCBC_SEOBNRV4PHM_HIS_MAX_STEPS",
        os.environ.get("PYCBC_SEOBNRV4PHM_HIS_MAX_STEPS", "2"),
    )
    lal_ref = _isolated_lal_phm_dynamics(params)
    torch_summary = swt._seobnrv4phm_native_boundary_summary(
        **params,
        t_attach_M=lal_ref["merger"]["t_attach_M"],
    )
    comparison = _phm_boundary_comparison(lal_ref, torch_summary)

    within_tolerance = (
        comparison["dn_adas"] == 0
        and comparison["dn_his"] == 0
        and comparison["dn_joined"] == 0
        and comparison["d_index_join_his"] == 0
        and comparison["d_index_join_attach"] == 0
        and abs(comparison["dt_start_his_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_join_his_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_attach_floor_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_joined_end_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dr_start"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
        and abs(comparison["dr_his_start"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
        and abs(comparison["dr_joined_end"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
    )
    assert within_tolerance, _format_boundary_diagnostic(lal_ref, torch_summary, comparison)


@pytest.mark.skipif(not RUN_IC_STEP4_DIAGNOSTIC, reason=IC_STEP4_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_ic_step4_diagnostic(monkeypatch):
    """Compare row-0 LAL IC inputs against torch Step 4 ingredients."""
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")

    comparison = _phm_ic_step4_comparison(_phm_boundary_params())
    row0_delta = comparison["row0_delta"]
    step4_at_lal = comparison["step4_at_lal_row0"]

    within_tolerance = (
        abs(row0_delta["r"]) < 1.0e-10
        and abs(row0_delta["px"]) < 1.0e-10
        and abs(row0_delta["py"]) < 1.0e-10
        and abs(row0_delta["pz"]) < 1.0e-10
        and abs(step4_at_lal["pr_star_minus_px"]) < 1.0e-10
    )
    assert within_tolerance, _format_ic_step4_diagnostic(comparison)


@pytest.mark.skipif(not RUN_CHECKPOINT_DIAGNOSTIC, reason=CHECKPOINT_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_checkpoint_diagnostic(monkeypatch):
    """Strict checkpoint diagnostic before the full FD parity path."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = _phm_boundary_params()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_AUTO_TMAX", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    lal_summary = _lal_phm_checkpoint_summary(params)
    try:
        torch_summary = swt._seobnrv4phm_native_checkpoint_summary(**params)
    except RuntimeError as exc:
        pytest.fail(f"SEOBNRv4PHM checkpoint path failed: {exc}")

    omega_ref = max(abs(lal_summary["omega_peak_dimless"]), 1.0e-12)
    comparison = {
        "dt_end_M": torch_summary["t_end_M"] - lal_summary["t_end_M"],
        "dt_peak_omega_M": torch_summary["t_peak_omega_argmax_M"] - lal_summary["t_peak_omega_M"],
        "domega_peak_rel": (torch_summary["omega_peak_dimless"] - lal_summary["omega_peak_dimless"]) / omega_ref,
        "dr_start": torch_summary["r_start"] - lal_summary["r_start"],
        "dr_end": torch_summary["r_end"] - lal_summary["r_end"],
    }
    tol = _PHM_CHECKPOINT_PARITY_TOLERANCES
    within_tolerance = (
        abs(comparison["dt_end_M"]) < tol["dt_M"]
        and abs(comparison["dt_peak_omega_M"]) < tol["dt_M"]
        and abs(comparison["domega_peak_rel"]) < tol["omega_peak_rel"]
        and abs(comparison["dr_start"]) < tol["r"]
        and abs(comparison["dr_end"]) < tol["r"]
    )
    assert within_tolerance, _format_checkpoint_diagnostic(lal_summary, torch_summary, comparison)


def test_adas_initial_step_matches_lal_delta_t_scaling():
    """AdaS should use LAL's INdeltaT/mTScaled initial RK step."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=55.0,
        f_ref=55.0,
        mode_array=((2, 2), (2, 1)),
    )
    delta_t = 1.0 / 4096.0
    assert math.isclose(
        swt._lal_adas_initial_step_M(params, delta_t),
        delta_t / params.M_sec,
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def test_lal_phm_dynamics_vector_layout_exposes_adas_his_join():
    """The isolated LAL helper should expose raw AdaS, HiS, and joined vectors."""
    params_dict = _phm_boundary_params()

    ref = _isolated_lal_phm_dynamics(params_dict)
    vectors = ref["vectors"]
    join = ref["join"]
    merger = ref["merger"]

    assert vectors["adas"]["n_dyn"] > join["index_join_his"]
    assert vectors["his"]["dt_start_M"] == pytest.approx(1.0 / 50.0)
    assert join["t_join_his_M"] == pytest.approx(join["t_his_start_M"])
    assert vectors["joined"]["n_dyn"] == join["index_join_his"] + join["n_his_in_join"]
    assert vectors["joined"]["t_end_M"] < vectors["his"]["t_end_M"]
    assert join["t_dynamics_end_M"] < merger["t_attach_M"]
    assert join["t_join_attach_M"] >= merger["t_attach_M"] - 1.0 / 50.0
    assert vectors["adas"]["samples"][0]["t_M"] == pytest.approx(0.0)
    assert vectors["adas"]["samples"][0]["r"] == pytest.approx(vectors["adas"]["r_start"])
    assert len(vectors["adas"]["samples"][0]["state"]) == 14


def test_lal_phm_joined_dynamics_excludes_attachment_floor_sample():
    """LAL's joined dynamics length is indexJoinAttach, excluding that sample."""
    params_dict = _phm_boundary_params()

    ref = _isolated_lal_phm_dynamics(params_dict)
    vectors = ref["vectors"]
    join = ref["join"]

    assert join["index_join_attach"] == vectors["joined"]["n_dyn"]
    assert join["t_dynamics_end_M"] == pytest.approx(vectors["joined"]["t_end_M"])
    assert join["t_join_attach_M"] == pytest.approx(join["t_first_his_excluded_M"])
    assert join["t_dynamics_end_M"] < join["t_join_attach_M"]
    assert join["t_join_attach_M"] - join["t_dynamics_end_M"] == pytest.approx(
        1.0 / 50.0
    )
    assert vectors["joined"]["t_peak_omega_M"] == pytest.approx(vectors["joined"]["t_end_M"])
    assert vectors["his"]["t_peak_omega_M"] > join["t_join_attach_M"]


def test_phm_integration_max_step_env(monkeypatch):
    """Native trajectory diagnostics should support global and stage step caps."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_MAX_STEPS", "11")
    assert swt._integration_max_steps("adas", 200000) == 11

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS", "7")
    assert swt._integration_max_steps("adas", 200000) == 7
    assert swt._integration_max_steps("his", 200000) == 11


def test_precessing_initial_conditions_match_lal_spherical_orbit(monkeypatch):
    """Projected ICs should solve the LAL-style precessing spherical orbit."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(y0[0], y0[1], y0[2], y0[3:6], y0[6:9], y0[9:12], params)

    # LAL SimIMRSpinPrecEOBWaveformAll first dynamics row for this fixture.
    torch.testing.assert_close(r_vec, torch.tensor([13.61567447, 0.0, 0.0], device=device, dtype=dtype), rtol=0.0, atol=2e-5)
    torch.testing.assert_close(p_vec[1:], torch.tensor([0.30381503, -0.000622258], device=device, dtype=dtype), rtol=0.0, atol=2e-6)
    torch.testing.assert_close(torch.linalg.cross(r_vec, p_vec), y0[3:6], rtol=1e-11, atol=1e-11)


def test_precessing_ic_radial_momentum_boundary_tracks_lal(monkeypatch):
    """STEP 4 and the returned radial momentum should track LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = _phm_boundary_params()
    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas")[0]
    params = swt._native_params_from_kwargs(params_dict)
    device = torch.device("cpu")
    dtype = torch.float64
    px_lal = lal_row[4]
    captured = {}
    radial_momentum_summary = dyn._precessing_ic_radial_momentum_summary

    def capture_radial_momentum_summary(*args, **kwargs):
        # Step 4 precedes LAL's spherical/Cartesian round trip.  Reapplying it
        # to the returned row is not equivalent: py and pz each round by one
        # ulp, enough to steer the adaptive finite-difference retry.  Capture
        # the actual pre-round-trip call instead.
        kwargs["include_flux_modes"] = True
        summary = radial_momentum_summary(*args, **kwargs)
        captured["summary"] = summary
        return summary

    monkeypatch.setattr(
        dyn,
        "_precessing_ic_radial_momentum_summary",
        capture_radial_momentum_summary,
    )
    initial_state = dyn.initial_cartesian_conditions(
        params,
        device=device,
        dtype=dtype,
    )
    summary = captured.get("summary")

    assert summary is not None
    for key in ("flux", "dEdr", "dxdt_probe0", "pr_star"):
        assert torch.isfinite(summary[key])
    assert "flux_modes" in summary
    flux_sum = sum(float(v) for v in summary["flux_modes"].values())
    assert flux_sum == pytest.approx(float(summary["flux"]), rel=0.0, abs=1.0e-18)
    assert float(summary["flux_modes"]["2,2"] / summary["flux"]) > 0.99
    assert 0.0 < float(summary["flux_modes"]["5,5"] / summary["flux"]) < 5.0e-5
    pr_star = float(summary["pr_star"])
    px_torch = float(initial_state[3])
    assert pr_star == pytest.approx(px_torch, rel=0.0, abs=5.0e-15)
    assert px_torch == pytest.approx(px_lal, rel=0.0, abs=5.0e-11)
    assert abs((px_torch - px_lal) / px_lal) < 1.0e-8


def test_precessing_ic_root_radius_uses_lal_extension_order():
    """The IC root callback mirrors the loaded LAL extension's split multiply/add."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    x_scaled = float.fromhex("0x1.0be0ded288ce8p-3")
    assert dyn._precessing_ic_root_radius(x_scaled).hex() == "0x1.80175b315b098p+2"


def test_precessing_spin_weight_uses_lal_v4p_order():
    """v4P stores spin states as chi*m_i*m_i/M/M, not chi*(m_i/M)^2."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    assert dyn._lal_spin_weight(0.03, 12.0, 20.0).hex() == "0x1.61e4f765fd8aep-7"
    assert (0.03 * (12.0 / 20.0) * (12.0 / 20.0)).hex() == "0x1.61e4f765fd8adp-7"


def test_lal_initial_row_rhs_velocity_matches_torch(monkeypatch):
    """At the LAL IC row, remaining drift is upstream of RHS/AdaS replay."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = _phm_boundary_params()
    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas", n_cols=26)[0]
    params = swt._native_params_from_kwargs(params_dict)
    device = torch.device("cpu")
    dtype = torch.float64

    dyn.initial_conditions(params, device=device, dtype=dtype)
    y_cart = torch.tensor(lal_row[1:15], device=device, dtype=dtype)
    rhs = dyn.rhs_cartesian_full(
        torch.tensor(lal_row[0], device=device, dtype=dtype),
        y_cart,
        params,
    )

    torch.testing.assert_close(
        rhs[:3],
        torch.tensor(lal_row[15:18], device=device, dtype=dtype),
        rtol=0.0,
        atol=8.0e-10,
    )
    assert lal_row[18] == pytest.approx(lal_row[1], rel=0.0, abs=1.0e-15)
    assert lal_row[20] == pytest.approx(lal_row[4], rel=0.0, abs=1.0e-15)


def test_precessing_initial_conditions_balance_projected_pdot(monkeypatch):
    """Default projected RHS should not immediately kick the rooted IC outward."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")
    monkeypatch.delenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", raising=False)

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dy = dyn.rhs_cartesian_projected(torch.tensor(0.0, device=device, dtype=dtype), y0, params)

    assert abs(float(dy[1])) < 1.0e-5
    assert float(dy[0]) < 0.0


def test_spins_almost_aligned_zeroes_inplane_and_skips_precessing_root(monkeypatch):
    """LAL's SpinsAlmostAligned branch zeroes xy spins and bypasses precessing ICs."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    def fail_precessing_root(*args, **kwargs):
        raise AssertionError("aligned-spin ICs should not call the precessing root")

    monkeypatch.setattr(dyn, "_precessing_spherical_initial_conditions", fail_precessing_root)
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=0.5 * dyn.EPS_ALIGN,
        spin1y=-0.25 * dyn.EPS_ALIGN,
        spin1z=PARAMS["spin1z"],
        spin2x=0.0,
        spin2y=0.5 * dyn.EPS_ALIGN,
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    assert params.aligned_spins is True
    assert params.spin1x == params.spin1y == params.spin2x == params.spin2y == 0.0

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)

    torch.testing.assert_close(y0[3:5], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(y0[6:8], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(y0[9:11], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)


def test_spins_almost_aligned_threshold_is_strict():
    """LAL uses a strict in-plane spin norm < EPS_ALIGN test."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=dyn.EPS_ALIGN,
        spin1y=0.0,
        spin1z=PARAMS["spin1z"],
        spin2x=0.0,
        spin2y=0.0,
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    assert params.aligned_spins is False
    assert params.spin1x == dyn.EPS_ALIGN


def test_cartesian_x_gradient_matches_finite_difference(monkeypatch):
    """Autograd fixed-P Cartesian force should mirror LAL's exact derivative block."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(y0[0], y0[1], y0[2], y0[3:6], y0[6:9], y0[9:12], params)

    grad_auto = dyn._dH_dx_cartesian_autograd(r_vec, p_vec, y0[6:9], y0[9:12], params)
    grad_fd = dyn._dH_dx_cartesian_fd(r_vec, p_vec, y0[6:9], y0[9:12], params)
    torch.testing.assert_close(grad_auto, grad_fd, rtol=1e-6, atol=1e-8)


def test_euler_j2p_initial_gamma_alignment():
    """Initial gamma should reproduce (n, lambda, L) alignment w.r.t. J-basis."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    J_hat = torch.tensor([0.3, 0.4, 0.86602540378], device=device, dtype=dtype)
    e1J, e2J, e3J = swt._build_J_frame(J_hat)
    Lhat = torch.tensor([0.2, 0.6, 0.75], device=device, dtype=dtype)
    Lhat = Lhat / torch.linalg.norm(Lhat)
    Lvec = Lhat.unsqueeze(0).repeat(5, 1)
    phi0 = torch.tensor(0.7, device=device, dtype=dtype)
    dt = 0.1

    alpha, beta, gamma = swt._euler_j2p(Lvec, e1J, e2J, e3J, dt, phi0=phi0)
    gamma_expected = swt._initial_gamma_from_sep(Lhat, e1J, e2J, e3J, phi0)

    torch.testing.assert_close(gamma[0], gamma_expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(gamma, torch.full_like(gamma, gamma_expected), atol=1e-12, rtol=0.0)


def test_euler_j2p_initial_gamma_uses_supplied_separation():
    """Initial gamma should use the dynamics separation vector, not projected phi."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    J_hat = torch.tensor([0.25, -0.35, 0.90277350426], device=device, dtype=dtype)
    e1J, e2J, e3J = swt._build_J_frame(J_hat)
    Lhat = torch.tensor([0.45, -0.25, 0.85], device=device, dtype=dtype)
    Lhat = Lhat / torch.linalg.norm(Lhat)
    n_raw = torch.tensor([0.1, 0.9, 0.35], device=device, dtype=dtype)
    n_hat = n_raw - torch.dot(n_raw, Lhat) * Lhat
    n_hat = n_hat / torch.linalg.norm(n_hat)
    Lvec = Lhat.unsqueeze(0).repeat(5, 1)
    t_vec = torch.linspace(0.0, 0.4, 5, device=device, dtype=dtype)

    _, _, gamma = swt._euler_j2p(
        Lvec,
        e1J,
        e2J,
        e3J,
        phi0=torch.tensor(1.9, device=device, dtype=dtype),
        n_hat0=n_hat,
        t_vec=t_vec,
    )
    gamma_expected = swt._initial_gamma_from_sep_vector(
        Lhat,
        n_hat,
        e1J,
        e2J,
        e3J,
    )

    torch.testing.assert_close(gamma[0], gamma_expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(gamma, torch.full_like(gamma, gamma_expected), atol=1e-12, rtol=0.0)


def test_spins_almost_aligned_euler_angles_are_zero():
    """LAL leaves I/J/P Euler rotations at zero in the aligned-spin branch."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([0.0, 0.3, 0.9, 1.7], dtype=torch.float64)
    alpha, beta, gamma = swt._zero_euler_angles_like(t)

    torch.testing.assert_close(alpha, torch.zeros_like(t), rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta, torch.zeros_like(t), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma, torch.zeros_like(t), rtol=0.0, atol=0.0)


def test_lal_output_time_grid_uses_floor_retlen():
    """Output grid length should mirror LAL's retLenTS floor calculation."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t_modes_m = torch.tensor([0.0, 0.7, 3.74], dtype=torch.float64)
    delta_t = 0.2
    m_sec = 2.0
    out_m = swt._lal_output_time_grid(t_modes_m, delta_t, m_sec)

    expected_len = math.floor((float(t_modes_m[-1]) - float(t_modes_m[0])) / (delta_t / m_sec))
    assert len(out_m) == expected_len
    torch.testing.assert_close(out_m, torch.arange(expected_len, dtype=torch.float64) * (delta_t / m_sec))


def test_amplitude_peak_from_22_21_matches_lal_discrete_rules():
    """The PHM epoch peak is the first discrete max of |h22|^2 plus requested |h21|^2."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float64)
    h22 = torch.tensor([1.0, 3.0, 3.0, 0.5], dtype=torch.complex128)
    h21 = torch.tensor([0.0, 0.0, 5.0, 0.0], dtype=torch.complex128)
    modes = {(2, 2): h22, (2, 1): h21}

    t_peak_22, idx_22 = swt._amplitude_peak_from_22_21(modes, ((2, 2),), t)
    assert idx_22 == 1
    torch.testing.assert_close(t_peak_22, t[1])

    t_peak_both, idx_both = swt._amplitude_peak_from_22_21(modes, ((2, 2), (2, 1)), t)
    assert idx_both == 2
    torch.testing.assert_close(t_peak_both, t[2])

    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        swt._amplitude_peak_from_22_21(modes, ((2, 1),), t)


def test_nyquist_check_rejects_ringdown_above_sample_rate():
    """LAL rejects delta_t values that put the checked ringdown mode above Nyquist."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.05,
        spin2x=-0.1,
        spin2y=0.07,
        spin2z=0.2,
        distance=300.0,
        inclination=0.6,
        f_lower=30.0,
        f_ref=50.0,
        mode_array=((2, 2), (5, 5)),
    )
    omega_rd = swt._ringdown_omega_for_nyquist(params, 5)
    dt_limit = math.pi / omega_rd

    swt._check_nyquist_frequency(params, 0.5 * dt_limit, 5, waveform_ell_max=5)
    with pytest.raises(ValueError, match="Nyquist"):
        swt._check_nyquist_frequency(params, 1.1 * dt_limit, 5, waveform_ell_max=5)
    with pytest.raises(ValueError, match=">= 2"):
        swt._check_nyquist_frequency(params, 0.5 * dt_limit, 1, waveform_ell_max=5)


def test_reduced_stop_quantities_match_cartesian_orbital_scalars():
    """Reduced-state stop checks should use the same Cartesian scalars as LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.0,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    y = torch.tensor(
        [6.0, -0.02, 0.4, 0.0, 0.0, 3.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2],
        dtype=torch.float64,
    )
    dy = torch.tensor(
        [0.15, -0.04, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )

    q = swt._lal_stop_quantities_from_reduced(y, dy, params)

    torch.testing.assert_close(q["drdt"], dy[0], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(q["omega"], dy[2], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(q["p_dot_r"], y[1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(q["pr_dot"], dy[1], rtol=0.0, atol=2e-7)
    assert bool(q["pphi_large"]) is False


def test_minimal_rotation_gamma_integrates_cubic_alpha_spline():
    """Gamma integration should use spline alpha_dot rather than sample differencing."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    dt = 0.25
    t = torch.arange(9, dtype=dtype) * dt
    rate = torch.tensor(0.37, dtype=dtype)
    beta0 = torch.tensor(0.63, dtype=dtype)
    gamma0 = torch.tensor(-0.2, dtype=dtype)
    alpha = 0.4 + rate * t
    beta = torch.full_like(alpha, beta0)

    gamma = swt._integrate_minimal_rotation_gamma(alpha, beta, dt, gamma0)
    expected = gamma0 - rate * torch.cos(beta0) * t

    torch.testing.assert_close(gamma, expected, rtol=1e-12, atol=1e-12)


def test_minimal_rotation_gamma_integrates_nonuniform_times():
    """Joined AdaS/HiS Euler gamma integration should use actual mode times."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 0.17, 0.41, 0.9, 1.4, 2.1], dtype=dtype)
    rate = torch.tensor(0.22, dtype=dtype)
    beta0 = torch.tensor(0.48, dtype=dtype)
    gamma0 = torch.tensor(0.13, dtype=dtype)
    alpha = -0.2 + rate * t
    beta = torch.full_like(alpha, beta0)

    gamma = swt._integrate_minimal_rotation_gamma_times(alpha, beta, t, gamma0)
    expected = gamma0 - rate * torch.cos(beta0) * t

    torch.testing.assert_close(gamma, expected, rtol=1e-12, atol=1e-12)


def test_euler_post_merger_extension_starts_at_attachment():
    """Post-attach dynamics angles are discarded before QNM precession extension."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    alpha = torch.tensor([0.0, 0.2, 0.5, 9.0, 9.5], dtype=dtype)
    beta = torch.tensor([0.3, 0.4, 0.6, 2.0, 2.5], dtype=dtype)
    gamma = torch.tensor([0.1, 0.05, -0.2, 8.0, 8.5], dtype=dtype)
    dt = 0.5
    prec_rate = torch.tensor(0.2, dtype=dtype)

    alpha_out, beta_out, gamma_out = swt._extend_euler_from_attach(
        alpha,
        beta,
        gamma,
        index_start=3,
        target_len=6,
        dt=dt,
        prec_rate=prec_rate,
    )

    torch.testing.assert_close(alpha_out[:3], alpha[:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta_out[:3], beta[:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[:3], gamma[:3], rtol=0.0, atol=0.0)

    t = torch.arange(1, 4, dtype=dtype) * dt
    torch.testing.assert_close(alpha_out[3:], alpha[2] + prec_rate * t, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beta_out[3:], torch.full((3,), beta[2], dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[3:], gamma[2] - torch.cos(beta[2]) * prec_rate * t, rtol=1e-12, atol=1e-12)


def test_euler_post_merger_extension_uses_joined_mode_times():
    """Post-attach extension should follow the non-uniform joined P-mode times."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_modes = torch.tensor([0.0, 0.4, 1.1, 1.3, 1.32, 1.34], dtype=dtype)
    alpha = torch.tensor([0.0, 0.2, 0.5, 9.0], dtype=dtype)
    beta = torch.tensor([0.3, 0.4, 0.6, 2.0], dtype=dtype)
    gamma = torch.tensor([0.1, 0.05, -0.2, 8.0], dtype=dtype)
    prec_rate = torch.tensor(0.7, dtype=dtype)

    alpha_out, beta_out, gamma_out = swt._extend_euler_from_attach_times(
        alpha,
        beta,
        gamma,
        t_modes,
        index_start=3,
        prec_rate=prec_rate,
    )

    torch.testing.assert_close(alpha_out[:3], alpha[:3], rtol=0.0, atol=0.0)
    dt = t_modes[3:] - t_modes[2]
    torch.testing.assert_close(alpha_out[3:], alpha[2] + prec_rate * dt, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beta_out[3:], torch.full((3,), beta[2], dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[3:], gamma[2] - torch.cos(beta[2]) * prec_rate * dt, rtol=1e-12, atol=1e-12)


def test_euler_qnm_precession_rate_matches_lal_checkpoint():
    """Use the Cardoso QNM table that drives LAL's Euler extension."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    rate = swt._euler_qnm_precession_rate(
        0.7041349531072489,
        0.9512185112429709,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert float(rate) == pytest.approx(0.08245331084837076, abs=1.0e-14)


def test_euler_attach_index_uses_lal_floor_sample():
    """LAL's joined Euler extension index is the last sample <= tAttach."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    assert swt._last_index_leq_increasing(t, 1.9) == 1
    assert swt._nearest_index_increasing(t, 1.9) == 2


def test_constant_inertial_rotation_matches_vectorized():
    """Constant-angle J→I rotation should match per-sample Wigner application."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    t = torch.linspace(0.0, 0.35, steps=9, device=device, dtype=dtype)
    h22 = torch.exp(1j * t)
    h21 = 0.3 * (torch.sin(t) + 1j * torch.cos(t))
    modes = {
        (2, 2): h22,
        (2, 1): h21,
        (2, -1): ((-1.0) ** 2) * torch.conj(h21),
        (2, -2): ((-1.0) ** 2) * torch.conj(h22),
    }
    alpha = torch.tensor(0.31, device=device, dtype=dtype)
    beta = torch.tensor(0.23, device=device, dtype=dtype)
    gamma = torch.tensor(-0.17, device=device, dtype=dtype)

    rot_vec = swt._rotate_modes(modes, alpha.repeat(len(t)), beta.repeat(len(t)), gamma.repeat(len(t)))
    rot_const = swt._rotate_modes_constant(modes, alpha, beta, gamma)

    for key in rot_vec.keys():
        torch.testing.assert_close(rot_const[key], rot_vec[key], rtol=1e-10, atol=1e-12)


def test_rotate_interpolate_matches_rotation_on_input_grid():
    """LAL-style rotate/interpolate reduces to direct rotation on source samples."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 0.3, 0.8, 1.2, 1.9], dtype=dtype)
    alpha = 0.2 + 0.13 * t
    beta = 0.4 + 0.03 * torch.sin(t)
    gamma = -0.1 + 0.07 * t
    h22 = (1.0 + 0.2 * t) * torch.exp(0.3j * t)
    h21 = (0.3 + 0.1 * t) * torch.exp(-0.4j * t)
    modes = {
        (2, 2): h22,
        (2, 1): h21,
        (2, -1): ((-1.0) ** 2) * torch.conj(h21),
        (2, -2): ((-1.0) ** 2) * torch.conj(h22),
    }

    got = swt._rotate_interpolate_modes_jframe(modes, alpha, beta, gamma, t, t, ((2, 2), (2, 1)))
    expected = swt._rotate_modes(modes, alpha, beta, gamma)

    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-10, atol=1e-12)


def test_rotate_interpolate_splines_pmode_phase():
    """P-frame modes should be interpolated as amp/phase, not complex samples."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    t_out = torch.tensor([1.5], dtype=dtype)
    amp = torch.tensor([1.0, 1.2, 1.4, 1.6], dtype=dtype)
    phase = torch.tensor([2.8, 3.1, 3.3, 3.5], dtype=dtype)
    h22 = amp * torch.exp(1j * phase)
    modes = {(2, 2): h22, (2, -2): torch.conj(h22)}
    zero = torch.zeros_like(t)

    got = swt._rotate_interpolate_modes_jframe(modes, zero, zero, zero, t, t_out, ((2, 2),))[(2, 2)]
    expected_amp = swt._interp_series_cubic(t_out, t, amp)
    expected_phase = swt._interp_series_cubic(t_out, t, phase)
    expected = expected_amp * torch.exp(1j * expected_phase)

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_wigner_d_element_matches_lal():
    """Rotation helper should use LAL's XLALWignerdMatrix sign convention."""
    try:
        import lal
    except Exception as exc:
        pytest.skip(f"lal unavailable: {exc}")

    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    beta = torch.tensor(0.7, dtype=torch.float64)
    for l, m, mp in [(2, 2, 1), (2, 2, 0), (3, -1, 2), (4, 3, -2)]:
        got = swt._wigner_d_element(l, mp, m, beta)
        expected = torch.tensor(lal.WignerdMatrix(l, m, mp, float(beta)), dtype=torch.float64)
        torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-13)


def test_build_j_frame_matches_lal_projection():
    """The final-J frame e1 axis is projected x, not x cross J."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64

    def projected_basis(J, ref):
        e3 = J / torch.linalg.norm(J)
        e1 = ref - torch.dot(ref, e3) * e3
        e1 = e1 / torch.linalg.norm(e1)
        e2 = torch.cross(e3, e1, dim=0)
        e2 = e2 / torch.linalg.norm(e2)
        return e1, e2, e3

    J = torch.tensor([0.24, -0.31, 0.92], dtype=dtype)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    got = swt._build_J_frame(J)
    expected = projected_basis(J, ex)
    for g, e in zip(got, expected, strict=False):
        torch.testing.assert_close(g, e, rtol=1e-13, atol=1e-13)

    J_near_x = torch.tensor([1.0, 1.0e-8, 2.0e-8], dtype=dtype)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    got = swt._build_J_frame(J_near_x)
    expected = projected_basis(J_near_x, ey)
    for g, e in zip(got, expected, strict=False):
        torch.testing.assert_close(g, e, rtol=1e-13, atol=1e-13)


def test_euler_from_basis_matches_lal_formula():
    """I->J Euler angles should use LAL's active-rotation matrix entries."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    J = torch.tensor([0.24, -0.31, 0.92], dtype=torch.float64)
    e1, e2, e3 = swt._build_J_frame(J)
    alpha, beta, gamma = swt._euler_from_basis(e1, e2, e3)

    expected_alpha = torch.atan2(e3[1], e3[0])
    expected_beta = torch.acos(e3[2])
    expected_gamma = torch.atan2(e2[2], -e1[2])

    torch.testing.assert_close(alpha, expected_alpha, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(beta, expected_beta, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(gamma, expected_gamma, rtol=1e-13, atol=1e-13)


def test_final_mass_spin_prec_matches_lal():
    """Torch remnant fit should reproduce LALSimIMREOBFinalMassSpinPrec."""
    try:
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    from pycbc.waveform import seobnrv4phm_torch as swt

    m1 = 25.0
    m2 = 18.0
    spin1 = (0.2, -0.15, 0.05)
    spin2 = (-0.1, 0.07, 0.2)

    status, mf_lal, af_lal = lalsim.SimIMREOBFinalMassSpinPrec(m1, m2, spin1, spin2, lalsim.SEOBNRv4PHM)
    if status != 0:
        pytest.skip("LAL remnant fit returned failure")

    mf, af = swt._final_mass_spin_prec(m1, m2, spin1, spin2)
    assert np.isclose(mf, mf_lal, rtol=0, atol=1e-13)
    assert np.isclose(af, af_lal, rtol=0, atol=1e-13)


def test_nqc_peak_delta_t_v4_matches_lal_fit():
    """Peak timing offset should mirror XLALSimIMREOBGetNRSpinPeakDeltaTv4."""
    from pycbc.waveform.seobnrv4phm_nqc import peak_delta_t_v4

    dt22 = peak_delta_t_v4(2, 2, 25.0, 18.0, 0.05, 0.2)
    dt55 = peak_delta_t_v4(5, 5, 25.0, 18.0, 0.05, 0.2)

    assert np.isclose(dt22, 4.561818695562277, rtol=0.0, atol=1e-13)
    assert np.isclose(dt55 - dt22, 10.0, rtol=0.0, atol=1e-15)


def test_nqc_22_peak_amplitude_fits_match_lal():
    """The 22 amplitude targets must include LAL's eta normalization."""
    from pycbc.waveform.seobnrv4phm_nqc import peak_addot_v4, peak_adot_v4, peak_amp_v4

    eta = 0.24
    chiS = 0.1
    chiA = -0.05

    assert np.isclose(peak_amp_v4(2, 2, eta, chiS, chiA), 0.37608626846721882, rtol=0.0, atol=1e-15)
    assert peak_adot_v4(2, 2, eta, chiS, chiA) == 0.0
    assert np.isclose(peak_addot_v4(2, 2, eta, chiS, chiA), -0.00092616014379767926, rtol=0.0, atol=1e-17)


def test_coprecessing_modes_can_skip_distance_scaling_for_nqc():
    """The NQC first pass must use LAL's dimensionless P-frame modes."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    n = 8
    r = torch.linspace(12.0, 10.5, n, dtype=dtype)
    pr = torch.linspace(-2.0e-3, -4.0e-3, n, dtype=dtype)
    phi = torch.linspace(0.0, 0.4, n, dtype=dtype)
    omega = torch.full((n,), 0.03 / params.M_sec, dtype=dtype)
    Lvec = torch.zeros((n, 3), dtype=dtype)
    Lvec[:, 2] = 3.8
    S1 = torch.tensor([[params.spin1x, params.spin1y, params.spin1z]], dtype=dtype).repeat(n, 1)
    S2 = torch.tensor([[params.spin2x, params.spin2y, params.spin2z]], dtype=dtype).repeat(n, 1)

    h_unscaled = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    h_scaled = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=True
    )[(2, 2)]
    scale = (params.M * swt._MRSUN_SI) / (params.distance * 1.0e6 * swt._PC_SI)

    torch.testing.assert_close(h_scaled, h_unscaled * scale, rtol=1e-12, atol=1e-30)

    params.nqc_b_map = {(2, 2): {"b1": 0.7, "b2": 0.0, "b3": 0.0, "b4": 0.0}}
    h_nqc = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    rOmega = r * torch.abs(omega * params.M_sec)
    expected_phase = 0.7 * pr / rOmega
    torch.testing.assert_close(h_nqc, h_unscaled * torch.exp(1j * expected_phase), rtol=1e-12, atol=1e-12)


def test_coprecessing_mode_uses_lal_newtonian_phase():
    """With fixed dynamics, the P-frame mode phase follows Y_{l-eps,-m}."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    phi = torch.tensor([0.0, 0.37], dtype=dtype)
    r = torch.full((2,), 12.0, dtype=dtype)
    pr = torch.full((2,), -2.0e-3, dtype=dtype)
    omega = torch.full((2,), 0.03 / params.M_sec, dtype=dtype)
    Lvec = torch.zeros((2, 3), dtype=dtype)
    Lvec[:, 2] = 3.8
    S1 = torch.tensor([[0.0, 0.0, params.spin1z]], dtype=dtype).repeat(2, 1)
    S2 = torch.tensor([[0.0, 0.0, params.spin2z]], dtype=dtype).repeat(2, 1)

    h22 = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    torch.testing.assert_close(h22[1] / h22[0], torch.exp(-2j * phi[1]), rtol=1e-11, atol=1e-12)


def test_high_mode_delta_vh9_terms_match_lal_nesting():
    """The 33/44/55 d9 terms contain three powers of H*Omega."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_waveform_coeffs as coeffs

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.05,
        spin2x=-0.1,
        spin2y=0.07,
        spin2z=0.2,
        distance=300.0,
        inclination=0.6,
        f_lower=30.0,
        f_ref=50.0,
    )
    v = torch.tensor([0.22, 0.31, 0.39], dtype=torch.float64)
    H = torch.tensor([0.991, 0.976, 0.951], dtype=torch.float64)
    eta = params.eta
    dM = (params.mass1 - params.mass2) / params.M
    chiS = 0.5 * (params.spin1z + params.spin2z)
    chiA = 0.5 * (params.spin1z - params.spin2z)
    a_delta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    omega = v**3
    vh3 = H * omega

    expected = {
        (3, 3): vh3
        * (
            13.0 / 10.0
            + vh3
            * (
                (-81.0 * a_delta) / 20.0 + (39.0 * np.pi) / 7.0
                + vh3
                * (-227827.0 / 3000.0 + (78.0 * np.pi * np.pi) / 7.0)
            )
        )
        + omega * v * v * (-80897.0 * eta / 2430.0),
        (4, 4): vh3
        * (
            (112.0 + 219.0 * eta) / (-120.0 * (-1.0 + 3.0 * eta))
            + vh3
            * (
                (-464.0 * a_delta) / 75.0 + (25136.0 * np.pi) / 3465.0
                + vh3
                * (-55144.0 / 375.0 + 201088.0 * np.pi * np.pi / 10395.0)
            )
        ),
        (5, 5): vh3
        * (
            (96875.0 + 857528.0 * eta) / (131250.0 * (1.0 - 2.0 * eta))
            + vh3
            * (
                3865.0 * np.pi / 429.0
                + vh3
                * ((-7686949127.0 + 954500400.0 * np.pi * np.pi) / 31783752.0)
            )
        ),
    }

    for (ell, emm), expected_delta in expected.items():
        _, _, delta = dyn._factorized_rho_aux_delta(
            ell,
            emm,
            v,
            params,
            waveform=True,
            H=H,
        )
        duplicate_delta = coeffs.delta_lm_full(
            ell,
            emm,
            eta,
            params.spin1z,
            params.spin2z,
            params.mass1,
            params.mass2,
            v,
            H,
            waveform=True,
        )
        torch.testing.assert_close(delta, expected_delta, rtol=2e-15, atol=2e-15)
        torch.testing.assert_close(duplicate_delta, expected_delta, rtol=2e-15, atol=2e-15)


def test_dynamic_spin_projection_tplspin_matches_lal_branches():
    """Main mode generation and 21/55 calibration use different LAL tplspin inputs."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 1),),
    )
    dtype = torch.float64
    Lvec = torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=dtype)
    S1 = torch.tensor([[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]], dtype=dtype)
    S2 = torch.tensor([[-0.2, 0.1, -0.1], [0.1, 0.3, -0.2]], dtype=dtype)

    chi1dotZ, chi2dotZ, chiS, chiA, weighted_tplspin = swt._dynamic_spin_projection_combos(
        Lvec,
        S1,
        S2,
        params,
        weighted_tplspin=True,
    )
    _, _, _, _, unweighted_tplspin = swt._dynamic_spin_projection_combos(
        Lvec,
        S1,
        S2,
        params,
        weighted_tplspin=False,
    )

    Lhat = Lvec / torch.linalg.norm(Lvec, dim=1).unsqueeze(-1)
    torch.testing.assert_close(chi1dotZ, torch.sum(S1 * Lhat, dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(chi2dotZ, torch.sum(S2 * Lhat, dim=1), rtol=1e-12, atol=1e-12)

    total_mass = params.mass1 + params.mass2
    dM = (params.mass1 - params.mass2) / total_mass
    s1dotZ = (params.mass1 / total_mass) ** 2 * chi1dotZ
    s2dotZ = (params.mass2 / total_mass) ** 2 * chi2dotZ
    expected_weighted = (1.0 - 2.0 * params.eta) * 0.5 * (s1dotZ + s2dotZ) + dM * 0.5 * (s1dotZ - s2dotZ)
    expected_unweighted = (1.0 - 2.0 * params.eta) * chiS + dM * chiA

    torch.testing.assert_close(weighted_tplspin, expected_weighted, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(unweighted_tplspin, expected_unweighted, rtol=1e-12, atol=1e-12)


def test_higher_mode_residual_calibration_matches_lal_formula(monkeypatch):
    """21/55 residual calibration should use the uncalibrated residual at attachment."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 1), (5, 5)),
    )
    params.cal21 = 12.0
    params.cal55 = -3.0
    params.nqc_a_map = {(2, 1): {"a1": 1.0}}
    params.nqc_b_map = {(2, 1): {"b1": 1.0}}

    dtype = torch.float64
    t_vec = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
    phi = torch.zeros(3, dtype=dtype)
    omega = torch.full((3,), 8.0 / params.M_sec, dtype=dtype)
    r = torch.full((3,), 10.0, dtype=dtype)
    pr = torch.zeros(3, dtype=dtype)
    Lvec = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype).repeat(3, 1)
    S1 = torch.zeros((3, 3), dtype=dtype)
    S2 = torch.zeros((3, 3), dtype=dtype)

    def fake_modes(*args, **kwargs):
        assert kwargs["weighted_tplspin"] is False
        assert args[7].cal21 == 0.0
        assert args[7].cal55 == 0.0
        return {
            (2, 1): torch.full((3,), 10.0 + 0.0j, dtype=torch.complex128),
            (5, 5): torch.full((3,), 10.0 + 0.0j, dtype=torch.complex128),
        }

    def fake_residual(mode_l, mode_m, *args, **kwargs):
        assert kwargs["cal21"] == 0.0
        assert kwargs["cal55"] == 0.0
        return torch.full((3,), 2.0 + 0.0j, dtype=torch.complex128), torch.zeros(3, dtype=dtype)

    monkeypatch.setattr(swt, "_build_coprecessing_modes", fake_modes)
    monkeypatch.setattr(swt, "_factorized_residual_power", fake_residual)
    monkeypatch.setattr(swt._dyn, "_eob_potentials", lambda *args, **kwargs: {"H": torch.ones(3, dtype=dtype)})
    monkeypatch.setattr(swt, "peak_delta_t_v4", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(swt, "peak_amp_v4", lambda l, m, eta, chiS, chiA: 20.0 if (l, m) == (2, 1) else 30.0)

    cal = swt._higher_mode_residual_calibration(
        t_vec,
        phi,
        omega,
        r,
        pr,
        Lvec,
        S1,
        S2,
        params,
        t_peak_omega_M=1.0,
        chi1L_peak=0.2,
        chi2L_peak=-0.1,
    )

    assert cal["cal21"] == pytest.approx((20.0 / 5.0 - 2.0) / (8.0 ** (7.0 / 3.0)))
    assert cal["cal55"] == pytest.approx((30.0 / 5.0 - 2.0) / (8.0 ** (5.0 / 3.0)))
    assert params.cal21 == 12.0
    assert params.cal55 == -3.0
    assert params.nqc_a_map == {(2, 1): {"a1": 1.0}}
    assert params.nqc_b_map == {(2, 1): {"b1": 1.0}}


def test_populate_nqc_coeffs_loops_requested_positive_modes(monkeypatch):
    """LAL computes NQC coefficients for requested positive modes, not derived -m modes."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2), (3, 3)),
    )
    calls = []

    def fake_solve(mode_l, mode_m, *args, **kwargs):
        calls.append((mode_l, mode_m))
        return {
            "a1": float(mode_l),
            "a2": float(mode_m),
            "a3": 0.0,
            "a4": 0.0,
            "a5": 0.0,
            "b1": 1.0,
            "b2": 0.0,
            "b3": 0.0,
            "b4": 0.0,
        }

    monkeypatch.setattr(swt, "_solve_nqc_coeffs_lal_series", fake_solve)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    modes = {
        (2, 2): torch.ones(4, dtype=torch.complex128),
        (2, -2): torch.ones(4, dtype=torch.complex128),
        (3, 3): torch.ones(4, dtype=torch.complex128),
        (3, -3): torch.ones(4, dtype=torch.complex128),
    }

    swt._populate_nqc_coeffs_lal_v4(
        params,
        params.mode_array,
        modes,
        t,
        torch.ones_like(t),
        torch.zeros_like(t),
        torch.ones_like(t),
        t_peak_omega_M=0.5,
        chi1L_peak=0.2,
        chi2L_peak=-0.1,
    )

    assert calls == [(2, 2), (3, 3)]
    assert set(params.nqc_a_map) == {(2, 2), (3, 3)}
    assert set(params.nqc_b_map) == {(2, 2), (3, 3)}
    assert params.nqc_a == params.nqc_a_map[(2, 2)]
    assert params.nqc_b == params.nqc_b_map[(2, 2)]


def test_polarization_projection_uses_lal_observer_azimuth():
    """SEOBNRv4PHM uses phi_sYlm = pi/2 - phiRef and LAL API polarity."""
    import math

    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    h22 = torch.tensor([1.0 + 0.0j, 0.5 - 0.25j], dtype=torch.complex128)
    inc = 0.7
    phi_ref = 0.3
    hp, hc = swt._polarizations_from_modes(
        {(2, 2): h22},
        inc,
        phi_ref,
        device=device,
        complex_dtype=torch.complex128,
    )
    y_lal = swt._sYlm_torch(
        -2,
        2,
        2,
        torch.tensor(inc, dtype=torch.float64),
        torch.tensor(math.pi / 2.0 - phi_ref, dtype=torch.float64),
    ).to(torch.complex128)
    hpc = h22 * y_lal
    torch.testing.assert_close(hp, -hpc.real, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hc, hpc.imag, rtol=1e-12, atol=1e-12)


def test_ringdown_attachment_snaps_to_lal_sample():
    """RD attachment should use the nearest HiS sample, not exact interpolation."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    dt = 0.5 * params.M_sec
    t_m = torch.arange(80, dtype=dtype) * 0.5
    h22 = (1.0 + 1.0e-3 * t_m) * torch.exp(0.2j * t_m)
    t_attach_m = 10.26
    idx_attach = swt._nearest_index_increasing(t_m, t_attach_m)

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=t_attach_m,
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
    )

    torch.testing.assert_close(out[(2, 2)][idx_attach], h22[idx_attach], rtol=1e-10, atol=1e-12)
    assert len(out[(2, 2)]) > idx_attach


def test_ringdown_attachment_respects_his_start_time():
    """HiS ringdown attachment times are absolute PHM times, not zero-based."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    dt_m = 0.5
    dt = dt_m * params.M_sec
    t_start_m = 100.0
    t_m = t_start_m + torch.arange(80, dtype=dtype) * dt_m
    h22 = (1.0 + 1.0e-3 * t_m) * torch.exp(0.2j * t_m)
    t_attach_m = 110.26
    idx_attach = swt._nearest_index_increasing(t_m, t_attach_m)

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=t_attach_m,
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
        t_start_M=t_start_m,
    )

    torch.testing.assert_close(out[(2, 2)][idx_attach], h22[idx_attach], rtol=1e-10, atol=1e-12)


def test_ringdown_attachment_keeps_lal_patch_length(monkeypatch):
    """LAL keeps the full HiS buffer and appends a longer RD patch."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    monkeypatch.setattr(swt, "_qnm_re_im", lambda *args: (1.0, 10.0))

    dtype = torch.float64
    dt_m = 1.0
    dt = dt_m * params.M_sec
    h22 = (1.0 + 1.0e-3 * torch.arange(20, dtype=dtype)) * torch.exp(
        0.2j * torch.arange(20, dtype=dtype)
    )
    idx_attach = 5
    expected_rd_len = max(1, int(swt._EOB_RD_EFOLDS / (10.0 * dt_m)))
    expected_patch_len = max(
        expected_rd_len,
        math.ceil(swt._EOB_RD_PATCH_EFOLDS / (10.0 * dt_m)),
    )

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=1.0,
        t_attach_M=float(idx_attach),
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
    )[(2, 2)]

    assert len(out) == len(h22) + expected_patch_len
    torch.testing.assert_close(out[:idx_attach], h22[:idx_attach], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        out[idx_attach + expected_rd_len : len(h22)],
        h22[idx_attach + expected_rd_len :],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(out[len(h22) :], torch.zeros_like(out[len(h22) :]), rtol=0.0, atol=0.0)


def test_ringdown_uses_peak_l_frame_spin_projections(monkeypatch):
    """RD coefficient fits use L-frame spins at peak Omega, matching LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.9,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.8,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    seen_chi = []

    def capture(value):
        def _fn(l, m, eta, chi, *, device, dtype):
            seen_chi.append(float(chi))
            return torch.tensor(value, device=device, dtype=dtype)

        return _fn

    monkeypatch.setattr(swt, "_rd_amp_coeff1", capture(0.2))
    monkeypatch.setattr(swt, "_rd_amp_coeff2", capture(0.5))
    monkeypatch.setattr(swt, "_rd_phase_coeff1", capture(0.1))
    monkeypatch.setattr(swt, "_rd_phase_coeff2", capture(1.0))

    dtype = torch.float64
    dt = 0.5 * params.M_sec
    t_m = torch.arange(80, dtype=dtype) * 0.5
    h22 = torch.exp(0.2j * t_m)
    chi1_l = 0.25
    chi2_l = -0.35

    swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=10.0,
        chi1L_attach=chi1_l,
        chi2L_attach=chi2_l,
    )

    d_m = (params.mass1 - params.mass2) / (params.mass1 + params.mass2)
    expected_chi = 0.5 * (chi1_l + chi2_l) + 0.5 * (chi1_l - chi2_l) * d_m / (1.0 - 2.0 * params.eta)
    assert seen_chi
    assert all(abs(chi - expected_chi) < 1e-14 for chi in seen_chi)


def test_non_keplerian_vphi_matches_fd():
    """non_keplerian_vphi should agree with finite-diff dH/dpphi (pr=0)."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)

    r = y0[0]
    Lvec = y0[3:6]
    zero_pr = y0[1] * 0.0

    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(
        r, zero_pr, y0[2], Lvec, y0[6:9], y0[9:12], params
    )
    omega_fd = torch.abs(
        dyn.dphi_dt_fd(
            r,
            zero_pr,
            y0[2],
            Lvec,
            y0[6:9],
            y0[9:12],
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            step=1e-4,
        )
    )
    coeff = 1.0 / (omega_fd * omega_fd * (r ** 3))
    vphi_fd = torch.abs(r * omega_fd * coeff.pow(1.0 / 3.0))

    rel_err = torch.abs(vphi_fd - dyn.non_keplerian_vphi(r, omega_fd, y0[2], Lvec, y0[6:9], y0[9:12], params)) / torch.clamp(vphi_fd, min=1e-12)
    assert rel_err < 1e-4


def test_non_keplerian_vphi_reuses_lal_rdot():
    """Supplying LAL's rdot derivative should not change vPhi."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dyn._refresh_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    r_vec = y_cart[0:3]
    p_vec = y_cart[3:6]
    S1 = y_cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)
    S2 = y_cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)
    r = torch.linalg.norm(r_vec)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    omega = torch.tensor(0.027852467644034994, device=device, dtype=dtype)
    rdot_vec = dyn._calcomega_rdot_lal_fd(r_vec, p_vec, S1, S2, params)

    vphi_direct = dyn.non_keplerian_vphi(
        r, omega, y0[2], L_vec, S1, S2, params, r_vec=r_vec, p_vec=p_vec
    )
    vphi_reused = dyn.non_keplerian_vphi(
        r,
        omega,
        y0[2],
        L_vec,
        S1,
        S2,
        params,
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=rdot_vec,
    )
    torch.testing.assert_close(vphi_reused, vphi_direct, rtol=0.0, atol=0.0)


def test_coprecessing_mode_uses_lal_calcomega_checkpoint():
    """The first-pass mode must use CalcOmega's non-Keplerian coefficient."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    # LALSuite v7.26.1 HiS checkpoint at 144.02 M.  The final two entries
    # are phiDMod and phiMod in LAL's 14-component Cartesian state.
    cart_state = torch.tensor(
        [
            2.4199303397609335,
            1.6711614236958754,
            -0.065210825480322684,
            -0.69446926858996627,
            0.68481375117182297,
            0.012550200016420781,
            -0.008578474902383915,
            -0.009200114156516219,
            0.043318444957915668,
            -0.0012957142107288893,
            0.0037612009862908126,
            -0.012681265409222112,
            33.367542049771906,
            4.935504129291122,
        ],
        dtype=torch.float64,
    )
    state = dyn.cartesian_state_to_reduced_state(cart_state, params).unsqueeze(0)
    omega_dimless = torch.tensor([0.1490381468581127], dtype=torch.float64)
    h22 = swt._build_coprecessing_modes(
        state[:, 2],
        omega_dimless / params.M_sec,
        state[:, 0],
        state[:, 1],
        state[:, 3:6],
        state[:, 6:9],
        state[:, 9:12],
        params,
        ((2, 2),),
        distance_scale=False,
        H=torch.tensor([0.97278008603797828], dtype=torch.float64),
    )[(2, 2)]

    torch.testing.assert_close(
        torch.abs(h22[0]),
        torch.tensor(0.42015328916193062, dtype=torch.float64),
        rtol=5.0e-8,
        atol=0.0,
    )


def test_calcomega_exact_path_batches_like_scalar():
    """CalcOmega's finite-difference port must preserve each batch member."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    state = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    cart = dyn.reduced_state_to_cartesian_state(state, params)
    r_vec = torch.stack((cart[0:3], cart[0:3] * torch.tensor([1.00001, 0.99999, 1.0])))
    p_vec = torch.stack((cart[3:6], cart[3:6] * torch.tensor([0.99999, 1.00001, 1.0])))
    S1 = (cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)).repeat(2, 1)
    S2 = (cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)).repeat(2, 1)

    batched = dyn._calcomega_lal_polar_derivative(r_vec, p_vec, S1, S2, params)
    scalar = torch.stack(
        [
            dyn._calcomega_lal_polar_derivative(
                r_vec[i], p_vec[i], S1[i], S2[i], params
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=0.0, atol=0.0)


def test_calcomega_rdot_refreshes_hcoeffs_per_momentum_perturbation(monkeypatch):
    """LAL's CalcOmega RvecDerivative recomputes hcoeffs inside dH/dP."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dyn._refresh_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    r_vec = y_cart[0:3]
    p_vec = y_cart[3:6]
    S1 = y_cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)
    S2 = y_cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)
    seen = []
    real_eob_potentials = dyn._eob_potentials

    def wrapped_eob_potentials(*args, **kwargs):
        seen.append((args[3].detach().clone(), kwargs.get("hcoeffs_override")))
        return real_eob_potentials(*args, **kwargs)

    monkeypatch.setattr(dyn, "_eob_potentials", wrapped_eob_potentials)
    dyn._calcomega_rdot_lal_fd(r_vec, p_vec, S1, S2, params)

    assert len(seen) >= 13
    assert all(hcoeffs_override is None for _, hcoeffs_override in seen)
    assert any(not torch.equal(L_vec, seen[0][0]) for L_vec, _ in seen[1:])


def test_rhs_flux_recomputes_lal_calcomega_rdot(monkeypatch):
    """The RHS must not pass its 2e-3 integrator dxdt into LAL's 1e-4 CalcOmega path."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    calls = []

    def fake_flux(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(args[7])

    monkeypatch.setattr(dyn, "_factorized_flux", fake_flux)
    dyn.rhs_cartesian_full(torch.tensor(0.0, device=device, dtype=dtype), y_cart, params)

    assert calls
    assert calls[0].get("rdot_vec") is None
    assert calls[0].get("velocity_vec") is not None


def test_dynamics_flux_spin_projection_uses_lal_lhat(monkeypatch):
    """v4P flux coefficients use Lhat, even when an LN velocity is available."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.0,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
        distance=300.0,
        inclination=0.0,
        f_lower=90.0,
        f_ref=90.0,
    )
    dtype = torch.float64
    r_vec = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    p_vec = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    velocity_vec = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    S1 = torch.tensor([0.0, 0.7, 0.2], dtype=dtype)
    S2 = torch.tensor([0.0, -0.3, -0.4], dtype=dtype)
    captured = []

    def fake_rho_aux(_l, _m, v, _params, chi1z=None, chi2z=None, **_kwargs):
        captured.append((float(chi1z), float(chi2z)))
        return torch.ones_like(v), torch.zeros_like(v)

    monkeypatch.setattr(dyn, "_rho_aux_flux", fake_rho_aux)
    dyn._factorized_flux(
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(-1.0e-3, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        L_vec,
        S1,
        S2,
        params,
        torch.tensor(0.03, dtype=dtype),
        torch.tensor(0.99, dtype=dtype),
        r_vec=r_vec,
        p_vec=p_vec,
        velocity_vec=velocity_vec,
    )

    assert captured
    assert captured[0][0] == pytest.approx(0.2, rel=0.0, abs=1.0e-15)
    assert captured[0][1] == pytest.approx(-0.4, rel=0.0, abs=1.0e-15)


def test_ic_radial_momentum_uses_lal_cartesian_flux_probe(monkeypatch):
    """LAL's STEP 4 IC flux probe uses rotated Cartesian dynamics and FD dH/dP."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device = torch.device("cpu")
    dtype = torch.float64
    r = 10.9
    py = 0.35
    pz = -7.0e-5
    omega = torch.tensor(0.0278, device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    L_vec = torch.tensor([0.0, -r * pz, r * py], device=device, dtype=dtype)
    potential_calls = []
    flux_calls = []

    def fake_second_derivative(_params, idx1, idx2, *_args, **_kwargs):
        assert idx1 == 0
        return torch.tensor(2.0 if idx2 == 0 else 3.0, device=device, dtype=dtype)

    def fake_spherical_derivatives(*_args, **_kwargs):
        raise AssertionError("STEP 4 dH/dpphi should come from Cartesian dH/dp_y / r")

    def fake_cartesian_potential(*args, **kwargs):
        potential_calls.append(kwargs)
        dH_dpvec = torch.tensor([0.2, 0.4 * r, 0.0], device=device, dtype=dtype)
        return {"H": torch.tensor(1.0, device=device, dtype=dtype), "deltaT": omega, "D": omega, "dH_dpvec": dH_dpvec}, L_vec

    def fake_flux(*args, **kwargs):
        flux_calls.append(kwargs)
        return torch.tensor(0.01, device=device, dtype=dtype)

    monkeypatch.setattr(dyn, "_ic_spherical_second_derivative", fake_second_derivative)
    monkeypatch.setattr(dyn, "_ic_spherical_derivatives", fake_spherical_derivatives)
    monkeypatch.setattr(dyn, "_ic_cartesian_potential", fake_cartesian_potential)
    monkeypatch.setattr(dyn, "_factorized_flux", fake_flux)

    pr_star = dyn._precessing_ic_radial_momentum(
        params,
        r,
        py,
        pz,
        L_vec,
        S1,
        S2,
        omega,
        final_tortoise=1,
        device=device,
        dtype=dtype,
    )

    assert torch.isfinite(pr_star)
    assert potential_calls[-1]["compute_grad_p"] is False
    assert potential_calls[-1]["fd_dpvec"] is True
    assert flux_calls
    torch.testing.assert_close(flux_calls[0]["r_vec"], torch.tensor([r, 0.0, 0.0], device=device, dtype=dtype))
    torch.testing.assert_close(flux_calls[0]["p_vec"], torch.tensor([0.0, (py * py + pz * pz) ** 0.5, 0.0], device=device, dtype=dtype))


@pytest.mark.parametrize("lm", [(2, 2), (3, 3), (4, 4), (5, 5)])
def test_tail_factor_complex_matches_scipy(lm):
    """Tail factor T_lm should track the LAL GSL branch (log-gamma)."""
    import torch
    from scipy import special

    from pycbc.waveform import seobnrv4phm_torch as swt

    l, m = lm
    device = torch.device("cpu")
    dtype = torch.float64
    omega = torch.tensor([0.0125, 0.045, 0.08], device=device, dtype=dtype)
    H = torch.tensor([1.05, 0.92, 0.88], device=device, dtype=dtype)

    tlm = swt._tail_factor_complex(l, m, omega, H).to(torch.complex128)

    k = m * omega.detach().cpu().double().numpy()
    hathatk = H.detach().cpu().double().numpy() * k
    ln_gamma = special.loggamma(l + 1 - 2j * hathatk)
    ln_gamma_l = special.gammaln(l + 1)
    ref = np.exp(
        ln_gamma
        - ln_gamma_l
        + np.pi * hathatk
        + 2j * hathatk * np.log(4.0 * np.maximum(np.abs(k), 1e-30) / np.sqrt(np.e))
    )
    ref_t = torch.as_tensor(ref, device=device, dtype=torch.complex128)
    torch.testing.assert_close(tlm, ref_t, rtol=1e-11, atol=1e-11)


def test_calibrated_hcoeffs_include_d1v2_and_dheffSSv2():
    """d1v2/dheffSSv2 should respond to augmented spin chi (v4 calibration)."""
    eta = 0.23
    a = 0.6
    h_chi0 = compute_spin_aligned_hcoeffs(eta, a, chi=0.0)
    h_chi1 = compute_spin_aligned_hcoeffs(eta, a, chi=0.5)

    assert h_chi0["d1"] == 0.0
    assert h_chi0["dheffSS"] == 0.0
    assert h_chi0["KK"] != pytest.approx(h_chi1["KK"])
    assert h_chi1["d1v2"] != 0.0
    assert h_chi1["dheffSSv2"] != 0.0
    assert h_chi0["d1v2"] != pytest.approx(h_chi1["d1v2"])
    assert h_chi0["dheffSSv2"] != pytest.approx(h_chi1["dheffSSv2"])


def test_eobparams_initial_hcoeffs_use_lal_full_sigma():
    """The IC-entry hcoeff cache should match LAL's pre-root v4P setup."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )

    total = params.mass1 + params.mass2
    s1 = np.array([params.spin1x, params.spin1y, params.spin1z]) * (params.mass1 / total) ** 2
    s2 = np.array([params.spin2x, params.spin2y, params.spin2z]) * (params.mass2 / total) ** 2
    sigma = s1 + s2
    sigma_norm = float(np.linalg.norm(sigma))
    denom = 1.0 - 2.0 * params.eta
    chi = sigma[2] / denom
    chi += (np.dot(s1[:2], sigma[:2]) + np.dot(s2[:2], sigma[:2])) / sigma_norm / denom / 2.0
    expected = compute_spin_aligned_hcoeffs(params.eta, sigma_norm, chi=chi)

    assert params.a_sigma == pytest.approx(sigma_norm, rel=0, abs=1.0e-15)
    for key, value in expected.items():
        assert params.hcoeffs[key] == pytest.approx(value, rel=0, abs=1.0e-13)


def test_initial_conditions_leave_lal_ic_probe_hcoeffs():
    """The precessing IC path keeps LAL's final derivative hcoeff state."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    hcoeffs_before = dict(params.hcoeffs)

    y0 = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)

    assert y0.numel() == 12
    assert params.hcoeffs.keys() == hcoeffs_before.keys()
    expected_hcoeffs, _, _ = dyn._instantaneous_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    assert any(abs(params.hcoeffs[key] - hcoeffs_before[key]) > 1.0e-14 for key in hcoeffs_before)
    for key, expected in expected_hcoeffs.items():
        expected = float(torch.as_tensor(expected).detach())
        assert params.hcoeffs[key] == pytest.approx(expected, rel=0, abs=1.0e-12)


def test_calibrated_hcoeffs_match_polynomial():
    """Check d1v2/dheffSSv2 against the v4 calibration polynomials (LAL)."""
    eta = 0.1875
    a = 0.55
    chi = 0.37

    coeff_K = (
        1.7336, -1.62045, -1.38086, 1.43659,
        10.2573, 2.26831, 0.0, -0.426958,
        -126.687, 17.3736, 6.16466, 0.0,
        267.788, -27.5201, 31.1746, -59.1658,
    )
    coeff_dSO = (
        -44.5324, 0.0, 0.0, 66.1987,
        0.0, 0.0, -343.313, -568.651,
        0.0, 2495.29, 0.0, 147.481,
        0.0, 0.0, 0.0, 0.0,
    )
    coeff_dSS = (
        6.06807, 0.0, 0.0, 0.0,
        -36.0272, 37.1964, 0.0, -41.0003,
        0.0, 0.0, -326.325, 528.511,
        706.958, 0.0, 1161.78, 0.0,
    )

    chi2 = chi * chi
    chi3 = chi2 * chi
    eta2 = eta * eta
    eta3 = eta2 * eta

    def _poly(c):
        (
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33,
        ) = c
        return (
            c00 + c01 * chi + c02 * chi2 + c03 * chi3
            + c10 * eta + c11 * eta * chi + c12 * eta * chi2 + c13 * eta * chi3
            + c20 * eta2 + c21 * eta2 * chi + c22 * eta2 * chi2 + c23 * eta2 * chi3
            + c30 * eta3 + c31 * eta3 * chi + c32 * eta3 * chi2 + c33 * eta3 * chi3
        )

    expected_KK = _poly(coeff_K)
    m1_plus_eta_KK = -1.0 + eta * expected_KK
    inv_m1_plus_eta_KK = 1.0 / m1_plus_eta_KK
    k0 = expected_KK * (m1_plus_eta_KK - 1.0)
    k1 = -2.0 * (k0 + expected_KK) * m1_plus_eta_KK
    k1p2 = k1 * k1
    k1p3 = k1 * k1p2
    k2 = 0.5 * k1 * (k1 - 4.0 * m1_plus_eta_KK) - a * a * k0 * m1_plus_eta_KK * m1_plus_eta_KK
    k3 = (
        -(k1 * k1) * k1 / 3.0
        + k1 * k2
        + (k1 * k1) * m1_plus_eta_KK
        - 2.0 * (k2 - m1_plus_eta_KK) * m1_plus_eta_KK
        - a * a * k1 * m1_plus_eta_KK * m1_plus_eta_KK
    )
    k4 = (
        (24.0 / 96.0) * (k1 * k1) * (k1 * k1)
        - (96.0 / 96.0) * (k1 * k1) * k2
        + (48.0 / 96.0) * k2 * k2
        - (64.0 / 96.0) * (k1 * k1) * k1 * m1_plus_eta_KK
        + (48.0 / 96.0) * a * a * (k1 * k1 - 2.0 * k2) * m1_plus_eta_KK * m1_plus_eta_KK
        + k1 * (k3 + 2.0 * k2 * m1_plus_eta_KK)
        - m1_plus_eta_KK * (2.0 * k3 + m1_plus_eta_KK * (-(3008.0 / 96.0) + (123.0 / 96.0) * math.pi * math.pi))
    )
    expected_k5 = m1_plus_eta_KK * m1_plus_eta_KK * (
        -4237.0 / 60.0
        + 128.0 / 5.0 * 0.577215664901532860606512090082402431
        + 2275.0 * math.pi * math.pi / 512.0
        - (a * a) * (k1p3 - 3.0 * k1 * k2 + 3.0 * k3) / 3.0
        - (
            k1p3 * k1p2
            - 5.0 * k1p3 * k2
            + 5.0 * k1 * k2 * k2
            + 5.0 * k1p2 * k3
            - 5.0 * k2 * k3
            - 5.0 * k1 * k4
        ) * 0.2 * inv_m1_plus_eta_KK * inv_m1_plus_eta_KK
        + (
            k1p2 * k1p2
            - 4.0 * k1p2 * k2
            + 2.0 * k2 * k2
            + 4.0 * k1 * k3
            - 4.0 * k4
        ) * 0.5 * inv_m1_plus_eta_KK
        + (256.0 / 5.0) * math.log(2.0)
        + (41.0 * math.pi * math.pi / 32.0 - 221.0 / 6.0) * eta
    )
    expected_d1v2 = _poly(coeff_dSO)
    expected_dheffSSv2 = _poly(coeff_dSS)

    h = compute_spin_aligned_hcoeffs(eta, a, chi=chi)
    assert h["KK"] == pytest.approx(expected_KK, rel=0, abs=1e-12)
    assert h["k5"] == pytest.approx(expected_k5, rel=0, abs=1e-12)
    assert h["d1v2"] == pytest.approx(expected_d1v2, rel=0, abs=1e-12)
    assert h["dheffSSv2"] == pytest.approx(expected_dheffSSv2, rel=0, abs=1e-12)


def test_augmented_spin_mapping_matches_lal_formula():
    """Augmented spin used in RHS should follow the LAL v4P mapping."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    r = torch.tensor(11.0, device=device, dtype=dtype)
    pr = torch.tensor(0.015, device=device, dtype=dtype)
    phi = torch.tensor(0.42, device=device, dtype=dtype)
    L_vec = torch.tensor([2.1, -1.4, 4.6], device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    y = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])

    # Trigger RHS to refresh hcoeffs (uses augmented chi internally)
    dyn.rhs(torch.tensor(0.0, device=device, dtype=dtype), y, params)

    M = params.mass1 + params.mass2
    eta = params.mass1 * params.mass2 / (M * M)
    s1_m2 = dyn._lal_spin_weight(S1, params.mass1, M)
    s2_m2 = dyn._lal_spin_weight(S2, params.mass2, M)
    sigma = s1_m2 + s2_m2
    sigma_norm = torch.linalg.norm(sigma)
    Lhat = L_vec / torch.linalg.norm(L_vec)

    S1_perp = s1_m2 - torch.dot(s1_m2, Lhat) * Lhat
    S2_perp = s2_m2 - torch.dot(s2_m2, Lhat) * Lhat
    denom = 1.0 - 2.0 * eta
    chi_aug = torch.dot(sigma, Lhat) / denom
    chi_aug = chi_aug + (torch.dot(S1_perp, sigma) + torch.dot(S2_perp, sigma)) / (sigma_norm * denom * 2.0)

    a_sigma = float(sigma_norm)
    h_expected = compute_spin_aligned_hcoeffs(eta, a_sigma, chi=float(chi_aug))

    assert params.hcoeffs["d1v2"] == pytest.approx(h_expected["d1v2"], rel=0, abs=1e-12)
    assert params.hcoeffs["dheffSSv2"] == pytest.approx(h_expected["dheffSSv2"], rel=0, abs=1e-12)


def test_vectorized_hamiltonian_refreshes_hcoeffs_per_sample():
    """Vector mode-building H should use LAL's per-sample hcoeff refresh."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    dtype = torch.float64
    r = torch.tensor([11.0, 8.7], dtype=dtype)
    pr = torch.tensor([0.015, -0.002], dtype=dtype)
    phi = torch.tensor([0.42, 1.1], dtype=dtype)
    L_vec = torch.tensor([[2.1, -1.4, 4.6], [-0.6, 0.8, 3.1]], dtype=dtype)
    S1 = torch.tensor([[0.18, -0.12, 0.27], [0.23, 0.04, -0.18]], dtype=dtype)
    S2 = torch.tensor([[-0.08, 0.11, 0.19], [-0.16, 0.07, 0.12]], dtype=dtype)

    coeffs, _, _ = dyn._instantaneous_hcoeffs(params, L_vec, S1, S2)
    assert abs(float(coeffs["KK"][0] - coeffs["KK"][1])) > 1.0e-8

    pot_vec = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        compute_grad_p=False,
        compute_base_grad=False,
        fd_pphi=False,
    )
    h_loop = []
    for i in range(2):
        pot_i = dyn._eob_potentials(
            r[i],
            pr[i],
            phi[i],
            L_vec[i],
            S1[i],
            S2[i],
            params,
            compute_grad_p=False,
            compute_base_grad=False,
            fd_pphi=False,
        )
        h_loop.append(pot_i["H"])
    torch.testing.assert_close(pot_vec["H"], torch.stack(h_loop), rtol=1e-12, atol=1e-12)


def test_vectorized_fd_dpvec_matches_lal_axis_layout():
    """Vectorized numerical dH/dP should return per-sample Cartesian rows."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    dtype = torch.float64
    r = torch.tensor([11.0, 8.7], dtype=dtype)
    pr = torch.tensor([0.015, -0.002], dtype=dtype)
    phi = torch.tensor([0.42, 1.1], dtype=dtype)
    L_vec = torch.tensor([[2.1, -1.4, 4.6], [-0.6, 0.8, 3.1]], dtype=dtype)
    S1 = torch.tensor([[0.18, -0.12, 0.27], [0.23, 0.04, -0.18]], dtype=dtype)
    S2 = torch.tensor([[-0.08, 0.11, 0.19], [-0.16, 0.07, 0.12]], dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)

    pot_vec = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=False,
        compute_base_grad=False,
        fd_dpvec=True,
        fd_pphi=False,
    )
    assert pot_vec["dH_dpvec"].shape == p_vec.shape

    dH_loop = []
    for i in range(2):
        pot_i = dyn._eob_potentials(
            r[i],
            pr[i],
            phi[i],
            L_vec[i],
            S1[i],
            S2[i],
            params,
            p_vec=p_vec[i],
            r_vec=r_vec[i],
            compute_grad_p=False,
            compute_base_grad=False,
            fd_dpvec=True,
            fd_pphi=False,
        )
        dH_loop.append(pot_i["dH_dpvec"])
    torch.testing.assert_close(pot_vec["dH_dpvec"], torch.stack(dH_loop), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tortoise_pdot", ["0", "1"])
def test_cartesian_projected_rhs_reconstructs_reduced_state(monkeypatch, tortoise_pdot):
    """Projected Cartesian RHS should preserve the reduced state's x cross P map."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", tortoise_pdot)

    params = dyn.EOBParams(
        mass1=28.0,
        mass2=17.0,
        spin1x=0.16,
        spin1y=-0.08,
        spin1z=0.22,
        spin2x=-0.12,
        spin2y=0.05,
        spin2z=0.14,
        distance=500.0,
        inclination=0.5,
        f_lower=25.0,
        f_ref=35.0,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    r = torch.tensor(9.5, device=device, dtype=dtype)
    pr = torch.tensor(-3.0e-4, device=device, dtype=dtype)
    phi = torch.tensor(0.73, device=device, dtype=dtype)
    L_vec = torch.tensor([0.36, -0.18, 3.2], device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    y = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])

    r_vec, p_vec, n_hat, _, _ = dyn._reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)
    torch.testing.assert_close(torch.linalg.cross(r_vec, p_vec), L_vec, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.dot(p_vec, n_hat), pr, rtol=1e-12, atol=1e-12)

    pot = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=True,
        compute_grad_spin=True,
        fd_dpvec=False,
        fd_pphi=False,
    )
    assert torch.isfinite(pot["dH_dS1"]).all()
    assert torch.isfinite(pot["dH_dS2"]).all()

    dy = dyn.rhs_cartesian_projected(torch.tensor(0.0, device=device, dtype=dtype), y, params)
    assert torch.isfinite(dy).all()
    assert abs(float(dy[2])) > 0.0
    torch.testing.assert_close(torch.dot(dy[6:9], S1), torch.zeros((), device=device, dtype=dtype), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.dot(dy[9:12], S2), torch.zeros((), device=device, dtype=dtype), rtol=0.0, atol=1e-12)


def test_full_cartesian_rhs_roundtrips_reduced_state(monkeypatch):
    """Internal Cartesian trajectory state should project back to reduced data."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", "1")
    params = dyn.EOBParams(
        mass1=26.0,
        mass2=19.0,
        spin1x=0.11,
        spin1y=-0.09,
        spin1z=0.18,
        spin2x=-0.07,
        spin2y=0.03,
        spin2z=0.16,
        distance=450.0,
        inclination=0.4,
        f_lower=25.0,
        f_ref=35.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [9.8, -2.0e-4, 0.4, 0.22, -0.31, 3.35, params.spin1x, params.spin1y, params.spin1z, params.spin2x, params.spin2y, params.spin2z],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)
    roundtrip = dyn.cartesian_state_to_reduced_state(cart, params)
    torch.testing.assert_close(roundtrip, reduced, rtol=1e-11, atol=1e-11)

    dy = dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=dtype), cart, params)
    assert torch.isfinite(dy).all()
    assert abs(float(dy[6])) > 0.0
    torch.testing.assert_close(torch.dot(dy[6:9], cart[6:9]), torch.zeros((), dtype=dtype), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.dot(dy[9:12], cart[9:12]), torch.zeros((), dtype=dtype), rtol=0.0, atol=1e-12)


def test_waveform_state_retains_integrated_cartesian_geometry(monkeypatch):
    """Mode and omega builders must receive integrated x/P without reconstruction."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=26.0,
        mass2=19.0,
        spin1x=0.11,
        spin1y=-0.09,
        spin1z=0.18,
        spin2x=-0.07,
        spin2y=0.03,
        spin2z=0.16,
        distance=450.0,
        inclination=0.4,
        f_lower=25.0,
        f_ref=35.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [
            9.8,
            -2.0e-4,
            0.4,
            0.22,
            -0.31,
            3.35,
            params.spin1x,
            params.spin1y,
            params.spin1z,
            params.spin2x,
            params.spin2y,
            params.spin2z,
        ],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)
    # The evolved phase variables are not the inertial-plane azimuth of x.
    cart[12] = 1.1
    cart[13] = -0.2
    state = swt._waveform_state_from_cartesian(cart, params).unsqueeze(0)
    assert state.shape == (1, 18)
    torch.testing.assert_close(state[0, 12:18], cart[0:6], rtol=0.0, atol=0.0)

    captured = {}

    def fake_omega(r, pr, phi, Lvec, S1vec, S2vec, params_in, **kwargs):
        assert params_in is params
        captured["omega_r_vec"] = kwargs["r_vec"]
        captured["omega_p_vec"] = kwargs["p_vec"]
        return torch.full_like(r, 0.03)

    monkeypatch.setattr(swt, "_omega_from_hamiltonian_velocity", fake_omega)
    phi, r, pr, Lvec, S1, S2, omega = swt._series_from_states(
        torch.zeros(1, dtype=dtype), state, params
    )
    r_vec, p_vec = cart[0:3].unsqueeze(0), cart[3:6].unsqueeze(0)
    expected_r = torch.linalg.norm(r_vec, dim=-1)
    expected_pr = torch.sum(p_vec * r_vec, dim=-1) / expected_r
    expected_L = torch.cross(r_vec, p_vec, dim=-1)
    torch.testing.assert_close(r, expected_r, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(pr, expected_pr, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(Lvec, expected_L, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(captured["omega_r_vec"], r_vec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(captured["omega_p_vec"], p_vec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(omega, torch.full_like(r, 0.03 / params.M_sec))

    def fake_potentials(*args, **kwargs):
        captured["mode_r_vec"] = kwargs["r_vec"]
        captured["mode_p_vec"] = kwargs["p_vec"]
        return {"H": torch.ones_like(r)}

    def fake_vphi(*args, **kwargs):
        captured["vphi_r_vec"] = kwargs["r_vec"]
        captured["vphi_p_vec"] = kwargs["p_vec"]
        return torch.ones_like(r)

    monkeypatch.setattr(dyn, "_eob_potentials", fake_potentials)
    monkeypatch.setattr(dyn, "non_keplerian_vphi", fake_vphi)
    modes = swt._build_coprecessing_modes(
        phi,
        omega,
        r,
        pr,
        Lvec,
        S1,
        S2,
        params,
        (),
        r_vec=r_vec,
        p_vec=p_vec,
    )
    assert modes == {}
    for name in ("mode_r_vec", "vphi_r_vec"):
        torch.testing.assert_close(captured[name], r_vec, rtol=0.0, atol=0.0)
    for name in ("mode_p_vec", "vphi_p_vec"):
        torch.testing.assert_close(captured[name], p_vec, rtol=0.0, atol=0.0)


def test_full_cartesian_analytic_spin_torque_matches_weighted_fd(monkeypatch):
    """Analytic chi gradients must produce rates for LAL's weighted spins."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    for name in (
        "PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE",
        "PYCBC_SEOBNRV4PHM_FD_SPIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_HAMILTONIAN_SPIN", "1")

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.21,
        spin1y=-0.13,
        spin1z=0.31,
        spin2x=-0.17,
        spin2y=0.09,
        spin2z=0.12,
        distance=450.0,
        inclination=0.7,
        f_lower=25.0,
        f_ref=25.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [
            8.7,
            -3.0e-4,
            0.63,
            0.24,
            -0.19,
            3.15,
            params.spin1x,
            params.spin1y,
            params.spin1z,
            params.spin2x,
            params.spin2y,
            params.spin2z,
        ],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)

    def spin_rates():
        L_vec = torch.linalg.cross(cart[0:3], cart[3:6])
        dyn._refresh_hcoeffs(params, L_vec, reduced[6:9], reduced[9:12])
        return dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=dtype), cart, params)[6:12]

    analytic = spin_rates()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_SPIN", "1")
    finite_difference = spin_rates()

    torch.testing.assert_close(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-10)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1=(0.1, -0.05, 0.2),
            spin2=(-0.02, 0.04, 0.15),
            f_min=20.0,
            inc=0.4,
        ),
        dict(
            mass1=22.0,
            mass2=16.0,
            spin1=(0.25, 0.12, -0.18),
            spin2=(-0.21, 0.06, 0.14),
            f_min=25.0,
            inc=0.7,
        ),
    ],
)
def test_hamiltonian_rhs_matches_lal_dynamics(cfg, monkeypatch):
    """SEOBNRv4PHM torch Hamiltonian/RHS should track LAL dynamics snapshots."""
    try:
        import lal
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    import numpy as np
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_DP", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_SPIN", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_X", "1")

    # Minimal mode set to keep the LAL call light
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 1)
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 3, 3)
    seobflags = lal.CreateDict()

    phi_c = 0.0
    delta_t = 1.0 / 4096.0
    distance_si = 500.0 * lal.PC_SI
    outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
        phi_c,
        delta_t,
        cfg["mass1"] * lal.MSUN_SI,
        cfg["mass2"] * lal.MSUN_SI,
        cfg["f_min"],
        distance_si,
        cfg["inc"],
        *cfg["spin1"],
        *cfg["spin2"],
        mode_array,
        seobflags,
    )

    # seobdynamicsAdaSHiSVector is index 6 (flattened shape [26 * N])
    seob_vec = outs[6]
    data = np.array(seob_vec.data, copy=False)
    dyn_vars = 26
    ret_len = seob_vec.length // dyn_vars
    snap_indices = sorted(
        set(
            [
                min(40, ret_len - 1),
                min(80, ret_len - 1),
                min(140, ret_len - 1),
            ]
        )
    )

    def col(k, idx):
        return data[k * ret_len + idx]

    for idx in snap_indices:
        r = torch.tensor(col(18, idx), dtype=torch.float64)
        pr_val = torch.tensor(col(20, idx), dtype=torch.float64)

        pos = torch.tensor([col(1, idx), col(2, idx), col(3, idx)], dtype=torch.float64)
        mom = torch.tensor([col(4, idx), col(5, idx), col(6, idx)], dtype=torch.float64)
        L_vec = torch.linalg.cross(pos, mom)

        S1_lal = torch.tensor([col(7, idx), col(8, idx), col(9, idx)], dtype=torch.float64)
        S2_lal = torch.tensor([col(10, idx), col(11, idx), col(12, idx)], dtype=torch.float64)
        S1, S2 = _lal_weighted_spins_to_chi(S1_lal, S2_lal, cfg["mass1"], cfg["mass2"])
        phi = torch.tensor(col(19, idx), dtype=torch.float64)

        params = dyn.EOBParams(
            mass1=cfg["mass1"],
            mass2=cfg["mass2"],
            spin1x=float(S1[0]),
            spin1y=float(S1[1]),
            spin1z=float(S1[2]),
            spin2x=float(S2[0]),
            spin2y=float(S2[1]),
            spin2z=float(S2[2]),
            distance=distance_si / lal.PC_SI,
            inclination=cfg["inc"],
            f_lower=cfg["f_min"],
            f_ref=cfg["f_min"],
        )
        params.tortoise = 1  # match LAL precessing dynamical evolution

        dyn._refresh_hcoeffs(params, L_vec, S1, S2)
        pr = pr_val
        pot = dyn._eob_potentials(r, pr, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos)

        H_lal = torch.tensor(col(25, idx), dtype=torch.float64)
        torch.testing.assert_close(pot["H"], H_lal, rtol=1e-12, atol=1e-12)

        # Derivative parity: omega = dphi/dt, rdot from velocity projection
        y_state = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])
        rhs_out = dyn.rhs_cartesian_projected(torch.tensor(0.0, dtype=torch.float64), y_state, params)

        omega_lal = torch.tensor(col(22, idx), dtype=torch.float64)
        torch.testing.assert_close(rhs_out[2], omega_lal, rtol=2e-5, atol=1e-8)

        v_vec = torch.tensor([col(15, idx), col(16, idx), col(17, idx)], dtype=torch.float64)
        rhat = pos / torch.linalg.norm(pos)
        rdot_lal = torch.dot(rhat, v_vec)
        torch.testing.assert_close(rhs_out[0], rdot_lal, rtol=1e-4, atol=5e-8)

        nhat_dot_dH = torch.dot(pot["dH_dpvec"], pot["n_hat"])
        v_torch = pot["dH_dpvec"] + (pot["csi"] - 1.0) * nhat_dot_dH * pot["n_hat"]
        torch.testing.assert_close(v_torch, v_vec, rtol=1e-4, atol=2e-6)

        if 0 < idx < ret_len - 1:
            dt_lal = torch.tensor(col(0, idx + 1) - col(0, idx - 1), dtype=torch.float64)
            S1_prev = torch.tensor([col(7, idx - 1), col(8, idx - 1), col(9, idx - 1)], dtype=torch.float64)
            S2_prev = torch.tensor([col(10, idx - 1), col(11, idx - 1), col(12, idx - 1)], dtype=torch.float64)
            S1_next = torch.tensor([col(7, idx + 1), col(8, idx + 1), col(9, idx + 1)], dtype=torch.float64)
            S2_next = torch.tensor([col(10, idx + 1), col(11, idx + 1), col(12, idx + 1)], dtype=torch.float64)
            S1_prev, S2_prev = _lal_weighted_spins_to_chi(S1_prev, S2_prev, cfg["mass1"], cfg["mass2"])
            S1_next, S2_next = _lal_weighted_spins_to_chi(S1_next, S2_next, cfg["mass1"], cfg["mass2"])
            torch.testing.assert_close(rhs_out[6:9], (S1_next - S1_prev) / dt_lal, rtol=2e-1, atol=5e-6)
            torch.testing.assert_close(rhs_out[9:12], (S2_next - S2_prev) / dt_lal, rtol=3e-1, atol=1e-5)

        dphi_fd = dyn.dphi_dt_fd(
            r, pr, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos, step=1e-4
        )
        torch.testing.assert_close(pot["dH_dpf"], dphi_fd, rtol=1e-6, atol=1e-8)

        y_full = torch.cat(
            [
                pos,
                mom,
                S1_lal,
                S2_lal,
                torch.tensor([col(13, idx), col(14, idx)], dtype=torch.float64),
            ]
        )
        rhs_full = dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=torch.float64), y_full, params)
        torch.testing.assert_close(rhs_full[:3], v_vec, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(rhs_full[12] + rhs_full[13], omega_lal, rtol=2e-5, atol=1e-8)


def test_first_clean_lal_rk_step_matches_lal_dynamics(monkeypatch):
    """With LAL's first row and step, torch RK/RHS should reproduce row two."""
    import torch

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        f_lower=90.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        mode_array=((2, 2), (2, 1)),
    )
    rows = _lal_phm_dynamics_rows(params_dict, 2, vector="adas")
    y_torch, y1_lal = _torch_replay_lal_rk_interval(params_dict, rows, 0)

    torch.testing.assert_close(y_torch[:12], y1_lal[:12], rtol=0.0, atol=5.0e-11)
    torch.testing.assert_close(y_torch[12:], y1_lal[12:], rtol=0.0, atol=1.2e-9)


def test_clean_lal_adas_rk_intervals_match_lal_dynamics(monkeypatch):
    """Replay representative raw AdaS LAL intervals with torch RK/RHS."""
    import torch

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        f_lower=90.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        mode_array=((2, 2), (2, 1)),
    )
    rows = _lal_phm_dynamics_rows(params_dict, 102, vector="adas")

    for idx in (1, 2, 10, 50, 100):
        y_torch, y_lal = _torch_replay_lal_rk_interval(params_dict, rows, idx)
        torch.testing.assert_close(y_torch[:12], y_lal[:12], rtol=0.0, atol=2.0e-10)
        torch.testing.assert_close(y_torch[12:], y_lal[12:], rtol=0.0, atol=6.0e-9)


def test_dphi_dt_fd_matches_gsl_snapshot():
    """Finite-diff dphi/dt (torch) should match LAL GSL pphi derivative on a snapshot."""
    try:
        import lal
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    cfg = dict(
        mass1=30.0,
        mass2=20.0,
        spin1=(0.1, -0.05, 0.2),
        spin2=(-0.02, 0.04, 0.15),
        f_min=20.0,
        inc=0.4,
    )

    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    seobflags = lal.CreateDict()

    phi_c = 0.0
    delta_t = 1.0 / 4096.0
    distance_si = 500.0 * lal.PC_SI
    outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
        phi_c,
        delta_t,
        cfg["mass1"] * lal.MSUN_SI,
        cfg["mass2"] * lal.MSUN_SI,
        cfg["f_min"],
        distance_si,
        cfg["inc"],
        *cfg["spin1"],
        *cfg["spin2"],
        mode_array,
        seobflags,
    )

    seob_vec = outs[6]
    data = np.array(seob_vec.data, copy=False)
    dyn_vars = 26
    ret_len = seob_vec.length // dyn_vars
    idx = min(80, ret_len - 1)
    def col(k):
        return data[k * ret_len + idx]

    r = torch.tensor(col(18), dtype=torch.float64)
    pr_val = torch.tensor(col(20), dtype=torch.float64)
    pos = torch.tensor([col(1), col(2), col(3)], dtype=torch.float64)
    mom = torch.tensor([col(4), col(5), col(6)], dtype=torch.float64)
    L_vec = torch.linalg.cross(pos, mom)

    S1_lal = torch.tensor([col(7), col(8), col(9)], dtype=torch.float64)
    S2_lal = torch.tensor([col(10), col(11), col(12)], dtype=torch.float64)
    S1, S2 = _lal_weighted_spins_to_chi(S1_lal, S2_lal, cfg["mass1"], cfg["mass2"])
    phi = torch.tensor(col(19), dtype=torch.float64)

    params = dyn.EOBParams(
        mass1=cfg["mass1"],
        mass2=cfg["mass2"],
        spin1x=float(S1[0]),
        spin1y=float(S1[1]),
        spin1z=float(S1[2]),
        spin2x=float(S2[0]),
        spin2y=float(S2[1]),
        spin2z=float(S2[2]),
        distance=distance_si / lal.PC_SI,
        inclination=cfg["inc"],
        f_lower=cfg["f_min"],
        f_ref=cfg["f_min"],
    )
    params.tortoise = 1

    pot = dyn._eob_potentials(r, pr_val, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos)
    dphi_fd = dyn.dphi_dt_fd(
        r, pr_val, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos, step=1e-4
    )
    omega_lal = torch.tensor(col(22), dtype=torch.float64)

    torch.testing.assert_close(dphi_fd, omega_lal, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(pot["dH_dpf"], dphi_fd, rtol=1e-6, atol=1e-8)


def test_rhs_derivative_options_enable_lal_numerical_bundle(monkeypatch):
    """One parity switch should enable LAL's numerical derivative choices."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    for name in (
        "PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE",
        "PYCBC_SEOBNRV4PHM_FD_DP",
        "PYCBC_SEOBNRV4PHM_FD_SPIN",
        "PYCBC_SEOBNRV4PHM_FD_X",
        "PYCBC_SEOBNRV4PHM_HAMILTONIAN_SPIN",
        "PYCBC_SEOBNRV4PHM_TORTOISE_PDOT",
        "PYCBC_SEOBNRV4PHM_X_GRAD",
    ):
        monkeypatch.delenv(name, raising=False)

    opts = dyn._rhs_derivative_options()
    assert opts["fd_dpvec"] is False
    assert opts["use_fd_spin"] is False
    assert opts["use_fd_x"] is False

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    opts = dyn._rhs_derivative_options()
    assert opts["fd_dpvec"] is True
    assert opts["use_fd_spin"] is True
    assert opts["use_fd_x"] is True
    assert opts["use_hamiltonian_spin"] is True
    assert opts["use_tortoise_pdot"] is True

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_X", "0")
    assert dyn._rhs_derivative_options()["use_fd_x"] is False


def test_rkf45_step_nonzero_error_and_accuracy():
    """rk45_step should produce non-zero error and accurate growth for y'=y."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    def f(t, y):
        return y

    state = ode.RKState(torch.tensor(0.0), torch.tensor([1.0], dtype=torch.float64), torch.tensor(0.5))
    new_state, err = ode.rk45_step(f, state)

    assert torch.all(err > 0)
    expected = torch.exp(torch.tensor(0.5, dtype=new_state.y.dtype, device=new_state.y.device))
    torch.testing.assert_close(new_state.y[0], expected, rtol=1e-3, atol=1e-3)


def test_integrate_simple_exp():
    """integrate should track y'=y over [0,1] within tolerance."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    def f(t, y):
        return y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(f, y0, t0=0.0, t1=1.0, h0=0.25, rtol=1e-7, atol=1e-10)

    assert len(traj) > 2  # multiple steps taken
    t_end, y_end = traj[-1]
    torch.testing.assert_close(t_end, torch.tensor(1.0, dtype=t_end.dtype, device=t_end.device), atol=1e-6, rtol=0.0)
    expected = torch.exp(torch.tensor(1.0, dtype=y_end.dtype, device=y_end.device))
    torch.testing.assert_close(y_end[0], expected, rtol=2e-3, atol=2e-3)


def test_integrate_reuses_cached_k1_after_rejection():
    """LAL's RK driver does not recompute dydt_in when retrying a rejected step."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    calls = []

    def f(t, y):
        calls.append(float(t))
        return y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(
        f,
        y0,
        t0=0.0,
        t1=1.0,
        h0=1.0,
        rtol=1e-14,
        atol=1e-14,
        stop_fn=lambda t, y: True,
    )

    assert len(traj) == 1
    assert sum(abs(t) < 1.0e-15 for t in calls) == 1


def test_integrate_accepts_lal_min_step_clamp():
    """NoInterpolate accepts when LAL's min-step clamp prevents a decrease."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=1e-14,
        atol=1e-14,
        max_steps=3,
        stop_fn=lambda t, y: True,
        h_min=1.0,
    )

    assert len(traj) == 1
    assert float(traj[0][0]) == 1.0


def test_integrate_debug_progress_label_and_max_steps(monkeypatch, capsys):
    """Debug output should identify bounded native integration stages."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_DEBUG", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROGRESS_INTERVAL", "0")

    with pytest.warns(UserWarning, match="max_steps"):
        ode.integrate(
            lambda t, y: y,
            torch.tensor([1.0], dtype=torch.float64),
            t0=0.0,
            t1=10.0,
            h0=0.1,
            rtol=1e-7,
            atol=1e-10,
            max_steps=1,
            h_min=0.0,
            progress_label="unit",
        )

    out = capsys.readouterr().out
    assert "ODE unit accept step=0" in out


def test_integrate_can_return_step_diagnostics():
    """Diagnostics expose accepted/rejected attempts for capped runs."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    with pytest.warns(UserWarning, match="max_steps"):
        traj, info = ode.integrate(
            lambda t, y: y,
            torch.tensor([1.0], dtype=torch.float64),
            t0=0.0,
            t1=10.0,
            h0=0.1,
            rtol=1e-7,
            atol=1e-10,
            max_steps=1,
            h_min=0.0,
            return_diagnostics=True,
        )

    assert len(traj) == 1
    assert info["accepted_steps"] == 1
    assert info["rejected_steps"] == 0
    assert info["attempted_steps"] == 1
    assert info["max_steps"] == 1
    assert info["hit_max_steps"] is True
    assert info["t_end"] == pytest.approx(0.1)


def test_integrate_can_trace_lal_hadjust_attempts(monkeypatch):
    """Optional diagnostics should expose accepted/rejected h-adjust decisions."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TRACE_STEPS", "2")
    y0 = torch.tensor([1.0], dtype=torch.float64)
    calls = []

    def fake_step(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([2.0 if len(calls) == 1 else 0.25], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step)
    traj, info = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: True,
        h_min=0.0,
        return_diagnostics=True,
    )

    expected_shrink = 0.9 * 2.0 ** (-1.0 / 5.0)
    expected_grow = expected_shrink * 0.9 * 0.25 ** (-1.0 / 6.0)
    assert float(traj[0][0]) == pytest.approx(expected_shrink)
    assert info["step_trace"] == [
        {
            "step": 0,
            "action": "reject_error",
            "t": 0.0,
            "h": 1.0,
            "err_norm": 2.0,
            "err_argmax": 0,
            "err_component": 2.0,
            "scale_component": 1.0,
            "h_next": pytest.approx(expected_shrink),
        },
        {
            "step": 1,
            "action": "accept",
            "t": 0.0,
            "h": pytest.approx(expected_shrink),
            "err_norm": 0.25,
            "err_argmax": 0,
            "err_component": 0.25,
            "scale_component": 1.0,
            "h_next": pytest.approx(expected_grow),
        },
    ]


def test_integrate_uses_gsl_rkf45_control_order(monkeypatch):
    """Old gsl_odeiv reports RKF45 step order 5 to the controller."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    y0 = torch.tensor([1.0], dtype=torch.float64)

    calls = []

    def fake_step_shrink(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([2.0 if len(calls) == 1 else 0.4], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step_shrink)
    traj = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: True,
        h_min=0.0,
    )

    expected_shrink = 0.9 * 2.0 ** (-1.0 / 5.0)
    assert calls[1] == pytest.approx(expected_shrink)
    assert float(traj[0][0]) == pytest.approx(expected_shrink)

    calls = []

    def fake_step_grow(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([0.25 if len(calls) == 1 else 1.0], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step_grow)
    ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: len(calls) >= 2,
        h_min=0.0,
    )

    expected_grow = 0.9 * 0.25 ** (-1.0 / 6.0)
    assert calls[1] == pytest.approx(expected_grow)


def test_final_mass_spin_uses_adas_r10m_lframe_state():
    """Final mass/spin fit should use LAL's AdaS sample nearest r=10M."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.3,
        spin2x=-0.1,
        spin2y=0.05,
        spin2z=0.2,
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )
    t_adas = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    y_adas = torch.zeros((4, 12), dtype=dtype)
    y_adas[:, 0] = torch.tensor([13.0, 11.0, 9.8, 8.0], dtype=dtype)
    y_adas[:, 1] = -1.0e-3
    y_adas[:, 2] = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=dtype)
    y_adas[:, 3:6] = torch.tensor([0.0, 0.0, 4.0], dtype=dtype)
    y_adas[:, 6:9] = torch.tensor([0.2, 0.1, 0.3], dtype=dtype)
    y_adas[:, 9:12] = torch.tensor([-0.1, 0.05, 0.2], dtype=dtype)

    (mf, af), time_10M, s1_l, s2_l = swt._final_mass_spin_from_adas_10M(t_adas, y_adas, params)

    assert time_10M == pytest.approx(2.0)
    expected_s1, expected_s2 = swt._l_frame_spin_vectors_from_state(y_adas[2], params)
    expected_mf, expected_af = swt._final_mass_spin_prec(
        params.mass1,
        params.mass2,
        tuple(float(x) for x in expected_s1),
        tuple(float(x) for x in expected_s2),
    )
    torch.testing.assert_close(s1_l, expected_s1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(s2_l, expected_s2, rtol=1e-12, atol=1e-12)
    assert mf == pytest.approx(expected_mf)
    assert af == pytest.approx(expected_af)


def test_final_spin_sign_and_clamp_match_lal():
    """LAL signs finalSpin with Lhat.J and clamps QNM interpolation input."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    assert swt._signed_clamped_final_spin(0.4, 0.1) == pytest.approx(0.4)
    assert swt._signed_clamped_final_spin(0.4, -0.1) == pytest.approx(-0.4)
    assert swt._signed_clamped_final_spin(1.2, 0.1) == pytest.approx(0.9996)
    assert swt._signed_clamped_final_spin(1.2, -0.1) == pytest.approx(-0.9996)


def test_adaptive_his_join_excludes_adas_join_sample():
    """AdaS/HiS join should keep AdaS before tstartHiS and HiS from tstartHiS."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    adas = [(torch.tensor(float(i), dtype=dtype), torch.tensor([float(i)], dtype=dtype)) for i in range(5)]
    his = [
        (torch.tensor(2.0, dtype=dtype), torch.tensor([20.0], dtype=dtype)),
        (torch.tensor(2.5, dtype=dtype), torch.tensor([25.0], dtype=dtype)),
        (torch.tensor(3.0, dtype=dtype), torch.tensor([30.0], dtype=dtype)),
    ]

    assert swt._last_traj_index_leq(adas, 2.6) == 2
    joined = swt._join_adaptive_his_trajectories(adas, his, 2)

    times = [float(t) for t, _ in joined]
    values = [float(y[0]) for _, y in joined]
    assert times == [0.0, 1.0, 2.0, 2.5, 3.0]
    assert values == [0.0, 1.0, 20.0, 25.0, 30.0]


def test_uniform_joined_dynamics_excludes_attachment_index():
    """SEOBJoinDynamics excludes the last sample <= tAttach from dynamics."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_adas = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    y_adas = torch.arange(4, dtype=dtype).reshape(4, 1)
    t_his = torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    y_his = (10.0 + torch.arange(5, dtype=dtype)).reshape(5, 1)

    t_join, y_join, index_attach = swt._join_uniform_dynamics_until_attach(
        t_adas,
        y_adas,
        t_his,
        y_his,
        index_start_his=2,
        t_attach_M=3.5,
    )

    assert index_attach == 5
    assert t_join.tolist() == [0.0, 1.0, 2.0, 2.5, 3.0]
    assert y_join[:, 0].tolist() == [0.0, 1.0, 10.0, 11.0, 12.0]


def test_phm_boundary_summary_from_segments_uses_supplied_attachment():
    """Boundary summaries should avoid merger timing when tAttach is supplied."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_adas = torch.tensor([0.0, 0.01, 0.02, 0.03], dtype=dtype)
    y_adas = torch.tensor([[10.0], [9.0], [8.0], [7.0]], dtype=dtype)
    t_his = torch.tensor([0.02, 0.04, 0.06, 0.08], dtype=dtype)
    y_his = torch.tensor([[8.0], [6.0], [4.0], [2.0]], dtype=dtype)
    segments = swt._TrajectorySegments(
        traj=[(t_his[i], y_his[i]) for i in range(len(t_his))],
        traj_adas=[(t_adas[i], y_adas[i]) for i in range(len(t_adas))],
        traj_his=[(t_his[i], y_his[i]) for i in range(len(t_his))],
        index_start_his=2,
        tstart_his=0.02,
        adas_stop_state={"stop_reason": "adas"},
        his_stop_state={"stop_reason": "his"},
    )
    params = type("Params", (), {"mode_array": ((2, 2),)})()

    summary = swt._phm_boundary_summary_from_segments(params, segments, 0.06)

    assert summary["n_adas"] == 4
    assert summary["n_his"] == 4
    assert summary["n_joined"] == 4
    assert summary["index_join_his"] == 2
    assert summary["index_join_attach"] == 4
    assert summary["t_join_attach_M"] == pytest.approx(0.06)
    assert summary["joined_end_M"] == pytest.approx(0.04)
    assert summary["r_his_start"] == pytest.approx(8.0)
    assert summary["r_end"] == pytest.approx(6.0)
    assert summary["adas_samples"][1]["t_M"] == pytest.approx(0.01)
    assert summary["adas_samples"][1]["r"] == pytest.approx(9.0)
    assert summary["joined_samples"][2]["state_prefix"] == [8.0]


def test_phm_adas_only_diagnostic_skips_his(monkeypatch):
    """The AdaS-only diagnostic path should not launch the HiS rerun."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.12,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2),),
    )
    y0 = torch.arange(12, dtype=torch.float64)
    calls = []

    def fake_integrate(_rhs, y_start, *_args, progress_label=None, return_diagnostics=False, **_kwargs):
        calls.append(progress_label)
        traj = [
            (torch.tensor(1.0, dtype=torch.float64), y_start + 1.0),
            (torch.tensor(2.0, dtype=torch.float64), y_start + 2.0),
        ]
        diag = {"accepted_steps": 2, "rejected_steps": 0, "attempted_steps": 2}
        return (traj, diag) if return_diagnostics else traj

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_ADAS_ONLY", "1")
    monkeypatch.setattr(dyn, "initial_conditions", lambda *_args, **_kwargs: y0)
    monkeypatch.setattr(swt, "integrate", fake_integrate)

    segments = swt._integrate_traj(
        params,
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        f_final=0.0,
        delta_t=1.0 / 4096.0,
        return_segments=True,
    )

    assert calls == ["adas"]
    assert segments.traj_his == []
    assert segments.his_stop_state["stop_reason"] == "adas_only"
    assert len(segments.traj_adas) == 2


def test_lal_local_interpolation_window_indices():
    """SEOBInterpolateDynamicsAtTime uses the last <= sample and +/-20 points."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.arange(50, dtype=torch.float64)

    assert swt._interpolation_window_indices(t, 22.4) == (2, 42)
    assert swt._interpolation_window_indices(t, 1.5) == (0, 21)
    assert swt._interpolation_window_indices(t, 48.5) == (28, 49)


def test_his_state_transfer_uses_lal_local_cubic_window():
    """HiS IC transfer should mirror SEOBInterpolateDynamicsAtTime."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.arange(50, dtype=dtype)
    y = torch.stack(
        [
            torch.sin(0.13 * t),
            torch.cos(0.07 * t),
            0.01 * t * t,
        ],
        dim=1,
    )
    traj = [(t[i], y[i]) for i in range(len(t))]
    q = 22.4

    got = swt._interpolate_traj_at_time(traj, q)
    expected = swt._interp_series_cubic(torch.tensor([q], dtype=dtype), t[2:43], y[2:43])[0]

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_sample_trajectory_states_batches_columnwise_cubic():
    """Trajectory resampling should match the previous column-wise cubic calls."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_src = torch.linspace(0.0, 3.0, 16, dtype=dtype)
    t_query = torch.linspace(0.15, 2.85, 23, dtype=dtype)
    y_src = torch.stack(
        [
            torch.sin(t_src),
            torch.cos(0.7 * t_src),
            t_src * t_src,
            torch.exp(-0.2 * t_src),
        ],
        dim=1,
    )

    got = swt._sample_trajectory_states(t_query, t_src, y_src)
    expected = torch.stack(
        [
            swt._interp_series_cubic(t_query, t_src, y_src[:, i])
            for i in range(y_src.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_sample_trajectory_states_handles_two_point_vector_fallback():
    """Capped HiS diagnostics may resample vector states from two support rows."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_src = torch.tensor([0.0, 2.0], dtype=dtype)
    y_src = torch.tensor(
        [
            [1.0, 10.0, -3.0],
            [5.0, 14.0, 9.0],
        ],
        dtype=dtype,
    )
    t_query = torch.tensor([0.5, 1.0, 1.5], dtype=dtype)

    got = swt._sample_trajectory_states(t_query, t_src, y_src)
    expected = torch.tensor(
        [
            [2.0, 11.0, 0.0],
            [3.0, 12.0, 3.0],
            [4.0, 13.0, 6.0],
        ],
        dtype=dtype,
    )

    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_series_from_states_uses_lal_velocity_omega(monkeypatch):
    """Mode series should use LAL's r x rdot omega, not sampled phase slope."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.zeros((3, 12), dtype=torch.float64)
    y[:, 0] = torch.tensor([11.0, 10.9, 10.7], dtype=torch.float64)
    y[:, 1] = -1.0e-3
    y[:, 2] = torch.tensor([0.0, 100.0, 200.0], dtype=torch.float64)
    y[:, 3:6] = torch.tensor([0.2, -0.1, 3.4], dtype=torch.float64)
    y[:, 6:9] = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64)
    y[:, 9:12] = torch.tensor([-0.02, 0.04, 0.15], dtype=torch.float64)
    omega_lal = torch.tensor([0.03, 0.031, 0.032], dtype=torch.float64)

    def fake_omega(r, pr, phi, Lvec, S1vec, S2vec, params_in):
        assert params_in is params
        return omega_lal

    monkeypatch.setattr(swt, "_omega_from_hamiltonian_velocity", fake_omega)
    *_, omega_orb = swt._series_from_states(t, y, params)
    torch.testing.assert_close(omega_orb, omega_lal / params.M_sec, rtol=0.0, atol=0.0)


def test_hamiltonian_velocity_omega_uses_lal_numerical_derivative(monkeypatch):
    """LAL's derived omegaVec path uses numerical Hamiltonian dH/dP by default."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    r = torch.tensor([10.0, 8.0], dtype=torch.float64)
    pr = torch.zeros_like(r)
    phi = torch.zeros_like(r)
    L = torch.zeros((2, 3), dtype=torch.float64)
    S1 = torch.zeros((2, 3), dtype=torch.float64)
    S2 = torch.zeros((2, 3), dtype=torch.float64)
    expected = torch.tensor([0.03, 0.04], dtype=torch.float64)

    def fake_reduced_to_cartesian(r_c, pr_c, phi_c, L_c, S1_c, S2_c, params_in):
        r_vec = torch.stack([r_c, torch.zeros_like(r_c), torch.zeros_like(r_c)], dim=-1)
        p_vec = torch.zeros_like(r_vec)
        n_hat = r_vec / r_c.unsqueeze(-1)
        return r_vec, p_vec, n_hat, torch.zeros_like(r_vec), torch.zeros_like(r_vec)

    def fake_potentials(*args, **kwargs):
        assert kwargs["compute_grad_p"] is False
        assert kwargs["fd_dpvec"] is True
        assert kwargs["fd_pphi"] is False
        r_vec = kwargs["r_vec"]
        omega = expected[: r_vec.shape[0]]
        dH_dpvec = torch.stack(
            [torch.zeros_like(omega), omega * r_vec[:, 0], torch.zeros_like(omega)],
            dim=-1,
        )
        return {
            "dH_dpvec": dH_dpvec,
            "n_hat": r_vec / r_vec[:, 0].unsqueeze(-1),
            "csi": torch.ones_like(omega),
        }

    monkeypatch.setattr(swt._dyn, "_reduced_to_cartesian", fake_reduced_to_cartesian)
    monkeypatch.setattr(swt._dyn, "_eob_potentials", fake_potentials)

    got = swt._omega_from_hamiltonian_velocity(r, pr, phi, L, S1, S2, params)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_nqc_spline_values_derivatives_batches_values_only():
    """Batched NQC spline values should preserve scalar values and derivatives."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt
    from pycbc.waveform.seobnrv4phm_peak import local_derivatives

    dtype = torch.float64
    t = torch.linspace(-2.0, 2.0, 31, dtype=dtype)
    series = (
        torch.sin(0.4 * t),
        torch.cos(0.3 * t) + 0.1 * t,
        0.2 * t * t - 0.05 * t,
    )
    t0 = 0.37
    got = swt._spline_values_derivatives(series, t, t0)
    q = torch.tensor([t0], dtype=dtype)

    for item, single in zip(got, series, strict=False):
        expected = (
            float(swt._interp_series_cubic(q, t, single)[0]),
            local_derivatives(single, t, t0, order=1),
            local_derivatives(single, t, t0, order=2),
        )
        assert item == pytest.approx(expected, rel=0.0, abs=1e-12)
