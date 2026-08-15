import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.imrphenomhm_torch import (  # noqa: E402
    _active_modes,
    imrphenomhm_native_supported,
)


_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_IMRPHENOMHM_NATIVE",
)


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(scheme):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme


def _run_case(params, *, use_native=True):
    env_backup = {key: os.environ.get(key) for key in _NATIVE_FLAGS}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMHM_NATIVE"] = "0"
        hp_cpu, hc_cpu = get_fd_waveform(
            approximant="IMRPhenomHM", **params
        )

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        enabled = "1" if use_native else "0"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = enabled
        os.environ["PYCBC_IMRPHENOMHM_NATIVE"] = enabled
        hp_torch, hc_torch = get_fd_waveform(
            approximant="IMRPhenomHM", **params
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return (hp_cpu, hc_cpu), (hp_torch, hc_torch)


def _assert_parity(cpu_polarizations, torch_polarizations):
    for cpu_series, torch_series in zip(
        cpu_polarizations, torch_polarizations
    ):
        assert len(torch_series) == len(cpu_series)
        assert float(torch_series.epoch) == pytest.approx(
            float(cpu_series.epoch)
        )
        assert isinstance(torch_series._data.tensor, torch.Tensor)
        assert torch_series._data.tensor.device.type == "cpu"

        cpu = cpu_series.numpy()
        actual = torch_series.numpy()
        scale = np.max(np.abs(cpu))
        if scale == 0.0:
            np.testing.assert_array_equal(actual, cpu)
        else:
            np.testing.assert_allclose(
                actual,
                cpu,
                rtol=5e-11,
                atol=scale * 1e-12,
            )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=50.0,
            mass2=35.0,
            spin1z=0.2,
            spin2z=0.1,
            delta_f=0.5,
            f_lower=15.0,
            f_final=0.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.7,
            coa_phase=1.0,
        ),
        dict(
            mass1=18.0,
            mass2=42.0,
            spin1z=-0.4,
            spin2z=0.7,
            delta_f=0.25,
            f_lower=17.3,
            f_final=133.3,
            f_ref=0.0,
            distance=700.0,
            inclination=0.8,
            coa_phase=0.6,
            long_asc_nodes=0.37,
        ),
        dict(
            mass1=67.0,
            mass2=43.5,
            spin1z=0.9,
            spin2z=-0.17,
            delta_f=0.125,
            f_lower=19.0,
            f_final=0.0,
            f_ref=245.0,
            distance=407.0,
            inclination=1.4,
            coa_phase=2.1,
            mode_array=[(2, 2), (3, 3), (4, 4)],
        ),
    ],
)
def test_imrphenomhm_torch_parity(params):
    _assert_parity(*_run_case(params))


@pytest.mark.parametrize(
    "mode",
    [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)],
)
def test_imrphenomhm_individual_mode_parity(mode):
    params = dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        inclination=1.1,
        coa_phase=0.4,
        mode_array=[mode],
    )
    _assert_parity(*_run_case(params))


def test_imrphenomhm_empty_mode_array_matches_lal():
    params = dict(
        mass1=40.0,
        mass2=30.0,
        delta_f=1.0,
        f_lower=20.0,
        distance=400.0,
        mode_array=[],
    )
    _assert_parity(*_run_case(params))


def test_imrphenomhm_empty_frequency_grid_matches_lal():
    params = dict(
        mass1=40.0,
        mass2=30.0,
        delta_f=1000.0,
        f_lower=20.0,
        f_final=100.0,
        distance=400.0,
    )
    _assert_parity(*_run_case(params))


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, True),
        ({"mode_array": [(2, 2), (3, 3)]}, True),
        ({"mode_array": []}, True),
        ({"mode_array": [(2, -2)]}, False),
        ({"mode_array": [(5, 5)]}, False),
        ({"mode_array": [(2.5, 2)]}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 10.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi4": 0.1}, False),
        ({"phase_order": 7}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ],
)
def test_imrphenomhm_native_support_boundary(params, expected):
    full_params = {"approximant": "IMRPhenomHM", **params}
    assert imrphenomhm_native_supported(full_params) is expected


def test_imrphenomhm_active_modes_preserve_model_order():
    requested = [(4, 4), (2, 1), (4, 4)]
    assert _active_modes(requested) == ((2, 1), (4, 4))


def test_imrphenomhm_global_switch_disabled_uses_lalsim():
    params = dict(
        mass1=25.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.1,
    )
    _assert_parity(*_run_case(params, use_native=False))


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomhm_native_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=40.0,
        mass2=15.0,
        spin1z=0.6,
        spin2z=-0.3,
        delta_f=0.5,
        f_lower=18.0,
        f_ref=25.0,
        distance=350.0,
        inclination=0.9,
        coa_phase=0.3,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(
        approximant="IMRPhenomHM", **params
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, cross = get_fd_waveform(
        approximant="IMRPhenomHM", **params
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert cross._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 1.0e-4 if device_name == "mps" else 1.0e-10
    assert relative_error < tolerance
