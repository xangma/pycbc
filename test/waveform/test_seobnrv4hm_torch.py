import os
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme
from pycbc.types import Array
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.waveform._seobnrv4_qnm import (
    seobnrv4_final_mass_spin,
    seobnrv4_qnm_omega,
)
from pycbc.waveform.seobnrv4hm_torch import (
    _active_mode_indices,
    _compute_i_max_LF_i_min_HF,
    seobnrv4hm_native_supported,
    seobnrv4hm_sequence_native_supported,
)

_ROM_FILENAMES = ("SEOBNRv4HMROM_v1.0.hdf5", "SEOBNRv4HMROM.hdf5")
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent
_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV4HM_NATIVE",
)
_BASE_PARAMS = dict(
    distance=500.0,
    inclination=0.7,
    coa_phase=0.3,
)


@pytest.fixture(autouse=True)
def preserve_process_state():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    old_environment = {name: os.environ.get(name) for name in _NATIVE_FLAGS}
    try:
        yield
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _generate(params, *, native, device=None):
    os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1" if native else "0"
    if device is None:
        _activate_scheme(_scheme.CPUScheme())
    else:
        _activate_scheme(_scheme.TorchScheme(device))
    return get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params)


def _generate_sequence(params, sample_points, *, native, device=None):
    os.environ["PYCBC_SEOBNRV4HM_NATIVE"] = "1" if native else "0"
    if device is None:
        _activate_scheme(_scheme.CPUScheme())
    else:
        _activate_scheme(_scheme.TorchScheme(device))
    return get_fd_waveform_sequence(
        approximant="SEOBNRv4HM_ROM",
        sample_points=sample_points,
        **params,
    )


def _snapshot_sequence(series_pair):
    return tuple(series.numpy().copy() for series in series_pair)


def _assert_sequence_parity(reference, actual, relative_tolerance):
    for expected, result in zip(reference, actual):
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected == 0.0,
        )
        nonzero = expected != 0.0
        assert nonzero.any(), "waveform contains no non-zero samples"
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < relative_tolerance


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
            long_asc_nodes=0.37,
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


@pytest.mark.parametrize(
    ("params", "sample_points"),
    [
        (
            {
                **_BASE_PARAMS,
                "mass1": 35.0,
                "mass2": 25.0,
                "spin1z": 0.2,
                "spin2z": -0.1,
                "f_ref": 30.0,
                "phase_order": 2,
                "amplitude_order": 3,
                "eccentricity_order": 4,
            },
            [20.0, 25.5, 50.0, 100.0, 300.0, 1000.0, 4000.0],
        ),
        (
            {
                **_BASE_PARAMS,
                "mass1": 18.0,
                "mass2": 42.0,
                "spin1z": -0.3,
                "spin2z": 0.45,
                "f_ref": 0.0,
                "long_asc_nodes": 0.73,
                "mode_array": [(2, -1), (5, -5)],
            },
            [20.0, 400.0, 22.5, 150.0, 4000.0],
        ),
    ],
)
def test_seobnrv4hm_default_sequence_matches_lal_without_calling_lal(
    params, sample_points, monkeypatch
):
    _require_rom_data()
    reference = _snapshot_sequence(
        _generate_sequence(params, sample_points, native=False)
    )

    import pycbc.waveform.seobnrv4hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4hm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(
            "native SEOBNRv4HM_ROM sequence called lalsimulation"
        )

    monkeypatch.setattr(
        native_module,
        "seobnrv4hm_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="SEOBNRv4HM_ROM",
        sample_points=sample_points,
        **params,
    )

    assert native_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_sequence_parity(reference, actual, relative_tolerance=1.0e-10)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        (
            {"mass1": 35.0, "mass2": 25.0, "f_lower": 9.0},
            True,
        ),
        (
            {"mass1": 35.0, "mass2": 25.0, "f_lower": 8.0},
            True,
        ),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "f_lower": 4.0,
                "mode_array": [(2, -2)],
            },
            True,
        ),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "f_lower": 3.0,
                "mode_array": [(2, -2)],
            },
            True,
        ),
        ({"spin1x": 0.1}, False),
        ({"phase_order": 2}, True),
        ({"amplitude_order": 3}, True),
        ({"eccentricity_order": 4}, True),
        ({"phase_order": 2.5}, True),
        ({"amplitude_order": "3"}, True),
        ({"eccentricity_order": 4.0}, False),
        ({"phase_order": 1 << 31}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 12}, False),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"mode_array": []}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ],
)
def test_seobnrv4hm_native_support_boundary(params, expected):
    assert seobnrv4hm_native_supported(params) is expected


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "sample_points": [9.0, 20.0],
            },
            True,
        ),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "sample_points": [20.0, 8.0],
            },
            True,
        ),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "sample_points": [4.0, 20.0],
                "mode_array": [(2, -2)],
            },
            True,
        ),
        (
            {
                "mass1": 35.0,
                "mass2": 25.0,
                "sample_points": [20.0, 3.0],
                "mode_array": [(2, -2)],
            },
            True,
        ),
        (
            {
                "phase_order": 2,
                "amplitude_order": 3,
                "eccentricity_order": 4,
            },
            True,
        ),
        ({"spin2y": 0.1}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 12}, False),
        ({"eccentricity": 0.1}, False),
    ],
)
def test_seobnrv4hm_sequence_native_support_boundary(params, expected):
    assert seobnrv4hm_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    ("params", "sample_points"),
    [
        (
            {
                **_BASE_PARAMS,
                "mass1": 35.0,
                "mass2": 25.0,
                "spin1z": 0.2,
                "spin2z": -0.1,
                "f_ref": 20.0,
            },
            [2.0, 3.0, 5.0, 9.0, 20.0, 100.0],
        ),
        (
            {
                **_BASE_PARAMS,
                "mass1": 30.0,
                "mass2": 30.0,
                "spin1z": 0.8,
                "spin2z": -0.8,
                "f_ref": 20.0,
                "mode_array": [(2, -1)],
            },
            [2.0, 3.0, 5.0, 9.0, 20.0, 100.0],
        ),
        (
            {
                **_BASE_PARAMS,
                "mass1": 18.0,
                "mass2": 42.0,
                "spin1z": -0.3,
                "spin2z": 0.45,
                "f_ref": 20.0,
                "mode_array": [(2, -1), (5, -5)],
            },
            [2.0, 4.0, 8.0, 20.0, 100.0],
        ),
    ],
)
def test_seobnrv4hm_sequence_low_frequency_is_native(
    params, sample_points, monkeypatch
):
    _require_rom_data()
    reference = _snapshot_sequence(
        _generate_sequence(params, sample_points, native=False)
    )

    import pycbc.waveform.seobnrv4hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4hm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(
            "native low-frequency HM sequence called lalsimulation"
        )

    monkeypatch.setattr(
        native_module,
        "seobnrv4hm_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    actual = _generate_sequence(
        params,
        sample_points,
        native=True,
        device="cpu",
    )

    assert native_calls == 1
    assert all(isinstance(series._data.tensor, torch.Tensor) for series in actual)
    _assert_sequence_parity(reference, actual, relative_tolerance=1.0e-10)


def test_seobnrv4hm_default_regular_low_frequency_is_native(monkeypatch):
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.25,
        "f_lower": 2.0,
        "f_final": 128.0,
        "f_ref": 20.0,
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
    reference = tuple(
        series.numpy().copy() for series in _generate(params, native=False)
    )

    import pycbc.waveform.seobnrv4hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4hm_fd_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(
            "native low-frequency HM waveform called lalsimulation"
        )

    monkeypatch.setattr(
        native_module,
        "seobnrv4hm_fd_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params)

    assert native_calls == 1
    for expected, result in zip(reference, actual):
        assert isinstance(result._data.tensor, torch.Tensor)
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1.0e-10,
            atol=scale * 1.0e-12,
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_seobnrv4hm_default_sequence_stays_on_requested_device(
    device_name, monkeypatch
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
    }
    sample_points = (
        [20.0, 30.0, 50.0, 100.0, 400.0]
        if device_name == "mps"
        else [2.0, 3.0, 5.0, 9.0, 20.0, 100.0, 400.0]
    )
    reference = _snapshot_sequence(
        _generate_sequence(params, sample_points, native=False)
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_sequence(
        approximant="SEOBNRv4HM_ROM",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert all(
        series._data.tensor.device.type == device_name for series in actual
    )
    assert all(
        series._data.tensor.dtype == expected_dtype for series in actual
    )
    tolerance = 1.0e-4 if device_name == "mps" else 1.0e-10
    _assert_sequence_parity(reference, actual, relative_tolerance=tolerance)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_seobnrv4hm_default_regular_stays_on_requested_device(
    device_name, monkeypatch
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 2.0,
        "f_lower": 20.0,
        "f_final": 128.0,
        "f_ref": 20.0,
        "mode_array": [(2, -2)],
    }
    reference = tuple(
        series.numpy().copy() for series in _generate(params, native=False)
    )

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(approximant="SEOBNRv4HM_ROM", **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = 1.0e-4 if device_name == "mps" else 1.0e-10
    for expected, result in zip(reference, actual):
        tensor = result._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        nonzero = expected != 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < tolerance


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_seobnrv4hm_default_mps_low_frequency_uses_lalsimulation(monkeypatch):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    original_regular = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    original_sequence = (
        waveform_module.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    regular_calls = 0
    sequence_calls = 0

    def recording_regular(*args, **kwargs):
        nonlocal regular_calls
        regular_calls += 1
        return original_regular(*args, **kwargs)

    def recording_sequence(*args, **kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        return original_sequence(*args, **kwargs)

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_regular,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("mps"))

    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
    }
    regular_support = {
        **params,
        "approximant": "SEOBNRv4HM_ROM",
        "f_lower": 2.0,
    }
    sequence_support = {
        **params,
        "approximant": "SEOBNRv4HM_ROM",
        "sample_points": [2.0, 20.0],
    }
    assert not seobnrv4hm_native_supported(regular_support)
    assert not seobnrv4hm_sequence_native_supported(sequence_support)
    assert not seobnrv4hm_sequence_native_supported(
        {**sequence_support, "sample_points": [4.0, 2.0, 20.0]}
    )
    assert seobnrv4hm_native_supported(
        {**regular_support, "f_lower": 4.0}
    )
    assert seobnrv4hm_sequence_native_supported(
        {**sequence_support, "sample_points": [4.0, 20.0]}
    )

    regular = get_fd_waveform(
        approximant="SEOBNRv4HM_ROM",
        delta_f=1.0,
        f_lower=2.0,
        f_final=128.0,
        **params,
    )
    sequence = get_fd_waveform_sequence(
        approximant="SEOBNRv4HM_ROM",
        sample_points=[4.0, 2.0, 5.0, 20.0, 100.0],
        **params,
    )

    assert regular_calls == 1
    assert sequence_calls == 1
    for results, expected_length in ((regular, 129), (sequence, 5)):
        for result in results:
            tensor = result._data.tensor
            assert len(result) == expected_length
            assert tensor.device.type == "mps"
            assert tensor.dtype == torch.complex64
            assert bool(torch.isfinite(tensor).all())


def test_seobnrv4hm_sequence_rejects_frequency_below_first_sample_domain():
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
    }

    with pytest.raises(ValueError, match="below the TaylorF2 spline domain"):
        _generate_sequence(
            params,
            [20.0, 0.5],
            native=True,
            device="cpu",
        )


def test_seobnrv4hm_default_sequence_avoids_host_transfer(monkeypatch):
    _require_rom_data()

    def reject_host_transfer(_self):
        raise AssertionError(
            "native SEOBNRv4HM_ROM sequence transferred to NumPy"
        )

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array([2.0, 3.0, 20.0, 100.0, 400.0])
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant="SEOBNRv4HM_ROM",
            sample_points=sample_points,
            mass1=35.0,
            mass2=25.0,
            spin1z=0.2,
            spin2z=-0.1,
            f_ref=20.0,
            distance=500.0,
            inclination=0.7,
            coa_phase=0.3,
            mode_array=[(2, -2)],
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


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
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "seobnrv4hm_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "seobnrv4hm_fd_sequence_torch",
        ),
    ),
)
def test_seobnrv4hm_default_unsupported_options_use_lal_fallback(
    interface,
    lal_name,
    native_name,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.seobnrv4hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported SEOBNRv4HM request reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_module, native_name, unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        lal_name,
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
        "dchi0": 0.01,
    }
    if interface == "regular":
        result = get_fd_waveform(
            approximant="SEOBNRv4HM_ROM",
            delta_f=2.0,
            f_lower=20.0,
            f_final=128.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="SEOBNRv4HM_ROM",
            sample_points=[20.0, 40.0, 80.0, 120.0],
            **params,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
@pytest.mark.parametrize(
    ("disabled_flag", "global_enabled"),
    (
        ("PYCBC_TORCH_NATIVE_PORTS", False),
        ("PYCBC_SEOBNRV4HM_NATIVE", True),
    ),
)
def test_seobnrv4hm_default_native_opt_out(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    _clear_native_flags(monkeypatch)
    if global_enabled:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv(disabled_flag, "0")
    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        lal_name,
        recording_lal,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
    }
    if interface == "regular":
        result = get_fd_waveform(
            approximant="SEOBNRv4HM_ROM",
            delta_f=2.0,
            f_lower=20.0,
            f_final=128.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="SEOBNRv4HM_ROM",
            sample_points=[20.0, 40.0, 80.0, 120.0],
            **params,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


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
    final_mass, final_spin = seobnrv4_final_mass_spin(*parameters)
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
    omega = seobnrv4_qnm_omega(35.0, 25.0, 0.2, -0.1, *mode)
    scaled_omega = seobnrv4_qnm_omega(70.0, 50.0, 0.2, -0.1, *mode)
    assert omega == pytest.approx(expected, rel=2e-14)
    assert scaled_omega == pytest.approx(omega, rel=2e-14)


def test_mode_array_uses_directly_modeled_negative_m_modes():
    assert _active_mode_indices([(2, -2), (4, -4)]) == (0, 3)
    with pytest.raises(ValueError, match="positive-m"):
        _active_mode_indices([(2, 2)])
    with pytest.raises(ValueError, match="not available"):
        _active_mode_indices([(3, -2)])


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_hybridization_indices_stay_on_requested_device(device_name):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    low = torch.tensor([0.001, 0.002, 0.003, 0.004], device=device_name)
    high = torch.tensor([0.002, 0.003, 0.004, 0.005], device=device_name)

    assert _compute_i_max_LF_i_min_HF(low, high, 0.003) == (1, 1)


@pytest.mark.parametrize("threshold", [0.001, 0.006])
def test_hybridization_indices_require_patch_overlap(threshold):
    low = torch.tensor([0.001, 0.002, 0.003, 0.004])
    high = torch.tensor([0.002, 0.003, 0.004, 0.005])

    with pytest.raises(ValueError, match="do not overlap"):
        _compute_i_max_LF_i_min_HF(low, high, threshold)


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
