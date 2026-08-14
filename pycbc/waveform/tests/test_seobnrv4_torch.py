import os
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.seobnrv4_torch import (  # noqa: E402
    _clear_rom_cache,
    seobnrv4_rom_native_supported,
)


_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent
_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV4_NATIVE",
)
_BASE_PARAMS = dict(
    distance=400.0,
    inclination=0.7,
    coa_phase=0.5,
)


@pytest.fixture(scope="module", autouse=True)
def require_rom_data():
    search_paths = [_WAVEFORM_DIR]
    search_paths.extend(
        Path(path)
        for path in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if path
    )
    if not any((path / _ROM_FILENAME).is_file() for path in search_paths):
        pytest.skip(f"{_ROM_FILENAME} is not available on LAL_DATA_PATH")
    yield
    _clear_rom_cache()


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


def _generate(params, *, native, device=None, approximant="SEOBNRv4_ROM"):
    os.environ["PYCBC_SEOBNRV4_NATIVE"] = "1" if native else "0"
    if device is None:
        _activate_scheme(_scheme.CPUScheme())
    else:
        _activate_scheme(_scheme.TorchScheme(device))
    return get_fd_waveform(approximant=approximant, **params)


def _snapshot(series_pair):
    return tuple(
        (len(series), series.delta_f, float(series.epoch), series.numpy().copy())
        for series in series_pair
    )


def _assert_parity(reference, actual, relative_tolerance):
    for expected, result in zip(reference, actual):
        expected_length, expected_delta_f, expected_epoch, expected_array = expected
        assert len(result) == expected_length
        assert result.delta_f == expected_delta_f
        assert float(result.epoch) == expected_epoch

        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected_array == 0.0)
        nonzero = expected_array != 0.0
        assert nonzero.any(), "waveform contains no non-zero bins"
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < relative_tolerance


@pytest.mark.parametrize(
    "params",
    [
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 0.25,
            "f_lower": 20.0,
            "f_final": 0.0,
            "f_ref": 20.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 12.0,
            "mass2": 40.0,
            "spin1z": -0.4,
            "spin2z": 0.7,
            "delta_f": 0.25,
            "f_lower": 17.3,
            "f_final": 133.3,
            "f_ref": 37.0,
            "long_asc_nodes": 0.23,
        },
        {
            **_BASE_PARAMS,
            "mass1": 40.0,
            "mass2": 1.0,
            "spin1z": 0.997,
            "spin2z": -0.3,
            "delta_f": 0.5,
            "f_lower": 10.0,
            "f_final": 0.0,
            "f_ref": 31.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 25.0,
            "mass2": 25.0,
            "spin1z": 1.0,
            "spin2z": -1.0,
            "delta_f": 0.5,
            "f_lower": 20.0,
            "f_final": 4096.0,
            "f_ref": 0.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 128.0 / 8192.0,
            "f_lower": 20.0,
            "f_final": 128.001,
            "f_ref": 20.0,
        },
    ],
)
def test_seobnrv4_rom_cpu_torch_parity(params):
    reference = _snapshot(_generate(params, native=False))
    actual = _generate(params, native=True, device="cpu")
    _assert_parity(reference, actual, relative_tolerance=1.0e-8)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"lambda1": 0.0, "dchi3": 0.0}, True),
        ({"approximant": "SEOBNRv4"}, False),
        ({"approximant": "SEOBNRv4_ROM_NRTidalv2"}, False),
        ({"spin1x": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"lambda1": 100.0}, False),
        ({"eccentricity": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ],
)
def test_seobnrv4_rom_native_support_boundary(params, expected):
    assert seobnrv4_rom_native_supported(params) is expected


def test_seobnrv4_rom_public_native_dispatch_avoids_lalsimulation(monkeypatch):
    params = {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 28.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.25,
        "f_lower": 20.0,
        "f_ref": 30.0,
        "long_asc_nodes": 0.37,
    }
    reference = _snapshot(_generate(params, native=False))

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4_fd_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4_ROM called lalsimulation")

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    actual = _generate(params, native=True, device="cpu")

    assert native_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual, relative_tolerance=1.0e-8)


def test_seobnrv4_rom_unsupported_options_use_lal_fallback(monkeypatch):
    params = {
        **_BASE_PARAMS,
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 20.0,
        "dchi3": 0.1,
    }
    reference = _snapshot(_generate(params, native=False))

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported parameters reached the native ROM")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    actual = _generate(params, native=True, device="cpu")

    assert lal_calls == 1
    assert all(isinstance(series._data.tensor, torch.Tensor) for series in actual)
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


def test_seobnrv4_rom_global_switch_fallback(monkeypatch):
    params = {
        **_BASE_PARAMS,
        "mass1": 20.0,
        "mass2": 15.0,
        "spin1z": 0.1,
        "spin2z": -0.05,
        "delta_f": 0.5,
        "f_lower": 20.0,
    }
    reference = _snapshot(_generate(params, native=False))
    os.environ.pop("PYCBC_SEOBNRV4_NATIVE", None)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="SEOBNRv4_ROM", **params)
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


def test_time_domain_seobnrv4_remains_on_lal_path(monkeypatch):
    params = dict(
        mass1=50.0,
        mass2=40.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=1.0,
        f_lower=30.0,
        distance=400.0,
        inclination=0.4,
        coa_phase=0.2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = _snapshot(
            _generate(params, native=False, approximant="SEOBNRv4")
        )

    import pycbc.waveform.seobnrv4_torch as native_module

    def unexpected_native(**_params):
        raise AssertionError("time-domain SEOBNRv4 reached the ROM evaluator")

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", unexpected_native)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = _generate(
            params, native=True, device="cpu", approximant="SEOBNRv4"
        )
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_seobnrv4_rom_stays_on_requested_device(device_name):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {
        **_BASE_PARAMS,
        "mass1": 30.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 20.0,
        "f_ref": 37.0,
        "long_asc_nodes": 0.23,
    }
    reference = _snapshot(_generate(params, native=False))
    actual = _generate(params, native=True, device=device_name)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert all(series._data.tensor.device.type == device_name for series in actual)
    assert all(series._data.tensor.dtype == expected_dtype for series in actual)
    tolerance = 2.0e-2 if device_name == "mps" else 1.0e-8
    _assert_parity(reference, actual, relative_tolerance=tolerance)


def test_seobnrv4_rom_reconstruction_uses_torch_tensors(monkeypatch):
    import pycbc.waveform.seobnrv4_torch as native_module

    evaluated_submodels = []
    spline_points = []
    evaluate_submodel = native_module._evaluate_submodel
    spline_eval = native_module._spline_eval

    def recording_submodel(*args, **kwargs):
        result = evaluate_submodel(*args, **kwargs)
        evaluated_submodels.extend(result)
        return result

    def recording_spline(points, *args, **kwargs):
        spline_points.append(points)
        return spline_eval(points, *args, **kwargs)

    monkeypatch.setattr(native_module, "_evaluate_submodel", recording_submodel)
    monkeypatch.setattr(native_module, "_spline_eval", recording_spline)
    actual = _generate(
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 0.5,
            "f_lower": 20.0,
        },
        native=True,
        device="cpu",
    )

    assert evaluated_submodels
    assert spline_points
    assert all(isinstance(value, torch.Tensor) for value in evaluated_submodels)
    assert all(isinstance(value, torch.Tensor) for value in spline_points)
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
