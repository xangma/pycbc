# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Waveform regressions for the Torch-native SEOBNRv4T surrogate."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence  # noqa: E402
from pycbc.waveform import seobnrv4t_surrogate_torch as surrogate  # noqa: E402


_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV4T_SURROGATE_NATIVE",
)
_BASE_PARAMETERS = {
    "mass1": 1.6,
    "mass2": 1.2,
    "spin1z": 0.1,
    "spin2z": -0.05,
    "lambda1": 400.0,
    "lambda2": 800.0,
    "distance": 100.0,
    "inclination": 0.4,
    "coa_phase": 0.3,
    "f_ref": 50.0,
}


@pytest.fixture(scope="module", autouse=True)
def require_surrogate_data():
    try:
        data_path = surrogate._find_rom_file()
    except FileNotFoundError:
        pytest.skip("SEOBNRv4T surrogate data is not available")

    old_data_path = os.environ.get("LAL_DATA_PATH")
    search_path = str(data_path.parent)
    if old_data_path:
        search_path += os.pathsep + old_data_path
    os.environ["LAL_DATA_PATH"] = search_path
    surrogate._clear_surrogate_cache()
    try:
        yield
    finally:
        if old_data_path is None:
            os.environ.pop("LAL_DATA_PATH", None)
        else:
            os.environ["LAL_DATA_PATH"] = old_data_path
        surrogate._clear_surrogate_cache()


@pytest.fixture(autouse=True)
def preserve_scheme():
    old_state = _scheme.mgr.state
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
        _scheme.mgr.state = old_state
        _scheme.Scheme._single = old_single


def _activate_torch_cpu():
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("cpu")


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _lal_regular(parameters):
    return lalsimulation.SimIMRSEOBNRv4TSurrogate(
        parameters["coa_phase"],
        parameters["delta_f"],
        parameters["f_lower"],
        parameters["f_final"],
        parameters["f_ref"],
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters["inclination"],
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["spin1z"],
        parameters["spin2z"],
        parameters["lambda1"],
        parameters["lambda2"],
        lalsimulation.SEOBNRv4TSurrogate_CUBIC,
    )


def _lal_sequence(parameters, frequencies):
    lal_frequencies = lal.CreateREAL8Sequence(len(frequencies))
    lal_frequencies.data[:] = frequencies
    return lalsimulation.SimIMRSEOBNRv4TSurrogateFrequencySequence(
        lal_frequencies,
        parameters["coa_phase"],
        parameters["f_ref"],
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters["inclination"],
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["spin1z"],
        parameters["spin2z"],
        parameters["lambda1"],
        parameters["lambda2"],
        lalsimulation.SEOBNRv4TSurrogate_CUBIC,
    )


def _assert_strain_close(expected, actual, *, tolerance=5.0e-8):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    np.testing.assert_array_equal(actual == 0.0, expected == 0.0)
    nonzero = expected != 0.0
    assert np.any(nonzero)
    relative_error = np.linalg.norm(actual[nonzero] - expected[nonzero])
    relative_error /= np.linalg.norm(expected[nonzero])
    assert relative_error < tolerance


@pytest.mark.parametrize(
    "intrinsic",
    [
        {},
        {
            "mass1": 1.2,
            "mass2": 1.8,
            "spin1z": -0.2,
            "spin2z": 0.3,
            "lambda1": 900.0,
            "lambda2": 300.0,
        },
    ],
)
def test_regular_waveform_matches_lal(intrinsic):
    parameters = (
        _BASE_PARAMETERS
        | intrinsic
        | {
            "delta_f": 1.0,
            "f_lower": 20.0,
            "f_final": 2048.0,
        }
    )
    expected = _lal_regular(parameters)
    _activate_torch_cpu()
    actual = surrogate.seobnrv4t_surrogate_fd_torch(**parameters)

    for expected_series, actual_series in zip(expected, actual):
        assert len(actual_series) == len(expected_series.data.data)
        assert actual_series.delta_f == expected_series.deltaF
        assert float(actual_series.epoch) == float(expected_series.epoch)
        _assert_strain_close(expected_series.data.data, actual_series.numpy())


def test_frequency_sequence_matches_lal_and_preserves_cutoff_zeros():
    frequencies = np.array([20.0, 500.0, 6000.0, 100.0, 2048.0, 6000.0])
    expected = _lal_sequence(_BASE_PARAMETERS, frequencies)
    _activate_torch_cpu()
    actual = surrogate.seobnrv4t_surrogate_fd_sequence_torch(
        **_BASE_PARAMETERS,
        sample_points=frequencies,
    )

    for expected_series, actual_array in zip(expected, actual):
        _assert_strain_close(expected_series.data.data, actual_array.numpy())
    assert actual[0].numpy()[2] == 0.0
    assert actual[0].numpy()[-1] == 0.0


def test_public_default_dispatch_stays_native_and_on_device(monkeypatch):
    import pycbc.waveform.seobnrv4t_surrogate_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    regular_native = native_module.seobnrv4t_surrogate_fd_torch
    sequence_native = native_module.seobnrv4t_surrogate_fd_sequence_torch
    regular_calls = 0
    sequence_calls = 0

    def recording_regular(**parameters):
        nonlocal regular_calls
        regular_calls += 1
        return regular_native(**parameters)

    def recording_sequence(**parameters):
        nonlocal sequence_calls
        sequence_calls += 1
        return sequence_native(**parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4T surrogate called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native SEOBNRv4T sequence transferred to NumPy")

    _clear_native_flags(monkeypatch)
    _activate_torch_cpu()
    monkeypatch.setattr(
        native_module,
        "seobnrv4t_surrogate_fd_torch",
        recording_regular,
    )
    monkeypatch.setattr(
        native_module,
        "seobnrv4t_surrogate_fd_sequence_torch",
        recording_sequence,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)

    regular = get_fd_waveform(
        approximant="SEOBNRv4T_surrogate",
        delta_f=1.0,
        f_lower=20.0,
        f_final=2048.0,
        **_BASE_PARAMETERS,
    )
    sequence = get_fd_waveform_sequence(
        approximant="SEOBNRv4T_surrogate",
        sample_points=Array([20.0, 500.0, 100.0, 2048.0, 6000.0]),
        **_BASE_PARAMETERS,
    )

    assert regular_calls == 1
    assert sequence_calls == 1
    for result in (*regular, *sequence):
        assert isinstance(result._data, TorchArrayData)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128


@pytest.mark.parametrize(
    "fallback",
    ("disabled", "unsupported"),
)
def test_public_dispatch_retains_lalsimulation_fallback(fallback, monkeypatch):
    import pycbc.waveform.waveform as waveform_module

    original = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    calls = 0

    def recording_lalsimulation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    _clear_native_flags(monkeypatch)
    if fallback == "disabled":
        monkeypatch.setenv("PYCBC_SEOBNRV4T_SURROGATE_NATIVE", "0")
        extra_parameters = {}
    else:
        extra_parameters = {"phase_order": 7}
    _activate_torch_cpu()
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lalsimulation,
    )

    hp, hc = get_fd_waveform(
        approximant="SEOBNRv4T_surrogate",
        delta_f=1.0,
        f_lower=20.0,
        f_final=2048.0,
        **_BASE_PARAMETERS,
        **extra_parameters,
    )

    assert calls == 1
    assert isinstance(hp._data, TorchArrayData)
    assert isinstance(hc._data, TorchArrayData)


@pytest.mark.parametrize("lambda_tidal", [0.0, 0.5, 1.0, 400.0, 5000.0])
def test_universal_quadrupole_matches_lal(lambda_tidal):
    expected = lalsimulation.SimUniversalRelationQuadMonVSlambda2Tidal(lambda_tidal)
    assert surrogate._universal_quadrupole(lambda_tidal) == pytest.approx(
        expected,
        rel=2.0e-15,
        abs=2.0e-15,
    )


def test_native_support_rejects_unimplemented_features():
    assert surrogate.seobnrv4t_surrogate_native_supported(_BASE_PARAMETERS)
    assert not surrogate.seobnrv4t_surrogate_native_supported(
        _BASE_PARAMETERS | {"spin1x": 0.1}
    )
    assert not surrogate.seobnrv4t_surrogate_sequence_native_supported(
        _BASE_PARAMETERS | {"phase_order": 7}
    )
