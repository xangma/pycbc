import os
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.imrphenomc_torch import (  # noqa: E402
    imrphenomc_fd_torch,
    imrphenomc_native_supported,
)


_ENV_KEYS = ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMC_NATIVE")


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""
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


def _run_case(params):
    env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMC_NATIVE"] = "0"
        _activate_scheme(_scheme.CPUScheme())
        reference = get_fd_waveform(approximant="IMRPhenomC", **params)
        reference_arrays = tuple(series.numpy().copy() for series in reference)
        reference_metadata = tuple(
            (len(series), series.delta_f, float(series.epoch))
            for series in reference
        )

        os.environ["PYCBC_IMRPHENOMC_NATIVE"] = "1"
        _activate_scheme(_scheme.TorchScheme("cpu"))
        actual = get_fd_waveform(approximant="IMRPhenomC", **params)
        actual_arrays = tuple(series.numpy().copy() for series in actual)
        actual_metadata = tuple(
            (len(series), series.delta_f, float(series.epoch))
            for series in actual
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return (
        reference_arrays,
        actual_arrays,
        reference_metadata,
        actual_metadata,
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=35.0,
            mass2=28.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=0.25,
            f_lower=20.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.4,
            coa_phase=1.1,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1z=0.6,
            spin2z=0.3,
            delta_f=0.5,
            f_lower=15.0,
            f_ref=30.0,
            distance=300.0,
            inclination=1.2,
            coa_phase=0.3,
        ),
        dict(
            mass1=18.0,
            mass2=42.0,
            spin1z=-0.4,
            spin2z=0.7,
            delta_f=0.25,
            f_lower=17.3,
            f_final=1333.3,
            f_ref=0.0,
            distance=700.0,
            inclination=0.8,
            coa_phase=0.6,
        ),
        dict(
            mass1=67.0,
            mass2=43.5,
            spin1z=0.7,
            spin2z=-0.17,
            delta_f=0.125,
            f_lower=19.0,
            f_ref=245.0,
            distance=407.0,
            inclination=0.68,
            coa_phase=2.17,
            long_asc_nodes=0.21,
        ),
        # The fixed cutoff lies just above an FFT power-of-two boundary.
        dict(
            mass1=85.65979841519399,
            mass2=33.13312144288014,
            spin1z=0.6483484819527008,
            spin2z=0.4788873806305084,
            delta_f=0.5,
            f_lower=18.448799600350895,
            f_ref=95.75337457534539,
            distance=192.2997726386339,
            inclination=2.9033595924330404,
            coa_phase=1.0479794788728685,
            long_asc_nodes=-0.8313650570795871,
        ),
    ],
)
def test_imrphenomc_torch_matches_lalsimulation(params):
    reference, actual, reference_metadata, actual_metadata = _run_case(params)
    assert actual_metadata == reference_metadata

    for expected, result in zip(reference, actual):
        np.testing.assert_array_equal(result == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 3.0e-10


def test_imrphenomc_native_emits_no_runtime_warnings():
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=100.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_case(params)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == []


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"f_ref": 100.0}, True),
        ({"lambda1": 0.0}, True),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"dalpha1": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"spin_order": 2}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "IMRPhenomD"}, False),
    ],
)
def test_imrphenomc_native_support_boundary(params, expected):
    assert imrphenomc_native_supported(params) is expected


def test_imrphenomc_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    params = dict(
        mass1=35.0,
        mass2=28.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.25,
        f_lower=20.0,
        f_ref=30.0,
        distance=500.0,
        inclination=0.4,
        coa_phase=1.1,
        long_asc_nodes=0.37,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomC", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomc_torch as imrphenomc_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenomc_mod.imrphenomc_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomC called lalsimulation")

    monkeypatch.setattr(
        imrphenomc_mod, "imrphenomc_fd_torch", recording_native
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomC", **params)

    assert calls == 1
    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < 3.0e-10


def test_imrphenomc_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
        dchi3=0.1,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomC", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomc_torch as imrphenomc_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomC parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomc_mod, "imrphenomc_fd_torch", unexpected_native
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform(approximant="IMRPhenomC", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomc_public_native_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=35.0,
        mass2=28.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.5,
        f_lower=20.0,
        distance=500.0,
        inclination=0.4,
        coa_phase=1.1,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant="IMRPhenomC", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, cross = get_fd_waveform(approximant="IMRPhenomC", **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for series in (actual, cross):
        assert series._data.tensor.device.type == device_name
        assert series._data.tensor.dtype == expected_dtype

    actual_array = actual.numpy()
    np.testing.assert_array_equal(
        actual_array == 0.0, reference_array == 0.0
    )
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 2.0e-3 if device_name == "mps" else 3.0e-10
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mass1": 0.0}, "masses must be positive"),
        ({"spin1z": 1.1}, "spins must be between"),
        ({"distance": 0.0}, "distance must be positive"),
        ({"f_final": 10.0}, "f_final is <= f_lower"),
        ({"mass1": 100.0, "mass2": 4.0}, "mass ratio"),
    ],
)
def test_imrphenomc_native_rejects_invalid_inputs(
    changes, message, preserve_scheme
):
    params = dict(
        approximant="IMRPhenomC",
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
    )
    params.update(changes)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomc_fd_torch(**params)
