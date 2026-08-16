import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.imrphenompv2_torch import (  # noqa: E402
    imrphenompv2_native_supported,
)


_ENV_KEYS = ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMPV2_NATIVE")


@pytest.fixture
def preserve_scheme():
    """Restore PyCBC's process-wide scheme singleton after a test."""

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
        os.environ["PYCBC_IMRPHENOMPV2_NATIVE"] = "0"
        _activate_scheme(_scheme.CPUScheme())
        reference = get_fd_waveform(approximant="IMRPhenomPv2", **params)
        reference_arrays = tuple(series.numpy().copy() for series in reference)
        reference_metadata = tuple(
            (len(series), series.delta_f, float(series.epoch))
            for series in reference
        )

        os.environ["PYCBC_IMRPHENOMPV2_NATIVE"] = "1"
        _activate_scheme(_scheme.TorchScheme("cpu"))
        actual = get_fd_waveform(approximant="IMRPhenomPv2", **params)
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
            mass1=40.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.05,
            spin2z=-0.2,
            distance=500.0,
            inclination=0.7,
            coa_phase=1.2,
            long_asc_nodes=0.3,
            delta_f=0.5,
            f_lower=20.0,
            f_final=512.0,
            f_ref=30.0,
        ),
        # Exercises component reordering, f_ref=0, and non-bin frequencies.
        dict(
            mass1=12.0,
            mass2=35.0,
            spin1x=0.15,
            spin1y=-0.25,
            spin1z=0.4,
            spin2x=0.05,
            spin2y=0.2,
            spin2z=-0.3,
            distance=320.0,
            inclination=1.1,
            coa_phase=0.0,
            long_asc_nodes=-0.4,
            delta_f=0.25,
            f_lower=17.3,
            f_final=900.3,
            f_ref=0.0,
        ),
        # The aligned-spin limit takes a special orientation-angle branch.
        dict(
            mass1=30.0,
            mass2=30.0,
            spin1z=0.2,
            spin2z=-0.1,
            distance=800.0,
            inclination=0.2,
            coa_phase=2.1,
            delta_f=0.5,
            f_lower=20.0,
            f_ref=20.0,
        ),
        dict(
            mass1=85.0,
            mass2=9.0,
            spin1x=0.6,
            spin1y=0.1,
            spin1z=-0.2,
            spin2x=-0.1,
            spin2y=0.2,
            spin2z=0.3,
            distance=1000.0,
            inclination=2.2,
            coa_phase=0.4,
            long_asc_nodes=0.7,
            delta_f=0.25,
            f_lower=15.0,
            f_ref=80.0,
        ),
    ],
)
def test_imrphenompv2_torch_matches_lalsimulation(params):
    reference, actual, reference_metadata, actual_metadata = _run_case(params)
    assert actual_metadata == reference_metadata

    for expected, result in zip(reference, actual):
        np.testing.assert_array_equal(result == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-9


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"spin1x": 0.2, "spin2y": -0.1}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"f_ref": 100.0}, True),
        ({"frame_axis": 0}, True),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        # LAL's PhenomPv2 driver accepts but ignores PN phase-order flags.
        ({"phase_order": 2}, True),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"modes_choice": 1}, False),
        ({"side_bands": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "IMRPhenomPv2_NRTidal"}, False),
    ],
)
def test_imrphenompv2_native_support_boundary(params, expected):
    assert imrphenompv2_native_supported(params) is expected


def test_imrphenompv2_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    params = dict(
        mass1=35.0,
        mass2=22.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        long_asc_nodes=0.37,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomPv2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenompv2_torch as imrphenompv2_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenompv2_mod.imrphenompv2_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomPv2 called lalsimulation")

    monkeypatch.setattr(
        imrphenompv2_mod,
        "imrphenompv2_fd_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomPv2", **params)

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
        assert relative_error < 1.0e-9


def test_imrphenompv2_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=500.0,
        delta_f=0.5,
        f_lower=20.0,
        dchi3=0.01,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomPv2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenompv2_torch as imrphenompv2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomPv2 parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenompv2_mod,
        "imrphenompv2_fd_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform(approximant="IMRPhenomPv2", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenompv2_public_native_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomPv2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(approximant="IMRPhenomPv2", **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        tolerance = 3.0e-3 if device_name == "mps" else 1.0e-9
        assert relative_error < tolerance
