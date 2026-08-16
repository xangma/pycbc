import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomnsbh_torch import (  # noqa: E402
    imrphenomnsbh_fd_sequence_torch,
    imrphenomnsbh_fd_torch,
    imrphenomnsbh_native_supported,
    imrphenomnsbh_sequence_native_supported,
)


_REGULAR_PARAMS = dict(
    mass1=7.0,
    mass2=1.4,
    spin1z=0.2,
    spin2z=0.0,
    lambda1=0.0,
    lambda2=500.0,
    delta_f=0.5,
    f_lower=20.0,
    f_final=1024.0,
    f_ref=20.0,
    distance=100.0,
    inclination=0.4,
    coa_phase=0.7,
    long_asc_nodes=0.31,
)

_SEQUENCE_PARAMS = {
    key: value
    for key, value in _REGULAR_PARAMS.items()
    if key not in ("delta_f", "f_lower", "f_final")
}


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


def _assert_waveform_parity(reference, actual, tolerance=3.0e-10):
    for expected, result in zip(reference, actual):
        expected_array = (
            expected if isinstance(expected, np.ndarray) else expected.numpy()
        )
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        assert nonzero.any(), "waveform contains no non-zero samples"
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < tolerance


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {
            "mass1": 5.0,
            "spin1z": 0.8,
            "lambda2": 1000.0,
            "f_ref": 30.0,
            "inclination": 0.9,
        },
        {
            "spin1z": 0.5,
            "lambda2": 0.0,
            "f_ref": 0.0,
            "coa_phase": 1.1,
        },
    ],
)
def test_imrphenomnsbh_regular_public_parity(
    changes,
    monkeypatch,
    preserve_scheme,
):
    params = dict(_REGULAR_PARAMS)
    params.update(changes)

    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomNSBH", **params)

    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomNSBH", **params)

    for expected, result in zip(reference, actual):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
    _assert_waveform_parity(reference, actual)


@pytest.mark.parametrize(
    "sample_points",
    [
        [20.0, 20.5, 30.0, 55.25, 100.0, 256.0, 1024.0],
        [20.0, 80.0, 35.0, 400.0, 250.0, 1024.0],
    ],
)
def test_imrphenomnsbh_sequence_public_parity(
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomNSBH",
        sample_points=sample_points,
        **_SEQUENCE_PARAMS,
    )

    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomNSBH",
        sample_points=sample_points,
        **_SEQUENCE_PARAMS,
    )

    assert all(result._data.tensor.device.type == "cpu" for result in actual)
    _assert_waveform_parity(reference, actual)


def test_imrphenomnsbh_public_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomnsbh_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    calls = {"regular": 0, "sequence": 0}
    regular_native = native_module.imrphenomnsbh_fd_torch
    sequence_native = native_module.imrphenomnsbh_fd_sequence_torch

    def recording_regular(**params):
        calls["regular"] += 1
        return regular_native(**params)

    def recording_sequence(**params):
        calls["sequence"] += 1
        return sequence_native(**params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomNSBH called lalsimulation")

    monkeypatch.setattr(
        native_module,
        "imrphenomnsbh_fd_torch",
        recording_regular,
    )
    monkeypatch.setattr(
        native_module,
        "imrphenomnsbh_fd_sequence_torch",
        recording_sequence,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    get_fd_waveform(approximant="IMRPhenomNSBH", **_REGULAR_PARAMS)
    get_fd_waveform_sequence(
        approximant="IMRPhenomNSBH",
        sample_points=[20.0, 30.0, 100.0],
        **_SEQUENCE_PARAMS,
    )

    assert calls == {"regular": 1, "sequence": 1}


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"lambda2": 0.0}, True),
        ({"lambda2": 5000.0}, True),
        ({"approximant": "IMRPhenomD_NRTidalv2"}, False),
        ({"lambda1": 10.0}, False),
        ({"lambda2": -1.0}, False),
        ({"lambda2": 5000.1}, False),
        ({"spin1x": 0.1}, False),
        ({"dquad_mon2": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"phase_order": 7}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"dchi3": 0.1}, False),
    ],
)
def test_imrphenomnsbh_native_support_boundary(changes, expected):
    params = {"approximant": "IMRPhenomNSBH", **changes}
    assert imrphenomnsbh_native_supported(params) is expected
    assert imrphenomnsbh_sequence_native_supported(params) is expected


def test_imrphenomnsbh_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomnsbh_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomNSBH reached Torch")

    lal_calls = 0

    def expected_lal(*_args, **_kwargs):
        nonlocal lal_calls
        lal_calls += 1
        raise RuntimeError("IMRPhenomNSBH LAL fallback reached")

    monkeypatch.setattr(
        native_module,
        "imrphenomnsbh_fd_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        expected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        get_fd_waveform(
            approximant="IMRPhenomNSBH",
            dchi3=0.1,
            **_REGULAR_PARAMS,
        )
    assert lal_calls == 1


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomnsbh_native_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomNSBH",
        **_REGULAR_PARAMS,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMNSBH_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    with torch.no_grad():
        actual = get_fd_waveform(
            approximant="IMRPhenomNSBH",
            **_REGULAR_PARAMS,
        )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for result in actual:
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
    tolerance = 3.0e-3 if device_name == "mps" else 3.0e-10
    _assert_waveform_parity(reference_arrays, actual, tolerance=tolerance)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mass1": 0.0}, "masses must be positive"),
        ({"mass1": 1.3}, "mass1 must be the black-hole mass"),
        ({"mass1": 150.0}, "mass ratio"),
        ({"mass2": 3.1}, "must not exceed 3"),
        ({"spin1z": 1.1}, "spins must be between"),
        ({"lambda2": 5000.1}, "not supported"),
        ({"distance": 0.0}, "distance must be positive"),
        ({"f_final": 10.0}, "f_final is below f_lower"),
    ],
)
def test_imrphenomnsbh_native_rejects_invalid_inputs(
    changes,
    message,
    preserve_scheme,
):
    params = dict(_REGULAR_PARAMS)
    params.update(changes)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomnsbh_fd_torch(
            approximant="IMRPhenomNSBH",
            **params,
        )


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([[20.0, 30.0]], "non-empty vector"),
        ([20.0, float("nan")], "finite"),
        ([20.0, 0.0], "positive"),
        ([30.0, 20.0], "below the first sample"),
    ],
)
def test_imrphenomnsbh_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomnsbh_fd_sequence_torch(
            approximant="IMRPhenomNSBH",
            sample_points=sample_points,
            **_SEQUENCE_PARAMS,
        )
