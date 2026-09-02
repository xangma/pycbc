import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
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


_ENV_KEYS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMNSBH_NATIVE",
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


def _clear_native_flags(monkeypatch):
    """Remove native switches so the registry default is exercised."""

    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


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

    _clear_native_flags(monkeypatch)
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

    _clear_native_flags(monkeypatch)
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
    _clear_native_flags(monkeypatch)
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
def test_imrphenomnsbh_native_support_boundary(
    changes,
    expected,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {"approximant": "IMRPhenomNSBH", **changes}
    assert imrphenomnsbh_native_supported(params) is expected
    assert imrphenomnsbh_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "imrphenomnsbh_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "imrphenomnsbh_fd_sequence_torch",
        ),
    ),
)
def test_imrphenomnsbh_unsupported_options_use_lal_fallback(
    interface,
    lal_name,
    native_name,
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
        native_name,
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        lal_name,
        expected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))

    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        if interface == "regular":
            get_fd_waveform(
                approximant="IMRPhenomNSBH",
                dchi3=0.1,
                **_REGULAR_PARAMS,
            )
        else:
            get_fd_waveform_sequence(
                approximant="IMRPhenomNSBH",
                sample_points=[20.0, 30.0, 100.0],
                dchi3=0.1,
                **_SEQUENCE_PARAMS,
            )
    assert lal_calls == 1


@pytest.mark.parametrize(
    ("interface", "native_name"),
    (
        ("regular", "imrphenomnsbh_fd_torch"),
        ("sequence", "imrphenomnsbh_fd_sequence_torch"),
    ),
)
@pytest.mark.parametrize(
    "opt_out_flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMNSBH_NATIVE"),
)
def test_imrphenomnsbh_default_native_opt_out_uses_lal(
    interface,
    native_name,
    opt_out_flag,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomnsbh_torch as native_module

    def unexpected_native(**_params):
        raise AssertionError("opted-out IMRPhenomNSBH reached Torch")

    monkeypatch.setattr(native_module, native_name, unexpected_native)
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv(opt_out_flag, "0")

    _activate_scheme(_scheme.CPUScheme())
    if interface == "regular":
        reference = get_fd_waveform(
            approximant="IMRPhenomNSBH",
            **_REGULAR_PARAMS,
        )
    else:
        reference = get_fd_waveform_sequence(
            approximant="IMRPhenomNSBH",
            sample_points=[20.0, 30.0, 100.0],
            **_SEQUENCE_PARAMS,
        )
    reference = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    if interface == "regular":
        result = get_fd_waveform(
            approximant="IMRPhenomNSBH",
            **_REGULAR_PARAMS,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomNSBH",
            sample_points=[20.0, 30.0, 100.0],
            **_SEQUENCE_PARAMS,
        )

    for expected, series in zip(reference, result):
        assert series._data.tensor.device.type == "cpu"
        np.testing.assert_array_equal(series.numpy(), expected)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
@pytest.mark.parametrize(
    ("interface", "changes", "expected"),
    (
        ("regular", {"delta_f": 1.0, "f_lower": 12.0}, False),
        ("regular", {"delta_f": 1.0, "f_lower": 13.0}, True),
        ("sequence", {"sample_points": [20.0, 12.0, 100.0]}, False),
        ("sequence", {"sample_points": [20.0, 13.0, 100.0]}, True),
    ),
)
def test_imrphenomnsbh_mps_support_boundary(
    interface,
    changes,
    expected,
    preserve_scheme,
):
    params = {
        "approximant": "IMRPhenomNSBH",
        "mass1": 7.0,
        "mass2": 1.4,
        **changes,
    }
    _activate_scheme(_scheme.TorchScheme("mps"))
    supported = (
        imrphenomnsbh_native_supported(params)
        if interface == "regular"
        else imrphenomnsbh_sequence_native_supported(params)
    )
    assert supported is expected


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "imrphenomnsbh_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "imrphenomnsbh_fd_sequence_torch",
        ),
    ),
)
def test_imrphenomnsbh_mps_low_frequency_uses_lal_fallback(
    interface,
    lal_name,
    native_name,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomnsbh_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsafe MPS IMRPhenomNSBH reached Torch")

    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

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
    _activate_scheme(_scheme.TorchScheme("mps"))
    if interface == "regular":
        params = dict(_REGULAR_PARAMS)
        params["f_lower"] = 12.0
        result = get_fd_waveform(
            approximant="IMRPhenomNSBH",
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomNSBH",
            sample_points=[12.0, 20.0, 100.0],
            **_SEQUENCE_PARAMS,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "mps" for series in result)


def test_imrphenomnsbh_sequence_avoids_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform_module

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomNSBH called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomNSBH copied through NumPy")

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array([20.0, 30.0, 100.0])
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    result = get_fd_waveform_sequence(
        approximant="IMRPhenomNSBH",
        sample_points=sample_points,
        **_SEQUENCE_PARAMS,
    )

    assert all(len(series) == 3 for series in result)
    assert all(series._data.tensor.device.type == "cpu" for series in result)


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

    _clear_native_flags(monkeypatch)
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
