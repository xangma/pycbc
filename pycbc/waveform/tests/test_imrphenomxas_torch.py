import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxas_torch import (  # noqa: E402
    imrphenomxas_native_supported,
    imrphenomxas_sequence_native_supported,
)


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


CASES = [
    dict(
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
        mass1=67.0,
        mass2=43.5,
        spin1z=0.9,
        spin2z=-0.17,
        delta_f=0.5,
        f_lower=19.0,
        f_ref=245.0,
        distance=407.0,
        inclination=0.8,
        coa_phase=0.6,
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
    ),
]

TIDAL_CASES = [
    dict(
        mass1=1.4,
        mass2=1.3,
        spin1z=0.03,
        spin2z=-0.02,
        lambda1=400.0,
        lambda2=700.0,
        delta_f=0.5,
        f_lower=20.0,
        f_final=2048.0,
        f_ref=30.0,
        distance=100.0,
        inclination=0.4,
    ),
    dict(
        mass1=1.2,
        mass2=1.6,
        spin1z=-0.04,
        spin2z=0.05,
        lambda1=800.0,
        lambda2=300.0,
        dquad_mon1=3.0,
        dquad_mon2=4.0,
        delta_f=0.25,
        f_lower=19.3,
        f_final=1024.0,
        f_ref=0.0,
        distance=130.0,
        inclination=0.8,
        long_asc_nodes=0.2,
    ),
    dict(
        mass1=1.7,
        mass2=1.1,
        spin1z=0.1,
        spin2z=-0.03,
        lambda1=0.0,
        lambda2=0.0,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        distance=90.0,
        inclination=0.3,
    ),
]

SEQUENCE_CASES = [
    (
        CASES[0],
        [20.0, 23.5, 30.0, 45.0, 100.0, 250.0, 400.0, 10000.0],
    ),
    (
        CASES[3],
        [17.3, 400.0, 22.0, 150.0],
    ),
]

TIDAL_SEQUENCE_CASES = [
    (
        TIDAL_CASES[0],
        [20.0, 23.5, 30.0, 50.0, 100.0, 500.0, 1000.0, 2048.0, 10000.0],
    ),
    (
        TIDAL_CASES[1],
        [19.3, 1500.0, 30.0, 1024.0],
    ),
]


def _sequence_params(params):
    return {
        key: value
        for key, value in params.items()
        if key not in {"delta_f", "f_lower", "f_final"}
    }


@pytest.mark.parametrize("params", CASES)
def test_imrphenomxas_matches_lal(
    params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128

        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < 1.0e-10


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
@pytest.mark.parametrize("params", TIDAL_CASES)
def test_imrphenomxas_nrtidal_matches_lal(
    approximant, params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant=approximant, **params)

    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128

        result_array = result.numpy()
        relative_error = np.linalg.norm(
            result_array - expected_array
        ) / np.linalg.norm(expected_array)
        assert relative_error < 5.0e-6

        # XHM's public LAL route is multibanded, so interpolation residuals
        # dominate pointwise relative error deep in the Planck-taper tail.
        significant = np.abs(expected_array) > (
            1.0e-4 * np.max(np.abs(expected_array))
        )
        point_error = np.max(
            np.abs(result_array[significant] - expected_array[significant])
            / np.abs(expected_array[significant])
        )
        assert point_error < 1.0e-4


def test_imrphenomxas_public_dispatch_does_not_call_lal(
    monkeypatch, preserve_scheme
):
    params = CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    native = xas_mod.imrphenomxas_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXAS called lalsimulation")

    monkeypatch.setattr(xas_mod, "imrphenomxas_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1.0e-10, atol=0.0
        )


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
def test_imrphenomxas_nrtidal_dispatch_does_not_call_lal(
    approximant, monkeypatch, preserve_scheme
):
    params = TIDAL_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} called LAL")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant=approximant, **params)

    for expected, result in zip(reference_arrays, actual):
        relative_error = np.linalg.norm(
            result.numpy() - expected
        ) / np.linalg.norm(expected)
        assert relative_error < 5.0e-6


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"eccentricity": 0.01}, False),
        ({"phase_order": 7}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "data.h5"}, False),
        ({"approximant": "IMRPhenomXP"}, False),
    ],
)
def test_imrphenomxas_native_support_boundary(changes, expected):
    params = {"approximant": "IMRPhenomXAS", **changes}
    assert imrphenomxas_native_supported(params) is expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"lambda1": 400.0, "lambda2": 700.0}, True),
        ({"lambda1": 400.0, "dquad_mon1": 3.0}, True),
        ({"lambda1": -1.0}, False),
        ({"lambda1": float("nan")}, False),
        ({"dquad_mon1": -1.0}, False),
        ({"lambda_octu1": 10.0}, False),
        ({"mode_array": [(2, 2)]}, False),
    ],
)
@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
def test_imrphenomxas_nrtidal_native_support_boundary(
    approximant, changes, expected
):
    params = {"approximant": approximant, **changes}
    assert imrphenomxas_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "extra", "last_nonzero"),
    [
        ("IMRPhenomXAS", {}, 4096),
        (
            "IMRPhenomXAS_NRTidalv2",
            {"lambda1": 400.0, "lambda2": 700.0},
            4095,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            {"lambda1": 400.0, "lambda2": 700.0},
            4095,
        ),
    ],
)
def test_imrphenomxas_power_of_two_layout_boundary(
    approximant,
    extra,
    last_nonzero,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=1.4,
        mass2=1.3,
        spin1z=0.03,
        spin2z=-0.02,
        delta_f=0.25,
        f_lower=20.0,
        f_final=1024.1,
        f_ref=30.0,
        distance=100.0,
        **extra,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    hp, _ = get_fd_waveform(approximant=approximant, **params)

    assert len(hp) == 4097
    assert np.flatnonzero(hp.numpy())[-1] == last_nonzero


def test_imrphenomxas_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = {**CASES[0], "dchi3": 0.1}
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomXAS parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xas_mod, "imrphenomxas_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    ("approximant", "params", "cpu_tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], 1.0e-10),
        ("IMRPhenomXAS_NRTidalv2", TIDAL_CASES[0], 5.0e-6),
        ("IMRPhenomXAS_NRTidalv3", TIDAL_CASES[0], 5.0e-6),
    ],
)
def test_imrphenomxas_stays_on_requested_device(
    device_name,
    approximant,
    params,
    cpu_tolerance,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant=approximant, **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant=approximant, **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else cpu_tolerance
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        ("IMRPhenomXAS", CASES[0]),
        ("IMRPhenomXAS_NRTidalv2", TIDAL_CASES[0]),
        ("IMRPhenomXAS_NRTidalv3", TIDAL_CASES[0]),
    ],
)
def test_imrphenomxas_native_avoids_host_transfer(
    approximant, params, monkeypatch, preserve_scheme
):
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXAS transferred data to NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    with torch.no_grad():
        hp, hc = get_fd_waveform(approximant=approximant, **params)

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)


@pytest.mark.parametrize(("params", "sample_points"), SEQUENCE_CASES)
def test_imrphenomxas_sequence_matches_lal(
    params, sample_points, monkeypatch, preserve_scheme
):
    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected == 0.0,
        )
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-10


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
@pytest.mark.parametrize(("params", "sample_points"), TIDAL_SEQUENCE_CASES)
def test_imrphenomxas_nrtidal_sequence_matches_lal(
    approximant, params, sample_points, monkeypatch, preserve_scheme
):
    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected == 0.0,
        )
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-9


@pytest.mark.parametrize(
    ("approximant", "params", "sample_points", "tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1], 1.0e-10),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
    ],
)
def test_imrphenomxas_sequence_public_dispatch_does_not_call_lal(
    approximant,
    params,
    sample_points,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    native = xas_mod.imrphenomxas_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} sequence called LAL")

    monkeypatch.setattr(
        xas_mod,
        "imrphenomxas_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        assert isinstance(result._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=tolerance, atol=0.0
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"approximant": "IMRPhenomXAS"}, True),
        ({"approximant": "IMRPhenomXAS_NRTidalv2"}, True),
        ({"approximant": "IMRPhenomXAS_NRTidalv3"}, True),
        (
            {"approximant": "IMRPhenomXAS_NRTidalv3", "lambda1": -1.0},
            False,
        ),
        ({"approximant": "IMRPhenomXP"}, False),
        ({"dchi3": 0.1}, False),
        ({"lambda1": 100.0}, False),
    ],
)
def test_imrphenomxas_sequence_native_support_boundary(changes, expected):
    assert imrphenomxas_sequence_native_supported(changes) is expected


def test_imrphenomxas_sequence_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = {**_sequence_params(CASES[0]), "dchi3": 0.1}
    sample_points = SEQUENCE_CASES[0][1]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported XAS sequence parameters reached Torch")

    lal_generator = (
        waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xas_mod,
        "imrphenomxas_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


def test_imrphenomxas_sequence_lal_fallback_supports_mps(
    monkeypatch, preserve_scheme
):
    if not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")

    params = {**_sequence_params(CASES[0]), "dchi3": 0.1}
    sample_points = SEQUENCE_CASES[0][1]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("mps"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    for expected, actual in zip(reference_arrays, fallback):
        assert actual._data.tensor.device.type == "mps"
        assert actual._data.tensor.dtype == torch.complex64
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-6, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    ("approximant", "params", "sample_points", "cpu_tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1], 1.0e-10),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
    ],
)
def test_imrphenomxas_sequence_stays_on_requested_device(
    device_name,
    approximant,
    params,
    sample_points,
    cpu_tolerance,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else cpu_tolerance
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "params", "sample_values"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1]),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
        ),
    ],
)
def test_imrphenomxas_sequence_native_avoids_host_transfer(
    approximant,
    params,
    sample_values,
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXAS sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    sample_points = Array(sample_values)
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=sample_points,
            **_sequence_params(params),
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
