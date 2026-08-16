import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenompv2_torch import (  # noqa: E402
    imrphenompv2_fd_sequence_torch,
    imrphenompv2_native_supported,
    imrphenompv2_sequence_native_supported,
)


_ENV_KEYS = ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMPV2_NATIVE")

_TIDAL_CASES = (
    (
        "IMRPhenomPv2_NRTidal",
        dict(
            mass1=1.4,
            mass2=1.2,
            spin1x=0.03,
            spin1y=-0.02,
            spin1z=0.04,
            spin2x=-0.02,
            spin2y=0.01,
            spin2z=-0.03,
            lambda1=400.0,
            lambda2=800.0,
            distance=100.0,
            inclination=0.7,
            coa_phase=0.4,
            long_asc_nodes=0.3,
            f_ref=30.0,
        ),
    ),
    (
        "IMRPhenomPv2_NRTidalv2",
        dict(
            mass1=1.55,
            mass2=1.15,
            spin1x=-0.02,
            spin1y=0.04,
            spin1z=-0.05,
            spin2x=0.03,
            spin2y=-0.01,
            spin2z=0.08,
            lambda1=300.0,
            lambda2=900.0,
            distance=120.0,
            inclination=1.1,
            coa_phase=0.2,
            long_asc_nodes=-0.25,
            f_ref=0.0,
        ),
    ),
)


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


@pytest.mark.parametrize(("approximant", "params"), _TIDAL_CASES)
def test_imrphenompv2_nrtidal_matches_lalsimulation(
    approximant,
    params,
    monkeypatch,
    preserve_scheme,
):
    regular_params = dict(
        params,
        delta_f=1.0,
        f_lower=20.0,
    )
    if approximant.endswith("v2"):
        regular_params["f_final"] = 2048.0
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant=approximant,
        **regular_params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native tidal IMRPhenomPv2 called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant=approximant,
        **regular_params,
    )

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
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
        ({"approximant": "IMRPhenomPv2_NRTidal"}, True),
        ({"approximant": "IMRPhenomPv2_NRTidalv2"}, True),
        (
            {
                "approximant": "IMRPhenomPv2_NRTidalv2",
                "lambda1": -1.0,
            },
            False,
        ),
        (
            {
                "approximant": "IMRPhenomPv2_NRTidal",
                "lambda2": float("inf"),
            },
            False,
        ),
        (
            {
                "approximant": "IMRPhenomPv2_NRTidalv2",
                "dquad_mon1": 1.0,
            },
            False,
        ),
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


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenompv2_nrtidal_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    approximant, tidal_params = _TIDAL_CASES[1]
    params = dict(
        tidal_params,
        delta_f=4.0,
        f_lower=20.0,
        f_final=2048.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native tidal IMRPhenomPv2 called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(approximant=approximant, **params)

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
        tolerance = 5.0e-3 if device_name == "mps" else 1.0e-9
        assert relative_error < tolerance


@pytest.mark.parametrize(
    ("params", "sample_points"),
    [
        (
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
                long_asc_nodes=0.73,
                f_ref=30.0,
            ),
            [20.0, 23.5, 30.0, 45.0, 100.0, 400.0, 700.0, 900.0],
        ),
        (
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
                f_ref=0.0,
            ),
            [17.3, 22.0, 150.0, 400.0, 850.0, 1000.0],
        ),
        (
            dict(
                mass1=30.0,
                mass2=30.0,
                spin1z=0.2,
                spin2z=-0.1,
                distance=800.0,
                inclination=0.2,
                coa_phase=2.1,
                f_ref=20.0,
            ),
            [20.0, 40.0, 100.0, 300.0, 600.0, 800.0],
        ),
    ],
)
def test_imrphenompv2_sequence_matches_lalsimulation(
    params,
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomPv2 sequence called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-9


@pytest.mark.parametrize(("approximant", "params"), _TIDAL_CASES)
def test_imrphenompv2_nrtidal_sequence_matches_lalsimulation(
    approximant,
    params,
    monkeypatch,
    preserve_scheme,
):
    sample_points = [
        20.0,
        23.5,
        30.0,
        100.0,
        500.0,
        1000.0,
        1400.0,
        1800.0,
        2200.0,
    ]
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(
            "native tidal IMRPhenomPv2 sequence called lalsimulation"
        )

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-9


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": float("nan")}, True),
        ({"dchi3": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"approximant": "IMRPhenomPv2_NRTidal"}, True),
        ({"approximant": "IMRPhenomPv2_NRTidalv2"}, True),
    ],
)
def test_imrphenompv2_sequence_native_support_boundary(changes, expected):
    assert imrphenompv2_sequence_native_supported(changes) is expected


@pytest.mark.parametrize(
    ("sample_points", "match"),
    [
        ([], "non-empty vector"),
        ([20.0, 20.0], "strictly increasing"),
        ([30.0, 20.0], "strictly increasing"),
        ([0.0, 20.0], "positive"),
        ([20.0, float("nan")], "finite"),
        ([1000.0, 1200.0], "fCut must exceed"),
    ],
)
def test_imrphenompv2_sequence_validates_frequencies(
    sample_points,
    match,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=match):
        imrphenompv2_fd_sequence_torch(
            approximant="IMRPhenomPv2",
            sample_points=sample_points,
            mass1=40.0,
            mass2=20.0,
            distance=500.0,
            f_ref=0.0,
        )


def test_imrphenompv2_sequence_dispatch_avoids_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

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
        f_ref=0.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array([20.0, 30.0, 100.0, 400.0, 800.0])

    import pycbc.waveform.imrphenompv2_torch as imrphenompv2_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenompv2_mod.imrphenompv2_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomPv2 sequence called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomPv2 sequence transferred to NumPy")

    monkeypatch.setattr(
        imrphenompv2_mod,
        "imrphenompv2_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant="IMRPhenomPv2",
            sample_points=sample_points,
            **params,
        )

    assert calls == 1
    assert len(hp) == len(sample_points)
    assert len(hc) == len(sample_points)
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


def test_imrphenompv2_sequence_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=500.0,
        f_ref=20.0,
        dchi3=0.01,
    )
    sample_points = [20.0, 30.0, 100.0, 400.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenompv2_torch as imrphenompv2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomPv2 sequence reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenompv2_mod,
        "imrphenompv2_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenompv2_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
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
        long_asc_nodes=0.37,
        f_ref=0.0,
    )
    sample_points = [20.0, 30.0, 100.0, 400.0, 800.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMPV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomPv2",
        sample_points=sample_points,
        **params,
    )

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
