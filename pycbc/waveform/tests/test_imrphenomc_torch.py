import os
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomc_torch import (  # noqa: E402
    imrphenomc_fd_sequence_torch,
    imrphenomc_fd_torch,
    imrphenomc_native_supported,
    imrphenomc_sequence_native_supported,
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
    assert imrphenomc_sequence_native_supported(params) is expected


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


SEQUENCE_PARAMS = dict(
    mass1=35.0,
    mass2=28.0,
    spin1z=0.2,
    spin2z=-0.1,
    f_ref=30.0,
    distance=500.0,
    inclination=0.4,
    coa_phase=1.1,
)


def test_imrphenomc_sequence_converges_to_grid_and_dispatches_native(
    monkeypatch,
    preserve_scheme,
):
    sample_points = [20.0, 23.5, 80.0, 150.0, 250.0]
    delta_f = 0.125
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    regular = get_fd_waveform(
        approximant="IMRPhenomC",
        delta_f=delta_f,
        f_lower=20.0,
        long_asc_nodes=0.0,
        **SEQUENCE_PARAMS,
    )
    indices = [int(frequency / delta_f) for frequency in sample_points]
    expected = tuple(series.numpy()[indices] for series in regular)

    import pycbc.waveform.imrphenomc_torch as imrphenomc_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenomc_mod.imrphenomc_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomC sequence called LAL")

    monkeypatch.setattr(
        imrphenomc_mod,
        "imrphenomc_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomC",
        sample_points=sample_points,
        long_asc_nodes=0.0,
        **SEQUENCE_PARAMS,
    )

    assert native_calls == 1
    for expected_array, actual_array in zip(expected, actual):
        assert actual_array._data.tensor.device.type == "cpu"
        assert actual_array._data.tensor.dtype == torch.complex128
        relative_error = np.linalg.norm(
            actual_array.numpy() - expected_array
        ) / np.linalg.norm(expected_array)
        assert relative_error < 2.0e-9


def test_imrphenomc_sequence_conventions(monkeypatch, preserve_scheme):
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    sample_points = [250.0, 20.0, 80.0, 10000.0]
    unordered = get_fd_waveform_sequence(
        approximant="IMRPhenomC",
        sample_points=sample_points,
        long_asc_nodes=0.73,
        **SEQUENCE_PARAMS,
    )
    sorted_points = sorted(sample_points)
    ordered = get_fd_waveform_sequence(
        approximant="IMRPhenomC",
        sample_points=sorted_points,
        long_asc_nodes=0.0,
        **SEQUENCE_PARAMS,
    )
    order = [sorted_points.index(frequency) for frequency in sample_points]
    for unordered_array, ordered_array in zip(unordered, ordered):
        torch.testing.assert_close(
            unordered_array._data.tensor,
            ordered_array._data.tensor[order],
            rtol=1.0e-13,
            atol=0.0,
        )
        assert unordered_array[-1] == 0.0


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "spin2z"),
    [
        (35.0, 28.0, 0.2, -0.1),
        (67.0, 43.5, 0.7, -0.17),
        (20.0, 20.0, -0.8, -0.8),
    ],
)
def test_imrphenomc_sequence_time_correction_is_phase_derivative(
    mass1,
    mass2,
    spin1z,
    spin2z,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomc_torch as imrphenomc_mod

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = dict(
        approximant="IMRPhenomC",
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=500.0,
    )
    inputs = imrphenomc_mod._imrphenomc_inputs(params, sequence=True)
    coefficients = imrphenomc_mod._imrphenomc_coefficients(inputs)
    step = coefficients.f_rd * 1.0e-6
    frequencies = torch.tensor(
        [coefficients.f_rd - step, coefficients.f_rd + step],
        dtype=torch.float64,
    )
    _, phase = imrphenomc_mod._imrphenomc_components(
        inputs,
        coefficients,
        frequencies,
    )
    finite_difference = float((phase[1] - phase[0]) / (2.0 * step))
    analytic = -2.0 * np.pi * (
        imrphenomc_mod._imrphenomc_ringdown_time_correction(
            inputs,
            coefficients,
        )
    )
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=1.0e-7,
        atol=1.0e-10,
    )


def test_imrphenomc_sequence_unsupported_options_reach_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomc_torch as imrphenomc_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomC sequence reached Torch")

    lal_calls = 0

    def expected_lal(*_args, **_kwargs):
        nonlocal lal_calls
        lal_calls += 1
        raise RuntimeError("IMRPhenomC sequence LAL fallback reached")

    monkeypatch.setattr(
        imrphenomc_mod,
        "imrphenomc_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        expected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        get_fd_waveform_sequence(
            approximant="IMRPhenomC",
            sample_points=[20.0, 30.0, 50.0],
            dchi3=0.1,
            **SEQUENCE_PARAMS,
        )

    assert lal_calls == 1


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomc_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    sample_points = [20.0, 27.5, 80.0, 250.0, 10000.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomC",
        sample_points=sample_points,
        **SEQUENCE_PARAMS,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    _activate_scheme(_scheme.TorchScheme(device_name))

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomC sequence copied through NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        actual = get_fd_waveform_sequence(
            approximant="IMRPhenomC",
            sample_points=sample_points,
            **SEQUENCE_PARAMS,
        )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = 3.0e-3 if device_name == "mps" else 1.0e-11
    for reference_array, actual_array in zip(reference_arrays, actual):
        tensor = actual_array._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        expected = torch.as_tensor(
            reference_array,
            dtype=expected_dtype,
            device=tensor.device,
        )
        torch.testing.assert_close(
            tensor,
            expected,
            rtol=tolerance,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([[20.0, 30.0]], "non-empty vector"),
        ([20.0, float("nan")], "finite"),
        ([20.0, 0.0], "positive"),
        ([20.0, -30.0], "positive"),
    ],
)
def test_imrphenomc_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomc_fd_sequence_torch(
            approximant="IMRPhenomC",
            sample_points=sample_points,
            **SEQUENCE_PARAMS,
        )
