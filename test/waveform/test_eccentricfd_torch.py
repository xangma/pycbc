import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.eccentricfd_torch import (  # noqa: E402
    eccentricfd_default_native_supported,
    eccentricfd_fd_sequence_torch,
    eccentricfd_fd_torch,
    eccentricfd_native_supported,
    eccentricfd_sequence_native_supported,
)


_BASE_PARAMS = dict(
    approximant="EccentricFD",
    mass1=25.0,
    mass2=10.0,
    distance=275.0,
    inclination=1.1,
    coa_phase=1.3,
    long_asc_nodes=0.8,
    eccentricity=0.15,
    delta_f=0.25,
    f_lower=18.0,
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


def _clear_native_flags(monkeypatch):
    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_ECCENTRICFD_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)


def _reference_and_native(params, monkeypatch, device="cpu"):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="EccentricFD", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device))
    native = get_fd_waveform(approximant="EccentricFD", **params)
    return reference, reference_arrays, native


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            delta_f=0.25,
            f_lower=20.0,
            f_final=300.0,
            distance=400.0,
            eccentricity=0.0,
        ),
        dict(
            mass1=25.0,
            mass2=10.0,
            spin1z=0.7,
            spin2z=-0.4,
            delta_f=0.125,
            f_lower=18.0,
            distance=275.0,
            inclination=1.1,
            coa_phase=1.3,
            long_asc_nodes=0.8,
            eccentricity=0.15,
            phase_order=7,
            f_ref=47.0,
            mean_per_ano=0.9,
        ),
        dict(
            mass1=12.0,
            mass2=7.0,
            delta_f=0.5,
            f_lower=17.3,
            f_final=633.7,
            distance=600.0,
            inclination=2.2,
            coa_phase=-0.7,
            long_asc_nodes=-0.37,
            eccentricity=0.38,
            phase_order=7.9,
        ),
        dict(
            mass1=60.0,
            mass2=40.0,
            delta_f=0.25,
            f_lower=20.0,
            f_final=10.0,
            distance=100.0,
            inclination=0.4,
            long_asc_nodes=0.3,
            eccentricity=0.1,
        ),
    ],
)
def test_eccentricfd_public_torch_matches_lalsimulation(
    params, monkeypatch, preserve_scheme
):
    reference, reference_arrays, native = _reference_and_native(params, monkeypatch)

    for expected, expected_array, actual in zip(reference, reference_arrays, native):
        assert len(actual) == len(expected)
        assert actual.delta_f == expected.delta_f
        assert float(actual.epoch) == float(expected.epoch)
        assert actual._data.tensor.device.type == "cpu"
        assert actual._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(
            actual.numpy() == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        if nonzero.any():
            relative_error = np.linalg.norm(
                actual.numpy()[nonzero] - expected_array[nonzero]
            ) / np.linalg.norm(expected_array[nonzero])
            assert relative_error < 2.0e-11


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"phase_order": 7}, True),
        ({"phase_order": 7.9}, True),
        ({"spin1z": 0.8, "spin2z": -0.7}, True),
        ({"f_ref": 100.0, "mean_per_ano": 0.4}, True),
        ({"eccentricity": 0.6}, True),
        ({"f_final": 10.0}, True),
        ({"phase_order": 6}, False),
        ({"amplitude_order": 0}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 10}, False),
        ({"eccentricity_order": 4}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"frame_axis": 1}, False),
        ({"modes_choice": 1}, False),
        ({"side_bands": 1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"mass1": 0.0}, False),
        ({"distance": 0.0}, False),
        ({"delta_f": 0.0}, False),
        ({"f_lower": 0.0}, False),
        ({"f_final": -1.0}, False),
        ({"eccentricity": float("nan")}, False),
        ({"approximant": "TaylorF2Ecc"}, False),
    ],
)
def test_eccentricfd_native_support_boundary(changes, expected):
    params = dict(_BASE_PARAMS)
    params.update(changes)
    assert eccentricfd_native_supported(params) is expected


def test_eccentricfd_sequence_support_ignores_regular_grid_controls():
    params = dict(
        _BASE_PARAMS,
        delta_f=-1.0,
        f_lower=-1.0,
        f_final=-1.0,
    )
    assert eccentricfd_sequence_native_supported(params)

    params["amplitude_order"] = 0
    assert not eccentricfd_sequence_native_supported(params)


def test_eccentricfd_default_supports_double_precision_devices(
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    assert eccentricfd_default_native_supported(_BASE_PARAMS)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_eccentricfd_default_rejects_mps(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("mps"))
    assert not eccentricfd_default_native_supported(_BASE_PARAMS)


def test_eccentricfd_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    params = dict(_BASE_PARAMS)
    params.pop("approximant")
    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="EccentricFD", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    native_generator = eccentricfd_mod.eccentricfd_fd_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native EccentricFD called lalsimulation")

    monkeypatch.setattr(eccentricfd_mod, "eccentricfd_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="EccentricFD", **params)

    assert native_calls == 1
    for expected, result in zip(reference_arrays, actual):
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 2.0e-11


def test_eccentricfd_unsupported_options_use_lal_fallback(monkeypatch, preserve_scheme):
    params = dict(_BASE_PARAMS, spin_order=2)
    params.pop("approximant")
    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="EccentricFD", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported EccentricFD parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(eccentricfd_mod, "eccentricfd_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform(approximant="EccentricFD", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert actual._data.tensor.device.type == "cpu"
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-14, atol=0.0)


@pytest.mark.parametrize(
    ("global_value", "component_value", "expected_native"),
    [
        (None, None, True),
        ("0", None, False),
        ("1", None, True),
        ("1", "0", False),
        ("0", "1", True),
    ],
)
def test_eccentricfd_native_flag_precedence(
    global_value,
    component_value,
    expected_native,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    native_generator = eccentricfd_mod.eccentricfd_fd_torch
    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    calls = {"native": 0, "lal": 0}

    def recording_native(**params):
        calls["native"] += 1
        return native_generator(**params)

    def recording_lal(*args, **kwargs):
        calls["lal"] += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(eccentricfd_mod, "eccentricfd_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    if global_value is None:
        monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    else:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", global_value)
    if component_value is None:
        monkeypatch.delenv("PYCBC_ECCENTRICFD_NATIVE", raising=False)
    else:
        monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", component_value)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    get_fd_waveform(**_BASE_PARAMS)

    assert calls == {
        "native": int(expected_native),
        "lal": int(not expected_native),
    }


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_eccentricfd_default_mps_uses_lalsimulation_fallbacks(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALSequenceFallbackReached(Exception):
        pass

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0
    sequence_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("default MPS EccentricFD request reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    def recording_sequence(*_args, **_kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        raise LALSequenceFallbackReached

    monkeypatch.setattr(
        eccentricfd_mod,
        "eccentricfd_fd_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        eccentricfd_mod,
        "eccentricfd_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("mps"))
    result = get_fd_waveform(**_BASE_PARAMS)
    with pytest.raises(LALSequenceFallbackReached):
        get_fd_waveform_sequence(
            approximant="EccentricFD",
            sample_points=[18.0, 30.0, 100.0],
            mass1=25.0,
            mass2=10.0,
            distance=275.0,
            eccentricity=0.15,
        )

    assert lal_calls == 1
    assert sequence_calls == 1
    for series in result:
        assert series._data.tensor.device.type == "mps"
        assert series._data.tensor.dtype == torch.complex64


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_eccentricfd_native_stays_on_requested_device_without_host_transfer(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    params = dict(_BASE_PARAMS)
    params.pop("approximant")
    _, reference_arrays, _ = _reference_and_native(params, monkeypatch, device="cpu")
    _activate_scheme(_scheme.TorchScheme(device_name))

    def reject_host_transfer(_self):
        raise AssertionError("native EccentricFD copied through NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        actual = get_fd_waveform(approximant="EccentricFD", **params)
        sample_points = [100.0, 18.0, 45.0, 18.0, 30.0]
        sequence = get_fd_waveform_sequence(
            approximant="EccentricFD",
            sample_points=sample_points,
            **params,
        )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = 1.5e-2 if device_name == "mps" else 2.0e-11
    for reference, series in zip(reference_arrays, actual):
        tensor = series._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        expected = torch.as_tensor(
            reference,
            dtype=expected_dtype,
            device=tensor.device,
        )
        torch.testing.assert_close(tensor == 0.0, expected == 0.0)
        nonzero = expected != 0.0
        inverse_scale = 1.0 / float(np.max(np.abs(reference)))
        scaled_tensor = tensor[nonzero] * inverse_scale
        scaled_expected = expected[nonzero] * inverse_scale
        squared_error = torch.sum(torch.abs(scaled_tensor - scaled_expected) ** 2)
        squared_norm = torch.sum(torch.abs(scaled_expected) ** 2)
        relative_error = torch.sqrt(squared_error / squared_norm)
        assert float(relative_error.cpu()) < tolerance

    indices = [
        round(frequency / params["delta_f"])
        for frequency in sample_points
    ]
    for reference, samples in zip(reference_arrays, sequence):
        tensor = samples._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        expected = torch.as_tensor(
            reference[indices],
            dtype=expected_dtype,
            device=tensor.device,
        )
        inverse_scale = 1.0 / float(np.max(np.abs(reference[indices])))
        torch.testing.assert_close(
            tensor * inverse_scale,
            expected * inverse_scale,
            rtol=tolerance,
            atol=tolerance,
        )


def test_eccentricfd_native_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(RuntimeError, match="requires TorchScheme"):
        eccentricfd_fd_torch(**_BASE_PARAMS)


def test_eccentricfd_sequence_native_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(RuntimeError, match="requires TorchScheme"):
        eccentricfd_fd_sequence_torch(
            **dict(_BASE_PARAMS, sample_points=[18.0, 30.0])
        )


def test_eccentricfd_public_sequence_matches_regular_lal_samples(
    monkeypatch, preserve_scheme
):
    params = dict(_BASE_PARAMS, f_ref=91.0)
    params.pop("approximant")
    sample_points = [100.0, 18.0, 45.0, 18.0, 30.0]

    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="EccentricFD", **params)
    indices = [
        round(frequency / params["delta_f"])
        for frequency in sample_points
    ]
    expected = tuple(series.numpy()[indices].copy() for series in reference)

    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    native_generator = eccentricfd_mod.eccentricfd_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native_generator(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(
            "native EccentricFD sequence called lalsimulation"
        )

    monkeypatch.setattr(
        eccentricfd_mod,
        "eccentricfd_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="EccentricFD",
        sample_points=sample_points,
        **params,
    )

    assert calls == 1
    for reference_samples, result in zip(expected, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        relative_error = np.linalg.norm(result.numpy() - reference_samples)
        relative_error /= np.linalg.norm(reference_samples)
        assert relative_error < 2.0e-11


def test_eccentricfd_sequence_is_order_independent_and_ignores_f_ref(
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    points = [45.0, 18.0, 30.0, 18.0]
    permutation = [1, 0, 3, 2]
    first = eccentricfd_fd_sequence_torch(
        **dict(_BASE_PARAMS, sample_points=points, f_ref=0.0)
    )
    second = eccentricfd_fd_sequence_torch(
        **dict(
            _BASE_PARAMS,
            sample_points=[points[index] for index in permutation],
            f_ref=90.0,
        )
    )

    for original, reordered in zip(first, second):
        tensor = original._data.tensor
        torch.testing.assert_close(
            reordered._data.tensor,
            tensor[permutation],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(tensor[1], tensor[3], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([[18.0, 30.0]], "non-empty vector"),
        ([18.0, float("nan")], "finite"),
        ([18.0, float("inf")], "finite"),
        ([18.0, 0.0], "positive"),
        ([18.0, -1.0], "positive"),
    ],
)
def test_eccentricfd_sequence_rejects_invalid_sample_points(
    sample_points, message, preserve_scheme
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        eccentricfd_fd_sequence_torch(
            **dict(_BASE_PARAMS, sample_points=sample_points)
        )


def test_eccentricfd_unsupported_sequence_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.eccentricfd_torch as eccentricfd_mod
    import pycbc.waveform.waveform as waveform_mod

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported EccentricFD sequence reached Torch")

    def expected_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("EccentricFD sequence LAL path reached")

    monkeypatch.setattr(
        eccentricfd_mod,
        "eccentricfd_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        expected_lal,
    )
    monkeypatch.setenv("PYCBC_ECCENTRICFD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(RuntimeError, match="LAL path reached"):
        get_fd_waveform_sequence(
            approximant="EccentricFD",
            mass1=25.0,
            mass2=10.0,
            distance=275.0,
            sample_points=[18.0, 30.0, 100.0],
            amplitude_order=0,
        )

    assert calls == 1
