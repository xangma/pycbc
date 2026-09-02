import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform, get_td_waveform_modes  # noqa: E402
from pycbc.waveform.taylort1_torch import (  # noqa: E402
    taylor_t1_coefficients,
    taylor_t1_orbit,
    taylor_t1_rhs,
    taylort1_modes_native_supported,
    taylort1_modes_torch,
    taylort1_native_supported,
    taylort1_td_torch,
)


_WAVEFORM_CASES = (
    {
        "mass1": 31.0,
        "mass2": 17.0,
        "distance": 230.0,
        "inclination": 0.73,
        "coa_phase": 0.37,
        "long_asc_nodes": 0.29,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 30.0,
        "f_ref": 30.0,
    },
    {
        "mass1": 1.5,
        "mass2": 1.3,
        "distance": 90.0,
        "inclination": 1.1,
        "coa_phase": -0.2,
        "long_asc_nodes": -0.17,
        "delta_t": 1.0 / 4096.0,
        "f_lower": 100.0,
        "f_ref": 120.0,
        "lambda1": 400.0,
        "lambda2": 800.0,
        "tidal_order": 12,
        "phase_order": 6,
        "amplitude_order": 5,
    },
)


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _lal_orbit(parameters):
    return lalsimulation.SimInspiralTaylorT1PNEvolveOrbit(
        parameters.get("coa_phase", 0.0),
        parameters["delta_t"],
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["f_lower"],
        parameters.get("f_ref", 0.0),
        parameters.get("lambda1", 0.0),
        parameters.get("lambda2", 0.0),
        parameters.get("tidal_order", -1),
        parameters.get("phase_order", -1),
    )


@pytest.mark.parametrize("phase_order", (0, 2, 3, 4, 5, 6, 7, -1))
def test_taylor_t1_orbit_matches_lalsuite_phase_orders(phase_order):
    parameters = {
        "mass1": 30.0,
        "mass2": 20.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 25.0,
        "f_ref": 25.0,
        "coa_phase": 0.3,
        "phase_order": phase_order,
    }
    expected_velocity, expected_phase = _lal_orbit(parameters)
    actual = taylor_t1_orbit(**parameters)

    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity.data.data,
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected_phase.data.data,
        rtol=2.0e-11,
        atol=2.0e-10,
    )
    assert actual.delta_t == expected_velocity.deltaT
    assert abs(actual.epoch - float(expected_velocity.epoch)) < 1.0e-9


@pytest.mark.parametrize(
    "parameters",
    (
        {
            "mass1": 35.0,
            "mass2": 15.0,
            "delta_t": 1.0 / 4096.0,
            "f_lower": 30.0,
            "f_ref": 0.0,
            "coa_phase": -0.2,
        },
        {
            "mass1": 25.0,
            "mass2": 20.0,
            "delta_t": 1.0 / 2048.0,
            "f_lower": 25.0,
            "f_ref": 40.0,
            "coa_phase": 0.7,
        },
        {
            "mass1": 1.5,
            "mass2": 1.3,
            "delta_t": 1.0 / 4096.0,
            "f_lower": 100.0,
            "f_ref": 120.0,
            "coa_phase": 0.1,
            "lambda1": 400.0,
            "lambda2": 800.0,
            "tidal_order": 12,
            "phase_order": 6,
        },
    ),
)
def test_taylor_t1_orbit_matches_lalsuite_reference_and_tides(parameters):
    expected_velocity, expected_phase = _lal_orbit(parameters)
    actual = taylor_t1_orbit(**parameters)

    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity.data.data,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected_phase.data.data,
        rtol=2.0e-10,
        atol=2.0e-9,
    )
    assert len(actual) == expected_velocity.data.length
    assert abs(actual.epoch - float(expected_velocity.epoch)) < 1.0e-9


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_taylor_t1_rhs_preserves_device_and_dtype(dtype):
    coefficients = taylor_t1_coefficients(
        30.0,
        20.0,
        device="cpu",
        dtype=dtype,
    )
    state = torch.tensor((0.2, 0.3), dtype=dtype)
    result = taylor_t1_rhs(state, coefficients)

    assert result.device.type == "cpu"
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_taylor_t1_orbit_stays_on_requested_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64

    orbit = taylor_t1_orbit(
        mass1=40.0,
        mass2=30.0,
        delta_t=1.0 / 1024.0,
        f_lower=50.0,
        f_ref=50.0,
        device=device_name,
        dtype=dtype,
    )

    assert orbit.velocity.device.type == device_name
    assert orbit.phase.device.type == device_name
    assert orbit.velocity.dtype == dtype
    assert orbit.phase.dtype == dtype
    assert torch.isfinite(orbit.velocity).all()
    assert torch.isfinite(orbit.phase).all()


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"mass1": 0.0}, "mass1"),
        ({"mass2": -1.0}, "mass2"),
        ({"delta_t": 0.0}, "delta_t"),
        ({"f_lower": -20.0}, "f_lower"),
        ({"f_ref": -1.0}, "f_ref"),
        ({"phase_order": 1}, "phase_order"),
        ({"tidal_order": 8}, "tidal_order"),
    ),
)
def test_taylor_t1_orbit_validates_inputs(arguments, message):
    parameters = {
        "mass1": 30.0,
        "mass2": 20.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 25.0,
    }
    parameters.update(arguments)
    with pytest.raises(ValueError, match=message):
        taylor_t1_orbit(**parameters)


def test_taylor_t1_orbit_rejects_reference_outside_evolution():
    with pytest.raises(ValueError, match="f_ref"):
        taylor_t1_orbit(
            mass1=30.0,
            mass2=20.0,
            delta_t=1.0 / 2048.0,
            f_lower=25.0,
            f_ref=10.0,
        )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"phase_order": 0, "amplitude_order": 0, "tidal_order": 0}, True),
        ({"phase_order": 7, "amplitude_order": 6, "tidal_order": 12}, True),
        ({"spin1z": 0.1}, False),
        ({"spin2x": -0.1}, False),
        ({"spin_order": 0}, False),
        ({"eccentricity_order": 0}, False),
        ({"phase_order": 1}, False),
        ({"amplitude_order": 7}, False),
        ({"tidal_order": 10}, True),
        ({"tidal_order": 8}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "TaylorT4"}, False),
    ),
)
def test_taylor_t1_native_support_boundary(parameters, expected):
    assert taylort1_native_supported(parameters) is expected


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"ell_max": 2}, True),
        ({"ell_max": np.int64(6)}, True),
        ({"ell_max": 1}, False),
        ({"ell_max": 7}, False),
        ({"ell_max": 5.0}, False),
        ({"ell_max": None}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"spin1z": 0.1}, False),
    ),
)
def test_taylor_t1_modes_native_support_boundary(parameters, expected):
    assert taylort1_modes_native_supported(parameters) is expected


@pytest.mark.parametrize("parameters", _WAVEFORM_CASES)
def test_taylor_t1_waveform_matches_lalsuite(parameters, preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT1", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort1_td_torch(**parameters)

    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < 1.0e-9
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float64
        relative_error = np.linalg.norm(result_array - expected_array) / np.linalg.norm(
            expected_array
        )
        assert relative_error < 2.0e-9


@pytest.mark.parametrize(
    "updates",
    (
        {"ell_max": 3},
        {
            "ell_max": 2,
            "coa_phase": -1.1,
            "lambda1": 400.0,
            "lambda2": 800.0,
            "tidal_order": 12,
            "phase_order": 6,
            "amplitude_order": 5,
        },
    ),
)
def test_taylor_t1_modes_match_lalsuite(updates, preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], **updates)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant="TaylorT1", **parameters)
    reference_arrays = {
        mode: real.numpy().copy() + 1j * imaginary.numpy().copy()
        for mode, (real, imaginary) in reference.items()
    }

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort1_modes_torch(approximant="TaylorT1", **parameters)

    assert set(actual) == set(reference)
    for mode, (actual_real, actual_imaginary) in actual.items():
        expected_real, _ = reference[mode]
        expected = reference_arrays[mode]
        result = actual_real.numpy() + 1j * actual_imaginary.numpy()
        assert len(actual_real) == len(expected_real)
        assert actual_real.delta_t == expected_real.delta_t
        assert abs(float(actual_real.start_time - expected_real.start_time)) < 1.0e-9
        if np.linalg.norm(expected) == 0.0:
            assert np.linalg.norm(result) == 0.0
        else:
            relative_error = np.linalg.norm(result - expected) / np.linalg.norm(
                expected
            )
            assert relative_error < 2.0e-9


def test_taylor_t1_modes_ignore_coalescence_phase(preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=2)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    first = taylort1_modes_torch(**parameters)
    second = taylort1_modes_torch(**dict(parameters, coa_phase=-2.1))

    for mode in first:
        for first_series, second_series in zip(first[mode], second[mode]):
            torch.testing.assert_close(
                first_series._data.tensor,
                second_series._data.tensor,
                rtol=0.0,
                atol=0.0,
            )


def test_taylor_t1_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = _WAVEFORM_CASES[0]
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT1_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT1", **parameters)

    import pycbc.waveform.taylort1_torch as taylort1_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = taylort1_module.taylort1_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT1 called lalsimulation")

    monkeypatch.setattr(taylort1_module, "taylort1_td_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT1_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="TaylorT1", **parameters)

    assert native_calls == 1
    for expected, result in zip(reference, actual):
        assert len(result) == len(expected)
        assert result._data.tensor.device.type == "cpu"


def test_taylor_t1_public_native_mode_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=2)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT1_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant="TaylorT1", **parameters)

    import pycbc.waveform.taylort1_torch as taylort1_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    native_generator = taylort1_module.taylort1_modes_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT1 modes called lalsimulation")

    monkeypatch.setattr(taylort1_module, "taylort1_modes_torch", recording_native)
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT1_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform_modes(approximant="TaylorT1", **parameters)

    assert native_calls == 1
    assert set(actual) == set(reference)
    assert all(
        isinstance(series._data.tensor, torch.Tensor)
        for pair in actual.values()
        for series in pair
    )


@pytest.mark.parametrize("component_enabled", ("0", "1"))
def test_taylor_t1_disabled_or_unsupported_uses_lal_fallback(
    component_enabled, monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0])
    if component_enabled == "1":
        parameters["eccentricity_order"] = 0
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT1", **parameters)

    import pycbc.waveform.taylort1_torch as taylort1_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported TaylorT1 parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(taylort1_module, "taylort1_td_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT1_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="TaylorT1", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert isinstance(actual._data.tensor, torch.Tensor)


def test_taylor_t1_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort1_td_torch(**_WAVEFORM_CASES[0])
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort1_modes_torch(**_WAVEFORM_CASES[0])
