import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform, get_td_waveform_modes  # noqa: E402
from pycbc.waveform.taylort2_torch import (  # noqa: E402
    taylor_t2_coefficients,
    taylor_t2_orbit,
    taylor_t2_phase,
    taylor_t2_timing,
    taylor_t2_timing_derivative,
    taylort2_modes_native_supported,
    taylort2_modes_torch,
    taylort2_native_supported,
    taylort2_td_torch,
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
    return lalsimulation.SimInspiralTaylorT2PNEvolveOrbit(
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


def _lal_modes(parameters):
    modes = lalsimulation.SimInspiralTaylorT2PNModes(
        1.0,
        parameters["delta_t"],
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["f_lower"],
        parameters.get("f_ref", 0.0),
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters.get("lambda1", 0.0),
        parameters.get("lambda2", 0.0),
        parameters.get("tidal_order", -1),
        parameters.get("amplitude_order", -1),
        parameters.get("phase_order", -1),
        parameters.get("ell_max", 5),
    )
    result = {}
    while modes is not None:
        mode = modes.mode
        result[(modes.l, modes.m)] = (
            np.array(mode.data.data, copy=True),
            mode.deltaT,
            float(mode.epoch),
        )
        modes = modes.next
    return result


def _assert_orbit_matches_lal(parameters):
    expected_velocity, expected_phase = _lal_orbit(parameters)
    actual = taylor_t2_orbit(**parameters)

    assert len(actual) == expected_velocity.data.length
    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity.data.data,
        rtol=5.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected_phase.data.data,
        rtol=5.0e-10,
        atol=1.0e-6,
    )
    assert actual.delta_t == expected_velocity.deltaT
    assert abs(actual.epoch - float(expected_velocity.epoch)) < 1.0e-9


@pytest.mark.parametrize("phase_order", (0, 2, 3, 4, 5, 6, 7, -1))
def test_taylor_t2_orbit_matches_lalsuite_phase_orders(phase_order):
    _assert_orbit_matches_lal(
        {
            "mass1": 30.0,
            "mass2": 20.0,
            "delta_t": 1.0 / 2048.0,
            "f_lower": 25.0,
            "f_ref": 25.0,
            "coa_phase": 0.3,
            "phase_order": phase_order,
        }
    )


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
        {
            "mass1": 1.4,
            "mass2": 10.0,
            "delta_t": 1.0 / 4096.0,
            "f_lower": 30.0,
            "f_ref": 0.0,
            "lambda1": 1000.0,
            "lambda2": 1000.0,
            "tidal_order": 12,
            "phase_order": 7,
        },
        {
            "mass1": 4.0,
            "mass2": 3.0,
            "delta_t": 1.0 / 512.0,
            "f_lower": 100.0,
            "f_ref": 100.0,
        },
    ),
)
def test_taylor_t2_orbit_matches_reference_tides_and_cutoffs(parameters):
    _assert_orbit_matches_lal(parameters)


def test_taylor_t2_timing_derivative_matches_autograd():
    coefficients = taylor_t2_coefficients(
        1.5,
        1.3,
        lambda1=400.0,
        lambda2=800.0,
        tidal_order=12,
        phase_order=6,
    )
    velocity = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    taylor_t2_timing(velocity, coefficients).backward()

    torch.testing.assert_close(
        taylor_t2_timing_derivative(velocity.detach(), coefficients),
        velocity.grad,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_taylor_t2_analytic_functions_preserve_device_and_dtype(dtype):
    coefficients = taylor_t2_coefficients(
        30.0,
        20.0,
        device="cpu",
        dtype=dtype,
    )
    velocity = torch.tensor((0.2, 0.25), dtype=dtype)

    for result in (
        taylor_t2_timing(velocity, coefficients),
        taylor_t2_timing_derivative(velocity, coefficients),
        taylor_t2_phase(velocity, coefficients),
    ):
        assert result.device.type == "cpu"
        assert result.dtype == dtype
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_taylor_t2_orbit_stays_on_requested_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64

    orbit = taylor_t2_orbit(
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
def test_taylor_t2_orbit_validates_inputs(arguments, message):
    parameters = {
        "mass1": 30.0,
        "mass2": 20.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 25.0,
    }
    parameters.update(arguments)
    with pytest.raises(ValueError, match=message):
        taylor_t2_orbit(**parameters)


def test_taylor_t2_orbit_rejects_invalid_frequency_ranges():
    with pytest.raises(ValueError, match="f_ref"):
        taylor_t2_orbit(30.0, 20.0, 1.0 / 2048.0, 25.0, f_ref=10.0)
    with pytest.raises(ValueError, match="sample rate"):
        taylor_t2_orbit(30.0, 20.0, 1.0 / 64.0, 40.0)


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
        ({"approximant": "TaylorT1"}, False),
    ),
)
def test_taylor_t2_native_support_boundary(parameters, expected):
    assert taylort2_native_supported(parameters) is expected


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
def test_taylor_t2_modes_native_support_boundary(parameters, expected):
    assert taylort2_modes_native_supported(parameters) is expected


@pytest.mark.parametrize("parameters", _WAVEFORM_CASES)
def test_taylor_t2_waveform_matches_lalsuite(parameters, preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT2", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort2_td_torch(**parameters)

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
        assert relative_error < 2.0e-7


@pytest.mark.parametrize(
    "updates",
    (
        {"ell_max": 3},
        {"ell_max": 2, **_WAVEFORM_CASES[1]},
    ),
)
def test_taylor_t2_modes_match_lalsuite(updates, preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], **updates)
    reference = _lal_modes(parameters)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort2_modes_torch(approximant="TaylorT2", **parameters)

    assert set(actual) == set(reference)
    for mode, (actual_real, actual_imaginary) in actual.items():
        expected, expected_delta_t, expected_epoch = reference[mode]
        result = actual_real.numpy() + 1j * actual_imaginary.numpy()
        assert len(actual_real) == len(expected)
        assert actual_real.delta_t == expected_delta_t
        assert abs(float(actual_real.start_time) - expected_epoch) < 1.0e-9
        if np.linalg.norm(expected) == 0.0:
            assert np.linalg.norm(result) == 0.0
        else:
            relative_error = np.linalg.norm(result - expected) / np.linalg.norm(
                expected
            )
            assert relative_error < 3.0e-7


def test_taylor_t2_modes_ignore_coalescence_phase(preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=2)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    first = taylort2_modes_torch(**parameters)
    second = taylort2_modes_torch(**dict(parameters, coa_phase=-2.1))

    for mode in first:
        for first_series, second_series in zip(first[mode], second[mode]):
            torch.testing.assert_close(
                first_series._data.tensor,
                second_series._data.tensor,
                rtol=0.0,
                atol=0.0,
            )


def test_taylor_t2_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = _WAVEFORM_CASES[0]
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT2", **parameters)

    import pycbc.waveform.taylort2_torch as taylort2_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = taylort2_module.taylort2_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT2 called lalsimulation")

    monkeypatch.setattr(taylort2_module, "taylort2_td_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="TaylorT2", **parameters)

    assert native_calls == 1
    for expected, result in zip(reference, actual):
        assert len(result) == len(expected)
        assert result._data.tensor.device.type == "cpu"


def test_taylor_t2_public_native_mode_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=2)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant="TaylorT2", **parameters)

    import pycbc.waveform.taylort2_torch as taylort2_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    native_generator = taylort2_module.taylort2_modes_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT2 modes called lalsimulation")

    monkeypatch.setattr(taylort2_module, "taylort2_modes_torch", recording_native)
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform_modes(approximant="TaylorT2", **parameters)

    assert native_calls == 1
    assert set(actual) == set(reference)
    assert all(
        isinstance(series._data.tensor, torch.Tensor)
        for pair in actual.values()
        for series in pair
    )


@pytest.mark.parametrize("component_enabled", ("0", "1"))
def test_taylor_t2_disabled_or_unsupported_uses_lal_fallback(
    component_enabled, monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0])
    if component_enabled == "1":
        parameters["eccentricity_order"] = 0
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT2", **parameters)

    import pycbc.waveform.taylort2_torch as taylort2_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported TaylorT2 parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(taylort2_module, "taylort2_td_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT2_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="TaylorT2", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert isinstance(actual._data.tensor, torch.Tensor)


def test_taylor_t2_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort2_td_torch(**_WAVEFORM_CASES[0])
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort2_modes_torch(**_WAVEFORM_CASES[0])
