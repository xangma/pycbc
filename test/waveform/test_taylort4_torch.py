import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform, get_td_waveform_modes  # noqa: E402
from pycbc.waveform.pn_modes_torch import (  # noqa: E402
    pn_modes_lal_convention,
)
from pycbc.waveform.pn_polarization_torch import (  # noqa: E402
    pn_polarizations,
)
from pycbc.waveform.taylort4_torch import (  # noqa: E402
    _rk4_scalar_step,
    _rk4_step,
    _taylor_t4_scalar_coefficients,
    _taylor_t4_scalar_rhs,
    taylor_t4_coefficients,
    taylor_t4_orbit,
    taylor_t4_rhs,
    taylort4_default_native_supported,
    taylort4_modes_native_supported,
    taylort4_modes_torch,
    taylort4_native_supported,
    taylort4_td_torch,
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


def _normalized_correlation(expected, actual):
    return np.dot(expected, actual) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )


def _lal_orbit(parameters):
    velocity, phase = lalsimulation.SimInspiralTaylorT4PNEvolveOrbit(
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
    return velocity, phase


def _lal_real_series(name, samples):
    series = lal.CreateREAL8TimeSeries(
        name,
        lal.LIGOTimeGPS(0),
        0.0,
        1.0,
        lal.DimensionlessUnit,
        len(samples),
    )
    series.data.data[:] = samples
    return series


@pytest.mark.parametrize(
    ("phase_order", "lambda1", "lambda2", "tidal_order"),
    (
        (0, 0.0, 0.0, -1),
        (5, 0.0, 0.0, 0),
        (-1, 400.0, 800.0, 12),
    ),
)
def test_taylor_t4_scalar_rhs_matches_torch_kernel(
    phase_order, lambda1, lambda2, tidal_order
):
    parameters = {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "phase_order": phase_order,
        "tidal_order": tidal_order,
    }
    scalar_coefficients = _taylor_t4_scalar_coefficients(
        30.0,
        20.0,
        **parameters,
    )
    tensor_coefficients = taylor_t4_coefficients(
        30.0,
        20.0,
        dtype=torch.float64,
        **parameters,
    )
    velocity = 0.23
    phase = -0.7

    scalar_rhs = _taylor_t4_scalar_rhs(velocity, scalar_coefficients)
    tensor_rhs = taylor_t4_rhs(
        torch.tensor((velocity, phase), dtype=torch.float64),
        tensor_coefficients,
    )
    np.testing.assert_allclose(scalar_rhs, tensor_rhs.numpy(), rtol=3.0e-15)

    scalar_step = _rk4_scalar_step(
        velocity,
        phase,
        1.0 / 4096.0,
        scalar_coefficients,
    )
    tensor_step = _rk4_step(
        torch.tensor((velocity, phase), dtype=torch.float64),
        1.0 / 4096.0,
        tensor_coefficients,
    )
    np.testing.assert_allclose(scalar_step, tensor_step.numpy(), rtol=3.0e-15)


def test_taylor_t4_compiled_orbit_matches_python_fallback(monkeypatch):
    import pycbc.waveform.taylort4_torch as taylort4_module

    if taylort4_module._evolve_taylor_t4_compiled is None:
        pytest.skip("TaylorT4 compiled extension unavailable")

    parameters = {
        "mass1": 25.0,
        "mass2": 20.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 25.0,
        "f_ref": 40.0,
        "coa_phase": 0.7,
        "lambda1": 400.0,
        "lambda2": 800.0,
        "tidal_order": 12,
        "phase_order": 6,
    }
    compiled = taylort4_module.taylor_t4_orbit(**parameters)
    monkeypatch.setattr(taylort4_module, "_evolve_taylor_t4_compiled", None)
    fallback = taylort4_module.taylor_t4_orbit(**parameters)

    assert len(compiled) == len(fallback)
    assert compiled.epoch == fallback.epoch
    torch.testing.assert_close(
        compiled.velocity,
        fallback.velocity,
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    torch.testing.assert_close(
        compiled.phase,
        fallback.phase,
        rtol=2.0e-13,
        atol=2.0e-11,
    )


@pytest.mark.parametrize("phase_order", (0, 2, 3, 4, 5, 6, 7, -1))
def test_taylor_t4_orbit_matches_lalsuite_phase_orders(phase_order):
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
    actual = taylor_t4_orbit(**parameters)

    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity.data.data,
        rtol=4.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected_phase.data.data,
        rtol=2.0e-10,
        atol=1.0e-8,
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
        },
    ),
)
def test_taylor_t4_orbit_matches_lalsuite_reference_and_tides(parameters):
    expected_velocity, expected_phase = _lal_orbit(parameters)
    actual = taylor_t4_orbit(**parameters)

    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity.data.data,
        rtol=1.0e-6,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected_phase.data.data,
        rtol=5.0e-8,
        atol=3.0e-5,
    )
    assert len(actual) == expected_velocity.data.length
    assert abs(actual.epoch - float(expected_velocity.epoch)) < 1.0e-9


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_taylor_t4_rhs_preserves_device_and_dtype(dtype):
    coefficients = taylor_t4_coefficients(
        30.0,
        20.0,
        device="cpu",
        dtype=dtype,
    )
    state = torch.tensor((0.2, 0.3), dtype=dtype)
    result = taylor_t4_rhs(state, coefficients)

    assert result.device.type == "cpu"
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_taylor_t4_orbit_stays_on_requested_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64

    orbit = taylor_t4_orbit(
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

    modes = pn_modes_lal_convention(
        orbit.velocity,
        orbit.phase,
        40.0,
        30.0,
        100.0,
        ell_max=2,
    )
    for real, imaginary in modes.values():
        assert real.device.type == device_name
        assert imaginary.device.type == device_name
        assert real.dtype == dtype
        assert imaginary.dtype == dtype


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
def test_taylor_t4_orbit_validates_inputs(arguments, message):
    parameters = {
        "mass1": 30.0,
        "mass2": 20.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 25.0,
    }
    parameters.update(arguments)
    with pytest.raises(ValueError, match=message):
        taylor_t4_orbit(**parameters)


def test_taylor_t4_orbit_rejects_reference_outside_evolution():
    with pytest.raises(ValueError, match="f_ref"):
        taylor_t4_orbit(
            mass1=30.0,
            mass2=20.0,
            delta_t=1.0 / 2048.0,
            f_lower=25.0,
            f_ref=10.0,
        )


@pytest.mark.parametrize("amplitude_order", (-1, 0, 1, 2, 3, 4, 5, 6))
def test_pn_polarizations_match_lalsuite(amplitude_order):
    parameters = {
        "mass1": 31.0,
        "mass2": 17.0,
        "distance": 230.0,
        "inclination": 0.73,
        "coa_phase": 0.37,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 30.0,
        "f_ref": 30.0,
    }
    velocity, phase = _lal_orbit(parameters)
    expected = lalsimulation.SimInspiralPNPolarizationWaveforms(
        velocity,
        phase,
        1.0,
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters["inclination"],
        amplitude_order,
    )
    actual = pn_polarizations(
        torch.tensor(velocity.data.data, dtype=torch.float64),
        torch.tensor(phase.data.data, dtype=torch.float64),
        parameters["mass1"],
        parameters["mass2"],
        parameters["distance"],
        parameters["inclination"],
        amplitude_order=amplitude_order,
    )

    for expected_series, actual_tensor in zip(expected, actual):
        np.testing.assert_allclose(
            actual_tensor.numpy(),
            expected_series.data.data,
            rtol=2.0e-13,
            atol=1.0e-30,
        )


@pytest.mark.parametrize("amplitude_order", (-1, 0, 1, 2, 3, 4, 5, 6))
def test_pn_modes_match_lalsuite(amplitude_order):
    velocity_samples = np.array((0.09, 0.13, 0.21, 0.31))
    phase_samples = np.array((-1.2, 0.0, 0.7, 3.4))
    mass1 = 31.0
    mass2 = 17.0
    distance = 230.0
    velocity = _lal_real_series("velocity", velocity_samples)
    phase = _lal_real_series("phase", phase_samples)

    actual = pn_modes_lal_convention(
        torch.tensor(velocity_samples, dtype=torch.float64),
        torch.tensor(phase_samples, dtype=torch.float64),
        mass1,
        mass2,
        distance,
        ell_max=6,
        amplitude_order=amplitude_order,
    )

    for (ell, emm), (actual_real, actual_imaginary) in actual.items():
        expected = (
            lalsimulation.CreateSimInspiralPNModeCOMPLEX16TimeSeriesLALConvention(
                velocity,
                phase,
                mass1 * lal.MSUN_SI,
                mass2 * lal.MSUN_SI,
                distance * 1.0e6 * lal.PC_SI,
                amplitude_order,
                ell,
                emm,
            ).data.data
        )
        result = actual_real.numpy() + 1j * actual_imaginary.numpy()
        np.testing.assert_allclose(result, expected, rtol=2.0e-13, atol=1.0e-29)

    # Preserve LAL's historical unit-amplitude placeholder for this mode.
    np.testing.assert_array_equal(actual[6, 0][0].numpy(), -1.0)
    np.testing.assert_array_equal(actual[6, 0][1].numpy(), 0.0)


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
        ({"frame_axis": 1}, False),
        ({"modes_choice": 1}, False),
        ({"side_bands": 1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "TaylorT1"}, False),
    ),
)
def test_taylor_t4_native_support_boundary(parameters, expected):
    assert taylort4_native_supported(parameters) is expected


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
def test_taylor_t4_modes_native_support_boundary(parameters, expected):
    assert taylort4_modes_native_supported(parameters) is expected


def test_taylor_t4_default_supports_cpu(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    assert taylort4_default_native_supported({})


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylor_t4_default_rejects_mps(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("mps"))
    assert not taylort4_default_native_supported({})


@pytest.mark.parametrize("parameters", _WAVEFORM_CASES)
def test_taylor_t4_waveform_matches_lalsuite(parameters, preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT4", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort4_td_torch(**parameters)

    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < 1.0e-9
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float64
        assert np.isfinite(result_array).all()
        relative_error = np.linalg.norm(result_array - expected_array) / np.linalg.norm(
            expected_array
        )
        assert relative_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.99999


@pytest.mark.parametrize(
    "updates",
    (
        {"ell_max": 6},
        {
            "ell_max": 3,
            "coa_phase": -1.1,
            "lambda1": 400.0,
            "lambda2": 800.0,
            "tidal_order": 12,
            "phase_order": 6,
            "amplitude_order": 5,
        },
    ),
)
def test_taylor_t4_modes_match_lalsuite(updates, preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], **updates)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant="TaylorT4", **parameters)
    reference_arrays = {
        mode: real.numpy().copy() + 1j * imaginary.numpy().copy()
        for mode, (real, imaginary) in reference.items()
    }

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = taylort4_modes_torch(approximant="TaylorT4", **parameters)

    assert set(actual) == set(reference)
    for mode, (actual_real, actual_imaginary) in actual.items():
        expected_real, _ = reference[mode]
        expected = reference_arrays[mode]
        result = actual_real.numpy() + 1j * actual_imaginary.numpy()
        assert len(actual_real) == len(expected_real)
        assert actual_real.delta_t == expected_real.delta_t
        assert abs(float(actual_real.start_time - expected_real.start_time)) < 1.0e-9
        assert actual_real._data.tensor.device.type == "cpu"
        assert actual_imaginary._data.tensor.device.type == "cpu"
        if np.linalg.norm(expected) == 0.0:
            assert np.linalg.norm(result) == 0.0
        else:
            relative_error = np.linalg.norm(result - expected) / np.linalg.norm(
                expected
            )
            assert relative_error < 5.0e-7


def test_taylor_t4_modes_ignore_coalescence_phase(preserve_scheme):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=2)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    first = taylort4_modes_torch(**parameters)
    second = taylort4_modes_torch(**dict(parameters, coa_phase=-2.1))

    for mode in first:
        for first_series, second_series in zip(first[mode], second[mode]):
            torch.testing.assert_close(
                first_series._data.tensor,
                second_series._data.tensor,
                rtol=0.0,
                atol=0.0,
            )


def test_taylor_t4_explicit_cpu_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform as waveform_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "1")
    native_td = taylort4_module.taylort4_td_torch
    native_modes = taylort4_module.taylort4_modes_torch
    calls = {"td": 0, "modes": 0}

    def recording_td(**parameters):
        calls["td"] += 1
        return native_td(**parameters)

    def recording_modes(**parameters):
        calls["modes"] += 1
        return native_modes(**parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("explicit TaylorT4 dispatch called lalsimulation")

    monkeypatch.setattr(taylort4_module, "taylort4_td_torch", recording_td)
    monkeypatch.setattr(
        taylort4_module,
        "taylort4_modes_torch",
        recording_modes,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )

    _activate_scheme(_scheme.TorchScheme("cpu"))
    polarizations = get_td_waveform(approximant="TaylorT4", **_WAVEFORM_CASES[0])
    modes = get_td_waveform_modes(
        approximant="TaylorT4",
        ell_max=2,
        **_WAVEFORM_CASES[0],
    )

    assert calls == {"td": 1, "modes": 1}
    assert all(series._data.tensor.device.type == "cpu" for series in polarizations)
    assert all(
        series._data.tensor.device.type == "cpu"
        for pair in modes.values()
        for series in pair
    )


def test_taylor_t4_default_cpu_modes_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORT4_NATIVE", raising=False)
    native_modes = taylort4_module.taylort4_modes_torch
    calls = 0

    def recording_modes(**parameters):
        nonlocal calls
        calls += 1
        return native_modes(**parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("default TaylorT4 modes called lalsimulation")

    monkeypatch.setattr(
        taylort4_module,
        "taylort4_modes_torch",
        recording_modes,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )

    _activate_scheme(_scheme.TorchScheme("cpu"))
    modes = get_td_waveform_modes(
        approximant="TaylorT4",
        ell_max=2,
        **_WAVEFORM_CASES[0],
    )

    assert calls == 1
    assert all(
        series._data.tensor.device.type == "cpu"
        for pair in modes.values()
        for series in pair
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylor_t4_default_mps_uses_lalsimulation_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform as waveform_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORT4_NATIVE", raising=False)
    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_modes_generator = (
        waveform_modes_module.lalsimulation.SimInspiralChooseTDModes
    )
    lal_calls = {"td": 0, "modes": 0}

    def unexpected_native(**_parameters):
        raise AssertionError("default MPS TaylorT4 request reached Torch")

    def recording_lal(*args, **kwargs):
        lal_calls["td"] += 1
        return lal_generator(*args, **kwargs)

    def recording_lal_modes(*args, **kwargs):
        lal_calls["modes"] += 1
        return lal_modes_generator(*args, **kwargs)

    monkeypatch.setattr(taylort4_module, "taylort4_td_torch", unexpected_native)
    monkeypatch.setattr(
        taylort4_module,
        "taylort4_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        recording_lal_modes,
    )
    _activate_scheme(_scheme.TorchScheme("mps"))
    result = get_td_waveform(approximant="TaylorT4", **_WAVEFORM_CASES[0])
    modes = get_td_waveform_modes(
        approximant="TaylorT4",
        ell_max=2,
        **_WAVEFORM_CASES[0],
    )

    assert lal_calls == {"td": 1, "modes": 1}
    assert all(series._data.tensor.device.type == "mps" for series in result)
    assert all(
        series._data.tensor.device.type == "mps"
        for pair in modes.values()
        for series in pair
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylor_t4_explicit_mps_override_uses_native(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform_module

    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "1")

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("explicit MPS TaylorT4 request called lalsimulation")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    _activate_scheme(_scheme.TorchScheme("mps"))
    result = get_td_waveform(approximant="TaylorT4", **_WAVEFORM_CASES[0])

    assert all(series._data.tensor.device.type == "mps" for series in result)


def test_taylor_t4_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = _WAVEFORM_CASES[0]
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT4", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = taylort4_module.taylort4_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT4 called lalsimulation")

    monkeypatch.setattr(taylort4_module, "taylort4_td_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="TaylorT4", **parameters)

    assert native_calls == 1
    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_error = np.linalg.norm(result_array - expected_array) / np.linalg.norm(
            expected_array
        )
        assert relative_error < 5.0e-5


def test_taylor_t4_public_native_mode_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0], ell_max=3)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant="TaylorT4", **parameters)

    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    native_generator = taylort4_module.taylort4_modes_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native TaylorT4 modes called lalsimulation")

    monkeypatch.setattr(taylort4_module, "taylort4_modes_torch", recording_native)
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform_modes(approximant="TaylorT4", **parameters)

    assert native_calls == 1
    assert set(actual) == set(reference)
    assert all(
        isinstance(series._data.tensor, torch.Tensor)
        for pair in actual.values()
        for series in pair
    )


@pytest.mark.parametrize("component_enabled", ("0", "1"))
def test_taylor_t4_disabled_or_unsupported_uses_lal_fallback(
    component_enabled, monkeypatch, preserve_scheme
):
    parameters = dict(_WAVEFORM_CASES[0])
    if component_enabled == "1":
        parameters["eccentricity_order"] = 0
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="TaylorT4", **parameters)

    import pycbc.waveform.taylort4_torch as taylort4_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported TaylorT4 parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(taylort4_module, "taylort4_td_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORT4_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="TaylorT4", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert isinstance(actual._data.tensor, torch.Tensor)
        assert actual._data.tensor.device.type == "cpu"


def test_taylor_t4_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort4_td_torch(**_WAVEFORM_CASES[0])
    with pytest.raises(TypeError, match="active TorchScheme"):
        taylort4_modes_torch(**_WAVEFORM_CASES[0])
