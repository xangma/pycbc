import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform  # noqa: E402
from pycbc.waveform._mode_rotation_torch import (  # noqa: E402
    wigner_d_element,
)
from pycbc.waveform._cubic_spline_torch import (  # noqa: E402
    _natural_cubic_coeff,
    _spline_eval,
)
from pycbc.waveform.imrphenomt_torch import (  # noqa: E402
    _build_imrphenomt_core,
)
from pycbc.waveform.imrphenomtp_torch import (  # noqa: E402
    _imrphenomtp_host_rhs,
    _uniform_natural_cubic_coeff,
    evolve_imrphenomtp_orbit,
    imrphenomtp_euler_angles,
    imrphenomtp_evolved_final_spin,
    imrphenomtp_initial_final_spin,
)
from pycbc.waveform.imrphenomx_spintaylor_torch import (  # noqa: E402
    imrphenomtp_spintaylor_vector_derivatives,
)
from pycbc.waveform.imrphenomtp_waveform_torch import (  # noqa: E402
    _build_imrphenomtp_modes,
    _build_imrphenomtp_state,
    imrphenomtp_native_supported,
    imrphenomtp_td_torch,
)


_CASES = (
    pytest.param(
        {
            "mass1": 35.0,
            "mass2": 25.0,
            "spin1": (0.2, -0.1, 0.3),
            "spin2": (-0.15, 0.08, -0.2),
            "delta_t": 1.0 / 4096.0,
            "f_lower": 20.0,
            "f_ref": 30.0,
        },
        id="backward-and-forward",
    ),
    pytest.param(
        {
            "mass1": 50.0,
            "mass2": 10.0,
            "spin1": (0.7, -0.2, 0.1),
            "spin2": (0.1, 0.3, -0.6),
            "delta_t": 1.0 / 2048.0,
            "f_lower": 25.0,
            "f_ref": 25.0,
        },
        id="forward-only-high-q",
    ),
)

_WAVEFORM_CASE = {
    "mass1": 80.0,
    "mass2": 40.0,
    "spin1x": 0.2,
    "spin1y": -0.1,
    "spin1z": 0.3,
    "spin2x": -0.1,
    "spin2y": 0.2,
    "spin2z": -0.2,
    "distance": 100.0,
    "inclination": 0.7,
    "coa_phase": 0.2,
    "delta_t": 1.0 / 2048.0,
    "f_lower": 30.0,
    "f_ref": 30.0,
}


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _normalized_correlation(expected, actual):
    return abs(np.vdot(expected, actual)) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )


def _lal_mode(mode_series, mode):
    return np.asarray(
        lalsimulation.SphHarmTimeSeriesGetMode(
            mode_series,
            *mode,
        ).data.data
    )


def _build_core(parameters):
    return _build_imrphenomt_core(
        {
            key: value
            for key, value in parameters.items()
            if key not in ("spin1", "spin2")
        }
        | {
            "spin1z": parameters["spin1"][2],
            "spin2z": parameters["spin2"][2],
        }
    )


def _build_tp_core(parameters):
    aligned_core = _build_core(parameters)
    final_spin_prec = imrphenomtp_initial_final_spin(
        aligned_core,
        parameters["spin1"],
        parameters["spin2"],
    )
    carrier_parameters = {
        key: value for key, value in parameters.items() if key not in ("spin1", "spin2")
    } | {
        "spin1z": parameters["spin1"][2],
        "spin2z": parameters["spin2"][2],
    }
    return _build_imrphenomt_core(
        carrier_parameters,
        final_spin_prec_override=final_spin_prec,
    )


def test_imrphenomtp_host_vector_field_matches_torch_kernel():
    step = 0.25
    velocity = [0.18, 0.2, 0.23, 0.27, 0.31]
    linear, quadratic, cubic = _uniform_natural_cubic_coeff(velocity, step)
    state = [
        0.63,
        0.1,
        -0.2,
        0.97,
        0.04,
        -0.02,
        0.08,
        -0.01,
        0.03,
        -0.04,
        0.99,
        0.05,
        -0.08,
    ]
    mass1_fraction = 0.6
    mass2_fraction = 0.4
    actual = torch.tensor(
        _imrphenomtp_host_rhs(
            state,
            1.0,
            mass1_fraction,
            mass2_fraction,
            step,
            velocity,
            linear,
            quadratic,
            cubic,
        ),
        dtype=torch.float64,
    )

    state_tensor = torch.tensor(state, dtype=torch.float64)
    knots = torch.arange(len(velocity), dtype=torch.float64) * step
    velocity_tensor = torch.tensor(velocity, dtype=torch.float64)
    torch_linear, torch_quadratic, torch_cubic = _natural_cubic_coeff(
        knots,
        velocity_tensor,
    )
    orbital_velocity = _spline_eval(
        state_tensor[0],
        knots,
        velocity_tensor,
        torch_linear,
        torch_quadratic,
        torch_cubic,
    )
    derivatives = imrphenomtp_spintaylor_vector_derivatives(
        orbital_velocity,
        state_tensor[1:4],
        state_tensor[4:7],
        state_tensor[7:10],
        state_tensor[10:13],
        mass1_fraction,
        mass2_fraction,
    )
    expected = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            derivatives.lnhat,
            derivatives.spin1,
            derivatives.spin2,
            derivatives.e1,
        )
    )
    torch.testing.assert_close(actual, expected, rtol=3.0e-14, atol=1.0e-16)


def _lal_orbit(parameters):
    return lalsimulation.SimIMRPhenomTPHM_EvolveOrbit(
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        *parameters["spin1"],
        *parameters["spin2"],
        parameters["delta_t"],
        parameters["f_lower"],
        parameters["f_ref"],
        0.0,
        lal.CreateDict(),
    )


def _lal_angles(parameters):
    # TP's frame setup reads LAL's shared powers-of-pi cache.  Initialize it
    # explicitly so the reference does not depend on which LAL model a test
    # process happened to call first.
    lalsimulation.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        parameters["f_ref"],
        parameters.get("coa_phase", 0.0),
        parameters.get("inclination", 0.0),
        *parameters["spin1"],
        *parameters["spin2"],
        lal.CreateDict(),
    )
    return lalsimulation.SimIMRPhenomTPHM_JModes(
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        *parameters["spin1"],
        *parameters["spin2"],
        parameters.get("distance", 100.0) * 1.0e6 * lal.PC_SI,
        parameters.get("inclination", 0.0),
        parameters["delta_t"],
        parameters["f_lower"],
        parameters["f_ref"],
        parameters.get("coa_phase", 0.0),
        lal.CreateDict(),
        1,
    )


@pytest.mark.parametrize("parameters", _CASES)
def test_imrphenomtp_orbit_matches_lalsuite(parameters, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = evolve_imrphenomtp_orbit(
        _build_core(parameters),
        parameters["spin1"],
        parameters["spin2"],
    )
    reference = _lal_orbit(parameters)

    expected_velocity = np.asarray(reference[0].data.data)
    expected_spin1 = np.column_stack(
        [np.asarray(series.data.data) for series in reference[1:4]]
    )
    expected_spin2 = np.column_stack(
        [np.asarray(series.data.data) for series in reference[4:7]]
    )
    expected_lnhat = np.column_stack(
        [np.asarray(series.data.data) for series in reference[7:10]]
    )
    expected_e1 = np.column_stack(
        [np.asarray(series.data.data) for series in reference[10:13]]
    )

    assert actual.velocity.numel() == expected_velocity.size
    assert actual.delta_t == parameters["delta_t"]
    assert abs(actual.epoch - float(reference[0].epoch)) < 1.0e-9
    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected_velocity,
        rtol=2.0e-6,
        atol=3.0e-8,
    )
    for result, expected in (
        (actual.spin1, expected_spin1),
        (actual.spin2, expected_spin2),
        (actual.lnhat, expected_lnhat),
        (actual.e1, expected_e1),
    ):
        np.testing.assert_allclose(result.numpy(), expected, rtol=2.0e-6, atol=5.0e-7)

    for vector in (actual.spin1, actual.spin2, actual.lnhat, actual.e1):
        norms = torch.linalg.vector_norm(vector, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.full_like(norms, norms[actual.reference_index]),
            rtol=0.0,
            atol=1.0e-8,
        )
    torch.testing.assert_close(
        torch.sum(actual.lnhat * actual.e1, dim=-1),
        torch.zeros_like(actual.time_m),
        rtol=0.0,
        atol=2.0e-10,
    )


def test_imrphenomtp_non_grid_reference_convention(preserve_scheme):
    parameters = _CASES[0].values[0]
    _activate_scheme(_scheme.TorchScheme("cpu"))
    core = _build_core(parameters)
    actual = evolve_imrphenomtp_orbit(
        core,
        parameters["spin1"],
        parameters["spin2"],
    )
    delta_time_m = float((core.time_m[1] - core.time_m[0]).item())
    reference_offset_m = abs(float((-core.time_m[0] + core.reference_time_m).item()))
    expected_index = math.floor(reference_offset_m / delta_time_m)

    assert actual.reference_index == expected_index
    assert expected_index > 0
    expected_vectors = (
        torch.tensor((0.0, 0.0, 1.0), dtype=core.binary.dtype),
        torch.tensor(parameters["spin1"], dtype=core.binary.dtype),
        torch.tensor(parameters["spin2"], dtype=core.binary.dtype),
        torch.tensor((1.0, 0.0, 0.0), dtype=core.binary.dtype),
    )
    for result, expected in zip(
        (actual.lnhat, actual.spin1, actual.spin2, actual.e1),
        expected_vectors,
    ):
        torch.testing.assert_close(
            result[expected_index], expected, rtol=0.0, atol=2.0e-16
        )
        # LAL's backward Hermite branch places its unevolved initial state in
        # the sample immediately before a non-grid-aligned reference time.
        torch.testing.assert_close(
            result[expected_index - 1], expected, rtol=0.0, atol=2.0e-16
        )


@pytest.mark.parametrize("parameters", _CASES)
def test_imrphenomtp_precessing_remnant_spin(parameters, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    aligned_core = _build_core(parameters)
    actual = imrphenomtp_initial_final_spin(
        aligned_core,
        parameters["spin1"],
        parameters["spin2"],
    )

    mass1 = parameters["mass1"]
    mass2 = parameters["mass2"]
    fraction1 = mass1 / (mass1 + mass2)
    fraction2 = mass2 / (mass1 + mass2)
    eta = fraction1 * fraction2
    aligned = lalsimulation.SimIMRPhenomXFinalSpin2017(
        eta,
        parameters["spin1"][2],
        parameters["spin2"][2],
    )
    perpendicular = np.linalg.norm(
        fraction1**2 * np.asarray(parameters["spin1"][:2])
        + fraction2**2 * np.asarray(parameters["spin2"][:2])
    )
    expected = math.copysign(math.sqrt(aligned**2 + perpendicular**2), aligned)
    assert float(actual) == pytest.approx(expected, rel=2.0e-14)

    precessing_core = _build_tp_core(parameters)
    torch.testing.assert_close(
        precessing_core.final_spin,
        aligned_core.final_spin,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        precessing_core.final_spin_prec,
        actual,
        rtol=2.0e-14,
        atol=0.0,
    )
    assert not torch.equal(
        precessing_core.phase_coefficients.omega_ring_prec,
        precessing_core.phase_coefficients.omega_ring,
    )


def test_imrphenomtp_reuses_prepared_carrier(monkeypatch, preserve_scheme):
    import pycbc.waveform.imrphenomt_torch as carrier_module

    original = carrier_module.get_remnant_fMs
    remnant_calls = 0

    def counting_remnant(*args, **kwargs):
        nonlocal remnant_calls
        remnant_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(carrier_module, "get_remnant_fMs", counting_remnant)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _build_imrphenomtp_state(
        dict(_WAVEFORM_CASE, f_lower=40.0, f_ref=40.0)
    )

    assert remnant_calls == 1


def test_imrphenomtp_evolved_final_spin(preserve_scheme):
    parameters = _CASES[0].values[0]
    _activate_scheme(_scheme.TorchScheme("cpu"))
    core = _build_tp_core(parameters)
    orbit = evolve_imrphenomtp_orbit(
        core,
        parameters["spin1"],
        parameters["spin2"],
    )
    actual = imrphenomtp_evolved_final_spin(core, orbit)

    fraction1 = parameters["mass1"] / (parameters["mass1"] + parameters["mass2"])
    fraction2 = 1.0 - fraction1
    lnhat = orbit.lnhat[-1].numpy()
    spin1 = fraction1**2 * orbit.spin1[-1].numpy()
    spin2 = fraction2**2 * orbit.spin2[-1].numpy()
    spin1_l = np.dot(spin1, lnhat)
    spin2_l = np.dot(spin2, lnhat)
    perpendicular = spin1 - spin1_l * lnhat + spin2 - spin2_l * lnhat
    aligned = lalsimulation.SimIMRPhenomXFinalSpin2017(
        fraction1 * fraction2,
        spin1_l / fraction1**2,
        spin2_l / fraction2**2,
    )
    expected = math.copysign(
        math.sqrt(aligned**2 + np.dot(perpendicular, perpendicular)),
        aligned,
    )
    assert float(actual) == pytest.approx(expected, rel=2.0e-14)


def test_imrphenomtp_euler_angles_match_lalsuite(preserve_scheme):
    parameters = {
        "mass1": 80.0,
        "mass2": 40.0,
        "spin1": (0.2, -0.1, 0.3),
        "spin2": (-0.1, 0.2, -0.2),
        "distance": 100.0,
        "inclination": 0.7,
        "coa_phase": 0.2,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 30.0,
        "f_ref": 30.0,
    }
    _activate_scheme(_scheme.TorchScheme("cpu"))
    core = _build_tp_core(parameters)
    orbit = evolve_imrphenomtp_orbit(
        core,
        parameters["spin1"],
        parameters["spin2"],
    )
    actual = imrphenomtp_euler_angles(core, orbit)
    reference = _lal_angles(parameters)

    assert actual.alpha.numel() == len(reference[1].data.data)
    for result, expected, tolerance in (
        (actual.alpha, reference[1].data.data, 3.0e-6),
        (actual.cosbeta, reference[2].data.data, 2.0e-8),
        (actual.gamma, reference[3].data.data, 3.0e-6),
    ):
        np.testing.assert_allclose(
            result.numpy(),
            np.asarray(expected),
            rtol=0.0,
            atol=tolerance,
        )

    reference_index = orbit.reference_index
    assert float(
        actual.alpha[reference_index] + actual.gamma[reference_index]
    ) == pytest.approx(0.0, abs=2.0e-14)
    torch.testing.assert_close(
        actual.cosbeta[orbit.time_m.numel() - 1 :],
        actual.cosbeta[orbit.time_m.numel() - 2].expand(
            core.time_m.numel() - orbit.time_m.numel() + 1
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_wigner_d_element_matches_lal():
    beta = torch.tensor(0.7, dtype=torch.float64)
    for ell, emm, mprime in (
        (2, 2, 1),
        (2, 2, 0),
        (3, -1, 2),
        (5, 3, -2),
    ):
        actual = wigner_d_element(ell, mprime, emm, beta)
        expected = lal.WignerdMatrix(ell, emm, mprime, float(beta))
        assert float(actual) == pytest.approx(expected, rel=1.0e-13)


def test_imrphenomtp_mode_frames_match_lalsuite(preserve_scheme):
    parameters = _WAVEFORM_CASE
    _activate_scheme(_scheme.TorchScheme("cpu"))
    state, coprecessing, j_frame, l0_frame = _build_imrphenomtp_modes(
        parameters,
        ((2, 2), (2, -2)),
    )

    lal_parameters = _CASES[0].values[0] | {
        "mass1": parameters["mass1"],
        "mass2": parameters["mass2"],
        "spin1": tuple(parameters[f"spin1{axis}"] for axis in "xyz"),
        "spin2": tuple(parameters[f"spin2{axis}"] for axis in "xyz"),
        "distance": parameters["distance"],
        "inclination": parameters["inclination"],
        "coa_phase": parameters["coa_phase"],
        "delta_t": parameters["delta_t"],
        "f_lower": parameters["f_lower"],
        "f_ref": parameters["f_ref"],
    }
    # Initialize LAL's process-global pi-power cache deterministically.
    _lal_angles(lal_parameters)
    arguments = (
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        *(parameters[f"spin1{axis}"] for axis in "xyz"),
        *(parameters[f"spin2{axis}"] for axis in "xyz"),
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters["inclination"],
        parameters["delta_t"],
        parameters["f_lower"],
        parameters["f_ref"],
        parameters["coa_phase"],
        lal.CreateDict(),
        1,
    )
    reference_coprecessing = lalsimulation.SimIMRPhenomTPHM_CoprecModes(*arguments)[0]
    reference_j = lalsimulation.SimIMRPhenomTPHM_JModes(*arguments)[0]
    reference_l0 = lalsimulation.SimIMRPhenomTPHM_L0Modes(*arguments)

    assert state.core.time_m.numel() == _lal_mode(reference_coprecessing, (2, 2)).size
    for mode, actual in coprecessing.items():
        expected = _lal_mode(reference_coprecessing, mode)
        if np.linalg.norm(expected) == 0.0:
            assert torch.count_nonzero(actual) == 0
        else:
            assert _normalized_correlation(expected, actual.numpy()) > 0.999999
    for reference, actual_modes in (
        (reference_j, j_frame),
        (reference_l0, l0_frame),
    ):
        for mode, actual in actual_modes.items():
            expected = _lal_mode(reference, mode)
            if np.linalg.norm(expected) == 0.0:
                assert torch.count_nonzero(actual) == 0
            else:
                assert _normalized_correlation(expected, actual.numpy()) > 0.99998


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"approximant": "IMRPhenomTP"}, True),
        ({"spin1x": 0.2, "spin2y": -0.1}, True),
        ({"long_asc_nodes": 0.3}, True),
        ({"phenom_x_prec_version": 300}, True),
        ({"phenom_xp_convention": 1}, True),
        ({"phenom_xp_final_spin_mod": 4}, True),
        ({"phenom_x_prec_version": 223}, False),
        ({"phenom_xp_convention": 0}, False),
        ({"phenom_xp_final_spin_mod": 2}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"approximant": "IMRPhenomT"}, False),
    ),
)
def test_imrphenomtp_native_support_boundary(parameters, expected):
    assert imrphenomtp_native_supported(parameters) is expected


def test_imrphenomtp_waveform_matches_lalsuite(preserve_scheme):
    parameters = dict(_WAVEFORM_CASE, long_asc_nodes=0.37)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTP", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomtp_td_torch(**parameters)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 8.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.99999


def test_imrphenomtp_public_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_WAVEFORM_CASE, f_lower=40.0, f_ref=40.0)
    import pycbc.waveform.imrphenomtp_waveform_torch as tp_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = tp_module.imrphenomtp_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomTP called lalsimulation")

    monkeypatch.setattr(tp_module, "imrphenomtp_td_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="IMRPhenomTP", **parameters)

    assert native_calls == 1
    for series in actual:
        assert series._data.tensor.device.type == "cpu"
        assert torch.isfinite(series._data.tensor).all()


@pytest.mark.parametrize(
    ("component_enabled", "modifications"),
    (("0", {}), ("1", {"phenom_x_prec_version": 223})),
)
def test_imrphenomtp_disabled_or_unsupported_uses_lal_fallback(
    component_enabled,
    modifications,
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_WAVEFORM_CASE, f_lower=40.0, f_ref=40.0)
    parameters.update(modifications)
    import pycbc.waveform.imrphenomtp_waveform_torch as tp_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported IMRPhenomTP parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(tp_module, "imrphenomtp_td_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTP_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="IMRPhenomTP", **parameters)

    assert lal_calls == 1
    for series in fallback:
        assert isinstance(series._data.tensor, torch.Tensor)
        assert series._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([0.1, 0.2], "scalar"),
        (float("nan"), "finite"),
        (1.01, r"\[-1, 1\]"),
    ),
)
def test_imrphenomtp_final_spin_override_validation(value, message, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _CASES[0].values[0]
    carrier_parameters = {
        key: item for key, item in parameters.items() if key not in ("spin1", "spin2")
    } | {
        "spin1z": parameters["spin1"][2],
        "spin2z": parameters["spin2"][2],
    }
    with pytest.raises(ValueError, match=message):
        _build_imrphenomt_core(
            carrier_parameters,
            final_spin_prec_override=value,
        )


@pytest.mark.parametrize(
    ("spin1", "spin2", "keywords", "message"),
    (
        ((0.1, 0.2), (0.0, 0.0, -0.2), {}, "length three"),
        ((0.1, 0.2, 0.4), (0.0, 0.0, -0.2), {}, "match the carrier"),
        ((float("nan"), 0.2, 0.3), (0.0, 0.0, -0.2), {}, "finite"),
        ((0.1, 0.2, 0.3), (0.0, 0.0, -0.2), {"rtol": 0.0}, "positive"),
        ((0.1, 0.2, 0.3), (0.0, 0.0, -0.2), {"max_steps": 1.5}, "integer"),
    ),
)
def test_imrphenomtp_input_validation(spin1, spin2, keywords, message, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    core = _build_core(_CASES[0].values[0])
    with pytest.raises(ValueError, match=message):
        evolve_imrphenomtp_orbit(core, spin1, spin2, **keywords)


def test_imrphenomtp_active_torch_device(torch_device, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme(torch_device))
    parameters = {
        "mass1": 80.0,
        "mass2": 40.0,
        "spin1": (0.2, -0.1, 0.3),
        "spin2": (-0.1, 0.2, -0.2),
        "delta_t": 1.0 / 2048.0,
        "f_lower": 30.0,
        "f_ref": 30.0,
    }
    core = _build_tp_core(parameters)
    actual = evolve_imrphenomtp_orbit(
        core,
        parameters["spin1"],
        parameters["spin2"],
    )
    angles = imrphenomtp_euler_angles(core, actual)
    expected_dtype = torch.float32 if torch_device == "mps" else torch.float64
    for tensor in (
        actual.time_m,
        actual.velocity,
        actual.lnhat,
        actual.spin1,
        actual.spin2,
        actual.e1,
        angles.alpha,
        angles.cosbeta,
        angles.gamma,
        angles.evolved_final_spin,
    ):
        assert tensor.device.type == torch_device
        assert tensor.dtype == expected_dtype
        assert torch.isfinite(tensor).all()
