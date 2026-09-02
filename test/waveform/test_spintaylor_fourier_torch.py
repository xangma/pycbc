import builtins
import cmath
import math

import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
import pycbc.waveform.spintaylor_fourier_torch as fourier_module  # noqa: E402
import pycbc.waveform.waveform as waveform_module  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.spintaylor_fourier_torch import (  # noqa: E402
    _SpinTaylorFourierTrajectory,
    _Spline,
    _adaptive_irregular_time_branch,
    _average_duplicate_time_knots,
    _assemble_spintaylor_fourier,
    _build_time_stepper,
    _harmonic_numbers,
    _spintaylor_fourier_fd_torch,
    _spintaylor_fourier_trajectory,
    _spintaylor_harmonic,
    spintaylor_t4_fourier_fd_torch,
    spintaylor_t4_fourier_native_supported,
    spintaylor_t5_fourier_fd_torch,
    spintaylor_t5_fourier_native_supported,
)
from pycbc.waveform.torch_waveform_registry import (  # noqa: E402
    TORCH_NATIVE_WAVEFORMS,
    native_approximants,
    try_torch_native_waveform,
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


_FOURIER_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SPINTAYLORT4FOURIER_NATIVE",
    "PYCBC_SPINTAYLORT5FOURIER_NATIVE",
)


def _clear_fourier_flags(monkeypatch):
    for name in _FOURIER_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _public_parameters(approximant="SpinTaylorT4Fourier"):
    return {
        "approximant": approximant,
        "mass1": 40.0,
        "mass2": 30.0,
        "distance": 300.0,
        "inclination": 0.7,
        "coa_phase": 0.37,
        "long_asc_nodes": 0.0,
        "delta_f": 2.0,
        "f_lower": 31.0,
        "f_final": 35.0,
        "f_ref": 31.0,
        "spin1x": 0.08,
        "spin1y": -0.04,
        "spin1z": 0.12,
        "spin2x": -0.03,
        "spin2y": 0.05,
        "spin2z": -0.07,
        "amplitude_order": 0,
        "phase_order": 7,
        "spin_order": 6,
        "tidal_order": 0,
        "eccentricity_order": -1,
        "frame_axis": 0,
        "modes_choice": 0,
        "side_bands": 0,
    }


@pytest.mark.parametrize(
    ("approximant", "spin_order"),
    (
        ("SpinTaylorT4Fourier", -1),
        ("SpinTaylorT4Fourier", 6),
        ("SpinTaylorT4Fourier", 7),
        ("SpinTaylorT5Fourier", -1),
        ("SpinTaylorT5Fourier", 6),
        ("SpinTaylorT5Fourier", 7),
    ),
)
@pytest.mark.parametrize("phase_order", (-1, 7, 8))
@pytest.mark.parametrize("frame_axis", (0, 2))
def test_public_support_accepts_conservative_cpu_contract(
    approximant,
    spin_order,
    phase_order,
    frame_axis,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters(approximant)
    parameters.update(
        spin_order=spin_order,
        phase_order=phase_order,
        frame_axis=frame_axis,
    )
    predicate = (
        spintaylor_t4_fourier_native_supported
        if approximant == "SpinTaylorT4Fourier"
        else spintaylor_t5_fourier_native_supported
    )

    assert predicate(parameters)
    parameters["tidal_order"] = -1
    assert predicate(parameters)


@pytest.mark.parametrize(
    ("approximant", "spin_order", "supported"),
    (
        ("SpinTaylorT4Fourier", -1, True),
        ("SpinTaylorT4Fourier", 6, True),
        ("SpinTaylorT4Fourier", 7, True),
        ("SpinTaylorT5Fourier", -1, True),
        ("SpinTaylorT5Fourier", 6, True),
        ("SpinTaylorT5Fourier", 7, True),
        ("SpinTaylorT4Fourier", 5, False),
        ("SpinTaylorT5Fourier", 5, False),
    ),
)
def test_public_support_has_exact_per_model_spin_orders(
    approximant, spin_order, supported, preserve_scheme
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters(approximant)
    parameters["spin_order"] = spin_order
    predicate = (
        spintaylor_t4_fourier_native_supported
        if approximant == "SpinTaylorT4Fourier"
        else spintaylor_t5_fourier_native_supported
    )
    assert predicate(parameters) is supported


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("amplitude_order", -1),
        ("amplitude_order", 1),
        ("phase_order", 6),
        ("tidal_order", 10),
        ("eccentricity_order", 0),
        ("frame_axis", 1),
        ("modes_choice", 1),
        ("side_bands", 1),
        ("mode_array", [(2, 2)]),
        ("numrel_data", "nr.h5"),
        ("phenom_x_prec_version", 300),
        ("phenom_xp_convention", 1),
        ("phenom_xp_final_spin_mod", 1),
    ),
)
def test_public_support_rejects_ignored_controls(name, value, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters()
    parameters[name] = value
    assert not spintaylor_t4_fourier_native_supported(parameters)


@pytest.mark.parametrize(
    "name",
    (
        "lambda1",
        "lambda2",
        "eccentricity",
        "mean_per_ano",
        *fourier_module._NL_TIDAL_KEYS,
        *fourier_module._TIDAL_EXTENSION_KEYS,
        *fourier_module._NON_GR_KEYS,
    ),
)
def test_public_support_rejects_each_unsupported_scalar_family(name, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters()
    parameters[name] = 1.0
    assert not spintaylor_t4_fourier_native_supported(parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("mass1", math.nan),
        ("distance", math.inf),
        ("spin1x", math.inf),
        ("f_ref", math.nan),
    ),
)
def test_public_support_rejects_nonfinite_fields(name, value, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters()
    parameters[name] = value
    assert not spintaylor_t4_fourier_native_supported(parameters)


def test_public_support_frequency_and_spin_boundaries(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = _public_parameters()
    total_mass_seconds = 70.0 * fourier_module._MTSUN_SI
    f_isco = 1.0 / (6.0**1.5 * math.pi * total_mass_seconds)

    parameters.update(f_final=0.0, f_ref=0.0)
    assert spintaylor_t4_fourier_native_supported(parameters)
    # Values at or below the snapped 30 Hz first bin request the full ISCO
    # output in LAL's regular-grid Fourier wrapper.
    parameters["f_final"] = 30.0
    assert spintaylor_t4_fourier_native_supported(parameters)
    parameters["f_final"] = f_isco
    assert spintaylor_t4_fourier_native_supported(parameters)

    parameters["f_final"] = math.nextafter(f_isco, math.inf)
    assert not spintaylor_t4_fourier_native_supported(parameters)
    parameters.update(f_final=35.0, f_ref=30.0)
    assert not spintaylor_t4_fourier_native_supported(parameters)
    parameters["f_ref"] = f_isco
    assert not spintaylor_t4_fourier_native_supported(parameters)
    parameters.update(f_ref=31.0, spin1x=1.0, spin1y=0.01)
    assert not spintaylor_t4_fourier_native_supported(parameters)


def test_public_support_is_exception_safe_and_cpu_only(preserve_scheme):
    parameters = _public_parameters()
    parameters["mass1"] = object()
    _activate_scheme(_scheme.TorchScheme("cpu"))
    assert not spintaylor_t4_fourier_native_supported(parameters)

    parameters = _public_parameters()
    for device_name in ("cuda", "mps"):
        state = _scheme.TorchScheme("cpu")
        state.torch_device = torch.device(device_name)
        _activate_scheme(state)
        assert not spintaylor_t4_fourier_native_supported(parameters)
        assert not spintaylor_t5_fourier_native_supported(
            _public_parameters("SpinTaylorT5Fourier")
        )

    _activate_scheme(_scheme.CPUScheme())
    assert not spintaylor_t4_fourier_native_supported(parameters)


def test_public_wrappers_reject_unsupported_calls(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    t4_parameters = _public_parameters()
    t4_parameters["amplitude_order"] = 1
    with pytest.raises(ValueError, match="unsupported native SpinTaylorT4"):
        spintaylor_t4_fourier_fd_torch(**t4_parameters)

    t5_parameters = _public_parameters("SpinTaylorT5Fourier")
    t5_parameters["spin_order"] = 5
    with pytest.raises(ValueError, match="unsupported native SpinTaylorT5"):
        spintaylor_t5_fourier_fd_torch(**t5_parameters)


@pytest.mark.parametrize(
    "approximant",
    ("SpinTaylorT4Fourier", "SpinTaylorT5Fourier"),
)
@pytest.mark.parametrize(
    ("global_value", "component_value", "enabled"),
    (
        (None, None, False),
        ("1", None, True),
        ("1", "0", False),
        ("0", "1", True),
    ),
)
def test_public_opt_in_routing_and_component_precedence(
    approximant,
    global_value,
    component_value,
    enabled,
    monkeypatch,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _clear_fourier_flags(monkeypatch)
    component_flag = TORCH_NATIVE_WAVEFORMS[approximant].component_flag
    if global_value is not None:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", global_value)
    if component_value is not None:
        monkeypatch.setenv(component_flag, component_value)

    sentinel = object()
    generator_name = TORCH_NATIVE_WAVEFORMS[approximant].fd_generator
    monkeypatch.setattr(
        fourier_module,
        generator_name,
        lambda **_parameters: sentinel,
    )
    actual = try_torch_native_waveform("fd", _public_parameters(approximant))
    assert (actual is sentinel) is enabled


def test_public_component_flags_are_distinct(monkeypatch, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _clear_fourier_flags(monkeypatch)
    monkeypatch.setenv("PYCBC_SPINTAYLORT4FOURIER_NATIVE", "1")
    monkeypatch.setattr(
        fourier_module,
        "spintaylor_t4_fourier_fd_torch",
        lambda **_parameters: "t4",
    )
    assert try_torch_native_waveform("fd", _public_parameters()) == "t4"
    assert (
        try_torch_native_waveform("fd", _public_parameters("SpinTaylorT5Fourier"))
        is None
    )


def test_public_ports_have_no_sequence_or_time_domain_registration():
    for approximant in ("SpinTaylorT4Fourier", "SpinTaylorT5Fourier"):
        port = TORCH_NATIVE_WAVEFORMS[approximant]
        assert port.fd_generator is not None
        assert port.fd_supported is not None
        assert port.td_generator is None
        assert port.td_modes_generator is None
        assert port.fd_modes_generator is None
        assert port.sequence_generator is None
        assert approximant in native_approximants("fd")
        assert approximant not in native_approximants("sequence")


def test_harmonic_numbers_follow_lal_amplitude_order():
    assert _harmonic_numbers(0) == (2,)
    assert _harmonic_numbers(1) == (1, 2, 3)
    assert _harmonic_numbers(2) == (1, 2, 3, 4)
    assert _harmonic_numbers(3) == (1, 2, 3, 4, 5)
    assert _harmonic_numbers(-1) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize("amplitude_order", (-1, 0, 1, 2, 3))
def test_fourier_harmonics_match_valid_lal_helper(amplitude_order):
    dtype = torch.float64
    x1, x2 = 0.6, 0.4
    velocity = torch.tensor([0.22], dtype=dtype)
    # The Fourier helper consumes LAL's total-mass-normalized internal spins,
    # not the component chi convention used by the public TD helper.
    spin1 = torch.tensor([[0.072, -0.036, 0.108]], dtype=dtype)
    spin2 = torch.tensor([[-0.008, 0.012, -0.016]], dtype=dtype)
    lnhat = torch.tensor([[0.25, -0.35, math.sqrt(0.815)]], dtype=dtype)
    e1 = torch.tensor([[0.813733471206735, 0.5812381937190965, 0.0]], dtype=dtype)

    for harmonic in range(1, 6):
        actual_plus, actual_cross = _spintaylor_harmonic(
            harmonic,
            velocity,
            spin1,
            spin2,
            lnhat,
            e1,
            x1,
            x2,
            amplitude_order,
        )
        expected_plus, expected_cross = (
            lalsimulation.SimInspiralPrecessingPolarizationWaveformHarmonic(
                float(velocity[0]),
                *spin1[0].tolist(),
                *spin2[0].tolist(),
                *lnhat[0].tolist(),
                *e1[0].tolist(),
                x1 - x2,
                x1 * x2,
                1.0,
                harmonic,
                amplitude_order,
            )
        )
        torch.testing.assert_close(
            actual_plus[0],
            torch.tensor(expected_plus, dtype=torch.complex128),
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        torch.testing.assert_close(
            actual_cross[0],
            torch.tensor(expected_cross, dtype=torch.complex128),
            rtol=2.0e-13,
            atol=2.0e-15,
        )


def test_time_solver_tracks_analytic_time_and_phase_and_retains_overshoot():
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0

    def time_rhs(state):
        derivative = torch.zeros_like(state)
        derivative[..., 0] = 2.0 * state[..., 1]
        derivative[..., 1] = state[..., 1] ** 2
        return derivative

    time, state, omega_rate = _adaptive_irregular_time_branch(
        reference,
        -1.0,
        time_rhs,
        lambda *_args: None,
        target_omega=0.8,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    assert torch.all(time[1:] < time[:-1])
    assert torch.all(state[1:, 1] < state[:-1, 1])
    assert state[-2, 1] >= 0.8
    assert state[-1, 1] < 0.8
    torch.testing.assert_close(
        time, 1.0 - torch.reciprocal(state[:, 1]), rtol=0.0, atol=2e-9
    )
    torch.testing.assert_close(
        state[:, 0],
        2.0 * torch.log(state[:, 1]),
        rtol=0.0,
        atol=2e-10,
    )
    torch.testing.assert_close(omega_rate, state[:, 1] ** 2)


def test_time_solver_retains_first_omega_acceleration_boundary_state():
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0

    def time_rhs(state):
        derivative = torch.zeros_like(state)
        omega = state[..., 1]
        derivative[..., 0] = omega
        derivative[..., 1] = 2.0 - (omega - 1.2) ** 2
        return derivative

    _, state, omega_rate = _adaptive_irregular_time_branch(
        reference,
        1.0,
        time_rhs,
        lambda *_args: None,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    # Every accepted state before the last follows increasing omega_dot.  The
    # first non-increasing candidate is retained and then terminates the branch.
    assert len(state) >= 3
    assert torch.all(omega_rate[1:-1] > omega_rate[:-2])
    assert omega_rate[-1] <= omega_rate[-2]
    assert state[-1, 1] > 1.2


def test_compiled_time_step_matches_eager_trajectory():
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0

    def time_rhs(state):
        derivative = torch.zeros_like(state)
        derivative[..., 0] = 2.0 * state[..., 1]
        derivative[..., 1] = state[..., 1] ** 2
        return derivative

    eager_step, eager_compiled = _build_time_stepper(
        reference, time_rhs, compile_step=False
    )
    compiled_step, is_compiled = _build_time_stepper(reference, time_rhs)
    assert not eager_compiled
    if not is_compiled:
        pytest.skip("this Torch build declined the optional JIT trace")

    eager = _adaptive_irregular_time_branch(
        reference,
        -1.0,
        time_rhs,
        lambda *_args: None,
        target_omega=0.8,
        stepper=eager_step,
    )
    compiled = _adaptive_irregular_time_branch(
        reference,
        -1.0,
        time_rhs,
        lambda *_args: None,
        target_omega=0.8,
        stepper=compiled_step,
    )
    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_time_step_unavailable_or_unsafe_trace_falls_back_cleanly(monkeypatch):
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0

    def time_rhs(state):
        derivative = torch.zeros_like(state)
        derivative[..., 0] = state[..., 1]
        derivative[..., 1] = state[..., 1] ** 2
        return derivative

    eager_step, _ = _build_time_stepper(reference, time_rhs, compile_step=False)

    def unavailable_trace(*_args, **_kwargs):
        raise RuntimeError("JIT tracing is unavailable")

    monkeypatch.setattr(torch.jit, "trace", unavailable_trace)
    fallback_step, is_compiled = _build_time_stepper(reference, time_rhs)
    assert not is_compiled

    state = reference.clone()
    step = reference.new_tensor(-0.03)
    slope = time_rhs(state)
    for actual, expected in zip(
        fallback_step(state, step, slope), eager_step(state, step, slope)
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def unsafe_trace(function, *_args, **_kwargs):
        def incorrect_step(state, step, slope):
            candidate, error, end_slope = function(state, step, slope)
            return candidate + 1.0, error, end_slope

        return incorrect_step

    monkeypatch.setattr(torch.jit, "trace", unsafe_trace)
    fallback_step, is_compiled = _build_time_stepper(reference, time_rhs)
    assert not is_compiled
    for actual, expected in zip(
        fallback_step(state, step, slope), eager_step(state, step, slope)
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_time_solver_reuses_rkf45_end_slope():
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0
    rhs_calls = 0
    attempts = 0

    def time_rhs(state):
        nonlocal rhs_calls
        rhs_calls += 1
        derivative = torch.zeros_like(state)
        derivative[..., 0] = 2.0 * state[..., 1]
        derivative[..., 1] = state[..., 1] ** 2
        return derivative

    eager_step, _ = _build_time_stepper(reference, time_rhs, compile_step=False)

    def counting_step(state, step, slope):
        nonlocal attempts
        attempts += 1
        return eager_step(state, step, slope)

    _adaptive_irregular_time_branch(
        reference,
        -1.0,
        time_rhs,
        lambda *_args: None,
        target_omega=0.8,
        stepper=counting_step,
    )

    # One initial-rate call and six RHS evaluations per RKF45 attempt.  The
    # accepted endpoint slope is cached as the next attempt's first stage.
    assert rhs_calls == 1 + 6 * attempts


def test_time_solver_rejected_trial_does_not_rerun_stop_callback():
    reference = torch.zeros(14, dtype=torch.float64)
    reference[1] = 1.0
    stop_checks = 0
    trial_count = 0

    def time_rhs(state):
        derivative = torch.zeros_like(state)
        derivative[1] = 1.0
        return derivative

    def physical_check(_state):
        nonlocal stop_checks
        stop_checks += 1

    def controlled_step(state, _step, slope):
        nonlocal trial_count
        trial_count += 1
        if trial_count == 1:
            # Force a controller rejection without changing the stored knot.
            return state, torch.ones_like(state), slope
        candidate = state.clone()
        candidate[1] = 0.7
        return candidate, torch.zeros_like(state), slope

    _, state, _ = _adaptive_irregular_time_branch(
        reference,
        -1.0,
        time_rhs,
        physical_check,
        target_omega=0.8,
        stepper=controlled_step,
    )

    assert trial_count == 2
    assert stop_checks == 1
    assert len(state) == 2
    assert state[-1, 1] == 0.7


def test_duplicate_time_runs_average_every_non_time_row():
    time = torch.tensor([0.0, 1.0, 1.0, 1.0, 2.0], dtype=torch.float64)
    state = torch.arange(70, dtype=torch.float64).reshape(5, 14)
    omega_rate = torch.arange(5, dtype=torch.float64) + 100.0

    actual_time, actual_state, actual_rate = _average_duplicate_time_knots(
        time, state, omega_rate
    )

    torch.testing.assert_close(
        actual_time, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual_state, torch.stack((state[0], state[1:4].mean(dim=0), state[4]))
    )
    torch.testing.assert_close(
        actual_rate,
        torch.stack((omega_rate[0], omega_rate[1:4].mean(), omega_rate[4])),
    )


def test_sua_shifted_spline_queries_are_explicitly_masked(monkeypatch):
    time = torch.arange(5, dtype=torch.float64)
    state = torch.zeros((5, 14), dtype=torch.float64)
    state[:, 0] = 0.2 * time
    state[:, 1] = 0.02 + 0.01 * time
    state[:, 4] = 1.0
    state[:, 11] = 1.0
    trajectory = _SpinTaylorFourierTrajectory(
        time=time,
        state=state,
        omega_rate=torch.full_like(time, 0.01),
        mass1_fraction=0.6,
        mass2_fraction=0.4,
    )
    frequencies = torch.tensor([0.03, 0.04, 0.05], dtype=torch.float64) / math.pi

    original_evaluate = _Spline.evaluate

    def checked_evaluate(spline, points):
        assert torch.all(points >= spline.knots[0])
        assert torch.all(points <= spline.knots[-1])
        return original_evaluate(spline, points)

    monkeypatch.setattr(_Spline, "evaluate", checked_evaluate)
    plus, cross, _ = _assemble_spintaylor_fourier(
        trajectory,
        frequencies,
        mass_seconds=1.0,
        distance=100.0,
        amplitude_order=0,
    )
    assert torch.isfinite(plus).all()
    assert torch.isfinite(cross).all()
    assert torch.any(plus != 0.0)


def test_sua_uses_distinct_shifted_states_coefficients_and_final_conjugation(
    monkeypatch,
):
    time = torch.linspace(0.0, 10.0, 6, dtype=torch.float64)
    state = torch.zeros((6, 14), dtype=torch.float64)
    state[:, 0] = 0.3 * time
    state[:, 1] = 0.02 + 0.004 * time
    state[:, 4] = 1.0
    state[:, 11] = 1.0
    trajectory = _SpinTaylorFourierTrajectory(
        time=time,
        state=state,
        omega_rate=torch.full_like(time, 2.0),
        mass1_fraction=0.6,
        mass2_fraction=0.4,
    )
    seen_velocities = []

    def synthetic_harmonic(_harmonic, velocity, *_args):
        seen_velocities.extend(velocity.tolist())
        velocity = velocity.to(torch.complex128)
        return velocity + 1j * velocity**2, 2.0 * velocity - 0.5j

    monkeypatch.setattr(fourier_module, "_spintaylor_harmonic", synthetic_harmonic)
    frequency = torch.tensor([0.04 / math.pi], dtype=torch.float64)
    plus, cross, t_isco = _assemble_spintaylor_fourier(
        trajectory,
        frequency,
        mass_seconds=1.0,
        distance=100.0,
        amplitude_order=0,
    )

    coefficients = (
        complex(-1.0 / 6.0, 17.0 / 18.0),
        complex(13.0 / 16.0, -7.0 / 16.0),
        complex(-1.0 / 4.0, -1.0 / 20.0),
        complex(1.0 / 48.0, 11.0 / 720.0),
    )
    shifted_velocities = [
        (0.02 + 0.004 * (5.0 + 0.5 * shift)) ** (1.0 / 3.0) for shift in range(-3, 4)
    ]
    expected_plus_sum = sum(
        coefficients[abs(shift)] * 0.5 * (velocity + 1j * velocity**2)
        for shift, velocity in zip(range(-3, 4), shifted_velocities)
    )
    expected_cross_sum = sum(
        coefficients[abs(shift)] * 0.5 * (2.0 * velocity - 0.5j)
        for shift, velocity in zip(range(-3, 4), shifted_velocities)
    )
    prefactor = (
        2.0
        * math.sqrt(2.0 * math.pi)
        * 0.24
        * fourier_module._C_SI
        / (100.0 * fourier_module._MPC_SI)
    )
    phase = 2.0 * math.pi * float(frequency[0]) * (5.0 - 10.0)
    phase -= 2.0 * 1.5 + math.pi / 4.0
    phase_factor = prefactor * cmath.exp(1j * phase)

    assert len(seen_velocities) == 7
    assert len(set(seen_velocities)) == 7
    torch.testing.assert_close(t_isco, torch.tensor(10.0, dtype=torch.float64))
    torch.testing.assert_close(
        plus[0],
        torch.tensor(
            (expected_plus_sum * phase_factor).conjugate(),
            dtype=torch.complex128,
        ),
        rtol=2.0e-13,
        atol=0.0,
    )
    torch.testing.assert_close(
        cross[0],
        torch.tensor(
            (expected_cross_sum * phase_factor).conjugate(),
            dtype=torch.complex128,
        ),
        rtol=2.0e-13,
        atol=0.0,
    )


# Generated from LALSuite commit c90e1175 after correcting only the three
# ``params = (void *)&paramsT*`` assignments in the irregular-orbit driver.
_CORRECTED_LAL_FOURIER_ORACLES = {
    "SpinTaylorT4": {
        "epoch": -2.407743300,
        "trajectory_length": 820,
        "forward_omega": (
            0.0083558985352737286,
            0.008356010819471002,
            0.0083565723240972122,
        ),
        "isco_last_hp": complex(-5.1025865069799564e-24, 1.9001732020726271e-24),
        "physical_last": (
            241,
            complex(9.1474746831420934e-27, -1.2904328074264107e-24),
            complex(-1.2904326368827134e-24, -9.147650743339802e-27),
            2.0e-8,
        ),
        "rows": {
            15: (
                complex(-8.112835528164946e-23, 1.287439041578701e-23),
                complex(1.287439042438688e-23, 8.112835518655139e-23),
            ),
            35: (
                complex(2.778734619013586e-23, 1.043790291564931e-23),
                complex(1.043790498858016e-23, -2.778734859878857e-23),
            ),
            49: (
                complex(8.22447027290886e-24, -1.765809125700482e-23),
                complex(-1.7658093586774e-23, -8.224494124719698e-24),
            ),
        },
    },
    "SpinTaylorT5": {
        "epoch": -2.406303437,
        "trajectory_length": 937,
        "forward_omega": (
            0.0083558985352737286,
            0.0083560107966633469,
            0.0083565721872105193,
        ),
        "isco_last_hp": complex(2.7853076915719165e-24, 1.6226911650423825e-24),
        # The terminal T5 knot is a numerically singular equal-time run.  Its
        # cleaned spectrum is stable at sub-1e-3 level, but its raw omega_dot
        # is intentionally not an oracle quantity.
        "physical_last": (
            194,
            complex(8.6869835785868333e-25, 2.5373500962671036e-25),
            complex(2.5373543627775434e-25, -8.6869774586542863e-25),
            8.0e-4,
        ),
        "rows": {
            15: (
                complex(-7.471473831723836e-23, 3.415609190961098e-23),
                complex(3.415609189234929e-23, 7.471473822323784e-23),
            ),
            35: (
                complex(2.763392642143667e-23, -1.082359820494521e-23),
                complex(-1.082359831372748e-23, -2.763392959167411e-23),
            ),
            49: (
                complex(-1.135329398087589e-23, -1.57834268469805e-23),
                complex(-1.578344856657191e-23, 1.135328426404473e-23),
            ),
        },
    },
}


@pytest.mark.parametrize(
    ("dynamics", "approximant", "component_flag", "spin_order"),
    (
        (
            "SpinTaylorT4",
            "SpinTaylorT4Fourier",
            "PYCBC_SPINTAYLORT4FOURIER_NATIVE",
            -1,
        ),
        (
            "SpinTaylorT5",
            "SpinTaylorT5Fourier",
            "PYCBC_SPINTAYLORT5FOURIER_NATIVE",
            7,
        ),
    ),
)
def test_public_fd_matches_corrected_lal_through_isco(
    dynamics,
    approximant,
    component_flag,
    spin_order,
    monkeypatch,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _clear_fourier_flags(monkeypatch)
    monkeypatch.setenv(component_flag, "1")

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("corrected-LAL oracle call fell back to LAL")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    plus, cross = get_fd_waveform(
        approximant=approximant,
        mass1=10.0,
        mass2=8.0,
        distance=100.0,
        inclination=0.0,
        coa_phase=0.37,
        long_asc_nodes=0.0,
        delta_f=2.0,
        f_lower=30.0,
        f_final=0.0,
        f_ref=30.0,
        spin1x=0.08,
        spin1y=-0.04,
        spin1z=0.12,
        spin2x=-0.03,
        spin2y=0.05,
        spin2z=-0.07,
        amplitude_order=0,
        phase_order=7,
        spin_order=spin_order,
        tidal_order=0,
    )
    plus_tensor = plus._data.tensor
    cross_tensor = cross._data.tensor
    oracle = _CORRECTED_LAL_FOURIER_ORACLES[dynamics]

    assert len(plus) == len(cross) == 123
    assert plus.delta_f == cross.delta_f == 2.0
    assert torch.count_nonzero(plus_tensor[:15]) == 0
    assert torch.count_nonzero(cross_tensor[:15]) == 0
    assert torch.all(plus_tensor[15:] != 0.0)
    assert torch.all(cross_tensor[15:] != 0.0)
    assert float(plus.start_time) == pytest.approx(oracle["epoch"], abs=1.1e-9)
    assert plus.start_time == cross.start_time
    for index, expected in oracle["rows"].items():
        torch.testing.assert_close(
            torch.stack((plus_tensor[index], cross_tensor[index])),
            torch.tensor(expected, dtype=torch.complex128),
            rtol=2.0e-9,
            atol=2.0e-31,
        )
    torch.testing.assert_close(
        plus_tensor[122],
        torch.tensor(oracle["isco_last_hp"], dtype=torch.complex128),
        rtol=2.0e-9,
        atol=2.0e-32,
    )


@pytest.mark.parametrize(
    "approximant",
    ("SpinTaylorT4Fourier", "SpinTaylorT5Fourier"),
)
@pytest.mark.parametrize("fallback_reason", ("disabled", "amplitude", "dquad_mon"))
def test_public_disabled_or_unsupported_calls_use_lal_fallback(
    approximant,
    fallback_reason,
    monkeypatch,
    preserve_scheme,
):
    class LALFallbackReached(Exception):
        pass

    def unexpected_native(**_parameters):
        raise AssertionError("disabled/unsupported Fourier call reached Torch")

    def recording_lal(*_args, **_kwargs):
        raise LALFallbackReached

    _activate_scheme(_scheme.TorchScheme("cpu"))
    _clear_fourier_flags(monkeypatch)
    port = TORCH_NATIVE_WAVEFORMS[approximant]
    monkeypatch.setattr(fourier_module, port.fd_generator, unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    if fallback_reason == "disabled":
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
        monkeypatch.setenv(port.component_flag, "0")
    else:
        monkeypatch.setenv(port.component_flag, "1")

    parameters = _public_parameters(approximant)
    if fallback_reason == "amplitude":
        parameters["amplitude_order"] = 1
    elif fallback_reason == "dquad_mon":
        parameters["dquad_mon1"] = 0.1
    with pytest.raises(LALFallbackReached):
        get_fd_waveform(**parameters)


@pytest.mark.parametrize("dynamics", ("SpinTaylorT4", "SpinTaylorT5"))
def test_private_fd_matches_corrected_lal_fourier_oracle(
    dynamics, monkeypatch, preserve_scheme
):
    """Lock the end-to-end contract from a pointer-corrected LAL 7.26.1."""

    _activate_scheme(_scheme.TorchScheme("cpu"))
    original_build_stepper = fourier_module._build_time_stepper
    original_trajectory = _spintaylor_fourier_trajectory
    compiled_steps = []
    recorded_trajectories = []

    def recording_build_stepper(*args, **kwargs):
        stepper, compiled = original_build_stepper(*args, **kwargs)
        compiled_steps.append(compiled)
        return stepper, compiled

    monkeypatch.setattr(fourier_module, "_build_time_stepper", recording_build_stepper)

    def recording_trajectory(*args, **kwargs):
        diagnostics = {}
        trajectory = original_trajectory(*args, **kwargs, diagnostics=diagnostics)
        recorded_trajectories.append((trajectory, diagnostics))
        return trajectory

    monkeypatch.setattr(
        fourier_module,
        "_spintaylor_fourier_trajectory",
        recording_trajectory,
    )
    plus, cross = _spintaylor_fourier_fd_torch(
        dynamics,
        mass1=10.0,
        mass2=8.0,
        distance=100.0,
        inclination=0.0,
        coa_phase=0.37,
        long_asc_nodes=0.0,
        delta_f=2.0,
        f_lower=30.0,
        f_final=600.0,
        f_ref=30.0,
        spin1x=0.08,
        spin1y=-0.04,
        spin1z=0.12,
        spin2x=-0.03,
        spin2y=0.05,
        spin2z=-0.07,
        amplitude_order=0,
        phase_order=7,
        spin_order=6,
        tidal_order=0,
    )
    plus_tensor = plus._data.tensor
    cross_tensor = cross._data.tensor
    oracle = _CORRECTED_LAL_FOURIER_ORACLES[dynamics]

    # Oracle correctness is independent of whether this Torch build accepts
    # the optional trace; the eager fallback has the same contract.
    assert len(compiled_steps) == 1
    assert len(recorded_trajectories) == 1
    trajectory, diagnostics = recorded_trajectories[0]
    assert len(trajectory.time) == oracle["trajectory_length"]
    assert diagnostics["low"] == {
        "attempts": 91,
        "accepted": 90,
        "rejected": 0,
        "stop_reason": "frequency",
    }
    assert diagnostics["high"]["stop_reason"] == (
        "physical" if dynamics == "SpinTaylorT4" else "omega_acceleration"
    )
    reference_index = diagnostics["low"]["accepted"]
    torch.testing.assert_close(
        trajectory.time[reference_index : reference_index + 3],
        torch.tensor((0.0, 1.0, 6.0), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        trajectory.state[reference_index : reference_index + 3, 1],
        torch.tensor(oracle["forward_omega"], dtype=torch.float64),
        rtol=2.0e-14,
        atol=0.0,
    )
    if dynamics == "SpinTaylorT4":
        assert diagnostics["high"]["accepted"] == 729
    terminal_index, terminal_plus, terminal_cross, terminal_rtol = oracle[
        "physical_last"
    ]
    assert len(plus) == len(cross) == 300
    assert plus.delta_f == cross.delta_f == 2.0
    assert torch.count_nonzero(plus_tensor[:15]) == 0
    assert torch.count_nonzero(cross_tensor[:15]) == 0
    assert torch.all(plus_tensor[15 : terminal_index + 1] != 0.0)
    assert torch.all(cross_tensor[15 : terminal_index + 1] != 0.0)
    assert torch.count_nonzero(plus_tensor[terminal_index + 1 :]) == 0
    assert torch.count_nonzero(cross_tensor[terminal_index + 1 :]) == 0
    assert float(plus.start_time) == pytest.approx(oracle["epoch"], abs=1.1e-9)
    assert plus.start_time == cross.start_time

    for index, expected in oracle["rows"].items():
        actual = torch.stack((plus_tensor[index], cross_tensor[index]))
        torch.testing.assert_close(
            actual,
            torch.tensor(expected, dtype=torch.complex128),
            rtol=2.0e-9,
            atol=2.0e-31,
        )

    # Bin 122 is 244 Hz, the last grid point below the Schwarzschild ISCO.
    torch.testing.assert_close(
        plus_tensor[122],
        torch.tensor(oracle["isco_last_hp"], dtype=torch.complex128),
        rtol=2.0e-9,
        atol=2.0e-32,
    )
    torch.testing.assert_close(
        torch.stack((plus_tensor[terminal_index], cross_tensor[terminal_index])),
        torch.tensor((terminal_plus, terminal_cross), dtype=torch.complex128),
        rtol=terminal_rtol,
        atol=2.0e-31,
    )


def test_private_fd_epoch_uses_requested_lower_frequency_not_grid_bin(
    monkeypatch, preserve_scheme
):
    _activate_scheme(_scheme.TorchScheme("cpu"))

    def synthetic_trajectory(
        _dynamics,
        mass1,
        mass2,
        _f_start,
        _f_ref,
        _spin1,
        _spin2,
        _lnhat,
        _e1,
        _coa_phase,
        _matter,
        device,
    ):
        mass_seconds = (mass1 + mass2) * fourier_module._MTSUN_SI
        knot_frequencies = torch.tensor(
            [20.0, 60.0], dtype=torch.float64, device=device
        )
        state = torch.zeros((2, 14), dtype=torch.float64, device=device)
        state[:, 1] = math.pi * mass_seconds * knot_frequencies
        return _SpinTaylorFourierTrajectory(
            time=knot_frequencies - 60.0,
            state=state,
            omega_rate=torch.ones(2, dtype=torch.float64, device=device),
            mass1_fraction=mass1 / (mass1 + mass2),
            mass2_fraction=mass2 / (mass1 + mass2),
        )

    def synthetic_assembler(trajectory, frequencies, *_args):
        zeros = torch.zeros_like(frequencies, dtype=torch.complex128)
        return zeros, zeros.clone(), trajectory.time[-1]

    monkeypatch.setattr(
        fourier_module, "_spintaylor_fourier_trajectory", synthetic_trajectory
    )
    monkeypatch.setattr(
        fourier_module, "_assemble_spintaylor_fourier", synthetic_assembler
    )
    plus, cross = _spintaylor_fourier_fd_torch(
        "SpinTaylorT4",
        mass1=40.0,
        mass2=30.0,
        distance=100.0,
        delta_f=2.0,
        f_lower=31.0,
        f_final=35.0,
        f_ref=31.0,
        amplitude_order=0,
        tidal_order=0,
    )

    mass_seconds = 70.0 * fourier_module._MTSUN_SI
    requested_epoch = fourier_module._gps_floor(-29.0 * mass_seconds)
    grid_bin_epoch = fourier_module._gps_floor(-30.0 * mass_seconds)
    assert float(plus.start_time) == pytest.approx(requested_epoch, abs=1.0e-12)
    assert plus.start_time == cross.start_time
    assert abs(float(plus.start_time) - grid_bin_epoch) > 1.0e-5


@pytest.mark.parametrize(
    ("approximant", "component_flag"),
    (
        (
            "SpinTaylorT4Fourier",
            "PYCBC_SPINTAYLORT4FOURIER_NATIVE",
        ),
        (
            "SpinTaylorT5Fourier",
            "PYCBC_SPINTAYLORT5FOURIER_NATIVE",
        ),
    ),
)
def test_public_fd_layout_rotation_device_and_no_lalsimulation(
    approximant, component_flag, monkeypatch, preserve_scheme
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _clear_fourier_flags(monkeypatch)
    monkeypatch.setenv(component_flag, "1")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lalsimulation" or name.startswith("lalsimulation."):
            raise AssertionError("public Torch Fourier port imported lalsimulation")
        return original_import(name, *args, **kwargs)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("public Torch Fourier port called lalsimulation")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    parameters = {
        "approximant": approximant,
        "mass1": 40.0,
        "mass2": 30.0,
        "distance": 300.0,
        "inclination": 0.7,
        "coa_phase": 0.37,
        "delta_f": 2.0,
        "f_lower": 31.0,
        "f_final": 35.0,
        "spin1x": 0.08,
        "spin1y": -0.04,
        "spin1z": 0.12,
        "spin2x": -0.03,
        "spin2y": 0.05,
        "spin2z": -0.07,
        "amplitude_order": 0,
        "phase_order": 7,
        "spin_order": 6,
        "tidal_order": 0,
    }
    baseline_plus, baseline_cross = get_fd_waveform(
        **parameters, f_ref=31.0, long_asc_nodes=0.0
    )
    node_angle = 0.23
    actual_plus, actual_cross = get_fd_waveform(
        **parameters, f_ref=0.0, long_asc_nodes=node_angle
    )

    # floor(nextafter(31) / 2) selects 30 Hz, and 35 Hz is excluded.
    assert len(actual_plus) == 18
    assert len(actual_cross) == 18
    assert torch.count_nonzero(actual_plus._data.tensor[:15]) == 0
    assert torch.count_nonzero(actual_cross._data.tensor[:15]) == 0
    assert actual_plus._data.tensor.dtype == torch.complex128
    assert actual_plus._data.tensor.device.type == "cpu"
    assert actual_plus.start_time == baseline_plus.start_time
    assert float(actual_plus.start_time) < 0.0

    cosine = math.cos(2.0 * node_angle)
    sine = math.sin(2.0 * node_angle)
    torch.testing.assert_close(
        actual_plus._data.tensor,
        cosine * baseline_plus._data.tensor + sine * baseline_cross._data.tensor,
        rtol=2.0e-13,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_cross._data.tensor,
        cosine * baseline_cross._data.tensor - sine * baseline_plus._data.tensor,
        rtol=2.0e-13,
        atol=0.0,
    )
