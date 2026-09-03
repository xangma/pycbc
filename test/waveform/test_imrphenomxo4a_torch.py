from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence  # noqa: E402
from pycbc.waveform.imrphenomxo4a_torch import (  # noqa: E402
    _antisymmetric_amplitude_ratio,
    _antisymmetric_phase,
    _build_pnr_angle_splines,
    _build_model,
    _pnr_base_transition_frequencies,
    _pnr_higher_mode_map,
    _pnr_interpolation_bounds,
    _pnr_interpolation_frequencies,
    _pnr_spline_angles,
    _xo4a_ringdowns,
    _xo4a_inputs,
    imrphenomxo4a_fd_sequence_torch,
    imrphenomxo4a_fd_torch,
    imrphenomxo4a_native_supported,
    imrphenomxo4a_sequence_native_supported,
    imrphenomxo4a_symmetric22_fd_sequence_torch,
    imrphenomxo4a_symmetric22_fd_torch,
)
from pycbc.waveform.imrphenomxas_torch import _get_cutoff_fMs  # noqa: E402
from pycbc.waveform.imrphenomxhm_mode21_torch import (  # noqa: E402
    _qnm_fring_21,
)
from pycbc.waveform.imrphenomxhm_mode32_torch import (  # noqa: E402
    qnm_fring32_fit,
)
from pycbc.waveform.imrphenomxhm_mode33_torch import (  # noqa: E402
    _qnm_fring_33,
)
from pycbc.waveform.imrphenomxhm_mode44_torch import (  # noqa: E402
    _qnm_fring_44,
)
from pycbc.waveform import imrphenomx_utils_torch as IMRPhenomX_utils  # noqa: E402


_BASE_PARAMS = dict(
    distance=400.0,
    inclination=0.9,
    coa_phase=0.3,
    delta_f=0.25,
    f_lower=20.0,
    f_final=512.0,
    f_ref=30.0,
)
_CALIBRATED_SOURCE = dict(
    mass1=40.0,
    mass2=20.0,
    spin1x=0.35,
    spin1y=-0.15,
    spin1z=0.2,
    spin2x=-0.1,
    spin2y=0.08,
    spin2z=-0.05,
)
_TAPERED_SOURCE = dict(
    mass1=60.0,
    mass2=4.0,
    spin1x=0.45,
    spin1y=-0.1,
    spin1z=0.35,
    spin2x=0.05,
    spin2y=0.03,
    spin2z=-0.2,
)
_ALIGNED_SOURCE = dict(
    mass1=30.0,
    mass2=20.0,
    spin1z=0.2,
    spin2z=-0.1,
)
_SOURCE_PARAMETERS = (
    pytest.param(
        _CALIBRATED_SOURCE,
        id="calibrated-q2",
    ),
    pytest.param(
        dict(
            mass1=15.0,
            mass2=35.0,
            spin1x=0.1,
            spin1y=0.2,
            spin1z=-0.1,
            spin2x=-0.25,
            spin2y=0.12,
            spin2z=0.3,
        ),
        id="reversed-masses",
    ),
    pytest.param(
        dict(
            mass1=50.0,
            mass2=5.0,
            spin1x=0.45,
            spin1y=-0.1,
            spin1z=0.35,
            spin2x=0.05,
            spin2y=0.03,
            spin2z=-0.2,
        ),
        id="high-q",
    ),
    pytest.param(
        dict(
            mass1=36.0,
            mass2=24.0,
            spin1x=0.3,
            spin1y=0.1,
            spin1z=0.25,
            spin2x=-0.12,
            spin2y=0.04,
            spin2z=-0.15,
            f_lower=19.3,
            f_ref=0.0,
        ),
        id="reference-at-lower-bound",
    ),
    pytest.param(
        _ALIGNED_SOURCE,
        id="aligned-spin-fallback",
    ),
)


@pytest.fixture
def torch_cpu_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("cpu")
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
        "PYCBC_IMRPHENOMXO4A_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)


def _symmetric22_lal_params():
    lal_params = lal.CreateDict()
    mode_array = lalsimulation.SimInspiralCreateModeArray()
    for emm in (2, -2):
        lalsimulation.SimInspiralModeArrayActivateMode(mode_array, 2, emm)
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(
        lal_params,
        mode_array,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        300,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPConvention(
        lal_params,
        1,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXAntisymmetricWaveform(
        lal_params,
        0,
    )
    return lal_params


def _antisymmetric_lal_params():
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        300,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPConvention(
        lal_params,
        1,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPNRUseTunedAngles(
        lal_params,
        1,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPNRUseTunedCoprec(
        lal_params,
        1,
    )
    return lal_params


def _default_lal_params():
    lal_params = _antisymmetric_lal_params()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXAntisymmetricWaveform(
        lal_params,
        1,
    )
    return lal_params


def _mode_lal_params(modes):
    lal_params = _antisymmetric_lal_params()
    mode_array = lalsimulation.SimInspiralCreateModeArray()
    for ell, emm in modes:
        lalsimulation.SimInspiralModeArrayActivateMode(
            mode_array,
            ell,
            emm,
        )
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(
        lal_params,
        mode_array,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXAntisymmetricWaveform(
        lal_params,
        1,
    )
    return lal_params


def _lalsimulation_regular(params, lal_params):
    spins = tuple(
        params.get(f"spin{body}{axis}", 0.0)
        for body in (1, 2)
        for axis in "xyz"
    )
    return lalsimulation.SimInspiralChooseFDWaveform(
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        *spins,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["inclination"],
        params["coa_phase"],
        0.0,
        0.0,
        0.0,
        params["delta_f"],
        params["f_lower"],
        params["f_final"],
        params["f_ref"],
        lal_params,
        lalsimulation.IMRPhenomXO4a,
    )


def _lalsimulation_symmetric22(params):
    return _lalsimulation_regular(params, _symmetric22_lal_params())


def _lalsimulation_default(params):
    return _lalsimulation_regular(params, _default_lal_params())


def _lalsimulation_sequence(params, sample_points, lal_params):
    frequencies = lal.CreateREAL8Vector(len(sample_points))
    frequencies.data[:] = sample_points
    spins = tuple(
        params.get(f"spin{body}{axis}", 0.0)
        for body in (1, 2)
        for axis in "xyz"
    )
    return lalsimulation.SimInspiralChooseFDWaveformSequence(
        params["coa_phase"],
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        *spins,
        params["f_ref"],
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["inclination"],
        lal_params,
        lalsimulation.IMRPhenomXO4a,
        frequencies,
    )


def _lalsimulation_symmetric22_sequence(params, sample_points):
    return _lalsimulation_sequence(
        params,
        sample_points,
        _symmetric22_lal_params(),
    )


def _lalsimulation_default_sequence(params, sample_points):
    return _lalsimulation_sequence(
        params,
        sample_points,
        _default_lal_params(),
    )


def _lalsimulation_antisymmetric_waveform(params):
    spins = tuple(
        params.get(f"spin{body}{axis}", 0.0)
        for body in (1, 2)
        for axis in "xyz"
    )
    return lalsimulation.SimIMRPhenomX_PNR_GenerateAntisymmetricWaveform(
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        *spins,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["inclination"],
        params["delta_f"],
        params["f_lower"],
        params["f_final"],
        params["f_ref"],
        params["coa_phase"],
        _antisymmetric_lal_params(),
    )


def _lalsimulation_antisymmetric_ratio(params):
    spins = tuple(
        params.get(f"spin{body}{axis}", 0.0)
        for body in (1, 2)
        for axis in "xyz"
    )
    return lalsimulation.SimIMRPhenomX_PNR_GenerateAntisymmetricAmpRatio(
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        *spins,
        params["inclination"],
        params["delta_f"],
        params["f_lower"],
        params["f_final"],
        params["f_ref"],
        _antisymmetric_lal_params(),
    )


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _antisymmetric_ratio_reference(model, frequencies):
    inputs = model.inputs
    single_spin = model.single_spin
    eta = float(single_spin.symmetric_mass_ratio)
    delta = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    theta = float(single_spin.antisymmetric_angle)
    chi = float(single_spin.antisymmetric_magnitude)
    coefficient = (
        18.0387
        + 15.4509 * eta
        + 55.1140 * theta
        - 203.6290 * eta * theta
    )
    ringdown_frequency = float(
        _get_cutoff_fMs(
            inputs.mass1,
            inputs.mass2,
            inputs.chi1_l,
            inputs.chi2_l,
            inputs.chip,
            final_spin=model.final_spin,
            coprecessing_deviations=model.coprecessing_deviations,
        )[0]
    )
    velocity = np.cbrt(np.pi * np.minimum(frequencies, ringdown_frequency))
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    ratio = (
        21.0
        * velocity2
        * (1.0 + delta)
        * chi
        * np.sin(theta)
        / (
            2.0
            * (
                42.0
                + 84.0 * np.pi * velocity3
                + velocity2 * (55.0 * eta - 107.0)
                - 28.0
                * velocity3
                * (1.0 + delta - eta)
                * chi
                * np.cos(theta)
            )
        )
        * (1.0 + coefficient * velocity3 * velocity2)
    )

    width = 80
    if width > len(ratio) - 1:
        width = len(ratio) // 2
    half_width = width // 2
    for start in range(len(ratio) - width - 1):
        differences = np.diff(frequencies[start : start + width + 2])
        ratio[start + half_width] = np.sum(
            ratio[start : start + width + 1] * differences
        ) / np.sum(differences)
    return ratio


def _calibrated_model(params):
    inputs = _xo4a_inputs(params)
    return _build_model(
        inputs,
        inclination=float(params["inclination"]),
        coa_phase=float(params["coa_phase"]),
    )


def _lalsimulation_pnr_diagnostic(function, params):
    spins = tuple(
        params.get(f"spin{body}{axis}", 0.0)
        for body in (1, 2)
        for axis in "xyz"
    )
    return function(
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        *spins,
        _antisymmetric_lal_params(),
    )


@pytest.mark.parametrize("source", (_CALIBRATED_SOURCE, _TAPERED_SOURCE))
def test_xo4a_final_spins_match_lalsimulation(source, torch_cpu_scheme):
    params = {**_BASE_PARAMS, **source}
    model = _calibrated_model(params)

    diagnostics = (
        (model.final_spin, lalsimulation.SimPhenomPNRafinal),
        (
            model.aligned_final_spin,
            lalsimulation.SimPhenomPNRafinal_nonprec,
        ),
        (
            model.precessing_final_spin,
            lalsimulation.SimPhenomPNRafinal_prec,
        ),
    )
    for actual, function in diagnostics:
        expected = _lalsimulation_pnr_diagnostic(function, params)
        assert float(actual) == pytest.approx(expected, rel=2.0e-14)


@pytest.mark.parametrize("source", (_CALIBRATED_SOURCE, _TAPERED_SOURCE))
def test_xo4a_effective_ringdowns_match_lal_shift(source, torch_cpu_scheme):
    params = {**_BASE_PARAMS, **source}
    model = _calibrated_model(params)
    ringdowns = _xo4a_ringdowns(model)
    expected_shift = _lalsimulation_pnr_diagnostic(
        lalsimulation.SimPhenomPNRfRINGEffShiftDividedByEmm,
        params,
    )
    assert float(ringdowns.shift_per_m) == pytest.approx(
        expected_shift,
        rel=3.0e-13,
        abs=1.0e-16,
    )

    remnant = IMRPhenomX_utils.get_remnant_fMs(
        model.inputs.mass1,
        model.inputs.mass2,
        model.inputs.chi1_l,
        model.inputs.chi2_l,
        final_spin=model.precessing_final_spin,
    )
    expected_mode21 = (
        _qnm_fring_21(model.precessing_final_spin)
        / (1.0 - remnant.radiated_energy)
        - expected_shift
    )
    torch.testing.assert_close(
        ringdowns.mode21,
        torch.as_tensor(expected_mode21),
        rtol=2.0e-14,
        atol=0.0,
    )
    expected_mode33 = (
        _qnm_fring_33(model.precessing_final_spin)
        / (1.0 - remnant.radiated_energy)
        - 3.0 * expected_shift
    )
    torch.testing.assert_close(
        ringdowns.mode33,
        torch.as_tensor(expected_mode33),
        rtol=2.0e-14,
        atol=0.0,
    )
    expected_mode32 = (
        qnm_fring32_fit(model.precessing_final_spin)
        / (1.0 - remnant.radiated_energy)
        - 2.0 * expected_shift
    )
    torch.testing.assert_close(
        ringdowns.mode32,
        torch.as_tensor(expected_mode32),
        rtol=2.0e-14,
        atol=0.0,
    )
    expected_mode44 = (
        _qnm_fring_44(model.precessing_final_spin)
        / (1.0 - remnant.radiated_energy)
        - 4.0 * expected_shift
    )
    torch.testing.assert_close(
        ringdowns.mode44,
        torch.as_tensor(expected_mode44),
        rtol=2.0e-14,
        atol=0.0,
    )


def test_pnr_higher_mode_transition_guards(torch_cpu_scheme):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    model = _calibrated_model(params)
    carrier = torch.tensor(0.08, dtype=torch.float64)
    guarded_model = replace(
        model,
        alpha_parameters=replace(
            model.alpha_parameters,
            mf_lower=torch.tensor(0.31, dtype=torch.float64),
        ),
        beta_parameters=replace(
            model.beta_parameters,
            mf_lower=torch.tensor(0.005, dtype=torch.float64),
        ),
    )

    lower, upper = _pnr_base_transition_frequencies(
        guarded_model,
        carrier,
    )

    torch.testing.assert_close(upper, carrier)
    torch.testing.assert_close(lower, 0.5 * carrier)


def test_higher_mode_mapping_extends_shared_angle_spline(torch_cpu_scheme):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    model = _calibrated_model(params)
    ringdowns = _xo4a_ringdowns(model)
    mode_map = _pnr_higher_mode_map(
        model,
        2,
        1,
        ringdowns.carrier,
        ringdowns.mode21,
    )
    bounds = torch.tensor(
        (_BASE_PARAMS["f_lower"], _BASE_PARAMS["f_final"]),
        dtype=torch.float64,
    )
    mapped = mode_map.evaluate(
        model.inputs.total_mass_seconds * bounds
    ) / model.inputs.total_mass_seconds
    lower, upper = _pnr_interpolation_bounds(
        model,
        *bounds.tolist(),
        (mode_map,),
    )
    assert lower <= float(torch.min(mapped))
    assert upper >= float(torch.max(mapped))

    angle_frequencies = _pnr_interpolation_frequencies(
        model,
        *bounds.tolist(),
        (mode_map,),
    )
    assert float(angle_frequencies[0]) <= lower
    assert float(angle_frequencies[-1]) >= upper
    splines = _build_pnr_angle_splines(model, angle_frequencies)
    angles = _pnr_spline_angles(splines, mapped)
    assert all(angle.shape == mapped.shape for angle in angles)
    assert all(bool(torch.all(torch.isfinite(angle))) for angle in angles)


def test_antisymmetric_amplitude_ratio_clamps_at_ringdown(torch_cpu_scheme):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    model = _calibrated_model(params)
    ringdown_frequency = float(
        _get_cutoff_fMs(
            model.inputs.mass1,
            model.inputs.mass2,
            model.inputs.chi1_l,
            model.inputs.chi2_l,
            model.inputs.chip,
            final_spin=model.final_spin,
            coprecessing_deviations=model.coprecessing_deviations,
        )[0]
    )
    frequencies = torch.tensor(
        [0.5 * ringdown_frequency, 1.5 * ringdown_frequency],
        dtype=torch.float64,
    )
    actual = _antisymmetric_amplitude_ratio(model, frequencies)
    expected = _antisymmetric_ratio_reference(model, frequencies.numpy())

    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-14)
    at_ringdown = _antisymmetric_ratio_reference(
        model,
        np.array([ringdown_frequency, ringdown_frequency]),
    )
    assert actual[1].item() == pytest.approx(at_ringdown[0], rel=2.0e-14)


def test_antisymmetric_amplitude_ratio_matches_lal_smoothing(
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    model = _calibrated_model(params)
    frequencies = np.geomspace(0.003, 0.29, 127)
    frequencies[1::3] *= 1.0001

    actual = _antisymmetric_amplitude_ratio(
        model,
        torch.tensor(frequencies, dtype=torch.float64),
    )
    expected = _antisymmetric_ratio_reference(model, frequencies.copy())

    np.testing.assert_allclose(actual.numpy(), expected, rtol=3.0e-14)


def test_antisymmetric_amplitude_ratio_matches_lalsimulation(
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    model = _calibrated_model(params)
    expected_ratio, expected_frequencies = (
        _lalsimulation_antisymmetric_ratio(params)
    )
    geometric_frequencies = (
        torch.from_numpy(np.asarray(expected_frequencies.data))
        * model.inputs.total_mass_seconds
    )
    actual = _antisymmetric_amplitude_ratio(model, geometric_frequencies)

    np.testing.assert_allclose(
        actual.numpy(),
        np.asarray(expected_ratio.data),
        rtol=2.0e-14,
    )


@pytest.mark.parametrize(
    "source",
    (_SOURCE_PARAMETERS[0], _SOURCE_PARAMETERS[2]),
)
def test_antisymmetric_phase_matches_lalsimulation(
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source}
    model = _calibrated_model(params)
    _, expected_phase = _lalsimulation_antisymmetric_waveform(params)
    frequencies = torch.arange(
        params["f_lower"],
        params["f_final"] + params["delta_f"],
        params["delta_f"],
        dtype=torch.float64,
    )
    actual = _antisymmetric_phase(
        model,
        frequencies * model.inputs.total_mass_seconds,
        -model.inputs.alpha0,
    )

    # The standalone LAL diagnostic includes zeta in the mode phase. The
    # native helper leaves that rotation to the common polarization assembly.
    expected = (
        np.asarray(expected_phase.data) - model.inputs.polarization_rotation
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=2.0e-11)


@pytest.mark.parametrize(
    "source",
    (_SOURCE_PARAMETERS[0], _SOURCE_PARAMETERS[2]),
)
def test_default_modes_regular_grid_match_lalsimulation(
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source}
    expected = _lalsimulation_default(params)
    actual = imrphenomxo4a_fd_torch(**params)

    for result, reference in zip(actual, expected):
        reference_array = np.asarray(reference.data.data)
        assert len(result) == len(reference_array)
        # LAL's regular-grid endpoint treatment differs from the native
        # carrier, as for explicit higher-mode requests below.
        assert (
            _relative_error(result.numpy()[:-1], reference_array[:-1])
            < 4.0e-4
        )


@pytest.mark.parametrize(
    ("modes", "tolerance"),
    (
        pytest.param(((2, 1),), 1.5e-3, id="mode21"),
        pytest.param(((3, 3),), 1.5e-3, id="mode33"),
        pytest.param(((3, 2),), 3.5e-3, id="mode32"),
        pytest.param(((4, 4),), 1.5e-3, id="mode44"),
        pytest.param(
            ((2, 2), (2, 1)),
            4.0e-4,
            id="modes22-and-21",
        ),
        pytest.param(
            ((2, 2), (2, 1), (3, 3)),
            1.5e-3,
            id="modes22-21-and-33",
        ),
        pytest.param(
            ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)),
            4.0e-4,
            id="all-native-modes",
        ),
    ),
)
@pytest.mark.parametrize("source", _SOURCE_PARAMETERS[:3])
def test_higher_modes_regular_grid_match_lalsimulation(
    modes,
    tolerance,
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source, "mode_array": modes}
    expected = _lalsimulation_regular(params, _mode_lal_params(modes))
    actual = imrphenomxo4a_fd_torch(**params)

    for result, reference in zip(actual, expected):
        reference_array = np.asarray(reference.data.data)
        assert len(result) == len(reference_array)
        # LAL leaves the exactly requested final regular-grid bin empty; the
        # native carrier evaluates it, as in the dominant-mode tests above.
        np.testing.assert_array_equal(
            result.numpy()[:-1] == 0.0,
            reference_array[:-1] == 0.0,
        )
        assert (
            _relative_error(result.numpy()[:-1], reference_array[:-1])
            < tolerance
        )


@pytest.mark.parametrize("source", _SOURCE_PARAMETERS)
def test_symmetric22_regular_grid_matches_lalsimulation(
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source}
    expected = _lalsimulation_symmetric22(params)
    actual = imrphenomxo4a_symmetric22_fd_torch(**params)

    for result, reference in zip(actual, expected):
        reference_array = np.asarray(reference.data.data)
        assert len(result) == len(reference_array)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        first_bin = int(params["f_lower"] / params["delta_f"])
        assert np.all(result.numpy()[:first_bin] == 0.0)
        # PNR gamma is integrated directly on the requested frequency grid.
        # LAL leaves an exactly requested final bin empty; the native XAS
        # carrier, like native XP, evaluates that bin.
        assert _relative_error(result.numpy(), reference_array) < 4.0e-4


def test_symmetric22_rejects_invalid_grid(torch_cpu_scheme):
    params = {
        **_BASE_PARAMS,
        "mass1": 40.0,
        "mass2": 20.0,
        "delta_f": 0.0,
    }
    with pytest.raises(ValueError, match="delta_f and f_lower must be positive"):
        imrphenomxo4a_symmetric22_fd_torch(**params)


@pytest.mark.parametrize(
    "source",
    (_SOURCE_PARAMETERS[0], _SOURCE_PARAMETERS[-1]),
)
def test_symmetric22_sequence_matches_lalsimulation(
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source}
    sample_points = np.array(
        [20.0, 23.5, 30.0, 45.0, 100.0, 250.0, 400.0, 1200.0]
    )
    expected = _lalsimulation_symmetric22_sequence(params, sample_points)
    actual = imrphenomxo4a_symmetric22_fd_sequence_torch(
        sample_points=sample_points,
        **params,
    )

    for result, reference in zip(actual, expected):
        result_array = result.numpy()
        reference_array = np.asarray(reference.data.data)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(
            result_array == 0.0,
            reference_array == 0.0,
        )
        nonzero = np.abs(reference_array) > 0.0
        assert nonzero.any()
        assert (
            _relative_error(
                result_array[nonzero],
                reference_array[nonzero],
            )
            < 1.0e-9
        )


@pytest.mark.parametrize(
    "source",
    (_SOURCE_PARAMETERS[0], _SOURCE_PARAMETERS[2]),
)
def test_default_modes_sequence_match_lalsimulation(
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source}
    sample_points = np.array(
        [20.0, 23.5, 30.0, 45.0, 100.0, 250.0, 400.0, 1200.0]
    )
    expected = _lalsimulation_default_sequence(params, sample_points)
    actual = imrphenomxo4a_fd_sequence_torch(
        sample_points=sample_points,
        **params,
    )

    for result, reference in zip(actual, expected):
        result_array = result.numpy()
        reference_array = np.asarray(reference.data.data)
        np.testing.assert_array_equal(
            result_array == 0.0,
            reference_array == 0.0,
        )
        nonzero = np.abs(reference_array) > 0.0
        assert nonzero.any()
        assert (
            _relative_error(
                result_array[nonzero],
                reference_array[nonzero],
            )
            < 5.0e-5
        )


@pytest.mark.parametrize(
    ("modes", "tolerance"),
    (
        pytest.param(((2, 1),), 1.0e-8, id="mode21"),
        pytest.param(((3, 3),), 1.0e-8, id="mode33"),
        pytest.param(((3, 2),), 2.5e-3, id="mode32"),
        pytest.param(((4, 4),), 1.0e-8, id="mode44"),
        pytest.param(
            ((2, 2), (2, 1)),
            1.0e-8,
            id="modes22-and-21",
        ),
        pytest.param(
            ((2, 2), (2, 1), (3, 3)),
            1.0e-8,
            id="modes22-21-and-33",
        ),
        pytest.param(
            ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)),
            5.0e-5,
            id="all-native-modes",
        ),
    ),
)
@pytest.mark.parametrize("source", _SOURCE_PARAMETERS[:3])
def test_higher_modes_sequence_matches_lalsimulation(
    modes,
    tolerance,
    source,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **source, "mode_array": modes}
    sample_points = np.array(
        [20.0, 23.5, 30.0, 45.0, 100.0, 250.0, 400.0, 1200.0]
    )
    expected = _lalsimulation_sequence(
        params,
        sample_points,
        _mode_lal_params(modes),
    )
    actual = imrphenomxo4a_fd_sequence_torch(
        sample_points=sample_points,
        **params,
    )

    for result, reference in zip(actual, expected):
        result_array = result.numpy()
        reference_array = np.asarray(reference.data.data)
        np.testing.assert_array_equal(
            result_array == 0.0,
            reference_array == 0.0,
        )
        nonzero = np.abs(reference_array) > 0.0
        assert nonzero.any()
        assert (
            _relative_error(
                result_array[nonzero],
                reference_array[nonzero],
            )
            < tolerance
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"mode_array": [(2, 2)]}, True),
        ({"mode_array": [(2, 2), (2, 2)]}, True),
        ({"mode_array": [(2, 1)]}, True),
        ({"mode_array": [(2, 1), (2, 1)]}, True),
        ({"mode_array": [(2, 1), (2, 2)]}, True),
        ({"mode_array": [(3, 3)]}, True),
        ({"mode_array": [(2, 2), (3, 3)]}, True),
        ({"mode_array": [(2, 1), (3, 3)]}, True),
        ({"mode_array": [(3, 2)]}, True),
        ({"mode_array": [(3, 2), (3, 2)]}, True),
        ({"mode_array": [(2, 2), (3, 2)]}, True),
        ({"mode_array": [(4, 4)]}, True),
        ({"mode_array": [(4, 4), (4, 4)]}, True),
        ({"mode_array": [(2, 2), (4, 4)]}, True),
        (
            {
                "mode_array": [(2, 2)],
                "phenom_x_prec_version": 300,
                "phenom_xp_convention": 1,
                "phenom_xp_final_spin_mod": 0,
            },
            True,
        ),
        ({}, True),
        ({"mode_array": None}, True),
        ({"mode_array": []}, False),
        ({"mode_array": (2, 2)}, False),
        ({"mode_array": [(2, -2)]}, False),
        ({"mode_array": [(2, 2), (2, -2)]}, False),
        ({"mode_array": [(5, 5)]}, False),
        ({"mode_array": [(2.0, 2.0)]}, False),
        ({"mode_array": ["22"]}, False),
        ({"mode_array": [(2, 2, 1)]}, False),
        (
            {"mode_array": [(2, 2)], "phenom_x_prec_version": 223},
            False,
        ),
        ({"mode_array": [(2, 2)], "phenom_xp_convention": 0}, False),
        ({"mode_array": [(2, 2)], "phenom_xp_final_spin_mod": 4}, False),
        ({"mode_array": [(2, 2)], "lambda1": 100.0}, False),
        ({"mode_array": [(2, 2)], "dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)], "eccentricity": 0.1}, False),
        ({"mode_array": [(2, 2)], "phase_order": 2.5}, True),
        ({"mode_array": [(2, 2)], "amplitude_order": "3"}, True),
        ({"mode_array": [(2, 2)], "spin_order": 4.5}, True),
        ({"mode_array": [(2, 2)], "tidal_order": 0}, True),
        ({"mode_array": [(2, 2)], "eccentricity_order": 4}, True),
        ({"mode_array": [(2, 2)], "eccentricity_order": 4.0}, False),
        ({"mode_array": [(2, 2)], "frame_axis": 1}, False),
        ({"mode_array": [(2, 2)], "numrel_data": "waveform.h5"}, False),
    ),
)
def test_imrphenomxo4a_native_support_boundary(changes, expected):
    params = {"approximant": "IMRPhenomXO4a", **changes}
    assert imrphenomxo4a_native_supported(params) is expected
    assert imrphenomxo4a_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    ("source", "modes", "expected"),
    (
        (_CALIBRATED_SOURCE, [(2, 1)], True),
        (_CALIBRATED_SOURCE, [(2, 2), (2, 1)], True),
        (_CALIBRATED_SOURCE, [(3, 3)], True),
        (_CALIBRATED_SOURCE, [(2, 1), (3, 3)], True),
        (_CALIBRATED_SOURCE, [(3, 2)], True),
        (_CALIBRATED_SOURCE, [(4, 4)], True),
        (_CALIBRATED_SOURCE, None, True),
        (_ALIGNED_SOURCE, [(2, 1)], False),
        (_ALIGNED_SOURCE, [(3, 3)], False),
        (_ALIGNED_SOURCE, [(3, 2)], False),
        (_ALIGNED_SOURCE, [(4, 4)], False),
        (_ALIGNED_SOURCE, None, False),
        (_ALIGNED_SOURCE, [(2, 2)], True),
        (
            {
                "mass1": 80.0,
                "mass2": 1.0,
                "spin1x": 0.1,
                "spin1z": 0.5,
            },
            [(2, 1)],
            False,
        ),
        (
            {
                "mass1": 80.0,
                "mass2": 1.0,
                "spin1x": 0.1,
                "spin1z": 0.5,
            },
            [(3, 3)],
            False,
        ),
        (
            {
                "mass1": 80.0,
                "mass2": 1.0,
                "spin1x": 0.1,
                "spin1z": 0.5,
            },
            [(3, 2)],
            False,
        ),
        (
            {
                "mass1": 80.0,
                "mass2": 1.0,
                "spin1x": 0.1,
                "spin1z": 0.5,
            },
            [(4, 4)],
            False,
        ),
        (
            {
                "mass1": 1.0,
                "mass2": 80.0,
                "spin2x": 0.1,
                "spin2z": 0.95,
            },
            [(2, 1)],
            True,
        ),
    ),
)
def test_imrphenomxo4a_higher_mode_native_source_boundary(
    source,
    modes,
    expected,
):
    params = {
        "approximant": "IMRPhenomXO4a",
        **source,
        "mode_array": modes,
    }
    assert imrphenomxo4a_native_supported(params) is expected
    assert imrphenomxo4a_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    ("modes", "tolerance"),
    (
        ([(2, 2)], 4.0e-4),
        ([(2, 1)], 7.0e-4),
        ([(3, 3)], 1.5e-3),
        ([(3, 2)], 1.5e-3),
        ([(4, 4)], 1.5e-3),
        ([(2, 2), (2, 1)], 2.0e-4),
        ([(2, 2), (2, 1), (3, 3)], 1.5e-3),
        ([(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)], 1.5e-4),
    ),
)
def test_imrphenomxo4a_public_regular_dispatch(
    modes,
    tolerance,
    monkeypatch,
    torch_cpu_scheme,
):
    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "mode_array": modes,
    }
    if modes == [(2, 2)]:
        params.update(
            phase_order=2.5,
            amplitude_order="3",
            spin_order=4.5,
            tidal_order=0,
            eccentricity_order=4,
        )
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXO4a", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXO4a called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXO4a", **params)

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
            result_array[:-1] == 0.0,
            expected_array[:-1] == 0.0,
        )
        assert (
            _relative_error(result_array[:-1], expected_array[:-1])
            < tolerance
        )


@pytest.mark.parametrize(
    ("modes", "tolerance"),
    (
        ([(2, 2)], 1.0e-9),
        ([(2, 1)], 1.0e-8),
        ([(3, 3)], 1.0e-8),
        ([(3, 2)], 3.0e-5),
        ([(4, 4)], 1.0e-8),
        ([(2, 2), (2, 1)], 1.0e-8),
        ([(2, 2), (2, 1), (3, 3)], 1.0e-8),
        ([(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)], 1.0e-7),
    ),
)
def test_imrphenomxo4a_public_sequence_dispatch(
    modes,
    tolerance,
    monkeypatch,
    torch_cpu_scheme,
):
    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "mode_array": modes,
    }
    if modes == [(2, 2)]:
        params.update(
            phase_order=2.5,
            amplitude_order="3",
            spin_order=4.5,
            tidal_order=0,
            eccentricity_order=4,
        )
    sample_points = [20.0, 23.5, 30.0, 45.0, 100.0, 400.0, 1200.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXO4a sequence called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxo4a_native_stays_on_requested_device(
    device_name,
    monkeypatch,
    torch_cpu_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "delta_f": 2.0,
        "f_final": 256.0,
    }
    sample_points = [20.0, 30.0, 50.0, 80.0, 128.0, 256.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference_grid = get_fd_waveform(
        approximant="IMRPhenomXO4a",
        **params,
    )
    reference_sequence = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )
    reference_grid = tuple(series.numpy().copy() for series in reference_grid)
    reference_sequence = tuple(
        series.numpy().copy() for series in reference_sequence
    )

    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual_grid = get_fd_waveform(
        approximant="IMRPhenomXO4a",
        **params,
    )
    actual_sequence = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    grid_tolerance = 5.0e-2 if device_name == "mps" else 2.0e-3
    sequence_tolerance = 2.5e-2 if device_name == "mps" else 1.0e-6
    for reference, actual, tolerance, comparison in (
        (reference_grid, actual_grid, grid_tolerance, slice(None, -1)),
        (reference_sequence, actual_sequence, sequence_tolerance, slice(None)),
    ):
        for expected, result in zip(reference, actual):
            tensor = result._data.tensor
            assert tensor.device.type == device_name
            assert tensor.dtype == expected_dtype
            assert bool(torch.isfinite(tensor).all())
            result_array = result.numpy()[comparison]
            expected = expected[comparison]
            np.testing.assert_array_equal(
                result_array == 0.0,
                expected == 0.0,
            )
            assert _relative_error(result_array, expected) < tolerance


def test_imrphenomxo4a_default_modes_regular_dispatch_natively(
    monkeypatch,
    torch_cpu_scheme,
):
    import pycbc.waveform.waveform as waveform

    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXO4a", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native default XO4a called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXO4a", **params)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert isinstance(result._data.tensor, torch.Tensor)
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array[:-1] == 0.0,
            expected_array[:-1] == 0.0,
        )
        assert (
            _relative_error(result_array[:-1], expected_array[:-1])
            < 1.5e-4
        )


def test_imrphenomxo4a_default_modes_sequence_dispatch_natively(
    monkeypatch,
    torch_cpu_scheme,
):
    import pycbc.waveform.waveform as waveform

    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    sample_points = [20.0, 30.0, 80.0, 200.0, 600.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXO4A_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native default XO4a called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXO4a",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert isinstance(result._data.tensor, torch.Tensor)
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 1.0e-7


@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
@pytest.mark.parametrize(
    ("disabled_flag", "global_enabled"),
    (
        ("PYCBC_TORCH_NATIVE_PORTS", False),
        ("PYCBC_IMRPHENOMXO4A_NATIVE", True),
    ),
)
def test_imrphenomxo4a_default_native_opt_out(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
    torch_cpu_scheme,
):
    import pycbc.waveform.waveform as waveform

    _clear_native_flags(monkeypatch)
    if global_enabled:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv(disabled_flag, "0")
    original = getattr(waveform.lalsimulation, lal_name)
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(waveform.lalsimulation, lal_name, record_lal)
    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "delta_f": 2.0,
        "f_final": 128.0,
    }
    if interface == "regular":
        result = get_fd_waveform(approximant="IMRPhenomXO4a", **params)
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomXO4a",
            sample_points=[20.0, 30.0, 80.0, 128.0],
            **params,
        )

    assert calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
def test_imrphenomxo4a_unsupported_default_modes_fall_back(
    interface,
    lal_name,
    monkeypatch,
    torch_cpu_scheme,
):
    import pycbc.waveform.waveform as waveform

    _clear_native_flags(monkeypatch)
    original = getattr(waveform.lalsimulation, lal_name)
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(waveform.lalsimulation, lal_name, record_lal)
    params = {
        **_BASE_PARAMS,
        **_ALIGNED_SOURCE,
        "delta_f": 2.0,
        "f_final": 128.0,
    }
    if interface == "regular":
        result = get_fd_waveform(approximant="IMRPhenomXO4a", **params)
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomXO4a",
            sample_points=[20.0, 30.0, 80.0, 128.0],
            **params,
        )

    assert calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


def test_symmetric22_sequence_uses_first_and_last_as_bounds(
    torch_cpu_scheme,
):
    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "f_ref": 23.5,
    }
    sample_points = np.array([20.0, 30.0, 23.0, 25.0])
    expected = _lalsimulation_symmetric22_sequence(params, sample_points)
    actual = imrphenomxo4a_symmetric22_fd_sequence_torch(
        sample_points=sample_points,
        **params,
    )

    for result, reference in zip(actual, expected):
        result_array = result.numpy()
        reference_array = np.asarray(reference.data.data)
        np.testing.assert_array_equal(
            result_array == 0.0,
            reference_array == 0.0,
        )
        assert _relative_error(result_array, reference_array) < 1.0e-9


@pytest.mark.parametrize(
    ("sample_points", "message"),
    (
        ([20.0], "at least two"),
        ([20.0, 20.0], "last.*must exceed"),
        ([20.0, 15.0, 30.0], "must not lie below"),
        ([20.0, np.nan, 30.0], "must be finite"),
        ([0.0, 30.0], "must be positive"),
    ),
)
def test_symmetric22_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    torch_cpu_scheme,
):
    params = {**_BASE_PARAMS, **_CALIBRATED_SOURCE}
    with pytest.raises(ValueError, match=message):
        imrphenomxo4a_symmetric22_fd_sequence_torch(
            sample_points=sample_points,
            **params,
        )


def test_symmetric22_sequence_rejects_reference_outside_bounds(
    torch_cpu_scheme,
):
    params = {
        **_BASE_PARAMS,
        **_CALIBRATED_SOURCE,
        "f_ref": 31.0,
    }
    with pytest.raises(ValueError, match="f_ref must lie between"):
        imrphenomxo4a_symmetric22_fd_sequence_torch(
            sample_points=[20.0, 25.0, 30.0],
            **params,
        )


def test_symmetric22_sequence_rejects_start_above_cutoff(
    torch_cpu_scheme,
):
    params = {
        **_BASE_PARAMS,
        "mass1": 2000.0,
        "mass2": 2000.0,
        "f_ref": 0.0,
    }
    with pytest.raises(ValueError, match="cutoff must exceed"):
        imrphenomxo4a_symmetric22_fd_sequence_torch(
            sample_points=[20.0, 30.0],
            **params,
        )
