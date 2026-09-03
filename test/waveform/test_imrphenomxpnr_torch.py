from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform.imrphenomx_utils_torch import (  # noqa: E402
    final_spin_2017,
    get_remnant_fMs,
    precessing_final_spin_2017,
    qnm_fdamp_21,
)
from pycbc.waveform.imrphenomx_spintaylor_torch import (  # noqa: E402
    SpinTaylorJFrame,
    SpinTaylorTrajectory,
    build_spintaylor_angle_spline,
    spintaylor_alpha_imr,
    spintaylor_beta_imr,
    spintaylor_inspiral_cosbeta,
)
from pycbc.waveform.imrphenomxp_msa_torch import (  # noqa: E402
    build_msa_state,
    msa_angles,
)
from pycbc.waveform.imrphenomxpnr_torch import (  # noqa: E402
    PNRSingleSpin,
    _arctan_window,
    build_pnr_alpha_parameters,
    build_pnr_spintaylor_alpha_parameters,
    build_pnr_beta_merger_parameters,
    build_pnr_beta_parameters,
    build_pnr_spintaylor_beta_parameters,
    build_pnr_spintaylor_integration,
    build_pnr_spintaylor_msa_state,
    build_pnr_spintaylor_remnant,
    build_pnr_coprecessing_deviations,
    build_pnr_single_spin_msa_state,
    generate_pnr_spintaylor_angles,
    pnr_alpha,
    pnr_angles_window,
    pnr_beta,
    pnr_coprecessing_fits,
    pnr_coprecessing_window,
    pnr_final_spin_model7,
    pnr_gamma,
    pnr_higher_mode_frequency_map,
    pnr_higher_mode_transition_frequencies,
    pnr_mr_beta,
    pnr_pn_waveform_beta,
    pnr_ringdown_beta,
    pnr_single_spin_mapping,
    pnr_spintaylor_alpha,
    pnr_spintaylor_beta,
    pnr_spintaylor_beta_imr,
    pnr_spintaylor_evolved_spins,
    pnr_spintaylor_final_spin_model4,
    pnr_spintaylor_interpolation_delta_f,
    pnr_spintaylor_single_spin_mapping,
)


_SOURCE_PARAMETERS = (
    (40.0, 20.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2),
    # Reversed component order exercises the body-labelled spin swap.
    (12.0, 35.0, 0.15, -0.25, 0.4, 0.05, 0.2, -0.3),
    (30.0, 30.0, 0.3, 0.0, 0.2, -0.2, 0.1, -0.1),
    (60.0, 5.0, 0.6, 0.2, -0.1, 0.2, -0.3, 0.4),
    (30.0, 20.0, 0.0, 0.0, 0.2, 0.0, 0.0, -0.1),
)

_FINAL_SPIN_PARAMETERS = _SOURCE_PARAMETERS + (
    # Negative cos(beta_RD) exercises model 7's signed-remnant branch.
    (40.0, 10.0, 0.05, 0.0, -0.99, 0.0, 0.0, -0.5),
)

_ALPHA_SOURCE_PARAMETERS = (
    pytest.param(
        (40.0, 20.0, 0.25, -0.1, 0.4, -0.15, 0.05, -0.2),
        id="calibrated",
    ),
    pytest.param(
        (50.0, 5.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2),
        id="transition-window",
    ),
    pytest.param(
        (60.0, 4.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2),
        id="outside-calibration",
    ),
)

_BETA_ANGLE_SOURCE_PARAMETERS = _ALPHA_SOURCE_PARAMETERS + (
    pytest.param(
        (40.0, 20.0, 0.25, -0.1, 0.4, 0.0, 0.0, 0.0),
        id="single-spin",
    ),
)

_BETA_REFERENCE_PARAMETERS = (
    (
        (40.0, 20.0, 0.25, -0.1, 0.4, -0.15, 0.05, -0.2),
        (
            0.0722111375547085,
            0.47319473808495266,
            -8.291192231347473,
            27.52185246921823,
            2070.986099921787,
            -0.08388635880129024,
            0.05034638176229571,
            0.09723245433888367,
        ),
        (0.10589261954730017, 0.09198342773023457, 0.057165596630495065),
    ),
    (
        (50.0, 5.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2),
        (
            0.23455303536475483,
            1.725079404339822,
            -48.037593973018346,
            340.69332818452943,
            1268.050972962383,
            -0.07799357334200792,
            0.03957868510344059,
            0.06957868510344059,
        ),
        (0.4056152950464175, 0.2640113148045688, 0.5230167140753904),
    ),
    (
        (60.0, 4.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2),
        (
            0.26742111091696574,
            4.607351187495475,
            -187.34255309538202,
            2016.739499573424,
            532.7789782492351,
            -0.08215966544568076,
            0.02698229272081727,
            0.044970487868028784,
        ),
        (0.8125121446262733, 1.5423816675217352, 6.606504176306552),
    ),
)

_COPRECESSING_FIT_NAMES = (
    "MU1",
    "MU2",
    "MU3",
    "NU0",
    "NU4",
    "NU5",
    "NU6",
    "ZETA1",
    "ZETA2",
)


def _lal_dict(*, tuned_coprecessing=False, final_spin_mod=None):
    params = lal.CreateDict()
    if tuned_coprecessing:
        lalsimulation.SimInspiralWaveformParamsInsertPhenomXPNRUseTunedCoprec(
            params,
            1,
        )
    if final_spin_mod is not None:
        lalsimulation.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
            params,
            final_spin_mod,
        )
    return params


def test_final_spin_2017_matches_lalsimulation():
    eta = torch.tensor([0.25, 2.0 / 9.0, 0.19, 0.071], dtype=torch.float64)
    chi1 = torch.tensor([0.1, 0.2, -0.6, 0.8], dtype=torch.float64)
    chi2 = torch.tensor([-0.2, 0.0, 0.3, -0.4], dtype=torch.float64)
    expected = np.array(
        [
            lalsimulation.SimIMRPhenomXFinalSpin2017(
                float(e),
                float(s1),
                float(s2),
            )
            for e, s1, s2 in zip(eta, chi1, chi2)
        ]
    )

    actual = final_spin_2017(eta, chi1, chi2)

    assert actual.device == eta.device
    assert actual.dtype == eta.dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-15, atol=2.0e-15)


def test_precessing_final_spin_2017_matches_lalsimulation():
    eta = torch.tensor([0.25, 2.0 / 9.0, 0.19, 0.071], dtype=torch.float64)
    chi1 = torch.tensor([0.1, 0.2, -0.6, 0.8], dtype=torch.float64)
    chi2 = torch.tensor([-0.2, 0.0, 0.3, -0.4], dtype=torch.float64)
    chi_inplane = torch.tensor([0.0, 0.2, 0.7, 0.4], dtype=torch.float64)
    expected = np.array(
        [
            lalsimulation.SimIMRPhenomXPrecessingFinalSpin2017(
                float(e),
                float(s1),
                float(s2),
                float(sp),
            )
            for e, s1, s2, sp in zip(eta, chi1, chi2, chi_inplane)
        ]
    )

    actual = precessing_final_spin_2017(eta, chi1, chi2, chi_inplane)

    assert actual.device == eta.device
    assert actual.dtype == eta.dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-15, atol=2.0e-15)


def _evolved_spin_trajectory(*, reversed_order=False):
    larger_mass = 40.0
    smaller_mass = 20.0
    larger_spin = torch.tensor((0.35, -0.2, 0.4), dtype=torch.float64)
    smaller_spin = torch.tensor((-0.15, 0.1, -0.25), dtype=torch.float64)
    if reversed_order:
        mass1, mass2 = smaller_mass, larger_mass
        spin1, spin2 = smaller_spin, larger_spin
    else:
        mass1, mass2 = larger_mass, smaller_mass
        spin1, spin2 = larger_spin, smaller_spin
    fraction1 = mass1 / (mass1 + mass2)
    fraction2 = mass2 / (mass1 + mass2)
    lnhat = torch.tensor((0.3, -0.4, np.sqrt(0.75)), dtype=torch.float64)
    state = torch.zeros((4, 14), dtype=torch.float64)
    state[:, 1] = torch.pi * torch.linspace(0.01, 0.04, 4, dtype=torch.float64)
    state[:, 2:5] = lnhat
    state[:, 5:8] = fraction1**2 * spin1
    state[:, 8:11] = fraction2**2 * spin2
    trajectory = SpinTaylorTrajectory(
        mf=torch.linspace(0.01, 0.04, 4, dtype=torch.float64),
        state=state,
    )
    return trajectory, mass1, mass2, spin1, spin2, lnhat


@pytest.mark.parametrize("reversed_order", (False, True))
def test_pnr_spintaylor_final_spin_model4_matches_lalsimulation(reversed_order):
    trajectory, mass1, mass2, spin1, spin2, lnhat = _evolved_spin_trajectory(
        reversed_order=reversed_order
    )
    cosbeta_max = torch.tensor(-0.35, dtype=torch.float64)
    actual = pnr_spintaylor_final_spin_model4(
        trajectory,
        mass1,
        mass2,
        cosbeta_max,
    )

    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1
    fraction1 = mass1 / (mass1 + mass2)
    fraction2 = mass2 / (mass1 + mass2)
    spin1_l = torch.dot(spin1, lnhat).item()
    spin2_l = torch.dot(spin2, lnhat).item()
    total_perp = fraction1**2 * (spin1 - lnhat * spin1_l)
    total_perp += fraction2**2 * (spin2 - lnhat * spin2_l)
    chi_perp = torch.linalg.vector_norm(total_perp).item() / fraction1**2
    expected = np.copysign(1.0, cosbeta_max.item()) * (
        lalsimulation.SimIMRPhenomXPrecessingFinalSpin2017(
            fraction1 * fraction2,
            spin1_l,
            spin2_l,
            chi_perp,
        )
    )

    assert actual.dtype == trajectory.state.dtype
    assert actual.device == trajectory.state.device
    assert actual.item() == pytest.approx(expected, rel=2.0e-15, abs=2.0e-15)


def test_build_pnr_spintaylor_remnant_uses_evolved_spin_and_qnm():
    trajectory, mass1, mass2, _, _, _ = _evolved_spin_trajectory()
    chi1_l = torch.tensor(0.3, dtype=torch.float64)
    chi2_l = torch.tensor(-0.2, dtype=torch.float64)
    cosbeta_max = torch.tensor(0.45, dtype=torch.float64)
    actual = build_pnr_spintaylor_remnant(
        trajectory,
        mass1,
        mass2,
        chi1_l,
        chi2_l,
        cosbeta_max,
    )
    expected_spin = pnr_spintaylor_final_spin_model4(
        trajectory,
        mass1,
        mass2,
        cosbeta_max,
    )
    expected = get_remnant_fMs(
        torch.tensor(mass1, dtype=torch.float64),
        torch.tensor(mass2, dtype=torch.float64),
        chi1_l,
        chi2_l,
        final_spin=expected_spin,
    )
    expected_final_mass = 1.0 - expected.radiated_energy

    torch.testing.assert_close(actual.final_spin, expected_spin)
    torch.testing.assert_close(actual.radiated_energy, expected.radiated_energy)
    torch.testing.assert_close(actual.final_mass, expected_final_mass)
    torch.testing.assert_close(actual.ringdown_frequency, expected.ringdown_frequency)
    torch.testing.assert_close(actual.damping_frequency, expected.damping_frequency)
    torch.testing.assert_close(
        actual.damping_difference,
        qnm_fdamp_21(expected_spin) / expected_final_mass
        - expected.damping_frequency,
    )


@pytest.mark.parametrize("reversed_order", (False, True))
def test_pnr_spintaylor_evolved_spins_use_final_orbital_frame(reversed_order):
    trajectory, mass1, mass2, spin1, spin2, lnhat = _evolved_spin_trajectory(
        reversed_order=reversed_order
    )
    actual1, actual2 = pnr_spintaylor_evolved_spins(
        trajectory,
        mass1,
        mass2,
    )
    azimuth = torch.atan2(lnhat[1], lnhat[0])
    polar = torch.acos(lnhat[2] / torch.linalg.vector_norm(lnhat))

    def expected(spin):
        cosine = torch.cos(azimuth)
        sine = torch.sin(azimuth)
        rotated = torch.stack(
            (
                spin[0] * cosine + spin[1] * sine,
                -spin[0] * sine + spin[1] * cosine,
                spin[2],
            )
        )
        cosine = torch.cos(polar)
        sine = torch.sin(polar)
        return torch.stack(
            (
                rotated[0] * cosine - rotated[2] * sine,
                rotated[1],
                rotated[0] * sine + rotated[2] * cosine,
            )
        )

    torch.testing.assert_close(actual1, expected(spin1))
    torch.testing.assert_close(actual2, expected(spin2))


def test_pnr_spintaylor_single_spin_mapping_uses_evolved_symmetric_spin():
    trajectory, mass1, mass2, _, _, _ = _evolved_spin_trajectory()
    initial_spin1 = (0.25, -0.1, 0.4)
    initial_spin2 = (-0.15, 0.05, -0.2)
    actual = pnr_spintaylor_single_spin_mapping(
        trajectory,
        mass1,
        mass2,
        *initial_spin1,
        *initial_spin2,
    )
    evolved_spin1, evolved_spin2 = pnr_spintaylor_evolved_spins(
        trajectory,
        mass1,
        mass2,
    )
    inverse_ratio = mass2 / mass1
    parallel = evolved_spin1[2] + inverse_ratio * evolved_spin2[2]
    symmetric_perpendicular = torch.hypot(
        evolved_spin1[0] + inverse_ratio**2 * evolved_spin2[0],
        evolved_spin1[1] + inverse_ratio**2 * evolved_spin2[1],
    )
    weight1 = 2.0 + 1.5 * inverse_ratio
    weight2 = 2.0 + 1.5 / inverse_ratio
    initial_chi_p = max(
        np.hypot(*initial_spin1[:2]),
        weight2
        * inverse_ratio**2
        * np.hypot(*initial_spin2[:2])
        / weight1,
    )
    expected_magnitude = torch.hypot(parallel, symmetric_perpendicular)
    expected_antisymmetric = torch.hypot(
        parallel,
        parallel.new_tensor(initial_chi_p),
    )

    torch.testing.assert_close(actual.magnitude, expected_magnitude)
    torch.testing.assert_close(actual.cosine, parallel / expected_magnitude)
    torch.testing.assert_close(
        actual.antisymmetric_magnitude,
        expected_antisymmetric,
    )
    torch.testing.assert_close(
        actual.antisymmetric_angle,
        torch.acos(parallel / expected_antisymmetric),
    )
    assert not torch.isclose(
        symmetric_perpendicular,
        symmetric_perpendicular.new_tensor(initial_chi_p),
    )


def test_pnr_final_spin_model7_matches_lalsimulation():
    batched = tuple(
        torch.tensor(values, dtype=torch.float64)
        for values in zip(*_FINAL_SPIN_PARAMETERS)
    )
    mass1, mass2, *spins = batched
    spin1 = spins[:3]
    spin2 = spins[3:]
    swap = mass2 > mass1
    larger_mass = torch.where(swap, mass2, mass1)
    smaller_mass = torch.where(swap, mass1, mass2)
    larger_spin = tuple(
        torch.where(swap, second, first)
        for first, second in zip(spin1, spin2)
    )
    smaller_spin = tuple(
        torch.where(swap, first, second)
        for first, second in zip(spin1, spin2)
    )
    total_mass = larger_mass + smaller_mass
    larger_fraction = larger_mass / total_mass
    smaller_fraction = smaller_mass / total_mass
    eta = larger_fraction * smaller_fraction
    larger_fraction2 = larger_fraction**2
    smaller_fraction2 = smaller_fraction**2
    chi_tot_perp = torch.hypot(
        larger_fraction2 * larger_spin[0]
        + smaller_fraction2 * smaller_spin[0],
        larger_fraction2 * larger_spin[1]
        + smaller_fraction2 * smaller_spin[1],
    ) / larger_fraction2
    beta_ringdown = pnr_ringdown_beta(pnr_single_spin_mapping(*batched))

    actual = pnr_final_spin_model7(
        eta,
        larger_spin[2],
        smaller_spin[2],
        chi_tot_perp,
        beta_ringdown,
    )
    params = _lal_dict(tuned_coprecessing=True, final_spin_mod=7)
    expected = np.array(
        [
            lalsimulation.SimPhenomPNRafinal_prec(
                mass1 * lal.MSUN_SI,
                mass2 * lal.MSUN_SI,
                *spins,
                params,
            )
            for mass1, mass2, *spins in _FINAL_SPIN_PARAMETERS
        ]
    )

    assert actual.device == batched[0].device
    assert actual.dtype == batched[0].dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-14, atol=2.0e-15)


def test_pnr_single_spin_ringdown_beta_matches_lalsimulation():
    batched = tuple(
        torch.tensor(values, dtype=torch.float64) for values in zip(*_SOURCE_PARAMETERS)
    )
    mapped = pnr_single_spin_mapping(*batched)
    actual = pnr_ringdown_beta(mapped)
    expected = []
    for mass1, mass2, *spins in _SOURCE_PARAMETERS:
        expected.append(
            lalsimulation.SimPhenomPNRbetaRD(
                mass1 * lal.MSUN_SI,
                mass2 * lal.MSUN_SI,
                *spins,
                _lal_dict(tuned_coprecessing=True),
            )
        )

    assert actual.device == batched[0].device
    assert actual.dtype == batched[0].dtype
    np.testing.assert_allclose(
        actual.numpy(),
        expected,
        rtol=2.0e-6,
        atol=3.0e-8,
    )

    single_precision = tuple(value.to(torch.float32) for value in batched)
    actual_single = pnr_ringdown_beta(pnr_single_spin_mapping(*single_precision))
    assert actual_single.dtype == torch.float32
    np.testing.assert_allclose(
        actual_single.numpy(),
        expected,
        rtol=1.0e-4,
        atol=8.0e-7,
    )


def test_pnr_angle_window_boundaries():
    mass_ratio = torch.tensor([8.5, 10.25, 12.0], dtype=torch.float64)
    spin = torch.tensor([0.85, 1.025, 1.2], dtype=torch.float64)

    actual = pnr_angles_window(mass_ratio, spin)

    torch.testing.assert_close(
        actual,
        torch.tensor([1.0, 0.25, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-16,
    )


def test_pnr_higher_mode_transition_frequencies_match_lal_formula():
    pnr_low = torch.tensor([0.04, 0.04, 0.04], dtype=torch.float64)
    pnr_high = torch.tensor([0.09, 0.03, 0.08], dtype=torch.float64)
    ring_22 = torch.tensor([0.08, 0.10, 0.10], dtype=torch.float64)
    ring_lm = torch.tensor([0.10, 0.01, 0.04], dtype=torch.float64)

    lower, upper = pnr_higher_mode_transition_frequencies(
        1,
        pnr_low,
        pnr_high,
        ring_22,
        ring_lm,
    )

    torch.testing.assert_close(
        lower,
        torch.full_like(lower, 0.013),
        rtol=0.0,
        atol=2.0e-17,
    )
    # The second case exercises LAL's negative-upper-frequency guard; the
    # third exercises its positive but too-small (2, 1) guard.
    torch.testing.assert_close(
        upper,
        torch.tensor([0.121, 0.03, 0.08], dtype=torch.float64),
        rtol=0.0,
        atol=3.0e-17,
    )


def test_pnr_higher_mode_frequency_map_matches_piecewise_lal_formula():
    frequencies = torch.tensor(
        [0.01, 0.03, 0.06, 0.09, 0.12],
        dtype=torch.float64,
    )

    actual = pnr_higher_mode_frequency_map(
        frequencies,
        3,
        3,
        0.03,
        0.09,
        0.08,
        0.12,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [2.0 / 300.0, 0.02, 0.035, 0.05, 0.08],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=2.0e-17,
    )


def test_pnr_higher_mode_frequency_map_identity_and_inspiral_only():
    frequencies = torch.tensor([0.02, 0.08, 0.14], dtype=torch.float32)

    identity = pnr_higher_mode_frequency_map(
        frequencies,
        2,
        2,
        0.03,
        0.09,
        0.08,
        0.12,
    )
    inspiral = pnr_higher_mode_frequency_map(
        frequencies,
        4,
        4,
        0.03,
        0.09,
        0.08,
        0.12,
        inspiral_only=True,
    )

    assert identity.data_ptr() == frequencies.data_ptr()
    assert identity.dtype == frequencies.dtype
    torch.testing.assert_close(inspiral, 0.5 * frequencies)


@pytest.mark.parametrize("ell,mprime", [(0, 2), (2, 0), (2.0, 2), (2, True)])
def test_pnr_higher_mode_frequency_map_rejects_invalid_modes(ell, mprime):
    with pytest.raises(ValueError, match="positive integer"):
        pnr_higher_mode_frequency_map(
            torch.tensor([0.02, 0.04]),
            ell,
            mprime,
            0.03,
            0.09,
            0.08,
            0.12,
        )


def test_pnr_higher_mode_frequency_map_rejects_unordered_transitions():
    with pytest.raises(ValueError, match="strictly ordered"):
        pnr_higher_mode_frequency_map(
            torch.tensor([0.02, 0.04]),
            2,
            1,
            0.09,
            0.09,
            0.08,
            0.12,
        )


def test_pnr_beta_merger_parameters_match_lalsimulation_reference():
    sources = [source for source, _, _ in _BETA_REFERENCE_PARAMETERS]
    batched = tuple(
        torch.tensor(values, dtype=torch.float64) for values in zip(*sources)
    )
    actual = build_pnr_beta_merger_parameters(pnr_single_spin_mapping(*batched))
    actual_parameters = torch.stack(
        (
            actual.b0,
            actual.b1,
            actual.b2,
            actual.b3,
            actual.b4,
            actual.b5,
            actual.mf_lower,
            actual.mf_upper,
        ),
        dim=-1,
    )
    expected_parameters = np.asarray(
        [expected for _, expected, _ in _BETA_REFERENCE_PARAMETERS]
    )

    assert actual_parameters.device == batched[0].device
    assert actual_parameters.dtype == batched[0].dtype
    np.testing.assert_allclose(
        actual_parameters.numpy(),
        expected_parameters,
        rtol=5.0e-12,
        atol=5.0e-12,
    )

    frequencies = torch.tensor((0.02, 0.07, 0.15), dtype=torch.float64)
    for index, (_, _, expected_beta) in enumerate(_BETA_REFERENCE_PARAMETERS):
        scalar_parameters = build_pnr_beta_merger_parameters(
            pnr_single_spin_mapping(*sources[index])
        )
        actual_beta = pnr_mr_beta(frequencies, scalar_parameters)
        np.testing.assert_allclose(
            actual_beta.numpy(),
            expected_beta,
            rtol=5.0e-12,
            atol=5.0e-12,
        )


def test_pnr_spintaylor_fits_apply_version_330_bounds():
    source = (60.0, 4.0, 0.7, 0.3, 0.4, 0.0, 0.0, 0.0)
    single_spin = pnr_single_spin_mapping(*source)
    spin_boundary = 0.80 - 0.20 * torch.exp(
        -((single_spin.mass_ratio - 6.0) / 1.5) ** 8
    )
    bounded_single_spin = replace(
        single_spin,
        symmetric_mass_ratio=torch.clamp(
            single_spin.symmetric_mass_ratio,
            min=0.09,
        ),
        magnitude=torch.minimum(single_spin.magnitude, spin_boundary),
    )

    beta = build_pnr_beta_merger_parameters(single_spin, prec_version=330)
    bounded_beta = build_pnr_beta_merger_parameters(
        bounded_single_spin,
        prec_version=330,
    )
    for field in ("b0", "b1", "b2", "b3", "b4", "b5"):
        torch.testing.assert_close(
            getattr(beta, field),
            getattr(bounded_beta, field),
        )

    angles = _synthetic_spintaylor_angles()
    alpha = build_pnr_spintaylor_alpha_parameters(
        single_spin,
        angles,
        sum(source[:2]) * lal.MTSUN_SI,
    )
    bounded_alpha = build_pnr_spintaylor_alpha_parameters(
        bounded_single_spin,
        angles,
        sum(source[:2]) * lal.MTSUN_SI,
    )
    for field in ("a1", "a2", "a3", "a4"):
        torch.testing.assert_close(
            getattr(alpha, field),
            getattr(bounded_alpha, field),
        )

    version_223 = build_pnr_beta_merger_parameters(single_spin)
    assert not torch.isclose(beta.b0, version_223.b0)


@pytest.mark.parametrize("source", _ALPHA_SOURCE_PARAMETERS)
def test_pnr_alpha_matches_lalsimulation(source):
    mass1, mass2, *spins = source
    spin1 = tuple(spins[:3])
    spin2 = tuple(spins[3:])
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    f_ref = 30.0
    msa_state = build_msa_state(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )
    single_spin = pnr_single_spin_mapping(*source)
    alpha_parameters = build_pnr_alpha_parameters(
        single_spin,
        msa_state,
        total_mass_seconds,
    )

    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        223,
    )
    lal_angles = lalsimulation.SimIMRPhenomX_PNR_GeneratePNRAngles(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        *spins,
        0.7,
        2.0,
        20.0,
        512.0,
        f_ref,
        lal_params,
    )
    frequencies = np.asarray(lal_angles[3].data)
    expected = np.asarray(lal_angles[0].data)
    geometric_frequencies = torch.tensor(
        frequencies * total_mass_seconds,
        dtype=torch.float64,
    )

    actual = pnr_alpha(
        geometric_frequencies,
        alpha_parameters,
        single_spin,
        msa_state,
    )

    assert actual.device == geometric_frequencies.device
    assert actual.dtype == geometric_frequencies.dtype
    np.testing.assert_allclose(
        actual.numpy(),
        expected,
        rtol=3.0e-12,
        atol=5.0e-12,
    )


def _synthetic_spintaylor_angles(mf_start=0.005):
    mf = torch.linspace(mf_start, 0.12, 48, dtype=torch.float64)
    alpha = 0.4 + 18.0 * mf + 7.0 * mf.square()
    beta = 0.35 + 0.07 * torch.sin(23.0 * mf)
    state = torch.zeros((mf.numel(), 14), dtype=mf.dtype)
    state[:, 2:5] = torch.stack(
        (
            torch.sin(beta) * torch.cos(alpha),
            torch.sin(beta) * torch.sin(alpha),
            torch.cos(beta),
        ),
        dim=-1,
    )
    zero = mf.new_zeros(())
    frame = SpinTaylorJFrame(
        phi_j_source=zero,
        theta_j_source=zero,
        kappa=zero,
        alpha0=mf.new_tensor(1.1),
        epsilon0=zero,
    )
    return build_spintaylor_angle_spline(
        SpinTaylorTrajectory(mf=mf, state=state),
        frame,
        mf[-1],
        damping_difference=0.013,
        ringdown_beta=0.4,
    )


def test_pnr_spintaylor_integration_matches_lal_buffer_logic():
    source = (40.0, 20.0, 0.25, -0.1, 0.4, -0.15, 0.05, -0.2)
    mass1, mass2, *spins = source
    msa_state = build_msa_state(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        (mass1 + mass2) * lal.MTSUN_SI,
        30.0,
    )
    single_spin = pnr_single_spin_mapping(*source)

    actual = build_pnr_spintaylor_integration(
        torch.tensor(20.0, dtype=torch.float64),
        2.0,
        0.09,
        single_spin,
        msa_state,
    )

    assert actual.interpolation_delta_f.item() == pytest.approx(
        1.7401687606207246,
        rel=2.0e-14,
    )
    assert actual.integration_buffer.item() == pytest.approx(
        2.4362362648690144,
        rel=2.0e-14,
    )
    assert actual.starting_frequency.item() == pytest.approx(
        15.66151884558652,
        rel=2.0e-14,
    )
    assert actual.trajectory_minimum_frequency.item() == pytest.approx(
        15.16151884558652,
        rel=2.0e-14,
    )
    assert actual.starting_frequency.dtype == torch.float64


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            (40.0, 20.0, 0.25, -0.1, 0.4, -0.15, 0.05, -0.2),
            id="ordered",
        ),
        pytest.param(
            (12.0, 35.0, 0.15, -0.25, 0.4, 0.05, 0.2, -0.3),
            id="reversed",
        ),
    ),
)
def test_generate_pnr_spintaylor_angles_matches_lalsimulation(source):
    mass1, mass2, *spins = source
    inclination = 0.7
    delta_f = 2.0
    f_min = 20.0
    f_max = 512.0
    f_ref = 30.0
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        330,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPNRUseTunedAngles(
        lal_params,
        1,
    )
    expected = lalsimulation.SimIMRPhenomX_PNR_GeneratePNRAngles(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        *spins,
        inclination,
        delta_f,
        f_min,
        f_max,
        f_ref,
        lal_params,
    )

    actual = generate_pnr_spintaylor_angles(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        inclination,
        delta_f,
        f_min,
        f_max,
        f_ref,
        dtype=torch.float64,
    )

    assert actual.frequencies.dtype == torch.float64
    assert actual.alpha.device == actual.frequencies.device
    np.testing.assert_allclose(actual.frequencies.numpy(), expected[3].data)
    for values, reference in zip(
        (actual.alpha, actual.beta, actual.gamma),
        expected[:3],
    ):
        np.testing.assert_allclose(
            values.numpy(),
            reference.data,
            rtol=3.0e-8,
            atol=3.0e-8,
        )
    assert actual.alpha_reference.item() == pytest.approx(
        expected[4],
        rel=3.0e-8,
        abs=3.0e-8,
    )
    assert actual.gamma_reference.item() == pytest.approx(
        expected[5],
        rel=3.0e-8,
        abs=3.0e-8,
    )


def test_pnr_spintaylor_interpolation_uses_output_spacing_when_aligned():
    msa_state = build_msa_state(
        40.0,
        20.0,
        (0.0, 0.0, 0.4),
        (0.0, 0.0, -0.2),
        60.0 * lal.MTSUN_SI,
        30.0,
    )

    actual = pnr_spintaylor_interpolation_delta_f(
        torch.tensor(20.0, dtype=torch.float64),
        msa_state,
        output_delta_f=2.0,
    )
    fallback = pnr_spintaylor_interpolation_delta_f(
        torch.tensor(20.0, dtype=torch.float64),
        msa_state,
    )

    torch.testing.assert_close(actual, actual.new_tensor(2.0))
    torch.testing.assert_close(fallback, fallback.new_tensor(0.1))


def test_pnr_spintaylor_msa_state_keeps_msa_dynamics_and_replaces_l_fit():
    msa_state = build_msa_state(
        40.0,
        20.0,
        (0.25, -0.1, 0.4),
        (-0.15, 0.05, -0.2),
        60.0 * lal.MTSUN_SI,
        30.0,
    )

    actual = build_pnr_spintaylor_msa_state(msa_state)

    assert actual is not msa_state
    assert actual["constants_L"] == msa_state["constants_L"]
    assert actual["L3"] == pytest.approx(-0.6296296296296297)
    assert actual["L5"] == pytest.approx(-1.0725651577503428)
    assert actual["L3"] != msa_state["L3"]
    assert msa_state["L3"] == pytest.approx(-0.42962962962962964)


def _spintaylor_beta_case(source):
    mass1, mass2, *spins = source
    msa_state = build_msa_state(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        (mass1 + mass2) * lal.MTSUN_SI,
        30.0,
    )
    single_spin = pnr_single_spin_mapping(*source)
    single_spin_msa_state = build_pnr_single_spin_msa_state(
        single_spin,
        msa_state,
    )
    angles = _synthetic_spintaylor_angles()
    parameters = build_pnr_spintaylor_beta_parameters(
        single_spin,
        angles,
        msa_state,
        single_spin_msa_state,
    )
    return single_spin, angles, msa_state, single_spin_msa_state, parameters


def _expected_spintaylor_tapered_beta(
    frequencies,
    parameters,
    angles,
    single_spin_msa_state,
    *,
    use_mr_beta,
):
    full_beta = pnr_spintaylor_beta_imr(
        frequencies,
        parameters,
        angles,
        use_mr_beta=use_mr_beta,
    )
    if single_spin_msa_state is None:
        return full_beta

    merger = parameters.merger
    velocity = torch.pow(np.pi * frequencies, 1.0 / 3.0)
    single_spin_beta = torch.acos(
        msa_angles(velocity, single_spin_msa_state)[2]
    )
    envelope = torch.cos(np.pi * frequencies / (2.0 * merger.mf_lower)) ** 2
    tapered = single_spin_beta + (full_beta - single_spin_beta) * envelope
    return torch.where(frequencies <= merger.mf_lower, tapered, single_spin_beta)


def test_pnr_spintaylor_alpha_uses_numerical_imr_provider():
    source = _ALPHA_SOURCE_PARAMETERS[0].values[0]
    single_spin = pnr_single_spin_mapping(*source)
    angles = _synthetic_spintaylor_angles()
    total_mass_seconds = sum(source[:2]) * lal.MTSUN_SI
    alpha_offset = angles.mf.new_tensor(-0.23)
    parameters = build_pnr_spintaylor_alpha_parameters(
        single_spin,
        angles,
        total_mass_seconds,
        alpha_offset=alpha_offset,
    )
    step = angles.mf.new_tensor(0.0005)
    lower_samples = parameters.mf_lower + step * torch.tensor(
        (-1.0, 0.0, 1.0),
        dtype=angles.mf.dtype,
    )
    pn_samples = spintaylor_alpha_imr(
        lower_samples,
        angles,
        offset=alpha_offset,
    )
    expected_derivative = (pn_samples[2] - pn_samples[0]) / (2.0 * step)

    below = parameters.mf_lower - 2.0 * step
    torch.testing.assert_close(
        pnr_spintaylor_alpha(
            below,
            parameters,
            single_spin,
            angles,
            alpha_offset=alpha_offset,
        ),
        spintaylor_alpha_imr(below, angles, offset=alpha_offset),
        rtol=0.0,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        parameters.interp0 * 2.0 * parameters.mf_lower
        + parameters.interp1
        - parameters.interp3 / parameters.mf_lower.square(),
        expected_derivative,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        pnr_spintaylor_alpha(
            parameters.mf_upper,
            parameters,
            single_spin,
            angles,
            alpha_offset=alpha_offset,
        ),
        spintaylor_alpha_imr(
            parameters.mf_upper,
            angles,
            offset=alpha_offset,
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_pnr_spintaylor_alpha_disables_connection_below_integration_domain():
    source = _ALPHA_SOURCE_PARAMETERS[0].values[0]
    single_spin = pnr_single_spin_mapping(*source)
    angles = _synthetic_spintaylor_angles(mf_start=0.08)
    parameters = build_pnr_spintaylor_alpha_parameters(
        single_spin,
        angles,
        sum(source[:2]) * lal.MTSUN_SI,
    )

    torch.testing.assert_close(
        parameters.mf_lower,
        parameters.mf_lower.new_tensor(100.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        parameters.mf_upper,
        parameters.mf_upper.new_tensor(100.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("two_spin", [False, True])
def test_pnr_spintaylor_beta_builds_lal_connection(two_spin):
    source = list(_ALPHA_SOURCE_PARAMETERS[0].values[0])
    if not two_spin:
        source[5:] = (0.0, 0.0, 0.0)
    mass1, mass2, *spins = source
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    msa_state = build_msa_state(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        total_mass_seconds,
        30.0,
    )
    single_spin = pnr_single_spin_mapping(*source)
    single_spin_msa_state = build_pnr_single_spin_msa_state(single_spin, msa_state)
    angles = _synthetic_spintaylor_angles()
    parameters = build_pnr_spintaylor_beta_parameters(
        single_spin,
        angles,
        msa_state,
        single_spin_msa_state,
    )
    merger = parameters.merger
    step = angles.mf.new_tensor(0.0005)
    mf_a = parameters.mf_interpolation_start
    mf_b = merger.mf_lower
    beta_a_samples = torch.acos(
        torch.clamp(
            spintaylor_inspiral_cosbeta(
                mf_a + step * angles.mf.new_tensor((-1.0, 0.0, 1.0)),
                angles,
            ),
            -1.0,
            1.0,
        )
    )
    derivative_a = (beta_a_samples[2] - beta_a_samples[0]) / (2.0 * step)
    connection_a = pnr_spintaylor_beta_imr(
        mf_a,
        parameters,
        angles,
        use_mr_beta=True,
    )
    connection_b = pnr_spintaylor_beta_imr(
        mf_b,
        parameters,
        angles,
        use_mr_beta=True,
    )
    derivative_at_a = (
        parameters.interp1
        + 2.0 * parameters.interp2 * mf_a
        + 3.0 * parameters.interp3 * mf_a.square()
    )

    # Version 330 deliberately pins both cubic endpoints to beta(MfA).
    torch.testing.assert_close(
        connection_a,
        beta_a_samples[1],
        atol=2.0e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        connection_b,
        beta_a_samples[1],
        atol=2.0e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        derivative_at_a,
        derivative_a,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert torch.isfinite(merger.rescale1)
    assert torch.isfinite(merger.rescale2)


def test_pnr_spintaylor_beta_imr_selects_generic_continuation():
    source = _ALPHA_SOURCE_PARAMETERS[0].values[0]
    mass1, mass2, *spins = source
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    msa_state = build_msa_state(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        total_mass_seconds,
        30.0,
    )
    single_spin = pnr_single_spin_mapping(*source)
    angles = _synthetic_spintaylor_angles()
    parameters = build_pnr_spintaylor_beta_parameters(
        single_spin,
        angles,
        msa_state,
        build_pnr_single_spin_msa_state(single_spin, msa_state),
    )
    frequencies = torch.tensor((0.03, 0.13, 0.2), dtype=torch.float64)

    torch.testing.assert_close(
        pnr_spintaylor_beta_imr(
            frequencies,
            parameters,
            angles,
            use_mr_beta=False,
        ),
        spintaylor_beta_imr(frequencies, angles),
    )


def test_pnr_spintaylor_beta_uses_calibrated_merger_regions():
    source = (40.0, 20.0, 0.25, -0.1, 0.4, 0.0, 0.0, 0.0)
    single_spin, angles, msa_state, single_spin_msa_state, parameters = (
        _spintaylor_beta_case(source)
    )
    merger = parameters.merger
    frequencies = torch.stack(
        (
            merger.mf_lower - 0.001,
            (merger.mf_lower + merger.mf_upper) / 2.0,
            merger.mf_upper + 0.001,
        )
    )
    dynamics_beta = pnr_spintaylor_beta_imr(
        frequencies,
        parameters,
        angles,
        use_mr_beta=True,
    )
    waveform_beta = pnr_pn_waveform_beta(
        frequencies,
        dynamics_beta,
        msa_state,
    )
    expected = torch.stack(
        (
            waveform_beta[0]
            * (
                1.0
                + merger.rescale1 * frequencies[0]
                + merger.rescale2 * frequencies[0].square()
            ),
            pnr_mr_beta(frequencies[1], merger),
            pnr_mr_beta(merger.mf_upper, merger),
        )
    )

    torch.testing.assert_close(
        pnr_spintaylor_beta(
            frequencies,
            parameters,
            single_spin,
            angles,
            msa_state,
            single_spin_msa_state,
        ),
        _arctan_window(expected),
    )
    torch.testing.assert_close(
        pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude),
        frequencies.new_tensor(1.0),
    )

    disabled_parameters = replace(
        parameters,
        merger=replace(merger, beta_upper=merger.beta_upper.new_tensor(-1.0)),
    )
    generic_dynamics = pnr_spintaylor_beta_imr(
        frequencies,
        disabled_parameters,
        angles,
        use_mr_beta=False,
    )
    torch.testing.assert_close(
        pnr_spintaylor_beta(
            frequencies,
            disabled_parameters,
            single_spin,
            angles,
            msa_state,
            single_spin_msa_state,
        ),
        _arctan_window(
            pnr_pn_waveform_beta(frequencies, generic_dynamics, msa_state)
        ),
    )


def test_pnr_spintaylor_beta_blends_tapered_transition_angles():
    source = (50.0, 5.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2)
    single_spin, angles, msa_state, single_spin_msa_state, parameters = (
        _spintaylor_beta_case(source)
    )
    merger = parameters.merger
    frequencies = torch.stack(
        (
            merger.mf_lower - 0.001,
            (merger.mf_lower + merger.mf_upper) / 2.0,
            merger.mf_upper + 0.001,
        )
    )
    tuned_dynamics = _expected_spintaylor_tapered_beta(
        frequencies,
        parameters,
        angles,
        single_spin_msa_state,
        use_mr_beta=True,
    )
    tuned_waveform = pnr_pn_waveform_beta(
        frequencies,
        tuned_dynamics,
        msa_state,
    )
    tuned = torch.stack(
        (
            tuned_waveform[0]
            * (
                1.0
                + merger.rescale1 * frequencies[0]
                + merger.rescale2 * frequencies[0].square()
            ),
            pnr_mr_beta(frequencies[1], merger),
            pnr_mr_beta(merger.mf_upper, merger),
        )
    )
    generic_dynamics = _expected_spintaylor_tapered_beta(
        frequencies,
        parameters,
        angles,
        single_spin_msa_state,
        use_mr_beta=False,
    )
    generic_waveform = pnr_pn_waveform_beta(
        frequencies,
        generic_dynamics,
        msa_state,
    )
    window = pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude)
    assert 0.0 < window < 1.0
    expected = _arctan_window(
        window * tuned + (1.0 - window) * generic_waveform
    )

    torch.testing.assert_close(
        pnr_spintaylor_beta(
            frequencies,
            parameters,
            single_spin,
            angles,
            msa_state,
            single_spin_msa_state,
        ),
        expected,
    )


def test_pnr_spintaylor_beta_outside_calibration_keeps_full_two_spin():
    source = (60.0, 4.0, 0.2, 0.1, 0.3, -0.1, 0.05, -0.2)
    single_spin, angles, msa_state, single_spin_msa_state, parameters = (
        _spintaylor_beta_case(source)
    )
    merger = parameters.merger
    frequencies = torch.stack(
        (merger.mf_lower - 0.001, merger.mf_lower + 0.001)
    )
    full_dynamics = pnr_spintaylor_beta_imr(
        frequencies,
        parameters,
        angles,
        use_mr_beta=False,
    )
    expected = _arctan_window(
        pnr_pn_waveform_beta(frequencies, full_dynamics, msa_state)
    )

    torch.testing.assert_close(
        pnr_spintaylor_beta(
            frequencies,
            parameters,
            single_spin,
            angles,
            msa_state,
            single_spin_msa_state,
        ),
        expected,
    )
    torch.testing.assert_close(
        pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude),
        frequencies.new_tensor(0.0),
    )
    tapered_dynamics = _expected_spintaylor_tapered_beta(
        frequencies,
        parameters,
        angles,
        single_spin_msa_state,
        use_mr_beta=False,
    )
    assert not torch.allclose(full_dynamics, tapered_dynamics)


@pytest.mark.parametrize("source", _BETA_ANGLE_SOURCE_PARAMETERS)
def test_pnr_beta_and_gamma_match_lalsimulation(source):
    mass1, mass2, *spins = source
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    f_ref = 30.0
    msa_state = build_msa_state(
        mass1,
        mass2,
        tuple(spins[:3]),
        tuple(spins[3:]),
        total_mass_seconds,
        f_ref,
    )
    single_spin = pnr_single_spin_mapping(*source)
    single_spin_msa_state = build_pnr_single_spin_msa_state(single_spin, msa_state)
    alpha_parameters = build_pnr_alpha_parameters(
        single_spin,
        msa_state,
        total_mass_seconds,
    )
    beta_parameters = build_pnr_beta_parameters(
        single_spin,
        msa_state,
        single_spin_msa_state,
    )

    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        223,
    )
    lal_angles = lalsimulation.SimIMRPhenomX_PNR_GeneratePNRAngles(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        *spins,
        0.7,
        2.0,
        20.0,
        512.0,
        f_ref,
        lal_params,
    )
    frequencies = np.asarray(lal_angles[3].data)
    expected_beta = np.asarray(lal_angles[1].data)
    expected_gamma = np.asarray(lal_angles[2].data)
    geometric_frequencies = torch.tensor(
        frequencies * total_mass_seconds,
        dtype=torch.float64,
    )

    actual_alpha = pnr_alpha(
        geometric_frequencies,
        alpha_parameters,
        single_spin,
        msa_state,
    )
    actual_beta = pnr_beta(
        geometric_frequencies,
        beta_parameters,
        single_spin,
        msa_state,
        single_spin_msa_state,
    )
    actual_gamma = pnr_gamma(
        geometric_frequencies,
        actual_alpha,
        actual_beta,
    )

    assert actual_gamma.device == geometric_frequencies.device
    assert actual_gamma.dtype == geometric_frequencies.dtype
    np.testing.assert_allclose(
        actual_beta.numpy(),
        expected_beta,
        rtol=5.0e-11,
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        actual_gamma.numpy(),
        expected_gamma,
        rtol=3.0e-12,
        atol=5.0e-12,
    )


@pytest.mark.parametrize("mass_ratio", [1.0, 8.5, 10.0, 12.0, 15.0, 20.0, 21.0])
def test_pnr_coprecessing_window_matches_lalsimulation(mass_ratio):
    spins = (0.2, 0.1, 0.3, -0.1, 0.05, -0.2)
    expected = lalsimulation.SimPhenomPNRwindow(
        mass_ratio * 30.0 * lal.MSUN_SI,
        30.0 * lal.MSUN_SI,
        *spins,
        _lal_dict(tuned_coprecessing=True),
    )

    actual = pnr_coprecessing_window(mass_ratio)

    assert actual.dtype == torch.float64
    assert actual.item() == pytest.approx(expected, rel=2.0e-15, abs=2.0e-15)


def test_pnr_coprecessing_fits_match_lalsimulation():
    theta = torch.tensor([0.0, 0.37, 1.2, 2.6], dtype=torch.float64)
    eta = torch.tensor([0.25, 0.21, 0.09876, 0.04], dtype=torch.float64)
    spin = torch.tensor([0.2, 0.5, 0.8, 1.1], dtype=torch.float64)
    actual = pnr_coprecessing_fits(theta, eta, spin)
    actual_values = torch.stack(
        tuple(getattr(actual, name.lower()) for name in _COPRECESSING_FIT_NAMES),
        dim=-1,
    )
    expected = np.asarray(
        [
            [
                getattr(lalsimulation, f"SimIMRPhenomXCP_{name}_l2m2")(
                    float(angle),
                    float(symmetric_mass_ratio),
                    float(spin_magnitude),
                )
                for name in _COPRECESSING_FIT_NAMES
            ]
            for angle, symmetric_mass_ratio, spin_magnitude in zip(
                theta,
                eta,
                spin,
            )
        ]
    )

    assert actual_values.device == theta.device
    assert actual_values.dtype == theta.dtype
    np.testing.assert_allclose(
        actual_values.numpy(),
        expected,
        rtol=3.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("prec_version", [300, 330])
def test_build_pnr_coprecessing_deviations_matches_lalsimulation(prec_version):
    mass_ratio = torch.tensor([2.0, 15.0, 25.0], dtype=torch.float64)
    eta = mass_ratio / (1.0 + mass_ratio) ** 2
    spin = torch.tensor([0.1, 0.9, 1.1], dtype=torch.float64)
    cosine = torch.tensor([0.25, -0.3, 0.8], dtype=torch.float64)
    zeros = torch.zeros_like(mass_ratio)
    single_spin = PNRSingleSpin(
        mass_ratio=mass_ratio,
        symmetric_mass_ratio=eta,
        magnitude=spin,
        cosine=cosine,
        antisymmetric_magnitude=zeros,
        antisymmetric_angle=zeros,
        final_cosine=zeros,
    )

    actual = build_pnr_coprecessing_deviations(
        single_spin,
        prec_version=prec_version,
    )
    if prec_version == 330:
        fitted_eta = torch.where(
            eta >= 0.09876,
            eta,
            0.09876 - (0.09876 - eta) * 0.1641,
        )
        fitted_spin = torch.maximum(
            torch.where(spin <= 0.8, spin, 0.8 + (spin - 0.8) / 12.0),
            spin.new_tensor(0.2),
        )
    else:
        fitted_eta = torch.clamp(eta, min=0.09876)
        fitted_spin = torch.clamp(spin, min=0.2, max=0.8)
    theta = torch.acos(cosine)
    expected_fits = np.asarray(
        [
            [
                getattr(lalsimulation, f"SimIMRPhenomXCP_{name}_l2m2")(
                    float(angle),
                    float(symmetric_mass_ratio),
                    float(spin_magnitude),
                )
                for name in _COPRECESSING_FIT_NAMES
            ]
            for angle, symmetric_mass_ratio, spin_magnitude in zip(
                theta,
                fitted_eta,
                fitted_spin,
            )
        ]
    )
    actual_fits = torch.stack(
        tuple(
            getattr(actual.fits, name.lower())
            for name in _COPRECESSING_FIT_NAMES
        ),
        dim=-1,
    )
    window = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    expected_strength = window * spin * torch.sqrt(1.0 - cosine * cosine)

    torch.testing.assert_close(actual.strength, expected_strength)
    np.testing.assert_allclose(
        actual_fits.numpy(),
        expected_fits,
        rtol=3.0e-12,
        atol=1.0e-12,
    )
