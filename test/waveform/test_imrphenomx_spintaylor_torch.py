import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform.imrphenomx_spintaylor_torch import (  # noqa: E402
    SpinTaylorJFrame,
    SpinTaylorTrajectory,
    build_spintaylor_angle_spline,
    build_spintaylor_alpha_mrd,
    build_spintaylor_beta_mrd,
    spintaylor_alpha_imr,
    spintaylor_alpha_mrd,
    spintaylor_alpha_mrd_derivative,
    spintaylor_alpha_reference_offset,
    spintaylor_beta_imr,
    spintaylor_beta_mrd,
    spintaylor_inspiral_alpha,
    spintaylor_inspiral_cosbeta,
    spintaylor_internal_spins,
    spintaylor_j_frame,
    spintaylor_t4_orbital_derivatives,
    spintaylor_t4_inspiral_angles,
    spintaylor_t4_rhs,
    spintaylor_t4_time_trajectory,
    spintaylor_t4_trajectory,
    spintaylor_t4_vector_derivatives,
    spintaylor_unwrap_angle,
)


_PARAMETERS = (
    pytest.param(
        35.0,
        20.0,
        (0.2, 0.1, 0.4),
        (-0.1, 0.15, -0.2),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        1.0,
        1.0,
        id="unequal-mass",
    ),
    pytest.param(
        40.0,
        10.0,
        (0.7, -0.2, 0.1),
        (0.1, 0.3, -0.6),
        (0.2, -0.3, np.sqrt(0.87)),
        (np.sqrt(0.87), 0.0, -0.2),
        1.0,
        1.0,
        id="rotated-frame",
    ),
    pytest.param(
        30.0,
        30.0,
        (0.3, 0.2, -0.4),
        (-0.2, 0.1, 0.5),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        2.5,
        4.0,
        id="quadrupole",
    ),
)


def _five_point_forward_derivative(series, delta_t):
    coefficients = np.array((-25.0, 48.0, -36.0, 16.0, -3.0))
    if isinstance(series, np.ndarray):
        values = series[:5]
    else:
        values = series.data.data[:5]
    return coefficients @ np.asarray(values) / (12.0 * delta_t)


@lru_cache
def _lal_derivatives(
    mass1,
    mass2,
    chi1,
    chi2,
    lnhat,
    e1,
    quadrupole1,
    quadrupole2,
    lambda1=0.0,
    lambda2=0.0,
):
    delta_t = 1.0 / 65536.0
    series = lalsimulation.SimInspiralSpinTaylorPNEvolveOrbit(
        delta_t,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        20.0,
        20.01,
        *chi1,
        *chi2,
        *lnhat,
        *e1,
        lambda1,
        lambda2,
        quadrupole1,
        quadrupole2,
        6,
        12,
        7,
        0,
        lalsimulation.SpinTaylorT4,
    )
    physical_derivatives = np.array(
        [_five_point_forward_derivative(item, delta_t) for item in series]
    )
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)

    return (
        float(series[0].data.data[0]),
        total_mass_seconds * physical_derivatives[1],
        total_mass_seconds
        * _five_point_forward_derivative(
            np.asarray(series[0].data.data[:5]) ** 3,
            delta_t,
        ),
        total_mass_seconds * physical_derivatives[8:11],
        total_mass_seconds * mass1_fraction**2 * physical_derivatives[2:5],
        total_mass_seconds * mass2_fraction**2 * physical_derivatives[5:8],
        total_mass_seconds * physical_derivatives[11:14],
    )


@lru_cache
def _lal_final_state(
    mass1,
    mass2,
    chi1,
    chi2,
    reference_frequency,
    target_frequency,
    quadrupole1=1.0,
    quadrupole2=1.0,
    lambda1=0.0,
    lambda2=0.0,
):
    series = lalsimulation.SimInspiralSpinTaylorPNEvolveOrbitOnlyFinal(
        1.0 / 16384.0,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        reference_frequency,
        target_frequency,
        *chi1,
        *chi2,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        lambda1,
        lambda2,
        quadrupole1,
        quadrupole2,
        6,
        12,
        7,
        0,
        lalsimulation.SpinTaylorT4,
    )
    values = np.array([item.data.data[0] for item in series])
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)
    state = np.concatenate(
        (
            (values[1], values[0] ** 3),
            values[8:11],
            mass1_fraction**2 * values[2:5],
            mass2_fraction**2 * values[5:8],
            values[11:14],
        )
    )
    # The LAL final-state helper may stop just beyond the requested frequency.
    # Return the frequency it actually reached for an apples-to-apples check.
    return values[0] ** 3 / math.pi, state


def _lal_time_branch(
    mass1,
    mass2,
    chi1,
    chi2,
    start_frequency,
    end_frequency,
    sample_interval,
    *,
    lnhat=(0.0, 0.0, 1.0),
    e1=(1.0, 0.0, 0.0),
):
    series = lalsimulation.SimInspiralSpinTaylorPNEvolveOrbit(
        sample_interval,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        start_frequency,
        end_frequency,
        *chi1,
        *chi2,
        *lnhat,
        *e1,
        0.0,
        0.0,
        1.0,
        1.0,
        6,
        12,
        7,
        0,
        lalsimulation.SpinTaylorT4,
    )
    values = np.stack([np.asarray(item.data.data) for item in series], axis=-1)
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)
    return np.concatenate(
        (
            values[:, 1:2],
            values[:, 0:1] ** 3,
            values[:, 8:11],
            mass1_fraction**2 * values[:, 2:5],
            mass2_fraction**2 * values[:, 5:8],
            values[:, 11:14],
        ),
        axis=-1,
    )


@lru_cache
def _lal_spintaylor_angles(
    mass1,
    mass2,
    chi1,
    chi2,
    fmin,
    fmax,
    delta_f,
    reference_frequency,
    phi_ref,
):
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(
        lal_params,
        330,
    )
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXPConvention(
        lal_params,
        1,
    )
    result = lalsimulation.SimIMRPhenomXPSpinTaylorAngles(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        *chi1,
        *chi2,
        fmin,
        fmax,
        delta_f,
        reference_frequency,
        phi_ref,
        lal_params,
    )
    return tuple(np.asarray(angle.data) for angle in result)


def test_spintaylor_internal_spin_convention_is_batchable():
    mass1 = torch.tensor([35.0, 40.0, 30.0], dtype=torch.float64)
    mass2 = torch.tensor([20.0, 10.0, 30.0], dtype=torch.float64)
    chi1 = torch.tensor([case.values[2] for case in _PARAMETERS], dtype=torch.float64)
    chi2 = torch.tensor([case.values[3] for case in _PARAMETERS], dtype=torch.float64)

    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)

    expected1 = (mass1 / (mass1 + mass2)).square().unsqueeze(-1) * chi1
    expected2 = (mass2 / (mass1 + mass2)).square().unsqueeze(-1) * chi2
    torch.testing.assert_close(spin1, expected1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(spin2, expected2, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    (
        "mass1",
        "mass2",
        "chi1",
        "chi2",
        "lnhat",
        "e1",
        "quadrupole1",
        "quadrupole2",
    ),
    _PARAMETERS,
)
def test_spintaylor_t4_vector_field_matches_lalsimulation(
    mass1,
    mass2,
    chi1,
    chi2,
    lnhat,
    e1,
    quadrupole1,
    quadrupole2,
):
    (
        v,
        _,
        _,
        expected_lnhat,
        expected_spin1,
        expected_spin2,
        expected_e1,
    ) = _lal_derivatives(
        mass1,
        mass2,
        chi1,
        chi2,
        lnhat,
        e1,
        quadrupole1,
        quadrupole2,
    )
    lnhat = torch.tensor(lnhat, dtype=torch.float64)
    e1 = torch.tensor(e1, dtype=torch.float64)
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    total_mass = mass1 + mass2

    actual = spintaylor_t4_vector_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        e1,
        mass1 / total_mass,
        mass2 / total_mass,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
    )

    np.testing.assert_allclose(
        actual.lnhat.numpy(), expected_lnhat, rtol=2.0e-8, atol=3.0e-13
    )
    np.testing.assert_allclose(
        actual.spin1.numpy(), expected_spin1, rtol=2.0e-8, atol=3.0e-13
    )
    np.testing.assert_allclose(
        actual.spin2.numpy(), expected_spin2, rtol=2.0e-8, atol=3.0e-13
    )
    np.testing.assert_allclose(
        actual.e1.numpy(), expected_e1, rtol=2.0e-8, atol=3.0e-13
    )

    assert actual.lnhat.device == lnhat.device
    assert actual.lnhat.dtype == lnhat.dtype


@pytest.mark.parametrize(
    (
        "mass1",
        "mass2",
        "chi1",
        "chi2",
        "lnhat",
        "e1",
        "quadrupole1",
        "quadrupole2",
    ),
    _PARAMETERS,
)
def test_spintaylor_t4_orbital_field_matches_lalsimulation(
    mass1,
    mass2,
    chi1,
    chi2,
    lnhat,
    e1,
    quadrupole1,
    quadrupole2,
):
    v, expected_phase, expected_omega, *_ = _lal_derivatives(
        mass1,
        mass2,
        chi1,
        chi2,
        lnhat,
        e1,
        quadrupole1,
        quadrupole2,
    )
    lnhat = torch.tensor(lnhat, dtype=torch.float64)
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    total_mass = mass1 + mass2

    actual = spintaylor_t4_orbital_derivatives(
        v,
        lnhat,
        spin1,
        spin2,
        mass1 / total_mass,
        mass2 / total_mass,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
    )

    np.testing.assert_allclose(
        actual.phase.numpy(), expected_phase, rtol=2.0e-8, atol=3.0e-13
    )
    np.testing.assert_allclose(
        actual.omega.numpy(), expected_omega, rtol=2.0e-8, atol=3.0e-13
    )
    assert actual.omega.device == lnhat.device
    assert actual.omega.dtype == lnhat.dtype


def test_spintaylor_t4_complete_rhs_matches_lalsimulation_with_tides():
    mass1 = 35.0
    mass2 = 20.0
    chi1 = (0.2, 0.1, 0.4)
    chi2 = (-0.1, 0.15, -0.2)
    lnhat = (0.0, 0.0, 1.0)
    e1 = (1.0, 0.0, 0.0)
    quadrupole1 = 2.5
    quadrupole2 = 4.0
    lambda1 = 500.0
    lambda2 = 800.0
    (
        v,
        expected_phase,
        expected_omega,
        expected_lnhat,
        expected_spin1,
        expected_spin2,
        expected_e1,
    ) = _lal_derivatives(
        mass1,
        mass2,
        chi1,
        chi2,
        lnhat,
        e1,
        quadrupole1,
        quadrupole2,
        lambda1,
        lambda2,
    )
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    state = torch.cat(
        (
            torch.tensor([0.0, v**3], dtype=torch.float64),
            torch.tensor(lnhat, dtype=torch.float64),
            spin1,
            spin2,
            torch.tensor(e1, dtype=torch.float64),
        )
    )
    expected = np.concatenate(
        (
            (expected_phase, expected_omega),
            expected_lnhat,
            expected_spin1,
            expected_spin2,
            expected_e1,
        )
    )

    actual = spintaylor_t4_rhs(
        state,
        mass1 / (mass1 + mass2),
        mass2 / (mass1 + mass2),
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-8, atol=3.0e-13)


def test_spintaylor_t4_rhs_validates_state_shape():
    with pytest.raises(ValueError, match="final dimension of 14"):
        spintaylor_t4_rhs(torch.zeros(13), 0.6, 0.4)


def test_spintaylor_t4_rhs_preserves_batch_and_device():
    dtype = torch.float64
    mass1 = torch.tensor([35.0, 40.0], dtype=dtype)
    mass2 = torch.tensor([20.0, 10.0], dtype=dtype)
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)
    chi1 = torch.tensor([[0.2, 0.1, 0.4], [0.7, -0.2, 0.1]], dtype=dtype)
    chi2 = torch.tensor([[-0.1, 0.15, -0.2], [0.1, 0.3, -0.6]], dtype=dtype)
    lnhat = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.3, np.sqrt(0.87)]], dtype=dtype)
    e1 = torch.tensor([[1.0, 0.0, 0.0], [np.sqrt(0.87), 0.0, -0.2]], dtype=dtype)
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    v = torch.tensor([0.257, 0.248], dtype=dtype)
    state = torch.cat(
        (
            torch.zeros((2, 1), dtype=dtype),
            v.pow(3).unsqueeze(-1),
            lnhat,
            spin1,
            spin2,
            e1,
        ),
        dim=-1,
    )
    quadrupole1 = torch.tensor([1.0, 2.5], dtype=dtype)
    quadrupole2 = torch.tensor([1.0, 4.0], dtype=dtype)
    lambda1 = torch.tensor([0.0, 500.0], dtype=dtype)
    lambda2 = torch.tensor([0.0, 800.0], dtype=dtype)

    actual = spintaylor_t4_rhs(
        state,
        mass1_fraction,
        mass2_fraction,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    expected = torch.stack(
        [
            spintaylor_t4_rhs(
                state[index],
                mass1_fraction[index],
                mass2_fraction[index],
                quadrupole1=quadrupole1[index],
                quadrupole2=quadrupole2[index],
                lambda1=lambda1[index],
                lambda2=lambda2[index],
            )
            for index in range(2)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=2.0e-18)
    assert actual.device == state.device
    assert actual.dtype == state.dtype


def test_spintaylor_t4_vector_field_preserves_vector_constraints():
    dtype = torch.float64
    lnhat = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.3, np.sqrt(0.87)]], dtype=dtype)
    e1 = torch.tensor([[1.0, 0.0, 0.0], [np.sqrt(0.87), 0.0, -0.2]], dtype=dtype)
    chi1 = torch.tensor([[0.2, 0.1, 0.4], [0.7, -0.2, 0.1]], dtype=dtype)
    chi2 = torch.tensor([[-0.1, 0.15, -0.2], [0.1, 0.3, -0.6]], dtype=dtype)
    mass1 = torch.tensor([35.0, 40.0], dtype=dtype)
    mass2 = torch.tensor([20.0, 10.0], dtype=dtype)
    spin1, spin2 = spintaylor_internal_spins(mass1, mass2, chi1, chi2)
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)

    derivatives = spintaylor_t4_vector_derivatives(
        torch.tensor([0.257, 0.248], dtype=dtype),
        lnhat,
        spin1,
        spin2,
        e1,
        mass1_fraction,
        mass2_fraction,
    )

    torch.testing.assert_close(
        torch.sum(spin1 * derivatives.spin1, dim=-1),
        torch.zeros(2, dtype=dtype),
        rtol=0.0,
        atol=2.0e-20,
    )
    torch.testing.assert_close(
        torch.sum(spin2 * derivatives.spin2, dim=-1),
        torch.zeros(2, dtype=dtype),
        rtol=0.0,
        atol=2.0e-20,
    )
    torch.testing.assert_close(
        torch.sum(lnhat * derivatives.lnhat, dim=-1),
        torch.zeros(2, dtype=dtype),
        rtol=0.0,
        atol=2.0e-20,
    )
    torch.testing.assert_close(
        torch.sum(e1 * derivatives.lnhat + lnhat * derivatives.e1, dim=-1),
        torch.zeros(2, dtype=dtype),
        rtol=0.0,
        atol=2.0e-20,
    )


@pytest.mark.parametrize(
    ("quadrupole1", "quadrupole2", "lambda1", "lambda2"),
    (
        pytest.param(1.0, 1.0, 0.0, 0.0, id="black-hole"),
        pytest.param(2.5, 4.0, 500.0, 800.0, id="matter"),
    ),
)
def test_spintaylor_t4_trajectory_matches_lalsimulation(
    quadrupole1,
    quadrupole2,
    lambda1,
    lambda2,
):
    mass1 = 35.0
    mass2 = 20.0
    chi1 = (0.2, 0.1, 0.4)
    chi2 = (-0.1, 0.15, -0.2)
    reference_frequency = 30.0
    target_frequencies = (20.0, 25.0, 35.0, 60.0)
    reference_mf = (mass1 + mass2) * lal.MTSUN_SI * reference_frequency
    lal_results = [
        _lal_final_state(
            mass1,
            mass2,
            chi1,
            chi2,
            reference_frequency,
            target,
            quadrupole1,
            quadrupole2,
            lambda1,
            lambda2,
        )
        for target in target_frequencies
    ]
    mf = torch.tensor([result[0] for result in lal_results], dtype=torch.float64)
    expected = np.stack([result[1] for result in lal_results])

    actual = spintaylor_t4_trajectory(
        mf,
        reference_mf,
        mass1,
        mass2,
        chi1,
        chi2,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    np.testing.assert_allclose(actual.phase.numpy(), expected[:, 0], atol=2.0e-7)
    np.testing.assert_allclose(actual.omega.numpy(), expected[:, 1], atol=2.0e-16)
    np.testing.assert_allclose(
        actual.state[:, 2:].numpy(), expected[:, 2:], rtol=2.0e-8, atol=3.0e-9
    )
    torch.testing.assert_close(actual.omega, math.pi * actual.mf)


def test_spintaylor_t4_time_trajectory_matches_lalsimulation():
    mass1 = 40.0
    mass2 = 20.0
    chi1 = (0.25, -0.1, 0.4)
    chi2 = (-0.15, 0.05, -0.2)
    start_frequency = 19.5
    reference_frequency = 30.0
    # This is the initial PhenomX ringdown-plus-eight-damping cutoff for the
    # binary.  Its uniform grid reaches the physical PN boundary first.
    end_frequency = 669.2463857284947
    sample_interval = 0.5 / end_frequency
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI

    backward = _lal_time_branch(
        mass1,
        mass2,
        chi1,
        chi2,
        reference_frequency,
        start_frequency,
        sample_interval,
    )
    forward = _lal_time_branch(
        mass1,
        mass2,
        chi1,
        chi2,
        reference_frequency,
        end_frequency,
        sample_interval,
    )
    expected = np.concatenate((backward, forward[1:]))

    actual = spintaylor_t4_time_trajectory(
        start_frequency * total_mass_seconds,
        reference_frequency * total_mass_seconds,
        end_frequency * total_mass_seconds,
        mass1,
        mass2,
        chi1,
        chi2,
    )

    assert actual.state.shape == expected.shape
    np.testing.assert_allclose(actual.phase.numpy(), expected[:, 0], atol=4.0e-8)
    np.testing.assert_allclose(
        actual.state[:, 1:].numpy(),
        expected[:, 1:],
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    torch.testing.assert_close(actual.omega, math.pi * actual.mf)
    assert actual.mf[-1] < end_frequency * total_mass_seconds


def test_spintaylor_t4_time_trajectory_matches_pnr_fine_grid():
    mass1 = 40.0
    mass2 = 20.0
    chi1 = (0.25, -0.1, 0.4)
    chi2 = (-0.15, 0.05, -0.2)
    start_frequency = 19.5
    reference_frequency = 30.0
    end_frequency = 669.2463857284947
    coarse_factor = 10.0
    coarse_interval = 0.5 * coarse_factor / end_frequency
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI

    backward = _lal_time_branch(
        mass1,
        mass2,
        chi1,
        chi2,
        reference_frequency,
        start_frequency,
        coarse_interval,
    )
    forward = _lal_time_branch(
        mass1,
        mass2,
        chi1,
        chi2,
        reference_frequency,
        end_frequency,
        coarse_interval,
    )
    # IMRPhenomX's appendTS overwrites the penultimate backward sample and
    # reserves one unused tail position in the combined coarse series.
    coarse = np.concatenate((backward[:-2], forward))
    coarse_length = coarse.shape[0] + 1
    transition_index = coarse_length - 1 - min(9, coarse_length - 1)
    transition = coarse[transition_index]
    mass1_fraction = mass1 / (mass1 + mass2)
    mass2_fraction = mass2 / (mass1 + mass2)
    fine = _lal_time_branch(
        mass1,
        mass2,
        tuple(transition[5:8] / mass1_fraction**2),
        tuple(transition[8:11] / mass2_fraction**2),
        transition[1] / (math.pi * total_mass_seconds),
        end_frequency,
        0.5 / end_frequency,
        lnhat=tuple(transition[2:5]),
        e1=tuple(transition[11:14]),
    )
    fine[:, 0] += transition[0]
    expected = np.concatenate((coarse[:transition_index], fine))

    actual = spintaylor_t4_time_trajectory(
        start_frequency * total_mass_seconds,
        reference_frequency * total_mass_seconds,
        end_frequency * total_mass_seconds,
        mass1,
        mass2,
        chi1,
        chi2,
        coarse_factor=coarse_factor,
        pnr_fine_grid=True,
    )

    assert actual.state.shape == expected.shape
    np.testing.assert_allclose(actual.phase.numpy(), expected[:, 0], atol=5.0e-8)
    np.testing.assert_allclose(
        actual.state[:, 1:].numpy(),
        expected[:, 1:],
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    torch.testing.assert_close(actual.omega, math.pi * actual.mf)
    assert actual.mf[-1] < end_frequency * total_mass_seconds


def test_spintaylor_t4_trajectory_keeps_reference_state_and_constraints():
    dtype = torch.float64
    mf = torch.tensor([0.004, 0.006, 0.008], dtype=dtype)
    chi1 = torch.tensor([0.2, 0.1, 0.4], dtype=dtype)
    chi2 = torch.tensor([-0.1, 0.15, -0.2], dtype=dtype)
    trajectory = spintaylor_t4_trajectory(mf, mf[1], 35.0, 20.0, chi1, chi2)
    spin1, spin2 = spintaylor_internal_spins(35.0, 20.0, chi1, chi2)

    torch.testing.assert_close(
        trajectory.state[1],
        torch.cat(
            (
                torch.tensor([0.0, math.pi * mf[1]], dtype=dtype),
                torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
                spin1,
                spin2,
                torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
            )
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(trajectory.lnhat, dim=-1),
        torch.ones(3, dtype=dtype),
        rtol=0.0,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        torch.sum(trajectory.lnhat * trajectory.e1, dim=-1),
        torch.zeros(3, dtype=dtype),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert trajectory.state.device == mf.device
    assert trajectory.state.dtype == mf.dtype


def test_spintaylor_t4_trajectory_stops_at_pn_energy_boundary():
    total_mass_seconds = 55.0 * lal.MTSUN_SI
    with pytest.raises(RuntimeError, match="PN orbital-energy boundary"):
        spintaylor_t4_trajectory(
            torch.tensor([100.0 * total_mass_seconds], dtype=torch.float64),
            30.0 * total_mass_seconds,
            35.0,
            20.0,
            (0.2, 0.1, 0.4),
            (-0.1, 0.15, -0.2),
            quadrupole1=2.5,
            quadrupole2=4.0,
            lambda1=500.0,
            lambda2=800.0,
        )


def test_spintaylor_t4_trajectory_can_retain_physical_prefix():
    total_mass_seconds = 55.0 * lal.MTSUN_SI
    mf = torch.arange(20.0, 181.0, dtype=torch.float64) * total_mass_seconds
    kwargs = {
        "quadrupole1": 2.5,
        "quadrupole2": 4.0,
        "lambda1": 500.0,
        "lambda2": 800.0,
    }

    truncated = spintaylor_t4_trajectory(
        mf,
        30.0 * total_mass_seconds,
        35.0,
        20.0,
        (0.2, 0.1, 0.4),
        (-0.1, 0.15, -0.2),
        truncate_at_boundary=True,
        **kwargs,
    )

    assert 4 <= truncated.mf.numel() < mf.numel()
    torch.testing.assert_close(truncated.mf, mf[: truncated.mf.numel()])
    expected = spintaylor_t4_trajectory(
        truncated.mf,
        30.0 * total_mass_seconds,
        35.0,
        20.0,
        (0.2, 0.1, 0.4),
        (-0.1, 0.15, -0.2),
        **kwargs,
    )
    torch.testing.assert_close(truncated.state, expected.state)


@pytest.mark.parametrize(
    ("frequencies", "message"),
    (
        ([0.006, 0.004], "strictly increasing"),
        ([0.0, 0.006], "finite and positive"),
        ([], "nonempty vector"),
    ),
)
def test_spintaylor_t4_trajectory_validates_frequency_grid(frequencies, message):
    with pytest.raises(ValueError, match=message):
        spintaylor_t4_trajectory(
            frequencies,
            0.006,
            35.0,
            20.0,
            (0.2, 0.1, 0.4),
            (-0.1, 0.15, -0.2),
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("rtol", math.nan, "tolerances"),
        ("max_steps", 10.5, "positive integer"),
        ("lambda1", math.inf, "matter parameters must be finite"),
        ("truncate_at_boundary", 1, "must be boolean"),
    ),
)
def test_spintaylor_t4_trajectory_validates_controls(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        spintaylor_t4_trajectory(
            [0.004, 0.006],
            0.005,
            35.0,
            20.0,
            (0.2, 0.1, 0.4),
            (-0.1, 0.15, -0.2),
            **{keyword: value},
        )


@pytest.mark.parametrize(
    (
        "mass1",
        "mass2",
        "chi1",
        "chi2",
        "fmin",
        "fmax",
        "reference_frequency",
        "phi_ref",
    ),
    (
        pytest.param(
            35.0,
            20.0,
            (0.2, 0.1, 0.4),
            (-0.1, 0.15, -0.2),
            20.0,
            50.0,
            30.0,
            0.0,
            id="unequal-mass",
        ),
        pytest.param(
            40.0,
            10.0,
            (0.7, -0.2, 0.1),
            (0.1, 0.3, -0.6),
            18.0,
            45.0,
            25.0,
            0.4,
            id="strong-precession",
        ),
        pytest.param(
            30.0,
            30.0,
            (0.3, 0.2, -0.4),
            (-0.2, 0.1, 0.5),
            20.0,
            45.0,
            30.0,
            -0.3,
            id="equal-mass",
        ),
    ),
)
def test_spintaylor_t4_inspiral_angles_match_lalsimulation(
    mass1,
    mass2,
    chi1,
    chi2,
    fmin,
    fmax,
    reference_frequency,
    phi_ref,
):
    delta_f = 1.0
    frequencies = np.arange(fmin, fmax + 0.5 * delta_f, delta_f)
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    mf = torch.tensor(frequencies * total_mass_seconds, dtype=torch.float64)
    expected_alpha, expected_cosbeta, expected_gamma = _lal_spintaylor_angles(
        mass1,
        mass2,
        chi1,
        chi2,
        fmin,
        fmax,
        delta_f,
        reference_frequency,
        phi_ref,
    )

    actual = spintaylor_t4_inspiral_angles(
        mf,
        reference_frequency * total_mass_seconds,
        mass1,
        mass2,
        chi1,
        chi2,
        phi_ref=phi_ref,
    )

    # LAL first integrates on a coarse uniform-time grid and splines in
    # frequency.  The native path integrates directly to the requested
    # frequencies, so the remaining milliradian differences measure that
    # sampling choice rather than a difference in the SpinTaylor equations.
    np.testing.assert_allclose(actual.alpha.numpy(), expected_alpha, atol=3.0e-3)
    np.testing.assert_allclose(
        actual.cosbeta.numpy(),
        expected_cosbeta,
        atol=3.0e-4,
    )
    np.testing.assert_allclose(actual.gamma.numpy(), expected_gamma, atol=4.0e-3)
    assert actual.alpha.device == mf.device
    assert actual.alpha.dtype == mf.dtype


def test_spintaylor_t4_angles_apply_reference_offsets():
    total_mass_seconds = 55.0 * lal.MTSUN_SI
    frequencies = torch.arange(20.0, 51.0, dtype=torch.float64)
    mf = frequencies * total_mass_seconds
    reference_mf = 30.0 * total_mass_seconds
    chi1 = (0.2, 0.1, 0.4)
    chi2 = (-0.1, 0.15, -0.2)
    frame = spintaylor_j_frame(
        reference_mf,
        35.0,
        20.0,
        chi1,
        chi2,
    )

    angles = spintaylor_t4_inspiral_angles(
        mf,
        reference_mf,
        35.0,
        20.0,
        chi1,
        chi2,
    )

    reference_index = 10
    torch.testing.assert_close(
        angles.alpha[reference_index],
        frame.alpha0,
        rtol=0.0,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        angles.gamma[reference_index],
        -frame.epsilon0,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_spintaylor_j_frame_canonicalizes_mass_order():
    chi1 = torch.tensor((0.2, 0.1, 0.4), dtype=torch.float64)
    chi2 = torch.tensor((-0.1, 0.15, -0.2), dtype=torch.float64)
    canonical = spintaylor_j_frame(0.008, 35.0, 20.0, chi1, chi2, phi_ref=0.3)
    reversed_order = spintaylor_j_frame(
        0.008,
        20.0,
        35.0,
        chi2,
        chi1,
        phi_ref=0.3,
    )

    for field in (
        "phi_j_source",
        "theta_j_source",
        "kappa",
        "alpha0",
        "epsilon0",
    ):
        torch.testing.assert_close(
            getattr(reversed_order, field),
            getattr(canonical, field),
            rtol=0.0,
            atol=0.0,
        )


def test_spintaylor_unwrap_angle_matches_single_turn_lal_rule():
    angle = torch.tensor((3.0, -3.0, -2.5, 3.1), dtype=torch.float64)
    actual = spintaylor_unwrap_angle(angle)
    expected = torch.tensor(
        (3.0, 2.0 * math.pi - 3.0, 2.0 * math.pi - 2.5, 3.1),
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-15)


def test_spintaylor_alpha_mrd_matches_inspiral_spline_constraints():
    mf = torch.linspace(0.01, 0.05, 17, dtype=torch.float64)
    alpha = torch.sin(37.0 * mf) + 0.2 * mf.square()
    fmax = mf[-1]
    parameters = build_spintaylor_alpha_mrd(mf, alpha, fmax)
    reference = CubicSpline(mf.numpy(), alpha.numpy(), bc_type="natural")
    f1 = 0.97 * float(fmax)
    f2 = 0.99 * float(fmax)

    np.testing.assert_allclose(
        spintaylor_alpha_mrd(
            torch.tensor((f1, f2), dtype=torch.float64),
            parameters,
        ).numpy(),
        reference((f1, f2)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        spintaylor_alpha_mrd_derivative(f1, parameters).numpy(),
        reference(f1, 1),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_spintaylor_beta_mrd_matches_inspiral_spline_constraints():
    mf = torch.linspace(0.01, 0.05, 17, dtype=torch.float64)
    cosbeta = 0.35 + 0.08 * torch.sin(29.0 * mf)
    fmax = mf[-1]
    parameters = build_spintaylor_beta_mrd(
        mf,
        cosbeta,
        fmax,
        damping_difference=0.013,
        ringdown_beta=0.42,
    )
    reference = CubicSpline(mf.numpy(), cosbeta.numpy(), bc_type="natural")
    f1 = 0.97 * float(fmax)
    f2 = 0.98 * float(fmax)
    expected = np.arccos(reference((f1, f2)))

    np.testing.assert_allclose(
        spintaylor_beta_mrd(
            torch.tensor((f1, f2), dtype=torch.float64),
            parameters,
        ).numpy(),
        expected,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    point = torch.tensor(f2, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(
        spintaylor_beta_mrd(point, parameters),
        point,
    )[0]
    expected_derivative = -reference(f2, 1) / np.sqrt(1.0 - reference(f2) ** 2)
    np.testing.assert_allclose(
        derivative.detach().numpy(),
        expected_derivative,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    assert not bool(parameters.flat)


@pytest.mark.parametrize(
    ("cosbeta", "expected"),
    ((1.01, 0.0), (-1.01, math.pi)),
)
def test_spintaylor_beta_mrd_flattens_invalid_inspiral_spline(cosbeta, expected):
    mf = torch.linspace(0.01, 0.05, 9, dtype=torch.float32)
    parameters = build_spintaylor_beta_mrd(
        mf,
        torch.full_like(mf, cosbeta),
        mf[-1],
        damping_difference=0.01,
        ringdown_beta=0.4,
    )
    actual = spintaylor_beta_mrd(torch.tensor((0.05, 0.08)), parameters)

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, expected),
        rtol=0.0,
        atol=2.0e-7,
    )
    assert bool(parameters.flat)
    assert actual.dtype == mf.dtype
    assert actual.device == mf.device


def _synthetic_angle_trajectory(dtype=torch.float64):
    mf = torch.linspace(0.01, 0.05, 17, dtype=dtype)
    alpha = 0.3 + 23.0 * mf + 11.0 * mf.square()
    beta = 0.4 + 0.08 * torch.sin(31.0 * mf)
    lnhat = torch.stack(
        (
            torch.sin(beta) * torch.cos(alpha),
            torch.sin(beta) * torch.sin(alpha),
            torch.cos(beta),
        ),
        dim=-1,
    )
    state = torch.zeros((mf.numel(), 14), dtype=dtype)
    state[:, 2:5] = lnhat
    frame = SpinTaylorJFrame(
        phi_j_source=mf.new_zeros(()),
        theta_j_source=mf.new_zeros(()),
        kappa=mf.new_zeros(()),
        alpha0=mf.new_tensor(1.2),
        epsilon0=mf.new_tensor(-0.2),
    )
    return SpinTaylorTrajectory(mf=mf, state=state), frame, alpha, torch.cos(beta)


def test_spintaylor_angle_spline_owns_inspiral_and_generic_imr_parts():
    trajectory, frame, alpha, cosbeta = _synthetic_angle_trajectory()
    angles = build_spintaylor_angle_spline(
        trajectory,
        frame,
        trajectory.mf[-1],
        damping_difference=0.013,
        ringdown_beta=0.42,
    )
    reference_alpha = CubicSpline(
        trajectory.mf.numpy(),
        alpha.numpy(),
        bc_type="natural",
    )
    reference_cosbeta = CubicSpline(
        trajectory.mf.numpy(),
        cosbeta.numpy(),
        bc_type="natural",
    )
    inspiral_points = torch.tensor((0.015, 0.031, 0.047), dtype=torch.float64)

    np.testing.assert_allclose(
        spintaylor_inspiral_alpha(inspiral_points, angles).numpy(),
        reference_alpha(inspiral_points.numpy()),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        spintaylor_inspiral_cosbeta(inspiral_points, angles).numpy(),
        reference_cosbeta(inspiral_points.numpy()),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    points = torch.tensor((0.047, 0.049, 0.08), dtype=torch.float64)
    expected_alpha = torch.stack(
        (
            spintaylor_inspiral_alpha(points[0], angles),
            spintaylor_alpha_mrd(points[1], angles.alpha_mrd),
            spintaylor_alpha_mrd(points[2], angles.alpha_mrd),
        )
    )
    expected_beta = torch.stack(
        (
            torch.acos(spintaylor_inspiral_cosbeta(points[0], angles)),
            spintaylor_beta_mrd(points[1], angles.beta_mrd),
            spintaylor_beta_mrd(points[2], angles.beta_mrd),
        )
    )
    torch.testing.assert_close(
        spintaylor_alpha_imr(points, angles),
        expected_alpha,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        spintaylor_beta_imr(points, angles),
        expected_beta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        angles.ftrans_mrd,
        0.98 * trajectory.mf[-1],
        rtol=0.0,
        atol=0.0,
    )


def test_spintaylor_angle_spline_reference_offset_uses_imr_alpha():
    trajectory, frame, _, _ = _synthetic_angle_trajectory(dtype=torch.float32)
    angles = build_spintaylor_angle_spline(
        trajectory,
        frame,
        trajectory.mf[-1],
        damping_difference=0.013,
        ringdown_beta=0.42,
    )
    mf_ref = trajectory.mf.new_tensor(0.0495)
    offset = spintaylor_alpha_reference_offset(mf_ref, angles)

    torch.testing.assert_close(
        spintaylor_alpha_imr(mf_ref, angles, offset=offset),
        frame.alpha0,
        rtol=0.0,
        atol=2.0e-6,
    )
    assert offset.dtype == trajectory.mf.dtype
    assert offset.device == trajectory.mf.device


@pytest.mark.parametrize(
    ("mf", "values", "fmax", "message"),
    (
        ([0.01], [0.2], 0.01, "at least two"),
        ([0.01, 0.02], [0.2], 0.02, "equal-length"),
        ([0.01, 0.02], [0.2, 0.3], 0.03, "spline domain"),
    ),
)
def test_spintaylor_alpha_mrd_validates_spline(mf, values, fmax, message):
    with pytest.raises(ValueError, match=message):
        build_spintaylor_alpha_mrd(mf, values, fmax)


@pytest.mark.parametrize(
    ("frequencies", "reference_mf", "message"),
    (
        ([0.004, 0.005, 0.007, 0.008], 0.005, "uniformly spaced"),
        ([0.004, 0.005, 0.006, 0.007], 0.008, "lie on the angle grid"),
        ([0.004, 0.005, 0.006], 0.005, "at least four"),
    ),
)
def test_spintaylor_t4_inspiral_angles_validate_grid(
    frequencies,
    reference_mf,
    message,
):
    with pytest.raises(ValueError, match=message):
        spintaylor_t4_inspiral_angles(
            frequencies,
            reference_mf,
            35.0,
            20.0,
            (0.2, 0.1, 0.4),
            (-0.1, 0.15, -0.2),
        )
