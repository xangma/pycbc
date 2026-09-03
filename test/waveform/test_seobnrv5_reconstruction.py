import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from pycbc.waveform import seobnrv5_torch as rom


@pytest.fixture(autouse=True)
def clear_rom_caches():
    rom._clear_rom_cache()
    yield
    rom._clear_rom_cache()


def _scipy_basis(breaks, value):
    knots = np.concatenate(
        (
            np.repeat(breaks[0], rom._SPLINE_DEGREE),
            breaks,
            np.repeat(breaks[-1], rom._SPLINE_DEGREE),
        )
    )
    return BSpline.design_matrix(
        [value], knots, rom._SPLINE_DEGREE, extrapolate=False
    ).toarray()[0]


@pytest.mark.parametrize("value", [0.0, 0.15, 0.7, 1.0])
def test_bspline_window_matches_scipy(value):
    breaks = (0.0, 0.15, 0.4, 0.7, 1.0)
    grid = torch.tensor(breaks, dtype=torch.float64)

    first, local = rom._bspline_window(breaks, grid, value)
    actual = torch.zeros(len(breaks) + 2, dtype=torch.float64)
    actual[first : first + 4] = local

    np.testing.assert_allclose(
        actual.numpy(), _scipy_basis(np.asarray(breaks), value), atol=2.0e-15
    )


def test_sparse_reconstruction_matches_independent_tensor_product():
    q_breaks = (1.0, 2.0, 4.0, 8.0)
    chi1_breaks = (-1.0, -0.2, 0.4, 1.0)
    chi2_breaks = (-1.0, 0.0, 1.0)
    shape = (3, 6, 6, 5)
    generator = torch.Generator().manual_seed(1729)

    def random(*sizes):
        return torch.randn(*sizes, generator=generator, dtype=torch.float64)

    submodel = rom._SubModel(
        q_breaks=q_breaks,
        chi1_breaks=chi1_breaks,
        chi2_breaks=chi2_breaks,
        qvec=torch.tensor(q_breaks, dtype=torch.float64),
        chi1vec=torch.tensor(chi1_breaks, dtype=torch.float64),
        chi2vec=torch.tensor(chi2_breaks, dtype=torch.float64),
        g_cmode=torch.linspace(0.01, 0.03, 4, dtype=torch.float64),
        g_phase=torch.linspace(0.01, 0.04, 5, dtype=torch.float64),
        basis_real=random(3, 4),
        basis_imag=random(3, 4),
        basis_phase=random(3, 5),
        coeff_real=random(*shape),
        coeff_imag=random(*shape),
        coeff_phase=random(*shape),
    )
    parameters = (2.7, 0.15, -0.35)

    actual = rom._evaluate_submodel(submodel, *parameters)
    basis = tuple(
        _scipy_basis(np.asarray(breaks), value)
        for breaks, value in zip((q_breaks, chi1_breaks, chi2_breaks), parameters)
    )

    def expected(coefficients, projection):
        projected = np.einsum("nijk,i,j,k->n", coefficients.numpy(), *basis)
        return projection.numpy().T @ projected

    np.testing.assert_allclose(
        actual.cmode_real.numpy(),
        expected(submodel.coeff_real, submodel.basis_real),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        actual.cmode_imag.numpy(),
        expected(submodel.coeff_imag, submodel.basis_imag),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected(submodel.coeff_phase, submodel.basis_phase),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert actual.cmode_frequency is submodel.g_cmode
    assert actual.phase_frequency is submodel.g_phase


def test_real_low_frequency_patch_matches_scipy_reference():
    try:
        rom._find_rom_file()
    except FileNotFoundError:
        pytest.skip("SEOBNRv5 ROM data is not available on LAL_DATA_PATH")

    submodel = rom._load_submodel("lowf", torch.float64, torch.device("cpu"))
    parameters = (5.3, 0.37, -0.41)
    actual = rom._evaluate_submodel(submodel, *parameters)
    basis = tuple(
        _scipy_basis(np.asarray(breaks), value)
        for breaks, value in zip(
            (
                submodel.q_breaks,
                submodel.chi1_breaks,
                submodel.chi2_breaks,
            ),
            parameters,
        )
    )

    def expected(coefficients, projection):
        projected = np.einsum("nijk,i,j,k->n", coefficients.numpy(), *basis)
        return projection.numpy().T @ projected

    np.testing.assert_allclose(
        actual.cmode_real.numpy(),
        expected(submodel.coeff_real, submodel.basis_real),
        rtol=3.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual.cmode_imag.numpy(),
        expected(submodel.coeff_imag, submodel.basis_imag),
        rtol=3.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected(submodel.coeff_phase, submodel.basis_phase),
        rtol=3.0e-14,
        atol=1.0e-10,
    )


def test_low_high_patch_hybridization_aligns_and_blends():
    low_frequency = torch.tensor(
        [0.001, 0.002, 0.0035, 0.005, 0.007, 0.009],
        dtype=torch.float64,
    )
    high_frequency = torch.tensor(
        [0.0025, 0.003, 0.004, 0.006, 0.008, 0.01],
        dtype=torch.float64,
    )

    def evaluation(frequency, phase_offset=False):
        phase = 3.0 * frequency + 1.0
        if phase_offset:
            phase = phase + 4.0 * frequency + 2.0
        return rom._PatchEvaluation(
            cmode_frequency=frequency,
            cmode_real=2.0 + 4.0 * frequency,
            cmode_imag=-1.0 + 3.0 * frequency,
            phase_frequency=frequency,
            phase=phase,
        )

    actual = rom._hybridize_evaluations(
        evaluation(low_frequency),
        evaluation(high_frequency, phase_offset=True),
        omega_qnm=2.0 * np.pi,
    )

    expected_phase_frequency = torch.tensor(
        [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01],
        dtype=torch.float64,
    )
    expected_cmode_frequency = torch.tensor(
        [0.001, 0.002, 0.0035, 0.005, 0.006, 0.008, 0.01],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual.phase_frequency, expected_phase_frequency)
    torch.testing.assert_close(
        actual.carrier_phase,
        -(7.0 * expected_phase_frequency + 3.0),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(actual.cmode_frequency, expected_cmode_frequency)
    torch.testing.assert_close(
        actual.cmode.real,
        2.0 + 4.0 * expected_cmode_frequency,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        actual.cmode.imag,
        -1.0 + 3.0 * expected_cmode_frequency,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_rom_mode_reconstruction_unwraps_and_extrapolates_carrier():
    carrier_frequency = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
    carrier_phase = 2.0 + 3.0 * carrier_frequency
    cmode_frequency = torch.tensor([0.002, 0.004, 0.006, 0.008], dtype=torch.float64)
    residual_phase = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64)
    approximation = 2.0 * (2.0 + 3.0 * cmode_frequency / 2.0) - np.pi / 4.0
    amplitude = 5.0 + cmode_frequency
    cmode = torch.polar(amplitude, approximation + residual_phase)
    evaluation = rom._HybridEvaluation(
        cmode_frequency=cmode_frequency,
        cmode=cmode,
        phase_frequency=carrier_frequency,
        carrier_phase=carrier_phase,
    )

    actual = rom._rom_mode_amp_phase(evaluation)

    torch.testing.assert_close(actual.amplitude, amplitude)
    torch.testing.assert_close(actual.phase, residual_phase)
    assert actual.amplitude_frequency is cmode_frequency
    assert actual.phase_frequency is cmode_frequency


def test_inspiral_grid_matches_v5_boundaries_and_is_ordered():
    start_mf = 0.0002
    actual = rom._inspiral_frequency_grid(
        start_mf, 4.0, torch.device("cpu"), torch.float64
    )

    assert actual[0].item() == min(start_mf / 2.0, rom._MF_LOW_22 / 2.0)
    assert actual[-1].item() == pytest.approx(1.1 * 2.0 * rom._MF_LOW_22 * 5.0 / 2.0)
    assert torch.all(torch.diff(actual) > 0.0)
    with pytest.raises(ValueError, match="starting geometric frequency"):
        rom._inspiral_frequency_grid(0.0, 4.0, torch.device("cpu"), torch.float64)


def test_taylorf2_rom_hybrid_has_continuous_domains():
    carrier_frequency = torch.linspace(0.0002, 0.012, 80, dtype=torch.float64)
    carrier_phase = 120.0 * carrier_frequency + 0.4
    cmode_frequency = torch.linspace(rom._MF_LOW_22, 0.02, 120, dtype=torch.float64)
    residual_phase = 250.0 * cmode_frequency - 1.2
    approximation = 2.0 * (120.0 * cmode_frequency / 2.0 + 0.4) - np.pi / 4.0
    amplitude = 1.0 + 8.0 * cmode_frequency
    evaluation = rom._HybridEvaluation(
        cmode_frequency=cmode_frequency,
        cmode=torch.polar(amplitude, approximation + residual_phase),
        phase_frequency=carrier_frequency,
        carrier_phase=carrier_phase,
    )

    actual = rom._hybridize_mode_with_taylorf2(
        evaluation,
        mass1=40.0,
        mass2=10.0,
        spin1z=0.3,
        spin2z=-0.2,
        start_mf=0.0002,
    )

    assert actual.amplitude_frequency[0].item() == pytest.approx(0.0001)
    assert actual.phase_frequency[0].item() == pytest.approx(0.0001)
    assert torch.all(torch.diff(actual.amplitude_frequency) > 0.0)
    assert torch.all(torch.diff(actual.phase_frequency) > 0.0)
    assert torch.all(torch.isfinite(actual.amplitude))
    assert torch.all(torch.isfinite(actual.phase))
    assert torch.all(actual.amplitude > 0.0)


def test_real_rom_taylorf2_hybrid_is_finite_and_mass_order_invariant():
    try:
        rom._find_rom_file()
    except FileNotFoundError:
        pytest.skip("SEOBNRv5 ROM data is not available on LAL_DATA_PATH")

    data = rom._load_rom(torch.float64, torch.device("cpu"))
    ordered = rom._evaluate_hybridized_mode(data, 40.0, 10.0, 0.3, -0.2, 0.0002)
    swapped = rom._evaluate_hybridized_mode(data, 10.0, 40.0, -0.2, 0.3, 0.0002)

    torch.testing.assert_close(ordered.amplitude_frequency, swapped.amplitude_frequency)
    torch.testing.assert_close(ordered.amplitude, swapped.amplitude)
    torch.testing.assert_close(ordered.phase_frequency, swapped.phase_frequency)
    torch.testing.assert_close(ordered.phase, swapped.phase)
    assert torch.all(torch.isfinite(ordered.amplitude))
    assert torch.all(torch.isfinite(ordered.phase))
