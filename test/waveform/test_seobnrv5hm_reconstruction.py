import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from pycbc.waveform import seobnrv5hm_torch as rom


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


def _random_submodel():
    q_breaks = (1.0, 2.0, 4.0, 8.0)
    chi1_breaks = (-1.0, -0.2, 0.4, 1.0)
    chi2_breaks = (-1.0, 0.0, 1.0)
    coefficient_shape = (3, 6, 6, 5)
    generator = torch.Generator().manual_seed(1729)

    def random(*sizes):
        return torch.randn(*sizes, generator=generator, dtype=torch.float64)

    return rom._SubModel(
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
        coeff_real=random(*coefficient_shape),
        coeff_imag=random(*coefficient_shape),
        coeff_phase=random(*coefficient_shape),
    )


def test_sparse_phase_and_cmode_reconstruction_match_scipy():
    submodel = _random_submodel()
    parameters = (2.7, 0.15, -0.35)
    local_basis = rom._parameter_basis(submodel, *parameters)
    phase = rom._evaluate_phase_submodel(submodel, local_basis)
    cmode = rom._evaluate_cmode_submodel(submodel, local_basis)
    full_basis = tuple(
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
        projected = np.einsum("nijk,i,j,k->n", coefficients.numpy(), *full_basis)
        return projection.numpy().T @ projected

    np.testing.assert_allclose(
        phase.phase.numpy(),
        expected(submodel.coeff_phase, submodel.basis_phase),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        cmode.real.numpy(),
        expected(submodel.coeff_real, submodel.basis_real),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        cmode.imag.numpy(),
        expected(submodel.coeff_imag, submodel.basis_imag),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert phase.frequency is submodel.g_phase
    assert cmode.frequency is submodel.g_cmode


def test_low_high_phase_and_cmode_hybridization():
    low_phase_frequency = torch.tensor(
        [0.001, 0.002, 0.0035, 0.005, 0.007, 0.009],
        dtype=torch.float64,
    )
    high_phase_frequency = torch.tensor(
        [0.0025, 0.003, 0.004, 0.006, 0.008, 0.01],
        dtype=torch.float64,
    )
    low_phase = rom._PatchPhaseEvaluation(
        low_phase_frequency, 3.0 * low_phase_frequency + 1.0
    )
    high_phase = rom._PatchPhaseEvaluation(
        high_phase_frequency, 7.0 * high_phase_frequency + 3.0
    )

    phase_frequency, carrier_phase = rom._hybridize_phase(
        low_phase, high_phase, omega_qnm=2.0 * math.pi
    )
    expected_phase_frequency = torch.tensor(
        [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01],
        dtype=torch.float64,
    )
    torch.testing.assert_close(phase_frequency, expected_phase_frequency)
    torch.testing.assert_close(
        carrier_phase,
        -(7.0 * expected_phase_frequency + 3.0),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    low_cmode_frequency = torch.tensor(
        [0.004, 0.007, 0.01, 0.013, 0.016, 0.019],
        dtype=torch.float64,
    )
    high_cmode_frequency = torch.tensor(
        [0.0085, 0.009, 0.012, 0.015, 0.018, 0.021],
        dtype=torch.float64,
    )

    def patch(frequency):
        return rom._PatchCModeEvaluation(
            frequency=frequency,
            real=2.0 + 4.0 * frequency,
            imag=-1.0 + 3.0 * frequency,
        )

    cmode = rom._hybridize_cmode(
        patch(low_cmode_frequency),
        patch(high_cmode_frequency),
        omega_qnm=2.0 * math.pi,
        mode_m=3,
    )
    expected_cmode_frequency = torch.tensor(
        [0.004, 0.007, 0.012, 0.015, 0.018, 0.021],
        dtype=torch.float64,
    )
    torch.testing.assert_close(cmode.frequency, expected_cmode_frequency)
    torch.testing.assert_close(
        cmode.cmode.real,
        2.0 + 4.0 * expected_cmode_frequency,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        cmode.cmode.imag,
        -1.0 + 3.0 * expected_cmode_frequency,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_real_low_frequency_mode_matches_scipy_reference():
    try:
        rom._find_rom_file()
    except FileNotFoundError:
        pytest.skip("SEOBNRv5HM ROM data is not available on LAL_DATA_PATH")

    submodel = rom._load_submodel("lowf", "43", torch.float64, torch.device("cpu"))
    parameters = (5.3, 0.37, -0.41)
    local_basis = rom._parameter_basis(submodel, *parameters)
    phase = rom._evaluate_phase_submodel(submodel, local_basis)
    cmode = rom._evaluate_cmode_submodel(submodel, local_basis)
    full_basis = tuple(
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
        projected = np.einsum("nijk,i,j,k->n", coefficients.numpy(), *full_basis)
        return projection.numpy().T @ projected

    np.testing.assert_allclose(
        phase.phase.numpy(),
        expected(submodel.coeff_phase, submodel.basis_phase),
        rtol=3.0e-14,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        cmode.real.numpy(),
        expected(submodel.coeff_real, submodel.basis_real),
        rtol=3.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        cmode.imag.numpy(),
        expected(submodel.coeff_imag, submodel.basis_imag),
        rtol=3.0e-14,
        atol=1.0e-14,
    )


def test_rom_mode_amp_phase_recovers_wrapped_residual_and_extrapolates():
    carrier_frequency = torch.tensor([0.001, 0.003, 0.005], dtype=torch.float64)
    carrier_phase = 0.1 + 10.0 * carrier_frequency
    mode_frequency = torch.tensor([0.003, 0.009, 0.015, 0.021], dtype=torch.float64)
    residual_phase = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64)
    amplitude = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    mode_m = 3
    expected_carrier = 0.1 + 10.0 * mode_frequency / mode_m
    approximate_phase = mode_m * expected_carrier + (1.0 - mode_m) * math.pi / 4.0
    cmode = amplitude * torch.exp(1j * (approximate_phase + residual_phase))
    evaluation = rom._SparseROMEvaluation(
        phase_frequency=carrier_frequency,
        carrier_phase=carrier_phase,
        modes={
            6: rom._HybridCModeEvaluation(
                frequency=mode_frequency,
                cmode=cmode,
            )
        },
    )

    reconstructed = rom._rom_mode_amp_phase(evaluation, 6)

    torch.testing.assert_close(reconstructed.amplitude, amplitude)
    torch.testing.assert_close(
        reconstructed.phase, residual_phase, rtol=2.0e-14, atol=2.0e-14
    )
    assert reconstructed.amplitude_frequency is mode_frequency
    assert reconstructed.phase_frequency is mode_frequency


@pytest.mark.parametrize("mode_index", range(len(rom._LM_MODES)))
def test_taylorf2_mode_amp_phase_matches_lal_formula(mode_index):
    q = 3.2
    spin1z = 0.37
    spin2z = -0.21
    frequency = torch.tensor([0.0003, 0.0008, 0.0017], dtype=torch.float64)
    coefficients = np.array([0.81, -0.13, 0.27, -0.09, 0.04, -0.02, 0.008, -0.003])
    log_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.011, -0.006, 0.0])
    phasing = SimpleNamespace(v=coefficients, vlogv=log_coefficients)

    amplitude, phase = rom._taylorf2_mode_amp_phase(
        q,
        spin1z,
        spin2z,
        phasing,
        mode_index,
        frequency,
    )

    ell, emm = rom._LM_MODES[mode_index]
    velocity = np.cbrt(math.pi * (2.0 / emm) * frequency.numpy())
    polynomial = np.zeros_like(velocity)
    for order in range(8):
        polynomial += (
            coefficients[order] + log_coefficients[order] * np.log(velocity)
        ) * velocity**order
    expected_phase = (
        polynomial / velocity**5 * (emm / 2.0)
        + rom._TAYLORF2_PHASE_SHIFTS[mode_index]
        - math.pi / 4.0
    )

    eta = q / (1.0 + q) ** 2
    delta = (q - 1.0 + np.finfo(np.float64).eps) / (1.0 + q)
    symmetric_spin = 0.5 * (spin1z + spin2z)
    antisymmetric_spin = 0.5 * (spin1z - spin2z)
    if (ell, emm) == (2, 2):
        mode_factor = np.ones_like(velocity)
    elif (ell, emm) == (2, 1):
        mode_factor = velocity * (
            delta / 3.0 - 0.5 * velocity * (antisymmetric_spin + delta * symmetric_spin)
        )
    elif (ell, emm) == (3, 3):
        mode_factor = velocity * 0.75 * np.sqrt(15.0 / 14.0) * delta
    elif (ell, emm) == (4, 4):
        mode_factor = velocity**2 * 8.0 * np.sqrt(35.0) / 63.0 * (1.0 - 3.0 * eta)
    elif (ell, emm) == (5, 5):
        mode_factor = (
            velocity**3 * 625.0 * np.sqrt(66.0) / 6336.0 * delta * (1.0 - 2.0 * eta)
        )
    elif (ell, emm) == (3, 2):
        mode_factor = (
            velocity**2
            * 9.0
            / 8.0
            * np.sqrt(5.0 / 7.0)
            * 8.0
            / 27.0
            * (-1.0 + 3.0 * eta)
        )
    else:
        mode_factor = (
            velocity**3
            * 8.0
            / 9.0
            * np.sqrt(10.0 / 7.0)
            * 81.0
            / 320.0
            * delta
            * (-1.0 + 2.0 * eta)
        )
    expected_amplitude = (
        math.pi
        * np.sqrt(2.0 * eta / 3.0)
        * velocity ** (-3.5)
        * np.sqrt(2.0 / emm)
        * mode_factor
    )
    np.testing.assert_allclose(amplitude.numpy(), expected_amplitude, rtol=2e-15)
    np.testing.assert_allclose(phase.numpy(), expected_phase, rtol=2e-15)


def test_inspiral_frequency_grid_matches_lal_boundaries():
    start_mf = 0.0002
    frequency = rom._inspiral_frequency_grid(
        start_mf, 4.0, torch.device("cpu"), torch.float64
    )

    assert frequency[0].item() == min(start_mf / 2.0, rom._mode_minimum_mf(2))
    assert frequency[-1].item() == (
        rom._PN_GRID_HIGH_FACTOR * rom._PN_HYBRID_END_FACTOR * rom._mode_minimum_mf(4)
    )
    assert bool(torch.all(torch.diff(frequency) > 0.0))


@pytest.mark.parametrize(
    ("start_mf", "q", "message"),
    [
        (0.0, 2.0, "starting geometric frequency"),
        (0.001, 0.9, "mass ratio"),
    ],
)
def test_inspiral_frequency_grid_rejects_invalid_inputs(start_mf, q, message):
    with pytest.raises(ValueError, match=message):
        rom._inspiral_frequency_grid(start_mf, q, torch.device("cpu"), torch.float64)


def test_phase_alignment_from_22_resolves_pi_ambiguity():
    frequency = torch.linspace(0.0004, 0.0018, 30, dtype=torch.float64)
    low_phase = 0.2 + 19.0 * frequency
    delta_time = torch.tensor(13.0, dtype=torch.float64)
    delta_phase = torch.tensor(-0.7, dtype=torch.float64)
    mode_m = 3
    alignment = 2.0 * math.pi * delta_time * frequency + mode_m / 2.0 * delta_phase
    high_phase = low_phase + alignment + 2.0 * math.pi

    actual = rom._phase_alignment_from_22(
        frequency,
        low_phase,
        frequency,
        high_phase,
        0.0007,
        0.0014,
        delta_time,
        delta_phase,
        mode_m,
    )

    torch.testing.assert_close(actual, high_phase, rtol=2.0e-13, atol=2.0e-13)


def test_hybridized_mode_subset_uses_22_alignment(monkeypatch):
    evaluation = rom._SparseROMEvaluation(
        phase_frequency=torch.linspace(0.0002, 0.003, 40, dtype=torch.float64),
        carrier_phase=torch.zeros(40, dtype=torch.float64),
        modes={},
    )
    reconstructed = []

    def fake_rom_mode_amp_phase(_evaluation, mode_index):
        reconstructed.append(mode_index)
        frequency = torch.linspace(0.0002, 0.003, 50, dtype=torch.float64)
        return rom._ModeAmpPhase(
            amplitude_frequency=frequency,
            amplitude=2.0 + frequency,
            phase_frequency=frequency,
            phase=0.3 * mode_index + 7.0 * frequency,
        )

    def fake_taylorf2(*args):
        frequency = args[-1]
        return 1.0 + frequency, 4.0 * frequency

    monkeypatch.setattr(rom, "_rom_mode_amp_phase", fake_rom_mode_amp_phase)
    monkeypatch.setattr(rom, "_taylorf2_mode_amp_phase", fake_taylorf2)
    monkeypatch.setattr(
        rom,
        "taylorf2_aligned_phasing",
        lambda *_args, **_kwargs: SimpleNamespace(v=np.zeros(8), vlogv=np.zeros(8)),
    )

    mode_data = rom._hybridized_mode_data(
        evaluation,
        40.0,
        20.0,
        0.2,
        -0.1,
        (6,),
        0.0004,
    )

    assert reconstructed == [0, 6]
    assert tuple(mode_data) == (6,)
    assert bool(torch.all(torch.diff(mode_data[6].amplitude_frequency) > 0.0))
    assert bool(torch.all(torch.diff(mode_data[6].phase_frequency) > 0.0))
    assert bool(torch.all(torch.isfinite(mode_data[6].amplitude)))
    assert bool(torch.all(torch.isfinite(mode_data[6].phase)))


def _polarization_inputs(
    active_mode_indices,
    *,
    sign_odd=1.0,
    long_asc_nodes=0.0,
):
    total_mass = 60.0
    total_mass_seconds = total_mass * rom.lal.MTSUN_SI
    cutoffs = (0.0018, 0.0030)
    return SimpleNamespace(
        evaluation=object(),
        mass1=40.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        active_mode_indices=active_mode_indices,
        sign_odd=sign_odd,
        total_mass=total_mass,
        total_mass_seconds=total_mass_seconds,
        distance=total_mass * total_mass_seconds * rom.lal.MRSUN_SI,
        inclination=0.9,
        coa_phase=0.37,
        long_asc_nodes=long_asc_nodes,
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        complex_dtype=torch.complex128,
        qnm_omega={
            mode: 2.0 * math.pi * cutoffs[index] / rom._CONST_FMAX[index]
            for index, mode in enumerate(rom._LM_MODES[:2])
        },
    )


def _assembly_mode(amplitude, phase):
    frequency = torch.tensor([0.0002, 0.0010, 0.0020, 0.0040], dtype=torch.float64)
    return rom._ModeAmpPhase(
        amplitude_frequency=frequency,
        amplitude=torch.full_like(frequency, amplitude),
        phase_frequency=frequency,
        phase=torch.full_like(frequency, phase),
    )


def test_polarization_assembly_sums_modes_and_applies_cutoffs(monkeypatch):
    mode_data = {
        0: _assembly_mode(2.0, 0.2),
        1: _assembly_mode(3.0, -0.4),
    }
    monkeypatch.setattr(
        rom,
        "_hybridized_mode_data",
        lambda *_args, **_kwargs: mode_data,
    )
    inputs = _polarization_inputs((0, 1))
    frequency = torch.tensor([0.0003, 0.0010, 0.0020, 0.0035], dtype=torch.float64)

    plus, cross = rom._seobnrv5hm_polarizations(inputs, frequency, 0.0004)

    expected_plus = torch.zeros_like(plus)
    expected_cross = torch.zeros_like(cross)
    observer_phi = math.pi / 2.0 - inputs.coa_phase
    for mode_index, (amplitude, phase) in enumerate(((2.0, 0.2), (3.0, -0.4))):
        ell, emm = rom._LM_MODES[mode_index]
        cutoff = (
            rom._CONST_FMAX[mode_index] * inputs.qnm_omega[(ell, emm)] / (2.0 * math.pi)
        )
        active = frequency <= cutoff
        shifted_mode = ((-1) ** ell) * torch.conj(
            torch.full(
                (int(active.sum()),),
                amplitude * np.exp(1j * phase),
                dtype=torch.complex128,
            )
            * torch.exp((-2j * math.pi * 1000.0) * frequency[active])
        )
        y_negative = rom.spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            ell,
            -emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        y_positive_conjugate = rom.spin_weighted_spherical_harmonic(
            inputs.inclination,
            observer_phi,
            -2,
            ell,
            emm,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ).conj()
        parity = (-1) ** ell
        expected_plus[active] += (
            0.5 * (y_negative + parity * y_positive_conjugate) * shifted_mode
        )
        expected_cross[active] += (
            0.5j * (y_negative - parity * y_positive_conjugate) * shifted_mode
        )

    torch.testing.assert_close(plus, expected_plus)
    torch.testing.assert_close(cross, expected_cross)
    assert plus[-1] == 0.0
    assert cross[-1] == 0.0


def test_polarization_assembly_applies_odd_mode_sign(monkeypatch):
    monkeypatch.setattr(
        rom,
        "_hybridized_mode_data",
        lambda *_args, **_kwargs: {1: _assembly_mode(3.0, -0.4)},
    )
    frequency = torch.tensor([0.0004, 0.0011], dtype=torch.float64)

    plus, cross = rom._seobnrv5hm_polarizations(
        _polarization_inputs((1,), sign_odd=1.0), frequency, 0.0004
    )
    swapped_plus, swapped_cross = rom._seobnrv5hm_polarizations(
        _polarization_inputs((1,), sign_odd=-1.0), frequency, 0.0004
    )

    torch.testing.assert_close(swapped_plus, -plus)
    torch.testing.assert_close(swapped_cross, -cross)


def test_polarization_assembly_rotates_ascending_node(monkeypatch):
    monkeypatch.setattr(
        rom,
        "_hybridized_mode_data",
        lambda *_args, **_kwargs: {0: _assembly_mode(2.0, 0.2)},
    )
    frequency = torch.tensor([0.0004, 0.0011], dtype=torch.float64)
    angle = 0.31

    plus, cross = rom._seobnrv5hm_polarizations(
        _polarization_inputs((0,)), frequency, 0.0004
    )
    rotated_plus, rotated_cross = rom._seobnrv5hm_polarizations(
        _polarization_inputs((0,), long_asc_nodes=angle),
        frequency,
        0.0004,
    )

    cosine = math.cos(2.0 * angle)
    sine = math.sin(2.0 * angle)
    torch.testing.assert_close(rotated_plus, cosine * plus + sine * cross)
    torch.testing.assert_close(rotated_cross, cosine * cross - sine * plus)


def test_polarization_assembly_rejects_frequency_below_spline_domain():
    with pytest.raises(ValueError, match="below the TaylorF2 spline domain"):
        rom._seobnrv5hm_polarizations(
            _polarization_inputs((0,)),
            torch.tensor([0.0001], dtype=torch.float64),
            0.0004,
        )
