import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform.imrphenompv2_torch import (  # noqa: E402
    phenomp_source_frame_parameters,
)
from pycbc.waveform._mode_rotation_torch import (  # noqa: E402
    wigner_d_columns,
)
from pycbc.waveform.imrphenomxp_msa_torch import (  # noqa: E402
    build_msa_state,
    msa_angles,
    orbital_angular_momentum_3pn,
)


PRECESSING_CASES = (
    (40.0, 20.0, 30.0, 0.4, 1.1, (0.3, -0.2, 0.45), (-0.1, 0.25, -0.3)),
    (35.0, 30.0, 22.0, -0.7, 0.8, (-0.2, 0.1, -0.4), (0.15, -0.3, 0.2)),
    (55.0, 8.0, 40.0, 1.2, 2.2, (0.05, 0.45, 0.1), (-0.2, 0.05, 0.6)),
)

PV3HM_WIGNER_COLUMNS = (
    (2, 2, "d22"),
    (2, 1, "d21"),
    (3, 3, "d33"),
    (3, 2, "d32"),
    (4, 4, "d44"),
    (4, 3, "d43"),
)


def _lal_vector(values):
    vector = lal.CreateREAL8Vector(len(values))
    vector.data[:] = values
    return vector


def _spherical_spin(spin):
    magnitude = math.sqrt(sum(component * component for component in spin))
    if magnitude == 0.0:
        return 1.0, 0.0, 0.0
    return (
        spin[2] / magnitude,
        math.atan2(spin[1], spin[0]),
        magnitude,
    )


@pytest.mark.parametrize("ell,mprime,field", PV3HM_WIGNER_COLUMNS)
def test_pv3hm_wigner_columns_match_lalsimulation(ell, mprime, field):
    beta = torch.tensor((0.0, 0.2, 0.73, 1.4, math.pi), dtype=torch.float64)
    actual = wigner_d_columns(ell, mprime, beta)
    expected = np.stack(
        [
            getattr(
                lalsimulation.SimIMRPhenomPv3HMComputeWignerdElements(
                    ell, mprime, -float(angle)
                ),
                field,
            )
            for angle in beta
        ],
        axis=-1,
    )

    assert actual[0].shape == (2 * ell + 1, beta.numel())
    np.testing.assert_allclose(
        actual[0].numpy(), expected[0], rtol=3.0e-14, atol=3.0e-14
    )
    np.testing.assert_allclose(
        actual[1].numpy(), expected[1], rtol=3.0e-14, atol=3.0e-14
    )


@pytest.mark.parametrize("case", PRECESSING_CASES)
def test_pv3_source_frame_matches_lalsimulation(case):
    mass1, mass2, f_ref, coa_phase, inclination, spin1, spin2 = case
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    msa_state = build_msa_state(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )
    actual = phenomp_source_frame_parameters(
        mass1,
        mass2,
        f_ref,
        coa_phase,
        inclination,
        spin1,
        spin2,
        orbital_angular_momentum=lambda velocity, _eta: (
            orbital_angular_momentum_3pn(velocity, msa_state)
        ),
    )
    expected = lalsimulation.SimIMRPhenomPCalculateModelParametersFromSourceFrame(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        f_ref,
        coa_phase,
        inclination,
        *spin1,
        *spin2,
        lalsimulation.IMRPhenomPv3_V,
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.parametrize("case", PRECESSING_CASES)
def test_pv3_msa_dynamics_match_lalsimulation(case):
    mass1, mass2, f_ref, _coa_phase, _inclination, spin1, spin2 = case
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    orbital_frequencies = np.array((8.0, 12.5, f_ref / 2.0, 31.0, 73.0))
    velocity = torch.tensor(
        np.cbrt(2.0 * math.pi * total_mass_seconds * orbital_frequencies),
        dtype=torch.float64,
    )
    msa_state = build_msa_state(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )

    frequency_vector = _lal_vector(orbital_frequencies)
    expected_angles = [_lal_vector(np.zeros(5)) for _ in range(3)]
    spherical1 = _spherical_spin(spin1)
    spherical2 = _spherical_spin(spin2)
    lal_args = (
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        1.0,
        0.0,
        *spherical1,
        *spherical2,
        f_ref,
        5,
    )
    lalsimulation.ComputeAngles3PN(
        *expected_angles,
        frequency_vector,
        *lal_args,
    )
    expected_angular_momentum = _lal_vector(np.zeros(5))
    lalsimulation.OrbitalAngMom3PNSpinning(
        expected_angular_momentum,
        frequency_vector,
        *lal_args,
    )

    for actual, expected in zip(msa_angles(velocity, msa_state), expected_angles):
        np.testing.assert_allclose(
            actual.numpy(),
            expected.data,
            rtol=3.0e-12,
            atol=2.0e-11,
        )
    np.testing.assert_allclose(
        orbital_angular_momentum_3pn(velocity, msa_state).numpy(),
        expected_angular_momentum.data,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
