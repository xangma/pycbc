# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Parity tests for the shared Torch NRTidal corrections."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform.nrtidal_torch import (  # noqa: E402
    nrtidal_amplitude,
    nrtidal_higher_order_spin_terms,
    nrtidal_kappa2t,
    nrtidal_merger_frequency,
    nrtidal_merger_frequency_v3,
    nrtidal_octupole_from_quadrupole,
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
    nrtidal_taper,
    nrtidal_version,
)


_MASSES = (1.4, 1.2)
_LAMBDAS = (400.0, 800.0)


@pytest.mark.parametrize(
    ("approximant", "version"),
    [
        ("IMRPhenomD_NRTidal", 1),
        ("IMRPhenomPv2_NRTidal", 1),
        ("SEOBNRv4_ROM_NRTidal", 1),
        ("IMRPhenomD_NRTidalv2", 2),
        ("IMRPhenomPv2_NRTidalv2", 2),
        ("IMRPhenomXAS_NRTidalv2", 2),
        ("IMRPhenomXP_NRTidalv2", 2),
        ("SEOBNRv4_ROM_NRTidalv2", 2),
        ("IMRPhenomXAS_NRTidalv3", 3),
        ("IMRPhenomXP_NRTidalv3", 3),
        ("SEOBNRv5_ROM_NRTidalv3", 3),
        ("IMRPhenomXAS", None),
    ],
)
def test_nrtidal_version(approximant, version):
    assert nrtidal_version(approximant) == version


def _lal_vector(values):
    vector = lal.CreateREAL8Vector(len(values))
    vector.data[:] = values
    return vector


def test_nrtidal_scalar_fits_match_lal():
    mass1, mass2 = _MASSES
    lambda1, lambda2 = _LAMBDAS
    expected_kappa = lalsimulation.SimNRTunedTidesComputeKappa2T(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        lambda1,
        lambda2,
    )
    actual_kappa = nrtidal_kappa2t(mass1, mass2, lambda1, lambda2)
    assert actual_kappa == pytest.approx(expected_kappa, rel=1.0e-14)

    expected_merger = lalsimulation.SimNRTunedTidesMergerFrequency(
        mass1 + mass2,
        expected_kappa,
        mass1 / mass2,
    )
    actual_merger = nrtidal_merger_frequency(mass1, mass2, lambda1, lambda2)
    assert actual_merger == pytest.approx(expected_merger, rel=1.0e-14)


@pytest.mark.parametrize(
    ("masses", "lambdas", "spins"),
    [
        (_MASSES, _LAMBDAS, (0.15, -0.08)),
        (_MASSES[::-1], _LAMBDAS[::-1], (-0.08, 0.15)),
    ],
)
def test_nrtidal_v3_merger_frequency_matches_lal(masses, lambdas, spins):
    mass1, mass2 = masses
    lambda1, lambda2 = lambdas
    spin1z, spin2z = spins
    primary_mass = max(mass1, mass2)
    secondary_mass = min(mass1, mass2)
    if mass1 >= mass2:
        primary_lambda, secondary_lambda = lambda1, lambda2
        primary_spin, secondary_spin = spin1z, spin2z
    else:
        primary_lambda, secondary_lambda = lambda2, lambda1
        primary_spin, secondary_spin = spin2z, spin1z
    expected = lalsimulation.SimNRTunedTidesMergerFrequency_v3(
        mass1 + mass2,
        primary_lambda,
        secondary_lambda,
        primary_mass / secondary_mass,
        primary_spin,
        secondary_spin,
    )
    actual = nrtidal_merger_frequency_v3(
        mass1,
        mass2,
        lambda1,
        lambda2,
        spin1z,
        spin2z,
    )
    assert actual == pytest.approx(expected, rel=1.0e-14)


@pytest.mark.parametrize("lambda_tidal", [0.0, 0.499, 0.5, 1.0, 400.0, 5000.0])
def test_nrtidal_quadrupole_fit_matches_legacy_lal_dispatch(lambda_tidal):
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda_tidal)
    lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(lal_params, 0.0)
    lalsimulation.SimInspiralSetQuadMonParamsFromLambdas(lal_params)
    expected = lalsimulation.SimInspiralWaveformParamsLookupdQuadMon1(lal_params) + 1.0
    assert nrtidal_quadrupole_from_lambda(lambda_tidal) == pytest.approx(
        expected, rel=1.0e-14
    )


def test_nrtidal_higher_order_spin_fits_match_lal():
    mass1, mass2 = _MASSES
    xa = mass1 / (mass1 + mass2)
    xb = mass2 / (mass1 + mass2)
    spin1z, spin2z = 0.15, -0.08
    quadrupole1 = nrtidal_quadrupole_from_lambda(_LAMBDAS[0])
    quadrupole2 = nrtidal_quadrupole_from_lambda(_LAMBDAS[1])

    for quadrupole in (quadrupole1, quadrupole2):
        expected = lalsimulation.SimUniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole(
            quadrupole
        )
        assert nrtidal_octupole_from_quadrupole(quadrupole) == pytest.approx(
            expected, rel=1.0e-14
        )

    expected = lalsimulation.SimInspiralGetHOSpinTerms(
        xa,
        xb,
        spin1z,
        spin2z,
        quadrupole1,
        quadrupole2,
    )
    actual = nrtidal_higher_order_spin_terms(
        mass1,
        mass2,
        spin1z,
        spin2z,
        quadrupole1,
        quadrupole2,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=0.0)


_LAL_NRTIDALV3 = getattr(lalsimulation, "NRTidalv3_V", None)
_NRTIDAL_VERSIONS = [
    (1, getattr(lalsimulation, "NRTidal_V", 1)),
    (2, getattr(lalsimulation, "NRTidalv2_V", 2)),
] + ([(3, _LAL_NRTIDALV3)] if _LAL_NRTIDALV3 is not None else [])


@pytest.mark.parametrize(
    ("version", "lal_version"),
    _NRTIDAL_VERSIONS,
)
def test_nrtidal_frequency_corrections_match_lal(version, lal_version):
    mass1, mass2 = _MASSES
    lambda1, lambda2 = _LAMBDAS
    spin1z, spin2z = 0.15, -0.08
    if version == 3:
        merger = nrtidal_merger_frequency_v3(
            mass1,
            mass2,
            lambda1,
            lambda2,
            spin1z,
            spin2z,
        )
        frequencies = np.linspace(20.0, 1.5 * merger, 512, dtype=np.float64)
    else:
        merger = nrtidal_merger_frequency(mass1, mass2, lambda1, lambda2)
        frequencies = np.array(
            [
                20.0,
                100.0,
                500.0,
                0.99 * merger,
                1.05 * merger,
                1.19 * merger,
                1.21 * merger,
            ],
            dtype=np.float64,
        )
    frequency_vector = _lal_vector(frequencies)
    phase = lal.CreateREAL8Vector(len(frequencies))
    amplitude = lal.CreateREAL8Vector(len(frequencies))
    taper = lal.CreateREAL8Vector(len(frequencies))
    status = lalsimulation.SimNRTunedTidesFDTidalPhaseFrequencySeries(
        phase,
        amplitude,
        taper,
        frequency_vector,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        lambda1,
        lambda2,
        spin1z,
        spin2z,
        lal_version,
    )
    assert status == 0

    torch_frequencies = torch.as_tensor(frequencies, dtype=torch.float64)
    phase_tolerance = 3.0e-13 if version == 3 else 2.0e-13
    np.testing.assert_allclose(
        nrtidal_phase(
            torch_frequencies,
            mass1,
            mass2,
            lambda1,
            lambda2,
            version,
            spin1z,
            spin2z,
        ).numpy(),
        phase.data,
        rtol=phase_tolerance,
        atol=phase_tolerance,
    )
    np.testing.assert_allclose(
        nrtidal_taper(torch_frequencies, merger).numpy(),
        taper.data,
        rtol=0.0,
        atol=3.0e-14 if version == 3 else 3.0e-15,
    )
    if version in (2, 3):
        np.testing.assert_allclose(
            nrtidal_amplitude(
                torch_frequencies,
                mass1,
                mass2,
                lambda1,
                lambda2,
            ).numpy(),
            amplitude.data,
            rtol=2.0e-13,
            atol=2.0e-13,
        )


def test_nrtidal_float32_taper_preserves_lal_support():
    merger = nrtidal_merger_frequency(*_MASSES, *_LAMBDAS)
    width = 0.2 * merger
    frequencies = [
        merger + 0.95 * width,
        merger + 0.97 * width,
        merger + 0.98 * width,
    ]
    expected = nrtidal_taper(
        torch.tensor(frequencies, dtype=torch.float64),
        merger,
    )
    actual = nrtidal_taper(
        torch.tensor(frequencies, dtype=torch.float32),
        merger,
    )

    np.testing.assert_array_equal(
        actual.numpy() == 0.0,
        expected.numpy() == 0.0,
    )
