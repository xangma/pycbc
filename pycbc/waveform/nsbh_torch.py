# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch evaluator for the SEOBNRv4-ROM NSBH amplitude correction.

The scalar compactness, disruption, and remnant fits follow
``LALSimNSBHProperties.c``, ``LALSimBHNSRemnantFits.c``, and
``LALSimIMRSEOBNRv4ROM_NSBHAmplitudeCorrection.c`` from LALSuite 7.26.1.
Scalar fit setup and the polynomial root solve remain on the host. The
frequency-dependent correction is evaluated on the Torch device of the
supplied frequency tensor.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import lal


_MASS_FIT_COEFFICIENTS = (
    -1.83417425e-3,
    2.39226041e-3,
    4.29407902e-3,
    9.79775571e-3,
    2.33868869e-7,
    -8.28090025e-7,
    -1.64315549e-6,
    8.08340931e-6,
    -2.00726981e-2,
    1.31986011e-1,
    6.50754064e-2,
    -1.42749961e-1,
)
_SPIN_FIT_COEFFICIENTS = (
    -5.44187381e-3,
    7.91165608e-3,
    2.33362046e-2,
    2.47764497e-2,
    -8.56844797e-7,
    -2.81727682e-6,
    6.61290966e-6,
    4.28979016e-5,
    -3.04174272e-2,
    2.54889050e-1,
    1.47549350e-1,
    -4.27905832e-1,
)


def nsbh_compactness_from_lambda(lambda_value: float) -> float:
    """Return the NS compactness fit, including its black-hole limit."""

    if lambda_value > 1.0:
        log_lambda = math.log(lambda_value)
        return 0.360 - 0.0355 * log_lambda + 0.000705 * log_lambda**2
    lambda_sq = lambda_value * lambda_value
    lambda_cu = lambda_sq * lambda_value
    return 0.5 + (3.0 * 0.360 + 0.0355 - 1.5) * lambda_sq + (
        -2.0 * 0.360 - 0.0355 + 1.0
    ) * lambda_cu


def nsbh_r_kerr_isco(spin: float) -> float:
    """Return the unit-mass Kerr ISCO radius."""

    z1 = 1.0 + (1.0 - spin**2) ** (1.0 / 3.0) * (
        (1.0 + spin) ** (1.0 / 3.0)
        + (1.0 - spin) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * spin**2 + z1**2)
    radical = math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 3.0 + z2 - radical if spin > 0.0 else 3.0 + z2 + radical


def nsbh_xi_tide(mass_ratio: float, spin: float, mu: float) -> float:
    """Return the relativistic mass-shedding-radius correction."""

    coefficients = np.array(
        [
            -3.0 * mass_ratio * mu**2 * spin**2,
            0.0,
            6.0 * mass_ratio * mu,
            0.0,
            -3.0 * mass_ratio,
            0.0,
            0.0,
            2.0 * spin * mu**1.5,
            -3.0 * mu,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )
    roots = np.roots(coefficients[::-1])
    candidates = roots.real[
        (np.abs(roots.imag) < 1.0e-5) & (roots.real > 0.0)
    ]
    if candidates.size == 0:
        return 0.0
    return float(np.max(candidates**2))


def nsbh_torus_mass_fit(
    mass_ratio: float, spin: float, compactness: float
) -> float:
    """Return the remnant torus baryonic-mass fraction."""

    mu = mass_ratio * compactness
    xi_tide = nsbh_xi_tide(mass_ratio, spin, mu)
    torus_mass = (
        0.296 * xi_tide * (1.0 - 2.0 * compactness)
        - 0.171 * mu * nsbh_r_kerr_isco(spin)
    )
    return max(torus_mass, 0.0)


def bbh_final_mass_non_precessing_uib2016(
    mass1: float, mass2: float, spin1z: float, spin2z: float
) -> float:
    """Return the aligned-spin UIB2016 BBH remnant mass."""

    total_mass = mass1 + mass2
    mass1_sq = mass1 * mass1
    mass2_sq = mass2 * mass2
    eta = min(max(mass1 * mass2 / total_mass**2, 0.0), 0.25)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    shat = (spin1z * mass1_sq + spin2z * mass2_sq) / (
        mass1_sq + mass2_sq
    )
    shat2 = shat * shat
    shat3 = shat2 * shat
    spin_difference = spin1z - spin2z
    if mass2 > mass1:
        spin_difference = -spin_difference
    sqrt_asymmetry = math.sqrt(1.0 - 4.0 * eta)

    nonspinning = (
        (1.0 - 2.0 * math.sqrt(2.0) / 3.0) * eta
        + 0.5609904135313374 * eta2
        - 0.84667563764404 * eta3
        + 3.145145224278187 * eta4
    )
    spin_numerator = (
        1.0
        + 0.346
        * -0.2091189048177395
        * shat
        * (
            1.8083565298668276
            + 15.738082204419655 * eta
            + (16.0 - 16.0 * 1.8083565298668276 - 4.0 * 15.738082204419655)
            * eta2
        )
        + 0.211
        * -0.19709136361080587
        * shat2
        * (
            4.271313308472851
            + (16.0 - 16.0 * 4.271313308472851) * eta2
        )
        + 0.128
        * -0.1588185739358418
        * shat3
        * (
            31.08987570280556
            - 243.6299258830685 * eta
            + (16.0 - 16.0 * 31.08987570280556 + 4.0 * 243.6299258830685)
            * eta2
        )
    )
    spin_denominator = (
        1.0
        - 0.212
        * 2.9852925538232014
        * shat
        * (
            1.5673498395263061
            - 0.5808669012986468 * eta
            + (16.0 - 16.0 * 1.5673498395263061 + 4.0 * 0.5808669012986468)
            * eta2
        )
    )
    radiated_energy = nonspinning * spin_numerator / spin_denominator
    radiated_energy += (
        -0.09803730445895877
        * sqrt_asymmetry
        * eta2
        * (1.0 - 3.2283713377939134 * eta)
        * spin_difference
        - 0.01978238971523653
        * shat
        * sqrt_asymmetry
        * eta
        * (1.0 - 4.91667749015812 * eta)
        * spin_difference
        + 0.01118530335431078 * eta3 * spin_difference**2
    )
    return total_mass * (1.0 - radiated_energy)


def bbh_final_spin_non_precessing_uib2016(
    mass1: float, mass2: float, spin1z: float, spin2z: float
) -> float:
    """Return the aligned-spin UIB2016 BBH remnant spin."""

    total_mass = mass1 + mass2
    total_mass_sq = total_mass * total_mass
    mass1_sq = mass1 * mass1
    mass2_sq = mass2 * mass2
    eta = min(max(mass1 * mass2 / total_mass_sq, 0.0), 0.25)
    eta2 = eta * eta
    eta3 = eta2 * eta
    total_spin = (
        spin1z * mass1_sq + spin2z * mass2_sq
    ) / total_mass_sq
    shat = (spin1z * mass1_sq + spin2z * mass2_sq) / (
        mass1_sq + mass2_sq
    )
    shat2 = shat * shat
    shat3 = shat2 * shat
    spin_difference = spin1z - spin2z
    if mass2 > mass1:
        spin_difference = -spin_difference
    sqrt_asymmetry = math.sqrt(1.0 - 4.0 * eta)

    orbital = (
        2.0 * math.sqrt(3.0) * eta
        + 5.24 * 3.8326341618708577 * eta2
        + 1.3 * -9.487364155598392 * eta3
    ) / (1.0 + 2.88 * 2.5134875145648374 * eta)
    orbital += (
        -0.194
        * 1.0009563702914628
        * shat
        * (
            4.409160174224525 * eta
            + 0.5118334706832706 * eta2
            + (64.0 - 16.0 * 4.409160174224525 - 4.0 * 0.5118334706832706)
            * eta3
        )
        + 0.0851
        * 0.7877509372255369
        * shat2
        * (
            8.77367320110712 * eta
            - 32.060648277652994 * eta2
            + (64.0 - 16.0 * 8.77367320110712 + 4.0 * 32.060648277652994)
            * eta3
        )
        + 0.00954
        * 0.6540138407185817
        * shat3
        * (
            22.830033250479833 * eta
            - 153.83722669033995 * eta2
            + (64.0 - 16.0 * 22.830033250479833 + 4.0 * 153.83722669033995)
            * eta3
        )
    ) / (
        1.0
        - 0.579
        * 0.8396665722805308
        * shat
        * (
            1.8804718791591157
            - 4.770246856212403 * eta
            + (64.0 - 64.0 * 1.8804718791591157 + 16.0 * 4.770246856212403)
            * eta3
        )
    )
    orbital += (
        0.3223660562764661
        * sqrt_asymmetry
        * eta2
        * (1.0 + 9.332575956437443 * eta)
        * spin_difference
        + 2.3170397514509933
        * shat
        * sqrt_asymmetry
        * eta3
        * (1.0 - 3.2624649875884852 * eta)
        * spin_difference
        - 0.059808322561702126 * eta3 * spin_difference**2
    )
    return orbital + total_spin


def _remnant_tidal_factor(
    symmetric_mass_ratio: float,
    spin: float,
    lambda_value: float,
    coefficients: tuple[float, ...],
) -> float:
    p11 = coefficients[0] * spin + coefficients[1]
    p12 = coefficients[2] * spin + coefficients[3]
    p21 = coefficients[4] * spin + coefficients[5]
    p22 = coefficients[6] * spin + coefficients[7]
    p31 = coefficients[8] * spin + coefficients[9]
    p32 = coefficients[10] * spin + coefficients[11]
    p0 = (p11 + p12 * symmetric_mass_ratio) * symmetric_mass_ratio
    p1 = (p21 + p22 * symmetric_mass_ratio) * symmetric_mass_ratio
    p2 = (p31 + p32 * symmetric_mass_ratio) * symmetric_mass_ratio
    factor = (1.0 + lambda_value * p0 + lambda_value**2 * p1) / (
        1.0 + lambda_value * p2**2
    ) ** 2
    if (spin < 0.0 and symmetric_mass_ratio < 0.188) or spin < -0.5:
        factor = 1.0
    return min(factor, 1.0)


def bhns_mass_aligned(
    black_hole_mass: float,
    neutron_star_mass: float,
    black_hole_spin: float,
    neutron_star_lambda: float,
) -> float:
    """Return the aligned-spin BHNS remnant-mass fit."""

    eta = black_hole_mass * neutron_star_mass / (
        black_hole_mass + neutron_star_mass
    ) ** 2
    factor = _remnant_tidal_factor(
        eta,
        black_hole_spin,
        neutron_star_lambda,
        _MASS_FIT_COEFFICIENTS,
    )
    return factor * bbh_final_mass_non_precessing_uib2016(
        black_hole_mass, neutron_star_mass, black_hole_spin, 0.0
    )


def bhns_spin_aligned(
    black_hole_mass: float,
    neutron_star_mass: float,
    black_hole_spin: float,
    neutron_star_lambda: float,
) -> float:
    """Return the aligned-spin BHNS remnant-spin fit."""

    eta = black_hole_mass * neutron_star_mass / (
        black_hole_mass + neutron_star_mass
    ) ** 2
    factor = _remnant_tidal_factor(
        eta,
        black_hole_spin,
        neutron_star_lambda,
        _SPIN_FIT_COEFFICIENTS,
    )
    return factor * bbh_final_spin_non_precessing_uib2016(
        black_hole_mass, neutron_star_mass, black_hole_spin, 0.0
    )


def _tanh_window(value, sign: float, center: float, width: float):
    argument = 4.0 * (value - center) / width
    if isinstance(value, torch.Tensor):
        return 0.5 * (1.0 + sign * torch.tanh(argument))
    return 0.5 * (1.0 + sign * math.tanh(argument))


def seobnrv4_nsbh_amplitude(
    frequencies_hz: torch.Tensor,
    black_hole_mass: float,
    neutron_star_mass: float,
    black_hole_spin: float,
    neutron_star_lambda: float,
) -> torch.Tensor:
    """Return the multiplicative SEOBNRv4-ROM NSBH amplitude correction."""

    total_mass = black_hole_mass + neutron_star_mass
    total_mass_si = total_mass * lal.MSUN_SI
    inverse_total_mass_seconds = lal.C_SI**3 / (
        lal.G_SI * total_mass_si
    )
    eta = black_hole_mass * neutron_star_mass / total_mass**2
    mass_ratio = black_hole_mass / neutron_star_mass
    compactness = nsbh_compactness_from_lambda(neutron_star_lambda)
    neutron_star_radius = neutron_star_mass / compactness

    final_mass = bhns_mass_aligned(
        black_hole_mass,
        neutron_star_mass,
        black_hole_spin,
        neutron_star_lambda,
    )
    final_spin = bhns_spin_aligned(
        black_hole_mass,
        neutron_star_mass,
        black_hole_spin,
        neutron_star_lambda,
    )
    ringdown_frequency = (
        (1.5251 - 1.1568 * (1.0 - final_spin) ** 0.1292)
        * total_mass
        / final_mass
        / (2.0 * math.pi)
    )

    xi_tide = nsbh_xi_tide(
        mass_ratio,
        black_hole_spin,
        black_hole_mass / neutron_star_radius,
    )
    tidal_radius = (
        xi_tide * neutron_star_radius * (1.0 - 2.0 * compactness)
        + 1.0e-15
    )
    tidal_frequency = abs(
        total_mass
        / (
            math.pi
            * (
                final_spin * final_mass
                + math.sqrt(tidal_radius**3 / final_mass)
            )
        )
    )
    torus_mass = nsbh_torus_mass_fit(
        mass_ratio, black_hole_spin, compactness
    )

    frequency_ratio = (
        (tidal_frequency - ringdown_frequency) / ringdown_frequency
    )
    nondisruptive_x = (
        frequency_ratio**2
        - 0.4865330927898738 * compactness
        - 0.03143937714260868 * black_hole_spin
    )
    nondisruptive_x_prime = (
        frequency_ratio**2
        + 0.4933764101669873 * compactness
        + 0.05691547067814197 * black_hole_spin
    )
    tidal_ringdown_fraction = _tanh_window(
        nondisruptive_x,
        1.0,
        -0.09236597801342522,
        0.01871545791809104,
    )
    nondisruptive_width = 0.022500562246265655 + 2.0 * _tanh_window(
        nondisruptive_x_prime,
        -1.0,
        -0.1773927624795226,
        0.771909557448921,
    )

    disruptive_x = (
        torus_mass
        + 0.8496732940251721 * compactness
        + 0.3022694700157108 * math.sqrt(eta)
        - 0.16594256718148745 * black_hole_spin
    )
    disruptive_x_prime = (
        torus_mass
        - 0.9904717980366731 * compactness
        + 1.1227719410457802 * math.sqrt(eta)
        + 0.002986871614045452 * black_hole_spin
        - 0.07136411471590108 * black_hole_spin**2
        - 0.11261503453409044 * black_hole_spin**3
    )
    disruption_scale = 1.2728043573489636 - 1.6873457237092873 * disruptive_x
    disruptive_width = (
        0.1853261083544252
        - 0.25347578534406 * disruptive_x_prime
    )
    disruptive_frequency = disruption_scale * tidal_frequency

    if tidal_frequency >= ringdown_frequency and torus_mass == 0.0:
        center = ringdown_frequency
        width = nondisruptive_width
        ringdown_fraction = tidal_ringdown_fraction
    elif tidal_frequency < ringdown_frequency and torus_mass > 0.0:
        center = disruptive_frequency
        width = disruptive_width
        ringdown_fraction = 0.0
    elif tidal_frequency < ringdown_frequency and torus_mass == 0.0:
        center = (
            (1.0 - mass_ratio**-1) * ringdown_frequency
            + mass_ratio**-1 * disruptive_frequency
        )
        width = 0.5 * (nondisruptive_width + disruptive_width)
        ringdown_fraction = 0.0
    else:
        center = disruption_scale * ringdown_frequency
        width = nondisruptive_width
        ringdown_fraction = tidal_ringdown_fraction

    dimensionless_frequency = frequencies_hz / inverse_total_mass_seconds
    return _tanh_window(
        dimensionless_frequency, -1.0, center, width
    ) + ringdown_fraction * _tanh_window(
        dimensionless_frequency, 1.0, center, width
    )


__all__ = [
    "bbh_final_mass_non_precessing_uib2016",
    "bbh_final_spin_non_precessing_uib2016",
    "bhns_mass_aligned",
    "bhns_spin_aligned",
    "nsbh_compactness_from_lambda",
    "nsbh_r_kerr_isco",
    "nsbh_torus_mass_fit",
    "nsbh_xi_tide",
    "seobnrv4_nsbh_amplitude",
]
