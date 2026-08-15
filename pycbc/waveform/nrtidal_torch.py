# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch implementation of the NRTidal v1, v2, and v3 corrections.

The frequency-dependent phase, amplitude, and merger taper follow
``LALSimNRTunedTides.c`` from LALSuite 7.26.1.  Scalar parameter validation
and the merger-frequency fit remain on the host; all array evaluation stays
on the Torch device of the supplied frequency tensor.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import lal
import torch


class _NRTidalV3PNCoefficients(NamedTuple):
    newton_a: float
    one_a: float
    three_halves_a: float
    two_a: float
    five_halves_a: float
    newton_b: float
    one_b: float
    three_halves_b: float
    two_b: float
    five_halves_b: float


class _NRTidalV3Coefficients(NamedTuple):
    s1: float
    s2: float
    exp_s2s3: float
    kappa_a: float
    kappa_b: float
    five_halves_a: float
    three_a: float
    denominator_one_a: float
    five_halves_b: float
    three_b: float
    denominator_one_b: float
    one_a: float
    three_halves_a: float
    two_a: float
    denominator_three_halves_a: float
    one_b: float
    three_halves_b: float
    two_b: float
    denominator_three_halves_b: float


NRTIDAL_V1_APPROXIMANTS = frozenset(
    {
        "IMRPhenomD_NRTidal",
        "SEOBNRv4_ROM_NRTidal",
    }
)
NRTIDAL_V2_APPROXIMANTS = frozenset(
    {
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS_NRTidalv2",
        "SEOBNRv4_ROM_NRTidalv2",
    }
)
NRTIDAL_V3_APPROXIMANTS = frozenset({"IMRPhenomXAS_NRTidalv3"})
NRTIDAL_APPROXIMANTS = (
    NRTIDAL_V1_APPROXIMANTS | NRTIDAL_V2_APPROXIMANTS | NRTIDAL_V3_APPROXIMANTS
)


def nrtidal_version(approximant: str) -> int | None:
    """Return the NRTidal generation number for a supported approximant."""

    if approximant in NRTIDAL_V1_APPROXIMANTS:
        return 1
    if approximant in NRTIDAL_V2_APPROXIMANTS:
        return 2
    if approximant in NRTIDAL_V3_APPROXIMANTS:
        return 3
    return None


def _ordered_matter_parameters(
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
) -> tuple[float, float, float, float]:
    values = (mass1, mass2, lambda1, lambda2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NRTidal masses and deformabilities must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("NRTidal component masses must be positive")
    if lambda1 < 0.0 or lambda2 < 0.0:
        raise ValueError("NRTidal deformabilities must be non-negative")
    if mass2 > mass1:
        return mass2, mass1, lambda2, lambda1
    return mass1, mass2, lambda1, lambda2


def nrtidal_kappa2t(
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
) -> float:
    """Return the effective quadrupolar tidal coupling constant."""

    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    total_mass = mass1 + mass2
    xa = mass1 / total_mass
    xb = mass2 / total_mass
    term1 = (1.0 + 12.0 * xb / xa) * xa**5 * lambda1
    term2 = (1.0 + 12.0 * xa / xb) * xb**5 * lambda2
    return (3.0 / 13.0) * (term1 + term2)


def nrtidal_merger_frequency(
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
) -> float:
    """Return the NRTidal v1/v2 merger frequency in hertz."""

    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    kappa2t = nrtidal_kappa2t(mass1, mass2, lambda1, lambda2)
    kappa2t_sq = kappa2t * kappa2t
    numerator = 1.0 + 3.35411203e-2 * kappa2t + 4.31460284e-5 * kappa2t_sq
    denominator = 1.0 + 7.54224145e-2 * kappa2t + 2.23626859e-4 * kappa2t_sq
    dimensionless_omega = 0.3586 / math.sqrt(mass1 / mass2) * numerator / denominator
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    return dimensionless_omega / (2.0 * math.pi * total_mass_seconds)


def nrtidal_merger_frequency_v3(
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
    spin1z: float,
    spin2z: float,
) -> float:
    """Return the spin-dependent NRTidal v3 merger frequency in hertz."""

    if not math.isfinite(spin1z) or not math.isfinite(spin2z):
        raise ValueError("NRTidal aligned spins must be finite")
    swap = mass2 > mass1
    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    if swap:
        spin1z, spin2z = spin2z, spin1z

    total_mass = mass1 + mass2
    xa = mass1 / total_mass
    xb = mass2 / total_mass
    symmetric_mass_ratio = xa * xb
    kappa2eff = 3.0 * symmetric_mass_ratio * (xa**3 * lambda1 + xb**3 * lambda2)
    mass_asymmetry = 1.0 - 4.0 * symmetric_mass_ratio

    spin_coefficient = 0.25 * (1.0 - 1.99 * mass_asymmetry)
    tidal_coefficients = (
        0.0485 * (1.0 + 1.80 * mass_asymmetry),
        5.86e-6 * (1.0 + 599.99 * mass_asymmetry),
        0.10 * (1.0 + 7.80 * mass_asymmetry),
        1.86e-4 * (1.0 + 84.76 * mass_asymmetry),
    )
    weighted_spin = xa**2 * spin1z + xb**2 * spin2z
    mass_fit = 1.0 + 0.80 * mass_asymmetry
    spin_fit = 1.0 + spin_coefficient * weighted_spin
    kappa2eff_sq = kappa2eff * kappa2eff
    tidal_fit = (
        1.0 + tidal_coefficients[0] * kappa2eff + tidal_coefficients[1] * kappa2eff_sq
    ) / (1.0 + tidal_coefficients[2] * kappa2eff + tidal_coefficients[3] * kappa2eff_sq)
    dimensionless_frequency = (
        0.22 * symmetric_mass_ratio * mass_fit * spin_fit * tidal_fit
    )
    return dimensionless_frequency / (total_mass * lal.MTSUN_SI)


def nrtidal_quadrupole_from_lambda(lambda_value: float) -> float:
    """Return the full quadrupole used by LAL's NRTidal dispatch.

    This mirrors ``XLALSimInspiralEOSQfromLambda``.  It is deliberately not
    the newer fit in ``XLALSimUniversalRelationQuadMonVSlambda2Tidal``:
    LAL's legacy IMRPhenomD/SEOBNRv4 NRTidal dispatch still uses the former
    when it fills otherwise-default quadrupole-monopole parameters.
    """

    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("tidal deformability must be finite and non-negative")
    if lambda_value < 0.5:
        return 1.0
    log_lambda = math.log(lambda_value)
    return math.exp(
        0.1940
        + 0.0936 * log_lambda
        + 0.0474 * log_lambda**2
        - 0.00421 * log_lambda**3
        + 0.000123 * log_lambda**4
    )


def nrtidal_octupole_from_quadrupole(quadrupole: float) -> float:
    """Return LAL's full spin-induced octupole coefficient."""

    if not math.isfinite(quadrupole) or quadrupole <= 0.0:
        raise ValueError("spin-induced quadrupole must be finite and positive")
    log_quadrupole = math.log(quadrupole)
    return math.exp(
        0.003131
        + 2.071 * log_quadrupole
        - 0.7152 * log_quadrupole**2
        + 0.2458 * log_quadrupole**3
        - 0.03309 * log_quadrupole**4
    )


def nrtidal_higher_order_spin_terms(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    quadrupole1: float,
    quadrupole2: float,
) -> tuple[float, float]:
    """Return LAL's 3.5PN spin-spin and spin-cubed coefficients."""

    values = (mass1, mass2, spin1z, spin2z, quadrupole1, quadrupole2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("higher-order spin parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")
    if quadrupole1 <= 0.0 or quadrupole2 <= 0.0:
        raise ValueError("spin-induced quadrupoles must be positive")

    total_mass = mass1 + mass2
    xa = mass1 / total_mass
    xb = mass2 / total_mass
    xa2 = xa * xa
    xb2 = xb * xb
    octupole1 = nrtidal_octupole_from_quadrupole(quadrupole1) - 1.0
    octupole2 = nrtidal_octupole_from_quadrupole(quadrupole2) - 1.0

    spin_spin = (
        -400.0 * math.pi * (quadrupole1 - 1.0) * spin1z**2 * xa2
        - 400.0 * math.pi * (quadrupole2 - 1.0) * spin2z**2 * xb2
    )
    spin_cubed = (
        10.0
        * ((xa2 + 308.0 / 3.0 * xa) * spin1z + (xb2 - 89.0 / 3.0 * xb) * spin2z)
        * (quadrupole1 - 1.0)
        * xa2
        * spin1z**2
        + 10.0
        * ((xb2 + 308.0 / 3.0 * xb) * spin2z + (xa2 - 89.0 / 3.0 * xa) * spin1z)
        * (quadrupole2 - 1.0)
        * xb2
        * spin2z**2
        - 440.0 * octupole1 * xa * xa2 * spin1z**3
        - 440.0 * octupole2 * xb * xb2 * spin2z**3
    )
    return spin_spin, spin_cubed


def nrtidal_higher_order_spin_phase(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    quadrupole1: float,
    quadrupole2: float,
) -> torch.Tensor:
    """Evaluate the NRTidal v2/v3 3.5PN matter-spin phase correction."""

    spin_spin, spin_cubed = nrtidal_higher_order_spin_terms(
        mass1,
        mass2,
        spin1z,
        spin2z,
        quadrupole1,
        quadrupole2,
    )
    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    velocity_squared = (math.pi * total_mass * lal.MTSUN_SI * frequencies).pow(
        2.0 / 3.0
    )
    return 3.0 / (128.0 * eta) * (spin_spin + spin_cubed) * velocity_squared


def nrtidal_self_spin_phase(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    dquad1: float,
    dquad2: float,
) -> torch.Tensor:
    """Evaluate SEOBNRv4 NRTidal's 2PN/3PN matter-spin correction."""

    values = (mass1, mass2, spin1z, spin2z, dquad1, dquad2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("self-spin parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("component masses must be positive")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    xa = mass1 / total_mass
    xb = mass2 / total_mass
    xa2 = xa * xa
    xb2 = xb * xb
    pn_sigma = -50.0 * (dquad1 * spin1z**2 * xa2 + dquad2 * spin2z**2 * xb2)
    pn_ss3 = (
        5.0 / 84.0 * (9407.0 + 8218.0 * xa - 2016.0 * xa2) * dquad1 * xa2 * spin1z**2
        + 5.0 / 84.0 * (9407.0 + 8218.0 * xb - 2016.0 * xb2) * dquad2 * xb2 * spin2z**2
    )
    normalization = 3.0 / (128.0 * eta)
    velocity = torch.pow(math.pi * total_mass * lal.MTSUN_SI * frequencies, 1.0 / 3.0)
    return normalization * (pn_sigma / velocity + pn_ss3 * velocity)


def nrtidal_taper(
    frequencies: torch.Tensor,
    merger_frequency: float,
) -> torch.Tensor:
    """Return one minus LAL's Planck taper at the supplied frequencies."""

    taper_end = 1.2 * merger_frequency
    interior = (frequencies > merger_frequency) & (frequencies < taper_end)
    midpoint = 0.5 * (merger_frequency + taper_end)
    safe_frequencies = torch.where(
        interior,
        frequencies,
        torch.as_tensor(midpoint, dtype=frequencies.dtype, device=frequencies.device),
    )
    width = taper_end - merger_frequency
    exponent = width / (safe_frequencies - merger_frequency) + width / (
        safe_frequencies - taper_end
    )
    # Keep the literal LAL expression. Besides numerical parity in the
    # transition, its floating-point cancellation gives the same exact-zero
    # bins near the lower end of the taper.
    tapered = 1.0 - 1.0 / (torch.exp(exponent) + 1.0)
    return torch.where(
        frequencies <= merger_frequency,
        torch.ones_like(frequencies),
        torch.where(
            frequencies >= taper_end,
            torch.zeros_like(frequencies),
            tapered,
        ),
    )


def _nrtidal_v3_pn_coefficients(xa: float) -> _NRTidalV3PNCoefficients:
    """Return the body-resolved 7.5PN tidal coefficients."""

    xb = 1.0 - xa

    def coefficients(mass_fraction, companion_fraction):
        fraction_sq = mass_fraction * mass_fraction
        fraction_cu = fraction_sq * mass_fraction
        fraction_fo = fraction_cu * mass_fraction
        fraction_fi = fraction_fo * mass_fraction
        denominator = 11.0 * mass_fraction - 12.0
        newton = -3.0 * denominator / (16.0 * mass_fraction * companion_fraction**2)
        one = (
            -1300.0 * fraction_cu
            + 11430.0 * fraction_sq
            + 4595.0 * mass_fraction
            - 15895.0
        ) / (672.0 * denominator)
        three_halves = -math.pi
        two = (
            22861440.0 * fraction_fi
            - 102135600.0 * fraction_fo
            + 791891100.0 * fraction_cu
            + 874828080.0 * fraction_sq
            + 216234195.0 * mass_fraction
            - 1939869350.0
        ) / (27433728.0 * denominator)
        five_halves = (
            -math.pi
            * (
                10520.0 * fraction_cu
                - 7598.0 * fraction_sq
                + 22415.0 * mass_fraction
                - 27719.0
            )
            / (672.0 * denominator)
        )
        return newton, one, three_halves, two, five_halves

    return _NRTidalV3PNCoefficients(
        *coefficients(xa, xb),
        *coefficients(xb, xa),
    )


def _nrtidal_v3_coefficients(
    xa: float,
    lambda1: float,
    lambda2: float,
    kappa2t: float,
    pn: _NRTidalV3PNCoefficients,
) -> _NRTidalV3Coefficients:
    """Return NRTidal v3's dynamical-tide and constrained Padé fits."""

    xb = 1.0 - xa
    mass_ratio = xa / xb
    s1 = 1.273000423 + 3.64169971e-3 * kappa2t
    s1 += 1.76144380e-3 * mass_ratio * kappa2t
    s2 = 27.8793291 + 1.18175396e-2 * kappa2t
    s2 -= 5.39996790e-3 * mass_ratio * kappa2t
    s3 = 0.142449682 - 1.70505852e-5 * kappa2t
    s3 += 3.38040594e-5 * mass_ratio * kappa2t
    exp_s2s3 = math.cosh(s2 * s3) + math.sinh(s2 * s3)

    kappa_a = 3.0 * xb * xa**4 * lambda1
    kappa_b = 3.0 * xa * xb**4 * lambda2

    def fitted_coefficients(mass_fraction, kappa, pn_coefficients):
        kappa_alpha = (kappa + 1.0) ** -8.08155404e-3
        fraction_beta = mass_fraction**-1.13695919
        five_halves = (
            -940.654388
            + 626.517157 * mass_fraction
            + 553.629706 * kappa_alpha
            + 88.4823087 * fraction_beta
        )
        three = (
            405.483848
            - 425.525054 * mass_fraction
            - 192.004957 * kappa_alpha
            - 51.0967553 * fraction_beta
        )
        denominator_one = (
            3.80343306 - 25.2026996 * mass_fraction - 3.08054443 * fraction_beta
        )
        _, pn_one, pn_three_halves, pn_two, pn_five_halves = pn_coefficients
        one = pn_one + denominator_one
        three_halves = (
            pn_one * pn_three_halves
            - pn_five_halves
            - pn_three_halves * denominator_one
            + five_halves
        ) / pn_one
        two = pn_two + pn_one * denominator_one
        denominator_three_halves = (
            -(pn_five_halves + pn_three_halves * denominator_one - five_halves) / pn_one
        )
        return (
            five_halves,
            three,
            denominator_one,
            one,
            three_halves,
            two,
            denominator_three_halves,
        )

    fit_a = fitted_coefficients(xa, kappa_a, pn[:5])
    fit_b = fitted_coefficients(xb, kappa_b, pn[5:])
    return _NRTidalV3Coefficients(
        s1,
        s2,
        exp_s2s3,
        kappa_a,
        kappa_b,
        fit_a[0],
        fit_a[1],
        fit_a[2],
        fit_b[0],
        fit_b[1],
        fit_b[2],
        fit_a[3],
        fit_a[4],
        fit_a[5],
        fit_a[6],
        fit_b[3],
        fit_b[4],
        fit_b[5],
        fit_b[6],
    )


def _nrtidal_v3_phase_components(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fitted NRTidal v3 and post-merger PN phases."""

    total_mass = mass1 + mass2
    xa = mass1 / total_mass
    kappa2t = nrtidal_kappa2t(mass1, mass2, lambda1, lambda2)
    pn = _nrtidal_v3_pn_coefficients(xa)
    fit = _nrtidal_v3_coefficients(xa, lambda1, lambda2, kappa2t, pn)

    angular_frequency = math.pi * total_mass * lal.MTSUN_SI * frequencies
    x = angular_frequency / torch.pow(angular_frequency, 1.0 / 3.0)
    x_two = x * x
    x_three = x_two * x
    x_three_halves = x * torch.sqrt(x)
    x_five_halves = x_three_halves * x

    exp_s2_frequency = torch.cosh(-2.0 * fit.s2 * angular_frequency)
    exp_s2_frequency += torch.sinh(-2.0 * fit.s2 * angular_frequency)
    enhancement = (
        1.0
        + (fit.s1 - 1.0) / (1.0 + exp_s2_frequency * fit.exp_s2s3)
        - (fit.s1 - 1.0) / (1.0 + fit.exp_s2s3)
        - 2.0
        * angular_frequency
        * (fit.s1 - 1.0)
        * fit.s2
        * fit.exp_s2s3
        / (1.0 + fit.exp_s2s3) ** 2
    )

    def fitted_phase(
        newton,
        kappa,
        one,
        three_halves,
        two,
        five_halves,
        three,
        denominator_one,
        denominator_three_halves,
    ):
        numerator = (
            1.0
            + one * x
            + three_halves * x_three_halves
            + two * x_two
            + five_halves * x_five_halves
            + three * x_three
        )
        denominator = (
            1.0 + denominator_one * x + denominator_three_halves * x_three_halves
        )
        return -newton * x_five_halves * kappa * enhancement * numerator / denominator

    fitted = fitted_phase(
        pn.newton_a,
        fit.kappa_a,
        fit.one_a,
        fit.three_halves_a,
        fit.two_a,
        fit.five_halves_a,
        fit.three_a,
        fit.denominator_one_a,
        fit.denominator_three_halves_a,
    )
    fitted += fitted_phase(
        pn.newton_b,
        fit.kappa_b,
        fit.one_b,
        fit.three_halves_b,
        fit.two_b,
        fit.five_halves_b,
        fit.three_b,
        fit.denominator_one_b,
        fit.denominator_three_halves_b,
    )

    def pn_phase(newton, kappa, one, three_halves, two, five_halves):
        polynomial = (
            1.0
            + one * x
            + three_halves * x_three_halves
            + two * x_two
            + five_halves * x_five_halves
        )
        return -newton * kappa * x_five_halves * polynomial

    post_merger = pn_phase(
        pn.newton_a,
        fit.kappa_a,
        pn.one_a,
        pn.three_halves_a,
        pn.two_a,
        pn.five_halves_a,
    )
    post_merger += pn_phase(
        pn.newton_b,
        fit.kappa_b,
        pn.one_b,
        pn.three_halves_b,
        pn.two_b,
        pn.five_halves_b,
    )
    return fitted, post_merger


def _nrtidal_v3_transition(
    frequencies: torch.Tensor,
    merger_frequency: float,
    *,
    hyperbolic_exponential: bool,
) -> torch.Tensor:
    """Return the v3 fitted-to-PN Planck transition."""

    start = 1.15 * merger_frequency
    end = 1.35 * merger_frequency
    interior = (frequencies > start) & (frequencies < end)
    safe_frequencies = torch.where(
        interior,
        frequencies,
        torch.as_tensor(
            0.5 * (start + end),
            dtype=frequencies.dtype,
            device=frequencies.device,
        ),
    )
    width = end - start
    exponent = width / (safe_frequencies - start)
    exponent += width / (safe_frequencies - end)
    if hyperbolic_exponential:
        exponential = torch.cosh(exponent) + torch.sinh(exponent)
    else:
        exponential = torch.exp(exponent)
    transition = 1.0 / (exponential + 1.0)
    return torch.where(
        frequencies <= start,
        torch.zeros_like(frequencies),
        torch.where(frequencies >= end, torch.ones_like(frequencies), transition),
    )


def _clamp_nrtidal_v3_minimum(
    frequencies: torch.Tensor,
    phase: torch.Tensor,
    merger_frequency: float,
) -> torch.Tensor:
    """Hold the sampled v3 phase at its first post-merger minimum."""

    if frequencies.ndim != 1 or frequencies.shape[0] < 2:
        return phase
    length = frequencies.shape[0]
    candidate = (frequencies[1:] >= 0.9 * merger_frequency) & (phase[1:] >= phase[:-1])
    positions = torch.arange(length, device=frequencies.device)
    left_positions = positions[:-1]
    sentinel = torch.full_like(left_positions, length - 1)
    first_minimum = torch.where(candidate, left_positions, sentinel).amin()
    minimum_value = phase.gather(0, first_minimum.reshape(1)).squeeze(0)
    return torch.where(positions > first_minimum, minimum_value, phase)


def nrtidal_v3_phase(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
    spin1z: float,
    spin2z: float,
    *,
    frequency_series: bool = True,
) -> torch.Tensor:
    """Evaluate the NRTidal v3 Fourier-phase correction.

    ``frequency_series`` enables LAL's sampled first-minimum clamp. Scalar
    reference and alignment evaluations instead use the smooth analytic phase
    employed by IMRPhenomX's internal phase helpers.
    """

    if not math.isfinite(spin1z) or not math.isfinite(spin2z):
        raise ValueError("NRTidal aligned spins must be finite")
    swap = mass2 > mass1
    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    if swap:
        spin1z, spin2z = spin2z, spin1z
    merger_frequency = nrtidal_merger_frequency_v3(
        mass1,
        mass2,
        lambda1,
        lambda2,
        spin1z,
        spin2z,
    )
    fitted, post_merger = _nrtidal_v3_phase_components(
        frequencies,
        mass1,
        mass2,
        lambda1,
        lambda2,
    )
    if frequency_series:
        fitted = _clamp_nrtidal_v3_minimum(
            frequencies,
            fitted,
            merger_frequency,
        )
    transition = _nrtidal_v3_transition(
        frequencies,
        merger_frequency,
        hyperbolic_exponential=not frequency_series,
    )
    return fitted * (1.0 - transition) + post_merger * transition


def nrtidal_phase(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
    version: int,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    *,
    frequency_series: bool = True,
) -> torch.Tensor:
    """Evaluate an NRTidal Fourier-phase correction."""

    if version == 3:
        return nrtidal_v3_phase(
            frequencies,
            mass1,
            mass2,
            lambda1,
            lambda2,
            spin1z,
            spin2z,
            frequency_series=frequency_series,
        )

    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    total_mass = mass1 + mass2
    xa = mass1 / total_mass
    xb = mass2 / total_mass
    kappa2t = nrtidal_kappa2t(mass1, mass2, lambda1, lambda2)
    x = (math.pi * total_mass * lal.MTSUN_SI * frequencies).pow(2.0 / 3.0)
    x_three_halves = x * torch.sqrt(x)
    x_two = x * x
    x_five_halves = x_two * torch.sqrt(x)

    if version == 1:
        numerator = (
            1.0
            - 17.428 * x
            + 31.867 * x_three_halves
            - 26.414 * x_two
            + 62.362 * x_five_halves
        )
        denominator = 1.0 + (-17.428 - 2.496) * x + 36.089 * x_three_halves
    elif version == 2:
        x_three = x_two * x
        numerator = (
            1.0
            - 12.615214237993088 * x
            + 19.0537346970349 * x_three_halves
            - 21.166863146081035 * x_two
            + 90.55082156324926 * x_five_halves
            - 60.25357801943598 * x_three
        )
        denominator = (
            1.0
            - 15.111207827736678 * x
            + 22.195327350624694 * x_three_halves
            + 8.064109635305156 * x_two
        )
    else:
        raise ValueError(f"unsupported NRTidal version: {version}")

    return -kappa2t * 2.4375 / (xa * xb) * x_five_halves * numerator / denominator


def nrtidal_amplitude(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
) -> torch.Tensor:
    """Evaluate the additive NRTidal v2/v3 amplitude correction."""

    mass1, mass2, lambda1, lambda2 = _ordered_matter_parameters(
        mass1, mass2, lambda1, lambda2
    )
    total_mass_seconds = (mass1 + mass2) * lal.MTSUN_SI
    kappa2t = nrtidal_kappa2t(mass1, mass2, lambda1, lambda2)
    x = (math.pi * total_mass_seconds * frequencies).pow(2.0 / 3.0)
    polynomial = (1.0 + 4.157407407407407 * x + 2519.111111111111 * x.pow(2.89)) / (
        1.0 + 13477.8073677 * x.pow(4.0)
    )
    return -9.0 * kappa2t * x.pow(3.25) * polynomial


def nrtidal_corrections(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
    version: int,
    *,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return phase, taper, and optional v2/v3 amplitude corrections."""

    if version == 3:
        merger_frequency = nrtidal_merger_frequency_v3(
            mass1,
            mass2,
            lambda1,
            lambda2,
            spin1z,
            spin2z,
        )
    else:
        merger_frequency = nrtidal_merger_frequency(
            mass1,
            mass2,
            lambda1,
            lambda2,
        )
    phase = nrtidal_phase(
        frequencies,
        mass1,
        mass2,
        lambda1,
        lambda2,
        version,
        spin1z,
        spin2z,
    )
    taper = nrtidal_taper(frequencies, merger_frequency)
    amplitude = None
    if version in (2, 3):
        amplitude = nrtidal_amplitude(frequencies, mass1, mass2, lambda1, lambda2)
    return phase, taper, amplitude


__all__ = [
    "NRTIDAL_APPROXIMANTS",
    "NRTIDAL_V1_APPROXIMANTS",
    "NRTIDAL_V2_APPROXIMANTS",
    "NRTIDAL_V3_APPROXIMANTS",
    "nrtidal_amplitude",
    "nrtidal_corrections",
    "nrtidal_higher_order_spin_phase",
    "nrtidal_higher_order_spin_terms",
    "nrtidal_kappa2t",
    "nrtidal_merger_frequency",
    "nrtidal_merger_frequency_v3",
    "nrtidal_octupole_from_quadrupole",
    "nrtidal_phase",
    "nrtidal_quadrupole_from_lambda",
    "nrtidal_self_spin_phase",
    "nrtidal_taper",
    "nrtidal_version",
    "nrtidal_v3_phase",
]
