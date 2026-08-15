# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch implementation of the NRTidal v1 and v2 corrections.

The frequency-dependent phase, amplitude, and merger taper follow
``LALSimNRTunedTides.c`` from LALSuite 7.26.1.  Scalar parameter validation
and the merger-frequency fit remain on the host; all array evaluation stays
on the Torch device of the supplied frequency tensor.
"""

from __future__ import annotations

import math

import lal
import torch


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
NRTIDAL_APPROXIMANTS = NRTIDAL_V1_APPROXIMANTS | NRTIDAL_V2_APPROXIMANTS


def nrtidal_version(approximant: str) -> int | None:
    """Return the NRTidal generation number for a supported approximant."""

    if approximant in NRTIDAL_V1_APPROXIMANTS:
        return 1
    if approximant in NRTIDAL_V2_APPROXIMANTS:
        return 2
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
    """Evaluate the NRTidalv2 3.5PN matter-spin phase correction."""

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


def nrtidal_phase(
    frequencies: torch.Tensor,
    mass1: float,
    mass2: float,
    lambda1: float,
    lambda2: float,
    version: int,
) -> torch.Tensor:
    """Evaluate the NRTidal v1 or v2 Fourier-phase correction."""

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
    """Evaluate the additive NRTidal v2 amplitude correction."""

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return phase, taper, and optional v2 amplitude corrections."""

    merger_frequency = nrtidal_merger_frequency(mass1, mass2, lambda1, lambda2)
    phase = nrtidal_phase(frequencies, mass1, mass2, lambda1, lambda2, version)
    taper = nrtidal_taper(frequencies, merger_frequency)
    amplitude = None
    if version == 2:
        amplitude = nrtidal_amplitude(frequencies, mass1, mass2, lambda1, lambda2)
    return phase, taper, amplitude


__all__ = [
    "NRTIDAL_APPROXIMANTS",
    "NRTIDAL_V1_APPROXIMANTS",
    "NRTIDAL_V2_APPROXIMANTS",
    "nrtidal_amplitude",
    "nrtidal_corrections",
    "nrtidal_higher_order_spin_phase",
    "nrtidal_higher_order_spin_terms",
    "nrtidal_kappa2t",
    "nrtidal_merger_frequency",
    "nrtidal_octupole_from_quadrupole",
    "nrtidal_phase",
    "nrtidal_quadrupole_from_lambda",
    "nrtidal_self_spin_phase",
    "nrtidal_taper",
    "nrtidal_version",
]
