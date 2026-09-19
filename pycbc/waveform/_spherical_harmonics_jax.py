# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""JAX evaluation of spin-weighted spherical harmonics."""

import math
from numbers import Integral

import jax.numpy as jnp
from pycbc.types.array_jax import _ensure_x64

_SPIN_MINUS_TWO_MODES = tuple(
    (ell, emm) for ell in range(2, 5) for emm in range(-ell, ell + 1)
)


def _spin_minus_two_terms():
    """Precompute scalar-equivalent Wigner-sum terms for ell 2 through 4."""
    terms = []
    spin_weight = -2
    wigner_m = -spin_weight
    for ell, emm in _SPIN_MINUS_TWO_MODES:
        prefactor = (-1) ** spin_weight * math.sqrt(
            (2 * ell + 1)
            / (4 * math.pi)
            * math.factorial(ell + wigner_m)
            * math.factorial(ell - wigner_m)
            * math.factorial(ell + emm)
            * math.factorial(ell - emm)
        )
        mode_terms = []
        for index in range(2 * ell + 1):
            denominator_indices = (
                ell + wigner_m - index,
                index,
                emm - wigner_m + index,
                ell - emm - index,
            )
            if min(denominator_indices) < 0:
                continue
            denominator = math.prod(
                math.factorial(value) for value in denominator_indices
            )
            coefficient = (-1) ** (emm - wigner_m + index) * prefactor / denominator
            mode_terms.append(
                (
                    coefficient,
                    2 * ell + wigner_m - emm - 2 * index,
                    emm - wigner_m + 2 * index,
                )
            )
        terms.append((ell, emm, tuple(mode_terms)))
    return tuple(terms)


_SPIN_MINUS_TWO_TERMS = {
    (ell, emm): terms for ell, emm, terms in _spin_minus_two_terms()
}


def spin_weighted_spherical_harmonic(
    theta,
    phi,
    spin_weight,
    ell,
    emm,
    *,
    dtype=jnp.float64,
):
    r"""Evaluate :math:`{}_{s}Y_{\ell m}(\theta, \phi)` with JAX.

    The finite Wigner-:math:`d` sum follows the convention used by
    ``lal.SpinWeightedSphericalHarmonic``.
    """
    _ensure_x64()
    indices = (spin_weight, ell, emm)
    if any(not isinstance(index, Integral) for index in indices):
        raise TypeError("spin weight, ell, and m must be integers")
    spin_weight, ell, emm = map(int, indices)
    if ell < 0 or abs(spin_weight) > ell or abs(emm) > ell:
        raise ValueError("spin weight and m must have magnitude at most ell")

    theta_j = jnp.asarray(theta, dtype=dtype)
    phi_j = jnp.asarray(phi, dtype=dtype)
    theta_j, phi_j = jnp.broadcast_arrays(theta_j, phi_j)

    wigner_m = -spin_weight
    prefactor = (-1) ** spin_weight * math.sqrt(
        (2 * ell + 1)
        / (4 * math.pi)
        * math.factorial(ell + wigner_m)
        * math.factorial(ell - wigner_m)
        * math.factorial(ell + emm)
        * math.factorial(ell - emm)
    )
    cos_half = jnp.cos(0.5 * theta_j)
    sin_half = jnp.sin(0.5 * theta_j)
    amplitude = jnp.zeros_like(theta_j)

    for index in range(2 * ell + 1):
        denominator_indices = (
            ell + wigner_m - index,
            index,
            emm - wigner_m + index,
            ell - emm - index,
        )
        if min(denominator_indices) < 0:
            continue
        denominator = math.prod(
            math.factorial(value) for value in denominator_indices
        )
        coefficient = (-1) ** (emm - wigner_m + index) * prefactor / denominator
        cos_power = 2 * ell + wigner_m - emm - 2 * index
        sin_power = emm - wigner_m + 2 * index
        amplitude = amplitude + coefficient * (cos_half ** cos_power) * (sin_half ** sin_power)

    phase = emm * phi_j
    c_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    return (amplitude * jnp.cos(phase) + 1j * amplitude * jnp.sin(phase)).astype(c_dtype)


def selected_spin_minus_two_spherical_harmonics(
    theta,
    phi,
    modes,
    *,
    dtype=jnp.float64,
):
    """Return dictionary mapping (ell, emm) modes to evaluated JAX spherical harmonics."""
    result = {}
    for ell, emm in modes:
        result[(ell, emm)] = spin_weighted_spherical_harmonic(
            theta, phi, -2, ell, emm, dtype=dtype
        )
    return result
