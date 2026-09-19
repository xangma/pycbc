# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX-specific helpers for :mod:`pycbc.conversions`."""

import operator
import jax
import jax.numpy as jnp

from pycbc.types.backend import coerce_jax_values

_qnm_spline_cache = {}


def broadcast_values(*values):
    """Broadcast conversion inputs on the device of their first JAX array."""
    module, converted = coerce_jax_values(*values)
    if module is None:
        return None, values
    return jax, jnp.broadcast_arrays(*converted)


def qnm_spline(pykerr, spin, mode_l, mode_m, overtone, reim):
    """Evaluate a cached pykerr QNM spline on JAX backend without host transfer."""
    try:
        mode_l = operator.index(mode_l)
        mode_m = operator.index(mode_m)
        overtone = operator.index(overtone)
    except TypeError as exc:
        raise TypeError("JAX QNM mode indices must be scalar integers") from exc

    max_spin = pykerr.qnm.MAX_SPIN
    if bool(jnp.any(jnp.abs(spin) > max_spin)):
        raise ValueError(f"|spin| must be < {max_spin}")

    key = (
        reim,
        mode_l,
        abs(mode_m),
        overtone,
        spin.dtype,
    )
    try:
        knots, coefficients = _qnm_spline_cache[key]
    except KeyError:
        if reim == "re":
            cache = pykerr.qnm._reomega_splines
        else:
            cache = pykerr.qnm._imomega_splines
        spline = pykerr.qnm._getspline("omega", reim, mode_l, mode_m, overtone, cache)
        knots = jnp.asarray(spline.x, dtype=spin.dtype)
        coefficients = jnp.asarray(spline.c, dtype=spin.dtype)
        _qnm_spline_cache[key] = knots, coefficients

    points = spin
    indices = jnp.searchsorted(knots, points) - 1
    indices = jnp.clip(indices, 0, len(knots) - 2)
    offset = points - knots[indices]
    coeff = coefficients[:, indices]
    return ((coeff[0] * offset + coeff[1]) * offset + coeff[2]) * offset + coeff[3]


def real_cuberoot(value):
    """Return the real cube root without losing JAX autograd state."""
    nonzero = value != 0
    magnitude = jnp.where(nonzero, jnp.abs(value), 1.0)
    result = jnp.sign(value) * (magnitude ** (1.0 / 3.0))
    return jnp.where(nonzero, result, 0.0)


def scale_ordered_params_jax(values, bounds):
    """Scale ordered parameters on JAX backend."""
    stacked = jnp.stack(values)
    assert bool(jnp.all((stacked >= bounds[0]) & (stacked <= bounds[1]))), (
        "Input parameters lie outside of given bounds"
    )
    scaled_params = (stacked - bounds[0]) / (bounds[1] - bounds[0])
    num_params = len(values)
    idx = jnp.arange(num_params, dtype=stacked.dtype)
    exponent = 1.0 / (num_params - idx)
    exponent = exponent.reshape((num_params,) + (1,) * (stacked.ndim - 1))
    out_scaled_params = 1.0 - jnp.cumprod(
        jnp.power(1.0 - scaled_params, exponent), axis=0
    )
    out_params = out_scaled_params * (bounds[1] - bounds[0]) + bounds[0]
    if stacked.ndim == 1:
        return list(out_params)
    return out_params


def mass2_from_mchirp_mass1_jax(mchirp, mass1):
    """Secondary mass from chirp mass and primary mass in JAX."""
    if jnp.issubdtype(mchirp.dtype, jnp.complexfloating) or jnp.issubdtype(
        mass1.dtype, jnp.complexfloating
    ):
        raise TypeError("component masses and chirp mass must be real")
    if bool(jnp.any(mchirp < 0)) or bool(jnp.any(mass1 <= 0)):
        raise ValueError("chirp mass must be nonnegative and mass1 positive")

    zero_chirp = mchirp == 0
    safe_chirp = jnp.where(zero_chirp, 1.0, mchirp)
    chirp_ratio = safe_chirp / mass1
    cubic_scale = chirp_ratio**5
    half_scale = cubic_scale / 2.0
    discriminant = half_scale**2 - (cubic_scale / 3.0) ** 3

    three_real_roots = discriminant < 0
    cardano_discriminant = jnp.where(three_real_roots, 1.0, discriminant)
    cardano_half_scale = jnp.where(three_real_roots, 1.0, half_scale)
    sqrt_discriminant = jnp.sqrt(cardano_discriminant)
    cardano = real_cuberoot(
        cardano_half_scale + sqrt_discriminant
    ) + real_cuberoot(cardano_half_scale - sqrt_discriminant)

    trigonometric_scale = jnp.where(three_real_roots, cubic_scale, 1.0)
    cosine_argument = jnp.where(
        three_real_roots,
        jnp.clip(1.5 * jnp.sqrt(3.0 / trigonometric_scale), -1.0, 1.0),
        0.0,
    )
    trigonometric = (
        2.0
        * jnp.sqrt(trigonometric_scale / 3.0)
        * jnp.cos(jnp.arccos(cosine_argument) / 3.0)
    )
    mass_ratio = jnp.where(three_real_roots, trigonometric, cardano)

    # One Newton step in log(q) removes cancellation error for q << 1
    log_ratio = jnp.log(mass_ratio)
    residual = (
        3.0 * log_ratio
        - jnp.logaddexp(0.0, log_ratio)
        - 5.0 * jnp.log(chirp_ratio)
    )
    sigmoid = 1.0 / (1.0 + jnp.exp(-log_ratio))
    log_ratio = log_ratio - residual / (3.0 - sigmoid)
    result = mass1 * jnp.exp(log_ratio)
    return jnp.where(zero_chirp, 0.0, result)


def mass_from_knownmass_eta_jax(known_mass, eta, known_is_secondary=False, force_real=True):
    """Return the other component mass from one mass and eta in JAX."""
    if jnp.issubdtype(known_mass.dtype, jnp.complexfloating) or jnp.issubdtype(eta.dtype, jnp.complexfloating):
        raise TypeError("component masses and eta must be real")

    zero_eta = eta == 0
    safe_eta = jnp.where(zero_eta, 1.0, eta)
    discriminant = 1.0 - 4.0 * safe_eta

    if force_real:
        real_roots = discriminant >= 0
        sqrt_discriminant = jnp.sqrt(
            jnp.where(real_roots, discriminant, 1.0)
        )
        numerator = 1.0 - 2.0 * safe_eta + sqrt_discriminant
        root_a = known_mass * numerator / (2.0 * safe_eta)
        root_b = known_mass * (2.0 * safe_eta) / numerator
        repeated_root = known_mass * (1.0 - 2.0 * safe_eta) / (2.0 * safe_eta)
        root_a = jnp.where(real_roots, root_a, repeated_root)
        root_b = jnp.where(real_roots, root_b, repeated_root)
    else:
        complex_dtype = (
            jnp.complex64
            if known_mass.dtype in (jnp.float16, jnp.bfloat16, jnp.float32)
            else jnp.complex128
        )
        complex_mass = known_mass.astype(complex_dtype)
        complex_eta = safe_eta.astype(complex_dtype)
        sqrt_discriminant = jnp.sqrt(1.0 - 4.0 * complex_eta)
        root_a = (
            complex_mass * (1.0 - 2.0 * complex_eta + sqrt_discriminant) / (2.0 * complex_eta)
        )
        root_b = (
            complex_mass * (1.0 - 2.0 * complex_eta - sqrt_discriminant) / (2.0 * complex_eta)
        )

    root_a_is_larger = (
        (jnp.real(root_a) > jnp.real(root_b))
        | ((jnp.real(root_a) == jnp.real(root_b)) & (jnp.imag(root_a) > jnp.imag(root_b)))
        if jnp.issubdtype(root_a.dtype, jnp.complexfloating)
        else root_a > root_b
    )
    larger = jnp.where(root_a_is_larger, root_a, root_b)
    smaller = jnp.where(root_a_is_larger, root_b, root_a)
    result = larger if known_is_secondary else smaller
    return jnp.where(zero_eta, 0.0, result)


def lambda_from_tov_jax(mass, distance, mass_from_file, lambda_from_file, redshift_func):
    """Interpolate tidal deformability from TOV data in JAX."""
    if jnp.issubdtype(mass.dtype, jnp.complexfloating) or jnp.issubdtype(
        distance.dtype, jnp.complexfloating
    ):
        raise TypeError("mass and distance must be real-valued")
    mass_src = mass / (1.0 + redshift_func(distance))
    mass_knots = jnp.asarray(mass_from_file, dtype=mass.dtype)
    lambda_knots = jnp.asarray(lambda_from_file, dtype=mass.dtype)
    if mass_knots.size == 1:
        return jnp.full_like(mass_src, lambda_knots[0])
    flat = mass_src.reshape(-1)
    res = jnp.interp(
        flat, mass_knots, lambda_knots, left=lambda_knots[0], right=lambda_knots[-1]
    )
    res = jnp.where(jnp.isnan(flat), flat, res)
    return res.reshape(mass_src.shape)
