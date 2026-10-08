# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused JAX compatibility tests for scientific helper modules."""

import numpy

import pytest

from pycbc import boundaries, conversions

from pycbc.coordinates import base as coordinates

jax = pytest.importorskip("jax")

jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

def test_boundary_conditioning_preserves_tensor_and_gradient():
    values = jnp.array(
        [-10.0, -4.0, -1.0, 0.25, 1.0, 3.0, 8.0],
        dtype=jnp.float64,
    )
    bounds = (
        boundaries.Bounds(
            -1.0, 1.0, btype_min="reflected", btype_max="reflected"
        ),
        boundaries.Bounds(-1.0, 1.0, btype_min="reflected"),
        boundaries.Bounds(-1.0, 1.0, btype_max="reflected"),
        boundaries.Bounds(-1.0, 1.0, cyclic=True),
    )
    expected = (
        [0.0, 0.0, -1.0, 0.25, 1.0, -1.0, 0.0],
        [8.0, 2.0, -1.0, 0.25, 1.0, 3.0, 8.0],
        [-10.0, -4.0, -1.0, 0.25, 1.0, -1.0, -6.0],
        [0.0, 0.0, -1.0, 0.25, -1.0, -1.0, 0.0],
    )

    actual = tuple(bound.apply_conditions(values) for bound in bounds)

    for result, target in zip(actual, expected):
        assert isinstance(result, (jnp.ndarray, jax.Array))
        assert result.dtype == values.dtype
        numpy.testing.assert_allclose(result, target)

    def total_loss(v):
        return sum(jnp.sum(bound.apply_conditions(v)) for bound in bounds)

    grad = jax.grad(total_loss)(values)
    assert bool(jnp.all(jnp.isfinite(grad)))

def test_coordinate_roundtrip_preserves_tensor_and_gradient():
    original = (
        jnp.array([0.75, -1.25, 0.4], dtype=jnp.float64),
        jnp.array([-0.5, 0.8, 1.1], dtype=jnp.float64),
        jnp.array([1.2, -0.3, 0.65], dtype=jnp.float64),
    )

    rho, phi, theta = coordinates.cartesian_to_spherical(*original)
    roundtrip = coordinates.spherical_to_cartesian(rho, phi, theta)
    origin_theta = coordinates.cartesian_to_spherical_polar(
        jnp.zeros(3, dtype=jnp.float64), 0.0, 0.0
    )

    for result, target in zip(roundtrip, original):
        assert isinstance(result, (jnp.ndarray, jax.Array))
        assert result.dtype == target.dtype
        numpy.testing.assert_allclose(result, target, rtol=1e-12)
    numpy.testing.assert_allclose(origin_theta, numpy.zeros(3))

    def loss(x, y, z):
        r, p, t = coordinates.cartesian_to_spherical(x, y, z)
        x_rec, y_rec, z_rec = coordinates.spherical_to_cartesian(r, p, t)
        return jnp.sum(x_rec + y_rec + z_rec)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*original)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

def test_conversions_mass_and_spin_derivatives_match_analytic():
    m1 = jnp.array([30.0, 15.0], dtype=jnp.float64)
    m2 = jnp.array([20.0, 10.0], dtype=jnp.float64)

    mc = conversions.mchirp_from_mass1_mass2(m1, m2)
    eta = conversions.eta_from_mass1_mass2(m1, m2)
    q = conversions.q_from_mass1_mass2(m1, m2)

    assert isinstance(mc, (jnp.ndarray, jax.Array))
    assert isinstance(eta, (jnp.ndarray, jax.Array))
    assert isinstance(q, (jnp.ndarray, jax.Array))

    # Test round-trip mass2 from mchirp, mass1
    m2_calc = conversions.mass2_from_mchirp_mass1(mc, m1)
    numpy.testing.assert_allclose(m2_calc, m2, rtol=1e-12)

    # Differentiability check
    def mc_loss(m_one, m_two):
        return jnp.sum(conversions.mchirp_from_mass1_mass2(m_one, m_two))

    dmc_dm1, dmc_dm2 = jax.grad(mc_loss, argnums=(0, 1))(m1, m2)
    assert bool(jnp.all(jnp.isfinite(dmc_dm1)))
    assert bool(jnp.all(jnp.isfinite(dmc_dm2)))

    # Tidal conversions
    lambda1 = jnp.array([300.0, 400.0], dtype=jnp.float64)
    lambda2 = jnp.array([200.0, 250.0], dtype=jnp.float64)
    ltilde = conversions.lambda_tilde(m1, m2, lambda1, lambda2)
    dltilde = conversions.delta_lambda_tilde(m1, m2, lambda1, lambda2)
    assert isinstance(ltilde, (jnp.ndarray, jax.Array))
    assert isinstance(dltilde, (jnp.ndarray, jax.Array))

    l1_rec = conversions.lambda1_from_delta_lambda_tilde_lambda_tilde(dltilde, ltilde, m1, m2)
    l2_rec = conversions.lambda2_from_delta_lambda_tilde_lambda_tilde(dltilde, ltilde, m1, m2)
    numpy.testing.assert_allclose(l1_rec, lambda1, rtol=1e-10)
    numpy.testing.assert_allclose(l2_rec, lambda2, rtol=1e-10)
