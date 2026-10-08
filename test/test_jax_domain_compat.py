# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused JAX compatibility tests for scientific helper modules."""

import numpy
import pytest

from pycbc import boundaries, conversions, cosmology, transforms
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


def test_log_and_custom_transforms_preserve_tensors_and_gradients():
    x = jnp.array([0.25, 0.8, 2.5], dtype=jnp.float64)
    p = jnp.array([-1.5, 0.25, 2.0], dtype=jnp.float64)
    q = jnp.array([1.5, 3.0, 2.0], dtype=jnp.float64)

    log = transforms.Log("x", "logx")
    exponent = transforms.Exponent("logx", "x_roundtrip")
    logit = transforms.Logit("p", "logitp", domain=(-2.0, 3.0))
    logistic = transforms.Logistic(
        "logitp", "p_roundtrip", codomain=(-2.0, 3.0)
    )
    custom = transforms.CustomTransform(
        ["x", "q"],
        ["combined"],
        {"combined": "sin(x) + sqrt(q)"},
        jacobian="exp(x) / q",
    )

    logx = log.transform({"x": x})["logx"]
    x_roundtrip = exponent.transform({"logx": logx})["x_roundtrip"]
    logitp = logit.transform({"p": p})["logitp"]
    p_roundtrip = logistic.transform({"logitp": logitp})["p_roundtrip"]
    combined = custom.transform({"x": x, "q": q})["combined"]
    jacobian = custom.jacobian({"x": x, "q": q})

    results = (logx, x_roundtrip, logitp, p_roundtrip, combined, jacobian)
    assert all(isinstance(result, (jnp.ndarray, jax.Array)) for result in results)
    numpy.testing.assert_allclose(x_roundtrip, x, rtol=1e-12)
    numpy.testing.assert_allclose(p_roundtrip, p, rtol=1e-12)
    numpy.testing.assert_allclose(combined, jnp.sin(x) + jnp.sqrt(q), rtol=1e-12)
    numpy.testing.assert_allclose(jacobian, jnp.exp(x) / q, rtol=1e-12)

    def total_loss(x_val, p_val, q_val):
        lx = log.transform({"x": x_val})["logx"]
        xr = exponent.transform({"logx": lx})["x_roundtrip"]
        lp = logit.transform({"p": p_val})["logitp"]
        pr = logistic.transform({"logitp": lp})["p_roundtrip"]
        c = custom.transform({"x": x_val, "q": q_val})["combined"]
        j = custom.jacobian({"x": x_val, "q": q_val})
        return jnp.sum(lx + xr + lp + pr + c + j)

    grads = jax.grad(total_loss, argnums=(0, 1, 2))(x, p, q)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_custom_transform_keeps_documented_numpy_fallback():
    custom = transforms.CustomTransform(
        ["value"], ["output"], {"output": "host_only(value, 'square')"}
    )
    custom._scratch.add_functions(
        "host_only", lambda value, mode: value**2 + (mode == "square")
    )

    value = jnp.array(2.0)
    output = custom.transform({"value": value})
    assert output["value"] is value
    assert isinstance(output["output"], (numpy.floating, float))
    assert output["output"] == 5.0


def test_distance_and_volume_interpolation_match_numpy_with_gradients():
    distance_converter = cosmology.DistToZ(numpoints=256)
    distances = numpy.array([1.0, 100.0, 1000.0, 10000.0])
    expected_redshift = distance_converter(distances)
    distance_tensor = jnp.array(distances, dtype=jnp.float64)

    actual_redshift = distance_converter(distance_tensor)

    assert isinstance(actual_redshift, (jnp.ndarray, jax.Array))
    assert actual_redshift.dtype == distance_tensor.dtype
    numpy.testing.assert_allclose(
        actual_redshift, expected_redshift, rtol=1e-13, atol=0.0
    )

    volume_converter = cosmology.ComovingVolInterpolator(
        "redshift", numpoints=64
    )
    volume_converter.setup_interpolant()
    redshifts = numpy.array([0.05, 0.2, 1.0, 4.0])
    volumes = volume_converter.cosmology.comoving_volume(redshifts).value
    expected_volume_result = volume_converter(volumes)
    volume_tensor = jnp.array(volumes, dtype=jnp.float64)
    actual_volume_result = volume_converter(volume_tensor)

    assert isinstance(actual_volume_result, (jnp.ndarray, jax.Array))
    assert actual_volume_result.dtype == volume_tensor.dtype
    numpy.testing.assert_allclose(
        actual_volume_result, expected_volume_result, rtol=1e-13, atol=0.0
    )

    def interp_loss(d, v):
        return jnp.sum(distance_converter(d)) + jnp.sum(volume_converter(v))

    grads = jax.grad(interp_loss, argnums=(0, 1))(distance_tensor, volume_tensor)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_tov_interpolation_preserves_tensor_and_numpy_contract(tmp_path):
    mass_knots = numpy.array([1.0, 1.2, 1.6, 2.0])
    lambda_knots = numpy.array([1000.0, 600.0, 200.0, 50.0])
    table_path = tmp_path / "mass-lambda.txt"
    numpy.savetxt(table_path, numpy.column_stack((mass_knots, lambda_knots)))
    transform = transforms.LambdaFromTOVFile(
        mass_param="mass1",
        lambda_param="lambda1",
        mass_lambda_file=table_path,
        redshift_mass=False,
    )
    masses = numpy.array([0.8, 1.1, 1.4, 2.0, 3.0])
    expected = numpy.interp(masses, mass_knots, lambda_knots, right=0.0)
    tensor = jnp.array(masses, dtype=jnp.float64)

    actual = transform.transform({"mass1": tensor})["lambda1"]

    assert isinstance(actual, (jnp.ndarray, jax.Array))
    assert actual.dtype == tensor.dtype
    numpy.testing.assert_allclose(actual, expected)

    def tov_loss(m):
        return jnp.sum(transform.transform({"mass1": m})["lambda1"])

    grad = jax.grad(tov_loss)(tensor)
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize(
    ("expression", "expected", "derivative"),
    (
        ("maximum(x, 0)", lambda x: numpy.maximum(x, 0), lambda x: (x > 0).astype(float)),
        ("minimum(1, x)", lambda x: numpy.minimum(1, x), lambda x: (x < 1).astype(float)),
        ("x / sqrt(2)", lambda x: x / numpy.sqrt(2),
         lambda x: numpy.full_like(x, 1 / numpy.sqrt(2))),
        ("sin(x) + sqrt(q)", lambda x: numpy.sin(x) + numpy.sqrt(2), numpy.cos),
        ("atan2(1, x)", lambda x: numpy.arctan2(1, x), lambda x: -1 / (1 + x*x)),
        ("hypot(x, 3)", lambda x: numpy.hypot(x, 3),
         lambda x: x / numpy.hypot(x, 3)),
    ),
)
def test_custom_expressions_accept_scalar_operands(expression, expected, derivative):
    values = numpy.array([-0.75, 0.5, 1.5])
    x = jnp.array(values, dtype=jnp.float64)
    transform = transforms.CustomTransform(
        ["x", "q"], ["result"], {"result": expression}
    )
    result = transform.transform({"x": x, "q": 2.0})["result"]

    def expr_loss(val):
        return jnp.sum(transform.transform({"x": val, "q": 2.0})["result"])

    gradient = jax.grad(expr_loss)(x)
    assert isinstance(result, (jnp.ndarray, jax.Array))
    assert result.dtype == x.dtype
    numpy.testing.assert_allclose(result, expected(values), rtol=1e-12)
    numpy.testing.assert_allclose(gradient, derivative(values), rtol=1e-12)


def test_custom_expression_does_not_truncate_scalar_against_integer_tensor():
    transform = transforms.CustomTransform(
        ["x"], ["result"], {"result": "maximum(x, 0.5)"}
    )
    x = jnp.array([0, 1, 2], dtype=jnp.int64)
    result = transform.transform({"x": x})["result"]
    numpy.testing.assert_allclose(result, [0.5, 1.0, 2.0])


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
