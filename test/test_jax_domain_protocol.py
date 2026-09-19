# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain consumers of the shared optional tensor protocol for JAX."""

import numpy
import pytest

from pycbc import boundaries, transforms
from pycbc.coordinates import base as coordinates

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)


def test_coordinate_mixed_broadcasting_and_gradients():
    original = jnp.array([[3.0], [4.0]], dtype=jnp.float64)
    y = numpy.array([4.0, 3.0], dtype=numpy.float64)
    actual = coordinates.cartesian_to_spherical_rho(original, y, 0)
    expected = jnp.sqrt(original**2 + jnp.array(y)**2)
    assert isinstance(actual, (jnp.ndarray, jax.Array))
    assert actual.dtype == original.dtype
    assert actual.shape == (2, 2)
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)

    def loss(arr):
        return jnp.sum(coordinates.cartesian_to_spherical_rho(arr, y, 0))

    grad = jax.grad(loss)(original)
    expected_grad = jnp.sum(original / expected, axis=1, keepdims=True)
    numpy.testing.assert_allclose(grad, expected_grad, rtol=1e-12)


def test_coordinate_integer_inputs_use_default_float():
    actual = coordinates.cartesian_to_spherical_rho(
        jnp.array([3, 5]), 4, 0
    )
    assert jnp.issubdtype(actual.dtype, jnp.floating)
    numpy.testing.assert_allclose(actual, [5.0, 41.0**0.5])


def test_expression_and_boundary_keep_autograd():
    original = jnp.array([-2.5, 0.25, 2.5], dtype=jnp.float64)
    custom = transforms.CustomTransform(
        ["x"], ["result"], {"result": "sin(x) + x**2"}
    )
    actual = custom.transform({"x": original})["result"]
    assert isinstance(actual, (jnp.ndarray, jax.Array))
    numpy.testing.assert_allclose(actual, jnp.sin(original) + original**2)

    def loss_expr(x):
        return jnp.sum(custom.transform({"x": x})["result"])

    grad_expr = jax.grad(loss_expr)(original)
    numpy.testing.assert_allclose(grad_expr, jnp.cos(original) + 2 * original)

    bounds = boundaries.Bounds(-1.0, 1.0, btype_min="reflected",
                               btype_max="reflected")
    reflected = bounds.apply_conditions(original)
    assert isinstance(reflected, (jnp.ndarray, jax.Array))
    numpy.testing.assert_allclose(reflected, [0.5, 0.25, -0.5])

    def loss_bounds(x):
        return jnp.sum(bounds.apply_conditions(x))

    grad_bounds = jax.grad(loss_bounds)(original)
    numpy.testing.assert_allclose(grad_bounds, [-1.0, 1.0, -1.0])
