# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

import math
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
lal = pytest.importorskip("lal")

from pycbc.waveform._spherical_harmonics_jax import (  # noqa: E402
    spin_weighted_spherical_harmonic,
)


@pytest.mark.parametrize("theta", [0.0, 0.2, 1.1, 2.7, math.pi])
@pytest.mark.parametrize("phi", [-0.3, 0.0, 0.7, math.pi / 2.0])
def test_jax_harmonics_match_lal(theta, phi):
    for ell in range(2, 6):
        for emm in range(-ell, ell + 1):
            expected = lal.SpinWeightedSphericalHarmonic(
                theta, phi, -2, ell, emm
            )
            actual = spin_weighted_spherical_harmonic(
                theta,
                phi,
                -2,
                ell,
                emm,
                dtype=jnp.float64,
            )
            assert complex(actual) == pytest.approx(expected, abs=2.0e-13)


def test_jax_harmonics_vectorized():
    theta = jnp.array([0.2, 0.8, 1.4], dtype=jnp.float64)
    actual = spin_weighted_spherical_harmonic(
        theta,
        0.7,
        -2,
        5,
        -3,
        dtype=jnp.float64,
    )
    expected = np.array(
        [
            lal.SpinWeightedSphericalHarmonic(
                float(angle), 0.7, -2, 5, -3
            )
            for angle in np.asarray(theta)
        ]
    )
    assert actual.dtype == jnp.complex128
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_jax_harmonics_are_differentiable():
    def loss(theta_val):
        h = spin_weighted_spherical_harmonic(
            theta_val,
            0.2,
            -2,
            4,
            3,
            dtype=jnp.float64,
        )
        return jnp.real(h)

    grad_fn = jax.grad(loss)
    grad_val = grad_fn(0.7)
    assert np.isfinite(float(grad_val))


@pytest.mark.parametrize(
    "indices, exception",
    [
        ((-2, 2.0, 2), TypeError),
        ((-3, 2, 2), ValueError),
        ((-2, 2, 3), ValueError),
    ],
)
def test_jax_harmonics_validate_indices(indices, exception):
    with pytest.raises(exception):
        spin_weighted_spherical_harmonic(
            0.2,
            0.3,
            *indices,
            dtype=jnp.float64,
        )
