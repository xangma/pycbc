# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused tests for public JAX detector-geometry dispatch."""

import subprocess

import sys

import numpy as np

import pytest

from pycbc.detector.ground import Detector, single_arm_frequency_response

jax = pytest.importorskip("jax")

jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

REFERENCE_TIME = 1126259462.0

def test_importing_ground_does_not_import_optional_jax():
    command = (
        "import sys; "
        "import pycbc.detector.ground; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)

def test_jax_single_arm_response_matches_numpy_and_differentiates():
    frequency_np = np.array([10.0, 100.0, 1000.0])
    direction_np = np.array([-0.8, 0.0, 0.8])
    expected = single_arm_frequency_response(
        frequency_np, direction_np, 4000.0
    )
    frequency = jnp.array(frequency_np, dtype=jnp.float64)
    direction = jnp.array(direction_np, dtype=jnp.float64)

    actual = single_arm_frequency_response(frequency, direction, 4000.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def loss(f, d):
        return jnp.sum(jnp.real(single_arm_frequency_response(f, d, 4000.0)))

    gf, gd = jax.grad(loss, argnums=(0, 1))(frequency, direction)
    assert bool(jnp.all(jnp.isfinite(gf)))
    assert bool(jnp.all(jnp.isfinite(gd)))

def test_jax_finite_arm_antenna_response_matches_numpy():
    detector = Detector("H1")
    frequency_np = np.array([20.0, 200.0, 800.0])
    scalar_results = [
        detector.antenna_pattern(
            1.2, -0.4, 0.7, REFERENCE_TIME, frequency=frequency
        )
        for frequency in frequency_np
    ]
    expected = tuple(
        np.asarray([result[index] for result in scalar_results])
        for index in range(2)
    )
    frequency = jnp.array(frequency_np, dtype=jnp.float64)
    actual = detector.antenna_pattern(
        jnp.array(1.2, dtype=jnp.float64),
        jnp.array(-0.4, dtype=jnp.float64),
        jnp.array(0.7, dtype=jnp.float64),
        REFERENCE_TIME,
        frequency=frequency,
    )

    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-12)

    def loss(f):
        fp, fc = detector.antenna_pattern(
            1.2, -0.4, 0.7, REFERENCE_TIME, frequency=f
        )
        return jnp.sum(jnp.real(fp) + jnp.real(fc))

    gf = jax.grad(loss)(frequency)
    assert bool(jnp.all(jnp.isfinite(gf)))

def test_jax_effective_distance_and_arrival_time_keep_tensor_contract():
    detector = Detector("L1")
    ra_np = np.array([0.3, 0.7])
    dec_np = np.array([-0.2, 0.4])
    polarization_np = np.array([0.1, 0.5])
    distance_np = np.array([100.0, 200.0])
    inclination_np = np.array([0.2, 0.8])
    time_np = REFERENCE_TIME + np.array([0.0, 0.25])

    expected_distance = detector.effective_distance(
        distance_np,
        ra_np,
        dec_np,
        polarization_np,
        time_np,
        inclination_np,
    )
    expected_arrival = detector.arrival_time(time_np, ra_np, dec_np)

    ra = jnp.array(ra_np, dtype=jnp.float64)
    dec = jnp.array(dec_np, dtype=jnp.float64)
    polarization = jnp.array(polarization_np, dtype=jnp.float64)
    distance = jnp.array(distance_np, dtype=jnp.float64)
    inclination = jnp.array(inclination_np, dtype=jnp.float64)
    time = jnp.array(time_np, dtype=jnp.float64)

    actual_distance = detector.effective_distance(
        distance, ra, dec, polarization, time, inclination
    )
    actual_arrival = detector.arrival_time(time, ra, dec)

    np.testing.assert_allclose(actual_distance, expected_distance, rtol=1e-12)
    np.testing.assert_allclose(actual_arrival, expected_arrival, rtol=1e-12)

    def loss(dist, r, d, pol, t, inc):
        ed = detector.effective_distance(dist, r, d, pol, t, inc)
        arr = detector.arrival_time(t, r, d)
        return jnp.sum(ed + arr)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        distance, ra, dec, polarization, time, inclination
    )
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))

def test_jax_geometry_rejects_unsupported_inputs():
    detector = Detector("H1")

    with pytest.raises(NotImplementedError, match="only the tensor response"):
        detector.antenna_pattern(
            jnp.array(0.2), 0.1, 0.3, REFERENCE_TIME,
            polarization_type="vector",
        )
    with pytest.raises(TypeError, match="angles must be floating"):
        detector.antenna_pattern(
            jnp.array([1], dtype=jnp.int64), 0.1, 0.3, REFERENCE_TIME
        )
    with pytest.raises(NotImplementedError, match="GMST reference time"):
        Detector("H1", reference_time=None).antenna_pattern(
            jnp.array([0.2]),
            jnp.array([0.1]),
            jnp.array([0.3]),
            jnp.array([REFERENCE_TIME]),
        )
