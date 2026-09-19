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

from pycbc.detector.ground import (
    Detector,
    NetworkGeometry,
    _scalar_antenna_pattern_and_time_delay,
    single_arm_frequency_response,
)

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


@pytest.mark.parametrize("name", ["H1", "L1", "V1"])
def test_combined_scalar_geometry_matches_public_numpy_methods(name):
    detector = Detector(name)
    ra = 1.234
    dec = 0.567
    polarization = 0.891

    fplus0, fcross0, delay0 = _scalar_antenna_pattern_and_time_delay(
        detector, ra, dec, REFERENCE_TIME
    )
    fplus, fcross, delay = detector.antenna_pattern_and_time_delay(
        ra, dec, polarization, REFERENCE_TIME
    )
    cos2psi = np.cos(2.0 * polarization)
    sin2psi = np.sin(2.0 * polarization)

    assert fplus == cos2psi * fplus0 + sin2psi * fcross0
    assert fcross == -sin2psi * fplus0 + cos2psi * fcross0
    assert delay == delay0

    public_fplus, public_fcross = detector.antenna_pattern(
        ra, dec, polarization, REFERENCE_TIME
    )
    public_delay = detector.time_delay_from_earth_center(
        ra, dec, REFERENCE_TIME
    )
    np.testing.assert_allclose(fplus, public_fplus, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(fcross, public_fcross, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(delay, public_delay, rtol=1e-14, atol=1e-16)


def test_combined_numpy_geometry_broadcasts_without_behavior_change():
    detector = Detector("H1")
    ra = np.linspace(0.1, 1.1, 6).reshape(2, 3)
    dec = np.array([[-0.4], [0.3]])
    polarization = np.linspace(0.2, 0.6, 3)
    time = REFERENCE_TIME + np.arange(3, dtype=np.float64)

    fplus, fcross, delay = detector.antenna_pattern_and_time_delay(
        ra, dec, polarization, time
    )
    expected_fplus, expected_fcross = detector.antenna_pattern(
        ra, dec, polarization, time
    )
    expected_delay = detector.time_delay_from_earth_center(ra, dec, time)

    assert fplus.shape == (2, 3)
    assert fcross.shape == (2, 3)
    assert delay.shape == (2, 3)
    np.testing.assert_allclose(fplus, expected_fplus, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(fcross, expected_fcross, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(delay, expected_delay, rtol=1e-14, atol=1e-16)


def test_jax_detector_geometry_matches_numpy_and_has_gradients():
    detector = Detector("H1")
    ra_np = np.linspace(0.1, 1.1, 6).reshape(2, 3)
    dec_np = np.array([[-0.4], [0.3]])
    polarization_np = np.linspace(0.2, 0.6, 3)
    time_np = REFERENCE_TIME + np.arange(3, dtype=np.float64)

    expected = detector.antenna_pattern_and_time_delay(
        ra_np, dec_np, polarization_np, time_np
    )

    ra = jnp.array(ra_np, dtype=jnp.float64)
    dec = jnp.array(dec_np, dtype=jnp.float64)
    polarization = jnp.array(polarization_np, dtype=jnp.float64)
    time = jnp.array(time_np, dtype=jnp.float64)

    actual = detector.antenna_pattern_and_time_delay(
        ra, dec, polarization, time
    )

    for value, reference in zip(actual, expected):
        assert isinstance(value, (jnp.ndarray, jax.Array))
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-12)

    def loss(r, d, p, t):
        fp, fc, dl = detector.antenna_pattern_and_time_delay(r, d, p, t)
        return jnp.sum(fp**2 + fc**2 + dl**2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(ra, dec, polarization, time)
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


def test_scalar_sky_over_retained_time_grid_matches_numpy_and_gradients():
    detector = Detector("H1", reference_time=REFERENCE_TIME)
    times = REFERENCE_TIME + np.linspace(-128.0, 0.0, 129).reshape(3, 43)
    sky = [1.37, -1.26, 0.23]
    angles = [jnp.array(value, dtype=jnp.float64) for value in sky]
    actual = detector.antenna_pattern_and_time_delay(*angles, times)
    expected = detector.antenna_pattern_and_time_delay(*sky, times)
    separate = (
        *detector.antenna_pattern(*angles, times),
        detector.time_delay_from_earth_center(*angles[:2], times),
    )
    for value, reference, single in zip(actual, expected, separate):
        assert value.shape == times.shape
        np.testing.assert_allclose(value, reference, rtol=1e-12)
        np.testing.assert_allclose(value, single, rtol=1e-12)

    def loss(r, d, p):
        res = detector.antenna_pattern_and_time_delay(r, d, p, times)
        return sum(jnp.mean(v) for v in res)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*angles)
    step = 1e-6
    for index, angle_grad in enumerate(grads):
        upper, lower = sky.copy(), sky.copy()
        upper[index] += step
        lower[index] -= step
        plus = detector.antenna_pattern_and_time_delay(*upper, times)
        minus = detector.antenna_pattern_and_time_delay(*lower, times)
        derivative = sum((hi - lo).mean() for hi, lo in zip(plus, minus))
        derivative /= 2.0 * step
        assert float(angle_grad) == pytest.approx(derivative, rel=1e-7, abs=1e-9)


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


def test_network_geometry_numpy_and_jax_match_individual_detectors():
    names = ["H1", "L1", "V1"]
    network = NetworkGeometry(names)
    ra_np = np.linspace(0.2, 1.0, 4)
    dec_np = np.linspace(-0.4, 0.3, 4)
    polarization_np = np.linspace(0.1, 0.7, 4)
    time_np = REFERENCE_TIME + np.arange(4, dtype=np.float64)

    numpy_result = network.antenna_pattern_and_time_delay(
        ra_np, dec_np, polarization_np, time_np
    )
    assert len(network) == len(names)
    assert network.detector_names == names
    assert all(value.shape == (3, 4) for value in numpy_result)
    for index, name in enumerate(names):
        expected = network[name].antenna_pattern_and_time_delay(
            ra_np, dec_np, polarization_np, time_np
        )
        for value, reference in zip(numpy_result, expected):
            np.testing.assert_allclose(
                value[index], reference, rtol=1e-12, atol=1e-14
            )
    assert set(network.to_dict(numpy_result[0])) == set(names)

    ra = jnp.array(ra_np, dtype=jnp.float64)
    dec = jnp.array(dec_np, dtype=jnp.float64)
    polarization = jnp.array(polarization_np, dtype=jnp.float64)
    time = jnp.array(time_np, dtype=jnp.float64)

    jax_result = network.antenna_pattern_and_time_delay(
        ra, dec, polarization, time
    )
    for value, reference in zip(jax_result, numpy_result):
        np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-12)

    def loss(r, d, p, t):
        res = network.antenna_pattern_and_time_delay(r, d, p, t)
        return sum(jnp.sum(v**2) for v in res)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(ra, dec, polarization, time)
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("override_location", ["subclass", "instance"])
@pytest.mark.parametrize("method_name", [
    "antenna_pattern", "time_delay_from_earth_center", "time_delay_from_location",
])
def test_combined_geometry_preserves_detector_overrides(
        monkeypatch, override_location, method_name):
    from types import MethodType

    original = getattr(Detector, method_name)
    calls = []

    def overridden(self, *args):
        calls.append(method_name)
        result = original(self, *args)
        if isinstance(result, tuple):
            return tuple(value + 0.2 for value in result)
        return result + 0.004

    class CustomDetector(Detector):
        pass

    detector = CustomDetector("H1")
    if override_location == "subclass":
        monkeypatch.setattr(CustomDetector, method_name, overridden)
    else:
        monkeypatch.setattr(detector, method_name, MethodType(overridden, detector))
    ra = jnp.array([0.3, 0.7], dtype=jnp.float64)
    dec = jnp.array([-0.2, 0.4], dtype=jnp.float64)
    args = (ra, dec, 0.1, REFERENCE_TIME)

    actual = detector.antenna_pattern_and_time_delay(*args)
    assert calls == [method_name]
    expected = (
        *detector.antenna_pattern(*args),
        detector.time_delay_from_earth_center(ra, dec, REFERENCE_TIME),
    )
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=1e-12)

    def loss(r):
        res = detector.antenna_pattern_and_time_delay(r, dec, 0.1, REFERENCE_TIME)
        return sum(jnp.sum(v) for v in res)

    gr = jax.grad(loss)(ra)
    assert bool(jnp.all(jnp.isfinite(gr)))


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
