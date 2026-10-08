# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent original geometry controls and native contract regressions."""

import os
from types import MethodType

import numpy as np
import pytest

from pycbc.detector.ground import (
    Detector,
    NetworkGeometry,
    single_arm_frequency_response,
)
from pycbc.scheme import JAXScheme

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)
DEVICE = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu").removeprefix("jax:")
TIME = 1126259462.0


def _exact(actual, expected):
    if isinstance(expected, tuple):
        for a, e in zip(actual, expected):
            _exact(a, e)
        return
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.dtype == e.dtype
    assert a.shape == e.shape
    assert a.tobytes() == e.tobytes()


def _inputs(dtype=np.float64):
    return tuple(
        np.asarray(v, dtype=dtype)
        for v in (
            [0.3, 0.7, 1.2, 2.0],
            [-0.2, 0.4, -0.5, 0.1],
            [0.1, 0.5, 0.3, 0.7],
            TIME + np.array([0.0, 0.125, 1.0, 3.0]),
        )
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name",
    [
        "antenna_pattern",
        "time_delay_from_location",
        "gmst_estimate",
        "effective_distance",
        "arrival_time",
    ],
)
def test_native_detector_operation_is_original(name, dtype):
    detector = Detector("H1")
    ra, dec, pol, time = _inputs(dtype)
    if name == "antenna_pattern":
        args = (ra, dec, pol, time)
    elif name == "time_delay_from_location":
        args = (np.zeros(3), ra, dec, time)
    elif name == "gmst_estimate":
        args = (time,)
    elif name == "effective_distance":
        args = (
            np.array([100.0, 200.0, 300.0, 400.0], dtype=dtype),
            ra,
            dec,
            pol,
            time,
            0.7,
        )
    else:
        args = (time, ra, dec)
    function = getattr(detector, name)
    expected = function(*args)
    with JAXScheme(
        DEVICE, reference_operations=["detector.Detector." + name]
    ) as context:
        actual = function(
            *(
                (
                    jax.device_put(v, context.jax_device)
                    if isinstance(v, np.ndarray)
                    else v
                )
                for v in args
            )
        )
        _exact(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_arm_transfer_preserves_zero_and_bytes(dtype):
    frequency = np.array([0.0, 1e-6, 10.0, 1000.0], dtype=dtype)
    direction = np.array([0.1, -0.8, 0.3, 0.8], dtype=dtype)
    with np.errstate(all="ignore"):
        expected = single_arm_frequency_response(frequency, direction, 4000.0)
        with JAXScheme(
            DEVICE, reference_operations=["detector.single_arm_frequency_response"]
        ):
            actual = single_arm_frequency_response(
                jnp.asarray(frequency), jnp.asarray(direction), 4000.0
            )
            _exact(actual, expected)
        with JAXScheme(DEVICE):
            default = single_arm_frequency_response(
                jnp.asarray(frequency), jnp.asarray(direction), 4000.0
            )
            assert np.isnan(default[0])


def test_combined_and_network_controls_compose_exactly():
    network = NetworkGeometry(["H1", "L1", "V1"])
    inputs = _inputs()
    expected = network.antenna_pattern_and_time_delay(*inputs)
    with JAXScheme(
        DEVICE,
        reference_operations=[
            "detector.Detector.antenna_pattern",
            "detector.Detector.time_delay_from_location",
        ],
    ):
        _exact(
            network.antenna_pattern_and_time_delay(*(jnp.asarray(v) for v in inputs)),
            expected,
        )


@pytest.mark.parametrize("vector_sky", [False, True])
def test_independent_antenna_and_distance_scale_restore_original(vector_sky):
    detector = Detector("L1")
    ra, dec, pol, time = _inputs() if vector_sky else (0.3, -0.2, 0.1, TIME)
    distance = np.array([100.0, 200.0, 300.0, 400.0])
    expected = detector.effective_distance(distance, ra, dec, pol, time, 0.7)
    with JAXScheme(
        DEVICE,
        reference_operations=[
            "detector.Detector.antenna_pattern",
            "detector.Detector.effective_distance_scale",
        ],
    ):
        actual = detector.effective_distance(
            jnp.asarray(distance), ra, dec, pol, time, 0.7
        )
        _exact(actual, expected)


def test_antenna_control_is_independent_and_default_avoids_original(monkeypatch):
    from pycbc.detector import ground_jax

    def forbidden(*args, **kwargs):
        raise AssertionError("JAX antenna kernel called")

    monkeypatch.setattr(ground_jax, "_jax_antenna_pattern", forbidden)
    detector = Detector("H1")
    with JAXScheme(DEVICE, reference_operations=["detector.Detector.antenna_pattern"]):
        detector.antenna_pattern(jnp.array(0.3), -0.2, 0.1, TIME)
    with JAXScheme(
        DEVICE, reference_operations=["detector.Detector.time_delay_from_location"]
    ):
        with pytest.raises(AssertionError, match="JAX antenna"):
            detector.antenna_pattern(jnp.array(0.3), -0.2, 0.1, TIME)


@pytest.mark.parametrize("shape", [(3, 7), (2, 3)])
def test_grid_reference_keeps_original_support_or_replays_scalar_rows(shape):
    detector = Detector("H1")
    ra = np.linspace(0.3, 1.2, np.prod(shape)).reshape(shape)
    if shape[0] == 3:
        expected = detector.antenna_pattern(ra, -0.2, 0.1, TIME)
        with JAXScheme(DEVICE):
            actual = detector.antenna_pattern(jnp.asarray(ra), -0.2, 0.1, TIME)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-15)
    else:
        with pytest.raises(ValueError, match="not aligned"):
            detector.antenna_pattern(ra, -0.2, 0.1, TIME)
        expected = tuple(
            np.array(
                [detector.antenna_pattern(r, -0.2, 0.1, TIME)[i] for r in ra.flat]
            ).reshape(shape)
            for i in range(2)
        )
    with JAXScheme(DEVICE, reference_operations=["detector.Detector.antenna_pattern"]):
        _exact(detector.antenna_pattern(jnp.asarray(ra), -0.2, 0.1, TIME), expected)


@pytest.mark.parametrize(
    "method", ["antenna_pattern", "time_delay_from_location", "gmst_estimate"]
)
@pytest.mark.parametrize("time_array", [False, True])
def test_network_honors_instance_overrides(method, time_array):
    detector = Detector("H1")
    original = getattr(detector, method)

    def override(self, *args, **kwargs):
        values = original(*args, **kwargs)
        return (
            tuple(v + 0.2 for v in values)
            if isinstance(values, tuple)
            else values + 0.004
        )

    setattr(detector, method, MethodType(override, detector))
    network = NetworkGeometry([detector])
    with JAXScheme(DEVICE):
        time = jnp.array([TIME, TIME + 0.125]) if time_array else TIME
        args = (jnp.array([0.3, 0.7]), -0.2, 0.1, time)
        actual = network.antenna_pattern_and_time_delay(*args)
        expected = detector.antenna_pattern_and_time_delay(*args)
        for a, e in zip(actual, expected):
            _exact(a[0], e)


def test_network_rereads_geometry_and_detector_clock_policy():
    detectors = [
        Detector("H1", reference_time=TIME),
        Detector("L1", reference_time=TIME - 10000),
    ]
    network = NetworkGeometry(detectors)
    with JAXScheme(DEVICE):
        args = (jnp.array([0.3, 0.7]), -0.2, 0.1, TIME)
        detectors[0].response = detectors[0].response * 0.5
        detectors[0].location = detectors[0].location + 1000.0
        actual = network.antenna_pattern_and_time_delay(*args)
        for i, detector in enumerate(detectors):
            expected = detector.antenna_pattern_and_time_delay(*args)
            for a, e in zip(actual, expected):
                _exact(a[i], e)


def test_numpy_complex_input_is_not_silently_cast():
    with JAXScheme(DEVICE):
        with pytest.raises(TypeError, match="real"):
            Detector("H1").antenna_pattern(
                jnp.array(0.3), np.array(0.2 + 0.3j), 0.1, TIME
            )
        with pytest.raises(ValueError, match="at least one"):
            NetworkGeometry([])


def test_scalar_frequency_zero_keeps_original_real_response():
    detector = Detector("H1")
    with JAXScheme(DEVICE):
        args = (jnp.array(0.3), -0.2, 0.1, TIME)
        expected = detector.antenna_pattern(*args, frequency=0.0)
        actual = detector.antenna_pattern(*args, frequency=jnp.array(0.0))
        _exact(actual, expected)
        assert not jnp.iscomplexobj(actual[0])


def test_float32_time_grid_has_consistent_eager_and_compiled_dtype():
    detector = Detector("H1")
    with JAXScheme(DEVICE):
        ra = jnp.array([.3, .7], dtype=jnp.float32)
        dec = jnp.array([-.2, .4], dtype=jnp.float32)
        time = jnp.array([TIME, TIME + 128], dtype=jnp.float32)
        eager = detector.time_delay_from_earth_center(ra, dec, time)
        compiled = jax.jit(detector.time_delay_from_earth_center)(ra, dec, time)
        # The physical speed-of-light constant is a native float64 scalar.
        assert eager.dtype == compiled.dtype == jnp.float64
        np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-8)


def test_geometry_constants_follow_first_raw_device():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    with JAXScheme("cpu:0"):
        ra = jax.device_put(np.array([0.3, 0.7]), devices[1])
        dec = jax.device_put(np.array([-0.2, 0.4]), devices[0])
        result = NetworkGeometry(["H1", "L1"]).antenna_pattern_and_time_delay(
            ra, dec, 0.1, TIME
        )
        assert all(value.device == devices[1] for value in result)
        compiled = jax.jit(
            lambda r: Detector("H1").antenna_pattern(r, -0.2, 0.1, TIME)
        )(ra)
        assert all(value.device == devices[1] for value in compiled)
        overridden = [Detector("H1"), Detector("L1")]
        for detector in overridden:
            detector.antenna_pattern = MethodType(
                lambda self, *args: (np.ones(2), np.ones(2)), detector
            )
        result = NetworkGeometry(overridden).antenna_pattern_and_time_delay(
            ra, dec, 0.1, TIME
        )
        assert all(value.device == devices[1] for value in result)
        compiled = jax.jit(
            NetworkGeometry(overridden).antenna_pattern_and_time_delay
        )(ra, jax.device_put(dec, devices[1]), 0.1, TIME)
        assert all(value.device == devices[1] for value in compiled)
        location = jax.device_put(np.zeros(3), devices[0])
        delay = Detector("H1").time_delay_from_location(location, ra, dec, TIME)
        assert delay.device == devices[0]
        time = jax.device_put(np.array([TIME, TIME + .125]), devices[1])
        frequency = jax.device_put(np.array([10., 100.]), devices[0])
        response = Detector("H1").antenna_pattern(.3, -.2, .1, time, frequency)
        assert all(value.device == devices[1] for value in response)


def test_network_is_exported_by_public_detector_package():
    from pycbc.detector import NetworkGeometry as public_network
    assert public_network is NetworkGeometry
