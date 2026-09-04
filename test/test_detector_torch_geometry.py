# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused tests for public Torch detector-geometry dispatch."""

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


REFERENCE_TIME = 1126259462.0


def _torch():
    return pytest.importorskip("torch")


@pytest.fixture(params=("cpu", "cuda"))
def torch_device(request):
    """Exercise CPU and CUDA when the optional accelerator is available."""
    torch = _torch()
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


def test_importing_ground_does_not_import_optional_torch():
    command = (
        "import sys; "
        "import pycbc.detector.ground; "
        "assert 'torch' not in sys.modules"
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
    np.testing.assert_allclose(
        fcross, expected_fcross, rtol=1e-14, atol=1e-16
    )
    np.testing.assert_allclose(delay, expected_delay, rtol=1e-14, atol=1e-16)


def test_torch_detector_geometry_matches_numpy_and_has_gradients(
    torch_device,
):
    torch = _torch()
    detector = Detector("H1")
    ra_np = np.linspace(0.1, 1.1, 6).reshape(2, 3)
    dec_np = np.array([[-0.4], [0.3]])
    polarization_np = np.linspace(0.2, 0.6, 3)
    time_np = REFERENCE_TIME + np.arange(3, dtype=np.float64)
    expected = detector.antenna_pattern_and_time_delay(
        ra_np, dec_np, polarization_np, time_np
    )

    ra = torch.tensor(
        ra_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    dec = torch.tensor(
        dec_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    polarization = torch.tensor(
        polarization_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    time = torch.tensor(
        time_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    actual = detector.antenna_pattern_and_time_delay(
        ra, dec, polarization, time
    )

    for value, reference in zip(actual, expected):
        assert value.device == ra.device
        torch.testing.assert_close(
            value,
            torch.as_tensor(reference, device=torch_device),
            rtol=1e-11,
            atol=1e-12,
        )

    loss = sum(value.square().sum() for value in actual)
    loss.backward()
    for value in (ra, dec, polarization, time):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_torch_single_arm_response_matches_numpy_and_differentiates(
    torch_device,
):
    torch = _torch()
    frequency_np = np.array([10.0, 100.0, 1000.0])
    direction_np = np.array([-0.8, 0.0, 0.8])
    expected = single_arm_frequency_response(
        frequency_np, direction_np, 4000.0
    )
    frequency = torch.tensor(
        frequency_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    direction = torch.tensor(
        direction_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )

    actual = single_arm_frequency_response(frequency, direction, 4000.0)

    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, device=torch_device),
        rtol=1e-12,
        atol=1e-12,
    )
    actual.real.sum().backward()
    assert torch.isfinite(frequency.grad).all()
    assert torch.isfinite(direction.grad).all()


def test_torch_finite_arm_antenna_response_matches_numpy(torch_device):
    torch = _torch()
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
    frequency = torch.tensor(
        frequency_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = detector.antenna_pattern(
        torch.tensor(1.2, device=torch_device, dtype=torch.float64),
        torch.tensor(-0.4, device=torch_device, dtype=torch.float64),
        torch.tensor(0.7, device=torch_device, dtype=torch.float64),
        REFERENCE_TIME,
        frequency=frequency,
    )

    for value, reference in zip(actual, expected):
        torch.testing.assert_close(
            value,
            torch.as_tensor(reference, device=torch_device),
            rtol=1e-11,
            atol=1e-12,
        )
    sum(value.real.sum() for value in actual).backward()
    assert torch.isfinite(frequency.grad).all()


def test_torch_effective_distance_and_arrival_time_keep_tensor_contract(
    torch_device,
):
    torch = _torch()
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

    ra = torch.tensor(
        ra_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    dec = torch.tensor(
        dec_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    polarization = torch.tensor(
        polarization_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distance = torch.tensor(
        distance_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    inclination = torch.tensor(
        inclination_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    time = torch.tensor(
        time_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )

    actual_distance = detector.effective_distance(
        distance, ra, dec, polarization, time, inclination
    )
    actual_arrival = detector.arrival_time(time, ra, dec)

    torch.testing.assert_close(
        actual_distance,
        torch.as_tensor(expected_distance, device=torch_device),
    )
    torch.testing.assert_close(
        actual_arrival,
        torch.as_tensor(expected_arrival, device=torch_device),
    )
    (actual_distance.sum() + actual_arrival.sum()).backward()
    for value in (ra, dec, polarization, distance, inclination, time):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_network_geometry_numpy_and_torch_match_individual_detectors(
    torch_device,
):
    torch = _torch()
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

    ra = torch.tensor(
        ra_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    dec = torch.tensor(
        dec_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    polarization = torch.tensor(
        polarization_np,
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    time = torch.tensor(
        time_np, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    torch_result = network.antenna_pattern_and_time_delay(
        ra, dec, polarization, time
    )
    for value, reference in zip(torch_result, numpy_result):
        torch.testing.assert_close(
            value,
            torch.as_tensor(reference, device=torch_device),
            rtol=1e-11,
            atol=1e-12,
        )
    sum(value.square().sum() for value in torch_result).backward()
    for value in (ra, dec, polarization, time):
        assert torch.isfinite(value.grad).all()


def test_torch_tensor_subclass_uses_public_dispatch_and_detector_hooks():
    torch = _torch()

    class TensorSubclass(torch.Tensor):
        pass

    class CustomDetector(Detector):
        def __init__(self):
            super().__init__("H1")
            self.antenna_calls = 0

        def antenna_pattern(self, *args, **kwargs):
            self.antenna_calls += 1
            return super().antenna_pattern(*args, **kwargs)

    detector = CustomDetector()
    ra = torch.tensor([0.3, 0.7], dtype=torch.float64).as_subclass(
        TensorSubclass
    )
    dec = torch.tensor([-0.2, 0.4], dtype=torch.float64)
    polarization = torch.tensor([0.1, 0.5], dtype=torch.float64)
    distance = torch.tensor([100.0, 200.0], dtype=torch.float64)
    inclination = torch.tensor([0.2, 0.8], dtype=torch.float64)

    result = detector.effective_distance(
        distance,
        ra,
        dec,
        polarization,
        REFERENCE_TIME,
        inclination,
    )

    assert detector.antenna_calls == 1
    assert torch.isfinite(result).all()


def test_torch_geometry_rejects_unsupported_inputs():
    torch = _torch()
    detector = Detector("H1")

    with pytest.raises(NotImplementedError, match="only the tensor response"):
        detector.antenna_pattern(
            torch.tensor(0.2), 0.1, 0.3, REFERENCE_TIME,
            polarization_type="vector",
        )
    with pytest.raises(TypeError, match="angles must be floating"):
        detector.antenna_pattern(
            torch.tensor([1], dtype=torch.int64), 0.1, 0.3, REFERENCE_TIME
        )
    with pytest.raises(NotImplementedError, match="GMST reference time"):
        Detector("H1", reference_time=None).antenna_pattern(
            torch.tensor([0.2]),
            torch.tensor([0.1]),
            torch.tensor([0.3]),
            torch.tensor([REFERENCE_TIME]),
        )
