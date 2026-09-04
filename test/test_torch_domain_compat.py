# Copyright (C) 2026  PyCBC developers
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused Torch compatibility tests for scientific helper modules."""

import numpy
import pytest

from pycbc import boundaries, cosmology, transforms
from pycbc.coordinates import base as coordinates

torch = pytest.importorskip("torch")


def test_boundary_conditioning_preserves_tensor_and_gradient():
    values = torch.tensor(
        [-10.0, -4.0, -1.0, 0.25, 1.0, 3.0, 8.0],
        dtype=torch.float64,
        requires_grad=True,
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
        assert isinstance(result, torch.Tensor)
        assert result.device == values.device
        assert result.dtype == values.dtype
        torch.testing.assert_close(result, values.new_tensor(target))
    sum(result.sum() for result in actual).backward()
    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all())


def test_coordinate_roundtrip_preserves_tensor_and_gradient():
    original = (
        torch.tensor([0.75, -1.25, 0.4], dtype=torch.float64,
                     requires_grad=True),
        torch.tensor([-0.5, 0.8, 1.1], dtype=torch.float64,
                     requires_grad=True),
        torch.tensor([1.2, -0.3, 0.65], dtype=torch.float64,
                     requires_grad=True),
    )

    rho, phi, theta = coordinates.cartesian_to_spherical(*original)
    roundtrip = coordinates.spherical_to_cartesian(rho, phi, theta)
    origin_theta = coordinates.cartesian_to_spherical_polar(
        torch.zeros(3, dtype=torch.float64), 0.0, 0.0
    )

    for result, target in zip(roundtrip, original):
        assert isinstance(result, torch.Tensor)
        assert result.device == target.device
        assert result.dtype == target.dtype
        torch.testing.assert_close(result, target)
    assert torch.equal(origin_theta, torch.zeros_like(origin_theta))
    sum(result.sum() for result in roundtrip).backward()
    assert all(value.grad is not None for value in original)
    assert all(bool(torch.isfinite(value.grad).all()) for value in original)


def test_log_and_custom_transforms_preserve_tensors_and_gradients():
    x = torch.tensor([0.25, 0.8, 2.5], dtype=torch.float64,
                     requires_grad=True)
    p = torch.tensor([-1.5, 0.25, 2.0], dtype=torch.float64,
                     requires_grad=True)
    q = torch.tensor([1.5, 3.0, 2.0], dtype=torch.float64,
                     requires_grad=True)

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
    assert all(isinstance(result, torch.Tensor) for result in results)
    torch.testing.assert_close(x_roundtrip, x)
    torch.testing.assert_close(p_roundtrip, p)
    torch.testing.assert_close(combined, torch.sin(x) + torch.sqrt(q))
    torch.testing.assert_close(jacobian, torch.exp(x) / q)
    sum(result.sum() for result in results).backward()
    assert all(value.grad is not None for value in (x, p, q))
    assert all(bool(torch.isfinite(value.grad).all()) for value in (x, p, q))


def test_custom_transform_keeps_documented_numpy_fallback():
    custom = transforms.CustomTransform(
        ["value"], ["output"], {"output": "host_only(value, 'square')"}
    )
    custom._scratch.add_functions(
        "host_only", lambda value, mode: value**2 + (mode == "square")
    )

    value = torch.tensor(2.0)
    output = custom.transform({"value": value})

    assert output["value"] is value
    assert isinstance(output["output"], numpy.floating)
    assert output["output"] == 5.0


def test_distance_and_volume_interpolation_match_numpy_with_gradients():
    distance_converter = cosmology.DistToZ(numpoints=256)
    distances = numpy.array([1.0, 100.0, 1000.0, 10000.0])
    expected_redshift = distance_converter(distances)
    distance_tensor = torch.tensor(
        distances, dtype=torch.float64, requires_grad=True
    )

    actual_redshift = distance_converter(distance_tensor)

    assert isinstance(actual_redshift, torch.Tensor)
    assert actual_redshift.device == distance_tensor.device
    assert actual_redshift.dtype == distance_tensor.dtype
    torch.testing.assert_close(
        actual_redshift, torch.as_tensor(expected_redshift), rtol=1e-13,
        atol=0.0
    )

    volume_converter = cosmology.ComovingVolInterpolator(
        "redshift", numpoints=64
    )
    volume_converter.setup_interpolant()
    redshifts = numpy.array([0.05, 0.2, 1.0, 4.0])
    volumes = volume_converter.cosmology.comoving_volume(redshifts).value
    expected_volume_result = volume_converter(volumes)
    volume_tensor = torch.tensor(
        volumes, dtype=torch.float64, requires_grad=True
    )
    actual_volume_result = volume_converter(volume_tensor)

    assert isinstance(actual_volume_result, torch.Tensor)
    assert actual_volume_result.device == volume_tensor.device
    assert actual_volume_result.dtype == volume_tensor.dtype
    torch.testing.assert_close(
        actual_volume_result,
        torch.as_tensor(expected_volume_result),
        rtol=1e-13,
        atol=0.0,
    )
    (actual_redshift.sum() + actual_volume_result.sum()).backward()
    assert distance_tensor.grad is not None
    assert volume_tensor.grad is not None
    assert bool(torch.isfinite(distance_tensor.grad).all())
    assert bool(torch.isfinite(volume_tensor.grad).all())


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
    tensor = torch.tensor(masses, dtype=torch.float64, requires_grad=True)

    actual = transform.transform({"mass1": tensor})["lambda1"]

    assert isinstance(actual, torch.Tensor)
    assert actual.device == tensor.device
    assert actual.dtype == tensor.dtype
    torch.testing.assert_close(actual, tensor.new_tensor(expected))
    actual.sum().backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())
