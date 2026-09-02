# Copyright (C) 2026  PyCBC developers
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

import numpy
import pytest

from pycbc import cosmology


@pytest.fixture(scope="module")
def comoving_volume_converters():
    converters = {
        parameter: cosmology.ComovingVolInterpolator(
            parameter, numpoints=64
        )
        for parameter in ("redshift", "luminosity_distance")
    }
    for converter in converters.values():
        converter.setup_interpolant()
    return converters


def test_distance_to_redshift_torch_matches_numpy_and_has_gradient():
    torch = pytest.importorskip("torch")
    converter = cosmology.DistToZ(numpoints=1000)
    distances = numpy.array([0.0, 1.0, 100.0, 1000.0, 10000.0, 1.0e6])
    expected = converter(distances)

    tensor = torch.tensor(distances, dtype=torch.float64, requires_grad=True)
    actual = converter(tensor)

    assert isinstance(actual, torch.Tensor)
    assert actual.device == tensor.device
    assert actual.dtype == tensor.dtype
    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=1e-13,
                               atol=0.0)
    actual.sum().backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())


def test_distance_to_redshift_torch_scalar_and_integer_dtype():
    torch = pytest.importorskip("torch")
    converter = cosmology.DistToZ(numpoints=1000)

    scalar = converter(torch.tensor(100.0, dtype=torch.float64))
    integer = converter(torch.tensor([0, 100, 1000]))

    assert scalar.shape == torch.Size([])
    assert integer.dtype == torch.get_default_dtype()
    torch.testing.assert_close(
        scalar, torch.tensor(converter(100.0), dtype=scalar.dtype))
    torch.testing.assert_close(
        integer,
        torch.tensor(converter(numpy.array([0, 100, 1000])),
                     dtype=integer.dtype),
        rtol=2e-6,
        atol=1e-7,
    )


def test_distance_to_redshift_torch_rejects_unsupported_values():
    torch = pytest.importorskip("torch")
    converter = cosmology.DistToZ(numpoints=128)
    converter.setup_interpolant()

    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            converter(torch.tensor(value))
    with pytest.raises(ValueError, match="precomputed redshift range"):
        converter(torch.tensor(converter.default_maxdist * 1.01))
    with pytest.raises(TypeError, match="real"):
        converter(torch.tensor(100.0 + 0.0j))


@pytest.mark.parametrize("parameter", ["redshift", "luminosity_distance"])
def test_comoving_volume_torch_matches_numpy_and_has_gradient(
    comoving_volume_converters, parameter
):
    torch = pytest.importorskip("torch")
    converter = comoving_volume_converters[parameter]
    redshifts = numpy.array([0.01, 0.2, 1.0, 4.0])
    volumes = converter.cosmology.comoving_volume(redshifts).value
    expected = converter(volumes)

    tensor = torch.tensor(volumes, dtype=torch.float64, requires_grad=True)
    actual = converter(tensor)

    assert isinstance(actual, torch.Tensor)
    assert actual.device == tensor.device
    assert actual.dtype == tensor.dtype
    torch.testing.assert_close(
        actual, torch.as_tensor(expected), rtol=1e-13, atol=0.0
    )
    actual.sum().backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())


def test_comoving_volume_torch_scalar_and_integer_dtype(
    comoving_volume_converters,
):
    torch = pytest.importorskip("torch")
    converter = comoving_volume_converters["redshift"]

    scalar = converter(torch.tensor(5000.0, dtype=torch.float64))
    integer = converter(torch.tensor([5000, 50000]))

    assert scalar.shape == torch.Size([])
    assert integer.dtype == torch.get_default_dtype()
    torch.testing.assert_close(
        scalar, torch.tensor(converter(5000.0), dtype=scalar.dtype)
    )
    torch.testing.assert_close(
        integer,
        torch.tensor(
            converter(numpy.array([5000, 50000])), dtype=integer.dtype
        ),
        rtol=2e-6,
        atol=1e-7,
    )


def test_comoving_volume_torch_rejects_unsupported_values(
    comoving_volume_converters,
):
    torch = pytest.importorskip("torch")
    converter = comoving_volume_converters["redshift"]

    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and > 0"):
            converter(torch.tensor(value))
    with pytest.raises(ValueError, match="precomputed redshift range"):
        converter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="precomputed redshift range"):
        converter(torch.tensor(numpy.exp(converter.default_maxvol) * 1.01))
    with pytest.raises(TypeError, match="real"):
        converter(torch.tensor(5000.0 + 0.0j))
