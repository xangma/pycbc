# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain consumers of the shared optional tensor protocol."""

import numpy
import pytest

from pycbc import boundaries, transforms
from pycbc.coordinates import base as coordinates

torch = pytest.importorskip("torch")


class UserTensor(torch.Tensor):
    """A tensor whose module name is unrelated to Torch."""


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    return request.param


def test_coordinate_subclass_mixed_broadcasting_and_gradients(device):
    original = torch.tensor([[3.], [4.]], device=device, requires_grad=True)
    tensor = original.as_subclass(UserTensor)
    y = numpy.array([4., 3.], dtype=numpy.float64)
    actual = coordinates.cartesian_to_spherical_rho(tensor, y, 0)
    expected = torch.sqrt(original**2 + original.new_tensor(y)**2)
    assert actual.device == original.device
    assert actual.dtype == original.dtype
    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    gradient = (original / expected).sum(dim=1, keepdim=True)
    torch.testing.assert_close(original.grad, gradient)


def test_coordinate_integer_inputs_use_default_float(device):
    actual = coordinates.cartesian_to_spherical_rho(
        torch.tensor([3, 5], device=device), 4, 0
    )
    assert actual.dtype == torch.get_default_dtype()
    torch.testing.assert_close(actual, actual.new_tensor([5., 41.**0.5]))


def test_expression_and_boundary_subclasses_keep_autograd(device):
    original = torch.tensor([-2.5, 0.25, 2.5], device=device,
                            requires_grad=True)
    tensor = original.as_subclass(UserTensor)
    custom = transforms.CustomTransform(
        ["x"], ["result"], {"result": "sin(x) + x**2"}
    )
    actual = custom.transform({"x": tensor})["result"]
    assert actual.device == original.device
    torch.testing.assert_close(actual, torch.sin(original) + original**2)
    grad = torch.autograd.grad(actual.sum(), original, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.cos(original) + 2 * original)

    bounds = boundaries.Bounds(-1., 1., btype_min="reflected",
                               btype_max="reflected")
    reflected = bounds.apply_conditions(tensor)
    assert reflected.device == original.device
    torch.testing.assert_close(reflected,
                               original.new_tensor([0.5, 0.25, -0.5]))
    reflected.sum().backward()
    torch.testing.assert_close(original.grad,
                               original.new_tensor([-1., 1., -1.]))
