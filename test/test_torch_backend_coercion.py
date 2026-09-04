# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared tensor detection/coercion contracts used by domain modules."""

import subprocess
import sys

import numpy
import pytest

from pycbc.types import backend


def test_host_protocol_does_not_import_torch():
    code = """
import builtins
import numpy
import sys
original_import = builtins.__import__
def reject_torch(name, *args, **kwargs):
    if name.partition('.')[0] == 'torch':
        raise AssertionError('host operations attempted to import torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_torch
from pycbc.types.backend import coerce_torch_values, torch_module_for
values = (numpy.array([1., 2.]), 3.)
module, converted = coerce_torch_values(*values)
assert module is None and converted[0] is values[0]
assert torch_module_for(values[0]) is None
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_mixed_coercion_preserves_device_shapes_and_gradients(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    from pycbc.conversions_torch import broadcast_values

    first = torch.tensor([[1.], [2.]], device=device, requires_grad=True)
    second = torch.tensor([3., 4., 5.], requires_grad=True)
    host = numpy.array([6., 7., 8.], dtype=numpy.float64)
    module, converted = backend.coerce_torch_values(first, second, host)
    assert module is torch
    assert [value.shape for value in converted] == [(2, 1), (3,), (3,)]
    assert all(value.device == first.device for value in converted)
    assert all(value.dtype == first.dtype for value in converted)
    _, broadcast = broadcast_values(first, second, host)
    assert all(value.shape == (2, 3) for value in broadcast)
    sum(value.sum() for value in broadcast).backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, 3.))
    torch.testing.assert_close(second.grad, torch.full_like(second, 2.))


@pytest.mark.parametrize("dtype", ["int64", "bool", "complex64", "float64"])
def test_coercion_dtype_and_public_storage(dtype):
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1, 0], dtype=getattr(torch, dtype))

    class PublicStorage:
        backend = "torch"

        def backend_array(self):
            return tensor

    module, values = backend.coerce_torch_values(PublicStorage(), [2, 3])
    expected = (torch.get_default_dtype() if dtype in ("int64", "bool")
                else tensor.dtype)
    assert module is torch
    assert all(value.dtype == expected for value in values)
    torch.testing.assert_close(values[0], tensor.to(dtype=expected))


def test_user_tensor_subclass_dispatch_and_gradient():
    torch = pytest.importorskip("torch")
    from pycbc import conversions

    class UserTensor(torch.Tensor):
        pass

    original = torch.tensor([3., 5.], requires_grad=True)
    value = original.as_subclass(UserTensor)
    assert backend.torch_module_for(value) is torch
    assert backend.backend_name(value) == "torch"
    result = conversions.primary_mass(value, 2.)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, original)
    result.sum().backward()
    torch.testing.assert_close(original.grad, torch.ones_like(original))


def test_type_cache_and_unhashable_metaclass():
    torch = pytest.importorskip("torch")
    classify = backend._torch_module_for_type
    classify.cache_clear()
    assert backend.torch_module_for(torch.tensor(1)) is torch
    before = classify.cache_info()
    assert backend.torch_module_for(
        torch.tensor(1., requires_grad=True)
    ) is torch
    assert classify.cache_info().hits == before.hits + 1

    class UnhashableType(type):
        __hash__ = None

    class HostValue(metaclass=UnhashableType):
        pass

    assert backend.torch_module_for(HostValue()) is None
