# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Small JAX-like helpers used by Torch-native waveform ports.

This is deliberately not a general JAX compatibility layer.  It only exposes
the elementwise, linear-algebra, and scalar-autodiff operations needed by the
independently validated IMRPhenomXAS coefficient implementation.  Keeping the
surface small makes device and dtype behavior explicit and prevents a runtime
dependency on JAX.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import torch


_ACTIVE_TENSOR = ContextVar("pycbc_torch_waveform_tensor", default=None)


@contextmanager
def torch_context(like):
    """Make new constants follow ``like`` for the duration of a model call."""

    token = _ACTIVE_TENSOR.set(like)
    try:
        yield
    finally:
        _ACTIVE_TENSOR.reset(token)


def _settings(reference=None):
    reference = reference if reference is not None else _ACTIVE_TENSOR.get()
    if isinstance(reference, torch.Tensor):
        dtype = reference.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float64
        return reference.device, dtype
    return torch.device("cpu"), torch.float64


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _as_tensor(value, reference=None):
    if reference is None:
        reference = _first_tensor(value)
    device, dtype = _settings(reference)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _array(value):
    reference = _ACTIVE_TENSOR.get()
    if reference is None:
        reference = _first_tensor(value)
    if isinstance(value, torch.Tensor):
        return _as_tensor(value, reference)
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, (torch.Tensor, list, tuple)) for item in value
    ):
        return torch.stack(
            [
                _array(item)
                if isinstance(item, (list, tuple))
                else _as_tensor(item, reference)
                for item in value
            ]
        )
    return _as_tensor(value, reference)


def _binary_tensors(left, right):
    reference = _first_tensor(left)
    if reference is None:
        reference = _first_tensor(right)
    return _as_tensor(left, reference), _as_tensor(right, reference)


def _maximum(left, right):
    left, right = _binary_tensors(left, right)
    return torch.maximum(left, right)


def _where(condition, left, right):
    reference = _first_tensor(left)
    if reference is None:
        reference = _first_tensor(right)
    condition = torch.as_tensor(condition, device=_settings(reference)[0])
    return torch.where(
        condition,
        _as_tensor(left, reference),
        _as_tensor(right, reference),
    )


def _heaviside(value, at_zero):
    value = _as_tensor(value)
    at_zero = _as_tensor(at_zero, value)
    # ``torch.heaviside`` is not implemented by MPS. Comparisons and where are
    # device-native on all supported Torch backends and have the same values.
    return torch.where(
        value > 0.0,
        torch.ones_like(value),
        torch.where(value < 0.0, torch.zeros_like(value), at_zero),
    )


def _cbrt(value):
    value = _as_tensor(value)
    return torch.sign(value) * torch.abs(value).pow(1.0 / 3.0)


def _unary(function, value):
    return function(_as_tensor(value))


def _copysign(left, right):
    left, right = _binary_tensors(left, right)
    return torch.copysign(left, right)


def _tree_detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_tree_detach(item) for item in value)
    if isinstance(value, list):
        return [_tree_detach(item) for item in value]
    return value


def _differentiable_argument(value):
    value = _as_tensor(value)
    return value.detach().clone().requires_grad_(True)


def _value_and_grad(function, *, has_aux=False):
    def evaluate(value, *args, **kwargs):
        with torch.enable_grad():
            argument = _differentiable_argument(value)
            result = function(argument, *args, **kwargs)
            if has_aux:
                scalar, auxiliary = result
            else:
                scalar, auxiliary = result, None
            gradient = torch.autograd.grad(
                scalar,
                argument,
                grad_outputs=(
                    torch.ones_like(scalar) if scalar.ndim else None
                ),
                create_graph=False,
            )[0]
        scalar = scalar.detach()
        gradient = gradient.detach()
        if has_aux:
            return (scalar, _tree_detach(auxiliary)), gradient
        return scalar, gradient

    return evaluate


def _grad(function):
    value_and_grad = _value_and_grad(function)

    def evaluate(value, *args, **kwargs):
        return value_and_grad(value, *args, **kwargs)[1]

    return evaluate


class _TorchNumpy:
    array = staticmethod(_array)
    asarray = staticmethod(_as_tensor)
    arctan = staticmethod(lambda value: _unary(torch.atan, value))
    cbrt = staticmethod(_cbrt)
    copysign = staticmethod(_copysign)
    cos = staticmethod(lambda value: _unary(torch.cos, value))
    exp = staticmethod(lambda value: _unary(torch.exp, value))
    fabs = staticmethod(lambda value: _unary(torch.abs, value))
    heaviside = staticmethod(_heaviside)
    log = staticmethod(lambda value: _unary(torch.log, value))
    maximum = staticmethod(_maximum)
    sign = staticmethod(lambda value: _unary(torch.sign, value))
    sqrt = staticmethod(lambda value: _unary(torch.sqrt, value))
    where = staticmethod(_where)
    linalg = SimpleNamespace(solve=torch.linalg.solve)

    @staticmethod
    def ones(shape):
        device, dtype = _settings()
        return torch.ones(shape, device=device, dtype=dtype)


jnp = _TorchNumpy()
jax = SimpleNamespace(
    grad=_grad,
    value_and_grad=_value_and_grad,
)
