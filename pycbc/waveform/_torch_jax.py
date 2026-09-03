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
from functools import lru_cache
from numbers import Number
import os
from types import SimpleNamespace

import torch

from .torch_switches import _parse_switch


_ACTIVE_TENSOR = ContextVar("pycbc_torch_waveform_tensor", default=None)
_IDENTITY_TENSOR_FASTPATH_ACTIVE = ContextVar(
    "pycbc_torch_jax_identity_tensor_fastpath",
    default=False,
)
_IDENTITY_TENSOR_FASTPATH_ENV = "PYCBC_TORCH_JAX_IDENTITY_TENSOR_FASTPATH"
_CUDA_DEVICE_SCALARS_ENV = "PYCBC_TORCH_CUDA_DEVICE_SCALARS"
_TORCH_FUNCTION_MODE_ENABLED = getattr(
    torch._C,
    "_is_torch_function_mode_enabled",
    None,
)
_TORCH_DISPATCH_STACK_LENGTH = getattr(
    torch._C,
    "_len_torch_dispatch_stack",
    None,
)


def _identity_tensor_fastpath_enabled():
    """Return whether exact no-op tensor normalization may be elided."""

    value = os.environ.get(_IDENTITY_TENSOR_FASTPATH_ENV)
    return (
        False if value is None else _parse_switch(_IDENTITY_TENSOR_FASTPATH_ENV, value)
    )


def _cuda_device_scalars_enabled():
    """Return whether Python scalars are constructed directly on CUDA."""

    value = os.environ.get(_CUDA_DEVICE_SCALARS_ENV)
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _numeric_sequence_key(value):
    """Return a hashable key for a nested sequence of Python numbers."""

    if isinstance(value, Number):
        return value
    if isinstance(value, (list, tuple)):
        items = tuple(_numeric_sequence_key(item) for item in value)
        if all(item is not None for item in items):
            return items
    return None


@lru_cache(maxsize=256)
def _cached_cuda_sequence(value, device, dtype):
    """Materialize an immutable numeric constant once per device and dtype."""

    return torch.as_tensor(value, device=torch.device(device), dtype=dtype)


@contextmanager
def torch_context(like):
    """Make new constants follow ``like`` for the duration of a model call."""

    identity_fastpath = _identity_tensor_fastpath_enabled()
    token = _ACTIVE_TENSOR.set(like)
    identity_token = _IDENTITY_TENSOR_FASTPATH_ACTIVE.set(identity_fastpath)
    try:
        yield
    finally:
        _IDENTITY_TENSOR_FASTPATH_ACTIVE.reset(identity_token)
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
    if (
        _IDENTITY_TENSOR_FASTPATH_ACTIVE.get()
        and type(value) is torch.Tensor
        and (value.dtype.is_floating_point or value.dtype.is_complex)
        and _TORCH_FUNCTION_MODE_ENABLED is not None
        and not _TORCH_FUNCTION_MODE_ENABLED()
        and _TORCH_DISPATCH_STACK_LENGTH is not None
        and _TORCH_DISPATCH_STACK_LENGTH() == 0
        and (
            reference is None
            or (
                type(reference) is torch.Tensor
                and (reference.dtype.is_floating_point or reference.dtype.is_complex)
                and value.device == reference.device
                and value.dtype == reference.dtype
            )
        )
    ):
        # ``Tensor.to`` returns the original plain tensor for this exact
        # dtype/device contract.  The legacy reference search and settings
        # extraction therefore cannot affect the result; avoid repeating all
        # three Python/dispatcher calls at hundreds of scalar call sites.
        return value
    if reference is None:
        reference = _first_tensor(value)
    device, dtype = _settings(reference)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if (
        device.type == "cuda"
        and _cuda_device_scalars_enabled()
        and isinstance(value, Number)
    ):
        # ``as_tensor`` stages Python scalars through pageable host memory.
        # ``scalar_tensor`` constructs the same typed value on-device and is
        # safe to use while a CUDA graph is being captured.
        return torch.scalar_tensor(value, device=device, dtype=dtype)
    if device.type == "cuda" and _cuda_device_scalars_enabled():
        sequence_key = _numeric_sequence_key(value)
        if isinstance(sequence_key, tuple):
            return _cached_cuda_sequence(sequence_key, str(device), dtype)
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


def _foreach_cbrt_supported(values):
    """Return whether three scalar roots can use the exact foreach lane."""

    if len(values) != 3:
        return False
    reference = values[0]
    return type(reference) is torch.Tensor and all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype in (torch.float32, torch.float64)
        and value.dtype == reference.dtype
        and value.device == reference.device
        and value.device.type in ("cpu", "cuda")
        and not value.is_conj()
        and not value.is_neg()
        for value in values
    )


def _foreach_cbrt(values, *, prevalidated=False):
    """Evaluate three independent scalar cube roots in fixed exact order.

    Each lane retains :func:`_cbrt`'s sign, absolute value, power, and
    multiplication sequence.  Unsupported tensors and unavailable foreach
    kernels use the ordinary per-value implementation.  ``prevalidated`` is
    reserved for call sites that have already established the same scalar
    tensor invariants before constructing ``values``.
    """

    values = tuple(values)
    if len(values) != 3 or (
        not prevalidated and not _foreach_cbrt_supported(values)
    ):
        return tuple(_cbrt(value) for value in values)
    try:
        signs = torch._foreach_sign(values)
        magnitudes = torch._foreach_abs(values)
        roots = torch._foreach_pow(magnitudes, 1.0 / 3.0)
        return tuple(torch._foreach_mul(signs, roots))
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return tuple(_cbrt(value) for value in values)


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
    foreach_cbrt = staticmethod(_foreach_cbrt)
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
