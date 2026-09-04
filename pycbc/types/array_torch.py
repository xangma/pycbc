# Copyright (C) 2025
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch backend for the PyCBC Array type.

All torch-specific logic is contained in this module to keep the core array
implementation scheme-agnostic.
"""

import operator
import warnings

import numpy as np
import torch

import pycbc.scheme as _scheme

_NUMPY_TRAPEZOID = getattr(np, "trapezoid", None)
if _NUMPY_TRAPEZOID is None:
    _NUMPY_TRAPEZOID = np.trapz

if not hasattr(np, "exceptions"):
    class _NumpyExceptionsShim:
        AxisError = getattr(np, "AxisError", ValueError)
        ComplexWarning = getattr(np, "ComplexWarning", UserWarning)
    np.exceptions = _NumpyExceptionsShim()

_TORCH_UINT16 = getattr(torch, "uint16", None)
_TORCH_UINT32 = getattr(torch, "uint32", None)

_NUMPY_TO_TORCH = {
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
}
if _TORCH_UINT32 is not None:
    _NUMPY_TO_TORCH[np.dtype(np.uint32)] = _TORCH_UINT32
_TORCH_TO_NUMPY = {v: k for k, v in _NUMPY_TO_TORCH.items()}


def _torch_rint(tensor):
    """Apply NumPy's component-wise complex rounding semantics."""
    if tensor.is_complex():
        return torch.complex(torch.round(tensor.real), torch.round(tensor.imag))
    return torch.round(tensor)


def _torch_round_decimals(tensor, decimals):
    """Apply NumPy's scale/rint/unscale decimal-rounding algorithm."""
    if tensor.is_complex():
        return torch.complex(
            _torch_round_decimals(tensor.real, decimals),
            _torch_round_decimals(tensor.imag, decimals),
        )
    if decimals == 0:
        return _torch_rint(tensor)
    factor = 10.0 ** abs(decimals)
    if decimals > 0:
        return _torch_rint(tensor * factor) / factor
    return _torch_rint(tensor / factor) * factor


def _torch_cbrt(tensor):
    """Apply a real cube root while preserving the sign of zero."""
    magnitude = torch.abs(tensor).pow(1.0 / 3.0)
    return torch.copysign(magnitude, tensor)


def _torch_cummax(tensor, dim):
    """Return cumulative maxima without Torch's accompanying indices."""
    return torch.cummax(tensor, dim=dim).values


def _torch_cummin(tensor, dim):
    """Return cumulative minima without Torch's accompanying indices."""
    return torch.cummin(tensor, dim=dim).values


_TORCH_UNARY_NUMPY_UFUNCS = {
    np.isfinite: torch.isfinite,
    np.isnan: torch.isnan,
    np.isinf: torch.isinf,
    np.logical_not: torch.logical_not,
    np.invert: torch.bitwise_not,
    np.absolute: torch.abs,
    np.fabs: torch.abs,
    np.sqrt: torch.sqrt,
    np.exp: torch.exp,
    np.exp2: torch.exp2,
    np.expm1: torch.expm1,
    np.log: torch.log,
    np.log10: torch.log10,
    np.log2: torch.log2,
    np.log1p: torch.log1p,
    np.sin: torch.sin,
    np.cos: torch.cos,
    np.tan: torch.tan,
    np.sinh: torch.sinh,
    np.cosh: torch.cosh,
    np.tanh: torch.tanh,
    np.arcsinh: torch.asinh,
    np.arccosh: torch.acosh,
    np.arctanh: torch.atanh,
    np.arcsin: torch.asin,
    np.arccos: torch.acos,
    np.arctan: torch.atan,
    np.sign: torch.sign,
    np.cbrt: _torch_cbrt,
    np.deg2rad: torch.deg2rad,
    np.radians: torch.deg2rad,
    np.rad2deg: torch.rad2deg,
    np.degrees: torch.rad2deg,
    np.floor: torch.floor,
    np.ceil: torch.ceil,
    np.trunc: torch.trunc,
    np.rint: _torch_rint,
    np.conjugate: torch.conj,
    np.negative: torch.neg,
    np.positive: torch.positive,
    np.square: torch.square,
    np.reciprocal: torch.reciprocal,
}
_TORCH_BOOLEAN_UNARY_NUMPY_UFUNCS = {
    np.isfinite,
    np.isnan,
    np.isinf,
    np.logical_not,
}
_TORCH_BITWISE_UNARY_NUMPY_UFUNCS = {np.invert}
_TORCH_REAL_UNARY_NUMPY_UFUNCS = {
    np.fabs,
    np.sign,
    np.cbrt,
    np.deg2rad,
    np.radians,
    np.rad2deg,
    np.degrees,
    np.floor,
    np.ceil,
    np.trunc,
}
_TORCH_BINARY_NUMPY_UFUNCS = {
    np.logical_and: torch.logical_and,
    np.logical_or: torch.logical_or,
    np.logical_xor: torch.logical_xor,
    np.bitwise_and: torch.bitwise_and,
    np.bitwise_or: torch.bitwise_or,
    np.bitwise_xor: torch.bitwise_xor,
    np.add: torch.add,
    np.subtract: torch.subtract,
    np.multiply: torch.multiply,
    np.divide: torch.true_divide,
    np.power: torch.pow,
    np.maximum: torch.maximum,
    np.minimum: torch.minimum,
    np.arctan2: torch.atan2,
    np.hypot: torch.hypot,
    np.fmod: torch.fmod,
    np.remainder: torch.remainder,
    np.copysign: torch.copysign,
    np.logaddexp: torch.logaddexp,
    np.logaddexp2: torch.logaddexp2,
}
_TORCH_LOGICAL_BINARY_NUMPY_UFUNCS = {
    np.logical_and,
    np.logical_or,
    np.logical_xor,
}
_TORCH_BITWISE_BINARY_NUMPY_UFUNCS = {
    np.bitwise_and,
    np.bitwise_or,
    np.bitwise_xor,
}
_TORCH_COMPARISON_NUMPY_UFUNCS = {
    np.equal: "eq",
    np.not_equal: "ne",
    np.less: "lt",
    np.less_equal: "le",
    np.greater: "gt",
    np.greater_equal: "ge",
}
_TORCH_REAL_BINARY_NUMPY_UFUNCS = {
    np.maximum,
    np.minimum,
    np.arctan2,
    np.hypot,
    np.fmod,
    np.remainder,
    np.copysign,
    np.logaddexp,
    np.logaddexp2,
}
_TORCH_REDUCE_NUMPY_UFUNCS = {
    np.add: torch.sum,
    np.multiply: torch.prod,
    np.maximum: torch.max,
    np.minimum: torch.min,
    np.logical_or: torch.any,
    np.logical_and: torch.all,
}
_TORCH_REAL_REDUCE_NUMPY_UFUNCS = {np.maximum, np.minimum}
_TORCH_LOGICAL_REDUCE_NUMPY_UFUNCS = {np.logical_or, np.logical_and}
_TORCH_ACCUMULATE_NUMPY_UFUNCS = {
    np.add: torch.cumsum,
    np.multiply: torch.cumprod,
    np.maximum: _torch_cummax,
    np.minimum: _torch_cummin,
}
_TORCH_REAL_ACCUMULATE_NUMPY_UFUNCS = {np.maximum, np.minimum}


def _normalized_reduction_axes(axis, ndim):
    """Normalize a NumPy reduction axis without using host array data."""
    if axis is None:
        return tuple(range(ndim))
    is_tuple = isinstance(axis, tuple)
    raw_axes = axis if is_tuple else (axis,)
    if ndim == 0:
        if not raw_axes:
            return ()
        if not is_tuple and operator.index(raw_axes[0]) in (0, -1):
            return ()
        raise IndexError("axis is out of bounds for array of dimension 0")

    axes = []
    for raw_axis in raw_axes:
        normalized = operator.index(raw_axis)
        if normalized < 0:
            normalized += ndim
        if normalized < 0 or normalized >= ndim:
            raise IndexError("axis is out of bounds")
        if normalized in axes:
            raise ValueError("duplicate value in 'axis'")
        axes.append(normalized)
    return tuple(axes)


def _reduction_output_shape(shape, axes, keepdims):
    """Return the shape produced by reducing ``axes``."""
    return tuple(
        1 if keepdims and dimension in axes else size
        for dimension, size in enumerate(shape)
        if keepdims or dimension not in axes
    )


def _torch_device():
    """Return the torch.device for the current scheme."""
    state = _scheme.mgr.state
    if hasattr(state, "torch_device"):
        return state.torch_device
    return torch.device("cpu")


def _device_matches_active(tensor):
    """Return whether a tensor resides on the active Torch device."""
    active = _torch_device()
    device = tensor.device
    return (
        device.type == active.type
        and (active.index is None or device.index == active.index)
    )


def _torch_dtype(dtype):
    """Normalize dtype to a torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _NUMPY_TO_TORCH[np.dtype(dtype)]
    except Exception as exc:  # pylint: disable=broad-except
        raise TypeError(f"{dtype} is not supported by the Torch backend") from exc


def _numpy_dtype(torch_dtype):
    """Convert torch dtype to numpy dtype."""
    try:
        return _TORCH_TO_NUMPY[torch_dtype]
    except KeyError as exc:
        raise TypeError(f"{torch_dtype} is not supported by the Torch backend") from exc


def _ensure_supported(device, torch_dtype):
    """Validate requested device / dtype combinations."""
    if device.type == "mps":
        if torch_dtype not in (
                torch.bool, torch.float32, torch.float16, torch.complex64,
                torch.int32, torch.int64):
            raise TypeError(
                "MPS backend only supports bool/float16/float32/complex64 "
                "and int32/int64 tensors"
            )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device requested but unavailable")
    return torch_dtype


def _accumulation_dtype(tensor, *, complex_result=False):
    """Choose the highest-precision reduction dtype supported on-device."""
    if tensor.device.type == "mps":
        return torch.complex64 if complex_result else torch.float32
    return torch.complex128 if complex_result else torch.float64


def _unwrap(other):
    """Return bare torch tensor for arithmetic helpers."""
    if isinstance(other, TorchArrayData):
        return other.tensor
    return other


def _tensor_from_any(other, device):
    """Cast different inputs to a torch tensor and numpy dtype tuple."""
    if isinstance(other, TorchArrayData):
        return other.tensor, other.dtype
    if isinstance(other, torch.Tensor):
        return other.to(device=device), _numpy_dtype(other.dtype)
    tensor = torch.as_tensor(other, device=device)
    return tensor, _numpy_dtype(tensor.dtype)


def _promote_tensors(tensor_a, dtype_a, tensor_b, dtype_b):
    target_np_dtype = np.result_type(np.dtype(dtype_a), np.dtype(dtype_b))
    target_torch = _torch_dtype(target_np_dtype)
    return (
        tensor_a.to(dtype=target_torch),
        tensor_b.to(dtype=target_torch),
        target_np_dtype,
    )


def _comparison_dtype(dtype_a, target_np, device, other_values=None):
    """Return a Torch dtype that preserves NumPy comparison semantics."""
    target_np = np.dtype(target_np)

    # Torch does not implement ordering for uint32. Promoting both operands
    # to int64 is lossless and preserves NumPy's result for uint32 values.
    if target_np == np.dtype(np.uint32):
        target_np = np.dtype(np.int64)

    try:
        return _ensure_supported(device, _torch_dtype(target_np))
    except TypeError:
        # MPS cannot represent NumPy's usual float64 scalar promotion. If
        # the host operand is exactly representable at the array's precision,
        # comparing at that precision has the same result and stays on-device.
        if device.type != "mps" or other_values is None:
            raise
        values = np.asarray(other_values)
        with np.errstate(all="ignore"):
            cast_values = values.astype(dtype_a)
            restored_values = cast_values.astype(values.dtype)
        if not np.array_equal(values, restored_values, equal_nan=True):
            raise
        return _ensure_supported(device, _torch_dtype(dtype_a))


def _comparison_tensors(tensor_a, dtype_a, other):
    """Prepare operands for a NumPy-compatible Torch comparison."""
    other = _unwrap(other)
    other_values = None
    outside_range = None

    if isinstance(other, torch.Tensor):
        tensor_b = other
        if other.dtype == torch.bool:
            dtype_b = np.dtype(np.bool_)
        elif other.dtype == torch.float16:
            dtype_b = np.dtype(np.float16)
        else:
            dtype_b = _numpy_dtype(other.dtype)
        target_np = np.result_type(np.dtype(dtype_a), dtype_b)
    else:
        other_values = np.asarray(other)
        if other_values.dtype.kind not in "biufc":
            raise TypeError("Torch comparisons require numeric operands")
        if other_values.ndim and not other_values.flags.c_contiguous:
            other_values = np.ascontiguousarray(other_values)

        # NumPy uses weak promotion for Python scalars, while arrays and
        # explicit NumPy scalars carry their own dtype into the comparison.
        if np.isscalar(other):
            target_np = np.result_type(np.dtype(dtype_a), other)
        else:
            target_np = np.result_type(
                np.dtype(dtype_a), other_values.dtype
            )

        # Python integers outside an integer array's range are compared by
        # value in NumPy instead of first being wrapped to the array dtype.
        if (
            isinstance(other, int)
            and not isinstance(other, bool)
            and np.dtype(dtype_a).kind in "iu"
        ):
            limits = np.iinfo(dtype_a)
            if other < limits.min:
                outside_range = -1
            elif other > limits.max:
                outside_range = 1

    target = _comparison_dtype(
        dtype_a,
        target_np,
        tensor_a.device,
        other_values=other_values,
    )
    if outside_range is not None:
        return tensor_a, None, outside_range
    if not isinstance(other, torch.Tensor):
        tensor_b = torch.as_tensor(other_values, dtype=target)
    return (
        tensor_a.to(dtype=target),
        tensor_b.to(device=tensor_a.device, dtype=target),
        None,
    )


def _resolve_for_numpy(tensor):
    """Ensure tensors with a conjugate bit are materialized before numpy conversion."""
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    return tensor


def _binary_ufunc_tensors(inputs, reference):
    """Prepare binary operands with NumPy scalar-promotion semantics."""
    dtype_inputs = []
    shapes = []
    for operand in inputs:
        if isinstance(operand, TorchArrayData):
            if np.dtype(operand.dtype).kind not in "fc":
                return None
            if operand.tensor.device != reference.tensor.device:
                return None
            dtype_inputs.append(np.dtype(operand.dtype))
            shapes.append(operand.shape)
        elif np.isscalar(operand):
            if np.asarray(operand).dtype.kind not in "biufc":
                return None
            dtype_inputs.append(operand)
            shapes.append(())
        else:
            return None

    try:
        target_np = np.result_type(*dtype_inputs)
        if target_np.kind not in "fc":
            return None
        target_torch = _ensure_supported(
            reference.tensor.device, _torch_dtype(target_np)
        )
        if tuple(torch.broadcast_shapes(*shapes)) != reference.shape:
            return None
        tensors = []
        for operand in inputs:
            if isinstance(operand, TorchArrayData):
                tensor = operand.tensor.to(dtype=target_torch)
            else:
                tensor = torch.as_tensor(
                    operand,
                    dtype=target_torch,
                    device=reference.tensor.device,
                )
            tensors.append(tensor)
    except (OverflowError, TypeError, ValueError, RuntimeError):
        return None
    return tensors


def _logical_ufunc_tensors(inputs, reference):
    """Prepare numeric operands for boolean Torch ufuncs."""
    tensors = []
    shapes = []
    for operand in inputs:
        if isinstance(operand, TorchArrayData):
            if (
                np.dtype(operand.dtype).kind not in "biufc"
                or operand.tensor.device != reference.tensor.device
            ):
                return None
            tensor = operand.tensor
            shapes.append(operand.shape)
        elif np.isscalar(operand):
            if np.asarray(operand).dtype.kind not in "biufc":
                return None
            tensor = torch.as_tensor(
                bool(operand),
                dtype=torch.bool,
                device=reference.tensor.device,
            )
            shapes.append(())
        else:
            return None
        tensors.append(tensor)

    try:
        if tuple(torch.broadcast_shapes(*shapes)) != reference.shape:
            return None
        return [tensor.to(dtype=torch.bool) for tensor in tensors]
    except (TypeError, ValueError, RuntimeError):
        return None


def _comparison_ufunc_result(inputs, reference, operation):
    """Evaluate an elementwise comparison without copying samples to NumPy."""
    shapes = []
    for operand in inputs:
        if isinstance(operand, TorchArrayData):
            if operand.tensor.device != reference.tensor.device:
                return None
            shapes.append(operand.shape)
        elif np.isscalar(operand):
            if np.asarray(operand).dtype.kind not in "biufc":
                return None
            shapes.append(())
        else:
            return None

    try:
        if tuple(torch.broadcast_shapes(*shapes)) != reference.shape:
            return None
        if isinstance(inputs[0], TorchArrayData):
            tensor = inputs[0].comparison(inputs[1], operation)
        else:
            reverse = {
                "eq": "eq",
                "ne": "ne",
                "lt": "gt",
                "le": "ge",
                "gt": "lt",
                "ge": "le",
            }
            tensor = inputs[1].comparison(inputs[0], reverse[operation])
    except (OverflowError, TypeError, ValueError, RuntimeError):
        return None
    return reference._wrap(tensor)


def _bitwise_ufunc_tensors(inputs, reference):
    """Prepare boolean or signed-integer operands for Torch bitwise ufuncs."""
    dtype_inputs = []
    shapes = []
    for operand in inputs:
        if isinstance(operand, TorchArrayData):
            if (
                np.dtype(operand.dtype).kind not in "biu"
                or operand.tensor.device != reference.tensor.device
            ):
                return None
            dtype_inputs.append(np.dtype(operand.dtype))
            shapes.append(operand.shape)
        elif np.isscalar(operand):
            if np.asarray(operand).dtype.kind not in "biu":
                return None
            dtype_inputs.append(operand)
            shapes.append(())
        else:
            return None

    try:
        target_np = np.result_type(*dtype_inputs)
        if target_np.kind not in "biu":
            return None
        target_torch = _ensure_supported(
            reference.tensor.device, _torch_dtype(target_np)
        )
        execution_torch = (
            torch.int64 if target_np.kind == "u" else target_torch
        )
        if tuple(torch.broadcast_shapes(*shapes)) != reference.shape:
            return None
        tensors = []
        for operand in inputs:
            if isinstance(operand, TorchArrayData):
                tensor = operand.tensor.to(dtype=execution_torch)
            else:
                if target_np.kind == "u":
                    # Match NumPy's overflow checks for weak Python scalars
                    # before using int64 as Torch's uint32 execution dtype.
                    np.asarray(operand, dtype=target_np)
                tensor = torch.as_tensor(
                    operand,
                    dtype=execution_torch,
                    device=reference.tensor.device,
                )
            tensors.append(tensor)
        return tensors, target_torch
    except (OverflowError, TypeError, ValueError, RuntimeError):
        return None


class TorchArrayData:
    """Lightweight wrapper around a torch tensor with numpy dtype semantics."""

    __slots__ = ("tensor", "dtype")
    __array_priority__ = 100.0
    backend = "torch"

    def __init__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("TorchArrayData requires a torch.Tensor")
        self.tensor = tensor
        self.dtype = _numpy_dtype(tensor.dtype)

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def device(self):
        return self.tensor.device

    @property
    def backend_array(self):
        """Return storage through the public PyCBC backend protocol."""
        return self.tensor

    def _wrap(self, tensor):
        return TorchArrayData(tensor)

    def _set_tensor(self, tensor):
        self.tensor = tensor
        self.dtype = _numpy_dtype(tensor.dtype)
        return self

    def to_cuda_async(self, stream=None, device="cuda"):
        return to_cuda_async(self, stream=stream, device=device)

    def _promote_with(self, other):
        other_t, other_np = _tensor_from_any(other, self.tensor.device)
        return _promote_tensors(self.tensor, self.dtype, other_t, other_np)

    def comparison(self, other, operation):
        """Return an elementwise boolean comparison as a Torch tensor."""
        functions = {
            "eq": torch.eq,
            "lt": torch.lt,
            "le": torch.le,
            "ne": torch.ne,
            "gt": torch.gt,
            "ge": torch.ge,
        }
        try:
            function = functions[operation]
        except KeyError as exc:
            raise ValueError(f"Unknown comparison operation {operation}") from exc

        left, right, outside_range = _comparison_tensors(
            self.tensor, self.dtype, other
        )
        if outside_range is not None:
            if operation == "eq":
                value = False
            elif operation == "ne":
                value = True
            elif outside_range > 0:
                value = operation in ("lt", "le")
            else:
                value = operation in ("gt", "ge")
            return torch.full(
                self.tensor.shape,
                value,
                dtype=torch.bool,
                device=self.tensor.device,
            )
        if operation in ("eq", "ne"):
            return function(left, right)
        if not left.is_complex():
            return function(left, right)

        # NumPy orders complex values lexicographically: compare the real
        # components first, then the imaginary components when they match.
        strict = torch.lt if operation in ("lt", "le") else torch.gt
        return strict(left.real, right.real) | (
            torch.eq(left.real, right.real)
            & function(left.imag, right.imag)
        )

    def array_equal(self, other):
        """Return exact equality without copying either tensor to the host."""
        if (
            not isinstance(other, TorchArrayData)
            or self.tensor.device != other.tensor.device
        ):
            return NotImplemented
        return torch.equal(self.tensor, other.tensor)

    def almost_equal_elem(self, other, tol, relative):
        """Reduce an elementwise tolerance comparison on the Torch device."""
        if (
            not isinstance(other, TorchArrayData)
            or self.tensor.device != other.tensor.device
            or np.dtype(self.dtype).kind not in "fc"
        ):
            return NotImplemented
        try:
            difference = torch.abs(self.tensor - other.tensor)
            limit = tol * torch.abs(self.tensor) if relative else tol
            return bool(torch.all(difference <= limit).item())
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def almost_equal_norm(self, other, tol, relative):
        """Reduce a normwise tolerance comparison on the Torch device."""
        if (
            not isinstance(other, TorchArrayData)
            or self.tensor.device != other.tensor.device
            or np.dtype(self.dtype).kind not in "fc"
        ):
            return NotImplemented
        try:
            difference_magnitude = torch.abs(self.tensor - other.tensor)
            difference_norm = torch.sqrt(
                torch.sum(difference_magnitude * difference_magnitude)
            )
            limit = tol
            if relative:
                magnitude = torch.abs(self.tensor)
                limit = tol * torch.sqrt(torch.sum(magnitude * magnitude))
            return bool((difference_norm <= limit).item())
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def numpy_variance(self, axes, dtype=None, ddof=0, keepdims=False):
        """Evaluate NumPy's two-pass variance without a host array copy."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        input_dtype = np.dtype(self.dtype)
        target_dtype = (
            np.dtype(np.float64)
            if dtype is None and input_dtype.kind in "biu"
            else input_dtype if dtype is None else np.dtype(dtype)
        )
        if target_dtype.kind not in "fc":
            return NotImplemented
        if input_dtype.kind == "c" and target_dtype.kind != "c":
            return NotImplemented
        try:
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_dtype)
            )
            tensor = self.tensor.to(dtype=target_torch)
            count = 1
            for dimension in axes:
                count *= tensor.shape[dimension]
            if axes:
                average = torch.mean(tensor, dim=axes, keepdim=True)
            else:
                average = tensor
            centered = tensor - average
            if input_dtype.kind == "c":
                squared = torch.view_as_real(centered).square().sum(dim=-1)
                if dtype is not None:
                    squared = squared.to(dtype=target_torch)
            else:
                squared = centered * centered
            denominator = count - ddof
            if denominator < 0:
                denominator = 0
            if axes:
                result = torch.sum(
                    squared, dim=axes, keepdim=keepdims
                ) / denominator
            else:
                result = squared / denominator
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if dtype is None and input_dtype.kind == "c":
            result_dtype = np.empty((), dtype=input_dtype).real.dtype
        else:
            result_dtype = target_dtype
        if result.ndim:
            return self._wrap(result)
        return result_dtype.type(result.item())

    def numpy_arg_reduce(self, operation, axis=None, keepdims=False):
        """Evaluate a real NumPy argument reduction on the Torch device."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        input_dtype = np.dtype(self.dtype)
        if input_dtype.kind not in "bfiu":
            return NotImplemented

        if axis is None:
            reduction_size = self.tensor.numel()
            result_shape = (1,) * self.tensor.ndim if keepdims else ()
        else:
            reduction_size = self.tensor.shape[axis]
            result_shape = tuple(
                1 if keepdims and dimension == axis else size
                for dimension, size in enumerate(self.tensor.shape)
                if keepdims or dimension != axis
            )
        if reduction_size == 0:
            raise ValueError(
                f"attempt to get arg{operation} of an empty sequence"
            )

        result_size = 1
        for size in result_shape:
            result_size *= size
        if result_shape and result_size == 0:
            return self._wrap(torch.empty(
                result_shape, dtype=torch.int64, device=self.tensor.device
            ))

        tensor = self.tensor
        if input_dtype.kind == "b":
            tensor = tensor.to(dtype=torch.float32)
        elif input_dtype.kind == "u":
            tensor = tensor.to(dtype=torch.int64)
        if axis is None:
            tensor = tensor.reshape(-1)
            reduction_axis = 0
            reduction_keepdims = False
        else:
            reduction_axis = axis
            reduction_keepdims = keepdims
        try:
            nan_mask = None
            if tensor.is_floating_point():
                nan_mask = torch.isnan(tensor)
            if operation == "max":
                index = torch.argmax(
                    tensor, dim=reduction_axis,
                    keepdim=reduction_keepdims,
                )
            elif operation == "min":
                index = torch.argmin(
                    tensor, dim=reduction_axis,
                    keepdim=reduction_keepdims,
                )
            else:
                return NotImplemented
            if nan_mask is not None:
                has_nan = torch.any(
                    nan_mask, dim=reduction_axis,
                    keepdim=reduction_keepdims,
                )
                if torch.any(has_nan):
                    nan_index = torch.argmax(
                        nan_mask.to(dtype=torch.int8),
                        dim=reduction_axis,
                        keepdim=reduction_keepdims,
                    )
                    index = torch.where(has_nan, nan_index, index)
            if axis is None and keepdims:
                index = index.reshape(result_shape)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        if index.ndim:
            return self._wrap(index)
        return np.intp(index.item())

    def numpy_searchsorted(self, values, side="left", sorter=None):
        """Find real-valued insertion points without a host array copy."""
        if not isinstance(side, str):
            raise TypeError(
                f"search side must be str, not {type(side).__name__}"
            )
        if side not in ("left", "right"):
            raise ValueError(
                "search side must be 'left' or 'right' "
                f"(got {side!r})"
            )
        if (
            not _device_matches_active(self.tensor)
            or np.dtype(self.dtype).kind not in "fiu"
        ):
            return NotImplemented

        sequence = self.tensor
        if sequence.ndim != 1:
            return NotImplemented
        if sorter is not None:
            if isinstance(sorter, TorchArrayData):
                if sorter.tensor.device != sequence.device:
                    return NotImplemented
                sorter_tensor = sorter.tensor
                sorter_dtype = np.dtype(sorter.dtype)
            elif isinstance(sorter, torch.Tensor):
                if sorter.device != sequence.device:
                    return NotImplemented
                sorter_tensor = sorter
                try:
                    sorter_dtype = _numpy_dtype(sorter.dtype)
                except TypeError:
                    return NotImplemented
            else:
                try:
                    sorter_values = np.asarray(sorter)
                    sorter_dtype = sorter_values.dtype
                    sorter_tensor = torch.as_tensor(
                        sorter_values, device=sequence.device
                    )
                except (OverflowError, TypeError, ValueError, RuntimeError):
                    return NotImplemented
            if sorter_dtype.kind not in "iu":
                return NotImplemented
            if (
                sorter_tensor.ndim != 1
                or sorter_tensor.numel() != sequence.numel()
            ):
                return NotImplemented
            sorter_tensor = sorter_tensor.to(dtype=torch.int64)
            if torch.any(
                (sorter_tensor < 0) | (sorter_tensor >= sequence.numel())
            ).item():
                raise ValueError("Sorter index out of range.")
            sequence = sequence[sorter_tensor]

        if isinstance(values, TorchArrayData):
            if values.tensor.device != sequence.device:
                return NotImplemented
            values = values.tensor
        elif (
            isinstance(values, torch.Tensor)
            and values.device != sequence.device
        ):
            return NotImplemented

        try:
            sequence, value_tensor, outside_range = _comparison_tensors(
                sequence, self.dtype, values
            )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented
        if outside_range is not None:
            return np.intp(0 if outside_range < 0 else sequence.numel())
        if sequence.is_complex():
            return NotImplemented

        scalar_result = value_tensor.ndim == 0
        try:
            if sequence.is_floating_point():
                sequence_nan = torch.isnan(sequence)
                value_nan = torch.isnan(value_tensor)
                finite_count = sequence.numel() - torch.count_nonzero(
                    sequence_nan
                )
                sequence = torch.where(
                    sequence_nan,
                    torch.full_like(sequence, torch.inf),
                    sequence,
                )
                searchable_values = torch.where(
                    value_nan,
                    torch.full_like(value_tensor, torch.inf),
                    value_tensor,
                )
            else:
                value_nan = None
                finite_count = None
                searchable_values = value_tensor

            result = torch.searchsorted(
                sequence.contiguous(),
                searchable_values.contiguous(),
                right=side == "right",
            )
            if value_nan is not None:
                if side == "right":
                    result = torch.where(
                        value_nan,
                        torch.full_like(result, sequence.numel()),
                        torch.minimum(result, finite_count),
                    )
                else:
                    result = torch.where(value_nan, finite_count, result)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        if scalar_result:
            return np.intp(result.item())
        return self._wrap(result)

    def numpy_argsort(
        self, axis=-1, kind=None, order=None, stable=None
    ):
        """Return real-valued sort indices without a host array copy."""
        if (
            not _device_matches_active(self.tensor)
            or np.dtype(self.dtype).kind not in "fiu"
            or order is not None
        ):
            return NotImplemented
        if kind not in (None, "stable", "mergesort"):
            return NotImplemented
        if kind is not None and stable is not None:
            raise ValueError(
                "`kind` and `stable` parameters can't be provided at the "
                "same time. Use only one of them."
            )

        tensor = self.tensor
        if axis is None:
            tensor = tensor.reshape(-1)
            dim = 0
        else:
            try:
                dim = operator.index(axis)
            except TypeError:
                return NotImplemented
            if dim < -tensor.ndim or dim >= tensor.ndim:
                return NotImplemented

        stable_sort = (
            bool(stable)
            if stable is not None
            else kind in ("stable", "mergesort")
        )
        try:
            result = torch.argsort(tensor, dim=dim, stable=stable_sort)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        return self._wrap(result)

    def numpy_nonzero(self):
        """Return C-ordered nonzero indices without a host array copy."""
        if (
            not _device_matches_active(self.tensor)
            or (self.tensor.device.type == "mps" and self.tensor.is_complex())
        ):
            return NotImplemented
        try:
            result = torch.nonzero(self.tensor, as_tuple=True)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        return tuple(self._wrap(index) for index in result)

    def _numpy_index_locations(self, function, args, kwargs):
        """Return grouped or flattened nonzero locations within Torch."""
        options = dict(kwargs)
        if len(args) > 1:
            return NotImplemented
        if args:
            if "a" in options:
                return NotImplemented
            array = args[0]
        else:
            sentinel = object()
            array = options.pop("a", sentinel)
            if array is sentinel:
                return NotImplemented
        if (
                options
                or not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)
                or (
                    array.tensor.device.type == "mps"
                    and array.tensor.is_complex()
                )
        ):
            return NotImplemented

        try:
            if function is np.flatnonzero:
                result = torch.nonzero(
                    array.tensor.reshape(-1), as_tuple=False
                ).reshape(-1)
            else:
                result = torch.argwhere(array.tensor)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        return self._wrap(result)

    def _numpy_where(self, args, kwargs):
        """Evaluate three-operand NumPy where without leaving Torch."""
        if kwargs or len(args) != 3:
            return NotImplemented

        condition, value_true, value_false = args
        operands = (condition, value_true, value_false)
        shapes = []
        for operand in operands:
            if isinstance(operand, TorchArrayData):
                if (
                        not _device_matches_active(operand.tensor)
                        or operand.tensor.device != self.tensor.device
                        or np.dtype(operand.dtype).kind not in "biufc"):
                    return NotImplemented
                shapes.append(operand.shape)
            elif np.isscalar(operand):
                if np.asarray(operand).dtype.kind not in "biufc":
                    return NotImplemented
                shapes.append(())
            else:
                return NotImplemented

        dtype_inputs = []
        for operand in (value_true, value_false):
            if isinstance(operand, TorchArrayData):
                dtype_inputs.append(np.dtype(operand.dtype))
            else:
                dtype_inputs.append(operand)

        try:
            target_np = np.result_type(*dtype_inputs)
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_np)
            )
            execution_torch = (
                torch.int64 if target_np.kind == "u" else target_torch
            )
            torch.broadcast_shapes(*shapes)

            if isinstance(condition, TorchArrayData):
                condition_tensor = condition.tensor.to(dtype=torch.bool)
            else:
                condition_tensor = torch.as_tensor(
                    bool(condition),
                    dtype=torch.bool,
                    device=self.tensor.device,
                )

            values = []
            for operand in (value_true, value_false):
                if isinstance(operand, TorchArrayData):
                    tensor = operand.tensor.to(dtype=execution_torch)
                else:
                    tensor = torch.as_tensor(
                        operand,
                        dtype=execution_torch,
                        device=self.tensor.device,
                    )
                values.append(tensor)
            result = torch.where(condition_tensor, *values)
            if result.dtype != target_torch:
                result = result.to(dtype=target_torch)
        except (OverflowError, TypeError, ValueError):
            return NotImplemented
        except RuntimeError as exc:
            if "broadcast" in str(exc).lower() or "size of tensor" in str(exc):
                raise ValueError(str(exc)) from exc
            return NotImplemented
        return self._wrap(result)

    def _numpy_diff(self, args, kwargs):
        """Evaluate NumPy differences without copying data to the host."""
        if not args or len(args) > 5:
            return NotImplemented

        options = dict(kwargs)
        parameters = {
            "n": 1,
            "axis": -1,
            "prepend": np._NoValue,
            "append": np._NoValue,
        }
        for name, value in zip(parameters, args[1:]):
            if name in options:
                return NotImplemented
            parameters[name] = value
        for name in tuple(parameters):
            if name in options:
                parameters[name] = options.pop(name)
        if options:
            return NotImplemented

        array = args[0]
        if not isinstance(array, TorchArrayData):
            return NotImplemented
        n = parameters["n"]
        try:
            if n == 0:
                return array
            if n < 0:
                raise ValueError(
                    "order must be non-negative but got " + repr(n)
                )
            n = operator.index(n)
        except TypeError:
            return NotImplemented

        if (
                not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented
        if array.ndim == 0:
            raise ValueError(
                "diff requires input that is at least one dimensional"
            )

        try:
            original_axis = operator.index(parameters["axis"])
        except TypeError:
            return NotImplemented
        axis = original_axis
        if axis < 0:
            axis += array.ndim
        if axis < 0 or axis >= array.ndim:
            raise np.exceptions.AxisError(original_axis, ndim=array.ndim)

        operands = []
        operand_dtypes = [np.dtype(array.dtype)]
        for name in ("prepend", "append"):
            value = parameters[name]
            if value is np._NoValue:
                operands.append(value)
                continue
            if isinstance(value, TorchArrayData):
                if (
                        not _device_matches_active(value.tensor)
                        or value.tensor.device != array.tensor.device):
                    return NotImplemented
                operand_dtypes.append(np.dtype(value.dtype))
            elif np.isscalar(value):
                scalar = np.asarray(value)
                if scalar.dtype.kind not in "biufc":
                    return NotImplemented
                operand_dtypes.append(scalar.dtype)
            else:
                return NotImplemented
            operands.append(value)

        try:
            target_np = np.result_type(*operand_dtypes)
            target_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(target_np)
            )
            execution_torch = (
                torch.int64 if target_np.kind == "u" else target_torch
            )
            tensor = array.tensor.to(dtype=execution_torch)
            combined = []
            for value in (operands[0], array, operands[1]):
                if value is np._NoValue:
                    continue
                if isinstance(value, TorchArrayData):
                    current = value.tensor.to(dtype=execution_torch)
                else:
                    shape = list(array.shape)
                    shape[axis] = 1
                    current = torch.as_tensor(
                        value,
                        dtype=execution_torch,
                        device=array.tensor.device,
                    ).reshape((1,) * array.ndim).expand(shape)
                combined.append(current)
            if len(combined) > 1:
                tensor = torch.cat(combined, dim=axis)

            slices_after = [slice(None)] * array.ndim
            slices_before = [slice(None)] * array.ndim
            slices_after[axis] = slice(1, None)
            slices_before[axis] = slice(None, -1)
            slices_after = tuple(slices_after)
            slices_before = tuple(slices_before)
            for _ in range(n):
                if target_np.kind == "b":
                    tensor = torch.ne(
                        tensor[slices_after], tensor[slices_before]
                    )
                else:
                    tensor = (
                        tensor[slices_after] - tensor[slices_before]
                    )
            if tensor.dtype != target_torch:
                tensor = tensor.to(dtype=target_torch)
        except (OverflowError, TypeError, ValueError):
            return NotImplemented
        except RuntimeError as exc:
            if "size" in str(exc).lower():
                raise ValueError(str(exc)) from exc
            return NotImplemented
        return self._wrap(tensor)

    def _numpy_diagonal(self, args, kwargs):
        """Select diagonals as Torch views."""
        if len(args) > 4:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"offset": 0, "axis1": 0, "axis2": 1}
        if args:
            if "a" in options:
                return NotImplemented
            array = args[0]
        else:
            try:
                array = options.pop("a")
            except KeyError:
                return NotImplemented
        for name, value in zip(parameters, args[1:]):
            if name in options:
                return NotImplemented
            parameters[name] = value
        for name in tuple(parameters):
            if name in options:
                parameters[name] = options.pop(name)
        if (
                options
                or not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented
        return array.numpy_diagonal(**parameters)

    def _numpy_diag(self, function, args, kwargs):
        """Construct or extract diagonals without copying array data."""
        if len(args) > 2:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"k": 0}
        for index, name in enumerate(("v", "k")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "v":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["v"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        tensor = array.tensor
        if function is np.diag and tensor.ndim not in (1, 2):
            raise ValueError("Input must be 1- or 2-d.")
        try:
            offset = operator.index(parameters["k"])
        except (TypeError, ValueError):
            return NotImplemented

        if function is np.diag and tensor.ndim == 2:
            return array.numpy_diagonal(offset=offset, axis1=0, axis2=1)
        if function is np.diagflat:
            tensor = tensor.reshape(-1)
        try:
            return self._wrap(torch.diag(tensor, diagonal=offset))
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def _numpy_triangle(self, function, args, kwargs):
        """Apply NumPy-compatible triangular masks on a Torch device."""
        if len(args) > 2:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"k": 0}
        for index, name in enumerate(("m", "k")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "m":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["m"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        tensor = array.tensor
        if tensor.ndim == 0:
            raise TypeError("tri() missing 1 required positional argument: 'N'")
        rows = tensor.shape[-2] if tensor.ndim > 1 else tensor.shape[-1]
        columns = tensor.shape[-1]

        # NumPy's tri builds integer arange metadata from ``-k``. Keeping
        # that one-dimensional operation preserves its unusual fractional-k
        # behavior without moving any input tensor data.
        try:
            tri_k = (
                parameters["k"]
                if function is np.tril
                else parameters["k"] - 1
            )
            column_thresholds = np.arange(
                -tri_k,
                columns - tri_k,
                dtype=np.int64,
            )
        except (OverflowError, TypeError, ValueError):
            return NotImplemented
        if column_thresholds.shape != (columns,):
            return NotImplemented

        row_indices = torch.arange(
            rows, dtype=torch.int64, device=tensor.device
        ).reshape(rows, 1)
        thresholds = torch.as_tensor(
            column_thresholds,
            dtype=torch.int64,
            device=tensor.device,
        ).reshape(1, columns)
        mask = row_indices >= thresholds
        if function is np.triu:
            mask = ~mask

        # Multiplication by a same-dtype mask supports PyCBC's uint32 dtype,
        # for which torch.where and masked_fill are not implemented.
        result = tensor * mask.to(dtype=tensor.dtype)
        return self._wrap(result)

    def _numpy_trace(self, args, kwargs):
        """Sum diagonals without copying array data to the host."""
        if len(args) > 6:
            return NotImplemented

        options = dict(kwargs)
        defaults = {
            "offset": 0,
            "axis1": 0,
            "axis2": 1,
            "dtype": None,
            "out": None,
        }
        parameters = {}
        for index, name in enumerate(
                ("a", "offset", "axis1", "axis2", "dtype", "out")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        out = parameters["out"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented
        if isinstance(out, np.ndarray):
            # A NumPy output explicitly requests host storage.
            return NotImplemented
        if out is not None and (
                not isinstance(out, TorchArrayData)
                or not _device_matches_active(out.tensor)
                or out.tensor.device != array.tensor.device):
            return NotImplemented

        diagonal = array.numpy_diagonal(
            offset=parameters["offset"],
            axis1=parameters["axis1"],
            axis2=parameters["axis2"],
        )
        if diagonal is NotImplemented:
            return NotImplemented

        input_dtype = np.dtype(array.dtype)
        dtype = parameters["dtype"]
        if dtype is None:
            if input_dtype.kind in "bi":
                target_dtype = np.dtype(np.int64)
            elif input_dtype.kind == "u":
                # NumPy promotes uint32 to uint64, which PyCBC Array cannot
                # represent. Preserve the established host fallback.
                return NotImplemented
            else:
                target_dtype = input_dtype
        else:
            try:
                target_dtype = np.dtype(dtype)
            except TypeError:
                return NotImplemented
        if target_dtype.kind not in "biufc":
            return NotImplemented

        try:
            target_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(target_dtype)
            )
            tensor = diagonal.tensor
            if input_dtype.kind == "c" and target_dtype.kind in "iuf":
                warnings.warn(
                    "Casting complex values to real discards the imaginary "
                    "part",
                    np.exceptions.ComplexWarning,
                    stacklevel=4,
                )
                tensor = tensor.real
            if target_dtype.kind in "biu":
                # Torch promotes bool and narrow integer sums to int64, and
                # cannot reduce uint32 directly. Cast each element first,
                # accumulate in int64, then restore NumPy's requested dtype;
                # this preserves boolean and modular integer sum semantics.
                result = torch.sum(
                    tensor.to(dtype=target_torch).to(dtype=torch.int64),
                    dim=-1,
                ).to(dtype=target_torch)
            else:
                result = torch.sum(
                    tensor.to(dtype=target_torch), dim=-1
                )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if out is not None:
            if out.tensor.shape != result.shape:
                raise ValueError(
                    "output parameter for reduction operation add has the "
                    "wrong number of dimensions or shape"
                )
            if result.is_complex() and not out.tensor.is_complex():
                if out.tensor.dtype != torch.bool:
                    warnings.warn(
                        "Casting complex values to real discards the "
                        "imaginary part",
                        np.exceptions.ComplexWarning,
                        stacklevel=4,
                    )
                    result = result.real
            out.tensor.copy_(result)
            return out
        if result.ndim == 0:
            return target_dtype.type(result.item())
        return self._wrap(result)

    def _numpy_roll(self, args, kwargs):
        """Roll array elements without copying data to the host."""
        if len(args) < 2 or len(args) > 3:
            return NotImplemented

        options = dict(kwargs)
        array = args[0]
        shift = args[1]
        if len(args) == 3:
            if "axis" in options:
                return NotImplemented
            axis = args[2]
        else:
            axis = options.pop("axis", None)
        if (
                options
                or not isinstance(array, TorchArrayData)
                or isinstance(shift, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        tensor = array.tensor
        original_shape = tensor.shape
        if axis is None:
            tensor = tensor.reshape(-1)
            axes = (0,)
        else:
            try:
                axes = (operator.index(axis),)
            except TypeError:
                try:
                    axes = tuple(operator.index(value) for value in axis)
                except TypeError:
                    return NotImplemented

            normalized_axes = []
            for original_axis in axes:
                normalized_axis = original_axis
                if normalized_axis < 0:
                    normalized_axis += tensor.ndim
                if normalized_axis < 0 or normalized_axis >= tensor.ndim:
                    raise np.exceptions.AxisError(
                        original_axis, ndim=tensor.ndim
                    )
                normalized_axes.append(normalized_axis)
            axes = tuple(normalized_axes)

        try:
            broadcasted = np.broadcast(shift, axes)
        except TypeError:
            return NotImplemented
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences"
            )

        shifts = dict.fromkeys(range(tensor.ndim), 0)
        try:
            for amount, dimension in broadcasted:
                shifts[dimension] += int(amount)
        except (TypeError, ValueError):
            return NotImplemented

        dimensions = []
        amounts = []
        for dimension, amount in shifts.items():
            amount %= tensor.shape[dimension] or 1
            if amount:
                dimensions.append(dimension)
                amounts.append(amount)
        if dimensions:
            result = torch.roll(
                tensor,
                shifts=tuple(amounts),
                dims=tuple(dimensions),
            )
        else:
            result = tensor.clone()
        if axis is None:
            result = result.reshape(original_shape)
        return self._wrap(result)

    def _numpy_flip(self, function, args, kwargs):
        """Reverse array dimensions without negative-stride host slicing."""
        options = dict(kwargs)
        if len(args) > (2 if function is np.flip else 1):
            return NotImplemented

        if args:
            if "m" in options:
                return NotImplemented
            array = args[0]
        else:
            try:
                array = options.pop("m")
            except KeyError:
                return NotImplemented

        if function is np.flip:
            if len(args) == 2:
                if "axis" in options:
                    return NotImplemented
                axis = args[1]
            else:
                axis = options.pop("axis", None)
        else:
            axis = 0 if function is np.flipud else 1

        if (
                options
                or not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        tensor = array.tensor
        if function is np.flipud and tensor.ndim < 1:
            raise ValueError("Input must be >= 1-d.")
        if function is np.fliplr and tensor.ndim < 2:
            raise ValueError("Input must be >= 2-d.")

        if axis is None:
            axes = tuple(range(tensor.ndim))
        else:
            try:
                axes = (operator.index(axis),)
            except TypeError:
                try:
                    axes = tuple(operator.index(value) for value in axis)
                except TypeError:
                    return NotImplemented

            normalized_axes = []
            for original_axis in axes:
                normalized_axis = original_axis
                if normalized_axis < 0:
                    normalized_axis += tensor.ndim
                if normalized_axis < 0 or normalized_axis >= tensor.ndim:
                    raise np.exceptions.AxisError(
                        original_axis, ndim=tensor.ndim
                    )
                if normalized_axis in normalized_axes:
                    raise ValueError("repeated axis")
                normalized_axes.append(normalized_axis)
            axes = tuple(normalized_axes)

        if not axes:
            return self._wrap(tensor)
        if tensor.dtype == _TORCH_UINT32:
            result = torch.flip(tensor.to(torch.int64), dims=axes)
            return self._wrap(result.to(_TORCH_UINT32))
        return self._wrap(torch.flip(tensor, dims=axes))

    def _numpy_rot90(self, args, kwargs):
        """Rotate a Torch tensor with NumPy's axis and ``k`` semantics."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {"k": 1, "axes": (0, 1)}
        for index, name in enumerate(("m", "k", "axes")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "m":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["m"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        # Keep NumPy's validation order. In particular, axes separated by
        # ndim are reported as the same axis before either is range checked.
        axes = tuple(parameters["axes"])
        if len(axes) != 2:
            raise ValueError("len(axes) must be 2.")
        if axes[0] == axes[1] or abs(axes[0] - axes[1]) == array.ndim:
            raise ValueError("Axes must be different.")
        if (
                axes[0] >= array.ndim or axes[0] < -array.ndim
                or axes[1] >= array.ndim or axes[1] < -array.ndim):
            raise ValueError(
                f"Axes={axes} out of range for array of ndim={array.ndim}."
            )

        k = parameters["k"]
        k %= 4
        tensor = array.tensor
        if k == 0:
            return self._wrap(tensor[:])

        def flip(value, axis):
            if value.dtype == _TORCH_UINT32:
                return torch.flip(
                    value.to(torch.int64), dims=(operator.index(axis),)
                ).to(_TORCH_UINT32)
            return torch.flip(value, dims=(operator.index(axis),))

        if k == 2:
            return self._wrap(flip(flip(tensor, axes[0]), axes[1]))

        # Construct the permutation through NumPy metadata to preserve its
        # indexing errors for non-integral axes without touching tensor data.
        dimensions = np.arange(tensor.ndim)
        dimensions[axes[0]], dimensions[axes[1]] = (
            dimensions[axes[1]], dimensions[axes[0]]
        )
        dimensions = tuple(int(dimension) for dimension in dimensions)
        if k == 1:
            result = flip(tensor, axes[1]).permute(dimensions)
        else:
            result = flip(tensor.permute(dimensions), axes[1])
        return self._wrap(result)

    def _numpy_expand_dims(self, args, kwargs):
        """Insert singleton dimensions as a Torch view."""
        options = dict(kwargs)
        if len(args) > 2:
            return NotImplemented

        if args:
            if "a" in options:
                return NotImplemented
            array = args[0]
        else:
            try:
                array = options.pop("a")
            except KeyError:
                return NotImplemented

        if len(args) == 2:
            if "axis" in options:
                return NotImplemented
            axis = args[1]
        else:
            try:
                axis = options.pop("axis")
            except KeyError:
                return NotImplemented

        if (
                options
                or not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        if isinstance(axis, (tuple, list)):
            try:
                axes = tuple(operator.index(value) for value in axis)
            except TypeError:
                return NotImplemented
        else:
            try:
                axes = (operator.index(axis),)
            except TypeError:
                return NotImplemented

        output_ndim = array.ndim + len(axes)
        normalized_axes = []
        for original_axis in axes:
            normalized_axis = original_axis
            if normalized_axis < 0:
                normalized_axis += output_ndim
            if normalized_axis < 0 or normalized_axis >= output_ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=output_ndim
                )
            if normalized_axis in normalized_axes:
                raise ValueError("repeated axis")
            normalized_axes.append(normalized_axis)

        result = array.tensor
        for normalized_axis in sorted(normalized_axes):
            result = torch.unsqueeze(result, normalized_axis)
        return self._wrap(result)

    @staticmethod
    def _normalize_movement_axes(axis, ndim, name):
        """Normalize unique axes for NumPy's movement operations."""
        try:
            axes = (operator.index(axis),)
        except TypeError:
            try:
                axes = tuple(operator.index(value) for value in axis)
            except TypeError:
                return None

        normalized_axes = []
        for original_axis in axes:
            normalized_axis = (
                original_axis + ndim if original_axis < 0 else original_axis
            )
            if normalized_axis < 0 or normalized_axis >= ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=ndim, msg_prefix=name
                )
            if normalized_axis in normalized_axes:
                raise ValueError(f"repeated axis in `{name}` argument")
            normalized_axes.append(normalized_axis)
        return tuple(normalized_axes)

    def _numpy_moveaxis(self, args, kwargs):
        """Move array axes using a Torch permutation view."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("a", "source", "destination")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            else:
                try:
                    parameters[name] = options.pop(name)
                except KeyError:
                    return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        source = self._normalize_movement_axes(
            parameters["source"], array.ndim, "source"
        )
        destination = self._normalize_movement_axes(
            parameters["destination"], array.ndim, "destination"
        )
        if source is None or destination is None:
            return NotImplemented
        if len(source) != len(destination):
            raise ValueError(
                "`source` and `destination` arguments must have the same "
                "number of elements"
            )

        dimensions = [
            dimension for dimension in range(array.ndim)
            if dimension not in source
        ]
        for destination_axis, source_axis in sorted(
                zip(destination, source)):
            dimensions.insert(destination_axis, source_axis)
        return self._wrap(array.tensor.permute(tuple(dimensions)))

    def _numpy_rollaxis(self, args, kwargs):
        """Roll one axis using NumPy's legacy positioning rules."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {"start": 0}
        for index, name in enumerate(("a", "axis", "start")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name != "start":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented

        try:
            original_axis = operator.index(parameters["axis"])
            start = operator.index(parameters["start"])
        except TypeError:
            return NotImplemented
        normalized_axis = (
            original_axis + array.ndim
            if original_axis < 0 else original_axis
        )
        if normalized_axis < 0 or normalized_axis >= array.ndim:
            raise np.exceptions.AxisError(original_axis, ndim=array.ndim)

        if start < 0:
            start += array.ndim
        if start < 0 or start >= array.ndim + 1:
            message = (
                "'start' arg requires "
                f"{-array.ndim} <= start < {array.ndim + 1}, "
                f"but {start} was passed in"
            )
            raise np.exceptions.AxisError(message)
        if normalized_axis < start:
            start -= 1

        dimensions = list(range(array.ndim))
        dimensions.remove(normalized_axis)
        dimensions.insert(start, normalized_axis)
        return self._wrap(array.tensor.permute(tuple(dimensions)))

    def _numpy_atleast_nd(self, function, args, kwargs):
        """Add leading or trailing dimensions using Torch views."""
        if kwargs or not args or not all(
                isinstance(array, TorchArrayData) for array in args):
            return NotImplemented
        if not all(
                _device_matches_active(array.tensor)
                and array.tensor.device == self.tensor.device
                for array in args):
            return NotImplemented

        minimum_ndim = {
            np.atleast_1d: 1,
            np.atleast_2d: 2,
            np.atleast_3d: 3,
        }[function]
        results = []
        for array in args:
            tensor = array.tensor
            if tensor.ndim >= minimum_ndim:
                result = tensor
            elif minimum_ndim == 1:
                result = tensor.reshape(1)
            elif minimum_ndim == 2:
                result = tensor.reshape(1, 1) if tensor.ndim == 0 \
                    else tensor.unsqueeze(0)
            elif tensor.ndim == 0:
                result = tensor.reshape(1, 1, 1)
            elif tensor.ndim == 1:
                result = tensor.reshape(1, -1, 1)
            else:
                result = tensor.unsqueeze(-1)
            results.append(
                array if result is tensor else array._wrap(result)
            )

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _numpy_broadcast(self, function, args, kwargs):
        """Broadcast Torch arrays using stride-zero views."""
        options = dict(kwargs)
        subok = options.pop("subok", False)
        try:
            if bool(subok):
                return NotImplemented
        except (TypeError, ValueError):
            return NotImplemented

        if function is np.broadcast_arrays:
            if options or not args or not all(
                    isinstance(array, TorchArrayData) for array in args):
                return NotImplemented
            if not all(
                    _device_matches_active(array.tensor)
                    and array.tensor.device == self.tensor.device
                    for array in args):
                return NotImplemented
            try:
                tensors = torch.broadcast_tensors(
                    *(array.tensor for array in args)
                )
            except RuntimeError:
                return NotImplemented
            return tuple(
                array if tensor is array.tensor else array._wrap(tensor)
                for array, tensor in zip(args, tensors)
            )

        if len(args) > 3:
            return NotImplemented
        if args:
            array = args[0]
        else:
            array = options.pop("array", None)
        if len(args) > 1:
            shape = args[1]
        else:
            shape = options.pop("shape", None)
        if len(args) > 2:
            try:
                if bool(args[2]):
                    return NotImplemented
            except (TypeError, ValueError):
                return NotImplemented
        if (
                options
                or not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        try:
            normalized_shape = (operator.index(shape),)
        except TypeError:
            try:
                normalized_shape = tuple(
                    operator.index(dimension) for dimension in shape
                )
            except (TypeError, ValueError):
                return NotImplemented
        if any(dimension < 0 for dimension in normalized_shape):
            return NotImplemented
        try:
            return array._wrap(
                torch.broadcast_to(array.tensor, normalized_shape)
            )
        except RuntimeError:
            return NotImplemented

    def _numpy_close(self, function, args, kwargs):
        """Evaluate NumPy tolerance comparisons on the Torch device."""
        options = dict(kwargs)
        if len(args) > 5:
            return NotImplemented

        defaults = {
            "rtol": 1e-5,
            "atol": 1e-8,
            "equal_nan": False,
        }
        parameters = {}
        for index, name in enumerate((
                "a", "b", "rtol", "atol", "equal_nan")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        operands = (parameters["a"], parameters["b"])
        arrays = [
            operand for operand in operands
            if isinstance(operand, TorchArrayData)
        ]
        if not arrays:
            return NotImplemented
        device = arrays[0].tensor.device
        if any(
                not _device_matches_active(array.tensor)
                or array.tensor.device != device
                or np.dtype(array.dtype).kind not in "biufc"
                for array in arrays):
            return NotImplemented

        dtype_inputs = []
        for operand in operands:
            if isinstance(operand, TorchArrayData):
                dtype_inputs.append(np.dtype(operand.dtype))
            elif np.isscalar(operand):
                if np.asarray(operand).dtype.kind not in "biufc":
                    return NotImplemented
                dtype_inputs.append(operand)
            else:
                return NotImplemented

        tolerances = []
        for name in ("rtol", "atol"):
            value = parameters[name]
            if (
                    not np.isscalar(value)
                    or np.asarray(value).dtype.kind not in "biuf"):
                return NotImplemented
            try:
                value = float(value)
            except (OverflowError, TypeError, ValueError):
                return NotImplemented
            if value < 0:
                return NotImplemented
            tolerances.append(value)
        if not np.isscalar(parameters["equal_nan"]):
            return NotImplemented
        try:
            equal_nan = bool(parameters["equal_nan"])
            target_np = np.result_type(*dtype_inputs)
            target_torch = _ensure_supported(
                device, _torch_dtype(target_np)
            )
            tensors = []
            for operand in operands:
                if isinstance(operand, TorchArrayData):
                    tensor = operand.tensor.to(dtype=target_torch)
                else:
                    tensor = torch.as_tensor(
                        np.asarray(operand),
                        dtype=target_torch,
                        device=device,
                    )
                tensors.append(tensor)
            result = torch.isclose(
                tensors[0],
                tensors[1],
                rtol=tolerances[0],
                atol=tolerances[1],
                equal_nan=equal_nan,
            )
        except (OverflowError, TypeError, ValueError):
            return NotImplemented
        except RuntimeError as exc:
            if "size" in str(exc).lower():
                raise ValueError(str(exc)) from exc
            return NotImplemented

        if function is np.allclose:
            return bool(torch.all(result).item())
        return self._wrap(result)

    def _numpy_array_equality(self, function, args, kwargs):
        """Reduce exact NumPy array equality on the Torch device."""
        options = dict(kwargs)
        names = ("a1", "a2", "equal_nan")
        maximum_args = 3 if function is np.array_equal else 2
        if len(args) > maximum_args:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(names[:maximum_args]):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "equal_nan":
                parameters[name] = False
            else:
                return NotImplemented
        if options:
            return NotImplemented

        left = parameters["a1"]
        right = parameters["a2"]
        if (
                not isinstance(left, TorchArrayData)
                or not isinstance(right, TorchArrayData)
                or left.tensor.device != right.tensor.device
                or not _device_matches_active(left.tensor)
                or not _device_matches_active(right.tensor)
                or np.dtype(left.dtype).kind not in "biufc"
                or np.dtype(right.dtype).kind not in "biufc"):
            return NotImplemented

        equal_nan = False
        if function is np.array_equal:
            value = parameters["equal_nan"]
            if not np.isscalar(value):
                return NotImplemented
            try:
                equal_nan = bool(value)
            except (TypeError, ValueError):
                return NotImplemented
            if left.shape != right.shape:
                return False
        else:
            try:
                torch.broadcast_shapes(left.shape, right.shape)
            except RuntimeError:
                return False

        try:
            left_tensor, right_tensor, outside_range = _comparison_tensors(
                left.tensor, left.dtype, right.tensor
            )
            if outside_range is not None:
                return False
            result = torch.eq(left_tensor, right_tensor)
            if equal_nan and (left_tensor.is_floating_point()
                              or left_tensor.is_complex()):
                result = result | (
                    torch.isnan(left_tensor) & torch.isnan(right_tensor)
                )
            return bool(torch.all(result).item())
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def _numpy_count_nonzero(self, args, kwargs):
        """Count nonzero values without copying array data to NumPy."""
        options = dict(kwargs)
        if len(args) > 2:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("a", "axis")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "axis":
                parameters[name] = None
            else:
                return NotImplemented
        keepdims = options.pop("keepdims", False)
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biufc"):
            return NotImplemented

        tensor = array.tensor
        axis = parameters["axis"]
        try:
            keepdims = bool(operator.index(keepdims))
            axes = _normalized_reduction_axes(axis, tensor.ndim)
        except (IndexError, TypeError, ValueError):
            return NotImplemented

        try:
            nonzero = (tensor != 0).to(dtype=torch.int64)
            if axis is None:
                result = torch.sum(nonzero)
            elif not axes:
                result = nonzero
            else:
                result = torch.sum(nonzero, dim=axes)
            if keepdims:
                result = result.reshape(
                    _reduction_output_shape(tensor.shape, axes, True)
                )
        except (TypeError, RuntimeError):
            return NotImplemented

        if result.ndim == 0:
            return np.intp(result.item())
        return self._wrap(result)

    def _numpy_average(self, args, kwargs):
        """Compute NumPy-compatible weighted averages on the Torch device."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {
            "axis": None,
            "weights": None,
            "returned": False,
            "keepdims": False,
        }
        parameters = {}
        for index, name in enumerate(
                ("a", "axis", "weights", "returned")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if "keepdims" in options:
            parameters["keepdims"] = options.pop("keepdims")
        else:
            parameters["keepdims"] = defaults["keepdims"]
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biufc"):
            return NotImplemented

        axis = parameters["axis"]
        if (
                array.tensor.ndim == 0
                and axis is not None
                and not (isinstance(axis, tuple) and not axis)):
            # NumPy accepts ``axis=None`` and the empty tuple for scalars,
            # but rejects even axis 0/-1 rather than treating it as a no-op.
            return NotImplemented

        try:
            axes = _normalized_reduction_axes(
                axis, array.tensor.ndim
            )
            keepdims = bool(operator.index(parameters["keepdims"]))
            returned = bool(parameters["returned"])
        except (IndexError, TypeError, ValueError):
            return NotImplemented

        tensor = array.tensor
        input_dtype = np.dtype(array.dtype)
        weights = parameters["weights"]
        if weights is None:
            output_dtype = (
                np.dtype(np.float64)
                if input_dtype.kind in "biu" else input_dtype
            )
            try:
                output_torch = _ensure_supported(
                    tensor.device, _torch_dtype(output_dtype)
                )
            except (TypeError, RuntimeError):
                return NotImplemented

            execution = tensor.to(dtype=output_torch)
            if axes:
                count = 1
                for axis_index in axes:
                    count *= tensor.shape[axis_index]
                if count == 0:
                    warnings.warn(
                        "Mean of empty slice.",
                        RuntimeWarning,
                        stacklevel=4,
                    )
                result = torch.mean(
                    execution, dim=axes, keepdim=keepdims
                )
            else:
                result = execution.clone()

            if returned:
                # NumPy derives this from total/result sizes. In particular,
                # ``returned=True`` raises for an empty, empty-shaped result.
                if result.numel() == 0:
                    raise ZeroDivisionError("division by zero")
                count = tensor.numel() / result.numel()
                scale = torch.full_like(result, count)
        else:
            if isinstance(weights, TorchArrayData):
                if (
                        not _device_matches_active(weights.tensor)
                        or weights.tensor.device != tensor.device):
                    return NotImplemented
                weight_tensor = weights.tensor
                weight_dtype = np.dtype(weights.dtype)
            elif isinstance(weights, torch.Tensor):
                if weights.device != tensor.device:
                    return NotImplemented
                weight_tensor = weights
                try:
                    weight_dtype = np.dtype(_numpy_dtype(weights.dtype))
                except TypeError:
                    # CPU Torch tensors support NumPy-compatible dtypes that
                    # PyCBC's Torch storage does not. Preserve NumPy's host
                    # fallback for those raw weight tensors.
                    return NotImplemented
            else:
                try:
                    weight_values = np.asanyarray(weights)
                    if type(weight_values) is not np.ndarray:
                        # ndarray subclasses such as MaskedArray and matrix
                        # carry semantics that torch.as_tensor would discard.
                        # Leave them to NumPy's host implementation.
                        return NotImplemented
                    weight_dtype = np.dtype(weight_values.dtype)
                    weight_torch = _ensure_supported(
                        tensor.device, _torch_dtype(weight_dtype)
                    )
                    weight_tensor = torch.as_tensor(
                        weight_values,
                        dtype=weight_torch,
                        device=tensor.device,
                    )
                except (TypeError, ValueError, RuntimeError):
                    return NotImplemented
            if weight_dtype.kind not in "biufc":
                return NotImplemented

            if tuple(weight_tensor.shape) != tuple(tensor.shape):
                if axis is None:
                    raise TypeError(
                        "Axis must be specified when shapes of a and weights "
                        "differ."
                    )
                expected_shape = tuple(tensor.shape[axis] for axis in axes)
                if tuple(weight_tensor.shape) != expected_shape:
                    raise ValueError(
                        "Shape of weights must be consistent with shape of "
                        "a along specified axis."
                    )
                permutation = tuple(
                    index for index, _axis in sorted(
                        enumerate(axes), key=lambda item: item[1]
                    )
                )
                if permutation:
                    weight_tensor = weight_tensor.permute(permutation)
                weight_tensor = weight_tensor.reshape(tuple(
                    tensor.shape[axis] if axis in axes else 1
                    for axis in range(tensor.ndim)
                ))

            if input_dtype.kind in "biu":
                output_dtype = np.result_type(
                    input_dtype, weight_dtype, np.dtype(np.float64)
                )
            else:
                output_dtype = np.result_type(input_dtype, weight_dtype)
            try:
                output_torch = _ensure_supported(
                    tensor.device, _torch_dtype(output_dtype)
                )
            except (TypeError, RuntimeError):
                return NotImplemented

            execution = tensor.to(dtype=output_torch)
            weight_tensor = weight_tensor.to(dtype=output_torch)
            if axes:
                scale = torch.sum(
                    weight_tensor, dim=axes, keepdim=keepdims
                )
                numerator = torch.sum(
                    execution * weight_tensor,
                    dim=axes,
                    keepdim=keepdims,
                )
            else:
                scale = weight_tensor
                numerator = execution * weight_tensor
            if bool(torch.any(scale == 0).item()):
                raise ZeroDivisionError(
                    "Weights sum to zero, can't be normalized"
                )
            result = numerator / scale
            if returned:
                if scale.shape != result.shape:
                    scale = torch.broadcast_to(scale, result.shape).clone()
                elif not axes:
                    # NumPy returns independent output storage even when an
                    # empty axis tuple makes the weighted reduction a no-op.
                    scale = scale.clone()

        def wrapped(value):
            if value.ndim == 0:
                return output_dtype.type(value.item())
            return self._wrap(value)

        result = wrapped(result)
        if returned:
            return result, wrapped(scale)
        return result

    def _numpy_median(self, args, kwargs):
        """Reduce real arrays without copying their data to NumPy."""
        options = dict(kwargs)
        if len(args) > 5:
            return NotImplemented

        defaults = {
            "axis": None,
            "out": None,
            "overwrite_input": False,
            "keepdims": False,
        }
        parameters = {}
        for index, name in enumerate((
                "a", "axis", "out", "overwrite_input", "keepdims")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        out = parameters["out"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biuf"):
            return NotImplemented
        if out is not None and (
                not isinstance(out, TorchArrayData)
                or not _device_matches_active(out.tensor)
                or out.tensor.device != array.tensor.device):
            return NotImplemented

        try:
            keepdims = bool(parameters["keepdims"])
        except (TypeError, ValueError):
            return NotImplemented

        tensor = array.tensor
        axis = parameters["axis"]
        flatten_all = axis is None
        if flatten_all:
            axes = tuple(range(tensor.ndim))
        else:
            try:
                axes = (operator.index(axis),)
            except TypeError:
                try:
                    axes = tuple(operator.index(value) for value in axis)
                except TypeError:
                    return NotImplemented

            normalized_axes = []
            for original_axis in axes:
                normalized_axis = original_axis
                if normalized_axis < 0:
                    normalized_axis += tensor.ndim
                if normalized_axis < 0 or normalized_axis >= tensor.ndim:
                    raise np.exceptions.AxisError(
                        original_axis, ndim=tensor.ndim
                    )
                if normalized_axis in normalized_axes:
                    raise ValueError("repeated axis")
                normalized_axes.append(normalized_axis)
            axes = tuple(normalized_axes)

        input_dtype = np.dtype(array.dtype)
        output_dtype = (
            input_dtype if input_dtype.kind == "f" else np.dtype(np.float64)
        )
        try:
            output_torch = _ensure_supported(
                tensor.device, _torch_dtype(output_dtype)
            )
        except TypeError:
            return NotImplemented

        # NumPy's empty no-axis reshape has an established error contract.
        if not flatten_all and not axes and tensor.numel() == 0:
            return NotImplemented

        if not axes and not flatten_all:
            result = tensor.to(dtype=output_torch, copy=True)
        else:
            kept_axes = tuple(
                dimension for dimension in range(tensor.ndim)
                if dimension not in axes
            )
            kept_shape = tuple(tensor.shape[dimension] for dimension in kept_axes)
            if flatten_all:
                reduced = tensor.reshape(-1)
                reduction_size = tensor.numel()
            else:
                reduction_size = 1
                for dimension in axes:
                    reduction_size *= tensor.shape[dimension]
                permutation = kept_axes + axes
                reduced = tensor.permute(permutation).reshape(
                    *kept_shape, reduction_size
                )

            reduced = reduced.to(dtype=output_torch)
            if reduction_size == 0:
                warnings.warn(
                    "Mean of empty slice.", RuntimeWarning, stacklevel=4
                )
                result = torch.full(
                    kept_shape,
                    torch.nan,
                    dtype=output_torch,
                    device=tensor.device,
                )
            else:
                sorted_values = torch.sort(reduced, dim=-1).values
                upper_index = reduction_size // 2
                result = sorted_values.select(-1, upper_index)
                if reduction_size % 2 == 0:
                    lower = sorted_values.select(-1, upper_index - 1)
                    result = (lower + result) / 2
                if input_dtype.kind == "f":
                    nan_slices = torch.isnan(reduced).any(dim=-1)
                    result = torch.where(
                        nan_slices,
                        torch.full_like(result, torch.nan),
                        result,
                    )

            if keepdims:
                result_shape = tuple(
                    1 if dimension in axes else tensor.shape[dimension]
                    for dimension in range(tensor.ndim)
                )
                result = result.reshape(result_shape)

        if out is not None:
            if out.tensor.shape != result.shape:
                raise ValueError(
                    "output parameter has the wrong shape: "
                    f"expected {tuple(result.shape)}, got {tuple(out.shape)}"
                )
            out.tensor.copy_(result.to(dtype=out.tensor.dtype))
            return out
        if result.ndim == 0 and not keepdims:
            return output_dtype.type(result.item())
        return self._wrap(result)

    def _numpy_ptp(self, args, kwargs):
        """Return real peak-to-peak ranges without a host-array copy."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {"axis": None, "out": None, "keepdims": False}
        parameters = {}
        for index, name in enumerate(("a", "axis", "out", "keepdims")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        out = parameters["out"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "iuf"):
            return NotImplemented
        if isinstance(out, np.ndarray):
            # A NumPy ``out`` explicitly requests host storage.  Let NumPy
            # perform both the transfer and its output-dtype type resolution.
            return np.ptp(
                array.tensor.detach().cpu().numpy(),
                axis=parameters["axis"],
                out=out,
                keepdims=parameters["keepdims"],
            )
        if out is not None:
            if not isinstance(out, TorchArrayData):
                return NotImplemented
            if (
                    not _device_matches_active(out.tensor)
                    or out.tensor.device != array.tensor.device):
                return NotImplemented

        tensor = array.tensor
        try:
            keepdims = bool(operator.index(parameters["keepdims"]))
            axes = _normalized_reduction_axes(
                parameters["axis"], tensor.ndim
            )
        except (IndexError, TypeError, ValueError):
            return NotImplemented

        if axes and any(tensor.shape[axis] == 0 for axis in axes):
            return NotImplemented

        try:
            reduction_input = tensor
            if tensor.dtype == _TORCH_UINT32:
                reduction_input = tensor.to(dtype=torch.int64)
            if not axes:
                maximum = reduction_input
                minimum = reduction_input
            else:
                maximum = torch.amax(
                    reduction_input, dim=axes, keepdim=keepdims
                )
                minimum = torch.amin(
                    reduction_input, dim=axes, keepdim=keepdims
                )

            if out is None:
                result = maximum - minimum
                if tensor.dtype == _TORCH_UINT32:
                    result = result.to(dtype=_TORCH_UINT32)
            else:
                input_dtype = np.dtype(array.dtype)
                output_dtype = np.dtype(out.dtype)
                subtraction_dtype = np.result_type(
                    output_dtype, input_dtype
                )
                if not np.can_cast(
                        subtraction_dtype,
                        output_dtype,
                        casting="same_kind"):
                    raise TypeError(
                        "Cannot cast ufunc 'subtract' output from "
                        f"dtype('{subtraction_dtype}') to "
                        f"dtype('{output_dtype}') with casting rule "
                        "'same_kind'"
                    )
                subtraction_torch_dtype = _ensure_supported(
                    tensor.device, _torch_dtype(subtraction_dtype)
                )
                if subtraction_torch_dtype == _TORCH_UINT32:
                    # Torch cannot subtract UInt32 tensors.  The operands are
                    # ordered extrema here, so Int64 is a lossless workspace.
                    subtraction_torch_dtype = torch.int64
                # NumPy first stores the maximum in ``out`` and then
                # subtracts the input-typed minimum into that same buffer.
                # The first cast matters for narrow and integer outputs.
                maximum = maximum.to(dtype=out.tensor.dtype)
                result = maximum.to(
                    dtype=subtraction_torch_dtype
                ) - minimum.to(dtype=subtraction_torch_dtype)
                result = result.to(dtype=out.tensor.dtype)
        except (TypeError, RuntimeError):
            if out is not None:
                raise
            return NotImplemented

        if out is not None:
            if out.shape != tuple(result.shape):
                raise ValueError("output parameter has the wrong shape")
            out.tensor.copy_(result)
            return out
        if result.ndim == 0:
            return np.dtype(array.dtype).type(result.item())
        return self._wrap(result)

    def _numpy_ravel(self, args, kwargs):
        """Flatten a Torch array through NumPy's public ravel entry point."""
        options = dict(kwargs)
        if not args or len(args) > 2:
            return NotImplemented
        array = args[0]
        if not isinstance(array, TorchArrayData) or array is not self:
            return NotImplemented
        if len(args) == 2:
            if "order" in options:
                return NotImplemented
            order = args[1]
        else:
            order = options.pop("order", "C")
        if options:
            return NotImplemented
        return self.numpy_ravel(order=order)

    def _numpy_copy(self, args, kwargs):
        """Copy an array on its Torch device with NumPy layout semantics."""
        if len(args) > 3:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"order": "K", "subok": False}
        if args:
            if "a" in options:
                return NotImplemented
            array = args[0]
        else:
            try:
                array = options.pop("a")
            except KeyError:
                return NotImplemented
        for name, value in zip(("order", "subok"), args[1:]):
            if name in options:
                return NotImplemented
            parameters[name] = value
        for name in tuple(parameters):
            if name in options:
                parameters[name] = options.pop(name)
        if (
                options
                or not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        order = parameters["order"]
        if order is None:
            order = "K"
        elif isinstance(order, (bytes, np.bytes_)):
            try:
                order = bytes(order).decode("ascii")
            except UnicodeDecodeError:
                order = ""
        elif isinstance(order, (str, np.str_)):
            order = str(order)
        else:
            raise TypeError(
                f"order must be str, not {type(order).__name__}"
            )
        order = order.upper()
        if order not in ("C", "F", "A", "K"):
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' "
                f"(got {parameters['order']!r})"
            )

        tensor = array.tensor
        if order == "A":
            order = "F" if (
                not tensor.is_contiguous()
                and self._is_fortran_contiguous(tensor)
            ) else "C"
        if order == "C":
            result = tensor.clone() if tensor.is_contiguous() \
                else tensor.contiguous()
        elif order == "F":
            if tensor.numel() == 0:
                strides = (0,) * tensor.ndim
            else:
                stride = 1
                strides = []
                for size in tensor.shape:
                    strides.append(stride)
                    stride *= size
                strides = tuple(strides)
            result = torch.empty_strided(
                tensor.shape,
                strides,
                dtype=tensor.dtype,
                device=tensor.device,
            ).copy_(tensor)
        else:
            result = tensor.clone(memory_format=torch.preserve_format)
        return self._wrap(result)

    @staticmethod
    def _numpy_like_shape(shape, source_shape):
        """Normalize a NumPy ``*_like`` shape without allocating on host."""
        if shape is None:
            return tuple(source_shape)
        if isinstance(shape, (bool, np.bool_)):
            raise TypeError(
                "expected a sequence of integers or a single integer, "
                f"got {shape!r}"
            )
        try:
            dimensions = (operator.index(shape),)
        except TypeError as scalar_error:
            if isinstance(shape, (str, bytes)) or not hasattr(
                    shape, "__iter__"):
                raise scalar_error
            dimensions = tuple(operator.index(value) for value in shape)
        if any(dimension < 0 for dimension in dimensions):
            raise ValueError("negative dimensions are not allowed")
        return dimensions

    @staticmethod
    def _numpy_like_order(order):
        """Normalize NumPy's C/F/A/K allocation-order argument."""
        if order is None:
            order = "K"
        elif isinstance(order, (bytes, np.bytes_)):
            try:
                order = bytes(order).decode("ascii")
            except UnicodeDecodeError:
                order = ""
        elif isinstance(order, (str, np.str_)):
            order = str(order)
        else:
            raise TypeError(
                f"order must be str, not {type(order).__name__}"
            )
        order = order.upper()
        if order not in ("C", "F", "A", "K"):
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' "
                f"(got {order!r})"
            )
        return order

    def _empty_like_ordered(self, source, shape, dtype, order):
        """Allocate a Torch tensor with NumPy ``empty_like`` strides."""
        if order == "A":
            order = "F" if (
                not source.is_contiguous()
                and self._is_fortran_contiguous(source)
            ) else "C"

        if not shape:
            return torch.empty(
                shape, dtype=dtype, device=source.device
            )
        if any(size == 0 for size in shape):
            return torch.empty_strided(
                shape,
                (0,) * len(shape),
                dtype=dtype,
                device=source.device,
            )

        if order == "K":
            if len(shape) != source.ndim:
                order = "C"
            elif source.is_contiguous() or source.ndim <= 1:
                order = "C"
            elif self._is_fortran_contiguous(source):
                order = "F"

        if order == "C":
            return torch.empty(
                shape, dtype=dtype, device=source.device
            )

        if order == "F":
            stride = 1
            strides = []
            for size in shape:
                strides.append(stride)
                stride *= size
            strides = tuple(strides)
        else:
            # NumPy's KEEPORDER allocator stably sorts axes by descending
            # absolute prototype stride, then packs the new shape in that
            # access order.  C/F-contiguous prototypes were handled above.
            permutation = sorted(
                range(source.ndim),
                key=lambda axis: (-abs(source.stride()[axis]), axis),
            )
            strides = [0] * len(shape)
            stride = 1
            for axis in reversed(permutation):
                strides[axis] = stride
                stride *= shape[axis]
            strides = tuple(strides)
        return torch.empty_strided(
            shape, strides, dtype=dtype, device=source.device
        )

    def _numpy_like_creator(self, function, args, kwargs):
        """Create NumPy-like arrays without copying their prototype to host."""
        options = dict(kwargs)
        is_full = function is np.full_like
        names = (
            ("a", "fill_value", "dtype", "order", "subok", "shape")
            if is_full else
            ("prototype", "dtype", "order", "subok", "shape")
        )
        if function is not np.empty_like:
            names = ("a",) + names[1:]
        if len(args) > len(names):
            return NotImplemented

        parameters = {
            "dtype": None,
            "order": "K",
            "subok": True,
            "shape": None,
        }
        for index, name in enumerate(names):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in ("a", "prototype", "fill_value"):
                return NotImplemented

        device = options.pop("device", None)
        if options:
            return NotImplemented

        array = parameters.get("a", parameters.get("prototype"))
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented
        if device is not None:
            if device != "cpu":
                raise ValueError(
                    'Device not understood. Only "cpu" is allowed, '
                    f"but received: {device}"
                )
            if array.tensor.device.type != "cpu":
                return NotImplemented

        # NumPy accepts integer-like values for this subclass toggle.  The
        # backend result is always wrapped as a plain PyCBC Array.
        operator.index(parameters["subok"])
        shape = self._numpy_like_shape(
            parameters["shape"], array.shape
        )
        order = self._numpy_like_order(parameters["order"])
        if parameters["dtype"] is None:
            numpy_dtype = np.dtype(array.dtype)
        else:
            try:
                numpy_dtype = np.dtype(parameters["dtype"])
            except TypeError:
                return NotImplemented
        try:
            torch_dtype = _ensure_supported(
                array.tensor.device, _torch_dtype(numpy_dtype)
            )
        except (TypeError, RuntimeError):
            return NotImplemented

        result = self._empty_like_ordered(
            array.tensor, shape, torch_dtype, order
        )
        if function is np.empty_like:
            return self._wrap(result)
        if function is np.zeros_like:
            result.zero_()
            return self._wrap(result)
        if function is np.ones_like:
            result.fill_(1)
            return self._wrap(result)

        fill_value = parameters["fill_value"]
        if isinstance(fill_value, TorchArrayData):
            if (
                    not _device_matches_active(fill_value.tensor)
                    or fill_value.tensor.device != result.device):
                return NotImplemented
            fill_tensor = fill_value.tensor
        else:
            try:
                fill_array = np.asarray(fill_value)
            except (TypeError, ValueError):
                return NotImplemented
            if fill_array.dtype.kind not in "biufc":
                return NotImplemented
            fill_tensor = torch.as_tensor(
                fill_array, dtype=result.dtype, device=result.device
            )
        try:
            result.copy_(fill_tensor)
        except RuntimeError as exc:
            if "shape" in str(exc).lower() or "size" in str(exc).lower():
                raise ValueError(str(exc)) from exc
            return NotImplemented
        return self._wrap(result)

    def _numpy_tile(self, args, kwargs):
        """Construct a tiled copy without materializing the source on host."""
        if len(args) > 2:
            return NotImplemented

        options = dict(kwargs)
        parameters = {}
        for index, name in enumerate(("A", "reps")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            else:
                return NotImplemented
        array = parameters["A"]
        if (
                options
                or not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        reps = parameters["reps"]
        if isinstance(reps, (TorchArrayData, torch.Tensor)):
            return NotImplemented
        try:
            repetitions = (operator.index(reps),)
        except TypeError as scalar_error:
            try:
                repetitions = tuple(operator.index(value) for value in reps)
            except TypeError:
                if isinstance(reps, (str, bytes)) or not hasattr(
                        reps, "__iter__"):
                    raise scalar_error
                raise
        if any(repetition < 0 for repetition in repetitions):
            raise ValueError("negative dimensions are not allowed")

        dimensions = array.tensor.ndim
        if len(repetitions) > dimensions:
            dimensions = len(repetitions)
        if dimensions == 0:
            result = array.tensor.clone()
            return self._wrap(result)

        shape = (1,) * (dimensions - array.tensor.ndim) + array.tensor.shape
        repetitions = (1,) * (dimensions - len(repetitions)) + repetitions
        interleaved_shape = tuple(
            value for size in shape for value in (1, size)
        )
        expanded_shape = tuple(
            value
            for repetition, size in zip(repetitions, shape)
            for value in (repetition, size)
        )
        output_shape = tuple(
            repetition * size
            for repetition, size in zip(repetitions, shape)
        )
        try:
            result = (
                array.tensor.reshape(shape)
                .reshape(interleaved_shape)
                .expand(expanded_shape)
                .clone()
                .reshape(output_shape)
            )
        except RuntimeError:
            return NotImplemented
        return self._wrap(result)

    def _numpy_pad(self, args, kwargs):
        """Pad common NumPy modes without materializing data on the host."""
        if len(args) > 3:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"mode": "constant"}
        for index, name in enumerate(("array", "pad_width", "mode")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name != "mode":
                return NotImplemented

        array = parameters["array"]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        mode = parameters["mode"]
        if mode not in ("constant", "edge"):
            return NotImplemented
        if mode == "constant":
            if any(key != "constant_values" for key in options):
                return NotImplemented
            constant_values = options.get("constant_values", 0)
        elif options:
            return NotImplemented

        pad_width = parameters["pad_width"]
        if isinstance(pad_width, (TorchArrayData, torch.Tensor)):
            return NotImplemented
        try:
            width_array = np.asarray(pad_width)
        except (TypeError, ValueError):
            return NotImplemented
        if width_array.dtype.kind not in "iu":
            return NotImplemented
        try:
            widths = np.broadcast_to(
                width_array, (array.tensor.ndim, 2)
            )
        except ValueError:
            return NotImplemented
        if np.any(widths < 0):
            return NotImplemented
        widths = tuple(
            (operator.index(pair[0]), operator.index(pair[1]))
            for pair in widths
        )

        if mode == "constant":
            if isinstance(constant_values, (TorchArrayData, torch.Tensor)):
                return NotImplemented
            try:
                constant_array = np.asarray(constant_values)
                constants = np.broadcast_to(
                    constant_array, (array.tensor.ndim, 2)
                )
            except (TypeError, ValueError):
                return NotImplemented
            if constant_array.dtype.kind not in "biufc":
                return NotImplemented
            if (
                    constant_array.dtype.kind == "c"
                    and not array.tensor.is_complex()):
                return NotImplemented
            try:
                cast_constants = []
                for pair in constants:
                    cast_pair = []
                    for value in pair:
                        scalar = np.empty((), dtype=np.dtype(array.dtype))
                        scalar[()] = value.item()
                        cast_pair.append(scalar.item())
                    cast_constants.append(tuple(cast_pair))
            except (OverflowError, TypeError, ValueError):
                return NotImplemented
            constants = tuple(cast_constants)

        result = array.tensor.clone()
        for dimension, (before, after) in enumerate(widths):
            if before == 0 and after == 0:
                continue
            if mode == "edge" and result.shape[dimension] == 0:
                return NotImplemented

            pieces = []
            for amount, side in ((before, 0), (after, -1)):
                if amount == 0:
                    pieces.append(None)
                    continue
                shape = list(result.shape)
                shape[dimension] = amount
                if mode == "constant":
                    value = constants[dimension][0 if side == 0 else 1]
                    try:
                        piece = torch.full(
                            shape,
                            value,
                            dtype=result.dtype,
                            device=result.device,
                        )
                    except (TypeError, ValueError, RuntimeError):
                        return NotImplemented
                else:
                    index = 0 if side == 0 else result.shape[dimension] - 1
                    piece = result.select(dimension, index)
                    piece = piece.unsqueeze(dimension).expand(shape)
                pieces.append(piece)

            sequence = []
            if pieces[0] is not None:
                sequence.append(pieces[0])
            sequence.append(result)
            if pieces[1] is not None:
                sequence.append(pieces[1])
            try:
                result = torch.cat(sequence, dim=dimension)
            except RuntimeError:
                return NotImplemented
        return self._wrap(result)

    def _numpy_boolean_selection(self, function, args, kwargs):
        """Select values by a boolean condition without leaving Torch."""
        options = dict(kwargs)
        if function is np.compress:
            if len(args) > 4:
                return NotImplemented
            parameters = {"axis": None, "out": None}
            names = ("condition", "a", "axis", "out")
        else:
            if len(args) > 2:
                return NotImplemented
            parameters = {}
            names = ("condition", "arr")

        for index, name in enumerate(names):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name not in parameters:
                return NotImplemented
        if options or parameters.get("out") is not None:
            return NotImplemented

        array = parameters["a" if function is np.compress else "arr"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        condition = parameters["condition"]
        if isinstance(condition, TorchArrayData):
            if (
                    not _device_matches_active(condition.tensor)
                    or condition.tensor.device != array.tensor.device):
                return NotImplemented
            condition_tensor = condition.tensor
        elif isinstance(condition, torch.Tensor):
            if condition.device != array.tensor.device:
                return NotImplemented
            condition_tensor = condition
        else:
            try:
                condition_array = np.asarray(condition)
            except (TypeError, ValueError):
                return NotImplemented
            if condition_array.dtype.kind not in "biufc":
                return NotImplemented
            try:
                condition_tensor = torch.as_tensor(
                    condition_array, device=array.tensor.device
                )
            except (TypeError, ValueError, RuntimeError):
                return NotImplemented

        source = array.tensor
        if function is np.extract:
            condition_tensor = condition_tensor.reshape(-1)
            source = source.reshape(-1)
            dimension = 0
        else:
            if condition_tensor.ndim != 1:
                raise ValueError("condition must be a 1-d array")
            axis = parameters["axis"]
            if axis is None:
                source = source.reshape(-1)
                dimension = 0
            else:
                try:
                    original_axis = operator.index(axis)
                except TypeError:
                    return NotImplemented
                dimension = (
                    original_axis + source.ndim
                    if original_axis < 0 else original_axis
                )
                if dimension < 0 or dimension >= source.ndim:
                    raise np.exceptions.AxisError(
                        original_axis, ndim=source.ndim
                    )

        try:
            indices = torch.nonzero(
                condition_tensor.to(dtype=torch.bool), as_tuple=False
            ).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        # NumPy's take loop has no outer iterations when a dimension before
        # the selected axis is empty.  It therefore returns the correctly
        # shaped empty result without inspecting otherwise out-of-range
        # indices.  Torch validates those indices eagerly.
        if dimension and 0 in source.shape[:dimension]:
            result_shape = list(source.shape)
            result_shape[dimension] = indices.numel()
            return array._wrap(source.reshape(result_shape))
        if condition_tensor.shape[0] > source.shape[dimension]:
            invalid = indices >= source.shape[dimension]
            if bool(torch.any(invalid).item()):
                raise IndexError("index out of range in self")
        try:
            selection_source = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32 else source
            )
            result = torch.index_select(
                selection_source, dimension, indices
            )
            if source.dtype == _TORCH_UINT32:
                result = result.to(dtype=_TORCH_UINT32)
        except IndexError:
            raise
        except RuntimeError:
            return NotImplemented
        return array._wrap(result)

    def _numpy_append(self, args, kwargs):
        """Append values with NumPy promotion while retaining Torch storage."""
        if len(args) > 3:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"axis": None}
        for index, name in enumerate(("arr", "values", "axis")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name != "axis":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["arr"]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        values = parameters["values"]
        if isinstance(values, TorchArrayData):
            if (
                    not _device_matches_active(values.tensor)
                    or values.tensor.device != array.tensor.device):
                return NotImplemented
            values_tensor = values.tensor
            values_dtype = np.dtype(values.dtype)
        elif isinstance(values, torch.Tensor):
            if values.device != array.tensor.device:
                return NotImplemented
            try:
                values_dtype = np.dtype(_numpy_dtype(values.dtype))
            except TypeError:
                return NotImplemented
            values_tensor = values
        else:
            try:
                values_array = np.asarray(values)
            except (TypeError, ValueError):
                return NotImplemented
            if values_array.dtype.kind not in "biufc":
                return NotImplemented
            values_dtype = values_array.dtype
            try:
                values_tensor = torch.as_tensor(
                    values_array, device=array.tensor.device
                )
            except (TypeError, ValueError, RuntimeError):
                return NotImplemented

        try:
            target_numpy = np.dtype(np.result_type(array.dtype, values_dtype))
            target_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(target_numpy)
            )
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        axis = parameters["axis"]
        if axis is None:
            left = array.tensor.reshape(-1)
            right = values_tensor.reshape(-1)
            dimension = 0
        else:
            if isinstance(axis, (bool, np.bool_)):
                raise TypeError("an integer is required for the axis")
            try:
                original_axis = operator.index(axis)
            except TypeError:
                raise TypeError(
                    f"'{type(axis).__name__}' object cannot be interpreted "
                    "as an integer"
                ) from None
            dimension = original_axis
            if dimension < 0:
                dimension += array.tensor.ndim
            if dimension < 0 or dimension >= array.tensor.ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=array.tensor.ndim
                )
            left = array.tensor
            right = values_tensor

        try:
            result = torch.cat(
                (
                    left.to(dtype=target_torch),
                    right.to(dtype=target_torch),
                ),
                dim=dimension,
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return self._wrap(result)

    def _numpy_resize(self, args, kwargs):
        """Repeat flattened data into a new shape without leaving Torch."""
        if len(args) > 2:
            return NotImplemented

        options = dict(kwargs)
        parameters = {}
        for index, name in enumerate(("a", "new_shape")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        new_shape = parameters["new_shape"]
        if isinstance(new_shape, (bool, np.bool_)):
            raise TypeError("an integer is required")
        if isinstance(new_shape, (int, np.integer)):
            shape = (operator.index(new_shape),)
        else:
            shape = tuple(operator.index(value) for value in new_shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(
                "all elements of `new_shape` must be non-negative"
            )

        new_size = 1
        for dimension in shape:
            new_size *= dimension

        flattened = array.tensor.reshape(-1)
        if flattened.numel() == 0 or new_size == 0:
            result = torch.zeros(
                shape, dtype=array.tensor.dtype, device=array.tensor.device
            )
            return self._wrap(result)

        repetitions = (
            new_size + flattened.numel() - 1
        ) // flattened.numel()
        result = flattened.repeat(repetitions)[:new_size].reshape(shape)
        return self._wrap(result)

    def _numpy_delete(self, args, kwargs):
        """Delete indexed slices without copying Torch data to the host."""
        if len(args) > 3:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"axis": None}
        for index, name in enumerate(("arr", "obj", "axis")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name != "axis":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["arr"]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        source = array.tensor
        axis = parameters["axis"]
        if axis is None:
            source = source.reshape(-1)
            dimension = 0
        else:
            try:
                original_axis = operator.index(axis)
            except TypeError:
                return NotImplemented
            dimension = (
                original_axis + source.ndim
                if original_axis < 0 else original_axis
            )
            if dimension < 0 or dimension >= source.ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=source.ndim
                )

        size = source.shape[dimension]
        obj = parameters["obj"]
        if isinstance(obj, slice):
            start, stop, step = obj.indices(size)
            if len(range(start, stop, step)) == 0:
                return self._wrap(source.clone())
            delete_indices = torch.arange(
                start,
                stop,
                step,
                dtype=torch.int64,
                device=source.device,
            )
        else:
            single_value = (
                isinstance(obj, (int, np.integer))
                and not isinstance(obj, (bool, np.bool_))
            )
            host_indices = False
            original_obj = obj
            if single_value:
                index_values = operator.index(obj)
                index_kind = "i"
            elif isinstance(obj, TorchArrayData):
                if (
                        obj.tensor.device != source.device
                        or not _device_matches_active(obj.tensor)):
                    return NotImplemented
                index_values = obj.tensor
                index_kind = np.dtype(obj.dtype).kind
                if index_values.numel() == 1 and index_kind in "iu":
                    index_values = int(index_values.item())
                    single_value = True
            else:
                try:
                    index_values = np.asarray(obj)
                except (OverflowError, TypeError, ValueError):
                    return NotImplemented
                if (
                        index_values.size == 0
                        and not isinstance(original_obj, np.ndarray)):
                    index_values = index_values.astype(np.intp)
                index_kind = index_values.dtype.kind
                if index_values.size == 1 and index_kind in "iu":
                    index_values = int(index_values.item())
                    single_value = True
                host_indices = True

            if single_value:
                if index_values < -size or index_values >= size:
                    raise IndexError(
                        f"index {index_values} is out of bounds for axis "
                        f"{dimension} with size {size}"
                    )
                if index_values < 0:
                    index_values += size
                delete_indices = torch.tensor(
                    [index_values], dtype=torch.int64, device=source.device
                )
            elif index_kind == "b":
                if tuple(index_values.shape) != (size,):
                    raise ValueError(
                        "boolean array argument obj to delete must be one "
                        "dimensional and match the axis length of "
                        f"{size}"
                    )
                try:
                    delete_mask = torch.as_tensor(
                        index_values,
                        dtype=torch.bool,
                        device=source.device,
                    )
                except (TypeError, ValueError, RuntimeError):
                    return NotImplemented
                delete_indices = torch.nonzero(
                    delete_mask, as_tuple=False
                ).reshape(-1)
            elif index_kind in "iu":
                if host_indices:
                    flattened_indices = np.asarray(index_values).reshape(-1)
                    invalid = flattened_indices >= size
                    if index_kind == "i":
                        invalid |= flattened_indices < -size
                    if np.any(invalid):
                        bad_index = int(flattened_indices[invalid][0])
                        raise IndexError(
                            f"index {bad_index} is out of bounds for axis "
                            f"{dimension} with size {size}"
                        )
                    flattened_indices = np.where(
                        flattened_indices < 0,
                        flattened_indices + size,
                        flattened_indices,
                    ).astype(np.int64, copy=False)
                    try:
                        delete_indices = torch.as_tensor(
                            flattened_indices, device=source.device
                        )
                    except (TypeError, ValueError, RuntimeError):
                        return NotImplemented
                else:
                    flattened_indices = index_values.reshape(-1)
                    invalid = flattened_indices >= size
                    if index_kind == "i":
                        invalid = invalid | (flattened_indices < -size)
                    if bool(torch.any(invalid).item()):
                        bad_index = int(flattened_indices[invalid][0].item())
                        raise IndexError(
                            f"index {bad_index} is out of bounds for axis "
                            f"{dimension} with size {size}"
                        )
                    if index_kind == "i":
                        flattened_indices = torch.where(
                            flattened_indices < 0,
                            flattened_indices + size,
                            flattened_indices,
                        )
                    delete_indices = flattened_indices.to(dtype=torch.int64)
            else:
                return NotImplemented

        delete_mask = torch.zeros(
            size, dtype=torch.bool, device=source.device
        )
        delete_mask[delete_indices] = True
        keep_indices = torch.nonzero(
            torch.logical_not(delete_mask), as_tuple=False
        ).reshape(-1)
        try:
            selection_source = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32 else source
            )
            result = torch.index_select(
                selection_source, dimension, keep_indices
            )
            if source.dtype == _TORCH_UINT32:
                result = result.to(dtype=_TORCH_UINT32)
        except RuntimeError:
            return NotImplemented
        return self._wrap(result)

    def _numpy_putmask(self, args, kwargs):
        """Replace selected values in place without leaving Torch."""
        if kwargs or len(args) != 3:
            return NotImplemented

        array, mask, values = args
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        device = array.tensor.device
        target_dtype = np.dtype(array.dtype)
        try:
            if isinstance(mask, TorchArrayData):
                if (
                        not _device_matches_active(mask.tensor)
                        or mask.tensor.device != device):
                    return NotImplemented
                mask_tensor = mask.tensor.to(dtype=torch.bool).reshape(-1)
            else:
                mask_values = np.asarray(mask, dtype=np.bool_)
                mask_tensor = torch.as_tensor(
                    mask_values.reshape(-1),
                    dtype=torch.bool,
                    device=device,
                )
            if mask_tensor.numel() != array.tensor.numel():
                raise ValueError(
                    "putmask: mask and data must be the same size"
                )

            target_torch = array.tensor.dtype
            if isinstance(values, TorchArrayData):
                if (
                        not _device_matches_active(values.tensor)
                        or values.tensor.device != device):
                    return NotImplemented
                values_dtype = np.dtype(values.dtype)
                if not np.can_cast(values_dtype, target_dtype, casting="safe"):
                    raise TypeError(
                        "Cannot cast array data from dtype"
                        f"('{values_dtype}') to dtype('{target_dtype}') "
                        "according to the rule 'safe'"
                    )
                values_tensor = values.tensor.to(
                    dtype=target_torch
                ).reshape(-1)
            elif isinstance(values, np.ndarray):
                if not np.can_cast(
                        values.dtype, target_dtype, casting="safe"):
                    raise TypeError(
                        "Cannot cast array data from dtype"
                        f"('{values.dtype}') to dtype('{target_dtype}') "
                        "according to the rule 'safe'"
                    )
                values_tensor = torch.as_tensor(
                    values.astype(target_dtype, copy=False).reshape(-1),
                    dtype=target_torch,
                    device=device,
                )
            else:
                converted = np.asarray(values, dtype=target_dtype)
                values_tensor = torch.as_tensor(
                    converted.reshape(-1),
                    dtype=target_torch,
                    device=device,
                )

            if values_tensor.numel() == 0 or array.tensor.numel() == 0:
                return None
            indices = torch.arange(
                array.tensor.numel(), device=device
            ).remainder(values_tensor.numel())
            execution_dtype = (
                torch.int64
                if target_torch == _TORCH_UINT32 else target_torch
            )
            replacements = values_tensor.to(
                dtype=execution_dtype
            )[indices]
            flattened = array.tensor.reshape(-1).to(
                dtype=execution_dtype
            )
            updated = torch.where(mask_tensor, replacements, flattened)
            if updated.dtype != target_torch:
                updated = updated.to(dtype=target_torch)
            array.tensor.copy_(updated.reshape(array.tensor.shape))
        except (OverflowError, TypeError, ValueError):
            raise
        except RuntimeError as exc:
            if "same size" in str(exc).lower():
                raise ValueError(str(exc)) from exc
            return NotImplemented
        return None

    def _numpy_complex_component(self, function, args, kwargs):
        """Extract real, imaginary, or phase data within Torch."""
        names = ("z", "deg") if function is np.angle else ("val",)
        if len(args) > len(names):
            return NotImplemented

        options = dict(kwargs)
        parameters = {"deg": False}
        for index, name in enumerate(names):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name != "deg":
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters[names[0]]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented

        if function is np.real:
            return array.real
        if function is np.imag:
            return array.imag

        degrees = parameters["deg"]
        if isinstance(degrees, (TorchArrayData, torch.Tensor)):
            return NotImplemented
        try:
            degrees = bool(degrees)
        except (TypeError, ValueError):
            return NotImplemented

        source_dtype = np.dtype(array.dtype)
        output_dtype = (
            np.dtype(np.float32)
            if source_dtype in (np.dtype(np.float32), np.dtype(np.complex64))
            else np.dtype(np.float64)
        )
        try:
            output_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(output_dtype)
            )
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        execution = array.tensor
        if source_dtype.kind in "biu":
            execution = execution.to(dtype=output_torch)
        result = torch.angle(execution)
        if degrees:
            result = torch.rad2deg(result)
        return self._wrap(result)

    def _numpy_unwrap(self, args, kwargs):
        """Unwrap real floating-point phase data within Torch."""
        names = ("p", "discont", "axis")
        if len(args) > len(names):
            return NotImplemented

        options = dict(kwargs)
        parameters = {"discont": None, "axis": -1}
        for index, name in enumerate(names):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "p":
                return NotImplemented
        period = options.pop("period", 2 * np.pi)
        if options:
            return NotImplemented

        array = parameters["p"]
        if (
                not isinstance(array, TorchArrayData)
                or array is not self
                or not _device_matches_active(array.tensor)):
            return NotImplemented
        if array.ndim == 0:
            raise ValueError(
                "diff requires input that is at least one dimensional"
            )

        axis_value = parameters["axis"]
        if isinstance(axis_value, (bool, np.bool_)):
            return NotImplemented
        try:
            original_axis = operator.index(axis_value)
        except TypeError:
            return NotImplemented
        axis = original_axis
        if axis < 0:
            axis += array.ndim
        if axis < 0 or axis >= array.ndim:
            raise np.exceptions.AxisError(original_axis, ndim=array.ndim)

        input_dtype = np.dtype(array.dtype)
        if input_dtype.kind != "f":
            return NotImplemented
        period_array = np.asarray(period)
        if period_array.ndim != 0 or period_array.dtype.kind not in "biuf":
            return NotImplemented
        discont = parameters["discont"]
        if discont is None:
            try:
                discont = period / 2
            except (TypeError, ValueError):
                return NotImplemented
        discont_array = np.asarray(discont)
        if (
                discont_array.ndim != 0
                or discont_array.dtype.kind not in "biuf"):
            return NotImplemented

        try:
            dtype_probe = np.empty((), dtype=input_dtype)
            target_dtype = np.dtype(np.result_type(dtype_probe, period))
            comparison_dtype = np.dtype(
                np.result_type(dtype_probe, discont)
            )
            target_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(target_dtype)
            )
            comparison_torch = _ensure_supported(
                array.tensor.device, _torch_dtype(comparison_dtype)
            )
            period_tensor = torch.as_tensor(
                period_array.item(),
                dtype=target_torch,
                device=array.tensor.device,
            )
            discont_tensor = torch.as_tensor(
                discont_array.item(),
                dtype=comparison_torch,
                device=array.tensor.device,
            )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        source = array.tensor
        source_target = source.to(dtype=target_torch)
        if source.shape[axis] < 2:
            return self._wrap(source_target.clone())

        delta = torch.diff(source, dim=axis)
        delta_target = delta.to(dtype=target_torch)
        interval_high = period_tensor / 2
        interval_low = -interval_high
        delta_mod = (
            torch.remainder(delta_target - interval_low, period_tensor)
            + interval_low
        )
        delta_mod = torch.where(
            (delta_mod == interval_low) & (delta_target > 0),
            interval_high,
            delta_mod,
        )
        correction = delta_mod - delta_target
        small_delta = (
            torch.abs(delta).to(dtype=comparison_torch) < discont_tensor
        )
        correction = torch.where(
            small_delta,
            torch.zeros_like(correction),
            correction,
        )

        prefix = source_target.narrow(axis, 0, 1)
        suffix = source_target.narrow(
            axis, 1, source.shape[axis] - 1
        ) + torch.cumsum(correction, dim=axis)
        return self._wrap(torch.cat((prefix, suffix), dim=axis))

    @staticmethod
    def _is_fortran_contiguous(tensor):
        """Return whether a tensor has NumPy-compatible Fortran strides."""
        expected_stride = 1
        for size, stride in zip(tensor.shape, tensor.stride()):
            if size == 0:
                return True
            if size != 1:
                if stride != expected_stride:
                    return False
                expected_stride *= size
        return True

    def _numpy_take_along_axis(self, args, kwargs):
        """Gather along one axis without exposing Torch backend storage."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("arr", "indices", "axis")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "axis":
                parameters[name] = -1
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["arr"]
        indices = parameters["indices"]
        if (
            not isinstance(array, TorchArrayData)
            or array is not self
            or not _device_matches_active(array.tensor)
        ):
            return NotImplemented

        if isinstance(indices, TorchArrayData):
            if indices.tensor.device != array.tensor.device:
                return NotImplemented
            index_tensor = indices.tensor
            index_kind = np.dtype(indices.dtype).kind
        elif isinstance(indices, torch.Tensor):
            if indices.device != array.tensor.device:
                return NotImplemented
            index_tensor = indices
            if indices.dtype == torch.bool:
                index_kind = "b"
            else:
                try:
                    index_kind = _numpy_dtype(indices.dtype).kind
                except TypeError:
                    index_kind = "O"
        else:
            try:
                index_values = np.asarray(indices)
                index_kind = index_values.dtype.kind
                if not index_values.flags.c_contiguous:
                    index_values = np.ascontiguousarray(index_values)
                index_tensor = torch.as_tensor(
                    index_values, device=array.tensor.device
                )
            except (OverflowError, TypeError, ValueError, RuntimeError):
                return NotImplemented
        if index_kind not in "iu":
            raise IndexError("`indices` must be an integer array")

        source = array.tensor
        axis = parameters["axis"]
        if axis is None:
            if index_tensor.ndim != 1:
                raise ValueError(
                    "when axis=None, `indices` must have a single dimension."
                )
            source = source.reshape(-1)
            dimension = 0
        else:
            try:
                original_axis = operator.index(axis)
            except TypeError:
                return NotImplemented
            dimension = (
                original_axis + source.ndim
                if original_axis < 0 else original_axis
            )
            if dimension < 0 or dimension >= source.ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=source.ndim
                )
            if index_tensor.ndim != source.ndim:
                raise ValueError(
                    "`indices` and `arr` must have the same number of "
                    "dimensions"
                )

        try:
            index_tensor = index_tensor.to(dtype=torch.int64)
            size = source.shape[dimension]
            if size:
                index_tensor = torch.where(
                    index_tensor < 0, index_tensor + size, index_tensor
                )
            execution = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32 else source
            )
            if execution.is_complex():
                result = torch.complex(
                    torch.take_along_dim(
                        execution.real, index_tensor, dim=dimension
                    ),
                    torch.take_along_dim(
                        execution.imag, index_tensor, dim=dimension
                    ),
                )
            else:
                result = torch.take_along_dim(
                    execution, index_tensor, dim=dimension
                )
            if source.dtype == _TORCH_UINT32:
                result = result.to(dtype=_TORCH_UINT32)
        except (IndexError, RuntimeError) as exc:
            raise IndexError(str(exc)) from exc
        return self._wrap(result)

    def _numpy_unique(self, args, kwargs):
        """Find flattened real unique values without a host-array copy."""
        options = dict(kwargs)
        if len(args) > 5:
            return NotImplemented

        defaults = {
            "return_index": False,
            "return_inverse": False,
            "return_counts": False,
            "axis": None,
            "equal_nan": True,
            "sorted": True,
        }
        parameters = {}
        for index, name in enumerate((
                "ar", "return_index", "return_inverse", "return_counts",
                "axis")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        for name in ("equal_nan", "sorted"):
            parameters[name] = options.pop(name, defaults[name])
        if options:
            return NotImplemented

        array = parameters["ar"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biuf"
                or parameters["axis"] is not None):
            return NotImplemented

        flags = {}
        for name in (
                "return_index", "return_inverse", "return_counts",
                "equal_nan", "sorted"):
            value = parameters[name]
            if not np.isscalar(value):
                return NotImplemented
            try:
                flags[name] = bool(value)
            except (TypeError, ValueError):
                return NotImplemented
        if not flags["sorted"]:
            return NotImplemented

        tensor = array.tensor
        flattened = tensor.reshape(-1)
        try:
            if np.dtype(array.dtype).kind == "f" and flags["equal_nan"]:
                nan_mask = torch.isnan(flattened)
                ordinary = flattened[~nan_mask]
                values, ordinary_inverse, counts = torch.unique(
                    ordinary,
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )
                inverse = torch.empty(
                    flattened.shape,
                    dtype=torch.int64,
                    device=tensor.device,
                )
                inverse[~nan_mask] = ordinary_inverse
                if torch.any(nan_mask):
                    nan_index = values.numel()
                    values = torch.cat((
                        values,
                        torch.full(
                            (1,), torch.nan,
                            dtype=tensor.dtype,
                            device=tensor.device,
                        ),
                    ))
                    inverse[nan_mask] = nan_index
                    counts = torch.cat((
                        counts,
                        nan_mask.sum(dtype=torch.int64).reshape(1),
                    ))
            else:
                values, inverse, counts = torch.unique(
                    flattened,
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )

            first_indices = None
            if flags["return_index"] or (
                    np.dtype(array.dtype).kind == "f" and values.numel()):
                order = torch.argsort(inverse, stable=True)
                offsets = torch.cumsum(counts, dim=0) - counts
                first_indices = order[offsets]
            if np.dtype(array.dtype).kind == "f" and values.numel():
                values = torch.where(
                    values == 0,
                    flattened[first_indices],
                    values,
                )

            inverse = inverse.reshape(tensor.shape)
            outputs = [self._wrap(values)]
            if flags["return_index"]:
                outputs.append(self._wrap(first_indices))
            if flags["return_inverse"]:
                outputs.append(self._wrap(inverse))
            if flags["return_counts"]:
                outputs.append(self._wrap(counts))
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _numpy_histogram_weights(self, value, shape):
        """Prepare real histogram weights on this Torch device."""
        if isinstance(value, TorchArrayData):
            if (
                    not _device_matches_active(value.tensor)
                    or value.tensor.device != self.tensor.device):
                raise TypeError
            weights = value.tensor
            dtype = np.dtype(value.dtype)
        elif isinstance(value, torch.Tensor):
            if value.device != self.tensor.device:
                raise TypeError
            weights = value
            dtype = _numpy_dtype(value.dtype)
        else:
            values = np.asarray(value)
            dtype = np.dtype(values.dtype)
            torch_dtype = _ensure_supported(
                self.tensor.device, _torch_dtype(dtype)
            )
            if values.ndim and not values.flags.c_contiguous:
                values = np.ascontiguousarray(values)
            weights = torch.as_tensor(
                values,
                dtype=torch_dtype,
                device=self.tensor.device,
            )
        if tuple(weights.shape) != tuple(shape):
            raise ValueError("weights should have the same shape as a.")
        if dtype.kind not in "iuf" or dtype.kind == "b":
            raise TypeError
        return weights.reshape(-1), dtype

    def _numpy_histogram_explicit_edges(self, bins):
        """Prepare explicit real histogram edges on this Torch device."""
        if isinstance(bins, TorchArrayData):
            if (
                    not _device_matches_active(bins.tensor)
                    or bins.tensor.device != self.tensor.device):
                raise TypeError
            edges = bins.tensor
            dtype = np.dtype(bins.dtype)
        elif isinstance(bins, torch.Tensor):
            if bins.device != self.tensor.device:
                raise TypeError
            edges = bins
            dtype = _numpy_dtype(bins.dtype)
        else:
            values = np.asarray(bins)
            if values.ndim != 1:
                raise ValueError("`bins` must be 1d, when an array")
            dtype = np.dtype(values.dtype)
            torch_dtype = _ensure_supported(
                self.tensor.device, _torch_dtype(dtype)
            )
            if values.ndim and not values.flags.c_contiguous:
                values = np.ascontiguousarray(values)
            edges = torch.as_tensor(
                values,
                dtype=torch_dtype,
                device=self.tensor.device,
            )
        if edges.ndim != 1:
            raise ValueError("`bins` must be 1d, when an array")
        if dtype.kind not in "biuf":
            raise TypeError
        if dtype.kind == "f" and bool(torch.any(torch.isnan(edges)).item()):
            # NumPy's search ordering for NaN edges is intentionally left to
            # its legacy implementation rather than approximated here.
            raise TypeError
        execution = (
            edges.to(dtype=torch.int64)
            if dtype.kind in "bu" else edges
        )
        if execution.numel() > 1 and bool(
                torch.any(execution[:-1] > execution[1:]).item()):
            raise ValueError(
                "`bins` must increase monotonically, when an array"
            )
        return edges, dtype

    def _numpy_histogram_generated_edges(self, tensor, dtype, count, limits):
        """Create NumPy-compatible uniform edges from device extrema."""
        if limits is None:
            if tensor.numel() == 0:
                first_edge, last_edge = 0, 1
            else:
                extrema = (
                    tensor.to(dtype=torch.int64)
                    if dtype.kind in "bu" else tensor
                )
                first_edge = dtype.type(torch.min(extrema).item())
                last_edge = dtype.type(torch.max(extrema).item())
            range_label = "autodetected"
        else:
            try:
                first_edge, last_edge = limits
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc
            if not (
                    np.isscalar(first_edge)
                    and np.isscalar(last_edge)
                    and np.asarray(first_edge).dtype.kind in "biuf"
                    and np.asarray(last_edge).dtype.kind in "biuf"):
                raise TypeError
            range_label = "supplied"

        if first_edge > last_edge:
            raise ValueError(
                "max must be larger than min in range parameter."
            )
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"{range_label} range of "
                f"[{first_edge}, {last_edge}] is not finite"
            )
        if first_edge == last_edge:
            first_edge = first_edge - 0.5
            last_edge = last_edge + 0.5

        edge_dtype = np.dtype(np.result_type(
            first_edge,
            last_edge,
            np.empty(0, dtype=dtype),
        ))
        if edge_dtype.kind in "biu":
            edge_dtype = np.dtype(np.result_type(edge_dtype, float))
        edge_torch = _ensure_supported(
            self.tensor.device, _torch_dtype(edge_dtype)
        )
        edge_values = np.linspace(
            first_edge,
            last_edge,
            count + 1,
            endpoint=True,
            dtype=edge_dtype,
        )
        if np.any(edge_values[:-1] >= edge_values[1:]):
            raise ValueError(
                f"Too many bins for data range. Cannot create {count} "
                "finite-sized bins."
            )
        edges = torch.as_tensor(
            edge_values,
            dtype=edge_torch,
            device=self.tensor.device,
        )
        return edges, edge_dtype

    def _numpy_histogram(self, args, kwargs):
        """Compute a numeric NumPy histogram without copying samples out."""
        options = dict(kwargs)
        if len(args) > 5:
            return NotImplemented

        defaults = {
            "bins": 10,
            "range": None,
            "density": None,
            "weights": None,
        }
        parameters = {}
        for index, name in enumerate((
                "a", "bins", "range", "density", "weights")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["a"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device):
            return NotImplemented
        input_dtype = np.dtype(array.dtype)
        if input_dtype.kind not in "biuf":
            return NotImplemented
        tensor = array.tensor.reshape(-1)

        density = parameters["density"]
        if density is not None and not np.isscalar(density):
            return NotImplemented
        try:
            density = bool(density)
        except (TypeError, ValueError):
            return NotImplemented

        weights = parameters["weights"]
        weight_dtype = None
        if weights is not None:
            try:
                weights, weight_dtype = self._numpy_histogram_weights(
                    weights, array.shape
                )
            except (OverflowError, TypeError, RuntimeError):
                return NotImplemented

        bins = parameters["bins"]
        uniform_count = None
        if isinstance(bins, str):
            return NotImplemented
        if isinstance(bins, (TorchArrayData, torch.Tensor)):
            explicit = True
        else:
            try:
                dimensions = np.ndim(bins)
            except (TypeError, ValueError):
                return NotImplemented
            explicit = dimensions != 0
            if not explicit:
                try:
                    uniform_count = operator.index(bins)
                except TypeError as exc:
                    raise TypeError(
                        "`bins` must be an integer, a string, or an array"
                    ) from exc
                if uniform_count < 1:
                    raise ValueError(
                        "`bins` must be positive, when an integer"
                    )

        try:
            if explicit:
                edges, edge_dtype = self._numpy_histogram_explicit_edges(
                    bins
                )
            else:
                edges, edge_dtype = self._numpy_histogram_generated_edges(
                    tensor,
                    input_dtype,
                    uniform_count,
                    parameters["range"],
                )
        except (OverflowError, TypeError, RuntimeError):
            return NotImplemented

        bin_count = edges.numel() - 1 if edges.numel() else 0
        if weights is None:
            histogram_dtype = np.dtype(np.int64)
            histogram_torch = torch.int64
        else:
            histogram_dtype = weight_dtype
            try:
                histogram_torch = _ensure_supported(
                    tensor.device, _torch_dtype(histogram_dtype)
                )
            except TypeError:
                return NotImplemented

        if bin_count == 0:
            histogram = torch.zeros(
                (0,), dtype=histogram_torch, device=tensor.device
            )
        else:
            try:
                comparison_dtype = np.dtype(np.result_type(
                    input_dtype, edge_dtype
                ))
                if comparison_dtype.kind in "bu":
                    comparison_torch = torch.int64
                else:
                    comparison_torch = _ensure_supported(
                        tensor.device, _torch_dtype(comparison_dtype)
                    )
                values = tensor.to(dtype=comparison_torch)
                boundaries = edges.to(dtype=comparison_torch)
                valid = (values >= boundaries[0]) & (
                    values <= boundaries[-1]
                )
                indices = torch.searchsorted(
                    boundaries, values, right=True
                ) - 1
                indices = torch.where(
                    values == boundaries[-1],
                    torch.full_like(indices, bin_count - 1),
                    indices,
                )
                valid &= (indices >= 0) & (indices < bin_count)
                indices = indices[valid].to(dtype=torch.int64)

                if weights is None:
                    source = torch.ones(
                        indices.shape,
                        dtype=torch.int64,
                        device=tensor.device,
                    )
                    histogram = torch.zeros(
                        (bin_count,),
                        dtype=torch.int64,
                        device=tensor.device,
                    ).scatter_add(0, indices, source)
                else:
                    execution_dtype = (
                        torch.int64
                        if histogram_dtype.kind == "u"
                        else histogram_torch
                    )
                    source = weights.to(dtype=execution_dtype)[valid]
                    histogram = torch.zeros(
                        (bin_count,),
                        dtype=execution_dtype,
                        device=tensor.device,
                    ).scatter_add(
                        0, indices, source
                    )
                    if histogram_dtype.kind == "u":
                        histogram = histogram.to(dtype=histogram_torch)
            except (IndexError, TypeError, ValueError, RuntimeError):
                return NotImplemented

        if density:
            density_torch = (
                torch.float32
                if tensor.device.type == "mps" else torch.float64
            )
            histogram = histogram.to(dtype=density_torch)
            widths = torch.diff(edges.to(dtype=density_torch))
            histogram = histogram / widths / torch.sum(histogram)

        return self._wrap(histogram), self._wrap(edges)

    def _numpy_digitize(self, args, kwargs):
        """Assign real values to monotonic bins without a host array copy."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("x", "bins", "right")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "right":
                parameters[name] = False
            else:
                return NotImplemented
        if options:
            return NotImplemented

        try:
            right = bool(parameters["right"])
        except (TypeError, ValueError):
            return NotImplemented

        device = self.tensor.device

        def describe(value):
            if isinstance(value, TorchArrayData):
                if (
                        not _device_matches_active(value.tensor)
                        or value.tensor.device != device):
                    raise TypeError
                dtype = np.dtype(value.dtype)
                tensor = value.tensor
                host = None
            elif isinstance(value, torch.Tensor):
                if value.device != device:
                    raise TypeError
                dtype = np.dtype(_numpy_dtype(value.dtype))
                tensor = value
                host = None
            else:
                host = np.asarray(value)
                dtype = np.dtype(host.dtype)
                if host.ndim and not host.flags.c_contiguous:
                    host = np.ascontiguousarray(host)
                tensor = None
            if dtype.kind not in "biuf":
                raise TypeError
            return tensor, host, dtype

        try:
            x_tensor, x_host, x_dtype = describe(parameters["x"])
            bins_tensor, bins_host, bins_dtype = describe(parameters["bins"])
            target_np = np.dtype(np.result_type(x_dtype, bins_dtype))
            if target_np == np.dtype(np.uint64):
                return NotImplemented
            try:
                target_torch = _ensure_supported(
                    device, _torch_dtype(target_np)
                )
            except TypeError:
                host_values = x_host if x_host is not None else bins_host
                device_dtype = (
                    bins_dtype if bins_tensor is not None else x_dtype
                )
                if host_values is None:
                    return NotImplemented
                target_torch = _comparison_dtype(
                    device_dtype,
                    target_np,
                    device,
                    other_values=host_values,
                )

            def materialize(tensor, host):
                if tensor is not None:
                    return tensor.to(dtype=target_torch)
                return torch.as_tensor(
                    host, dtype=target_torch, device=device
                )

            values = materialize(x_tensor, x_host)
            bins = materialize(bins_tensor, bins_host)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if bins.ndim != 1:
            return NotImplemented
        if bins.is_floating_point() and bool(torch.any(torch.isnan(bins))):
            return NotImplemented

        if bins.dtype == torch.bool or bins.dtype in (
                _TORCH_UINT16, _TORCH_UINT32):
            bins = bins.to(dtype=torch.int64)
            values = values.to(dtype=torch.int64)

        try:
            if bins.numel() < 2:
                increasing = True
                decreasing = True
            else:
                increasing = not bool(
                    torch.any(bins[:-1] > bins[1:]).item()
                )
                decreasing = not bool(
                    torch.any(bins[:-1] < bins[1:]).item()
                )
            if not (increasing or decreasing):
                raise ValueError(
                    "bins must be monotonically increasing or decreasing"
                )

            side_right = not right
            if increasing:
                result = torch.searchsorted(
                    bins.contiguous(),
                    values.contiguous(),
                    right=side_right,
                )
            else:
                result = bins.numel() - torch.searchsorted(
                    torch.flip(bins, dims=(0,)).contiguous(),
                    values.contiguous(),
                    right=side_right,
                )
        except (TypeError, RuntimeError):
            return NotImplemented

        if result.ndim == 0:
            return np.intp(result.item())
        return self._wrap(result)

    def _numpy_trapezoid(self, args, kwargs):
        """Integrate along one axis without copying data to the host."""
        if not args or len(args) > 4:
            return NotImplemented

        options = dict(kwargs)
        parameters = {"x": None, "dx": 1.0, "axis": -1}
        for name, value in zip(parameters, args[1:]):
            if name in options:
                return NotImplemented
            parameters[name] = value
        for name in tuple(parameters):
            if name in options:
                parameters[name] = options.pop(name)
        if options:
            return NotImplemented

        device = self.tensor.device

        def describe(value):
            if isinstance(value, TorchArrayData):
                if (
                        not _device_matches_active(value.tensor)
                        or value.tensor.device != device):
                    raise TypeError
                tensor = value.tensor
                host = None
                dtype = np.dtype(value.dtype)
            elif isinstance(value, torch.Tensor):
                if value.device != device:
                    raise TypeError
                tensor = value
                host = None
                dtype = np.dtype(_numpy_dtype(value.dtype))
            else:
                host = np.asarray(value)
                tensor = None
                dtype = np.dtype(host.dtype)
            if dtype.kind not in "biufc" or dtype.kind == "u":
                raise TypeError
            _ensure_supported(device, _torch_dtype(dtype))
            return tensor, host, dtype

        def materialize(tensor, host, dtype):
            torch_dtype = _ensure_supported(device, _torch_dtype(dtype))
            if tensor is not None:
                return tensor.to(dtype=torch_dtype)
            return torch.as_tensor(host, dtype=torch_dtype, device=device)

        try:
            y_tensor, y_host, y_dtype = describe(args[0])
            y = materialize(y_tensor, y_host, y_dtype)
            axis = operator.index(parameters["axis"])
            if y.ndim == 0 or not -y.ndim <= axis < y.ndim:
                return NotImplemented
            y_axis = axis % y.ndim

            lower = [slice(None)] * y.ndim
            upper = list(lower)
            lower[y_axis] = slice(None, -1)
            upper[y_axis] = slice(1, None)
            if y.dtype == torch.bool:
                y_pair = torch.logical_or(y[tuple(lower)], y[tuple(upper)])
            else:
                y_pair = y[tuple(lower)] + y[tuple(upper)]

            if parameters["x"] is None:
                dx_host = np.asarray(parameters["dx"])
                if dx_host.ndim != 0 or dx_host.dtype.kind not in "biufc":
                    return NotImplemented
                product_dtype = np.dtype(
                    np.result_type(np.empty(0, dtype=y_dtype), parameters["dx"])
                )
                distance = torch.as_tensor(
                    parameters["dx"],
                    dtype=_ensure_supported(
                        device, _torch_dtype(product_dtype)
                    ),
                    device=device,
                )
            else:
                x_tensor, x_host, x_dtype = describe(parameters["x"])
                x = materialize(x_tensor, x_host, x_dtype)
                if x.ndim == 0:
                    return NotImplemented
                if x.ndim == 1:
                    if x.dtype == torch.bool:
                        distance = torch.logical_xor(x[1:], x[:-1])
                    else:
                        distance = x[1:] - x[:-1]
                    shape = [1] * y.ndim
                    shape[y_axis] = distance.shape[0]
                    distance = distance.reshape(shape)
                else:
                    if not -x.ndim <= axis < x.ndim:
                        return NotImplemented
                    x_axis = axis % x.ndim
                    x_lower = [slice(None)] * x.ndim
                    x_upper = list(x_lower)
                    x_lower[x_axis] = slice(None, -1)
                    x_upper[x_axis] = slice(1, None)
                    if x.dtype == torch.bool:
                        distance = torch.logical_xor(
                            x[tuple(x_upper)], x[tuple(x_lower)]
                        )
                    else:
                        distance = (
                            x[tuple(x_upper)] - x[tuple(x_lower)]
                        )
                product_dtype = np.dtype(
                    np.result_type(
                        np.empty(0, dtype=y_dtype),
                        np.empty(0, dtype=x_dtype),
                    )
                )

            product_torch = _ensure_supported(
                device, _torch_dtype(product_dtype)
            )
            y_pair = y_pair.to(dtype=product_torch)
            distance = distance.to(dtype=product_torch)
            torch.broadcast_shapes(y_pair.shape, distance.shape)
            if product_torch == torch.bool:
                product = torch.logical_and(distance, y_pair)
            else:
                product = distance * y_pair

            if product_dtype.kind in "biu":
                output_dtype = np.dtype(np.float64)
            else:
                output_dtype = product_dtype
            output_torch = _ensure_supported(
                device, _torch_dtype(output_dtype)
            )
            result = torch.sum(
                product.to(dtype=output_torch) / 2.0,
                dim=y_axis,
            )
        except (IndexError, OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if result.ndim == 0:
            return output_dtype.type(result.item())
        return self._wrap(result)

    def _numpy_interp(self, args, kwargs):
        """Linearly interpolate real coordinates without a host array copy."""
        options = dict(kwargs)
        names = ("x", "xp", "fp", "left", "right", "period")
        if len(args) > len(names):
            return NotImplemented

        parameters = {"left": None, "right": None, "period": None}
        for index, name in enumerate(names):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name not in parameters:
                return NotImplemented
        if options:
            return NotImplemented

        device = self.tensor.device
        # NumPy always evaluates interpolation in double precision. MPS does
        # not provide float64 or complex128, so preserve the host fallback.
        if device.type == "mps":
            return NotImplemented

        def describe(value, *, complex_ok):
            if isinstance(value, TorchArrayData):
                if (
                        not _device_matches_active(value.tensor)
                        or value.tensor.device != device):
                    raise TypeError
                tensor = value.tensor
                dtype = np.dtype(value.dtype)
                host = None
            elif isinstance(value, torch.Tensor):
                if value.device != device:
                    raise TypeError
                tensor = value
                dtype = np.dtype(_numpy_dtype(value.dtype))
                host = None
            else:
                host = np.asarray(value)
                if host.ndim and not host.flags.c_contiguous:
                    host = np.ascontiguousarray(host)
                tensor = None
                dtype = np.dtype(host.dtype)
            valid_kinds = "biufc" if complex_ok else "biuf"
            if dtype.kind not in valid_kinds:
                raise TypeError
            return tensor, host, dtype

        def materialize(tensor, host, dtype):
            if tensor is not None:
                return tensor.to(dtype=dtype)
            return torch.as_tensor(host, dtype=dtype, device=device)

        try:
            x_tensor, x_host, _ = describe(
                parameters["x"], complex_ok=False
            )
            xp_tensor, xp_host, _ = describe(
                parameters["xp"], complex_ok=False
            )
            fp_tensor, fp_host, fp_dtype = describe(
                parameters["fp"], complex_ok=True
            )
            output_np = np.dtype(
                np.complex128 if fp_dtype.kind == "c" else np.float64
            )
            output_torch = _ensure_supported(
                device, _torch_dtype(output_np)
            )
            x = materialize(x_tensor, x_host, torch.float64)
            xp = materialize(xp_tensor, xp_host, torch.float64)
            fp = materialize(fp_tensor, fp_host, output_torch)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if xp.ndim != 1 or fp.ndim != 1:
            raise ValueError("Data points must be 1-D sequences")
        if xp.shape[0] != fp.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        if xp.numel() == 0:
            raise ValueError("array of sample points is empty")

        period = parameters["period"]
        if period is not None:
            try:
                period_array = np.asarray(period)
                if (
                        period_array.ndim != 0
                        or period_array.dtype.kind not in "biuf"):
                    return NotImplemented
                period_value = float(period_array)
            except (OverflowError, TypeError, ValueError):
                return NotImplemented
            if period_value == 0:
                raise ValueError("period must be a non-zero value")
            if not np.isfinite(period_value):
                return NotImplemented

            period_value = abs(period_value)
            x = torch.remainder(x, period_value)
            xp = torch.remainder(xp, period_value)
            if bool(torch.any(~torch.isfinite(xp)).item()):
                return NotImplemented
            order = torch.argsort(xp, stable=True)
            xp = xp[order]
            fp = fp[order]
            # NumPy's default quicksort does not define a portable order for
            # equal periodic coordinates. Preserve its host implementation
            # for that degenerate input rather than choosing a device order.
            if xp.numel() > 1 and bool(
                    torch.any(xp[1:] == xp[:-1]).item()):
                return NotImplemented
            xp = torch.cat((
                xp[-1:] - period_value,
                xp,
                xp[:1] + period_value,
            ))
            fp = torch.cat((fp[-1:], fp, fp[:1]))
            left = fp[0]
            right = fp[-1]
        else:
            if bool(torch.any(~torch.isfinite(xp)).item()):
                return NotImplemented
            if xp.numel() > 1 and bool(
                    torch.any(xp[1:] < xp[:-1]).item()):
                return NotImplemented

            def boundary(value, default):
                if value is None:
                    return default
                tensor, host, dtype = describe(
                    value, complex_ok=output_np.kind == "c"
                )
                candidate = materialize(tensor, host, output_torch)
                if candidate.ndim != 0:
                    raise TypeError
                if output_np.kind != "c" and dtype.kind == "c":
                    raise TypeError
                return candidate

            try:
                left = boundary(parameters["left"], fp[0])
                right = boundary(parameters["right"], fp[-1])
            except (OverflowError, TypeError, ValueError, RuntimeError):
                return NotImplemented

        # NumPy's compiled interpolation loop has special cancellation
        # handling for infinite samples that a direct Torch slope expression
        # does not reproduce. Keep those rare inputs on the host path.
        if bool(torch.any(~torch.isfinite(fp)).item()):
            return NotImplemented

        flat_x = x.reshape(-1)
        if xp.numel() == 1:
            result = fp[0].expand(flat_x.shape)
        else:
            try:
                upper = torch.searchsorted(
                    xp.contiguous(), flat_x.contiguous(), right=True
                ).clamp(1, xp.numel() - 1)
                lower = upper - 1
                x_lower = xp[lower]
                x_upper = xp[upper]
                denominator = x_upper - x_lower
                safe_denominator = torch.where(
                    denominator == 0,
                    torch.ones_like(denominator),
                    denominator,
                )
                weight = (flat_x - x_lower) / safe_denominator
                result = fp[lower] + weight * (fp[upper] - fp[lower])
                result = torch.where(
                    flat_x == x_lower, fp[lower], result
                )
            except (IndexError, TypeError, RuntimeError):
                return NotImplemented

        result = torch.where(flat_x < xp[0], left, result)
        result = torch.where(flat_x > xp[-1], right, result)
        result = torch.where(flat_x == xp[-1], fp[-1], result)
        if xp.numel() > 1:
            nan_value = torch.full(
                (), float("nan"), dtype=output_torch, device=device
            )
            result = torch.where(torch.isnan(flat_x), nan_value, result)
        result = result.reshape(x.shape)
        if result.ndim == 0:
            return output_np.type(result.item())
        return self._wrap(result)

    def _numpy_set_operand(self, value):
        """Prepare a real set-operation operand on this Torch device."""
        if isinstance(value, TorchArrayData):
            if (
                    not _device_matches_active(value.tensor)
                    or value.tensor.device != self.tensor.device
                    or np.dtype(value.dtype).kind not in "biuf"):
                raise TypeError
            return value, np.dtype(value.dtype)

        values = np.asanyarray(value)
        dtype = np.dtype(values.dtype)
        if dtype.kind not in "biuf":
            raise TypeError
        torch_dtype = _ensure_supported(
            self.tensor.device, _torch_dtype(dtype)
        )
        if values.ndim and not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        tensor = torch.as_tensor(
            values,
            dtype=torch_dtype,
            device=self.tensor.device,
        )
        return TorchArrayData(tensor), dtype

    def _numpy_sorted_unique_values(self, tensor, dtype):
        """Return NumPy-compatible sorted unique values on-device."""
        dtype = np.dtype(dtype)
        flattened = tensor.reshape(-1)
        execution = (
            flattened.to(dtype=torch.int64)
            if tensor.dtype == _TORCH_UINT32 else flattened
        )
        ordered = torch.sort(execution).values
        if ordered.numel():
            duplicates = ordered[1:] == ordered[:-1]
            if dtype.kind == "f":
                duplicates |= (
                    torch.isnan(ordered[1:])
                    & torch.isnan(ordered[:-1])
                )
            keep = torch.ones(
                ordered.shape,
                dtype=torch.bool,
                device=ordered.device,
            )
            keep[1:] = ~duplicates
            values = ordered[keep]
        else:
            values = ordered

        # NumPy's default sort chooses negative zero whenever the input
        # contains one, independent of its original position.
        if dtype.kind == "f" and values.numel():
            has_negative_zero = torch.any(
                (flattened == 0) & torch.signbit(flattened)
            )
            values = torch.where(
                (values == 0) & has_negative_zero,
                torch.copysign(values, -torch.ones_like(values)),
                values,
            )
        if tensor.dtype == _TORCH_UINT32:
            values = values.to(dtype=_TORCH_UINT32)
        return values

    def _numpy_union1d(self, args, kwargs):
        """Find sorted flattened union values without a host-array copy."""
        options = dict(kwargs)
        if len(args) > 2:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("ar1", "ar2")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            else:
                return NotImplemented
        if options:
            return NotImplemented

        try:
            left_data, left_dtype = self._numpy_set_operand(
                parameters["ar1"]
            )
            right_data, right_dtype = self._numpy_set_operand(
                parameters["ar2"]
            )
            target_dtype = np.dtype(np.result_type(
                left_dtype, right_dtype
            ))
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_dtype)
            )
            combined = torch.cat((
                left_data.tensor.reshape(-1).to(dtype=target_torch),
                right_data.tensor.reshape(-1).to(dtype=target_torch),
            ))
            values = self._numpy_sorted_unique_values(
                combined, target_dtype
            )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented
        return self._wrap(values)

    def _numpy_setdiff1d(self, args, kwargs):
        """Return flattened values unique to the first operand on-device."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("ar1", "ar2", "assume_unique")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "assume_unique":
                parameters[name] = False
            else:
                return NotImplemented
        if options or not np.isscalar(parameters["assume_unique"]):
            return NotImplemented
        try:
            assume_unique = bool(parameters["assume_unique"])
        except (TypeError, ValueError):
            return NotImplemented

        try:
            left_data, left_dtype = self._numpy_set_operand(
                parameters["ar1"]
            )
            right_data, right_dtype = self._numpy_set_operand(
                parameters["ar2"]
            )
            if assume_unique:
                left = left_data.tensor.reshape(-1)
                right = right_data.tensor.reshape(-1)
            else:
                left = self._numpy_sorted_unique_values(
                    left_data.tensor, left_dtype
                )
                right = self._numpy_sorted_unique_values(
                    right_data.tensor, right_dtype
                )
            left_values = self._wrap(left)
            mask = left_values._numpy_isin(
                (left_values, self._wrap(right)),
                {"assume_unique": True, "invert": True, "kind": "sort"},
            )
            if mask is NotImplemented:
                return NotImplemented
            indices = torch.nonzero(
                mask.tensor, as_tuple=False
            ).reshape(-1)
            execution = (
                left.to(dtype=torch.int64)
                if left.dtype == _TORCH_UINT32 else left
            )
            values = torch.index_select(execution, 0, indices)
            if left.dtype == _TORCH_UINT32:
                values = values.to(dtype=_TORCH_UINT32)
            return self._wrap(values)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def _numpy_setxor1d(self, args, kwargs):
        """Return the sorted exclusive union of two operands on-device."""
        options = dict(kwargs)
        if len(args) > 3:
            return NotImplemented

        parameters = {}
        for index, name in enumerate(("ar1", "ar2", "assume_unique")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name == "assume_unique":
                parameters[name] = False
            else:
                return NotImplemented
        if options or not np.isscalar(parameters["assume_unique"]):
            return NotImplemented
        try:
            assume_unique = bool(parameters["assume_unique"])
        except (TypeError, ValueError):
            return NotImplemented

        try:
            left_data, left_dtype = self._numpy_set_operand(
                parameters["ar1"]
            )
            right_data, right_dtype = self._numpy_set_operand(
                parameters["ar2"]
            )
            if assume_unique:
                left = left_data.tensor.reshape(-1)
                right = right_data.tensor.reshape(-1)
            else:
                left = self._numpy_sorted_unique_values(
                    left_data.tensor, left_dtype
                )
                right = self._numpy_sorted_unique_values(
                    right_data.tensor, right_dtype
                )

            target_dtype = np.dtype(np.result_type(
                left_dtype, right_dtype
            ))
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_dtype)
            )
            combined = torch.cat((
                left.to(dtype=target_torch),
                right.to(dtype=target_torch),
            ))
            execution = (
                combined.to(dtype=torch.int64)
                if target_torch == _TORCH_UINT32 else combined
            )
            ordered = torch.sort(execution).values
            if ordered.numel():
                different = ordered[1:] != ordered[:-1]
                boundaries = torch.cat((
                    torch.ones(
                        (1,), dtype=torch.bool, device=ordered.device
                    ),
                    different,
                    torch.ones(
                        (1,), dtype=torch.bool, device=ordered.device
                    ),
                ))
                values = ordered[boundaries[1:] & boundaries[:-1]]
            else:
                values = ordered
            if target_torch == _TORCH_UINT32:
                values = values.to(dtype=_TORCH_UINT32)
            return self._wrap(values)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def _numpy_intersect1d(self, args, kwargs):
        """Find common flattened values without copying Torch data to host."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {
            "assume_unique": False,
            "return_indices": False,
        }
        parameters = {}
        for index, name in enumerate((
                "ar1", "ar2", "assume_unique", "return_indices")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        flags = {}
        for name in ("assume_unique", "return_indices"):
            value = parameters[name]
            if not np.isscalar(value):
                return NotImplemented
            try:
                flags[name] = bool(value)
            except (TypeError, ValueError):
                return NotImplemented

        try:
            left_data, left_dtype = self._numpy_set_operand(
                parameters["ar1"]
            )
            right_data, right_dtype = self._numpy_set_operand(
                parameters["ar2"]
            )
            has_negative_zero = torch.zeros(
                (), dtype=torch.bool, device=self.tensor.device
            )
            for data, dtype in (
                    (left_data, left_dtype),
                    (right_data, right_dtype)):
                if dtype.kind == "f":
                    has_negative_zero |= torch.any(
                        (data.tensor == 0) & torch.signbit(data.tensor)
                    )

            if flags["assume_unique"]:
                left = left_data.tensor.reshape(-1)
                right = right_data.tensor.reshape(-1)
                left_first = right_first = None
            else:
                left_unique = left_data._numpy_unique(
                    (left_data,), {"return_index": True}
                )
                right_unique = right_data._numpy_unique(
                    (right_data,), {"return_index": True}
                )
                if (
                        left_unique is NotImplemented
                        or right_unique is NotImplemented):
                    return NotImplemented
                left = left_unique[0].tensor
                left_first = left_unique[1].tensor
                right = right_unique[0].tensor
                right_first = right_unique[1].tensor

            target_dtype = np.dtype(np.result_type(
                left_dtype, right_dtype
            ))
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_dtype)
            )
            left = left.to(dtype=target_torch)
            right = right.to(dtype=target_torch)
            left_size = left.numel()
            combined = torch.cat((left, right))
            execution = (
                combined.to(dtype=torch.int64)
                if target_torch == _TORCH_UINT32 else combined
            )

            if flags["return_indices"]:
                sorted_result = torch.sort(execution, stable=True)
                ordered = sorted_result.values
                order = sorted_result.indices
            else:
                order = None
                ordered = torch.sort(execution).values

            matches = ordered[1:] == ordered[:-1]
            values = ordered[:-1][matches]

            # NumPy's default quicksort places negative zero first. Stable
            # sorting is required for return_indices, where the value instead
            # comes from the first input. Preserve both contracts on-device.
            if (
                    not flags["return_indices"]
                    and target_dtype.kind == "f"
                    and values.numel()):
                values = torch.where(
                    (values == 0) & has_negative_zero,
                    torch.copysign(values, -torch.ones_like(values)),
                    values,
                )
            if target_torch == _TORCH_UINT32:
                values = values.to(dtype=_TORCH_UINT32)

            outputs = [self._wrap(values)]
            if flags["return_indices"]:
                left_indices = order[:-1][matches]
                right_indices = order[1:][matches] - left_size
                if not flags["assume_unique"]:
                    left_indices = left_first[left_indices]
                    right_indices = right_first[right_indices]
                outputs.extend((
                    self._wrap(left_indices),
                    self._wrap(right_indices),
                ))
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _numpy_isin(self, args, kwargs):
        """Test membership without copying Torch array data to the host."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {
            "assume_unique": False,
            "invert": False,
            "kind": None,
        }
        parameters = {}
        for index, name in enumerate((
                "element", "test_elements", "assume_unique", "invert")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        parameters["kind"] = options.pop("kind", None)
        if options:
            return NotImplemented

        element = parameters["element"]
        if (
                not isinstance(element, TorchArrayData)
                or not _device_matches_active(element.tensor)
                or np.dtype(element.dtype).kind not in "biufc"):
            return NotImplemented

        test_elements = parameters["test_elements"]
        test_values = None
        if isinstance(test_elements, TorchArrayData):
            if (
                    test_elements.tensor.device != element.tensor.device
                    or not _device_matches_active(test_elements.tensor)
                    or np.dtype(test_elements.dtype).kind not in "biufc"):
                return NotImplemented
            test_operand = test_elements.tensor
            test_dtype = np.dtype(test_elements.dtype)
        else:
            try:
                test_values = np.asarray(test_elements)
            except (TypeError, ValueError):
                return NotImplemented
            if test_values.dtype.kind not in "biufc":
                return NotImplemented
            if test_values.ndim and not test_values.flags.c_contiguous:
                test_values = np.ascontiguousarray(test_values)
            test_operand = test_values
            test_dtype = test_values.dtype

        flags = {}
        for name in ("assume_unique", "invert"):
            value = parameters[name]
            if not np.isscalar(value):
                return NotImplemented
            try:
                flags[name] = bool(value)
            except (TypeError, ValueError):
                return NotImplemented

        kind = parameters["kind"]
        try:
            valid_kind = kind in {None, "sort", "table"}
        except (TypeError, ValueError):
            return NotImplemented
        if not valid_kind:
            raise ValueError(
                f"Invalid kind: '{kind}'. Please use None, 'sort' or "
                "'table'."
            )
        if kind == "table" and not all(
                dtype.kind in "biu"
                for dtype in (np.dtype(element.dtype), test_dtype)):
            raise ValueError(
                "The 'table' method is only supported for boolean or "
                "integer arrays. Please select 'sort' or None for kind."
            )

        # NumPy rejects an explicit table lookup when subtracting the test
        # extrema would overflow its integer dtype.  A scalar synchronization
        # is sufficient to preserve that error without staging either array.
        if kind == "table" and test_dtype.kind in "iu":
            if isinstance(test_operand, torch.Tensor):
                flattened_test = test_operand.reshape(-1)
                if flattened_test.numel():
                    test_min = int(torch.min(flattened_test).item())
                    test_max = int(torch.max(flattened_test).item())
                else:
                    test_min = test_max = 0
            elif test_values.size:
                test_min = int(np.min(test_values))
                test_max = int(np.max(test_values))
            else:
                test_min = test_max = 0
            if test_max - test_min > np.iinfo(test_dtype).max:
                raise RuntimeError(
                    "You have specified kind='table', but the range of "
                    "values in `ar2` or `ar1` exceed the maximum integer "
                    "of the datatype. Please set `kind` to None or 'sort'."
                )
            if test_max - test_min >= np.iinfo(np.intp).max:
                raise ValueError("Maximum allowed dimension exceeded")

        try:
            left, right, outside_range = _comparison_tensors(
                element.tensor, element.dtype, test_operand
            )
            if outside_range is not None:
                return NotImplemented
            flattened_test = right.reshape(-1)

            if flattened_test.numel() == 0:
                result = torch.zeros_like(left, dtype=torch.bool)
            elif left.dtype == torch.bool:
                contains_true = torch.any(flattened_test)
                contains_false = torch.any(torch.logical_not(flattened_test))
                result = (
                    (left & contains_true)
                    | (torch.logical_not(left) & contains_false)
                )
            elif not left.is_complex():
                try:
                    result = torch.isin(left, flattened_test)
                except (NotImplementedError, RuntimeError):
                    result = None
            else:
                result = None

            if result is None:
                flattened = left.reshape(-1)
                matched = torch.zeros_like(flattened, dtype=torch.bool)
                # Bound each temporary comparison matrix to roughly one
                # million values while retaining exact complex semantics.
                comparison_budget = 1 << 20
                element_count = flattened.numel()
                if element_count == 0:
                    element_count = 1
                chunk_size = comparison_budget // element_count
                if chunk_size < 1:
                    chunk_size = 1
                if chunk_size > flattened_test.numel():
                    chunk_size = flattened_test.numel()
                for start in range(0, flattened_test.numel(), chunk_size):
                    chunk = flattened_test[start:start + chunk_size]
                    matched |= torch.any(
                        flattened[:, None] == chunk[None, :], dim=1
                    )
                result = matched.reshape(left.shape)

            if flags["invert"]:
                result = torch.logical_not(result)
            return self._wrap(result)
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

    def _numpy_sort(self, args, kwargs):
        """Sort real values without copying the array to the host."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {
            "axis": -1,
            "kind": None,
            "order": None,
            "stable": None,
        }
        parameters = {}
        for index, name in enumerate(("a", "axis", "kind", "order")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        parameters["stable"] = options.pop("stable", None)
        if options:
            return NotImplemented

        array = parameters["a"]
        kind = parameters["kind"]
        stable = parameters["stable"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biuf"
                or parameters["order"] is not None
                or kind not in (None, "stable", "mergesort")):
            return NotImplemented
        if kind is not None and stable is not None:
            raise ValueError(
                "`kind` and `stable` parameters can't be provided at the "
                "same time. Use only one of them."
            )

        tensor = array.tensor
        axis = parameters["axis"]
        if axis is None:
            tensor = tensor.reshape(-1)
            dimension = 0
        else:
            try:
                dimension = operator.index(axis)
            except TypeError:
                return NotImplemented
            if dimension < -tensor.ndim or dimension >= tensor.ndim:
                return NotImplemented

        stable_sort = (
            bool(stable)
            if stable is not None
            else kind in ("stable", "mergesort")
        )
        try:
            result = torch.sort(
                tensor, dim=dimension, stable=stable_sort
            ).values
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        # MPS orders negative zero before positive zero even for a stable
        # sort. Restore the input order of equal zero values so the result
        # follows NumPy's stable-sort contract without leaving the device.
        if (
                stable_sort
                and tensor.device.type == "mps"
                and tensor.dtype.is_floating_point):
            size = tensor.shape[dimension]
            if size:
                shape = [1] * tensor.ndim
                shape[dimension] = size
                positions = torch.arange(
                    size, device=tensor.device, dtype=torch.int64
                ).reshape(shape)
                positions = positions.expand(tensor.shape)
                source_positions = torch.where(
                    tensor == 0, positions, size
                )
                source_positions = torch.sort(
                    source_positions, dim=dimension
                ).values
                source_positions = source_positions.clamp_max(size - 1)
                source_zeros = torch.gather(
                    tensor, dimension, source_positions
                )
                result_zero = result == 0
                zero_rank = torch.cumsum(
                    result_zero.to(torch.int64), dim=dimension
                ) - 1
                stable_zeros = torch.gather(
                    source_zeros, dimension, zero_rank.clamp_min(0)
                )
                result = torch.where(result_zero, stable_zeros, result)
        return self._wrap(result)

    def _numpy_dot_product(self, function, args, kwargs):
        """Evaluate NumPy dot-product functions without a host array copy."""
        options = dict(kwargs)
        if function in (np.dot, np.outer):
            if len(args) > 3:
                return NotImplemented
            parameters = {"a": None, "b": None, "out": None}
            for name, value in zip(parameters, args):
                if name in options:
                    return NotImplemented
                parameters[name] = value
            for name in tuple(parameters):
                if name in options:
                    parameters[name] = options.pop(name)
            if options or parameters["a"] is None or parameters["b"] is None:
                return NotImplemented
            if parameters["out"] is not None:
                return NotImplemented
            left, right = parameters["a"], parameters["b"]
        else:
            if len(args) != 2 or options:
                return NotImplemented
            left, right = args

        def operand_tensor(value):
            if isinstance(value, TorchArrayData):
                if (
                        not _device_matches_active(value.tensor)
                        or value.tensor.device != self.tensor.device):
                    raise TypeError
                return value.tensor, np.dtype(value.dtype)
            if isinstance(value, torch.Tensor):
                if value.device != self.tensor.device:
                    raise TypeError
                return value, np.dtype(_numpy_dtype(value.dtype))
            values = np.asarray(value)
            dtype = np.dtype(values.dtype)
            torch_dtype = _ensure_supported(
                self.tensor.device, _torch_dtype(dtype)
            )
            return (
                torch.as_tensor(
                    values,
                    dtype=torch_dtype,
                    device=self.tensor.device,
                ),
                dtype,
            )

        try:
            left_tensor, left_dtype = operand_tensor(left)
            right_tensor, right_dtype = operand_tensor(right)
            target_np = np.result_type(left_dtype, right_dtype)
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_np)
            )
            left_tensor = left_tensor.to(dtype=target_torch)
            right_tensor = right_tensor.to(dtype=target_torch)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        def contraction(first, second, dimensions):
            if target_np.kind == "b":
                result = torch.tensordot(
                    first.to(torch.int64),
                    second.to(torch.int64),
                    dims=dimensions,
                )
                return result.to(dtype=torch.bool)
            if target_np.kind == "u":
                result = torch.tensordot(
                    first.to(torch.int64),
                    second.to(torch.int64),
                    dims=dimensions,
                )
                return result.to(dtype=target_torch)
            return torch.tensordot(first, second, dims=dimensions)

        try:
            if function is np.outer:
                result = (
                    left_tensor.reshape(-1, 1)
                    * right_tensor.reshape(1, -1)
                )
            elif function is np.vdot:
                left_flat = left_tensor.reshape(-1)
                right_flat = right_tensor.reshape(-1)
                if left_flat.numel() != right_flat.numel():
                    raise ValueError(
                        "cannot reshape array of size "
                        f"{right_flat.numel()} into shape "
                        f"({left_flat.numel()},)"
                    )
                if target_np.kind in "bu":
                    result = contraction(
                        left_flat, right_flat, ([0], [0])
                    )
                else:
                    result = torch.sum(torch.conj(left_flat) * right_flat)
            elif left_tensor.ndim == 0 or right_tensor.ndim == 0:
                result = left_tensor * right_tensor
            elif function is np.inner:
                result = contraction(
                    left_tensor,
                    right_tensor,
                    ([-1], [-1]),
                )
            else:
                right_axis = -2 if right_tensor.ndim > 1 else 0
                result = contraction(
                    left_tensor,
                    right_tensor,
                    ([-1], [right_axis]),
                )
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        if result.ndim == 0:
            return np.dtype(target_np).type(result.item())
        return self._wrap(result)

    def _numpy_linalg_norm(self, args, kwargs):
        """Evaluate common vector and Frobenius norms on the device."""
        options = dict(kwargs)
        if len(args) > 4:
            return NotImplemented

        defaults = {"ord": None, "axis": None, "keepdims": False}
        parameters = {}
        for index, name in enumerate(("x", "ord", "axis", "keepdims")):
            if index < len(args):
                if name in options:
                    return NotImplemented
                parameters[name] = args[index]
            elif name in options:
                parameters[name] = options.pop(name)
            elif name in defaults:
                parameters[name] = defaults[name]
            else:
                return NotImplemented
        if options:
            return NotImplemented

        array = parameters["x"]
        if (
                not isinstance(array, TorchArrayData)
                or not _device_matches_active(array.tensor)
                or array.tensor.device != self.tensor.device
                or np.dtype(array.dtype).kind not in "biufc"):
            return NotImplemented

        try:
            keepdims = bool(parameters["keepdims"])
        except (TypeError, ValueError):
            return NotImplemented

        tensor = array.tensor
        axis = parameters["axis"]
        order = parameters["ord"]
        mode = None
        dimensions = None

        def normalize_axis(value):
            original = operator.index(value)
            normalized = original + tensor.ndim if original < 0 else original
            if normalized < 0 or normalized >= tensor.ndim:
                raise IndexError
            return normalized

        if axis is None:
            if order is None:
                mode = "vector"
            elif tensor.ndim == 1:
                mode = "vector"
                dimensions = 0
            elif tensor.ndim == 2:
                mode = "matrix"
                dimensions = (0, 1)
            else:
                return NotImplemented
        else:
            try:
                dimensions = normalize_axis(axis)
                mode = "vector"
            except (IndexError, TypeError):
                if not isinstance(axis, tuple) or len(axis) != 2:
                    return NotImplemented
                try:
                    dimensions = tuple(normalize_axis(value) for value in axis)
                except (IndexError, TypeError):
                    return NotImplemented
                if dimensions[0] == dimensions[1]:
                    return NotImplemented
                mode = "matrix"

        if mode == "matrix":
            if order is not None and not (
                    isinstance(order, (str, np.str_))
                    and order in ("fro", "f")):
                return NotImplemented
            vector_order = 2.0
        elif order is None:
            vector_order = 2.0
        else:
            if not np.isscalar(order):
                return NotImplemented
            try:
                order_dtype = np.asarray(order).dtype
                vector_order = float(order)
            except (OverflowError, TypeError, ValueError):
                return NotImplemented
            if (
                    order_dtype.kind not in "biuf"
                    or np.isnan(vector_order)
                    or (vector_order < 0 and vector_order != -np.inf)):
                return NotImplemented

        input_dtype = np.dtype(array.dtype)
        if input_dtype == np.dtype(np.complex64):
            output_dtype = np.dtype(np.float32)
        elif input_dtype == np.dtype(np.complex128):
            output_dtype = np.dtype(np.float64)
        elif input_dtype.kind == "f":
            output_dtype = input_dtype
        else:
            output_dtype = np.dtype(np.float64)

        try:
            output_torch = _ensure_supported(
                tensor.device, _torch_dtype(output_dtype)
            )
            execution = tensor if input_dtype.kind == "c" else tensor.to(
                dtype=output_torch
            )

            # NumPy defines the positive-infinity norm of an empty vector as
            # zero, while Torch's reduction has no empty-set identity.
            if vector_order == np.inf and (
                    execution.numel() == 0
                    and (dimensions is None
                         or execution.shape[dimensions] == 0)):
                if dimensions is None:
                    shape = ((1,) * execution.ndim) if keepdims else ()
                else:
                    shape = list(execution.shape)
                    if keepdims:
                        shape[dimensions] = 1
                    else:
                        shape.pop(dimensions)
                result = torch.zeros(
                    shape, dtype=output_torch, device=tensor.device
                )
            else:
                result = torch.linalg.vector_norm(
                    execution,
                    ord=vector_order,
                    dim=dimensions,
                    keepdim=keepdims,
                )
        except (IndexError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if result.ndim == 0:
            return output_dtype.type(result.item())
        return self._wrap(result)

    def numpy_array_function(self, function, *args, **kwargs):
        """Evaluate supported NumPy array functions within Torch."""
        if function in (np.argwhere, np.flatnonzero):
            return self._numpy_index_locations(function, args, kwargs)
        if function in (np.dot, np.vdot, np.inner, np.outer):
            return self._numpy_dot_product(function, args, kwargs)
        if function is np.linalg.norm:
            return self._numpy_linalg_norm(args, kwargs)
        if function is np.where:
            return self._numpy_where(args, kwargs)
        if function is np.diff:
            return self._numpy_diff(args, kwargs)
        if function in (np.diag, np.diagflat):
            return self._numpy_diag(function, args, kwargs)
        if function is np.diagonal:
            return self._numpy_diagonal(args, kwargs)
        if function is np.trace:
            return self._numpy_trace(args, kwargs)
        if function in (np.tril, np.triu):
            return self._numpy_triangle(function, args, kwargs)
        if function in (np.flip, np.flipud, np.fliplr):
            return self._numpy_flip(function, args, kwargs)
        if function is np.rot90:
            return self._numpy_rot90(args, kwargs)
        if function is np.expand_dims:
            return self._numpy_expand_dims(args, kwargs)
        if function is np.moveaxis:
            return self._numpy_moveaxis(args, kwargs)
        if function is np.rollaxis:
            return self._numpy_rollaxis(args, kwargs)
        if function in (np.atleast_1d, np.atleast_2d, np.atleast_3d):
            return self._numpy_atleast_nd(function, args, kwargs)
        if function in (np.broadcast_arrays, np.broadcast_to):
            return self._numpy_broadcast(function, args, kwargs)
        if function in (np.isclose, np.allclose):
            return self._numpy_close(function, args, kwargs)
        if function in (np.array_equal, np.array_equiv):
            return self._numpy_array_equality(function, args, kwargs)
        if function is np.count_nonzero:
            return self._numpy_count_nonzero(args, kwargs)
        if function is np.average:
            return self._numpy_average(args, kwargs)
        if function is np.median:
            return self._numpy_median(args, kwargs)
        if function is np.ptp:
            return self._numpy_ptp(args, kwargs)
        if function is np.take_along_axis:
            return self._numpy_take_along_axis(args, kwargs)
        if function is np.copy:
            return self._numpy_copy(args, kwargs)
        if function in (
                np.empty_like, np.zeros_like, np.ones_like, np.full_like):
            return self._numpy_like_creator(function, args, kwargs)
        if function is np.tile:
            return self._numpy_tile(args, kwargs)
        if function is np.pad:
            return self._numpy_pad(args, kwargs)
        if function in (np.compress, np.extract):
            return self._numpy_boolean_selection(function, args, kwargs)
        if function is np.append:
            return self._numpy_append(args, kwargs)
        if function is np.delete:
            return self._numpy_delete(args, kwargs)
        if function is np.putmask:
            return self._numpy_putmask(args, kwargs)
        if function is np.resize:
            return self._numpy_resize(args, kwargs)
        if function in (np.real, np.imag, np.angle):
            return self._numpy_complex_component(function, args, kwargs)
        if function is np.unwrap:
            return self._numpy_unwrap(args, kwargs)
        if function is np.ravel:
            return self._numpy_ravel(args, kwargs)
        if function is np.roll:
            return self._numpy_roll(args, kwargs)
        if function is np.sort:
            return self._numpy_sort(args, kwargs)
        if function is np.unique:
            return self._numpy_unique(args, kwargs)
        if function is np.histogram:
            return self._numpy_histogram(args, kwargs)
        if function is np.digitize:
            return self._numpy_digitize(args, kwargs)
        if function is _NUMPY_TRAPEZOID:
            return self._numpy_trapezoid(args, kwargs)
        if function is np.interp:
            return self._numpy_interp(args, kwargs)
        if function is np.intersect1d:
            return self._numpy_intersect1d(args, kwargs)
        if function is np.setdiff1d:
            return self._numpy_setdiff1d(args, kwargs)
        if function is np.setxor1d:
            return self._numpy_setxor1d(args, kwargs)
        if function is np.union1d:
            return self._numpy_union1d(args, kwargs)
        if function is np.isin:
            return self._numpy_isin(args, kwargs)
        join_functions = (
            np.concatenate,
            np.stack,
            np.hstack,
            np.vstack,
            np.dstack,
            np.column_stack,
        )
        if function not in join_functions or not args:
            return NotImplemented

        options = dict(kwargs)
        arrays = args[0]
        out = None
        if function in (np.concatenate, np.stack):
            if len(args) > 3:
                return NotImplemented
            if len(args) > 1:
                if "axis" in options:
                    return NotImplemented
                axis = args[1]
            else:
                axis = options.pop("axis", 0)
            if len(args) > 2:
                if "out" in options:
                    return NotImplemented
                out = args[2]
            else:
                out = options.pop("out", None)
            dtype = options.pop("dtype", None)
            casting = options.pop("casting", "same_kind")
        elif function in (np.hstack, np.vstack):
            if len(args) != 1:
                return NotImplemented
            axis = None
            dtype = options.pop("dtype", None)
            casting = options.pop("casting", "same_kind")
        else:
            if len(args) != 1:
                return NotImplemented
            axis = None
            dtype = None
            casting = "same_kind"
        if options or not isinstance(arrays, (tuple, list)) or not arrays:
            return NotImplemented
        if dtype is not None and out is not None:
            raise TypeError(
                "concatenate() only takes `out` or `dtype` as an argument, "
                "but both were provided."
            )
        if not all(isinstance(array, TorchArrayData) for array in arrays):
            return NotImplemented
        if not all(
                _device_matches_active(array.tensor)
                and array.tensor.device == self.tensor.device
                for array in arrays):
            return NotImplemented
        if out is not None and (
                not isinstance(out, TorchArrayData)
                or not _device_matches_active(out.tensor)
                or out.tensor.device != self.tensor.device):
            return NotImplemented

        if axis is not None:
            try:
                axis = operator.index(axis)
            except TypeError:
                return NotImplemented
        elif function is np.stack:
            return NotImplemented

        try:
            input_dtypes = [np.dtype(array.dtype) for array in arrays]
            if out is not None:
                target_np = np.dtype(out.dtype)
            elif dtype is not None:
                target_np = np.dtype(dtype)
            else:
                target_np = np.result_type(*input_dtypes)
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_np)
            )
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented

        try:
            for input_dtype in input_dtypes:
                if not np.can_cast(input_dtype, target_np, casting=casting):
                    raise TypeError(
                        f"Cannot cast array data from {input_dtype!r} to "
                        f"{target_np!r} according to the rule {casting!r}"
                    )

            tensors = [
                array.tensor.to(dtype=target_torch) for array in arrays
            ]
            if function is np.concatenate:
                if axis is None:
                    tensors = [tensor.reshape(-1) for tensor in tensors]
                    axis = 0
                result = torch.cat(tensors, dim=axis)
            elif function is np.stack:
                result = torch.stack(tensors, dim=axis)
            else:
                if function is np.hstack:
                    tensors = [
                        tensor.reshape(1) if tensor.ndim == 0 else tensor
                        for tensor in tensors
                    ]
                    axis = 0 if tensors[0].ndim == 1 else 1
                elif function is np.vstack:
                    tensors = [
                        tensor.reshape(1, 1) if tensor.ndim == 0
                        else tensor.reshape(1, -1) if tensor.ndim == 1
                        else tensor
                        for tensor in tensors
                    ]
                    axis = 0
                elif function is np.dstack:
                    tensors = [
                        tensor.reshape(1, 1, 1) if tensor.ndim == 0
                        else tensor.reshape(1, -1, 1)
                        if tensor.ndim == 1
                        else tensor.unsqueeze(-1) if tensor.ndim == 2
                        else tensor
                        for tensor in tensors
                    ]
                    axis = 2
                else:
                    tensors = [
                        tensor.reshape(1, 1) if tensor.ndim == 0
                        else tensor.reshape(-1, 1) if tensor.ndim == 1
                        else tensor
                        for tensor in tensors
                    ]
                    axis = 1
                result = torch.cat(tensors, dim=axis)
        except (IndexError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc

        if out is not None:
            if out.shape != tuple(result.shape):
                raise ValueError(
                    "Output array is the wrong shape: expected "
                    f"{tuple(result.shape)}, got {out.shape}"
                )
            out.tensor.copy_(result)
            return out
        return self._wrap(result)

    def numpy_ufunc(self, ufunc, method, *inputs, **kwargs):
        """Evaluate supported NumPy ufunc calls within Torch."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        if method == "accumulate":
            if len(inputs) != 1 or inputs[0] is not self:
                return NotImplemented
            options = dict(kwargs)
            axis = options.pop("axis", 0)
            out = options.pop("out", None)
            if out is not None and not (
                isinstance(out, tuple)
                and len(out) == 1
                and out[0] is None
            ):
                return NotImplemented
            dtype = options.pop("dtype", None)
            if options:
                return NotImplemented
            operation = _TORCH_ACCUMULATE_NUMPY_UFUNCS.get(ufunc)
            if operation is None or np.dtype(self.dtype).kind not in "fc":
                return NotImplemented
            try:
                target_dtype = np.dtype(
                    self.dtype if dtype is None else dtype
                )
                if target_dtype.kind not in "fc":
                    return NotImplemented
                target_torch = _ensure_supported(
                    self.tensor.device, _torch_dtype(target_dtype)
                )
                tensor = self.tensor.to(dtype=target_torch)
                if axis is None:
                    tensor = tensor.reshape(-1)
                    dimension = 0
                else:
                    dimension = operator.index(axis)
                    if dimension < 0:
                        dimension += tensor.ndim
                    if dimension < 0 or dimension >= tensor.ndim:
                        return NotImplemented
            except (TypeError, ValueError, RuntimeError):
                return NotImplemented
            if (
                ufunc in _TORCH_REAL_ACCUMULATE_NUMPY_UFUNCS
                and tensor.is_complex()
            ):
                return NotImplemented
            return self._wrap(operation(tensor, dim=dimension))
        if method == "reduce":
            if len(inputs) != 1 or inputs[0] is not self:
                return NotImplemented
            options = dict(kwargs)
            axis = options.pop("axis", 0)
            out = options.pop("out", None)
            if out is not None and not (
                isinstance(out, tuple)
                and len(out) == 1
                and out[0] is None
            ):
                return NotImplemented
            dtype = options.pop("dtype", None)
            initial = options.pop("initial", None)
            keepdims = options.pop("keepdims", False)
            if options:
                return NotImplemented
            operation = _TORCH_REDUCE_NUMPY_UFUNCS.get(ufunc)
            if operation is None:
                return NotImplemented
            try:
                keepdims = bool(operator.index(keepdims))
                axes = _normalized_reduction_axes(axis, self.tensor.ndim)
            except (IndexError, TypeError, ValueError):
                return NotImplemented

            source_dtype = np.dtype(self.dtype)
            logical_reduction = (
                ufunc in _TORCH_LOGICAL_REDUCE_NUMPY_UFUNCS
            )
            if logical_reduction:
                if dtype is not None and np.dtype(dtype) != np.dtype(bool):
                    return NotImplemented
                target_dtype = np.dtype(bool)
                target_torch = torch.bool
            else:
                if source_dtype.kind not in "biufc":
                    return NotImplemented
                if dtype is None:
                    if (
                        ufunc in (np.add, np.multiply)
                        and source_dtype.kind in "bi"
                    ):
                        target_dtype = np.dtype(np.int64)
                    elif (
                        ufunc in (np.add, np.multiply)
                        and source_dtype.kind == "u"
                    ):
                        return NotImplemented
                    else:
                        target_dtype = source_dtype
                else:
                    target_dtype = np.dtype(dtype)
                if target_dtype.kind not in "biufc":
                    return NotImplemented
                if (
                    ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                    and (
                        source_dtype.kind == "c"
                        or target_dtype.kind == "c"
                    )
                ):
                    return NotImplemented
                try:
                    target_torch = _ensure_supported(
                        self.tensor.device, _torch_dtype(target_dtype)
                    )
                except (TypeError, RuntimeError):
                    return NotImplemented
            tensor = self.tensor.to(dtype=target_torch)

            if (
                ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                and tensor.is_complex()
            ):
                return NotImplemented
            output_shape = _reduction_output_shape(
                tuple(tensor.shape), axes, keepdims
            )
            output_size = int(np.prod(output_shape, dtype=np.int64))
            reduction_size = 1
            for dimension in axes:
                reduction_size *= tensor.shape[dimension]
            if (
                ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                and reduction_size == 0
                and initial is None
            ):
                raise ValueError(
                    "zero-size array to reduction operation "
                    f"{ufunc.__name__} which has no identity"
                )
            try:
                if (
                    logical_reduction
                    and reduction_size == 0
                ):
                    reduced = torch.full(
                        output_shape,
                        ufunc is np.logical_and,
                        dtype=torch.bool,
                        device=tensor.device,
                    )
                elif (
                    ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                    and reduction_size == 0
                    and initial is not None
                    and output_size
                ):
                    reduced = torch.full(
                        output_shape,
                        initial,
                        dtype=target_torch,
                        device=tensor.device,
                    )
                elif not axes:
                    reduced = tensor.clone()
                elif (
                    ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                    and output_size == 0
                ):
                    reduced = torch.empty(
                        output_shape,
                        dtype=target_torch,
                        device=tensor.device,
                    )
                elif ufunc is np.add:
                    reduced = torch.sum(
                        tensor, dim=axes, keepdim=keepdims
                    )
                elif ufunc is np.multiply:
                    reduced = tensor
                    dimensions = sorted(axes, reverse=not keepdims)
                    for dimension in dimensions:
                        reduced = torch.prod(
                            reduced, dim=dimension, keepdim=keepdims
                        )
                elif ufunc is np.maximum:
                    reducer = torch.any if tensor.dtype == torch.bool \
                        else torch.amax
                    reduced = reducer(
                        tensor, dim=axes, keepdim=keepdims
                    )
                elif ufunc is np.minimum:
                    reducer = torch.all if tensor.dtype == torch.bool \
                        else torch.amin
                    reduced = reducer(
                        tensor, dim=axes, keepdim=keepdims
                    )
                elif ufunc is np.logical_or:
                    reduced = torch.any(
                        tensor, dim=axes, keepdim=keepdims
                    )
                else:
                    reduced = torch.all(
                        tensor, dim=axes, keepdim=keepdims
                    )

                if initial is not None:
                    initial_tensor = torch.as_tensor(
                        initial,
                        device=tensor.device,
                        dtype=target_torch,
                    )
                    if initial_tensor.ndim:
                        return NotImplemented
                    if (
                        ufunc in _TORCH_REAL_REDUCE_NUMPY_UFUNCS
                        and reduction_size == 0
                        and output_size
                    ):
                        reduced = torch.full(
                            output_shape,
                            initial_tensor.item(),
                            dtype=target_torch,
                            device=tensor.device,
                        )
                    elif ufunc is np.add:
                        reduced = reduced + initial_tensor
                    elif ufunc is np.multiply:
                        reduced = reduced * initial_tensor
                    elif ufunc is np.maximum:
                        reduced = torch.maximum(reduced, initial_tensor)
                    elif ufunc is np.minimum:
                        reduced = torch.minimum(reduced, initial_tensor)
                    elif ufunc is np.logical_or:
                        reduced = torch.logical_or(reduced, initial_tensor)
                    else:
                        reduced = torch.logical_and(reduced, initial_tensor)
            except (IndexError, TypeError, ValueError, RuntimeError):
                return NotImplemented

            if reduced.ndim == 0:
                return target_dtype.type(reduced.item())
            return self._wrap(reduced)
        if method != "__call__" or kwargs:
            return NotImplemented
        if len(inputs) == 1:
            operand = inputs[0]
            if not isinstance(operand, TorchArrayData):
                return NotImplemented
            operation = _TORCH_UNARY_NUMPY_UFUNCS.get(ufunc)
            if operation is None:
                return NotImplemented
            supported_kinds = (
                "biufc"
                if ufunc in _TORCH_BOOLEAN_UNARY_NUMPY_UFUNCS
                else (
                    "biu"
                    if ufunc in _TORCH_BITWISE_UNARY_NUMPY_UFUNCS
                    else "fc"
                )
            )
            if np.dtype(operand.dtype).kind not in supported_kinds:
                return NotImplemented
            if (
                ufunc in _TORCH_REAL_UNARY_NUMPY_UFUNCS
                and operand.tensor.is_complex()
            ):
                return NotImplemented
            tensor = operand.tensor
            if (
                ufunc in _TORCH_BITWISE_UNARY_NUMPY_UFUNCS
                and np.dtype(operand.dtype).kind == "u"
            ):
                return operand._wrap(
                    operation(tensor.to(torch.int64)).to(tensor.dtype)
                )
            return operand._wrap(operation(tensor))
        if len(inputs) == 2:
            comparison = _TORCH_COMPARISON_NUMPY_UFUNCS.get(ufunc)
            if comparison is not None:
                result = _comparison_ufunc_result(
                    inputs, self, comparison
                )
                return NotImplemented if result is None else result
            operation = _TORCH_BINARY_NUMPY_UFUNCS.get(ufunc)
            if operation is None:
                return NotImplemented
            if ufunc in _TORCH_LOGICAL_BINARY_NUMPY_UFUNCS:
                tensors = _logical_ufunc_tensors(inputs, self)
            elif ufunc in _TORCH_BITWISE_BINARY_NUMPY_UFUNCS:
                prepared = _bitwise_ufunc_tensors(inputs, self)
                if prepared is None:
                    return NotImplemented
                tensors, output_dtype = prepared
                return self._wrap(
                    operation(*tensors).to(dtype=output_dtype)
                )
            else:
                tensors = _binary_ufunc_tensors(inputs, self)
            if tensors is None:
                return NotImplemented
            if (
                ufunc in _TORCH_REAL_BINARY_NUMPY_UFUNCS
                and any(tensor.is_complex() for tensor in tensors)
            ):
                return NotImplemented
            return self._wrap(operation(*tensors))
        return NotImplemented

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        res = self.tensor.__getitem__(idx)
        if isinstance(res, torch.Tensor):
            if res.ndim == 0:
                return res.item()
            return self._wrap(res)
        return res

    def __setitem__(self, idx, value):
        if isinstance(value, TorchArrayData):
            value = value.tensor
        if isinstance(value, torch.Tensor):
            value = value.to(device=self.tensor.device)
        elif isinstance(value, (np.ndarray, list, tuple)):
            value = torch.as_tensor(value, device=self.tensor.device)
        self.tensor.__setitem__(idx, value)

    def fill(self, value):
        self.tensor.fill_(value)

    def __neg__(self):
        return self._wrap(-self.tensor)

    def __abs__(self):
        return self._wrap(torch.abs(self.tensor))

    def __add__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a + b)

    def __radd__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a + b)

    def __iadd__(self, other):
        other_t, _ = _tensor_from_any(other, self.tensor.device)
        self.tensor += other_t
        return self

    def __sub__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a - b)

    def __rsub__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(b - a)

    def __isub__(self, other):
        other_t, _ = _tensor_from_any(other, self.tensor.device)
        self.tensor -= other_t
        return self

    def __mul__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a * b)

    def __rmul__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a * b)

    def __imul__(self, other):
        other_t, _ = _tensor_from_any(other, self.tensor.device)
        self.tensor *= other_t
        return self

    def __truediv__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a / b)

    def __rtruediv__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(b / a)

    def __itruediv__(self, other):
        other_t, _ = _tensor_from_any(other, self.tensor.device)
        self.tensor /= other_t
        return self

    def __pow__(self, other):
        a, b, _ = self._promote_with(other)
        return self._wrap(a ** b)

    def conj(self):
        return self._wrap(torch.conj(self.tensor))

    @property
    def real(self):
        if not self.tensor.is_complex():
            return self
        return self._wrap(self.tensor.real)

    @property
    def imag(self):
        if not self.tensor.is_complex():
            zeros = torch.zeros_like(self.tensor, dtype=self.tensor.dtype)
            return self._wrap(zeros)
        return self._wrap(self.tensor.imag)

    def astype(self, dtype):
        tdtype = _torch_dtype(dtype)
        return self._wrap(self.tensor.to(dtype=tdtype))

    def view(self, dtype):
        target_np = np.dtype(dtype)
        target_torch = _torch_dtype(target_np)
        if target_torch == self.tensor.dtype:
            return self

        # ``Tensor.view(dtype)`` reinterprets the same storage on every Torch
        # device. A NumPy round-trip here would silently turn a view into a
        # copy and move CUDA/MPS data through host memory.
        try:
            return self._wrap(self.tensor.view(target_torch))
        except RuntimeError as exc:
            raise TypeError(
                f"Cannot view {self.dtype} data as {target_np}"
            ) from exc

    def copy(self):
        return self._wrap(self.tensor.clone())

    @staticmethod
    def _normalize_numpy_order(order, allowed):
        """Normalize NumPy's string-like memory-order argument."""
        if order is None:
            order = "C"
        if isinstance(order, bytes):
            try:
                order = order.decode("ascii")
            except UnicodeDecodeError:
                return None
        if not isinstance(order, str):
            return None
        order = order.upper()
        return order if order in allowed else None

    def numpy_reshape(self, shape, order="C", copy=None):
        """Implement NumPy's C-, F-, and A-order reshape on Torch."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        order = self._normalize_numpy_order(order, ("C", "F", "A"))
        if order is None:
            return NotImplemented
        if order == "A":
            order = (
                "F"
                if self._is_fortran_contiguous(self.tensor)
                and not self.tensor.is_contiguous()
                else "C"
            )
        if copy is not None:
            if isinstance(copy, (str, bytes)):
                return NotImplemented
            try:
                copy = bool(copy)
            except (TypeError, ValueError):
                return NotImplemented

        try:
            try:
                normalized_shape = (operator.index(shape),)
            except TypeError:
                normalized_shape = tuple(
                    operator.index(length) for length in shape
                )

            tensor = self.tensor
            if order == "F":
                source_axes = tuple(reversed(range(tensor.ndim)))
                if source_axes != tuple(range(tensor.ndim)):
                    tensor = tensor.permute(source_axes)
                normalized_shape = tuple(reversed(normalized_shape))

            if copy is False:
                result = tensor.view(normalized_shape)
            else:
                result = torch.reshape(tensor, normalized_shape)
            if order == "F":
                result_axes = tuple(reversed(range(result.ndim)))
                if result_axes != tuple(range(result.ndim)):
                    result = result.permute(result_axes)
            if copy is True:
                result = result.clone()
        except (TypeError, ValueError):
            return NotImplemented
        except RuntimeError as exc:
            if copy is False:
                raise ValueError(
                    "Unable to avoid a copy while reshaping"
                ) from exc
            return NotImplemented
        return self._wrap(result)

    def numpy_transpose(self, axes=None):
        """Implement NumPy's transpose using a Torch permutation view."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        if axes is None:
            dimensions = tuple(reversed(range(self.ndim)))
        else:
            try:
                dimensions = tuple(operator.index(axis) for axis in axes)
                normalized = tuple(
                    axis + self.ndim if axis < 0 else axis
                    for axis in dimensions
                )
            except (TypeError, ValueError):
                return NotImplemented
            if (
                len(normalized) != self.ndim
                or len(set(normalized)) != self.ndim
                or any(axis < 0 or axis >= self.ndim for axis in normalized)
            ):
                return NotImplemented
            dimensions = normalized
        return self._wrap(self.tensor.permute(dimensions))

    def numpy_swapaxes(self, axis1, axis2):
        """Implement NumPy's swapaxes as a Torch view."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        try:
            axis1 = operator.index(axis1)
            axis2 = operator.index(axis2)
            axis1 = axis1 + self.ndim if axis1 < 0 else axis1
            axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        except (TypeError, ValueError):
            return NotImplemented
        if not (0 <= axis1 < self.ndim and 0 <= axis2 < self.ndim):
            return NotImplemented
        return self._wrap(torch.swapaxes(self.tensor, axis1, axis2))

    def numpy_squeeze(self, axis=None):
        """Implement NumPy's squeeze while retaining exact axis checks."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        if axis is None:
            return self._wrap(torch.squeeze(self.tensor))

        axes = axis if isinstance(axis, tuple) else (axis,)
        try:
            axes = tuple(operator.index(value) for value in axes)
            axes = tuple(
                value + self.ndim if value < 0 else value for value in axes
            )
        except (TypeError, ValueError):
            return NotImplemented
        if (
            len(set(axes)) != len(axes)
            or any(value < 0 or value >= self.ndim for value in axes)
            or any(self.shape[value] != 1 for value in axes)
        ):
            return NotImplemented
        return self._wrap(torch.squeeze(self.tensor, dim=axes))

    def numpy_diagonal(self, offset=0, axis1=0, axis2=1):
        """Implement NumPy's diagonal selection as a Torch view."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        if self.ndim < 2:
            raise ValueError("diag requires an array of at least two dimensions")
        try:
            offset = operator.index(offset)
            original_axis1 = operator.index(axis1)
            original_axis2 = operator.index(axis2)
        except (TypeError, ValueError):
            return NotImplemented

        normalized_axes = []
        for original_axis in (original_axis1, original_axis2):
            normalized_axis = original_axis
            if normalized_axis < 0:
                normalized_axis += self.ndim
            if normalized_axis < 0 or normalized_axis >= self.ndim:
                raise np.exceptions.AxisError(
                    original_axis, ndim=self.ndim
                )
            normalized_axes.append(normalized_axis)
        if normalized_axes[0] == normalized_axes[1]:
            raise ValueError("axis1 and axis2 cannot be the same")

        try:
            result = torch.diagonal(
                self.tensor,
                offset=offset,
                dim1=normalized_axes[0],
                dim2=normalized_axes[1],
            )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented
        return self._wrap(result)

    def _flattened_tensor(self, order):
        """Return a one-dimensional tensor in NumPy's requested order."""
        order = self._normalize_numpy_order(order, ("C", "F", "A", "K"))
        if order is None:
            return NotImplemented

        tensor = self.tensor
        if order == "A":
            order = (
                "F"
                if self._is_fortran_contiguous(tensor)
                and not tensor.is_contiguous()
                else "C"
            )

        if order == "C":
            axes = tuple(range(tensor.ndim))
        elif order == "F":
            axes = tuple(reversed(range(tensor.ndim)))
        else:
            # Torch tensors have no negative strides.  Zero or repeated
            # nontrivial strides denote overlapping storage, for which NumPy's
            # K-order traversal cannot be represented by an axis permutation.
            nontrivial_strides = [
                stride for length, stride in zip(
                    tensor.shape, tensor.stride()
                ) if length > 1
            ]
            if (
                    any(stride <= 0 for stride in nontrivial_strides)
                    or len(set(nontrivial_strides))
                    != len(nontrivial_strides)):
                return NotImplemented
            axes = tuple(sorted(
                range(tensor.ndim),
                key=lambda axis: tensor.stride(axis),
                reverse=True,
            ))

        if axes != tuple(range(tensor.ndim)):
            tensor = tensor.permute(axes)
        return tensor.reshape(-1)

    def numpy_ravel(self, order="C"):
        """Implement NumPy-order ravel calls on the Torch device."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        tensor = self._flattened_tensor(order)
        if tensor is NotImplemented:
            return NotImplemented
        return self._wrap(tensor)

    def numpy_flatten(self, order="C"):
        """Implement NumPy-order flatten calls with copy semantics."""
        if not _device_matches_active(self.tensor):
            return NotImplemented
        tensor = self._flattened_tensor(order)
        if tensor is NotImplemented:
            return NotImplemented
        return self._wrap(tensor.clone())

    def cumsum(self):
        return self._wrap(self.tensor.cumsum(dim=0))

    def numpy_take(self, indices, axis=None, mode="raise"):
        """Evaluate NumPy take modes along any axis on the Torch device."""
        if not _device_matches_active(self.tensor) or mode not in (
                "raise", "wrap", "clip"):
            return NotImplemented

        source = self.tensor
        if axis is None:
            source = source.reshape(-1)
            dimension = 0
        else:
            try:
                dimension = operator.index(axis)
            except TypeError:
                return NotImplemented
            dimension = dimension + source.ndim if dimension < 0 else dimension
            if dimension < 0 or dimension >= source.ndim:
                return NotImplemented

        if isinstance(indices, TorchArrayData):
            if indices.tensor.device != self.tensor.device:
                return NotImplemented
            if np.dtype(indices.dtype).kind not in "iub":
                return NotImplemented
            index_tensor = indices.tensor
        elif isinstance(indices, torch.Tensor):
            if indices.device != self.tensor.device:
                return NotImplemented
            if indices.is_floating_point() or indices.is_complex():
                return NotImplemented
            index_tensor = indices
        else:
            try:
                index_values = np.asarray(indices)
                if (
                    isinstance(indices, np.ndarray)
                    and not np.can_cast(
                        index_values.dtype, np.intp, casting="same_kind"
                    )
                ):
                    return NotImplemented
                if index_values.dtype.kind == "c" or (
                    index_values.dtype.kind == "f"
                    and not np.all(np.isfinite(index_values))
                ):
                    return NotImplemented
                index_tensor = torch.as_tensor(
                    index_values, device=self.tensor.device
                )
            except (TypeError, ValueError, RuntimeError):
                return NotImplemented

        try:
            index_tensor = index_tensor.to(dtype=torch.int64)
            size = source.shape[dimension]
            if index_tensor.numel() and size == 0:
                raise IndexError(
                    "cannot do a non-empty take from an empty axes."
                )
            if mode == "wrap" and size:
                index_tensor = torch.remainder(index_tensor, size)
            elif mode == "clip" and size:
                index_tensor = torch.clamp(index_tensor, 0, size - 1)
            execution = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32 else source
            )
            selectors = [slice(None)] * execution.ndim
            selectors[dimension] = index_tensor
            result = execution[tuple(selectors)]
            if source.dtype == _TORCH_UINT32:
                result = result.to(dtype=_TORCH_UINT32)
        except IndexError:
            raise
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        return self._wrap(result)

    def numpy_repeat(self, repeats, axis=None):
        """Evaluate NumPy repeat along any axis on the Torch device."""
        if not _device_matches_active(self.tensor):
            return NotImplemented

        source = self.tensor
        if axis is None:
            source = source.reshape(-1)
            dimension = 0
        else:
            try:
                dimension = operator.index(axis)
            except TypeError:
                return NotImplemented
            dimension = dimension + source.ndim if dimension < 0 else dimension
            if dimension < 0 or dimension >= source.ndim:
                return NotImplemented

        if isinstance(repeats, TorchArrayData):
            if repeats.tensor.device != self.tensor.device:
                return NotImplemented
            repeat_tensor = repeats.tensor
            repeat_dtype = np.dtype(repeats.dtype)
        elif isinstance(repeats, torch.Tensor):
            if repeats.device != self.tensor.device:
                return NotImplemented
            repeat_tensor = repeats
            try:
                repeat_dtype = _numpy_dtype(repeats.dtype)
            except TypeError:
                if repeats.dtype != torch.bool:
                    return NotImplemented
                repeat_dtype = np.dtype(np.bool_)
        else:
            try:
                repeat_values = np.asarray(repeats)
            except (TypeError, ValueError):
                return NotImplemented
            repeat_dtype = repeat_values.dtype
            repeat_tensor = torch.as_tensor(
                repeat_values, device=self.tensor.device
            )

        if repeat_dtype.kind not in "iub" or repeat_tensor.ndim > 1:
            return NotImplemented
        try:
            execution = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32 else source
            )
            result = torch.repeat_interleave(
                execution,
                repeat_tensor.to(dtype=torch.int64),
                dim=dimension,
            )
            if source.dtype == _TORCH_UINT32:
                result = result.to(dtype=_TORCH_UINT32)
        except RuntimeError:
            return NotImplemented
        return self._wrap(result)

    def numpy_round(self, decimals=0):
        """Evaluate NumPy-compatible rounding without a host array copy."""
        if (
            not _device_matches_active(self.tensor)
            or not isinstance(decimals, (int, np.integer))
        ):
            return NotImplemented

        decimals = int(decimals)
        kind = np.dtype(self.dtype).kind
        try:
            if kind in "fc":
                result = _torch_round_decimals(self.tensor, decimals)
            elif kind in "iu" and decimals >= 0:
                result = self.tensor.clone()
            else:
                return NotImplemented
        except (OverflowError, RuntimeError):
            return NotImplemented
        return self._wrap(result)

    def numpy_clip(self, minimum=None, maximum=None, out=None, **kwargs):
        """Evaluate NumPy-compatible clipping without a host array copy."""
        if (
            kwargs
            or not _device_matches_active(self.tensor)
            or np.dtype(self.dtype).kind not in "fi"
        ):
            return NotImplemented

        bounds = []
        dtype_inputs = [np.dtype(self.dtype)]
        shapes = [self.shape]
        for bound in (minimum, maximum):
            if bound is None:
                bounds.append(None)
                continue
            if isinstance(bound, TorchArrayData):
                if (
                    bound.tensor.device != self.tensor.device
                    or np.dtype(bound.dtype).kind not in "fiu"
                ):
                    return NotImplemented
                value = bound.tensor
                dtype_inputs.append(np.dtype(bound.dtype))
            elif isinstance(bound, torch.Tensor):
                if bound.dtype not in _TORCH_TO_NUMPY:
                    return NotImplemented
                value = bound.to(device=self.tensor.device)
                dtype_inputs.append(_numpy_dtype(bound.dtype))
            else:
                try:
                    values = np.asarray(bound)
                except (TypeError, ValueError):
                    return NotImplemented
                if values.dtype.kind not in "fiu":
                    return NotImplemented
                value = values
                dtype_inputs.append(
                    bound if np.isscalar(bound) else values.dtype
                )
            bounds.append(value)
            shapes.append(tuple(value.shape) if hasattr(value, "shape") else ())

        try:
            target_dtype = np.dtype(np.result_type(*dtype_inputs))
            if target_dtype.kind not in "fi":
                return NotImplemented
            target_torch = _ensure_supported(
                self.tensor.device, _torch_dtype(target_dtype)
            )
            if tuple(torch.broadcast_shapes(*shapes)) != self.shape:
                return NotImplemented
            tensor_bounds = [
                None if bound is None else torch.as_tensor(
                    bound, dtype=target_torch, device=self.tensor.device
                )
                for bound in bounds
            ]
            if minimum is None and maximum is None:
                result = self.tensor.to(dtype=target_torch).clone()
            else:
                result = torch.clamp(
                    self.tensor.to(dtype=target_torch),
                    min=tensor_bounds[0],
                    max=tensor_bounds[1],
                )
        except (OverflowError, TypeError, ValueError, RuntimeError):
            return NotImplemented

        if out is None:
            return self._wrap(result)
        if (
            not isinstance(out, TorchArrayData)
            or out.tensor.device != self.tensor.device
            or out.shape != self.shape
            or not np.can_cast(
                target_dtype, np.dtype(out.dtype), casting="same_kind"
            )
        ):
            return NotImplemented
        try:
            out.tensor.copy_(result)
        except RuntimeError:
            return NotImplemented
        return out

    def numpy(self):
        tensor = _resolve_for_numpy(self.tensor)
        return tensor.detach().cpu().numpy()

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    def __repr__(self):
        return f"TorchArrayData({repr(self.tensor)})"


def _scheme_matches_base_array(array):
    return isinstance(array, TorchArrayData) and _device_matches_active(array.tensor)


def _copy_base_array(array):
    return TorchArrayData(array.tensor.clone())


def _to_device(array):
    device = _torch_device()
    torch_dtype = _ensure_supported(device, _torch_dtype(array.dtype))
    tensor = torch.tensor(np.ascontiguousarray(array), device=device, dtype=torch_dtype)
    return TorchArrayData(tensor)


def zeros(length, dtype=np.float64):
    device = _torch_device()
    torch_dtype = _ensure_supported(device, _torch_dtype(dtype))
    return TorchArrayData(torch.zeros(length, device=device, dtype=torch_dtype))


def zeros_pinned(shape, dtype=np.float64):
    """Return a zero-initialized page-locked (pinned) host memory buffer.

    Parameters
    ----------
    shape : int or tuple of ints
        Size or shape of the requested buffer.
    dtype : numpy.dtype or torch.dtype, optional
        Data type of the requested buffer (default numpy.float64).

    Returns
    -------
    TorchArrayData
        Array data wrapping the pinned CPU tensor.
    """
    torch_dtype = _torch_dtype(dtype)
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    else:
        shape = tuple(int(s) for s in shape)
    if torch.cuda.is_available():
        tensor = torch.zeros(shape, dtype=torch_dtype, pin_memory=True)
    else:
        tensor = torch.zeros(shape, dtype=torch_dtype)
    return TorchArrayData(tensor)


def to_cuda_async(tensor, stream=None, device="cuda"):
    """Transfer tensor or Array to CUDA asynchronously using non-blocking copy.

    Parameters
    ----------
    tensor : torch.Tensor or TorchArrayData or Array or TimeSeries or FrequencySeries
        Input pinned host or device tensor / array.
    stream : torch.cuda.Stream, optional
        CUDA stream to execute the transfer on.
    device : str or torch.device, optional
        Target CUDA device (default 'cuda').

    Returns
    -------
    TorchArrayData or torch.Tensor
        Target device tensor on CUDA, retaining the input encapsulation.
    """
    is_wrapped = isinstance(tensor, TorchArrayData)
    if is_wrapped:
        raw = tensor.tensor
    elif hasattr(tensor, "_data") and isinstance(tensor._data, TorchArrayData):
        raw = tensor._data.tensor
    elif isinstance(tensor, torch.Tensor):
        raw = tensor
    elif hasattr(tensor, "_data") and isinstance(tensor._data, torch.Tensor):
        raw = tensor._data
    else:
        raw = torch.as_tensor(tensor)

    if not torch.cuda.is_available():
        out = raw.to(device=raw.device)
    elif stream is not None:
        with torch.cuda.stream(stream):
            out = raw.to(device=device, non_blocking=True)
    else:
        out = raw.to(device=device, non_blocking=True)

    if is_wrapped:
        return TorchArrayData(out)
    return out


def empty(length, dtype=np.float64):
    device = _torch_device()
    torch_dtype = _ensure_supported(device, _torch_dtype(dtype))
    return TorchArrayData(torch.empty(length, device=device, dtype=torch_dtype))


def ptr(self):
    return int(self._data.tensor.data_ptr())


def dot(self, other):
    other_t, other_np = _tensor_from_any(other, self._data.tensor.device)
    self_t, other_t, _ = _promote_tensors(self._data.tensor, self._data.dtype,
                                          other_t, other_np)
    return torch.dot(self_t, other_t).item()


def min(self):
    return self._data.tensor.min().item()


def abs_max_loc(self):
    tensor = self._data.tensor
    val, idx = torch.max(torch.abs(tensor).flatten(), dim=0)
    return val.item(), int(idx.item())


def cumsum(self):
    return self._data.cumsum()


def max(self):
    return self._data.tensor.max().item()


def max_loc(self):
    tensor = self._data.tensor
    val, idx = torch.max(tensor.flatten(), dim=0)
    return val.item(), int(idx.item())

def multiply_and_add(self, other, mult_fac):
    """ other * mult_fac + self (self mutated) """
    other_t, other_np = _tensor_from_any(other, self._data.tensor.device)
    a, other_t, _ = _promote_tensors(self._data.tensor, self._data.dtype,
                                     other_t, other_np)
    if isinstance(mult_fac, torch.Tensor):
        mult_fac = mult_fac.to(device=a.device)
    elif hasattr(mult_fac, "item"):
        mult_fac = mult_fac.item()
    if a.dtype == self._data.tensor.dtype and a.data_ptr() == self._data.tensor.data_ptr():
        self._data.tensor.add_(other_t, alpha=mult_fac)
    else:
        result = a + other_t * mult_fac
        self._data = TorchArrayData(result)
    self._saved = None
    return self._data


def take(self, indices):
    tensor = self._data.tensor
    index_data = getattr(indices, "_data", indices)
    if isinstance(index_data, TorchArrayData):
        idx = index_data.tensor.to(device=tensor.device, dtype=torch.long)
    elif isinstance(index_data, torch.Tensor):
        idx = index_data.to(device=tensor.device, dtype=torch.long)
    else:
        idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return TorchArrayData(tensor.reshape(-1)[idx])


def weighted_inner(self, other, weight):
    if weight is None:
        return inner(self, other)

    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    w, w_np = _tensor_from_any(weight, a.device)
    a, b, target_np = _promote_tensors(a, self._data.dtype, b, b_np)
    w, _, _ = _promote_tensors(w, w_np, b, target_np)

    is_same = (
        self is other
        or a is b
        or (a.data_ptr() == b.data_ptr() and a.shape == b.shape)
    )
    if is_same and a.is_complex():
        complex_weight = np.dtype(w_np).kind == "c"
        sq_mag = torch.view_as_real(a).square().sum(dim=-1)
        if not complex_weight:
            w_real = w.real if w.is_complex() else w
            accum_dtype = _accumulation_dtype(a, complex_result=False)
            return torch.sum(
                sq_mag / w_real,
                dtype=accum_dtype,
            ).item()
        accum_dtype = _accumulation_dtype(a, complex_result=True)
        return torch.sum(
            sq_mag / w, dtype=accum_dtype
        ).item()

    accum_dtype = _accumulation_dtype(
        a, complex_result=b.dtype.is_complex or a.dtype.is_complex
    )
    return torch.sum(torch.conj(a) * b / w, dtype=accum_dtype).item()


def abs_arg_max(self):
    tensor = self._data.tensor
    if tensor.is_complex():
        idx = torch.argmax(torch.view_as_real(tensor).square().sum(dim=-1))
    else:
        idx = torch.argmax(torch.abs(tensor))
    return int(idx.item())


def inner(self, other):
    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    a, b, _ = _promote_tensors(a, self._data.dtype, b, b_np)

    is_same = (
        self is other
        or a is b
        or (a.data_ptr() == b.data_ptr() and a.shape == b.shape)
    )
    if is_same and a.is_complex():
        accum_dtype = _accumulation_dtype(a, complex_result=False)
        sq_mag = torch.view_as_real(a).square().sum(dim=-1)
        return torch.sum(
            sq_mag, dtype=accum_dtype
        ).item()

    complex_result = a.dtype.is_complex or b.dtype.is_complex
    accum_dtype = _accumulation_dtype(a, complex_result=complex_result)
    if complex_result:
        acc = torch.sum(torch.conj(a) * b, dtype=accum_dtype)
    else:
        acc = torch.sum(a * b, dtype=accum_dtype)
    return acc.item()


def vdot(self, other):
    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    a, b, _ = _promote_tensors(a, self._data.dtype, b, b_np)
    return torch.vdot(a, b).item()


def squared_norm(self):
    tensor = self._data.tensor
    if tensor.is_complex():
        out = torch.view_as_real(tensor).square().sum(dim=-1)
    else:
        out = tensor.square()
    return TorchArrayData(out)


def numpy(self):
    tensor = _resolve_for_numpy(self._data.tensor)
    return tensor.detach().cpu().numpy()


def _copy(self, self_ref, other_ref):
    dst = self_ref.tensor if isinstance(self_ref, TorchArrayData) else self_ref
    src = other_ref.tensor if isinstance(other_ref, TorchArrayData) else other_ref
    dst.copy_(src)


def _getvalue(self, index):
    tensor = self._data.tensor
    index_data = getattr(index, "_data", index)
    if isinstance(index_data, TorchArrayData):
        index = index_data.tensor.to(device=tensor.device)
    elif isinstance(index_data, torch.Tensor):
        index = index_data.to(device=tensor.device)
    res = tensor[index]
    if res.ndim == 0:
        return _resolve_for_numpy(res).item()
    if hasattr(self, "_return"):
        return self._return(TorchArrayData(res))
    from pycbc.types.array import Array
    return Array(TorchArrayData(res), copy=False)


def sum(self):
    tensor = self._data.tensor
    accum_dtype = _accumulation_dtype(
        tensor, complex_result=tensor.is_complex()
    )
    return torch.sum(tensor, dtype=accum_dtype).item()


def clear(self):
    self._data.tensor.zero_()
