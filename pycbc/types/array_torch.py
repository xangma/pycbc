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

"""Core storage and scheme operations for the PyCBC Torch array backend.

NumPy interoperability is implemented separately in
:mod:`pycbc.types.array_torch_numpy`.  Keeping this module as the stable
backend entry point preserves the public ``TorchArrayData`` import and the
``@schemed`` function lookup convention.
"""

import operator

import numpy as np
import torch

from .array_torch_numpy import (
    _TORCH_TO_NUMPY,
    _TORCH_UINT32,
    TorchArrayNumpyCompatibilityMixin,
    _accumulation_dtype,
    _comparison_tensors,
    _device_matches_active,
    _ensure_supported,
    _numpy_dtype,
    _promote_tensors,
    _resolve_for_numpy,
    _tensor_from_any,
    _torch_device,
    _torch_dtype,
    _torch_round_decimals,
)


class TorchArrayData(TorchArrayNumpyCompatibilityMixin):
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

        left, right, outside_range = _comparison_tensors(self.tensor, self.dtype, other)
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
            torch.eq(left.real, right.real) & function(left.imag, right.imag)
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
        return self._wrap(a**b)

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
            raise TypeError(f"Cannot view {self.dtype} data as {target_np}") from exc

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
                normalized_shape = tuple(operator.index(length) for length in shape)

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
                raise ValueError("Unable to avoid a copy while reshaping") from exc
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
                    axis + self.ndim if axis < 0 else axis for axis in dimensions
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
            axes = tuple(value + self.ndim if value < 0 else value for value in axes)
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
                raise np.exceptions.AxisError(original_axis, ndim=self.ndim)
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
                if self._is_fortran_contiguous(tensor) and not tensor.is_contiguous()
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
                stride
                for length, stride in zip(tensor.shape, tensor.stride(), strict=True)
                if length > 1
            ]
            if any(stride <= 0 for stride in nontrivial_strides) or len(
                set(nontrivial_strides)
            ) != len(nontrivial_strides):
                return NotImplemented
            axes = tuple(
                sorted(
                    range(tensor.ndim),
                    key=lambda axis: tensor.stride(axis),
                    reverse=True,
                )
            )

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
            "raise",
            "wrap",
            "clip",
        ):
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
                if isinstance(indices, np.ndarray) and not np.can_cast(
                    index_values.dtype, np.intp, casting="same_kind"
                ):
                    return NotImplemented
                if index_values.dtype.kind == "c" or (
                    index_values.dtype.kind == "f"
                    and not np.all(np.isfinite(index_values))
                ):
                    return NotImplemented
                index_tensor = torch.as_tensor(index_values, device=self.tensor.device)
            except (TypeError, ValueError, RuntimeError):
                return NotImplemented

        try:
            index_tensor = index_tensor.to(dtype=torch.int64)
            size = source.shape[dimension]
            if index_tensor.numel() and size == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            if mode == "wrap" and size:
                index_tensor = torch.remainder(index_tensor, size)
            elif mode == "clip" and size:
                index_tensor = torch.clamp(index_tensor, 0, size - 1)
            execution = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32
                else source
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
            repeat_tensor = torch.as_tensor(repeat_values, device=self.tensor.device)

        if repeat_dtype.kind not in "iub" or repeat_tensor.ndim > 1:
            return NotImplemented
        try:
            execution = (
                source.to(dtype=torch.int64)
                if source.dtype == _TORCH_UINT32
                else source
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
        if not _device_matches_active(self.tensor) or not isinstance(
            decimals, (int, np.integer)
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
                dtype_inputs.append(bound if np.isscalar(bound) else values.dtype)
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
                None
                if bound is None
                else torch.as_tensor(
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
            or not np.can_cast(target_dtype, np.dtype(out.dtype), casting="same_kind")
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
    if isinstance(array, TorchArrayData):
        return TorchArrayData(array.tensor.to(device=device, dtype=torch_dtype))
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
    self_t, other_t, _ = _promote_tensors(
        self._data.tensor, self._data.dtype, other_t, other_np
    )
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
    """other * mult_fac + self (self mutated)"""
    other_t, other_np = _tensor_from_any(other, self._data.tensor.device)
    a, other_t, _ = _promote_tensors(
        self._data.tensor, self._data.dtype, other_t, other_np
    )
    if isinstance(mult_fac, torch.Tensor):
        mult_fac = mult_fac.to(device=a.device)
    elif hasattr(mult_fac, "item"):
        mult_fac = mult_fac.item()
    if (
        a.dtype == self._data.tensor.dtype
        and a.data_ptr() == self._data.tensor.data_ptr()
    ):
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
        self is other or a is b or (a.data_ptr() == b.data_ptr() and a.shape == b.shape)
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
        return torch.sum(sq_mag / w, dtype=accum_dtype).item()

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
        self is other or a is b or (a.data_ptr() == b.data_ptr() and a.shape == b.shape)
    )
    if is_same and a.is_complex():
        accum_dtype = _accumulation_dtype(a, complex_result=False)
        sq_mag = torch.view_as_real(a).square().sum(dim=-1)
        return torch.sum(sq_mag, dtype=accum_dtype).item()

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
    accum_dtype = _accumulation_dtype(tensor, complex_result=tensor.is_complex())
    return torch.sum(tensor, dtype=accum_dtype).item()


def clear(self):
    self._data.tensor.zero_()
