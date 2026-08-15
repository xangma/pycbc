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

import numpy as np
import torch

import pycbc.scheme as _scheme


_NUMPY_TO_TORCH = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
    np.dtype(np.uint32): torch.uint32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
}
_TORCH_TO_NUMPY = {v: k for k, v in _NUMPY_TO_TORCH.items()}


def _torch_device():
    """Return the torch.device for the current scheme."""
    state = _scheme.mgr.state
    if hasattr(state, "torch_device"):
        return state.torch_device
    return torch.device("cpu")


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
                torch.float32, torch.float16, torch.complex64):
            raise TypeError(
                "MPS backend only supports float16/float32/complex64 tensors"
            )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA device requested but unavailable")
    return torch_dtype


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


def _resolve_for_numpy(tensor):
    """Ensure tensors with a conjugate bit are materialized before numpy conversion."""
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    return tensor


class TorchArrayData:
    """Lightweight wrapper around a torch tensor with numpy dtype semantics."""

    __slots__ = ("tensor", "dtype")
    __array_priority__ = 100.0

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

    def _wrap(self, tensor):
        return TorchArrayData(tensor)

    def _set_tensor(self, tensor):
        self.tensor = tensor
        self.dtype = _numpy_dtype(tensor.dtype)
        return self

    def _promote_with(self, other):
        other_t, other_np = _tensor_from_any(other, self.tensor.device)
        return _promote_tensors(self.tensor, self.dtype, other_t, other_np)

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

    def cumsum(self):
        return self._wrap(self.tensor.cumsum(dim=0))

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
    return isinstance(array, TorchArrayData)


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
    abs_vals = torch.abs(tensor)
    idx = torch.argmax(abs_vals)
    return abs_vals[idx].item(), int(idx.item())


def cumsum(self):
    return self._data.cumsum()


def max(self):
    return self._data.tensor.max().item()


def max_loc(self):
    tensor = self._data.tensor
    idx = torch.argmax(tensor)
    return tensor[idx].item(), int(idx.item())

def multiply_and_add(self, other, mult_fac):
    """ other * mult_fac + self (self mutated) """
    other_t, other_np = _tensor_from_any(other, self._data.tensor.device)
    a, other_t, _ = _promote_tensors(self._data.tensor, self._data.dtype,
                                     other_t, other_np)
    if hasattr(mult_fac, "item"):
        mult_fac = mult_fac.item()
    result = a + other_t * mult_fac
    self._data = TorchArrayData(result)
    return self._data


def take(self, indices):
    tensor = self._data.tensor
    idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return TorchArrayData(torch.take(tensor, idx))


def weighted_inner(self, other, weight):
    if weight is None:
        return inner(self, other)

    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    w, w_np = _tensor_from_any(weight, a.device)
    a, b, target_np = _promote_tensors(a, self._data.dtype, b, b_np)
    w, _, _ = _promote_tensors(w, w_np, b, target_np)

    accum_dtype = torch.complex128 if b.dtype.is_complex or a.dtype.is_complex \
        else torch.float64
    return torch.sum(torch.conj(a) * b / w, dtype=accum_dtype).item()


def abs_arg_max(self):
    idx = torch.argmax(torch.abs(self._data.tensor))
    return int(idx.item())


def inner(self, other):
    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    a, b, _ = _promote_tensors(a, self._data.dtype, b, b_np)

    if a.dtype.is_complex or b.dtype.is_complex:
        acc = torch.sum(torch.conj(a) * b, dtype=torch.complex128)
    else:
        acc = torch.sum(a * b, dtype=torch.float64)
    return acc.item()


def vdot(self, other):
    a = self._data.tensor
    b, b_np = _tensor_from_any(other, a.device)
    a, b, _ = _promote_tensors(a, self._data.dtype, b, b_np)
    return torch.vdot(a, b).item()


def squared_norm(self):
    tensor = self._data.tensor
    if tensor.is_complex():
        real = tensor.real
        imag = tensor.imag
        out = real * real + imag * imag
    else:
        out = tensor * tensor
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
    res = tensor[index]
    if res.ndim == 0:
        return _resolve_for_numpy(res).item()
    return TorchArrayData(res)


def sum(self):
    tensor = self._data.tensor
    if tensor.is_complex():
        accum_dtype = torch.complex128
    else:
        accum_dtype = torch.float64
    return torch.sum(tensor, dtype=accum_dtype).item()


def clear(self):
    self._data.tensor.zero_()
