# Copyright (C) 2025
#
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
Torch backend for matched filtering primitives.
"""

import ctypes
import functools
import os
import threading

import numpy as np
import torch
from pycbc import PYCBC_ALIGNMENT
from .matchedfilter import _BaseCorrelator


# The established Cython kernel wins once each OpenMP worker receives enough
# elements, while Torch is faster for one-thread and over-threaded calls.  Keep
# this deliberately conservative: these boundaries are covered by crossover
# benchmarks on the production CPU host.
_CPU_NATIVE_MIN_THREADS = 4
_CPU_NATIVE_MAX_THREADS = 64
_CPU_NATIVE_ELEMENTS_PER_THREAD = 512
_CPU_NATIVE_MAX_LENGTH = 2**32 - 1
_CPU_NATIVE_BATCH_GATE = "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"
_CUDA_NATIVE_BATCH_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE"
_CUDA_NATIVE_BATCH_PEAK_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"
_ASYNC_STREAMS_GATE = "PYCBC_TORCH_ASYNC_STREAMS"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_TORCH_IS_INFERENCE = getattr(torch, "is_inference", None)


def _torch_inference_mode_context():
    """Return torch.inference_mode() or fallback to torch.no_grad()."""
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    if hasattr(torch, "no_grad"):
        return torch.no_grad()
    from contextlib import nullcontext
    return nullcontext()


def _environment_flag(name, default=False):
    """Read a strict boolean environment switch."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    choices = ", ".join(sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES))
    raise ValueError(f"{name} must be one of: {choices}; got {value!r}")


def _has_autograd_state(tensor):
    """Whether bypassing Torch would discard reverse- or forward-mode AD."""
    if tensor.requires_grad:
        return True
    try:
        return torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
    except (AttributeError, RuntimeError):
        return True


def _has_dynamic_autograd_state(tensors):
    """Check AD state immediately before entering the opaque CPU kernel."""
    if any(tensor.requires_grad for tensor in tensors):
        return True

    # Avoid unpacking every tensor in the ordinary search path.  Forward duals
    # can only exist inside an active dual level.
    current_level = getattr(torch.autograd.forward_ad, "_current_level", None)
    if current_level == -1:
        return False
    return any(_has_autograd_state(tensor) for tensor in tensors)


def _storage_span(tensor):
    """Return the occupied byte interval of a contiguous tensor."""
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _spans_overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _logical_storage_span(tensor, size):
    """Return the byte interval occupied by a contiguous logical prefix."""
    start = tensor.data_ptr()
    return start, start + size * tensor.element_size()


def _is_inference_tensor(tensor):
    if _TORCH_IS_INFERENCE is None:
        return False
    try:
        return _TORCH_IS_INFERENCE(tensor)
    except RuntimeError:
        return True


def _cpu_native_static_eligible(x, y, z):
    """Whether fixed buffers satisfy the Cython correlation ABI."""
    tensors = (x, y, z)
    if not all(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.complex64
        and tensor.layout == torch.strided
        and tensor.ndim == 1
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and not _has_autograd_state(tensor)
        for tensor in tensors
    ):
        return False
    if not (x.numel() == y.numel() == z.numel() <= _CPU_NATIVE_MAX_LENGTH):
        return False

    # Torch rejects partially overlapping ``out=`` operations.  Keep all
    # output aliases on that path rather than weakening its safety contract.
    z_span = _storage_span(z)
    return not (
        _spans_overlap(_storage_span(x), z_span)
        or _spans_overlap(_storage_span(y), z_span)
    )


@functools.lru_cache(maxsize=None)
def _cpu_native_openmp_runtime(module):
    """Resolve OpenMP state through the Cython extension or system OpenMP runtime."""
    import ctypes.util
    candidates = []
    if hasattr(module, "__file__") and module.__file__:
        candidates.append(module.__file__)
    candidates.append(None)
    for name in ("omp", "gomp", "iomp5"):
        lib = ctypes.util.find_library(name)
        if lib:
            candidates.append(lib)
    for target in candidates:
        try:
            handle = ctypes.CDLL(target)
            get_max_threads = handle.omp_get_max_threads
            get_dynamic = handle.omp_get_dynamic
            get_max_threads.argtypes = []
            get_max_threads.restype = ctypes.c_int
            get_dynamic.argtypes = []
            get_dynamic.restype = ctypes.c_int
            return handle, get_max_threads, get_dynamic
        except (AttributeError, OSError, TypeError):
            continue
    return None


def _cpu_native_dispatch_is_beneficial(length, openmp_runtime):
    """Select native OpenMP only beyond its measured thread crossover.

    ``torch.get_num_threads`` is not by itself authoritative for a Cython
    OpenMP loop.  Only reuse the native kernel when its linked runtime reports
    the same fixed team size and dynamic adjustment is disabled.
    """
    if openmp_runtime is None:
        return False
    _, get_max_threads, get_dynamic = openmp_runtime
    try:
        torch_threads = torch.get_num_threads()
        openmp_threads = get_max_threads()
        dynamic = get_dynamic()
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        not dynamic
        and openmp_threads == torch_threads
        and _CPU_NATIVE_MIN_THREADS
        <= openmp_threads
        <= _CPU_NATIVE_MAX_THREADS
        and length
        >= openmp_threads * _CPU_NATIVE_ELEMENTS_PER_THREAD
    )


def _cpu_native_batch_runtime_is_stable(openmp_runtime):
    """Validate that Torch and the Cython loop see one fixed OMP team."""
    if openmp_runtime is None:
        return False
    _, get_max_threads, get_dynamic = openmp_runtime
    try:
        torch_threads = torch.get_num_threads()
        openmp_threads = get_max_threads()
        dynamic = get_dynamic()
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        not dynamic
        and 1 <= openmp_threads <= _CPU_NATIVE_MAX_THREADS
        and openmp_threads == torch_threads
    )


def _batch_tensor_contract(tensor, size):
    """Whether a tensor can be passed through the native pointer ABI."""
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.complex64
        and tensor.layout == torch.strided
        and tensor.ndim == 1
        and tensor.numel() >= size
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and not _is_inference_tensor(tensor)
        and not _has_autograd_state(tensor)
        and tensor.data_ptr() % PYCBC_ALIGNMENT == 0
    )


def _batch_buffers_are_disjoint(x_tensors, y_tensor, z_tensors, size):
    """Reject every output alias and parallel output/output overlap."""
    input_spans = [
        _logical_storage_span(tensor, size)
        for tensor in (*x_tensors, y_tensor)
    ]
    return _batch_outputs_are_disjoint(input_spans, z_tensors, size)


def _batch_outputs_are_disjoint(input_spans, z_tensors, size):
    """Reject output/input and output/output logical-span overlap."""
    output_spans = [
        _logical_storage_span(tensor, size) for tensor in z_tensors
    ]
    for index, output_span in enumerate(output_spans):
        if any(
            _spans_overlap(output_span, input_span)
            for input_span in input_spans
        ):
            return False
        if any(
            _spans_overlap(output_span, other)
            for other in output_spans[index + 1 :]
        ):
            return False
    return True


def _same_array_tensors(arrays, tensors, *args):
    """Check that arrays hold the exact cached torch.Tensor instances."""
    if len(args) == 2:
        owners, tensors, pointers = tensors, args[0], args[1]
        if len(arrays) != len(owners):
            return False
        for array, owner, tensor, pointer in zip(
            arrays, owners, tensors, pointers
        ):
            try:
                current = array._data.tensor
            except AttributeError:
                return False
            if (
                array is not owner
                or current is not tensor
                or current.data_ptr() != pointer
            ):
                return False
        return True

    if len(arrays) != len(tensors):
        return False
    for array, tensor in zip(arrays, tensors):
        try:
            current = array._data.tensor
        except AttributeError:
            return False
        if current is not tensor or current.data_ptr() != tensor.data_ptr():
            return False
    return True


class _CPUNativeBatchCorrelationState:
    """Strong-owner state for fixed batch buffers and a dynamic data input."""

    def __init__(self, batch, matchedfilter_cpu, openmp_runtime):
        size = int(batch.size)
        num_vectors = int(batch.num_vectors)
        batch_xs = getattr(batch, "_xs", getattr(batch, "xs", None))
        batch_zs = getattr(batch, "_zs", getattr(batch, "zs", None))
        if (
            size <= 0
            or size > _CPU_NATIVE_MAX_LENGTH
            or num_vectors <= 1
            or num_vectors > _CPU_NATIVE_MAX_LENGTH
            or batch_xs is None
            or batch_zs is None
            or len(batch_xs) != num_vectors
            or len(batch_zs) != num_vectors
            or not _cpu_native_batch_runtime_is_stable(openmp_runtime)
            or not hasattr(torch.autograd.graph, "increment_version")
        ):
            raise ValueError("batch correlation contract is unsupported")

        x_arrays = tuple(batch_xs)
        z_arrays = tuple(batch_zs)
        x_tensors = tuple(array._data.tensor for array in x_arrays)
        z_tensors = tuple(array._data.tensor for array in z_arrays)
        tensors = (*x_tensors, *z_tensors)
        if not all(_batch_tensor_contract(tensor, size) for tensor in tensors):
            raise ValueError("batch correlation tensor contract is unsupported")
        if not _batch_outputs_are_disjoint(
            [_logical_storage_span(tensor, size) for tensor in x_tensors],
            z_tensors,
            size,
        ):
            raise ValueError("batch correlation buffers overlap")

        # These private pointer tables are tiny and immutable.  The waveform
        # fixed buffers remain zero-copy, while strong Tensor owners keep every
        # address alive until the opaque Cython call returns.  The live data
        # input is deliberately validated and viewed anew for each execute.
        x_pointers = tuple(tensor.data_ptr() for tensor in x_tensors)
        z_pointers = tuple(tensor.data_ptr() for tensor in z_tensors)
        pointer_dtype = np.dtype(np.int_)
        if pointer_dtype.itemsize != ctypes.sizeof(ctypes.c_long):
            raise ValueError("NumPy integer does not match C long")
        self._x_pointer_table = np.asarray(
            x_pointers, dtype=pointer_dtype
        )
        self._z_pointer_table = np.asarray(
            z_pointers, dtype=pointer_dtype
        )
        self._x_pointer_table.flags.writeable = False
        self._z_pointer_table.flags.writeable = False
        self._pid = os.getpid()
        self._thread_id = threading.get_ident()
        self._size = size
        self._num_vectors = num_vectors
        self._x_arrays = x_arrays
        self._z_arrays = z_arrays
        self._x_tensors = x_tensors
        self._z_tensors = z_tensors
        self._x_pointers = x_pointers
        self._z_pointers = z_pointers
        self._z_spans = tuple(
            _logical_storage_span(tensor, size) for tensor in z_tensors
        )
        self._openmp_runtime = openmp_runtime
        self._matchedfilter_cpu = matchedfilter_cpu
        self._function = matchedfilter_cpu._batch_correlate
        self._epoch = getattr(batch, "_epoch", 0)

    _same_array_tensors = staticmethod(_same_array_tensors)

    def can_execute(self, batch):
        try:
            batch_xs = getattr(batch, "_xs", getattr(batch, "xs", None))
            batch_zs = getattr(batch, "_zs", getattr(batch, "zs", None))
            if (
                os.getpid() != self._pid
                or threading.get_ident() != self._thread_id
                or int(batch.size) != self._size
                or int(batch.num_vectors) != self._num_vectors
                or getattr(batch, "_epoch", 0) != self._epoch
                or batch_xs is None
                or batch_zs is None
                or not _same_array_tensors(
                    batch_xs,
                    self._x_arrays,
                    self._x_tensors,
                    self._x_pointers,
                )
                or not _same_array_tensors(
                    batch_zs,
                    self._z_arrays,
                    self._z_tensors,
                    self._z_pointers,
                )
                or not _cpu_native_batch_runtime_is_stable(
                    self._openmp_runtime
                )
            ):
                return False
            tensors = (
                *self._x_tensors,
                *self._z_tensors,
            )
            return (
                all(
                    _batch_tensor_contract(tensor, self._size)
                    for tensor in tensors
                )
                and _batch_outputs_are_disjoint(
                    [
                        _logical_storage_span(tensor, self._size)
                        for tensor in self._x_tensors
                    ],
                    self._z_tensors,
                    self._size,
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def execute(self, batch, y):
        """Run with this call's eligible live data tensor, or decline it."""
        try:
            batch_xs = getattr(batch, "_xs", getattr(batch, "xs", None))
            batch_zs = getattr(batch, "_zs", getattr(batch, "zs", None))
            if (
                os.getpid() != self._pid
                or threading.get_ident() != self._thread_id
                or int(batch.size) != self._size
                or int(batch.num_vectors) != self._num_vectors
                or getattr(batch, "_epoch", 0) != self._epoch
                or batch_xs is None
                or batch_zs is None
                or not _same_array_tensors(
                    batch_xs,
                    self._x_arrays,
                    self._x_tensors,
                    self._x_pointers,
                )
                or not _same_array_tensors(
                    batch_zs,
                    self._z_arrays,
                    self._z_tensors,
                    self._z_pointers,
                )
                or not _cpu_native_batch_runtime_is_stable(
                    self._openmp_runtime
                )
                or _has_dynamic_autograd_state(self._x_tensors)
                or _has_dynamic_autograd_state(self._z_tensors)
            ):
                return False
            y_tensor = getattr(getattr(y, "_data", None), "tensor", None)
            if (
                y_tensor is None
                or not _batch_tensor_contract(y_tensor, self._size)
            ):
                return False
            y_span = _logical_storage_span(y_tensor, self._size)
            if any(_spans_overlap(y_span, z_span) for z_span in self._z_spans):
                return False
            y_pointer = y_tensor.data_ptr()
            y_view = y_tensor.detach().numpy()
            if y_view.__array_interface__["data"][0] != y_pointer:
                return False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        self._function(
            self._x_pointer_table,
            y_view,
            self._z_pointer_table,
            self._size,
            self._num_vectors,
        )
        for tensor in self._z_tensors:
            torch.autograd.graph.increment_version(tensor)
        return True


def _try_cpu_native_batch_correlate(batch, y):
    """Execute the gated native batch route, or request Torch fallback."""
    state = getattr(batch, "_torch_cpu_native_batch_state", None)
    if state is not None:
        return state.execute(batch, y)
    try:
        from . import matchedfilter_cpu

        openmp_runtime = _cpu_native_openmp_runtime(matchedfilter_cpu)
        state = _CPUNativeBatchCorrelationState(
            batch, matchedfilter_cpu, openmp_runtime
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    batch._torch_cpu_native_batch_state = state
    return state.execute(batch, y)


def _cuda_batch_tensor_contract(tensor, size):
    """Whether a tensor can be used in CUDA native batched operations."""
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cuda"
        and tensor.dtype == torch.complex64
        and tensor.layout == torch.strided
        and tensor.ndim == 1
        and tensor.numel() >= size
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and not _is_inference_tensor(tensor)
        and not _has_autograd_state(tensor)
    )


def _find_uniform_stride(tensors, size):
    """Return the uniform row stride in elements if tensors form a strided block, else None."""
    num_vectors = len(tensors)
    if num_vectors == 0:
        return None
    if num_vectors == 1:
        return tensors[0].numel()
    itemsize = tensors[0].element_size()
    ptr0 = tensors[0].data_ptr()
    ptr1 = tensors[1].data_ptr()
    step_bytes = ptr1 - ptr0
    if step_bytes % itemsize != 0:
        return None
    stride = step_bytes // itemsize
    if stride < size:
        return None
    for i in range(num_vectors):
        if tensors[i].data_ptr() != ptr0 + i * stride * itemsize:
            return None
        if not tensors[i].is_contiguous():
            return None
        if tensors[i].numel() < size:
            return None
    return stride


class _CUDANativeBatchCorrelationState:
    """Zero-copy 2D batched tensor correlation state for CUDA."""

    def __init__(self, batch):
        size = int(batch.size)
        num_vectors = int(batch.num_vectors)
        if (
            size <= 0
            or size > _CPU_NATIVE_MAX_LENGTH
            or num_vectors <= 1
            or num_vectors > _CPU_NATIVE_MAX_LENGTH
            or len(batch.xs) != num_vectors
            or len(batch.zs) != num_vectors
            or not hasattr(torch.autograd.graph, "increment_version")
        ):
            raise ValueError("CUDA batch correlation contract is unsupported")

        x_arrays = tuple(batch.xs)
        z_arrays = tuple(batch.zs)
        x_tensors = tuple(array._data.tensor for array in x_arrays)
        z_tensors = tuple(array._data.tensor for array in z_arrays)
        tensors = (*x_tensors, *z_tensors)
        if not all(_cuda_batch_tensor_contract(tensor, size) for tensor in tensors):
            raise ValueError("CUDA batch correlation tensor contract is unsupported")

        device = x_tensors[0].device
        if any(t.device != device for t in tensors):
            raise ValueError("CUDA batch correlation tensors must be on the same device")

        if not _batch_outputs_are_disjoint(
            [_logical_storage_span(tensor, size) for tensor in x_tensors],
            z_tensors,
            size,
        ):
            raise ValueError("CUDA batch correlation buffers overlap")

        x_stride = _find_uniform_stride(x_tensors, size)
        z_stride = _find_uniform_stride(z_tensors, size)
        if z_stride is None:
            raise ValueError("CUDA batch correlation output tensors are not uniformly strided")

        if x_stride is not None:
            # Construct zero-copy 2D views over the underlying storage.
            # Strong Tensor owners keep all addresses and buffers alive.
            self._packed_x = x_tensors[0].as_strided(
                size=(num_vectors, size),
                stride=(x_stride, 1),
            )
            self._is_stacked_x = False
        elif num_vectors <= 1024:
            # Pre-allocate contiguous 2D view for batch sizes up to B=1024
            self._packed_x = torch.stack(
                [t[:size] for t in x_tensors], dim=0
            ).contiguous()
            self._is_stacked_x = True
        else:
            raise ValueError("CUDA batch correlation tensors are not uniformly strided")

        self._packed_z = z_tensors[0].as_strided(
            size=(num_vectors, size),
            stride=(z_stride, 1),
        )
        self._conj_packed_x = torch.conj(self._packed_x)
        self._pid = os.getpid()
        self._thread_id = threading.get_ident()
        self._size = size
        self._num_vectors = num_vectors
        self._device = device
        self._x_arrays = x_arrays
        self._z_arrays = z_arrays
        self._x_tensors = x_tensors
        self._z_tensors = z_tensors
        self._x_pointers = tuple(t.data_ptr() for t in x_tensors)
        self._z_spans = tuple(
            _logical_storage_span(tensor, size) for tensor in z_tensors
        )
        self._x_stride = x_stride
        self._z_stride = z_stride
        self._epoch = getattr(batch, "_epoch", 0)

    def can_execute(self, batch):
        try:
            if (
                os.getpid() != self._pid
                or threading.get_ident() != self._thread_id
                or int(batch.size) != self._size
                or int(batch.num_vectors) != self._num_vectors
                or getattr(batch, "_epoch", 0) != self._epoch
                or not _CPUNativeBatchCorrelationState._same_array_tensors(
                    batch.xs,
                    self._x_arrays,
                    self._x_tensors,
                    self._x_pointers,
                )
                or not _CPUNativeBatchCorrelationState._same_array_tensors(
                    batch.zs,
                    self._z_arrays,
                    self._z_tensors,
                    self._z_pointers,
                )
            ):
                return False
            tensors = (*self._x_tensors, *self._z_tensors)
            if not all(_cuda_batch_tensor_contract(t, self._size) for t in tensors):
                return False
            if any(t.device != self._device for t in tensors):
                return False
            if not _batch_outputs_are_disjoint(
                [_logical_storage_span(t, self._size) for t in self._x_tensors],
                self._z_tensors,
                self._size,
            ):
                return False
            if not self._is_stacked_x and _find_uniform_stride(self._x_tensors, self._size) != self._x_stride:
                return False
            if _find_uniform_stride(self._z_tensors, self._size) != self._z_stride:
                return False
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def execute(self, batch, y):
        try:
            if (
                os.getpid() != self._pid
                or threading.get_ident() != self._thread_id
                or int(batch.size) != self._size
                or int(batch.num_vectors) != self._num_vectors
                or getattr(batch, "_epoch", 0) != self._epoch
                or not _CPUNativeBatchCorrelationState._same_array_tensors(
                    batch.xs,
                    self._x_arrays,
                    self._x_tensors,
                    self._x_pointers,
                )
                or not _CPUNativeBatchCorrelationState._same_array_tensors(
                    batch.zs,
                    self._z_arrays,
                    self._z_tensors,
                    self._z_pointers,
                )
                or _has_dynamic_autograd_state(self._x_tensors)
                or _has_dynamic_autograd_state(self._z_tensors)
            ):
                return False
            y_tensor = getattr(getattr(y, "_data", None), "tensor", None)
            if (
                y_tensor is None
                or not _cuda_batch_tensor_contract(y_tensor, self._size)
                or y_tensor.device != self._device
            ):
                return False

            y_span = _logical_storage_span(y_tensor, self._size)
            if any(_spans_overlap(y_span, z_span) for z_span in self._z_spans):
                return False

            y_sub = (
                y_tensor
                if y_tensor.numel() == self._size
                else y_tensor[: self._size]
            )
            torch.mul(
                self._conj_packed_x,
                y_sub,
                out=self._packed_z,
            )
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False


def _try_cuda_native_batch_correlate(batch, y):
    """Execute the gated CUDA native batch route, or request Torch fallback."""
    state = getattr(batch, "_torch_cuda_native_batch_state", None)
    if state is not None:
        return state.execute(batch, y)
    try:
        state = _CUDANativeBatchCorrelationState(batch)
    except (
        AttributeError,
        ImportError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    batch._torch_cuda_native_batch_state = state
    return state.execute(batch, y)


def standard_peak_tensor(values):
    """Peak extraction matching PyCBC legacy scan semantics.

    Computes squared magnitudes and extracts peak indices and values on device.
    """
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must be a non-empty 2D tensor")
    device = values.device
    if device.type in ("cpu", "mps") or values.dtype in (torch.complex64, torch.float32):
        if values.is_complex():
            sq_mag = torch.view_as_real(values).square().sum(dim=-1)
        else:
            sq_mag = values.square()
    else:
        if values.is_complex():
            sq_mag = (
                torch.view_as_real(values)
                .to(torch.float64)
                .square()
                .sum(dim=-1)
            )
        else:
            sq_mag = values.to(torch.float64).square()
    indices = torch.argmax(sq_mag, dim=-1)
    gather_indices = indices.unsqueeze(1)
    if values.is_complex():
        peaks = torch.complex(
            values.real.gather(1, gather_indices),
            values.imag.gather(1, gather_indices),
        ).squeeze(1)
    else:
        peaks = values.gather(1, gather_indices).squeeze(1)
    return indices, peaks


def _torch_batch_peak_and_threshold_gpu(
    values, norms, snr_threshold, snr_abort_threshold=None
):
    """Compute squared magnitude and compare against snr_threshold on device.

    Parameters
    ----------
    values : torch.Tensor
        2D tensor of shape (template_count, segment_len) on CUDA/MPS.
    norms : torch.Tensor or numpy.ndarray
        1D array or tensor of length template_count containing template SNR
        normalizations.
    snr_threshold : float
        SNR threshold to record triggers.
    snr_abort_threshold : float, optional
        SNR threshold above which to abort processing.

    Returns
    -------
    survivor_indices : numpy.ndarray (int64)
        Indices of templates that crossed snr_threshold (empty if none).
    peak_indices : numpy.ndarray (int64)
        Peak index within the segment for each survivor template (empty
        if none).
    peak_values : numpy.ndarray (complex64 or float32)
        Complex/real peak value for each survivor template (empty if none).
    aborted : bool
        True if snr_abort_threshold was exceeded by any template, else False.
    """
    if values.ndim != 2 or values.shape[1] == 0 or values.shape[0] == 0:
        dtype = np.complex64 if values.is_complex() else np.float32
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=dtype),
            False,
        )

    device = values.device
    float_dtype = torch.float32 if device.type in ("mps", "cpu") else torch.float64
    if isinstance(norms, torch.Tensor):
        norms_t = norms.to(device=device, dtype=float_dtype)
    else:
        norms_t = torch.as_tensor(norms, device=device, dtype=float_dtype)

    if values.is_complex():
        if device.type in ("mps", "cpu") and values.dtype == torch.complex64:
            sq_mag = torch.view_as_real(values).square().sum(dim=-1)
        else:
            sq_mag = (
                torch.view_as_real(values)
                .to(float_dtype)
                .square()
                .sum(dim=-1)
            )
    else:
        if device.type in ("mps", "cpu") and values.dtype == torch.float32:
            sq_mag = values.square()
        else:
            sq_mag = values.to(float_dtype).square()

    clean_mag = torch.nan_to_num(sq_mag, nan=0.0)
    max_sq_mag, indices = torch.max(clean_mag, dim=-1)

    max_snr_sq = max_sq_mag * norms_t.square()

    dtype = np.complex64 if values.is_complex() else np.float32

    # Single-reduction check on device: evaluates global maximum SNR to determine
    # threshold crossing and abort conditions without redundant PCIe sync round-trips
    max_val_t = torch.max(max_snr_sq)
    max_val = max_val_t.item()

    thresh_sq = float(snr_threshold) ** 2
    if max_val < thresh_sq:
        # Common fast path in >99% of search data blocks: no triggers and no abort
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=dtype),
            False,
        )

    if snr_abort_threshold is not None:
        abort_thresh_sq = float(snr_abort_threshold) ** 2
        if max_val > abort_thresh_sq:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=dtype),
                True,
            )

    crossing_mask = max_snr_sq >= thresh_sq
    survivor_indices_gpu = torch.nonzero(crossing_mask, as_tuple=True)[0]
    surv_indices_within_seg = indices[survivor_indices_gpu]
    surv_values = values[survivor_indices_gpu]
    g_idx = surv_indices_within_seg.unsqueeze(1)

    if values.is_complex():
        surv_peaks = torch.complex(
            surv_values.real.gather(1, g_idx),
            surv_values.imag.gather(1, g_idx),
        ).squeeze(1)
    else:
        surv_peaks = surv_values.gather(1, g_idx).squeeze(1)

    return (
        survivor_indices_gpu.detach().cpu().numpy(),
        surv_indices_within_seg.detach().cpu().numpy(),
        surv_peaks.detach().cpu().numpy(),
        False,
    )


def correlate(x, y, z):
    """Elementwise z = conj(x) * y."""
    torch.mul(
        torch.conj(x._data.tensor), y._data.tensor, out=z._data.tensor
    )


class TorchCorrelator(_BaseCorrelator):
    def __init__(self, x, y, z):
        self.x = x._data.tensor
        self.y = y._data.tensor
        self.z = z._data.tensor
        self._conjugate_source = None
        self._conjugated_x = None
        self._cpu_native_static = _cpu_native_static_eligible(
            self.x, self.y, self.z
        )
        self._cpu_native = None
        if self._cpu_native_static:
            from . import matchedfilter_cpu

            openmp_runtime = _cpu_native_openmp_runtime(matchedfilter_cpu)
            if _cpu_native_dispatch_is_beneficial(
                self.x.numel(), openmp_runtime
            ):
                self._setup_cpu_native(matchedfilter_cpu, openmp_runtime)
                # A reusable correlator fixes its buffers and is normally
                # called many times.  Bind the selected implementation once
                # so the one-thread Torch path has no per-call dispatch cost.
                self.correlate = self._correlate_cpu_native
            else:
                # ``torch.conj`` is a lazy metadata view for this tightly
                # guarded, non-AD buffer.  Reuse it across the repeated
                # Torch route instead of allocating the same view each call.
                self._conjugate_source = self.x
                self._conjugated_x = torch.conj(self.x)

    def _setup_cpu_native(self, matchedfilter_cpu, openmp_runtime):
        """Cache zero-copy ABI views after selecting the native route."""
        self._cpu_native = (
            matchedfilter_cpu._correlate,
            self.x.detach().numpy(),
            self.y.detach().numpy(),
            self.z.detach().numpy(),
            openmp_runtime,
        )

    def correlate(self):
        """Execute the Torch route, reusing its safe lazy conjugate view."""
        conjugated_x = self._conjugated_x
        if conjugated_x is None or self.x is not self._conjugate_source:
            # Attribute replacement is outside the reusable-correlator
            # contract, but retaining a fresh view here preserves the prior
            # fallback semantics for dynamic AD and defensive callers.
            conjugated_x = torch.conj(self.x)
        torch.mul(conjugated_x, self.y, out=self.z)

    _correlate_torch = correlate

    def _correlate_cpu_native(self):
        """Execute the selected native route, rechecking mutable state."""
        function, x_view, y_view, z_view, _ = self._cpu_native
        if _has_dynamic_autograd_state((self.x, self.y, self.z)):
            return self._correlate_torch()
        function(x_view, y_view, z_view)
        torch.autograd.graph.increment_version(self.z)


def _correlate_factory(x, y, z):
    return TorchCorrelator


def batch_correlate_execute(self, y):
    """Vectorised batch correlation for BatchCorrelator."""
    if _environment_flag(_CPU_NATIVE_BATCH_GATE, default=False):
        if _try_cpu_native_batch_correlate(self, y):
            return self
    if _environment_flag(_CUDA_NATIVE_BATCH_GATE, default=False):
        if _try_cuda_native_batch_correlate(self, y):
            return self
    device = getattr(getattr(self.xs[0], "_data", None), "tensor", None)
    device = device.device if device is not None else None
    if hasattr(y, "_data") and hasattr(y._data, "tensor"):
        y_t = y._data.tensor
    else:
        y_raw = y._data if hasattr(y, "_data") else y
        y_t = torch.as_tensor(y_raw, device=device)
    size = getattr(self, "size", y_t.numel())
    y_sub = y_t if y_t.numel() == size else y_t[:size]

    num_vectors = len(self.xs)
    if num_vectors > 0 and len(self.zs) == num_vectors:
        try:
            x_tensors = tuple(
                getattr(getattr(x, "_data", None), "tensor", None)
                for x in self.xs
            )
            z_tensors = tuple(
                getattr(getattr(z, "_data", None), "tensor", None)
                for z in self.zs
            )
            all_tensors = (*x_tensors, *z_tensors)
            if all(isinstance(t, torch.Tensor) for t in all_tensors):
                target_device = y_sub.device
                if (
                    all(
                        t.device == target_device
                        and t.dtype == y_sub.dtype
                        and t.layout == torch.strided
                        and not t.is_conj()
                        and not t.is_neg()
                        for t in all_tensors
                    )
                ):
                    x_stride = _find_uniform_stride(x_tensors, size)
                    z_stride = _find_uniform_stride(z_tensors, size)
                    epoch = getattr(self, "_epoch", 0)
                    cached_conj_x = getattr(self, "_cached_conj_packed_x", None)
                    cached_epoch = getattr(self, "_cached_conj_packed_x_epoch", -1)

                    if cached_conj_x is None or cached_epoch != epoch:
                        if x_stride is not None:
                            packed_x = x_tensors[0].as_strided(
                                size=(num_vectors, size),
                                stride=(x_stride, 1),
                            )
                            cached_conj_x = torch.conj(packed_x)
                        elif num_vectors <= 1024:
                            stacked_x = torch.stack(
                                [t[:size] for t in x_tensors], dim=0
                            ).contiguous()
                            cached_conj_x = torch.conj(stacked_x)
                        else:
                            cached_conj_x = None
                        self._cached_conj_packed_x = cached_conj_x
                        self._cached_conj_packed_x_epoch = epoch

                    if cached_conj_x is not None and z_stride is not None:
                        packed_z = z_tensors[0].as_strided(
                            size=(num_vectors, size),
                            stride=(z_stride, 1),
                        )
                        torch.mul(
                            cached_conj_x,
                            y_sub,
                            out=packed_z,
                        )
                        return self
                    elif cached_conj_x is not None and num_vectors <= 1024:
                        out_2d = torch.mul(cached_conj_x, y_sub)
                        for i, z_t in enumerate(z_tensors):
                            z_t[:size].copy_(out_2d[i])
                        return self
        except Exception:
            pass

    for x, z in zip(self.xs, self.zs):
        if hasattr(x._data, "tensor"):
            x_t = x._data.tensor
        else:
            x_t = torch.as_tensor(x._data, device=device)
        if hasattr(z._data, "tensor"):
            z_t = z._data.tensor
        else:
            z_t = torch.as_tensor(z._data, device=device)
        z_out = z_t if z_t.numel() == size else z_t[:size]
        torch.mul(
            torch.conj(x_t[:size]),
            y_sub,
            out=z_out,
        )
    return self
