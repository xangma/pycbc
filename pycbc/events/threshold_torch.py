# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch backend for thresholding and clustering event triggers.
The implementations mirror the CUDA/CuPy backends but use Torch tensor
operations and keep scheme plumbing scheme‑agnostic.
"""

from __future__ import division

from functools import lru_cache
import os

import numpy as np
import torch
import torch.nn.functional as F

import pycbc.scheme as _scheme
from pycbc import opt
from pycbc.types import Array
from pycbc.types.array_torch import TorchArrayData

from .eventmgr import _BaseThresholdCluster
from .simd_threshold_cython import parallel_thresh_cluster

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_symm_block_reduction_kernel(
        in_ptr,
        block_max_ptr,
        block_idx_ptr,
        N,
        window,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * window
        if block_start >= N:
            return

        block_end = block_start + window
        if block_end > N:
            block_end = N

        curr_max_val = -1.0
        curr_max_idx = block_start

        for offset in range(block_start, block_end, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < block_end
            re = tl.load(in_ptr + 2 * cols, mask=mask, other=0.0)
            im = tl.load(in_ptr + 2 * cols + 1, mask=mask, other=0.0)
            mag_sq = tl.where(mask, re * re + im * im, -1.0)

            tile_max = tl.max(mag_sq, axis=0)
            tile_argmax = tl.argmax(mag_sq, axis=0)
            tile_idx = offset + tile_argmax

            if tile_max > curr_max_val:
                curr_max_val = tile_max
                curr_max_idx = tile_idx

        tl.store(block_max_ptr + pid, curr_max_val)
        tl.store(block_idx_ptr + pid, curr_max_idx)

    def _triton_symmetric_block_reduce(tensor, window, out_max=None, out_idx=None):
        """Run Triton fused 1D block reduction on CUDA complex64 tensor."""
        slen = tensor.numel()
        nb = (slen + window - 1) // window
        if out_max is None or out_max.numel() != nb:
            block_max = torch.empty(nb, device=tensor.device, dtype=torch.float32)
        else:
            block_max = out_max

        if out_idx is None or out_idx.numel() != nb:
            block_idx = torch.empty(nb, device=tensor.device, dtype=torch.int64)
        else:
            block_idx = out_idx

        block_size = min(triton.next_power_of_2(window), 1024)
        if block_size < 32:
            block_size = 32

        in_real = tensor.view(torch.float32)
        grid = (nb,)
        _triton_symm_block_reduction_kernel[grid](
            in_real,
            block_max,
            block_idx,
            slen,
            window,
            BLOCK_SIZE=block_size,
        )
        return block_max, block_idx

    @triton.jit
    def _triton_symm_mask_tensor_kernel(
        max_vals_ptr,
        thresh_sq_ptr,
        out_keep_ptr,
        nb,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < nb

        thresh_sq = tl.load(thresh_sq_ptr)
        curr = tl.load(max_vals_ptr + offsets, mask=mask, other=-1e30)
        thresh_ok = curr > thresh_sq

        # Left neighbor
        left_offsets = offsets - 1
        left_mask = mask & (offsets > 0)
        left_val = tl.load(
            max_vals_ptr + left_offsets, mask=left_mask, other=-1e30
        )
        left_ok = tl.where(offsets > 0, curr > left_val, True)

        # Right neighbor
        right_offsets = offsets + 1
        right_mask = mask & (offsets < nb - 1)
        right_val = tl.load(
            max_vals_ptr + right_offsets, mask=right_mask, other=-1e30
        )
        right_ok = tl.where(offsets < nb - 1, curr >= right_val, True)

        # First element special: curr[0] > curr[1] when nb > 1
        first_special = tl.where(
            (offsets == 0) & (nb > 1), curr > right_val, True
        )

        keep = thresh_ok & left_ok & right_ok & first_special
        tl.store(out_keep_ptr + offsets, keep.to(tl.int1), mask=mask)

    @triton.jit
    def _triton_symm_mask_scalar_kernel(
        max_vals_ptr,
        thresh_sq,
        out_keep_ptr,
        nb,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < nb

        curr = tl.load(max_vals_ptr + offsets, mask=mask, other=-1e30)
        thresh_ok = curr > thresh_sq

        # Left neighbor
        left_offsets = offsets - 1
        left_mask = mask & (offsets > 0)
        left_val = tl.load(
            max_vals_ptr + left_offsets, mask=left_mask, other=-1e30
        )
        left_ok = tl.where(offsets > 0, curr > left_val, True)

        # Right neighbor
        right_offsets = offsets + 1
        right_mask = mask & (offsets < nb - 1)
        right_val = tl.load(
            max_vals_ptr + right_offsets, mask=right_mask, other=-1e30
        )
        right_ok = tl.where(offsets < nb - 1, curr >= right_val, True)

        # First element special: curr[0] > curr[1] when nb > 1
        first_special = tl.where(
            (offsets == 0) & (nb > 1), curr > right_val, True
        )

        keep = thresh_ok & left_ok & right_ok & first_special
        tl.store(out_keep_ptr + offsets, keep.to(tl.int1), mask=mask)

    def _triton_symmetric_cluster_mask(max_vals, threshold_sq, out=None):
        """Run Triton fused symmetric neighbor clustering mask on CUDA tensor."""
        nb = max_vals.numel()
        if nb == 0:
            return torch.empty(0, device=max_vals.device, dtype=torch.bool)
        if out is None or out.numel() != nb:
            out = torch.empty(nb, device=max_vals.device, dtype=torch.bool)
        BLOCK_SIZE = 512
        grid = ((nb + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        if isinstance(threshold_sq, torch.Tensor) and threshold_sq.is_cuda:
            _triton_symm_mask_tensor_kernel[grid](
                max_vals,
                threshold_sq,
                out,
                nb,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _triton_symm_mask_scalar_kernel[grid](
                max_vals,
                float(threshold_sq),
                out,
                nb,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return out


_l2_size = opt.get_l2_cache_size() if hasattr(opt, "get_l2_cache_size") else getattr(opt, "LEVEL2_CACHE_SIZE", None)
if _l2_size is not None:
    _CPU_NATIVE_SEGSIZE = _l2_size // np.dtype(np.complex64).itemsize
else:
    _CPU_NATIVE_SEGSIZE = 32768

_CPU_NATIVE_INT_MAX = np.iinfo(np.int32).max
_CPU_NATIVE_VALUE_DTYPE = np.dtype(np.complex64)
_CPU_NATIVE_INDEX_DTYPE = np.dtype(np.int64)
_TRUSTED_NATIVE_RESULT_ENV = (
    "PYCBC_TORCH_CPU_THRESHOLD_TRUSTED_ARRAYS"
)

_THRESHOLD_COMPILE_CACHE_SIZE = 16
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_TORCH_IS_INFERENCE = getattr(torch, "is_inference", None)
_TORCH_INFERENCE_MODE_ENABLED = getattr(
    torch, "is_inference_mode_enabled", lambda: False
)


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


def _threshold_compile_requested():
    """Whether the experimental threshold compiler route is enabled."""
    if not _environment_flag("PYCBC_TORCH_COMPILE", default=False):
        return False
    return _environment_flag(
        "PYCBC_TORCH_COMPILE_THRESHOLD", default=True
    )


def _threshold_compile_configuration():
    """Return the explicitly attributable compiler configuration."""
    backend = os.environ.get(
        "PYCBC_TORCH_COMPILE_BACKEND", "inductor"
    ).strip()
    mode = os.environ.get("PYCBC_TORCH_COMPILE_MODE", "default").strip()
    if not backend:
        raise ValueError("PYCBC_TORCH_COMPILE_BACKEND must not be empty")
    if not mode:
        raise ValueError("PYCBC_TORCH_COMPILE_MODE must not be empty")
    return backend, mode


def _native_float32_scalar(value):
    """Return a native scalar, or ``None`` for tensor/array-like inputs."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return np.float32(value)


def _has_forward_ad_state(tensor):
    """Whether exporting ``tensor`` would silently discard a tangent."""
    current_level = getattr(torch.autograd.forward_ad, "_current_level", None)
    if current_level == -1:
        return False
    try:
        return (
            torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
        )
    except (AttributeError, RuntimeError):
        return True


def _source_tensor(series):
    """Return the tensor backing a PyCBC or raw Torch series.

    Accepts pycbc Array/TimeSeries/FrequencySeries with TorchArrayData,
    a raw torch.Tensor, or anything convertible via torch.as_tensor.
    """
    # ``torch.Tensor.data`` is detached from autograd, so handle a raw tensor
    # before consulting the array-like ``data`` attributes.
    if isinstance(series, torch.Tensor):
        return series
    data = getattr(series, "_data", getattr(series, "data", series))
    if hasattr(data, "tensor"):
        return data.tensor
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data)


def _as_tensor(series):
    """Return a 1D torch tensor view of the input series data."""
    tensor = _source_tensor(series)
    # The production search workspace is already one-dimensional.  Returning
    # it directly avoids constructing a redundant Torch view for every
    # template while retaining reshape semantics for scalar and N-D inputs.
    return tensor if tensor.ndim == 1 else tensor.reshape(-1)


def _threshold_mask(tensor, threshold):
    """Boolean mask of samples whose magnitude exceeds threshold."""
    thresh_sq = torch.as_tensor(threshold, device=tensor.device,
                                dtype=tensor.real.dtype)
    thresh_sq = thresh_sq * thresh_sq
    mag_sq = _magnitude_squared(tensor)
    return mag_sq > thresh_sq, mag_sq


def _magnitude_squared(tensor, out=None, component_out=None):
    """Return squared magnitudes, optionally reusing Torch CPU buffers."""
    if tensor.device.type == "cpu":
        if tensor.is_complex():
            # Preserve the real**2 + imag**2 evaluation order while reusing
            # the real-square allocation for the sum.  Search clustering
            # calls this once per template, so avoiding a third full-series
            # temporary matters on Torch CPU.
            if out is None and component_out is None:
                return torch.view_as_real(tensor).square().sum(dim=-1)
            if out is None:
                magnitude = tensor.real.square()
            else:
                torch.square(tensor.real, out=out)
                magnitude = out
            if component_out is None:
                magnitude.add_(tensor.imag.square())
            else:
                torch.square(tensor.imag, out=component_out)
                magnitude.add_(component_out)
            return magnitude
        if out is None:
            return tensor.square()
        return torch.square(tensor, out=out)
    if tensor.is_complex():
        return torch.view_as_real(tensor).square().sum(dim=-1)
    return tensor.square()


def _array_from_tensor(tensor):
    """Wrap a result tensor as a PyCBC array without leaving its device."""
    return Array(TorchArrayData(tensor), copy=False)


def threshold(series, value):
    """Return device-backed locations and values above the threshold."""
    tensor = _as_tensor(series)
    mask, _ = _threshold_mask(tensor, value)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    vals = tensor[mask]
    return _array_from_tensor(idx), _array_from_tensor(vals)


def threshold_only(series, value):
    """Alias for threshold; kept for API parity."""
    return threshold(series, value)


def _findchirp_cluster_indices(times, values, window_length):
    """Return FindChirp survivor positions without copying candidates.

    ``times`` must be sorted. Vectorized sliding-window maximum filtering
    via ``torch.nn.functional.max_pool1d`` extracts forward-window candidate
    maxima on-device, preserving FindChirp's earlier tie-breaking rule,
    strict window bounds, and NaN handling.
    """
    assert window_length > 0, "Clustering window length is not positive"
    window_length = int(window_length)
    count = times.numel()
    if count == 0:
        return torch.empty(0, device=times.device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=times.device, dtype=torch.long)

    if values.is_complex():
        magnitudes = torch.view_as_real(values).square().sum(dim=-1)
    else:
        magnitudes = torch.abs(values)

    clean_mags = torch.nan_to_num(
        magnitudes,
        nan=float("-inf"),
        posinf=float("inf"),
        neginf=float("-inf"),
    )

    t0 = times[0]
    span = int((times[-1] - t0).item()) + 1

    offset_indices = (times - t0).long()
    if span <= max(2 * count, 10_000_000):
        dense_mag = torch.full(
            (span,),
            float("-inf"),
            device=times.device,
            dtype=clean_mags.dtype,
        )
        dense_mag.scatter_(0, offset_indices, clean_mags)
        padded = F.pad(
            dense_mag.view(1, 1, -1), (0, window_length), value=float("-inf")
        )
        pooled = F.max_pool1d(
            padded, kernel_size=window_length + 1, stride=1
        ).view(-1)
        fwd_max = pooled[offset_indices]
    else:
        # Memory-safe chunked evaluation for sparse spans > 10M samples
        right = torch.searchsorted(times, times + window_length, right=True)
        fwd_max = torch.empty(count, device=times.device, dtype=clean_mags.dtype)
        chunk_size = 2048
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            c_right = right[start:end]
            c_starts = torch.arange(start, end, device=times.device)[:, None]
            max_span = int((c_right - c_starts.squeeze(1)).max().item())
            if max_span <= 0:
                fwd_max[start:end] = clean_mags[start:end]
                continue
            offsets = torch.arange(max_span, device=times.device)[None, :]
            c_indices = torch.clamp(c_starts + offsets, max=count - 1)
            valid = (c_indices >= c_starts) & (c_indices < c_right[:, None])
            fwd_max[start:end] = torch.where(
                valid, clean_mags[c_indices], float("-inf")
            ).max(dim=1).values

    is_fwd_max = (clean_mags >= fwd_max)
    fwd_indices = torch.nonzero(is_fwd_max, as_tuple=False).flatten()
    num_fwd = fwd_indices.numel()
    if num_fwd == 0:
        return torch.empty(0, device=times.device, dtype=torch.long)
    if num_fwd == 1:
        return fwd_indices

    fwd_times = times[fwd_indices]

    # Vectorized searchsorted: evaluate all successors in 1 parallel kernel launch
    target_t = fwd_times + window_length
    next_idx = torch.searchsorted(fwd_times, target_t, right=True)
    next_idx_cpu = next_idx.detach().cpu().numpy()

    survivor_positions = []
    curr_idx = 0
    while curr_idx < num_fwd:
        survivor_positions.append(curr_idx)
        curr_idx = int(next_idx_cpu[curr_idx])

    if len(survivor_positions) == 0:
        return torch.empty(0, device=times.device, dtype=torch.long)
    survivor_tensor = torch.as_tensor(
        survivor_positions,
        device=fwd_indices.device,
        dtype=torch.long,
    )
    return fwd_indices[survivor_tensor]


def _findchirp_cluster_positions(times, values, window_length):
    """Co-locate candidate arrays and return survivor positions on-device."""
    time_tensor = _as_tensor(times)
    value_tensor = _as_tensor(values)
    if value_tensor.device.type != "cpu":
        device = value_tensor.device
    else:
        device = time_tensor.device

    time_tensor = time_tensor.to(device=device)
    value_tensor = value_tensor.to(device=device)

    # Fast-path for CPU: delegate directly to O(K) Cython single-pass engine
    if device.type == "cpu":
        count = time_tensor.numel()
        if count == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if count == 1:
            return torch.zeros(1, device=device, dtype=torch.long)
        times_np = time_tensor.numpy().astype(np.int32, copy=False)
        if value_tensor.is_complex():
            mags_np = np.abs(value_tensor.numpy())
        else:
            mags_np = np.abs(value_tensor.numpy()).astype(np.float32, copy=False)
        indices_np = np.zeros(count, dtype=np.int32)
        from .eventmgr_cython import findchirp_cluster_over_window_cython
        k = findchirp_cluster_over_window_cython(times_np, mags_np, int(window_length), indices_np, count)
        return torch.from_numpy(indices_np[:k+1].astype(np.int64))

    return _findchirp_cluster_indices(
        time_tensor, value_tensor, window_length
    )


def _threshold_and_cluster_findchirp(series, value, window, real=False):
    """Threshold and FindChirp-cluster before transferring survivors."""
    tensor = _as_tensor(series)
    if real:
        mask = tensor > value
    else:
        mask, _ = _threshold_mask(tensor, value)

    candidate_times = torch.nonzero(mask, as_tuple=False).flatten()
    candidate_values = tensor[candidate_times]
    survivor_positions = _findchirp_cluster_indices(
        candidate_times, candidate_values, window
    )
    survivor_times = candidate_times[survivor_positions]
    survivor_values = candidate_values[survivor_positions]
    return (
        survivor_times.cpu().numpy(),
        survivor_values.cpu().numpy(),
    )


def threshold_and_cluster_findchirp(series, value, window):
    """Threshold complex data and copy only clustered triggers to the host."""
    return _threshold_and_cluster_findchirp(series, value, window)


def threshold_real_and_cluster_findchirp(series, value, window):
    """Threshold real data and copy only clustered triggers to the host."""
    return _threshold_and_cluster_findchirp(series, value, window, real=True)


def _cluster_candidates(mag_sq, window):
    """Return per-window maxima and indices (before neighbor clustering)."""
    length = mag_sq.numel()
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")

    num_blocks = (length + window - 1) // window
    if num_blocks == 0:
        return (torch.empty(0, device=mag_sq.device, dtype=mag_sq.dtype),
                torch.empty(0, device=mag_sq.device, dtype=torch.long))

    pad = num_blocks * window - length
    if pad:
        pad_val = torch.full((pad,), float("-inf"), device=mag_sq.device,
                             dtype=mag_sq.dtype)
        mag_padded = torch.cat((mag_sq, pad_val))
    else:
        mag_padded = mag_sq

    mag_blocks = mag_padded.view(num_blocks, window)
    # ``torch.max`` propagates NaN, whereas the legacy clustering path ignores
    # an invalid candidate and can still retain a finite maximum from the same
    # block. Accelerators use a fixed-shape masked reduction to avoid the host
    # synchronization inherent in boolean advanced indexing. CPU first takes
    # its ordinary finite-data max and allocates repair rows only when needed.
    if mag_sq.device.type != "cpu":
        finite_blocks = torch.nan_to_num(mag_blocks, nan=float("-inf"))
        block_max, block_idx = torch.max(finite_blocks, dim=-1)
    else:
        block_max, block_idx = torch.max(mag_blocks, dim=-1)
        nan_blocks = torch.isnan(block_max)
        if bool(nan_blocks.any()):
            nan_rows = mag_blocks[nan_blocks]
            nan_rows = torch.nan_to_num(nan_rows, nan=float("-inf"))
            repaired_max, repaired_idx = torch.max(nan_rows, dim=-1)
            block_max[nan_blocks] = repaired_max
            block_idx[nan_blocks] = repaired_idx
    # ``block_idx`` is an otherwise-discarded relative-index result.  Promote
    # it to global indices in place instead of allocating product and sum
    # temporaries for every template.
    block_idx.add_(
        torch.arange(num_blocks, device=mag_sq.device), alpha=window
    )
    return block_max, block_idx


def _fixed_shape_threshold_core(tensor, threshold_sq, window):
    """Return fixed-shape block candidates and their survivor mask."""
    mag_sq = _magnitude_squared(tensor)
    block_max, block_idx = _cluster_candidates(mag_sq, window)
    keep = _symmetric_cluster_mask(block_max, threshold_sq)
    return block_max, block_idx, keep


def _plain_strided_compile_tensor(tensor):
    """Whether compiler dispatch may consume ``tensor`` directly."""
    return (
        type(tensor) is torch.Tensor
        and tensor.layout == torch.strided
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(tensor)
        )
    )


def _threshold_compile_eligible(tensor, threshold_sq, *, single_series):
    """Whether a production search tensor may enter the compiled CUDA core."""
    return (
        _threshold_compile_requested()
        and single_series
        # Compiled execution must not bypass Tensor-subclass dispatch or
        # inference-tensor semantics.  Both inputs are ordinary tensors in
        # the qualified production call, so unusual tensors retain eager
        # execution.
        and _plain_strided_compile_tensor(tensor)
        and _plain_strided_compile_tensor(threshold_sq)
        and tensor.ndim == 1
        and tensor.numel() > 0
        and tensor.device.type == "cuda"
        and tensor.dtype == torch.complex64
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and not tensor.requires_grad
        and not _has_forward_ad_state(tensor)
        and threshold_sq.ndim == 0
        and threshold_sq.device == tensor.device
        and threshold_sq.dtype == tensor.real.dtype
        and not threshold_sq.requires_grad
        and not _has_forward_ad_state(threshold_sq)
    )


@lru_cache(maxsize=_THRESHOLD_COMPILE_CACHE_SIZE)
def _compiled_threshold_core(
    length, window, device_type, device_index, dtype, backend, mode
):
    """Compile and retain one exact fixed-shape threshold specialization."""
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        raise RuntimeError(
            "PYCBC_TORCH_COMPILE_THRESHOLD requires torch.compile"
        )

    # Shape, device, and dtype participate in this bounded cache key even
    # though Dynamo also guards them. Keeping one callable per exact search
    # geometry prevents PyCBC from accidentally routing a different workload
    # through an existing specialization.
    def fixed_shape_core(tensor, threshold_sq):
        return _fixed_shape_threshold_core(tensor, threshold_sq, window)

    return compiler(
        fixed_shape_core,
        backend=backend,
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )


def _tensor_bitwise_equal(left, right):
    """Return true only when tensor metadata and element bits are identical."""
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    left_bytes = left.contiguous().view(torch.uint8)
    right_bytes = right.contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _run_eligible_compiled_threshold_core(tensor, threshold_sq, window):
    """Run the compiled core after its production gate has accepted a call."""
    backend, mode = _threshold_compile_configuration()
    verify = _environment_flag(
        "PYCBC_TORCH_COMPILE_VERIFY", default=False
    )
    compiled = _compiled_threshold_core(
        tensor.numel(),
        window,
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        backend,
        mode,
    )
    compiled_max, compiled_idx, compiled_keep = compiled(tensor, threshold_sq)

    if verify:
        eager_max, eager_idx, eager_keep = _fixed_shape_threshold_core(
            tensor, threshold_sq, window
        )
        if not _tensor_bitwise_equal(compiled_max, eager_max):
            raise RuntimeError(
                "compiled threshold block maxima differ from eager output"
            )
        if not _tensor_bitwise_equal(compiled_idx, eager_idx):
            raise RuntimeError(
                "compiled threshold block indices differ from eager output"
            )
        if not _tensor_bitwise_equal(compiled_keep, eager_keep):
            raise RuntimeError(
                "compiled threshold survivor mask differs from eager output"
            )

    return compiled_max, compiled_idx, compiled_keep


def _run_threshold_core(tensor, threshold_sq, window, *, single_series):
    """Run eager or explicitly requested compiled fixed-shape processing."""
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    if not _threshold_compile_eligible(
        tensor, threshold_sq, single_series=single_series
    ):
        return _fixed_shape_threshold_core(tensor, threshold_sq, window)
    return _run_eligible_compiled_threshold_core(
        tensor, threshold_sq, window
    )


def _symmetric_cluster_mask(max_vals, threshold_sq, out=None):
    """Return the fixed-shape local-maximum mask before survivor extraction."""
    nb = max_vals.numel()
    if nb == 0:
        return torch.empty(0, device=max_vals.device, dtype=torch.bool)

    if (
        _TRITON_AVAILABLE
        and max_vals.is_cuda
        and max_vals.dtype == torch.float32
        and max_vals.is_contiguous()
    ):
        return _triton_symmetric_cluster_mask(max_vals, threshold_sq, out=out)

    if out is None or out.numel() != nb:
        keep = max_vals > threshold_sq
    else:
        torch.gt(max_vals, threshold_sq, out=out)
        keep = out

    if nb > 1:
        # Match parallel_thresh_cluster exactly: each candidate must be
        # strictly greater than its previous neighbor and greater than or
        # equal to its next neighbor. The first boundary is also strict, so a
        # plateau keeps only its leftmost interior maximum and a two-block tie
        # keeps neither endpoint.
        keep[1:] &= max_vals[1:] > max_vals[:-1]
        keep[:-1] &= max_vals[:-1] >= max_vals[1:]
        keep[0] &= max_vals[0] > max_vals[1]
    return keep


def _symmetric_cluster(max_vals, max_idx, threshold_sq):
    """Apply symmetric neighbor clustering (keep local maxima)."""
    nb = max_vals.numel()
    if nb == 0:
        return (torch.empty(0, device=max_vals.device, dtype=max_vals.dtype),
                torch.empty(0, device=max_idx.device, dtype=max_idx.dtype))

    keep = _symmetric_cluster_mask(max_vals, threshold_sq)
    return max_vals[keep], max_idx[keep]


def threshold_and_cluster(series, threshold, window):
    """Return device-backed clustered values and indices above threshold."""
    source = _source_tensor(series)
    single_series = source.ndim == 1
    tensor = source if single_series else source.reshape(-1)
    # Preserve the public eager path's validation order: the legacy
    # _cluster_candidates call rejected the window before converting the
    # threshold. _run_threshold_core repeats this check after receiving the
    # normalized integer so compiled and direct callers remain fail-closed.
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    thresh_sq = torch.as_tensor(
        threshold, device=tensor.device, dtype=tensor.real.dtype
    )
    thresh_sq = thresh_sq * thresh_sq

    if (
        _TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype == torch.complex64
        and tensor.is_contiguous()
    ):
        block_max, block_idx = _triton_symmetric_block_reduce(tensor, window)
        keep = _symmetric_cluster_mask(block_max, thresh_sq)
    else:
        _, block_idx, keep = _run_threshold_core(
            tensor, thresh_sq, window, single_series=single_series
        )

    kept_idx = block_idx[keep]
    flat_series = tensor.reshape(-1)
    kept_vals = flat_series[kept_idx]
    return _array_from_tensor(kept_vals), _array_from_tensor(kept_idx)


class TorchThresholdCluster(_BaseThresholdCluster):
    """Torch implementation of ThresholdCluster using tensor ops."""
    def __init__(self, series):
        # Retain the source tensor as well as its flat view. Recreating the
        # view per call makes a later ``requires_grad_(True)`` on a bound raw
        # tensor visible to autograd.
        self._source = _source_tensor(series)
        self.series = (
            self._source
            if self._source.ndim == 1
            else self._source.reshape(-1)
        )
        # A contiguous CPU complex64 tensor can be exposed to NumPy without a
        # copy and searched by PyCBC's SIMD/OpenMP clustering kernel.  This is
        # substantially cheaper than dispatching several full-series eager
        # Torch operations for every sequential template.  Keep reusable
        # native output buffers here; returned Torch arrays are copied only at
        # the sparse-survivor boundary so they remain stable across calls.
        self._native_values = None
        self._native_indices = None
        # Creating a NumPy ndarray wrapper for unchanged Torch storage costs a
        # measurable fraction of the Python/native boundary on every template.
        # Retain that zero-copy view while its pointer and length are stable;
        # these guards also make in-place resize/set_ operations refresh it.
        self._native_series = None
        self._native_series_pointer = None
        self._native_series_size = None
        # Like the established CPU/CUDA engines, retain scratch storage across
        # repeated calls bound to the same matched-filter output.  Keep this to
        # CPU, where _magnitude_squared uses real/imag component operations;
        # accelerator backends retain their native complex-abs path.
        native_cpu_candidate = (
            # Direct NumPy/Cython access must not bypass Tensor-subclass
            # dispatch or inference-tensor restrictions.
            type(self.series) is torch.Tensor
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self.series)
            )
            and self.series.device.type == "cpu"
            and self.series.layout == torch.strided
            and self.series.dtype == torch.complex64
            and self.series.is_contiguous()
            and not self.series.is_conj()
            and not self.series.is_neg()
            and not self.series.requires_grad
            and not _has_forward_ad_state(self.series)
            and 0 < self.series.numel() <= _CPU_NATIVE_INT_MAX
        )
        # This opt-in is latched per engine so the sparse-result fast path
        # does not spend its measured constructor saving on an environment
        # lookup for every template.  The trusted construction below remains
        # restricted to exact tensors produced by the closed native CPU call
        # site and to the Torch CPU scheme active when this engine was made.
        trusted_results_requested = _environment_flag(
            _TRUSTED_NATIVE_RESULT_ENV, default=False
        )
        active_scheme = _scheme.mgr.state
        self._trusted_native_result_scheme = None
        if (
            trusted_results_requested
            and native_cpu_candidate
            and type(self.series) is torch.Tensor
            and self.series.layout == torch.strided
            and isinstance(active_scheme, _scheme.TorchScheme)
            and active_scheme.torch_device.type == "cpu"
        ):
            self._trusted_native_result_scheme = active_scheme
        if native_cpu_candidate:
            import multiprocessing
            try:
                nthreads = multiprocessing.cpu_count()
            except (NotImplementedError, AttributeError):
                nthreads = 1
            segsize = int(_CPU_NATIVE_SEGSIZE)
            self._ws_cvals = np.empty(nthreads * segsize, dtype=np.complex64)
            self._ws_norms = np.empty(nthreads * segsize, dtype=np.float32)
            self._ws_mlocs = np.empty(nthreads * (segsize // 2 + 2), dtype=np.int64)
            self._ws_seglens = np.empty(nthreads, dtype=np.int64)
        else:
            self._ws_cvals = None
            self._ws_norms = None
            self._ws_mlocs = None
            self._ws_seglens = None

        self._triton_scratch_nb = 0
        self._triton_block_max = None
        self._triton_block_idx = None
        self._triton_keep = None

        if (
            # Reusable ``out=`` kernels bypass Tensor-subclass dispatch, so
            # retain them only for the exact base strided tensor qualified by
            # this eager implementation.
            type(self.series) is torch.Tensor
            and self.series.layout == torch.strided
            and self.series.device.type == "cpu"
            and not self.series.requires_grad
            and not _has_forward_ad_state(self.series)
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self.series)
            )
            # Scratch created in inference mode cannot be written after the
            # engine leaves that mode. The uncommon cross-mode engine keeps
            # the allocation-free eager fallback instead.
            and not _TORCH_INFERENCE_MODE_ENABLED()
            and not native_cpu_candidate
        ):
            self._magnitude = torch.empty_like(self.series.real)
            self._magnitude_component = (
                torch.empty_like(self._magnitude)
                if self.series.is_complex()
                else None
            )
        else:
            self._magnitude = None
            self._magnitude_component = None

    def _can_use_native_cpu(self):
        """Whether the zero-copy CPU clustering fast path is safe."""
        return (
            type(self.series) is torch.Tensor
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self.series)
            )
            and self.series.device.type == "cpu"
            and self.series.layout == torch.strided
            and self.series.dtype == torch.complex64
            and self.series.is_contiguous()
            and not self.series.is_conj()
            and not self.series.is_neg()
            and not self.series.requires_grad
            and not _has_forward_ad_state(self.series)
            and 0 < self.series.numel() <= _CPU_NATIVE_INT_MAX
        )

    def _native_cpu_threshold_and_cluster(self, threshold, window):
        """Run the established CPU kernel on shared Torch/NumPy storage."""
        series_size = self.series.numel()
        candidate_count = (series_size + window - 1) // window
        if (
            self._native_values is None
            or self._native_values.size < candidate_count
        ):
            self._native_values = np.empty(candidate_count, dtype=np.complex64)
            self._native_indices = np.empty(candidate_count, dtype=np.uint32)

        series_pointer = self.series.data_ptr()
        if (
            self._native_series is None
            or self._native_series_pointer != series_pointer
            or self._native_series_size != series_size
        ):
            self._native_series = self.series.numpy()
            self._native_series_pointer = series_pointer
            self._native_series_size = series_size

        count = parallel_thresh_cluster(
            # Native eligibility excludes reverse- and forward-mode AD as well
            # as lazy conjugate/negative views. The cached NumPy wrapper still
            # aliases current Torch storage and is refreshed after a rebind.
            self._native_series,
            series_size,
            self._native_values,
            self._native_indices,
            threshold,
            window,
            _CPU_NATIVE_SEGSIZE,
            self._ws_cvals,
            self._ws_norms,
            self._ws_mlocs,
            self._ws_seglens,
        )
        # The native buffers are reused on the next call.  Survivors are
        # sparse (at most one per clustering window), so copying only these
        # values is cheap and preserves the existing Torch engine's stable
        # result and int64-index contracts.
        sparse_values = self._native_values[:count].copy()
        values = torch.from_numpy(sparse_values)
        indices = torch.from_numpy(
            self._native_indices[:count].astype(np.int64, copy=True)
        )
        trusted_scheme = self._trusted_native_result_scheme
        if trusted_scheme is not None and _scheme.mgr.state is trusted_scheme:
            # These tensors were just created above from private, fresh NumPy
            # copies with fixed CPU dtypes.  Inline the two lightweight
            # wrappers to bypass only constructor validation already proved by
            # this closed call site.  Do not generalize this pattern to
            # arbitrary tensors: Array and TorchArrayData constructors are the
            # validation boundary everywhere else.
            values_data = object.__new__(TorchArrayData)
            values_data.tensor = values
            values_data.dtype = _CPU_NATIVE_VALUE_DTYPE
            values_result = object.__new__(Array)
            values_result._scheme = trusted_scheme
            values_result._saved = None
            values_result._data = values_data

            indices_data = object.__new__(TorchArrayData)
            indices_data.tensor = indices
            indices_data.dtype = _CPU_NATIVE_INDEX_DTYPE
            indices_result = object.__new__(Array)
            indices_result._scheme = trusted_scheme
            indices_result._saved = None
            indices_result._data = indices_data
            return values_result, indices_result
        return _array_from_tensor(values), _array_from_tensor(indices)

    def threshold_and_cluster(self, threshold, window):
        # A raw tensor may have requires_grad toggled after this engine is
        # constructed.  ``out=`` kernels are incompatible with autograd, so
        # decide whether to reuse scratch from the tensor's current state.
        self.series = (
            self._source
            if self._source.ndim == 1
            else self._source.reshape(-1)
        )
        window = int(window)
        if window <= 0:
            raise ValueError("window must be positive")
        native_threshold = _native_float32_scalar(threshold)
        if (
            self._can_use_native_cpu()
            and window <= _CPU_NATIVE_INT_MAX
            and native_threshold is not None
        ):
            return self._native_cpu_threshold_and_cluster(
                native_threshold, window
            )

        use_scratch = (
            # Re-check the current binding because a reusable engine may see
            # its source acquire special semantics after construction.
            type(self.series) is torch.Tensor
            and self.series.layout == torch.strided
            and self.series.device.type == "cpu"
            and not self.series.requires_grad
            and not _has_forward_ad_state(self.series)
            and self._magnitude is not None
            # ``Tensor.data`` rebinding can change shape, dtype, or real versus
            # complex storage without replacing the source object.  Never let
            # stale ``out=`` buffers narrow, resize, or change the eager
            # magnitude calculation in that case.
            and self._magnitude.shape == self.series.shape
            and self._magnitude.device == self.series.device
            and self._magnitude.dtype == self.series.real.dtype
            and (
                (
                    self.series.is_complex()
                    and self._magnitude_component is not None
                    and self._magnitude_component.shape == self.series.shape
                    and self._magnitude_component.device == self.series.device
                    and self._magnitude_component.dtype
                    == self.series.real.dtype
                )
                or (
                    not self.series.is_complex()
                    and self._magnitude_component is None
                )
            )
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self.series)
            )
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self._magnitude)
            )
            and (
                self._magnitude_component is None
                or _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(self._magnitude_component)
            )
        )
        thresh_sq = torch.as_tensor(
            threshold, device=self.series.device, dtype=self.series.real.dtype
        )
        thresh_sq = thresh_sq * thresh_sq
        # Reusable ``out=`` storage is a CPU-only eager optimization. The
        # compiled route is restricted to a single contiguous CUDA search
        # series, where the native accelerator magnitude operation is retained.
        if (
            _TRITON_AVAILABLE
            and self.series.is_cuda
            and self.series.dtype == torch.complex64
            and self.series.is_contiguous()
        ):
            slen = self.series.numel()
            nb = (slen + window - 1) // window
            if self._triton_block_max is None or self._triton_scratch_nb != nb:
                self._triton_scratch_nb = nb
                self._triton_block_max = torch.empty(
                    nb, device=self.series.device, dtype=torch.float32
                )
                self._triton_block_idx = torch.empty(
                    nb, device=self.series.device, dtype=torch.int64
                )
                self._triton_keep = torch.empty(
                    nb, device=self.series.device, dtype=torch.bool
                )

            block_max, block_idx = _triton_symmetric_block_reduce(
                self.series,
                window,
                out_max=self._triton_block_max,
                out_idx=self._triton_block_idx,
            )
            keep = _symmetric_cluster_mask(
                block_max, thresh_sq, out=self._triton_keep
            )
        elif use_scratch:
            mag_sq = _magnitude_squared(
                self.series,
                out=self._magnitude,
                component_out=self._magnitude_component,
            )
            block_max, block_idx = _cluster_candidates(mag_sq, window)
            keep = _symmetric_cluster_mask(block_max, thresh_sq)
        else:
            _, block_idx, keep = _run_threshold_core(
                self.series,
                thresh_sq,
                window,
                single_series=self._source.ndim == 1,
            )

        kept_idx = block_idx[keep]
        flat_series = self.series.reshape(-1)
        kept_vals = flat_series[kept_idx]
        return _array_from_tensor(kept_vals), _array_from_tensor(kept_idx)


def _threshold_cluster_factory(series):
    return TorchThresholdCluster
