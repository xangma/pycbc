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
Torch backend for chisq accumulation and shift_sum.
"""

import os
from functools import lru_cache

import numpy as np
import torch

from pycbc.types import Array
from pycbc.types.array import _convert_to_scheme
from pycbc.types.array_torch import TorchArrayData

try:
    from . import chisq_cpu
except (ImportError, OSError):
    chisq_cpu = None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _triton_pointwise_chisq_bin_kernel(
        corr_real_ptr,
        corr_imag_ptr,
        pts_ptr,
        bin_starts_ptr,
        bin_ends_ptr,
        out_real_ptr,
        out_imag_ptr,
        two_pi_over_N,
        stride_out_p,
        stride_out_b,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_p = tl.program_id(0)
        pid_b = tl.program_id(1)

        p_val = tl.load(pts_ptr + pid_p)
        k_start = tl.load(bin_starts_ptr + pid_b)
        k_end = tl.load(bin_ends_ptr + pid_b)

        num_k = k_end - k_start
        if num_k <= 0:
            out_ptr_re = out_real_ptr + pid_p * stride_out_p + pid_b * stride_out_b
            out_ptr_im = out_imag_ptr + pid_p * stride_out_p + pid_b * stride_out_b
            tl.store(out_ptr_re, 0.0)
            tl.store(out_ptr_im, 0.0)
            return

        acc_real = 0.0
        acc_imag = 0.0

        scale = p_val.to(tl.float64) * two_pi_over_N

        for offset in range(0, num_k, BLOCK_SIZE):
            k_offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = k_offsets < num_k
            k_indices = k_start + k_offsets

            c_re = tl.load(corr_real_ptr + k_indices, mask=mask, other=0.0)
            c_im = tl.load(corr_imag_ptr + k_indices, mask=mask, other=0.0)

            # Double-precision angle range reduction to prevent phase drift
            theta_f64 = scale * k_indices.to(tl.float64)
            turns = (theta_f64 * (1.0 / 6.283185307179586476925286766559)).to(tl.int64)
            theta_red = (
                theta_f64 - turns.to(tl.float64) * 6.283185307179586476925286766559
            ).to(tl.float32)

            cos_t = tl.cos(theta_red)
            sin_t = tl.sin(theta_red)

            term_re = c_re * cos_t - c_im * sin_t
            term_im = c_re * sin_t + c_im * cos_t

            acc_real += tl.sum(tl.where(mask, term_re, 0.0))
            acc_imag += tl.sum(tl.where(mask, term_im, 0.0))

        out_ptr_re = out_real_ptr + pid_p * stride_out_p + pid_b * stride_out_b
        out_ptr_im = out_imag_ptr + pid_p * stride_out_p + pid_b * stride_out_b
        tl.store(out_ptr_re, acc_real)
        tl.store(out_ptr_im, acc_imag)


# ``point_chisq_code`` uses this literal internally.  Correcting the input
# shift lets its double-precision specialization evaluate phases with Torch's
# full-precision value of pi.
_CPU_POINT_CHISQ_PI = 3.141592653
_CPU_POINT_CHISQ_SHIFT_SCALE = torch.pi / _CPU_POINT_CHISQ_PI
_CPU_NATIVE_INT_MAX = np.iinfo(np.int32).max
_CPU_NATIVE_UINT_MAX = np.iinfo(np.uint32).max
_TORCH_IS_INFERENCE = getattr(torch, "is_inference", None)
_functorch_module = getattr(getattr(torch, "_C", None), "_functorch", None)
_is_functorch_wrapped_tensor = getattr(
    _functorch_module, "is_functorch_wrapped_tensor", None
)


def _has_forward_ad_state(tensor):
    """Whether exporting ``tensor`` would silently discard a tangent."""
    # Avoid the Python/dispatcher cost of unpacking ordinary tensors in this
    # small, frequently called sparse-search path.  No dual tensor can exist
    # outside an active forward-AD level.
    current_level = getattr(torch.autograd.forward_ad, "_current_level", None)
    if current_level == -1:
        return False
    try:
        return torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
    except (AttributeError, RuntimeError):
        # If forward-AD state cannot be inspected, retain the Torch route.
        return True


def chisq_accum_bin(chisq, q):
    """Accumulate ``|q|^2`` into chisq."""
    _convert_to_scheme(chisq)
    _convert_to_scheme(q)
    tensor = q._data.tensor
    if tensor.is_complex():
        chisq._data.tensor += torch.view_as_real(tensor).square().sum(dim=-1)
    else:
        chisq._data.tensor += torch.square(tensor)


def _point_tensor(corr, points):
    """Return point indices in the phase dtype used by ``corr``."""
    device = corr.device
    point_data = points._data if isinstance(points, Array) else points
    point_dtype = torch.float32 if device.type == "mps" else torch.float64
    if isinstance(point_data, TorchArrayData):
        return point_data.tensor.to(device=device, dtype=point_dtype)
    if isinstance(point_data, torch.Tensor):
        return point_data.to(device=device, dtype=point_dtype)
    return torch.as_tensor(points, device=device, dtype=point_dtype)


@lru_cache(maxsize=512)
def _cpu_native_bins(bin_edges):
    """Cache the small Cython metadata array shared by repeated templates."""
    return np.asarray(bin_edges, dtype=np.uint32)


def _cpu_native_corr_eligible(corr, bin_edges):
    """Whether Cython can consume ``corr`` and the bin metadata safely."""
    return (
        # NumPy/Cython execution must not bypass Tensor-subclass dispatch or
        # inference-tensor semantics.
        type(corr) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(corr))
        and corr.device.type == "cpu"
        and corr.layout == torch.strided
        and corr.dtype == torch.complex64
        and corr.is_contiguous()
        and not corr.requires_grad
        and not _has_forward_ad_state(corr)
        and not corr.is_conj()
        and not corr.is_neg()
        and len(bin_edges) >= 2
        # The Cython ABI declares these lengths as signed ``int`` and bin
        # edges as ``uint32_t``.  Reject values that would narrow silently.
        and 0 < corr.numel() <= _CPU_NATIVE_INT_MAX
        and len(bin_edges) - 1 <= _CPU_NATIVE_INT_MAX
        and 0 <= bin_edges[0] <= bin_edges[-1] <= corr.numel()
        and bin_edges[-1] <= _CPU_NATIVE_UINT_MAX
        and all(
            start <= end for start, end in zip(bin_edges, bin_edges[1:], strict=False)
        )
    )


def _cpu_native_eligible(corr, pts, bin_edges):
    """Whether Cython can consume the Torch tensors without copying them."""
    return (
        _cpu_native_corr_eligible(corr, bin_edges)
        and type(pts) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(pts))
        and pts.device.type == "cpu"
        and pts.layout == torch.strided
        and pts.dtype == torch.float64
        and pts.is_contiguous()
        and not pts.requires_grad
        and not _has_forward_ad_state(pts)
        and not pts.is_neg()
        and pts.numel() <= _CPU_NATIVE_INT_MAX
    )


def _points_are_empty(points):
    """Return true when a supported points container is known to be empty."""
    point_data = points._data if isinstance(points, Array) else points
    if isinstance(point_data, np.ndarray):
        return point_data.size == 0
    if isinstance(point_data, torch.Tensor):
        return point_data.numel() == 0
    if isinstance(point_data, (tuple, list)):
        return len(point_data) == 0

    if isinstance(point_data, TorchArrayData):
        return point_data.tensor.numel() == 0
    return False


def _cpu_native_index_view(points):
    """Return an exact zero-copy NumPy ABI view of eligible Torch indices.

    The native point-chi-squared route accepts contiguous NumPy ``int64``
    indices and the historical PyCBC ``float64`` result of adding a scalar
    offset to an integer Array.  Torch CPU storage has the same ABI, so expose
    either without changing PyCBC's established scalar-promotion semantics,
    but only when doing so cannot discard device, autograd, or special-view
    semantics.
    The view is request-local: keeping it beyond the native call would extend
    storage lifetime and could make later mutations surprising.
    """
    point_data = points._data if isinstance(points, Array) else points
    if isinstance(point_data, TorchArrayData):
        tensor = point_data.tensor
    elif isinstance(point_data, torch.Tensor):
        tensor = point_data
    else:
        return points

    if not (
        type(tensor) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(tensor))
        and tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and tensor.dtype in (torch.int64, torch.float64)
        and tensor.ndim == 1
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and not _has_forward_ad_state(tensor)
        and not tensor.is_conj()
        and not tensor.is_neg()
    ):
        return points
    try:
        return tensor.numpy()
    except (RuntimeError, TypeError):
        # NumPy interop can be disabled in unusual Torch builds.  The generic
        # Torch path remains exact, so an ABI-export failure is not fatal.
        return points


def _cpu_native_sparse_search_eligible(corr, points, bin_edges, snr):
    """Whether native search kernels can consume sparse NumPy points."""
    return (
        _cpu_native_corr_eligible(corr, bin_edges)
        and type(points) is np.ndarray
        and points.dtype in (np.dtype(np.int64), np.dtype(np.float64))
        and points.ndim == 1
        and 0 < points.size <= _CPU_NATIVE_INT_MAX
        and points.flags.c_contiguous
        and type(snr) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(snr))
        and snr.device.type == "cpu"
        and snr.layout == torch.strided
        and snr.dtype == torch.complex64
        and snr.ndim == 1
        and snr.numel() == points.size
        and snr.is_contiguous()
        and not snr.requires_grad
        and not _has_forward_ad_state(snr)
        and not snr.is_conj()
        and not snr.is_neg()
    )


def _search_compat_storage(value):
    """Unwrap storage without copying or changing tensor dispatch semantics."""
    if isinstance(value, Array):
        value = value._data
    return value.tensor if isinstance(value, TorchArrayData) else value


def _search_compat_array_eligible(value, numpy_dtypes, torch_dtypes):
    """Whether a plain search array can be evaluated by the CPU kernel."""
    if type(value) is np.ndarray:
        return (
            value.dtype in numpy_dtypes and value.ndim == 1 and value.flags.c_contiguous
        )
    return (
        type(value) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(value))
        and value.device.type in ("cpu", "cuda")
        and value.layout == torch.strided
        and value.dtype in torch_dtypes
        and value.ndim == 1
        and value.is_contiguous()
        and not value.requires_grad
        and not _has_forward_ad_state(value)
        and (
            _is_functorch_wrapped_tensor is None
            or not _is_functorch_wrapped_tensor(value)
        )
        and not value.is_conj()
        and not value.is_neg()
    )


_CHISQ_BIN_DEVICE_CACHE = {}


def _get_device_bin_edges(bins, device):
    """Return pre-allocated torch.int32 device tensors for bin starts and ends."""
    key = (tuple(bins) if not isinstance(bins, tuple) else bins, device)
    cached = _CHISQ_BIN_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    bin_starts = torch.as_tensor(bins[:-1], device=device, dtype=torch.int32)
    bin_ends = torch.as_tensor(bins[1:], device=device, dtype=torch.int32)
    if len(_CHISQ_BIN_DEVICE_CACHE) > 2048:
        _CHISQ_BIN_DEVICE_CACHE.clear()
    _CHISQ_BIN_DEVICE_CACHE[key] = (bin_starts, bin_ends)
    return bin_starts, bin_ends


_CUDA_EXACT_CHISQ_MODULE = None
_CUDA_EXACT_CHISQ_TRIED = False


def _get_cuda_exact_chisq_module():
    """Lazily compile and cache the exact single-precision CUDA chi-squared kernel."""
    global _CUDA_EXACT_CHISQ_MODULE, _CUDA_EXACT_CHISQ_TRIED
    if _CUDA_EXACT_CHISQ_TRIED:
        return _CUDA_EXACT_CHISQ_MODULE
    _CUDA_EXACT_CHISQ_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline

        cuda_src = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __device__ const double CPU_PI = 3.141592653;

        __global__ void chisq_fast_kernel(
            const float2* __restrict__ corr,
            const int64_t* __restrict__ points,
            const int32_t* __restrict__ bin_starts,
            const int32_t* __restrict__ bin_ends,
            float* __restrict__ out,
            int N,
            int B,
            int stride_out_p
        ) {
            int pid_p = blockIdx.x;
            int pid_b = blockIdx.y;

            int bstart = bin_starts[pid_b];
            int bend = bin_ends[pid_b];

            if (bend <= bstart) {
                out[pid_p * stride_out_p + pid_b] = 0.0f;
                return;
            }

            double two_pi_c = 2.0 * CPU_PI;
            double pt = (double)points[pid_p];
            double bstart_d = (double)bstart;

            double angle_init = (two_pi_c / (double)N) * (pt * bstart_d);
            float pr = (float)cos(angle_init);
            float pi = (float)sin(angle_init);

            double angle_vs = (two_pi_c / (double)N) * pt;
            float vsr_val = (float)cos(angle_vs);
            float vsi_val = (float)sin(angle_vs);

            float outr = 0.0f;
            float outi = 0.0f;

            for (int j = bstart; j < bend; ++j) {
                float2 c = corr[j];
                float vr = c.x;
                float vi = c.y;
                float vs = vr + vi;
                float va = vi - vr;
                float t1 = pr;
                float t2 = pi;

                float k1 = vr * (t1 + t2);
                float k2 = t1 * va;
                float k3 = t2 * vs;

                outr += (k1 - k3);
                outi += (k1 + k2);

                pr = (t1 * vsr_val) - (t2 * vsi_val);
                pi = (t1 * vsi_val) + (t2 * vsr_val);
            }

            float sq = outr * outr + outi * outi;
            out[pid_p * stride_out_p + pid_b] = sq;
        }

        torch::Tensor launch_chisq_fast(
            torch::Tensor corr,
            torch::Tensor points,
            torch::Tensor bin_starts,
            torch::Tensor bin_ends
        ) {
            int P = points.size(0);
            int B = bin_starts.size(0);
            int N = corr.size(0);
            auto out = torch::empty({P, B}, corr.options().dtype(torch::kFloat32));

            dim3 grid(P, B);
            chisq_fast_kernel<<<grid, 1>>>(
                reinterpret_cast<const float2*>(corr.data_ptr<c10::complex<float>>()),
                points.data_ptr<int64_t>(),
                bin_starts.data_ptr<int32_t>(),
                bin_ends.data_ptr<int32_t>(),
                out.data_ptr<float>(),
                N,
                B,
                out.stride(0)
            );
            return out;
        }
        """
        cpp_src = """
        torch::Tensor launch_chisq_fast(
            torch::Tensor corr,
            torch::Tensor points,
            torch::Tensor bin_starts,
            torch::Tensor bin_ends
        );
        """
        _CUDA_EXACT_CHISQ_MODULE = load_inline(
            name="chisq_exact_cuda_nofmad_v2",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["launch_chisq_fast"],
            extra_cuda_cflags=["--fmad=false", "-O3"],
        )
    except Exception:
        _CUDA_EXACT_CHISQ_MODULE = None
    return _CUDA_EXACT_CHISQ_MODULE


def _cuda_exact_point_chisq(cuda_mod, corr, points, bin_edges, snr, snr_norm):
    """Execute on-GPU exact recurrence chi-squared calculation."""
    num_bins = len(bin_edges) - 1
    device = corr.device

    if isinstance(points, torch.Tensor):
        pts = points.to(device=device, dtype=torch.int64)
    else:
        pts = torch.as_tensor(points, device=device, dtype=torch.int64)

    P = pts.numel()
    if P == 0:
        return torch.empty(0, device=device, dtype=torch.float32)

    bin_starts, bin_ends = _get_device_bin_edges(bin_edges, device)

    out = cuda_mod.launch_chisq_fast(corr, pts, bin_starts, bin_ends)

    if isinstance(snr, torch.Tensor):
        snr_dev = snr.to(device=device)
    else:
        snr_dev = torch.as_tensor(snr, device=device)

    chisq_raw = out.sum(dim=1)
    snr_sq = (snr_dev.conj() * snr_dev).real
    chisq = (chisq_raw * num_bins - snr_sq) * float(snr_norm**2.0)
    return chisq


def _search_compat_point_chisq(corr, points, bin_edges, snr, snr_norm):
    """Return the unchanged CPU search statistic, or None for general inputs.

    The original complex64 search uses float32 phase recurrence and serial
    accumulation, including its historical pi literal and multiply order.
    Direct phases and higher precision accumulation produce a different
    statistic. Reuse that kernel without changing the active scheme. CPU
    storage is exposed through zero-copy NumPy ABI views; CUDA storage is
    copied synchronously to the host and the result returned to its device.

    Tensor normalization and special/differentiable storage retain the Torch
    mathematical path. In particular, never export a tensor after a dtype
    conversion that could have discarded its inference or special-view state.
    """
    points = _search_compat_storage(points)
    snr = _search_compat_storage(snr)
    if not (
        type(snr_norm) in (float, int, np.float32, np.float64)
        and _search_compat_array_eligible(corr, (), (torch.complex64,))
        and _search_compat_array_eligible(
            points,
            (np.dtype(np.int64), np.dtype(np.float64)),
            (torch.int64, torch.float64),
        )
        and _search_compat_array_eligible(
            snr, (np.dtype(np.complex64),), (torch.complex64,)
        )
        and 0 < len(points) <= _CPU_NATIVE_INT_MAX
        and len(snr) == len(points)
        and 0 < corr.numel() <= _CPU_NATIVE_INT_MAX
        and 2 <= len(bin_edges) <= _CPU_NATIVE_INT_MAX + 1
        and 0 <= bin_edges[0] <= bin_edges[-1] <= corr.numel()
        and bin_edges[-1] <= _CPU_NATIVE_UINT_MAX
        and all(
            start <= end for start, end in zip(bin_edges, bin_edges[1:], strict=False)
        )
    ):
        return None

    if corr.device.type == "cuda" and (
        chisq_cpu is None
        or not hasattr(chisq_cpu, "point_chisq_code")
        or os.environ.get("PYCBC_TORCH_CUDA_EXACT_CHISQ", "0") == "1"
    ):
        cuda_mod = _get_cuda_exact_chisq_module()
        if cuda_mod is not None:
            return _cuda_exact_point_chisq(
                cuda_mod, corr, points, bin_edges, snr, snr_norm
            )

    if chisq_cpu is None or not hasattr(chisq_cpu, "point_chisq_code"):
        return None
    point_chisq_code = chisq_cpu.point_chisq_code

    def host_view(value):
        if type(value) is np.ndarray:
            return value
        if value.device.type == "cpu":
            return value.numpy()
        return value.cpu().numpy()

    corr_host = host_view(corr)
    snr_host = host_view(snr)
    shifts = np.asarray(host_view(points), dtype=np.float32)
    num_bins = len(bin_edges) - 1
    chisq = np.zeros(len(shifts), dtype=np.float32)
    point_chisq_code(
        chisq,
        corr_host,
        len(shifts),
        len(corr_host),
        shifts,
        _cpu_native_bins(bin_edges),
        num_bins,
    )
    # Keep the original expression and scalar type: rounding the norm before
    # squaring or folding the SNR subtraction into the accumulator changes it.
    chisq = (chisq * num_bins - (snr_host.conj() * snr_host).real) * (snr_norm**2.0)
    res = torch.from_numpy(chisq)
    if corr.device.type == "cpu":
        return res
    return res.to(device=corr.device)


def _cpu_native_point_chisq(corr, pts, bin_edges, snr=None, snr_norm=None):
    """Run the fused CPU kernel on zero-copy views of sparse storage.

    The CPU kernel has independent real and complex fused types.  Its
    float64/complex64 specialization therefore retains double-precision phase
    recurrence and accumulation without promoting or copying the correlation.
    All writable buffers are allocated by Torch; NumPy only supplies zero-copy
    views required by the existing Cython ABI. Search indices already provided
    as contiguous NumPy int64 storage are consumed directly rather than copied
    through a temporary Torch tensor. Autograd inputs never enter this helper
    and continue through the native Torch implementation.
    """
    if chisq_cpu is None or not hasattr(chisq_cpu, "point_chisq_code"):
        raise RuntimeError("Cython chisq_cpu extension is unavailable")
    point_chisq_code = chisq_cpu.point_chisq_code

    if isinstance(pts, np.ndarray):
        count = pts.size
        pts_np = pts
    else:
        count = pts.numel()
        pts_np = pts.numpy()
    num_bins = len(bin_edges) - 1
    workspace = torch.empty(3 * count, device="cpu", dtype=torch.float64)
    corrected_pts = workspace[:count]
    accum = workspace[count : 2 * count]
    scratch = workspace[2 * count :]

    # Eligibility excludes autograd, forward-AD, and special views, making
    # direct NumPy export safe and avoiding redundant detach wrappers.
    corrected_pts_np = corrected_pts.numpy()
    np.multiply(
        pts_np,
        _CPU_POINT_CHISQ_SHIFT_SCALE,
        out=corrected_pts_np,
    )

    accum_np = accum.numpy()
    if snr is None:
        accum_np.fill(0.0)
    elif count == 1:
        value = snr.numpy()[0]
        real = float(value.real)
        imag = float(value.imag)
        accum_np[0] = -(real * real + imag * imag) / num_bins
    else:
        snr_np = snr.numpy()
        scratch_np = scratch.numpy()
        np.multiply(snr_np.real, snr_np.real, out=accum_np, dtype=np.float64)
        np.multiply(
            snr_np.imag,
            snr_np.imag,
            out=scratch_np,
            dtype=np.float64,
        )
        np.add(accum_np, scratch_np, out=accum_np)
        np.multiply(accum_np, -1.0 / num_bins, out=accum_np)

    point_chisq_code(
        accum_np,
        corr.numpy(),
        count,
        corr.numel(),
        corrected_pts_np,
        _cpu_native_bins(bin_edges),
        num_bins,
    )

    if snr is not None:
        scale = num_bins * float(snr_norm) * float(snr_norm)
        # This is private, freshly allocated workspace and native eligibility
        # excludes autograd, so scaling in-place avoids a redundant float64
        # allocation before the required result-dtype conversion.
        accum.mul_(scale)
    return accum.to(dtype=corr.real.dtype)


def _cumprod_phases(phases):
    """Accumulate phases in-place unless forward AD requires a result."""
    if _has_forward_ad_state(phases):
        return torch.cumprod(phases, dim=1)
    torch.cumprod(phases, dim=1, out=phases)
    return phases


def _cpu_shifted_bin_sum(corr, pts, start, end, step_angle, step):
    """Sum one frequency bin at selected points on a Torch CPU.

    Constructing every Fourier phase with a complex exponential dominates
    sparse pointwise chi-squared calculations.  A phase recurrence needs only
    the bin's initial phase and the shared phase step.  Evaluate that
    recurrence in complex128, even for complex64 correlations, so rounding the
    phases back to the correlation dtype retains the accuracy of the direct
    exponential implementation.
    """
    width = end - start
    if width <= 0:
        return torch.zeros(pts.numel(), device=pts.device, dtype=corr.dtype)

    initial = torch.polar(torch.ones_like(step_angle), step_angle * start)
    phases = step[:, None].expand(-1, width).clone()
    phases[:, 0] = initial
    phases = _cumprod_phases(phases)
    if phases.dtype != corr.dtype:
        phases = phases.to(corr.dtype)
    return torch.sum(corr[start:end] * phases, dim=1)


def _cpu_single_point_bin_sums(corr, bins, step_angle, step):
    """Sum all contiguous bins for one point with one phase recurrence.

    Production searches commonly evaluate the pointwise chi-squared test for
    a single surviving trigger.  The bins partition one contiguous frequency
    band, so generating a separate, max-bin-width recurrence row for every
    bin duplicates most of that band.  Generate its phases once, then reduce
    the corresponding slices.  Complex128 recurrence still makes the phases
    round to the same complex64 values as the direct-exponential path for
    production-sized spectra, while limiting temporary storage to the actual
    band width.
    """
    band_start = bins[0]
    band_end = bins[-1]
    width = band_end - band_start
    if width <= 0:
        return torch.zeros((1, len(bins) - 1), device=corr.device, dtype=corr.dtype)

    phases = step[:, None].expand(-1, width).clone()
    phases[:, 0] = torch.polar(torch.ones_like(step_angle), step_angle * band_start)
    phases = _cumprod_phases(phases)
    if phases.dtype != corr.dtype:
        phases = phases.to(corr.dtype)

    weighted = corr[band_start:band_end] * phases[0]

    # One high-precision prefix sum is both faster than launching a reduction
    # for every bin and more accurate than accumulating each complex64 slice.
    # Strictly increasing edges are the normal search case; retain support for
    # empty bins without introducing an extra full-band prefix allocation.
    if all(end > start for start, end in zip(bins, bins[1:], strict=False)):
        cumulative = torch.cumsum(weighted, dim=0, dtype=torch.complex128)
        end_offsets = torch.as_tensor(
            [end - band_start - 1 for end in bins[1:]],
            device=corr.device,
        )
        boundaries = cumulative[end_offsets]
        bin_sums = torch.diff(boundaries, prepend=boundaries.new_zeros(1))
    else:
        bin_sums = torch.stack(
            [
                torch.sum(
                    weighted[start - band_start : end - band_start],
                    dtype=torch.complex128,
                )
                for start, end in zip(bins, bins[1:], strict=False)
            ],
            dim=0,
        )
    return bin_sums.to(corr.dtype).reshape(1, -1)


def _accelerator_phase_reuse_eligible(corr, pts, bins):
    """Whether accelerator phase vector reuse preserves current semantics."""
    return (
        corr.device.type != "cpu"
        # The optimization coalesces one phase-construction dispatch per bin
        # into a single union-band dispatch for all points. Preserve
        # Tensor-subclass dispatch and side-effect semantics by retaining the
        # legacy route for subclassed point metadata.
        and type(pts) is torch.Tensor
        and pts.numel() > 0
        # Sharing a phase graph changes the grouping of point-coordinate
        # gradients. Search points are metadata, so retain the legacy path for
        # the uncommon differentiable-point case.
        and not pts.requires_grad
        and not _has_forward_ad_state(pts)
        and len(bins) >= 2
        and 0 <= bins[0] <= bins[-1] <= corr.shape[-1]
        and all(start <= end for start, end in zip(bins, bins[1:], strict=False))
    )


def _accelerator_batched_bin_sums(corr, pts, bins):
    """Sum accelerator bins with batched (P x K) phase matrix or fused Triton operations."""
    length = corr.shape[-1]
    band_start = bins[0]
    band_end = bins[-1]
    width = band_end - band_start
    if width <= 0:
        return torch.zeros(
            (pts.numel(), len(bins) - 1), device=corr.device, dtype=corr.dtype
        )

    # Fast-path for CUDA GPUs via fused SRAM register accumulation
    if _HAS_TRITON and corr.device.type == "cuda" and corr.dtype == torch.complex64:
        P = pts.numel()
        B = len(bins) - 1
        two_pi_over_N = float(2.0 * np.pi / length)
        bin_starts, bin_ends = _get_device_bin_edges(bins, corr.device)
        pts_f64 = pts.to(torch.float64)
        corr_re = corr.real.contiguous()
        corr_im = corr.imag.contiguous()
        out_re = torch.empty((P, B), device=corr.device, dtype=torch.float32)
        out_im = torch.empty((P, B), device=corr.device, dtype=torch.float32)
        grid = (P, B)
        _triton_pointwise_chisq_bin_kernel[grid](
            corr_re,
            corr_im,
            pts_f64,
            bin_starts,
            bin_ends,
            out_re,
            out_im,
            # Runtime Python floats become fp32 Triton arguments. Preserve
            # this coefficient for the kernel's fp64 phase range reduction.
            tl.constexpr(two_pi_over_N),
            out_re.stride(0),
            out_re.stride(1),
            BLOCK_SIZE=512,
        )
        return torch.complex(out_re, out_im)

    frequency = torch.arange(band_start, band_end, device=corr.device, dtype=pts.dtype)
    phase = torch.exp(
        2j * torch.pi * pts[:, None] * frequency[None, :] / float(length)
    ).to(corr.dtype)

    weighted = phase * corr[band_start:band_end][None, :]

    if all(end > start for start, end in zip(bins, bins[1:], strict=False)):
        if corr.device.type == "mps":
            cum = torch.complex(
                torch.cumsum(weighted.real, dim=1),
                torch.cumsum(weighted.imag, dim=1),
            )
        else:
            cum = torch.cumsum(weighted, dim=1, dtype=torch.complex128)
        end_offsets = torch.as_tensor(
            [end - band_start - 1 for end in bins[1:]],
            device=corr.device,
        )
        boundaries = cum[:, end_offsets]
        out = torch.diff(
            boundaries, dim=1, prepend=boundaries.new_zeros((pts.numel(), 1))
        )
        return out.to(corr.dtype)

    out = torch.stack(
        [
            torch.sum(weighted[:, start - band_start : end - band_start], dim=1)
            for start, end in zip(bins, bins[1:], strict=False)
        ],
        dim=1,
    )
    return out.to(corr.dtype)


def _accelerator_single_point_bin_sums(corr, pts, bins):
    """Sum accelerator bins after constructing their shared phases once."""
    length = corr.shape[-1]
    band_start = bins[0]
    band_end = bins[-1]
    frequency = torch.arange(band_start, band_end, device=corr.device, dtype=pts.dtype)
    phase = torch.exp(
        2j * torch.pi * pts[:, None] * frequency[None, :] / float(length)
    ).to(corr.dtype)

    out = torch.zeros((1, len(bins) - 1), device=corr.device, dtype=corr.dtype)
    for index, (start, end) in enumerate(zip(bins, bins[1:], strict=False)):
        offset_start = start - band_start
        offset_end = end - band_start
        out[:, index] = torch.sum(
            corr[start:end] * phase[:, offset_start:offset_end], dim=1
        )
    return out


def _accelerator_single_point_phase_reuse_eligible(corr, pts, bins):
    """Whether single-point accelerator phase vector reuse preserves current semantics."""
    return getattr(
        pts, "numel", lambda: 0
    )() == 1 and _accelerator_phase_reuse_eligible(corr, pts, bins)


def shift_sum(corr, points, bins):
    """
    Calculate time-shifted sums of corr over provided bins at given points.
    """
    device = corr._data.tensor.device
    dtype = corr._data.tensor.dtype
    N = corr._data.tensor.shape[-1]

    pts = _point_tensor(corr._data.tensor, points)
    if pts.numel() == 0:
        empty = torch.empty(0, device=device, dtype=corr._data.tensor.real.dtype)
        return Array(TorchArrayData(empty), copy=False)
    if N == 0:
        raise ValueError("correlation must contain at least one sample")
    bin_edges = tuple(int(edge) for edge in bins)
    nbins = len(bin_edges) - 1

    if _cpu_native_eligible(corr._data.tensor, pts, bin_edges):
        chisq = _cpu_native_point_chisq(corr._data.tensor, pts, bin_edges)
        return Array(TorchArrayData(chisq), copy=False)

    if device.type == "cpu" and dtype == torch.complex64:
        step_angle = 2 * torch.pi * pts / float(N)
        step = torch.polar(torch.ones_like(step_angle), step_angle)
        if pts.numel() == 1:
            out = _cpu_single_point_bin_sums(
                corr._data.tensor, bin_edges, step_angle, step
            )
        else:
            out = torch.zeros((pts.numel(), nbins), device=device, dtype=dtype)
            for j in range(nbins):
                s, e = bin_edges[j], bin_edges[j + 1]
                out[:, j] = _cpu_shifted_bin_sum(
                    corr._data.tensor, pts, s, e, step_angle, step
                )
    elif _accelerator_single_point_phase_reuse_eligible(
        corr._data.tensor, pts, bin_edges
    ):
        out = _accelerator_single_point_bin_sums(corr._data.tensor, pts, bin_edges)
    elif _accelerator_phase_reuse_eligible(corr._data.tensor, pts, bin_edges):
        out = _accelerator_batched_bin_sums(corr._data.tensor, pts, bin_edges)
    else:
        out = torch.zeros((pts.numel(), nbins), device=device, dtype=dtype)
        for j in range(nbins):
            s, e = bin_edges[j], bin_edges[j + 1]
            point_dtype = pts.dtype
            idx = torch.arange(s, e, device=device, dtype=point_dtype)
            phase = torch.exp(2j * torch.pi * pts[:, None] * idx[None, :] / float(N))
            phase = phase.to(dtype)
            sl = corr._data.tensor[s:e]
            out[:, j] = torch.sum(sl * phase, dim=1)

    chisq = torch.sum(torch.conj(out) * out, dim=1).real

    # Wrap tensor so Array sees torch-backed storage and avoids host copies
    return Array(TorchArrayData(chisq), copy=False)


def power_chisq_at_points_from_precomputed(corr, snr, snr_norm, bins, indices):
    """Calculate the power chisq at points using Torch-backed storage.

    Ordinary complex64 CPU/CUDA searches use the original CPU statistic.
    General mathematical, MPS and differentiable inputs retain Torch paths.
    """
    _convert_to_scheme(corr)

    device = corr._data.tensor.device
    dtype = corr._data.tensor.dtype

    if _points_are_empty(indices):
        empty = torch.empty(0, device=device, dtype=corr._data.tensor.real.dtype)
        return Array(TorchArrayData(empty), copy=False)
    if corr._data.tensor.numel() == 0:
        raise ValueError("correlation must contain at least one sample")

    num_bins = len(bins) - 1
    bin_edges = tuple(int(edge) for edge in bins)
    compatible = _search_compat_point_chisq(
        corr._data.tensor, indices, bin_edges, snr, snr_norm
    )
    if compatible is not None:
        return Array(TorchArrayData(compatible), copy=False)
    native_indices = _cpu_native_index_view(indices)

    if isinstance(snr, TorchArrayData):
        snr_t = snr.tensor
        if snr_t.device != device or snr_t.dtype != dtype:
            snr_t = snr_t.to(device=device, dtype=dtype)
    elif hasattr(snr, "_scheme"):
        _convert_to_scheme(snr)
        snr_t = snr._data.tensor
    else:
        snr_t = torch.as_tensor(snr, device=device, dtype=dtype)

    norm_value = None
    if isinstance(snr_norm, torch.Tensor):
        if (
            # Extracting a Python scalar must not bypass Tensor-subclass or
            # inference-tensor dispatch semantics.
            type(snr_norm) is torch.Tensor
            and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(snr_norm))
            and snr_norm.device.type == "cpu"
            and snr_norm.layout == torch.strided
            and snr_norm.numel() == 1
            and not snr_norm.requires_grad
            and not _has_forward_ad_state(snr_norm)
        ):
            norm_value = snr_norm.item()
    else:
        try:
            norm_value = float(snr_norm)
        except (TypeError, ValueError):
            pass

    native_snr = (
        _cpu_native_sparse_search_eligible(
            corr._data.tensor, native_indices, bin_edges, snr_t
        )
        and norm_value is not None
    )
    if native_snr:
        chisq_t = _cpu_native_point_chisq(
            corr._data.tensor,
            native_indices,
            bin_edges,
            snr=snr_t,
            snr_norm=norm_value,
        )
        return Array(TorchArrayData(chisq_t), copy=False)

    pts = _point_tensor(corr._data.tensor, indices)
    native_snr = (
        _cpu_native_eligible(corr._data.tensor, pts, bin_edges)
        and type(snr_t) is torch.Tensor
        and (_TORCH_IS_INFERENCE is None or not _TORCH_IS_INFERENCE(snr_t))
        and snr_t.device.type == "cpu"
        and snr_t.layout == torch.strided
        and snr_t.dtype == torch.complex64
        and snr_t.is_contiguous()
        and not snr_t.requires_grad
        and not _has_forward_ad_state(snr_t)
        and not snr_t.is_conj()
        and not snr_t.is_neg()
        and snr_t.numel() == pts.numel()
        and norm_value is not None
    )
    if native_snr:
        chisq_t = _cpu_native_point_chisq(
            corr._data.tensor,
            pts,
            bin_edges,
            snr=snr_t,
            snr_norm=norm_value,
        )
        return Array(TorchArrayData(chisq_t), copy=False)

    chisq_arr = shift_sum(corr, pts, bin_edges)
    chisq_t = chisq_arr._data.tensor

    snr_term = (torch.conj(snr_t) * snr_t).real
    chisq_t = chisq_t * num_bins - snr_term

    if isinstance(snr_norm, (int, float, complex, np.number, np.bool_)):
        scale = float(snr_norm) ** 2
        chisq_t = chisq_t * scale
    elif isinstance(snr_norm, torch.Tensor):
        chisq_t = chisq_t * (snr_norm.to(device=device, dtype=chisq_t.dtype) ** 2)
    else:
        snr_norm_t = torch.as_tensor(snr_norm, device=device, dtype=chisq_t.dtype)
        chisq_t = chisq_t * (snr_norm_t**2)

    return Array(TorchArrayData(chisq_t), copy=False)
