# Copyright (C) 2026  The PyCBC team
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

"""Torch interpolation backend for compressed frequency-domain waveforms."""

import sys

import numpy
import torch

from pycbc.types.backend import backend_array

try:
    from . import decompress_cpu_cython
    from .decompress_cpu_cython import (
        decomp_ccode_double,
        decomp_ccode_float,
    )
except ImportError:
    decompress_cpu_cython = None
    decomp_ccode_double = None
    decomp_ccode_float = None

_STENCIL_OFFSETS = {
    1: (0, 1),
    2: (-1, 0, 1),
    3: (-1, 0, 1, 2),
    4: (-1, 0, 1, 2, 3),
}


def _as_tensor(value, device, dtype):
    """Return ``value`` as a tensor on the output device."""
    value = backend_array(value)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(numpy.asarray(value), device=device, dtype=dtype)


def _lagrange_weights(nodes, points):
    """Evaluate Lagrange basis weights for each row of ``nodes``."""
    weights = []
    for column in range(nodes.shape[1]):
        numerator = torch.ones_like(points)
        denominator = torch.ones_like(points)
        for other in range(nodes.shape[1]):
            if other == column:
                continue
            numerator = numerator * (points - nodes[:, other])
            denominator = denominator * (nodes[:, column] - nodes[:, other])
        weights.append(numerator / denominator)
    return torch.stack(weights, dim=1)


def _grid_indices(frequencies, df):
    """Truncate frequencies to output-grid indices like the CPU backend."""
    ratios = frequencies / df
    if ratios.dtype == torch.float64:
        # The optimized C++ reference treats a quotient that is only a few
        # double-precision ulps from an integer as that grid boundary.  Avoid
        # assigning such a point to the adjacent interpolation segment merely
        # because tensor division rounded in the opposite direction.
        nearest = torch.round(ratios)
        scale = torch.maximum(torch.abs(ratios), torch.ones_like(ratios))
        tolerance = 8 * torch.finfo(ratios.dtype).eps * scale
        ratios = torch.where(
            torch.abs(ratios - nearest) <= tolerance,
            nearest,
            ratios,
        )
    return torch.trunc(ratios).to(torch.int64)


def _cpu_numpy_view(value):
    """Expose ordinary CPU storage without copying or bypassing Torch AD."""
    value = backend_array(value)
    if type(value) is torch.Tensor:
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype
            not in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            or value.requires_grad
            or value.is_conj()
            or value.is_neg()
            or torch.is_inference(value)
        ):
            return None
        try:
            dual = torch.autograd.forward_ad.unpack_dual(value)
            if dual.tangent is not None:
                return None
        except (AttributeError, RuntimeError):
            return None
        value = value.numpy()
    if (
        type(value) is not numpy.ndarray
        or value.ndim != 1
        or not value.flags.c_contiguous
        or not value.flags.aligned
        or not value.flags.writeable
    ):
        return None
    return value


_FREQ_VALID_CACHE = {}


def _is_frequencies_valid(frequencies, key_obj=None):
    """Check if frequencies are non-negative, finite, and strictly increasing.

    Results are cached across calls since template bank frequencies are fixed.
    """
    key = (
        id(key_obj) if key_obj is not None else None,
        frequencies.ctypes.data,
        frequencies.size,
        frequencies.dtype.num,
        frequencies.strides,
        float(frequencies[0]),
        float(frequencies[-1]),
    )
    res = _FREQ_VALID_CACHE.get(key)
    if res is not None:
        return res

    if (
        frequencies[0] < 0
        or not numpy.all(numpy.isfinite(frequencies))
        or not numpy.all(frequencies[1:] > frequencies[:-1])
    ):
        valid = False
    else:
        valid = True

    if len(_FREQ_VALID_CACHE) >= 64:
        _FREQ_VALID_CACHE.clear()
    _FREQ_VALID_CACHE[key] = valid
    return valid


def _cpu_linear_interp(amp, phase, sample_frequencies, output, df, imin, start_index):
    """Use the existing Cython/C++ linear kernel for compatible CPU storage."""
    out = backend_array(output, "torch")
    if (
        type(out) is not torch.Tensor
        or out.device.type != "cpu"
        or out.dtype not in (torch.complex64, torch.complex128)
        or not hasattr(torch.autograd.graph, "increment_version")
        or not isinstance(df, (int, float, numpy.integer, numpy.floating))
        or not isinstance(imin, (int, numpy.integer))
        or not isinstance(start_index, (int, numpy.integer))
    ):
        return False

    h = _cpu_numpy_view(out)
    arrays = tuple(_cpu_numpy_view(value) for value in (sample_frequencies, amp, phase))
    if h is None or any(value is None for value in arrays):
        return False
    frequencies, amplitudes, phases = arrays
    dtype = numpy.float32 if out.dtype == torch.complex64 else numpy.float64
    sample_count = frequencies.size
    # The Python-facing Cython wrappers accept lengths/indices as C ints,
    # then pass int64_t values and unchecked contiguous pointers to C++.
    if (
        not all(value.dtype == dtype and value.size == sample_count for value in arrays)
        or not 2 <= sample_count <= numpy.iinfo(numpy.int32).max
        or not 0 <= start_index < h.size <= numpy.iinfo(numpy.int32).max
        or not 0 <= imin < sample_count - 1
        or any(numpy.shares_memory(h, value) for value in arrays)
    ):
        return False

    # Match the single-precision Cython argument rounding before checking its
    # index arithmetic. Short first intervals can move the C++ cursor below
    # start_index; keep those cases in the existing Torch implementation.
    delta_f = dtype(df).item()
    if not numpy.isfinite(delta_f) or delta_f <= 0:
        return False
    if not _is_frequencies_valid(frequencies, sample_frequencies):
        return False
    last_ratio = float(frequencies[-1]) / delta_f
    first_end = float(frequencies[imin + 1]) / delta_f
    if (
        not last_ratio < numpy.iinfo(numpy.int32).max
        or int(first_end) + (imin == sample_count - 2) < start_index
    ):
        return False

    if (
        decompress_cpu_cython is None
        or sys.modules.get("pycbc.waveform.decompress_cpu_cython") is None
    ):
        return False
    function = (
        getattr(decompress_cpu_cython, "decomp_ccode_float", decomp_ccode_float)
        if out.dtype == torch.complex64
        else getattr(
            decompress_cpu_cython, "decomp_ccode_double", decomp_ccode_double
        )
    )
    if function is None:
        return False
    try:
        function(
            h,
            float(df),
            h.size,
            start_index,
            frequencies,
            amplitudes,
            phases,
            sample_count,
            imin,
        )
    finally:
        # NumPy is only the zero-copy ABI: notify Torch and every shared view
        # of the native write, including writes made before a kernel error.
        torch.autograd.graph.increment_version(out)
    return True


def _inline_interp(
    amp, phase, sample_frequencies, output, df, imin, start_index, degree
):
    """Interpolate amplitude and phase with CPU-backend stencil semantics."""
    out = backend_array(output, "torch")
    if out is None:
        raise TypeError("Torch decompression requires Torch-backed output")

    # The CPU implementation promotes interpolation arithmetic to double,
    # including for single-precision output. MPS cannot represent float64,
    # but CPU and CUDA follow that behavior exactly.
    input_dtype = torch.float32 if out.dtype == torch.complex64 else torch.float64
    calc_dtype = torch.float32 if out.device.type == "mps" else torch.float64
    # The single-precision Cython entry points accept ``df`` as a C float
    # before promoting it for the C++ interpolation arithmetic.  Preserve
    # that rounding here as it also determines whether the final grid point
    # is inside the half-open output interval.
    calc_df = numpy.float32(df).item() if out.dtype == torch.complex64 else float(df)
    frequencies = _as_tensor(sample_frequencies, out.device, input_dtype).to(calc_dtype)
    amplitudes = _as_tensor(amp, out.device, input_dtype).to(calc_dtype)
    phases = _as_tensor(phase, out.device, input_dtype).to(calc_dtype)

    out.zero_()
    sample_count = frequencies.numel()
    if sample_count < 2:
        return output

    if (
        hasattr(sample_frequencies, "__getitem__")
        and not (
            isinstance(sample_frequencies, torch.Tensor)
            and sample_frequencies.device.type != "cpu"
        )
    ):
        try:
            last_freq = torch.as_tensor(
                sample_frequencies[-1], dtype=calc_dtype
            )
        except Exception:
            last_freq = frequencies[-1]
    else:
        last_freq = frequencies[-1]
    last_index = int(_grid_indices(last_freq, calc_df))
    end_index = min(out.numel(), last_index + 1)
    if end_index <= start_index:
        return output

    output_indices = torch.arange(
        start=start_index,
        end=end_index,
        device=out.device,
        dtype=torch.int64,
    )

    # Match the CPU segment boundaries exactly. For non-final segments the
    # compiled backend stops at int(f[i + 1] / df), while the final segment
    # includes that index. ``right=True`` assigns each boundary index to the
    # following segment, and the clamp restores the final-segment exception.
    segment_ends = _grid_indices(frequencies[1:], calc_df)
    segments = torch.searchsorted(
        segment_ends.contiguous(), output_indices, right=True
    )
    segments.clamp_(min=int(imin), max=sample_count - 2)

    if degree == 1:
        points = output_indices.to(calc_dtype) * calc_df
        f0 = frequencies[segments]
        f1 = frequencies[segments + 1]
        w1 = (points - f0) / (f1 - f0)
        w0 = 1.0 - w1
        interp_amp = amplitudes[segments] * w0 + amplitudes[segments + 1] * w1
        interp_phase = phases[segments] * w0 + phases[segments + 1] * w1
        waveform = torch.complex(
            interp_amp * torch.cos(interp_phase),
            interp_amp * torch.sin(interp_phase),
        ).to(dtype=out.dtype)
        out[start_index:start_index + waveform.numel()] = waveform
        return output

    max_degree = min(degree, sample_count - 1)
    degrees = torch.full_like(segments, max_degree)
    degrees.masked_fill_(segments == 0, 1)
    if max_degree > 3:
        degrees.masked_fill_(segments >= sample_count - 3, 3)
    if max_degree > 2:
        degrees.masked_fill_(segments >= sample_count - 2, 2)

    for current_degree in range(1, max_degree + 1):
        positions = torch.nonzero(degrees == current_degree, as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        selected_segments = segments.index_select(0, positions)
        offsets = torch.tensor(
            _STENCIL_OFFSETS[current_degree],
            device=out.device,
            dtype=torch.int64,
        )
        stencil = selected_segments[:, None] + offsets[None, :]
        nodes = frequencies[stencil]
        points = output_indices.index_select(0, positions).to(calc_dtype)
        points = points * calc_df
        weights = _lagrange_weights(nodes, points)

        interp_amp = (amplitudes[stencil] * weights).sum(dim=1)
        interp_phase = (phases[stencil] * weights).sum(dim=1)
        waveform = torch.complex(
            interp_amp * torch.cos(interp_phase),
            interp_amp * torch.sin(interp_phase),
        ).to(dtype=out.dtype)
        out.index_copy_(0, output_indices.index_select(0, positions), waveform)

    return output


def inline_linear_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    if _cpu_linear_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index
    ):
        return output
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 1
    )


def inline_quadratic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 2
    )


def inline_cubic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 3
    )


def inline_quartic_interp(
    amp, phase, sample_frequencies, output, df, f_lower, imin, start_index
):
    return _inline_interp(
        amp, phase, sample_frequencies, output, df, imin, start_index, 4
    )


class AsyncWaveformPrefetcher:
    """Asynchronous double-buffered waveform prefetcher for PyCBC on CUDA.

    Prefetches and decompresses template N+1 on a secondary CUDA stream while
    template N is being processed (matched-filtered) on the primary CUDA
    stream, using double-buffered GPU FrequencySeries memory and CUDA event
    synchronization.

    Provides a clean Python iterator yielding (t_num, template) and
    automatically falls back to sequential execution on CPU schemes.
    """

    def __init__(self, bank, indices=None, scheme_ctx=None, preload=False):
        self.bank = bank
        self.indices = (
            list(range(len(bank))) if indices is None else list(indices)
        )
        self.scheme_ctx = scheme_ctx
        self.preload = preload

        from pycbc.scheme import TorchScheme, mgr

        state = scheme_ctx if scheme_ctx is not None else mgr.state
        self.is_cuda = (
            isinstance(state, TorchScheme)
            and state.torch_device.type == "cuda"
            and torch.cuda.is_available()
            and getattr(bank, "has_compressed_waveforms", False)
            and getattr(bank, "enable_compressed_waveforms", False)
        )

        if self.is_cuda:
            from pycbc.types import FrequencySeries, zeros

            self.device = state.torch_device
            self.decomp_stream = torch.cuda.Stream(device=self.device)
            self.compute_stream = torch.cuda.current_stream(device=self.device)

            self.buffers = [
                zeros(bank.filter_length, dtype=bank.dtype),
                zeros(bank.filter_length, dtype=bank.dtype),
            ]
            self.fs_buffers = [
                FrequencySeries(
                    self.buffers[0], delta_f=bank.delta_f, copy=False
                ),
                FrequencySeries(
                    self.buffers[1], delta_f=bank.delta_f, copy=False
                ),
            ]

            self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]
            self.done_events = [torch.cuda.Event(), torch.cuda.Event()]

            for event in self.done_events:
                event.record(self.compute_stream)
        else:
            self.buffers = None
            self.fs_buffers = None
            self.decomp_stream = None
            self.compute_stream = None
            self.ready_events = None
            self.done_events = None

        self._compressed_cache = {}
        if preload and getattr(bank, "has_compressed_waveforms", False):
            from pycbc.waveform import compress

            for idx in self.indices:
                tmplt_hash = bank.table.template_hash[idx]
                self._compressed_cache[idx] = (
                    compress.CompressedWaveform.from_hdf(
                        bank.filehandler, tmplt_hash, load_now=True
                    )
                )

    def _decompress_into(self, slot, t_num):
        """Decompress template t_num into buffer slot."""
        from pycbc.waveform import compress
        from pycbc.waveform.bank import find_variable_start_frequency
        from pycbc.waveform.waveform import (
            get_waveform_filter_length_in_time,
            props,
        )

        bank = self.bank
        approximant = bank.approximant(t_num)
        f_low = find_variable_start_frequency(
            approximant,
            bank.table[t_num],
            bank.f_lower,
            bank.max_template_length,
        )

        self.buffers[slot].clear()

        if t_num in self._compressed_cache:
            cw = self._compressed_cache[t_num]
        else:
            tmplt_hash = bank.table.template_hash[t_num]
            cw = compress.CompressedWaveform.from_hdf(
                bank.filehandler, tmplt_hash, load_now=True
            )

        method = bank.waveform_decompression_method or cw.interpolation
        fs = self.fs_buffers[slot]
        hdecomp = cw.decompress(
            out=fs,
            f_lower=f_low,
            interpolation=method,
        )

        p = props(bank.table[t_num])
        p.pop("approximant", None)
        try:
            tmpltdur = bank.table[t_num].template_duration
        except AttributeError:
            tmpltdur = None
        if tmpltdur is None or tmpltdur == 0.0:
            tmpltdur = get_waveform_filter_length_in_time(approximant, **p)

        f_end = bank.end_frequency(t_num)
        if f_end is None or f_end >= (bank.filter_length * bank.delta_f):
            f_end = (bank.filter_length - 1) * bank.delta_f

        from pycbc.waveform.bank import sigma_cached

        hdecomp.f_lower = f_low
        hdecomp.min_f_lower = bank.min_f_lower
        hdecomp.end_idx = int(f_end / hdecomp.delta_f)
        hdecomp.params = bank.table[t_num]
        hdecomp.chirp_length = tmpltdur
        hdecomp.length_in_time = tmpltdur
        hdecomp.approximant = approximant
        hdecomp.end_frequency = f_end
        hdecomp.sigmasq = sys.modules["types"].MethodType(sigma_cached, hdecomp)
        hdecomp._sigmasq = {}
        return hdecomp

    def __iter__(self):
        if not self.is_cuda:
            for t_num in self.indices:
                yield t_num, self.bank[t_num]
            return

        n = len(self.indices)
        if n == 0:
            return

        try:
            slot_0 = 0
            self.decomp_stream.wait_event(self.done_events[slot_0])
            with torch.cuda.stream(self.decomp_stream):
                self._decompress_into(slot_0, self.indices[0])
                self.ready_events[slot_0].record(self.decomp_stream)

            for i in range(n):
                curr_slot = i % 2
                next_slot = (i + 1) % 2
                curr_idx = self.indices[i]

                if i + 1 < n:
                    next_idx = self.indices[i + 1]
                    self.decomp_stream.wait_event(self.done_events[next_slot])
                    with torch.cuda.stream(self.decomp_stream):
                        self._decompress_into(next_slot, next_idx)
                        self.ready_events[next_slot].record(self.decomp_stream)

                self.compute_stream.wait_event(self.ready_events[curr_slot])
                curr_template = self.fs_buffers[curr_slot]

                yield curr_idx, curr_template

                self.done_events[curr_slot].record(self.compute_stream)
        finally:
            self.compute_stream.synchronize()
            self.decomp_stream.synchronize()
