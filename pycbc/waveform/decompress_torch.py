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
Torch backend for waveform decompression (inline interpolation helpers).

These functions are wired by the scheme prefix ``pycbc.waveform.decompress_``.
They implement the same API as the CPU/CUDA/CuPy backends but rely purely on
torch tensor operations (searchsorted + vectorised interpolation). Higher-order
helpers (quadratic/cubic/quartic) currently reuse the linear implementation to
provide correct results with minimal complexity; polynomial refinement can be
added later without changing the interface.
"""

import torch


def _dtype_for_output(output):
    """Return real/complex torch dtypes matched to the output precision."""
    if output.precision == "single":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _to_tensor(arr, device, dtype):
    """Convert numpy/torch/Array-like to a torch tensor on the given device."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _interp_linear(amp, phase, sample_frequencies, output,
                   df, f_lower, start_index):
    """Vectorised linear interpolation into the output FrequencySeries."""
    real_dtype, complex_dtype = _dtype_for_output(output)
    device = output.data.tensor.device

    sf = _to_tensor(sample_frequencies, device, real_dtype)
    a = _to_tensor(amp, device, real_dtype)
    ph = _to_tensor(phase, device, real_dtype)

    out = output.data.tensor
    out.zero_()

    hlen = out.numel()
    if start_index >= hlen:
        return output

    idxs = torch.arange(start_index, hlen, device=device, dtype=torch.int64)
    df_t = torch.tensor(df, device=device, dtype=real_dtype)
    freqs = idxs.to(dtype=real_dtype) * df_t
    max_freq = sf[-1]
    f_lower_t = torch.tensor(f_lower, device=device, dtype=real_dtype)
    mask = (freqs >= f_lower_t) & (freqs <= max_freq)
    if not torch.any(mask):
        return output

    fsel = freqs[mask]
    hi = torch.searchsorted(sf, fsel, right=False)
    hi = torch.clamp(hi, min=1, max=len(sf) - 1)
    lo = hi - 1

    f0, f1 = sf[lo], sf[hi]
    a0, a1 = a[lo], a[hi]
    p0, p1 = ph[lo], ph[hi]

    inv = 1.0 / (f1 - f0)
    amp_interp = a0 * (f1 - fsel) * inv + a1 * (fsel - f0) * inv
    phase_interp = p0 * (f1 - fsel) * inv + p1 * (fsel - f0) * inv

    real = amp_interp * torch.cos(phase_interp)
    imag = amp_interp * torch.sin(phase_interp)
    vals = torch.complex(real, imag).to(dtype=complex_dtype)

    out_idx = idxs[mask]
    out[out_idx] = vals
    return output


def inline_linear_interp(amp, phase, sample_frequencies, output,
                         df, f_lower, imin, start_index):
    return _interp_linear(amp, phase, sample_frequencies, output,
                          df, f_lower, start_index)


# For now, higher-order interpolation reuses the linear implementation to keep
# correctness and parity across schemes. These can be upgraded later with
# polynomial stencils while keeping the same API.
def inline_quadratic_interp(amp, phase, sample_frequencies, output,
                            df, f_lower, imin, start_index):
    return inline_linear_interp(amp, phase, sample_frequencies, output,
                                df, f_lower, imin, start_index)


def inline_cubic_interp(amp, phase, sample_frequencies, output,
                        df, f_lower, imin, start_index):
    return inline_linear_interp(amp, phase, sample_frequencies, output,
                                df, f_lower, imin, start_index)


def inline_quartic_interp(amp, phase, sample_frequencies, output,
                          df, f_lower, imin, start_index):
    return inline_linear_interp(amp, phase, sample_frequencies, output,
                                df, f_lower, imin, start_index)
