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
PyTorch FFT backend for PyCBC.

Implements the same API as numpy/fftw backends but operates on TorchArrayData
objects allocated by the torch scheme.
"""

import torch
from .core import _check_fft_args, _BaseFFT, _BaseIFFT


def _ensure_match(invec, outvec):
    if outvec._data.tensor.device != invec._data.tensor.device:
        raise ValueError("Input and output must be on the same torch device")


def _copy_result(outvec, result):
    if result.dtype != outvec._data.tensor.dtype:
        result = result.to(dtype=outvec._data.tensor.dtype)
    outvec._data.tensor.copy_(result)


def fft(invec, outvec, _, itype, otype):
    _ensure_match(invec, outvec)
    fin = invec._data.tensor
    # NOTE: PyTorch FFT kernels are not bitwise-identical to numpy/FFTW
    # in float32. In parity testing we observe rfft diffs up to ~2e-5
    # (and downstream SNR/chi^2 diffs at 1e-6 / 1e-4). If exact parity
    # is required, run the torch scheme in float64 or route through the
    # CPU FFT backend.
    if itype == 'complex' and otype == 'complex':
        res = torch.fft.fft(fin, n=fin.shape[-1])
    elif itype == 'real' and otype == 'complex':
        res = torch.fft.rfft(fin, n=fin.shape[-1])
    else:
        raise ValueError("Unsupported dtype combination for torch fft")
    _copy_result(outvec, res)


def ifft(invec, outvec, _, itype, otype):
    _ensure_match(invec, outvec)
    fin = invec._data.tensor
    n_out = outvec._data.tensor.shape[-1]
    if itype == 'complex' and otype == 'complex':
        res = torch.fft.ifft(fin, n=n_out) * n_out
    elif itype == 'complex' and otype == 'real':
        res = torch.fft.irfft(fin, n=n_out) * n_out
    else:
        raise ValueError("Unsupported dtype combination for torch ifft")
    _copy_result(outvec, res)


class FFT(_BaseFFT):
    """Class-based torch FFT."""
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super().__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)

    def execute(self):
        fft(self.invec, self.outvec, self.prec, self.itype, self.otype)


class IFFT(_BaseIFFT):
    """Class-based torch inverse FFT."""
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super().__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)

    def execute(self):
        ifft(self.invec, self.outvec, self.prec, self.itype, self.otype)
