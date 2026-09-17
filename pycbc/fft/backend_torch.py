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
Torch FFT backend registration for the torch scheme.
"""

import pycbc

_backend_dict = {"torch": "torchfft"}
_backend_list = ["torch"]


class _LazyTorchFFT:
    def insert_fft_options(self, fft_group):
        pass

    def __getattr__(self, name):
        from pycbc.fft import torchfft
        return getattr(torchfft, name)


_alist = []
_adict = {}

if getattr(pycbc, "HAVE_TORCH", False):
    _alist = ["torch"]
    _adict = {"torch": _LazyTorchFFT()}

torch_backend = None


def set_backend(backend_list):
    global torch_backend
    for backend in backend_list:
        if backend in _alist:
            torch_backend = backend
            break


def get_backend():
    if torch_backend == "torch":
        from pycbc.fft import torchfft
        return torchfft
    return _adict[torch_backend]


set_backend(_backend_list)
