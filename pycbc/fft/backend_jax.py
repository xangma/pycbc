# Copyright (C) 2026
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

"""JAX FFT backend registration for the jax scheme."""

import importlib.util
import pycbc

_backend_dict = {"jax": "jaxfft"}
_backend_list = ["jax"]


class _LazyJAXFFT:
    def insert_fft_options(self, fft_group):
        pass

    def __getattr__(self, name):
        from pycbc.fft import jaxfft

        return getattr(jaxfft, name)


def _check_have_jax():
    if getattr(pycbc, "HAVE_JAX", False):
        return True
    try:
        return importlib.util.find_spec("jax") is not None
    except Exception:
        return False


_alist = []
_adict = {}

if _check_have_jax():
    _alist = ["jax"]
    _adict = {"jax": _LazyJAXFFT()}

jax_backend = None


def set_backend(backend_list):
    global jax_backend, _alist, _adict
    if not _alist and _check_have_jax():
        _alist = ["jax"]
        _adict = {"jax": _LazyJAXFFT()}
    for backend in backend_list:
        if backend in _alist:
            jax_backend = backend
            break


def get_backend():
    if jax_backend is None:
        set_backend(_backend_list)
    if jax_backend == "jax":
        from pycbc.fft import jaxfft

        return jaxfft
    return _adict.get(jax_backend)


set_backend(_backend_list)
