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

"""JAX FFT backend for PyCBC fast Fourier transforms."""

import functools
import numpy as np

from pycbc.types.array_jax import _ensure_x64, is_jax_array, to_jax
from .core import _BaseFFT, _BaseIFFT, _check_fft_args

_INV_FFT_MSG = (
    "I cannot perform an {} between data with an input type of "
    "{} and an output type of {}"
)


def _batched_view(vec, nbatch, dist):
    """View a flat array as its nbatch rows separated by dist."""
    data = getattr(vec, "data", vec)
    if hasattr(data, "tensor"):
        data = data.tensor
    elif hasattr(data, "_data"):
        data = data._data
    return data[: nbatch * dist].reshape(nbatch, dist)


def _assign_to_vec(outvec, result, nbatch, size, odist):
    """Assign JAX or NumPy result into destination vector data."""
    target = getattr(outvec, "data", outvec)
    narr = np.asarray(result)
    copy_len = min(size, narr.shape[-1])
    if nbatch == 1:
        target[:copy_len] = narr[0, :copy_len]
    else:
        for b in range(nbatch):
            target[b * odist : b * odist + copy_len] = narr[b, :copy_len]


# -------------------------------------------------------------------------
# Pure JAX Functional Primitives
# -------------------------------------------------------------------------

def jax_fft(in_array, n=None, axis=-1):
    """Compute 1D discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = in_array if is_jax_array(in_array) else to_jax(in_array)
    return jnp.fft.fft(jarr, n=n, axis=axis)


def jax_ifft(in_array, n=None, axis=-1, unnormalized=True):
    """Compute 1D inverse discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = in_array if is_jax_array(in_array) else to_jax(in_array)
    res = jnp.fft.ifft(jarr, n=n, axis=axis)
    if unnormalized:
        transform_len = n if n is not None else jarr.shape[axis]
        res = res * transform_len
    return res


def jax_rfft(in_array, n=None, axis=-1):
    """Compute real-input 1D discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = in_array if is_jax_array(in_array) else to_jax(in_array)
    return jnp.fft.rfft(jarr, n=n, axis=axis)


def jax_irfft(in_array, n=None, axis=-1, unnormalized=True):
    """Compute complex-input real 1D inverse discrete Fourier transform."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = in_array if is_jax_array(in_array) else to_jax(in_array)
    transform_len = n if n is not None else 2 * (jarr.shape[axis] - 1)
    res = jnp.fft.irfft(jarr, n=transform_len, axis=axis)
    if unnormalized:
        res = res * transform_len
    return res


# -------------------------------------------------------------------------
# PyCBC Scheme Functional Hook API
# -------------------------------------------------------------------------

def fft(invec, outvec, _, itype, otype):
    """Execute functional FFT into outvec."""
    if invec.ptr == outvec.ptr:
        raise NotImplementedError(
            "JAX backend of pycbc.fft does not support in-place transforms"
        )
    _ensure_x64()
    jin = to_jax(invec)
    if itype == "complex" and otype == "complex":
        res = jax_fft(jin)
    elif itype == "real" and otype == "complex":
        res = jax_rfft(jin)
    else:
        raise ValueError(_INV_FFT_MSG.format("FFT", itype, otype))

    outvec.data[:] = np.asarray(res, dtype=outvec.dtype)


def ifft(invec, outvec, _, itype, otype):
    """Execute functional IFFT into outvec."""
    if invec.ptr == outvec.ptr:
        raise NotImplementedError(
            "JAX backend of pycbc.fft does not support in-place transforms"
        )
    _ensure_x64()
    jin = to_jax(invec)
    out_len = len(outvec)
    if itype == "complex" and otype == "complex":
        res = jax_ifft(jin, n=out_len, unnormalized=True)
    elif itype == "complex" and otype == "real":
        res = jax_irfft(jin, n=out_len, unnormalized=True)
    else:
        raise ValueError(_INV_FFT_MSG.format("IFFT", itype, otype))

    outvec.data[:] = np.asarray(res, dtype=outvec.dtype)


# -------------------------------------------------------------------------
# PyCBC Class-Based API (FFT, IFFT)
# -------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _compile_jax_fwd(itype, otype, size):
    """Return a JIT-compiled forward transform."""
    import jax
    import jax.numpy as jnp

    if itype == "complex" and otype == "complex":
        return jax.jit(lambda x: jnp.fft.fft(x[:, :size], axis=-1))
    elif itype == "real" and otype == "complex":
        return jax.jit(lambda x: jnp.fft.rfft(x[:, :size], axis=-1))
    raise ValueError(_INV_FFT_MSG.format("FFT", itype, otype))


@functools.lru_cache(maxsize=32)
def _compile_jax_inv(itype, otype, idist, size):
    """Return a JIT-compiled unnormalized inverse transform."""
    import jax
    import jax.numpy as jnp

    if itype == "complex" and otype == "complex":
        return jax.jit(
            lambda x: jnp.fft.ifft(x[:, :size], axis=-1) * size
        )
    elif itype == "complex" and otype == "real":
        return jax.jit(
            lambda x: jnp.fft.irfft(x[:, :idist], n=size, axis=-1) * size
        )
    raise ValueError(_INV_FFT_MSG.format("IFFT", itype, otype))


class FFT(_BaseFFT):
    """PyCBC class-based FFT plan via JAX backend."""

    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(FFT, self).__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)
        _ensure_x64()
        self._compiled = _compile_jax_fwd(self.itype, self.otype, self.size)

    def execute(self):
        """Compute the forward FFT of the input vector."""
        if self.invec.ptr == self.outvec.ptr:
            raise NotImplementedError(
                "JAX backend of pycbc.fft does not support in-place transforms"
            )
        jin = to_jax(self.invec)
        jin_batch = jin[: self.nbatch * self.idist].reshape(
            self.nbatch, self.idist
        )
        res = self._compiled(jin_batch)
        _assign_to_vec(self.outvec, res, self.nbatch, self.size, self.odist)


class IFFT(_BaseIFFT):
    """PyCBC class-based IFFT plan via JAX backend."""

    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(IFFT, self).__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)
        _ensure_x64()
        self._compiled = _compile_jax_inv(
            self.itype, self.otype, self.idist, self.size
        )

    def execute(self):
        """Compute the unnormalized inverse FFT of the input vector."""
        if self.invec.ptr == self.outvec.ptr:
            raise NotImplementedError(
                "JAX backend of pycbc.fft does not support in-place transforms"
            )
        jin = to_jax(self.invec)
        jin_batch = jin[: self.nbatch * self.idist].reshape(
            self.nbatch, self.idist
        )
        res = self._compiled(jin_batch)
        _assign_to_vec(self.outvec, res, self.nbatch, self.size, self.odist)
