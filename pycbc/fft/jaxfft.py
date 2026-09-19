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
from types import FunctionType

from pycbc.types import Array
from pycbc.types.aligned import zeros as _aligned_zeros
from pycbc.types.array_jax import (
    _as_jax_array,
    _ensure_x64,
    _reference_enabled,
    to_jax,
)
from .core import _BaseFFT, _BaseIFFT, _check_fft_args

_INV_FFT_MSG = (
    "I cannot perform an {} between data with an input type of "
    "{} and an output type of {}"
)


def _assign_to_vec(outvec, result):
    """Update the packed output buffer, retaining its write-through views."""
    outvec.data.set_array(result.reshape(-1).astype(outvec.dtype))


def _jax_input(value):
    """Unwrap storage without converting a JAX tracer to a host array."""
    raw = _as_jax_array(value)
    return to_jax(value) if raw is None else raw


class _HostArray(Array):
    """Aligned CPU storage for original FFT kernels, independent of scheme."""

    def __init__(self, length, dtype):
        self._data = _aligned_zeros(length, dtype)

    @property
    def data(self):
        return self._data

    @property
    def ptr(self):
        return self._data.ctypes.data

    def __imul__(self, other):
        self._data *= other
        return self


def _cpu_namespace(backend):
    """Bind only original CPU planning code to host scratch allocation.

    FFTW/MKL planning helpers call scheme-dispatched ``zeros`` even when
    handed host buffers. Rebind their unchanged code in a private namespace;
    otherwise FFTW could receive JAX object IDs as native memory addresses.
    This adapter is limited to the existing NumPy, FFTW and MKL backends.
    """
    name = backend.__name__.rsplit(".", 1)[-1]
    helpers = {"npfft": (), "fftw": ("plan", "_fftw_setup"),
               "mkl": ("create_descriptor",)}
    if name not in helpers:
        raise RuntimeError("Unsupported CPU FFT backend for JAX validation")
    namespace = vars(backend).copy()
    namespace["zeros"] = _HostArray
    for name in helpers[name]:
        function = getattr(backend, name, None)
        if not hasattr(function, "__code__"):
            raise RuntimeError("CPU FFT planning helper is unavailable")
        namespace[name] = FunctionType(function.__code__, namespace,
                                      function.__name__, function.__defaults__,
                                      function.__closure__)
    return namespace


def _cpu_function(function, namespace):
    if not hasattr(function, "__code__"):
        raise RuntimeError("CPU FFT entry point is unavailable")
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)


def _cpu_buffers(invec, outvec):
    source = _HostArray(len(invec), invec.dtype)
    target = _HostArray(len(outvec), outvec.dtype)
    source.data[:] = np.asarray(to_jax(invec))
    return source, target


def _reference_transform(operation, invec, outvec, prec, itype, otype):
    from .backend_cpu import get_backend

    backend = get_backend()
    source, target = _cpu_buffers(invec, outvec)
    execute = _cpu_function(getattr(backend, operation, None),
                            _cpu_namespace(backend))
    execute(source, target, prec, itype, otype)
    _assign_to_vec(outvec, to_jax(target.data))


def _reference_plan(operation, invec, outvec, nbatch, size):
    from .backend_cpu import get_backend

    backend = get_backend()
    source, target = _cpu_buffers(invec, outvec)
    cls = getattr(backend, operation.upper(), None)
    if not isinstance(cls, type):
        raise RuntimeError("CPU FFT plan class is unavailable")
    plan = object.__new__(cls)
    initialize = _cpu_function(cls.__init__, _cpu_namespace(backend))
    initialize(plan, source, target, nbatch=nbatch, size=size)
    return plan


# -------------------------------------------------------------------------
# Pure JAX Functional Primitives
# -------------------------------------------------------------------------

def jax_fft(in_array, n=None, axis=-1):
    """Compute 1D discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = _jax_input(in_array)
    return jnp.fft.fft(jarr, n=n, axis=axis)


def jax_ifft(in_array, n=None, axis=-1, unnormalized=True):
    """Compute 1D inverse discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = _jax_input(in_array)
    res = jnp.fft.ifft(jarr, n=n, axis=axis)
    if unnormalized:
        transform_len = n if n is not None else jarr.shape[axis]
        res = res * transform_len
    return res


def jax_rfft(in_array, n=None, axis=-1):
    """Compute real-input 1D discrete Fourier transform on JAX array."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = _jax_input(in_array)
    return jnp.fft.rfft(jarr, n=n, axis=axis)


def jax_irfft(in_array, n=None, axis=-1, unnormalized=True):
    """Compute complex-input real 1D inverse discrete Fourier transform."""
    _ensure_x64()
    import jax.numpy as jnp

    jarr = _jax_input(in_array)
    transform_len = n if n is not None else 2 * (jarr.shape[axis] - 1)
    res = jnp.fft.irfft(jarr, n=transform_len, axis=axis)
    if unnormalized:
        res = res * transform_len
    return res


# -------------------------------------------------------------------------
# PyCBC Scheme Functional Hook API
# -------------------------------------------------------------------------

def fft(invec, outvec, prec, itype, otype):
    """Execute functional FFT into outvec."""
    if invec.ptr == outvec.ptr:
        raise NotImplementedError(
            "JAX backend of pycbc.fft does not support in-place transforms"
        )
    if _reference_enabled("fft"):
        _reference_transform("fft", invec, outvec, prec, itype, otype)
        return
    _ensure_x64()
    jin = to_jax(invec)
    if itype == "complex" and otype == "complex":
        res = jax_fft(jin)
    elif itype == "real" and otype == "complex":
        res = jax_rfft(jin)
    else:
        raise ValueError(_INV_FFT_MSG.format("FFT", itype, otype))

    _assign_to_vec(outvec, res)


def ifft(invec, outvec, prec, itype, otype):
    """Execute functional IFFT into outvec."""
    if invec.ptr == outvec.ptr:
        raise NotImplementedError(
            "JAX backend of pycbc.fft does not support in-place transforms"
        )
    if _reference_enabled("ifft"):
        _reference_transform("ifft", invec, outvec, prec, itype, otype)
        return
    _ensure_x64()
    jin = to_jax(invec)
    out_len = len(outvec)
    if itype == "complex" and otype == "complex":
        res = jax_ifft(jin, n=out_len, unnormalized=True)
    elif itype == "complex" and otype == "real":
        res = jax_irfft(jin, n=out_len, unnormalized=True)
    else:
        raise ValueError(_INV_FFT_MSG.format("IFFT", itype, otype))

    _assign_to_vec(outvec, res)


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
        self._reference = None
        if _reference_enabled("fft"):
            self._reference = _reference_plan("fft", invec, outvec, nbatch, size)
            return
        self._compiled = _compile_jax_fwd(self.itype, self.otype, self.size)

    def execute(self):
        """Compute the forward FFT of the input vector."""
        if self.invec.ptr == self.outvec.ptr:
            raise NotImplementedError(
                "JAX backend of pycbc.fft does not support in-place transforms"
            )
        if self._reference is not None:
            self._reference.invec.data[:] = np.asarray(to_jax(self.invec))
            self._reference.execute()
            # CPU device_put may borrow NumPy storage. This host scratch
            # buffer is reused by the next execution, so publish a snapshot.
            _assign_to_vec(self.outvec,
                           to_jax(self._reference.outvec.data.copy()))
            return
        jin = to_jax(self.invec)
        jin_batch = jin[: self.nbatch * self.idist].reshape(
            self.nbatch, self.idist
        )
        res = self._compiled(jin_batch)
        _assign_to_vec(self.outvec, res)


class IFFT(_BaseIFFT):
    """PyCBC class-based IFFT plan via JAX backend."""

    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(IFFT, self).__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)
        _ensure_x64()
        self._reference = None
        if _reference_enabled("ifft"):
            self._reference = _reference_plan("ifft", invec, outvec, nbatch, size)
            return
        self._compiled = _compile_jax_inv(
            self.itype, self.otype, self.idist, self.size
        )

    def execute(self):
        """Compute the unnormalized inverse FFT of the input vector."""
        if self.invec.ptr == self.outvec.ptr:
            raise NotImplementedError(
                "JAX backend of pycbc.fft does not support in-place transforms"
            )
        if self._reference is not None:
            self._reference.invec.data[:] = np.asarray(to_jax(self.invec))
            self._reference.execute()
            # CPU device_put may borrow NumPy storage. This host scratch
            # buffer is reused by the next execution, so publish a snapshot.
            _assign_to_vec(self.outvec,
                           to_jax(self._reference.outvec.data.copy()))
            return
        jin = to_jax(self.invec)
        jin_batch = jin[: self.nbatch * self.idist].reshape(
            self.nbatch, self.idist
        )
        res = self._compiled(jin_batch)
        _assign_to_vec(self.outvec, res)
