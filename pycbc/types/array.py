# Copyright (C) 2012  Alex Nitz, Josh Willis, Andrew Miller, Tito Dal Canton
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This modules provides a device independent Array class based on PyCUDA and Numpy.
"""

BACKEND_PREFIX="pycbc.types.array_"

import os as _os
import operator as _operator
import warnings as _warnings

from functools import wraps

import h5py
import numpy as _numpy
from numpy import float32, float64, complex64, complex128, ones
from numpy.linalg import norm

import pycbc.scheme as _scheme
from pycbc import lal_compat as _lal_compat
from pycbc.scheme import schemed, cpuonly
from pycbc.opt import LimitedSizeDict

_NUMPY_TRAPEZOID = getattr(_numpy, "trapezoid", None)
if _NUMPY_TRAPEZOID is None:
    _NUMPY_TRAPEZOID = _numpy.trapz

#! FIXME: the uint32 datatype has not been fully tested,
# we should restrict any functions that do not allow an
# array of uint32 integers
_ALLOWED_DTYPES = [_numpy.bool_, _numpy.float32, _numpy.float64,
                   _numpy.complex64, _numpy.complex128, _numpy.uint32,
                   _numpy.int32, int]
try:
    _ALLOWED_SCALARS = [int, long, float, complex] + _ALLOWED_DTYPES
except NameError:
    _ALLOWED_SCALARS = [int, float, complex] + _ALLOWED_DTYPES

def _convert_to_scheme(ary):
    if ary._scheme is _scheme.mgr.state:
        return
    if not isinstance(ary._scheme, _scheme.mgr.state.__class__):
        converted_array = Array(ary, dtype=ary._data.dtype)
        ary._data = converted_array._data
        ary._saved = None
        ary._scheme = _scheme.mgr.state



def _array_function_arrays(value):
    """Yield PyCBC arrays nested in a NumPy array-function argument."""
    if isinstance(value, Array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _array_function_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _array_function_arrays(item)


def _array_function_backend_value(value):
    """Replace nested PyCBC arrays with their active backend storage."""
    if isinstance(value, Array):
        return value._data
    if isinstance(value, tuple):
        return tuple(_array_function_backend_value(item) for item in value)
    if isinstance(value, list):
        return [_array_function_backend_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _array_function_backend_value(item)
            for key, item in value.items()
        }
    return value


def _array_function_numpy_value(value):
    """Replace nested PyCBC arrays with legacy NumPy storage."""
    if isinstance(value, Array):
        return value.numpy()
    if isinstance(value, tuple):
        return tuple(_array_function_numpy_value(item) for item in value)
    if isinstance(value, list):
        return [_array_function_numpy_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _array_function_numpy_value(item)
            for key, item in value.items()
        }
    return value


_UNSUPPORTED_REDUCTION_AXES = object()


def _normalized_reduction_axes(shape, axis):
    """Return normalized reduction axes or an unsupported sentinel."""
    ndim = len(shape)
    if axis is None:
        return tuple(range(ndim))
    raw_axes = axis if isinstance(axis, tuple) else (axis,)
    axes = []
    try:
        for raw_axis in raw_axes:
            normalized = _operator.index(raw_axis)
            if normalized < 0:
                normalized += ndim
            if normalized < 0 or normalized >= ndim:
                return _UNSUPPORTED_REDUCTION_AXES
            if normalized in axes:
                return _UNSUPPORTED_REDUCTION_AXES
            axes.append(normalized)
    except TypeError:
        return _UNSUPPORTED_REDUCTION_AXES
    return tuple(axes)

def _convert(func):
    @wraps(func)
    def convert(self, *args, **kwargs):
        _convert_to_scheme(self)
        return func(self, *args, **kwargs)
    return convert
    
def _nocomplex(func):
    @wraps(func)
    def nocomplex(self, *args, **kwargs):
        if self.kind == 'real':
            return func(self, *args, **kwargs)
        else:
            raise TypeError( func.__name__ + " does not support complex types")
    return nocomplex

def _noreal(func):
    @wraps(func)
    def noreal(self, *args, **kwargs):
        if self.kind == 'complex':
            return func(self, *args, **kwargs)
        else:
            raise TypeError( func.__name__ + " does not support real types")
    return noreal

def force_precision_to_match(scalar, precision):
    if _numpy.iscomplexobj(scalar):
        if precision == 'single':
            return _numpy.complex64(scalar)
        else:
            return _numpy.complex128(scalar)
    else:
        if precision == 'single':
            return _numpy.float32(scalar)
        else:
            return _numpy.float64(scalar)

def common_kind(*dtypes):
    for dtype in dtypes:
        if dtype.kind == 'c':
            return dtype
    return dtypes[0]
   
@schemed(BACKEND_PREFIX) 
def _to_device(array):
    """ Move input to device """
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)
    
@schemed(BACKEND_PREFIX)
def _copy_base_array(array):
    """ Copy a backend array"""
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

@schemed(BACKEND_PREFIX)
def _scheme_matches_base_array(array):
    """ Check that input matches array type for scheme """
    err_msg = "This function is a stub that should be overridden using the "
    err_msg += "scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

def check_same_len_precision(a, b):
    """Check that the two arguments have the same length and precision.
    Raises ValueError if they do not.
    """
    if len(a) != len(b):
        msg = 'lengths do not match ({} vs {})'.format(
                len(a), len(b))
        raise ValueError(msg)
    if a.precision != b.precision:
        msg = 'precisions do not match ({} vs {})'.format(
                a.precision, b.precision)
        raise TypeError(msg)

class Array(object):
    """Array used to do numeric calculations on a various compute
    devices. It is a convience wrapper around numpy, and
    pycuda.
    """

    def __init__(self, initial_array, dtype=None, copy=True):
        """ initial_array: An array-like object as specified by NumPy, this
        also includes instances of an underlying data type as described in
        section 3 or an instance of the PYCBC Array class itself. This
        object is used to populate the data of the array.

        dtype: A NumPy style dtype that describes the type of
        encapsulated data (float32,compex64, etc)

        copy: This defines whether the initial_array is copied to instantiate
        the array or is simply referenced. If copy is false, new data is not
        created, and so all arguments that would force a copy are ignored.
        The default is to copy the given object.
        """
        self._scheme=_scheme.mgr.state
        # Most short-lived Arrays are never sliced.  Allocate the bounded
        # slice cache only when the first slice is successfully produced.
        self._saved = None
        
        #Unwrap initial_array
        if isinstance(initial_array, Array):
            initial_array = initial_array._data

        if not copy:
            if not _scheme_matches_base_array(initial_array):
                raise TypeError("Cannot avoid a copy of this array")
            else:
                self._data = initial_array

            # Check that the dtype is supported.
            if self._data.dtype not in _ALLOWED_DTYPES:
                raise TypeError(str(self._data.dtype) + ' is not supported')

            if dtype and dtype != self._data.dtype:
                raise TypeError("Can only set dtype when allowed to copy data")


        if copy:
            # First we will check the dtype that we are given
            if not hasattr(initial_array, 'dtype'):
                initial_array = _numpy.array(initial_array)

            # Determine the dtype to use
            if dtype is not None:  
                dtype = _numpy.dtype(dtype)
                if dtype not in _ALLOWED_DTYPES:
                    raise TypeError(str(dtype) + ' is not supported')
                if dtype.kind != 'c' and initial_array.dtype.kind == 'c':
                    raise TypeError(str(initial_array.dtype) + ' cannot be cast as ' + str(dtype))          
            elif initial_array.dtype in _ALLOWED_DTYPES:
                dtype = initial_array.dtype
            else:
                if initial_array.dtype.kind == 'c':
                    dtype = complex128
                else:
                    dtype = float64
                     
            # Cast to the final dtype if needed
            if initial_array.dtype != dtype:
                initial_array = initial_array.astype(dtype)
                                              
            #Create new instance with initial_array as initialization.
            if issubclass(type(self._scheme), _scheme.CPUScheme):
                if hasattr(initial_array, 'get'):
                    self._data = _numpy.array(initial_array.get())
                else:
                    self._data = _numpy.array(initial_array, dtype=dtype, ndmin=1)
            elif _scheme_matches_base_array(initial_array):
                self._data = _copy_base_array(initial_array) # pylint:disable=assignment-from-no-return
            else:
                initial_array = _numpy.array(initial_array, dtype=dtype, ndmin=1)
                self._data = _to_device(initial_array) # pylint:disable=assignment-from-no-return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        active_scheme = _scheme.mgr.state
        backend_ufunc = None
        if self._scheme is active_scheme:
            backend_ufunc = getattr(self._data, "numpy_ufunc", None)
        if backend_ufunc is not None and all(
                not isinstance(value, Array)
                or value._scheme is active_scheme
                for value in inputs):
            backend_inputs = [
                value._data if isinstance(value, Array) else value
                for value in inputs
            ]
            ret = backend_ufunc(
                ufunc, method, *backend_inputs, **kwargs
            )
            if ret is not NotImplemented:
                if hasattr(ret, 'shape') and ret.shape == self.shape:
                    ret = self._return(ret)
                elif isinstance(ret, type(self._data)):
                    ret = Array(ret, copy=False)
                return ret

        inputs = [i.numpy() if isinstance(i, Array) else i for i in inputs]
        ret = getattr(ufunc, method)(*inputs, **kwargs)
        if hasattr(ret, 'shape') and ret.shape == self.shape:
            if _numpy.dtype(ret.dtype) in _ALLOWED_DTYPES:
                if not _scheme_matches_base_array(ret):
                    ret = Array(ret)
                ret = self._return(ret)
        return ret

    def __array_function__(self, func, types, args, kwargs):
        """Keep supported NumPy array functions on the active backend."""
        if not all(
                issubclass(array_type, (Array, _numpy.ndarray))
                for array_type in types):
            return NotImplemented

        if func in (
                _numpy.concatenate,
                _numpy.argwhere,
                _numpy.stack,
                _numpy.hstack,
                _numpy.vstack,
                _numpy.dstack,
                _numpy.column_stack,
                _numpy.copy,
                _numpy.empty_like,
                _numpy.zeros_like,
                _numpy.ones_like,
                _numpy.full_like,
                _numpy.take_along_axis,
                _numpy.tile,
                _numpy.append,
                _numpy.delete,
                _numpy.putmask,
                _numpy.resize,
                _numpy.real,
                _numpy.imag,
                _numpy.angle,
                _numpy.unwrap,
                _numpy.compress,
                _numpy.count_nonzero,
                _numpy.average,
                _numpy.diag,
                _numpy.diagflat,
                _numpy.diagonal,
                _numpy.trace,
                _NUMPY_TRAPEZOID,
                _numpy.diff,
                _numpy.digitize,
                _numpy.dot,
                _numpy.expand_dims,
                _numpy.extract,
                _numpy.flip,
                _numpy.flipud,
                _numpy.fliplr,
                _numpy.flatnonzero,
                _numpy.inner,
                _numpy.histogram,
                _numpy.interp,
                _numpy.intersect1d,
                _numpy.setdiff1d,
                _numpy.setxor1d,
                _numpy.union1d,
                _numpy.isin,
                _numpy.isclose,
                _numpy.allclose,
                _numpy.array_equal,
                _numpy.array_equiv,
                _numpy.atleast_1d,
                _numpy.atleast_2d,
                _numpy.atleast_3d,
                _numpy.broadcast_arrays,
                _numpy.broadcast_to,
                _numpy.linalg.norm,
                _numpy.median,
                _numpy.moveaxis,
                _numpy.outer,
                _numpy.pad,
                _numpy.ptp,
                _numpy.ravel,
                _numpy.roll,
                _numpy.rollaxis,
                _numpy.rot90,
                _numpy.sort,
                _numpy.tril,
                _numpy.triu,
                _numpy.unique,
                _numpy.vdot,
                _numpy.where,
        ):
            active_scheme = _scheme.mgr.state
            arrays = list(_array_function_arrays((args, kwargs)))
            backend_function = getattr(
                self._data, "numpy_array_function", None
            )
            if (
                    arrays
                    and backend_function is not None
                    and all(array._scheme is active_scheme for array in arrays)
            ):
                backend_args = _array_function_backend_value(args)
                backend_kwargs = _array_function_backend_value(kwargs)
                result = backend_function(
                    func, *backend_args, **backend_kwargs
                )
                if result is not NotImplemented:
                    if func is _numpy.putmask and result is None:
                        return None
                    if func in (
                            _numpy.allclose,
                            _numpy.array_equal,
                            _numpy.array_equiv):
                        return result
                    if (
                            func is _numpy.count_nonzero
                            and _numpy.isscalar(result)):
                        return result
                    if func is _numpy.average:
                        def wrap_average(value):
                            if _numpy.isscalar(value):
                                return value
                            return Array(value, copy=False)

                        if isinstance(result, tuple):
                            return tuple(
                                wrap_average(value) for value in result
                            )
                        return wrap_average(result)
                    if (
                            func is _numpy.digitize
                            and _numpy.isscalar(result)):
                        return result
                    if (
                            func is _numpy.interp
                            and _numpy.isscalar(result)):
                        return result
                    if (
                            func is _NUMPY_TRAPEZOID
                            and _numpy.isscalar(result)):
                        return result
                    if func in (_numpy.median, _numpy.ptp):
                        out = kwargs.get("out")
                        if len(args) > 2:
                            out = args[2]
                        if isinstance(out, Array) and result is out._data:
                            return out
                        if isinstance(out, _numpy.ndarray) and result is out:
                            return out
                        if _numpy.isscalar(result):
                            return result
                    if func is _numpy.trace:
                        out = kwargs.get("out")
                        if len(args) > 5:
                            out = args[5]
                        if isinstance(out, Array) and result is out._data:
                            return out
                        if isinstance(out, _numpy.ndarray) and result is out:
                            return out
                        if _numpy.isscalar(result):
                            return result
                    if (
                            func is _numpy.linalg.norm
                            and _numpy.isscalar(result)):
                        return result
                    if func in (
                            _numpy.dot,
                            _numpy.inner,
                            _numpy.outer,
                            _numpy.vdot,
                    ) and _numpy.isscalar(result):
                        return result
                    if func in (
                            _numpy.histogram,
                            _numpy.intersect1d,
                            _numpy.setdiff1d,
                            _numpy.setxor1d,
                            _numpy.union1d,
                            _numpy.unique):
                        if isinstance(result, tuple):
                            return tuple(
                                Array(value, copy=False) for value in result
                            )
                        return Array(result, copy=False)
                    if func in (_numpy.concatenate, _numpy.stack):
                        out = kwargs.get("out")
                        if len(args) > 2:
                            out = args[2]
                        if isinstance(out, Array) and result is out._data:
                            return out
                    if func is _numpy.diff and result is self._data:
                        return self
                    if func in (
                            _numpy.atleast_1d,
                            _numpy.atleast_2d,
                            _numpy.atleast_3d):
                        if isinstance(result, tuple):
                            return tuple(
                                original
                                if isinstance(original, Array)
                                and value is original._data
                                else Array(value, copy=False)
                                for original, value in zip(args, result)
                            )
                        if (
                                len(args) == 1
                                and isinstance(args[0], Array)
                                and result is args[0]._data):
                            return args[0]
                    if func is _numpy.broadcast_arrays:
                        return tuple(
                            original
                            if isinstance(original, Array)
                            and value is original._data
                            else Array(value, copy=False)
                            for original, value in zip(args, result)
                        )
                    return Array(result, copy=False)
                if func in (
                        _numpy.histogram,
                        _numpy.setdiff1d,
                        _numpy.setxor1d):
                    implementation = getattr(func, "_implementation", None)
                    if implementation is not None:
                        return implementation(
                            *_array_function_numpy_value(args),
                            **_array_function_numpy_value(kwargs),
                        )

        # NumPy's private implementation bypasses this protocol and preserves
        # the legacy Array method/host fallback for every unsupported function.
        implementation = getattr(func, "_implementation", None)
        if implementation is None:
            return NotImplemented
        return implementation(*args, **kwargs)

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    @property
    def shape(self):
        return self._data.shape

    @_convert
    def reshape(self, *shape, order='C', copy=None):
        """Return a reshaped plain :class:`Array`.

        NumPy dispatches ``numpy.reshape(Array, ...)`` through this method.
        Backends may therefore preserve device residency for C-order shapes;
        unsupported layout requests retain the legacy host-array behavior.
        """
        newshape = shape[0] if len(shape) == 1 else shape
        backend_reshape = getattr(self._data, "numpy_reshape", None)
        if backend_reshape is not None:
            result = backend_reshape(newshape, order=order, copy=copy)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.reshape(
            self.numpy(), newshape, order=order, copy=copy
        )

    @_convert
    def transpose(self, *axes):
        """Permute dimensions without preserving series metadata."""
        requested_axes = None
        if axes:
            requested_axes = axes[0] if len(axes) == 1 else axes

        backend_transpose = getattr(self._data, "numpy_transpose", None)
        if backend_transpose is not None:
            result = backend_transpose(requested_axes)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.transpose(self.numpy(), requested_axes)

    @property
    def T(self):
        """Return the array with its dimensions reversed."""
        return self.transpose()

    @_convert
    def swapaxes(self, axis1, axis2):
        """Interchange two dimensions without preserving series metadata."""
        backend_swapaxes = getattr(self._data, "numpy_swapaxes", None)
        if backend_swapaxes is not None:
            result = backend_swapaxes(axis1, axis2)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.swapaxes(self.numpy(), axis1, axis2)

    @_convert
    def squeeze(self, axis=None):
        """Remove length-one dimensions without preserving metadata."""
        backend_squeeze = getattr(self._data, "numpy_squeeze", None)
        if backend_squeeze is not None:
            result = backend_squeeze(axis)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.squeeze(self.numpy(), axis=axis)

    @_convert
    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Return a diagonal view without preserving series metadata."""
        backend_diagonal = getattr(self._data, "numpy_diagonal", None)
        if backend_diagonal is not None:
            result = backend_diagonal(offset, axis1, axis2)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.diagonal(
            self.numpy(), offset=offset, axis1=axis1, axis2=axis2
        )

    @_convert
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Return the sum along diagonals of the array."""
        return _numpy.trace(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            dtype=dtype,
            out=out,
        )

    @_convert
    def ravel(self, order='C'):
        """Return a flattened view when the backend can provide one."""
        backend_ravel = getattr(self._data, "numpy_ravel", None)
        if backend_ravel is not None:
            result = backend_ravel(order=order)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return _numpy.ravel(self.numpy(), order=order)

    @_convert
    def flatten(self, order='C'):
        """Return an independent flattened copy of the array."""
        backend_flatten = getattr(self._data, "numpy_flatten", None)
        if backend_flatten is not None:
            result = backend_flatten(order=order)
            if result is not NotImplemented:
                return Array(result, copy=False)

        return self.numpy().flatten(order=order)
     
    def _memoize_single(func):
        @wraps(func)
        def memoize_single(self, arg):
            badh = str(arg)
            saved = self._saved

            if saved is not None and badh in saved:
                return saved[badh]

            res = func(self, arg) # pylint:disable=not-callable
            saved = self._saved
            if saved is None:
                saved = LimitedSizeDict(size_limit=2**5)
                self._saved = saved
            saved[badh] = res
            return res
        return memoize_single

    def _returnarray(func):
        @wraps(func)
        def returnarray(self, *args, **kwargs):
            return Array(func(self, *args, **kwargs), copy=False) # pylint:disable=not-callable
        return returnarray

    def _returntype(func):
        @wraps(func)
        def returntype(self, *args, **kwargs):
            ary = func(self, *args, **kwargs) # pylint:disable=not-callable
            if ary is NotImplemented:
                return NotImplemented
            return self._return(ary)
        return returntype
        
    def _return(self, ary):
        """Wrap the ary to return an Array type """
        if isinstance(ary, Array):
            return ary
        if _scheme_matches_base_array(ary):
            return Array(ary, copy=False)
        return Array(ary)

    def _checkother(func):
        @wraps(func)
        def checkother(self, *args):
            nargs = ()
            for other in args:
                self._typecheck(other)
                if type(other) in _ALLOWED_SCALARS:
                    other = force_precision_to_match(other, self.precision)
                    nargs +=(other,)
                elif isinstance(other, type(self)) or type(other) is Array:
                    check_same_len_precision(self, other)
                    _convert_to_scheme(other)
                    nargs += (other._data,)
                else:
                    return NotImplemented

            return func(self, *nargs) # pylint:disable=not-callable
        return checkother

    def _vcheckother(func):
        @wraps(func)
        def vcheckother(self, *args):
            nargs = ()
            for other in args:
                self._typecheck(other)
                if isinstance(other, type(self)) or type(other) is Array:
                    check_same_len_precision(self, other)
                    _convert_to_scheme(other)
                    nargs += (other._data,)
                else:
                    raise TypeError('array argument required')

            return func(self, *nargs) # pylint:disable=not-callable
        return vcheckother
        
    def _vrcheckother(func):
        @wraps(func)
        def vrcheckother(self, *args):
            nargs = ()
            for other in args:
                if isinstance(other, type(self)) or type(other) is Array:
                    check_same_len_precision(self, other)
                    _convert_to_scheme(other)
                    nargs += (other._data,)
                else:
                    raise TypeError('array argument required')

            return func(self, *nargs) # pylint:disable=not-callable
        return vrcheckother

    def _icheckother(func):
        @wraps(func)
        def icheckother(self, other):
            """ Checks the input to in-place operations """
            self._typecheck(other)
            if type(other) in _ALLOWED_SCALARS:
                if self.kind == 'real' and type(other) == complex:
                    raise TypeError('dtypes are incompatible')
                # Preserve integer and boolean scalars for integer-like
                # arrays.  Coercing (for example) ``1`` to float64 makes an
                # otherwise valid ``int64_array += 1`` fail on Torch and does
                # not match NumPy's in-place casting rules.
                if self.kind in ('real', 'complex'):
                    other = force_precision_to_match(other, self.precision)
            elif isinstance(other, type(self)) or type(other) is Array:
                check_same_len_precision(self, other)
                if self.kind == 'real' and other.kind == 'complex':
                    raise TypeError('dtypes are incompatible')
                _convert_to_scheme(other)
                other = other._data
            else:
                return NotImplemented

            return func(self, other) # pylint:disable=not-callable
        return icheckother

    def _typecheck(self, other):
        """ Additional typechecking for other. Placeholder for use by derived
        types. 
        """
        pass

    @_returntype
    @_convert
    @_checkother
    def __mul__(self,other):
        """ Multiply by an Array or a scalar and return an Array. """
        return self._data * other

    __rmul__ = __mul__

    @_convert
    @_icheckother
    def __imul__(self,other):
        """ Multiply by an Array or a scalar and return an Array. """
        self._data *= other
        return self

    @_returntype
    @_convert
    @_checkother
    def __add__(self,other):
        """ Add Array to Array or scalar and return an Array. """
        return self._data + other

    __radd__ = __add__
       
    def fill(self, value):
        self._data.fill(value)

    @_convert
    @_icheckother
    def __iadd__(self,other):
        """ Add Array to Array or scalar and return an Array. """
        self._data += other
        return self

    @_convert
    @_checkother
    @_returntype
    def __truediv__(self,other):
        """ Divide Array by Array or scalar and return an Array. """
        return self._data / other

    @_returntype
    @_convert
    @_checkother
    def __rtruediv__(self,other):
        """ Divide Array by Array or scalar and return an Array. """
        return self._data.__rtruediv__(other)

    @_convert
    @_icheckother
    def __itruediv__(self,other):
        """ Divide Array by Array or scalar and return an Array. """
        self._data /= other
        return self
        
    __div__ = __truediv__
    __idiv__ = __itruediv__
    __rdiv__ = __rtruediv__

    @_returntype
    @_convert
    def __neg__(self):
        """ Return negation of self """
        return - self._data

    @_returntype
    @_convert
    @_checkother
    def __sub__(self,other):
        """ Subtract Array or scalar from Array and return an Array. """
        return self._data - other

    @_returntype
    @_convert
    @_checkother
    def __rsub__(self,other):
        """ Subtract Array or scalar from Array and return an Array. """
        return self._data.__rsub__(other)

    @_convert
    @_icheckother
    def __isub__(self,other):
        """ Subtract Array or scalar from Array and return an Array. """
        self._data -= other
        return self

    @_returntype
    @_convert
    @_checkother
    def __pow__(self,other):
        """ Exponentiate Array by scalar """
        return self._data ** other

    @_returntype
    @_convert
    def __abs__(self):
        """ Return absolute value of Array """
        return abs(self._data)

    def __invert__(self):
        """Return the elementwise bitwise inverse of the array."""
        return _numpy.invert(self)

    def __and__(self, other):
        """Return the elementwise bitwise AND of two operands."""
        return _numpy.bitwise_and(self, other)

    def __rand__(self, other):
        """Return the reflected elementwise bitwise AND."""
        return _numpy.bitwise_and(other, self)

    def __or__(self, other):
        """Return the elementwise bitwise OR of two operands."""
        return _numpy.bitwise_or(self, other)

    def __ror__(self, other):
        """Return the reflected elementwise bitwise OR."""
        return _numpy.bitwise_or(other, self)

    def __xor__(self, other):
        """Return the elementwise bitwise XOR of two operands."""
        return _numpy.bitwise_xor(self, other)

    def __rxor__(self, other):
        """Return the reflected elementwise bitwise XOR."""
        return _numpy.bitwise_xor(other, self)

    def __len__(self):
        """ Return length of Array """
        return len(self._data)

    def __str__(self):
        return str(self._data)
        
    @property
    def ndim(self):
        return self._data.ndim

    def __eq__(self,other):
        """
        This is the Python special method invoked whenever the '=='
        comparison is used.  It will return true if the data of two
        PyCBC arrays are identical, and all of the numeric meta-data
        are identical, irrespective of whether or not the two
        instances live in the same memory (for that comparison, the
        Python statement 'a is b' should be used instead).

        Thus, this method returns 'True' if the types of both 'self'
        and 'other' are identical, as well as their lengths, dtypes
        and the data in the arrays, element by element. Same-device
        Torch arrays are reduced on their device, synchronizing only the
        final boolean. Mixed backends retain the CPU comparison path.
        Neither object is relocated nor has its scheme changed.

        Note in particular that this function returns a single boolean,
        and not an array of booleans as Numpy does.  If the numpy
        behavior is instead desired it can be obtained using the numpy()
        method of the PyCBC type to get a numpy instance from each
        object, and invoking '==' on those two instances.

        Parameters
        ----------
        other: another Python object, that should be tested for equality
            with 'self'.

        Returns
        -------
        boolean: 'True' if the types, dtypes, lengths, and data of the
            two objects are each identical.
        """

        # Writing the first test as below allows this method to be safely
        # called from subclasses.
        if type(self) != type(other):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self) != len(other):
            return False

        backend_equal = getattr(self._data, "array_equal", None)
        if backend_equal is not None:
            result = backend_equal(other._data)
            if result is not NotImplemented:
                return result

        sary = self.numpy()
        oary = other.numpy()

        # Now we know that both sary and oary are numpy arrays. The
        # '==' statement returns an array of booleans, and the all()
        # method of that array returns 'True' only if every element
        # of that array of booleans is True.
        return (sary == oary).all()

    def almost_equal_elem(self,other,tol,relative=True):
        """
        Compare whether two array types are almost equal, element
        by element.

        If the 'relative' parameter is 'True' (the default) then the
        'tol' parameter (which must be positive) is interpreted as a
        relative tolerance, and the comparison returns 'True' only if
        abs(self[i]-other[i]) <= tol*abs(self[i])
        for all elements of the array.

        If 'relative' is 'False', then 'tol' is an absolute tolerance,
        and the comparison is true only if
        abs(self[i]-other[i]) <= tol
        for all elements of the array.

        Other meta-data (type, dtype, and length) must be exactly equal.
        Same-device Torch arrays are reduced on their device,
        synchronizing only the final boolean. Mixed backends retain the
        CPU comparison path. Neither object is relocated nor has its
        scheme changed.

        Parameters
        ----------
        other
            Another Python object, that should be tested for
            almost-equality with 'self', element-by-element.
        tol
            A non-negative number, the tolerance, which is interpreted
            as either a relative tolerance (the default) or an absolute
            tolerance.
        relative
            A boolean, indicating whether 'tol' should be interpreted
            as a relative tolerance (if True, the default if this argument
            is omitted) or as an absolute tolerance (if tol is False).

        Returns
        -------
        boolean 
            'True' if the data agree within the tolerance, as
            interpreted by the 'relative' keyword, and if the types,
            lengths, and dtypes are exactly the same.
        """
        # Check that the tolerance is non-negative and raise an
        # exception otherwise.
        if (tol<0):
            raise ValueError("Tolerance cannot be negative")
        # Check that the meta-data agree; the type check is written in
        # this way so that this method may be safely called from
        # subclasses as well.
        if type(other) != type(self):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self) != len(other):
            return False

        backend_comparison = getattr(self._data, "almost_equal_elem", None)
        if backend_comparison is not None:
            result = backend_comparison(other._data, tol, relative)
            if result is not NotImplemented:
                return result

        diff = abs(self.numpy()-other.numpy())
        if relative:
            cmpary = tol*abs(self.numpy())
        else:
            cmpary = tol*ones(len(self),dtype=self.dtype)

        return (diff<=cmpary).all()

    def almost_equal_norm(self,other,tol,relative=True):
        """
        Compare whether two array types are almost equal, normwise.

        If the 'relative' parameter is 'True' (the default) then the
        'tol' parameter (which must be positive) is interpreted as a
        relative tolerance, and the comparison returns 'True' only if
        abs(norm(self-other)) <= tol*abs(norm(self)).

        If 'relative' is 'False', then 'tol' is an absolute tolerance,
        and the comparison is true only if
        abs(norm(self-other)) <= tol

        Other meta-data (type, dtype, and length) must be exactly equal.
        Same-device Torch arrays are reduced on their device,
        synchronizing only the final boolean. Mixed backends retain the
        CPU comparison path. Neither object is relocated nor has its
        scheme changed.

        Parameters
        ----------
        other
            another Python object, that should be tested for
            almost-equality with 'self', based on their norms.
        tol 
            a non-negative number, the tolerance, which is interpreted
            as either a relative tolerance (the default) or an absolute
            tolerance.
        relative
            A boolean, indicating whether 'tol' should be interpreted
            as a relative tolerance (if True, the default if this argument
            is omitted) or as an absolute tolerance (if tol is False).

        Returns
        -------
        boolean
            'True' if the data agree within the tolerance, as
            interpreted by the 'relative' keyword, and if the types,
            lengths, and dtypes are exactly the same.
        """
        # Check that the tolerance is non-negative and raise an
        # exception otherwise.
        if (tol<0):
            raise ValueError("Tolerance cannot be negative")
        # Check that the meta-data agree; the type check is written in
        # this way so that this method may be safely called from
        # subclasses as well.
        if type(other) != type(self):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self) != len(other):
            return False

        backend_comparison = getattr(self._data, "almost_equal_norm", None)
        if backend_comparison is not None:
            result = backend_comparison(other._data, tol, relative)
            if result is not NotImplemented:
                return result

        diff = self.numpy()-other.numpy()
        dnorm = norm(diff)
        if relative:
            return (dnorm <= tol*norm(self))
        else:
            return (dnorm <= tol)

    @_returntype
    @_convert
    def real(self):
        """ Return real part of Array """
        return Array(self._data.real, copy=True)

    @_returntype
    @_convert
    def imag(self):
        """ Return imaginary part of Array """
        return Array(self._data.imag, copy=True)

    @_returntype
    @_convert
    def conj(self):
        """ Return complex conjugate of Array. """
        return self._data.conj()
        
    @_returntype
    @_convert
    @schemed(BACKEND_PREFIX)
    def squared_norm(self):
        """ Return the elementwise squared norm of the array """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_returntype
    @_checkother
    @_convert
    @schemed(BACKEND_PREFIX)
    def multiply_and_add(self, other, mult_fac):
        """ Return other multiplied by mult_fac and with self added.
        Self is modified in place and returned as output.
        Precisions of inputs must match.
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_vrcheckother
    @_convert
    @schemed(BACKEND_PREFIX)
    def inner(self, other):
        """ Return the inner product of the array with complex conjugation.
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_vrcheckother
    @_convert
    @schemed(BACKEND_PREFIX)
    def vdot(self, other):
        """ Return the inner product of the array with complex conjugation.
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_convert
    @schemed(BACKEND_PREFIX)
    def clear(self): 
        """ Clear out the values of the array. """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_vrcheckother
    @_convert
    @schemed(BACKEND_PREFIX)
    def weighted_inner(self, other, weight):
        """ Return the inner product of the array with complex conjugation.
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_convert
    @schemed(BACKEND_PREFIX)
    def sum(self):
        """ Return the sum of the the array. """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    _scheme_sum = sum

    def sum(self, *args, **kwargs):
        """Return a legacy or NumPy-compatible sum of the array."""
        if not args and not kwargs:
            return self._scheme_sum()
        return _numpy.add.reduce(self, *args, **kwargs)

    def prod(
            self, axis=None, dtype=None, out=None, keepdims=False,
            initial=_numpy._NoValue, where=_numpy._NoValue):
        """Return the product with NumPy-compatible options."""
        kwargs = {
            "axis": axis,
            "dtype": dtype,
            "out": out,
            "keepdims": keepdims,
        }
        if initial is not _numpy._NoValue:
            kwargs["initial"] = initial
        if where is not _numpy._NoValue:
            kwargs["where"] = where
        return _numpy.multiply.reduce(self, **kwargs)

    def any(
            self, axis=None, out=None, keepdims=False, *,
            where=_numpy._NoValue):
        """Return whether any array element evaluates to true."""
        kwargs = {
            "axis": axis,
            "out": out,
            "keepdims": keepdims,
        }
        if where is not _numpy._NoValue:
            kwargs["where"] = where
        return _numpy.logical_or.reduce(self, **kwargs)

    def all(
            self, axis=None, out=None, keepdims=False, *,
            where=_numpy._NoValue):
        """Return whether all array elements evaluate to true."""
        kwargs = {
            "axis": axis,
            "out": out,
            "keepdims": keepdims,
        }
        if where is not _numpy._NoValue:
            kwargs["where"] = where
        return _numpy.logical_and.reduce(self, **kwargs)

    @_convert
    def clip(self, min=None, max=None, out=None, **kwargs):
        """Clip values while retaining an active device backend when possible."""
        active_scheme = _scheme.mgr.state
        backend_clip = getattr(self._data, "numpy_clip", None)
        backend_bounds = []
        backend_supported = backend_clip is not None
        for bound in (min, max):
            if isinstance(bound, Array):
                _convert_to_scheme(bound)
                backend_supported &= bound._scheme is active_scheme
                backend_bounds.append(bound._data)
            else:
                backend_bounds.append(bound)

        backend_out = out
        if isinstance(out, Array):
            _convert_to_scheme(out)
            backend_supported &= out._scheme is active_scheme
            backend_out = out._data
        elif out is not None:
            backend_supported = False

        if backend_supported:
            result = backend_clip(
                *backend_bounds, out=backend_out, **kwargs
            )
            if result is not NotImplemented:
                if isinstance(out, Array):
                    return out
                return self._return(result)

        host_bounds = [
            bound.numpy() if isinstance(bound, Array) else bound
            for bound in (min, max)
        ]
        host_out = out.numpy() if isinstance(out, Array) else out
        result = _numpy.clip(
            self.numpy(), *host_bounds, out=host_out, **kwargs
        )
        if isinstance(out, Array):
            out[:] = Array(result, dtype=out.dtype)
            return out
        if out is None and getattr(result, 'shape', None) == self.shape:
            return self._return(result)
        return result

    def mean(
            self, axis=None, dtype=None, out=None, keepdims=False, *,
            where=True):
        """Return the arithmetic mean with NumPy-compatible options."""
        axes = _normalized_reduction_axes(self.shape, axis)
        input_dtype = _numpy.dtype(self.dtype)
        output_dtype = (
            _numpy.dtype(_numpy.float64)
            if dtype is None and input_dtype.kind in "biu"
            else input_dtype if dtype is None else _numpy.dtype(dtype)
        )
        if (
            axes is _UNSUPPORTED_REDUCTION_AXES
            or out is not None
            or where is not True
            or output_dtype.kind not in "fc"
        ):
            return _numpy.mean(
                self.numpy(), axis=axis, dtype=dtype, out=out,
                keepdims=keepdims, where=where,
            )

        count = 1
        for dimension in axes:
            count *= self.shape[dimension]
        result_shape = tuple(
            1 if keepdims and dimension in axes else size
            for dimension, size in enumerate(self.shape)
            if keepdims or dimension not in axes
        )
        result_size = 1
        for size in result_shape:
            result_size *= size
        if count == 0:
            _warnings.warn(
                "Mean of empty slice.", RuntimeWarning, stacklevel=2
            )
            if result_size:
                _warnings.warn(
                    "invalid value encountered in divide",
                    RuntimeWarning,
                    stacklevel=2,
                )
        total = _numpy.add.reduce(
            self, axis=axis, dtype=output_dtype, keepdims=keepdims
        )
        if isinstance(total, Array):
            return total / count
        result = _numpy.asarray(total)
        with _numpy.errstate(divide="ignore", invalid="ignore"):
            _numpy.true_divide(
                result, count, out=result, casting="unsafe", subok=False
            )
        return result[()]

    def _backend_variance(
            self, axis, dtype, out, ddof, keepdims, where, mean):
        """Return a device variance result or ``NotImplemented``."""
        axes = _normalized_reduction_axes(self.shape, axis)
        if (
            axes is _UNSUPPORTED_REDUCTION_AXES
            or out is not None
            or where is not True
            or mean is not None
            or not isinstance(
                ddof, (int, float, _numpy.integer, _numpy.floating)
            )
        ):
            return NotImplemented

        backend_variance = getattr(self._data, "numpy_variance", None)
        if self._scheme is not _scheme.mgr.state or backend_variance is None:
            return NotImplemented
        result = backend_variance(
            axes=axes, dtype=dtype, ddof=ddof, keepdims=keepdims
        )
        if result is NotImplemented:
            return NotImplemented

        count = 1
        for dimension in axes:
            count *= self.shape[dimension]
        result_shape = tuple(
            1 if keepdims and dimension in axes else size
            for dimension, size in enumerate(self.shape)
            if keepdims or dimension not in axes
        )
        result_size = 1
        for size in result_shape:
            result_size *= size
        if ddof >= count:
            _warnings.warn(
                "Degrees of freedom <= 0 for slice",
                RuntimeWarning,
                stacklevel=3,
            )
        if result_size and count == 0:
            _warnings.warn(
                "invalid value encountered in divide",
                RuntimeWarning,
                stacklevel=3,
            )
        if result_size and count - ddof <= 0:
            _warnings.warn(
                "invalid value encountered in divide",
                RuntimeWarning,
                stacklevel=3,
            )
        if hasattr(result, "shape") and result.shape != ():
            return Array(result, copy=False)
        return result

    def var(
            self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=False, *, where=True, mean=None):
        """Return the variance with NumPy-compatible options."""
        result = self._backend_variance(
            axis, dtype, out, ddof, keepdims, where, mean
        )
        if result is not NotImplemented:
            return result
        return _numpy.var(
            self.numpy(), axis=axis, dtype=dtype, out=out, ddof=ddof,
            keepdims=keepdims, where=where, mean=mean,
        )

    def std(
            self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=False, *, where=True, mean=None):
        """Return the standard deviation with NumPy-compatible options."""
        result = self._backend_variance(
            axis, dtype, out, ddof, keepdims, where, mean
        )
        if result is not NotImplemented:
            if isinstance(result, Array):
                return _numpy.sqrt(result)
            return result.dtype.type(_numpy.sqrt(result))
        return _numpy.std(
            self.numpy(), axis=axis, dtype=dtype, out=out, ddof=ddof,
            keepdims=keepdims, where=where, mean=mean,
        )

    def _backend_arg_reduce(self, operation, axis, out, keepdims):
        """Return a device argument reduction or ``NotImplemented``."""
        normalized_axis = None
        if axis is not None:
            try:
                normalized_axis = _operator.index(axis)
            except TypeError:
                return NotImplemented
            if normalized_axis < 0:
                normalized_axis += len(self.shape)
            if normalized_axis < 0 or normalized_axis >= len(self.shape):
                return NotImplemented
        if out is not None:
            return NotImplemented

        backend_arg_reduce = getattr(self._data, "numpy_arg_reduce", None)
        if self._scheme is not _scheme.mgr.state or backend_arg_reduce is None:
            return NotImplemented
        result = backend_arg_reduce(
            operation, axis=normalized_axis, keepdims=keepdims
        )
        if hasattr(result, "shape") and result.shape != ():
            return Array(result, copy=False)
        return result

    def argmax(self, axis=None, out=None, *, keepdims=False):
        """Return the index of the maximum with NumPy-compatible options."""
        result = self._backend_arg_reduce("max", axis, out, keepdims)
        if result is not NotImplemented:
            return result
        return _numpy.argmax(
            self.numpy(), axis=axis, out=out, keepdims=keepdims
        )

    def argmin(self, axis=None, out=None, *, keepdims=False):
        """Return the index of the minimum with NumPy-compatible options."""
        result = self._backend_arg_reduce("min", axis, out, keepdims)
        if result is not NotImplemented:
            return result
        return _numpy.argmin(
            self.numpy(), axis=axis, out=out, keepdims=keepdims
        )

    @_convert
    def searchsorted(self, values, side='left', sorter=None):
        """Find insertion points while retaining an active backend."""
        backend_search = getattr(self._data, "numpy_searchsorted", None)
        backend_values = values
        backend_sorter = sorter
        if backend_search is not None:
            if isinstance(values, Array):
                _convert_to_scheme(values)
                backend_values = values._data
            if isinstance(sorter, Array):
                _convert_to_scheme(sorter)
                backend_sorter = sorter._data
            result = backend_search(
                backend_values, side=side, sorter=backend_sorter
            )
            if result is not NotImplemented:
                if getattr(result, 'shape', ()) != ():
                    return self._return(result)
                return result

        host_values = (
            values.numpy() if isinstance(values, Array) else values
        )
        host_sorter = (
            sorter.numpy() if isinstance(sorter, Array) else sorter
        )
        return _numpy.searchsorted(
            self.numpy(), host_values, side=side, sorter=host_sorter
        )

    @_convert
    def argsort(self, axis=-1, kind=None, order=None, *, stable=None):
        """Return sort indices while retaining an active backend."""
        backend_argsort = getattr(self._data, "numpy_argsort", None)
        if backend_argsort is not None:
            result = backend_argsort(
                axis=axis, kind=kind, order=order, stable=stable
            )
            if result is not NotImplemented:
                return Array(result, copy=False)
        kwargs = {"axis": axis, "kind": kind, "order": order}
        if stable is not None:
            kwargs["stable"] = stable
        return _numpy.argsort(self.numpy(), **kwargs)

    @_convert
    def nonzero(self):
        """Return nonzero indices while retaining an active backend."""
        backend_nonzero = getattr(self._data, "numpy_nonzero", None)
        if backend_nonzero is not None:
            result = backend_nonzero()
            if result is not NotImplemented:
                return tuple(Array(index, copy=False) for index in result)
        return _numpy.nonzero(self.numpy())

    @_returntype
    @_convert
    @schemed(BACKEND_PREFIX)
    def cumsum(self):
        """ Return the cumulative sum of the the array. """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    _scheme_cumsum = cumsum

    def cumsum(self, *args, **kwargs):
        """Return a legacy or NumPy-compatible cumulative sum."""
        if not args and not kwargs:
            return self._scheme_cumsum()
        return _numpy.add.accumulate(self, *args, **kwargs)

    def cumprod(self, axis=None, dtype=None, out=None):
        """Return the cumulative product with NumPy-compatible options."""
        return _numpy.multiply.accumulate(
            self, axis=axis, dtype=dtype, out=out
        )
     
    @_convert
    @_nocomplex
    @schemed(BACKEND_PREFIX)
    def max(self):
        """ Return the maximum value in the array. """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    _scheme_max = max

    @_nocomplex
    def max(self, *args, **kwargs):
        """Return a legacy or NumPy-compatible maximum of the array."""
        if not args and not kwargs:
            return self._scheme_max()
        return _numpy.maximum.reduce(self, *args, **kwargs)
            
    @_convert
    @_nocomplex
    @schemed(BACKEND_PREFIX)
    def max_loc(self):
        """Return the maximum value in the array along with the index location """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_convert
    @schemed(BACKEND_PREFIX)
    def abs_arg_max(self):
        """ Return location of the maximum argument max """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_convert
    @schemed(BACKEND_PREFIX)
    def abs_max_loc(self):
        """Return the maximum elementwise norm in the array along with the index location"""
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_convert
    @_nocomplex
    @schemed(BACKEND_PREFIX)
    def min(self):
        """ Return the maximum value in the array. """ 
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    _scheme_min = min

    @_nocomplex
    def min(self, *args, **kwargs):
        """Return a legacy or NumPy-compatible minimum of the array."""
        if not args and not kwargs:
            return self._scheme_min()
        return _numpy.minimum.reduce(self, *args, **kwargs)
        
    @_returnarray
    @_convert
    @schemed(BACKEND_PREFIX)
    def take(self, indices):
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    _scheme_take = take

    @_convert
    def take(self, indices, axis=None, out=None, mode='raise'):
        """Return elements selected by indices.

        Give backends an opportunity to implement NumPy's axis-aware
        indexing modes before falling back to a host array.
        """
        backend_take = getattr(self._data, "numpy_take", None)
        backend_indices = indices
        if backend_take is not None:
            if isinstance(indices, Array):
                _convert_to_scheme(indices)
                backend_indices = indices._data
            result = backend_take(
                backend_indices, axis=axis, mode=mode
            )
            if result is not NotImplemented:
                selected = Array(result, copy=False)
                if out is None:
                    return selected
                if isinstance(out, Array):
                    out[:] = selected
                else:
                    out[...] = selected.numpy()
                return out

        if axis in (None, 0, -1) and mode == 'raise':
            result = self._scheme_take(indices)
            if out is not None:
                if isinstance(out, Array):
                    out[:] = result
                    return out
                out[...] = result.numpy()
                return out
            return result

        host_indices = (
            indices.numpy() if isinstance(indices, Array) else indices
        )
        result = _numpy.take(
            self.numpy(), host_indices, axis=axis, mode=mode
        )
        if out is not None:
            out[...] = result
            return out
        return result

    @_convert
    def repeat(self, repeats, axis=None):
        """Repeat elements, preserving a supported active backend.

        NumPy dispatches ``numpy.repeat(Array, ...)`` through this method.
        Give backends an opportunity to implement that operation before the
        legacy host-array fallback materializes device data through
        :meth:`numpy`.
        """
        backend_repeat = getattr(self._data, "numpy_repeat", None)
        backend_repeats = repeats
        if backend_repeat is not None:
            if isinstance(repeats, Array):
                _convert_to_scheme(repeats)
                backend_repeats = repeats._data
            result = backend_repeat(backend_repeats, axis=axis)
            if result is not NotImplemented:
                return Array(result, copy=False)

        host_repeats = (
            repeats.numpy() if isinstance(repeats, Array) else repeats
        )
        return _numpy.repeat(self.numpy(), host_repeats, axis=axis)

    @_convert
    def round(self, decimals=0, out=None):
        """Round values while retaining shape-preserving array metadata."""
        backend_round = getattr(self._data, "numpy_round", None)
        if backend_round is not None:
            result = backend_round(decimals=decimals)
            if result is not NotImplemented:
                rounded = self._return(result)
                if out is None:
                    return rounded
                if isinstance(out, Array):
                    out[:] = rounded
                else:
                    out[...] = rounded.numpy()
                return out

        result = _numpy.round(self.numpy(), decimals=decimals)
        if out is None:
            return self._return(Array(result))
        out[...] = result
        return out

    @_convert
    @_vcheckother
    @schemed(BACKEND_PREFIX)
    def dot(self, other):
        """ Return the dot product"""
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)
    
    @schemed(BACKEND_PREFIX)
    def _getvalue(self, index):
        """Helper function to return a single value from an array. May be very
           slow if the memory is on a gpu.
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)

    @_memoize_single
    @_returntype
    def _getslice(self, index):
        return self._return(self._data[index])
    
    @_convert
    def __getitem__(self, index):
        """ Return items from the Array. This not guaranteed to be fast for
            returning single values. 
        """
        if isinstance(index, slice):
            return self._getslice(index)
        else:
            return self._getvalue(index)

    @_convert
    def resize(self, new_size):
        """Resize self to new_size
        """
        if new_size == len(self):
            return
        else:
            new_arr = zeros(new_size, dtype=self.dtype)
            if len(self) <= new_size:
                new_arr[0:len(self)] = self
            else:
                new_arr[:] = self[0:new_size]
                
            self._data = new_arr._data
            self._saved = None

    @_convert
    def roll(self, shift):
        """shift vector
        """
        new_arr = zeros(len(self), dtype=self.dtype)

        if shift < 0:
            shift = shift - len(self) * (shift // len(self))
        
        if shift == 0:
            return
        
        new_arr[0:shift] = self[len(self)-shift: len(self)]
        new_arr[shift:len(self)] = self[0:len(self)-shift]
        
        self._data = new_arr._data
        self._saved = None

    @_returntype
    @_convert
    def astype(self, dtype):
        if _numpy.dtype(self.dtype) == _numpy.dtype(dtype):
            return self
        else:
            return self._data.astype(dtype)
    
    @schemed(BACKEND_PREFIX)
    def _copy(self, self_ref, other_ref):
        """Helper function to copy between two arrays. The arrays references
           should be bare array types and not `Array` class instances. 
        """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)
                
    @_convert
    def __setitem__(self, index, other):
        if isinstance(other,Array):
            _convert_to_scheme(other)

            if self.kind == 'real' and other.kind == 'complex':
                raise ValueError('Cannot set real value with complex')

            if isinstance(index,slice):          
                self_ref = self._data[index]
                other_ref = other._data
            else:
                self_ref = self._data[index:index+1]
                other_ref = other._data

            self._copy(self_ref, other_ref)

        elif type(other) in _ALLOWED_SCALARS:
            if isinstance(index, slice):
                self[index].fill(other)
            else:
                self[index:index+1].fill(other)
        else:
            raise TypeError('Can only copy data from another Array')

    @property
    def precision(self):
        if self.dtype == float32 or self.dtype == complex64:
            return 'single'
        else:
            return 'double'        
                
    @property
    def kind(self):
        if self.dtype == float32 or self.dtype == float64:
            return 'real'
        elif self.dtype == complex64 or self.dtype == complex128:
            return 'complex'
        else:
            return 'unknown'

    @property
    @_convert
    def data(self):
        """Returns the internal python array """
        return self._data

    @data.setter
    def data(self,other):
        dtype = None
        if hasattr(other,'dtype'):
            dtype = other.dtype
        temp = Array(other, dtype=dtype)
        self._data = temp._data
        self._saved = None

    @property
    @_convert
    @schemed(BACKEND_PREFIX)
    def ptr(self):
        """ Returns a pointer to the memory of this array """
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)
        
    @property
    def itemsize(self):
        return self.dtype.itemsize
    
    @property
    def nbytes(self):
        return len(self.data) * self.itemsize

    @property
    @cpuonly
    @_convert
    def _swighelper(self):
        """ Used internally by SWIG typemaps to ensure @_convert 
            is called and scheme is correct  
        """
        return self;

    @_convert
    @schemed(BACKEND_PREFIX)
    def numpy(self):
        """ Returns a Numpy Array that contains this data """     
        err_msg = "This function is a stub that should be overridden using "
        err_msg += "the scheme. You shouldn't be seeing this error!"
        raise ValueError(err_msg)
    
    @_convert
    def lal(self):
        """ Returns a LAL Object that contains this data """

        lal = _lal_compat.require_lal("Array.lal() conversion")

        lal_data = None
        if self._data.dtype == float32:
            lal_data = lal.CreateREAL4Vector(len(self))
        elif self._data.dtype == float64:
            lal_data = lal.CreateREAL8Vector(len(self))
        elif self._data.dtype == complex64:
            lal_data = lal.CreateCOMPLEX8Vector(len(self))
        elif self._data.dtype == complex128:
            lal_data = lal.CreateCOMPLEX16Vector(len(self))

        lal_data.data[:] = self.numpy()

        return lal_data

    @property
    def dtype(self):
        return self._data.dtype
    
    def save(self, path, group=None):
        """
        Save array to a Numpy .npy, hdf, or text file. When saving a complex array as
        text, the real and imaginary parts are saved as the first and second
        column respectively. When using hdf format, the data is stored
        as a single vector, along with relevant attributes.

        Parameters
        ----------
        path: string
            Destination file path. Must end with either .hdf, .npy or .txt.
            
        group: string 
            Additional name for internal storage use. Ex. hdf storage uses
            this as the key value.

        Raises
        ------
        ValueError
            If path does not end in .npy or .txt.
        """

        ext = _os.path.splitext(path)[1]
        if ext == '.npy':
            _numpy.save(path, self.numpy())
        elif ext == '.txt':
            if self.kind == 'real':
                _numpy.savetxt(path, self.numpy())
            elif self.kind == 'complex':
                output = _numpy.vstack((self.numpy().real,
                                        self.numpy().imag)).T
                _numpy.savetxt(path, output)
        elif ext == '.hdf':
            key = 'data' if group is None else group
            with h5py.File(path, 'a') as f:
                f.create_dataset(key, data=self.numpy(), compression='gzip',
                                 compression_opts=9, shuffle=True)
        else:
            raise ValueError('Path must end with .npy, .txt, or .hdf')
           
    @_convert 
    def trim_zeros(self):
        """Remove the leading and trailing zeros.
        """
        if isinstance(self._scheme, _scheme.TorchScheme):
            import torch

            nonzero = torch.nonzero(
                self._data.tensor != 0,
                as_tuple=False,
            ).flatten()
            if nonzero.numel() == 0:
                return self[len(self):0]
            first = int(nonzero[0].item())
            last = int(nonzero[-1].item()) + 1
            return self[first:last]

        tmp = self.numpy()
        f = len(self)-len(_numpy.trim_zeros(tmp, trim='f'))
        b = len(self)-len(_numpy.trim_zeros(tmp, trim='b'))
        return self[f:len(self)-b]

    @_returntype
    @_convert
    def view(self, dtype):
        """
        Return a 'view' of the array with its bytes now interpreted according
        to 'dtype'. The location in memory is unchanged and changing elements
        in a view of an array will also change the original array.

        Parameters
        ----------
        dtype : numpy dtype (one of float32, float64, complex64 or complex128)
            The new dtype that should be used to interpret the bytes of self
        """
        return self._data.view(dtype)

    def copy(self):
        """ Return copy of this array """
        return self._return(self.data.copy())

    @_convert
    def _elementwise_compare(self, other, operation):
        """Compare values while preserving the NumPy boolean-array API."""
        comparison = getattr(self._data, "comparison", None)
        if comparison is None:
            if isinstance(other, Array):
                other = other.numpy()
            return getattr(self.numpy(), f"__{operation}__")(other)

        if isinstance(other, Array):
            _convert_to_scheme(other)
            other = other._data

        try:
            result = comparison(other, operation)
        except (TypeError, ValueError, NotImplementedError):
            if hasattr(other, "numpy"):
                other = other.numpy()
            return getattr(self.numpy(), f"__{operation}__")(other)
        if hasattr(self._data, "tensor"):
            import torch
            from pycbc.types.array_torch import TorchArrayData
            if isinstance(result, torch.Tensor):
                return Array(TorchArrayData(result), copy=False)
            elif isinstance(result, TorchArrayData):
                return Array(result, copy=False)
        return result.detach().cpu().numpy()

    def __lt__(self, other):
        return self._elementwise_compare(other, "lt")

    def __le__(self, other):
        return self._elementwise_compare(other, "le")

    def __ne__(self, other):
        return self._elementwise_compare(other, "ne")

    def __gt__(self, other):
        return self._elementwise_compare(other, "gt")

    def __ge__(self, other):
        return self._elementwise_compare(other, "ge")

# Convenience functions for determining dtypes
def real_same_precision_as(data):
    if data.precision == 'single':
        return float32
    elif data.precision == 'double':
        return float64

def complex_same_precision_as(data):
    if data.precision == 'single':
        return complex64
    elif data.precision == 'double':
        return complex128

def _regular_grid(length, spacing, offset=None):
    """Build a regularly spaced coordinate array in the active scheme."""
    state = _scheme.mgr.state
    if isinstance(state, _scheme.TorchScheme):
        import torch
        from pycbc.types.array_torch import TorchArrayData

        dtype = torch.float32 if state.torch_device.type == 'mps' \
            else torch.float64
        values = torch.arange(
            length, device=state.torch_device, dtype=dtype
        )
        values.mul_(spacing)
        if offset is not None:
            values.add_(offset)
        return Array(TorchArrayData(values), copy=False)

    values = Array(range(length)) * spacing
    if offset is not None:
        values = values + offset
    return values

def _return_array(func):
    @wraps(func)
    def return_array(*args, **kwds):
        return Array(func(*args, **kwds), copy=False)
    return return_array

@_return_array
@schemed(BACKEND_PREFIX)
def zeros(length, dtype=float64):
    """ Return an Array filled with zeros.
    """
    err_msg = "This function is a stub that should be overridden using "
    err_msg += "the scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

@_return_array
@schemed(BACKEND_PREFIX)
def empty(length, dtype=float64):
    """ Return an empty Array (no initialization)
    """
    err_msg = "This function is a stub that should be overridden using "
    err_msg += "the scheme. You shouldn't be seeing this error!"
    raise ValueError(err_msg)

def load_array(path, group=None):
    """Load an Array from an HDF5, ASCII or Numpy file. The file type is
    inferred from the file extension, which must be `.hdf`, `.txt` or `.npy`.

    For ASCII and Numpy files with a single column, a real array is returned.
    For files with two columns, the columns are assumed to contain the real
    and imaginary parts of a complex array respectively.

    The default data types will be double precision floating point.

    Parameters
    ----------
    path : string
        Input file path. Must end with either `.npy`, `.txt` or `.hdf`.

    group: string
        Additional name for internal storage use. When reading HDF files, this
        is the path to the HDF dataset to read.

    Raises
    ------
    ValueError
        If path does not end with a supported extension. For Numpy and ASCII
        input files, this is also raised if the array does not have 1 or 2
        dimensions.
    """
    ext = _os.path.splitext(path)[1]
    if ext == '.npy':
        data = _numpy.load(path)
    elif ext == '.txt':
        data = _numpy.loadtxt(path)
    elif ext == '.hdf':
        key = 'data' if group is None else group
        with h5py.File(path, 'r') as f:
            array = Array(f[key])
        return array
    else:
        raise ValueError('Path must end with .npy, .hdf, or .txt')

    if data.ndim == 1:
        return Array(data)
    elif data.ndim == 2:
        return Array(data[:,0] + 1j*data[:,1])

    raise ValueError('File has %s dimensions, cannot convert to Array, \
                      must be 1 (real) or 2 (complex)' % data.ndim)
