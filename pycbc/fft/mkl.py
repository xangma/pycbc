import ctypes
import os
import threading
import weakref

import numpy as np

import pycbc.libutils
import pycbc.scheme as _scheme
from pycbc.types import zeros

from .core import _BaseFFT, _BaseIFFT

lib = pycbc.libutils.get_ctypes_library("mkl_rt", [])
if lib is None:
    raise ImportError

# MKL constants  taken from mkl_df_defines.h
DFTI_FORWARD_DOMAIN = 0
DFTI_DIMENSION = 1
DFTI_LENGTHS = 2
DFTI_PRECISION = 3
DFTI_FORWARD_SCALE = 4
DFTI_BACKWARD_SCALE = 5
DFTI_NUMBER_OF_TRANSFORMS = 7
DFTI_COMPLEX_STORAGE = 8
DFTI_REAL_STORAGE = 9
DFTI_CONJUGATE_EVEN_STORAGE = 10
DFTI_PLACEMENT = 11
DFTI_INPUT_STRIDES = 12
DFTI_OUTPUT_STRIDES = 13
DFTI_INPUT_DISTANCE = 14
DFTI_OUTPUT_DISTANCE = 15
DFTI_WORKSPACE = 17
DFTI_ORDERING = 18
DFTI_TRANSPOSE = 19
DFTI_DESCRIPTOR_NAME = 20
DFTI_PACKED_FORMAT = 21
DFTI_COMMIT_STATUS = 22
DFTI_VERSION = 23
DFTI_NUMBER_OF_USER_THREADS = 26
DFTI_THREAD_LIMIT = 27
DFTI_COMMITTED = 30
DFTI_UNCOMMITTED = 31
DFTI_COMPLEX = 32
DFTI_REAL = 33
DFTI_SINGLE = 35
DFTI_DOUBLE = 36
DFTI_COMPLEX_COMPLEX = 39
DFTI_COMPLEX_REAL = 40
DFTI_REAL_COMPLEX = 41
DFTI_REAL_REAL = 42
DFTI_INPLACE = 43
DFTI_NOT_INPLACE = 44
DFTI_ORDERED = 48
DFTI_BACKWARD_SCRAMBLED = 49
DFTI_ALLOW = 51
DFTI_AVOID = 52
DFTI_NONE = 53
DFTI_CCS_FORMAT = 54
DFTI_PACK_FORMAT = 55
DFTI_PERM_FORMAT = 56
DFTI_CCE_FORMAT = 57

mkl_domain = {
    "real": {"complex": DFTI_REAL},
    "complex": {
        "real": DFTI_REAL,
        "complex": DFTI_COMPLEX,
    },
}

mkl_descriptor = {
    "single": lib.DftiCreateDescriptor_s_1d,
    "double": lib.DftiCreateDescriptor_d_1d,
}


def check_status(status):
    """Check the status of a mkl functions and raise a python exeption if
    there is an error.
    """
    if status:
        lib.DftiErrorMessage.restype = ctypes.c_char_p
        msg = lib.DftiErrorMessage(status)
        raise RuntimeError(msg)


def create_descriptor(size, idtype, odtype, inplace):
    invec = zeros(1, dtype=idtype)
    outvec = zeros(1, dtype=odtype)
    desc = ctypes.c_void_p()

    domain = mkl_domain[str(invec.kind)][str(outvec.kind)]
    f = mkl_descriptor[invec.precision]
    f.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_long]

    status = f(ctypes.byref(desc), domain, size)
    check_status(status)
    if not desc.value:
        raise RuntimeError("MKL returned an empty FFT descriptor")
    try:
        placement = DFTI_INPLACE if inplace else DFTI_NOT_INPLACE
        check_status(lib.DftiSetValue(desc, DFTI_PLACEMENT, placement))
        nthreads = _scheme.mgr.state.num_threads
        check_status(lib.DftiSetValue(desc, DFTI_THREAD_LIMIT, nthreads))
        # Match MKL's default interleaved complex representation. CCS_FORMAT
        # belongs to PACKED_FORMAT, not CONJUGATE_EVEN_STORAGE.
        check_status(
            lib.DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
        )
        check_status(lib.DftiCommitDescriptor(desc))
    except BaseException:
        lib.DftiFreeDescriptor(ctypes.byref(desc))
        raise
    return desc


# Only these one-thread conditioning transforms have qualified descriptor
# reuse. Keep this separate from the class API and its bound-buffer plans.
_FUNCTION_CACHE_CONFIGS = frozenset(
    {
        ("fft", 16384, 8193, "<f4", "<c8", "single", "real", "complex"),
        ("fft", 16777216, 8388609, "<f4", "<c8", "single", "real", "complex"),
        ("ifft", 8388609, 16777216, "<c8", "<f4", "single", "complex", "real"),
    }
)
_FUNCTION_CACHE_LIMIT = 3
_function_cache = {}
_function_cache_pid = os.getpid()


def _free_function_descriptor(free, value, pid):
    # A forked child must never destroy an inherited parent's native plan.
    if os.getpid() == pid:
        check_status(free(ctypes.byref(ctypes.c_void_p(value))))


class _FunctionDescriptor:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.busy = False
        self.close = weakref.finalize(
            self,
            _free_function_descriptor,
            lib.DftiFreeDescriptor,
            descriptor.value,
            os.getpid(),
        )


def clear_function_cache():
    """Release idle function-API plans in their owning main process/thread.

    At most three plans are retained; their finalizers also run at interpreter
    exit. No input/output arrays are retained. Class-API plans are independent.
    """
    if (
        os.getpid() != _function_cache_pid
        or threading.current_thread() is not threading.main_thread()
    ):
        return
    if any(plan.busy for plan in _function_cache.values()):
        raise RuntimeError("Cannot clear an active MKL FFT cache")
    plans = list(_function_cache.values())
    _function_cache.clear()
    error = None
    for plan in plans:
        try:
            plan.close()
        except BaseException as exc:
            error = exc
    if error is not None:
        raise error


def _function_cache_key(direction, invec, outvec, prec, itype, otype):
    state = _scheme.mgr.state
    if (
        os.getpid() != _function_cache_pid
        or threading.current_thread() is not threading.main_thread()
        or not isinstance(state, _scheme.CPUScheme)
        or state.num_threads != 1
    ):
        return None
    config = (
        direction,
        len(invec),
        len(outvec),
        np.dtype(invec.dtype).str,
        np.dtype(outvec.dtype).str,
        prec,
        itype,
        otype,
    )
    if config not in _FUNCTION_CACHE_CONFIGS:
        return None
    for vector in (invec, outvec):
        data = vector._data
        if (
            type(data) is not np.ndarray
            or data.ndim != 1
            or not data.flags.c_contiguous
        ):
            return None
    if max(invec.ptr, outvec.ptr) < min(
        invec.ptr + invec._data.nbytes, outvec.ptr + outvec._data.nbytes
    ):
        return None
    return config + (type(state), state.num_threads)


def _execute_function(direction, invec, outvec, prec, itype, otype):
    key = _function_cache_key(direction, invec, outvec, prec, itype, otype)
    plan = _function_cache.get(key)
    if plan is not None and plan.busy:
        plan, key = None, None  # Reentrant calls use an independent descriptor.
    if plan is None:
        descriptor = create_descriptor(
            max(len(invec), len(outvec)),
            invec.dtype,
            outvec.dtype,
            invec.ptr == outvec.ptr,
        )
        plan = _FunctionDescriptor(descriptor)
        if key is not None and len(_function_cache) < _FUNCTION_CACHE_LIMIT:
            _function_cache[key] = plan
        else:
            key = None
    plan.busy = True
    try:
        f = lib.DftiComputeForward if direction == "fft" else lib.DftiComputeBackward
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        check_status(f(plan.descriptor, invec.ptr, outvec.ptr))
    except BaseException:
        _function_cache.pop(key, None)
        try:
            plan.close()
        except BaseException:
            pass  # Preserve the compute error; finalization cannot retry a free.
        raise
    finally:
        plan.busy = False
    if key is None:
        plan.close()


def fft(invec, outvec, prec, itype, otype):
    _execute_function("fft", invec, outvec, prec, itype, otype)


def ifft(invec, outvec, prec, itype, otype):
    _execute_function("ifft", invec, outvec, prec, itype, otype)


# Class based API


def _get_desc(fftobj):
    desc = ctypes.c_void_p(1)
    domain = mkl_domain[str(fftobj.invec.kind)][str(fftobj.outvec.kind)]

    f = mkl_descriptor[fftobj.invec.precision]
    f.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_long]
    status = f(ctypes.byref(desc), domain, int(fftobj.size))

    check_status(status)
    # Now we set various things depending on exactly what kind of transform we're
    # performing.

    lib.DftiSetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    # The following only matters if the transform is C2R or R2C
    status = lib.DftiSetValue(desc, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
    check_status(status)

    # In-place or out-of-place:
    if fftobj.inplace:
        status = lib.DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE)
    else:
        status = lib.DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    check_status(status)

    # If we are performing a batched transform:
    if fftobj.nbatch > 1:
        status = lib.DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, fftobj.nbatch)
        check_status(status)
        status = lib.DftiSetValue(desc, DFTI_INPUT_DISTANCE, fftobj.idist)
        check_status(status)
        status = lib.DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, fftobj.odist)
        check_status(status)

    # Knowing how many threads will be allowed may help select a better transform
    nthreads = _scheme.mgr.state.num_threads
    status = lib.DftiSetValue(desc, DFTI_THREAD_LIMIT, nthreads)
    check_status(status)

    # Now everything's ready, so commit
    status = lib.DftiCommitDescriptor(desc)
    check_status(status)

    return desc


class FFT(_BaseFFT):
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(FFT, self).__init__(invec, outvec, nbatch, size)
        self.iptr = self.invec.ptr
        self.optr = self.outvec.ptr
        self._efunc = lib.DftiComputeForward
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.desc = _get_desc(self)

    def execute(self):
        self._efunc(self.desc, self.iptr, self.optr)


class IFFT(_BaseIFFT):
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(IFFT, self).__init__(invec, outvec, nbatch, size)
        self.iptr = self.invec.ptr
        self.optr = self.outvec.ptr
        self._efunc = lib.DftiComputeBackward
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.desc = _get_desc(self)

    def execute(self):
        self._efunc(self.desc, self.iptr, self.optr)
