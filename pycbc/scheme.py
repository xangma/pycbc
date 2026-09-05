# Copyright (C) 2014  Alex Nitz, Andrew Miller
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
This modules provides python contexts that set the default behavior for PyCBC
objects.
"""

import ctypes
import logging
import os
from functools import lru_cache, wraps

import pycbc

from .libutils import get_ctypes_library
from .pool import use_mpi

logger = logging.getLogger("pycbc.scheme")


@lru_cache(maxsize=1)
def _resolve_libgomp():
    """Resolve the process-global GNU OpenMP runtime once."""
    return get_ctypes_library("gomp", ["gomp"], mode=ctypes.RTLD_GLOBAL)


class _SchemeManager(object):
    _single = None

    def __init__(self):
        if _SchemeManager._single is not None:
            raise RuntimeError("SchemeManager is a private class")
        _SchemeManager._single = self

        self.state = None
        self._lock = False

    def lock(self):
        self._lock = True

    def unlock(self):
        self._lock = False

    def shift_to(self, state):
        if self._lock is False:
            self.state = state
        else:
            raise RuntimeError("The state is locked, cannot shift schemes")


# Create the global processing scheme manager
mgr = _SchemeManager()
default_context = None


class Scheme(object):
    """Context that sets PyCBC objects to use CPU processing."""

    _single = None

    def __init__(self):
        self._owns_singleton = False
        if DefaultScheme is type(self):
            return
        if Scheme._single is not None:
            raise RuntimeError("Only one processing scheme can be used")
        Scheme._single = True
        self._owns_singleton = True

    def __enter__(self):
        mgr.shift_to(self)
        mgr.lock()
        return self

    def __exit__(self, type, value, traceback):
        mgr.unlock()
        mgr.shift_to(default_context)

    def __del__(self):
        if Scheme is not None and getattr(self, "_owns_singleton", False):
            Scheme._single = None


_cuda_cleanup_list = []


def register_clean_cuda(function):
    _cuda_cleanup_list.append(function)


def clean_cuda(context):
    # Before cuda context is destroyed, all item destructions dependent on cuda
    # must take place. This calls all functions that have been registered
    # with _register_clean_cuda() in reverse order
    # So the last one registered, is the first one cleaned
    _cuda_cleanup_list.reverse()
    for func in _cuda_cleanup_list:
        func()

    context.pop()
    from pycuda.tools import clear_context_caches

    clear_context_caches()


class CUDAScheme(Scheme):
    """Context that sets PyCBC objects to use a CUDA processing scheme."""

    def __init__(self, device_num=0):
        Scheme.__init__(self)
        if not pycbc.HAVE_CUDA:
            raise RuntimeError("Install PyCUDA to use CUDA processing")
        import pycuda.driver

        pycuda.driver.init()
        self.device = pycuda.driver.Device(device_num)
        self.context = self.device.make_context(
            flags=pycuda.driver.ctx_flags.SCHED_BLOCKING_SYNC
        )
        import atexit

        atexit.register(clean_cuda, self.context)


class CUPYScheme(Scheme):
    """Scheme for using CUPY.

    Supports using CUPY with MPI. If MPI is enabled, will use all available
    devices. The environment variable `CUDA_VISIBLE_DEVICES` can be used to
    restrict the devices used.

    Parameters
    ----------
    device_num : int, optional
        The device number to use. If not provided, will use the default, 0.
        Should not be provided when using MPI to parallelize across devices.
    """

    def __init__(self, device_num=None):
        import cupy  # Fail now if cupy is not there.
        import cupy.cuda

        do_mpi, _, rank = use_mpi(require_mpi=False, log=False)

        if device_num is not None and do_mpi:
            logger.warning("MPI is enabled, but a device number was provided.")

        if device_num is None and do_mpi:
            # Logical device numbers will always be 0, 1, 2, ... etc. irrespective
            # of the physical device numbers.
            device_num = rank % cupy.cuda.runtime.getDeviceCount()
            logger.debug("MPI enabled, using CUDA device %s", device_num)

        self.device_num = device_num
        self.cuda_device = cupy.cuda.Device(self.device_num)

    def __enter__(self):
        super().__enter__()
        self.cuda_device.__enter__()
        logger.warning(
            "You are using the CUPY GPU backend for PyCBC. This backend is "
            "still only a prototype. It may be useful for your application "
            "but it may fail unexpectedly, run slowly, or not give correct "
            "output. Please do contribute to the effort to develop this "
            "further."
        )

    def __exit__(self, *args):
        super().__exit__(*args)
        self.cuda_device.__exit__(*args)


class TorchScheme(Scheme):
    """Context that sets PyCBC objects to use a Torch processing scheme."""

    def __init__(self, device=None, num_threads=None):
        # A Torch scheme does not create a process-global driver context.
        # Scheme.__enter__ still prevents simultaneous active schemes, but
        # lightweight Torch scheme objects may safely coexist.
        if not pycbc.HAVE_TORCH:
            raise RuntimeError("Install PyTorch to use the Torch processing scheme.")

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "PyTorch was found but could not be imported; install a "
                "working PyTorch package to use the Torch processing scheme."
            ) from exc

        self._torch = torch
        self.device_spec = "cpu" if device in (None, "") else device
        self.torch_device = torch.device(self.device_spec)

        if self.torch_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Torch CUDA device requested but CUDA is unavailable."
                )
        elif self.torch_device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Torch MPS device requested but MPS is unavailable.")
        elif self.torch_device.type != "cpu":
            raise RuntimeError(f"Unsupported Torch device {self.device_spec}")

        # Alias used by backends to locate the target device
        self.device = self.torch_device
        self.prefix = "torch"
        if num_threads is not None:
            num_threads = int(num_threads)
            if num_threads <= 0:
                raise ValueError(f"num_threads must be positive, got {num_threads}")
        self.num_threads = num_threads
        self._thread_state = None

    def _restore_thread_state(self):
        """Restore thread settings saved by the current context entry."""
        state = self._thread_state
        self._thread_state = None
        if state is None:
            return

        torch_threads, openmp_runtime, openmp_threads = state
        try:
            self._torch.set_num_threads(torch_threads)
        finally:
            if openmp_runtime is not None:
                openmp_runtime.omp_set_num_threads(openmp_threads)

    def __enter__(self):
        super().__enter__()
        try:
            if self.device.type == "cpu" and self.num_threads is not None:
                torch_threads = self._torch.get_num_threads()
                openmp_runtime = None
                openmp_threads = None
                try:
                    runtime = _resolve_libgomp()
                    if runtime is not None:
                        openmp_threads = runtime.omp_get_max_threads()
                        openmp_runtime = runtime
                except Exception:
                    pass

                self._thread_state = (
                    torch_threads,
                    openmp_runtime,
                    openmp_threads,
                )
                self._torch.set_num_threads(self.num_threads)
                if openmp_runtime is not None:
                    openmp_runtime.omp_set_num_threads(self.num_threads)
        except Exception:
            try:
                self._restore_thread_state()
            finally:
                super().__exit__(None, None, None)
            raise
        return self

    def __exit__(self, type, value, traceback):
        try:
            self._restore_thread_state()
        finally:
            super().__exit__(type, value, traceback)


class CPUScheme(Scheme):
    def __init__(self, num_threads=1):
        if isinstance(num_threads, int):
            self.num_threads = num_threads
        elif num_threads == "env" and "PYCBC_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["PYCBC_NUM_THREADS"])
        else:
            import multiprocessing

            self.num_threads = multiprocessing.cpu_count()
        self._libgomp = None

    def __enter__(self):
        Scheme.__enter__(self)
        # CPUScheme loads libgomp globally below.  If MKL keeps its default
        # Intel OpenMP layer, threaded DFTI calls can silently return corrupt
        # data once libgomp (including Torch's copy) is present. Default to
        # the compatible GNU layer, preserving explicit process settings.
        if pycbc.HAVE_MKL:
            os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        try:
            self._libgomp = _resolve_libgomp()
        except Exception:
            # Should we fail or give a warning if we cannot import
            # libgomp? Seems to work even for MKL scheme, but
            # not entirely sure why...
            pass

        num_threads_str = str(self.num_threads)
        if os.environ.get("OMP_NUM_THREADS") != num_threads_str:
            os.environ["OMP_NUM_THREADS"] = num_threads_str
        if self._libgomp is not None:
            self._libgomp.omp_set_num_threads(int(self.num_threads))

    def __exit__(self, type, value, traceback):
        if os.environ.get("OMP_NUM_THREADS") != "1":
            os.environ["OMP_NUM_THREADS"] = "1"
        if self._libgomp is not None:
            self._libgomp.omp_set_num_threads(1)
        Scheme.__exit__(self, type, value, traceback)


class MKLScheme(CPUScheme):
    def __init__(self, num_threads=1):
        CPUScheme.__init__(self, num_threads)
        if not pycbc.HAVE_MKL:
            raise RuntimeError("Can't find MKL libraries")


class NumpyScheme(CPUScheme):
    pass


scheme_prefix = {
    CUDAScheme: "cuda",
    CPUScheme: "cpu",
    CUPYScheme: "cupy",
    MKLScheme: "mkl",
    NumpyScheme: "numpy",
    TorchScheme: "torch",
}
_scheme_map = {v: k for (k, v) in scheme_prefix.items()}

_default_scheme_raw = os.getenv("PYCBC_SCHEME", "cpu")
_default_scheme_prefix, _, _default_scheme_extra = _default_scheme_raw.partition(":")
try:
    _default_scheme_class = _scheme_map[_default_scheme_prefix]
except KeyError:
    raise RuntimeError(
        "PYCBC_SCHEME={!r} not recognised, please select one of: {}".format(
            _default_scheme_prefix,
            ", ".join(map(repr, _scheme_map)),
        ),
    )


def _parse_torch_scheme_extra(extra):
    if not extra:
        return None, None
    if extra.isdigit():
        return "cpu", int(extra)
    if extra.startswith("cpu:") and extra[4:].isdigit():
        return "cpu", int(extra[4:])
    return extra, None


def _torch_device_from_cli(device, device_id):
    """Apply the shared CLI device ID to an unindexed Torch accelerator."""
    if device in ("cuda", "mps"):
        return f"{device}:{device_id}"
    return device


class DefaultScheme(_default_scheme_class):
    def __init__(self):
        extra = _default_scheme_extra if _default_scheme_extra else None

        if _default_scheme_prefix == "torch":
            dev, numt = _parse_torch_scheme_extra(extra)
            super().__init__(device=dev, num_threads=numt)
        elif _default_scheme_prefix == "cuda":
            dev = int(extra) if extra and extra.isdigit() else 0
            super().__init__(device_num=dev)
        elif _default_scheme_prefix in ("cpu", "mkl"):
            if extra is None:
                super().__init__()
            else:
                numt = extra if not extra.isdigit() else int(extra)
                super().__init__(num_threads=numt)
        elif _default_scheme_prefix == "cupy":
            dev = int(extra) if extra and extra.isdigit() else None
            super().__init__(device_num=dev)
        else:
            super().__init__()


default_context = DefaultScheme()
mgr.state = default_context
scheme_prefix[DefaultScheme] = _default_scheme_prefix


def current_prefix():
    return scheme_prefix[type(mgr.state)]


def current_backend_key():
    """Return a hashable identity for scheme-owned reusable resources."""
    state = mgr.state
    return (
        current_prefix(),
        type(state),
        getattr(state, "device", None),
        getattr(state, "device_num", None),
        getattr(state, "num_threads", None),
    )


_import_cache = {}


def schemed(prefix):
    def scheming_function(func):
        @wraps(func)
        def _scheming_function(*args, **kwds):
            try:
                return _import_cache[mgr.state][func](*args, **kwds)
            except KeyError:
                exc_errors = []
                for sch in mgr.state.__class__.__mro__[0:-2]:
                    try:
                        backend = __import__(
                            prefix + scheme_prefix[sch], fromlist=[func.__name__]
                        )
                        schemed_fn = getattr(backend, func.__name__)
                    except (ImportError, AttributeError) as e:
                        exc_errors += [e]
                        continue

                    if mgr.state not in _import_cache:
                        _import_cache[mgr.state] = {}

                    _import_cache[mgr.state][func] = schemed_fn

                    return schemed_fn(*args, **kwds)

                err = (
                    f"Failed to find implementation of {func.__name__} "
                    f"for {current_prefix()} scheme. "
                )
                for emsg in exc_errors:
                    err += str(emsg) + " "
                raise RuntimeError(err)

        return _scheming_function

    return scheming_function


def cpuonly(func):
    @wraps(func)
    def _cpuonly(*args, **kwds):
        if not issubclass(type(mgr.state), CPUScheme):
            raise TypeError(
                func.__name__ + " can only be called from a CPU processing scheme."
            )
        return func(*args, **kwds)

    return _cpuonly


def insert_processing_option_group(parser):
    """
    Adds the options used to choose a processing scheme. This should be used
    if your program supports the ability to select the processing scheme.

    Parameters
    ----------
    parser : object
        OptionParser instance
    """
    processing_group = parser.add_argument_group(
        "Options for selecting the processing scheme in this program."
    )
    processing_group.add_argument(
        "--processing-scheme",
        help="The choice of processing scheme. "
        "Choices are "
        + str(list(set(scheme_prefix.values())))
        + ". (optional for CPU scheme) The number of "
        "execution threads "
        "can be indicated by cpu:NUM_THREADS, "
        "where NUM_THREADS "
        "is an integer. The default is a single thread. "
        "If the scheme is provided as cpu:env, the number "
        "of threads can be provided by the PYCBC_NUM_THREADS "
        "environment variable. If the environment variable "
        "is not set, the number of threads matches the number "
        "of logical cores. ",
        default="cpu",
    )

    processing_group.add_argument(
        "--processing-device-id",
        help="(optional) ID of GPU to use for accelerated processing",
        default=0,
        type=int,
    )


def from_cli(opt):
    """Parses the command line options and returns a processing scheme.

    Parameters
    ----------
    opt: object
        Result of parsing the CLI with OptionParser, or any object with
        the required attributes.

    Returns
    -------
    ctx: Scheme
        Returns the requested processing scheme.
    """
    scheme_str = opt.processing_scheme.split(":", 1)
    name = scheme_str[0]
    extra = scheme_str[1] if len(scheme_str) > 1 else None

    if name == "cuda":
        logger.info("Running with CUDA support")
        ctx = CUDAScheme(opt.processing_device_id)
    elif name == "torch":
        dev, numt = _parse_torch_scheme_extra(extra)
        dev = _torch_device_from_cli(dev, opt.processing_device_id)
        ctx = TorchScheme(device=dev, num_threads=numt)
        logger.info("Running with Torch support on device %s", ctx.torch_device)
    elif name == "mkl":
        if extra:
            numt = extra
            if numt.isdigit():
                numt = int(numt)
            ctx = MKLScheme(num_threads=numt)
        else:
            ctx = MKLScheme()
        logger.info("Running with MKL support: %s threads" % ctx.num_threads)
    elif name == "cupy":
        logger.info("Running with CUPY support")
        ctx = CUPYScheme()
    else:
        if extra:
            numt = extra
            if numt.isdigit():
                numt = int(numt)
            ctx = CPUScheme(num_threads=numt)
        else:
            ctx = CPUScheme()
        logger.info("Running with CPU support: %s threads" % ctx.num_threads)
    return ctx


def verify_processing_options(opt, parser):
    """Parses the  processing scheme options and verifies that they are
       reasonable.


    Parameters
    ----------
    opt : object
        Result of parsing the CLI with OptionParser, or any object with the
        required attributes.
    parser : object
        OptionParser instance.
    """
    scheme_types = scheme_prefix.values()
    if opt.processing_scheme.split(":")[0] not in scheme_types:
        parser.error("(%s) is not a valid scheme type.")


class ChooseBySchemeDict(dict):
    """This class represents a dictionary whose purpose is to chose objects
    based on their processing scheme. The keys are intended to be processing
    schemes.
    """

    def __getitem__(self, scheme):
        for base in scheme.__mro__[0:-1]:
            try:
                return dict.__getitem__(self, base)
                break
            except:
                pass
