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

import ctypes
import os
import platform
import threading
import weakref

import numpy as np
import torch

from pycbc import PYCBC_ALIGNMENT
from pycbc.types import aligned as aligned_array
from .core import _check_fft_args, _BaseFFT, _BaseIFFT


_FFTW_MIN_LENGTH = 4096
# Linux/x86-64 is the only platform on which the direct-plan precision and
# performance route has been validated.  Only the explicitly measured sizes
# below use it; every other size and platform uses the promoted complex128
# workspace, except for the measured Apple/Torch preference below.
_FFTW_DIRECT_PLATFORM_SUPPORTED = (
    platform.system() == "Linux"
    and platform.machine().lower() in {"x86_64", "amd64"}
)
_FFTW_DIRECT_SIZES = frozenset({4096, 8192, 16384, 32768})
# The large search transform is validated only in the inverse direction.
# Direct single-precision FFTW is bitwise identical to legacy PyCBC FFTW at
# this size, while avoiding the promoted workspace's two conversion passes.
_FFTW_DIRECT_IFFT_SIZES = _FFTW_DIRECT_SIZES | frozenset({131072})
_FFTW_DIRECT_MAX_LENGTH = max(_FFTW_DIRECT_IFFT_SIZES)
_FFTW_DIRECT_BATCH_SIZES = frozenset({131072})
_FFTW_DIRECT_BATCH_GATE = "PYCBC_TORCH_CPU_FFTW_BATCH"
# Direct FFTW output to the live Torch buffer at the production search size is
# unusually sensitive to its physical placement.  Execute this one validated
# inverse size from the bound live source into the retained planning output,
# then make one exact complex64 copy to the live target.  This retains bitwise
# legacy-FFTW arithmetic without the large live-output tail or an input copy.
_FFTW_RETAINED_WORKSPACE_IFFT_SIZES = frozenset({131072})
# A user-selected FFTW_MEASURE plan is released only for the retained search
# IFFT above.  Its private planning buffers may be overwritten safely, and the
# measured kernel has passed the precision and steady-state workflow gates.
# More expensive PATIENT/EXHAUSTIVE planning and every other transform retain
# the conservative promoted workspace.
_FFTW_RETAINED_DIRECT_MEASURE_LEVEL = 1
# Apple Silicon's torch.fft wins at exactly 32768, between smaller promoted
# work-plan sizes and the accurate promoted-workspace route above that size.
_TORCH_FFT_PREFERRED_CPU_SIZES = (
    frozenset({32768})
    if platform.system() == "Darwin" and platform.machine() == "arm64"
    else frozenset()
)
# Direct single-precision MKL IFFT execution is released only for the aligned
# size-32768 search transform.  Across the qualification matrix it is bitwise
# identical to standard PyCBC's MKL backend and faster than direct FFTW.  The
# size-131072 candidate remains rejected because a broad seed matrix exceeded
# legacy FFTW's error.  ``PYCBC_TORCH_CPU_MKL_IFFT=0`` retains the FFTW route
# for debugging and controlled performance comparisons.
_MKL_DIRECT_PLATFORM_SUPPORTED = (
    platform.system() == "Linux"
    and platform.machine().lower() in {"x86_64", "amd64"}
)
_MKL_DIRECT_IFFT_SIZES = frozenset({32768})
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_TORCH_IS_INFERENCE = getattr(torch, "is_inference", None)
_CUDA_PROMOTED_ROWS_GATE = "PYCBC_TORCH_CUDA_PROMOTED_ROWS"
_DIRECT_BATCH_IFFT_GATE = "PYCBC_TORCH_DIRECT_BATCH_IFFT"
_CUDA_PROMOTED_BATCH_MAX_ELEMENTS = 2**22
# Retain enough rows to preserve the measured batch crossover without making
# workspace memory scale with a full template bank.  At size 131072 this is
# eight CPU/MPS rows (32 MiB for two complex128 workspaces) or sixteen CUDA
# rows (64 MiB).  MPS adds 16 MiB of CPU transfer staging at that size.
_PROMOTED_BATCH_MAX_ELEMENTS = {
    "cpu": 2**20,
    "cuda": 2**21,
    "mps": 2**20,
}


def from_cli(opt):
    """Apply FFTW planning options selected while Torch is the backend."""
    measure_level = getattr(opt, "fftw_measure_level", None)
    if measure_level is None:
        return
    try:
        from . import fftw
    except (ImportError, OSError, AttributeError):
        return
    try:
        fftw.set_measure_level(measure_level)
    except AttributeError:
        return


def _is_retained_measured_ifft(
    size, forward, nthreads, measure_level
):
    """Return whether explicit FFTW_MEASURE is qualified for this plan."""
    return (
        measure_level == _FFTW_RETAINED_DIRECT_MEASURE_LEVEL
        and nthreads == 1
        and not forward
        and size in _FFTW_RETAINED_WORKSPACE_IFFT_SIZES
    )


def _free_mkl_descriptor(free_descriptor, descriptor_value, owner_pid):
    """Release an MKL descriptor only in the process which created it."""
    if os.getpid() == owner_pid:
        descriptor = ctypes.c_void_p(descriptor_value)
        free_descriptor(ctypes.byref(descriptor))


class _MKLCPUDirectIFFTPlan:
    """Own a pointer-bound, out-of-place single-precision MKL IFFT."""

    def __init__(self, mkl, size, source, target, nthreads=None):
        if nthreads is None:
            nthreads = torch.get_num_threads()
        nthreads = int(nthreads)
        if nthreads < 1:
            raise ValueError("direct MKL IFFT requires at least one thread")
        if _tensors_overlap(source, target):
            raise ValueError("direct MKL IFFT requires disjoint buffers")

        self._pid = os.getpid()
        # Retain and bind the exact tensors used by the released precision and
        # timing measurements.  DFTI accepts execution pointers after commit,
        # but silently accepting replacement storage would escape that gate.
        self._source = source
        self._target = target
        self._source_ptr = source.data_ptr()
        self._target_ptr = target.data_ptr()
        self._size = size
        self._nthreads = nthreads

        free_descriptor = mkl.lib.DftiFreeDescriptor
        free_descriptor.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        free_descriptor.restype = ctypes.c_int
        descriptor = ctypes.c_void_p()
        create = mkl.mkl_descriptor["single"]
        create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_long,
        ]
        create.restype = ctypes.c_int
        try:
            status = create(
                ctypes.byref(descriptor), mkl.DFTI_COMPLEX, size
            )
            mkl.check_status(status)
            if not descriptor.value:
                raise RuntimeError("MKL returned a null descriptor")

            set_value = mkl.lib.DftiSetValue
            # Both properties passed here have integer values.  DftiSetValue
            # is variadic, so constrain only the signature used by this plan.
            set_value.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int
            ]
            set_value.restype = ctypes.c_int
            mkl.check_status(
                set_value(
                    descriptor,
                    mkl.DFTI_PLACEMENT,
                    mkl.DFTI_NOT_INPLACE,
                )
            )
            mkl.check_status(
                set_value(
                    descriptor,
                    mkl.DFTI_THREAD_LIMIT,
                    nthreads,
                )
            )

            commit = mkl.lib.DftiCommitDescriptor
            commit.argtypes = [ctypes.c_void_p]
            commit.restype = ctypes.c_int
            mkl.check_status(commit(descriptor))

            execute = mkl.lib.DftiComputeBackward
            execute.argtypes = [ctypes.c_void_p] * 3
            execute.restype = ctypes.c_int
        except BaseException:
            if descriptor.value:
                _free_mkl_descriptor(
                    free_descriptor, descriptor.value, self._pid
                )
            raise

        self._check_status = mkl.check_status
        self._descriptor = descriptor
        self._execute = execute
        self._increment_version = torch.autograd.graph.increment_version
        try:
            self._finalizer = weakref.finalize(
                self,
                _free_mkl_descriptor,
                free_descriptor,
                descriptor.value,
                self._pid,
            )
        except BaseException:
            _free_mkl_descriptor(
                free_descriptor, descriptor.value, self._pid
            )
            raise

    def can_execute(self, source, target):
        # Planning already validated device, dtype, alignment, layout, and
        # disjoint storage.  Recheck only state that can change on the exact
        # bound Tensor objects.  Keeping this guard plan-local avoids running
        # the complete generic FFT eligibility matrix for every template in a
        # reusable matched-filter plan.
        return (
            getattr(self, "_pid", os.getpid()) == os.getpid()
            and torch.get_num_threads() == self._nthreads
            and source is self._source
            and target is self._target
            and source.data_ptr() == self._source_ptr
            and target.data_ptr() == self._target_ptr
            and source.is_cpu
            and target.is_cpu
            and source.dtype == torch.complex64
            and target.dtype == torch.complex64
            and source.ndim == 1
            and target.ndim == 1
            and source.numel() == self._size
            and target.numel() == self._size
            and source.layout == torch.strided
            and target.layout == torch.strided
            and source.is_contiguous()
            and target.is_contiguous()
            and not source.is_conj()
            and not target.is_conj()
            and not source.is_neg()
            and not target.is_neg()
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(source)
            )
            and (
                _TORCH_IS_INFERENCE is None
                or not _TORCH_IS_INFERENCE(target)
            )
            and not _has_autograd_state(source)
            and not _has_autograd_state(target)
        )

    def execute(self, source, target):
        status = self._execute(
            self._descriptor,
            source.data_ptr(),
            target.data_ptr(),
        )
        # DFTI writes outside Torch's dispatcher, so publish the mutation to
        # version-counter consumers just as the direct FFTW plan does.
        self._increment_version(target)
        self._check_status(status)


class _FFTWWorkspaceVector:
    """Minimal PyCBC-vector interface around an internal Torch tensor."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.dtype = np.dtype(np.complex128)

    @property
    def ptr(self):
        return self.tensor.data_ptr()


def _destroy_batch_plan_in_owner(fftw, destroy, plan, owner_pid):
    """Destroy an FFTW plan only in the process which created it."""
    if os.getpid() == owner_pid:
        fftw._destroy_plan(destroy, plan)


class _FFTWCPUWorkPlan:
    """Own an in-place double-precision FFTW plan and work buffer."""

    def __init__(self, fftw, size, forward, measure_level):
        self._pid = os.getpid()
        self._fftw = fftw
        self._nthreads = 1
        self._measure_level = measure_level
        self._workspace = _FFTWWorkspaceVector(
            torch.empty(size, dtype=torch.complex128, device="cpu")
        )
        # Plan and execute on this exact retained pointer. Planning on a
        # separate PyCBC scratch allocation and using FFTW's new-array API on
        # the Torch workspace would require their native FFTW alignment
        # classes to match; a fixed-byte modulus cannot establish that
        # portably. Measured planners may overwrite this private workspace,
        # which is safe because execute() replaces it with the user input.
        with fftw._FFTW_PLANNING_LOCK:
            if not fftw._fftw_threaded_set:
                fftw.set_threads_backend()
            if fftw._fftw_current_nthreads != self._nthreads:
                fftw._fftw_plan_with_nthreads(self._nthreads)
            direction = (
                fftw.FFTW_FORWARD if forward else fftw.FFTW_BACKWARD
            )
            flags = fftw.get_flag(
                measure_level,
                self._workspace.ptr % PYCBC_ALIGNMENT == 0,
            )
            planner = fftw.plan_function["complex128"]["complex128"]
            planner.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            planner.restype = ctypes.c_void_p
            execute = fftw.execute_function["complex128"]["complex128"]
            execute.argtypes = [ctypes.c_void_p] * 3
            destroy = fftw.double_lib.fftw_destroy_plan
            destroy.argtypes = [ctypes.c_void_p]
            plan = planner(
                size,
                self._workspace.ptr,
                self._workspace.ptr,
                direction,
                flags,
            )
        if not plan:
            raise RuntimeError("FFTW returned a null plan")
        self._execute = execute
        self._plan = plan
        try:
            self._finalizer = weakref.finalize(
                self,
                _destroy_batch_plan_in_owner,
                fftw,
                destroy,
                plan,
                self._pid,
            )
        except BaseException:
            _destroy_batch_plan_in_owner(fftw, destroy, plan, self._pid)
            raise

    def execute(self, source, target):
        # Converting to a reusable complex128 workspace both avoids host
        # copies and improves accuracy over either float32 FFT backend.
        self._workspace.tensor.copy_(source)
        self._execute(
            self._plan, self._workspace.ptr, self._workspace.ptr
        )
        target.copy_(self._workspace.tensor)

    def can_execute(self, source, target):
        return (
            getattr(self, "_pid", os.getpid()) == os.getpid()
            and torch.get_num_threads() == self._nthreads
        )


class _FFTWCPUDirectPlan:
    """Own a reusable, out-of-place single-precision FFTW plan."""

    def __init__(
        self,
        fftw,
        size,
        forward,
        source,
        target,
        aligned,
        measure_level,
        nthreads=1,
    ):
        self._pid = os.getpid()
        retained_search_ifft = (
            nthreads == 1
            and not forward
            and size in _FFTW_RETAINED_WORKSPACE_IFFT_SIZES
        )
        if measure_level != 0 and not _is_retained_measured_ifft(
            size, forward, nthreads, measure_level
        ):
            raise ValueError(
                "direct FFTW planning mode is not precision-qualified"
            )
        self._aligned = aligned
        self._measure_level = measure_level
        self._nthreads = nthreads
        # Bind execution to these exact live buffers and keep them alive for
        # the lifetime of the native plan.  Planning itself uses the legacy-
        # compatible scratch buffers below.
        self._source = source
        self._target = target
        self._source_ptr = source.data_ptr()
        self._target_ptr = target.data_ptr()
        buffers_aligned = (
            self._source_ptr % PYCBC_ALIGNMENT == 0
            and self._target_ptr % PYCBC_ALIGNMENT == 0
        )
        if buffers_aligned != aligned:
            raise ValueError("FFTW planner alignment does not match buffers")
        if not fftw._fftw_threaded_set:
            fftw.set_threads_backend()
        if fftw._fftw_current_nthreads != nthreads:
            fftw._fftw_plan_with_nthreads(nthreads)

        transform_size = np.asarray([size], dtype=np.int32)
        embedding = np.asarray([size], dtype=np.int32)
        # Match the legacy class backend: plan on PyCBC-aligned scratch
        # buffers, then execute on the caller's arrays.  In particular, an
        # FFTW_UNALIGNED plan made on an aligned scratch buffer can select a
        # different measured-wisdom kernel from a plan made on an unaligned
        # live pointer, even though both plans are valid.
        self._scratch_input = aligned_array.zeros(size, np.complex64)
        self._scratch_output = aligned_array.zeros(size, np.complex64)
        self._use_retained_workspace = retained_search_ifft
        if self._use_retained_workspace:
            # This tensor is a zero-copy alias of the retained NumPy output.
            # Torch copy_ publishes the final live-target mutation to version-
            # counter consumers without a manual bump.
            self._scratch_output_tensor = torch.from_numpy(
                self._scratch_output
            )
        scratch_input_ptr = self._scratch_input.ctypes.data
        scratch_output_ptr = self._scratch_output.ctypes.data
        if aligned:
            # FFTW's new-array contract is expressed in terms of its native
            # alignment class, not a portable fixed-byte modulus.  Fall back
            # to the promoted workspace if Torch and legacy scratch buffers
            # have different native classes.  FFTW_UNALIGNED removes this
            # constraint for the unaligned route.
            alignment_of = fftw.float_lib.fftwf_alignment_of
            alignment_of.argtypes = [ctypes.c_void_p]
            alignment_of.restype = ctypes.c_int
            if (
                alignment_of(self._source_ptr)
                != alignment_of(scratch_input_ptr)
                or alignment_of(self._target_ptr)
                != alignment_of(scratch_output_ptr)
            ):
                raise ValueError(
                    "Torch and legacy FFTW alignment classes differ"
                )
        direction = (
            fftw.FFTW_FORWARD if forward else fftw.FFTW_BACKWARD
        )
        flags = fftw.get_flag(measure_level, aligned)
        execute = fftw.execute_function["complex64"]["complex64"]
        execute.argtypes = [ctypes.c_void_p] * 3
        destroy = fftw.float_lib.fftwf_destroy_plan
        destroy.argtypes = [ctypes.c_void_p]
        plan = fftw.plan_many_c2c_f(
            1,
            transform_size.ctypes.data,
            1,
            scratch_input_ptr,
            embedding.ctypes.data,
            1,
            size,
            scratch_output_ptr,
            embedding.ctypes.data,
            1,
            size,
            direction,
            flags,
        )
        if not plan:
            raise RuntimeError("FFTW returned a null plan")
        self._execute = execute
        self._increment_version = torch.autograd.graph.increment_version
        self._plan = plan
        try:
            self._finalizer = weakref.finalize(
                self,
                _destroy_batch_plan_in_owner,
                fftw,
                destroy,
                plan,
                self._pid,
            )
        except BaseException:
            _destroy_batch_plan_in_owner(fftw, destroy, plan, self._pid)
            raise

    def can_execute(self, source, target):
        buffers_aligned = (
            source.data_ptr() % PYCBC_ALIGNMENT == 0
            and target.data_ptr() % PYCBC_ALIGNMENT == 0
        )
        return (
            getattr(self, "_pid", os.getpid()) == os.getpid()
            and torch.get_num_threads() == self._nthreads
            and not _tensors_overlap(source, target)
            # Bind new-array execution to the live buffers whose native
            # alignment classes were validated when the plan was built.
            and source.data_ptr() == self._source_ptr
            and target.data_ptr() == self._target_ptr
            # Match legacy PyCBC's planner class on every execution.  An
            # FFTW_UNALIGNED plan is safe on aligned buffers, but it can select
            # a different algorithm (and rounding path) from the aligned plan.
            and buffers_aligned == self._aligned
        )

    def execute(self, source, target):
        if getattr(self, "_use_retained_workspace", False):
            self._execute(
                self._plan,
                source.data_ptr(),
                self._scratch_output.ctypes.data,
            )
            target.copy_(self._scratch_output_tensor)
            return
        self._execute(
            self._plan, source.data_ptr(), target.data_ptr()
        )
        # Native writes must participate in Torch's mutation tracking. This
        # is a metadata-only operation; it does not add another array pass.
        self._increment_version(target)


class _FFTWCPUDirectBatchPlan:
    """Own a legacy-compatible pointer-bound complex64 FFTW batch plan."""

    def __init__(
        self,
        fftw,
        size,
        batch,
        forward,
        source,
        target,
        aligned,
        measure_level,
        nthreads=1,
    ):
        total = size * batch
        if (
            size <= 0
            or batch <= 1
            or size > np.iinfo(np.int32).max
            or batch > np.iinfo(np.int32).max
            or total > np.iinfo(np.int32).max
            or source.numel() != total
            or target.numel() != total
        ):
            raise ValueError("FFTW batch geometry exceeds the legacy ABI")

        self._pid = os.getpid()
        self._thread_id = threading.get_ident()
        self._size = size
        self._batch = batch
        self._forward = forward
        self._aligned = aligned
        self._measure_level = measure_level
        self._nthreads = nthreads
        self._source = source
        self._target = target
        self._source_ptr = source.data_ptr()
        self._target_ptr = target.data_ptr()
        buffers_aligned = (
            self._source_ptr % PYCBC_ALIGNMENT == 0
            and self._target_ptr % PYCBC_ALIGNMENT == 0
        )
        if buffers_aligned != aligned:
            raise ValueError("FFTW planner alignment does not match buffers")

        # Reproduce fftw._fftw_setup_unlocked exactly for the C2C case.  In
        # particular, legacy PyCBC uses total flat lengths for the rank-one
        # embedding arrays and aligned scratch storage even for an UNALIGNED
        # plan.  The resulting plan therefore has the same wisdom identity.
        transform_size = np.asarray([size], dtype=np.int32)
        input_embedding = np.asarray([total], dtype=np.int32)
        output_embedding = np.asarray([total], dtype=np.int32)
        scratch_input = aligned_array.zeros(total, np.complex64)
        scratch_output = aligned_array.zeros(total, np.complex64)
        scratch_input_ptr = scratch_input.ctypes.data
        scratch_output_ptr = scratch_output.ctypes.data
        if aligned:
            alignment_of = fftw.float_lib.fftwf_alignment_of
            alignment_of.argtypes = [ctypes.c_void_p]
            alignment_of.restype = ctypes.c_int
            if (
                alignment_of(self._source_ptr)
                != alignment_of(scratch_input_ptr)
                or alignment_of(self._target_ptr)
                != alignment_of(scratch_output_ptr)
            ):
                raise ValueError(
                    "Torch and legacy FFTW alignment classes differ"
                )

        direction = (
            fftw.FFTW_FORWARD if forward else fftw.FFTW_BACKWARD
        )
        flags = fftw.get_flag(measure_level, aligned)
        execute = fftw.execute_function["complex64"]["complex64"]
        execute.argtypes = [ctypes.c_void_p] * 3
        destroy = fftw.float_lib.fftwf_destroy_plan
        destroy.argtypes = [ctypes.c_void_p]
        plan = fftw.plan_many_c2c_f(
            1,
            transform_size.ctypes.data,
            batch,
            scratch_input_ptr,
            input_embedding.ctypes.data,
            1,
            size,
            scratch_output_ptr,
            output_embedding.ctypes.data,
            1,
            size,
            direction,
            flags,
        )
        if not plan:
            raise RuntimeError("FFTW returned a null batch plan")
        self._execute = execute
        self._increment_version = torch.autograd.graph.increment_version
        self._plan = plan
        try:
            self._finalizer = weakref.finalize(
                self,
                _destroy_batch_plan_in_owner,
                fftw,
                destroy,
                plan,
                self._pid,
            )
        except BaseException:
            fftw._destroy_plan(destroy, plan)
            raise

    def can_execute(self, source, target, *, size, batch, forward):
        buffers_aligned = (
            source.data_ptr() % PYCBC_ALIGNMENT == 0
            and target.data_ptr() % PYCBC_ALIGNMENT == 0
        )
        return (
            os.getpid() == self._pid
            and threading.get_ident() == self._thread_id
            and source is self._source
            and target is self._target
            and source.data_ptr() == self._source_ptr
            and target.data_ptr() == self._target_ptr
            and size == self._size
            and batch == self._batch
            and forward is self._forward
            and torch.get_num_threads() == self._nthreads
            and not _tensors_overlap(source, target)
            and buffers_aligned == self._aligned
        )

    def execute(self, source, target):
        self._execute(
            self._plan, source.data_ptr(), target.data_ptr()
        )
        self._increment_version(target)


def _create_fftw_cpu_plan(
    size,
    forward,
    direct=False,
    aligned=True,
    nthreads=1,
    source=None,
    target=None,
):
    """Create a guarded FFTW plan, or return None when unavailable."""
    cache_entry = None
    wisdom_cache = None
    try:
        from . import fftw

        # Use the same re-entrant planner lock as the legacy FFTW backend.
        with fftw._FFTW_PLANNING_LOCK:
            measure_level = fftw.get_measure_level()
            requested_measure_level = measure_level
            if (
                direct
                and aligned
                and nthreads == 1
                and not forward
                and size in _FFTW_RETAINED_WORKSPACE_IFFT_SIZES
            ):
                from . import wisdom_cache

                measure_level, cache_entry = wisdom_cache.prepare_plan(
                    fftw,
                    size=size,
                    forward=forward,
                    direct=direct,
                    aligned=aligned,
                    nthreads=nthreads,
                    batch=1,
                    requested_measure_level=measure_level,
                )
            # Direct FFTW_ESTIMATE is the default.  A user-selected MEASURE
            # plan is also precision-qualified for the retained production
            # search IFFT; PATIENT/EXHAUSTIVE and all other non-default cases
            # keep the promoted complex128 workspace.
            retained_measured_ifft = _is_retained_measured_ifft(
                size, forward, nthreads, measure_level
            )
            if direct and (measure_level == 0 or retained_measured_ifft):
                try:
                    plan = _FFTWCPUDirectPlan(
                        fftw,
                        size,
                        forward,
                        source,
                        target,
                        aligned,
                        measure_level,
                        nthreads=nthreads,
                    )
                    if cache_entry is not None:
                        wisdom_cache.record_plan(
                            fftw, cache_entry, measure_level
                        )
                    return plan
                except (AttributeError, KeyError, RuntimeError, ValueError):
                    # A native alignment-class mismatch (or unavailable
                    # direct-plan API) must retain the accuracy guarantee.
                    # The promoted complex128 workspace is the conservative
                    # fallback when a legacy-compatible direct plan cannot be
                    # constructed.
                    return _FFTWCPUWorkPlan(
                        fftw, size, forward, requested_measure_level
                    )
            return _FFTWCPUWorkPlan(
                fftw, size, forward, requested_measure_level
            )
    except (ImportError, OSError, AttributeError, KeyError,
            RuntimeError, ValueError):
        return None
    finally:
        if cache_entry is not None and wisdom_cache is not None:
            wisdom_cache.cancel_plan(cache_entry)


def _create_mkl_cpu_ifft_plan(
    size, source, target, nthreads=None
):
    """Create the validated direct MKL IFFT plan, or return ``None``."""
    try:
        from . import mkl

        if nthreads is None:
            nthreads = torch.get_num_threads()
        return _MKLCPUDirectIFFTPlan(
            mkl, size, source, target, nthreads
        )
    except (ImportError, OSError, AttributeError, KeyError,
            RuntimeError, ValueError, ctypes.ArgumentError):
        return None


def _has_autograd_state(tensor):
    if tensor.requires_grad:
        return True
    # Avoid unpacking two ordinary search buffers on every reusable FFT.  A
    # forward dual cannot exist outside an active dual level.
    current_level = getattr(torch.autograd.forward_ad, "_current_level", None)
    if current_level == -1:
        return False
    try:
        return (
            torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
        )
    except (AttributeError, RuntimeError):
        # An uninspectable tensor should stay on Torch's autograd-aware path.
        return True


def _environment_flag(name, default=False):
    """Read a strict boolean environment switch."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    choices = ", ".join(sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES))
    raise ValueError(f"{name} must be one of: {choices}; got {value!r}")


def _can_use_fftw_cpu(fftobj):
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    return (
        fftobj.nbatch == 1
        and fftobj.size >= _FFTW_MIN_LENGTH
        and fftobj.size <= np.iinfo(np.int32).max
        and fftobj.prec == "single"
        and fftobj.itype == "complex"
        and fftobj.otype == "complex"
        # Native pointer execution must not bypass Tensor-subclass dispatch or
        # Torch's restrictions on mutating inference tensors.  Keep these
        # fail-closed checks ahead of every tensor operation below.
        and type(fin) is torch.Tensor
        and type(fout) is torch.Tensor
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fin)
        )
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fout)
        )
        and torch.get_num_threads() == 1
        and fin.device.type == "cpu"
        and fout.device.type == "cpu"
        and fin.dtype == torch.complex64
        and fout.dtype == torch.complex64
        and fin.ndim == 1
        and fout.ndim == 1
        and fin.numel() == fftobj.size
        and fout.numel() == fftobj.size
        and fin.layout == torch.strided
        and fout.layout == torch.strided
        and fin.is_contiguous()
        and fout.is_contiguous()
        and not fin.is_conj()
        and not fout.is_conj()
        and not fin.is_neg()
        and not fout.is_neg()
        and not _has_autograd_state(fin)
        and not _has_autograd_state(fout)
    )


def _can_use_fftw_cpu_batch(fftobj):
    """Whether a class transform can use the gated legacy batch ABI."""
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    total = fftobj.size * fftobj.nbatch
    return (
        _environment_flag(_FFTW_DIRECT_BATCH_GATE, default=False)
        and _FFTW_DIRECT_PLATFORM_SUPPORTED
        and fftobj.nbatch > 1
        and fftobj.size in _FFTW_DIRECT_BATCH_SIZES
        and fftobj.size <= np.iinfo(np.int32).max
        and fftobj.nbatch <= np.iinfo(np.int32).max
        and total <= np.iinfo(np.int32).max
        and fftobj.prec == "single"
        and fftobj.itype == "complex"
        and fftobj.otype == "complex"
        and type(fin) is torch.Tensor
        and type(fout) is torch.Tensor
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fin)
        )
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fout)
        )
        and torch.get_num_threads() >= 1
        and fin.device.type == "cpu"
        and fout.device.type == "cpu"
        and fin.dtype == torch.complex64
        and fout.dtype == torch.complex64
        and fin.ndim == 1
        and fout.ndim == 1
        and fin.numel() == total
        and fout.numel() == total
        and fin.layout == torch.strided
        and fout.layout == torch.strided
        and fin.is_contiguous()
        and fout.is_contiguous()
        and not fin.is_conj()
        and not fout.is_conj()
        and not fin.is_neg()
        and not fout.is_neg()
        and not _has_autograd_state(fin)
        and not _has_autograd_state(fout)
        and not _tensors_overlap(fin, fout)
        and hasattr(torch.autograd.graph, "increment_version")
    )


def _create_fftw_cpu_batch_plan(fftobj, forward):
    """Create a direct plan_many matching the standard PyCBC backend."""
    try:
        from . import fftw

        with fftw._FFTW_PLANNING_LOCK:
            nthreads = torch.get_num_threads()
            if not fftw._fftw_threaded_set:
                fftw.set_threads_backend()
            if fftw._fftw_current_nthreads != nthreads:
                fftw._fftw_plan_with_nthreads(nthreads)
            measure_level = fftw.get_measure_level()
            fin = fftobj.invec._data.tensor
            fout = fftobj.outvec._data.tensor
            aligned = (
                fin.data_ptr() % PYCBC_ALIGNMENT == 0
                and fout.data_ptr() % PYCBC_ALIGNMENT == 0
            )
            return _FFTWCPUDirectBatchPlan(
                fftw,
                fftobj.size,
                fftobj.nbatch,
                forward,
                fin,
                fout,
                aligned,
                measure_level,
                nthreads=nthreads,
            )
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        ctypes.ArgumentError,
    ):
        return None


def _setup_fftw_cpu_batch_plan(fftobj, forward):
    fftobj._fftw_batch_plan = None
    if _can_use_fftw_cpu_batch(fftobj):
        fftobj._fftw_batch_plan = _create_fftw_cpu_batch_plan(
            fftobj, forward
        )


def _execute_fftw_cpu_batch_plan(fftobj, forward):
    plan = fftobj._fftw_batch_plan
    if plan is None or not _can_use_fftw_cpu_batch(fftobj):
        return False
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    if not plan.can_execute(
        fin,
        fout,
        size=fftobj.size,
        batch=fftobj.nbatch,
        forward=forward,
    ):
        return False
    plan.execute(fin, fout)
    return True


def _can_use_mkl_cpu_ifft(fftobj):
    """Return whether ``fftobj`` is in the validated direct-MKL matrix."""
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    return (
        _environment_flag("PYCBC_TORCH_CPU_MKL_IFFT", default=True)
        and _MKL_DIRECT_PLATFORM_SUPPORTED
        and not fftobj.forward
        and fftobj.size in _MKL_DIRECT_IFFT_SIZES
        and fftobj.nbatch == 1
        and fftobj.size <= np.iinfo(np.int32).max
        and fftobj.prec == "single"
        and fftobj.itype == "complex"
        and fftobj.otype == "complex"
        and type(fin) is torch.Tensor
        and type(fout) is torch.Tensor
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fin)
        )
        and (
            _TORCH_IS_INFERENCE is None
            or not _TORCH_IS_INFERENCE(fout)
        )
        and torch.get_num_threads() >= 1
        and fin.device.type == "cpu"
        and fout.device.type == "cpu"
        and fin.dtype == torch.complex64
        and fout.dtype == torch.complex64
        and fin.ndim == 1
        and fout.ndim == 1
        and fin.numel() == fftobj.size
        and fout.numel() == fftobj.size
        and fin.layout == torch.strided
        and fout.layout == torch.strided
        and fin.is_contiguous()
        and fout.is_contiguous()
        and not fin.is_conj()
        and not fout.is_conj()
        and not fin.is_neg()
        and not fout.is_neg()
        and not _has_autograd_state(fin)
        and not _has_autograd_state(fout)
        and fin.data_ptr() % PYCBC_ALIGNMENT == 0
        and fout.data_ptr() % PYCBC_ALIGNMENT == 0
        and not _tensors_overlap(fin, fout)
        and hasattr(torch.autograd.graph, "increment_version")
    )


def _setup_mkl_cpu_ifft_plan(fftobj):
    fftobj._mkl_plan = None
    if _can_use_mkl_cpu_ifft(fftobj):
        fftobj._mkl_plan = _create_mkl_cpu_ifft_plan(
            fftobj.size,
            fftobj.invec._data.tensor,
            fftobj.outvec._data.tensor,
            nthreads=torch.get_num_threads(),
        )


def _execute_mkl_cpu_ifft_plan(fftobj):
    if fftobj._mkl_plan is None:
        return False
    if not _environment_flag("PYCBC_TORCH_CPU_MKL_IFFT", default=True):
        return False
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    if not fftobj._mkl_plan.can_execute(fin, fout):
        return False
    fftobj._mkl_plan.execute(fin, fout)
    return True


def _setup_fftw_cpu_plan(fftobj, forward):
    fftobj._fftw_plan = None
    if (
        _can_use_fftw_cpu(fftobj)
        and fftobj.size not in _TORCH_FFT_PREFERRED_CPU_SIZES
    ):
        fin = fftobj.invec._data.tensor
        fout = fftobj.outvec._data.tensor
        direct_sizes = (
            _FFTW_DIRECT_SIZES if forward else _FFTW_DIRECT_IFFT_SIZES
        )
        direct = (
            _FFTW_DIRECT_PLATFORM_SUPPORTED
            and fftobj.size in direct_sizes
            and not _tensors_overlap(fin, fout)
            and hasattr(torch.autograd.graph, "increment_version")
        )
        # Use the same planner alignment class as legacy PyCBC.  Forcing
        # FFTW_UNALIGNED on aligned Torch buffers can select different imported
        # wisdom and produce a slightly less accurate result.
        aligned = (
            fin.data_ptr() % PYCBC_ALIGNMENT == 0
            and fout.data_ptr() % PYCBC_ALIGNMENT == 0
        )
        fftobj._fftw_plan = _create_fftw_cpu_plan(
            fftobj.size,
            forward,
            direct=direct,
            aligned=aligned,
            nthreads=1,
            source=fin,
            target=fout,
        )


def _execute_fftw_cpu_plan(fftobj):
    if fftobj._fftw_plan is None or not _can_use_fftw_cpu(fftobj):
        return False
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    if not fftobj._fftw_plan.can_execute(fin, fout):
        return False
    fftobj._fftw_plan.execute(fin, fout)
    return True


def _ensure_match(invec, outvec):
    if outvec._data.tensor.device != invec._data.tensor.device:
        raise ValueError("Input and output must be on the same torch device")


def _shares_storage(invec, outvec):
    """Return whether the input and output tensor byte ranges overlap."""
    return _tensors_overlap(
        invec._data.tensor, outvec._data.tensor
    )


def _tensors_overlap(input_tensor, output_tensor):
    """Return whether two tensor byte ranges overlap.

    Comparing Torch storage identities is insufficient because separately
    wrapped NumPy views can describe overlapping memory with distinct Storage
    objects.  FFTW-eligible tensors are contiguous, so their pointer and byte
    length define the exact accessed interval.  Retain the prior conservative
    storage-identity check for non-contiguous tensors whose accessed bytes may
    not form one interval.
    """
    if input_tensor.device != output_tensor.device:
        return False
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        return (
            input_tensor.untyped_storage().data_ptr()
            == output_tensor.untyped_storage().data_ptr()
        )
    input_start = input_tensor.data_ptr()
    output_start = output_tensor.data_ptr()
    input_end = (
        input_start + input_tensor.numel() * input_tensor.element_size()
    )
    output_end = (
        output_start + output_tensor.numel() * output_tensor.element_size()
    )
    return input_start < output_end and output_start < input_end


def _copy_result(outvec, result):
    if result.dtype != outvec._data.tensor.dtype:
        result = result.to(dtype=outvec._data.tensor.dtype)
    outvec._data.tensor.copy_(result)


def fft(invec, outvec, _, itype, otype):
    _ensure_match(invec, outvec)
    fin = invec._data.tensor
    fout = outvec._data.tensor
    aliased = _shares_storage(invec, outvec)
    # NOTE: PyTorch FFT kernels are not bitwise-identical to numpy/FFTW
    # in float32. In parity testing we observe rfft diffs up to ~2e-5
    # (and downstream SNR/chi^2 diffs at 1e-6 / 1e-4). If exact parity
    # is required, run the torch scheme in float64 or route through the
    # CPU FFT backend.
    if itype == 'complex' and otype == 'complex':
        transform = torch.fft.fft
    elif itype == 'real' and otype == 'complex':
        transform = torch.fft.rfft
    else:
        raise ValueError("Unsupported dtype combination for torch fft")

    # Real/complex in-place transforms use overlapping typed views.  Preserve
    # the allocation-before-copy behaviour for every aliased transform; write
    # directly into the caller's output for the common out-of-place case.
    if aliased:
        _copy_result(outvec, transform(fin, n=fin.shape[-1]))
    else:
        transform(fin, n=fin.shape[-1], out=fout)


def ifft(invec, outvec, _, itype, otype):
    _ensure_match(invec, outvec)
    fin = invec._data.tensor
    fout = outvec._data.tensor
    n_out = fout.shape[-1]
    aliased = _shares_storage(invec, outvec)
    if itype == 'complex' and otype == 'complex':
        transform = torch.fft.ifft
    elif itype == 'complex' and otype == 'real':
        transform = torch.fft.irfft
    else:
        raise ValueError("Unsupported dtype combination for torch ifft")

    # PyCBC's inverse-FFT backend contract is unnormalised.  Asking Torch for
    # forward normalisation produces that result directly and avoids a second
    # full-output pass to multiply the conventional inverse transform by N.
    if aliased:
        _copy_result(outvec, transform(fin, n=n_out, norm="forward"))
    else:
        transform(fin, n=n_out, norm="forward", out=fout)


def _batch_views(fftobj):
    """Expose the flat class-API buffers as independent transform rows."""
    fin = fftobj.invec._data.tensor
    fout = fftobj.outvec._data.tensor
    return (
        fin.view(fftobj.nbatch, fftobj.idist),
        fout.view(fftobj.nbatch, fftobj.odist),
    )


def _batch_logical_views(fftobj):
    """Return the logical rows, excluding real-transform padding."""
    fin, fout = _batch_views(fftobj)
    if fftobj.itype == "real":
        fin = fin[..., :fftobj.size]
    if fftobj.otype == "real":
        fout = fout[..., :fftobj.size]
    return fin, fout


class _TorchPromotedBatchPlan:
    """Reusable double-precision workspaces for single-precision batches.

    Native single-precision Torch FFTs are less accurate than legacy FFTW at
    the large batch size used by ``LiveBatchMatchedFilter``.  Copying through
    retained double-precision workspaces improves on legacy precision while
    preserving the public dtype and avoiding per-execution allocations.
    """

    _PROMOTED_DTYPES = {
        torch.float32: torch.float64,
        torch.complex64: torch.complex128,
    }

    def __init__(self, source, target):
        self._source_shape = tuple(source.shape)
        self._target_shape = tuple(target.shape)
        self._source_dtype = source.dtype
        self._target_dtype = target.dtype
        self._device = source.device
        # MPS cannot represent the promoted dtypes.  Stage its public
        # single-precision buffers through CPU tensors, then perform the same
        # accuracy-preserving Torch transform on CPU.  Direct cross-device
        # complex64 -> complex128 copies are unsupported, hence the explicit
        # single-precision staging buffers on both sides.
        workspace_device = (
            torch.device("cpu")
            if self._device.type == "mps"
            else self._device
        )
        element_budget = _PROMOTED_BATCH_MAX_ELEMENTS[self._device.type]
        promoted_rows_enabled = (
            self._device.type == "cuda"
            and _environment_flag(_CUDA_PROMOTED_ROWS_GATE, default=False)
        )
        if promoted_rows_enabled:
            element_budget = _CUDA_PROMOTED_BATCH_MAX_ELEMENTS
        row_elements = max(source.shape[-1], target.shape[-1])
        self.rows = min(
            source.shape[0], max(1, element_budget // row_elements)
        )
        self._promoted_rows_enabled = promoted_rows_enabled
        source_workspace_shape = (self.rows, source.shape[-1])
        target_workspace_shape = (self.rows, target.shape[-1])
        self._source_staging = None
        self._target_staging = None
        if workspace_device != self._device:
            self._source_staging = torch.empty(
                source_workspace_shape,
                dtype=source.dtype,
                device=workspace_device,
            )
            self._target_staging = torch.empty(
                target_workspace_shape,
                dtype=target.dtype,
                device=workspace_device,
            )
        self.source = torch.empty(
            source_workspace_shape,
            dtype=self._PROMOTED_DTYPES[source.dtype],
            device=workspace_device,
        )
        self.target = torch.empty(
            target_workspace_shape,
            dtype=self._PROMOTED_DTYPES[target.dtype],
            device=workspace_device,
        )

    def can_execute(self, source, target):
        """Whether workspaces can safely serve the current class buffers."""
        promoted_rows_enabled = (
            self._device.type == "cuda"
            and _environment_flag(_CUDA_PROMOTED_ROWS_GATE, default=False)
        )
        return (
            promoted_rows_enabled == getattr(self, "_promoted_rows_enabled", False)
            and source.device == target.device == self._device
            and tuple(source.shape) == self._source_shape
            and tuple(target.shape) == self._target_shape
            and source.dtype == self._source_dtype
            and target.dtype == self._target_dtype
            and source.layout == target.layout == torch.strided
        )

    def execute(self, source, target, transform, size, inverse):
        execution_source = source
        if (
            _tensors_overlap(source, target)
            and source.data_ptr() != target.data_ptr()
        ):
            # A chunk write can otherwise corrupt a not-yet-read source row.
            # Partial overlap is rare, so snapshot its public-precision input
            # once; allocation failure deliberately propagates rather than
            # falling through to a lower-precision transform.
            execution_source = source.clone()
        for start in range(0, source.shape[0], self.rows):
            stop = min(start + self.rows, source.shape[0])
            count = stop - start
            source_work = self.source[:count]
            target_work = self.target[:count]
            if self._source_staging is None:
                source_work.copy_(execution_source[start:stop])
            else:
                source_stage = self._source_staging[:count]
                source_stage.copy_(execution_source[start:stop])
                source_work.copy_(source_stage)
            kwargs = {"n": size, "dim": -1, "out": target_work}
            if inverse:
                kwargs["norm"] = "forward"
            transform(source_work, **kwargs)
            if self._target_staging is None:
                target[start:stop].copy_(target_work)
            else:
                target_stage = self._target_staging[:count]
                target_stage.copy_(target_work)
                target[start:stop].copy_(target_stage)


def _setup_promoted_torch_batch_plan(fftobj):
    """Build the validated precision path for float32 Torch batches."""
    fftobj._promoted_batch_plan = None
    if fftobj.nbatch <= 1 or fftobj.prec != "single":
        return
    fin, fout = _batch_logical_views(fftobj)
    if fin.device.type not in {"cpu", "cuda", "mps"}:
        return
    fftobj._promoted_batch_plan = _TorchPromotedBatchPlan(fin, fout)


def _execute_promoted_torch_batch_plan(
    fftobj, source, target, transform, inverse
):
    """Execute through reusable promoted workspaces when AD is inactive."""
    if source.device.type not in {"cpu", "cuda", "mps"}:
        return False
    if _has_autograd_state(source) or _has_autograd_state(target):
        # Preserve Torch's established out=/in-place autograd behaviour.
        return False
    plan = fftobj._promoted_batch_plan
    if plan is None or not plan.can_execute(source, target):
        # These workspaces bind no external pointers, so same-shape storage
        # replacements reuse safely; rebuild for a changed tensor contract.
        plan = _TorchPromotedBatchPlan(source, target)
        fftobj._promoted_batch_plan = plan
    plan.execute(source, target, transform, fftobj.size, inverse)
    return True


def _execute_batched_fft(fftobj):
    """Execute the class API's independent forward transforms with Torch."""
    _ensure_match(fftobj.invec, fftobj.outvec)
    fin, fout = _batch_logical_views(fftobj)
    if fftobj.itype == "complex" and fftobj.otype == "complex":
        transform = torch.fft.fft
        transform_input = fin
    elif fftobj.itype == "real" and fftobj.otype == "complex":
        transform = torch.fft.rfft
        transform_input = fin
    else:
        raise ValueError("Unsupported dtype combination for torch fft")

    if fftobj.prec == "single" and _execute_promoted_torch_batch_plan(
        fftobj, transform_input, fout, transform, inverse=False
    ):
        return
    if _shares_storage(fftobj.invec, fftobj.outvec):
        fout.copy_(
            transform(transform_input, n=fftobj.size, dim=-1)
        )
    else:
        transform(
            transform_input, n=fftobj.size, dim=-1, out=fout
        )


def _can_use_direct_batch_ifft(fftobj):
    """Whether batch IFFT can execute directly with torch.fft.ifft."""
    if fftobj.nbatch <= 1 or fftobj.prec != "single":
        return False
    if fftobj.itype != "complex" or fftobj.otype != "complex":
        return False
    fin = getattr(getattr(fftobj.invec, "_data", None), "tensor", None)
    fout = getattr(getattr(fftobj.outvec, "_data", None), "tensor", None)
    if fin is None or fout is None:
        return False
    if fin.dtype != torch.complex64 or fout.dtype != torch.complex64:
        return False
    if _has_autograd_state(fin) or _has_autograd_state(fout):
        return False
    default_direct = (fin.device.type == "cuda")
    return _environment_flag(_DIRECT_BATCH_IFFT_GATE, default=default_direct)


def _execute_batched_ifft_direct(fftobj):
    """Execute direct single-precision 2D batched IFFT with hardware cache tiling."""
    _ensure_match(fftobj.invec, fftobj.outvec)
    fin, fout = _batch_logical_views(fftobj)
    kwargs = {
        "n": fftobj.size,
        "dim": -1,
        "norm": "forward",
    }
    nbatch = fin.shape[0]
    is_cuda = (fin.device.type == "cuda")
    device_id = fin.device.index or 0 if is_cuda else 0
    from pycbc.hardware import get_optimal_batch_tile_size
    tile_size = get_optimal_batch_tile_size(
        fftobj.size, is_cuda=is_cuda, device_id=device_id
    )

    if nbatch > tile_size:
        for start in range(0, nbatch, tile_size):
            stop = min(start + tile_size, nbatch)
            fin_chunk = fin[start:stop]
            fout_chunk = fout[start:stop]
            if _shares_storage(fftobj.invec, fftobj.outvec):
                if fin_chunk.data_ptr() == fout_chunk.data_ptr():
                    torch.fft.ifft(fin_chunk, out=fout_chunk, **kwargs)
                else:
                    res = torch.fft.ifft(fin_chunk, **kwargs)
                    fout_chunk.copy_(res)
            else:
                torch.fft.ifft(fin_chunk, out=fout_chunk, **kwargs)
        return

    if _shares_storage(fftobj.invec, fftobj.outvec):
        if fin.data_ptr() == fout.data_ptr():
            torch.fft.ifft(fin, out=fout, **kwargs)
        else:
            res = torch.fft.ifft(fin, **kwargs)
            fout.copy_(res)
    else:
        torch.fft.ifft(fin, out=fout, **kwargs)


def _execute_batched_ifft(fftobj):
    """Execute the class API's independent inverse transforms with Torch."""
    _ensure_match(fftobj.invec, fftobj.outvec)
    if _can_use_direct_batch_ifft(fftobj):
        _execute_batched_ifft_direct(fftobj)
        return
    fin, fout = _batch_logical_views(fftobj)
    if fftobj.itype == "complex" and fftobj.otype == "complex":
        transform = torch.fft.ifft
        transform_output = fout
    elif fftobj.itype == "complex" and fftobj.otype == "real":
        transform = torch.fft.irfft
        transform_output = fout
    else:
        raise ValueError("Unsupported dtype combination for torch ifft")

    if fftobj.prec == "single" and _execute_promoted_torch_batch_plan(
        fftobj, fin, transform_output, transform, inverse=True
    ):
        return
    kwargs = {
        "n": fftobj.size,
        "dim": -1,
        # PyCBC's inverse class API is unnormalised transform-by-transform.
        "norm": "forward",
    }
    if _shares_storage(fftobj.invec, fftobj.outvec):
        if fftobj.itype == "complex" and fftobj.otype == "complex":
            transform(fin, out=fin, **kwargs)
            if fin.data_ptr() != transform_output.data_ptr():
                transform_output.copy_(fin)
        else:
            transform_output.copy_(transform(fin, **kwargs))
    else:
        transform(fin, out=transform_output, **kwargs)


class FFT(_BaseFFT):
    """Class-based torch FFT."""
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super().__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)
        _setup_fftw_cpu_batch_plan(self, forward=True)
        if self._fftw_batch_plan is None:
            _setup_promoted_torch_batch_plan(self)
        else:
            # Do not retain c128 conversion workspaces while the explicitly
            # gated direct plan owns the batch.  Any later gate, pointer, or
            # tensor-contract drift constructs the conservative fallback on
            # demand in _execute_promoted_torch_batch_plan().
            self._promoted_batch_plan = None
        _setup_fftw_cpu_plan(self, forward=True)

    def execute(self):
        if self.nbatch > 1:
            if not _execute_fftw_cpu_batch_plan(self, forward=True):
                _execute_batched_fft(self)
            return
        if not _execute_fftw_cpu_plan(self):
            fft(self.invec, self.outvec, self.prec, self.itype, self.otype)


class IFFT(_BaseIFFT):
    """Class-based torch inverse FFT."""
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super().__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)
        _setup_fftw_cpu_batch_plan(self, forward=False)
        if (
            self._fftw_batch_plan is None
            and not _can_use_direct_batch_ifft(self)
        ):
            _setup_promoted_torch_batch_plan(self)
        else:
            self._promoted_batch_plan = None
        _setup_mkl_cpu_ifft_plan(self)
        if self._mkl_plan is None:
            _setup_fftw_cpu_plan(self, forward=False)
        else:
            # Avoid allocating the promoted complex128 workspace on the
            # measured direct route.  If the caller later replaces either
            # bound tensor, execute() constructs that conservative fallback
            # lazily before touching the replacement storage.
            self._fftw_plan = None

    def execute(self):
        if self.nbatch > 1:
            if not _execute_fftw_cpu_batch_plan(self, forward=False):
                _execute_batched_ifft(self)
            return
        if _execute_mkl_cpu_ifft_plan(self):
            return
        if self._fftw_plan is None:
            _setup_fftw_cpu_plan(self, forward=False)
        if not _execute_fftw_cpu_plan(self):
            ifft(self.invec, self.outvec, self.prec, self.itype, self.otype)
