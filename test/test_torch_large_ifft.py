# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Precision, ownership and dispatch contracts for promoted CPU MKL IFFTs."""

import ctypes
import gc
import types

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, zeros

torch = pytest.importorskip("torch")


class _FakeMKL:
    """Execute the requested native precision through real memory pointers."""

    DFTI_COMPLEX = 32
    DFTI_PLACEMENT = 11
    DFTI_INPLACE = 43
    DFTI_NOT_INPLACE = 44
    DFTI_THREAD_LIMIT = 27

    def __init__(self):
        self.calls = []
        self.status = 0
        owner = self

        def create(precision):
            def create_descriptor(descriptor, domain, size):
                owner.precision = precision
                owner.size = size
                owner.calls.append(("create", precision, domain, size))
                descriptor._obj.value = 8123
                return 0

            return create_descriptor

        class Lib:
            @staticmethod
            def DftiFreeDescriptor(descriptor):
                owner.calls.append(("free", descriptor._obj.value))
                return 0

            @staticmethod
            def DftiSetValue(descriptor, parameter, value):
                owner.calls.append(("set", parameter, value))
                return 0

            @staticmethod
            def DftiCommitDescriptor(descriptor):
                owner.calls.append(("commit", descriptor.value))
                return 0

            @staticmethod
            def DftiComputeBackward(descriptor, source, target):
                owner.calls.append(("execute", source, target))
                real = (
                    ctypes.c_double if owner.precision == "double" else ctypes.c_float
                )
                complex_type = (
                    np.complex128 if real is ctypes.c_double else np.complex64
                )
                storage = real * (2 * owner.size)
                input_view = np.ctypeslib.as_array(storage.from_address(source)).view(
                    complex_type
                )
                output_view = np.ctypeslib.as_array(storage.from_address(target)).view(
                    complex_type
                )
                output_view[:] = (
                    np.fft.ifft(input_view.astype(np.complex128)) * owner.size
                )
                return owner.status

        self.lib = Lib()
        self.mkl_descriptor = {key: create(key) for key in ("single", "double")}

    @staticmethod
    def check_status(status):
        if status:
            raise RuntimeError("native MKL failure")


def _values(size, seed=9021):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(np.complex64)


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("promote", (False, True))
def test_mkl_workspace_precision_normalization_and_reuse(one_thread, promote):
    from pycbc.fft import torchfft

    mkl = _FakeMKL()
    source = torch.from_numpy(_values(256))
    target = torch.empty_like(source)
    plan = torchfft._MKLCPUDirectIFFTPlan(mkl, 256, source, target, promote=promote)
    assert mkl.precision == ("double" if promote else "single")
    assert ("set", mkl.DFTI_PLACEMENT, mkl.DFTI_NOT_INPLACE) in mkl.calls
    assert ("set", mkl.DFTI_THREAD_LIMIT, 1) in mkl.calls
    pointers = (plan._native_source.data_ptr(), plan._native_target.data_ptr())
    if promote:
        assert (
            plan._native_source.dtype == plan._native_target.dtype == torch.complex128
        )
        assert (
            plan._native_source.device.type == plan._native_target.device.type == "cpu"
        )
        assert source.data_ptr() not in pointers and target.data_ptr() not in pointers

    for scale in (1, 1e-10, 1e10):
        source.copy_(torch.from_numpy(_values(256)) * scale)
        preserved = source.clone()
        source_version, target_version = source._version, target._version
        expected = (np.fft.ifft(source.numpy().astype(np.complex128)) * 256).astype(
            np.complex64
        )
        assert plan.can_execute(source, target)
        plan.execute(source, target)
        np.testing.assert_array_equal(target.numpy(), expected)
        assert torch.equal(source, preserved)
        assert source._version == source_version
        assert target._version == target_version + 1
        assert (
            plan._native_source.data_ptr(),
            plan._native_target.data_ptr(),
        ) == pointers
    assert sum(call[0] == "create" for call in mkl.calls) == 1
    assert sum(call[0] == "commit" for call in mkl.calls) == 1
    del plan
    gc.collect()
    assert sum(call[0] == "free" for call in mkl.calls) == 1


def test_promoted_mkl_failure_does_not_publish_partial_workspace(one_thread):
    from pycbc.fft import torchfft

    mkl = _FakeMKL()
    source = torch.from_numpy(_values(64))
    target = torch.full_like(source, 123 + 45j)
    plan = torchfft._MKLCPUDirectIFFTPlan(mkl, 64, source, target, promote=True)
    preserved, version = target.clone(), target._version
    mkl.status = 1
    with pytest.raises(RuntimeError, match="native MKL failure"):
        plan.execute(source, target)
    assert torch.equal(target, preserved)
    assert target._version == version


@pytest.mark.parametrize("fail", (False, True))
def test_inplace_mkl_abi_ownership_and_publication(monkeypatch, one_thread, fail):
    from pycbc.fft import torchfft

    assert torchfft._MKL_PROMOTED_INPLACE_IFFT_SIZES == frozenset({2097152})
    monkeypatch.setattr(
        torchfft, "_MKL_PROMOTED_INPLACE_IFFT_SIZES", frozenset({256})
    )
    mkl = _FakeMKL()
    original = mkl.lib.DftiComputeBackward

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def execute(descriptor, pointer):
        return original(ctypes.c_void_p(descriptor), pointer, pointer)

    # Emulate the signature already installed by another out-of-place plan.
    execute.argtypes = [ctypes.c_void_p] * 3
    mkl.lib.DftiComputeBackward = execute
    source = torch.from_numpy(_values(256))
    target = torch.full_like(source, 123 + 45j)
    plan = torchfft._MKLCPUDirectIFFTPlan(
        mkl, 256, source, target, promote=True
    )
    assert plan._inplace
    assert plan._native_source is plan._native_target
    assert execute.argtypes == [ctypes.c_void_p] * 3
    assert ("set", mkl.DFTI_PLACEMENT, mkl.DFTI_INPLACE) in mkl.calls
    assert sum(call[0] == "commit" for call in mkl.calls) == 1
    pointer = plan._native_source.data_ptr()
    assert pointer not in (source.data_ptr(), target.data_ptr())
    mkl.status = int(fail)
    for scale in (1, 1e-12, 1e12):
        source.copy_(torch.from_numpy(_values(256)) * scale)
        preserved = source.clone(), target.clone()
        versions = source._version, target._version
        assert plan.can_execute(source, target)
        assert not plan.can_execute(source.clone(), target)
        if fail:
            with pytest.raises(RuntimeError, match="native MKL failure"):
                plan.execute(source, target)
            assert torch.equal(target, preserved[1])
            assert target._version == versions[1]
        else:
            plan.execute(source, target)
            expected = (np.fft.ifft(source.numpy().astype(np.complex128))
                        * 256).astype(np.complex64)
            np.testing.assert_array_equal(target.numpy(), expected)
            assert target._version == versions[1] + 1
        assert torch.equal(source, preserved[0])
        assert source._version == versions[0]
        assert plan._native_source.data_ptr() == pointer
        assert mkl.calls[-1] == ("execute", pointer, pointer)
    del plan
    gc.collect()
    assert sum(call[0] == "free" for call in mkl.calls) == 1


@pytest.mark.parametrize(
    "size,nthreads,promote", ((256, 1, False), (512, 1, True), (256, 2, True))
)
def test_inplace_mkl_keeps_other_placements(
    monkeypatch, one_thread, size, nthreads, promote
):
    from pycbc.fft import torchfft

    monkeypatch.setattr(
        torchfft, "_MKL_PROMOTED_INPLACE_IFFT_SIZES", frozenset({256})
    )
    source = torch.from_numpy(_values(size))
    target = torch.empty_like(source)
    mkl = _FakeMKL()
    plan = torchfft._MKLCPUDirectIFFTPlan(
        mkl, size, source, target, nthreads=nthreads, promote=promote
    )
    assert not plan._inplace
    assert plan._native_source is not plan._native_target
    assert ("set", mkl.DFTI_PLACEMENT, mkl.DFTI_NOT_INPLACE) in mkl.calls


@pytest.mark.parametrize(
    "drift", ("source", "target", "threads", "pid", "shape", "grad")
)
def test_promoted_mkl_plan_rejects_contract_drift(monkeypatch, one_thread, drift):
    from pycbc.fft import torchfft

    source = torch.from_numpy(_values(64))
    target = torch.empty_like(source)
    plan = torchfft._MKLCPUDirectIFFTPlan(_FakeMKL(), 64, source, target, promote=True)
    assert plan.can_execute(source, target)
    if drift == "source":
        source = source.clone()
    elif drift == "target":
        target = target.clone()
    elif drift == "threads":
        torch.set_num_threads(2)
    elif drift == "pid":
        child_pid = plan._pid + 1
        monkeypatch.setattr(torchfft.os, "getpid", lambda: child_pid)
    elif drift == "shape":
        source.resize_(8, 8)
    elif drift == "grad":
        source.requires_grad_(True)
    assert not plan.can_execute(source, target)
    # Destroy in the owner so this fixture also frees its native descriptor.
    monkeypatch.undo()
    del plan
    gc.collect()


@pytest.mark.parametrize(
    "excluded", ("none", "gate", "alias", "unaligned", "batch", "forward", "size")
)
def test_promoted_mkl_dispatch_is_narrow(monkeypatch, one_thread, excluded):
    from pycbc.fft import torchfft

    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_MKL_DIRECT_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(torchfft, "_MKL_PROMOTED_IFFT_SIZES", frozenset({64}))
    source, target = (
        torch.empty(64, dtype=torch.complex64),
        torch.empty(64, dtype=torch.complex64),
    )
    if excluded == "alias":
        target = source
    elif excluded == "unaligned":
        source = torch.empty(65, dtype=torch.complex64)[1:]
    elif excluded == "gate":
        monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "0")
    engine = types.SimpleNamespace(
        forward=excluded == "forward",
        size=128 if excluded == "size" else 64,
        nbatch=2 if excluded == "batch" else 1,
        prec="single",
        itype="complex",
        otype="complex",
        invec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=source)),
        outvec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=target)),
    )
    assert torchfft._can_use_mkl_cpu_ifft(engine) == (excluded == "none")


def test_promoted_mkl_setup_selects_double_without_expanding_direct_sizes(
    monkeypatch, one_thread
):
    from pycbc.fft import torchfft

    assert torchfft._MKL_DIRECT_IFFT_SIZES == frozenset(
        {32768, 1048576, 2097152, 4194304}
    )
    calls = []
    monkeypatch.setattr(torchfft, "_MKL_PROMOTED_IFFT_SIZES", frozenset({64}))
    monkeypatch.setattr(torchfft, "_can_use_mkl_cpu_ifft", lambda obj: True)

    def create(size, source, target, nthreads, **kwargs):
        calls.append((size, nthreads, kwargs))
        return None

    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create)
    with scheme.TorchScheme("cpu"):
        engine = torchfft.IFFT(Array(_values(64)), zeros(64, dtype=np.complex64))
        engine.execute()
    assert calls == [(64, 1, {"promote": True})]


@pytest.mark.parametrize("size", (1048576, 2097152, 4194304))
def test_direct_mkl_setup_single_precision_default_and_env_promotion(
    monkeypatch, one_thread, size
):
    from pycbc.fft import torchfft

    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_MKL_DIRECT_PLATFORM_SUPPORTED", True)
    source = torch.empty(size, dtype=torch.complex64)
    target = torch.empty_like(source)
    engine = types.SimpleNamespace(
        forward=False,
        size=size,
        nbatch=1,
        prec="single",
        itype="complex",
        otype="complex",
        invec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=source)),
        outvec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=target)),
    )
    calls = []

    def create(sz, src, tgt, nthreads, **kwargs):
        calls.append((sz, nthreads, kwargs))
        return object()

    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create)

    # By default, large sizes use direct single-precision MKL (no promote kwarg)
    assert torchfft._can_use_mkl_cpu_ifft(engine)
    torchfft._setup_mkl_cpu_ifft_plan(engine)
    assert engine._mkl_plan is not None
    assert calls == [(size, 1, {})]

    # Multiple threads are supported in direct single-precision mode
    torch.set_num_threads(4)
    assert torchfft._can_use_mkl_cpu_ifft(engine)
    calls.clear()
    torchfft._setup_mkl_cpu_ifft_plan(engine)
    assert calls == [(size, 4, {})]

    # When promotion is requested via environment variable, double precision is selected
    # and multithreading is rejected (single thread required)
    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_PROMOTED_IFFT", "1")
    assert not torchfft._can_use_mkl_cpu_ifft(engine)  # 4 threads rejected
    torch.set_num_threads(1)
    assert torchfft._can_use_mkl_cpu_ifft(engine)
    calls.clear()
    torchfft._setup_mkl_cpu_ifft_plan(engine)
    assert calls == [(size, 1, {"promote": True})]


@pytest.mark.parametrize("pattern", ("dense", "banded", "impulse"))
def test_native_promoted_mkl_matches_double_reference_and_legacy_error(
    one_thread, pattern
):
    from pycbc.fft import torchfft

    if not torchfft._MKL_DIRECT_PLATFORM_SUPPORTED:
        pytest.skip("native MKL route is qualified only on Linux x86-64")
    mkl = pytest.importorskip("pycbc.fft.mkl")
    fftw = pytest.importorskip("pycbc.fft.fftw")
    values = _values(4096)
    if pattern == "banded":
        values[:100] = 0
        values[1200:] = 0
    elif pattern == "impulse":
        values[:] = 0
        values[0] = 2 - 3j
    truth = np.fft.ifft(values.astype(np.complex128)) * len(values)
    source, target = (
        torch.from_numpy(values.copy()),
        torch.empty(len(values), dtype=torch.complex64),
    )
    plan = torchfft._MKLCPUDirectIFFTPlan(
        mkl, len(values), source, target, promote=True
    )
    plan.execute(source, target)
    actual = target.numpy().copy()
    with scheme.CPUScheme(num_threads=1):
        double_source = Array(values.astype(np.complex128))
        double_target = zeros(len(values), dtype=np.complex128)
        mkl.IFFT(double_source, double_target).execute()
        np.testing.assert_array_equal(
            actual, double_target.numpy().astype(np.complex64)
        )
        legacy_source, legacy_target = (
            Array(values),
            zeros(len(values), dtype=np.complex64),
        )
        fftw.IFFT(legacy_source, legacy_target).execute()
        legacy = legacy_target.numpy().copy()
    error, legacy_error = actual - truth, legacy - truth
    assert np.linalg.norm(error) <= np.linalg.norm(legacy_error)
    assert np.max(np.abs(error)) <= np.max(np.abs(legacy_error))
    np.testing.assert_array_equal(source.numpy(), values)


@pytest.mark.parametrize("size", (64, 32768))
@pytest.mark.parametrize("nthreads", (1, 2, 4))
def test_mkl_dispatch_thread_matrix(monkeypatch, one_thread, size, nthreads):
    from pycbc.fft import torchfft

    torch.set_num_threads(nthreads)
    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_MKL_DIRECT_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(torchfft, "_MKL_PROMOTED_IFFT_SIZES", frozenset({64}))
    source = torch.empty(size, dtype=torch.complex64)
    target = torch.empty_like(source)
    engine = types.SimpleNamespace(
        forward=False,
        size=size,
        nbatch=1,
        prec="single",
        itype="complex",
        otype="complex",
        invec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=source)),
        outvec=types.SimpleNamespace(_data=types.SimpleNamespace(tensor=target)),
    )
    calls = []

    def create(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create)
    expected = size == 32768 or nthreads == 1
    assert torchfft._can_use_mkl_cpu_ifft(engine) == expected
    torchfft._setup_mkl_cpu_ifft_plan(engine)
    assert (engine._mkl_plan is not None) == expected
    if expected:
        assert calls == [
            {"nthreads": nthreads, **({"promote": True} if size == 64 else {})}
        ]
    else:
        assert calls == []
