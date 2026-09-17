# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""CPU FFT specialization eligibility and descriptor thread selection."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycbc import hardware, scheme
from pycbc.types import zeros

torch = pytest.importorskip("torch")


@pytest.fixture
def cache_domain(monkeypatch):
    monkeypatch.setattr(hardware, "get_cpu_l3_cache_size", lambda **kw: 2**24)
    monkeypatch.setattr(hardware, "get_cpu_cores_per_numa_node", lambda: 8)
    hardware.get_optimal_1d_fft_threads.cache_clear()
    yield
    hardware.get_optimal_1d_fft_threads.cache_clear()


@pytest.mark.parametrize(
    "size, requested, expected",
    ((32768, 32, 8), (1048576, 32, 8), (1048577, 32, 32),
     (32768, 4, 4), (32768, 0, 1)),
)
def test_descriptor_thread_selection_respects_cache_and_request(
    cache_domain, size, requested, expected
):
    assert hardware.get_optimal_1d_fft_threads(size, requested) == expected


@pytest.mark.parametrize(
    "size, eligible",
    ((32768, True), (65536, True), (131072, True), (262144, True),
     (524288, True), (1048576, False), (16384, False)),
)
def test_expanded_mkl_sizes_use_selected_descriptor_threads(
    monkeypatch, cache_domain, size, eligible
):
    from pycbc.fft import torchfft

    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_MKL_DIRECT_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 32)
    calls = []
    plan = object()

    def create_plan(length, source, target, nthreads):
        calls.append((length, nthreads))
        return plan

    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create_plan)
    with scheme.TorchScheme("cpu"):
        fftobj = SimpleNamespace(
            invec=zeros(size, dtype=np.complex64),
            outvec=zeros(size, dtype=np.complex64),
            size=size, nbatch=1, forward=False, prec="single",
            itype="complex", otype="complex",
        )
        torchfft._setup_mkl_cpu_ifft_plan(fftobj)

    assert calls == ([(size, 8)] if eligible else [])
    assert fftobj._mkl_plan is (plan if eligible else None)


@pytest.mark.parametrize("size", (1048576, 2097152, 4194304))
@pytest.mark.parametrize("threads", (1, 4))
def test_large_sizes_require_qualified_promoted_dispatch(
    monkeypatch, cache_domain, size, threads
):
    from pycbc.fft import torchfft

    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_MKL_DIRECT_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads)
    calls = []
    plan = object()

    def create_plan(length, source, target, nthreads, **kwargs):
        calls.append((length, nthreads, kwargs))
        return plan

    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create_plan)
    with scheme.TorchScheme("cpu"):
        fftobj = SimpleNamespace(
            invec=zeros(size, dtype=np.complex64),
            outvec=zeros(size, dtype=np.complex64),
            size=size, nbatch=1, forward=False, prec="single",
            itype="complex", otype="complex",
        )
        torchfft._setup_mkl_cpu_ifft_plan(fftobj)

    assert calls == ([(size, 1, {"promote": True})] if threads == 1 else [])
    assert fftobj._mkl_plan is (plan if threads == 1 else None)


def test_plan_tracks_active_threads_separately_from_descriptor_limit(monkeypatch):
    from pycbc.fft import torchfft

    properties = []

    def success(*args):
        return 0

    def create_descriptor(reference, *args):
        reference._obj.value = 9999
        return 0

    def set_value(descriptor, parameter, value):
        properties.append((parameter, value))
        return 0

    mkl = SimpleNamespace(
        DFTI_COMPLEX=32, DFTI_PLACEMENT=11, DFTI_NOT_INPLACE=44,
        DFTI_THREAD_LIMIT=27,
        lib=SimpleNamespace(
            DftiFreeDescriptor=success, DftiSetValue=set_value,
            DftiCommitDescriptor=success, DftiComputeBackward=success,
        ),
        mkl_descriptor={"single": create_descriptor},
        check_status=success,
    )
    source = torch.empty(64, dtype=torch.complex64)
    target = torch.empty_like(source)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    plan = torchfft._MKLCPUDirectIFFTPlan(mkl, 64, source, target, nthreads=2)

    assert (mkl.DFTI_THREAD_LIMIT, 2) in properties
    assert plan.can_execute(source, target)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    assert not plan.can_execute(source, target)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    assert plan.can_execute(source, target)
