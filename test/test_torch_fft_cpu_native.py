# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch fft cpu native regression tests."""

import gc
import types
import numpy as np
import pytest
import pycbc
from pycbc import scheme
from pycbc.types import Array, zeros
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


FFTW_BATCH_GATE = "PYCBC_TORCH_CPU_FFTW_BATCH"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(rows, size)) + 1j * rng.normal(size=(rows, size))).astype(
        np.complex64
    )


class _TensorSubclass(torch.Tensor):
    pass


@pytest.fixture
def direct_batch_fftw(monkeypatch):
    from pycbc.fft import fftw, torchfft

    old_threads = torch.get_num_threads()
    old_measure = fftw.get_measure_level()
    torch.set_num_threads(1)
    fftw.set_measure_level(0)
    monkeypatch.setenv(FFTW_BATCH_GATE, "1")
    monkeypatch.setattr(torchfft, "_FFTW_DIRECT_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(torchfft, "_FFTW_DIRECT_BATCH_SIZES", frozenset({64}))
    try:
        yield torchfft
    finally:
        fftw.set_measure_level(old_measure)
        torch.set_num_threads(old_threads)


def _legacy_batch_fft(values, inverse, aligned=True):
    from pycbc.fft import fftw

    rows, size = values.shape
    total = rows * size
    with scheme.CPUScheme(num_threads=1):
        source_storage = zeros(total + int(not aligned), dtype=np.complex64)
        target_storage = zeros(total + int(not aligned), dtype=np.complex64)
        if aligned:
            source = source_storage
            target = target_storage
        else:
            source = Array(source_storage.data[1:], copy=False)
            target = Array(target_storage.data[1:], copy=False)
        source.data[:] = values.ravel()
        engine_type = fftw.IFFT if inverse else fftw.FFT
        engine_type(source, target, nbatch=rows, size=size).execute()
        return target.numpy().copy()


def _torch_batch_fft_arrays(values, aligned=True):
    total = values.size
    if aligned:
        source_tensor = torch.empty(total, dtype=torch.complex64)
        target_tensor = torch.empty_like(source_tensor)
    else:
        source_tensor = torch.empty(total + 1, dtype=torch.complex64)[1:]
        target_tensor = torch.empty(total + 1, dtype=torch.complex64)[1:]
    source_tensor.copy_(torch.from_numpy(values.ravel()))
    assert (source_tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT == 0) is aligned
    assert (target_tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT == 0) is aligned
    return (
        Array(TorchArrayData(source_tensor), copy=False),
        Array(TorchArrayData(target_tensor), copy=False),
    )


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("aligned", (False, True))
def test_direct_batch_fftw_is_bitwise_legacy_exact(direct_batch_fftw, inverse, aligned):
    torchfft = direct_batch_fftw
    values = _complex_values(3, 64, seed=6201 + inverse)
    replacement = _complex_values(3, 64, seed=6301 + inverse)
    expected = _legacy_batch_fft(values, inverse, aligned=aligned)
    replacement_expected = _legacy_batch_fft(replacement, inverse, aligned=aligned)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values, aligned=aligned)
        engine_type = torchfft.IFFT if inverse else torchfft.FFT
        engine = engine_type(source, target, nbatch=3, size=64)
        if engine._fftw_batch_plan is None:
            pytest.skip("single-precision FFTW is unavailable")
        plan = engine._fftw_batch_plan
        assert isinstance(plan, torchfft._FFTWCPUDirectBatchPlan)
        assert engine._promoted_batch_plan is None
        assert plan._source is source._data.tensor
        assert plan._target is target._data.tensor
        assert plan._size == 64
        assert plan._batch == 3
        assert plan._forward is (not inverse)
        assert plan._aligned is aligned
        source_before = source._data.tensor.clone()
        source_version = source._data.tensor._version
        target_version = target._data.tensor._version
        engine.execute()
        assert torch.equal(source._data.tensor, source_before)
        assert source._data.tensor._version == source_version
        assert target._data.tensor._version == target_version + 1
        np.testing.assert_array_equal(target.numpy(), expected)

        # The plan remains bound while contents change in place.
        source._data.tensor.copy_(torch.from_numpy(replacement.ravel()))
        target_version = target._data.tensor._version
        engine.execute()
        assert target._data.tensor._version == target_version + 1
        np.testing.assert_array_equal(target.numpy(), replacement_expected)


def test_direct_batch_fftw_gate_is_strict_and_default_off(monkeypatch):
    from pycbc.fft import torchfft

    monkeypatch.setattr(torchfft, "_FFTW_DIRECT_BATCH_SIZES", frozenset({64}))
    monkeypatch.delenv(FFTW_BATCH_GATE, raising=False)
    with scheme.TorchScheme("cpu"):
        source = zeros(128, dtype=np.complex64)
        target = zeros(128, dtype=np.complex64)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        assert engine._fftw_batch_plan is None

    monkeypatch.setenv(FFTW_BATCH_GATE, "sometimes")
    with scheme.TorchScheme("cpu"):
        source = zeros(128, dtype=np.complex64)
        target = zeros(128, dtype=np.complex64)
        with pytest.raises(ValueError, match=FFTW_BATCH_GATE):
            torchfft.IFFT(source, target, nbatch=2, size=64)


@pytest.mark.parametrize(
    "drift", ("gate", "source", "target", "pid", "thread", "threads")
)
def test_direct_batch_fftw_runtime_drift_falls_back(
    direct_batch_fftw, monkeypatch, drift
):
    torchfft = direct_batch_fftw
    values = _complex_values(2, 64, seed=6401)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        if engine._fftw_batch_plan is None:
            pytest.skip("single-precision FFTW is unavailable")
        plan = engine._fftw_batch_plan
        assert engine._promoted_batch_plan is None

        def fail_stale_plan(*args):
            raise AssertionError("invalidated direct plan executed")

        monkeypatch.setattr(plan, "execute", fail_stale_plan)
        if drift == "gate":
            monkeypatch.setenv(FFTW_BATCH_GATE, "0")
        elif drift == "source":
            replacement = source._data.tensor.clone().mul_(2 - 1j)
            source._data._set_tensor(replacement)
        elif drift == "target":
            target._data._set_tensor(torch.empty_like(target._data.tensor))
        elif drift == "pid":
            monkeypatch.setattr(torchfft.os, "getpid", lambda: plan._pid + 1)
        elif drift == "thread":
            monkeypatch.setattr(
                torchfft.threading,
                "get_ident",
                lambda: plan._thread_id + 1,
            )
        else:
            monkeypatch.setattr(torchfft.torch, "get_num_threads", lambda: 2)

        current = source._data.tensor.detach().numpy().reshape(2, 64)
        truth = np.fft.ifft(current.astype(np.complex128), axis=-1) * 64
        engine.execute()
        assert engine._promoted_batch_plan is not None
        if drift == "pid":
            # Restore owner identity before the plan becomes unreachable so
            # this parent-process test does not deliberately leak its plan.
            monkeypatch.setattr(torchfft.os, "getpid", lambda: plan._pid)
        np.testing.assert_allclose(
            target.numpy().reshape(2, 64), truth, rtol=2e-6, atol=2e-6
        )


def test_direct_batch_fftw_does_not_change_single_batch_dispatch(
    direct_batch_fftw,
):
    torchfft = direct_batch_fftw
    values = _complex_values(1, 64, seed=6451)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        engine = torchfft.IFFT(source, target, nbatch=1, size=64)
        assert engine._fftw_batch_plan is None
        assert engine._promoted_batch_plan is None


def test_direct_batch_fftw_rejects_unsafe_contracts(direct_batch_fftw, monkeypatch):
    torchfft = direct_batch_fftw
    values = _complex_values(2, 64, seed=6501)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        source._data._set_tensor(source._data.tensor.as_subclass(_TensorSubclass))
        assert torchfft.IFFT(source, target, nbatch=2, size=64)._fftw_batch_plan is None

        source, target = _torch_batch_fft_arrays(values)
        storage = torch.empty(256, dtype=torch.complex64)
        noncontiguous = storage[::2]
        noncontiguous.copy_(source._data.tensor)
        source._data._set_tensor(noncontiguous)
        assert not source._data.tensor.is_contiguous()
        assert torchfft.IFFT(source, target, nbatch=2, size=64)._fftw_batch_plan is None

        source, target = _torch_batch_fft_arrays(values)
        target._data._set_tensor(source._data.tensor)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        assert engine._fftw_batch_plan is None
        engine.execute()
        truth = np.fft.ifft(values.astype(np.complex128), axis=-1) * 64
        np.testing.assert_allclose(
            target.numpy().reshape(2, 64), truth, rtol=2e-6, atol=2e-6
        )

        source, target = _torch_batch_fft_arrays(values)
        source._data.tensor.requires_grad_(True)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        assert engine._fftw_batch_plan is None
        with pytest.raises(RuntimeError):
            engine.execute()

    with torch.inference_mode(), scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        assert engine._fftw_batch_plan is None


def test_direct_batch_fftw_rejects_forward_ad(direct_batch_fftw):
    torchfft = direct_batch_fftw
    values = _complex_values(2, 64, seed=6601)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        with torch.autograd.forward_ad.dual_level():
            source._data._set_tensor(
                torch.autograd.forward_ad.make_dual(
                    source._data.tensor,
                    torch.ones_like(source._data.tensor),
                )
            )
            engine = torchfft.IFFT(source, target, nbatch=2, size=64)
            assert engine._fftw_batch_plan is None
            with pytest.raises(NotImplementedError, match="forward AD"):
                engine.execute()


def test_direct_batch_fftw_setup_failure_falls_back(direct_batch_fftw, monkeypatch):
    torchfft = direct_batch_fftw
    values = _complex_values(2, 64, seed=6701)
    monkeypatch.setattr(
        torchfft, "_create_fftw_cpu_batch_plan", lambda fftobj, forward: None
    )
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        engine = torchfft.IFFT(source, target, nbatch=2, size=64)
        assert engine._fftw_batch_plan is None
        engine.execute()
        truth = np.fft.ifft(values.astype(np.complex128), axis=-1) * 64
        np.testing.assert_allclose(
            target.numpy().reshape(2, 64), truth, rtol=2e-6, atol=2e-6
        )


def test_batch_plan_destructor_is_fork_safe(monkeypatch):
    from pycbc.fft import torchfft

    calls = []

    class FakeFFTW:
        @staticmethod
        def _destroy_plan(destroy, plan):
            calls.append((destroy, plan))

    monkeypatch.setattr(torchfft.os, "getpid", lambda: 42)
    torchfft._destroy_batch_plan_in_owner(FakeFFTW, "destroy", "plan", 41)
    assert calls == []
    torchfft._destroy_batch_plan_in_owner(FakeFFTW, "destroy", "plan", 42)
    assert calls == [("destroy", "plan")]
    gc.collect()


def test_mkl_descriptor_destructor_is_fork_safe(monkeypatch):
    from pycbc.fft import torchfft

    calls = []

    def fake_free(descriptor_ptr):
        calls.append(descriptor_ptr)

    monkeypatch.setattr(torchfft.os, "getpid", lambda: 42)
    torchfft._free_mkl_descriptor(fake_free, 12345, 41)
    assert calls == []
    torchfft._free_mkl_descriptor(fake_free, 12345, 42)
    assert len(calls) == 1
    gc.collect()


@pytest.mark.parametrize("drift", ("none", "gate", "source", "target"))
def test_single_ifft_rechecks_mkl_plan_before_execution(monkeypatch, drift):
    from pycbc.fft import torchfft

    calls = []

    def create_plan(size, source, target, nthreads):
        def execute(current_source, current_target):
            calls.append("mkl")
            current_target.copy_(torch.fft.ifft(current_source, norm="forward"))

        return types.SimpleNamespace(
            can_execute=lambda current_source, current_target: (
                current_source is source and current_target is target
            ),
            execute=execute,
        )

    monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "1")
    monkeypatch.setattr(torchfft, "_can_use_mkl_cpu_ifft", lambda obj: True)
    monkeypatch.setattr(torchfft, "_create_mkl_cpu_ifft_plan", create_plan)
    with scheme.TorchScheme("cpu"):
        source = Array(_complex_values(1, 64, seed=8301)[0])
        target = zeros(64, dtype=np.complex64)
        engine = torchfft.IFFT(source, target)
        if drift == "gate":
            monkeypatch.setenv("PYCBC_TORCH_CPU_MKL_IFFT", "0")
        elif drift == "source":
            source._data._set_tensor(source._data.tensor.clone().mul_(2 - 1j))
        elif drift == "target":
            target._data._set_tensor(torch.empty_like(target._data.tensor))

        truth = np.fft.ifft(source.numpy().astype(np.complex128)) * 64
        engine.execute()
        assert calls == (["mkl"] if drift == "none" else [])
        np.testing.assert_allclose(target.numpy(), truth, rtol=2e-6, atol=2e-6)


def test_mkl_direct_ifft_plan_multithreaded_support(monkeypatch):
    from pycbc.fft import torchfft

    calls = []

    class FakeMKL:
        DFTI_COMPLEX = 32
        DFTI_PLACEMENT = 11
        DFTI_NOT_INPLACE = 44
        DFTI_THREAD_LIMIT = 27

        def __init__(self):
            class Lib:
                @staticmethod
                def DftiFreeDescriptor(desc_ref):
                    return 0

                @staticmethod
                def DftiSetValue(desc, param, val):
                    calls.append(("SetValue", param, val))
                    return 0

                @staticmethod
                def DftiCommitDescriptor(desc):
                    calls.append(("Commit", desc))
                    return 0

                @staticmethod
                def DftiComputeBackward(desc, in_ptr, out_ptr):
                    calls.append(("ComputeBackward", in_ptr, out_ptr))
                    return 0

            self.lib = Lib()
            self.mkl_descriptor = {
                "single": lambda desc_ref, dom, sz: (
                    setattr(desc_ref._obj, "value", 9999) or 0
                )
            }

        @staticmethod
        def check_status(status):
            assert status == 0

    fake_mkl = FakeMKL()
    source = torch.empty(32768, dtype=torch.complex64, device="cpu")
    target = torch.empty(32768, dtype=torch.complex64, device="cpu")

    orig_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        plan = torchfft._MKLCPUDirectIFFTPlan(
            fake_mkl, 32768, source, target, nthreads=4
        )
        assert plan._nthreads == 4
        assert ("SetValue", FakeMKL.DFTI_THREAD_LIMIT, 4) in calls
        assert plan.can_execute(source, target)

        # Thread drift should fail can_execute
        torch.set_num_threads(2)
        assert not plan.can_execute(source, target)
        torch.set_num_threads(4)
        assert plan.can_execute(source, target)
    finally:
        torch.set_num_threads(orig_threads)
