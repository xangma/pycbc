# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Tests for guarded native reuse by the Torch CPU correlator."""

from types import SimpleNamespace

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def _arrays(size=4096, dtype=np.complex64):
    rng = np.random.default_rng(12617)
    x_values = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(dtype)
    y_values = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(dtype)
    return Array(x_values), Array(y_values), Array(np.empty_like(x_values))


def test_reusable_cpu_native_correlation_is_zero_copy_and_matches_cpu(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    observed = {}
    original = matchedfilter_cpu._correlate

    def record_native(x, y, z):
        observed.update(
            x_pointer=x.__array_interface__["data"][0],
            y_pointer=y.__array_interface__["data"][0],
            z_pointer=z.__array_interface__["data"][0],
            x_owner=isinstance(x.base, torch.Tensor),
            y_owner=isinstance(y.base, torch.Tensor),
            z_owner=isinstance(z.base, torch.Tensor),
        )
        return original(x, y, z)

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays()
        expected = torch.conj(x._data.tensor) * y._data.tensor
        x_numpy = x._data.tensor.detach().numpy()
        y_numpy = y._data.tensor.detach().numpy()
        legacy_expected = np.empty_like(x_numpy)
        original(x_numpy, y_numpy, legacy_expected)
        truth = np.conj(x_numpy.astype(np.complex128)) * y_numpy.astype(
            np.complex128
        )
        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: True,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", record_native)
        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)
        correlator.correlate()

    actual = z._data.tensor.detach().numpy()
    np.testing.assert_array_equal(actual, legacy_expected)
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(
        actual, expected.detach().numpy(), rtol=4 * eps, atol=4 * eps
    )
    relative_l2 = np.linalg.norm(actual.astype(np.complex128) - truth)
    relative_l2 /= np.linalg.norm(truth)
    assert relative_l2 <= eps
    assert isinstance(z._data, TorchArrayData)
    assert observed == {
        "x_pointer": x._data.tensor.data_ptr(),
        "y_pointer": y._data.tensor.data_ptr(),
        "z_pointer": z._data.tensor.data_ptr(),
        "x_owner": True,
        "y_owner": True,
        "z_owner": True,
    }


def test_reusable_correlator_avoids_native_one_thread_regression(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays(size=32768)
        expected = torch.conj(x._data.tensor) * y._data.tensor

        def fail_native(*args):
            raise AssertionError("one-thread call entered native correlation")

        monkeypatch.setattr(matchedfilter_torch.torch, "get_num_threads", lambda: 1)
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        matchedfilter_torch.TorchCorrelator(x, y, z).correlate()

    assert torch.equal(z._data.tensor, expected)


def test_reusable_correlator_selects_thread_crossover_once(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    calls = []
    dispatches = []
    original = matchedfilter_cpu._correlate

    def record_native(*args):
        calls.append("native")
        return original(*args)

    def select_native(length, runtime):
        dispatches.append(length)
        return True

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays(size=32768)
        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            select_native,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", record_native)
        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)
        correlator.correlate()
        correlator.correlate()

    assert dispatches == [32768]
    assert calls == ["native", "native"]


def test_openmp_runtime_resolution_is_cached(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    class FakeSymbol:
        def __init__(self, value):
            self.value = value

        def __call__(self):
            return self.value

    class FakeHandle:
        omp_get_max_threads = FakeSymbol(8)
        omp_get_dynamic = FakeSymbol(0)

    class FakeModule:
        __file__ = "/tmp/fake-matchedfilter-cpu.so"

    loads = []

    def load_runtime(path):
        loads.append(path)
        return FakeHandle()

    resolve = matchedfilter_torch._cpu_native_openmp_runtime
    resolve.cache_clear()
    try:
        monkeypatch.setattr(matchedfilter_torch.ctypes, "CDLL", load_runtime)
        first = resolve(FakeModule)
        second = resolve(FakeModule)
    finally:
        resolve.cache_clear()

    assert first is second
    assert loads == [FakeModule.__file__]
    assert first[1]() == 8
    assert first[2]() == 0


def test_torch_route_defers_numpy_views_and_reuses_lazy_conjugate(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    conjugate_calls = []
    original_conj = torch.conj

    def record_conj(tensor):
        conjugate_calls.append(tensor)
        return original_conj(tensor)

    def reject_numpy(*_args, **_kwargs):
        raise AssertionError("unselected native route created a NumPy view")

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays(size=32768)
        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: False,
        )
        monkeypatch.setattr(matchedfilter_torch.torch, "conj", record_conj)
        monkeypatch.setattr(torch.Tensor, "numpy", reject_numpy)

        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)
        assert correlator._cpu_native is None
        correlator.correlate()
        x._data.tensor.add_(1 + 2j)
        correlator.correlate()

        expected = original_conj(x._data.tensor) * y._data.tensor
        assert torch.equal(z._data.tensor, expected)
        assert conjugate_calls == [x._data.tensor]

        # Defensive attribute replacement must not reuse the stale view.
        replacement = x._data.tensor.clone().mul_(2 - 1j)
        correlator.x = replacement
        correlator.correlate()
        expected = original_conj(replacement) * y._data.tensor

    assert torch.equal(z._data.tensor, expected)
    assert conjugate_calls == [x._data.tensor, replacement]


def test_reusable_correlator_rechecks_reverse_ad_at_execution(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays(size=32768)
        expected = torch.conj(x._data.tensor) * y._data.tensor

        def fail_native(*args):
            raise AssertionError("mutable autograd input entered native")

        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: True,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)
        correlator.x.requires_grad_(True)
        with torch.no_grad():
            correlator.correlate()

    assert torch.equal(z._data.tensor, expected)


def test_reusable_correlator_rechecks_forward_ad_at_execution(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    fallback_calls = []
    original_mul = torch.mul

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays(size=32768)

        def fail_native(*args):
            raise AssertionError("forward dual entered native")

        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: True,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)

        with torch.autograd.forward_ad.dual_level():
            correlator.x = torch.autograd.forward_ad.make_dual(
                correlator.x, torch.ones_like(correlator.x)
            )

            def record_mul(left, right, *, out=None):
                fallback_calls.append(out)
                return original_mul(left, right, out=out)

            monkeypatch.setattr(matchedfilter_torch.torch, "mul", record_mul)
            with pytest.raises(NotImplementedError, match="forward AD"):
                correlator.correlate()

    assert fallback_calls == [z._data.tensor]


@pytest.mark.parametrize("kind", ("autograd", "complex128", "strided", "conj"))
def test_reusable_cpu_native_rejects_nonstandard_inputs(kind, monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    with scheme.TorchScheme("cpu"):
        if kind == "complex128":
            x, y, z = _arrays(dtype=np.complex128)
        else:
            x, y, z = _arrays()
        if kind == "autograd":
            x._data.tensor.requires_grad_(True)
        elif kind == "strided":
            x._data.tensor = x._data.tensor[::2]
            y._data.tensor = y._data.tensor[::2]
            z._data.tensor = z._data.tensor[::2]
        elif kind == "conj":
            x._data.tensor = x._data.tensor.conj()

        expected = torch.conj(x._data.tensor) * y._data.tensor

        def fail_native(*args):
            raise AssertionError(f"{kind} input entered native correlation")

        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: True,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        correlator = matchedfilter_torch.TorchCorrelator(x, y, z)
        if kind == "autograd":
            with torch.no_grad():
                correlator.correlate()
        else:
            correlator.correlate()

    assert torch.equal(z._data.tensor, expected)


def test_reusable_cpu_native_rejects_negative_bit_view(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    neg_view = getattr(torch, "_neg_view", None)
    if neg_view is None:
        pytest.skip("this Torch version cannot construct a negative-bit view")

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays()
        x._data.tensor = neg_view(x._data.tensor)
        expected = torch.conj(x._data.tensor) * y._data.tensor

        def fail_native(*args):
            raise AssertionError("negative-bit view entered native correlation")

        monkeypatch.setattr(
            matchedfilter_torch,
            "_cpu_native_dispatch_is_beneficial",
            lambda length, runtime: True,
        )
        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        matchedfilter_torch.TorchCorrelator(x, y, z).correlate()

    assert torch.equal(z._data.tensor, expected)


def test_cpu_native_crossover_is_conservative(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    cases = (
        (1, 1, 0, 131072, False),
        (16, 8, 0, 131072, False),
        (16, 16, 1, 131072, False),
        (16, 16, 0, 8191, False),
        (16, 16, 0, 8192, True),
        (64, 64, 0, 32767, False),
        (64, 64, 0, 32768, True),
        (65, 65, 0, 131072, False),
    )
    for torch_threads, openmp_threads, dynamic, length, expected in cases:
        monkeypatch.setattr(
            matchedfilter_torch.torch,
            "get_num_threads",
            lambda threads=torch_threads: threads,
        )
        runtime = (
            object(),
            lambda threads=openmp_threads: threads,
            lambda dynamic=dynamic: dynamic,
        )
        assert (
            matchedfilter_torch._cpu_native_dispatch_is_beneficial(
                length, runtime
            )
            is expected
        )
    assert not matchedfilter_torch._cpu_native_dispatch_is_beneficial(
        131072, None
    )


def test_cpu_native_static_eligibility_enforces_abi_and_no_alias(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    x = torch.zeros(8, dtype=torch.complex64)
    y = torch.ones(8, dtype=torch.complex64)
    z = torch.empty(8, dtype=torch.complex64)
    assert matchedfilter_torch._cpu_native_static_eligible(x, y, z)
    assert not matchedfilter_torch._cpu_native_static_eligible(x, y, x)
    assert not matchedfilter_torch._cpu_native_static_eligible(
        x, y[:-1], z
    )

    storage = torch.empty(9, dtype=torch.complex64)
    assert not matchedfilter_torch._cpu_native_static_eligible(
        storage[:-1], y, storage[1:]
    )

    monkeypatch.setattr(matchedfilter_torch, "_CPU_NATIVE_MAX_LENGTH", 7)
    assert not matchedfilter_torch._cpu_native_static_eligible(x, y, z)


def test_functional_and_batch_correlation_remain_on_torch(monkeypatch):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    with scheme.TorchScheme("cpu"):
        x, y, z = _arrays()
        batch_z = Array(np.empty(len(z), dtype=np.complex64))
        expected = torch.conj(x._data.tensor) * y._data.tensor

        def fail_native(*args):
            raise AssertionError("non-reusable operation entered native correlation")

        monkeypatch.setattr(matchedfilter_cpu, "_correlate", fail_native)
        matchedfilter_torch.correlate(x, y, z)
        batch = SimpleNamespace(xs=[x], zs=[batch_z])
        matchedfilter_torch.batch_correlate_execute(batch, y)

    assert torch.equal(z._data.tensor, expected)
    assert torch.equal(batch_z._data.tensor, expected)


def test_cpu_standard_peak_tensor_avoids_float64_and_matches_exact(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    # Create test 2D tensor on CPU
    rows = np.array([
        [0.0, 3.0 + 4.0j, 1.0, 2.0],
        [1.0, 2.0, -10.0j, 5.0],
        [7.0, -8.0, 2.0, 3.0],
    ], dtype=np.complex64)
    values = torch.from_numpy(rows)

    # Verify that .to(torch.float64) is NEVER called for CPU complex64
    original_to = torch.Tensor.to

    def guard_to(self, *args, **kwargs):
        dtype = kwargs.get("dtype", None)
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype == torch.float64:
            raise AssertionError("standard_peak_tensor converted CPU tensor to float64!")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guard_to)

    indices, peaks = matchedfilter_torch.standard_peak_tensor(values)

    np.testing.assert_array_equal(indices.numpy(), [1, 2, 1])
    np.testing.assert_allclose(
        peaks.numpy(),
        [3.0 + 4.0j, -10.0j, -8.0 + 0.0j],
    )
    assert peaks.dtype == torch.complex64


def test_cpu_standard_peak_tensor_real_float32(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    rows = np.array([
        [1.0, 5.0, 2.0],
        [-8.0, 3.0, 4.0],
    ], dtype=np.float32)
    values = torch.from_numpy(rows)

    original_to = torch.Tensor.to

    def guard_to(self, *args, **kwargs):
        dtype = kwargs.get("dtype", None)
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype == torch.float64:
            raise AssertionError("standard_peak_tensor converted CPU tensor to float64!")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guard_to)

    indices, peaks = matchedfilter_torch.standard_peak_tensor(values)
    np.testing.assert_array_equal(indices.numpy(), [1, 0])
    np.testing.assert_allclose(peaks.numpy(), [5.0, -8.0])
    assert peaks.dtype == torch.float32


def test_cpu_batch_peak_and_threshold_avoids_float64(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    rows = np.zeros((10, 16), dtype=np.complex64)
    rows[3, 5] = 6.0 + 8.0j  # magnitude 10.0
    values = torch.from_numpy(rows)
    norms = np.ones(10, dtype=np.float32)

    original_to = torch.Tensor.to

    def guard_to(self, *args, **kwargs):
        dtype = kwargs.get("dtype", None)
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype == torch.float64:
            raise AssertionError("CPU thresholding converted tensor to float64!")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guard_to)

    surv, p_idx, p_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=9.0
        )
    )
    assert not aborted
    np.testing.assert_array_equal(surv, [3])
    np.testing.assert_array_equal(p_idx, [5])
    np.testing.assert_allclose(p_val, [6.0 + 8.0j])

