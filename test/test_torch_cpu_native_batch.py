# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch filtering regression tests."""


import os
import threading
import numpy as np
import pytest
import pycbc
from pycbc import scheme
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.types import Array, zeros
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


CORRELATION_GATE = "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(np.complex64)


def _make_batch(rows=3, size=128, total=None):
    if total is None:
        total = size
    x_values = _complex_values(rows, total, seed=6101)
    y_values = _complex_values(1, total, seed=6102)[0]
    xs = [Array(row) for row in x_values]
    y = Array(y_values)
    zs = [
        Array(np.full(total, 19 - 7j, dtype=np.complex64))
        for _ in range(rows)
    ]
    return BatchCorrelator(xs, zs, size), y


def _legacy_correlation(batch, y):
    from pycbc.filter import matchedfilter_cpu

    expected = []
    y_values = y._data.tensor.detach().numpy()[: batch.size]
    for x in batch.xs:
        output = np.empty(batch.size, dtype=np.complex64)
        matchedfilter_cpu._correlate(
            x._data.tensor.detach().numpy()[: batch.size],
            y_values,
            output,
        )
        expected.append(output)
    return expected


def _torch_correlation(batch, y):
    return [
        torch.conj(x._data.tensor[: batch.size])
        * y._data.tensor[: batch.size]
        for x in batch.xs
    ]


def _enable_native_correlation(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    monkeypatch.setenv(CORRELATION_GATE, "1")
    monkeypatch.setattr(
        matchedfilter_torch,
        "_cpu_native_batch_runtime_is_stable",
        lambda runtime: True,
    )


def test_batch_correlation_gate_is_strict_and_default_off(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    monkeypatch.delenv(CORRELATION_GATE, raising=False)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch()
        expected = _torch_correlation(batch, y)

        def fail_native(*args):
            raise AssertionError("default-off route entered native code")

        monkeypatch.setattr(matchedfilter_cpu, "_batch_correlate", fail_native)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

    monkeypatch.setenv(CORRELATION_GATE, "sometimes")
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch()
        with pytest.raises(ValueError, match=CORRELATION_GATE):
            batch.execute(y)


def test_native_batch_correlation_is_zero_copy_exact_and_reusable(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_correlation(monkeypatch)
    native = matchedfilter_cpu._batch_correlate
    y_pointers = []

    def observed_native(x_pointers, y_view, z_pointers, size, rows):
        y_pointers.append(y_view.__array_interface__["data"][0])
        native(x_pointers, y_view, z_pointers, size, rows)

    monkeypatch.setattr(
        matchedfilter_cpu, "_batch_correlate", observed_native
    )
    size, total = 129, 193
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(rows=4, size=size, total=total)
        expected = _legacy_correlation(batch, y)
        versions = [z._data.tensor._version for z in batch.zs]
        tails = [z._data.tensor[size:].clone() for z in batch.zs]
        batch.execute(y)

        state = batch._torch_cpu_native_batch_state
        assert y_pointers == [y._data.tensor.data_ptr()]
        np.testing.assert_array_equal(
            state._x_pointer_table,
            [x._data.tensor.data_ptr() for x in batch.xs],
        )
        np.testing.assert_array_equal(
            state._z_pointer_table,
            [z._data.tensor.data_ptr() for z in batch.zs],
        )
        assert not state._x_pointer_table.flags.writeable
        assert not state._z_pointer_table.flags.writeable
        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            np.testing.assert_array_equal(output.numpy()[:size], truth)
            assert torch.equal(output._data.tensor[size:], tails[index])
            assert output._data.tensor._version == versions[index] + 1

        # Content mutation is the reusable-buffer contract, not pointer drift.
        batch.xs[0]._data.tensor.mul_(2 - 0.5j)
        expected = _legacy_correlation(batch, y)
        versions = [z._data.tensor._version for z in batch.zs]
        batch.execute(y)
        assert batch._torch_cpu_native_batch_state is state
        assert y_pointers[-1] == y._data.tensor.data_ptr()
        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            np.testing.assert_array_equal(output.numpy()[:size], truth)
            assert torch.equal(output._data.tensor[size:], tails[index])
            assert output._data.tensor._version == versions[index] + 1


@pytest.mark.parametrize(
    "drift",
    ("x_rebind", "z_rebind", "pid", "thread", "openmp", "epoch"),
)
def test_native_batch_correlation_drift_fails_closed(monkeypatch, drift):
    from pycbc.filter import matchedfilter_torch

    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch()
        batch.execute(y)
        state = batch._torch_cpu_native_batch_state

        def fail_stale_state(*args):
            raise AssertionError("invalidated native state executed")

        monkeypatch.setattr(state, "_function", fail_stale_state)
        if drift == "x_rebind":
            replacement = batch.xs[0]._data.tensor.clone().mul_(2 + 1j)
            batch.xs[0]._data._set_tensor(replacement)
        elif drift == "z_rebind":
            replacement = torch.empty_like(batch.zs[0]._data.tensor)
            batch.zs[0]._data._set_tensor(replacement)
        elif drift == "pid":
            monkeypatch.setattr(
                matchedfilter_torch.os, "getpid", lambda: state._pid + 1
            )
        elif drift == "thread":
            monkeypatch.setattr(
                matchedfilter_torch.threading,
                "get_ident",
                lambda: state._thread_id + 1,
            )
        elif drift == "epoch":
            batch.mark_dirty()
        else:
            monkeypatch.setattr(
                matchedfilter_torch,
                "_cpu_native_batch_runtime_is_stable",
                lambda runtime: False,
            )

        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert batch._torch_cpu_native_batch_state is state
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor[: batch.size], truth)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("target", ("x", "y", "z"))
@pytest.mark.parametrize("drift", ("stride", "rank", "length", "autograd"))
def test_native_batch_rechecks_inplace_tensor_metadata(
    monkeypatch, device, target, drift
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _enable_native_correlation(monkeypatch)
    monkeypatch.setenv("PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE", "1")
    size, total = 64, 128
    with scheme.TorchScheme(device):
        batch, y = _make_batch(rows=3, size=size, total=total)
        # Pack both sides so native admission does not depend on the allocator
        # placing separate template allocations at uniformly spaced addresses.
        for arrays in (batch.xs, batch.zs):
            memory = zeros(batch.num_vectors * total, dtype=np.complex64)
            packed = [
                memory[row * total:(row + 1) * total]
                for row in range(batch.num_vectors)
            ]
            for source, target_array in zip(arrays, packed):
                target_array._data.tensor.copy_(source._data.tensor)
            arrays[:] = packed
        batch.execute(y)
        state = getattr(batch, f"_torch_{device}_native_batch_state")
        tensor = {
            "x": batch.xs[0]._data.tensor,
            "y": y._data.tensor,
            "z": batch.zs[0]._data.tensor,
        }[target]
        pointer = tensor.data_ptr()
        if drift == "stride":
            tensor.as_strided_((size,), (2,))
        elif drift == "rank":
            tensor.as_strided_((2, size), (size, 1))
        elif drift == "length":
            tensor.as_strided_((size - 1,), (1,))
        else:
            tensor.requires_grad_(True)
        assert tensor.data_ptr() == pointer
        before = [z._data.tensor.detach().clone() for z in batch.zs]

        assert state.execute(batch, y) is False
        for output, unchanged in zip(batch.zs, before):
            assert torch.equal(output._data.tensor, unchanged)

        if drift == "stride":
            expected = _torch_correlation(batch, y)
            batch.execute(y)
            for output, truth in zip(batch.zs, expected):
                torch.testing.assert_close(output._data.tensor[:size], truth)


def test_native_batch_correlation_accepts_dynamic_y_and_recovers(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_correlation(monkeypatch)
    native = matchedfilter_cpu._batch_correlate
    y_pointers = []

    def observed_native(x_pointers, y_view, z_pointers, size, rows):
        y_pointers.append(y_view.__array_interface__["data"][0])
        native(x_pointers, y_view, z_pointers, size, rows)

    monkeypatch.setattr(
        matchedfilter_cpu, "_batch_correlate", observed_native
    )
    with scheme.TorchScheme("cpu"):
        batch, y1 = _make_batch(rows=3, size=64)
        for y in (
            y1,
            Array(_complex_values(1, 64, seed=6151)[0]),
        ):
            expected = _legacy_correlation(batch, y)
            versions = [z._data.tensor._version for z in batch.zs]
            batch.execute(y)
            assert y_pointers[-1] == y._data.tensor.data_ptr()
            for index, (output, truth) in enumerate(zip(batch.zs, expected)):
                np.testing.assert_array_equal(output.numpy(), truth)
                assert output._data.tensor._version == versions[index] + 1

        state = batch._torch_cpu_native_batch_state
        calls_before_fallback = len(y_pointers)
        storage = torch.empty(65, dtype=torch.complex64)
        unsafe_tensor = storage[1:]
        unsafe_tensor.copy_(
            torch.from_numpy(_complex_values(1, 64, seed=6152)[0])
        )
        assert unsafe_tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT != 0
        unsafe_y = Array(TorchArrayData(unsafe_tensor), copy=False)
        expected = _torch_correlation(batch, unsafe_y)
        batch.execute(unsafe_y)
        assert len(y_pointers) == calls_before_fallback
        assert batch._torch_cpu_native_batch_state is state
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

        y3 = Array(_complex_values(1, 64, seed=6153)[0])
        expected = _legacy_correlation(batch, y3)
        versions = [z._data.tensor._version for z in batch.zs]
        batch.execute(y3)
        assert len(y_pointers) == calls_before_fallback + 1
        assert y_pointers[-1] == y3._data.tensor.data_ptr()
        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            np.testing.assert_array_equal(output.numpy(), truth)
            assert output._data.tensor._version == versions[index] + 1


def test_native_batch_correlation_dynamic_ad_y_uses_torch(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_correlation(monkeypatch)
    calls = 0
    native = matchedfilter_cpu._batch_correlate

    def observed_native(*args):
        nonlocal calls
        calls += 1
        native(*args)

    monkeypatch.setattr(
        matchedfilter_cpu, "_batch_correlate", observed_native
    )
    with scheme.TorchScheme("cpu"):
        batch, y1 = _make_batch(rows=2, size=64)
        batch.execute(y1)
        assert calls == 1
        ad_y = Array(_complex_values(1, 64, seed=6161)[0])
        ad_y._data.tensor.requires_grad_(True)
        with pytest.raises(RuntimeError):
            batch.execute(ad_y)
        assert calls == 1
        y3 = Array(_complex_values(1, 64, seed=6162)[0])
        expected = _legacy_correlation(batch, y3)
        batch.execute(y3)
        assert calls == 2
        for output, truth in zip(batch.zs, expected):
            np.testing.assert_array_equal(output.numpy(), truth)


@pytest.mark.parametrize("native_enabled", [False, True])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_batch_correlation_dynamic_ad_y_preserves_outputs(
    monkeypatch, native_enabled, reverse_rows
):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    _enable_native_correlation(monkeypatch)
    monkeypatch.setenv(CORRELATION_GATE, str(int(native_enabled)))
    calls = []
    native = matchedfilter_cpu._batch_correlate

    def observed_native(*args):
        calls.append(True)
        native(*args)

    monkeypatch.setattr(matchedfilter_cpu, "_batch_correlate", observed_native)
    with scheme.TorchScheme("cpu"):
        size = 64
        x_mem = Array(_complex_values(2, size, seed=6163).reshape(-1))
        z_mem = Array(np.full(2 * size, 19 - 7j, dtype=np.complex64))
        order = (1, 0) if reverse_rows else (0, 1)
        xs = [x_mem[i * size:(i + 1) * size] for i in order]
        zs = [z_mem[i * size:(i + 1) * size] for i in order]
        batch = BatchCorrelator(xs, zs, size)
        # Descending row addresses deterministically select the copied-output
        # fallback; ascending rows exercise the packed out= path.
        for arrays in (xs, zs):
            stride = matchedfilter_torch._find_uniform_stride(
                tuple(array._data.tensor for array in arrays), size
            )
            assert stride == (None if reverse_rows else size)

        y = Array(_complex_values(1, size, seed=6164)[0])
        batch.execute(y)
        assert len(calls) == int(native_enabled)
        tensors = tuple(z._data.tensor for z in zs)
        snapshots = tuple(tensor.clone() for tensor in tensors)
        versions = tuple(tensor._version for tensor in tensors)

        ad_y = Array(_complex_values(1, size, seed=6165)[0])
        ad_y._data.tensor.requires_grad_(True)
        with pytest.raises(RuntimeError, match="automatic differentiation"):
            batch.execute(ad_y)
        assert len(calls) == int(native_enabled)
        for tensor, snapshot, version in zip(tensors, snapshots, versions):
            assert torch.equal(tensor, snapshot)
            assert tensor._version == version
            assert not tensor.requires_grad
            assert tensor.grad_fn is None

        expected = (_legacy_correlation(batch, y) if native_enabled
                    else _torch_correlation(batch, y))
        batch.execute(y)
        assert len(calls) == 2 * int(native_enabled)
        for tensor, truth in zip(tensors, expected):
            torch.testing.assert_close(tensor, torch.as_tensor(truth),
                                       rtol=0, atol=0)


def test_native_batch_correlation_does_not_change_single_batch(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_correlation(monkeypatch)

    def fail_native(*args):
        raise AssertionError("B1 entered native batch correlation")

    monkeypatch.setattr(matchedfilter_cpu, "_batch_correlate", fail_native)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(rows=1, size=64)
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")
        assert torch.equal(batch.zs[0]._data.tensor, expected[0])


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_native_batch_correlation_real_thread_and_fork_decline(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cpu"):
        batch, y1 = _make_batch(rows=2, size=64)
        batch.execute(y1)
        state = batch._torch_cpu_native_batch_state

        def fail_native(*args):
            raise AssertionError("non-owner entered native correlation")

        monkeypatch.setattr(state, "_function", fail_native)
        y2 = Array(_complex_values(1, 64, seed=6171)[0])
        expected = _torch_correlation(batch, y2)
        errors = []

        def execute_in_thread():
            try:
                batch.execute(y2)
            except BaseException as error:  # pragma: no cover - diagnostic
                errors.append(error)

        worker = threading.Thread(target=execute_in_thread)
        worker.start()
        worker.join()
        assert errors == []
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:  # pragma: no cover - asserted through the pipe
            try:
                os.close(read_fd)
                os.write(write_fd, b"1" if state.can_execute(batch) else b"0")
            finally:
                os._exit(0)
        os.close(write_fd)
        child_result = os.read(read_fd, 1)
        os.close(read_fd)
        _, status = os.waitpid(child, 0)
        assert status == 0
        assert child_result == b"0"


class _TensorSubclass(torch.Tensor):
    pass


@pytest.mark.parametrize("contract", ("unaligned", "noncontiguous", "subclass"))
def test_native_batch_correlation_rejects_unsafe_tensor_contracts(
    monkeypatch, contract
):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(size=64)
        if contract == "unaligned":
            storage = torch.empty(65, dtype=torch.complex64)
            replacement = storage[1:]
            replacement.copy_(batch.xs[0]._data.tensor)
            assert replacement.data_ptr() % pycbc.PYCBC_ALIGNMENT != 0
        elif contract == "noncontiguous":
            storage = torch.empty(128, dtype=torch.complex64)
            replacement = storage[::2]
            replacement.copy_(batch.xs[0]._data.tensor)
            assert not replacement.is_contiguous()
        else:
            replacement = batch.xs[0]._data.tensor.as_subclass(
                _TensorSubclass
            )
        batch.xs[0]._data._set_tensor(replacement)
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_native_batch_correlation_rejects_output_alias(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(size=64)
        original = batch.xs[0]._data.tensor.clone()
        expected = torch.conj(original) * y._data.tensor
        batch.zs[0]._data._set_tensor(batch.xs[0]._data.tensor)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")
        assert torch.equal(batch.zs[0]._data.tensor, expected)


def test_native_batch_correlation_preserves_ad_and_inference_dispatch(
    monkeypatch,
):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(size=64)
        batch.xs[0]._data.tensor.requires_grad_(True)
        with pytest.raises(RuntimeError):
            batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")

        batch, y = _make_batch(size=64)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                batch.xs[0]._data.tensor,
                torch.ones_like(batch.xs[0]._data.tensor),
            )
            batch.xs[0]._data._set_tensor(dual)
            with pytest.raises(NotImplementedError, match="forward AD"):
                batch.execute(y)
            assert not hasattr(batch, "_torch_cpu_native_batch_state")

    with torch.inference_mode(), scheme.TorchScheme("cpu"):
        batch, y = _make_batch(size=64)
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cpu_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_native_batch_correlation_setup_failure_uses_torch(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    _enable_native_correlation(monkeypatch)

    def fail_setup(*args):
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(
        matchedfilter_torch, "_CPUNativeBatchCorrelationState", fail_setup
    )
    with scheme.TorchScheme("cpu"):
        batch, y = _make_batch(size=64)
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_native_batch_correlation_2d_packed_layout(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_correlation(monkeypatch)
    native = matchedfilter_cpu._batch_correlate
    called = []

    def observed_native(x_pointers, y_view, z_pointers, size, rows):
        called.append((len(x_pointers), len(z_pointers), size, rows))
        native(x_pointers, y_view, z_pointers, size, rows)

    monkeypatch.setattr(
        matchedfilter_cpu, "_batch_correlate", observed_native
    )
    rows, size = 4, 128
    with scheme.TorchScheme("cpu"):
        x_values = _complex_values(rows, size, seed=8101)
        y_values = _complex_values(1, size, seed=8102)[0]
        x_mem = zeros(rows * size, dtype=np.complex64)
        x_mem._data.tensor.copy_(torch.from_numpy(x_values.reshape(-1)))
        xs = [x_mem[i * size:(i + 1) * size] for i in range(rows)]
        y = Array(y_values)
        z_mem = zeros(rows * size, dtype=np.complex64)
        z_mem._data.tensor.fill_(19 - 7j)
        zs = [z_mem[i * size:(i + 1) * size] for i in range(rows)]

        batch = BatchCorrelator(xs, zs, size)
        expected = _legacy_correlation(batch, y)
        batch.execute(y)

        assert len(called) == 1
        assert called[0] == (rows, rows, size, rows)
        for output, truth in zip(batch.zs, expected):
            np.testing.assert_array_equal(output.numpy(), truth)


def test_torch_batch_correlation_2d_packed_layout_default(monkeypatch):
    monkeypatch.delenv(CORRELATION_GATE, raising=False)
    rows, size = 4, 128
    with scheme.TorchScheme("cpu"):
        x_values = _complex_values(rows, size, seed=8201)
        y_values = _complex_values(1, size, seed=8202)[0]
        x_mem = zeros(rows * size, dtype=np.complex64)
        x_mem._data.tensor.copy_(torch.from_numpy(x_values.reshape(-1)))
        xs = [x_mem[i * size:(i + 1) * size] for i in range(rows)]
        y = Array(y_values)
        z_mem = zeros(rows * size, dtype=np.complex64)
        z_mem._data.tensor.fill_(19 - 7j)
        zs = [z_mem[i * size:(i + 1) * size] for i in range(rows)]

        batch = BatchCorrelator(xs, zs, size)
        expected = _torch_correlation(batch, y)
        batch.execute(y)

        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_torch_batch_nonuniform_cache_tracks_in_place_mutation(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    monkeypatch.delenv(CORRELATION_GATE, raising=False)
    with scheme.TorchScheme("cpu"):
        rows, size = 3, 64
        x_values = _complex_values(rows, size, seed=8251)
        x_mem = zeros(4 * size, dtype=np.complex64)
        offsets = (0, size, 3 * size)
        xs = []
        for offset, values in zip(offsets, x_values):
            x = x_mem[offset:offset + size]
            x._data.tensor.copy_(torch.from_numpy(values))
            xs.append(x)
        y = Array(_complex_values(1, size, seed=8252)[0])
        z_mem = zeros(rows * size, dtype=np.complex64)
        zs = [z_mem[i * size:(i + 1) * size] for i in range(rows)]
        batch = BatchCorrelator(xs, zs, size)
        x_tensors = tuple(x._data.tensor for x in batch.xs)
        assert matchedfilter_torch._find_uniform_stride(
            x_tensors, batch.size
        ) is None

        batch.execute(y)
        batch.xs[1]._data.tensor.mul_(2.0 - 0.5j)
        expected = _torch_correlation(batch, y)
        batch.execute(y)

        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

        batch.xs[1].numpy()[:] *= -0.25 + 1.5j
        expected = _torch_correlation(batch, y)
        batch.execute(y)

        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)
