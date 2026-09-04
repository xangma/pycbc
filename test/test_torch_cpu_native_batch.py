# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Qualification tests for the opt-in Torch-CPU native batch routes."""

import gc
import os
import threading
import types

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.types import Array, zeros
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


CORRELATION_GATE = "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"
FFTW_BATCH_GATE = "PYCBC_TORCH_CPU_FFTW_BATCH"
PEAK_GATE = "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"


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


def _enable_native_peak(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    monkeypatch.setenv(PEAK_GATE, "1")
    monkeypatch.setattr(
        matchedfilter_torch,
        "_cpu_native_batch_runtime_is_stable",
        lambda runtime: True,
    )


def _standard_peak_values(rows, segment):
    from pycbc.types import array_cpu

    start, stop, step = segment.indices(rows.shape[1])
    assert step == 1 and start < stop
    indices = np.empty(len(rows), dtype=np.int64)
    peaks = np.empty(len(rows), dtype=np.complex64)
    for row_index, row in enumerate(rows):
        values = np.ascontiguousarray(row[start:stop])
        index = array_cpu.abs_arg_max_complex(values)
        indices[row_index] = index
        peaks[row_index] = row[start + index]
    return indices, peaks


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
        # CUDA's packed output requires rows in one allocation.
        z_memory = zeros(batch.num_vectors * total, dtype=np.complex64)
        batch.zs[:] = [
            z_memory[row * total:(row + 1) * total]
            for row in range(batch.num_vectors)
        ]
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


def _peak_test_rows():
    rows = np.zeros((6, 12), dtype=np.complex64)
    rows[:, 0] = np.complex64(300 + 400j)
    rows[:, 11] = np.complex64(-600j)
    rows[0, 2:10] = np.array(
        [0, -0j, 3 + 4j, -3 - 4j, 1, 2, 4, 0],
        dtype=np.complex64,
    )
    rows[1, 2:10] = np.array(
        [1, np.nan + 0j, 6, 5, 4, 3, 2, 1],
        dtype=np.complex64,
    )
    rows[2, 2:10] = np.array(
        [1, 2, np.inf + 0j, -np.inf + 0j, 9, 8, 7, 6],
        dtype=np.complex64,
    )
    rows[3, 2:10] = np.array(
        [np.nan + 1j, np.nan - 2j, 0, -0j, 0, 0, 0, 0],
        dtype=np.complex64,
    )
    rows[4, 2:10] = np.array(
        [1e20 + 1e20j, -2e20 + 0j, 3, 4, 5, 6, 7, 8],
        dtype=np.complex64,
    )
    rows[5, 2:10] = np.array(
        [complex(-0.0, 0.0), complex(0.0, -0.0), 0, 0, 0, 0, 0, 0],
        dtype=np.complex64,
    )
    return rows


def test_native_batch_peak_is_zero_copy_standard_exact_and_reusable(
    monkeypatch,
):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_peak(monkeypatch)
    rows = _peak_test_rows()
    segment = slice(-10, -2)
    expected_indices, expected_peaks = _standard_peak_values(rows, segment)
    calls = []
    native = matchedfilter_cpu._batch_abs_arg_max_complex64

    def capture_native(values, indices, peaks, *geometry):
        calls.append(
            (
                values.__array_interface__["data"][0],
                indices.dtype,
                peaks.dtype,
                geometry,
            )
        )
        return native(values, indices, peaks, *geometry)

    monkeypatch.setattr(
        matchedfilter_cpu,
        "_batch_abs_arg_max_complex64",
        capture_native,
    )
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        tensor = output._data.tensor
        pointer = tensor.data_ptr()
        version = tensor._version
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, len(rows), rows.shape[1], segment
        )

        assert calls == [
            (
                pointer,
                np.dtype(np.int64),
                np.dtype(np.complex64),
                (rows.shape[1], 2, 10, len(rows)),
            )
        ]
        assert tensor.data_ptr() == pointer
        assert tensor._version == version
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(
            peaks.view(np.uint64), expected_peaks.view(np.uint64)
        )

        # Reusing the workspace with new contents must read the current bytes.
        tensor[2] = 11 - 13j
        current = tensor.detach().numpy().reshape(rows.shape).copy()
        expected_indices, expected_peaks = _standard_peak_values(
            current, segment
        )
        version = tensor._version
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, len(rows), rows.shape[1], segment
        )
        assert tensor._version == version
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(
            peaks.view(np.uint64), expected_peaks.view(np.uint64)
        )


def test_native_batch_peak_gate_is_strict_default_off_and_corrects_nan(
    monkeypatch,
):
    from pycbc.filter import matchedfilter_cpu

    native = matchedfilter_cpu._batch_abs_arg_max_complex64
    rows = np.zeros((2, 8), dtype=np.complex64)
    rows[0, 2:6] = [1, np.nan, 3, 2]
    rows[1, 2:6] = [2, -2, 1, 0]
    segment = slice(2, 6)
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        monkeypatch.delenv(PEAK_GATE, raising=False)

        def fail_default_off(*args):
            raise AssertionError("default-off route entered native code")

        monkeypatch.setattr(
            matchedfilter_cpu,
            "_batch_abs_arg_max_complex64",
            fail_default_off,
        )
        helper_indices, helper_peaks = (
            matchedfilter._torch_batch_peak_values(output, 2, 8, segment)
        )
        assert helper_indices[0] == 1
        assert np.isnan(helper_peaks[0])

        monkeypatch.setenv(PEAK_GATE, "sometimes")
        with pytest.raises(ValueError, match=PEAK_GATE):
            matchedfilter._torch_batch_peak_values(output, 2, 8, segment)

    monkeypatch.setattr(
        matchedfilter_cpu,
        "_batch_abs_arg_max_complex64",
        native,
    )
    _enable_native_peak(monkeypatch)
    expected_indices, expected_peaks = _standard_peak_values(rows, segment)
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, 2, 8, segment
        )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(
        peaks.view(np.uint64), expected_peaks.view(np.uint64)
    )
    assert indices[0] == expected_indices[0]


def test_native_batch_peak_supports_single_batch(monkeypatch):
    from pycbc.filter import matchedfilter_cpu

    _enable_native_peak(monkeypatch)
    rows = _complex_values(1, 16, seed=6751)
    calls = []
    native = matchedfilter_cpu._batch_abs_arg_max_complex64

    def capture_native(values, indices, peaks, *geometry):
        calls.append(
            (
                values.__array_interface__["data"][0],
                indices.dtype,
                peaks.dtype,
                geometry,
            )
        )
        return native(values, indices, peaks, *geometry)

    monkeypatch.setattr(
        matchedfilter_cpu,
        "_batch_abs_arg_max_complex64",
        capture_native,
    )
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        tensor = output._data.tensor
        pointer = tensor.data_ptr()
        result = matchedfilter._torch_batch_peak_values(
            output, 1, 16, slice(2, 14)
        )
    assert calls == [
        (
            pointer,
            np.dtype(np.int64),
            np.dtype(np.complex64),
            (16, 2, 14, 1),
        )
    ]
    expected_index = int(np.argmax(np.abs(rows[0, 2:14])))
    assert result[0].tolist() == [expected_index]
    assert result[1][0] == rows[0, 2 + expected_index]


@pytest.mark.parametrize(
    "contract",
    ("unaligned", "noncontiguous", "subclass", "requires_grad"),
)
def test_native_batch_peak_unsafe_tensor_contracts_use_helper(
    monkeypatch, contract
):
    from pycbc.filter import matchedfilter_cpu

    rows = _complex_values(3, 16, seed=6801)
    flat = torch.from_numpy(rows.reshape(-1).copy())
    with scheme.TorchScheme("cpu"):
        if contract == "unaligned":
            storage = torch.empty(flat.numel() + 1, dtype=torch.complex64)
            tensor = storage[1:]
            tensor.copy_(flat)
            assert tensor.is_contiguous()
            assert tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT != 0
        elif contract == "noncontiguous":
            storage = torch.empty(flat.numel() * 2, dtype=torch.complex64)
            tensor = storage[::2]
            tensor.copy_(flat)
            assert not tensor.is_contiguous()
        elif contract == "subclass":
            tensor = flat.as_subclass(_TensorSubclass)
        else:
            tensor = flat.requires_grad_(True)
        output = Array(TorchArrayData(tensor), copy=False)

        monkeypatch.setenv(PEAK_GATE, "0")
        expected_indices, expected_peaks = (
            matchedfilter._torch_batch_peak_values(
                output, 3, 16, slice(2, 14)
            )
        )
        _enable_native_peak(monkeypatch)

        def fail_native(*args):
            raise AssertionError("unsafe tensor entered native code")

        monkeypatch.setattr(
            matchedfilter_cpu,
            "_batch_abs_arg_max_complex64",
            fail_native,
        )
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, 3, 16, slice(2, 14)
        )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(
        peaks.view(np.uint64), expected_peaks.view(np.uint64)
    )


def test_native_batch_peak_rejects_forward_ad_and_inference(monkeypatch):
    _enable_native_peak(monkeypatch)
    rows = _complex_values(2, 16, seed=6802)
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                output._data.tensor,
                torch.ones_like(output._data.tensor),
            )
            dual_output = Array(TorchArrayData(dual), copy=False)
            assert matchedfilter._try_torch_cpu_native_batch_peak_values(
                dual_output, dual, 2, 16, slice(2, 14)
            ) is None

    with torch.inference_mode(), scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        tensor = output._data.tensor
        assert matchedfilter._try_torch_cpu_native_batch_peak_values(
            output, tensor, 2, 16, slice(2, 14)
        ) is None
        result = matchedfilter._torch_batch_peak_values(
            output, 2, 16, slice(2, 14)
        )
        assert result[0].dtype == np.int64
        assert result[1].dtype == np.complex64


@pytest.mark.parametrize(
    "drift", ("mutation", "rebind", "pid", "thread", "openmp", "failure")
)
def test_native_batch_peak_drift_and_failure_use_helper(monkeypatch, drift):
    from pycbc.filter import matchedfilter_cpu, matchedfilter_torch

    rows = _complex_values(3, 16, seed=6803)
    _enable_native_peak(monkeypatch)
    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        tensor = output._data.tensor
        monkeypatch.setenv(PEAK_GATE, "0")
        expected_indices, expected_peaks = (
            matchedfilter._torch_batch_peak_values(
                output, 3, 16, slice(2, 14)
            )
        )
        monkeypatch.setenv(PEAK_GATE, "1")

        if drift in ("mutation", "rebind"):
            real_empty = np.empty
            calls = 0

            def drift_during_setup(*args, **kwargs):
                nonlocal calls
                calls += 1
                result = real_empty(*args, **kwargs)
                if calls == 1:
                    if drift == "mutation":
                        tensor.add_(0)
                    else:
                        output._data._set_tensor(tensor.clone())
                return result

            numpy_proxy = types.SimpleNamespace(
                empty=drift_during_setup,
                int64=np.int64,
                complex64=np.complex64,
            )
            monkeypatch.setattr(matchedfilter, "numpy", numpy_proxy)
        elif drift == "pid":
            pids = iter((101, 102))
            os_proxy = types.SimpleNamespace(
                environ=os.environ,
                getpid=lambda: next(pids, 102),
            )
            monkeypatch.setattr(matchedfilter, "os", os_proxy)
        elif drift == "thread":
            identities = iter((201, 202))
            threading_proxy = types.SimpleNamespace(
                get_ident=lambda: next(identities, 202)
            )
            monkeypatch.setattr(matchedfilter, "threading", threading_proxy)
        elif drift == "openmp":
            monkeypatch.setattr(
                matchedfilter_torch,
                "_cpu_native_batch_runtime_is_stable",
                lambda runtime: False,
            )
        else:
            calls = 0

            def fail_native(*args):
                nonlocal calls
                calls += 1
                raise RuntimeError("synthetic private-kernel failure")

            monkeypatch.setattr(
                matchedfilter_cpu,
                "_batch_abs_arg_max_complex64",
                fail_native,
            )

        if drift != "failure":
            def reject_native(*args):
                raise AssertionError("invalidated native route executed")

            monkeypatch.setattr(
                matchedfilter_cpu,
                "_batch_abs_arg_max_complex64",
                reject_native,
            )
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, 3, 16, slice(2, 14)
        )
        if drift == "failure":
            assert calls == 1

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(
        peaks.view(np.uint64), expected_peaks.view(np.uint64)
    )


@pytest.fixture
def direct_batch_fftw(monkeypatch):
    from pycbc.fft import fftw, torchfft

    old_threads = torch.get_num_threads()
    old_measure = fftw.get_measure_level()
    torch.set_num_threads(1)
    fftw.set_measure_level(0)
    monkeypatch.setenv(FFTW_BATCH_GATE, "1")
    monkeypatch.setattr(
        torchfft, "_FFTW_DIRECT_PLATFORM_SUPPORTED", True
    )
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
        source_storage = zeros(
            total + int(not aligned), dtype=np.complex64
        )
        target_storage = zeros(
            total + int(not aligned), dtype=np.complex64
        )
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
    assert (
        source_tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT == 0
    ) is aligned
    assert (
        target_tensor.data_ptr() % pycbc.PYCBC_ALIGNMENT == 0
    ) is aligned
    return (
        Array(TorchArrayData(source_tensor), copy=False),
        Array(TorchArrayData(target_tensor), copy=False),
    )


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("aligned", (False, True))
def test_direct_batch_fftw_is_bitwise_legacy_exact(
    direct_batch_fftw, inverse, aligned
):
    torchfft = direct_batch_fftw
    values = _complex_values(3, 64, seed=6201 + inverse)
    replacement = _complex_values(3, 64, seed=6301 + inverse)
    expected = _legacy_batch_fft(values, inverse, aligned=aligned)
    replacement_expected = _legacy_batch_fft(
        replacement, inverse, aligned=aligned
    )
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
            monkeypatch.setattr(
                torchfft.os, "getpid", lambda: plan._pid + 1
            )
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
            monkeypatch.setattr(
                torchfft.os, "getpid", lambda: plan._pid
            )
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


def test_direct_batch_fftw_rejects_unsafe_contracts(
    direct_batch_fftw, monkeypatch
):
    torchfft = direct_batch_fftw
    values = _complex_values(2, 64, seed=6501)
    with scheme.TorchScheme("cpu"):
        source, target = _torch_batch_fft_arrays(values)
        source._data._set_tensor(
            source._data.tensor.as_subclass(_TensorSubclass)
        )
        assert (
            torchfft.IFFT(source, target, nbatch=2, size=64)._fftw_batch_plan
            is None
        )

        source, target = _torch_batch_fft_arrays(values)
        storage = torch.empty(256, dtype=torch.complex64)
        noncontiguous = storage[::2]
        noncontiguous.copy_(source._data.tensor)
        source._data._set_tensor(noncontiguous)
        assert not source._data.tensor.is_contiguous()
        assert (
            torchfft.IFFT(source, target, nbatch=2, size=64)._fftw_batch_plan
            is None
        )

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


def test_direct_batch_fftw_setup_failure_falls_back(
    direct_batch_fftw, monkeypatch
):
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
