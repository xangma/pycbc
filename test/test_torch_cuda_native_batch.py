# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch filtering regression tests."""


import numpy as np
import pytest
from pycbc import scheme
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.types import Array, zeros
from pycbc.types.array_torch import TorchArrayData
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


if not torch.cuda.is_available():
    pytest.skip("CUDA device is required", allow_module_level=True)


CORRELATION_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(np.complex64)


def _make_batch(rows=3, size=128, total=None):
    if total is None:
        total = size
    x_values = _complex_values(rows, total, seed=7101)
    y_values = _complex_values(1, total, seed=7102)[0]

    # Allocate contiguous memory on CUDA to ensure uniform striding for zero-copy views
    x_mem = zeros(rows * total, dtype=np.complex64)
    x_mem._data.tensor.copy_(torch.from_numpy(x_values.reshape(-1)))
    xs = [x_mem[i * total : (i + 1) * total] for i in range(rows)]

    y = Array(y_values)

    z_mem = zeros(rows * total, dtype=np.complex64)
    z_mem._data.tensor.fill_(19 - 7j)
    zs = [z_mem[i * total : (i + 1) * total] for i in range(rows)]

    return BatchCorrelator(xs, zs, size), y


def _torch_correlation(batch, y):
    return [
        torch.conj(x._data.tensor[: batch.size])
        * y._data.tensor[: batch.size]
        for x in batch.xs
    ]


def _enable_native_correlation(monkeypatch):
    monkeypatch.setenv(CORRELATION_GATE, "1")


def test_cuda_batch_correlation_gate_is_strict_and_default_off(monkeypatch):
    monkeypatch.delenv(CORRELATION_GATE, raising=False)
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch()
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cuda_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

    monkeypatch.setenv(CORRELATION_GATE, "sometimes")
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch()
        with pytest.raises(ValueError, match=CORRELATION_GATE):
            batch.execute(y)


def test_cuda_native_batch_correlation_is_zero_copy_exact_and_reusable(monkeypatch):
    _enable_native_correlation(monkeypatch)
    size, total = 129, 193
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch(rows=4, size=size, total=total)
        expected = _torch_correlation(batch, y)
        versions = [z._data.tensor._version for z in batch.zs]
        tails = [z._data.tensor[size:].clone() for z in batch.zs]
        batch.execute(y)

        state = getattr(batch, "_torch_cuda_native_batch_state", None)
        assert state is not None
        assert state.can_execute(batch)
        assert state._z_pointers == tuple(
            z._data.tensor.data_ptr() for z in batch.zs
        )
        assert state._packed_x.data_ptr() == batch.xs[0]._data.tensor.data_ptr()
        assert state._packed_z.data_ptr() == batch.zs[0]._data.tensor.data_ptr()
        assert state._packed_x.shape == (4, size)
        assert state._packed_z.shape == (4, size)

        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            assert torch.equal(output._data.tensor[:size], truth)
            assert torch.equal(output._data.tensor[size:], tails[index])
            assert output._data.tensor._version == versions[index] + 1

        # In-place content mutation is the reusable contract
        batch.xs[0]._data.tensor.mul_(2 - 0.5j)
        expected = _torch_correlation(batch, y)
        versions = [z._data.tensor._version for z in batch.zs]
        batch.execute(y)
        assert batch._torch_cuda_native_batch_state is state
        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            assert torch.equal(output._data.tensor[:size], truth)
            assert torch.equal(output._data.tensor[size:], tails[index])
            assert output._data.tensor._version == versions[index] + 1


def test_cuda_native_batch_correlation_refreshes_nonuniform_inputs(monkeypatch):
    from pycbc.filter import matchedfilter_torch

    _enable_native_correlation(monkeypatch)
    rows, size = 3, 64
    with scheme.TorchScheme("cuda"):
        x_values = _complex_values(rows, size, seed=7141)
        x_mem = zeros(4 * size, dtype=np.complex64)
        offsets = (0, size, 3 * size)
        xs = []
        for offset, values in zip(offsets, x_values):
            x = x_mem[offset:offset + size]
            x._data.tensor.copy_(torch.from_numpy(values))
            xs.append(x)
        y = Array(_complex_values(1, size, seed=7142)[0])
        z_mem = zeros(rows * size, dtype=np.complex64)
        zs = [z_mem[i * size:(i + 1) * size] for i in range(rows)]
        batch = BatchCorrelator(xs, zs, size)

        assert matchedfilter_torch._find_uniform_stride(
            tuple(x._data.tensor for x in xs), size
        ) is None
        batch.execute(y)
        state = batch._torch_cuda_native_batch_state
        assert state._is_stacked_x

        batch.xs[1]._data.tensor.mul_(2 - 0.5j)
        expected = _torch_correlation(batch, y)
        batch.execute(y)

        assert batch._torch_cuda_native_batch_state is state
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


@pytest.mark.parametrize(
    "drift",
    ("x_rebind", "z_rebind", "pid", "thread"),
)
def test_cuda_native_batch_correlation_drift_fails_closed(monkeypatch, drift):
    from pycbc.filter import matchedfilter_torch

    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch()
        batch.execute(y)
        state = batch._torch_cuda_native_batch_state

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

        expected = _torch_correlation(batch, y)
        batch.execute(y)
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor[: batch.size], truth)


def test_cuda_native_batch_correlation_accepts_dynamic_y_and_recovers(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y1 = _make_batch(rows=3, size=64)
        for y in (
            y1,
            Array(_complex_values(1, 64, seed=7151)[0]),
        ):
            expected = _torch_correlation(batch, y)
            versions = [z._data.tensor._version for z in batch.zs]
            batch.execute(y)
            for index, (output, truth) in enumerate(zip(batch.zs, expected)):
                assert torch.equal(output._data.tensor, truth)
                assert output._data.tensor._version == versions[index] + 1

        state = batch._torch_cuda_native_batch_state

        # Non-contiguous y tensor falls back to torch loop
        storage = torch.empty(128, dtype=torch.complex64, device="cuda")
        unsafe_tensor = storage[::2]
        unsafe_tensor.copy_(
            torch.from_numpy(_complex_values(1, 64, seed=7152)[0])
        )
        assert not unsafe_tensor.is_contiguous()
        unsafe_y = Array(TorchArrayData(unsafe_tensor), copy=False)
        expected = _torch_correlation(batch, unsafe_y)
        batch.execute(unsafe_y)
        assert batch._torch_cuda_native_batch_state is state
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)

        # Subsequent valid y recovers native execution
        y3 = Array(_complex_values(1, 64, seed=7153)[0])
        expected = _torch_correlation(batch, y3)
        versions = [z._data.tensor._version for z in batch.zs]
        batch.execute(y3)
        for index, (output, truth) in enumerate(zip(batch.zs, expected)):
            assert torch.equal(output._data.tensor, truth)
            assert output._data.tensor._version == versions[index] + 1


def test_cuda_native_batch_correlation_dynamic_ad_y_uses_torch(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y1 = _make_batch(rows=2, size=64)
        batch.execute(y1)
        assert hasattr(batch, "_torch_cuda_native_batch_state")

        ad_y = Array(_complex_values(1, 64, seed=7161)[0])
        ad_y._data.tensor.requires_grad_(True)
        with pytest.raises(RuntimeError):
            batch.execute(ad_y)

        y3 = Array(_complex_values(1, 64, seed=7162)[0])
        expected = _torch_correlation(batch, y3)
        batch.execute(y3)
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_cuda_native_batch_correlation_does_not_change_single_batch(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch(rows=1, size=64)
        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cuda_native_batch_state")
        assert torch.equal(batch.zs[0]._data.tensor, expected[0])


class _TensorSubclass(torch.Tensor):
    pass


@pytest.mark.parametrize("contract", ("noncontiguous", "subclass", "non_uniform_stride"))
def test_cuda_native_batch_correlation_rejects_unsafe_tensor_contracts(
    monkeypatch, contract
):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        if contract == "non_uniform_stride":
            # Separate disjoint allocations have non-uniform strides
            xs = [Array(_complex_values(1, 64, seed=7170 + i)[0]) for i in range(3)]
            zs = [Array(np.full(64, 19 - 7j, dtype=np.complex64)) for _ in range(3)]
            # Force non-uniform addresses
            t0 = torch.empty(64, dtype=torch.complex64, device="cuda")
            torch.empty(13, dtype=torch.float32, device="cuda")
            t1 = torch.empty(64, dtype=torch.complex64, device="cuda")
            torch.empty(27, dtype=torch.float32, device="cuda")
            t2 = torch.empty(64, dtype=torch.complex64, device="cuda")
            xs[0]._data._set_tensor(t0)
            xs[1]._data._set_tensor(t1)
            xs[2]._data._set_tensor(t2)
            batch = BatchCorrelator(xs, zs, 64)
            y = Array(_complex_values(1, 64, seed=7175)[0])
        else:
            batch, y = _make_batch(size=64)
            if contract == "noncontiguous":
                storage = torch.empty(128, dtype=torch.complex64, device="cuda")
                replacement = storage[::2]
                replacement.copy_(batch.xs[0]._data.tensor)
                assert not replacement.is_contiguous()
            else:
                replacement = batch.xs[0]._data.tensor.as_subclass(_TensorSubclass)
            batch.xs[0]._data._set_tensor(replacement)

        expected = _torch_correlation(batch, y)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cuda_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)


def test_cuda_native_batch_correlation_rejects_output_alias(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch(size=64)
        original = batch.xs[0]._data.tensor.clone()
        expected = torch.conj(original) * y._data.tensor
        batch.zs[0]._data._set_tensor(batch.xs[0]._data.tensor)
        batch.execute(y)
        assert not hasattr(batch, "_torch_cuda_native_batch_state")
        assert torch.equal(batch.zs[0]._data.tensor, expected)


def test_cuda_native_batch_correlation_stream_safety(monkeypatch):
    _enable_native_correlation(monkeypatch)
    with scheme.TorchScheme("cuda"):
        batch, y = _make_batch(rows=4, size=128)
        expected = _torch_correlation(batch, y)

        custom_stream = torch.cuda.Stream()
        with torch.cuda.stream(custom_stream):
            batch.execute(y)
        custom_stream.synchronize()

        assert hasattr(batch, "_torch_cuda_native_batch_state")
        for output, truth in zip(batch.zs, expected):
            assert torch.equal(output._data.tensor, truth)
