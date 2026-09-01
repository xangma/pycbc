# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Qualification tests for the opt-in Torch-CUDA native batch routes."""

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

if not torch.cuda.is_available():
    pytest.skip("CUDA device is required for CUDA native batch tests", allow_module_level=True)


CORRELATION_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE"
PEAK_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"
PROMOTED_ROWS_GATE = "PYCBC_TORCH_CUDA_PROMOTED_ROWS"


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


def _enable_native_peak(monkeypatch):
    monkeypatch.setenv(PEAK_GATE, "1")


def _standard_peak_values_reference(rows, segment):
    """Reference legacy complex abs-arg-max implementation."""
    start, stop, step = segment.indices(rows.shape[1])
    assert step == 1 and start < stop
    indices = np.empty(len(rows), dtype=np.int64)
    peaks = np.empty(len(rows), dtype=np.complex64)
    for row_index, row in enumerate(rows):
        values = row[start:stop]
        magmax = 0.0
        best = 0
        for i, val in enumerate(values):
            # double precision squared magnitude matching legacy scan semantics
            mag = float(val.real) * float(val.real) + float(val.imag) * float(val.imag)
            if mag > magmax:
                magmax = mag
                best = i
        indices[row_index] = best
        peaks[row_index] = values[best]
    return indices, peaks


# =============================================================================
# Correlation Gate & Execution Tests
# =============================================================================

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
            _pad = torch.empty(13, dtype=torch.float32, device="cuda")
            t1 = torch.empty(64, dtype=torch.complex64, device="cuda")
            _pad2 = torch.empty(27, dtype=torch.float32, device="cuda")
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


# =============================================================================
# Peak Extraction Tests
# =============================================================================

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


def test_cuda_native_batch_peak_is_exact_legacy_semantics(monkeypatch):
    _enable_native_peak(monkeypatch)
    rows = _peak_test_rows()
    segment = slice(2, 10)
    expected_indices, expected_peaks = _standard_peak_values_reference(rows, segment)

    with scheme.TorchScheme("cuda"):
        output = Array(rows.reshape(-1))
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, len(rows), rows.shape[1], segment
        )

        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(
            peaks.view(np.uint64), expected_peaks.view(np.uint64)
        )


def test_cuda_native_batch_peak_gate_is_strict_default_off_and_corrects_nan(monkeypatch):
    rows = np.zeros((2, 8), dtype=np.complex64)
    rows[0, 2:6] = [1, np.nan, 3, 2]
    rows[1, 2:6] = [2, -2, 1, 0]
    segment = slice(2, 6)

    with scheme.TorchScheme("cuda"):
        output = Array(rows.reshape(-1))
        monkeypatch.delenv(PEAK_GATE, raising=False)

        # Default off returns the first NaN index
        helper_indices, helper_peaks = (
            matchedfilter._torch_batch_peak_values(output, 2, 8, segment)
        )
        assert helper_indices[0] == 1
        assert np.isnan(helper_peaks[0])

        monkeypatch.setenv(PEAK_GATE, "sometimes")
        with pytest.raises(ValueError, match=PEAK_GATE):
            matchedfilter._torch_batch_peak_values(output, 2, 8, segment)

    _enable_native_peak(monkeypatch)
    expected_indices, expected_peaks = _standard_peak_values_reference(rows, segment)
    with scheme.TorchScheme("cuda"):
        output = Array(rows.reshape(-1))
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, 2, 8, segment
        )
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(
        peaks.view(np.uint64), expected_peaks.view(np.uint64)
    )
    assert indices[0] == 2  # value 3 wins over NaN at index 1


def test_cuda_native_batch_peak_does_not_change_single_batch(monkeypatch):
    _enable_native_peak(monkeypatch)
    rows = _complex_values(1, 16, seed=7751)
    with scheme.TorchScheme("cuda"):
        output = Array(rows.reshape(-1))
        result = matchedfilter._torch_batch_peak_values(
            output, 1, 16, slice(2, 14)
        )
    expected_index = int(np.argmax(np.abs(rows[0, 2:14])))
    assert result[0].tolist() == [expected_index]
    assert result[1][0] == rows[0, 2 + expected_index]


def test_cuda_native_batch_peak_stream_safety(monkeypatch):
    _enable_native_peak(monkeypatch)
    rows = _peak_test_rows()
    segment = slice(2, 10)
    expected_indices, expected_peaks = _standard_peak_values_reference(rows, segment)

    with scheme.TorchScheme("cuda"):
        output = Array(rows.reshape(-1))
        custom_stream = torch.cuda.Stream()
        with torch.cuda.stream(custom_stream):
            indices, peaks = matchedfilter._torch_batch_peak_values(
                output, len(rows), rows.shape[1], segment
            )
        custom_stream.synchronize()

        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(
            peaks.view(np.uint64), expected_peaks.view(np.uint64)
        )


def test_standard_peak_tensor_direct():
    from pycbc.filter.matchedfilter_torch import standard_peak_tensor

    rows = torch.tensor(_peak_test_rows(), dtype=torch.complex64, device="cuda")
    segment_rows = rows[:, 2:10]
    expected_indices, expected_peaks = _standard_peak_values_reference(
        _peak_test_rows(), slice(2, 10)
    )

    indices, peaks = standard_peak_tensor(segment_rows)
    assert indices.device.type == "cuda"
    assert peaks.device.type == "cuda"
    np.testing.assert_array_equal(indices.cpu().numpy(), expected_indices)
    np.testing.assert_array_equal(
        peaks.cpu().numpy().view(np.uint64),
        expected_peaks.view(np.uint64),
    )


# =============================================================================
# Promoted Rows Workspace Tests (B=32)
# =============================================================================

def test_cuda_promoted_rows_gate_is_strict_and_default_off(monkeypatch):
    from pycbc.fft import torchfft

    size = 131072
    batch = 32
    with scheme.TorchScheme("cuda"):
        invec = zeros(batch * size, dtype=np.complex64)
        outvec = zeros(batch * size, dtype=np.complex64)

        monkeypatch.delenv(PROMOTED_ROWS_GATE, raising=False)
        fftobj = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert fftobj._promoted_batch_plan is not None
        assert fftobj._promoted_batch_plan.rows == 16

        monkeypatch.setenv(PROMOTED_ROWS_GATE, "sometimes")
        with pytest.raises(ValueError, match=PROMOTED_ROWS_GATE):
            torchfft.IFFT(invec, outvec, nbatch=batch, size=size)

        monkeypatch.setenv(PROMOTED_ROWS_GATE, "1")
        fftobj32 = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert fftobj32._promoted_batch_plan is not None
        assert fftobj32._promoted_batch_plan.rows == 32


def test_cuda_promoted_rows_fft_execution_and_accuracy(monkeypatch):
    from pycbc.fft import torchfft

    size = 131072
    batch = 32
    monkeypatch.setenv(PROMOTED_ROWS_GATE, "1")

    with scheme.TorchScheme("cuda"):
        values = _complex_values(batch, size, seed=7901)
        invec = zeros(batch * size, dtype=np.complex64)
        outvec = zeros(batch * size, dtype=np.complex64)
        invec._data.tensor.copy_(torch.from_numpy(values.reshape(-1)))

        ifftobj = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert ifftobj._promoted_batch_plan.rows == 32
        ifftobj.execute()

        # Compare with complex128 torch.fft.ifft reference
        torch_ref = torch.fft.ifft(
            torch.from_numpy(values).to(torch.complex128),
            n=size,
            dim=-1,
            norm="forward",
        ).to(torch.complex64).numpy()

        actual = outvec.numpy().reshape(batch, size)
        np.testing.assert_allclose(actual, torch_ref, rtol=1e-5, atol=1e-5)
