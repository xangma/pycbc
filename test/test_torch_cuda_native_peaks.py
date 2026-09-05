# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch live-search regression tests."""


import numpy as np
import pytest
from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import Array
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


if not torch.cuda.is_available():
    pytest.skip("CUDA device is required", allow_module_level=True)


PEAK_GATE = "PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(np.complex64)


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
        magmax = np.float32(0.0)
        best = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for i, val in enumerate(values):
                # The compiled complex64 kernels evaluate these operations in
                # float before assigning the result to their double variable.
                mag = np.float32(
                    np.float32(val.real) * np.float32(val.real)
                    + np.float32(val.imag) * np.float32(val.imag)
                )
                if mag > magmax:
                    magmax = mag
                    best = i
        indices[row_index] = best
        peaks[row_index] = values[best]
    return indices, peaks


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
