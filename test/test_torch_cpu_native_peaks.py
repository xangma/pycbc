# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Qualification tests for optional native CPU peak extraction."""

import os
import types

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import Array
from pycbc.types.array_torch import TorchArrayData

torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)

PEAK_GATE = "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, size))
        + 1j * rng.normal(size=(rows, size))
    ).astype(np.complex64)


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



class _TensorSubclass(torch.Tensor):
    pass



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
