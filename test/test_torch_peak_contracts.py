# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch live-search regression tests."""


import numpy as np
import pytest
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


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
