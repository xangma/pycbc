# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Tests for direct sparse-index dispatch in Torch CPU pointwise chi-squared."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def _inputs(point_count):
    rng = np.random.default_rng(202608250 + point_count)
    size = 4096
    correlation = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(np.complex64)
    points = np.linspace(101, 3901, point_count, dtype=np.int64)
    snr = torch.from_numpy(
        (
            rng.normal(size=point_count)
            + 1j * rng.normal(size=point_count)
        ).astype(np.complex64)
    )
    return correlation, points, snr


@pytest.mark.parametrize("point_count", (2, 4, 8))
def test_sparse_numpy_indices_are_consumed_directly_and_bitwise(
    point_count, monkeypatch
):
    from pycbc.vetoes import chisq_torch

    values, points, snr = _inputs(point_count)
    bins = (29, 211, 619, 1481, 3073)
    norm = 0.117
    observed_points = []

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        corr_tensor = correlation._data.tensor
        expected = chisq_torch._cpu_native_point_chisq(
            corr_tensor,
            torch.as_tensor(points, dtype=torch.float64),
            bins,
            snr=snr,
            snr_norm=norm,
        )
        corr_before = corr_tensor.clone()
        snr_before = snr.clone()
        points_before = points.copy()
        corr_version = corr_tensor._version
        snr_version = snr._version
        original_native = chisq_torch._cpu_native_point_chisq

        def record_native(corr, pts, edges, snr=None, snr_norm=None):
            observed_points.append(pts)
            return original_native(corr, pts, edges, snr, snr_norm)

        def fail_point_tensor(*args, **kwargs):
            raise AssertionError("eligible NumPy indices were copied to Torch")

        monkeypatch.setattr(
            chisq_torch, "_cpu_native_point_chisq", record_native
        )
        monkeypatch.setattr(chisq_torch, "_point_tensor", fail_point_tensor)
        actual = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr),
            norm,
            bins,
            points,
        )

    assert len(observed_points) == 1
    assert observed_points[0] is points
    assert isinstance(actual._data, TorchArrayData)
    assert actual._data.tensor.device.type == "cpu"
    assert actual._data.tensor.dtype == torch.float32
    assert torch.equal(actual._data.tensor, expected)
    assert torch.equal(corr_tensor, corr_before)
    assert torch.equal(snr, snr_before)
    assert np.array_equal(points, points_before)
    assert corr_tensor._version == corr_version
    assert snr._version == snr_version


@pytest.mark.parametrize("kind", ("int32", "strided"))
def test_unsupported_numpy_indices_retain_generic_native_fallback(
    kind, monkeypatch
):
    from pycbc.vetoes import chisq_torch

    values, contiguous_points, snr = _inputs(4)
    if kind == "int32":
        points = contiguous_points.astype(np.int32)
    else:
        storage = np.empty(8, dtype=np.int64)
        storage[::2] = contiguous_points
        points = storage[::2]
        assert not points.flags.c_contiguous
    bins = (29, 211, 619, 1481, 3073)
    observed_points = []
    original_native = chisq_torch._cpu_native_point_chisq

    def record_native(corr, pts, edges, snr=None, snr_norm=None):
        if snr is not None:
            observed_points.append(pts)
        return original_native(corr, pts, edges, snr, snr_norm)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        monkeypatch.setattr(
            chisq_torch, "_cpu_native_point_chisq", record_native
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr),
            0.117,
            bins,
            points,
        )

    assert len(observed_points) == 1
    assert isinstance(observed_points[0], torch.Tensor)
    assert observed_points[0].dtype == torch.float64
    assert observed_points[0].is_contiguous()
    assert result._data.tensor.shape == (4,)
    assert torch.isfinite(result._data.tensor).all()


def test_sparse_numpy_fast_path_preserves_multi_point_autograd(monkeypatch):
    from pycbc.vetoes import chisq_torch

    values, points, _ = _inputs(4)
    bins = (29, 211, 619, 1481, 3073)

    def fail_native(*args, **kwargs):
        raise AssertionError("autograd input entered a native CPU kernel")

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        corr_tensor = correlation._data.tensor.detach().requires_grad_(True)
        correlation._data.tensor = corr_tensor
        snr = torch.tensor(
            [1.0 + 0.5j] * points.size,
            dtype=torch.complex64,
            requires_grad=True,
        )
        monkeypatch.setattr(
            chisq_torch, "_cpu_native_point_chisq", fail_native
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr),
            0.117,
            bins,
            points,
        )
        corr_grad, snr_grad = torch.autograd.grad(
            result._data.tensor.sum(), (corr_tensor, snr)
        )

    assert torch.isfinite(torch.view_as_real(corr_grad)).all()
    assert torch.isfinite(torch.view_as_real(snr_grad)).all()
    assert torch.count_nonzero(corr_grad[bins[0]:bins[-1]]) > 0
    assert torch.count_nonzero(snr_grad) > 0


def test_sparse_numpy_empty_result_skips_native_dispatch(monkeypatch):
    from pycbc.vetoes import chisq_torch

    def fail_native(*args, **kwargs):
        raise AssertionError("empty indices entered native eligibility")

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(4096, dtype=np.complex64), delta_f=0.125
        )
        monkeypatch.setattr(
            chisq_torch, "_cpu_native_sparse_search_eligible", fail_native
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(torch.empty(0, dtype=torch.complex64)),
            0.117,
            (29, 211, 619, 1481, 3073),
            np.empty(0, dtype=np.int64),
        )

    assert isinstance(result._data, TorchArrayData)
    assert result._data.tensor.device.type == "cpu"
    assert result._data.tensor.dtype == torch.float32
    assert result._data.tensor.numel() == 0
