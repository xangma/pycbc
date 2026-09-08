# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Preserve chi-square arithmetic while scheduling scalar CUDA transfers."""

from contextlib import nullcontext

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("norm_type", (float, int, np.float32, np.float64))
@pytest.mark.parametrize("custom_stream", (False, True))
def test_cuda_scalar_norm_preserves_cast_then_square(
    monkeypatch, norm_type, custom_stream
):
    norm = norm_type(2 if norm_type is int else 0.117311)
    _check_normalization(monkeypatch, norm, "ordinary",
                         custom_stream, "cuda")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "special", ("mixed_precision", "tensor_norm", "corr_grad", "snr_grad",
                "norm_grad", "corr_subclass", "snr_subclass", "inference",
                "points_subclass", "points_forward_ad", "forward_ad")
)
def test_special_inputs_retain_late_normalization(
    monkeypatch, device, special
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _check_normalization(monkeypatch, 0.117311, special, False, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("special", ("corr_grad", "forward_ad"))
def test_native_cuda_corr_autodiff_matches_existing_formula(
    monkeypatch, special
):
    _check_normalization(monkeypatch, 0.117311, "native_" + special,
                         False, "cuda")


def _check_normalization(monkeypatch, norm, special, custom_stream, device):
    from pycbc.vetoes import chisq_torch

    native_corr_ad = special.startswith("native_")
    special = special.removeprefix("native_")
    if special in ("corr_grad", "forward_ad") and not native_corr_ad:
        # The existing Triton bin kernel has no correlation AD rule. Exercise
        # supported Torch derivatives here and compare native behavior above.
        monkeypatch.setattr(chisq_torch, "_HAS_TRITON", False)

    class DispatchTensor(torch.Tensor):
        pass

    # Exercise the Torch formula on CPU too; native CPU fusion is unchanged.
    monkeypatch.setattr(chisq_torch, "_cpu_native_eligible", lambda *a: False)
    monkeypatch.setattr(
        chisq_torch, "_cpu_native_sparse_search_eligible", lambda *a: False
    )
    stream_context = (
        torch.cuda.stream(torch.cuda.Stream())
        if custom_stream else nullcontext()
    )
    ad_context = (
        torch.autograd.forward_ad.dual_level()
        if special in ("forward_ad", "points_forward_ad") else nullcontext()
    )
    inference_context = (
        torch.inference_mode() if special == "inference" else nullcontext()
    )
    with (stream_context, ad_context, inference_context,
          scheme.TorchScheme(device)):
        generator = np.random.default_rng(731)
        values = (generator.normal(size=64)
                  + 1j * generator.normal(size=64)).astype(np.complex64)
        corr = FrequencySeries(values, delta_f=0.125)
        snr_dtype = (torch.complex128 if special == "mixed_precision"
                     else torch.complex64)
        snr_t = torch.tensor([1.25 - 0.75j, -0.25 + 0.5j], device=device,
                             dtype=snr_dtype)
        if special == "corr_grad":
            corr._data.tensor.requires_grad_(True)
        if special == "snr_grad":
            snr_t.requires_grad_(True)
        if special == "corr_subclass":
            corr._data.tensor = corr._data.tensor.as_subclass(DispatchTensor)
        if special == "snr_subclass":
            snr_t = snr_t.as_subclass(DispatchTensor)
        if special in ("tensor_norm", "norm_grad"):
            norm = torch.tensor(norm, dtype=torch.float64, device=device,
                                requires_grad=special == "norm_grad")
        if special == "forward_ad":
            corr._data.tensor = torch.autograd.forward_ad.make_dual(
                corr._data.tensor, torch.ones_like(corr._data.tensor)
            )
        # A PyCBC Array preserves mixed SNR precision in the existing API.
        snr = Array(TorchArrayData(snr_t), copy=False)
        bins = (1, 13, 31, 63)
        points = np.array([3, 17], dtype=np.int64)
        if special in ("points_subclass", "points_forward_ad"):
            points = torch.as_tensor(
                points, device=device, dtype=torch.float64
            )
            if special == "points_subclass":
                points = points.as_subclass(DispatchTensor)
            else:
                points = torch.autograd.forward_ad.make_dual(
                    points, torch.ones_like(points)
                )
        pts = chisq_torch._point_tensor(corr._data.tensor, points)
        expected = chisq_torch.shift_sum(corr, pts, bins)._data.tensor
        expected = expected * 3 - (torch.conj(snr_t) * snr_t).real
        expected = expected * torch.as_tensor(
            norm, device=device, dtype=expected.dtype
        )**2

        calls = []
        original_as_tensor = torch.as_tensor
        original_shift_sum = chisq_torch.shift_sum

        def as_tensor(value, *args, **kwargs):
            if value is norm:
                calls.append("norm")
            return original_as_tensor(value, *args, **kwargs)

        def shift_sum(*args, **kwargs):
            calls.append("bins")
            return original_shift_sum(*args, **kwargs)

        monkeypatch.setattr(torch, "as_tensor", as_tensor)
        monkeypatch.setattr(chisq_torch, "shift_sum", shift_sum)
        actual = chisq_torch.power_chisq_at_points_from_precomputed(
            corr, snr, norm, bins, points
        )._data.tensor
        expected_order = (["norm", "bins"] if special == "ordinary"
                          else ["bins", "norm"])
        assert calls == expected_order
        assert actual.dtype == expected.dtype
        assert actual.requires_grad == expected.requires_grad
        if special == "corr_grad" and not native_corr_ad:
            assert actual.requires_grad
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if special.endswith("grad") and actual.requires_grad:
            variable = {"corr_grad": corr._data.tensor, "snr_grad": snr_t,
                        "norm_grad": norm}[special]
            actual_grad = torch.autograd.grad(actual.sum(), variable)[0]
            expected_grad = torch.autograd.grad(expected.sum(), variable)[0]
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=0, atol=0
            )
        if special in ("forward_ad", "points_forward_ad"):
            actual_dual = torch.autograd.forward_ad.unpack_dual(actual)
            expected_dual = torch.autograd.forward_ad.unpack_dual(expected)
            if native_corr_ad and expected_dual.tangent is None:
                assert actual_dual.tangent is None
            else:
                assert actual_dual.tangent is not None
                torch.testing.assert_close(
                    actual_dual.tangent, expected_dual.tangent, rtol=0, atol=0
                )
        if device == "cuda":
            torch.cuda.current_stream().synchronize()
