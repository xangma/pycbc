# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Precision tests for the sparse Torch CPU pointwise chi-squared path."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def _direct_bin_sums(corr, points, bins):
    """Evaluate the pre-optimization direct-exponential bin sums."""
    point_dtype = (
        torch.float32 if corr.device.type == "mps" else torch.float64
    )
    pts = torch.as_tensor(points, device=corr.device, dtype=point_dtype)
    output = torch.zeros(
        (pts.numel(), len(bins) - 1), device=corr.device, dtype=corr.dtype
    )
    for column, (start, end) in enumerate(zip(bins, bins[1:])):
        frequencies = torch.arange(
            start, end, device=corr.device, dtype=point_dtype
        )
        phases = torch.exp(
            2j
            * torch.pi
            * pts[:, None]
            * frequencies[None, :]
            / float(corr.shape[-1])
        ).to(corr.dtype)
        output[:, column] = torch.sum(
            corr[start:end] * phases, dim=1
        )
    return output


def _direct_shift_sum(corr, points, bins):
    """Evaluate the pre-optimization direct-exponential formulation."""
    output = _direct_bin_sums(corr, points, bins)
    return torch.sum(torch.conj(output) * output, dim=1).real


def test_forward_ad_guard_skips_unpack_outside_dual_level(monkeypatch):
    from pycbc.vetoes import chisq_torch

    def fail_unpack(*args, **kwargs):
        raise AssertionError("ordinary tensor reached forward-AD unpack")

    monkeypatch.setattr(torch.autograd.forward_ad, "unpack_dual", fail_unpack)
    assert not chisq_torch._has_forward_ad_state(torch.ones(1))


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("point_count", (1, 4, 32))
def test_cpu_shift_sum_recurrence_matches_direct_phases(dtype, point_count):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(7341 + point_count)
    size = 4096
    values = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(dtype)
    points = rng.integers(0, size, size=point_count, dtype=np.int64)
    bins = (29, 211, 619, 1481, 3073)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        expected = _direct_shift_sum(
            correlation._data.tensor, points, bins
        )
        actual = chisq_torch.shift_sum(correlation, Array(points), bins)

    assert isinstance(actual._data, TorchArrayData)
    assert actual._data.tensor.device.type == "cpu"
    assert actual._data.tensor.dtype == expected.dtype
    if dtype == np.complex128:
        assert torch.equal(actual._data.tensor, expected)
    else:
        torch.testing.assert_close(
            actual._data.tensor, expected, rtol=3e-6, atol=3e-6
        )


def test_cpu_shift_sum_uses_high_precision_recurrence(monkeypatch):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(8341)
    values = (
        rng.normal(size=2048) + 1j * rng.normal(size=2048)
    ).astype(np.complex64)
    bins = (17, 193, 541, 1201)
    observed_dtypes = []
    original_cumprod = torch.cumprod

    def record_cumprod(input_tensor, dim, *, out=None, dtype=None):
        observed_dtypes.append(input_tensor.dtype)
        return original_cumprod(
            input_tensor, dim, out=out, dtype=dtype
        )

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        correlation._data.tensor.requires_grad_(True)
        monkeypatch.setattr(chisq_torch.torch, "cumprod", record_cumprod)
        result = chisq_torch.shift_sum(
            correlation, np.array([101, 701], dtype=np.int64), bins
        )

    assert result._data.tensor.dtype == torch.float32
    assert observed_dtypes == [torch.complex128] * (len(bins) - 1)


def test_cpu_single_point_shift_sum_batches_bin_recurrences(monkeypatch):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(9341)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    bins = (29, 211, 619, 1481, 3073)
    observed_dtypes = []
    observed_shapes = []
    original_cumprod = torch.cumprod

    def record_cumprod(input_tensor, dim, *, out=None, dtype=None):
        observed_dtypes.append(input_tensor.dtype)
        observed_shapes.append(tuple(input_tensor.shape))
        return original_cumprod(
            input_tensor, dim, out=out, dtype=dtype
        )

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        correlation._data.tensor.requires_grad_(True)
        expected = _direct_shift_sum(
            correlation._data.tensor, np.array([701]), bins
        )
        monkeypatch.setattr(chisq_torch.torch, "cumprod", record_cumprod)
        result = chisq_torch.shift_sum(
            correlation, np.array([701], dtype=np.int64), bins
        )

    torch.testing.assert_close(
        result._data.tensor, expected, rtol=3e-6, atol=3e-6
    )
    assert observed_dtypes == [torch.complex128]
    assert observed_shapes == [(1, bins[-1] - bins[0])]


@pytest.mark.parametrize(
    "bins",
    (
        (29, 211, 619, 1481, 3073),
        (29, 211, 619, 619, 3073),
        (29, 29, 29),
    ),
)
def test_accelerator_single_point_phase_reuse_matches_direct_bins(bins):
    from pycbc.vetoes import chisq_torch

    generator = torch.Generator().manual_seed(9541)
    correlation = torch.randn(
        4096, dtype=torch.complex128, generator=generator
    )
    points = torch.tensor([701.0], dtype=torch.float64)

    expected = _direct_bin_sums(correlation, points, bins)
    actual = chisq_torch._accelerator_single_point_bin_sums(
        correlation, points, bins
    )

    assert torch.equal(actual, expected)


def test_accelerator_single_point_phase_reuse_launches_one_phase(monkeypatch):
    from pycbc.vetoes import chisq_torch

    correlation = torch.ones(4096, dtype=torch.complex128)
    points = torch.tensor([701.0], dtype=torch.float64)
    bins = (29, 211, 619, 1481, 3073)
    calls = {"arange": 0, "exp": 0}
    original_arange = torch.arange
    original_exp = torch.exp

    def record_arange(*args, **kwargs):
        calls["arange"] += 1
        return original_arange(*args, **kwargs)

    def record_exp(*args, **kwargs):
        calls["exp"] += 1
        return original_exp(*args, **kwargs)

    monkeypatch.setattr(chisq_torch.torch, "arange", record_arange)
    monkeypatch.setattr(chisq_torch.torch, "exp", record_exp)
    chisq_torch._accelerator_single_point_bin_sums(
        correlation, points, bins
    )

    assert calls == {"arange": 1, "exp": 1}


def test_accelerator_phase_reuse_eligibility_preserves_legacy_cases(
    monkeypatch,
):
    from types import SimpleNamespace

    from pycbc.vetoes import chisq_torch

    correlation = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), shape=(4096,)
    )
    points = torch.tensor([701.0], dtype=torch.float64)
    bins = (29, 211, 619, 1481, 3073)
    eligible = chisq_torch._accelerator_single_point_phase_reuse_eligible

    assert eligible(correlation, points, bins)
    assert not eligible(correlation, points.repeat(2), bins)
    assert not eligible(correlation, points.clone().requires_grad_(), bins)
    assert not eligible(correlation, points, (29, 211, 101, 3073))
    assert not eligible(correlation, points, (-1, 211, 3073))
    assert not eligible(correlation, points, (29, 4097))
    assert not eligible(correlation, points, (29,))
    assert not eligible(correlation, points, ())

    class PointTensor(torch.Tensor):
        pass

    assert not eligible(correlation, points.as_subclass(PointTensor), bins)

    monkeypatch.setattr(
        chisq_torch, "_has_forward_ad_state", lambda tensor: True
    )
    assert not eligible(correlation, points, bins)


def test_shift_sum_dispatches_single_accelerator_point_to_phase_reuse(
    monkeypatch,
):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(9641)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex128)
    bins = (29, 211, 619, 1481, 3073)
    calls = []
    original = chisq_torch._accelerator_single_point_bin_sums

    def record_dispatch(correlation, points, bin_edges):
        calls.append((points.numel(), bin_edges))
        return original(correlation, points, bin_edges)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        correlation._data.tensor.requires_grad_(True)
        expected = _direct_shift_sum(
            correlation._data.tensor, np.array([701]), bins
        )
        monkeypatch.setattr(
            chisq_torch,
            "_accelerator_single_point_phase_reuse_eligible",
            lambda *args: True,
        )
        monkeypatch.setattr(
            chisq_torch,
            "_accelerator_single_point_bin_sums",
            record_dispatch,
        )
        actual = chisq_torch.shift_sum(
            correlation, np.array([701], dtype=np.int64), bins
        )

    assert calls == [(1, bins)]
    assert torch.equal(actual._data.tensor, expected)


def test_accelerator_single_point_shift_sum_is_bitwise_exact():
    from pycbc.vetoes import chisq_torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        pytest.skip("no Torch accelerator is available")

    rng = np.random.default_rng(9741)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    bins = (29, 211, 619, 1481, 3073)

    with scheme.TorchScheme(device):
        correlation = FrequencySeries(values, delta_f=0.125)
        expected = _direct_shift_sum(
            correlation._data.tensor, np.array([701]), bins
        )
        actual = chisq_torch.shift_sum(
            correlation, np.array([701], dtype=np.int64), bins
        )

    torch.testing.assert_close(
        actual._data.tensor.cpu(), expected.cpu(), rtol=1e-5, atol=1e-3
    )


def test_cpu_native_shift_sum_is_zero_copy_and_torch_owned(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    rng = np.random.default_rng(9841)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    points = np.array([701], dtype=np.int64)
    bins = (29, 211, 619, 1481, 3073)
    observed = {}
    original = chisq_cpu.point_chisq_code

    def record_native(output, corr, count, length, shifts, edges, num_bins):
        observed.update(
            output_dtype=output.dtype,
            corr_dtype=corr.dtype,
            shift_dtype=shifts.dtype,
            edge_dtype=edges.dtype,
            corr_pointer=corr.__array_interface__["data"][0],
            output_pointer=output.__array_interface__["data"][0],
            corr_owner=isinstance(corr.base, torch.Tensor),
            output_owner=isinstance(output.base, torch.Tensor),
            shift_owner=isinstance(shifts.base, torch.Tensor),
            corrected_shift=shifts.copy(),
        )
        return original(
            output, corr, count, length, shifts, edges, num_bins
        )

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        expected = _direct_shift_sum(
            correlation._data.tensor, points, bins
        )
        monkeypatch.setattr(chisq_cpu, "point_chisq_code", record_native)
        result = chisq_torch.shift_sum(correlation, points, bins)

    torch.testing.assert_close(
        result._data.tensor, expected, rtol=3e-6, atol=3e-6
    )
    assert observed["output_dtype"] == np.float64
    assert observed["corr_dtype"] == np.complex64
    assert observed["shift_dtype"] == np.float64
    assert observed["edge_dtype"] == np.uint32
    assert observed["corr_pointer"] == correlation._data.tensor.data_ptr()
    assert observed["output_pointer"] != result._data.tensor.data_ptr()
    assert observed["corr_owner"]
    assert observed["output_owner"]
    assert observed["shift_owner"]
    np.testing.assert_allclose(
        observed["corrected_shift"],
        points * np.pi / chisq_torch._CPU_POINT_CHISQ_PI,
        rtol=0,
        atol=1e-12,
    )
    # The native accumulator is Torch-owned, while the final cast returns an
    # independent Torch allocation that has never been exported to NumPy.
    original_count = result._data.tensor.numel()
    result._data.tensor.resize_(original_count + 1)
    result._data.tensor.resize_(original_count)


def test_cpu_native_power_chisq_preserves_autograd_fallback(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    generator = torch.Generator().manual_seed(9941)
    bins = (29, 211, 619, 1481, 3073)
    points = np.array([701], dtype=np.int64)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(4096, dtype=np.complex64), delta_f=0.125
        )
        correlation._data.tensor = torch.randn(
            4096,
            dtype=torch.complex64,
            generator=generator,
            requires_grad=True,
        )
        snr_tensor = torch.tensor(
            [1.0 + 0.5j], dtype=torch.complex64, requires_grad=True
        )

        def fail_native(*args, **kwargs):
            raise AssertionError("autograd input entered the native CPU path")

        monkeypatch.setattr(chisq_cpu, "point_chisq_code", fail_native)
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr_tensor),
            0.1,
            bins,
            points,
        )
        corr_grad, snr_grad = torch.autograd.grad(
            result._data.tensor.sum(),
            (correlation._data.tensor, snr_tensor),
        )

    assert torch.isfinite(torch.view_as_real(corr_grad)).all()
    assert torch.isfinite(torch.view_as_real(snr_grad)).all()
    assert torch.count_nonzero(corr_grad[bins[0]:bins[-1]]) > 0
    assert torch.count_nonzero(snr_grad) > 0


def test_cpu_native_single_power_chisq_is_zero_copy_and_bitwise(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    rng = np.random.default_rng(9891)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    points = np.array([701], dtype=np.int64)
    bins = (29, 211, 619, 1481, 3073)
    norm = 0.117
    observed = {}
    original_single = chisq_cpu.point_chisq_code_single_double

    def record_single(corr, snr, length, shift, edges, num_bins, snr_norm):
        observed.update(
            corr_pointer=corr.__array_interface__["data"][0],
            snr_pointer=snr.__array_interface__["data"][0],
            corr_owner=isinstance(corr.base, torch.Tensor),
            snr_owner=isinstance(snr.base, torch.Tensor),
            corr_dtype=corr.dtype,
            snr_dtype=snr.dtype,
            shift=shift,
            edge_dtype=edges.dtype,
        )
        return original_single(
            corr, snr, length, shift, edges, num_bins, snr_norm
        )

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        snr_tensor = torch.tensor([1.25 - 0.75j], dtype=torch.complex64)
        generic_points = torch.as_tensor(points, dtype=torch.float64)
        expected = chisq_torch._cpu_native_point_chisq(
            correlation._data.tensor,
            generic_points,
            bins,
            snr=snr_tensor,
            snr_norm=norm,
        )

        def fail_generic(*args, **kwargs):
            raise AssertionError("single search point used the generic kernel")

        monkeypatch.setattr(
            chisq_cpu, "point_chisq_code_single_double", record_single
        )
        monkeypatch.setattr(chisq_cpu, "point_chisq_code", fail_generic)
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr_tensor),
            norm,
            bins,
            points,
        )

    assert torch.equal(result._data.tensor, expected)
    assert observed["corr_pointer"] == correlation._data.tensor.data_ptr()
    assert observed["snr_pointer"] == snr_tensor.data_ptr()
    assert observed["corr_owner"]
    assert observed["snr_owner"]
    assert observed["corr_dtype"] == np.complex64
    assert observed["snr_dtype"] == np.complex64
    assert observed["edge_dtype"] == np.uint32
    assert observed["shift"] == pytest.approx(
        points[0] * np.pi / chisq_torch._CPU_POINT_CHISQ_PI,
        rel=0,
        abs=1e-12,
    )
    original_count = result._data.tensor.numel()
    result._data.tensor.resize_(original_count + 1)
    result._data.tensor.resize_(original_count)


@pytest.mark.parametrize("point", (0, 701, 2047, 4095))
def test_cpu_native_single_kernel_matches_generic_high_precision(point):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(9911 + point)
    correlation = torch.from_numpy(
        (
            rng.normal(size=4096) + 1j * rng.normal(size=4096)
        ).astype(np.complex64)
    )
    snr = torch.tensor([0.625 + 1.75j], dtype=torch.complex64)
    points = torch.tensor([point], dtype=torch.float64)
    bins = (29, 211, 211, 619, 1481, 3073)
    norm = 0.083

    generic = chisq_torch._cpu_native_point_chisq(
        correlation, points, bins, snr=snr, snr_norm=norm
    )
    specialized = chisq_torch._cpu_native_single_point_chisq(
        correlation, point, bins, snr, norm
    )

    assert torch.equal(specialized, generic)


@pytest.mark.parametrize(
    "points",
    (
        np.array([701, 1701], dtype=np.int64),
        np.array([701], dtype=np.int32),
    ),
)
def test_cpu_native_single_power_chisq_preserves_generic_fallback(
    points, monkeypatch
):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    rng = np.random.default_rng(9991 + points.size)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    bins = (29, 211, 619, 1481, 3073)
    snr_tensor = torch.full(
        (points.size,), 1.25 - 0.75j, dtype=torch.complex64
    )
    generic_calls = []
    original_generic = chisq_cpu.point_chisq_code

    def fail_single(*args, **kwargs):
        raise AssertionError("unsupported points used the scalar kernel")

    def record_generic(*args, **kwargs):
        generic_calls.append(True)
        return original_generic(*args, **kwargs)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        monkeypatch.setattr(
            chisq_cpu, "point_chisq_code_single_double", fail_single
        )
        monkeypatch.setattr(chisq_cpu, "point_chisq_code", record_generic)
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr_tensor),
            0.117,
            bins,
            points,
        )

    assert generic_calls == [True]
    assert result._data.tensor.shape == (points.size,)
    assert torch.isfinite(result._data.tensor).all()


@pytest.mark.parametrize("dual_input", ("correlation", "snr", "norm"))
def test_cpu_native_power_chisq_preserves_forward_ad_fallback(
    dual_input, monkeypatch
):
    from pycbc.vetoes import chisq_torch

    generator = torch.Generator().manual_seed(10041)
    bins = (29, 211, 619, 1481, 3073)
    points = np.array([701], dtype=np.int64)
    native_calls = []
    original_native = chisq_torch._cpu_native_point_chisq

    def record_native(corr, pts, edges, snr=None, snr_norm=None):
        native_calls.append(snr is not None)
        if snr is not None:
            raise AssertionError("forward dual entered fused native path")
        return original_native(corr, pts, edges, snr, snr_norm)

    monkeypatch.setattr(
        chisq_torch, "_cpu_native_point_chisq", record_native
    )

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(4096, dtype=np.complex64), delta_f=0.125
        )
        corr_primal = torch.randn(
            4096, dtype=torch.complex64, generator=generator
        )
        snr_primal = torch.tensor([1.0 + 0.5j], dtype=torch.complex64)
        norm_primal = torch.tensor(0.1, dtype=torch.float32)

        with torch.autograd.forward_ad.dual_level():
            if dual_input == "correlation":
                correlation._data.tensor = torch.autograd.forward_ad.make_dual(
                    corr_primal, torch.ones_like(corr_primal)
                )
                snr_tensor = snr_primal
                norm_tensor = norm_primal
            elif dual_input == "snr":
                correlation._data.tensor = corr_primal
                snr_tensor = torch.autograd.forward_ad.make_dual(
                    snr_primal, torch.ones_like(snr_primal)
                )
                norm_tensor = norm_primal
            else:
                correlation._data.tensor = corr_primal
                snr_tensor = snr_primal
                norm_tensor = torch.autograd.forward_ad.make_dual(
                    norm_primal, torch.ones_like(norm_primal)
                )

            result = chisq_torch.power_chisq_at_points_from_precomputed(
                correlation,
                TorchArrayData(snr_tensor),
                norm_tensor,
                bins,
                points,
            )
            tangent = torch.autograd.forward_ad.unpack_dual(
                result._data.tensor
            ).tangent
            assert tangent is not None
            assert torch.isfinite(tangent).all()
            assert torch.count_nonzero(tangent) > 0

    assert True not in native_calls


def test_cpu_native_shift_sum_preserves_forward_point_tangent(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    generator = torch.Generator().manual_seed(10141)
    values = torch.randn(4096, dtype=torch.complex64, generator=generator)
    bins = (29, 211, 619, 1481, 3073)

    def fail_native(*args, **kwargs):
        raise AssertionError("forward point dual entered native CPU path")

    monkeypatch.setattr(chisq_cpu, "point_chisq_code", fail_native)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(4096, dtype=np.complex64), delta_f=0.125
        )
        correlation._data.tensor = values
        with torch.autograd.forward_ad.dual_level():
            points = torch.autograd.forward_ad.make_dual(
                torch.tensor([701.0], dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
            )
            result = chisq_torch.shift_sum(correlation, points, bins)
            tangent = torch.autograd.forward_ad.unpack_dual(
                result._data.tensor
            ).tangent
            assert tangent is not None
            assert torch.isfinite(tangent).all()
            assert torch.count_nonzero(tangent) > 0


def test_cpu_native_shift_sum_rejects_negative_bit_view(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    neg_view = getattr(torch, "_neg_view", None)
    if neg_view is None:
        pytest.skip("this Torch version cannot construct a negative-bit view")

    generator = torch.Generator().manual_seed(10941)
    bins = (29, 211, 619, 1481, 3073)
    points = np.array([701], dtype=np.int64)
    base = torch.randn(4096, dtype=torch.complex64, generator=generator)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(4096, dtype=np.complex64), delta_f=0.125
        )
        correlation._data.tensor = neg_view(base)
        expected = _direct_shift_sum(correlation._data.tensor, points, bins)

        def fail_native(*args, **kwargs):
            raise AssertionError("negative-bit view entered native CPU path")

        monkeypatch.setattr(chisq_cpu, "point_chisq_code", fail_native)
        result = chisq_torch.shift_sum(correlation, points, bins)

    torch.testing.assert_close(
        result._data.tensor, expected, rtol=3e-6, atol=3e-6
    )


@pytest.mark.parametrize("dtype", (np.int64, np.float64))
def test_cpu_native_index_view_is_request_local_and_zero_copy(dtype):
    from pycbc.vetoes import chisq_torch

    with scheme.TorchScheme("cpu"):
        indices = Array(np.array([101, 701], dtype=dtype))
        tensor = indices._data.tensor
        first = chisq_torch._cpu_native_index_view(indices)
        second = chisq_torch._cpu_native_index_view(indices._data)

        assert isinstance(first, np.ndarray)
        assert first.dtype == np.dtype(dtype)
        assert first.flags.c_contiguous
        assert first.__array_interface__["data"][0] == tensor.data_ptr()
        assert second.__array_interface__["data"][0] == tensor.data_ptr()
        assert first is not second

        tensor[0] = 303
        assert first[0] == 303
        first[1] = 909
        assert tensor[1].item() == 909


@pytest.mark.parametrize(
    "tensor",
        (
            torch.tensor([101], dtype=torch.int32),
            torch.arange(4, dtype=torch.int64).reshape(2, 2),
        torch.arange(8, dtype=torch.int64)[::2],
        torch.tensor([101 + 0j], dtype=torch.complex128).conj(),
        torch.sparse_coo_tensor(
            torch.tensor([[0]]), torch.tensor([101]), size=(1,)
        ),
    ),
)
def test_cpu_native_index_view_fails_closed_for_unsupported_storage(tensor):
    from pycbc.vetoes import chisq_torch

    assert chisq_torch._cpu_native_index_view(tensor) is tensor


def test_cpu_native_index_view_rejects_forward_ad_state(monkeypatch):
    from pycbc.vetoes import chisq_torch

    indices = torch.tensor([101], dtype=torch.int64)
    monkeypatch.setattr(
        chisq_torch, "_has_forward_ad_state", lambda tensor: True
    )
    assert chisq_torch._cpu_native_index_view(indices) is indices


def test_cpu_native_index_view_fails_closed_without_numpy_abi(monkeypatch):
    from pycbc.vetoes import chisq_torch

    indices = torch.tensor([101], dtype=torch.int64)

    def fail_export(*args, **kwargs):
        raise RuntimeError("NumPy support unavailable")

    monkeypatch.setattr(torch.Tensor, "numpy", fail_export)
    assert chisq_torch._cpu_native_index_view(indices) is indices


def test_cpu_native_index_view_rejects_negative_bit_view():
    from pycbc.vetoes import chisq_torch

    neg_view = getattr(torch, "_neg_view", None)
    if neg_view is None:
        pytest.skip("this Torch version cannot construct a negative-bit view")
    indices = neg_view(torch.tensor([101], dtype=torch.int64))
    assert indices.is_neg()
    assert chisq_torch._cpu_native_index_view(indices) is indices


@pytest.mark.parametrize("device", ("cuda", "mps"))
def test_cpu_native_index_view_rejects_accelerator_storage(device):
    from pycbc.vetoes import chisq_torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    indices = torch.tensor([101], dtype=torch.int64, device=device)
    assert chisq_torch._cpu_native_index_view(indices) is indices


def test_torch_int64_array_dispatches_single_search_with_bitwise_parity(
    monkeypatch,
):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(10871)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    bins = (29, 211, 619, 1481, 3073)
    norm = 0.117
    calls = []
    original_single = chisq_torch._cpu_native_single_point_chisq

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        snr_tensor = torch.tensor([1.25 - 0.75j], dtype=torch.complex64)
        indices = Array(np.array([700], dtype=np.int64)) + 1
        # PyCBC historically promotes scalar arithmetic on integer Arrays to
        # float64.  The zero-copy native path must accept that public contract
        # rather than changing global Array promotion for this optimization.
        assert indices._data.tensor.dtype == torch.float64
        expected = chisq_torch._cpu_native_point_chisq(
            correlation._data.tensor,
            torch.tensor([701.0], dtype=torch.float64),
            bins,
            snr=snr_tensor,
            snr_norm=norm,
        )

        def record_single(*args, **kwargs):
            calls.append(True)
            return original_single(*args, **kwargs)

        def fail_generic(*args, **kwargs):
            raise AssertionError("eligible Torch int64 index missed scalar path")

        monkeypatch.setattr(
            chisq_torch, "_cpu_native_single_point_chisq", record_single
        )
        monkeypatch.setattr(
            chisq_torch, "_cpu_native_point_chisq", fail_generic
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr_tensor),
            norm,
            bins,
            indices,
        )

    assert calls == [True]
    assert torch.equal(result._data.tensor, expected)


def test_torch_int64_array_dispatches_sparse_search_with_bitwise_parity(
    monkeypatch,
):
    from pycbc.vetoes import chisq_torch

    rng = np.random.default_rng(10891)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    bins = (29, 211, 619, 1481, 3073)
    norm = 0.117
    calls = []
    original_generic = chisq_torch._cpu_native_point_chisq

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(values, delta_f=0.125)
        snr_tensor = torch.tensor(
            [1.25 - 0.75j, -0.5 + 0.25j], dtype=torch.complex64
        )
        indices = Array(np.array([700, 1700], dtype=np.int64)) + 1
        assert indices._data.tensor.dtype == torch.float64
        expected = original_generic(
            correlation._data.tensor,
            torch.tensor([701.0, 1701.0], dtype=torch.float64),
            bins,
            snr=snr_tensor,
            snr_norm=norm,
        )

        def record_generic(*args, **kwargs):
            calls.append(isinstance(args[1], np.ndarray))
            return original_generic(*args, **kwargs)

        monkeypatch.setattr(
            chisq_torch, "_cpu_native_point_chisq", record_generic
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            TorchArrayData(snr_tensor),
            norm,
            bins,
            indices,
        )

    assert calls == [True]
    assert torch.equal(result._data.tensor, expected)


def test_cpu_native_eligibility_respects_cython_integer_bounds(monkeypatch):
    from pycbc.vetoes import chisq_torch

    correlation = torch.zeros(8, dtype=torch.complex64)
    points = torch.zeros(2, dtype=torch.float64)
    bins = (0, 4, 8)
    assert chisq_torch._cpu_native_eligible(correlation, points, bins)

    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_INT_MAX", 7)
    assert not chisq_torch._cpu_native_eligible(correlation, points, bins)

    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_INT_MAX", 8)
    too_many_points = torch.zeros(9, dtype=torch.float64)
    assert not chisq_torch._cpu_native_eligible(
        correlation, too_many_points, bins
    )
    too_many_bins = (0,) * 9 + (8,)
    assert not chisq_torch._cpu_native_eligible(
        correlation, points, too_many_bins
    )

    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_INT_MAX", 9)
    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_UINT_MAX", 7)
    assert not chisq_torch._cpu_native_eligible(correlation, points, bins)

    search_points = np.array([3], dtype=np.int64)
    snr = torch.ones(1, dtype=torch.complex64)
    assert not chisq_torch._cpu_native_single_search_eligible(
        correlation, search_points, bins, snr
    )

    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_UINT_MAX", 8)
    assert chisq_torch._cpu_native_single_search_eligible(
        correlation, search_points, bins, snr
    )
    monkeypatch.setattr(chisq_torch, "_CPU_NATIVE_INT_MAX", 7)
    assert not chisq_torch._cpu_native_single_search_eligible(
        correlation, search_points, bins, snr
    )


def test_cpu_native_paths_reject_tensor_subclasses(monkeypatch):
    from pycbc.vetoes import chisq_cpu, chisq_torch

    class DispatchTensor(torch.Tensor):
        pass

    def subclass(tensor):
        return torch.Tensor._make_subclass(
            DispatchTensor, tensor, tensor.requires_grad
        )

    bins = (0, 4, 8)
    correlation = torch.ones(8, dtype=torch.complex64)
    points = torch.tensor([3.0], dtype=torch.float64)
    sparse_points = np.array([3], dtype=np.int64)
    snr = torch.ones(1, dtype=torch.complex64)

    assert not chisq_torch._cpu_native_corr_eligible(
        subclass(correlation), bins
    )
    assert not chisq_torch._cpu_native_eligible(
        correlation, subclass(points), bins
    )
    subclass_points = subclass(torch.tensor([3], dtype=torch.int64))
    assert chisq_torch._cpu_native_index_view(subclass_points) is subclass_points
    assert not chisq_torch._cpu_native_sparse_search_eligible(
        correlation, sparse_points, bins, subclass(snr)
    )

    def fail_native(*args, **kwargs):
        raise AssertionError("Tensor subclass entered native CPU path")

    monkeypatch.setattr(chisq_cpu, "point_chisq_code", fail_native)
    with scheme.TorchScheme("cpu"):
        series = FrequencySeries(
            np.ones(8, dtype=np.complex64), delta_f=0.125
        )
        series._data.tensor = subclass(correlation)
        result = chisq_torch.shift_sum(series, sparse_points, bins)

    expected = _direct_shift_sum(series._data.tensor, sparse_points, bins)
    torch.testing.assert_close(result._data.tensor, expected)


@pytest.mark.parametrize(
    "special_input", ("correlation", "indices", "snr", "snr_norm")
)
@pytest.mark.parametrize("special_kind", ("subclass", "inference"))
def test_cpu_native_power_chisq_rejects_special_tensor_dispatch(
    monkeypatch, special_input, special_kind
):
    from pycbc.vetoes import chisq_torch

    if special_kind == "inference" and chisq_torch._TORCH_IS_INFERENCE is None:
        pytest.skip("this Torch version cannot identify inference tensors")

    class DispatchTensor(torch.Tensor):
        pass

    def make_special(tensor):
        if special_kind == "subclass":
            return torch.Tensor._make_subclass(
                DispatchTensor, tensor, tensor.requires_grad
            )
        with torch.inference_mode():
            return tensor.clone()

    bins = (0, 4, 8)
    correlation = torch.tensor(
        [
            1 + 0.25j,
            -0.5 + 0.75j,
            0.125 - 0.25j,
            0.5 + 0.5j,
            -0.25 + 0.125j,
            0.75 - 0.5j,
            -0.125 - 0.75j,
            0.25 + 0.5j,
        ],
        dtype=torch.complex64,
    )
    indices = torch.tensor([3.0], dtype=torch.float64)
    snr = torch.tensor([1.25 - 0.75j], dtype=torch.complex64)
    snr_norm = torch.tensor(0.117, dtype=torch.float64)

    def evaluate(corr_tensor, point_tensor, snr_tensor, norm_tensor):
        with scheme.TorchScheme("cpu"):
            series = FrequencySeries(
                np.ones(8, dtype=np.complex64), delta_f=0.125
            )
            series._data.tensor = corr_tensor
            return chisq_torch.power_chisq_at_points_from_precomputed(
                series,
                TorchArrayData(snr_tensor),
                norm_tensor,
                bins,
                point_tensor,
            )._data.tensor

    expected = evaluate(correlation, indices, snr, snr_norm).clone()
    inputs = {
        "correlation": correlation,
        "indices": indices,
        "snr": snr,
        "snr_norm": snr_norm,
    }
    inputs[special_input] = make_special(inputs[special_input])

    original_point = chisq_torch._cpu_native_point_chisq

    def fail_single(*args, **kwargs):
        raise AssertionError("special tensor entered native power-chisq path")

    def reject_power_only(*args, **kwargs):
        # The fallback may still use the safe native shift-sum calculation;
        # it does not consume snr or snr_norm. Only reject the fused native
        # power-chi-squared form that would bypass their dispatch semantics.
        if "snr" in kwargs or len(args) > 3:
            raise AssertionError(
                "special tensor entered native power-chisq path"
            )
        return original_point(*args, **kwargs)

    monkeypatch.setattr(
        chisq_torch, "_cpu_native_single_point_chisq", fail_single
    )
    monkeypatch.setattr(
        chisq_torch, "_cpu_native_point_chisq", reject_power_only
    )
    context = (
        torch.inference_mode()
        if special_kind == "inference"
        else torch.enable_grad()
    )
    with context:
        actual = evaluate(
            inputs["correlation"],
            inputs["indices"],
            inputs["snr"],
            inputs["snr_norm"],
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_single_point_bin_sums_preserves_correlation_autograd():
    from pycbc.vetoes.chisq_torch import _cpu_single_point_bin_sums

    generator = torch.Generator().manual_seed(10341)
    correlation = torch.randn(
        4096, dtype=torch.complex64, generator=generator, requires_grad=True
    )
    bins = (29, 211, 619, 1481, 3073)
    point = torch.tensor([701.0], dtype=torch.float64)
    step_angle = 2 * torch.pi * point / correlation.numel()
    step = torch.polar(torch.ones_like(step_angle), step_angle)

    bin_sums = _cpu_single_point_bin_sums(
        correlation, bins, step_angle, step
    )
    loss = torch.view_as_real(bin_sums).square().sum()
    gradient = torch.autograd.grad(loss, correlation)[0]

    assert gradient.dtype == correlation.dtype
    assert torch.isfinite(torch.view_as_real(gradient)).all()
    assert torch.count_nonzero(gradient[bins[0]:bins[-1]]) > 0
    assert torch.count_nonzero(gradient[:bins[0]]) == 0
    assert torch.count_nonzero(gradient[bins[-1]:]) == 0


def test_cpu_single_point_bin_sums_improves_reduction_precision():
    from pycbc.vetoes.chisq_torch import _cpu_single_point_bin_sums

    rng = np.random.default_rng(11341)
    values = (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    ).astype(np.complex64)
    correlation = torch.from_numpy(values)
    bins = (29, 211, 619, 1481, 3073)
    point = torch.tensor([701.0], dtype=torch.float64)
    step_angle = 2 * torch.pi * point / correlation.numel()
    step = torch.polar(torch.ones_like(step_angle), step_angle)

    phases = step[:, None].expand(-1, bins[-1] - bins[0]).clone()
    phases[:, 0] = torch.polar(
        torch.ones_like(step_angle), step_angle * bins[0]
    )
    torch.cumprod(phases, dim=1, out=phases)
    weighted = correlation[bins[0]:bins[-1]] * phases[0].to(
        correlation.dtype
    )
    conventional = torch.stack(
        [
            torch.sum(weighted[start - bins[0]:end - bins[0]])
            for start, end in zip(bins, bins[1:])
        ]
    )
    reference = torch.stack(
        [
            torch.sum(
                weighted[start - bins[0]:end - bins[0]],
                dtype=torch.complex128,
            )
            for start, end in zip(bins, bins[1:])
        ]
    )

    actual = _cpu_single_point_bin_sums(
        correlation, bins, step_angle, step
    )[0]

    actual_error = torch.linalg.vector_norm(
        actual.to(reference.dtype) - reference
    )
    conventional_error = torch.linalg.vector_norm(
        conventional.to(reference.dtype) - reference
    )
    assert actual_error < conventional_error


def test_empty_power_chisq_skips_shift_sum(monkeypatch):
    from pycbc.vetoes import chisq_torch

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(2048, dtype=np.complex64), delta_f=0.125
        )
        empty_indices = Array(np.empty(0, dtype=np.int64))
        empty_snr = Array(np.empty(0, dtype=np.complex64))

        def fail_shift_sum(*args, **kwargs):
            raise AssertionError("empty chi-squared launched shift-sum work")

        def fail_point_tensor(*args, **kwargs):
            raise AssertionError("empty chi-squared converted point storage")

        monkeypatch.setattr(chisq_torch, "shift_sum", fail_shift_sum)
        monkeypatch.setattr(chisq_torch, "_point_tensor", fail_point_tensor)
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            empty_snr,
            1.0,
            (17, 193, 541, 1201),
            empty_indices,
        )

    assert isinstance(result._data, TorchArrayData)
    assert result._data.tensor.device.type == "cpu"
    assert result._data.tensor.dtype == torch.float32
    assert result._data.tensor.numel() == 0


def test_power_chisq_consumes_sliced_torch_storage_without_host_conversion(
    monkeypatch,
):
    from pycbc.vetoes import chisq_torch

    original_as_tensor = torch.as_tensor

    def reject_storage_conversion(value, *args, **kwargs):
        if isinstance(value, TorchArrayData):
            raise AssertionError("Torch storage passed through torch.as_tensor")
        return original_as_tensor(value, *args, **kwargs)

    with scheme.TorchScheme("cpu"):
        correlation = FrequencySeries(
            np.ones(2048, dtype=np.complex64), delta_f=0.125
        )
        snr_series = TimeSeries(
            np.ones(2048, dtype=np.complex64), delta_t=1 / 2048
        )
        selected_snr = snr_series[np.array([101], dtype=np.int64)]
        assert isinstance(selected_snr._data, TorchArrayData)
        monkeypatch.setattr(
            chisq_torch.torch, "as_tensor", reject_storage_conversion
        )
        result = chisq_torch.power_chisq_at_points_from_precomputed(
            correlation,
            selected_snr,
            0.1,
            (17, 193, 541, 1201),
            np.array([101], dtype=np.int64),
        )

    assert isinstance(result._data, TorchArrayData)
    assert torch.isfinite(result._data.tensor).all()
