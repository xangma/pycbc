# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused tests for the device-native LAL-style TimeSeries taper."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import TimeSeries


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)

from pycbc.types.array_torch import TorchArrayData  # noqa: E402
import pycbc.types.timeseries as timeseries_module  # noqa: E402
from pycbc.types.timeseries import (  # noqa: E402
    _torch_constant_taper_parameters,
    _torch_lal_taper,
)


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_ctx(request):
    """Provide each locally available Torch device."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    ctx = scheme.TorchScheme(device)
    try:
        yield ctx, device
    finally:
        del ctx
        scheme.Scheme._single = None


def _edge_cases(dtype):
    """Return taper inputs exercising sparse, fallback, and flat peaks."""
    cases = {}

    cases["all-zero"] = np.zeros(192, dtype=dtype)

    single = np.zeros(192, dtype=dtype)
    single[73] = 2
    cases["single-nonzero"] = single

    adjacent = np.zeros(192, dtype=dtype)
    adjacent[73:75] = (2, -1)
    cases["adjacent-nonzero"] = adjacent

    separated = np.zeros(192, dtype=dtype)
    separated[17] = 2
    separated[174] = -1
    cases["separated-nonzero"] = separated

    monotonic = np.zeros(192, dtype=dtype)
    monotonic[17:175] = np.linspace(0.25, 2.0, 158, dtype=dtype)
    cases["midpoint-fallback"] = monotonic

    flat = np.zeros(192, dtype=dtype)
    flat[11:181] = np.resize(
        np.asarray((0.5, 1.0, 2.0, 2.0, 1.0), dtype=dtype), 170
    )
    cases["flat-peaks"] = flat

    oscillatory = np.zeros(192, dtype=dtype)
    samples = np.arange(160)
    oscillatory[17:177] = (
        np.sin(2 * np.pi * samples / 23 + 0.3)
        * (1 + 0.002 * samples)
    ).astype(dtype)
    cases["oscillatory"] = oscillatory
    return cases


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "location",
    (
        "TAPER_NONE",
        "TAPER_START",
        "TAPER_END",
        "TAPER_STARTEND",
        "start",
        "end",
        "startend",
    ),
)
def test_torch_lal_taper_matches_lal_edge_cases(
        torch_device_ctx, dtype, location):
    """Match installed LAL without moving or mutating the Torch input."""
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    delta_t = 1 / 2048
    epoch = 123.125
    tolerance = 2e-6 if dtype == np.float32 else 1e-14
    for name, data in _edge_cases(dtype).items():
        expected = TimeSeries(
            data.copy(), delta_t=delta_t, epoch=epoch
        ).taper_timeseries(location=location).numpy()

        with ctx:
            series = TimeSeries(
                data.copy(), delta_t=delta_t, epoch=epoch
            )
            original = series._data.tensor.clone()
            actual = series.taper_timeseries(location=location)

        assert isinstance(actual._data, TorchArrayData)
        assert actual._data.tensor.device.type == device
        assert actual.dtype == np.dtype(dtype)
        assert actual.delta_t == series.delta_t
        assert actual.start_time == series.start_time
        assert actual is not series
        assert actual._data.tensor.data_ptr() != series._data.tensor.data_ptr()
        assert torch.equal(series._data.tensor, original)
        np.testing.assert_allclose(
            actual._data.tensor.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )


def test_torch_lal_taper_has_no_data_dependent_host_boundary(
        torch_device_ctx, monkeypatch):
    """Reject scalar extraction, dynamic compaction, and host conversion."""
    ctx, device = torch_device_ctx
    data = _edge_cases(np.float32)["flat-peaks"]
    expected = TimeSeries(
        data.copy(), delta_t=1 / 1024, epoch=17
    ).taper_timeseries(location="startend").numpy()

    with ctx:
        series = TimeSeries(data, delta_t=1 / 1024, epoch=17)
        original = series._data.tensor.clone()
        original_getitem = torch.Tensor.__getitem__

        def reject_host_boundary(*_args, **_kwargs):
            raise AssertionError("Torch taper crossed a scalar/host boundary")

        def reject_dynamic_index(tensor, key):
            keys = key if isinstance(key, tuple) else (key,)
            if any(isinstance(index, torch.Tensor) for index in keys):
                raise AssertionError("Torch taper used a data-dependent index")
            return original_getitem(tensor, key)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "nonzero", reject_host_boundary)
            patch.setattr(torch.Tensor, "nonzero", reject_host_boundary)
            patch.setattr(torch.Tensor, "item", reject_host_boundary)
            patch.setattr(torch.Tensor, "tolist", reject_host_boundary)
            patch.setattr(torch.Tensor, "numpy", reject_host_boundary)
            patch.setattr(torch.Tensor, "cpu", reject_host_boundary)
            patch.setattr(torch.Tensor, "__bool__", reject_host_boundary)
            patch.setattr(
                torch.Tensor, "__getitem__", reject_dynamic_index
            )
            patch.setattr(TorchArrayData, "numpy", reject_host_boundary)
            patch.setattr(TimeSeries, "lal", reject_host_boundary)
            actual = series.taper_timeseries(location="startend")

    assert actual._data.tensor.device.type == device
    assert torch.equal(series._data.tensor, original)
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )


def test_torch_lal_taper_empty_private_input(torch_device_ctx):
    """The fixed-shape helper keeps its static empty-input guard."""
    ctx, device = torch_device_ctx
    with ctx:
        source = torch.empty(0, dtype=torch.float32, device=device)
        actual = _torch_lal_taper(source, "startend")
    assert actual.shape == source.shape
    assert actual.device == source.device
    assert actual.dtype == source.dtype
    # Empty tensors have the sentinel data_ptr() == 0 even after cloning.
    assert actual is not source


def test_torch_lal_taper_preserves_autograd(torch_device_ctx):
    """Taper application remains differentiable with respect to samples."""
    ctx, device = torch_device_ctx
    data = _edge_cases(np.float32)["oscillatory"]

    with ctx:
        source = torch.tensor(
            data, dtype=torch.float32, device=device, requires_grad=True
        )
        original = source.detach().clone()
        series = TimeSeries(
            TorchArrayData(source), delta_t=1 / 2048, epoch=31, copy=False
        )
        tapered = series.taper_timeseries(location="startend")
        output = tapered._data.tensor
        output.square().sum().backward()
        gradient = source.grad.detach().clone()

        weights = torch.where(
            source.detach() != 0,
            output.detach() / source.detach(),
            torch.ones_like(source),
        )
        expected_gradient = 2 * source.detach() * weights.square()

    assert output.device.type == device
    assert output.dtype == source.dtype
    assert output.grad_fn is not None
    assert gradient.device.type == device
    assert torch.isfinite(gradient).all()
    assert torch.equal(source.detach(), original)
    torch.testing.assert_close(
        gradient, expected_gradient, rtol=2e-6, atol=2e-6
    )


def test_torch_lal_taper_return_lal_remains_explicit_host_boundary():
    """return_lal=True retains the established wrapped-LAL behavior."""
    data = _edge_cases(np.float64)["oscillatory"]
    delta_t = 1 / 4096
    epoch = 42.25
    expected = TimeSeries(
        data.copy(), delta_t=delta_t, epoch=epoch
    ).taper_timeseries(location="startend", return_lal=True)

    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            series = TimeSeries(
                data.copy(), delta_t=delta_t, epoch=epoch
            )
            original = series._data.tensor.clone()
            actual = series.taper_timeseries(
                location="startend", return_lal=True
            )
    finally:
        del ctx
        scheme.Scheme._single = None

    assert type(actual) is type(expected)
    assert actual.deltaT == expected.deltaT
    assert actual.epoch == expected.epoch
    assert torch.equal(series._data.tensor, original)
    np.testing.assert_allclose(
        actual.data.data,
        expected.data.data,
        rtol=1e-14,
        atol=1e-14,
    )


def _constant_taper_data(dtype):
    """Return data with GPS-rounding-sensitive nonzero boundaries."""
    data = np.zeros(97, dtype=dtype)
    data[3:89] = np.linspace(0.25, 2.0, 86, dtype=dtype)
    return data


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "location",
    ("TAPER_START", "TAPER_END", "TAPER_STARTEND", "startend"),
)
def test_torch_constant_taper_matches_legacy_on_device(
        torch_device_ctx, dtype, location):
    """Match constant-gate values, metadata, mutation, and residency."""
    ctx, device = torch_device_ctx
    if device == "mps" and dtype == np.float64:
        pytest.skip("Torch MPS does not support float64")

    rate = 4096
    window = 10 / rate
    epoch = 123.125000001
    data = _constant_taper_data(dtype)
    expected_series = TimeSeries(
        data.copy(), delta_t=1 / rate, epoch=epoch
    )
    expected = expected_series.taper_timeseries(
        location=location, tapermethod="constant", taper_window=window,
    ).numpy()

    with ctx:
        series = TimeSeries(
            data.copy(), delta_t=1 / rate, epoch=epoch
        )
        pointer = series._data.tensor.data_ptr()
        actual = series.taper_timeseries(
            location=location,
            tapermethod="constant",
            taper_window=window,
        )

    assert actual is series
    assert actual._data.tensor.data_ptr() == pointer
    assert actual._data.tensor.device.type == device
    assert actual.dtype == np.dtype(dtype)
    assert actual.delta_t == 1 / rate
    assert actual.start_time == expected_series.start_time
    tolerance = 2e-6 if dtype == np.float32 else 1e-14
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )
    # With a first nonzero at index 3, GPS nanosecond quantization places
    # the start gate at -6 rather than the naive sample-space value -7.
    if location in ("TAPER_START", "TAPER_STARTEND", "startend"):
        assert expected[3] > 0


def test_torch_constant_taper_has_no_data_dependent_host_boundary(
        torch_device_ctx, monkeypatch):
    """The eligible path uses no scalar extraction or host conversion."""
    ctx, device = torch_device_ctx
    data = _constant_taper_data(np.float32)
    expected = TimeSeries(
        data.copy(), delta_t=1 / 4096, epoch=17.000000001
    ).taper_timeseries(
        location="startend", tapermethod="constant",
        taper_window=10.75 / 4096,
    ).numpy()

    with ctx:
        series = TimeSeries(
            data.copy(), delta_t=1 / 4096, epoch=17.000000001
        )

        def reject_host_boundary(*_args, **_kwargs):
            raise AssertionError("constant taper crossed a host boundary")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "nonzero", reject_host_boundary)
            patch.setattr(torch.Tensor, "nonzero", reject_host_boundary)
            patch.setattr(torch.Tensor, "item", reject_host_boundary)
            patch.setattr(torch.Tensor, "tolist", reject_host_boundary)
            patch.setattr(torch.Tensor, "numpy", reject_host_boundary)
            patch.setattr(torch.Tensor, "cpu", reject_host_boundary)
            patch.setattr(torch.Tensor, "__bool__", reject_host_boundary)
            patch.setattr(TorchArrayData, "numpy", reject_host_boundary)
            actual = series.taper_timeseries(
                location="startend", tapermethod="constant",
                taper_window=10.75 / 4096,
            )

    assert actual is series
    assert actual._data.tensor.device.type == device
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )


def test_torch_constant_taper_preserves_fallback(monkeypatch):
    """Unusual rates and scalar classes retain the synchronized path."""
    data = _constant_taper_data(np.float64)
    cases = (
        (1 / 3, 0.75),
        (1 / 4096, np.float64(10 / 4096)),
    )

    def reject_fast_path(*_args, **_kwargs):
        raise AssertionError("unsupported input reached the fast path")

    monkeypatch.setattr(
        timeseries_module, "_torch_constant_taper", reject_fast_path
    )
    ctx = scheme.TorchScheme("cpu")
    try:
        for delta_t, window in cases:
            expected = TimeSeries(
                data.copy(), delta_t=delta_t, epoch=31
            ).taper_timeseries(
                location="startend", tapermethod="constant",
                taper_window=window,
            ).numpy()
            with ctx:
                actual = TimeSeries(
                    data.copy(), delta_t=delta_t, epoch=31
                )
                result = actual.taper_timeseries(
                    location="startend", tapermethod="constant",
                    taper_window=window,
                )
            assert result is actual
            np.testing.assert_allclose(
                actual._data.tensor.detach().cpu().numpy(), expected,
                rtol=1e-14, atol=1e-14,
            )
    finally:
        del ctx
        scheme.Scheme._single = None


def test_torch_constant_taper_preserves_nonfinite_and_leaf_edges():
    """All-zero and requires-grad leaves retain legacy early/error paths."""
    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            zeros = TimeSeries(
                np.zeros(32, dtype=np.float32), delta_t=1 / 4096
            )
            result = zeros.taper_timeseries(
                location="startend", tapermethod="constant",
                taper_window=np.nan,
            )
            assert result is zeros
            assert torch.count_nonzero(zeros._data.tensor) == 0

            leaf = torch.zeros(32, dtype=torch.float32, requires_grad=True)
            zero_leaf = TimeSeries(
                TorchArrayData(leaf), delta_t=1 / 4096, copy=False
            )
            assert zero_leaf.taper_timeseries(
                location="start", tapermethod="constant",
                taper_window=10 / 4096,
            ) is zero_leaf

            nonzero_leaf_tensor = torch.ones(
                32, dtype=torch.float32, requires_grad=True
            )
            nonzero_leaf = TimeSeries(
                TorchArrayData(nonzero_leaf_tensor),
                delta_t=1 / 4096, copy=False,
            )
            with pytest.raises(RuntimeError, match="leaf Variable"):
                nonzero_leaf.taper_timeseries(
                    location="start", tapermethod="constant",
                    taper_window=10 / 4096,
                )
    finally:
        del ctx
        scheme.Scheme._single = None


def test_torch_constant_taper_preserves_nonleaf_autograd(
        torch_device_ctx):
    """The requires-grad fallback retains the legacy differentiable graph."""
    ctx, device = torch_device_ctx
    data = _constant_taper_data(np.float32)
    from pycbc.strain import gate_data

    weights = TimeSeries(
        np.ones_like(data), delta_t=1 / 4096, epoch=9.000000001
    )
    expected = gate_data(weights, (
        (weights.start_time + 3 / 4096, 0, 10 / 4096),
        (weights.start_time + 88 / 4096, 0, 10 / 4096),
    )).numpy()

    with ctx:
        leaf = torch.tensor(
            data, dtype=torch.float32, device=device, requires_grad=True
        )
        backing = leaf * 1
        series = TimeSeries(
            TorchArrayData(backing), delta_t=1 / 4096,
            epoch=9.000000001, copy=False,
        )
        actual = series.taper_timeseries(
            location="startend", tapermethod="constant",
            taper_window=10 / 4096,
        )
        actual._data.tensor.sum().backward()

    assert actual is series
    assert actual._data.tensor.device.type == device
    assert leaf.grad.device.type == device
    np.testing.assert_allclose(
        leaf.grad.detach().cpu().numpy(), expected,
        rtol=2e-6, atol=2e-6,
    )


def test_torch_constant_taper_all_zero_nonleaf_preserves_saved_graph():
    """An all-zero nonleaf remains untouched for an earlier saved graph."""
    ctx = scheme.TorchScheme("cpu")
    try:
        with ctx:
            leaf = torch.ones(32, dtype=torch.float32, requires_grad=True)
            backing = leaf * 0
            saved = backing.square().sum()
            version = backing._version
            series = TimeSeries(
                TorchArrayData(backing), delta_t=1 / 4096, copy=False
            )
            assert series.taper_timeseries(
                location="startend", tapermethod="constant",
                taper_window=10 / 4096,
            ) is series
            assert backing._version == version
            saved.backward()
    finally:
        del ctx
        scheme.Scheme._single = None

    assert torch.equal(leaf.grad, torch.zeros_like(leaf))


def test_torch_constant_taper_parameter_gate():
    """Only ordinary positive integral power-of-two rates are eligible."""
    epoch = TimeSeries(
        np.zeros(1, dtype=np.float32), delta_t=1, epoch=10
    ).start_time
    assert _torch_constant_taper_parameters(
        32, 1 / 4096, epoch, 10 / 4096
    ) is not None
    assert _torch_constant_taper_parameters(
        32, 1 / 3, epoch, 0.25
    ) is None
    assert _torch_constant_taper_parameters(
        32, 1 / 4096, epoch, 0.0
    ) is None
