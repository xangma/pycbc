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
from pycbc.events import coinc
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.mark.parametrize("method", ("python", "cython"))
def test_cluster_over_time_stays_on_torch_device(monkeypatch, method):
    host_stat = np.array(
        [5.0, 2.0, 7.0, 7.0, -1.0, 4.0, 8.0, 3.0]
    )
    host_time = np.array(
        [4.0, 0.0, 1.0, 1.25, 8.0, 4.25, 7.75, 12.0]
    )
    expected = coinc.cluster_over_time(
        host_stat, host_time, window=0.5, method=method
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("clustering used the NumPy/Cython path")

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("clustering copied or synchronized Torch data")

    with scheme.TorchScheme("cpu"):
        torch_stat = Array(host_stat)
        torch_time = Array(host_time)
        monkeypatch.setattr(coinc, "timecluster_cython", reject_host_path)
        monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        monkeypatch.setattr(torch.Tensor, "cpu", reject_host_transfer)
        monkeypatch.setattr(torch.Tensor, "item", reject_host_transfer)
        actual = coinc.cluster_over_time(
            torch_stat, torch_time, window=0.5, method=method
        )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == "cpu"
    assert actual._data.tensor.dtype == torch.int64
    np.testing.assert_array_equal(
        actual._data.tensor.detach().numpy(), expected
    )


def test_cluster_over_time_raw_torch_nan_and_validation():
    times = torch.tensor([0.0, 0.1, 0.2, 2.0], dtype=torch.float64)
    cases = (
        ([1.0, np.nan, 3.0, 4.0], [1, 3], [2, 3]),
        ([np.nan, 4.0, 3.0, 2.0], [0, 3], [0, 3]),
        ([1.0, 2.0, np.nan, np.nan], [2, 3], [1, 3]),
    )
    for statistics, python_expected, cython_expected in cases:
        stat = torch.tensor(statistics, dtype=torch.float64)
        for method, expected in (
            ("python", python_expected),
            ("cython", cython_expected),
        ):
            actual = coinc.cluster_over_time(
                stat, times, window=0.5, method=method
            )
            assert isinstance(actual, torch.Tensor)
            np.testing.assert_array_equal(actual.numpy(), expected)

    empty = coinc.cluster_over_time(
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        window=0.5,
    )
    assert isinstance(empty, torch.Tensor)
    assert empty.dtype == torch.int64
    assert empty.numel() == 0

    with pytest.raises(NotImplementedError):
        coinc.cluster_over_time(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            window=0.5,
            argmax=lambda value: value.argmax(),
        )


def test_coincidence_torch_backend_public_dispatch():
    """Torch public APIs retain parity after moving to their backend module."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0], dtype=np.float64)
    expected = coinc.time_coincidence(time1, time2, 0.25)
    actual = coinc.time_coincidence(
        torch.as_tensor(time1), torch.as_tensor(time2), 0.25
    )
    for torch_value, numpy_value in zip(actual, expected):
        assert isinstance(torch_value, torch.Tensor)
        np.testing.assert_array_equal(torch_value.numpy(), numpy_value)

    stat = np.array([2.0, 7.0, 4.0, 8.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    expected = coinc.cluster_coincs(
        stat, time1, time2, slide_ids, 1.0, 0.5
    )
    actual = coinc.cluster_coincs(
        torch.as_tensor(stat),
        torch.as_tensor(time1),
        torch.as_tensor(time2),
        torch.as_tensor(slide_ids),
        1.0,
        0.5,
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.numpy(), expected)

    detector_times = (
        time1 + 10.0,
        time2 + 10.0,
        np.array([-1.0, 10.7, 11.0, 12.9], dtype=np.float64),
    )
    expected = coinc.cluster_coincs_multiifo(
        stat, detector_times, slide_ids, 1.0, 0.5
    )
    actual = coinc.cluster_coincs_multiifo(
        torch.as_tensor(stat),
        tuple(torch.as_tensor(value) for value in detector_times),
        torch.as_tensor(slide_ids),
        1.0,
        0.5,
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("storage_kind", ["protocol", "tensor_subclass"])
def test_search_public_storage_preserves_device_and_gradients(
    device, storage_kind,
):
    from pycbc.events import cuts, ranking, single, veto
    from pycbc.types.backend import backend_array

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class PublicStorage:
        backend = "torch"

        def __init__(self, tensor):
            self.backend_array = tensor

    class ExternalTensor(torch.Tensor):
        pass

    source = torch.tensor([8.0, 12.0, 16.0], dtype=torch.float64,
                          device=device, requires_grad=True)
    reduced_chisq = torch.tensor(
        [1.0, 2.0, 4.0], dtype=torch.float64, device=device
    )
    if storage_kind == "protocol":
        values = PublicStorage(source)
        chisq = PublicStorage(reduced_chisq)
        tensor = source
    else:
        tensor = source.as_subclass(ExternalTensor)
        values = tensor
        chisq = reduced_chisq.as_subclass(ExternalTensor)
    with scheme.TorchScheme(device):
        for accessor in (cuts._torch_cut_tensor, single._torch_tensor,
                         veto._torch_veto_tensor):
            assert accessor(values) is tensor
        result = backend_array(ranking.newsnr(values, chisq))
        expected_weight = ((1 + reduced_chisq ** 3) / 2) ** (-1 / 6)
        torch.testing.assert_close(result, source * expected_weight)
        assert result.device == source.device
        result.sum().backward()
        torch.testing.assert_close(source.grad, expected_weight)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_strain_overwhitening_preserves_storage_contract(device, monkeypatch):
    from types import SimpleNamespace
    from pycbc.strain.strain import StrainBuffer
    from pycbc.types.backend import backend_array, wrap_backend_array

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.linspace(0.0, 1.0, 64, dtype=torch.float64,
                            device=device, requires_grad=True)
    expected = torch.fft.rfft(source[-32:]) / 64.0
    with scheme.TorchScheme(device):
        strain = TimeSeries(wrap_backend_array(source), delta_t=1 / 32,
                            epoch=123.0, copy=False)
        psd = FrequencySeries(np.full(17, 2.0), delta_f=1.0)
        psd.psdt = psd
        buffer = SimpleNamespace(strain=strain, segments={}, sample_rate=32,
                                 reduced_pad=0, psds={1.0: psd})

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("strain overwhitening copied to the host")

        monkeypatch.setattr(Array, "numpy", reject_host_transfer)
        result = StrainBuffer.overwhitened_data(buffer, 1.0)
        tensor = backend_array(result, "torch")
        torch.testing.assert_close(tensor, expected)
        assert result.delta_f == 1.0
        assert float(result.epoch) == 124.0
        assert result.psd is psd
        assert StrainBuffer.overwhitened_data(buffer, 1.0) is result
        tensor.abs().square().sum().backward()
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert source.grad[-32:].abs().sum() > 0
