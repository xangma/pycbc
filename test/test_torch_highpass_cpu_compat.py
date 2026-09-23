"""CPU Torch highpass parity and native dispatch for special tensors."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter import resample
from pycbc.types import TimeSeries
from pycbc.types.backend import backend_array, wrap_backend_array

torch = pytest.importorskip("torch")


def _strain():
    rng = np.random.default_rng(20050914)
    values = rng.normal(size=4096).astype(np.float32)
    values[100] += 50
    return values


def test_search_highpass_matches_lal_bitwise(monkeypatch):
    values = _strain()
    with scheme.CPUScheme():
        expected = resample.highpass(
            TimeSeries(values, delta_t=1 / 1024, epoch=1234567890.25),
            20, filter_order=8, attenuation=0.1,
        )

    def reject_native(*args, **kwargs):
        raise AssertionError("ordinary CPU data should use LAL highpass")

    monkeypatch.setattr(resample, "_torch_butterworth_filter", reject_native)
    with scheme.TorchScheme("cpu"):
        source = TimeSeries(values, delta_t=1 / 1024, epoch=1234567890.25)
        actual = resample._highpass_cpu_compatible(
            source, 20, filter_order=8, attenuation=0.1,
        )
        assert backend_array(actual, "torch").device.type == "cpu"
        assert actual.start_time == expected.start_time
        assert actual.delta_t == expected.delta_t
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(source.numpy(), values)


@pytest.mark.parametrize("kind", ["grad", "subclass", "float64", "ordinary"])
def test_nonsearch_highpass_keeps_native_path(monkeypatch, kind):
    class UserTensor(torch.Tensor):
        pass

    native = resample._torch_butterworth_filter
    calls = []

    def count_native(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(resample, "_torch_butterworth_filter", count_native)
    with scheme.TorchScheme("cpu"):
        values = torch.from_numpy(_strain())
        if kind == "grad":
            values = values.requires_grad_()
        elif kind == "subclass":
            values = values.as_subclass(UserTensor)
        elif kind == "float64":
            values = values.double()
        source = TimeSeries(
            wrap_backend_array(values), delta_t=1 / 1024, copy=False,
        )
        filter_func = (
            resample.highpass if kind == "ordinary"
            else resample._highpass_cpu_compatible
        )
        actual = filter_func(source, 20)
        assert calls == [True]
        assert backend_array(actual, "torch").dtype == values.dtype
        if kind == "grad":
            backend_array(actual, "torch").square().sum().backward()
            assert values.grad is not None
            assert torch.isfinite(values.grad).all()


def test_forward_mode_highpass_keeps_native_path(monkeypatch):
    native = resample._torch_butterworth_filter
    calls = []

    def count_native(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(resample, "_torch_butterworth_filter", count_native)
    with scheme.TorchScheme("cpu"), torch.autograd.forward_ad.dual_level():
        primal = torch.from_numpy(_strain())
        dual = torch.autograd.forward_ad.make_dual(
            primal, torch.ones_like(primal)
        )
        source = TimeSeries(
            wrap_backend_array(dual), delta_t=1 / 1024, copy=False,
        )
        result = resample._highpass_cpu_compatible(source, 20)
        assert calls == [True]
        tangent = torch.autograd.forward_ad.unpack_dual(
            backend_array(result, "torch")
        ).tangent
        assert tangent is not None
        assert torch.isfinite(tangent).all()
