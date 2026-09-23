"""CUDA search conditioning must honor its CPU-compatibility option."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter import resample
from pycbc.strain import strain as strain_module
from pycbc.types import TimeSeries
from pycbc.types.backend import backend_array, wrap_backend_array

torch = pytest.importorskip("torch")


def test_torch_cpu_search_highpass_defaults_to_lal(monkeypatch):
    rng = np.random.default_rng(2901)
    values = rng.normal(size=4096).astype(np.float32)
    with scheme.CPUScheme():
        expected = resample.highpass(
            TimeSeries(values, delta_t=1 / 1024), frequency=20
        )

    def reject_native(*args, **kwargs):
        raise AssertionError("CPU search used native Torch highpass")

    monkeypatch.setattr(resample, "_torch_butterworth_filter", reject_native)
    with scheme.TorchScheme("cpu"):
        source = TimeSeries(values, delta_t=1 / 1024)
        actual = strain_module._search_highpass(source, 20, None)
        assert backend_array(actual, "torch").device.type == "cpu"
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_search_highpass_cpu_compatibility(monkeypatch):
    rng = np.random.default_rng(2902)
    values = rng.normal(size=8192).astype(np.float32)
    values[211] += 40
    with scheme.CPUScheme():
        expected = resample.highpass(
            TimeSeries(values, delta_t=1 / 2048, epoch=1187007048),
            frequency=25,
        )

    def reject_native(*args, **kwargs):
        raise AssertionError("CPU compatibility mode used CUDA highpass")

    with scheme.TorchScheme("cuda"):
        source = TimeSeries(values, delta_t=1 / 2048, epoch=1187007048)
        with monkeypatch.context() as patch:
            patch.setattr(resample, "_torch_butterworth_filter", reject_native)
            actual = strain_module._search_highpass(source, 25, False)
            with pytest.raises(AssertionError, match="CUDA highpass"):
                strain_module._search_highpass(source, 25, True)
            with pytest.raises(AssertionError, match="CUDA highpass"):
                strain_module._search_highpass(source, 25, None)

        assert backend_array(actual, "torch").device.type == "cuda"
        assert actual.start_time == expected.start_time
        assert actual.delta_t == expected.delta_t
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(source.numpy(), values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_search_highpass_preserves_autograd(monkeypatch):
    rng = np.random.default_rng(2903)
    values = rng.normal(size=4096).astype(np.float32)
    native = resample._torch_butterworth_filter
    calls = []

    def count_native(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(resample, "_torch_butterworth_filter", count_native)
    with scheme.TorchScheme("cuda"):
        tensor = torch.as_tensor(values, device="cuda").requires_grad_()
        source = TimeSeries(
            wrap_backend_array(tensor), delta_t=1 / 1024, copy=False
        )
        result = strain_module._search_highpass(source, 20, False)
        backend_array(result, "torch").square().sum().backward()

    assert calls == [True]
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
