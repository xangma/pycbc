"""Coordinate precision and differentiable transfers between Torch contexts."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc.types import Array, TimeSeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402


def _require_device(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


@pytest.mark.parametrize("source,target", (
    ("cpu", "cpu"), ("cpu", "mps"), ("mps", "cpu"), ("mps", "mps"),
    ("cpu", "cuda"), ("cuda", "cpu"),
))
@pytest.mark.parametrize("operation", ("sin", "concatenate", "add"))
def test_context_transfer_preserves_autograd(
    monkeypatch, source, target, operation
):
    _require_device(source)
    _require_device(target)
    with scheme.TorchScheme(source):
        leaf = torch.tensor([0.2, 0.4, 0.8], device=source, requires_grad=True)
        array = Array(TorchArrayData(leaf), copy=False)

    def reject_host_conversion(_self):
        raise AssertionError("Torch context transfer must not use NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_conversion)
    with scheme.TorchScheme(target) as active:
        if operation == "sin":
            result = np.sin(array)
            expected = leaf.cos()
        elif operation == "concatenate":
            result = np.concatenate((array, array))
            expected = torch.full_like(leaf, 2.0)
        else:
            result = array + 1
            expected = torch.ones_like(leaf)
        tensor = result.backend_array
        assert tensor.device.type == target
        assert result._scheme is active
        gradient, = torch.autograd.grad(tensor.sum(), leaf)
        torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_absolute_sample_times_preserve_gps_spacing(device):
    _require_device(device)
    with scheme.TorchScheme(device):
        series = TimeSeries(np.ones(8, dtype=np.float32),
                            delta_t=1 / 4096, epoch=1126259462.125)
        times = series.sample_times.backend_array
        assert times.dtype == torch.float64
        expected = 1126259462.125 + torch.arange(
            8, device=device, dtype=torch.float64
        ) / 4096
        torch.testing.assert_close(times, expected, rtol=0, atol=0)
        assert torch.all(times[1:] > times[:-1])


def test_mps_absolute_sample_times_reject_loss_of_precision():
    _require_device("mps")
    with scheme.TorchScheme("mps"):
        series = TimeSeries(np.ones(8, dtype=np.float32),
                            delta_t=1 / 4096, epoch=1126259462.125)
        with pytest.raises(
            TypeError, match="Absolute sample times require float64"
        ):
            _ = series.sample_times
