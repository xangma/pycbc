"""PSD public storage operations preserve metadata and differentiability."""

import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc.psd import analytical, estimate, variation  # noqa: E402
from pycbc.types import Array, FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.backend import (  # noqa: E402
    backend_array, wrap_backend_array,
)


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return request.param


def test_psd_interpolation_preserves_metadata_and_gradients(device):
    source = torch.tensor(
        [1.0, 2.0, 4.0], dtype=torch.float32, device=device,
        requires_grad=True,
    )
    with scheme.TorchScheme(device):
        series = FrequencySeries(
            wrap_backend_array(source), delta_f=2, epoch=123, copy=False,
        )
        result = estimate.interpolate(series, delta_f=1)
        values = backend_array(result, "torch")
        torch.testing.assert_close(
            values, source.new_tensor([1.0, 1.5, 2.0, 3.0, 4.0]),
        )
        assert result.delta_f == 1
        assert result.epoch == series.epoch
        values.sum().backward()
        torch.testing.assert_close(
            source.grad, source.new_tensor([1.5, 2.0, 1.5]),
        )


def test_trigger_psd_interpolation_preserves_device_and_gradients(device):
    source = torch.tensor(
        [2.0, 4.0, 8.0], dtype=torch.float32, device=device,
        requires_grad=True,
    )
    indices = source.new_tensor([-1.0, 0.0, 1.0, 3.0, 4.0, 5.0])
    with scheme.TorchScheme(device):
        series = TimeSeries(
            wrap_backend_array(source), delta_t=1, epoch=123, copy=False,
        )
        result = variation.find_trigger_value(
            series, Array(wrap_backend_array(indices), copy=False),
            start=123, sample_rate=2,
        )
        values = backend_array(result, "torch")
        torch.testing.assert_close(
            values, source.new_tensor([1.0, 2.0, 3.0, 6.0, 8.0, 1.0]),
        )
        values.sum().backward()
        torch.testing.assert_close(
            source.grad, source.new_tensor([1.5, 1.0, 1.5]),
        )


def test_flat_psd_preserves_scheme_device_and_precision(device):
    dtype = torch.float32 if device == "mps" else torch.float64
    with scheme.TorchScheme(device):
        result = analytical.flat_unity(4, delta_f=2, low_freq_cutoff=4)
        torch.testing.assert_close(
            backend_array(result, "torch"),
            torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype),
        )
        assert result.delta_f == 2
