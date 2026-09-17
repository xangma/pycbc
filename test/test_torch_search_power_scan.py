"""Equal-power bins must use the original serial float32 accumulation."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter import sigmasq_series
from pycbc.types import FrequencySeries

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_long_scan_and_bin_edges_match_cpu(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    values = np.ones(2**20 + 1, dtype=np.complex64)
    values[:4096] = 100
    with scheme.CPUScheme(1):
        expected = sigmasq_series(FrequencySeries(values, delta_f=0.25)).numpy().copy()
    with scheme.TorchScheme(device):
        actual = sigmasq_series(FrequencySeries(values, delta_f=0.25))
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert actual.dtype == np.float32
        assert actual._data.tensor.device.type == device
        boundaries = np.arange(1, 16) * expected[-2] / 16
        np.testing.assert_array_equal(
            np.searchsorted(actual.numpy()[:-1], boundaries),
            np.searchsorted(expected[:-1], boundaries),
        )
        # Ensure the fixture detects replacing serial addition by Torch's scan.
        native = torch.as_tensor(np.abs(values[1:-1]) ** 2, device=device).cumsum(0)
        assert native[-1].item() != expected[-2]
