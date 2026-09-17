"""Rolling frame buffers accept integral sample counts in every scheme."""

import numpy
import pytest

from pycbc.frame.frame import DataBuffer
from pycbc.scheme import CPUScheme, TorchScheme


@pytest.fixture
def buffer_metadata(monkeypatch):
    monkeypatch.setattr(DataBuffer, "update_cache",
                        lambda self: setattr(self, "stream", None))
    monkeypatch.setattr(DataBuffer, "_retrieve_metadata",
                        staticmethod(lambda *_: (None, 1024.0)))


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_integral_float_buffer_size(buffer_metadata, device):
    if device is not None:
        torch = pytest.importorskip("torch")
        if device.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    scheme = CPUScheme() if device is None else TorchScheme(device=device)
    with scheme:
        buffer = DataBuffer([], "H1:TEST", 1000000000,
                            max_buffer=0.25, dtype=numpy.float64)
        assert len(buffer.raw_buffer) == 256
        assert float(buffer.raw_buffer.start_time) == 999999999.75
        assert buffer.raw_buffer.delta_t == 1 / 1024


@pytest.mark.parametrize("duration", [0, -1, 0.0001, numpy.inf, numpy.nan])
def test_invalid_buffer_size(buffer_metadata, duration):
    with pytest.raises(ValueError, match="integer number of samples"):
        DataBuffer([], "H1:TEST", 1000000000, max_buffer=duration)
