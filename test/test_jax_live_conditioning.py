"""Live conditioning retains JAX buffers through PSD and FFT operations."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
from pycbc import scheme
from pycbc.types.array_jax import to_jax


def _devices():
    devices = ['cpu']
    try:
        if jax.devices('gpu'):
            devices.append('cuda:0')
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize('device', _devices())
def test_frame_buffer_accepts_floating_sample_rate(device, monkeypatch):
    from pycbc.frame.frame import DataBuffer

    def update_cache(self):
        self.stream = None

    monkeypatch.setattr(DataBuffer, 'update_cache', update_cache)
    monkeypatch.setattr(DataBuffer, '_retrieve_metadata',
                        staticmethod(lambda stream, channel: (None, 4096.0)))
    with scheme.JAXScheme(device) as ctx:
        buffer = DataBuffer([], 'H1:STRAIN', 1000000000,
                            max_buffer=0.5, dtype=np.float32)
        assert len(buffer.raw_buffer) == 2048
        assert buffer.raw_buffer.delta_t == 1 / 4096
        assert float(buffer.raw_buffer.start_time) == 999999999.5
        assert to_jax(buffer.raw_buffer).devices() == {ctx.jax_device}
        np.testing.assert_array_equal(buffer.raw_buffer.numpy(), 0)


