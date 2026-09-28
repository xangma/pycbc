"""Device-resident frame staging for bounded JAX replays."""

import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme
from pycbc.frame.frame import DataBuffer
from pycbc.frame.frame_jax import configure_jax_replay
from pycbc.types import TimeSeries
from pycbc.types.array_jax import to_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize("device", _devices())
def test_bounded_replay_decodes_once_and_advances_on_device(device):
    calls = []
    with scheme.JAXScheme(device) as ctx:
        buffer = object.__new__(DataBuffer)
        buffer.channel_name = "H1:STRAIN"
        buffer.raw_sample_rate = 4
        buffer.read_pos = 10.0
        buffer.force_update_cache = True
        buffer.increment_update_cache = None
        buffer.raw_buffer = TimeSeries(
            np.zeros(12, dtype=np.float64), delta_t=0.25, epoch=7.0
        )

        def read_frame(self, duration):
            calls.append((self.read_pos, duration))
            count = int(duration * self.raw_sample_rate)
            values = np.arange(count, dtype=np.float64)
            return TimeSeries(
                values, delta_t=0.25, epoch=self.read_pos
            )

        def unexpected_cache_update(self):
            raise AssertionError("bounded replay reopened the frame cache")

        buffer._read_frame = types.MethodType(read_frame, buffer)
        buffer.update_cache = types.MethodType(unexpected_cache_update, buffer)
        configure_jax_replay(
            buffer, end_time=16.0, blocksize=2.0,
            read_ahead_seconds=16.0,
        )

        blocks = [buffer.attempt_advance(2.0) for _ in range(3)]

        assert calls == [(10.0, 6.0)]
        assert [float(block.start_time) for block in blocks] == [10, 12, 14]
        assert all(to_jax(block).devices() == {ctx.jax_device}
                   for block in blocks)
        assert to_jax(buffer.raw_buffer).devices() == {ctx.jax_device}
        np.testing.assert_array_equal(
            np.asarray(buffer.raw_buffer), np.arange(12, 24)
        )
        assert buffer.read_pos == 16
        assert float(buffer.raw_buffer.start_time) == 13


@pytest.mark.parametrize("device", _devices())
def test_replay_read_ahead_is_bounded_and_refills(device):
    calls = []
    with scheme.JAXScheme(device):
        buffer = object.__new__(DataBuffer)
        buffer.channel_name = "H1:STRAIN"
        buffer.raw_sample_rate = 2
        buffer.read_pos = 20.0
        buffer.force_update_cache = False
        buffer.increment_update_cache = None
        buffer.raw_buffer = TimeSeries(
            np.zeros(8, dtype=np.float32), delta_t=0.5, epoch=16.0
        )

        def read_frame(self, duration):
            calls.append((self.read_pos, duration))
            values = np.full(
                int(duration * self.raw_sample_rate), self.read_pos,
                dtype=np.float32,
            )
            return TimeSeries(values, delta_t=0.5, epoch=self.read_pos)

        buffer._read_frame = types.MethodType(read_frame, buffer)
        configure_jax_replay(
            buffer, end_time=32.0, blocksize=2.0,
            read_ahead_seconds=4.0,
        )
        for _ in range(6):
            buffer.attempt_advance(2.0)

        assert calls == [(20.0, 4.0), (24.0, 4.0), (28.0, 4.0)]
        np.testing.assert_array_equal(
            np.asarray(buffer.raw_buffer),
            np.array([28, 28, 28, 28, 28, 28, 28, 28], dtype=np.float32),
        )


def test_incremental_frame_discovery_is_not_replaced():
    buffer = object.__new__(DataBuffer)
    buffer.read_pos = 10.0
    buffer.increment_update_cache = "/frames/GPS4/*.gwf"
    buffer.raw_buffer = TimeSeries(
        np.zeros(8, dtype=np.float64), delta_t=0.25, epoch=8.0
    )
    configure_jax_replay(buffer, end_time=14.0, blocksize=2.0)
    assert not hasattr(buffer, "_jax_replay_reader")


def test_nested_status_and_idq_readers_are_configured():
    def reader(start):
        current = object.__new__(DataBuffer)
        current.read_pos = start
        current.increment_update_cache = None
        current.raw_buffer = TimeSeries(
            np.zeros(8, dtype=np.float32), delta_t=0.5, epoch=start - 4
        )
        current._read_frame = types.MethodType(lambda self, duration: None,
                                                current)
        return current

    buffer = reader(10.0)
    buffer.state = reader(10.0)
    buffer.dq = reader(10.0)
    buffer.idq = types.SimpleNamespace(
        idq=reader(10.0), idq_state=reader(10.0)
    )

    configure_jax_replay(buffer, end_time=14.0, blocksize=2.0)

    configured = (
        buffer, buffer.state, buffer.dq,
        buffer.idq.idq, buffer.idq.idq_state,
    )
    assert all(hasattr(current, "_jax_replay_reader")
               for current in configured)


@pytest.mark.parametrize("device", _devices())
def test_noncontiguous_read_ahead_falls_back_to_incremental_reader(device):
    calls = []
    cache_updates = []
    with scheme.JAXScheme(device):
        buffer = object.__new__(DataBuffer)
        buffer.channel_name = "H1:STRAIN"
        buffer.raw_sample_rate = 2
        buffer.read_pos = 30.0
        buffer.force_update_cache = False
        buffer.increment_update_cache = None
        buffer.raw_buffer = TimeSeries(
            np.zeros(8, dtype=np.float32), delta_t=0.5, epoch=26.0
        )

        def read_frame(self, duration):
            calls.append(duration)
            if duration > 2:
                raise RuntimeError("span crosses a gap")
            return TimeSeries(
                np.arange(4, dtype=np.float32),
                delta_t=0.5,
                epoch=self.read_pos,
            )

        def update_cache(self):
            cache_updates.append(self.read_pos)

        buffer._read_frame = types.MethodType(read_frame, buffer)
        buffer.update_cache = types.MethodType(update_cache, buffer)
        configure_jax_replay(
            buffer, end_time=36.0, blocksize=2.0,
            read_ahead_seconds=6.0,
        )

        result = buffer.attempt_advance(2.0)

        assert calls == [6.0, 2.0]
        assert cache_updates == [30.0]
        assert buffer._jax_replay_reader is None
        assert to_jax(result).shape == (4,)
        np.testing.assert_array_equal(
            np.asarray(buffer.raw_buffer),
            np.array([0, 0, 0, 0, 0, 1, 2, 3], dtype=np.float32),
        )
