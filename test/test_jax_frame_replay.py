"""Device-resident frame staging for bounded JAX replays."""

import types
import os
import subprocess
import sys
import zlib

import lal
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme
from pycbc.frame import frame_jax, gwf_replay_jax
from pycbc.frame.frame import (
    DataBuffer, StatusBuffer, locations_to_cache, write_frame,
)
from pycbc.frame.frame_jax import (
    JAXReplayFrameReader, JAXReplayReadError, configure_jax_replay,
    retrieve_frame_metadata_jax,
)
from pycbc.types import TimeSeries
from pycbc.types.array_jax import to_jax
import lalframe


_CHANNEL_DTYPES = [np.float32, np.float64, np.complex64, np.complex128,
                   np.uint32, np.int32]


def _write_samples(path, dtype, duration=16, rate=16):
    values = np.arange(duration * rate, dtype=np.float64).astype(dtype)
    if np.dtype(dtype).kind == 'c':
        values += (np.arange(len(values)) % 7 * 1j).astype(dtype)
    with scheme.CPUScheme():
        write_frame(str(path), 'H1:TEST', TimeSeries(
            values, delta_t=1.0 / rate, epoch=1000000000))


def _shifted_channel_frame(tmp_path):
    original = tmp_path / 'original.gwf'
    _write_samples(original, np.float64)
    source = lalframe.FrameUFrFileOpen(str(original), 'r')
    channel = lalframe.FrameUFrChanRead(source, 'H1:TEST', 0)
    lalframe.FrameUFrChanSetTimeOffset(channel, 0.25)
    lalframe.FrameUFrChanVectorSetStartX(channel, 0.03125)
    frame = lalframe.FrameUFrameHAlloc('TEST', 1000000000, 0.125, 16, 0)
    lalframe.FrameUFrameHFrChanAdd(frame, channel)
    path = tmp_path / 'shifted.gwf'
    output = lalframe.FrameUFrFileOpen(str(path), 'w')
    lalframe.FrameUFrameHWrite(output, frame)
    del output
    return path, lal.LIGOTimeGPS(1000000000, 406250000)


def _native_buffer(path, start, rate=16):
    """Use known metadata to isolate FrStream's sample/epoch calculation."""
    buffer = object.__new__(DataBuffer)
    buffer.frame_src = [str(path)]
    buffer.stream = lalframe.FrStreamCacheOpen(locations_to_cache(buffer.frame_src))
    buffer.channel_name = 'H1:TEST'
    buffer.channel_type = lal.D_TYPE_CODE
    buffer.raw_sample_rate = rate
    buffer.read_pos = start
    buffer.increment_update_cache = None
    return buffer


def _write_gapped_frames(tmp_path):
    original = tmp_path / 'original.gwf'
    _write_samples(original, np.float64, duration=4)
    source = lalframe.FrameUFrFileOpen(str(original), 'r')
    channel = lalframe.FrameUFrChanRead(source, 'H1:TEST', 0)
    path = tmp_path / 'gapped.gwf'
    output = lalframe.FrameUFrFileOpen(str(path), 'w')
    for start in (1000000000, 1000000006):
        frame = lalframe.FrameUFrameHAlloc('TEST', start, 0, 4, 0)
        lalframe.FrameUFrameHFrChanAdd(frame, channel)
        lalframe.FrameUFrameHWrite(output, frame)
    del output
    return path


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize('dtype', _CHANNEL_DTYPES)
def test_local_frame_metadata_matches_stream(tmp_path, dtype):
    path = tmp_path / 'H-test-1000000000-2.gwf'
    write_frame(
        str(path), 'H1:TEST',
        TimeSeries(np.ones(8, dtype=dtype), delta_t=0.25,
                   epoch=1000000000),
    )
    stream = lalframe.FrStreamCacheOpen(locations_to_cache([str(path)]))
    expected = DataBuffer._retrieve_metadata(stream, 'H1:TEST')
    actual = retrieve_frame_metadata_jax(stream, 'H1:TEST', [str(path)])
    assert actual == expected


def test_metadata_shortcut_excludes_non_direct_sources():
    assert retrieve_frame_metadata_jax(
        None, 'H1:TEST', ['/frames/H-*.gwf']
    ) is None


@pytest.mark.parametrize('dtype', _CHANNEL_DTYPES)
@pytest.mark.parametrize('device', _devices())
def test_typed_single_frame_replay_matches_stream_and_bounds_device_data(
        tmp_path, monkeypatch, dtype, device):
    path = tmp_path / 'H-test-1000000000-16.gwf'
    _write_samples(path, dtype)
    with scheme.CPUScheme():
        expected = DataBuffer([str(path)], 'H1:TEST', 1000000002,
                              max_buffer=4, dtype=dtype)
        blocks = [expected.advance(2) for _ in range(3)]
        expected_epoch = expected.stream.epoch.ns()
        expected_state = expected.stream.state
        expected_data = np.asarray(expected.raw_buffer).copy()
    with scheme.JAXScheme(device) as ctx:
        actual = DataBuffer([str(path)], 'H1:TEST', 1000000002,
                            max_buffer=4, dtype=dtype)
        configure_jax_replay(actual, end_time=1000000008, blocksize=2,
                             read_ahead_seconds=4)
        # Exercise the typed compatibility reader after native admission fails.
        def unsupported_native(*_args, **_kwargs):
            raise gwf_replay_jax.GWFReplayUnsupported('test compatibility path')

        monkeypatch.setattr(gwf_replay_jax, 'GWFReplaySource', unsupported_native)
        reader, native_dtype = frame_jax._FRAME_FILE_READERS[actual.channel_type]
        calls = []

        def typed_read(*args):
            calls.append(args[2])
            return reader(*args)

        def reject_stream(_duration):
            raise AssertionError('admitted replay used the FrStream reader')

        monkeypatch.setitem(frame_jax._FRAME_FILE_READERS, actual.channel_type,
                            (typed_read, native_dtype))
        monkeypatch.setattr(actual, '_read_frame', reject_stream)
        sizes = []
        for want in blocks:
            got = actual.attempt_advance(2)
            assert got.dtype == want.dtype and got.delta_t == want.delta_t
            assert got.start_time == want.start_time
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
            sizes.append(len(actual._jax_replay_reader._data))
            assert to_jax(got).devices() == {ctx.jax_device}
        assert calls == [0, 0]
        assert sizes == [64, 64, 32]
        assert np.asarray(actual.raw_buffer).tobytes() == expected_data.tobytes()
        assert actual.stream.epoch.ns() == expected_epoch
        assert actual.stream.state == expected_state


@pytest.mark.parametrize('offset_ns', [-1000, 0, 1, 6103, 6104, 31250000,
                                      62500000])
def test_typed_span_preserves_shifted_epochs_and_subsample_rounding(
        tmp_path, offset_ns):
    path, epoch = _shifted_channel_frame(tmp_path)
    start_ns = epoch.ns() + offset_ns
    start = lal.LIGOTimeGPS(start_ns // 1000000000, start_ns % 1000000000)
    expected = _native_buffer(path, start)
    actual = _native_buffer(path, start)
    with scheme.JAXScheme('cpu'):
        want = expected._read_frame(2)
        got = frame_jax._read_single_frame_span(actual, 2)
        assert got is not None
        assert got.start_time == want.start_time == float(start)
        assert got.delta_t == want.delta_t
        assert got.dtype == want.dtype
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        assert actual.stream.epoch.ns() == expected.stream.epoch.ns()
        assert (actual.stream.pos, actual.stream.state) == (
            expected.stream.pos, expected.stream.state)


def test_typed_span_preserves_stream_error_before_channel_epoch(tmp_path):
    path, epoch = _shifted_channel_frame(tmp_path)
    start_ns = epoch.ns() - 2000
    start = lal.LIGOTimeGPS(start_ns // 1000000000, start_ns % 1000000000)
    buffer = _native_buffer(path, start)
    assert frame_jax._read_single_frame_span(buffer, 2) is None
    with pytest.raises(RuntimeError):
        buffer._read_frame(2)


@pytest.mark.parametrize('offset', [0, 1, 2, 3, 7, 17, 1024])
def test_typed_span_preserves_native_absolute_gps_rounding(tmp_path, offset):
    path = tmp_path / 'fractional-grid.gwf'
    _write_samples(path, np.float64, duration=4, rate=4096)
    start = lal.LIGOTimeGPS(1000000000) + offset / 4096
    expected = _native_buffer(path, start, rate=4096)
    actual = _native_buffer(path, start, rate=4096)
    with scheme.JAXScheme('cpu'):
        want = expected._read_frame(1)
        got = frame_jax._read_single_frame_span(actual, 1)
        assert got is not None
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        assert got.start_time == want.start_time
        assert actual.stream.epoch.ns() == expected.stream.epoch.ns()
        assert (actual.stream.pos, actual.stream.state) == (
            expected.stream.pos, expected.stream.state)


@pytest.mark.parametrize('gapped,offset,duration', [
    (False, 14, 2), (False, 14, 4),
    (True, 2, 2), (True, 2, 4), (True, 4, 2),
])
def test_typed_boundary_fallback_preserves_stream_state(
        tmp_path, monkeypatch, gapped, offset, duration):
    path = tmp_path / 'single.gwf'
    if gapped:
        path = _write_gapped_frames(tmp_path)
    else:
        _write_samples(path, np.float64)
    start = 1000000000 + offset
    expected = _native_buffer(path, start)
    actual = _native_buffer(path, start)

    def reject_decode(*_args):
        raise AssertionError('obvious boundary span decoded the full vector')

    monkeypatch.setitem(frame_jax._FRAME_FILE_READERS, lal.D_TYPE_CODE,
                        (reject_decode, np.float64))
    calls = []
    original = actual._read_frame

    def stream_read(span):
        calls.append(span)
        return original(span)

    monkeypatch.setattr(actual, '_read_frame', stream_read)
    replay = JAXReplayFrameReader(start + duration, duration)
    with scheme.JAXScheme('cpu'):
        try:
            want = expected._read_frame(duration)
        except RuntimeError:
            with pytest.raises(JAXReplayReadError):
                replay._refill(actual)
        else:
            replay._refill(actual)
            assert np.asarray(replay._data).tobytes() == np.asarray(want).tobytes()
        assert calls == [duration]
        assert (actual.stream.epoch.ns(), actual.stream.pos,
                actual.stream.state, actual.stream.mode) == (
            expected.stream.epoch.ns(), expected.stream.pos,
            expected.stream.state, expected.stream.mode)


@pytest.mark.parametrize('source', ['glob', 'multiple', 'string', 'missing',
                                   'incremental'])
def test_typed_span_excludes_non_direct_sources(tmp_path, monkeypatch, source):
    path = tmp_path / 'single.gwf'
    _write_samples(path, np.float64)
    actual = _native_buffer(path, 1000000002)
    if source == 'glob':
        actual.frame_src = [str(tmp_path / '*.gwf')]
    elif source == 'multiple':
        actual.frame_src = [str(path), str(path)]
    elif source == 'string':
        actual.frame_src = str(path)
    elif source == 'missing':
        actual.frame_src = [str(tmp_path / 'missing.gwf')]
    else:
        actual.increment_update_cache = '/frames/*.gwf'

    def reject_decode(*_args):
        raise AssertionError('unsupported source decoded a typed vector')

    monkeypatch.setitem(frame_jax._FRAME_FILE_READERS, lal.D_TYPE_CODE,
                        (reject_decode, np.float64))
    with scheme.JAXScheme('cpu'):
        expected = actual._read_frame(2)
        replay = JAXReplayFrameReader(1000000004, 2)
        replay._refill(actual)
        assert np.asarray(replay._data).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize('invalid', ['rate', 'zero_dt', 'nan_dt', 'dtype',
                                    'short_vector', 'negative_offset',
                                    'reader_error'])
def test_typed_span_metadata_guards_use_stream_fallback(
        tmp_path, monkeypatch, invalid):
    path = tmp_path / 'single.gwf'
    _write_samples(path, np.float64)
    actual = _native_buffer(path, 1000000002)
    native_reader, native_dtype = frame_jax._FRAME_FILE_READERS[lal.D_TYPE_CODE]

    def invalid_native(*args):
        if invalid == 'reader_error':
            raise RuntimeError('typed vector unavailable')
        original = native_reader(*args)
        data = original.data.data
        epoch = original.epoch
        delta_t = original.deltaT
        if invalid == 'rate':
            delta_t *= 2
        elif invalid == 'zero_dt':
            delta_t = 0
        elif invalid == 'nan_dt':
            delta_t = np.nan
        elif invalid == 'dtype':
            data = data.astype(np.float32)
        elif invalid == 'short_vector':
            data = data[:1]
        else:
            epoch = lal.LIGOTimeGPS(1000000003)
        return types.SimpleNamespace(deltaT=delta_t, epoch=epoch,
                                     data=types.SimpleNamespace(data=data))

    monkeypatch.setitem(frame_jax._FRAME_FILE_READERS, lal.D_TYPE_CODE,
                        (invalid_native, native_dtype))
    expected = _native_buffer(path, actual.read_pos)
    with scheme.JAXScheme('cpu'):
        want = expected._read_frame(2)
        replay = JAXReplayFrameReader(1000000004, 2)
        replay._refill(actual)
        assert np.asarray(replay._data).tobytes() == np.asarray(want).tobytes()
        assert actual.stream.epoch.ns() == expected.stream.epoch.ns()


def test_typed_span_keeps_fractional_duration_and_unknown_type_on_stream(
        tmp_path, monkeypatch):
    path = tmp_path / 'single.gwf'
    _write_samples(path, np.float64)
    actual = _native_buffer(path, 1000000002)

    def reject_seek(*_args):
        raise AssertionError('ineligible input sought the frame stream')

    monkeypatch.setattr(lalframe, 'FrStreamSeek', reject_seek)
    for duration in (1.5, 0, -1, np.inf, np.nan):
        assert frame_jax._read_single_frame_span(actual, duration) is None
    actual.channel_type = -1
    assert frame_jax._read_single_frame_span(actual, 2) is None


def test_jax_buffer_uses_single_read_and_cpu_keeps_stream_path(
        tmp_path, monkeypatch):
    path = tmp_path / 'H-test-1000000000-2.gwf'
    write_frame(
        str(path), 'H1:TEST',
        TimeSeries(np.ones(8, dtype=np.float64), delta_t=0.25,
                   epoch=1000000000),
    )
    original = DataBuffer._retrieve_metadata
    calls = []

    def stream_metadata(stream, channel):
        calls.append(channel)
        return original(stream, channel)

    monkeypatch.setattr(DataBuffer, '_retrieve_metadata',
                        staticmethod(stream_metadata))
    with scheme.JAXScheme('cpu'):
        reader = DataBuffer([str(path)], 'H1:TEST', 1000000000,
                            max_buffer=1)
        assert reader.raw_sample_rate == 4
        assert calls == []
    with scheme.CPUScheme():
        def reject_typed_span(*_args):
            raise AssertionError('CPU advance used the JAX replay reader')

        monkeypatch.setattr(frame_jax, '_read_single_frame_span',
                            reject_typed_span)
        reader = DataBuffer([str(path)], 'H1:TEST', 1000000000,
                            max_buffer=1)
        assert reader.raw_sample_rate == 4
        assert calls == ['H1:TEST']
        np.testing.assert_array_equal(np.asarray(reader.advance(1)),
                                      np.ones(4))
    with scheme.JAXScheme('cpu'):
        reader = DataBuffer([str(tmp_path / 'H-*.gwf')], 'H1:TEST',
                            1000000000, max_buffer=1)
        assert reader.raw_sample_rate == 4
        assert calls == ['H1:TEST', 'H1:TEST']


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


def test_replay_prefers_direct_gwf_source(monkeypatch):
    calls = []

    class Source:
        def __init__(self, frame_src, channel_name, sample_rate):
            calls.append((frame_src, channel_name, sample_rate))

        def read(self, start, duration):
            calls.append((start, duration))
            return TimeSeries(
                np.arange(int(duration * 2), dtype=np.float32),
                delta_t=0.5,
                epoch=start,
            )

    monkeypatch.setattr(gwf_replay_jax, "GWFReplaySource", Source)
    with scheme.JAXScheme("cpu"):
        buffer = object.__new__(DataBuffer)
        buffer.frame_src = ["direct.gwf"]
        buffer.channel_name = "H1:STRAIN"
        buffer.raw_sample_rate = 2
        buffer.read_pos = 20.0
        buffer.force_update_cache = False
        buffer.increment_update_cache = None
        buffer.raw_buffer = TimeSeries(
            np.zeros(8, dtype=np.float32), delta_t=0.5, epoch=16.0
        )
        buffer._read_frame = types.MethodType(
            lambda self, duration: pytest.fail("compatibility read used"),
            buffer,
        )
        configure_jax_replay(
            buffer, end_time=24.0, blocksize=2.0,
            read_ahead_seconds=4.0,
        )

        buffer.attempt_advance(2.0)

    assert calls == [
        (["direct.gwf"], "H1:STRAIN", 2),
        (20.0, 4.0),
    ]


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
        current._read_frame = types.MethodType(
            lambda self, duration: None, current
        )
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


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("device", _devices())
def test_status_buffer_advances_with_configured_replay_reader(device, gap):
    calls = []
    cache_updates = []
    with scheme.JAXScheme(device) as ctx:
        buffer = object.__new__(StatusBuffer)
        buffer.channel_name = "H1:DQ"
        buffer.raw_sample_rate = 1
        buffer.read_pos = 10.0
        buffer.force_update_cache = False
        buffer.increment_update_cache = None
        buffer.valid_mask = 3
        buffer.valid_on_zero = False
        buffer.raw_buffer = TimeSeries(
            np.zeros(6, dtype=np.int32), delta_t=1.0, epoch=4.0
        )

        def read_frame(self, duration):
            calls.append((self.read_pos, duration))
            if gap and duration > 2:
                raise RuntimeError("read-ahead crosses a gap")
            return TimeSeries(
                np.full(int(duration), 3, dtype=np.int32),
                delta_t=1.0,
                epoch=self.read_pos,
            )

        buffer._read_frame = types.MethodType(read_frame, buffer)
        buffer.update_cache = lambda: cache_updates.append(buffer.read_pos)
        configure_jax_replay(
            buffer, end_time=14.0, blocksize=2.0,
            read_ahead_seconds=4.0,
        )

        assert buffer.advance(2.0)
        assert buffer.advance(2.0)
        assert calls == ([(10.0, 4.0), (10.0, 2.0), (12.0, 2.0)]
                         if gap else [(10.0, 4.0)])
        assert cache_updates == ([10.0] if gap else [])
        assert (buffer._jax_replay_reader is None) == gap
        assert to_jax(buffer.raw_buffer).devices() == {ctx.jax_device}
        np.testing.assert_array_equal(
            np.asarray(buffer.raw_buffer), np.array([0, 0, 3, 3, 3, 3])
        )


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


@pytest.mark.parametrize("device", _devices())
def test_replay_preserves_original_rolling_buffer_view_behavior(device):
    with scheme.JAXScheme(device):
        buffer = object.__new__(DataBuffer)
        buffer.read_pos, buffer.raw_sample_rate = 10.0, 2
        buffer.raw_buffer = TimeSeries(
            np.zeros(8, np.float32), delta_t=0.5, epoch=6)
        buffer._read_frame = types.MethodType(
            lambda self, duration: TimeSeries(
                np.arange(4, dtype=np.float32), delta_t=0.5,
                epoch=self.read_pos), buffer)
        view, storage = buffer.raw_buffer[4:], buffer.raw_buffer.data
        reader = JAXReplayFrameReader(12, 2)
        reader.advance(buffer, 2)
        assert buffer.raw_buffer.data is not storage
        np.testing.assert_array_equal(view.numpy(), 0)
        np.testing.assert_array_equal(buffer.raw_buffer.numpy()[4:], np.arange(4))


@pytest.mark.parametrize("device", _devices())
def test_replay_rejects_incomplete_final_block(device):
    with scheme.JAXScheme(device):
        buffer = object.__new__(DataBuffer)
        buffer.read_pos, buffer.raw_sample_rate = 0.0, 2
        buffer.raw_buffer = TimeSeries(
            np.zeros(8, np.float32), delta_t=0.5, epoch=-4)
        buffer._read_frame = types.MethodType(
            lambda self, duration: TimeSeries(
                np.ones(int(2 * duration), np.float32), delta_t=0.5,
                epoch=self.read_pos), buffer)
        reader = JAXReplayFrameReader(3, 2, 2)
        reader.advance(buffer, 2)
        before = buffer.raw_buffer.copy()
        with pytest.raises(JAXReplayReadError, match="complete block"):
            reader.advance(buffer, 2)
        assert buffer.read_pos == 2
        np.testing.assert_array_equal(buffer.raw_buffer.numpy(), before.numpy())


def test_cpu_frame_retry_ignores_jax_replay_mode(monkeypatch):
    monkeypatch.setenv("PYCBC_REPLAY_CLOCK", "1")
    attempts, sleeps = [], []
    with scheme.CPUScheme():
        buffer = object.__new__(DataBuffer)
        buffer.force_update_cache = False
        buffer.increment_update_cache = None
        buffer.raw_buffer = TimeSeries(np.zeros(8), delta_t=0.5, epoch=96)
        buffer._jax_replay_reader = types.SimpleNamespace(
            advance=lambda *a: pytest.fail("CPU used JAX read-ahead"))

        def advance(self, blocksize):
            attempts.append(blocksize)
            if len(attempts) == 1:
                raise RuntimeError("frame not available yet")
            return "original reader"

        monkeypatch.setattr(DataBuffer, "advance", advance)
        monkeypatch.setattr("pycbc.gps_now", lambda: 100)
        monkeypatch.setattr("pycbc.frame.frame.time.sleep", sleeps.append)
        assert buffer.attempt_advance(2, timeout=10) == "original reader"
        assert attempts == [2, 2]
        assert sleeps == [0.1]


@pytest.mark.parametrize("channel_dtype,buffer_dtype", [
    (np.float64, np.float32), (np.uint32, np.int32),
])
@pytest.mark.parametrize("device", _devices())
def test_replay_preserves_buffer_dtype_when_channel_differs(
        tmp_path, channel_dtype, buffer_dtype, device):
    path = tmp_path / 'H-test-1000000000-16.gwf'
    values = np.arange(256, dtype=channel_dtype)
    if channel_dtype == np.float64:
        values /= 7
    else:
        values += np.uint32(2**31)
    with scheme.CPUScheme():
        write_frame(str(path), 'H1:TEST', TimeSeries(
            values, delta_t=1/16, epoch=1000000000))
        native = DataBuffer([str(path)], 'H1:TEST', 1000000002,
                            max_buffer=4, dtype=buffer_dtype)
        expected = [native.advance(2) for _ in range(2)]
        expected_raw = native.raw_buffer.numpy().copy()
    with scheme.JAXScheme(device) as ctx:
        actual = DataBuffer([str(path)], 'H1:TEST', 1000000002,
                            max_buffer=4, dtype=buffer_dtype)
        configure_jax_replay(actual, end_time=1000000006, blocksize=2,
                             read_ahead_seconds=4)
        actual._read_frame = lambda *args: pytest.fail("direct replay fell back")
        for want in expected:
            got = actual.attempt_advance(2)
            assert got.dtype == want.dtype
            assert got.numpy().tobytes() == want.numpy().tobytes()
            assert actual.raw_buffer.dtype == np.dtype(buffer_dtype)
            assert to_jax(got).device == ctx.jax_device
        assert actual.raw_buffer.numpy().tobytes() == expected_raw.tobytes()
        assert not actual._jax_replay_reader._gwf_disabled


def test_cpu_status_advance_ignores_configured_jax_replay():
    with scheme.CPUScheme():
        buffer = object.__new__(StatusBuffer)
        buffer.increment_update_cache = None
        buffer.raw_sample_rate = 1
        buffer.read_pos = 10
        buffer.valid_mask = 3
        buffer.valid_on_zero = False
        buffer.raw_buffer = TimeSeries(
            np.zeros(4, np.int32), delta_t=1, epoch=6)
        calls = []

        def read_frame(self, duration):
            calls.append((self.read_pos, duration))
            return TimeSeries(np.full(duration, 3, np.int32),
                              delta_t=1, epoch=self.read_pos)

        buffer._read_frame = types.MethodType(read_frame, buffer)
        buffer._jax_replay_reader = types.SimpleNamespace(
            advance=lambda *args: pytest.fail("CPU used JAX status replay"))
        assert buffer.advance(2)
        assert calls == [(10, 2)]
        assert buffer.read_pos == 12
        np.testing.assert_array_equal(buffer.raw_buffer.numpy(), [0, 0, 3, 3])


def test_replay_honors_selected_device_for_cached_and_host_data():
    code = """
import tempfile, types
from pathlib import Path
import jax
import numpy as np
from pycbc import scheme
from pycbc.frame.frame import DataBuffer, write_frame
from pycbc.frame.frame_jax import configure_jax_replay
from pycbc.frame.gwf_replay_jax import GWFReplaySource
from pycbc.frame.gwf_deflate_jax import GWFDeflateUnavailable
from pycbc.types import TimeSeries
from pycbc.types.array_jax import to_jax
cpu0, cpu1 = jax.devices('cpu')
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'H-test-1000000000-16.gwf'
    values = np.arange(256, dtype=np.float64) / 7
    with scheme.CPUScheme():
        write_frame(str(path), 'H1:TEST', TimeSeries(
            values, delta_t=1/16, epoch=1000000000))
    with scheme.JAXScheme('0'):
        buffer = DataBuffer([str(path)], 'H1:TEST', 1000000002, max_buffer=4)
        configure_jax_replay(buffer, 1000000006, 2, read_ahead_seconds=4)
        buffer.attempt_advance(2)
        original = buffer._jax_replay_reader._data
    with scheme.JAXScheme('1'), jax.default_device(cpu0):
        block = buffer.attempt_advance(2)
        assert to_jax(block).device == to_jax(buffer.raw_buffer).device == cpu1
        assert original.device == cpu0
        assert block.numpy().tobytes() == values[64:96].tobytes()
        source = GWFReplaySource([str(path)], 'H1:TEST', 16, cuda_codec='host')
        result = source.read(1000000002, 2)
        assert to_jax(result).device == cpu1
        assert result.numpy().tobytes() == values[32:64].tobytes()
    source = GWFReplaySource([str(path)], 'H1:TEST', 16)
    attempts = []
    def codec(self, frame, vector):
        device = scheme.mgr.state.jax_device
        attempts.append(device)
        if device == cpu0:
            raise GWFDeflateUnavailable('device backend unavailable')
        return jax.device_put(values, device)
    source._cuda_values = types.MethodType(codec, source)
    for target in ['0', '0', '1']:
        with scheme.JAXScheme(target), jax.default_device(cpu0):
            result = source.read(1000000002, 2)
            assert to_jax(result).device == scheme.mgr.state.jax_device
            assert result.numpy().tobytes() == values[32:64].tobytes()
    assert attempts == [cpu0, cpu1]
    assert source._cuda_codec == 'deflate'
    assert source._cuda_unavailable_devices == {cpu0}
"""
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_replay_rejects_misnamed_frame_gps_span(tmp_path):
    path = tmp_path / 'H-test-1000000100-16.gwf'
    _write_samples(path, np.float64)
    with scheme.JAXScheme('cpu'):
        source = gwf_replay_jax.GWFReplaySource([str(path)], 'H1:TEST', 16)
        with pytest.raises(gwf_replay_jax.GWFReplayUnsupported, match='FrameH GPS'):
            source.read(1000000102, 2)
    with scheme.CPUScheme():
        with pytest.raises(RuntimeError):
            DataBuffer([str(path)], 'H1:TEST', 1000000102,
                       max_buffer=4)._read_frame(2)


def test_parent_channel_time_offset_uses_original_reader(tmp_path):
    original = tmp_path / 'original.gwf'
    _write_samples(original, np.float64)
    source = lalframe.FrameUFrFileOpen(str(original), 'r')
    channel = lalframe.FrameUFrChanRead(source, 'H1:TEST', 0)
    lalframe.FrameUFrChanSetTimeOffset(channel, 0.25)
    frame = lalframe.FrameUFrameHAlloc('TEST', 1000000000, 0, 16, 0)
    lalframe.FrameUFrameHFrChanAdd(frame, channel)
    path = tmp_path / 'H-shifted-1000000000-16.gwf'
    output = lalframe.FrameUFrFileOpen(str(path), 'w')
    lalframe.FrameUFrameHWrite(output, frame)
    del output
    with scheme.CPUScheme():
        native = DataBuffer([str(path)], 'H1:TEST', 1000000002, max_buffer=4)
        expected = [native.advance(2) for _ in range(2)]
    with scheme.JAXScheme('cpu'):
        source = gwf_replay_jax.GWFReplaySource([str(path)], 'H1:TEST', 16)
        with pytest.raises(gwf_replay_jax.GWFReplayUnsupported, match='time offset'):
            source.read(1000000002, 2)
        actual = DataBuffer([str(path)], 'H1:TEST', 1000000002, max_buffer=4)
        configure_jax_replay(actual, 1000000006, 2, read_ahead_seconds=4)
        for want in expected:
            got = actual.attempt_advance(2)
            assert got.numpy().tobytes() == want.numpy().tobytes()
            assert got.start_time == want.start_time
        assert actual._jax_replay_reader._gwf_disabled


def test_gzip_vector_keeps_cuda_preference_for_later_zlib(tmp_path, monkeypatch):
    from test_gwf_jax import _frvect, _replay_container

    first = np.arange(8, dtype=np.float32)
    second = first + 8
    compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    gzip = compressor.compress(first.tobytes()) + compressor.flush()
    paths = []
    for start, payload in [(200, gzip), (202, zlib.compress(second.tobytes()))]:
        vector = _frvect(8, 'little', payload, 3, compression=257,
                         shape=(8,), n_data=8, spacing=0.25, origin=0)
        path = tmp_path / ('H-test-%s-2.gwf' % start)
        path.write_bytes(_replay_container(8, 'little', vector, start=start))
        paths.append(str(path))
    cuda_calls, host_calls = [], []
    native_host = gwf_replay_jax._host_decompress

    def host(vector):
        host_calls.append(True)
        assert vector.payload[:2] == b'\x1f\x8b'
        return native_host(vector)

    with scheme.JAXScheme('cpu'):
        source = gwf_replay_jax.GWFReplaySource(paths, 'H1:TEST', 4)

        def codec(frame, vector):
            assert vector.payload[:2] != b'\x1f\x8b'
            cuda_calls.append(frame.start)
            return jax.device_put(second, scheme.mgr.state.jax_device)

        monkeypatch.setattr(source, '_cuda_values', codec)
        monkeypatch.setattr(gwf_replay_jax, '_host_decompress', host)
        actual = source.read(200, 4)
        assert actual.numpy().tobytes() == np.concatenate((first, second)).tobytes()
        assert source._cuda_codec == 'deflate'
        assert source._cuda_unavailable_devices == set()
    assert cuda_calls == [202]
    assert len(host_calls) == 1
