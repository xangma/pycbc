# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX data staging for bounded historical frame replays.

Local GWF inputs first use the narrow clean-room source in
``gwf_replay_jax.py``.  Its raw and zero-suppressed codecs run with JAX, while
RFC 1950 streams use installed ``cuda-zlib`` on CUDA, with bounded host
decoding followed by one transfer when the optional backend is unavailable.
RFC 1952 streams also use host decoding. The separate ``gwf_deflate_jax.py``
module adapts the external codec to JAX.
Unsupported layouts and codecs retain the existing LALFrame compatibility
reader.  Read-ahead then performs block extraction and rolling-buffer updates
on the selected JAX device.

The normal streaming reader remains authoritative.  This path is configured
only for an explicitly bounded ``--replay-clock`` run and falls back to the
streaming reader if a contiguous read-ahead span is unavailable.
"""

import functools
import logging
import math
import os

import jax
from jax import lax
import jax.numpy as jnp
import lal
import lalframe
import numpy as np

from pycbc.benchmark import stage_event
from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData, to_jax


DEFAULT_READ_AHEAD_SECONDS = 1024


logger = logging.getLogger(__name__)


_FRAME_VECTOR_TYPES = {
    lalframe.FRAMEU_FR_VECT_4R: lal.S_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_8R: lal.D_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_8C: lal.C_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_16C: lal.Z_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_4U: lal.U4_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_4S: lal.I4_TYPE_CODE,
}


_FRAME_FILE_READERS = {
    lal.S_TYPE_CODE: (lalframe.FrFileReadREAL4TimeSeries, np.float32),
    lal.D_TYPE_CODE: (lalframe.FrFileReadREAL8TimeSeries, np.float64),
    lal.C_TYPE_CODE: (lalframe.FrFileReadCOMPLEX8TimeSeries, np.complex64),
    lal.Z_TYPE_CODE: (lalframe.FrFileReadCOMPLEX16TimeSeries, np.complex128),
    lal.U4_TYPE_CODE: (lalframe.FrFileReadUINT4TimeSeries, np.uint32),
    lal.I4_TYPE_CODE: (lalframe.FrFileReadINT4TimeSeries, np.int32),
}


def _single_local_frame(frame_src):
    return (isinstance(frame_src, (list, tuple)) and len(frame_src) == 1
            and isinstance(frame_src[0], str)
            and frame_src[0].endswith('.gwf')
            and os.path.isfile(frame_src[0]))


def retrieve_frame_metadata_jax(stream, channel_name, frame_src):
    """Get type and rate with one channel read for a single local GWF.

    The generic stream path makes three full channel queries.  Restrict this
    shortcut to an unambiguous local frame; caches, globs and changing live
    sources retain the normal stream metadata lookup.
    """
    if not _single_local_frame(frame_src):
        return None
    try:
        frame = lalframe.FrameUFrFileOpen(frame_src[0], 'r')
        channel = lalframe.FrameUFrChanRead(
            frame, channel_name, stream.pos
        )
        if lalframe.FrameUFrChanVectorQueryNDim(channel) != 1:
            return None
        channel_type = _FRAME_VECTOR_TYPES.get(
            lalframe.FrameUFrChanVectorQueryType(channel)
        )
        delta_t = lalframe.FrameUFrChanVectorQueryDx(channel, 0)
        if channel_type is None or not math.isfinite(delta_t) or delta_t <= 0:
            return None
        return channel_type, int(1.0 / delta_t)
    except (AttributeError, RuntimeError, ValueError):
        return None


def _read_single_frame_span(buffer, duration):
    """Read an interior span once; leave boundary handling to FrStream."""
    if (not _single_local_frame(getattr(buffer, 'frame_src', None))
            or getattr(buffer, 'increment_update_cache', None)):
        return None
    try:
        reader = _FRAME_FILE_READERS.get(buffer.channel_type)
        if (reader is None or not math.isfinite(duration) or duration <= 0
                or int(duration) != duration):
            return None
        stream = buffer.stream
        requested = lal.LIGOTimeGPS(buffer.read_pos)
        if lalframe.FrStreamSeek(stream, requested) != 0:
            return None
        if stream.state & (lalframe.FR_STREAM_ERR | lalframe.FR_STREAM_END
                           | lalframe.FR_STREAM_GAP):
            return None
        frame_start = lalframe.FrFileQueryGTime(
            lal.LIGOTimeGPS(), stream.file, stream.pos)
        frame_duration = lalframe.FrFileQueryDt(stream.file, stream.pos)
        if not math.isfinite(frame_duration) or frame_duration <= 0:
            return None
        frame_end = frame_start + frame_duration
        if requested + duration >= frame_end:
            return None
        native = reader[0](stream.file, buffer.channel_name, stream.pos)
        delta_t = native.deltaT
        if (not math.isfinite(delta_t) or delta_t <= 0
                or 1.0 / delta_t != buffer.raw_sample_rate):
            return None
        # FrStream truncates duration/deltaT after DataBuffer's int(duration).
        count = int(int(duration) / delta_t)
        if count != int(round(duration * buffer.raw_sample_rate)) or count <= 0:
            return None
        if requested.ns() + 1000 < native.epoch.ns():
            return None
        offset = math.ceil(((requested.ns() - native.epoch.ns()) * 1e-9
                            - 0.1 / 16384) / delta_t)
        values = np.asarray(native.data.data)
        if (values.ndim != 1 or values.dtype != reader[1] or offset < 0
                or offset + count > len(values)):
            return None
        # FrStream adds the rounded offset to GPS ns in double precision.
        sample_ns = int(float(native.epoch.ns()) + math.floor(
            1e9 * offset * delta_t + 0.5))
        sample_epoch = lal.LIGOTimeGPS(sample_ns // 1000000000,
                                       sample_ns % 1000000000)
        sample_end = sample_epoch + count * delta_t
        if sample_end >= frame_end:
            return None
        selected = values[offset:offset + count].copy()
        series = TimeSeries(selected, delta_t=delta_t,
                            epoch=buffer.read_pos, dtype=reader[1])
        stream.epoch = sample_end
        return series
    except (AttributeError, RuntimeError, ValueError, TypeError, OverflowError):
        return None


class JAXReplayReadError(RuntimeError):
    """A contiguous read-ahead span could not serve the next replay block."""


@functools.partial(jax.jit, static_argnames=("block_samples",))
def _advance_replay_buffer(raw_buffer, replay_data, offset, block_samples):
    """Extract one replay block and append it to a rolling raw buffer."""
    block = lax.dynamic_slice_in_dim(
        replay_data, offset, block_samples, axis=0
    )
    updated = jnp.concatenate((raw_buffer[block_samples:],
                               block.astype(raw_buffer.dtype)))
    return updated, block


class JAXReplayFrameReader:
    """Bounded, device-resident read-ahead for one frame channel."""

    def __init__(self, raw_end_time, blocksize,
                 read_ahead_seconds=DEFAULT_READ_AHEAD_SECONDS):
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        if read_ahead_seconds < blocksize:
            read_ahead_seconds = blocksize
        self.raw_end_time = float(raw_end_time)
        self.blocksize = float(blocksize)
        self.read_ahead_seconds = float(read_ahead_seconds)
        self._data = None
        self._offset = 0
        self._gwf_source = None
        self._gwf_disabled = False

    def _read_ahead_duration(self, read_pos):
        remaining = self.raw_end_time - float(read_pos)
        if remaining <= 0:
            return 0.0
        duration = min(remaining, self.read_ahead_seconds)
        blocks = max(1, int(math.floor(
            (duration + 1e-9) / self.blocksize
        )))
        return min(remaining, blocks * self.blocksize)

    def _refill(self, buffer):
        duration = self._read_ahead_duration(buffer.read_pos)
        if duration <= 0:
            raise JAXReplayReadError("JAX replay read-ahead is exhausted")
        series = None
        if not self._gwf_disabled and hasattr(buffer, "frame_src"):
            from pycbc.frame.gwf_replay_jax import (
                GWFReplaySource,
                GWFReplayUnsupported,
            )
            try:
                if self._gwf_source is None:
                    self._gwf_source = GWFReplaySource(
                        buffer.frame_src,
                        buffer.channel_name,
                        buffer.raw_sample_rate,
                    )
                series = self._gwf_source.read(buffer.read_pos, duration)
            except GWFReplayUnsupported as exc:
                logger.info(
                    "Direct JAX GWF replay unavailable for %s; using the "
                    "compatibility reader: %s", buffer.channel_name, exc
                )
                self._gwf_disabled = True
                self._gwf_source = None
        if series is None:
            try:
                series = _read_single_frame_span(buffer, duration)
                if series is None:
                    series = buffer._read_frame(duration)
            except RuntimeError as exc:
                raise JAXReplayReadError(str(exc)) from exc
        expected = int(round(duration * buffer.raw_sample_rate))
        if len(series) != expected:
            raise JAXReplayReadError(
                "frame read-ahead returned %d samples, expected %d"
                % (len(series), expected)
            )
        self._data = to_jax(series)
        self._offset = 0

    def advance(self, buffer, blocksize):
        """Advance ``buffer`` without returning samples to the host."""
        if abs(float(blocksize) - self.blocksize) > 1e-9:
            raise ValueError("JAX replay blocksize changed during a run")
        block_samples = int(round(blocksize * buffer.raw_sample_rate))
        needs_refill = (
            self._data is None
            or self._offset + block_samples > len(self._data)
        )
        if needs_refill:
            self._refill(buffer)
        if self._offset + block_samples > len(self._data):
            raise JAXReplayReadError("JAX replay has no complete block left")
        self._data = to_jax(self._data)

        stage_event("gwf_replay_buffer", "start")
        try:
            updated, block = _advance_replay_buffer(
                to_jax(buffer.raw_buffer),
                self._data,
                to_jax(np.int32(self._offset)),
                block_samples,
            )
        finally:
            stage_event("gwf_replay_buffer", "end")
        buffer.raw_buffer._data = JAXArrayData(updated)
        buffer.raw_buffer._saved.clear()

        result = TimeSeries(
            JAXArrayData(block),
            delta_t=buffer.raw_buffer.delta_t,
            epoch=buffer.read_pos,
            copy=False,
        )
        self._offset += block_samples
        buffer.read_pos += blocksize
        buffer.raw_buffer.start_time += blocksize
        return result


def advance_status_replay_jax(buffer, blocksize):
    """Advance an explicitly configured status replay with gap fallback."""
    try:
        return buffer._jax_replay_reader.advance(buffer, blocksize)
    except JAXReplayReadError as exc:
        from pycbc.frame.frame import DataBuffer

        logger.warning(
            "JAX replay read-ahead unavailable for %s; falling back "
            "to frame reads: %s", buffer.channel_name, exc
        )
        buffer._jax_replay_reader = None
        buffer.update_cache()
        return DataBuffer.advance(buffer, blocksize)


def _data_buffers(root):
    """Yield a strain reader and any nested state/data-quality readers."""
    pending = [root]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if hasattr(current, "raw_buffer") and hasattr(current, "_read_frame"):
            yield current
        for name in ("state", "dq", "idq", "idq_state"):
            child = getattr(current, name, None)
            if child is not None:
                pending.append(child)


def configure_jax_replay(buffer, end_time, blocksize,
                         read_ahead_seconds=DEFAULT_READ_AHEAD_SECONDS):
    """Configure bounded JAX read-ahead for a historical replay.

    The raw input must cover every analysis advance, including the initial
    conditioning corruption preceding ``buffer.end_time``.  Each nested data
    buffer receives the same raw end point so state and strain stay in
    lockstep.
    Buffers that discover frame paths incrementally retain their existing
    streaming implementation.
    """
    if not math.isfinite(float(end_time)):
        return buffer
    current_end = (
        buffer.end_time
        if hasattr(type(buffer), "end_time")
        else buffer.raw_buffer.end_time
    )
    remaining = max(0.0, float(end_time) - float(current_end))
    blocks = int(math.ceil((remaining - 1e-9) / float(blocksize)))
    raw_end_time = float(buffer.read_pos) + blocks * float(blocksize)
    for current in _data_buffers(buffer):
        if getattr(current, "increment_update_cache", None):
            continue
        current._jax_replay_reader = JAXReplayFrameReader(
            raw_end_time,
            blocksize,
            read_ahead_seconds=read_ahead_seconds,
        )
    return buffer
