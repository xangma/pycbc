# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX data staging for bounded historical frame replays.

GWF decompression is a host operation provided by LALFrame.  Re-reading a
compressed frame for every analysis chunk needlessly repeats that operation.
This module amortizes the unavoidable host work over a bounded replay span,
transfers the decoded samples once, and then performs block extraction and
rolling-buffer updates on the selected JAX device.

The normal streaming reader remains authoritative.  This path is configured
only for an explicitly bounded ``--replay-clock`` run and falls back to the
streaming reader if a contiguous read-ahead span is unavailable.
"""

import functools
import math
import os

import jax
from jax import lax
import jax.numpy as jnp
import lal
import lalframe

from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData, to_jax


DEFAULT_READ_AHEAD_SECONDS = 1024


_FRAME_VECTOR_TYPES = {
    lalframe.FRAMEU_FR_VECT_4R: lal.S_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_8R: lal.D_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_8C: lal.C_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_16C: lal.Z_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_4U: lal.U4_TYPE_CODE,
    lalframe.FRAMEU_FR_VECT_4S: lal.I4_TYPE_CODE,
}


def retrieve_frame_metadata_jax(stream, channel_name, frame_src):
    """Get type and rate with one channel read for a single local GWF.

    The generic stream path makes three full channel queries.  Restrict this
    shortcut to an unambiguous local frame; caches, globs and changing live
    sources retain the normal stream metadata lookup.
    """
    if (not isinstance(frame_src, (list, tuple)) or len(frame_src) != 1
            or not isinstance(frame_src[0], str)
            or not frame_src[0].endswith('.gwf')
            or not os.path.isfile(frame_src[0])):
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


class JAXReplayReadError(RuntimeError):
    """A contiguous read-ahead span could not serve the next replay block."""


@functools.partial(jax.jit, static_argnames=("block_samples",))
def _advance_replay_buffer(raw_buffer, replay_data, offset, block_samples):
    """Extract one replay block and append it to a rolling raw buffer."""
    block = lax.dynamic_slice_in_dim(
        replay_data, offset, block_samples, axis=0
    )
    updated = jnp.concatenate((raw_buffer[block_samples:], block))
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
        try:
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
        """Advance ``buffer`` by one block without returning samples to host."""
        if abs(float(blocksize) - self.blocksize) > 1e-9:
            raise ValueError("JAX replay blocksize changed during a run")
        block_samples = int(round(blocksize * buffer.raw_sample_rate))
        if self._data is None or self._offset + block_samples > len(self._data):
            self._refill(buffer)

        updated, block = _advance_replay_buffer(
            to_jax(buffer.raw_buffer),
            self._data,
            jnp.asarray(self._offset, dtype=jnp.int32),
            block_samples,
        )
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
    buffer receives the same raw end point so state and strain stay in lockstep.
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
