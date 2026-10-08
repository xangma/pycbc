# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Narrow clean-room GWF source for bounded JAX replay.

Container and native codec details live in :mod:`pycbc.frame.gwf_jax`.  This
integration layer selects a channel and time interval from ordinary local GWF
files.  Layouts it cannot prove safe are rejected so the caller can retain the
normal LALFrame reader as an explicit compatibility fallback.

RFC 1950 streams use the optional cuda-zlib package on CUDA. Unavailable
backends and RFC 1952 streams use the bounded standard-library host decoder.
Raw and zero-suppressed vectors use JAX.
No LAL/LALFrame implementation detail is used here.
"""

from dataclasses import dataclass
import functools
import glob
import logging
import math
import os
from pathlib import Path
import re
import time
import zlib

import numpy as np

from pycbc import scheme
from pycbc.benchmark import stage_event
from pycbc.frame.gwf_jax import (
    COMMON_HEADER_SIZE,
    CompressionKind,
    GWFFormatError,
    UnsupportedGWFChecksum,
    UnsupportedGWFFormat,
    _Cursor,
    decode_fr_vect,
    parse_gwf,
)
from pycbc.types import TimeSeries
from pycbc.types.array import _ALLOWED_DTYPES
from pycbc.types.array_jax import JAXArrayData


__all__ = ("GWFReplaySource", "GWFReplayUnsupported")

logger = logging.getLogger(__name__)


_FRAME_NAME = re.compile(r"-(?P<start>[0-9]+)-(?P<duration>[0-9]+)\.gwf$")


class GWFReplayUnsupported(RuntimeError):
    """The source needs the existing compatibility frame reader."""


@dataclass(frozen=True)
class _FrameFile:
    path: str
    start: int
    duration: int

    @property
    def end(self):
        return self.start + self.duration


@dataclass(frozen=True)
class _ChannelVector:
    name: str
    time_offset: float
    sample_rate: float | None
    time_series: bool
    vector: object


@dataclass(frozen=True)
class _ReplayFrame:
    seconds: int
    nanoseconds: int
    duration: float
    channels: tuple[_ChannelVector, ...]


def _replay_metadata(data, container):
    """Admit one FrameH and its linked ADC/processed channel vectors.

    Field order and pointers follow the public GWF v8 tables 9/10/17/18 and
    v9 tables 10/11/18/19. No channel or epoch is inferred from a vector name.
    """
    names = dict(container.structure_names)
    headers = [item for item in container.structures
               if names.get(item.class_id) == "FrameH"]
    if len(headers) != 1:
        raise GWFReplayUnsupported("direct replay requires one FrameH per file")
    structures = {}
    for item in container.structures:
        if item.class_id in (1, 2):  # FrSH/FrSE dictionaries are not targets.
            continue
        key = (item.class_id, item.instance)
        if key in structures:
            raise GWFReplayUnsupported("ambiguous frame structure pointers")
        structures[key] = item
    vectors = {(item.structure.class_id, item.structure.instance): item
               for item in container.vectors}

    def cursor(item):
        return _Cursor(data, item.offset + COMMON_HEADER_SIZE,
                       item.offset + item.length - 4,
                       container.header.struct_prefix)

    def pointer(reader):
        return (reader.uint(2, "pointer class"),
                reader.uint(4, "pointer instance"))

    def resolve(reference, expected):
        item = structures.get(reference)
        if item is None or names.get(item.class_id) != expected:
            raise GWFReplayUnsupported("unresolved %s frame pointer" % expected)
        return item

    def finish(reader, name):
        if reader.pos != reader.end:
            raise GWFReplayUnsupported("unrecognized %s metadata layout" % name)

    reader = cursor(headers[0])
    reader.string("frame name")
    reader.take(12, "frame run, number and quality")
    seconds = reader.uint(4, "GPS seconds")
    nanoseconds = reader.uint(4, "GPS nanoseconds")
    if nanoseconds >= 1000000000:
        raise GWFFormatError("invalid FrameH GPS nanoseconds")
    if container.header.version == 8:
        reader.uint(2, "leap seconds")
    duration = reader.float64("frame duration")
    if not math.isfinite(duration) or duration <= 0:
        raise GWFFormatError("invalid FrameH duration")
    links = [pointer(reader) for _ in range(13)]
    finish(reader, "FrameH")
    first_adc = (0, 0)
    if links[5] != (0, 0):
        reader = cursor(resolve(links[5], "FrRawData"))
        reader.string("raw data name")
        pointer(reader)  # firstSer
        first_adc = pointer(reader)
        for _ in range(3):
            pointer(reader)  # firstTable, logMsg, more
        finish(reader, "FrRawData")

    channels = []
    for first, kind in [(first_adc, "FrAdcData"), (links[6], "FrProcData")]:
        seen = set()
        reference = first
        while reference != (0, 0):
            if reference in seen:
                raise GWFReplayUnsupported("cyclic channel pointer list")
            seen.add(reference)
            reader = cursor(resolve(reference, kind))
            name = reader.string("channel name")
            # Comments are opaque and may have an empty encoded payload.
            reader.take(reader.uint(2, "channel comment length"),
                        "channel comment")
            sample_rate = None
            time_series = True
            if kind == "FrAdcData":
                reader.take(20, "ADC identifiers and calibration")
                reader.string("ADC units")
                sample_rate = reader.float64("ADC sample rate")
                time_offset = reader.float64("channel time offset")
                reader.take(12, "ADC frequency shift and phase")
                time_series = reader.uint(2, "ADC validity") == 0
                data_pointer = pointer(reader)
                pointer(reader)  # aux
            else:
                time_series = reader.uint(2, "processed data type") == 1
                reader.uint(2, "processed data subtype")
                time_offset = reader.float64("channel time offset")
                reader.take(36, "processed range, shift, phase and bandwidth")
                auxiliary = reader.uint(2, "auxiliary parameter count")
                reader.take(8 * auxiliary, "auxiliary parameters")
                for _ in range(auxiliary):
                    reader.string("auxiliary parameter name")
                data_pointer = pointer(reader)
                for _ in range(3):
                    pointer(reader)  # aux, table, history
            reference = pointer(reader)
            finish(reader, kind)
            resolve(data_pointer, "FrVect")
            vector = vectors.get(data_pointer)
            if vector is None:
                raise GWFReplayUnsupported("unresolved channel data vector")
            channels.append(_ChannelVector(name, time_offset, sample_rate,
                                           time_series, vector))
    return _ReplayFrame(seconds, nanoseconds, duration, tuple(channels))


def _timed_stage(name, function, **metadata):
    started = time.perf_counter()
    stage_event(name, "start", **metadata)
    try:
        return function()
    finally:
        stage_event(
            name,
            "end",
            elapsed_seconds=time.perf_counter() - started,
            **metadata,
        )


@functools.lru_cache(maxsize=2)
def _parse_cached(path, size, mtime_ns):
    del size, mtime_ns

    def parse():
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise GWFReplayUnsupported(
                "cannot read local GWF file %s" % path
            ) from exc
        try:
            return _replay_metadata(data, parse_gwf(data))
        except (UnsupportedGWFChecksum, UnsupportedGWFFormat) as exc:
            raise GWFReplayUnsupported(
                "unsupported GWF format or checksum uses "
                "the compatibility reader"
            ) from exc

    return _timed_stage("gwf_parse", parse, path=path)


def _frame_file(path):
    match = _FRAME_NAME.search(os.path.basename(path))
    if match is None:
        raise GWFReplayUnsupported(
            "GWF filename does not declare its GPS span: %s" % path
        )
    return _FrameFile(
        os.path.abspath(path),
        int(match.group("start")),
        int(match.group("duration")),
    )


def _resolve_files(frame_src):
    if isinstance(frame_src, (str, os.PathLike)):
        frame_src = [frame_src]
    files = []
    for source in frame_src:
        source = os.fspath(source)
        if source.endswith((".cache", ".lcf")):
            raise GWFReplayUnsupported("frame cache files are not direct GWFs")
        matches = glob.glob(source)
        if not matches:
            raise GWFReplayUnsupported("no local GWF files match %s" % source)
        for match in matches:
            if not match.endswith(".gwf"):
                raise GWFReplayUnsupported(
                    "non-GWF replay source %s" % match
                )
            files.append(_frame_file(match))
    files = sorted(set(files), key=lambda item: (item.start, item.path))
    if not files:
        raise GWFReplayUnsupported("no direct GWF replay files")
    return tuple(files)


def _sample_index(seconds, sample_rate, label):
    samples = float(seconds) * sample_rate
    index = int(round(samples))
    if not math.isclose(samples, index, rel_tol=0.0, abs_tol=1e-6):
        raise GWFReplayUnsupported(
            "%s is not aligned to the channel sample grid" % label
        )
    return index


def _gzip_wrapper(payload):
    if len(payload) >= 2 and payload[:2] == b"\x1f\x8b":
        return 16 + zlib.MAX_WBITS
    if len(payload) >= 2:
        cmf, flg = payload[:2]
        if cmf & 0x0F == 8 and (cmf << 8 | flg) % 31 == 0:
            return zlib.MAX_WBITS
    raise GWFReplayUnsupported(
        "GWF gzip payload is neither an RFC 1950 nor RFC 1952 stream"
    )


def _host_decompress(vector):
    dtype = np.dtype(vector.dtype)
    expected = vector.n_data * dtype.itemsize
    wrapper = _gzip_wrapper(vector.payload)
    decoder = zlib.decompressobj(wrapper)
    try:
        pending = vector.payload
        chunks = []
        total = 0
        while pending:
            chunk = decoder.decompress(pending, expected + 1 - total)
            chunks.append(chunk)
            total += len(chunk)
            if total > expected:
                raise GWFFormatError(
                    "compressed FrVect expands beyond its declared size"
                )
            next_pending = decoder.unconsumed_tail
            stalled = (
                next_pending
                and len(next_pending) == len(pending)
                and not chunk
            )
            if stalled:
                raise GWFFormatError("stalled compressed FrVect decoder")
            pending = next_pending
        chunk = decoder.flush(expected + 1 - total)
        chunks.append(chunk)
        raw = b"".join(chunks)
    except zlib.error as exc:
        raise GWFFormatError("invalid compressed FrVect payload") from exc
    if not decoder.eof:
        raise GWFFormatError("truncated compressed FrVect payload")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise GWFFormatError("trailing compressed FrVect payload data")
    if len(raw) != expected:
        raise GWFFormatError(
            "compressed FrVect expands to %d bytes; expected %d"
            % (len(raw), expected)
        )
    wire_dtype = dtype.newbyteorder(
        "<" if vector.compression.endian == "little" else ">"
    )
    values = np.frombuffer(raw, dtype=wire_dtype, count=vector.n_data)
    return values.astype(dtype.newbyteorder("="), copy=False)


def _sync_requested():
    return os.environ.get("PYCBC_BENCHMARK_GWF_SYNC") == "1"


def _device_put(values, path, channel):
    import jax

    def transfer():
        result = jax.device_put(
            values, getattr(scheme.mgr.state, "jax_device", None))
        if _sync_requested():
            result.block_until_ready()
        return result

    return _timed_stage(
        "gwf_h2d", transfer, path=path, channel=channel,
        bytes=int(values.nbytes), synchronized=_sync_requested()
    )


class GWFReplaySource:
    """Read exact sample spans from a conservative subset of local GWFs."""

    def __init__(self, frame_src, channel_name, sample_rate, *,
                 cuda_codec=None):
        self.files = _resolve_files(frame_src)
        self.channel_name = str(channel_name)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._host_cache_key = None
        self._host_cache = None
        self._cuda_codec = (
            os.environ.get("PYCBC_GWF_CUDA_DECOMPRESS", "deflate")
            if cuda_codec is None else cuda_codec
        )
        if self._cuda_codec not in ("host", "deflate"):
            raise ValueError("GWF CUDA codec must be host or deflate")
        self._cuda_cache_vector = None
        self._cuda_cache_device = None
        self._cuda_cache = None
        self._cuda_unavailable_devices = set()

    def _container(self, frame):
        try:
            stat = os.stat(frame.path)
        except OSError as exc:
            raise GWFReplayUnsupported(
                "cannot stat local GWF file %s" % frame.path
            ) from exc
        return _parse_cached(frame.path, stat.st_size, stat.st_mtime_ns)

    def _vector(self, frame):
        container = self._container(frame)
        if (container.seconds != frame.start or container.nanoseconds
                or container.duration != frame.duration):
            raise GWFReplayUnsupported(
                "FrameH GPS span does not match the filename")
        channels = [item for item in container.channels
                    if item.name == self.channel_name]
        if len(channels) != 1:
            raise GWFReplayUnsupported(
                "expected one %s channel in %s; found %d"
                % (self.channel_name, frame.path, len(channels))
            )
        channel = channels[0]
        if not channel.time_series or channel.time_offset != 0:
            raise GWFReplayUnsupported(
                "channel type, validity or time offset needs compatibility metadata")
        if (channel.sample_rate is not None
                and channel.sample_rate != self.sample_rate):
            raise GWFReplayUnsupported("ADC sample rate differs from stream metadata")
        vector = channel.vector
        if vector.next_class or vector.next_instance:
            raise GWFReplayUnsupported("linked channel vectors need compatibility metadata")
        if vector.type_code == 8:
            raise GWFReplayUnsupported(
                "string FrVect data use the compatibility reader"
            )
        if np.dtype(vector.dtype) not in _ALLOWED_DTYPES:
            raise GWFReplayUnsupported(
                "FrVect dtype %s is unsupported by TimeSeries" % vector.dtype
            )
        if vector.n_data_valid or vector.data_valid:
            raise GWFReplayUnsupported(
                "FrVect validity metadata use the compatibility reader"
            )
        if len(vector.shape) != 1 or len(vector.spacing) != 1:
            raise GWFReplayUnsupported("only one-dimensional FrVect data")
        spacing = vector.spacing[0]
        if not math.isfinite(spacing) or spacing <= 0:
            raise GWFFormatError("invalid FrVect sample spacing")
        if not math.isclose(
            spacing * self.sample_rate, 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise GWFReplayUnsupported(
                "FrVect sample spacing does not match stream metadata"
            )
        if not math.isclose(
            vector.origins[0], 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise GWFReplayUnsupported(
                "nonzero FrVect origin needs parent topology metadata"
            )
        expected = frame.duration * self.sample_rate
        if vector.n_data != expected:
            raise GWFReplayUnsupported(
                "FrVect length does not match the filename GPS span"
            )
        return vector

    def _host_values(self, frame, vector):
        if self._host_cache_key is not vector:
            self._host_cache = _timed_stage(
                "gwf_host_decompress",
                lambda: _host_decompress(vector),
                path=frame.path,
                channel=self.channel_name,
                compressed_bytes=len(vector.payload),
            )
            self._host_cache_key = vector
        return self._host_cache

    def _cuda_values(self, frame, vector):
        from pycbc.frame.gwf_deflate_jax import decode_fr_vect_deflate

        device = getattr(scheme.mgr.state, "jax_device", None)
        if (self._cuda_cache_vector is not vector
                or self._cuda_cache_device != device):
            values = _timed_stage(
                "gwf_cuda_decompress",
                lambda: decode_fr_vect_deflate(vector, device=device),
                path=frame.path, channel=self.channel_name,
                compressed_bytes=len(vector.payload), synchronized=False,
                codec_validation="eager", conversion="asynchronous",
            )
            self._cuda_cache_vector = vector
            self._cuda_cache_device = device
            self._cuda_cache = values
        return self._cuda_cache

    def _segment(self, frame, start, end):
        vector = self._vector(frame)
        first = _sample_index(
            start - frame.start, self.sample_rate, "GWF segment start"
        )
        last = _sample_index(
            end - frame.start, self.sample_rate, "GWF segment end"
        )
        if vector.compression.kind == CompressionKind.GZIP:
            device = getattr(scheme.mgr.state, "jax_device", None)
            if (vector.payload[:2] != b"\x1f\x8b"
                    and self._cuda_codec == "deflate"
                    and device not in self._cuda_unavailable_devices):
                from pycbc.frame.gwf_deflate_jax import GWFDeflateUnavailable

                try:
                    return self._cuda_values(frame, vector)[first:last]
                except GWFDeflateUnavailable as exc:
                    logger.info(
                        "CUDA zlib unavailable for %s; using bounded host "
                        "decompression: %s", self.channel_name, exc
                    )
                    self._cuda_unavailable_devices.add(device)
            values = self._host_values(frame, vector)[first:last]
            return _device_put(values, frame.path, self.channel_name)
        if vector.compression.kind in (
            CompressionKind.RAW,
            CompressionKind.ZERO_SUPPRESS,
        ):
            def decode():
                values = decode_fr_vect(vector)[first:last]
                if _sync_requested():
                    values.block_until_ready()
                return values

            return _timed_stage(
                "gwf_jax_decompress",
                decode,
                path=frame.path,
                channel=self.channel_name,
                compressed_bytes=len(vector.payload),
                synchronized=_sync_requested(),
            )
        raise GWFReplayUnsupported(
            "FrVect codec %s uses the compatibility reader"
            % vector.compression.kind.value
        )

    def read(self, start, duration):
        """Return a device-resident ``TimeSeries`` for ``[start, end)``."""
        start = float(start)
        duration = float(duration)
        if not math.isfinite(start) or not math.isfinite(duration):
            raise ValueError("GWF replay bounds must be finite")
        if duration <= 0:
            raise ValueError("GWF replay duration must be positive")
        _sample_index(duration, self.sample_rate, "GWF replay duration")
        end = start + duration
        cursor = start
        segments = []
        while cursor < end - 1e-12:
            candidates = [
                frame for frame in self.files
                if frame.start <= cursor < frame.end
            ]
            if len(candidates) != 1:
                raise GWFReplayUnsupported(
                    "GWF files do not uniquely cover GPS %.9f" % cursor
                )
            frame = candidates[0]
            segment_end = min(end, float(frame.end))
            segments.append(self._segment(frame, cursor, segment_end))
            cursor = segment_end

        import jax.numpy as jnp

        values = (
            segments[0]
            if len(segments) == 1
            else jnp.concatenate(segments)
        )
        expected = _sample_index(duration, self.sample_rate, "GWF replay span")
        if len(values) != expected:
            raise GWFFormatError(
                "direct GWF read returned %d samples; expected %d"
                % (len(values), expected)
            )
        return TimeSeries(
            JAXArrayData(values),
            delta_t=1.0 / self.sample_rate,
            epoch=start,
            copy=False,
        )
