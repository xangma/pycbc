# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Direct synthetic tests for the clean-room GWF FrVect decoder."""

from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import struct
import types
import zlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from pycbc import scheme  # noqa: E402
from pycbc.frame import gwf_jax  # noqa: E402
from pycbc.frame.gwf_jax import (  # noqa: E402
    CompressionKind,
    GWFFormatError,
    UnsupportedGWFCompression,
    UnsupportedGWFChecksum,
    compression_info,
    decode_fr_vect,
    parse_gwf,
    parse_header,
    posix_cksum,
)
from pycbc.frame.gwf_replay_jax import (  # noqa: E402
    GWFReplaySource,
    GWFReplayUnsupported,
)


_DTYPES = {
    0: np.dtype("int8"),
    1: np.dtype("int16"),
    2: np.dtype("float64"),
    3: np.dtype("float32"),
    4: np.dtype("int32"),
    5: np.dtype("int64"),
    6: np.dtype("complex64"),
    7: np.dtype("complex128"),
    9: np.dtype("uint16"),
    10: np.dtype("uint32"),
    11: np.dtype("uint64"),
    12: np.dtype("uint8"),
}


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _prefix(endian):
    return "<" if endian == "little" else ">"


def _pack(endian, code, *values):
    return struct.pack(_prefix(endian) + code, *values)


def _string(endian, value):
    raw = value.encode("ascii") + b"\0"
    return _pack(endian, "H", len(raw)) + raw


def _header(version, endian):
    prefix = _prefix(endian)
    result = bytearray(b"IGWD\0")
    result.extend((version, 1, 2, 4, 8, 4, 8))
    result.extend(struct.pack(prefix + "H", 0x1234))
    result.extend(struct.pack(prefix + "I", 0x12345678))
    result.extend(struct.pack(prefix + "Q", 0x0123456789ABCDEF))
    result.extend(struct.pack(prefix + "f", math.pi))
    result.extend(struct.pack(prefix + "d", math.pi))
    result.extend((0, 0))
    assert len(result) == 40
    return bytes(result)


def _fixture_cksum(data):
    """Independent, bit-at-a-time POSIX CRC for small fixture records."""
    suffix = bytearray()
    length = len(data)
    while length:
        suffix.append(length & 255)
        length >>= 8
    crc = 0
    for value in bytes(data) + suffix:
        crc ^= value << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0))
            crc &= 0xFFFFFFFF
    return crc ^ 0xFFFFFFFF


def _structure(endian, class_id, body, instance=0, checksum_type=0):
    length = 14 + len(body)
    result = (
        _pack(endian, "Q", length)
        + bytes((checksum_type, class_id))
        + _pack(endian, "I", instance)
        + body
    )
    if checksum_type == 1:
        result = result[:-4] + _pack(endian, "I", _fixture_cksum(result[:-4]))
    return result


def _frsh(endian, class_id=42, name="FrVect", checksum_type=0):
    body = (
        _string(endian, name)
        + _pack(endian, "H", class_id)
        + _string(endian, "synthetic")
        + _pack(endian, "I", 0)
    )
    return _structure(endian, 1, body, checksum_type=checksum_type)


def _raw_code(version, endian):
    if version == 8:
        return 256 if endian == "little" else 0
    return 0x8000 if endian == "little" else 0


def _zero_code(version, endian, word_bytes):
    if version == 8:
        base = {2: 5, 4: 8, 8: 10}[word_bytes]
        return base + (256 if endian == "little" else 0)
    return 0x8001 if endian == "little" else 1


def _frvect(
    version,
    endian,
    payload,
    type_code,
    compression=None,
    shape=None,
    name="H1:TEST",
    n_data=None,
    data_valid=b"",
    spacing=0.25,
    origin=-1.0,
    checksum_type=0,
):
    if n_data is None:
        n_data = int(np.prod(shape)) if shape is not None else 0
    if shape is None:
        shape = (n_data,)
    if compression is None:
        compression = _raw_code(version, endian)
    body = (
        _string(endian, name)
        + _pack(endian, "HHQQ", compression, type_code, n_data, len(payload))
        + payload
        + _pack(endian, "I", len(shape))
        + b"".join(_pack(endian, "Q", size) for size in shape)
        + b"".join(_pack(endian, "d", spacing) for _ in shape)
        + b"".join(_pack(endian, "d", origin) for _ in shape)
        + b"".join(_string(endian, "s") for _ in shape)
        + _string(endian, "count")
    )
    if version == 9:
        n_valid = len(data_valid)
        body += (
            _pack(endian, "QHQ", n_valid, 0, len(data_valid))
            + data_valid
        )
    body += _pack(endian, "HII", 0, 0, 0)
    return _structure(
        endian, 42, body, checksum_type=checksum_type
    )


def _container(version, endian, vector):
    return _header(version, endian) + _frsh(endian) + vector


def _checked_container(version, endian, vector, *, before=b"", after=b""):
    header = bytearray(_header(version, endian))
    header[39] = 1
    prefix = (
        bytes(header)
        + _frsh(endian, checksum_type=1)
        + _frsh(endian, 43, "FrEndOfFile", checksum_type=1)
        + before
        + vector
        + after
    )
    eof_length = 46 if version == 8 else 50
    fields = _pack(endian, "IQQ", 1, len(prefix) + eof_length, 0)
    if version == 9:
        fields += _pack(endian, "I", 0)  # No TOC uniqueness checksum.
    fields += _pack(endian, "III", _fixture_cksum(header), 0, 0)
    eof = bytearray(_structure(endian, 43, fields, checksum_type=0))
    eof[8] = 1
    eof[-8:-4] = _pack(endian, "I", _fixture_cksum(eof[:-8]))
    eof[-4:] = _pack(endian, "I", _fixture_cksum(prefix + eof[:-4]))
    return prefix + eof


def _replay_container(version, endian, vector, *, start=200, duration=None,
                      channel=None, time_offset=0):
    """A checked, single-channel time series with explicit frame pointers."""
    descriptor = parse_gwf(_container(version, endian, vector)).vectors[0]
    span = descriptor.n_data * descriptor.spacing[0]
    duration = span if duration is None else duration
    channel = descriptor.name if channel is None else channel
    seconds = math.floor(start)
    nanoseconds = round((start - seconds) * 1e9)
    null = _pack(endian, "HI", 0, 0)
    pointers = [null] * 13
    pointers[6] = _pack(endian, "HI", 45, 0)
    frame = (_string(endian, "TEST")
             + _pack(endian, "iIIII", 0, 0, 0, seconds, nanoseconds))
    if version == 8:
        frame += _pack(endian, "H", 18)
    frame += (_pack(endian, "d", duration) + b"".join(pointers)
              + _pack(endian, "I", 0))
    parent = (_string(endian, channel) + _string(endian, "")
              + _pack(endian, "HHdddfddH", 1, 0, time_offset, span,
                      0, 0, 0, 0, 0)
              + _pack(endian, "HI", 42, descriptor.structure.instance)
              + null * 4 + _pack(endian, "I", 0))
    before = (
        _frsh(endian, 44, "FrameH", checksum_type=1)
        + _frsh(endian, 45, "FrProcData", checksum_type=1)
        + _frsh(endian, 46, "FrEndOfFrame", checksum_type=1)
        + _structure(endian, 44, frame, checksum_type=1)
        + _structure(endian, 45, parent, checksum_type=1)
    )
    after = _structure(
        endian, 46, _pack(endian, "iIIII", 0, 0, seconds, nanoseconds, 0),
        checksum_type=1)
    return _checked_container(
        version, endian, vector, before=before, after=after)


def _file_bytes(values, endian):
    dtype = values.dtype.newbyteorder("<" if endian == "little" else ">")
    return values.astype(dtype, copy=False).tobytes()


def _components_as_words(values, type_code, endian):
    dtype = _DTYPES[type_code]
    if type_code in (6, 7):
        component_dtype = np.dtype("float32" if type_code == 6 else "float64")
        components = np.concatenate((values.real, values.imag)).astype(
            component_dtype
        )
    else:
        component_dtype = dtype
        components = values.astype(dtype)
    item_size = component_dtype.itemsize
    raw = _file_bytes(components, endian)
    return [
        int.from_bytes(raw[index:index + item_size], endian)
        for index in range(0, len(raw), item_size)
    ], item_size


def _minimal_width(block, word_bits):
    for width in range(1, word_bits):
        offset = (1 << (width - 1)) - 1
        if all(-offset <= value <= offset for value in block):
            return width
    return word_bits


def _zero_payload(values, type_code, endian, block_size):
    words, word_bytes = _components_as_words(values, type_code, endian)
    modulus = 1 << (word_bytes * 8)
    previous = 0
    differences = []
    for word in words:
        difference = (word - previous) % modulus
        if difference >= modulus // 2:
            difference -= modulus
        differences.append(difference)
        previous = word

    bit_buffer = 0
    bit_position = 0
    width_field_bits = int(math.log2(word_bytes * 8))
    for start in range(0, len(differences), block_size):
        block = differences[start:start + block_size]
        width = _minimal_width(block, word_bytes * 8)
        bit_buffer |= (width - 1) << bit_position
        bit_position += width_field_bits
        offset = (1 << (width - 1)) - 1
        for difference in block:
            # Full-width words also represent the most-negative difference
            # through the specification's useful nB bits (modular arithmetic).
            encoded = (difference + offset) & ((1 << width) - 1)
            bit_buffer |= encoded << bit_position
            bit_position += width
        bit_position += (block_size - len(block)) * width

    output = bytearray(block_size.to_bytes(2, endian))
    for shift in range(0, bit_position, 16):
        output.extend(((bit_buffer >> shift) & 0xFFFF).to_bytes(2, endian))
    return bytes(output), word_bytes


@pytest.mark.parametrize("version", (8, 9))
@pytest.mark.parametrize("endian", ("little", "big"))
def test_parse_container_versions_endian_and_v9_validity(version, endian):
    values = np.array([2, -3, 9], dtype=np.int16)
    payload = _file_bytes(values, endian)
    valid = b"\0\1\0" if version == 9 else b""
    vector = _frvect(
        version,
        endian,
        payload,
        1,
        shape=(3,),
        n_data=3,
        data_valid=valid,
    )
    parsed = parse_gwf(_container(version, endian, vector))

    assert parsed.header.version == version
    assert parsed.header.endian == endian
    assert parsed.structure_names == ((42, "FrVect"),)
    assert len(parsed.vectors) == 1
    descriptor = parsed.vectors[0]
    assert descriptor.name == "H1:TEST"
    assert descriptor.shape == (3,)
    assert descriptor.spacing == (0.25,)
    assert descriptor.origins == (-1.0,)
    assert descriptor.data_valid == valid
    np.testing.assert_array_equal(
        np.asarray(decode_fr_vect(descriptor)), values
    )


_RAW_VALUES = {
    0: np.array([-5, 0, 120], dtype=np.int8),
    1: np.array([-300, 0, 1234], dtype=np.int16),
    2: np.array([-1.25, 0.0, math.pi], dtype=np.float64),
    3: np.array([-1.25, 0.0, math.pi], dtype=np.float32),
    4: np.array([-70000, 0, 123456], dtype=np.int32),
    5: np.array([-2**40, 0, 2**42], dtype=np.int64),
    6: np.array([1 + 2j, -3.5 + 0.25j], dtype=np.complex64),
    7: np.array([1 + 2j, -3.5 + 0.25j], dtype=np.complex128),
    9: np.array([0, 7, 65530], dtype=np.uint16),
    10: np.array([0, 7, 2**32 - 2], dtype=np.uint32),
    11: np.array([0, 7, 2**63 + 9], dtype=np.uint64),
    12: np.array([0, 7, 255], dtype=np.uint8),
}


@pytest.mark.parametrize("type_code", tuple(_RAW_VALUES))
@pytest.mark.parametrize("endian", ("little", "big"))
def test_raw_decode_all_numeric_dtypes(type_code, endian):
    values = _RAW_VALUES[type_code]
    vector = _frvect(
        9,
        endian,
        _file_bytes(values, endian),
        type_code,
        shape=(len(values),),
        n_data=len(values),
    )
    descriptor = parse_gwf(_container(9, endian, vector)).vectors[0]
    actual = np.asarray(decode_fr_vect(descriptor))

    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("type_code", (1, 2, 3, 5, 6, 7, 10, 11))
@pytest.mark.parametrize("endian", ("little", "big"))
@pytest.mark.parametrize("compressed", (False, True))
@pytest.mark.parametrize("device", _devices())
def test_numeric_decode_preserves_all_bits(
        type_code, endian, compressed, device):
    dtype = _DTYPES[type_code]
    if dtype.kind in "fc":
        bits = 32 if type_code in (3, 6) else 64
        # Signed zeros, subnormals, infinities and distinct NaN payloads;
        # construct complex components by viewing bytes, without arithmetic.
        words = ((0, 1, 0x80000000, 0x807FFFFF, 0x7F800000, 0xFF800000,
                  0x7FC00001, 0x7FA00042, 0xFFC00013, 0x3F800000)
                 if bits == 32 else
                 (0, 1, 0x8000000000000000, 0x800FFFFFFFFFFFFF,
                  0x7FF0000000000000, 0xFFF0000000000000,
                  0x7FF8000000000001, 0x7FF0000000000042,
                  0xFFF8000000000013, 0x3FF0000000000000))
    else:
        bits = dtype.itemsize * 8
        words = (0, 1 << (bits - 1), (1 << bits) - 1, 0, 1, 0)
    values = np.asarray(words, dtype="uint%d" % bits).view(dtype)
    if compressed:
        payload, word_bytes = _zero_payload(values, type_code, endian, 3)
        code = _zero_code(9, endian, word_bytes)
    else:
        payload, code = _file_bytes(values, endian), _raw_code(9, endian)
    vector = _frvect(9, endian, payload, type_code, compression=code,
                     shape=(len(values),))
    descriptor = parse_gwf(_container(9, endian, vector)).vectors[0]
    with scheme.JAXScheme(device) as context:
        decoded = decode_fr_vect(descriptor)
        actual = np.asarray(decoded)
        assert decoded.devices() == {context.jax_device}
    assert actual.dtype == values.dtype
    assert actual.tobytes() == values.tobytes()


@pytest.mark.parametrize("version", (8, 9))
@pytest.mark.parametrize("endian", ("little", "big"))
def test_zero_suppress_official_example_and_partial_block(version, endian):
    values = np.array([82, 85, 85, 81, 80, 82, 84, 85], dtype=np.int16)
    payload, word_bytes = _zero_payload(values, 1, endian, block_size=3)
    vector = _frvect(
        version,
        endian,
        payload,
        1,
        compression=_zero_code(version, endian, word_bytes),
        shape=(len(values),),
        n_data=len(values),
    )
    descriptor = parse_gwf(_container(version, endian, vector)).vectors[0]

    encoded_words = [
        int.from_bytes(payload[index:index + 2], endian)
        for index in range(0, len(payload), 2)
    ]
    assert encoded_words == [3, 0x2D17, 0x37F8, 0x2963, 0x0025]
    np.testing.assert_array_equal(
        np.asarray(decode_fr_vect(descriptor)), values
    )


@pytest.mark.parametrize(
    "type_code,values",
    (
        (3, np.array([1.0, 1.25, -2.5, 8.0], dtype=np.float32)),
        (6, np.array([1 + 2j, 1.5 + 3j, 2 + 4j], dtype=np.complex64)),
        (12, np.array([3, 4, 4, 2, 255], dtype=np.uint8)),
    ),
)
def test_v9_zero_suppress_float_complex_and_byte(type_code, values):
    endian = "little"
    payload, _ = _zero_payload(values, type_code, endian, block_size=3)
    vector = _frvect(
        9,
        endian,
        payload,
        type_code,
        compression=_zero_code(9, endian, values.dtype.itemsize),
        shape=(len(values),),
        n_data=len(values),
    )
    descriptor = parse_gwf(_container(9, endian, vector)).vectors[0]
    actual = np.asarray(decode_fr_vect(descriptor))

    np.testing.assert_array_equal(actual.view("uint8"), values.view("uint8"))


@pytest.mark.parametrize(
    "version,code,kind,endian",
    (
        (8, 1, CompressionKind.GZIP, "big"),
        (8, 259, CompressionKind.DIFF_GZIP, "little"),
        (9, 2, CompressionKind.GZIP, "big"),
        (9, 0x8004, CompressionKind.DIFF_GZIP, "little"),
        (9, 8, CompressionKind.ZSTD, "big"),
        (9, 0x8010, CompressionKind.DIFF_ZSTD, "little"),
        (9, 0x8020, CompressionKind.UNKNOWN, "little"),
    ),
)
def test_known_unsupported_compressions_are_explicit(
    version, code, kind, endian
):
    info = compression_info(version, code)
    assert info.kind == kind
    assert info.endian == endian
    assert not info.supported

    raw = _frvect(
        version,
        endian,
        b"compressed",
        1,
        compression=code,
        shape=(1,),
        n_data=1,
    )
    descriptor = parse_gwf(_container(version, endian, raw)).vectors[0]
    with pytest.raises(UnsupportedGWFCompression, match=kind.value):
        decode_fr_vect(descriptor)


def test_empty_raw_and_zero_suppress_vectors():
    raw = _frvect(9, "little", b"", 3, shape=(0,), n_data=0)
    descriptor = parse_gwf(_container(9, "little", raw)).vectors[0]
    assert np.asarray(decode_fr_vect(descriptor)).shape == (0,)

    zero = _frvect(
        9,
        "little",
        (4).to_bytes(2, "little"),
        1,
        compression=_zero_code(9, "little", 2),
        shape=(0,),
        n_data=0,
    )
    descriptor = parse_gwf(_container(9, "little", zero)).vectors[0]
    assert np.asarray(decode_fr_vect(descriptor)).shape == (0,)


def test_malformed_headers_structures_strings_and_dimensions():
    with pytest.raises(GWFFormatError, match="truncated GWF"):
        parse_header(b"short")

    header = bytearray(_header(9, "little"))
    header[:5] = b"OTHER"
    with pytest.raises(GWFFormatError, match="signature"):
        parse_header(header)

    header = bytearray(_header(9, "little"))
    header[12:14] = b"xx"
    with pytest.raises(GWFFormatError, match="byte-order"):
        parse_header(header)

    valid = _container(
        9,
        "little",
        _frvect(9, "little", b"\0\0", 1, shape=(1,), n_data=1),
    )
    with pytest.raises(GWFFormatError, match="truncated structure"):
        parse_gwf(valid[:-1])

    bad_frsh_body = (
        _pack("little", "H", 3)
        + b"bad"
        + _pack("little", "H", 42)
        + _string("little", "")
        + _pack("little", "I", 0)
    )
    bad_frsh = _header(9, "little") + _structure(
        "little", 1, bad_frsh_body
    )
    with pytest.raises(GWFFormatError, match="NUL terminated"):
        parse_gwf(bad_frsh)

    mismatch = _frvect(
        9, "little", b"\0" * 6, 1, shape=(2,), n_data=3
    )
    with pytest.raises(GWFFormatError, match="dimensions"):
        parse_gwf(_container(9, "little", mismatch))


def test_decode_rejects_raw_size_zero_word_size_and_truncation():
    bad_raw = _frvect(9, "little", b"\0", 1, shape=(1,), n_data=1)
    descriptor = parse_gwf(_container(9, "little", bad_raw)).vectors[0]
    with pytest.raises(GWFFormatError, match="expected 2"):
        decode_fr_vect(descriptor)

    values = np.array([82, 85, 85, 81], dtype=np.int16)
    payload, _ = _zero_payload(values, 1, "little", block_size=3)
    base = _frvect(
        9,
        "little",
        payload,
        1,
        compression=_zero_code(9, "little", 2),
        shape=(len(values),),
        n_data=len(values),
    )
    descriptor = parse_gwf(_container(9, "little", base)).vectors[0]

    with pytest.raises(GWFFormatError, match="block size is zero"):
        decode_fr_vect(replace(descriptor, payload=b"\0\0"))
    with pytest.raises(GWFFormatError, match="truncated"):
        decode_fr_vect(replace(descriptor, payload=payload[:-2]))


def test_v8_zero_suppress_word_size_must_match_dtype():
    values = np.array([1, 2], dtype=np.int16)
    payload, _ = _zero_payload(values, 1, "big", block_size=2)
    vector = _frvect(
        8,
        "big",
        payload,
        1,
        compression=8,
        shape=(2,),
        n_data=2,
    )
    descriptor = parse_gwf(_container(8, "big", vector)).vectors[0]
    with pytest.raises(GWFFormatError, match="word size"):
        decode_fr_vect(descriptor)


@pytest.mark.parametrize("device", _devices())
def test_codec_device_shape_dtype_and_static_cache(device):
    values = np.array([1.25, -3.5, 8.0], dtype=np.float64)
    vector = _frvect(
        9,
        "little",
        _file_bytes(values, "little"),
        2,
        shape=values.shape,
        n_data=len(values),
    )
    descriptor = parse_gwf(_container(9, "little", vector)).vectors[0]
    gwf_jax._raw_kernel.cache_clear()

    with scheme.JAXScheme(device) as context:
        first = decode_fr_vect(descriptor)
        second = decode_fr_vect(descriptor)

    assert first.shape == values.shape
    assert first.dtype == values.dtype
    assert first.devices() == {context.jax_device}
    np.testing.assert_array_equal(np.asarray(first), values)
    np.testing.assert_array_equal(np.asarray(second), values)
    cache = gwf_jax._raw_kernel.cache_info()
    assert cache.misses == 1
    assert cache.hits == 1
    assert cache.currsize == 1


@pytest.mark.parametrize("compressed", (False, True))
def test_codec_honors_scheme_device_over_ambient_default(compressed):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    values = np.array([1, 2, 2, -1], dtype=np.int32)
    if compressed:
        payload, word_bytes = _zero_payload(values, 4, "little", 3)
        code = _zero_code(9, "little", word_bytes)
    else:
        payload, code = values.tobytes(), _raw_code(9, "little")
    vector = _frvect(9, "little", payload, 4, compression=code,
                     shape=values.shape)
    descriptor = parse_gwf(_container(9, "little", vector)).vectors[0]
    with scheme.JAXScheme("cpu:1") as context:
        with jax.default_device(devices[0]):
            decoded = decode_fr_vect(descriptor)
        assert decoded.devices() == {context.jax_device}
        assert np.asarray(decoded).tobytes() == values.tobytes()


@pytest.mark.parametrize("compression", ("raw", "zlib", "gzip"))
def test_replay_source_exact_channel_slice(tmp_path, compression):
    values = np.arange(16, dtype=np.float64)
    raw = _file_bytes(values, "little")
    if compression == "raw":
        payload = raw
        code = _raw_code(8, "little")
    elif compression == "zlib":
        payload = zlib.compress(raw)
        code = 257
    else:
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        payload = compressor.compress(raw) + compressor.flush()
        code = 257
    vector = _frvect(
        8,
        "little",
        payload,
        2,
        compression=code,
        shape=(len(values),),
        n_data=len(values),
        spacing=0.25,
        origin=0.0,
    )
    path = tmp_path / "H-TEST-100-4.gwf"
    path.write_bytes(_replay_container(8, "little", vector, start=100))

    with scheme.JAXScheme("cpu"):
        source = GWFReplaySource([str(path)], "H1:TEST", 4)
        actual = source.read(101, 2)

    assert float(actual.start_time) == 101
    assert actual.delta_t == 0.25
    assert actual._data.backend == "jax"
    np.testing.assert_array_equal(np.asarray(actual), values[4:12])


def test_replay_source_rejects_malformed_and_unsupported_payloads(tmp_path):
    values = np.arange(8, dtype=np.int32)
    raw = _file_bytes(values, "little")
    cases = (
        ("truncated", zlib.compress(raw)[:-2], 257, GWFFormatError),
        ("differential", zlib.compress(raw), 259, GWFReplayUnsupported),
    )
    for name, payload, code, error in cases:
        vector = _frvect(
            8,
            "little",
            payload,
            4,
            compression=code,
            shape=(len(values),),
            n_data=len(values),
            spacing=0.5,
            origin=0.0,
        )
        path = tmp_path / ("H-%s-200-4.gwf" % name)
        path.write_bytes(_replay_container(8, "little", vector))
        with scheme.JAXScheme("cpu"):
            source = GWFReplaySource([str(path)], "H1:TEST", 2)
            with pytest.raises(error):
                source.read(200, 4)


@pytest.mark.parametrize("data,expected", (
    (b"", 4294967295),
    (b"123456789", 930766865),
    (bytes(range(256)), 1313719201),
    (bytes(range(256)) * 4097, 1076062463),
))
def test_posix_cksum_known_values_and_chunk_boundary(data, expected):
    # Independent POSIX cksum utility outputs, including the length suffix.
    assert posix_cksum(data) == expected
    if len(data) <= 256:
        assert _fixture_cksum(data) == expected


@pytest.mark.parametrize("version", (8, 9))
@pytest.mark.parametrize("endian", ("little", "big"))
@pytest.mark.parametrize("compressed", (False, True))
def test_replay_validates_file_header_and_vector_checksums(
    tmp_path, version, endian, compressed
):
    values = np.arange(8, dtype=np.int32)
    payload = _file_bytes(values, endian)
    code = _raw_code(version, endian)
    if compressed:
        payload = zlib.compress(payload)
        code = (257 if endian == "little" else 1) if version == 8 else (
            0x8002 if endian == "little" else 2
        )
    vector = _frvect(
        version,
        endian,
        payload,
        4,
        compression=code,
        shape=(len(values),),
        n_data=len(values),
        spacing=0.5,
        origin=0.0,
        checksum_type=1,
    )
    path = tmp_path / "H-CHECKSUM-200-4.gwf"
    content = _replay_container(version, endian, vector)
    path.write_bytes(content)
    parsed = parse_gwf(content)
    assert all(item.checksum_type == 1 for item in parsed.structures)

    with scheme.JAXScheme("cpu"):
        source = GWFReplaySource([str(path)], "H1:TEST", 2)
        actual = source.read(201, 2)
    assert actual.delta_t == 0.5
    assert float(actual.start_time) == 201
    assert actual.dtype == values.dtype
    assert actual._data.backend == "jax"
    np.testing.assert_array_equal(np.asarray(actual), values[2:6])


def _checked_replay_fixture():
    vector = _frvect(
        8, "little", zlib.compress(np.arange(8, dtype="<i4").tobytes()),
        4, compression=257, shape=(8,), n_data=8, spacing=0.5,
        origin=0.0, checksum_type=1,
    )
    return _replay_container(8, "little", vector)


@pytest.mark.parametrize(
    "part", ("dictionary", "vector", "eof", "header", "file")
)
def test_checksum_corruption_is_rejected_before_decode(
    tmp_path, part, monkeypatch
):
    content = bytearray(_checked_replay_fixture())
    parsed = parse_gwf(content)
    if part == "dictionary":
        content[parsed.structures[0].offset + 20] ^= 1
    elif part == "vector":
        vector = parsed.vectors[0]
        payload_byte = vector.structure.offset + 14 + 2 + len(vector.name) + 21
        content[payload_byte] ^= 1
    elif part == "eof":
        content[-8] ^= 1
    elif part == "header":
        # Stored header CRC; keep EOF CRC independently valid.
        content[-12] ^= 1
        content[-8:-4] = _pack(
            "little", "I", _fixture_cksum(content[-46:-8])
        )
    else:
        content[-4] ^= 1

    from pycbc.frame import gwf_replay_jax
    monkeypatch.setattr(
        gwf_replay_jax, "_host_decompress",
        lambda vector: pytest.fail("unverified compressed bytes decoded"),
    )
    path = tmp_path / "H-CORRUPT-200-4.gwf"
    path.write_bytes(content)
    with scheme.JAXScheme("cpu"):
        with pytest.raises(GWFFormatError, match="checksum mismatch"):
            GWFReplaySource([str(path)], "H1:TEST", 2).read(200, 4)


def test_checksum_corruption_never_switches_to_compatibility_reader(tmp_path):
    from pycbc.frame.frame_jax import JAXReplayFrameReader
    from pycbc.types import TimeSeries

    content = bytearray(_checked_replay_fixture())
    content[-4] ^= 1
    path = tmp_path / "H-CORRUPT-200-4.gwf"
    path.write_bytes(content)
    with scheme.JAXScheme("cpu"):
        buffer = types.SimpleNamespace(
            frame_src=[str(path)], channel_name="H1:TEST", raw_sample_rate=2,
            read_pos=200.0,
            raw_buffer=TimeSeries(np.zeros(8), delta_t=0.5, epoch=196),
            _read_frame=lambda duration: pytest.fail(
                "corrupt frame fell back"
            ),
        )
        reader = JAXReplayFrameReader(204, 2, read_ahead_seconds=4)
        with pytest.raises(GWFFormatError, match="file checksum mismatch"):
            reader.advance(buffer, 2)
        assert not reader._gwf_disabled
        assert buffer.read_pos == 200


def test_checksummed_replay_reader_uses_native_source(tmp_path):
    from pycbc.frame.frame_jax import JAXReplayFrameReader
    from pycbc.types import TimeSeries

    path = tmp_path / "H-CHECKED-200-4.gwf"
    path.write_bytes(_checked_replay_fixture())
    with scheme.JAXScheme("cpu"):
        buffer = types.SimpleNamespace(
            frame_src=[str(path)], channel_name="H1:TEST", raw_sample_rate=2,
            read_pos=200.0,
            raw_buffer=TimeSeries(np.zeros(8), delta_t=0.5, epoch=196),
            _read_frame=lambda duration: pytest.fail(
                "valid CRC frame fell back"
            ),
        )
        reader = JAXReplayFrameReader(204, 2, read_ahead_seconds=4)
        blocks = [reader.advance(buffer, 2) for _ in range(2)]
    assert not reader._gwf_disabled
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(x) for x in blocks]),
        np.arange(8, dtype=np.int32),
    )
    assert [float(block.start_time) for block in blocks] == [200, 202]


def test_declared_file_checksum_requires_valid_eof_and_byte_count():
    content = _checked_replay_fixture()
    with pytest.raises(GWFFormatError, match="requires FrEndOfFile"):
        parse_gwf(content[:-46])
    wrong_count = bytearray(content)
    wrong_count[-28:-20] = _pack("little", "Q", len(content) + 1)
    wrong_count[-8:-4] = _pack(
        "little", "I", _fixture_cksum(wrong_count[-46:-8])
    )
    with pytest.raises(GWFFormatError, match="byte count"):
        parse_gwf(wrong_count)


def test_standalone_vector_parser_checks_structure_crc():
    content = _checked_replay_fixture()
    offset = parse_gwf(content).vectors[0].structure.offset
    assert gwf_jax.parse_fr_vect(content, offset).name == "H1:TEST"
    corrupt = bytearray(content)
    corrupt[offset + 20] ^= 1
    with pytest.raises(GWFFormatError, match="structure checksum mismatch"):
        gwf_jax.parse_fr_vect(corrupt, offset)


def test_structure_checksum_is_checked_without_file_checksum():
    checked = _checked_replay_fixture()
    vector = parse_gwf(checked).vectors[0].structure
    content = _container(
        8, "little", checked[vector.offset:vector.offset + vector.length]
    )
    assert parse_gwf(content).header.checksum_scheme == 0
    corrupt = bytearray(content)
    corrupt[-4] ^= 1
    with pytest.raises(GWFFormatError, match="structure checksum mismatch"):
        parse_gwf(corrupt)


@pytest.mark.parametrize("version", (8, 9))
@pytest.mark.parametrize("endian", ("little", "big"))
@pytest.mark.parametrize("part", ("dictionary", "vector", "eof"))
def test_unknown_structure_checksum_cannot_hide_known_file_corruption(
        version, endian, part):
    vector = _frvect(version, endian, _file_bytes(
        np.arange(8, dtype=np.int32), endian), 4,
        shape=(8,), spacing=0.5, origin=0.0, checksum_type=1)
    content = bytearray(_checked_container(version, endian, vector))
    parsed = parse_gwf(content)
    structure = (parsed.structures[0] if part == "dictionary" else
                 parsed.vectors[0].structure if part == "vector" else
                 parsed.structures[-1])
    content[structure.offset + 8] = 2
    content[-4:] = _pack(endian, "I", _fixture_cksum(content[:-4]))
    with pytest.raises(UnsupportedGWFChecksum, match="structure checksum"):
        parse_gwf(content)
    content[-4] ^= 1
    with pytest.raises(GWFFormatError, match="file checksum mismatch"):
        parse_gwf(content)


@pytest.mark.parametrize("endian", ("little", "big"))
def test_v9_nonzero_toc_checksum_defers_without_hiding_corruption(
    tmp_path, endian
):
    vector = _frvect(
        9, endian, _file_bytes(np.arange(8, dtype=np.int32), endian), 4,
        shape=(8,), n_data=8, spacing=0.5, origin=0.0, checksum_type=1,
    )
    content = bytearray(_replay_container(9, endian, vector))
    content[-16:-12] = _pack(endian, "I", 1)
    content[-8:-4] = _pack(endian, "I", _fixture_cksum(content[-50:-8]))
    content[-4:] = _pack(endian, "I", _fixture_cksum(content[:-4]))
    with pytest.raises(UnsupportedGWFChecksum, match="version-9 TOC"):
        parse_gwf(content)
    path = tmp_path / "H-TOC-200-4.gwf"
    path.write_bytes(content)
    with scheme.JAXScheme("cpu"):
        with pytest.raises(GWFReplayUnsupported, match="unsupported GWF"):
            GWFReplaySource([str(path)], "H1:TEST", 2).read(200, 4)
    content[-4] ^= 1
    with pytest.raises(GWFFormatError, match="file checksum mismatch"):
        parse_gwf(content)


@pytest.mark.parametrize("type_code", (0, 1, 9, 11, 12))
def test_checked_vector_unsupported_timeseries_dtype_falls_back(
    tmp_path, type_code
):
    values = np.arange(8, dtype=_DTYPES[type_code])
    vector = _frvect(
        8, "little", _file_bytes(values, "little"), type_code,
        shape=(8,), n_data=8, spacing=0.5, origin=0.0, checksum_type=1,
    )
    path = tmp_path / "H-DTYPE-200-4.gwf"
    path.write_bytes(_replay_container(8, "little", vector))
    with scheme.JAXScheme("cpu"):
        with pytest.raises(
            GWFReplayUnsupported, match="unsupported by TimeSeries"
        ):
            GWFReplaySource([str(path)], "H1:TEST", 2).read(200, 4)


@pytest.mark.parametrize("unsupported", ("checksum", "dtype"))
def test_checked_reader_retains_unsupported_input_fallback(
    tmp_path, unsupported
):
    from pycbc.frame.frame_jax import JAXReplayFrameReader
    from pycbc.types import TimeSeries

    content = bytearray(_checked_replay_fixture())
    if unsupported == "checksum":
        content[39] = 2
    else:
        vector = _frvect(
            8, "little", np.arange(8, dtype="<i2").tobytes(), 1,
            shape=(8,), n_data=8, spacing=0.5, origin=0.0, checksum_type=1,
        )
        content = _replay_container(8, "little", vector)
    path = tmp_path / "H-UNSUPPORTED-200-4.gwf"
    path.write_bytes(content)
    calls = []
    with scheme.JAXScheme("cpu"):
        def compatibility_read(duration):
            calls.append(duration)
            return TimeSeries(
                np.arange(8, dtype=np.int32), delta_t=0.5, epoch=200
            )

        buffer = types.SimpleNamespace(
            frame_src=[str(path)], channel_name="H1:TEST", raw_sample_rate=2,
            read_pos=200.0,
            raw_buffer=TimeSeries(np.zeros(8), delta_t=0.5, epoch=196),
            _read_frame=compatibility_read,
        )
        reader = JAXReplayFrameReader(204, 2, read_ahead_seconds=4)
        actual = reader.advance(buffer, 2)
    assert reader._gwf_disabled
    assert calls == [4]
    np.testing.assert_array_equal(
        np.asarray(actual), np.arange(4, dtype=np.int32)
    )


@pytest.mark.parametrize("ifo,digest", (
    ("H1",
     "580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865"),
    ("L1",
     "ff9743efc1555a47bc3f2effcaabba252a2244f10fd3a8f85a7c5f677b3c772e"),
))
def test_pinned_losc_checked_replay_matches_public_reader(ifo, digest):
    """Optional real-frame oracle, selected by PYCBC_TEST_GWF_DIR."""
    from pycbc.frame import read_frame
    from pycbc.frame.frame_jax import JAXReplayFrameReader
    from pycbc.types import TimeSeries

    fixture_dir = Path(__file__).resolve().parents[1] / "docs/_include"
    directory = Path(os.environ.get("PYCBC_TEST_GWF_DIR", fixture_dir))
    name = "%s-%s_LOSC_CLN_4_V1-1187007040-2048.gwf" % (ifo[0], ifo)
    path = directory / name
    if not path.is_file():
        pytest.skip("pinned LOSC frame not available")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    for suffix, rate, duration in (
        ("STRAIN", 4096, 64), ("DQMASK", 1, 64), ("INJMASK", 1, 64),
        ("STRAIN", 4096, 640),
    ):
        channel = "%s:LOSC-%s" % (ifo, suffix)
        start = 1187007080
        with scheme.CPUScheme():
            expected = read_frame(
                str(path), channel, start_time=start,
                duration=duration, check_integrity=True,
            )
        with scheme.JAXScheme("cpu"):
            buffer = types.SimpleNamespace(
                frame_src=[str(path)], channel_name=channel,
                raw_sample_rate=rate,
                read_pos=float(start),
                raw_buffer=TimeSeries(
                    np.zeros(8 * rate, dtype=expected.dtype),
                    delta_t=1.0 / rate, epoch=start - 8,
                ),
                _read_frame=lambda duration: pytest.fail(
                    "LOSC replay used compatibility reader"
                ),
            )
            reader = JAXReplayFrameReader(start + duration, 2, duration)
            blocks = [reader.advance(buffer, 2) for _ in range(duration // 2)]
            actual = np.concatenate([np.asarray(block) for block in blocks])
        assert not reader._gwf_disabled
        assert reader._gwf_source is not None
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.numpy().tobytes()
        assert blocks[0].start_time == expected.start_time
        assert blocks[0].delta_t == expected.delta_t
        assert buffer.read_pos == start + duration


def test_replay_source_defers_unknown_checksum_format_string_and_validity(
    tmp_path
):
    cases = []

    checked = bytearray(_container(
        8,
        "little",
        _frvect(
            8, "little", b"\0" * 32, 4, shape=(8,), n_data=8,
            spacing=0.5, origin=0.0,
        ),
    ))
    checked[39] = 2
    cases.append(("file-checksum", bytes(checked), "unsupported GWF"))
    with pytest.raises(UnsupportedGWFChecksum):
        parse_gwf(checked)

    unknown_structure = bytearray(_checked_replay_fixture())
    unknown_structure[48] = 2
    unknown_structure[-4:] = _pack(
        "little", "I", _fixture_cksum(unknown_structure[:-4]))
    cases.append((
        "structure-checksum", bytes(unknown_structure), "unsupported GWF"
    ))
    with pytest.raises(UnsupportedGWFChecksum):
        parse_gwf(unknown_structure)

    unknown_version = bytearray(checked)
    unknown_version[5] = 7
    cases.append(("version", bytes(unknown_version), "unsupported GWF"))

    string = _frvect(
        9, "little", b"abcdefgh", 8, shape=(8,), n_data=8,
        spacing=0.5, origin=0.0,
    )
    cases.append(("string", _replay_container(9, "little", string), "string"))

    validity = _frvect(
        9, "little", b"\0" * 32, 4, shape=(8,), n_data=8,
        data_valid=b"\1" * 8, spacing=0.5, origin=0.0,
    )
    cases.append((
        "validity", _replay_container(9, "little", validity),
        "validity metadata"
    ))

    for name, content, message in cases:
        path = tmp_path / ("H-%s-200-4.gwf" % name)
        path.write_bytes(content)
        with scheme.JAXScheme("cpu"):
            source = GWFReplaySource([str(path)], "H1:TEST", 2)
            with pytest.raises(GWFReplayUnsupported, match=message):
                source.read(200, 4)


def test_replay_source_contiguous_boundary_gap_and_overlap(tmp_path):
    def write_frame(name, start, values):
        vector = _frvect(
            8,
            "little",
            _file_bytes(values, "little"),
            2,
            shape=(len(values),),
            n_data=len(values),
            spacing=0.25,
            origin=0.0,
        )
        path = tmp_path / ("H-%s-%d-2.gwf" % (name, start))
        path.write_bytes(_replay_container(8, "little", vector, start=start))
        return str(path)

    first_values = np.arange(8, dtype=np.float64)
    second_values = np.arange(8, 16, dtype=np.float64)
    first = write_frame("FIRST", 100, first_values)
    contiguous = write_frame("CONTIGUOUS", 102, second_values)
    gap = write_frame("GAP", 103, second_values)
    overlap = write_frame("OVERLAP", 101, second_values)

    with scheme.JAXScheme("cpu"):
        source = GWFReplaySource([first, contiguous], "H1:TEST", 4)
        actual = source.read(101, 2)
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.concatenate((first_values[4:], second_values[:4])),
        )

        with pytest.raises(GWFReplayUnsupported, match="uniquely cover"):
            GWFReplaySource([first, gap], "H1:TEST", 4).read(101, 3)
        with pytest.raises(GWFReplayUnsupported, match="uniquely cover"):
            GWFReplaySource([first, overlap], "H1:TEST", 4).read(101, 1)
