# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Small, clean-room GWF FrVect parser and JAX payload decoder.

This module is derived solely from the published IGWD frame-format
specifications, not from LAL, LALFrame, or another frame-library
implementation:

* format 8: LIGO-T970130-v2, tables 5--8 and 26, appendices B--C
  https://dcc.ligo.org/public/0000/T970130/002/T970130-v2.pdf
* format 9: LIGO-T970130-v4, tables 4, 6--9 and 26, appendices A--B
  https://dcc.ligo.org/public/0000/T970130/004/T970130-v4.pdf

The deliberately narrow surface parses the file header, common structure
headers, FrSH dictionaries, and FrVect structures.  Raw and zero-suppressed
numeric FrVect payloads can be decoded on a JAX device.  Gzip, differential
gzip, zstd, differential zstd, strings, and unknown compression identifiers
are recognized but rejected explicitly by :func:`decode_fr_vect`.

The container parser validates declared POSIX structure, file-header, and
whole-file checksums before returning descriptors. The standalone vector parser
checks its structure CRC. Higher-level frame topology, channel selection, and
time metadata remain the integration layer's job.
"""

from dataclasses import dataclass
from enum import Enum
import binascii
import functools
import math
import struct

import numpy as np


__all__ = (
    "CompressionInfo",
    "CompressionKind",
    "FrVectDescriptor",
    "GWFContainer",
    "GWFFormatError",
    "GWFHeader",
    "StructureDescriptor",
    "UnsupportedGWFCompression",
    "UnsupportedGWFChecksum",
    "UnsupportedGWFFormat",
    "compression_info",
    "decode_fr_vect",
    "iter_structures",
    "parse_fr_vect",
    "parse_gwf",
    "parse_header",
    "posix_cksum",
)

FILE_HEADER_SIZE = 40
COMMON_HEADER_SIZE = 14


class GWFFormatError(ValueError):
    """The byte stream does not satisfy the supported GWF specification."""


class UnsupportedGWFFormat(GWFFormatError):
    """The version or scalar layout needs a compatibility reader."""


class UnsupportedGWFCompression(NotImplementedError):
    """The FrVect compression is known but not implemented by this module."""


class UnsupportedGWFChecksum(NotImplementedError):
    """The declared checksum scheme needs a compatibility reader."""


_REVERSE_BYTE = bytes(int(format(value, "08b")[::-1], 2)
                      for value in range(256))


def posix_cksum(data):
    """Return the POSIX ``cksum`` CRC required by GWF checksum scheme 1.

    POSIX uses the unreflected 0x04C11DB7 polynomial, an initial remainder of
    zero, the byte count appended least-significant byte first, and a final
    complement. Reversing each byte maps this to the standard library's
    reflected CRC. Chunking bounds temporary storage for full-file checks.
    """
    raw = memoryview(data).cast("B")
    crc = 0xFFFFFFFF  # binascii complements this to the required zero state.
    for start in range(0, len(raw), 1024 * 1024):
        chunk = bytes(raw[start:start + 1024 * 1024]).translate(_REVERSE_BYTE)
        crc = binascii.crc32(chunk, crc)
    length = len(raw)
    suffix = bytearray()
    while length:
        suffix.append(length & 0xFF)
        length >>= 8
    crc = binascii.crc32(bytes(suffix).translate(_REVERSE_BYTE), crc)
    return int.from_bytes(crc.to_bytes(4, "little").translate(_REVERSE_BYTE),
                          "big")


class CompressionKind(str, Enum):
    """Published FrVect compression families."""

    RAW = "raw"
    ZERO_SUPPRESS = "zero-suppress"
    GZIP = "gzip"
    DIFF_GZIP = "differential-gzip"
    ZSTD = "zstd"
    DIFF_ZSTD = "differential-zstd"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GWFHeader:
    """The fixed 40-byte GWF file header."""

    originator: str
    version: int
    library_minor: int
    endian: str
    library_id: int
    checksum_scheme: int

    @property
    def struct_prefix(self):
        return "<" if self.endian == "little" else ">"


@dataclass(frozen=True)
class StructureDescriptor:
    """Common header shared by every on-media frame structure."""

    offset: int
    length: int
    checksum_type: int
    class_id: int
    instance: int


@dataclass(frozen=True)
class CompressionInfo:
    """Decoded meaning of an on-media compression identifier."""

    code: int
    kind: CompressionKind
    endian: str | None
    word_bytes: int | None = None

    @property
    def supported(self):
        return self.kind in (
            CompressionKind.RAW,
            CompressionKind.ZERO_SUPPRESS,
        )


@dataclass(frozen=True)
class FrVectDescriptor:
    """Validated FrVect metadata plus its still-compressed payload."""

    structure: StructureDescriptor
    version: int
    file_endian: str
    name: str
    compression: CompressionInfo
    type_code: int
    dtype: str
    n_data: int
    payload: bytes
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origins: tuple[float, ...]
    unit_x: tuple[str, ...]
    unit_y: str
    n_data_valid: int
    data_valid_compression: int | None
    data_valid: bytes
    next_class: int
    next_instance: int
    checksum: int


@dataclass(frozen=True)
class GWFContainer:
    """The bounded subset of a GWF container understood by this module."""

    header: GWFHeader
    structures: tuple[StructureDescriptor, ...]
    structure_names: tuple[tuple[int, str], ...]
    vectors: tuple[FrVectDescriptor, ...]


@dataclass(frozen=True)
class _TypeInfo:
    name: str
    component_bytes: int
    components: int = 1

    @property
    def item_bytes(self):
        return self.component_bytes * self.components


_TYPES = {
    0: _TypeInfo("int8", 1),
    1: _TypeInfo("int16", 2),
    2: _TypeInfo("float64", 8),
    3: _TypeInfo("float32", 4),
    4: _TypeInfo("int32", 4),
    5: _TypeInfo("int64", 8),
    6: _TypeInfo("complex64", 4, 2),
    7: _TypeInfo("complex128", 8, 2),
    8: _TypeInfo("string", 1),
    9: _TypeInfo("uint16", 2),
    10: _TypeInfo("uint32", 4),
    11: _TypeInfo("uint64", 8),
    12: _TypeInfo("uint8", 1),
}


class _Cursor:
    def __init__(self, data, start, end, prefix):
        self.data = memoryview(data)
        self.pos = start
        self.end = end
        self.prefix = prefix

    def take(self, size, label):
        if size < 0 or self.pos + size > self.end:
            raise GWFFormatError("truncated %s at byte %d" % (label, self.pos))
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def uint(self, size, label):
        codes = {2: "H", 4: "I", 8: "Q"}
        raw = self.take(size, label)
        return struct.unpack(self.prefix + codes[size], raw)[0]

    def float64(self, label):
        return struct.unpack(self.prefix + "d", self.take(8, label))[0]

    def string(self, label):
        size = self.uint(2, label + " length")
        raw = bytes(self.take(size, label))
        if not raw or raw[-1] != 0:
            raise GWFFormatError("%s is not NUL terminated" % label)
        value = raw.rstrip(b"\0")
        if b"\0" in value:
            raise GWFFormatError("%s contains an embedded NUL" % label)
        try:
            return value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise GWFFormatError("%s is not ASCII" % label) from exc


def parse_header(data):
    """Parse and validate the fixed GWF v8/v9 file header."""
    raw = memoryview(data)
    if len(raw) < FILE_HEADER_SIZE:
        raise GWFFormatError("truncated GWF file header")
    if bytes(raw[:5]) != b"IGWD\0":
        raise GWFFormatError("invalid GWF file signature")
    version = raw[5]
    if version not in (8, 9):
        raise UnsupportedGWFFormat(
            "unsupported GWF format version %d" % version
        )
    if tuple(raw[7:12]) != (2, 4, 8, 4, 8):
        raise UnsupportedGWFFormat("unsupported GWF scalar sizes")

    marker = bytes(raw[12:14])
    if marker == b"\x34\x12":
        endian = "little"
        prefix = "<"
    elif marker == b"\x12\x34":
        endian = "big"
        prefix = ">"
    else:
        raise GWFFormatError("invalid GWF byte-order marker")
    if struct.unpack(prefix + "I", raw[14:18])[0] != 0x12345678:
        raise GWFFormatError("invalid GWF INT_4 byte-order marker")
    if struct.unpack(prefix + "Q", raw[18:26])[0] != 0x0123456789ABCDEF:
        raise GWFFormatError("invalid GWF INT_8 byte-order marker")
    pi4 = struct.unpack(prefix + "f", raw[26:30])[0]
    pi8 = struct.unpack(prefix + "d", raw[30:38])[0]
    if not math.isclose(pi4, math.pi, rel_tol=1e-6):
        raise GWFFormatError("invalid GWF REAL_4 marker")
    if not math.isclose(pi8, math.pi, rel_tol=1e-14):
        raise GWFFormatError("invalid GWF REAL_8 marker")
    if raw[39] not in (0, 1):
        raise UnsupportedGWFChecksum("unsupported GWF file checksum scheme")
    return GWFHeader(
        originator="IGWD",
        version=version,
        library_minor=raw[6],
        endian=endian,
        library_id=raw[38],
        checksum_scheme=raw[39],
    )


def compression_info(version, code):
    """Return the published meaning and payload endianness of ``code``."""
    if version == 8:
        table = {
            0: (CompressionKind.RAW, "big", None),
            256: (CompressionKind.RAW, "little", None),
            1: (CompressionKind.GZIP, "big", None),
            257: (CompressionKind.GZIP, "little", None),
            3: (CompressionKind.DIFF_GZIP, "big", None),
            259: (CompressionKind.DIFF_GZIP, "little", None),
            5: (CompressionKind.ZERO_SUPPRESS, "big", 2),
            261: (CompressionKind.ZERO_SUPPRESS, "little", 2),
            8: (CompressionKind.ZERO_SUPPRESS, "big", 4),
            264: (CompressionKind.ZERO_SUPPRESS, "little", 4),
            10: (CompressionKind.ZERO_SUPPRESS, "big", 8),
            266: (CompressionKind.ZERO_SUPPRESS, "little", 8),
        }
        kind, endian, width = table.get(
            code, (CompressionKind.UNKNOWN, None, None)
        )
        return CompressionInfo(code, kind, endian, width)
    if version == 9:
        endian = "little" if code & 0x8000 else "big"
        table = {
            0: CompressionKind.RAW,
            1: CompressionKind.ZERO_SUPPRESS,
            2: CompressionKind.GZIP,
            4: CompressionKind.DIFF_GZIP,
            8: CompressionKind.ZSTD,
            16: CompressionKind.DIFF_ZSTD,
        }
        return CompressionInfo(
            code, table.get(code & 0x7FFF, CompressionKind.UNKNOWN), endian
        )
    raise GWFFormatError("unsupported GWF format version %d" % version)


def _parse_common(data, offset, header):
    if offset < FILE_HEADER_SIZE or offset + COMMON_HEADER_SIZE > len(data):
        raise GWFFormatError("truncated structure header at byte %d" % offset)
    cursor = _Cursor(data, offset, len(data), header.struct_prefix)
    length = cursor.uint(8, "structure length")
    checksum_type = cursor.take(1, "structure checksum type")[0]
    class_id = cursor.take(1, "structure class")[0]
    instance = cursor.uint(4, "structure instance")
    if length < COMMON_HEADER_SIZE:
        raise GWFFormatError("invalid structure length %d at byte %d" % (
            length, offset
        ))
    if offset + length > len(data):
        raise GWFFormatError("truncated structure at byte %d" % offset)
    return StructureDescriptor(
        offset, length, checksum_type, class_id, instance
    )


def _validate_structure_checksum(
    data, structure, header, checksum_offset=None
):
    if not structure.checksum_type:
        return
    if structure.checksum_type != 1:
        raise UnsupportedGWFChecksum("unsupported structure checksum type")
    if checksum_offset is None:
        checksum_offset = structure.offset + structure.length - 4
    if checksum_offset < structure.offset + COMMON_HEADER_SIZE:
        raise GWFFormatError("truncated structure checksum")
    stored = struct.unpack_from(header.struct_prefix + "I", data,
                                checksum_offset)[0]
    actual = posix_cksum(memoryview(data)[structure.offset:checksum_offset])
    if stored != actual:
        raise GWFFormatError("structure checksum mismatch at byte %d"
                             % structure.offset)


def _validate_end_of_file(data, structure, header):
    # Tables 12 (v8) / 13 (v9): v9 adds chkSumTOC before chkSumFrHeader.
    expected_length = 46 if header.version == 8 else 50
    if (structure.length != expected_length
            or structure.offset + structure.length != len(data)):
        raise GWFFormatError("invalid FrEndOfFile location or length")
    cursor = _Cursor(data, structure.offset + COMMON_HEADER_SIZE,
                     len(data), header.struct_prefix)
    cursor.uint(4, "FrEndOfFile.nFrames")
    n_bytes = cursor.uint(8, "FrEndOfFile.nBytes")
    cursor.uint(8, "FrEndOfFile.seekTOC")
    toc_checksum = 0
    if header.version == 9:
        toc_checksum = cursor.uint(4, "FrEndOfFile.chkSumTOC")
    header_checksum = cursor.uint(4, "FrEndOfFile.chkSumFrHeader")
    structure_checksum_offset = cursor.pos
    cursor.uint(4, "FrEndOfFile.chkSum")
    file_checksum_offset = cursor.pos
    file_checksum = cursor.uint(4, "FrEndOfFile.chkSumFile")
    if n_bytes and n_bytes != len(data):
        raise GWFFormatError("FrEndOfFile byte count does not match file")
    if header.checksum_scheme:
        if header_checksum != posix_cksum(memoryview(data)[:FILE_HEADER_SIZE]):
            raise GWFFormatError("file header checksum mismatch")
        actual_file_checksum = posix_cksum(
            memoryview(data)[:file_checksum_offset]
        )
        if file_checksum != actual_file_checksum:
            raise GWFFormatError("file checksum mismatch")
    # An unknown structure scheme cannot hide a known file/header CRC failure.
    # parse_gwf reports it after all known integrity checks have completed.
    if structure.checksum_type in (0, 1):
        _validate_structure_checksum(
            data, structure, header, structure_checksum_offset)
    if toc_checksum:
        raise UnsupportedGWFChecksum(
            "nonzero version-9 TOC checksum requires the compatibility reader"
        )


def iter_structures(data, header=None):
    """Yield validated common structure headers in on-media order."""
    if header is None:
        header = parse_header(data)
    offset = FILE_HEADER_SIZE
    while offset < len(data):
        structure = _parse_common(data, offset, header)
        yield structure
        offset += structure.length


def _parse_frsh(data, structure, header):
    cursor = _Cursor(
        data,
        structure.offset + COMMON_HEADER_SIZE,
        structure.offset + structure.length,
        header.struct_prefix,
    )
    name = cursor.string("FrSH.name")
    class_id = cursor.uint(2, "FrSH.class")
    cursor.string("FrSH.comment")
    cursor.uint(4, "FrSH.checksum")
    if cursor.pos != cursor.end:
        raise GWFFormatError("unexpected bytes at end of FrSH")
    if class_id > 255:
        raise GWFFormatError("FrSH class does not fit the common header")
    return class_id, name


def parse_fr_vect(data, offset, header=None):
    """Parse one FrVect at an absolute structure offset.

    This entry point is useful when an integration layer already obtained a
    FrVect location from frame topology.  :func:`parse_gwf` instead discovers
    FrVect's class through the file's FrSH dictionary. This entry point checks
    the vector's declared structure CRC; file integrity requires parse_gwf.
    """
    if header is None:
        header = parse_header(data)
    structure = _parse_common(data, offset, header)
    _validate_structure_checksum(data, structure, header)
    cursor = _Cursor(
        data,
        offset + COMMON_HEADER_SIZE,
        offset + structure.length,
        header.struct_prefix,
    )
    name = cursor.string("FrVect.name")
    compression_code = cursor.uint(2, "FrVect.compress")
    type_code = cursor.uint(2, "FrVect.type")
    n_data = cursor.uint(8, "FrVect.nData")
    n_bytes = cursor.uint(8, "FrVect.nBytes")
    payload = bytes(cursor.take(n_bytes, "FrVect.data"))
    n_dim = cursor.uint(4, "FrVect.nDim")
    if n_dim == 0:
        raise GWFFormatError("FrVect.nDim must be positive")
    shape = tuple(cursor.uint(8, "FrVect.nx") for _ in range(n_dim))
    spacing = tuple(cursor.float64("FrVect.dx") for _ in range(n_dim))
    origins = tuple(cursor.float64("FrVect.startX") for _ in range(n_dim))
    unit_x = tuple(cursor.string("FrVect.unitX") for _ in range(n_dim))
    unit_y = cursor.string("FrVect.unitY")

    if type_code not in _TYPES:
        raise GWFFormatError("unknown FrVect type code %d" % type_code)
    if math.prod(shape) != n_data:
        raise GWFFormatError("FrVect dimensions do not match nData")

    n_data_valid = 0
    data_valid_compression = None
    data_valid = b""
    if header.version == 9:
        n_data_valid = cursor.uint(8, "FrVect.nDataValid")
        data_valid_compression = cursor.uint(
            2, "FrVect.dataValidCompScheme"
        )
        data_valid_bytes = cursor.uint(8, "FrVect.nDataValidCompBytes")
        data_valid = bytes(cursor.take(data_valid_bytes, "FrVect.dataValid"))
        if n_data_valid:
            if not n_data or n_data % n_data_valid:
                raise GWFFormatError(
                    "FrVect data-valid length does not divide nData"
                )
            block = n_data // n_data_valid
            if shape[-1] % block:
                raise GWFFormatError(
                    "FrVect data-valid blocks do not divide its inner "
                    "dimension"
                )
        elif data_valid_bytes:
            raise GWFFormatError(
                "FrVect has bytes for an empty data-valid array"
            )

    next_class = cursor.uint(2, "FrVect.next.class")
    next_instance = cursor.uint(4, "FrVect.next.instance")
    checksum = cursor.uint(4, "FrVect.checksum")
    if cursor.pos != cursor.end:
        raise GWFFormatError("unexpected bytes at end of FrVect")
    return FrVectDescriptor(
        structure=structure,
        version=header.version,
        file_endian=header.endian,
        name=name,
        compression=compression_info(header.version, compression_code),
        type_code=type_code,
        dtype=_TYPES[type_code].name,
        n_data=n_data,
        payload=payload,
        shape=shape,
        spacing=spacing,
        origins=origins,
        unit_x=unit_x,
        unit_y=unit_y,
        n_data_valid=n_data_valid,
        data_valid_compression=data_valid_compression,
        data_valid=data_valid,
        next_class=next_class,
        next_instance=next_instance,
        checksum=checksum,
    )


def parse_gwf(data):
    """Scan a GWF v8/v9 container, validate CRCs, and return FrVect objects.

    Every declared structure CRC is checked, including compressed payloads.
    File scheme 1 also requires a terminal FrEndOfFile with matching header
    and whole-file CRCs. Unsupported checksum schemes are distinct from
    corrupt data so replay callers can fall back only for unsupported input.
    Nonzero version-9 TOC checksums require the compatibility reader.
    """
    header = parse_header(data)
    structures = []
    names = {}
    vectors = []
    end_of_file = None
    unsupported_checksum = False
    for structure in iter_structures(data, header):
        structures.append(structure)
        supported_checksum = structure.checksum_type in (0, 1)
        unsupported_checksum |= not supported_checksum
        if structure.class_id == 1:
            if supported_checksum:
                _validate_structure_checksum(data, structure, header)
            class_id, name = _parse_frsh(data, structure, header)
            previous = names.get(class_id)
            if previous is not None and previous != name:
                raise GWFFormatError("conflicting FrSH class definitions")
            names[class_id] = name
        elif names.get(structure.class_id) == "FrVect":
            if supported_checksum:
                vectors.append(parse_fr_vect(data, structure.offset, header))
        elif names.get(structure.class_id) == "FrEndOfFile":
            if end_of_file is not None:
                raise GWFFormatError("multiple FrEndOfFile structures")
            end_of_file = structure
        elif supported_checksum:
            _validate_structure_checksum(data, structure, header)
    if end_of_file is not None:
        _validate_end_of_file(data, end_of_file, header)
    elif header.checksum_scheme:
        raise GWFFormatError("file checksum requires FrEndOfFile")
    if unsupported_checksum:
        raise UnsupportedGWFChecksum("unsupported structure checksum type")
    return GWFContainer(
        header=header,
        structures=tuple(structures),
        structure_names=tuple(sorted(names.items())),
        vectors=tuple(vectors),
    )


def _require_jax_width(component_bytes):
    try:
        import jax
    except ImportError as exc:
        raise ImportError(
            "JAX is required to decode an FrVect payload"
        ) from exc
    if component_bytes == 8 and not jax.config.x64_enabled:
        raise RuntimeError(
            "exact 64-bit FrVect decoding requires JAX x64 mode"
        )
    return jax


def _assemble_words(jnp, payload, word_bytes, endian):
    bits = word_bytes * 8
    dtype = getattr(jnp, "uint%d" % bits)
    raw = payload.reshape((payload.shape[0] // word_bytes, word_bytes)).astype(
        dtype
    )
    result = jnp.zeros((raw.shape[0],), dtype=dtype)
    for index in range(word_bytes):
        shift = 8 * (index if endian == "little" else word_bytes - 1 - index)
        result = jnp.bitwise_or(result, jnp.left_shift(raw[:, index], shift))
    return result


def _words_to_values(jax, words, type_code, n_data, planar_complex=False):
    jnp = jax.numpy
    lax = jax.lax
    signed = {0: jnp.int8, 1: jnp.int16, 4: jnp.int32, 5: jnp.int64}
    unsigned = {9: jnp.uint16, 10: jnp.uint32, 11: jnp.uint64, 12: jnp.uint8}
    floats = {2: jnp.float64, 3: jnp.float32}
    if type_code in signed:
        return lax.bitcast_convert_type(words, signed[type_code])
    if type_code in unsigned:
        return words.astype(unsigned[type_code])
    if type_code in floats:
        return lax.bitcast_convert_type(words, floats[type_code])
    if type_code in (6, 7):
        float_dtype = jnp.float32 if type_code == 6 else jnp.float64
        parts = lax.bitcast_convert_type(words, float_dtype)
        if planar_complex:
            real, imag = parts[:n_data], parts[n_data:]
        else:
            pairs = parts.reshape((n_data, 2))
            real, imag = pairs[:, 0], pairs[:, 1]
        return lax.complex(real, imag)
    raise UnsupportedGWFCompression("FrVect string data are not supported")


@functools.lru_cache(maxsize=64)
def _raw_kernel(type_code, n_data, endian):
    info = _TYPES[type_code]
    jax = _require_jax_width(info.component_bytes)
    jnp = jax.numpy

    @jax.jit
    def kernel(payload):
        words = _assemble_words(jnp, payload, info.component_bytes, endian)
        return _words_to_values(jax, words, type_code, n_data)

    return kernel


def _host_read_bits(words, position, width):
    value = 0
    for index in range(width):
        bit_position = position + index
        word_index = bit_position // 16
        if word_index >= len(words):
            raise GWFFormatError("truncated zero-suppressed FrVect payload")
        value |= ((words[word_index] >> (bit_position % 16)) & 1) << index
    return value


def _zero_layout(payload, endian, word_bits, n_words):
    if len(payload) < 2 or len(payload) % 2:
        raise GWFFormatError(
            "zero-suppressed FrVect payload must contain 16-bit words"
        )
    block_size = int.from_bytes(payload[:2], endian)
    if not block_size:
        raise GWFFormatError("zero-suppressed FrVect block size is zero")
    words = [
        int.from_bytes(payload[index:index + 2], endian)
        for index in range(2, len(payload), 2)
    ]
    width_bits = int(math.log2(word_bits))
    position = 0
    remaining = n_words
    layout = []
    while remaining:
        width = _host_read_bits(words, position, width_bits) + 1
        position += width_bits
        if width > word_bits:
            raise GWFFormatError("invalid zero-suppressed FrVect word width")
        output_count = min(block_size, remaining)
        layout.append((position, width, output_count))
        position += block_size * width
        remaining -= output_count
    total_bits = len(words) * 16
    if position > total_bits:
        raise GWFFormatError("truncated zero-suppressed FrVect payload")
    if total_bits - position >= 16:
        raise GWFFormatError(
            "trailing words in zero-suppressed FrVect payload"
        )
    return tuple(layout)


@functools.lru_cache(maxsize=64)
def _zero_kernel(type_code, n_data, endian, layout):
    info = _TYPES[type_code]
    jax = _require_jax_width(info.component_bytes)
    jnp = jax.numpy
    word_bits = info.component_bytes * 8
    word_dtype = getattr(jnp, "uint%d" % word_bits)
    component_count = n_data * info.components

    @jax.jit
    def kernel(payload):
        packed = _assemble_words(jnp, payload, 2, endian)
        differences = []
        for start, width, count in layout:
            positions = (
                start
                + jnp.arange(count, dtype=jnp.int32)[:, None] * width
                + jnp.arange(width, dtype=jnp.int32)[None, :]
            )
            source = packed[positions // 16]
            bits = jnp.bitwise_and(
                jnp.right_shift(source, positions % 16), 1
            ).astype(word_dtype)
            powers = jnp.left_shift(
                jnp.ones((width,), dtype=word_dtype),
                jnp.arange(width, dtype=word_dtype),
            )
            encoded = jnp.sum(bits * powers, axis=1, dtype=word_dtype)
            offset = jnp.asarray((1 << (width - 1)) - 1, dtype=word_dtype)
            differences.append(encoded - offset)
        if differences:
            differences = jnp.concatenate(differences)[:component_count]
            words = jax.lax.associative_scan(jnp.add, differences)
        else:
            words = jnp.empty((0,), dtype=word_dtype)
        return _words_to_values(
            jax, words, type_code, n_data, planar_complex=True
        )

    return kernel


def _device_payload(jax, data):
    """Honor the active scheme even if another JAX default device is set."""
    from pycbc import scheme

    device = getattr(scheme.mgr.state, "jax_device", None)
    return jax.device_put(np.frombuffer(data, dtype=np.uint8), device=device)


def decode_fr_vect(vector):
    """Decode a numeric FrVect and return a one-dimensional JAX array."""
    info = _TYPES[vector.type_code]
    if vector.type_code == 8:
        raise UnsupportedGWFCompression("FrVect string data are not supported")
    compression = vector.compression
    if compression.kind == CompressionKind.RAW:
        expected = vector.n_data * info.item_bytes
        if len(vector.payload) != expected:
            raise GWFFormatError(
                "raw FrVect payload has %d bytes; expected %d"
                % (len(vector.payload), expected)
            )
        jax = _require_jax_width(info.component_bytes)
        payload = _device_payload(jax, vector.payload)
        return _raw_kernel(
            vector.type_code, vector.n_data, compression.endian
        )(payload)
    if compression.kind == CompressionKind.ZERO_SUPPRESS:
        if compression.word_bytes not in (None, info.component_bytes):
            raise GWFFormatError(
                "zero-suppress word size does not match FrVect type"
            )
        component_count = vector.n_data * info.components
        word_bits = info.component_bytes * 8
        layout = _zero_layout(
            vector.payload, compression.endian, word_bits, component_count
        )
        jax = _require_jax_width(info.component_bytes)
        payload = _device_payload(jax, vector.payload[2:])
        return _zero_kernel(
            vector.type_code,
            vector.n_data,
            compression.endian,
            layout,
        )(payload)
    raise UnsupportedGWFCompression(
        "FrVect compression %s (code %d) is not supported"
        % (compression.kind.value, compression.code)
    )
