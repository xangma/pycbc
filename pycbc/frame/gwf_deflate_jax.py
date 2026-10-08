# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Optional cuda-zlib bridge for numeric GWF vectors.

The external codec owns CUDA compilation, allocation and stream validation.
Its JAX bridge returns independently owned device arrays. The replay reader
handles unavailable backends; malformed streams remain explicit errors.
"""

import numpy as np

from .gwf_jax import CompressionKind, GWFFormatError, _raw_kernel


class GWFDeflateUnavailable(NotImplementedError):
    """The optional CUDA zlib backend is unavailable or unsupported."""


def _backend():
    try:
        import cuda_zlib
        from cuda_zlib import jax as bridge
    except ImportError as exc:
        raise GWFDeflateUnavailable(
            "CUDA GWF decoding requires cuda-zlib with JAX support"
        ) from exc
    return cuda_zlib, bridge


def _device(device):
    if device is None:
        from pycbc import scheme
        device = getattr(scheme.mgr.state, "jax_device", None)
    return device


def decompress_zlib(payload, expected_bytes, device=None):
    """Decode to independent JAX storage using the installed byte codec."""
    # Avoid importing optional dependencies for an explicitly supplied CPU.
    if device is not None and getattr(device, "platform", None) != "gpu":
        raise GWFDeflateUnavailable("CUDA JAX device required")
    codec, bridge = _backend()
    try:
        device = _device(device)
        # The byte API admits sizes and framing before selecting a device;
        # the JAX convenience API requires a device before input validation.
        decode = (bridge.decompress_zlib if device is not None
                  else codec.decompress_zlib)
        return decode(payload, expected_bytes, device=device)
    except codec.CodecError as exc:
        raise GWFFormatError(str(exc)) from exc
    except codec.UnsupportedStream as exc:
        # Keep the existing GWF error for preset dictionaries.
        if isinstance(payload, bytes) and len(payload) > 1 and payload[1] & 32:
            raise GWFFormatError(str(exc)) from exc
        raise GWFDeflateUnavailable(str(exc)) from exc
    except codec.BackendUnavailable as exc:
        raise GWFDeflateUnavailable(str(exc)) from exc


def decode_fr_vect_deflate(vector, device=None):
    """Decode a CRC-admitted numeric zlib FrVect and convert its wire dtype."""
    if (vector.compression.kind != CompressionKind.GZIP
            or vector.type_code == 8):
        raise GWFDeflateUnavailable("native CUDA codec requires numeric zlib")
    raw = decompress_zlib(vector.payload,
                          vector.n_data * np.dtype(vector.dtype).itemsize,
                          device=device)
    result = _raw_kernel(vector.type_code, vector.n_data,
                         vector.compression.endian)(raw)
    return result
