# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""CPU checks for the optional cuda-zlib adapter and replay fallback."""

import builtins
import importlib.util
import sys
import types
import zlib

import numpy as np
import pytest

from pycbc.frame import gwf_deflate_jax as adapter
from pycbc.frame.gwf_jax import CompressionKind, GWFFormatError


@pytest.fixture
def fake_codec(monkeypatch):
    codec = types.ModuleType("cuda_zlib")
    bridge = types.ModuleType("cuda_zlib.jax")
    codec.CodecError = type("CodecError", (ValueError,), {})
    codec.BackendUnavailable = type("BackendUnavailable",
                                    (NotImplementedError,), {})
    codec.UnsupportedStream = type("UnsupportedStream",
                                   (NotImplementedError,), {})
    codec.jax = bridge
    monkeypatch.setitem(sys.modules, "cuda_zlib", codec)
    monkeypatch.setitem(sys.modules, "cuda_zlib.jax", bridge)
    return codec, bridge


@pytest.fixture
def cuda_device():
    return types.SimpleNamespace(
        platform="gpu", local_hardware_id=np.int64(3),
        client=types.SimpleNamespace(platform_version="CUDA 12"),
    )


@pytest.fixture
def forbid_cpu_inflation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("direct cuda-zlib adapter attempted CPU inflation")

    monkeypatch.setattr(zlib, "decompress", forbidden)
    monkeypatch.setattr(zlib, "decompressobj", forbidden)


def test_import_is_lazy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"cuda_zlib", "cupy", "jax"}:
            pytest.fail("adapter import loaded an optional backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location(
        "pycbc.frame._cuda_zlib_import_probe", adapter.__file__,
    )
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert callable(fresh.decompress_zlib)


@pytest.mark.parametrize("missing", ["package", "jax-bridge"])
def test_missing_dependency_is_unavailable_without_host_inflation(
    monkeypatch, fake_codec, cuda_device, forbid_cpu_inflation, missing,
):
    codec, _ = fake_codec
    if missing == "package":
        monkeypatch.setitem(sys.modules, "cuda_zlib", None)
    else:
        monkeypatch.delattr(codec, "jax")
        original = builtins.__import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cuda_zlib" and fromlist and "jax" in fromlist:
                raise ModuleNotFoundError("No module named 'cuda_zlib.jax'",
                                          name="cuda_zlib.jax")
            return original(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(adapter.GWFDeflateUnavailable, match="cuda-zlib"):
        adapter.decompress_zlib(zlib.compress(b"valid"), 5, cuda_device)


def test_forwards_original_bytes_extent_device_and_owned_result(
    fake_codec, cuda_device, forbid_cpu_inflation,
):
    _, bridge = fake_codec
    payload = zlib.compress(b"exact original bytes")
    result = object()
    calls = []

    def decode(data, expected_bytes, *, device):
        calls.append((data, expected_bytes, device))
        return result

    bridge.decompress_zlib = decode
    assert adapter.decompress_zlib(payload, 20, cuda_device) is result
    assert len(calls) == 1
    assert calls[0][0] is payload
    assert calls[0][1:] == (20, cuda_device)


def test_numeric_conversion_does_not_fence_the_pipeline(monkeypatch):
    class PendingConversion:
        def block_until_ready(self):
            pytest.fail('wire conversion inserted a device fence')

    result, raw = PendingConversion(), object()
    vector = types.SimpleNamespace(
        compression=types.SimpleNamespace(kind=CompressionKind.GZIP,
                                          endian='little'),
        type_code=1, n_data=4, dtype=np.dtype('int16'), payload=b'admitted')
    monkeypatch.setattr(adapter, 'decompress_zlib', lambda *a, **k: raw)
    monkeypatch.setattr(adapter, '_raw_kernel',
                        lambda *a: lambda data: (
                            result if data is raw else None))
    assert adapter.decode_fr_vect_deflate(vector, device=object()) is result


@pytest.mark.parametrize("error, expected", [
    ("CodecError", GWFFormatError),
    ("BackendUnavailable", adapter.GWFDeflateUnavailable),
    ("UnsupportedStream", adapter.GWFDeflateUnavailable),
    ("RuntimeError", RuntimeError),
])
def test_codec_errors_preserve_failure_boundary(
    fake_codec, cuda_device, forbid_cpu_inflation, error, expected,
):
    codec, bridge = fake_codec
    failure = (RuntimeError if error == "RuntimeError"
               else getattr(codec, error))("failure from codec")

    def decode(*args, **kwargs):
        raise failure

    bridge.decompress_zlib = decode
    with pytest.raises(expected, match="failure from codec") as caught:
        adapter.decompress_zlib(zlib.compress(b"test"), 4, cuda_device)
    if error == "RuntimeError":
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure


def test_preset_dictionary_remains_format_error(fake_codec, cuda_device):
    codec, bridge = fake_codec
    compressor = zlib.compressobj(zdict=b"dictionary")
    payload = compressor.compress(b"dictionary test") + compressor.flush()
    assert payload[1] & 32

    def decode(*args, **kwargs):
        raise codec.UnsupportedStream("preset dictionary unsupported")

    bridge.decompress_zlib = decode
    with pytest.raises(GWFFormatError, match="preset dictionary"):
        adapter.decompress_zlib(payload, 15, cuda_device)


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize("type_code, values", [
    (1, np.array([-32768, -2, 0, 32767], dtype=np.int16)),
    (2, np.array([-0., np.nextafter(0., 1.), np.inf, -np.inf],
                 dtype=np.float64)),
    (7, np.array([1 + 2j, -3 + 4j, 0j, 5 - 6j], dtype=np.complex128)),
])
def test_numeric_wire_conversion(
    fake_codec, cuda_device, forbid_cpu_inflation, endian, type_code, values,
):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pycbc import scheme

    _, bridge = fake_codec
    raw = values.astype(values.dtype.newbyteorder(
        "<" if endian == "little" else ">"), copy=False).tobytes()
    payload = zlib.compress(raw)
    calls = []

    def decode(data, extent, *, device):
        calls.append((data, extent, device))
        return jax.numpy.asarray(np.frombuffer(raw, dtype=np.uint8))

    bridge.decompress_zlib = decode
    vector = types.SimpleNamespace(
        compression=types.SimpleNamespace(kind=CompressionKind.GZIP,
                                          endian=endian),
        type_code=type_code, dtype=values.dtype,
        n_data=len(values), payload=payload,
    )
    with scheme.JAXScheme("cpu"):
        actual = adapter.decode_fr_vect_deflate(vector, cuda_device)
    assert actual.dtype == values.dtype
    assert actual.shape == values.shape
    assert np.asarray(actual).tobytes() == values.tobytes()
    assert calls == [(payload, values.nbytes, cuda_device)]


