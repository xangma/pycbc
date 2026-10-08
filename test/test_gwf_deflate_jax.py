# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.


"""Optional CUDA codec integration: exact GWF bytes and replay boundaries."""

import builtins
import types
import zlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def decoder():
    from pycbc.frame import gwf_deflate_jax
    return gwf_deflate_jax.decompress_zlib


@pytest.fixture(scope="module")
def cuda_device():
    # Collection and framing-only tests must not initialize a JAX backend.
    pytest.importorskip("cuda_zlib", reason="cuda_zlib is optional")
    jax = pytest.importorskip("jax", reason="JAX is optional")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("CUDA JAX device unavailable")
    devices = [device for device in devices
               if "cuda" in str(device.client.platform_version).lower()]
    if not devices:
        pytest.skip("CUDA JAX device unavailable")
    device = devices[0]
    return device


@pytest.fixture
def forbid_cpu_inflation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fresh CUDA decoding attempted CPU zlib inflation")

    # Install before the first candidate call, not after cache preparation.
    monkeypatch.setattr(zlib, "decompress", forbidden)
    monkeypatch.setattr(zlib, "decompressobj", forbidden)


def _compressed(raw, level=6, strategy=zlib.Z_DEFAULT_STRATEGY,
                wbits=zlib.MAX_WBITS, flush=zlib.Z_FINISH):
    compressor = zlib.compressobj(level, zlib.DEFLATED, wbits, 8, strategy)
    return compressor.compress(raw) + compressor.flush(flush)


@pytest.fixture(scope="module")
def gwf_builders(cuda_device):
    # Reuse the independently checksummed GWF fixture builder only after the
    # CUDA availability gate; importing its test module during collection
    # would initialize JAX devices in otherwise CPU-only framing tests.
    import test_gwf_jax
    return test_gwf_jax


def _vector_record(builders, values, endian, version=8, wrapper="zlib"):
    raw = builders._file_bytes(values, endian)
    type_code = next(code for code, sample in builders._RAW_VALUES.items()
                     if sample.dtype == values.dtype)
    payload = _compressed(raw, wbits=15 if wrapper == "zlib" else 31)
    code = (257 if endian == "little" else 1) if version == 8 else (
        0x8002 if endian == "little" else 2)
    return builders._frvect(
        version, endian, payload, type_code, compression=code,
        shape=values.shape, n_data=len(values), spacing=0.5, origin=0.0,
        name="H1:TEST", checksum_type=1,
    )


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize("type_code", [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12])
def test_fresh_fr_vect_all_numeric_types_and_byte_orders(
    cuda_device, gwf_builders, forbid_cpu_inflation, endian, type_code,
):
    import jax
    from pycbc import scheme
    from pycbc.frame.gwf_deflate_jax import decode_fr_vect_deflate
    from pycbc.frame.gwf_jax import FrVectDescriptor, parse_gwf

    values = gwf_builders._RAW_VALUES[type_code]
    record = _vector_record(gwf_builders, values, endian, version=9)
    container = gwf_builders._checked_container(9, endian, record)
    vector = parse_gwf(container).vectors[0]
    assert isinstance(vector, FrVectDescriptor)
    with scheme.JAXScheme("cuda:%d" % cuda_device.id):
        actual = decode_fr_vect_deflate(vector, device=cuda_device)
        actual.block_until_ready()
    assert isinstance(actual, jax.Array)
    assert actual.devices() == {cuda_device}
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert np.asarray(actual).tobytes() == values.tobytes()


@pytest.mark.parametrize("change", ["checksum", "extent", "trailing"])
def test_admitted_zlib_errors_propagate_from_cuda(
    cuda_device, gwf_builders, forbid_cpu_inflation, change,
):
    """A valid GWF CRC must not hide corruption inside its zlib vector."""
    from pycbc.frame.gwf_deflate_jax import decode_fr_vect_deflate
    from pycbc.frame.gwf_jax import GWFFormatError, parse_gwf

    values = np.arange(8, dtype=np.float64)
    raw = values.tobytes()
    payload = zlib.compress(raw[:-8] if change == "extent" else raw)
    if change == "checksum":
        payload = payload[:-1] + bytes((payload[-1] ^ 1,))
    elif change == "trailing":
        payload += b"\0"
    record = gwf_builders._frvect(
        8, "little", payload, 2, compression=257, shape=values.shape,
        n_data=len(values), spacing=0.5, origin=0., checksum_type=1,
    )
    vector = parse_gwf(gwf_builders._checked_container(
        8, "little", record)).vectors[0]
    with pytest.raises(GWFFormatError):
        decode_fr_vect_deflate(vector, device=cuda_device)


def test_cpu_device_rejected_before_optional_backend(decoder, monkeypatch):
    from pycbc.frame.gwf_deflate_jax import GWFDeflateUnavailable

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "cupy" or name.startswith("jax"):
            pytest.fail("CPU admission loaded optional CUDA dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(GWFDeflateUnavailable, match="CUDA.*device"):
        decoder(_compressed(b"valid"), 5,
                types.SimpleNamespace(platform="cpu"))
