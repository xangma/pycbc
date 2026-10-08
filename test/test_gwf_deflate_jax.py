# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.


"""Optional CUDA codec integration: exact GWF bytes and replay boundaries."""

import builtins
import gc
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


def _fresh_frame(tmp_path, builders, values, start, endian="little",
                 wrapper="zlib"):
    assert len(values) % 2 == 0
    record = _vector_record(builders, values, endian, wrapper=wrapper)
    path = tmp_path / ("H-TEST-%d-%d.gwf" % (start, len(values) // 2))
    path.write_bytes(builders._replay_container(
        8, endian, record, start=start, duration=len(values) / 2,
        channel="H1:TEST"))
    return str(path)


def _assert_series(series, values, start, device):
    from pycbc.types.array_jax import to_jax

    actual = to_jax(series)
    actual.block_until_ready()
    assert actual.devices() == {device}
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert float(series.start_time) == start
    assert float(series.end_time) == start + len(values) / 2
    assert series.sample_rate == 2
    assert np.asarray(actual).tobytes() == values.tobytes()


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


@pytest.mark.parametrize("endian", ["little", "big"])
def test_fresh_deflate_replay_full_sliced_joined_and_retained_reads(
    tmp_path, monkeypatch, cuda_device, gwf_builders, forbid_cpu_inflation,
    endian,
):
    from pycbc import scheme
    from pycbc.frame import gwf_replay_jax as replay

    # Include signed zero, a subnormal and infinities in the bytewise oracle.
    first = np.array([-0.0, np.nextafter(0., 1.), np.inf, -np.inf,
                      1.25, -1.25, np.pi, -np.pi], dtype=np.float64)
    second = np.arange(8, 16, dtype=np.float64)
    paths = [_fresh_frame(tmp_path, gwf_builders, first, 100, endian),
             _fresh_frame(tmp_path, gwf_builders, second, 104, endian)]
    monkeypatch.setattr(replay, "_host_decompress", lambda *args: pytest.fail(
        "fresh Deflate replay attempted host decompression"))
    with scheme.JAXScheme("cuda:%d" % cuda_device.id):
        source = replay.GWFReplaySource(paths, "H1:TEST", 2,
                                        cuda_codec="deflate")
        full = source.read(100, 4)
        sliced = source.read(100.5, 1.5)
        joined = source.read(103, 3)
        final = source.read(106, 2)
        del source
        gc.collect()
    _assert_series(full, first, 100, cuda_device)
    _assert_series(sliced, first[1:4], 100.5, cuda_device)
    _assert_series(joined, np.concatenate((first[6:], second[:4])),
                   103, cuda_device)
    _assert_series(final, second[4:], 106, cuda_device)


def test_fresh_deflate_replay_advance_keeps_device_blocks_and_epochs(
    tmp_path, monkeypatch, cuda_device, gwf_builders, forbid_cpu_inflation,
):
    from pycbc import scheme
    from pycbc.frame import gwf_replay_jax as replay
    from pycbc.frame.frame_jax import JAXReplayFrameReader
    from pycbc.types import TimeSeries

    values = np.arange(16, dtype=np.float64)
    paths = [_fresh_frame(tmp_path, gwf_builders, values[:8], 100),
             _fresh_frame(tmp_path, gwf_builders, values[8:], 104)]
    monkeypatch.setenv("PYCBC_GWF_CUDA_DECOMPRESS", "deflate")
    monkeypatch.setattr(replay, "_host_decompress", lambda *args: pytest.fail(
        "fresh Deflate reader attempted host decompression"))
    with scheme.JAXScheme("cuda:%d" % cuda_device.id):
        buffer = types.SimpleNamespace(
            frame_src=paths, channel_name="H1:TEST", raw_sample_rate=2,
            read_pos=100.,
            raw_buffer=TimeSeries(np.zeros(8), delta_t=0.5, epoch=96),
            _read_frame=lambda duration: pytest.fail("compatibility fallback"),
        )
        reader = JAXReplayFrameReader(108, 2, read_ahead_seconds=6)
        blocks = [reader.advance(buffer, 2) for _ in range(4)]
        del reader
        gc.collect()
    assert buffer.read_pos == 108
    _assert_series(buffer.raw_buffer, values[8:], 104, cuda_device)
    for index, block in enumerate(blocks):
        _assert_series(block, values[index * 4:(index + 1) * 4],
                       100 + index * 2, cuda_device)


def test_rfc1952_gzip_vector_is_explicitly_unsupported_without_fallback(
    tmp_path, monkeypatch, cuda_device, gwf_builders, forbid_cpu_inflation,
):
    from pycbc import scheme
    from pycbc.frame import gwf_replay_jax as replay
    from pycbc.frame.gwf_deflate_jax import (
        GWFDeflateUnavailable, decode_fr_vect_deflate,
    )
    from pycbc.frame.gwf_jax import parse_gwf

    values = np.arange(8, dtype=np.float64)
    monkeypatch.setattr(replay, "_host_decompress", lambda *args: pytest.fail(
        "unsupported GZIP wrapper fell back to host decompression"))
    with scheme.JAXScheme("cuda:%d" % cuda_device.id):
        with pytest.raises(GWFDeflateUnavailable, match="zlib|RFC 1950"):
            record = _vector_record(gwf_builders, values, "little",
                                    wrapper="gzip")
            vector = parse_gwf(gwf_builders._checked_container(
                8, "little", record)).vectors[0]
            decode_fr_vect_deflate(vector, device=cuda_device)


def test_rfc1952_gzip_replay_uses_cached_host_fallback(
    tmp_path, monkeypatch, cuda_device, gwf_builders,
):
    from pycbc import scheme
    from pycbc.frame import gwf_replay_jax as replay

    values = np.arange(8, dtype=np.float64)
    path = _fresh_frame(tmp_path, gwf_builders, values, 100, wrapper="gzip")
    calls = []
    original = replay._host_decompress

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(replay, "_host_decompress", decode)
    with scheme.JAXScheme("cuda:%d" % cuda_device.id):
        source = replay.GWFReplaySource(path, "H1:TEST", 2)
        full = source.read(100, 4)
        sliced = source.read(101, 2)
    assert len(calls) == 1
    _assert_series(full, values, 100, cuda_device)
    _assert_series(sliced, values[2:6], 101, cuda_device)


@pytest.fixture
def cpu_replay():
    pytest.importorskip("jax", reason="JAX is optional")
    from pycbc import scheme
    from pycbc.frame import gwf_replay_jax
    import test_gwf_jax

    with scheme.JAXScheme("cpu"):
        yield gwf_replay_jax, test_gwf_jax


def test_host_replay_does_not_load_optional_cuda(tmp_path, monkeypatch,
                                                 cpu_replay):
    replay, builders = cpu_replay
    values = np.arange(8, dtype=np.float64)
    path = _fresh_frame(tmp_path, builders, values, 100)
    monkeypatch.delenv("PYCBC_GWF_CUDA_DECOMPRESS", raising=False)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"cuda_zlib", "cupy"} \
                or name.startswith("nvidia") \
                or name.endswith("gwf_deflate_jax"):
            pytest.fail("host replay loaded an optional CUDA decoder")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    got = replay.GWFReplaySource(path, "H1:TEST", 2,
                                 cuda_codec="host").read(100, 4)
    assert np.asarray(got).tobytes() == values.tobytes()


@pytest.mark.parametrize("codec", ["lookahead", "unknown", ""])
def test_replay_rejects_unsupported_codec(tmp_path, cpu_replay, codec):
    replay, builders = cpu_replay
    path = _fresh_frame(tmp_path, builders, np.arange(8.), 100)
    with pytest.raises(ValueError, match="host or deflate"):
        replay.GWFReplaySource(path, "H1:TEST", 2, cuda_codec=codec)


@pytest.mark.parametrize("error", ["corrupt", "runtime"])
def test_explicit_deflate_errors_propagate_without_host_decode(
    tmp_path, monkeypatch, cpu_replay, error,
):
    replay, builders = cpu_replay
    from pycbc.frame import gwf_deflate_jax as cuda
    from pycbc.frame.gwf_jax import GWFFormatError

    path = _fresh_frame(tmp_path, builders, np.arange(8.), 100)
    failure = {"corrupt": GWFFormatError("bad checksum"),
               "runtime": RuntimeError("decoder failed")}[error]

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cuda, "decode_fr_vect_deflate", fail)
    monkeypatch.setattr(replay, "_host_decompress", lambda *args: pytest.fail(
        "explicit CUDA decode fell back to host inflation"))
    source = replay.GWFReplaySource(path, "H1:TEST", 2, cuda_codec="deflate")
    with pytest.raises(type(failure), match=str(failure)):
        source.read(100, 4)
    assert source._cuda_cache is None


def test_replay_decodes_each_vector_once_and_retains_old_spans(
    tmp_path, monkeypatch, cpu_replay,
):
    replay, builders = cpu_replay
    import jax.numpy as jnp
    from pycbc.frame import gwf_deflate_jax as cuda

    first = np.arange(8.)
    second = np.arange(10., 22.)
    paths = [_fresh_frame(tmp_path, builders, first, 100),
             _fresh_frame(tmp_path, builders, second, 104)]
    calls = []

    def decode(vector, **kwargs):
        calls.append(vector)
        return jnp.asarray(replay._host_decompress(vector))

    monkeypatch.setattr(cuda, "decode_fr_vect_deflate", decode)
    source = replay.GWFReplaySource(paths, "H1:TEST", 2, cuda_codec="deflate")
    retained = [source.read(100, 2), source.read(102, 2)]
    crossed = source.read(103, 3)
    later = source.read(108, 2)
    del source
    gc.collect()
    assert len(calls) == len({id(vector) for vector in calls}) == 2
    np.testing.assert_array_equal(np.asarray(retained[0]), first[:4])
    np.testing.assert_array_equal(np.asarray(retained[1]), first[4:])
    np.testing.assert_array_equal(np.asarray(crossed), [6, 7, 10, 11, 12, 13])
    np.testing.assert_array_equal(np.asarray(later), second[-4:])


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
