# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Regression tests for resident JAX single-trigger ring appends."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

import pycbc
from pycbc import scheme
from pycbc.events import coinc, coinc_jax

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


def _legacy_add(monkeypatch, buffer, ids, columns):
    """Run the retained eager implementation as the exact JAX oracle."""
    with monkeypatch.context() as patch:
        patch.setattr(coinc_jax, "_can_append_singles_columns",
                      lambda *_args: False)
        buffer.add(ids, columns)


def _assert_column(actual, expected):
    got, want = np.asarray(actual), np.asarray(expected)
    assert got.dtype == want.dtype and got.shape == want.shape
    if got.dtype.kind == "O":
        np.testing.assert_equal(got.tolist(), want.tolist())
    else:
        assert got.tobytes() == want.tobytes()
    if isinstance(expected, jax.Array):
        assert isinstance(actual, jax.Array)
        assert actual.devices() == expected.devices()
    else:
        assert isinstance(actual, np.ndarray)


def _assert_state(actual, expected):
    assert (actual.time, actual.filled_time, actual.valid_ends, actual.nbytes) == (
        expected.time, expected.filled_time, expected.valid_ends, expected.nbytes)
    for got, want in zip(actual.buffer, expected.buffer):
        if want is None:
            assert got is None
        else:
            assert list(got) == list(want)
            for key in want:
                _assert_column(got[key], want[key])
    for got, want in zip(actual.buffer_expire, expected.buffer_expire):
        _assert_column(got, want)
    for got, want in zip(actual._expire_times, expected._expire_times):
        _assert_column(got, want)


@pytest.mark.parametrize("max_time", [0, 1, 2])
@pytest.mark.parametrize("x64", [False, True])
def test_ring_append_matches_full_eager_state(monkeypatch, device, max_time, x64):
    """Ordered duplicates, pruning and backout retain every typed column."""
    tiny = np.nextafter(np.float32(0), np.float32(1))
    with scheme.JAXScheme(device), enable_x64(x64):
        actual = coinc_jax.JAXMultiRingBuffer(4, max_time)
        expected = coinc_jax.JAXMultiRingBuffer(4, max_time)
        snapshots = []
        for block, ids in enumerate(([2, 0, 2, 1], [1, 2, 0, 2], [], [2, 0, 2])):
            ids = np.asarray(ids, dtype=np.int32)
            count = ids.size
            host = {
                "snr": np.resize(np.array([-0., tiny, -tiny, np.nan], "f4"), count),
                "metadata": np.array([f"{block}/{i}" if i % 2 else np.nan
                                      for i in range(count)], dtype=object),
                "time": 1e9 + (block * 10 + np.arange(count)) / 1024,
                "event_id": np.arange(count, dtype=np.uint64) + 2**53 + 1,
                "vector": np.arange(count * 2, dtype=np.int16).reshape(count, 2),
                "complex": np.resize(np.array([-0. + 1j, np.nan + 0j], "c8"), count),
            }
            columns = {key: value if key == "metadata" else jnp.asarray(value)
                       for key, value in host.items()}
            actual.add(ids, columns)
            _legacy_add(monkeypatch, expected, ids, columns)
            _assert_state(actual, expected)
            for ring in range(4):
                got, want = actual.data(ring), expected.data(ring)
                assert list(got) == list(want)
                _assert_column(actual.expire_vector(ring), expected.expire_vector(ring))
            _assert_state(actual, expected)
            if block == 1:
                snapshots = [(row, {key: np.asarray(value).copy()
                                    for key, value in row.items()})
                             for row in actual.buffer if row is not None]
            if block == 3:
                actual.discard_last([2, 2, 0])
                expected.discard_last([2, 2, 0])
                _assert_state(actual, expected)
        for row, saved in snapshots:
            for key, want in saved.items():
                got = np.asarray(row[key])
                if got.dtype.kind == "O":
                    np.testing.assert_equal(got.tolist(), want.tolist())
                else:
                    assert got.tobytes() == want.tobytes()


def test_ring_fused_dispatch_and_large_incoming_vector(monkeypatch, device):
    """Select all numeric columns once per group, including production sizes."""
    calls = []
    gathers = []
    core = coinc_jax._append_selected_singles_groups
    gather = coinc_jax._gather_singles_column_groups

    def counted(selected, previous, expiries, clock):
        calls.append((len(selected), tuple(len(row) for row in previous),
                      tuple(row[0].size for row in selected), int(clock)))
        return core(selected, previous, expiries, clock)

    def counted_gather(columns, indices, *, counts):
        gathers.append((len(columns), indices.size, counts))
        return gather(columns, indices, counts=counts)

    def reject_serial(*_args):
        raise AssertionError("eligible ring group dispatched per-ring append")

    monkeypatch.setattr(coinc_jax, "_append_selected_singles_groups", counted)
    monkeypatch.setattr(coinc_jax, "_gather_singles_column_groups", counted_gather)
    monkeypatch.setattr(coinc_jax, "_append_singles_columns", reject_serial)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(1024, 20)
        ids = np.tile(np.array([2, 0, 1], np.int32), 44)
        columns = {"snr": jnp.arange(132, dtype=jnp.float32),
                   "time": jnp.arange(132, dtype=jnp.float64),
                   "metadata": np.array(["TaylorF2"] * 132)}
        actual.add(ids, columns)
        actual.add(ids, columns)
        assert calls == [(3, (0, 0, 0), (44, 44, 44), 0),
                         (3, (2, 2, 2), (44, 44, 44), 1)]
        assert gathers == [(2, 132, (44, 44, 44))] * 2
        for ring in (2, 0, 1):
            assert actual.buffer[ring]["snr"].size == 88
            positions = np.flatnonzero(ids == ring)
            for key in columns:
                expected = np.concatenate((np.asarray(columns[key])[positions],) * 2)
                _assert_column(actual.buffer[ring][key],
                               jnp.asarray(expected) if key != "metadata" else expected)
            np.testing.assert_array_equal(actual.buffer_expire[ring],
                                          np.repeat([0, 1], 44))
        assert actual.buffer[3] is None
        assert actual.buffer_expire[3] is actual.buffer_expire[4]


def test_ring_kernel_reuses_dynamic_clock_and_selection(monkeypatch, device):
    """Incoming batch sizes do not multiply selected-append specializations."""
    traces = []
    gather_traces = []
    gather = coinc_jax._gather_singles_columns.__wrapped__
    append = coinc_jax._append_selected_singles_columns.__wrapped__

    def traced_append(*args):
        traces.append(1)
        return append(*args)

    def traced_gather(*args):
        gather_traces.append(args[0][0].shape[0])
        return gather(*args)

    monkeypatch.setattr(coinc_jax, "_gather_singles_columns",
                        jax.jit(traced_gather))
    monkeypatch.setattr(coinc_jax, "_append_selected_singles_columns",
                        jax.jit(traced_append))
    with scheme.JAXScheme(device):
        dtypes = (jnp.float32,) * 11 + (jnp.float64,) * 4 + (jnp.int32,) * 2
        previous = tuple(jnp.array([6], dtype=dtype) for dtype in dtypes)
        expiry = jnp.array([0], dtype=jnp.int32)
        incoming_sizes = (2, 3, 8, 32, 64, 123, 132)
        for incoming, clock in zip(incoming_sizes, (1, 7, -3, 0, 4, 2, -1)):
            values = tuple(jnp.arange(incoming, dtype=dtype) for dtype in dtypes)
            idx = jnp.array([incoming - 1, 0], dtype=jnp.int64)
            result, clocks = coinc_jax._append_singles_columns(
                values, previous, expiry, idx, np.int32(clock))
            for actual, old, value in zip(result, previous, values):
                expected = jnp.concatenate((old, value[idx]))
                _assert_column(actual, expected)
            np.testing.assert_array_equal(np.asarray(clocks), [0, clock, clock])
        assert len(traces) == 1
        assert gather_traces == list(incoming_sizes)


def test_ring_resident_append_does_not_read_scientific_arrays(monkeypatch, device):
    """Only host insertion metadata crosses the append control boundary."""
    with scheme.JAXScheme(device):
        columns = {"snr": jnp.arange(4, dtype=jnp.float32),
                   "time": jnp.arange(4, dtype=jnp.float64)}
        actual = coinc_jax.JAXMultiRingBuffer(2, 20)
        expected = coinc_jax.JAXMultiRingBuffer(2, 20)
        # Compile both the empty-prior and nonempty-prior shapes first.
        actual.add([0, 1, 0, 1], columns)
        actual.add([0, 1, 0, 1], columns)
        expected.add([0, 1, 0, 1], columns)

        def reject_read(*_args, **_kwargs):
            raise AssertionError("resident scientific column read by the host")

        with monkeypatch.context() as patch:
            patch.setattr(type(columns["snr"]), "__array__", reject_read)
            expected.add([0, 1, 0, 1], columns)
        _assert_state(actual, expected)


def test_ring_int32_clock_rollover_matches_eager(monkeypatch, device):
    """The optimized in-range clock and overflow fallback retain int32 bits."""
    with scheme.JAXScheme(device):
        for clock in (np.iinfo(np.int32).min, np.iinfo(np.int32).max):
            actual = coinc_jax.JAXMultiRingBuffer(1, 20)
            expected = coinc_jax.JAXMultiRingBuffer(1, 20)
            actual.time = expected.time = int(clock)
            columns = {"stat": jnp.array([-0.], dtype=jnp.float32)}
            for _ in range(2):
                actual.add([0], columns)
                _legacy_add(monkeypatch, expected, [0], columns)
                _assert_state(actual, expected)


def test_ring_numeric_dtype_promotion_keeps_eager_boundaries(monkeypatch, device):
    """Mixed dtype concatenation retains subnormal and integer cast behavior."""
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 10)
        expected = coinc_jax.JAXMultiRingBuffer(2, 10)
        for dtype in (np.float32, np.float64, np.uint64):
            value = (np.nextafter(np.float32(0), np.float32(1))
                     if dtype == np.float32 else 2**53 + 1)
            columns = {"stat": jnp.asarray([value], dtype=dtype)}
            actual.add([0], columns)
            _legacy_add(monkeypatch, expected, [0], columns)
            _assert_state(actual, expected)


@pytest.mark.parametrize("clock", [.5, 2**31 + 1, -2**31 - 1])
def test_ring_unusual_clocks_use_eager_fallback(monkeypatch, device, clock):
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 10)
        expected = coinc_jax.JAXMultiRingBuffer(2, 10)
        actual.time = expected.time = clock
        columns = {"snr": jnp.array([1.], dtype=jnp.float32)}

        def reject_core(*_args):
            raise AssertionError("unusual clock reached fused append")

        monkeypatch.setattr(coinc_jax, "_append_singles_columns", reject_core)
        actual.add([0], columns)
        _legacy_add(monkeypatch, expected, [0], columns)
        _assert_state(actual, expected)


def test_ring_empty_omitted_keys_and_error_state(monkeypatch, device):
    """None rings, omitted fields and failures keep old mutation ordering."""
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(3, 1)
        expected = coinc_jax.JAXMultiRingBuffer(3, 1)
        for ids, columns in (([0, 1], {"a": jnp.arange(2), "b": jnp.arange(2)}),
                             ([], {"unused": object()}),
                             ([0], {"a": jnp.array([7])})):
            actual.add(ids, columns)
            _legacy_add(monkeypatch, expected, ids, columns)
            _assert_state(actual, expected)
        assert list(actual.buffer[0]) == ["a"]
        for ids, columns in (([0], {"missing": jnp.array([3])}),
                             ([5], {"a": jnp.array([3])}),
                             ([0], {"a": jnp.array(3)}),
                             ([0, 0], {"metadata": np.array(["one"]),
                                       "a": jnp.arange(2)})):
            with pytest.raises(Exception) as legacy:
                _legacy_add(monkeypatch, expected, ids, columns)
            with pytest.raises(type(legacy.value)):
                actual.add(ids, columns)
            _assert_state(actual, expected)


def test_ring_failure_preserves_earlier_ring_updates(monkeypatch, device):
    """A later missing field retains earlier appends and the unadvanced clock."""
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 20)
        expected = coinc_jax.JAXMultiRingBuffer(2, 20)
        for ids, columns in (([0], {"a": jnp.array([1])}),
                             ([1], {"other": jnp.array([2])})):
            actual.add(ids, columns)
            _legacy_add(monkeypatch, expected, ids, columns)
        columns = {"a": jnp.array([3, 4])}
        with pytest.raises(KeyError):
            actual.add([0, 1], columns)
        with pytest.raises(KeyError):
            _legacy_add(monkeypatch, expected, [0, 1], columns)
        _assert_state(actual, expected)
        assert actual.time == 2 and actual.valid_ends == [2, 1]


def test_ring_bounded_admission_preserves_generic_paths(monkeypatch, device):
    """Wide or unusually stored rings retain the original implementation."""
    with scheme.JAXScheme(device):
        for columns in ({str(i): jnp.ones(2) for i in range(33)},
                        {"a": jnp.ones(4097)}):
            actual = coinc_jax.JAXMultiRingBuffer(2, 10)
            expected = coinc_jax.JAXMultiRingBuffer(2, 10)

            def reject_core(*_args):
                raise AssertionError("unbounded input reached fused append")

            with monkeypatch.context() as patch:
                patch.setattr(coinc_jax, "_append_singles_columns", reject_core)
                actual.add([0, 1], columns)
                _legacy_add(monkeypatch, expected, [0, 1], columns)
            _assert_state(actual, expected)


def test_ring_append_matches_native_expiration(device):
    """The optimized device path retains native active rows and clocks."""
    dtype = np.dtype([("snr", "f4"), ("time", "f8"), ("metadata", object)])
    native = coinc.MultiRingBuffer(3, 2, dtype)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(3, 2)
        for block, ids in enumerate(([2, 0, 2], [1, 2], [], [0, 2])):
            rows = np.empty(len(ids), dtype=dtype)
            rows["snr"] = np.arange(len(ids)) + block
            rows["time"] = 1e9 + block + np.arange(len(ids)) / 1024
            rows["metadata"] = "TaylorF2"
            native.add(np.asarray(ids), rows)
            actual.add(ids, {key: rows[key] for key in dtype.names})
            for ring in range(3):
                want, got = native.data(ring), actual.data(ring)
                # An untouched JAX ring retains its public empty mapping.
                if not got:
                    assert not len(want)
                    continue
                for key in dtype.names:
                    _assert_column(got[key], jnp.asarray(want[key])
                                   if key != "metadata" else want[key])
                np.testing.assert_array_equal(np.asarray(actual.expire_vector(ring)),
                                              native.expire_vector(ring))
