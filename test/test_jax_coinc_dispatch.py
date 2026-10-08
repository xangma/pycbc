# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Exact state and executable reuse of grouped live coincidence dispatch."""

from contextlib import contextmanager, nullcontext

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc_jax
from jax_test_helpers import CompilationGuard, COUNTERS

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


def _assert_exact(actual, expected):
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_exact(got, want)
    elif expected is None:
        assert actual is None
    else:
        got, want = np.asarray(actual), np.asarray(expected)
        assert got.dtype == want.dtype and got.shape == want.shape
        if want.dtype.kind == "O":
            np.testing.assert_equal(got.tolist(), want.tolist())
        else:
            assert got.tobytes() == want.tobytes()
        if isinstance(expected, jax.Array):
            assert isinstance(actual, jax.Array)
            assert actual.devices() == expected.devices()


def _ring_state(buffer):
    try:
        nbytes = buffer.nbytes
    except (AttributeError, TypeError) as error:
        nbytes = (type(error).__name__, str(error))
    return {"time": buffer.time, "filled_time": buffer.filled_time,
            "valid_ends": buffer.valid_ends.copy(), "nbytes": nbytes,
            "rows": [dict(row) if isinstance(row, dict) else row
                     for row in buffer.buffer],
            "expiry": list(buffer.buffer_expire),
            "host_clocks": [value.copy() for value in buffer._expire_times]}


@contextmanager
def _serial(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(coinc_jax, "_can_append_singles_columns", lambda *_args: False)
        yield


def _columns(count, block=0):
    # Seventeen mixed scientific columns, including payload bits which must
    # survive a pure gather/append without floating point arithmetic.
    floats = np.resize(np.array([0x80000000, 1, 0x7fc00017, 0x3f800000], "u4")
                       .view("f4"), count)
    columns = {"f32": jnp.asarray(floats),
               "time": jnp.asarray(1e9 + (block * 100 + np.arange(count)) / 1024),
               "id": jnp.asarray(np.arange(count, dtype="u8") + 2**53 + 1),
               "complex": jnp.asarray(floats.astype("c8")),
               "vector": jnp.asarray(np.arange(count * 2, dtype="i2")
                                      .reshape(count, 2))}
    columns.update({f"field{i}": jnp.asarray(np.arange(count, dtype="i4") + i)
                    for i in range(12)})
    columns["metadata"] = np.array([f"{block}:{i}" if i % 2 else None
                                     for i in range(count)], dtype=object)
    return columns


@pytest.mark.parametrize('profiling,fail_dispatch',
                         [(False, False), (True, False), (True, True)])
def test_grouped_singles_stage_markers_preserve_optional_dispatch(
        monkeypatch, device, profiling, fail_dispatch):
    """Group-level markers balance failures and never fence device work."""
    from pycbc import benchmark

    records = []

    def emit(stage, event, **metadata):
        assert profiling, 'disabled profiling reached the stage emitter'
        records.append((stage, event, metadata))

    def fail(*_args, **_kwargs):
        raise RuntimeError('append dispatch failed')

    monkeypatch.delenv('PYCBC_BENCHMARK_STAGES', raising=False)
    monkeypatch.setattr(benchmark, 'stage_event', emit)
    with scheme.JAXScheme(device):
        buffer = coinc_jax.JAXMultiRingBuffer(2, 2)
        values = jnp.array([1., 2.], dtype=jnp.float32)
        if profiling:
            monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
        if fail_dispatch:
            monkeypatch.setattr(coinc_jax, '_gather_singles_column_groups', fail)
        context = (pytest.raises(RuntimeError, match='append dispatch failed')
                   if fail_dispatch else nullcontext())
        with context:
            prepared = coinc_jax._prepare_singles_ring_group(
                buffer, ((0, [0]), (1, [1])), {'snr': values}, ('snr',),
                (values,), next(iter(values.devices())))
            assert list(prepared) == [0, 1]
    if profiling:
        assert len(records) == 8
        assert [stage for stage, event, _ in records if event == 'start'] == [
            'coinc_singles_group_host_prepare',
            'coinc_singles_group_index_dispatch',
            'coinc_singles_group_host_prepare',
            'coinc_singles_group_append_dispatch']
        for start, end in zip(records[::2], records[1::2]):
            assert start[0] == end[0]
            assert (start[1], end[1]) == ('start', 'end')
            assert start[2] == end[2]
            assert start[2]['synchronize'] is False
    else:
        assert records == []


def test_grouped_rings_preserve_all_state_and_old_snapshots(monkeypatch, device):
    """More than one dispatch group retains ring order, expiry and backout."""
    gather = coinc_jax._gather_singles_column_groups
    dispatches = []

    def record(columns, indices, *, counts):
        dispatches.append(counts)
        return gather(columns, indices, counts=counts)

    monkeypatch.setattr(coinc_jax, "_gather_singles_column_groups", record)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(18, 2)
        expected = coinc_jax.JAXMultiRingBuffer(18, 2)
        snapshots = []
        schedules = ([9, 0, 1, 2, 9, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
                      15, 16, 17, 0], [17, 0, 9, 9, 3, 6, 1], [],
                     [9, 0, 1, 2, 9, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
                      15, 16, 17, 0])
        for block, ids in enumerate(schedules):
            columns = _columns(len(ids), block)
            actual.add(np.asarray(ids, "i4"), columns)
            with _serial(monkeypatch):
                expected.add(np.asarray(ids, "i4"), columns)
            _assert_exact(_ring_state(actual), _ring_state(expected))
            if block == 0:
                snapshots = [(dict(row), {key: np.asarray(value).copy()
                                          for key, value in row.items()})
                             for row in actual.buffer if row is not None]
            for ring in range(actual.num_rings):
                _assert_exact(actual.data(ring), expected.data(ring))
                _assert_exact(actual.expire_vector(ring),
                              expected.expire_vector(ring))
            _assert_exact(_ring_state(actual), _ring_state(expected))
        actual.discard_last(np.array([9, 9, 0], "i4"))
        expected.discard_last(np.array([9, 9, 0], "i4"))
        _assert_exact(_ring_state(actual), _ring_state(expected))
        for row, saved in snapshots:
            for key in saved:
                _assert_exact(np.asarray(row[key]), saved[key])
    assert [len(counts) for counts in dispatches] == [8, 8, 2, 6, 8, 8, 2]
    assert dispatches[0] == (2, 2, 1, 1, 1, 1, 1, 1)


def test_grouped_metadata_failure_keeps_serial_partial_state(monkeypatch, device):
    """A later ring's metadata error must leave earlier ring commits intact."""
    append = coinc_jax._append_selected_singles_groups
    dispatches = []

    def record(*args):
        dispatches.append(len(args[0]))
        return append(*args)

    monkeypatch.setattr(coinc_jax, "_append_selected_singles_groups", record)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(3, 10)
        expected = coinc_jax.JAXMultiRingBuffer(3, 10)
        metadata = np.array([(1, 2)], dtype=[("id", "i4"), ("tag", "i4")])
        incompatible = np.array([(1, 2)],
                                dtype=[("id", "i4"), ("other", "i4")])
        for buffer in (actual, expected):
            buffer.add(np.array([0]), {"snr": jnp.array([1.], dtype=jnp.float32),
                                      "metadata": metadata})
            buffer.add(np.array([1]), {"snr": jnp.array([2.], dtype=jnp.float32),
                                      "metadata": incompatible})
        columns = {"snr": jnp.array([3., 4., 5.], dtype=jnp.float32),
                   "metadata": np.repeat(metadata, 3)}
        for buffer, context in ((actual, nullcontext()),
                                (expected, _serial(monkeypatch))):
            with context:
                with pytest.raises(TypeError, match="field names"):
                    buffer.add(np.array([0, 1, 2]), columns)
        _assert_exact(_ring_state(actual), _ring_state(expected))
        assert actual.valid_ends == [2, 1, 0]
        assert actual.time == 2 and actual.buffer[2] is None
    assert dispatches == [3]


def test_later_malformed_ring_keeps_serial_partial_state(monkeypatch, device):
    """Admission must not raise for a future row before earlier commits."""
    def reject_group(*_args, **_kwargs):
        raise AssertionError("malformed ring storage reached grouped dispatch")

    monkeypatch.setattr(coinc_jax, "_gather_singles_column_groups", reject_group)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 2)
        expected = coinc_jax.JAXMultiRingBuffer(2, 2)
        actual.buffer[1] = expected.buffer[1] = 0
        columns = {"snr": jnp.array([3., 4.], dtype=jnp.float32)}
        errors = []
        for buffer, context in ((actual, nullcontext()),
                                (expected, monkeypatch.context())):
            with context as patch:
                if patch is not None:
                    # Retain the preexisting per-ring guard/error boundary;
                    # disabling that guard would change the error message.
                    patch.setattr(coinc_jax, "_prepare_singles_ring_group",
                                  lambda *_args: {})
                with pytest.raises(TypeError) as error:
                    buffer.add(np.array([0, 1]), columns)
                errors.append(str(error.value))
        _assert_exact(_ring_state(actual), _ring_state(expected))
        assert errors[0] == errors[1]
        assert actual.valid_ends == [1, 0] and actual.time == 0


@pytest.mark.parametrize("storage", ["metadata", "clocks"])
def test_custom_host_storage_preserves_serial_side_effects(monkeypatch, device,
                                                         storage):
    """A custom host concatenation can change a later scientific row."""
    class MutatingMetadata(np.ndarray):
        def __array_function__(self, function, types, args, kwargs):
            if function is np.concatenate:
                self.target.buffer[1]["snr"] += 10
                return np.concatenate(tuple(np.asarray(value)
                                            for value in args[0]), **kwargs)
            return super().__array_function__(function, types, args, kwargs)

    def reject_group(*_args, **_kwargs):
        raise AssertionError("custom host storage reached grouped dispatch")

    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 10)
        expected = coinc_jax.JAXMultiRingBuffer(2, 10)
        columns = {"snr": jnp.array([1., 2.], dtype=jnp.float32),
                   "metadata": np.array(["a", "b"])}
        for buffer in (actual, expected):
            buffer.add([0, 1], columns)
            if storage == "metadata":
                value = buffer.buffer[0]["metadata"].view(MutatingMetadata)
                buffer.buffer[0]["metadata"] = value
            else:
                value = buffer._expire_times[0].view(MutatingMetadata)
                buffer._expire_times[0] = value
            value.target = buffer
        monkeypatch.setattr(coinc_jax, "_gather_singles_column_groups", reject_group)
        columns = {"snr": jnp.array([3., 4.], dtype=jnp.float32),
                   "metadata": np.array(["c", "d"])}
        actual.add([0, 1], columns)
        with monkeypatch.context() as patch:
            # Compare against the unchanged prior per-ring implementation.
            patch.setattr(coinc_jax, "_prepare_singles_ring_group",
                          lambda *_args: {})
            expected.add([0, 1], columns)
        _assert_exact(_ring_state(actual), _ring_state(expected))
        _assert_exact(actual.buffer[1]["snr"],
                      jnp.array([12., 4.], dtype=jnp.float32))


@pytest.mark.parametrize("ids", [[0, -2], [-1, 1], [-2, 0, -2]])
def test_negative_ring_aliases_retain_serial_append_order(monkeypatch, device, ids):
    """Negative NumPy-style IDs can address the same physical ring twice."""
    def reject_group(*_args, **_kwargs):
        raise AssertionError("aliased ring IDs reached grouped dispatch")

    monkeypatch.setattr(coinc_jax, "_gather_singles_column_groups", reject_group)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXMultiRingBuffer(2, 2)
        expected = coinc_jax.JAXMultiRingBuffer(2, 2)
        for block in range(3):
            columns = _columns(len(ids), block)
            actual.add(np.asarray(ids, "i4"), columns)
            with _serial(monkeypatch):
                expected.add(np.asarray(ids, "i4"), columns)
            _assert_exact(_ring_state(actual), _ring_state(expected))


def test_grouped_cores_reuse_indices_and_clock_without_compiler_events(device):
    """The reusable signature describes shapes, not clocks or selected rows."""
    with scheme.JAXScheme(device):
        columns = tuple(value for value in _columns(132).values()
                        if isinstance(value, jax.Array))
        counts = (2, 1, 3, 2)
        indices = jnp.array([131, 0, 3, 9, 17, 1, 5, 2], dtype=jnp.int64)
        previous = tuple(tuple(value[:1] for value in columns) for _ in counts)
        expiries = tuple(jnp.array([1], dtype=jnp.int32) for _ in counts)

        def run(selected_indices, clock):
            selected = coinc_jax._gather_singles_column_groups(
                columns, selected_indices, counts=counts)
            return coinc_jax._append_selected_singles_groups(
                selected, previous, expiries, np.int32(clock))

        observer = CompilationGuard()
        try:
            jax.block_until_ready(run(indices, 2))
            changed = indices[::-1]
            # Warm input preparation as well; only production dispatch is
            # inside the guard, without introducing test-only slice kernels.
            changed.block_until_ready()
            before = observer.snapshot()
            with observer.timed_guard():
                actual = run(changed, 7)
                jax.block_until_ready(actual)
            assert observer.snapshot() == before
            assert {key: observer.snapshot()[key] - before[key]
                    for key in COUNTERS} == dict.fromkeys(COUNTERS, 0)
            positions = np.asarray(changed)
            offset = 0
            for (row, expiry), count, old in zip(actual, counts, previous):
                for value, prior, joined in zip(columns, old, row):
                    want = np.concatenate((np.asarray(prior),
                                           np.asarray(value)[positions[offset:offset + count]]))
                    _assert_exact(np.asarray(joined), want)
                np.testing.assert_array_equal(expiry, [1] + [7] * count)
                offset += count
        finally:
            observer.close()


def _triggers(templates, block, offset):
    templates = np.asarray(templates, dtype="i4")
    size = templates.size
    rows = {"snr": jnp.asarray(np.full(size, 7. + offset, "f4")),
            "chisq": jnp.asarray(np.full(size, 2., "f4")),
            "chisq_dof": jnp.asarray(np.full(size, 2, "u4")),
            "template_id": jnp.asarray(templates),
            "end_time": jnp.asarray(100. + block + templates * .1 +
                                     np.arange(size) * .0001 + offset * .001),
            "mass1": jnp.asarray(np.full(size, 10., "f8")),
            "mass2": jnp.asarray(np.full(size, 9., "f8")),
            "approximant": np.array(["TaylorF2"] * size)}
    rows.update({f"extra{i}": jnp.asarray(np.arange(size, dtype="i4") + i)
                 for i in range(10)})
    return rows


def _estimator():
    estimator = coinc_jax.JAXLiveCoincTimeslideBackgroundEstimator(
        12, 8, "single_ranking_only", "snr", [], ["H1", "L1"],
        ifar_limit=1, timeslide_interval=.1, return_background=True)
    estimator.buffer_size = 2
    estimator.coincs = coinc_jax.JAXCoincExpireBuffer(2, estimator.ifos,
                                                   initial_size=32)
    return estimator


def _estimator_state(estimator):
    return {"memory": estimator.trig_stat_memory,
            "singles": {ifo: _ring_state(buffer)
                        for ifo, buffer in estimator.singles.items()},
            "background": {"buffer": estimator.coincs.buffer,
                           "timers": dict(estimator.coincs.timer),
                           "index": estimator.coincs.index,
                           "time": dict(estimator.coincs.time)}}


def test_warmed_estimator_grouped_dispatch_matches_serial_bytes(monkeypatch, device):
    """A fresh repeated search retains outputs/state without executable loads."""
    with scheme.JAXScheme(device):
        ids = [9, 0, 1, 2, 9, 3, 4, 5, 6, 7, 8, 0]
        blocks = [{ifo: _triggers(ids, block, offset)
                   for offset, ifo in enumerate(("H1", "L1"))}
                  for block in (0, 1)]
        blocks += [{ifo: _triggers([], 2, offset)
                    for offset, ifo in enumerate(("H1", "L1"))},
                   {"H1": False, "L1": _triggers([0, 1, 1, 9], 3, 1)},
                   {ifo: _triggers(ids[::-1], 4, offset)
                    for offset, ifo in enumerate(("H1", "L1"))}]
        originals = [{ifo: False if rows is False else dict(rows)
                      for ifo, rows in block.items()} for block in blocks]

        def replay(estimator):
            outputs, states = [], []
            for block in blocks:
                outputs.append(estimator.add_singles(block))
                states.append(_estimator_state(estimator))
            jax.block_until_ready(jax.tree.leaves((outputs, states)))
            return outputs, states

        with _serial(monkeypatch):
            expected = replay(_estimator())
        observer = CompilationGuard()
        try:
            replay(_estimator())
            measured = _estimator()
            before = observer.snapshot()
            with observer.timed_guard():
                actual = replay(measured)
            assert observer.snapshot() == before
        finally:
            observer.close()
        _assert_exact(actual, expected)
        _assert_exact(blocks, originals)
        assert any("foreground/stat" in output for output in actual[0])
        assert any(output.get("background/count", 0) for output in actual[0])
