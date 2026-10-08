# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Exact ordered payloads and state publication in live coincidence batches."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc_jax
from test_jax_coinc_dispatch import _assert_exact, _estimator_state, _ring_state
from test_jax_coinc_match_batch import _estimator, _triggers
from jax_test_helpers import CompilationGuard, COUNTERS

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


@pytest.mark.parametrize("memory_dtype", ["f4", "f8"])
@pytest.mark.parametrize("fixed_dtype", ["f4", "f8"])
def test_group_payloads_equal_sequential_gathers_and_workspace(
        device, memory_dtype, fixed_dtype):
    """Interleaved query shapes retain pruned IDs and the 8→2→4 memory tail."""
    tiny = np.nextafter(np.float32(0), np.float32(1))
    with scheme.JAXScheme(device):
        memory = jnp.asarray(np.arange(4, dtype=memory_dtype) + 11)
        fixed_stats = jnp.asarray(np.array([-0., tiny, 7.], fixed_dtype))
        counts = (8, 2, 4)
        selections = ([3, 0, 3, 1] * 2, [1, 0], [3, 0, 3, 1])
        rows, controls = [], []
        for number, (count, selected) in enumerate(zip(counts, selections)):
            start = 1 if number == 1 else 2
            length = 4 if number == 1 else 6
            position = 0 if number != 2 else 1
            indices = np.zeros((2, length * 3), "i8")
            indices[position, :count] = selected
            slides = np.zeros_like(indices)
            slides[position, :count] = np.resize([-1, 0, 1], count)
            fixed_times = jnp.array([1e9 + .01, -1e9 + .02])
            times = jnp.array(1e9 + np.arange(length) / 1024)
            dtype = "f4" if number != 1 else "f8"
            stats = jnp.asarray(np.resize(np.array([tiny, -0., 5., 7.], dtype),
                                          length))
            expiry = jnp.asarray(np.arange(length, dtype="i4") + 4)
            rows.append((jnp.asarray(indices), jnp.asarray(slides), fixed_times,
                         times, stats, expiry))
            controls.append((position, number, start, 2 - number, 8))
        control = jnp.asarray(np.asarray(controls, "i8").T)
        actual = coinc_jax._prepare_live_match_group(
            memory, fixed_stats, tuple(rows), control, counts=counts)
        expected = []
        for row, values, count in zip(rows, controls, counts):
            indices, slides, fixed_times, times, stats, expiry = row
            position, index, start, template, clock = values
            selected, slide, fixed_time = coinc_jax._compact_live_query(
                indices, slides, fixed_times, np.int64(position), count=count)
            while memory.size < count:
                memory = jnp.pad(memory, (0, memory.size))
            memory, fixed, shifted, chirp, payload = coinc_jax._prepare_live_match(
                memory, fixed_stats, None, index, fixed_time, times[start:],
                stats[start:], expiry[start:], selected, template, clock)
            assert chirp is None
            expected.append((memory, fixed, shifted, selected, slide, fixed_time,
                             payload))
        _assert_exact(actual, tuple(expected))
        assert actual[-1][0].size == 8
        # Later shorter writes leave the signed-zero tail from the first row.
        _assert_exact(actual[-1][0][4:], jnp.full(4, -0., dtype=memory.dtype))
        for result, selected in zip(actual, selections):
            np.testing.assert_array_equal(result[-1][-2], selected)


def test_payload_batch_replay_preserves_order_state_and_warmed_reuse(
        monkeypatch, device):
    """Compare full search with query batching enabled in both replay paths."""
    with scheme.JAXScheme(device):
        ids = np.resize([0, 1, 0, 2, 3, 2, 4, 5, 6], 22)
        blocks = [{ifo: _triggers(ids, 100. + block + ids * .0625 +
                                 np.arange(ids.size) * .0001 + offset * .001,
                                 "f4" if offset == 0 else "f8")
                   for offset, ifo in enumerate(("H1", "L1"))}
                  for block in (0, 1)]
        blocks += [{"H1": False, "L1": _triggers([0, 0, 7],
                                                  [102., 102.001, 102.4])},
                   {ifo: _triggers([], []) for ifo in ("H1", "L1")},
                   {ifo: _triggers(ids[::-1], 104. + ids[::-1] * .0625 +
                                  np.arange(ids.size) * .0001 + offset * .001)
                    for offset, ifo in enumerate(("H1", "L1"))}]
        originals = [{ifo: rows if rows is False else dict(rows)
                      for ifo, rows in block.items()} for block in blocks]
        joined, dispatched = [], []
        join = coinc_jax._join_live_matches
        gather = coinc_jax._prepare_live_match_group

        def record_join(payloads):
            result = join(payloads)
            joined.append(result)
            return result

        def record_gather(*args, counts):
            dispatched.append(counts)
            return gather(*args, counts=counts)

        monkeypatch.setattr(coinc_jax, "_join_live_matches", record_join)
        monkeypatch.setattr(coinc_jax, "_prepare_live_match_group", record_gather)

        def replay():
            estimator = _estimator()
            outputs, states = [], []
            joined.clear()
            for block in blocks:
                outputs.append(estimator.add_singles(block))
                states.append(_estimator_state(estimator))
            result = outputs, states, list(joined)
            jax.block_until_ready(jax.tree.leaves(result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(coinc_jax, "_prepare_live_payloads", lambda *_args: {})
            expected = replay()
        assert not dispatched
        observer = CompilationGuard()
        try:
            replay()
            assert dispatched
            before = observer.snapshot()
            with observer.timed_guard():
                actual = replay()
            assert observer.snapshot() == before
            assert {key: observer.snapshot()[key] - before[key]
                    for key in COUNTERS} == dict.fromkeys(COUNTERS, 0)
        finally:
            observer.close()
        _assert_exact(actual, expected)
        _assert_exact(blocks, originals)
        assert actual[2]
        assert any("foreground/stat" in output for output in actual[0])


def test_uncovered_singleton_query_retains_serial_payload_path(
        monkeypatch, device):
    """A singleton shape between grouped queries cannot skip a workspace write."""
    with scheme.JAXScheme(device):
        estimator = _estimator()
        ring = coinc_jax.JAXMultiRingBuffer(8, 3)
        fixed_ring = coinc_jax.JAXMultiRingBuffer(8, 3)
        history = _triggers([0, 1, 1], [100., 100., 100.001])
        history["stat"] = history["snr"]
        ring.add(np.array([0, 1, 1], "i4"), history)
        trigs = _triggers([0, 1, 0], [100., 100., 100.])
        trigs["stat"] = trigs["snr"]
        templates = np.asarray(trigs["template_id"])
        queries = coinc_jax._prepare_live_queries(
            ring, trigs, templates, 0, 3, estimator.stat_calculator, .01, 1.)
        assert set(queries) == {0, 2}
        before = _ring_state(ring)
        memory = jnp.array([3., 4., 5., 6.], dtype=jnp.float32)

        def reject(*_args, **_kwargs):
            pytest.fail("an uncovered nonempty query must use serial payloads")

        monkeypatch.setattr(coinc_jax, "_prepare_live_match_group", reject)
        assert coinc_jax._prepare_live_payloads(
            ring, fixed_ring, trigs, templates, 0, 3, queries, memory) == {}
        _assert_exact(_ring_state(ring), before)
        np.testing.assert_array_equal(memory, [3., 4., 5., 6.])


@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_count_queries_do_not_write_or_resize_workspace(
        monkeypatch, device, all_zero):
    """An unmatched final row cannot publish its statistic over a prior match."""
    with scheme.JAXScheme(device):
        estimator = _estimator()
        times = [100.25] * 4 if all_zero else [100., 100.25, 100., 100.25]
        trigs = _triggers([0] * 4, times)
        trigs["stat"] = trigs["snr"]
        estimator.singles["H1"] = coinc_jax.JAXMultiRingBuffer(8, 3)
        estimator.singles["L1"] = coinc_jax.JAXMultiRingBuffer(8, 3)
        estimator.singles["H1"].add(np.zeros(4, "i4"), trigs)
        history = _triggers([0], [100.])
        history["stat"] = history["snr"]
        estimator.singles["L1"].add(np.zeros(1, "i4"), history)
        estimator.trig_stat_memory = jnp.array([3., 4., 5., 6.], jnp.float32)
        prepare = coinc_jax._prepare_live_payloads
        prepared_keys = []

        def record(*args):
            result = prepare(*args)
            prepared_keys.append(set(result))
            return result

        monkeypatch.setattr(coinc_jax, "_prepare_live_payloads", record)
        coinc_jax.find_coincs_jax(estimator, {"H1": trigs}, ["H1"])
        assert prepared_keys == [set() if all_zero else {0, 2}]
        np.testing.assert_array_equal(estimator.trig_stat_memory,
                                      [3. if all_zero else 9., 4., 5., 6.])
