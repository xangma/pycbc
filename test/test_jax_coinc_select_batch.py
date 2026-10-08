# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Exact stable post-cluster selections without per-column dispatch."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc_jax
from test_jax_coinc_dispatch import _assert_exact, _columns, _estimator_state
from test_jax_coinc_match_batch import _estimator, _triggers
from jax_test_helpers import CompilationGuard, COUNTERS

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


def _selection_inputs(dtype):
    # Gather-only kernels must retain payload NaNs, subnormals and signed zero.
    if dtype == "f4":
        stat = np.array([0x80000000, 1, 0x7fc00017, 0x7f800000,
                         0xff800000, 0, 0x3f800000, 0x80000000,
                         0x7fc00031], "u4").view("f4")
    else:
        stat = np.array([0x8000000000000000, 1, 0x7ff8000000000017,
                         0x7ff0000000000000, 0xfff0000000000000, 0,
                         0x3ff0000000000000, 0x8000000000000000,
                         0x7ff8000000000031], "u8").view("f8")
    return (jnp.asarray(stat),
            jnp.asarray(np.arange(9, dtype="i4") + 10),
            jnp.asarray(np.arange(9, dtype="i8") + 20),
            jnp.asarray(np.arange(9, dtype="i4") % 3),
            jnp.asarray(np.arange(9, dtype="i4") - 1),
            jnp.asarray(np.arange(9, dtype="i8") + 5))


def _eager_selection(columns, cidx, offsets):
    stat, expiry0, expiry1, templates, ids0, ids1 = columns
    selected_offsets = offsets[cidx]
    background = selected_offsets != 0
    zerolag = selected_offsets == 0
    num_background, num_zerolag = background.sum(), zerolag.sum()
    zero_stat = stat[cidx][zerolag]
    if int(num_zerolag):
        first = cidx[zerolag][0]
        first_values = (zero_stat[0], templates[first], ids0[first], ids1[first])
    else:
        first_values = tuple(jnp.zeros((), dtype=value.dtype)
                             for value in (stat, templates, ids0, ids1))
    return (stat[cidx][background], expiry0[cidx][background],
            expiry1[cidx][background], zero_stat, *first_values,
            num_background, num_zerolag)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("selected", [[], [7, 2, 0, 7], [6, 3, 1, 6],
                                      [6, 7, 2, 1, 0, 3, 7, 4],
                                      [-1, 1, -9, 2]])
def test_cluster_selection_matches_eager_bytes_and_order(device, dtype, selected):
    """Empty, all-zero, all-shifted and mixed partitions retain duplicates."""
    with scheme.JAXScheme(device):
        columns = _selection_inputs(dtype)
        offsets = jnp.asarray([0, 1, 0, -2, 0, 3, 1, 0, 0], dtype=jnp.int32)
        cidx = jnp.asarray(selected, dtype=jnp.int64)
        background, zerolag, counts = coinc_jax._partition_live_cluster(cidx,
                                                                        offsets)
        host_counts = np.asarray(counts)
        assert counts.dtype == jnp.int64 and counts.shape == (2,)
        assert background.shape == zerolag.shape == cidx.shape
        mask = offsets[cidx] != 0
        _assert_exact(background[:int(host_counts[0])], cidx[mask])
        _assert_exact(zerolag[:int(host_counts[1])], cidx[~mask])
        actual = coinc_jax._select_live_cluster(
            *columns, background, zerolag,
            background_count=int(host_counts[0]), zerolag_count=int(host_counts[1]))
        _assert_exact(actual, _eager_selection(columns, cidx, offsets))


def test_cluster_selection_reuses_warmed_compilation(device):
    """Replay exactly the admitted signatures without compiling in timing."""
    with scheme.JAXScheme(device):
        columns = _selection_inputs("f8")
        offsets = jnp.asarray([0, 1, 0, -2, 0, 3, 1, 0, 0], dtype=jnp.int32)
        cidx = jnp.asarray([6, 7, 2, 1, 0, 3, 7, 4], dtype=jnp.int64)

        def select():
            background, zerolag, counts = coinc_jax._partition_live_cluster(cidx,
                                                                            offsets)
            background_count, zerolag_count = map(int, np.asarray(counts))
            values = coinc_jax._select_live_cluster(
                *columns, background, zerolag,
                background_count=background_count, zerolag_count=zerolag_count)
            jax.block_until_ready(values)
            return values

        observer = CompilationGuard()
        try:
            expected = select()
            before = observer.snapshot()
            with observer.timed_guard():
                actual = select()
            assert observer.snapshot() == before
            assert {key: observer.snapshot()[key] - before[key]
                    for key in COUNTERS} == dict.fromkeys(COUNTERS, 0)
        finally:
            observer.close()
        _assert_exact(actual, expected)


@pytest.mark.parametrize("index", [0, 3, -1])
def test_foreground_numeric_tuple_gather_preserves_column_types(device, index):
    """Scalar gathers retain mixed typed scalar/vector fields and host metadata."""
    with scheme.JAXScheme(device):
        stored = _columns(4)
        numeric = tuple(value for value in stored.values()
                        if isinstance(value, jax.Array))
        position = jnp.asarray(index, dtype=jnp.int32)
        actual = coinc_jax._gather_singles_columns(numeric, position)
        expected = tuple(value[position] for value in numeric)
        _assert_exact(actual, expected)
        # Native metadata follows its original NumPy indexing path.
        assert stored["metadata"][index] == (f"0:{index % 4}"
                                            if index % 4 % 2 else None)


def test_foreground_column_hook_retains_serial_read_order(device):
    """A custom column may change a later column before its original read."""
    with scheme.JAXScheme(device):
        position = jnp.asarray(0, dtype=jnp.int32)

        def stored_row():
            stored = {}

            class MutatingColumn:
                def __getitem__(self, _index):
                    stored["later"] = jnp.asarray([19., 23.], dtype=jnp.float64)
                    return "changed"

            stored.update({"hook": MutatingColumn(),
                           "first": jnp.asarray([3., 5.], dtype=jnp.float32),
                           "later": jnp.asarray([7., 11.], dtype=jnp.float64)})
            return stored

        eager = stored_row()
        expected = {key: value[position] for key, value in eager.items()}
        grouped = stored_row()
        prepared = coinc_jax._prepare_live_foreground(grouped, position)
        assert prepared == {}
        actual = {key: prepared[key] if key in prepared else value[position]
                  for key, value in grouped.items()}
        _assert_exact(actual, expected)


@pytest.mark.parametrize("case", ["host_index", "boolean_index", "host_column",
                                 "unequal_length", "wrapped_column"])
def test_cluster_selection_unusual_inputs_keep_eager_fallback(device, case):
    """Admission should retain unusual wrappers and original indexing errors."""
    with scheme.JAXScheme(device):
        columns = list(_selection_inputs("f8"))
        offsets = jnp.asarray([0, 1, 0, -2, 0, 3, 1, 0, 0], dtype=jnp.int32)
        cidx = jnp.asarray([6, 7, 2, 1, 0, 3, 7, 4], dtype=jnp.int64)
        if case == "host_index":
            cidx = np.asarray(cidx)
        elif case == "boolean_index":
            cidx = jnp.asarray([True, False] * 4)
        elif case == "host_column":
            columns[1] = np.asarray(columns[1])
        elif case == "unequal_length":
            columns[1] = columns[1][:-1]
        else:
            from pycbc.types import Array
            columns[1] = Array(columns[1])
        assert coinc_jax._prepare_live_cluster(cidx, offsets, tuple(columns)) is None


def test_post_cluster_replay_preserves_order_state_and_warmed_reuse(
        monkeypatch, device):
    """Ranking and payload batching remain enabled in the exact eager oracle."""
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
        clustered, gathered = [], []
        prepare_cluster = coinc_jax._prepare_live_cluster
        prepare_foreground = coinc_jax._prepare_live_foreground

        def record_cluster(*args):
            result = prepare_cluster(*args)
            if result is not None:
                clustered.append(result)
            return result

        def record_foreground(*args):
            result = prepare_foreground(*args)
            if result:
                gathered.append(result)
            return result

        monkeypatch.setattr(coinc_jax, "_prepare_live_cluster", record_cluster)
        monkeypatch.setattr(coinc_jax, "_prepare_live_foreground", record_foreground)

        def replay():
            estimator = _estimator()
            outputs, states = [], []
            for block in blocks:
                outputs.append(estimator.add_singles(block))
                states.append(_estimator_state(estimator))
            result = outputs, states
            jax.block_until_ready(jax.tree.leaves(result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(coinc_jax, "_prepare_live_cluster", lambda *_args: None)
            patch.setattr(coinc_jax, "_prepare_live_foreground", lambda *_args: {})
            expected = replay()
        assert not clustered and not gathered
        observer = CompilationGuard()
        try:
            replay()
            assert clustered and gathered
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
        assert any("foreground/stat" in output for output in actual[0])
