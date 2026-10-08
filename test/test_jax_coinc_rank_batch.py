# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Eager native arithmetic and state parity of grouped coincidence ranking."""

from contextlib import contextmanager
import sys
from types import MethodType

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc_jax
from test_jax_coinc_dispatch import _assert_exact, _estimator_state
from test_jax_coinc_match_batch import _estimator, _triggers
from jax_test_helpers import CompilationGuard, COUNTERS

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


@contextmanager
def _native_rank_calls():
    """Count native Python calls without changing its identity or behavior."""
    calls = []
    previous = sys.getprofile()
    native = coinc_jax._quadrature_rank_stat.__code__

    def record(frame, event, argument):
        if event == "call" and frame.f_code is native:
            calls.append(True)
        if previous is not None:
            previous(frame, event, argument)

    sys.setprofile(record)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _payload(fixed, shifted):
    count = fixed.size
    indices = jnp.arange(count, dtype=jnp.int64)
    slides = jnp.resize(jnp.array([-1, 0, 1], jnp.int32), (count,))
    fixed_time = jnp.array([1e9 + .01])
    times = jnp.full(count, fixed_time[0], jnp.float64)
    columns = (times, times, jnp.full(count, 3, jnp.int32),
               jnp.full(count, 4, jnp.int32), jnp.full(count, 2, jnp.int32),
               indices.astype(jnp.int32), jnp.full(count, -1, jnp.int32))
    return fixed, fixed, shifted, indices, slides, fixed_time, columns


def _native_rank(ranker, payload):
    result = ranker.rank_stat_coinc(
        [["H1", payload[1]], ["L1", payload[2]]], payload[4], 1., [0, -1])
    return jnp.asarray(result, dtype=jnp.float64)


@pytest.mark.parametrize("fixed_dtype", ["f4", "f8"])
@pytest.mark.parametrize("shifted_dtype", ["f4", "f8"])
def test_grouped_rank_preserves_eager_native_arithmetic_bits(
        device, fixed_dtype, shifted_dtype):
    """Subnormals, overflow and nonfinite values retain each eager boundary."""
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        fixed_limits = np.finfo(fixed_dtype)
        shifted_limits = np.finfo(shifted_dtype)
        fixed_tiny = np.nextafter(np.array(0, fixed_dtype),
                                 np.array(1, fixed_dtype))
        shifted_tiny = np.nextafter(np.array(0, shifted_dtype),
                                   np.array(1, shifted_dtype))
        fixed = np.array([-0., fixed_tiny, -fixed_tiny, 3., -1.,
                          fixed_limits.max / 2, np.inf, np.nan], fixed_dtype)
        shifted = np.array([-0., shifted_tiny, -shifted_tiny, 4., -1.,
                            shifted_limits.max / 2, np.nan, np.inf], shifted_dtype)
        payloads = {index: _payload(jnp.asarray(np.resize(fixed, count)),
                                    jnp.asarray(np.resize(shifted, count)))
                    for index, count in zip((5, 1, 7), (8, 2, 4))}
        originals = dict(payloads)
        with _native_rank_calls() as serial_calls:
            expected = {index: _native_rank(ranker, payload)
                        for index, payload in payloads.items()}
        with _native_rank_calls() as grouped_calls:
            actual = coinc_jax._rank_live_matches(ranker, payloads)
        assert set(actual) == set(payloads)
        for index in payloads:
            _assert_exact(actual[index], expected[index])
            assert actual[index].dtype == jnp.float64
        _assert_exact(payloads, originals)
        assert len(serial_calls) == 3
        assert len(grouped_calls) < len(serial_calls)


def test_interleaved_dtype_groups_preserve_incoming_associations(device):
    """Grouping by operand dtypes must not associate ranks with another row."""
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        payloads = {}
        for number, (index, count) in enumerate(zip((7, 0, 6, 2), (8, 2, 4, 3))):
            fixed_dtype = jnp.float32 if number % 2 else jnp.float64
            shifted_dtype = jnp.float64 if number % 2 else jnp.float32
            fixed = jnp.arange(count, dtype=fixed_dtype) + 2 + number
            shifted = jnp.arange(count, dtype=shifted_dtype) + 7 - number
            payloads[index] = _payload(fixed, shifted)
        expected = {index: _native_rank(ranker, payload)
                    for index, payload in payloads.items()}
        with _native_rank_calls() as calls:
            actual = coinc_jax._rank_live_matches(ranker, payloads)
        assert set(actual) == set(expected)
        for index in expected:
            _assert_exact(actual[index], expected[index])
        assert len(calls) < len(payloads)


@pytest.mark.parametrize("weak_operand", ["fixed", "shifted"])
def test_interleaved_weak_types_preserve_native_promotion(device, weak_operand):
    """Identical dtypes can require different native arithmetic precision."""
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        payloads = {}
        for number, (index, count) in enumerate(zip((7, 0, 6, 2), (1, 2, 3, 4))):
            weak = number % 2 == 0
            value = (jnp.broadcast_to(jnp.asarray(2.718281828), (count,))
                     if weak else jnp.full(count, 2.718281828, jnp.float64))
            assert value.dtype == jnp.float64 and value.weak_type == weak
            strong = jnp.full(count, 3.1415925, jnp.float32)
            fixed, shifted = ((value, strong) if weak_operand == "fixed" else
                              (strong, value))
            payloads[index] = _payload(fixed, shifted)
        expected = {index: _native_rank(ranker, payload)
                    for index, payload in payloads.items()}
        # The weak double operand keeps the other operand's float32 precision.
        assert np.asarray(expected[7])[0] != np.asarray(expected[0])[0]
        with _native_rank_calls() as calls:
            actual = coinc_jax._rank_live_matches(ranker, payloads)
        assert set(actual) == set(expected)
        for index in expected:
            _assert_exact(actual[index], expected[index])
        assert len(calls) < len(payloads)


@pytest.mark.parametrize("kind", ["hook", "subclass"])
def test_custom_rankers_keep_scalar_hook_boundaries(device, kind):
    """Batch preparation cannot invoke a custom ranker before its serial turn."""
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator

        def reject(*_args, **_kwargs):
            pytest.fail("a custom ranker must retain its scalar call boundary")

        if kind == "hook":
            ranker.rank_stat_coinc = MethodType(reject, ranker)
        else:
            class CustomRanker(type(ranker)):
                rank_stat_coinc = reject

            ranker = object.__new__(CustomRanker)
        payload = _payload(jnp.array([3.]), jnp.array([4.]))
        assert coinc_jax._rank_live_matches(ranker, {0: payload, 1: payload}) == {}


def test_single_payload_keeps_existing_native_dispatch(device):
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        payload = _payload(jnp.array([3.]), jnp.array([4.]))
        with _native_rank_calls() as calls:
            assert coinc_jax._rank_live_matches(ranker, {}) == {}
            assert coinc_jax._rank_live_matches(ranker, {0: payload}) == {}
        assert not calls


def test_grouped_rank_replay_preserves_state_order_and_warmed_reuse(
        monkeypatch, device):
    """Leave payload batching enabled in both full-search comparison paths."""
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
        joined, admitted = [], []
        join = coinc_jax._join_live_matches
        rank = coinc_jax._rank_live_matches

        def record_join(payloads):
            result = join(payloads)
            joined.append(result)
            return result

        def record_rank(*args):
            result = rank(*args)
            if result:
                admitted.append(len(result))
            return result

        monkeypatch.setattr(coinc_jax, "_join_live_matches", record_join)
        monkeypatch.setattr(coinc_jax, "_rank_live_matches", record_rank)

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
            patch.setattr(coinc_jax, "_rank_live_matches", lambda *_args: {})
            with _native_rank_calls() as serial_calls:
                expected = replay()
        assert not admitted
        observer = CompilationGuard()
        try:
            replay()
            assert admitted
            before = observer.snapshot()
            with observer.timed_guard(), _native_rank_calls() as grouped_calls:
                actual = replay()
            assert observer.snapshot() == before
            assert {key: observer.snapshot()[key] - before[key]
                    for key in COUNTERS} == dict.fromkeys(COUNTERS, 0)
        finally:
            observer.close()
        assert len(grouped_calls) < len(serial_calls)
        _assert_exact(actual, expected)
        _assert_exact(blocks, originals)
        assert actual[2]
        assert any("foreground/stat" in output for output in actual[0])
