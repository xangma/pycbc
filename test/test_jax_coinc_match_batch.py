# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under GPLv3; see the repository COPYING file.
"""Scientific ordering and executable reuse of batched live time matching."""

from contextlib import contextmanager, nullcontext
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc, coinc_jax
from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData
from test_jax_coinc_dispatch import _assert_exact, _estimator_state
from jax_test_helpers import CompilationGuard, COUNTERS

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    return request.param


def _estimator():
    result = coinc_jax.JAXLiveCoincTimeslideBackgroundEstimator(
        8, 8, "single_ranking_only", "snr", [], ["H1", "L1"],
        ifar_limit=1, timeslide_interval=1., return_background=True)
    result.time_window = .01
    result.coincs = coinc_jax.JAXCoincExpireBuffer(3, result.ifos,
                                                initial_size=32)
    result.buffer_size = 3
    return result


def _triggers(templates, times, dtype="f4"):
    templates = np.asarray(templates, "i4")
    count = templates.size
    return {"template_id": jnp.asarray(templates),
            "end_time": jnp.asarray(times, dtype=jnp.float64),
            "snr": jnp.asarray(np.arange(count, dtype=dtype) + 7),
            "chisq": jnp.asarray(np.full(count, 2., dtype=dtype)),
            "chisq_dof": jnp.asarray(np.full(count, 2, dtype="u4")),
            "mass1": jnp.asarray(np.full(count, 10., dtype="f8")),
            "mass2": jnp.asarray(np.full(count, 9., dtype="f8")),
            "approximant": np.array(["TaylorF2"] * count)}


def _direct_case():
    estimator = _estimator()
    result = _triggers([0, 0], [100., 100.2])
    result["stat"] = result["snr"]
    estimator.singles["H1"] = coinc_jax.JAXMultiRingBuffer(8, 3)
    estimator.singles["L1"] = coinc_jax.JAXMultiRingBuffer(8, 3)
    estimator.singles["H1"].add([0, 0], result)
    estimator.singles["L1"].add(
        [0], {"end_time": jnp.array([100.]),
              "stat": jnp.array([9.], dtype=jnp.float32)})
    return estimator, {"H1": result}


@contextmanager
def _serial(monkeypatch):
    # Disable only the new batching boundary, retaining all prior JAX kernels.
    with monkeypatch.context() as patch:
        patch.setattr(coinc_jax, "_can_prepare_live_queries", lambda *_args: False)
        yield


@pytest.mark.parametrize('failure', [None, 'dispatch', 'readback'])
def test_live_query_stage_markers_balance_real_boundaries(
        monkeypatch, device, failure):
    """Dispatch and existing count readback remain distinct, unfenced ranges."""
    from pycbc import benchmark

    records = []

    def emit(stage, event, **metadata):
        records.append((stage, event, metadata))

    def fail(*_args, **_kwargs):
        raise RuntimeError('query boundary failed')

    class FailReadback:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, value, *args, **kwargs):
            if isinstance(value, jax.Array):
                fail()
            return np.asarray(value, *args, **kwargs)

    monkeypatch.delenv('PYCBC_BENCHMARK_STAGES', raising=False)
    with scheme.JAXScheme(device):
        estimator, results = _direct_case()
        trigs = results['H1']
        ring = estimator.singles['L1']
        monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
        monkeypatch.setattr(benchmark, 'stage_event', emit)
        if failure == 'dispatch':
            monkeypatch.setattr(coinc_jax, '_time_coincidence_query_core', fail)
        elif failure == 'readback':
            monkeypatch.setattr(coinc_jax, 'np', FailReadback())
        context = (pytest.raises(RuntimeError, match='query boundary failed')
                   if failure else nullcontext())
        with context:
            prepared = coinc_jax._prepare_live_queries(
                ring, trigs, np.asarray([0, 0], 'i4'), 0, 2,
                estimator.stat_calculator, .01, 1.)
            assert list(prepared) == [0, 1]
    stages = ['coinc_query_host_prepare', 'coinc_query_host_prepare',
              'coinc_query_dispatch', 'coinc_query_count_readback',
              'coinc_query_host_publish']
    if failure == 'dispatch':
        stages = stages[:3]
    elif failure == 'readback':
        stages = stages[:4]
    assert [stage for stage, event, _ in records if event == 'start'] == stages
    assert len(records) == 2 * len(stages)
    for start, end in zip(records[::2], records[1::2]):
        assert start[0] == end[0]
        assert (start[1], end[1]) == ('start', 'end')
        assert start[2] == end[2]
        assert start[2]['synchronize'] is False


def test_repeated_template_search_retains_order_state_and_warmed_reuse(
        monkeypatch, device):
    """Fresh scientific state yields the serial outputs without new executables."""
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
        joined = []
        join = coinc_jax._join_live_matches

        def record(payloads):
            result = join(payloads)
            joined.append(result)
            return result

        monkeypatch.setattr(coinc_jax, "_join_live_matches", record)

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

        with _serial(monkeypatch):
            expected = replay()
        observer = CompilationGuard()
        try:
            replay()
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
        assert actual[2]  # Compare pre-clustering ordered columns as well.


@pytest.mark.parametrize('history', [1, 2, 5])
def test_resident_match_rank_cluster_keeps_lengths_on_device_until_publication(
        monkeypatch, device, history):
    """Actual ring searches preserve outputs/state with one final control read."""
    with scheme.JAXScheme(device):
        # CPU execution qualifies the same numerical core; production admission
        # remains GPU-only. MPI supplies these routing IDs as host metadata.
        monkeypatch.setattr(coinc_jax, '_resident_live_device_supported', lambda _: True)
        ids = np.resize([0, 1, 0, 2, 3, 2, 4, 5, 6], 22)
        blocks = [{ifo: _triggers(ids, 100. + block + ids * .0625 +
                                 np.arange(ids.size) * .0001 + offset * .001,
                                 'f4' if offset == 0 else 'f8')
                   for offset, ifo in enumerate(('H1', 'L1'))}
                  for block in range(history + 2)]
        blocks += [{ifo: _triggers([], []) for ifo in ('H1', 'L1')}]
        for block in blocks:
            for rows in block.values():
                rows['template_id'] = np.asarray(rows['template_id'])

        def replay():
            estimator = _estimator()
            estimator.coincs = coinc_jax.JAXCoincExpireBuffer(
                3, estimator.ifos, initial_size=4096)
            outputs, states = [], []
            for block in blocks:
                outputs.append(estimator.add_singles(
                    {ifo: dict(rows) for ifo, rows in block.items()}))
                states.append(_estimator_state(estimator))
            jax.block_until_ready(jax.tree.leaves((outputs, states)))
            return outputs, states

        with _serial(monkeypatch):
            expected = replay()
        readbacks = []

        class ObserveNumpy:
            def __getattr__(self, name):
                return getattr(np, name)

            def asarray(self, value, *args, **kwargs):
                if isinstance(value, jax.Array):
                    readbacks.append((value.shape, value.dtype))
                return np.asarray(value, *args, **kwargs)

        # State inspection belongs to the test's terminal oracle boundary.
        # Matching/ranking/clustering/background expose final control ints only.
        monkeypatch.setattr(coinc_jax, 'np', ObserveNumpy())
        observer = CompilationGuard()
        try:
            replay()
            before = observer.snapshot()
            readbacks.clear()
            with observer.timed_guard():
                actual = replay()
            assert observer.snapshot() == before
        finally:
            observer.close()
        _assert_exact(actual, expected)
        assert readbacks and all(shape == (9,) and dtype == np.int64
                                 for shape, dtype in readbacks)


def test_resident_numeric_chain_has_no_device_readback(monkeypatch, device):
    """Query lengths and matching payloads reach cluster under transfer guard."""
    with scheme.JAXScheme(device):
        monkeypatch.setattr(coinc_jax, '_resident_live_device_supported', lambda _: True)
        estimator, results = _direct_case()
        estimator.coincs = coinc_jax.JAXCoincExpireBuffer(
            3, estimator.ifos, initial_size=4096)
        results['H1']['template_id'] = np.array([0, 0], np.int32)
        with jax.transfer_guard_device_to_host('disallow'):
            prepared = coinc_jax._prepare_resident_live_matches(estimator, results, ['H1'])
            assert prepared is not None
            assert estimator._jax_resident_coinc_counters['admitted'] == 1
            cluster = coinc_jax._resident_live_cluster(
                prepared[0], prepared[1], estimator.timeslide_interval,
                estimator.analysis_block + 2 * estimator.time_window)
            foreground = coinc_jax._prepare_resident_live_foreground(
                estimator, prepared[4], cluster[0], cluster[2])
            background = coinc_jax._resident_live_background(
                estimator.coincs.buffer, tuple(estimator.coincs.timer.values()),
                cluster[0], cluster[1], cluster[2], cluster[3], np.int64(0),
                (np.int32(-1),), prepared[3],
                np.asarray([estimator.background_time, coinc_jax.conv.YRJUL_SI]),
                work_size=16, active=(0,))
            jax.block_until_ready(background)
            jax.block_until_ready(foreground)
            jax.block_until_ready(cluster)
        counts = np.asarray(cluster[3])
        assert counts[2] > 0
        assert all(isinstance(value, jax.Array) for value in jax.tree.leaves(cluster))


@pytest.mark.parametrize('fixed_dtype', ['f4', 'f8'])
@pytest.mark.parametrize('shifted_dtype', ['f4', 'f8'])
def test_resident_fused_rank_preserves_native_arithmetic_bytes(
        device, fixed_dtype, shifted_dtype):
    """Fusion preserves each operand's original square rounding and promotion."""
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        arrays = []
        for dtype in (fixed_dtype, shifted_dtype):
            limits = np.finfo(dtype)
            tiny = np.nextafter(np.array(0, dtype), np.array(1, dtype))
            values = np.array([-0., tiny, -tiny, -1., 3., limits.max / 2,
                               np.inf, np.nan], dtype)
            arrays.append(jnp.asarray(np.resize(values, 257)))
        fixed, shifted = arrays
        expected = ranker.rank_stat_coinc(
            [['H1', fixed], ['L1', shifted]], None, None, None).astype(jnp.float64)
        fused = jax.jit(lambda first, second:
                        coinc_jax._resident_live_quadrature(first, second)
                        .astype(jnp.float64))
        with jax.transfer_guard_device_to_host('disallow'):
            actual = fused(fixed, shifted)
        _assert_exact(actual, expected)


@pytest.mark.parametrize('weak_operand', ['fixed', 'shifted'])
def test_resident_fused_rank_preserves_weak_operand_promotion(device, weak_operand):
    with scheme.JAXScheme(device):
        ranker = _estimator().stat_calculator
        weak = jnp.broadcast_to(jnp.asarray(2.718281828), (17,))
        strong = jnp.full(17, 3.1415925, jnp.float32)
        fixed, shifted = ((weak, strong) if weak_operand == 'fixed' else
                          (strong, weak))
        expected = ranker.rank_stat_coinc(
            [['H1', fixed], ['L1', shifted]], None, None, None).astype(jnp.float64)
        actual = jax.jit(lambda first, second:
                         coinc_jax._resident_live_quadrature(first, second)
                         .astype(jnp.float64))(fixed, shifted)
        _assert_exact(actual, expected)


@pytest.mark.parametrize('seconds', [0., 1., 4096., 31557600., 1e20])
@pytest.mark.parametrize('threshold', [0., 1., 99., np.nan])
def test_resident_background_ifar_remains_on_device_with_native_bytes(
        device, seconds, threshold):
    """Count, two IEEE divisions and saturation finish before publication."""
    with scheme.JAXScheme(device):
        values = np.array([0., 1., 2., 3., np.nan, -99., -99., -99.], 'f4')
        buffer = jnp.asarray(values)
        timers = (jnp.zeros(8, jnp.int32), jnp.zeros(8, jnp.int32))
        floating = jnp.array([threshold], jnp.float64)
        integer = jnp.array([0], jnp.int32)
        columns = (floating, integer, floating, floating,
                   integer, integer, integer, integer, integer)
        rows = jnp.array([0], jnp.int64)
        counts = jnp.array([0, 1, 1], jnp.int64)
        expected_count = np.count_nonzero(values[:5] > threshold)
        expected = np.float64(coinc_jax.conv.sec_to_year(seconds) / (expected_count + 1))
        with jax.transfer_guard_device_to_host('disallow'):
            updated, _, control, (ifar, saturated) = coinc_jax._resident_live_background(
                buffer, timers, columns, rows, rows, counts, np.int64(5), (),
                np.int64(0), np.array([seconds, coinc_jax.conv.YRJUL_SI]),
                work_size=8, active=())
        assert isinstance(ifar, jax.Array) and isinstance(saturated, jax.Array)
        _assert_exact(ifar, expected)
        _assert_exact(saturated, np.bool_(expected_count == 0))
        _assert_exact(updated[0], buffer)
        assert np.asarray(control)[5] == expected_count


@pytest.mark.parametrize('count', [2, 32, 40])
def test_resident_warmed_preparation_has_no_eager_primitive_dispatch(
        monkeypatch, device, count):
    """Device slices/conversions remain inside bounded compiled dispatches."""
    from jax._src import dispatch

    with scheme.JAXScheme(device):
        monkeypatch.setattr(coinc_jax, '_resident_live_device_supported', lambda _: True)
        estimator, results = _estimator(), {}
        ids = np.arange(count, dtype=np.int32)
        for position, ifo in enumerate(estimator.ifos):
            trigs = _triggers(ids, 100. + ids * .03125 + position * .001,
                              'f4' if position == 0 else 'f8')
            trigs['template_id'], trigs['stat'] = ids.copy(), trigs['snr']
            results[ifo] = trigs
            estimator.singles[ifo] = coinc_jax.JAXMultiRingBuffer(count, 3)
            estimator.singles[ifo].add(ids, trigs)
        estimator.trig_stat_memory = jnp.zeros(32, jnp.float32)

        def prepare():
            return coinc_jax._prepare_resident_live_matches_impl(
                estimator, results, estimator.ifos)

        prepared = prepare()
        cluster = coinc_jax._resident_live_cluster(
            prepared[0], prepared[1], 1., 8.02)

        def foreground():
            return coinc_jax._prepare_resident_live_foreground(
                estimator, prepared[4], cluster[0], cluster[2])

        foreground()
        primitive_calls = []
        previous = sys.getprofile()

        def observe(frame, event, argument):
            if event == 'call' and frame.f_code is dispatch.apply_primitive.__code__:
                primitive_calls.append(str(frame.f_locals.get('prim')))
            if previous is not None:
                previous(frame, event, argument)

        with jax.transfer_guard_device_to_host('disallow'):
            sys.setprofile(observe)
            try:
                prepared = prepare()
                selected = foreground()
            finally:
                sys.setprofile(previous)
        jax.block_until_ready((prepared, selected))
        assert not primitive_calls


@pytest.mark.parametrize('unusual,reason', [
    ('routing', 'device_template_routing_ids'),
    ('ranker', 'query_geometry_dtype_or_hooks'),
    ('background', 'background_storage_or_clock'),
    ('conversion', 'custom_background_time_or_conversion'),
    ('background_time', 'custom_background_time_or_conversion'),
    ('zero_slide', 'zero_timeslide_background_time'),
])
def test_resident_compatibility_reasons_do_not_collect_device_data(
        monkeypatch, device, unusual, reason):
    with scheme.JAXScheme(device):
        monkeypatch.setattr(coinc_jax, '_resident_live_device_supported', lambda _: True)
        estimator, results = _direct_case()
        if unusual != 'routing':
            results['H1']['template_id'] = np.array([0, 0], np.int32)
        if unusual == 'ranker':
            estimator.stat_calculator.rank_stat_coinc = lambda *_args, **_kwargs: None
        elif unusual == 'background':
            estimator.coincs.index = -1
        elif unusual == 'conversion':
            monkeypatch.setattr(coinc_jax.conv, 'sec_to_year', lambda value: value)
        elif unusual == 'background_time':
            monkeypatch.setattr(type(estimator), 'background_time',
                                property(lambda _: 1.))
        elif unusual == 'zero_slide':
            estimator.timeslide_interval = 0
        with jax.transfer_guard_device_to_host('disallow'):
            assert coinc_jax._prepare_resident_live_matches(estimator, results, ['H1']) is None
        assert estimator._jax_resident_coinc_counters == {
            'admitted': 0, 'compatibility': {reason: 1}}


def test_resident_static_capacity_reservation_does_not_collect_query_counts(
        monkeypatch, device):
    """Worst-case storage is reserved on-device without serial count fallback."""
    with scheme.JAXScheme(device):
        monkeypatch.setattr(coinc_jax, '_resident_live_device_supported', lambda _: True)
        estimator, results = _direct_case()
        estimator.coincs = coinc_jax.JAXCoincExpireBuffer(
            3, estimator.ifos, initial_size=2)
        results['H1']['template_id'] = np.array([0, 0], np.int32)
        with jax.transfer_guard_device_to_host('disallow'):
            prepared = coinc_jax._prepare_resident_live_matches(estimator, results, ['H1'])
            assert prepared is not None
            assert prepared[5][0].size >= prepared[1].size
            assert estimator._jax_resident_coinc_counters['admitted'] == 1
        assert estimator.coincs.buffer.size == 2  # Published only at the final boundary.
        np.testing.assert_array_equal(np.asarray(prepared[5][0]), 0)


@pytest.mark.parametrize('count', [0, 1, 3, 7, 33, 129])
@pytest.mark.parametrize('unusual', ['ordinary', 'ties', 'nan_stat', 'nan_time'])
def test_resident_cluster_masks_match_compact_native_selection(device, count, unusual):
    """Padding cannot affect endpoint/tie/nonfinite handling or match order."""
    with scheme.JAXScheme(device):
        size = max(count * 2 + 3, 5)
        selected = np.arange(count) * 2 + 1
        mask = np.zeros(size, bool)
        mask[selected] = True
        stats = np.arange(count, dtype=np.float64) % 5 + 7
        times = 1e9 + np.arange(count, dtype=np.float64) * .0625
        if unusual == 'ties':
            stats[:] = 7
            times[:] = 1e9
        elif unusual == 'nan_stat' and count:
            stats[::2] = np.nan
        elif unusual == 'nan_time' and count:
            times[0] = np.nan
        slides = np.resize(np.array([-1, 0, 1], np.int32), count)
        compact = (stats, slides, times, times + .001,
                   np.full(count, 1, np.int32), np.full(count, 2, np.int32),
                   np.arange(count, dtype=np.int32),
                   np.full(count, -1, np.int32), np.arange(count, dtype=np.int32))
        columns = []
        for values in compact:
            padded = np.zeros(size, dtype=values.dtype)
            padded[selected] = values
            columns.append(jnp.asarray(padded))
        packed, background, zerolag, counts = coinc_jax._resident_live_cluster(
            tuple(columns), jnp.asarray(mask), 1., .25)
        counts = np.asarray(counts)
        if count:
            cidx = np.asarray(coinc_jax.cluster_coincs(
                *(jnp.asarray(compact[index]) for index in (0, 2, 3, 1)),
                1., .25, method='cython'))
            expected_background = cidx[slides[cidx] != 0]
            expected_zerolag = cidx[slides[cidx] == 0]
        else:
            expected_background = expected_zerolag = np.array([], np.int64)
        np.testing.assert_array_equal(counts, [expected_background.size,
                                              expected_zerolag.size, count])
        np.testing.assert_array_equal(np.asarray(background)[:counts[0]],
                                      expected_background)
        np.testing.assert_array_equal(np.asarray(zerolag)[:counts[1]], expected_zerolag)
        for got, want in zip(packed, compact):
            np.testing.assert_array_equal(np.asarray(got)[:count], want)


@pytest.mark.parametrize("hook", ["rank", "matcher", "fixed_time", "chirp"])
def test_custom_hook_mutations_keep_serial_trigger_boundaries(
        monkeypatch, device, hook):
    """A hook changing the next query's ring cannot use a stale group snapshot."""
    with scheme.JAXScheme(device):
        observed = []
        outputs = []
        for serial in (True, False):
            estimator, results = _direct_case()
            calls = []

            def change():
                calls.append("call")
                if len(calls) == 1:
                    estimator.singles["L1"].buffer[0]["end_time"] = jnp.array([100.2])

            with monkeypatch.context() as patch:
                def reject(*_args, **_kwargs):
                    pytest.fail("custom hook must retain serial matching")

                patch.setattr(coinc_jax, "_time_coincidence_query_core", reject)
                if hook == "rank":
                    rank = estimator.stat_calculator.rank_stat_coinc

                    def wrapped(*args, **kwargs):
                        result = rank(*args, **kwargs)
                        change()
                        return result

                    patch.setattr(estimator.stat_calculator, "rank_stat_coinc", wrapped)
                elif hook == "matcher":
                    matcher = coinc.time_coincidence

                    def wrapped(*args, **kwargs):
                        result = matcher(*args, **kwargs)
                        change()
                        return result

                    patch.setattr(coinc, "time_coincidence", wrapped)
                elif hook == "fixed_time":
                    fixed = coinc_jax._live_fixed_time_jax

                    def wrapped(*args, **kwargs):
                        result = fixed(*args, **kwargs)
                        change()
                        return result

                    patch.setattr(coinc_jax, "_live_fixed_time_jax", wrapped)
                else:
                    conversion = coinc_jax.conv.mchirp_from_mass1_mass2

                    class Chirps(np.ndarray):
                        def __getitem__(self, item):
                            value = super().__getitem__(item)
                            if np.isscalar(item):
                                change()
                            return value

                    def wrapped(*args):
                        return np.asarray(conversion(*args)).view(Chirps)

                    patch.setattr(coinc_jax.conv, "mchirp_from_mass1_mass2", wrapped)
                with _serial(patch) if serial else nullcontext():
                    result = coinc_jax.find_coincs_jax(estimator, results, ["H1"])
                outputs.append((result, _estimator_state(estimator)))
                observed.append(calls)
        _assert_exact(outputs[1], outputs[0])
        assert observed == [["call", "call"], ["call", "call"]]


def test_later_invalid_time_column_keeps_serial_partial_state(monkeypatch, device):
    """Validation of a later trigger must not precede the first ranking update."""
    with scheme.JAXScheme(device):
        outcomes = []
        for serial in (True, False):
            estimator = _estimator()
            trigs = _triggers([0, 1], [100., 100.])
            trigs["stat"] = jnp.array([3., 4.], dtype=jnp.float32)
            for ifo in estimator.ifos:
                estimator.singles[ifo] = coinc_jax.JAXMultiRingBuffer(8, 3)
                estimator.singles[ifo].add([0, 1], trigs)
            estimator.singles["L1"].buffer[1]["end_time"] = jnp.array([100], dtype=jnp.int64)
            with _serial(monkeypatch) if serial else nullcontext():
                with pytest.raises(TypeError) as caught:
                    coinc_jax.find_coincs_jax(estimator, {"H1": trigs}, ["H1"])
            outcomes.append((str(caught.value), _estimator_state(estimator)))
        _assert_exact(outcomes[1], outcomes[0])
        np.testing.assert_array_equal(outcomes[1][1]["memory"], [3.])


def _query(rows, fixed, control, window, step, pruning=False):
    fixed = coinc_jax._prepare_live_query_times(fixed, control)
    indices, slides, counts = coinc_jax._time_coincidence_query_core(
        rows, fixed, control, window, step,
        sliding=bool(step), pruning=pruning)
    # Exactly one host control-vector read, without scientific column copies.
    counts = np.asarray(counts)
    return [coinc_jax._compact_live_query(indices, slides, fixed, i,
                                         count=int(count))
            for i, count in enumerate(counts)]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_grouped_queries_preserve_singleton_order_and_precision(device, dtype):
    """Unordered folds, endpoints, repeated slides and nonfinite times are exact."""
    cases = [
        ([.2, .1], [.1, .2], .1, 0.),
        ([1e9 + .75, 1e9 + .25], [1e9 + .125, 1e9 + .75], .2, 1.),
        ([-.8, -.2], [-.1, -.8], .2, 1.),
        ([.2, .1], [.1, .2], 2., 1.),
        ([.1, .1], [.1, .1], 2., 1.),
        ([0., .5], [.25, .5], .25, 0.),
        ([np.nextafter(0., 1.), np.nextafter(.5, 0.)], [.25, 0.], .25, 0.),
        ([np.nextafter(0., -1.), np.nextafter(.5, 1.)], [.25, 0.], .25, 0.),
        ([-.5, np.nextafter(.5, 0.)], [0., -.25], 1., 1.),
        ([np.nan, .2], [.1, np.nan], .3, 1.),
        ([np.inf, -np.inf], [.1, np.inf], .3, 1.),
        ([.2, .1], [.1, .9], 0., 1.),
        ([np.nextafter(np.float32(0), np.float32(1)), -0.], [0., -0.], .1, 0.),
        ([np.nextafter(np.float32(0), np.float32(1)),
          -np.nextafter(np.float32(0), np.float32(1))],
         [0., float(np.nextafter(np.float32(0), np.float32(1)))], 2e-45, 0.),
    ]
    with scheme.JAXScheme(device):
        for history, incoming, window, step in cases:
            rows = tuple(jnp.asarray(row, dtype=dtype)
                         for row in (history, history[::-1]))
            # Live fixed GPS time is always promoted to float64.
            fixed = jnp.asarray([incoming[1], incoming[0]], dtype=jnp.float64)
            control = jnp.array([[1, 0], [0, 0]], dtype=jnp.int64)
            expected = [coinc_jax.time_coincidence(
                row, coinc_jax._live_fixed_time_jax(fixed, i), window, step)
                for row, i in zip(rows, (1, 0))]
            actual = _query(rows, fixed, control, window, step)
            for got, want, i in zip(actual, expected, (1, 0)):
                _assert_exact(got[0], want[0])
                _assert_exact(got[1], want[2])
                _assert_exact(got[2], coinc_jax._live_fixed_time_jax(fixed, i))
                assert got[0].dtype == np.int64
                assert got[1].dtype == np.int32
                assert got[2].dtype == np.float64


def test_grouped_query_matches_native_cpu_endpoint_oracle(device):
    """The independent C matcher checks asymmetrical endpoints and pair order."""
    histories = ([.2, .1], [1e9 + .75, 1e9 + .25], [-.8, -.2], [.2, .1])
    incoming = (.1, 1e9 + .125, -.1, .1)
    windows, steps = (.1, .2, .2, 1.), (0., 1., 1., 1.)
    expected = [coinc.time_coincidence(np.asarray(row, "f8"),
                                     np.asarray([fixed], "f8"), window, step)
                for row, fixed, window, step in zip(histories, incoming, windows, steps)]
    with scheme.JAXScheme(device):
        for row, fixed, window, step, want in zip(histories, incoming, windows, steps, expected):
            rows = (jnp.asarray(row), jnp.asarray(row))
            control = jnp.array([[0, 0], [0, 0]], dtype=jnp.int64)
            actual = _query(rows, jnp.array([fixed]), control, window, step)
            for indices, slides, _ in actual:
                np.testing.assert_array_equal(indices, want[0])
                np.testing.assert_array_equal(slides, want[2])


def test_grouped_expiry_masks_keep_pruned_relative_indices_and_nonfinite_order(device):
    """Expired rows cannot leak matches or change indices in the pruned ring."""
    with scheme.JAXScheme(device):
        rows = (jnp.array([np.nan, .75, .1, .5]),
                jnp.array([.2, -.3, np.nan, .1]),
                jnp.array([.1, .2, .3, .4]))
        fixed = jnp.array([.1, .2, .3])
        control = jnp.array([[0, 1, 2], [1, 2, 4]], dtype=jnp.int64)
        actual = _query(rows, fixed, control, .5, 1., pruning=True)
        for pos, (indices, slides, fixed_time) in enumerate(actual):
            expected = coinc_jax.time_coincidence(
                rows[pos][(1, 2, 4)[pos]:], fixed[pos:pos + 1], .5, 1.)
            _assert_exact(indices, expected[0])
            _assert_exact(slides, expected[2])
            _assert_exact(fixed_time, fixed[pos:pos + 1])


def test_grouped_match_core_reuses_executable_for_changed_queries_and_expiry(device):
    """Count changes are data; the bounded matching geometry reuses its program."""
    with scheme.JAXScheme(device):
        rows = (jnp.array([.1, .3]), jnp.array([.5, .7]))
        changed = (jnp.array([.2, .4]), jnp.array([.4, .8]))
        fixed = jnp.array([.1, .5, .2, .4])
        controls = (jnp.array([[0, 1], [0, 0]], dtype=jnp.int64),
                    jnp.array([[2, 3], [1, 2]], dtype=jnp.int64))
        query_times = tuple(coinc_jax._prepare_live_query_times(fixed, control)
                            for control in controls)

        def run(values, times, control):
            return coinc_jax._time_coincidence_query_core(
                values, times, control, .11, 1., sliding=True, pruning=True)

        observer = CompilationGuard()
        try:
            jax.block_until_ready(run(rows, query_times[0], controls[0]))
            jax.block_until_ready((changed, query_times, controls))
            before = observer.snapshot()
            with observer.timed_guard():
                actual = run(changed, query_times[1], controls[1])
                jax.block_until_ready(actual)
            assert observer.snapshot() == before
            assert np.asarray(actual[2]).tolist() == [0, 0]
            # A new history geometry must still be rejected during timing.
            larger = tuple(jnp.pad(value, (0, 1)) for value in rows)
            jax.block_until_ready(larger)
            with pytest.raises(RuntimeError, match="trac"):
                with observer.timed_guard():
                    run(larger, query_times[0], controls[0])
        finally:
            observer.close()


def test_preparation_reads_one_count_vector_and_keeps_science_resident(
        monkeypatch, device):
    """Eight singleton queries require one host count read with no time download."""
    with scheme.JAXScheme(device):
        estimator = _estimator()
        trigs = _triggers([0, 1, 0, 2, 3, 2, 0, 1],
                          [100., 100.1, 100.001, 100.2, 100.3, 100.201, 100., 100.1])
        trigs["stat"] = trigs["snr"]
        ring = coinc_jax.JAXMultiRingBuffer(8, 3)
        ring.add([0, 1, 2, 3], {"end_time": jnp.array([100., 100.1, 100.2, 100.3]),
                                "stat": jnp.ones(4, dtype=jnp.float32)})
        ring.add([4], {"end_time": jnp.array([100.4]),
                       "stat": jnp.ones(1, dtype=jnp.float32)})
        counts_read, calls = [], []
        core = coinc_jax._time_coincidence_query_core

        class ObserveNumpy:
            def __getattr__(self, name):
                return getattr(np, name)

            def asarray(self, value, *args, **kwargs):
                if isinstance(value, jax.Array):
                    assert value.dtype == np.int64 and value.shape == (8,)
                    counts_read.append(value.shape)
                return np.asarray(value, *args, **kwargs)

        def record(*args, **kwargs):
            calls.append(len(args[0]))
            return core(*args, **kwargs)

        monkeypatch.setattr(coinc_jax, "np", ObserveNumpy())
        monkeypatch.setattr(coinc_jax, "_time_coincidence_query_core", record)
        prepared = coinc_jax._prepare_live_queries(
            ring, trigs, np.asarray([0, 1, 0, 2, 3, 2, 0, 1], "i4"), 0, 8,
            estimator.stat_calculator, .01, 1.)
        assert list(prepared) == list(range(8))
        assert calls == [8] and counts_read == [(8,)]
        for index, query in prepared.items():
            packed, slides, fixed, position, count = query
            actual = coinc_jax._compact_live_query(
                packed, slides, fixed, np.int64(position), count=count)
            expected = coinc_jax.time_coincidence(
                ring.data(int(np.asarray(trigs["template_id"])[index]))["end_time"],
                coinc_jax._live_fixed_time_jax(trigs["end_time"], index), .01, 1.)
            _assert_exact(actual[0], expected[0])
            _assert_exact(actual[1], expected[2])
            assert actual[2].devices() == trigs["end_time"].devices()


@pytest.mark.parametrize("unusual", ["large_history", "large_incoming", "host_times",
                                     "wrapped_times", "wrapped_f32", "clock_order", "custom_data",
                                     "negative_template", "nonfinite_window",
                                     "negative_slide", "query_cap"])
def test_unusual_query_admission_preserves_serial_fallback(monkeypatch, device, unusual):
    """Unsupported geometry and storage never eagerly validate or decode a group."""
    with scheme.JAXScheme(device):
        estimator, results = _direct_case()
        trigs = results["H1"]
        ring = estimator.singles["L1"]
        templates = np.asarray([0, 0], "i4")
        first, last, window, step = 0, 2, .01, 1.
        assert coinc_jax._can_prepare_live_queries(
            ring, trigs, templates, first, last, estimator.stat_calculator, window, step)
        if unusual == "large_history":
            ring = coinc_jax.JAXMultiRingBuffer(8, 3)
            ring.add(np.zeros(65, "i4"),
                     {"end_time": jnp.arange(65, dtype=jnp.float64),
                      "stat": jnp.ones(65, dtype=jnp.float32)})
        elif unusual in ("large_incoming", "query_cap"):
            size = 4097 if unusual == "large_incoming" else 9
            trigs = _triggers(np.zeros(size, "i4"), np.full(size, 100.))
            trigs["stat"] = trigs["snr"]
            templates = np.zeros(size, "i4")
            last = 2 if unusual == "large_incoming" else 9
        elif unusual == "host_times":
            trigs["end_time"] = np.asarray(trigs["end_time"])
        elif unusual in ("wrapped_times", "wrapped_f32"):
            if unusual == "wrapped_f32":
                trigs["end_time"] = jnp.asarray(
                    [np.nextafter(np.float32(0), np.float32(1)), -0.], dtype=jnp.float32)
            trigs["end_time"] = Array(JAXArrayData(trigs["end_time"]), copy=False)
        elif unusual == "clock_order":
            ring.add([0], {"end_time": jnp.array([100.2]),
                           "stat": jnp.array([10.], dtype=jnp.float32)})
            ring._expire_times[0] = np.array([1, 0], "i8")
        elif unusual == "custom_data":
            original = ring.data
            ring.data = lambda *args: original(*args)
        elif unusual == "negative_template":
            templates[0] = -1
        elif unusual == "nonfinite_window":
            window = np.nan
        elif unusual == "negative_slide":
            step = -1.

        def reject(*_args, **_kwargs):
            pytest.fail("unusual query must not dispatch grouped matching")

        monkeypatch.setattr(coinc_jax, "_time_coincidence_query_core", reject)
        assert not coinc_jax._can_prepare_live_queries(
            ring, trigs, templates, first, last, estimator.stat_calculator, window, step)
        assert coinc_jax._prepare_live_queries(
            ring, trigs, templates, first, last, estimator.stat_calculator, window, step) == {}


@pytest.mark.parametrize("history", [64, 65])
def test_history_capacity_boundary_keeps_complete_search_outputs(monkeypatch, device, history):
    """The bounded matcher and its larger-ring fallback both keep serial results."""
    with scheme.JAXScheme(device):
        outcomes = []
        for serial in (True, False):
            estimator, results = _direct_case()
            ring = coinc_jax.JAXMultiRingBuffer(8, 3)
            rows = _triggers(np.zeros(history, "i4"), 99. + np.arange(history) * .001)
            rows["stat"] = rows["snr"]
            ring.add(np.zeros(history, "i4"), rows)
            estimator.singles["L1"] = ring
            with _serial(monkeypatch) if serial else nullcontext():
                result = coinc_jax.find_coincs_jax(estimator, results, ["H1"])
            outcomes.append((result, _estimator_state(estimator)))
        _assert_exact(outcomes[1], outcomes[0])


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_fixed_query_gather_preserves_raw_slice_promotion(device, dtype):
    """A device gather retains raw time casts, including signed zero/subnormals."""
    with scheme.JAXScheme(device):
        tiny = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))
        times = jnp.asarray([tiny, -tiny, 0., -0., 1e9 + .25], dtype=dtype)
        ids = [4, 1, 3, 0, 2]
        control = jnp.asarray([ids, [0] * len(ids)], dtype=jnp.int64)
        actual = coinc_jax._prepare_live_query_times(times, control)
        expected = jnp.concatenate([coinc_jax._live_fixed_time_jax(times, index)
                                    for index in ids])
        _assert_exact(actual, expected)


def test_x64_disabled_retains_first_query_cast_before_matcher_enable(
        monkeypatch, device):
    """Enabling precision early must not invent a first GPS coincidence."""
    original_flag = jax.config.jax_enable_x64
    try:
        with scheme.JAXScheme(device):
            outcomes, observed = [], []
            compact = coinc_jax._compact_time_coincidence
            for serial in (True, False):
                jax.config.update("jax_enable_x64", True)
                estimator = _estimator()
                estimator.time_window = .001
                trigs = _triggers([0, 0], [1e9 + .1, 1e9 + .1])
                trigs["stat"] = trigs["snr"]
                for ifo in estimator.ifos:
                    estimator.singles[ifo] = coinc_jax.JAXMultiRingBuffer(8, 3)
                estimator.singles["H1"].add([0, 0], trigs)
                estimator.singles["L1"].add(
                    [0], {key: value[:1] for key, value in trigs.items()})
                counts = []

                def record(*args, count):
                    counts.append(count)
                    return compact(*args, count=count)

                def reject(*_args, **_kwargs):
                    pytest.fail("x64-disabled first chunk must retain singleton dispatch")

                with monkeypatch.context() as patch:
                    patch.setattr(coinc_jax, "_compact_time_coincidence", record)
                    patch.setattr(coinc_jax, "_time_coincidence_query_core", reject)
                    jax.config.update("jax_enable_x64", False)
                    assert coinc_jax._prepare_live_queries(
                        estimator.singles["L1"], trigs, np.array([0, 0], "i4"),
                        0, 2, estimator.stat_calculator, .001, 1.) == {}
                    assert not jax.config.jax_enable_x64
                    with _serial(patch) if serial else nullcontext():
                        with pytest.warns(UserWarning, match="float64"):
                            result = coinc_jax.find_coincs_jax(
                                estimator, {"H1": trigs}, ["H1"])
                    assert jax.config.jax_enable_x64
                outcomes.append((result, _estimator_state(estimator)))
                observed.append(counts)
            _assert_exact(outcomes[1], outcomes[0])
            # The first narrowed query misses; the second query runs after
            # the native matcher has enabled x64 and matches the stored time.
            assert observed == [[0, 1], [0, 1]]
    finally:
        jax.config.update("jax_enable_x64", original_flag)
