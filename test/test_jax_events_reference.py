"""Native dispatch and isolated numerical controls for JAX events."""

import os
import subprocess
import sys
import copy
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from pycbc import scheme
from pycbc.events import cuts, ranking, veto
from pycbc.events.eventmgr import EventManager, findchirp_cluster_over_window
from pycbc.events.eventmgr_jax import JAXEventManager
from pycbc.events import coinc, coinc_jax
from pycbc.events.stat import QuadratureSumStatistic


def exact(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_cpu_events_do_not_import_jax_or_change_getter_types():
    code = """
import sys
import numpy as np
from pycbc.events import ranking, cuts, coinc
trigs = {'snr': np.array([5., 8.], dtype=np.float32),
         'chisq': np.array([1., 4.], dtype=np.float32),
         'chisq_dof': np.array([2, 3], dtype=np.uint32)}
for result in (ranking.get_snr(trigs), ranking.get_newsnr(trigs),
               ranking.newsnr(trigs['snr'], np.ones(2)),
               ranking.effsnr(trigs['snr'], np.ones(2)),
               cuts.apply_trigger_cuts(trigs, {('snr', np.greater): 6})):
    assert isinstance(result, np.ndarray)
assert 'jax' not in sys.modules
background = coinc.LiveCoincTimeslideBackgroundEstimator(
    1, 1, 'single_ranking_only', 'snr', [], ['H1', 'L1'], ifar_limit=.00001)
assert type(background.coincs) is coinc.CoincExpireBuffer
background.set_singles_buffer({'H1': {'snr': np.array([1.]),
    'end_time': np.array([1.]), 'template_id': np.array([0])}})
assert all(type(value) is coinc.MultiRingBuffer for value in background.singles.values())
assert 'jax' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.parametrize('operation', ['newsnr', 'effsnr'])
@pytest.mark.parametrize('scalar', [False, True])
def test_original_ranking_controls(operation, scalar):
    snr = np.float32(5.500001) if scalar else np.array([5.500001, 8.25], np.float32)
    chisq = np.float32(1.000001) if scalar else np.array([1.000001, 2.75], np.float32)
    with scheme.CPUScheme():
        expected = getattr(ranking, operation)(snr, chisq)
    with scheme.JAXScheme('cpu', reference_operations=[operation]):
        actual = getattr(ranking, operation)(jnp.asarray(snr), jnp.asarray(chisq))
        exact(actual, expected)
        assert actual.device == scheme.mgr.state.jax_device


@pytest.mark.parametrize('outside', [False, True])
def test_unsorted_segment_veto_and_native_tie_order(outside):
    function = veto.indices_outside_times if outside else veto.indices_within_times
    times = np.array([.5, 2., 3.5])
    start, end = np.array([3., 0.]), np.array([4., 1.])
    with scheme.CPUScheme():
        expected = function(times, start, end)
    with scheme.JAXScheme('cpu'):
        exact(function(times, start, end), expected)
    tied = np.tile(np.array([0., 1., 2.]), 12)
    with scheme.CPUScheme():
        expected = function(tied, np.array([0.]), np.array([3.]))
    with scheme.JAXScheme('cpu', reference_operations=['segment_veto']):
        exact(function(jnp.asarray(tied), jnp.array([0.]), jnp.array([3.])), expected)


def test_empty_segment_veto_dtype():
    with scheme.CPUScheme():
        expected = veto.indices_within_times(np.array([.5]), np.array([]), np.array([]))
    with scheme.JAXScheme('cpu'):
        exact(veto.indices_within_times(np.array([.5]), np.array([]), np.array([])), expected)


def test_trigger_cuts_keep_derived_values_on_device():
    triggers = {'snr': np.array([5., 8., 9.], np.float32),
                'chisq': np.array([1., 4., 8.], np.float32),
                'chisq_dof': np.array([2, 3, 3], np.uint32)}
    for parameter, threshold in [('snr', 6), ('traditional_chisq', 1.5), ('newsnr', 7)]:
        options = {(parameter, np.greater): threshold}
        with scheme.CPUScheme():
            expected = cuts.apply_trigger_cuts(triggers, options)
        with scheme.JAXScheme('cpu'):
            actual = cuts.apply_trigger_cuts(triggers, options)
            exact(actual, expected)
            assert isinstance(actual, jax.Array)


def test_findchirp_preserves_original_int32_times():
    times, values = np.array([.9, 1.95]), np.array([1., 2.])
    with scheme.CPUScheme():
        expected = findchirp_cluster_over_window(times, values, 1)
    with scheme.JAXScheme('cpu'):
        exact(findchirp_cluster_over_window(times, values, 1), expected)
    with scheme.JAXScheme('cpu', reference_operations=['findchirp_cluster']):
        exact(findchirp_cluster_over_window(times, values, 1), expected)


def manager(cls, dof_dtype=int):
    result = cls(SimpleNamespace(chisq_bins=True),
                 ['time_index', 'snr', 'chisq', 'chisq_dof'],
                 [int, complex, float, dof_dtype])
    result.new_template()
    result.add_template_events(['time_index', 'snr', 'chisq', 'chisq_dof'],
                              [np.array([1, 2]), np.array([5+0j, 5+0j]),
                               np.array([3., 5.]), np.array([3, 4])])
    result.finalize_template_events()
    return result


@pytest.mark.parametrize('method, options', [
    ('newsnr_threshold', {'threshold': 1}),
    ('chisq_threshold', {'value': 2, 'num_bins': 4}),
    ('keep_loudest_in_interval', {'window': 10, 'num_keep': 1}),
])
def test_pending_event_selections_match_original(method, options):
    with scheme.CPUScheme():
        cpu = manager(EventManager)
        getattr(cpu, method)(**options)
        expected = cpu.events.copy()
    with scheme.JAXScheme('cpu'):
        device = manager(JAXEventManager)
        getattr(device, method)(**options)
        for name in expected.dtype.names:
            exact(device.events[name], expected[name])
            assert isinstance(device.events[name], jax.Array)


@pytest.mark.parametrize('method, selector, options', [
    ('newsnr_threshold', 'event_newsnr_threshold', {'threshold': 1}),
    ('chisq_threshold', 'event_chisq_threshold', {'value': 2, 'num_bins': 4}),
    ('keep_loudest_in_interval', 'event_loudest', {'window': 10, 'num_keep': 1}),
])
def test_original_event_selection_controls(method, selector, options):
    with scheme.CPUScheme():
        cpu = manager(EventManager)
        getattr(cpu, method)(**options)
        expected = cpu.events.copy()
    with scheme.JAXScheme('cpu', reference_operations=[selector]):
        device = manager(JAXEventManager)
        getattr(device, method)(**options)
        for name in expected.dtype.names:
            exact(device.events[name], expected[name])
            assert device.events[name].device == scheme.mgr.state.jax_device


def test_event_boundaries_align_cross_device_inputs():
    code = """
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from pycbc import scheme
from pycbc.events import ranking, cuts, veto, coinc
from pycbc.events.eventmgr import findchirp_cluster_over_window
from pycbc.events.eventmgr_jax import JAXEventManager
from pycbc.events.coinc_jax import JAXMultiRingBuffer, JAXCoincExpireBuffer
source = jax.devices('cpu')[0]
snr = jax.device_put(np.array([5., 8.]), source)
chisq = jax.device_put(np.array([1., 2.]), source)
for controls in [(), ('newsnr', 'segment_veto', 'findchirp_cluster',
                      'time_coincidence', 'cluster_over_time')]:
    with scheme.JAXScheme('1', reference_operations=controls):
        target = scheme.mgr.state.jax_device
        outputs = [ranking.newsnr(snr, chisq),
                   ranking.get_snr({'snr': snr}),
                   ranking.get_newsnr({'snr': snr, 'chisq': chisq, 'chisq_dof': np.array([2, 3])}),
                   cuts.apply_trigger_cuts({'snr': snr}, {('snr', np.greater): 6}),
                   veto.indices_within_times(snr, np.array([0.]), np.array([10.])),
                   findchirp_cluster_over_window(np.array([1, 2]), snr, 1),
                   coinc.cluster_over_time(snr, np.array([1., 2.]), .1),
                   *coinc.time_coincidence(snr, np.array([5.]), .1)]
        assert all(value.device == target for value in outputs)
        manager = JAXEventManager(SimpleNamespace(), ['time_index', 'snr'], [int, float])
        manager.new_template()
        manager.add_template_events(['time_index', 'snr'], [np.array([1, 2]), snr])
        assert all(value.device == target for value in manager.template_events.values())
        manager.finalize_template_events()
        assert all(value.device == target for value in manager.events.values())
        rings = JAXMultiRingBuffer(1, 2)
        rings.add(np.array([0, 0]), {'snr': snr})
        assert rings.data(0)['snr'].device == target
        buffer = JAXCoincExpireBuffer(2, ['H1', 'L1'], initial_size=2)
        buffer.add(snr, {'H1': np.array([0, 0]), 'L1': np.array([0, 0])}, ['H1', 'L1'])
        assert buffer.data.device == target
        if not controls:
            result = jax.jit(ranking.newsnr)(jnp.array([5., 8.]), jnp.array([1., 2.]))
            assert result.device == target
"""
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    subprocess.run([sys.executable, '-c', code], check=True, env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.parametrize('operation', ['time_coincidence', 'cluster_over_time',
                                      'cluster_coincs', 'cluster_coincs_multiifo'])
def test_original_coincidence_controls_preserve_tie_order(operation):
    times = np.tile(np.array([10., 11., 12.]), 12)
    stat = np.ones(times.size)
    if operation == 'time_coincidence':
        args = (times, np.array([10.]), .1)
    elif operation == 'cluster_over_time':
        args = (stat, times, .1)
    elif operation == 'cluster_coincs':
        args = (stat, times, times, np.zeros(times.size, np.int32), 0., .1)
    else:
        args = (stat, (times, times), np.zeros(times.size, np.int32), 0., .1)
    function = getattr(coinc, operation)
    with scheme.CPUScheme():
        expected = function(*args)
    with scheme.JAXScheme('cpu', reference_operations=[operation]):
        actual = function(*args)
        if operation == 'time_coincidence':
            for got, want in zip(actual, expected):
                exact(got, want)
        else:
            exact(actual, expected)


def test_original_coincidence_geometry_at_gps_rounding_boundary():
    base = 1e9
    first = np.array([base, base + .25])
    second = np.array([base + np.spacing(base), base + .25])
    args = (np.array([1., 2.]), first, second, np.zeros(2, np.int32), 0., .25)
    with scheme.CPUScheme():
        expected = coinc.cluster_coincs(*args)
    with scheme.JAXScheme('cpu', reference_operations=['cluster_coincs']):
        exact(coinc.cluster_coincs(*args), expected)


def test_original_quadrature_sum_control():
    first = np.random.default_rng(42).normal(size=100).astype(np.float32)
    second = np.random.default_rng(43).normal(size=100).astype(np.float32)
    args = ([['H1', first], ['L1', second]], None, None, None)
    statistic = object.__new__(QuadratureSumStatistic)
    with scheme.CPUScheme():
        expected = statistic.rank_stat_coinc(*args)
    with scheme.JAXScheme('cpu', reference_operations=['quadrature_sum']):
        exact(statistic.rank_stat_coinc(*args), expected)


def test_quadrature_default_uses_original_square_and_sqrt_operations():
    first = np.random.default_rng(42).normal(size=100).astype(np.float32)
    second = np.random.default_rng(43).normal(size=100).astype(np.float32)
    statistic = object.__new__(QuadratureSumStatistic)
    inputs = [['H1', first], ['L1', second]]
    with scheme.CPUScheme():
        expected = statistic.rank_stat_coinc(inputs, None, None, None)
    with scheme.JAXScheme('cpu'):
        actual = statistic.rank_stat_coinc(inputs, None, None, None)
        exact(actual, expected)
        exact(coinc_jax._resident_live_quadrature(
            jnp.asarray(first), jnp.asarray(second)), expected)


@pytest.mark.parametrize('ifars, stats', [
    ((0., 0.), (0., 0.)), ((-1., -2.), (1., 2.)),
    ((np.nan, 2.), (0., 0.)), ((2., 2.), (1., np.nan)),
    ((2., 2.), (np.nan, 1.)), ((2., 2.), (1., 2.)),
])
def test_best_coincidence_keeps_original_ordered_comparisons(ifars, stats):
    values = [{'coinc_possible': True, 'foreground/ifar': ifar,
               'foreground/stat': np.array([stat]), 'foreground/type': str(index)}
              for index, (ifar, stat) in enumerate(zip(ifars, stats))]
    estimator = coinc.LiveCoincTimeslideBackgroundEstimator
    with scheme.CPUScheme():
        expected = estimator.pick_best_coinc(copy.deepcopy(values))
    with scheme.JAXScheme('cpu'):
        actual = coinc_jax.JAXLiveCoincTimeslideBackgroundEstimator.pick_best_coinc(
            copy.deepcopy(values))
        assert actual['foreground/type'] == expected['foreground/type']
        exact(actual['foreground/ifar'], expected['foreground/ifar'])
