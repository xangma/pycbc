"""Regression tests for backend-independent runtime optimizations."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pycbc.events.eventmgr import EventManager
from pycbc.frame import frame
from pycbc.types import TimeSeries
from pycbc.waveform import plugin


@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize('bounds', ['both', 'start', 'end', 'neither'])
def test_frame_read_preserves_dtype_values_and_bounds(tmp_path, dtype, bounds):
    """The REAL8 shortcut must fall back without losing non-REAL8 data."""
    values = np.arange(32, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * values[::-1]
    series = TimeSeries(values, delta_t=0.25, epoch=1000000000)
    path = str(tmp_path / 'roundtrip.gwf')
    frame.write_frame(path, ['H1:X', 'H1:Y'], [series, series * 2])
    kwargs = {}
    if bounds in ('both', 'start'):
        kwargs['start_time'] = series.start_time
    if bounds in ('both', 'end'):
        kwargs['end_time'] = series.end_time
    actual = frame.read_frame(path, ['H1:X', 'H1:Y'], **kwargs)
    for result, expected in zip(actual, [series, series * 2]):
        assert result.dtype == expected.dtype
        assert result.start_time == expected.start_time
        assert result.end_time == expected.end_time
        np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_explicit_frame_bounds_skip_duration_metadata(tmp_path, monkeypatch):
    series = TimeSeries(np.arange(32, dtype=np.float64),
                        delta_t=0.25, epoch=1000000000)
    path = str(tmp_path / 'bounded.gwf')
    frame.write_frame(path, 'H1:X', series)

    def unexpected_metadata(*args):
        raise AssertionError('Unexpected metadata with explicit bounds')

    monkeypatch.setattr(frame.lalframe, 'FrStreamGetVectorLength',
                        unexpected_metadata)
    result = frame.read_frame(path, 'H1:X',
                              start_time=series.start_time + 1,
                              end_time=series.start_time + 3)
    np.testing.assert_array_equal(result.numpy(), series.numpy()[4:12])
    assert result.start_time == series.start_time + 1


@pytest.mark.parametrize('group,registration', [
    ('fd', 'waveform'), ('fd_det', 'waveform'),
    ('fd_sequence', 'waveform'), ('fd_det_sequence', 'waveform'),
    ('td', 'waveform'), ('length', 'length'), ('end_freq', 'end_freq'),
])
def test_plugin_registration_is_lazy_and_loads_once(monkeypatch, group,
                                                    registration):
    loaded = []
    registered = {}

    def target(value=0):
        return value + 1

    target.description = 'test plugin'

    def load():
        loaded.append(True)
        return target

    entry = SimpleNamespace(name='TestLazyPlugin', load=load)
    selected = 'pycbc.waveform.' + group
    monkeypatch.setattr(
        plugin, 'entry_points',
        lambda *, group: [entry] if group == selected else []
    )
    for kind, name in [('waveform', 'add_custom_waveform'),
                       ('length', 'add_length_estimator'),
                       ('end_freq', 'add_end_frequency_estimator')]:
        def register(name, function, *args, kind=kind, **kwargs):
            registered[kind] = function
        monkeypatch.setattr(plugin, name, register)

    plugin.retrieve_waveform_plugins()
    assert not loaded
    wrapper = registered[registration]
    assert wrapper(value=4) == 5
    assert wrapper.description == 'test plugin'
    assert wrapper(9) == 10
    assert loaded == [True]


@pytest.mark.parametrize('ncores,expected_templates,expected_filters', [
    (None, 200., 50.), (1, 200., 50.), (2, 100., 25.),
])
def test_event_performance_handles_unspecified_ncores(
        tmp_path, ncores, expected_templates, expected_filters):
    opts = SimpleNamespace(channel_name='H1:TEST', trig_start_time=None,
                           trig_end_time=None, gps_start_time=1000000000,
                           gps_end_time=1000000020, segment_start_pad=0,
                           segment_end_pad=0)
    manager = EventManager(opts, [], [], array_minsize=1)
    manager.write_performance = True
    manager.ncores = ncores
    manager.ntemplates = 100
    manager.nfilters = 500
    manager.run_time = 10
    manager.setup_time = 2
    path = str(tmp_path / 'events.hdf')
    manager.write_to_hdf(path)
    with h5py.File(path) as result:
        assert result['H1/search/templates_per_core'][0] == expected_templates
        assert result['H1/search/filter_rate_per_core'][0] == expected_filters


def test_calibration_file_selection_uses_run_lookup(tmp_path, monkeypatch):
    from pycbc.frame import gwosc
    from pycbc.strain.recalibrate import get_calibration_files

    calls = []

    def get_run(gps_time):
        calls.append(gps_time)
        return 'O4a'

    monkeypatch.setattr(gwosc, 'get_run', get_run)
    directory = tmp_path / 'H1_O4a'
    directory.mkdir()
    for gps in (1000000000, 1000000010):
        (directory / f'calibration_{gps}.txt').touch()
    actual = get_calibration_files(['H1'], 1000000002, str(tmp_path))
    assert actual == {'H1': str(directory / 'calibration_1000000000.txt')}
    assert calls == [1000000002]
