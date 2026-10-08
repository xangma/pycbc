# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Detector admission and failure ordering around batched PSD preparation."""

from types import SimpleNamespace

import pytest

from pycbc.strain import live_data_jax as live


def _prepare(monkeypatch, readers, counts, recalculated, updates, events):
    from pycbc.strain import live_psd_jax
    monkeypatch.setattr(live, '_native_readers',
                        lambda readers, ifos: tuple(readers[i] for i in ifos))

    def prepare(buffers):
        events.append(('prepare', tuple(b.name for b in buffers)))
        return tuple(updates[b.name] for b in buffers)

    monkeypatch.setattr(live_psd_jax, 'prepare_psd_updates_jax', prepare)
    return live.prepare_live_data_jax(
        readers, tuple(readers), counts, recalculated, increment=8, timeout=30,
        interval=3, min_distance=5, max_distance=20,
        stage_event=lambda *a, **k: None, data_start=100)


def _reader(name, events, *, available=True, distance_good=True, error=None):
    def advance(increment, timeout):
        events.append(('advance', name, increment, timeout))
        if error is not None:
            raise error
        return available

    def check(minimum, maximum):
        events.append(('range', name, minimum, maximum))
        return distance_good

    return SimpleNamespace(name=name, advance=advance, check_psd_dist=check)


def _pending(name, events, *, valid=True, error=None):
    def result():
        events.append(('resolve', name))
        if error is not None:
            raise error
        return valid

    return SimpleNamespace(result=result)


def test_filter_turn_precedes_later_refresh_failure(monkeypatch):
    events = []
    readers = {i: _reader(i, events) for i in ('H1', 'L1')}
    failure = ValueError('math domain error')
    updates = {'H1': _pending('H1', events),
               'L1': _pending('L1', events, error=failure)}
    counts = {'H1': 0, 'L1': 0}
    recalculated = {'H1': False, 'L1': False}
    prepared = _prepare(monkeypatch, readers, counts, recalculated,
                        updates, events)
    assert not any(e[0] == 'resolve' for e in events)
    assert prepared.status('H1') is True
    events.append(('filter', 'H1'))
    with pytest.raises(ValueError) as caught:
        prepared.status('L1')
    assert caught.value is failure
    assert events.index(('filter', 'H1')) < events.index(('resolve', 'L1'))
    assert counts == {'H1': 2, 'L1': 0}
    assert recalculated == {'H1': True, 'L1': False}


def test_later_frame_error_retains_earlier_detector_admission(monkeypatch):
    events = []
    failure = OSError('later frame unavailable')
    readers = {'H1': _reader('H1', events),
               'L1': _reader('L1', events, error=failure),
               'V1': _reader('V1', events)}
    counts = dict.fromkeys(readers, 0)
    recalculated = dict.fromkeys(readers, False)
    prepared = _prepare(monkeypatch, readers, counts, recalculated,
                        {'H1': _pending('H1', events)}, events)
    assert prepared.status('H1') is True
    events.append(('filter', 'H1'))
    with pytest.raises(OSError) as caught:
        prepared.status('L1')
    assert caught.value is failure
    assert not any(e[:2] == ('advance', 'V1') for e in events)
    assert ('prepare', ('H1',)) in events
    assert counts == {'H1': 2, 'L1': 0, 'V1': 0}


def test_unavailable_range_and_recompute_counters(monkeypatch):
    events = []
    readers = {'H1': _reader('H1', events, available=False),
               'L1': _reader('L1', events, distance_good=False),
               'V1': _reader('V1', events)}
    counts = {'H1': 2, 'L1': 1, 'V1': 0}
    recalculated = dict.fromkeys(readers, False)
    prepared = _prepare(monkeypatch, readers, counts, recalculated,
                        {'V1': _pending('V1', events, valid=False)}, events)
    assert prepared.status('H1') is False
    assert prepared.status('L1') is False
    assert prepared.status('V1') is False
    assert counts == {'H1': 0, 'L1': 0, 'V1': 2}
    assert recalculated == {'H1': False, 'L1': False, 'V1': True}
    # A repeated status lookup cannot apply the update or decrement twice.
    assert prepared.status('V1') is False
    assert events.count(('resolve', 'V1')) == 1
    assert ('prepare', ('V1',)) in events


@pytest.mark.parametrize('kind', ['custom', 'shared', 'overridden',
                                 'shared-strain', 'shared-raw'])
def test_non_native_readers_keep_original_serial_path(kind):
    if kind == 'custom':
        readers = {'H1': object(), 'L1': object()}
    else:
        first = live.StrainBuffer.__new__(live.StrainBuffer)
        second = live.StrainBuffer.__new__(live.StrainBuffer)
        first.strain, first.raw_buffer = object(), object()
        second.strain, second.raw_buffer = object(), object()
        if kind == 'shared':
            second = first
        elif kind == 'shared-strain':
            second.strain = first.strain
        elif kind == 'shared-raw':
            second.raw_buffer = first.raw_buffer
        else:
            second.advance = lambda *a, **k: pytest.fail('custom advance ran')
        readers = {'H1': first, 'L1': second}
    assert live.prepare_live_data_jax(
        readers, tuple(readers), {}, {}, increment=8, timeout=30, interval=1,
        min_distance=0, max_distance=100,
        stage_event=lambda *a, **k: pytest.fail('fallback advanced a reader'),
        data_start=100) is None


@pytest.mark.parametrize('count,available,refreshed,distance,expected,new_count', [
    (0, True, True, True, True, 2),
    (0, True, False, True, False, 2),
    (0, True, True, False, False, 2),
    (2, True, True, True, True, 1),
    (2, False, True, True, False, 0),
])
def test_single_reader_serial_admission(count, available, refreshed, distance,
                                       expected, new_count):
    events = []
    reader = _reader('H1', events, available=available,
                     distance_good=distance)

    def recalculate():
        events.append(('refresh', 'H1'))
        return refreshed

    reader.recalculate_psd = recalculate
    counts, recalculated = {'H1': count}, {'H1': False}
    options = dict(increment=8, timeout=30, interval=3,
                   min_distance=5, max_distance=20,
                   stage_event=lambda *a, **k: None, data_start=100)
    assert live.prepare_live_data_jax(
        {'H1': reader}, ('H1',), counts, recalculated, **options) is None
    assert live.advance_live_data_jax(
        reader, 'H1', counts, recalculated, **options) is expected
    assert counts == {'H1': new_count}
    assert recalculated == {'H1': available and count == 0}
    assert events == ([('advance', 'H1', 8, 30)]
                      + ([('refresh', 'H1')] if available and count == 0 else [])
                      + [('range', 'H1', 5, 20)])


def test_custom_readers_keep_serial_failure_and_filter_order():
    events = []
    failure = OSError('later reader unavailable')
    readers = {'H1': _reader('H1', events),
               'L1': _reader('L1', events, error=failure)}
    counts, recalculated = dict.fromkeys(readers, 1), dict.fromkeys(readers, False)
    options = dict(increment=8, timeout=30, interval=3,
                   min_distance=5, max_distance=20,
                   stage_event=lambda *a, **k: None, data_start=100)
    assert live.prepare_live_data_jax(
        readers, tuple(readers), counts, recalculated, **options) is None
    assert live.advance_live_data_jax(
        readers['H1'], 'H1', counts, recalculated, **options) is True
    events.append(('filter', 'H1'))
    with pytest.raises(OSError) as caught:
        live.advance_live_data_jax(
            readers['L1'], 'L1', counts, recalculated, **options)
    assert caught.value is failure
    assert counts == {'H1': 0, 'L1': 1}
    assert events.index(('filter', 'H1')) < events.index(('advance', 'L1', 8, 30))
