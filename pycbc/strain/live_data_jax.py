# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Batch independent Live PSD work without moving detector admission decisions."""

from pycbc.strain.strain import StrainBuffer


_NATIVE_METHODS = {name: getattr(StrainBuffer, name) for name in (
    'advance', 'recalculate_psd', 'check_psd_dist')}


def _native_readers(readers, ifos):
    """Keep overridden and shared reader state on the original serial path."""
    selected = tuple(readers.get(ifo) for ifo in ifos)
    if len(selected) < 2:
        return None
    if len({id(reader) for reader in selected}) != len(selected):
        return None
    for reader in selected:
        if type(reader) is not StrainBuffer:
            return None
        for name, original in _NATIVE_METHODS.items():
            if getattr(getattr(reader, name), '__func__', None) is not original:
                return None
    for name in ('strain', 'raw_buffer'):
        storage = tuple(getattr(reader, name, None) for reader in selected)
        if any(value is None for value in storage):
            return None
        if len({id(value) for value in storage}) != len(storage):
            return None
    return selected


class _LiveDataAdmission:
    """Resolve each detector only when the original Live loop reaches it."""

    def __init__(self, steps, counts, recalculated, interval, bounds):
        self._steps = steps
        self._counts = counts
        self._recalculated = recalculated
        self._interval = interval
        self._bounds = bounds

    def status(self, ifo):
        step = self._steps[ifo]
        if 'resolved' in step:
            return step['resolved']
        if step['error'] is not None:
            raise step['error']
        status = step['status']
        if step['pending'] is not None:
            status = step['pending'].result()
            self._recalculated[ifo] = True
            self._counts[ifo] = self._interval - 1
        elif not status:
            self._counts[ifo] = 0
        else:
            self._counts[ifo] -= 1
        status &= step['reader'].check_psd_dist(*self._bounds)
        step['resolved'] = status
        return status


def advance_live_data_jax(reader, ifo, counts, recalculated, *, increment,
                          timeout, interval, min_distance, max_distance,
                          stage_event, data_start):
    """Admit a single or custom reader in the original detector order."""
    stage_event('frame_read', 'start', ifo=ifo, data_start=data_start)
    status = reader.advance(increment, timeout=timeout)
    stage_event('frame_read', 'end', ifo=ifo, data_start=data_start,
                status=bool(status))
    if status and counts[ifo] == 0:
        status = reader.recalculate_psd()
        recalculated[ifo] = True
        counts[ifo] = interval - 1
    elif not status:
        counts[ifo] = 0
    else:
        counts[ifo] -= 1
    status &= reader.check_psd_dist(min_distance, max_distance)
    return status


def prepare_live_data_jax(readers, ifos, counts, recalculated, *, increment,
                          timeout, interval, min_distance, max_distance,
                          stage_event, data_start):
    """Advance native readers and enqueue their refreshes before one collection.

    Later detector failures are retained until its original admission turn,
    so earlier detector decisions and filtering keep their exception priority.
    No candidate PSD is installed here. Initial PSD-None refreshes inside
    ``advance`` retain the ordinary checked path. Return None for custom or
    aliased readers, allowing the caller to use its original serial loop.
    """
    ifos = tuple(ifos)
    selected = _native_readers(readers, ifos)
    if selected is None:
        return None
    from .live_psd_jax import prepare_psd_updates_jax

    steps = {}
    updates = []
    for ifo, reader in zip(ifos, selected):
        step = {'reader': reader, 'status': False, 'error': None,
                'pending': None}
        steps[ifo] = step
        stage_event('frame_read', 'start', ifo=ifo, data_start=data_start)
        try:
            status = reader.advance(increment, timeout=timeout)
            stage_event('frame_read', 'end', ifo=ifo, data_start=data_start,
                        status=bool(status))
            step['status'] = status
            if status and counts[ifo] == 0:
                updates.append((ifo, reader))
        except Exception as error:
            step['error'] = error
            # Preserve the first advance failure; later readers never advance.
            break

    if updates:
        stage_event('psd_refresh_batch', 'start', data_start=data_start,
                    detectors=[ifo for ifo, _ in updates])
    pending = prepare_psd_updates_jax(tuple(reader for _, reader in updates))
    if updates:
        stage_event('psd_refresh_batch', 'end', data_start=data_start,
                    detectors=[ifo for ifo, _ in updates])
    for (ifo, _), handle in zip(updates, pending):
        steps[ifo]['pending'] = handle
    return _LiveDataAdmission(steps, counts, recalculated, interval,
                              (min_distance, max_distance))
