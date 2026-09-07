#!/usr/bin/env python3
"""Profile the template loop of an unchanged pycbc_inspiral executable.

The interval includes decompression, normalizations, filtering, vetoes and event
handling. It begins at the first FilterBank lookup and ends just before the
executable saves its performance metadata, after final event consolidation.
Instrumented wall times are never used as benchmark timings.
"""
import argparse
import cProfile
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--pstats', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('An executable command is required')
    if args.receipt.exists() or (args.pstats and args.pstats.exists()):
        parser.error('Refusing existing profile outputs')
    from pycbc.waveform.bank import FilterBank
    from pycbc.events.eventmgr import EventManager
    from pycbc.scheme import mgr

    profile = cProfile.Profile() if args.pstats else None
    filtering_device = None
    record = dict(state='running', clock='CLOCK_MONOTONIC', command=command,
                  interval='First FilterBank lookup through final event consolidation',
                  indices=[], start_monotonic_ns=None, stop_monotonic_ns=None)
    getitem = FilterBank.__getitem__
    save_performance = EventManager.save_performance

    def synchronize():
        torch = sys.modules.get('torch')
        device = filtering_device or getattr(mgr.state, 'device', None)
        if torch is not None and getattr(device, 'type', None) == 'cuda':
            torch.cuda.synchronize(device)

    def observed_getitem(bank, index):
        nonlocal filtering_device
        if record['start_monotonic_ns'] is None:
            filtering_device = getattr(mgr.state, 'device', None)
            record['filtering_device'] = str(filtering_device)
            synchronize()
            record['bank_templates'] = len(bank)
            record['start_monotonic_ns'] = time.monotonic_ns()
            if profile:
                profile.enable()
        record['indices'].append(int(index))
        return getitem(bank, index)

    def observed_performance(manager, ncores, nfilters, ntemplates, *rest, **kwargs):
        synchronize()
        if profile:
            profile.disable()
        record['stop_monotonic_ns'] = time.monotonic_ns()
        record.update(host_cores=ncores, segments_per_template=nfilters,
                      completed_templates=ntemplates)
        return save_performance(manager, ncores, nfilters, ntemplates, *rest, **kwargs)

    FilterBank.__getitem__ = observed_getitem
    EventManager.save_performance = observed_performance
    previous_argv = sys.argv
    code = 0
    try:
        sys.argv = command
        try:
            runpy.run_path(command[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        assert record['indices'] == list(range(record['bank_templates']))
        assert record['completed_templates'] == record['bank_templates']
        assert record['start_monotonic_ns'] < record['stop_monotonic_ns']
        record['filtering_window_seconds'] = (
            record['stop_monotonic_ns'] - record['start_monotonic_ns']) / 1e9
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=repr(exc))
        code = 1
        raise
    finally:
        if profile:
            profile.disable()
            profile.dump_stats(str(args.pstats))
        FilterBank.__getitem__ = getitem
        EventManager.save_performance = save_performance
        sys.argv = previous_argv
        record['returncode'] = code
        args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
