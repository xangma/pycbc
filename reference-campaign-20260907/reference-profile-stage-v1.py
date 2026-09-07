#!/usr/bin/env python3
"""Select the measured normal reference and profile it before any Torch run."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/home/xangma/pycbc-torch-maintainer-benchmark-20260906/original')
state = dict(state='waiting-for-sweep', pid=os.getpid(), pgid=os.getpgrp(),
             started=time.time(), completed=[])


def save(name, data):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    save('reference-profile-status.json', state)
    while True:
        sweep = json.loads((ROOT / 'status.json').read_text())
        if sweep['state'] != 'running':
            assert sweep['state'] == 'complete', sweep
            break
        time.sleep(10)
    import h5py
    rows = []
    for length in (256, 512, 1024):
        for pad in (96, 112):
            samples = []
            for repeat in range(1, 4):
                folder = ROOT / f'runs/original-l{length}-p{pad}-r{repeat}'
                receipt = json.loads((folder / 'receipt.json').read_text())
                assert receipt['state'] == 'complete'
                assert receipt['input_sha256'] == receipt['input_sha256_after']
                assert receipt['source_info']['status'] == receipt['source_status_after'] == ''
                with h5py.File(folder / 'triggers.hdf') as f:
                    starts, stops = f['H1/search/start_time'][:], f['H1/search/end_time'][:]
                    assert sum(stops-starts) == 1904
                    assert all(starts[1:] >= stops[:-1])
                    samples.append(dict(case=receipt['case'], wall=receipt['elapsed_wall_seconds'],
                                        internal_seconds=float(f['H1/search/run_time'][0]),
                                        setup_fraction=float(f['H1/search/setup_time_fraction'][0]),
                                        segments=len(starts), triggers=len(f['H1/snr'])))
            walls = [s['wall'] for s in samples]
            rows.append(dict(length=length, start_pad=pad, end_pad=16, samples=samples,
                             median=statistics.median(walls), minimum=min(walls), maximum=max(walls)))
    fastest = min(r['median'] for r in rows)
    eligible = [r for r in rows if r['median'] <= fastest*1.03]
    selected = sorted(eligible, key=lambda r: (-r['start_pad'], r['length'], r['median']))[0]
    save('tuning-decision.json', dict(rows=rows, fastest_median=fastest,
                                    selection_rule='Within 3%: larger start pad, then smaller FFT.',
                                    selected=selected, original_only=True))
    state['state'] = 'running'
    plans = [('qual-selected-original', 'qualify', 'run-case.py'),
             ('profile-original-cprofile', 'cprofile', 'run-case.py'),
             ('profile-original-filter-cprofile', 'filter-cprofile', 'run-profile-case.py'),
             ('profile-original-filter-perf', 'filter-perf', 'run-profile-case.py')]
    save('reference-profile-plan.json', plans)
    for case, mode, runner in plans:
        command = [sys.executable, str(ROOT / runner), '--config', str(ROOT / 'config.json'),
                   '--case', case, '--mode', mode, '--scheme', 'cpu:1', '--source', str(ORIGINAL),
                   '--bank', str(ROOT / 'inputs/bank-compressed.hdf'),
                   '--segment-length', str(selected['length']), '--start-pad', str(selected['start_pad'])]
        state.update(current=case, command=command)
        save('reference-profile-status.json', state)
        with (ROOT / (case + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=900, check=True)
        state['completed'].append(case)
        save('reference-profile-status.json', state)
    window = json.loads((ROOT / 'runs/profile-original-filter-perf/filtering-window.json').read_text())
    assert window['state'] == 'complete'
    folder = ROOT / 'runs/profile-original-filter-perf'
    for name, extra in [('perf-report-full.txt', []), ('perf-report-filtering.txt',
            ['--time', f"{window['start_monotonic_ns']/1e9:.9f},{window['stop_monotonic_ns']/1e9:.9f}"])]:
        command = ['perf', 'report', '--stdio', '--no-children', '--call-graph', 'none',
                   '--percent-limit', '0', '-i', str(folder / 'perf.data'), *extra]
        with (folder / name).open('x') as output, (folder / (name + '.stderr')).open('x') as error:
            subprocess.run(command, stdout=output, stderr=error, check=True)
    state.update(state='complete', finished=time.time())
    save('reference-profile-status.json', state)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=time.time())
        save('reference-profile-status.json', state)
        raise
