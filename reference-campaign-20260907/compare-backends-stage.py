#!/usr/bin/env python3
"""Run the frozen comparison after the original reference has been reviewed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/home/xangma/pycbc-torch-maintainer-benchmark-20260906/original')
FINAL = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906/source-v6')
ROUTES = [('original-cpu', ORIGINAL, 'cpu:1'),
          ('torch-cpu', FINAL, 'torch:cpu:1'),
          ('torch-cuda', FINAL, 'torch:cuda:0')]
state = dict(state='running', pid=os.getpid(), pgid=os.getpgrp(),
             started=time.time(), completed=[])


def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert json.loads((ROOT / 'reference-profile-status.json').read_text())['state'] == 'complete'
    accepted = json.loads((ROOT / 'reference-accepted.json').read_text())
    assert accepted['accepted'] is True
    selected = json.loads((ROOT / 'tuning-decision.json').read_text())['selected']
    plan = []
    for name, source, scheme in [('corrected-cpu', FINAL, 'cpu:1'), *ROUTES[1:]]:
        plan.append(dict(case='qual-selected-' + name, mode='qualify',
                         source=str(source), scheme=scheme, runner='run-case.py'))
    for repeat in range(3):
        for name, source, scheme in ROUTES[repeat:] + ROUTES[:repeat]:
            plan.append(dict(case=f'matched-{name}-r{repeat+1}', mode='timing',
                             source=str(source), scheme=scheme, runner='run-case.py'))
    for name, source, scheme in ROUTES[1:]:
        for mode in ('cprofile', 'filter-cprofile', 'filter-perf'):
            plan.append(dict(case=f'profile-{name}-{mode}', mode=mode,
                             source=str(source), scheme=scheme,
                             runner='run-profile-case.py' if mode.startswith('filter-') else 'run-case.py'))
    save('comparison-plan.json', plan)
    for item in plan:
        command = [sys.executable, str(ROOT / item['runner']), '--config', str(ROOT / 'config.json'),
                   '--case', item['case'], '--mode', item['mode'], '--scheme', item['scheme'],
                   '--source', item['source'], '--bank', str(ROOT / 'inputs/bank-compressed.hdf'),
                   '--segment-length', str(selected['length']), '--start-pad', str(selected['start_pad'])]
        state.update(current=item['case'], command=command)
        save('comparison-status.json', state)
        with (ROOT / (item['case'] + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        state['completed'].append(item['case'])
        save('comparison-status.json', state)
        if item['mode'] == 'filter-perf':
            folder = ROOT / 'runs' / item['case']
            window = json.loads((folder / 'filtering-window.json').read_text())
            assert window['state'] == 'complete'
            for name, extra in [('perf-report-full.txt', []), ('perf-report-filtering.txt',
                    ['--time', f"{window['start_monotonic_ns']/1e9:.9f},{window['stop_monotonic_ns']/1e9:.9f}"])]:
                command = ['perf', 'report', '--stdio', '--no-children', '--call-graph', 'none',
                           '--percent-limit', '0', '-i', str(folder / 'perf.data'), *extra]
                with (folder / name).open('x') as output, (folder / (name + '.stderr')).open('x') as error:
                    subprocess.run(command, stdout=output, stderr=error, check=True)
    comparisons = [
        ('original-trigger-comparison', ROOT / 'runs/matched-original-cpu-r1/triggers.hdf',
         [ROOT / f'runs/matched-{name}-r{repeat}/triggers.hdf'
          for name in ('torch-cpu', 'torch-cuda') for repeat in range(1, 4)]),
        ('corrected-trigger-comparison', ROOT / 'runs/qual-selected-corrected-cpu/triggers.hdf',
         [ROOT / f'runs/matched-{name}-r{repeat}/triggers.hdf'
          for name in ('torch-cpu', 'torch-cuda') for repeat in range(1, 4)]),
    ]
    # Compare profiling outputs against each backend's own unprofiled output.
    for name, _, _ in ROUTES:
        prefix = 'profile-original' if name == 'original-cpu' else 'profile-' + name
        comparisons.append((name + '-profile-parity', ROOT / f'runs/matched-{name}-r1/triggers.hdf',
                            [ROOT / f'runs/{prefix}-{mode}/triggers.hdf'
                             for mode in ('cprofile', 'filter-cprofile', 'filter-perf')]))
    results = {}
    for name, baseline, candidates in comparisons:
        command = [sys.executable, str(ROOT / 'compare-triggers.py'), str(baseline), *map(str, candidates)]
        with (ROOT / (name + '.json')).open('x') as output, (ROOT / (name + '.stderr')).open('x') as error:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=error, timeout=120)
        results[name] = result.returncode
    state.update(state='complete', comparison_returncodes=results, finished=time.time())
    save('comparison-status.json', state)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=time.time())
        save('comparison-status.json', state)
        raise
