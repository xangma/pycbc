#!/usr/bin/env python3
"""Measure original upstream against final Torch with the frozen workload."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906')
FINAL = OLD / 'source-v6'
BASE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
HEAD = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
ORIGINAL = ROOT / 'original'

def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + '\n')

def main():
    plan = []
    routes = [('original-cpu', ORIGINAL, 'cpu:1'), ('torch-cpu', FINAL, 'torch:cpu:1'), ('torch-cuda', FINAL, 'torch:cuda:0')]
    for repeat in range(3):
        for name, source, scheme in routes[repeat:] + routes[:repeat]:
            plan.append(dict(case=f'{name}-r{repeat+1}', mode='timing', source=str(source), scheme=scheme))
    for name, source, scheme in routes:
        for mode in ('cprofile', 'perf'):
            plan.append(dict(case=f'{name}-{mode}', mode=mode, source=str(source), scheme=scheme))
    save('corrected-plan.json', plan)
    state = dict(pid=os.getpid(), state='running', completed=[], started=time.time())
    save('status.json', state)
    for item in plan:
        receipt = ROOT / 'runs' / item['case'] / 'receipt.json'
        if receipt.exists():
            d = json.loads(receipt.read_text())
            if d['state'] == 'complete':
                state['completed'].append(item['case']); save('status.json', state)
                continue
            shutil.move(str(receipt.parent), str(ROOT / ('failed-' + item['case'])))
        state['current'] = item['case']; save('status.json', state)
        command = [sys.executable, str(ROOT / 'run-case.py'), '--config', str(ROOT / 'config.json'), '--case', item['case'], '--mode', item['mode'], '--scheme', item['scheme'], '--segment-length', '512', '--source', item['source'], '--bank', str(OLD / 'inputs/bank-compressed-1e5.hdf')]
        with (ROOT / (item['case'] + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed'].append(item['case']); save('status.json', state)
    # Preserve a numerical failure as evidence rather than changing tolerances.
    base = ROOT / 'runs/original-cpu-r1/triggers.hdf'
    candidates = [ROOT / f'runs/{name}-r1/triggers.hdf' for name in ('torch-cpu','torch-cuda')]
    with (ROOT / 'original-trigger-comparison.json').open('w') as out:
        result = subprocess.run([sys.executable, str(ROOT / 'compare-triggers.py'), str(base), *map(str,candidates)], stdout=out, stderr=subprocess.PIPE, text=True)
    (ROOT / 'comparison.stderr.log').write_text(result.stderr)
    state.update(state='complete', comparison_returncode=result.returncode, finished=time.time())
    save('status.json', state)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('failure.json', dict(error=repr(exc), time=time.time()))
        raise
