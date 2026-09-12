#!/usr/bin/env python3
"""Record target-host environment and final/baseline lint diagnostics."""
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path('/home/xangma/pycbc-remediation-20260912')
REPO = ROOT / 'qualified'
BASELINE = ROOT / 'baseline'
LOGS = ROOT / 'logs'


def run(command, cwd=REPO):
    result = subprocess.run(command, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'command': command, 'cwd': str(cwd), 'returncode': result.returncode,
            'output': result.stdout}


changed = run(['git', 'diff', '--name-only',
               '9ff3a7ec5b5643fe7b0a3b94d082799c05d31775', 'HEAD'])['output'].splitlines()
changed = [p for p in changed if p.endswith('.py') or p in
           ('bin/pycbc_live', 'bin/pycbc_inspiral')]
lint = {}
for label, cwd in (('final', REPO), ('baseline', BASELINE)):
    paths = [p for p in changed if (cwd / p).is_file()]
    result = run([sys.executable, '-m', 'flake8', *paths], cwd)
    (LOGS / f'flake8-{label}-changed.log').write_text(result['output'])
    lint[f'{label}_changed_full'] = {k: v for k, v in result.items() if k != 'output'}
    lint[f'{label}_changed_full']['diagnostic_lines'] = len(result['output'].splitlines())
    result = run([sys.executable, '-m', 'flake8', '--select', 'F401', *paths], cwd)
    (LOGS / f'flake8-{label}-changed-f401.log').write_text(result['output'])
    lint[f'{label}_changed_f401'] = result


def normalized(path):
    return collections.Counter(re.sub(r':\d+:\d+:', ':', line)
                               for line in path.read_text().splitlines())


old = normalized(LOGS / 'flake8-f401-baseline.log')
new = normalized(LOGS / 'flake8-f401-final.log')
lint['repository_f401'] = {'baseline_count': sum(old.values()),
                          'final_count': sum(new.values()),
                          'new_diagnostics': list((new - old).elements()),
                          'removed_diagnostics': list((old - new).elements())}
lint['qlty'] = {'path': shutil.which('qlty'), 'executed': False}
if lint['qlty']['path']:
    lint['qlty']['version'] = run(['qlty', '--version'])
(LOGS / 'lint-comparison.json').write_text(json.dumps(lint, indent=2) + '\n')

import torch

packages = {}
for name in ('torch', 'numpy', 'scipy', 'pycbc', 'torchwave', 'lalsuite',
             'h5py', 'pyfftw', 'pytest', 'flake8', 'mpi4py'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
environment = {
    'host': platform.node(), 'platform': platform.platform(),
    'python': sys.version, 'executable': sys.executable,
    'packages': packages, 'torch_cuda_runtime': torch.version.cuda,
    'torch_config': torch.__config__.show(),
    'note': 'Post-acquisition identity snapshot; per-run thread/device settings are in commands and receipts. GPU snapshot is not an in-run utilization trace.',
    'source': run(['git', 'rev-parse', 'HEAD']),
    'tracked_status': run(['git', 'status', '--porcelain', '--untracked-files=no']),
    'cpu': run(['lscpu']),
    'gpu': run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,power.draw,clocks.gr,temperature.gpu', '--format=csv']),
    'filesystem': run(['df', '-T', str(ROOT)]),
    'environment': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPATH')},
}
(LOGS / 'final-environment.json').write_text(json.dumps(environment, indent=2) + '\n')
print(json.dumps({'lint': lint, 'packages': packages, 'host': environment['host']}, indent=2))
