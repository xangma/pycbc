#!/usr/bin/env python3
"""Check scheme selection/runtime against the frozen executable source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
source = root / 'source-v5'
out = root / 'unit-tests-v5.json'
assert not out.exists()
for path in root.glob('*.status.json'):
    assert json.loads(path.read_text()).get('state') != 'running', path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_info():
    return {key: subprocess.check_output(
        ['git', '-C', str(source), *args], text=True).strip()
        for key, args in [('commit', ['rev-parse', 'HEAD']),
                          ('status', ['status', '--porcelain'])]}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


before = source_info()
assert before == dict(commit=json.loads((root / 'source-v5.json').read_text())['commit'], status='')
tests = [
    'test/test_scheme_runtime.py', 'test/test_scheme_selection.py',
    'test/test_matchedfilter.py', 'test/test_chisq.py',
    'test/test_psd.py', 'test/test_strain.py',
    'test/test_sigmasq_series_precision.py', 'test/test_chisq_precision.py',
    'test/test_strain_psd_precision.py', 'test/test_torch_chisq_cpu_optimization.py',
    'test/test_torch_chisq_sparse_dispatch.py', 'test/test_torch_filter_pipeline.py',
    'test/test_torch_psd_pipeline.py', 'test/test_torch_psd_protocol.py',
    'test/test_torch_versioned_data_psd.py', 'test/test_torch_matchedfilter_cpu_optimization.py',
]

cfg = json.loads((root / 'config.json').read_text())
env = dict(os.environ, **cfg['environment'], PYTHONPATH=str(source),
           PYTHONDONTWRITEBYTECODE='1')
command = ['taskset', '-c', str(cfg['core']), sys.executable, '-m', 'pytest',
           '-q', '-p', 'no:cacheprovider', *tests]
inputs = [Path(__file__), root / 'config.json', source / 'bin/pycbc_inspiral',
          source / 'pycbc/scheme.py', root / 'source-v5.json',
          *[source / p for p in ['pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq.py', 'pycbc/vetoes/chisq_torch.py',
              'pycbc/psd/__init__.py', 'pycbc/strain/strain.py', *tests]]]
record = dict(state='running', passed=False, command=command, cwd=str(source),
              host=os.uname().nodename, pid=os.getpid(), started_utc=utc(),
              source_info=before, environment={key: env[key] for key in
              [*cfg['environment'], 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE']},
              input_sha256={str(p): digest(p) for p in inputs})
out.write_text(json.dumps(record, indent=2) + '\n')
started = time.monotonic()
with (root / 'unit-tests-v5.log').open('x') as log:
    code = subprocess.call(command, cwd=source, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
record.update(returncode=code, finished_utc=utc(),
              wall_seconds=time.monotonic() - started,
              source_after=source_info(),
              input_sha256_after={str(p): digest(p) for p in inputs},
              log_sha256=digest(root / 'unit-tests-v5.log'))
unchanged = (record['source_after'] == before and
             record['input_sha256_after'] == record['input_sha256'])
record.update(state='complete' if unchanged else 'invalid-input-mutation',
              passed=code == 0 and unchanged)
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
sys.exit(0 if record['passed'] else 1)
