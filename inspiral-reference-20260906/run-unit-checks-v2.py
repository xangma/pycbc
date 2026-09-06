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
source = root / 'source-v2'
out = root / 'unit-tests.json'
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
assert before == dict(commit='968bcd558117262af0d603710b054174659adb51', status='')
tests = ['test/test_scheme_runtime.py', 'test/test_scheme_selection.py']
cfg = json.loads((root / 'config.json').read_text())
env = dict(os.environ, **cfg['environment'], PYTHONPATH=str(source),
           PYTHONDONTWRITEBYTECODE='1')
command = ['taskset', '-c', str(cfg['core']), sys.executable, '-m', 'pytest',
           '-q', '-p', 'no:cacheprovider', *tests]
inputs = [Path(__file__), root / 'config.json', source / 'bin/pycbc_inspiral',
          source / 'pycbc/scheme.py', *[source / p for p in tests]]
record = dict(state='running', passed=False, command=command, cwd=str(source),
              host=os.uname().nodename, pid=os.getpid(), started_utc=utc(),
              source_info=before, environment={key: env[key] for key in
              [*cfg['environment'], 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE']},
              input_sha256={str(p): digest(p) for p in inputs})
out.write_text(json.dumps(record, indent=2) + '\n')
started = time.monotonic()
with (root / 'unit-tests.log').open('x') as log:
    code = subprocess.call(command, cwd=source, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
record.update(returncode=code, finished_utc=utc(),
              wall_seconds=time.monotonic() - started,
              source_after=source_info(),
              input_sha256_after={str(p): digest(p) for p in inputs},
              log_sha256=digest(root / 'unit-tests.log'))
unchanged = (record['source_after'] == before and
             record['input_sha256_after'] == record['input_sha256'])
record.update(state='complete' if unchanged else 'invalid-input-mutation',
              passed=code == 0 and unchanged)
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
sys.exit(0 if record['passed'] else 1)
