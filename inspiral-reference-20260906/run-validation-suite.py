#!/usr/bin/env python3
"""Run scientific diagnostics sequentially after benchmark timing has stopped."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
status = root / 'scientific-validation.status.json'
assert not status.exists()
for path in root.glob('*.status.json'):
    assert json.loads(path.read_text()).get('state') != 'running', path
cfg = json.loads((root / 'config.json').read_text())
env = dict(os.environ, **cfg['environment'])
env.update(PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1')
common = ['--bank-metadata', str(root / 'inputs/bank-metadata.json'),
          '--compression-receipt', str(root / 'compression.json'),
          '--source', str(root / 'source')]
qualifications = [root / 'runs' / name / 'qualification.json' for name in
                  ('qual-v2-cpu-l256', 'qual-v2-cpu-l512', 'qual-v2-cpu-l1024',
                   'qual-cpu-l256-s96', 'qual-cpu-l512-s96', 'qual-cpu-l1024-s96')]
commands = [
    ('waveform-validation', [sys.executable, '-u', str(root / 'validate-waveforms.py'),
      *common, '--output', str(root / 'waveform-validation.json'),
      *[arg for path in qualifications for arg in ('--qualification', str(path))]]),
    ('boundary-injections', [sys.executable, '-u', str(root / 'check-boundary-injections.py'),
      *common, '--output', str(root / 'boundary-injections.json'),
      '--qualification', str(qualifications[2])]),
]
record = dict(state='running', pid=os.getpid(), host=os.uname().nodename,
              cwd=str(root), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              commands=commands, completed=[])
record['environment'] = {key: env[key] for key in
                         [*cfg['environment'], 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE']}
inputs = [Path(__file__), root / 'validate-waveforms.py', root / 'check-boundary-injections.py']
record['runner_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}


def save():
    temp = status.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(status)


save()
for name, command in commands:
    command = ['taskset', '-c', str(cfg['core']), *command]
    with (root / f'{name}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        record.update(current=command, child_pid=child.pid)
        save()
        code = child.wait()
    record['completed'].append(dict(name=name, returncode=code))
    save()
    if code:
        break
record['runner_sha256_after'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
record.update(state='complete' if code == 0 else 'failed', returncode=code,
              current=None, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if record['runner_sha256_after'] != record['runner_sha256']:
    record.update(state='invalid-input-mutation', returncode=1)
save()
sys.exit(record['returncode'])
