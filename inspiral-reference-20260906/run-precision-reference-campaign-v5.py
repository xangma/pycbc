#!/usr/bin/env python3
"""Validate the corrected CPU reference before retuning its segment length."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
status = root / 'precision5-reference-campaign.status.json'
assert not status.exists()
for path in root.glob('*.status.json'):
    assert json.loads(path.read_text()).get('state') != 'running', path
cfg = json.loads((root / 'config.json').read_text())
fixed_env = dict(cfg['environment'], PYTHONPATH=str(root / 'source-v5'), PYTHONDONTWRITEBYTECODE='1')
env = dict(os.environ, **fixed_env)
common = ['--bank-metadata', str(root / 'inputs/bank-metadata.json'),
          '--compression-receipt', str(root / 'compression-1e5.json'),
          '--source', str(root / 'source-v5')]
qualifications = [root / 'runs' / f'qual-precision5-cpu-l{length}' / 'qualification.json'
                  for length in (256, 512, 1024)]
commands = [
    ('waveform-validation-precision5', ['taskset', '-c', str(cfg['core']), sys.executable,
      '-u', str(root / 'validate-waveforms.py'), *common,
      '--output', str(root / 'waveform-validation-precision5.json'),
      *[arg for path in qualifications for arg in ('--qualification', str(path))]]),
    ('boundary-injections-precision5', ['taskset', '-c', str(cfg['core']), sys.executable,
      '-u', str(root / 'check-boundary-injections.py'), *common,
      '--output', str(root / 'boundary-injections-precision5.json'),
      '--qualification', str(qualifications[2])]),
    ('precision5-reference-timings', [sys.executable, '-u', str(root / 'run-series-v5.py'),
                         str(root / 'precision5-reference-timings-plan.json')]),
]
inputs = [Path(__file__), *[root / name for name in
          ('source-v5.json', 'run-series-v5.py', 'run-case.py',
           'qualify-inspiral.py', 'validate-waveforms.py', 'check-boundary-injections.py',
           'precision5-reference-qualifications-plan.json', 'precision5-reference-timings-plan.json', 'config.json')]]
record = dict(state='running', pid=os.getpid(), host=os.uname().nodename,
              cwd=str(root), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              commands=commands, completed=[], environment=fixed_env,
              input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})


def save():
    temp = status.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(status)


save()
for name, command in commands:
    with (root / f'{name}-stage.log').open('x') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        record.update(current_stage=name, current=command, child_pid=child.pid)
        save()
        code = child.wait()
    record['completed'].append(dict(name=name, returncode=code))
    save()
    if code:
        break
record['input_sha256_after'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
record.update(state='complete' if code == 0 else 'failed', returncode=code,
              current=None, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if record['input_sha256_after'] != record['input_sha256']:
    record.update(state='invalid-input-mutation', returncode=1)
save()
sys.exit(record['returncode'])
