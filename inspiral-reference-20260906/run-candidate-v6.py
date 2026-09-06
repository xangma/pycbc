#!/usr/bin/env python3
"""Run candidate unit tests and the serial large-IFFT qualification."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
output = root / 'candidate-v6.status.json'
assert not output.exists()
cfg = json.loads((root / 'config.json').read_text())
source = root / 'source-v6-candidate1'
receipt = root / 'source-v6-candidate1.json'
env = dict(os.environ, **cfg['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
commands = [
    [sys.executable, str(root / 'run-unit-checks-v6.py'), 'v6-candidate1'],
    ['taskset', '-c', str(cfg['core']), sys.executable, str(root / 'qualify-large-ifft-v6.py'),
     '--source', str(source), '--output', str(root / 'large-ifft-v6-candidate1.json'), '--threads', '1'],
]
inputs = [Path(__file__), receipt, root / 'config.json', root / 'run-unit-checks-v6.py',
          root / 'qualify-large-ifft-v6.py']
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
record = dict(state='running', pid=os.getpid(), host=os.uname().nodename, cwd=str(root),
              command=[sys.executable, *sys.argv], source_commit=json.loads(receipt.read_text())['commit'],
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              input_sha256={str(path): digest(path) for path in inputs}, steps=[])
started = time.monotonic()

def save():
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

try:
    # The unit runner requires no running search status. Save this launcher
    # record only after that runner has completed its preflight and execution.
    for index, command in enumerate(commands):
        log_path = root / f'candidate-v6-step{index + 1}.log'
        with log_path.open('x') as log:
            step = dict(command=command, log=str(log_path), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            step['pid'] = process.pid
            code = process.wait()
        step.update(returncode=code, log_sha256=digest(log_path))
        record['steps'].append(step)
        if code:
            raise RuntimeError(f'Candidate step {index + 1} failed with status {code}')
        if index:
            save()
    record.update(state='complete', returncode=0)
except Exception as error:
    record.update(state='failed', returncode=1, error=f'{type(error).__name__}: {error}')
finally:
    record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  wall_seconds=time.monotonic() - started,
                  input_sha256_after={str(path): digest(path) for path in inputs})
    if record['input_sha256_after'] != record['input_sha256']:
        record.update(state='invalid-input-mutation', returncode=1)
    save()
print(json.dumps(record), flush=True)
sys.exit(record['returncode'])
