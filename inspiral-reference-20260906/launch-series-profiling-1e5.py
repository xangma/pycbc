#!/usr/bin/env python3
"""Detach one named benchmark series, preserving a launch receipt."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
plan = (root / sys.argv[1]).resolve()
assert plan.parent == root and plan.is_file()
assert not plan.with_suffix('.status.json').exists()
for path in root.glob('*.status.json'):
    previous = json.loads(path.read_text())
    if previous.get('state') == 'running':
        raise RuntimeError(f'Existing running series: {path}')
name = plan.stem.removesuffix('-plan')
log = root / f'{name}-launch.log'
receipt = root / f'{name}-launch.json'
assert not log.exists() and not receipt.exists()
command = [sys.executable, '-u', str(root / 'run-series-profiling-1e5.py'), str(plan)]
with log.open('x') as stream:
    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
record = dict(pid=child.pid, host=os.uname().nodename, cwd=str(root), command=command,
              log=str(log), stop=f'kill -TERM -- -{child.pid}',
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [plan, root / 'run-series-profiling-1e5.py',
                                       root / 'run-case-profiling.py', Path(__file__)]})
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
