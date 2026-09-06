#!/usr/bin/env python3
"""Run a frozen list of case arguments sequentially, stopping on any failure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

plan = Path(sys.argv[1]).resolve()
root = plan.parent
cases = json.loads(plan.read_text())
status_path = plan.with_suffix('.status.json')
assert not status_path.exists()
record = dict(state='running', pid=os.getpid(), plan=str(plan),
              plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              completed=[], current=None)


def save():
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(status_path)


save()
for case in cases:
    command = [sys.executable, '-u', str(root / 'run-case-profiling.py'),
               '--config', str(root / 'config.json'), '--source', str(root / 'source-v5'),
               '--bank', str(root / 'inputs/bank-compressed-1e5.hdf'), *case]
    record['current'] = command
    save()
    child = subprocess.Popen(command, cwd=root)
    record['child_pid'] = child.pid
    save()
    code = child.wait()
    if code:
        record.update(state='failed', returncode=code,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        sys.exit(code)
    record['completed'].append(case)
    record['current'] = None
    save()
record.update(state='complete', returncode=0,
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
assert hashlib.sha256(plan.read_bytes()).hexdigest() == record['plan_sha256']
save()
