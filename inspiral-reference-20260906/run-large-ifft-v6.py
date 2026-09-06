#!/usr/bin/env python3
"""Record a serial large-IFFT matrix on a clean, assembled source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
version, tag = sys.argv[1:]
assert (version, tag) in (('v6-candidate1', 'v6-candidate1b'), ('v6', 'v6'))
source = root / ('source-' + version)
output = root / ('large-ifft-' + tag + '.json')
status = root / ('large-ifft-' + tag + '.status.json')
log = root / ('large-ifft-' + tag + '.log')
assert not any(p.exists() for p in (output, status, log))
for p in root.glob('*.status.json'):
    assert json.loads(p.read_text()).get('state') != 'running', p
cfg = json.loads((root / 'config.json').read_text())
receipt = root / ('source-' + version + '.json')
expected = json.loads(receipt.read_text())['commit']
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == expected
assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip() == ''
env = dict(os.environ, **cfg['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
command = ['taskset', '-c', str(cfg['core']), sys.executable, str(root / 'qualify-large-ifft-v6b.py'),
           '--source', str(source), '--output', str(output), '--threads', '1']
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
paths = [Path(__file__), receipt, root / 'config.json', root / 'qualify-large-ifft-v6b.py']
record = dict(state='running', pid=os.getpid(), host=os.uname().nodename, cwd=str(root),
              command=command, source_commit=expected, log=str(log),
              input_sha256={str(path): digest(path) for path in paths},
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
status.write_text(json.dumps(record, indent=2) + '\n')
started = time.monotonic()
with log.open('x') as stream:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record['child_pid'] = process.pid
    status.write_text(json.dumps(record, indent=2) + '\n')
    code = process.wait()
record.update(state='complete' if code == 0 else 'failed', returncode=code,
              wall_seconds=time.monotonic() - started, output_sha256=digest(output), log_sha256=digest(log),
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              input_sha256_after={str(path): digest(path) for path in paths})
if record['input_sha256_after'] != record['input_sha256']:
    record.update(state='invalid-input-mutation', returncode=1)
status.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
sys.exit(record['returncode'])
