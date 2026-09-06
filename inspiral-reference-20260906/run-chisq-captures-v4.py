#!/usr/bin/env python3
"""Sequential observational captures; no source mutation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root = Path(__file__).resolve().parent
status_path = root / 'chisq-captures-v4.status.json'
assert not status_path.exists()
script = root / 'capture-chisq-inputs-v4.py'
sha = hashlib.sha256(script.read_bytes()).hexdigest()
record = dict(state='running', pid=os.getpid(), completed=[], script_sha256=sha)
def save():
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(status_path)
save()
try:
    for backend in ['cpu', 'torch-cuda']:
        case = f'qual-precision-{backend}-l512'
        output = f'chisq-capture-v4-{backend}'
        command = [sys.executable, '-B', '-u', str(script), '--case', case,
                   '--template-hash=-6051532311927781382', '--output-name', output]
        record.update(current=backend, command=command)
        save()
        with (root / f'{output}.log').open('x') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (backend, result.returncode)
        captured = json.loads((root / output / 'status.json').read_text())
        assert captured['state'] == 'complete'
        record['completed'].append(backend)
        save()
    assert hashlib.sha256(script.read_bytes()).hexdigest() == sha
    record.update(state='complete')
except BaseException as exc:
    record.update(state='failed', error=repr(exc))
    raise
finally:
    record.update(finished_unix=time.time())
    save()
