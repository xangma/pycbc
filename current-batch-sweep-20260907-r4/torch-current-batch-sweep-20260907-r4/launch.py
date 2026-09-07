#!/usr/bin/env python3
"""Verify staged inputs and launch exactly one campaign in a fresh root."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
assert root == Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r4')
assert not (root / 'launch-receipt.json').exists()
assert not (root / 'stages').exists()
source = Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3/source')
for name, expected in json.loads((root / 'stage-manifest.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '9578a710479b924e882857c4dffab6ed372a634b'
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
for item in json.loads((root / 'native-provenance.json').read_text())['extensions']:
    assert hashlib.sha256((source / item['relative_path']).read_bytes()).hexdigest() == item['sha256']
(root / 'source').symlink_to(source, target_is_directory=True)
command = ['/home/xangma/pycbc-torch-split-20260905/venv/bin/python', str(root / 'batch-campaign.py'), '--shared-host', '--snr-policy-v2']
with (root / 'launch.log').open('x') as log:
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
receipt = {'host': 'len', 'cwd': str(root), 'pid': process.pid, 'pgid': os.getpgid(process.pid), 'command': command, 'log': str(root / 'launch.log'), 'started': time.time(), 'expected_next_check': 'after smoke qualification, about two minutes', 'stop_command': f'ssh len "kill -TERM {process.pid}"', 'source_symlink_target': str(source), 'source_revision': revision, 'native_hashes_verified': 11}
(root / 'launch-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
