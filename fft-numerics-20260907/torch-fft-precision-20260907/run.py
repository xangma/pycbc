#!/usr/bin/env python3
"""Bounded serial diagnostics, with a shared process group for stopping."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
campaign = Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3')
status = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'state': 'running', 'started': time.time(), 'completed': []}


def save():
    temporary = root / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(root / 'status.json')


try:
    for batch in (1, 8):
        for route in ('torch_cpu', 'torch_cuda'):
            name = f'b{batch}-{route}'
            command = ['taskset', '-c', '8', sys.executable, str(root / 'diagnose.py'),
                       '--campaign-root', str(campaign), '--output-root', str(root / name),
                       '--route', route, '--batch', str(batch)]
            status.update(current=name, command=command, log=str(root / f'{name}.log'))
            save()
            with (root / f'{name}.log').open('xb') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=360)
            status['completed'].append({'name': name, 'exit_code': result.returncode})
            if result.returncode:
                raise RuntimeError(f'{name} diagnostic failed with {result.returncode}')
    status['state'] = 'complete'
except BaseException as exc:
    status.update(state='failed', error=repr(exc))
    raise
finally:
    status['finished'] = time.time()
    save()
