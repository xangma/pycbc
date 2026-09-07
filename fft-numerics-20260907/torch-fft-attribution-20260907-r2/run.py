#!/usr/bin/env python3
"""One bounded replay process; children share the supervisor's process group."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
command = ['taskset', '-c', '8', sys.executable, str(root / 'replay.py')]
status = dict(state='running', pid=os.getpid(), pgid=os.getpgrp(), command=command,
              started=time.time(), log=str(root / 'replay.log'))


def save():
    temporary = root / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(root / 'status.json')


save()
try:
    with (root / 'replay.log').open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=360)
    status.update(state='complete' if result.returncode == 0 else 'failed', exit_code=result.returncode)
except BaseException as exc:
    status.update(state='failed', error=repr(exc))
    raise
finally:
    status['finished'] = time.time()
    save()
