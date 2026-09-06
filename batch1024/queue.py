import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/xangma/pycbc-torch-batch1024-20260906')
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
MAIN = '4885b64560e9f39b740e85b6a976898869dd360e'
CPU = 'd544420232428225c214a4be84fbe1262a6d307b'
ACTIVE = None
STATE = {'pid': os.getpid(), 'cwd': str(ROOT / 'main'), 'command': [PYTHON, '-B', str(ROOT / 'queue.py')], 'runs': [], 'waiting_for': 3924196}

def save():
    temporary = ROOT / 'queue-status.json.tmp'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(ROOT / 'queue-status.json')

def stop(signum, _frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.send_signal(signal.SIGTERM)
        try:
            ACTIVE.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
    STATE['interrupted_signal'] = signum
    save()
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
save()
while True:
    triton = json.loads((ROOT / 'triton/manifest.json').read_text())
    if 'finished_utc' in triton:
        break
    try:
        os.kill(3924196, 0)
    except ProcessLookupError:
        raise RuntimeError('Triton runner stopped without a completed manifest')
    time.sleep(5)
assert len(triton['jobs']) == 72
assert all(j.get('status') == 'ok' and j.get('returncode') == 0 for j in triton['jobs'])
STATE['waiting_for'] = None
common = ['--main-root', str(ROOT / 'main'), '--cpu-root', str(ROOT / 'cpu'), '--expected-main', MAIN, '--expected-cpu', CPU, '--python', PYTHON]
commands = [
    ('waveform', ['taskset', '-c', '8-11', PYTHON, '-B', str(ROOT / 'wave-support/waveform/harness/run.py'), '--root', str(ROOT / 'main'), '--python', PYTHON, '--expected-sha', MAIN, '--batches', '1', '8', '32', '128', '512', '1024', '--out', str(ROOT / 'waveform')]),
    ('live', [PYTHON, '-B', str(ROOT / 'live-support/run-live.py'), *common, '--output', str(ROOT / 'live')]),
    ('probes', [PYTHON, '-B', str(ROOT / 'live-support/run-probes.py'), *common, '--output', str(ROOT / 'probes')]),
]
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
environment.pop('PYTHONPATH', None)
for name, command in commands:
    record = {'name': name, 'command': command, 'log': str(ROOT / 'logs' / (name + '.log')), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(record['log'], 'x') as log:
        ACTIVE = subprocess.Popen(command, cwd=ROOT / 'main', env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = ACTIVE.pid
        STATE['runs'].append(record)
        STATE['active'] = name
        save()
        print(json.dumps(record), flush=True)
        record['returncode'] = ACTIVE.wait()
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps(record), flush=True)
STATE['active'] = None
STATE['finished'] = True
STATE['passed'] = all(record['returncode'] == 0 for record in STATE['runs'])
save()
raise SystemExit(0 if STATE['passed'] else 1)
