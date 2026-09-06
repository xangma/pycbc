"""Run recorded correctness checks only after the benchmark queue finishes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/xangma/pycbc-torch-batch1024-20260906')
PY = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
OUT = ROOT / 'correctness'
OUT.mkdir(exist_ok=True)
REVISIONS = {'main': '4885b64560e9f39b740e85b6a976898869dd360e',
             'fft': 'd840198592a0ca128f4f89d90987a7469ec37c8f',
             'cpu': 'd544420232428225c214a4be84fbe1262a6d307b'}
STATE = {'pid': os.getpid(), 'waiting_for_queue': 4085887, 'runs': []}
ACTIVE = None

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save():
    temporary = OUT / 'status.json.tmp'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(OUT / 'status.json')

def stop(signum, frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
        try:
            ACTIVE.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
    STATE['interrupted_signal'] = signum
    save()
    raise SystemExit(128 + signum)

def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
save()
while True:
    queue = json.loads((ROOT / 'queue-status.json').read_text())
    if queue.get('finished'):
        break
    os.kill(4085887, 0)
    time.sleep(10)
STATE['waiting_for_queue'] = None
STATE['benchmark_queue_passed'] = queue.get('passed')
test = ROOT / 'test_torch_large_batches.py'
assert hashlib.sha256(test.read_bytes()).hexdigest() == 'c558857122e7bed462ad2eb0c72d7c109058ac583f2246c72d5c590435766dee'
STATE['test_sha256'] = hashlib.sha256(test.read_bytes()).hexdigest()
for name, revision in REVISIONS.items():
    root = ROOT / name
    assert git(root, 'rev-parse', 'HEAD') == revision
    assert not git(root, 'status', '--porcelain', '--untracked-files=no')
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    environment = {key: value for key, value in environment.items()
                   if not key.startswith('PYCBC_TORCH_') and key != 'PYCBC_TAYLORF2_TRITON'}
    check = ['taskset', '-c', '8-11', PY, '-B', '-c',
             'import pycbc,torch; print(pycbc.__file__); print(torch.__version__); '
             'assert pycbc.__file__.startswith(' + repr(str(root)) + '); '
             'assert torch.cuda.is_available()']
    imported = subprocess.check_output(check, cwd=root, env=environment, text=True)
    command = ['taskset', '-c', '8-11', PY, '-B', '-m', 'pytest', '-q', '-rs',
               '-p', 'no:cacheprovider', '--junitxml=' + str(OUT / (name + '.xml')),
               str(test), 'test/test_torch_fft_cpu_native.py',
               'test/test_torch_cpu_native_batch.py', 'test/test_torch_cuda_native_batch.py',
               'test/test_torch_cuda_native_peaks.py', 'test/test_live_batch_torch_fft_integration.py',
               'test/test_live_batch_torch_peaks.py']
    if name == 'fft':
        command.extend(['test/test_fft_batched_backends.py',
                        'test/test_fft_cli_wisdom.py', 'test/test_fftw_wisdom_cache.py'])
    record = {'name': name, 'revision': revision, 'command': command, 'cwd': str(root),
              'imports': imported, 'started_utc': now(), 'log': str(OUT / (name + '.log')),
              'environment': {key: environment[key] for key in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                                                               'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                               'OPENBLAS_NUM_THREADS')}}
    with open(record['log'], 'x') as log:
        ACTIVE = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = ACTIVE.pid
        STATE['runs'].append(record)
        STATE['active'] = name
        save()
        print(json.dumps(record), flush=True)
        record['returncode'] = ACTIVE.wait()
    record['finished_utc'] = now()
    record['head_after'] = git(root, 'rev-parse', 'HEAD')
    record['tracked_changes_after'] = git(root, 'status', '--porcelain', '--untracked-files=no')
    record['log_sha256'] = hashlib.sha256(Path(record['log']).read_bytes()).hexdigest()
    assert record['head_after'] == revision and not record['tracked_changes_after']
    save()
    print(json.dumps(record), flush=True)
STATE['active'] = None
STATE['finished'] = True
STATE['finished_utc'] = now()
STATE['passed'] = all(row['returncode'] == 0 for row in STATE['runs'])
save()
