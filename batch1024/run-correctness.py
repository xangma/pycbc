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
PREPARED = json.loads((ROOT / 'correctness-preparation.json').read_text())
REVISIONS = {row['name']: row['revision'] for row in PREPARED}
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
assert hashlib.sha256(test.read_bytes()).hexdigest() == '7cfe32a8e20125b0e21dd1abcb831f821fec4f974f845fe23b4c564e97219fb6'
STATE['test_sha256'] = hashlib.sha256(test.read_bytes()).hexdigest()
for name, revision in REVISIONS.items():
    prepared = next(row for row in PREPARED if row['name'] == name)
    root = Path(prepared['cwd'])
    assert root == ROOT / 'correctness-sources' / name
    assert git(root, 'rev-parse', 'HEAD^') == prepared['base_revision']
    assert git(root, 'diff', '--name-only', prepared['base_revision'], revision) == 'pycbc/filter/matchedfilter.py'
    for path, checksum in prepared['file_sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == checksum
    for path, checksum in {**prepared['native_sha256'], **prepared['generated_metadata_sha256']}.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == checksum
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
               'test/test_live_batch_torch_peaks.py', 'test/test_chisq_torch.py',
               'test/test_torch_filter_pipeline.py']
    if name == 'fft':
        command.extend(['test/test_fft_batched_backends.py',
                        'test/test_fft_cli_wisdom.py', 'test/test_fftw_wisdom_cache.py'])
    record = {**prepared, 'command': command, 'cwd': str(root),
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
    for path, checksum in {**prepared['file_sha256'], **prepared['native_sha256'], **prepared['generated_metadata_sha256']}.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == checksum
    save()
    print(json.dumps(record), flush=True)
STATE['active'] = None
STATE['finished'] = True
STATE['finished_utc'] = now()
STATE['passed'] = all(row['returncode'] == 0 for row in STATE['runs'])
save()
