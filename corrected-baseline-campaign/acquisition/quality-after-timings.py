"""Check rebuilt main with existing Qlty after the benchmark releases its lock."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

C = Path(__file__).resolve().parent
Q = C.parent / 'corrected-baseline-quality'
LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')
QLTY = '/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty'
HEAD = 'f582b6fd250d0b82612492979e01e645d5c07afc'
PARENT = 'f2eb803de6f2d34f2227e69e35d460bd481fff05'
CHILD = None


def status(state, **details):
    (Q / 'status.json').write_text(json.dumps(dict(state=state, pid=os.getpid(),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details), indent=2) + '\n')


def terminate(*_):
    if CHILD is not None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    status('terminated')
    sys.exit(143)


def main():
    global CHILD
    Q.mkdir(exist_ok=False)
    status('waiting_for_benchmark_lock')
    with LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        timing = json.loads((C / 'timing-status.json').read_text())
        assert timing['state'] == 'complete', timing
        subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(C / 'repo'), str(Q / 'source')], check=True)
        subprocess.run(['git', 'checkout', '--detach', HEAD], cwd=Q / 'source', check=True)
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=Q / 'source').strip()
        env = dict(os.environ)
        env['PATH'] = '/home/xangma/pycbc-torch-split-20260905/venv/bin:' + env['PATH']
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = ['taskset', '-c', '9', QLTY, 'check', '--no-upgrade-check', '--no-fix',
                   '--sarif', '--no-progress', '--jobs', '1', '--upstream', PARENT]
        status('running', head=HEAD, upstream=PARENT, cwd=str(Q / 'source'), command=command)
        with (Q / 'qlty.sarif').open('w') as out, (Q / 'qlty.log').open('w') as err:
            CHILD = subprocess.Popen(command, cwd=Q / 'source', env=env, stdout=out, stderr=err, start_new_session=True)
            code = CHILD.wait()
        CHILD = None
        clean = not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=Q / 'source').strip()
        assert clean
        status('complete', returncode=code, head=HEAD, upstream=PARENT, tracked_clean=clean,
               cwd=str(Q / 'source'), command=command)
        return code


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
