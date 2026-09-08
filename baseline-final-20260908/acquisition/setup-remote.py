"""Prepare and rebuild both pinned source checkouts in an isolated directory."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = '/home/xangma/pycbc-torch-maintainer-benchmark-20260906/original'
LOCK = '/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock'
config = json.loads((ROOT / 'config.json').read_text())
state = dict(state='running', pid=os.getpid(), pgid=os.getpgrp(), started=time.time(), completed=[])


def save():
    (ROOT / 'setup-status.json').write_text(json.dumps(state, indent=2) + '\n')


def run(command, **kwargs):
    print(json.dumps(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


try:
    save()
    with open(LOCK) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(['git', 'clone', '--shared', '--no-checkout', OLD, str(ROOT / 'repo')])
        run(['git', '-C', str(ROOT / 'repo'), 'fetch', str(ROOT / 'proposal.bundle'),
             'refs/heads/codex/torch-doc-audit-20260908'])
        for name, commit in config['source_commits'].items():
            source = ROOT / name
            run(['git', '-C', str(ROOT / 'repo'), 'worktree', 'add', '--detach', str(source), commit])
            state['current'] = name
            save()
            env = os.environ.copy()
            env.update(config['environment'], PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source))
            with (ROOT / (name + '-build.log')).open('x') as log:
                run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=source,
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
            assert not status, status
            native = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((source / 'pycbc').rglob('*.so'))}
            assert len(native) >= 11, native
            (ROOT / (name + '-build.json')).write_text(json.dumps(dict(
                commit=commit, source=str(source), command=[sys.executable, 'setup.py', 'build_ext', '--inplace'],
                native=native, version_sha256=hashlib.sha256((source / 'pycbc/version.py').read_bytes()).hexdigest(),
                status=status), indent=2) + '\n')
            state['completed'].append(name)
            save()
        packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                    if d.metadata['Name']}
        (ROOT / 'dependencies.json').write_text(json.dumps(dict(python=sys.version,
            executable=sys.executable, packages=packages), indent=2) + '\n')
        for path, expected in config['input_pins'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        state.update(state='complete', finished=time.time())
except BaseException as error:
    state.update(state='failed', error=repr(error), finished=time.time())
    raise
finally:
    save()
