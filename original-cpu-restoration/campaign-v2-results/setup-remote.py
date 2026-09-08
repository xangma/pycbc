"""Prepare isolated source checkouts using verified unchanged native binaries."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FROZEN = Path('/home/xangma/pycbc-torch-baseline-final-20260908')
CONFIG = json.loads((ROOT / 'config.json').read_text())
REFERENCES = {'original': FROZEN / 'original', 'proposed': FROZEN / 'proposed'}
STATE = dict(state='running', pid=os.getpid(), started=time.time(), completed=[])
LOCK = Path('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def run(*args):
    return subprocess.check_output(list(args), text=True).strip()


try:
    save(ROOT / 'setup-status.json', STATE)
    with LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run('git', 'clone', '--shared', '--no-checkout', str(FROZEN / 'original'), str(ROOT / 'repo'))
        run('git', '-C', str(ROOT / 'repo'), 'fetch', str(ROOT / 'sources.bundle'),
            'refs/heads/codex/original-cpu-restoration-v2-20260908-pr15')
        for name, head in CONFIG['source_commits'].items():
            source, reference = ROOT / name, REFERENCES[name]
            assert not run('git', '-C', str(reference), 'status', '--porcelain')
            run('git', '-C', str(ROOT / 'repo'), 'worktree', 'add', '--detach', str(source), head)
            reference_head = run('git', '-C', str(reference), 'rev-parse', 'HEAD')
            native_changes = run('git', '-C', str(source), 'diff', '--name-only', reference_head, head,
                '--', '*.pyx', '*.pxd', '*.pxi', '*.c', '*.cpp', '*.h', '*.cu', 'setup.py', 'pycbc/lib')
            assert not native_changes, native_changes
            build = json.loads((FROZEN / ('original-build.json' if name == 'original' else 'proposed-build.json')).read_text())
            native = {str(p.relative_to(reference)): sha(p) for p in (reference / 'pycbc').rglob('*.so')}
            assert native == build['native'] and len(native) >= 11
            for relative in native:
                shutil.copy2(reference / relative, source / relative)
                assert sha(source / relative) == native[relative]
            shutil.copy2(reference / 'pycbc/version.py', source / 'pycbc/version.py')
            assert not run('git', '-C', str(source), 'status', '--porcelain')
            save(ROOT / (name + '-build.json'), dict(commit=head, source=str(source),
                native=native, version_sha256=sha(source / 'pycbc/version.py'),
                method='Reused frozen binaries after exact native-source and binary-hash verification',
                reference_source=str(reference), reference_commit=reference_head,
                reference_build_sha256=sha(FROZEN / ('original-build.json' if name == 'original' else 'proposed-build.json'))))
            STATE['completed'].append(name)
            save(ROOT / 'setup-status.json', STATE)
        for path, expected in CONFIG['input_pins'].items():
            assert sha(path) == expected
        save(ROOT / 'dependencies.json', dict(python=sys.version, executable=sys.executable,
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}))
        STATE.update(state='complete', finished=time.time(), bundle_sha256=sha(ROOT / 'sources.bundle'))
except BaseException as error:
    STATE.update(state='failed', error=repr(error), finished=time.time())
    raise
finally:
    save(ROOT / 'setup-status.json', STATE)
