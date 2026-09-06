#!/usr/bin/env python3
"""Measure original upstream against final Torch with the frozen workload."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906')
FINAL = OLD / 'source-v6'
BASE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
HEAD = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
ORIGINAL = ROOT / 'original'

def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + '\n')

def main():
    assert not ORIGINAL.exists()
    assert git(FINAL, 'rev-parse', 'HEAD') == HEAD
    assert not git(FINAL, 'status', '--porcelain')
    subprocess.run(['git', 'clone', '--no-hardlinks', str(FINAL), str(ORIGINAL)], check=True)
    subprocess.run(['git', '-C', str(ORIGINAL), 'checkout', '--detach', BASE], check=True)
    native_diff = git(FINAL, 'diff', '--name-only', BASE, HEAD, '--', '*.pyx', '*.pxd', '*.cpp', '*.c', '*.h')
    assert native_diff == ''
    # Compare all native-build definitions, excluding the Python-only extras.
    def build_ast(source):
        tree = ast.parse((source / 'setup.py').read_text())
        begin = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cythonext' for t in n.targets))
        return [ast.dump(n, include_attributes=False) for n in tree.body[begin:-1]]
    assert build_ast(ORIGINAL) == build_ast(FINAL)
    receipt = json.loads((OLD / 'source-v6.json').read_text())
    native = {}
    for name, expected in receipt['native_modules_sha256'].items():
        assert digest(FINAL / name) == expected
        shutil.copy2(FINAL / name, ORIGINAL / name)
        native[name] = digest(ORIGINAL / name)
    assert len(native) == 11
    env = os.environ.copy()
    env.update(json.loads((OLD / 'config.json').read_text())['environment'])
    env.update(PYTHONPATH=str(ORIGINAL), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, 'setup.py', '--version'], cwd=ORIGINAL, env=env, check=True)
    assert not git(ORIGINAL, 'status', '--porcelain')
    probe = subprocess.check_output([sys.executable, '-c', 'import json,pycbc,pycbc.version; from pycbc.fft import mkl; print(json.dumps(dict(pycbc=pycbc.__file__,mkl=mkl.__file__,version=pycbc.version.git_hash)))'], cwd=ROOT, env=env, text=True)
    save('source.json', dict(original=BASE, final=HEAD, original_import_probe=json.loads(probe), native_sources_unchanged=True, native_build_definitions_unchanged=True, reused_native_sha256=native, original_status=git(ORIGINAL,'status','--porcelain'), final_status=git(FINAL,'status','--porcelain')))
    for name in ('config.json', 'run-case.py', 'compare-triggers.py'):
        shutil.copy2(OLD / name, ROOT / name)
    plan = []
    routes = [('original-cpu', ORIGINAL, 'cpu:1'), ('torch-cpu', FINAL, 'torch:cpu:1'), ('torch-cuda', FINAL, 'torch:cuda:0:1')]
    for repeat in range(3):
        for name, source, scheme in routes[repeat:] + routes[:repeat]:
            plan.append(dict(case=f'{name}-r{repeat+1}', mode='timing', source=str(source), scheme=scheme))
    for name, source, scheme in routes:
        for mode in ('cprofile', 'perf'):
            plan.append(dict(case=f'{name}-{mode}', mode=mode, source=str(source), scheme=scheme))
    save('plan.json', plan)
    state = dict(pid=os.getpid(), state='running', completed=[], started=time.time())
    save('status.json', state)
    for item in plan:
        state['current'] = item['case']; save('status.json', state)
        command = [sys.executable, str(ROOT / 'run-case.py'), '--config', str(ROOT / 'config.json'), '--case', item['case'], '--mode', item['mode'], '--scheme', item['scheme'], '--segment-length', '512', '--source', item['source'], '--bank', str(OLD / 'inputs/bank-compressed-1e5.hdf')]
        with (ROOT / (item['case'] + '.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed'].append(item['case']); save('status.json', state)
    # Preserve a numerical failure as evidence rather than changing tolerances.
    base = ROOT / 'runs/original-cpu-r1/triggers.hdf'
    candidates = [ROOT / f'runs/{name}-r1/triggers.hdf' for name in ('torch-cpu','torch-cuda')]
    with (ROOT / 'original-trigger-comparison.json').open('w') as out:
        result = subprocess.run([sys.executable, str(ROOT / 'compare-triggers.py'), str(base), *map(str,candidates)], stdout=out, stderr=subprocess.PIPE, text=True)
    (ROOT / 'comparison.stderr.log').write_text(result.stderr)
    state.update(state='complete', comparison_returncode=result.returncode, finished=time.time())
    save('status.json', state)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('failure.json', dict(error=repr(exc), time=time.time()))
        raise
