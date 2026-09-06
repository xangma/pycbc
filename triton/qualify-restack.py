"""Validate the exact prepared publication prefixes after timed work ends."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/xangma/pycbc-taylorf2-triton-20260906')
REPO = ROOT / 'validation'
PYTHON = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
QLTY = '/home/xangma/pycbc-torch-finish-20260904/qlty/qlty-x86_64-unknown-linux-gnu/qlty'
LOGS = ROOT / 'validation-logs'
PARTS = [p for p in json.loads((ROOT / 'candidate-stack.json').read_text()) if 'old_sha' in p]
ENV = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           PYCBC_TEST_SCHEME='torch:cuda', PYCBC_TAYLORF2_TRITON='1',
           TRITON_CACHE_DIR=str(ROOT / 'validation-triton-cache'),
           PATH=str(Path(PYTHON).parent) + ':' + os.environ['PATH'])
RESULTS = []


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def execute(label, args, stderr=None):
    started = time.time()
    print('START ' + label, flush=True)
    with (LOGS / (label + '.log')).open('w') as output:
        if stderr is None:
            code = subprocess.run(args, cwd=REPO, env=ENV, stdout=output,
                                  stderr=subprocess.STDOUT).returncode
        else:
            with (LOGS / stderr).open('w') as errors:
                code = subprocess.run(args, cwd=REPO, env=ENV, stdout=output,
                                      stderr=errors).returncode
    result = dict(label=label, sha=git('rev-parse', 'HEAD'), command=args,
                  returncode=code, seconds=time.time() - started)
    xml = LOGS / (label + '.xml')
    if xml.exists():
        root = ET.parse(xml).getroot()
        suites = list(root) if root.tag == 'testsuites' else [root]
        counts = {k: sum(int(s.get(k, '0')) for s in suites)
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
        result['counts'] = counts
    RESULTS.append(result)
    (ROOT / 'restack-validation.json').write_text(json.dumps(RESULTS, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return code


def tests(label, paths):
    return execute(label, [PYTHON, '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider',
                           *paths, '--junitxml=' + str(LOGS / (label + '.xml'))])


while True:
    manifest = ROOT / 'full-v3-clean/manifest.json'
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state.get('finished_utc'):
            if state.get('status') != 'ok':
                raise SystemExit('Full benchmark did not qualify; no restack validation started')
            break
    time.sleep(5)
LOGS.mkdir(exist_ok=True)
assert not REPO.exists(), 'Use a fresh validation checkout'
subprocess.run(['git', 'clone', '--shared', str(ROOT / 'source'), str(REPO)], check=True)
for source in (ROOT / 'source/pycbc').rglob('*.so'):
    destination = REPO / source.relative_to(ROOT / 'source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
git('fetch', str(ROOT / 'publication.bundle'),
    '+refs/heads/codex/torch-triton-restack-20260906/*:refs/remotes/publication/*')
for part in PARTS:
    key = part['key']
    git('checkout', '--detach', part['sha'])
    if execute(key + '-build', [PYTHON, 'setup.py', 'build_ext', '--inplace', '-j', '8']):
        raise SystemExit(1)
    paths = part['tests'] if key != 'format-fft' else [
        'test/test_torch_batched_fft.py', 'test/test_torch_fft_cpu_native.py',
        'test/test_torch_fft_writes.py', 'test/test_hardware.py']
    if tests(key, paths):
        raise SystemExit(1)
    execute(key + '-qlty', [QLTY, 'check', '--no-upgrade-check', '--no-fix', '--sarif',
                            '--no-progress', '--upstream', part['parent']], key + '-qlty-stderr.log')
    saved = LOGS / (key + '-qlty-generated-links')
    saved.mkdir()
    for name in ('logs', 'out', 'plugin_cachedir', 'results'):
        path = REPO / '.qlty' / name
        if path.is_symlink():
            path.rename(saved / name)
    if git('status', '--porcelain'):
        raise RuntimeError('Unexpected validation checkout changes: ' + git('status', '--porcelain'))
    native = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (REPO / 'pycbc').rglob('*.so')}
    (LOGS / (key + '-native-sha256.json')).write_text(json.dumps(native, indent=2) + '\n')
main = next(p for p in PARTS if p['key'] == 'pr11')
git('checkout', '--detach', main['sha'])
if execute('main-final-build', [PYTHON, 'setup.py', 'build_ext', '--inplace', '-j', '8']):
    raise SystemExit(1)
tests('main-final-taylorf2', ['test/waveform/test_taylorf2_batch.py',
                             'test/waveform/test_taylorf2_torch.py'])
execute('main-f401-ci', [PYTHON, '/home/xangma/pycbc-torch-split-20260905/validate-f401.py'])
execute('main-extension-scope', [PYTHON, '-c',
    'from pycbc.filter import matchedfilter_cpu as m; from pycbc.vetoes import chisq_cpu as c; '
    'assert not hasattr(m, "_batch_abs_arg_max_complex64"); '
    'assert not hasattr(c, "point_chisq_code_single_double"); print("No optional kernels in main stack")'])
status = dict(finished=True, results=len(RESULTS),
              nonzero=[r['label'] for r in RESULTS if r['returncode']])
(ROOT / 'restack-validation-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
