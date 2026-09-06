"""Qualify the integrated CPU candidate in an isolated source checkout."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
PUB = ROOT / 'publication'
OLD = ROOT / 'cpu-candidate-v2'
SOURCE = ROOT / 'cpu-integration-v6-checkout'
OUT = ROOT / 'cpu-integration-v6-tests'
HEAD = 'b776a97a0477ca922b829adcf14475b0d3791363'
OLD_HEAD = '665fa5a0f41946c0c2873d8a79a16c8b6975a090'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    paths = git(root, 'ls-files').splitlines()
    return dict(head=git(root, 'rev-parse', 'HEAD'), tree=git(root, 'rev-parse', 'HEAD^{tree}'),
        status=git(root, 'status', '--porcelain', '--untracked-files=all'),
        files={p: sha(root / p) for p in paths if (root / p).is_file()},
        native_binaries={str(p.relative_to(root)): sha(p) for p in sorted((root / 'pycbc').rglob('*.so'))})


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(record):
    (OUT / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


assert not SOURCE.exists() and not OUT.exists()
assert git(OLD, 'rev-parse', 'HEAD') == OLD_HEAD
assert not git(OLD, 'status', '--porcelain', '--untracked-files=all')
stack = json.loads((PUB / 'candidate-stack.json').read_bytes())
part = next(p for p in stack['candidates'] if p['number'] == 17)
assert part['head'] == HEAD
OUT.mkdir()
record = dict(state='preparing', passed=False, host=os.uname().nodename, pid=os.getpid(),
    cwd=str(ROOT), command=[sys.executable, *sys.argv], started_utc=utc(),
    expected_next_check_seconds=30, stop_command=f'kill -TERM {os.getpid()}',
    input_sha256={str(p): sha(p) for p in (Path(__file__), PUB / 'candidate-stack.json', PUB / 'publication.bundle')})
save(record)
subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(OLD), str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', str(PUB / 'publication.bundle'),
    'refs/heads/' + part['candidate_branch'] + ':refs/remotes/publication/cpu'], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', HEAD], check=True)
native_inputs = {}
for relative in git(SOURCE, 'ls-files').splitlines():
    p = Path(relative)
    if p.suffix.lower() in {'.c', '.cc', '.cpp', '.h', '.hpp', '.pyx', '.pxd', '.cu', '.f', '.f90'} or relative in {'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in'}:
        assert (SOURCE / p).read_bytes() == (OLD / p).read_bytes(), relative
        native_inputs[relative] = sha(OLD / p)
for p in sorted((OLD / 'pycbc').rglob('*.so')):
    assert p.is_file() and not p.is_symlink()
    target = SOURCE / p.relative_to(OLD)
    assert not target.exists()
    shutil.copy2(p, target)
    assert sha(target) == sha(p)
before = snapshot(SOURCE)
assert before['head'] == HEAD and before['tree'] == part['tree'] and not before['status']
assert len(before['native_binaries']) == 11
tests = [
    'test/test_torch_cpu_fft_tuning.py', 'test/test_torch_large_ifft.py',
    'test/test_torch_fft_cpu_native.py', 'test/test_torch_fft_writes.py',
    'test/test_torch_fft_cuda_workspace.py', 'test/test_torch_decompress_cpu.py',
    'test/test_decompress.py', 'test/test_matchedfilter.py', 'test/test_chisq.py',
    'test/test_torch_filter_pipeline.py', 'test/test_torch_matchedfilter_cpu_optimization.py',
    'test/test_cpu_batch_peaks.py', 'test/test_torch_cpu_native_peaks.py',
    'test/test_torch_chisq_cpu_optimization.py', 'test/test_torch_chisq_sparse_dispatch.py',
    'test/test_chisq_torch.py', 'test/test_torch_performance_artifacts.py',
]
env = {k: v for k, v in os.environ.items() if not k.startswith('PYCBC_') and k not in {'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONSTARTUP'}}
env.update(PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
command = ['taskset', '-c', '8', sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests]
record.update(state='running', source_before=before, binary_source=str(OLD), binary_source_head=OLD_HEAD,
    native_inputs=native_inputs, test_command=command, test_cwd=str(SOURCE),
    environment={k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
save(record)
with (OUT / 'tests.log').open('x') as log:
    process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record.update(test_pid=process.pid, test_stop_command=f'kill -TERM -- -{process.pid}')
    save(record)
    code = process.wait()
after = snapshot(SOURCE)
record.update(state='complete', passed=code == 0 and after == before,
    returncode=code, source_after=after, finished_utc=utc(), log_sha256=sha(OUT / 'tests.log'))
assert all(sha(Path(p)) == value for p, value in record['input_sha256'].items())
save(record)
print(json.dumps({'state': record['state'], 'passed': record['passed'], 'returncode': code}))
sys.exit(0 if record['passed'] else 1)
