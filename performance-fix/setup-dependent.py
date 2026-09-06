"""Prepare isolated dependent sources after the paired benchmark completes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

R = Path('/home/xangma/pycbc-torch-performance-fix-20260906')
OLD = Path('/home/xangma/pycbc-torch-batch1024-20260906')

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

while True:
    status = json.loads((R/'comparison-status.json').read_text())
    if status['state'] == 'complete':
        break
    if status['state'] != 'running':
        raise RuntimeError('Benchmark did not complete: ' + status['state'])
    time.sleep(30)

manifest = json.loads((R/'dependent-sources.json').read_text())
assert digest(R/'dependent-sources.bundle') == manifest['bundles'][0]['sha256']
assert git(R/'baseline', 'rev-parse', 'HEAD') == 'dfd42bf76766cadca0eecf609a1eaeac73534676'
assert not git(R/'baseline', 'status', '--porcelain')
plan = [
    ('fft-candidate', 'd6407d32742a57e4f026c461a26ef7b3929c5849',
     '4adcca5ff8c9ab3aa6769ff0cced6f7da19c2fa7', 'fft',
     'd840198592a0ca128f4f89d90987a7469ec37c8f', 'dependent-sources.bundle'),
    ('cpu-candidate', '1514327669fc7be125523b991c847868c3a2a17e',
     '0bbf15de82c467649cb88fb6378aaf74d829cfca', 'cpu',
     'd544420232428225c214a4be84fbe1262a6d307b', 'dependent-sources.bundle'),
    ('cpu-baseline', 'bd53914be6d2e4324cc867d52b3842b77cc6729a',
     None, 'cpu', 'd544420232428225c214a4be84fbe1262a6d307b',
     'optional-cpu-baseline.bundle'),
]
assert all(not (R/name).exists() for name, *_ in plan)
assert not (R/'dependent-environment.json').exists()
receipt = {}
for name, sha, tree, old_name, old_sha, bundle in plan:
    root, old = R/name, OLD/old_name
    assert git(old, 'rev-parse', 'HEAD') == old_sha
    assert not git(old, 'status', '--porcelain')
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout',
                    str(R/'baseline'), str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'fetch', str(R/bundle),
                    '+refs/heads/*:refs/remotes/bundle/*'], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--detach', sha], check=True)
    actual_tree = git(root, 'rev-parse', 'HEAD^{tree}')
    assert tree is None or actual_tree == tree
    inputs = {}
    for relative in git(root, 'ls-files').splitlines():
        path = Path(relative)
        if path.suffix.lower() in {'.c', '.cc', '.cpp', '.h', '.hpp', '.pyx', '.pxd', '.cu', '.f', '.f90'} or relative in {'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in'}:
            assert (root/path).read_bytes() == (old/path).read_bytes(), relative
            assert git(root, 'ls-files', '-s', relative).split()[0] == git(old, 'ls-files', '-s', relative).split()[0], relative
            inputs[relative] = digest(root/path)
    binaries = {}
    for source in sorted((old/'pycbc').rglob('*.so')):
        assert source.is_file() and not source.is_symlink(), str(source)
        relative = source.relative_to(old)
        destination = root/relative
        assert destination.parent.is_dir() and not destination.exists(), str(destination)
        shutil.copy2(source, destination)
        binaries[str(relative)] = digest(source)
        assert digest(destination) == binaries[str(relative)]
    assert len(binaries) == 11
    assert not git(root, 'status', '--porcelain')
    receipt[name] = dict(root=str(root), sha=sha, tree=actual_tree, status='',
                         binary_source=str(old), binary_source_sha=old_sha,
                         native_inputs=inputs, native_binaries=binaries,
                         bundle_sha256=digest(R/bundle))
    save(R/'dependent-environment.partial.json', receipt)
save(R/'dependent-environment.json', receipt)
print(json.dumps({name: r['sha'] for name, r in receipt.items()}), flush=True)
