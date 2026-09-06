"""Prepare isolated, committed sources for the disabled-veto regression fix."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/xangma/pycbc-torch-batch1024-20260906')
OUT = ROOT / 'correctness'
DEST = ROOT / 'correctness-sources'
PATH = 'pycbc/filter/matchedfilter.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

old = json.loads((OUT / 'status.json').read_text())
assert old['finished'] and old['passed'] is False
assert all(row['returncode'] == 1 for row in old['runs'])
failed = OUT / 'disabled-veto-failed'
failed.mkdir()
for path in list(OUT.iterdir()):
    if path.is_file():
        path.rename(failed / path.name)
shutil.copy2(ROOT / 'test_torch_large_batches.py', failed / 'test_torch_large_batches.py')
shutil.copy2(ROOT / 'run-correctness.py', failed / 'run-correctness.py')
DEST.mkdir()
prepared = []
for source in json.loads((ROOT / 'preparation.json').read_text()):
    name, base = source['name'], source['sha']
    root, original = DEST / name, ROOT / name
    assert git(original, 'rev-parse', 'HEAD') == base
    assert not git(original, 'status', '--porcelain')
    git(original, 'worktree', 'add', '--detach', str(root), base)
    for relative, checksum in source['binaries'].items():
        assert sha(original / relative) == checksum
        shutil.copy2(original / relative, root / relative)
    git(root, 'apply', '--check', str(ROOT / 'disabled-veto-fix.patch'))
    git(root, 'apply', str(ROOT / 'disabled-veto-fix.patch'))
    assert git(root, 'diff', '--name-only') == PATH
    git(root, 'diff', '--check')
    git(root, 'add', '--', PATH)
    git(root, 'commit', '-m', 'Preserve disabled chi-square values in Torch live fallback')
    revision = git(root, 'rev-parse', 'HEAD')
    assert git(root, 'rev-parse', 'HEAD^') == base
    assert not git(root, 'status', '--porcelain')
    delta = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', '--full-index', base, revision])
    (OUT / (name + '-runtime.patch')).write_bytes(delta)
    (OUT / (name + '-matchedfilter.py')).write_bytes((root / PATH).read_bytes())
    prepared.append(dict(name=name, base_revision=base, revision=revision, cwd=str(root),
                         runtime_patch_sha256=hashlib.sha256(delta).hexdigest(),
                         file_sha256={PATH: sha(root / PATH)}, native_sha256=source['binaries']))
(ROOT / 'correctness-preparation.json').write_text(json.dumps(prepared, indent=2) + '\n')
print(json.dumps(prepared, indent=2))
