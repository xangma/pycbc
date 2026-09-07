#!/usr/bin/env python3
"""Stage the corrected source without changing the failed campaign."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r2')
REVISION = '9578a710479b924e882857c4dffab6ed372a634b'
OLD_REVISION = 'a23401d299ba2dfb93a0437483db02a960b9bfdb'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    assert ROOT == Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r3')
    manifest = json.loads((ROOT / 'staged-files.json').read_text())
    for name, expected in manifest.items():
        assert digest(ROOT / name) == expected, name
    assert json.loads((OLD / 'batch-status.json').read_text())['state'] == 'failed'
    predecessors = [OLD, Path('/home/xangma/pycbc-torch-convergence-20260907'),
                    Path('/home/xangma/pycbc-torch-device-profile-20260907')]
    assert json.loads((predecessors[1] / 'convergence-status.json').read_text())['state'] == 'complete'
    groups = {340874, 1609662, 3526622, 1505900}
    for root in predecessors:
        for receipt in (root / 'stages').glob('*.json'):
            value = json.loads(receipt.read_text())
            for key in ('pgid', 'child_pgid'):
                if isinstance(value.get(key), int):
                    groups.add(value[key])
    alive = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if fields[0] != 'Z' and int(fields[2]) in groups:
                alive.append(int(proc.name))
    assert not alive, alive
    old_source = OLD / 'source'
    assert git(old_source, 'rev-parse', 'HEAD') == OLD_REVISION
    assert not git(old_source, 'status', '--porcelain', '--untracked-files=no')
    source = ROOT / 'source'
    assert not source.exists(), source
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(old_source), str(source)], check=True)
    git(source, 'fetch', str(ROOT / 'source.bundle'), 'HEAD')
    git(source, 'checkout', '--detach', REVISION)
    changed = git(source, 'diff', '--name-only', OLD_REVISION, REVISION).splitlines()
    assert not any(name.endswith(('.pyx', '.pxd', '.c', '.h', '.cu'))
                   or name in ('setup.py', 'pyproject.toml')
                   or name.startswith('pycbc/lib/') for name in changed), changed
    original = json.loads((OLD / 'native-provenance.json').read_text())
    extensions = []
    for row in original['extensions']:
        src = old_source / row['relative_path']
        dst = source / row['relative_path']
        assert digest(src) == row['sha256'], str(src)
        assert not dst.exists(), str(dst)
        shutil.copy2(src, dst)
        assert digest(dst) == row['sha256'], str(dst)
        extensions.append(dict(row, staged_from=str(src)))
    native = dict(original, target_revision=REVISION,
                  copied_from_revision=OLD_REVISION,
                  previous_provenance_sha256=digest(OLD / 'native-provenance.json'),
                  native_build_inputs_diff='', extensions=extensions)
    (ROOT / 'native-provenance.json').write_text(json.dumps(native, indent=2) + '\n')
    assert git(source, 'rev-parse', 'HEAD') == REVISION
    assert not git(source, 'status', '--porcelain', '--untracked-files=no')
    result = {'state': 'staged', 'time': time.time(), 'revision': REVISION,
              'source': str(source), 'extensions_verified': len(extensions),
              'no_live_predecessor_groups': sorted(groups),
              'files_verified': manifest, 'changed_paths': changed}
    (ROOT / 'staging-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'state': 'staged', 'revision': REVISION,
                      'extensions_verified': len(extensions)}))


if __name__ == '__main__':
    main()
