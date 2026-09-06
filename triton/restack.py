"""Prepare local publication refs with the reviewed TaylorF2 delta."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = Path('/private/tmp/pycbc-taylorf2-triton-20260906')
CANDIDATE = '6829dc9bcba07ca7d3da44de7589cc4e9fb84da5'
OLD_MAIN = '607bce53ead14f12af32552a5b2441d3bc667267'
PARTS = json.loads((ROOT.parent / 'torch-stack-format-20260906/stack.json').read_text())
EARLY = ['pycbc/waveform/taylorf2_torch.py', 'pycbc/waveform/taylorf2_triton.py',
         'test/waveform/test_taylorf2_batch.py', 'docs/waveform.rst']
LATE = ['docs/torch_optimizations.rst', 'docs/torch_parity.rst']


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True, **kwargs).strip()


assert not git('status', '--porcelain')
assert set(git('diff', '--name-only', OLD_MAIN, CANDIDATE).splitlines()) == set(EARLY + LATE)
mapping, updated, audit = {}, [], []
for part in PARTS:
    old, old_parent = part['sha'], part['parent']
    if part['key'] in ('format-main', 'pr1', 'pr2', 'pr3', 'pr4', 'pr5', 'pr6'):
        mapping[old] = old
        updated.append(part)
        continue
    paths = EARLY + (LATE if part['key'] in ('pr11', 'format-fft', 'fft-followup', 'cpu-followup') else [])
    for path in paths:
        if path == 'pycbc/waveform/taylorf2_triton.py':
            assert not git('ls-tree', old, '--', path)
        elif (part['key'], path) != ('cpu-followup', 'docs/torch_optimizations.rst'):
            assert git('rev-parse', f'{old}:{path}') == git('rev-parse', f'{OLD_MAIN}:{path}'), (part['key'], path)
    patch = git('diff', '--binary', OLD_MAIN, CANDIDATE, '--', *paths) + '\n'
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / 'index'))
        git('read-tree', old, env=env)
        git('apply', '--cached', '--whitespace=nowarn', input=patch, env=env)
        tree = git('write-tree', env=env)
        git('apply', '--cached', '--reverse', '--whitespace=nowarn', input=patch, env=env)
        assert git('write-tree', env=env) == git('rev-parse', old + '^{tree}')
    parent = mapping[old_parent]
    author = git('show', '-s', '--format=%an%n%ae%n%aI', old).splitlines()
    env = dict(os.environ, GIT_AUTHOR_NAME=author[0], GIT_AUTHOR_EMAIL=author[1], GIT_AUTHOR_DATE=author[2])
    sha = git('commit-tree', tree, '-p', parent, input=git('show', '-s', '--format=%B', old) + '\n', env=env)
    mapping[old] = sha
    entry = dict(part, old_sha=old, old_parent=old_parent, sha=sha, parent=parent,
                 tested_sha=None, files=git('diff', '--name-only', parent, sha).splitlines())
    updated.append(entry)
    assert set(git('diff', '--name-only', old, sha).splitlines()) == set(paths)
    if part['key'] not in ('pr7', 'pr11'):
        old_patch = git('diff', '--binary', old_parent, old) + '\n'
        new_patch = git('diff', '--binary', parent, sha) + '\n'
        assert git('patch-id', '--stable', input=old_patch).split()[0] == git('patch-id', '--stable', input=new_patch).split()[0], part['key']
    assert git('rev-list', '--count', f'{parent}..{sha}') == '1'
    ref = f"refs/heads/codex/torch-triton-restack-20260906/{part['key']}"
    git('update-ref', ref, sha)
    audit.append(dict(key=part['key'], old_sha=old, sha=sha, parent=parent,
                      candidate_files=paths, changed_from_old=git('diff', '--stat', old, sha),
                      feature_patch_unchanged=part['key'] not in ('pr7', 'pr11')))
main = next(p for p in updated if p['key'] == 'pr11')
assert git('rev-parse', main['sha'] + '^{tree}') == git('rev-parse', CANDIDATE + '^{tree}')
for name, value in [('candidate-stack.json', updated), ('restack-audit.json', audit), ('sha-map.json', mapping)]:
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')
git('bundle', 'create', str(ROOT / 'publication.bundle'),
    *[f"refs/heads/codex/torch-triton-restack-20260906/{p['key']}" for p in updated if 'old_sha' in p],
    '^' + PARTS[6]['sha'])
print(json.dumps([dict(key=p['key'], sha=p['sha'], parent=p['parent']) for p in updated if 'old_sha' in p], indent=2))
