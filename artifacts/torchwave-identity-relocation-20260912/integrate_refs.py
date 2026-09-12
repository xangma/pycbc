"""Verify and integrate the prepared local stack with compare-and-swap refs."""
import json
from pathlib import Path
import subprocess

REPO = Path('/Users/xangma/repos/pycbc')
ARTIFACTS = REPO / 'artifacts/torchwave-identity-relocation-20260912'
OLD = {
    'torch-pr6-waveform': '0faa84ce02620b8b9865f4ba6f4e76e4f965f431',
    'torch-pr7-taylorf2': '0a8812d0275159b8a7e5f9f9eb67f3f4ca514769',
    'torch-pr8-domain-compat': 'ab4d97308e6e9eb1b8bc33195fb69f0f0427e618',
    'torch-pr9-detector': '09e06d566729c7c189a385de02f72203fd7602e3',
    'torch-pr10-inference': '0376dd3b385bc8271d4cb52b3c40171581ffe672',
    'torch-pr11-performance-evidence': 'dc4f0f87fc3c961d959bc038978781e3020b3064',
}


def git(*args, **kwargs):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
                                   text=True, **kwargs).strip()


assert not git('status', '--porcelain', '--untracked-files=no')
assert git('branch', '--show-current') == 'torch-pr11-performance-evidence'
new = {}
previous = git('rev-parse', 'torch-pr5-search')
for number, (branch, old) in enumerate(OLD.items(), 6):
    assert git('rev-parse', branch) == old, branch
    tip = git('rev-parse', f'codex/provider-identity-pr{number}-20260912')
    paths = set(git('diff', '--name-only', old, tip).splitlines())
    assert paths == {'pycbc/waveform/torchwave.py',
                     'test/test_torchwave_provider_identity.py'}, (branch, paths)
    assert git('show', f'{tip}:pycbc/waveform/torchwave.py') == git(
        'show', 'codex/provider-identity-pr6-20260912:pycbc/waveform/torchwave.py')
    git('merge-base', '--is-ancestor', previous, tip)
    previous = tip
    new[branch] = tip

# The detached checkout lets all actual branch heads advance atomically.
git('switch', '--detach', previous)
commands = ['start']
for branch, old in OLD.items():
    commands.append(f'create refs/heads/codex/backup-identity-20260912/{branch} {old}')
    commands.append(f'update refs/heads/{branch} {new[branch]} {old}')
commands += ['prepare', 'commit']
try:
    git('update-ref', '--stdin', input='\n'.join(commands) + '\n')
finally:
    git('switch', 'torch-pr11-performance-evidence')
assert not git('status', '--porcelain', '--untracked-files=no')
(ARTIFACTS / 'integrated-refs.json').write_text(
    json.dumps({'before': OLD, 'after': new}, indent=2) + '\n')
print(json.dumps({'integrated': new, 'tracked_clean': True}, indent=2))
