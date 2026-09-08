"""Local-only, exact-range replay. Stops at each conflict for manual review."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/Users/xangma/repos/pycbc')
OUT = Path(__file__).resolve().parent
WT = Path('/private/tmp/pycbc-cpu-base-restack-20260908')
CPU = '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
BASE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
STATE = OUT / 'manifest.json'


def git(*args, cwd=WT):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()


def save(state):
    STATE.write_text(json.dumps(state, indent=2) + '\n')


if not STATE.exists():
    rows = []
    for line in (OUT.parent / 'stack-cleanup-plan.md').read_text().splitlines():
        match = re.match(r'\| \[#(\d+)\]\(([^)]+)\) \| `([^`]+)` \| `([0-9a-f]{40})` \| `([^`]+)` \| (\d+) \|', line)
        if not match:
            continue
        number, url, branch, head, base_ref, count = match.groups()
        parent = next((r for r in rows if r['published_ref'] == base_ref), None)
        base = parent['old_head'] if parent else BASE
        commits = git('rev-list', '--reverse', '--first-parent', f'{base}..{head}').splitlines()
        assert len(commits) == int(count)
        rows.append(dict(pr=int(number), url=url, published_ref=branch,
                         old_head=head, old_base=base, published_base_ref=base_ref,
                         parent_pr=parent['pr'] if parent else None,
                         staging_ref=f'codex/cpu-base-restack-20260908-pr{number}',
                         old_commits=commits, commits=[], conflict_decisions=[]))
    assert len(rows) == 15
    state = dict(status='rebuilding', worktree=str(WT), cpu_base=CPU,
                 frozen_original=BASE, prs=rows, validation=[],
                 refs_before=git('for-each-ref', '--format=%(refname) %(objectname)', cwd=ROOT).splitlines())
    assert not any('refs/heads/codex/cpu-base-restack-20260908-' in r for r in state['refs_before'])
    save(state)
else:
    state = json.loads(STATE.read_text())

target = int(sys.argv[1])
row = next(r for r in state['prs'] if r['pr'] == target)
if 'new_base' not in row:
    parent = next((r for r in state['prs'] if r['pr'] == row['parent_pr']), None)
    row['new_base'] = parent['new_head'] if parent else CPU
    assert git('status', '--porcelain', '--untracked-files=no') == ''
    git('switch', '-c', row['staging_ref'], row['new_base'])
    save(state)

for old in row['old_commits'][len(row['commits']):]:
    cherry = subprocess.run(['git', '-C', str(WT), 'rev-parse', '-q', '--verify', 'CHERRY_PICK_HEAD'], capture_output=True, text=True)
    if cherry.returncode == 0:
        assert cherry.stdout.strip() == old
        assert not git('diff', '--name-only', '--diff-filter=U'), 'Unresolved conflicts'
        result = subprocess.run(['git', '-C', str(WT), '-c', 'core.editor=true', 'cherry-pick', '--continue'], capture_output=True, text=True)
    else:
        result = subprocess.run(['git', '-C', str(WT), '-c', 'rerere.enabled=false', 'cherry-pick', old], capture_output=True, text=True)
    with (OUT / 'replay.log').open('a') as log:
        log.write(f'\nPR {target} old {old}\n{result.stdout}{result.stderr}')
    if result.returncode:
        files = git('diff', '--name-only', '--diff-filter=U').splitlines()
        row['pending_conflict'] = dict(old_commit=old, files=files)
        save(state)
        print(json.dumps(row['pending_conflict'], indent=2))
        sys.exit(1)
    row.pop('pending_conflict', None)
    row['commits'].append(dict(old=old, new=git('rev-parse', 'HEAD'), subject=git('show', '-s', '--format=%s', old)))
    save(state)

row['new_head'] = git('rev-parse', 'HEAD')
assert git('merge-base', row['new_base'], row['new_head']) == row['new_base']
save(state)
print(json.dumps({k: row[k] for k in ('pr', 'staging_ref', 'new_base', 'new_head')}, indent=2))
