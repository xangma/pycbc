"""Attach reviewed documentation to PR15 and replay its three descendants."""
import json
from pathlib import Path
import subprocess
import sys

O = Path(__file__).resolve().parent
R = Path('/Users/xangma/repos/pycbc')
W = Path('/private/tmp/pycbc-torch-parity-final-20260908')
M = json.loads((O / 'manifest.json').read_text())
DOC = sys.argv[1]

def git(*args):
    p = subprocess.run(['git', *args], cwd=W if W.exists() else R, text=True, capture_output=True)
    with (O / 'finalize-docs.log').open('a') as f:
        f.write('$ git ' + ' '.join(args) + '\n' + p.stdout + p.stderr)
    p.check_returncode()
    return p.stdout.strip()

changes = git('diff-tree', '--no-commit-id', '--name-only', '-r', DOC).splitlines()
assert len(changes) == 7 and all(p.startswith('docs/') and p.endswith('.rst') for p in changes), changes
assert not W.exists()
main = next(r for r in M['prs'] if r['pr'] == 15)
git('worktree', 'add', '--detach', str(W), main['new_head'])
for row in M['prs']:
    row['tested_head'] = row['new_head']
    if row['pr'] not in (15, 19, 16, 17):
        continue
    parent = next(r for r in M['prs'] if r['pr'] == row['parent_pr'])
    branch = f"codex/torch-parity-final-20260908-pr{row['pr']:02d}"
    if row['pr'] == 15:
        git('checkout', '-b', branch, row['new_head'])
        git('cherry-pick', DOC)
    else:
        commits = git('rev-list', '--reverse', row['new_base'] + '..' + row['new_head']).splitlines()
        git('checkout', '-b', branch, parent['new_head'])
        for commit in commits:
            git('cherry-pick', commit)
        row['new_base'] = parent['new_head']
    row['new_head'] = git('rev-parse', 'HEAD')
    row['staging_ref'] = branch
    (O / 'manifest-final.json').write_text(json.dumps(M, indent=2) + '\n')
    print(row['pr'], row['new_head'], flush=True)
M['documentation_commit'] = DOC
M['status'] = 'complete'
(O / 'manifest-final.json').write_text(json.dumps(M, indent=2) + '\n')
