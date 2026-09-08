from pathlib import Path
import hashlib
import json
import subprocess

V = Path(__file__).resolve().parent
ROOT = Path('/Users/xangma/repos/pycbc')
WT = Path('/private/tmp/pycbc-torch-benchmark-summary-20260908')


def git(*args, cwd=WT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


old = json.loads((ROOT / 'artifacts/torch-docs-cleanup-20260908/manifest-final.json').read_text())
rows = {r['pr']: dict(r) for r in old['prs']}
main = git('rev-parse', 'HEAD')
changed = git('diff', '--name-only', rows[15]['new_head'], main).splitlines()
assert len(changed) == 1 and all(p.startswith('docs/') and p.endswith('.rst') for p in changed)
assert not git('status', '--porcelain')
for r in rows.values():
    r['old_head'], r['old_base'] = r['new_head'], r['new_base']
rows[15]['new_head'] = main
rows[15]['staging_ref'] = 'codex/torch-benchmark-summary-20260908'
for n in [19, 16, 17]:
    row = rows[n]
    parent = rows[row['parent_pr']]['new_head']
    branch = f'codex/torch-benchmark-summary-20260908-pr{n}'
    commits = git('rev-list', '--reverse', row['old_base'] + '..' + row['old_head']).splitlines()
    git('checkout', '-b', branch, parent)
    for commit in commits:
        git('cherry-pick', commit)
    row.update(new_head=git('rev-parse', 'HEAD'), new_base=parent, staging_ref=branch)
    print(n, row['new_head'], flush=True)
for row in rows.values():
    diff = git('diff', '--name-only', row['tested_head'], row['new_head']).splitlines()
    assert all(p.startswith('docs/') and p.endswith('.rst') for p in diff), (row['pr'], diff)
    diff_before = git('diff', '--name-only', row['old_head'], row['new_head']).splitlines()
    assert all(p.startswith('docs/') and p.endswith('.rst') for p in diff_before)
    row['documentation_files_since_test'] = diff
    row['benchmark_summary_files'] = diff_before
    git('merge-base', '--is-ancestor', old['cpu_base'], row['new_head'])
    git('merge-base', '--is-ancestor', row['new_base'], row['new_head'])
hashes = json.loads((V / 'sphinx-source-hashes.json').read_text())
for name, sha in hashes.items():
    data = subprocess.check_output(['git', 'show', main + ':docs/' + name], cwd=ROOT)
    assert hashlib.sha256(data).hexdigest() == sha, name
mapping = dict(cpu_base=old['cpu_base'], status='complete', documentation_commit=main, prs=list(rows.values()),
               non_documentation_files_byte_identical_to_tested_heads=True,
               non_documentation_files_byte_identical_to_previous_publication=True,
               rendered_main_source_hashes_match=True,
               previous_evidence='https://github.com/xangma/pycbc/tree/32eaadc79d07d0e4f81cf54641e288c85032b4fd/torch-docs-cleanup')
(V / 'manifest-final.json').write_text(json.dumps(mapping, indent=2) + '\n')
print('PASS: all 15 runtime trees retain tested bytes; 19 rendered sources match main with benchmark summary.')
