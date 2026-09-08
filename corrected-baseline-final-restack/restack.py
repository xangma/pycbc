"""Locally replay the frozen CPU-base stack with one supplied format patch."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path('/Users/xangma/repos/pycbc')
V = REPO / 'artifacts/torch-precision-validation-20260908'
OUT = V / 'corrected-baseline-final-restack'
OLD = V / 'cpu-base-restack'
QUALITY = V / 'corrected-baseline-quality-v2'
WT = Path('/private/tmp/pycbc-corrected-baseline-final-restack-20260908')
DOCS = Path('/private/tmp/pycbc-corrected-baseline-docs-20260908')
PREFIX = 'codex/corrected-baseline-final-20260908-pr'
MEASURED = 'f582b6fd250d0b82612492979e01e645d5c07afc'
FILES = ['pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq.py']


def git(*args, cwd=REPO, binary=False):
    proc = subprocess.run(['git', *args], cwd=cwd, capture_output=True, check=True)
    with (OUT / 'git-operations.log').open('ab') as log:
        log.write(('git ' + ' '.join(args) + '\n').encode())
        log.write(proc.stderr)
    return proc.stdout if binary else proc.stdout.decode().strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(path):
    return {str(p.relative_to(path)): digest(p.read_bytes())
            for p in sorted(path.rglob('*')) if p.is_file()}


def refs():
    return dict(line.split(' ', 1) for line in
                git('for-each-ref', '--format=%(refname) %(objectname)').splitlines())


def save():
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def audit_pair(before, after, changed):
    names = git('diff', '--name-only', before, after).splitlines()
    assert names == sorted(changed), (before, after, names)
    result = {}
    expected = patched_files(before) if changed else {}
    for path in changed:
        a = git('show', f'{before}:{path}', binary=True)
        b = git('show', f'{after}:{path}', binary=True)
        assert ast.dump(ast.parse(a), include_attributes=False) == ast.dump(
            ast.parse(b), include_attributes=False), path
        assert b == expected[path], path
        if before == MEASURED:
            assert digest(a) == receipt['files'][path]['before_sha256'], path
            assert digest(b) == receipt['files'][path]['after_sha256'], path
        result[path] = {'before_sha256': digest(a), 'after_sha256': digest(b),
                        'full_module_ast_identical': True}
    git('diff', '--check', before, after)
    return {'status': 'PASS', 'changed_files': names, 'files': result,
            'all_other_tracked_paths_identical': True}


def patched_files(head):
    with tempfile.TemporaryDirectory(prefix='pycbc-format-expected-', dir='/private/tmp') as tmp:
        root = Path(tmp)
        for path in FILES:
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(git('show', f'{head}:{path}', binary=True))
        git('apply', str(OUT / 'format.patch'), cwd=root)
        return {p: (root / p).read_bytes() for p in FILES}


resume = (OUT / 'manifest.json').exists()
assert resume or not WT.exists()
old = json.loads((OLD / 'manifest.json').read_text())
receipt = json.loads((QUALITY / 'primary-format-review.json').read_text())
assert receipt['status'] == 'PASS'
assert receipt['measured_source'] == MEASURED
patch = (QUALITY / 'format.patch').read_bytes()
assert digest(patch) == receipt['patch_sha256']
assert set(receipt['files']) == set(FILES)
assert receipt['qlty_after_exit'] == 0
assert (QUALITY / 'qlty-after-exit-code.txt').read_text().strip() == '0'
sarif = json.loads((QUALITY / 'qlty-after.sarif').read_text())
assert not any(run.get('results') for run in sarif['runs'])
assert git('rev-parse', 'HEAD', cwd=DOCS) == MEASURED
assert not git('diff', '--cached', '--name-only', cwd=DOCS)
draft_files = git('diff', '--name-only', cwd=DOCS).splitlines()
assert len(draft_files) == 5 and all(p.startswith('docs/') for p in draft_files)
(OUT / 'draft-before-restack.diff').write_bytes(git('diff', '--binary', cwd=DOCS, binary=True))
for path in draft_files:
    dest = OUT / 'draft-before-restack' / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DOCS / path, dest)
for name in ['format.patch', 'primary-format-review.json', 'qlty-after.sarif',
             'qlty-after.log', 'qlty-after-exit-code.txt']:
    shutil.copyfile(QUALITY / name, OUT / name)
before_refs = (json.loads((OUT / 'manifest.json').read_text())['refs_before']
               if resume else refs())
for p in old['prs']:
    assert before_refs['refs/heads/' + p['staging_ref']] == p['new_head']
    assert 'refs/heads/' + PREFIX + str(p['pr']) not in before_refs
manifest = json.loads((OUT / 'manifest.json').read_text()) if resume else {
    'status': 'replaying_format_only_stack',
    'worktree': str(WT), 'cpu_base': old['cpu_base'],
    'measured_main_head': MEASURED,
    'input_manifest': str(OLD / 'manifest.json'),
    'original_artifact_hashes': snapshot(OLD), 'refs_before': before_refs,
    'formatter_patch_sha256': digest(patch),
    'primary_quality_receipt': receipt,
    'draft_before': {'worktree': str(DOCS), 'head': MEASURED,
                     'branch': git('branch', '--show-current', cwd=DOCS),
                     'files': {p: digest((DOCS / p).read_bytes()) for p in draft_files}},
    'prs': [],
}
save()
if not resume:
    git('worktree', 'add', '--detach', str(WT), old['cpu_base'])
by_pr = {}
for p in old['prs']:
    number = p['pr']
    completed = next((r for r in manifest['prs'] if r['pr'] == number), None)
    if completed:
        assert git('rev-parse', completed['staging_ref']) == completed['new_head']
        by_pr[number] = completed
        continue
    new_base = by_pr[p['parent_pr']]['new_head'] if p['parent_pr'] else old['cpu_base']
    row = {k: p[k] for k in ['pr', 'url', 'published_ref', 'published_base_ref', 'parent_pr']}
    row.update({'old_ref': p['staging_ref'], 'old_head': p['new_head'],
                'old_base': p['new_base'], 'new_base': new_base,
                'staging_ref': PREFIX + str(number), 'commits': []})
    old_commits = git('rev-list', '--reverse', f"{p['new_base']}..{p['new_head']}").splitlines()
    assert old_commits == [c['new'] for c in p['commits']]
    if number in [18, 5, 6, 7, 8]:
        in_progress = git('branch', '--show-current', cwd=WT) == row['staging_ref']
        if not in_progress:
            git('switch', '-c', row['staging_ref'], p['new_head'], cwd=WT)
        row['commits'] = [{'old': c, 'new': c} for c in old_commits]
        if number == 8:
            if not in_progress:
                git('apply', '--check', str(OUT / 'format.patch'), cwd=WT)
                git('apply', str(OUT / 'format.patch'), cwd=WT)
            assert git('diff', '--name-only', cwd=WT).splitlines() == FILES
            expected = patched_files(p['new_head'])
            for path in FILES:
                a = git('show', f"{p['new_head']}:{path}", binary=True)
                b = (WT / path).read_bytes()
                assert b == expected[path]
                assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
            git('diff', '--check', cwd=WT)
            git('add', '--', *FILES, cwd=WT)
            git('commit', '-m', 'Format corrected CPU filtering wrappers', cwd=WT)
            manifest['formatting_commit'] = git('rev-parse', 'HEAD', cwd=WT)
            row['added_formatting_commit'] = manifest['formatting_commit']
    else:
        git('switch', '-c', row['staging_ref'], new_base, cwd=WT)
        for commit in old_commits:
            git('cherry-pick', commit, cwd=WT)
            row['commits'].append({'old': commit, 'new': git('rev-parse', 'HEAD', cwd=WT)})
    row['new_head'] = git('rev-parse', 'HEAD', cwd=WT)
    row['equivalence'] = audit_pair(row['old_head'], row['new_head'],
                                    [] if number in [18, 5, 6, 7] else FILES)
    git('merge-base', '--is-ancestor', new_base, row['new_head'])
    assert not git('status', '--porcelain', cwd=WT)
    manifest['prs'].append(row)
    by_pr[number] = row
    if number == 15:
        manifest['formatting_only_main_head'] = row['new_head']
        print('STABLE_FORMATTING_ONLY_MAIN_HEAD=' + row['new_head'], flush=True)
    save()
after_refs = refs()
assert all(after_refs.get(ref) == sha for ref, sha in before_refs.items())
assert snapshot(OLD) == manifest['original_artifact_hashes']
manifest['original_refs_preserved'] = True
manifest['original_artifact_unchanged'] = True
manifest['status'] = 'format_only_stack_complete_docs_pending'
git('switch', by_pr[15]['staging_ref'], cwd=WT)
(OUT / 'measured-to-formatted.diff').write_bytes(
    git('diff', '--binary', MEASURED, by_pr[15]['new_head'], binary=True))
save()
print(json.dumps({'status': manifest['status'], 'formatting_commit': manifest['formatting_commit'],
                  'main_head': manifest['formatting_only_main_head'], 'refs': len(by_pr)}), flush=True)
