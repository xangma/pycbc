"""Record the immutable format-only stack and pending documentation draft."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
REPO = Path('/Users/xangma/repos/pycbc')
DOCS = Path('/private/tmp/pycbc-corrected-baseline-docs-20260908')
m = json.loads((OUT / 'manifest.json').read_text())


def git(*args, cwd=REPO):
    return subprocess.check_output(['git', *args], cwd=cwd)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(head):
    result = {}
    for entry in git('ls-tree', '-rz', head).split(b'\0'):
        if entry:
            meta, path = entry.split(b'\t', 1)
            result[path.decode()] = meta.decode()
    return result


measured = m['measured_main_head']
formatted = m['formatting_only_main_head']
before, after = tree(measured), tree(formatted)
changes = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
assert changes == ['pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq.py']
for p in m['prs']:
    assert git('rev-parse', p['old_ref']).decode().strip() == p['old_head']
    assert git('rev-parse', p['staging_ref']).decode().strip() == p['new_head']
    expected = [c['new'] for c in p['commits']]
    if 'added_formatting_commit' in p:
        expected.append(p['added_formatting_commit'])
    actual = git('rev-list', '--reverse', f"{p['new_base']}..{p['new_head']}").decode().splitlines()
    assert actual == expected
old_artifacts = Path(m['input_manifest']).parent
current_artifacts = {str(p.relative_to(old_artifacts)): sha(p.read_bytes())
                     for p in sorted(old_artifacts.rglob('*')) if p.is_file()}
assert current_artifacts == m['original_artifact_hashes']
current_refs = dict(line.split(' ', 1) for line in
                    git('for-each-ref', '--format=%(refname) %(objectname)').decode().splitlines())
assert all(current_refs.get(ref) == value for ref, value in m['refs_before'].items())
namespaces = {}
for namespace in ['bin', 'test', 'tools', '.github', 'docs']:
    a = git('rev-parse', f'{measured}:{namespace}').decode().strip()
    b = git('rev-parse', f'{formatted}:{namespace}').decode().strip()
    assert a == b
    namespaces[namespace] = {'before_tree': a, 'after_tree': b, 'byte_identical': True,
                             'tracked_files': sum(p.startswith(namespace + '/') for p in before)}
extensions = {'.pyx', '.pxd', '.pxi', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.so', '.a'}
native = {p: obj for p, obj in before.items()
          if Path(p).suffix.lower() in extensions or p.startswith('pycbc/lib/')}
assert all(after[p] == obj for p, obj in native.items())
format_files = {}
for path in changes:
    a = git('show', f'{measured}:{path}')
    b = git('show', f'{formatted}:{path}')
    assert ast.dump(ast.parse(a), include_attributes=False) == ast.dump(ast.parse(b), include_attributes=False)
    assert sha(b) == m['primary_quality_receipt']['files'][path]['after_sha256']
    format_files[path] = {'before_git_entry': before[path], 'after_git_entry': after[path],
                          'before_sha256': sha(a), 'after_sha256': sha(b),
                          'full_module_ast_identical_excluding_locations': True}
audit = {'status': 'PASS', 'measured_head': measured, 'formatted_head': formatted,
         'tracked_files': len(before), 'changed_files': format_files,
         'all_other_tracked_paths_identical': True, 'namespaces': namespaces,
         'native_files': native, 'native_file_count': len(native),
         'all_original_refs_preserved': True, 'original_artifact_unchanged': True,
         'all_15_heads_preserve_exact_commit_ranges_and_parent_ancestry': True,
         'all_15_tree_comparisons': {str(p['pr']): p['equivalence'] for p in m['prs']}}
(OUT / 'structural-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
assert git('rev-parse', 'HEAD', cwd=DOCS).decode().strip() == formatted
draft_paths = git('diff', '--name-only', cwd=DOCS).decode().splitlines()
assert sorted(draft_paths) == sorted(m['draft_before']['files'])
assert not git('diff', '--cached', '--name-only', cwd=DOCS)
git('diff', '--check', cwd=DOCS)
draft = {'status': 'awaiting_verified_final_timings_and_immutable_links',
         'worktree': str(DOCS), 'branch': git('branch', '--show-current', cwd=DOCS).decode().strip(),
         'base_head': formatted, 'commit': None, 'original_draft_transferred_byte_identically': True,
         'source_mapping_prose_updated': True,
         'files': {p: sha((DOCS / p).read_bytes()) for p in draft_paths},
         'pending_tokens': {p: sorted(set(re.findall(r'PENDING_[A-Z_]+', (DOCS / p).read_text())))
                            for p in draft_paths if 'PENDING_' in (DOCS / p).read_text()},
         'sphinx': {'status': 'prepared_not_run_pending_final_evidence',
                    'command': ['/private/tmp/pycbc-torch-publication-20260908-venv/bin/python',
                                str(OUT / 'build_docs.py')],
                    'documentation_base': formatted, 'runtime_source': measured,
                    'runtime_mapping': 'Full module AST equivalence verified for the two format-only files; all other tracked source identical.'}}
(OUT / 'draft.diff').write_bytes(git('diff', '--binary', cwd=DOCS))
(OUT / 'draft-stat.txt').write_bytes(git('diff', '--stat', cwd=DOCS))
lint_command = ['/Users/xangma/miniconda3/bin/flake8', *changes, '--select', 'F401']
lint = subprocess.run(lint_command, cwd=DOCS, capture_output=True)
(OUT / 'flake8-f401.log').write_bytes(lint.stdout + lint.stderr)
assert lint.returncode == 0
m['status'] = 'format_only_stack_complete_docs_draft_ready_awaiting_final_evidence'
m['documentation'] = draft
m['validation'] = {'structural_audit': str(OUT / 'structural-audit.json'),
                   'primary_final_format_source_review': {
                       'path': str(OUT / 'primary-final-format-source-review.json'),
                       'reviewed_manifest': str(OUT / 'manifest-format-only.json'),
                       'reviewed_manifest_sha256': sha((OUT / 'manifest-format-only.json').read_bytes()),
                       'status': 'PASS'},
                   'changed_modules_flake8_f401': {'command': lint_command, 'returncode': lint.returncode},
                   'primary_qlty': {'status': 'PASS', 'returncode': 0, 'issues': 0,
                                    'scope': 'Primary isolated scratch at measured f582 + supplied patch; final main formatted-file hashes independently match.',
                                    'published_head_rerun_owner': 'primary'},
                   'unit_tests': 'Not rerun for line wrapping only; all 15 full-tree and changed-module AST comparisons passed. Existing frozen validation stays in the unchanged cpu-base-restack artifact.',
                   'sphinx': 'Deferred until verified final timings and immutable evidence links arrive.'}
table = '\n'.join(f"| #{p['pr']} | {('#' + str(p['parent_pr'])) if p['parent_pr'] else 'CPU base'} | `{p['old_head']}` | `{p['new_head']}` |" for p in m['prs'])
text = f'''# Corrected-baseline final restack

Formatting-only main is **`{formatted}`**, on `codex/corrected-baseline-final-20260908-pr15`.
The complete 15-ref stack is ready for primary validation. The five-file documentation draft is on `{draft['branch']}`, based on this main. Verified final statistics and immutable archive links remain pending; the docs have not been committed or built yet.

## Change and measured-source mapping

- Measured main remains `{measured}`; standalone corrected CPU remains `{m['cpu_base']}`.
- Added commit `{m['formatting_commit']}` to owner PR #8, directly after its frozen head. It contains exactly the supplied two-file formatter patch, SHA256 `{m['formatter_patch_sha256']}`.
- Replayed each original descendant's exact commit range onto its new parent, including #19 → #16 and #17. #18/#5/#6/#7 are unchanged heads under new refs. Original staging refs and all preexisting refs retain their original object IDs.
- Final main differs from measured main only in `pycbc/filter/matchedfilter.py` and `pycbc/vetoes/chisq.py`. Both complete module ASTs match after excluding source locations; their source bytes differ and match the primary formatter receipt exactly.
- Every other tracked file is byte-identical. Whole `bin/`, `test/`, `tools/`, `.github/`, and `docs/` tree IDs match. All {len(native)} native files match. Per-head audits show only the same supplied formatting changes at each changed prefix, preserving all optional-leaf contents relative to its own frozen head.
- The eventual documentation-bearing publication adds only docs to the formatted main. Its final commit ID will be recorded after final evidence arrives; no benchmark has been rerun on the formatted or documentation-bearing head.

## Frozen-to-formatted mapping

Each new ref uses `codex/corrected-baseline-final-20260908-prNN`. `old_head` refers to the previous CPU-base restack, not an older published stack. Full ref/base/commit mappings are in [manifest.json](manifest.json).

| PR | Parent | Frozen CPU-base head | New formatted head |
| --- | --- | --- | --- |
{table}

## Validation

[structural-audit.json](structural-audit.json) records all 15 tree/AST comparisons, exact commit-range and ancestry checks, namespace tree IDs, native file entries, and preservation of every preexisting ref and every original `cpu-base-restack` artifact file.

The primary's Qlty run on frozen `f582` plus this patch passed with zero issues. Its raw SARIF, log, exit code and review receipt are copied here. The final formatted file hashes match that receipt. Primary owns the requested Qlty rerun against `{formatted}`; this sidecar performed no network operations. Local F401 lint for both modified modules and `git diff --check` passed.

Primary independently confirmed the final formatted main's exact two-file difference, full-module AST equality, and native/ancestry checks for all 15 heads in [primary-final-format-source-review.json](primary-final-format-source-review.json). Its manifest hash matches the preserved [format-only manifest snapshot](manifest-format-only.json); the current manifest additionally tracks the evolving documentation draft.

No unit tests were rerun for these line-wrapping changes. All changed modules pass complete AST comparisons at every prefix, and the original frozen unit-test evidence remains unchanged at [the CPU-base report](../cpu-base-restack/report.md). Optional leaves were structurally checked; no new optional runtime claims are made.

## Documentation draft and remaining work

Worktree: `{DOCS}`. The original five-file draft was backed up and transferred byte-for-byte before updating the measured/publication source mapping in performance, reference-campaign and protocol prose. Current [draft.diff](draft.diff) changes only the same five docs; its whitespace check passes. Timings and archive URLs remain explicit placeholders.

The draft also states that the campaign's acquired `sources.bundle` and hash-verified bank preserve the measured source and input after published refs advance. The reference page points readers to reproduction instructions for the bundle's required frozen base. Primary's archive and independent verification remain in progress; no timing values were copied from unverified drafts.

After verified final numbers and immutable links arrive: fill the draft, run the prepared actual-Torch-page Sphinx build with `-W`, then commit the docs and record the final publication mapping. The helper verifies the formatted-source mapping before using the frozen runtime and records both pins. The optional descendants currently start at the formatting-only main; they will need to inherit the final docs commit for a publication stack that includes those docs.

Build command: `/private/tmp/pycbc-torch-publication-20260908-venv/bin/python {OUT / 'build_docs.py'}`. No build or other owned long-running job is active.

Original CPU-base worktree, refs, measured campaign artifacts and original `cpu-base-restack` artifacts were not modified. Publication remains with primary.
'''
(OUT / 'report.md').write_text(text)
m['artifact_sha256'] = {str(p.relative_to(OUT)): sha(p.read_bytes()) for p in sorted(OUT.rglob('*'))
                        if p.is_file() and p.name != 'manifest.json'}
(OUT / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps({'status': m['status'], 'main': formatted, 'formatting_commit': m['formatting_commit'],
                  'refs': len(m['prs']), 'native_files_identical': len(native),
                  'documentation_files': len(draft_paths), 'pending_tokens': draft['pending_tokens']}))
