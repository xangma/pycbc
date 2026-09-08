import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
WT = Path('/private/tmp/pycbc-corrected-baseline-final-restack-20260908')
DOCS = Path('/private/tmp/pycbc-corrected-baseline-docs-20260908')
PREFIX = 'codex/corrected-baseline-publish-20260908-pr'


def git(*args, cwd=WT):
    return subprocess.check_output(['git', *args], cwd=cwd).decode().strip()


def mutate(*args):
    p = subprocess.run(['git', *args], cwd=WT, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (OUT / 'publication-replay.log').open('a') as log:
        log.write('git ' + ' '.join(args) + '\n' + p.stdout + '\n')
    p.check_returncode()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((OUT / 'manifest.json').read_text())
assert manifest['status'] == 'format_only_stack_complete_docs_draft_ready_awaiting_final_evidence'
formatted = copy.deepcopy(manifest['prs'])
dochead = git('rev-parse', 'HEAD', cwd=DOCS)
formhead = manifest['formatting_only_main_head']
assert git('rev-parse', dochead + '^') == formhead
assert not git('status', '--porcelain', cwd=DOCS)
assert not git('status', '--porcelain')
docfiles = git('diff', '--name-only', formhead, dochead).splitlines()
assert len(docfiles) == 5 and all(f.startswith('docs/') for f in docfiles)
source_hashes = json.loads((OUT / 'sphinx-source-hashes.json').read_text())
for name, expected in source_hashes.items():
    assert sha(DOCS / 'docs' / name) == expected, name
for p in formatted:
    assert git('rev-parse', p['staging_ref']) == p['new_head']
    assert git('rev-parse', p['old_ref']) == p['old_head']
    assert not git('for-each-ref', '--format=%(refname)', 'refs/heads/' + PREFIX + str(p['pr']))

heads = {}
replay = {}
for p in manifest['prs']:
    num = p['pr']
    p['formatted_head'] = p['new_head']
    p['formatted_base'] = p['new_base']
    p['format_ref'] = p['staging_ref']
    p['format_commits'] = copy.deepcopy(p['commits'])
    p['format_equivalence'] = p.pop('equivalence')
    p['staging_ref'] = PREFIX + str(num)
    p['new_base'] = heads[p['parent_pr']] if p['parent_pr'] else manifest['cpu_base']
    if num in (19, 16, 17):
        mutate('switch', '-c', p['staging_ref'], p['new_base'])
        commits = git('rev-list', '--reverse', p['formatted_base'] + '..' + p['formatted_head']).splitlines()
        assert commits == [c['new'] for c in p['format_commits']]
        stage_map = []
        for c in commits:
            mutate('cherry-pick', '-x', c)
            head = git('rev-parse', 'HEAD')
            replay[c] = head
            stage_map.append({'formatted': c, 'final': head})
        p['formatted_to_final_commits'] = stage_map
        p['new_head'] = git('rev-parse', 'HEAD')
        for c in p['commits']:
            c['new'] = replay[c['new']]
    else:
        p['new_head'] = dochead if num == 15 else p['formatted_head']
        mutate('branch', p['staging_ref'], p['new_head'])
    if num == 15:
        p['added_documentation_commit'] = dochead
    heads[num] = p['new_head']
    assert subprocess.run(['git', 'merge-base', '--is-ancestor', p['new_base'], p['new_head']], cwd=WT).returncode == 0
    changed = git('diff', '--name-only', p['formatted_head'], p['new_head']).splitlines()
    assert changed == (docfiles if num in (15, 19, 16, 17) else []), (num, changed)
    for f in docfiles if changed else []:
        assert git('rev-parse', p['new_head'] + ':' + f) == git('rev-parse', dochead + ':' + f)
    p['documentation_inheritance'] = {'status': 'PASS', 'changed_from_formatted': changed,
                                      'all_other_tracked_files_and_modes_identical': True}
    print('#' + str(num), p['new_head'], flush=True)

mutate('switch', PREFIX + '15')
audit = {'status': 'PASS', 'documentation_commit': dochead, 'prs': []}
formatfiles = ['pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq.py']
for p in manifest['prs']:
    changed = git('diff', '--name-only', p['old_head'], p['new_head']).splitlines()
    expected = ([] if p['pr'] in (18, 5, 6, 7) else formatfiles)
    expected = sorted(expected + (docfiles if p['pr'] in (15, 19, 16, 17) else []))
    assert changed == expected, (p['pr'], changed)
    for f in set(changed) & set(formatfiles):
        before = git('show', p['old_head'] + ':' + f)
        after = git('show', p['new_head'] + ':' + f)
        assert ast.dump(ast.parse(before), include_attributes=False) == ast.dump(ast.parse(after), include_attributes=False)
    assert git('rev-parse', p['old_ref']) == p['old_head']
    assert git('rev-parse', p['format_ref']) == p['formatted_head']
    assert git('rev-parse', p['staging_ref']) == p['new_head']
    namespaces = {}
    for namespace in ('bin', 'test', 'tools', '.github'):
        oldtree = git('rev-parse', p['old_head'] + ':' + namespace)
        newtree = git('rev-parse', p['new_head'] + ':' + namespace)
        assert oldtree == newtree
        namespaces[namespace] = newtree
    p['equivalence'] = {'status': 'PASS', 'changed_files': changed,
                        'full_module_ast_identical_excluding_locations': True,
                        'all_other_tracked_files_and_modes_identical': True,
                        'native_sources_unchanged': True,
                        'identical_namespace_tree_ids': namespaces}
    audit['prs'].append({'pr': p['pr'], 'old_head': p['old_head'],
                         'formatted_head': p['formatted_head'], 'new_head': p['new_head'],
                         'new_base': p['new_base'], **p['equivalence']})
    subprocess.run(['git', 'diff', '--check', p['old_head'], p['new_head']], cwd=WT, check=True)

oldartifact = Path(manifest['input_manifest']).parent
for name, expected in manifest['original_artifact_hashes'].items():
    assert sha(oldartifact / name) == expected, name
audit['frozen_cpu_base_refs_and_all_format_refs_preserved'] = True
audit['original_cpu_base_artifact_hashes_preserved'] = True
audit['built_source_hashes_match_final_commit'] = True
dump('final-structural-audit.json', audit)
for fname, args in [('final-docs.diff', ('diff', formhead, dochead)),
                    ('final-main.diff', ('diff', manifest['measured_main_head'], dochead)),
                    ('final-docs-stat.txt', ('diff', '--stat', formhead, dochead))]:
    (OUT / fname).write_text(git(*args) + '\n')

manifest['format_only_prs'] = formatted
manifest['status'] = 'complete_final_docs_and_all_15_local_refs_ready'
manifest['final_main_head'] = dochead
manifest['old_head_semantics'] = 'Frozen CPU-base-restack head, not a live published head. Primary composes the publication manifest using its live snapshot.'
manifest['evidence_commit'] = 'e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c'
manifest['final_evidence_inputs'] = str(OUT / 'final-evidence-inputs.json')
manifest['documentation'].update(status='committed_and_validated', commit=dochead,
                                 files={f: sha(DOCS / f) for f in docfiles}, pending_tokens={})
build = json.loads((OUT / 'sphinx-build-result.json').read_text())
assert build['returncode'] == 0 and len(build['pages']) == 19
manifest['documentation']['sphinx'] = {'status': 'PASS', 'receipt': str(OUT / 'sphinx-build-result.json'),
                                      'source_hashes': str(OUT / 'sphinx-source-hashes.json'),
                                      'built_bytes_match_documentation_commit': True, **build}
manifest['validation']['final_structural_audit'] = str(OUT / 'final-structural-audit.json')
manifest['validation']['primary_qlty'].pop('published_head_rerun_owner', None)
manifest['validation']['primary_qlty']['scope'] = 'PASS (0 issues) on frozen f582 plus the supplied formatter patch. Primary independently verified the formatted main has identical formatted file bytes and complete-module AST equality. Final main adds only five documentation files; no new Qlty run claimed or outstanding.'
manifest['validation']['sphinx'] = 'PASS: strict -b html -E -a -W --keep-going, all 19 actual selected pages; committed docs match every built-source hash.'
manifest['validation']['render'] = {'status': 'PASS_WITH_PREEXISTING_CONSOLE_ERRORS',
    'pages_visually_reviewed': ['torch_performance.html', 'torch_reference_campaign.html'],
    'timing_table': 'All four values and ranges verified in browser DOM; theme horizontal overflow is scrollable.',
    'screenshots': ['/Users/xangma/repos/pycbc/output/playwright/corrected-baseline-performance.png',
                    '/Users/xangma/repos/pycbc/output/playwright/corrected-baseline-reference.png'],
    'limitation': 'Two preexisting browser syntax errors: unchanged docs/conf.py registers terminal.css and theme_overrides.css as JavaScript. Content renders; no configuration changes made.',
    'external_page_requests': 'Blocked during browser inspection.'}
manifest['validation']['unit_tests'] = 'Not repeated for source line wrapping and documentation. Complete AST equality and tracked-tree comparisons pass at all 15 prefixes; original frozen CPU-base-restack test evidence is unchanged.'
manifest['publication_performed'] = False
manifest['owned_running_jobs'] = []
table = '\n'.join('| #' + str(p['pr']) + ' | ' + ('#' + str(p['parent_pr']) if p['parent_pr'] else 'CPU base') + ' | `' + p['new_head'] + '` |' for p in manifest['prs'])
report = f'''# Corrected-baseline final restack

Final documentation-bearing main: **`{dochead}`**. All 15 local refs are ready under `{PREFIX}NN`. The five documentation files are committed, strict Sphinx passed all 19 pages, and optional #19 → #16 and #17 now inherit this exact documentation commit. Publication remains with primary.

## Final heads

| PR | Parent | Final head |
| --- | --- | --- |
{table}

[manifest.json](manifest.json) contains all 15 refs, parent bases and commit mappings, plus the preserved formatting stage. Its `old_head` continues to mean the frozen CPU-base restack, not the live published branch. Primary will compose the separate publication manifest from its live snapshot.

## Evidence and source mapping

The final documentation links the immutable verified campaign at [e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c](https://github.com/xangma/pycbc/tree/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign). All 20 processes, five trigger comparisons and 16 timing-to-qualification comparisons pass. CPU versus CPU passes the full scientific comparison. Torch full-PSD comparisons fail below 30 Hz (2375/3105 finite-bin violations per segment for CPU/CUDA); timings remain descriptive and no equivalent-output Torch speedup is claimed. Independent evidence verification is PASS_WITH_LIMITATIONS and the full scientific result is FAIL. The documentation retains the timing-policy amendment, Torch distribution/import metadata discrepancy and receipt-based verification limitations, and separates the standalone CPU correction cost.

Measured main remains `{manifest['measured_main_head']}`; corrected CPU remains `{manifest['cpu_base']}`. Formatter commit `{manifest['formatting_commit']}` belongs to #8; replayed formatted main is `{formhead}`. Relative to the measured main, final main differs only in two Python formatting files and five documentation files. Complete module ASTs match excluding source locations; Python source bytes differ. All other tracked paths and modes, including native source, `bin/`, `test/`, `tools/` and `.github/`, are identical. No benchmark was rerun on the formatted or documentation-bearing head. Every original frozen CPU-base ref and every formatting-stage ref is preserved.

## Validation

- [Final structural audit](final-structural-audit.json): all 15 ancestry, source/AST, documentation inheritance and ref checks pass. Original CPU-base artifact hashes remain unchanged. Optional source contents match their corresponding formatted heads exactly.
- [Sphinx receipt](sphinx-build-result.json) and [log](sphinx-build.log): exit 0 with `-b html -E -a -W --keep-going`. All 19 actual Torch and required waveform/plugin/installation pages were built with the repository extensions/theme and real plot/command directives. Scope excludes unrelated manual/include generators, external intersphinx inventories and the remote logo. The runtime pin is frozen measured main; [all built-source hashes](sphinx-source-hashes.json) match the committed documentation. An initial short-title-underline warning was corrected before the successful run; its receipt remains archived.
- Browser inspection of performance/reference pages confirms rendered prose, immutable links and all timing-table values. Wide tables use the theme's horizontal scrolling. Two preexisting console syntax errors arise because unchanged `docs/conf.py` registers `terminal.css` and `theme_overrides.css` as JavaScript; this does not prevent the reviewed content rendering. Screenshots are under `/Users/xangma/repos/pycbc/output/playwright/`.
- Qlty PASS with zero issues applies to the primary run on frozen measured main plus the exact supplied formatter patch. Primary independently verified formatted-main file bytes, whole-module AST equality and all-15 ancestry/native equivalence in [its receipt](primary-final-format-source-review.json), bound to [the preserved format-only manifest](manifest-format-only.json). Final main adds only documentation. No additional Qlty run is claimed or outstanding.
- Changed-module F401 and final whitespace checks pass. Unit tests were not repeated for line wrapping and documentation; the unchanged [frozen CPU-base validation](../cpu-base-restack/report.md) remains the runtime evidence.

[Final docs diff](final-docs.diff), [final main diff](final-main.diff), and [replay log](publication-replay.log) make the completed changes reviewable. No remote mutations or PR publication were performed by this task.
'''
(OUT / 'report.md').write_text(report)
manifest['artifact_sha256'] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.iterdir())
                              if p.is_file() and p.name not in ('manifest.json', 'render-http.log')}
dump('manifest.json', manifest)
print('FINAL_MAIN=' + dochead, flush=True)
print('FINAL_MANIFEST=' + str(OUT / 'manifest.json'), flush=True)
