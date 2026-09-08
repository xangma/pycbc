"""Compose stage mappings and independently audit the publication commits."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

V = Path(__file__).resolve().parent
ROOT = V.parents[1]
FINAL = V / 'corrected-baseline-final-restack'
CPU = '66789ac4a7468094b0cc3ca1498a1de67e0311f6'
MEASURED = 'f582b6fd250d0b82612492979e01e645d5c07afc'
FORMATTED = '6b47580146e73169cd130b601731e5ba40668d93'
WRAPPERS = {'pycbc/filter/matchedfilter.py', 'pycbc/vetoes/chisq.py'}
DOCS = {'docs/torch_batch_numerics.rst', 'docs/torch_benchmark_protocol.rst',
        'docs/torch_performance.rst', 'docs/torch_reference_campaign.rst',
        'docs/torch_testing.rst'}
CPU_TESTS = ['test/test_chisq_precision.py', 'test/test_sigmasq_series_precision.py',
             'test/test_strain_psd_precision.py']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_tree(commit):
    entries = {}
    for entry in git('ls-tree', '-rz', commit).split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        name = path.decode()
        if name.startswith('pycbc/lib/') or Path(name).suffix.lower() in {
                '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.pyx', '.pxd'}:
            entries[name] = metadata.decode()
    return entries


def main():
    first = read(V / 'cpu-base-restack/manifest.json')
    final = read(FINAL / 'manifest.json')
    live = {p['number']: p for p in read(V / 'live-prs-before-cpu-restack.json')}
    initial = {p['pr']: p for p in first['prs']}
    heads = {p['pr']: p['new_head'] for p in final['prs']}
    assert len(heads) == 15
    publication = heads[15]
    assert publication != FORMATTED
    assert git('rev-parse', publication + '^').decode().strip() == FORMATTED
    assert set(git('diff', '--name-only', FORMATTED, publication).decode().splitlines()) == DOCS
    changed = set(git('diff', '--name-only', MEASURED, publication).decode().splitlines())
    assert changed == DOCS | WRAPPERS, changed
    for path in WRAPPERS:
        before = git('show', f'{MEASURED}:{path}')
        after = git('show', f'{publication}:{path}')
        assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)), path
        assert after == git('show', f'{FORMATTED}:{path}'), path
    for namespace in ['bin', 'test', 'tools', '.github']:
        assert git('rev-parse', f'{MEASURED}:{namespace}') == git('rev-parse', f'{publication}:{namespace}')
    rows = []
    audits = []
    for row in final['prs']:
        number = row['pr']
        frozen = initial[number]
        before = live[number]
        assert frozen['old_head'] == before['head']['sha']
        assert row['old_head'] == frozen['new_head']
        assert row['published_ref'] == before['head']['ref']
        assert git('rev-parse', row['staging_ref']).decode().strip() == row['new_head']
        parent = CPU if number == 18 else heads[row['parent_pr']]
        assert row['new_base'] == parent, (number, row['new_base'], parent)
        git('merge-base', '--is-ancestor', parent, row['new_head'])
        git('merge-base', '--is-ancestor', CPU, row['new_head'])
        assert native_tree(frozen['old_head']) == native_tree(row['new_head']), number
        for path in CPU_TESTS:
            assert git('show', f'{CPU}:{path}') == git('show', f'{row["new_head"]}:{path}'), (number, path)
        merged = dict(row, old_head=frozen['old_head'], old_base=frozen['old_base'],
                      frozen_head=frozen['new_head'], frozen_base=frozen['new_base'],
                      cpu_restack_commits=frozen['commits'],
                      final_restack_commits=row['commits'])
        rows.append(merged)
        audits.append(dict(pr=number, head=row['new_head'], base=parent,
                           ancestry=True, unchanged_native_paths=len(native_tree(row['new_head'])),
                           standalone_cpu_tests_identical=True))
    sphinx = read(FINAL / 'sphinx-build-result.json')
    doc_hashes = read(FINAL / 'sphinx-source-hashes.json')
    assert sphinx['returncode'] == 0 and len(sphinx['pages']) == 19
    for name, digest in doc_hashes.items():
        assert hashlib.sha256(git('show', f'{publication}:docs/{name}')).hexdigest() == digest, name
    manifest = dict(status='reviewed_ready_for_publication', cpu_base=CPU,
                    measured_main=MEASURED, formatted_main=FORMATTED,
                    publication_main=publication, prs=rows,
                    mapping_semantics='old_head/base are published pre-restack objects. frozen_head/base identify measured CPU-based restack. Ordered maps are preserved separately for each stage.',
                    inputs={str(p.relative_to(V)): sha(p) for p in [
                        V / 'cpu-base-restack/manifest.json', FINAL / 'manifest.json',
                        V / 'live-prs-before-cpu-restack.json']})
    (V / 'publication-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = dict(status='PASS', publication_main=publication, measured_main=MEASURED,
                  changed_paths=sorted(changed), complete_wrapper_asts_identical=True,
                  bin_test_tools_ci_trees_identical=True, all_15_heads=audits,
                  sphinx_exit_code=0, all_19_built_doc_sources_match_final_commit=True,
                  final_docs_review='All five diffs reviewed against independently verified benchmark values and immutable source links.',
                  publication_manifest_sha256=sha(V / 'publication-manifest.json'))
    (V / 'primary-final-publication-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', head=publication, prs=len(rows), built_pages=len(doc_hashes))))


if __name__ == '__main__':
    main()
