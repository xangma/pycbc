"""Bounded structural audit of staged immutable refs, no source writes."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/xangma/repos/pycbc')
manifest = json.loads((OUT / 'manifest.json').read_text())
CPU = manifest['cpu_base']


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def source(head, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{head}:{path}'], text=True)


def tree(head):
    return {line.split('\t', 1)[1]: line.split()[2]
            for line in git('ls-tree', '-r', head).splitlines()}


def native(path):
    return path.startswith('pycbc/lib/') or Path(path).suffix in {
        '.pyx', '.pxd', '.pxi', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh'}


cpu_tests = [f'test/test_{name}_precision.py' for name in ('chisq', 'sigmasq_series', 'strain_psd')]
cpu_blobs = {p: git('rev-parse', f'{CPU}:{p}') for p in cpu_tests}
rows = []
for row in manifest['prs']:
    old, new = row['old_head'], row['new_head']
    assert git('rev-parse', row['staging_ref']) == new
    assert git('merge-base', CPU, new) == CPU
    assert git('merge-base', row['new_base'], new) == row['new_base']
    commits = git('rev-list', '--reverse', f"{row['new_base']}..{new}").splitlines()
    assert commits == [c['new'] for c in row['commits']]
    assert len(commits) == len(row['old_commits'])
    old_tree, new_tree = tree(old), tree(new)
    old_native = {p: h for p, h in old_tree.items() if native(p)}
    new_native = {p: h for p, h in new_tree.items() if native(p)}
    assert old_native == new_native
    for p in cpu_tests:
        assert new_tree[p] == cpu_blobs[p]
        assert 'torch' not in source(new, p).lower()
        assert 'pytest.skip' not in source(new, p)
    docs = {p: h for p, h in old_tree.items() if p.startswith(('docs/', 'examples/'))}
    assert docs == {p: h for p, h in new_tree.items() if p.startswith(('docs/', 'examples/'))}
    rows.append(dict(pr=row['pr'], old_head=old, new_head=new,
                     native_blobs_identical=True, native_count=len(old_native),
                     native_manifest_sha256=hashlib.sha256(json.dumps(old_native, sort_keys=True).encode()).hexdigest(),
                     cpu_test_blobs_identical=True, docs_and_examples_identical=True,
                     exact_replay_count=len(commits),
                     delta=git('diff', '--name-status', old, new).splitlines()))

main = next(r for r in manifest['prs'] if r['pr'] == 15)
old, new = main['old_head'], main['new_head']
functions = {'pycbc/psd/__init__.py': 'from_cli',
             'pycbc/filter/matchedfilter.py': 'sigmasq_series',
             'pycbc/strain/strain.py': 'fourier_segments',
             'pycbc/vetoes/chisq.py': 'power_chisq_at_points_from_precomputed'}
expected = set(functions) | set(cpu_tests) | {p.replace('/test_', '/test_torch_') for p in cpu_tests}
changed = git('diff', '--name-only', old, new).splitlines()
assert set(changed) == expected
details = {}
for path, name in functions.items():
    old_ast, new_ast = ast.parse(source(old, path)), ast.parse(source(new, path))
    old_fn = next(n for n in ast.walk(old_ast) if isinstance(n, ast.FunctionDef) and n.name == name)
    new_fn = next(n for n in ast.walk(new_ast) if isinstance(n, ast.FunctionDef) and n.name == name)
    old_copy, new_copy = copy.deepcopy(old_fn), copy.deepcopy(new_fn)
    old_fn.body = new_fn.body = [ast.Pass()]
    assert ast.dump(old_ast) == ast.dump(new_ast), path
    details[path] = dict(only_changed_function=name, other_module_ast_identical=True)
    if name == 'power_chisq_at_points_from_precomputed':
        cpu_fn = next(n for n in ast.walk(ast.parse(source(CPU, path))) if isinstance(n, ast.FunctionDef) and n.name == name)
        # New body is the CPU function with just the published Torch early dispatch.
        assert ast.dump(new_copy.body[1]) == ast.dump(old_copy.body[1])
        assert [ast.dump(n) for n in new_copy.body[2:]] == [ast.dump(n) for n in cpu_fn.body[1:]]
        # Inline the split `shifts` assignment to establish old/new arithmetic identity.
        for block in ast.walk(new_copy):
            if isinstance(block, ast.If) and len(block.body) > 3:
                for i in range(len(block.body)-1):
                    a, b = block.body[i:i+2]
                    if (isinstance(a, ast.Assign) and isinstance(b, ast.Assign)
                            and ast.unparse(a.targets[0]) == ast.unparse(b.targets[0]) == 'shifts'):
                        b.value.left = a.value
                        del block.body[i]
                        break
        assert ast.dump(old_copy) == ast.dump(new_copy)
        details[path]['cpu_body_exact_standalone_ast'] = True
        details[path]['torch_dispatch_identical_ast'] = True
        details[path]['old_new_arithmetic_identical_after_inlining_shifts'] = True

for label, args in {
    'final-main.diff': ['diff', '--binary', old, new],
    'final-main-stat.txt': ['diff', '--stat', old, new],
    'final-main-production.diff': ['diff', old, new, '--', 'pycbc'],
}.items():
    (OUT / label).write_text(git(*args) + '\n')

result = dict(status='pass', main_old_head=old, main_new_head=new,
              changed_files=changed, functions=details, prs=rows,
              main_native_blobs={p: h for p, h in tree(old).items() if native(p)},
              cpu_test_blobs=cpu_blobs,
              scope='Only four production functions and six precision test files differ from prior main; all other tracked files identical.')
(OUT / 'structural-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'main_new_head', 'scope')}, indent=2))
