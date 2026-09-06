#!/usr/bin/env python3
"""Create the deepcopy-fix source and prove the scope of the revision."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
old = root / 'source'
new = root / 'source-v2'
bundle = root / 'inspiral-source-v2.bundle'
old_head = 'fb4b335eeeeaeaa907c1143b45e0191e2d977761'
new_head = sys.argv[1]
assert len(new_head) == 40 and set(new_head) <= set('0123456789abcdef')
assert not new.exists()
assert not (root / 'source-v2.json').exists()
assert not (root / 'reference-source-equivalence.json').exists()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


assert git(old, 'rev-parse', 'HEAD') == old_head
assert git(old, 'status', '--porcelain') == ''
subprocess.run(['git', 'clone', '--no-hardlinks', str(old), str(new)], check=True)
subprocess.run(['git', '-C', str(new), 'fetch', str(bundle), new_head], check=True)
subprocess.run(['git', '-C', str(new), 'checkout', '--detach', new_head], check=True)
assert git(new, 'rev-parse', 'HEAD^') == old_head
changed = git(new, 'diff', '--name-only', old_head, new_head).splitlines()
assert changed == ['pycbc/scheme.py', 'test/test_scheme_runtime.py']
old_ast = ast.parse((old / 'pycbc/scheme.py').read_text())
new_ast = ast.parse((new / 'pycbc/scheme.py').read_text())
torch_class = next(node for node in new_ast.body
                   if isinstance(node, ast.ClassDef) and node.name == 'TorchScheme')
methods = [node for node in torch_class.body
           if isinstance(node, ast.FunctionDef) and node.name == '__deepcopy__']
assert len(methods) == 1
expected = ast.parse('def __deepcopy__(self, memo):\n    memo[id(self)] = self\n    return self\n').body[0]
assert ast.dump(methods[0], include_attributes=False) == ast.dump(expected, include_attributes=False)
torch_class.body.remove(methods[0])
assert ast.dump(old_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False)

old_record = json.loads((root / 'source.json').read_text())
native = {}
for name, expected_hash in old_record['native_modules_sha256'].items():
    source, target = old / name, new / name
    assert digest(source) == expected_hash and not target.exists()
    shutil.copy2(source, target)
    native[name] = digest(target)
assert native == old_record['native_modules_sha256']
assert git(new, 'status', '--porcelain') == ''
assert git(old, 'status', '--porcelain') == ''
record = dict(source=str(new), commit=new_head, parent=old_head,
              changed_paths=changed, native_modules_sha256=native,
              bundle_sha256=digest(bundle))
save(root / 'source-v2.json', record)
proof = dict(schema_version=1, status='pass', old_source_commit=old_head,
             new_source_commit=new_head, changed_paths=changed,
             checks=dict(only_TorchScheme_deepcopy_added=True,
                         all_other_tracked_blobs_identical=True,
                         native_hashes_equal=True, both_sources_clean=True),
             scope='Earlier normal CPU tuning and waveform/boundary validation remain applicable: all source outside the added TorchScheme deepcopy method and its regression test is identical. New matched runs must all use the new source. No old Torch results are promoted.',
             exact_git_diff=git(new, 'diff', '--no-ext-diff', '--no-textconv', old_head, new_head),
             input_sha256={str(p): digest(p) for p in
                           [Path(__file__).resolve(), root / 'source.json',
                            root / 'source-v2.json', root / 'inspiral-source.bundle', bundle]})
save(root / 'reference-source-equivalence.json', proof)
print(json.dumps(record))
