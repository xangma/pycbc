#!/usr/bin/env python3
"""Create an isolated precision-corrected source with unchanged native modules."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
old, new = root / 'source-v4', root / 'source-v5'
bundle = root / 'inspiral-source-v5.bundle'
old_head = 'f2c0abe61e787a26f41208f489c62c877bbd5667'
reference_base = '968bcd558117262af0d603710b054174659adb51'
new_head = sys.argv[1]
assert len(new_head) == 40 and set(new_head) <= set('0123456789abcdef')
assert not new.exists()
assert not (root / 'source-v5.json').exists()
assert not (root / 'source-precision-provenance-v5.json').exists()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


allowed = ['pycbc/filter/matchedfilter.py', 'pycbc/psd/__init__.py',
           'pycbc/strain/strain.py', 'pycbc/vetoes/chisq.py',
           'pycbc/vetoes/chisq_torch.py',
           'test/test_chisq_precision.py', 'test/test_sigmasq_series_precision.py',
           'test/test_strain_psd_precision.py']
assert git(old, 'rev-parse', 'HEAD') == old_head
assert git(old, 'status', '--porcelain') == ''
subprocess.run(['git', 'clone', '--no-hardlinks', str(old), str(new)], check=True)
subprocess.run(['git', '-C', str(new), 'fetch', str(bundle), new_head], check=True)
subprocess.run(['git', '-C', str(new), 'checkout', '--detach', new_head], check=True)
assert git(new, 'rev-parse', 'HEAD^') == old_head
parent_delta = git(new, 'diff', '--name-only', old_head, new_head).splitlines()
assert parent_delta == ['pycbc/vetoes/chisq_torch.py', 'test/test_chisq_precision.py'], parent_delta
old_tree = ast.parse((old / 'pycbc/vetoes/chisq_torch.py').read_text())
new_tree = ast.parse((new / 'pycbc/vetoes/chisq_torch.py').read_text())
def kernel(tree):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                and node.name == '_triton_pointwise_chisq_bin_kernel')
assert ast.dump(kernel(old_tree)) == ast.dump(kernel(new_tree))
changed = git(new, 'diff', '--name-only', reference_base, new_head).splitlines()
assert changed == allowed, changed
previous = json.loads((root / 'source-v4.json').read_text())
native = {}
for name, expected in previous['native_modules_sha256'].items():
    source, target = old / name, new / name
    assert digest(source) == expected and not target.exists()
    shutil.copy2(source, target)
    native[name] = digest(target)
assert len(native) == 11 and native == previous['native_modules_sha256']
assert git(old, 'status', '--porcelain') == git(new, 'status', '--porcelain') == ''
record = dict(source=str(new), commit=new_head, parent=old_head,
              reference_base=reference_base, parent_source_commit=old_head,
              changed_paths=changed, native_modules_sha256=native,
              changed_files_sha256={name: digest(new / name) for name in allowed},
              bundle_sha256=digest(bundle))
save(root / 'source-v5.json', record)
proof = dict(schema_version=1, status='pass', old_source_commit=reference_base,
             parent_source_commit=old_head, reference_base=reference_base,
             new_source_commit=new_head, changed_paths=changed,
             parent_changed_paths=parent_delta,
             exact_parent_git_diff=git(new, 'diff', '--no-ext-diff', '--no-textconv', old_head, new_head),
             normal_cpu_outputs_changed=True, prior_science_reused=False,
             checks=dict(only_reviewed_python_and_test_paths_changed=True,
                         native_hashes_equal=True, both_sources_clean=True,
                         all_final_runs_require_new_source=True, host_launch_and_regression_only_parent_delta=True),
             scope='Normal CPU and supported Torch arithmetic changed. All final reference tuning, scientific validation, matched timings and profiles must use this revision. Earlier timings are supplemental and are not source-equivalent.',
             exact_git_diff=git(new, 'diff', '--no-ext-diff', '--no-textconv', reference_base, new_head),
             input_sha256={str(p): digest(p) for p in
                           [Path(__file__).resolve(), root / 'source-v4.json',
                            root / 'source-v5.json', root / 'source-v3.json', root / 'source-v2.json',
                            root / 'inspiral-source-v2.bundle', root / 'inspiral-source-v3.bundle', root / 'inspiral-source-v4.bundle', bundle]})
save(root / 'source-precision-provenance-v5.json', proof)
print(json.dumps(record))
