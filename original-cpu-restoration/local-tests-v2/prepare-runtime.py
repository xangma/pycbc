"""Reuse local unchanged native binaries; generate version only in staging."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WT = Path('/private/tmp/pycbc-original-cpu-validation-20260908')
SOURCE = Path('/private/tmp/pycbc-cpu-precision-corrections-20260908')
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', '-C', str(WT), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not git('status', '--porcelain', '--untracked-files=no')
head = git('rev-parse', 'HEAD')
native_patterns = ['*.pyx', '*.pxd', '*.pxi', '*.c', '*.cc', '*.cpp', '*.h', '*.hpp', '*.cu', 'pycbc/lib/**']
assert not git('diff', '--name-only', '66789ac4a7', head, '--', *native_patterns)
reused = {}
for source in (SOURCE / 'pycbc').rglob('*.so'):
    relative = source.relative_to(SOURCE)
    target = WT / relative
    shutil.copy2(source, target)
    assert sha(target) == sha(source)
    reused[str(relative)] = dict(source=str(source), sha256=sha(target))
assert len(reused) == 11
os.chdir(WT)
sys.path.insert(0, str(WT))
setup = ast.parse((WT / 'setup.py').read_text())
fn = next(n for n in setup.body if isinstance(n, ast.FunctionDef) and n.name == 'get_version_info')
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WT / 'setup.py'), 'exec'), namespace)
namespace['get_version_info']()
spec = importlib.util.spec_from_file_location('staging_version', WT / 'pycbc/version.py')
version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version)
assert version.git_hash == head
record = dict(head=head, worktree=str(WT), interpreter=sys.executable,
              native_source_policy='All native source files identical to CPU baseline, including headers and pycbc/lib.',
              reused_native=reused, version_sha256=sha(WT / 'pycbc/version.py'))
(OUT / f'runtime-{head[:12]}.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(head=head, native_count=len(reused), version=version.git_hash)))
