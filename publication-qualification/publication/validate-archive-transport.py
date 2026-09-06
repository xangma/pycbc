"""Record isolated transport validation and streaming checks of the three large inputs."""
import ast
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import archive_transport as transport


root = Path(__file__).resolve().parent
reference = root.parent.parent / 'torch-inspiral-reference-20260906'
started = time.monotonic()
result = dict(state='running', host=socket.gethostname(), cwd=str(Path.cwd()),
              pid=os.getpid(), command=[sys.executable, '-B', str(Path(__file__).resolve())],
              started_utc=datetime.now(timezone.utc).isoformat(), real_archive_copied=False,
              scientific_campaigns_run=False, original_artifacts_modified=False)
print(json.dumps(result), flush=True)
paths = [root / name for name in ('stage-evidence.py', 'archive_transport.py',
         'ARCHIVE_TRANSPORT.md', 'test_archive_transport.py', 'validate-archive-transport.py')]
paths += [root.parent / 'write-docs.py', reference / 'build-reference-report-v4.py']
before = {str(path): transport.file_record(path) for path in paths}
backup = root / 'stage-evidence.before-gzip-e9e0e15fbac7.py'
old = ast.parse(backup.read_text())
new = ast.parse((root / 'stage-evidence.py').read_text())
functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                         for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != 'main'}
assert functions(old) == functions(new)
result['unchanged_stager_functions'] = sorted(functions(old))
assert before[str(reference / 'build-reference-report-v4.py')]['sha256'] == 'ad303e00008dd4792353ae23cb0856087bf7a643388d1e9b4391de2f8801ccda'
assert before[str(root.parent / 'write-docs.py')]['sha256'] == '627e1500f6d2b6bca5c28574721b4af3a949be5133216134a436744a488e0db8'
for path in paths:
    if path.suffix == '.py':
        compile(path.read_text(), str(path), 'exec')
test = subprocess.run([sys.executable, '-B', str(root / 'test_archive_transport.py')],
                      capture_output=True, text=True)
print(test.stdout + test.stderr, flush=True)
assert test.returncode == 0
result['isolated_tests'] = dict(returncode=test.returncode, output=test.stdout + test.stderr)
large = sorted(path for path in (reference / 'runs').rglob('perf.data')
               if path.stat().st_size > transport.THRESHOLD)
assert len(large) == 3
logical = {str(path.relative_to(reference)): dict(source=str(path), **transport.file_record(path)) for path in large}
cache = root / 'gzip-transport-validation-cache'
actual, mapping = transport.prepare_transport(logical, cache)
manifest = transport.verify_archive(cache)
assert manifest['logical_files'] == {name: transport.metadata(item) for name, item in logical.items()}
assert transport.prepare_transport(logical, cache, create=False) == (actual, mapping)
assert {name: transport.file_record(Path(item['source'])) for name, item in logical.items()} == manifest['logical_files']
result['large_input_checks'] = dict(original_files=logical, actual_cached_files=actual, transport=mapping,
                                  all_decompressed_hashes_and_sizes_match=True,
                                  frozen_cache_reuse_passed=True, original_inputs_unchanged=True,
                                  caveat='Validation cache contains only the three large inputs; use a fresh staging cache for the full supplement.')
diff = ''.join(difflib.unified_diff(backup.read_text().splitlines(True),
        (root / 'stage-evidence.py').read_text().splitlines(True),
        fromfile=str(backup), tofile=str(root / 'stage-evidence.py')))
for name in ('archive_transport.py', 'ARCHIVE_TRANSPORT.md', 'test_archive_transport.py', 'validate-archive-transport.py'):
    path = root / name
    diff += ''.join(difflib.unified_diff([], path.read_text().splitlines(True), fromfile='/dev/null', tofile=str(path)))
diff_path = root / 'gzip-transport-review.diff'
diff_path.write_text(diff)
after = {str(path): transport.file_record(path) for path in paths}
assert before == after
result.update(state='passed', finished_utc=datetime.now(timezone.utc).isoformat(),
              wall_seconds=time.monotonic() - started, input_files=before, input_files_after=after,
              full_diff=dict(path=str(diff_path), **transport.file_record(diff_path)),
              original_stager=dict(path=str(backup), **transport.file_record(backup)))
output = root / 'gzip-transport-validation.json'
output.write_bytes(transport.canonical(result))
print(json.dumps(dict(state=result['state'], seconds=result['wall_seconds'],
                     receipt=str(output), receipt_record=transport.file_record(output),
                     transport=mapping,
                     compressed=[dict(original=name, original_bytes=logical[name]['bytes'], **item)
                                 for name, item in manifest['compressed_files'].items()])), flush=True)
