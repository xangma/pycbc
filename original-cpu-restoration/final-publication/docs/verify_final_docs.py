"""Check the six-file delivery against strict build and measured-source receipts."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

OUT = Path(__file__).resolve().parent
WT = Path('/private/tmp/pycbc-original-cpu-restoration-docs-20260908')
BASE = '1d22031fd31c6e5bcb48a68fe11772e960bd406b'
MEASURED = 'aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0'
FILES = ['docs/torch_batch_numerics.rst', 'docs/torch_benchmark_protocol.rst',
         'docs/torch_optimizations.rst', 'docs/torch_performance.rst', 'docs/torch_reference_campaign.rst',
         'docs/torch_testing.rst']

changed = subprocess.check_output(
    ['git', '-C', str(WT), 'diff', '--name-only', BASE], text=True).splitlines()
assert changed == FILES, changed
subprocess.run(['git', '-C', str(WT), 'diff', '--check', BASE], check=True)
build = json.loads((OUT / 'sphinx-build-result.json').read_text())
assert build['returncode'] == 0, build
assert build['measured_runtime'] == MEASURED
assert build['runtime']['version'] == MEASURED
assert len(build['pages']) == 19
assert all(x in build['command'] for x in ('-E', '-a', '-W', '--keep-going'))
hashes = json.loads((OUT / 'sphinx-source-hashes.json').read_text())
assert set(hashes) == set(build['pages'])
for name, expected in hashes.items():
    assert hashlib.sha256((WT / 'docs' / name).read_bytes()).hexdigest() == expected
links = []
for name in FILES:
    text = (WT / name).read_text()
    assert not re.search(r'PENDING_|FINAL_|\*\*DRAFT', text), name
    found = re.findall(r'<(https://github.com/xangma/pycbc/(?:tree|blob)/[0-9a-f]{40}/[^>]+)>', text)
    assert found, name
    links.extend(found)
checks = {
    'status': 'pass', 'documentation_base': BASE, 'measured_runtime': MEASURED,
    'files': {name: hashlib.sha256((WT / name).read_bytes()).hexdigest() for name in FILES},
    'strict_build_pages': len(build['pages']), 'immutable_links': sorted(set(links)),
    'source_proof': str(OUT / 'docs-source-claims-verified.json'),
    'build_result': str(OUT / 'sphinx-build-result.json'),
}
(OUT / 'final-docs-checks.json').write_text(json.dumps(checks, indent=2) + '\n')
print('PASS: six documentation files; strict 19-page build; aa6 imports; immutable links; no placeholders.')
