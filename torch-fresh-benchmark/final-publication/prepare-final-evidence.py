"""Archive rendered benchmark documentation and exact final source mapping."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

O = Path(__file__).resolve().parent
R = Path('/Users/xangma/repos/pycbc')
F = O.parent / 'torch-fresh-benchmark-20260908'
E = O.parent / 'cpu-precision-evidence-20260908'
D = E / 'torch-fresh-benchmark/final-publication'
M = json.loads((O / 'manifest-final.json').read_text())
S = json.loads((F / 'remote/summary.json').read_text())

def git(*args):
    return subprocess.check_output(['git', *args], cwd=R)

assert M['status'] == 'complete' and len(M['prs']) == 15
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=E)
D.mkdir()
mapping = []
for row in M['prs']:
    changed = git('diff', '--name-only', row['tested_head'], row['new_head']).decode().splitlines()
    assert all(p.startswith('docs/') and p.endswith('.rst') for p in changed)
    mapping.append(dict(pr=row['pr'], tested_head=row['tested_head'], final_head=row['new_head'],
                        final_base=row['new_base'], non_documentation_files_identical=True,
                        documentation_changes=changed))
main = next(r for r in M['prs'] if r['pr'] == 15)
measured = S['source_commits']['proposed']
benchmark_diff = git('diff', '--name-only', measured, main['new_head']).decode().splitlines()
assert benchmark_diff == ['docs/torch_performance.rst'], benchmark_diff
hashes = json.loads((O / 'sphinx-source-hashes.json').read_text())
assert len(hashes) == 19
for name, sha in hashes.items():
    assert hashlib.sha256(git('show', main['new_head'] + ':docs/' + name)).hexdigest() == sha, name
build = json.loads((O / 'sphinx-build-result.json').read_text())
assert build['returncode'] == 0 and build['measured_runtime'] == main['tested_head']
(D / 'source-mapping.json').write_text(json.dumps(dict(status='pass', documentation_sources_match_final_main=True,
    original_cpu=M['cpu_base'], regression_tested_main=main['tested_head'], benchmark_measured_main=measured,
    final_main=main['new_head'], benchmark_non_documentation_files_identical=True,
    benchmark_documentation_changes=benchmark_diff, prs=mapping), indent=2) + '\n')
for name in ['manifest-final.json', 'restack.py', 'write-results.py', 'prepare-final-evidence.py',
             'build_docs.py', 'sphinx-build-result.json', 'sphinx-build.log', 'sphinx-source-hashes.json']:
    shutil.copy2(O / name, D / name)
for name in hashes:
    for source, target in [(O / 'render-final/docs' / name, D / 'sources' / name),
                           (O / 'render-final/html' / Path(name).with_suffix('.html'), D / 'html' / Path(name).with_suffix('.html'))]:
        assert source.is_file(), source
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
git('bundle', 'create', str(D / 'final-stack.bundle'), *(r['staging_ref'] for r in M['prs']), '^' + M['cpu_base'])
(D / 'README.md').write_text(f'''# Fresh benchmark documentation and final branch mapping

The fresh complete-executable benchmark measured `{measured}`. Final main `{main['new_head']}` changes only `docs/torch_performance.rst`, adding the current result table and its measurement scope. Its runtime code is byte-identical to the benchmarked main. Regression tests previously measured `{main['tested_head']}`; all 15 final heads retain their respective tested bytes outside documentation. [Source mapping](source-mapping.json) records both kinds of validation separately.

Strict Sphinx built all 19 scoped pages with no warnings. Their recorded source hashes match final main. The [build receipt](sphinx-build-result.json) records the exact scope and exclusions. Sources and rendered HTML are included; shared HTML assets are omitted. The bundle includes all final stack heads and requires original CPU `{M['cpu_base']}`.

The four qualifications and 16 timed samples remain unchanged in the parent evidence folder. Existing PR numbers, branch names, dependencies, draft states and agent-assisted labels are retained. Optional FFT and native CPU optimization leaves remain outside the executable benchmark.
''')
p = E / 'torch-fresh-benchmark/README.md'
p.write_text(p.read_text().replace('Documentation and final branch mappings will be appended separately; benchmark records are immutable.',
    'Final documentation and branch mappings are in [final-publication](final-publication/README.md); benchmark records are unchanged.'))
files = {str(p.relative_to(E)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in E.rglob('*') if p.is_file() and '.git' not in p.parts and p.name != 'SHA256SUMS.json'}
(E / 'SHA256SUMS.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
print('PASS: all 15 tested-source mappings and 19 rendered sources verified; final evidence prepared')
