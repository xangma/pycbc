"""Archive final source mapping, documentation and optional-leaf receipts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

O = Path(__file__).resolve().parent
R = Path('/Users/xangma/repos/pycbc')
E = R / 'artifacts/cpu-precision-evidence-20260908'
D = E / 'torch-parity-fix/final-publication'
M = json.loads((O / 'manifest-final.json').read_text())

def git(*args):
    return subprocess.check_output(['git', *args], cwd=R)

assert M['status'] == 'complete' and len(M['prs']) == 15
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=E)
D.mkdir()
mapping = []
for row in M['prs']:
    changed = git('diff', '--name-only', row['tested_head'], row['new_head']).decode().splitlines()
    assert all(p.startswith('docs/') and p.endswith('.rst') for p in changed), (row['pr'], changed)
    mapping.append(dict(pr=row['pr'], tested_head=row['tested_head'], final_head=row['new_head'],
                        final_base=row['new_base'], non_documentation_files_identical=True,
                        documentation_changes=changed))
main = next(r for r in M['prs'] if r['pr'] == 15)
hashes = json.loads((O / 'docs-audit/sphinx-source-hashes.json').read_text())
assert len(hashes) == 19
for name, sha in hashes.items():
    assert hashlib.sha256(git('show', main['new_head'] + ':docs/' + name)).hexdigest() == sha, name
build = json.loads((O / 'docs-audit/sphinx-build-result.json').read_text())
assert build['returncode'] == 0 and build['measured_runtime'] == main['tested_head']
(D / 'source-mapping.json').write_text(json.dumps(dict(status='pass', documentation_sources_match_final_main=True,
    original_cpu=M['cpu_base'], measured_main=main['tested_head'], final_main=main['new_head'], prs=mapping), indent=2) + '\n')
for name in ['manifest-final.json', 'source-preservation-final.json', 'verify-source-preservation-final.py',
             'finalize-docs.py', 'prepare-final-evidence.py']:
    shutil.copy2(O / name, D / name)
shutil.copytree(O / 'optional-pr17', D / 'optional-pr17')
docs = D / 'docs-audit'
docs.mkdir()
for name in ['build_docs.py', 'runtime-build.json', 'sphinx-build-result.json', 'sphinx-build.log', 'sphinx-source-hashes.json']:
    shutil.copy2(O / 'docs-audit' / name, docs / name)
for name in hashes:
    for source, target in [(O/'docs-audit/render-final/docs'/name, docs/'sources'/name),
                           (O/'docs-audit/render-final/html'/Path(name).with_suffix('.html'), docs/'html'/Path(name).with_suffix('.html'))]:
        assert source.is_file(), source
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
git('bundle', 'create', str(D/'final-stack.bundle'), *(r['staging_ref'] for r in M['prs']), '^'+M['cpu_base'])
(D/'README.md').write_text(f'''# Final Torch stack publication

The executable qualification measured `{main['tested_head']}`. Final main `{main['new_head']}` differs only in the seven reviewed documentation pages. All other final branch heads are byte-identical to their tested source outside documentation. [Source mapping](source-mapping.json) lists every tested and final head; [manifest](manifest-final.json) records the dependency stack. The bundle contains all final heads with original CPU `{M['cpu_base']}` as its prerequisite.

Strict Sphinx built all 19 scoped pages successfully using the measured runtime. Their recorded source hashes match the final main Git tree. [Build receipt](docs-audit/sphinx-build-result.json) records the exact scope and exclusions; source pages and rendered HTML are archived alongside it.

The separate optional CPU optimization leaf passed 266 tests with 42 skipped. Its [receipt](optional-pr17/provenance.json) pins the tested head, native sources and reused binaries; [XML](optional-pr17/pytest.xml) and logs are retained. This leaf and the optional FFT leaf remain outside the four-route executable qualification.

Original CPU and the withdrawn PR20 remain outside the changes. Publication keeps all existing PR numbers, branch names, draft states and agent-assisted labels. Review descriptions and live publication verification are retained by the publishing workspace.
''')
p=E/'torch-parity-fix/README.md'
p.write_text(p.read_text().replace('subsequent documentation changes and restacked optional leaves are mapped in final-publication when available.',
    'subsequent documentation changes and restacked optional leaves are recorded in [final-publication](final-publication/README.md).'))
files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in E.rglob('*') if p.is_file() and '.git' not in p.parts and p.name!='SHA256SUMS.json'}
(E/'SHA256SUMS.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
print('PASS: all 15 source mappings and 19 rendered sources verified; final evidence prepared')
