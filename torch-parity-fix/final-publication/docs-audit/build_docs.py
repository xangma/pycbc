"""Build actual Torch pages with the previously validated Sphinx scope."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--runtime-hash', required=True)
args = parser.parse_args()
WORKTREE = Path('/private/tmp/pycbc-torch-parity-docs-20260908')
RUNTIME = args.runtime.resolve()
FROZEN = args.runtime_hash
DOC_BASE = '8f71727e28ede9de3e3bbed82bb616eab710cbe7'
render = OUT / 'render-final'
source = render / 'docs'
source.mkdir(parents=True, exist_ok=True)
assert subprocess.check_output(['git', '-C', str(RUNTIME), 'rev-parse', 'HEAD'], text=True).strip() == FROZEN
assert not subprocess.check_output(['git', '-C', str(RUNTIME), 'status', '--porcelain'], text=True)
changed = subprocess.check_output(['git', '-C', str(WORKTREE), 'diff', '--name-only', DOC_BASE], text=True).splitlines()
assert changed and all(p.startswith('docs/') for p in changed), changed
assert len(changed) == 7, changed
pages = sorted(p.name for p in (WORKTREE / 'docs').glob('torch*.rst')) + [
    'waveform.rst', 'waveform_plugin.rst', 'install.rst', 'install_cuda.rst',
    'install_lalsuite.rst', 'install_virtualenv.rst', 'docker.rst']
for name in pages:
    text = (WORKTREE / 'docs' / name).read_text()
    assert 'PENDING_' not in text and 'FINAL_' not in text, f'Final evidence is still pending in {name}'
    shutil.copy2(WORKTREE / 'docs' / name, source / name)
for name in ('images', 'data', '_static', '_templates'):
    path = WORKTREE / 'docs' / name
    if path.exists():
        shutil.copytree(path, source / name, dirs_exist_ok=True)
shutil.copytree(WORKTREE / 'examples', render / 'examples', dirs_exist_ok=True)
(source / 'index.rst').write_text('Original CPU baseline and Torch\n================================\n\n.. toctree::\n   :maxdepth: 2\n\n   torch\n   waveform\n   waveform_plugin\n   install\n   api\n')
(source / 'api.rst').write_text('Referenced API objects\n======================\n\n.. autoclass:: pycbc.scheme.TorchScheme\n\n.. autofunction:: pycbc.waveform.compress.fd_decompress\n\n.. autofunction:: pycbc.waveform.plugin.add_custom_waveform\n\n.. autoclass:: pycbc.types.frequencyseries.FrequencySeries\n')
conf = str(WORKTREE / 'docs/conf.py')
(source / 'conf.py').write_text(f"from pathlib import Path\nexec(compile(Path({conf!r}).read_text(), {conf!r}, 'exec'))\nintersphinx_mapping = {{}}\nhtml_logo = None\n")
env = os.environ.copy()
env.update(SKIP_PYCBC_DOCS_INCLUDE='1', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=str(RUNTIME), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
# Import the prepared runtime without changing its version or build files.
provenance = json.loads(subprocess.check_output([
    sys.executable, '-c', 'import json,pycbc,pycbc.version,sphinx; print(json.dumps(dict(pycbc=pycbc.__file__,version=pycbc.version.git_hash,sphinx=sphinx.__version__)))'
], cwd=source, env=env, text=True))
assert Path(provenance['pycbc']).parent == RUNTIME / 'pycbc', provenance
assert provenance['version'] == FROZEN, provenance
command = [sys.executable, '-m', 'sphinx', '-b', 'html', '-E', '-a', '-W', '--keep-going', str(source), str(render / 'html')]
log = OUT / 'sphinx-build.log'
with log.open('w') as output:
    process = subprocess.Popen(command, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(json.dumps(dict(pid=process.pid, host='local', cwd=str(source), command=command,
                          log=str(log), stop_command=f'kill -TERM {process.pid}')), flush=True)
    code = process.wait()
result = dict(command=command, returncode=code, pages=pages, runtime=provenance,
              documentation_base=DOC_BASE, measured_runtime=FROZEN,
              source_mapping='Seven documentation-only changes from PR15 8f71727e28. API imports use the independently named measured runtime; publication onto the restacked source is a separate mapping.',
              source_head=subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip(),
              finished=time.time(), environment={k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'SKIP_PYCBC_DOCS_INCLUDE', 'MPLBACKEND', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
              scope='All actual Torch pages plus waveform/plugin/installation dependencies, repository extensions and theme, real plot/command directives, targeted API context; excludes unrelated manual and _include generators, no external intersphinx inventory or remote logo.')
assert not subprocess.check_output(['git', '-C', str(RUNTIME), 'status', '--porcelain'], text=True)
assert all((source / name).read_bytes() == (WORKTREE / 'docs' / name).read_bytes() for name in pages)
(OUT / 'sphinx-build-result.json').write_text(json.dumps(result, indent=2) + '\n')
(OUT / 'sphinx-source-hashes.json').write_text(json.dumps({name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in pages}, indent=2) + '\n')
print('Sphinx exit', code, flush=True)
sys.exit(code)
