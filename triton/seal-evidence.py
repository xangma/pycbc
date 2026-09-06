"""Seal completed remote TaylorF2 evidence without source/build/cache trees."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/home/xangma/pycbc-taylorf2-triton-20260906')
SHA = '6829dc9bcba07ca7d3da44de7589cc4e9fb84da5'
status = json.loads((ROOT / 'restack-validation-status.json').read_text())
assert status['finished'] and set(status['nonzero']) <= {'format-fft-qlty'}, status
manifest = json.loads((ROOT / 'full-v3-clean/manifest.json').read_text())
assert manifest['status'] == 'ok' and manifest['finished_utc']

def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

source = ROOT / 'source'
assert command(['git', 'rev-parse', 'HEAD'], source) == SHA
assert command(['git', 'status', '--porcelain'], source) == ''
native = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (source / 'pycbc').rglob('*.so')}
prepared = json.loads((ROOT / 'preparation.json').read_text())
assert native == prepared['binaries'], 'Benchmark native binaries changed'
final = dict(finished_utc=datetime.now(timezone.utc).isoformat(), sha=SHA,
             tree=command(['git', 'rev-parse', 'HEAD^{tree}'], source),
             status=command(['git', 'status', '--porcelain'], source),
             native_sha256=native, native_unchanged=True,
             nvidia_smi=command(['nvidia-smi']),
             packages=json.loads(command([
                 '/home/xangma/pycbc-torch-split-20260905/venv/bin/python',
                 '-m', 'pip', 'list', '--format=json'])))
(ROOT / 'finalization.json').write_text(json.dumps(final, indent=2) + '\n')

files = []
for name in ('harness', 'logs', 'validation-logs'):
    files += [p for p in (ROOT / name).rglob('*') if p.is_file() and not p.is_symlink()]
for name in ('full-v3-clean', 'full-v3', 'smoke-v2'):
    files += [p for p in (ROOT / name).iterdir() if p.is_file() and not p.is_symlink()]
files += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.json', '.py', '.sh')]
files = sorted(set(files))
inventory = {str(p.relative_to(ROOT)): dict(size=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
sealed = ROOT / 'sealed-manifest.json'
sealed.write_text(json.dumps(inventory, indent=2) + '\n')
with tarfile.open(ROOT / 'sealed-evidence.tar.gz', 'w:gz') as archive:
    for p in files + [sealed]:
        archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(files=len(files), bytes=sum(v['size'] for v in inventory.values()),
                      native_unchanged=True)))
