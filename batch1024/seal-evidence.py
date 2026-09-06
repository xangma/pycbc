"""Archive completed batch-1024 evidence, preserving failures and raw receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/home/xangma/pycbc-torch-batch1024-20260906')
PY = '/home/xangma/pycbc-torch-split-20260905/venv/bin/python'
queue = json.loads((ROOT / 'queue-status.json').read_text())
correctness = json.loads((ROOT / 'correctness/status.json').read_text())
triton = json.loads((ROOT / 'triton/manifest.json').read_text())
assert queue.get('finished') and correctness.get('finished') and triton.get('finished_utc')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

sources = []
for preparation in json.loads((ROOT / 'preparation.json').read_text()):
    root = ROOT / preparation['name']
    revision = command(['git', 'rev-parse', 'HEAD'], root)
    status = command(['git', 'status', '--porcelain'], root)
    binaries = {str(p.relative_to(root)): digest(p) for p in (root / 'pycbc').rglob('*.so')}
    assert revision == preparation['sha'] and not status
    assert binaries == preparation['binaries'], 'Native binaries changed during the campaign'
    sources.append(dict(name=preparation['name'], revision=revision, status=status,
                        tree=command(['git', 'rev-parse', 'HEAD^{tree}'], root),
                        native_sha256=binaries, native_unchanged=True))
correctness_sources = []
for prepared in json.loads((ROOT / 'correctness-preparation.json').read_text()):
    root = Path(prepared['cwd'])
    assert root == ROOT / 'correctness-sources' / prepared['name']
    assert command(['git', 'rev-parse', 'HEAD'], root) == prepared['revision']
    assert command(['git', 'rev-parse', 'HEAD^'], root) == prepared['base_revision']
    assert not command(['git', 'status', '--porcelain'], root)
    assert command(['git', 'diff', '--name-only', prepared['base_revision'], 'HEAD'], root) == 'pycbc/filter/matchedfilter.py'
    for path, checksum in {**prepared['file_sha256'], **prepared['native_sha256'],
                           **prepared['generated_metadata_sha256']}.items():
        assert digest(root / path) == checksum
    patch = subprocess.check_output(['git', 'diff', '--binary', '--full-index', prepared['base_revision'], 'HEAD'], cwd=root)
    assert hashlib.sha256(patch).hexdigest() == prepared['runtime_patch_sha256']
    assert patch == (ROOT / 'correctness' / (prepared['name'] + '-runtime.patch')).read_bytes()
    assert digest(ROOT / 'correctness' / (prepared['name'] + '-matchedfilter.py')) == prepared['file_sha256']['pycbc/filter/matchedfilter.py']
    (ROOT / 'correctness' / (prepared['name'] + '-version.py')).write_bytes((root / 'pycbc/version.py').read_bytes())
    correctness_sources.append(dict(**prepared, status='', verified_unchanged=True))
final = dict(finished_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
             correctness_sources=correctness_sources,
             nvidia_smi=command(['nvidia-smi']), uname=command(['uname', '-a']),
             lscpu=command(['lscpu']), memory=command(['free', '-b']),
             packages=json.loads(command([PY, '-m', 'pip', 'list', '--format=json'])),
             queue_passed=queue.get('passed'), correctness_passed=correctness.get('passed'))
(ROOT / 'finalization.json').write_text(json.dumps(final, indent=2) + '\n')
files = []
for name in ('live-support', 'wave-support', 'triton-original', 'logs', 'correctness', 'live', 'probes'):
    files.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and not p.is_symlink()
                 and '__pycache__' not in p.parts and p.suffix != '.pyc')
for name in ('triton', 'waveform'):
    files.extend(p for p in (ROOT / name).iterdir() if p.is_file() and not p.is_symlink())
files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.json', '.py', '.sh')
             and p.name != 'sealed-manifest.json')
files = sorted(set(files))
inventory = {str(p.relative_to(ROOT)): dict(size=p.stat().st_size, sha256=digest(p)) for p in files}
sealed = ROOT / 'sealed-manifest.json'
sealed.write_text(json.dumps(inventory, indent=2) + '\n')
with tarfile.open(ROOT / 'sealed-evidence.tar.gz', 'w:gz') as archive:
    for path in files + [sealed]:
        archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
print(json.dumps(dict(files=len(files), bytes=sum(row['size'] for row in inventory.values()),
                      archive_sha256=digest(ROOT / 'sealed-evidence.tar.gz'),
                      queue_passed=queue.get('passed'), correctness_passed=correctness.get('passed'))))
