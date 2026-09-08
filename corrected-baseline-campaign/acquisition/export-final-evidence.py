"""Freeze completed acquisition evidence without source checkouts or binaries."""
import hashlib
import json
from pathlib import Path
import tarfile

C = Path(__file__).resolve().parent
assert json.loads((C / 'timing-status.json').read_text())['state'] == 'complete'
assert (C / 'torch-environment-diagnostic.json').is_file()
archive = C / 'final-evidence.tar.gz'
assert not archive.exists()
files = sorted(p for p in C.iterdir() if p.is_file() and p.suffix in {'.py', '.json', '.log'})
files += sorted(p for folder in ('runs', 'comparisons') for p in (C / folder).rglob('*') if p.is_file())
assert len([p for p in (C / 'runs').iterdir() if p.is_dir()]) == 20
hashes = {str(p.relative_to(C)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest = C / 'final-evidence-sha256.json'
manifest.write_text(json.dumps(hashes, indent=2) + '\n')
with tarfile.open(archive, 'w:gz') as output:
    for path in [*files, manifest]:
        output.add(path, arcname=str(path.relative_to(C)), recursive=False)
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=len(files))))
