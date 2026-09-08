"""Verify the complete five-package publication tree; no scientific code execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    expected = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        if name in expected or name.startswith('/') or any(p in ('','..','.') for p in name.split('/')):
            raise ValueError('Invalid publication checksum path')
        expected[name] = sha
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in publication')
        if path.is_file() and path != root/'SHA256SUMS':
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Publication inventory mismatch')
    layout = json.loads((root/'publication-layout.json').read_text())
    results = []
    for item in layout['packages']:
        path = root/item['directory']/'verify.py'
        process = subprocess.run([sys.executable,'-I',str(path)],check=True,text=True,capture_output=True)
        results.append(json.loads(process.stdout))
    print(json.dumps(dict(state='pass', physical_files=len(actual)+1, packages=results,
                         reconstructed_files=sum(r['reconstructed_files'] for r in results),
                         original_archives=sum(r['original_archives'] for r in results),
                         science_executed=False, remote_access=False),indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
