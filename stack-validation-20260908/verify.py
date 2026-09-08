"""Verify this exact validation-publication tree without executing evidence."""
from pathlib import Path
import hashlib
if not __debug__:
    raise RuntimeError('Run verification without -O or PYTHONOPTIMIZE')
root=Path(__file__).resolve().parent
expected={}
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest,path=line.split('  ',1)
    if path.startswith('/') or any(p in ('','.','..') for p in path.split('/')) or path in expected:
        raise ValueError('Invalid checksum path')
    expected[path]=digest
actual={}
for path in root.rglob('*'):
    if path.is_symlink(): raise ValueError('Symlink in evidence')
    if path.is_file() and path != root/'SHA256SUMS':
        actual[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
assert actual==expected,'Validation evidence inventory mismatch'
print('Verified',len(expected),'files')
