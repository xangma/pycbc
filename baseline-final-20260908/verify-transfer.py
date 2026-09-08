"""Compare transferred raw acquisition bytes to a read-only remote inventory."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
probe = '''import hashlib,json,pathlib
p=pathlib.Path('/home/xangma/pycbc-torch-baseline-final-20260908')
assert json.loads((p/'timing-status.json').read_text())['state']=='complete'
skip={'original','proposed','repo','proposal.bundle'}
files={}
for item in p.rglob('*'):
 rel=item.relative_to(p)
 if rel.parts[0] in skip or '__pycache__' in rel.parts or not item.is_file(): continue
 files[rel.as_posix()]=hashlib.sha256(item.read_bytes()).hexdigest()
print(json.dumps(files,sort_keys=True))
'''
manifest = json.loads(subprocess.check_output(['ssh', 'len', 'python3 -'], input=probe, text=True))
for name, digest in manifest.items():
    assert hashlib.sha256((root / 'acquisition' / name).read_bytes()).hexdigest() == digest, name
receipt = dict(host='len', remote_root='/home/xangma/pycbc-torch-baseline-final-20260908',
               excluded_source_directories=['original', 'proposed', 'repo'],
               excluded_other=['proposal.bundle', '**/__pycache__/**'],
               raw_files_sha256=manifest, all_transferred_raw_bytes_match_remote=True)
(root / 'acquisition-transfer-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(f'PASS: {len(manifest)} raw acquisition files match the completed remote campaign')
