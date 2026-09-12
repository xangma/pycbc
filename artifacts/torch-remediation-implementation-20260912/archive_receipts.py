#!/usr/bin/env python3
"""Archive a finished acquisition with relative names and per-file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
paths = set()
for pattern in ('logs/final*', 'final-cli-*', 'final-live', 'cli-fixture',
                'logs/flake8-f401-final.log', 'logs/flake8-f401-baseline.log',
                'logs/flake8-*-changed*.log', 'logs/lint-comparison.json',
                'logs/final-environment.json', 'final-power-chisq-before-after',
                'run-final-*.sh', 'compare_offline_campaigns.py',
                'offline_pycbc_contract.json', 'profile_candidate_pipeline.py',
                'make_cli_fixture.py', 'live_cli_smoke.py', 'summarize_results.py',
                'archive_receipts.py', 'audit_final_environment.py',
                'qualified/artifacts/torch-remediation-20260912/compare_native_taylorf2_generation.py'):
    for path in root.glob(pattern):
        paths.update(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else paths.add(path)
manifest = {}
for path in sorted(paths):
    data = path.read_bytes()
    manifest[str(path.relative_to(root))] = {
        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
args.output.mkdir(parents=True, exist_ok=True)
archive = args.output / 'raw-receipts.tar.gz'
with tarfile.open(archive, 'w:gz') as output:
    for path in sorted(paths):
        output.add(path, arcname=str(path.relative_to(root)), recursive=False)
record = {'source_root': str(root), 'archive': archive.name,
          'archive_bytes': archive.stat().st_size,
          'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'file_count': len(manifest), 'files': manifest}
(args.output / 'raw-receipts-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key != 'files'}))
