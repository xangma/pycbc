"""Read-only post-acquisition inspection of Torch metadata and imported bytes."""
import base64
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import socket
import sys

root = Path(__file__).resolve().parent
status = json.loads((root / 'timing-status.json').read_text())
assert status['state'] == 'complete'
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


all_distributions = list(metadata.distributions())
collapsed = {d.metadata['Name']: d.version for d in all_distributions}
rows = []
for distribution in all_distributions:
    if distribution.metadata['Name'].lower().replace('_', '-') != 'torch':
        continue
    path = Path(distribution._path)
    matches = []
    for name, recorded, size in csv.reader(io.StringIO(distribution.read_text('RECORD'))):
        if name not in {'torch/__init__.py', 'torch/version.py'}:
            continue
        file = Path(distribution.locate_file(name))
        digest = sha(file)
        calculated = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
        matches.append(dict(path=str(file), relative_path=name, recorded=recorded,
                            observed_sha256=digest, matches_record=calculated == recorded))
    rows.append(dict(name=distribution.metadata['Name'], version=distribution.version,
                     metadata_path=str(path),
                     metadata_sha256={name: sha(path / name) for name in ('METADATA', 'WHEEL', 'RECORD')},
                     selected_file_checks=matches))
result = dict(
    schema_version=1, kind='post_acquisition_read_only_dependency_diagnostic',
    observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), hostname=socket.gethostname(),
    executable=sys.executable, timing_status_sha256=sha(root / 'timing-status.json'),
    dependency_manifest_sha256=sha(root / 'dependencies.json'),
    imported_torch_version=torch.__version__, imported_cuda_version=torch.version.cuda,
    imported_files={str(path): sha(path) for path in
                    (torch.__file__, torch.version.__file__, torch._C.__file__)},
    metadata_first_match_version=metadata.version('torch'),
    metadata_last_wins_torch_version=collapsed['torch'], distributions=rows,
    scope='Observed after all recorded processes finished. This diagnoses metadata selection; '
          'it is not a retrospective hash of Torch binaries loaded by benchmark workers.')
print(json.dumps(result, indent=2))
