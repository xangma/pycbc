"""Read-only post-campaign audit; run on len with python3 on stdin."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/xangma/pycbc-torch-current-batch-sweep-20260907-r4')
REVISION = '9578a710479b924e882857c4dffab6ed372a634b'


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


status = read('batch-status.json')
assert status['state'] == status['phase'] == 'complete'
assert status['current'] is None and status['child_pid'] is None
assert status['qualifications_passed'] == 36 and status['timing_workers'] == 54
assert status['source_unchanged'] is True
labels = status['completed']
assert len(labels) == len(set(labels)) == 102
stages = [read('stages/' + label + '.json') for label in labels]
assert all(s['state'] == 'complete' and s['returncode'] == 0 for s in stages)
tracked_pids = {status['pid']} | {s['pid'] for s in stages}
tracked_pgids = {status['pgid']} | {s['pgid'] for s in stages}
processes = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,comm='], text=True)
present = []
for line in processes.splitlines():
    pid, pgid, command = line.strip().split(None, 2)
    if int(pid) in tracked_pids or int(pgid) in tracked_pgids:
        present.append(dict(pid=int(pid), pgid=int(pgid), command=command))
source = ROOT / 'source'
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain=v1'], text=True)
assert head == REVISION and dirty == ''
helpers = read('stage-manifest.json')
assert all(digest(ROOT / name) == expected for name, expected in helpers.items())
extensions = read('native-provenance.json')['extensions']
assert len(extensions) == 11
assert all(digest(source / item['relative_path']) == item['sha256'] for item in extensions)
receipt = dict(
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    host='len', cwd=str(ROOT), source_revision=head, source_status=dirty,
    stage_manifest_sha256=digest(ROOT / 'stage-manifest.json'),
    batch_status_sha256=digest(ROOT / 'batch-status.json'),
    helper_files_verified=len(helpers), native_binaries_verified=len(extensions),
    completed_workers=len(stages), controller_pid=status['pid'],
    recorded_pids=sorted(tracked_pids), recorded_pgids=sorted(tracked_pgids),
    present_processes=present, all_recorded_processes_absent=not present,
    scope='Read-only terminal/process/source/native integrity audit; no scientific rerun',
)
print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
assert not present, 'Recorded process IDs/groups still present; inspect possible PID reuse'
