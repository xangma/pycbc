#!/usr/bin/env python3
"""Run the bounded diagnostic with frozen source, inputs and environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    source = root / 'source-v2'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == '968bcd558117262af0d603710b054174659adb51'
    assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True) == ''
    for name in ('cpu', 'cuda'):
        assert json.loads((root / f'chisq-input-capture-{name}/status.json').read_text())['state'] != 'running'
    for path in root.glob('*.status.json'):
        assert json.loads(path.read_text()).get('state') != 'running', path
    cfg = json.loads((root / 'config.json').read_text())
    inputs = [Path(__file__).resolve(), root / 'diagnose-snr-psd.py', root / 'config.json', root / 'source-v2.json', root / 'inputs/bank-compressed-1e5.hdf', root / 'chisq-input-capture-cpu/conditioned-strain.npy', root / 'chisq-input-capture-cpu/capture-3-corr.npy', root / 'chisq-input-capture-cpu/status.json']
    for name in ('cpu', 'torch-cpu', 'torch-cuda'):
        run = root / 'runs' / f'qual-final-{name}-l512'
        qual = json.loads((run / 'qualification.json').read_text())
        inputs += [run / 'qualification.json', run / 'receipt.json', run / qual['observations']['psd_arrays'][0]['relative_path']]
    inputs += [source / path for path in ('pycbc/psd/estimate.py', 'pycbc/filter/matchedfilter.py', 'pycbc/waveform/decompress_torch.py', 'pycbc/scheme.py')]
    assert all(p.is_file() for p in inputs), [str(p) for p in inputs if not p.is_file()]
    hashes = {str(p): digest(p) for p in inputs}
    out = root / 'snr-psd-diagnostic-v1'
    assert not out.exists()
    status = root / 'snr-psd-diagnostic-v1.status.json'
    assert not status.exists()
    command = ['taskset', '-c', str(cfg['core']), sys.executable, '-u', str(root / 'diagnose-snr-psd.py'), '--root', str(root), '--output-dir', str(out), '--template-hash', '4366715446675002219', '--strain', str(root / 'chisq-input-capture-cpu/conditioned-strain.npy'), '--correlation', str(root / 'chisq-input-capture-cpu/capture-3-corr.npy'), '--indices', '1884341']
    env = dict(os.environ, **cfg['environment'], PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
    record = dict(state='running', pid=os.getpid(), command=command, cwd=str(root), hostname=os.uname().nodename, input_sha256=hashes, started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status)

    save()
    started = time.perf_counter()
    code = subprocess.call(command, cwd=root, env=env)
    record.update(state='complete' if code == 0 else 'failed', returncode=code, elapsed_wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat(), input_sha256_after={str(p): digest(p) for p in inputs}, source_status_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
    if record['input_sha256_after'] != hashes or record['source_status_after']:
        record['state'] = 'invalid-input-mutation'
    report = out / 'report.json'
    if report.is_file():
        record['report_sha256'] = digest(report)
    save()
    return 0 if record['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
