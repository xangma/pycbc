#!/usr/bin/env python3
"""Compress the frozen workload once, preserving preparation provenance."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parent
metadata = json.loads((root / 'inputs/bank-metadata.json').read_text())
config = json.loads((root / 'config.json').read_text())
source = Path(os.environ['PYTHONPATH'])
output = root / 'inputs/bank-compressed-1e5.hdf'
pilot = root / 'inputs/bank-pilot-compressed-1e5.hdf'
assert not output.exists() and not pilot.exists()
assert metadata['longest']['duration_seconds'] + 16 <= 112
command = ['/usr/bin/time', '-v', '-o', str(root / 'compression-1e5-time.txt'),
           'taskset', '-c', str(config['core']), sys.executable,
           str(source / 'bin/pycbc_compress_bank'), '--verbose',
           '--bank-file', str(root / 'inputs/bank.hdf'), '--output', str(output),
           '--sample-rate', '4096', '--segment-length',
           str(metadata['compression']['segment_length_seconds']),
           '--compression-algorithm', 'spa', '--interpolation', 'inline_linear',
           '--precision', 'single', '--tolerance', '0.00001', '--nprocesses', '1',
           '--psd-model', 'aLIGOZeroDetHighPower', '--low-frequency-cutoff', '30',
           '--approximant', 'IMRPhenomD']
inputs = [root / 'inputs/bank.hdf', root / 'inputs/bank-metadata.json',
          source / 'bin/pycbc_compress_bank', Path(__file__)]
record = dict(state='running', pid=os.getpid(), command=command, cwd=str(root),
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              environment={k: os.environ.get(k) for k in config['environment']},
              pythonpath=str(source), input_sha256={str(p): sha(p) for p in inputs})
receipt = root / 'compression-1e5.json'


def save():
    receipt.write_text(json.dumps(record, indent=2) + '\n')


save()
started = time.perf_counter()
with open(root / 'compression-1e5.log', 'w') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=root)
    record['child_pid'] = child.pid
    save()
    code = child.wait()
record.update(returncode=code, elapsed_wall_seconds=time.perf_counter() - started,
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              state='failed' if code else 'compressed')
save()
if code:
    sys.exit(code)
import h5py
from pycbc.waveform.bank import TemplateBank

with h5py.File(output, 'r') as stream:
    hashes = [str(int(value)) for value in stream['template_hash'][:]]
    assert len(hashes) == 96
    assert set(stream['compressed_waveforms']) == set(hashes)
    assert all(set(stream['compressed_waveforms'][key]) ==
               {'sample_points', 'amplitude', 'phase'} for key in hashes)
    bank = TemplateBank(file_handler=stream, approximant='IMRPhenomD')
    bank.table = bank.table[metadata['pilot']['row_indices_in_main_bank']]
    with h5py.File(pilot, 'x') as target:
        bank.write_to_hdf(str(pilot), file_handler=target, write_compressed_waveforms=True)
with h5py.File(pilot, 'r') as stream:
    assert [int(value) for value in stream['template_hash'][:]] == metadata['pilot']['template_hashes']
    assert set(stream['compressed_waveforms']) == {str(value) for value in metadata['pilot']['template_hashes']}
record['input_sha256_after'] = {str(p): sha(p) for p in inputs}
assert record['input_sha256_after'] == record['input_sha256']
record.update(state='complete', output_sha256={str(p): sha(p) for p in [output, pilot]},
              compressed_records=96, pilot_compressed_records=8)
save()
print(json.dumps(record))
