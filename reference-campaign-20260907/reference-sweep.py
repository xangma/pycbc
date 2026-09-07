#!/usr/bin/env python3
"""Prepare a unique low-mass bank and tune the untouched single-core reference."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906')
PREVIOUS = Path('/home/xangma/pycbc-torch-maintainer-benchmark-20260906')
ORIGINAL = PREVIOUS / 'original'
FINAL = OLD / 'source-v6'
state = dict(state='running', pid=os.getpid(), pgid=os.getpgrp(),
             started=time.time(), completed=[])


def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(name, command, env):
    state.update(current=name, command=command)
    save('status.json', state)
    with (ROOT / (name + '.log')).open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=900)
    state['completed'].append(name)
    save('status.json', state)


def main():
    config = json.loads((PREVIOUS / 'config.json').read_text())
    config['description'] = ('384 unique compressed BNS/NSBH templates, '
                             '256/128 split; deterministic performance set.')
    save('config.json', config)
    for name, origin in [('run-case.py', PREVIOUS),
                         ('compare-triggers.py', OLD),
                         ('qualify-inspiral.py', OLD)]:
        shutil.copy2(origin / name, ROOT / name)
    for source, expected in [(ORIGINAL, '40e94792b3edf59f39b18b65102b28a4f74433a7'),
                             (FINAL, 'a4d77a6d1863c0515e8dace64c5609b63d40b51e')]:
        assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                       text=True).strip() == expected
        assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'])
    shutil.copy2(PREVIOUS / 'source.json', ROOT / 'source.json')
    env = dict(os.environ, **config['environment'], PYTHONPATH=str(ORIGINAL),
               PYTHONDONTWRITEBYTECODE='1')
    save('protocol.json', dict(
        source_order='Tune and profile original CPU before running Torch.',
        sources={'original': str(ORIGINAL), 'final': str(FINAL)},
        threads=config['environment'], core=config['core'],
        bank_templates=384, bns_templates=256, nsbh_templates=128,
        valid_detector_seconds=1904, end_pad_seconds=16,
        lengths_seconds=[256, 512, 1024], start_pads_seconds=[96, 112],
        repetitions=3, ordering='Cyclic rotation of six candidates each repeat.',
        selection='Lowest full-process median; if within 3% choose larger start padding, then smaller FFT.',
        limits='Finite candidate sweep, no global optimality claim; fixed end pad conservatively covers the 16-s inverse-spectrum kernel.',
        science='Qualification must confirm compression, complete work and intervals. Compare final Torch against original at identical settings and retain failures.',
        profiles='Separate cProfile and native perf, both full-process and filtering-window attribution. Never use profiled runs for timing medians.',
        expected_profile='FFTs expected to dominate filtering. Report measured shares; do not force the proposed three/four kernels near 10%.'))
    execute('prepare-bank', [sys.executable, str(ROOT / 'prepare-bank.py'),
                            '--output-dir', str(ROOT / 'inputs')], env)
    metadata = json.loads((ROOT / 'inputs/bank-metadata.json').read_text())
    assert len(metadata['templates']) == 384
    assert metadata['longest']['duration_seconds'] + 16 < 96
    command = ['taskset', '-c', str(config['core']), sys.executable,
               str(ORIGINAL / 'bin/pycbc_compress_bank'), '--verbose',
               '--bank-file', str(ROOT / 'inputs/bank.hdf'),
               '--output', str(ROOT / 'inputs/bank-compressed.hdf'),
               '--sample-rate', '4096', '--segment-length', '256',
               '--compression-algorithm', 'spa', '--interpolation', 'inline_linear',
               '--precision', 'single', '--tolerance', '0.00001', '--nprocesses', '1',
               '--psd-model', 'aLIGOZeroDetHighPower', '--low-frequency-cutoff', '30',
               '--approximant', 'IMRPhenomD']
    before = time.perf_counter()
    execute('compress-bank', command, env)
    import h5py
    with h5py.File(ROOT / 'inputs/bank-compressed.hdf') as bank:
        hashes = [str(int(v)) for v in bank['template_hash'][:]]
        assert len(set(hashes)) == len(hashes) == 384
        assert set(bank['compressed_waveforms']) == set(hashes)
        assert all(set(bank['compressed_waveforms'][h]) ==
                   {'sample_points', 'amplitude', 'phase'} for h in hashes)
    save('compression.json', dict(command=command, elapsed=time.perf_counter()-before,
                                 source=str(ORIGINAL), sha256=sha(ROOT / 'inputs/bank-compressed.hdf'),
                                 template_count=384, complete=True))
    candidates = [(length, pad) for length in (256, 512, 1024) for pad in (96, 112)]
    plan = []
    for repeat in range(3):
        for length, pad in candidates[repeat:]+candidates[:repeat]:
            plan.append(dict(case=f'original-l{length}-p{pad}-r{repeat+1}',
                             length=length, pad=pad))
    save('sweep-plan.json', plan)
    for item in plan:
        command = [sys.executable, str(ROOT / 'run-case.py'), '--config', str(ROOT / 'config.json'),
                   '--case', item['case'], '--source', str(ORIGINAL), '--scheme', 'cpu:1',
                   '--bank', str(ROOT / 'inputs/bank-compressed.hdf'),
                   '--segment-length', str(item['length']), '--start-pad', str(item['pad'])]
        execute(item['case'], command, env)
    state.update(state='complete', finished=time.time())
    save('status.json', state)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished=time.time())
        save('status.json', state)
        raise
