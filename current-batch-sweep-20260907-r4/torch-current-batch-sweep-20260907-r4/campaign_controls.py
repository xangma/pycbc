#!/usr/bin/env python3
"""Shared-host workload convergence; run only after copying to its new root.

Run with the reference PyCBC venv: python convergence-campaign.py --shared-host
ROOT is this file's parent. Requires config.json, prepare-scaling-bank.py,
run-case.py, qualify-inspiral.py and compare-triggers.py alongside this file.
Fresh outputs only; a failed/interrupted campaign is never silently resumed.
Stop with kill -TERM <campaign PID>; each stage owns a separate process group.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/home/xangma/pycbc-torch-maintainer-benchmark-20260906/original')
FINAL = Path('/home/xangma/pycbc-torch-inspiral-reference-20260906/source-v6')
ORIGINAL_COMMIT = '40e94792b3edf59f39b18b65102b28a4f74433a7'
FINAL_COMMIT = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
DEPENDENCY = Path('/home/xangma/pycbc-torch-reference-protocol-20260907/comparison-status.json')
SIZES = (384, 768, 1536, 3072, 6144)
VALID_SECONDS = 1904
ROUTES = [('original-cpu', ORIGINAL, 'cpu:1', ORIGINAL_COMMIT),
          ('corrected-cpu', FINAL, 'cpu:1', FINAL_COMMIT),
          ('torch-cpu', FINAL, 'torch:cpu:1', FINAL_COMMIT),
          ('torch-cuda', FINAL, 'torch:cuda:0', FINAL_COMMIT)]
POLICY = {
    'scope': 'shared-host diagnostic; no exclusive reservation',
    'sizes': list(SIZES), 'repeats': 3, 'valid_detector_seconds': VALID_SECONDS,
    'throughput': 'templates * valid_detector_seconds / executable_full_wall_seconds. CPU convergence uses one physical core; CUDA results additionally use the GPU.',
    'selection': 'All boundaries plus SHA256-ranked interior hashes per population; BNS:NSBH = 2:1.',
    'criterion': 'At the largest size, BOTH last successive doubling changes in median full-wall throughput have absolute relative change < 0.05; ALL three repeats at EACH of those last three sizes have (setup + full_wall - internal_runtime) / full_wall < 0.10.',
    'selection_on_pass': 6144,
    'selection_on_fail': 'finite-workload only; do not run final matched campaign',
    'matched': 'Three fresh repeats of all four routes; rotate route order each repeat. No reuse of convergence samples.',
    'timing': 'Unprofiled run-case timing only. Qualification times excluded.',
    'geometry': {'segment_length': 512, 'start_pad': 112, 'end_pad': 16},
    'stage_timeout_seconds': 3600, 'dependency_wait_seconds': 14400,
}


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def signed_hash(value):
    value = int(value)
    if not -(1 << 63) <= value < (1 << 64):
        raise ValueError('Template hash outside 64-bit range')
    return value - (1 << 64) if value >= (1 << 63) else value


def nested_indices(metadata, sizes=SIZES):
    """Rank by hash rather than source row order, retaining every boundary."""
    rows = metadata['templates']
    hashes = [signed_hash(row['template_hash']) for row in rows]
    if len(set(hashes)) != len(rows):
        raise ValueError('Duplicate template hash')
    if sorted(row['row_index'] for row in rows) != list(range(len(rows))):
        raise ValueError('Row indices must be unique and contiguous')
    if any(row['population'] not in ('BNS', 'NSBH') or
           row['selection'] not in ('boundary', 'interior-stratum') for row in rows):
        raise ValueError('Unknown population or selection')

    def rank(row):
        key = 'pycbc-convergence-v1:' + str(signed_hash(row['template_hash']))
        return hashlib.sha256(key.encode('ascii')).hexdigest()

    result = {}
    for size in sizes:
        if size <= 0 or size % 3:
            raise ValueError('Size must be a positive multiple of three')
        selected = []
        for population, count in [('BNS', size * 2 // 3), ('NSBH', size // 3)]:
            group = [r for r in rows if r['population'] == population]
            boundaries = sorted((r for r in group if r['selection'] == 'boundary'), key=rank)
            interiors = sorted((r for r in group if r['selection'] != 'boundary'), key=rank)
            if not len(boundaries) <= count <= len(group):
                raise ValueError('Insufficient rows or too many boundaries')
            selected.extend(r['row_index'] for r in boundaries + interiors[:count - len(boundaries)])
        result[size] = sorted(selected)
    previous = set()
    for size in sorted(result):
        current = set(result[size])
        if not previous <= current or len(current) != size:
            raise ValueError('Invalid nested membership')
        previous = current
    return result


def subset_bank(source, destination, indices, expected_hashes):
    """Slice row datasets recursively; retain attributes and selected waveforms."""
    import h5py
    import numpy as np

    with h5py.File(source, 'r') as src, h5py.File(destination, 'x') as dst:
        hashes = [signed_hash(h) for h in src['template_hash'][:]]
        if hashes != list(expected_hashes) or len(set(hashes)) != len(hashes):
            raise ValueError('Compressed bank hashes differ from unique metadata rows')
        selected = {str(hashes[i]) for i in indices}
        if 'compressed_waveforms' not in src:
            raise ValueError('Bank lacks compressed waveforms')
        if not selected <= set(src['compressed_waveforms']):
            raise ValueError('Missing compressed hash group')

        def copy_group(old, new):
            for key, value in old.attrs.items():
                new.attrs[key] = value
            for key, obj in old.items():
                if obj.name == '/compressed_waveforms':
                    group = new.create_group(key)
                    for attr, value in obj.attrs.items():
                        group.attrs[attr] = value
                    for template_hash in sorted(selected):
                        obj.copy(template_hash, group)
                elif isinstance(obj, h5py.Group):
                    copy_group(obj, new.create_group(key))
                elif obj.ndim and obj.shape[0] == len(hashes):
                    # Preserve the complete row (including trailing dimensions),
                    # dtype and attributes; storage layout may be recomputed.
                    data = obj[np.asarray(indices, dtype=np.int64)]
                    new.create_dataset(key, data=data, dtype=obj.dtype)
                    for attr, value in obj.attrs.items():
                        new[key].attrs[attr] = value
                else:
                    old.copy(key, new)
        copy_group(src, dst)
        if [signed_hash(h) for h in dst['template_hash'][:]] != [hashes[i] for i in indices]:
            raise ValueError('Subset readback mismatch')


def make_subsets(root):
    source = root / 'inputs/bank-compressed.hdf'
    metadata = read(root / 'inputs/bank-metadata.json')
    rows = sorted(metadata['templates'], key=lambda r: r['row_index'])
    selections = nested_indices(metadata)
    hashes = [signed_hash(r['template_hash']) for r in rows]
    result = {'selection': POLICY['selection'], 'source_sha256': digest(source), 'banks': {}}
    for size, indices in selections.items():
        path = root / f'inputs/bank-{size}-compressed.hdf'
        subset_bank(source, path, indices, hashes)
        result['banks'][str(size)] = {
            'path': str(path), 'sha256': digest(path), 'template_count': size,
            'row_indices_in_full_bank': indices,
            'template_hashes': [hashes[i] for i in indices],
            'population_counts': {p: sum(rows[i]['population'] == p for i in indices)
                                  for p in ('BNS', 'NSBH')},
            'boundary_indices': [i for i in indices if rows[i]['selection'] == 'boundary'],
        }
    if digest(source) != result['source_sha256']:
        raise ValueError('Full compressed bank mutated during selection')
    save(root / 'nested-banks.json', result)
    return result


def dependency_ready(value):
    state = value.get('state')
    if state in ('failed', 'invalid-input-mutation', 'cancelled'):
        raise ValueError('Reference dependency failed')
    if state != 'complete':
        return False
    codes = value.get('comparison_returncodes', {})
    required = {'corrected-trigger-comparison', 'original-trigger-comparison'}
    required.update(name + '-' + kind + '-parity'
                    for name, *_ in ROUTES for kind in ('repeat', 'profile'))
    if not required <= codes.keys():
        raise ValueError('Completed dependency lacks required comparison outcomes')
    if codes['original-trigger-comparison'] not in (0, 1, 2):
        raise ValueError('Original comparison did not finish normally')
    if any(code != 0 for name, code in codes.items() if name != 'original-trigger-comparison'):
        raise ValueError('Corrected/reference repeat or profile comparison failed')
    return True


def source_record(path, expected):
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
    if commit != expected or status:
        raise ValueError(f'Source must be clean at {expected}: {path}')
    return {'path': str(path), 'commit': commit, 'status': status}


def qualify(path, size, bank_hash):
    q = read(path)
    checks = q.get('checks', {})
    required = {'no_generation_fallback', 'bank_0_all_templates_once',
                'bank_0_compression_enabled', 'scalar_ifft_executed_for_every_pair'}
    if q.get('status') != 'success' or not required <= checks.keys() or not all(checks.values()):
        raise ValueError(f'Qualification checks failed: {path}')
    observations = q['observations']
    banks = observations['banks']
    geometry = observations['segment_geometry']
    if len(banks) != 1 or len(geometry) != 1:
        raise ValueError('Qualification lacks one bank and geometry')
    bank, geometry = banks[0], geometry[0]
    segments = len(geometry['segments'])
    if (bank['selected_template_count'] != size or
            bank['expected_filter_calls'] != size * segments or
            bank['file']['sha256'] != bank_hash or
            geometry['unique_analyzed_seconds'] != VALID_SECONDS or
            geometry['gap_samples'] != 0 or geometry['overlap_samples'] != 0):
        raise ValueError('Qualification work, bank or intervals differ')
    return {'path': str(path), 'sha256': digest(path), 'checks': checks,
            'templates': size, 'segments': segments, 'valid_seconds': VALID_SECONDS}


def timing_metrics(receipt, internal, setup_fraction, size):
    wall = float(receipt['elapsed_wall_seconds'])
    if not all(math.isfinite(x) for x in (wall, internal, setup_fraction)):
        raise ValueError('Nonfinite timing value')
    if not 0 < internal <= wall or not 0 <= setup_fraction <= 1:
        raise ValueError('Internal/setup timing inconsistent with full wall time')
    setup = internal * setup_fraction
    return {'templates': size, 'full_wall_seconds': wall, 'internal_seconds': internal,
            'setup_seconds': setup, 'outside_internal_seconds': wall - internal,
            'overhead_fraction': (setup + wall - internal) / wall,
            'template_seconds_per_wall_second': size * VALID_SECONDS / wall}


def timing(path, size, expected_commit, expected_bank_hash, qualification):
    import h5py

    receipt = read(path / 'receipt.json')
    if (receipt.get('state') != 'complete' or receipt.get('returncode') != 0 or
            receipt.get('mode') != 'timing' or
            receipt['source_info']['commit'] != expected_commit or
            receipt['source_info']['status'] or receipt['source_status_after'] or
            receipt['input_sha256'] != receipt['input_sha256_after'] or
            expected_bank_hash not in receipt['input_sha256'].values()):
        raise ValueError(f'Invalid timing/source/input receipt: {path}')
    trigger_path = path / 'triggers.hdf'
    trigger_hash = digest(trigger_path)
    if receipt.get('trigger_sha256') != trigger_hash:
        raise ValueError('Trigger output SHA256 differs from its receipt')
    with h5py.File(trigger_path) as f:
        search = f['H1/search']
        starts, ends = search['start_time'][:], search['end_time'][:]
        if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(ends) or not len(starts):
            raise ValueError('Invalid trigger interval arrays')
        intervals = sorted((float(a), float(b)) for a, b in zip(starts, ends))
        if (any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a, b in intervals)
                or any(a < previous_end for (_, previous_end), (a, _) in zip(intervals, intervals[1:]))
                or len(set(intervals)) != len(intervals)):
            raise ValueError('Trigger intervals must be unique, nonempty and disjoint')
        valid = math.fsum(b - a for a, b in intervals)
        if valid != VALID_SECONDS or qualification['valid_seconds'] != valid:
            raise ValueError('Trigger interval differs from qualified work')

        def scalar(key):
            value = search[key][:]
            if value.size != 1 or not math.isfinite(float(value.flat[0])):
                raise ValueError(f'Invalid performance scalar: {key}')
            return float(value.flat[0])

        internal = scalar('run_time')
        templates = scalar('templates_per_core') * internal / valid
        segments = scalar('filter_rate_per_core') * internal
        if not math.isclose(templates, size, rel_tol=0, abs_tol=1e-6) or qualification['templates'] != size:
            raise ValueError('Timed template count differs from qualified bank')
        if (not math.isclose(segments, round(segments), rel_tol=0, abs_tol=1e-6)
                or round(segments) != qualification['segments']):
            raise ValueError('Timed segment count differs from qualification')
        metrics = timing_metrics(receipt, internal, scalar('setup_time_fraction'), size)
    if digest(trigger_path) != trigger_hash:
        raise ValueError('Trigger output mutated while reading performance')
    metrics.update(case=path.name, receipt=str(path / 'receipt.json'),
                   scheme=receipt['scheme'], trigger_sha256=trigger_hash,
                   valid_intervals=intervals, inferred_templates=templates,
                   inferred_segments=segments, qualification=qualification['path'],
                   receipt_sha256=digest(path / 'receipt.json'))
    return metrics


def convergence(samples, sizes=SIZES):
    summaries = []
    for size in sizes:
        rows = samples[size]
        if len(rows) != 3:
            raise ValueError('Three fresh timing repeats required per size')
        summaries.append({'templates': size, 'samples': rows,
                          'median_throughput': statistics.median(
                              r['template_seconds_per_wall_second'] for r in rows),
                          'max_overhead_fraction': max(r['overhead_fraction'] for r in rows)})
    changes = [abs(b['median_throughput'] / a['median_throughput'] - 1)
               for a, b in zip(summaries, summaries[1:])]
    passed = (len(sizes) >= 3 and all(b == a * 2 for a, b in zip(sizes[-3:], sizes[-2:]))
              and all(x < 0.05 for x in changes[-2:])
              and all(row['max_overhead_fraction'] < 0.10 for row in summaries[-3:]))
    return {'criterion': POLICY['criterion'], 'converged': passed,
            'selected_templates': sizes[-1] if passed else None,
            'classification': 'converged shared-host diagnostic' if passed else 'finite-workload; convergence not demonstrated',
            'successive_absolute_relative_changes': changes, 'sizes': summaries}


class Cancelled(Exception):
    pass


_stop_deferrals = 0
_pending_stop = None


@contextmanager
def defer_stop_signals():
    """Defer our handler's exception without passing blocked signals to children."""
    global _stop_deferrals, _pending_stop
    _stop_deferrals += 1
    try:
        yield
    finally:
        _stop_deferrals -= 1
        if _stop_deferrals == 0 and _pending_stop is not None:
            signum, _pending_stop = _pending_stop, None
            raise Cancelled(f'Received signal {signum}')


def clean_group(child):
    """Terminate only the stage's own group, even if its leader exited first."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(0.3)
    child.wait(timeout=10)


class Campaign:
    def __init__(self, root, timeout=3600):
        self.root, self.timeout = Path(root), timeout
        self.state = {'state': 'starting', 'pid': os.getpid(), 'pgid': os.getpgrp(),
                      'host': socket.gethostname(), 'cwd': str(root), 'started': time.time(),
                      'command': sys.argv, 'scope': POLICY['scope'], 'completed': [],
                      'stop_command': f'kill -TERM {os.getpid()}', 'next_check_seconds': 5}
        self.stop = threading.Event()
        self.observer = None

    def update(self, **values):
        self.state.update(values)
        save(self.root / 'convergence-status.json', self.state)

    def observe(self):
        with (self.root / 'convergence-host-samples.jsonl').open('x') as log:
            while not self.stop.is_set():
                sample = {'utc': time.time(), 'monotonic_ns': time.monotonic_ns(),
                          'phase': self.state.get('phase'), 'case': self.state.get('current'),
                          'child_pid': self.state.get('child_pid')}
                for name in ('stat', 'loadavg', 'meminfo'):
                    try:
                        sample['proc_' + name] = Path('/proc', name).read_text()
                    except OSError as exc:
                        sample['proc_' + name + '_error'] = str(exc)
                try:
                    sample['processes'] = subprocess.check_output(
                        ['ps', '-eo', 'pid,ppid,psr,nlwp,pcpu,pmem,comm', '--sort=-pcpu'],
                        text=True, stderr=subprocess.DEVNULL, timeout=3)
                except (OSError, subprocess.SubprocessError) as exc:
                    sample['processes_error'] = str(exc)
                log.write(json.dumps(sample) + '\n')
                log.flush()
                self.stop.wait(5)

    def stage(self, name, command, env=None, allowed=(0,)):
        log_path = self.root / 'stages' / (name + '.log')
        err_path = self.root / 'stages' / (name + '.stderr')
        receipt_path = self.root / 'stages' / (name + '.json')
        record = {'name': name, 'command': list(map(str, command)), 'cwd': str(self.root),
                  'host': socket.gethostname(), 'started': time.time(),
                  'log': str(log_path), 'stderr': str(err_path), 'state': 'starting',
                  'timeout_seconds': self.timeout, 'environment_overrides': env or {}}
        self.update(state='running', current=name, command=record['command'],
                    log=str(log_path), stage_receipt=str(receipt_path), child_pid=None, child_pgid=None)
        child = None
        start = time.monotonic()
        try:
            with log_path.open('x') as output, err_path.open('x') as error:
                with defer_stop_signals():
                    child = subprocess.Popen(record['command'], cwd=self.root,
                                             env=dict(os.environ, **(env or {})),
                                             stdout=output, stderr=error, start_new_session=True)
                    record.update(state='running', pid=child.pid, pgid=child.pid)
                    save(receipt_path, record)
                    self.update(child_pid=child.pid, child_pgid=child.pid)
                code = child.wait(timeout=self.timeout)
            record['returncode'] = code
            if code not in allowed:
                raise RuntimeError(f'{name} exited {code}; see {err_path}')
            record['state'] = 'complete'
        except BaseException as exc:
            record.update(state='failed', error=repr(exc))
            raise
        finally:
            with defer_stop_signals():
                if child is not None:
                    clean_group(child)
                    record['returncode'] = child.returncode
                record.update(finished=time.time(), elapsed_wall_seconds=time.monotonic() - start)
                save(receipt_path, record)
                wrapper = self.root / 'runs' / name / 'receipt.json'
                if wrapper.exists():
                    value = read(wrapper)
                    if value.get('state') == 'running':
                        value.update(state='interrupted', campaign_stage_receipt=str(receipt_path),
                                     campaign_error=record.get('error', 'Wrapper exited without final receipt'))
                        save(wrapper, value)
                self.update(child_pid=None, child_pgid=None)
        self.state['completed'].append(name)
        self.update()
        return record

    def wait_dependency(self, path, timeout=14400):
        self.update(state='waiting', phase='dependency', dependency=str(path),
                    dependency_deadline=time.time() + timeout)
        deadline = time.monotonic() + timeout
        while True:
            try:
                value = read(path)
            except FileNotFoundError:
                value = {}
            if dependency_ready(value):
                save(self.root / 'dependency-receipt.json',
                     {'path': str(path), 'sha256': digest(path), 'observed': time.time(), 'status': value})
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Reference dependency did not succeed within four hours')
            time.sleep(min(5, max(0, deadline - time.monotonic())))

    def run_case(self, name, mode, route, size):
        _, source, scheme, commit = route
        source_record(source, commit)
        bank = self.root / f'inputs/bank-{size}-compressed.hdf'
        command = [sys.executable, self.root / 'run-case.py', '--config', self.root / 'config.json',
                   '--case', name, '--mode', mode, '--scheme', scheme, '--source', source,
                   '--bank', bank, '--segment-length', '512', '--start-pad', '112', '--end-pad', '16']
        self.stage(name, command)
        source_record(source, commit)
        folder = self.root / 'runs' / name
        receipt = read(folder / 'receipt.json')
        if (receipt.get('state') != 'complete' or receipt.get('returncode') != 0 or
                receipt['source_info']['commit'] != commit or receipt['source_info']['status'] or
                receipt['source_status_after'] or receipt['input_sha256'] != receipt['input_sha256_after']):
            raise ValueError(f'Invalid run receipt: {folder}')
        return folder

    def compare(self, name, baseline, candidates):
        self.stage(name, [sys.executable, self.root / 'compare-triggers.py', baseline, *candidates],
                   allowed=(0, 1, 2))
        result = read(self.root / 'stages' / (name + '.log'))
        save(self.root / (name + '.json'), result)
        if result.get('status') not in ('pass', 'fail', 'review') or 'error' in result:
            raise ValueError(f'Comparison did not produce a scientific outcome: {name}')
        return result['status']


def run(campaign):
    root = campaign.root
    campaign.wait_dependency(DEPENDENCY)
    cfg = read(root / 'config.json')
    if cfg['core'] != 8:
        raise ValueError('Frozen config must pin worker to CPU 8')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        if cfg['environment'].get(name) != '1':
            raise ValueError(f'Expected single thread: {name}')
    if cfg['environment'].get('MKL_DYNAMIC') != 'FALSE' or cfg['environment'].get('PYTHONHASHSEED') != '0':
        raise ValueError('Frozen thread/hash environment differs')
    provenance = {'original': source_record(ORIGINAL, ORIGINAL_COMMIT),
                  'final': source_record(FINAL, FINAL_COMMIT)}
    pins = [root / name for name in ('config.json', 'convergence-campaign.py',
                                    'prepare-scaling-bank.py', 'run-case.py',
                                    'qualify-inspiral.py', 'compare-triggers.py')]
    pins.extend(map(Path, cfg['input_files']))
    frozen = {str(path): digest(path) for path in pins}
    save(root / 'convergence-inputs.json', {'sources': provenance, 'sha256': frozen})
    env = dict(cfg['environment'], PYTHONPATH=str(ORIGINAL), PYTHONDONTWRITEBYTECODE='1')
    campaign.update(phase='prepare')
    campaign.stage('prepare-6144', ['taskset', '-c', '8', sys.executable,
                                  root / 'prepare-scaling-bank.py', '--output-dir', root / 'inputs'], env)
    command = ['taskset', '-c', '8-15', sys.executable, ORIGINAL / 'bin/pycbc_compress_bank',
               '--verbose', '--bank-file', root / 'inputs/bank.hdf',
               '--output', root / 'inputs/bank-compressed.hdf', '--sample-rate', '4096',
               '--segment-length', '256', '--compression-algorithm', 'spa',
               '--interpolation', 'inline_linear', '--precision', 'single',
               '--tolerance', '0.00001', '--nprocesses', '8', '--psd-model', 'aLIGOZeroDetHighPower',
               '--low-frequency-cutoff', '30', '--approximant', 'IMRPhenomD']
    bank_before = digest(root / 'inputs/bank.hdf')
    campaign.update(phase='compression')
    receipt = campaign.stage('compress-6144', command, env)
    source_record(ORIGINAL, ORIGINAL_COMMIT)
    if bank_before != digest(root / 'inputs/bank.hdf'):
        raise ValueError('Compression input mutated')
    save(root / 'compression.json', dict(receipt, source=provenance['original'],
                                       input_sha256=bank_before, complete=True, template_count=6144,
                                       sha256=digest(root / 'inputs/bank-compressed.hdf')))
    campaign.update(phase='nested-selection')
    campaign.stage('select-nested-banks', [sys.executable, root / 'convergence-campaign.py', '--select-only'])
    banks = read(root / 'nested-banks.json')['banks']
    qualifications, samples = {}, {size: [] for size in SIZES}
    campaign.update(phase='original-qualification')
    for size in SIZES:
        folder = campaign.run_case(f'qual-original-{size}', 'qualify', ROUTES[0], size)
        qualifications[str(size)] = qualify(folder / 'qualification.json', size, banks[str(size)]['sha256'])
        save(root / 'convergence-qualifications.json', qualifications)
    campaign.update(phase='original-timing')
    for repeat in range(1, 4):
        for size in SIZES:
            folder = campaign.run_case(f'convergence-original-{size}-r{repeat}', 'timing', ROUTES[0], size)
            samples[size].append(timing(folder, size, ORIGINAL_COMMIT, banks[str(size)]['sha256'],
                                        qualifications[str(size)]))
            save(root / 'convergence-samples.json', samples)
    decision = convergence(samples)
    save(root / 'convergence-decision.json', decision)
    if decision['converged']:
        size = SIZES[-1]
        campaign.update(phase='matched-qualification')
        for route in ROUTES[1:]:
            folder = campaign.run_case('qual-largest-' + route[0], 'qualify', route, size)
            qualifications[route[0]] = qualify(folder / 'qualification.json', size, banks[str(size)]['sha256'])
            save(root / 'convergence-qualifications.json', qualifications)
        campaign.update(phase='matched-timing')
        matched = {name: [] for name, *_ in ROUTES}
        for repeat in range(3):
            for route in ROUTES[repeat:] + ROUTES[:repeat]:
                folder = campaign.run_case(f'matched-{route[0]}-r{repeat + 1}', 'timing', route, size)
                qualified = qualifications[str(size) if route[0] == 'original-cpu' else route[0]]
                matched[route[0]].append(timing(folder, size, route[3], banks[str(size)]['sha256'], qualified))
                save(root / 'matched-samples.json', matched)
        campaign.update(phase='comparisons')

        def trigger(name, repeat):
            return root / f'runs/matched-{name}-r{repeat}/triggers.hdf'

        results = {}
        comparisons = [
            ('original-trigger-comparison', trigger('original-cpu', 1),
             [trigger(name, r) for name in ('torch-cpu', 'torch-cuda') for r in (1, 2, 3)]),
            ('corrected-trigger-comparison', trigger('corrected-cpu', 1),
             [trigger(name, r) for name in ('torch-cpu', 'torch-cuda') for r in (1, 2, 3)])]
        comparisons.extend((name + '-repeat-parity', trigger(name, 1), [trigger(name, r) for r in (2, 3)])
                           for name, *_ in ROUTES)
        for name, baseline, candidates in comparisons:
            results[name] = campaign.compare(name, baseline, candidates)
            save(root / 'matched-comparisons.json', results)
        if any(status != 'pass' for name, status in results.items() if name != 'original-trigger-comparison'):
            raise ValueError('Corrected or own-repeat parity failed/requires review; all outcomes retained')
    else:
        save(root / 'matched-skipped.json', {'reason': decision['classification']})
    if frozen != {str(path): digest(path) for path in pins}:
        raise ValueError('Campaign inputs or helpers mutated')
    source_record(ORIGINAL, ORIGINAL_COMMIT)
    source_record(FINAL, FINAL_COMMIT)
    campaign.update(state='complete', phase='complete', finished=time.time(),
                    converged=decision['converged'], selected_templates=decision['selected_templates'],
                    classification=decision['classification'], current=None, command=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-host', action='store_true', help='Required: observations do not imply a reservation.')
    parser.add_argument('--select-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.select_only:
        make_subsets(ROOT)
        return 0
    if not args.shared_host:
        parser.error('--shared-host is required; no reservation is claimed')
    if not sys.platform.startswith('linux'):
        parser.error('Campaign execution requires Linux; helpers/tests are portable')
    # An exclusive directory creation prevents concurrent starts and unsafe resume.
    (ROOT / 'stages').mkdir(exist_ok=False)
    campaign = Campaign(ROOT)
    interrupted = False

    def stop(signum, frame):
        global _pending_stop
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if _stop_deferrals:
                _pending_stop = signum
                return
            raise Cancelled(f'Received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    save(ROOT / 'convergence-plan.json', POLICY)
    campaign.update()
    campaign.observer = threading.Thread(target=campaign.observe, daemon=True)
    campaign.observer.start()
    try:
        run(campaign)
        return 0
    except BaseException as exc:
        campaign.update(state='cancelled' if isinstance(exc, Cancelled) else 'failed',
                        error=repr(exc), finished=time.time())
        raise
    finally:
        campaign.stop.set()
        campaign.observer.join(timeout=10)


if __name__ == '__main__':
    sys.exit(main())
