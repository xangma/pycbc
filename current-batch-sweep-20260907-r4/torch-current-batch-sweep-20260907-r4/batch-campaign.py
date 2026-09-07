#!/usr/bin/env python3
"""Current-revision batch sweep, queued behind the owned convergence campaign.

All science workers are serial, pinned to CPU 8, with one numerical thread.
This is an observed shared-host experiment, not a resource reservation.
"""
import argparse
import hashlib
import json
import importlib.util
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

import campaign_controls as control

ROOT = Path(__file__).resolve().parent
REVISION = '9578a710479b924e882857c4dffab6ed372a634b'
SOURCE = ROOT / 'source'
DEPENDENCY = Path('/home/xangma/pycbc-torch-convergence-20260907/convergence-status.json')
BATCHES = (1, 8, 32, 128, 512, 1024)
ROUTES = ('branch_standard', 'torch_cpu', 'torch_cuda')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
PLAN = {
    'revision': REVISION, 'batches': list(BATCHES), 'routes': list(ROUTES),
    'bank_templates': 1024, 'fft_samples': 131072, 'blocks': 3,
    'replicates': 3, 'warmups': 2, 'samples': 5, 'seed': 7102,
    'qualification_seeds': [7102, 7103], 'numerical_policy_version': 2,
    'physical_core_logical_cpu': 8, 'smt_sibling': 72, 'numerical_threads': 1,
    'qualification': 'Two fresh seeds, fixed banks across batches. Every complex SNR sample compared against an independent complex128 oracle and standard CPU batch 1 under policy v2; raw v1 errors retained as diagnostics. Actual route normalization is included. All trigger/veto gates unchanged.',
    'timing': 'Public LiveBatchMatchedFilter.process_data calls, uninstrumented; three fresh processes per cell, two warmups and five timed calls per process. Timing summaries are published only when all required qualifications pass.',
    'scope': 'shared-host diagnostic; no exclusive reservation',
    'excludes': ['frame I/O', 'PSD estimation', 'waveform generation',
                 'bank loading', 'CLI startup/teardown', 'full-machine capacity'],
    'ordering': 'Batch order rotates between replicates; route order rotates and reverses between cells.',
    'worker_timeout_seconds': 3600,
}


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'start_ticks': fields[19]}
    except FileNotFoundError:
        return None


def dependency_ready(status, identity=None):
    if status.get('state') in ('failed', 'cancelled', 'invalid-input-mutation'):
        raise ValueError('Convergence dependency failed; batch sweep will not start')
    if status.get('state') != 'complete':
        return False
    required = ('finished', 'pid', 'converged', 'selected_templates')
    if any(key not in status for key in required):
        raise ValueError('Completed dependency lacks final campaign evidence')
    return identity is None or identity.get('state') == 'Z'


def timing_order():
    result = []
    for replicate in range(3):
        batches = BATCHES[replicate:] + BATCHES[:replicate]
        for index, batch in enumerate(batches):
            shift = (replicate + index) % len(ROUTES)
            routes = ROUTES[shift:] + ROUTES[:shift]
            if replicate % 2:
                routes = tuple(reversed(routes))
            result.extend((replicate + 1, batch, route) for route in routes)
    return result


def environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('PYCBC_TORCH_') or key in (
                'PYCBC_ENABLE_CUDA_GRAPHS', 'PYCBC_BATCH_MAXELEMENTS'):
            del env[key]
    env.update({key: '1' for key in THREADS})
    env.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
               PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES='0')
    return env


def helper_hashes():
    return {name: control.digest(ROOT / name) for name in
            ('batch-campaign.py', 'batch-worker.py', 'campaign_controls.py',
             'native-provenance.json', 'NUMERICAL-POLICY.md', 'policy-decision.json')}


def require_qualification(result):
    if result.get('status') != 'pass':
        raise ValueError('Scientific qualification failed or did not finish')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def require_hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Missing or invalid qualification hash')


def validate_pointwise(result, bank, size, blocks, reference):
    """Independently check v2 evidence, including rows hidden by a pass rollup."""
    pointwise = result['pointwise_blocks']
    expected_shape = [blocks, bank, size]
    for name, dtype, filename in (('oracle', 'complex128', 'oracle.npy'),
                                  ('reference', 'complex64', 'outputs.npy')):
        item = result[name]
        if (item['shape'] != expected_shape or item['dtype'] != dtype
                or item['filename'] != filename
                or item['created'] is not (reference is None)
                or not item.get('directory')):
            raise ValueError('Missing or invalid oracle/reference provenance')
        if reference is not None and item['directory'] != str(reference):
            raise ValueError('Oracle/reference directory mismatch')
    ref = control.read(reference / 'result.json') if reference is not None else None
    if len(pointwise) != blocks:
        raise ValueError('Incomplete pointwise block coverage')

    def metric(record, samples):
        if (record['samples'] != samples or record['failed_samples'] != 0
                or record['nonfinite_actual'] != 0 or record['nonfinite_reference'] != 0
                or not math.isfinite(record['max_abs_error'])
                or not 0 <= record['max_abs_error'] <= 1e-3
                or not math.isfinite(record['max_tolerance_ratio'])
                or not 0 <= record['max_tolerance_ratio'] <= 1
                or not math.isclose(record['max_abs_error'] / 1e-3,
                                    record['max_tolerance_ratio'], rel_tol=1e-12)):
            raise ValueError('Incomplete or failed normalized complex SNR evidence')

    def normalization(record):
        fields = {key: record[key] for key in ('norm', 'sigmasq', 'source')}
        if (not fields['source'] or any(not math.isfinite(fields[key]) or fields[key] <= 0
                                      for key in ('norm', 'sigmasq'))
                or record['sha256'] != canonical_hash(fields)):
            raise ValueError('Missing or invalid observed normalization')

    for b, block in enumerate(pointwise):
        if (block['block'] != b or not block['processed_once']
                or block['processing_counts'] != [1] * bank or len(block['rows']) != bank
                or block['normalization_failed_rows'] != 0):
            raise ValueError('Incomplete or failed pointwise coverage')
        for key in (None, 'normalized_oracle', 'normalized_mkl_compatibility'):
            metric(block if key is None else block[key], bank * size)
        for i, row in enumerate(block['rows']):
            if row['template_id'] != 1000+i:
                raise ValueError('Pointwise template coverage mismatch')
            for key in (None, 'normalized_oracle', 'normalized_mkl_compatibility'):
                metric(row if key is None else row[key], size)
            normalization(row['normalization'])
            normalization(row['oracle']['normalization'])
            sources = {'observed_scalar_sigmasq_callback'}
            if result['route'] != 'branch_standard':
                sources.add('observed_bulk_magnitude_return_locals')
            if row['normalization']['source'] not in sources:
                raise ValueError('Normalization was not observed on an allowed default path')
            if row['oracle']['normalization']['source'] != 'independent_float64_power_sum':
                raise ValueError('Oracle normalization lacks independent provenance')
            for value in (row['sha256'], row['reference_sha256'], row['oracle']['sha256']):
                require_hash(value)
            sigma = row['sigmasq_oracle']
            actual_error = abs(row['normalization']['sigmasq']
                               / row['oracle']['normalization']['sigmasq'] - 1)
            if (sigma['passed'] is not True or not math.isfinite(sigma['relative_error'])
                    or not 0 <= sigma['relative_error'] <= 1e-3
                    or not math.isclose(actual_error, sigma['relative_error'],
                                        rel_tol=1e-10, abs_tol=1e-15)):
                raise ValueError('Failed or inconsistent independent sigma comparison')
            old = row['raw_complex_v1_audit']
            if (old['classification'] != 'audit_only_not_v2_acceptance'
                    or old['samples'] != size):
                raise ValueError('Missing original raw comparison audit')
            if row['normwise_diagnostics']['classification'] != 'diagnostic_only_no_threshold':
                raise ValueError('Missing separate normwise diagnostics')
            reference_row = row if ref is None else ref['pointwise_blocks'][b]['rows'][i]
            if (row['reference_sha256'] != reference_row['sha256']
                    or row['oracle'] != reference_row['oracle']):
                raise ValueError('Pointwise oracle/reference hash mismatch')
        for key in ('max_abs_error', 'max_tolerance_ratio'):
            if block[key] != max(row[key] for row in block['rows']):
                raise ValueError('Pointwise summary contradicts row evidence')


def validate_result(result, route, batch, mode, reference=None, smoke=False, seed=None):
    """Reject incomplete evidence independently of the worker's status flag."""
    if seed is None:
        seed = PLAN['seed']
    bank = 8 if smoke else PLAN['bank_templates']
    size = 32768 if smoke else PLAN['fft_samples']
    blocks = PLAN['blocks']
    for key, value in dict(schema_version=1, route=route, batch=batch, mode=mode,
                           expected_head=REVISION).items():
        if result.get(key) != value:
            raise ValueError(f'Worker metadata mismatch: {key}')
    if result['source']['revision'] != REVISION or result['source']['tracked_dirty']:
        raise ValueError('Worker source mismatch')
    if result['worker_sha256'] != control.digest(ROOT / 'batch-worker.py'):
        raise ValueError('Worker script differs from frozen campaign')
    if result['arguments']['seed'] != seed or result['inputs']['seed'] != seed:
        raise ValueError('Worker seed differs from the frozen matrix')
    spec = importlib.util.spec_from_file_location('policy_worker', ROOT / 'batch-worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    if result['tolerance_policy'] != worker.POLICY or worker.POLICY['version'] != 2:
        raise ValueError('Worker numerical policy differs from v2')
    if result['policy_sha256'] != canonical_hash(worker.POLICY):
        raise ValueError('Worker numerical policy hash mismatch')
    geometry = result['inputs']['geometry']
    if any(geometry[key] != value for key, value in
           dict(bank_templates=bank, fft_samples=size, blocks=blocks).items()):
        raise ValueError('Workload dimensions mismatch')
    if not result.get('input_sha256') or any(value != '1' for value in
                                             result['thread_environment'].values()):
        raise ValueError('Missing input identity or wrong numerical thread count')
    if result['affinity'] != [8]:
        raise ValueError('Worker affinity differs from the declared host core')
    if route != 'branch_standard' and any(result['runtime'][key] != 1 for key in
                                          ('torch_num_threads', 'torch_num_interop_threads')):
        raise ValueError('Torch thread count differs from the declared setting')
    groups = result['group_layout']
    if (sorted(t for group in groups for t in group) != list(range(1000, 1000+bank))
            or max(map(len, groups)) != batch):
        raise ValueError('Missing/duplicate templates or wrong execution batch')
    plans = result['fft_plans']
    if len(plans) != len(groups) or any(p['nbatch'] != len(g)
                                       for p, g in zip(plans, groups)):
        raise ValueError('FFT plan does not execute the declared batch')
    if route == 'branch_standard' and any(p['class'] != 'pycbc.fft.mkl.IFFT' for p in plans):
        raise ValueError('CPU reference is not MKL')
    if reference is not None:
        ref = control.read(reference / 'result.json')
        if (result['input_sha256'] != ref['input_sha256']
                or result['tolerance_policy'] != ref['tolerance_policy']
                or result['reference_result_sha256'] != control.digest(reference / 'result.json')):
            raise ValueError('Reference/input/policy identity mismatch')
    if result['status'] != 'pass':
        return  # Preserve scientific mismatches; they never permit timing.
    if result.get('failures') or result.get('exit_code') != 0:
        raise ValueError('Pass status contradicts failure evidence')
    triggers = result['trigger_blocks']
    if len(triggers) != blocks or any(t['count'] <= 0 for t in triggers):
        raise ValueError('Missing nonempty trigger evidence')
    comparisons = result['trigger_comparisons']
    expected_checks = blocks * (1 + PLAN['warmups'] + PLAN['samples']) if mode == 'timing' else blocks
    if reference is not None and (len(comparisons) != expected_checks
                                 or any(not check['passed'] for check in comparisons)):
        raise ValueError('Missing or failed trigger comparisons')
    if mode == 'qualify':
        validate_pointwise(result, bank, size, blocks, reference)
    else:
        timing = result['timing']
        samples = timing['warm_block_ms']
        totals = timing['warm_iteration_ms']
        if (timing['instrumented'] or timing['templates_per_iteration'] != bank * blocks
                or len(samples) != PLAN['samples'] or len(totals) != PLAN['samples']
                or any(len(row) != blocks or any(not math.isfinite(t) or t <= 0 for t in row)
                       for row in samples)
                or any(not math.isclose(sum(row), total, rel_tol=1e-12)
                       for row, total in zip(samples, totals))):
            raise ValueError('Invalid timing samples or work denominator')


def summarize(timings):
    expected = {(r, b, route) for r, b, route in timing_order()}
    if len(timings) != len(expected) or {(r['repeat'], r['batch'], r['route'])
                                       for r in timings} != expected:
        raise ValueError('Incomplete or duplicate timing matrix')
    cells = []
    for batch in BATCHES:
        for route in ROUTES:
            workers = [r['result'] for r in timings if r['batch'] == batch and r['route'] == route]
            elapsed = [statistics.median(r['timing']['warm_iteration_ms']) / 1000 for r in workers]
            work = PLAN['bank_templates'] * PLAN['blocks']
            g = workers[0]['inputs']['geometry']
            duration = (g['valid_end'] - g['valid_start']) / g['sample_rate']
            rates = [work / value for value in elapsed]
            cells.append({'batch': batch, 'route': route, 'workers': 3,
                          'worker_median_iteration_seconds': elapsed,
                          'median_iteration_seconds': statistics.median(elapsed),
                          'median_templates_per_second': statistics.median(rates),
                          'templates_per_second_range': [min(rates), max(rates)],
                          'valid_seconds_per_block': duration,
                          'median_template_seconds_per_second': statistics.median(rates) * duration})
    return {'plan': PLAN, 'cells': cells,
            'denominator': '1024 templates x 3 blocks per iteration; capacity multiplies by valid seconds per block. One affinity-pinned host CPU per worker; CUDA additionally uses one GPU.',
            'scope': 'Warm public live-filter API throughput on a shared host; excludes end-to-end CLI work and does not establish full-machine capacity.'}


class Campaign(control.Campaign):
    def __init__(self, root, timeout=3600):
        super().__init__(root, timeout)
        self.state['scope'] = PLAN['scope']
        self.state['launcher_command'] = list(sys.argv)

    def update(self, **values):
        self.state.update(values)
        control.save(self.root / 'batch-status.json', self.state)

    def observe(self):
        with (self.root / 'host-samples.jsonl').open('x') as log:
            while not self.stop.is_set():
                sample = {'utc': time.time(), 'phase': self.state.get('phase'),
                          'case': self.state.get('current'),
                          'child_pid': self.state.get('child_pid')}
                for name in ('stat', 'loadavg', 'meminfo'):
                    try:
                        sample['proc_' + name] = Path('/proc', name).read_text()
                    except OSError as exc:
                        sample[name + '_error'] = str(exc)
                try:
                    sample['gpu'] = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.free',
                         '--format=csv,noheader,nounits'], text=True, timeout=3).strip()
                    sample['processes'] = subprocess.check_output(
                        ['ps', '-eo', 'pid,ppid,psr,nlwp,pcpu,pmem,comm', '--sort=-pcpu'],
                        text=True, timeout=3)
                except (OSError, subprocess.SubprocessError) as exc:
                    sample['observer_error'] = str(exc)
                log.write(json.dumps(sample) + '\n')
                log.flush()
                self.stop.wait(10)

    def wait_dependency(self, path, timeout=21600):
        self.update(state='waiting', phase='dependency', dependency=str(path),
                    dependency_deadline=time.time() + timeout)
        deadline = time.monotonic() + timeout
        while True:
            status = control.read(path)
            identity = process_identity(status['pid']) if 'pid' in status else None
            if dependency_ready(status, identity):
                control.save(self.root / 'dependency-receipt.json',
                             {'path': str(path), 'sha256': control.digest(path),
                              'status': status, 'observed': time.time()})
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Convergence dependency did not finish within six hours')
            self.stop.wait(min(10, max(0, deadline - time.monotonic())))

    def capacity(self):
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available_gib = int(fields['MemAvailable'].split()[0]) / 1048576
        free_mib = float(subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=5).strip())
        record = {'available_ram_gib': available_gib, 'gpu_free_mib': free_mib,
                  'free_disk_gib': __import__('shutil').disk_usage(self.root).free / 2**30}
        control.save(self.root / 'capacity.json', record)
        if available_gib < 20 or free_mib < 14000 or record['free_disk_gib'] < 30:
            raise RuntimeError('Insufficient free capacity; see capacity.json')

    def worker(self, label, route, batch, mode, reference=None, smoke=False, seed=None):
        if seed is None:
            seed = PLAN['seed']
        target = self.root / 'runs' / label
        command = ['taskset', '-c', '8', sys.executable, self.root / 'batch-worker.py',
                   '--source-root', SOURCE, '--route', route, '--batch', str(batch),
                   '--mode', mode, '--output-dir', target, '--seed', str(seed),
                   '--size', str(32768 if smoke else PLAN['fft_samples']),
                   '--bank-size', str(8 if smoke else PLAN['bank_templates']),
                   '--num-blocks', str(PLAN['blocks']), '--warmups', str(PLAN['warmups']),
                   '--samples', str(PLAN['samples']), '--cuda-device', '0']
        if reference is not None:
            command.extend(['--reference-dir', reference])
        receipt = self.stage(label, command, environment(), allowed=(0, 1))
        result = control.read(target / 'result.json')
        control.save(target / 'acquisition.json',
                     {'stage_receipt': str(self.root / 'stages' / (label + '.json')),
                      'result_sha256': control.digest(target / 'result.json'),
                      'whole_worker_wall_seconds': receipt['elapsed_wall_seconds']})
        validate_result(result, route, batch, mode, reference, smoke, seed)
        return result


def require_matrix(results, phase, batches):
    """Every cell of both frozen seeds must qualify before any timing."""
    expected = {f'{phase}-s{seed}-b{batch}-{route}'
                for seed in PLAN['qualification_seeds']
                for batch in batches for route in ROUTES}
    if set(results) != expected:
        raise ValueError('Incomplete qualification matrix; timing skipped')
    failures = [name for name, result in results.items() if result.get('status') != 'pass']
    if failures:
        raise ValueError(f'Qualification failures; timing skipped: {failures}')


def run(campaign):
    campaign.update(phase='preflight', state='running')
    if helper_hashes() != control.read(ROOT / 'queued-inputs.json'):
        raise ValueError('Queued helpers changed while waiting')
    decision = control.read(ROOT / 'policy-decision.json')
    if decision.get('adopted') is not True or decision.get('policy_version') != 2:
        raise ValueError('Numerical policy v2 has not been adopted')
    source = control.source_record(SOURCE, REVISION)
    helpers = [ROOT / name for name in ('batch-campaign.py', 'batch-worker.py',
                                        'campaign_controls.py', 'native-provenance.json',
                                        'NUMERICAL-POLICY.md', 'policy-decision.json')]
    frozen = {str(path): control.digest(path) for path in helpers}
    native = control.read(ROOT / 'native-provenance.json')
    for entry in native['extensions']:
        path = SOURCE / entry['relative_path']
        if control.digest(path) != entry['sha256']:
            raise ValueError(f'Native extension hash changed: {path}')
    control.save(ROOT / 'provenance.json', {'source': source, 'helper_sha256': frozen,
                                          'native': native, 'plan': PLAN})
    campaign.capacity()
    qualifications = {}
    for smoke, phase, batches in ((True, 'smoke', (1, 8)), (False, 'qual', BATCHES)):
        campaign.update(phase='smoke-qualification' if smoke else 'qualification')
        phase_results = {}
        for seed in PLAN['qualification_seeds']:
            reference = ROOT / f'runs/{phase}-s{seed}-b1-branch_standard'
            for batch in batches:
                for route in ROUTES:
                    label = f'{phase}-s{seed}-b{batch}-{route}'
                    result = campaign.worker(label, route, batch, 'qualify',
                                             None if label == reference.name else reference,
                                             smoke=smoke, seed=seed)
                    phase_results[label] = result
                    control.save(ROOT / f'{phase}-qualifications.json', phase_results)
                    if label == reference.name:
                        require_qualification(result)
        require_matrix(phase_results, phase, batches)
        if not smoke:
            qualifications = phase_results
    reference = ROOT / f"runs/qual-s{PLAN['seed']}-b1-branch_standard"
    campaign.update(phase='timing')
    timings = []
    for repeat, batch, route in timing_order():
        label = f'time-r{repeat}-b{batch}-{route}'
        result = campaign.worker(label, route, batch, 'timing', reference)
        require_qualification(result)
        timings.append({'repeat': repeat, 'batch': batch, 'route': route, 'result': result})
        control.save(ROOT / 'timings.json', timings)
    control.save(ROOT / 'summary.json', summarize(timings))
    if frozen != {str(path): control.digest(path) for path in helpers}:
        raise ValueError('Campaign helper or native provenance changed')
    control.source_record(SOURCE, REVISION)
    for entry in native['extensions']:
        if control.digest(SOURCE / entry['relative_path']) != entry['sha256']:
            raise ValueError('Native extension changed during campaign')
    campaign.update(state='complete', phase='complete', current=None, command=None,
                    finished=time.time(), qualifications_passed=len(qualifications),
                    timing_workers=len(timings), source_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-host', action='store_true')
    parser.add_argument('--snr-policy-v2', action='store_true',
                        help='Explicitly select the separately versioned complex-SNR criterion')
    args = parser.parse_args()
    if not args.shared_host or not args.snr_policy_v2 or not sys.platform.startswith('linux'):
        parser.error('Linux, --shared-host and --snr-policy-v2 are required')
    (ROOT / 'stages').mkdir(exist_ok=False)
    (ROOT / 'runs').mkdir(exist_ok=False)
    campaign = Campaign(ROOT)
    clean_environment = environment()
    os.environ.clear()
    os.environ.update(clean_environment)
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if control._stop_deferrals:
                control._pending_stop = signum
                return
            raise control.Cancelled(f'Received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    control.save(ROOT / 'batch-plan.json', PLAN)
    control.save(ROOT / 'queued-inputs.json', helper_hashes())
    campaign.update()
    # The observer starts after waiting, avoiding extra GPU queries during the
    # predecessor's timed measurements.
    try:
        campaign.wait_dependency(DEPENDENCY)
        campaign.observer = threading.Thread(target=campaign.observe, daemon=True)
        campaign.observer.start()
        run(campaign)
        return 0
    except BaseException as exc:
        campaign.update(state='cancelled' if isinstance(exc, control.Cancelled) else 'failed',
                        error=repr(exc), finished=time.time())
        raise
    finally:
        campaign.stop.set()
        if campaign.observer is not None:
            campaign.observer.join(timeout=10)


if __name__ == '__main__':
    sys.exit(main())
