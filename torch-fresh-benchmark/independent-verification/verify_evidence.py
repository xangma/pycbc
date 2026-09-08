"""Read archived evidence; write only independent verification artifacts.

Does not import PyCBC or execute campaign/setup code. The pinned comparator is
replayed without changing it, then numerical metrics and identities are checked
again directly from arrays. No remote access or scientific code changes.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import posixpath
from pathlib import Path
import subprocess
import sys
import math
import statistics
import re
import csv

import h5py
import numpy as np

sys.dont_write_bytecode = True

OUT = Path(__file__).resolve().parent
O = OUT.parent
REPO = Path('/Users/xangma/repos/pycbc')
ORIGINAL = '40e94792b3edf59f39b18b65102b28a4f74433a7'
PROPOSED = 'eb8fef9ed1d06378b59cae8439fd40af63827575'
TIMING = {'search/' + name for name in (
    'filter_rate_per_core', 'run_time', 'setup_time_fraction', 'templates_per_core')}
TOL = dict(rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5, phase_atol=1e-4, max_examples=12)
ARMS = ('original-cpu', 'proposed-cpu', 'torch-cpu', 'torch-cuda')
PAIRS = [('original-cpu', arm) for arm in ARMS[1:]] + [
    ('proposed-cpu', 'torch-cpu'), ('proposed-cpu', 'torch-cuda')]


def sha_data(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def snapshot(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def git_hashes(commit):
    entries, modes = {}, {}
    for row in git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
        if row:
            meta, path = row.split(b'\t', 1)
            mode, kind, oid = meta.split()
            require(kind == b'blob', 'Unexpected non-blob source entry')
            entries[path.decode()] = oid.decode()
            modes[path.decode()] = mode.decode()
    unique = sorted(set(entries.values()))
    process = subprocess.run(['git', '-C', str(REPO), 'cat-file', '--batch'],
                             input=('\n'.join(unique) + '\n').encode(), capture_output=True, check=True)
    stream = io.BytesIO(process.stdout)
    hashes, link_targets = {}, {}
    link_oids = {entries[p] for p in entries if modes[p] == "120000"}
    for oid in unique:
        found, kind, size = stream.readline().split()
        require(found.decode() == oid and kind == b'blob', 'Git blob response mismatch')
        content = stream.read(int(size))
        hashes[oid] = sha_data(content)
        if oid in link_oids:
            link_targets[oid] = content.decode()
        require(stream.read(1) == b'\n', 'Git blob delimiter missing')
    def resolve(name, seen=()):
        require(name in entries and name not in seen, 'Invalid Git symlink target')
        if modes[name] == '120000':
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link_targets[entries[name]]))
            return resolve(target, seen + (name,))
        return hashes[entries[name]]
    return {name: resolve(name) for name in entries}


def datasets(folder):
    result = {}
    with h5py.File(folder / 'triggers.hdf', 'r') as stream:
        require(set(stream) == {'H1'}, 'Unexpected top-level HDF objects')
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = obj[()]
        stream['H1'].visititems(collect)
    return result


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def descriptor(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), nbytes=a.nbytes,
                data_sha256=sha_data(a.tobytes()))


def field_comparison(a, b):
    names_a, names_b = set(a), set(b)
    rows = {}
    for name in sorted(names_a & names_b):
        x, y = a[name], b[name]
        rows[name] = dict(left=descriptor(x), right=descriptor(y),
                          dtype_equal=x.dtype == y.dtype, shape_equal=x.shape == y.shape,
                          bytes_equal=x.tobytes() == y.tobytes(), exact=exact(x, y))
        if name in TIMING:
            rows[name].update(left_value=x.tolist(), right_value=y.tolist())
    return dict(left_dataset_names=sorted(names_a), right_dataset_names=sorted(names_b),
                dataset_field_sets_equal=names_a == names_b,
                scientific_field_sets_equal=names_a - TIMING == names_b - TIMING,
                left_only=sorted(names_a - names_b), right_only=sorted(names_b - names_a),
                datasets=rows)


def conditioning(left, right, arrays):
    a, b = left['observations'], right['observations']
    psds = []
    require(len(a['psd_arrays']) == len(b['psd_arrays']), 'PSD field sets differ')
    for x, y in zip(a['psd_arrays'], b['psd_arrays']):
        require(x['segment_indices'] == y['segment_indices'], 'PSD segment mapping differs')
        av, bv = arrays[x['path']], arrays[y['path']]
        require(av.shape == bv.shape, 'PSD shapes differ')
        finite = np.isfinite(av) & np.isfinite(bv)
        d = np.abs(av[finite].astype(float) - bv[finite].astype(float))
        budget = 1e-4 * np.maximum(np.abs(av[finite].astype(float)), np.abs(bv[finite].astype(float)))
        psds.append(dict(exact=exact(av, bv),
                         nonfinite_masks_equal=bool(np.array_equal(np.isposinf(av), np.isposinf(bv))),
                         relative_budget=1e-4, absolute_floor=0,
                         violations=int(np.count_nonzero(d > budget)),
                         max_absolute_difference=float(d.max(initial=0))))
    return dict(conditioned_strain_exact=a['conditioned_strain'] == b['conditioned_strain'],
                geometry_exact=a['segment_geometry'] == b['segment_geometry'], psds=psds)


def validate_run(folder, config, pins, arrays):
    receipt, q, runtime = [read(folder / name) for name in ('receipt.json', 'qualification.json', 'runtime.json')]
    arm = receipt['arm']
    source_name = 'original' if arm == 'original-cpu' else 'proposed'
    source = pins[source_name]
    root = receipt['source']
    require(receipt['state'] == runtime['state'] == 'complete' and receipt['returncode'] == 0,
            f'{arm}: incomplete receipt')
    require(receipt['source_info'] == source['info'] and receipt['source_status_after'] == '',
            f'{arm}: source receipt mismatch')
    require(receipt['input_sha256'] == receipt['input_sha256_after'], f'{arm}: changed inputs')
    expected_inputs = dict(config['input_pins'])
    expected_inputs[root + '/bin/pycbc_inspiral'] = source['tracked']['bin/pycbc_inspiral']
    for path, value in expected_inputs.items():
        require(receipt['input_sha256'][path] == value, f'{arm}: input pin mismatch {path}')
    for path, value in receipt['input_sha256'].items():
        if path not in expected_inputs:
            require(sha(folder.parents[1] / Path(path).name) == value, f'{arm}: harness receipt mismatch')
    require(sha(folder / 'triggers.hdf') == receipt['trigger_sha256'], f'{arm}: HDF receipt mismatch')
    require(q['status'] == 'success' and q['executable_exit_code'] == 0, f'{arm}: qualification failed')
    require(q['argv'] == q['executed_argv'] == receipt['executable_cli'], f'{arm}: CLI disagreement')
    require(q['source_root'] == runtime['source'] == runtime['imported_source'] == root,
            f'{arm}: imported source mismatch')
    require(q['executable']['sha256'] == source['tracked']['bin/pycbc_inspiral'], f'{arm}: executable mismatch')
    module_counts = {}
    for label, modules in [('runtime', runtime['source_modules']), ('qualification', q['source_modules'])]:
        for path, value in modules.items():
            if isinstance(value, dict):
                path, value = value["path"], value["sha256"]
            rel = str(Path(path).relative_to(root))
            expected = source['generated_version'] if rel == 'pycbc/version.py' else (
                source['native'].get(rel) or source['tracked'].get(rel))
            require(expected == value, f'{arm}: module pin mismatch {rel}')
        module_counts[label] = len(modules)
    require(runtime['helper_sha256'] == sha(folder.parents[1] / 'checked-inspiral.py'), 'Helper mismatch')
    env = dict(config['environment'], OMP_DYNAMIC='FALSE', CUDA_VISIBLE_DEVICES='0',
               PYTHONPATH=root, PYTHONDONTWRITEBYTECODE='1')
    require(receipt['environment'] == runtime['environment'] == env, f'{arm}: environment mismatch')
    scheme = {'original-cpu': 'cpu:1', 'proposed-cpu': 'cpu:1',
              'torch-cpu': 'torch:cpu:1', 'torch-cuda': 'torch:cuda:0'}[arm]
    require(receipt['scheme'] == runtime['processing_scheme'] == scheme, f'{arm}: scheme mismatch')
    expected_threads = 1 if arm.startswith('torch') else None
    thread_stages = {}
    for stage in ('before_executable', 'at_first_bank', 'after_executable'):
        value = runtime[stage]
        require(value['affinity'] == [8] and value['smt_siblings'] == {'8': '8,72'}, 'Affinity mismatch')
        require(value['intra_op'] == value['inter_op'] == expected_threads, 'Torch threads mismatch')
        require(value['environment'] == env and value['threadpools'] and
                all(p['num_threads'] == 1 for p in value['threadpools']), 'Threadpool mismatch')
        require(value['threadpoolctl']['sha256'] == sha(folder.parents[1] / 'threadpoolctl.py'), 'Thread observer mismatch')
        thread_stages[stage] = {k: value[k] for k in ('affinity', 'intra_op', 'inter_op', 'scheme', 'device', 'threadpools')}
    expected_scheme = ('TorchScheme', 'cuda:0' if arm == 'torch-cuda' else 'cpu') if arm.startswith('torch') else ('CPUScheme', '')
    require((runtime['at_first_bank']['scheme'], runtime['at_first_bank']['device']) == expected_scheme, 'Runtime route mismatch')
    obs = q['observations']
    require(q['checks'] and all(v is True for v in q['checks'].values()), 'Recorded qualification check failed')
    require(len(obs['banks']) == len(obs['matched_filter_controllers']) == len(obs['segment_geometry']) == 1,
            'Bank/controller/geometry count mismatch')
    bank, = obs['banks']
    require(bank['selected_template_count'] == bank['expected_decompressions'] == 384 and
            bank['expected_filter_calls'] == 1920 and bank['has_compressed_waveforms'] and bank['enable_compressed_waveforms'],
            'Bank qualification mismatch')
    require(bank['file']['sha256'] == config['input_pins'][config['bank']], 'Bank hash mismatch')
    require(set(bank['templates']) == {str(i) for i in range(384)}, 'Template field set mismatch')
    for i, row in bank['templates'].items():
        require(row['index'] == int(i) and row['getitem_attempts'] == row['getitem_successes'] == row['decompression_successes'] == 1,
                'Template decompression mismatch')
        require(row['filter_attempts_by_segment'] == row['filter_successes_by_segment'] == {str(i): 1 for i in range(5)},
                'Template/segment coverage mismatch')
    calls = obs['decompression_calls']
    require(len(calls) == obs['decompression_success_count'] == 384 and
            all(c['success'] and c['actual_interpolation'] == 'inline_linear' for c in calls) and
            {c['index'] for c in calls} == set(range(384)) and not obs['waveform_generation_attempts'],
            'Decompression/fallback evidence mismatch')
    ctrl, = obs['matched_filter_controllers']
    require(ctrl['segment_count'] == 5 and [ctrl['filter_bin_start'], ctrl['filter_bin_stop']] == [15360, 1048576],
            'Filter geometry mismatch')
    engine = obs['fft_engines'][ctrl['ifft_engine_id']]
    require(engine['execute_attempts'] == engine['execute_successes'] == 1920 and engine['nbatch'] == 1 and
            engine['size'] == 2097152 and engine['input_dtype'] == engine['output_dtype'] == 'complex64', 'IFFT count mismatch')
    geom, = obs['segment_geometry']
    intervals = []
    for seg in geom['segments']:
        start = seg['segment_slice']['start']
        require(seg['segment_slice']['stop'] - start == 2097152, 'FFT window mismatch')
        interval = [start + seg['analyze_slice'][k] for k in ('start', 'stop')]
        require(interval == seg['analyzed_sample_interval'], 'Analyzed interval mismatch')
        intervals.append(interval)
    require(len(intervals) == 5 and all(x[1] == y[0] for x, y in zip(intervals, intervals[1:])) and
            [intervals[0][0], intervals[-1][1]] == [458752, 8257536] and
            sum(y - x for x, y in intervals) == geom['unique_analyzed_samples'] == 7798784 and
            geom['unique_analyzed_seconds'] == 1904 and geom['gap_samples'] == geom['overlap_samples'] == 0 and
            geom['sample_rate_hz'] == 4096 and geom['fft_samples'] == 2097152, 'Geometry coverage mismatch')
    require(len(obs['conditioned_strain']) == 1, 'Strain receipt missing')
    psd_checks = []
    require(sorted(i for p in obs['psd_arrays'] for i in p['segment_indices']) == list(range(5)), 'Missing segment PSD')
    for row in obs['psd_arrays']:
        path = folder / row['relative_path']
        a = np.load(path, allow_pickle=False)
        require(sha(path) == row['sha256'] and sha_data(a.tobytes()) == row['data_sha256'] and
                list(a.shape) == row['shape'] and a.dtype.str == row['dtype_str'], 'PSD hash/schema mismatch')
        require(a.shape == (1048577,) and a.dtype.str == '<f4' and
                row['n_samples'] == a.size and row['nbytes'] == a.nbytes and
                row['delta_f_hz'] == 1 / 512 and row['dyn_range_factor'] == float(2 ** 69) and
                row['scaling'] == 'DYN_RANGE_FAC**2', 'PSD extent/frequency/scaling mismatch')
        require(not np.isnan(a).any() and not np.isneginf(a).any() and not np.any(a <= 0) and
                np.all(np.isfinite(a[15360:1048576])) and row['validity']['valid_for_filter'], 'Invalid filter PSD')
        arrays[row['path']] = a
        psd_checks.append(dict(file=row['relative_path'], **descriptor(a), file_sha256=sha(path),
                               positive_infinity_count=int(np.isposinf(a).sum()), segments=row['segment_indices']))
    return q, dict(status='pass', source_commit=source['info']['commit'], trigger_sha256=receipt['trigger_sha256'],
                   input_hash_receipts_consistent=True, source_module_hash_counts=module_counts,
                   qualification_checks=q['checks'], independently_checked_templates=384, template_segment_pairs=1920,
                   thread_stages=thread_stages, fft_engine=engine, psds=psd_checks,
                   conditioned_strain_evidence='recorded digest/schema only; raw strain not included',
                   conditioned_strain=obs['conditioned_strain'], segment_geometry=obs['segment_geometry'])


def independent_metrics(a, b, result):
    require(set(a) - TIMING == set(b) - TIMING, 'Scientific dataset field sets differ')
    fields = {k for k in a if '/' not in k}
    require(fields == {k for k in b if '/' not in k}, 'Event field sets differ')
    def indices(data):
        ticks = np.rint(data['end_time'] * 4096).astype(np.int64)
        require(np.all(np.abs(data['end_time'] * 4096 - ticks) <= .01), 'Off-grid trigger')
        keys = list(zip(map(int, data['template_hash']), map(int, ticks)))
        require(len(set(keys)) == len(keys), 'Duplicate event identity')
        return dict(zip(keys, range(len(keys))))
    ia, ib = indices(a), indices(b)
    keys = sorted(ia.keys() & ib.keys())
    h = result['detectors']['H1']
    require(h['baseline_count'] == len(ia) and h['candidate_count'] == len(ib) and h['matched_count'] == len(keys),
            'Recomputed count mismatch')
    require(h['unmatched']['baseline_only']['count'] == len(ia.keys() - ib.keys()) and
            h['unmatched']['candidate_only']['count'] == len(ib.keys() - ia.keys()), 'Unmatched count mismatch')
    for field in sorted(fields - {'template_hash', 'end_time'}):
        x = np.asarray([a[field][ia[key]] for key in keys], dtype=np.float64)
        y = np.asarray([b[field][ib[key]] for key in keys], dtype=np.float64)
        difference = abs(x - y)
        scale = np.maximum(abs(x), abs(y))
        if field == 'coa_phase':
            difference = abs(np.angle(np.exp(1j * (x - y))))
            budget = np.full(len(x), 1e-4)
        elif field.endswith('_dof') or a[field].dtype.kind in 'iu':
            budget = np.zeros(len(x))
        elif field == 'sigmasq':
            budget = 1e-5 * scale
        else:
            budget = 1e-5 + 1e-4 * scale
        worst = int(np.argmax(difference))
        actual = dict(violations=int(np.count_nonzero(difference > budget)),
                      max_absolute_error=float(difference[worst]),
                      max_relative_error=float(np.max(np.divide(difference, scale, out=np.zeros_like(scale), where=scale > 0))),
                      worst_identity=list(keys[worst]), baseline_value=float(x[worst]),
                      candidate_value=float(y[worst]), budget_at_worst=float(budget[worst]))
        require(actual == h['metrics'][field], f'Independent metric mismatch {field}')
    return dict(scientific_field_sets_equal=True, event_field_sets_equal=True,
                independently_recomputed_metric_fields=sorted(fields - {'template_hash', 'end_time'}),
                baseline_only_identities=[list(k) for k in sorted(ia.keys() - ib.keys())],
                candidate_only_identities=[list(k) for k in sorted(ib.keys() - ia.keys())])



SCHEMES = {'original-cpu': 'cpu:1', 'proposed-cpu': 'cpu:1',
           'torch-cpu': 'torch:cpu:1', 'torch-cuda': 'torch:cuda:0'}
ORDERING = [list(ARMS[i:] + ARMS[:i]) for i in range(4)]
INPUT_PINS = {
    '/home/xangma/pycbc-torch-reference-protocol-20260907/inputs/bank-compressed.hdf':
        '26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f',
    '/home/xangma/pycbc_bench_repo/docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf':
        '580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865',
}
WORK = 384 * 1904


def utc(value):
    return datetime.datetime.fromisoformat(value).timestamp()


def range_stats(values):
    require(values and all(math.isfinite(v) for v in values), 'Invalid summary values')
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def observation_proof(folder):
    samples = [json.loads(line) for line in (folder / 'host-samples.jsonl').read_text().splitlines()]
    require(len(samples) >= 2 and samples[0]['phase'] == 'before' and
            samples[-1]['phase'] == 'after' and
            all(s['phase'] == 'during' for s in samples[1:-1]), 'Host observation phases missing')
    percentages = {'host': [], 'core_8': [], 'smt_sibling_72': []}
    for before, after in zip(samples, samples[1:]):
        a, b = before['cpu_ticks'], after['cpu_ticks']
        total = busy = 0
        per_cpu = {}
        complete = a.keys() == b.keys()
        for cpu in b.keys() & a.keys():
            elapsed = b[cpu]['total'] - a[cpu]['total']
            idle = b[cpu]['idle'] - a[cpu]['idle']
            if elapsed <= 0 or idle < 0 or idle > elapsed:
                complete = False
                continue
            total += elapsed
            busy += elapsed - idle
            per_cpu[cpu] = 100.0 * (elapsed - idle) / elapsed
        actual = dict(complete=complete and bool(total),
                      host_percent=100.0 * busy / total if total else None,
                      per_cpu_percent=per_cpu,
                      interval_seconds=after['monotonic_seconds'] - before['monotonic_seconds'])
        require(actual == after['utilization'] and actual['interval_seconds'] > 0,
                'Host utilization recomputation mismatch')
        if actual['host_percent'] is not None:
            percentages['host'].append(actual['host_percent'])
        for label, cpu in [('core_8', '8'), ('smt_sibling_72', '72')]:
            if cpu in per_cpu:
                percentages[label].append(per_cpu[cpu])
    gpu_rows = [read(folder / 'gpu-before.json'), read(folder / 'gpu-after.json')]
    during = folder / 'gpu-during.jsonl'
    if during.exists():
        gpu_rows += [json.loads(line) for line in during.read_text().splitlines()]
    gpu_records = []
    for row in gpu_rows:
        require(row['returncode'] == 0, 'GPU observation failed')
        for fields in csv.reader(row['output'].splitlines()):
            require(len(fields) == 5, 'Unexpected GPU observation columns')
            name, uuid, utilization, memory, capacity = [s.strip() for s in fields]
            gpu_records.append(dict(name=name, uuid=uuid, utilization_percent=float(utilization),
                                    memory_used_mib=float(memory), memory_total_mib=float(capacity)))
    require(gpu_records, 'Missing GPU observations')
    return dict(host_samples=len(samples), gpu_samples=len(gpu_rows),
                host_sample_span_seconds=samples[-1]['monotonic_seconds'] - samples[0]['monotonic_seconds'],
                utilization_percent={k: range_stats(v) for k, v in percentages.items()},
                gpu_uuid=sorted({r['uuid'] for r in gpu_records}),
                gpu_utilization_percent=range_stats([r['utilization_percent'] for r in gpu_records]),
                gpu_memory_used_mib=range_stats([r['memory_used_mib'] for r in gpu_records]),
                interpretation='Observed shared-host activity; these samples do not establish a host or GPU reservation.')


def validate_receipt(folder, config, pins, dependencies):
    """Common source, command, timer and runtime checks for every process."""
    receipt, runtime = [read(folder / name) for name in ('receipt.json', 'runtime.json')]
    arm, mode, case = receipt['arm'], receipt['mode'], receipt['case']
    require(arm in ARMS and mode in ('qualify', 'timing'), 'Unknown arm/mode')
    repeat = receipt['repeat']
    expected_case = 'qual-' + arm if mode == 'qualify' else f'timing-{arm}-r{repeat}'
    require(folder.name == case == expected_case, 'Case name mismatch')
    require(repeat is None if mode == 'qualify' else type(repeat) is int and 1 <= repeat <= 4,
            'Invalid repeat number')
    require(receipt['state'] == runtime['state'] == 'complete' and receipt['returncode'] == 0,
            f'{case}: run did not complete')
    source_name = 'original' if arm == 'original-cpu' else 'proposed'
    pin = pins[source_name]
    root, cwd = receipt['source'], receipt['cwd']
    require(root == cwd + '/' + source_name and receipt['hostname'] == runtime['hostname'] == 'len',
            f'{case}: unexpected source/host')
    require(receipt['source_info'] == pin['info'] and receipt['source_status_after'] == '',
            f'{case}: changed source')
    require(runtime['source'] == runtime['imported_source'] == root and
            receipt['scheme'] == runtime['processing_scheme'] == SCHEMES[arm],
            f'{case}: wrong imported source or route')
    expected_cli = [root + '/bin/pycbc_inspiral', *config['common_args'],
                    '--bank-file', config['bank'], '--processing-scheme', SCHEMES[arm],
                    '--segment-length', '512', '--segment-start-pad', '112', '--segment-end-pad', '16',
                    '--output', cwd + '/runs/' + case + '/triggers.hdf']
    require(receipt['executable_cli'] == expected_cli, f'{case}: unexpected executable command')
    expected_inner = expected_cli if mode == 'timing' else [
        cwd + '/qualify-inspiral.py', '--receipt',
        cwd + '/runs/' + case + '/qualification.json', '--', *expected_cli]
    require(runtime['command'] == expected_inner, f'{case}: unexpected instrumentation')
    command = receipt['command']
    prefix = ['/usr/bin/time', '-v', '-o', cwd + '/runs/' + case + '/time.txt',
              'taskset', '-c', '8', dependencies['executable'], cwd + '/checked-inspiral.py',
              '--receipt', cwd + '/runs/' + case + '/runtime.json',
              '--config', cwd + '/config.json', '--source', root, '--scheme', SCHEMES[arm], '--lock-fd']
    require(command[:len(prefix)] == prefix and
            command[len(prefix) + 1:] == ['--', *expected_inner], f'{case}: launch wrapper mismatch')
    require(int(command[len(prefix)]) == runtime['inherited_lock']['fd'] >= 3,
            f'{case}: wrong inherited lock descriptor')
    env = dict(config['environment'], OMP_DYNAMIC='FALSE', CUDA_VISIBLE_DEVICES='0',
               PYTHONPATH=root, PYTHONDONTWRITEBYTECODE='1')
    require(receipt['environment'] == runtime['environment'] == env, f'{case}: environment mismatch')
    expected_inputs = dict(config['input_pins'])
    expected_inputs[root + '/bin/pycbc_inspiral'] = pin['tracked']['bin/pycbc_inspiral']
    for name in ('config.json', 'checked-inspiral.py'):
        expected_inputs[cwd + '/' + name] = sha(folder.parents[1] / name)
    require(receipt['input_sha256'] == receipt['input_sha256_after'] == expected_inputs,
            f'{case}: input/hash fields differ')
    require(sha(folder / 'triggers.hdf') == receipt['trigger_sha256'] and
            runtime['helper_sha256'] == sha(folder.parents[1] / 'checked-inspiral.py'),
            f'{case}: output/helper hash mismatch')
    modules = runtime['source_modules']
    require(len(modules) > 20, f'{case}: incomplete module provenance')
    native_modules = {}
    for path, value in modules.items():
        rel = str(Path(path).relative_to(root))
        expected = pin['generated_version'] if rel == 'pycbc/version.py' else (
            pin['native'].get(rel) or pin['tracked'].get(rel))
        require(expected == value, f'{case}: imported module hash mismatch {rel}')
        if rel in pin['native']:
            native_modules[rel] = value
    require(native_modules, f'{case}: no native module receipts')
    stages = {}
    for name in ('before_executable', 'at_first_bank', 'after_executable'):
        stage = runtime[name]
        threads = 1 if arm.startswith('torch') else None
        require(stage['intra_op'] == stage['inter_op'] == threads and stage['affinity'] == [8] and
                stage['smt_siblings'] == {'8': '8,72'} and stage['environment'] == env,
                f'{case}: wrong thread count/affinity')
        require(stage['threadpools'] and all(p['num_threads'] == 1 for p in stage['threadpools']),
                f'{case}: nonscalar native threadpool')
        require(stage['threadpoolctl'] == dict(path=cwd + '/threadpoolctl.py', version='3.6.0',
                sha256=sha(folder.parents[1] / 'threadpoolctl.py')), f'{case}: observer provenance mismatch')
        stages[name] = {k: stage[k] for k in ('intra_op', 'inter_op', 'affinity', 'scheme', 'device', 'torch_imported')}
    expected_route = ('TorchScheme', 'cuda:0' if arm == 'torch-cuda' else 'cpu') if arm.startswith('torch') else ('CPUScheme', '')
    require((runtime['at_first_bank']['scheme'], runtime['at_first_bank']['device']) == expected_route,
            f'{case}: runtime route mismatch')
    elapsed = receipt['elapsed_wall_seconds']
    require(math.isfinite(elapsed) and elapsed > 0, f'{case}: invalid elapsed timer')
    time_text = (folder / 'time.txt').read_text()
    match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', time_text)
    require(match is not None and re.search(r'Exit status: 0(?:\n|$)', time_text), f'{case}: GNU time missing/failed')
    gnu_elapsed = 0.0
    for part in match.group(1).split(':'):
        gnu_elapsed = gnu_elapsed * 60 + float(part)
    require(gnu_elapsed - .02 <= elapsed <= gnu_elapsed + 2,
            f'{case}: independent elapsed clocks disagree')
    require(utc(receipt['started_utc']) <= runtime['started_at'] <=
            runtime['before_executable']['observed_at'] <= runtime['at_first_bank']['observed_at'] <=
            runtime['after_executable']['observed_at'] <= runtime['finished_at'] <= utc(receipt['finished_utc']),
            f'{case}: invalid process chronology')
    require(runtime['finished_at'] - runtime['started_at'] <= elapsed + .02,
            f'{case}: runtime not contained within elapsed timing')
    require(not (folder / 'qualification.json').exists() if mode == 'timing' else
            (folder / 'qualification.json').exists(), f'{case}: wrong qualification artifact presence')
    return dict(status='pass', arm=arm, mode=mode, repeat=repeat, elapsed_wall_seconds=elapsed,
                gnu_elapsed_wall_seconds=gnu_elapsed, source_commit=pin['info']['commit'],
                actual_torch_version=runtime['torch_version'], actual_torch_cuda_version=runtime['torch_cuda_version'],
                distribution_inventory_torch_version=dependencies['packages'].get('torch'),
                source_module_count=len(modules), imported_native_module_sha256=native_modules,
                trigger_sha256=receipt['trigger_sha256'], stages=stages,
                observations=observation_proof(folder)), receipt


def compare_archived(folder, comparator, loaded, data, left, right, name):
    a, b = loaded[left], loaded[right]
    raw = comparator.compare(a, b, TOL)
    require(raw == read(folder / 'comparisons' / (name + '-raw.json')), f'{name}: raw comparison mismatch')
    fixed, substitutions = copy.deepcopy(b), []
    for key in ('source_snapshot', 'consumed_input_sha256'):
        if fixed['metadata'][key] != a['metadata'][key]:
            substitutions.append(dict(field=key, before=fixed['metadata'][key], after=a['metadata'][key]))
            fixed['metadata'][key] = copy.deepcopy(a['metadata'][key])
    result = comparator.compare(a, fixed, TOL)
    require(read(folder / 'comparisons' / (name + '.json')) ==
            dict(result=result, provenance_substitutions=substitutions, tolerances=TOL),
            f'{name}: adjusted comparison mismatch')
    extra = independent_metrics(data[left], data[right], result)
    require(result['status'] == 'pass', f'{name}: scientific comparison failed')
    return result, extra


def source_proof(folder, config, pins):
    results = {}
    for name in ('original', 'proposed'):
        source, commit = pins[name], config['source_commits'][name]
        require(source['info'] == dict(commit=commit, status='', tracked_diff=''), 'Dirty source receipt')
        hashes = git_hashes(commit)
        require(hashes == source['tracked'], f'{name}: tracked source hash/field set mismatch')
        build = read(folder / (name + '-build.json'))
        reference_path = OUT / (name + '-reference-build.json')
        reference = read(reference_path)
        require(build['commit'] == commit and build['native'] == source['native'] == reference['native'] and
                build['version_sha256'] == source['generated_version'] and
                build['reference_build_sha256'] == sha(reference_path) and
                build['reference_commit'] == reference['commit'] and build['reference_source'] == reference['source'],
                f'{name}: native build provenance mismatch')
        changed = git('diff', '--name-only', reference['commit'], commit, '--', '*.pyx', '*.pxd',
                      '*.pxi', '*.c', '*.cpp', '*.h', '*.cu', 'setup.py', 'pycbc/lib').decode().splitlines()
        require(not changed, f'{name}: native source differs from frozen build')
        version_path = folder.parent / 'additional-source-metadata' / (name + '-version.py')
        version_sha = sha(version_path)
        require(version_sha == source['generated_version'], f'{name}: generated version bytes mismatch')
        version_fields = {}
        for node in ast.parse(version_path.read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                version_fields[node.targets[0].id] = ast.literal_eval(node.value)
        require(version_fields['git_hash'] == commit and
                version_fields['git_status'] == 'CLEAN: All modifications committed',
                f'{name}: generated version identifies another commit or dirty tree')
        results[name] = dict(commit=commit, tracked_files=len(hashes), tracked_hashes_match_git=True,
                            native_source_matches_frozen_build=True, native_hashes=source['native'],
                            generated_version_sha256=version_sha, generated_version_fields=version_fields,
                            frozen_build_receipt_sha256=sha(reference_path),
                            native_evidence='Native sources verified against Git; binary hash receipts match frozen build. Linux binary bytes absent locally.')
    return results


def verify(folder):
    before = snapshot(folder)
    metadata_folder = folder.parent / 'additional-source-metadata'
    metadata_before = snapshot(metadata_folder)
    config, pins, summary, status, dependencies = [
        read(folder / name) for name in ('config.json', 'source-pins.json', 'summary.json',
                                        'status.json', 'dependencies.json')]
    require(config['source_commits'] == {'original': ORIGINAL, 'proposed': PROPOSED}, 'Wrong source commits')
    require(config['input_pins'] == INPUT_PINS and config['repeats'] == 4 and
            config['ordering'] == ORDERING and config['core'] == 8, 'Protocol inputs/order/core changed')
    planned = read(OUT / 'audited-harness-pins.json')
    harness = read(folder / 'harness-pins.json')
    require({k: harness[k] for k in planned} == planned and
            all(before.get(k) == v for k, v in harness.items()), 'Harness differs from audited pins')
    setup = read(folder / 'setup-status.json')
    require(setup['state'] == 'complete' and setup['completed'] == ['original', 'proposed'],
            'Incomplete setup')
    require(sha(folder / 'sources.bundle') == setup['bundle_sha256'], 'Source bundle hash mismatch')
    proof = source_proof(folder, config, pins)
    expected_cases = ['qual-' + arm for arm in ARMS] + [
        f'timing-{arm}-r{repeat}' for repeat, order in enumerate(ORDERING, 1) for arm in order]
    require(status['state'] == 'complete' and status['completed'] == expected_cases and
            {p.name for p in (folder / 'runs').iterdir()} == set(expected_cases),
            'Missing/extra/out-of-order/unfinished run')
    spec = importlib.util.spec_from_file_location('audited_comparator', folder / 'compare-triggers.py')
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    require(comparator.DEFAULTS == summary['tolerances'] == TOL, 'Scientific tolerances changed')
    arrays, qualifications, qualification_proof, run_proof = {}, {}, {}, {}
    loaded, data, receipts = {}, {}, {}
    for case in expected_cases:
        path = folder / 'runs' / case
        run_proof[case], receipts[case] = validate_receipt(path, config, pins, dependencies)
        require(receipts[case]['parent_pid'] == status['pid'], 'Wrong campaign parent')
        runtime = read(path / 'runtime.json')
        require(runtime['inherited_lock']['inode'] == status['lock']['inode'], 'Benchmark lock changed')
        data[case] = datasets(path)
        loaded[case] = comparator.load(path / 'triggers.hdf')
        require(not loaded[case]['metadata']['issues'], f'{case}: comparator receipt problem')
        expected_consumed = dict(config['input_pins'])
        source_name = 'original' if receipts[case]['arm'] == 'original-cpu' else 'proposed'
        expected_consumed[receipts[case]['executable_cli'][0]] = pins[source_name]['tracked']['bin/pycbc_inspiral']
        require(loaded[case]['metadata']['consumed_input_sha256'] == expected_consumed,
                f'{case}: consumed inputs mismatch')
        loaded[case]['path'] = receipts[case]['executable_cli'][-1]
        if case.startswith('qual-'):
            arm = receipts[case]['arm']
            qualifications[arm], qualification_proof[arm] = validate_run(path, config, pins, arrays)
    for previous, following in zip(expected_cases, expected_cases[1:]):
        require(utc(receipts[previous]['finished_utc']) <= utc(receipts[following]['started_utc']),
                'Runs overlap or contradict declared balanced order')
    torch_versions = {
        (row['actual_torch_version'], row['actual_torch_cuda_version'])
        for row in run_proof.values() if row['arm'].startswith('torch')
    }
    require(len(torch_versions) == 1 and all(all(v is not None for v in row) for row in torch_versions),
            'Actual Torch/CUDA runtime versions differ across measured processes')
    require(all(row['actual_torch_version'] is None and row['actual_torch_cuda_version'] is None
                for row in run_proof.values() if not row['arm'].startswith('torch')),
            'CPU route unexpectedly reported explicit Torch setup')
    actual_torch, actual_cuda = next(iter(torch_versions))
    inventory_version = dependencies['packages'].get('torch')
    dependency_proof = dict(actual_torch_version=actual_torch, actual_torch_cuda_version=actual_cuda,
                            actual_runtime_consistent_across_all_torch_processes=True,
                            distribution_inventory_torch_version=inventory_version,
                            inventory_matches_imported_torch=inventory_version == actual_torch,
                            interpretation='Use Torch-route runtime receipts for Torch/CUDA versions. The inventory is a name-keyed distribution map and may hide duplicate installed distributions. CPU-route null version/thread fields mean the wrapper did not explicitly configure Torch; they do not prove Torch was never imported. Each stage retains the observed torch_imported flag.')
    supplement_path = metadata_folder / 'dependencies-actual.json'
    supplement = read(supplement_path)
    require(supplement['torch']['version'] == actual_torch and supplement['torch']['cuda'] == actual_cuda and
            supplement['executable'] == dependencies['executable'] and
            supplement['python'] == dependencies['python'] and
            supplement['original_flat_inventory_version'] == inventory_version,
            'Supplemental dependency/runtime identities disagree')
    require(utc(supplement['observed_utc']) >= utc(status['finished_utc']) and
            supplement['cwd'] == receipts[expected_cases[0]]['cwd'],
            'Dependency inspection was not after timing or used another cwd')
    distributions = supplement['all_torch_distributions']
    selected = supplement['selected_distribution']
    require(selected in distributions and
            selected['version'].split('+')[0] == actual_torch.split('+')[0] and
            any(d['version'] == inventory_version for d in distributions),
            'Duplicate distribution evidence does not explain flat inventory')
    module_hashes = supplement['torch']['module_sha256']
    require(len(module_hashes) == 2 and str(Path(supplement['torch']['file'])) in module_hashes and
            all(re.fullmatch('[0-9a-f]{64}', value) for value in module_hashes.values()),
            'Supplemental Torch module hash receipts missing')
    dependency_proof.update(supplemental_receipt_sha256=sha(supplement_path),
                            supplemental_inspection_after_campaign=True,
                            selected_distribution=selected, all_torch_distributions=distributions,
                            imported_torch_module_hash_receipts=module_hashes)
    qualification_comparisons, conditions, checks = {}, {}, {}
    for left, right in PAIRS:
        name = left + '-vs-' + right
        result, supplemental = compare_archived(folder, comparator, loaded, data,
                                               'qual-' + left, 'qual-' + right, name)
        qualification_comparisons[name], checks[name] = result, supplemental
        conditions[name] = conditioning(qualifications[left], qualifications[right], arrays)
        require(conditions[name] == read(folder / 'comparisons' / (name + '-conditioning.json')),
                'Full PSD/conditioning comparison mismatch')
    require(read(folder / 'qualification-summary.json') ==
            dict(comparisons=qualification_comparisons, conditioning=conditions), 'Qualification summary mismatch')
    cpu = field_comparison(data['qual-original-cpu'], data['qual-proposed-cpu'])
    scientific = {k: v['exact'] for k, v in cpu['datasets'].items() if k not in TIMING}
    runtime_fields = {k: v['exact'] for k, v in cpu['datasets'].items() if k in TIMING}
    c = conditions['original-cpu-vs-proposed-cpu']
    preserved = bool(cpu['scientific_field_sets_equal'] and scientific and all(scientific.values()) and
                     c['conditioned_strain_exact'] and c['geometry_exact'] and
                     all(p['exact'] and p['nonfinite_masks_equal'] for p in c['psds']))
    gates = all(c['conditioned_strain_exact'] and c['geometry_exact'] and
                all(p['nonfinite_masks_equal'] and p['violations'] == 0 for p in c['psds'])
                for c in conditions.values())
    require(preserved and gates and summary['cpu_preserved'] is True and
            summary['scientific_gates_pass'] is True and summary['cpu_exact_datasets'] == scientific and
            summary['runtime_metadata_exact'] == runtime_fields, 'Scientific preservation summary mismatch')
    counts = {arm: len(data['qual-' + arm]['snr']) for arm in ARMS}
    require(summary['trigger_counts'] == counts and
            summary['comparison_status'] == {k: r['status'] for k, r in qualification_comparisons.items()},
            'Qualification counts/verdicts summary mismatch')
    timing_comparisons, rows = {}, {}
    for arm in ARMS:
        cases = [f'timing-{arm}-r{r}' for r in range(1, 5)]
        for case in cases:
            result, supplemental = compare_archived(folder, comparator, loaded, data,
                                                    'qual-' + arm, case, case + '-vs-own-qualification')
            timing_comparisons[case] = dict(comparison=result, independently_recomputed=supplemental)
        values = [receipts[case]['elapsed_wall_seconds'] for case in cases]
        median = statistics.median(values)
        rows[arm] = dict(source=config['source_commits']['original' if arm == 'original-cpu' else 'proposed'],
                         scheme=SCHEMES[arm], samples_seconds=values, median_seconds=median,
                         min_seconds=min(values), max_seconds=max(values),
                         template_seconds_per_wall_second=WORK / median, trigger_count=counts[arm])
    require(summary['arms'] == rows and summary['source_commits'] == config['source_commits'] and
            summary['scope'] == config['scope'] and summary['timing_boundary'] == config['timing_boundary'] and
            summary['timing_policy'] == config['qualification_policy'] and
            summary['timing_outputs_pass'] is True and summary['equal_output_speedup_eligible'] is True and
            summary['sustained_capacity_established'] is False, 'Timing summary mismatch')
    speedup = {arm: rows['original-cpu']['median_seconds'] / rows[arm]['median_seconds'] for arm in ARMS}
    machine = read(folder / 'machine.json')
    require(machine['reservation'] == 'none; shared host and GPU' and machine['uname'][1] == 'len',
            'Unexpected host/reservation claim')
    require(before == snapshot(folder), 'Archive changed during verification')
    require(metadata_before == snapshot(metadata_folder), 'Supplemental source metadata changed during verification')
    return dict(status='pass', verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                archive=str(folder), archive_unchanged=True, archive_file_sha256=before,
                verifier_sha256=sha(__file__), source_proof=proof, harness_hashes_verified=harness,
                source_commits=config['source_commits'], external_input_hash_receipts=INPUT_PINS,
                dependencies=dependencies, machine=machine, qualification_proof=qualification_proof,
                dependency_version_proof=dependency_proof, additional_source_metadata_sha256=metadata_before,
                run_proof=run_proof, ordered_cases=expected_cases, timing_sample_count=16,
                all_declared_samples_preserved=True, h1_cpu_field_comparison=cpu,
                scientific_dataset_count=len(scientific), scientific_cpu_datasets_exact=preserved,
                timing_exclusions=['H1/' + k for k in sorted(TIMING)],
                qualification_comparisons=qualification_comparisons, independent_qualification_metrics=checks,
                conditioning=conditions, scientific_gates_pass=gates, trigger_counts=counts,
                timing_comparisons=timing_comparisons, timing_outputs_pass=True, arms=rows,
                median_speedup_relative_to_original_cpu=speedup, workload_template_seconds=WORK,
                equal_output_speedup_eligible=True, scope=config['scope'], timing_boundary=config['timing_boundary'],
                sustained_capacity_established=False,
                limitations=['Conditioned strain equality uses recorded digest/schema; raw strain is not archived.',
                             'External input and Linux binary bytes are not included; their receipts are cross-checked.',
                             'Timings include runtime verification overhead and fresh process startup; qualification runs are excluded.',
                             'Only this shared-host finite workload is measured, without host/GPU reservation.'])


def main():
    global REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, nargs='?', default=O / 'remote')
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, default=OUT / 'verification.json')
    args = parser.parse_args()
    REPO = args.repo.resolve()
    folder = args.archive.resolve()
    require(not args.output.resolve().is_relative_to(folder),
            'Verification output must be outside the read-only archive')
    result = verify(folder)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in (
        'status', 'scientific_cpu_datasets_exact', 'scientific_dataset_count', 'trigger_counts',
        'scientific_gates_pass', 'timing_sample_count', 'timing_outputs_pass',
        'median_speedup_relative_to_original_cpu', 'archive_unchanged')}))


if __name__ == '__main__':
    main()
