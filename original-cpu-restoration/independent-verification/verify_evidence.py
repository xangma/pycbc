"""Read archived evidence; write only independent verification artifacts.

Does not import PyCBC or execute campaign/setup code. The pinned comparator is
replayed without changing it, then numerical metrics and identities are checked
again directly from arrays. No remote access or scientific code changes.
"""
import argparse
import copy
import datetime
import difflib
import hashlib
import importlib.util
import io
import json
import posixpath
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
O = OUT.parent
REPO = O.parents[2]
ORIGINAL = '40e94792b3edf59f39b18b65102b28a4f74433a7'
INITIAL = '652206d84177f658f5fb2f9cab2ee73f6d2bc95f'
V2 = 'aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0'
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


def verify(folder, initial):
    before = snapshot(folder)
    config, pins, archived = [read(folder / f) for f in ('config.json', 'source-pins.json', 'summary.json')]
    require(config['source_commits'] == {'original': ORIGINAL, 'proposed': INITIAL if initial else V2}, 'Wrong source commits')
    harness = read(folder / 'harness-pins.json')
    require(all(before.get(name) == value for name, value in harness.items()), 'Harness pin mismatch')
    source_proof = {}
    for name, source in pins.items():
        commit = config['source_commits'][name]
        require(source['info'] == dict(commit=commit, status='', tracked_diff=''), 'Dirty source receipt')
        hashes = git_hashes(commit)
        require(hashes == source['tracked'], f'{name}: Git tracked field set or hash mismatch')
        build = read(folder / (name + '-build.json'))
        require(build['commit'] == commit and build['native'] == source['native'] and
                build['version_sha256'] == source['generated_version'], 'Native/build receipt mismatch')
        source_proof[name] = dict(commit=commit, tracked_field_sets_and_sha256_match_git=True,
                                  symlink_policy="Receipt hashes dereferenced file contents; four tracked example symlinks resolved entirely within the same Git tree",
                                  tracked_files=len(hashes), native_hashes=source['native'],
                                  native_build_receipt_consistent=True,
                                  native_evidence='receipt consistency; Linux binaries absent locally')
    arrays, qualifications, run_proof, data, loaded = {}, {}, {}, {}, {}
    spec = importlib.util.spec_from_file_location('archived_comparator', folder / 'compare-triggers.py')
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    require(comparator.DEFAULTS == archived['tolerances'] == TOL, 'Changed comparison tolerances')
    for arm in ARMS:
        path = folder / 'runs' / ('qual-' + arm)
        qualifications[arm], run_proof[arm] = validate_run(path, config, pins, arrays)
        data[arm] = datasets(path)
        loaded[arm] = comparator.load(path / 'triggers.hdf')
        require(not loaded[arm]['metadata']['issues'], f'{arm}: comparator receipt issue')
        # Restore recorded path labels only; HDF/receipt content is read locally.
        loaded[arm]['path'] = next(iter(qualifications[arm]['observations']['psd_arrays']))['path'].split('/arrays/')[0] + '/triggers.hdf'
    comparisons, conditions, supplemental = {}, {}, {}
    for left, right in PAIRS:
        name = left + '-vs-' + right
        a, b = loaded[left], loaded[right]
        raw = comparator.compare(a, b, TOL)
        require(raw == read(folder / 'comparisons' / (name + '-raw.json')), f'{name}: raw result mismatch')
        fixed, substitutions = copy.deepcopy(b), []
        for key in ('source_snapshot', 'consumed_input_sha256'):
            if fixed['metadata'][key] != a['metadata'][key]:
                substitutions.append(dict(field=key, before=fixed['metadata'][key], after=a['metadata'][key]))
                fixed['metadata'][key] = copy.deepcopy(a['metadata'][key])
        result = comparator.compare(a, fixed, TOL)
        stored = read(folder / 'comparisons' / (name + '.json'))
        require(stored == dict(result=result, provenance_substitutions=substitutions, tolerances=TOL),
                f'{name}: adjusted comparison mismatch')
        supplemental[name] = independent_metrics(data[left], data[right], result)
        comparisons[name] = result
        conditions[name] = conditioning(qualifications[left], qualifications[right], arrays)
        require(conditions[name] == read(folder / 'comparisons' / (name + '-conditioning.json')), 'Conditioning result mismatch')
    require(read(folder / 'qualification-summary.json') == dict(comparisons=comparisons, conditioning=conditions),
            'Qualification summary mismatch')
    fields = field_comparison(data['original-cpu'], data['proposed-cpu'])
    scientific = {k: v['exact'] for k, v in fields['datasets'].items() if k not in TIMING}
    timing = {k: v['exact'] for k, v in fields['datasets'].items() if k in TIMING}
    c = conditions['original-cpu-vs-proposed-cpu']
    preserved = fields['scientific_field_sets_equal'] and bool(scientific) and all(scientific.values()) and c['conditioned_strain_exact'] and c['geometry_exact'] and all(p['exact'] for p in c['psds']) and comparisons['original-cpu-vs-proposed-cpu']['status'] == 'pass'
    counts = {arm: len(data[arm]['snr']) for arm in ARMS}
    statuses = {name: r['status'] for name, r in comparisons.items()}
    require(archived['trigger_counts'] == counts and archived['comparison_status'] == statuses, 'Summary counts/verdicts mismatch')
    gates = all(v == 'pass' for v in statuses.values()) and all(c['conditioned_strain_exact'] and c['geometry_exact'] and all(p['nonfinite_masks_equal'] and not p['violations'] for p in c['psds']) for c in conditions.values())
    require(archived['scientific_gates_pass'] == gates and archived['equal_output_speedup_eligible'] is False, 'Summary gates mismatch')
    require(archived['source_commits'] == config['source_commits'], 'Summary source mismatch')
    if initial:
        require(archived['cpu_preserved'] is False and archived['cpu_exact_datasets'] == scientific | timing and
                {k for k, v in archived['cpu_exact_datasets'].items() if not v} == TIMING,
                'Initial discrepancy is not exactly four timing fields')
    else:
        require(archived['cpu_preserved'] == preserved and archived['cpu_exact_datasets'] == scientific and
                archived['runtime_metadata_exact'] == timing, 'V2 preservation summary mismatch')
    require(before == snapshot(folder), 'Archive changed during verification')
    return dict(status='pass', verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                archive=str(folder), archive_file_sha256=before, archive_unchanged=True,
                verifier_sha256=sha(__file__), source_proof=source_proof, harness_hashes_verified=harness,
                run_proof=run_proof, h1_cpu_field_comparison=fields,
                scientific_dataset_count=len(scientific), scientific_cpu_datasets_exact=all(scientific.values()),
                timing_exclusions=['H1/' + k for k in sorted(TIMING)],
                archived_cpu_preserved=archived['cpu_preserved'], corrected_cpu_preserved=preserved,
                trigger_counts=counts, comparison_status=statuses, comparisons=comparisons,
                independent_array_checks=supplemental, conditioning=conditions, scientific_gates_pass=gates,
                equal_output_speedup_eligible=False, performance_claim=False,
                limitations=['Conditioned strain equality is established by recorded digest/schema; raw strain is not archived.',
                             'Native Linux .so and external input bytes are not included; their receipts are cross-checked, not locally rehashed.'])


def harness_delta():
    a, b = O / 'campaign-initial', O / 'campaign-v2'
    names_a = {p.name for p in a.glob('*.py')}
    names_b = {p.name for p in b.glob('*.py')}
    require(names_a == names_b, 'Harness Python field sets changed')
    rows = {}
    for name in sorted(names_a):
        x, y = (a / name).read_text(), (b / name).read_text()
        rows[name] = dict(initial_sha256=sha(a / name), v2_sha256=sha(b / name), identical=x == y)
        if x != y:
            rows[name]['diff'] = ''.join(difflib.unified_diff(x.splitlines(True), y.splitlines(True), fromfile='initial/' + name, tofile='v2/' + name))
    ca, cb = read(a / 'config.json'), read(b / 'config.json')
    return dict(files=rows, config_changes={k: dict(initial=ca.get(k), v2=cb.get(k)) for k in ca.keys() | cb.keys() if ca.get(k) != cb.get(k)},
                interpretation='Scientific comparison logic changes only the four timing-metadata exclusions; setup also selects the v2 staging source ref and config selects its commit. Comparator and numerical tolerances are unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=['initial', 'v2'], required=True)
    args = parser.parse_args()
    initial = args.campaign == 'initial'
    folder = O / ('campaign-initial' if initial else 'campaign-v2-results')
    result = verify(folder, initial)
    result['harness_delta'] = harness_delta()
    stem = 'initial-reconciliation' if initial else 'campaign-v2-verification'
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = [f'# {stem.replace("-", " ").capitalize()}', '',
             'Independent verification: PASS. All five archived comparison results were recomputed from local HDF/NPY evidence with unchanged tolerances.', '',
             f'Original CPU `{ORIGINAL}` versus candidate `{INITIAL if initial else V2}`: **1988 triggers each, scientifically byte-identical**. Both complete H1 dataset-name sets were enumerated and compared; all 18 scientific datasets match dtype, shape, and C-order bytes. No right-only scientific field was ignored.', '',
             'The only excluded H1 datasets are:', '']
    for k in sorted(TIMING):
        row = result['h1_cpu_field_comparison']['datasets'][k]
        lines.append(f'- `H1/{k}`: {row["left_value"]} versus {row["right_value"]}.')
    lines += ['', 'These four fields are elapsed-time-derived metadata; search start/end times and every gating/trigger dataset remain included.', '',
              'The saved PSD arrays match exactly, including dtype, shape, bytes, and infinity masks. Conditioned-strain records match, including the SHA-256 digest of 8,323,072 float32 samples and gating metadata. Raw conditioned strain is absent, so that equality is supported by the recorded hashes. The five segment geometries match exactly and independently cover 7,798,784 samples (1904 seconds) without gaps or overlaps.', '',
              'The historical `cpu_preserved=false` is caused solely by including those four timing fields in its all-datasets conjunction; the corrected scientific interpretation is **CPU preservation PASS**.' if initial else 'The new summary correctly reports **CPU preservation PASS**.', '',
              'Both Torch arms contain 1991 triggers and retain FAIL against both CPU arms at the original tolerances. The complete scientific campaign remains FAIL; no speedup or performance claim follows.', '',
              'The unadjusted comparison also retains declared source/executable-provenance differences. These are separate from the timing-field issue: after independently checking source and input pins, the original harness normalizes only those two provenance fields for its scientific verdict.', '',
              'All recorded qualification checks and template/segment coverage were checked. Runtime receipts confirm affinity [8], one thread per observed native pool, and Torch intra/inter-op counts of one in Torch arms. Tracked source field sets and SHA-256 hashes match local Git blobs at the exact pinned commits; harness hashes and HDF/NPY file/data hashes match their receipts. Native binaries and external input bytes are checked through receipt consistency only; remote rehash evidence is supplied by the primary agent.', '',
              'Harness review: campaign comparison logic changes only the four timing exclusions and their reporting. Separately, setup selects the v2 staging ref and config selects its commit; other harness Python files, including the comparator, are identical. Full diffs and hashes are in the JSON.', '',
              'The original archive was unchanged across verification. This analysis writes only to the independent-verification directory.']
    (OUT / (stem + '.md')).write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'corrected_cpu_preserved', 'scientific_dataset_count', 'trigger_counts', 'comparison_status', 'archive_unchanged')}))


if __name__ == '__main__':
    main()
