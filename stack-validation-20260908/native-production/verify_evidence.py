#!/usr/bin/env python3
"""Verify archived production qualification without Linux, CUDA or PyCBC.

Requires Python 3.11+, numpy and h5py. All reads are relative to this file;
original remote paths remain unchanged in receipts. No workloads are rerun.
"""
if not __debug__:
    raise RuntimeError('Run without -O/-OO or PYTHONOPTIMIZE; validation assertions must remain enabled')

import copy
import gzip
import hashlib
import json
from pathlib import Path
import posixpath
import tarfile
from types import ModuleType
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
REMOTE = Path('/home/xangma/pycbc-torch-residual-production-20260908/native-v1')
SOURCE = REMOTE / 'source'
GENERATED_VERSION = dict(
    path='/home/xangma/pycbc-torch-split-20260905/validation/pycbc/version.py',
    bytes=850, sha256='fc9141167333f201ed01fcdc4e383e9e3e505fd98d889fc6160fe4bb3ab06184')


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def module(name, filename):
    path = ROOT / filename
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == read('manifest.json')[filename]
    result = ModuleType(name)
    result.__file__ = str(path)
    # Compile pinned source directly: do not execute untracked cached bytecode.
    exec(compile(source, str(path), 'exec'), result.__dict__)
    return result


def verify_source(pins):
    with gzip.open(ROOT / 'source.tar.gz', 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == pins['archives']['source.tar']
    # Inspect members directly; never extract archive paths or symlinks.
    actual = {}
    with tarfile.open(ROOT / 'source.tar.gz', 'r:gz') as archive:
        for item in archive:
            assert not item.name.startswith('/') and '..' not in Path(item.name).parts
            if item.isdir():
                continue
            assert item.name not in actual and (item.isfile() or item.issym()), item.name
            if item.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(item.name), item.linkname))
                assert not target.startswith('/') and '..' not in Path(target).parts
                assert archive.getmember(target).isfile()
            actual[item.name] = hashlib.file_digest(archive.extractfile(item), 'sha256').hexdigest()
    assert actual == pins['tracked_files']


def verify_native():
    status = read('native-status.json')
    assert status['state'] == 'complete' and status['source_unchanged']
    expected = {'contracts': (43, 0), 'regression': (292, 52)}
    assert {row['name'] for row in status['runs']} == set(expected)
    for row in status['runs']:
        assert row['returncode'] == 0
        cases = ET.parse(ROOT / (row['name'] + '.xml')).findall('.//testcase')
        skipped = [case.find('skipped') for case in cases if case.find('skipped') is not None]
        assert (len(cases), len(skipped)) == expected[row['name']]
        assert not any(case.find('error') is not None or case.find('failure') is not None for case in cases)
        assert all(case.get('message') == 'Torch MPS device unavailable' for case in skipped)
        assert row['counts'] == dict(tests=len(cases), skipped=len(skipped), errors=0, failures=0)
    assert read('pending-allocation-result.json')['rows'] == [
        dict(protected=True, pending_at_mutation=True, original_pointer_reused=False, sentinel_preserved=True),
        dict(protected=False, pending_at_mutation=True, original_pointer_reused=True, sentinel_preserved=False)]


def verify_runtime(name, scheme, pins):
    receipt = read(f'runs/{name}/receipt.json')
    runtime = read(f'runs/{name}/runtime.json')
    assert receipt['source_info'] == dict(commit=pins['commit'], status='', tracked_diff='')
    assert receipt['state'] == 'complete' and receipt['returncode'] == 0
    assert receipt['source_status_after'] == ''
    expected_inputs = dict(read('input-pins.json'), **{str(SOURCE / 'bin/pycbc_inspiral'): pins['tracked_files']['bin/pycbc_inspiral']})
    assert receipt['input_sha256'] == receipt['input_sha256_after'] == expected_inputs
    assert receipt['runtime_sha256'] == sha(ROOT / f'runs/{name}/runtime.json')
    assert receipt['trigger_sha256'] == sha(ROOT / f'runs/{name}/triggers.hdf')
    assert runtime['state'] == 'complete' and runtime['processing_scheme'] == scheme
    assert runtime['source'] == runtime['imported_source'] == str(SOURCE)
    config = read('config-graph.json' if name == 'cuda' else 'config.json')
    checker = module('runtime_checker', 'checked-inspiral.py')
    environment = checker.fixed_environment(config, SOURCE)
    assert receipt['environment'] == runtime['environment'] == environment
    # Only the helper's archive location changes; its bytes/version must match.
    checker.expected_threadpoolctl = lambda: dict(path=str(REMOTE / 'threadpoolctl.py'),
                                                 version='3.6.0', sha256=sha(ROOT / 'threadpoolctl.py'))
    for stage in ('before_executable', 'at_first_bank', 'after_executable'):
        checker.check(runtime[stage], environment, bank=stage == 'at_first_bank', scheme=scheme)
    assert runtime['helper_sha256'] == sha(ROOT / 'checked-inspiral.py')
    assert runtime['inherited_lock']['inode'] == read('science-status-v3.json')['lock']['inode']
    extensions = read('native-extension-pins.json')
    for path, value in runtime['source_modules'].items():
        relative = str(Path(path).relative_to(SOURCE))
        expected = extensions[relative]['sha256'] if path.endswith('.so') else pins['tracked_files'][relative]
        assert value == expected


def verify_qualification(name, scheme, pins):
    q = read(f'runs/{name}/qualification.json')
    assert q['status'] == 'success' and q['executable_exit_code'] == 0
    assert q['checks'] and all(q['checks'].values())
    assert q['source_root'] == str(SOURCE)
    assert q['executable']['sha256'] == pins['tracked_files']['bin/pycbc_inspiral']
    modules = q['source_modules']
    assert {'pycbc.filter.matchedfilter', 'pycbc.types.array_torch', 'pycbc.version'} <= modules.keys()
    if name == 'cuda':
        assert 'pycbc.filter._torch_cuda_graph' in modules
    assert sha(ROOT / 'generated-version.py') == GENERATED_VERSION['sha256']
    assert (ROOT / 'generated-version.py').stat().st_size == GENERATED_VERSION['bytes']
    extensions = read('native-extension-pins.json')
    for module_name, record in modules.items():
        if module_name == 'pycbc.version':
            assert record == GENERATED_VERSION
            continue
        path = Path(record['path'])
        relative = str(path.relative_to(SOURCE))
        expected = extensions[relative]['sha256'] if path.suffix == '.so' else pins['tracked_files'][relative]
        assert record['sha256'] == expected, module_name
    obs = q['observations']
    bank, = obs['banks']
    assert bank['selected_template_count'] == 384 and bank['expected_filter_calls'] == 1920
    assert bank['file']['sha256'] == read('input-pins.json')[bank['file']['path']]
    assert obs['parsed_options']['chisq_bins'] == '16' and obs['parsed_options']['cluster_window'] == 1.0
    control, = obs['matched_filter_controllers']
    assert control['segment_count'] == 5
    assert [control['filter_bin_start'], control['filter_bin_stop']] == [15360, 1048576]
    psds = []
    for row in obs['psd_arrays']:
        path = ROOT / 'runs' / name / row['relative_path']
        assert row['path'] == str(REMOTE / 'runs' / name / row['relative_path'])
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == row['shape'] and array.dtype.str == row['dtype_str']
        assert array.nbytes == row['nbytes']
        assert hashlib.sha256(array.tobytes()).hexdigest() == row['data_sha256']
        assert row['validity']['valid_for_filter']
        psds.append({key: value for key, value in row.items() if key != 'path'})
    assert dict(conditioned_strain=obs['conditioned_strain'], psd_arrays=psds,
                segment_geometry=obs['segment_geometry']) == read('science-reference.json')['by_scheme'][scheme]
    if name != 'cuda':
        return
    graph = obs['offline_graph']
    assert len(graph['calls']) == graph['replays'] == 1920
    assert len(graph['captures']) == len(graph['final_bindings']) == 5
    assert graph['capture_contexts'] == list(range(5))
    assert [(row['template'], row['segment']) for row in graph['calls']] == [
        ([0, template], segment) for template in range(384) for segment in range(5)]
    assert all(all(row[key] for key in ('full_bytes_equal', 'sparse_bytes_equal',
               'inputs_unchanged', 'binding_unchanged')) for row in graph['calls'])
    assert graph['retained_nonempty'] == sum(not row['empty'] for row in graph['calls']) > 1
    offsets = [row['analyzed_sample_interval'][0] for row in obs['segment_geometry'][0]['segments']]
    for row in graph['calls']:
        if row['empty']:
            continue
        consumer = row['consumer']
        assert consumer['verified'] and consumer['offset'] == offsets[row['segment']]
        assert consumer['stage'] == 'complete' and consumer['transition_count'] == 1
        assert consumer['observed_after'] == consumer['expected_after']
        assert consumer['observed_versions_after'] == consumer['expected_versions_after']
        assert consumer['expected_versions_after'] == [consumer['versions_before'][0] + 1, consumer['versions_before'][1]]
        assert consumer['before'] == row['sparse']


def verify_comparison(key, left, right, pins, exact):
    comparator = module('trigger_comparator', 'compare-triggers.py')
    saved = read('v2-' + key + '.json')
    baseline, candidate = [comparator.load(ROOT / folder / 'triggers.hdf') for folder in (left, right)]
    # Restore provenance display paths only, to reproduce the report verbatim.
    baseline['path'], candidate['path'] = saved['raw_result']['baseline'], saved['raw_result']['candidate']
    for folder, loaded in ((left, baseline), (right, candidate)):
        receipt = read(folder + '/receipt.json')
        assert not loaded['metadata']['issues']
        executable = receipt['executable_cli'][0]
        assert loaded['metadata']['consumed_input_sha256'] == dict(read('input-pins.json'), **{executable: pins['tracked_files']['bin/pycbc_inspiral']})
        assert receipt['source_info']['status'] == receipt['source_info']['tracked_diff'] == ''
    raw = comparator.compare(baseline, candidate, comparator.DEFAULTS)
    assert raw == saved['raw_result'] == read('v2-' + key + '-raw.json')
    fixed = copy.deepcopy(candidate)
    substitutions = []
    for field in ('source_snapshot', 'consumed_input_sha256'):
        before, after = fixed['metadata'][field], baseline['metadata'][field]
        if before == after:
            continue
        if field == 'consumed_input_sha256':
            old, new = read(right + '/receipt.json')['executable_cli'][0], read(left + '/receipt.json')['executable_cli'][0]
            renamed = dict(before)
            assert old != new and renamed[old] == after[new]
            renamed[new] = renamed.pop(old)
            assert renamed == after
        substitutions.append(dict(field=field, before=before, after=after))
        fixed['metadata'][field] = copy.deepcopy(after)
    assert substitutions == saved['provenance_substitutions']
    assert saved['tolerances'] == comparator.DEFAULTS
    result = comparator.compare(baseline, fixed, comparator.DEFAULTS)
    assert result == saved['result'] and result['status'] == 'pass'
    assert result['detectors']['H1']['matched_count'] == 1991
    if exact:
        certificates = {}
        with h5py.File(ROOT / left / 'triggers.hdf') as a, h5py.File(ROOT / right / 'triggers.hdf') as b:
            datasets = []
            for handle in (a, b):
                paths = {}
                handle['H1'].visititems(lambda key, value: paths.update({key: value}) if isinstance(value, h5py.Dataset) else None)
                datasets.append(paths)
            assert set(datasets[0]) == set(datasets[1])
            telemetry = {'search/run_time', 'search/filter_rate_per_core', 'search/setup_time_fraction', 'search/templates_per_core'}
            assert telemetry <= datasets[0].keys()
            keys = sorted(datasets[0].keys() - telemetry)
            assert len(keys) == 18
            for key in keys:
                av, bv = datasets[0][key][()], datasets[1][key][()]
                assert av.dtype == bv.dtype and av.shape == bv.shape and av.tobytes() == bv.tobytes(), key
                certificates[key] = dict(shape=list(av.shape), dtype=av.dtype.str, sha256=hashlib.sha256(av.tobytes()).hexdigest())
        assert certificates == saved['exact_H1']


def main():
    manifest = read('manifest.json')
    actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json'}
    assert actual == set(manifest)
    for name, expected in manifest.items():
        assert sha(ROOT / name) == expected, name
    for filename in ('science-manifest.json', 'science-manifest-v2.json', 'science-manifest-v3.json', 'science-reuse-pins.json', 'negative-reuse-pins.json', 'remote-output-pins.json'):
        for name, expected in read(filename).items():
            assert sha(ROOT / name) == expected, (filename, name)
    pins = read('source-pins.json')
    verify_source(pins)
    verify_native()
    status = read('science-status-v3.json')
    assert status['state'] == 'complete' and status['source_unchanged']
    assert [row['name'] for row in status['runs']] == ['standard', 'cpu', 'cuda']
    for name, reference in read('science-reference-pins.json').items():
        for path, expected in reference['sha256'].items():
            assert sha(ROOT / 'references' / name / path) == expected
    negative = read('negative-result-v3.json')
    q = read('runs/cuda-negative/qualification.json')
    assert negative['detected_suppressed_replay'] and negative['returncode'] != 0
    assert q['status'] == 'failed' and q['error']['type'] == 'AssertionError'
    assert q['error']['message'] == 'Full correlation/SNR byte mismatch'
    assert q['source_root'] == str(SOURCE) and q['executable']['sha256'] == pins['tracked_files']['bin/pycbc_inspiral']
    assert negative['qualification_sha256'] == sha(ROOT / 'runs/cuda-negative/qualification.json')
    assert negative['log_sha256'] == sha(ROOT / 'runs/cuda-negative/stderr.log')
    assert negative['original_result_sha256'] == sha(ROOT / 'negative-result.json')
    graph = q['observations']['offline_graph']
    assert graph['replays'] == negative['replays'] == 6
    assert len(graph['calls']) == negative['completed_comparisons'] == 5
    assert len(graph['captures']) == negative['captures'] == 5
    expected_comparisons = set()
    for name, scheme in [('standard', 'cpu:1'), ('cpu', 'torch:cpu:1'), ('cuda', 'torch:cuda:0')]:
        verify_runtime(name, scheme, pins)
        verify_qualification(name, scheme, pins)
        key = name + '-vs-own-reference'
        verify_comparison(key, 'references/' + name, 'runs/' + name, pins, True)
        expected_comparisons.add(key)
        if name != 'standard':
            key = name + '-vs-standard'
            verify_comparison(key, 'runs/standard', 'runs/' + name, pins, False)
            expected_comparisons.add(key)
    assert set(status['comparisons']) == expected_comparisons
    for key, value in status['comparisons'].items():
        assert value['sha256'] == sha(ROOT / ('v2-' + key + '.json'))
    print(json.dumps(dict(status='pass', source_commit=pins['commit'], manifest_files=len(manifest),
                          native_passed=283, native_skipped=52, scientific_schemes=3,
                          triggers_per_scheme=1991, exact_H1_datasets_per_reference=18,
                          production_graph_replays=1920, negative_controls=2)))


if __name__ == '__main__':
    main()
