"""Independent offline parent review; no remote access or science execution."""
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
E = ROOT / 'executable-remote-evidence'
REMOTE = '/home/xangma/pycbc-torch-python-optimization-r4-20260907'
BASE = '9578a710479b924e882857c4dffab6ed372a634b'
CAND = '9e6a688a5190d6e1ddc655fbe352cc206085d5c6'
DIFF = '06c76d2bd544f9ac96fbb407356e59bb7ef675a1edc47860f9470cf0e92f3b09'
EXEC = 'd4af378d77aa5f66bcc018db32fe372360e53b22542e539ea3a2e53d31d8fa8a'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


manifest = read(ROOT / 'FINAL-MANIFEST.json')
for name, expected in manifest.items():
    assert sha(ROOT / name) == expected, name
audit = read(ROOT / 'executable-terminal-audit.json')
assert sha(ROOT / 'executable-terminal-audit.json') == '7810dbe5d24eb285ff83a0d1f60469bb54c1686804a3ea7312274f5b8b2dd954'
assert audit['state'] == 'complete' and not audit['active_owned']
assert audit['shared_lock_reacquired'] and audit['source_unchanged']
assert len(audit['evidence_sha256']) == 190
assert {str(p.relative_to(E)) for p in E.rglob('*') if p.is_file()} == set(audit['evidence_sha256'])
for name, expected in audit['evidence_sha256'].items():
    assert sha(E / name) == expected, name
status = read(E / 'executable-status.json')
summary = read(E / 'executable-summary.json')
assert sha(E / 'executable-summary.json') == status['result_sha256'] == 'f21b5db87509c9e94701bcb050f97d3a0d104986df40bf51c47af3c2df06ce66'
assert sha(E / 'executable-status.json') == '55015e8cc2fde4214f4e2e38cdb402f6807c993c2d3087a09d4bb66086b73727'
assert status['state'] == 'complete' and status['source_unchanged']
assert status['child_pid'] is None and summary['science_parity'] == 'pass'
before = read(E / 'executable-before.json')
for later in [read(E / 'executable-after-qualification.json'), read(E / 'executable-after.json'), audit['fresh_source_snapshot']]:
    for key in ['changed_files', 'diff_sha256', 'helpers', 'input_sha256', 'sources']:
        assert later[key] == before[key], key
assert before['diff_sha256'] == DIFF
assert before['changed_files'] == ['pycbc/types/array_torch.py', 'test/test_torch_squared_norm.py']
assert before['sources']['baseline']['native_sha256'] == before['sources']['candidate']['native_sha256']
assert len(before['sources']['baseline']['native_sha256']) == 11
for role, rev in [('baseline', BASE), ('candidate', CAND)]:
    src = before['sources'][role]
    assert src['source_snapshot'] == dict(commit=rev, status='', tracked_diff='')
    assert src['executable_sha256'] == EXEC

spec = importlib.util.spec_from_file_location('frozen_comparator', E / 'compare-triggers.py')
frozen = importlib.util.module_from_spec(spec)
assert sha(E / 'compare-triggers.py') == 'f0af115a2bf2d3a5a85152cb1f61d9b7570a460efd6d420566c5a2b74ac8f1b5'
spec.loader.exec_module(frozen)
loaded, receipts, order, pids = {}, {}, [], []
for role, rev in [('baseline', BASE), ('candidate', CAND)]:
    q = read(E / 'runs' / ('qualify-' + role) / 'qualification.json')
    assert q['status'] == 'success' and all(q['checks'].values())
    obs = q['observations']
    bank, = obs['banks']
    assert bank['selected_template_count'] == 384 and bank['enable_compressed_waveforms']
    assert set(bank['templates']) == {str(i) for i in range(384)}
    for t in bank['templates'].values():
        assert t['getitem_successes'] == t['decompression_successes'] == 1
        assert t['filter_successes_by_segment'] == {str(i): 1 for i in range(5)}
    geometry, = obs['segment_geometry']
    assert geometry['gap_samples'] == geometry['overlap_samples'] == 0
    assert geometry['unique_analyzed_seconds'] == 1904 and len(geometry['segments']) == 5
    engine, = obs['fft_engines']
    assert engine['execute_attempts'] == engine['execute_successes'] == 1920
    values = []
    for name in ['qualify-' + role] + [role + '-r' + str(i) for i in (1, 2, 3)]:
        folder = E / 'runs' / name
        receipt = receipts[name] = read(folder / 'receipt.json')
        runtime = read(folder / 'runtime.json')
        assert receipt['state'] == runtime['state'] == 'complete' and receipt['returncode'] == 0
        assert receipt['source_info'] == before['sources'][role]['source_snapshot']
        assert receipt['input_sha256'] == receipt['input_sha256_after']
        assert receipt['runtime_sha256'] == sha(folder / 'runtime.json')
        assert receipt['trigger_sha256'] == sha(folder / 'triggers.hdf')
        assert runtime['module_sha256'] == before['sources'][role]['module_sha256']
        assert runtime['source'] == runtime['imported_source'] == before['sources'][role]['path']
        assert runtime['environment'] == receipt['environment']
        assert runtime['torch_version'] == '2.13.0+cu130'
        start = datetime.fromisoformat(receipt['started_utc']).timestamp()
        end = datetime.fromisoformat(receipt['finished_utc']).timestamp()
        times = [runtime['started_at']]
        for phase in ['before_executable', 'at_first_bank', 'after_executable']:
            state = runtime[phase]
            assert state['intra_op'] == state['inter_op'] == 1
            assert state['affinity'] == [8] and state['smt_siblings'] == {'8': '8,72'}
            assert all(pool['num_threads'] == 1 for pool in state['threadpools'])
            times.append(state['observed_at'])
        times.append(runtime['finished_at'])
        assert [start] + times + [end] == sorted([start] + times + [end])
        pids.append(runtime['pid'])
        loaded[name] = frozen.load(folder / 'triggers.hdf')
        loaded[name]['path'] = REMOTE + '/runs/' + name + '/triggers.hdf'
        assert not loaded[name]['metadata']['issues']
        if name.startswith(role):
            assert receipt['mode'] == 'timing'
            values.append(receipt['elapsed_wall_seconds'])
            order.append((start, end, name))
            metric = summary['metrics'][role][len(values)-1]
            assert metric['receipt_sha256'] == sha(folder / 'receipt.json')
            assert metric['full_wall_seconds'] == values[-1]
            assert metric['templates'] == 384 and metric['inferred_segments'] == 5
    assert summary['full_wall'][role] == dict(seconds=values, median=statistics.median(values), minimum=min(values), maximum=max(values))
assert len(set(pids)) == 8
order.sort()
assert [n for _, _, n in order] == ['baseline-r1', 'candidate-r1', 'candidate-r2', 'baseline-r2', 'baseline-r3', 'candidate-r3']
assert all(a[1] <= b[0] for a, b in zip(order, order[1:]))
for role in ['baseline', 'candidate']:
    saved = read(E / (role + '-repeat-parity.json'))
    for i in (1, 2, 3):
        result = frozen.compare(loaded['qualify-' + role], loaded[role + '-r' + str(i)], frozen.DEFAULTS)
        assert result['status'] == 'pass' and result == saved['comparisons'][i-1]
cases = [('qualification-parity', 'qualify-baseline', 'qualify-candidate')]
cases += [('timing-parity-r' + str(i), 'baseline-r' + str(i), 'candidate-r' + str(i)) for i in (1, 2, 3)]
for name, base, candidate in cases:
    raw = frozen.compare(loaded[base], loaded[candidate], frozen.DEFAULTS)
    assert raw == read(E / name / 'raw-strict-comparison.json')
    assert raw['status'] == 'fail'
    assert raw['failures'] == ['configuration mismatch: consumed_input_sha256', 'configuration mismatch: source_snapshot']
    items = copy.deepcopy([loaded[base], loaded[candidate]])
    for item, case in zip(items, [base, candidate]):
        metadata = item['metadata']
        executable = receipts[case]['executable_cli'][0]
        hashes = metadata['consumed_input_sha256']
        assert hashes == dict(before['input_sha256'], **{executable: EXEC})
        del hashes[executable]
        hashes['role:byte-identical-bin/pycbc_inspiral'] = EXEC
        metadata['source_snapshot'] = dict(authorized_pair=[BASE, CAND], diff_sha256=DIFF)
    result = frozen.compare(*items, frozen.DEFAULTS)
    saved = read(E / name / 'cross-revision-comparison.json')
    assert result == saved['comparison'] and result['status'] == saved['status'] == 'pass'
    assert saved['tolerances'] == frozen.DEFAULTS
    detector = result['detectors']['H1']
    assert detector['matched_count'] == detector['baseline_count'] == detector['candidate_count'] == 1991
    assert len(detector['metrics']) == 11
    assert all(m['max_absolute_error'] == m['violations'] == 0 for m in detector['metrics'].values())
wall = summary['full_wall']
ratio = wall['baseline']['median'] / wall['candidate']['median']
assert ratio == summary['ratio'] == audit['ratio']
assert wall['candidate']['maximum'] < wall['baseline']['minimum']
result = dict(state='pass', sealed_files=len(manifest), remote_files=190,
              source_snapshot_pins=True, eight_fresh_workers=True,
              exact_strict_comparisons=4, exact_wrapped_science=4,
              within_revision_comparisons=6, matched_triggers=1991,
              zero_error_fields=11, sequential_order=[n for _, _, n in order],
              full_wall=wall, ratio=ratio,
              percent_less_wall=100*(1-1/ratio), remote_work_performed=False)
(ROOT / 'parent-executable-review.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
