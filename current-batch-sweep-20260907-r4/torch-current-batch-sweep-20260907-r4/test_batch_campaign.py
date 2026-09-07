"""Controls tests use local temporary processes; no PyCBC or GPU is imported."""
import importlib.util
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('batch_campaign', HERE / 'batch-campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def done():
    return dict(state='complete', pid=123, finished=42, converged=True, selected_templates=6144)


def test_dependency_waits_until_predecessor_exits():
    assert not c.dependency_ready({'state': 'running'})
    assert not c.dependency_ready(done(), {'state': 'S', 'start_ticks': '12'})
    assert c.dependency_ready(done())
    assert c.dependency_ready(done(), {'state': 'Z', 'start_ticks': '12'})


@pytest.mark.parametrize('state', ['failed', 'cancelled', 'invalid-input-mutation'])
def test_dependency_failure_stops_new_work(state):
    with pytest.raises(ValueError, match='failed'):
        c.dependency_ready({'state': state})


@pytest.mark.parametrize('field', ['finished', 'pid', 'converged', 'selected_templates'])
def test_incomplete_final_state_rejected(field):
    state = done()
    del state[field]
    with pytest.raises(ValueError, match='lacks'):
        c.dependency_ready(state)


def test_timing_matrix_has_three_fresh_replicates_per_route_and_batch():
    order = c.timing_order()
    assert len(order) == len(set(order)) == 54
    assert set(order) == {(r, b, route) for r in (1, 2, 3)
                          for b in c.BATCHES for route in c.ROUTES}
    assert order[:3] == [(1, 1, route) for route in c.ROUTES]
    assert order[18][1] != order[0][1]


def test_single_thread_environment_removes_opt_in_acceleration(monkeypatch):
    monkeypatch.setenv('PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK', '1')
    monkeypatch.setenv('PYCBC_BATCH_MAXELEMENTS', '5')
    monkeypatch.setenv('OMP_NUM_THREADS', '64')
    env = c.environment()
    assert not any(name.startswith('PYCBC_TORCH_') for name in env)
    assert 'PYCBC_BATCH_MAXELEMENTS' not in env
    assert all(env[name] == '1' for name in c.THREADS)
    assert env['MKL_DYNAMIC'] == 'FALSE'
    assert env['PYTHONPATH'] == str(c.SOURCE)


@pytest.mark.parametrize('value', [{}, {'status': 'failed'}, {'status': 'review'}])
def test_bad_science_outcomes_are_not_promoted(value):
    with pytest.raises(ValueError):
        c.require_qualification(value)


def evidence():
    policy = __import__('runpy').run_path(str(HERE / 'batch-worker.py'))['POLICY']
    def metric(samples):
        return dict(samples=samples, failed_samples=0, nonfinite_actual=0,
                    nonfinite_reference=0, max_abs_error=0, max_tolerance_ratio=0)

    def normalization(source):
        fields = dict(norm=1.0, sigmasq=1.0, source=source)
        return {**fields, 'sha256': c.canonical_hash(fields)}

    rows = [{**metric(32768), 'template_id': 1000+i, 'sha256': 'a'*64,
             'reference_sha256': 'a'*64,
             'oracle': {'sha256': 'b'*64,
                        'normalization': normalization('independent_float64_power_sum')},
             'normalization': normalization('observed_scalar_sigmasq_callback'),
             'normalized_oracle': metric(32768),
             'normalized_mkl_compatibility': metric(32768),
             'sigmasq_oracle': {'passed': True, 'relative_error': 0.0},
             'raw_complex_v1_audit': {**metric(32768),
                                      'classification': 'audit_only_not_v2_acceptance'},
             'normwise_diagnostics': {'classification': 'diagnostic_only_no_threshold'}}
            for i in range(8)]
    pointwise = [{**metric(8*32768), 'block': b, 'processed_once': True,
                  'processing_counts': [1]*8, 'rows': copy.deepcopy(rows),
                  'normalization_failed_rows': 0,
                  'normalized_oracle': metric(8*32768),
                  'normalized_mkl_compatibility': metric(8*32768)} for b in range(3)]
    return {'schema_version': 1, 'route': 'branch_standard', 'batch': 8,
            'mode': 'qualify', 'expected_head': c.REVISION,
            'arguments': {'seed': c.PLAN['seed']},
            'tolerance_policy': policy, 'policy_sha256': c.canonical_hash(policy),
            'source': {'revision': c.REVISION, 'tracked_dirty': False},
            'worker_sha256': c.control.digest(HERE / 'batch-worker.py'),
            'inputs': {'seed': c.PLAN['seed'], 'geometry': {'bank_templates': 8, 'fft_samples': 32768, 'blocks': 3}},
            'input_sha256': 'fixed-input', 'thread_environment': {'OMP_NUM_THREADS': '1'},
            'affinity': [8],
            'group_layout': [list(range(1000, 1008))],
            'fft_plans': [{'nbatch': 8, 'class': 'pycbc.fft.mkl.IFFT'}],
            'status': 'pass', 'failures': [], 'exit_code': 0,
            'trigger_blocks': [{'count': 4} for _ in range(3)],
            'trigger_comparisons': [],
            'pointwise_blocks': pointwise,
            'oracle': dict(created=True, directory='reference', filename='oracle.npy',
                           dtype='complex128', shape=[3, 8, 32768]),
            'reference': dict(created=True, directory='reference', filename='outputs.npy',
                              dtype='complex64', shape=[3, 8, 32768])}


def test_pass_flag_cannot_hide_missing_work_or_wrong_backend():
    result = evidence()
    c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)
    result['pointwise_blocks'][2]['processing_counts'][0] = 0
    with pytest.raises(ValueError, match='coverage'):
        c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)
    result = evidence()
    result['fft_plans'][0]['class'] = 'pycbc.fft.fftw.IFFT'
    with pytest.raises(ValueError, match='MKL'):
        c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)


@pytest.mark.parametrize('times', [[float('nan'), 1, 1], [0, 1, 1], [-1, 1, 1], [1, 1]])
def test_invalid_timing_is_rejected(times):
    result = evidence()
    result.update(mode='timing', timing={'instrumented': False, 'templates_per_iteration': 24,
                                        'warm_block_ms': [times] * 5,
                                        'warm_iteration_ms': [sum(times)] * 5})
    with pytest.raises(ValueError, match='timing'):
        c.validate_result(result, 'branch_standard', 8, 'timing', smoke=True)


def test_summary_requires_all_cells_and_uses_constant_bank_denominator():
    timings = [{'repeat': r, 'batch': b, 'route': route,
                'result': {'timing': {'warm_iteration_ms': [2000.] * 5},
                           'inputs': {'geometry': {'valid_end': 126976, 'valid_start': 12288,
                                                  'sample_rate': 2048}}}}
               for r, b, route in c.timing_order()]
    with pytest.raises(ValueError, match='Incomplete'):
        c.summarize(timings[:-1])
    summary = c.summarize(timings)
    assert len(summary['cells']) == 18
    for cell in summary['cells']:
        assert cell['median_templates_per_second'] == 1536
        assert cell['median_template_seconds_per_second'] == 1536 * 56
        assert cell['templates_per_second_range'] == [1536, 1536]


def runner(tmp_path, timeout=5):
    (tmp_path / 'stages').mkdir()
    return c.Campaign(tmp_path, timeout)


def test_stage_logs_and_failed_worker_are_retained(tmp_path):
    campaign = runner(tmp_path)
    campaign.stage('pass', [sys.executable, '-c', 'print("passed")'])
    assert (tmp_path / 'stages/pass.log').read_text().strip() == 'passed'
    with pytest.raises(RuntimeError):
        campaign.stage('failure', [sys.executable, '-c', 'raise SystemExit(7)'])
    receipt = c.control.read(tmp_path / 'stages/failure.json')
    assert receipt['returncode'] == 7 and receipt['state'] == 'failed'
    assert campaign.state['child_pid'] is None
    assert (tmp_path / 'batch-status.json').exists()


HEARTBEAT = '''import pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    pathlib.Path('heartbeat').write_text(str(time.time_ns()))
    time.sleep(0.02)
'''


def child_command(exit_early=False):
    code = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {HEARTBEAT!r}]); time.sleep(0.2)'
    code += '; raise SystemExit(3)' if exit_early else '; time.sleep(120)'
    return [sys.executable, '-c', code]


@pytest.mark.parametrize('exit_early', [False, True])
def test_failure_and_timeout_remove_owned_descendants(tmp_path, exit_early):
    campaign = runner(tmp_path, timeout=0.6)
    with pytest.raises(RuntimeError if exit_early else subprocess.TimeoutExpired):
        campaign.stage('descendant', child_command(exit_early))
    heartbeat = tmp_path / 'heartbeat'
    previous = heartbeat.read_text()
    time.sleep(0.15)
    assert heartbeat.read_text() == previous
    assert campaign.state['child_pid'] is None


def test_wait_timeout_launches_no_worker(tmp_path):
    path = tmp_path / 'dependency.json'
    c.control.save(path, {'state': 'running'})
    campaign = c.Campaign(tmp_path)
    with pytest.raises(TimeoutError):
        campaign.wait_dependency(path, timeout=0)
    assert campaign.state['completed'] == []


def test_signal_cancels_only_new_campaign_and_cleans_child(tmp_path):
    harness = f'''import sys
sys.path.insert(0, {str(HERE)!r})
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('c', {str(HERE / 'batch-campaign.py')!r})
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.ROOT = pathlib.Path({str(tmp_path)!r})
c.sys.platform = 'linux'
c.helper_hashes = lambda: {{}}
c.Campaign.wait_dependency = lambda *args: None
c.Campaign.observe = lambda *args: None
c.run = lambda runner: runner.stage('signal-child', {child_command()!r})
sys.argv = ['batch-campaign.py', '--shared-host', '--snr-policy-v2']
c.main()
'''
    with (tmp_path / 'harness.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', harness], stdout=log,
                                 stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / 'heartbeat').exists():
                assert child.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.02)
            os.kill(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
            before = (tmp_path / 'heartbeat').read_text()
            time.sleep(0.15)
            assert (tmp_path / 'heartbeat').read_text() == before
            state = c.control.read(tmp_path / 'batch-status.json')
            assert state['state'] == 'cancelled' and state['child_pid'] is None
        finally:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGTERM)
                child.wait(timeout=10)


def test_both_fresh_seed_matrices_must_pass_before_timing():
    complete = {f'qual-s{seed}-b{batch}-{route}': {'status': 'pass'}
                for seed in c.PLAN['qualification_seeds']
                for batch in c.BATCHES for route in c.ROUTES}
    assert len(complete) == 36
    assert c.PLAN['qualification_seeds'] == [7102, 7103]
    c.require_matrix(complete, 'qual', c.BATCHES)
    missing = dict(complete)
    del missing['qual-s7103-b1024-torch_cuda']
    with pytest.raises(ValueError, match='Incomplete'):
        c.require_matrix(missing, 'qual', c.BATCHES)
    complete['qual-s7103-b1024-torch_cuda'] = {'status': 'fail'}
    with pytest.raises(ValueError, match='timing skipped'):
        c.require_matrix(complete, 'qual', c.BATCHES)


def test_result_cannot_substitute_previously_examined_seed():
    result = evidence()
    result['arguments']['seed'] = 7101
    with pytest.raises(ValueError, match='seed'):
        c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)


@pytest.mark.parametrize('damage', ['oracle_missing', 'oracle_dtype', 'oracle_hash',
                                  'normalization', 'hidden_row_failure', 'oversized_error',
                                  'sigma_mismatch', 'old_policy_hash', 'fixture_normalization'])
def test_pass_summary_cannot_hide_invalid_v2_evidence(damage):
    result = evidence()
    row = result['pointwise_blocks'][1]['rows'][4]
    if damage == 'oracle_missing':
        del result['oracle']
    elif damage == 'oracle_dtype':
        result['oracle']['dtype'] = 'complex64'
    elif damage == 'oracle_hash':
        row['oracle']['sha256'] = ''
    elif damage == 'normalization':
        row['normalization']['norm'] = 2
    elif damage == 'hidden_row_failure':
        row['normalized_oracle']['failed_samples'] = 1
    elif damage == 'oversized_error':
        row['normalized_mkl_compatibility']['max_abs_error'] = 0.0011
    elif damage == 'sigma_mismatch':
        row['normalization']['sigmasq'] = 1.01
        fields = {k: row['normalization'][k] for k in ('norm', 'sigmasq', 'source')}
        row['normalization']['sha256'] = c.canonical_hash(fields)
    elif damage == 'fixture_normalization':
        row['normalization']['source'] = 'shared_fixture_value'
        fields = {k: row['normalization'][k] for k in ('norm', 'sigmasq', 'source')}
        row['normalization']['sha256'] = c.canonical_hash(fields)
    else:
        result['policy_sha256'] = 'c'*64
    with pytest.raises((ValueError, KeyError)):
        c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)


def test_original_raw_failures_remain_diagnostics_under_v2():
    result = evidence()
    result['pointwise_blocks'][0]['rows'][0]['raw_complex_v1_audit']['failed_samples'] = 5
    c.validate_result(result, 'branch_standard', 8, 'qualify', smoke=True)


def test_candidate_cannot_substitute_oracle_with_another_valid_hash(tmp_path):
    reference = evidence()
    c.control.save(tmp_path / 'result.json', reference)
    candidate = evidence()
    for key in ('oracle', 'reference'):
        candidate[key].update(created=False, directory=str(tmp_path))
    c.validate_pointwise(candidate, 8, 32768, 3, tmp_path)
    candidate['pointwise_blocks'][1]['rows'][2]['oracle']['sha256'] = 'c'*64
    with pytest.raises(ValueError, match='hash mismatch'):
        c.validate_pointwise(candidate, 8, 32768, 3, tmp_path)
