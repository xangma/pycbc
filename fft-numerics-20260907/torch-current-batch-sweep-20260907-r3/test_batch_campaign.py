"""Controls tests use local temporary processes; no PyCBC or GPU is imported."""
import importlib.util
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
    return {'schema_version': 1, 'route': 'branch_standard', 'batch': 8,
            'mode': 'qualify', 'expected_head': c.REVISION,
            'source': {'revision': c.REVISION, 'tracked_dirty': False},
            'worker_sha256': c.control.digest(HERE / 'batch-worker.py'),
            'inputs': {'geometry': {'bank_templates': 8, 'fft_samples': 32768, 'blocks': 3}},
            'input_sha256': 'fixed-input', 'thread_environment': {'OMP_NUM_THREADS': '1'},
            'affinity': [8],
            'group_layout': [list(range(1000, 1008))],
            'fft_plans': [{'nbatch': 8, 'class': 'pycbc.fft.mkl.IFFT'}],
            'status': 'pass', 'failures': [], 'exit_code': 0,
            'trigger_blocks': [{'count': 4} for _ in range(3)],
            'trigger_comparisons': [],
            'pointwise_blocks': [{'processed_once': True, 'samples': 8 * 32768,
                                  'failed_samples': 0, 'nonfinite_actual': 0,
                                  'nonfinite_reference': 0, 'processing_counts': [1] * 8,
                                  'rows': [{}] * 8} for _ in range(3)]}


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
sys.argv = ['batch-campaign.py', '--shared-host']
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
