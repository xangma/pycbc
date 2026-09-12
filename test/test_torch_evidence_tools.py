"""Evidence gates must fail closed independently of performance measurements."""

import base64
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmarking.evidence import array_error, git_snapshot, timed_command, write_receipt


def test_complex_errors_reject_phase_nonfinite_and_shape():
    reference = np.array([1 + 1j, 2 - 1j])
    assert not array_error(reference, reference * 1j, rtol=1e-3)['passed']
    assert not array_error(reference, [np.nan, 1], rtol=1e-3)['passed']
    assert not array_error(reference, [1], rtol=1e-3)['passed']
    assert array_error([0], [0], rtol=0)['passed']


def test_candidate_comparison_rejects_equal_magnitude_wrong_phase():
    from tools.benchmarking.benchmark_gpu_search import compare_template_candidates
    reference = {'sample_idx': 4, 'snr': 8 + 1j, 'chisq_dof': 14, 'red_chisq': 1.}
    with pytest.raises(RuntimeError, match='SNR mismatch'):
        compare_template_candidates([dict(reference, snr=1 - 8j)], [reference])
    with pytest.raises(RuntimeError, match='missing GPU trigger'):
        compare_template_candidates([], [reference])
    with pytest.raises(RuntimeError, match='sample position mismatch'):
        compare_template_candidates([dict(reference, sample_idx=5)], [reference])


def test_receipt_serialization_preserves_previous_evidence(tmp_path):
    path = tmp_path / 'receipt.json'
    write_receipt(path, {'status': 'passed'})
    with pytest.raises(ValueError):
        write_receipt(path, {'metric': float('nan')})
    assert json.loads(path.read_text()) == {'status': 'passed'}


def test_git_snapshot_reconstructs_tracked_and_untracked_sources(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()

    def git(*args, cwd=source, **kwargs):
        return subprocess.run(['git', *args], cwd=cwd, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    git('init')
    (source / 'module.py').write_text('value = 1\n')
    git('add', 'module.py')
    git('-c', 'user.name=Evidence Test', '-c', 'user.email=test@example.invalid',
        '-c', 'core.hooksPath=/dev/null', 'commit', '-m', 'fixture')
    (source / 'module.py').write_text('value = 2\n')
    (source / 'provider.py').write_text('provider = "native"\n')
    snapshot = git_snapshot(source)
    restored = tmp_path / 'restored'
    git('clone', '--no-hardlinks', str(source), str(restored))
    git('apply', '-', cwd=restored, input=base64.b64decode(snapshot['tracked_patch_base64']))
    for name, value in snapshot['untracked_source_base64'].items():
        (restored / name).write_bytes(base64.b64decode(value))
    for name in ('module.py', 'provider.py'):
        assert (restored / name).read_bytes() == (source / name).read_bytes()
    assert snapshot['git_dirty'] is True


def test_process_timer_includes_launch_exit_and_output(tmp_path):
    output = tmp_path / 'worker-output.txt'
    log = tmp_path / 'worker.log'
    command = [sys.executable, '-c',
               'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("complete"); print("done")',
               str(output)]
    receipt = timed_command(command, log, cwd=tmp_path)
    assert receipt['returncode'] == 0
    assert output.read_text() == 'complete'
    assert log.read_text().strip() == 'done'
    assert receipt['process_wall_time_sec'] > 0
    assert receipt['timer_boundary'] == 'before subprocess launch through process exit; includes output'


def test_requested_cuda_unavailable_is_skipped_without_reference_fallback(tmp_path, monkeypatch):
    from tools import verify_torchwave_gpu_search as verify
    monkeypatch.setattr(verify.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(verify, 'execution_provenance', lambda: {})
    monkeypatch.setattr(verify, 'generate_bank', lambda *_: pytest.fail('must not generate on CPU'))
    path = tmp_path / 'skipped.json'
    assert verify.main(['--device', 'cuda:0', '--output', str(path)]) == 2
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'skipped'
    assert receipt['requested_device'] == 'cuda:0'
    assert receipt['effective_device'] is None


def test_verifier_checks_failure_after_first_eight_rows(tmp_path, monkeypatch):
    from tools import verify_torchwave_gpu_search as verify
    from pycbc.types import FrequencySeries
    monkeypatch.setattr(verify, 'execution_provenance', lambda: {})
    args = verify.parse_args(['--batch-size', '12', '--flen', '129', '--delta-f', '1',
                              '--output', str(tmp_path / 'rows.json')])
    reference = np.ones((12, 129), np.complex64)
    actual = reference.copy()
    actual[10] *= 1j

    def generate(_path, _args, native):
        values = actual if native else reference
        return (verify.torch.tensor(values),
                [FrequencySeries(row.copy(), delta_f=1) for row in values],
                [{'provider': 'torchwave', 'index': i} for i in range(12)])

    monkeypatch.setattr(verify, 'generate_bank', generate)
    monkeypatch.setattr(verify, 'sigmasq', lambda *a, **k: 1.)
    monkeypatch.setattr(verify, 'matched_filter_core', lambda row, *a, **k: (row.copy(), None, 1.))
    monkeypatch.setattr(verify, 'power_chisq_bins', lambda *a: np.array([1, 20]))
    monkeypatch.setattr(verify, 'compute_canonical_cpu_reference_template', lambda *a, **k: [])
    monkeypatch.setattr(verify, 'engine_candidates', lambda *a: ({i: [] for i in range(12)}, None))
    result = verify.qualify(args)
    assert result['status'] == 'failed'
    assert result['checked_rows'] == 12
    assert result['rows'][9]['passed'] is True
    assert result['rows'][10]['passed'] is False
    monkeypatch.setattr(verify, 'qualify', lambda args: result)
    assert verify.main(['--output', args.output]) == 1


@pytest.mark.parametrize('size', [2**17, 2**19, 2**20, 2**21])
def test_geometry_propagates_to_every_cli_mode(size, tmp_path, monkeypatch):
    """Large-N routing is tested without allocating the full experiment matrix."""
    from tools.benchmarking import benchmark_gpu_search as bench
    calls = {}
    fake_cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'test',
                                get_device_properties=lambda _: SimpleNamespace(total_memory=1))
    monkeypatch.setattr(bench, 'torch', SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda='test')))
    monkeypatch.setattr(bench, 'capture_execution_provenance', lambda **k: {})
    for name in ('tile_scaling', 'cuda_graphs', 'live_streaming_latency', 'production_inspiral'):
        def record(name=name, **kw):
            calls.setdefault(name, []).append(kw)
            return {}
        monkeypatch.setattr(bench, 'benchmark_' + name, record)
    monkeypatch.setattr(bench, 'save_qualification_receipt', lambda *a, **k: Path(a[1]))
    bench.main(['--size', str(size), '--production-size', str(size), '--num-templates', '2',
                '--production-templates', '2', '--tile-size', '2', '--num-segments', '2',
                '--iterations', '1', '--num-blocks', '1', '--include-production-inspiral',
                '--output', str(tmp_path / 'geometry.json')])
    assert {name: [r['size'] for r in rows] for name, rows in calls.items()} == {
        'tile_scaling': [size, size], 'cuda_graphs': [size],
        'live_streaming_latency': [size], 'production_inspiral': [size]}
    assert calls['tile_scaling'][0]['num_templates'] == calls['tile_scaling'][1]['num_templates'] == 2


def test_v2_receipt_rejects_inconsistent_geometry_and_memory_counts():
    from tools.benchmarking.benchmark_gpu_search import validate_qualification_receipt
    live = {'transform_length': 1024, 'delta_f': 1 / 1024, 'sample_rate_hz': 1.,
            'num_blocks': 2, 'raw_latency_ms': [1., 1.],
            'memory_samples': [{'block': i, 'allocated_bytes': 0, 'reserved_bytes': 0,
                                'peak_allocated_bytes': 0} for i in range(-1, 2)]}
    report = {'schema_version': 2, 'benchmarks': {'live_streaming_latency': live}}
    assert validate_qualification_receipt(report, require_full_workload=False)
    live['sample_rate_hz'] = 4096.
    with pytest.raises(ValueError, match='inconsistent transform_length'):
        validate_qualification_receipt(report, require_full_workload=False)
    live['sample_rate_hz'] = 1.
    live['num_blocks'] = 500
    with pytest.raises(ValueError, match='count must equal recorded repetitions'):
        validate_qualification_receipt(report, require_full_workload=False)


@pytest.mark.parametrize('size', ['7', '9', '-8'])
def test_cli_rejects_invalid_geometry_before_running(size, monkeypatch):
    from tools.benchmarking import benchmark_gpu_search as bench
    monkeypatch.setattr(bench, 'benchmark_tile_scaling', lambda **kw: pytest.fail('must not execute'))
    with pytest.raises(SystemExit):
        bench.main(['--size', size])


@pytest.mark.parametrize('size', [2**17, 2**19, 2**20, 2**21])
def test_direct_large_geometry_with_two_templates(size):
    """Direct acquisition uses two rows and one repeat, with explicit geometry."""
    from tools.benchmarking import benchmark_gpu_search as bench
    report = bench.benchmark_tile_scaling('cpu', num_templates=2, size=size,
                                         tile_sizes=[2], warmup=1, iterations=1)
    row = report['tile_size_2']
    assert row['transform_length'] == size
    assert row['delta_f'] * size == row['sample_rate_hz']
    assert row['num_templates'] == 2
    assert len(row['raw_submit_drain_seconds']) == 1
