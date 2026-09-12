"""Evidence gates must fail closed independently of performance measurements."""

import base64
import copy
import hashlib
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


def candidate_ticket(snr=8 + 1j, **flags):
    values = {'completed': True, 'aborted': False, 'overflow': False,
              'results': [{'template_id': np.array([0]), 'sample_idx': np.array([4]),
                           'snr': np.array([snr]), 'chisq': np.array([14.]),
                           'chisq_dof': np.array([14])}]}
    values.update(flags)
    return SimpleNamespace(**values)


def test_search_snapshot_owns_values_and_rejects_bad_ticket():
    from tools.benchmarking.evidence import search_snapshot, compare_search_snapshots
    ticket = candidate_ticket()
    before = search_snapshot([ticket], 1)
    ticket.results[0]['snr'][0] *= 1j
    after = search_snapshot([ticket], 1)
    assert before['snr'][0] == 8 + 1j
    with pytest.raises(ValueError, match='output mismatch: snr'):
        compare_search_snapshots(before, after)
    with pytest.raises(ValueError, match='one completed ticket'):
        search_snapshot([], 1)
    ticket.overflow = True
    with pytest.raises(ValueError, match='overflow'):
        search_snapshot([ticket], 1)
    ticket.overflow = False
    ticket.completed = False
    with pytest.raises(ValueError, match='not completed'):
        search_snapshot([ticket], 1)


def test_each_timed_result_requires_candidate_veto_and_newsnr_parity(monkeypatch):
    from tools import bench_torchwave_pipeline as pipeline
    expected = {0: [{'sample_idx': 4, 'snr': 8 + 1j, 'chisq': 14.,
                     'chisq_dof': 14, 'red_chisq': 1.}]}
    args = SimpleNamespace(batch_size=1)
    assert pipeline.validate_timed_result([candidate_ticket()], expected, args, 'first')['passed']
    bad = candidate_ticket(1 - 8j)
    assert not pipeline.validate_timed_result([bad], expected, args, 'warm_0')['passed']
    bad = candidate_ticket()
    bad.overflow = True
    assert not pipeline.validate_timed_result([bad], expected, args, 'warm_1')['passed']
    bad = candidate_ticket()
    bad.results[0]['chisq'][0] = 28.
    assert not pipeline.validate_timed_result([bad], expected, args, 'warm_2')['passed']
    # A within-tolerance perturbation can still cross an acceptance boundary.
    expected[0][0]['snr'] = 5.0001 + 0j
    monkeypatch.setattr(pipeline.verification, 'newsnr', lambda snr, chisq: snr)
    result = pipeline.validate_timed_result([candidate_ticket(4.9999 + 0j)], expected, args, 'warm_3')
    assert not result['passed']
    assert 'NewSNR accepted identities differ' in result['reason']


def pipeline_fixture():
    provenance = {'repositories': {'pycbc': {'root': '/source', 'git_commit': 'commit',
        'tracked_patch_sha256': 'patch', 'untracked_source_base64': {}}},
        'library_versions': {'pycbc': 'version'}, 'imported_module_paths': {'pycbc': '/source/pycbc'},
        'loaded_extension_binaries': {'pycbc.test': {'path': '/link.so', 'resolved_path': '/actual.so',
                                                    'sha256': 'binary'}},
        'mapped_reference_libraries': {}, 'reference_fft': {'cpu_backend': 'fftw'},
        'reference_library_versions': {'lal': {'version': 'version'}}}
    diagnostics = [{'index': 0, 'provider': 'torchwave'}]
    parent = {'status': 'passed', 'requested_device': 'cpu', 'requested_dtype': 'complex64',
              'configuration': {'batch_size': 1}, 'physical_manifest': {'mass1': [10.]},
              'geometry': {'transform_length': 128, 'delta_f': 1.},
              'provider_diagnostics': diagnostics,
              'input_hashes': {'reference_waveforms': 'ref', 'native_waveforms': 'native',
                               'strain': 'strain', 'psd': 'psd'}, 'provenance': provenance}
    child = {'status': 'passed', 'provider': 'torchwave', 'requested_device': 'cpu',
             'effective_device': 'cpu', 'effective_dtype': 'torch.complex64',
             'manifest': parent['physical_manifest'], 'provider_diagnostics': diagnostics,
             'geometry': parent['geometry'],
             'input_hashes': {'reference_waveforms': 'ref', 'waveforms': 'native',
                              'strain': 'strain', 'psd': 'psd'},
             'raw_prepared_submit_drain_sec': [1.],
             'result_qualification': [{'passed': True, 'rows': [{'index': 0}],
                                        'output_hashes': {'snr': 'output'}} for _ in range(2)],
             'provenance': copy.deepcopy(provenance)}
    return parent, child


@pytest.mark.parametrize('failure', ['status', 'fixture', 'waveform', 'dispatch', 'dtype',
                                     'samples', 'rows', 'source', 'binary'])
def test_parent_rejects_successful_process_with_unqualified_child(failure):
    from tools import bench_torchwave_pipeline as pipeline
    parent, child = pipeline_fixture()
    assert pipeline.validate_child_measurement(child, parent, 'torchwave', 1)['passed']
    if failure == 'status':
        child['status'] = 'failed'
    elif failure == 'fixture':
        child['input_hashes']['strain'] = 'different'
    elif failure == 'waveform':
        child['input_hashes']['waveforms'] = 'different'
    elif failure == 'dispatch':
        child['provider_diagnostics'] = [{'index': 0, 'provider': 'reference'}]
    elif failure == 'dtype':
        child['effective_dtype'] = 'torch.complex128'
    elif failure == 'samples':
        child['result_qualification'].pop()
    elif failure == 'rows':
        child['result_qualification'][1]['rows'] = []
    elif failure == 'source':
        child['provenance']['repositories']['pycbc']['tracked_patch_sha256'] = 'different'
    elif failure == 'binary':
        child['provenance']['loaded_extension_binaries']['pycbc.test']['sha256'] = 'different'
    assert not pipeline.validate_child_measurement(child, parent, 'torchwave', 1)['passed']


def test_pipeline_main_fails_when_zero_exit_child_has_failed_result(tmp_path, monkeypatch):
    from tools import bench_torchwave_pipeline as pipeline
    parent, child = pipeline_fixture()
    child['status'] = 'failed'
    monkeypatch.setattr(pipeline.verification, 'qualify', lambda args: parent)

    def command(command, logfile):
        write_receipt(command[command.index('--output') + 1], child)
        return {'returncode': 0}

    monkeypatch.setattr(pipeline, 'timed_command', command)
    output = tmp_path / 'campaign.json'
    assert pipeline.main(['--cold-runs', '1', '--warm-samples', '1', '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert all(not row['child_qualification']['passed'] for row in report['runs'])


def test_worker_validates_its_actual_first_and_every_prepared_drain(monkeypatch, tmp_path):
    from tools import bench_torchwave_pipeline as pipeline
    expected = [{'sample_idx': 4, 'snr': 8 + 1j, 'chisq': 14.,
                 'chisq_dof': 14, 'red_chisq': 1.}]

    class Tensor:
        device = 'cpu'
        dtype = pipeline.torch.complex64

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((1, 65), np.complex64)

    class Engine:
        def __init__(self, *args, **kwargs):
            self.count = 0

        def submit(self, *args):
            self.count += 1

        def drain(self):
            return [candidate_ticket(1 - 8j if self.count == 2 else 8 + 1j)]

        def close(self):
            pass

    args = SimpleNamespace(device='cpu', dtype='complex64', batch_size=1, flen=65, delta_f=1.)
    monkeypatch.setattr(pipeline, 'execution_provenance', lambda: {})
    monkeypatch.setattr(pipeline.verification, 'create_fixture_bank', lambda *a: {'mass1': [10.]})
    monkeypatch.setattr(pipeline.verification, 'generate_bank', lambda path, args, native:
        (Tensor(), [object()], [{'index': 0, 'provider': 'torchwave' if native else 'reference'}]))
    monkeypatch.setattr(pipeline.verification, 'prepare_fixture', lambda *a:
        (Tensor(), Tensor(), 30., 40., (8, 56)))
    monkeypatch.setattr(pipeline.verification, 'sigmasq', lambda *a, **k: 1.)
    monkeypatch.setattr(pipeline.verification, 'power_chisq_bins', lambda *a: [1, 20])
    monkeypatch.setattr(pipeline.verification, 'compute_canonical_cpu_reference_template',
                        lambda *a, **k: expected)
    for name in ('prepare_bank', 'bind_psd', 'prepare_power_chisq_plan'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'SearchEngine', Engine)
    result = pipeline.worker(args, 'torchwave', 2)
    assert result['status'] == 'failed'
    assert [row['passed'] for row in result['result_qualification']] == [True, False, True]
    assert len(result['raw_prepared_submit_drain_sec']) == 2
    monkeypatch.setattr(pipeline, 'worker', lambda *a: result)
    assert pipeline.main(['--worker-provider', 'torchwave', '--output', str(tmp_path / 'child.json')]) == 1


@pytest.mark.parametrize('failure', [None, 'phase', 'overflow', 'missing', 'no_graph'])
def test_graph_speedup_requires_every_actual_result_to_match(failure, monkeypatch):
    from tools.benchmarking import benchmark_gpu_search as bench
    fake_cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
    monkeypatch.setattr(bench, 'torch', SimpleNamespace(cuda=fake_cuda))
    series = SimpleNamespace(numpy=lambda: np.ones(5, np.complex64))
    monkeypatch.setattr(bench, 'generate_synthetic_bank', lambda *a: [series])
    monkeypatch.setattr(bench, 'generate_synthetic_data', lambda *a: (series, series))
    monkeypatch.setattr(bench, 'prepare_bank', lambda *a, **k: None)
    monkeypatch.setattr(bench, 'bind_psd', lambda *a, **k: None)

    class Engine:
        def __init__(self, *args, use_cuda_graphs=False, **kwargs):
            self.graph = use_cuda_graphs
            self.submissions = 0
            self.graph_stats = {'capture_count': int(use_cuda_graphs),
                                'replay_count': int(use_cuda_graphs and failure != 'no_graph')}

        def submit(self, *args):
            self.submissions += 1

        def drain(self):
            ticket = candidate_ticket()
            if self.graph and self.submissions == 2:
                if failure == 'phase':
                    ticket.results[0]['snr'][0] *= 1j
                elif failure == 'overflow':
                    ticket.overflow = True
                elif failure == 'missing':
                    return []
            return [ticket]

        def close(self):
            pass

    monkeypatch.setattr(bench, 'SearchEngine', Engine)
    result = bench.benchmark_cuda_graphs(num_templates=1, size=8, tile_size=1, warmup=1, iterations=2)
    assert len(result['output_equivalence']['checks']) == 6
    assert len(result['raw_graph_ms']) == len(result['raw_eager_ms']) == 2
    if failure is None:
        assert result['status'] == 'passed'
        assert np.isfinite(result['speedup'])
    else:
        assert result['status'] == 'failed'
        assert result['speedup'] is None
    receipt = {'schema_version': 2, 'benchmarks': {'cuda_graph_comparison': result}}
    assert bench.validate_qualification_receipt(receipt, require_full_workload=False)
    if failure is not None:
        result['speedup'] = 1.1
        with pytest.raises(ValueError, match='graph speedup requires'):
            bench.validate_qualification_receipt(receipt, require_full_workload=False)


def test_loaded_extension_identity_resolves_abi_symlinks(tmp_path, monkeypatch):
    from tools.benchmarking.evidence import loaded_implementation_identities
    actual = tmp_path / 'extension.cpython-311.so'
    actual.write_bytes(b'compiled extension fixture')
    alias = tmp_path / 'extension.cpython-313.so'
    alias.symlink_to(actual)
    monkeypatch.setitem(sys.modules, 'pycbc.test_binary', SimpleNamespace(__file__=str(alias)))
    identities = loaded_implementation_identities()
    identity = identities['loaded_extension_binaries']['pycbc.test_binary']
    assert identity['path'] == str(alias)
    assert identity['resolved_path'] == str(actual)
    assert identity['sha256'] == hashlib.sha256(actual.read_bytes()).hexdigest()
    assert 'cpu_backend' in identities['reference_fft']
